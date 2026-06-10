"""
One-time (re-runnable) backfill of the sweep registry from existing history.

Seeds logs/sweep/sweep_registry.json from three automatic sources, deduped by
sweep_id, then leaves anything else for you to hand-add via:
    python reporting/registry.py --id <id> --name <name> [...]

Sources (in priority order — earlier sources win the name/command, later ones
only fill in sweep_ids not already seen):
  1. result.json under output/    → all train_al sweeps, full command + date
  2. one-shot `sacct` via ssh      → recent non-train_al jobs (in SLURM retention)
  3. logs/sweep/*.out, *.sh names  → older jobs aged out of sacct

Idempotent: re-running merges and never clobbers manual edits or existing
category/note fields (see registry.upsert).

Usage:
    python reporting/backfill_registry.py                 # all three sources
    python reporting/backfill_registry.py --no-sacct      # skip the ssh call
    python reporting/backfill_registry.py --sacct-since 2026-01-01
    python reporting/backfill_registry.py --dry-run       # print, don't write
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reporting.registry import (  # noqa: E402
    REGISTRY_FILE,
    load_registry,
    make_entry,
    save_registry,
)
from reporting.scanner import load_all_results  # noqa: E402

LOG_DIR = REPO_ROOT / "logs" / "sweep"
SSH_HOST = "phoenix"
SLURM_USER = "alongol"

# logs/sweep/<name>_<jobid>_<task>.out  →  (name, jobid)
_LOG_OUT_RX = re.compile(r"^(?P<name>.+?)_(?P<jobid>\d{5,})_(?P<task>\d+)\.out$")
# logs/sweep/<id>.sh  (e.g. local_20260603_104536.sh, 30726412.sh)
_LOG_SH_RX = re.compile(r"^(?P<id>[A-Za-z0-9_]+)\.sh$")


def _array_base(jobid: str) -> str:
    """'30726412_5', '30726412_5.batch' → '30726412'."""
    return jobid.split("_", 1)[0].split(".", 1)[0]


# ── Source 1: result.json ────────────────────────────────────────────────────────


def from_result_json() -> "OrderedDict[str, dict]":
    out: "OrderedDict[str, dict]" = OrderedDict()
    df = load_all_results()
    if df.empty:
        return out
    for (sid, sname), g in df.groupby(["sweep_id", "sweep_name"], sort=False):
        g = g.sort_values("created_at")
        first = g.iloc[0]
        created = first.get("created_at")
        created_iso = created.isoformat() if created is not None and not _isna(created) else ""
        out.setdefault(
            str(sid),
            make_entry(
                sweep_id=str(sid),
                sweep_name=str(sname),
                command=str(first.get("command", "")),
                created_at=created_iso,
                source="result_json",
            ),
        )
    return out


def _isna(v) -> bool:
    try:
        import pandas as pd
        return bool(pd.isna(v))
    except Exception:
        return v is None


# ── Source 2: sacct (one-shot ssh) ───────────────────────────────────────────────


def from_sacct(since: str) -> "OrderedDict[str, dict]":
    out: "OrderedDict[str, dict]" = OrderedDict()
    fmt = "JobID,JobName%60,Start,State"
    remote = (
        f"sacct -u {SLURM_USER} --starttime {since} "
        f"-X -P -n -o {fmt}"
    )
    cmd = ["ssh", SSH_HOST, "bash", "-lc", f'"{remote}"']
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! sacct skipped ({exc})")
        return out
    if res.returncode != 0:
        print(f"  ! sacct returned {res.returncode}: {res.stderr.strip()[:200]}")
        return out

    for line in res.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jobid, jobname, start, _state = parts[0], parts[1], parts[2], parts[3]
        base = _array_base(jobid)
        if not base.isdigit():
            continue
        created = "" if start in ("Unknown", "None", "") else start
        out.setdefault(
            base,
            make_entry(
                sweep_id=base,
                sweep_name=jobname or base,
                command=jobname,  # sacct only gives the script name
                created_at=created,
                source="sacct",
            ),
        )
    return out


# ── Source 3: logs/sweep filenames ────────────────────────────────────────────────


def from_logs() -> "OrderedDict[str, dict]":
    out: "OrderedDict[str, dict]" = OrderedDict()
    if not LOG_DIR.exists():
        return out
    for f in sorted(LOG_DIR.iterdir()):
        m = _LOG_OUT_RX.match(f.name)
        if m:
            base = m.group("jobid")
            name = m.group("name")
            out.setdefault(
                base,
                make_entry(base, name, command=name, source="logs"),
            )
            continue
        m = _LOG_SH_RX.match(f.name)
        if m:
            sid = m.group("id")
            if sid in ("dashboard_hidden_sweeps", "sweep_registry"):
                continue
            out.setdefault(sid, make_entry(sid, sid, command="", source="logs"))
    return out


# ── Orchestration ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-sacct", action="store_true", help="skip the one-shot ssh sacct call")
    ap.add_argument("--sacct-since", default="2026-01-01", help="sacct --starttime (default 2026-01-01)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be added, don't write")
    a = ap.parse_args()

    print("Backfilling sweep registry…")
    existing = {e["sweep_id"]: e for e in load_registry() if e.get("sweep_id")}
    print(f"  existing entries: {len(existing)}")

    # Priority merge: result_json first, then sacct, then logs.
    merged: "OrderedDict[str, dict]" = OrderedDict()

    s1 = from_result_json()
    print(f"  result.json     : {len(s1)} sweeps")
    for k, v in s1.items():
        merged.setdefault(k, v)

    if not a.no_sacct:
        s2 = from_sacct(a.sacct_since)
        new2 = [k for k in s2 if k not in merged]
        print(f"  sacct           : {len(s2)} jobs ({len(new2)} new)")
        for k, v in s2.items():
            merged.setdefault(k, v)
    else:
        print("  sacct           : skipped (--no-sacct)")

    s3 = from_logs()
    new3 = [k for k in s3 if k not in merged]
    print(f"  logs/sweep      : {len(s3)} jobs ({len(new3)} new)")
    for k, v in s3.items():
        merged.setdefault(k, v)

    added = [k for k in merged if k not in existing]
    print(f"\n  → {len(merged)} total candidates, {len(added)} new vs current registry")

    # Summarise categories.
    from collections import Counter
    cats = Counter(v["category"] for v in merged.values())
    print("  categories:", dict(cats))

    if a.dry_run:
        print("\n(dry-run — nothing written)")
        for k in added[:30]:
            v = merged[k]
            print(f"    + {v['sweep_id']:>18}  [{v['category']:<14}] {v['sweep_name']}")
        return

    # Merge into the registry, preserving manual edits + existing category/note.
    final = OrderedDict(existing)
    from reporting.registry import upsert
    for v in merged.values():
        upsert(v)  # handles preserve-user-fields
    n = len(load_registry())
    print(f"\nDone. Registry now holds {n} entries → {REGISTRY_FILE}")


if __name__ == "__main__":
    main()
