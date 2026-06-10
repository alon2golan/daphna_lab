"""
Sweep registry: a single JSON ledger of every sweep/job you've run, of any
kind (train_al experiments, ν'-caching jobs, other-repo jobs, misc tasks).

This is the data source for the new dashboard's **Sweeps** tab. Unlike the
result.json scanner (which only sees train_al runs), the registry can hold ANY
job — seeded from history by reporting/backfill_registry.py and appended to as
you launch new sweeps.

Schema — logs/sweep/sweep_registry.json is a list of entries:
    {
      "sweep_id":   "30726412",                    # SLURM array job id, or local_/manual_ id
      "sweep_name": "fig55_daphna_20260603_104536",
      "kind":       "train_al",                    # train_al | nu_caching | other_repo | misc
      "category":   "fig55",                        # finer purpose tag (see CATEGORY_RULES)
      "command":    "...",                          # representative command / script
      "created_at": "2026-06-03T10:45:38",          # ISO; "" if unknown
      "source":     "result_json",                  # result_json | sacct | logs | manual
      "note":       ""                              # free text, never overwritten by backfill
    }

The (kind, category) pair is auto-derived by ``categorize()`` but is freely
overridable — the dashboard writes user edits straight back here, and the
backfill preserves any entry whose ``source == "manual"`` plus the ``note`` and
``category`` of every existing entry.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = REPO_ROOT / "logs" / "sweep" / "sweep_registry.json"

# ── Categorisation ─────────────────────────────────────────────────────────────
#
# Ordered; first match wins. Each rule matches against "<sweep_name> <command>".
# Edit freely — this is the single source of truth for auto-categories.
#
# `category` is the finer purpose tag used to scope the Graphs tab; `kind` is the
# coarse bucket the Sweeps ledger groups by when a job isn't a train_al run.

KINDS = ["train_al", "nu_caching", "other_repo", "misc"]

# Purpose tags for train_al runs (the Graphs tab scopes on these).
CATEGORY_RULES = [
    ("fig55",           re.compile(r"fig55")),
    ("oracle_vs_noisy", re.compile(r"probcover_random|oracle_vs_noisy")),
    # noise_mechanism before nu_estimation: "bypass_noise_mechanism" sweeps embed
    # a "_nu0.3" in their command, which the nu_estimation rule would match first.
    ("noise_mechanism", re.compile(r"bypass_noise|noise_mechanism")),
    ("nu_estimation",   re.compile(r"oneshot|seeddecoupled|nu_sweep|_nu[0-9.]")),
]
KNOWN_CATEGORIES = (
    [c for c, _ in CATEGORY_RULES]
    + ["uncategorized", "nu_caching", "other_repo", "misc"]
)

_NU_CACHE_RX = re.compile(r"nu_prime|nu_optcompare|sweep_nu", re.IGNORECASE)
_TRAIN_AL_RX = re.compile(r"train_al\.py")


def categorize(sweep_name: str, command: str = "") -> dict:
    """Return {"kind": ..., "category": ...} derived from name + command."""
    hay = f"{sweep_name or ''} {command or ''}"

    if _TRAIN_AL_RX.search(hay):
        kind = "train_al"
        category = "uncategorized"
        for name, rx in CATEGORY_RULES:
            if rx.search(hay):
                category = name
                break
        return {"kind": kind, "category": category}

    if _NU_CACHE_RX.search(hay):
        return {"kind": "nu_caching", "category": "nu_caching"}

    # Unknown script we can still name → other_repo if it looks like a real job,
    # else misc. We can't reliably tell, so default everything else to misc.
    return {"kind": "misc", "category": "misc"}


# ── Load / save ─────────────────────────────────────────────────────────────────


def load_registry() -> list[dict]:
    if REGISTRY_FILE.exists():
        try:
            data = json.loads(REGISTRY_FILE.read_text())
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_registry(entries: list[dict]) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Stable order: newest first by created_at, unknown dates last.
    entries = sorted(
        entries,
        key=lambda e: (e.get("created_at") or "", e.get("sweep_id") or ""),
        reverse=True,
    )
    REGISTRY_FILE.write_text(json.dumps(entries, indent=2))


def _index(entries: list[dict]) -> dict[str, dict]:
    return {e["sweep_id"]: e for e in entries if e.get("sweep_id")}


def upsert(entry: dict, *, preserve_user_fields: bool = True) -> None:
    """
    Insert or merge a single entry, keyed on sweep_id.

    When merging an existing entry, user-owned fields (``category``, ``note``,
    and any entry whose ``source == "manual"``) are preserved by default so a
    re-run of the backfill never clobbers hand edits.
    """
    entries = load_registry()
    idx = _index(entries)
    sid = entry.get("sweep_id")
    if not sid:
        return
    if sid in idx and preserve_user_fields:
        existing = idx[sid]
        if existing.get("source") == "manual":
            return  # fully user-owned; leave untouched
        merged = {**entry}
        for k in ("category", "note"):
            if existing.get(k):
                merged[k] = existing[k]
        idx[sid].update(merged)
    elif sid in idx:
        idx[sid].update(entry)
    else:
        entries.append(entry)
    save_registry(entries)


def set_category(sweep_id: str, category: str) -> None:
    """Override the category of one sweep (called by the dashboard dropdown)."""
    entries = load_registry()
    for e in entries:
        if e.get("sweep_id") == sweep_id:
            e["category"] = category
            save_registry(entries)
            return


def make_entry(
    sweep_id: str,
    sweep_name: str,
    command: str = "",
    created_at: str = "",
    source: str = "manual",
    note: str = "",
) -> dict:
    cat = categorize(sweep_name, command)
    return {
        "sweep_id": str(sweep_id),
        "sweep_name": sweep_name or str(sweep_id),
        "kind": cat["kind"],
        "category": cat["category"],
        "command": command,
        "created_at": created_at,
        "source": source,
        "note": note,
    }


# ── CLI: hand-add an entry ───────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Manually add/edit a sweep registry entry")
    p.add_argument("--id", required=True, help="sweep_id (SLURM array job id or any unique id)")
    p.add_argument("--name", required=True, help="human-readable sweep name")
    p.add_argument("--command", default="", help="representative command / script")
    p.add_argument("--created-at", default="", help="ISO timestamp (optional)")
    p.add_argument("--category", default=None, help="override auto-category")
    p.add_argument("--note", default="", help="free-text note")
    a = p.parse_args()

    e = make_entry(a.id, a.name, a.command, a.created_at, source="manual", note=a.note)
    if a.category:
        e["category"] = a.category
    upsert(e, preserve_user_fields=False)
    print(f"Registry now has {len(load_registry())} entries → {REGISTRY_FILE}")
