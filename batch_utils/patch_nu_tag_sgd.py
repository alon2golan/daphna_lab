#!/usr/bin/env python
"""
One-shot backfill for the live SGD 3000cap sweep (job 30925479, sweep_name
train_al_3000cap_sgd__*), which launched BEFORE train_al gained --tag/--track_progress.

Writes, into each run's result.json:
  - tags["nu_tag"] = "sgd_b<nucache>"   (provenance of the nu_prime value)
  - params["track_progress"] = true     (so it shows in the dashboard Comparison tab)

Only patches COMPLETED runs: a still-running run's ResultWriter rewrites result.json
on every episode and would clobber the patch (its in-memory state has no tags). So
re-run this once the sweep fully finishes to catch the rest. Until then the dashboard
still shows these correctly — the scanner derives nu_tag=sgd_b<N> and the
train_noise_nu_prime category marks them track-progress as a fallback.

Idempotent and atomic (.tmp + rename). Future SGD sweeps get these directly from the
launcher (--track_progress --tag ...), so this script is only for the one in-flight sweep.

Usage:
    python batch_utils/patch_nu_tag_sgd.py            # dry-run: list what would change
    python batch_utils/patch_nu_tag_sgd.py --apply    # write the changes
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "output"
SWEEP_PREFIX = "train_al_3000cap_sgd__"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    args = ap.parse_args()

    changed, skipped = 0, 0
    for rf in sorted(OUTPUT.glob("*/*/*/*/result.json")):
        try:
            data = json.load(open(rf))
        except (json.JSONDecodeError, OSError):
            continue
        if not str(data.get("sweep_name", "")).startswith(SWEEP_PREFIX):
            continue
        # Skip in-flight runs — their ResultWriter would clobber the patch on the
        # next episode write. Re-run after the sweep completes to catch them.
        if data.get("status") != "completed":
            skipped += 1
            continue

        hay = (data.get("command", "") or "") + " " + str(rf.parent)
        m = re.search(r"nucache(\d+)", hay)
        nu_tag = f"sgd_b{m.group(1)}" if m else "sgd"

        tags = dict(data.get("tags") or {})
        params = dict(data.get("params") or {})
        already = tags.get("nu_tag") == nu_tag and params.get("track_progress") is True
        if already:
            skipped += 1
            continue

        tags["nu_tag"] = nu_tag
        params["track_progress"] = True
        data["tags"] = tags
        data["params"] = params

        rel = rf.relative_to(OUTPUT)
        if args.apply:
            tmp = rf.with_suffix(".json.tmp")
            json.dump(data, open(tmp, "w"), indent=2)
            os.replace(tmp, rf)
            print(f"patched  {rel}  -> nu_tag={nu_tag}, track_progress=True")
        else:
            print(f"WOULD patch  {rel}  -> nu_tag={nu_tag}, track_progress=True")
        changed += 1

    verb = "patched" if args.apply else "would patch"
    print(f"\n{verb} {changed} run(s); {skipped} already up-to-date.")
    if not args.apply and changed:
        print("Dry-run. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
