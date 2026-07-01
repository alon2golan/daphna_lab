"""
migrate_nu_prime_cache_optnest.py
---------------------------------
One-time migration: promote results/<dataset>/nu_prime_cache.json from the
legacy shape  {al_method: {budget: {tau: nu'}}}  (implicitly Adam) to the
optimizer-nested shape  {al_method: {optimizer: {budget: {tau: nu'}}}}.

Sources:
  - the existing dense cache            -> the "adam" column (authoritative)
  - nu_prime_cache_optcompare.json      -> the "sgd" column for {probcover,random}
                                           + any adam budgets the dense cache
                                           lacks (e.g. B=3000)

maxherding keeps adam only (no SGD estimate exists). No cross-optimizer
fallback is invented; a method/optimizer with no source data is simply absent.

Usage:
  python batch_utils/migrate_nu_prime_cache_optnest.py            # dry-run (prints plan)
  python batch_utils/migrate_nu_prime_cache_optnest.py --apply    # backup + write
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def is_legacy(method_dict: dict) -> bool:
    return bool(method_dict) and all(str(k).isdigit() for k in method_dict)


def migrate(dataset: str, apply: bool) -> None:
    res = REPO / "results" / dataset
    live_path = res / "nu_prime_cache.json"
    oc_path = res / "nu_prime_cache_optcompare.json"

    live = json.loads(live_path.read_text()) if live_path.exists() else {}
    oc = json.loads(oc_path.read_text()) if oc_path.exists() else {}

    if not live:
        print(f"[skip] no live cache at {live_path}")
        return

    # Already migrated? (top-of-method keys are optimizer names, not budgets)
    if not any(is_legacy(md) for md in live.values()):
        print(f"[skip] {live_path} already uses the optimizer-nested schema")
        return

    new: dict = {}
    for al, md in live.items():
        # dense cache = authoritative adam column
        new[al] = {"adam": copy.deepcopy(md)}

    # overlay optcompare: sgd column + any adam budgets the dense cache lacks
    for al, opts in oc.items():
        new.setdefault(al, {})
        if "sgd" in opts:
            new[al]["sgd"] = copy.deepcopy(opts["sgd"])
        for b, taus in opts.get("adam", {}).items():
            new[al].setdefault("adam", {}).setdefault(b, {})
            for t, v in taus.items():
                new[al]["adam"][b].setdefault(t, v)  # don't clobber dense values

    # report
    print(f"== migration plan for {live_path} ==")
    for al in sorted(new):
        for opt in sorted(new[al]):
            budgets = sorted(int(b) for b in new[al][opt])
            print(f"  {al:12s} {opt:4s}  budgets={budgets}")

    if not apply:
        print("\nDry-run. Re-run with --apply to back up and write.")
        return

    bak = live_path.with_suffix(".json.bak_optmerge")
    shutil.copy2(live_path, bak)
    live_path.write_text(json.dumps(new, indent=2))
    print(f"\nBacked up -> {bak}")
    print(f"Wrote     -> {live_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--apply", action="store_true")
    migrate(ap.parse_args().dataset, ap.parse_args().apply)
