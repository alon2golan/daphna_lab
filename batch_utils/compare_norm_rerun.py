"""
Before/after comparison for the normalized-features rerun
(batch_utils/sweep_bayes_norm_rerun.py) vs the historical unnormalized sweeps
behind docs/bayesian_coverage_SUMMARY.md §4:

  compare   : BAYESCOV_compare_cifar100_*    (job 30978366)  vs BAYESCOV_norm_compare_cifar100_*
  trainmode : BAYESCOV_trainmode_cifar100_*  (job 30980646)  vs BAYESCOV_norm_trainmode_cifar100_*
  pseudo    : BAYESCOV_pseudo_cifar100_*     (job 31047129)  vs BAYESCOV_norm_pseudo_cifar100_*

Selection is untouched by the flag (AL-side features were always normalized and
training never feeds back into selection), so runs pair per-seed: every delta is
attributable purely to L2-normalizing the classifier's input features.

Writes docs/bayesian_coverage_NORMALIZATION.md (full doc, regenerated each run)
+ results/bayes_norm_rerun/norm_compare.csv (per-run rows for plotting).

Usage:  python batch_utils/compare_norm_rerun.py
"""
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_GLOB = "output/CIFAR100/resnet18/*/cfg*/result.json"
DOC = REPO / "docs" / "bayesian_coverage_NORMALIZATION.md"
CSV_OUT = REPO / "results" / "bayes_norm_rerun" / "norm_compare.csv"

FAMILIES = {  # family -> (old sweep_name prefix, norm sweep_name prefix)
    "compare":   ("BAYESCOV_compare_cifar100_",   "BAYESCOV_norm_compare_cifar100_"),
    "trainmode": ("BAYESCOV_trainmode_cifar100_", "BAYESCOV_norm_trainmode_cifar100_"),
    "pseudo":    ("BAYESCOV_pseudo_cifar100_",    "BAYESCOV_norm_pseudo_cifar100_"),
}
BUDGET_EPISODES = {100: 0, 300: 2, 1000: 9}  # labeled size -> episode index
AGG50 = "1,2,3,5,8,13,20,30,40,50"


def arm_of(family, params):
    if family == "compare":
        return params.get("al_method")
    if family == "trainmode":
        return params.get("train_label_mode")
    scope, sched = params.get("pseudo_scope"), params.get("pseudo_k")
    if scope == "all":
        return "pseudo_all"
    return "pseudo_agg50" if sched == AGG50 else "pseudo_lin10"


def collect():
    """-> {(family, side, arm, noise, seed): {budget: test_acc}}, side in {old,norm}"""
    runs = {}
    for rj in sorted(REPO.glob(OUT_GLOB)):
        try:
            d = json.load(open(rj))
        except Exception:
            continue
        sw = d.get("sweep_name") or ""
        for family, (old_pref, norm_pref) in FAMILIES.items():
            side = "norm" if sw.startswith(norm_pref) else \
                   "old" if sw.startswith(old_pref) else None
            if side is None:
                continue
            if d.get("status") != "completed":
                print(f"  SKIP not-completed: {sw} {rj.parent.name} ({d.get('status')})")
                continue
            if d.get("total_episodes_planned", 0) < 10:  # 1-episode smokes share keys with real runs
                print(f"  SKIP smoke ({d.get('total_episodes_planned')} eps): {rj.parent.name}")
                continue
            p = d["params"]
            key = (family, side, arm_of(family, p), p["noise_rate"], p["seed"])
            accs = {ep["labeled_size"]: ep["test_acc"] for ep in d.get("episodes", [])}
            if key in runs:  # keep the newest (dir sort order); flag it
                print(f"  DUP {key} — keeping latest {rj.parent.name}")
            runs[key] = {b: accs.get(b) for b in BUDGET_EPISODES}
    return runs


def seed_mean(runs, family, side, arm, noise, budget):
    vals = [v[budget] for (f, s, a, n, sd), v in runs.items()
            if (f, s, a, n) == (family, side, arm, noise) and v.get(budget) is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def fmt(x):
    return f"{x:.1f}" if x is not None else "—"


def main():
    runs = collect()
    noises = sorted({k[3] for k in runs})
    lines = [
        "# Bayesian coverage — classifier-input L2 normalization: before vs after",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `batch_utils/compare_norm_rerun.py`._",
        "",
        "**What changed:** the `from_features` classifier historically trained on RAW",
        "SimCLR features while the fixed 1-NN eval arm L2-normalizes (~+18 pts @100).",
        "The rerun adds `--normalize_features` (L2-normalized classifier inputs, train",
        "+ test). The AL side (kernel/posterior/selection) was ALWAYS normalized, so",
        "lSet trajectories pair per-seed and deltas isolate the training-input change.",
        "The probcover+1nn rows of the SUMMARY doc are unaffected (already normalized).",
        "",
        "**Reading:** cells are mean test acc over seeds (n in parens when < 3);",
        "Δ = normalized − raw, positive = normalization helps.",
        "",
        "## Takeaways (sweep 31048960 vs originals, 144 paired runs each)",
        "",
        "1. **The effect flips sign with budget.** @1000 labels normalization helps",
        "   almost every arm (+1 to +7 pts; observed +6.4/+6.8 at noise .8/.5,",
        "   per-seed Δ tight: 7.3/6.6/6.5). @100 labels it HURTS badly (−5 to −15)",
        "   for every arm that trains on only the ~100 labeled points; @300 is the",
        "   crossover. Exception: `pseudo_all` (trains on ~10k pseudo points every",
        "   round) is flat-to-positive at ALL budgets — consistent with the low-",
        "   budget losses being a CONVERGENCE artifact: at fixed BASE_LR/200 epochs,",
        "   tiny-budget training underfits isotropic inputs (smoke: train acc 50%",
        "   vs 95% @100) while high-norm raw points produce larger logits and fit",
        "   fast. The scale-preserving rescale fixed the gross collapse (unit-L2",
        "   gave 3.3 @100) but not this residual. Low-budget normalized cells need",
        "   an LR/epoch retune before being read as 'normalization is bad there'.",
        "2. **§4 qualitative conclusions SURVIVE normalization** (read @1000, the",
        "   converged regime): product ≫ additive everywhere (47.5 vs 29.9 clean);",
        "   pseudo is still the best training arm at every noise (49.0/46.5+/42.8/",
        "   29.9) though its clean-data edge over probcover+training closes to a",
        "   tie (49.0 vs 49.0); posterior-training still beats observed under noise",
        "   but the margin SHRINKS a lot (noise .5: +10 raw → +4.3 norm; noise .8:",
        "   2.2× → 1.5×) because observed gains the most from normalization —",
        "   i.e., part of what soft/map-posterior was fixing was feature-norm",
        "   pathology, not only label noise.",
        "3. **Absolute doc numbers are understated @1000** under a normalized",
        "   convention (everything +2-7). The NPC+AUM rows in SUMMARY §4 were NOT",
        "   rerun and remain raw-feature numbers — cross-compare with care.",
        "4. **probcover+1nn vs pseudo @100 clean:** normalized `pseudo_all` (38.4)",
        "   now ties 1-NN (38.1); on raw features 1-NN led (38.1 vs 37.6).",
        "5. lin10 ≡ agg50 @100/300 in BOTH arms (schedules coincide) — determinism",
        "   check passes on the normalized side too.",
        "",
    ]
    csv_rows = [("family", "arm", "noise", "budget", "raw", "norm", "delta", "n_raw", "n_norm")]
    for family in FAMILIES:
        arms = sorted({k[2] for k in runs if k[0] == family})
        if not arms:
            lines += [f"## {family}", "", "_No runs found yet._", ""]
            continue
        lines += [f"## {family}", ""]
        for budget in BUDGET_EPISODES:
            lines += [f"### {family} @ {budget} labels", "",
                      "| arm | " + " | ".join(f"n{n} raw | n{n} norm | Δ" for n in noises) + " |",
                      "|---|" + "---|" * (3 * len(noises))]
            for arm in arms:
                cells = []
                for n in noises:
                    old, n_old = seed_mean(runs, family, "old", arm, n, budget)
                    new, n_new = seed_mean(runs, family, "norm", arm, n, budget)
                    delta = f"{new - old:+.1f}" if (old is not None and new is not None) else "—"
                    o = fmt(old) + (f" ({n_old})" if 0 < n_old < 3 else "")
                    v = fmt(new) + (f" ({n_new})" if 0 < n_new < 3 else "")
                    cells += [o, v, delta]
                    csv_rows.append((family, arm, n, budget,
                                     fmt(old), fmt(new),
                                     f"{new - old:.2f}" if (old is not None and new is not None) else "",
                                     n_old, n_new))
                lines.append(f"| {arm} | " + " | ".join(cells) + " |")
            lines.append("")
    n_old = sum(1 for k in runs if k[1] == "old")
    n_new = sum(1 for k in runs if k[1] == "norm")
    lines += ["---", f"_Runs found: {n_old} unnormalized (historical), {n_new} normalized "
              f"(expected 144 each: compare 48 / trainmode 60 / pseudo 36)._", ""]

    DOC.write_text("\n".join(lines))
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"Wrote {DOC}\nWrote {CSV_OUT}")
    print(f"old runs: {n_old}, norm runs: {n_new}")


if __name__ == "__main__":
    main()
