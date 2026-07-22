#!/usr/bin/env python3
"""Generate docs/classifier_model_comparison.md — the classifier-model comparison:
linear head vs 1-NN vs MLP per sampling method, + the §9.8 soft-loss comparison
(soft-CE vs reverse KL) on the pseudo-all arm.

Protocols (DELTA_ROOTCAUSE pitfall 3 — each head runs ITS best protocol):
  linear head  warm-started across episodes (FINE_TUNE=True), best-val checkpoint
  MLP head     COLD per episode + best-val epoch selection (sklearn partial_fit
               path / the from_mlp_torch alias); raw (un-normalized) features
  1-NN         training-free, unit-L2 features, observed labels (or the stated
               target set)

Pulls entirely from results_master.csv — the same single source of truth as
RESULTS §4.1 — so it never drifts from the main tables.

Run:  python3 batch_utils/render_classifier_comparison.py
      (after build_results_master.py; needs sweeps 31085802/31085854/31085867 +
       eval_heads_fill.py collected)
"""
import os

from render_results_tables import BUDGETS, NOISES, cell, load, mean_of

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "classifier_model_comparison.md")

HEADER = ["Classifier model"] + [str(b) for b in BUDGETS]


def spec(label, sampling, model, labels, sweep, ref=False):
    """One display row. ref=True rows (Δ baselines) are marked with † in the doc."""
    return dict(label=label, sampling=sampling, model=model, labels=labels,
                sweep=sweep, ref=ref)


# (display sampling name, [row specs]) — the sampled point set is held fixed
# inside each block (selection never depends on the eval head), only the
# classifier / loss is swapped.
BLOCKS = [
    ("random", [
        spec("linear head (warm)", "random", "linear head", "observed", "30978366", ref=True),
        spec("1-NN", "random", "1-NN", "observed", "eval_1nn_all_arms"),
        spec("MLP head (cold+best-val)", "random", "MLP head", "observed", "31085867"),
    ]),
    ("probcover", [
        spec("linear head (warm)", "probcover", "linear head", "observed", "30978366", ref=True),
        spec("1-NN", "probcover", "1-NN", "observed", "31048649"),
        spec("MLP head (cold+best-val)", "probcover", "MLP head", "observed", "31085867"),
    ]),
    ("bayes_misp (Itai)", [
        spec("linear head (warm)", "bayes_misp", "linear head", "observed", "31084429", ref=True),
        spec("1-NN", "bayes_misp", "1-NN", "observed", "31084429"),
        spec("MLP head (cold+best-val)", "bayes_misp", "MLP head", "observed", "31085802"),
    ]),
    ("pseudo all (bayes_product + pseudo-labeling)", [
        spec("linear head (warm, norm), soft-CE", "bayes_product", "linear head", "pseudo (all)", "31048960", ref=True),
        spec("linear head (warm, norm), rKL", "bayes_product", "linear head", "pseudo (all), rkl", "31085854"),
        spec("MLP head torch (cold+best-val, raw), soft-CE", "bayes_product", "MLP head (torch)", "pseudo (all)", "31088628"),
        spec("MLP head torch (cold+best-val, raw), rKL", "bayes_product", "MLP head (torch)", "pseudo (all), rkl", "31088628"),
        spec("1-NN on the pseudo-all target set", "bayes_product", "1-NN", "pseudo (all) targets", "30978366"),
    ]),
]

# torch-vs-sklearn MLP architecture parity (probcover, clean, observed labels)
PARITY = [
    spec("MLP head sklearn (cold+best-val)", "probcover", "MLP head", "observed", "31085867"),
    spec("MLP head torch (cold+best-val)", "probcover", "MLP head (torch)", "observed", "31088628"),
]


def series(acc, s, noise):
    return [cell(acc.get((s["sampling"], s["model"], s["labels"], s["sweep"], noise, b)))
            for b in BUDGETS]


def block_table(acc, rows_spec, noise, deltas_vs_ref=True):
    lines = ["| " + " | ".join(HEADER) + " |",
             "|" + "|".join(["---"] * len(HEADER)) + "|"]
    ser = [(s, series(acc, s, noise)) for s in rows_spec]
    ref_row = next((vals for s, vals in ser if s["ref"]), None)
    # bold the column winner among present rows
    best = []
    for j in range(len(BUDGETS)):
        ms = [mean_of(vals[j]) for _, vals in ser]
        present = [m for m in ms if m is not None]
        best.append(max(present) if present else None)
    for s, vals in ser:
        disp = []
        for j, v in enumerate(vals):
            m = mean_of(v)
            if v is None:
                disp.append("—")
            elif best[j] is not None and m == best[j]:
                disp.append(f"**{v}**")
            else:
                disp.append(v)
        mark = "†" if s["ref"] else ""
        lines.append(f"| {s['label']}{mark} | " + " | ".join(disp) + " |")
        if deltas_vs_ref and not s["ref"] and ref_row is not None:
            ds = []
            for a, b in zip(ref_row, vals):
                ma, mb = mean_of(a), mean_of(b)
                ds.append("—" if ma is None or mb is None else f"{mb - ma:+.1f}")
            lines.append("| &nbsp;&nbsp;Δ vs linear† | " + " | ".join(ds) + " |")
    return "\n".join(lines)


DOC_HEAD = """---
title: "Classifier model comparison — linear head vs 1-NN vs MLP (CIFAR100)"
---

<!-- GENERATED by batch_utils/render_classifier_comparison.py from
     results/cifar100/results_master.csv — do not edit by hand. -->

**Companion to `bayesian_coverage_RESULTS.md`** (kept separate so the main results
file stays readable). Same data source (`results_master.csv`), same regime: CIFAR100
SimCLR features, tophat δ = 0.75, budgets 100–1000, mean±std over seeds {1,2,3}.
Interactive twin: `docs/classifier_model_comparison.html`.

**What this compares.** For each **sampling** method the selected point set is held
fixed (selection never depends on the eval head) and the classifier is swapped:

- **linear head** — `FeaturesNet(512→100)` probe, **warm-started** across episodes
  (`FINE_TUNE=True`), best-val checkpoint. Its established study protocol.
- **1-NN** — `KNeighborsClassifier(1)` on unit-L2 features, no training.
- **MLP head** — one hidden layer (256, ReLU), **cold per episode + best-val epoch
  selection** on the clean valSet, raw features. sklearn `partial_fit` path for the
  observed-label arms; the torch twin (`from_mlp_torch`) for soft-target/loss arms
  (arch parity checked below).

Per DELTA_ROOTCAUSE pitfall 3 each head runs **its own best protocol** — warm-start
helps the linear probe (+2–4 pts) and *hurts* the MLP (plasticity loss), so a shared
protocol would handicap one of them. The old final-iterate MLP rows (31084429) are
superseded; with best-val selection **the MLP no longer "inverts" under noise** —
most of that was protocol, not memorization.

**Loss comparison (§9.8).** The pseudo-all block also compares **soft-CE** against
**reverse KL(model‖target)** (`--soft_loss rkl`, mode-seeking) on both heads. KL is
skipped for `bayes_misp`: it trains on hard observed labels, where soft-CE ≡ CE and
reverse KL is degenerate.

**Reading.** † marks the Δ baseline row. **Bold** = column best within the block.
The recurring §8.3 pattern — 1-NN wins at low budget, the trained heads win once
converged — now has a third player: the cold+best-val MLP.
"""


def main():
    acc = load()
    parts = [DOC_HEAD]
    for noise in NOISES:
        tag = "clean" if noise == 0.0 else f"{int(noise * 100)}% label noise"
        parts.append(f"\n## Noise {noise} ({tag})\n")
        for name, rows_spec in BLOCKS:
            parts.append(f"\n### {name}\n")
            parts.append(block_table(acc, rows_spec, noise))
    parts.append("\n## Torch-vs-sklearn MLP architecture parity (probcover, clean, observed labels)\n")
    parts.append("Same widths (512→256→100), same cold+best-val protocol, adam ≈ same "
                 "hyperparams — the torch twin exists so soft targets / rKL can reach the "
                 "MLP. The two rows should track within noise:\n")
    parts.append(block_table(acc, PARITY, 0.0, deltas_vs_ref=False))
    open(OUT, "w").write("\n".join(parts) + "\n")
    print(f"wrote {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
