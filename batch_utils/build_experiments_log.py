#!/usr/bin/env python3
"""Build the chronological experiments log of the Bayesian-coverage study.

    python3 batch_utils/build_experiments_log.py
        -> docs/bayes_experiments_log.html   (single file, works offline)

One entry per experiment, in the order they ran: what was tested, which sweep
ran it, a comparison graph, the key numbers, and the conclusion (adopted /
not adopted / measurement). Companion to docs/bayes_coverage_summary.html
(the story + interactive explorer); this file is the lab notebook.

Data: results/cifar100/results_master.csv for every trained arm, plus the
offline-study CSVs (results/bayes_pseudo/*, noise_rate_estimators,
delta_replay). Palette/theme reused from build_summary_report.py.
"""
import csv
import json
import os
import statistics as st
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "batch_utils"))
from build_summary_report import HUES, color, PLOTLY_CANDIDATES  # noqa: E402

CSV = os.path.join(REPO, "results", "cifar100", "results_master.csv")
OUT = os.path.join(REPO, "docs", "bayes_experiments_log.html")
BUDGETS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------
def load_master():
    acc = defaultdict(list)
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            dlt = f'{float(r.get("delta") or 0.75):g}'
            key = (r["sampling"], r["model"], r["train_labels"], r["sweep_id"],
                   dlt, r["noise"], int(r["budget"]))
            acc[key].append(float(r["test_acc"]))
    return acc


ACC = load_master()


def series(sampling, model, labels, sweep, noise, delta="0.75"):
    """(budgets, means, stds) for one arm at one noise level."""
    pts = sorted((b, v) for (s, m, l, w, d, n, b), v in ACC.items()
                 if (s, m, l, w, d, n) == (sampling, model, labels, sweep,
                                           f'{float(delta):g}', noise))
    if not pts:
        raise SystemExit(f"no data: {sampling}/{model}/{labels}/{sweep}/δ{delta}/n{noise}")
    return ([b for b, _ in pts],
            [round(st.mean(v), 2) for _, v in pts],
            [round(st.stdev(v), 2) if len(v) > 1 else 0.0 for _, v in pts])


def tr(label, hue, x, y, s=None, dash="solid", tint=False, width=2):
    c = color(hue, tint)
    return dict(label=label, x=x, y=y, s=s or [0] * len(x),
                cl=c["light"], cd=c["dark"], dash=dash, w=width)


def arm_tr(label, hue, sampling, model, labels, sweep, noise,
           dash="solid", tint=False, width=2, delta="0.75"):
    b, m, s = series(sampling, model, labels, sweep, noise, delta)
    return tr(label, hue, b, m, s, dash, tint, width)


def csv_rows(path):
    with open(os.path.join(REPO, path)) as fh:
        return list(csv.DictReader(fh))


def mean_by(rows, keyf, valf):
    g = defaultdict(list)
    for r in rows:
        v = valf(r)
        if v is not None:
            g[keyf(r)].append(v)
    return {k: st.mean(v) for k, v in g.items()}


def panel(pid, title, traces, ytitle="test accuracy (%)",
          xtitle="annotation budget (labels)", zero=False, xticks=None):
    return dict(id=pid, title=title, traces=traces, ytitle=ytitle,
                xtitle=xtitle, zero=zero, xticks=xticks or BUDGETS[::2] + [1000])


# ---------------------------------------------------------------------------
# offline-study panels
# ---------------------------------------------------------------------------
def nw_delta_panels(path, arms, pid_prefix, noises=("0.0", "0.8")):
    """Δ-vs-paired-baseline panels from a collect_nw_* deltas CSV."""
    rows = csv_rows(path)
    out = []
    for nz in noises:
        traces = []
        for a, hue, dash in arms:
            m = mean_by([r for r in rows if r["noise"] == nz and r["arm"] == a],
                        lambda r: int(r["labeled"]), lambda r: float(r["delta"]))
            xs = sorted(m)
            traces.append(tr(a, hue, xs, [round(m[x], 2) for x in xs], dash=dash))
        out.append(panel(f"{pid_prefix}_{nz.replace('.', '')}",
                         f"noise {nz} — Δ test acc vs paired baseline",
                         traces, ytitle="Δ accuracy (pts)", zero=True))
    return out


def bakeoff_panel(ntag, nz_label, pid):
    rows = [r for r in csv_rows("results/cifar100/noise_rate_estimators/noise_rate_estimators.csv")
            if r["ntag"] == ntag]
    traces = []
    for col, label, hue, dash in [("ds_em", "DS-EM", "blue", "solid"),
                                  ("loo_corr", "impurity-corrected LOO", "green", "solid"),
                                  ("raw_loo", "raw LOO", "orange", "dash")]:
        m = mean_by(rows, lambda r: int(r["budget"]), lambda r, c=col: float(r[c]))
        xs = sorted(m)
        traces.append(tr(label, hue, xs, [round(m[x], 3) for x in xs], dash=dash))
    real = st.mean(float(r["realized"]) for r in rows)
    traces.append(tr("realized noise (truth)", "gray",
                     [100, 1000], [round(real, 3)] * 2, dash="dot"))
    return panel(pid, f"noise {nz_label} — estimated noise rate vs budget",
                 traces, ytitle="estimated noise rate")


def delta_replay_panel():
    rows = [r for r in csv_rows("results/cifar100/delta_replay/delta_replay.csv")
            if int(r["budget"]) == 1000]
    deltas = [0.5, 0.6, 0.75, 0.9, 1.1]
    traces = []
    for nz, hue in [("0.0", "blue"), ("0.2", "cyan"), ("0.5", "orange"), ("0.8", "red")]:
        m = mean_by([r for r in rows if r["noise"] == nz],
                    lambda r: float(r["delta"]), lambda r: 100 * float(r["post_acc"]))
        traces.append(tr(f"denoising @ noise {nz}", hue, deltas,
                         [round(m[d], 1) for d in deltas]))
    m = mean_by([r for r in rows if r["noise"] == "0.0"],
                lambda r: float(r["delta"]), lambda r: 100 * float(r["cov_frac"]))
    traces.append(tr("coverage (clean)", "gray", deltas,
                     [round(m[d], 1) for d in deltas], dash="dot"))
    return panel("e15_replay", "stage 1 replay @ B=1000 — labeled-plane denoising accuracy vs δ",
                 traces, ytitle="%", xtitle="ball radius δ", xticks=deltas)


# ---------------------------------------------------------------------------
# the experiments, chronological
# ---------------------------------------------------------------------------
V_POS, V_NEG, V_NEU = "pos", "neg", "neu"

EXPERIMENTS = [
    dict(
        date="2026-07-02", verdict=V_POS, sweeps="30978366 (48/48) · CIFAR-10 pilot 30969432",
        title="Bayesian-coverage selection: product vs additive vs the baselines",
        tested="Does noise-aware coverage selection (per-point posteriors from δ-ball neighbors + annotator confusion matrix) beat ProbCover and random, and which combination rule — product of experts or additive pseudo-counts?",
        explain="All arms share the linear-probe evaluator and train on observed labels, so this isolates <b>selection</b>. The additive curve here is the <i>original</i> acquisition (later found buggy — see the 07-14 fix entry).",
        conclusion="<b>bayes_product ≈ probcover on clean data and holds up under noise; additive collapses at low budget</b> (13.4 @100 clean — diagnosed 07-14 as an acquisition-ceiling bug, not the combination rule itself). Product became the study default; the product ≫ additive gate passed.",
        panels=lambda: [
            panel("e1_n0", "noise 0 (clean)", [
                arm_tr("random", "gray", "random", "linear head", "observed", "30978366", "0.0"),
                arm_tr("probcover", "blue", "probcover", "linear head", "observed", "30978366", "0.0"),
                arm_tr("bayes_product", "cyan", "bayes_product", "linear head", "observed", "30978366", "0.0"),
                arm_tr("bayes_additive (orig)", "cyan", "bayes_additive", "linear head", "observed", "30978366", "0.0", dash="dash", tint=True)]),
            panel("e1_n5", "noise 0.5", [
                arm_tr("random", "gray", "random", "linear head", "observed", "30978366", "0.5"),
                arm_tr("probcover", "blue", "probcover", "linear head", "observed", "30978366", "0.5"),
                arm_tr("bayes_product", "cyan", "bayes_product", "linear head", "observed", "30978366", "0.5"),
                arm_tr("bayes_additive (orig)", "cyan", "bayes_additive", "linear head", "observed", "30978366", "0.5", dash="dash", tint=True)]),
        ]),

    dict(
        date="2026-07-02", verdict=V_POS, sweeps="30980646 (60/60)",
        title="Training on the posterior: five label modes",
        tested="Once the posterior exists, what should the classifier train on? observed labels, the posterior MAP relabel, the full soft posterior, an LOO-filtered clean subset, or LOO-reweighted labels.",
        explain="Same bayes_product selection everywhere; only the training targets change. This is where the posterior first pays off on the <b>classification plane</b>, not just selection.",
        conclusion="<b>soft/map posterior training beats observed 2.2× at noise .8</b> and beats Netta's NPC+AUM at noise ≥ .5; <code>filtered</code>/<code>reweighted</code> break at low budget (LOO threshold too aggressive — still an open engineering guard). soft_posterior became the noise-regime recommendation until pseudo-labeling superseded it.",
        panels=lambda: [
            panel("e2_n5", "noise 0.5", [
                arm_tr("observed", "cyan", "bayes_product", "linear head", "observed", "30978366", "0.5"),
                arm_tr("map_posterior", "yellow", "bayes_product", "linear head", "map_posterior", "30980646", "0.5", dash="dash"),
                arm_tr("soft_posterior", "magenta", "bayes_product", "linear head", "soft_posterior", "30980646", "0.5"),
                arm_tr("filtered", "violet", "bayes_product", "linear head", "filtered", "30980646", "0.5", dash="dash"),
                arm_tr("reweighted", "violet", "bayes_product", "linear head", "reweighted", "30980646", "0.5", dash="dot")]),
            panel("e2_n8", "noise 0.8", [
                arm_tr("observed", "cyan", "bayes_product", "linear head", "observed", "30978366", "0.8"),
                arm_tr("map_posterior", "yellow", "bayes_product", "linear head", "map_posterior", "30980646", "0.8", dash="dash"),
                arm_tr("soft_posterior", "magenta", "bayes_product", "linear head", "soft_posterior", "30980646", "0.8"),
                arm_tr("filtered", "violet", "bayes_product", "linear head", "filtered", "30980646", "0.8", dash="dash"),
                arm_tr("reweighted", "violet", "bayes_product", "linear head", "reweighted", "30980646", "0.8", dash="dot")]),
        ]),

    dict(
        date="2026-07-09", verdict=V_POS, sweeps="31047129 (36/36)",
        title="Pseudo-labeling: train on all the points (§7.5)",
        tested="The posterior covers ~50% of the pool — use it to pseudo-label unlabeled points and train on labeled + covered pseudo rows. Scopes: all covered points, top-k per class linear (lin10) or aggressive (agg50).",
        explain="TRAIN_LABEL_MODE=pseudo. The training set jumps from B rows to thousands, all soft-targeted by the posterior.",
        conclusion="<b>Best arm at every noise level including clean</b> (47.8 vs 45.1 observed @1000 clean; 45.8 vs NPC+AUM 43.3 @noise .2 — closed Netta's last lead). <code>all</code> wins at low budget, <code>agg50</code> at noise .8. <code>pseudo all</code> became the champion training mode.",
        panels=lambda: [
            panel("e3_n0", "noise 0 (clean)", [
                arm_tr("observed", "cyan", "bayes_product", "linear head", "observed", "30978366", "0.0"),
                arm_tr("pseudo all", "orange", "bayes_product", "linear head", "pseudo (all)", "31047129", "0.0"),
                arm_tr("pseudo lin10", "orange", "bayes_product", "linear head", "pseudo (lin10)", "31047129", "0.0", dash="dot", tint=True),
                arm_tr("pseudo agg50", "orange", "bayes_product", "linear head", "pseudo (agg50)", "31047129", "0.0", dash="dash", tint=True)]),
            panel("e3_n8", "noise 0.8", [
                arm_tr("observed", "cyan", "bayes_product", "linear head", "observed", "30978366", "0.8"),
                arm_tr("pseudo all", "orange", "bayes_product", "linear head", "pseudo (all)", "31047129", "0.8"),
                arm_tr("pseudo lin10", "orange", "bayes_product", "linear head", "pseudo (lin10)", "31047129", "0.8", dash="dot", tint=True),
                arm_tr("pseudo agg50", "orange", "bayes_product", "linear head", "pseudo (agg50)", "31047129", "0.8", dash="dash", tint=True)]),
        ]),

    dict(
        date="2026-07-09", verdict=V_POS, sweeps="31048649 (12/12); L2-norm fix commit 0fb2a91",
        title="ProbCover + 1-NN evaluator (paper framework)",
        tested="Evaluate the same selected points with a training-free 1-NN classifier instead of the linear probe — the ProbCover paper's own low-budget evaluation.",
        explain="1-NN needs <b>L2-normalized</b> features (raw features cost −18 pts @100; the first sweep 31048609 ran raw and was discarded).",
        conclusion="<b>1-NN 38.1 vs training 23.6 @100 clean</b> (matches the paper's ~35) and wins budgets 100–300 at every noise level; the trained probe wins once converged. Established the convention: never a bare “probcover” row again — always “+ 1-NN” or “+ training”.",
        panels=lambda: [
            panel("e4_n0", "noise 0 (clean)", [
                arm_tr("probcover + training", "blue", "probcover", "linear head", "observed", "30978366", "0.0"),
                arm_tr("probcover + 1-NN", "blue", "probcover", "1-NN", "observed", "31048649", "0.0", dash="dash")]),
            panel("e4_n5", "noise 0.5", [
                arm_tr("probcover + training", "blue", "probcover", "linear head", "observed", "30978366", "0.5"),
                arm_tr("probcover + 1-NN", "blue", "probcover", "1-NN", "observed", "31048649", "0.5", dash="dash")]),
        ]),

    dict(
        date="2026-07-10", verdict=V_POS, sweeps="31048960 (144/144)",
        title="Normalized classifier inputs — every arm re-run",
        tested="L2-normalize the classifier's input features (scale-preserving: x/‖x‖ × mean train norm — plain unit-L2 collapses low-budget training 13.3 → 3.3 @100). Paired seeds against every raw arm.",
        explain="The effect flips with budget: at ≤300 labels the labeled-only arms <i>lose</i> (fixed-LR convergence artifact), crossover ~350–400, and by 600–1000 every arm gains +2..+8. <code>pseudo all</code> — always enough training rows to converge — is immune and gains ~+1 at every budget.",
        conclusion="<b>Adopted for the champion arm</b>: pseudo all (norm) reaches 38.4 @100 and 48.4 @1000 clean — the study's red curve. Honest note: normalization helps <code>observed</code> most at ≥400 labels, so part of the posterior-training margin on raw features was norm pathology, not label noise (margins shrink 2.2× → 1.5× but don't vanish).",
        panels=lambda: [
            panel("e5_n0", "noise 0 (clean) — pseudo all, raw vs normalized", [
                arm_tr("pseudo all (raw)", "orange", "bayes_product", "linear head", "pseudo (all)", "31047129", "0.0"),
                arm_tr("pseudo all (norm)", "red", "bayes_product", "linear head", "pseudo (all)", "31048960", "0.0", width=3)]),
            panel("e5_n5", "noise 0.5 — pseudo all, raw vs normalized", [
                arm_tr("pseudo all (raw)", "orange", "bayes_product", "linear head", "pseudo (all)", "31047129", "0.5"),
                arm_tr("pseudo all (norm)", "red", "bayes_product", "linear head", "pseudo (all)", "31048960", "0.5", width=3)]),
        ]),

    dict(
        date="2026-07-11", verdict=V_NEG, sweeps="31050702 + 31050847 (36/36)",
        title="NW out-of-ball pseudo-targets (§7.9) — constant weight",
        tested="Extend pseudo-labeling beyond the covered set: give UNcovered points Nadaraya–Watson soft targets from the labeled set, at constant weight w ∈ {1.0, 0.3, 0.1} vs the paired covered-only baseline.",
        explain="If far-field targets help, accuracy should rise with w (dose-response). It didn't: the loss was weight-insensitive.",
        conclusion="<b>Not adopted.</b> Clean −2.1..−1.1 and noise .2 −1.0 at <i>every</i> weight (far targets are confidently wrong, not over-weighted); noise .8 +0.5..+0.9 is label smoothing, not signal. Covered-only pseudo-labeling stands. Spin-off kept: in-ball RBF-weighted product exponents (replay 87→96.7 denoising) — still open.",
        panels=lambda: nw_delta_panels("results/bayes_pseudo/nw_oobw_deltas.csv",
                                       [("nw_w1.0", "red", "solid"),
                                        ("nw_w0.3", "orange", "dash"),
                                        ("nw_w0.1", "yellow", "dot")], "e6"),
    ),

    dict(
        date="2026-07-12", verdict=V_NEU, sweeps="offline replay — eval_1nn_all_arms.py (840 cells, sanity-gated)",
        title="Training-free evaluators over every arm (§8.3 Q2 / §9.1)",
        tested="Replace the trained probe with 1-NN or soft-kNN (Nadaraya–Watson) evaluators for every §4 arm — including 1-NN on the posterior's denoised labels (“denoise → 1-NN”).",
        explain="Pure evaluation replay on the recorded selections; no training anywhere.",
        conclusion="<b>Denoise-then-1-NN ≈ the best trained arm at noise .8 with zero training</b> (28.0 vs 29.2 @1000); NW κ=3 is the best training-free evaluator on clean (46.4 @1000). Kept as a benchmark family (explorer group “training-free”), not adopted as the main pipeline.",
        panels=lambda: [
            panel("e7_n0", "noise 0 (clean) — training-free evaluators", [
                arm_tr("probcover + 1-NN", "blue", "probcover", "1-NN", "observed", "31048649", "0.0", dash="dash"),
                arm_tr("product + 1-NN", "cyan", "bayes_product", "1-NN", "observed", "eval_1nn_all_arms", "0.0", dash="dash"),
                arm_tr("product + NW κ=3", "cyan", "bayes_product", "NW (k=3)", "observed", "eval_1nn_all_arms", "0.0", dash="longdash"),
                arm_tr("denoise → 1-NN", "magenta", "bayes_product", "1-NN", "posterior-argmax", "eval_1nn_all_arms", "0.0", dash="dashdot")]),
            panel("e7_n8", "noise 0.8 — denoise→1-NN vs the best trained arm", [
                arm_tr("pseudo all (norm), trained", "red", "bayes_product", "linear head", "pseudo (all)", "31048960", "0.8", width=3),
                arm_tr("denoise → 1-NN (no training)", "magenta", "bayes_product", "1-NN", "posterior-argmax", "eval_1nn_all_arms", "0.8", dash="dashdot")]),
        ]),

    dict(
        date="2026-07-13", verdict=V_NEG, sweeps="31058388 (12/12)",
        title="w(d) distance-decayed out-of-ball weight (§9.1)",
        tested="§7.9's rescue attempt: instead of a constant weight, ramp the out-of-ball weight down with distance w(d), so only the trustworthy near-shell contributes.",
        explain="Paired against the same covered-only baseline. Even the best shell (55% NW accuracy) has to beat the covered diet it dilutes — which is 72% accurate.",
        conclusion="<b>Not adopted — out-of-ball closed for good.</b> Clean −1.35..−1.11 at every budget: any dilution of the covered diet hurts. Covered-only pseudo-labeling confirmed as the right scope.",
        panels=lambda: nw_delta_panels("results/bayes_pseudo/nw_wdist_deltas.csv",
                                       [("nw_wramp", "violet", "solid")], "e8",
                                       noises=("0.0", "0.2")),
    ),

    dict(
        date="2026-07-14", verdict=V_POS, sweeps="31066809 (12/12); commit 5e6fb8e",
        title="bayes_additive acquisition fix",
        tested="Alon flagged the additive arm's low-budget collapse. Diagnosis: additive selection reused the product's 1−maxP ceiling on a posterior that never saturates → covered points never retire → the greedy clumps.",
        explain="Fix: Itai's saturating ceiling f₁ = (max π + 1)/(Σ π + 1) — the dominant-class mass after exactly one in-ball label (= BAYES_MISP diff_method='max' for tophat). One clean label now retires a point.",
        conclusion="<b>Fixed:</b> additive 13.4 → 24.8 @100 clean, now ≈ probcover ≈ product at low budget for every noise. Product still ≫ at high budget — that residual gap is an intrinsic additive weakness (impure balls keep a residual deficit), so the product-selection gate holds.",
        panels=lambda: [
            panel("e9_n0", "noise 0 (clean) — additive before/after the fix", [
                arm_tr("additive (orig acquisition)", "cyan", "bayes_additive", "linear head", "observed", "30978366", "0.0", dash="dot", tint=True),
                arm_tr("additive (fixed)", "cyan", "bayes_additive", "linear head", "observed", "31066809", "0.0", dash="dash"),
                arm_tr("product", "cyan", "bayes_product", "linear head", "observed", "30978366", "0.0"),
                arm_tr("probcover", "blue", "probcover", "linear head", "observed", "30978366", "0.0")]),
            panel("e9_n5", "noise 0.5", [
                arm_tr("additive (orig acquisition)", "cyan", "bayes_additive", "linear head", "observed", "30978366", "0.5", dash="dot", tint=True),
                arm_tr("additive (fixed)", "cyan", "bayes_additive", "linear head", "observed", "31066809", "0.5", dash="dash"),
                arm_tr("product", "cyan", "bayes_product", "linear head", "observed", "30978366", "0.5"),
                arm_tr("probcover", "blue", "probcover", "linear head", "observed", "30978366", "0.5")]),
        ]),

    dict(
        date="2026-07-14", verdict=V_NEU, sweeps="offline — eval_noise_rate_estimators.py",
        title="Noise-rate estimator bake-off (§9.4)",
        tested="Every run feeds the annotator's TRUE noise rate into the confusion matrix — an oracle assumption. Can it be estimated from the labeled data alone? Candidates: raw LOO disagreement, impurity-corrected LOO, Dawid–Skene EM, Netta's q̂.",
        explain="Measurement only, on the recorded runs — nothing wired into production.",
        conclusion="<b>DS-EM is the best assumption-free estimator</b> (separates noise 0 vs .2 at every budget; +~0.25 bias at low noise); <b>impurity-corrected LOO is the most accurate</b> (MAE ≈ .016 @1000) but needs a clean anchor; raw LOO saturates at the ball-impurity floor; Netta's q̂ under-estimates at low budget. Adoption decision deferred — the “unknown noise” story is still open.",
        panels=lambda: [bakeoff_panel("n20", "0.2", "e10_n2"),
                        bakeoff_panel("n50", "0.5", "e10_n5")],
    ),

    dict(
        date="2026-07-15", verdict=V_NEU, sweeps="offline replica — pc1nn_delta_replay.py",
        title="Why is probcover+1-NN 38.1 @100? — δ root cause",
        tested="Our 38.1 beats the paper's ~34–35. Candidate causes: embedding, selection code, seeds, or the ball radius δ. An offline replica (reproduces the sweep to the decimal) swept δ.",
        explain="Embedding (Itai's ≈ ours ≈ TypiClust's), selection and seeds all ruled out — the gap is <i>entirely</i> δ: 0.75 is ProbCover's CIFAR-10 radius; the paper's purity criterion gives δ=0.50 on CIFAR-100 (our balls are only 39% pure at 0.75).",
        conclusion="<b>38.1 is δ-inflated</b> — the accuracy-vs-δ curve peaks exactly at our 0.75 (large δ selects typical, TypiClust-like points); the paper's ~34–35 lands at δ≈0.60. Kept as a footnote at first; the full δ decision came from §9.2 (see the 07-22 entry — 0.75 survived end-to-end).",
        panels=lambda: [panel("e11_curve", "offline replica @ B=100, clean — 1-NN accuracy vs δ",
                              [tr("probcover + 1-NN @100", "blue",
                                  [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
                                  [23.0, 26.3, 28.6, 32.3, 34.7, 36.1, 37.8, 38.1, 37.5, 36.2, 30.8])],
                              xtitle="ball radius δ", xticks=[0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9])],
    ),

    dict(
        date="2026-07-16", verdict=V_POS, sweeps="31078514 (3/3 parity) + 31084429 (24/24); commit acf7d2d",
        title="Reproducing Itai's numbers + porting BAYES_MISP",
        tested="Itai's bayes_al_2025 reported ~45 @1000 where we saw ~41. Is his “bayes” method better, or is something else different?",
        explain="Root cause was twofold: his “bayes” is <code>BAYES_MISP</code> with diff_method='max' (additive-in-spirit, needs --sparse_K) — not our bayes_additive — and his evaluator is an sklearn <b>MLP head</b> (hidden 256), not a linear probe. Both were ported; parity confirmed (our misp 45.0 @1000 ≈ his 45.2/44.9).",
        conclusion="<b>Port faithful; no selection advantage.</b> Under the same linear head, bayes_misp ≈ probcover ≈ bayes_product at every noise level — his edge was the evaluator, which became its own investigation (next entry).",
        panels=lambda: [
            panel("e12_n0", "noise 0 (clean) — same linear head", [
                arm_tr("bayes_misp (Itai)", "yellow", "bayes_misp", "linear head", "observed", "31084429", "0.0", dash="dashdot", width=3),
                arm_tr("probcover", "blue", "probcover", "linear head", "observed", "30978366", "0.0"),
                arm_tr("bayes_product", "cyan", "bayes_product", "linear head", "observed", "30978366", "0.0")]),
            panel("e12_n5", "noise 0.5 — same linear head", [
                arm_tr("bayes_misp (Itai)", "yellow", "bayes_misp", "linear head", "observed", "31084429", "0.5", dash="dashdot", width=3),
                arm_tr("probcover", "blue", "probcover", "linear head", "observed", "30978366", "0.5"),
                arm_tr("bayes_product", "cyan", "bayes_product", "linear head", "observed", "30978366", "0.5")]),
        ]),

    dict(
        date="2026-07-16", verdict=V_POS, sweeps="31085802 (12/12) + 31085867 (24/24); 31085570 invalid (accidental warm-start)",
        title="The MLP head: cold vs warm, final-iterate vs early stopping",
        tested="Itai's MLP head under a fair protocol. Two knobs that turned out to matter more than the architecture: warm-starting across episodes, and which iterate you keep (final vs best-val epoch).",
        explain="The linear probe <i>wants</i> warm-starting (+2.2 @1000, log-verified); the MLP is <i>hurt</i> by it (−4.3 @1000, plasticity loss) and wants a cold start per episode. With the final iterate the MLP appeared to “memorize noise” and invert under noise — best-val epoch selection (early stopping on the clean valSet) recovers +5..+10 under noise and retracts that reading entirely. Each head gets its own best protocol; comparing across mismatched protocols fabricates effects (DELTA_ROOTCAUSE pitfall 3).",
        conclusion="<b>MLP (cold + best-val) leads the warm linear probe at every noise and budget on observed labels</b> (@1000: 46.8/42.5/35.0/17.2 vs 46.7/41.8/31.2/13.6 at noise 0/.2/.5/.8) — real learned nonlinearity, +10 @100 clean over the converged linear probe. The “MLP memorizes noise” caveat is retracted as a protocol artifact.",
        panels=lambda: [
            panel("e13_n0", "noise 0 (clean) — probcover selection, heads compared fairly", [
                arm_tr("linear head (warm)", "blue", "probcover", "linear head", "observed", "30978366", "0.0"),
                arm_tr("MLP head (cold + best-val)", "violet", "probcover", "MLP head", "observed", "31085867", "0.0", width=3),
                arm_tr("1-NN", "blue", "probcover", "1-NN", "observed", "31048649", "0.0", dash="dash")]),
            panel("e13_n8", "noise 0.8 — bayes_misp selection", [
                arm_tr("linear head (warm)", "yellow", "bayes_misp", "linear head", "observed", "31084429", "0.8", dash="dashdot"),
                arm_tr("MLP head (cold + best-val)", "violet", "bayes_misp", "MLP head", "observed", "31085802", "0.8", width=3)]),
        ]),

    dict(
        date="2026-07-16 → 07-22", verdict=V_NEG, sweeps="31085854 (torch rows invalid: wd 200× too strong) → wd-fix 31088628 (27/27); commit 1f628a4",
        title="Stacking the MLP on pseudo-all (§9.7) + reverse-KL loss (§9.8)",
        tested="Two bets at once. §9.7: MLP + observed does 46.8 with zero posterior machinery, pseudo-all + linear does 48.4 — do the gains stack? §9.8: for soft targets, does mode-seeking reverse KL(model‖target) beat soft-CE? (Forward KL needs no experiment: with fixed targets, <b>forward KL = soft-CE − H(target)</b>, a constant — identical gradients, identical optima. Soft-CE <i>is</i> forward KL.)",
        explain="Needed a torch MLP twin (sklearn can't fit soft targets). The first sweep's torch rows were invalid — sklearn's alpha=3e-2 mapped 1:1 onto Adam weight_decay is ~200× too strong (sklearn divides the penalty by batch size), making the head degrade with budget. With wd=1.5e-4 the arch-parity gate passes (torch 34.5→44.9 vs sklearn 34.9→46.8).",
        conclusion="<b>Both negative.</b> §9.7: MLP+pseudo-all trails linear+pseudo-all at every noise and budget (clean −1, noise .5/.8 ≈ −5, deficit grows with noise — the posterior's targets already linearize the problem; extra capacity only adds variance). §9.8: rKL ties soft-CE under noise, loses −0.7..−1.7 on clean, slightly worse on the MLP — soft-CE stands. <b>Champion config unchanged.</b>",
        panels=lambda: [
            panel("e14_par", "parity gate — torch twin vs sklearn head (probcover, clean, observed)", [
                arm_tr("MLP sklearn (cold+best-val)", "violet", "probcover", "MLP head", "observed", "31085867", "0.0", width=3),
                arm_tr("MLP torch, wd fixed", "violet", "probcover", "MLP head (torch)", "observed", "31088628", "0.0", dash="dash")]),
            panel("e14_n5", "noise 0.5 — the stack + loss test on pseudo-all", [
                arm_tr("linear (norm), soft-CE", "red", "bayes_product", "linear head", "pseudo (all)", "31048960", "0.5", width=3),
                arm_tr("linear (norm), rKL", "red", "bayes_product", "linear head", "pseudo (all), rkl", "31085854", "0.5", dash="dash", tint=True),
                arm_tr("MLP torch, soft-CE", "violet", "bayes_product", "MLP head (torch)", "pseudo (all)", "31088628", "0.5", dash="longdash"),
                arm_tr("MLP torch, rKL", "violet", "bayes_product", "MLP head (torch)", "pseudo (all), rkl", "31088628", "0.5", dash="dot", tint=True)]),
        ]),

    dict(
        date="2026-07-22", verdict=V_POS, sweeps="offline eval_delta_replay.py (60 cells, gate 10/10 ×12) + 31108061 (24/24); commits 94a4a78 + 355faa2",
        title="The ball radius δ, properly: replay + paired training sweep (§9.2)",
        tested="δ=0.75 was inherited, never re-examined, and the 07-15 entry showed it inflates the 1-NN cell. Stage 1: replay the recorded runs at δ ∈ {0.5, 0.6, 0.75, 0.9, 1.1} (posterior quality only). Stage 2: re-run the champion arm end-to-end at δ {0.6, 0.9}, where δ shapes selection too.",
        explain="Replay: purity and calibration are monotone in −δ (clean covered-accuracy 83.9% @0.5 vs 66.7% @0.75), but denoising and noise-ID AUC peak at 0.75 for noise ≥ .5 (tiny balls can't cross-vote), and coverage collapses below it (5%/17%/50% of the pool at δ .5/.6/.75). The open bet — δ=0.6's purity on clean data — went to training.",
        conclusion="<b>δ=0.75 confirmed at every noise level and every budget</b> (@1000 clean 48.4 vs 47.9 (δ.6) / 40.3 (δ.9); noise .8: 29.2 vs 24.4/24.8). Volume beats purity while coverage is scarce; impurity poisons cheap coverage — 0.75 sits at the crossing. No retune for CIFAR-100/SimCLR; new datasets/embeddings should re-run the cheap replay first.",
        panels=lambda: [
            delta_replay_panel(),
            panel("e15_n0", "stage 2 training — champion arm per δ, noise 0 (clean)", [
                arm_tr("δ = 0.75 (baseline)", "red", "bayes_product", "linear head", "pseudo (all)", "31048960", "0.0", width=3),
                arm_tr("δ = 0.6", "red", "bayes_product", "linear head", "pseudo (all)", "31108061", "0.0", dash="dash", tint=True, delta="0.6"),
                arm_tr("δ = 0.9", "red", "bayes_product", "linear head", "pseudo (all)", "31108061", "0.0", dash="dot", tint=True, delta="0.9")]),
            panel("e15_n8", "stage 2 training — champion arm per δ, noise 0.8", [
                arm_tr("δ = 0.75 (baseline)", "red", "bayes_product", "linear head", "pseudo (all)", "31048960", "0.8", width=3),
                arm_tr("δ = 0.6", "red", "bayes_product", "linear head", "pseudo (all)", "31108061", "0.8", dash="dash", tint=True, delta="0.6"),
                arm_tr("δ = 0.9", "red", "bayes_product", "linear head", "pseudo (all)", "31108061", "0.8", dash="dot", tint=True, delta="0.9")]),
        ]),
]


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
VERDICT_META = {
    V_POS: ("ADOPTED / POSITIVE", "vpos"),
    V_NEG: ("NEGATIVE — NOT ADOPTED", "vneg"),
    V_NEU: ("MEASUREMENT", "vneu"),
}

CSS = r"""
:root{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --border:rgba(11,11,11,.10); --accent:#2a78d6;
  --pos:#008300; --neg:#c93a39; --neu:#2a78d6;
  --posbg:rgba(0,131,0,.08); --negbg:rgba(201,58,57,.08); --neubg:rgba(42,120,214,.07);
}
html[data-theme="dark"]{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --border:rgba(255,255,255,.10); --accent:#3987e5;
  --pos:#39b039; --neg:#e66767; --neu:#3987e5;
  --posbg:rgba(57,176,57,.10); --negbg:rgba(230,103,103,.10); --neubg:rgba(57,135,229,.12);
}
@media (prefers-color-scheme: dark){
  html:not([data-theme]){
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --border:rgba(255,255,255,.10); --accent:#3987e5;
    --pos:#39b039; --neg:#e66767; --neu:#3987e5;
    --posbg:rgba(57,176,57,.10); --negbg:rgba(230,103,103,.10); --neubg:rgba(57,135,229,.12);
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
     font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 28px 80px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;padding:18px 0 6px;flex-wrap:wrap}
h1{font-size:26px;line-height:1.25;margin:0}
.sub{color:var(--ink2);margin:.35em 0 0;max-width:860px}
button{background:var(--surface);color:var(--ink);border:1px solid var(--border);
       border-radius:8px;padding:6px 12px;font-size:13.5px;cursor:pointer}
.toc{border:1px solid var(--border);border-radius:12px;background:var(--surface);
     padding:14px 20px;margin:20px 0 8px;font-size:14.5px}
.toc table{border-collapse:collapse;width:100%}
.toc td{padding:3px 10px 3px 0;vertical-align:top}
.toc .d{color:var(--muted);white-space:nowrap;font-variant-numeric:tabular-nums}
.toc a{color:var(--ink);text-decoration:none}
.toc a:hover{color:var(--accent)}
.badge{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.4px;
       border-radius:5px;padding:1px 7px;vertical-align:1px;white-space:nowrap}
.vpos{color:var(--pos);background:var(--posbg)}
.vneg{color:var(--neg);background:var(--negbg)}
.vneu{color:var(--neu);background:var(--neubg)}
.exp{border:1px solid var(--border);border-radius:14px;background:var(--surface);
     padding:22px 26px;margin:26px 0}
.exp h2{font-size:19px;margin:0 0 2px}
.meta{color:var(--muted);font-size:13.5px;margin:0 0 12px}
.meta code{font-size:12.5px}
.q{margin:.6em 0;max-width:960px}
.q b.klabel{color:var(--ink2);font-weight:600}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px;margin:14px 0 4px}
.fig{border:1px solid var(--border);border-radius:10px;padding:8px 6px 0;background:var(--page)}
.fig .t{font-size:13px;color:var(--ink2);padding:2px 10px 4px;font-weight:600}
.concl{border-left:4px solid var(--neu);background:var(--neubg);border-radius:0 10px 10px 0;
       padding:10px 16px;margin:14px 0 2px;max-width:none}
.exp.vneg .concl{border-left-color:var(--neg);background:var(--negbg)}
.exp.vpos .concl{border-left-color:var(--pos);background:var(--posbg)}
code{background:rgba(127,127,127,.12);border-radius:4px;padding:0 4px;font-size:14px}
.small{font-size:13.5px;color:var(--ink2)}
@media(max-width:820px){.figs{grid-template-columns:1fr}}
"""

JS = r"""
const CHROME = {
  light:{ink:"#0b0b0b", ink2:"#52514e", grid:"#e1e0d9", paper:"rgba(0,0,0,0)"},
  dark:{ink:"#ffffff", ink2:"#c3c2b7", grid:"#2c2c2a", paper:"rgba(0,0,0,0)"}
};
function mode(){
  const t = document.documentElement.getAttribute("data-theme");
  if (t) return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function renderAll(){
  const m = mode(), c = CHROME[m];
  FIGS.forEach(f => {
    const traces = f.traces.map(t => ({
      x: t.x, y: t.y, name: t.label, mode: "lines+markers", type: "scatter",
      line: {color: m === "dark" ? t.cd : t.cl, dash: t.dash, width: t.w},
      marker: {size: 5},
      error_y: t.s && t.s.some(v => v > 0)
        ? {type: "data", array: t.s, visible: true, thickness: 1, width: 2} : undefined,
      hovertemplate: t.label + " · %{x}: %{y}" + (t.s ? "" : "") + "<extra></extra>"
    }));
    const shapes = f.zero ? [{type:"line", xref:"paper", x0:0, x1:1, y0:0, y1:0,
                              line:{color:c.ink2, width:1, dash:"dot"}}] : [];
    Plotly.newPlot(f.id, traces, {
      margin:{l:52,r:12,t:6,b:44}, height:320,
      paper_bgcolor:c.paper, plot_bgcolor:c.paper,
      font:{color:c.ink2, size:11.5},
      xaxis:{title:{text:f.xtitle,font:{size:11.5}}, gridcolor:c.grid, zeroline:false,
             tickvals:f.xticks},
      yaxis:{title:{text:f.ytitle,font:{size:11.5}}, gridcolor:c.grid, zeroline:false},
      legend:{orientation:"h", y:-0.32, x:0, font:{size:10.5}},
      shapes: shapes, hovermode:"x unified"
    }, {displayModeBar:false, responsive:true});
  });
}
document.getElementById("themeBtn").addEventListener("click", () => {
  const cur = mode();
  document.documentElement.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
  renderAll();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
renderAll();
"""


def build():
    plotly_js = None
    for p in PLOTLY_CANDIDATES:
        if os.path.exists(p):
            plotly_js = open(p, encoding="utf-8").read()
            break
    if plotly_js is None:
        raise SystemExit("plotly.min.js not found")

    figs, entries_html, toc_rows = [], [], []
    for i, e in enumerate(EXPERIMENTS, 1):
        panels = e["panels"]()
        vlabel, vcls = VERDICT_META[e["verdict"]]
        anchor = f"e{i}"
        toc_rows.append(
            f'<tr><td class="d">{e["date"]}</td>'
            f'<td><a href="#{anchor}">{i}. {e["title"]}</a></td>'
            f'<td><span class="badge {vcls}">{vlabel}</span></td></tr>')
        fig_divs = "".join(
            f'<div class="fig"><div class="t">{p["title"]}</div><div id="{p["id"]}"></div></div>'
            for p in panels)
        for p in panels:
            figs.append(p)
        entries_html.append(f"""
<section class="exp {vcls}" id="{anchor}">
  <h2>{i} · {e['title']} <span class="badge {vcls}">{vlabel}</span></h2>
  <p class="meta">{e['date']} &nbsp;·&nbsp; sweep(s): <code>{e['sweeps']}</code></p>
  <p class="q"><b class="klabel">What we tested —</b> {e['tested']}</p>
  <p class="q"><b class="klabel">How —</b> {e['explain']}</p>
  <div class="figs">{fig_divs}</div>
  <p class="q concl"><b class="klabel">Conclusion —</b> {e['conclusion']}</p>
</section>""")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bayesian Coverage — chronological experiments log</title>
<style>{CSS}</style>
<script>{plotly_js}</script>
</head>
<body><div class="wrap">
<header>
  <div>
    <h1>Bayesian Coverage — chronological experiments log</h1>
    <p class="sub">Every experiment of the study in the order it ran: what it tested, the sweep
    that ran it, the comparison graph, and the conclusion. CIFAR-100, SimCLR features,
    test accuracy mean±std over seeds {{1,2,3}}, symmetric annotator noise {{0, .2, .5, .8}}.
    The story-form companion (theory + interactive explorer over all arms):
    <code>bayes_coverage_summary.html</code>.</p>
  </div>
  <button id="themeBtn" title="Toggle light/dark">◐ theme</button>
</header>
<div class="toc"><table>{''.join(toc_rows)}</table></div>
{''.join(entries_html)}
<p class="small">Generated by <code>batch_utils/build_experiments_log.py</code> from
<code>results/cifar100/results_master.csv</code> + the offline-study CSVs
(<code>results/bayes_pseudo/</code>, <code>noise_rate_estimators/</code>,
<code>delta_replay/</code>). Regenerate after refreshing the master CSV. Generated 2026-07-22.</p>
</div>
<script>const FIGS = {json.dumps(figs)};</script>
<script>{JS}</script>
</body></html>"""

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(OUT, REPO)}  "
          f"({os.path.getsize(OUT)/1e6:.1f} MB, {len(EXPERIMENTS)} experiments, {len(figs)} panels)")


if __name__ == "__main__":
    build()
