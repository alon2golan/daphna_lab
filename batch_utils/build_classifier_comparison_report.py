#!/usr/bin/env python3
"""Build the self-contained interactive classifier-model comparison report.

    python3 batch_utils/build_classifier_comparison_report.py
        -> docs/classifier_model_comparison.html   (single file, works offline)

The interactive twin of docs/classifier_model_comparison.md: for each sampling
method {random, probcover, bayes_misp, pseudo-all} the point set is held fixed
and the classifier is swapped {linear head, 1-NN, MLP}; the pseudo-all block
additionally compares soft-CE vs reverse-KL (§9.8) on both trained heads.

Look, palette and interaction are IDENTICAL to docs/bayes_coverage_summary.html:
the CSS and the plotly/explorer JS scaffolding are sliced verbatim out of
build_summary_report.TEMPLATE at build time (single source for the machinery,
the two reports can't drift apart visually). Data comes from
results/cifar100/results_master.csv, same sweep pinning as the .md renderer.
"""
import json
import os

import build_summary_report as bsr
from build_summary_report import GROUPS as _unused_groups  # noqa: F401 (import check)
from build_summary_report import arm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "cifar100", "results_master.csv")
OUT = os.path.join(REPO, "docs", "classifier_model_comparison.html")

NOISES = bsr.NOISES

# ---------------------------------------------------------------------------
# arm definitions — sweep pins mirror render_classifier_comparison.py BLOCKS
# ---------------------------------------------------------------------------
GROUPS = [
    ("rand",   "random selection"),
    ("pc",     "probcover selection"),
    ("misp",   "bayes_misp (Itai's bayes_al_2025 — diff max, α=0.5)"),
    ("pseudo", "pseudo-all (bayes_product selection + pseudo-labeling) — incl. the soft-CE vs reverse-KL loss comparison"),
    ("qa",     "protocol QA: torch-vs-sklearn MLP parity + the superseded final-iterate MLP"),
]

ARMS = [
    arm("r_lh", "rand", "random", "linear head", "observed", "30978366",
        "random · linear (warm)", "gray", on=True),
    arm("r_1nn", "rand", "random", "1-NN", "observed", "eval_1nn_all_arms",
        "random · 1-NN", "gray", dash="dash"),
    arm("r_mlp", "rand", "random", "MLP head", "observed", "31085867",
        "random · MLP (best-val)", "gray", markers=True, dash="dashdot", d_model="MLP head (best-val)"),

    arm("p_lh", "pc", "probcover", "linear head", "observed", "30978366",
        "probcover · linear (warm)", "blue", on=True),
    arm("p_1nn", "pc", "probcover", "1-NN", "observed", "31048649",
        "probcover · 1-NN", "blue", dash="dash", on=True),
    arm("p_mlp", "pc", "probcover", "MLP head", "observed", "31085867",
        "probcover · MLP (best-val)", "blue", markers=True, dash="dashdot", on=True,
        d_model="MLP head (best-val)"),

    arm("m_lh", "misp", "bayes_misp", "linear head", "observed", "31084429",
        "misp · linear (warm)", "yellow", d_sampling="bayes_misp (Itai)"),
    arm("m_1nn", "misp", "bayes_misp", "1-NN", "observed", "31084429",
        "misp · 1-NN", "yellow", dash="dash", d_sampling="bayes_misp (Itai)"),
    arm("m_mlp", "misp", "bayes_misp", "MLP head", "observed", "31085802",
        "misp · MLP (best-val)", "yellow", markers=True, dash="dashdot", on=True,
        d_sampling="bayes_misp (Itai)", d_model="MLP head (best-val)"),

    arm("ps_lh_ce", "pseudo", "bayes_product", "linear head", "pseudo (all)", "31048960",
        "pseudo-all · linear, soft-CE", "red", width=3, on=True,
        d_model="linear head (warm, norm)"),
    arm("ps_lh_rkl", "pseudo", "bayes_product", "linear head", "pseudo (all), rkl", "31085854",
        "pseudo-all · linear, rKL", "magenta", width=3,
        d_model="linear head (warm, norm)"),
    arm("ps_mlp_ce", "pseudo", "bayes_product", "MLP head (torch)", "pseudo (all)", "31088628",
        "pseudo-all · MLP, soft-CE", "orange", markers=True, dash="dashdot", on=True,
        d_model="MLP torch (best-val, raw)"),
    arm("ps_mlp_rkl", "pseudo", "bayes_product", "MLP head (torch)", "pseudo (all), rkl", "31088628",
        "pseudo-all · MLP, rKL", "violet", markers=True, dash="dashdot",
        d_model="MLP torch (best-val, raw)"),
    arm("ps_1nn", "pseudo", "bayes_product", "1-NN", "pseudo (all) targets", "30978366",
        "pseudo-all targets · 1-NN", "cyan", dash="dashdot",
        d_labels="pseudo (all) targets (offline)"),

    arm("qa_sk", "qa", "probcover", "MLP head", "observed", "31085867",
        "parity: sklearn MLP", "blue", dash="dot", markers=True,
        d_model="MLP sklearn (best-val)"),
    arm("qa_torch", "qa", "probcover", "MLP head (torch)", "observed", "31088628",
        "parity: torch MLP", "green", dash="dot", markers=True,
        d_model="MLP torch (best-val)"),
    arm("qa_old", "qa", "bayes_misp", "MLP head", "observed", "31084429",
        "misp · MLP final-iterate (superseded)", "yellow", dash="dot",
        d_sampling="bayes_misp (Itai)", d_model="MLP head (final-iterate ✗)"),
]


def load_series():
    """Same aggregation as build_summary_report.load_series, over THIS file's ARMS."""
    import csv as _csv
    import statistics as st
    from collections import defaultdict
    acc = defaultdict(list)
    with open(CSV) as fh:
        for r in _csv.DictReader(fh):
            key = (r["sampling"], r["model"], r["train_labels"], r["sweep_id"],
                   r["noise"], int(r["budget"]))
            acc[key].append(float(r["test_acc"]))
    out, missing = [], []
    for a in ARMS:
        series = {}
        for nz in NOISES:
            pts = sorted((b, v) for (s, m, l, w, n, b), v in
                         ((k, acc[k]) for k in acc)
                         if (s, m, l, w, n) == (a["sampling"], a["model"], a["labels"], a["sweep"], nz))
            if not pts:
                continue
            series[nz] = {
                "b": [b for b, _ in pts],
                "m": [round(st.mean(v), 2) for _, v in pts],
                "s": [round(st.stdev(v), 2) if len(v) > 1 else 0.0 for _, v in pts],
            }
        if not series:
            missing.append(a["id"])
        d = dict(a)
        d["series"] = series
        out.append(d)
    if missing:
        # Non-fatal: in-flight sweeps land episode by episode; the explorer
        # shows "—" for missing cells. Check the match keys if this persists
        # after the pinned sweeps are fully collected.
        print(f"WARNING: arms with no data yet: {missing}")
    return out


# ---------------------------------------------------------------------------
# body (CSS + JS scaffolding are sliced from build_summary_report.TEMPLATE)
# ---------------------------------------------------------------------------
BODY = r"""
<header>
  <div>
    <h1>Classifier model comparison — linear head vs 1-NN vs MLP</h1>
    <p class="sub">One question, held fixed per sampling method: <b>given the same selected
    points, which classifier extracts the most?</b> CIFAR-100 SimCLR features (512-d),
    tophat δ=0.75, budgets 100–1000, symmetric label noise {0, .2, .5, .8}, mean±std over
    seeds {1,2,3}. Companion to <code>classifier_model_comparison.md</code> and the main
    <code>bayes_coverage_summary.html</code>.</p>
  </div>
  <button id="themeBtn" title="Toggle light/dark">◐ theme</button>
</header>

<div class="toc"><ol>
  <li><a href="#s1">Protocols: each head gets its own best training recipe</a></li>
  <li><a href="#s2">Headline: the three classifiers per sampling method</a></li>
  <li><a href="#s3">The MLP "noise inversion" was a protocol artifact</a></li>
  <li><a href="#s4">Loss comparison: soft-CE vs reverse KL on pseudo-all</a></li>
  <li><a href="#s5">Explorer — all arms, all noise levels (interactive)</a></li>
  <li><a href="#s6">Provenance &amp; how to regenerate</a></li>
</ol></div>

<h2 id="s1">1 · Protocols: each head gets its own best training recipe</h2>
<p>The two trained heads never ran the same protocol historically, and that manufactured a fake
crossover (DELTA_ROOTCAUSE, pitfall 3). The decomposition on identical seed-1 lSets
(clean, probcover selection):</p>
<div class="minitables"><table>
<caption>test acc — protocol decomposition (offline replay, clean)</caption>
<tr><th>protocol</th><th>100</th><th>300</th><th>500</th><th>700</th><th>1000</th></tr>
<tr><td>linear, cold, final epoch</td><td>25.2</td><td>35.8</td><td>40.7</td><td>41.8</td><td>44.9</td></tr>
<tr><td>linear, <b>warm</b> ← protocol used</td><td>25.6</td><td>40.0</td><td>43.6</td><td>44.4</td><td>47.1</td></tr>
<tr><td>MLP, cold, final ← superseded</td><td>36.1</td><td>39.9</td><td>41.8</td><td>43.4</td><td>44.9</td></tr>
<tr><td>MLP, <b>cold, best-val</b> ← protocol used</td><td>36.0</td><td>40.0</td><td>43.3</td><td>43.8</td><td>46.7</td></tr>
<tr><td>MLP, warm, best-val (worst)</td><td>36.0</td><td>39.9</td><td>41.3</td><td>41.3</td><td>42.4</td></tr>
</table></div>
<p><b>Warm-start helps the linear probe (+2–4 pts, effectively extra optimization time) and
hurts the MLP (plasticity loss, −4.3 @1000)</b> — so every arm here runs its head's own best
recipe: <b>linear = warm-started</b> across episodes (the established study protocol),
<b>MLP = cold per episode + best-val epoch selection</b> against the same clean valSet the
probe checkpoints on, <b>1-NN = training-free</b> (unit-L2 features). The first best-val MLP
rerun (sweep 31085570) accidentally kept the model <i>warm</i> across episodes
(<code>partial_fit</code> continues where <code>.fit()</code> re-initializes — the best-val
epoch collapsing to ~5 in late episodes was the tell); it was fixed with a cold clone per
episode and rerun as sweep 31085802, which reproduces the decomposition's cold+best-val
numbers (46.8 vs predicted 46.7 @1000 clean).</p>

<h2 id="s2">2 · Headline: the three classifiers per sampling method</h2>
<p>The §8.3 pattern — <b>1-NN wins at low budget</b> (nothing to under-train at ~1 label/class),
<b>trained heads win once converged</b> — now has a third player, and the cold+best-val MLP
takes the low-budget crown from 1-NN on clean data while staying competitive late:</p>
<div class="chart"><div class="chart-title">probcover selection: linear (warm) vs 1-NN vs MLP (cold + best-val)</div>
<div class="chart-sub">the same selected points, three ways to classify with them</div>
<div id="chartA" style="height:560px"></div></div>

<h2 id="s3">3 · The MLP "noise inversion" was a protocol artifact</h2>
<p>Under the superseded final-iterate protocol the MLP appeared to <i>memorize label noise</i>
(9.9 @1000 noise .8 — below the linear head's 12.7 — and R3–R5 carried the same story). With
best-val selection on a clean valSet the picture flips: the cold MLP stops training before
memorization and lands <b>above</b> the linear protocol under every noise level
(misp selection @1000: 42.5 vs 37.1 at noise .2, 35.0 vs 24.6 at .5, 17.2 vs 9.9 at .8 —
old-vs-new MLP rows toggleable in the explorer's QA group). Takeaway: <i>"high-capacity head
memorizes noise" claims are protocol-sensitive — check the iterate-selection rule before
believing them.</i></p>
<div class="chart"><div class="chart-title">bayes_misp selection: the MLP head before and after the protocol fix</div>
<div class="chart-sub">final-iterate (superseded) vs cold+best-val, vs the warm linear head</div>
<div id="chartB" style="height:560px"></div></div>

<h2 id="s4">4 · Loss comparison: soft-CE vs reverse KL on pseudo-all</h2>
<p>The soft-target arms train with soft cross-entropy (= forward KL up to the targets'
entropy — identical gradients). The variant with content is <b>reverse KL(model‖target)</b>:
mode-seeking, it lets the classifier commit to a single class inside a spread-out posterior
instead of averaging over it (NEXT_STEPS §9.8) — potentially a better fit for impure balls at
high noise. Both losses run on both heads of the <b>pseudo-all</b> arm (the recommended
configuration): linear (warm, normalized inputs) and torch-MLP (cold + best-val, raw inputs,
architecture-parity-checked against the sklearn head in the QA group). KL is skipped for
<code>bayes_misp</code>: it trains on hard observed labels, where soft-CE ≡ CE and reverse KL
is degenerate.</p>
<div class="chart"><div class="chart-title">pseudo-all: {linear, MLP} × {soft-CE, reverse KL}</div>
<div class="chart-sub">paired seeds; the 1-NN on the pseudo-all target set (cyan) is the training-free reference</div>
<div id="chartC" style="height:560px"></div></div>

<h2 id="s5">5 · Explorer — all arms, all noise levels</h2>
<p class="hint">Click a row to add/remove its curve. Hover a row to spotlight its curve; hover the
chart for exact mean±std of every visible arm. Values are test accuracy (%), mean±std over seeds;
<b>bold = column best</b>.</p>
<div class="controls">
  <span class="seg" id="noiseSeg"></span>
  <label class="ctl"><input type="checkbox" id="bandsCk"> ±std bands</label>
  <label class="ctl"><input type="checkbox" id="fullXCk" style="display:none"></label>
  <button class="btn" id="resetBtn">reset selection</button>
  <button class="btn" id="clearBtn">clear all</button>
</div>
<div class="explorer">
  <div class="tblwrap"><table class="res" id="resTable"></table>
    <p class="hint">Sweep pinning mirrors <code>docs/classifier_model_comparison.md</code>
    (generated from the same <code>results_master.csv</code>). The QA group holds the
    torch-vs-sklearn parity rows and the superseded final-iterate MLP — off by default.</p></div>
  <div class="chartwrap"><div class="chart" style="margin:0">
    <div class="chart-title" id="expTitle"></div>
    <div class="chart-sub">test accuracy (%) vs annotation budget</div>
    <div id="expChart" style="height:620px"></div></div></div>
</div>

<h2 id="s6">6 · Provenance &amp; how to regenerate</h2>
<table class="prov">
<tr><th>Rows</th><th>Source</th><th>Grid</th></tr>
<tr><td>linear heads</td><td>sweeps 30978366 / 31084429 / 31048960 (pseudo-all, norm)</td><td>warm protocol, per §4 tables</td></tr>
<tr><td>1-NN</td><td>sweep 31048649 + offline replays (<code>eval_1nn_all_arms.py</code>, <code>eval_heads_fill.py</code> — gate-checked to ≤0.05)</td><td>all four samplings + pseudo-all target set</td></tr>
<tr><td>MLP (sklearn, cold+best-val)</td><td>sweeps 31085802 (misp) + 31085867 (random, probcover)</td><td>noise × 3 seeds</td></tr>
<tr><td>MLP (torch twin) + rKL loss</td><td>sweep 31085854</td><td>pseudo-all × {LH,MLP} × {soft-CE,rKL} + parity gate</td></tr>
</table>
<p class="small" style="margin-top:12px">
Single source of truth: <code>results/cifar100/results_master.csv</code>
(<code>batch_utils/build_results_master.py</code>). This file was generated by
<code>python3 batch_utils/build_classifier_comparison_report.py</code> — rerun after refreshing
the master CSV. The markdown twin comes from <code>batch_utils/render_classifier_comparison.py</code>.
Protocol investigation: <code>docs/bayesian_coverage_DELTA_ROOTCAUSE.md</code> (pitfall 3).
Generated 2026-07-16.</p>
"""

RENDER_ALL = r"""
function renderAll(){
  smallMultiples("chartA", ["p_lh","p_1nn","p_mlp"], 1000);
  smallMultiples("chartB", ["m_lh","m_mlp","qa_old"], 1000);
  smallMultiples("chartC", ["ps_lh_ce","ps_lh_rkl","ps_mlp_ce","ps_mlp_rkl","ps_1nn"], 1000);
  renderTable(); renderChart();
}
renderAll();
"""


def slice_template():
    """Extract the CSS block and the JS scaffolding from the summary report's
    TEMPLATE so the two reports share one implementation of look + interaction."""
    t = bsr.TEMPLATE
    css = t.split("<style>", 1)[1].split("</style>", 1)[0]
    js_start = t.index('"use strict";')
    js_end = t.index("/* ---------- render everything ---------- */")
    js = t[js_start:js_end]
    return css, js


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Classifier model comparison — linear head vs 1-NN vs MLP (CIFAR-100)</title>
<style>__CSS__</style>
</head>
<body>
<div class="wrap">
__BODY__
</div>
<script>__PLOTLY__</script>
<script>
__JS__
__RENDER_ALL__
</script>
</body>
</html>
"""


def main():
    plotly_js = None
    for p in bsr.PLOTLY_CANDIDATES:
        if os.path.exists(p):
            plotly_js = open(p, encoding="utf-8").read()
            break
    if plotly_js is None:
        raise SystemExit(f"plotly.min.js not found in any of: {bsr.PLOTLY_CANDIDATES}")

    css, js = slice_template()
    arms = load_series()
    payload = {"groups": GROUPS, "arms": arms}
    html = (TEMPLATE
            .replace("__CSS__", css)
            .replace("__BODY__", BODY)
            .replace("__PLOTLY__", plotly_js)
            .replace("__JS__", js)
            .replace("__RENDER_ALL__", RENDER_ALL)
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False)))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(OUT, REPO)}  ({os.path.getsize(OUT)/1e6:.1f} MB, {len(arms)} arms)")


if __name__ == "__main__":
    main()
