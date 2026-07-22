#!/usr/bin/env python3
"""Build the self-contained interactive summary report of the Bayesian-coverage study.

    python3 batch_utils/build_summary_report.py
        -> docs/bayes_coverage_summary.html   (single file, works offline)

The report tells the story of the work (noise-aware ProbCover -> per-point
posteriors -> product-vs-additive -> training modes -> pseudo-labeling ->
normalization) with the R2-R5 tables rendered as an *interactive explorer*:
click a table row to add/remove its curve from the chart beside it; hover the
chart for exact mean±std; switch noise levels with the segmented control.

Data source: results/cifar100/results_master.csv (the single source of truth,
built by build_results_master.py). Arm definitions mirror
render_results_tables.py's ROW_SPEC — same sweep pinning (e.g. bayes_additive
comes from the fixed rerun 31066809, never the superseded 30978366 rows).

Everything is inlined (plotly.min.js + data JSON), so the output opens from
file:// with no network and can be mailed / dropped in Slack as-is.
"""
import csv
import json
import os
import statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "cifar100", "results_master.csv")
OUT = os.path.join(REPO, "docs", "bayes_coverage_summary.html")
PLOTLY_CANDIDATES = [
    os.path.join(REPO, "venv", "lib", "python3.11", "site-packages",
                 "plotly", "package_data", "plotly.min.js"),
    "/cs/labs/daphna/alongol/envs/venv311/lib/python3.11/site-packages/plotly/package_data/plotly.min.js",
]

NOISES = ["0.0", "0.2", "0.5", "0.8"]
TBL_BUDGETS = [100, 300, 500, 700, 1000]

# ---------------------------------------------------------------------------
# palette (validated against the dataviz reference palette, light+dark modes)
# ---------------------------------------------------------------------------
HUES = {  # name -> (light, dark)
    "blue":    ("#2a78d6", "#3987e5"),
    # true cyan replacing the reference teal-green aqua; dark step teal-leaning
    # so it clears the normal-vision floor against dark blue (#3987e5)
    "cyan":    ("#00b3d9", "#0ba69d"),
    "yellow":  ("#eda100", "#c98500"),
    "violet":  ("#4a3aa7", "#9085e9"),
    # dark step darkened from the reference #d55181: keeps deutan separation
    # from the cyan product hue when soft_posterior and product co-plot in dark
    "magenta": ("#e87ba4", "#b13b64"),
    "orange":  ("#eb6834", "#d95926"),
    "red":     ("#e34948", "#e66767"),
    "green":   ("#008300", "#008300"),
    "gray":    ("#898781", "#898781"),
}


def _mix(hexcol, other, t):
    a = [int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def color(hue, tint=False):
    lt, dk = HUES[hue]
    if tint:  # normalized-input variants: same hue, lightened + markers
        return {"light": _mix(lt, "#ffffff", 0.35), "dark": _mix(dk, "#ffffff", 0.30)}
    return {"light": lt, "dark": dk}


# ---------------------------------------------------------------------------
# arm definitions — mirrors render_results_tables.py ROW_SPEC (same sweep pins)
# ---------------------------------------------------------------------------
def arm(aid, group, sampling, model, labels, sweep, short, hue, dash="solid",
        width=2, tint=False, markers=False, on=False,
        d_sampling=None, d_model=None, d_labels=None, delta="0.75"):
    return dict(id=aid, group=group, sampling=sampling, model=model, labels=labels,
                sweep=sweep, short=short, color=color(hue, tint), dash=dash,
                width=width, markers=markers, on=on, delta=delta,
                d_sampling=d_sampling or sampling, d_model=d_model or model,
                d_labels=d_labels or labels)


GROUPS = [
    ("base",   "Baselines"),
    ("sel",    "Bayesian coverage — selection only (train on observed labels)"),
    ("corr",   "Bayesian coverage + label-correction training"),
    ("pseudo", "Bayesian coverage + pseudo-labeling"),
    ("postcls", "Posterior as classifier (no trained model)"),
    ("netta",  "Netta NPC / AUM"),
    ("misp",   "BAYES_MISP port (Itai's bayes_al_2025 — diff max, α=0.5, tophat δ=0.75; sweeps 31084429 + 31085802 + 31078514; MLP row = cold start + best-val selection)"),
    ("heads",  "Classifier heads on the champion diet (§9.7/§9.8; sweeps 31085867 + 31088628 + 31085854) — MLP is cold + best-val, its own best protocol"),
    ("delta",  "δ ablation (§9.2, sweep 31108061) — champion arm re-run with the ball radius at 0.6 / 0.9 vs the study's 0.75"),
    ("norm",   "Normalized classifier inputs (sweep 31048960) — same arms, scale-preserving L2"),
    ("tfree",  "Training-free evaluator study (offline replay)"),
]

ARMS = [
    arm("rand_lin", "base", "random", "linear head", "observed", "30978366",
        "random", "gray", on=True),
    arm("pc_lin", "base", "probcover", "linear head", "observed", "30978366",
        "probcover", "blue", on=True),
    arm("pc_1nn", "base", "probcover", "1-NN", "observed", "31048649",
        "probcover + 1-NN", "blue", dash="dash", on=True),

    arm("add_lin", "sel", "bayes_additive", "linear head", "observed", "31066809",
        "additive", "cyan", dash="dash"),
    arm("prod_lin", "sel", "bayes_product", "linear head", "observed", "30978366",
        "product", "cyan", on=True),

    arm("map", "corr", "bayes_product", "linear head", "map_posterior", "30980646",
        "map_posterior", "yellow", dash="dash"),
    arm("soft", "corr", "bayes_product", "linear head", "soft_posterior", "30980646",
        "soft_posterior", "magenta", on=True),
    arm("filt", "corr", "bayes_product", "linear head", "filtered", "30980646",
        "filtered", "violet", dash="dash"),
    arm("rew", "corr", "bayes_product", "linear head", "reweighted", "30980646",
        "reweighted", "violet", dash="dot"),

    arm("ps_lin10", "pseudo", "bayes_product", "linear head", "pseudo (lin10)", "31047129",
        "pseudo lin10", "orange", dash="dot"),
    arm("ps_agg50", "pseudo", "bayes_product", "linear head", "pseudo (agg50)", "31047129",
        "pseudo agg50", "orange", dash="dash"),
    arm("ps_all", "pseudo", "bayes_product", "linear head", "pseudo (all)", "31047129",
        "pseudo all", "orange"),
    arm("ps_all_n", "pseudo", "bayes_product", "linear head", "pseudo (all)", "31048960",
        "pseudo all (norm)", "red", width=3, on=True, d_model="linear head (norm)"),

    arm("post_arg", "postcls", "bayes_product", "posterior-argmax", "—", "test_posterior_acc",
        "posterior argmax", "cyan", dash="dot"),

    arm("pc_aum", "netta", "probcover", "linear head", "AUM clean-train", "30863746",
        "PC + AUM", "green", dash="dash"),
    arm("npc_sel", "netta", "npc", "linear head", "observed (sel. only)", "30863746",
        "NPC (sel. only)", "green", dash="dot"),
    arm("npc_aum", "netta", "npc", "linear head", "AUM clean-train", "30863746",
        "NPC + AUM", "green", on=True),

    # -- BAYES_MISP port (Itai's method run in our repo) ----------------------
    # linear-head row is directly comparable to every other curve; the MLP-head
    # rows live ~10 pts above the linear protocol (different evaluator, not a
    # better selector) — compare them only against each other.
    arm("misp_lin", "misp", "bayes_misp", "linear head", "observed", "31084429",
        "bayes_misp (linear)", "yellow", dash="dashdot", width=3, on=True,
        d_sampling="bayes_misp (Itai)"),
    # pinned to the best-val rerun 31085802 (from_mlp_sklearn now checkpoints on
    # the clean valSet); the final-iterate 31084429 MLP rows stay in the CSV only.
    arm("misp_mlp", "misp", "bayes_misp", "MLP head", "observed", "31085802",
        "bayes_misp (MLP, best-val)", "yellow", width=3, markers=True,
        d_sampling="bayes_misp (Itai)", d_model="MLP head (best-val)"),
    arm("pc_mlp", "misp", "probcover", "MLP head", "observed", "31078514",
        "probcover (MLP, seed 1)", "blue", dash="dashdot", markers=True,
        d_model="MLP head (seed 1)"),
    arm("add_mlp", "misp", "bayes_additive", "MLP head", "observed", "31078514",
        "additive (MLP, seed 1)", "cyan", dash="dot", markers=True,
        d_model="MLP head (seed 1)"),

    # -- classifier heads (§9.7 stack test + §9.8 loss test) -------------------
    # sklearn MLP (cold+best-val) on observed labels — the best observed-label head
    arm("h_pc_mlp", "heads", "probcover", "MLP head", "observed", "31085867",
        "probcover + MLP", "blue", dash="longdash", markers=True,
        d_model="MLP head (cold+best-val)"),
    arm("h_rand_mlp", "heads", "random", "MLP head", "observed", "31085867",
        "random + MLP", "gray", dash="longdash", markers=True,
        d_model="MLP head (cold+best-val)"),
    # torch twin parity gate (wd-fix 31088628; the 31085854 torch rows were invalid)
    arm("h_pc_mlpt", "heads", "probcover", "MLP head (torch)", "observed", "31088628",
        "probcover + MLP torch (parity)", "blue", dash="dot", markers=True,
        d_model="MLP head torch (parity gate)"),
    # §9.7: the stack test — pseudo-all posterior targets through the MLP
    arm("h_ps_mlpt", "heads", "bayes_product", "MLP head (torch)", "pseudo (all)", "31088628",
        "pseudo all + MLP torch", "orange", dash="longdash", markers=True, on=True,
        d_model="MLP head torch"),
    # §9.8: reverse-KL loss on both heads (soft-CE ≡ forward KL for fixed targets)
    arm("h_ps_rkl", "heads", "bayes_product", "linear head", "pseudo (all), rkl", "31085854",
        "pseudo all (norm), rKL", "red", dash="dash", markers=True,
        d_model="linear head (norm)", d_labels="pseudo (all), reverse-KL"),
    arm("h_ps_mlpt_rkl", "heads", "bayes_product", "MLP head (torch)", "pseudo (all), rkl", "31088628",
        "pseudo all + MLP torch, rKL", "orange", dash="dot", markers=True,
        d_model="MLP head torch", d_labels="pseudo (all), reverse-KL"),

    # -- §9.2 δ ablation (champion arm, delta shapes selection AND posterior) --
    arm("d_06", "delta", "bayes_product", "linear head", "pseudo (all)", "31108061",
        "pseudo all (norm), δ=0.6", "red", dash="dash", tint=True, markers=True,
        delta="0.6", d_model="linear head (norm)", d_labels="pseudo (all), δ=0.6"),
    arm("d_09", "delta", "bayes_product", "linear head", "pseudo (all)", "31108061",
        "pseudo all (norm), δ=0.9", "red", dash="dot", tint=True, markers=True,
        delta="0.9", d_model="linear head (norm)", d_labels="pseudo (all), δ=0.9"),

    # -- normalized-input reruns (paired seeds vs the raw originals) ----------
    arm("n_rand", "norm", "random", "linear head", "observed", "31048960",
        "random (norm)", "gray", tint=True, markers=True, d_model="linear head (norm)"),
    arm("n_pc", "norm", "probcover", "linear head", "observed", "31048960",
        "probcover (norm)", "blue", tint=True, markers=True, d_model="linear head (norm)"),
    arm("n_prod", "norm", "bayes_product", "linear head", "observed", "31048960",
        "product (norm)", "cyan", tint=True, markers=True, d_model="linear head (norm)"),
    arm("n_map", "norm", "bayes_product", "linear head", "map_posterior", "31048960",
        "map (norm)", "yellow", dash="dash", tint=True, markers=True, d_model="linear head (norm)"),
    arm("n_soft", "norm", "bayes_product", "linear head", "soft_posterior", "31048960",
        "soft (norm)", "magenta", tint=True, markers=True, d_model="linear head (norm)"),
    arm("n_filt", "norm", "bayes_product", "linear head", "filtered", "31048960",
        "filtered (norm)", "violet", dash="dash", tint=True, markers=True, d_model="linear head (norm)"),
    arm("n_rew", "norm", "bayes_product", "linear head", "reweighted", "31048960",
        "reweighted (norm)", "violet", dash="dot", tint=True, markers=True, d_model="linear head (norm)"),
    arm("n_lin10", "norm", "bayes_product", "linear head", "pseudo (lin10)", "31048960",
        "pseudo lin10 (norm)", "orange", dash="dot", tint=True, markers=True, d_model="linear head (norm)"),
    arm("n_agg50", "norm", "bayes_product", "linear head", "pseudo (agg50)", "31048960",
        "pseudo agg50 (norm)", "orange", dash="dash", tint=True, markers=True, d_model="linear head (norm)"),

    # -- training-free evaluator study (eval_1nn_all_arms replay) -------------
    arm("t_rand1nn", "tfree", "random", "1-NN", "observed", "eval_1nn_all_arms",
        "random + 1-NN", "gray", dash="dash"),
    arm("t_prod1nn", "tfree", "bayes_product", "1-NN", "observed", "eval_1nn_all_arms",
        "product + 1-NN", "cyan", dash="dash"),
    arm("t_arg1nn", "tfree", "bayes_product", "1-NN", "posterior-argmax", "eval_1nn_all_arms",
        "denoise → 1-NN", "cyan", dash="dashdot"),
    arm("t_nw3", "tfree", "bayes_product", "NW (k=3)", "observed", "eval_1nn_all_arms",
        "product + NW κ=3", "cyan", dash="longdash"),
]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def load_series():
    acc = defaultdict(list)
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            # delta column added 2026-07-22 (§9.2); normalize "0.6"/"0.60" etc.
            dlt = f'{float(r.get("delta") or 0.75):g}'
            key = (r["sampling"], r["model"], r["train_labels"], r["sweep_id"],
                   dlt, r["noise"], int(r["budget"]))
            acc[key].append(float(r["test_acc"]))
    out, missing = [], []
    for a in ARMS:
        a_delta = f'{float(a["delta"]):g}'
        series = {}
        for nz in NOISES:
            pts = sorted((b, v) for (s, m, l, w, d, n, b), v in
                         ((k, acc[k]) for k in acc)
                         if (s, m, l, w, d, n) == (a["sampling"], a["model"], a["labels"], a["sweep"], a_delta, nz))
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
        raise SystemExit(f"arms with NO data in {CSV}: {missing} — check the match keys")
    return out


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bayesian Coverage — noise-aware active learning on CIFAR-100</title>
<style>
:root{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,.10);
  --wash:rgba(42,120,214,.07); --accent:#2a78d6;
}
html[data-theme="dark"]{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
  --wash:rgba(57,135,229,.14); --accent:#3987e5;
}
@media (prefers-color-scheme: dark){
  html:not([data-theme]){
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
    --wash:rgba(57,135,229,.14); --accent:#3987e5;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
     font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:24px 28px 80px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;
       padding:18px 0 6px;flex-wrap:wrap}
h1{font-size:26px;line-height:1.25;margin:0}
h2{font-size:21px;margin:2.4em 0 .5em;padding-top:.6em;border-top:1px solid var(--grid)}
h3{font-size:17px;margin:1.6em 0 .4em}
p{max-width:900px}
.sub{color:var(--ink2);margin:.35em 0 0;max-width:900px}
.small{font-size:13px;color:var(--muted)}
a{color:var(--accent)}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.88em;
     background:var(--wash);padding:.08em .32em;border-radius:4px}
#themeBtn{background:var(--surface);border:1px solid var(--border);color:var(--ink2);
          border-radius:8px;padding:6px 12px;cursor:pointer;font:13px system-ui}
/* formulas */
.math{background:var(--surface);border:1px solid var(--border);border-radius:10px;
      padding:12px 18px;margin:14px 0;font-size:16.5px;overflow-x:auto;white-space:nowrap}
.math em, em.v{font-style:italic}
.math .op{color:var(--ink2);padding:0 .15em}
.note{border-left:3px solid var(--accent);background:var(--wash);border-radius:0 8px 8px 0;
      padding:10px 16px;margin:14px 0;max-width:880px}
/* charts */
.chart{background:var(--surface);border:1px solid var(--border);border-radius:12px;
       padding:8px 6px 2px;margin:18px 0}
.chart-title{font-size:14px;font-weight:600;padding:8px 14px 0;color:var(--ink)}
.chart-sub{font-size:12.5px;color:var(--muted);padding:0 14px 4px}
/* explorer */
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:14px 0}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:9px;overflow:hidden;background:var(--surface)}
.seg button{border:0;background:transparent;color:var(--ink2);padding:7px 15px;cursor:pointer;
            font:13.5px system-ui;border-right:1px solid var(--border)}
.seg button:last-child{border-right:0}
.seg button.sel{background:var(--wash);color:var(--ink);font-weight:650}
.ctl{font-size:13.5px;color:var(--ink2);display:inline-flex;align-items:center;gap:6px;cursor:pointer}
.btn{background:var(--surface);border:1px solid var(--border);color:var(--ink2);border-radius:8px;
     padding:5px 11px;cursor:pointer;font:13px system-ui}
.btn:hover{background:var(--wash)}
.explorer{display:flex;gap:18px;align-items:flex-start}
.explorer .tblwrap{flex:1 1 auto;min-width:0;overflow-x:auto}
.explorer .chartwrap{flex:0 0 470px;min-width:380px;position:sticky;top:10px}
@media (max-width:1080px){.explorer{flex-direction:column}.explorer .chartwrap{position:static;width:100%;flex:1 1 auto}}
table.res{border-collapse:collapse;font-size:12.5px;background:var(--surface);
          border:1px solid var(--border);border-radius:10px;width:100%}
table.res th,table.res td{padding:4px 6px;text-align:right;border-bottom:1px solid var(--grid);
                          white-space:nowrap;font-variant-numeric:tabular-nums}
table.res th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--surface);z-index:1}
table.res td.k,table.res th.k{text-align:left}
table.res tr.sep td{color:var(--ink2);font-style:italic;background:var(--page);font-size:12.5px;padding-top:9px}
table.res tr.armrow{cursor:pointer}
table.res tr.armrow:hover{background:var(--wash)}
table.res tr.armrow.on td.k1{box-shadow:inset 3px 0 0 var(--rowcol)}
table.res tr.armrow.off{opacity:.62}
table.res td b{font-weight:700}
.sw{display:inline-block;vertical-align:middle}
.hint{font-size:12.5px;color:var(--muted);margin:6px 0 0}
/* static mini tables */
.minitables{display:flex;gap:14px;overflow-x:auto;padding:4px 0}
.minitables table{border-collapse:collapse;font-size:12.5px;background:var(--surface);
                  border:1px solid var(--border)}
.minitables caption{font-size:12.5px;color:var(--ink2);padding:4px;font-weight:600;white-space:nowrap}
.minitables th,.minitables td{padding:3px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;
                              font-variant-numeric:tabular-nums}
.minitables td:first-child,.minitables th:first-child{text-align:left}
details{margin:12px 0}
summary{cursor:pointer;color:var(--ink2);font-weight:600;font-size:14.5px}
ol.next li{margin:.5em 0;max-width:880px}
table.prov{border-collapse:collapse;font-size:13px;background:var(--surface);border:1px solid var(--border)}
table.prov th,table.prov td{padding:4px 10px;border-bottom:1px solid var(--grid);text-align:left}
table.prov th{color:var(--muted)}
.toc{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:10px 22px;margin:16px 0}
.toc a{text-decoration:none;color:var(--ink2)}
.toc a:hover{color:var(--accent)}
.toc li{margin:.22em 0}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div>
    <h1>Bayesian Coverage — noise-aware active learning on CIFAR-100</h1>
    <p class="sub">Low-budget active learning with noisy annotators, on pretrained SimCLR features.
    Building on <b>ProbCover</b> (Yehuda et&nbsp;al.) and <b>low-budget AUM</b> (Netta's NPC+AUM stack).
    Test accuracy, mean±std over seeds {1,2,3}, budgets 100–1000, symmetric label noise {0, .2, .5, .8}.</p>
  </div>
  <button id="themeBtn" title="Toggle light/dark">◐ theme</button>
</header>

<div class="toc"><ol>
  <li><a href="#s1">The idea: make ProbCover noise-aware</a></li>
  <li><a href="#s2">From covered bits to per-point posteriors</a></li>
  <li><a href="#s3">Selection: spend the budget where uncertainty mass remains</a></li>
  <li><a href="#s4">Product vs additive — two combination rules</a></li>
  <li><a href="#s5">The classification plane: five ways to train on the posterior</a></li>
  <li><a href="#s6">Pseudo-labeling: train on <i>all</i> the points</a></li>
  <li><a href="#s7">Normalized features: the final +1</a></li>
  <li><a href="#s8">Which classifier head? linear vs 1-NN vs MLP (cold/warm, early stopping)</a></li>
  <li><a href="#s9">The ball radius δ: replay + paired training sweep — 0.75 confirmed</a></li>
  <li><a href="#s10">The loss: soft-CE ≡ forward KL; reverse KL tested</a></li>
  <li><a href="#s11">Results explorer — all arms, all noise levels (interactive)</a></li>
  <li><a href="#s12">Next steps</a></li>
  <li><a href="#s13">Provenance &amp; how to regenerate</a></li>
</ol></div>

<h2 id="s1">1 · The idea: make ProbCover noise-aware</h2>
<p>ProbCover selects the point whose δ-ball covers the most <i>uncovered</i> points — a binary
notion of coverage that implicitly trusts every annotation. Our starting question:
<b>what if the annotator is noisy?</b> A label from a 50%-accurate annotator should not
"cover" its ball the way a clean label does. So we feed the annotator's noise level —
currently the <i>real</i> noise rate via its confusion matrix; estimating it from the data
is future work (a §9.4 estimator bake-off already ran offline, DS-EM looks usable) —
directly into the algorithm. This one change forces the two upgrades below, and ends up
paying on <i>two</i> planes: point <b>selection</b> and classifier <b>training</b>.</p>

<h2 id="s2">2 · From covered bits to per-point posteriors</h2>
<p>Instead of a covered/uncovered bit, every pool point <em>u</em> carries a full distribution
over the <em>C</em>=100 classes, <em>P</em>(<em>y<sub>u</sub></em>&nbsp;=&nbsp;·), starting from a prior
π (uniform, 1/<em>C</em>). It is built in three steps:</p>

<p><b>Step A — annotator noise enters.</b> For a labeled point ℓ with observed label
<em>ỹ</em><sub>ℓ</sub> from annotator <em>a</em> with confusion matrix
<em>M<sup>a</sup></em>[<em>k</em>,<em>j</em>] = <em>P</em>(says <em>j</em> | true <em>k</em>),
the belief about ℓ's <i>own</i> true label is Bayes' rule through the confusion matrix:</p>
<div class="math"><em>b</em><sub>ℓ</sub>(<em>k</em>) <span class="op">∝</span> π(<em>k</em>) · <em>M<sup>a</sup></em>[<em>k</em>, <em>ỹ</em><sub>ℓ</sub>]
&nbsp;&nbsp;&nbsp;<span style="color:var(--muted)">— symmetric annotator with accuracy ρ:&nbsp;</span>
<em>b</em><sub>ℓ</sub>(<em>k</em>) = ρ <span style="color:var(--muted)">if</span> <em>k</em>=<em>ỹ</em><sub>ℓ</sub>,
<span style="color:var(--muted)">else</span> (1−ρ)/(<em>C</em>−1)</div>

<p><b>Step B — geometry enters.</b> The belief propagates to any point <em>u</em> through the
kernel <em>K</em>[<em>u</em>,ℓ] (tophat in all runs: 1 inside the δ-ball, 0 outside; δ=0.75),
diluted toward the prior with distance:</p>
<div class="math"><em>q</em><sub>ℓ</sub>(<em>u</em>,<em>k</em>) <span class="op">=</span>
<em>K</em>[<em>u</em>,ℓ] · <em>b</em><sub>ℓ</sub>(<em>k</em>)
<span class="op">+</span> (1 − <em>K</em>[<em>u</em>,ℓ]) · π(<em>k</em>)</div>
<p class="small">Read: with probability <em>K</em> "u and ℓ share a true label" (ProbCover's ball
assumption) so ℓ's belief transfers as-is; otherwise ℓ says nothing and the message is the prior.</p>

<p><b>Step C — fuse all neighbors (product of experts).</b> All labeled neighbors combine in
log space (naive Bayes over neighbors):</p>
<div class="math"><em>P</em>(<em>y<sub>u</sub></em>=<em>k</em>) <span class="op">∝</span>
π(<em>k</em>) · ∏<sub>ℓ∈L</sub> <em>q</em><sub>ℓ</sub>(<em>u</em>,<em>k</em>)
&nbsp;&nbsp;&nbsp;<span style="color:var(--muted)">i.e.</span>&nbsp;
<em>C</em>[<em>u</em>] = log&nbsp;π + Σ<sub>ℓ</sub> log <em>q</em><sub>ℓ</sub>(<em>u</em>,·),&nbsp;
<em>P</em> = softmax(<em>C</em>[<em>u</em>])</div>
<p>Agreeing neighbors <i>sharpen</i> the posterior; a single confident-but-wrong neighbor is
overridable. Noise-awareness falls out for free: a noisy annotator's message is nearly flat,
so its ball barely sharpens.</p>
<p class="small">Future work (like the noise-rate estimation): the code also supports raising each
message to a per-neighbor exponent β·<em>w<sub>u</sub></em><sub>ℓ</sub> (uniform tempering β and
kernel weighting <em>w</em>=<em>K</em>) to damp the overconfidence of correlated neighbors — unused
so far: all runs use β=1 and the tophat kernel, i.e. the exact plain product above; a β-sweep is
on deck (§9).</p>

<h2 id="s3">3 · Selection: spend the budget where uncertainty mass remains</h2>
<p>Coverage is now graded — a point is well-covered when its posterior is peaked. Each point's
remaining uncertainty is <b>1&nbsp;−&nbsp;max<sub>k</sub>&nbsp;<em>P</em>(<em>y<sub>u</sub></em>=<em>k</em>)</b>,
and a candidate is scored by the unresolved mass inside its ball:</p>
<div class="math">uncertainty(<em>u</em>) <span class="op">=</span> 1 − max<sub><em>k</em></sub> <em>P</em>(<em>y<sub>u</sub></em> = <em>k</em>)
&nbsp;&nbsp;&nbsp;&nbsp;
score(<em>v</em>) <span class="op">=</span> Σ<sub><em>u</em></sub> <em>K</em>[<em>v</em>,<em>u</em>] · uncertainty(<em>u</em>)
&nbsp;&nbsp;&nbsp;&nbsp;
pick&nbsp;argmax<sub><em>v</em></sub> score(<em>v</em>)</div>
<p>— the soft generalization of ProbCover's "ball covering the most uncovered points". Each pick
is annotated <i>in-loop</i> and folded into the posterior before the next pick. This is where
<b>depth-vs-breadth</b> becomes automatic: a clean label saturates its ball (retired — go
explore <i>breadth</i>), while a noisy label barely sharpens it, so the region stays attractive
and gets re-sampled (<i>depth</i>) — the algorithm decides per region, from the annotator's
accuracy, with no extra knob.</p>

<h2 id="s4">4 · Product vs additive — two combination rules</h2>
<p>Two ways to fuse the neighbors' messages were tried: the <b>product</b> rule multiplies experts
(§2 above), and an <b>additive</b> rule accumulates evidence as pseudo-counts instead. Both live
behind the same <code>COMBINE</code> switch, so the comparison is like-for-like
(same kernel, same in-loop annotation). (Itai's <code>BAYES_MISP</code> is also additive in
spirit — his actual method is ported and compared separately in the explorer's BAYES_MISP
group, and the head it introduced gets §8.)</p>
<div class="math"><b>product:</b>&nbsp;
<em>C</em>[<em>u</em>] = log&nbsp;π + Σ<sub>ℓ</sub> log <em>q</em><sub>ℓ</sub>(<em>u</em>,·),&nbsp;
<em>P</em> = softmax(<em>C</em>[<em>u</em>])</div>
<div class="math"><b>additive:</b>&nbsp;
<em>C</em>[<em>u</em>] init = π&nbsp;(pseudo-counts);&nbsp;
<em>C</em>[<em>u</em>, <em>ỹ</em><sub>ℓ</sub>] += <em>K</em>[<em>u</em>,ℓ];&nbsp;
<em>P</em>(<em>y<sub>u</sub></em>=·) = <em>C</em>[<em>u</em>]&nbsp;/&nbsp;Σ<sub><em>k</em></sub><em>C</em>[<em>u</em>,<em>k</em>]</div>
<p>The two need <i>different</i> acquisition ceilings. The product posterior saturates to ~1
after one clean in-ball label, so <code>1−max&nbsp;P</code> retires covered points. The additive
posterior only reaches <em>n</em>/(<em>n</em>+1) after <em>n</em> labels — scored against a ceiling
of 1 it never retires anything, the greedy clumps into dense regions, and low-budget accuracy
collapses (13.4 @100 — the bug we found and fixed on 2026-07-14). The fix borrows the
saturating ceiling from Itai's <code>BAYES_MISP</code> (his idea, <code>diff_method='max'</code>):</p>
<div class="math"><b>additive deficit:</b>&nbsp;
max( <em>f</em><sub>1</sub> − max<sub><em>k</em></sub> <em>P</em>(<em>y<sub>u</sub></em>=<em>k</em>), 0 ),&nbsp;&nbsp;
<em>f</em><sub>1</sub> = (max<sub><em>k</em></sub> π + 1)&nbsp;/&nbsp;(Σ<sub><em>k</em></sub> π + 1)
&nbsp;&nbsp;<span style="color:var(--muted)">— the dominant-class mass after exactly one tophat in-ball label
(= BAYES_MISP diff_method='max'); one clean label now retires a point.</span></div>
<h2 id="s5">5 · The classification plane: five ways to train on the posterior</h2>
<p>Selection alone ≈ ProbCover — the posterior's real payoff is that <b>the same posterior can
build the training targets</b>. Five arms (<code>TRAIN_LABEL_MODE</code>), all swept:</p>
<table class="prov" style="margin:10px 0">
<tr><th>Mode</th><th>Classifier trains on</th><th>Loss</th></tr>
<tr><td><code>observed</code></td><td>raw annotated (noisy) labels — the baseline</td><td>CE</td></tr>
<tr><td><code>map_posterior</code></td><td>each labeled point relabeled to argmax of its posterior (self-correction)</td><td>CE</td></tr>
<tr><td><code>soft_posterior</code></td><td>the full posterior vector as a soft label</td><td>soft CE</td></tr>
<tr><td><code>filtered</code></td><td>only points whose leave-one-out posterior agrees with their annotation
    (argmax<sub>k</sub> <em>P</em><sup>LOO</sup><sub>ℓ</sub> = <em>ỹ</em><sub>ℓ</sub> — neighbors judging you without your own vote)</td><td>CE, subset</td></tr>
<tr><td><code>reweighted</code></td><td>one-hot on the observed label scaled by
    <em>w</em><sub>ℓ</sub> = <em>P</em><sup>LOO</sup><sub>ℓ</sub>(<em>ỹ</em><sub>ℓ</sub>)</td><td>soft CE (row sum = weight)</td></tr>
</table>
<p><b>Result:</b> at noise ≥ 0.5 posterior training crushes raw labels —
<code>soft_posterior</code> 27.5 vs 12.6 observed at noise 0.8 @1000 (<b>2.2×</b>) — and beats
Netta's NPC+AUM (15.0) exactly where noise is heaviest: AUM's judge network was itself trained
on mostly-wrong labels, while the graph posterior needs no trained model. At noise 0.2 NPC+AUM
kept a small lead (43.3 vs 41.5) — until §6. On clean data every correction mode costs a little
(the ball-impurity price: ~32% of labeled points sit in balls whose neighbor-vote prefers
another class at δ=0.75). <code>filtered</code>/<code>reweighted</code> collapse at budget 100
(~1 label/class → the LOO judge has no neighbors) — kept for completeness, not recommended.</p>
<div class="note">Measured at the source (offline replay, no classifier): at noise 0.8 @1000 the
posterior-argmax relabel is <b>61.6% correct vs the annotator's 20.6%</b> — of 811 relabels, 462
fix a wrong label, only 52 break a correct one. At noise ≤ 0.2 relabeling is net-negative
(ball impurity) — the training-mode choice should be noise-gated… until pseudo-labeling removed
the need (§6). This beat NPC + clean-data training everywhere the noise was real.</div>
<div class="chart"><div class="chart-title">Training plane: observed vs map/soft posterior vs NPC+AUM (all bayes_product selection except NPC and probcover)</div>
<div class="chart-sub">The gap grows with noise: soft targets carry what the annotations know without committing to their mistakes.</div>
<div id="chartB" style="height:560px"></div></div>
<details><summary>Posterior quality vs ground truth (offline replay, R6–R9): denoising + pseudo-label supply</summary>
<div class="minitables">__MINITABLES__</div>
<p class="small">labeled rows = the labels <code>map_posterior</code> would train on (denoising);
pool rows = what pseudo-labeling can harvest. At noise ≥ 0.2 @1000 the top-10-per-class
pseudo-labels are <i>cleaner than the annotations we paid for</i> (79.3 vs 81.0 @.2; 69.4 vs 49.5 @.5;
42.8 vs 20.6 @.8) — §6 was justified before a single training job ran.</p>
</details>

<h2 id="s6">6 · Pseudo-labeling: train on <i>all</i> the points</h2>
<p>Then the question: why train only on the ~100–1000 <i>labeled</i> points? Every <b>covered</b>
pool point (≥1 labeled neighbor in its δ-ball) already carries an informative posterior — use it
as a free soft training target. <code>TRAIN_LABEL_MODE='pseudo'</code>: labeled points train on
their soft posterior, plus covered unlabeled points enter with their posterior as target
(guard: prior-only/uncovered points are never pseudo-labeled). Three scopes:
<code>all</code> = every covered point (~10–30k targets); <code>lin10</code>/<code>agg50</code> =
top-k most-confident per predicted class, k growing per round (linear / aggressive schedule).</p>
<p><b>This was the big jump.</b> <code>pseudo all</code> is the best arm at <b>every noise level
including clean</b> — 47.8 vs 45.1 (observed) @1000 clean, and it closes NPC+AUM's last lead at
noise 0.2 (45.8 vs 43.3). It even beats the training-free <b>probcover&nbsp;+&nbsp;1-NN</b>
evaluator by 1000 labels (47.8 vs 45.6) — though below ~400 labels 1-NN is still the best known
cell (38.1 @100 clean; if the budget stops at 300, skip training entirely). Volume is the
mechanism: ~10–30k soft targets dominate the data diet, and soft wrongness (spread over classes,
not committed to a hard wrong label) is cheap. At noise 0.8 the safer scope is
<code>agg50</code> (29.0 vs 28.1 for <code>all</code>) — per-ball caps (guard&nbsp;b) are still
unimplemented, and <code>all</code> amplifies noisy balls hardest exactly where labels are least
reliable. Out-of-ball targets (NW soft-kNN on the uncovered ~80%) were tested at several weights
and <b>rejected</b> — far targets are confidently wrong, covered-only stands.</p>
<div class="chart"><div class="chart-title">Pseudo-labeling: the jump from training on 100–1000 points to training on every covered point</div>
<div class="chart-sub">pseudo all (norm) is the recommended configuration (§7). probcover + 1-NN shown as the training-free reference; soft_posterior as the best labeled-only arm (§5).</div>
<div id="chartC" style="height:560px"></div></div>

<h2 id="s7">7 · Normalized features: the final +1</h2>
<p>All classifier-trained arms were re-run with L2-normalized classifier inputs — the
<b>scale-preserving</b> scheme <em>x</em> → <em>x</em>/‖<em>x</em>‖ · <span style="text-decoration:overline">‖x<sub>train</sub>‖</span>
(mean train norm 2.2761; plain unit-L2 collapses low-budget training 13.3→3.3 @100). The effect is
<b>convergence-gated</b>: at ≤300 labels the labeled-only arms lose (fixed-LR optimization
artifact), crossover ~350–400, and by 600–1000 every arm gains +2..+8 — e.g. probcover+training
+8.1 @600 at noise 0.2. <code>pseudo all</code> — which always trains on enough points to
converge — gains ~+1&nbsp;pt at <i>every</i> budget and noise, reaching <b>38.4 @100 clean</b>,
edging past even probcover+1-NN's 38.1. Hence the recommended configuration:
<b>bayes_product selection + pseudo all + normalized inputs</b> (the red curve everywhere in this
report). Toggle the "normalized" rows in the explorer below to see the paired raw-vs-norm effect
per arm. One honest accounting note: normalization helps <code>observed</code> the most at ≥400
labels, so part of what posterior-training fixed on raw features was feature-norm pathology, not
label noise — the mid-budget noise-robustness margins shrink (but don't vanish) under
normalization.</p>

<h2 id="s8">8 · Which classifier head? linear vs 1-NN vs MLP (cold/warm, early stopping)</h2>
<p>Porting Itai's <code>bayes_al_2025</code> revealed his numbers came from a different
<i>evaluator</i>, not a better selector: a one-hidden-layer <b>MLP head</b> (sklearn
MLPClassifier, hidden 256) instead of our linear probe. Reproducing it surfaced a protocol
lesson that matters more than the architecture: <b>each head has its own best protocol, and
mixing them fabricates effects.</b> The linear probe <i>wants warm-starting</i> across episodes
(<code>FINE_TUNE=True</code>, +2.2 @1000 — log-verified) — the MLP is <i>hurt</i> by it
(−4.3 @1000, plasticity loss) and wants a <b>cold start every episode</b>. And the MLP needs
<b>early stopping</b> (best-val epoch selection on the clean valSet): with the final iterate
instead, it looked like it "memorizes noise" and inverted under noise — that reading was
retracted once best-val selection recovered +5..+10 under noise. A protocol artifact, not
memorization (RESULTS §4.7).</p>
<p>Under matched-best protocols (sweep 31085867; explorer group "Classifier heads"), the story
by budget is the familiar low-budget pattern with a third player: <b>1-NN wins the low-budget regime</b>
(38.1 @100 on probcover, clean), <b>the MLP wins once converged</b> (46.8 vs linear 46.5 @1000
clean; 17.2 vs 13.6 @noise .8 — it leads the warm linear at <i>every</i> noise level on
observed labels), and the linear probe is never best on observed labels alone.</p>
<p><b>The stack test (§9.7) failed, and that is the interesting result:</b> MLP + the champion
<code>pseudo all</code> diet does <i>not</i> add up. A torch twin (<code>from_mlp_torch</code>)
was built so soft targets could reach the MLP (sklearn's <code>.fit</code> can't take soft rows);
after fixing a 200× too-strong weight-decay mapping (sklearn's α is divided by batch size —
parity gate now passes, 34.5→44.9 vs sklearn 34.9→46.8, sweep 31088628), MLP+pseudo-all trails
the warm-norm linear+pseudo-all at every noise and budget — clean −0.9..−1.4, noise .2 ≈ −3,
noise .5/.8 ≈ −5, the deficit <i>growing</i> with noise. Reading: the posterior's pseudo-targets
are built from the feature-space ball structure, so they already <i>linearize</i> the problem —
the MLP's extra capacity has nothing left to model and only adds variance under noise.
<b>Recommended config unchanged</b>: warm-norm linear + pseudo all (48.4 @1000 clean); the MLP
(cold + best-val) is the best <i>observed-label</i> head. Full per-head tables:
<code>docs/classifier_model_comparison.html</code>.</p>

<h2 id="s9">9 · The ball radius δ: replay + paired training sweep — 0.75 confirmed</h2>
<p>Everything above runs δ=0.75 — a value inherited from ProbCover's CIFAR-10 setting and never
re-examined, even though the repo's own purity criterion says ~0.50 for CIFAR-100 (balls are
only 39% pure at 0.75) and the two mechanisms that dominate our results pull in opposite
directions: ball <i>impurity</i> drives relabel damage, coverage <i>volume</i> feeds
<code>pseudo all</code>. §9.2 tested it in two stages. <b>Stage 1 (offline replay,</b>
<code>eval_delta_replay.py</code><b>)</b> rebuilt the recorded posteriors at
δ ∈ {0.5, 0.6, 0.75, 0.9, 1.1}: purity metrics are monotone in −δ (covered-point accuracy on
clean: 83.9% @0.5 → 66.7% @0.75 → 26.2% @1.1, calibration likewise), but <b>denoising and
noise-ID peak at 0.75</b> once noise ≥ .5 (tiny balls can't cross-vote — at δ=0.5 the posterior
just parrots the noisy label, 24.5% vs 61.6% relabel accuracy at noise .8) and coverage
collapses below it (5%/17%/50% of the pool @δ .5/.6/.75, B=1000). <b>Stage 2 (paired training
sweep, job 31108061)</b> re-ran the champion arm end-to-end at δ {0.6, 0.9} — where δ shapes
<i>selection</i> too: <b>δ=0.75 wins at every noise level and every budget</b> (@1000 clean:
48.4 vs 47.9/40.3; noise .8: 29.2 vs 24.4/24.8). δ=0.6's purity never overcomes its 3× coverage
loss; δ=0.9's extra covered mass is impurity-poisoned. Volume beats purity while coverage is
scarce, impurity wins once coverage is cheap — 0.75 sits at the crossing at every noise level.
The inherited default survives a proper examination; no retune for CIFAR-100/SimCLR (new
datasets should re-run the cheap replay first — the crossing is geometry-dependent). Toggle the
"δ ablation" group in the explorer to see both curves against the red champion.</p>

<h2 id="s10">10 · The loss: soft-CE ≡ forward KL; reverse KL tested</h2>
<p>The soft-target arms train with soft cross-entropy. One candidate "improvement" needs no
experiment: with <i>fixed</i> targets t, <b>forward KL(t‖model) = soft-CE − H(t)</b>, and the
entropy term is a constant of the optimization — identical gradients, identical optima,
identical best-val checkpoints. <b>Soft-CE <i>is</i> forward KL</b>; running it separately would
reproduce the same numbers. The only direction with new content is <b>reverse KL(model‖target)</b>
(<code>--soft_loss rkl</code>): mode-seeking, it lets the classifier commit to one class inside a
spread-out posterior instead of averaging over it — plausibly better inside impure balls at high
noise. Tested on both heads at every noise level (sweeps 31085854 + 31088628; explorer group
"Classifier heads"): <b>no win anywhere</b> — on the linear head it ties soft-CE under noise
(+0.2 at noise .8, within seed noise) and loses −0.7..−1.7 on clean (committing early wastes the
calibrated spread), on the MLP it is uniformly slightly worse. Soft-CE stands for all
soft-target arms. (The equivalence breaks only if targets stop being fixed — e.g. the §12
classifier-feedback loop, where the posterior would move during training.)</p>

<h2 id="s11">11 · Results explorer — all arms, all noise levels</h2>
<p class="hint">Click a row to add/remove its curve. Hover a row to spotlight its curve; hover the
chart for exact mean±std of every visible arm. Values are test accuracy (%), mean±std over seeds;
<b>bold = column best</b>. The Netta arms have no noise-0 runs (never swept).
Head protocols differ by design: linear-head rows warm-start across episodes
(<code>FINE_TUNE=True</code>), MLP-head rows are cold with best-val epoch selection — see
RESULTS §4.7 / DELTA_ROOTCAUSE pitfall 3 before reading across heads.</p>
<div class="controls">
  <span class="seg" id="noiseSeg"></span>
  <label class="ctl"><input type="checkbox" id="bandsCk"> ±std bands</label>
  <label class="ctl"><input type="checkbox" id="fullXCk"> full x-range (AUM sweeps reach 3000)</label>
  <button class="btn" id="resetBtn">reset selection</button>
  <button class="btn" id="clearBtn">clear all</button>
</div>
<div class="explorer">
  <div class="tblwrap"><table class="res" id="resTable"></table>
    <p class="hint">Row order and sweep pinning mirror the generated R2–R5 tables in
    <code>docs/bayesian_coverage_RESULTS.md</code>. The last four groups (classifier heads,
    δ ablation, normalized-input reruns, training-free evaluators) are extras beyond
    R2–R5 — off by default.</p></div>
  <div class="chartwrap"><div class="chart" style="margin:0">
    <div class="chart-title" id="expTitle"></div>
    <div class="chart-sub">test accuracy (%) vs annotation budget</div>
    <div id="expChart" style="height:620px"></div></div></div>
</div>

<h2 id="s12">12 · Next steps</h2>
<p class="small">Three former entries are now <i>answered</i> and moved into the tale above:
the MLP head + pseudo-stack (§8 — does not stack, config unchanged), the δ calibration
(§9 — 0.75 confirmed end-to-end), and the KL loss (§10 — soft-CE ≡ forward KL; reverse KL
never wins). What remains:</p>
<ol class="next">
  <li><b>Different kernels for covering vs training.</b> The tophat gate is right for
      <i>selection</i>, but the training targets could use a smooth in-ball kernel: RBF-weighted
      product exponents already replay to 87→96.7 labeled-set denoising @100 clean —
      not yet in live code.</li>
  <li><b>The AUM idea, chicken-and-egg.</b> Feed the trained classifier's estimates back into
      the posterior (posterior → classifier → updated posterior → next selection round) — AUM's
      loop, but with the graph posterior as the anchor instead of a noisily-trained judge.</li>
  <li><b>Estimate the annotator noise level instead of using the real one.</b> Every run above
      feeds the annotator's <i>true</i> noise rate into Step A's confusion matrix — an oracle
      assumption a real deployment doesn't get. Replace it with an estimate ρ̂ from the labeled
      data itself. An offline bake-off (§9.4) already ran: <b>DS-EM</b> is the best
      assumption-free estimator (separates noise 0 vs 0.2 at every budget, with a small
      +~0.25 bias at low noise), <b>impurity-corrected LOO</b> is the most accurate
      (MAE ≈ 0.016 @1000) but needs a clean anchor, raw LOO saturates at the ball-impurity
      floor. Next: wire ρ̂ into the live pipeline and measure how much of the oracle-ρ
      accuracy survives.</li>
  <li><b>Compare to Itai's formalism.</b> The BAYES_MISP port shows the two methods land in the
      same place empirically (linear head: misp ≈ probcover ≈ product at every noise level) —
      the open work is the <i>theoretical</i> side-by-side: map his pseudo-count accumulation +
      diff-max acquisition onto our posterior/uncertainty language, state exactly when the two
      selection rules pick different points, and see whether either formalism predicts the
      high-budget product-vs-additive gap. (A first written pass exists:
      <code>docs/probabilistic_kernel_AL_section2_comparison.md</code>.)</li>
  <li class="small"><b>Also on deck:</b> β-temper sweep for correlated-neighbor
      overconfidence; per-ball caps (guard b) to make <code>pseudo all</code> safe at noise 0.8.</li>
</ol>

<h2 id="s13">13 · Provenance &amp; how to regenerate</h2>
<table class="prov">
<tr><th>Sweep</th><th>SLURM job</th><th>Grid</th><th>Status</th></tr>
<tr><td>Bayes comparison</td><td>30978366</td><td>{random, probcover, bayes_additive, bayes_product} × noise × 3 seeds</td><td>48/48 ✔</td></tr>
<tr><td>Bayes additive re-run (fixed acquisition)</td><td>31066809</td><td>bayes_additive × noise × 3 seeds</td><td>12/12 ✔</td></tr>
<tr><td>Bayes train-mode</td><td>30980646</td><td>bayes_product × 5 modes × noise × 3 seeds</td><td>60/60 ✔</td></tr>
<tr><td>§7.5 pseudo-labeling</td><td>31047129</td><td>bayes_product + pseudo × {all, lin10, agg50} × noise × 3 seeds</td><td>36/36 ✔</td></tr>
<tr><td>Normalized-inputs rerun</td><td>31048960</td><td>all trained arms × noise × 3 seeds, paired</td><td>144/144 ✔</td></tr>
<tr><td>ProbCover + 1-NN eval</td><td>31048649</td><td>probcover × NN1 evaluator × noise × 3 seeds</td><td>12/12 ✔</td></tr>
<tr><td>Netta NPC repro / AUM ablation</td><td>30811243 / 30863746</td><td>{npc, probcover, random} × clean-train × noise ≥ .2 × 3 seeds</td><td>27/27, 54/54 ✔</td></tr>
<tr><td>Posterior-as-classifier / training-free study</td><td colspan="2">offline replays: <code>test_posterior_acc.py</code>, <code>eval_1nn_all_arms.py</code></td><td>✔</td></tr>
<tr><td>BAYES_MISP port (parity + noise sweep + best-val MLP)</td><td>31078514 / 31084429 / 31085802</td><td>bayes_misp × {linear, MLP} heads × noise × 3 seeds</td><td>3/3, 24/24, 12/12 ✔</td></tr>
<tr><td>Classifier-head comparison (§8)</td><td>31085867 / 31085854 / 31088628</td><td>{random, probcover, pseudo-all} × {MLP sklearn, MLP torch, rKL} × noise × 3 seeds</td><td>24/24, 24/24, 27/27 ✔</td></tr>
<tr><td>δ ablation stage 2 (§9)</td><td>31108061</td><td>champion arm × δ {0.6, 0.9} × noise × 3 seeds (stage 1 offline: <code>eval_delta_replay.py</code>)</td><td>24/24 ✔</td></tr>
</table>
<p class="small" style="margin-top:12px">
Single source of truth: <code>results/cifar100/results_master.csv</code> (one row per per-seed cell,
built by <code>batch_utils/build_results_master.py</code>). This file was generated by
<code>python3 batch_utils/build_summary_report.py</code> — rerun after refreshing the master CSV;
the R2–R5 markdown/HTML tables come from <code>batch_utils/render_results_tables.py</code> against
the same CSV. Theory: <code>docs/bayesian_coverage_THEORY.md</code> · results prose:
<code>docs/bayesian_coverage_RESULTS.md</code> · open work: <code>docs/bayesian_coverage_NEXT_STEPS.md</code>.
Regime: CIFAR-100, SimCLR ResNet-18 features (512-d), linear probe, tophat δ=0.75, budget 100/episode,
symmetric annotator noise, seeds {1,2,3}. Generated 2026-07-22. Chronological experiment log
(what ran when, sweep by sweep, with per-experiment graphs and conclusions):
<code>docs/bayes_experiments_log.html</code>.</p>

</div>

<script>__PLOTLY__</script>
<script>
"use strict";
const DATA = __DATA__;
const NOISES = ["0.0","0.2","0.5","0.8"];
const NOISE_NAME = {"0.0":"0 (clean)","0.2":"0.2","0.5":"0.5","0.8":"0.8"};
const TBL_BUDGETS = [100,300,500,700,1000];
const byId = {}; DATA.arms.forEach(a => byId[a.id] = a);
const DEFAULT_ON = DATA.arms.filter(a => a.on).map(a => a.id);

const state = { noise:"0.2", on:new Set(DEFAULT_ON), bands:false, fullX:false };

/* ---------- theme ---------- */
function mode(){
  const t = document.documentElement.getAttribute("data-theme");
  if (t) return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
const CHROME = {
  light:{ink:"#0b0b0b", ink2:"#52514e", muted:"#898781", grid:"#e1e0d9",
         baseline:"#c3c2b7", surface:"#fcfcfb", hoverbg:"#fcfcfb"},
  dark:{ink:"#ffffff", ink2:"#c3c2b7", muted:"#898781", grid:"#2c2c2a",
        baseline:"#383835", surface:"#1a1a19", hoverbg:"#222221"}
};
function armColor(a){ return a.color[mode()]; }
function hexA(hex, alpha){
  const v = [1,3,5].map(i => parseInt(hex.slice(i,i+2),16));
  return `rgba(${v[0]},${v[1]},${v[2]},${alpha})`;
}

document.getElementById("themeBtn").addEventListener("click", () => {
  const cur = mode();
  document.documentElement.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
  renderAll();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => renderAll());

/* ---------- shared plotly scaffolding ---------- */
function baseLayout(){
  const c = CHROME[mode()];
  return {
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
    font:{family:'system-ui,-apple-system,"Segoe UI",sans-serif', size:12.5, color:c.ink2},
    hovermode:"x unified",
    hoverlabel:{bgcolor:c.hoverbg, bordercolor:c.grid, font:{color:c.ink, size:12}},
    margin:{l:46,r:14,t:26,b:42},
    dragmode:false
  };
}
function ax(extra){
  const c = CHROME[mode()];
  return Object.assign({gridcolor:c.grid, gridwidth:1, zeroline:false,
                        linecolor:c.baseline, tickcolor:c.baseline,
                        tickfont:{color:c.muted}}, extra || {});
}
const CONFIG = {responsive:true, displaylogo:false, displayModeBar:false, scrollZoom:false};

function lineTrace(a, nz, opts){
  const s = a.series[nz]; if (!s) return null;
  const col = armColor(a);
  return Object.assign({
    type:"scatter", x:s.b, y:s.m, name:a.short,
    mode:a.markers ? "lines+markers" : "lines",
    line:{color:col, width:a.width, dash:a.dash, shape:"linear"},
    marker:{size:5.5, color:col},
    customdata:s.s,
    hovertemplate:"%{y:.1f}±%{customdata:.1f}"
  }, opts || {});
}
function bandTraces(a, nz, opts){
  const s = a.series[nz]; if (!s) return [];
  const col = hexA(armColor(a), 0.13);
  const hi = s.m.map((m,i) => m + s.s[i]), lo = s.m.map((m,i) => m - s.s[i]);
  const common = Object.assign({type:"scatter", mode:"lines", line:{width:0},
    hoverinfo:"skip", showlegend:false, name:a.short + " ±σ"}, opts || {});
  return [Object.assign({}, common, {x:s.b, y:hi}),
          Object.assign({}, common, {x:s.b, y:lo, fill:"tonexty", fillcolor:col})];
}

/* ---------- chapter small-multiples (2×2 by noise) ---------- */
function smallMultiples(divId, armIds, maxBudget){
  const c = CHROME[mode()];
  const traces = [], layout = baseLayout();
  layout.grid = {rows:2, columns:2, pattern:"independent", xgap:0.14, ygap:0.22};
  layout.legend = {orientation:"h", y:1.14, x:0, font:{size:12, color:c.ink2}};
  layout.margin = {l:46,r:14,t:64,b:40};
  layout.annotations = [];
  NOISES.forEach((nz,p) => {
    const suf = p === 0 ? "" : String(p+1);
    armIds.forEach((id,k) => {
      const t = lineTrace(byId[id], nz,
        {xaxis:"x"+suf, yaxis:"y"+suf, legendgroup:id, showlegend:p===NOISES.findIndex(n=>byId[id].series[n])});
      if (t){ if (maxBudget){ const cut = t.x.filter(b=>b<=maxBudget).length;
                t.x = t.x.slice(0,cut); t.y = t.y.slice(0,cut); t.customdata = t.customdata.slice(0,cut); }
              traces.push(t); }
    });
    layout["xaxis"+suf] = ax({range:[50,1050], tickvals:[100,300,500,700,1000]});
    layout["yaxis"+suf] = ax({rangemode:"tozero"});
    layout.annotations.push({text:"noise " + NOISE_NAME[nz], showarrow:false,
      xref:"x"+suf+" domain", yref:"y"+suf+" domain", x:0.02, y:1.06,
      font:{size:12.5, color:c.ink, weight:600}, xanchor:"left"});
  });
  Plotly.react(divId, traces, layout, CONFIG);
}

/* ---------- explorer ---------- */
function fmt(m,s){ return m.toFixed(1) + "±" + s.toFixed(1); }

function tableValue(a, nz, b){
  const s = a.series[nz]; if (!s) return null;
  const i = s.b.indexOf(b); if (i < 0) return null;
  return [s.m[i], s.s[i]];
}

function swatchSVG(a){
  const col = armColor(a);
  const dashmap = {solid:"", dot:"2,3", dash:"6,4", dashdot:"6,4,2,4",
                   longdash:"10,4", longdashdot:"10,4,2,4"};
  const dot = a.markers ? '<circle cx="17" cy="6" r="3" fill="'+col+'"/>' : "";
  return '<svg class="sw" width="34" height="12" aria-hidden="true">' +
    '<line x1="1" y1="6" x2="33" y2="6" stroke="'+col+'" stroke-width="'+Math.min(a.width,3)+'"' +
    (dashmap[a.dash] ? ' stroke-dasharray="'+dashmap[a.dash]+'"' : "") + '/>' + dot + "</svg>";
}

function renderTable(){
  const tbl = document.getElementById("resTable");
  tbl.textContent = "";
  const thead = tbl.createTHead(); const hr = thead.insertRow();
  ["", "Sampling", "Classifier", "Train labels"].forEach(h => {
    const th = document.createElement("th"); th.className="k"; th.textContent=h; hr.appendChild(th); });
  TBL_BUDGETS.forEach(b => {
    const th = document.createElement("th"); th.textContent=String(b); hr.appendChild(th); });

  // column max per budget at DISPLAY precision (ties all get bold)
  const best = {};
  TBL_BUDGETS.forEach(b => {
    let bm = -1;
    DATA.arms.forEach(a => { const v = tableValue(a, state.noise, b);
      if (v){ const r = Math.round(v[0]*10); if (r > bm) bm = r; } });
    best[b] = bm;
  });

  const tb = tbl.createTBody();
  DATA.groups.forEach(([gid, glabel]) => {
    const sep = tb.insertRow(); sep.className = "sep";
    const td = sep.insertCell(); td.colSpan = 4 + TBL_BUDGETS.length; td.textContent = glabel;
    DATA.arms.filter(a => a.group === gid).forEach(a => {
      const on = state.on.has(a.id);
      const tr = tb.insertRow();
      tr.className = "armrow " + (on ? "on" : "off");
      tr.style.setProperty("--rowcol", armColor(a));
      tr.title = (on ? "click to hide curve" : "click to show curve") + " · sweep " + a.sweep;
      const c0 = tr.insertCell(); c0.className = "k k1"; c0.innerHTML = on ? swatchSVG(a) : "";
      const names = [a.d_sampling, a.d_model, a.d_labels];
      names.forEach(n => { const c = tr.insertCell(); c.className = "k"; c.textContent = n; });
      TBL_BUDGETS.forEach(b => {
        const c = tr.insertCell();
        const v = tableValue(a, state.noise, b);
        if (!v){ c.textContent = "—"; return; }
        if (Math.round(v[0]*10) === best[b]){ const bb = document.createElement("b"); bb.textContent = fmt(v[0], v[1]); c.appendChild(bb); }
        else c.textContent = fmt(v[0], v[1]);
      });
      tr.addEventListener("click", () => {
        state.on.has(a.id) ? state.on.delete(a.id) : state.on.add(a.id);
        renderTable(); renderChart();
      });
      tr.addEventListener("mouseenter", () => highlight(a.id));
      tr.addEventListener("mouseleave", () => highlight(null));
    });
  });
}

let traceOwner = [];  // arm id per trace index in the explorer chart
function renderChart(){
  const c = CHROME[mode()];
  const traces = []; traceOwner = [];
  const visible = DATA.arms.filter(a => state.on.has(a.id) && a.series[state.noise]);
  if (state.bands) visible.forEach(a => {
    bandTraces(a, state.noise).forEach(t => { traces.push(t); traceOwner.push(a.id); }); });
  visible.forEach(a => { traces.push(lineTrace(a, state.noise)); traceOwner.push(a.id); });

  const layout = baseLayout();
  layout.legend = {orientation:"h", y:-0.14, x:0, font:{size:11.5, color:c.ink2}};
  layout.xaxis = ax(state.fullX ? {rangemode:"tozero", title:{text:"annotation budget (labels)", font:{size:12}}}
    : {range:[50,1050], tickvals:TBL_BUDGETS, title:{text:"annotation budget (labels)", font:{size:12}}});
  layout.yaxis = ax({rangemode:"tozero", title:{text:"test accuracy (%)", font:{size:12}}});
  document.getElementById("expTitle").textContent =
    "Noise " + NOISE_NAME[state.noise] + " — " + visible.length + " arm" + (visible.length===1?"":"s");
  Plotly.react("expChart", traces, layout, CONFIG);
}

function highlight(armId){
  const div = document.getElementById("expChart");
  if (!div || !div.data || !div.data.length) return;
  const op = traceOwner.map(id => armId === null ? 1 : (id === armId ? 1 : 0.18));
  Plotly.restyle(div, {opacity: op});
}

/* ---------- controls ---------- */
const seg = document.getElementById("noiseSeg");
NOISES.forEach(nz => {
  const b = document.createElement("button");
  b.textContent = NOISE_NAME[nz];
  b.addEventListener("click", () => { state.noise = nz;
    seg.querySelectorAll("button").forEach(x => x.classList.toggle("sel", x === b));
    renderTable(); renderChart(); });
  if (nz === state.noise) b.classList.add("sel");
  seg.appendChild(b);
});
document.getElementById("bandsCk").addEventListener("change", e => { state.bands = e.target.checked; renderChart(); });
document.getElementById("fullXCk").addEventListener("change", e => { state.fullX = e.target.checked; renderChart(); });
document.getElementById("resetBtn").addEventListener("click", () => {
  state.on = new Set(DEFAULT_ON); renderTable(); renderChart(); });
document.getElementById("clearBtn").addEventListener("click", () => {
  state.on = new Set(); renderTable(); renderChart(); });

/* ---------- render everything ---------- */
function renderAll(){
  smallMultiples("chartB", ["pc_lin","prod_lin","map","soft","npc_aum"], 1000);
  smallMultiples("chartC", ["pc_lin","pc_1nn","prod_lin","soft","ps_all","ps_all_n"], 1000);
  renderTable(); renderChart();
}
renderAll();
</script>
</body>
</html>
"""

# R6-R9 posterior-quality mini tables (from docs/bayesian_coverage_RESULTS.md §4.5 —
# offline-replay numbers, not in results_master.csv)
MINI = {
    "0.0": [("labeled: observed acc", "100.0", "100.0", "100.0", "100.0", "100.0"),
            ("labeled: posterior argmax", "87.0", "75.7", "71.9", "70.7", "68.3"),
            ("pool: covered (%)", "25.6", "36.4", "41.6", "45.6", "49.5"),
            ("pool: acc on covered", "71.7", "69.0", "68.1", "67.4", "66.7"),
            ("pool: acc top-10/class", "76.4", "70.1", "68.4", "68.1", "67.9")],
    "0.2": [("labeled: observed acc", "81.0", "78.8", "79.9", "80.2", "81.0"),
            ("labeled: posterior argmax", "71.7", "70.8", "68.5", "67.1", "66.2"),
            ("pool: covered (%)", "24.4", "35.1", "40.1", "43.9", "47.8"),
            ("pool: acc on covered", "58.5", "61.6", "62.8", "62.2", "62.6"),
            ("pool: acc top-10/class", "64.2", "69.1", "71.8", "74.9", "79.3")],
    "0.5": [("labeled: observed acc", "48.3", "49.9", "49.5", "49.1", "49.5"),
            ("labeled: posterior argmax", "65.0", "70.2", "66.7", "68.5", "67.7"),
            ("pool: covered (%)", "19.9", "30.7", "36.1", "39.9", "44.0"),
            ("pool: acc on covered", "43.9", "52.8", "54.1", "54.6", "55.0"),
            ("pool: acc top-10/class", "35.2", "52.2", "59.2", "65.7", "69.4")],
    "0.8": [("labeled: observed acc", "19.7", "20.1", "20.4", "20.6", "20.6"),
            ("labeled: posterior argmax", "39.0", "51.9", "58.0", "59.7", "61.6"),
            ("pool: covered (%)", "12.6", "22.9", "28.1", "32.1", "36.0"),
            ("pool: acc on covered", "20.4", "30.1", "33.3", "35.3", "37.5"),
            ("pool: acc top-10/class", "11.1", "23.0", "29.4", "35.5", "42.8")],
}


def minitables_html():
    parts = []
    for nz in NOISES:
        rows = "".join("<tr><td>{}</td>{}</tr>".format(r[0], "".join(f"<td>{v}</td>" for v in r[1:]))
                       for r in MINI[nz])
        parts.append(
            f"<table><caption>noise {nz}</caption>"
            "<tr><th>measure</th><th>100</th><th>300</th><th>500</th><th>700</th><th>1000</th></tr>"
            f"{rows}</table>")
    return "".join(parts)


def main():
    plotly_js = None
    for p in PLOTLY_CANDIDATES:
        if os.path.exists(p):
            plotly_js = open(p, encoding="utf-8").read()
            break
    if plotly_js is None:
        raise SystemExit(f"plotly.min.js not found in any of: {PLOTLY_CANDIDATES}")

    arms = load_series()
    payload = {"groups": GROUPS, "arms": arms}
    html = (TEMPLATE
            .replace("__MINITABLES__", minitables_html())
            .replace("__PLOTLY__", plotly_js)
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False)))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(OUT, REPO)}  ({os.path.getsize(OUT)/1e6:.1f} MB, "
          f"{len(arms)} arms)")


if __name__ == "__main__":
    main()
