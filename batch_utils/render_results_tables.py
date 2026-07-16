#!/usr/bin/env python3
"""Render the §4.1 results-by-noise tables (R2-R5) from results_master.csv.

The CSV (built by build_results_master.py) is the single source of truth; this
script is the *pivot*: it aggregates per-seed rows to mean±std and writes the wide
per-noise tables into the docs between marker comments, e.g.

    <!-- BEGIN:R2 -->   ... generated table ...   <!-- END:R2 -->

Only the block between the markers is rewritten; all hand-written prose is left
untouched. Re-run after build_results_master.py to refresh every table at once.

Each display row is pinned to a specific sweep_id (ROW_SPEC) so provenance is
unambiguous — e.g. bayes_additive is drawn from the fixed rerun 31066809, never the
superseded 30978366 rows. Adding a new arm (e.g. `training + soft-kNN`) = collect it
into the CSV, then add one ROW_SPEC line.

Usage:  python3 batch_utils/render_results_tables.py            # writes into the .md + .html
        python3 batch_utils/render_results_tables.py --check    # print tables, do not write
"""
import argparse
import csv
import os
import statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "cifar100", "results_master.csv")
MD = os.path.join(REPO, "docs", "bayesian_coverage_RESULTS.md")
HTML = os.path.join(REPO, "results", "cifar100", "results_tables.html")

BUDGETS = [100, 300, 500, 700, 1000]
NOISES = [0.0, 0.2, 0.5, 0.8]
NOISE_LABEL = {0.0: "R2", 0.2: "R3", 0.5: "R4", 0.8: "R5"}

# --- ROW SPEC: the recipe for the table. One entry per display row, in table order.
# match keys: (sampling, model, train_labels, sweep_id) -> selects CSV rows.
# display   : what the three key cells print (defaults to the match value).
SEP = "SEP"  # italic separator marker


def row(sampling, model, labels, sweep, disp_sampling=None, disp_model=None, disp_labels=None):
    return dict(sampling=sampling, model=model, labels=labels, sweep=sweep,
                d_sampling=disp_sampling or sampling, d_model=disp_model or model,
                d_labels=disp_labels or labels)


ROW_SPEC = [
    (SEP, "Baselines"),
    row("random", "linear head", "observed", "30978366"),
    row("probcover", "linear head", "observed", "30978366"),
    row("probcover", "1-NN", "observed", "31048649"),
    (SEP, "Bayesian coverage — selection only (train on observed labels)"),
    row("bayes_additive", "linear head", "observed", "31066809"),
    row("bayes_product", "linear head", "observed", "30978366"),
    (SEP, "Bayesian coverage + label-correction training"),
    row("bayes_product", "linear head", "map_posterior", "30980646"),
    row("bayes_product", "linear head", "soft_posterior", "30980646"),
    row("bayes_product", "linear head", "filtered", "30980646"),
    row("bayes_product", "linear head", "reweighted", "30980646"),
    (SEP, "Bayesian coverage + pseudo-labeling (§4.6)"),
    row("bayes_product", "linear head", "pseudo (lin10)", "31047129"),
    row("bayes_product", "linear head", "pseudo (agg50)", "31047129"),
    row("bayes_product", "linear head", "pseudo (all)", "31047129"),
    row("bayes_product", "linear head", "pseudo (all)", "31048960", disp_model="linear head (norm)"),
    (SEP, "Posterior as classifier"),
    row("bayes_product", "posterior-argmax", "—", "test_posterior_acc"),
    (SEP, "Netta NPC / AUM"),
    row("probcover", "linear head", "AUM clean-train", "30863746"),
    row("npc", "linear head", "observed (sel. only)", "30863746"),
    row("npc", "linear head", "AUM clean-train", "30863746"),
    (SEP, "BAYES_MISP port (Itai's bayes_al_2025 — diff max, α=0.5, tophat δ=0.75; §4.7)"),
    row("bayes_misp", "linear head", "observed", "31084429", disp_sampling="bayes_misp (Itai)"),
    row("bayes_misp", "MLP head", "observed", "31084429", disp_sampling="bayes_misp (Itai)"),
    row("probcover", "MLP head", "observed", "31078514", disp_model="MLP head (seed 1)"),
    row("bayes_additive", "MLP head", "observed", "31078514", disp_model="MLP head (seed 1)"),
]

N_KEYCOLS = 3
HEADER = ["Sampling", "Classifier model", "Train labels"] + [str(b) for b in BUDGETS]


def load():
    """acc[(sampling,model,labels,sweep,noise,budget)] = list of per-seed test_acc."""
    acc = defaultdict(list)
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            key = (r["sampling"], r["model"], r["train_labels"], r["sweep_id"],
                   float(r["noise"]), int(r["budget"]))
            acc[key].append(float(r["test_acc"]))
    return acc


def cell(vals):
    if not vals:
        return None
    if len(vals) == 1:
        return f"{vals[0]:.1f}±0.0"
    return f"{st.mean(vals):.1f}±{st.stdev(vals):.1f}"


def mean_of(text):
    return None if text is None else float(text.split("±")[0])


def build_table(acc, noise):
    """Return list of markdown row-strings for one noise level (no header)."""
    grid = []  # (kind, cells) where kind in {'sep','data'}
    for spec in ROW_SPEC:
        if isinstance(spec, tuple):  # (SEP, "label")
            grid.append(("sep", spec[1]))
            continue
        cells = [cell(acc.get((spec["sampling"], spec["model"], spec["labels"], spec["sweep"], noise, b)))
                 for b in BUDGETS]
        grid.append(("data", spec, cells))

    # bold the column-max per budget across data rows — separately for the MLP-head
    # rows and everything else: the sklearn-MLP evaluator sits ~10 pts above the
    # linear protocol, so a single global bold would land on the MLP rows in every
    # column and stop marking the best arm within the established linear tables.
    for j in range(len(BUDGETS)):
        for head_is_mlp in (False, True):
            col = [(i, mean_of(g[2][j])) for i, g in enumerate(grid)
                   if g[0] == "data" and g[2][j] is not None
                   and ("MLP" in g[1]["model"]) == head_is_mlp]
            if not col:
                continue
            best_i = max(col, key=lambda t: t[1])[0]
            txt = grid[best_i][2][j]
            grid[best_i][2][j] = f"**{txt}**"

    lines = []
    for g in grid:
        if g[0] == "sep":
            lines.append("| *" + g[1] + "* |" + " |" * (len(HEADER) - 1))
        else:
            spec, cells = g[1], g[2]
            key = [spec["d_sampling"], spec["d_model"], spec["d_labels"]]
            vals = [c if c is not None else "—" for c in cells]
            lines.append("| " + " | ".join(key + vals) + " |")
    return lines


def md_block(acc, noise):
    note = "<!-- generated by batch_utils/render_results_tables.py from results_master.csv — do not edit by hand -->"
    hdr = "| " + " | ".join(HEADER) + " |"
    sep = "|" + "|".join(["---"] * len(HEADER)) + "|"
    return "\n".join([note, hdr, sep] + build_table(acc, noise))


def inject(text, tag, block):
    b, e = f"<!-- BEGIN:{tag} -->", f"<!-- END:{tag} -->"
    if b not in text or e not in text:
        raise SystemExit(f"markers {b}/{e} not found in doc — add them around the {tag} table first")
    pre = text[: text.index(b) + len(b)]
    post = text[text.index(e):]
    return pre + "\n" + block + "\n" + post


def render_html(acc):
    parts = ["<meta charset='utf-8'><style>",
             "body{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}",
             "table{border-collapse:collapse;margin:1rem 0;font-size:14px}",
             "th,td{border:1px solid #ccc;padding:3px 8px;text-align:right}",
             "td:nth-child(-n+3),th:nth-child(-n+3){text-align:left}",
             ".sep td{background:#f0f0f0;font-style:italic;font-weight:bold}",
             "</style>",
             "<h1>CIFAR100 — test accuracy by method (generated from results_master.csv)</h1>",
             "<p>Do not edit by hand — regenerate with <code>render_results_tables.py</code>.</p>"]
    for noise in NOISES:
        parts.append(f"<h2>{NOISE_LABEL[noise]} — noise {noise}</h2>")
        parts.append("<table><tr>" + "".join(f"<th>{h}</th>" for h in HEADER) + "</tr>")
        for g in build_table_struct(acc, noise):
            if g[0] == "sep":
                parts.append(f"<tr class='sep'><td colspan='{len(HEADER)}'>{g[1]}</td></tr>")
            else:
                key, vals = g[1], g[2]
                cells = "".join(f"<td>{c}</td>" for c in key + vals)
                parts.append("<tr>" + cells + "</tr>")
        parts.append("</table>")
    return "\n".join(parts)


def build_table_struct(acc, noise):
    """Same as build_table but returns structured rows for HTML (bold via <b>)."""
    out = []
    md = build_table(acc, noise)
    for line in md:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0].startswith("*") and cells[0].endswith("*"):
            out.append(("sep", cells[0].strip("*").strip()))
        else:
            fmt = [c.replace("**", "").strip() if "**" not in c else "<b>" + c.replace("**", "") + "</b>"
                   for c in cells]
            out.append(("data", fmt[:N_KEYCOLS], fmt[N_KEYCOLS:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print tables, do not write files")
    args = ap.parse_args()
    acc = load()

    if args.check:
        for noise in NOISES:
            print(f"\n### {NOISE_LABEL[noise]} — noise {noise}\n")
            print(md_block(acc, noise))
        return

    text = open(MD).read()
    for noise in NOISES:
        text = inject(text, NOISE_LABEL[noise], md_block(acc, noise))
    open(MD, "w").write(text)
    open(HTML, "w").write(render_html(acc))
    print(f"updated {os.path.relpath(MD, REPO)} (R2-R5) and {os.path.relpath(HTML, REPO)}")


if __name__ == "__main__":
    main()
