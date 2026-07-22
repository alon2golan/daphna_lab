# Aggregates the per-seed coarse-budget CSVs into ONE mean +/- std curve per variant, so runs that
# differ ONLY by seed collapse into a single line with a shaded band instead of N doubled lines.
# Groups by the variant TAG parsed from each filename (the seed is dropped), matches seeds on
# num_labeled, and writes a tidy mean/std CSV plus a PNG with error bands. Frozen is aggregated the
# same way from whichever files carry it. Read-only over outputs/finetune/; safe to re-run.

import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")                                          # file-only backend, no display
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "finetune"

# coarse_budget_sweep_<selector>_seed<seed>[_<tag>].csv -- pull out seed and the (optional) tag.
NAME_RE = re.compile(r"coarse_budget_sweep_(?P<selector>\w+?)_seed(?P<seed>\d+)(?:_(?P<tag>.+))?\.csv")


def load_csv(path):
    """Reads one sweep CSV into {num_labeled: (frozen_acc, finetuned_acc)} (percent). frozen_acc is
    NaN in --skip_frozen files; we keep it so frozen aggregates only over the files that have it."""
    rows = {}
    with open(path) as handle:
        header = handle.readline().strip().split(",")          # column order per CSV_HEADER
        idx = {name: i for i, name in enumerate(header)}
        for line in handle:
            cells = line.strip().split(",")
            if len(cells) < len(header):
                continue
            n = int(cells[idx["num_labeled"]])
            frozen = float(cells[idx["frozen_acc"]])
            tuned = float(cells[idx["finetuned_acc"]])
            rows[n] = (frozen, tuned)
    return rows


def collect(selector):
    """Groups every matching CSV so each curve has exactly ONE value per SEED per budget (so the std
    is genuine seed-to-seed spread, not file-to-file). Returns {tag: {num_labeled: [one val per seed]}}
    for the finetuned curves, and the same for frozen. Frozen is computed in several files at the same
    seed (each identical), so we dedup it PER SEED first -- otherwise a seed with many frozen files
    would dominate the mean and inflate n_seeds."""
    tuned = defaultdict(lambda: defaultdict(dict))             # tag -> budget -> {seed: acc}
    frozen = defaultdict(dict)                                 # budget -> {seed: acc}  (one per seed)
    for path in sorted(RESULTS_DIR.glob(f"coarse_budget_sweep_{selector}_seed*.csv")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        tag = match.group("tag") or "default"                 # untagged original run -> 'default'
        if tag.startswith("shard"):                           # skip per-shard partials, not variants
            continue
        seed = match.group("seed")
        for n, (fz, tn) in load_csv(path).items():
            if not np.isnan(tn):
                tuned[tag][n][seed] = tn                       # last write per (tag,seed) wins (dedup)
            if not np.isnan(fz):
                frozen[n][seed] = fz                           # one frozen value per seed (dedup)
    # collapse the per-seed dicts to lists of one-value-per-seed
    tuned_by_tag = {t: {n: list(sd.values()) for n, sd in b.items()} for t, b in tuned.items()}
    frozen_by_budget = {n: list(sd.values()) for n, sd in frozen.items()}
    return tuned_by_tag, frozen_by_budget


def summarize(by_budget):
    """{num_labeled: [vals]} -> sorted (budgets, mean, std, n) arrays. std is population std (ddof=0);
    a single-seed budget gets std 0."""
    budgets = sorted(by_budget)
    mean = np.array([np.mean(by_budget[n]) for n in budgets])
    std = np.array([np.std(by_budget[n]) for n in budgets])
    count = np.array([len(by_budget[n]) for n in budgets])
    return np.array(budgets), mean, std, count


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-seed coarse-budget CSVs to mean+/-std.")
    parser.add_argument("--selector", default="random")
    parser.add_argument("--max_budget", type=int, default=None,
                        help="only plot/write budgets <= this (e.g. 3000 for the low-budget region)")
    parser.add_argument("--tags", nargs="*", default=None,
                        help="restrict to these variant tags (default: all found)")
    parser.add_argument("--out", default="coarse_budget_seed_mean_std",
                        help="output basename (writes <out>.csv and <out>.png under outputs/finetune/)")
    args = parser.parse_args()

    tuned_by_tag, frozen_by_budget = collect(args.selector)
    tags = args.tags if args.tags else sorted(tuned_by_tag)
    cap = args.max_budget or 10**12

    # --- CSV: one row per (curve, budget) with mean, std, n_seeds ---
    csv_path = RESULTS_DIR / f"{args.out}.csv"
    with open(csv_path, "w") as handle:
        handle.write("curve,num_labeled,mean_acc,std_acc,n_seeds\n")
        if frozen_by_budget:                                   # frozen first, as the shared baseline
            b, m, s, c = summarize(frozen_by_budget)
            for i in range(len(b)):
                if b[i] <= cap:
                    handle.write(f"frozen,{b[i]},{m[i]:.4f},{s[i]:.4f},{c[i]}\n")
        for tag in tags:
            b, m, s, c = summarize(tuned_by_tag.get(tag, {}))
            for i in range(len(b)):
                if b[i] <= cap:
                    handle.write(f"{tag},{b[i]},{m[i]:.4f},{s[i]:.4f},{c[i]}\n")
    print(f"saved -> {csv_path}", flush=True)

    # --- PNG: mean line + std band per curve ---
    fig, ax = plt.subplots(figsize=(9, 6))
    if frozen_by_budget:
        b, m, s, _ = summarize(frozen_by_budget)
        keep = b <= cap
        ax.plot(b[keep], m[keep], "o--", color="grey", label="frozen")
        ax.fill_between(b[keep], (m - s)[keep], (m + s)[keep], color="grey", alpha=0.15)
    for tag in tags:
        b, m, s, _ = summarize(tuned_by_tag.get(tag, {}))
        keep = b <= cap
        line, = ax.plot(b[keep], m[keep], "o-", label=tag)     # mean line
        ax.fill_between(b[keep], (m - s)[keep], (m + s)[keep], color=line.get_color(), alpha=0.15)  # +/-std band
    ax.set_xscale("log")
    ax.set_xlabel("number of coarse-labeled training points")
    ax.set_ylabel("test 20-way superclass accuracy (%)")
    ax.set_title(f"Superclass accuracy vs budget (mean +/- std over seeds), selector={args.selector}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = RESULTS_DIR / f"{args.out}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"saved -> {png_path}", flush=True)


if __name__ == "__main__":
    main()
