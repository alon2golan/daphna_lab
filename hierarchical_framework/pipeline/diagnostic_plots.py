# Post-run diagnostic figures (+ CSVs) written to outputs/figures/al_picks/. They
# only visualize what the strategy picked; they never affect the run itself.

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg") # render straight to PNG files, no GUI/display needed
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None # only touched when a run passes use_wandb=True

FIGDIR = Path(__file__).resolve().parents[1] / "outputs" / "figures" / "al_picks" # where every diagnostic lands


def plot_picks_per_class(picked_classes_by_round, num_classes, out_base, use_wandb=False):
    n_rounds = len(picked_classes_by_round)
    counts = np.zeros((num_classes, n_rounds), dtype=int) # counts[c, r] = how many class-c points were chosen in round r
    for r, classes_this_round in enumerate(picked_classes_by_round): # tally each round's picks into their class rows
        for c in classes_this_round:
            counts[c, r] += 1
    total_per_class = counts.sum(axis=1) # how many times each class was chosen over the whole run

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(counts, aspect="auto", cmap="viridis", interpolation="nearest") # left: class x round heatmap, bright = picked then
    ax[0].set_xlabel("round"); ax[0].set_ylabel("class")
    ax[0].set_title("picks per class per round")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].bar(np.arange(num_classes), total_per_class, color="steelblue") # right: total picks per class
    n_classes_hit = int((total_per_class > 0).sum()) # how many distinct classes were ever chosen -- a class-coverage summary
    ax[1].set_xlabel("class"); ax[1].set_ylabel("total picks")
    ax[1].set_title(f"picks per class (distinct chosen={n_classes_hit}/{num_classes})")
    fig.tight_layout()

    png = FIGDIR / f"{out_base}_classpicks.png"
    png.parent.mkdir(parents=True, exist_ok=True) # out_base is a nested <dataset>/<technique>/... path
    fig.savefig(png, dpi=130); print(f"saved -> {png}", flush=True)

    csv = FIGDIR / f"{out_base}_classpicks.csv" # companion table: per class, its total and the exact rounds it was chosen in
    with open(csv, "w") as fh:
        fh.write("class,total,rounds\n")
        for c in range(num_classes):
            rounds_hit = " ".join(str(r) for r in range(n_rounds) if counts[c, r] > 0)
            fh.write(f"{c},{int(total_per_class[c])},{rounds_hit}\n")
    print(f"saved -> {csv}", flush=True)

    if use_wandb:
        wandb.log({"class_picks": wandb.Image(str(png))}) # mirror the figure into the run
        table = wandb.Table(columns=["class", "total", "rounds"]) # and the same per-class table
        for c in range(num_classes):
            rounds_hit = " ".join(str(r) for r in range(n_rounds) if counts[c, r] > 0)
            table.add_data(int(c), int(total_per_class[c]), rounds_hit)
        wandb.log({"class_picks_table": table})
    plt.close(fig)
    return counts


def plot_coverage_gain(coverage_gains_by_round, labels_per_class_by_round, out_base, use_wandb=False):
    # one curve point per round: mean coverage gain of that round's picks, with a min..max whisker
    x, mean_gain, min_gain, max_gain = [], [], [], []
    for labels_per_class, gains in zip(labels_per_class_by_round, coverage_gains_by_round):
        if len(gains) == 0: # nothing recorded (e.g. the random strategy) -> skip this round
            continue
        gains = np.asarray(gains, dtype=float)
        x.append(labels_per_class)
        mean_gain.append(gains.mean()); min_gain.append(gains.min()); max_gain.append(gains.max())
    x = np.array(x); mean_gain = np.array(mean_gain)
    yerr = np.vstack([mean_gain - np.array(min_gain), np.array(max_gain) - mean_gain]) # error bars as distances from the mean, not absolute min/max

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, mean_gain, yerr=yerr, fmt="o-", capsize=4, color="darkred", ecolor="gray") # marker = round mean, whisker = round min..max
    ax.set_xlabel("budget (labels per class, B)")
    ax.set_ylabel("newly covered points per chosen point")
    ax.set_title("coverage gain per pick: round mean (bar = min..max)")
    fig.tight_layout()

    png = FIGDIR / f"{out_base}_pickdegree.png"
    png.parent.mkdir(parents=True, exist_ok=True) # out_base is a nested <dataset>/<technique>/... path
    fig.savefig(png, dpi=130); print(f"saved -> {png}", flush=True)
    if use_wandb:
        wandb.log({"pick_degree": wandb.Image(str(png))}) # push the figure into the run
    plt.close(fig)
