# Post-run diagnostics. After an active-learning run finishes, turn the record of
# WHICH points got chosen into two figures (+ CSVs) under outputs/figures/al_picks/.
# These never affect the run -- they only visualize what the strategy did, so we can
# sanity-check its behaviour:
#
#   plot_picks_per_class  -- did the strategy spread its labels across all classes,
#                            or pile them onto a few? (class balance of the picks)
#   plot_coverage_gain    -- for coverage strategies (ProbCover): how much NEW ground
#                            each labeled point covered, and how that fades over the run.
#
# Split out of run_active_learning.py so that file stays purely the AL loop.

from pathlib import Path

import numpy as np

# every diagnostic lands under hierarchical_framework/outputs/figures/al_picks/
FIGDIR = Path(__file__).resolve().parents[1] / "outputs" / "figures" / "al_picks"


def plot_picks_per_class(picked_classes_by_round, num_classes, out_base, use_wandb=False):
    # picked_classes_by_round[r] = the class labels of the points chosen in round r.
    # Purpose: reveal whether the strategy touches every class or fixates on a few.
    import matplotlib
    matplotlib.use("Agg")                 # write straight to a PNG, no GUI/display needed
    import matplotlib.pyplot as plt

    n_rounds = len(picked_classes_by_round)
    # counts[c, r] = how many points of class c were chosen in round r
    counts = np.zeros((num_classes, n_rounds), dtype=int)
    for r, classes_this_round in enumerate(picked_classes_by_round):
        for c in classes_this_round:
            counts[c, r] += 1
    # total_per_class[c] = how many times class c was chosen over the whole run
    total_per_class = counts.sum(axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # LEFT panel: heatmap of counts -- rows = classes, columns = rounds, brightness = how
    # many of that class were taken that round (bright spots show when a class gets picked)
    im = ax[0].imshow(counts, aspect="auto", cmap="viridis", interpolation="nearest")
    ax[0].set_xlabel("round"); ax[0].set_ylabel("class")
    ax[0].set_title("picks per class per round")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    # RIGHT panel: bar chart of the per-class totals; title reports how many of the
    # num_classes classes were EVER chosen -- a one-glance class-coverage summary
    ax[1].bar(np.arange(num_classes), total_per_class, color="steelblue")
    ax[1].set_xlabel("class"); ax[1].set_ylabel("total picks")
    n_classes_hit = int((total_per_class > 0).sum())
    ax[1].set_title(f"picks per class (distinct chosen={n_classes_hit}/{num_classes})")
    fig.tight_layout()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / f"{out_base}_classpicks.png"
    fig.savefig(png, dpi=130); print(f"saved -> {png}", flush=True)

    # CSV companion: one row per class = its total picks and the exact rounds it was chosen in
    csv = FIGDIR / f"{out_base}_classpicks.csv"
    with open(csv, "w") as fh:
        fh.write("class,total,rounds\n")
        for c in range(num_classes):
            rounds_hit = " ".join(str(r) for r in range(n_rounds) if counts[c, r] > 0)
            fh.write(f"{c},{int(total_per_class[c])},{rounds_hit}\n")
    print(f"saved -> {csv}", flush=True)

    if use_wandb:
        import wandb
        # mirror the figure + a per-class table into the wandb run
        wandb.log({"class_picks": wandb.Image(str(png))})
        table = wandb.Table(columns=["class", "total", "rounds"])
        for c in range(num_classes):
            rounds_hit = " ".join(str(r) for r in range(n_rounds) if counts[c, r] > 0)
            table.add_data(int(c), int(total_per_class[c]), rounds_hit)
        wandb.log({"class_picks_table": table})
    plt.close(fig)
    return counts


def plot_coverage_gain(coverage_gains_by_round, labels_per_class_by_round, out_base, use_wandb=False):
    # coverage_gains_by_round[r] = for each point chosen in round r, how many previously
    # UNCOVERED points labeling it newly covered (only ProbCover-family strategies fill this).
    # Purpose: watch the gain shrink as the pool fills -- early picks cover lots of new
    # ground, later picks little; a healthy coverage strategy trends downward.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # one point on the curve per round: x = labels-per-class so far, y = that round's mean
    # gain, with a whisker spanning the round's smallest..largest gain
    x, mean_gain, min_gain, max_gain = [], [], [], []
    for labels_per_class, gains in zip(labels_per_class_by_round, coverage_gains_by_round):
        if len(gains) == 0:                        # nothing recorded (e.g. the random strategy) -> skip the round
            continue
        gains = np.asarray(gains, dtype=float)
        x.append(labels_per_class)
        mean_gain.append(gains.mean()); min_gain.append(gains.min()); max_gain.append(gains.max())
    x = np.array(x); mean_gain = np.array(mean_gain)
    # matplotlib wants error bars as DISTANCES from the mean (how far down, how far up),
    # not absolute min/max values
    yerr = np.vstack([mean_gain - np.array(min_gain), np.array(max_gain) - mean_gain])

    fig, ax = plt.subplots(figsize=(8, 5))
    # marker = the round's mean gain; whisker = that round's min..max spread
    ax.errorbar(x, mean_gain, yerr=yerr, fmt="o-", capsize=4, color="darkred", ecolor="gray")
    ax.set_xlabel("budget (labels per class, B)")
    ax.set_ylabel("newly covered points per chosen point")
    ax.set_title("coverage gain per pick: round mean (bar = min..max)")
    fig.tight_layout()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / f"{out_base}_pickdegree.png"
    fig.savefig(png, dpi=130); print(f"saved -> {png}", flush=True)
    if use_wandb:
        import wandb
        wandb.log({"pick_degree": wandb.Image(str(png))})
    plt.close(fig)
