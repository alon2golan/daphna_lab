# Two diagnostic figures written after an AL run, into outputs/figures/al_picks/:
#   report_picks_per_class -- which classes the strategy chose, and in which round
#   report_coverage_gain   -- how many new points each chosen point covered
# Kept out of run_active_learning.py so that file stays just the AL loop.

from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs" # figures land under hierarchical_framework/outputs/


def report_picks_per_class(picked_classes_by_round, num_classes, out_base, use_wandb=False):
    import matplotlib
    matplotlib.use("Agg") # render straight to a file, no display needed
    import matplotlib.pyplot as plt
    n_rounds = len(picked_classes_by_round)
    counts = np.zeros((num_classes, n_rounds), dtype=int) # counts[c, r] = how many class-c points were chosen in round r
    for r, classes in enumerate(picked_classes_by_round):
        for c in classes:
            counts[c, r] += 1
    total_per_class = counts.sum(axis=1) # how often each class was chosen across the whole run

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(counts, aspect="auto", cmap="viridis", interpolation="nearest") # left: which class was chosen in which round
    ax[0].set_xlabel("round"); ax[0].set_ylabel("class")
    ax[0].set_title("picks per class per round")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].bar(np.arange(num_classes), total_per_class, color="steelblue") # right: total picks per class
    ax[1].set_xlabel("class"); ax[1].set_ylabel("total picks")
    ax[1].set_title(f"picks per class (distinct chosen={int((total_per_class > 0).sum())}/{num_classes})")
    fig.tight_layout()

    figdir = OUTPUTS / "figures" / "al_picks"
    figdir.mkdir(parents=True, exist_ok=True)
    png = figdir / f"{out_base}_classpicks.png"
    fig.savefig(png, dpi=130); print(f"saved -> {png}", flush=True)

    csv = figdir / f"{out_base}_classpicks.csv" # table: per class, its total and the exact rounds it was chosen in
    with open(csv, "w") as fh:
        fh.write("class,total,rounds\n")
        for c in range(num_classes):
            rounds = " ".join(str(r) for r in range(n_rounds) if counts[c, r] > 0)
            fh.write(f"{c},{int(total_per_class[c])},{rounds}\n")
    print(f"saved -> {csv}", flush=True)

    if use_wandb:
        import wandb
        wandb.log({"class_picks": wandb.Image(str(png))}) # push the figure into the run
        tbl = wandb.Table(columns=["class", "total", "rounds"])
        for c in range(num_classes):
            rounds = " ".join(str(r) for r in range(n_rounds) if counts[c, r] > 0)
            tbl.add_data(int(c), int(total_per_class[c]), rounds)
        wandb.log({"class_picks_table": tbl})
    plt.close(fig)
    return counts


def report_coverage_gain(coverage_gains_by_round, labels_per_class_by_round, out_base, use_wandb=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, mean_gain, min_gain, max_gain = [], [], [], []
    for labels_per_class, gains in zip(labels_per_class_by_round, coverage_gains_by_round): # per round: summarize its picks' coverage gains
        if len(gains) == 0:
            continue
        gains = np.asarray(gains, dtype=float)
        x.append(labels_per_class); mean_gain.append(gains.mean()); min_gain.append(gains.min()); max_gain.append(gains.max())
    x = np.array(x); mean_gain = np.array(mean_gain)
    yerr = np.vstack([mean_gain - np.array(min_gain), np.array(max_gain) - mean_gain]) # error bar spans min..max around the mean

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, mean_gain, yerr=yerr, fmt="o-", capsize=4, color="darkred", ecolor="gray") # point = round mean, bar = round min..max
    ax.set_xlabel("budget (labels per class, B)")
    ax.set_ylabel("newly covered points per chosen point")
    ax.set_title("coverage gain per pick: round mean (bar = min..max)")
    fig.tight_layout()

    figdir = OUTPUTS / "figures" / "al_picks"
    figdir.mkdir(parents=True, exist_ok=True)
    png = figdir / f"{out_base}_pickdegree.png"
    fig.savefig(png, dpi=130); print(f"saved -> {png}", flush=True)
    if use_wandb:
        import wandb
        wandb.log({"pick_degree": wandb.Image(str(png))})
    plt.close(fig)
