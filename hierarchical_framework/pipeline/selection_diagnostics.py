# Post-run diagnostic figures (+ CSVs) written to outputs/figures/selection_diagnostics/.
# They only visualize what the strategy picked; they never affect the run itself.

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Render straight to PNG files; no GUI environment or display window needed
import matplotlib.pyplot as plt

# Resolve the absolute path where all diagnostic output images and tables will be saved
FIGDIR = Path(__file__).resolve().parents[1] / "outputs" / "figures" / "selection_diagnostics"


def plot_picks_per_class(picked_classes_by_round, num_classes, out_base):
    """
    Audits class balance across active learning rounds. Generates a dual-plot figure:
    1. A heatmap showing which classes were targeted round-by-round.
    2. A bar chart showing total accumulated selections per class.
    Also saves a raw companion CSV table. Local files only -- nothing goes to wandb.
    """
    n_rounds = len(picked_classes_by_round)

    # Initialize a 2D grid matrix: counts[class_id, round_id]
    counts = np.zeros((num_classes, n_rounds), dtype=int)

    # Tally up the occurrences of each class choice for every active learning round
    for r, classes_this_round in enumerate(picked_classes_by_round):
        for c in classes_this_round:
            counts[c, r] += 1

    # Calculate the total aggregate number of times each class was picked across the whole run
    total_per_class = counts.sum(axis=1)

    # Create a 1-row, 2-column figure frame for the side-by-side plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Left Plot: Generate the heatmap. Brighter colors indicate intense focus on that class in that round
    im = ax[0].imshow(counts, aspect="auto", cmap="viridis", interpolation="nearest")
    ax[0].set_xlabel("round")
    ax[0].set_ylabel("class")
    ax[0].set_title("picks per class per round")
    fig.colorbar(im, ax=ax[0], fraction=0.046)  # Add a color scale legend next to the heatmap

    # Right Plot: Generate a total aggregate bar chart of selections per class
    ax[1].bar(np.arange(num_classes), total_per_class, color="steelblue")

    # Count how many distinct classes managed to get at least one single pick during the run
    n_classes_hit = int((total_per_class > 0).sum())
    ax[1].set_xlabel("class")
    ax[1].set_ylabel("total picks")
    ax[1].set_title(f"picks per class (distinct chosen={n_classes_hit}/{num_classes})")
    fig.tight_layout()  # Auto-adjust layout spacing to prevent labels from overlapping

    # Save the combined visualization as a PNG (out_base is a nested <dataset>/<technique>/... path)
    png = FIGDIR / f"{out_base}_classpicks.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130)
    print(f"saved -> {png}", flush=True)

    # Generate a companion CSV spreadsheet tracking total counts and active round histories for each class
    csv = FIGDIR / f"{out_base}_classpicks.csv"
    with open(csv, "w") as fh:
        fh.write("class,total,rounds\n")
        for c in range(num_classes):
            # Extract a space-separated string of only the round numbers where this class was selected
            rounds_hit = " ".join(str(r) for r in range(n_rounds) if counts[c, r] > 0)
            fh.write(f"{c},{int(total_per_class[c])},{rounds_hit}\n")
    print(f"saved -> {csv}", flush=True)

    plt.close(fig)  # Free up system memory by explicitly closing the Matplotlib figure
    return counts


def plot_coverage_gain(coverage_gains_by_round, labels_per_class_by_round, out_base):
    """
    Audits strategy efficiency by plotting a geometric utility curve over time.
    It tracks how many brand-new points were covered per pick, showing whether the strategy
    is still discovering massive data clusters or hitting diminishing returns.
    """
    x, mean_gain, min_gain, max_gain = [], [], [], []

    # Process the metrics gathered across each round
    for labels_per_class, gains in zip(labels_per_class_by_round, coverage_gains_by_round):
        # Skip strategies (like random selection) that don't calculate or report coverage metrics
        if len(gains) == 0:
            continue
        gains = np.asarray(gains, dtype=float)
        x.append(labels_per_class)  # Store the current cumulative budget status (X-axis point)

        # Calculate statistical boundaries for the current batch of choices
        mean_gain.append(gains.mean())
        min_gain.append(gains.min())
        max_gain.append(gains.max())

    x = np.array(x)
    mean_gain = np.array(mean_gain)

    # Convert absolute minimum/maximum values into directional error bar offsets from the mean value
    yerr = np.vstack([mean_gain - np.array(min_gain), np.array(max_gain) - mean_gain])

    # Set up the plot window
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw a line chart where dots represent the round average efficiency, and whiskers represent the min/max range
    ax.errorbar(x, mean_gain, yerr=yerr, fmt="o-", capsize=4, color="darkred", ecolor="gray")
    ax.set_xlabel("budget (labels per class, B)")
    ax.set_ylabel("newly covered points per chosen point")
    ax.set_title("coverage gain per pick: round mean (bar = min..max)")
    fig.tight_layout()

    # Save the generated efficiency trend graph as a PNG file (out_base is a nested path)
    png = FIGDIR / f"{out_base}_pickdegree.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130)
    print(f"saved -> {png}", flush=True)

    plt.close(fig)  # Clean up memory allocations