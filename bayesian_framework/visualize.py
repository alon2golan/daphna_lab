"""Visualize a figures matrix (matrix 2) as a graph over the labeled pool.

The figures is stored sparsely as a k-NN pair (idx.npy + sim.npy), so the
dense N x N is never touched. Given the point labels, this renders three views
that actually fit on screen and say something:

  1. class-block heatmap   -- (C x C) fraction of each class's edges landing in
                              every class; a good matrix is strongly diagonal.
  2. degree histogram      -- edges per point (how many neighbors within the cut).
  3. sorted sub-block      -- dense adjacency among a few whole classes, sorted by
                              label, so the block-diagonal structure is visible.

It also prints summary stats (edges, degree, isolated points, same-class purity).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # render to file, no display needed on the cluster
import matplotlib.pyplot as plt


def load_similarity(sim_path):
    # sim_path is the *_sim.npy; its partner *_idx.npy holds the neighbor indices
    sim_path = Path(sim_path)
    idx_path = sim_path.with_name(sim_path.name.replace("_sim.npy", "_idx.npy"))
    knn_idx = np.load(idx_path)          # (N, k) neighbor indices
    knn_sim = np.load(sim_path)          # (N, k) figures weights (tophat: 0/1)
    return knn_idx, knn_sim


def class_block_matrix(knn_idx, knn_sim, labels, num_classes):
    # accumulate edge weight between class(i) and class(neighbor) into a C x C grid
    block = np.zeros((num_classes, num_classes), dtype=np.float64)
    # label of every neighbor, same (N, k) shape as the index array
    nbr_labels = labels[knn_idx]
    src = labels[:, None]                # each point's own label, broadcast over k
    # add every edge's weight to block[src_label, neighbor_label]
    np.add.at(block, (np.broadcast_to(src, nbr_labels.shape).ravel(),
                      nbr_labels.ravel()), knn_sim.ravel())
    return block


def summarize(knn_sim, block):
    # an edge exists where the figures weight is nonzero
    edges = knn_sim > 0
    # per-point degree = number of neighbors kept after the cut
    degree = edges.sum(axis=1)
    total = float(block.sum())
    # diagonal = edges that stay within the point's own class
    purity = float(np.trace(block)) / total if total else 0.0
    stats = {
        "n_points": int(knn_sim.shape[0]),
        "total_edges": int(edges.sum()),
        "degree_mean": float(degree.mean()),
        "degree_median": float(np.median(degree)),
        "degree_max": int(degree.max()),
        "isolated": int((degree == 0).sum()),
        "same_class_fraction": purity,
    }
    return stats, degree


def sorted_subblock(knn_idx, knn_sim, labels, classes):
    # gather every point belonging to the chosen classes, ordered by class so the
    # image comes out block-diagonal
    keep = np.isin(labels, classes)
    order = np.where(keep)[0][np.argsort(labels[keep], kind="stable")]
    # map global point id -> its row/col in the small dense sub-adjacency
    pos = -np.ones(labels.shape[0], dtype=np.int64)
    pos[order] = np.arange(order.shape[0])
    m = order.shape[0]
    A = np.zeros((m, m), dtype=np.float32)
    # for each kept point, drop its in-subset neighbors into the dense block
    for local_i, i in enumerate(order):
        j = knn_idx[i]
        w = knn_sim[i]
        in_sub = pos[j] >= 0            # neighbor also belongs to the subset
        A[local_i, pos[j[in_sub]]] = w[in_sub]
    return A, labels[order]


def visualize(sim_path, labels_path, out_path=None, subblock_classes=10):
    knn_idx, knn_sim = load_similarity(sim_path)
    labels = np.load(labels_path)
    num_classes = int(labels.max()) + 1

    # C x C class-to-class edge counts, and row-normalized fractions for the heatmap
    block = class_block_matrix(knn_idx, knn_sim, labels, num_classes)
    row = block.sum(axis=1, keepdims=True)
    block_frac = np.divide(block, row, out=np.zeros_like(block), where=row > 0)

    stats, degree = summarize(knn_sim, block)
    print("summary:", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)

    # dense adjacency among the first `subblock_classes` classes, sorted by label
    classes = np.arange(min(subblock_classes, num_classes))
    A, _ = sorted_subblock(knn_idx, knn_sim, labels, classes)

    # three panels side by side
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    # panel 1: class-block heatmap (diagonal = within-class edges)
    im0 = ax[0].imshow(block_frac, cmap="viridis", vmin=0, vmax=1)
    ax[0].set_title(f"class-block edge fraction\nsame-class = {stats['same_class_fraction']:.3f}")
    ax[0].set_xlabel("neighbor class"); ax[0].set_ylabel("point class")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    # panel 2: degree distribution
    ax[1].hist(degree, bins=60, color="steelblue")
    ax[1].set_title(f"degree (edges/point)\nmean={stats['degree_mean']:.1f} "
                    f"max={stats['degree_max']} isolated={stats['isolated']}")
    ax[1].set_xlabel("neighbors within cut"); ax[1].set_ylabel("points")

    # panel 3: sorted sub-block adjacency for a handful of classes
    ax[2].imshow(A, cmap="Greys", interpolation="nearest", aspect="auto")
    ax[2].set_title(f"adjacency, first {len(classes)} classes\n(sorted by label)")
    ax[2].set_xlabel("point"); ax[2].set_ylabel("point")

    fig.tight_layout()

    if out_path:
        out = Path(out_path)
    else:
        # default: a figures/ folder in the framework dir, named after the matrix
        out_dir = Path(__file__).resolve().parent / "figures" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(sim_path).name.replace("_sim.npy", "")
        out = out_dir / f"{base}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sim", required=True,
                   help="path to the *_sim.npy from figures.py (--npy output)")
    p.add_argument("--labels", required=True,
                   help="path to the <split>_labels.npy for the same points")
    p.add_argument("--out_path", default=None,
                   help="where to save the .png (default: framework figures/ folder)")
    p.add_argument("--subblock_classes", type=int, default=10,
                   help="how many classes to draw in the sorted adjacency panel")
    args = p.parse_args()

    visualize(args.sim, args.labels, out_path=args.out_path,
              subblock_classes=args.subblock_classes)


if __name__ == "__main__":
    main()
