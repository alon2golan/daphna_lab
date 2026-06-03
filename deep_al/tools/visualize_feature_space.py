"""
Reduce pre-computed embeddings to 2D and plot them colored by class label.

Produces, under --out_dir (default: results/feature-space/):
    {dataset}_labels.npy                   shape (N,)        ground-truth labels
    {dataset}_{method}_embeddings.npy      shape (N, 2)      2D coordinates
    {dataset}_{method}_indices.npy         shape (N,)        original indices used
    {dataset}_{method}.png                 scatter plot

The _indices.npy file is the identity np.arange(N) unless --max_samples is
used, in which case it lets the AL overlay (in deep_al/tools/utils.py)
remap full-dataset lSet/activeSet indices into plot positions.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def get_distinct_palette(n):
    """Return a list of n RGB tuples chosen to be visually distinct.

    Strategy: matplotlib tab10 for n<=10, tab20 for n<=20, otherwise
    colorcet's glasbey (256 maximally-distinct colors, falls back to husl
    if colorcet is unavailable).
    """
    import matplotlib.colors as mcolors
    if n <= 10:
        cmap = plt.get_cmap('tab10')
        return [cmap(i) for i in range(n)]
    if n <= 20:
        cmap = plt.get_cmap('tab20')
        return [cmap(i) for i in range(n)]
    try:
        import colorcet as cc
        hexes = cc.b_glasbey_bw_minc_20[:n]
        return [mcolors.to_rgb(h) for h in hexes]
    except ImportError:
        return sns.color_palette("husl", n)

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import deep_al.pycls.datasets.utils as ds_utils
from deep_al.pycls.datasets.utils.features import DATASET_FEATURES_DICT, get_dataset_name_with_tag

REPO_BASE = _repo_root
DEFAULT_CIFAR_ROOT = os.path.join(REPO_BASE, "data")
DEFAULT_OUT_DIR = os.path.join(REPO_BASE, "results", "feature-space")


def load_class_names(dataset, train=True, cifar_root=DEFAULT_CIFAR_ROOT, num_classes=None):
    """Return human-readable class names aligned with integer label ids, or None."""
    if dataset.startswith("CIFAR10") and not dataset.startswith("CIFAR100"):
        import torchvision
        return list(torchvision.datasets.CIFAR10(cifar_root, train=train, download=False).classes)
    if dataset.startswith("CIFAR100"):
        import torchvision
        return list(torchvision.datasets.CIFAR100(cifar_root, train=train, download=False).classes)
    if dataset in ("SCENARIO_A", "HALF_MOON") and num_classes is not None:
        return [f"class {i}" for i in range(num_classes)]
    return None


def classmap_layout(n):
    """Pick (ncols, nrows) for the class-name legend panel based on class count."""
    ncols = 1 if n <= 12 else (2 if n <= 30 else 5)
    nrows = int(np.ceil(n / ncols))
    return ncols, nrows


def draw_classmap_on_axes(ax, labels, colors, title=None):
    """Render a color-swatch + label grid onto an existing matplotlib axes.

    labels and colors are aligned lists drawn in the order given (callers
    presort them — e.g. alphabetically — and supply matching colors).
    """
    n = len(labels)
    ncols, nrows = classmap_layout(n)
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        c, r = i % ncols, i // ncols
        ax.add_patch(mpatches.Rectangle((c + 0.05, r + 0.15), 0.25, 0.7,
                                        facecolor=col, edgecolor='black', linewidth=0.5))
        ax.text(c + 0.38, r + 0.5, lbl,
                va='center', ha='left', fontsize=9)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)


def build_sorted_classmap(classes, palette, class_names=None):
    """Return (labels, colors) for the legend sorted alphabetically by class
    name, with each label prefixed by its integer class id so the mapping
    stays unambiguous. Falls back to id order when no names are known.
    """
    triples = [(int(c), (class_names[int(c)] if class_names else str(int(c))), palette[i])
               for i, c in enumerate(classes)]
    if class_names:
        triples.sort(key=lambda t: t[1].lower())
    labels = [f"{cid}: {name}" for cid, name, _ in triples]
    colors = [color for _, _, color in triples]
    return labels, colors


def load_labels(dataset, train=True, cifar_root=DEFAULT_CIFAR_ROOT, features_tag=None):
    """Return ground-truth labels aligned with load_features() ordering."""
    split = "train" if train else "test"
    if dataset.startswith("CIFAR10") and not dataset.startswith("CIFAR100"):
        import torchvision
        ds = torchvision.datasets.CIFAR10(cifar_root, train=train, download=False)
        return np.asarray(ds.targets)
    if dataset.startswith("CIFAR100"):
        import torchvision
        ds = torchvision.datasets.CIFAR100(cifar_root, train=train, download=False)
        return np.asarray(ds.targets)
    if dataset in ("SCENARIO_A", "HALF_MOON"):
        full_name = get_dataset_name_with_tag(dataset, features_tag)
        path = DATASET_FEATURES_DICT[split][full_name]
        return np.loadtxt(path)[:, 2].astype(np.int64)
    raise NotImplementedError(
        f"Label loading not implemented for dataset={dataset!r}. "
        "Add a branch in load_labels()."
    )


def stratified_subsample(labels, max_samples, seed):
    """Pick a per-class-balanced subset of size <= max_samples."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = max(1, max_samples // len(classes))
    picked = []
    for c in classes:
        idx = np.where(labels == c)[0]
        k = min(per_class, len(idx))
        picked.append(rng.choice(idx, size=k, replace=False))
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out[:max_samples]


def reduce_2d(features, method, seed, perplexity, n_neighbors, pca_init_components):
    """Reduce features to 2D using method in {pca, tsne, umap}."""
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(features)
    if method == "tsne":
        from sklearn.manifold import TSNE
        x = features
        if pca_init_components and x.shape[1] > pca_init_components:
            from sklearn.decomposition import PCA
            x = PCA(n_components=pca_init_components, random_state=seed).fit_transform(x)
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=seed,
            n_jobs=-1,
            verbose=1,
        ).fit_transform(x)
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise RuntimeError(
                "umap-learn not installed. Install with 'pip install umap-learn' "
                "or pick --method tsne / pca."
            ) from e
        return umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=seed,
            verbose=True,
        ).fit_transform(features)
    raise ValueError(f"Unknown method: {method!r}")


def plot_scatter(emb2d, labels, save_path, title, class_names=None,
                 point_size=6, alpha=0.6, aug2d=None,
                 aug_point_size=2, aug_alpha=0.2):
    """Render scatter + class-name colormap side-by-side into a single PNG.

    aug2d, if given, has shape (N, K, 2) and is drawn as a faint cloud
    underneath the clean points, colored the same way per parent image.
    """
    classes = np.unique(labels)
    num_classes = len(classes)
    palette = get_distinct_palette(num_classes)
    name_for = (lambda i: class_names[int(i)]) if class_names else (lambda i: f"{int(i)}")
    names_ordered = [name_for(c) for c in classes]
    hue = np.array([name_for(c) for c in labels])
    palette_dict = {names_ordered[i]: palette[i] for i in range(num_classes)}

    ncols, nrows = classmap_layout(num_classes)
    legend_w = max(2.6, 2.6 * ncols)
    legend_h_needed = 0.32 * nrows + 0.6
    fig_h = max(10, legend_h_needed)
    fig = plt.figure(figsize=(10 + legend_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[10, legend_w], wspace=0.02)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])

    if aug2d is not None:
        aug_flat = aug2d.reshape(-1, 2)
        aug_hue = np.repeat(hue, aug2d.shape[1])
        sns.scatterplot(
            ax=ax_scatter,
            x=aug_flat[:, 0], y=aug_flat[:, 1],
            hue=aug_hue, hue_order=names_ordered, palette=palette_dict,
            s=aug_point_size, alpha=aug_alpha, linewidth=0,
            legend=False,
        )

    sns.scatterplot(
        ax=ax_scatter,
        x=emb2d[:, 0], y=emb2d[:, 1],
        hue=hue, hue_order=names_ordered, palette=palette_dict,
        s=point_size, alpha=alpha, linewidth=0,
        legend=False,
    )
    ax_scatter.set_title(title)
    ax_scatter.set_axis_off()

    legend_labels, legend_colors = build_sorted_classmap(classes, palette, class_names)
    draw_classmap_on_axes(ax_legend, legend_labels, legend_colors,
                          title="color → class (A→Z)")

    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_method(features, labels, indices, method, dataset, out_dir, seed,
               perplexity, n_neighbors, pca_init_components, class_names=None,
               aug_features=None):
    """Reduce `features` to 2D and plot.

    If aug_features (shape (N, K, D)) is given, clean and augmented features
    are reduced JOINTLY so the augmented points land in the same 2D space.
    The augmented cloud is overlaid in a separate <method>_augK<K>.png file
    next to the standard plot; the standard outputs are unchanged.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    if aug_features is None:
        print(f"[{method}] reducing {features.shape} ...")
        emb2d = reduce_2d(
            features, method, seed=seed,
            perplexity=perplexity, n_neighbors=n_neighbors,
            pca_init_components=pca_init_components,
        )
        print(f"[{method}] done in {time.time() - t0:.1f}s")
        emb_path = os.path.join(out_dir, f"{dataset}_{method}_embeddings.npy")
        idx_path = os.path.join(out_dir, f"{dataset}_{method}_indices.npy")
        png_path = os.path.join(out_dir, f"{dataset}_{method}.png")
        np.save(emb_path, emb2d)
        is_identity = (len(indices) == len(emb2d) and
                       np.array_equal(indices, np.arange(len(emb2d))))
        if not is_identity:
            np.save(idx_path, indices)
        plot_scatter(
            emb2d, labels, png_path,
            title=f"{method.upper()} of {dataset} ({len(labels)} pts, {len(np.unique(labels))} classes)",
            class_names=class_names,
        )
        print(f"[{method}] saved {emb_path}\n[{method}] saved {png_path}")
        return

    # --augmented_features path: reduce clean+aug jointly and write only the
    # _aug{K}-suffixed files. The standard {dataset}_{method}_embeddings.npy
    # and {dataset}_{method}.png from a prior clean-only run are NOT touched.
    N, K, D = aug_features.shape
    assert N == features.shape[0] and D == features.shape[1], (
        f"aug_features shape {aug_features.shape} incompatible with "
        f"clean features {features.shape}"
    )
    stacked = np.concatenate([features, aug_features.reshape(N * K, D)], axis=0)
    print(f"[{method}] reducing {stacked.shape} (clean+aug joint) ...")
    emb2d_all = reduce_2d(
        stacked, method, seed=seed,
        perplexity=perplexity, n_neighbors=n_neighbors,
        pca_init_components=pca_init_components,
    )
    emb2d = emb2d_all[:N]
    aug2d = emb2d_all[N:].reshape(N, K, 2)
    print(f"[{method}] done in {time.time() - t0:.1f}s")
    aug_emb_path = os.path.join(out_dir, f"{dataset}_{method}_aug{K}_clean_embeddings.npy")
    aug_cloud_path = os.path.join(out_dir, f"{dataset}_{method}_aug{K}_embeddings.npy")
    aug_idx_path = os.path.join(out_dir, f"{dataset}_{method}_aug{K}_indices.npy")
    aug_png_path = os.path.join(out_dir, f"{dataset}_{method}_aug{K}.png")
    np.save(aug_emb_path, emb2d)
    np.save(aug_cloud_path, aug2d)
    is_identity = (len(indices) == len(emb2d) and
                   np.array_equal(indices, np.arange(len(emb2d))))
    if not is_identity:
        np.save(aug_idx_path, indices)
    plot_scatter(
        emb2d, labels, aug_png_path,
        title=(f"{method.upper()} of {dataset} "
               f"({len(labels)} pts, {len(np.unique(labels))} classes, "
               f"K={K} augmentations per image)"),
        class_names=class_names, aug2d=aug2d,
    )
    print(f"[{method}] saved {aug_emb_path}\n[{method}] saved {aug_cloud_path}\n[{method}] saved {aug_png_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="CIFAR10, CIFAR100, TINYIMAGENET, IMAGENET, SCENARIO_A, HALF_MOON, ...")
    parser.add_argument("--method", choices=["pca", "tsne", "umap", "all"], default="all")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--tag", default=None, help="features_tag for load_features (e.g. 'alon')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="If set, stratified-subsample to this many points before reduction.")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Skip L2 row-normalization of features before reduction.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cifar_root", default=DEFAULT_CIFAR_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--umap_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--pca_init_components", type=int, default=50,
                        help="If >0, PCA-reduce to this many dims before t-SNE (default 50). "
                             "Set 0 to disable.")
    parser.add_argument("--augmented_features", default=None,
                        help="Path to a (N, K, D) .npy file produced by "
                             "scan/extract_augmented_features.py. When set, K augmented "
                             "views per image are reduced jointly with the clean features "
                             "and overlaid on a second plot suffixed _aug{K}.png. Clean "
                             "outputs are not affected.")
    args = parser.parse_args()

    features = ds_utils.load_features(
        args.dataset, seed=args.seed, train=(args.split == "train"),
        normalized=not args.no_normalize, features_tag=args.tag,
    )
    labels = load_labels(args.dataset, train=(args.split == "train"),
                         cifar_root=args.cifar_root, features_tag=args.tag)
    assert features.shape[0] == labels.shape[0], (
        f"feature/label length mismatch: {features.shape[0]} vs {labels.shape[0]}"
    )

    aug_features = None
    if args.augmented_features:
        aug_features = np.load(args.augmented_features)
        assert aug_features.ndim == 3, (
            f"augmented features must be (N, K, D); got {aug_features.shape}"
        )
        assert aug_features.shape[0] == features.shape[0], (
            f"augmented N={aug_features.shape[0]} does not match clean N={features.shape[0]}"
        )
        assert aug_features.shape[2] == features.shape[1], (
            f"augmented D={aug_features.shape[2]} does not match clean D={features.shape[1]}"
        )
        if not args.no_normalize:
            norms = np.linalg.norm(aug_features, axis=2, keepdims=True)
            norms[norms == 0] = 1.0
            aug_features = aug_features / norms
        print(f"Loaded augmented features: {aug_features.shape}")

    if args.max_samples and args.max_samples < features.shape[0]:
        keep = stratified_subsample(labels, args.max_samples, args.seed)
        features = features[keep]
        labels = labels[keep]
        if aug_features is not None:
            aug_features = aug_features[keep]
        indices = keep
        print(f"Subsampled to {len(keep)} points (stratified).")
    else:
        indices = np.arange(features.shape[0])

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, f"{args.dataset}_labels.npy"), labels)
    print(f"Saved labels: {args.out_dir}/{args.dataset}_labels.npy  shape={labels.shape}")

    num_classes = len(np.unique(labels))
    class_names = load_class_names(
        args.dataset, train=(args.split == "train"),
        cifar_root=args.cifar_root, num_classes=num_classes,
    )
    if class_names is not None:
        names_path = os.path.join(args.out_dir, f"{args.dataset}_class_names.json")
        with open(names_path, "w") as f:
            json.dump(class_names, f, indent=2)
        print(f"Saved class names: {names_path}")
    else:
        print(f"No class names known for dataset={args.dataset}; legend will show integer ids.")

    methods = ["pca", "tsne", "umap"] if args.method == "all" else [args.method]
    for m in methods:
        try:
            run_method(
                features, labels, indices, m,
                dataset=args.dataset, out_dir=args.out_dir, seed=args.seed,
                perplexity=args.perplexity, n_neighbors=args.umap_neighbors,
                pca_init_components=args.pca_init_components,
                class_names=class_names, aug_features=aug_features,
            )
        except RuntimeError as e:
            print(f"[{m}] SKIPPED: {e}")


if __name__ == "__main__":
    main()
