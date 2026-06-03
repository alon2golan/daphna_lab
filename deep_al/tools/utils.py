import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import os

# Ensure repo root is in sys.path for clean imports
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Get the repository base directory
REPO_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import deep_al.pycls.datasets.utils as ds_utils


def _load_precomputed_2d(dataset_name, method, in_dir):
    """Read 2D embeddings + labels + (optional) original indices saved by
    deep_al/tools/visualize_feature_space.py.

    Returns (embeddings, labels, full_to_plot) where full_to_plot maps an
    original full-dataset index -> position in the plotted array, or None
    if no _indices file is present (i.e. plot covers the full dataset and
    the identity map suffices).
    """
    emb_path = os.path.join(in_dir, f"{dataset_name}_{method}_embeddings.npy")
    lab_path = os.path.join(in_dir, f"{dataset_name}_labels.npy")
    idx_path = os.path.join(in_dir, f"{dataset_name}_{method}_indices.npy")
    embeddings = np.load(emb_path)
    labels = np.load(lab_path)
    full_to_plot = None
    if os.path.exists(idx_path):
        orig_indices = np.load(idx_path)
        if len(orig_indices) != len(embeddings) or not np.array_equal(orig_indices, np.arange(len(embeddings))):
            full_to_plot = {int(o): p for p, o in enumerate(orig_indices)}
    return embeddings, labels, full_to_plot


def _remap(indices, full_to_plot):
    if indices is None or len(indices) == 0:
        return np.array([], dtype=np.int64)
    if full_to_plot is None:
        return np.asarray(indices, dtype=np.int64)
    return np.array([full_to_plot[int(i)] for i in indices if int(i) in full_to_plot], dtype=np.int64)


def visualize_feature_space(
    method: str = 'tsne',
    labeled_indices: np.ndarray = None,
    sampled_indices: np.ndarray = None,
    dataset_name: str = 'CIFAR10',
    algo_name: str = 'default',
    in_dir: str = None,
    out_dir: str = None,
    seed: int = None,
):
    """Overlay labeled / newly-sampled points on a precomputed 2D embedding
    of the feature space (PCA / t-SNE / UMAP), colored by class.

    Files expected in in_dir:
        {dataset_name}_{method}_embeddings.npy
        {dataset_name}_labels.npy
        {dataset_name}_{method}_indices.npy   (optional; needed if subsampled)

    Generate them with deep_al/tools/visualize_feature_space.py.
    """
    in_dir = in_dir or os.path.join(REPO_BASE, "results", "feature-space")
    out_dir = out_dir or os.path.join(REPO_BASE, "results", "feature-space-visual")
    os.makedirs(out_dir, exist_ok=True)

    from deep_al.tools.visualize_feature_space import (
        classmap_layout, draw_classmap_on_axes, get_distinct_palette, build_sorted_classmap,
    )

    embeddings, labels, full_to_plot = _load_precomputed_2d(dataset_name, method, in_dir)
    classes = np.unique(labels)
    num_classes = len(classes)

    names_path = os.path.join(in_dir, f"{dataset_name}_class_names.json")
    class_names = None
    if os.path.exists(names_path):
        with open(names_path) as f:
            class_names = json.load(f)

    name_for = (lambda i: class_names[int(i)]) if class_names else (lambda i: f"{int(i)}")
    palette = get_distinct_palette(num_classes)
    names_ordered = [name_for(c) for c in classes]
    palette_dict = {names_ordered[i]: palette[i] for i in range(num_classes)}
    hue_all = np.array([name_for(c) for c in labels])

    labeled_plot_idx = _remap(labeled_indices, full_to_plot)
    sampled_plot_idx = _remap(sampled_indices, full_to_plot)

    ncols, nrows = classmap_layout(num_classes)
    legend_w = max(2.6, 2.6 * ncols)
    legend_h_needed = 0.32 * nrows + 0.6
    fig_h = max(10, legend_h_needed)
    fig = plt.figure(figsize=(10 + legend_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[10, legend_w], wspace=0.02)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])

    sns.scatterplot(
        ax=ax_scatter,
        x=embeddings[:, 0], y=embeddings[:, 1],
        hue=hue_all, hue_order=names_ordered, palette=palette_dict,
        legend=False,
        s=15, alpha=0.7, linewidth=0,
    )
    if len(labeled_plot_idx) > 0:
        sns.scatterplot(
            ax=ax_scatter,
            x=embeddings[labeled_plot_idx, 0], y=embeddings[labeled_plot_idx, 1],
            hue=hue_all[labeled_plot_idx], hue_order=names_ordered, palette=palette_dict,
            marker='*', s=180, edgecolor='#222222', linewidth=1.6, legend=False,
        )
    if len(sampled_plot_idx) > 0:
        sns.scatterplot(
            ax=ax_scatter,
            x=embeddings[sampled_plot_idx, 0], y=embeddings[sampled_plot_idx, 1],
            hue=hue_all[sampled_plot_idx], hue_order=names_ordered, palette=palette_dict,
            marker='*', s=220, edgecolor='black', linewidth=2.0, legend=False,
        )
    ax_scatter.set_title(f"{method.upper()} of {dataset_name} — {algo_name}")
    ax_scatter.set_axis_off()

    legend_labels, legend_colors = build_sorted_classmap(classes, palette, class_names)
    draw_classmap_on_axes(ax_legend, legend_labels, legend_colors, title="color → class (A→Z)")

    num_labeled_samples = len(labeled_plot_idx)
    seed_tag = f"_seed{int(seed)}" if seed is not None else ""
    save_full_path = os.path.join(
        out_dir, f"{algo_name}_{dataset_name}_{method}{seed_tag}_labeled_{num_labeled_samples}.png"
    )
    fig.savefig(save_full_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return save_full_path


def visualize_tsne(
    labeled_indices: np.ndarray = None,
    sampled_indices: np.ndarray = None,
    dataset_name: str = 'CIFAR10',
    algo_name: str = 'default',
):
    """Back-compat shim. Prefer visualize_feature_space(method='tsne', ...)."""
    return visualize_feature_space(
        method='tsne',
        labeled_indices=labeled_indices,
        sampled_indices=sampled_indices,
        dataset_name=dataset_name,
        algo_name=algo_name,
    )


# def save_embeddings_to_file(embeddings: np.ndarray, labels: np.ndarray, dataset_name: str):
#     """
#     Saves embeddings and labels to a text file.
#
#     :param embeddings: NumPy array of shape (N, D)
#     :param labels: NumPy array of shape (N,)
#     :param file_path: Path to save the embeddings
#     """
#     assert embeddings.shape[0] == labels.shape[0], "Embeddings and labels must have same length"
#     embed_output_path = os.path.join(REPO_BASE, f"results/t-sne/{dataset_name}_t-sne_embeddings.npy")
#     label_output_path = os.path.join(REPO_BASE, f"results/t-sne/{dataset_name}_t-sne_labels.npy")
#     tsne = TSNE(n_components=2,
#                 perplexity=30,  # Experiment with values like 20, 30, 50
#                 learning_rate=200,  # Experiment with values like 100, 200, 500, 1000
#                 n_iter=1000,  # Number of iterations, 1000-5000 is common
#                 verbose=1,  # Show progress
#                 random_state=42,  # For reproducibility
#                 init='pca',  # Initialize with PCA results
#                 n_jobs=-1  # Use all available CPU cores
#                 )
#     tsne_results = tsne.fit_transform(embeddings)
#     # Save the embeddings
#     np.save(embed_output_path, tsne_results)
#     print(f"t-SNE embeddings saved to {embed_output_path}")
#
#     # Save the corresponding labels
#     np.save(label_output_path, labels)
#     print(f"Labels saved to {label_output_path}")
#
#
# if __name__ == "__main__":
#     # Example usage
#     all_features = ds_utils.load_features('CIFAR10', train=True)
#     # embeddings = np.random.rand(1000, 128)  # Example embeddings
#     labels = np.zeros_like(all_features)  # Example labels (10 classes)
#
#     # visualize_tsne(embeddings, labels, max_samples=500, include_indices=np.array([0, 1, 2]))
#     save_embeddings_to_file(all_features, labels, "CIFAR10")


# path = os.path.join(REPO_BASE, "output/CIFAR10/resnet18/CIFAR10_resnet18_2025_6_10_171959_564799/plot_episode_yvalues.npy")
#
# x = np.load(path)
# print(x)