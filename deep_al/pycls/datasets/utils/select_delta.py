"""
Delta selection for ProbCover.

WARNING: The automatic delta selection implemented here is a heuristic
estimation and is NOT proven to be theoretically optimal. It serves as
a practical starting point. Always test empirically with nearby delta
values (e.g. ±0.05–0.10) to verify that the selected value is suitable
for your dataset and task.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import matplotlib.pyplot as plt
from typing import Optional, Tuple

SEED = 0 # Constant seed for stability, otherwise delta might change between runs for the same embedding

def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device if available (and preferred), else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def choose_delta_grid() -> np.ndarray:
    """
    For future use
    """
    delta_values = np.arange(0, 2.55, 0.05)

    return delta_values

def compute_min_mismatch_distance(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 500,
    device: Optional[torch.device] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    Compute d_min[x] = min_{y: label[y] != label[x]} ||x - y||_2.
    If no mismatching y exists, d_min[x] = +inf.

    Then purity at delta is: mean(d_min >= delta)  (matches dist < delta rule).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)

    n = X.shape[0]
    dmin = torch.full((n,), float("inf"), device=device, dtype=torch.float32)

    with torch.no_grad():
        num_batches = (n + batch_size - 1) // batch_size
        batch_iter = range(num_batches)

        if debug:
            from tqdm import tqdm
            batch_iter = tqdm(batch_iter, desc="d_min batches", leave=False)

        for b in batch_iter:
            start = b * batch_size
            end = min((b + 1) * batch_size, n)

            cur = X[start:end]                       # [B, d]
            dist = torch.cdist(cur, X)               # [B, n]

            same = (labels_t[start:end].unsqueeze(1) == labels_t.unsqueeze(0))  # [B, n]
            dist = dist.masked_fill(same, float("inf"))

            dmin[start:end] = dist.min(dim=1).values

    return dmin

def purity_curve_from_dmin(dmin: torch.Tensor, delta_values: np.ndarray) -> np.ndarray:
    """
    purity(delta) = mean(d_min >= delta)
    (because impure iff exists mismatching y with dist < delta).
    """
    deltas = torch.as_tensor(
        np.asarray(delta_values),
        device=dmin.device,
        dtype=dmin.dtype,
    )
    purity = (dmin.unsqueeze(1) >= deltas.unsqueeze(0)).float().mean(dim=0)
    return purity.detach().cpu().numpy()

def _compute_curvature(delta: np.ndarray, purity: np.ndarray) -> np.ndarray:
    """Signed curvature kappa via central differences on a non-uniform grid."""
    n = delta.size
    y_prime = np.full(n, np.nan, dtype=float)
    y_second = np.full(n, np.nan, dtype=float)

    for i in range(1, n - 1):
        x_im1, x_i, x_ip1 = delta[i - 1], delta[i], delta[i + 1]
        y_im1, y_i, y_ip1 = purity[i - 1], purity[i], purity[i + 1]

        dx_total = x_ip1 - x_im1
        if dx_total == 0:
            continue

        y_prime[i] = (y_ip1 - y_im1) / dx_total

        dx_f = x_ip1 - x_i
        dx_b = x_i - x_im1
        if dx_f == 0 or dx_b == 0:
            continue

        y_second[i] = 2.0 * (((y_ip1 - y_i) / dx_f) - ((y_i - y_im1) / dx_b)) / dx_total

    y_prime[0], y_prime[-1] = y_prime[1], y_prime[-2]
    y_second[0], y_second[-1] = y_second[1], y_second[-2]

    kappa = y_second / np.power(1.0 + y_prime ** 2, 1.5)
    kappa[~np.isfinite(kappa)] = np.inf
    return kappa


def find_knee_point(
    delta: np.ndarray,
    purity: np.ndarray,
    debug: bool = False,
    debug_plot_path: str = "k_vs_delta_debug.png",
) -> Tuple[float, float]:
    """
    Knee point: delta where signed curvature is most negative (argmin kappa).
    Intended for raw-image neural network classifiers.
    """
    delta = np.asarray(delta, dtype=float)
    purity = np.asarray(purity, dtype=float)

    if delta.ndim != 1 or purity.ndim != 1 or delta.size != purity.size:
        raise ValueError("delta and purity must be 1D arrays of equal length")
    if delta.size < 3:
        raise ValueError("Need at least 3 points for curvature")

    if np.any(np.diff(delta) < 0):
        order = np.argsort(delta)
        delta, purity = delta[order], purity[order]

    kappa = _compute_curvature(delta, purity)

    if debug:
        plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(delta, kappa, linewidth=2)
        plt.ylabel(r"$\kappa$")
        plt.xlabel(r"$\delta$")
        plt.title(r"$\delta$ vs. $\kappa$ (knee)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(debug_plot_path, dpi=300)
        plt.close()

    idx_knee = int(np.argmin(kappa))
    return float(delta[idx_knee]), float(purity[idx_knee])


def find_zero_curvature_point(
    delta: np.ndarray,
    purity: np.ndarray,
    debug: bool = False,
    debug_plot_path: str = "k_vs_delta_debug.png",
) -> Tuple[float, float]:
    """
    Inflection point between the negative and positive curvature peaks.

    Algorithm (matches delta_episode_graphs.py):
      1. Find idx_neg = argmin(kappa)            — the knee (most negative curvature)
      2. Find idx_pos = next argmax after idx_neg — the recovery peak
      3. Find idx_zero = argmin(|kappa|) in the window (idx_neg, idx_pos)
         — the zero-crossing between the two peaks

    Intended for representation-space classifiers (1-NN, MLP, linear, etc.).
    """
    delta = np.asarray(delta, dtype=float)
    purity = np.asarray(purity, dtype=float)

    if delta.ndim != 1 or purity.ndim != 1 or delta.size != purity.size:
        raise ValueError("delta and purity must be 1D arrays of equal length")
    if delta.size < 3:
        raise ValueError("Need at least 3 points for curvature")

    if np.any(np.diff(delta) < 0):
        order = np.argsort(delta)
        delta, purity = delta[order], purity[order]

    kappa = _compute_curvature(delta, purity)

    idx_neg = int(np.argmin(kappa))
    # Next positive peak strictly after the knee
    tail = kappa[idx_neg + 1:]
    idx_pos = idx_neg + 1 + int(np.argmax(tail))

    if idx_pos - idx_neg <= 1:
        # No room for a zero-crossing — fall back to the knee itself
        idx_zero = idx_neg
    else:
        window = kappa[idx_neg + 1:idx_pos]
        idx_zero = idx_neg + 1 + int(np.argmin(np.abs(window)))

    if debug:
        plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(delta, kappa, linewidth=2)
        plt.axvline(delta[idx_neg],  linestyle="--", color="red",   label=f"neg peak ({delta[idx_neg]:.3f})")
        plt.axvline(delta[idx_zero], linestyle=":",  color="black",  label=f"zero ({delta[idx_zero]:.3f})")
        plt.axvline(delta[idx_pos],  linestyle="--", color="blue",  label=f"pos peak ({delta[idx_pos]:.3f})")
        plt.ylabel(r"$\kappa$")
        plt.xlabel(r"$\delta$")
        plt.title(r"$\delta$ vs. $\kappa$ (inflection)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(debug_plot_path, dpi=300)
        plt.close()

    return float(delta[idx_zero]), float(purity[idx_zero])

def kmeans(features: np.ndarray, num_clusters: int, seed: int = 1, debug: bool = False) -> np.ndarray:
    """
    KMeans for small K, MiniBatchKMeans for large K.
    Returns labels of shape (n_samples,).
    """
    if num_clusters <= 100:
        if debug:
            print(f"[kmeans] Using KMeans (k={num_clusters}, seed={seed})")
        km = KMeans(n_clusters=num_clusters, n_init="auto", random_state=seed)
        labels = km.fit_predict(features)
        return labels

    if debug:
        print(f"[kmeans] Using MiniBatchKMeans (k={num_clusters}, seed={seed}, batch_size=10000)")
    km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=10000, random_state=seed)
    labels = km.fit_predict(features)
    return labels

def select_delta(
    features: np.ndarray,
    num_classes: int,
    classifier_on_raw_images: bool = False,
    prefer_cuda: bool = True,
    debug: bool = False,
) -> float:
    """
    Select delta from the purity(delta) curve built on KMeans pseudo-labels.

    classifier_on_raw_images=True  → knee point (argmin kappa); suited for NNs
                                     trained on raw pixels.
    classifier_on_raw_images=False → inflection point (argmin |kappa|); suited
                                     for 1-NN / MLP / linear classifiers on the
                                     representation space.
    """
    print(
        "[select_delta] WARNING: automatic delta selection is a heuristic "
        "estimation and is NOT proven to be optimal. Always verify empirically "
        "with nearby delta values (e.g. ±0.05–0.10) before relying on this result."
    )

    if features.ndim != 2:
        raise ValueError(f"Features for delta calculation must be 2D, got shape {features.shape}")

    device = get_device(prefer_cuda)
    np.random.seed(SEED)

    if debug:
        mode = "knee (raw-image NN)" if classifier_on_raw_images else "inflection (representation-space)"
        print(f"[select_delta] mode={mode}, features: shape={features.shape}, dtype={features.dtype}, device={device}")

    pseudo_labels = kmeans(features, num_classes, seed=SEED, debug=debug)

    if debug:
        print(f"[select_delta] clustered into {num_classes} clusters for pseudo-labels")

    delta_values = choose_delta_grid()

    dmin = compute_min_mismatch_distance(
        features,
        pseudo_labels,
        batch_size=500,
        device=device,
        debug=debug,
    )

    purity_values = purity_curve_from_dmin(dmin, delta_values)

    if debug:
        for d, p in zip(delta_values, purity_values):
            print(f"delta: {d:.6f}, purity: {p:.6f}")

    if classifier_on_raw_images:
        x_sel, y_sel = find_knee_point(
            delta_values,
            purity_values,
            debug=debug,
            debug_plot_path="k_vs_delta.png",
        )
    else:
        x_sel, y_sel = find_zero_curvature_point(
            delta_values,
            purity_values,
            debug=debug,
            debug_plot_path="k_vs_delta.png",
        )

    if debug:
        # ProbCover's alpha threshold selection (reference only)
        alpha = 0.95
        eligible = [d for d, p in zip(delta_values, purity_values) if p >= alpha]
        old_delta = max(eligible) if len(eligible) > 0 else float(delta_values[0])

        plt.figure(figsize=(6, 4), dpi=150)
        plt.title("Purity vs. delta")
        plt.plot(delta_values, purity_values, "bo-", label="Purity", zorder=1)

        plt.axvline(
            x=old_delta,
            color="deepskyblue",
            linestyle="dashed",
            label=rf"Old Method $\delta$ = {old_delta:.2f}",
            zorder=2,
        )

        plt.axvline(
            x=x_sel,
            color="red",
            linestyle="dashed",
            label=rf"$\delta^*$ = {x_sel:.2f}",
            zorder=3,
        )

        plt.xlabel(r"$\delta$", fontsize=14)
        plt.ylabel(r"Purity", fontsize=14)
        plt.ylim(0, 1.05)
        plt.xlim(float(delta_values.min()), float(delta_values.max()))
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_name = "purity_vs_delta_with_delta_selection.png"
        plt.savefig(out_name, dpi=300)
        plt.close()

        print(f"Saved purity plot: {out_name}")
        print(
            f"[select_delta] selected delta={x_sel:.6f} (purity={y_sel:.6f}), "
            f"old delta={old_delta:.6f}"
        )
    return x_sel

def main() -> None:
    """
    Example usage
    """
    import os
    import sys

    from features import load_features

    DATASET = "CIFAR100"
    CLASSES = 100
    features_tag=None

    features = np.asarray(load_features(DATASET, features_tag=features_tag))
    selected_delta = select_delta(features, CLASSES, debug=True)
    print(f"Selected delta: {selected_delta:.6f}")

if __name__ == "__main__":
    main()
