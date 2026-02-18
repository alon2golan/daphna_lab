import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import matplotlib.pyplot as plt
from typing import Optional, Tuple

def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device if available (and preferred), else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def choose_delta_grid() -> np.ndarray:
    """
    For future use
    """
    delta_values = np.arange(0, 1.05, 0.05)

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

def find_knee_point(
    delta: np.ndarray,
    purity: np.ndarray,
    debug: bool = False,
    debug_plot_path: str = "k_vs_delta_debug.png",
) -> Tuple[float, float]:
    """
    Knee point via signed curvature with central differences on a non-uniform grid.
    """
    delta = np.asarray(delta, dtype=float)
    purity = np.asarray(purity, dtype=float)

    if delta.ndim != 1 or purity.ndim != 1 or delta.size != purity.size:
        raise ValueError("delta and purity must be 1D arrays of equal length")
    n = delta.size
    if n < 3:
        raise ValueError("Need at least 3 points for curvature")

    if np.any(np.diff(delta) < 0):
        order = np.argsort(delta)
        delta = delta[order]
        purity = purity[order]

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

    if debug:
        plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(delta, kappa, linewidth=2)
        plt.ylabel(r"$\kappa$")
        plt.xlabel(r"$\delta$")
        plt.title(r"$\delta$ vs. $\kappa$")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(debug_plot_path, dpi=300)
        plt.close()

    idx_knee = int(np.argmin(kappa))
    return float(delta[idx_knee]), float(purity[idx_knee])

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
    seed: int = 1,
    prefer_cuda: bool = True,
    debug: bool = False,
) -> float:
    """
       Select delta via knee point on a purity(delta) curve.
       Pseudo-labels are obtained via KMeans.
   """

    if features.ndim != 2:
        raise ValueError(f"Features for delta calculation must be 2D, got shape {features.shape}")

    device = get_device(prefer_cuda)
    np.random.seed(seed)

    if debug:
        print(f"[select_delta] features: shape={features.shape}, dtype={features.dtype}, device={device}")

    pseudo_labels = kmeans(features, num_classes, seed=seed, debug=debug)

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

    x_knee, y_knee = find_knee_point(
        delta_values,
        purity_values,
        debug=debug,
        debug_plot_path="k_vs_delta.png",
    )

    if debug:
        # ProbCover's alpha threshold selection
        alpha = 0.95
        eligible = [d for d, p in zip(delta_values, purity_values) if p >= alpha]
        old_delta = max(eligible) if len(eligible) > 0 else float(delta_values[0])

        # Plot purity curve with both selections
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
            x=x_knee,
            color="red",
            linestyle="dashed",
            label=rf"$\delta^*$ = {x_knee:.2f}",
            zorder=3,
        )

        plt.xlabel(r"$\delta$", fontsize=14)
        plt.ylabel(r"Purity", fontsize=14)
        plt.ylim(0, 1.05)
        plt.xlim(float(delta_values.min()), float(delta_values.max()))
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_name = f"purity_vs_delta_with_delta_selection.png"
        plt.savefig(out_name, dpi=300)
        plt.close()

        print(f"Saved purity plot: {out_name}")
        print(
            f"[select_delta] knee delta={x_knee:.6f} (purity={y_knee:.6f}), "
            f"old delta={old_delta:.6f}"
        )
    return x_knee

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
