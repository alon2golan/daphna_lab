"""Self-contained, per-sample propagation replay, with a fixed PCA layout."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds


def unit_rows(features):
    rows = np.asarray(features, dtype=np.float64)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def project_pca(features, fit_limit=10000):
    """Fit PCA on a deterministic sample; project every row in bounded batches."""
    n, width = features.shape
    indices = np.sort(np.random.RandomState(0).choice(n, min(n, fit_limit), replace=False))
    fit = unit_rows(features[indices])
    mean = fit.mean(axis=0)
    fit -= mean
    if min(fit.shape) <= 2 or not np.any(fit):
        _, _, components = np.linalg.svd(fit, full_matrices=False)
        components = components[:2]
    else:
        _, singular, components = svds(fit, k=2, which="LM", random_state=0)
        components = components[np.argsort(singular)[::-1]]
    # Choose signs deterministically, so equivalent fits do not mirror the plane.
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    xy = np.zeros((n, 2), dtype=np.float32)
    for start in range(0, n, 4096):
        # Only two dot products per row; avoid BLAS matmul warnings observed
        # with NumPy 2 / Accelerate on macOS, keeping the same PCA projection.
        centered = unit_rows(features[start:start + 4096]) - mean
        block = np.einsum("ij,kj->ik", centered, components, optimize=False)
        if not np.isfinite(block).all():
            raise ValueError("PCA projection produced nonfinite coordinates")
        xy[start:start + len(block), :block.shape[1]] = block
    return xy, {"method": "PCA", "geometry": "L2-normalized train features",
                "fit_count": len(indices), "fit_seed": 0, "fixed_across_iterations": True}


class PropagationTrace:
    def __init__(self, directory, features, sample_ids, labeled_mask, class_ids):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        xy, projection = project_pca(features)
        np.savez_compressed(self.directory / "samples.npz", xy=xy, sample_ids=sample_ids,
                            labeled=labeled_mask)
        self.metadata = {"version": 1, "sample_count": len(features), "class_ids": list(class_ids),
                         "projection": projection, "frames": [], "confirmed_reproduction": False,
                         "semantics": "Frame 0 is initialization. Later frames store sender/receiver masks before the update, and predictions/confidence/frozen masks after it. Roles can overlap."}
        self._save_metadata()

    def _save_metadata(self):
        temp = self.directory / "metadata.tmp"
        temp.write_text(json.dumps(self.metadata, indent=2) + "\n")
        temp.replace(self.directory / "metadata.json")

    def __call__(self, iteration, probabilities, senders, receivers, frozen, newly_frozen):
        filename = f"round_{iteration:06d}.npz"
        temp = self.directory / (filename + ".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, predicted=probabilities.argmax(axis=1).astype(np.int32),
                                confidence=probabilities.max(axis=1).astype(np.float32),
                                senders=senders, receivers=receivers, frozen=frozen,
                                newly_frozen=newly_frozen)
        temp.replace(self.directory / filename)
        self.metadata["frames"].append({"iteration": iteration, "file": filename})
        self._save_metadata()
