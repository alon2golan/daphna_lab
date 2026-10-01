"""Paper graph construction; author's logarithmic kernel correction."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from .upstream import _apply_degree_normalization_sparse

GRAPH_VERSION = "vast-connected-union-rbf-log-v1"


def normalize_features(features):
    z = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if not np.isfinite(z).all() or np.any(norms == 0):
        raise ValueError("Graph requires finite, nonzero feature vectors")
    return np.ascontiguousarray(z / norms, dtype=np.float32)


def neighbor_indices(z, k, backend="faiss"):
    n = len(z)
    if backend == "dense":
        if n > 4096:
            raise ValueError("dense backend is a reference limited to 4096 samples; use faiss")
        distances = cdist(z, z, metric="sqeuclidean")
        np.fill_diagonal(distances, np.inf)
        return np.argsort(distances, axis=1, kind="stable")[:, :k]
    if backend != "faiss":
        raise ValueError("Unknown neighbor backend")
    # FAISS and PyTorch wheels can bundle incompatible OpenMP runtimes (macOS).
    # Isolate search rather than enabling unsafe KMP_DUPLICATE_LIB_OK.
    with tempfile.TemporaryDirectory(prefix="vast-neighbors-") as folder:
        features_path, result_path = Path(folder) / "features.npy", Path(folder) / "neighbors.npy"
        np.save(features_path, z)
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("_faiss_worker.py")),
                                 str(features_path), str(result_path), str(k)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("FAISS worker failed. Install faiss-cpu or faiss-gpu in this environment.\n"
                               + result.stderr[-4000:])
        return np.load(result_path, allow_pickle=False)


def union_graph(neighbors, k):
    n = len(neighbors)
    rows = np.repeat(np.arange(n), k)
    directed = sp.csr_matrix((np.ones(n * k, dtype=np.uint8),
                              (rows, neighbors[:, :k].ravel())), shape=(n, n))
    graph = directed.maximum(directed.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def build_kernel(features, backend="faiss", max_neighbors=512):
    z = normalize_features(features)
    n = len(z)
    if n < 2:
        raise ValueError("At least two training samples are required")
    cap = min(max_neighbors, n - 1)
    if cap < 1:
        raise ValueError("max_neighbors must be positive")
    width = min(16, cap)
    while True:
        neighbors = neighbor_indices(z, width, backend)
        if connected_components(union_graph(neighbors, width), directed=False, return_labels=False) == 1:
            break
        if width == cap:
            raise ValueError(f"Graph disconnected at k={cap}; increase max_neighbors explicitly")
        width = min(2 * width, cap)
    lo, hi = 1, width
    while lo < hi:
        mid = (lo + hi) // 2
        if connected_components(union_graph(neighbors, mid), directed=False, return_labels=False) == 1:
            hi = mid
        else:
            lo = mid + 1
    graph = union_graph(neighbors, lo).tocoo()
    weights = np.empty(graph.nnz, dtype=np.float64)
    for start in range(0, graph.nnz, 65536):
        end = min(start + 65536, graph.nnz)
        delta = z[graph.row[start:end]].astype(np.float64) - z[graph.col[start:end]]
        weights[start:end] = np.exp(-np.einsum("ij,ij->i", delta, delta))
    raw = sp.csr_matrix((weights, (graph.row, graph.col)), shape=(n, n))
    kernel = _apply_degree_normalization_sparse(raw, 1.0, "log")
    return kernel, {"k": lo, "directed_nonzeros": kernel.nnz,
                    "undirected_edges": kernel.nnz // 2, "backend": backend,
                    "graph_version": GRAPH_VERSION, "train_count": n}


def cached_kernel(features, backend, max_neighbors, cache_dir=None):
    if cache_dir is None:
        return build_kernel(features, backend, max_neighbors)
    z = normalize_features(features)
    digest = hashlib.sha256(memoryview(z))
    digest.update(f"{z.shape}:{backend}:{max_neighbors}:{GRAPH_VERSION}".encode())
    key = digest.hexdigest()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    matrix_path, meta_path = cache / f"{key}.npz", cache / f"{key}.json"
    if matrix_path.exists() and meta_path.exists():
        kernel = sp.load_npz(matrix_path)
        meta = json.loads(meta_path.read_text())
        if kernel.shape != (len(z), len(z)) or not np.isfinite(kernel.data).all():
            raise ValueError("Invalid graph cache; remove this cache entry")
        return kernel, {**meta, "cache_hit": True, "cache_key": key}
    # Use original features here: a second normalization of float32 z changes ties.
    kernel, meta = build_kernel(features, backend, max_neighbors)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as f:
        temp = Path(f.name)
    try:
        sp.save_npz(temp, kernel)
        temp.replace(matrix_path)
    finally:
        temp.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=cache, suffix=".json", delete=False) as f:
        json.dump(meta, f)
        temp = Path(f.name)
    temp.replace(meta_path)
    return kernel, {**meta, "cache_hit": False, "cache_key": key}
