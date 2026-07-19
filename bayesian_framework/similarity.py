"""Matrix 2 of the bayesian framework: the figures matrix.

Takes the k-NN L2 distance matrix from distance.py (matrix 1) and turns each
distance into a figures via one of four kernels. Default is tophat: keep the
edge when distance <= delta, drop it otherwise. Operates purely on the saved
distances, so it is cheap to re-run with different kernels/knobs.

delta (tophat) / sigma (gaussian, laplacian) is the ONE semantic cut of the
pipeline. Matrix 1's k is only a capacity bound; when it is too small to hold a
point's whole delta-ball, that shows up as a silent second cut -- build() checks
for this and warns so it never happens unnoticed.
"""

import argparse
from pathlib import Path

import numpy as np

METHODS = ("tophat", "gaussian", "laplacian", "cosine")


def load_distance_matrix(npz_path):
    # read the k-NN distance matrix written by distance.py
    z = np.load(npz_path)
    # neighbor indices, their L2 distances, and how many neighbors per point
    return z["knn_idx"], z["knn_dist"], int(z["k"])


def similarity_from_distances(knn_dist, method="tophat", delta=None, sigma=None):
    # error out on an unknown kernel name
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    # work in float32; d has the same (N, k) shape as the input distances
    d = np.asarray(knn_dist, dtype=np.float32)

    if method == "tophat":
        # no delta given -> use the median neighbor distance as a sane default
        if delta is None:
            delta = float(np.median(d))
        # binary adjacency: 1 where the point is within delta, 0 beyond it
        sim = (d <= delta).astype(np.float32)
        params = {"method": method, "delta": delta}

    elif method == "gaussian":
        # RBF width sigma defaults to the median distance (median heuristic)
        if sigma is None:
            sigma = float(np.median(d))
        # standard gaussian kernel exp(-d^2 / (2 sigma^2)); sigma^2 is the variance
        sim = np.exp(-(d ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
        params = {"method": method, "sigma": sigma}

    elif method == "laplacian":
        if sigma is None:
            sigma = float(np.median(d))
        # standard laplacian kernel exp(-d / sigma); linear in d, so no sigma^2 here
        sim = np.exp(-(d / sigma)).astype(np.float32)
        params = {"method": method, "sigma": sigma}

    else:  # cosine
        # distances are L2 on unit-norm vectors, so d^2 = 2 - 2cos, i.e. cos = 1 - d^2/2
        sim = (1.0 - (d ** 2) / 2.0).astype(np.float32)
        params = {"method": method}

    return sim, params


def count_k_binding(knn_dist, params, tol=1e-2):
    # per point, the distance to its farthest STORED neighbor (the k-NN radius)
    last = np.asarray(knn_dist, dtype=np.float32)[:, -1]
    method = params["method"]
    if method == "tophat":
        # if the farthest stored neighbor is still inside delta, the delta-ball
        # spills past the k we kept -> those edges were dropped by matrix 1's k cut
        binding = last <= params["delta"]
    elif method in ("gaussian", "laplacian"):
        # if the kernel still has weight > tol at the k-NN radius, real neighbors
        # beyond k carry non-negligible figures that we silently threw away
        wlast, _ = similarity_from_distances(last[:, None], method, sigma=params["sigma"])
        binding = wlast[:, 0] > tol
    else:  # cosine has no cut -- k itself is the only support, nothing spills
        binding = np.zeros(last.shape[0], dtype=bool)
    return int(binding.sum())


def to_sparse_csr(knn_idx, knn_sim, n, symmetric=True):
    import scipy.sparse as sp

    # rebuild an N x N sparse figures matrix from (row, col, sim) triplets
    k = knn_idx.shape[1]
    # row index i repeated k times
    rows = np.repeat(np.arange(n), k)
    # the k neighbor columns of every point, flattened
    cols = knn_idx.reshape(-1)
    # the matching similarities, flattened
    data = knn_sim.reshape(-1).astype(np.float32)
    csr = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # tophat drops edges to 0; prune them so they aren't stored
    csr.eliminate_zeros()

    if symmetric:
        # k-NN is one-directional, so symmetrize by keeping the stronger figures
        # (for figures, larger = closer, unlike the distance matrix)
        csr = csr.maximum(csr.transpose().tocsr())
    return csr


def build(matrix1_path, method="tophat", delta=None, sigma=None,
          save=True, out_path=None, npy=False):
    # load matrix 1 (the k-NN distances)
    knn_idx, knn_dist, k = load_distance_matrix(matrix1_path)
    print(f"loaded matrix 1: knn_dist {knn_dist.shape} from {matrix1_path}", flush=True)

    # turn distances into similarities with the chosen kernel
    knn_sim, params = similarity_from_distances(knn_dist, method, delta, sigma)
    print(f"figures matrix ({method}) {knn_sim.shape} | params {params}", flush=True)

    # guard against the second, silent cut: is k big enough to hold each point's
    # neighborhood, or is matrix 1's k cut biting before our delta/sigma cut?
    n_bind = count_k_binding(knn_dist, params)
    if n_bind:
        print(f"WARNING: k is binding for {n_bind}/{knn_dist.shape[0]} points -- their "
              f"neighborhood reaches past the {k} stored neighbors, so some edges are "
              f"missing. Raise --k in matrix 1 (distance.py) or shrink delta/sigma.",
              flush=True)

    if save:
        if out_path:
            out = Path(out_path)
        else:
            # tag the filename with the cut value so different deltas/sigmas don't
            # overwrite each other: tophat -> d<delta>, gaussian/laplacian -> s<sigma>
            if "delta" in params:
                ptag = f"_d{params['delta']:.3g}"
            elif "sigma" in params:
                ptag = f"_s{params['sigma']:.3g}"
            else:  # cosine has no knob
                ptag = ""
            # sit next to matrix 1 (the framework's matrices/ folder), swap the
            # 'knn_dist' tag for 'knn_sim_<method><ptag>' so runs don't collide
            src = Path(matrix1_path)
            name = src.stem.replace("knn_dist", f"knn_sim_{method}{ptag}")
            out = src.with_name(name + ".npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        if npy:
            # save the two arrays as plain .npy files (sparse k-NN form: the sim
            # matrix is (N, k), NOT the dense N x N which would be terabytes)
            idx_path = out.with_name(out.stem + "_idx.npy")
            sim_path = out.with_name(out.stem + "_sim.npy")
            np.save(idx_path, knn_idx)
            np.save(sim_path, knn_sim)
            print(f"saved -> {idx_path}\n         {sim_path}", flush=True)
        else:
            # default: one .npz bundling neighbor indices, similarities, and params
            np.savez(out, knn_idx=knn_idx, knn_sim=knn_sim, **params)
            print(f"saved -> {out}", flush=True)

    return knn_idx, knn_sim, params


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--matrix1", required=True,
                   help="path to the k-NN distance .npz written by distance.py")
    p.add_argument("--method", default="tophat", choices=list(METHODS))
    p.add_argument("--delta", type=float, default=None,
                   help="tophat cutoff; default = median neighbor distance")
    p.add_argument("--sigma", type=float, default=None,
                   help="gaussian/laplacian width sigma; default = median neighbor distance")
    p.add_argument("--out_path", default=None,
                   help="where to save (default: next to matrix 1)")
    p.add_argument("--npy", action="store_true",
                   help="save two .npy files (<base>_idx.npy, <base>_sim.npy) instead of one .npz")
    args = p.parse_args()

    build(args.matrix1, method=args.method, delta=args.delta,
          sigma=args.sigma, out_path=args.out_path, npy=args.npy)


if __name__ == "__main__":
    main()
