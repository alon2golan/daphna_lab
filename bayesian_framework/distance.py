"""Matrix 1 of the bayesian framework: the k-NN L2 distance matrix.

Reads the frozen, already-L2-normalized backbone features written by
solo-learn/extract_embeddings.py and, for every point, keeps its k nearest
neighbors and the L2 distance to them. The full N x N matrix is never built.

k here is a CAPACITY bound -- how many neighbors we can afford to store -- not a
semantic cut. The real cut (delta / kernel width) happens in matrix 2
(figures.py). Keep k generous so it never binds; figures.py warns if it
does, i.e. if a point's delta-ball reaches past its k stored neighbors.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def load_normalized_embeddings(embeddings_dir, split="train"):
    d = Path(embeddings_dir)

    # extract_embeddings.py saved a unit-norm copy as <split>_features_normalized.npy
    feats = np.load(d / f"{split}_features_normalized.npy").astype(np.float32)

    # length of every row; should be 1.0 because the file is the normalized copy
    norms = np.linalg.norm(feats, axis=1)
    # bail out loudly if it is not -- L2-on-normalized only holds when rows are unit norm
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            f"{split}_features_normalized.npy rows are not unit norm "
            f"(min={norms.min():.4f}, max={norms.max():.4f}); expected ~1.0"
        )

    # labels are optional; load them only when the file is there
    labels_path = d / f"{split}_labels.npy"
    labels = np.load(labels_path) if labels_path.exists() else None
    return feats, labels


def knn_distance_matrix(features, k=50, batch_size=1024, device=None,
                        dtype=torch.float32):
    # use the GPU when one is available, else fall back to CPU
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # move the features onto the compute device once, up front
    feats = torch.as_tensor(np.asarray(features), dtype=dtype, device=device)
    # N = number of points
    n = feats.shape[0]
    # each point has at most N-1 real neighbors, so k must be smaller than N
    if k >= n:
        raise ValueError(f"k={k} must be < number of points N={n}")

    # outputs: neighbor indices and their distances, one row per point
    idx_out = torch.empty(n, k, dtype=torch.int64)
    dist_out = torch.empty(n, k, dtype=torch.float32)

    # walk the points in row-blocks so we only ever hold a (batch_size x N) slab
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # distances from this block of points to every point -> shape (b, N)
        block = torch.cdist(feats[start:end], feats)

        # take the k+1 smallest per row (the extra one covers the point's own 0-distance self)
        d, i = torch.topk(block, k + 1, dim=1, largest=False)

        # the global index of each row in this block, as a column for broadcasting
        rows = torch.arange(start, end, device=device).unsqueeze(1)
        # mark, per row, which of the k+1 columns is the point itself
        self_mask = i == rows
        # a duplicate point could outrank self so self never appears; then just
        # drop the farthest (last) column instead so every row still loses one
        no_self = ~self_mask.any(dim=1)
        if no_self.any():
            self_mask[no_self, -1] = True
        # keep everything that is not the dropped column -> exactly k per row
        keep = ~self_mask

        # number of rows actually in this block
        b = end - start
        # drop the masked column and reshape back to (b, k)
        d = d[keep].view(b, k)
        i = i[keep].view(b, k)

        # copy this block's results back to CPU output arrays
        dist_out[start:end] = d.float().cpu()
        idx_out[start:end] = i.cpu()

    return idx_out.numpy(), dist_out.numpy()


def to_sparse_csr(knn_idx, knn_dist, n, symmetric=True):
    import scipy.sparse as sp

    # rebuild N x N sparse matrix from the (row, col, distance) triplets
    k = knn_idx.shape[1]
    # row index i repeated k times: i i i ... j j j ...
    rows = np.repeat(np.arange(n), k)
    # the k neighbor column indices of every point, flattened
    cols = knn_idx.reshape(-1)
    # the matching distances, flattened
    data = knn_dist.reshape(-1).astype(np.float32)
    csr = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    if symmetric:
        # the transpose has an edge j->i wherever the original has i->j
        csr_t = csr.transpose().tocsr()
        # True only where BOTH directions are present (product of two nonzeros)
        both = csr.multiply(csr_t).astype(bool)
        # union of edges, taking the larger distance where they overlap
        union = csr.maximum(csr_t)
        # the smaller distance where they overlap
        smaller = csr.minimum(csr_t)
        # on the overlap, swap the larger value for the smaller one
        csr = (union - union.multiply(both) + smaller.multiply(both)).tocsr()
        # a genuine 0 would read as "absent" in CSR, so prune stored zeros
        csr.eliminate_zeros()
    return csr


def build(embeddings_dir, split="train", k=50, batch_size=1024, device=None,
          save=True, out_path=None):
    # load the normalized features (labels come along if present)
    feats, labels = load_normalized_embeddings(embeddings_dir, split)
    print(f"loaded {split}: features {feats.shape} from {embeddings_dir}", flush=True)

    # compute the k nearest neighbors and their L2 distances
    knn_idx, knn_dist = knn_distance_matrix(
        feats, k=k, batch_size=batch_size, device=device
    )
    print(f"k-NN distance matrix: knn_idx {knn_idx.shape}, knn_dist {knn_dist.shape}",
          flush=True)

    if save:
        if out_path:
            out = Path(out_path)
        else:
            # default: a matrices/ folder inside this framework dir (NOT next to
            # the solo-learn embeddings -- those are just an input to us)
            out_dir = Path(__file__).resolve().parent / "matrices"
            out_dir.mkdir(parents=True, exist_ok=True)
            # tag the file with the embeddings it came from (e.g. simclr_cifar10_1000ep)
            # so different sources don't overwrite each other in the shared folder
            tag = "_".join(Path(embeddings_dir).resolve().parts[-3:])
            out = out_dir / f"{tag}_{split}_knn_dist_k{k}.npz"
        # make sure the target's folder exists even for a custom out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, knn_idx=knn_idx, knn_dist=knn_dist, k=k)
        print(f"saved -> {out}", flush=True)

    return knn_idx, knn_dist


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--embeddings_dir", required=True,
                   help="dir with <split>_features_normalized.npy (from extract_embeddings.py)")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--k", type=int, default=50, help="neighbors to keep per point")
    p.add_argument("--batch_size", type=int, default=1024,
                   help="rows per distance block (memory knob)")
    p.add_argument("--out_path", default=None,
                   help="where to save the .npz (default: next to the embeddings)")
    args = p.parse_args()

    build(args.embeddings_dir, split=args.split, k=args.k,
          batch_size=args.batch_size, out_path=args.out_path)


if __name__ == "__main__":
    main()
