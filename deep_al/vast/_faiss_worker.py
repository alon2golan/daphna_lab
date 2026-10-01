"""Isolated exact FAISS search: do not import torch in this process."""
import os
import sys

import faiss
import numpy as np


def main():
    features_path, result_path, k = sys.argv[1:]
    k = int(k)
    threads = int(os.environ.get("VAST_FAISS_THREADS", min(os.cpu_count() or 1, 8)))
    if threads < 1:
        raise ValueError("VAST_FAISS_THREADS must be positive")
    faiss.omp_set_num_threads(threads)
    z = np.load(features_path, mmap_mode="r", allow_pickle=False)
    index = faiss.IndexFlatL2(z.shape[1])
    index.add(z)
    neighbors = np.empty((len(z), k), dtype=np.int64)
    for start in range(0, len(z), 1024):
        distances, indices = index.search(z[start:start + 1024], min(k + 1, len(z)))
        for offset, (d, ids) in enumerate(zip(distances, indices)):
            valid = (ids != start + offset) & (ids >= 0)
            ids, d = ids[valid], d[valid]
            order = np.lexsort((ids, d))
            neighbors[start + offset] = ids[order[:k]]
    np.save(result_path, neighbors)


if __name__ == "__main__":
    main()
