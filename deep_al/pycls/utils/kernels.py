import sys
import os
import numpy as np

# HACK: Remove the directory containing this script from sys.path to avoid shadowing the standard library 'logging' module
# which is imported by torch. This happens because there is a 'logging.py' in this directory.
# Probably theres a better way TODO this
script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] == script_dir:
    sys.path.pop(0)

import faiss
import torch
import abc

# This file is for GPU optimized kernel function usage,
# computing gram matrices, and related operations.
# For scenarios where N is very big (>100k), look into using *KerOps library*

# ======== Abstract class kernel ======= #

class Kernel(abc.ABC):
    @abc.abstractmethod
    def compute(self, x, y):
        """Compute the kernel function between two points x and y."""
        pass

    @abc.abstractmethod
    def compute_gram(self, x):
        """Compute the full gram matrix for input X"""
        pass

    @abc.abstractmethod
    def compute_sub_gram(self, x1, x2):
        """
        Computes a submatrix of the gram matrix between inputs x1 and x2
        - Usually x1,x2 are subsets of X
        """
        pass

# ============= RBF Kernel ============= #

@torch.compile
def rbf_fused_math(dists, gamma):
    return torch.exp(dists.pow(2) * gamma)

class RBFKernel(Kernel):
    def _get_gamma(self, sigma):
        # Gamma is negative so we can just multiply: exp(dist_sq * gamma)
        return -1.0 / (2 * (sigma ** 2))

    def compute(self, x, y, sigma=1.0):
        return torch.exp(-torch.norm(x - y) ** 2 / (2 * (sigma ** 2)))

    def compute_gram(self, X, comp_device='cuda', sigma=1.0, batch_size=8192, final_type=torch.float16):
        return self.compute_sub_gram(X, X, comp_device=comp_device, sigma=sigma, batch_size=batch_size, final_type=final_type)

    def compute_sparse_gram(self, X, comp_device='cuda', sigma=1.0, final_type=torch.float16, k=1000):
        """
        Computes a sparse gram matrix using Faiss for NN search. (GPU)
        """
        N, D = X.shape
        gamma = self._get_gamma(sigma)

        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X.astype(np.float32)

        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(D)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index.add(X_np) # Load data to GPU index

        # distances_sq_np: (N, k) Array of L2 distances SQUARED (||x-y||^2)
        # indices_np:      (N, k) Array of integer indices of the neighbors
        distances_sq_np, indices_np = gpu_index.search(X_np, k)

        distances_sq = torch.from_numpy(distances_sq_np).to(comp_device)
        neighbor_indices = torch.from_numpy(indices_np).to(comp_device).flatten()

        values = torch.exp(distances_sq * gamma).flatten().to(dtype=final_type)
        crow_indices = torch.arange(0, (N + 1) * k, step=k, device=comp_device, dtype=torch.long)

        # Create sparse tensor
        K_sparse = torch.sparse_csr_tensor(
            crow_indices,
            neighbor_indices,
            values,
            size=(N, N),
            dtype=final_type,
            device=comp_device
        )

        # Cleanup faiss resources
        del gpu_index, res, distances_sq, neighbor_indices, values, crow_indices

        return K_sparse


    def compute_sub_gram(self, x1, x2, comp_device='cuda', sigma=1.0, batch_size=8192, final_type=torch.float16):
        n1 = x1.shape[0]
        n2 = x2.shape[0]

        K = torch.empty((n1, n2), dtype=final_type)
        # Precompute the scaling factor for RBF
        gamma = self._get_gamma(sigma)

        with torch.no_grad():
            x1_device = x1.to(comp_device)
            rounds = (n2 + batch_size - 1) // batch_size

            for i in range(rounds):
                start = i * batch_size
                end = min((i + 1) * batch_size, n2)

                x2_batch = x2[start:end].to(comp_device)
                dists = torch.cdist(x1_device, x2_batch, p=2.0)

                # In-place Optimization:
                rbf_vals = rbf_fused_math(dists, gamma)

                K[:, start:end] = rbf_vals.cpu().to(dtype=final_type)
                del x2_batch, dists
        del x1_device

        return K

# ============ TopHat Kernel ============= #

@torch.compile
def tophat_fused(dists, h, target_dtype):
    return (dists < h).to(dtype=target_dtype)

class TopHatKernel(Kernel):
    def compute(self, x, y, h=1.0):
        dist = torch.norm(x - y)
        return (dist < h).float()

    def compute_gram(self, X, comp_device='cuda', h=1.0, batch_size=8192, final_type=torch.float16):
        return self.compute_sub_gram(X, X, comp_device=comp_device, h=h, batch_size=batch_size, final_type=final_type)

    def compute_sub_gram(self, x1, x2, comp_device='cuda', h=1.0, batch_size=8192, final_type=torch.float16):
        n1 = x1.shape[0]
        n2 = x2.shape[0]

        K = torch.empty((n1, n2), dtype=final_type)

        with torch.no_grad():
            x1_device = x1.to(comp_device)
            rounds = (n2 + batch_size - 1) // batch_size

            for i in range(rounds):
                start = i * batch_size
                end = min((i + 1) * batch_size, n2)

                x2_batch = x2[start:end].to(comp_device)

                dists = torch.cdist(x1_device, x2_batch, p=2.0)

                K_batch = tophat_fused(dists, h, final_type)
                K[:, start:end] = K_batch.cpu()

                del x2_batch, dists, K_batch
        del x1_device

        return K