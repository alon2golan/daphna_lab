import sys
import os

# HACK: Remove the directory containing this script from sys.path to avoid shadowing the standard library 'logging' module
# which is imported by torch. This happens because there is a 'logging.py' in this directory.
# Probably a better way TODO this?
script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] == script_dir:
    sys.path.pop(0)

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

# @torch.compile() 
def rbf_fused_math(dists, gamma):
    return torch.exp(dists.pow(2) * gamma)

class RBFKernel(Kernel):
    def compute(self, x, y, sigma=1.0):
        return torch.exp(-torch.norm(x - y) ** 2 / (2 * (sigma ** 2)))
    
    def compute_gram(self, X, comp_device='cuda', sigma=1.0, batch_size=8192, final_type=torch.float16):
        return self.compute_sub_gram(X, X, comp_device=comp_device, sigma=sigma, batch_size=batch_size, final_type=final_type)
        
    def compute_sub_gram(self, x1, x2, comp_device='cuda', sigma=1.0, batch_size=8192, final_type=torch.float16):
        n1 = x1.shape[0]
        n2 = x2.shape[0]

        K = torch.empty((n1, n2), dtype=final_type)
        # Precompute the scaling factor for RBF
        gamma = -1.0 / (2 * (sigma ** 2))
        
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

@torch.compile()
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