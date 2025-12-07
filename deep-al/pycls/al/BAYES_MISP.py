import numpy as np
import pandas as pd
import torch
import gc
import pycls.datasets.utils as ds_utils
import os
from torch.profiler import profile, record_function, ProfilerActivity
from . import prior_selection
from pycls.utils.kernels import RBFKernel


class BAYES_MISP:
    def __init__(self, cfg, budgetSize, train_labels, lset, uset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        
        self.features = torch.from_numpy(ds_utils.load_features(self.ds_name, train=True, seed=self.seed, normalized=False))
        self.rel_indices = np.concatenate([lset, uset]).astype(int)
        self.features = self.features[self.rel_indices]
        self.alpha = self.cfg.ALPHA if 'ALPHA' in self.cfg else 0.0
        self.alpha_upper = self.cfg.ALPHA_UPPER if 'ALPHA_UPPER' in self.cfg else 0.1
        self.alpha_lower = self.cfg.ALPHA_LOWER if 'ALPHA_LOWER' in self.cfg else 0.001
        
        self.debug = self.cfg.DEBUG
        self.cont_method = self.cfg.CONT_METHOD if 'CONT_METHOD' in self.cfg else 'positive'

        self.budgetSize = budgetSize
        self.delta = delta # delta for topHat, sigma for RBF

        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff'
        kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if kernel_type == 'tophat':
            pass
            # self.kernel_fn = ('cuda') # TODO
        else:
            self.kernel_fn = RBFKernel()
            
        self.K = self.kernel_fn.compute_gram(self.features, sigma=self.delta, batch_size=8192, final_type=torch.float16)
        
        self.unique_labels = np.unique(train_labels)
        self.num_of_classes = self.unique_labels.size
        self.train_labels = np.array(train_labels)[self.rel_indices] # train_labels is now locally indexed
        
        self.init_C(lset)
                
        # Track labeled points information
        self.chosen_labels_num = torch.zeros(self.num_of_classes)
        self.activeSet = []
        self.coverages = [] # EXP
        self.coverages.append(self.get_point_coverage()) # EXP get initial coverage
        
    
    def get_priors(self):
        if self.alpha == 0.0:
            # Set sample specific prior
            # rel_measures = prior_selection.compute_reliability(self.K, self.train_labels, batch_size=8192, normalized=True)
            rel_measures = prior_selection.compute_clarity_kp(self.K, self.train_labels, self.num_of_classes, batch_size=8192)
            priors = prior_selection.get_temp_priors(rel_measures, lb=self.alpha_lower, ub=self.alpha_upper) # (N,)
        else:
            # Uniform prior of self.alpha
            priors = torch.tensor([self.alpha] * len(self.features))
        return priors
        
        
    def init_C(self, lset):
        """
        Init the main matrix C with the priors. If lset != empty, Init C accordingly.
        NOTE: Attention! float16
        """
        self.priors = self.get_priors() # This is (N,) tensor on CPU
        priors_tensor = torch.as_tensor(self.priors, device='cuda', dtype=torch.float16).unsqueeze(1) # (N, 1)
        self.C = priors_tensor.repeat(1, self.num_of_classes)
        
        if len(lset) > 0:
            print("Using a method which is not yet tested --- :(")
            self.load_C_from_lset(lset)
        
            
    def load_C_from_lset(self, lset):
        """
        Load matrix C from lset as if lset was labeled in the algorithm process.

        Assumes self.K is already computed.
        Assumes K is on CPU to save GPU memory.
        Assumes C is on GPU.
        """
        # Since self.features (and K) are sorted as [lset, uset], the indices of lset are simply 0..len(lset)-1
        lset_indices = np.arange(len(lset))
        lset_labels = np.array(self.train_labels)[lset_indices]
        
        for label in self.unique_labels:
            is_label = (lset_labels == label)
            indices_to_slice = lset_indices[is_label] 
            med_K = self.K[indices_to_slice].to('cuda')
            self.C[:, label] += torch.sum(med_K, axis=0)

            del med_K
        torch.cuda.empty_cache()
        
        
    def get_point_coverage(self): # EXP
        """
        Compute the coverage vector: max(C_i)/sum(C_i) for each point i.
        """
        sum_C = torch.sum(self.C, dim=1)
        max_C = torch.max(self.C, dim=1).values
        coverage = (max_C / sum_C).detach().cpu().numpy()
        return coverage
        

    def select_samples(self, lset, uset):

        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        # curr_l_set contains the lset with the local indices
        curr_l_set = np.where(np.isin(self.rel_indices, lset))[0].tolist()
        
        
        for i in range(self.budgetSize):
            
            if self.diff_method == 'margin':
                point_total_contribution = self.margin_old_proxy()
            elif self.diff_method == 'max':  ### old proxy with alphas vector without prior
                pass # FIXME
            elif self.diff_method == 'top2_weighted_max':
                vals, inds = torch.topk(self.C, k=2, dim=1)
                point_total_contribution = batched_diffs_weighted(self.K, self.C, vals, inds, diff_method="weighted_max", cont_method=self.cont_method)
            elif self.diff_method == 'full_weighted_max': ### the method with the expectation
                if len(self.K.shape) == 2:
                    self.K.unsqueeze_(2)
                point_total_contribution = expectation_proxy_max(self.K, self.C)
            elif self.diff_method in ['prob_cover', 'max_herding']:
                max_vals, indices = torch.max(self.C, dim=1)
                point_total_contribution = batched_diffs(self.K, max_vals, 0, self.num_of_classes,
                                                         diff_method="abs_diff")
            else:
                point_total_contribution = batched_diffs(self.K, self.C, diff_method=self.diff_method)
            # TODO: Make sure we don't even calculate contribution for selected points in the first place
            point_total_contribution[curr_l_set] = -np.inf
            # Note: This is for stability - argmax is weird sometimes.
            sampled_point = np.argsort(point_total_contribution.cpu().numpy(), kind='stable')[::-1][0].item()
            chosen_label = self.train_labels[sampled_point].item()

            self.chosen_labels_num[chosen_label] += 1

            if self.diff_method in ['prob_cover', 'max_herding']:
                self.C[:, chosen_label] = torch.maximum(self.C[:, chosen_label], self.K[sampled_point].to('cuda').squeeze())
            else:
                self.C[:, chosen_label] += self.K[sampled_point].squeeze().to('cuda')
                
            
            coverage = self.get_point_coverage() # EXP
            self.coverages.append(coverage) # EXP
        

            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)
            curr_l_set.append(sampled_point)

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        self.activeSet = selected
        
        # Convert activeSet to the actual indices in the dataset
        self.activeSet = self.rel_indices[np.array(self.activeSet)].astype(int)

        # Updating the general C matrix with changes after point selections
        remainSet = np.array(sorted(list(set(uset) - set(self.activeSet)))).astype(int)
        
        print(f'Finished the selection of {len(self.activeSet)} samples.')
        print(f'Active set is {self.activeSet}')
        
        # EXP - Write the coverage and contributions to the experiment directory
        if hasattr(self.cfg, 'EXP_DIR'):
            save_dir = self.cfg.EXP_DIR
            
            coverages_array = np.array(self.coverages)
            np.save(os.path.join(save_dir, 'coverages.npy'), coverages_array)
        else:
            raise ValueError("No EXP_DIR attribute in cfg provided for saving coverages.")
        # EXP

        return self.activeSet, remainSet


    def margin_old_proxy(self):
        """
        Calculates the acquisition score for all unlabeled points 'i' simultaneously
        using the vectorized form of the objective function (6) with the margin score (9):
        contribution(i) = sum over all j's of max{f(p^{{i}}[j]) - f(p^{L}[j]), 0}
        """
        # 1. Setup constants and dimension checks
        N = len(self.features)
        M = self.num_of_classes
        
        C_cpu = self.C.to('cpu', dtype=torch.float32)  # (N, M)

        # 2. Calculate f(p^L[j]) (old_margin) for all j
        # This is the margin of the normalized count vector p^L[j].
        # sum_C[j] = alpha*M + sum_{l in L} K_{lj}
        sum_C = torch.sum(C_cpu, dim=1, keepdim=True) # (N, 1)
        norm_C = C_cpu / sum_C                        # (N, M) -> p^L[j]
        
        # Find the margin (highest - second highest element) for all j
        vals = torch.topk(norm_C, k=2, dim=1).values   # (N, 2)
        old_margin = vals[:, 0] - vals[:, 1]           # (N,) vector -> f(p^L[j]) for all j

        # 3. Calculate f(p^{i}[j]) (Matrix F_i_j) for all i and j
        
        # Calculate the alpha*M term (indexed by j)
        # alpha_M_j[j] = M * alpha_j
        alpha_M_j = M * self.priors # (N,) vector

        # Calculate the denominator: alpha*M + K_ij for all (i, j) pairs
        # Broadcasting ensures alpha_M_j[j] is added to every K[i, j].
        # (N, N) matrix + (N,) vector = (N, N) matrix (broadcasting over rows/dim 0)
        denominator_i_j = alpha_M_j + self.K

        # Calculate F_i_j[i, j] = K_ij / denominator_i_j
        # This is f(p^{i}[j]) for all pairs (i, j)
        F_i_j = self.K / denominator_i_j # (N, N) matrix

        # 4. Apply max{F_i_j - old_margin, 0} and sum over j
        
        # Subtract old_margin[j] from every element in column j of F_i_j
        # This is the difference term R[i, j] = f(p^{i}[j]) - f(p^{L}[j])
        difference_matrix = F_i_j - old_margin # (N, N) matrix
        
        # Apply max{R[i, j], 0} element-wise
        max_term_matrix = torch.max(difference_matrix, torch.tensor(0.0)) # (N, N) matrix

        # Sum along dim=1 (the j dimension) to get the final score for each i.
        total_point_contribution = torch.sum(max_term_matrix, dim=1) # (N,) vector

        return total_point_contribution       
        
# ----------------------- Functions for Contribution Computation ----------------------- #

def expectation_proxy_max(K: torch.Tensor, C: torch.Tensor, chunk_size: int = 8192):
    
    N, _, _ = K.shape
    result = torch.empty((N, )).to(device=C.device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    norm_C = (C / sum_C)
    old_max = (max_C / sum_C)
    C_diff = (C - max_C).unsqueeze(0)
    num_iterations = int(N)
    max_C.unsqueeze_(0)
    
    for i in range(0, num_iterations, chunk_size):
            end = min(i + chunk_size, N)
            K_batched = K[i:end]
            K_batched = K_batched.to('cuda')
            weights_batched = norm_C[i:end]

            future_sum = K_batched + sum_C
            state_add = max_C + K_batched

            new_state_vec = torch.maximum(-K_batched, C_diff)

            new_state_vec.add_(state_add)
            new_state_vec.div_(future_sum)
            new_state_vec.sub_(old_max)

            new_state_vec.clamp_(min=0)
                
            result[i:end] = torch.einsum('ijk,ik->i', new_state_vec, weights_batched)
            del new_state_vec
            del K_batched
            del weights_batched

    return result


def batched_diffs_efficient_weighted_v2(K: torch.Tensor, C: torch.Tensor, chunk_size: int = 256, diff_method: str = "abs_diff", cont_method: int = 0):
    D, N, _ = K.shape
    results_list = []
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    norm_C = (C / sum_C)
    old_max = (max_C / sum_C)
    C_diff = (C - max_C).unsqueeze(0)
    num_iterations = int(N)
    cont_method = int(cont_method)
    max_C = max_C.unsqueeze(0)
    n_labels = C_diff.shape[-1]

    s_C_diff = C_diff * sum_C
    s_min_c1 = sum_C - max_C
    s_square = sum_C * sum_C

    for i in range(0, num_iterations, int(chunk_size)):
            end = min(i + chunk_size, D)
            K_batch = K[i:end]
            p1 = s_min_c1 * K_batch
            p2 = p1 + s_C_diff
            nom = s_square + sum_C * K_batch
            new_state_vec = p2 / nom
            if cont_method == 0:
                new_state_vec.clamp_(min=0)
            elif cont_method == 1:
                 torch.abs(new_state_vec, out=new_state_vec)

            weighted_point_diff = torch.einsum('ijk,ik->i',new_state_vec, norm_C[i:end])
            results_list.append(weighted_point_diff)
    result = torch.cat(results_list)
    return result

def batched_diffs(K, C, alpha, number_of_classes, chunk_size=1024, diff_method="abs_diff"):
    D, N = K.shape
    result = torch.empty(D).to(device=C.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if diff_method == "abs_diff":
            result[start:end] = torch.sum(torch.maximum(K[start:end] - C, torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        elif diff_method == "max":
            K_batched = K[start:end]
            K_batched = K_batched.to('cuda')
            result[start:end] = torch.sum(
                torch.maximum(((K_batched + alpha) / (torch.maximum( K_batched + alpha * number_of_classes, torch.full_like(K_batched, 1e-8)))) - C, torch.zeros_like(K_batched).to(device=C.device)), dim=1)
        elif diff_method == 'margin':
            result[start:end] = torch.sum(
                torch.maximum((K[start:end] / (K[start:end] + alpha * number_of_classes)) - C, torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    return result

def batched_diffs_weighted(K, C, vals, inds, chunk_size=1024, diff_method="abs_diff", cont_method="positive"):
    D, N = K.shape
    result = torch.empty((D, )).to(device=C.device)
    sum_C = torch.sum(C, axis=1)
    norm_C = (C / sum_C.unsqueeze(1))
    num_iterations = N
    C_max_diff = vals[:, 1] - vals[:, 0]
    partial_sum = torch.sum(vals, dim=1)
    weights = torch.gather(norm_C, 1, inds).unsqueeze(1)
    # old_max = (max_C.squeeze() / sum_C.squeeze())
    old_max = vals[:, 0] / partial_sum
    # timing each iteration
    for i in range(0, num_iterations, chunk_size):
        if diff_method == "weighted_max":
            end = i + chunk_size
            K_batched = K[i:end]
            K_batched = K_batched.to('cuda')
            weights_batched = norm_C[i:end, inds]
            future_sum = K_batched + partial_sum
            new_state_vec = torch.stack([torch.zeros_like(K_batched), torch.maximum(-K_batched, C_max_diff)], dim=0).to(device=C.device) + vals[:, 0] + K_batched
            cont_vec = (new_state_vec / future_sum) - old_max
            if cont_method == "positive":
                cont_vec.clamp_(min=0)
            elif cont_method == "abs":
                cont_vec = torch.abs(cont_vec)
            # weighted_point_diff = weights[i:end] @ cont_vec.permute(1, 0, 2)
            # result[i:end] = torch.nansum(weighted_point_diff, dim=2)
            result[i:end] = torch.einsum('ijk,jki->j', cont_vec, weights_batched)

            del new_state_vec
            del K_batched
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    return result