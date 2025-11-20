import numpy as np
import pandas as pd
import torch
import gc
import pycls.datasets.utils as ds_utils
from tools.utils import visualize_tsne
import time
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
###MISP = maximum importance sampling points
torch.cuda.empty_cache()


class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, sigma=1.0, batch_size=8192, final_type=torch.float32):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device)
        K = []
        rounds = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        x1_sq = (x1 * x1).sum(dim=-1, keepdim=True)     # (1, n1, 1)
        for i in range(rounds):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            x2_sq = (x2_subset * x2_subset).sum(dim=-1, keepdim=True)  # (1, b, 1)
            x2_sq = x2_sq.transpose(1, 2)  # → (1, 1, b)   <-- only transpose x2
            # dot product
            xy = x1 @ x2_subset.transpose(1, 2)
            dist2 = x1_sq + x2_sq - 2.0 * xy
            dist2 = torch.clamp(dist2, min=0)
            # Using in-place operation to save memory
            K_batch = torch.exp_(-dist2 / (2 * (sigma ** 2))) 

            K.append(K_batch.squeeze(0).cpu().to(dtype=final_type))
            del x2_subset, x2_sq, xy, dist2, K_batch

        K = torch.cat(K, dim=-1)
        return K
    

class TopHatKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device) # 1 x n x d, 1 x n' x d
        dist_matrix = []
        batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset)
            dist = (dist < h).to(dtype=torch.float16)
            dist_matrix.append(dist.cpu())
            del dist
        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        # k = (dist_matrix < h).to(dtype=torch.float16)
        return dist_matrix

# ----------------------- Main Logic ----------------------- #

class BAYES_MISP:
    def __init__(self, cfg, budgetSize, train_labels, lset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        
        self.all_features = ds_utils.load_features(self.ds_name, train=True)
        self.alpha = self.cfg.ALPHA if self.diff_method not in ['prob_cover', 'max_herding'] else 0
        self.debug = self.cfg.DEBUG
        self.cont_method = self.cfg.CONT_METHOD if 'CONT_METHOD' in self.cfg else 'positive'

        self.budgetSize = budgetSize
        self.delta = delta # delta for topHat, sigma for RBF

        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff'
        kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
        else:
            self.kernel_fn = RBFKernel('cuda')

        self.train_labels_general = np.array(train_labels)
        self.unique_labels = np.unique(self.train_labels_general)
        self.num_of_classes = self.unique_labels.size
        # Attention! float16 
        self.C_general = torch.full((self.all_features.shape[0], self.num_of_classes), self.alpha, device='cuda', dtype=torch.float16)
        # Track labeled points information - to CUDA??
        self.chosen_labels_num = torch.zeros(self.num_of_classes).to('cuda')
        torch.cuda.empty_cache()


    def load_C_from_lset(self):
        """
        Load matrix C as if lset was labeled in the algorithm process.
        -- Assumes self.K is already computed --
        """
        class_indices = {label: np.where(self.train_labels_general[self.lSet.astype(int)] == label)[0] for label in self.unique_labels}
        for label in self.unique_labels:
            K_lset_label = self.K[class_indices[label]]  # K subset for already labeled points with current label
            self.C[:, label] += torch.sum(K_lset_label, axis=0)
    

    def init_sampling_loop(self, lset, uset):
        self.lSet = lset
        self.uSet = uset
        torch.cuda.empty_cache()
        self.set_rel_features()
        self.activeSet = []
        self.K = self.kernel_fn.compute_kernel(
            self.rel_features, self.rel_features, self.delta)
        self.C = self.C_general[self.relevant_indices_order].to('cuda') # C stays on the GPU throught the process?
        self.train_labels = self.train_labels_general[self.relevant_indices_order]
        # TODO: Load C from lset should be here. Important.


    def set_rel_features(self):
        """
        Reorganizing the features to [lset and then uset].
        """
        self.relevant_indices_order = np.concatenate([self.lSet, self.uSet]).astype(int)
        if isinstance(self.all_features, torch.Tensor):
            self.rel_features = self.all_features[self.relevant_indices_order]
        elif isinstance(self.all_features, np.ndarray):
            self.rel_features = torch.from_numpy(self.all_features[self.relevant_indices_order])


    def select_samples(self, lset, uset):
        
        self.lSet = lset
        self.uSet = uset
        self.set_rel_features()
        self.C = self.C_general[self.relevant_indices_order].to('cuda') # C stays on the GPU throught the process?
        self.train_labels = self.train_labels_general[self.relevant_indices_order]
        self.K = self.K[self.relevant_indices_order, :][:, self.relevant_indices_order]

        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        curr_l_set = list(range(len(lset)))
        for i in range(self.budgetSize):
            
            if self.diff_method == 'margin':
                vals, inds = torch.topk(self.C, k=2, dim=1)
                old_margin = vals[:, 0] - vals[:, 1]
                point_total_contribution = batched_diffs(self.K, old_margin, self.alpha, self.num_of_classes, diff_method="margin")
            elif self.diff_method == 'max':  ### old proxy with alphas vector without prior
                max_vals, indices = torch.max(self.C, dim=1) # TODO
                point_total_contribution = batched_diffs(self.K, max_vals, self.alpha, self.num_of_classes, diff_method="max")
            elif self.diff_method == 'top2_weighted_max':
                vals, inds = torch.topk(self.C, k=2, dim=1)
                point_total_contribution = batched_diffs_weighted(self.K, self.C, vals, inds, diff_method="weighted_max", cont_method=self.cont_method)
            elif self.diff_method == 'full_weighted_max': ### the method with the excepectation
                if len(self.K.shape) == 2:
                    self.K.unsqueeze_(2)
                class_corr = (self.C.T - self.alpha) @ (self.C -self.alpha)
                points_intres_class = (self.C - self.alpha) @ class_corr
                point_total_contribution = batched_diffs_efficient_weighted(self.K, self.C,
                                                          diff_method="efficient_full_weighted_max",cont_method=self.cont_method, class_corr=points_intres_class)
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

            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)
            curr_l_set.append(sampled_point)

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices_order[selected]

        # Updating the general C matrix with changes after point selections
        self.C_general[self.relevant_indices_order] = self.C
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        del self.C

        return activeSet, remainSet

    def plot_tsne(self):
        labeled_indices = np.array(self.lSet).astype(int)
        sampled_indices = np.array(self.activeSet).astype(int)
        visualize_tsne(labeled_indices, sampled_indices, algo_name='MISP')

# ----------------------- Functions for Contribution Computation ----------------------- #

def batched_diffs_efficient_weighted(K: torch.Tensor, C: torch.Tensor, chunk_size: int = 1024, diff_method: str = "abs_diff", cont_method: str = "positive", class_corr=None):
    D, N, _ = K.shape
    result = torch.empty((D, )).to(device=C.device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    norm_C = (C / sum_C)
    old_max = (max_C / sum_C)
    C_diff = (C - max_C).unsqueeze(0)
    num_iterations = int(N)
    cont_method = cont_method
    max_C.unsqueeze_(0)
    class_corr = class_corr.unsqueeze(1).to(torch.bool)
    for i in range(0, num_iterations, int(chunk_size)):
            end = min(i + chunk_size, D)
            K_batched = K[i:end]
            K_batched = K_batched.to('cuda')
            weights_batched = norm_C[i:end]


            future_sum = K_batched + sum_C
            state_add = max_C + K_batched

            new_state_vec = torch.maximum(-K_batched, C_diff)

            new_state_vec.add_(state_add)
            new_state_vec.div_(future_sum)
            new_state_vec.sub_(old_max)

            if cont_method == "positive": ### regular method
                new_state_vec.clamp_(min=0)
            elif cont_method == 'abs': ### take all contribution
                 torch.abs(new_state_vec, out=new_state_vec)
            elif cont_method == "fusion":
                 class_corr_batched = class_corr[i:end]
                 is_neg = new_state_vec < 0
                 new_state_vec[is_neg & ~class_corr_batched] = 0
                 new_state_vec[is_neg & class_corr_batched] *= -1
            elif cont_method == "reg_sum_postive": ## take average contribution (not weighted by the prior)
                 new_state_vec.clamp_(min=0)
                 result[i:end] = torch.sum(new_state_vec, dim=(1, 2))

                 del new_state_vec
                 del K_batched
                 del weights_batched

                 continue
            elif cont_method == "reg_sum_abs":
                 torch.abs(new_state_vec, out=new_state_vec)
                 result[i:end] = torch.sum(new_state_vec, dim=(1, 2))

                 del new_state_vec
                 del K_batched
                 del weights_batched

                 continue

            result[i:end] = torch.bmm(new_state_vec, weights_batched.unsqueeze_(2)).sum(dim=1).squeeze(1)
            del new_state_vec
            del K_batched
            del weights_batched
            # result[i:end] = torch.einsum('ijk,ik->i',new_state_vec, weights_batched)
            # res = new_state_vec * weights_batched

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

