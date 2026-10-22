import numpy as np
import torch
import scipy.sparse as sp
import deep_al.pycls.datasets.utils as ds_utils
from deep_al.pycls.al.kernel_utils import (
    build_K_general_matrix,
    update_C_with_label_connections,
    RBFKernel,
    TopHatKernel,
)
from deep_al.pycls.al import prior_selection

torch.cuda.empty_cache()


class BAYES_MISP:
    """
    Bayesian Maximum Importance Sampling Points (MISP): greedy sample selection
    that maintains a per-point, per-class evidence matrix C, accumulated from a
    kernel similarity matrix K over already-chosen points, and repeatedly picks
    the point whose addition would change C the most (by one of several
    calc_method/cont_method/weight_method scoring rules below).
    """

    def __init__(self, cfg, budgetSize, train_labels, lset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        features_tag = self.cfg['DATASET'].get('FEATURES_TAG', None)
        self.all_features = ds_utils.load_features(self.ds_name, train=True, features_tag=features_tag)

        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff'
        self.alpha = self.cfg.ALPHA if self.diff_method not in ['prob_cover', 'max_herding'] else 0
        self.debug = self.cfg.DEBUG
        self.use_sparse = getattr(self.cfg, 'SPARSE_K', False)
        mat_type = getattr(self.cfg, 'MATRIX_DTYPE', None)
        self.matrices_type = getattr(torch, mat_type) if mat_type else (torch.float32 if self.use_sparse else torch.float16)
        self.cont_method = self.cfg.CONT_METHOD if 'CONT_METHOD' in self.cfg else 'positive'
        self.distribution_cont_weight_method = getattr(self.cfg, 'DISTRIBUTION_CONT_WEIGHT_METHOD', 'weighted')
        self.c_normalization = getattr(self.cfg, 'C_NORMALIZATION', 'sum')
        self.calc_method = getattr(self.cfg, 'CALC_METHOD', 'max')
        self.class_weighting_method = getattr(self.cfg, 'CLASS_WEIGHTING_METHOD', 'none')

        self.budgetSize = budgetSize
        self.K_sparsity_threshold = getattr(self.cfg, 'K_SPARSITY_THRESHOLD', 0.0)
        self.sigma = cfg.ACTIVE_LEARNING.INITIAL_SIGMA if 'INITIAL_SIGMA' in cfg.ACTIVE_LEARNING else 1.0
        self.update_K_matrix = getattr(self.cfg, 'UPDATE_K_MATRIX', False)

        self.delta = delta

        self.train_labels_general = np.array(train_labels)
        unique_labels = np.unique(self.train_labels_general)
        self.unique_labels = unique_labels
        self.num_of_classes = unique_labels.size

        self.local_alpha = getattr(self.cfg, 'LOCAL_ALPHA', False)
        self.local_alpha_oracle_method = getattr(self.cfg, 'LOCAL_ALPHA_ORACLE_METHOD', 'entropy')
        self.alpha_lower = getattr(self.cfg, 'ALPHA_LOWER_BOUND', 0.01)
        self.alpha_upper = getattr(self.cfg, 'ALPHA_UPPER_BOUND', 1.0)

        self.kernel_build_batch_size = getattr(self.cfg, 'KERNEL_BUILD_BATCH_SIZE', 1024)

        self.chosen_labels_num = torch.zeros(self.num_of_classes).to('cuda')
        self.cum_labels_info = torch.zeros(self.num_of_classes).to('cuda')
        self.labeled_points_mask_general = torch.zeros(self.all_features.shape[0], dtype=torch.bool).to('cuda')

        self.kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if self.kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
            initial_threshold = self.delta
        else:
            self.kernel_fn = RBFKernel('cuda')
            initial_threshold = self.K_sparsity_threshold

        self.K_general = build_K_general_matrix(
            self.all_features,
            threshold=initial_threshold,
            use_sparse=self.use_sparse,
            kernel_type=self.kernel_type,
            delta=self.delta,
            sigma=self.sigma,
            kernel_build_batch_size=self.kernel_build_batch_size,
            matrices_type=self.matrices_type,
            kernel_fn=self.kernel_fn,
            zero_indices=None,
        )

        self.total_connections_chosen = 0

        if self.local_alpha:
            self.init_C(lset, self.K_general, self.local_alpha_oracle_method)
        else:
            self.C_general = torch.full((self.all_features.shape[0], unique_labels.size), self.alpha, device='cuda',
                                        dtype=self.matrices_type)

        total_points = self.all_features.shape[0]
        if sp.issparse(self.K_general):
            kernel_sum = float((self.K_general > 0).sum())
        elif torch.is_tensor(self.K_general):
            kernel_sum = float((self.K_general > 0).sum().item())
        else:
            kernel_sum = float(np.sum((self.K_general > 0)))
        total_pairs = total_points ** 2
        sparsity_ratio = 1 - (kernel_sum / total_pairs)
        # General sparsity ratio scaled to the full K size, minus self-connections,
        # halved because the matrix is symmetric.
        self.initial_sparse_index = max(
            (int(sparsity_ratio * total_pairs) - total_points) // 2,
            0
        )

        if lset is not None and lset.size > 0 and not self.use_sparse:
            temp_K = self.K_general
            class_indices = {label: np.where(self.train_labels_general[lset.astype(int)] == label)[0] for label in unique_labels}

            for label in unique_labels:
                label_positions = class_indices[label]
                if label_positions.size == 0:
                    continue
                curr_labels_sim = temp_K[label_positions]
                self.C_general[:, label] = torch.max(curr_labels_sim, axis=0).values.to(device='cuda')
                del curr_labels_sim
            del temp_K, class_indices
        torch.cuda.empty_cache()

    def _extract_sparse_submatrix(self, K_csr, indices):
        """
        Efficiently extract a symmetric submatrix K[indices, :][:, indices] from a CSR matrix
        via a COO intermediate, avoiding the slow/sparse-efficiency-warning-prone
        double CSR fancy indexing.
        """
        indices = np.asarray(indices, dtype=np.int64)
        n_new = len(indices)

        old_to_new = np.full(K_csr.shape[0], -1, dtype=np.int64)
        old_to_new[indices] = np.arange(n_new)

        K_coo = K_csr.tocoo()
        row_new = old_to_new[K_coo.row]
        col_new = old_to_new[K_coo.col]

        keep_mask = (row_new >= 0) & (col_new >= 0)
        if not keep_mask.any():
            return sp.csr_matrix((n_new, n_new), dtype=K_csr.dtype)

        new_coo = sp.coo_matrix(
            (K_coo.data[keep_mask], (row_new[keep_mask], col_new[keep_mask])),
            shape=(n_new, n_new),
            dtype=K_csr.dtype
        )
        return new_coo.tocsr()

    def get_priors(self, K, oracle_method):
        if sp.issparse(K):
            K_csr = K.tocsr().astype(np.float32, copy=False)
            n_points = K_csr.shape[0]
            y_indices = self.train_labels_general.astype(np.int64)
            one_hot = np.zeros((n_points, self.num_of_classes), dtype=np.float32)
            one_hot[np.arange(n_points), y_indices] = 1.0

            class_accum = K_csr @ one_hot
            class_accum += 0.1
            row_sums = class_accum.sum(axis=1, keepdims=True)
            np.maximum(row_sums, 1e-12, out=row_sums)
            norm_c = class_accum / row_sums

            if oracle_method == 'entropy':
                entropy = -np.sum(norm_c * np.log(norm_c + 1e-9), axis=1)
                clarity = 1.0 - entropy / np.log(self.num_of_classes)
            elif oracle_method == 'max':
                clarity = np.max(norm_c, axis=1)
            else:
                raise ValueError(f"Unknown oracle method: {oracle_method}")

            rel_measures = torch.from_numpy(clarity.astype(np.float32))
        else:
            rel_measures = prior_selection.compute_clarity_kp(K, self.train_labels_general, self.num_of_classes, oracle_method, batch_size=8192)

        priors = prior_selection.get_temp_priors(rel_measures, lb=self.alpha_lower, ub=self.alpha_upper)  # (N,)
        return priors

    def init_C(self, lset, K, oracle_method):
        """Init the main matrix C with per-point priors derived from K's class clarity."""
        self.priors = self.get_priors(K, oracle_method)  # (N,) tensor on CPU
        priors_tensor = torch.as_tensor(self.priors, device='cuda', dtype=torch.float16).unsqueeze(1)  # (N, 1)
        self.C_general = priors_tensor.repeat(1, self.num_of_classes).to(dtype=self.matrices_type)

    def init_sampling_loop(self, lset, uset):
        torch.cuda.empty_cache()
        self.set_rel_features(lset, uset)
        self.activeSet = []
        if self.use_sparse:
            K_csr_shuffled = self._extract_sparse_submatrix(self.K_general, self.relevant_indices)
            crow_indices = torch.from_numpy(K_csr_shuffled.indptr).to(torch.int64)
            col_indices = torch.from_numpy(K_csr_shuffled.indices).to(torch.int64)
            values = torch.from_numpy(K_csr_shuffled.data).to(torch.float32)

            self.K = torch.sparse_csr_tensor(
                crow_indices=crow_indices,
                col_indices=col_indices,
                values=values,
                size=K_csr_shuffled.shape,
                dtype=values.dtype
            )
            del K_csr_shuffled, values, col_indices, crow_indices
        else:
            self.K = self.K_general[self.relevant_indices, :][:, self.relevant_indices]

        self.C = self.C_general[self.relevant_indices].to('cuda')
        self.train_labels = self.train_labels_general[self.relevant_indices]
        self.labeled_points_mask = self.labeled_points_mask_general[self.relevant_indices]

    def set_rel_features(self, lset, uset):
        self.lSet = lset
        self.uSet = uset
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        if isinstance(self.all_features, torch.Tensor):
            self.rel_features = self.all_features[self.relevant_indices]
        elif isinstance(self.all_features, np.ndarray):
            self.rel_features = torch.from_numpy(self.all_features[self.relevant_indices])

    def update_K_general_matrix(self, lset):
        torch.cuda.empty_cache()
        if self.kernel_type == 'tophat':
            old_threshold = self.delta
            rebuild_threshold = self.delta
        else:
            old_threshold = self.K_sparsity_threshold
            rebuild_threshold = self.K_sparsity_threshold
        zero_indices = np.asarray(lset, dtype=np.int64)
        self.K_general = build_K_general_matrix(
            self.all_features,
            threshold=rebuild_threshold,
            use_sparse=self.use_sparse,
            kernel_type=self.kernel_type,
            delta=self.delta,
            sigma=self.sigma,
            kernel_build_batch_size=self.kernel_build_batch_size,
            matrices_type=self.matrices_type,
            kernel_fn=self.kernel_fn,
            zero_indices=zero_indices,
            prev_threshold=old_threshold,
            C_general=self.C_general,
            train_labels_general=self.train_labels_general,
            labeled_points_mask_general=self.labeled_points_mask_general,
            diff_method=self.diff_method,
            cum_labels_info=self.cum_labels_info,
        )

    def select_samples(self, lset, uset):
        """
        Greedy selection: iteratively pick the unlabeled point whose kernel row
        would most increase the evidence matrix C for its (predicted) class.
        """
        if len(lset) > 0 and len(lset) % 10000 == 0 and self.use_sparse and self.update_K_matrix:
            self.update_K_general_matrix(lset)

        self.init_sampling_loop(lset, uset)

        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        for i in range(self.budgetSize):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)

            class_gains_weights = self.get_class_gains_weighting_vector()

            if self.diff_method == 'max':
                C_sum = torch.sum(self.C, dim=1, keepdim=True)
                norm_C = self.C / C_sum
                max_vals, indices = torch.max(norm_C, dim=1)
                point_total_contribution = batched_diffs_sparse_max(self.K, max_vals, self.alpha, self.num_of_classes)
            elif self.use_sparse:
                point_total_contribution = batched_diffs_efficient_weighted_sparse(
                    self.K, self.C, cont_method=self.cont_method,
                    weight_method=self.distribution_cont_weight_method,
                    class_gains_weights=class_gains_weights,
                    c_normalization=self.c_normalization, calc_method=self.calc_method)
            else:
                if len(self.K.shape) == 2:
                    self.K.unsqueeze_(2)
                point_total_contribution = batched_diffs_efficient_weighted(
                    self.K, self.C, diff_method="efficient_full_weighted_max",
                    cont_method=self.cont_method, weight_method=self.distribution_cont_weight_method,
                    c_normalization=self.c_normalization)

            point_total_contribution[curr_l_set] = -np.inf

            sampled_point = np.argsort(point_total_contribution.cpu().numpy(), kind='stable')[::-1][0].item()
            chosen_label = self.train_labels[sampled_point].item()

            self.chosen_labels_num[chosen_label] += 1

            K_row_dense = self.K[sampled_point].to_dense().to('cuda').squeeze()

            self.labeled_points_mask[sampled_point] = True
            self.C[sampled_point, :] = torch.zeros(self.num_of_classes).to('cuda')
            self.C[sampled_point, chosen_label] = 1.0
            self.C[~self.labeled_points_mask, chosen_label] += K_row_dense[~self.labeled_points_mask]

            self.cum_labels_info[chosen_label] += K_row_dense.sum()
            self.total_connections_chosen += torch.sum(K_row_dense > 0).item()

            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)
            del K_row_dense

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]

        self.C_general[self.relevant_indices] = self.C
        self.labeled_points_mask_general[self.relevant_indices] = self.labeled_points_mask
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')

        del self.K
        del self.C

        return activeSet, remainSet

    def get_class_gains_weighting_vector(self):
        if self.class_weighting_method == 'none':
            return None
        elif self.class_weighting_method == 'linear':
            norm_class_gains_vector = self.cum_labels_info / (torch.sum(self.cum_labels_info) + 1e-8)
            return 1 / (norm_class_gains_vector + 1)
        elif self.class_weighting_method == 'square':
            square_class_gains_vector = torch.sqrt(self.cum_labels_info + 1) - 1
            norm_class_gains_vector = square_class_gains_vector / (torch.sum(square_class_gains_vector) + 1e-8)
            return 1 / (norm_class_gains_vector + 1)
        elif self.class_weighting_method == 'exp':
            square_class_gains_vector = torch.exp(self.cum_labels_info + 1) - 1
            norm_class_gains_vector = square_class_gains_vector / (torch.sum(square_class_gains_vector) + 1e-8)
            return 1 / (norm_class_gains_vector + 1)
        else:
            raise ValueError(f"Unknown class weighting method: {self.class_weighting_method}")

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                          checkpoint_file=None):
        """
        Standard interface for active learning sampling.

        Note: clf_model, trainDataset, supportingModels, and checkpoint_file are
        ignored -- this method only uses kernel similarity over features.
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), \
            f"BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {len(uSet)} and budgetSize: {self.cfg.ACTIVE_LEARNING.BUDGET_SIZE}"

        return self.select_samples(lSet, uSet)


def batched_diffs_efficient_weighted(K: torch.Tensor, C: torch.Tensor, chunk_size: int = 1024, diff_method: str = "abs_diff", cont_method: str = "positive", weight_method: str = "weighted", c_normalization: str = "sum"):
    D, N, _ = K.shape
    result = torch.empty((D, )).to(device=C.device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    if c_normalization == 'softmax':
        norm_C = torch.softmax(C, dim=1)
    else:
        norm_C = (C / sum_C)
    old_max = norm_C.max(dim=1, keepdim=True).values
    C_diff = (C - max_C).unsqueeze(0)
    num_iterations = int(N)
    max_C.unsqueeze_(0)
    use_softmax_norm = (c_normalization == 'softmax')
    if use_softmax_norm:
        exp_C = torch.exp(C - C.max(dim=1, keepdim=True).values)  # numerically stable exp (N, classes)
        Z_old = exp_C.sum(dim=1, keepdim=True)  # (N, 1)
        exp_C_max = exp_C.max(dim=1, keepdim=True).values  # (N, 1) -- exp of the max class
    for i in range(0, num_iterations, int(chunk_size)):
        end = min(i + chunk_size, D)
        K_batched = K[i:end]
        K_batched = K_batched.to('cuda')
        weights_batched = norm_C[i:end]

        if use_softmax_norm:
            # For each class c, adding K[i,j] to C[j,c]:
            # Z_new[j,c] = Z_old[j] + exp_C[j,c] * (exp(K[i,j]) - 1)
            # new_max_softmax[j,c] = max(exp_C_max[j], exp_C[j,c]*exp(K[i,j])) / Z_new[j,c]
            exp_K = torch.exp(K_batched)  # (batch, N, 1)
            Z_new = Z_old + exp_C.unsqueeze(0) * (exp_K - 1)  # (batch, N, classes)
            boosted_exp = exp_C.unsqueeze(0) * exp_K  # (batch, N, classes) -- exp(C[c]+K)
            new_max_num = torch.maximum(exp_C_max.unsqueeze(0).expand_as(boosted_exp), boosted_exp)
            new_state_vec = (new_max_num / Z_new) - old_max
            del exp_K, Z_new, boosted_exp, new_max_num
        else:
            future_sum = K_batched + sum_C
            state_add = max_C + K_batched

            new_state_vec = torch.maximum(-K_batched, C_diff)

            new_state_vec.add_(state_add)
            new_state_vec.div_(future_sum)
            new_state_vec.sub_(old_max)

        if cont_method == "positive":  # regular method
            new_state_vec.clamp_(min=0)
        elif cont_method == 'abs':  # take all contribution
            torch.abs(new_state_vec, out=new_state_vec)
        elif cont_method == "reg_sum_postive":  # take average contribution (not weighted by the prior)
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

        if weight_method == "weighted":
            result[i:end] = torch.einsum('ijk,ik->i', new_state_vec, weights_batched)
        elif weight_method == "equal":
            result[i:end] = torch.sum(new_state_vec, dim=(1, 2))
        else:
            raise ValueError(f"Unknown weight_method: {weight_method}. Must be 'weighted' or 'equal'.")

        del new_state_vec
        del K_batched
        del weights_batched
    return result


def batched_diffs_efficient_weighted_sparse(K_csr: torch.Tensor, C: torch.Tensor, chunk_size: int = 2048, cont_method: str = "positive", weight_method: str = "weighted", class_gains_weights: torch.Tensor = None, c_normalization: str = "sum", calc_method: str = "max"):
    D, N = K_csr.shape
    dev = C.device
    crow = K_csr.crow_indices().to(dev)  # shape (D+1,)
    ccol = K_csr.col_indices().to(dev)  # shape (nnz,)
    cvals = K_csr.values().to(dev)  # shape (nnz,)
    D = crow.numel() - 1
    classes = C.shape[1]
    uniform_default_val = 1.0 / classes

    use_softmax_norm = (c_normalization == 'softmax')

    result = torch.empty((D, )).to(device=C.device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)

    if use_softmax_norm:
        norm_C = torch.softmax(C, dim=1)
        old_max = norm_C.max(dim=1, keepdim=True).values
        C_stable = C - C.max(dim=1, keepdim=True).values  # for numerical stability
        exp_C = torch.exp(C_stable)  # (N, classes)
        Z_old = exp_C.sum(dim=1, keepdim=True)  # (N, 1)
        exp_C_max = exp_C.max(dim=1, keepdim=True).values  # (N, 1)
        exp_C = exp_C.squeeze()
        Z_old = Z_old.squeeze()
        exp_C_max = exp_C_max.squeeze()
    else:
        has_mass = (sum_C != 0)
        safe_sum = torch.where(has_mass, sum_C, torch.ones_like(sum_C))
        norm_C = torch.where(has_mass, C / safe_sum, torch.full_like(C, uniform_default_val))
        old_max = torch.where(has_mass, max_C / safe_sum, torch.zeros_like(max_C))

    C_diff = (C - max_C)
    max_C = max_C.squeeze()
    sum_C = sum_C.squeeze()
    C_diff = C_diff.squeeze()
    old_max = old_max.squeeze()

    if calc_method == "entropy":
        p_log2_p = norm_C * torch.log2(norm_C)
        h_before_global = -torch.sum(p_log2_p, dim=1)  # (N,)
        h_before_global = torch.where(has_mass.squeeze(), h_before_global, torch.zeros_like(h_before_global))

    elif calc_method in ("margin", "margin_not_normalized"):
        top2 = torch.topk(C, 2, dim=1)
        max1_val_global = top2.values[:, 0]
        max2_val_global = top2.values[:, 1]
        max1_idx_global = top2.indices[:, 0]
        max2_idx_global = top2.indices[:, 1]

        if calc_method == "margin":
            m_before_global = (max1_val_global - max2_val_global) / safe_sum.squeeze()
            m_before_global = torch.where(has_mass.squeeze(), m_before_global, torch.zeros_like(m_before_global))
        else:
            m_before_global = max1_val_global - max2_val_global

    for row_start in range(0, D, chunk_size):
        row_end = min(row_start + chunk_size, D)
        b = row_end - row_start

        starts = crow[row_start:row_end]  # shape (b,)
        ends = crow[row_start + 1: row_end + 1]  # shape (b,)
        lengths = (ends - starts).to(torch.long)  # (b,)

        total_nnz = int(lengths.sum().item())

        if total_nnz == 0:
            continue

        slice_start = int(starts[0].item())
        slice_end = int(ends[-1].item())

        cols_all = ccol[slice_start:slice_end]  # (total_nnz,)
        vals_all = cvals[slice_start:slice_end]  # (total_nnz,)

        row_indices = torch.repeat_interleave(torch.arange(b, device=dev, dtype=torch.long),
                                              lengths)  # (total_nnz,)

        global_rows = torch.arange(row_start, row_end, device=dev, dtype=torch.long)  # (b,)
        kvals = vals_all  # (total_nnz,)

        if calc_method == "entropy":
            c_vecs = C[cols_all]
            sum_C_cols = sum_C[cols_all].unsqueeze(1)
            h_before_cols = h_before_global[cols_all].unsqueeze(1)
            k_vals_2d = kvals.unsqueeze(1)

            future_base_vec_sum = sum_C_cols + k_vals_2d
            safe_future_sum = torch.where(future_base_vec_sum != 0, future_base_vec_sum, torch.ones_like(future_base_vec_sum))

            future_norm_base_vec = c_vecs / safe_future_sum

            future_base_log_vec = torch.sum(future_norm_base_vec * torch.log2(future_norm_base_vec), dim=1, keepdim=True)

            vec_plus_k = c_vecs + k_vals_2d
            first_ele = k_vals_2d * torch.log2(safe_future_sum / (vec_plus_k))
            second_ele = c_vecs * torch.log2((c_vecs) / (vec_plus_k))

            delta_vec = first_ele + second_ele
            final_vec = -(future_base_log_vec - (delta_vec / safe_future_sum))

            new_state = final_vec - h_before_cols

            del c_vecs, future_norm_base_vec, vec_plus_k, first_ele, second_ele, delta_vec
        elif calc_method == "max":

            old_max_cols = old_max[cols_all]  # (total_nnz,) or (total_nnz, classes)

            if use_softmax_norm:
                exp_K = torch.exp(kvals)  # (total_nnz,)
                exp_C_cols = exp_C[cols_all]  # (total_nnz, classes)
                Z_old_cols = Z_old[cols_all]  # (total_nnz,)
                exp_C_max_cols = exp_C_max[cols_all]  # (total_nnz,)

                exp_K_minus1 = (exp_K - 1).unsqueeze(1).expand_as(exp_C_cols)  # (total_nnz, classes)
                Z_new = Z_old_cols.unsqueeze(1) + exp_C_cols * exp_K_minus1  # (total_nnz, classes)

                boosted = exp_C_cols * exp_K.unsqueeze(1)  # (total_nnz, classes)
                new_max_num = torch.maximum(exp_C_max_cols.unsqueeze(1).expand_as(boosted), boosted)

                safe_Z = torch.where(Z_new != 0, Z_new, torch.ones_like(Z_new))
                new_state = (new_max_num / safe_Z) - old_max_cols.unsqueeze(1)

                del exp_K, exp_C_cols, Z_old_cols, exp_C_max_cols, exp_K_minus1, Z_new, boosted, new_max_num, safe_Z
            else:
                sumC_cols = sum_C[cols_all]  # (total_nnz,)
                maxC_cols = max_C[cols_all]  # (total_nnz,)
                Cdiff_cols = C_diff[cols_all]

                negk = -kvals  # (total_nnz,)
                negk_expand = negk.expand(classes, -1).T  # (total_nnz, classes)
                new_state = torch.maximum(negk_expand, Cdiff_cols)  # (total_nnz, classes)

                del negk_expand, Cdiff_cols

                state_add = maxC_cols + kvals  # (total_nnz,)
                new_state = new_state + state_add.expand(classes, -1).T

                future_sum = (kvals + sumC_cols)  # (total_nnz,)

                valid_denom = (future_sum != 0)
                safe_future_sum = torch.where(valid_denom, future_sum, torch.ones_like(future_sum))
                safe_denom_expanded = safe_future_sum.expand(classes, -1).T
                mask_expanded = valid_denom.expand(classes, -1).T
                new_state = new_state / safe_denom_expanded

                del safe_denom_expanded

                new_state = torch.where(mask_expanded, new_state, torch.zeros_like(new_state))
                new_state = new_state - old_max_cols.expand(classes, -1).T
        elif calc_method in ["margin", "margin_not_normalized"]:
            max1_val = max1_val_global[cols_all].unsqueeze(1)  # (total_nnz, 1)
            max2_val = max2_val_global[cols_all].unsqueeze(1)
            max1_idx = max1_idx_global[cols_all].unsqueeze(1)
            max2_idx = max2_idx_global[cols_all].unsqueeze(1)
            m_before_cols = m_before_global[cols_all].unsqueeze(1)

            c_vecs = C[cols_all]
            sum_C_cols = sum_C[cols_all].unsqueeze(1)
            k_vals_2d = kvals.unsqueeze(1)

            future_base_vec_sum = sum_C_cols + k_vals_2d
            safe_future_sum = torch.where(future_base_vec_sum != 0, future_base_vec_sum,
                                          torch.ones_like(future_base_vec_sum))

            new_val = c_vecs + k_vals_2d

            new_margin = torch.maximum(max1_val, new_val) - torch.maximum(max2_val, torch.minimum(new_val, max1_val))

            fixed_max1_margin = torch.abs((max1_val + k_vals_2d) - max2_val)
            fixed_max2_margin = torch.abs((max2_val + k_vals_2d) - max1_val)

            class_indices = torch.arange(classes, device=dev).unsqueeze(0)  # (1, classes)
            is_max1 = (class_indices == max1_idx)  # (total_nnz, classes)
            is_max2 = (class_indices == max2_idx)

            new_margin = torch.where(is_max1, fixed_max1_margin, new_margin)
            new_margin = torch.where(is_max2, fixed_max2_margin, new_margin)

            final_margin = new_margin / safe_future_sum if calc_method == "margin" else new_margin
            new_state = final_margin - m_before_cols

            del c_vecs, new_val, new_margin, is_max1, is_max2, fixed_max1_margin, fixed_max2_margin
        else:
            raise ValueError(f"Unknown calc_method: {calc_method}")

        if cont_method == "positive":
            new_state.clamp_(min=0.0)

        if class_gains_weights is not None:
            new_state = new_state * class_gains_weights

        weights_chunk = norm_C[global_rows]  # (b, classes)
        weights_for_nnz = weights_chunk[row_indices]  # (total_nnz, classes)

        if weight_method == "weighted":
            per_nnz_weighted = (new_state * weights_for_nnz).sum(dim=1)  # (total_nnz,)
        elif weight_method == "equal":
            per_nnz_weighted = new_state.sum(dim=1)  # (total_nnz,) - equal contribution
        else:
            raise ValueError(f"Unknown weight_method: {weight_method}. Must be 'weighted' or 'equal'.")

        chunk_result = torch.zeros((b,), device=dev, dtype=C.dtype)
        chunk_result.scatter_add_(0, row_indices, torch.nan_to_num(per_nnz_weighted, nan=0.0))

        result[row_start:row_end] = chunk_result
        torch.cuda.empty_cache()

    return result


def batched_diffs_sparse_max(
        K_csr: torch.Tensor,
        C: torch.Tensor,
        alpha: float,
        number_of_classes: int,
        chunk_size: int = 1024
):
    """
    Sparse equivalent of the dense diff_method="max" scoring rule.
    Computes row-wise sums of max( ((K + alpha)/(K + alpha * classes)) - C, 0 ).
    """
    D, N = K_csr.shape
    dev = C.device

    crow = K_csr.crow_indices().to(dev)  # shape (D+1,)
    ccol = K_csr.col_indices().to(dev)  # shape (nnz,)
    cvals = K_csr.values().to(dev)  # shape (nnz,)

    C_flat = C.squeeze()

    # In a sparse matrix, absent values are 0, so K_ij=0 evaluates to 1/number_of_classes.
    base_val = 1.0 / number_of_classes
    base_diffs = torch.maximum(base_val - C_flat, torch.zeros_like(C_flat))
    base_row_sum = torch.sum(base_diffs)

    result = torch.empty(D, device=dev, dtype=C.dtype)

    for row_start in range(0, D, chunk_size):
        row_end = min(row_start + chunk_size, D)
        b = row_end - row_start

        starts = crow[row_start:row_end]
        ends = crow[row_start + 1: row_end + 1]
        lengths = (ends - starts).to(torch.long)

        total_nnz = int(lengths.sum().item())

        if total_nnz == 0:
            result[row_start:row_end] = base_row_sum
            continue

        slice_start = int(starts[0].item())
        slice_end = int(ends[-1].item())

        cols_all = ccol[slice_start:slice_end]
        vals_all = cvals[slice_start:slice_end]

        row_indices = torch.repeat_interleave(
            torch.arange(b, device=dev, dtype=torch.long), lengths
        )

        C_cols = C_flat[cols_all]
        base_diffs_cols = base_diffs[cols_all]

        transformed_K = (vals_all + alpha) / (vals_all + alpha * number_of_classes)
        active_diffs = torch.maximum(transformed_K - C_cols, torch.zeros_like(transformed_K))

        deltas = active_diffs - base_diffs_cols

        chunk_deltas = torch.zeros(b, device=dev, dtype=C.dtype)
        chunk_deltas.scatter_add_(0, row_indices, torch.nan_to_num(deltas, nan=0.0))

        result[row_start:row_end] = base_row_sum + chunk_deltas

        del cols_all, vals_all, row_indices, C_cols, base_diffs_cols, transformed_K, active_diffs, deltas
        torch.cuda.empty_cache()

    return result
