import json
import os
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import deep_al.pycls.datasets.utils as ds_utils
from deep_al.pycls.al.kernel_utils import build_K_general_matrix, RBFKernel, TopHatKernel, compute_local_sigmas

torch.cuda.empty_cache()


class   CoverageMatrixMethod:
    def __init__(self, cfg, train_labels):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        features_tag = self.cfg['DATASET'].get('FEATURES_TAG', None)
        features_normalized = bool(getattr(self.cfg, 'FEATURES_NORMALIZED', True))
        self.all_features = ds_utils.load_features(
            self.ds_name,
            train=True,
            normalized=features_normalized,
            features_tag=features_tag,
        )
        self.debug = self.cfg.DEBUG
        self.use_sparse = getattr(self.cfg, 'SPARSE_K', False)
        self.device = 'cuda' if (self.use_sparse or self.all_features.shape[0] <= 100000) else 'cpu'

        mat_type = getattr(self.cfg, 'MATRIX_DTYPE', 'float32')
        self.matrices_type = getattr(torch, mat_type)
        
        self.features_gpu = torch.from_numpy(self.all_features).to(self.device, dtype=self.matrices_type)
        self.budgetSize = cfg.ACTIVE_LEARNING.BUDGET_SIZE
        self.sigma = cfg.ACTIVE_LEARNING.INITIAL_SIGMA if 'INITIAL_SIGMA' in cfg.ACTIVE_LEARNING else 1.0


        self.alpha = 0
        if cfg.ACTIVE_LEARNING.INITIAL_DELTA < 0:
            print("Delta not specified. Starting automated selection process...")
    
            # Track time for performance monitoring
            start_time = time.time()

            print("Running Delta selection algorithm...")

            cfg.ACTIVE_LEARNING.INITIAL_DELTA = ds_utils.select_delta(
                features=self.all_features,
                num_classes=self.cfg.MODEL.NUM_CLASSES,
                seed=self.seed
            )

            elapsed = time.time() - start_time
            print(f"Delta selection complete | Value: {cfg.ACTIVE_LEARNING.INITIAL_DELTA:.4f} | Time: {elapsed:.2f}s")

        self.delta = cfg.ACTIVE_LEARNING.INITIAL_DELTA

        self.train_labels_general = np.array(train_labels)
        unique_labels = np.unique(self.train_labels_general)
        self.unique_labels = unique_labels
        self.num_of_classes = unique_labels.size

        self.kernel_build_batch_size = getattr(self.cfg, 'KERNEL_BUILD_BATCH_SIZE', 1024)

        self.kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        self.rbf_cutoff_mode = getattr(self.cfg, 'RBF_CUTOFF_MODE', 'none')
        self.use_local_sigma = bool(getattr(self.cfg, 'LOCAL_SIGMA', False))
        self.local_sigma_k = int(getattr(self.cfg, 'LOCAL_SIGMA_K', 7))

        self.local_sigma = None
        if self.use_local_sigma:
            print(f"Computing local sigma with K={self.local_sigma_k} (self-tuning kernel)")
            self.local_sigma = compute_local_sigmas(
                self.all_features,
                k=self.local_sigma_k,
                batch_size=self.kernel_build_batch_size,
                device=self.device,
                dtype=torch.float32,
            )
            self._record_local_sigma_stats()

        if self.kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
            initial_threshold = self.delta
            rbf_keep_mode = 'kernel_value'
        else:
            self.kernel_fn = RBFKernel('cuda')
            if self.rbf_cutoff_mode == 'probcover':
                initial_threshold = self.delta
                rbf_keep_mode = 'local_probcover' if self.use_local_sigma else 'distance'
            elif self.use_sparse:
                if self.use_local_sigma:
                    # Preserve the existing sparse local-sigma behavior by default.
                    initial_threshold = self.delta
                    rbf_keep_mode = 'distance'
                else:
                    initial_threshold = np.exp(-1.0 * (self.delta / self.sigma) ** 2)
                    rbf_keep_mode = 'kernel_value'
            else:
                initial_threshold = 0.0
                rbf_keep_mode = 'kernel_value'

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
            local_sigma=self.local_sigma,
            rbf_keep_mode=rbf_keep_mode,
        )

        self.C_general = torch.full((self.all_features.shape[0], unique_labels.size), self.alpha, device=self.device,
                                        dtype=self.matrices_type)

    def _build_local_sigma_stats(self):
        if self.local_sigma is None:
            return None

        sigma_cpu = self.local_sigma.detach().cpu().to(torch.float64)
        quantile_levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
        quantiles = torch.quantile(sigma_cpu, quantile_levels)

        stats = {
            "dataset": self.ds_name,
            "seed": int(self.seed),
            "kernel_type": self.kernel_type,
            "local_sigma_k": int(self.local_sigma_k),
            "num_points": int(sigma_cpu.numel()),
            "mean": float(torch.mean(sigma_cpu).item()),
            "std": float(torch.std(sigma_cpu, unbiased=False).item()),
            "min": float(torch.min(sigma_cpu).item()),
            "q05": float(quantiles[0].item()),
            "q25": float(quantiles[1].item()),
            "median": float(quantiles[2].item()),
            "q75": float(quantiles[3].item()),
            "q95": float(quantiles[4].item()),
            "max": float(torch.max(sigma_cpu).item()),
        }
        stats["iqr"] = stats["q75"] - stats["q25"]
        stats["range"] = stats["max"] - stats["min"]
        stats["cv"] = stats["std"] / stats["mean"] if stats["mean"] > 0 else None
        return stats

    def _record_local_sigma_stats(self):
        stats = self._build_local_sigma_stats()
        if stats is None:
            return

        print("Local sigma stats:")
        print(
            "  mean={mean:.6f} std={std:.6f} median={median:.6f} min={min:.6f} "
            "q05={q05:.6f} q25={q25:.6f} q75={q75:.6f} q95={q95:.6f} max={max:.6f}".format(**stats)
        )

        exp_dir = getattr(self.cfg, 'EXP_DIR', None)
        if not exp_dir:
            return

        os.makedirs(exp_dir, exist_ok=True)
        json_path = os.path.join(exp_dir, "local_sigma_stats.json")
        txt_path = os.path.join(exp_dir, "local_sigma_stats.txt")

        with open(json_path, "w", encoding="ascii") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)
            handle.write("\n")

        ordered_keys = [
            "dataset",
            "seed",
            "kernel_type",
            "local_sigma_k",
            "num_points",
            "mean",
            "std",
            "cv",
            "min",
            "q05",
            "q25",
            "median",
            "q75",
            "q95",
            "max",
            "iqr",
            "range",
        ]
        with open(txt_path, "w", encoding="ascii") as handle:
            for key in ordered_keys:
                handle.write(f"{key}: {stats[key]}\n")


    def init_sampling_loop(self, lset, uset):
        torch.cuda.empty_cache()
        self.set_rel_features(lset, uset)
        self.activeSet = []
        
        if self.use_sparse:
            K_csr_shuffled = self.K_general[self.relevant_indices, :][:, self.relevant_indices]
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
        self.K =  self.K.to(self.device) if self.K is not None else None
        self.C = self.C_general[self.relevant_indices]
        self.train_labels = self.train_labels_general[self.relevant_indices]
        
    def set_rel_features(self, lset, uset):
        self.lSet = lset
        self.uSet = uset
        # print(lset)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        if isinstance(self.all_features, torch.Tensor):
            self.rel_features = self.all_features[self.relevant_indices]
        elif isinstance(self.all_features, np.ndarray):
            self.rel_features = torch.from_numpy(self.all_features[self.relevant_indices])
    
    @torch.inference_mode()
    def select_samples(self, lset, uset):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """

        self.init_sampling_loop(lset, uset)

        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        for i in tqdm(range(self.budgetSize), desc="Selecting Samp"):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)

            max_vals, indices = torch.max(self.C, dim=1)
            if self.use_sparse:
                point_total_contribution = batched_diffs_sparse(self.K, max_vals, 0, self.num_of_classes,
                                                         diff_method="abs_diff")
            else:
                point_total_contribution = batched_diffs(self.K, max_vals, 0, self.num_of_classes,
                                                     diff_method="abs_diff")


            point_total_contribution[curr_l_set] = -np.inf
            
            sampled_point = torch.argmax(point_total_contribution).item()
            # sampled_point = np.argsort(-point_total_contribution.cpu().numpy(), kind='stable')[0].item()
            chosen_label = self.train_labels[sampled_point].item()

            K_row_dense = self.K[sampled_point].to_dense().squeeze()

            self.C[:, chosen_label] = torch.maximum(self.C[:, chosen_label], K_row_dense)


            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)
            del K_row_dense

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]

        self.C_general[self.relevant_indices] = self.C
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')

        del self.K
        del self.C

        return activeSet, remainSet
    
    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                        checkpoint_file=None):
        """
        Standard interface for active learning sampling.
        Adapts the algorithm to work with the unified AL interface.
        
        Note: checkpoint_file is ignored for non-DCoM algorithms.
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), \
            f"BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {len(uSet)} and budgetSize: {self.cfg.ACTIVE_LEARNING.BUDGET_SIZE}"
        
        return self.select_samples(lSet, uSet)



def batched_diffs(K, C, alpha, number_of_classes, chunk_size=1024, diff_method="abs_diff"):
    D, N = K.shape
    result = torch.empty(D).to(device=C.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if diff_method == "abs_diff":
            
            K_batched = K[start:end]
            K_batched = K_batched.to(C.device)
            diff = K_batched.sub(C) 
            diff.clamp_(min=0)
            result[start:end] = diff.sum(dim=1)
            del diff
            # result[start:end] = torch.sum(torch.maximum(K_batched - C, torch.zeros_like(K_batched).to(device=C.device)),
            #                               dim=1)
        elif diff_method == "max":
            K_batched = K[start:end]
            K_batched = K_batched.to('cuda')
            result[start:end] = torch.sum(
                torch.maximum(((K_batched + alpha) / (
                    torch.maximum(K_batched + alpha * number_of_classes, torch.full_like(K_batched, 1e-8)))) - C,
                              torch.zeros_like(K_batched).to(device=C.device)), dim=1)
        elif diff_method == 'margin':
            result[start:end] = torch.sum(
                torch.maximum((K[start:end] / (K[start:end] + alpha * number_of_classes)) - C,
                              torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    return result


def batched_diffs_sparse(K, C, alpha=None, number_of_classes=None, chunk_size=1024, diff_method="abs_diff"):
    """
    Computes batched diffs for a Sparse CSR Tensor K and a dense tensor/scalar C.

    Assumes:
    1. K is a torch.sparse_csr_tensor.
    2. C >= 0 (This allows us to ignore the zero-entries in K, keeping operations sparse).
    """
    D, N = K.shape
    device = C.device

    # Extract CSR components and move to device
    crow = K.crow_indices().to(device)  # shape (D+1,)
    ccol = K.col_indices().to(device)   # shape (nnz,)
    cvals = K.values().to(device)       # shape (nnz,)

    # Pre-allocate result on the correct device
    result = torch.zeros(D, device=device, dtype=cvals.dtype)

    for row_start in range(0, D, chunk_size):
        row_end = min(row_start + chunk_size, D)
        b = row_end - row_start

        # CSR pointers for the chunk
        starts = crow[row_start:row_end]           # shape (b,)
        ends = crow[row_start + 1: row_end + 1]    # shape (b,)
        lengths = (ends - starts).to(torch.long)   # (b,)

        total_nnz = int(lengths.sum().item())

        if total_nnz == 0:
            # nothing in this chunk, skip
            continue

        # global slice of indices/values for this chunk
        slice_start = int(starts[0].item())
        slice_end = int(ends[-1].item())

        cols_all = ccol[slice_start:slice_end]   # (total_nnz,)
        vals_all = cvals[slice_start:slice_end]  # (total_nnz,)

        # row index for each nnz entry within the chunk: 0..b-1 repeated by lengths
        row_indices = torch.repeat_interleave(
            torch.arange(b, device=device, dtype=torch.long),
            lengths
        )  # (total_nnz,)

        if diff_method == "abs_diff":
            # --- SPARSE OPTIMIZATION EXPLAINED ---
            # The operation is sum(max(K - C, 0)).
            # For sparse entries where K=0, this is max(0 - C, 0).
            # If C >= 0, this result is 0.
            # Therefore, we only need to work with the non-zero values.

            # Handle broadcasting for C
            if C.numel() == 1:
                # C is a scalar
                c_mapped = C
            elif C.ndim == 1 and C.shape[0] == N:
                # C is a vector matching the columns (feature dim).
                # We gather only the C values that align with the non-zero columns.
                c_mapped = C[cols_all]
            else:
                raise NotImplementedError("C shape not supported for sparse optimization")

            # Perform ReLU(Values - C)
            new_vals = torch.relu(vals_all - c_mapped)

            # Aggregate per row via scatter_add
            chunk_result = torch.zeros(b, device=device, dtype=cvals.dtype)
            chunk_result.scatter_add_(0, row_indices, new_vals)

            result[row_start:row_end] = chunk_result

    return result
