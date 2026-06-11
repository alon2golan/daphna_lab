import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import deep_al.pycls.datasets.utils as ds_utils
from deep_al.pycls.al.kernel_utils import build_K_general_matrix, RBFKernel, TopHatKernel

torch.cuda.empty_cache()


class   CoverageMatrixMethod:
    def __init__(self, cfg, annotator_manager=None):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        features_tag = self.cfg['DATASET'].get('FEATURES_TAG', None)
        self.all_features = ds_utils.load_features(self.ds_name, train=True, features_tag=features_tag)
        self.debug = self.cfg.DEBUG
        self.use_sparse = getattr(self.cfg, 'SPARSE_K', False)
        self.device = 'cuda' if (self.use_sparse or self.all_features.shape[0] <= 100000) else 'cpu'

        mat_type = getattr(self.cfg, 'MATRIX_DTYPE', 'float32')
        self.matrices_type = getattr(torch, mat_type)
        
        self.features_gpu = torch.from_numpy(self.all_features).to(self.device, dtype=self.matrices_type)
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

        self.num_of_classes = cfg.MODEL.NUM_CLASSES

        self.kernel_build_batch_size = getattr(self.cfg, 'KERNEL_BUILD_BATCH_SIZE', 1024)

        self.kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if self.kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
            initial_threshold = self.delta
        else:
            self.kernel_fn = RBFKernel('cuda')
            initial_threshold = np.exp(-1.0 * (self.delta / self.sigma) ** 2) if self.use_sparse else 0.0

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

        self.C_general = torch.full((self.all_features.shape[0], self.num_of_classes), self.alpha, device=self.device,
                                        dtype=self.matrices_type)

        self.annotator_manager = annotator_manager


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
    def select_samples(self, lset, uset, num_samples, annotator_id=None):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """

        self.init_sampling_loop(lset, uset)

        print(f'Start selecting {num_samples} samples.')
        selected = []
        for i in tqdm(range(num_samples), desc="Selecting Samp"):
            self._select_one(selected, annotator_id)

        return self._finalize_selection(selected, num_samples)

    def _select_one(self, selected, annotator_id):
        """
        One greedy step: pick the highest-out-degree point, annotate it in-loop
        (label may be noisy), and fold its coverage into C. Mutates `selected`.
        Returns (sampled_point_local, global_idx, chosen_label).
        """
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
        global_idx = int(self.relevant_indices[sampled_point])
        result = self.annotator_manager.annotate_samples([global_idx], annotator_id)
        chosen_label = result['annotated_labels'][global_idx]

        K_row_dense = self.K[sampled_point].to_dense().squeeze()
        self.C[:, chosen_label] = torch.maximum(self.C[:, chosen_label], K_row_dense)


        assert sampled_point not in selected, 'sample was already selected'
        selected.append(sampled_point)
        del K_row_dense
        return sampled_point, global_idx, chosen_label

    def _finalize_selection(self, selected, num_samples):
        """Common tail: persist C, build activeSet/remainSet, free buffers."""
        assert len(selected) == num_samples, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]

        self.C_general[self.relevant_indices] = self.C
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')

        del self.K
        del self.C

        return activeSet, remainSet

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                        checkpoint_file=None, num_samples=None, annotator_id=None):
        """
        Standard interface for active learning sampling.
        Adapts the algorithm to work with the unified AL interface.

        num_samples: number of samples to select this round (computed from budget / annotator cost).
        annotator_id: If provided, each selected point is annotated in-loop via annotator_manager
                      so that C is updated with the actual (possibly noisy) label.
        Note: checkpoint_file is ignored for non-DCoM algorithms.
        """
        assert num_samples is not None and num_samples > 0, "Expected a positive num_samples"
        assert num_samples < len(uSet), \
            f"num_samples cannot exceed length of unlabelled set. Length of unlabelled set: {len(uSet)} and num_samples: {num_samples}"

        return self.select_samples(lSet, uSet, num_samples, annotator_id=annotator_id)


class NoiseAwareCoverage(CoverageMatrixMethod):
    """
    NPC — Noise-aware ProbCover (Netta's thesis, the query-side contribution).

    Wraps the greedy coverage selection in a Noise-Aware Sampling loop: select in
    inner batches of `b`, and between batches re-run a noise filter over the
    labeled set and rebuild the coverage matrix C from only the points predicted
    clean. A noisy labeled point therefore stops "covering" its neighbourhood, so
    that region re-opens and can be re-sampled.
    """

    def __init__(self, cfg, annotator_manager=None, noise_filter=None):
        super().__init__(cfg, annotator_manager=annotator_manager)
        self.noise_filter = noise_filter
        b = getattr(cfg.NOISE_FILTER, 'INNER_BATCH_SIZE', -1)
        self.inner_batch_size = self.num_of_classes if (b is None or int(b) < 0) else int(b)
        # Set per-round in sample_from_uSet so the filter can train on noisy labels.
        self.train_data = None

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                         checkpoint_file=None, num_samples=None, annotator_id=None):
        # trainDataset must serve the (possibly noisy) annotated labels — i.e. the
        # NoisyLabelDatasetWrapper — so the AUM filter sees what was actually labelled.
        self.train_data = trainDataset
        return super().sample_from_uSet(
            clf_model, lSet, uSet, trainDataset, supportingModels=supportingModels,
            checkpoint_file=checkpoint_file, num_samples=num_samples, annotator_id=annotator_id,
        )

    @torch.inference_mode()
    def select_samples(self, lset, uset, num_samples, annotator_id=None):
        assert self.noise_filter is not None, "NoiseAwareCoverage requires a noise_filter"
        self.init_sampling_loop(lset, uset)

        print(f'NPC | Start selecting {num_samples} samples (inner batch b={self.inner_batch_size}).')
        n_lset = len(self.lSet)
        selected = []
        round_idx = 0
        while len(selected) < num_samples:
            k = min(self.inner_batch_size, num_samples - len(selected))

            # Current labeled set = original lSet (locals 0..n_lset-1) + points
            # already selected this episode (their local indices in relevant_indices).
            labeled_local = list(range(n_lset)) + list(selected)
            labeled_global = [int(self.relevant_indices[l]) for l in labeled_local]

            self._rebuild_C_from_clean(labeled_local, labeled_global, annotator_id, round_idx)

            for _ in range(k):
                self._select_one(selected, annotator_id)
            round_idx += 1

        return self._finalize_selection(selected, num_samples)

    def _rebuild_C_from_clean(self, labeled_local, labeled_global, annotator_id, round_idx):
        """
        Reset C and re-accumulate coverage from only the labeled points the noise
        filter predicts clean. Noisy-predicted points contribute nothing, so their
        coverage region re-opens for the next inner batch.
        """
        self.C = torch.full((len(self.relevant_indices), self.num_of_classes),
                            self.alpha, device=self.device, dtype=self.matrices_type)

        if len(labeled_global) == 0:
            # Bootstrap round with an empty labeled set — plain coverage from scratch.
            return

        # select_samples runs under torch.inference_mode(); the AUM filter trains a
        # model, so autograd must be re-enabled for the filter call.
        with torch.inference_mode(False):
            with torch.enable_grad():
                clean_mask = self.noise_filter.predict_clean(
                    self.cfg, self.train_data,
                    np.asarray(labeled_global, dtype=int), np.asarray(self.uSet, dtype=int),
                    self.annotator_manager,
                )

        n_clean = int(np.sum(clean_mask))
        print(f'NPC | round {round_idx}: filter kept {n_clean}/{len(labeled_global)} labeled points clean.')

        for local, g, is_clean in zip(labeled_local, labeled_global, clean_mask):
            if not is_clean:
                continue
            label = int(self.annotator_manager.annotated_labels[int(g)])
            K_row = self.K[local].to_dense().squeeze()
            self.C[:, label] = torch.maximum(self.C[:, label], K_row)
            del K_row



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