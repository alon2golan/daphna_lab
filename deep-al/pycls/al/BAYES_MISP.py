import numpy as np
import pycls.datasets.utils as ds_utils
import os
from . import prior_selection
from pycls.utils.kernels import RBFKernel, TopHatKernel
import torch


class BAYES_MISP:
    def __init__(self, cfg, budgetSize, train_labels, lset, uset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.features = torch.as_tensor(ds_utils.load_features(self.ds_name, train=True, seed=self.seed))
        self.rel_indices = np.concatenate([lset, uset]).astype(int)
        self.features = self.features[self.rel_indices]
        
        # Load pre-computed clarity scores
        self.clarity_scores = self._load_clarity_scores()
        
        self.alpha = self.cfg.ALPHA if 'ALPHA' in self.cfg else 0.0
        self.alpha_upper = self.cfg.ALPHA_UPPER if 'ALPHA_UPPER' in self.cfg else 0.1
        self.alpha_lower = self.cfg.ALPHA_LOWER if 'ALPHA_LOWER' in self.cfg else 0.001
        
        self.debug = self.cfg.DEBUG
        # self.cont_method = self.cfg.CONT_METHOD if 'CONT_METHOD' in self.cfg else 'positive'
        self.sparsity_percent = self.cfg.SPARSITY_PERCENT if 'SPARSITY_PERCENT' in self.cfg else 0.0
        print(f"Sparsity percent value =========== : {self.sparsity_percent}")

        self.budgetSize = budgetSize
        self.delta = delta # delta for topHat, sigma for RBF

        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff'
        kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel()
        else:
            self.kernel_fn = RBFKernel()
        
        if self.sparsity_percent > 0:
            print("Computing sparse K matrix...")
            # In case of sparse K, we store it on GPU
            self.K = self.kernel_fn.compute_sparse_gram(self.features, sigma=self.delta, sparsity_percent=self.sparsity_percent)
            self.K_vals = self.K.values()
            self.K_col_indices = self.K.col_indices()
            self.K_row_indices = torch.repeat_interleave(
                torch.arange(self.K.shape[0], device=self.K.device, dtype=torch.long), 
                torch.diff(self.K.crow_indices())
            )
        else:
            print("Computing dense K matrix...")
            self.K = self.kernel_fn.compute_gram(self.features, sigma=self.delta, batch_size=8192, final_type=torch.float16)
        
        self.unique_labels = np.unique(train_labels)
        self.num_of_classes = self.unique_labels.size
        self.train_labels = np.array(train_labels)[self.rel_indices] # train_labels is now locally indexed
        
        self.init_C(lset)
                
        # Track labeled points information
        self.chosen_labels_num = torch.zeros(self.num_of_classes, device=self.C.device, dtype=torch.int32)
        self.activeSet = []
        self.coverages = [] # EXP
        self.coverages.append(self.get_point_coverage()) # EXP get initial coverage
        
    
    def get_priors(self):
        if self.alpha == 0.0:
            # Use pre-computed clarity scores if available
            if self.clarity_scores is not None:
                print("Using pre-computed clarity scores for prior selection...")
                rel_measures = self.clarity_scores
            # Fallback to sparse K calculation if Laplace clarity is missing
            elif self.sparsity_percent > 0:
                K_for_rel = self.kernel_fn.compute_sparse_gram_knn(
                    self.features, sigma=self.delta, final_type=torch.float16, k=500
                )
                rel_measures = prior_selection.oracle_clarity_K_sparse(
                    K_for_rel, self.train_labels, self.num_of_classes, matrix_peakedness=prior_selection.compute_margin_peakedness)
                del K_for_rel
            else:
                raise NotImplementedError("No clarity scores found and non-sparse prior selection not implemented.")
            
            priors = prior_selection.get_priors_logistic(
                rel_measures.numpy() if isinstance(rel_measures, torch.Tensor) else rel_measures, 
                alpha_min=self.alpha_lower, 
                alpha_max=self.alpha_upper, 
                normalize_input=False
            ) # (N,)
        else:
            # Uniform prior of self.alpha
            priors = torch.tensor([self.alpha] * len(self.features))
            
        # Ensure priors is a tensor before returning
        if not isinstance(priors, torch.Tensor):
            priors = torch.tensor(priors)
            
        return priors
    
    def _load_clarity_scores(self):
        """
        Load pre-computed Laplace clarity scores and return the (N,) clarity vector properly indexed.
        """
        if 'CLARITY_PATH' in self.cfg:
            clarity_path = self.cfg.CLARITY_PATH
        else:
            # Try multiple possible locations for the clarity file
            base_paths = [
                f'laplace_full_train_clarity.npy',  # current directory
                f'../../../laplace_full_train_clarity.npy',  # relative to tools dir
                f'/cs/labs/daphna/dvir_mateless/laplace_full_train_clarity.npy',  # absolute workspace root
            ]
            clarity_path = None
            for path in base_paths:
                if os.path.exists(path):
                    clarity_path = path
                    break
            if clarity_path is None:
                clarity_path = base_paths[0]  # default for error message
        
        if os.path.exists(clarity_path):
            full_clarity = np.load(clarity_path)
            clarity_aligned = torch.from_numpy(full_clarity[self.rel_indices]).float()
            print(f"Loaded Clarity from {clarity_path}, computed clarity scores shape: {clarity_aligned.shape}")
            return clarity_aligned
        else:
            print(f"Clarity file not found at {clarity_path}, will use kernel-based prior selection if alpha=0.0")
            return None
        
        
    def init_C(self, lset):
        """
        Init the main matrix C with the priors. If lset != empty, Init C accordingly.
        NOTE: Attention! float16
        """
        self.priors = self.get_priors().to(self.device, dtype=torch.float16) # This is (N,) tensor on GPU
        self.C = self.priors.unsqueeze(1).repeat(1, self.num_of_classes).contiguous() # (N, M)
        
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
        
        if self.sparsity_percent > 0:
            raise NotImplementedError("Loading C from lset not implemented for sparse matrices.") # TODO
        else:
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
            
            elif self.diff_method == 'full_weighted_max': ### the method with the expectation
                if self.sparsity_percent > 0:
                    # point_total_contribution = self.expectation_proxy_max_sparse()
                    point_total_contribution = batched_diffs_efficient_weighted_sparse(
                        self.K, self.C, chunk_size=5120, cont_method="positive"
                    )
                else:
                    point_total_contribution = expectation_proxy_max(self.K, self.C)

            # TODO: Make sure we don't even calculate contribution for selected points in the first place
            point_total_contribution[curr_l_set] = -np.inf
            # Note: This is for stability - argmax is weird sometimes.
            # sampled_point = np.argsort(point_total_contribution.cpu().numpy(), kind='stable')[::-1][0].item()
            sampled_point = torch.argmax(point_total_contribution).item()
            chosen_label = self.train_labels[sampled_point].item()

            self.chosen_labels_num[chosen_label] += 1

            if self.sparsity_percent > 0:
                similarity_row = self.K[sampled_point].to_dense().to('cuda').squeeze()
            else:
                similarity_row = self.K[sampled_point].to('cuda').squeeze()

            if self.diff_method in ['prob_cover', 'max_herding']:
                self.C[:, chosen_label] = torch.maximum(self.C[:, chosen_label], similarity_row)
            else:
                self.C[:, chosen_label] += similarity_row.to(dtype=torch.float16)
                
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
    
    
    def expectation_proxy_max_sparse(self, batch_size=8192):
        """
        Calculates expectation proxy using sparse K.
        Optimized: Batch-Row + FP16 Precision (No Compile).
        """
        N, M = self.C.shape
        device = self.device
        
        # === USE FP16 FOR SPEED (Crucial for bandwidth) ===
        dtype_compute = torch.float16
        
        # Cast Inputs
        C_curr = self.C.to(dtype_compute)
        
        # --- 1. Pre-calculate Global Stats (Dense) ---
        # Keep clean 1D tensors (N,)
        max_C = torch.max(C_curr, dim=1).values # (N,)
        sum_C = torch.sum(C_curr, dim=1)        # (N,)
        old_max_norm = max_C / sum_C            # (N,)
        
        C_diff = C_curr - max_C.unsqueeze(1)    
        weights_all = C_curr / sum_C.unsqueeze(1) 

        total_scores = torch.zeros(N, device=device, dtype=torch.float32)

        # --- 2. Reshape for Batch Processing (N, k) ---
        vals = self.K_vals.to(dtype_compute)
        cols = self.K_col_indices
        num_edges = vals.shape[0]
        
        if N > 0:
            k = num_edges // N
        else:
            k = 0

        # View as (N, k) matrix
        vals_matrix = vals.view(N, k)
        cols_matrix = cols.view(N, k)
        
        # --- 3. Iterate over Batches ---
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            
            b_cols = cols_matrix[start:end]
            
            # Gather & Compute (Eager Mode)
            # We use contiguous() to keep memory access clean and fast
            batch_scores = compute_gain_kernel_batch(
                vals    = vals_matrix[start:end].contiguous(),     # (B, k)
                sum_C   = sum_C[b_cols].contiguous(),              # (B, k)
                max_C   = max_C[b_cols].contiguous(),              # (B, k)
                old_max = old_max_norm[b_cols].contiguous(),       # (B, k)
                C_diff  = C_diff[b_cols].contiguous(),             # (B, k, M)
                weights = weights_all[start:end].unsqueeze(1).contiguous() # (B, 1, M)
            )
            
            total_scores[start:end] = batch_scores.to(torch.float32)

        return total_scores
    
    def margin_old_proxy(self):
        """
        Calculates the acquisition score for all unlabeled points 'i' simultaneously
        using the vectorized form of the objective function (6) with the margin score (9):
        contribution(i) = sum over all j's of max{f(p^{{i}}[j]) - f(p^{L}[j]), 0}
        """
        N = len(self.features)
        M = self.num_of_classes
        
        C_cpu = self.C.to('cpu', dtype=torch.float32)  # (N, M)

        sum_C = torch.sum(C_cpu, dim=1, keepdim=True) # (N, 1)
        norm_C = C_cpu / sum_C                        # (N, M) -> p^L[j]
        
        vals = torch.topk(norm_C, k=2, dim=1).values   # (N, 2)
        old_margin = vals[:, 0] - vals[:, 1]           # (N,) vector -> f(p^L[j]) for all j

        alpha_M_j = M * self.priors # (N,) vector
        denominator_i_j = alpha_M_j + self.K

        F_i_j = self.K / denominator_i_j # (N, N) matrix

        difference_matrix = F_i_j - old_margin # (N, N) matrix
        
        max_term_matrix = torch.max(difference_matrix, torch.tensor(0.0)) # (N, N) matrix
        
        total_point_contribution = torch.sum(max_term_matrix, dim=1) # (N,) vector

        return total_point_contribution       
        
        
# ----------------------- Functions for Contribution Computation ----------------------- #

def compute_gain_kernel_batch(vals, sum_C, max_C, old_max, C_diff, weights):
    """
    Computes gain for (B, k) neighbors.
    Runs in standard Eager mode (Stable & Fast).
    """
    # 1. Reshape for Broadcasting: (B, k) -> (B, k, 1)
    # We do this explicitly to allow broadcasting against (B, k, M)
    vals_u    = vals.unsqueeze(-1)
    sum_C_u   = sum_C.unsqueeze(-1)
    max_C_u   = max_C.unsqueeze(-1)
    old_max_u = old_max.unsqueeze(-1)
    
    # 2. Denominator: (B, k, 1)
    future_sum = sum_C_u + vals_u
    
    # 3. Numerator: (B, k, M)
    potential_overtake = (C_diff + vals_u).relu()
    new_max_vals = max_C_u + potential_overtake
    
    # 4. Weighted Sum
    # Gain = (New_Prob - Old_Prob) * Weight
    raw_gain = (new_max_vals / future_sum) - old_max_u
    
    # Sum over Neighbors (dim=1) and Classes (dim=2)
    return (raw_gain.clamp(min=0) * weights).sum(dim=[1, 2])

@torch.compile
def compute_gain_kernel(K_batch, C_max, C_diff, weights, sum_C, old_max_norm):
    """
    Computes the gain for a batch. 
    This function will be fused into a single Triton kernel by torch.compile.
    """
    # Denominator (updates regardless of class)
    future_sum = sum_C + K_batch 
    # Numerator (The ReLU here calculates the NEW MAX VALUE)
    potential_overtake = (C_diff + K_batch).relu() 
    # Calculate the Raw Gain for every possible label update 'm'
    # Shape: (B, N, M)
    new_max_vals = C_max + potential_overtake
    new_probs = new_max_vals / future_sum
    
    raw_gain = new_probs - old_max_norm # Can be negative
    
    gain_clamped = raw_gain.clamp(min=0)
    
    return (gain_clamped * weights).sum(dim=-1).sum(dim=1)

def expectation_proxy_max(K: torch.Tensor, C: torch.Tensor, chunk_size: int = 16384):
    """
    K: (N, N) Similarity matrix (CPU or GPU)
    C: (N, M) Class counts/alphas (GPU) (we transpose it before passing)
    """
    device = C.device
    N, M = C.shape
    
    if len(K.shape) == 2:
        K = K.unsqueeze(2)

    result = torch.empty((N, ), device=device)
    
    max_C, _ = torch.max(C, dim=1, keepdim=True) # (N, 1)
    sum_C = torch.sum(C, dim=1, keepdim=True)    # (N, 1)
    norm_C = (C / sum_C)                         # (N, M) -> Weights p^L[j]
    old_max = (max_C / sum_C)                    # (N, 1)
    
    # C_diff: Distance of every class from the max class.
    # Values are <= 0. Zero indicates the current max class.
    C_diff = (C - max_C).unsqueeze(0)
    
    # Reshape other constants for broadcasting against Batch dim
    max_C_sq = max_C.unsqueeze(0)   # (1, N, 1)
    sum_C_sq = sum_C.unsqueeze(0)   # (1, N, 1)
    old_max_sq = old_max.unsqueeze(0) # (1, N, 1)

    if K.device.type == 'cpu':
        K = K.pin_memory()

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        
        # Async transfer to GPU
        # Shape: (B, N, 1)
        K_batched = K[i:end].to(device, non_blocking=True)
        
        weights_batched = norm_C[i:end].unsqueeze(1) 

        # Call the compiled kernel
        batch_scores = compute_gain_kernel(
            K_batched, 
            max_C_sq, 
            C_diff, 
            weights_batched, 
            sum_C_sq, 
            old_max_sq
        )
        result[i:end] = batch_scores
        
    return result


def batched_diffs_efficient_weighted_sparse(K_csr: torch.Tensor, C: torch.Tensor, chunk_size: int = 5120, cont_method: str = "positive"):
    D, N = K_csr.shape
    dev = C.device
    crow = K_csr.crow_indices().to(dev)  # shape (D+1,)
    ccol = K_csr.col_indices().to(dev)  # shape (nnz,)
    cvals = K_csr.values().to(dev)  # shape (nnz,)
    D = crow.numel() - 1
    classes = C.shape[1]
    uniform_default_val = 1.0 / classes

    result = torch.empty((D, )).to(device=C.device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    has_mass = (sum_C != 0)
    safe_sum = torch.where(has_mass, sum_C, torch.ones_like(sum_C))
    norm_C = torch.where(has_mass, C / safe_sum, torch.full_like(C, uniform_default_val))
    old_max = torch.where(has_mass, max_C / safe_sum, torch.zeros_like(max_C))

    # norm_C = (C / sum_C)
    # old_max = (max_C / sum_C)

    C_diff = (C - max_C)
    max_C = max_C.squeeze()
    sum_C = sum_C.squeeze()
    C_diff = C_diff.squeeze()
    old_max = old_max.squeeze()
    num_iterations = int(N)
    for row_start in range(0, D, chunk_size):
        row_end = min(row_start + chunk_size, D)
        b = row_end - row_start

        # CSR pointers for the chunk
        starts = crow[row_start:row_end]  # shape (b,)
        ends = crow[row_start + 1: row_end + 1]  # shape (b,)
        lengths = (ends - starts).to(torch.long)  # (b,)

        total_nnz = int(lengths.sum().item())

        if total_nnz == 0:
            # nothing in this chunk, skip
            continue

            # global slice of indices/values for this chunk
        slice_start = int(starts[0].item())
        slice_end = int(ends[-1].item())

        cols_all = ccol[slice_start:slice_end]  # (total_nnz,)
        vals_all = cvals[slice_start:slice_end]  # (total_nnz,)

        # row index for each nnz entry within the chunk: 0..b-1 repeated by lengths
        row_indices = torch.repeat_interleave(torch.arange(b, device=dev, dtype=torch.long),
                                              lengths)  # (total_nnz,)

        # Map chunk row-local indices -> global row indices (if needed)
        global_rows = torch.arange(row_start, row_end, device=dev, dtype=torch.long)  # (b,)

        kvals = vals_all  # (total_nnz, 1)
        sumC_cols = sum_C[cols_all]  # (total_nnz, 1)
        maxC_cols = max_C[cols_all]  # (total_nnz, 1)
        old_max_cols = old_max[cols_all]  # (total_nnz, 1)
        Cdiff_cols = C_diff[cols_all]

        negk = -kvals  # (total_nnz,1)
        # maximum between negk and Cdiff_cols: broadcast negk on classes dimension
        # torch.maximum requires same shape; expand negk to (total_nnz, classes)
        negk_expand = negk.expand(classes, -1).T  # (total_nnz, classes)
        new_state = torch.maximum(negk_expand, Cdiff_cols)  # (total_nnz, classes)

        del negk_expand, Cdiff_cols

        state_add = maxC_cols + kvals  # (total_nnz,1)
        new_state = new_state + state_add.expand(classes, -1).T  # add per-row scalar across classes

        future_sum = (kvals + sumC_cols)  # (total_nnz,1)

        valid_denom = (future_sum != 0)
        safe_future_sum = torch.where(valid_denom, future_sum, torch.ones_like(future_sum))
        safe_denom_expanded = safe_future_sum.expand(classes, -1).T
        mask_expanded = valid_denom.expand(classes, -1).T
        # divide
        new_state = new_state / safe_denom_expanded

        del safe_denom_expanded

        new_state = torch.where(mask_expanded, new_state, torch.zeros_like(new_state))
        # new_state = new_state / future_sum.expand(classes, -1).T
        # subtract old_max (per column)
        new_state = new_state - old_max_cols.expand(classes, -1).T

        # Now apply continuation method
        if cont_method == "positive":
            new_state.clamp_(min=0.0)

        weights_chunk = norm_C[global_rows]  # (b, classes)
        # Now map per-nnz: weights_for_nnz = weights_chunk[row_indices]
        weights_for_nnz = weights_chunk[row_indices]  # (total_nnz, classes)

        # Multiply elementwise and sum over classes -> per-nnz scalar
        per_nnz_weighted = (new_state * weights_for_nnz).sum(dim=1)  # (total_nnz,)

        # Aggregate per row via scatter_add
        chunk_result = torch.zeros((b,), device=dev, dtype=C.dtype)
        chunk_result.scatter_add_(0, row_indices, per_nnz_weighted)



        # result[i:end] = torch.bmm(new_state_vec, weights_batched.unsqueeze_(2)).sum(dim=1).squeeze(1)
        result[row_start:row_end] = chunk_result
        torch.cuda.empty_cache()

    return result