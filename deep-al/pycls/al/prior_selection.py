import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA

# ============ Actual Prior Selection Functions ============ #

def get_temp_priors(rel_measures, lb, ub, mode='log', normalize_input=True, gamma=8.0, threshold=0.5, steepness=20.0):
    
    if normalize_input:
        rmin = rel_measures.min()
        rmax = rel_measures.max()
        
        if rmax - rmin < 1e-9:
            raise ValueError("rel_measures has zero variance; Probably mistake.")
            
        x = (rel_measures - rmin) / (rmax - rmin)
    else:
        x = np.clip(rel_measures, 0.0, 1.0)
    
    if mode == 'linear':
        scaled_priors = ub - x * (ub - lb)
        
    elif mode == 'log':
        if lb <= 0 or ub <= 0:
            raise ValueError("Lower and Upper bounds must be positive for log mode.")
        ratio = lb / ub
        scaled_priors = ub * np.power(ratio, x)
        
    elif mode == 'poly':
        decay_factor = np.power(1.0 - x, gamma)
        scaled_priors = lb + (ub - lb) * decay_factor
        
    elif mode == 'sigmoid':
        sigmoid_input = (x - threshold) * steepness
        sig = 1 / (1 + np.exp(-sigmoid_input))
        scaled_priors = lb + (ub - lb) * (1.0 - sig)
        
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'linear', 'log', 'poly', or 'sigmoid'.")

    return scaled_priors


def get_priors_logistic(clarity_scores, alpha_min=0.01, alpha_max=2.0, c0=0.5, k=20, normalize_input=False, analysis=False):
    
    if normalize_input:
        cmin = clarity_scores.min()
        cmax = clarity_scores.max()
        
        if cmax - cmin < 1e-9:
            raise ValueError("clarity_scores has zero variance; Probably a mistake.")
        
        c = (clarity_scores - cmin) / (cmax - cmin)
    else:
        c = np.clip(clarity_scores, 0.0, 1.0)
    
    def sigmoid_base(x):
        return 1.0 / (1.0 + np.exp(k * (x - c0)))
    
    S_max = sigmoid_base(0.0)
    S_min = sigmoid_base(1.0)
    
    normalized_S = (sigmoid_base(c) - S_min) / (S_max - S_min)
    alpha_priors = alpha_min + (alpha_max - alpha_min) * normalized_S
    
    if analysis:
        _plot_priors_logistic(c, alpha_priors, clarity_scores, alpha_min, alpha_max, c0, k)
    
    return alpha_priors


def _plot_priors_logistic(c_normalized, alpha_priors, clarity_original, alpha_min, alpha_max, c0, k):
    
    c_smooth = np.linspace(0, 1, 300)
    
    def sigmoid_base(x):
        return 1.0 / (1.0 + np.exp(k * (x - c0)))
    
    S_max = sigmoid_base(0.0)
    S_min = sigmoid_base(1.0)
    normalized_S = (sigmoid_base(c_smooth) - S_min) / (S_max - S_min)
    alpha_smooth = alpha_min + (alpha_max - alpha_min) * normalized_S
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    
    ax1.plot(c_smooth, alpha_smooth, color='purple', lw=3.5, label='Sigmoid Prior Mapping', zorder=2)
    
    hb = ax1.hexbin(c_normalized, alpha_priors, gridsize=30, cmap='YlOrRd', mincnt=1, edgecolors='black', linewidths=0.2, zorder=3)
    cbar = plt.colorbar(hb, ax=ax1)
    cbar.set_label('Point Density', fontsize=10)
    
    ax1.axvspan(0.0, 0.3, color='red', alpha=0.08, label='Danger Zone (High α)', zorder=1)
    ax1.axvspan(0.3, 0.8, color='orange', alpha=0.08, label='Transition Zone', zorder=1)
    ax1.axvspan(0.8, 1.0, color='blue', alpha=0.08, label='Safe Zone (Low α)', zorder=1)
    
    ax1.set_title("Phase-Transition Mapping: Clarity to Prior Strength", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Clarity (c)", fontsize=11)
    ax1.set_ylabel("Prior Strength (α)", fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    
    ax2.hist(alpha_priors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(alpha_priors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(alpha_priors):.4f}')
    ax2.axvline(np.median(alpha_priors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(alpha_priors):.4f}')
    
    ax2.set_title("Distribution of Prior Strengths", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Prior Strength (α)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig("clarity_to_alpha_mapping.png", dpi=300)

# ============ Actual Prior Selection Functions ============ #

def proba_gmm(embeddings: torch.Tensor, n_comp_guess: int, rnd_state: int = 0, target_inc_var: float = None):
    
    embeddings = embeddings.numpy()
    
    if target_inc_var is not None:
        pca = PCA(n_components=target_inc_var, random_state=rnd_state, whiten=True)
        X = pca.fit_transform(embeddings)
        print(f"PCA reduced embeddings from {embeddings.shape[1]} to {X.shape[1]} dimensions.")
    
    gmm = GaussianMixture(
                    n_components=int(n_comp_guess * 1.2),
                    covariance_type='diag',
                    random_state=rnd_state,
                    reg_covar=1e-2,
                    n_init=1,
                    verbose=1
                            )
    
    print(f"Fitting Regular GMM on {X.shape[0]} samples...")
    gmm.fit(X)
    probs = gmm.predict_proba(X)
    active_components = np.sum(gmm.weights_ > 1e-3)
    print(f"Regular GMM fitting complete. Active components: {active_components}/{int(n_comp_guess * 1.2)}")
    
    return probs


def gmm_clarity(data, n_clusters, rnd_state=0, pca_dim=None, target_inc_var=None, baysian=False):
    if baysian:
        probs = baysian_proba_gmm(data, n_clusters, rnd_state, target_inc_var=target_inc_var)
    else:
        probs = proba_gmm(data, n_clusters, rnd_state, target_inc_var=target_inc_var)
    return 1.0 - compute_matrix_entropy(torch.as_tensor(probs), normalize=True)


def oracle_clarity_K(K, y, M, batch_size=8192, matrix_peakedness=None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = K.shape[0]
    
    K.fill_diagonal_(0.0)

    y_tensor = torch.as_tensor(y, device=device, dtype=torch.int64).view(-1)
    Y_onehot = torch.zeros((N, M), device=device, dtype=K.dtype)
    Y_onehot.scatter_(1, y_tensor.unsqueeze(1), 1.0)
    unif_prior = 0.01

    scores = []

    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        
        K_batch = K[i:end_idx, :].to(device)
        
        C_batch = torch.mm(K_batch, Y_onehot) + unif_prior
        norm_C_batch = C_batch / C_batch.sum(dim=1, keepdim=True)
        
        clarity_batch = matrix_peakedness(norm_C_batch)
        
        scores.append(clarity_batch.cpu())
        
        del K_batch, C_batch, norm_C_batch, clarity_batch

    return torch.cat(scores)


def oracle_clarity_K_sparse(K, y, M, matrix_peakedness):
    
    device = K.device
    N = K.shape[0]
    unif_prior = 0.1
    
    y_tensor = torch.as_tensor(y, device=device, dtype=torch.int64).view(-1)
    Y_onehot = torch.zeros((N, M), device=device, dtype=torch.float32)
    Y_onehot.scatter_(1, y_tensor.unsqueeze(1), 1.0)
    
    with torch.no_grad():
        K_float32 = K.float()
        total_evidence = torch.sparse.mm(K_float32, Y_onehot)
        external_evidence = total_evidence - Y_onehot
        C = external_evidence + unif_prior
        norm_C = C / C.sum(dim=1, keepdim=True)
        clarity_scores = matrix_peakedness(norm_C)
        
    return clarity_scores.cpu()


def slice_csr_matrix(csr_tensor, start_row, end_row):
    
    crow = csr_tensor.crow_indices()
    col = csr_tensor.col_indices()
    val = csr_tensor.values()
    
    p_start = crow[start_row]
    p_end = crow[end_row]
    
    new_crow = crow[start_row : end_row + 1] - p_start
    new_col = col[p_start : p_end]
    new_val = val[p_start : p_end]
    
    new_shape = (end_row - start_row, csr_tensor.shape[1])
    return torch.sparse_csr_tensor(
        new_crow, new_col, new_val, 
        size=new_shape, 
        device=csr_tensor.device
    )


def compute_reliability(K, y, batch_size=8192, normalized=False):

    if not torch.is_tensor(K):
        raise ValueError("K must be a torch tensor")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = K.shape[0]

    y_t = torch.as_tensor(y, device=device).view(-1)

    scores = []

    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        K_batch = K[i:end_idx, :].to(device)
        y_batch = y_t[i:end_idx]

        agree_mask = (y_batch[:, None] == y_t[None, :])
        disagree_mask = ~agree_mask

        agree = (K_batch * agree_mask).sum(dim=1)
        disagree = (K_batch * disagree_mask).sum(dim=1)

        if normalized:
            same_class_counts = agree_mask.sum(dim=1)
            diff_class_counts = disagree_mask.sum(dim=1)
            rel = (agree / torch.clamp(same_class_counts, min=1.0) -
                   disagree / torch.clamp(diff_class_counts, min=1.0))
        else:
            rel = agree - disagree

        scores.append(rel.cpu())

        del K_batch, y_batch, agree_mask, disagree_mask, agree, disagree, rel

    return torch.cat(scores)


def baysian_proba_gmm(embeddings: torch.Tensor, n_comp_guess: int, rnd_state: int = 0, target_inc_var: float = None):
    
    embeddings = embeddings.numpy()
    
    if target_inc_var is not None:
        pca = PCA(n_components=target_inc_var, random_state=rnd_state, whiten=True)
        X = pca.fit_transform(embeddings)
        print(f"PCA reduced embeddings from {embeddings.shape[1]} to {X.shape[1]} dimensions.")
    
    bgmm = BayesianGaussianMixture(
                n_components=int(n_comp_guess * 1.2),
                covariance_type='diag',
                weight_concentration_prior_type='dirichlet_distribution',
                weight_concentration_prior=100.0,
                mean_precision_prior=1e-9,
                max_iter=500,
                n_init=1,
                random_state=rnd_state,
                reg_covar=1e-4,
                verbose=1)
    
    print(f"Fitting Bayesian GMM on {X.shape[0]} samples...")
    bgmm.fit(X)
    probs = bgmm.predict_proba(X)
    active_components = np.sum(bgmm.weights_ > 1e-3)
    print(f"Bayesian GMM fitting complete. Active components: {active_components}/{n_comp_guess}")
    
    return probs

# ============== Estimators for Clarity\Representativeness ============ #

def cohesion_sparse(K_sparse, batch_size=128, rank_normalize=False):
    
    n_points = K_sparse.shape[0]
    
    K_transpose_csc = K_sparse.transpose(0, 1).to_sparse_csc()
    
    if K_transpose_csc.dtype == torch.float16:
        K_transpose_csc = K_transpose_csc.float()

    cohesion_scores = torch.zeros(n_points, dtype=torch.float32, device='cpu')

    for start_idx in range(0, n_points, batch_size):
        end_idx = min(start_idx + batch_size, n_points)

        K_batch_csr = slice_csr_matrix(K_sparse, start_idx, end_idx)
        V = (K_batch_csr.to_dense() != 0).float()
        
        degrees = V.sum(dim=1)
        
        V_T = V.t()
        Temp_T = torch.sparse.mm(K_transpose_csc, V_T)
        Temp = Temp_T.t()
        
        subgraph_weight = (Temp * V).sum(dim=1)

        normalization = torch.clamp(degrees * degrees, min=1.0)
        batch_scores = subgraph_weight / normalization
        
        cohesion_scores[start_idx:end_idx] = batch_scores.cpu()
        
        del V, V_T, Temp, Temp_T, K_batch_csr

    if rank_normalize:
        ranks = torch.argsort(torch.argsort(cohesion_scores))
        cohesion_scores = ranks.float() / (n_points - 1)
        
    return cohesion_scores

    
def calculate_cohesion(K_cpu, k, device='cuda', batch_size=128,):
    
    n_points = K_cpu.shape[0]

    if k >= n_points:
        raise ValueError("k must be smaller than the number of points N.")

    cohesion_cpu = torch.zeros(n_points, device='cpu')
    
    print("Pre-calculating neighbor indices on CPU...")
    top_k_indices_all = torch.topk(K_cpu, k+1, dim=0).indices[1:, :]
    print("Done.")

    for i in range(0, n_points, batch_size):
        batch_end = min(i + batch_size, n_points)

        top_k_indices_batch_cpu = top_k_indices_all[:, i:batch_end]

        row_indices_cpu = torch.repeat_interleave(top_k_indices_batch_cpu, k, dim=0)
        col_indices_cpu = top_k_indices_batch_cpu.tile((k, 1))

        unique_cols_cpu, inverse_indices = torch.unique(col_indices_cpu, return_inverse=True)

        K_slice_gpu = K_cpu[:, unique_cols_cpu].to(device, dtype=torch.float32)

        row_indices_gpu = row_indices_cpu.to(device)
        col_indices_remapped_gpu = inverse_indices.to(device)
        
        submatrix_values = K_slice_gpu[row_indices_gpu, col_indices_remapped_gpu]
        
        submatrix_values = submatrix_values.view(k*k, -1)

        cohesion_batch_gpu = torch.sum(submatrix_values, dim=0) / (k * k)
        
        cohesion_cpu[i:batch_end] = cohesion_batch_gpu.to('cpu')
        del K_slice_gpu, row_indices_gpu, col_indices_remapped_gpu, submatrix_values, cohesion_batch_gpu
        torch.cuda.empty_cache()

    return cohesion_cpu


def sum_K_sparse_estimator(sparse_K):
    
    with torch.no_grad():
        ones = torch.ones((sparse_K.shape[1], 1), device=sparse_K.device, dtype=sparse_K.dtype)
        sums = torch.sparse.mm(sparse_K, ones).squeeze()
    return sums.cpu()


def est_reliability(K, transformation, batch_size=8192, p=2, a=1.0):
    
    eps = 1e-5
    if torch.is_tensor(K):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n = K.shape[0]
        scores = []
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            K_batch = K[i:end_idx, :].to(device)
            if transformation == 'reg':
                transformed_K = K_batch
            elif transformation == 'exp':
                pass
            elif transformation == 'pow':
                transformed_K = K_batch ** p
            else:
                raise ValueError(f"Unknown transformation: {transformation}")
            rel_batch = transformed_K.sum(dim=1)
            scores.append(rel_batch.cpu())
            
            del K_batch, transformed_K, rel_batch

        return torch.cat(scores)
    else:
        raise ValueError("K must be a torch tensor")
    
    
def lid_clarity_estimate(K_sparse, epsilon=1e-6):
    
    N = K_sparse.shape[0]
    k = K_sparse.values().shape[0] // N
    sim_values = K_sparse.values().view(N, k).detach()
    
    dists = torch.clamp(1.0 - sim_values, min=0.0)
    sorted_dists, _ = torch.sort(dists, dim=1)
    
    neighbor_dists = torch.clamp(sorted_dists[:, 1:], min=epsilon)
    
    r_max = neighbor_dists[:, -1].unsqueeze(1)
    log_ratios = torch.log(neighbor_dists / torch.clamp(r_max, min=epsilon))
    mean_log_ratios = torch.mean(log_ratios, dim=1)
    
    raw_lid = -1.0 / (mean_log_ratios - epsilon)
    clarity_scores = -1.0 * raw_lid
    
    return clarity_scores.cpu()

# ============== Matrix Peakedness Implementations ============ #

def compute_entropy_peakedness(A):
    
    entropy = -torch.sum(A * torch.log(A + 1e-9), dim=1)
    max_entropy = math.log(A.size(1))
    normalized_entropy = entropy / max_entropy
    return 1.0 - normalized_entropy

def compute_margin_peakedness(A):
    
    top2 = torch.topk(A, 2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return margin

def compute_max_peakedness(A):
    
    max_vals = torch.max(A, dim=1).values
    return max_vals