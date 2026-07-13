import math
import torch

# Used by BAYES_MISP's LOCAL_ALPHA init path (per-point priors from K's class clarity).


def get_temp_priors(rel_measures, lb, ub):
    """
    Create per-point prior values scaled between lb and ub based on reliability
    measures: max reliability -> min prior (lb), min reliability -> max prior (ub).
    """
    rel_min = rel_measures.min()
    rel_max = rel_measures.max()
    scaled_priors = ub - (rel_measures - rel_min) * (ub - lb) / (rel_max - rel_min)
    return scaled_priors


def compute_matrix_entropy(A, normalize=True):
    """
    Computes entropy for each row of the matrix and returns the scores in order.

    input: A torch tensor (N, M)
    output: A torch tensor (N,)
    """
    entropy = -torch.sum(A * torch.log(A + 1e-9), dim=1)
    if normalize:
        max_entropy = math.log(A.size(1))
        entropy = entropy / max_entropy
    return entropy


def compute_clarity_kp(K, y, M, oracle_method, batch_size=8192):
    """
    For each point, compute its per-class evidence vector from K (as if the full
    labeled set contributed), then score clarity as either 1 - normalized entropy
    or the max class share of that vector.

    M is the number of classes. K is a torch tensor on CPU. y is the labels array.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = K.shape[0]

    y_tensor = torch.as_tensor(y, device=device, dtype=torch.int64).view(-1)
    Y_onehot = torch.zeros((N, M), device=device, dtype=K.dtype)
    Y_onehot.scatter_(1, y_tensor.unsqueeze(1), 1.0)

    scores = []

    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)

        K_batch = K[i:end_idx, :].to(device)  # (b, N)

        # NOTE: +0.1 uniform prior for mathematical formalization; negligible when N is large.
        C_batch = torch.mm(K_batch, Y_onehot) + 0.1
        norm_C_batch = C_batch / C_batch.sum(dim=1, keepdim=True)

        if oracle_method == 'entropy':
            clarity_batch = 1 - compute_matrix_entropy(norm_C_batch, normalize=True)
        elif oracle_method == 'max':
            clarity_batch = torch.max(norm_C_batch, axis=1)[0]
        else:
            raise ValueError(f"Unknown oracle method: {oracle_method}")

        scores.append(clarity_batch.cpu())

        del K_batch, C_batch, norm_C_batch, clarity_batch

    return torch.cat(scores)
