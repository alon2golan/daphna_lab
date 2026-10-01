"""Selected unchanged author components; see PROVENANCE.md and LICENSE.upstream."""
import numpy as np
import scipy.sparse as sp
from torch import nn

def _apply_degree_normalization_sparse(csr_matrix, alpha, degree_normalization_method='none'):
    """
    Apply degree normalization to a sparse CSR matrix.
    - 'none': K_ij <- K_ij / (degree_i * degree_j)^(alpha/2)
    - 'sum' or 'max': K_ij <- K_ij / (normalized_degree_i * normalized_degree_j)
    - 'log': K_ij <- K_ij / log(degree_i * degree_j)

    Degrees are computed excluding diagonal elements.
    Degrees can be optionally normalized before applying the transformation.
    If degree_i or degree_j is 0, K_ij is set to 0.

    Args:
        csr_matrix: Sparse CSR matrix (N, N)
        alpha: Normalization parameter (only used for 'none' method)
        degree_normalization_method: Method to normalize degrees ('none', 'sum', 'max', 'log')

    Returns:
        Normalized sparse CSR matrix
    """
    csr_no_diag = csr_matrix.copy()
    csr_no_diag.setdiag(0)
    csr_no_diag.eliminate_zeros()

    degrees = np.asarray(csr_no_diag.sum(axis=1)).flatten()

    # Normalize degrees based on the specified method
    if degree_normalization_method == 'sum':
        degree_sum = degrees.sum()
        if degree_sum > 0:
            degrees = degrees / degree_sum
    elif degree_normalization_method == 'max':
        degree_max = degrees.max()
        if degree_max > 0:
            degrees = degrees / degree_max
    # else: 'none' or 'log' - no pre-normalization

    coo = csr_matrix.tocoo()

    row_degrees = degrees[coo.row]
    col_degrees = degrees[coo.col]

    degree_product = row_degrees * col_degrees

    zero_mask = (degree_product == 0)

    degree_product_safe = degree_product.copy()
    degree_product_safe[zero_mask] = 1.0

    # Apply transformation based on method
    if degree_normalization_method == 'none':
        normalization_factor = np.power(degree_product_safe, alpha / 2.0)
    elif degree_normalization_method == 'log':
        normalization_factor = np.power(np.log(1 + degree_product_safe), 0.5)
    else:  # 'sum' or 'max'
        normalization_factor = degree_product_safe

    normalized_data = coo.data / normalization_factor
    normalized_data[zero_mask] = 0.0

    normalized_coo = sp.coo_matrix(
        (normalized_data, (coo.row, coo.col)),
        shape=csr_matrix.shape
    )

    return normalized_coo.tocsr()

class MLPClassifierTorch(nn.Module):
    """
    PyTorch MLP for classification matching sklearn's MLPClassifier behavior.

    This is a simplified version that can be used as an nn.Module for training
    with custom loss functions and distillation.

    For full sklearn API compatibility, use the MLPClassifier from tools.comapre_archs.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, activation='relu'):
        super(MLPClassifierTorch, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Build network matching sklearn's architecture
        layers = []
        # Support for tuple of hidden layer sizes (like sklearn)
        if isinstance(hidden_dim, (tuple, list)):
            in_size = input_dim
            for h in hidden_dim:
                layers.append(nn.Linear(in_size, h))
                layers.append(self._get_activation(activation))
                in_size = h
            layers.append(nn.Linear(in_size, num_classes))
        else:
            # Single hidden layer (default)
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Linear(hidden_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation):
        """Get activation function matching sklearn's options."""
        mapping = {
            'identity': nn.Identity(),
            'logistic': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
        }
        if activation not in mapping:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(mapping.keys())}.")
        return mapping[activation]

    def forward(self, x):
        return self.network(x)
