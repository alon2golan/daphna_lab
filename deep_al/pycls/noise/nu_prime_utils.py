"""
Utilities for Equilibrium Noise Rate (ν') estimation.

This module provides functions for:
- Loading/saving ν' values
- Feature extraction from models
- Pseudo-label generation via clustering
- Noise injection for simulation
- Lightweight training for calibration
"""

import json
import numpy as np
import torch
from sklearn.cluster import KMeans
import deep_al.pycls.utils.logging as lu

logger = lu.get_logger(__name__)


def load_nu_prime(filepath):
    """Load equilibrium noise rate from JSON file.
    
    Args:
        filepath: Path to JSON file containing nu_prime
        
    Returns:
        tuple: (nu_prime, metadata_dict) where metadata contains budget, costs, etc.
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'nu_prime' not in data:
            raise ValueError(f"Invalid nu_prime file format: missing 'nu_prime' key")
        
        nu_prime = float(data['nu_prime'])
        metadata = data.get('metadata', {})
        
        logger.info(f"Loaded ν' = {nu_prime} from {filepath}")
        
        if metadata:
            logger.info(f"Metadata: {metadata}")
        
        return nu_prime, metadata
        
    except FileNotFoundError:
        raise FileNotFoundError(f"ν' file not found: {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in: {filepath}")


def save_nu_prime(nu_prime, filepath, metadata=None):
    """Save equilibrium noise rate to JSON file.
    
    Args:
        nu_prime: The equilibrium noise rate value
        filepath: Output file path
        metadata: Optional dict with additional information
    """
    data = {
        'nu_prime': float(nu_prime),
        'metadata': metadata or {}
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved ν' = {nu_prime} to {filepath}")


@torch.no_grad()
def extract_features(model, dataloader, device='cuda', max_samples=None):
    """Extract feature embeddings from a model.
    
    Args:
        model: PyTorch model with a feature extraction capability
        dataloader: DataLoader for the dataset
        device: Device to run on ('cuda' or 'cpu')
        max_samples: Maximum number of samples to process (None for all)
        
    Returns:
        tuple: (features_array, labels_array, indices_array)
    """
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
    
    model.eval()
    
    all_features = []
    all_labels = []
    all_indices = []
    
    sample_count = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if max_samples and sample_count >= max_samples:
            break
        
        if torch.cuda.is_available() and device == 'cuda':
            inputs = inputs.cuda()
        
        inputs = inputs.type(torch.FloatTensor if device == 'cpu' else torch.cuda.FloatTensor)
        
        # Extract features (assume model has a feature extraction mode)
        # Try to get features before final classification layer
        try:
            # Try common attribute names
            if hasattr(model, 'features'):
                features = model.features(inputs)
            elif hasattr(model, 'forward_features'):
                features = model.forward_features(inputs)
            elif hasattr(model, 'extract_features'):
                features = model.extract_features(inputs)
            else:
                # Fallback: use the model output as features
                features = model(inputs)
            
            # Flatten if needed
            if len(features.shape) > 2:
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            
        except Exception as e:
            logger.warning(f"Feature extraction method failed: {e}. Using model output.")
            features = model(inputs)
        
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
        
        # Track indices
        batch_size = len(labels)
        batch_indices = np.arange(sample_count, sample_count + batch_size)
        all_indices.append(batch_indices)
        
        sample_count += batch_size
    
    features_array = np.vstack(all_features)
    labels_array = np.concatenate(all_labels)
    indices_array = np.concatenate(all_indices)
    
    logger.info(f"Extracted features: shape={features_array.shape}")
    
    return features_array, labels_array, indices_array


def get_pseudo_labels_kmeans(features, num_classes, seed=42, normalize=True):
    """Generate pseudo-labels using K-Means clustering.
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        num_classes: Number of clusters (classes)
        seed: Random seed for reproducibility
        normalize: Whether to normalize features before clustering (L2 norm)
        
    Returns:
        np.ndarray: Cluster assignments as pseudo-labels
    """
    logger.info(f"Generating pseudo-labels using K-Means (k={num_classes})")
    
    if normalize:
        # Use L2 normalization (important for SimCLR features)
        features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    else:
        features_normalized = features
    
    from tqdm import tqdm
    import sys
    n_init = 10
    best_inertia = np.inf
    best_labels  = None
    pbar = tqdm(range(n_init), desc=f"K-Means (k={num_classes})", unit="init", file=sys.stdout)
    for i in pbar:
        km = KMeans(n_clusters=num_classes, random_state=seed + i, n_init=1, max_iter=300)
        km.fit(features_normalized)
        pbar.set_postfix(inertia=f"{km.inertia_:.1f}")
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_labels  = km.labels_
    pseudo_labels = best_labels

    logger.info(f"K-Means clustering complete. Label distribution: {np.bincount(pseudo_labels)}")
    
    return pseudo_labels


def add_symmetric_noise(labels, noise_rate, num_classes, seed=42):
    """Add symmetric label noise to labels.
    
    Args:
        labels: Original labels (numpy array or list)
        noise_rate: Fraction of labels to flip (0.0 to 1.0)
        num_classes: Number of classes for random flipping
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Noisy labels
    """
    labels = np.array(labels)
    n_samples = len(labels)
    n_noisy = int(noise_rate * n_samples)
    
    rng = np.random.RandomState(seed)
    
    # Select samples to flip
    noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)
    
    # Create noisy labels
    noisy_labels = labels.copy()
    
    for idx in noisy_indices:
        # Randomly select a different class
        original_class = labels[idx]
        possible_classes = [c for c in range(num_classes) if c != original_class]
        noisy_labels[idx] = rng.choice(possible_classes)
    
    noise_actual = (noisy_labels != labels).sum() / n_samples
    logger.info(f"Added symmetric noise: target={noise_rate:.3f}, actual={noise_actual:.3f}")
    
    return noisy_labels


def add_asymmetric_noise(labels, noise_rate, num_classes, seed=42):
    """Add asymmetric label noise (class-dependent confusion).
    
    For example in CIFAR-10: truck->automobile, bird->airplane, deer->horse, cat->dog
    
    Args:
        labels: Original labels
        noise_rate: Fraction of labels to flip
        num_classes: Number of classes
        seed: Random seed
        
    Returns:
        np.ndarray: Noisy labels
    """
    labels = np.array(labels)
    n_samples = len(labels)
    
    rng = np.random.RandomState(seed)
    
    # Create confusion pairs (cycle: 0->1->2->...->0)
    confusion_pairs = {i: (i + 1) % num_classes for i in range(num_classes)}
    
    noisy_labels = labels.copy()
    
    for i in range(n_samples):
        if rng.rand() < noise_rate:
            noisy_labels[i] = confusion_pairs[labels[i]]
    
    noise_actual = (noisy_labels != labels).sum() / n_samples
    logger.info(f"Added asymmetric noise: target={noise_rate:.3f}, actual={noise_actual:.3f}")
    
    return noisy_labels


def simple_train_epoch(model, train_loader, optimizer, loss_fn, device='cuda'):
    """Train model for one epoch (lightweight version).
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        
    Returns:
        float: Average training loss
    """
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
    
    model.train()
    
    total_loss = 0.0
    n_batches = 0
    
    for inputs, labels in train_loader:
        if torch.cuda.is_available() and device == 'cuda':
            inputs, labels = inputs.cuda(), labels.cuda()
        
        inputs = inputs.type(torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor)
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    return avg_loss


@torch.no_grad()
def simple_eval_epoch(model, eval_loader, device='cuda'):
    """Evaluate model (lightweight version).
    
    Args:
        model: PyTorch model
        eval_loader: Evaluation data loader
        device: Device to evaluate on
        
    Returns:
        float: Accuracy (0-100)
    """
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
    
    model.eval()
    
    correct = 0
    total = 0
    
    for inputs, labels in eval_loader:
        if torch.cuda.is_available() and device == 'cuda':
            inputs, labels = inputs.cuda(), labels.cuda()
        
        inputs = inputs.type(torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return accuracy
