#!/usr/bin/env python3
"""
Estimate Equilibrium Noise Rate (ν') for Active Learning with Noisy Annotators.

This script implements Algorithm 1 from the paper: a budget-aware comparison between
oracle and noisy annotator strategies to find the equilibrium noise rate.

Usage:
    python estimate_nu_prime.py \\
        --cfg configs/cifar10/al/config.yaml \\
        --output results/nu_prime_cifar10.json \\
        --budget 1000 \\
        --oracle_cost 10.0 \\
        --noisy_cost 1.0

Algorithm 1: Estimating Equilibrium Noise Rate ν'
    1. Generate pseudo-labels via K-Means clustering
    2. Train oracle baseline: sample B/$o samples, train R times, compute average accuracy
    3. Binary search for ν':
       - For each candidate noise rate ν_mid:
         * Sample B/$n samples (more samples with cheaper annotator)
         * Inject noise at rate ν_mid
         * Train R times, compute average accuracy
         * Compare: if Acc[oracle] > Acc[noisy], noise too high; else acceptable
    4. Return equilibrium point where both strategies have equal performance
    
ν' represents the noise rate where: Oracle(B/$o samples) ≈ Noisy(B/$n samples, noise=ν')
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Ensure repo root is in sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from deep_al.pycls.core.config import cfg
from deep_al.pycls.datasets.data import Data
import deep_al.pycls.core.losses as losses
import deep_al.pycls.core.optimizer as optim
import deep_al.pycls.utils.logging as lu
import deep_al.pycls.utils.model_handler as mh
from deep_al.pycls.noise.nu_prime_utils import (
    extract_features,
    get_pseudo_labels_kmeans,
    add_symmetric_noise,
    add_asymmetric_noise,
    simple_train_epoch,
    simple_eval_epoch,
    save_nu_prime
)
from deep_al.pycls.datasets.utils.features import load_features

logger = lu.get_logger(__name__)


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Estimate Equilibrium Noise Rate (ν\') for Active Learning (Algorithm 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate ν' for CIFAR-10 with budget 1000, oracle costs 10x more than noisy
  python estimate_nu_prime.py \\
      --cfg configs/cifar10/al/config.yaml \\
      --output nu_prime_cifar10.json \\
      --budget 1000 \\
      --oracle_cost 10.0 \\
      --noisy_cost 1.0

  # Custom parameters with more repeats
  python estimate_nu_prime.py \\
      --cfg configs/cifar100/al/config.yaml \\
      --output results/nu_prime.json \\
      --budget 5000 \\
      --oracle_cost 10.0 \\
      --noisy_cost 1.0 \\
      --repeats 5 \\
      --epsilon 0.03
        """
    )
    
    parser.add_argument('--cfg', dest='cfg_file', required=True, type=str,
                        help='Path to config file')
    parser.add_argument('--output', required=True, type=str,
                        help='Output path for nu_prime JSON file')
    
    # Budget and cost parameters (REQUIRED for Algorithm 1)
    parser.add_argument('--budget', type=float, required=True,
                        help='Query budget B (total labeling budget)')
    parser.add_argument('--oracle_cost', type=float, required=True,
                        help='Cost per oracle label $o (e.g., 10.0)')
    parser.add_argument('--noisy_cost', type=float, required=True,
                        help='Cost per noisy label $n (e.g., 1.0)')
    
    # Algorithm parameters
    parser.add_argument('--repeats', type=int, default=3,
                        help='Number of training repeats R for averaging (default: 3)')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Binary search tolerance (default: 0.05)')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='Maximum training epochs per repeat (default: 5)')
    parser.add_argument('--noise_type', choices=['sym', 'asym'], default='sym',
                        help='Type of noise to simulate: symmetric or asymmetric (default: sym)')
    
    # Dataset parameters
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to use for estimation (None = all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: use cfg.TRAIN.BATCH_SIZE)')
    parser.add_argument('--use_simclr_features', action='store_true',
                        help='Use pre-extracted SimCLR features from representations_bank')
    
    return parser


class DatasetWithLabelMap(torch.utils.data.Dataset):
    """Wraps a dataset subset and replaces labels using a provided mapping."""
    
    def __init__(self, subset, label_map, global_indices):
        self.subset = subset
        self.label_map = label_map
        self.global_indices = global_indices
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img, _ = self.subset[idx]
        global_idx = self.global_indices[idx]
        new_label = self.label_map[global_idx]
        return img, new_label


class SimpleFeatureDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for features and labels."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def estimate_nu_prime_binary_search(
    features, 
    true_labels,
    num_classes, 
    cfg,
    dataset,
    budget,
    oracle_cost,
    noisy_cost,
    repeats=3,
    epsilon=0.05, 
    max_epochs=5,
    noise_type='sym',
    seed=42,
    pseudo_labels=None
):
    """
    Estimate equilibrium noise rate using Algorithm 1 from the paper.
    
    Algorithm: Budget-Aware Oracle vs Noisy Comparison
        1. Generate pseudo-labels from features using K-Means
        2. Split into train/test sets
        3. Compute Oracle Baseline:
           - Sample |Do| = B/$o samples (fewer, expensive)
           - Train R times, compute average accuracy Acc[Mo]
        4. Binary Search for ν':
           - For each candidate noise rate ν_mid:
             * Sample |Dn| = B/$n samples (more, cheaper)
             * Inject noise at rate ν_mid
             * Train R times, compute average accuracy Acc[Mn]
             * if Acc[Mo] > Acc[Mn]: ν_high = ν_mid (noise too high)
             * else: ν_low = ν_mid (noise acceptable)
        5. Return ν' = (ν_low + ν_high) / 2
    
    Args:
        features: Feature embeddings (used only for K-Means pseudo-labels)
        true_labels: True labels (for test set evaluation only)
        num_classes: Number of classes
        cfg: Configuration object
        dataset: Original image dataset
        budget: Total labeling budget B
        oracle_cost: Cost per oracle label $o
        noisy_cost: Cost per noisy label $n
        repeats: Number of training repeats R for averaging
        epsilon: Binary search tolerance
        max_epochs: Training epochs per repeat
        noise_type: 'sym' or 'asym'
        seed: Random seed
        
    Returns:
        float: Estimated ν' (equilibrium noise rate)
    """
    logger.info("=" * 70)
    logger.info("Algorithm 1: Estimating Equilibrium Noise Rate (ν')")
    logger.info("=" * 70)
    logger.info(f"Budget: {budget}, Oracle Cost: ${oracle_cost}, Noisy Cost: ${noisy_cost}")
    
    # Calculate sample sizes based on budget
    n_samples_oracle = int(budget / oracle_cost)
    n_samples_noisy = int(budget / noisy_cost)
    
    logger.info(f"Oracle Strategy: {n_samples_oracle} samples (expensive, clean)")
    logger.info(f"Noisy Strategy:  {n_samples_noisy} samples (cheap, will add noise)")
    logger.info("=" * 70)
    
    # Step 1: Generate pseudo-labels using K-Means (if not provided)
    if pseudo_labels is None:
        logger.info("\n[Step 1] Generating pseudo-labels via K-Means clustering...")
        pseudo_labels = get_pseudo_labels_kmeans(features, num_classes, seed=seed)
    else:
        logger.info("\n[Step 1] Using provided pseudo-labels")
    
    # Step 2: Train/Test Split
    n_total = len(features)
    n_train = int(0.8 * n_total)
    
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    pseudo_labels_train = pseudo_labels[train_indices]
    pseudo_labels_test = pseudo_labels[test_indices]  # Use pseudo-labels for consistent evaluation
    
    logger.info(f"Split: {len(train_indices)} train, {len(test_indices)} test")
    
    # Step 3: Compute Oracle Baseline
    logger.info(f"\n[Step 2] Computing Oracle Baseline ({repeats} repeats)...")
    logger.info(f"  Training: {n_samples_oracle} samples (clean pseudo-labels)")
    logger.info(f"  Testing: {len(test_indices)} samples (pseudo-labels)")
    oracle_accuracies = []
    
    for r in range(repeats):
        logger.info(f"  Oracle repeat {r+1}/{repeats}...")
        
        # Sample Do ⊂ Dtrain, |Do| = B/$o
        sample_indices = rng.choice(len(train_indices), size=min(n_samples_oracle, len(train_indices)), replace=False)
        sampled_train_indices = train_indices[sample_indices]
        sampled_labels = pseudo_labels_train[sample_indices]
        
        # Train model on Do
        acc = train_and_evaluate(
            cfg, dataset, 
            sampled_train_indices, sampled_labels,
            test_indices, pseudo_labels_test,
            max_epochs, seed + r
        )
        oracle_accuracies.append(acc)
        logger.info(f"    → Accuracy on pseudo-labels (test set): {acc:.2f}%")
    
    acc_oracle = np.mean(oracle_accuracies)
    logger.info(f"\n  Oracle Baseline: Acc[Mo] = {acc_oracle:.2f}% (avg over {repeats} repeats)")
    logger.info(f"  [Trained on {n_samples_oracle} clean pseudo-labels, evaluated on {len(test_indices)} pseudo-labels]")
    
    # Step 4: Binary Search for ν'
    low = 0.0
    high = 1.0
    iteration = 0
    
    noise_add_fn = add_symmetric_noise if noise_type == 'sym' else add_asymmetric_noise
    
    logger.info(f"\n[Step 3] Binary Search for ν' (epsilon={epsilon})")
    logger.info(f"Target: Find ν where Acc[Mn] ≈ Acc[Mo] = {acc_oracle:.2f}%\n")
    
    while high - low > epsilon:
        iteration += 1
        mid = (low + high) / 2.0
        
        logger.info(f"Iteration {iteration}: Testing ν = {mid:.4f} (range: [{low:.4f}, {high:.4f}])")
        logger.info(f"  Training: {n_samples_noisy} samples (pseudo-labels with {mid:.1%} noise)")
        logger.info(f"  Testing: {len(test_indices)} samples (pseudo-labels)")
        
        # Test noisy annotator strategy at noise rate ν_mid
        noisy_accuracies = []
        
        for r in range(repeats):
            logger.info(f"  Noisy repeat {r+1}/{repeats} with ν={mid:.4f}...")
            
            # Sample Dn ⊂ Dtrain, |Dn| = B/$n (MORE samples than oracle)
            sample_indices = rng.choice(len(train_indices), size=min(n_samples_noisy, len(train_indices)), replace=False)
            sampled_train_indices = train_indices[sample_indices]
            sampled_pseudo_labels = pseudo_labels_train[sample_indices]
            
            # Inject noise at rate ν_mid
            noisy_labels = noise_add_fn(
                sampled_pseudo_labels,
                noise_rate=mid,
                num_classes=num_classes,
                seed=seed + iteration * 100 + r
            )
            
            # Train model on Dn with noisy labels
            acc = train_and_evaluate(
                cfg, dataset,
                sampled_train_indices, noisy_labels,
                test_indices, pseudo_labels_test,
                max_epochs, seed + iteration * 100 + r
            )
            noisy_accuracies.append(acc)
            logger.info(f"    → Accuracy on pseudo-labels (test set): {acc:.2f}%")
        
        acc_noisy = np.mean(noisy_accuracies)
        logger.info(f"  Noisy (ν={mid:.4f}): Acc[Mn] = {acc_noisy:.2f}% (avg over {repeats} repeats)")
        logger.info(f"  [Trained on {n_samples_noisy} noisy pseudo-labels, evaluated on {len(test_indices)} pseudo-labels]")
        
        # Compare: Oracle vs Noisy
        if acc_oracle > acc_noisy:
            # Oracle is better → noise too high
            high = mid
            logger.info(f"  → Acc[Mo]={acc_oracle:.2f}% > Acc[Mn]={acc_noisy:.2f}%: Noise TOO HIGH, search lower\n")
        else:
            # Noisy is competitive → can handle this noise
            low = mid
            logger.info(f"  → Acc[Mo]={acc_oracle:.2f}% ≤ Acc[Mn]={acc_noisy:.2f}%: Noise ACCEPTABLE, search higher\n")
    
    nu_prime = (low + high) / 2.0
    
    logger.info("=" * 70)
    logger.info(f"Binary Search Complete!")
    logger.info(f"Estimated ν' = {nu_prime:.4f}")
    logger.info(f"Final range: [{low:.4f}, {high:.4f}]")
    logger.info(f"\nInterpretation:")
    logger.info(f"  If annotator noise < {nu_prime:.4f}: Use Noisy Annotator (cost-effective)")
    logger.info(f"  If annotator noise > {nu_prime:.4f}: Use Oracle (quality over quantity)")
    logger.info("=" * 70)
    
    return nu_prime


def train_and_evaluate(cfg, dataset, train_indices, labels, test_indices, test_labels, max_epochs, seed):
    """
    Train a model on given samples and evaluate on test set.
    
    Args:
        cfg: Configuration object
        dataset: Original dataset
        train_indices: Indices for training samples
        labels: Labels for training (may be pseudo or noisy)
        test_indices: Indices for test samples
        test_labels: True labels for test set
        max_epochs: Number of training epochs
        seed: Random seed
        
    Returns:
        float: Test accuracy percentage
    """
    # Create datasets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    label_map_train = {idx: int(label) for idx, label in zip(train_indices, labels)}
    label_map_test = {idx: int(label) for idx, label in zip(test_indices, test_labels)}
    
    train_dataset = DatasetWithLabelMap(train_subset, label_map_train, train_indices)
    test_dataset = DatasetWithLabelMap(test_subset, label_map_test, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize model
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    model = mh.get_model(cfg)
    optimizer_obj = optim.construct_optimizer(cfg, model)
    loss_fn = losses.get_loss_fun()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train
    for epoch in range(max_epochs):
        train_loss = simple_train_epoch(model, train_loader, optimizer_obj, loss_fn, device)
    
    # Evaluate
    test_accuracy = simple_eval_epoch(model, test_loader, device)
    
    return test_accuracy


def main(cfg, args):
    """Main estimation workflow implementing Algorithm 1."""
    
    print("\n" + "=" * 70)
    print("  Algorithm 1: Equilibrium Noise Rate (ν') Estimation")
    print("=" * 70)
    print(f"Dataset: {cfg.DATASET.NAME}")
    print(f"Model: {cfg.MODEL.TYPE}")
    print(f"Output: {args.output}")
    print(f"\nBudget Parameters:")
    print(f"  Budget (B): {args.budget}")
    print(f"  Oracle Cost ($o): {args.oracle_cost}")
    print(f"  Noisy Cost ($n): {args.noisy_cost}")
    print(f"  → Oracle samples: {int(args.budget / args.oracle_cost)}")
    print(f"  → Noisy samples: {int(args.budget / args.noisy_cost)}")
    print(f"\nAlgorithm Parameters:")
    print(f"  Repeats (R): {args.repeats}")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Max Epochs: {args.max_epochs}")
    print(f"  Noise Type: {args.noise_type}")
    print("=" * 70 + "\n")
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load dataset
    logger.info("Loading dataset...")
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(
        save_dir=cfg.DATASET.ROOT_DIR, 
        isTrain=True, 
        isDownload=True
    )
    
    num_classes = cfg.MODEL.NUM_CLASSES
    
    # Extract features
    if args.use_simclr_features:
        logger.info("Loading pre-extracted SimCLR features...")
        features = load_features(cfg.DATASET.NAME, seed=args.seed, train=True, normalized=True)
        logger.info(f"Loaded SimCLR features: shape={features.shape}")
        
        # Get true labels from dataset
        logger.info("Loading dataset for labels...")
        data_obj = Data(cfg)
        train_data, train_size = data_obj.getDataset(
            save_dir=cfg.DATASET.ROOT_DIR, 
            isTrain=True, 
            isDownload=True
        )
        
        # Extract labels
        true_labels = np.array([train_data[i][1] for i in range(len(train_data))])
        logger.info(f"Extracted labels: {len(true_labels)} samples")
        
        # Generate pseudo-labels on FULL dataset for stability
        logger.info("Generating pseudo-labels via K-Means on FULL dataset...")
        all_pseudo_labels = get_pseudo_labels_kmeans(features, num_classes)
        agreement = (all_pseudo_labels == true_labels).mean() * 100
        logger.info(f"Pseudo-label agreement: {agreement:.2f}%")
        logger.info(f"Dataset: {cfg.DATASET.NAME}, {len(features)} samples, {num_classes} classes")
        
        # Use subset if requested
        if args.max_samples and args.max_samples < len(features):
            logger.info(f"Using subset: {args.max_samples} samples")
            indices = np.random.choice(len(features), args.max_samples, replace=False)
            features = features[indices]
            true_labels = true_labels[indices]
            pseudo_labels = all_pseudo_labels[indices]
            train_size = args.max_samples
        else:
            pseudo_labels = all_pseudo_labels
    else:
        logger.info("Initializing random model for feature extraction (WARNING: Not recommended!)")
        logger.info("Use --use_simclr_features for pretrained SimCLR features.")
        
        # Use a subset if requested
        if args.max_samples and args.max_samples < train_size:
            logger.info(f"Using subset: {args.max_samples} samples")
            indices = np.random.choice(train_size, args.max_samples, replace=False)
            train_data = Subset(train_data, indices)
            train_size = args.max_samples
        
        # Create dataloader for feature extraction
        batch_size = args.batch_size if args.batch_size else cfg.TRAIN.BATCH_SIZE
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS
        )
        
        model = mh.get_model(cfg)
        logger.info("Extracting features from training data...")
        features, true_labels, _ = extract_features(model, train_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
        pseudo_labels = None  # Will be generated inside estimate_nu_prime_binary_search
    
    # Estimate nu_prime using Algorithm 1
    nu_prime = estimate_nu_prime_binary_search(
        features=features,
        true_labels=true_labels,
        pseudo_labels=pseudo_labels,
        num_classes=num_classes,
        cfg=cfg,
        dataset=train_data,
        budget=args.budget,
        oracle_cost=args.oracle_cost,
        noisy_cost=args.noisy_cost,
        repeats=args.repeats,
        epsilon=args.epsilon,
        max_epochs=args.max_epochs,
        noise_type=args.noise_type,
        seed=args.seed
    )
    
    # Prepare metadata
    metadata = {
        'dataset': cfg.DATASET.NAME,
        'num_classes': num_classes,
        'model': cfg.MODEL.TYPE,
        'num_samples': train_size,
        'budget': args.budget,
        'oracle_cost': args.oracle_cost,
        'noisy_cost': args.noisy_cost,
        'oracle_samples': int(args.budget / args.oracle_cost),
        'noisy_samples': int(args.budget / args.noisy_cost),
        'repeats': args.repeats,
        'epsilon': args.epsilon,
        'max_epochs': args.max_epochs,
        'noise_type': args.noise_type,
        'seed': args.seed,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': 'Algorithm 1: Budget-Aware Oracle vs Noisy Comparison'
    }
    
    # Save result
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_nu_prime(nu_prime, args.output, metadata)
    
    print("\n" + "=" * 70)
    print(f"✓ Estimation Complete!")
    print(f"✓ ν' = {nu_prime:.4f}")
    print(f"✓ Saved to: {args.output}")
    print("=" * 70 + "\n")
    
    print("Interpretation:")
    print(f"  ν' = {nu_prime:.4f} is the equilibrium noise rate where:")
    print(f"    Oracle({int(args.budget/args.oracle_cost)} samples, clean) ≈ Noisy({int(args.budget/args.noisy_cost)} samples, noise={nu_prime:.4f})")
    print(f"\n  Decision Rule:")
    print(f"    • If your annotator's noise < {nu_prime:.4f}: Use Noisy Annotator (cost-effective)")
    print(f"    • If your annotator's noise > {nu_prime:.4f}: Use Oracle (quality over quantity)")
    print("")


if __name__ == "__main__":
    # Parse arguments
    args = argparser().parse_args()
    
    # Load configuration file
    cfg.merge_from_file(args.cfg_file)
    cfg.freeze()
    
    # Setup logging
    lu.setup_logging(cfg)
    
    # Run estimation
    main(cfg, args)
