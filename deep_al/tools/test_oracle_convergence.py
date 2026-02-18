#!/usr/bin/env python3
"""
Test Oracle Training Convergence

Simple script to verify that training on pseudo-labels (from K-Means) 
converges to reasonable accuracy. Trains a linear classifier on pre-extracted 
features instead of training CNN from scratch.

Usage:
    python test_oracle_convergence.py \\
        --cfg deep_al/configs/cifar100/al/RESNET18.yaml \\
        --max_samples 2000 \\
        --max_epochs 100 \\
        --train_samples 100
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root is in sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from deep_al.pycls.core.config import cfg
from deep_al.pycls.datasets.data import Data
import deep_al.pycls.utils.logging as lu
import deep_al.pycls.utils.model_handler as mh
from deep_al.pycls.noise.nu_prime_utils import (
    extract_features,
    get_pseudo_labels_kmeans
)
from deep_al.pycls.datasets.utils.features import load_features

logger = lu.get_logger(__name__)


class LinearClassifier(nn.Module):
    """Simple linear classifier for features."""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    return avg_loss, train_acc


@torch.no_grad()
def eval_epoch(model, eval_loader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    
    for features, labels in eval_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def main(args):
    """Main test workflow."""
    
    print("\n" + "=" * 70)
    print("  Oracle Training Convergence Test")
    print("=" * 70)
    print(f"Dataset: {cfg.DATASET.NAME}")
    print(f"Model: {cfg.MODEL.TYPE}")
    print(f"Max Samples: {args.max_samples}")
    print(f"Train Samples: {args.train_samples}")
    print(f"Max Epochs: {args.max_epochs}")
    print("=" * 70 + "\n")
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load dataset
    print("Loading dataset...")
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(
        save_dir=cfg.DATASET.ROOT_DIR, 
        isTrain=True, 
        isDownload=True
    )
    
    num_classes = cfg.MODEL.NUM_CLASSES
    print(f"Dataset loaded: {train_size} samples, {num_classes} classes")
    
    # Use subset if specified
    if args.max_samples and args.max_samples < train_size:
        from torch.utils.data import Subset
        print(f"Using subset: {args.max_samples} samples")
        indices = np.random.choice(train_size, args.max_samples, replace=False)
        train_data = Subset(train_data, indices)
        train_size = args.max_samples
    
    # Create dataloader
    batch_size = cfg.TRAIN.BATCH_SIZE
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS
    )
    
    # Extract features
    if args.use_simclr_features:
        print("\nLoading pre-extracted SimCLR features...")
        features_tag = cfg.DATASET.get('FEATURES_TAG', None)
        features = load_features(cfg.DATASET.NAME, seed=args.seed, train=True, normalized=True, features_tag=features_tag)
        print(f"Loaded SimCLR features: shape={features.shape}")
        
        # Get true labels
        print("Loading dataset for labels...")
        data_obj = Data(cfg)
        train_data, train_size = data_obj.getDataset(
            save_dir=cfg.DATASET.ROOT_DIR, 
            isTrain=True, 
            isDownload=True
        )
        
        true_labels = np.array([train_data[i][1] for i in range(len(train_data))])
        print(f"Extracted labels: {len(true_labels)} samples")
        
        # Generate pseudo-labels on FULL dataset for stability
        print(f"\nGenerating pseudo-labels via K-Means on FULL dataset...")
        all_pseudo_labels = get_pseudo_labels_kmeans(features, num_classes)
        agreement = (all_pseudo_labels == true_labels).mean() * 100
        print(f"Pseudo-label agreement with true labels: {agreement:.2f}%")
        
        # NOW take subset for training
        if args.max_samples and args.max_samples < len(features):
            indices = np.random.choice(len(features), args.max_samples, replace=False)
            features = features[indices]
            true_labels = true_labels[indices]
            pseudo_labels = all_pseudo_labels[indices]
            print(f"Using subset: {args.max_samples} samples")
        else:
            pseudo_labels = all_pseudo_labels
    else:
        print("\nExtracting features from RANDOM model (not recommended!)")
        print("Use --use_simclr_features for meaningful features.")
        
        if args.max_samples and args.max_samples < train_size:
            indices = np.random.choice(train_size, args.max_samples, replace=False)
            train_data = Subset(train_data, indices)
        
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS
        )
        
        model = mh.get_model(cfg)
        features, true_labels, _ = extract_features(model, train_loader, device=device)
    
    print(f"Features extracted: shape={features.shape}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Pseudo-label agreement with true labels (for this subset): {(pseudo_labels == true_labels).mean() * 100:.2f}%")
    print()
    
    # Split into train/test
    n_total = len(features)
    n_train_split = int(0.8 * n_total)
    
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n_total)
    train_indices = indices[:n_train_split]
    test_indices = indices[n_train_split:]
    
    # Sample subset for training (oracle budget)
    if args.train_samples < len(train_indices):
        sample_indices = rng.choice(len(train_indices), size=args.train_samples, replace=False)
        sampled_train_indices = train_indices[sample_indices]
    else:
        sampled_train_indices = train_indices
    
    # Prepare datasets
    train_features = torch.FloatTensor(features[sampled_train_indices])
    train_labels_pseudo = torch.LongTensor(pseudo_labels[sampled_train_indices])
    
    test_features = torch.FloatTensor(features[test_indices])
    test_labels_pseudo = torch.LongTensor(pseudo_labels[test_indices])  # Use pseudo-labels for consistent evaluation
    
    train_dataset = TensorDataset(train_features, train_labels_pseudo)
    test_dataset = TensorDataset(test_features, test_labels_pseudo)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("=" * 70)
    print("Training Configuration:")
    print(f"  Train samples: {len(sampled_train_indices)} (pseudo-labels)")
    print(f"  Test samples: {len(test_indices)} (pseudo-labels)")
    print(f"  Feature dim: {features.shape[1]}")
    print(f"  Num classes: {num_classes}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 70 + "\n")
    
    # Initialize linear classifier
    input_dim = features.shape[1]
    classifier = LinearClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_obj = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9)
    
    # Training loop with progress monitoring
    print("Starting training...\n")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Acc (Pseudo)':<18}")
    print("-" * 70)
    
    best_test_acc = 0.0
    
    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_acc = train_epoch(classifier, train_loader, optimizer_obj, criterion, device)
        
        # Evaluate every N epochs or on last epoch
        if epoch % args.eval_freq == 0 or epoch == args.max_epochs:
            test_acc = eval_epoch(classifier, test_loader, device)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_marker = " ← BEST"
            else:
                best_marker = ""
            
            print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.2f}% {test_acc:<18.2f}%{best_marker}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Test Accuracy (on pseudo-labels): {best_test_acc:.2f}%")
    print(f"Final Test Accuracy (on pseudo-labels): {test_acc:.2f}%")
    print("\nInterpretation:")
    print("  • Train Acc shows model learns pseudo-label patterns")
    print("  • Test Acc shows generalization to held-out pseudo-labels")
    print("  • High test acc means model successfully learns K-Means clusters")
    print("  • This validates the oracle baseline for Algorithm 1")
    print("=" * 70 + "\n")


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test Oracle Training Convergence on Features'
    )
    
    parser.add_argument('--cfg', dest='cfg_file', required=True, type=str,
                        help='Path to config file')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Maximum samples to use (default: 2000)')
    parser.add_argument('--train_samples', type=int, default=100,
                        help='Number of samples for training (default: 100)')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Evaluation frequency in epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--use_simclr_features', action='store_true',
                        help='Use pre-extracted SimCLR features from representations_bank')
    
    return parser


if __name__ == "__main__":
    # Parse arguments
    args = argparser().parse_args()
    
    # Load configuration
    cfg.merge_from_file(args.cfg_file)
    cfg.freeze()
    
    # Setup logging
    lu.setup_logging(cfg)
    
    # Run test
    main(args)
