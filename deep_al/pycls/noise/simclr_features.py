"""
Utilities for extracting features from pretrained SimCLR models.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


def load_simclr_model(checkpoint_path, backbone='resnet18', features_dim=128):
    """
    Load a pretrained SimCLR model.
    
    Args:
        checkpoint_path: Path to SimCLR checkpoint (.pth.tar file)
        backbone: Backbone architecture (default: resnet18)
        features_dim: Feature dimension (default: 128)
        
    Returns:
        model: Loaded SimCLR model
    """
    # Import from SCAN codebase
    import sys
    import os
    scan_path = os.path.join(os.path.dirname(__file__), '../../../../scan')
    if scan_path not in sys.path:
        sys.path.insert(0, scan_path)
    
    from models.models import ContrastiveModel
    
    # Create model
    model = ContrastiveModel(backbone, head='mlp', features_dim=features_dim)
    
    # Load checkpoint
    print(f"Loading SimCLR checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


@torch.no_grad()
def extract_simclr_features(model, dataloader, device='cuda'):
    """
    Extract features from SimCLR model.
    
    Args:
        model: Pretrained SimCLR model
        dataloader: DataLoader for the dataset
        device: Device to use
        
    Returns:
        features: numpy array of features
        labels: numpy array of labels
        indices: numpy array of indices
    """
    model.to(device)
    model.eval()
    
    features_list = []
    labels_list = []
    indices_list = []
    
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 2:
            images, labels = batch
            batch_indices = torch.arange(batch_idx * dataloader.batch_size, 
                                        batch_idx * dataloader.batch_size + len(labels))
        elif len(batch) == 3:
            images, labels, batch_indices = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        
        images = images.to(device)
        
        # Extract features (before projection head)
        features = model.backbone(images)
        features = features.view(features.size(0), -1)
        
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.numpy())
        indices_list.append(batch_indices.numpy())
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    indices = np.concatenate(indices_list, axis=0)
    
    print(f"Extracted SimCLR features: shape={features.shape}")
    
    return features, labels, indices
