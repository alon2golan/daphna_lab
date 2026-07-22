# The solo-learn interface layer: load a pretrained SimCLR backbone from a checkpoint,
# build CIFAR datasets whose row order matches the extracted .npy features, and turn
# images into backbone features. Everything that touches solo-learn lives here so the
# rest of the finetune package stays plain torch + numpy.

import re
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf

# solo-learn is a sibling project, not an installed package; put it on the path before
# importing from it (same trick compare_hierarchical uses for the pipeline dir). This
# file is .../hierarchical_framework/pipeline/finetune/solo_backbone.py, so parents[3]
# is the daphna_lab root that holds solo-learn/.
SOLO_LEARN_DIR = Path(__file__).resolve().parents[3] / "solo-learn"
if str(SOLO_LEARN_DIR) not in sys.path:
    sys.path.insert(0, str(SOLO_LEARN_DIR))

from solo.data.classification_dataloader import prepare_datasets, prepare_transforms
from solo.methods import METHODS


# --- Configuration Constants ---

# Batch size used only for the (no-grad) feature-extraction passes; unrelated to training
EXTRACTION_BATCH_SIZE = 256

# Worker processes for the extraction dataloaders
EXTRACTION_WORKERS = 4

# CIFAR normalization used by solo-learn's cifar pipeline (train AND val). The strong-aug train
# transform below MUST end with these exact stats so finetune-time inputs match the eval-time
# (T_val) inputs the features are extracted with -- otherwise the embedding shifts and breaks.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.247, 0.243, 0.261)


def strong_train_transform():
    """SimCLR-style train augmentation: the existing RandomResizedCrop + flip PLUS the photometric
    half that the default finetune transform lacks -- color jitter and random grayscale. This is
    the pretraining-era aug, so it stays in-distribution for the SimCLR backbone. Same crop size
    (32) and same Normalize stats as T_val, so only the augmentation strength changes."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),          # same spatial crop as default
        transforms.RandomHorizontalFlip(),                                  # same flip as default
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # SimCLR jitter
        transforms.RandomGrayscale(p=0.2),                                  # SimCLR grayscale drop
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),                        # MUST match T_val stats
    ])


# --- Backbone Loading ---

def load_backbone(checkpoint_dir, device, multi_gpu=False):
    """Rebuilds the pretrained method from its args.json, loads the latest checkpoint,
    and returns ONLY the backbone encoder (projection/prediction heads dropped). When
    multi_gpu is set and more than one GPU is visible, wraps it in DataParallel so each
    batch is split across GPUs (transparent: .parameters()/.train()/backbone(x) still work)."""
    checkpoint_dir = Path(checkpoint_dir)

    # Collect every .ckpt in the dir; keep_prev saves one per 100 epochs
    checkpoints = [f for f in checkpoint_dir.iterdir() if f.suffix == ".ckpt"]
    if not checkpoints:
        raise FileNotFoundError(f"no .ckpt found in {checkpoint_dir}")

    # Filenames end with '-ep=<N>.ckpt'; pick the highest epoch (ep=999 here)
    def epoch_of(path):
        match = re.search(r"ep=(\d+)", path.name)
        return int(match.group(1)) if match else -1
    checkpoint_path = max(checkpoints, key=epoch_of)

    # args.json holds the exact method + backbone config the checkpoint was trained with
    with open(checkpoint_dir / "args.json") as handle:
        cfg = OmegaConf.create(json.load(handle))

    # Reconstruct the SimCLR/BYOL/... module and load weights; strict=False tolerates the
    # momentum/head buffers we intentionally drop by taking .backbone
    model = METHODS[cfg.method].load_from_checkpoint(
        checkpoint_path, strict=False, cfg=cfg
    ).backbone

    model = model.to(device)  # move the encoder onto the training device

    # Split each batch across all visible GPUs when asked (helps the large-budget finetunes)
    if multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def infer_feature_dim(backbone, eval_dataset, device):
    """Runs a single image through the backbone to read off its output width (512 for resnet18)."""
    backbone.eval()  # no dropout/bn updates for this probe forward
    with torch.no_grad():
        image, _label = eval_dataset[0]                       # first sample, deterministically ordered
        features = backbone(image.unsqueeze(0).to(device))    # add a batch dim of 1
    return int(features.shape[1])                             # the feature dimension


# --- Datasets (index-aligned to the .npy features) ---

def build_datasets(dataset, data_dir, aug="default"):
    """Returns three CIFAR datasets sharing one index order:
      train_aug  -- T_train augmentations, for the finetune loop,
      train_eval -- T_val (deterministic), for extracting train features,
      test_eval  -- T_val, for extracting test features.
    Row i is image i in every one of them, matching train_features.npy row i.
    aug="strong" swaps the finetune-loop T_train for the SimCLR-style transform (adds color jitter
    + grayscale); train_eval/test_eval always use T_val, so feature extraction is unaffected."""
    data_dir = str(data_dir)

    # solo-learn's transform pair: T_train has random crop/flip, T_val is ToTensor+Normalize
    T_train, T_val = prepare_transforms(dataset)
    if aug == "strong":
        T_train = strong_train_transform()  # only the finetune-loop augmentation changes

    # Augmented train + clean test, using T_train on the training split
    train_aug, test_eval = prepare_datasets(
        dataset, T_train, T_val,
        train_data_path=data_dir, val_data_path=data_dir, data_format="image_folder",
    )

    # Clean (un-augmented) train, using T_val on the training split, for feature extraction
    train_eval, _ = prepare_datasets(
        dataset, T_val, T_val,
        train_data_path=data_dir, val_data_path=data_dir, data_format="image_folder",
    )

    return train_aug, train_eval, test_eval


# --- Feature Extraction ---

def l2_normalize(features):
    """Row-wise L2 normalization, matching how the pipeline's .npy features are stored."""
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)


@torch.no_grad()
def extract_features(backbone, dataset, device, normalize=True):
    """Runs the whole dataset through the (frozen-for-this-call) backbone in original order
    and returns (features, labels) as numpy arrays; optionally L2-normalizes the features."""
    backbone.eval()  # deterministic forward: no bn/dropout noise while extracting

    # No shuffle, no drop_last -> row i of the output is sample i of the dataset
    loader = DataLoader(
        dataset, batch_size=EXTRACTION_BATCH_SIZE, shuffle=False,
        num_workers=EXTRACTION_WORKERS, pin_memory=True, drop_last=False,
    )

    feature_batches, label_batches = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)   # move the batch to the GPU
        feature_batches.append(backbone(images).cpu())  # backbone output, back to CPU
        label_batches.append(labels.clone())            # fine labels, kept aligned

    features = torch.cat(feature_batches).numpy()  # stack into (N, feat_dim)
    labels = torch.cat(label_batches).numpy()      # stack into (N,)

    if normalize:
        features = l2_normalize(features)
    return features, labels
