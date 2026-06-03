"""
Extract K augmented views per image through a frozen pretrained SimCLR encoder.

Whereas extract_features_only.py runs each image through the encoder ONCE with
the val transform (clean image), this script runs each image K times through
the same encoder with the SimCLR/standard train transform, so every pass
produces a different random augmentation.

Output: a single .npy file of shape (N, K, D) for the train split, where
    N = number of training images
    K = --num_views
    D = encoder feature dim (e.g. 512 for ResNet-18 SimCLR)

Image order matches the val_dataloader iteration order, i.e. the same order
extract_features_only.py uses, so augmented[i, :, :] corresponds to the same
image as train_features[i] in the existing pretext_features file.

The model weights are FROZEN (no training). Cost is roughly K x one pass of
extract_features_only.py. Per-image augmentations are sampled independently
across the K passes (each pass calls __getitem__ which re-randomizes the
transform).
"""
import argparse
import os
import sys
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import (get_model, get_train_dataset,
                                 get_val_dataloader, get_train_transformations)
from utils.memory import MemoryBank
from utils.utils import fill_memory_bank
from termcolor import colored


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config_env', required=True, help='Config file for the environment')
    parser.add_argument('--config_exp', required=True, help='Config file for the experiment')
    parser.add_argument('--model_path', required=True, help='Path to the pretrained SimCLR weights')
    parser.add_argument('--output_path', default=None,
                        help='Where to save the (N, K, D) .npy file. '
                             'Default: <model_path dir>/train_features_aug{K}.npy')
    parser.add_argument('--num_views', type=int, default=8,
                        help='K: number of augmented views to sample per image (default 8)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    K = args.num_views
    assert K >= 1, 'num_views must be >= 1'

    p = create_config(args.config_env, args.config_exp, args.seed)
    print(colored(p, 'red'))

    # Model — frozen
    print(colored('Loading pretrained model', 'blue'))
    model = get_model(p)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model not found at {args.model_path}')
    print(colored(f'Loading weights from {args.model_path}', 'blue'))
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model = model.cuda()
    model.eval()
    torch.backends.cudnn.benchmark = True

    # Dataset with TRAIN transform (augmented), but using val_dataloader
    # (no shuffle, drop_last=False) so the iteration order is deterministic
    # and matches extract_features_only.py.
    print(colored('Building augmented train dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    train_dataset = get_train_dataset(p, train_transforms, split='train')
    train_dataloader = get_val_dataloader(p, train_dataset)
    N = len(train_dataset)
    D = p['model_kwargs']['features_dim']
    print(f'Dataset contains {N} train samples; features_dim={D}; K={K}')

    # K passes. Each call to fill_memory_bank reshuffles the per-sample
    # augmentation RNG (the transform contains RandomResizedCrop, etc., which
    # call into the global torch RNG on each __getitem__).
    memory_bank = MemoryBank(N, D, p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank.cuda()
    views = np.empty((N, K, D), dtype=np.float32)
    for k in range(K):
        # Bump the seed per view so each pass gets a distinct augmentation
        # sequence, even when this script is rerun with the same --seed.
        torch.manual_seed(args.seed * 10_000 + k)
        np.random.seed(args.seed * 10_000 + k)
        print(colored(f'Pass {k + 1}/{K}', 'blue'))
        with torch.no_grad():
            fill_memory_bank(train_dataloader, model, memory_bank)
        views[:, k, :] = memory_bank.pre_lasts.cpu().numpy()

    out_path = args.output_path
    if out_path is None:
        out_dir = os.path.dirname(os.path.abspath(args.model_path))
        out_path = os.path.join(out_dir, f'train_features_aug{K}.npy')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    np.save(out_path, views)
    print(colored(f'Saved augmented features to: {out_path}', 'green'))
    print(colored(f'Shape: {views.shape}  dtype: {views.dtype}', 'green'))


if __name__ == '__main__':
    main()
