"""Extract backbone embeddings from a solo-learn pretrained checkpoint.

After pretraining (main_pretrain.py) you get a checkpoint directory holding a
`.ckpt` file and an `args.json`. This script loads that model, keeps ONLY the
backbone encoder (dropping the projection/prediction heads), runs a dataset
through it, and saves the resulting features + labels as .npy files -- the same
kind of frozen-feature matrix the downstream active-learning code consumes.

The embedding is the backbone output (512-dim for resnet18), NOT the projector
output; the projector is only used during contrastive training.

Example:
    python extract_embeddings.py \
        --checkpoint_dir trained_models/simclr/<run-id> \
        --dataset cifar10 \
        --output_dir embeddings/simclr_cifar10

Produces, under --output_dir:
    train_features.npy  (N_train, feat_dim)   test_features.npy  (N_test, feat_dim)
    train_labels.npy    (N_train,)            test_labels.npy    (N_test,)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from solo.data.classification_dataloader import prepare_data
from solo.methods import METHODS


@torch.no_grad()
def extract(model, loader, device):
    """Run every batch through the backbone and stack features + labels."""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats.append(model(x).cpu())
        labels.append(y.clone())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", required=True,
                   help="dir with the .ckpt and args.json produced by pretraining")
    p.add_argument("--dataset", required=True,
                   help="cifar10 | cifar100 | stl10 | custom")
    p.add_argument("--train_data_path", default="./datasets")
    p.add_argument("--val_data_path", default="./datasets")
    p.add_argument("--data_format", default="image_folder",
                   help="only relevant for custom/imagenet-style folders")
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_path = next(f for f in ckpt_dir.iterdir() if f.suffix == ".ckpt")
    with open(ckpt_dir / "args.json") as f:
        cfg = OmegaConf.create(json.load(f))
    print(f"Loading {cfg.method} backbone from {ckpt_path}")

    # Reconstruct the pretrained method from its config, load weights, keep the
    # backbone encoder only. strict=False tolerates the missing head/momentum
    # buffers that we intentionally drop.
    model = METHODS[cfg.method].load_from_checkpoint(
        ckpt_path, strict=False, cfg=cfg
    ).backbone

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_loader, val_loader = prepare_data(
        args.dataset,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=args.data_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        auto_augment=False,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.split in ("train", "both"):
        f, y = extract(model, train_loader, device)
        np.save(out / "train_features.npy", f)
        np.save(out / "train_labels.npy", y)
        print(f"train: features {f.shape} -> {out / 'train_features.npy'}")

    if args.split in ("val", "both"):
        f, y = extract(model, val_loader, device)
        np.save(out / "test_features.npy", f)
        np.save(out / "test_labels.npy", y)
        print(f"val:   features {f.shape} -> {out / 'test_features.npy'}")


if __name__ == "__main__":
    main()
