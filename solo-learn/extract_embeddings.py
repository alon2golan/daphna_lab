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
import re
from pathlib import Path

print("loading dependencies (torch + solo-learn), this takes ~15-30s ...", flush=True)

import numpy as np
import torch
from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from solo.data.classification_dataloader import prepare_datasets, prepare_transforms
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
    ckpts = [f for f in ckpt_dir.iterdir() if f.suffix == ".ckpt"]
    if not ckpts:
        raise FileNotFoundError(f"no .ckpt found in {ckpt_dir}")
    # When keep_prev saved several checkpoints (e.g. every 100 epochs), pick the
    # latest epoch. Filenames end with '-ep=<N>.ckpt'; fall back to name sort.
    def _epoch(p):
        m = re.search(r"ep=(\d+)", p.name)
        return int(m.group(1)) if m else -1
    ckpt_path = max(ckpts, key=_epoch)
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

    # We want a faithful frozen-feature matrix of the ACTUAL images, so both splits
    # use the deterministic val transform (T_val: ToTensor + Normalize -- no random
    # crop/flip). prepare_data's default train loader would instead apply T_train
    # (RandomResizedCrop scale 0.08-1.0 + RandomHorizontalFlip) and use shuffle=True
    # + drop_last=True, i.e. augmented, reordered, and missing the last partial
    # batch. We build the loaders ourselves with T_val for both, shuffle off,
    # drop_last off, so row i is image i and every sample is included.
    _, T_val = prepare_transforms(args.dataset)
    train_dataset, val_dataset = prepare_datasets(
        args.dataset,
        T_val,
        T_val,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=args.data_format,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def l2norm(f):
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)

    def save_split(f, y, prefix):
        # Save both the raw backbone features and an L2-normalized copy, so
        # downstream code can use whichever it needs without re-normalizing.
        np.save(out / f"{prefix}_features.npy", f)
        np.save(out / f"{prefix}_features_normalized.npy", l2norm(f))
        np.save(out / f"{prefix}_labels.npy", y)
        print(f"{prefix}: features {f.shape} (raw + normalized) -> {out}")

    if args.split in ("train", "both"):
        f, y = extract(model, train_loader, device)
        save_split(f, y, "train")

    if args.split in ("val", "both"):
        f, y = extract(model, val_loader, device)
        save_split(f, y, "test")


if __name__ == "__main__":
    main()
