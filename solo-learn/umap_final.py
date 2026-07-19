"""Plot a UMAP of the final checkpoint of each run.

Loads only the last (highest-epoch) checkpoint per run, extracts frozen backbone
features from the test set, and saves one UMAP panel per run.

Usage:
    python -u umap_final.py \
        --runs trained_models/simclr/cifar10/1000ep \
               trained_models/simclr/cifar100/1000ep \
               trained_models/byol/cifar10/1000ep \
               trained_models/byol/cifar100/1000ep \
        --data_path /cs/labs/daphna/$USER/projects/daphna_lab/data \
        --output_dir analysis
"""

import argparse
import json
import re
from pathlib import Path

print("loading dependencies (torch + solo-learn + umap) ...", flush=True)

import numpy as np
import torch
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from solo.data.classification_dataloader import prepare_data
from solo.methods import METHODS


def epoch_of(p):
    m = re.search(r"ep=(\d+)", p.name)
    return int(m.group(1)) if m else -1


def last_checkpoint(run_dir):
    ckpts = [f for f in run_dir.iterdir() if f.suffix == ".ckpt"]
    if not ckpts:
        raise FileNotFoundError(f"no .ckpt files in {run_dir}")
    return max(ckpts, key=epoch_of)


@torch.no_grad()
def extract(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device, non_blocking=True)).cpu())
        labels.append(y.clone())
    f = torch.cat(feats).numpy()
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    return f, torch.cat(labels).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data_path", default="./datasets")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output_dir", default="analysis")
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for run in args.runs:
        run_dir = Path(run)
        cfg = OmegaConf.create(json.load(open(run_dir / "args.json")))
        dataset = cfg.data.dataset
        run_name = f"{cfg.method}-{dataset}"

        ckpt = last_checkpoint(run_dir)
        ep = epoch_of(ckpt)
        print(f"\n{run_name}: loading ep={ep} from {ckpt.name}", flush=True)

        backbone = METHODS[cfg.method].load_from_checkpoint(
            ckpt, strict=False, cfg=cfg
        ).backbone.to(device)

        _, val_loader = prepare_data(
            dataset,
            train_data_path=args.data_path,
            val_data_path=args.data_path,
            data_format="image_folder",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            auto_augment=False,
        )

        print("  extracting features ...", flush=True)
        feats, labels = extract(backbone, val_loader, device)

        print("  fitting UMAP ...", flush=True)
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=0).fit_transform(feats)

        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, s=3,
                        cmap="tab20", alpha=0.6)
        plt.colorbar(sc, ax=ax, fraction=0.03)
        ax.set_title(f"{run_name}  ep={ep}")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        out_path = out / f"umap_final_{run_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
