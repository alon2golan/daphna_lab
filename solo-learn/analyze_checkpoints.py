"""Analyze the evolution of solo-learn backbones across their saved checkpoints.

You only ran main_pretrain.py, so on disk you have, per run, ~11 checkpoints
(epoch 0, 99, 199, ... 999) plus args.json. This script walks every checkpoint,
runs the dataset through the frozen backbone, L2-normalizes the features, and
computes a battery of representation-quality metrics -- then plots how each one
evolves over training and how much the representation drifts between epochs.

For each (run, checkpoint) it reports, on the L2-normalized TEST features:
  knn_acc        weighted k-NN top-1 accuracy (feature quality, no probe trained)
  uniformity     Wang & Isola uniformity: log E exp(-2||fi-fj||^2). Lower = points
                 spread more evenly over the unit sphere ("where on the ball").
  intra_align    mean intra-class squared distance. Lower = same-class points tight.
  eff_rank       effective rank exp(H(singular values)). How many dims are used.
  silhouette     class-cluster separation in feature space (subsampled).
And BETWEEN consecutive checkpoints of a run:
  cka_prev       linear CKA vs the previous checkpoint. 1 = identical geometry,
                 lower = the representation changed a lot this interval.

Outputs under --output_dir:
  metrics.csv                 one row per (run, epoch)
  metric_<name>_vs_epoch.png  one curve per run, for each metric
  umap_<run>.png              UMAP snapshots (a few epochs) colored by class

Run it on a GPU node via run_analyze.sbatch (44 extractions over full CIFAR).
"""

import argparse
import json
import re
from pathlib import Path

print("loading dependencies (torch + solo-learn + umap), ~30s ...", flush=True)

import numpy as np
import torch
from omegaconf import OmegaConf

from solo.data.classification_dataloader import prepare_data
from solo.methods import METHODS


# ------------------------------------------------------------------ extraction
@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats.append(model(x).cpu())
        labels.append(y.clone())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def load_backbone(ckpt_path, cfg, device):
    model = METHODS[cfg.method].load_from_checkpoint(
        ckpt_path, strict=False, cfg=cfg
    ).backbone
    return model.to(device)


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


# -------------------------------------------------------------------- metrics
def knn_acc(train_f, train_y, test_f, test_y, k=20):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    clf.fit(train_f, train_y)
    return float((clf.predict(test_f) == test_y).mean())


def uniformity(f, t=2.0, sample=4000, rng=None):
    idx = rng.choice(len(f), min(sample, len(f)), replace=False)
    x = f[idx]
    sq = np.maximum(0.0, 2 - 2 * (x @ x.T))          # squared dists on unit sphere
    iu = np.triu_indices(len(x), k=1)
    return float(np.log(np.exp(-t * sq[iu]).mean()))


def intra_align(f, y):
    tot, n = 0.0, 0
    for c in np.unique(y):
        g = f[y == c]
        mu = g.mean(0)
        tot += ((g - mu) ** 2).sum()
        n += len(g)
    return float(tot / n)


def eff_rank(f):
    s = np.linalg.svd(f - f.mean(0), compute_uv=False)
    p = s / s.sum()
    h = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(h))


def silhouette(f, y, sample=3000, rng=None):
    from sklearn.metrics import silhouette_score
    idx = rng.choice(len(f), min(sample, len(f)), replace=False)
    return float(silhouette_score(f[idx], y[idx], metric="cosine"))


def linear_cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    hsic = np.linalg.norm(a.T @ b, "fro") ** 2
    na = np.linalg.norm(a.T @ a, "fro"); nb = np.linalg.norm(b.T @ b, "fro")
    return float(hsic / (na * nb + 1e-12))


# ----------------------------------------------------------------------- main
def epoch_of(p):
    m = re.search(r"ep=(\d+)", p.name)
    return int(m.group(1)) if m else -1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="checkpoint dirs, e.g. trained_models/simclr/cifar10/1000ep")
    p.add_argument("--data_path", default="./datasets")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--umap_epochs", nargs="+", type=int, default=[0, 499, 999],
                   help="which epochs to draw UMAP snapshots for")
    p.add_argument("--output_dir", default="analysis")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    rows = []                      # (run, epoch, metric dict)
    umap_cache = {}                # run -> {epoch: (feat, label)}

    for run in args.runs:
        run_dir = Path(run)
        cfg = OmegaConf.create(json.load(open(run_dir / "args.json")))
        dataset = cfg.data.dataset
        run_name = f"{cfg.method}-{dataset}"
        print(f"\n=== {run_name}  ({run_dir}) ===", flush=True)

        train_loader, val_loader = prepare_data(
            dataset, train_data_path=args.data_path, val_data_path=args.data_path,
            data_format="image_folder", batch_size=args.batch_size,
            num_workers=args.num_workers, auto_augment=False,
        )

        ckpts = sorted((f for f in run_dir.iterdir() if f.suffix == ".ckpt"),
                       key=epoch_of)
        prev_test = None
        for ck in ckpts:
            ep = epoch_of(ck)
            model = load_backbone(ck, cfg, device)
            tr_f, tr_y = extract(model, train_loader, device)
            te_f, te_y = extract(model, val_loader, device)
            tr_f, te_f = l2(tr_f), l2(te_f)

            m = dict(
                run=run_name, dataset=dataset, method=cfg.method, epoch=ep,
                knn_acc=knn_acc(tr_f, tr_y, te_f, te_y),
                uniformity=uniformity(te_f, rng=rng),
                intra_align=intra_align(te_f, te_y),
                eff_rank=eff_rank(te_f),
                silhouette=silhouette(te_f, te_y, rng=rng),
                cka_prev=linear_cka(prev_test, te_f) if prev_test is not None else np.nan,
            )
            prev_test = te_f
            rows.append(m)
            print(f"  ep={ep:4d}  knn={m['knn_acc']:.3f}  unif={m['uniformity']:.2f}"
                  f"  align={m['intra_align']:.3f}  rank={m['eff_rank']:.1f}"
                  f"  sil={m['silhouette']:.3f}  cka_prev={m['cka_prev']:.3f}", flush=True)

            if ep in args.umap_epochs:
                umap_cache.setdefault(run_name, {})[ep] = (te_f, te_y)

    # ---- write csv
    import csv
    keys = ["run", "dataset", "method", "epoch", "knn_acc", "uniformity",
            "intra_align", "eff_rank", "silhouette", "cka_prev"]
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print(f"\nwrote {out / 'metrics.csv'}", flush=True)

    # ---- one figure per metric: value vs epoch, one line per run
    runs = sorted({r["run"] for r in rows})
    for metric in ["knn_acc", "uniformity", "intra_align", "eff_rank",
                   "silhouette", "cka_prev"]:
        plt.figure(figsize=(6, 4))
        for rn in runs:
            pts = sorted([(r["epoch"], r[metric]) for r in rows if r["run"] == rn])
            xs = [e for e, v in pts if not np.isnan(v)]
            ys = [v for e, v in pts if not np.isnan(v)]
            plt.plot(xs, ys, marker="o", label=rn)
        plt.xlabel("epoch"); plt.ylabel(metric); plt.title(f"{metric} vs epoch")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(out / f"metric_{metric}_vs_epoch.png", dpi=130)
        plt.close()
    print("wrote metric_*_vs_epoch.png", flush=True)

    # ---- UMAP snapshots per run
    import umap
    for rn, snaps in umap_cache.items():
        eps = sorted(snaps)
        fig, axes = plt.subplots(1, len(eps), figsize=(4 * len(eps), 4), squeeze=False)
        for ax, ep in zip(axes[0], eps):
            f, y = snaps[ep]
            emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                            random_state=0).fit_transform(f)
            ax.scatter(emb[:, 0], emb[:, 1], c=y, s=3, cmap="tab20", alpha=0.6)
            ax.set_title(f"{rn}  ep={ep}"); ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(out / f"umap_{rn}.png", dpi=130); plt.close()
        print(f"wrote umap_{rn}.png", flush=True)


if __name__ == "__main__":
    main()
