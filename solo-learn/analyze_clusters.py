"""Count, per class, how many sub-clusters a class fractures into in feature space.

The UMAP plots show that a single semantic class does not always land as one blob
in an SSL backbone's feature space -- it can split into several density islands.
This script quantifies that: for each run, at a few checkpoints, for each class on
its own, it reduces that class's features with UMAP and then clusters them with
HDBSCAN. HDBSCAN needs no k -- it returns however many density clusters exist,
plus a "noise" bucket for points too sparse to belong to any of them.

Why UMAP first: HDBSCAN on the raw 512-d L2-normalized features is degenerate --
in that many dimensions a class is one diffuse lump with no density separation, so
HDBSCAN grabs a tiny densest knot and calls ~90% of the class noise. Reducing each
class to a few UMAP dims (the standard UMAP->HDBSCAN recipe) makes density
meaningful and lets real sub-structure show up. Why HDBSCAN and not k-means: no k to
pick, and leftover points become noise instead of being forced into a blob.

TWO STAGES, decoupled so you only pay the GPU cost once:
  1. extract   run the frozen backbone over the split, cache features to
               <output_dir>/feats/<run>_ep<ep>.npz. Needs torch + a GPU.
  2. cluster   per class: UMAP-reduce -> HDBSCAN -> counts. Reads the cache, so
               it runs on a plain login node (no torch, no GPU) in seconds.
Rerun with different --umap_dim / --min_frac etc. and stage 1 is skipped entirely
as long as the cache exists (delete <output_dir>/feats or pass --force_extract to
recompute). torch/solo/umap are imported lazily so a cache-only run needs neither.

For each (run, checkpoint, class) it reports:
  n_clusters     how many density clusters that class split into (noise excluded)
  noise_frac     fraction of the class HDBSCAN left as noise (no dense cluster)
  largest_frac   fraction of the class in its biggest sub-cluster (1.0 = one blob)
  sizes          the sub-cluster sizes, largest first

Outputs under --output_dir:
  feats/<run>_ep<ep>.npz          cached (features, labels) per checkpoint
  clusters.csv                    one row per (run, epoch, class)
  clusters_dist_<run>.png         box of per-class n_clusters at each epoch
  clusters_per_class_<run>.png    bar of n_clusters per class at the final epoch

First run on a GPU node via run_clusters.sbatch (12 extractions). After that,
re-cluster anywhere with run_clusters.sh.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def epoch_of(p):
    m = re.search(r"ep=(\d+)", p.name)
    return int(m.group(1)) if m else -1


# --------------------------------------------------------- stage 1: extraction
def extract_to_cache(run_dir, epochs, split, data_path, batch_size, num_workers,
                     cache_dir, force):
    """Return [(run_name, epoch, npz_path)] for the requested epochs, extracting
    (and caching) any that are missing. torch/solo are imported here so a fully
    cached run never touches them."""
    cfg_raw = json.load(open(run_dir / "args.json"))
    method = cfg_raw["method"]
    dataset = cfg_raw["data"]["dataset"]
    run_name = f"{method}-{dataset}"

    want = set(epochs)
    ckpts = sorted((f for f in run_dir.iterdir() if f.suffix == ".ckpt"),
                   key=epoch_of)
    ckpts = [c for c in ckpts if epoch_of(c) in want]

    out = []
    missing = []
    for ck in ckpts:
        ep = epoch_of(ck)
        npz = cache_dir / f"{run_name}_ep{ep}.npz"
        out.append((run_name, ep, npz))
        if force or not npz.exists():
            missing.append((ck, ep, npz))

    if not missing:
        print(f"  [{run_name}] all {len(out)} checkpoints cached, skipping extract",
              flush=True)
        return out

    # Only now do we need the heavy stack.
    print(f"  [{run_name}] extracting {len(missing)} checkpoint(s) ...", flush=True)
    import torch
    from omegaconf import OmegaConf
    from solo.data.classification_dataloader import prepare_data
    from solo.methods import METHODS

    cfg = OmegaConf.create(cfg_raw)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = prepare_data(
        dataset, train_data_path=data_path, val_data_path=data_path,
        data_format="image_folder", batch_size=batch_size,
        num_workers=num_workers, auto_augment=False,
    )
    loader = train_loader if split == "train" else val_loader

    @torch.no_grad()
    def extract(backbone):
        backbone.eval()
        feats, labels = [], []
        for x, y in loader:
            feats.append(backbone(x.to(device, non_blocking=True)).cpu())
            labels.append(y.clone())
        f = torch.cat(feats).numpy()
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        return f.astype(np.float32), torch.cat(labels).numpy().astype(np.int64)

    for ck, ep, npz in missing:
        backbone = METHODS[method].load_from_checkpoint(
            ck, strict=False, cfg=cfg).backbone.to(device)
        f, y = extract(backbone)
        np.savez_compressed(npz, feats=f, labels=y)
        print(f"    ep={ep:4d}  cached {f.shape} -> {npz.name}", flush=True)
    return out


# ---------------------------------------------------------- stage 2: clustering
def cluster_one_class(f, umap_dim, umap_neighbors, min_frac, min_floor):
    """UMAP-reduce one class, then HDBSCAN. Returns (n_clusters, noise_frac,
    largest_frac, sizes) with sizes largest first."""
    from sklearn.cluster import HDBSCAN

    n = len(f)
    mcs = max(min_floor, int(round(min_frac * n)))
    if n < 2 * mcs:                            # too few points to split
        return 1, 0.0, 1.0, [n]

    # per-class UMAP: fit on this class alone so within-class structure spreads out.
    import umap
    nn = min(umap_neighbors, n - 1)
    emb = umap.UMAP(n_components=min(umap_dim, n - 2), n_neighbors=nn,
                    min_dist=0.0, metric="cosine",
                    random_state=0).fit_transform(f)

    # allow_single_cluster: a genuinely unimodal class should come back as 1, not
    # as all-noise / zero clusters.
    labels = HDBSCAN(min_cluster_size=mcs, metric="euclidean",
                     allow_single_cluster=True).fit_predict(emb)
    noise = int((labels == -1).sum())
    sizes = sorted(
        (int((labels == c).sum()) for c in set(labels) if c != -1), reverse=True)
    n_clusters = len(sizes)
    if n_clusters == 0:                        # diffuse blob with no dense core
        n_clusters, sizes = 1, [n - noise]
    return n_clusters, noise / n, (sizes[0] / n if sizes else 0.0), sizes


# ----------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="checkpoint dirs, e.g. trained_models/simclr/cifar10/1000ep")
    p.add_argument("--data_path", default="./datasets")
    p.add_argument("--epochs", nargs="+", type=int, default=[0, 499, 999],
                   help="which checkpoint epochs to cluster (first/middle/last)")
    p.add_argument("--split", choices=["train", "val"], default="train",
                   help="which split to cluster (train is denser -> more reliable)")
    p.add_argument("--umap_dim", type=int, default=5,
                   help="UMAP target dimensionality before HDBSCAN")
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--min_frac", type=float, default=0.05,
                   help="min sub-cluster size as a fraction of the class")
    p.add_argument("--min_floor", type=int, default=15,
                   help="absolute floor for min sub-cluster size")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--force_extract", action="store_true",
                   help="re-extract features even if a cache file exists")
    p.add_argument("--output_dir", default="analysis")
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "feats"; cache_dir.mkdir(exist_ok=True)

    # ---- stage 1: make sure every (run, epoch) has cached features
    print("stage 1: features (extract only what is missing)", flush=True)
    catalog = []                               # (run_name, epoch, npz_path)
    for run in args.runs:
        catalog += extract_to_cache(
            Path(run), args.epochs, args.split, args.data_path,
            args.batch_size, args.num_workers, cache_dir, args.force_extract)

    # ---- stage 2: cluster each class from the cache
    print("\nstage 2: per-class UMAP -> HDBSCAN", flush=True)
    rows = []
    for run_name, ep, npz in catalog:
        d = np.load(npz)
        f, y = d["feats"], d["labels"]
        n_list = []
        for c in np.unique(y):
            fc = f[y == c]
            nc, nz, lg, sizes = cluster_one_class(
                fc, args.umap_dim, args.umap_neighbors, args.min_frac, args.min_floor)
            rows.append(dict(
                run=run_name, epoch=ep, label=int(c), n_points=len(fc),
                n_clusters=nc, noise_frac=nz, largest_frac=lg, sizes=sizes))
            n_list.append(nc)
        n_arr = np.array(n_list)
        print(f"  {run_name:16s} ep={ep:4d}  classes={len(n_arr):3d}  "
              f"mean={n_arr.mean():.2f} med={int(np.median(n_arr))} "
              f"max={n_arr.max()}  split={(n_arr > 1).mean():.2f}", flush=True)

    # ---- csv
    import csv
    keys = ["run", "epoch", "label", "n_points", "n_clusters",
            "noise_frac", "largest_frac", "sizes"]
    with open(out / "clusters.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            row = dict(r); row["sizes"] = " ".join(map(str, r["sizes"]))
            w.writerow({k: row[k] for k in keys})
    print(f"\nwrote {out / 'clusters.csv'}", flush=True)

    # ---- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted({r["run"] for r in rows})
    eps = sorted({r["epoch"] for r in rows})

    for rn in runs:                            # distribution across epochs
        data = [[r["n_clusters"] for r in rows if r["run"] == rn and r["epoch"] == e]
                for e in eps]
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, labels=[str(e) for e in eps], showmeans=True)
        plt.xlabel("epoch"); plt.ylabel("sub-clusters per class")
        plt.title(f"{rn}: per-class sub-cluster count")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(out / f"clusters_dist_{rn}.png", dpi=130); plt.close()

    last = eps[-1]
    for rn in runs:                            # per-class bar at final epoch
        pts = sorted(((r["label"], r["n_clusters"]) for r in rows
                      if r["run"] == rn and r["epoch"] == last))
        if not pts:
            continue
        labels = [l for l, _ in pts]; ncs = [n for _, n in pts]
        plt.figure(figsize=(max(6, len(labels) * 0.12), 4))
        plt.bar(range(len(labels)), ncs)
        plt.axhline(np.mean(ncs), color="r", ls="--", lw=1,
                    label=f"mean={np.mean(ncs):.2f}")
        plt.xlabel("class"); plt.ylabel("sub-clusters")
        plt.title(f"{rn}: sub-clusters per class (ep={last})")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(out / f"clusters_per_class_{rn}.png", dpi=130); plt.close()

    print("wrote clusters_dist_*.png and clusters_per_class_*.png", flush=True)


if __name__ == "__main__":
    main()
