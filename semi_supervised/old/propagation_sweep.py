"""Preview kernel label propagation across a hyperparameter grid -- no training.

Choosing bandwidth / sim_threshold / conf_threshold blind is wasteful: each
combination changes the pseudo-labeled set, and a full SupCon run per combo is
expensive. This tool runs ONLY the propagation step (cheap: a kernel over frozen
features) across a grid and reports, per combination, how many pseudo-labels it
would produce and -- using the held-out true labels for diagnostics only -- how
accurate they are. Read the table, pick the promising corners of the grid, then
launch the real supcon_finetune runs there.

It reads the base features saved by a supcon_finetune run (output_dir/base_features)
or extracts them fresh from a checkpoint. The labeled set is reconstructed from
the same (n_labeled, seed) so it matches the run you will launch.

    python -m semi_supervised.algorithms.propagation_sweep \
        --features runs/supcon_ft_cifar100_l1000/base_features \
        --dataset cifar100 --n_labeled 1000 --seed 1 \
        --kernel gaussian \
        --bandwidth 2,5,10,20 --sim_threshold 0.0,0.1,0.2 --conf_threshold 0.8,0.9,0.95
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from ..backends import solo_learn
from ..labeling import RandomSelector
from .. import propagation as prop

_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def _floats(s):
    return [float(x) for x in str(s).split(",") if x != ""]


def _load_features(args):
    """Base train features + labels, from a saved dir or a fresh extraction."""
    if args.features:
        d = Path(args.features)
        return np.load(d / "train_features.npy"), np.load(d / "train_labels.npy")
    backbone = solo_learn.load_backbone_from_checkpoint(args.base_checkpoint)
    import torch
    backbone = backbone.to("cuda" if torch.cuda.is_available() else "cpu")
    emb = solo_learn.extract_embeddings(backbone, args.dataset, args.data_dir,
                                        split="train", num_workers=args.num_workers)
    return emb["train"]


def run(args):
    num_classes = _NUM_CLASSES[args.dataset]
    feats, labels = _load_features(args)
    n_pool = feats.shape[0]
    labeled = RandomSelector().select(n_pool, args.n_labeled, args.seed)
    labeled_targets = np.asarray(labels)[labeled]

    grid = list(itertools.product(_floats(args.bandwidth),
                                  _floats(args.sim_threshold),
                                  _floats(args.conf_threshold)))
    rows = []
    print(f"pool={n_pool} labeled={len(labeled)} kernel={args.kernel} | {len(grid)} combos\n")
    print(f"{'bandwidth':>10}{'sim_thr':>10}{'conf_thr':>10}"
          f"{'n_pseudo':>10}{'pseudo_acc':>12}{'no_evid':>10}")
    for h, st, ct in grid:
        r = prop.propagate(feats, labeled, labeled_targets, num_classes,
                           kernel=args.kernel, bandwidth=h, sim_threshold=st,
                           conf_threshold=ct, agg=args.agg,
                           confidence_metric=args.confidence_metric,
                           max_pseudo=args.max_pseudo, true_targets=labels)
        s = r.stats
        acc = s.get("pseudo_accuracy")
        rows.append({"bandwidth": h, "sim_threshold": st, "conf_threshold": ct, **s})
        print(f"{h:>10.3g}{st:>10.3g}{ct:>10.3g}"
              f"{s['n_pseudo']:>10}{(acc if acc is not None else float('nan')):>12.4f}"
              f"{s['n_no_evidence']:>10}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nsaved -> {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="dir with train_features.npy/train_labels.npy")
    src.add_argument("--base_checkpoint", help="encoder run dir / .ckpt to extract from")
    p.add_argument("--dataset", default="cifar100", choices=list(_NUM_CLASSES))
    p.add_argument("--data_dir", default=str(solo_learn.DEFAULT_DATA_DIR))
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_labeled", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--kernel", default="gaussian", choices=list(prop.KERNELS))
    p.add_argument("--bandwidth", default="2,5,10,20", help="comma-separated h values")
    p.add_argument("--sim_threshold", default="0.0,0.1,0.2", help="comma-separated floors")
    p.add_argument("--conf_threshold", default="0.8,0.9,0.95", help="comma-separated cutoffs")
    p.add_argument("--agg", default="max", choices=list(prop.AGGREGATIONS))
    p.add_argument("--confidence_metric", default="prob", choices=list(prop.CONFIDENCE_METRICS))
    p.add_argument("--max_pseudo", type=int, default=None)
    p.add_argument("--out", default=None, help="optional json path for the table")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
