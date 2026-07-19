"""Rank propagation thresholds BEFORE spending SupCon runs on them.

The propagation condition only helps if its pseudo-labels are both plentiful
(they add supervised signal) and pure (wrong labels pull the wrong classes
together and can hurt). The right (kernel, bandwidth, sim_threshold,
conf_threshold) is not obvious and depends on the feature scale, so this tool
sweeps a grid -- using only the frozen features, no training -- and ranks every
combination by how much *net* correct supervision it would add.

For each combination it reports, on the labeled set of a given per-class budget:
    n_pseudo    how many points get a hard pseudo-label
    purity      fraction of those that are correct (uses true labels, for
                selection only -- never fed to training)
    coverage    n_pseudo / (pool - labeled)
    net_correct n_pseudo * (purity - lambda * (1 - purity))
                = expected correct-minus-penalised-wrong labels added. With
                lambda=1 this is n_pseudo*(2*purity-1): positive only when the
                pseudo-set is better than a coin flip, and larger when a pure set
                also has reach. This is the ranking key.

It then prints a ranked table, a short "why" (how purity/coverage move with each
threshold, and whether ANY combo is worth training), and writes the full grid +
a recommended shortlist to json -- feed that shortlist to supcon_finetune's
list-valued --sim_threshold / --conf_threshold / --bandwidth to train only the
combinations that look worth it.

    python -m ssl_framework.algorithms.threshold_search \
        --features results/cifar100/simclr/base_features \
        --dataset cifar100 --per_class_budget 10 --seed 1 \
        --kernels gaussian,laplacian,cosine \
        --bandwidths auto,0.5x,2x --sim_thresholds 0.0,0.1,0.2 \
        --conf_thresholds 0.6,0.8,0.9,0.95
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from ..backends import solo_learn
from ..labeling import RandomSelector, class_counts
from .. import propagation as prop

_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def _floats(s):
    return [float(x) for x in str(s).split(",") if x != ""]


def _load_features(args):
    if args.features:
        d = Path(args.features)
        return np.load(d / "train_features.npy"), np.load(d / "train_labels.npy")
    backbone = solo_learn.load_backbone_from_checkpoint(args.base_checkpoint)
    import torch
    backbone = backbone.to("cuda" if torch.cuda.is_available() else "cpu")
    emb = solo_learn.extract_embeddings(backbone, args.dataset, args.data_dir,
                                        split="train", num_workers=args.num_workers)
    return emb["train"]


def _resolve_bandwidths(tokens, median):
    """Each token is 'auto', a bare number (absolute h), or '<k>x' (k*median)."""
    out = []
    for t in [t.strip() for t in str(tokens).split(",") if t.strip()]:
        if t == "auto":
            out.append(("auto", median))
        elif t.endswith("x"):
            out.append((t, float(t[:-1]) * median))
        else:
            out.append((t, float(t)))
    return out


def rank_combos(feats, labels, labeled, num_classes, *, kernels, bandwidths,
                sims, confs, agg="max", confidence_metric="prob", wrong_penalty=1.0):
    """Score every (kernel, bandwidth, sim, conf) combo on the frozen features and
    return rows sorted by net_correct (expected correct-minus-penalised-wrong
    pseudo-labels). `bandwidths` is a list of (token, value); cosine ignores it.
    Reused by both the CLI report and supcon_finetune's --auto_thresholds.
    """
    labels = np.asarray(labels)
    labeled_targets = labels[np.asarray(labeled)]
    n_unlabeled = max(1, len(feats) - len(labeled))
    rows = []
    for kernel in kernels:
        kbws = [bandwidths[0]] if kernel == "cosine" else bandwidths
        for (bw_tok, bw), sim, conf in itertools.product(kbws, sims, confs):
            pr = prop.propagate(feats, labeled, labeled_targets, num_classes,
                                kernel=kernel, bandwidth=bw, sim_threshold=sim,
                                conf_threshold=conf, agg=agg,
                                confidence_metric=confidence_metric, true_targets=labels)
            n = pr.stats["n_pseudo"]
            acc = pr.stats.get("pseudo_accuracy", 0.0) if n else 0.0
            net = n * (acc - wrong_penalty * (1 - acc))
            rows.append({"kernel": kernel,
                         "bw_token": ("-" if kernel == "cosine" else bw_tok),
                         "bandwidth": None if kernel == "cosine" else round(float(bw), 4),
                         "sim_threshold": sim, "conf_threshold": conf,
                         "n_pseudo": int(n), "purity": round(float(acc), 4),
                         "coverage": round(n / n_unlabeled, 4),
                         "net_correct": round(float(net), 1)})
    rows.sort(key=lambda r: r["net_correct"], reverse=True)
    return rows


def select_per_kernel(rows, kernels, *, per_kernel=1, min_purity=0.6):
    """Best `per_kernel` combos for each kernel, keeping only those that add
    positive net-correct supervision above a purity floor. Returns a flat list."""
    chosen = []
    for k in kernels:
        good = [r for r in rows if r["kernel"] == k
                and r["net_correct"] > 0 and r["purity"] >= min_purity]
        chosen.extend(good[:per_kernel])
    return chosen


def run(args):
    num_classes = _NUM_CLASSES[args.dataset]
    feats, labels = _load_features(args)
    labels = np.asarray(labels)
    n_pool = feats.shape[0]
    labeled = RandomSelector().select(n_pool, args.n_labeled, args.seed)
    labeled_targets = labels[labeled]
    counts = class_counts(labeled, labels, num_classes)
    median = prop.median_bandwidth(feats)
    n_unlabeled = n_pool - len(labeled)

    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    bandwidths = _resolve_bandwidths(args.bandwidths, median)
    sims = _floats(args.sim_thresholds)
    confs = _floats(args.conf_thresholds)
    lam = args.wrong_penalty

    print(f"dataset={args.dataset} L={args.per_class_budget} n_labeled={len(labeled)} "
          f"(empty classes {int((counts==0).sum())}) | median dist={median:.3g} | "
          f"grid={len(kernels)}x{len(bandwidths)}x{len(sims)}x{len(confs)}\n")

    rows = rank_combos(feats, labels, labeled, num_classes, kernels=kernels,
                       bandwidths=bandwidths, sims=sims, confs=confs, agg=args.agg,
                       confidence_metric=args.confidence_metric, wrong_penalty=lam)

    print(f"{'rank':>4} {'kernel':<9}{'bw':>7}{'sim':>6}{'conf':>6}"
          f"{'n_pseudo':>9}{'purity':>8}{'cover':>7}{'net_correct':>12}")
    for i, r in enumerate(rows[:args.top], 1):
        print(f"{i:>4} {r['kernel']:<9}{str(r['bw_token']):>7}{r['sim_threshold']:>6.2g}"
              f"{r['conf_threshold']:>6.2g}{r['n_pseudo']:>9}{r['purity']:>8.3f}"
              f"{r['coverage']:>7.3f}{r['net_correct']:>12.1f}")

    _explain(rows, sims, confs, lam)

    # recommended shortlist: best per kernel above the purity floor with net > 0.
    shortlist = select_per_kernel(rows, kernels, per_kernel=1, min_purity=args.min_purity)
    _print_shortlist(shortlist)

    out = {"dataset": args.dataset, "per_class_budget": args.per_class_budget,
           "seed": args.seed, "n_labeled": len(labeled), "median_distance": median,
           "wrong_penalty": lam, "grid": rows, "recommended": shortlist}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nfull grid + recommendation -> {args.out}")
    return out


def _explain(rows, sims, confs, lam):
    print("\nwhy:")
    if not rows or rows[0]["net_correct"] <= 0:
        print(f"  NO combination has positive net_correct (lambda={lam}): every "
              "pseudo-set adds more wrong-than-right supervision at this budget. "
              "Propagation is unlikely to help here -- rely on supcon_only, raise "
              "the budget, or raise conf_threshold further.")
        return
    best = rows[0]
    print(f"  best: {best['kernel']} bw={best['bw_token']} sim={best['sim_threshold']:g} "
          f"conf={best['conf_threshold']:g} -> {best['n_pseudo']} pseudo at "
          f"{best['purity']:.2f} purity (+{best['net_correct']:.0f} net correct).")
    # trend of purity/coverage with each threshold (averaged over the grid)
    def trend(key):
        vals = sorted({r[key] for r in rows})
        line = []
        for v in vals:
            sub = [r for r in rows if r[key] == v]
            line.append(f"{v:g}:(pur {np.mean([r['purity'] for r in sub]):.2f}, "
                        f"cov {np.mean([r['coverage'] for r in sub]):.2f})")
        return "  ".join(line)
    print(f"  raising conf_threshold  {trend('conf_threshold')}")
    print(f"  raising sim_threshold   {trend('sim_threshold')}")
    print("  (purity should rise and coverage fall as either threshold rises; the "
          "winner is the sweet spot where reach survives while wrong labels are cut.)")


def _print_shortlist(shortlist):
    print("\nrecommended for training (best positive-net per kernel):")
    if not shortlist:
        print("  none passed the purity floor with positive net_correct.")
        return
    kernels = ",".join(r["kernel"] for r in shortlist)
    sims = ",".join(f"{r['sim_threshold']:g}" for r in shortlist)
    confs = ",".join(f"{r['conf_threshold']:g}" for r in shortlist)
    bws = ",".join(str(r["bw_token"]) for r in shortlist if r["kernel"] != "cosine") or "auto"
    for r in shortlist:
        print(f"  {r['kernel']:<9} bw={r['bw_token']} sim={r['sim_threshold']:g} "
              f"conf={r['conf_threshold']:g}  (n_pseudo={r['n_pseudo']} "
              f"purity={r['purity']:.2f} net=+{r['net_correct']:.0f})")
    print("\n  feed to a training run, e.g.:")
    print(f"    --kernels {kernels} --bandwidth {bws} \\")
    print(f"    --sim_threshold {sims} --conf_threshold {confs}")
    print("  (list-valued: each kernel/threshold combination becomes its own "
          "supcon_prop leaf + wandb run.)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="dir with train_features.npy/train_labels.npy")
    src.add_argument("--base_checkpoint", help="encoder run dir / .ckpt to extract from")
    p.add_argument("--dataset", default="cifar100", choices=list(_NUM_CLASSES))
    p.add_argument("--data_dir", default=str(solo_learn.DEFAULT_DATA_DIR))
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_labeled", type=int, default=None,
                   help="total labeled size; if omitted, derived from --per_class_budget")
    p.add_argument("--per_class_budget", type=int, required=True,
                   help="per-class budget L (n_labeled = L * num_classes if --n_labeled omitted)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--kernels", default="gaussian,laplacian,cosine")
    p.add_argument("--bandwidths", default="auto,0.5x,0.25x,0.1x",
                   help="comma list; each is 'auto', a number (absolute h), or '<k>x' (k*median). "
                        "Tighter (smaller) bandwidths give purer pseudo-labels")
    p.add_argument("--sim_thresholds", default="0.0,0.1,0.2,0.3",
                   help="higher figures floors cut weak, impure votes")
    p.add_argument("--conf_thresholds", default="0.8,0.9,0.95")
    p.add_argument("--agg", default="max", choices=list(prop.AGGREGATIONS))
    p.add_argument("--confidence_metric", default="prob", choices=list(prop.CONFIDENCE_METRICS))
    p.add_argument("--wrong_penalty", type=float, default=3.0,
                   help="lambda: cost of a wrong pseudo-label relative to a correct one "
                        "(higher = the ranking favours purity over coverage)")
    p.add_argument("--min_purity", type=float, default=0.9,
                   help="purity floor for the recommended shortlist")
    p.add_argument("--recommend", type=int, default=3, help="max combos in the shortlist")
    p.add_argument("--top", type=int, default=15, help="rows to print in the ranked table")
    p.add_argument("--out", default=None, help="json path for the full grid + recommendation")
    args = p.parse_args()
    if args.n_labeled is None:
        args.n_labeled = args.per_class_budget * _NUM_CLASSES[args.dataset]
    run(args)


if __name__ == "__main__":
    main()
