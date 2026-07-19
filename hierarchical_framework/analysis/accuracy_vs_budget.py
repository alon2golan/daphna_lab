# Error-structure analysis for an AL run: at every budget, split the 1-NN test
# accuracy into fine (subclass) vs coarse (superclass) accuracy, and break the
# errors into "near miss" (right superclass, wrong subclass) vs "far miss"
# (wrong superclass). Also dumps a per-class-per-budget table. Reuses the exact
# same selectors as run_active_learning.py so the numbers match the probcover results.

import argparse
import sys
from pathlib import Path

import numpy as np

# hierarchical_framework/ root (analysis/ -> up one); make the pipeline package
# importable so we share the identical selectors + delta resolution, figures->outputs/
HF_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = HF_ROOT / "outputs"
if str(HF_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(HF_ROOT / "pipeline"))

from selection_strategies import SELECTORS


# canonical torchvision CIFAR-100 fine(0-99) -> coarse(0-19) superclass map.
# index = fine label id, value = its superclass id. verified 20 groups x 5.
CIFAR100_COARSE = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
], dtype=np.int64)


def eval_1nn_preds(train_feats, train_labels, labeled_idx, test_feats):
    # deep_al's NN1_normalized: 1-NN on the (already L2-normalized) labeled feats.
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=1)
    # fit on the labeled subset, return raw predictions for the whole test split
    clf.fit(train_feats[labeled_idx], train_labels[labeled_idx])
    return clf.predict(test_feats)


def eval_oracle_preds(train_feats, train_labels, labeled_idx, test_feats,
                      test_fine, coarse_map):
    # "superclass known in advance" upper bound: for each test point we're TOLD
    # its true superclass for free, so 1-NN only searches labeled points inside
    # that superclass. this can never miss the coarse target, so any remaining
    # error is a pure within-family (sibling) mistake. the gain over plain 1-NN
    # is exactly how much a perfect coarse oracle would buy you.
    from sklearn.neighbors import KNeighborsClassifier
    lab_feats = train_feats[labeled_idx]
    lab_fine = train_labels[labeled_idx]
    lab_coarse = coarse_map[lab_fine]
    test_coarse = coarse_map[test_fine]
    preds = np.empty(len(test_fine), dtype=lab_fine.dtype)
    for s in np.unique(test_coarse):
        # candidate neighbors = labeled points whose class lives in superclass s
        cand = lab_coarse == s
        tmask = test_coarse == s
        if cand.any():
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(lab_feats[cand], lab_fine[cand])
            preds[tmask] = clf.predict(test_feats[tmask])
        else:
            # no labeled point in this superclass yet -> fall back to global 1-NN
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(lab_feats, lab_fine)
            preds[tmask] = clf.predict(test_feats[tmask])
    return preds


def round_metrics(preds, test_fine, coarse_map, num_classes):
    # map both prediction and truth up to the superclass level
    pred_coarse = coarse_map[preds]
    true_coarse = coarse_map[test_fine]
    # overall fine (subclass) and coarse (superclass) accuracy, in percent
    fine_acc = 100.0 * float((preds == test_fine).mean())
    coarse_acc = 100.0 * float((pred_coarse == true_coarse).mean())
    # look only at the points the fine classifier got wrong
    err = preds != test_fine
    n_err = int(err.sum())
    # split those errors into two buckets that sum to all errors:
    #   near = same superclass (right family, wrong member)
    #   far  = different superclass entirely
    same = pred_coarse == true_coarse
    near = int((err & same).sum())
    far = n_err - near
    near_frac = (near / n_err) if n_err else 0.0
    far_frac = (far / n_err) if n_err else 0.0
    # per-fine-class recall (accuracy among test points whose true class is c)
    per_class = np.full(num_classes, np.nan)
    for c in range(num_classes):
        m = test_fine == c
        if m.any():
            per_class[c] = 100.0 * float((preds[m] == c).mean())
    return dict(fine_acc=fine_acc, coarse_acc=coarse_acc, n_err=n_err,
                near_frac=near_frac, far_frac=far_frac, per_class=per_class)


def curve_figure(rows, out_base):
    import matplotlib
    matplotlib.use("Agg")                       # render to file, no display
    import matplotlib.pyplot as plt
    xs = [r["num_labeled"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # left: fine vs coarse vs oracle accuracy as the budget grows. oracle = fine
    # accuracy if the true superclass were handed to you for free, so the gap
    # (oracle - fine) is exactly what perfect coarse knowledge would add.
    ax[0].plot(xs, [r["fine_acc"] for r in rows], "o-", color="steelblue", label="fine (subclass)")
    ax[0].plot(xs, [r["coarse_acc"] for r in rows], "s-", color="darkorange", label="coarse (superclass)")
    ax[0].plot(xs, [r.get("oracle_acc", np.nan) for r in rows], "d--", color="purple", label="fine | superclass known")
    ax[0].set_xlabel("num labeled"); ax[0].set_ylabel("test accuracy (%)")
    ax[0].set_title("fine vs coarse vs superclass-oracle accuracy"); ax[0].legend(); ax[0].grid(alpha=0.3)
    # right: the two error buckets, which sum to 100% of the mistakes
    ax[1].plot(xs, [100 * r["near_frac"] for r in rows], "o-", color="seagreen", label="near (same superclass)")
    ax[1].plot(xs, [100 * r["far_frac"] for r in rows], "^-", color="firebrick", label="far (other superclass)")
    ax[1].set_xlabel("num labeled"); ax[1].set_ylabel("share of errors (%)")
    ax[1].set_title("error type per budget"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    figdir = run_dir(out_base)
    png = figdir / "errorcurve.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    print(f"saved -> {png}", flush=True)
    return png


def run_dir(out_base):
    # one folder per run under outputs/figures/error_analysis/, so files stay grouped
    d = OUTPUTS / "figures" / "error_analysis" / out_base
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_tables(rows, num_classes, coarse_map, out_base):
    figdir = run_dir(out_base)
    # summary: one row per budget with the aggregate metrics
    summ = figdir / "summary.csv"
    with open(summ, "w") as fh:
        fh.write("num_labeled,budget_per_class,fine_acc,coarse_acc,oracle_acc,"
                 "n_errors,near_miss_frac,far_miss_frac\n")
        for r in rows:
            fh.write(f"{r['num_labeled']},{r['bpc']:.3f},{r['fine_acc']:.4f},"
                     f"{r['coarse_acc']:.4f},{r.get('oracle_acc', float('nan')):.4f},"
                     f"{r['n_err']},{r['near_frac']:.4f},{r['far_frac']:.4f}\n")
    print(f"saved -> {summ}", flush=True)
    # long-format per-class table: (num_labeled, class, coarse, recall) one row each
    perc = figdir / "perclass.csv"
    with open(perc, "w") as fh:
        fh.write("num_labeled,class,coarse,recall\n")
        for r in rows:
            pc = r["per_class"]
            for c in range(num_classes):
                val = "" if np.isnan(pc[c]) else f"{pc[c]:.4f}"
                fh.write(f"{r['num_labeled']},{c},{int(coarse_map[c])},{val}\n")
    print(f"saved -> {perc}", flush=True)


def slope_stats(rows):
    # per consecutive budget pair, the slope (accuracy gained per 100 labels) of
    # each curve, plus how far apart the two curves are (the vertical gap).
    segs = []
    for a, b in zip(rows[:-1], rows[1:]):
        dx = (b["num_labeled"] - a["num_labeled"]) / 100.0   # width in 100-label units
        fine_slope = (b["fine_acc"] - a["fine_acc"]) / dx
        coarse_slope = (b["coarse_acc"] - a["coarse_acc"]) / dx
        segs.append(dict(
            x_mid=(a["num_labeled"] + b["num_labeled"]) / 2,   # segment midpoint for plotting
            num_labeled=b["num_labeled"],                      # step to log the slope at
            fine_slope=fine_slope, coarse_slope=coarse_slope,
            slope_diff=coarse_slope - fine_slope,              # 0 => the two rise identically
        ))
    # gap = coarse - fine at every budget (constant gap => "same curve, shifted up")
    gaps = np.array([r["coarse_acc"] - r["fine_acc"] for r in rows])
    fs = np.array([s["fine_slope"] for s in segs])
    cs = np.array([s["coarse_slope"] for s in segs])
    # correlation of the matching slopes: 1.0 => the two curves bend together
    corr = float(np.corrcoef(fs, cs)[0, 1]) if len(fs) > 1 and fs.std() and cs.std() else float("nan")
    summary = dict(gap_mean=float(gaps.mean()), gap_std=float(gaps.std()),
                   slope_corr=corr, slope_diff_mean=float((cs - fs).mean()),
                   slope_diff_std=float((cs - fs).std()))
    return segs, gaps, summary


def run(embeddings_dir, strategy="probcover", delta="auto_delta", budget=100, rounds=10,
        split="train", seed=0, dataset="cifar100", coarse_map_path=None,
        use_wandb=False, wandb_kwargs=None):
    emb = Path(embeddings_dir)
    # pool features/labels selection runs over
    train_feats = np.load(emb / f"{split}_features_normalized.npy").astype(np.float32)
    train_labels = np.load(emb / f"{split}_labels.npy")
    # test features/labels used only to score
    test_feats = np.load(emb / "test_features_normalized.npy").astype(np.float32)
    test_labels = np.load(emb / "test_labels.npy")
    num_classes = int(train_labels.max()) + 1
    # fine->coarse map: explicit file (any dataset) else the built-in CIFAR-100 one.
    if coarse_map_path:
        coarse_map = np.load(coarse_map_path)
    elif dataset == "cifar100":
        coarse_map = CIFAR100_COARSE
    else:
        raise SystemExit(f"no coarse map for dataset={dataset}; pass --coarse_map")

    # resolve --delta the deep_al way (explicit float | knee_delta | auto_delta table);
    # dataset keys the auto_delta lookup, so it matches the run_active_learning run
    from delta_selection import resolve_delta
    delta = resolve_delta(delta, train_feats, num_classes, seed=seed, dataset=dataset)
    # same selector construction as run_active_learning.py, so the picks are identical
    selector = SELECTORS[strategy](train_feats, delta=delta, seed=seed)
    print(f"strategy={strategy} on {train_feats.shape[0]} points", flush=True)

    if use_wandb:
        import wandb
        wandb.init(**(wandb_kwargs or {}))
        # every curve shares num_labeled as its x-axis, so they overlay cleanly
        for k in ("fine_acc", "coarse_acc", "oracle_acc", "gap", "oracle_gain",
                  "near_pct", "far_pct"):
            wandb.define_metric(k, step_metric="num_labeled")

    labeled, rows = [], []
    for r in range(rounds):
        # this round's picks; the selector mutates its own covered/selected state
        chosen = selector.select_round(budget)
        labeled.extend(chosen)
        # 1-NN predictions on the whole test split from the current labeled set
        lab = np.array(labeled)
        preds = eval_1nn_preds(train_feats, train_labels, lab, test_feats)
        m = round_metrics(preds, test_labels, coarse_map, num_classes)
        # oracle accuracy: same labels, but the true superclass is given for free
        oracle = eval_oracle_preds(train_feats, train_labels, lab, test_feats,
                                   test_labels, coarse_map)
        m["oracle_acc"] = 100.0 * float((oracle == test_labels).mean())
        m["num_labeled"] = len(labeled)
        m["bpc"] = len(labeled) / num_classes            # budget per class (paper's "B")
        rows.append(m)
        print(f"round {r}: labeled={len(labeled)} fine={m['fine_acc']:.2f} "
              f"coarse={m['coarse_acc']:.2f} oracle={m['oracle_acc']:.2f} "
              f"near={100*m['near_frac']:.1f}% far={100*m['far_frac']:.1f}%", flush=True)
        if use_wandb:
            import wandb
            # the accuracy trio + gaps + the three-way error-type split
            wandb.log({"num_labeled": m["num_labeled"], "budget_per_class": m["bpc"],
                       "fine_acc": m["fine_acc"], "coarse_acc": m["coarse_acc"],
                       "oracle_acc": m["oracle_acc"],
                       "gap": m["coarse_acc"] - m["fine_acc"],
                       "oracle_gain": m["oracle_acc"] - m["fine_acc"],
                       "near_pct": 100 * m["near_frac"], "far_pct": 100 * m["far_frac"]})

    # slope of each curve between consecutive budgets + how alike the two are
    segs, gaps, summary = slope_stats(rows)
    if use_wandb:
        import wandb
        wandb.define_metric("fine_slope", step_metric="num_labeled")
        wandb.define_metric("coarse_slope", step_metric="num_labeled")
        wandb.define_metric("slope_diff", step_metric="num_labeled")
        # log the per-segment slopes as their own overlaid curves
        for s in segs:
            wandb.log({"num_labeled": s["num_labeled"], "fine_slope": s["fine_slope"],
                       "coarse_slope": s["coarse_slope"], "slope_diff": s["slope_diff"]})
        # scalars: constant gap + slope_corr~1 == "same shape, just shifted up"
        wandb.summary.update(summary)
        print(f"gap_mean={summary['gap_mean']:.2f} gap_std={summary['gap_std']:.2f} "
              f"slope_corr={summary['slope_corr']:.3f}", flush=True)

    # output path mirrors embeddings/<dataset>/<technique>/<old_deep_al|new_solo_learn>;
    # strategy + budget are defaults (tracked in wandb), so left out of the folder name
    ds, technique, family = emb.parts[-3], emb.parts[-2], emb.parts[-1]
    out_base = f"{ds}/{technique}/{family}_d{delta}"
    png = curve_figure(rows, out_base)
    write_tables(rows, num_classes, coarse_map, out_base)
    if use_wandb:
        import wandb
        xs = [r["num_labeled"] for r in rows]
        # native interactive multi-line charts (not images): the accuracy trio and
        # the three-way error split, each rendered as one overlaid line panel.
        wandb.log({"accuracy_curves": wandb.plot.line_series(
            xs=xs, ys=[[r["fine_acc"] for r in rows], [r["coarse_acc"] for r in rows],
                       [r["oracle_acc"] for r in rows]],
            keys=["fine (subclass)", "coarse (superclass)", "fine | superclass known"],
            title="accuracy vs budget", xname="num_labeled")})
        wandb.log({"error_type_curves": wandb.plot.line_series(
            xs=xs, ys=[[100 * r["near_frac"] for r in rows], [100 * r["far_frac"] for r in rows]],
            keys=["near (same superclass)", "far (other superclass)"],
            title="error type share vs budget", xname="num_labeled")})
        # slopes as their own overlaid chart (fine vs coarse gain per +100 labels)
        sx = [s["num_labeled"] for s in segs]
        wandb.log({"slope_curves": wandb.plot.line_series(
            xs=sx, ys=[[s["fine_slope"] for s in segs], [s["coarse_slope"] for s in segs]],
            keys=["fine slope", "coarse slope"], title="slope per segment", xname="num_labeled")})
        wandb.finish()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", required=True,
                   help="dir with <split>_features_normalized.npy + labels + test_*")
    p.add_argument("--strategy", default="probcover", choices=list(SELECTORS))
    p.add_argument("--delta", default="auto_delta",
                   help="a float, 'knee_delta', or 'auto_delta' (default: pinned table, knee fallback)")
    p.add_argument("--budget", type=int, default=100, help="points labeled per round")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="cifar100", help="chooses the built-in fine->coarse map")
    p.add_argument("--coarse_map", default=None,
                   help="path to a (num_classes,) fine->coarse .npy (overrides --dataset)")
    # wandb: interactive curves; default project for these prior/oracle studies
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="applying_priors")
    p.add_argument("--name", default=None)
    p.add_argument("--group", default="error_analysis")
    args = p.parse_args()
    # wandb run name = <dataset>_<technique>_<family>, so overlaying old vs new is trivial
    p = Path(args.embeddings_dir)
    tag = "_".join(p.parts[-3:])
    name = args.name or f"{tag}_{args.strategy}_d{args.delta}"
    wandb_kwargs = dict(project=args.project, name=name, group=args.group,
                        config=dict(embeddings_dir=args.embeddings_dir, strategy=args.strategy,
                                    delta=args.delta, budget=args.budget, rounds=args.rounds,
                                    split=args.split, seed=args.seed))
    run(args.embeddings_dir, strategy=args.strategy, delta=args.delta,
        budget=args.budget, rounds=args.rounds, split=args.split,
        seed=args.seed, dataset=args.dataset, coarse_map_path=args.coarse_map,
        use_wandb=args.wandb, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    main()
