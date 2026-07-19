# Strategy-agnostic active-learning runner: each round asks a Selector for
# `budget` points, reveals their labels, scores with 1-NN, logs to wandb.
# Swap --strategy to change WHICH points get picked; this loop stays the same.

import argparse
from pathlib import Path

import numpy as np

from selectors_al import SELECTORS

# with_priors/ root (pipeline/ -> up one); all rendered figures land under outputs/
OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"


def eval_1nn(train_feats, train_labels, labeled_idx, test_feats, test_labels):
    # deep_al's NN1_normalized: 1-NN on the (already L2-normalized) labeled features
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=1)
    # train points = the labeled subset, predict the whole test split
    clf.fit(train_feats[labeled_idx], train_labels[labeled_idx])
    preds = clf.predict(test_feats)
    # accuracy in percent
    return 100.0 * float((preds == test_labels).mean())


def class_pick_report(picks_by_round, num_classes, out_base, use_wandb=False):
    import matplotlib
    matplotlib.use("Agg")                 # render to file, no display
    import matplotlib.pyplot as plt
    R = len(picks_by_round)
    # counts[c, r] = how many class-c points were chosen in round r
    counts = np.zeros((num_classes, R), dtype=int)
    for r, classes in enumerate(picks_by_round):
        for c in classes:
            counts[c, r] += 1
    # total picks per class over the whole run
    total = counts.sum(axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # left: which class got picked in which round
    im = ax[0].imshow(counts, aspect="auto", cmap="viridis", interpolation="nearest")
    ax[0].set_xlabel("round"); ax[0].set_ylabel("class")
    ax[0].set_title("picks per class per round")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    # right: how often each class was chosen in total
    ax[1].bar(np.arange(num_classes), total, color="steelblue")
    ax[1].set_xlabel("class"); ax[1].set_ylabel("total picks")
    ax[1].set_title(f"picks per class (distinct chosen={int((total > 0).sum())}/{num_classes})")
    fig.tight_layout()

    figdir = OUTPUTS / "figures" / "al_picks"
    figdir.mkdir(parents=True, exist_ok=True)
    png = figdir / f"{out_base}_classpicks.png"
    fig.savefig(png, dpi=130)
    print(f"saved -> {png}", flush=True)

    # table: for each class, its total and the exact rounds it was chosen in
    csv = figdir / f"{out_base}_classpicks.csv"
    with open(csv, "w") as fh:
        fh.write("class,total,rounds\n")
        for c in range(num_classes):
            rounds = " ".join(str(r) for r in range(R) if counts[c, r] > 0)
            fh.write(f"{c},{int(total[c])},{rounds}\n")
    print(f"saved -> {csv}", flush=True)

    if use_wandb:
        import wandb
        # log the figure and the per-class table into the run
        wandb.log({"class_picks": wandb.Image(str(png))})
        tbl = wandb.Table(columns=["class", "total", "rounds"])
        for c in range(num_classes):
            rounds = " ".join(str(r) for r in range(R) if counts[c, r] > 0)
            tbl.add_data(int(c), int(total[c]), rounds)
        wandb.log({"class_picks_table": tbl})
    plt.close(fig)
    return counts


def pick_degree_report(neighbors_by_round, bpc_by_round, out_base, use_wandb=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, means, lows, highs = [], [], [], []
    # per round: mean / min / max of the uncovered-neighbor count of the chosen points
    for bpc, scores in zip(bpc_by_round, neighbors_by_round):
        if len(scores) == 0:
            continue
        a = np.asarray(scores, dtype=float)
        xs.append(bpc); means.append(a.mean()); lows.append(a.min()); highs.append(a.max())
    xs = np.array(xs); means = np.array(means)
    # error bar spans min..max around the mean point
    yerr = np.vstack([means - np.array(lows), np.array(highs) - means])

    fig, ax = plt.subplots(figsize=(8, 5))
    # point = round average; bar = that round's min and max
    ax.errorbar(xs, means, yerr=yerr, fmt="o-", capsize=4, color="darkred", ecolor="gray")
    ax.set_xlabel("budget (labels per class, B)")
    ax.set_ylabel("uncovered neighbors of chosen points")
    ax.set_title("coverage gain per pick: round mean (bar = min..max)")
    fig.tight_layout()

    figdir = OUTPUTS / "figures" / "al_picks"
    figdir.mkdir(parents=True, exist_ok=True)
    png = figdir / f"{out_base}_pickdegree.png"
    fig.savefig(png, dpi=130)
    print(f"saved -> {png}", flush=True)
    if use_wandb:
        import wandb
        wandb.log({"pick_degree": wandb.Image(str(png))})
    plt.close(fig)


def run(embeddings_dir, strategy="probcover", delta=0.65, budget=100, rounds=10,
        split="train", seed=0, use_wandb=False, wandb_kwargs=None):
    emb = Path(embeddings_dir)
    # pool features/labels -- selection happens over these
    train_feats = np.load(emb / f"{split}_features_normalized.npy").astype(np.float32)
    train_labels = np.load(emb / f"{split}_labels.npy")
    # test features/labels -- only used to score the selection
    test_feats = np.load(emb / "test_features_normalized.npy").astype(np.float32)
    test_labels = np.load(emb / "test_labels.npy")
    # number of classes -> lets us report the budget as labels-per-class (the paper's
    # x-axis "B": e.g. 100 labeled over 100 CIFAR-100 classes = 1B)
    num_classes = int(train_labels.max()) + 1

    # resolve delta: a float, or "knee" -> deep_al's knee-point algorithm on these feats
    from knee_delta import resolve_delta
    delta = resolve_delta(delta, train_feats, num_classes, seed=seed)
    # build the chosen strategy; kwargs it doesn't use (delta/seed) are ignored
    cls = SELECTORS[strategy]
    selector = cls(train_feats, delta=delta, seed=seed)
    print(f"strategy={strategy} on {train_feats.shape[0]} points", flush=True)

    if use_wandb:
        import wandb
        wandb.init(**(wandb_kwargs or {}))
        # x-axis is the budget in labels-per-class (1B..10B), not raw count
        wandb.define_metric("test_accuracy", step_metric="budget_per_class")

    labeled = []
    # the class of each pick, grouped by round (labels read only for diagnostics)
    picks_by_round = []
    # the coverage score of each pick, grouped by round (coverage-family only)
    neighbors_by_round, bpc_by_round = [], []
    n_scores = 0                                  # how many pick_scores we've consumed
    for r in range(rounds):
        # this round's points; the selector updates its own state
        chosen = selector.select_round(budget)
        labeled.extend(chosen)
        picks_by_round.append(train_labels[np.array(chosen)] if chosen else [])
        # slice off the scores this round appended (empty for non-coverage strategies)
        neighbors_by_round.append(selector.pick_scores[n_scores:])
        n_scores = len(selector.pick_scores)
        # score the growing labeled set
        acc = eval_1nn(train_feats, train_labels, np.array(labeled), test_feats, test_labels)
        # budget expressed as average labels per class (the paper's "B")
        bpc = len(labeled) / num_classes
        bpc_by_round.append(bpc)
        # coverage exists only for the coverage-family strategies
        cov = selector.coverage() if hasattr(selector, "coverage") else None
        msg = f"round {r}: labeled={len(labeled)} ({bpc:.1f}B) test_acc={acc:.2f}"
        if cov is not None:
            msg += f" coverage={cov:.3f}"
        print(msg, flush=True)
        if use_wandb:
            import wandb
            log = {"budget_per_class": bpc, "num_labeled": len(labeled),
                   "test_accuracy": acc, "episode": r}
            if cov is not None:
                log["coverage"] = cov
            wandb.log(log)

    # class-provenance report: which class was picked when, how often
    out_base = f"{Path(embeddings_dir).name}_{strategy}_d{delta}_b{budget}"
    class_pick_report(picks_by_round, num_classes, out_base, use_wandb=use_wandb)
    # per-round coverage gain of the chosen points (skip if no scores, e.g. random)
    if any(len(s) for s in neighbors_by_round):
        pick_degree_report(neighbors_by_round, bpc_by_round, out_base, use_wandb=use_wandb)

    if use_wandb:
        import wandb
        wandb.finish()
    return labeled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", required=True,
                   help="dir with <split>_features_normalized.npy + labels + test_*")
    p.add_argument("--strategy", default="probcover", choices=list(SELECTORS),
                   help="which point-selection strategy to run")
    p.add_argument("--delta", default="0.65", help="a float, or 'knee' for deep_al's knee algorithm")
    p.add_argument("--budget", type=int, default=100, help="points labeled per round")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=0, help="tie-break seed (random strategy)")
    # wandb -- a new project, separate from deep_al's daphna_lab
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="bayesian")
    p.add_argument("--name", default=None)
    p.add_argument("--group", default=None)
    args = p.parse_args()

    # default run name/group encode the strategy and knobs
    name = args.name or f"{args.strategy}_d{args.delta}_b{args.budget}"
    group = args.group or args.strategy
    wandb_kwargs = dict(project=args.project, name=name, group=group,
                        config={"strategy": args.strategy, "delta": args.delta,
                                "budget": args.budget, "rounds": args.rounds,
                                "seed": args.seed, "embeddings_dir": args.embeddings_dir})
    run(args.embeddings_dir, strategy=args.strategy, delta=args.delta,
        budget=args.budget, rounds=args.rounds, split=args.split, seed=args.seed,
        use_wandb=args.wandb, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    main()
