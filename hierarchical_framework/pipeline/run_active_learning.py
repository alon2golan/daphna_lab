# Strategy-agnostic active-learning runner: each round asks a Selector for
# `budget` points, reveals their labels, scores with 1-NN, logs to wandb.
# Swap --strategy to change WHICH points get picked; this loop stays the same.
#
# --embeddings_dir points at one of two embedding families (see embeddings/):
#   old_embeddings/  itai's SimCLR features (the deep_al pipeline; the embeddings
#                    used in the ProbCover paper) -> the baseline to reproduce/compare
#   new_embeddings/  solo-learn features -> the current work
# Output figure/run names are prefixed with the family (old_embeddings_ / new_embeddings_).

import argparse
from pathlib import Path

import numpy as np

from selection_strategies import SELECTORS
from pick_reports import report_picks_per_class, report_coverage_gain


def eval_1nn(train_feats, train_labels, labeled_idx, test_feats, test_labels):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=1) # deep_al's NN1_normalized scorer, on the already-L2-normalized features
    clf.fit(train_feats[labeled_idx], train_labels[labeled_idx]) # fit on just the labeled subset
    preds = clf.predict(test_feats) # predict every test point
    return 100.0 * float((preds == test_labels).mean()) # accuracy in percent


def run(embeddings_dir, strategy="probcover", delta=0.65, budget=100, rounds=10,
        split="train", seed=0, use_wandb=False, wandb_kwargs=None):
    emb = Path(embeddings_dir)
    train_feats = np.load(emb / f"{split}_features_normalized.npy").astype(np.float32) # pool features -- selection happens over these
    train_labels = np.load(emb / f"{split}_labels.npy")
    test_feats = np.load(emb / "test_features_normalized.npy").astype(np.float32) # test features -- only used to score the selection
    test_labels = np.load(emb / "test_labels.npy")
    num_classes = int(train_labels.max()) + 1 # lets us report the budget as labels-per-class (the paper's x-axis "B")

    from delta_selection import resolve_delta
    delta = resolve_delta(delta, train_feats, num_classes, seed=seed) # a float, or "knee" -> deep_al's knee-point algorithm
    selector = SELECTORS[strategy](train_feats, delta=delta, seed=seed) # kwargs a strategy ignores (delta/seed) are harmless
    print(f"strategy={strategy} on {train_feats.shape[0]} points", flush=True)

    if use_wandb:
        import wandb
        wandb.init(**(wandb_kwargs or {}))
        wandb.define_metric("test_accuracy", step_metric="budget_per_class") # x-axis is labels-per-class (1B..10B), not raw count

    labeled = [] # every point labeled so far, across all rounds
    picked_classes_by_round = [] # the class of each pick, grouped by round (labels read only for diagnostics)
    coverage_gains_by_round = [] # per round, the coverage gain of each pick (empty for non-coverage strategies)
    labels_per_class_by_round = [] # per round, the running labels-per-class ("B") -- the plots' x-axis
    gains_seen = 0 # how many of the selector's coverage_gains we've already split off per round
    for r in range(rounds):
        newly_labeled = selector.select_round(budget) # this round's points; the selector updates its own state
        labeled.extend(newly_labeled)
        picked_classes_by_round.append(train_labels[np.array(newly_labeled)] if newly_labeled else [])
        coverage_gains_by_round.append(selector.coverage_gains[gains_seen:]) # just the gains this round appended
        gains_seen = len(selector.coverage_gains)
        test_acc = eval_1nn(train_feats, train_labels, np.array(labeled), test_feats, test_labels) # score the growing labeled set
        labels_per_class = len(labeled) / num_classes # budget as average labels per class (the paper's "B")
        labels_per_class_by_round.append(labels_per_class)
        coverage = selector.coverage() if hasattr(selector, "coverage") else None # coverage exists only for coverage-family strategies
        msg = f"round {r}: labeled={len(labeled)} ({labels_per_class:.1f}B) test_acc={test_acc:.2f}"
        if coverage is not None:
            msg += f" coverage={coverage:.3f}"
        print(msg, flush=True)
        if use_wandb:
            import wandb
            log = {"budget_per_class": labels_per_class, "num_labeled": len(labeled),
                   "test_accuracy": test_acc, "episode": r}
            if coverage is not None:
                log["coverage"] = coverage
            wandb.log(log)

    # out_base = <embedding family>_<dataset> (old_embeddings_* / new_embeddings_*), so runs never collide
    out_base = f"{emb.parent.name}_{emb.name}_{strategy}_d{delta}_b{budget}"
    report_picks_per_class(picked_classes_by_round, num_classes, out_base, use_wandb=use_wandb) # which classes got picked, when
    if any(len(g) for g in coverage_gains_by_round): # coverage-gain plot only for strategies that produce gains (skip random)
        report_coverage_gain(coverage_gains_by_round, labels_per_class_by_round, out_base, use_wandb=use_wandb)

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
