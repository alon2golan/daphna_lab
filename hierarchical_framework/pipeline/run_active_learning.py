# Strategy-agnostic active-learning runner. Each round: a Selector picks `budget`
# points, their labels are revealed, and an Evaluator scores the test set from the
# growing labeled set. --strategy sets WHICH points get picked; --eval sets HOW they
# are scored (1nn / mlp / superclass_oracle). The loop below stays the same.
#
# --embeddings_dir points at embeddings/<dataset>/<technique>/<source>, where source
# is old_deep_al (itai's SimCLR, the ProbCover-paper baseline) or new_solo_learn
# (solo-learn, the current work). Output figure/run names mirror that path.

import argparse
from pathlib import Path

import numpy as np

from selection_strategies import SELECTORS
from evaluators import EVALUATORS
from coarse_maps import load_coarse_map
from delta_selection import resolve_delta
from selection_diagnostics import plot_picks_per_class, plot_coverage_gain
from run_logging import RunLogger


def run(embeddings_dir, strategy="probcover", evaluator="1nn", delta="auto_delta",
        budget=100, rounds=10, split="train", seed=0, dataset="cifar100",
        figures=False, use_wandb=False, wandb_kwargs=None):
    emb = Path(embeddings_dir)
    train_feats = np.load(emb / f"{split}_features_normalized.npy").astype(np.float32) # pool features -- selection runs over these
    train_labels = np.load(emb / f"{split}_labels.npy")
    test_feats = np.load(emb / "test_features_normalized.npy").astype(np.float32) # test features -- only used to score
    test_labels = np.load(emb / "test_labels.npy")
    num_classes = int(train_labels.max()) + 1 # lets us report the budget as labels-per-class (the paper's "B")

    # resolve --delta the deep_al way (explicit float | knee_delta | auto_delta table); prints a [delta] banner
    delta = resolve_delta(delta, train_feats, num_classes, seed=seed, dataset=dataset)
    selector = SELECTORS[strategy](train_feats, delta=delta, seed=seed) # kwargs a strategy ignores (delta/seed) are harmless
    coarse_map = load_coarse_map(embeddings_dir, dataset) # only the superclass_oracle evaluator reads this
    scorer = EVALUATORS[evaluator](train_feats, train_labels, test_feats, test_labels,
                                   coarse_map=coarse_map, seed=seed)
    print(f"strategy={strategy} eval={evaluator} on {train_feats.shape[0]} points", flush=True)

    logger = RunLogger(use_wandb=use_wandb, wandb_kwargs=wandb_kwargs)
    logger.set_config(resolved_delta=float(delta)) # record the delta actually used

    labeled = [] # every point labeled so far, across all rounds
    picked_classes_by_round = [] # the class of each pick, grouped by round (labels read only for diagnostics)
    coverage_gains_by_round = [] # per round, the coverage gain of each pick (empty for non-coverage strategies)
    labels_per_class_by_round = [] # per round, the running labels-per-class ("B") -- the diagnostics' x-axis
    gains_seen = 0 # how many of the selector's coverage_gains we've already split off per round
    for r in range(rounds):
        newly_labeled = selector.select_round(budget) # this round's points; the selector updates its own state
        labeled.extend(newly_labeled)
        picked_classes_by_round.append(train_labels[np.array(newly_labeled)] if newly_labeled else [])
        coverage_gains_by_round.append(selector.coverage_gains[gains_seen:]) # just the gains this round appended
        gains_seen = len(selector.coverage_gains)
        accuracy = scorer.accuracy(np.array(labeled)) # HOW we score is set by --eval
        labels_per_class = len(labeled) / num_classes # budget as average labels per class (the paper's "B")
        labels_per_class_by_round.append(labels_per_class)
        coverage = selector.coverage() if hasattr(selector, "coverage") else None # coverage-family strategies only
        logger.log_round(r, len(labeled), labels_per_class, accuracy, coverage=coverage)

    # diagnostics are opt-in (--figures): generate them once per seed, not on every run. local files only.
    if figures:
        ds, technique, family = emb.parts[-3], emb.parts[-2], emb.parts[-1]
        out_base = f"{ds}/{technique}/{family}_d{delta}" # mirrors the embeddings path; strategy/budget are defaults
        plot_picks_per_class(picked_classes_by_round, num_classes, out_base) # which classes got picked, when
        if any(len(g) for g in coverage_gains_by_round): # coverage-gain plot only for strategies that produce gains
            plot_coverage_gain(coverage_gains_by_round, labels_per_class_by_round, out_base)

    logger.finish()
    return labeled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", required=True,
                   help="dir with <split>_features_normalized.npy + labels + test_*")
    p.add_argument("--strategy", default="probcover", choices=list(SELECTORS),
                   help="which point-selection strategy to run")
    p.add_argument("--eval", dest="evaluator", default="1nn", choices=list(EVALUATORS),
                   help="how to score the labeled set: 1nn | mlp | superclass_oracle")
    p.add_argument("--delta", default="auto_delta",
                   help="a float, 'knee_delta' (run the knee algorithm), or 'auto_delta' "
                        "(default: pinned table for this dataset, knee fallback)")
    p.add_argument("--dataset", default="cifar100", help="dataset name for the auto_delta table + coarse map")
    p.add_argument("--budget", type=int, default=100, help="points labeled per round")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=0, help="tie-break seed (random strategy)")
    p.add_argument("--figures", action="store_true",
                   help="save the selection-diagnostic figures (do this once per seed, not every run)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="superclass_analysis")
    p.add_argument("--name", default=None)
    p.add_argument("--group", default=None)
    args = p.parse_args()

    # default run name/group encode the strategy, evaluator, and knobs
    name = args.name or f"{args.strategy}_{args.evaluator}_d{args.delta}_b{args.budget}"
    group = args.group or args.strategy
    wandb_kwargs = dict(project=args.project, name=name, group=group,
                        config={"strategy": args.strategy, "eval": args.evaluator,
                                "delta": args.delta, "budget": args.budget,
                                "rounds": args.rounds, "seed": args.seed,
                                "embeddings_dir": args.embeddings_dir})
    run(args.embeddings_dir, strategy=args.strategy, evaluator=args.evaluator, delta=args.delta,
        budget=args.budget, rounds=args.rounds, split=args.split, seed=args.seed,
        dataset=args.dataset, figures=args.figures, use_wandb=args.wandb, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    main()
