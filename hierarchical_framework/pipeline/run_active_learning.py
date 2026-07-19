# Strategy-agnostic active-learning runner.
#
# Each round: a Selector picks `budget` points, their labels are revealed, and an
# Evaluator scores the test set from the growing labeled set. Three pluggable pieces,
# set from the command line, decide the experiment:
#
#   --strategy  WHICH points get picked   (probcover / random)
#   --eval      HOW they are scored       (1nn / mlp)
#   --oracle    optionally restrict every prediction to the test point's TRUE
#               superclass (an upper bound; composes with any --eval)
#
# The loop below never changes -- only those pieces do. This runner makes NO figures;
# it just logs accuracy (+ the regular AL scalars) to wandb. The figure tools in
# analysis/ replay a run separately when you actually want plots.
#
# --embeddings_dir points at embeddings/<dataset>/<technique>/<source>, where source
# is old_deep_al (itai's SimCLR, the ProbCover-paper baseline) or new_solo_learn
# (solo-learn, the current work).

import argparse
from pathlib import Path

import numpy as np

from selection_strategies import SELECTORS
from evaluators import EVALUATORS
from coarse_maps import load_coarse_map
from delta_selection import resolve_delta
from run_logging import RunLogger


def run(embeddings_dir, strategy="probcover", evaluator="1nn", superclass_oracle=False,
        delta="auto_delta", budget=100, rounds=10, split="train", seed=0,
        dataset="cifar100", use_wandb=False, wandb_kwargs=None):

    embeddings_path = Path(embeddings_dir)

    # The pool we select from: features and their (only-revealed-once-picked) fine labels.
    train_features = np.load(embeddings_path / f"{split}_features_normalized.npy").astype(np.float32)
    train_labels = np.load(embeddings_path / f"{split}_labels.npy")

    # The held-out test split: used only to score the current labeled set.
    test_features = np.load(embeddings_path / "test_features_normalized.npy").astype(np.float32)
    test_labels = np.load(embeddings_path / "test_labels.npy")

    # Number of fine classes, so we can report the budget as labels-per-class (the paper's "B").
    num_classes = int(train_labels.max()) + 1

    # Resolve --delta the deep_al way (explicit float | knee_delta | auto_delta table).
    # This prints a [delta] banner stating which value was chosen.
    delta = resolve_delta(delta, train_features, num_classes, seed=seed, dataset=dataset)

    # Build the pluggable pieces once, up front.
    selector = SELECTORS[strategy](train_features, delta=delta, seed=seed)      # WHICH points to label
    coarse_map = load_coarse_map(embeddings_dir, dataset)                       # only the --oracle path reads this
    scorer = EVALUATORS[evaluator](train_features, train_labels,               # HOW to score a labeled set
                                   test_features, test_labels,
                                   coarse_map=coarse_map, seed=seed)

    print(f"strategy={strategy} eval={evaluator} oracle={superclass_oracle} "
          f"on {train_features.shape[0]} points", flush=True)

    # All console + wandb logging lives in RunLogger; wandb receives accuracy + the regular scalars only.
    logger = RunLogger(use_wandb=use_wandb, wandb_kwargs=wandb_kwargs)
    logger.set_config(resolved_delta=float(delta))

    labeled_indices = []   # every point labeled so far, in pick order
    for round_index in range(rounds):

        # Ask the strategy for this round's points; it updates its own internal state.
        labeled_indices.extend(selector.select_round(budget))

        # Score the growing labeled set with the chosen evaluator (and the optional superclass oracle).
        accuracy = scorer.accuracy(np.array(labeled_indices), use_superclass_oracle=superclass_oracle)

        # Express the budget as average labels per class (the paper's "B") -- the wandb x-axis.
        labels_per_class = len(labeled_indices) / num_classes

        # Coverage only exists for the coverage-family strategies (ProbCover); others return None.
        coverage = selector.coverage() if hasattr(selector, "coverage") else None

        logger.log_round(round_index, len(labeled_indices), labels_per_class, accuracy, coverage=coverage)

    logger.finish()
    return labeled_indices


def main():
    parser = argparse.ArgumentParser(
        description="Run one active-learning experiment: pick points, reveal labels, score with an evaluator.")

    # --- what to run on ---
    parser.add_argument(
        "--embeddings_dir", required=True,
        help="Path to embeddings/<dataset>/<technique>/<source>; must contain "
             "<split>_features_normalized.npy, <split>_labels.npy, and test_*.npy.")
    parser.add_argument(
        "--dataset", default="cifar100",
        help="Dataset name. Keys the auto_delta table lookup and the built-in coarse (superclass) map.")

    # --- the pluggable pieces ---
    parser.add_argument(
        "--strategy", default="probcover", choices=list(SELECTORS),
        help="Which points to label each round: probcover (coverage-greedy) or random (baseline).")
    parser.add_argument(
        "--eval", dest="evaluator", default="1nn", choices=list(EVALUATORS),
        help="How to score the labeled set each round: 1nn (nearest neighbour) or mlp (small MLP head).")
    parser.add_argument(
        "--oracle", dest="superclass_oracle", action="store_true",
        help="Restrict every prediction to the test point's TRUE superclass -- an upper bound on what "
             "knowing the coarse label would buy. Composes with any --eval method.")

    # --- ProbCover's coverage radius ---
    parser.add_argument(
        "--delta", default="auto_delta",
        help="Coverage-ball radius. A float uses that value as-is; 'knee_delta' runs deep_al's knee "
             "algorithm; 'auto_delta' (default) reads the pinned table for the dataset and falls back "
             "to the knee if the dataset is unlisted.")

    # --- budget schedule ---
    parser.add_argument("--budget", type=int, default=100, help="Points labeled per round.")
    parser.add_argument("--rounds", type=int, default=6, help="Number of active-learning rounds.")
    parser.add_argument("--split", default="train", help="Which split to draw the pool from (the features-file prefix).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (random-strategy tie-breaks; MLP init).")

    # --- outputs (this runner only logs to wandb; use analysis/ tools for figures) ---
    parser.add_argument("--wandb", action="store_true", help="Log accuracy + the regular scalars to Weights & Biases.")
    parser.add_argument("--project", default="superclass_analysis", help="wandb project name.")
    parser.add_argument("--name", default=None, help="wandb run name (defaults to an auto name built from the knobs).")
    parser.add_argument("--group", default=None, help="wandb group (defaults to the strategy).")

    args = parser.parse_args()

    # Auto run name = the pieces that actually distinguish a run. Budget is already
    # visible in the wandb x-axis (labels-per-class) and delta is usually the default,
    # so we leave both out of the name; both are still saved in the run config below.
    oracle_tag = "_oracle" if args.superclass_oracle else ""
    run_name = args.name or f"{args.strategy}_{args.evaluator}{oracle_tag}"
    if args.delta != "auto_delta":                 # only note delta when it isn't the default
        run_name += f"_d{args.delta}"
    run_group = args.group or args.strategy

    wandb_kwargs = dict(
        project=args.project,
        name=run_name,
        group=run_group,
        config={
            "strategy": args.strategy,
            "eval": args.evaluator,
            "oracle": args.superclass_oracle,
            "delta": args.delta,
            "budget": args.budget,
            "rounds": args.rounds,
            "seed": args.seed,
            "embeddings_dir": args.embeddings_dir,
        },
    )

    run(args.embeddings_dir, strategy=args.strategy, evaluator=args.evaluator,
        superclass_oracle=args.superclass_oracle, delta=args.delta, budget=args.budget,
        rounds=args.rounds, split=args.split, seed=args.seed, dataset=args.dataset,
        use_wandb=args.wandb, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    main()
