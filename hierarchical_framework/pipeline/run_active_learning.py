# Strategy-agnostic active-learning runner.
#
# Each round: a Selector picks `budget` points, their labels are revealed, and an
# Evaluator scores the test set from the growing labeled set. Three pluggable pieces,
# set from the command line, decide the experiment:
#
#   --strategy            WHICH points get picked   (probcover / random)
#   --eval                HOW they are scored       (1nn / mlp)
#   --label_granularity   WHAT each pick reveals     (fine subclass / coarse superclass); coarse
#                         trains+scores on superclasses, selection is unchanged either way
#   --oracle              optionally restrict every prediction to the test point's TRUE
#                         superclass (an upper bound; composes with any --eval; fine labeling only)
#
# The loop below never changes -- only those pieces do. This runner makes NO figures;
# it just logs accuracy (+ the regular AL scalars) to wandb. The figure tools in
# analysis/ replay a run separately when you actually want plots.
#
# --embeddings_dir points at embeddings/<dataset>/<technique>/<source>, where source
# is old_deep_al (itai's SimCLR, the ProbCover-paper baseline) or new_solo_learn
# (solo-learn, the current work).

import argparse
import csv
from pathlib import Path

import numpy as np

from selection import SELECTORS
from evaluators import EVALUATORS
from shared.superclass_maps import load_coarse_map
from shared.delta_resolution import resolve_delta
from shared.wandb_logger import RunLogger


def run(embeddings_dir, strategy="probcover", evaluator="1nn", superclass_oracle=False,
        log_coarse=False, label_granularity="fine", label_cost_ratio=5.0, delta="auto_delta",
        budget=100, rounds=10, split="train", seed=0, dataset="cifar100",
        use_wandb=False, wandb_kwargs=None, csv_path=None):

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
    coarse_map = load_coarse_map(embeddings_dir, dataset)                       # --oracle and coarse labeling read this

    # Fine labeling (default): the evaluator sees the true fine labels of the picked points.
    # Coarse labeling: reveal only each picked point's SUPERCLASS, so the evaluator both trains and is
    # scored on superclasses. Selection + delta above stay on the fine problem, so probcover picks the
    # SAME points in both regimes -- only the labels handed to the evaluator change.
    if label_granularity == "coarse":
        if coarse_map is None:
            raise SystemExit("--label_granularity coarse needs a coarse map "
                             "(coarse_map.npy beside the embeddings, or a built-in dataset like cifar100)")
        if superclass_oracle:
            raise SystemExit("--oracle is meaningless with --label_granularity coarse (the task is already coarse)")
        eval_train_labels = coarse_map[train_labels]   # fine id -> superclass id for every pool point
        eval_test_labels = coarse_map[test_labels]     # and for every test point we score against
    else:
        eval_train_labels, eval_test_labels = train_labels, test_labels

    scorer = EVALUATORS[evaluator](train_features, eval_train_labels,          # HOW to score a labeled set
                                   test_features, eval_test_labels,
                                   coarse_map=coarse_map, seed=seed)

    print(f"strategy={strategy} eval={evaluator} oracle={superclass_oracle} "
          f"on {train_features.shape[0]} points", flush=True)

    # Coarse labeling scores the 20-way superclass task, not the 100-way fine task, so its accuracy
    # belongs on the coarse metric. Routing it there keeps the test_accuracy panel a fine-task-only
    # comparison (no apples-to-oranges 20-way-vs-100-way overlay) and lands the fine-rolled-up and
    # coarse-direct curves together on test_coarse_accuracy -- the valid same-task cross comparison.
    acc_metric = "test_coarse_accuracy" if label_granularity == "coarse" else "test_accuracy"
    logger = RunLogger(use_wandb=use_wandb, wandb_kwargs=wandb_kwargs, acc_metric=acc_metric)

    # Cost per revealed label: a coarse label costs 1 unit, a fine label costs label_cost_ratio units
    # (a fine annotation is the harder ask). annotation_cost = num_labeled * this, so fine and coarse
    # runs line up on the wandb x-axis at equal SPEND -- the coarse run gets more points for the same
    # cost, which is the whole reason coarse labeling might win the coarse task.
    cost_per_label = label_cost_ratio if label_granularity == "fine" else 1.0
    logger.set_config(resolved_delta=float(delta), cost_per_label=float(cost_per_label))

    labeled_indices = []   # every point labeled so far, in pick order
    csv_rows = []          # one row per round, mirrored to --csv (same numbers as wandb)
    for round_index in range(rounds):

        # Ask the strategy for this round's points; it updates its own internal state.
        labeled_indices.extend(selector.select_round(budget))
        labeled_array = np.array(labeled_indices)

        # Score the growing labeled set.
        if label_granularity == "coarse":
            # The evaluator predicts superclasses, so its plain accuracy IS the coarse accuracy. It is
            # logged under the coarse metric via acc_metric above (never on test_accuracy), so leave the
            # separate coarse slot empty here.
            accuracy = scorer.accuracy(labeled_array)
            coarse_accuracy = None
        elif log_coarse:
            # Fine labeling + --coarse: also report superclass accuracy (did the fine prediction land in
            # the right superclass?) from the same prediction pass.
            accuracy, coarse_accuracy = scorer.fine_and_coarse_accuracy(
                labeled_array, use_superclass_oracle=superclass_oracle)
        else:
            accuracy = scorer.accuracy(labeled_array, use_superclass_oracle=superclass_oracle)
            coarse_accuracy = None

        # Express the budget as average labels per class (the paper's "B") and as total annotation cost.
        labels_per_class = len(labeled_indices) / num_classes
        annotation_cost = len(labeled_indices) * cost_per_label

        # Coverage only exists for the coverage-family strategies (ProbCover); others return None.
        coverage = selector.coverage() if hasattr(selector, "coverage") else None

        logger.log_round(round_index, len(labeled_indices), labels_per_class, accuracy,
                         coverage=coverage, coarse_accuracy=coarse_accuracy,
                         annotation_cost=annotation_cost)

        # Same scalars the logger just sent to wandb, kept as a row for the CSV mirror below.
        csv_rows.append({
            "round": round_index,
            "num_labeled": len(labeled_indices),
            "budget_per_class": labels_per_class,
            "annotation_cost": annotation_cost,
            acc_metric: accuracy,
            "coarse_accuracy": coarse_accuracy,
            "coverage": coverage,
        })

    logger.finish()

    # Mirror the per-round scalars to a CSV so results persist off-wandb. Written once at the
    # end (the loop already streams live to wandb); parent dir is created if missing.
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[csv] wrote {len(csv_rows)} rows -> {csv_path}", flush=True)

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
    parser.add_argument(
        "--coarse", dest="log_coarse", action="store_true",
        help="Also log superclass (coarse) accuracy -- did the prediction land in the right superclass? "
             "Gives you a fine-accuracy curve AND a superclass-accuracy curve in wandb.")
    parser.add_argument(
        "--label_granularity", default="fine", choices=["fine", "coarse"],
        help="What label each picked point reveals: 'fine' (subclass, the default) trains/scores on fine "
             "classes; 'coarse' reveals only the superclass, so the evaluator trains and is scored on "
             "superclasses. Selection stays identical either way. 'coarse' predict-fine is impossible.")
    parser.add_argument(
        "--label_cost_ratio", type=float, default=5.0,
        help="Cost of a fine label relative to a coarse one (coarse = 1). Sets the annotation_cost "
             "x-axis (num_labeled * per-label cost) so fine and coarse runs are compared at equal spend. "
             "Default 5 = CIFAR-100's ~5 subclasses per superclass.")

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
    parser.add_argument(
        "--csv", default=None,
        help="Write per-round scalars to this CSV path (same numbers as wandb). If a directory is "
             "given, the file is named <run_name>_seed<seed>.csv inside it.")

    args = parser.parse_args()

    # Auto run name = the pieces that actually distinguish a run. Budget is already
    # visible in the wandb x-axis (labels-per-class) and delta is usually the default,
    # so we leave both out of the name; both are still saved in the run config below.
    oracle_tag = "_oracle" if args.superclass_oracle else ""
    run_name = args.name or f"{args.strategy}_{args.evaluator}_{args.label_granularity}{oracle_tag}"
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
            "coarse": args.log_coarse,
            "label_granularity": args.label_granularity,
            "label_cost_ratio": args.label_cost_ratio,
            "delta": args.delta,
            "budget": args.budget,
            "rounds": args.rounds,
            "seed": args.seed,
            "embeddings_dir": args.embeddings_dir,
        },
    )

    # Resolve --csv: a directory gets an auto filename from the run name + seed; a full path is
    # used as-is. Distinct run names (they carry the evaluator) keep the two flat runs separate.
    csv_path = args.csv
    if csv_path is not None:
        csv_path = Path(csv_path)
        if csv_path.suffix != ".csv":            # treat a suffix-less arg as a directory
            csv_path = csv_path / f"{run_name}_seed{args.seed}.csv"

    run(args.embeddings_dir, strategy=args.strategy, evaluator=args.evaluator,
        superclass_oracle=args.superclass_oracle, log_coarse=args.log_coarse,
        label_granularity=args.label_granularity, label_cost_ratio=args.label_cost_ratio,
        delta=args.delta, budget=args.budget, rounds=args.rounds, split=args.split, seed=args.seed,
        dataset=args.dataset, use_wandb=args.wandb, wandb_kwargs=wandb_kwargs, csv_path=csv_path)


if __name__ == "__main__":
    main()
