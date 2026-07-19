# Accuracy-structure analysis for one AL run. At every budget it scores the growing
# labeled set three ways and writes the breakdown as curves + CSV tables:
#
#   fine    -- ordinary fine-class (subclass) 1-NN accuracy
#   coarse  -- accuracy after mapping predictions up to the 20 superclasses
#   oracle  -- fine accuracy if the true superclass were given for free (an upper bound)
#
# and splits every fine error into "near miss" (right superclass, wrong subclass) vs
# "far miss" (wrong superclass). It reuses the pipeline's selector and 1-NN evaluator,
# so the numbers line up exactly with run_active_learning.

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")                        # render figures straight to file, no display
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None                             # only used for --wandb runs; absent in .venv_clean

# hierarchical_framework/ root (analysis/ -> up one); put pipeline/ on the path so we
# share the identical selector, evaluator, delta resolution, coarse map, and naming.
HF_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = HF_ROOT / "outputs"
if str(HF_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(HF_ROOT / "pipeline"))

from selection_strategies import SELECTORS
from evaluators import OneNearestNeighbourEvaluator
from coarse_maps import load_coarse_map
from delta_selection import resolve_delta
from output_paths import figure_output_base, wandb_run_tag


def round_metrics(fine_predictions, test_labels, coarse_map, num_classes):
    # Turn one budget's fine predictions into the fine/coarse accuracies and the
    # near/far error split. Returns a dict; oracle accuracy is filled in by the caller.
    predicted_superclasses = coarse_map[fine_predictions]
    true_superclasses = coarse_map[test_labels]

    fine_accuracy = 100.0 * float((fine_predictions == test_labels).mean())
    coarse_accuracy = 100.0 * float((predicted_superclasses == true_superclasses).mean())

    # Of the points the fine classifier got wrong, how many at least landed in the right superclass.
    is_error = fine_predictions != test_labels
    error_count = int(is_error.sum())
    landed_in_right_superclass = predicted_superclasses == true_superclasses
    near_miss_count = int((is_error & landed_in_right_superclass).sum())   # right superclass, wrong subclass
    far_miss_count = error_count - near_miss_count                         # wrong superclass entirely
    near_miss_fraction = (near_miss_count / error_count) if error_count else 0.0
    far_miss_fraction = (far_miss_count / error_count) if error_count else 0.0

    # Per-fine-class recall: accuracy among the test points whose true class is that class.
    per_class_recall = np.full(num_classes, np.nan)
    for fine_class in range(num_classes):
        is_this_class = test_labels == fine_class
        if is_this_class.any():
            per_class_recall[fine_class] = 100.0 * float((fine_predictions[is_this_class] == fine_class).mean())

    return dict(
        fine_acc=fine_accuracy,
        coarse_acc=coarse_accuracy,
        n_err=error_count,
        near_frac=near_miss_fraction,
        far_frac=far_miss_fraction,
        per_class=per_class_recall,
    )


def run_output_dir(output_base):
    # One folder per run under outputs/figures/error_analysis/ (shared with superclass_confusion).
    directory = OUTPUTS / "figures" / "error_analysis" / output_base
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_accuracy_curves(per_round_metrics, output_base):
    x = [row["num_labeled"] for row in per_round_metrics]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Left: fine vs coarse vs oracle accuracy as the budget grows. The (oracle - fine)
    # gap is exactly what perfect coarse knowledge would add.
    ax[0].plot(x, [row["fine_acc"] for row in per_round_metrics], "o-", color="steelblue", label="fine (subclass)")
    ax[0].plot(x, [row["coarse_acc"] for row in per_round_metrics], "s-", color="darkorange", label="coarse (superclass)")
    ax[0].plot(x, [row["oracle_acc"] for row in per_round_metrics], "d--", color="purple", label="fine | superclass known")
    ax[0].set_xlabel("num labeled")
    ax[0].set_ylabel("test accuracy (%)")
    ax[0].set_title("fine vs coarse vs superclass-oracle accuracy")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # Right: the two error buckets, which together are 100% of the mistakes.
    ax[1].plot(x, [100 * row["near_frac"] for row in per_round_metrics], "o-", color="seagreen", label="near (same superclass)")
    ax[1].plot(x, [100 * row["far_frac"] for row in per_round_metrics], "^-", color="firebrick", label="far (other superclass)")
    ax[1].set_xlabel("num labeled")
    ax[1].set_ylabel("share of errors (%)")
    ax[1].set_title("error type per budget")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()

    png = run_output_dir(output_base) / "accuracy_curves.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"saved -> {png}", flush=True)
    return png


def save_tables(per_round_metrics, num_classes, coarse_map, output_base):
    directory = run_output_dir(output_base)

    # Summary: one row per budget with the aggregate metrics.
    summary_csv = directory / "accuracy_summary.csv"
    with open(summary_csv, "w") as handle:
        handle.write("num_labeled,budget_per_class,fine_acc,coarse_acc,oracle_acc,"
                     "n_errors,near_miss_frac,far_miss_frac\n")
        for row in per_round_metrics:
            handle.write(f"{row['num_labeled']},{row['labels_per_class']:.3f},{row['fine_acc']:.4f},"
                         f"{row['coarse_acc']:.4f},{row['oracle_acc']:.4f},"
                         f"{row['n_err']},{row['near_frac']:.4f},{row['far_frac']:.4f}\n")
    print(f"saved -> {summary_csv}", flush=True)

    # Long-format per-class recall table: one (budget, class) row each.
    per_class_csv = directory / "accuracy_per_class.csv"
    with open(per_class_csv, "w") as handle:
        handle.write("num_labeled,class,coarse,recall\n")
        for row in per_round_metrics:
            recall = row["per_class"]
            for fine_class in range(num_classes):
                value = "" if np.isnan(recall[fine_class]) else f"{recall[fine_class]:.4f}"
                handle.write(f"{row['num_labeled']},{fine_class},{int(coarse_map[fine_class])},{value}\n")
    print(f"saved -> {per_class_csv}", flush=True)


def log_wandb_curves(per_round_metrics):
    # Interactive multi-line charts (not images): the accuracy trio and the error-type split.
    x = [row["num_labeled"] for row in per_round_metrics]

    wandb.log({"accuracy_curves": wandb.plot.line_series(
        xs=x,
        ys=[[row["fine_acc"] for row in per_round_metrics],
            [row["coarse_acc"] for row in per_round_metrics],
            [row["oracle_acc"] for row in per_round_metrics]],
        keys=["fine (subclass)", "coarse (superclass)", "fine | superclass known"],
        title="accuracy vs budget", xname="num_labeled")})

    wandb.log({"error_type_curves": wandb.plot.line_series(
        xs=x,
        ys=[[100 * row["near_frac"] for row in per_round_metrics],
            [100 * row["far_frac"] for row in per_round_metrics]],
        keys=["near (same superclass)", "far (other superclass)"],
        title="error type share vs budget", xname="num_labeled")})


def run(embeddings_dir, strategy="probcover", delta="auto_delta", budget=100, rounds=10,
        split="train", seed=0, dataset="cifar100", coarse_map_path=None,
        use_wandb=False, wandb_kwargs=None):

    embeddings_path = Path(embeddings_dir)

    # The pool we select from, and the test split we score on.
    train_features = np.load(embeddings_path / f"{split}_features_normalized.npy").astype(np.float32)
    train_labels = np.load(embeddings_path / f"{split}_labels.npy")
    test_features = np.load(embeddings_path / "test_features_normalized.npy").astype(np.float32)
    test_labels = np.load(embeddings_path / "test_labels.npy")
    num_classes = int(train_labels.max()) + 1

    # Fine -> coarse map: an explicit file wins, otherwise the built-in for the dataset.
    coarse_map = np.load(coarse_map_path) if coarse_map_path else load_coarse_map(embeddings_dir, dataset)
    if coarse_map is None:
        raise SystemExit(f"no coarse map for dataset={dataset}; pass --coarse_map")

    # Resolve --delta exactly as run_active_learning does, so the replayed picks match.
    delta = resolve_delta(delta, train_features, num_classes, seed=seed, dataset=dataset)

    # Same selector + 1-NN evaluator as the runner: identical picks, identical scoring.
    selector = SELECTORS[strategy](train_features, delta=delta, seed=seed)
    scorer = OneNearestNeighbourEvaluator(train_features, train_labels, test_features, test_labels,
                                          coarse_map=coarse_map, seed=seed)
    print(f"strategy={strategy} on {train_features.shape[0]} points", flush=True)

    if use_wandb and wandb is not None:
        wandb.init(**(wandb_kwargs or {}))
        for metric in ("fine_acc", "coarse_acc", "oracle_acc", "near_pct", "far_pct"):
            wandb.define_metric(metric, step_metric="num_labeled")   # every curve shares num_labeled as its x-axis

    labeled_indices = []
    per_round_metrics = []
    for round_index in range(rounds):

        labeled_indices.extend(selector.select_round(budget))   # this round's picks; selector mutates itself
        labeled_array = np.array(labeled_indices)

        # Fine predictions drive fine/coarse/near/far; the oracle predictions give the upper bound.
        fine_predictions = scorer.predict(labeled_array)
        oracle_predictions = scorer.predict_within_true_superclass(labeled_array)

        metrics = round_metrics(fine_predictions, test_labels, coarse_map, num_classes)
        metrics["oracle_acc"] = 100.0 * float((oracle_predictions == test_labels).mean())
        metrics["num_labeled"] = len(labeled_indices)
        metrics["labels_per_class"] = len(labeled_indices) / num_classes      # the paper's "B"
        per_round_metrics.append(metrics)

        print(f"round {round_index}: labeled={metrics['num_labeled']} "
              f"fine={metrics['fine_acc']:.2f} coarse={metrics['coarse_acc']:.2f} "
              f"oracle={metrics['oracle_acc']:.2f} "
              f"near={100 * metrics['near_frac']:.1f}% far={100 * metrics['far_frac']:.1f}%", flush=True)

        if use_wandb and wandb is not None:
            wandb.log({
                "num_labeled": metrics["num_labeled"], "budget_per_class": metrics["labels_per_class"],
                "fine_acc": metrics["fine_acc"], "coarse_acc": metrics["coarse_acc"],
                "oracle_acc": metrics["oracle_acc"],
                "near_pct": 100 * metrics["near_frac"], "far_pct": 100 * metrics["far_frac"],
            })

    output_base = figure_output_base(embeddings_dir, delta)
    save_accuracy_curves(per_round_metrics, output_base)
    save_tables(per_round_metrics, num_classes, coarse_map, output_base)

    if use_wandb and wandb is not None:
        log_wandb_curves(per_round_metrics)
        wandb.finish()

    return per_round_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Fine vs coarse vs superclass-oracle accuracy and the near/far error split, vs budget.")

    parser.add_argument("--embeddings_dir", required=True, help="embeddings/<dataset>/<technique>/<source>")
    parser.add_argument("--strategy", default="probcover", choices=list(SELECTORS), help="selection strategy to replay")
    parser.add_argument("--delta", default="auto_delta",
                        help="a float, 'knee_delta', or 'auto_delta' (default: pinned table, knee fallback)")
    parser.add_argument("--dataset", default="cifar100", help="keys the auto_delta table and the built-in coarse map")
    parser.add_argument("--coarse_map", default=None,
                        help="path to a (num_classes,) fine->coarse .npy (overrides the built-in --dataset map)")
    parser.add_argument("--budget", type=int, default=100, help="points labeled per round")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="log to wandb (off by default)")
    parser.add_argument("--project", default="superclass_analysis")
    parser.add_argument("--name", default=None)
    parser.add_argument("--group", default="error_analysis")
    args = parser.parse_args()

    run_name = args.name or f"{wandb_run_tag(args.embeddings_dir)}_{args.strategy}"
    wandb_kwargs = dict(project=args.project, name=run_name, group=args.group,
                        config=dict(embeddings_dir=args.embeddings_dir, strategy=args.strategy,
                                    delta=args.delta, budget=args.budget, rounds=args.rounds,
                                    split=args.split, seed=args.seed))
    run(args.embeddings_dir, strategy=args.strategy, delta=args.delta, budget=args.budget,
        rounds=args.rounds, split=args.split, seed=args.seed, dataset=args.dataset,
        coarse_map_path=args.coarse_map, use_wandb=args.wandb, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    main()
