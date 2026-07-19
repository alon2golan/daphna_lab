# Compare hierarchical (superclass -> subclass) evaluators across selection strategies.
#
# Grows a labeled set per strategy and, at every budget, scores it with every evaluator; writes
# one figure + a CSV. Each evaluator is a floor (flat, ignores hierarchy), a method (uses the
# PREDICTED superclass -- what we test), or a ceiling (oracle, uses the TRUE superclass for free).
# Strategies: probcover + random grow over rounds (curves); "all" = full supervision (one point,
# drawn as a horizontal reference). Owns its own figure; does not touch the wandb runner.

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")                        # render figures straight to file, no display
import matplotlib.pyplot as plt

# pipeline/ is this file's own directory, so the shared modules import directly.
PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from selectors import SELECTORS
from evaluators import EVALUATORS
from shared.superclass_maps import load_coarse_map
from shared.delta_resolution import resolve_delta
from shared.run_naming import figure_output_base

HF_ROOT = PIPELINE_DIR.parent
FIGURE_DIR = HF_ROOT / "outputs" / "figures" / "hierarchical_comparison"

# Each evaluator we score with: a display label, the registry name of its base classifier,
# whether to turn the superclass oracle on, its family, and its role (floor/method/ceiling).
# oracle rows reuse the flat classifiers with use_superclass_oracle=True -- no separate class.
EVALUATION_SPECS = [
    # label            eval name        oracle  family  role
    ("flat-1nn",      "1nn",           False,  "nn",   "floor"),
    ("hknn",          "hknn",          False,  "nn",   "method"),
    ("hknn-marginal", "hknn_marginal", False,  "nn",   "method"),
    ("oracle-1nn",    "1nn",           True,   "nn",   "ceiling"),
    ("flat-mlp",      "mlp",           False,  "mlp",  "floor"),
    ("hmlp-routed",   "hmlp_routed",   False,  "mlp",  "method"),
    ("hmlp-marginal", "hmlp_marginal", False,  "mlp",  "method"),
    ("hmlp-logits",   "hmlp_logits",   False,  "mlp",  "method"),
    ("oracle-mlp",    "mlp",           True,   "mlp",  "ceiling"),
]

CURVE_STRATEGIES = ["probcover", "random"]   # strategies that grow over rounds -> plotted as curves
REFERENCE_STRATEGY = "all"                   # full supervision -> plotted as a horizontal reference line

# Role -> line style for the growing-budget curves.
ROLE_LINESTYLE = {"floor": ":", "method": "-", "ceiling": "--"}


def replay_labeled_sets(strategy, train_features, delta, seed, budget, rounds):
    # Replay one strategy and return the cumulative labeled-index array after each round.
    # "all" reveals the whole pool in a single round, so it yields exactly one (full) set.
    selector = SELECTORS[strategy](train_features, delta=delta, seed=seed)
    number_of_rounds = 1 if strategy == REFERENCE_STRATEGY else rounds
    labeled_indices = []
    labeled_sets = []
    for _ in range(number_of_rounds):
        labeled_indices.extend(selector.select_round(budget))
        labeled_sets.append(np.array(labeled_indices))
    return labeled_sets


def score_strategy(strategy, labeled_sets, scorers, num_classes):
    # For every labeled set (one per round) score all evaluators, returning a list of per-round
    # dicts: the budget plus, per evaluator label, its (fine, coarse) accuracy.
    per_round = []
    for labeled_array in labeled_sets:
        row = {
            "num_labeled": int(len(labeled_array)),
            "labels_per_class": len(labeled_array) / num_classes,   # the paper's "B", the x-axis
            "accuracy": {},
        }
        for label, eval_name, use_oracle, _family, _role in EVALUATION_SPECS:
            fine_accuracy, coarse_accuracy = scorers[eval_name].fine_and_coarse_accuracy(
                labeled_array, use_superclass_oracle=use_oracle)
            row["accuracy"][label] = (fine_accuracy, coarse_accuracy)
        per_round.append(row)
        print(f"  [{strategy}] labeled={row['num_labeled']} "
              + " ".join(f"{label}={row['accuracy'][label][0]:.1f}" for label, *_ in EVALUATION_SPECS),
              flush=True)
    return per_round


def write_csv(results, output_base):
    # One tidy row per (strategy, evaluator, budget): fine + coarse accuracy and the metadata
    # needed to slice it any way later.
    csv_path = FIGURE_DIR / output_base / "hierarchical_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as handle:
        handle.write("strategy,evaluator,family,role,oracle,num_labeled,budget_per_class,fine_acc,coarse_acc\n")
        for strategy, per_round in results.items():
            for row in per_round:
                for label, eval_name, use_oracle, family, role in EVALUATION_SPECS:
                    fine_accuracy, coarse_accuracy = row["accuracy"][label]
                    handle.write(f"{strategy},{label},{family},{role},{int(use_oracle)},"
                                 f"{row['num_labeled']},{row['labels_per_class']:.3f},"
                                 f"{fine_accuracy:.4f},{coarse_accuracy:.4f}\n")
    print(f"saved -> {csv_path}", flush=True)


def save_figure(results, output_base):
    # Grid of cells: one row per family (nn, mlp), one column per growing-budget strategy. In each
    # cell the family's floor/method/ceiling curves grow with budget, and each evaluator's full-
    # supervision ("all") accuracy is drawn as a faint same-colour horizontal reference line.
    families = ["nn", "mlp"]

    # Fix one colour per evaluator label so it reads the same across every cell.
    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    label_colour = {label: colour_cycle[i % len(colour_cycle)]
                    for i, (label, *_ ) in enumerate(EVALUATION_SPECS)}

    reference_round = results[REFERENCE_STRATEGY][0]["accuracy"] if REFERENCE_STRATEGY in results else {}

    fig, axes = plt.subplots(len(families), len(CURVE_STRATEGIES), figsize=(14, 10), sharex=True, sharey=True)
    for row_index, family in enumerate(families):
        family_specs = [spec for spec in EVALUATION_SPECS if spec[3] == family]
        for column_index, strategy in enumerate(CURVE_STRATEGIES):
            ax = axes[row_index][column_index]
            per_round = results[strategy]
            budgets = [entry["labels_per_class"] for entry in per_round]

            for label, eval_name, use_oracle, _family, role in family_specs:
                fine_curve = [entry["accuracy"][label][0] for entry in per_round]
                ax.plot(budgets, fine_curve, ROLE_LINESTYLE[role], marker="o", markersize=3,
                        color=label_colour[label], label=label)
                # Full-supervision reference for the same evaluator (faint horizontal line).
                if label in reference_round:
                    ax.axhline(reference_round[label][0], color=label_colour[label],
                               linestyle="-", linewidth=0.8, alpha=0.25)

            if row_index == 0:
                ax.set_title(f"strategy = {strategy}")
            if column_index == 0:
                ax.set_ylabel(f"{family} family\ntest fine accuracy (%)")
            if row_index == len(families) - 1:
                ax.set_xlabel("budget (labels per class, B)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Hierarchical evaluators vs selection strategy "
                 "(dotted=floor, solid=method, dashed=ceiling; faint line = full supervision)")
    fig.tight_layout()
    png_path = FIGURE_DIR / output_base / "hierarchical_comparison.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"saved -> {png_path}", flush=True)


def run(embeddings_dir, delta="auto_delta", budget=100, rounds=6, split="train", seed=0, dataset="cifar100"):
    embeddings_path = Path(embeddings_dir)

    # Pool we select from, and the held-out test split we score on.
    train_features = np.load(embeddings_path / f"{split}_features_normalized.npy").astype(np.float32)
    train_labels = np.load(embeddings_path / f"{split}_labels.npy")
    test_features = np.load(embeddings_path / "test_features_normalized.npy").astype(np.float32)
    test_labels = np.load(embeddings_path / "test_labels.npy")
    num_classes = int(train_labels.max()) + 1

    # Coarse map is required: every hierarchical method and every oracle needs fine -> superclass.
    coarse_map = load_coarse_map(embeddings_dir, dataset)
    if coarse_map is None:
        raise SystemExit(f"no coarse map for dataset={dataset}; add coarse_map.npy beside the embeddings")

    # Resolve --delta exactly as run_active_learning does, so probcover picks match a real run.
    delta = resolve_delta(delta, train_features, num_classes, seed=seed, dataset=dataset)

    # Build each base classifier once and reuse it across strategies and rounds (they hold no
    # per-round state -- they take the labeled indices fresh each call).
    scorers = {
        eval_name: EVALUATORS[eval_name](train_features, train_labels, test_features, test_labels,
                                         coarse_map=coarse_map, seed=seed)
        for _label, eval_name, *_ in EVALUATION_SPECS
    }

    all_strategies = CURVE_STRATEGIES + [REFERENCE_STRATEGY]
    results = {}
    for strategy in all_strategies:
        print(f"strategy={strategy}", flush=True)
        labeled_sets = replay_labeled_sets(strategy, train_features, delta, seed, budget, rounds)
        results[strategy] = score_strategy(strategy, labeled_sets, scorers, num_classes)

    output_base = figure_output_base(embeddings_dir, delta)   # cifar100/simclr/<source>_d<delta>
    write_csv(results, output_base)
    save_figure(results, output_base)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare hierarchical superclass->subclass evaluators across selection strategies.")

    parser.add_argument("--embeddings_dir", required=True, help="embeddings/<dataset>/<technique>/<source>")
    parser.add_argument("--dataset", default="cifar100", help="keys the auto_delta table and the built-in coarse map")
    parser.add_argument("--delta", default="auto_delta",
                        help="a float, 'knee_delta', or 'auto_delta' (default: pinned table, knee fallback)")
    parser.add_argument("--budget", type=int, default=100, help="points labeled per round (probcover/random)")
    parser.add_argument("--rounds", type=int, default=6, help="rounds for the growing-budget strategies")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(args.embeddings_dir, delta=args.delta, budget=args.budget, rounds=args.rounds,
        split=args.split, seed=args.seed, dataset=args.dataset)


if __name__ == "__main__":
    main()
