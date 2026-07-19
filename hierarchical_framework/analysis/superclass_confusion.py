# What did the 1-NN classifier actually confuse? Builds superclass confusion heatmaps
# and lists the concrete (true -> guessed) mistakes, for TWO predictors side by side:
#
#   ceiling  = 1-NN fit on ALL train labels           -> the embedding's own ceiling
#              (how good the representation is, ignoring active learning)
#   selected = 1-NN fit only on the strategy's picks   -> what the AL run actually achieves
#
# Outputs go to outputs/figures/error_analysis/<dataset>/<technique>/<source>_d<delta>/.
# Reuses the pipeline's selector, 1-NN evaluator, coarse map, and names.

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

# hierarchical_framework/ root (analysis/ -> up one); put pipeline/ on the path for the
# shared selector, evaluator, coarse map + names, delta resolution, and output naming.
HF_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = HF_ROOT / "outputs"
if str(HF_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(HF_ROOT / "pipeline"))

from selection_strategies import SELECTORS
from evaluators import OneNearestNeighbourEvaluator
from coarse_maps import CIFAR100_COARSE, CIFAR100_COARSE_NAMES as SUPERCLASS_NAMES, CIFAR100_FINE_NAMES as FINE_NAMES
from delta_selection import resolve_delta
from output_paths import figure_output_base, wandb_run_tag

NUM_SUPERCLASSES = len(SUPERCLASS_NAMES)     # 20 for CIFAR-100


def coarse_confusion_matrix(fine_predictions, fine_truth, coarse_map):
    # 20x20 count matrix: entry [t, p] = how many points whose TRUE superclass is t were
    # PREDICTED into superclass p (the diagonal is the correct-superclass count).
    true_superclasses = coarse_map[fine_truth]
    predicted_superclasses = coarse_map[fine_predictions]
    matrix = np.zeros((NUM_SUPERCLASSES, NUM_SUPERCLASSES), dtype=int)
    np.add.at(matrix, (true_superclasses, predicted_superclasses), 1)
    return matrix


def draw_heatmap(ax, confusion_matrix, title):
    # Row-normalize to percentages, so every true-superclass row sums to 100.
    row_totals = confusion_matrix.sum(axis=1, keepdims=True)
    row_percentages = 100.0 * confusion_matrix / np.clip(row_totals, 1, None)

    image = ax.imshow(row_percentages, cmap="magma", vmin=0, vmax=100)
    ax.set_xticks(range(NUM_SUPERCLASSES))
    ax.set_yticks(range(NUM_SUPERCLASSES))
    ax.set_xticklabels(SUPERCLASS_NAMES, rotation=90, fontsize=6)
    ax.set_yticklabels(SUPERCLASS_NAMES, fontsize=6)
    ax.set_xlabel("predicted superclass")
    ax.set_ylabel("true superclass")

    # Coarse accuracy = trace / total = fraction that landed in the right superclass.
    coarse_accuracy = 100.0 * np.trace(confusion_matrix) / confusion_matrix.sum()
    ax.set_title(f"{title}\ncoarse acc = {coarse_accuracy:.1f}%  (bright diagonal = good)")
    return image


def write_mistakes_table(fine_predictions, fine_truth, coarse_map, out_csv):
    # One row per wrong test point: what it was, what it was guessed as, and whether the
    # guess at least landed in the right superclass.
    wrong_indices = np.where(fine_predictions != fine_truth)[0]
    with open(out_csv, "w") as handle:
        handle.write("test_idx,true_fine,true_name,pred_fine,pred_name,true_super,pred_super,same_superclass\n")
        for test_index in wrong_indices:
            true_class = int(fine_truth[test_index])
            predicted_class = int(fine_predictions[test_index])
            same_superclass = int(coarse_map[true_class] == coarse_map[predicted_class])
            handle.write(f"{test_index},{true_class},{FINE_NAMES[true_class]},"
                         f"{predicted_class},{FINE_NAMES[predicted_class]},"
                         f"{SUPERCLASS_NAMES[coarse_map[true_class]]},{SUPERCLASS_NAMES[coarse_map[predicted_class]]},"
                         f"{same_superclass}\n")
    return len(wrong_indices)


def write_confused_pairs(fine_predictions, fine_truth, coarse_map, out_csv, top_k=30):
    # Aggregate the mistakes into (true_fine -> pred_fine) pairs, most frequent first.
    is_error = fine_predictions != fine_truth
    pair_counts = {}
    for true_class, predicted_class in zip(fine_truth[is_error], fine_predictions[is_error]):
        key = (int(true_class), int(predicted_class))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    ranked_pairs = sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)

    with open(out_csv, "w") as handle:
        handle.write("true_name,pred_name,count,same_superclass\n")
        for (true_class, predicted_class), count in ranked_pairs:
            same_superclass = int(coarse_map[true_class] == coarse_map[predicted_class])
            handle.write(f"{FINE_NAMES[true_class]},{FINE_NAMES[predicted_class]},{count},{same_superclass}\n")

    # Return the top-k as readable tuples for the console.
    return [
        ((FINE_NAMES[true_class], FINE_NAMES[predicted_class]), count,
         int(coarse_map[true_class] == coarse_map[predicted_class]))
        for (true_class, predicted_class), count in ranked_pairs[:top_k]
    ]


def log_wandb_confusion(predictions_by_name, test_labels, coarse_map, wandb_kwargs, figure_png=None):
    wandb.init(**wandb_kwargs)
    true_superclasses = coarse_map[test_labels]
    for name, fine_predictions in predictions_by_name.items():
        predicted_superclasses = coarse_map[fine_predictions]
        # wandb's interactive confusion matrix: hover a cell to read the count.
        wandb.log({f"coarse_confusion_{name}": wandb.plot.confusion_matrix(
            y_true=true_superclasses.tolist(), preds=predicted_superclasses.tolist(),
            class_names=SUPERCLASS_NAMES)})
    if figure_png is not None:
        wandb.log({"confusion_matrix": wandb.Image(str(figure_png))})
    wandb.finish()


def run(embeddings_dir, strategy="probcover", delta="auto_delta", budget=100, rounds=10,
        split="train", seed=0, dataset="cifar100", top_k=20, use_wandb=False, wandb_kwargs=None):

    embeddings_path = Path(embeddings_dir)
    train_features = np.load(embeddings_path / f"{split}_features_normalized.npy").astype(np.float32)
    train_labels = np.load(embeddings_path / f"{split}_labels.npy")
    test_features = np.load(embeddings_path / "test_features_normalized.npy").astype(np.float32)
    test_labels = np.load(embeddings_path / "test_labels.npy")
    coarse_map = CIFAR100_COARSE
    num_classes = int(train_labels.max()) + 1
    print(f"test set = {len(test_labels)} images, {num_classes} classes", flush=True)

    # Resolve --delta the same way run_active_learning does, so the replayed picks match.
    delta = resolve_delta(delta, train_features, num_classes, seed=seed, dataset=dataset)

    # Replay the AL run to get the exact set of selected points.
    selector = SELECTORS[strategy](train_features, delta=delta, seed=seed)
    selected_indices = []
    for _ in range(rounds):
        selected_indices.extend(selector.select_round(budget))
    selected_indices = np.array(selected_indices)

    # Two 1-NN predictors via the shared evaluator: the embedding ceiling (all labels) vs
    # what the AL selection achieves.
    scorer = OneNearestNeighbourEvaluator(train_features, train_labels, test_features, test_labels, seed=seed)
    ceiling_predictions = scorer.predict(np.arange(len(train_labels)))
    selected_predictions = scorer.predict(selected_indices)
    predictions_by_name = {"ceiling": ceiling_predictions, "selected": selected_predictions}

    # One output folder per run, shared with accuracy_vs_budget.
    output_dir = OUTPUTS / "figures" / "error_analysis" / figure_output_base(embeddings_dir, delta)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Side-by-side coarse confusion heatmaps: ceiling vs selected.
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    draw_heatmap(ax[0], coarse_confusion_matrix(ceiling_predictions, test_labels, coarse_map),
                 f"embedding ceiling (1-NN on all {len(train_labels)} labels)")
    image = draw_heatmap(ax[1], coarse_confusion_matrix(selected_predictions, test_labels, coarse_map),
                         f"selected (1-NN on {len(selected_indices)} selected labels)")
    fig.colorbar(image, ax=ax, fraction=0.02, label="% of true-superclass row")
    figure_png = output_dir / "confusion_matrix.png"
    fig.savefig(figure_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {figure_png}", flush=True)

    # Per predictor: the concrete mistakes + the most confused fine-class pairs.
    for name, fine_predictions in predictions_by_name.items():
        error_count = write_mistakes_table(fine_predictions, test_labels, coarse_map,
                                           output_dir / f"confusion_mistakes_{name}.csv")
        top_pairs = write_confused_pairs(fine_predictions, test_labels, coarse_map,
                                         output_dir / f"confusion_pairs_{name}.csv", top_k)
        fine_accuracy = 100.0 * float((fine_predictions == test_labels).mean())

        print(f"\n[{name}] fine acc={fine_accuracy:.2f}%  errors={error_count}/{len(test_labels)}", flush=True)
        print(f"  saved -> {output_dir / f'confusion_mistakes_{name}.csv'}", flush=True)
        print(f"  saved -> {output_dir / f'confusion_pairs_{name}.csv'}", flush=True)
        print("  top confusions (true -> guessed, count, same-superclass?):", flush=True)
        for (true_name, predicted_name), count, same_superclass in top_pairs[:12]:
            mark = "same-super" if same_superclass else "CROSS-super"
            print(f"    {true_name:>16} -> {predicted_name:<16} {count:4d}  [{mark}]", flush=True)

    if use_wandb and wandb is not None:
        log_wandb_confusion(predictions_by_name, test_labels, coarse_map, wandb_kwargs, figure_png=figure_png)


def main():
    parser = argparse.ArgumentParser(
        description="Superclass confusion heatmaps + confused-pair lists: embedding ceiling vs the AL selection.")

    parser.add_argument("--embeddings_dir", required=True, help="embeddings/<dataset>/<technique>/<source>")
    parser.add_argument("--strategy", default="probcover", choices=list(SELECTORS), help="selection strategy to replay")
    parser.add_argument("--delta", default="auto_delta",
                        help="a float, 'knee_delta', or 'auto_delta' (default: pinned table, knee fallback)")
    parser.add_argument("--dataset", default="cifar100", help="dataset name for the auto_delta table lookup")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", dest="top_k", type=int, default=20, help="how many confused pairs to keep")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", default="applying_priors")
    parser.add_argument("--group", default="confusion")
    args = parser.parse_args()

    run_name = f"{wandb_run_tag(args.embeddings_dir)}_{args.strategy}"
    wandb_kwargs = dict(project=args.project, group=args.group, name=run_name,
                        config=dict(embeddings_dir=args.embeddings_dir, strategy=args.strategy,
                                    delta=args.delta, budget=args.budget, rounds=args.rounds))
    run(args.embeddings_dir, strategy=args.strategy, delta=args.delta, budget=args.budget,
        rounds=args.rounds, split=args.split, seed=args.seed, dataset=args.dataset,
        top_k=args.top_k, use_wandb=args.wandb, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    main()
