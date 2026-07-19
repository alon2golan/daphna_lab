import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

# Render figures straight to file; no GUI/display needed
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None                 # only needed for --wandb runs; absent in .venv_clean

# --- Import Path Setup ---

# pipeline/ is this file's own directory; put it on the path so the sibling packages import
PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from selection import SELECTORS
from evaluators import EVALUATORS
from shared.superclass_maps import load_coarse_map
from shared.delta_resolution import resolve_delta
from shared.run_naming import figure_output_base, wandb_run_tag


# --- Configuration ---

HF_ROOT = PIPELINE_DIR.parent
FIGURE_DIR = HF_ROOT / "outputs" / "figures" / "hierarchical_comparison"

# Every evaluator we score with: display label, base-classifier registry name, whether to turn
# the superclass oracle on, its family (nn/mlp), its role, and a colour_key. A method and its own
# MATCHED ceiling (the same method handed the TRUE superclass via use_superclass_oracle=True) share
# a colour_key so they plot in the same colour -- the method solid, its ceiling dashed. Roles:
#   floor        = flat classifier, ignores the hierarchy (family lower bound).
#   method       = uses the PREDICTED superclass (what we are actually testing).
#   ceiling      = that same method handed the TRUE superclass (its own matched upper bound).
#   flat_ceiling = the flat classifier handed the TRUE superclass (the whole family's oracle).
EVALUATION_SPECS = [
    # label                  eval name        oracle  family  role            colour_key
    ("flat-1nn",            "1nn",           False,  "nn",   "floor",        "flat-nn"),
    ("oracle-1nn",          "1nn",           True,   "nn",   "flat_ceiling", "flat-nn"),
    ("hknn",                "hknn",          False,  "nn",   "method",       "hknn"),
    ("hknn ceiling",        "hknn",          True,   "nn",   "ceiling",      "hknn"),
    ("hknn-marginal",       "hknn_marginal", False,  "nn",   "method",       "hknn-marg"),
    ("hknn-marg ceiling",   "hknn_marginal", True,   "nn",   "ceiling",      "hknn-marg"),
    ("flat-mlp",            "mlp",           False,  "mlp",  "floor",        "flat-mlp"),
    ("oracle-mlp",          "mlp",           True,   "mlp",  "flat_ceiling", "flat-mlp"),
    ("hmlp-routed",         "hmlp_routed",   False,  "mlp",  "method",       "routed"),
    ("hmlp-routed ceiling", "hmlp_routed",   True,   "mlp",  "ceiling",      "routed"),
    ("hmlp-marginal",       "hmlp_marginal", False,  "mlp",  "method",       "marg"),
    ("hmlp-marg ceiling",   "hmlp_marginal", True,   "mlp",  "ceiling",      "marg"),
    ("hmlp-logits",         "hmlp_logits",   False,  "mlp",  "method",       "logits"),
    ("hmlp-logits ceiling", "hmlp_logits",   True,   "mlp",  "ceiling",      "logits"),
]

# Strategies that grow over rounds are drawn as curves; "all" is full supervision (one point)
CURVE_STRATEGIES = ["probcover", "random"]
REFERENCE_STRATEGY = "all"

# Role -> line style for the growing-budget curves
ROLE_LINESTYLE = {"floor": ":", "method": "-", "ceiling": "--", "flat_ceiling": "-."}


# --- Selection Replay + Scoring ---

def replay_labeled_sets(strategy, train_features, delta, seed, budget, rounds):
    """Replays a strategy and returns the cumulative labeled-index array after each round."""
    selector = SELECTORS[strategy](train_features, delta=delta, seed=seed)

    # "all" reveals the whole pool in one round -> exactly one (full) set; the others grow over rounds
    number_of_rounds = 1 if strategy == REFERENCE_STRATEGY else rounds

    labeled_indices = []
    labeled_sets = []
    for _ in range(number_of_rounds):
        # Add this round's picks and snapshot the labeled set so far
        labeled_indices.extend(selector.select_round(budget))
        labeled_sets.append(np.array(labeled_indices))
    return labeled_sets


def score_strategy(strategy, labeled_sets, scorers, num_classes, log_live=False):
    """Scores every evaluator on each round's labeled set; returns a per-round list of accuracies.

    With log_live, each round's fine accuracies are streamed to wandb the moment they are
    computed, so the run's curves appear and update online while the job is still running."""
    per_round = []
    for labeled_array in labeled_sets:
        # One record per budget: the size, the labels-per-class x-axis, and an accuracy dict to fill
        row = {
            "num_labeled": int(len(labeled_array)),
            "labels_per_class": len(labeled_array) / num_classes,   # the paper's "B", the x-axis
            "accuracy": {},
        }

        # Score every evaluator on this same labeled set
        for label, eval_name, use_oracle, _family, _role, _colour in EVALUATION_SPECS:
            fine_accuracy, coarse_accuracy = scorers[eval_name].fine_and_coarse_accuracy(
                labeled_array, use_superclass_oracle=use_oracle)
            row["accuracy"][label] = (fine_accuracy, coarse_accuracy)
        per_round.append(row)

        # Echo the fine accuracies for this budget to the console/slurm log
        print(f"  [{strategy}] labeled={row['num_labeled']} "
              + " ".join(f"{label}={row['accuracy'][label][0]:.1f}" for label, *_ in EVALUATION_SPECS),
              flush=True)

        # Stream this round to wandb right away so the chart fills in live, one point per round.
        # Each metric is namespaced by strategy and plotted against that strategy's own budget step.
        if log_live:
            metrics = {f"{strategy}/budget_per_class": row["labels_per_class"]}
            for label, *_ in EVALUATION_SPECS:
                metrics[f"{strategy}/{label}"] = row["accuracy"][label][0]
            wandb.log(metrics)
    return per_round


# --- Output: CSV + Figure ---

def write_csv(results, output_base):
    """Writes one tidy row per (strategy, evaluator, budget) so the results can be sliced any way later."""
    csv_path = FIGURE_DIR / output_base / "hierarchical_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as handle:
        handle.write("strategy,evaluator,family,role,oracle,num_labeled,budget_per_class,fine_acc,coarse_acc\n")
        # Flatten the nested results into rows
        for strategy, per_round in results.items():
            for row in per_round:
                for label, eval_name, use_oracle, family, role, _colour in EVALUATION_SPECS:
                    fine_accuracy, coarse_accuracy = row["accuracy"][label]
                    handle.write(f"{strategy},{label},{family},{role},{int(use_oracle)},"
                                 f"{row['num_labeled']},{row['labels_per_class']:.3f},"
                                 f"{fine_accuracy:.4f},{coarse_accuracy:.4f}\n")
    print(f"saved -> {csv_path}", flush=True)


def build_figure(results):
    """Builds the family x strategy grid (each method vs its own matched ceiling); returns the Figure."""
    families = ["nn", "mlp"]

    # One colour per colour_key, assigned in first-seen order. A method and its matched ceiling share
    # a colour_key, so they draw in the same colour (method solid, ceiling dashed).
    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colour_keys = list(dict.fromkeys(spec[5] for spec in EVALUATION_SPECS))
    key_colour = {key: colour_cycle[i % len(colour_cycle)] for i, key in enumerate(colour_keys)}

    # The full-supervision accuracies, drawn as a faint horizontal reference (methods only, to limit clutter)
    reference_round = results[REFERENCE_STRATEGY][0]["accuracy"] if REFERENCE_STRATEGY in results else {}

    # One row per family, one column per growing-budget strategy
    fig, axes = plt.subplots(len(families), len(CURVE_STRATEGIES), figsize=(14, 10), sharex=True, sharey=True)
    for row_index, family in enumerate(families):
        # The evaluators belonging to this family (spec[3] is the family field)
        family_specs = [spec for spec in EVALUATION_SPECS if spec[3] == family]
        for column_index, strategy in enumerate(CURVE_STRATEGIES):
            ax = axes[row_index][column_index]
            per_round = results[strategy]
            budgets = [entry["labels_per_class"] for entry in per_round]

            # One curve per evaluator: colour by colour_key, line style by role
            # (floor dotted, method solid, own ceiling dashed, flat family ceiling dash-dot)
            for label, eval_name, use_oracle, _family, role, colour_key in family_specs:
                fine_curve = [entry["accuracy"][label][0] for entry in per_round]
                ax.plot(budgets, fine_curve, ROLE_LINESTYLE[role], marker="o", markersize=3,
                        color=key_colour[colour_key], label=label)

                # For methods only: a faint same-colour horizontal line = its full-supervision accuracy
                if role == "method" and label in reference_round:
                    ax.axhline(reference_round[label][0], color=key_colour[colour_key],
                               linestyle="-", linewidth=0.8, alpha=0.25)

            # Label the outer edges of the grid only
            if row_index == 0:
                ax.set_title(f"strategy = {strategy}")
            if column_index == 0:
                ax.set_ylabel(f"{family} family\ntest fine accuracy (%)")
            if row_index == len(families) - 1:
                ax.set_xlabel("budget (labels per class, B)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Hierarchical evaluators vs selection strategy -- each method (solid) against its OWN "
                 "matched ceiling (dashed, same colour)\n"
                 "dotted = flat floor, dash-dot = flat family oracle, faint line = method at full supervision")
    fig.tight_layout()
    return fig


def save_figure_png(figure, output_base):
    """Writes the figure to outputs/figures/... -- only used when NOT logging to wandb."""
    png_path = FIGURE_DIR / output_base / "hierarchical_comparison.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=130)
    print(f"saved -> {png_path}", flush=True)
    return png_path


# --- wandb ---

def init_wandb(wandb_kwargs):
    """Starts the wandb run BEFORE scoring so per-round points stream in live, and declares each
    strategy's own budget as the x-axis for its evaluator curves."""
    wandb.init(**(wandb_kwargs or {}))
    # One budget step-metric per strategy; every evaluator curve for that strategy plots against it
    for strategy in CURVE_STRATEGIES + [REFERENCE_STRATEGY]:
        step = f"{strategy}/budget_per_class"
        wandb.define_metric(step)
        for label, *_ in EVALUATION_SPECS:
            wandb.define_metric(f"{strategy}/{label}", step_metric=step)


def log_wandb_summary(results, figure):
    """End-of-run extras on the ALREADY-OPEN wandb run: the interactive multi-line charts,
    a full results table, and the composite figure. The live per-round scalars are logged
    separately in score_strategy; this just adds the polished summary views."""
    families = ["nn", "mlp"]

    # One interactive multi-line chart per family x growing-budget strategy. Each line is an
    # evaluator's fine-accuracy curve (method, its ceiling, floor, flat oracle) vs budget.
    for strategy in CURVE_STRATEGIES:
        per_round = results[strategy]
        budgets = [entry["labels_per_class"] for entry in per_round]
        for family in families:
            family_specs = [spec for spec in EVALUATION_SPECS if spec[3] == family]
            curves = [[entry["accuracy"][label][0] for entry in per_round] for label, *_ in family_specs]
            keys = [label for label, *_ in family_specs]
            wandb.log({f"{family}_{strategy}_fine_acc": wandb.plot.line_series(
                xs=budgets, ys=curves, keys=keys,
                title=f"{family} family | {strategy}: fine accuracy vs budget",
                xname="budget (labels per class)")})

    # A tidy table of every (strategy, evaluator, budget) row, including the full-supervision "all"
    table = wandb.Table(columns=["strategy", "evaluator", "family", "role", "oracle",
                                 "num_labeled", "budget_per_class", "fine_acc", "coarse_acc"])
    for strategy, per_round in results.items():
        for row in per_round:
            for label, eval_name, use_oracle, family, role, _colour in EVALUATION_SPECS:
                fine_accuracy, coarse_accuracy = row["accuracy"][label]
                table.add_data(strategy, label, family, role, int(use_oracle), row["num_labeled"],
                               round(row["labels_per_class"], 3), fine_accuracy, coarse_accuracy)
    wandb.log({"results_table": table})

    # The composite grid figure as an image, logged straight from the Figure (no disk file)
    wandb.log({"comparison_figure": wandb.Image(figure)})


# --- Orchestration ---

def run(embeddings_dir, delta="auto_delta", budget=100, rounds=6, split="train", seed=0, dataset="cifar100",
        use_wandb=False, wandb_kwargs=None):
    """Loads the embeddings, runs every strategy x evaluator combination, and writes the CSV + figure (+ wandb)."""
    embeddings_path = Path(embeddings_dir)

    # The pool we select from, and the held-out test split we score on
    train_features = np.load(embeddings_path / f"{split}_features_normalized.npy").astype(np.float32)
    train_labels = np.load(embeddings_path / f"{split}_labels.npy")
    test_features = np.load(embeddings_path / "test_features_normalized.npy").astype(np.float32)
    test_labels = np.load(embeddings_path / "test_labels.npy")
    num_classes = int(train_labels.max()) + 1

    # Coarse map is required: every hierarchical method and every oracle needs fine -> superclass
    coarse_map = load_coarse_map(embeddings_dir, dataset)
    if coarse_map is None:
        raise SystemExit(f"no coarse map for dataset={dataset}; add coarse_map.npy beside the embeddings")

    # Resolve --delta exactly as run_active_learning does, so probcover's picks match a real run
    delta = resolve_delta(delta, train_features, num_classes, seed=seed, dataset=dataset)

    # Build each base classifier once and reuse it across strategies and rounds -- they hold no
    # per-round state, they take the labeled indices fresh on every call
    scorers = {
        eval_name: EVALUATORS[eval_name](train_features, train_labels, test_features, test_labels,
                                         coarse_map=coarse_map, seed=seed)
        for _label, eval_name, *_ in EVALUATION_SPECS
    }

    # Open the wandb run BEFORE any scoring so the curves show up online straightaway and update
    # round by round, instead of appearing all at once when the job finishes.
    live_wandb = use_wandb and wandb is not None
    if live_wandb:
        init_wandb(wandb_kwargs)

    # Run each strategy (the two growing ones plus full supervision) and collect its per-round scores
    all_strategies = CURVE_STRATEGIES + [REFERENCE_STRATEGY]
    results = {}
    for strategy in all_strategies:
        print(f"strategy={strategy}", flush=True)
        labeled_sets = replay_labeled_sets(strategy, train_features, delta, seed, budget, rounds)
        results[strategy] = score_strategy(strategy, labeled_sets, scorers, num_classes,
                                           log_live=live_wandb)

    # Build the grid figure once; where it goes depends on --wandb
    output_base = figure_output_base(embeddings_dir, delta)
    figure = build_figure(results)

    # With --wandb the plots + table live in the project only; otherwise fall back to disk (figure + CSV)
    if live_wandb:
        log_wandb_summary(results, figure)   # summary charts/table/figure on the already-open run
        wandb.finish()
    else:
        write_csv(results, output_base)
        save_figure_png(figure, output_base)
    plt.close(figure)
    return results


def main():
    """Parses the command-line knobs and launches one comparison run."""
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

    # wandb: send the plots to the project instead of only writing them to disk
    parser.add_argument("--wandb", action="store_true", help="log the comparison to Weights & Biases")
    parser.add_argument("--project", default="superclass_analysis", help="wandb project name")
    parser.add_argument("--name", default=None, help="wandb run name (defaults to an auto name from the embeddings path)")
    parser.add_argument("--group", default="hierarchical_comparison", help="wandb group")
    args = parser.parse_args()

    # Auto run name from the embeddings path, e.g. cifar100_simclr_new_solo_learn_hierarchical
    run_name = args.name or f"{wandb_run_tag(args.embeddings_dir)}_hierarchical"
    wandb_kwargs = dict(project=args.project, name=run_name, group=args.group,
                        config=dict(embeddings_dir=args.embeddings_dir, dataset=args.dataset, delta=args.delta,
                                    budget=args.budget, rounds=args.rounds, seed=args.seed))

    run(args.embeddings_dir, delta=args.delta, budget=args.budget, rounds=args.rounds,
        split=args.split, seed=args.seed, dataset=args.dataset,
        use_wandb=args.wandb, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    main()
