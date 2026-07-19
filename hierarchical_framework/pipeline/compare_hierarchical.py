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


def score_strategy(strategy, labeled_sets, scorers, num_classes, wandb_base=None, flat_x=None):
    """Scores every evaluator across this strategy's rounds; returns a per-round list of accuracies.

    The wandb x-axis is the TOTAL number of labeled samples (reaches budget*rounds), not per-class.
    With wandb_base set, EACH evaluator gets its OWN wandb run named
    <strategy>_<evaluator>_<base>, streaming live. Besides the plain fine_acc/coarse_acc (all runs
    overlaid on one panel), each point is ALSO logged under namespaced keys -- by_selection/
    <strategy>/... and by_evaluation/<evaluator>/... . wandb makes one panel per distinct key, so a
    "by_selection" section gives a panel per selection method (evaluators overlaid) and a
    "by_evaluation" section gives a panel per evaluation method (selection methods overlaid), each
    showing the individual run lines -- no UI grouping/aggregation needed.

    flat_x: for a single-set reference strategy ('all', full supervision) whose accuracy does not
    change with budget, pass the curve budgets to replicate its one accuracy across, so it draws as
    a flat horizontal line spanning the same x-range as the growing-budget curves."""
    # One record per budget/round; the accuracy dict is filled evaluator by evaluator below
    rows = [{"num_labeled": int(len(labeled_array)),
             "labels_per_class": len(labeled_array) / num_classes,   # the paper's "B" (kept for the CSV)
             "accuracy": {}} for labeled_array in labeled_sets]

    # Evaluator-outer so each evaluator's whole curve streams into its own run, one active run at a time
    for spec in EVALUATION_SPECS:
        label, eval_name, use_oracle, _family, _role, _colour = spec
        eval_slug = label.replace(" ", "_")   # metric-key-safe evaluator name
        if wandb_base is not None:
            init_combo_run(wandb_base, strategy, spec)   # this (strategy, evaluator) run

        # Score this evaluator on each round's labeled set
        for row, labeled_array in zip(rows, labeled_sets):
            fine_accuracy, coarse_accuracy = scorers[eval_name].fine_and_coarse_accuracy(
                labeled_array, use_superclass_oracle=use_oracle)
            row["accuracy"][label] = (fine_accuracy, coarse_accuracy)

        # One console line per evaluator: its fine-accuracy curve across the budgets
        print(f"  [{strategy}/{label}] "
              + " ".join(f"n{row['num_labeled']}={row['accuracy'][label][0]:.1f}" for row in rows),
              flush=True)

        # Stream to wandb: x = total num_labeled. For the flat reference strategy, replicate the one
        # accuracy across flat_x so it renders as a flat line; otherwise one point per round.
        if wandb_base is not None:
            if flat_x is not None:
                fine_accuracy, coarse_accuracy = rows[0]["accuracy"][label]
                points = [(x, fine_accuracy, coarse_accuracy) for x in flat_x]
            else:
                points = [(row["num_labeled"], *row["accuracy"][label]) for row in rows]
            # Same keys across the matching runs so wandb draws each contributing run as its own line
            for x, fine_accuracy, coarse_accuracy in points:
                wandb.log({
                    "num_labeled": x,
                    "fine_acc": fine_accuracy, "coarse_acc": coarse_accuracy,
                    f"by_selection/{strategy}/fine_acc": fine_accuracy,
                    f"by_selection/{strategy}/coarse_acc": coarse_accuracy,
                    f"by_evaluation/{eval_slug}/fine_acc": fine_accuracy,
                    f"by_evaluation/{eval_slug}/coarse_acc": coarse_accuracy,
                })
            wandb.finish()   # close before the next evaluator's run
    return rows


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
    """Builds one panel per strategy with EVERY classification method overlaid, so the methods can be
    compared head-to-head. Methods are bold solid coloured lines; each method's matched superclass
    ceiling, the flat floors, and the full-supervision level are drawn faint for reference."""
    # The methods we are actually comparing -- one bold coloured line each, across both families
    method_specs = [spec for spec in EVALUATION_SPECS if spec[4] == "method"]
    floor_specs = [spec for spec in EVALUATION_SPECS if spec[4] == "floor"]

    # One distinct colour per method, assigned in listing order; a method and its matched ceiling
    # share a colour_key (spec[5]) so the faint ceiling draws in its method's colour
    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_colour = {spec[5]: colour_cycle[i % len(colour_cycle)] for i, spec in enumerate(method_specs)}
    # colour_key -> the matched ceiling's label, so we can look up each method's ceiling curve
    ceiling_label = {spec[5]: spec[0] for spec in EVALUATION_SPECS if spec[4] == "ceiling"}

    # Full-supervision accuracies (the "all" round), drawn as a faint horizontal target per method
    reference_round = results[REFERENCE_STRATEGY][0]["accuracy"] if REFERENCE_STRATEGY in results else {}

    # One panel per growing-budget strategy, side by side, sharing axes
    fig, axes = plt.subplots(1, len(CURVE_STRATEGIES), figsize=(15, 7), sharex=True, sharey=True)
    for column_index, strategy in enumerate(CURVE_STRATEGIES):
        ax = axes[column_index]
        per_round = results[strategy]
        budgets = [entry["num_labeled"] for entry in per_round]   # x-axis = total labeled (reaches budget*rounds)

        # Each method: a bold solid line in its own colour -- this is the head-to-head comparison
        for label, _eval, _oracle, _family, _role, colour_key in method_specs:
            colour = method_colour[colour_key]
            fine_curve = [entry["accuracy"][label][0] for entry in per_round]
            ax.plot(budgets, fine_curve, "-", marker="o", markersize=4, linewidth=2,
                    color=colour, label=label)

            # Faint dashed same-colour line = this method's matched superclass ceiling (reference)
            ceil = ceiling_label.get(colour_key)
            if ceil is not None:
                ceil_curve = [entry["accuracy"][ceil][0] for entry in per_round]
                ax.plot(budgets, ceil_curve, "--", linewidth=1, alpha=0.35, color=colour)

            # Faint horizontal same-colour line = this method at full supervision (its target)
            if label in reference_round:
                ax.axhline(reference_round[label][0], linestyle="-", linewidth=0.8, alpha=0.2, color=colour)

        # The flat floors (1nn / mlp, no hierarchy) as faint grey dotted reference baselines
        for label, _eval, _oracle, _family, _role, _colour in floor_specs:
            floor_curve = [entry["accuracy"][label][0] for entry in per_round]
            ax.plot(budgets, floor_curve, ":", linewidth=1.2, alpha=0.5, color="grey", label=label)

        ax.set_title(f"strategy = {strategy}")
        ax.set_xlabel("number of labeled samples")
        if column_index == 0:
            ax.set_ylabel("test fine accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle("Classification methods head-to-head (solid = method, one colour each)\n"
                 "faint dashed = that method's matched superclass ceiling, faint grey dotted = flat floor, "
                 "faint horizontal = method at full supervision")
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

def init_combo_run(base_kwargs, strategy, spec):
    """Opens the wandb run for ONE (selection strategy x evaluator) pair, named
    <strategy>_<evaluator>_<base> so the selection+evaluation lead the name and the embeddings tag
    trails at the end, and tags strategy/evaluator/family/role/oracle in the config. Pins the total
    labeled count as the x-axis."""
    label, _eval_name, use_oracle, family, role, _colour = spec
    kwargs = dict(base_kwargs or {})
    base_name = kwargs.get("name") or "run"
    kwargs["name"] = f"{strategy}_{label.replace(' ', '_')}_{base_name}"   # selection + evaluation first, tag last
    kwargs["group"] = kwargs.get("group") or "hierarchical_comparison"     # keep the whole sweep together
    kwargs["job_type"] = strategy                                          # quick filter by selection method
    config = dict(kwargs.get("config") or {})
    config.update(strategy=strategy, evaluator=label, family=family, role=role, oracle=int(use_oracle))
    kwargs["config"] = config
    wandb.init(**kwargs)
    # Total labeled count is the x-axis for every metric, including by_selection/* and by_evaluation/*
    wandb.define_metric("num_labeled")
    wandb.define_metric("*", step_metric="num_labeled")


def log_combined_figure(base_kwargs, figure):
    """Opens one small 'summary_<name>' run holding just the cross-strategy grid figure (methods
    head-to-head, probcover and random side by side) -- the one artifact that spans strategies."""
    kwargs = dict(base_kwargs or {})
    kwargs["name"] = f"summary_{(kwargs.get('name') or 'run')}"
    kwargs["group"] = kwargs.get("group") or "hierarchical_comparison"
    wandb.init(**kwargs)
    wandb.log({"comparison_figure": wandb.Image(figure)})
    wandb.finish()


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

    # Run each strategy (the two growing ones plus full supervision) and collect its per-round scores.
    # With --wandb, EACH (strategy, evaluator) pair gets its OWN wandb run named
    # <base>_<strategy>_<evaluator>, streaming fine_acc + coarse_acc live. Group the runs by
    # config.strategy or config.evaluator in the UI to get the two panel views you want.
    live_wandb = use_wandb and wandb is not None
    all_strategies = CURVE_STRATEGIES + [REFERENCE_STRATEGY]
    results = {}
    for strategy in all_strategies:
        print(f"strategy={strategy}", flush=True)
        labeled_sets = replay_labeled_sets(strategy, train_features, delta, seed, budget, rounds)
        # 'all' is full supervision (one labeled set): replicate its accuracy across the curve budgets
        # so it draws as a flat line spanning the same x-range as probcover/random
        flat_x = None
        if strategy == REFERENCE_STRATEGY and CURVE_STRATEGIES[0] in results:
            flat_x = [row["num_labeled"] for row in results[CURVE_STRATEGIES[0]]]
        results[strategy] = score_strategy(strategy, labeled_sets, scorers, num_classes,
                                           wandb_base=wandb_kwargs if live_wandb else None, flat_x=flat_x)

    # Build the cross-strategy grid figure once; where it goes depends on --wandb
    output_base = figure_output_base(embeddings_dir, delta)
    figure = build_figure(results)

    # With --wandb the plots + table live in the project only; otherwise fall back to disk (figure + CSV)
    if live_wandb:
        log_combined_figure(wandb_kwargs, figure)   # one extra '<name>_summary' run for the grid figure
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
