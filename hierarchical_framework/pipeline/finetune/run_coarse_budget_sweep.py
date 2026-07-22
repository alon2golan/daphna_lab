# Sweeps the coarse-label budget and produces TWO comparable curves of 20-way superclass
# accuracy vs how many training points we treat as (coarse-)labeled:
#
#   curve 1 "frozen"    -- train a 20-way MLP on the FROZEN SimCLR embeddings of the labeled
#                          subset, test on the test embeddings.
#   curve 2 "finetuned" -- reload a fresh backbone, finetune it with the COARSE HEAD ONLY (no
#                          fine labels used) on the SAME labeled subset, re-extract embeddings,
#                          train the SAME 20-way MLP on them, test the same way.
#
# The two curves share the exact same labeled subset at each budget, so the only difference is
# whether the embedding moved -- the comparison is honest. Selection is random by default but
# pluggable. Writes a CSV always; streams to Weights & Biases with --wandb.

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

# Render straight to file; no GUI/display on the slurm node
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None                 # only needed for --wandb runs; absent in some envs

# pipeline/ on the path so the shared coarse map imports; this file lives in pipeline/finetune/
FINETUNE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = FINETUNE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from shared.superclass_maps import CIFAR100_COARSE
from solo_backbone import load_backbone, infer_feature_dim, build_datasets, extract_features
from labeled_budgets import LabeledSet, random_selector
from two_head_finetune import (
    finetune_two_heads,
    FINETUNE_EPOCHS, LR_BACKBONE, LR_HEADS, WEIGHT_DECAY,
    COARSE_LOSS_WEIGHT, FINE_LOSS_WEIGHT, FREEZE_BACKBONE, TRAIN_BATCH_SIZE,
)
from probes import coarse_mlp_accuracy, MLP_HIDDEN_LAYER_SIZES, MLP_MAX_ITERATIONS


# --- Path + Sweep Defaults ---

HF_ROOT = PIPELINE_DIR.parent                 # hierarchical_framework/
DAPHNA_ROOT = HF_ROOT.parent                  # daphna_lab/
DEFAULT_CHECKPOINT = DAPHNA_ROOT / "solo-learn/trained_models/simclr/cifar100/1000ep"
DEFAULT_DATA_DIR = DAPHNA_ROOT / "data"       # holds cifar-100-python
RESULTS_DIR = HF_ROOT / "outputs" / "finetune"

# The x-axis: total training points treated as coarse-labeled. Fine 100s, then 1000-step jumps
# through 10k, then the big coarse budgets.
DEFAULT_BUDGETS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                   2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 25000, 50000]

# The pluggable selection strategies (extend this dict to add coreset / typiclust / margin).
SELECTORS = {"random": random_selector}

# The wandb group -- all runs of this experiment land here. Frozen and finetuned are logged as
# SEPARATE runs (config.curve) so the sidebar shows them apart and each is its own line.
DEFAULT_WANDB_GROUP = "superclass_frozen_vs_finetuned"


# --- Labeled subset for one budget ---

def labeled_set_for_budget(train_fine, budget, selector, seed):
    """Selects `budget` points at random from the WHOLE training pool -- so a budget of 100 is
    ~5 per superclass ON AVERAGE, not exactly 5 each. Only the coarse head is trained, so the
    points are tagged coarse-only (fine_indices stays empty -> no fine labels in the finetune)."""
    rng = np.random.default_rng(seed)                              # seeded, reproducible draw
    pool = np.arange(len(train_fine))                              # every training row is selectable
    indices = np.sort(np.asarray(selector(pool, budget, rng), dtype=np.int64))  # the labeled points
    return LabeledSet(coarse_indices=indices)                     # coarse-only; fine head gets no data


# --- One budget: both curves ---

def scores_for_budget(budget, checkpoint_dir, dataset, data_dir, coarse_map, selector, seed,
                      frozen_train, frozen_test, train_fine, test_fine, test_coarse,
                      train_aug, train_eval, test_eval, device, multi_gpu, epochs, lr_backbone):
    """Computes (frozen_acc, finetuned_acc, num_labeled) for a single budget."""
    labeled = labeled_set_for_budget(train_fine, budget, selector, seed)
    indices = labeled.coarse_indices                                # the labeled points (coarse-only)
    labeled_coarse = coarse_map[train_fine[indices]]                # their superclass labels

    # Curve 1: 20-way MLP on the FROZEN embeddings of exactly these points
    frozen_acc = coarse_mlp_accuracy(frozen_train[indices], labeled_coarse,
                                     frozen_test, test_coarse, seed=seed)

    # Curve 2: fresh backbone -> finetune on these points -> re-extract -> same 20-way MLP
    backbone = load_backbone(checkpoint_dir, device, multi_gpu=multi_gpu)  # independent per budget
    feature_dim = infer_feature_dim(backbone, train_eval, device)
    num_fine = int(coarse_map.shape[0])                             # 100
    num_coarse = int(coarse_map.max()) + 1                          # 20
    backbone = finetune_two_heads(backbone, feature_dim, train_aug, labeled,
                                  coarse_map, num_fine, num_coarse, device, seed,
                                  epochs=epochs, lr_backbone=lr_backbone)
    tuned_train, _ = extract_features(backbone, train_eval, device)
    tuned_test, _ = extract_features(backbone, test_eval, device)
    finetuned_acc = coarse_mlp_accuracy(tuned_train[indices], labeled_coarse,
                                        tuned_test, test_coarse, seed=seed)

    return frozen_acc, finetuned_acc, len(indices)


# --- Output: CSV ---

CSV_HEADER = ("selector,seed,num_labeled,avg_per_super,coarse_weight,fine_weight,lr_backbone,"
              "finetune_epochs,freeze_backbone,frozen_acc,finetuned_acc,gain")


def append_csv(row, csv_path):
    """Appends one result row (one per budget) to this run's OWN csv (one writer per file, so
    parallel shards never race). Writes the header when the file is first created."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()                            # header only the first time
    with open(csv_path, "a") as handle:
        if write_header:
            handle.write(CSV_HEADER + "\n")
        handle.write(",".join(str(v) for v in row) + "\n")
    return csv_path


def save_figure(results, selector, seed, tag=""):
    """Plots the two curves (frozen vs finetuned 20-way accuracy) against num_labeled and writes
    a PNG under outputs/finetune/. One figure per (selector, seed, tag) run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    budgets = [entry["num_labeled"] for entry in results]          # x-axis: labeled points
    frozen = [entry["frozen_acc"] for entry in results]            # curve 1
    finetuned = [entry["finetuned_acc"] for entry in results]      # curve 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(budgets, frozen, "o--", color="grey", label="frozen (probe on SimCLR embedding)")
    ax.plot(budgets, finetuned, "o-", color="tab:blue", label="finetuned (coarse-head, then probe)")
    ax.set_xscale("log")                                           # budgets span 100 -> 50000
    ax.set_xlabel("number of coarse-labeled training points")
    ax.set_ylabel("test 20-way superclass accuracy (%)")
    ax.set_title(f"Superclass accuracy vs coarse-label budget\nselector={selector}, seed={seed}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    tag_suffix = f"_{tag}" if tag else ""                          # keep tagged variants' PNGs apart
    png_path = RESULTS_DIR / f"coarse_budget_sweep_{selector}_seed{seed}{tag_suffix}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"saved -> {png_path}", flush=True)
    return png_path


# --- wandb ---

def init_curve_run(project, group, curve, selector, seed, embeddings_tag, num_coarse, shard,
                   epochs, lr_backbone, tag=""):
    """Opens ONE wandb run for a single curve ('frozen' or 'finetuned'). Both curves use the SAME
    metric key 'coarse_acc', and differ only by config.curve + the run name, so the sidebar shows
    them as separate runs and overlaying/grouping-by-curve draws exactly two lines. `tag` labels a
    finetune VARIANT (e.g. ep80_lr3e5) so several finetune settings can coexist as distinct runs.
    Returns the run object so the caller can log to it directly (two runs stay open at once)."""
    tag_suffix = f"_{tag}" if tag else ""                          # distinguishes finetune variants
    shard_suffix = f"_shard{shard}" if shard is not None else ""   # unique per parallel worker
    run = wandb.init(
        project=project,
        group=group,                                               # the shared experiment group
        name=f"{curve}_{selector}_seed{seed}{tag_suffix}{shard_suffix}",  # curve first -> frozen/finetuned split
        job_type=curve,                                            # one-click filter: frozen vs finetuned
        reinit=True,                                               # allow a second run open in this process
        config=dict(
            experiment="coarse_budget_sweep", curve=curve, tag=(tag or "default"),  # group/colour keys
            embeddings_tag=embeddings_tag, selector=selector, seed=seed,
            shard=(shard if shard is not None else "all"), num_superclasses=num_coarse,
            # finetune knobs (only meaningful for the finetuned curve)
            lr_backbone=lr_backbone, lr_heads=LR_HEADS, weight_decay=WEIGHT_DECAY,
            finetune_epochs=epochs, coarse_weight=COARSE_LOSS_WEIGHT,
            fine_weight=FINE_LOSS_WEIGHT, freeze_backbone=int(FREEZE_BACKBONE),
            train_batch_size=TRAIN_BATCH_SIZE,
            # probe knobs (both curves)
            probe_mlp_hidden=str(MLP_HIDDEN_LAYER_SIZES), probe_mlp_iters=MLP_MAX_ITERATIONS,
        ),
    )
    run.define_metric("num_labeled")                               # the x-axis
    run.define_metric("coarse_acc", step_metric="num_labeled")     # y vs num_labeled
    return run


# --- Orchestration ---

def run(checkpoint_dir, data_dir, dataset, budgets, selector_name, seed,
        use_wandb, project, group, multi_gpu, epochs, lr_backbone, tag="", shard=None):
    """Extracts the frozen embeddings once, then sweeps this worker's budgets through both curves.
    With `shard` set (one parallel worker over a slice of budgets), outputs are suffixed by the
    shard so parallel workers never collide; each worker is its own LIVE wandb run in the shared
    group, so overlaying the group's runs shows the combined curves filling in live."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    coarse_map = CIFAR100_COARSE
    num_coarse = int(coarse_map.max()) + 1
    selector = SELECTORS[selector_name]                            # the pluggable selection strategy
    embeddings_tag = Path(checkpoint_dir).name                    # e.g. "1000ep", for wandb config
    tag_suffix = f"_{tag}" if tag else ""                         # labels the finetune variant
    shard_suffix = f"_shard{shard}" if shard is not None else ""  # names outputs per parallel worker
    csv_path = RESULTS_DIR / f"coarse_budget_sweep_{selector_name}_seed{seed}{tag_suffix}{shard_suffix}.csv"

    # Use every visible GPU (DataParallel) when asked and more than one is present
    use_multi = multi_gpu and torch.cuda.device_count() > 1
    print(f"device={device}, visible GPUs={torch.cuda.device_count()}, data_parallel={use_multi}",
          flush=True)

    # Load the pristine backbone and build the datasets ONCE
    backbone = load_backbone(checkpoint_dir, device, multi_gpu=use_multi)
    train_aug, train_eval, test_eval = build_datasets(dataset, data_dir)

    # Extract the FROZEN embeddings ONCE (before any finetuning); curve 1 reuses them at every budget
    frozen_train, train_fine = extract_features(backbone, train_eval, device)
    frozen_test, test_fine = extract_features(backbone, test_eval, device)
    test_coarse = coarse_map[test_fine]                           # test superclass labels, fixed
    print(f"frozen embeddings ready on {device}: train {frozen_train.shape}, test {frozen_test.shape}",
          flush=True)

    # Frozen and finetuned are SEPARATE wandb runs (same key 'coarse_acc'); both stay open so each
    # budget streams live into its own run. Group/colour by config.curve to get the two lines.
    live_wandb = use_wandb and wandb is not None
    frozen_run = finetuned_run = None
    if live_wandb:
        frozen_run = init_curve_run(project, group, "frozen", selector_name, seed,
                                    embeddings_tag, num_coarse, shard, epochs, lr_backbone, tag=tag)
        finetuned_run = init_curve_run(project, group, "finetuned", selector_name, seed,
                                       embeddings_tag, num_coarse, shard, epochs, lr_backbone, tag=tag)

    # Sweep budgets; each budget reloads a fresh backbone for curve 2 inside scores_for_budget
    results = []                                                    # collected for the end-of-run plot
    for budget in budgets:
        frozen_acc, finetuned_acc, num_labeled = scores_for_budget(
            budget, checkpoint_dir, dataset, data_dir, coarse_map, selector, seed,
            frozen_train, frozen_test, train_fine, test_fine, test_coarse,
            train_aug, train_eval, test_eval, device, use_multi, epochs, lr_backbone)
        gain = finetuned_acc - frozen_acc
        print(f"[budget={num_labeled}] frozen={frozen_acc:.2f} finetuned={finetuned_acc:.2f} "
              f"gain={gain:+.2f}", flush=True)
        results.append({"num_labeled": num_labeled, "frozen_acc": frozen_acc,
                        "finetuned_acc": finetuned_acc})

        avg_per_super = num_labeled // num_coarse                  # ~points per superclass on average
        append_csv([selector_name, seed, num_labeled, avg_per_super,      # this worker's own csv
                    COARSE_LOSS_WEIGHT, FINE_LOSS_WEIGHT, lr_backbone,
                    epochs, int(FREEZE_BACKBONE),
                    f"{frozen_acc:.4f}", f"{finetuned_acc:.4f}", f"{gain:.4f}"], csv_path)
        if live_wandb:
            # Same metric key in both runs -> two lines when grouped/overlaid by curve
            frozen_run.log({"num_labeled": num_labeled, "coarse_acc": frozen_acc})
            finetuned_run.log({"num_labeled": num_labeled, "coarse_acc": finetuned_acc})

    # A single (unsharded) run plots the combined PNG; parallel shards leave that to the merge step
    if shard is None:
        save_figure(results, selector_name, seed, tag=tag)
    if live_wandb:
        frozen_run.finish()
        finetuned_run.finish()
    print(f"saved -> {csv_path}", flush=True)


def main():
    """Parses the command-line knobs and launches one budget sweep."""
    parser = argparse.ArgumentParser(
        description="Coarse-budget sweep: frozen-probe vs finetuned-embedding 20-way accuracy.")
    parser.add_argument("--checkpoint_dir", default=str(DEFAULT_CHECKPOINT),
                        help="dir with the .ckpt + args.json (default: cifar100 simclr 1000ep -> ep=999)")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR), help="holds cifar-100-python")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
                        help="total labeled points per budget (random draw from the whole pool, "
                             "~budget/20 per superclass on average)")
    parser.add_argument("--selector", default="random", choices=sorted(SELECTORS),
                        help="how the labeled points are chosen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="stream both curves to Weights & Biases")
    parser.add_argument("--project", default="superclass_analysis", help="wandb project name")
    parser.add_argument("--group", default=DEFAULT_WANDB_GROUP, help="wandb group for this experiment")
    parser.add_argument("--no_multi_gpu", action="store_true",
                        help="disable DataParallel even if several GPUs are visible")
    parser.add_argument("--finetune_epochs", type=int, default=FINETUNE_EPOCHS,
                        help="epochs for the coarse-head finetune (curve 2 only)")
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE,
                        help="backbone learning rate for the finetune (curve 2 only)")
    parser.add_argument("--tag", default="",
                        help="label for a finetune VARIANT (e.g. ep80_lr3e5); suffixes the csv/png + "
                             "wandb run names + config.tag so several finetune settings coexist as "
                             "distinct runs you can compare")
    parser.add_argument("--shard", default=None,
                        help="label for one parallel worker over a slice of --budgets; suffixes the "
                             "csv + wandb run names so parallel workers never collide")
    args = parser.parse_args()

    run(args.checkpoint_dir, args.data_dir, args.dataset, args.budgets, args.selector,
        args.seed, args.wandb, args.project, args.group,
        multi_gpu=not args.no_multi_gpu, epochs=args.finetune_epochs,
        lr_backbone=args.lr_backbone, tag=args.tag, shard=args.shard)


if __name__ == "__main__":
    main()
