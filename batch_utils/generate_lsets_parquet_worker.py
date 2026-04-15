"""
Generate labeled-set index samples for one (al_method, budget, tau) combination
and save to a partial parquet file.  Called per SLURM array job by
sweep_generate_lsets.py.

Schema of each row written:
    budget                int    – number of samples in the lset
    oracle_teacher_ratio  float  – tau (cost ratio) this lset was generated for
    alpha                 int    – 1 = oracle annotator, 0 = noisy annotator
    lsets                 list   – [np.ndarray of selected indices]

For oracle rows:  budget == args.budget
For noisy rows:   budget == int(args.budget / args.tau)

Usage (standalone test):
    python batch_utils/generate_lsets_parquet_worker.py \\
        --cfg deep_al/configs/cifar100/al/RESNET18_noisy.yaml \\
        --al probcover \\
        --budget 100 \\
        --tau 0.3 \\
        --initial_delta 0.75 \\
        --n_runs 3 \\
        --seed 42 \\
        --exp-name gen_parquet_test \\
        --eval_model_type from_features
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Re-use the exact same infrastructure as train_al.py
from deep_al.tools.train_al import (
    argparser as _base_argparser,
    get_annotators,
    prepare_datasets,
)
from deep_al.pycls.core.config import setup_cfg
from deep_al.pycls.noise.annotator_manager import AnnotatorManager
from deep_al.pycls.al.coverage_matrix_methods import CoverageMatrixMethod
from deep_al.pycls.noise.annotator import Annotator


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser():
    """Extend the train_al argparser with worker-specific arguments."""
    parser = _base_argparser()

    # Make --exp-name optional: the sweep auto-generates one, but standalone
    # invocations can omit it.
    for action in parser._actions:
        if "--exp-name" in getattr(action, "option_strings", []):
            action.required = False
            action.default = "generate_parquet"

    parser.add_argument(
        "--tau",
        type=float,
        required=True,
        help="oracle_teacher_ratio stored in the parquet (cost of noisy / cost of oracle)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of independent lset runs per (budget, tau, alpha) cell",
    )
    return parser


# ---------------------------------------------------------------------------
# CoverageMatrixMethod helpers
# ---------------------------------------------------------------------------

def _configure_cmm_kernel(al_method: str, cfg) -> None:
    """Set cfg KERNEL_TYPE / SPARSE_K to match what active_learning.py does."""
    if al_method in ("probcover", "prob_cover"):
        cfg["KERNEL_TYPE"] = "tophat"
        cfg["SPARSE_K"] = True
        cfg["K_SPARSE"] = True   # mirror the key set by active_learning.py
    elif al_method in ("maxherding", "max_herding"):
        cfg["KERNEL_TYPE"] = "rbf"
    else:
        raise ValueError(
            f"Unsupported al_method '{al_method}'. "
            "Expected 'probcover' or 'maxherding'."
        )


def _reset_coverage(cmm: CoverageMatrixMethod) -> None:
    """Zero out C_general so the next select_samples starts from a clean slate."""
    cmm.C_general.fill_(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Config setup (identical to train_al.py)
    # -----------------------------------------------------------------------
    cfg = setup_cfg(args)

    # The worker does not run a full experiment, so we use a temp dir for any
    # annotator state files instead of creating a real experiment directory.
    cfg.EXP_DIR = tempfile.mkdtemp(prefix="gen_parquet_")

    # -----------------------------------------------------------------------
    # Identify output path
    # -----------------------------------------------------------------------
    dataset_lower = cfg.DATASET.NAME.lower()   # "cifar100"
    model_lower   = cfg.MODEL.TYPE.lower()     # "resnet18"
    noise_type    = cfg.ANNOTATORS.NOISE_TYPE  # "sym"
    al_method     = args.al                    # "probcover" / "maxherding"

    tau_str   = f"{args.tau:.2f}".replace(".", "p")   # "0p30"
    partial_dir = os.path.join(
        _repo_root, "output", cfg.DATASET.NAME.upper(), "partial"
    )
    os.makedirs(partial_dir, exist_ok=True)
    partial_path = os.path.join(
        partial_dir,
        f"{dataset_lower}_{model_lower}_{al_method}_{noise_type}"
        f"_tau{tau_str}_budget{args.budget}_seed{args.seed}.parquet",
    )

    if os.path.exists(partial_path):
        print(f"[worker] Partial file already exists, skipping: {partial_path}")
        return

    # -----------------------------------------------------------------------
    # Data loading (no LUV splits needed — just train_data for annotators)
    # -----------------------------------------------------------------------
    _, train_data, _, train_size, _ = prepare_datasets(cfg, args)
    full_uset = np.arange(train_size, dtype=int)

    budget     = args.budget
    noisy_size = int(budget / args.tau)

    if noisy_size >= train_size:
        raise ValueError(
            f"noisy_size={noisy_size} (budget={budget}/tau={args.tau}) "
            f"must be less than train_size={train_size}"
        )

    # -----------------------------------------------------------------------
    # Configure kernel for the chosen AL method
    # -----------------------------------------------------------------------
    _configure_cmm_kernel(al_method, cfg)

    # -----------------------------------------------------------------------
    # Build K_general ONCE (expensive GPU operation) and reuse across runs
    # -----------------------------------------------------------------------
    # We create a throw-away annotator manager here just so __init__ can
    # complete; we swap it out per-run below.
    _initial_annotators = get_annotators(cfg, args, train_data)
    _initial_mgr = AnnotatorManager(
        annotators=_initial_annotators,
        selection_strategy="random",
        seed=args.seed,
        save_dir=cfg.EXP_DIR,
    )
    cmm = CoverageMatrixMethod(cfg, annotator_manager=_initial_mgr)

    # -----------------------------------------------------------------------
    # Generate n_runs independent lsets
    # -----------------------------------------------------------------------
    rows = []

    for run_i in range(args.n_runs):
        run_seed = args.seed + run_i * 100   # spread seeds apart

        # Create fresh annotators with this run's seed so each run has an
        # independent noise realization (relevant for the noisy annotator).
        import types
        run_args = types.SimpleNamespace(**vars(args))
        run_args.seed = run_seed
        run_annotators = get_annotators(cfg, run_args, train_data)

        # Identify oracle annotator (noise_rate == 0) and noisy annotator.
        oracle_id = None
        noisy_id  = None
        for i, ann in enumerate(run_annotators):
            if ann.noise_rate == 0.0:
                oracle_id = i
            else:
                noisy_id = i

        if oracle_id is None or noisy_id is None:
            raise RuntimeError(
                f"Expected exactly 1 oracle (noise=0) and 1 noisy annotator; "
                f"got {[(i, a.noise_rate) for i, a in enumerate(run_annotators)]}"
            )

        run_mgr = AnnotatorManager(
            annotators=run_annotators,
            selection_strategy="random",
            seed=run_seed,
            save_dir=cfg.EXP_DIR,
        )
        # Swap annotators into the shared CMM instance
        cmm.annotator_manager = run_mgr

        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        print(f"\n[run {run_i + 1}/{args.n_runs}] seed={run_seed}, "
              f"al={al_method}, budget={budget}, tau={args.tau}")

        # -- Oracle run ------------------------------------------------------
        _reset_coverage(cmm)
        oracle_set, _ = cmm.select_samples(
            lset=np.array([], dtype=int),
            uset=full_uset.copy(),
            num_samples=budget,
            annotator_id=oracle_id,
        )
        rows.append({
            "budget":               budget,
            "oracle_teacher_ratio": args.tau,
            "alpha":                1,
            "lsets":                [oracle_set],
        })

        # -- Noisy run -------------------------------------------------------
        _reset_coverage(cmm)
        noisy_set, _ = cmm.select_samples(
            lset=np.array([], dtype=int),
            uset=full_uset.copy(),
            num_samples=noisy_size,
            annotator_id=noisy_id,
        )
        rows.append({
            "budget":               noisy_size,
            "oracle_teacher_ratio": args.tau,
            "alpha":                0,
            "lsets":                [noisy_set],
        })

    # -----------------------------------------------------------------------
    # Save partial parquet
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_parquet(partial_path, index=False)
    print(f"\n[worker] Saved {len(df)} rows → {partial_path}")


if __name__ == "__main__":
    main()
