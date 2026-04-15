"""
nu_prime_sweeper.py
--------------------
Runs estimate_nu for ONE (dataset, al_method, budget, seed) combination and
writes the results to a per-seed JSON file under results/<dataset>/nu_prime_seeds/.

Each call is reproducible and independent: running with seed=0, seed=1, seed=2
in separate SLURM jobs (or sequentially) gives three files that can be inspected
and averaged with `sweep_nu_prime.py --average`.

Matches the direct CLI call exactly:
    python deep_al/pycls/noise/estimating_delta_nu.py \\
        --dataset cifar100 --model linear --budget 100 --tau 0.5 \\
        --epochs 100 --al_method random --noise_mode random \\
        --representation_model simclr --seed <SEED>

Usage
-----
  # single seed (called by SLURM array job)
  python -m deep_al.pycls.noise.nu_prime_sweeper \\
      --dataset cifar100 --al_method probcover --budget 100 --seed 0

  # dry-run to see what would be written
  python -m deep_al.pycls.noise.nu_prime_sweeper \\
      --dataset cifar100 --al_method random --budget 200 --seed 1 --no_skip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from deep_al.pycls.noise.estimating_delta_nu import estimate_nu, _set_seed

# ── constants ─────────────────────────────────────────────────────────────────

GRID_TAUS  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenet50": 50}

# Subdirectory under results/<dataset>/ that holds per-seed raw files
SEEDS_SUBDIR = "nu_prime_seeds"


# ── per-seed file I/O ─────────────────────────────────────────────────────────

def seed_file_path(dataset: str, al_method: str, budget: int, seed: int) -> Path:
    """Return the Path to the per-seed result file (creates parent dirs)."""
    p = Path(_repo_root) / "results" / dataset.lower() / SEEDS_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{al_method.lower()}_{budget}_seed{seed}.json"


def load_seed_file(dataset: str, al_method: str, budget: int, seed: int) -> dict:
    """Return the taus dict from the seed file, or {} if not found."""
    p = seed_file_path(dataset, al_method, budget, seed)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f).get("taus", {})


def save_seed_file(dataset: str, al_method: str, budget: int, seed: int,
                  taus: dict) -> None:
    """Write per-seed results as JSON."""
    p = seed_file_path(dataset, al_method, budget, seed)
    data = {
        "dataset":   dataset,
        "al_method": al_method,
        "budget":    budget,
        "seed":      seed,
        "taus":      taus,
    }
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(p)  # atomic write


# ── dataset loader ─────────────────────────────────────────────────────────────

def _load_dataset(dataset_name: str):
    from deep_al.pycls.core.config import cfg
    from deep_al.pycls.datasets.data import Data
    cfg_file = _repo_root + f"/deep_al/configs/{dataset_name}/al/RESNET18.yaml"
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.LINEAR_FROM_FEATURES = True
    data_obj = Data(cfg)
    data, _ = data_obj.getDataset(
        save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True
    )
    return data


# ── single estimate_nu call ─────────────────────────────────────────────────────

def _run_estimate_nu(dataset, tau: float, budget: int, dataset_name: str,
                     al_method: str, seed: int, epsilon: float) -> float:
    """One estimate_nu call with runs=1 — identical to the direct CLI."""
    from copy import deepcopy
    num_classes = NUM_CLASSES[dataset_name.lower()]
    _, nu_prime, _ = estimate_nu(
        dataset=deepcopy(dataset),
        tau=tau,
        budget=budget,
        num_classes=num_classes,
        model_name="resnet18",       # parquet lookup key; auto-switches to linear internally
        representation_model="simclr",
        epsilon=epsilon,
        test_size=0.2,
        batch_size=32,
        epochs=100,
        lr=1e-3,
        seed=seed,
        cluster_labels=True,
        cluster_method="kmeans",
        runs=1,                      # matches direct CLI default
        noise_mode="random",
        al_method=al_method,
        dataset_name=dataset_name,
    )
    return nu_prime


# ── grid sweep for one seed ────────────────────────────────────────────────────

def run_single_seed(dataset_name: str, al_method: str, budget: int,
                    seed: int, epsilon: float, skip_existing: bool) -> None:
    """
    Run estimate_nu at every tau in GRID_TAUS for a single seed.
    Results are written to a per-seed JSON file; the shared cache is NOT touched.
    Use sweep_nu_prime.py --average to combine seed files into the cache.
    """
    existing = load_seed_file(dataset_name, al_method, budget, seed)

    if skip_existing and len(existing) >= len(GRID_TAUS):
        print(f"  [skip] seed file already complete: "
              f"{seed_file_path(dataset_name, al_method, budget, seed)}")
        return

    _set_seed(seed)
    dataset = _load_dataset(dataset_name)

    taus_result = dict(existing)  # carry over any partially-written results

    for tau in GRID_TAUS:
        t_key = f"{tau:.4f}"
        if skip_existing and t_key in taus_result:
            print(f"  [skip] tau={tau:.2f} already in seed file")
            continue

        print(f"  [run]  tau={tau:.2f} seed={seed} ...", flush=True)
        nu = _run_estimate_nu(dataset, tau, budget, dataset_name, al_method, seed, epsilon)
        taus_result[t_key] = round(nu, 6)
        # Write after every tau so a killed job doesn't lose progress
        save_seed_file(dataset_name, al_method, budget, seed, taus_result)
        print(f"         -> nu_prime={nu:.4f}", flush=True)

    print(f"  Seed file written: {seed_file_path(dataset_name, al_method, budget, seed)}",
          flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run nu_prime estimate for ONE seed and write to a per-seed file")
    parser.add_argument("--dataset",   default="cifar100",
                        choices=["cifar10", "cifar100", "imagenet50"])
    parser.add_argument("--al_method", default="probcover",
                        choices=["probcover", "maxherding", "random"])
    parser.add_argument("--budget",    type=int, default=100)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--epsilon",   type=float, default=0.01)
    parser.add_argument("--no_skip",   action="store_true",
                        help="Re-run even if the seed file already exists")
    args = parser.parse_args()

    out_path = seed_file_path(args.dataset, args.al_method, args.budget, args.seed)
    print(f"\n{'='*60}")
    print(f"nu_prime sweeper  (single seed)")
    print(f"  dataset   : {args.dataset}")
    print(f"  al_method : {args.al_method}")
    print(f"  budget    : {args.budget}")
    print(f"  seed      : {args.seed}")
    print(f"  epsilon   : {args.epsilon}")
    print(f"  taus      : {GRID_TAUS}")
    print(f"  output    : {out_path}")
    print(f"{'='*60}\n")

    run_single_seed(args.dataset, args.al_method, args.budget,
                    args.seed, args.epsilon, skip_existing=not args.no_skip)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

