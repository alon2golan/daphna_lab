Low-budget ProbCover K-sweep package for local-sigma probing.

Purpose:
- compare fixed global sigma / fixed delta ProbCover against local-sigma ProbCover
- keep the same low-budget setup used in the latest MaxHerding runs
- run both unnormalized and normalized feature variants
- keep separate unnormalized sweeps for `from_features` and paper-style `from_images`
- keep a parallel normalized-feature linear-eval sweep for comparison

Datasets and budgets:
- CIFAR-10: labels `10, 20, ..., 100` via `budget=10`, `max_iter=9`
- CIFAR-100: labels `200, 400, ..., 3000` via `budget=200`, `max_iter=14`

Eval modes:
- unnormalized feature sweep: `from_features`
- unnormalized image sweep: `from_images`
- normalized sweep: `from_features`

Feature variants:
- unnormalized features: `--no_features_normalized`
- normalized features: default normalized-feature path

Global reference deltas:
- CIFAR-10: `0.75`
- CIFAR-100: `0.65`

Local-sigma K grid:
- `3, 7, 60, 480, 1440, 2000, 3500, 5000, 10000, 20000, 30000, 35000, 40000`

Scripts:
- `probcover_cifar10_seed2_linear_global_reference.sh`
- `probcover_cifar100_seed2_linear_global_reference.sh`
- `probcover_cifar10_seed2_linear_sigma_probe.sh`
- `probcover_cifar100_seed2_linear_sigma_probe.sh`
- `probcover_cifar10_seed2_images_global_reference.sh`
- `probcover_cifar100_seed2_images_global_reference.sh`
- `probcover_cifar10_seed2_images_sigma_probe.sh`
- `probcover_cifar100_seed2_images_sigma_probe.sh`
- `probcover_cifar10_seed2_linear_normalized_global_reference.sh`
- `probcover_cifar100_seed2_linear_normalized_global_reference.sh`
- `probcover_cifar10_seed2_linear_normalized_sigma_probe.sh`
- `probcover_cifar100_seed2_linear_normalized_sigma_probe.sh`

Outputs:
- Slurm logs under `outputs/probcover/low_budget_k_sweep/...`
- experiment directories under `output/` with local-sigma runs also saving:
  - `local_sigma_stats.json`
  - `local_sigma_stats.txt`
