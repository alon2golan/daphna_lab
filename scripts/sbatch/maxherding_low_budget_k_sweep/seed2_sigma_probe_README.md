Seed-2 sigma probe package for low-budget MaxHerding.

Purpose:
- reproduce the large-K local-sigma trend with a new seed (`seed=2`)
- compare `NN1` vs linear-separator evaluation (`from_features`)
- record local-sigma distribution statistics for every local-sigma run

Datasets and budgets:
- CIFAR-10: labels `10, 20, ..., 100` via `budget=10`, `max_iter=9`
- CIFAR-100: labels `200, 400, ..., 3000` via `budget=200`, `max_iter=14`

Eval modes:
- `NN1`: 1-nearest-neighbor on fixed SSL features
- `from_features`: linear separator on fixed SSL features
- `logistic_from_features`: scikit-learn logistic regression on fixed SSL features

Global reference scripts:
- `maxherding_cifar10_seed2_nn1_global_reference.sh`
- `maxherding_cifar100_seed2_nn1_global_reference.sh`
- `maxherding_cifar10_seed2_linear_global_reference.sh`
- `maxherding_cifar100_seed2_linear_global_reference.sh`
- `maxherding_cifar10_seed2_logistic_global_reference.sh`
- `maxherding_cifar100_seed2_logistic_global_reference.sh`

Local-sigma sigma-probe scripts:
- `maxherding_cifar10_seed2_nn1_sigma_probe.sh`
- `maxherding_cifar100_seed2_nn1_sigma_probe.sh`
- `maxherding_cifar10_seed2_linear_sigma_probe.sh`
- `maxherding_cifar100_seed2_linear_sigma_probe.sh`
- `maxherding_cifar10_seed2_logistic_sigma_probe.sh`
- `maxherding_cifar100_seed2_logistic_sigma_probe.sh`

Unified K grid used by all sigma-probe arrays:
- `3, 7, 60, 480, 1440, 2000, 3500, 5000, 10000, 20000, 30000, 35000, 40000`

Outputs:
- Slurm logs go under `outputs/maxherding/sigma_probe_seed2/...`
- Experiment directories under `output/` will now include:
  - `local_sigma_stats.json`
  - `local_sigma_stats.txt`

Logged local-sigma statistics:
- mean
- std
- coefficient of variation
- min / max
- 5%, 25%, 50% (median), 75%, 95% quantiles
- interquartile range
- full range
