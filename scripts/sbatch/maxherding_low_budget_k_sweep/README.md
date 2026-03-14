## MaxHerding Low-Budget K Sweep

This directory contains a clean first-pass experiment suite for low-budget
MaxHerding comparisons.

### Goals

- Re-run one global-sigma reference experiment on CIFAR-10.
- Re-run one global-sigma reference experiment on CIFAR-100.
- Sweep several `local_sigma_k` values for MaxHerding on both datasets.

### Label ranges

- CIFAR-10: budget `10`, `max_iter=9` -> labeled set sizes `10, 20, ..., 100`
- CIFAR-100: budget `200`, `max_iter=14` -> labeled set sizes `200, 400, ..., 3000`

### Scripts

- `maxherding_cifar10_global_reference.sh`
- `maxherding_cifar100_global_reference.sh`
- `maxherding_cifar10_local_sigma_k_sweep.sh`
- `maxherding_cifar100_local_sigma_k_sweep.sh`

### Output layout

- `outputs/maxherding/low_budget_k_sweep/cifar10/global_sigma/reference/`
- `outputs/maxherding/low_budget_k_sweep/cifar10/local_sigma/k_sweep/`
- `outputs/maxherding/low_budget_k_sweep/cifar100/global_sigma/reference/`
- `outputs/maxherding/low_budget_k_sweep/cifar100/local_sigma/k_sweep/`

### Notes

- The reference runs default to `SEED=0`.
- The CIFAR-10 K-sweep script uses array jobs over `K in {3, 7, 15, 30, 60}`
  with `SEED=0` by default.
- The CIFAR-100 K-sweep script uses array jobs over
  `K in {7, 15, 30, 60, 120}` with `SEED=0` by default.
- Override the seed at submission time if needed, for example:
  `sbatch --export=SEED_OVERRIDE=1 <script>`
