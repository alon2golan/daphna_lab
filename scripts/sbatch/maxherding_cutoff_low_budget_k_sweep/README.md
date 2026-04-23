MaxHerding low-budget cutoff experiments.

Purpose:
- keep MaxHerding's smooth RBF kernel
- zero out entries outside a ProbCover-style distance cutoff
- compare global sigma vs local sigma K sweeps under the validated `NN1` evaluator

Selector details:
- method: `max_herding`
- kernel: `rbf`
- cutoff: `--rbf_cutoff_mode probcover`
- evaluator: `NN1`

Budgets:
- CIFAR-10: `10, 20, ..., 100`
- CIFAR-100: `100, 200, ..., 1000`

Seeds:
- default sweep uses seeds `3, 4, 5`
- global references run all three seeds
- local-sigma sweeps run all `(seed, K)` pairs

Chosen K grids:
- CIFAR-10: `10, 60, 480, 1440, 2000, 3500, 5000, 10000`
- CIFAR-100: `10, 60, 480, 960, 1440, 2000, 3500, 5000`

Rationale:
- CIFAR-10 previously looked best in the `1440-10000` range, so the sweep is denser there
- CIFAR-100 previously looked best in the `480-3500` range, so the sweep is denser there

Concurrency:
- sigma sweeps use `--array=0-6%1`
- global references use one GPU each
- if you submit all four scripts together, the maximum concurrent usage is 4 GPUs total

Outputs:
- slurm logs: `outputs/maxherding/cutoff_seeds345/...`
- experiment dirs: `output/CIFAR10/resnet18/...` and `output/CIFAR100/resnet18/...`

Post-run utilities:
- CSV generator:
  - `scripts/generate_maxherding_cutoff_seeds345_tables.py`
- plot generator:
  - `scripts/plot_maxherding_cutoff_seeds345.py`
