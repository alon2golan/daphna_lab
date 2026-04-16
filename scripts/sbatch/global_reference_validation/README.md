Global-only reference validation package.

Purpose:
- recover paper-faithful global-sigma references before comparing against local sigma
- separate true paper references from feature-linear sanity checks
- avoid mixing the current SGD `from_features` path with paper baselines
- focus on the low-budget paper region:
  - CIFAR-10: `10, 20, ..., 100`
  - CIFAR-100: `100, 200, ..., 1000`

Why the current `from_features` results diverged:
- DCoM / ProbCover main tables are evaluated with `from_images`, not `from_features`
- MaxHerding main benchmark figures are evaluated with `NN1`
- the current repo `from_features` path is a PyTorch linear layer trained with the generic SGD image-training stack, which is not the paper-faithful reference
- the closest stable feature-linear sanity check is `logistic_from_features`

Paper-faithful references in this package:
- ProbCover: `from_images`
- MaxHerding: `NN1`

Feature-linear sanity checks in this package:
- both methods: `logistic_from_features`
- MaxHerding uses normalized features
- ProbCover uses normalized features in this validation package for consistency across evaluators

Additional ProbCover feature-space sanity checks:
- `NN1`
- `logistic_from_features`

Concurrency:
- each array is `0-2%1`, so each script uses at most 1 GPU at a time
- if you run four scripts together, total concurrent usage is at most 4 GPUs

Budget schedule used in this package:
- CIFAR-10: `budget=10`, `max_iter=9`
- CIFAR-100: `budget=100`, `max_iter=9`

Scripts:
- `maxherding_cifar10_paper_nn1_global_seeds012.sh`
- `maxherding_cifar100_paper_nn1_global_seeds012.sh`
- `maxherding_cifar10_logistic_global_seeds012.sh`
- `maxherding_cifar100_logistic_global_seeds012.sh`
- `probcover_cifar10_paper_images_global_seeds012.sh`
- `probcover_cifar100_paper_images_global_seeds012.sh`
- `probcover_cifar10_nn1_global_seeds012.sh`
- `probcover_cifar100_nn1_global_seeds012.sh`
- `probcover_cifar10_logistic_global_seeds012.sh`
- `probcover_cifar100_logistic_global_seeds012.sh`
