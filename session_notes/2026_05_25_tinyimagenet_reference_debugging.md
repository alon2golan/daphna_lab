# Session Notes: Tiny ImageNet MaxHerding Reference Debugging

## Goal

We started extending the MaxHerding locality and density study from CIFAR10 and
CIFAR100 to Tiny ImageNet.

The purpose is to test whether the same main conclusion appears on a harder
dataset:

- local sigma mainly affects the graph degree
- graph degree may explain the behavior more than locality itself
- density and selected-point structure may be more important than the exact
  local-sigma K value

## Tiny ImageNet Setup

Tiny ImageNet has 200 classes and 100000 training images.

For low-budget active learning we used:

- method: MaxHerding
- classifier: 1NN
- features: fixed Tiny ImageNet SSL features
- budget per round: 200 labels
- labeled-set sizes: 200 to 2000 labels
- seed: 3 for the initial feasibility run

This matches the MaxHerding paper budget convention for Tiny ImageNet: one
batch contains C labels, where C is the number of classes.

## Problem Observed

The first Tiny ImageNet global run produced a monotonic curve, but the accuracy
was much lower than expected:

```text
200 labels:  1.88%
400 labels:  2.93%
600 labels:  4.10%
800 labels:  4.22%
1000 labels: 4.66%
1200 labels: 4.82%
1400 labels: 5.23%
1600 labels: 5.41%
1800 labels: 5.50%
2000 labels: 5.58%
```

The curve shape looked reasonable, but the scale did not match the MaxHerding
paper. The paper reports Tiny ImageNet MaxHerding with 1NN at roughly the
`17%` to `25%` range for the same low-budget regime.

Therefore, the Tiny ImageNet result should not yet be treated as a valid paper
reference.

## What We Checked

The run that gave `1.88%` to `5.58%` was not pure paper-style MaxHerding.

It used:

```bash
--rbf_cutoff_mode probcover
--initial_delta 0.15
--sparse_K
```

This means it was MaxHerding with a ProbCover-style cutoff, not the smooth
global RBF MaxHerding reference from the paper.

We also checked the available Tiny ImageNet feature files using full-train 1NN:

```text
features_seed1 / test_features_seed1: 0.36%
maor_train / maor_test:              0.45%
3112_train_set / 2912_valset:        13.13%
```

The best aligned feature pair currently available is:

```text
3112_train_set_features.npy
2912_valset_features.npy
2912_valset_labels.npy
```

However, even this gives only `13.13%` full-train 1NN accuracy, which is low
relative to the paper numbers. This suggests that feature quality or feature
alignment may still be a major issue.

## Current Hypotheses

The low Tiny ImageNet accuracy may come from one or more of the following:

- the ProbCover-style cutoff is too aggressive for Tiny ImageNet
- the available Tiny ImageNet SSL features are weaker than the paper features
- the exact feature extraction or evaluation protocol differs from the paper
- the sparse RBF implementation is an approximation of the full smooth RBF
  kernel

We do not yet think sparse K is necessarily the main problem. The next most
direct test is to keep the sparse RBF path but remove the ProbCover-style
cutoff.

## Scripts Added

The new scripts are under:

```text
scripts/sbatch/tinyimagenet_paper_reference/
```

First diagnostic run:

```bash
sbatch scripts/sbatch/tinyimagenet_paper_reference/maxherding_tinyimagenet_seed3_nn1_no_cutoff_sparse.sh
```

This run keeps `--sparse_K` but removes `--rbf_cutoff_mode probcover`.

Exact dense paper-style option:

```bash
sbatch scripts/sbatch/tinyimagenet_paper_reference/maxherding_tinyimagenet_seed3_nn1_global_paper.sh
```

This run removes both the ProbCover-style cutoff and sparse K. It requests a
larger GPU group because Tiny ImageNet needs a dense `100000 x 100000` RBF
kernel.

## Next Step

Run the no-cutoff sparse script first.

If the result jumps closer to the paper range, then the ProbCover-style cutoff
was the main issue.

If the result remains very low, then the likely blocker is the Tiny ImageNet
feature representation or some remaining mismatch in the evaluation protocol.

## Update: No-Cutoff Sparse Run Failed

The first no-cutoff sparse run started and then was killed by Slurm:

```text
Detected 1 oom-kill event
```

This was not a path issue.

The reason is that `--sparse_K` only saves memory when the threshold actually
keeps the graph sparse. With global RBF, no ProbCover cutoff, and
`initial_delta=1.0`, the current sparse implementation keeps entries with
kernel value above `exp(-1)`, which is equivalent to keeping distances below
about `1.0`.

On the current normalized Tiny ImageNet features, a sampled estimate showed:

```text
delta=0.15 -> average degree about 1
delta=0.25 -> average degree about 19
delta=0.50 -> average degree about 5834
delta=0.75 -> average degree about 48786
delta=1.00 -> average degree about 93426
```

So with `delta=1.0`, the matrix is about `93%` nonzero. That is not practically
sparse. The sparse construction stores row indices, column indices, and values,
so it can use more RAM than the dense `float16` matrix.

Conclusion: sparse K may not be an algorithmic problem, but for the no-cutoff
Tiny ImageNet reference it is a memory problem in this implementation.

The cleaner next run is the dense paper-style script:

```bash
sbatch scripts/sbatch/tinyimagenet_paper_reference/maxherding_tinyimagenet_seed3_nn1_global_paper.sh
```

## Update: Dense No-Cutoff Run Still Too Low

The dense no-cutoff paper-style run completed and improved over the cutoff run,
but it was still far below the paper scale:

```text
200 labels:  3.52%
400 labels:  4.39%
600 labels:  5.12%
800 labels:  5.68%
1000 labels: 5.86%
1200 labels: 6.13%
1400 labels: 6.33%
1600 labels: 6.66%
1800 labels: 6.69%
2000 labels: 6.82%
```

This showed that the cutoff was not the main issue.

We then rechecked the available Tiny ImageNet feature files with the correct
validation label orders:

```text
features_seed1 / test_features_seed1 with val_annotations labels: 0.44%
features_seed1 / test_features_seed1 with bank labels:            0.36%
maor_train / maor_test with val_annotations labels:               29.94%
maor_train / maor_test with bank labels:                          0.45%
3112_train_set / 2912_valset with bank labels:                    13.13%
3112_train_set / 2912_valset with val_annotations labels:         0.56%
```

The correct feature pair for reproducing the paper scale is therefore likely:

```text
maor_train_features.npy
maor_test_features.npy
```

with the normal `val_annotations.txt` validation label order.

We added a `maor` Tiny ImageNet feature tag and a new reference script:

```bash
sbatch scripts/sbatch/tinyimagenet_paper_reference/maxherding_tinyimagenet_seed3_nn1_global_paper_maor_features.sh
```

This should be the next run.

## Update: Maor Features Improve the Scale

The Maor-feature paper-style run produced:

```text
200 labels:  11.49%
400 labels:  13.34%
600 labels:  15.18%
800 labels:  16.11%
1000 labels: 17.14%
1200 labels: 17.73%
1400 labels: 18.48%
1600 labels: 19.09%
1800 labels: 19.39%
2000 labels: 19.71%
```

This is much closer to the MaxHerding paper than the previous feature pair, but
it is still somewhat lower than the reported curve.

One remaining protocol difference is that our Tiny ImageNet low-budget config
uses:

```yaml
DATASET:
  VAL_RATIO: 0.05
```

This removes 5000 images from the active-learning pool. The run therefore
selects labels from 95000 training images instead of the full 100000-image Tiny
ImageNet train pool.

For `1NN`, this validation split is not needed because there is no neural model
selection. To test this protocol difference, we added:

```text
deep_al/configs/tinyimagenet/al/RESNET18_LOW_BUDGET_NO_VAL.yaml
```

and the sbatch:

```bash
sbatch scripts/sbatch/tinyimagenet_paper_reference/maxherding_tinyimagenet_seed3_nn1_global_paper_maor_features_no_val.sh
```

The no-validation run produced almost the same result:

```text
200 labels:  11.28%
400 labels:  13.81%
600 labels:  15.42%
800 labels:  16.32%
1000 labels: 17.35%
1200 labels: 18.00%
1400 labels: 18.50%
1600 labels: 19.11%
1800 labels: 19.46%
2000 labels: 19.84%
```

So the validation split was not the reason for the remaining gap to the paper.

We checked selected class coverage for the no-validation run:

```text
200 labels:  124 unique classes
400 labels:  173 unique classes
600 labels:  187 unique classes
800 labels:  195 unique classes
1000 labels: 200 unique classes
2000 labels: 200 unique classes
```

This suggests the selection is not failing due to missing many classes.

We also computed a quick random-selection 1NN baseline on the same Maor features
over five random seeds:

```text
200 labels:  7.58% +/- 0.51
400 labels:  9.61% +/- 0.64
600 labels:  10.91% +/- 0.34
800 labels:  12.11% +/- 0.45
1000 labels: 12.97% +/- 0.23
1200 labels: 13.56% +/- 0.11
1400 labels: 14.17% +/- 0.22
1600 labels: 14.72% +/- 0.29
1800 labels: 15.14% +/- 0.31
2000 labels: 15.45% +/- 0.22
```

MaxHerding is therefore clearly above random on Tiny ImageNet with Maor
features. The setting is reliable enough for our purpose: studying whether the
density/locality conclusions from CIFAR10 and CIFAR100 also appear on Tiny
ImageNet.

We created a separate Maor-feature Tiny degree-study directory:

```text
scripts/sbatch/tinyimagenet_maor_degree_study/
```

These scripts should be used for all Tiny ImageNet locality/density experiments
going forward.

## Update: Tiny Maor Cutoff K-Sweep Results

The Maor-feature Tiny ImageNet cutoff experiment completed for global sigma and
the planned local-sigma K sweep.

Common setup:

```text
dataset: Tiny ImageNet
features: Maor SSL features
classifier: 1NN
method: MaxHerding
cutoff: ProbCover-style RBF cutoff
delta: 0.5
seed: 3
budget: 200 labels per episode
labeled sizes: 200 to 2000
```

Final accuracies:

```text
global sigma: 20.29%

K=10:     15.15%
K=60:     15.49%
K=250:    17.25%
K=500:    17.96%
K=1000:   19.51%
K=1500:   19.41%
K=3000:   20.39%
K=5000:   20.45%
K=10000:  20.81%
```

Mean local sigma increased with K:

```text
K=10:     mean sigma 0.688
K=60:     mean sigma 0.754
K=250:    mean sigma 0.816
K=500:    mean sigma 0.849
K=1000:   mean sigma 0.885
K=1500:   mean sigma 0.906
K=3000:   mean sigma 0.946
K=5000:   mean sigma 0.977
K=10000:  mean sigma 1.023
```

This is consistent with our CIFAR10 and CIFAR100 conclusions:

- small K hurts strongly
- larger K becomes more global-like
- the useful region is large K
- the differences near global are modest
- local sigma seems to matter mostly through the effective graph scale/degree,
  not through strong locality by itself

For Tiny ImageNet, the best local K slightly beats global sigma, but the gain is
small:

```text
K=10000: 20.81%
global:  20.29%
difference: +0.52 points
```

Lightweight CSV summaries were written to:

```text
results_summaries/maxherding_tinyimagenet_maor_density_selection/accuracy_sigma_summary.csv
results_summaries/maxherding_tinyimagenet_maor_density_selection/accuracy_by_episode.csv
results_summaries/maxherding_tinyimagenet_maor_density_selection/selection_class_coverage_by_episode.csv
```

Plots were written to:

```text
visual_outputs/tinyimagenet_maor_degree_study/tinyimagenet_maor_accuracy_curves.png
visual_outputs/tinyimagenet_maor_degree_study/tinyimagenet_maor_final_accuracy_by_k.png
visual_outputs/tinyimagenet_maor_degree_study/tinyimagenet_maor_sigma_mean_by_k.png
```

The heavier selected-point degree and dataset-density diagnostics still need to
run on the cluster, because the local machine GPU is too old for the installed
PyTorch build. Run:

```bash
sbatch scripts/sbatch/tinyimagenet_maor_degree_study/maxherding_tinyimagenet_maor_density_selection_seed3.sh
```

## Update: Tiny Maor Density Diagnostics

The cluster diagnostics completed and wrote:

```text
results_summaries/maxherding_tinyimagenet_maor_density_selection/density_summary.csv
results_summaries/maxherding_tinyimagenet_maor_density_selection/selection_episode_summary.csv
results_summaries/maxherding_tinyimagenet_maor_density_selection/selected_points.csv
```

We also created compact derived summaries:

```text
results_summaries/maxherding_tinyimagenet_maor_density_selection/density_selection_comparison_summary.csv
results_summaries/maxherding_tinyimagenet_maor_density_selection/selected_density_ratio_by_episode.csv
```

The main density result is that final accuracy tracks the effective graph degree
very clearly:

```text
global:  effective degree 0.729, final accuracy 20.29
K=10:    effective degree 0.001, final accuracy 15.15
K=60:    effective degree 0.004, final accuracy 15.49
K=250:   effective degree 0.028, final accuracy 17.25
K=500:   effective degree 0.077, final accuracy 17.96
K=1000:  effective degree 0.160, final accuracy 19.51
K=1500:  effective degree 0.244, final accuracy 19.41
K=3000:  effective degree 0.538, final accuracy 20.39
K=5000:  effective degree 0.889, final accuracy 20.45
K=10000: effective degree 1.416, final accuracy 20.81
```

Selected points are denser than the dataset average for the successful settings.
For example:

```text
global:  selected/dataset degree ratio 5.41
K=3000:  selected/dataset degree ratio 7.20
K=5000:  selected/dataset degree ratio 5.80
K=10000: selected/dataset degree ratio 5.03
```

Very small K values make the graph so sparse that the selected effective degree
is essentially zero, and performance collapses. This supports the same conclusion
we reached on CIFAR10 and CIFAR100: local sigma is mostly affecting the useful
graph scale and degree. Strong locality by itself is not what helps. The good
regime is where the graph is dense enough to provide meaningful coverage, and
large K behaves close to global sigma.
