# Local Sigma Experiments Summary

## Goal
Implement and evaluate a **local-sigma kernel** (self-tuning) for **MaxHerding** and **ProbCover**, and compare against the existing **global-sigma** baseline.

## What We Implemented

### 1) Local-Sigma Kernel (Self-Tuning)
Based on Zelnik-Manor & Perona (2004):

- Local scale per point:
```
sigma_i = distance from x_i to its K-th nearest neighbor
```
  (K=7 by default, configurable)

- RBF kernel with local scales:
```
k(x_i, x_j) = exp( - d(x_i, x_j)^2 / (sigma_i * sigma_j) )
```

### 2) ProbCover (Tophat) with Local Sigma
ProbCover uses a binary kernel, so we apply **normalized distance**:

- Keep edge if:
```
d(x_i, x_j)^2 / (sigma_i * sigma_j) < delta^2
```

### 3) Sparse Kernel Thresholds
- **Global sigma** (RBF): threshold uses kernel values as before.
- **Local sigma** (RBF): threshold uses **distance cutoff** `d < delta` to preserve sparsity behavior.

## Feature Normalization (Per Paper)

- **MaxHerding paper**: assumes **normalized features** (sigma=1 is “sensible for normalized features”).
- **ProbCover / DCoM**: CIFAR features are **not normalized** (normalization only explicit for ImageNet).

### Code Adjustment
We added a flag to control feature normalization:
- Default: normalized
- For ProbCover CIFAR: run **unnormalized**

## What Was Run
We duplicated the seed sbatches and ran **local-sigma** versions:

- MaxHerding CIFAR-10/100 (seeds 0–2)
- ProbCover CIFAR-10/100 (seeds 0–2)

Outputs organized as:
```
outputs/<method>/<dataset>/<global_sigma|local_sigma>/
```

## Results So Far (MaxHerding)

### CIFAR-10 (avg over seeds, 11 episodes)
Global sigma:
```
[70.86, 73.98, 77.26, 78.09, 78.10, 78.80, 79.39, 78.70, 79.22, 81.00, 81.16]
```
Local sigma:
```
[36.77, 55.15, 60.67, 62.80, 67.78, 68.47, 68.38, 68.21, 72.11, 73.53, 72.99]
```
Gap: **very large early (~-34%)**, narrows to ~-7 to -8% late.

### CIFAR-100 (avg over seeds, 11 episodes)
Global sigma:
```
[35.19, 39.74, 41.88, 43.16, 44.08, 44.68, 45.25, 45.72, 46.07, 46.40, 46.75]
```
Local sigma:
```
[23.26, 31.03, 35.12, 37.51, 38.94, 40.17, 41.31, 42.12, 42.77, 43.42, 43.90]
```
Gap: starts ~-12%, ends ~-3%.

## Interpretation (Working Hypotheses)
1. **Local sigma makes kernel too sparse early**, hurting coverage.
2. **Delta tuned for global sigma** may be inappropriate for local sigma.
3. Local scaling emphasizes dense regions early, reducing diversity of picks.

## Next Steps
- Re-run ProbCover local-sigma with **unnormalized** features (now configured).
- Small **K sweep**: K ∈ {7, 15, 30} on CIFAR-10 to test sensitivity.
- Log sigma stats (min/median/max) and kernel density to diagnose over-sparsity.
