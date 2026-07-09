# Bayesian coverage — classifier-input L2 normalization: before vs after

_Generated 2026-07-09 23:41 by `batch_utils/compare_norm_rerun.py`._

**What changed:** the `from_features` classifier historically trained on RAW
SimCLR features while the fixed 1-NN eval arm L2-normalizes (~+18 pts @100).
The rerun adds `--normalize_features` (L2-normalized classifier inputs, train
+ test). The AL side (kernel/posterior/selection) was ALWAYS normalized, so
lSet trajectories pair per-seed and deltas isolate the training-input change.
The probcover+1nn rows of the SUMMARY doc are unaffected (already normalized).

**Reading:** cells are mean test acc over seeds (n in parens when < 3);
Δ = normalized − raw, positive = normalization helps.

## compare

### compare @ 100 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes_additive | 13.4 | — | — | 7.7 | — | — | 5.6 | — | — | 2.0 | — | — |
| bayes_product | 24.6 | — | — | 21.6 | — | — | 14.9 | — | — | 4.6 | — | — |
| probcover | 23.7 | — | — | 19.7 | — | — | 13.1 | — | — | 5.9 | — | — |
| random | 8.5 | — | — | 5.8 | — | — | 4.6 | — | — | 2.6 | — | — |

### compare @ 300 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes_additive | 18.2 | — | — | 12.7 | — | — | 6.0 | — | — | 3.3 | — | — |
| bayes_product | 38.1 | — | — | 32.8 | — | — | 23.1 | — | — | 9.7 | — | — |
| probcover | 39.9 | — | — | 33.4 | — | — | 21.5 | — | — | 8.2 | — | — |
| random | 25.1 | — | — | 19.9 | — | — | 12.7 | — | — | 4.9 | — | — |

### compare @ 1000 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes_additive | 26.9 | — | — | 19.5 | — | — | 12.4 | — | — | 5.4 | — | — |
| bayes_product | 45.1 | — | — | 39.2 | — | — | 29.5 | — | — | 12.6 | — | — |
| probcover | 46.6 | — | — | 41.2 | — | — | 30.3 | — | — | 12.7 | — | — |
| random | 42.6 | — | — | 37.7 | — | — | 27.1 | — | — | 10.3 | — | — |

## trainmode

### trainmode @ 100 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| filtered | 5.6 | — | — | 22.4 | — | — | 4.3 | — | — | 1.3 | — | — |
| map_posterior | 22.5 | — | — | 20.6 | — | — | 15.2 | — | — | 5.8 | — | — |
| observed | 24.6 | — | — | 21.6 | — | — | 14.9 | — | — | 4.6 | — | — |
| reweighted | 5.8 | — | — | 8.6 | — | — | 5.0 | — | — | 1.3 | — | — |
| soft_posterior | 23.7 | — | — | 19.1 | — | — | 12.3 | — | — | 5.3 | — | — |

### trainmode @ 300 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| filtered | 24.2 | — | — | 28.9 | — | — | 16.5 | — | — | 6.5 | — | — |
| map_posterior | 35.9 | — | — | 33.7 | — | — | 27.9 | — | — | 14.2 | — | — |
| observed | 38.1 | — | — | 32.8 | — | — | 23.1 | — | — | 9.7 | — | — |
| reweighted | 25.1 | — | — | 26.3 | — | — | 17.6 | — | — | 5.2 | — | — |
| soft_posterior | 36.3 | — | — | 33.4 | — | — | 27.8 | — | — | 13.3 | — | — |

### trainmode @ 1000 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| filtered | 38.6 | — | — | 40.8 | — | — | 34.3 | — | — | 20.6 | — | — |
| map_posterior | 41.8 | — | — | 40.6 | — | — | 37.6 | — | — | 26.2 | — | — |
| observed | 45.1 | — | — | 39.2 | — | — | 29.5 | — | — | 12.6 | — | — |
| reweighted | 39.6 | — | — | 40.2 | — | — | 36.5 | — | — | 23.4 | — | — |
| soft_posterior | 42.5 | — | — | 41.5 | — | — | 39.1 | — | — | 27.5 | — | — |

## pseudo

### pseudo @ 100 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pseudo_agg50 | 26.3 | — | — | 19.7 | — | — | 12.5 | — | — | 5.2 | — | — |
| pseudo_all | 37.6 | — | — | 28.6 | — | — | 18.0 | — | — | 6.2 | — | — |
| pseudo_lin10 | 26.3 | — | — | 19.7 | — | — | 12.5 | — | — | 5.2 | — | — |

### pseudo @ 300 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pseudo_agg50 | 40.6 | — | — | 37.6 | — | — | 30.1 | — | — | 14.3 | — | — |
| pseudo_all | 42.6 | — | — | 37.7 | — | — | 31.1 | — | — | 14.4 | — | — |
| pseudo_lin10 | 40.6 | — | — | 37.6 | — | — | 30.1 | — | — | 14.3 | — | — |

### pseudo @ 1000 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pseudo_agg50 | 47.8 | — | — | 44.1 | — | — | 41.3 | — | — | 29.0 | — | — |
| pseudo_all | 47.5 | — | — | 45.8 | — | — | 41.8 | — | — | 28.1 | — | — |
| pseudo_lin10 | 46.3 | — | — | 43.8 | — | — | 40.9 | — | — | 28.0 | — | — |

---
_Runs found: 144 unnormalized (historical), 0 normalized (expected 144 each: compare 48 / trainmode 60 / pseudo 36)._
