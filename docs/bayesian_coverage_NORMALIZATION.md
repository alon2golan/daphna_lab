# Bayesian coverage — classifier-input L2 normalization: before vs after

_Generated 2026-07-10 10:24 by `batch_utils/compare_norm_rerun.py`._

**What changed:** the `from_features` classifier historically trained on RAW
SimCLR features while the fixed 1-NN eval arm L2-normalizes (~+18 pts @100).
The rerun adds `--normalize_features` (L2-normalized classifier inputs, train
+ test). The AL side (kernel/posterior/selection) was ALWAYS normalized, so
lSet trajectories pair per-seed and deltas isolate the training-input change.
The probcover+1nn rows of the SUMMARY doc are unaffected (already normalized).

**Reading:** cells are mean test acc over seeds (n in parens when < 3);
Δ = normalized − raw, positive = normalization helps.

## Takeaways (sweep 31048960 vs originals, 144 paired runs each)

1. **The effect flips sign with budget.** @1000 labels normalization helps
   almost every arm (+1 to +7 pts; observed +6.4/+6.8 at noise .8/.5,
   per-seed Δ tight: 7.3/6.6/6.5). @100 labels it HURTS badly (−5 to −15)
   for every arm that trains on only the ~100 labeled points; @300 is the
   crossover. Exception: `pseudo_all` (trains on ~10k pseudo points every
   round) is flat-to-positive at ALL budgets — consistent with the low-
   budget losses being a CONVERGENCE artifact: at fixed BASE_LR/200 epochs,
   tiny-budget training underfits isotropic inputs (smoke: train acc 50%
   vs 95% @100) while high-norm raw points produce larger logits and fit
   fast. The scale-preserving rescale fixed the gross collapse (unit-L2
   gave 3.3 @100) but not this residual. Low-budget normalized cells need
   an LR/epoch retune before being read as 'normalization is bad there'.
2. **§4 qualitative conclusions SURVIVE normalization** (read @1000, the
   converged regime): product ≫ additive everywhere (47.5 vs 29.9 clean);
   pseudo is still the best training arm at every noise (49.0/46.5+/42.8/
   29.9) though its clean-data edge over probcover+training closes to a
   tie (49.0 vs 49.0); posterior-training still beats observed under noise
   but the margin SHRINKS a lot (noise .5: +10 raw → +4.3 norm; noise .8:
   2.2× → 1.5×) because observed gains the most from normalization —
   i.e., part of what soft/map-posterior was fixing was feature-norm
   pathology, not only label noise.
3. **Absolute doc numbers are understated @1000** under a normalized
   convention (everything +2-7). The NPC+AUM rows in SUMMARY §4 were NOT
   rerun and remain raw-feature numbers — cross-compare with care.
4. **probcover+1nn vs pseudo @100 clean:** normalized `pseudo_all` (38.4)
   now ties 1-NN (38.1); on raw features 1-NN led (38.1 vs 37.6).
5. lin10 ≡ agg50 @100/300 in BOTH arms (schedules coincide) — determinism
   check passes on the normalized side too.

## compare

### compare @ 100 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes_additive | 13.4 | 2.4 | -10.9 | 7.7 | 1.9 | -5.8 | 5.6 | 2.5 | -3.1 | 2.0 | 1.2 | -0.8 |
| bayes_product | 24.6 | 11.7 | -12.9 | 21.6 | 11.3 | -10.3 | 14.9 | 10.1 | -4.8 | 4.6 | 3.8 | -0.8 |
| probcover | 23.7 | 13.1 | -10.6 | 19.7 | 10.8 | -8.9 | 13.1 | 6.4 | -6.7 | 5.9 | 3.8 | -2.0 |
| random | 8.5 | 6.8 | -1.7 | 5.8 | 4.3 | -1.5 | 4.6 | 3.8 | -0.7 | 2.6 | 2.1 | -0.5 |

### compare @ 300 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes_additive | 18.2 | 13.6 | -4.7 | 12.7 | 11.1 | -1.5 | 6.0 | 5.9 | -0.2 | 3.3 | 3.1 | -0.1 |
| bayes_product | 38.1 | 35.3 | -2.8 | 32.8 | 32.4 | -0.4 | 23.1 | 25.1 | +2.0 | 9.7 | 11.7 | +2.0 |
| probcover | 39.9 | 37.7 | -2.1 | 33.4 | 31.9 | -1.5 | 21.5 | 21.2 | -0.3 | 8.2 | 7.5 | -0.7 |
| random | 25.1 | 24.8 | -0.3 | 19.9 | 19.6 | -0.3 | 12.7 | 12.7 | -0.0 | 4.9 | 4.9 | -0.0 |

### compare @ 1000 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes_additive | 26.9 | 29.9 | +3.0 | 19.5 | 25.4 | +5.9 | 12.4 | 16.7 | +4.3 | 5.4 | 7.0 | +1.6 |
| bayes_product | 45.1 | 47.5 | +2.4 | 39.2 | 43.8 | +4.6 | 29.5 | 36.3 | +6.8 | 12.6 | 19.0 | +6.4 |
| probcover | 46.6 | 49.0 | +2.5 | 41.2 | 45.1 | +3.9 | 30.3 | 34.9 | +4.7 | 12.7 | 14.8 | +2.2 |
| random | 42.6 | 44.6 | +2.0 | 37.7 | 39.9 | +2.2 | 27.1 | 30.0 | +2.9 | 10.3 | 11.6 | +1.3 |

## trainmode

### trainmode @ 100 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| filtered | 5.6 | 5.8 | +0.1 | 22.4 | 14.4 | -8.0 | 4.3 | 4.4 | +0.1 | 1.3 | 1.4 | +0.0 |
| map_posterior | 22.5 | 8.2 | -14.3 | 20.6 | 9.4 | -11.2 | 15.2 | 10.0 | -5.2 | 5.8 | 5.5 | -0.3 |
| observed | 24.6 | 11.7 | -12.9 | 21.6 | 11.3 | -10.3 | 14.9 | 10.1 | -4.8 | 4.6 | 3.8 | -0.8 |
| reweighted | 5.8 | 4.6 | -1.2 | 8.6 | 4.5 | -4.1 | 5.0 | 2.6 | -2.4 | 1.3 | 1.2 | -0.1 |
| soft_posterior | 23.7 | 9.0 | -14.7 | 19.1 | 9.2 | -9.9 | 12.3 | 10.1 | -2.1 | 5.3 | 5.1 | -0.2 |

### trainmode @ 300 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| filtered | 24.2 | 20.6 | -3.7 | 28.9 | 25.9 | -3.0 | 16.5 | 15.2 | -1.3 | 6.5 | 6.7 | +0.1 |
| map_posterior | 35.9 | 32.8 | -3.1 | 33.7 | 31.1 | -2.5 | 27.9 | 27.3 | -0.6 | 14.2 | 13.7 | -0.6 |
| observed | 38.1 | 35.3 | -2.8 | 32.8 | 32.4 | -0.4 | 23.1 | 25.1 | +2.0 | 9.7 | 11.7 | +2.0 |
| reweighted | 25.1 | 19.9 | -5.2 | 26.3 | 20.4 | -5.9 | 17.6 | 9.0 | -8.6 | 5.2 | 2.7 | -2.6 |
| soft_posterior | 36.3 | 33.0 | -3.3 | 33.4 | 30.6 | -2.8 | 27.8 | 27.3 | -0.5 | 13.3 | 13.5 | +0.1 |

### trainmode @ 1000 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| filtered | 38.6 | 39.7 | +1.1 | 40.8 | 42.4 | +1.6 | 34.3 | 34.5 | +0.2 | 20.6 | 20.1 | -0.5 |
| map_posterior | 41.8 | 44.1 | +2.3 | 40.6 | 42.6 | +2.1 | 37.6 | 39.6 | +2.0 | 26.2 | 28.5 | +2.2 |
| observed | 45.1 | 47.5 | +2.4 | 39.2 | 43.8 | +4.6 | 29.5 | 36.3 | +6.8 | 12.6 | 19.0 | +6.4 |
| reweighted | 39.6 | 40.5 | +1.0 | 40.2 | 40.7 | +0.5 | 36.5 | 35.5 | -1.0 | 23.4 | 19.9 | -3.5 |
| soft_posterior | 42.5 | 44.6 | +2.0 | 41.5 | 43.3 | +1.8 | 39.1 | 40.6 | +1.5 | 27.5 | 29.4 | +1.9 |

## pseudo

### pseudo @ 100 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pseudo_agg50 | 26.3 | 11.4 | -14.9 | 19.7 | 9.6 | -10.1 | 12.5 | 10.1 | -2.4 | 5.2 | 5.1 | -0.1 |
| pseudo_all | 37.6 | 38.4 | +0.7 | 28.6 | 29.7 | +1.1 | 18.0 | 18.4 | +0.5 | 6.2 | 6.2 | +0.0 |
| pseudo_lin10 | 26.3 | 11.4 | -14.9 | 19.7 | 9.6 | -10.1 | 12.5 | 10.1 | -2.4 | 5.2 | 5.1 | -0.1 |

### pseudo @ 300 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pseudo_agg50 | 40.6 | 40.3 | -0.3 | 37.6 | 36.8 | -0.8 | 30.1 | 29.9 | -0.2 | 14.3 | 14.4 | +0.1 |
| pseudo_all | 42.6 | 43.3 | +0.8 | 37.7 | 38.7 | +1.1 | 31.1 | 32.2 | +1.2 | 14.4 | 15.0 | +0.5 |
| pseudo_lin10 | 40.6 | 40.3 | -0.3 | 37.6 | 36.8 | -0.8 | 30.1 | 29.9 | -0.2 | 14.3 | 14.4 | +0.1 |

### pseudo @ 1000 labels

| arm | n0.0 raw | n0.0 norm | Δ | n0.2 raw | n0.2 norm | Δ | n0.5 raw | n0.5 norm | Δ | n0.8 raw | n0.8 norm | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pseudo_agg50 | 47.8 | 49.0 | +1.2 | 44.1 | 46.5 | +2.4 | 41.3 | 42.8 | +1.5 | 29.0 | 29.9 | +0.9 |
| pseudo_all | 47.5 | 48.4 | +0.9 | 45.8 | 46.6 | +0.8 | 41.8 | 42.6 | +0.8 | 28.1 | 29.2 | +1.1 |
| pseudo_lin10 | 46.3 | 47.8 | +1.4 | 43.8 | 45.8 | +2.0 | 40.9 | 42.5 | +1.5 | 28.0 | 29.5 | +1.5 |

---
_Runs found: 144 unnormalized (historical), 144 normalized (expected 144 each: compare 48 / trainmode 60 / pseudo 36)._
