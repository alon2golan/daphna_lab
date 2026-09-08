# Session Notes: MaxHerding Density Normalization Summary

## Goal Since The Last Meeting

We focused on MaxHerding with the validated 1NN evaluation path and tried to
understand whether local kernel modifications actually improve the selection
process.

The main research question was:

- Can we modify the MaxHerding kernel so that it balances dense and sparse
  regions better than the global-sigma kernel?
- If yes, does the benefit depend on the label budget?
- Can we identify which K/alpha setting is best for each budget regime?

We kept the same basic active-learning setup:

- method: MaxHerding
- classifier/evaluation: 1NN on SSL features
- kernel: RBF with ProbCover-style cutoff
- CIFAR10 budget: 10 to 100 labels
- CIFAR100 budget: 100 to 1000 labels

## What We Did

### 1. Density and Selection Diagnostics

We analyzed existing local-sigma K sweeps and measured:

- effective graph degree,
- local sigma statistics,
- selected-point density,
- selected-class diversity,
- final accuracy.

Main result:

- K mostly controls graph density.
- Small K makes the graph too sparse.
- Large K makes the local sigmas more uniform, so the kernel becomes closer to
  global sigma.
- Selected points are usually denser than the dataset average, meaning
  MaxHerding already tends to select central/dense points.

This suggests that local sigma alone does not strongly solve the dense-vs-sparse
balance problem.

### 2. Density-Normalized Kernel

We added alpha density normalization:

```text
K_ij <- K_ij / degree_i^alpha degree_j^alpha
```

in the symmetric version, implemented as:

```text
K_ij <- K_ij / (degree_i degree_j)^(alpha / 2)
```

Intuition:

- positive alpha penalizes points in dense regions,
- alpha = 0 is the original kernel,
- larger alpha makes the penalty stronger.

We tested:

- global sigma,
- best local K from prior experiments,
- weighted and effective degree,
- several fixed alphas,
- then dynamic alpha schedules.

Weighted and effective degree behaved almost the same, so we kept weighted
degree going forward.

### 3. Wide Fixed-Alpha Sweep

We tested fixed alpha values:

```text
alpha = 0, 0.1, 0.25, 0.5, 0.75, 1.0
```

For:

- CIFAR10: global and K=3500
- CIFAR100: global and K=1440

Main result:

- alpha normalization can help at very low budgets.
- alpha = 1.0 is too aggressive and collapses performance.
- alpha is not consistently good at later budgets.

Best observed fixed-alpha settings by budget, seed 3:

```text
CIFAR10:
10 labels:  global, alpha=0
20 labels:  K=3500, alpha=0.75
30 labels:  K=3500, alpha=0.75
40 labels:  K=3500, alpha=0.5
50 labels:  K=3500, alpha=0
60 labels:  K=3500, alpha=0
70 labels:  K=3500, alpha=0.1
80 labels:  K=3500, alpha=0
90 labels:  global, alpha=0.5
100 labels: K=3500, alpha=0.25

CIFAR100:
100 labels:  K=1440, alpha=0.5
200 labels:  K=1440, alpha=0.5
300 labels:  K=1440, alpha=0.5
400 labels:  K=1440, alpha=0.5
500 labels:  global, alpha=0.1
600 labels:  K=1440, alpha=0
700 labels:  global, alpha=0.25
800 labels:  global, alpha=0.1
900 labels:  global, alpha=0
1000 labels: K=1440, alpha=0.1
```

Important caveat:

- This exact best-by-budget table is from the wide alpha sweep on seed 3.
- It is useful for identifying trends, but should not yet be treated as a final
  robust rule until repeated across more seeds.

### 4. Dynamic Alpha / Decay Schedules

Because alpha helped early and hurt later, we tried dynamic schedules:

- early-only alpha,
- linear decay,
- three-phase decay.

These were run on seeds 3 and 4.

Main result:

- Decay is not useful for final accuracy.
- It preserves some early-budget gains, especially on CIFAR100.
- But every tested decay schedule ended below the alpha=0 baseline at the final
  budget.

Best final deltas vs alpha=0:

```text
CIFAR10 global:  best decay final delta = -1.965
CIFAR10 K=3500:  best decay final delta = -0.675
CIFAR100 global: best decay final delta = -0.490
CIFAR100 K=1440: best decay final delta = -0.545
```

So the conclusion is not that alpha normalization is useless. The conclusion is:

- fixed alpha can be useful at very low budgets,
- simple decay schedules are not enough to preserve that early gain through the
  full budget,
- optimal alpha appears budget-dependent.

## Main Interpretation

The current picture is:

- Local sigma K mostly changes the graph density.
- Very small K makes the graph too disconnected.
- Very large K becomes too similar to global sigma.
- Medium/large K is the only useful region.
- Alpha normalization penalizes dense regions.
- This can help early, probably by reducing redundant dense-mode selection.
- Later, that same penalty can overcorrect and hurt coverage of representative
  dense regions.

This explains why alpha can help at the very low budget but not necessarily
later. At very low budgets, MaxHerding may over-focus on the densest regions, so
some density penalty helps diversify. Later, after the main dense regions are
already represented, continuing the penalty may push selection away from useful
representative points.

## What We Can Tell The Professor

The clean summary is:

> We validated that local K and alpha normalization affect the density structure
> of the MaxHerding kernel. Small K is too sparse, large K is close to global,
> and the useful K range is medium/large. Adding alpha density normalization can
> improve very low-budget performance, but it is budget-sensitive. Fixed alpha
> helps around 20-30 labels on CIFAR10 and around 100-400 labels on CIFAR100,
> but it often hurts later. We tried simple decay schedules to keep the early
> gains and remove the late penalty; however, the tested decays did not improve
> final accuracy. This suggests that the right rule is not a simple monotonic
> decay, but a budget-dependent or adaptive choice of K and alpha.

More cautious version:

> The current evidence does not show that local sigma or alpha normalization
> gives a robust final-accuracy improvement. It does show that the best kernel
> configuration changes with budget, which may be the more interesting direction.

## Best Graphs To Send

Use these graphs first:

1. `visual_outputs/maxherding/density_norm/professor_summary/cifar10_effective_degree_vs_final_accuracy.png`
2. `visual_outputs/maxherding/density_norm/professor_summary/cifar100_effective_degree_vs_final_accuracy.png`

These explain the K story:

- small K is too sparse,
- medium/large K is useful,
- very large K becomes global-like.

Then send:

3. `visual_outputs/maxherding/density_norm/professor_summary/cifar10_best_fixed_alpha_by_budget_seed3.png`
4. `visual_outputs/maxherding/density_norm/professor_summary/cifar100_best_fixed_alpha_by_budget_seed3.png`

These show that the best alpha/K setting changes with budget.

Then send:

5. `visual_outputs/maxherding/density_norm/professor_summary/cifar10_best_decay_delta_by_budget.png`
6. `visual_outputs/maxherding/density_norm/professor_summary/cifar100_best_decay_delta_by_budget.png`

These show that decay helps only in early budgets and does not solve the final
accuracy problem.

Optional detailed graphs:

7. `visual_outputs/maxherding/density_norm/wide_alpha_summary/cifar10_k3500_all_alphas_delta.png`
8. `visual_outputs/maxherding/density_norm/wide_alpha_summary/cifar100_k1440_all_alphas_delta.png`
9. `visual_outputs/maxherding/density_norm/decay_tuned/cifar10_k3500_delta.png`
10. `visual_outputs/maxherding/density_norm/decay_tuned/cifar100_k1440_delta.png`

## Tables To Reference

Professor-summary tables:

- `results_summaries/maxherding_density_norm/professor_summary/best_fixed_alpha_by_budget_seed3.csv`
- `results_summaries/maxherding_density_norm/professor_summary/best_fixed_alpha_within_setting_seed3.csv`
- `results_summaries/maxherding_density_norm/professor_summary/best_decay_by_budget_avg_seeds34.csv`
- `results_summaries/maxherding_density_norm/professor_summary/decay_final_summary_avg_seeds34.csv`
- `results_summaries/maxherding_density_norm/professor_summary/decay_phase_summary_avg_seeds34.csv`
- `results_summaries/maxherding_density_norm/professor_summary/k_density_final_accuracy_avg_seeds345.csv`

## Proposed Next Step

The next experiment should explicitly study budget-dependent optimal
configuration.

Rather than testing more decay schedules blindly, test a compact grid over:

- K: global and the best local K values,
- alpha: 0, 0.1, 0.25, 0.5, 0.75,
- seeds: at least 3 seeds,
- same MaxHerding + 1NN + cutoff setup.

Then report:

- best configuration at each budget,
- whether the best configuration is stable across seeds,
- whether the transition point from alpha > 0 to alpha = 0 is consistent.

If the budget-dependent pattern is stable, it could motivate an adaptive rule:

- use density normalization only in the earliest rounds,
- choose alpha based on graph density or selected-point density,
- turn it off once the selected set becomes diverse enough.

The current decay results suggest that a fixed schedule is too crude. The next
interesting question is whether the switch point can be inferred from density
statistics rather than manually chosen.
