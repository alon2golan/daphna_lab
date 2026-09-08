# Tiny ImageNet Degree Study

Goal: test whether the main conclusion from CIFAR10/CIFAR100 also holds on
Tiny ImageNet: MaxHerding behavior is explained more by graph degree than by
locality itself.

## Phase 0: Delta Calibration

Run one delta probe first:

```bash
sbatch scripts/sbatch/tinyimagenet_degree_study/tinyimagenet_delta_probe_seed3.sh
```

After it finishes, read:

```bash
cat results_summaries/maxherding_tinyimagenet_density_selection/tinyimagenet_delta_seed3.txt
```

Use the printed value as `DELTA_OVERRIDE` for the MaxHerding jobs.

## Phase 1: Seed 3 Feasibility Study

Run a global reference and a local-sigma K sweep:

```bash
DELTA_OVERRIDE=<delta_from_probe> sbatch scripts/sbatch/tinyimagenet_degree_study/maxherding_tinyimagenet_seed3_nn1_cutoff_global.sh
DELTA_OVERRIDE=<delta_from_probe> sbatch scripts/sbatch/tinyimagenet_degree_study/maxherding_tinyimagenet_seed3_nn1_cutoff_k_sweep.sh
```

The K sweep uses:

- 10
- 60
- 250
- 500
- 1000
- 1500
- 3000
- 5000
- 10000

This intentionally includes very small K values, medium K values, the
CIFAR100/CIFAR10 useful range, and one larger K value. The goal is to see
whether Tiny ImageNet also has the same pattern: small K creates sparse graphs,
medium K is useful, and large K becomes closer to global sigma.

## Phase 2: Diagnostics

After the runs finish:

```bash
sbatch scripts/sbatch/tinyimagenet_degree_study/maxherding_tinyimagenet_density_selection_seed3.sh
```

The diagnostics write CSV files under:

```text
results_summaries/maxherding_tinyimagenet_density_selection/
```

## Interpretation Target

The main plot we want for the research summary is:

- X-axis: effective graph degree
- Y-axis: final accuracy
- Points: global sigma and local-sigma K values

If Tiny ImageNet follows CIFAR10/CIFAR100, we expect:

- Very small effective degree should hurt
- Very large K should become global-like
- The best region should be a medium or medium-large effective degree
- Degree should explain performance better than K itself

If this does not happen, the degree explanation may be CIFAR-specific or may
depend on the feature representation.
