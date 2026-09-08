# Professor Notes Update - 2026-06-13

Baseline document used: `Research Summary after notes.docx`

Updated document created: `Research Summary updated 2026-06-13.docx`

## Main Document Updates

- Added CIFAR100-only absolute-accuracy plots for the new notes.
- Kept the existing CIFAR10 figures for now, but did not add new CIFAR10 analysis.
- Strengthened the phrasing around the positive CIFAR100 result: localization and density normalization do improve accuracy in the low-budget regime when K or alpha are selected appropriately.
- Kept the conclusion nuanced: the gains are budget-dependent, and fixed K/alpha schedules are still too rigid.

## New Figures Added

- Figure 3: CIFAR100 global sigma compared with the best local-sigma K at each budget.
- Figure 12: CIFAR100 alpha=0 compared with the best alpha at each budget for global sigma and local-sigma K=1440.
- Figure 13: CIFAR100 global/local sigma with alpha=0 and selected alpha at each budget.
- Figure 18: CIFAR100 absolute accuracy across delta values with alpha=0.5.
- Figure 23: CIFAR100 absolute accuracy of alpha decay schedules with global sigma.
- Figure 24: CIFAR100 absolute accuracy of alpha decay schedules with local-sigma K=1440.

The document was renumbered by actual figure order. It now contains Figure 1 through Figure 24 sequentially. Figure 13 labels the normalized curves as selected-alpha curves and annotates the selected alpha value at each budget.

## Main Results Added

- In the cutoff K sweep averaged over seeds 3, 4, and 5, the best local K beats global sigma at every CIFAR100 budget from 100 to 1000 labels.
- The best K is K=2000 from 100 to 600 labels and K=1440 from 700 to 1000 labels.
- Best-alpha density normalization improves CIFAR100 low-budget accuracy, especially around 100 to 400 labels.
- For local-sigma K=1440, alpha=0.5 is best from 100 to 400 labels, and alpha=0 is usually preferred later.
- Delta changes the absolute accuracy level and the size of the density-normalization effect, but the qualitative trend remains: density normalization is most useful early and weakens later.
- Existing fixed decay schedules do not solve the late-budget problem; absolute-accuracy plots make this clearer than gain/loss plots.

## New Experiment Plan Added In Code

Created:

- `scripts/sbatch/maxherding_density_norm/maxherding_cifar100_seeds34_density_norm_decay_followup.sh`

This runs 8 jobs:

- Seeds: 3 and 4
- Sigma settings: global sigma and local-sigma K=1440
- Schedule 1: `early_only` with `steps=5`, keeping alpha active through 500 labeled points before turning it off.
- Schedule 2: `three_phase_floor` with `steps=4`, using alpha, then half alpha, then quarter alpha instead of zeroing alpha in the final phase.

Code support added:

- `three_phase_floor` schedule in `coverage_matrix_methods.py`
- CLI support for `three_phase_floor` in `train_al.py`
- Aggregation support for `three_phase_floor` in `plot_density_norm_decay_tuned.py`

Run command:

```bash
sbatch scripts/sbatch/maxherding_density_norm/maxherding_cifar100_seeds34_density_norm_decay_followup.sh
```

After the jobs finish, aggregate with:

```bash
python scripts/plot_density_norm_decay_tuned.py
```

## Notes From Ambiguous Meeting Comments

Comment: "show dependency on delta in absolute way"

Likely meaning: avoid gain/loss relative to alpha=0 and instead plot actual accuracy curves for different delta values. This is now Figure 18.

Comment: "global, local, global normalized and local normalized with alpha"

Likely meaning: compare four curves in one place:

- Global sigma without density normalization
- Local sigma without density normalization
- Global sigma with density normalization
- Local sigma with density normalization

I generated this candidate visual but did not insert it into the main document yet:

- `visual_outputs/professor_notes_2026_06_13/cifar100_global_local_best_alpha_four_curves.png`

This graph may be useful if we want a compact visual showing how localization and alpha normalization interact.
