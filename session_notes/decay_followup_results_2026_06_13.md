# Decay Follow-Up Results - 2026-06-13

Experiments analyzed:

- CIFAR100
- MaxHerding with 1NN
- Global sigma and local-sigma K=1440
- Density normalization alpha=0.5
- Seeds 3 and 4

New schedules:

- `early_only_5`: alpha is active through 500 labels, then set to 0.
- `three_phase_floor_4`: alpha is full strength through 400 labels, half strength through 800 labels, then quarter strength instead of zero.

## Output Files

Clean plots:

- `visual_outputs/professor_notes_2026_06_13/decay_followup/cifar100_decay_followup_absolute_accuracy.png`
- `visual_outputs/professor_notes_2026_06_13/decay_followup/cifar100_decay_followup_early_only_absolute.png`
- `visual_outputs/professor_notes_2026_06_13/decay_followup/cifar100_decay_followup_three_phase_absolute.png`
- `visual_outputs/professor_notes_2026_06_13/decay_followup/cifar100_decay_followup_best_schedule_per_budget.png`
- `visual_outputs/professor_notes_2026_06_13/decay_followup/cifar100_decay_followup_final_accuracy_bars.png`

Clean tables:

- `results_summaries/professor_notes_2026_06_13/decay_followup/cifar100_decay_followup_by_episode_mean.csv`
- `results_summaries/professor_notes_2026_06_13/decay_followup/cifar100_decay_followup_final_mean.csv`
- `results_summaries/professor_notes_2026_06_13/decay_followup/cifar100_decay_followup_phase_summary.csv`
- `results_summaries/professor_notes_2026_06_13/decay_followup/cifar100_decay_followup_best_schedule_by_budget.csv`

## Main Findings

The new schedules improve some parts of the decay behavior, but they do not beat the alpha=0 baseline at the final budget.

For global sigma:

- `early_only_5` improves over `early_only_4` around 500, 800, and 900 labels.
- Final accuracy is unchanged compared with `early_only_4`: both end at 46.755.
- `three_phase_floor_4` is only slightly better than the old `three_phase_4` at 1000 labels: 46.580 versus 46.550.
- Alpha=0 baseline final accuracy is 47.393.

For local-sigma K=1440:

- `early_only_5` improves mid-budget accuracy compared with `early_only_4`, especially around 500 to 900 labels, but final accuracy is slightly lower: 46.875 versus 46.910.
- `three_phase_floor_4` is the best decay schedule at final budget: 47.065.
- It improves over old `three_phase_4` by +0.245 at 1000 labels.
- It is still below the alpha=0 local-sigma baseline final accuracy of 47.590.

## Interpretation

Keeping some alpha in the final phase helps local-sigma K=1440 more than fully zeroing alpha, so the idea is directionally useful. However, the fixed schedule still does not recover the alpha=0 final baseline.

This supports the same broader conclusion:

- Density normalization helps early.
- Keeping it too strong for too long can hurt.
- Turning it off or reducing it by a fixed hand-written schedule is still too crude.
- The next useful step is probably an adaptive rule based on graph/selection statistics, not another fixed schedule.

The most useful visual for discussion is probably:

- `cifar100_decay_followup_best_schedule_per_budget.png`

It shows that the best fixed schedule changes across the budget, which supports the need for budget-dependent or statistic-dependent adaptation.
