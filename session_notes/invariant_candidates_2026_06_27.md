# Invariant Candidate Analysis - 2026-06-27

## Motivation

Daphna suggested that a useful finishing question is whether we can identify statistics that are robust to budget changes. The main candidate we checked is the average graph degree of the remaining unlabeled nodes when the other hyperparameters are chosen at their empirical optimum. We also checked whether useful graph-degree ranges appear related to dataset-level quantities such as the number of classes and class separability in the feature space.

This analysis did not launch new active-learning jobs. It uses the completed MaxHerding runs and recomputes diagnostics from the saved `uSet.npy`, `lSet.npy`, and `activeSet.npy` files.

## What Was Computed

For each dataset and budget, we first selected the empirically best local-sigma setting from the completed K sweep:

- CIFAR10 used the 10-90 label budgets available in the saved selection diagnostics.
- CIFAR100 used the 100-900 label budgets available in the saved selection diagnostics.
- TinyImageNet used the 200-1800 label budgets available in the saved selection diagnostics.

For the best setting at each budget, we estimated the remaining-pool graph degree:

- We sampled 512 anchors from the current unlabeled pool `uSet`.
- For each anchor, we counted its graph neighbors inside the same remaining unlabeled pool.
- For local sigma, the effective neighbor condition was the same local scaled distance used in the graph.
- The result is a sampled estimate of the average remaining-node degree, not an exact all-pairs average.

We also computed simple feature-space separability statistics:

- Number of classes.
- Mean distance from each point to its true-class centroid.
- Mean distance between true-class centroids.
- Separability ratio: mean inter-centroid distance divided by mean intra-class distance.
- Nearest-centroid train accuracy in the SSL feature space.

## Main Outputs

CSV outputs:

- `results_summaries/invariant_candidates_2026_06_27/best_setting_by_budget.csv`
- `results_summaries/invariant_candidates_2026_06_27/remaining_degree_at_best_setting_by_seed.csv`
- `results_summaries/invariant_candidates_2026_06_27/remaining_degree_at_best_setting_mean.csv`
- `results_summaries/invariant_candidates_2026_06_27/remaining_degree_stability_by_dataset.csv`
- `results_summaries/invariant_candidates_2026_06_27/dataset_class_separability.csv`
- `results_summaries/invariant_candidates_2026_06_27/dataset_cross_summary.csv`

Visual outputs:

- `visual_outputs/invariant_candidates_2026_06_27/cifar10_best_setting_remaining_degree.png`
- `visual_outputs/invariant_candidates_2026_06_27/cifar100_best_setting_remaining_degree.png`
- `visual_outputs/invariant_candidates_2026_06_27/tinyimagenet_best_setting_remaining_degree.png`
- `visual_outputs/invariant_candidates_2026_06_27/remaining_degree_budget_stability_by_dataset.png`
- `visual_outputs/invariant_candidates_2026_06_27/dataset_separability_vs_best_degree.png`

## Findings

CIFAR100 gives the clearest candidate invariant. When we choose the empirically best K at each budget, the estimated average remaining-pool effective degree stays in a relatively narrow range: roughly 14-21, with a budget-wise CV of about 0.14. The best K is K=2000 up to 600 labels and K=1440 from 700 labels onward. This suggests that for CIFAR100, the useful configurations may be maintaining a similar effective graph degree rather than depending on locality itself.

CIFAR10 is less stable. Its best K changes between K=10000, K=5000, and K=3500, and the estimated remaining effective degree ranges roughly from 134 to 302, with a budget-wise CV of about 0.37. This does not look like a clean invariant in absolute degree. It may still indicate a broad dense-regime preference, but not a stable target degree.

TinyImageNet is also not stable in this estimate. The best setting is usually K=10000, but the remaining effective degree is extremely small and noisy, roughly 0.4-1.7, with only one seed available. Because TinyImageNet has one seed and a much sparser effective graph under this delta, this should be treated as suggestive rather than conclusive.

Across datasets, the useful final effective degree decreases sharply as the number of classes increases:

- CIFAR10: 10 classes, best final effective degree about 154.
- CIFAR100: 100 classes, best final effective degree about 20.
- TinyImageNet: 200 classes, best final effective degree about 1.4.

This supports Daphna's point that the important quantity may be degree, but the target degree is probably dataset-dependent rather than universal in raw absolute value. The number of classes and the separability of the SSL representation likely affect what degree range is useful.

The separability statistics are directionally consistent with task difficulty:

- CIFAR10 has the highest nearest-centroid train accuracy in feature space: about 0.838.
- CIFAR100 is lower: about 0.536.
- TinyImageNet is much lower: about 0.294.

However, the simple separability ratio alone does not fully explain the pattern. CIFAR100 has the largest inter/intra centroid ratio but is harder than CIFAR10 by nearest-centroid accuracy. This means the geometry is more subtle than one scalar centroid-ratio statistic.

## Interpretation

The most useful phrasing for the professor is:

The strongest candidate invariant we found is not a fixed K and not locality itself, but an effective graph-degree regime. On CIFAR100, when K is chosen optimally per budget, the remaining-pool effective degree stays fairly stable across budget. This supports the hypothesis that MaxHerding is mainly sensitive to the degree/density structure induced by the kernel. For CIFAR10 and TinyImageNet the same statistic is less stable, so the invariant is likely dataset-dependent and should be normalized or conditioned on dataset complexity, such as number of classes and feature-space separability.

## Limitations

- Remaining-pool degree was estimated with 512 sampled anchors per episode, not exact all-pairs degree.
- TinyImageNet currently has one seed in this diagnostic.
- The saved selection diagnostics cover active-selection episodes, not always the final post-run labeled size used in some accuracy tables.
- The separability analysis used simple true-class centroid statistics. More advanced cluster-based separability could be added, but this is probably beyond the necessary finishing scope unless Daphna asks for it.

## Suggested Response to Daphna

We evaluated the invariant candidate she suggested by taking, for each budget, the empirically best local-sigma K and measuring the average effective degree of the remaining unlabeled pool. On CIFAR100 this statistic is relatively stable across budgets, around 14-21 remaining-neighbor degree, even though the best K switches from 2000 to 1440. This supports the idea that the algorithm is primarily controlled by the induced graph degree rather than by K directly.

For CIFAR10 and TinyImageNet the same raw degree statistic is less stable, so a universal absolute-degree invariant is unlikely. The target degree seems dataset-dependent and probably depends on the number of classes and on how separable the SSL features are. The next possible refinement would be to normalize the target degree by dataset complexity, for example by class count or by a separability statistic, but for the current project this already gives a useful finishing conclusion: degree/density is the more robust explanatory variable, while locality is mainly one mechanism for changing that degree.
