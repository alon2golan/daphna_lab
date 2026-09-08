# CIFAR100 Degree Evidence - 2026-06-27

## Goal

The remaining question from Daphna was whether we can identify invariants that are robust to budget changes. For now we focus only on CIFAR100, because this is the most relevant dataset and the previous multi-dataset comparison was too broad for a clean conclusion.

The main hypothesis is:

MaxHerding performance is controlled more by the effective graph degree induced by the kernel than by the local-sigma K itself.

## Existing-Run Evidence

I aggregated the completed CIFAR100 MaxHerding + 1NN cutoff K sweep over seeds 3, 4, and 5.

Main output directory:

`results_summaries/cifar100_degree_evidence_2026_06_27`

Main visual directory:

`visual_outputs/cifar100_degree_evidence_2026_06_27`

### Average Graph Degree of Remaining Nodes

For each budget, I chose the empirically best setting and measured the effective graph degree.

Two related degree statistics were computed:

- Dataset effective degree of the best setting.
- Sampled remaining-pool effective degree, measured on the unlabeled nodes left at that budget.

Results:

- Best-setting dataset effective degree has mean 22.75 and CV 0.10.
- Best-setting remaining-pool effective degree has mean 18.40 and CV 0.13.
- The best K changes from K=2000 at 100-600 labels to K=1440 at 700-900 labels.
- Even though K changes, the useful effective degree remains in a relatively narrow band.

This is the strongest current evidence for a degree-like invariant on CIFAR100.

Useful files:

- `cifar100_best_degree_stability_summary.csv`
- `cifar100_best_setting_degree_and_remaining_degree.png`
- `cifar100_top3_setting_degree_by_budget.png`

### Accuracy as a Function of Effective Degree

The top settings per budget mostly lie in the same effective-degree region.

Observed best-degree band:

- K=2000 gives dataset effective degree about 24.3.
- K=1440 gives dataset effective degree about 19.7.
- These two settings dominate the best results across budgets.

The degree-performance correlation is strongest at the lowest budgets and weakens later:

- At 100 labels, Spearman correlation between accuracy and degree is about 0.93.
- At 900 labels, Spearman correlation drops to about 0.30.

Interpretation:

Degree matters most at very low budget. As the labeled set grows, several nearby graph settings become competitive, so the exact degree matters less.

Useful files:

- `cifar100_accuracy_by_budget_and_degree.csv`
- `cifar100_accuracy_degree_correlations_by_budget.csv`
- `cifar100_accuracy_vs_effective_degree_by_budget.png`

### Selected-Point Degree Bias

Even at the best setting, MaxHerding selects denser-than-average points.

The selected batch effective degree divided by dataset effective degree:

- About 3.20 at 100 labels.
- About 2.00 at 300 labels.
- About 1.52 at 600 labels.
- About 1.48 at 900 labels.

Interpretation:

The algorithm initially strongly favors high-degree points. This bias decreases with budget, but remains above 1. The selected points are still denser than average even when using the best K.

Useful files:

- `cifar100_selected_degree_ratio_at_best_setting.csv`
- `cifar100_selected_to_dataset_degree_ratio_best_setting.png`

### Normalized Degree

The remaining effective degree can also be normalized by the number of remaining nodes, but on CIFAR100 this is less interpretable because the remaining pool size changes only from about 49,900 to 49,100 over our budget range. The raw effective degree is more readable and essentially captures the same trend here.

## Cluster Separability / Number of Classes

Daphna also mentioned dependence on number of classes or separability when clustered to k classes. Since we are currently focusing on CIFAR100 only, I tested this by clustering the CIFAR100 SSL features into k groups:

- k=10
- k=20
- k=50
- k=100
- k=200

The separability ratio increases with k, mostly because clusters become smaller and tighter. However, silhouette scores remain low, around 0.04-0.06, and true-label purity is still only about 0.43 even at k=200.

Interpretation:

The CIFAR100 feature space is structured but still highly overlapping. This supports the idea that there is no universal absolute degree that should work for every dataset; the useful degree should depend on the feature geometry and effective class/cluster structure.

Useful files:

- `cifar100_cluster_separability_by_k.csv`
- `cifar100_cluster_separability_by_k.png`

## Stronger Evidence: Controlled-Degree Experiment

The retrospective evidence is useful, but it does not fully prove that degree is the causal factor. K, delta, sigma, and graph degree all change together. The stronger experiment is to control the induced degree.

I created this sbatch:

`scripts/sbatch/cifar100_degree_controlled/maxherding_cifar100_degree_controlled_seeds345.sh`

It runs CIFAR100 MaxHerding + 1NN with the same low-budget setup:

- Budget 100 per episode.
- 9 active-learning episodes.
- Seeds 3, 4, and 5.
- ProbCover-style cutoff.
- g0 GPU group.
- At most 4 GPUs concurrently.

It tests three target degree bands:

- Low degree: around 10-13 effective neighbors.
- Optimal degree: around 18-23 effective neighbors.
- High degree: around 30-36 effective neighbors.

For each target band, it uses multiple different mechanisms to reach a similar degree. For example, the optimal band includes global sigma with larger delta, small K with larger delta, and larger K with smaller delta.

### What Would Support the Degree Hypothesis

The degree hypothesis is supported if:

- Configurations inside the same target-degree band produce similar accuracy, even when they use different K and delta.
- The optimal-degree band consistently outperforms the low-degree and high-degree bands.
- Differences between degree bands are larger than differences within the same band.

### What Would Weaken the Degree Hypothesis

The hypothesis is weakened if:

- Same-degree configurations produce very different accuracies.
- A specific K wins even when its induced degree is matched by other K/delta combinations.
- Delta or locality explains performance better than effective degree.

### Controlled-Degree Results

All 36 runs completed successfully:

- 12 configurations.
- 3 seeds: 3, 4, and 5.
- Three target degree bands: low, optimal, and high.
- CIFAR100 MaxHerding with 1NN and ProbCover-style cutoff.

Main output directory:

`results_summaries/cifar100_degree_controlled_2026_06_27`

Main visual directory:

`visual_outputs/cifar100_degree_controlled_2026_06_27`

At the final budget of 1000 labels, the average accuracy by degree band was:

- Low degree, around 10-13 effective neighbors: 47.41.
- Optimal degree, around 18-23 effective neighbors: 47.17.
- High degree, around 30-36 effective neighbors: 46.62.

The best individual final configuration was still inside the intended optimal-degree band:

- Local sigma with K=1440, delta=0.65, estimated effective degree about 18.3.
- Final accuracy: 47.59.

However, the low-degree band was very competitive and had the best average final accuracy across its configurations. This means the result should not be phrased as "one fixed optimal degree wins everywhere." The more accurate conclusion is that very high degree consistently hurts, while low-to-moderate degree settings are the useful regime.

The best configuration also changed across budget:

- From 100 to 600 labels, the best setting was K=480, delta=0.75, estimated effective degree about 21.8.
- From 700 to 1000 labels, the best setting was K=1440, delta=0.65, estimated effective degree about 18.3.

This supports a budget-dependent interpretation. Earlier budgets can benefit from a slightly denser graph, while later budgets prefer a somewhat lower/moderate degree.

The correlation between estimated effective degree and accuracy also changes with budget:

- At 100 labels, Spearman correlation is about 0.80.
- At 200 labels, Spearman correlation is about 0.82.
- At 500 labels, Spearman correlation is about -0.24.
- At 900 labels, Spearman correlation is about -0.72.
- At 1000 labels, Spearman correlation is about -0.76.

This is important: degree is not monotonically beneficial. High degree helps explain early performance, but later in the run too much degree becomes harmful.

Useful files:

- `cifar100_degree_controlled_by_seed_episode.csv`
- `cifar100_degree_controlled_by_config.csv`
- `cifar100_degree_controlled_final_by_config.csv`
- `cifar100_degree_controlled_by_band.csv`
- `cifar100_degree_controlled_final_by_band.csv`
- `cifar100_degree_controlled_budget_best.csv`
- `cifar100_degree_controlled_correlations.csv`
- `cifar100_degree_controlled_band_curves.png`
- `cifar100_degree_controlled_final_accuracy_vs_degree.png`
- `cifar100_degree_controlled_best_degree_per_budget.png`
- `cifar100_degree_controlled_final_accuracy_by_config.png`

### Updated Interpretation

The controlled-degree experiment strengthens the claim that graph degree is a major factor, but refines the hypothesis.

The evidence does not support a simple rule such as "larger degree is better" or "one fixed degree is optimal for all budgets." Instead:

- K is mainly useful because it changes the induced graph degree.
- Very high degree is consistently worse at later budgets.
- The best settings lie in a low-to-moderate degree regime.
- The preferred degree appears budget-dependent.
- The transition is visible: degree around 21.8 is best early, while degree around 18.3 is best later.

This suggests that the practical invariant may be a stable useful degree range, or an upper degree threshold to avoid over-dense graphs, rather than a single constant degree.

## Suggested Current Message to Daphna

For CIFAR100, the strongest invariant candidate is the effective graph degree. When choosing the best K at each budget, the best K changes from 2000 to 1440, but the induced effective degree remains relatively stable. The dataset effective degree of the best setting has CV about 0.10, and the sampled remaining-pool degree has CV about 0.13. This suggests that K is not the main object; K matters because it controls the induced graph degree.

The remaining selected points are also informative: MaxHerding selects points with substantially higher degree than the dataset average, especially at very low budgets. This density bias decreases with budget but remains above one.

To test whether degree is actually the controlling factor, I designed a controlled-degree experiment on CIFAR100. It runs several different K/delta combinations that induce similar effective degrees. If configurations with similar degree behave similarly despite different K and delta, that would provide stronger evidence that the graph degree is the relevant invariant.
