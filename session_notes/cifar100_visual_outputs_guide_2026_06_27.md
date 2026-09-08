# CIFAR100 Visual Outputs Guide - 2026-06-27

This note explains the CIFAR100 degree-analysis figures and how they relate to the final research question: whether MaxHerding behavior is explained more by graph degree/density than by the specific local-sigma K.

Main visual directories:

`visual_outputs/cifar100_degree_evidence_2026_06_27`

`visual_outputs/invariant_candidates_2026_06_27`

`visual_outputs/cifar100_degree_controlled_2026_06_27`

## 1. Terms Used in the Figures

### Dataset Effective Degree

For a fixed kernel setting, each point has a degree equal to the total amount of connection it has to other points. In the cutoff setting, this is close to counting how many neighbors the point has inside the cutoff radius.

Dataset effective degree is the average of this quantity over the CIFAR100 training set.

This is not an active-learning result by itself. It is a statistic of the graph created by a specific setting, such as global sigma or local sigma with a particular K.

### Remaining-Pool Effective Degree

Remaining-pool effective degree is measured only on the unlabeled pool left after some number of labels has already been selected.

This statistic is useful because active learning changes the pool. If MaxHerding first selects dense representative points, then the remaining unlabeled pool should become less dense over time. That tells us something about what the algorithm is removing from the pool.

### Selected-to-Dataset Degree Ratio

This compares the average degree of the selected batch to the average degree of the dataset.

A ratio of 1 means the selected points have average density.

A ratio above 1 means the selected points come from denser-than-average regions.

This is important because it directly tests whether MaxHerding is biased toward high-degree points.

### K, Delta, and Degree

K is not the degree. K is the number of neighbors used to compute the local sigma of each point.

Delta is the cutoff threshold.

K and delta together determine the graph degree. Larger K usually makes local sigmas smoother and more global-like. Larger delta directly allows more neighbors through the cutoff. Therefore, both can increase the dataset effective degree.

## 2. Retrospective K-Sweep Figures

These figures do not come from experiments that directly controlled degree. They reuse the completed MaxHerding K-sweep experiments.

For each setting, we already had:

- The AL accuracy curve.
- The K value.
- The resulting local sigmas.
- The induced graph degree.

So the question here is retrospective:

If we take the experiments we already ran, does accuracy line up better with K or with the graph degree induced by K?

### `cifar100_accuracy_vs_effective_degree_by_budget.png`

What it shows:

- X axis is dataset effective degree.
- Y axis is accuracy.
- Each point corresponds to one completed setting, usually one K value.
- Each curve is a different labeling budget.

How the X values were measured:

- For each K, we used the local sigmas induced by that K.
- We built the corresponding graph/cutoff relation.
- We measured the average number/weight of neighbors per dataset point.
- That average is the dataset effective degree.

How the Y values were measured:

- The Y value is the accuracy from the same AL experiment and same K.
- For example, if the point has degree 24.3 and budget 100, that point is the accuracy at 100 labels for the K that induced degree 24.3.

What the shaded band means:

- It is a post-hoc range, not an experimental constraint.
- It marks the approximate dataset-degree region where the best settings from the previous K sweep tended to appear.
- It was measured after the experiments by looking at the degrees of the best-performing settings.

What we can learn:

- The graph degree is a useful way to organize the K-sweep results.
- Very small degree is bad.
- Very large degree is not consistently better.
- The useful degree region changes with budget.

Important caveat:

- This graph is correlational. It does not prove degree causes the accuracy difference, because K and degree changed together in the original sweep.

### `cifar100_top3_setting_degree_by_budget.png`

What it shows:

- For every budget, all K-sweep settings are ranked by accuracy.
- Rank 1 is the best setting at that budget.
- Rank 2 and rank 3 are the next best.
- The Y axis is the dataset effective degree of those top-ranked settings.

What it is meant to answer:

Do the best settings have a stable degree, even when the actual K changes?

What we can learn:

- Early budgets tend to include higher-degree settings among the top results.
- Later budgets shift toward lower/moderate degrees.
- This suggests that the useful graph scale is budget-dependent.

Important caveat:

- This again comes from the old K sweep. It ranks existing results; it does not directly control degree.

### `cifar100_selected_to_dataset_degree_ratio_best_setting.png`

What it shows:

- At each budget, we first choose the best-performing setting at that budget.
- Then we compute the degree of the batch selected by that setting.
- The plotted value is selected-batch degree divided by dataset degree.

So this is not one fixed K across all budgets. It is the per-budget best setting.

What we can learn:

- The selected points are much denser than the dataset average at the start.
- The ratio decreases as budget increases.
- Even later, selected points remain denser than average.

Why this matters:

- It directly supports the idea that point degree/density is central to MaxHerding selection behavior.

### `cifar100_best_setting_degree_and_remaining_degree.png`

What it shows:

- Blue curve: dataset effective degree of the per-budget best setting.
- Orange curve: remaining-pool effective degree under that same setting.
- Labels above points show which K was best at each budget.

Why remaining-pool degree is interesting:

- It measures what is left after the algorithm selected labels.
- If the remaining pool becomes less dense, it suggests the algorithm selected and removed denser points earlier.
- This is another way to see the density bias of the selection process.

Why it is not enough alone:

- It does not directly show selected-point degree.
- It should be read together with the selected-to-dataset degree ratio.

### `invariant_candidates_2026_06_27/cifar100_best_setting_remaining_degree.png`

What it shows:

- Blue curve: remaining-pool effective degree at the per-budget best local setting.
- Red curve: accuracy of that same best setting.

What we can learn:

- Accuracy increases while remaining-pool degree generally decreases.
- This is consistent with MaxHerding selecting dense/representative regions early, leaving a sparser pool.
- It is useful as a pool-dynamics diagnostic, not as a direct accuracy predictor.

## 3. Controlled-Degree Experiment

The controlled-degree experiment was designed because the retrospective K sweep was not enough.

In the original K sweep, K and degree changed together. To test whether degree itself matters, we chose different `(K, delta)` combinations that target similar effective-degree ranges.

The controlled experiment used CIFAR100 MaxHerding with 1NN and cutoff, over seeds 3, 4, and 5.

Target degree bands:

- Low degree: about 10-13 effective neighbors.
- Moderate degree: about 18-23 effective neighbors.
- High degree: about 30-36 effective neighbors.

The degree values were estimated before running the AL experiments by sampling CIFAR100 training points and counting/averaging their graph neighbors under each `(K, delta)` setting.

### `cifar100_degree_controlled_band_curves.png`

What it shows:

- Accuracy over budget for the low, moderate, and high target-degree bands.
- Each band averages several configurations and seeds.

What it is meant to answer:

If we group settings by induced graph degree, do the accuracy curves separate by degree band?

What we can learn:

- High degree is worse later in the AL run.
- Low and moderate degree are close and often competitive.
- This supports degree as an important explanatory factor, but not a simple monotone one.

### `cifar100_degree_controlled_fixed_k1440_curves.png`

What it shows:

- Same K: 1440.
- Three different delta values.
- These produce low, moderate, and high effective degree.

Why this is clearer:

- K is fixed, so the main thing changing is the cutoff-induced degree.
- This isolates the degree effect better than comparing many unrelated K values.

What we can learn:

- The moderate-degree curve is best for most of the budget range.
- High degree is better than very low degree early, but it is not best later.
- This supports a budget-dependent useful degree regime.

### `cifar100_degree_controlled_band_panels.png`

What it shows:

- The controlled experiment split into three panels.
- Each panel contains only the configurations inside one target degree band.

Why this is useful:

- It replaces the unreadable all-config plot.
- It lets us see whether configurations inside the same degree band behave similarly.

What we can learn:

- Curves inside a band are relatively close, especially compared to the bad very-small-degree cases from the older K sweep.
- There are still differences within a band, so degree is important but not the only factor.

### Supporting Controlled-Degree Figures

The following figures exist but should not be the main focus:

- `cifar100_degree_controlled_final_accuracy_vs_degree.png`
- `cifar100_degree_controlled_final_accuracy_by_config.png`
- `cifar100_degree_controlled_all_config_curves.png`
- `cifar100_degree_controlled_best_degree_per_budget.png`

Reason:

- Final accuracy at 1000 labels is arbitrary for our current question.
- The research question is about behavior across budget, not only the final endpoint.
- The all-config curve is too dense to be a good presentation figure.

## 4. Effective Degree Grid

### `cifar100_delta_k_effective_degree_grid.png`

What it shows:

- For each `(K, delta)` pair, it shows the estimated dataset effective degree.

How it was measured:

- Sample CIFAR100 training points.
- For each sampled point, count how many points are connected under that `(K, delta)` setting.
- Average those counts.

What it is useful for:

- Designing controlled-degree experiments.
- Finding different `(K, delta)` pairs that produce similar graph degrees.

What it is not:

- It is not an AL result.
- It is not meant to be a main conclusion figure.

## 5. Cluster Separability

### `cifar100_cluster_separability_by_k.png`

This is a feature-space diagnostic, not an AL experiment.

What we did:

- Took the CIFAR100 SSL feature vectors.
- Ran k-means clustering with different numbers of clusters.
- Compared cluster geometry and label purity.

Inter-centroid distance:

- Distance between cluster centers.
- Larger means clusters are farther apart.

Intra-cluster distance:

- Average distance from points to their assigned cluster center.
- Smaller means clusters are tighter.

Separability ratio:

- Inter-centroid distance divided by intra-cluster distance.
- Larger means clusters are more separated relative to their internal spread.

True-label purity:

- For each cluster, find the most common true CIFAR100 class.
- Purity is the fraction of all points that belong to their cluster's majority class.
- Higher purity means clusters align better with true labels.

What we can learn:

- CIFAR100 features are structured but not cleanly separated.
- Increasing the number of clusters improves purity, but purity is still far from perfect.
- This suggests that a useful degree threshold may depend on dataset geometry, not only on the number of classes.

How strongly to use this:

- This is useful background evidence.
- It is not central enough to be a main presentation figure unless Daphna specifically wants separability analysis.

## 6. Main Presentation Recommendation

The clearest figures to show are:

- `cifar100_selected_to_dataset_degree_ratio_best_setting.png`
- `invariant_candidates_2026_06_27/cifar100_best_setting_remaining_degree.png`
- `cifar100_accuracy_vs_effective_degree_by_budget.png`
- `cifar100_degree_controlled_band_curves.png`
- `cifar100_degree_controlled_fixed_k1440_curves.png`
- `cifar100_degree_controlled_band_panels.png`

The story these figures support:

- MaxHerding strongly prefers high-degree points early.
- The unlabeled pool becomes less dense as labels are selected.
- K matters mainly because it changes the graph degree.
- Degree is not monotonically good.
- Very high degree can hurt later.
- A useful low-to-moderate degree regime appears more stable than any single K.

## 7. Current Finishing-Touch Interpretation

The main finishing result is not that local sigma alone gives a large accuracy improvement.

The stronger result is that the behavior of MaxHerding is strongly tied to graph degree and density. Local sigma is one mechanism for changing that degree, but the degree itself appears to be the more useful explanatory quantity.

This reframes the research:

Instead of asking only "which K is best?", we should ask "what graph degree regime does MaxHerding need at each budget?"

The current evidence suggests:

- Early budgets benefit from selecting representative dense regions.
- Later budgets should avoid overly dense graphs because they blur too much of the geometry.
- The best setting is budget-dependent.
- A future adaptive method could choose K or delta to maintain a useful degree regime as the unlabeled pool changes.
