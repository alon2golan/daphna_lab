# Professor Notes Addressed - Change Summary

Source document preserved: `Research Project Summary.docx`

Updated document created: `Research Project Summary - notes addressed.docx`

The updated Word document does not explicitly label edits as professor-note responses. This file is the separate mapping from each note to the corresponding document change.

## 1. Degree Definition

Comment addressed: clarify what "degree" means and whether it is the clipped graph degree or the weighted sum over the kernel.

Document change:

- Section 6 now defines degree as the graph degree after the same ProbCover-style cutoff used in the experiment.
- It separates effective degree from weighted degree.
- Effective degree is the number of nonzero neighbors after removing the diagonal.
- Weighted degree is the weighted-degree formula from Section 1.6, computed after removing the diagonal.
- The document avoids the ugly inline `sum_j K_ij` text and refers back to the already formatted formula.
- It now states that effective-degree and weighted-degree normalization gave very similar results, so the main effect appears to be penalizing high-connectivity points rather than the exact degree definition.

## 2. Alpha Definition

Comment addressed: confirm whether the alpha in the formula is the same alpha used in the result graphs.

Document change:

- Section 6 now states that the plotted alpha is the same total exponent alpha in the symmetric normalization formula.
- It explains that the formula contains `alpha/2` because the density penalty is split equally between the two edge endpoints.

## 3. Alpha And Geometric Mean Interpretation

Comment addressed: clarify whether alpha=0.5 means a geometric mean of the endpoint ranks.

Document change:

- Section 6 now states that the current implementation uses degree, not percentile rank.
- It explains that alpha=1 divides an edge by the geometric mean of the endpoint degrees.
- It explains that alpha=0.5 is weaker, with denominator `(degree_i degree_j)^0.25`.

## 4. Average Rank / Degree By Budget

Comment addressed: compute average rank by budget, or a statistic that shows whether selected points become less extreme over time.

Document change:

- Section 5 now explains that selected degree divided by dataset-average degree is a rank-like proxy, not an exact percentile rank.
- It states that exact percentile rank would require the full per-point degree distribution for every configuration.
- Added Figure 3 and Figure 4.

Added figures:

- Figure 3: selected-point effective-degree mean and within-batch standard deviation by budget for CIFAR10.
- Figure 4: selected-point effective-degree mean and within-batch standard deviation by budget for CIFAR100.

## 5. Figure A/B Standard Deviation Issue

Problem fixed: the earlier version claimed the figures showed standard deviation, but the plots only made the mean visually obvious.

Document change:

- The degree-by-budget plots were regenerated as two-panel figures.
- The top panel shows mean selected effective degree.
- The bottom panel shows within-batch standard deviation.

## 6. Delta Sensitivity

Comment addressed: test whether the density-normalization benefit depends on delta in the budget range where alpha helps.

Document change:

- Section 7 now includes a new subsection, "Delta Sensitivity And ProbCover Comparison."
- It summarizes the delta sweep:
- CIFAR10 deltas: 0.65, 0.70, 0.75, 0.80, 0.85.
- CIFAR100 deltas: 0.55, 0.60, 0.65, 0.70, 0.75.
- Settings: global sigma and best local-K setting.
- Alphas: 0 and 0.5.
- Seeds: three seeds per configuration.

Added figures:

- Figure 13: CIFAR10 alpha=0.5 gain over alpha=0 across deltas for local-sigma K=3500.
- Figure 14: CIFAR100 alpha=0.5 gain over alpha=0 across deltas for local-sigma K=1440.

Main conclusion added:

- Delta changes the size of the density-normalization gain.
- The qualitative trend is stable: density normalization can help at very low budgets, then weakens or becomes harmful later.
- This is clearest on CIFAR100.

## 7. ProbCover Comparison

Comment addressed: compare performance to ProbCover in the same plots.

Document change:

- Section 7 now explains why the ProbCover comparison is included.
- It says the plots compare MaxHerding with different density-normalization strengths against ProbCover under the same low-budget x-axis and classifier setup.
- It clarifies that the goal is to check whether density-normalized MaxHerding improves over a binary coverage baseline or mainly reproduces a similar coverage effect with a smoother kernel.

Added figures:

- Figure 11: CIFAR10 MaxHerding density normalization compared to ProbCover for both global sigma and local-sigma K=3500.
- Figure 12: CIFAR100 MaxHerding density normalization compared to ProbCover for both global sigma and local-sigma K=1440.

Additional fix:

- The comparison is now symmetric across datasets: each dataset shows both global sigma and the best local-sigma setting.

## 8. Single Sigma / Global-K Experiment

Comment addressed: repeat the decay experiment with a single global sigma.

Document change:

- Section 8 now states that the decay schedules were run both with the best local-sigma setting and with a single global sigma.
- Added Figure 17 and Figure 18 for the global-sigma decay schedule gain/loss.
- The existing decay figures are now explicitly labeled as local-sigma K=3500 for CIFAR10 and local-sigma K=1440 for CIFAR100.

Added figures:

- Figure 17: global-sigma decay schedule gain/loss for CIFAR10.
- Figure 18: global-sigma decay schedule gain/loss for CIFAR100.

## 9. Greedy Max Coverage Definition

Comment addressed: avoid implying that MaxHerding with cutoff is exactly classical greedy max coverage.

Document change:

- Section 1.8 now says the cutoff makes MaxHerding closer to a greedy coverage-style rule, but not exactly classical greedy max coverage.
- It explains that ProbCover uses binary coverage counts, while MaxHerding still uses weighted kernel similarities and its herding objective.

## 10. "Does Not Persist" Versus "Degrades Later Performance"

Comment addressed: make the result more precise.

Document change:

- Section 9 now says the density-normalization benefit often does not persist and, in several settings, becomes negative at later budgets.

## 11. Degree Versus Local Kernel

Comment addressed: avoid treating degree and local kernel as unrelated.

Document change:

- Section 9 now says degree and locality are closely related because both are affected by local density.
- It frames degree as the clearest measured statistic of the graph induced by local sigma.

## 12. Dynamic Alpha Meaning

Comment addressed: clarify the meaning of the dynamic-alpha/decay discussion.

Document change:

- Section 8 now defines each decay schedule explicitly.
- Early-only alpha uses the chosen alpha for the first three active-learning rounds and then sets alpha to 0.
- Linear decay starts from the chosen alpha and decreases it linearly to 0 over the first five rounds.
- Three-phase decay uses the chosen alpha for three rounds, half alpha for the next three rounds, and then alpha=0.
- Section 8 also explains that these are fixed schedules by episode number, not adaptive rules based on selected-set density, graph degree, or rank-like statistics.

## 13. Three-Phase Experiment Clarity

Comment addressed: clarify whether the proposed adaptive idea is the same as the three-phase experiment.

Document change:

- Section 11.3 now explicitly says the three-phase schedule is not enough by itself because it does not check whether graph degree is too small, selected points are still unusually dense, or the coverage signal has weakened.

## 14. Average Degree And Standard Deviation By Budget

Comment addressed: add graph of average degree and standard deviation as a function of budget.

Document change:

- Figure 3 and Figure 4 now show both mean selected effective degree and within-batch standard deviation across budget.

Limitation:

- These figures summarize selected points by budget. They are not a full recomputation of the remaining-pool graph degree after every episode.

## 15. Degree-Based Switching / Random Mixture

Comment addressed: mention that degree could indicate when the method stops being useful and should switch to random or a mixture strategy.

Document change:

- Section 12.2 now adds a future direction about switching strategy when graph degree becomes uniformly very small.
- It mentions switching to random selection or mixing MaxHerding with random/diversity sampling.
- It connects this to SelectAL.

## 16. ProbCover Figure 9 / Delta Decreases With Budget

Comment addressed: relate the proposed heuristic to the ProbCover observation that useful delta may decrease as budget increases.

Document change:

- Section 10 now explains that delta may also need to be budget-dependent.
- It connects decreasing delta to the idea that, after central regions are covered, a smaller threshold can reduce redundant coverage and change which points remain useful.
- It states that the new delta sweep supports this interaction, especially on CIFAR100.
