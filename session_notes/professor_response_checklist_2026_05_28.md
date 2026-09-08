# Professor Response Checklist - 2026-05-28

This note maps every PDF comment to the concrete response before the next meeting.

## 1. Degree Definition

Comment: "is the degree" and "remind me how the degree is computed?"

What she is asking: define the degree used in density normalization and say whether it is the clipped graph degree or `sum_j K_ij`.

Status: addressed in the updated summary text.

Response:

- We used the graph after the ProbCover-style cutoff.
- `effective` degree is the number of nonzero neighbors after removing the diagonal.
- `weighted` degree is `sum_j K_ij` after removing the diagonal.
- The later alpha-wide experiments used `weighted` degree.
- The density diagnostics often report `effective` degree because it is easier to interpret as graph connectivity.

Needed action: cite the new selected-degree plots in the meeting.

## 2. Alpha Definition

Comment: "is this the alpha you use in the results graphs below?"

What she is asking: confirm that the formula alpha is the same as the plotted alpha.

Status: clarification only.

Response:

- Yes. The plotted alpha is the same total exponent in `D^{-alpha/2} K D^{-alpha/2}`.
- The `/2` appears because symmetric normalization splits the total exponent equally between the two endpoints.

Needed action: done in the updated summary text.

## 3. Alpha And Geometric Mean

Comment: "do I understand correctly that alpha is the actual power, and therefore alpha=0.5 is the geometric mean of the rank of the two edge-nodes?"

What she is asking: verify the interpretation of symmetric normalization.

Status: clarification plus one correction.

Response:

- It is degree, not rank, in the current implementation.
- For `alpha=1`, the denominator is the geometric mean of the two endpoint degrees.
- For `alpha=0.5`, the penalty is weaker: `(degree_i degree_j)^0.25`.

Needed action: use this exact explanation in the meeting.

## 4. Average Rank Per Budget

Comment: "can you compute the average rank for each budget?"

What she is asking: check whether selected points become less extreme in degree over time.

Status: partially addressed.

Existing output:

- `results_summaries/professor_notes_2026_05_28/selected_degree_by_budget.csv`
- `results_summaries/professor_notes_2026_05_28/selected_degree_by_budget_mean.csv`
- `visual_outputs/professor_notes_2026_05_28/*selected_degree_ratio_by_budget.png`

Important limitation: existing CSVs contain selected-point degrees and dataset degree summaries, but not the full per-point degree distribution. Therefore the current plot is a degree-ratio/rank-proxy, not an exact percentile rank.

Needed action: if she specifically wants percentile rank, run an additional offline rank computation over sampled anchor degrees or exact full degrees.

## 5. Delta Sensitivity And ProbCover Comparison

Comment: "in 100-400, check sensitivity to delta; compare to ProbCover."

What she is asking: test whether alpha gains depend on optimal delta, and add ProbCover to the same plots.

Status: addressed. The delta-sensitivity runs completed successfully and were aggregated.

Existing output:

- `results_summaries/professor_notes_2026_05_28/probcover_vs_maxherding_alpha.csv`
- `visual_outputs/professor_notes_2026_05_28/*probcover_vs_alpha.png`
- `results_summaries/professor_notes_2026_05_28/delta_sensitivity_by_episode.csv`
- `results_summaries/professor_notes_2026_05_28/delta_sensitivity_mean_by_episode.csv`
- `results_summaries/professor_notes_2026_05_28/delta_sensitivity_gain_vs_alpha0.csv`
- `visual_outputs/professor_notes_2026_05_28/delta_sensitivity/*`

Run verification:

- Expected runs: 120
- Completed runs with result arrays: 120
- Seeds per dataset/setting/delta/alpha: 3
- No OOM, timeout, cancellation, or traceback was found in the logs.

Main result:

- CIFAR10: alpha=0.5 often helps early, but the effect is unstable across delta values. The best final result is still alpha=0 with delta around 0.75 to 0.80.
- CIFAR100: alpha=0.5 helps at low budgets, especially 100 to 300 labels. Smaller deltas, 0.55 and 0.60, preserve the gain longer; larger deltas, 0.70 and 0.75, make alpha harmful later.
- The delta sensitivity supports the conclusion that density normalization is useful only in a budget- and graph-scale-dependent regime.

Needed action: discuss these results in the meeting and decide whether the next adaptive rule should control alpha, delta, or both.

## 6. "Meaning?" In Dynamic Alpha Section

What she is asking: a sentence was too vague.

Status: phrasing change.

Response: say that fixed alpha helped some early budgets, but fixed hand-written schedules did not beat alpha=0 at final budget.

Needed action: done in the updated summary text.

## 7. Single Sigma / Global K

Comment: "can you repeat this experiment with a single sigma?"

What she is asking: separate density normalization from local sigma.

Status: already addressed by existing global-sigma alpha experiments.

Evidence:

- The alpha-wide and alpha-gain plots include `global`.
- The new ProbCover comparison plots include global-sigma MaxHerding with alpha normalization.

Needed action: done in the response plots and checklist; optional to add one sentence in the final Word document.

## 8. Greedy Max Coverage Definition

Comment: "is this the definition of greedy max coverage?"

What she is asking: avoid an algorithmic overclaim.

Status: phrasing change.

Response: call it a greedy coverage-style selection rule, not exactly classical greedy max coverage.

Needed action: done in the updated summary text.

## 9. "Does Not Persist" Versus "Degrades Later Performance"

What she is asking: make the result more precise.

Status: phrasing change.

Response: say "the early gain often does not persist, and in several settings becomes negative at later budgets."

Needed action: done in the updated summary text.

## 10. Degree Versus Local Kernel

Comment: "These scores are not totally different..."

What she is asking: avoid separating degree and locality too strongly.

Status: interpretation change.

Response: degree and locality are related. Local sigma affects the kernel graph, and degree is the measurable graph statistic that best explains the behavior we observed.

Needed action: done in the updated summary text.

## 11. Average Degree And Std As Function Of Budget

Comment: asks for graph of average degree and standard deviation as a function of budget.

Status: selected-batch version done from existing data; remaining-pool/full-graph version still requires recomputation if requested.

Existing output:

- `visual_outputs/professor_notes_2026_05_28/*selected_degree_mean_by_budget.png`
- `visual_outputs/professor_notes_2026_05_28/*selected_degree_ratio_by_budget.png`

Needed action: clarify in the meeting whether the selected-batch version answers her question or whether she wants average degree/std over the entire remaining graph after each episode.

## 12. Switch To Random Or Mixture Strategy

What she is asking: connect degree collapse to strategy switching.

Status: conceptual addition.

Response: when average degree becomes very small, the coverage signal may be weak. This suggests switching to random or mixing strategies, related to Hacohen and Weinshall, NeurIPS 2023, SelectAL.

Needed action: add this to future work in the final polished document if we keep a dedicated future-work section.

## 13. Three-Phase Experiment Clarity

Comment: "isn't this what you did in the 3 phase experiments?"

What she is asking: distinguish fixed schedules from adaptive rules.

Status: clarification.

Response: the three-phase schedule was fixed by episode number. The proposed next step is adaptive because it would use measured graph statistics such as degree/rank.

Needed action: done in the updated summary text.

## 14. ProbCover Figure 9 / Delta Decreases With Budget

What she is asking: reconcile our degree heuristic with the observation that optimal delta decreases as budget increases.

Status: partially addressed by the delta-sensitivity experiment.

Response: decreasing delta can reduce redundant coverage at higher budgets and change which points remain useful after already-covered regions are removed. In the delta sweep, smaller deltas preserved the alpha benefit longer on CIFAR100, while larger deltas made alpha harmful later.

Needed action: discuss whether this motivates an adaptive delta/alpha rule rather than a fixed-alpha decay.

## 15. Positive Feedback

Comment: "Very thoughtful and well put."

Status: no action.

Response: keep the thesis-style takeaway, but update it with the refined degree/locality wording.
