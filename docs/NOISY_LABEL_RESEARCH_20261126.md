# Auditing influential noisy labels and testing the sources of pseudo-label confidence

## Result

**A re-query is more useful when the resulting prediction changes agree with evidence from the other annotations.** On CIFAR-100 BYOL, a graph audit score that rewards those changes and debits harmful ones improved held-out accuracy over leave-one-out disagreement in all 15 matched comparisons at 800 noisy labels plus 80 clean requests. The mean gain was **3.43 pp** at symmetric 50%, **0.78 pp** at asymmetric 50%, and **0.40 pp** on CIFAR-100N. The 80 answers were simulated from the clean training labels. Each comparison used the same 10,000 held-out images.

## How earlier experiments narrowed the search

- In the latest 2,160-fit VAST study, mutable propagation with Itai's admission rule averaged −0.06 pp against protected propagation across 720 matched cells. Balancing the top 80% averaged −0.68 pp. The strongest ten-step JS broadcast gained +0.69 pp over the better herding control under symmetric 50%, with one acquisition seed. These results put more weight on the quality of the original observations and pseudo-targets than on a wholesale change to propagation or acquisition. [Prior experiment chat](codex://threads/01a0f6cf-f712-79e3-bf47-13f0f6ac63cf)
- The lab's earlier noisy-AL analysis found that direct AUM filtering cost 3.66 and 4.31 pp in the warm linear comparison at 400 and 800 labels. Hard checkpoint and temporal relabeling made more harmful changes than correct repairs under asymmetric 50%. The Bayesian study found substantial gains from propagating uncertain targets and identified correlated pseudo-evidence as an open problem. [Earlier noisy-AL analyses](../noisy_al_report_448/README.md) · [Bayesian coverage results](https://github.com/alon2golan/daphna_lab/blob/bayesian_coverage/docs/bayesian_coverage_RESULTS.md)
- The first active-audit experiment tried counting the predictions affected by a proposed correction. Adding coverage diversity helped little beyond the footprint itself. At 800 noisy labels the greedy footprint beat leave-one-out disagreement by 1.69 and 1.19 pp under symmetric and asymmetric 50%, respectively. It trailed by 0.47 pp on CIFAR-100N. The unsigned score's failure on human noise motivated checking each proposed change against the other annotations. [First audit study](../active_audit_study/RESULTS.md)

## Signed graph audit score

For each acquired annotation, compute a leave-one-out margin between its observed class and the strongest alternative. Move that annotation's graph votes to the alternative and recompute the winning classes of the affected unqueried training points. For each change, estimate its effect as the difference in probability assigned to the new and old winning classes by the other annotations. Sum the signed differences and multiply by the positive leave-one-out margin. This is a deterministic score on a frozen graph. It uses the noisy answers already acquired and the unlabeled training features.

At fixed graph depth, the exact contribution from each annotation is a column of walk counts. Replacing a label moves one column between two classes. For affected row \(v\), the peer distribution is the original class vote vector with this column removed, normalized by its remaining mass. If the mass is zero, that row contributes zero. A negative peer probability difference reduces the audit score.

Existing Impact Sampling already prioritizes relabels by counterfactual prediction change. CINCER uses influence to find incompatible examples. The specific idea here is to score a hypothetical graph prediction change with the votes of the remaining annotations and keep its sign. NAS motivates attention to noisy representatives and coverage, and is a complementary new-query method in a different annotation-cost setting. [Impact Sampling](https://homes.cs.washington.edu/~mausam/papers/aaai16b.pdf) · [CINCER](https://arxiv.org/abs/2106.03922) · [Noise-Aware Active Sampling](https://arxiv.org/abs/2504.04506)

### New acquisition seeds 5–9

Mean test accuracy on the three-hop graph classifier. The no-additional-label reference used 800 noisy calls. Each other arm spent 80 more clean calls. All settings use the same scoring rule.

| Training annotations | 800 only | Clean expansion | Random audits | Leave-one-out audits | Unsigned footprint | Signed audit score | Signed vs leave-one-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| Symmetric 50% | 29.60% | 31.18% | 32.93% | 35.73% | 38.06% | **39.16%** | **+3.43 pp** |
| Asymmetric 50% | 29.42% | 30.80% | 32.64% | 33.75% | **34.98%** | 34.53% | **+0.78 pp** |
| CIFAR-100N | 45.52% | 46.34% | 46.40% | 49.95% | 49.38% | **50.35%** | **+0.40 pp** |

On human labels, the signed score corrected 53.8 of 80 labels on average, compared with leave-one-out's 58.0. The signed score's graph gains ranged from +0.27 to +0.57 pp. Its cosine 1NN change relative to leave-one-out was +0.87 pp for symmetric 50%, +0.03 pp for asymmetric 50%, and −0.23 pp on CIFAR-100N. The primary human graph gain exceeded the prespecified +0.25 pp threshold. The unsigned footprint still led under asymmetric 50%. This follow-up used one frozen representation, one holdout test set, and one fixed human annotation vector. Signed-score performance at 400 labels and under mild noise is unknown. [Protocol](../active_audit_study/SIGNED_PROTOCOL.md) · [Per-seed scores](../active_audit_study/signed_metrics.csv) · [CIFAR-N](https://arxiv.org/abs/2110.12088)

## Separate result · Why source redundancy helps at the top of the VAST ranking

For each unlabeled prediction, trace the class evidence to its original acquired annotations through the protected VAST masks. Subtract the strongest single annotation supporting the predicted class and recompute its signed margin against the nearest rival. Multiple paths from the same annotation count as a single source. This is exact conditional on the fixed masks. All 20 archive replays reconstructed the historical probabilities within 6.56e−7.

Replacing the full VAST gate with the deletion-margin ranking failed. Mean pseudo-label precision fell by 2.60 pp at symmetric 50%, and the two larger budgets admitted the entire pool in both arms. At fixed 10% coverage and matched predicted-class counts, source stability improved precision by **5.07 pp** at symmetric 50% and **2.83 pp** at symmetric 20% over four budgets. It lost **3.69 pp** at asymmetric 50%. The archive used a single seed and acquired one example per clean class per round. The new channel comparison used five label-free acquisition trajectories. [VAST](https://arxiv.org/abs/2609.14451) · [Exact replay data](../source_support_study/replay_results.json)

### Hold the corrupted images fixed and change their wrong labels

The controlled follow-up used the same acquired examples, same corrupted rows, same 50% corruption rate, and same protected CkNN graph. Within each channel, the compared rankings kept predicted class counts identical. The three noise channels chose a wrong label uniformly, via a fixed random cycle across the 100 classes, or via a fixed cycle within each superclass. The first and second conditions can both produce mistakes across superclasses. The second and third send each clean class to exactly one wrong class.

| Wrong labels at 50%, 800 acquired labels | Shortlist precision change vs confidence | Same direction in paired seeds |
|---|---:|---:|
| Uniform off-diagonal | **+12.73 pp** | 5/5 gains |
| Fixed cycle across classes | +0.84 pp | 3/5 gains |
| Fixed cycle within superclass | −1.14 pp | 4/5 losses |

Under a simple same-class model with independently generated errors, two annotations agree on a wrong label with probability equal to the sum of the squared off-diagonal entries of the noise transition row. At 50% noise on 100 classes, the conditional error among agreeing pairs is 1% for uniform wrong labels and 50% for a single systematic wrong label. The controlled replay supports the importance of this collision mechanism. Real neighborhoods are mixed, and the channel changes can interact with the geometry. [Higher-Order Consensus](https://proceedings.mlr.press/v139/zhu21e.html) · [Controlled replay results](../source_support_study/channel_results.json)

### Train the same VAST student on equal-size shortlists

Both shortlist arms used the same 300 epochs, validation split, probabilities, loss weights, acquired labels, and optimizer-step count. The only intervention was which unlabeled examples entered distillation. The two symmetric cells were selected after seeing shortlist precision. The clean and asymmetric cells are controls. The original full gate is a historical reference with different exposure.

| Labels and noise | Confidence shortlist | Source-stability shortlist | Matched change | Archived full gate |
|---|---:|---:|---:|---:|
| 500, symmetric 50% | 31.90% | 32.15% | +0.25 pp | 32.90% |
| 1,000, symmetric 50% | 40.23% | **41.31%** | **+1.08 pp** | 40.03% |
| 1,000, asymmetric 50% | 29.64% | 29.79% | +0.15 pp | 28.94% |
| 1,000, clean | 53.67% | 54.01% | +0.34 pp | 57.15% |

At 1,000 labels under symmetric 50%, the source shortlist used 16,200 optimizer steps compared with 135,000 in the original full gate. Its test gain over the full gate is a single-seed observation. Under clean labels the original full-gate model was 3.14 pp better than the source shortlist. [Verified student scores](../source_support_study/student_results.json)

## Reproduction and scope

The study reused the frozen CIFAR-100 BYOL bank. It used three-hop inductive graph voting and cosine 1NN for active audits, and the pinned CkNN propagation and 256-unit VAST MLP for the shortlist experiments. Seeds vary acquisition and synthetic noise. The audit branches share the exact frozen graph, and all paid requests return a perfectly accurate label in this replay. The stronger human comparison is between clean verification and clean expansion at equal per-request cost. If verified answers cost more in practice, the cost tradeoff changes.

| Slurm run | Verified output |
|---|---|
| 32667916, 32667917 | Prepared graph and 300 synthetic audit evaluations |
| 32668078 | 60 CIFAR-100N audit evaluations |
| 32668199 | 105 signed-audit follow-up evaluations |
| 32668091 | 20 corrected exact VAST replay cells |
| 32668187 | 15 noise-channel cells, five matched seeds |
| 32668024, 32668193 | Four final matched MLP pairs, 300 epochs each |

An independent review found round-off could perturb an exact tie after removing an annotation. The correction computes the margin directly from excess evidence. All 20 replay cells were regenerated. The two symmetric MLP pairs were rerun after shortlist membership changed. The clean and asymmetric MLP fits were retained after verifying their ordered training memberships matched the corrected selection exactly. The aggregate files and run instructions are linked here. [Audit code](../active_audit_study/README.md) · [VAST replay and student code](../source_support_study/README.md)
