# Grouped backup coverage pilot

Frozen protocol for three seeds 70–72 at 800 and 1,600 total annotations on the existing CIFAR-100 BYOL geometry. Every trajectory starts with the same seed-specific 100 binary coverage queries. Use batches of 100 and the prior sparse exact RBF affinity support. Candidate rows, target eligibility at each batch start, and tie-break rule follow the original sparse backup pilot.

Compare maximum-only coverage, the established top-two objective with backup weight 0.5, grouped top-two with weight 0.5, and the same grouped objective using shuffled group assignments. The group definition is the nearest of 512 cosine landmarks picked by reverse-13NN coverage with seed 764. Fit the partition once from the eligible training features without labels or test inputs. The shuffle permutes group assignments with seed 765 and preserves their sizes. For each demand row, a group contributes its maximum known affinity. The rule rewards only the two strongest distinct groups. Singleton groups recover ordinary top-two and one group recovers max-only.

Under an independent all-or-none failure per group, the exact expected usable similarity is a discounted sum of ranked group maxima. The pilot uses its two-term truncation with the same fixed coefficient as ordinary backup coverage. The monotone submodularity argument follows by integrating the number of eligible hit groups capped at two at every affinity threshold. Grouped coverage does not intrinsically correct wrong labels or deterministic cyclic class shifts.

Synthetic IID and cyclic noise share exactly the same uniform sample mask with 50% corruptions across the eligible pool. For cluster-correlated corruption, an independent feature-based partition uses 256 reverse-13NN landmarks with seed 188. Sample one independent Bernoulli(0.5) flip indicator per stress group. Affected rows have independently drawn uniform wrong destinations, using the same per-image draw as the IID arm. Match its exact realized error count in a separate uniformly sampled row mask. Use clean and CIFAR-100N as controls. Each noise vector is fixed across all acquisition arms within a seed.

The primary endpoint is paired held-out six-hop graph accuracy grouped minus ordinary top-two in cluster-correlated conditions, averaged over the two budgets within a seed. Report separately versus the shuffled groups and in rate-matched IID noise. Report clean, cyclic and human comparisons and six-hop support counts. Graph and cosine 1NN use identical selected annotations. This pilot shares one held-out test set and one human annotation vector. Report the three seedwise differences without a statistical significance claim.

References

- [Noisy-annotator active sampling](https://arxiv.org/html/2504.04506#S3.SS2)
- [MaxHerding](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11051.pdf)
- [Backup facility assignment](https://doi.org/10.1287/trsc.1040.0107)
- [Facility location under correlated disruptions](https://doi.org/10.1016/j.trb.2009.09.004)
