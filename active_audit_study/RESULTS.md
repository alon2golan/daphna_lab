# Counterfactual footprint audits

Five matched CIFAR-100 BYOL replays tested what to do after acquiring 400 or 800 noisy labels. Each policy had 40 or 80 requests to a perfect verifier. Policies could verify existing labels or buy the next geometry-selected labels clean. All arms reused the same noisy acquisition prefix within each seed. The paired test set had 10,000 images and was excluded from acquisition and audit ranking. The learner was inductive three-hop graph voting, with cosine 1NN as a secondary readout.

Our candidate ranks noisy labels by the train-pool predictions that would change if the annotation were replaced with its most supported leave-one-out alternative. It discounts overlap between the predicted effects of labels chosen for the batch. The suspicion baseline ranks annotations by the same leave-one-out disagreement score and requests the same number of clean answers.

## Synthetic corruption

| Condition | Initial labels | No audit | Fresh clean | Random audit | LOO audit | Footprint, independent | Footprint, greedy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 400 | 53.60 | 54.14 | 53.60 | 53.60 | 53.60 | 53.60 |
| Symmetric 20% | 400 | 43.80 | 44.92 | 44.80 | 47.41 | 46.66 | 46.66 |
| Symmetric 50% | 400 | 27.06 | 28.97 | 29.88 | 32.78 | 32.19 | 32.25 |
| Asymmetric 20% | 400 | 43.42 | 44.48 | 44.56 | 46.60 | 46.17 | 46.25 |
| Asymmetric 50% | 400 | 27.73 | 29.56 | 30.33 | 31.78 | 31.73 | 31.79 |
| Clean | 800 | 56.27 | 56.40 | 56.27 | 56.27 | 56.27 | 56.27 |
| Symmetric 20% | 800 | 47.39 | 48.18 | 48.34 | 51.35 | 50.83 | 50.91 |
| Symmetric 50% | 800 | 30.02 | 31.55 | 33.23 | 36.12 | 37.74 | **37.81** |
| Asymmetric 20% | 800 | 46.72 | 47.35 | 47.63 | 50.39 | 50.14 | 50.18 |
| Asymmetric 50% | 800 | 29.27 | 30.55 | 31.96 | 33.95 | 34.91 | **35.14** |

Entries are mean held-out test accuracy in percent over five seeds. Greedy footprint audited 10 fewer wrong labels on average than LOO in the 800-label symmetric 50% condition (46.6 versus 56.6 corrected) while improving test accuracy in all five seeds by 0.93, 1.80, 2.74, 2.27, and 0.72 percentage points. In the corresponding asymmetric setting the five changes were +0.46, +1.90, +0.71, +2.68, and +0.21 points. The overlap discount added 0.06 and 0.23 points over independent footprint ranking. Most of the benefit came from the footprint itself.

LOO auditing led at 20% corruption and on the cosine 1NN classifier. The greedy footprint's average advantage over LOO at 400 labels was −0.53 points in symmetric 50% noise and +0.01 points in asymmetric 50% noise. This limits the observed effect to a sufficiently populated graph under heavy noise. Across all noisy synthetic cells, graph voting improved by 4.37 points with LOO audits, 4.45 with greedy footprint audits, 1.91 with random audits, and 1.27 with equal-cost clean expansion relative to the initial labels alone.

The five repeats used seeded greedy sparse coverage on the exact historical 13NN graph. Synthetic seed 0 exactly reproduced the lab's archived observed-label vectors. The evaluator used unweighted walks and excluded the 5,000 historical validation rows. The experiment tests this graph learner, the fixed feature bank, and retrospective perfect verification. Historical full Gaussian MaxHerding and MLP accuracies use different acquisition and training.

## Replay provenance

Slurm preparation job `32667916` and array `32667917` completed successfully. The collector checked all 300 synthetic evaluations, the ordered request budgets, range and finiteness of scores, and identical clean controls. The synthetic source SHA256 is `c22dfd6d8f5e66a3525cc2cab15bd7efd750862a0d44d958b4d3ab9d00dec388` and the prepared graph SHA256 is `486509610f2947676c2bf5dfc7d3e6950e79805580290f9ca0d54790058e682b`.

## Human annotations

The CIFAR-100N follow-up varied acquisition seeds while holding the human annotation vector fixed. That vector disagreed with the canonical labels on 20,100 of 50,000 training images. Its clean-label companion matched the BYOL targets in every row. All five acquisition orders and the full prepared geometry matched the synthetic study exactly.

| Initial labels | No audit | Fresh clean | Random audit | LOO audit | Footprint, independent | Footprint, greedy |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 43.36 | 44.50 | 44.19 | **46.77** | 46.42 | 46.42 |
| 800 | 45.67 | 46.31 | 46.63 | **50.05** | 49.49 | 49.58 |

LOO audits beat the greedy footprint on held-out graph accuracy in all five repeats at each budget. With 800 initial labels, LOO added 4.38 points over the initial labels and 3.73 points over buying the next 80 labels clean. The same choices on cosine 1NN gave 39.49%, 40.26%, and 43.60% for no audit, fresh clean, and LOO audit. The synthetic footprint gain was specific to severe corruption under the graph learner. The comparative benefit of verifying existing labels over expanding transferred to the human annotations.

Slurm array `32668078` completed successfully and produced all 60 uniquely matched policy evaluations. The human replay source SHA256 is `f01a95358801019634840ba54bdc0cd51c29dfb68ebe15a721825ad01dad9ab6`. The verified human annotation archive SHA256 is `bd2d80409754d420292d622e15e25248ba21e37d27429efac49c8da723f44394`.
