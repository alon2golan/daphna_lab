# Strong-baseline follow-up on CIFAR-100 BYOL

**Runner-up capping is the strongest candidate for noisy graph votes.** It improved all 24 fresh noise/budget means over raw NPC and Weighted NPC, and 16/24 over the strongest available raw comparison. Square-root votes help more selectively. The VAST winner swap is the best new learned-model lead under symmetric 50% noise.

## Comparisons

- The acquisition experiments use 45,000 eligible images, 100 shared first queries, batches of 100, and 10,000 test images. All arms observe the same noisy annotation vector within a condition and seed. The acquisition comparison uses the same graph votes across orders.
- The archive replays seed 0 for clean, symmetric and superclass-cycle asymmetric 20%, 30% and 50%. It includes exact MaxHerding, the lab's fixed-bandwidth U-Herding, one-step JS, and both observed- and soft-broadcast JS-10. Published U-Herding also adapts bandwidth. [Archive protocol](archive/PROTOCOL.md) · [U-Herding](https://arxiv.org/abs/2412.20644)
- Netta's ProbCover, NPC, Weighted NPC, and Gaussian MaxHerding+NAS use the same seed-0 labels and an additional CIFAR-100N replay. The NPC rules use the official fixed-radius decision logic, the lab's pinned LowBudgetAUM implementation, and the existing 0.50 BYOL radius. AUM is used at acquisition, and the comparison also scores all labels and the AUM-filtered subset on separate common graph heads. The published study used its own feature, radius, noise and classifier protocol. [Matched protocol](netta/PROTOCOL.md) · [Netta's paper](https://arxiv.org/abs/2504.04506) · [Author code](https://github.com/nettashafir/active_learning_with_label_noise/tree/733651bb0943a96392895103eb2ffc4e99a7c7d1)
- Three fresh NPC noise seeds, 51–53, cover symmetric and asymmetric 30%, 40% and 50%. Both NPC policies and their backup variants acquire against their own revealed labels. The warm order and detector settings are shared. Budgets are nested within each seed.

## Runner-up cap on NPC acquisitions

Use $w_i(x)=\min(h_i(x),h_{(2)}(x))$ where $h_{(2)}$ is the second largest path weight among distinct original annotations. Keep the weight when only one annotation is reachable. A 16-path annotation with a four-path runner-up gets weight 4. The rule changes at most one source per image. If the largest weight is $a$ and the other weights sum to $S$ with squared sum $V$, effective support is $(a+S)^2/(a^2+V)$. While $a$ is the largest, $aS\ge V$, so effective support increases as $a$ falls to the runner-up.

Each fresh order was regenerated and verified against its saved hash. At 800–2,000 labels, six-hop capping improved all 24 three-seed means for both NPC methods. Average gains were **+1.10 pp** for NPC and **+1.15 pp** for Weighted NPC. With the best raw three- or six-hop head, either NPC policy and either AUM mask chosen per cell and seed after scoring, capped NPC gained **+0.65 pp** and won **16/24** mean cells.

| Fresh noise | Cap vs raw NPC, six hops | Cap vs raw Weighted NPC, six hops | Cap on NPC vs best raw |
|---|---:|---:|---:|
| Symmetric 30% | +1.49 | +1.24 | +0.91 |
| Symmetric 40% | +1.81 | +2.08 | +1.45 |
| Symmetric 50% | +2.32 | +2.48 | +1.86 |
| Asymmetric 30% | +0.50 | +0.55 | +0.11 |
| Asymmetric 40% | +0.25 | +0.43 | −0.46 |
| Asymmetric 50% | +0.25 | +0.11 | +0.04 |

At symmetric 50% and 800 labels, capped Weighted NPC reached **44.04% vs 40.43%**. The seed-0 cap gained +0.70 pp on CIFAR-100N and changed clean NPC accuracy by −0.04 pp over 800–2,000 labels. With hard AUM filtering, the fresh mean changed by +0.09 pp for NPC and −0.07 pp for Weighted NPC. The strongest evidence is for graph voting that uses all acquired labels. [Seed-0 replay and fresh confirmation](capped_votes/results)

## Square-root votes on NPC acquisitions

Replace each annotation's total path weight \(h_i(x)\) with \(\sqrt{h_i(x)}\) before voting for its observed class. This lowers concentration across independent sources. It shares the acquisition order and path support of the raw vote. [Effective support derivation](../README.md#square-root-votes)

With all acquired labels and ordinary NPC, three-hop compression gained **1.97 pp** over raw three-hop votes across the 800–2,000-label grid. It gained in **23/24** budget/noise cells by three-seed mean and **69/72** individual pairs. Weighted NPC gained **2.25 pp** at three hops, with positive means in 24/24 cells. Raw six hops is a stronger reference.

| Fresh NPC noise | Square root vs raw at six hops | Square root vs best available raw head |
|---|---:|---:|
| Symmetric 30% | +0.69 | +0.11 |
| Symmetric 40% | +1.33 | +0.97 |
| Symmetric 50% | +2.44 | +1.98 |
| Asymmetric 30% | +0.50 | +0.11 |
| Asymmetric 40% | +0.18 | −0.53 |
| Asymmetric 50% | −0.27 | −0.48 |

Values average matched changes at 800, 1,200, 1,600 and 2,000 labels. The last column uses the better raw count among three and six hops, all or AUM-filtered labels, and NPC or Weighted NPC at each seed, budget and condition after scoring. Against that reference the mean gain was **+0.36 pp**, with positive three-seed means in **12/24** cells. Symmetric 40% and 50% gained at all four budgets. At symmetric 50% the gains were +3.52, +2.22, +1.43 and +0.74 pp. On the archived NPC and CIFAR-100N labels, square-root six-hop votes gained **0.53 pp** on average compared with raw six hops. Hard AUM filtering reversed the fresh-seed six-hop effect to −0.48 pp. [Replays](npc_backup/results)

## Winner swap for protected VAST

Let \(a=\arg\max p_\mathrm{raw}\) and \(b=\arg\max p_\mathrm{sqrt}\). Swap \(p_\mathrm{raw}[a]\) with \(p_\mathrm{raw}[b]\). This keeps the original target probability multiset, confidence and entropy while assigning its highest value to the corrected class. There were no argmax mismatches among admitted rows. Soft square-root targets made predictions less sharp and lost all six trained matches. The swap changes only class assignment for the two probabilities. The same seed-0 protected teacher gate and 300-epoch student use the same acquired labels, admitted rows, loss weights and step counts. [Verified fits](../vast/results/winner_student)

| Noise | Labels | Protected VAST | Full square root | Winner swap |
|---|---:|---:|---:|---:|
| Clean | 800 | 56.55% | 55.65% | 56.47% |
| Symmetric 30% | 800 | 49.48% | 48.01% | 49.87% |
| Symmetric 50% | 800 | 38.48% | 38.13% | **38.99%** |
| Asymmetric 50% | 800 | 29.29% | 28.76% | 29.28% |
| Symmetric 30% | 1,200 | 52.77% | 51.75% | 52.67% |
| Symmetric 50% | 1,200 | 43.93% | 42.31% | **45.32%** |

The gains of **0.51** and **1.39 pp** under symmetric 50% both came from one archived seed. The swap is a narrow learned-model candidate.

## Acquisition methods

**Backup coverage.** Sum \(m_1 + 0.5m_2\) over the pool, where \(m_k\) is similarity to the \(k\)-th closest acquired image. A second neighbor covers failures of the first. The original paired comparisons gained in 24/24 synthetic cells over exact MaxHerding at 800–2,000 labels. [Theory and original controls](../README.md#backup-coverage)

In the seed-0 archive, backup raw-six-hop voting gained **0.83 pp** over the strongest of MaxHerding, fixed-bandwidth U-Herding, JS-1 and either JS-10 per cell, with **26/35** wins. Against the best available Netta arm under symmetric 50%, backup lagged by **7.02 pp** at 800–2,000 labels. At 2,000 labels, backup put the common six-hop classifier at 46.55%, compared with 42.12% for MaxHerding and 53.39% for NPC. [Archived comparison](archive/RESULTS.md) · [Netta counts](netta/results/scores.json)

The shared 300-epoch MLP used ordered noisy annotations and the same validation split and test images as the archived MaxHerding, lab U-Herding and JS arms. Across five noise conditions and budgets of 800, 1,200 and 2,000, backup gained 0.07 pp over MaxHerding and won 7/15 seed-0 cells. It lost **0.69 pp** against the strongest archived MLP per cell, winning 4/15. Under symmetric 50% at 2,000 labels it reached **50.32% vs 50.13%** for the strongest archived MLP. Clean losses against that reference were 0.94–2.09 pp. [Matched MLP fits](backup_mlp/results)

**NPC + backup.** For radius coverage, add half-credit when a pool image has a second predicted-clean covering center. On the archived noise draw this lost **0.83 pp** vs NPC and **0.86 pp** vs Weighted NPC with all labels and raw six hops. The fresh noise draws gave **+0.34** and **+0.14 pp**, respectively. With the AUM-filtered head, fresh differences were **−0.14** and **−0.49 pp**. Larger geometric redundancy does not consistently help after NPC reopens flagged neighborhoods. [Paired replays](npc_backup/results)

**Source fragility.** Keep Gaussian coverage and change just \(\omega(x)\). Sample an anchor in proportion to its path weight and compute the probability that deleting it removes the current winner's strict lead. Use one plus that probability as the target weight. This improved MaxHerding on several noisy seed-0 settings and trailed the strongest NPC variants. [Scores](fragility/results)

**Grouped backup.** Use two distinct feature groups as coverage sources. Two-group coverage at 1,600 labels rose from 70.80% to 83.82%. In three paired starts under cluster-correlated noise, graph accuracy still fell 0.48 pp on average versus ordinary backup. The rate-matched IID control lost 1.70 pp. [Protocol and results](grouped_backup/RESULTS.md)

**Compress only AUM-flagged votes.** Leave accepted labels at full weight and square-root the path total of flagged labels. This was near the hard AUM filter at six hops, about +0.19 pp for NPC. A fixed half-weight for flagged labels gained **1.01 pp** against hard NPC filtering and **0.31 pp** against NPC with all labels. The half-weight control held up better than selective source compression and still trailed the strongest raw envelope by 0.53 pp. This diagnostic is one draw. [Scores](conditional_votes/results)

## Run record

| Slurm | Scope |
|---|---|
| 32691453 | Exact lab U-Herding, JS and MaxHerding order replay |
| 32691465, 32691492 | Netta parity and NPC, Weighted NPC, MaxHerding+NAS |
| 32691496 | Grouped backup and clustered-noise control |
| 32691499 | Six matched VAST winner-swap fits |
| 32691506 | Source-fragility weights on exact Gaussian acquisitions |
| 32698667 | NPC backup plus matched raw and square-root heads |
| 32698678 | Shared 300-epoch classifier transfer for backup coverage |
| 32698705 | Selective compression and half-weight control |
| 32698715 | Cap at the second source |
| 32698723 | Confirm the cap on three fresh NPC noise draws |

[All tested ideas](../METHODS.md) · [Earlier graph study](../README.md)
