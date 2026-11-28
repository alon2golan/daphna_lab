# Signed label verification on fresh seeds

Signed auditing improved the held-out three-hop graph in **21 of 32** settings against the strongest equal-cost control in each setting. It gained **+1.13 percentage points in all 12 symmetric settings** and **+0.40 pp in all four CIFAR-100N settings**. The strongest control is the best five-seed mean among the other six policies that spent the same extra 10% clean-query budget.

| Annotation type | Vs strongest control | Vs leave-one-out | Vs released LowBudgetAUM rank |
|---|---:|---:|---:|
| Symmetric 30/40/50% | +1.13 · 12/12 | +2.82 · 12/12 | +4.77 · 12/12 |
| Cyclic 30/40/50% | −0.34 · 5/12 | +1.19 · 12/12 | +3.69 · 12/12 |
| CIFAR-100N | +0.40 · 4/4 | +1.24 · 4/4 | +2.36 · 4/4 |
| Clean | −0.31 · 0/4 | 0.00 · 4 ties | 0.00 · 4 ties |

Entries give the mean change in percentage points and wins over budget/noise means. Individual paired runs won all 60 symmetric comparisons against the strongest control for their setting and all 20 CIFAR-100N comparisons. At 1,600 labels and 50% symmetric noise, the strongest control was greedy unsigned footprint. Its mean was 46.30% and signed auditing reached 47.34%. At the same budget, CIFAR-100N moved from 54.30% to 54.77%.

The ordering changes which labels get corrected. In 49 of 60 symmetric pairs and all 20 CIFAR-100N pairs, signed auditing scored higher than AUM while correcting fewer labels. The CIFAR-100N advantage reversed with cosine 1NN, where the signed method averaged −0.32 pp against the strongest same-cost control. The results support a graph-prediction claim on this frozen CIFAR-100 BYOL bank with perfect verification answers. Under cyclic noise the greedy unsigned policy often performs better. Under clean data, spending the budget on new labels performs better.

All five fresh seeds of Slurm array `32702727` completed. The paired test uses the same 45,000 candidate images, 10,000 held-out images, initial labels and graph for each verification rule. The AUM ranking control runs the released detector and shares that graph. The original NPC pipeline was tested separately in the final-filter and graph-predictor experiments. [Protocol](PROTOCOL.md) · [All 1,280 evaluations](results/metrics.csv) · [Paired setting summaries](results/comparisons.csv) · [Source and data provenance](results/provenance.json)
