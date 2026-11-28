# What AUM adds to peer agreement

Peer-only ranking improved the released NPC classifier by **0.51 percentage points** on average under symmetric noise, with gains in 13 of 15 settings. The AUM and peer rank fusion gained **0.48 pp** and improved all 15. Under cyclic noise, peer-only lost **1.00 pp** and fell in all 15 settings. Adding the AUM rank recovered **0.93 pp**, improving every cyclic setting relative to peer-only. The fused cyclic mean was 0.07 pp below released LowBudgetAUM.

| Noise | Peer-only vs LowBudgetAUM | Fusion vs peer-only | Fusion vs LowBudgetAUM |
|---|---:|---:|---:|
| Clean | −0.07 | +0.08 | +0.00 |
| Symmetric 30% | +0.25 | 0.00 | +0.24 |
| Symmetric 40% | +0.66 | −0.18 | +0.48 |
| Symmetric 50% | +0.64 | +0.07 | +0.71 |
| Cyclic 30% | −0.78 | +0.82 | +0.03 |
| Cyclic 40% | −1.32 | +1.09 | −0.23 |
| Cyclic 50% | −0.88 | +0.87 | −0.01 |
| All settings | −0.22 | +0.39 | +0.18 |

Each entry averages five budgets and three paired seeds, in percentage points. At symmetric 40% and 400 labels, the three-seed mean was 34.92% for LowBudgetAUM, 36.15% for peer-only, and 35.82% for fusion. At cyclic 40% and 1,200 labels, peer-only lost 2.19 pp, and fusion recovered 1.47 pp. [All 35 setting means](results/by_setting.csv) · [All 105 seed pairs](results/per_seed.csv)

At the same retained training size, peer-only kept 8.8 more correct labels per fit under symmetric noise than LowBudgetAUM. Fusion kept 6.9 more. Under cyclic noise, peer-only kept 4.6 fewer correct labels per fit and replaced an average of 122.5 training rows. Fusion kept 0.7 more correct labels than LowBudgetAUM and replaced 55.9 rows. These results distinguish a ranking signal that works under uniform wrong-class flips from a fusion that is less sensitive to the cyclic channel. Both candidates retain LowBudgetAUM for NPC acquisition and the final per-class quotas.

The canary (`32702724`) and full array (`32702750`) completed on RTX 2080 Ti GPUs using the archived runtime. All 105 identity replays matched the original fitted-model and test-prediction hashes. The collector checked the snapshots and class quotas for all three arms. The comparison uses one frozen BYOL bank and test split, with nested budgets and lab-generated corruptions. [Protocol](PROTOCOL.md) · [Run provenance](results/provenance.json) · [Checksums](results/SHA256SUMS)
