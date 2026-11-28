# Official NPC labels with graph prediction

All 105 pairs use the released NPC trajectory and final AUM mask from Slurm array `32699969`. The only module replaced in the full-pipeline comparison is the predictor. The raw and capped graph arms share the same graph, source labels and walk counts.

| Comparison | Mean change | Wins over 35 settings |
|---|---:|---:|
| Raw six-hop vs released linear NPC | +3.06 pp | 34 |
| Capped six-hop vs released linear NPC | +2.32 pp | 33 |
| Capped vs raw six-hop | −0.74 pp | 4 |

Raw votes improved all 15 symmetric settings by a mean +3.71 pp and all 15 cyclic asymmetric settings by +2.87 pp. Runner-up capping lost to raw votes in 30 settings and tied one. At 1,600 labels and symmetric 50%, the three means are 49.68% released NPC, 52.80% raw graph, and 52.14% capped graph.

The published classifier comparison measures a graph predictor replacement. The raw-graph comparison isolates capping. Both comparisons are in the full grid. [All seed pairs](../results/npc_cap/per_seed.csv) · [Budget/noise means](../results/npc_cap/by_setting.csv) · [Formulas and exact controls](PROTOCOL.md)

Slurm jobs `32702725` and `32702726` built the 50,000-image graph and verified all 21 episode groups. Evidence is archived under `/cs/labs/daphna/tomn/projects/cifar_npc_cap_exact_20261128`.

## One-weight follow-up on the same seeds

Slurm array `32702761` reproduced every raw six-hop score before changing the source weight. Square-root votes averaged **−0.65 pp** and won **2/35** settings. AUM-rank votes averaged **+0.22 pp** and won **34/35**. By family, AUM-rank votes gained **+0.17 pp** on clean, **+0.26 pp** on symmetric, and **+0.20 pp** on cyclic asymmetric settings. Its best setting was symmetric 50% at 400 labels, **+0.64 pp**. This is the exploratory grid used to generate the fresh-seed confirmation. [All pairs](../results/npc_graph_weights/per_seed.csv) · [Budget/noise means](../results/npc_graph_weights/by_setting.csv)

The 105 fresh pairs on seeds 74–76 confirmed **+0.22 pp and 33/35** settings against raw six-hop. Raw-depth comparisons and the released classifier appear alongside the matched raw six-hop control. [Fresh confirmation](../rank_confirm/RESULTS.md)
