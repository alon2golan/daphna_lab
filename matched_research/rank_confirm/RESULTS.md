# Fresh seeds confirm the AUM-rank weight

The predeclared six-hop AUM-rank weight gained **+0.22 percentage points over raw six-hop votes** on seeds 74–76. It won **33 of 35** budget/noise means and 87 of 105 individual seed pairs, with one seed-pair tie. The exploratory seeds 71–73 gave +0.22 pp and 34/35 wins.

| Fresh noise family | Vs raw six-hop | Vs strongest raw depth per setting | Vs released NPC classifier |
|---|---:|---:|---:|
| Clean | +0.21 · 5/5 | −0.02 · 1/5 | +2.10 · 5/5 |
| Symmetric 30/40/50% | +0.25 · 15/15 | +0.18 · 13/15 | +4.02 · 15/15 |
| Cyclic 30/40/50% | +0.18 · 13/15 | +0.13 · 13/15 | +3.01 · 15/15 |
| All | +0.22 · 33/35 | +0.13 · 27/35 | +3.31 · 35/35 |

Cells show mean gain in percentage points and wins over budget/noise means. The strongest raw depth per setting is the best of 3–8 hops by mean accuracy after scoring. The best single raw depth overall was five hops. AUM-rank six-hop averaged +0.19 pp above it. Against raw six-hop, the fresh mean gain per seed was +0.18, +0.31 and +0.15 pp. Raw six-hop itself gained +3.10 pp over the released classifier, so the predictor swap explains most of the end-to-end gain. At 1,600 labels and 50% symmetric noise the means are 50.24% released NPC, 54.25% raw six-hop, and 54.47% AUM-rank six-hop.

The fresh run covers a single frozen CIFAR-100 BYOL bank, the same held-out test set, nested label budgets, and synthetic corruptions. It tests the new vote weight against stronger raw graph controls and compares the full predictor replacement against the released NPC classifier.

Official array `32702794` and graph array `32704757` each finished all 21 tasks. Parity job `32704756` matched all five old clean seed-71 raw and rank results before scoring the fresh grid. The collector verifies the per-case artifacts against the full official snapshots and records the source, geometry, and input identities. [Protocol and formulas](PROTOCOL.md) · [All paired scores](../results/rank_confirm/per_seed.csv) · [Budget and noise means](../results/rank_confirm/by_setting.csv) · [Provenance](../results/rank_confirm/provenance.json)
