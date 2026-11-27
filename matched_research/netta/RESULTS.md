# Peer-ranked AUM on official NPC

Peer-ranked AUM improved **all 15 symmetric-noise settings** by a mean **+0.48 percentage points**. Eight settings improved on all three seeds. At 50% symmetric noise, the gain averaged **+0.71 pp** across five budgets. At 1,600 labels the mean accuracy moved from **49.68% to 50.99%**.

Across the full grid it won **23 of 35** three-seed means and averaged **+0.18 pp**. Clean performance was flat on average. Cyclic asymmetric noise averaged **−0.07 pp** with four wins in 15 settings. This is a symmetric-noise result on CIFAR-100 BYOL.

| Noise | 400 | 800 | 1,200 | 1,600 | 2,000 | Mean | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | +0.03 | +0.05 | +0.10 | −0.19 | +0.04 | +0.00 | 4/5 |
| Symmetric 30% | +0.31 | +0.05 | +0.13 | +0.22 | +0.53 | +0.24 | 5/5 |
| Symmetric 40% | +0.90 | +0.48 | +0.45 | +0.08 | +0.47 | +0.48 | 5/5 |
| Symmetric 50% | +0.30 | +0.66 | +0.71 | +1.31 | +0.58 | +0.71 | 5/5 |
| Cyclic asymmetric 30% | +0.25 | −0.03 | −0.27 | −0.02 | +0.24 | +0.03 | 2/5 |
| Cyclic asymmetric 40% | −0.04 | −0.04 | −0.72 | −0.05 | −0.31 | −0.23 | 0/5 |
| Cyclic asymmetric 50% | +0.34 | +0.10 | −0.21 | −0.23 | −0.04 | −0.01 | 2/5 |

Every cell is candidate minus reference test accuracy in percentage points, averaged over paired seeds 71–73. Budgets within a seed share an acquisition trajectory. The peer filter retained about **6.9 more correct labels** on average under symmetric noise at the same retained training size, compared with **0.7 more** under cyclic asymmetric noise. [Budget and noise means](../results/npc/by_setting.csv) · [All 105 pairs and retention diagnostics](../results/npc/per_seed.csv)

## What changed

The reference is the pinned Netta NPC release with LowBudgetAUM and the full 500-epoch supervised linear classifier on BYOL. Each variant fit uses the saved reference acquisition trajectory, observed labels, initialization, optimizer and minibatch permutation stream. It changes the final training mask by adding an AUM midrank and a local label-agreement midrank, then filling the original quota within each observed class. All 105 identity replays matched the original model and test-prediction hashes. The exported masks independently passed the per-class quota check. [Full pseudocode and formulas](PROTOCOL.md) · [Verification code](collect.py)

Slurm array `32699969` completed all 21 tasks. The fit inputs, output masks and prediction hashes are archived under `/cs/labs/daphna/tomn/projects/cifar_matched_npc_20261127`.

In [Streamlit](http://127.0.0.1:18529), open **NPC · final AUM filter**. The grid shows every paired mean. Choose **Symmetric 50%** to see both models and all three seeds for each budget. Expand **NPC formulas and pseudocode** for the method.
