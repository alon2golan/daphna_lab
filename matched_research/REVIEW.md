## What held up

| Change | Exact reference | Evidence |
|---|---|---|
| Peer-ranked AUM | Released NPC and its 500-epoch linear classifier on BYOL | +0.48 pp and 15/15 symmetric settings. Cyclic asymmetric mean −0.07 pp. |
| Signed label verification | Six equal-cost controls including leave-one-out, released LowBudgetAUM rank and buying fresh clean labels | Fresh strongest-control comparison gives +1.13 pp, 12/12 symmetric and +0.40 pp, 4/4 CIFAR-100N. Cyclic −0.34 pp, 5/12. |
| Raw six-hop predictor | Released NPC linear classifier with the same acquired labels and final AUM keep set | +3.06 pp and 34/35 settings. A predictor replacement using a standard graph method. |
| AUM-rank votes | Raw six-hop votes on the same official AUM keep set | +0.22 pp and 33/35 fresh settings across both noise families and clean data. +0.13 pp against the strongest raw depth per setting. |
| Runner-up cap | Raw six-hop predictor on the same official AUM keep set | −0.74 pp and 4/35 wins. |
| Square-root votes | Raw six-hop predictor on the same official AUM keep set | −0.65 pp and 2/35 wins. |
| VAST rank alignment | Protected VAST with the same student, admitted rows and confidence weights | 3 wins, 4 ties and 1 loss on one archived seed. Symmetric 50% gained +0.55 and +1.56 pp. |

The earlier cap claim used a lab NPC acquisition replay and a raw graph comparator. The official-runs replay holds Netta's label set fixed. Comparing the cap only with the linear classifier credits the cap for switching to graph prediction.

For peer-ranked AUM, replacing the rank sum by neighbor agreement alone gives +0.51 pp on symmetric noise and −1.00 pp on cyclic noise. The AUM term recovers +0.93 pp on cyclic noise against that peer-only control.

Fresh AUM-rank votes gained +3.31 pp against the released NPC classifier in 35/35 settings. Raw graph voting accounted for +3.10 pp. The AUM weight supplies a small improvement on top of the predictor swap.

Choose a method tab for its comparator, full budget/noise grid, and formulas. **Pilot inventory** lists the earlier trials and their limits.

The acquisition trials in the archive include exact MaxHerding and the lab's fixed-bandwidth U-Herding. They provide no broadly winning new trained acquisition variant. The published adaptive-bandwidth U-Herding protocol is outside those comparisons. [U-Herding paper](https://arxiv.org/abs/2412.20644)
