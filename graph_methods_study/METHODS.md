## 26–27 November experiment inventory

The runner-up cap is the strongest graph candidate. It improved all 24 fresh noise/budget means against raw six-hop NPC and Weighted NPC. The VAST winner swap is a smaller learned-model lead. Evidence covers one representation and test split.

We matched label budgets and test heads, replayed fixed noise channels, and checked transfer to the original trained models. The acquisition comparisons included MaxHerding, ProbCover, Netta’s NPC variants, MaxHerding+NAS, lab U-Herding and both JS-10 variants.

| Method | Change | Finding |
|---|---|---|
| Runner-up cap | Limit the strongest annotation to the second strongest weight | +1.10 pp vs six-hop NPC in 24/24 fresh cells. +0.65 pp vs the strongest raw comparison in 16/24. |
| Square-root votes | Sum graph paths by original annotation and square-root each weight | Clear gains under symmetric noise, including on NPC acquisitions. Hard AUM filtering reduces the gain. |
| Backup coverage | Add half the second-closest similarity | Clear graph gains over MaxHerding. Netta's NPC and the strongest shared MLP references were stronger. |
| Soft AUM votes | Keep flagged annotations at half weight, or compress their paths alone | Half weight beat hard filtering more often. It trailed the strongest raw envelope. |
| NPC + backup | Add half-credit for a second covering center that passes AUM | Mixed across noise draws. Seed-0 results were weaker than ordinary NPC. |
| Square-root VAST targets | Apply source compression to the full soft distribution | Teacher labels improved under symmetric noise. Student lost all six matches. |
| VAST winner swap | Give the original winner’s probability to the compressed winner by swapping the two entries | Two symmetric-50 gains. Near flat in clean and asymmetric controls. |
| Source fragility | Weight each target by one plus its loss-of-lead probability after deleting one source | Improves on MaxHerding in the seed-0 noisy replay. Trails the strongest NPC arms. |
| Grouped backup | Count the closest two different feature groups | More group coverage. Lower accuracy than ordinary backup in the clustered-noise pilot. |
| Adaptive depth | Pick depth by a source-standardized margin | Lost to the best fixed depth in 32/32 cells. |
| Equal source mass | Normalize the total contribution from each original label | Lost to the best fixed depth in 32/32 cells. |
| Binary votes | Give each reached annotation one vote | Lost substantial accuracy at six hops. |
| Non-backtracking walks | Remove immediate two-edge returns | Small effect. Fixed depth four was stronger. |
| Signed graph audits | Rank potential corrections by their downstream peer support | Strong under a separate budget for verified labels. |
| Unsigned and diverse impact audits | Count or diversify the predictions touched by a potential correction | Gains in synthetic pilots. The unsigned footprint lost to leave-one-out on CIFAR-100N before signed auditing. |
| Source-stability shortlist | Rank VAST pseudo-labels by margin after deleting the strongest source | Helped a small shortlist in symmetric noise. Poorer at full coverage and under heavy asymmetric noise. |

The benchmark uses one representation and one fixed test split. Comparisons share the displayed classifier, except where the two classifier selectors specify different rules.
