# Two changes for noisy CIFAR-100 labels

**Backup coverage is the broader result.** It beat archived MaxHerding in mean graph test accuracy at every tested budget from 800 to 2,000 labels under symmetric and asymmetric 30%, 40% and 50% noise. **Square-root votes had the larger symmetric-noise gains.** They beat the strongest raw vote depth by 3.76–7.33 percentage points across 800–1,600 labels in ten fresh starts.

Both changes use the same annotation budget as their references. Positive values in the tables are improvements in percentage points.

The later comparisons add Netta NPC, Weighted NPC, MaxHerding+NAS, lab U-Herding and JS-10. They narrow the acquisition claim and add one useful VAST variant. [Strong-baseline follow-up](sota/RESULTS.md) · [Research scorecard](METHODS.md)

The earlier signed-audit and VAST studies traced mistakes back to individual source annotations. These experiments use the same idea at two earlier decisions. Backup coverage changes which images get queried. Square-root votes change how much influence each acquired annotation receives. [Previous lab results](../docs/NOISY_LABEL_RESEARCH_20261126.md)

## Backup coverage

For each pool image, let $m_1$ and $m_2$ be its two highest similarities to distinct queried images. MaxHerding sums $m_1$. Add half of $m_2$ and use the same greedy selection. With similarities 0.9 and 0.1, a new neighbor at 0.8 earns 0 under max coverage and 0.35 under backup coverage. [MaxHerding, Bae et al.](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11051.pdf)

If each queried label is independently unusable with probability $q$, the expected similarity to the best usable neighbor is

$$
(1-q)(m_1 + q m_2 + q^2 m_3 + \cdots).
$$

The objective keeps the first two terms up to a constant, at $q=0.5$. The weight 0.5 is fixed across all experiments and the selector never observes a noise rate. The formula models an unusable label as absent. Wrong labels can actively mislead a classifier, especially when nearby labels have systematic errors. The objective adapts a reliability idea from facility location to noisy queries. [Snyder and Daskin](https://doi.org/10.1287/trsc.1040.0107) · [Noise-aware active sampling](https://arxiv.org/abs/2504.04506)

For a candidate similarity $s$, the marginal score per pool image is $(1-\lambda)[s-m_1]_+ + \lambda[s-m_2]_+$. Here $\lambda=0.5$ and $[x]_+=\max(0,x)$. Both thresholds increase as queries accumulate, so the objective is monotone submodular and lazy greedy can use old marginal gains as upper bounds.

### Full Gaussian affinities

The two orders share MaxHerding's first 100 queries. Subsequent choices use the same archived 45,000 × 45,000 Gaussian matrix. The max-only implementation recovered the second archived batch exactly. The evaluation replays the full archived MaxHerding order and uses a six-hop graph classifier in both arms. Five paired seeds redraw synthetic noise on the two fixed orders. CIFAR-100N and clean each have one paired order and one fixed label vector.

| Annotations | 400 | 800 | 1,200 | 1,600 | 2,000 |
|---|---:|---:|---:|---:|---:|
| Symmetric 30% | −0.70 | +0.69 | +1.62 | +1.99 | +2.67 |
| Symmetric 40% | −0.95 | +0.90 | +2.04 | +3.02 | +3.70 |
| Symmetric 50% | −0.14 | +1.62 | +2.39 | +3.95 | **+4.72** |
| Asymmetric 30% | −0.15 | +2.15 | +1.41 | +1.94 | +2.37 |
| Asymmetric 40% | +0.18 | +1.93 | +0.97 | +1.06 | +1.19 |
| Asymmetric 50% | +0.48 | +1.63 | +0.81 | +1.14 | +0.62 |
| CIFAR-100N | +0.51 | +1.26 | +1.59 | +1.17 | **+2.14** |
| Clean | −0.64 | −0.15 | +0.17 | −0.26 | +0.04 |

The majority of seeds gained in 23 of the 24 synthetic cells at 800–2,000 labels. At symmetric 50% and 2,000 labels, mean accuracy rose from 40.28% to 45.00%, with all five seeds gaining. Backup coverage acquired 0.4 more incorrect labels on average in that cell. At the same budget, CIFAR-100N rose from 50.75% to 52.89% and the new order selected 43 fewer incorrect annotations. Under symmetric noise, cosine 1NN lost ground on average at each tested budget.

### Sparse affinities and fresh starts

The independent sparse confirmation re-generated the shared first 100 queries for seeds 30–34. The primary endpoint was the average three-hop test gain across 1,600 and 2,000 labels under symmetric 50%. It gained **3.81 pp**, with gains between 2.18 and 5.76 pp across all five seeds. At six hops and 800–2,000 labels, every symmetric noise cell gained 3.11–5.54 pp and all five seeds gained in each cell. At 2,000 labels, the share of unlabeled images with at least two supported annotations grew 7.03 pp and single-annotation coverage was essentially tied. The CIFAR-100N gains were +1.36 to +1.99 pp across these budgets. Asymmetric 50% was effectively flat at 800–1,200 and lost up to 0.81 pp at larger budgets. [Frozen protocol](sparse_backup/PROTOCOL.md) · [Seed counts](sparse_backup/results)

## Square-root votes

Sum all path weights $h_i$ from a given test image to each original annotation $i$. The raw graph vote gives annotation $i$ weight $h_i$. Give it $\sqrt{h_i}$ and apply the same class vote. With unit edge weights, 16 paths to one annotation give weight 4. Four annotations with one path apiece give the same total weight. The transform preserves the ordering of annotation weights and the set of reachable annotations. The direct comparison shares the same acquisitions and depth.

For normalized weights $w_i = h_i^\alpha / \sum_j h_j^\alpha$, the effective support is $N_{\mathrm{eff}}=1/\sum_i w_i^2$. For positive $h_i$, differentiating gives

$$
\frac{d}{d\alpha}\log N_{\mathrm{eff}}
=2\left(\mathbb E_{w(\alpha)}[\log h]-\mathbb E_{w(2\alpha)}[\log h]\right)\le 0
$$

for $\alpha \ge 0$, since the mean of $\log h$ increases with the tilt exponent. Lowering $\alpha$ from 1 to 0.5 increases effective support. For fixed weights and independent, bounded annotation-level vote margins, the Hoeffding concentration bound uses $\sum_i w_i^2$ as its variance proxy. The expected margin can also change when weak paths come from other classes. At 50% superclass-cycle noise, each clean class sends all of its errors to the same rival. A more even vote can reinforce that rival. [Prior channel study](../docs/NOISY_LABEL_RESEARCH_20261126.md) · [Adaptive Diffusions](https://arxiv.org/abs/1804.02081) · [Label Propagation for Deep Semi-Supervised Learning](https://openaccess.thecvf.com/content_CVPR_2019/html/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.html)

The one-component change is applying a concave transform *after* aggregating paths by annotation. The pilot tested depth three, with a binary one-vote-per-reached-annotation control. The follow-up froze strengths 1, 0.5 and 0 and evaluated square root at depths three and six. One vote per reachable annotation performed poorly at six hops. On the confirmatory fresh starts at 1,600 labels, the mean six-hop effective support grew from 28.7 to 142.8.

### Ten fresh acquisition starts

The table compares six-hop square-root votes with the strongest mean raw depth among three, four, six and eight in each cell. Choosing the raw depth after scoring gives the reference the benefit of depth selection. Both classifiers see identical noisy annotations in each seed.

| Annotations | 400 | 800 | 1,200 | 1,600 |
|---|---:|---:|---:|---:|
| Symmetric 30% | +1.55 | +4.03 | +4.23 | +3.76 |
| Symmetric 40% | +1.67 | +4.79 | +5.56 | +5.43 |
| Symmetric 50% | +1.91 | +5.55 | +6.98 | **+7.33** |
| Asymmetric 30% | −0.03 | +1.26 | +1.58 | +1.80 |
| Asymmetric 40% | −0.13 | +0.45 | +0.71 | +1.17 |
| Asymmetric 50% | −0.46 | −0.41 | −0.59 | −0.44 |
| CIFAR-100N | +0.06 | +1.75 | +1.46 | +1.48 |
| Clean | −1.42 | −0.79 | −1.10 | −1.61 |

Every seed gained in every symmetric cell, even when the raw depth was chosen separately for every seed. At 1,600 labels and symmetric 50%, the result was 39.13% → 46.46% against the best mean raw depth. Every seed also gained on CIFAR-100N at 800, 1,200 and 1,600 labels. On the fixed archived MaxHerding order, square-root votes gained 2.47–7.77 pp under symmetric 30–50% at 800–2,000 labels, against the same strongest-mean-depth comparison. [Counts and acquisition orders](confirmation/results) · [CIFAR-100N](https://arxiv.org/abs/2110.12088)

The separately measured combination reached 50.76% against 40.28% for archived MaxHerding with raw six-hop votes at symmetric 50% and 2,000 labels. Backup coverage alone reached 45.00% and square-root votes alone 48.05%. At asymmetric 50% and the same budget, the combination was essentially tied with the reference.

### Fixed corrupted images

Re-use the ten fresh acquisition orders and corrupt the same images at 50% three ways. Choose each wrong class uniformly, use a fixed class cycle spread across the class list, or cycle only within superclasses. The global cycle is sampled once per seed. At the matched six-hop depth, the mean gain across 800, 1,200 and 1,600 labels was **+6.84 pp** with uniform wrong classes, **−0.36 pp** with the global cycle and **−0.21 pp** with the within-superclass cycle. Every seed gained with uniform wrong classes at every budget. Single-rival errors erased the gain even when the rival class was assigned without using semantic groups. [Matched counts](confirmation/collision_results)

### Protected VAST transfer

Reconstruct the archived seed-0 Itai protected teacher at five updates, including the original sender and receiver masks. Apply square root per annotation and rescale the revised class evidence to the original per-image total. This leaves the uniform prior's share, protected rows, admission and confidence weights fixed. The archive was reconstructed within $5.43\times 10^{-7}$ at 800 and 1,200 labels. The VAST graph tracks per-annotation evidence in blocks, so storage grows with block size rather than the total label budget. It adds graph work compared with propagating class totals alone. [VAST](https://arxiv.org/abs/2609.14451) · [Teacher counts](vast/results/teacher)

On the full unlabeled pool, symmetric 30% gains were 5.40 and 4.44 pp at 800 and 1,200 labels. Symmetric 50% gains were 7.34 and 9.35 pp. For the original admitted rows the corresponding gains were 3.73, 2.59, 3.02 and 9.35 pp. At asymmetric 50% the full-pool change was below 0.2 pp at both budgets.

The original 300-epoch MLP used these frozen targets with matching membership, confidence, optimizer steps and validation schedule. The 800-label symmetric-50 reference re-fit on the same GPU reproduced the archived test predictions exactly. Test accuracy fell by 0.35–1.62 pp in all six verified student cases.

| Labels and noise | Itai protected VAST | Square-root targets | Change (pp) |
|---|---:|---:|---:|
| 800, clean | 56.55% | 55.65% | −0.90 |
| 800, symmetric 30% | 49.48% | 48.01% | −1.47 |
| 800, symmetric 50% | 38.48% | 38.13% | −0.35 |
| 800, asymmetric 50% | 29.29% | 28.76% | −0.53 |
| 1,200, symmetric 30% | 52.77% | 51.75% | −1.02 |
| 1,200, symmetric 50% | 43.93% | 42.31% | −1.62 |

The revised distribution had higher entropy in all eight teacher cells. Mean probability on the true class fell 8.12–13.50 percentage points across their admitted sets, even where top-1 accuracy rose. Negative log likelihood was mixed. [Probability diagnostic](vast/results/target_probabilities.json)

## Other tests

- **Adaptive depth.** Selecting depth by an annotation-weighted standardized margin lost to the best fixed depth in all 32 cells, by 1.87 pp on average. Equalizing each annotation's total influence also lost in all 32. [Five paired replays](pilots/adaptive_depth/results)
- **Non-backtracking walks.** Removing immediate reversals gave about +0.17 pp over raw depth three across the pilot grid. Increasing to raw depth four was the stronger change. [Pilot](pilots/nonbacktracking) · [Compression follow-up](pilots/powers)

## Reproduce

The dashboard reads checked-in, per-seed aggregate counts. The graph runs share a frozen CIFAR-100 BYOL representation, 45,000 eligible training rows, and one 10,000-image test set. Budgets within a seed are nested. Synthetic asymmetric noise follows the fixed within-superclass cycle. The VAST experiments use one historical allocation and initialization seed. The tests cover multiple budgets and noise settings on this dataset and representation.

```bash
python -m pip install 'streamlit>=1.65' 'plotly>=6' 'pandas>=2' 'numpy>=1.26'
python -m streamlit run graph_methods_study/streamlit_app.py --server.port 18527
```

Replay scripts expect the frozen lab inputs and the archived NumPy, SciPy and, for VAST, CUDA environment. The Slurm scripts give the cluster paths. The sparse and power pilots used `active_audit_study/active_audit.py` from revision `da533b8623a605307dd6dcb7f486b1acebc20c4a`. The new depth and joint confirmation jobs used the helper from `90840eb6cd9ec487b646a5f563ce25faf6228a72`. The VAST archive pins source revision `9e70f229c6eefba74ec887b2d9fe8589227792ef`.

| Slurm job | Experiment |
|---|---|
| 32670019, 32670173, 32670110 | Non-backtracking, source weights and adaptive depth pilots |
| 32670162, 32670268 | Sparse backup coverage pilot and fresh-seed confirmation |
| 32670229 | Full Gaussian backup trajectories |
| 32679287 | Ten fresh graph starts and five paired noise draws on full Gaussian trajectories |
| 32679975 | Matched corruption channels on the ten fresh graph starts |
| 32679334, 32679570 | Protected VAST teacher and student transfer |
| 32679990 | VAST target probability diagnostic |
