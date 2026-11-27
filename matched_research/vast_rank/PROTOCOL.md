# Protected VAST rank alignment pilot

Use the depth-five protected VAST archive under `cifar100_balanced_depths_20261123/study`. Test the eight cells formed by 800 and 1,200 anchors with clean, symmetric 30%, symmetric 50%, and superclass-cycle asymmetric 50% annotations. This is an archived seed-0 study with the archive's class-balanced acquisition split. The student's 5,000 clean validation labels were already part of the reference checkpoint-selection protocol. The rank policy itself uses only acquired annotations.

Five seeded folds use acquisition positions. For each held-out fold, rebuild exact CK initialization and the five protected recurrences using only other-fold anchors. Compute raw and square-root-source probabilities without supplying that fold's labels to either predictor. Match the top zero, one, or three square-root ranks by sequentially swapping the corresponding original ordered probability values into position. In a tie the stable ranks use class ID order. Choose the depth with the lowest pooled held-out observed-label cross-entropy, breaking ties in favor of the smaller depth. The fit-free phase writes all per-anchor scores and the decision before fitting starts. It does not load validation or test labels or train canonical labels.

For two row-wise permutations $p,r$ of the same positive probability vector, let $\Delta_y=-\log p_y+\log r_y$. Then $\sum_y\Delta_y=0$. Under uniform flips to a different class at rate $\rho$, conditional on $Y=y$ and the fold-excluded predictors, $\mathbb{E}[\Delta_{\tilde Y}]=(1-\rho-\rho/(C-1))\Delta_y$. This applies to fixed candidate comparisons when held-out flips are independent of the predictions. It does not identify cyclic-noise preferences, guarantee that minimizing sample cross-entropy selects the clean-risk minimizer, or turn teacher loss into a student-accuracy guarantee. Zhou et al. prove the related fixed-permutation result. Sun et al. swap logits with the known ground-truth class. [Zhou et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_Learning_With_Noisy_Labels_via_Sparse_Regularization_ICCV_2021_paper.html) · [Sun et al., ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Knowledge_Distillation_with_Refined_Logits_ICCV_2025_paper.html)

The archive samples a fixed number of symmetric flips without replacement. Conditional on the retained fold's true labels and flip indicators, replace $\rho$ in the identity with the residual flip rate $(M-K)/(N-s)$, where $M$ is the total number of flips across $N$ training rows and $K$ occurred among the $s$ retained anchors. Every held-out row in that fold has the same conditional rate. The corresponding loss scaling stays positive for these budgets and rates. The scale can vary across folds.

Apply the chosen depth to the archived raw and square-root distributions on the original admitted unlabeled rows. The target's original probability multiset, gate, confidence, hard labels, student membership and order, configuration, seed, 300 epochs, and checkpoint-selection rule stay fixed. Verify the archive and revised NPZs by hash and recompute the exact optimizer steps and membership. A numerically identical target takes the archived verified fit. Nonidentical targets run the standard training and verification against all 10,000 test images, with identical training metadata aside from validation-selected statistics.

## Formulas and pseudocode

For annotation $i$ with observed class $\tilde y_i$, its direct evidence at an unprotected node $x$ is $b_{xi}=K_{ix}$, using the frozen CkNN kernel $K$. Start $h^0_{xi}=b_{xi}$. Let $\alpha=1/100$, and $S_t$ be unprotected nodes with current maximum probability greater than $\alpha+10^{-6}$. At each of the five rounds, a node that can still receive updates gets

$$
h^{t+1}_{xi}=b_{xi}+\sum_{j\in S_t}K_{xj}h^t_{ji}.
$$

A node freezes once its maximum probability exceeds $0.75$. Acquired nodes are protected. For an unprotected node after five rounds, let $H_x=\sum_i h_{xi}^5$ and $R_x=\sum_i\sqrt{h_{xi}^5}$. The original teacher and the proposal used for ordering are

$$
p_{xc}=\frac{\alpha+\sum_{i:\tilde y_i=c}h_{xi}^5}{1+H_x},\qquad
q_{xc}=\frac{\alpha+(H_x/R_x)\sum_{i:\tilde y_i=c}\sqrt{h_{xi}^5}}{1+H_x},
$$

with $H_x/R_x=0$ when $R_x=0$. Compute sender and receiver masks with $p$. Reuse those masks to decompose the same teacher into individual annotation contributions $h$.

Define $T_d(p,q)$ by moving the $d$ largest values of $p$, in descending order, to the classes ranked first through $d$ by $q$. Move each value by swapping it with the current value at its destination. $T_0(p,q)=p$.

~~~text
Split acquired indices into five folds with a fixed seed.
For each fold:
    Rebuild the protected teacher with only the other folds' labels.
    Compute p and q on this fold.
For d in (0, 1, 3):
    loss[d] = mean over acquired i of -log(T_d(p_i, q_i)[observed_label_i])
Choose the smallest d attaining the minimum loss.
Rebuild p and q with all acquired labels.
At each originally admitted unlabeled image x, set the target to T_d(p_x, q_x).
Use the original hard targets, confidence weights, and admitted rows.
Fit the original protected VAST student with the revised soft targets.
~~~

For each batch, let $H$ be acquired rows and $U$ admitted unlabeled rows, with logits $f_x$ and confidence $w_x=\max_c p_{xc}$. Both VAST fits use the student loss below with $r_x=p_x$ for the reference and $r_x=T_d(p_x,q_x)$ for rank alignment.

$$
\mathcal L=\frac{1}{|H|}\sum_{x\in H}\operatorname{CE}(f_x,\tilde y_x)
+64\frac{\sum_{x\in U}w_x\operatorname{KL}
\left(\operatorname{softmax}(r_x)\,\|\,\operatorname{softmax}(f_x)\right)}
{\sum_{x\in U}w_x}.
$$

An empty partition contributes zero. The pinned temperature is 1. The fold selector scores the graph probabilities before the stock student's softmax. The target coordinates are the only input to the student fit that the permutation changes. [VAST](https://arxiv.org/abs/2609.14451)

## Isolated cluster run

In a new directory under `/cs/labs/daphna/tomn/projects`, copy `rank.py`, `fit.py`, `test_rank.py`, and `run.slurm.sh`. Make a `helpers` folder containing the unchanged `graph_methods_study/vast/transfer.py` and `graph_methods_study/vast/train_student.py` from the research branch. The archive and square-root proposal directories are read by the Slurm entry point.

```bash
mkdir -p logs
sbatch --parsable --array=0-7 --output="$PWD/logs/%x_%A_%a.out" run.slurm.sh "$PWD" oof
```

Check that all eight `targets/*/result.json` exist, then run the matching fit array. Resource values mirror the preceding winner-swap experiment.

```bash
sbatch --parsable --array=0-7 --job-name=vast-rank-fit --time=01:00:00 --mem=16G \
  --gres=gg:g0:1,vmem:8g --output="$PWD/logs/%x_%A_%a.out" run.slurm.sh "$PWD" fit
```
