# Equal-cost signed label verification

Freeze the verified CIFAR-100 BYOL geometry with 45,000 eligible training images, 13 neighbors per image and 10,000 held-out test images. Use three-hop unweighted walk counts, the archived seeded greedy coverage rule and fresh acquisition and noise seeds 80–84. Initial noisy budgets are 400, 800, 1,200 and 1,600. Each audit or expansion arm spends an additional 10% of its initial budget on perfect clean answers. Use the lab's symmetric and within-superclass cyclic asymmetric noise at 30%, 40% and 50%, plus clean and the aligned CIFAR-100N annotation vector. Rank requests before reading any clean answer or test label.

Let $A_{xj}=\mathbf{1}[j\in N_{13}(x)]$ on eligible training images. Let $A_*$ link each test image to its 13 nearest training neighbors. The train and test vote weights are entries of $A^3$ and $A_*A^2$, respectively, restricted to acquired columns. Let $w_{xi}$ be the resulting walk count from point $x$ to acquired annotation $i$. Let $g_x(c)=\sum_i w_{xi}\mathbf{1}[\tilde y_i=c]$. At an acquired point $i$, drop its own votes, select the highest-scoring other class $a_i$, and set $m_i=[g^{-i}_i(a_i)-g^{-i}_i(\tilde y_i)]/\max(1,\sum_{j\ne i} w_{ij})$. Leave-one-out verification ranks $m_i$ in descending order.

For an eligible target $x$ outside the common initial and clean expansion prefixes, set $c_x=\arg\max_c g_x(c)$. Move $i$'s vote mass from $\tilde y_i$ to $a_i$ and call the resulting winner $c'_{xi}$. Define $q^{-i}_x(c)=[g_x(c)-w_{xi}\mathbf{1}[\tilde y_i=c]]/[\sum_j w_{xj}-w_{xi}]$ when the denominator is positive. The signed score is

$$
S_i=\max(0,m_i)\sum_{x\,:\, w_{xi}>0,\;\sum_{j\ne i}w_{xj}>0}
     \big[q^{-i}_x(c'_{xi})-q^{-i}_x(c_x)\big].
$$

Rank by decreasing $S_i$, decreasing $m_i$, then acquisition order. Every verifier corrects selected labels with the oracle and keeps all initial anchors. The unsigned footprint is $F_i=\{x:c_x=\tilde y_i,\ c'_{xi}\ne c_x\}$. The independent unsigned baseline ranks $\max(0,m_i)|F_i|$. The greedy unsigned baseline builds a batch $J$ by adding the largest marginal gain in $\sum_x\max_{i\in J:x\in F_i}\max(0,m_i)$, with zero for an empty maximum. The other controls are random auditing, leave-one-out auditing and buying the next coverage selections clean. Coverage chooses anchors by the largest number of still-uncovered training images that have the anchor in their 13-neighbor lists or are the anchor itself. The initial run without new requests is a zero-cost reference.

Add a learned noise-ranking control using the unmodified LowBudgetAUM detector from the pinned NPC release. Give the detector the raw BYOL features and the same acquired noisy labels. It trains the released 101-class linear detector for 40 epochs with its standard fake-threshold examples, optimizer and batch sizes. Use one detector fit per seed, budget and condition. Seed it with $700000+10000\sigma+B$ for acquisition seed $\sigma$. For each acquired image, its margin score is $\operatorname{AUM}_i=|\mathcal{T}_i|^{-1}\sum_{t\in\mathcal{T}_i}[z_{it}(\tilde y_i)-\max_{c\ne\tilde y_i}z_{it}(c)]$, where $\mathcal{T}_i$ is its set of training visits. Verify the $B/10$ lowest-score images, breaking ties by acquisition order. This adapts the detector to an equal-cost ranking problem. The acquisition and prediction stages are those of this graph replay.

All graph arms predict $\arg\max_c\sum_i w_{xi}\mathbf 1[y_i^{\rm after}=c]$ with the labels and anchor sets produced by their respective requests. The primary readout is held-out graph accuracy. Also report cosine 1NN accuracy and the number of corrections. Compare signed auditing against each control, and report wins and losses against the strongest equal-cost control per setting. Budgets share prefixes, and all seeds share the same test set. The original verification results remain untouched.

## Reproduce

Stage the runner and three checked-in dependencies in a fresh cluster directory. The verified NPC runtime and frozen graph remain readable inputs.

```sh
mkdir "$job_root" "$job_root/source"
cp matched_research/signed_audit/audit.py "$job_root/source/"
cp active_audit_study/active_audit.py active_audit_study/collect.py "$job_root/source/"
cp matched_research/netta/run_official.py "$job_root/source/"
cp matched_research/signed_audit/run.slurm.sh "$job_root/"
sbatch --array=80-84%2 --output="$job_root/replay_%A_%a.log" \
  "$job_root/run.slurm.sh" "$job_root" "$geometry_npz" "$pinned_npc_root"
"$lab_python" "$job_root/source/audit.py" collect "$job_root"
"$lab_python" matched_research/signed_audit/summarize.py \
  "$job_root/metrics.csv" "$job_root/comparisons.csv"
```

The AUM comparator takes its margin statistic from Pleiss et al. and its low-budget detector implementation from the released Shafir et al. code. [AUM paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html) · [Pinned NPC release](https://github.com/nettashafir/active_learning_with_label_noise/tree/733651bb0943a96392895103eb2ffc4e99a7c7d1)
