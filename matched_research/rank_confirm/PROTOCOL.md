# Fresh confirmation of AUM-rank graph votes

Freeze the graph vote before running new seeds 74, 75 and 76. Run clean, symmetric 30/40/50%, and the lab's CIFAR-100 superclass-cycle asymmetric 30/40/50%. Score 400, 800, 1,200, 1,600 and 2,000 acquired labels. The earlier seeds 71–73 motivated the rule. They do not enter these results.

Each trajectory calls the unchanged released NPC BYOL wrapper from `cifar_matched_npc_20261127`. It preserves the full 50,000-image BYOL pool, acquisition from empty in batches of 100, Euclidean radius 0.65, released LowBudgetAUM, final official keep mask, and released 500-epoch linear classifier. The official baseline's predictions on all 10,000 held-out test images are computed and verified by its wrapper in each run. The graph and bank match the archive. The graph has 13 cosine edges per training row with self excluded, and 13 edges from each test row into training. The training graph has no test nodes.

At each budget let $K$ be the released final AUM keep set, $K_c$ its retained examples with observed class $c$, and $r_i$ the zero-based AUM midrank of $i$ in $K_c$. Define

$$
\omega_i=\tfrac12+\frac{r_i+\tfrac12}{|K_c|}.
$$

The class-average weight is exactly one. Let $h_i(x)=(TA^5)_{xi}$ count six-hop walks from test image $x$ to $i\in K$, where $T$ is the test-to-train adjacency and $A$ is the train adjacency. Score the same reached labels with unit weights or AUM ranks.

$$
\hat y(x)=\arg\max_c \sum_{i\in K_c} \omega_i h_i(x).
$$

The primary comparison is rank-weighted six-hop versus raw six-hop on identical retained labels and test inputs, with lowest class ID resolving exact vote ties. Report every pair and three-seed setting mean. Also report the unchanged supervised linear baseline and raw three-, four-, five-, seven- and eight-hop scores. The strongest raw depth per setting is selected from the three-seed means after scoring. A graph readout replacing the classifier is a separate change from the rank weight. Budgets are nested within a seed. The evaluation covers one frozen representation and synthetic noise channels.

The scripts use original archive source and inputs read-only, and write predictions and aggregates under a fresh project directory. The frozen six-hop geometry hash is `b76897f7fb38c60d56a9865ca7b3acd8aba7d7aa4af0e9eb4e336b29fda7b892`. The pinned official wrapper hash is `d5db57a7116111953ffdaa046b9f6369b6f6682f0fd02cbc04511330631db6b4`.

## Reproduce

Put `run_fresh.py`, `run.slurm.sh`, `score.py` and `score.slurm.sh` under `source/` in a fresh lab project directory. Create `logs`, `runs` and `results` there. First run the official array, then run graph parity against the archived clean seed-71 result. After both have passed, score the new trajectories. `work` is the fresh project path and `archived_weights` is the previous weight replay directory.

```sh
sbatch --array=0-20%16 --gres=gpu:rtx2080:1 --output="$work/logs/run_%A_%a.log" \
  "$work/source/run.slurm.sh" "$work"
sbatch --output="$work/logs/parity_%j.log" \
  "$work/source/score.slurm.sh" "$work" --parity "$archived_weights/results/clean_s71.json"
sbatch --array=0-20%12 --output="$work/logs/score_%A_%a.log" \
  "$work/source/score.slurm.sh" "$work"
```

From the repository root, point `archive` to the completed run with `parity.json`, `results/`, and the official `runs/*/summary.json` and `runs/*/provenance.json` files.

```sh
PYTHONPATH=matched_research/netta:matched_research/npc_cap:active_audit_study:graph_methods_study/sota/netta:graph_methods_study/sota/capped_votes \
  python matched_research/rank_confirm/collect.py "$archive" "$output"
```
