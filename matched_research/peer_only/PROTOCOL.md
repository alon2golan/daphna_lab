# Peer-only ablation on the released NPC classifier

Test whether the local agreement signal alone explains the gain from adding its rank to the LowBudgetAUM rank. Reuse all 105 frozen fit snapshots from the released NPC experiment (CIFAR-100 BYOL, three seeds, five budgets and seven noise settings). The snapshot fixes acquired IDs, observed labels, AUM scores, initialization, optimizer and sampling state. The reference and fusion scores come from those previously verified fits. [Original protocol](../netta/PROTOCOL.md)

For each acquired row, use the same locally scaled weighted vote of the 13 nearest other acquired rows as the fusion arm. Within each observed class, fill the original LowBudgetAUM quota in descending agreement order. Break exact agreement ties by original retention, AUM, then acquisition order. The fusion method adds the within-class AUM midrank to the agreement midrank. The peer-only comparison removes the AUM term from ranking while sharing its quota and tie rule.

The closest published comparison is SimiFeat's classwise rank filter. It ranks nearest-neighbor label evidence normalized by its vector norm and estimates class quotas from the observed noise statistics. Its paper also discusses ranking by the assigned-class component used in this ablation. [SimiFeat](https://proceedings.mlr.press/v162/zhu22a/zhu22a.pdf)

Use the untouched `run_official.py fit` entry point from the archived NPC study. It must match the original classifier's test-prediction hash and model hash before running the peer-only fit. Slurm requests the RTX 2080 GPU type, and the launcher checks the actual archived RTX 2080 Ti device name, fit wrapper hash, and package versions. The canary is symmetric 50%, seed 71, 400 labels. Verify its two fit outputs before launching the remaining grid. The full grid reuses this canary result.

From an isolated cluster directory containing the files in this folder, set `reference` to the archived NPC directory containing `runs`, `peer`, `source`, `inputs`, and `runtime_deps`.

```bash
mkdir -p logs
sbatch --array=3 --time=00:12:00 --job-name=npc-peer-canary --output="$PWD/logs/%x_%A_%a.out" \
  run.slurm.sh canary "$PWD" "$reference"
```

After the canary completes, verify its `identity_result.json` against the original `baseline_result.json`, then launch the grid.

```bash
sbatch --array=0-20%10 --output="$PWD/logs/%x_%A_%a.out" \
  run.slurm.sh grid "$PWD" "$reference"
```

The collector verifies the original and fusion artifacts, peer-only masks, identity hashes and input lineage. It writes all 105 pairs and all 35 three-seed setting means for peer-only versus LowBudgetAUM and fusion versus peer-only.

```bash
PYTHONPATH="$reference/source:$reference/runtime_deps" \
  python collect_peer_only.py "$reference" "$PWD" scores
```
