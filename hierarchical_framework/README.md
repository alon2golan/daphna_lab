# hierarchical_framework

Active-learning experiments testing whether class hierarchy priors (superclass /
coarse structure) help ProbCover-style selection. Everything runs over
pre-extracted, L2-normalized SimCLR embeddings and is scored with 1-NN — no
network training.

## Two embedding families

Embeddings are organized `embeddings/<dataset>/<technique>/<source>`, where
`<source>` is one of two families the runner reads via `--embeddings_dir`:

- old_deep_al — itai's features via the deep_al pipeline (the embeddings used in
  the ProbCover paper). The baseline to reproduce / compare against.
- new_solo_learn — solo-learn features (extracted by
  `solo-learn/extract_embeddings.py`). The current work.

Every output figure/run mirrors the same path, e.g.
`cifar100/simclr/old_deep_al_d0.65` (strategy and budget are defaults, kept in
wandb config rather than the folder name).

## Layout

```
pipeline/     the runner and its swappable parts. Logs accuracy to wandb; makes NO figures.
  run_active_learning.py   the runner: each round, a Selector picks and an Evaluator scores -> wandb
  selection_strategies.py  WHICH points to pick (delta-graph, ProbCover, Random) + SELECTORS registry
  evaluators.py            HOW to score a labeled set (1nn, mlp) + optional superclass oracle + EVALUATORS registry
  delta_selection.py       resolves --delta via deep_al's resolve_initial_delta (auto_delta table / knee)
  coarse_maps.py           fine->coarse (superclass) maps + names, shared with analysis/
  output_paths.py          run-naming from the embeddings path, shared with analysis/
  run_logging.py           RunLogger: console line + wandb (accuracy + regular scalars only)

analysis/     on-demand figure tools; each replays a run, imports pipeline/, writes to outputs/
  selection_diagnostics.py picks-per-class + coverage-gain plots (what the strategy chose)
  accuracy_vs_budget.py    fine vs coarse vs superclass-oracle accuracy + near/far miss split vs budget
  superclass_confusion.py  20x20 superclass confusion heatmap + confused-pair lists, ceiling vs selected

scripts/      slurm launchers (run from the repo root)
  run_active_learning.sbatch  one AL run on a given embeddings dir

embeddings/   (gitignored) <dataset>/<technique>/<source>
  cifar100/simclr/old_deep_al/     itai's SimCLR features, L2-normalized
  cifar100/simclr/new_solo_learn   symlink into solo-learn/embeddings/simclr/cifar100/1000ep

outputs/      (gitignored) figures/, wandb/, slurm_logs/
              run folders mirror the embeddings path, e.g.
              figures/error_analysis/cifar100/simclr/old_deep_al_d0.65/

archive/      retired, incorrect TinyImageNet work on the old embeddings
  data_prep/   prepare_tinyimagenet.py, build_tinyimagenet_coarse.py
  scripts/     run_tinyimagenet.sbatch
```

`cifar100/simclr/old_deep_al` is byte-for-byte itai's `cifar100_simclr_train.npy`
after L2-normalization. The raw source lives at
`/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/`, so the copy is
regenerable in one line if ever needed.

## Running

Scripts are self-contained and expect to be launched from the repo root
(`daphna_lab/`). analysis/ scripts add `pipeline/` to sys.path so the selectors
and delta resolution stay identical across the whole pipeline.

```
# current work: solo-learn (new) embeddings
sbatch hierarchical_framework/scripts/run_active_learning.sbatch \
    hierarchical_framework/embeddings/cifar100/simclr/new_solo_learn

# baseline: itai's (old) embeddings, as in the ProbCover paper
sbatch hierarchical_framework/scripts/run_active_learning.sbatch \
    hierarchical_framework/embeddings/cifar100/simclr/old_deep_al
```

`--delta` is resolved by deep_al's `resolve_initial_delta` (the same rule ProbCover
uses): a float is used as-is, `knee_delta` runs the data-driven knee algorithm, and
`auto_delta` (the default) looks up the pinned `PROBCOVER_AUTO_DELTAS` table for the
dataset (CIFAR-100 -> 0.65) and falls back to the knee if the row is missing. The
run prints a `[delta]` line saying which value was chosen. Add `--strategy random`
for the baseline.

`--eval` picks the base classifier used to score the labeled set each round: `1nn`
(default) or `mlp`. Add `--oracle` to restrict every prediction to the test point's
true superclass — an upper bound that composes with either base (1-NN searches only
same-superclass neighbours; the MLP keeps only same-superclass output classes). Only
the accuracy + the regular AL scalars go to wandb (project `superclass_analysis`).

The runner never makes figures. When you want plots, run an `analysis/` tool on the
same knobs — it replays the run and writes figures under `outputs/figures/`:

```
python hierarchical_framework/analysis/selection_diagnostics.py --embeddings_dir <EMB>
python hierarchical_framework/analysis/accuracy_vs_budget.py   --embeddings_dir <EMB>
python hierarchical_framework/analysis/superclass_confusion.py --embeddings_dir <EMB>
```

## An embeddings dir must contain

```
<split>_features_normalized.npy   train_labels.npy
test_features_normalized.npy      test_labels.npy
coarse_map.npy                    # (num_classes,) fine->coarse, for analysis/
```
