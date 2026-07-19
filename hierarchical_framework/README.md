# hierarchical_framework

Active-learning experiments testing whether class hierarchy priors (superclass /
coarse structure) help ProbCover-style selection. Everything runs over
pre-extracted, L2-normalized SimCLR embeddings and is scored with 1-NN — no
network training.

Current work uses the solo-learn embeddings (extracted by
`solo-learn/extract_embeddings.py`), passed via `--embeddings_dir`. The retired
deep_al embeddings and their results live under `old/` (see below).

## Layout

```
pipeline/     the AL loop and its swappable parts
  run_al.py            strategy-agnostic runner: pick budget points/round, 1-NN eval, wandb
  selectors_al.py      selection strategies (delta-graph, ProbCover, Random) + SELECTORS registry
  knee_delta.py        bridge to deep_al's knee-point --delta selector

analysis/     post-hoc breakdowns of a run (both import pipeline/)
  accuracy_vs_budget.py   fine vs coarse vs superclass-oracle accuracy + near/far miss split,
                          traced as the label budget grows
  superclass_confusion.py 20x20 superclass confusion heatmap + confused-pair lists at the
                          full budget, embedding-ceiling vs ProbCover

scripts/      slurm launchers (run from the repo root)
  run_al.sbatch  one AL run on a given embeddings dir

data/         embedding banks (gitignored)
outputs/      figures/, wandb/, slurm_logs/  (gitignored)

old/          retired work on the old embeddings, kept for reference (mirrors the layout above)
  data_prep/   prepare_tinyimagenet.py, build_tinyimagenet_coarse.py
  scripts/     run_tinyimagenet.sbatch
  data/        embeddings_from_deep_al/<dataset>/  (old embeddings copied out of the deep_al pipeline; gitignored)
  outputs/     figures/  (gitignored)
```

`embeddings_from_deep_al/` holds the old embeddings that were copied out of the
deep_al pipeline — kept because their features exist nowhere else in the repo (the
label files happen to match solo-learn's, but the features are distinct). The
deep_al TinyImageNet results were found to be incorrect and were removed; only the
deep_al CIFAR-100 embeddings + figures were archived under `old/`.

## Running

Scripts are self-contained and expect to be launched from the repo root
(`daphna_lab/`). analysis/ scripts add `pipeline/` to sys.path so the selectors
and delta resolution stay identical across the whole pipeline.

```
# one ProbCover run on the solo-learn CIFAR-100 SimCLR embeddings
sbatch hierarchical_framework/scripts/run_al.sbatch \
    solo-learn/embeddings/simclr/cifar100/1000ep probcover 0.65 100 10
```

`--delta` accepts a float or the keyword `knee` (runs deep_al's knee algorithm on
the features). Add `--strategy random` for the baseline.

## An embeddings dir must contain

```
<split>_features_normalized.npy   train_labels.npy
test_features_normalized.npy      test_labels.npy
coarse_map.npy                    # (num_classes,) fine->coarse, for analysis/
```
