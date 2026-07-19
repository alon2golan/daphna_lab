# with_priors

Active-learning experiments testing whether class priors (superclass / coarse
structure) help ProbCover-style selection. Everything runs over pre-extracted,
L2-normalized SimCLR embeddings and is scored with 1-NN — no network training.

## Layout

```
pipeline/     the AL loop and its swappable parts
  run_al.py            strategy-agnostic runner: pick budget points/round, 1-NN eval, wandb
  selectors_al.py      selection strategies (delta-graph, ProbCover, Random) + SELECTORS registry
  knee_delta.py        bridge to deep_al's knee-point --delta selector

analysis/     post-hoc breakdowns of a run (both import pipeline/)
  error_analysis.py    fine vs coarse vs superclass-oracle accuracy; near/far miss split
  confusion.py         superclass confusion heatmaps + confused-pair lists (ceiling vs ProbCover)

data_prep/    build the embedding banks the pipeline reads
  prepare_tinyimagenet.py       TinyImageNet embeddings dir from itai.david's bank
  build_tinyimagenet_coarse.py  WordNet-derived superclass map (coarse_map.npy)

scripts/      slurm launchers (run from the repo root)
  run_al.sbatch            one AL run on a given embeddings dir
  run_tinyimagenet.sbatch  full TinyImageNet pipeline (prep -> knee delta -> run_al -> error_analysis)

data/         embeddings_deepal/<dataset>/  (gitignored, ~360MB)
outputs/      figures/, wandb/, slurm_logs/  (gitignored)
```

## Running

Scripts are self-contained and expect to be launched from the repo root
(`daphna_lab/`). analysis/ scripts add `pipeline/` to sys.path so the selectors
and delta resolution stay identical across the whole pipeline.

```
# one ProbCover run
sbatch with_priors/scripts/run_al.sbatch \
    with_priors/data/embeddings_deepal/cifar100_simclr probcover 0.65 100 10

# full TinyImageNet pipeline
sbatch with_priors/scripts/run_tinyimagenet.sbatch 200 10
```

`--delta` accepts a float or the keyword `knee` (runs deep_al's knee algorithm on
the features). Add `--strategy random` for the baseline.

## An embeddings dir must contain

```
<split>_features_normalized.npy   train_labels.npy
test_features_normalized.npy      test_labels.npy
coarse_map.npy                    # (num_classes,) fine->coarse, for analysis/
```
