# Archived: SimCLR<->SupCon concept work

> Shelved, not under active development. These modules now live in
> `ssl_framework/old/` and the run scripts in `ssl_framework/scripts/`; the
> shared modules they use (`labeling.py`, `propagation.py`, `evaluation.py`,
> `backends/`) still sit one level up in `ssl_framework/`. The current line of
> work is `algorithms/joint_finetune.py`. Paths below are relative to
> `ssl_framework/` unless noted.

Workspace for semi-supervised learning ideas built on top of self-supervised
encoders. The base encoders and their training (SimCLR, SupCon, BYOL, ...) come
from the vendored [solo-learn](https://github.com/vturrisi/solo-learn) repo at
`../solo-learn`; this package holds the research methods built on top of them.

## Layout

- `backends/`   adapters to external training frameworks. `solo_learn.py` is the
                only module that imports solo-learn: it runs a training phase,
                warm-starts a backbone across phases, feeds a supervised method a
                labeled subset, and extracts embeddings. Everything else goes
                through it, so the backend can be swapped in one place.
- `old/`        this archived concept work. `simclr_supcon.py` iterates
                SimCLR/SupCon on a shared backbone; `supcon_finetune.py`
                fine-tunes an existing encoder with SupCon and compares
                conditions; `propagation_sweep.py` previews the propagation knobs
                without training; `threshold_search.py` ranks them.
- `labeling.py` strategies for choosing which pool samples to label. `RandomSelector`
                picks a random subset now; an active-learning selector (e.g.
                backed by `deep_al/`) can implement the same interface later.
                `class_counts` records the (unbalanced) per-class makeup of a set.
- `propagation.py` kernel label propagation: gaussian/laplacian/cosine similarity
                with a floor (`sim_threshold`), a per-class coverage matrix, and a
                confidence cutoff (`conf_threshold`) that turns confident soft
                labels into hard pseudo-labels.
- `evaluation.py` one shared yardstick for every backbone: offline linear-probe
                and weighted-kNN top-1/5 (val_acc1/5, val_knn_acc1/5).

## Running the SimCLR<->SupCon iteration

One process, one job (uses the CIFAR copies under `../data`):

```
python -m ssl_framework.old.simclr_supcon \
    --dataset cifar100 --n_labeled 1000 \
    --num_iterations 3 --simclr_epochs 200 --supcon_epochs 100 \
    --output_dir runs/simclr_supcon_cifar100_l1000
```

Outputs, under `--output_dir`: `labeled_indices.npy`, a `backbone_iter*.pt` per
iteration, `backbone_final.pt`, and `train_/val_features.npy` (+ labels) from the
final backbone.

Requires the solo-learn dependencies in the project venv (lightning, hydra,
timm, ...) and a GPU for real runs.

## SupCon fine-tuning on an existing encoder (+ kernel propagation)

Take a trained SimCLR/BYOL checkpoint and ask whether a SupCon pass on a small
labeled set improves it, and whether kernel-propagated pseudo-labels improve it
further. Three conditions -- `baseline`, `supcon_only`, `supcon_prop` -- are
scored on the SAME offline linear-probe + weighted-kNN (`val_acc1/5`,
`val_knn_acc1/5`); the trained ones also stream solo's native online metrics to
wandb.

### The full sweep (SLURM)

One launcher submits everything: `{simclr,byol}` x `{cifar10,cifar100}` x per-class
budgets `L` x kernels `{gaussian,laplacian,cosine}`. The run scripts live in
`scripts/`; invoke them from `ssl_framework/` (they locate the project root
themselves, so the exact cwd does not matter):

```
./scripts/run_supcon_sweep_all.sh                 # budgets L = 2,4,6,10,25,50
./scripts/run_supcon_sweep_all.sh 2,4,6,10        # custom budgets
DRY=1 ./scripts/run_supcon_sweep_all.sh           # print the job graph, submit nothing
```

Per (embedding, dataset) it submits ONE fast `baseline` job (extracts + caches
the frozen features, scores the raw encoder -- no labels, identical across
budgets), then one `budget` job per `L`, each depending (`afterok`) on that
baseline so it reuses the cache. `n_labeled = L * num_classes` (cifar10 x10,
cifar100 x100); selection stays random/unbalanced and the per-class makeup is
recorded. Budgets go small -> large so wandb fills in quickly.

### Results tree

Everything lands under `results/<dataset>/<embedding>/`, split by condition, then
kernel, then hyperparameters, so multiple attempts sit side by side:

```
results/<dataset>/<embedding>/
    base_features/                                  cached frozen features (shared)
    baseline/scores.json                            just the embedding
    supcon_only/L<L>_s<seed>/                        embedding + supcon
    supcon_prop/<kernel>/bw<bw>_sim<st>_conf<ct>/L<L>_s<seed>/   + propagation
```

Each leaf holds `scores.json` (the four metrics + full metadata) plus its
artifacts (`backbone.pt`, `pseudo_*.npy`, `propagation_stats.json`,
`labeled_indices.npy`). Slurm logs go to `slurm_logs/<jobname>_<jobid>.{out,err}`
(absolute `%u` path, so it works no matter where you `sbatch` from).

### wandb

New project `semi-supervised-supcon`. Every run carries: `group` = `<embedding>-<dataset>`,
`job_type` = `baseline|supcon_only|supcon_prop`, `tags` = `[embedding, dataset,
condition, kernel]`, and a config with `base_method, dataset, condition, kernel,
per_class_budget, n_labeled, bandwidth, sim_threshold, conf_threshold, seed`. To
get the headline plot, make a line panel with X = `per_class_budget`, Y =
`offline/val_knn_acc1`, grouped by `base_method` + `dataset` + `job_type` +
`kernel` -- one curve per (embedding, dataset, condition, kernel).

### One case, no SLURM

```
./scripts/run_supcon_finetune.sh trained_models/simclr/cifar100/1000ep cifar100 10   # L=10
```

runs `--mode all` (baseline if not cached, then the budget) for a single case on
the local GPU.

### Choosing propagation hyperparameters cheaply

Fine-tuning is budgeted in gradient steps, not epochs: `--supcon_target_steps`
(default 5000) auto-picks epochs; `--supcon_lr` defaults to a gentle 0.1.

Each `(kernel, bandwidth, sim_threshold, conf_threshold)` changes the pseudo set
and needs its own SupCon run, so preview the grid first -- no training -- with
`propagation_sweep`, which reports pseudo-count + pseudo-label accuracy per combo:

```
python -m ssl_framework.old.propagation_sweep \
    --features results/cifar100/simclr/base_features \
    --dataset cifar100 --n_labeled 1000 --seed 1 --kernel gaussian \
    --bandwidth 5,10,20 --sim_threshold 0.0,0.1,0.2 --conf_threshold 0.7,0.8,0.9
```

`--bandwidth auto` (the default in the runners) picks the median-distance
heuristic per case; pass an explicit value to override. With the default
thresholds propagation is conservative (few/zero pseudo-labels at small budgets),
so tune from the sweep before trusting `supcon_prop`.

### Ranking thresholds (no training)

`threshold_search` sweeps `(kernel, bandwidth, sim_threshold, conf_threshold)` on
the frozen features and ranks each by `net_correct = n_pseudo * (purity - lambda *
(1 - purity))` -- expected correct-minus-penalised-wrong pseudo-labels. It prints
a ranked table + a short "why" + a recommended shortlist, and saves the grid to
`results/<dataset>/<embedding>/threshold_search/L<L>.json`. Run it across every
case with `./scripts/run_threshold_search.sh`. Then train the shortlist:
`--kernels`, `--bandwidth`, `--sim_threshold`, `--conf_threshold` are all
list-valued, so each combination becomes its own `supcon_prop` leaf + wandb run.

### Parameters: what is swept vs fixed

Swept (each value is a separate run/folder/wandb run):
- embedding: simclr, byol (which checkpoint) -- the pretrained encoder.
- dataset: cifar10, cifar100.
- per_class_budget L: 2,4,6,10,25,50 -> n_labeled = L * num_classes. The labeled
  size, and the plot x-axis. Selection is random, so counts per class vary.
- kernel: gaussian, laplacian, cosine -- the similarity used to propagate.
- bandwidth h: kernel width. gaussian is exp(-(d/h)^2); small h = only very close
  points count (pure, sparse), large h = distant points count too (more reach,
  more noise). auto = median pairwise distance; 0.5x = half of it. Cosine ignores
  it. Swept in threshold_search; the training sweep defaults to 0.5x.
- sim_threshold: kernel floor; pairs with similarity below it add 0 to the
  coverage matrix ("far points don't help"). Raises purity. Default 0.1.
- conf_threshold: how confident a soft label must be to become a hard
  pseudo-label. Default 0.8.

Fixed (one value everywhere, change by hand if you want to study them):
- seed = 1 (which points are labeled; fix for comparable runs, vary for repeats).
- supcon_target_steps = 5000 -> auto-derives epochs (an epoch over n points is
  ceil(n/batch) steps), CAPPED by supcon_max_epochs = 200 so tiny budgets do not
  run thousands of one-step epochs. Net effect: up to 200 passes over the labeled
  set (tiny budgets ~200 steps, large budgets ~2000-4000).
- supcon_lr = 0.1, classifier_lr = 0.1, batch_size = 256 (capped at n), warmup auto.
- knn_k = 20, knn_distance = cosine, probe_epochs = 100 (the eval yardstick).
- agg = max (per-class coverage pooling), confidence_metric = prob, max_pseudo = none.

### wandb (project semi-supervised-supcon)

Confirmed logging to entity `sapir-tobias-47-hebrew-university-of-jerusalem`.
Sort / group / filter runs by any of: `group` (embedding-dataset), `job_type`
(baseline / supcon_only / supcon_prop), `tags` (embedding, dataset, condition,
kernel, L), or config/summary fields `base_method, dataset, condition, kernel,
per_class_budget, n_labeled, bandwidth, bandwidth_token, sim_threshold,
conf_threshold, seed`, and metrics `offline/val_acc1`, `offline/val_acc5`,
`offline/val_knn_acc1`, `offline/val_knn_acc5`, `n_pseudo`, `pseudo_accuracy`.
