# VAST on frozen lab features

Dedicated single-seed experiments implementing the approved specification in
`docs/VAST_REPRODUCTION_SPEC.md`. Source reuse is listed in `PROVENANCE.md`.

**Propagation is still scientifically unresolved.** The default blocks training.
An explicitly selected `literal-paper` mode runs the written equations and is
labeled experimental in every result. It does not establish reproduction of
the reported results. At 1/2 labels per class it stops after one round, as the
specification proves.

## Environment

On the cluster, run `bash onboarding/VAST_setup_env.sh` from the repo root.
It installs a shared Python 3.13.5 with uv, repairs the venv's interpreter links
without clearing installed packages, and installs the pinned requirements.
The bootstrap uses `/usr/bin/python3`; override `VAST_BOOTSTRAP_PYTHON` if needed.
The helper backs up existing interpreter links and `pyvenv.cfg` under
`results/vast/env_backup_*` before repair.

The cluster's Debian 12 nodes lack `/usr/bin/python3.13`; a venv created with
that path on a Debian 13 node cannot run there. Shared Python avoids this
dependency. GPU nodes still need a working NVIDIA driver; the temporary
Debian 13 test node `dumfries-004` lacked the driver library during diagnosis.

For local setup:

Use Python 3.11, 3.12 or 3.13 in a separate environment:

```bash
python3.13 -m venv .venv-vast
source .venv-vast/bin/activate
python -m pip install --only-binary=:all: -r deep_al/vast/requirements.txt
```

The pins use NumPy 2.2.6, PyTorch 2.6.0 and FAISS 1.9.0.post1, which have
Python 3.13 wheels. A fresh Python 3.13.13 environment passed all 33 tests
and the CPU smoke launcher, including replay and viewer checks. Cluster GPU
smoke job `31831088` completed on `dumfries-007` on 2026-10-01 in 2 minutes
2 seconds (Slurm exit `0:0`), with Python 3.13.5 and PyTorch 2.6.0+cu124.
It trained both epochs on CUDA and saved the checkpoint and propagation replay.
The submission excluded the temporary Debian 13 reservation's nodes while
their driver setup was being diagnosed. This is a synthetic execution check;
real-data alignment and paper accuracy remain unverified.

FAISS CPU builds the graph; PyTorch uses CUDA for training when available, or
CPU otherwise. FAISS runs in a separate process to avoid conflicting OpenMP
runtimes in macOS wheels. `VAST_FAISS_THREADS` controls search threads (default
up to 8); `--cpu-threads` controls PyTorch threads (default 1).

## Prepare a manifest

Copy `manifest.example.yaml` and supply real aligned arrays. All paths resolve
relative to the manifest; absolute paths are also accepted. Arrays are ordinary
NPY files without pickle objects:

- Features: finite floating-point `[N,D]` matrices with nonzero rows.
- Labels: integer `[N]` arrays indexing the ordered class-ID list.
- Sample IDs: unique, nonempty string `[N]` arrays, with no train/test overlap.
- Class IDs: an ordered JSON list of 100 strings (CIFAR/ImageNet100) or 200
  strings (TinyImageNet).
- `alignment`: a truthful description of how feature and label ordering was
  established. Validation checks counts and IDs but cannot infer provenance.

**Existing lab files need metadata:** CIFAR uses original clean fine labels;
TinyImageNet requires labels/IDs in its actual feature-extraction order;
ImageNet100 must be selected by verified row metadata from the full-ImageNet
DINOv2 bank. Do not point the ImageNet100 manifest at the full bank or assume
the first rows are its subset. The feature files and this alignment metadata
are not available in this Mac checkout.

Graph geometry uses L2-normalized features. The MLP consumes raw frozen
features, following Itai's implementation. Its hidden layer is 256; input width
is read from the arrays. Current lab representations differ from paper BYOL
representations, so paper means are comparison references only.

Validate without creating output files or selecting a propagation rule:

```bash
python deep_al/tools/train_vast.py \
  --manifest /path/cifar100.yaml --labels-per-class 1 --preflight
```

## One experimental run

```bash
python deep_al/tools/train_vast.py \
  --manifest /path/cifar100.yaml \
  --labels-per-class 1 --seed 0 \
  --propagation literal-paper \
  --cache-dir /path/vast/graph_cache \
  --output-dir /path/vast/cifar100_simclr_labels1_seed0
```

Every invocation uses exactly one integer seed (default 0). Defaults are 300
epochs, Adam lr 0.001 and weight decay 0.03, paper batch sizes and distillation
weights. `--config config.yaml` can provide settings; explicitly provided CLI
values override them. Supported config keys are `seed`, `epochs`, `batch_size`,
`distillation_weight`, `device`, `backend`, `max_neighbors`, `max_rounds`,
`propagation`, `cache_dir` and `cpu_threads`. Cache paths are relative to the
working directory. Unknown keys are rejected.

For a quick plumbing check use `--epochs 2 --device cpu`. This is a smoke run,
not the 300-epoch protocol. `--backend dense` is limited to 4096 examples.
Production graph search finds the smallest connected k up to `--max-neighbors`
(default 512). If the graph remains disconnected it fails and asks for an
explicit higher cap; it never substitutes a disconnected graph. A propagation
`--max-rounds` cap also fails rather than treating truncation as convergence.

An empty admitted pool results in labeled-only training, explicitly recorded.
Output directories must not already exist. Failures leave a `status.json`
explaining the error; select a fresh directory for a retry.

## Nine settings, one chosen seed

```bash
python deep_al/tools/run_vast_grid.py \
  --manifests /path/cifar100.yaml /path/imagenet100.yaml /path/tinyimagenet.yaml \
  --seed 0 --propagation literal-paper --output-dir /path/vast/grid_seed0 \
  --dry-run
```

The dry run validates all three inputs for the largest budget and prints nine
commands. Remove `--dry-run` to execute sequentially. Graphs are cached across
budgets and `summary.csv` is updated after each successful run. Use a separate
grid root for another representation. There is no implicit multi-seed loop.

To submit datasets independently (300 epochs for each of 1/2/4 labels per class):

```bash
VAST_MANIFEST="$PWD/data/vast/cifar100.yaml" sbatch onboarding/VAST_dataset.sh
```

Repeat with the ImageNet100 and TinyImageNet manifests once their inputs are
verified and readable. Each job writes a separate output root and updates its
`summary.csv` after every completed budget. Per-budget folders retain
`training.jsonl`, `result.json`, predictions, input hashes, the final checkpoint,
and `propagation_trace/`. Evaluation labels and sample IDs are saved alongside
predictions after training, allowing per-class statistics and confusion
matrices from a copied folder. Copy a per-budget folder to your laptop and
select that folder in Streamlit. Graph caches are shared across budgets within a job.
The per-dataset launcher requests 24 hours and 8 GB of GPU memory: features
stay on CPU and the classifier transfers only each batch to CUDA. These are
starting allocations; adjust the wall time if a real run needs longer.

## Author-source propagation experiment

To run the source-based recurrence found in Itai's public code, use the
separate launcher below. It runs budgets 1, 2, and 4 for one dataset and seed,
uses the lab's initial receiver-plateau stopping rule, and defaults to the
inspected CIFAR100-log thresholds (sender 0.05, receiver 0.75) with a 50-round
cap. The sender threshold is documented for Itai's CIFAR100 log; applying it
to TinyImageNet is an experimental assumption. Provide a pre-existing graph
cache from a run with the same verified feature bank to avoid rebuilding it.
The output is explicitly marked `author-source` and is not a confirmed paper
reproduction.

```bash
VAST_MANIFEST="$PWD/data/vast/cifar100.yaml" \
VAST_CACHE_DIR="$PWD/results/vast/cifar100_seed0_<previous-job>/graph_cache" \
sbatch onboarding/VAST_author_source_dataset.sh
```

## Results and inference

Each run saves resolved configuration, input/code hashes and provenance,
labeled indices/IDs, graph diagnostics, propagation history, gate thresholds,
admitted indices, soft targets, per-epoch loss, final checkpoint, predictions
and accuracy. `result.json` contains one seed's accuracy and its difference
from the paper mean. SD and SEM are null; no statistical significance is claimed.
The final epoch checkpoint is used, with no validation or test-based selection.

For new raw feature vectors from the **same frozen encoder**:

```python
from deep_al.vast.training import predict_checkpoint
predicted_class_indices = predict_checkpoint("/path/run/final.pt", raw_features)
```

Resolve class names using `provenance.json`'s ordered `class_ids` list. No graph
or reference labels are needed for inductive prediction.

## Tests

```bash
python -m pytest deep_al/vast/tests -q
```

Tests cover graph/reference equivalence, explicit self exclusion, kernel/evidence
arithmetic, the propagation discrepancy, gate cases, reverse KL and gradients,
balanced batches, manifest checks, one-seed grids, final-checkpoint reload,
repeatability and training invariance when only test labels change.

## Propagation replay

Every new run writes `propagation_trace/` automatically. `samples.npz` stores
all training sample IDs, labeled-seed membership and a fixed 2D PCA projection
of normalized features. PCA is fitted on a deterministic sample of at most
10,000 rows (visualization seed 0, independent of the experiment RNG); all rows
are projected in batches. This layout is only for display and never enters the
propagation or classifier.

Frame 0 records initialization with no sender/receiver activity. Each later
`round_*.npz` records the algorithm's sender and receiver eligibility masks
**before** the synchronous update and predicted class, maximum confidence,
frozen mask and newly frozen mask **after** that update. This includes the final
no-new-freeze round. Labeled seeds count as frozen from initialization; initial
high-confidence unlabeled samples may also already be frozen. Sender and receiver
roles overlap. Eligibility does not imply that a sample has an active neighbor;
these logs do not record individual messages along graph edges.

Frames are compressed and written incrementally; memory does not grow with the
number of rounds. Storage is O(samples × rounds), without full class-probability
matrices. The metadata index is updated atomically after each completed frame.
The trace is self-contained; no original features or training environment are
needed to view it. Install `requirements-viewer.txt` alongside the base requirements
and run the command in the README. The viewer defaults to at most 20,000 displayed
samples with a stable sample across rounds; counters always cover the full dataset.
Raise the display limit to see all samples, or select `Newly frozen` to inspect
that set directly. Symbols identify freeze status, colors identify propagation
role, and hovering shows the sample ID, estimated class and confidence. The class
estimate shown for labeled seeds is also the propagated estimate, not their known
training label. The table lists the same displayed samples.

For a cluster run, copy its output directory to your laptop and launch Streamlit
locally. The app does not submit jobs or require a GPU. Old aggregate logs cannot
recover missing sample roles: rerun with the updated code to produce replay files.
