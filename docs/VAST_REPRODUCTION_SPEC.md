# VAST reproduction specification

Status: pipeline implemented; propagation semantics awaiting clarification from Itai.

Updated: 2026-10-01. Branch: `codex/vast-paper-reproduction`, based on
`tomn_noisy_al` at `be0cd78b65632ada6bf2c0dcc75fae220a60c20e`.

## 1. Objective and agreed decisions

Build a dedicated VAST runner inside `daphna_lab`, reusing Itai's implementation
where possible, to support the paper's nine main dataset/budget settings.
The first deliverable was this specification. The dedicated pipeline is now
implemented; the propagation rule remains an explicit unresolved item.

### Implementation status (2026-10-01)

- Entrypoint: `deep_al/tools/train_vast.py`; one integer seed per invocation.
- Grid: `deep_al/tools/run_vast_grid.py`; exactly nine settings at one selected seed.
- Setup, manifests and commands: `deep_al/vast/README.md`.
- Reuse/corrections and pinned source: `deep_al/vast/PROVENANCE.md`.
- Default propagation remains `unresolved` and prevents training. An explicit
  `--propagation literal-paper` option runs the written equations as an
  **unconfirmed diagnostic**, always recording `confirmed_reproduction: false`.
  This does not resolve the author question or select a reproduction default.
- The FAISS exact-search worker is isolated from PyTorch to avoid native OpenMP
  runtime conflicts. Graphs are cached by normalized features and graph settings.
- No real-data result is claimed; the lab feature arrays/alignment metadata
  are absent from this checkout.
- Validation: 29 automated tests passed on Python 3.12 / CPU, including FAISS
  neighbor search, checkpoint reload and test-label isolation. A separate
  one-seed synthetic run completed all 300 default epochs with FAISS and
  nonempty soft-label distillation. CUDA and real lab datasets are not yet tested.

| Decision | Agreed behavior |
|---|---|
| Scientific reference | The attached August paper, *Follow the Geometry, Not the Model: Cold Start Semi-Supervised Learning*. |
| Authority when sources disagree | Follow the paper, recording every change from Itai's code, except the unresolved propagation rule below. |
| Initial representations | Use current lab features; clearly identify their differences from paper embeddings. |
| Integration | Dedicated feature-only runner, not an extension of active-learning episodes. |
| Randomness | One integer seed per invocation; default `0`. One chosen seed for the initial nine-run grid. |
| Initial experiment scope | VAST only, three datasets and 1/2/4 labels per class. |
| Deferred | Competing methods, appendix ablations, new SSL pretraining, repeated-seed sweeps. |
| Propagation discrepancy | Document it and obtain Itai's clarification rather than guess a correction. |

One seed controls labeled selection, model initialization, and training sampling.
It does not represent a separate seed per epoch, nor a request to run a list of
seeds internally. Users may explicitly launch another invocation with another
seed later. Changing the seed does not regenerate the frozen features.

The initial claim is **VAST on current lab representations**, not reproduction
of the published numbers. Exact numerical reproduction additionally requires
resolution of propagation and compatible embeddings, sample ordering and splits.

### References

- Primary PDF: [Itai 08 2026.pdf](</Users/tomno/Downloads/Itai 08 2026.pdf>),
  especially sections 3.1-3.5, Algorithms 1-3 and supplements B.1/C.1/D.2/D.6.
- [Public paper](https://arxiv.org/abs/2609.14451), submitted September 2026;
  use the attached August version when versions differ.
- [Pinned author code](https://github.com/itaidavid1/bayes_al_2025/tree/3d4cbd7d543b1d1290799bdfb02687519af3a264),
  commit `3d4cbd7d543b1d1290799bdfb02687519af3a264` (2026-07-02).
- Local author port: `origin/itai_bayes_misp_core` at
  `c1cb89fcc7bd2e50ee4915d53485db8cb63d0635` (2026-07-13).

## 2. Experiment matrix and reporting

Each cell in the following table is one run at the configured seed: **nine runs
total**, rather than the paper's repeated-seed experiment grid.

| Dataset | Classes | Labels per class | Total labels | Current lab representation | Paper representation | Batch | Distillation weight |
|---|---:|---|---|---|---|---:|---:|
| CIFAR-100 | 100 | 1, 2, 4 | 100, 200, 400 | SimCLR | BYOL | 200 | 64 |
| ImageNet100 | 100 | 1, 2, 4 | 100, 200, 400 | DINOv2, subset alignment required | BYOL | 200 | 64 |
| TinyImageNet | 200 | 1, 2, 4 | 200, 400, 800 | SimCLR, provenance/order to verify | SimCLR | 100 | 32 |

Draw exactly the requested number of clean labels per class, uniformly without
replacement, and save the selected sample IDs. Do not use noisy-label files or
an active-learning acquisition method for these initial experiments.

### Published comparison references

These are reference results, not required accuracy thresholds for current lab
features or a single seed.

| Dataset | 1 label/class | 2 labels/class | 4 labels/class | Paper dispersion |
|---|---:|---:|---:|---|
| CIFAR-100 | 39.2 +/- 0.9 | 47.8 +/- 0.3 | 54.4 +/- 0.4 | SEM over 5 seeds |
| ImageNet100 | 34.9 +/- 1.1 | 43.1 +/- 0.9 | 49.3 +/- 0.2 | SEM over 6 seeds |
| TinyImageNet | 17.71 +/- 1.26 | 22.83 +/- 1.40 | 27.55 +/- 1.61 | Sample SD over 5 seeds |

Report one test accuracy per run, the actual seed, representation provenance and
the difference from the published mean. Do not produce SD, SEM, confidence
intervals or significance tests from one seed, and do not treat the three
budgets as independent repeated runs of one setting. If repeated seeds are
explicitly added later, follow the paper's dataset-specific dispersion convention.

## 3. Data contract and feature alignment

Use an explicit manifest with dataset name, representation name/source,
train/test feature arrays, clean label arrays, sample identifiers and ordered
class identifiers. Resolve relative manifest paths relative to the manifest.
Record file hashes, representation provenance and feature dimensions in outputs.

Validate before graph construction:

- Feature, label and identifier counts agree for each split.
- Identifiers are unique within their split and train/test sample sets are disjoint.
- Features are finite, have nonzero row norms, and have matching train/test widths.
- Labels use the same ordered class mapping for both splits.
- The selected dataset has the intended class count and enough training examples
  for the requested balanced label budget.
- Row alignment has documented provenance; matching array lengths alone is not
  proof that features and labels belong to the same examples.

Reuse path resolution from the lab feature loader, but do not instantiate the
current image wrappers as the dedicated runner's data pipeline. They contain
assumptions that are unsuitable for verified feature-only experiments.

### Current feature bank

The repository points into
`/cs/labs/daphna/itai.david/representations_bank/`. These files are not present
in the local Mac checkout; execution requires access to the lab data or supplied
copies. Paths are configuration inputs, not embedded constants in new code.

- **CIFAR-100:** `cifar-100_simclr/cifar100_simclr_{train,test}.npy`.
  Use clean CIFAR fine labels and verified original dataset ordering.
- **TinyImageNet:** `tiny-imagenet_simclr/pretext/{features,test_features}_seed1.npy`.
  Obtain labels/IDs matching feature extraction order. ImageFolder order,
  annotation-file order and SCAN pickle order must not be assumed equivalent.
- **ImageNet100:** current keys point to full-ImageNet DINOv2 arrays,
  `imagenet_dinov2/{train,val}_features.npy`. Use verified feature-row image IDs
  to select the classes in the checked-in ImageNet100 subset list and preserve
  its ordered class mapping. Never take the first rows as an ImageNet100 subset.

Keep raw features for classifier input and a separate L2-normalized view for
graph geometry. Raw classifier input preserves Itai's source behavior; the
paper does not clearly settle classifier-input normalization. Record this as
an explicit default. Infer input width from the manifest rather than forcing
512 for the current DINOv2 features.

## 4. Reuse ledger and architecture

Entrypoint: `deep_al/tools/train_vast.py`.
Implementation package: `deep_al/vast/`.

Expose `--manifest`, `--labels-per-class`, `--seed` (one integer, default 0),
`--config` and `--output-dir`. CLI overrides configuration only for explicitly
provided values. A grid launcher enumerates dataset/budget combinations using
one selected seed; it does not add a hidden seed loop.

Port selected source functions, not the entire upstream training script. Each
ported component must cite its original path and pinned commit and describe
intentional changes. Preserve MIT copyright and license notices.

All paths below are in the pinned author repository unless marked as local.

| Component | Source to reuse | Required adaptation |
|---|---|---|
| Small-data neighbor reference | `deep-al/pycls/al/prepare_sparse_K_advanced.py`, `compute_topk_mask` | Explicit self-ID removal, union symmetry, connectedness-based k selection. |
| Sparse degree correction | `deep-al/pycls/al/kernel_utils.py`, `_apply_degree_normalization_sparse` | Reuse logarithmic branch after removing self-edges. |
| RBF edge weights | `deep-al/pycls/al/kernel_utils.py` | Apply author weight formula to graph edges; do not reuse kernel-threshold sparsification as kNN construction. |
| Evidence accumulation | `BAYES_MISP.py` / `ck_matrix_manager.py` | Retain additive accumulation patterns; replace max-based or prior-overwriting initialization with the paper's sum. |
| Propagation helpers | `deep-al/tools/train_al.py`, mask helpers and sparse multiplication; local `propagation.py` port | Recurrence/stopping unresolved; see section 6. |
| Gate preparation | `deep-al/tools/train_al.py`, `prepare_distillation_targets` | Add paper mode heuristic, top-class statistic, chance floor and unlabeled-only admission. |
| Classifier | `deep-al/pycls/core/builders.py`, `MLPClassifierTorch` | Reuse architecture with hidden width 256 on all three datasets. |
| Training loop | `deep-al/tools/train_al.py`, `train_mlp_with_distillation` | Correct sample weighting, fixed 300 epochs, balanced batches and final-checkpoint policy. |
| Distillation | `deep-al/pycls/core/losses.py`, `distillation_loss` | Reverse upstream KL direction to follow paper, temperature 1, weight samples before reducing. |
| Balanced label selection | `deep-al/pycls/al/Sampling.py`, `random_1c` | Extend one-per-class selection to 1/2/4; persist selected IDs. |

### Graph and seeded evidence

For the training pool only, find the smallest positive k whose union-symmetric
kNN graph is connected. Use exact FAISS neighbor search in production and a
small dense/batched reference for validation. Exclude self by identity, not by
assuming the first nearest neighbor is self. Use sparse storage rather than a
full N-by-N kernel.

For L2-normalized feature vectors z, define edge weights and degrees as:

```text
K_ij = exp(-||z_i - z_j||^2)
d_i = sum_j K_ij
Khat_ij = K_ij / sqrt(log(1 + d_i*d_j))
```

Use natural logarithms, no self-edges, no additional row normalization and no
clipping Khat to 1. Seed the evidence matrix for M classes with:

```text
V_ic = 1/M + sum_{j in labeled, label_j=c} Khat_ij
q_ic = V_ic / sum_c V_ic
confidence_i = max_c q_ic
```

The reported k values 13/5/3 are diagnostics for the paper's embeddings, not
defaults to force on different lab features. Test examples never enter this
graph, evidence matrix or gate.

### Adaptive gate

Follow supplement B.1 and Algorithms 2-3 over post-propagation beliefs:

1. Histogram unlabeled confidences into 100 bins between their minimum and maximum.
2. Smooth with a width-three moving average. Default: zero-padded, same-length
   convolution; this boundary convention is an explicit implementation assumption.
3. Include boundary peaks. Use the midpoint, rounded down, for a flat peak;
   break equal-height ties toward the lower bin index.
4. Select the tallest primary peak, then the tallest secondary peak separated
   by at least five bins and at least 15% of primary height.
5. For two qualifying peaks, take the left edge of the right-hand peak's bin.
   This follows the algorithm's stated bin boundary, not an invented valley-minimum search.
6. With no second peak, or constant scores, use base threshold 0.75. Empty
   unlabeled input admits no samples.
7. For each class, average its top floor(N/M) probabilities over the full train
   pool, following formal equation S1. Normalize these means by their largest value.
8. Set each class threshold to max(base threshold times its normalized mean, 1/M).
   Admit only unlabeled points strictly above their predicted-class threshold.

Keep full soft targets and each admitted point's confidence. Do not replace
targets with hard argmax labels. The source's Otsu/Bayesian path is not a
substitute for this histogram procedure.

### Classifier and loss

Use `Linear(input_width, 256) -> ReLU -> Linear(256, M)` with frozen input
representations. No encoder fitting is part of these runs.

Train for exactly 300 epochs with Adam, fixed learning rate 0.001, weight decay
0.03, betas (0.9, 0.999), epsilon 1e-8. Use the batch sizes and loss weights
from the experiment table in section 2 of this document.

```text
loss = mean_labeled CE(logits_i, true_label_i)
     + lambda * mean_admitted [confidence_i * KL(p_model_i || q_i)]
```

This follows the paper's displayed KL direction, which differs from upstream
`F.kl_div(log_model, target)` (that computes KL(target || model)). Test with
asymmetric probability vectors so the directions cannot be confused.
Likewise, mean(confidence times KL) is not mean(confidence) times mean(KL).

Default batch policy, where the paper is incomplete: reserve
`q = max(1, batch_size // 8)` labeled slots; traverse a freshly shuffled admitted
pool once per epoch in chunks of at most `batch_size - q`; fill labeled slots
by cycling freshly shuffled labeled examples, reshuffling at exhaustion. Keep
the final short batch. If no points are admitted, use shuffled labeled-only
batches and zero distillation loss; explicitly record that outcome.

Default selection policy: private NumPy RandomState(seed), classes in sorted
ID order, uniform sampling without replacement within each class. Generate
each budget's selection independently from that seed; do not claim nested
budgets or equality to unpublished author split files. Seed Python, NumPy and
PyTorch for model/training operations and record the software/hardware context.

Report the final epoch checkpoint. The paper does not clearly specify VAST
checkpoint selection; final-epoch selection is an explicit default. Test labels
are used for evaluation only, never checkpoint selection or hyperparameter tuning.

## 5. Outputs and execution status

Each run records dataset, label budget, one seed, representation identity/hashes,
class/sample mapping, selected labeled IDs, resolved configuration, source revision,
graph k/edge counts, propagation status/diagnostics, gate threshold/admission
counts, training losses, final checkpoint and final test accuracy.

Use separate output directories per dataset, representation, budget and seed.
Do not overwrite a completed run implicitly. A grid launcher must preflight
input manifests before scheduling compute. Local synthetic tests do not require
the lab's feature files; full experiments do.

Until the question below is resolved, no default propagation implementation is
approved for claims of reproduction. Independent components can be implemented
and verified; the full runner must report the unresolved status rather than
silently omit propagation or choose the public-code recurrence.

## 6. Propagation discrepancy: explanation and author question

### Plain-language explanation

The paper starts by giving each example a distribution over possible classes.
Its confidence is the largest probability in that distribution. A point is
frozen when confidence reaches 75%.

Under the written initialization, with one labeled example per class, no point
can initially be more than about 55% confident. With two examples per class,
the upper bound is about 71%. These are best-case mathematical ceilings, not
measured average accuracies.

The written propagation rule mixes existing evidence using positive weights.
After normalization, each new distribution is a weighted average of existing
distributions. A weighted average cannot exceed the largest value being mixed.
An individual point can become more confident, but the maximum confidence
anywhere in the dataset cannot increase beyond its initial ceiling.

Consequently no point reaches 75% at either small label budget. The written
stopping rule stops after a round in which no new point freezes, so it stops
after the first round. This does not mean beliefs stopped changing, and does
not mean all variants of label propagation have this limitation.

The reported 12-27 rounds therefore require a different or omitted detail.
For example, making labeled points certain about their own class would change
the argument, but that is only a possible explanation, not a verified fix.
Using one random seed instead of several does not affect this contradiction.

### Technical derivation

This is our derivation from the paper's literal formulas, not a claim that we
have executed the authors' unpublished experimental implementation.

**Assumptions:** symmetric nonnegative graph, no self-loops, raw RBF weights at
most 1, degree correction as above, r clean labels per class, prior 1/M for
every class, no overwrite of labeled rows with one-hot/strong evidence, full
unnormalized V sent additively, and the written no-new-freeze stopping rule.

For an edge of raw weight t, both endpoint degrees are at least t. Hence:

```text
Khat_ij <= t / sqrt(log(1 + t^2)) <= 1 / sqrt(log 2) = B
B = 1.2011224088..., for 0 < t <= 1.
```

The last inequality follows because t/sqrt(log(1+t^2)) is increasing on (0,1].
With r labeled examples of class c, its initial evidence contribution a_ic is
at most rB. The total prior across M classes is 1, so:

```text
q_ic = (1/M + a_ic) / (1 + sum_d a_id)
     <= (1/M + a_ic) / (1 + a_ic)
     <= (1/M + rB) / (1 + rB).
```

| Labels per class r | Ceiling for M=100 | Ceiling for M=200 | Freezing threshold |
|---:|---:|---:|---:|
| 1 | 0.55023 | 0.54796 | 0.75 |
| 2 | 0.70902 | 0.70755 | 0.75 |
| 4 | 0.82944 | 0.82858 | 0.75 |

The four-label bound is above 0.75, so this argument does not force one-round
termination at that budget. It also does not guarantee that any point freezes.

Let s_i be the sum of row V_i. The normalized additive update for a receiver is:

```text
q'_i = (s_i*q_i + sum_{j in senders} Khat_ji*s_j*q_j)
       / (s_i + sum_{j in senders} Khat_ji*s_j).
```

All mixing weights are nonnegative and sum to one after division. Thus every
coordinate of q'_i is bounded by the previous global maximum coordinate.
Frozen rows are unchanged. Inductively the initial global ceiling holds for
every round, independent of features or random seed under these assumptions.
At r=1 and r=2, no unlabeled receiver ever reaches 0.75; the first round has
zero newly frozen points and triggers the specified stopping condition.

### Question to resolve with Itai

> Which initialization, propagation update and stopping rules produced the
> paper's reported 12-27 rounds? Were labeled rows overwritten with one-hot or
> stronger evidence, was the prior subtracted during propagation, or was a
> different stopping criterion used? Please identify the source function and
> exact configuration for at least one one-label-per-class experiment.

The public source uses a different recurrence:

```text
V_new = alpha + fixed_labeled_contribution
        + K @ ((V_old - alpha) * unlabeled_sender_mask)
```

It also uses fixed iteration counts, with an optional decay factor. The
convex-combination argument above does not apply unchanged to that recurrence.
It cannot be selected as the reproduction rule without confirming the actual
experimental settings. No message has been sent to Itai as part of this work.

## 7. Validation and acceptance

1. **Source parity:** compare unchanged ported helpers against the pinned source
   on small inputs. For each correction, test the paper formula separately.
2. **Geometry:** compare sparse graph, minimum connected k, edge weights and
   degree correction against a tiny dense reference; include duplicate features
   and explicit self-edge exclusion. Reject zero-norm feature rows.
3. **Evidence:** hand-calculate additive class evidence and priors. Detect
   accidental max accumulation, prior overwrites and access to unlabeled truth.
4. **Propagation discrepancy:** verify the convex-combination identity and the
   confidence bound on synthetic examples. These are diagnostic tests of the
   literal paper, not approval to use that rule for reproduction. The explicit
   diagnostic mode is tested now; confirmed reproduction recurrence/termination
   tests will follow author clarification.
5. **Gate:** golden tests for two modes, boundary/flat/equal peaks, unimodal,
   constant and empty scores, top-class statistics and strict threshold equality.
6. **Training:** distinguish the two KL directions, validate per-sample weighting,
   labeled quota, empty distillation pool and 300-epoch final-checkpoint selection.
7. **Data:** verify labels/IDs/class maps; reject misaligned or mixed train/test
   arrays. Confirm ImageNet100 selects the intended subset.
8. **Single-seed behavior:** one invocation yields one setting/seed result; the
   default grid enumerates exactly nine settings at seed 0. Save split IDs;
   repeated CPU smoke runs with the same seed are reproducible. Report no
   multi-seed uncertainty statistics from this grid.
9. **Real-data smoke:** after propagation is resolved and feature alignment is
   available, run CIFAR-100 with one configured seed and inspect all intermediate
   diagnostics before launching the nine-run grid.

Progress states are distinct: specification complete; independent components
tested; propagation confirmed; lab-feature runs complete; published results
reproduced. Do not treat completion of an earlier state as evidence of a later one.
