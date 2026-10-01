# Source reuse and paper corrections

Author source: [itaidavid1/bayes_al_2025](https://github.com/itaidavid1/bayes_al_2025/tree/3d4cbd7d543b1d1290799bdfb02687519af3a264),
pinned to `3d4cbd7d543b1d1290799bdfb02687519af3a264` (2026-07-02).
The upstream MIT notice is preserved in `LICENSE.upstream`.

## Copied without functional changes

`upstream.py` contains two extracted definitions. Their ASTs were compared with
the pinned originals and are identical; imports are local to the extracted file.

| Definition | Original location |
|---|---|
| `_apply_degree_normalization_sparse` | `deep-al/pycls/al/kernel_utils.py`, lines 39-103 |
| `MLPClassifierTorch` | `deep-al/pycls/core/builders.py`, lines 113-159 |

The original kernel helper's prose describes its log mode imprecisely; its
actual code uses `sqrt(log(1 + degree_i * degree_j))`, as required by the paper.
We call only that mode and remove self-edges beforehand.

## Adapted source logic

- `inference.seed_evidence`: upstream `train_al.propagate_labels` class-wise
  labeled contribution accumulation, with symmetric CSR column sums and the
  paper's prior. Avoids the other source initializer's maximum/overwritten prior.
- `inference.prepare_targets`: sum normalization, max confidence, predicted-class
  lookup and top-class mean scaling from `prepare_distillation_targets`.
  Added paper histogram gate, chance floor and labeled exclusion.
- `inference.propagate_literal`: upstream sparse evidence multiplication and
  frozen-row structure; changed to the paper's full-evidence additive recurrence
  and no-new-freeze stopping. This is explicitly an **unconfirmed diagnostic**.
- `training.train_classifier`: MLP, Adam hyperparameters and separated
  CE/distillation structure from `train_mlp_with_distillation`. Fixed hidden
  width 256, exactly configured epochs (default 300), no early stopping or
  best-validation selection, and a new balanced sampler.
- `training.distillation_loss`: log-softmax/per-example KL structure from
  `pycls/core/losses.py:distillation_loss`. Direction is `KL(model || target)`
  per the paper, temperature is 1, and confidence multiplies each example before
  reduction. The author code instead uses `KL(target || model)` and its training
  path multiplies two separate means.
- `data.select_labeled`: class-wise random selection from `Sampling.random_1c`,
  extended to 1/2/4 labels, a private seed-controlled RNG and saved sample IDs.

## New infrastructure required by the specification

The connected-minimum-k union graph, histogram heuristic, balanced batch sampler,
canonical manifest validation, caching, CLI, outputs and tests are new code.
Upstream threshold-based all-pairs sparsification cannot implement the paper's
graph criterion by configuration alone. Its fixed-k batched search informed the
small reference path, but was not copied as a production graph implementation.

FAISS uses exact `IndexFlatL2` in a subprocess to isolate its native OpenMP runtime
from PyTorch. The dense SciPy backend is a small-data reference only. At a tied
kth-neighbor boundary FAISS can select a different equally distant neighbor than
the dense backend; self removal is by ID, returned ties are ordered by sample
index, and graph caches record backend identity. Near-tie floating-point or
version changes can alter a graph; do not claim cross-platform bitwise identity.

## Unresolved scientific item

There is no confirmed reproduction propagation mode. The default runner refuses
to train until a mode is explicitly chosen. `literal-paper` executes the equations
for diagnostics and writes `confirmed_reproduction: false` into its outputs and
checkpoint. See the specification's section 6 for the confidence-bound proof and
the question for Itai. No inferred repair, one-hot clamping or author-code mode
has been silently introduced.

## Lab integration and launch scripts

The lab repository supplies the package location, current feature-bank paths and
dataset conventions. Its legacy noisy-AL runner and image wrappers are not copied
into this feature-only runner. The new `onboarding/VAST_smoke.sh` and
`VAST_full.sh` follow existing onboarding Slurm resource/path conventions.
The synthetic smoke-data generator is new and does not measure paper accuracy.
