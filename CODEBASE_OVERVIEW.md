# Codebase Overview

This repository implements active learning methods from TypiClust, ProbCover, and DCoM, with extensions around MISP and BAYES_MISP. The core training/selection loop lives under `deep_al/`, while `scan/` provides self-supervised feature extraction used to build embeddings and kernels.

## Top-Level Structure

- `deep_al/`: Main active learning framework and experiments.
- `scan/`: Self-supervised learning (SCAN/SimCLR) used to generate features.
- `bayes_misp_runner.py`: Generates command lines for AL experiments.
- `USAGE.md`, `README.md`: High-level usage and paper references.
- `results/`, `stats_parquets/`, `visual_outputs/`: Outputs and analysis artifacts.

## Core Training/Selection Flow

Entry point:

- `deep_al/tools/train_al.py`: CLI for training and active learning episodes. It:
  - Loads datasets and features.
  - Builds `ActiveLearning` with `train_labels` and initial labeled set.
  - Runs an episode loop: train/eval, then query new samples.

Sampler dispatch:

- `deep_al/pycls/al/ActiveLearning.py`:
  - Picks a sampler based on `cfg.ACTIVE_LEARNING.SAMPLING_FN`.
  - Instantiates methods like `ProbCover`, `MaxHerding`, or `BAYES_MISP`.

Feature loading:

- `deep_al/pycls/datasets/utils/features.py`:
  - Loads pretrained embedding matrices used to build kernels (RBF/top-hat).

## Active Learning Methods (Key Files)

- `deep_al/pycls/al/prob_cover.py`: Classic ProbCover using a graph built by thresholding distances.
- `deep_al/pycls/al/maxherding.py`: MaxHerding with kernel matrices (RBF, TopHat, etc.).
- `deep_al/pycls/al/BAYES_MISP.py`: Kernel-based MISP variant with optional Bayesian priors.
- `deep_al/pycls/al/Prob_Misp.py`, `MISP_probcover.py`, `MAX_MISP.py`, `ALL_MISP.py`: Related variants.

## How BAYES_MISP Fits In

`deep_al/pycls/al/BAYES_MISP.py` is a kernel-based selection method that can behave like classic ProbCover or MaxHerding depending on config:

- Kernel construction happens in `__init__`:
  - Computes pairwise distances and builds a kernel matrix (RBF or TopHat).
  - Optionally sparsifies K.
- Bayesian priors are injected only if `LOCAL_ALPHA` is enabled:
  - `init_C()` calls `prior_selection.compute_clarity_kp()` to build per-sample priors.
- Selection logic in `select_samples()`:
  - `diff_method` controls behavior:
    - `prob_cover` and `max_herding` bypass alpha (alpha=0) and use `batched_diffs`.
    - Other diff methods use Bayesian-style updates.

Key configuration knobs (from `deep_al/tools/train_al.py`):

- `--al bayes_misp`: select BAYES_MISP sampler.
- `--diff_method prob_cover` or `--diff_method max_herding`: classic behavior paths.
- `--kernel_type rbf|tophat`: similarity kernel type.
- `--local_alpha`: enables Bayesian priors (omit for classic mode).
- `--initial_sigma`, `--initial_delta`: kernel scale/threshold parameters.

Important caveat:

- In `BAYES_MISP`, even `prob_cover`/`max_herding` paths still update `C` by the chosen sample’s true class label. If you want fully label-agnostic classic coverage, you may need to:
  - Pass a single dummy class for all labels, or
  - Modify `BAYES_MISP` to use a single coverage vector instead of per-class `C`.

## What to Focus On Next

If the goal is classic ProbCover/MaxHerding on a similarity matrix K (without Bayesian priors):

- Start with `deep_al/pycls/al/BAYES_MISP.py`:
  - Replace or inject your precomputed K where `self.K_general_dense_cpu` is built.
  - Ensure `LOCAL_ALPHA` is disabled.
- Compare with `deep_al/pycls/al/maxherding.py` and `deep_al/pycls/al/prob_cover.py` for baseline behavior.
- Use `deep_al/pycls/al/ActiveLearning.py` and `deep_al/tools/train_al.py` to wire configs and CLI flags.

## Recent Session Notes (ProbCover/MaxHerding on CIFAR)

This section captures the latest debugging context so a new session can resume quickly.

### Scripts Reviewed

- `sbatch_cifar100_probcover_kernel.sh`:
  - Fixes needed:
    - Use `cd deep_al/tools` (not `deep_al/tools`).
    - Use `--eval_model_type` (not `--eval_model`).
  - Intended flags:
    - `--al bayes_misp`
    - `--diff_method prob_cover`
    - `--kernel_type tophat`
    - `--initial_delta 0.65`
  - This uses BAYES_MISP in a non-Bayesian mode (no `--local_alpha`), but still label-aware.

- `sbatch_cifar100_maxhearding.sh`:
  - Fixes needed:
    - `--diff_method max_herding` (typo in script used `max_hearding`).
  - Intended flags:
    - `--al bayes_misp`
    - `--kernel_type rbf`
    - `--initial_sigma 1.0`

### Runtime Errors Observed

- Overflow error in `deep_al/pycls/al/BAYES_MISP.py` at `initial_sparse_index` when running dense K on CIFAR-100:
  - `OverflowError: cannot convert float infinity to integer`
  - Root cause: `self.K_general.sum()` in float16 overflowed to `inf`.
- CSR slicing error when using `--sparse_K`:
  - `RuntimeError: Sparse CSR tensors do not have strides`
  - Root cause: `batched_diffs` assumes dense slicing; CSR tensors do not support `K[start:end]`.

### Code Fix Applied

- File: `deep_al/pycls/al/BAYES_MISP.py`
- Change: compute `K` sum in float64 when building `initial_sparse_index` to avoid overflow.
- Effect: allows running dense K without `--sparse_K` for CIFAR-10/100.

### Expected Safe Runs (No Sparse K)

- CIFAR-10 ProbCover (tophat): `--diff_method prob_cover`, no `--sparse_K`.
- CIFAR-100 ProbCover (tophat): same.
- CIFAR-100 MaxHerding (rbf): `--diff_method max_herding`, no `--sparse_K`.

### Important Caveat

Even with `diff_method prob_cover` / `max_herding`, `BAYES_MISP` updates coverage per true class label. If fully label-agnostic behavior is needed, `BAYES_MISP` should be adjusted to use a single coverage vector, or labels should be made uniform.
