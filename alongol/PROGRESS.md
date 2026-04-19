# Algorithmic Changes & Project Progress

> **Owner: Alon Golan (alongol)** — personal notes, not team-wide.
> Summary of work done in the noisy-label active learning project (Apr 2025).
> Intended for context handoff between chat sessions.

---

## Background

The project extends active learning (probcover, maxherding, random) to the noisy-label setting.
The key parameter is **τ (tau)** — the cost ratio of a noisy label vs an oracle label — and
**ν' (nu_prime)** — the effective noise fraction the algorithm can tolerate under that budget constraint.

The pipeline has two stages:

1. **ν' estimation**: Pre-compute how much noise each (method, budget) pair incurs at each τ.
2. **AL training**: Run `train_al.py` with `--tau` or `--target_nu`; the cache maps τ ↔ ν' at query time.

---

## Session Summary (Apr 2025)

### Problem Discovered: Cache Discrepancy

Running `estimating_delta_nu.py` directly (CLI) produced ν' ≈ 0.45 for random/budget=100/τ=0.5,
but the pre-built cache had ν' ≈ 0.52. Root cause: the old sweeper (`nu_prime_sweeper.py`) used
`runs=3` (averaged 3 binary searches per step internally), whereas the CLI used `runs=1`.

**Fix**: redesign the sweep to call `estimating_delta_nu.py` directly, once per seed.

---

### Redesign: ν' Sweep Infrastructure

**Before**: `nu_prime_sweeper.py` orchestrated multiple runs internally; opaque averaging.

**After**: each SLURM job = one call to `estimating_delta_nu.py` with a fixed `--seed`.
- Output: one JSON file per job — `results/cifar100/nu_prime_raw/{method}_{budget}_tau{tau}_seed{seed}.json`
- Separate averaging step reads all raw files and writes cache.

This makes every data point **transparent and reproducible**: raw files are never overwritten,
and the cache can be rebuilt at any time with `--average`.

---

### Changes to `estimating_delta_nu.py`

- Added **`--output_file`** CLI argument: saves `{"dataset","al_method","budget","tau","seed","nu_prime"}` as JSON atomically (write-to-tmp + rename).
- Added **`maxherding`** to `--al_method` choices.
- Fixed **`parquet_model_name` bug**: files on disk are named `resnet18` but the code was looking up `linear` (the model config name). Added remapping `"linear" → "resnet18"` at both `estimate_nu` call sites.

---

### New Script: `batch_utils/sweep_nu_prime.py`

Replaces the old sweeper for submitting ν' estimation jobs.

- **`--submit`**: SSHes to phoenix-gw and submits one SLURM job per (method, budget, τ, seed).
- **`--average`**: reads all matching raw files, averages seeds, upserts cache.
- **`--dry_run`** (with `--average`): prints table without writing cache.
- Grid configurable via `--methods`, `--budgets`, `--taus`, `--seeds`.
- Default grid: 3 methods × 3 budgets × 9 taus × 3 seeds = **243 jobs**.

---

### ν' Sweep Results

243 SLURM jobs completed successfully. Cache rebuilt from scratch.

**Cache coverage:**
- Methods: `probcover`, `maxherding`, `random`
- Budgets: 100, 200, 500
- τ: 0.1, 0.2, ..., 0.9 (measured, 3 seeds each) + oracle anchor τ=1.0 → ν'=0.0
- Oracle anchor is injected by `nu_prime_cache.py` automatically

**6 additional jobs** submitted for `random/budgets=[100,200]/τ=0.95/seeds=[0,1,2]`
to add a ~10% noise data point near τ=1.

---

### Changes to `batch_utils/sweep_train_al.py`

Added full CLI configurability so sweeps can be controlled without editing the file:

```
--tau         τ values to sweep (mutually exclusive with --target_nu)
--target_nu   target_nu values to sweep
--al          AL methods
--seed        seeds
--budget      budget
--cfg         config yaml
--submit      submit to SLURM
--local       run locally
```

Previously all these were hardcoded; editing was required for every new experiment config.

---

### Dashboard Improvements (`reporting/dashboard.py`)

Added three side-by-side filters to the **📈 Compare** tab:
- **Sweep(s)**: filter to specific sweep submission IDs
- **AL method(s)**: e.g. show only `probcover`
- **target_nu**: compare across noise levels (e.g. oracle vs τ=0.3)

All filters are additive; leaving a filter empty shows all values.

---

## Key Architectural Facts (for new sessions)

| Topic | Detail |
|-------|--------|
| Python | `python3` (python3.11), venv at `venv/bin/activate` |
| SLURM login node | `phoenix-gw`; sbatch at `/vol/slurm/phoenix/bindir/bin/sbatch` |
| Parquet files | `output/CIFAR100/cifar100_resnet18_{probcover,maxherding}_sym_all.parquet` |
| `random` method | does not need parquet (uses np.random.choice) |
| Cache location | `results/cifar100/nu_prime_cache.json` |
| Raw results | `results/cifar100/nu_prime_raw/` (243 files) |
| Cache API | `nu_prime_cache.py`: `upsert()`, `lookup_nu_for_tau()`, `lookup_tau_for_nu()` — interpolates linearly |
| Oracle anchor | τ=1.0 → ν'=0.0 injected automatically by the cache layer |
| `runs` param | must be 1 in `estimating_delta_nu.py` calls (old sweeper used 3 — was the source of the discrepancy) |

---

## Files Changed This Session

| File | Change |
|------|--------|
| `deep_al/pycls/noise/estimating_delta_nu.py` | Added `--output_file`, added maxherding, fixed parquet name `linear→resnet18` |
| `batch_utils/sweep_nu_prime.py` | New design: one job per seed, transparent raw files, `--submit`/`--average`/`--dry_run` |
| `batch_utils/sweep_train_al.py` | Added CLI args: `--tau`, `--target_nu`, `--al`, `--seed`, `--budget`, `--cfg`, `--submit`, `--local` |
| `reporting/dashboard.py` | Added Sweep / AL method / target_nu filters to Compare tab |
| `results/cifar100/nu_prime_cache.json` | Deleted stale data; rebuilt cleanly from 243 raw files |
| `results/cifar100/nu_prime_raw/` | 243 new JSON files (one per method × budget × tau × seed) |
| `USAGE.md` | Added noisy-label cheatsheet section |
| `PROGRESS.md` | This file — new |
