# Getting Started

> **Owner: Alon Golan (alongol)** — personal cheatsheet, not team-wide.

## Setup

Clone the repository
```
git clone https://github.com/alon2golan/daphna_lab
cd TypiClust
```

Create an environment using
```
conda create --name typiclust --file typiclust-env.txt
conda activate typiclust
pip install pyyaml easydict termcolor tqdm simplejson yacs
```

If that fails, it might be due to incompatible CUDA version.
In that case, try installing by
```
conda create --name typiclust python=3.7
conda activate typiclust
conda install pytorch torchvision torchaudio cudatoolkit=<CUDA_VERSION> -c pytorch
conda install matplotlib scipy scikit-learn pandas
conda install -c conda-forge faiss-gpu
pip install pyyaml easydict termcolor tqdm simplejson yacs
```
Select the GPU to be used by running
```
CUDA_VISIBLE_DEVICES=0
```

## Representation Learning
Both TypiClust variants and ProbCover rely on representation learning. 
To train CIFAR-10 on simclr please run
```
cd scan
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
cd ..
```
When this finishes, the file `./results/cifar-10/pretext/features_seed1.npy` should exist.

To save time, you can download the features of CIFAR-10/100 from here:

| Dataset          | Download link |
|------------------|---------------| 
|CIFAR10           | [Download](https://drive.google.com/file/d/1Le1ZuZOpfxBfxL3nnNahZcCt-lLWLQSB/view?usp=sharing)  |
|CIFAR100          | [Download](https://drive.google.com/file/d/1o2nz_SKLdcaTCB9XVA44qCTVSUSmktUb/view?usp=sharing)  |


and locate the files here:`./results/cifar-10/pretext/features_seed1.npy`.

## TypiClust - K-Means variant
To select samples according to TypiClust (K-Means) where the `initial_size=0` and the `budget=100` please run 
```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_rp --exp-name auto --initial_size 0 --budget 100
cd ../../
```


## TypiClust - SCAN variant
In this section we select `budget=10` samples without an initial set. 
We first must run SCAN clustering algorithm, as TypiClust uses its features cluster assignments.
Please repeat the following command  `for k in [10, 20, 30, 40, 50, 60]`

```
cd scan
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --num_clusters k
cd .. 
```
You can use other representations and change the path in the file `deep-al/pycls/datasets/utils/features.py`.
Then, you can run the active learning experiment by running
```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_dc --exp-name auto --initial_size 0 --budget 10
cd ../../
```

## ProbCover
Example ProbCover script
```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al probcover --exp-name auto --initial_size 0 --budget 10 --delta 0.75
cd ../../
```

## DCoM
Example DCoM script:
```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al dcom --exp-name auto --initial_size 0 --budget 10 --initial_delta 0.75
cd ../../
```

You can add the `a_logistic` and `k_logistic` parameters to the run using `--a_logistic` and `--k_logistic`.

---

# Noisy-Label Experiments — Cheatsheet

> This section documents the noisy-label / ν'-budget workflow added on top of the base active-learning code.

---

## Environment

```bash
cd /cs/labs/daphna/alongol/projects/daphna_lab
source venv/bin/activate   # python3.11
```

---

## Train AL (single run, local debug)

```bash
python3 deep_al/tools/train_al.py \
  --cfg deep_al/configs/cifar100/al/RESNET18.yaml \
  --al probcover \
  --budget 100 \
  --seed 1 \
  --tau 0.5
```

**Key flags:**
| Flag | Meaning |
|------|---------|
| `--tau 0.5` | Cost ratio τ = cost_noisy / cost_oracle; ν' looked up from cache |
| `--target_nu 0.2` | Target noise fraction; τ looked up from cache. **Mutually exclusive with `--tau`** |
| `--al` | `probcover` / `random` / `maxherding` |
| `--budget` | Labeled-set budget per round |
| `--seed` | Random seed |

---

## Train AL Sweep — SLURM

```bash
# dry-run (prints job plan, no submission)
python3 batch_utils/sweep_train_al.py --tau 0.2 0.5 0.8 1.0

# submit
python3 batch_utils/sweep_train_al.py --tau 0.2 0.5 0.8 1.0 --submit

# target_nu instead of tau
python3 batch_utils/sweep_train_al.py --target_nu 0.0 0.1 0.2 0.3 0.5 --submit

# narrow to specific methods / seeds / budget
python3 batch_utils/sweep_train_al.py \
  --tau 0.2 0.5 0.8 1.0 \
  --al probcover maxherding \
  --seed 1 2 3 \
  --budget 200 \
  --submit
```

**All CLI flags:**
```
--tau         τ values to sweep              (mutually exclusive with --target_nu)
--target_nu   target_nu values to sweep
--al          AL methods    (default: probcover random)
--seed        seeds         (default: 1 2 3)
--budget      budget        (default: 100)
--cfg         config yaml   (default: cifar100/al/RESNET18.yaml)
--submit      submit to SLURM via SSH to phoenix-gw
--local       run sequentially on this machine
```

---

## ν' Cache — Sweep and Average

The cache maps **(method, budget, τ) → ν'** and lives at:
`results/cifar100/nu_prime_cache.json`

`train_al.py` uses it to resolve `--tau` → ν' or `--target_nu` → τ.

### Run sweep (243 jobs: 3 methods × 3 budgets × 9 taus × 3 seeds):
```bash
python3 batch_utils/sweep_nu_prime.py --submit
```

### After jobs finish — average seeds and write cache:
```bash
python3 batch_utils/sweep_nu_prime.py --average

# preview without writing
python3 batch_utils/sweep_nu_prime.py --average --dry_run

# narrow scope (e.g. after adding tau=0.95)
python3 batch_utils/sweep_nu_prime.py --methods random --budgets 100 200 --taus 0.95 --average
```

### Reset and rebuild cache:
```bash
rm results/cifar100/nu_prime_cache.json
python3 batch_utils/sweep_nu_prime.py --average
```

### Print cache as table:
```bash
python3 - << 'EOF'
import json
from pathlib import Path
cache = json.loads(Path('results/cifar100/nu_prime_cache.json').read_text())
taus = sorted({t for m in cache.values() for b in m.values() for t in b.keys()}, key=float)
header = f"{'method':<12} {'budget':>7}  " + "  ".join(f"t={float(t):.2f}" for t in taus)
print(header); print("-" * len(header))
for method in sorted(cache):
    for budget in sorted(cache[method], key=int):
        vals = "  ".join(f"{cache[method][budget][t]:.3f}" if t in cache[method][budget] else "  -- " for t in taus)
        print(f"{method:<12} {budget:>7}  {vals}")
EOF
```

**Raw per-seed files** (safe to re-average any time):
`results/cifar100/nu_prime_raw/{method}_{budget}_tau{tau}_seed{seed}.json`

**Current cache coverage (as of Apr 2025):**
- Methods: probcover, maxherding, random
- Budgets: 100, 200, 500
- τ: 0.1 – 0.9 (step 0.1) + oracle anchor τ=1.0 → ν'=0.0
- 3 seeds averaged per point

---

## Dashboard

```bash
streamlit run reporting/dashboard.py
```

| Tab | What it shows |
|-----|--------------|
| 📊 Sweeps | All sweep runs, status, hide stale sweeps |
| 📈 Compare | Accuracy curves with ±σ bands. **Filters: Sweep, AL method, target_nu** |
| 🔍 Run Details | Per-seed curves and logs for one run |
| 📉 ν' Cache | Interactive ν'(τ) curves — select methods and budgets |
| 📤 Export | Download result CSVs |
| 🚀 Launch | Submit sweeps from browser |

---

## Key Files

| File | Purpose |
|------|---------|
| `batch_utils/sweep_train_al.py` | Submit/run train_al sweeps |
| `batch_utils/sweep_nu_prime.py` | Submit/run ν' sweeps + average |
| `deep_al/tools/train_al.py` | Main AL training script |
| `deep_al/pycls/noise/estimating_delta_nu.py` | Single ν'(τ) measurement |
| `deep_al/pycls/noise/nu_prime_cache.py` | Cache read/write API |
| `reporting/dashboard.py` | Streamlit dashboard |
| `results/cifar100/nu_prime_cache.json` | ν' cache (do not edit by hand) |
| `results/cifar100/nu_prime_raw/` | Per-seed raw ν' results |
| `output/CIFAR100/*.parquet` | Precomputed AL label sets (resnet18, probcover + maxherding) |

---

## Common Gotchas

- Use **`python3`**, not `python` — the venv is python3.11.
- Parquet files for AL index lookup are named **`resnet18`**, not `linear`. This remapping is handled automatically in `estimating_delta_nu.py`.
- **`random`** AL method does not need a parquet file (uses `np.random.choice`).
- `--tau` and `--target_nu` are **mutually exclusive** in `train_al.py`.
- Cache only covers budgets **100, 200, 500** — other budgets will fail the cache lookup.
- SLURM jobs must be submitted from **phoenix-gw**; the sweep scripts SSH there automatically.
