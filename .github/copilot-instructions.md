# Copilot Instructions for Daphna Lab

This repository contains implementations of active learning strategies for image classification, including TypiClust, ProbCover, and DCoM. Below are key details to work effectively in this codebase.

## Setup and Environment

- **Python**: Use Python 3.11+ (works with 3.9+)
- **Virtual Environment**: Create with `virtualenv -p python3.11 venv` in a parent directory, symlink as `./venv` in repo root
- **Installation**: Run `pip install -r deep_al/requirements.txt` (includes linting tools: black, flake8, isort)
- **GPU**: Select GPU via environment variable: `export CUDA_VISIBLE_DEVICES=0`

## Building, Testing, and Linting

### Linting
```bash
# Run code formatter and linters
black deep_al/ scan/
flake8 deep_al/ scan/
isort deep_al/ scan/
```

### Testing
The repository includes limited tests in `deep_al/tools/`:
```bash
# Run individual test files
cd deep_al/tools
python test_model.py
python test_classes_correlations.py
```

The main validation approach is through full experimental runs (not unit test suites). Run experiments with:
```bash
cd deep_al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_rp --exp-name auto --initial_size 0 --budget 100 --eval_model_type from_images
```

## High-Level Architecture

The codebase implements a factory-pattern active learning framework with two main modules:

### `deep_al/` - Active Learning Module
- **Core Logic**: `pycls/al/active_learning.py` - Orchestrator using factory pattern to dispatch to concrete AL strategies
- **Strategies**:
  - `typiclust.py` - Selects typical (dense) samples from K clusters
  - `prob_cover.py` - Covers unlabeled data with δ-radius balls
  - `DCoM.py` - Coverage with dynamic radius list updated by model training feedback
  - `maxherding.py` - Herding-based selection
  - `Prob_Misp.py` - Probabilistic MISP variant
  - `Sampling.py` - Low-level sampling algorithms (CoreSet, Adversary)
- **Utilities**: 
  - `pycls/core/config.py` - YACS-based configuration system
  - `pycls/datasets/utils/features.py` - Feature loading with support for different representation tags
  - `pycls/models/` - Neural network models (ResNet, VGG)
- **Tools**: `tools/train_al.py` - Main training loop; `tools/ensemble_al.py` - Ensemble methods

### `scan/` - Self-Supervised Clustering Module
- **Representation Learning**: 
  - `simclr.py` - SimCLR pretext task
  - `scan.py` - Self-supervised clustering algorithm
  - `moco.py` - Momentum Contrast implementation
- **Features**: Extract deep representations used by AL strategies
- **Configuration**: YAML-based config in `configs/`

### Data Flow
1. Load features from pre-computed representation (SimCLR, SCAN, or MoCo)
2. Initialize labeled set (random or empty)
3. Iterate: Train model → Apply AL strategy → Sample from unlabeled pool → Update features if needed
4. Metrics tracked in `output/` with experiment name (auto-generated or custom via `--exp-name`)

## Key Conventions

### Configuration System (YACS)
- All experiments use YAML configs in `configs/{dataset}/{task}/` (e.g., `configs/cifar10/al/`)
- Config overrides via CLI: `--cfg {config.yaml}` (required), `--budget 100`, `--seed 1`, `--initial_size 0`
- Special parameters for algorithms:
  - ProbCover/DCoM: `--initial_delta 0.75` (ball radius)
  - DCoM only: `--a_logistic`, `--k_logistic` (logistic parameters)
- `EXP_NAME` in configs: `"auto"` generates timestamp-based folder; custom names via CLI override

### Features and Representation Tags
- Features stored in `scan/results/{dataset}/pretext/features_{seed}.npy`
- Support for multiple representation sets via `--features_tag` (e.g., `"alon"`, `"itai"`)
- Feature loading handles different paths per contributor in `pycls/datasets/utils/features.py`
- Download pre-computed CIFAR features from [README.md](README.md) to skip representation learning

### Algorithm Implementation Pattern
Each AL strategy (e.g., `TypiClust`) follows this pattern:
```python
class TypiClust:
    def __init__(self, data_obj, cfg, train_labels=None, lset=None, plotting_manager=None):
        # Initialize with data and config
    
    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None, checkpoint_file=None):
        # Return indices of samples to label next
        # checkpoint_file used by DCoM to update delta list
```
Strategies delegate low-level sampling via `Sampling` module (CoreSet, Adversary, etc.).

### State Updates and Checkpointing
- DCoM maintains and updates `delta_list` (per-point radii) based on model predictions
- After each training round, checkpoint model and pass to `sample_from_uSet()` for delta updates
- Training state saved in `output/{exp_name}/logs/` for resumption

### Code Style
- **Format**: Black (line length 88)
- **Linting**: Flake8 for style issues, isort for imports
- **Logging**: Use `logging.get_logger(__name__)` from `pycls/utils/logging.py` (not print)
- **Config access**: Use global `cfg` from `pycls/core/config.py` after `setup_cfg(cfg, args)`
- **Imports**: Absolute imports at repo root (sys.path includes repo root in `train_al.py`)

### Data Handling Conventions
- Datasets use YACS config paths and loader factory in `pycls/datasets/data.py`
- Support for CIFAR-10, CIFAR-100, TinyImageNet, scenario-specific datasets
- Evaluation models can train from raw images (`--eval_model_type from_images`) or pre-extracted features

### Output and Metrics
- Experiments save to `output/{exp_name}/` with structured subdirectories
- Metrics tracked via `TrainMeter`, `ValMeter`, `TestMeter` in `pycls/utils/meters.py`
- Plotting manager (`pycls/utils/plotting_state.py`) handles visualization state
- Results can be extracted via `extract_stats_from_dir.py` or `bayes_misp_runner.py`

## Cluster Job Submission (SLURM)

Submit jobs from `phoenix-gw.cs.huji.ac.il`:
```bash
sbatch /path/to/sbatch_script.sh
```

Track progress:
```bash
squeue -u $USER
```

Example template: See `onboarding/sbatch_print_arguments.sh` and `onboarding/sbatch_gpu_example.sh`

## Quick Start Commands

**TypiClust (K-Means variant)**:
```bash
cd deep_al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_rp --exp-name auto --initial_size 0 --budget 100
```

**TypiClust (SCAN variant)** - Requires SCAN first:
```bash
# Step 1: Run SCAN for each k in [10, 20, 30, 40, 50, 60]
cd scan
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --num_clusters 10
# Step 2: Run AL with clustering features
cd ../deep_al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_dc --exp-name auto --initial_size 0 --budget 10
```

**ProbCover**:
```bash
cd deep_al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al probcover --exp-name auto --initial_size 0 --budget 10 --delta 0.75
```

**DCoM**:
```bash
cd deep_al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al dcom --exp-name auto --initial_size 0 --budget 10 --initial_delta 0.75
```

## Key Papers and References

- **TypiClust**: [Active Learning on a Budget](https://arxiv.org/abs/2202.02794)
- **ProbCover**: [Active Learning Through a Covering Lens](https://arxiv.org/abs/2205.11320)
- **DCoM**: [Active Learning for All Learners](https://arxiv.org/abs/2407.01804)
- Base frameworks: [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification), [Deep-AL](https://github.com/acl21/deep-active-learning-pytorch)
