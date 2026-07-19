# Semi- and Self-Supervised

Workspace for semi-supervised learning ideas built on top of self-supervised
encoders. The base encoders and their training (SimCLR, SupCon, BYOL, ...) come
from the vendored [solo-learn](https://github.com/vturrisi/solo-learn) repo at
`../solo-learn`; this package holds the research methods built on top of them.

## Layout

- `algorithms/`  the current line of work. `joint_finetune.py` jointly
                 fine-tunes an existing SimCLR encoder with a supervised SupCon
                 term (labeled subset) plus an unsupervised SimCLR term (full
                 pool), sweeping the unsupervised weight, and scores conditions on
                 `val_acc1/5` + `val_knn_acc1/5`.
- `backends/`    adapters to external training frameworks. `solo_learn.py` is the
                 only module that imports solo-learn; `joint.py` backs
                 `joint_finetune`. Everything reaches solo-learn through here.
- `scripts/`     SLURM (`.sbatch`) and local (`.sh`) runners. They find the
                 project root themselves, so the exact cwd does not matter.
- `labeling.py`  strategies for choosing which pool samples to label
                 (`RandomSelector`, `class_counts`).
- `propagation.py` kernel label propagation (gaussian/laplacian/cosine) into hard
                 pseudo-labels. Used by the archived work below.
- `evaluation.py` one shared yardstick: offline linear-probe + weighted-kNN
                 top-1/5.
- `old/`         archived SimCLR<->SupCon concept work (shelved). See
                 `old/README.md`. Still runnable via
                 `python -m semi_supervised.old.<module>`.

## Running the joint fine-tuning

```
# local GPU
./scripts/run_joint_finetune.sh trained_models/simclr/cifar100/1000ep cifar100 10

# SLURM
sbatch --job-name=joint_simclr_cifar100_L10 \
    scripts/run_joint_finetune.sbatch trained_models/simclr/cifar100/1000ep cifar100 10
```

Positional args: `BASE_CKPT` (relative to `../solo-learn` or absolute),
`DATASET` (cifar10 | cifar100), `L` (per-class budget; `n_labeled = L *
num_classes`). Env knobs: `EPOCHS` (50), `UNSUP_WEIGHTS` (1,2,5,10), `SEED` (1),
`WANDB_OFFLINE=1`, `EXTRA="..."`. Logs to its own wandb project
`semi-supervised-joint`; results under `results/<dataset>/<embedding>/`.
