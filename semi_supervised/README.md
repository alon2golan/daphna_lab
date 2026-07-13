# Semi- and Self-Supervised

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
- `algorithms/` research methods. `simclr_supcon.py` iterates between
                unsupervised SimCLR (full pool) and supervised SupCon (labeled
                subset) on a shared backbone.
- `labeling.py` strategies for choosing which pool samples to label. `RandomSelector`
                picks a random subset now; an active-learning selector (e.g.
                backed by `deep_al/`) can implement the same interface later.

## Running the SimCLR<->SupCon iteration

One process, one job (uses the CIFAR copies under `../data`):

```
python -m semi_supervised.algorithms.simclr_supcon \
    --dataset cifar100 --n_labeled 1000 \
    --num_iterations 3 --simclr_epochs 200 --supcon_epochs 100 \
    --output_dir runs/simclr_supcon_cifar100_l1000
```

Outputs, under `--output_dir`: `labeled_indices.npy`, a `backbone_iter*.pt` per
iteration, `backbone_final.pt`, and `train_/val_features.npy` (+ labels) from the
final backbone.

Requires the solo-learn dependencies in the project venv (lightning, hydra,
timm, ...) and a GPU for real runs.
