# Getting Started

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

## BAYES_MISP

BAYES_MISP is a greedy sample-selection method: for every point it maintains a per-class evidence
score built up from a kernel similarity matrix (`K`) over the points chosen so far, then repeatedly
picks the unlabeled point whose kernel row would most increase that evidence (i.e. the point that
"covers" the most information still missing from the current pick). It supports both an `rbf` and a
`tophat` kernel, a sparse or dense `K`, and several scoring rules for what "most increase" means.

Example BAYES_MISP script:
```
cd deep_al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name auto --initial_size 0 --budget 100 --sparse_K --kernel_type rbf --alpha 0.5
cd ../../
```

### BAYES_MISP arguments

| Argument | Default | Description |
|---|---|---|
| `--kernel_type` | `rbf` | Kernel used to build `K`: `rbf` (Gaussian) or `tophat` (indicator/ball). |
| `--sparse_K` | off | Build `K` as a sparse CSR matrix instead of dense. Recommended for large datasets. |
| `--k_sparsity_threshold` | `0.0` | Minimum kernel value kept when `--sparse_K` is set (smaller values are dropped as zero). |
| `--initial_sigma` | `1.0` | Bandwidth (`sigma`) for the `rbf` kernel. |
| `--alpha` | `0.5` | Baseline/prior evidence value every point starts with in the `C` matrix (ignored if `--local_alpha` is set). |
| `--calc_method` | `max` | Scoring rule for how a candidate point's contribution is computed: `max`, `entropy`, `margin`, or `margin_not_normalized`. |
| `--cont_method` | `positive` | How per-class contributions are aggregated: `positive` (clip negative contributions to 0), `abs` (use magnitude), `reg_sum_postive`/`reg_sum_abs` (unweighted sum variants). |
| `--distribution_cont_weight_method` | `weighted` | Whether each class's contribution is weighted by its current share of `C` (`weighted`) or summed unweighted (`equal`). |
| `--c_normalization` | `sum` | How `C` is normalized before scoring: `sum` (divide by row sum) or `softmax`. |
| `--class_weighting_method` | `none` | Optional per-class gain reweighting to counter class imbalance in what's been selected so far: `none`, `linear`, `square`, or `exp`. |
| `--local_alpha` | off | Initialize `C` from per-point priors (derived from `K`'s class clarity) instead of a constant `--alpha`. |
| `--local_alpha_oracle_method` | `entropy` | How per-point clarity is scored for `--local_alpha`: `entropy` or `max`. |
| `--alpha_lower_bound` / `--alpha_upper_bound` | `0.01` / `1.0` | Range the per-point priors are scaled into when `--local_alpha` is set. |
| `--kernel_build_batch_size` | `1024` | Row-chunk size used while building `K` (lower this if you hit out-of-memory errors). |
| `--matrix_dtype` | `float32` (sparse) / `float16` (dense) | Torch dtype for `K`/`C`: `float32` or `float16`. |

## Distillation & Propagation

Distillation and propagation are optional, opt-in training-time additions that make use of the
evidence matrix (`C_general`) an AL algorithm builds while selecting samples (currently only
BAYES_MISP and CoverageMatrixMethod-based methods expose this). They're independent of *which* AL
method chose the labeled set, but need one of those methods to be the active `--al` method so that
`C_general` exists to draw from.

**Distillation** turns `C_general` into a soft, per-class probability target for every point, and
adds a temperature-scaled KL-divergence loss (alongside the normal cross-entropy loss) that
encourages the model to also match those soft targets on unlabeled points the evidence matrix is
confident about. This gives the model extra training signal from unlabeled data without needing
ground-truth labels for it.

**Propagation** is an optional preprocessing step run before distillation: it spreads each labeled
point's evidence outward through `K` to its similar neighbors, so points that are close to a labeled
point (but weren't picked by the AL algorithm) also accumulate some evidence before soft targets are
computed. Both which points are allowed to *send* their evidence forward and which points are
allowed to *receive* it can be controlled independently, either by a confidence threshold or by
asking the AL algorithm itself which points it would currently pick.

Example: BAYES_MISP with distillation (no propagation):
```
cd deep_al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name auto --initial_size 0 --budget 100 --sparse_K --distillation_training --distillation_threshold 0.5
cd ../../
```

Example: BAYES_MISP with propagation feeding into distillation:
```
cd deep_al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name auto --initial_size 0 --budget 100 --sparse_K --distillation_training --propagate_c_labels --propagation_sender_mode high_confidence --propagation_sender_threshold 0.5
cd ../../
```

### Distillation arguments

| Argument | Default | Description |
|---|---|---|
| `--distillation_training` | off | Master switch: enables the soft-target KL loss during training. Everything else in this section is a no-op unless this is set. |
| `--distillation_threshold` | `0.5` | Minimum normalized max-probability a point needs to participate in distillation (points below this are excluded from the KL loss). |
| `--distillation_temperature` | `3.0` | Temperature used to soften both the model's predictions and the targets before computing the KL loss. Higher = softer/smoother targets. |
| `--distill_factor` | `1.0` | Scalar multiplier applied to the distillation loss term before it's added to the cross-entropy loss. |
| `--ck_c_normalization` | `sum` | How `C_general` is normalized into probabilities: `sum` (divide by row sum) or `softmax`. |

### Propagation arguments

| Argument | Default | Description |
|---|---|---|
| `--propagate_c_labels` | off | Enable the propagation preprocessing step before distillation targets are computed. Requires `--distillation_training`. |
| `--propagation_iterations` | `1` | Number of propagation rounds. Each round re-evaluates which points can send/receive before spreading evidence again. |
| `--propagation_sender_mode` | `all` | Which unlabeled points are allowed to send their evidence forward: `all`, `high_confidence` (threshold-gated), `al_constant` (the AL algorithm's most recent picks), or `al_updated` (re-ask the AL algorithm each iteration). |
| `--propagation_sender_threshold` | `0.5` | Confidence threshold for `--propagation_sender_mode high_confidence`. |
| `--propagation_sender_budget` | none | Optional cap on the number of senders for `high_confidence` mode (keeps only the top-K most confident points instead of thresholding). |
| `--propagation_receiver_mode` | `all` | Which unlabeled points are allowed to receive propagated evidence: same mode options as sender. Once a point is frozen out under `high_confidence` mode it stays frozen for the rest of the run. |
| `--propagation_receiver_threshold` | none | Confidence threshold for `--propagation_receiver_mode high_confidence`. |
| `--propagation_receiver_budget` | none | Optional cap on the number of receivers for `high_confidence` mode (keeps only the bottom-K least confident points instead of thresholding). |

**Note:** distillation and propagation are wired into the training loop but have only been validated
with CPU smoke tests on synthetic data so far — please run and sanity-check a small real experiment
before relying on them.
