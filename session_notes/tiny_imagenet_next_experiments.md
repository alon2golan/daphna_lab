# Tiny ImageNet Next Experiments

## Goal

The next experiment should test whether our main CIFAR conclusion also holds on
Tiny ImageNet:

> MaxHerding behavior is explained more by the effective graph degree than by
> locality itself.

Tiny ImageNet is a useful next dataset because it is larger and has more
classes than CIFAR10 and CIFAR100:

- 100,000 training images
- 10,000 validation images
- 200 classes
- 64x64 images

This makes it a stronger test of whether the degree-based explanation is
dataset-specific or more general.

## Experimental Setup

Use the same validated path as our CIFAR experiments:

- Active learning method: MaxHerding
- Evaluation classifier: 1-nearest neighbor
- Representation: fixed SimCLR/SSL features
- Kernel: RBF
- Cutoff: ProbCover-style distance cutoff
- Initial labeled set: 0
- Budget per round: 200 labels
- Episodes: 10 total labeled-set sizes from 200 to 2000
- Seed: start with seed 3 only

The budget range is chosen to match the same low-budget logic used for
CIFAR100. CIFAR100 has 100 classes and we used 100 to 1000 labels, which is
roughly 1 to 10 labels per class. Tiny ImageNet has 200 classes, so 200 to 2000
labels gives the same rough labels-per-class regime.

## Delta Calibration

Tiny ImageNet does not have a paper-provided ProbCover delta in our current
setup. We should therefore calibrate delta first using the existing automatic
delta-selection method.

The delta probe writes:

```text
results_summaries/maxherding_tinyimagenet_density_selection/tinyimagenet_delta_seed3.txt
```

After this value is known, it should be passed to the MaxHerding jobs through
`DELTA_OVERRIDE`.

## K Sweep

Start with a compact K sweep:

- 10
- 60
- 250
- 500
- 1000
- 1500
- 3000
- 5000
- 10000

This range is designed to answer several questions:

- Do very small K values make the graph too sparse?
- Does the CIFAR100 useful region around K=1440 transfer approximately?
- Does the CIFAR10 useful region around K=3500 transfer approximately?
- Does Tiny ImageNet need larger K because it has more points and classes?
- Do very large K values become global-like again?

Earlier candidate values were:

- 480
- 1440
- 3500
- 7000
- 10000
- 20000

We replaced them with the current sweep because we want more resolution in the
small and medium K range before spending compute on extremely large K values.

## Expected Outcomes

If Tiny ImageNet behaves like CIFAR10 and CIFAR100, we expect:

- Small K will produce low effective degree and poor accuracy
- Medium or medium-large K will be best
- Very large K will become closer to global sigma
- Effective degree will explain performance better than K itself

If this pattern does not appear, then the degree explanation may depend on the
dataset, the feature representation, or the scale of the label space.

## Outputs To Add To The Research Summary

After the runs finish, we should produce:

- Final accuracy versus K
- Effective graph degree versus final accuracy
- Selected-point degree divided by dataset-average degree
- Global sigma versus best local K accuracy curves by budget
- A short comparison to the CIFAR10 and CIFAR100 patterns

The most important figure for the summary is effective degree versus final
accuracy. This directly tests whether the degree-based interpretation
generalizes.

## Scripts

The sbatches are under:

```text
scripts/sbatch/tinyimagenet_degree_study/
```

The dedicated Tiny ImageNet low-budget config is:

```text
deep_al/configs/tinyimagenet/al/RESNET18_LOW_BUDGET.yaml
```
