# Research Summary Draft: Local Sigma, Density, and MaxHerding

## Document Goal

This document summarizes the research process and main conclusions of the
project. It is written for a reader who may not know active learning or the
details of our experiments. The focus is on the scientific story: what we tried,
why we tried it, what the results showed, and what should be studied next.

## Executive Summary

We studied whether local kernel scaling can improve MaxHerding active learning.
The original idea came from the Local Sigma paper: instead of using one global
kernel scale for the whole dataset, each point gets a local scale based on its
neighborhood. We expected this might help balance dense and sparse regions of
the data.

The main conclusion so far is that locality alone does not explain most of the
behavior. The effective graph degree induced by the kernel appears to be more
important. Small K creates graphs that are too sparse and hurts accuracy. Very
large K behaves more like global sigma. Medium-degree regimes are usually best,
but the gains are modest. Density normalization can help at very low budgets,
but the benefit is budget-dependent and simple decay schedules did not solve the
problem.

The next research direction should test whether controlling degree directly is
more robust than choosing K, and whether this degree-based explanation holds on
datasets beyond CIFAR10 and CIFAR100.

## 1. Background: Active Learning

In standard supervised learning, we usually assume that we already have a large
labeled training set. In many real problems this assumption is expensive or
unrealistic. Images, medical scans, satellite data, biological measurements, and
many other datasets may be easy to collect but expensive to label.

Active learning studies how to train useful models while asking for as few
labels as possible.

The basic idea is:

- Start with a large unlabeled dataset
- Choose a small number of examples to label
- Train or evaluate a model using those labeled examples
- Use the current information to choose the next examples to label
- Repeat until the labeling budget is used

The goal is not only to get high accuracy, but to get high accuracy with a small
number of labels.

### 1.1 Pool-Based Active Learning

The setting in this project is pool-based active learning.

There is a fixed unlabeled pool:

```text
U = all unlabeled examples
```

At each active learning round, the algorithm chooses a batch of points from the
pool to label. These points move from the unlabeled set to the labeled set:

```text
L = labeled examples
U = remaining unlabeled examples
```

The algorithm then evaluates how useful the labeled set is, usually by training
or applying a classifier and measuring accuracy.

### 1.2 CIFAR10 And CIFAR100

CIFAR10 and CIFAR100 are standard image-classification benchmark datasets. They
are commonly used because they are small enough to run many experiments, but
still complex enough to test whether an algorithm behaves sensibly.

CIFAR10 contains:

- 60,000 color images
- 50,000 training images
- 10,000 test images
- 10 object classes
- 32x32 pixel images

The 10 CIFAR10 classes are broad categories such as airplane, automobile, bird,
cat, deer, dog, frog, horse, ship, and truck.

CIFAR100 has the same number and size of images, but the labels are split into
100 classes instead of 10.

This makes CIFAR100 harder:

- There are fewer examples per class
- Classes are more fine-grained
- A small label budget covers fewer examples per class
- Active learning selection must spread labels across many more categories

In this project, CIFAR10 is useful as the simpler benchmark, while CIFAR100 is a
harder test of whether the method can cover many classes and more complex
feature-space structure.

### 1.3 Budget

The budget is the total number of labels the algorithm is allowed to request.
The examples below use CIFAR10 and CIFAR100, which were introduced above.

For example:

- CIFAR10 budget 100 means the algorithm can label only 100 training examples
- CIFAR100 budget 1000 means the algorithm can label only 1000 training
  examples.

The budget is usually much smaller than the full dataset. This is why the choice
of which points to label matters.

There is no universal numeric definition of low, mid, and high budget in active
learning. The terms are usually relative to the dataset, the number of classes,
the model, and the difficulty of the task. For this project, we use them as
practical regimes rather than fixed thresholds.

Low budget means that only a very small number of labels is available compared
with the dataset size and the number of classes. In this regime, each selected
point matters a lot. A bad selection can leave whole classes or regions of the
feature space uncovered. In our experiments, this is the main regime we study:
10 to 100 labels for CIFAR10 and 100 to 1000 labels for CIFAR100.

Mid budget means that the labeled set is large enough to cover many common
regions, but still far from the full dataset. The algorithm still matters, but
the effect of each individual selected point is smaller.

High budget means that many labels are available relative to the task. At this
stage, the marginal value of each additional selected point may be smaller
because the labeled set already covers more of the dataset. However, this does
not mean that all active learning methods behave similarly. Some methods are
designed specifically for higher-budget regimes, and the best strategy can
change as the budget grows.

We decided to focus on the low-budget regime because this is where active
learning is most important. If labels are very expensive, the main question is
how much can be learned from the first few selected points. It is also the
regime where kernel structure, density, and redundancy should matter the most.

Our low-budget experiments therefore use:

- CIFAR10 used 10 labels per round from 10 to 100 labels
- CIFAR100 used 100 labels per round from 100 to 1000 labels

### 1.4 What Makes A Good Active Learning Selection?

A good selected set should be informative and representative.

Important selection goals include:

- Cover the main regions of the data
- Avoid choosing many redundant points
- Include examples from different classes or clusters
- Include rare or sparse regions when they matter
- Improve downstream classification accuracy

There is a natural tension:

- Dense regions contain many points and are often representative
- Sparse regions may contain rare but important structure
- Selecting only dense points can be redundant
- Selecting too many sparse/outlier points can waste budget

Much of this project is about understanding this dense-versus-sparse tradeoff.

### 1.5 Feature Space

The algorithms in this project usually do not operate directly on raw image
pixels. Instead, each image is represented by a feature vector from a pretrained
self-supervised model, such as SimCLR.

So each image becomes a point in feature space:

```text
image -> feature vector
```

Distances and similarities are then computed between feature vectors. If two
feature vectors are close, the corresponding images are considered similar.

This is important because active learning selection depends heavily on the
geometry of the feature space.

### 1.6 Kernels, Graphs, And Degree

Many coverage-based active learning methods build a similarity graph over the
dataset.

In this graph:

- Each data point is a node
- An edge means two points are similar or close
- The edge weight measures how similar they are

The kernel matrix stores these similarities:

```text
K_ij = similarity between point i and point j
```

An RBF kernel is one common way to build this similarity matrix. RBF stands for
radial basis function. In this project, it means that we take the distance
between two feature vectors and convert it into a similarity value:

```text
K_ij = exp(-distance(i, j)^2 / sigma^2)
```

If two points are very close, the distance is small, so the exponent is close to
zero and the similarity is close to one. If two points are far apart, the
distance is large, so the exponent becomes strongly negative and the similarity
gets close to zero.

The parameter sigma controls how quickly similarity decays with distance.

Intuitively:

- Small sigma creates a local kernel where only very close points are similar
- Large sigma creates a smoother kernel where many points are similar
- Global sigma uses one scale for the whole dataset
- Local sigma gives each point its own scale based on nearby points

The degree of a point is the total amount of connectivity it has to other
points. In an unweighted graph this is the number of neighbors. In a weighted
graph this is the sum of edge weights.

For example, in a binary graph like ProbCover, a point with degree 30 is
connected to 30 other points inside the distance threshold. In a weighted kernel
graph, the degree is not just a neighbor count. A point may have many weak
connections or fewer strong connections, and its weighted degree sums those
similarity values:

```text
degree(i) = sum_j K_ij
```

When we talk about the average degree of selected points, we mean the average
connectivity of the points chosen by the active learning algorithm. Comparing
this to the average degree of the full dataset tells us whether the algorithm is
selecting unusually dense points, unusually sparse points, or points that look
typical under the graph.

Intuitively:

- High-degree points are in dense or central regions
- Low-degree points are in sparse or isolated regions

This project eventually found that degree is one of the most important
quantities for understanding the behavior of MaxHerding.

### 1.7 The Local Sigma Paper

The Local Sigma idea comes from the paper Self-Tuning Spectral Clustering by
Zelnik-Manor and Perona. That paper is not an active learning paper. It studies
spectral clustering, where the goal is to group unlabeled points using a
similarity matrix.

Standard spectral clustering uses an affinity matrix such as:

```text
A_ij = exp(-distance(i, j)^2 / sigma^2)
```

Here sigma is a global scale parameter. It decides when two points should be
considered similar. The problem is that one global sigma may not work well when
the data has regions with different densities or different cluster scales. A
sigma that is good for a dense cluster may be too small for a sparse cluster. A
sigma that is good for a sparse cluster may over-connect points in a dense
cluster.

The paper proposes replacing the single global sigma with a local sigma for each
point. For point i:

```text
sigma_i = distance from point i to its K-th nearest neighbor
```

Then the affinity between two points is computed as:

```text
A_ij = exp(-distance(i, j)^2 / (sigma_i sigma_j))
```

This makes the scale adaptive. In dense regions, the K-th neighbor is close, so
sigma_i is small. In sparse regions, the K-th neighbor is farther away, so
sigma_i is larger. The paper shows that this can help spectral clustering on
multi-scale data because similarities are measured relative to the local
neighborhood of each point rather than using one global distance scale for the
whole dataset.

This motivated our research because MaxHerding also depends on a pairwise
similarity matrix. If a global sigma can fail in clustering when the data has
non-uniform density, then it is natural to ask whether a global sigma can also
be suboptimal for active learning selection. Our starting hypothesis was that
local sigma might let MaxHerding compare points more fairly across dense and
sparse regions.

There is an important difference between the paper and our setting. The Local
Sigma paper used local scaling for clustering, while we use it for selecting
points under a label budget. Also, the paper reports that a small fixed K worked
well for its clustering examples, but our experiments showed that K behaves very
differently inside MaxHerding on SSL features. In our setting, K strongly
changes graph degree and sparsity, so choosing K is not just a technical detail.
It becomes part of the behavior we need to understand.

### 1.8 ProbCover

ProbCover is a coverage-based active learning method. It creates a graph using a
distance threshold. If two points are close enough, they are connected. If not,
they are not connected.

This creates a binary coverage graph:

```text
edge(i, j) = 1 if distance(i, j) <= threshold
edge(i, j) = 0 otherwise
```

The algorithm then tries to select points that cover many currently uncovered
points.

The important idea for this project is the threshold graph: it gives a concrete
notion of which points are close enough to count as neighbors.

We also used this idea inside MaxHerding as a cutoff. Without a cutoff, the RBF
kernel gives every pair of points some positive similarity, even if that
similarity is extremely small. With a ProbCover-style cutoff, pairs of points
that are farther than the ProbCover threshold are treated as disconnected:

```text
K_ij = RBF similarity if distance(i, j) <= threshold
K_ij = 0 otherwise
```

So the kernel remains smooth for nearby points, but distant points do not
contribute at all. This makes MaxHerding closer to the coverage logic of
ProbCover while still keeping weighted similarities between nearby points. It
should be interpreted as a greedy coverage-style selection rule, not exactly as
the classical binary greedy max-coverage algorithm.

### 1.9 MaxHerding

MaxHerding is another coverage/representativeness-based active learning method.
Instead of selecting points only by a binary coverage graph, it uses a smooth
kernel matrix.

A point is valuable if it helps represent or cover the remaining unlabeled
distribution. The kernel controls how much one selected point represents nearby
points.

Therefore, MaxHerding depends strongly on the kernel:

- If the kernel is too sparse, points do not cover enough neighbors
- If the kernel is too dense, many points may look similar
- If the kernel overemphasizes dense regions, selection may become redundant

This is why we focused on how sigma, local sigma, graph degree, and density
normalization affect the kernel.

## 2. Research Goal

The project started from a question about whether localizing the kernel scale in
active learning can improve the behavior of coverage-based selection methods.

The main method we focused on is MaxHerding. In this setting, the algorithm
selects points using a similarity/kernel matrix between unlabeled examples. The
kernel determines which points are considered close, redundant, representative,
or useful for covering the dataset.

The core idea was:

- A single global sigma may not fit all parts of the dataset
- Dense regions and sparse regions may need different scales
- Local sigma could adapt the kernel to the local geometry
- This might improve the balance between selecting representative dense points
  and not ignoring sparse or rare regions

This direction was motivated by the Local Sigma paper, where the kernel scale is
adapted per point using local neighborhood distances rather than a single global
bandwidth. We used that idea as the starting point and asked whether the same
kind of local scaling is useful inside MaxHerding active learning.

Over time, the research question became more specific:

> Is the benefit coming from locality itself, or mostly from changing the graph
> degree/density induced by the kernel?

The current evidence points toward the second explanation: the graph degree
appears to be a dominant factor in MaxHerding behavior, possibly more than the
locality mechanism itself.

## 3. Experimental Setup

The main validated setup used:

- Active learning method: MaxHerding
- Evaluation classifier: 1-nearest neighbor on SSL features
- Features: fixed SimCLR/SSL features
- Kernel: RBF kernel
- Cutoff: ProbCover-style distance cutoff applied to the RBF kernel
- Main datasets so far: CIFAR10 and CIFAR100
- CIFAR10 budget range: 10 to 100 labels
- CIFAR100 budget range: 100 to 1000 labels

This setup was chosen because the global-sigma MaxHerding + 1NN path reproduced
the expected reference behavior most reliably.

## 4. First Direction: Local Sigma Through K

The first main experiment applied the Local Sigma idea by varying the local
sigma parameter K. For each point, the local sigma is based on the distance to
its K-th neighbor. Small K gives a very local scale; large K gives a smoother,
more global-like scale.

The expectation was:

- Small K might capture local structure better
- Large K might behave more like global sigma
- Some middle K might improve the dense/sparse selection balance

### Main Result

K mostly controls the effective graph density.

Small K values made the graph too sparse and disconnected. These settings
performed badly because many points had almost no meaningful neighbors under the
cutoff rule.

Large K values made local sigmas more similar across points. These settings
became close to global sigma, so they were stable but did not provide a strong
localization advantage.

The useful range was medium-to-large K:

- CIFAR10: around K=3500
- CIFAR100: around K=1440

The improvements in this range were modest.

### Interpretation

Local sigma does change the kernel, but much of the effect can be explained by
the induced graph degree:

- Too low degree means the graph is fragmented
- Moderate degree gives useful coverage
- Very high degree becomes close to global behavior

This suggests that the key variable may not be K itself, but the effective
degree distribution produced by K.

Suggested visual:

- Effective degree versus final accuracy for CIFAR10 and CIFAR100
  (`visual_outputs/maxherding/density_norm/professor_summary/cifar10_effective_degree_vs_final_accuracy.png`,
  `visual_outputs/maxherding/density_norm/professor_summary/cifar100_effective_degree_vs_final_accuracy.png`)

This figure supports the claim that K mainly acts through the graph density it
induces. It should show that small-degree graphs fail, medium/large-degree
settings are the useful region, and very large K does not create a strong
advantage over global sigma.

## 5. Density and Selection Diagnostics

To understand what MaxHerding actually selects, we measured:

- Dataset-level graph degree
- Selected-point graph degree
- Selected-point degree normalized by dataset-average degree
- Selected label diversity
- Local sigma statistics
- Accuracy

### Main Findings

MaxHerding selects points that are denser than the dataset average.

For example, under global sigma:

- CIFAR10 selected points had much higher degree than the average dataset point
- CIFAR100 showed the same pattern

This means MaxHerding naturally prioritizes central or high-coverage points.
That is not necessarily bad: dense regions are representative and should be
covered early. But it also means local sigma would need to do something
nontrivial if we want it to improve sparse/dense balance.

### Local Sigma Versus Global Selection

When comparing local sigma to global sigma, two density notions are important:

- Raw/global density: density under the same raw ProbCover graph
- Effective/local density: density under the graph actually used by the method

Local sigma often selects points with lower raw/global density than global
MaxHerding. However, under the local graph, those same selected points can still
look highly connected.

This suggests that local sigma shifts the geometry: it may make some points look
locally central even when they are less central under the global graph.

This is important, but it did not translate into a strong robust accuracy gain.
Therefore, locality changes selection behavior, but not always in a clearly
beneficial way.

Suggested visuals:

- Selected-point degree normalized by dataset-average degree
  (`visual_outputs/maxherding/density_selection/comparisons/cifar10_selected_density_ratio_avg.png`,
  `visual_outputs/maxherding/density_selection/comparisons/cifar100_selected_density_ratio_avg.png`)

- Raw/global selected density ratio
  (`visual_outputs/maxherding/density_selection/comparisons/cifar10_selected_raw_density_ratio_final.png`,
  `visual_outputs/maxherding/density_selection/comparisons/cifar100_selected_raw_density_ratio_final.png`)

These figures should make the density story concrete. The first shows whether
MaxHerding selects points that are denser than the dataset average. The second
shows whether local sigma selects points that are globally denser or sparser
than the global-sigma selected points.

## 6. Second Direction: Density-Normalized Kernel

After seeing that graph degree seemed central, we tried modifying the kernel
directly by normalizing by degree.

The symmetric version was:

```text
K_ij <- K_ij / (degree_i degree_j)^(alpha / 2)
```

Here:

- `degree_i` is the graph degree of point i
- `alpha = 0` gives the original kernel
- Larger alpha penalizes dense points more strongly

In our implementation, degree is measured on the kernel graph after the
ProbCover-style cutoff. We used two degree definitions. Effective degree counts
how many nonzero neighbors a point has, excluding the self-edge. Weighted degree
uses the kernel mass, `sum_j K_ij`, also excluding the self-edge. The later
alpha-wide experiments used weighted degree for the normalization, while many of
the diagnostic plots report effective degree because it is easier to interpret
as graph connectivity.

The alpha shown in the result graphs is the same alpha in the formula above. It
appears as `alpha / 2` only because symmetric normalization splits the total
penalty equally between the two endpoints. For `alpha = 1`, the denominator is
the geometric mean of the two endpoint degrees. For `alpha = 0.5`, the penalty
is weaker: `(degree_i degree_j)^0.25`.

The motivation was:

- MaxHerding already tends to pick dense/central points
- This may create redundancy
- A controlled density penalty might encourage more balanced selection

## 7. Fixed Alpha Results

We tested fixed alpha values:

```text
alpha = 0, 0.1, 0.25, 0.5, 0.75, 1.0
```

for global sigma and the best local K settings.

### Main Result

Alpha normalization can help at very low budgets, but it is not consistently
helpful later.

For CIFAR10:

- The clearest gains appeared around 20 to 30 labels
- Stronger alpha values helped early
- The benefit weakened or became harmful later

For CIFAR100:

- The clearest gains appeared around 100 to 400 labels
- Alpha=0.5 was often useful early
- Later budgets preferred weaker alpha or alpha=0

Alpha=1.0 was too aggressive and collapsed performance.

### Interpretation

This supports the idea that density is important. Penalizing dense points can
help when the budget is extremely small, likely because it reduces redundant
selection inside the densest modes.

However, dense regions are also representative. Penalizing them too much, or for
too long, can hurt performance. This creates a budget-dependent tradeoff.

The best alpha is not constant across the active learning trajectory.

Suggested visual:

- Alpha gain over alpha=0 by budget for CIFAR10 and CIFAR100
  (`visual_outputs/maxherding/density_norm/alpha_gain/cifar10_global_and_bestk_alpha_gain.png`,
  `visual_outputs/maxherding/density_norm/alpha_gain/cifar100_global_and_bestk_alpha_gain.png`)

This figure should show that density normalization helps mainly at very low
budgets, and that the gain is not stable across the full active learning
trajectory.

## 8. Third Direction: Dynamic Alpha / Decay

Because fixed alpha helped early and hurt later, we tested simple dynamic alpha
schedules:

- Early-only alpha
- Linear decay
- Three-phase decay

The hope was that these schedules would preserve early gains while avoiding late
penalties.

### Main Result

The simple decay schedules did not improve final accuracy.

They preserved some early gains, especially for CIFAR100, but every tested
schedule ended below the alpha=0 baseline at the final budget. In other words,
the early gain often did not persist, and in several settings it became a
negative gain at later budgets.

### Interpretation

This suggests that the correct adaptation is probably not a simple hand-written
monotonic decay. The schedules we tested were fixed by episode number; they did
not use the actual graph degree, selected-set density, or coverage state.

The useful rule may need to depend on the actual density/selection state of the
run, not only on the episode number.

In other words, alpha should perhaps be chosen based on what the selected set
already covers, or on graph statistics, rather than following a fixed schedule.

Suggested visual:

- Decay schedule gain/loss for CIFAR10 and CIFAR100
  (`visual_outputs/maxherding/density_norm/decay_tuned/cifar10_k3500_delta.png`,
  `visual_outputs/maxherding/density_norm/decay_tuned/cifar100_k1440_delta.png`)

This figure should support the conclusion that simple decay preserves some early
gains but does not improve final accuracy.

## 9. Main Scientific Conclusion So Far

The current evidence suggests:

1. Local sigma affects MaxHerding mainly by changing graph degree.
2. Very sparse graphs hurt badly.
3. Very dense or very smooth local graphs become close to global sigma.
4. Medium-degree graphs are the most useful regime.
5. Selected-point density is consistently higher than dataset-average density.
6. Explicit density normalization can help at very low budgets.
7. The early benefit is budget-dependent and often degrades at later budgets.
8. Simple decay schedules are too crude.

The most important takeaway is:

> The graph degree induced by the kernel appears to be a stronger explanatory
> variable than locality by itself.

This does not mean locality is irrelevant or separate from degree. Local sigma
changes the kernel graph, and degree is one measurable graph statistic that
captures much of that change. The current evidence suggests that the observed
benefit or failure of localization is largely mediated by the degree and
coverage structure it induces.

## 10. What We Can Learn

The results shift the research direction.

Initially, the question was:

> Which local sigma K is best?

Now the better question seems to be:

> What graph degree or density profile should the active learning algorithm
> induce at each budget?

This reframes the problem from choosing K manually to controlling the graph
structure more directly.

Possible future rules could be based on:

- Target effective degree
- Selected-point degree relative to dataset-average degree
- Budget-dependent density targets
- Adaptive alpha chosen from current selection statistics

A useful future visual would summarize the best observed K/alpha configuration
at each budget across seeds. This would directly test whether the optimal
configuration changes in a stable way during the active learning trajectory.

## 11. Open Questions

The current results answer part of the original question, but they also open
several new questions.

### 11.1 Generality Across Datasets

The strongest conclusion so far is based on CIFAR10 and CIFAR100. We still need
to know whether it holds on other datasets.

Open questions:

- Does graph degree remain the dominant factor on larger datasets?
- Does the same small-K failure appear on datasets with more classes or more
  visual diversity?
- Does the useful degree range scale with dataset size, number of classes, or
  feature quality?
- Is the degree-based interpretation still true on Tiny ImageNet, or is it a
  CIFAR-specific effect?

Why this matters:

- If Tiny ImageNet shows the same behavior, the degree explanation becomes much
  stronger.
- If it does not, then locality may interact with dataset structure in a more
  complicated way.

### 11.2 Degree Versus Locality

The main current hypothesis is that degree matters more than locality itself.
But this is not fully proven.

Open questions:

- If two kernels have similar effective degree distributions, do they produce
  similar selections and accuracy even if one is global and one is local?
- Can local sigma provide benefits beyond degree control?
- Is K only a proxy for graph degree?
- Would directly targeting a desired degree perform better than choosing K?

Why this matters:

- If degree is the true control variable, future algorithms should control
  degree directly rather than tune K.
- If locality has an independent benefit, then we need diagnostics that separate
  locality from degree.

### 11.3 Selection Dynamics Across Budget

The best configuration appears to change over the active learning trajectory.

Open questions:

- Why does alpha normalization help early but hurt later?
- Does MaxHerding first select central/dense points and only later move outward?
- Can we identify the point where dense-region penalization should be reduced or
  turned off?
- Is the best alpha determined by budget, selected-set diversity, selected-set
  density, or something else?

Why this matters:

- A fixed K or fixed alpha is probably too rigid
- The next useful algorithm may need to adapt based on the current selected set

### 11.4 Best Statistic For Prediction

We measured several density-related statistics, but it is not yet clear which
one best predicts performance.

Candidate predictors:

- Dataset effective degree
- Selected-point effective degree
- Selected-point degree divided by dataset-average degree
- Raw/global selected density
- Local sigma mean and coefficient of variation
- Selected label entropy
- Class diversity of the selected batch

Open questions:

- Which statistic correlates best with accuracy?
- Which statistic predicts when a K value is too sparse?
- Which statistic predicts when alpha normalization becomes harmful?
- Can these statistics be used online during active learning?

### 11.5 Better Adaptive Rules

The simple decay schedules were not enough.

Open questions:

- Can alpha be chosen dynamically from density statistics rather than episode
  number?
- Can K be chosen to hit a target effective degree?
- Can we design a rule that uses density normalization only when selected points
  are too concentrated in dense regions?
- Can we avoid hurting late-budget accuracy while preserving early gains?

## 12. Future Research Directions

These are directions that other people could continue from this project.

### 12.1 Degree-Targeted Active Learning

Instead of manually choosing local sigma K, choose K or sigma so that the graph
has a desired effective degree.

Possible research question:

> Is controlling graph degree directly more robust than choosing local sigma K?

Possible approach:

- Define a target average effective degree
- Choose K or sigma to match that target
- Compare against global sigma and fixed local K

### 12.2 Adaptive Density Penalization

Instead of using fixed alpha, choose alpha based on the current state of the
selected set.

Possible research question:

> Can density normalization improve early-budget selection without hurting later
> budgets if alpha is selected adaptively?

Possible approach:

- Measure selected/dataset degree ratio after each episode
- Apply alpha only if selected points are much denser than the dataset average
- Reduce alpha when selected diversity is high enough

### 12.3 Separating Degree From Geometry

Design controlled experiments where two kernels have similar degree but
different geometric/locality structure.

Possible research question:

> Does locality matter after controlling for degree?

Possible approach:

- Compare global and local kernels matched by average effective degree
- Compare their selected sets
- Measure overlap, class diversity, and accuracy

### 12.4 Dataset Scaling Study

Repeat the same analysis on additional datasets.

Candidate datasets:

- Tiny ImageNet
- STL10
- ImageNet subset
- Domain-specific datasets if available

Possible research question:

> Is the degree-dominance phenomenon universal, or does it depend on dataset
> scale and feature quality?

The most useful figure for this direction would be an effective-degree versus
final-accuracy plot for Tiny ImageNet, matching the CIFAR10 and CIFAR100 plots.

### 12.5 Selection-Set Geometry Analysis

Analyze not only individual selected-point density but the geometry of the full
selected set.

Possible metrics:

- Pairwise distance between selected points
- Class coverage
- Cluster coverage
- Redundancy among selected points
- Distance from selected points to nearest dataset modes

Possible research question:

> Does MaxHerding fail when selected points are individually dense, mutually
> redundant, or both?

## 13. Short Thesis-Style Takeaway

The project suggests a shift in how we should think about local kernels in
coverage-based active learning. Local sigma is not automatically useful because
it is local. Its main effect is to change the graph induced by the kernel,
especially the effective degree of points. Once viewed this way, many results
become more coherent: small K fails because the graph is too sparse, large K
resembles global sigma, and density normalization helps only when dense-region
selection is too dominant. Future work should therefore focus less on manually
tuning locality and more on controlling graph density and selected-set density
through adaptive rules.
