# Research Summary: The Effect of Locality and Density on Coverage-Based Active Learning

## Executive Summary

We studied whether local kernel scaling can improve active learning.
Active learning is useful when labels are expensive: instead of labeling
many examples, the algorithm chooses which examples should be labeled.

The original idea came from the Self-Tuning Spectral Clustering paper. Instead of using one
global kernel scale for the whole dataset, each point receives a local
scale based on its neighborhood. We expected this might help balance
dense and sparse regions of the data.

The main conclusion is that locality alone does not explain most of the
behavior. The effective graph degree induced by the kernel appears to be
more important. If the graph is too sparse, the method performs badly
because points do not cover enough neighbors. If the graph is too dense,
many points become too similar and dense regions can dominate. The
useful behavior is between these extremes.

The research therefore shifted from asking which local neighborhood size
K is best to asking what graph degree or density profile MaxHerding
should induce at each budget. On CIFAR100, the strongest evidence points
to a useful low-to-moderate degree regime, roughly around 18-25
effective neighbors, but this regime is budget-dependent rather than one
fixed global optimum.

## 1. Background: Active Learning

In many machine-learning problems, unlabeled data is easy to collect but
labels are expensive. Active learning studies how to choose a small set
of examples to label so that the resulting labeled set is as useful as
possible.

This project uses pool-based active learning. We start with a fixed
unlabeled pool. At each round, the algorithm selects a batch of points
to label, and the selected points move from the unlabeled pool to the
labeled set. The total number of labels the algorithm is allowed to
request is called the budget.

We focused on low-budget active learning, where every selected point
matters. Our main ranges were 10-100 labels for CIFAR10 and 100-1000
labels for CIFAR100.

### 1.1 CIFAR10 And CIFAR100

CIFAR10 and CIFAR100 are standard image-classification benchmarks.
CIFAR10 has 10 object classes, while CIFAR100 has 100 finer-grained
classes.

We used both datasets, but the final analysis focuses mostly on
CIFAR100. CIFAR10 baselines often reach high accuracy with very few
labels, so there is less room for improvement. CIFAR100 is harder
because the budget must cover many more classes, making it more useful
for studying how kernel density and selection behavior affect
performance.

### 1.2 What Makes A Good Active Learning Selection?

A good selected set should be informative and representative.

Important selection goals include:

-   Cover the main regions of the data
-   Avoid choosing many redundant points
-   Include examples from different classes or clusters
-   Include rare or sparse regions when they matter
-   Improve downstream classification accuracy

There is a natural tension:

-   Dense regions contain many points and are often representative
-   Sparse regions may contain rare but important structure
-   Selecting only dense points can be redundant
-   Selecting too many sparse/outlier points can waste budget

Much of this project is about understanding this dense-versus-sparse
tradeoff.

### 1.3 Feature Space

The algorithms in this project usually do not operate directly on raw
image pixels. Instead, each image is represented by a feature vector
from a pretrained self-supervised model, such as SimCLR.

So each image becomes a point in feature space:

    image -> feature vector

Distances and similarities are then computed between feature vectors. If
two feature vectors are close, the corresponding images are considered
similar. Active learning selection therefore depends heavily on the
geometry of the feature space.

### 1.4 Kernels, Graphs, And Degree

Many coverage-based active learning methods build a similarity graph
over the dataset. In this graph, each data point is a node, and an edge
means two points are similar or close.

An RBF kernel converts distance into similarity:

$$K_{ij} = \exp\left(-\frac{d(i,j)^2}{\sigma^2}\right)$$

The parameter σ controls how quickly similarity decays with
distance. Small σ creates a more local kernel, while large σ
creates a smoother kernel where many points are similar.

The degree of a point is the total amount of connectivity it has to
other points. In an unweighted graph this is the number of neighbors. In
a weighted graph this is the sum of edge weights:

$$\text{degree}(i) = \sum_j K_{ij}$$

High-degree points are usually in dense or central regions. Low-degree
points are usually in sparse or isolated regions. Comparing the degree
of selected points to the dataset-average degree tells us whether the
algorithm is selecting unusually dense points, unusually sparse points,
or points that look typical under the graph.

In our cutoff experiments, δ is the threshold that decides which
connections remain. Points outside the δ radius are treated as
disconnected. Therefore, for local σ, graph degree is induced by the
full `(K, δ)` setting: K determines the local scale, and δ
determines which scaled distances remain connected.

### 1.5 MaxHerding

MaxHerding is another coverage/representativeness-based active learning
method. Instead of selecting points only by a binary coverage graph, it
uses a smooth kernel matrix.

A point is valuable if it helps represent or cover the remaining
unlabeled distribution. The kernel controls how much one selected point
represents nearby points.

Therefore, MaxHerding depends strongly on the kernel:

-   If the kernel is too sparse, points do not cover enough neighbors
-   If the kernel is too dense, many points may look similar
-   If the kernel overemphasizes dense regions, selection may become
    redundant

This is why we focused on how σ, local σ, graph degree, and
density normalization affect the kernel.

### 1.6 The Self-Tuning Spectral Clustering Paper

The local-σ idea comes from the paper Self-Tuning Spectral
Clustering by Zelnik-Manor and Perona. That paper is not an active
learning paper. It studies spectral clustering, where the goal is to
group unlabeled points using a similarity matrix.

Standard spectral clustering uses an affinity matrix such as:

$$A_{ij} = \exp\left(-\frac{d(i,j)^2}{\sigma^2}\right)$$

Here σ is a global scale parameter. It decides when two points
should be considered similar. The problem is that one global σ may
not work well when the data has regions with different densities or
different cluster scales. A σ that is good for a dense cluster may
be too small for a sparse cluster. A σ that is good for a sparse
cluster may over-connect points in a dense cluster.

The paper proposes replacing the single global σ with a local σ
for each point. For point i:

*σ*<sub>*i*</sub> = distance from point *i* to its K-th nearest neighbor

Then the affinity between two points is computed as:

$$A_{ij} = \exp\left(-\frac{d(i,j)^2}{\sigma_i\sigma_j}\right)$$

This makes the scale adaptive. In dense regions, the K-th neighbor is
close, so *σ*<sub>*i*</sub> is small. In sparse regions, the K-th
neighbor is farther away, so *σ*<sub>*i*</sub> is larger. The paper
shows that this can help spectral clustering on multi-scale data because
similarities are measured relative to the local neighborhood of each
point rather than using one global distance scale for the whole dataset.

This motivated our research because MaxHerding also depends on a
pairwise similarity matrix. If a global σ can fail in clustering
when the data has non-uniform density, then it is natural to ask whether
a global σ can also be suboptimal for active learning selection. Our
starting hypothesis was that local σ might let MaxHerding compare
points more fairly across dense and sparse regions.

There is an important difference between the paper and our setting.
Self-Tuning Spectral Clustering paper used local scaling for clustering,
while we use it for selecting points under a label budget. Also, the
paper reports that a small fixed K worked well for its clustering
examples, but our experiments showed that K behaves very differently
inside MaxHerding on SSL features. In our setting, K strongly changes
graph degree and sparsity, so choosing K is not just a technical detail.
It becomes part of the behavior we need to understand.


### 1.7 ProbCover

ProbCover is a coverage-based active learning method. It creates a graph
using a distance threshold. If two points are close enough, they are
connected. If not, they are not connected.

This creates a binary coverage graph:

$$
\text{edge}(i,j) =
\begin{cases}
1, & d(i,j) \leq \text{threshold} \\
0, & d(i,j) > \text{threshold}
\end{cases}
$$

The algorithm then tries to select points that cover many currently
uncovered points.

The important idea for this project is the threshold graph: it gives a
concrete notion of which points are close enough to count as neighbors.

We also used this idea inside MaxHerding as a cutoff. Without a cutoff,
the RBF kernel gives every pair of points some positive similarity, even
if that similarity is extremely small. With a ProbCover-style cutoff,
pairs of points that are farther than the ProbCover threshold are
treated as disconnected:

$$
K_{ij} =
\begin{cases}
\text{RBF similarity}, & d(i,j) \leq \text{threshold} \\
0, & d(i,j) > \text{threshold}
\end{cases}
$$

So the kernel remains smooth for nearby points, but distant points do
not contribute at all. This makes MaxHerding closer to a greedy
coverage-style rule, but it is not exactly classical greedy max
coverage: ProbCover uses binary coverage counts, while MaxHerding still
uses weighted kernel similarities and its own herding objective.


## 2. Research Goal

The project started from a question about whether localizing the kernel
scale in active learning can improve the behavior of coverage-based
selection methods.

The main method we focused on is MaxHerding. In this setting, the
algorithm selects points using a similarity/kernel matrix between
unlabeled examples. The kernel determines which points are considered
close, redundant, representative, or useful for covering the dataset.

The core idea was:

-   A single global σ may not fit all parts of the dataset
-   Dense regions and sparse regions may need different scales
-   Local σ could adapt the kernel to the local geometry
-   This might improve the balance between selecting representative
    dense points and not ignoring sparse or rare regions

This direction was motivated by the Self-Tuning Spectral Clustering paper, where the kernel
scale is adapted per point using local neighborhood distances rather
than a single global bandwidth. We used that idea as the starting point
and asked whether the same kind of local scaling is useful inside
MaxHerding active learning.

Over time, the research question became more specific:

> Is the benefit coming from locality itself, or mostly from changing
> the graph degree/density induced by the kernel?

The current evidence points toward the second explanation: the graph
degree appears to be a dominant factor in the algorithm’s behavior,
possibly more than the locality mechanism itself.

## 3. Experimental Setup

The main validated setup used:

-   Active learning method: MaxHerding
-   Evaluation classifier: 1-nearest neighbor on SSL features
-   Features: fixed SimCLR/SSL features
-   Kernel: RBF kernel
-   Cutoff: ProbCover-style distance cutoff applied to the RBF kernel
-   Main datasets: CIFAR10 and CIFAR100
-   CIFAR10 budget range: 10 to 100 labels
-   CIFAR100 budget range: 100 to 1000 labels

This setup was chosen because the global-σ MaxHerding + 1NN path
reproduced the expected reference behavior most reliably.

## 4. First Direction: Local σ Through K

The first main experiment applied the local-σ idea by varying the
local σ parameter K. For each point, the local σ is based on the
distance to its K-th neighbor. Small K gives a very local scale; large K
gives a smoother, more global-like scale.

The expectation was:

-   Small K might capture local structure better
-   Large K might behave more like global σ
-   Some middle K might improve the dense/sparse selection balance

### Main Result

K mostly controls the effective graph density.

Small K values made the graph too sparse and disconnected. These
settings performed badly because many points had almost no meaningful
neighbors under the cutoff rule.

Large K values made local σ values more similar across points. These
settings became close to global σ, so they were stable but did not
provide a strong localization advantage.

The useful range was medium-to-large K:

-   CIFAR10: around K=3500
-   CIFAR100: around K=1440

On CIFAR100, the budget-wise best local K improves over global σ
across the full 100 to 1000 label range. The gain is largest at very low
budgets and becomes smaller later, but the direction is consistent in
the averaged seed-3/4/5 results.

### Interpretation

Local σ does change the kernel, but much of the effect can be
explained by the induced graph degree:

-   Too low degree means the graph is fragmented
-   Moderate degree gives useful coverage
-   Very high degree becomes close to global behavior

This suggests that the key variable may not be K itself, but the
effective degree distribution produced by K.

![Figure 1. Effective degree versus final accuracy for CIFAR10.](/tmp/summary_update_inspect_long/media/image1.png){width=4.69015in}

Figure 1. Effective degree versus final accuracy for CIFAR10.

![Figure 2. Effective degree versus final accuracy for CIFAR100.](/tmp/summary_update_inspect_long/media/image2.png){width=5.83333in}

Figure 2. Effective degree versus final accuracy for CIFAR100.

This figure supports the claim that K mainly acts through the graph
density it induces. It shows that small-degree graphs fail,
medium/large-degree settings are the useful region, and very large K
does not create a strong advantage over global σ.

![Figure 3. CIFAR100 global σ compared with the best local-σ K at each budget.](/tmp/summary_update_inspect_long/media/image3.png){width=5.83333in}

Figure 3. CIFAR100 global σ compared with the best local-σ K at
each budget.

Figure 3 makes the budget dependence explicit. The best local K is
K=2000 from 100 to 600 labels and K=1440 from 700 to 1000 labels. This
supports a positive but nuanced conclusion: localization can improve
MaxHerding on CIFAR100, but the useful locality scale is
budget-dependent rather than fixed once for the whole run.

## 5. Density and Selection Diagnostics

To understand what MaxHerding actually selects, we measured:

-   Dataset-level graph degree
-   Selected-point graph degree
-   Selected-point degree normalized by dataset-average degree
-   Selected label diversity
-   Local σ statistics
-   Accuracy

### Main Findings

MaxHerding selects points that are denser than the dataset average.

This means MaxHerding naturally prioritizes central or high-coverage
points. That is not necessarily bad: dense regions are representative
and should be covered early. But it also means local σ would need to
do something nontrivial if we want it to improve sparse/dense balance.

### Local σ Versus Global Selection

When comparing local σ to global σ, two density notions are
important:

-   Raw/global density: density under the same raw ProbCover graph
-   Effective/local density: density under the graph actually used by
    the method

Local σ often selects points with lower raw/global density than
global MaxHerding. However, under the local graph measurement, those
same selected points can still look highly connected.

This suggests that local σ shifts the geometry: it may make some
points look locally central even when they are less central under the
global graph.

To compare selected points to the dataset more directly, we also
measured the selected-point degree divided by the dataset-average
degree. This is a rank-like proxy rather than an exact percentile rank:
it tells us whether selected points are more connected than a typical
dataset point. Exact average percentile rank at every budget would
require storing or recomputing the full per-point degree distribution
for each configuration.

![Figure 4. Selected-point effective degree mean and within-batch standard deviation by budget for CIFAR10.](/tmp/summary_update_inspect_long/media/image4.png){width=5.84483in}

Figure 4. Selected-point effective degree mean and within-batch standard
deviation by budget for CIFAR10.

![Figure 5. Selected-point effective degree mean and within-batch standard deviation by budget for CIFAR100.](/tmp/summary_update_inspect_long/media/image5.png){width=5.85343in}

Figure 5. Selected-point effective degree mean and within-batch standard
deviation by budget for CIFAR100.

![Figure 6. Selected-point degree normalized by dataset-average degree for CIFAR10.](/tmp/summary_update_inspect_long/media/image6.png){width=5.83333in}

Figure 6. Selected-point degree normalized by dataset-average degree for
CIFAR10.

![Figure 7. Selected-point degree normalized by dataset-average degree for CIFAR100.](/tmp/summary_update_inspect_long/media/image7.png){width=5.83333in}

Figure 7. Selected-point degree normalized by dataset-average degree for
CIFAR100.

![Figure 8. Raw/global selected density ratio for CIFAR10.](/tmp/summary_update_inspect_long/media/image8.png){width=5.83333in}

Figure 8. Raw/global selected density ratio for CIFAR10.

![Figure 9. Raw/global selected density ratio for CIFAR100.](/tmp/summary_update_inspect_long/media/image9.png){width=5.83333in}

Figure 9. Raw/global selected density ratio for CIFAR100.

These figures make the density story concrete. The first shows
whether MaxHerding selects points that are denser than the dataset
average. The second shows whether local σ selects points that are
globally denser or sparser than the global-σ selected points.

## 6. Second Direction: Density-Normalized Kernel

After seeing that graph degree seemed central, we tried modifying the
kernel directly by normalizing by degree.

The symmetric version was:

$$
K_{ij} \leftarrow
\frac{K_{ij}}{\left(\text{degree}_i \text{degree}_j\right)^{\alpha/2}}
$$

Here:

-   `degree_i` is the graph degree of point i after the same
    ProbCover-style cutoff used by the experiment. We considered two
    degree definitions: effective degree, the number of nonzero
    neighbors after removing the diagonal, and weighted degree, computed
    as the weighted-degree formula from Section 1.4 after removing the
    diagonal. We tested both effective and weighted degree for
    normalization; their results were very similar, suggesting that the
    main effect comes from penalizing high-connectivity points in
    general rather than from the exact degree definition. The α-wide
    experiments used weighted degree, while many diagnostic plots report
    effective degree because it is easier to interpret as graph
    connectivity.
-   `α = 0` gives the original kernel
-   Larger α penalizes dense points more strongly
-   It appears as α/2 because in symmetric normalization the total
    density penalty α is split equally between the two endpoints of
    the edge

The α shown in the result graphs is the same total exponent α in
this formula. Because the normalization is symmetric, α=1 divides an
edge by the geometric mean of the two endpoint degrees. α=0.5 is
weaker: the denominator is (degree_i degree_j)^0.25. In these
experiments this is degree, not percentile rank.

The motivation was:

-   MaxHerding already tends to pick dense/central points
-   This may create redundancy
-   A controlled density penalty might encourage more balanced selection

## 7. Fixed α Results

We tested fixed α values:

    α = 0, 0.1, 0.25, 0.5, 0.75, 1.0

for global σ and the best local K settings.

### Main Result

On CIFAR100, density normalization improves the low-budget regime when
α is chosen appropriately. The improvement is strongest around 100
to 400 labels and becomes weaker later, which suggests that α should
be treated as a budget-dependent control rather than a fixed global
setting.

For CIFAR10:

-   The clearest gains appeared around 20 to 30 labels
-   Stronger α values helped early
-   The benefit weakened or became harmful later

For CIFAR100:

-   The clearest gains appeared around 100 to 400 labels
-   α=0.5 was often useful early
-   Later budgets preferred weaker α or α=0

α=1.0 was too aggressive and collapsed performance.

### Interpretation

This supports the idea that density is important. Penalizing dense
points can help when the budget is extremely small, likely because it
reduces redundant selection inside the densest modes.

However, dense regions are also representative. Penalizing them too
much, or for too long, can hurt performance. This creates a
budget-dependent tradeoff.

The best α is not constant across the active learning trajectory.

![Figure 10. α gain over α=0 by budget for CIFAR10.](/tmp/summary_update_inspect_long/media/image10.png){width=5.83333in}

Figure 10. α gain over α=0 by budget for CIFAR10.

![Figure 11. α gain over α=0 by budget for CIFAR100.](/tmp/summary_update_inspect_long/media/image11.png){width=5.83333in}

Figure 11. α gain over α=0 by budget for CIFAR100.

This figure shows that density normalization helps mainly at very
low budgets, and that the gain is not stable across the full active
learning trajectory.

![Figure 12. CIFAR100 α=0 compared with the best α at each budget for global σ and local-σ K=1440.](/tmp/summary_update_inspect_long/media/image12.png){width=5.57068in}

Figure 12. CIFAR100 α=0 compared with the best α at each budget
for global σ and local-σ K=1440.

Figure 12 shows the same result in absolute accuracy rather than gain
over α=0. For both global σ and local-σ K=1440, the best
α improves the early budgets. For local-σ K=1440, α=0.5 is
best from 100 to 400 labels, after which α=0 is usually preferred.
This strengthens the interpretation that density normalization is useful
early, but should be reduced as the selected set grows.

![Figure 13. CIFAR100 global/local σ with α=0 and selected α at each budget.](/tmp/summary_update_inspect_long/media/image13.png){width=5.56731in}

Figure 13. CIFAR100 global/local σ with α=0 and selected α
at each budget.

This figure separates the two effects in one place: local σ changes
the kernel scale, while density normalization changes the influence of
high-degree points. The selected-α curves annotate the α chosen
at each budget. The strongest improvement appears when local σ
K=1440 is combined with α=0.5 in the first 400 labels; later, the
selected α mostly returns to 0, which again suggests that
normalization should be budget-dependent.

### δ Sensitivity And ProbCover Comparison

To test whether the early α benefit depends on the cutoff threshold,
we reran global σ and the best local-K setting with α=0 and
α=0.5 across several δ values.

ProbCover is included here as a coverage baseline, not as a
density-normalization method. Both ProbCover and MaxHerding with cutoff
use graph connectivity to choose representative points: ProbCover uses a
hard binary threshold graph, while MaxHerding uses a smooth weighted
kernel on nearby points. Density normalization changes how much
high-degree points influence the smooth MaxHerding kernel. Comparing to
ProbCover checks whether this smooth, density-normalized coverage
behavior improves over the simpler binary coverage baseline.

![Figure 14. CIFAR10 MaxHerding density normalization compared to ProbCover for global σ and local-σ K=3500.](/tmp/summary_update_inspect_long/media/image14.png){width=5.8381in}

Figure 14. CIFAR10 MaxHerding density normalization compared to
ProbCover for global σ and local-σ K=3500.

![Figure 15. CIFAR100 MaxHerding density normalization compared to ProbCover for global σ and local-σ K=1440.](/tmp/summary_update_inspect_long/media/image15.png){width=5.84081in}

Figure 15. CIFAR100 MaxHerding density normalization compared to
ProbCover for global σ and local-σ K=1440.

![Figure 16. CIFAR10 α=0.5 gain over α=0 across δ values for local-σ K=3500.](/tmp/summary_update_inspect_long/media/image16.png){width=5.57104in}

Figure 16. CIFAR10 α=0.5 gain over α=0 across δ values for
local-σ K=3500.

![Figure 17. CIFAR100 α=0.5 gain over α=0 across δ values for local-σ K=1440.](/tmp/summary_update_inspect_long/media/image17.png){width=5.60785in}

Figure 17. CIFAR100 α=0.5 gain over α=0 across δ values for
local-σ K=1440.

Figures 14 to 18 suggest that δ affects the size of the
density-normalization gain, but not the qualitative trend. Density
normalization can help at very low budgets, especially on CIFAR100, but
the gain weakens as the budget grows and often becomes negative later.
Smaller δ values preserve the early gain longer in some settings, while
larger δ values make the late penalty appear sooner. This supports the
interpretation that α and δ are coupled graph-scale parameters:
a fixed α is too rigid, and useful density penalization probably
needs to weaken or turn off as the selected set grows.

![Figure 18. CIFAR100 absolute accuracy across δ values with α=0.5.](/tmp/summary_update_inspect_long/media/image18.png){width=5.83333in}

Figure 18. CIFAR100 absolute accuracy across δ values with
α=0.5.

Figure 18 presents the δ dependency directly in absolute accuracy.
The cutoff threshold changes the absolute level of performance and the
strength of the α effect, but the main pattern remains: density
normalization is most useful early and becomes less useful later. This
suggests that δ and α should be considered together as
graph-scale parameters.

## 8. Third Direction: Dynamic α / Decay

Because fixed α helped early and hurt later, we tested simple
dynamic α schedules:

-   Early-only α: use the chosen α for the first three
    active-learning rounds, then set α back to 0 for the rest of the
    run.

-   Linear decay: start from the chosen α and reduce it linearly to
    0 over the first five active-learning rounds, then keep α at 0.

-   Three-phase decay: use the chosen α for the first three rounds,
    use half of that α for the next three rounds, and then use
    α=0 for the remaining rounds.

The hope was that these schedules would preserve early gains while
avoiding late penalties.

### Main Result

The simple decay schedules did not improve final accuracy.

They preserved some early gains, especially for CIFAR100, but every
tested schedule ended below the α=0 baseline at the final budget.

### Interpretation

This suggests that the correct adaptation is probably not a simple
hand-written monotonic decay.

The useful rule may need to depend on the actual density/selection state
of the run, not only on the episode number.

These schedules were run both with the best local-σ setting and with
a single global σ. This was important because otherwise the decay
result could be confused with the local-σ effect. In both cases, the
schedules were fixed by episode number, not adaptive to measured graph
statistics. A more principled adaptive rule would change α only when
the selected-set density, graph degree, or rank-like statistics indicate
that the coverage signal has changed.

In other words, α should perhaps be chosen based on what the
selected set already covers, or on graph statistics, rather than
following a fixed schedule.

![Figure 19. Local-σ K=3500 decay schedule gain/loss for CIFAR10.](/tmp/summary_update_inspect_long/media/image19.png){width=5.83333in}

Figure 19. Local-σ K=3500 decay schedule gain/loss for CIFAR10.

![Figure 20. Local-σ K=1440 decay schedule gain/loss for CIFAR100.](/tmp/summary_update_inspect_long/media/image20.png){width=5.91877in}

Figure 20. Local-σ K=1440 decay schedule gain/loss for CIFAR100.

This figure supports the conclusion that simple decay preserves
some early gains but does not improve final accuracy.

![Figure 21. Global-σ decay schedule gain/loss for CIFAR10.](/tmp/summary_update_inspect_long/media/image21.png){width=5.83333in}

Figure 21. Global-σ decay schedule gain/loss for CIFAR10.

![Figure 22. Global-σ decay schedule gain/loss for CIFAR100.](/tmp/summary_update_inspect_long/media/image22.png){width=5.83333in}

Figure 22. Global-σ decay schedule gain/loss for CIFAR100.

Figures 19 to 24 show that the fixed decay schedules do not solve the
main problem. Some schedules preserve early gains in some settings, but
the final-budget effect is still usually worse than α=0. The
global-σ repeats are important because they show that this is not
only a local-σ artifact: simple hand-written decay is too crude even
when σ is fixed globally.

![Figure 23. CIFAR100 absolute accuracy of α decay schedules with global σ.](/tmp/summary_update_inspect_long/media/image23.png){width=5.83333in}

Figure 23. CIFAR100 absolute accuracy of α decay schedules with
global σ.

![Figure 24. CIFAR100 absolute accuracy of α decay schedules with local-σ K=1440.](/tmp/summary_update_inspect_long/media/image24.png){width=5.83333in}

Figure 24. CIFAR100 absolute accuracy of α decay schedules with
local-σ K=1440.

Figures 23 and 24 show the decay experiments in absolute accuracy. The
schedules can preserve part of the early improvement, but none of the
tested fixed schedules gives the desired late-budget behavior. This
motivates the next two schedule variants: keeping α active longer in
the early-only schedule, and using a nonzero final α in the
three-phase schedule instead of turning it off completely.

## 9. Main Scientific Conclusion So Far

The current evidence suggests:

1.  Local σ affects MaxHerding mainly by changing graph degree.
2.  Very sparse graphs hurt badly.
3.  Very dense or very smooth local graphs become close to global σ.
4.  Medium-degree graphs are the most useful regime.
5.  Selected-point density is consistently higher than dataset-average
    density.
6.  Explicit density normalization improves CIFAR100 low-budget
    performance when α is chosen appropriately.
7.  The benefit is budget-dependent: the strongest gains appear early,
    while later budgets require weaker normalization or a different
    adaptive rule.
8.  Simple decay schedules are too crude.

The most important takeaway is:

> Degree remains the strongest measurable explanatory variable in these
> experiments, and it gives a practical path toward improving the method
> through budget-dependent graph control.

This does not mean locality is irrelevant. Degree and locality are
closely related because both are affected by local density. The current
interpretation is that local σ matters mainly through the graph it
induces, and degree is the clearest statistic we found for describing
that graph.

## 10. What We Can Learn

The results shift the research direction.

Initially, the question was:

> Which local σ K is best?

Now the better question seems to be:

> What graph degree or density profile should the active learning
> algorithm induce at each budget?

This reframes the problem from choosing K manually to controlling the
graph structure more directly.

Possible future rules could be based on:

-   Target effective degree
-   Selected-point degree relative to dataset-average degree
-   Budget-dependent density targets
-   Adaptive α chosen from current selection statistics

### Final Degree-Control Evidence

The earlier results suggested that the important object is not K itself,
but the graph degree induced by the kernel and cutoff. To test this more
directly, we performed a final CIFAR100 analysis using MaxHerding with
1NN.

First, we took the completed K-sweep experiments and measured the
dataset effective degree induced by each setting. In this retrospective
analysis, each point in the figure is one completed setting: the x-axis
is the effective degree induced by that setting, and the y-axis is the
matching accuracy at that budget.

![Figure 25. CIFAR100 accuracy versus dataset effective degree.](/tmp/summary_update_inspect_long/media/image25.png){width=5.83333in}

Figure 25. CIFAR100 accuracy versus dataset effective degree.

This figure shows that very small degree performs poorly. Higher degree
helps more at early budgets, but the relationship is not monotone later.
The shaded range is post-hoc: it marks the approximate degree range
where strong settings tended to appear. This analysis is correlational
because K and degree changed together, but it strongly suggests that
graph degree is a better explanatory variable than K alone.

To test the degree explanation more directly, we then ran
controlled-degree experiments. Different `(K, δ)` pairs were chosen
to target low, moderate, and high graph-degree bands:

-   Low degree: about 10-13 effective neighbors
-   Moderate degree: about 18-23 effective neighbors
-   High degree: about 30-36 effective neighbors

![Figure 26. CIFAR100 controlled-degree accuracy by target degree band.](/tmp/summary_update_inspect_long/media/image26.png){width=5.83333in}

Figure 26. CIFAR100 controlled-degree accuracy by target degree band.

The controlled-degree curves show that high degree is worse later in the
active learning run. Low and moderate degree are close and competitive.
This means the useful conclusion is not “higher degree is better.”
Instead, too much degree can hurt, and MaxHerding seems to prefer a
low-to-moderate degree regime on CIFAR100.

The cleanest controlled comparison fixes K and changes δ, so the
main changing factor is the induced graph degree.

![Figure 27. CIFAR100 controlled-degree curves with fixed K=1440.](/tmp/summary_update_inspect_long/media/image27.png){width=5.83333in}

Figure 27. CIFAR100 controlled-degree curves with fixed K=1440.

With K fixed to 1440, the moderate-degree setting is best for most of
the budget range. High degree is not best later, and very low degree is
weaker early. This supports the final interpretation: the useful
graph-density regime is budget-dependent. On CIFAR100, the strongest
settings are roughly in the 18-25 effective-neighbor region, but this
should be understood as a useful regime rather than one exact optimal
degree.

A useful future visual would summarize the best observed K/α
configuration at each budget across seeds. This would directly test
whether the optimal configuration changes in a stable way during the
active learning trajectory.

δ may also need to be budget-dependent. This connects to the
ProbCover observation that the useful δ can decrease as budget
increases: after central regions are already covered, a smaller
threshold can reduce redundant coverage and change which points remain
useful. The δ sweep supports this interaction, especially on
CIFAR100, where smaller δ values preserved the early α benefit longer
than larger δ values.

## 11. Open Questions

The current results answer part of the original question, but they also
open several new questions.

### 11.1 Generality Across Datasets

The strongest conclusion so far is based on CIFAR10 and CIFAR100. We
still need to know whether it holds on other datasets.

Open questions:

-   Does graph degree remain the dominant factor on larger datasets?
-   Does the same small-K failure appear on datasets with more classes
    or more visual diversity?
-   Does the useful degree range scale with dataset size, number of
    classes, or feature quality?
-   Is the degree-based interpretation still true on Tiny ImageNet, or
    is it a CIFAR-specific effect?

Why this matters:

-   If Tiny ImageNet shows the same behavior, the degree explanation
    becomes much stronger.
-   If it does not, then locality may interact with dataset structure in
    a more complicated way.

### 11.2 Degree Versus Locality

The main current hypothesis is that degree matters more than locality
itself. But this is not fully proven.

Open questions:

-   If two kernels have similar effective degree distributions, do they
    produce similar selections and accuracy even if one is global and
    one is local?
-   Can local σ provide benefits beyond degree control?
-   Is K only a proxy for graph degree?
-   Would directly targeting a desired degree perform better than
    choosing K?

Why this matters:

-   If degree is the true control variable, future algorithms should
    control degree directly rather than tune K.
-   If locality has an independent benefit, then we need diagnostics
    that separate locality from degree.

### 11.3 Selection Dynamics Across Budget

The best configuration appears to change over the active learning
trajectory.

Open questions:

-   Why does α normalization help early but hurt later?
-   Does MaxHerding first select central/dense points and only later
    move outward?
-   Can we identify the point where dense-region penalization should be
    reduced or turned off?
-   Is the best α determined by budget, selected-set diversity,
    selected-set density, or something else?

Why this matters:

-   A fixed K or fixed α is probably too rigid
-   The next useful algorithm may need to adapt based on the current
    selected set

This is also why the three-phase schedule is not enough by itself. It
changes α according to a predefined budget schedule, but it does not
look at whether the graph degree has actually become too small, whether
selected points are still unusually dense, or whether the coverage
signal has already weakened.

### 11.4 Best Statistic For Prediction

We measured several density-related statistics, but it is not yet clear
which one best predicts performance.

Candidate predictors:

-   Dataset effective degree
-   Selected-point effective degree
-   Selected-point degree divided by dataset-average degree
-   Raw/global selected density
-   Local σ mean and coefficient of variation
-   Selected label entropy
-   Class diversity of the selected batch

Open questions:

-   Which statistic correlates best with accuracy?
-   Which statistic predicts when a K value is too sparse?
-   Which statistic predicts when α normalization becomes harmful?
-   Can these statistics be used online during active learning?

### 11.5 Better Adaptive Rules

The simple decay schedules were not enough.

Open questions:

-   Can α be chosen dynamically from density statistics rather than
    episode number?
-   Can K be chosen to hit a target effective degree?
-   Can we design a rule that uses density normalization only when
    selected points are too concentrated in dense regions?
-   Can we avoid hurting late-budget accuracy while preserving early
    gains?

## 12. Future Research Directions

These are directions that other people could continue from this project.

### 12.1 Degree-Targeted Active Learning

Instead of manually choosing local σ K, choose K or σ so that
the graph has a desired effective degree.

Possible research question:

> Is controlling graph degree directly more robust than choosing local
> σ K?

Possible approach:

-   Define a target average effective degree
-   Choose K or σ to match that target
-   Compare against global σ and fixed local K

### 12.2 Adaptive Density Penalization

Instead of using fixed α, choose α based on the current state of
the selected set.

Possible research question:

> Can density normalization improve early-budget selection without
> hurting later budgets if α is selected adaptively?

Possible approach:

-   Measure selected/dataset degree ratio after each episode
-   Apply α only if selected points are much denser than the dataset
    average
-   Reduce α when selected diversity is high enough

Another possible adaptive rule is to switch strategy when the graph
signal becomes weak. If the average degree becomes uniformly very small,
MaxHerding may no longer have a meaningful coverage signal. In that
case, the method could switch to random selection or mix MaxHerding with
random/diversity sampling, related to the motivation behind SelectAL
(Hacohen and Weinshall, NeurIPS 2023).

### 12.3 Separating Degree From Geometry

Design controlled experiments where two kernels have similar degree but
different geometric/locality structure.

Possible research question:

> Does locality matter after controlling for degree?

Possible approach:

-   Compare global and local kernels matched by average effective degree
-   Compare their selected sets
-   Measure overlap, class diversity, and accuracy

### 12.4 Dataset Scaling Study

Repeat the same analysis on additional datasets.

Candidate datasets:

-   Tiny ImageNet
-   STL10
-   ImageNet subset
-   Domain-specific datasets if available

Possible research question:

> Is the degree-dominance phenomenon universal, or does it depend on
> dataset scale and feature quality?

The most useful figure for this direction would be an effective-degree
versus final-accuracy plot for Tiny ImageNet, matching the CIFAR10 and
CIFAR100 plots.

### 12.5 Selection-Set Geometry Analysis

Analyze not only individual selected-point density but the geometry of
the full selected set.

Possible metrics:

-   Pairwise distance between selected points
-   Class coverage
-   Cluster coverage
-   Redundancy among selected points
-   Distance from selected points to nearest dataset modes

Possible research question:

> Does MaxHerding fail when selected points are individually dense,
> mutually redundant, or both?

## 13. Short Thesis-Style Takeaway

The project suggests a shift in how we should think about local kernels
in coverage-based active learning. We started from local σ and K,
expecting localization to help balance dense and sparse regions. The
experiments showed that local σ is not automatically useful just
because it is local. Its main effect is to change the graph induced by
the kernel, especially the effective degree of points.

Once viewed this way, many results become more coherent: small K fails
because the graph is too sparse, large K resembles global σ, density
normalization helps mainly when dense-region selection is too dominant,
and simple decay schedules fail because they do not react to the actual
graph state.

The final CIFAR100 degree-control analysis strengthens this
interpretation. MaxHerding tends to select high-degree representative
points early, the remaining pool becomes less dense, and
controlled-degree experiments show that too-high degree hurts later. The
useful regime appears to be low-to-moderate graph degree, roughly around
18-25 effective neighbors in our CIFAR100 setting, but it is
budget-dependent rather than globally fixed.

The main lesson is that future work should focus less on manually tuning
K and more on controlling graph density, cutoff scale, and selected-set
density through adaptive rules.
