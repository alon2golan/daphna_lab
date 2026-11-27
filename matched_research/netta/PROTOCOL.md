# Paired NPC and peer-ranked AUM on BYOL

The earlier runner-up cap, square-root votes, backup coverage, and VAST results are pilots. The cap measured a change to graph voting. This experiment measures a change to the final training mask of the released Netta NPC pipeline.

## Reference

Run Shafir et al.'s released NPC + LowBudgetAUM and their Framework 2 linear fit from pinned revision `733651bb0943a96392895103eb2ffc4e99a7c7d1`. Use the aligned CIFAR-100 BYOL feature bank with 50,000 training and 10,000 test images. Start acquisition from an empty labeled set, in batches of 100. Use the release's constant Euclidean radius 0.65 on unit-norm features for query selection. Use the official feature handling for the detector and final classifier. Fit a linear classifier for 500 epochs using batch size 100, SGD, Nesterov momentum 0.9, weight decay 0.0003, and cosine learning rate from 0.025. Source and data fingerprints accompany the runs. [Paper and protocol](https://arxiv.org/html/2504.04506v1) · [Pinned release](https://github.com/nettashafir/active_learning_with_label_noise/tree/733651bb0943a96392895103eb2ffc4e99a7c7d1)

The baseline and candidate use the same acquisition trajectory, observed labels, AUM model outputs, model initialization, scheduled optimization, minibatch permutation stream, and test images. Score the baseline before fitting the candidate from its saved training state. Check that an identity replay reproduces the released fit. The class counts and total number of training items are paired by construction.

Three fresh seeds, 71–73, share the paired feature bank and noise generator. Checkpoints are 400, 800, 1,200, 1,600, and 2,000 queried labels. Conditions are clean, symmetric 30%, 40%, 50%, and within-superclass cyclic asymmetric 30%, 40%, 50%. The asymmetric channel is the lab's existing protocol. Report every three-seed budget/condition mean, all per-seed results, and the win counts by noise condition. Budgets within a seed are nested.

## Algorithm

Let `L` be acquired images, `U` be unqueried images, and `ỹ[i]` the noisy annotator label. Let `D_i = {j : ‖z_i − z_j‖₂ < 0.65}` for unit-norm query features. The released acquisition loop is shared.

```text
L = empty, U = all training images, K = empty
for each round:
    Q = empty
    repeat 100 times:
        i = argmax over U minus Q of |D_i minus union(D_j for j in K ∪ Q)|
        add i to Q
    run the release's internal NPC AUM refresh after the batch
    L = L ∪ Q, U = U minus Q
    AUM scores, K = released outer AUM pass on L and U
    reference training set = K
    fit the released supervised classifier from its saved initialization
```

For AUM, add `ceil(|L|/100)` threshold examples from `U` with one fake class and fit the detector for 40 epochs. The per-image score is the mean observed-class logit minus strongest other logit across its training visits. The reference keeps `K = {i in L : AUM[i] ≥ P80(AUM[fakes])}` as its final training set.

The candidate changes final retention only. Use locally scaled neighbor agreement among the acquired images. For each image `i`, let `N_i` be its 13 closest other acquired images by Euclidean distance between unit BYOL embeddings. Let `b_i` be the largest squared distance in `N_i`. When `b_i=0`, set it to 1.

```text
peer[i] = sum(j in N_i) exp(-dist²(i,j) / b_i) * [ỹ[j] == ỹ[i]]
          / sum(j in N_i) exp(-dist²(i,j) / b_i)

for each observed class c:
    rows = [i in L with ỹ[i] == c]
    quota = count(i in rows and i in K)
    score[i] = midrank_in_class(AUM[i]) + midrank_in_class(peer[i])
    keep the top quota rows by score
    break ties by baseline-kept status, higher AUM, then acquisition order

train the released linear classifier on the resulting set
predict argmax of its logits on each test image
```

Ranks increase with agreement and AUM. Within each class, keeping the highest rank sums maximizes the total retained-over-rejected pairwise comparisons favored by the two signals, with tied comparisons counting one half. Comparisons within the retained set sum to a constant for a fixed quota. This score allows peer evidence to offset the AUM rank difference while preserving the released filter's class counts. Under independent uniform wrong-class flips, the noisy class probability is an increasing affine function of clean class probability for rates below 99%. This motivates the peer score under symmetric noise. The behavior under asymmetric noise is empirical. [Related representation-based filtering](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f4e3ce3e7b581ff32e40968298ba013d-Abstract.html)
