---
title: "Bayesian Coverage — How It Works + Results vs. Netta's NPC/AUM"
date: 2026-07-08
---

**Branch:** `bayesian_coverage` (HEAD `236cf81`)
**Companion docs:** design spec in `docs/bayesian_coverage_DESIGN.md`; implementation in
`deep_al/pycls/al/bayesian_coverage.py`; training wiring in `deep_al/tools/train_al.py`
(`_bayes_training_loader`) + `deep_al/pycls/noise/noisy_label_wrapper.py`.
Everything in this file is verified against the code and the actual `result.json` files on disk.

---

## 0. TL;DR

We work with **noisy annotators** and act on **two planes**:

1. **Point selection (AL):** replace ProbCover's binary covered/not-covered with a
   **per-class posterior distribution** per point, built from neighbor annotations,
   each weighted by its **annotator's confusion matrix** and its **kernel distance**,
   fused as a **product of experts**. Select the point whose ball contains the most
   remaining *uncertainty mass*.
2. **Training:** reuse the same posterior to build the training targets — six
   configurable policies (`TRAIN_LABEL_MODE`), from raw noisy labels to full soft
   labels to curriculum pseudo-labeling of unlabeled points (`pseudo`, §4.6).

**Headline result (CIFAR100):** the selection plane alone $\approx$ ProbCover; the *training*
plane is where the method pays. At noise 0.5/0.8, training on the posterior
(`soft_posterior` / `map_posterior`) beats both raw-label training ($2.2\times$ at noise 0.8)
**and Netta's NPC+AUM**. The §7.5 **pseudo-labeling arm (§4.6) then beats every other
arm at every noise level *including clean*** (e.g. 47.8 vs 45.1 observed at noise 0
@1000; 45.8 vs NPC+AUM's 43.3 at noise 0.2) — closing the one regime where Netta's
stack had led.

---

## 1. The probabilistic model (selection plane)

Class: `BayesianCoverage(CoverageMatrixMethod)` — a new subclass; the base
ProbCover/NPC path is byte-untouched.

### 1.0 Notation — every symbol used below

| Symbol | Meaning | In the code |
|---|---|---|
| $C$ | number of classes (100 for CIFAR100) | `num_of_classes` |
| $u, v$ | an arbitrary / candidate **pool point** | row index into `self.C` |
| $\ell$ | a **labeled point** (a "neighbor" once annotated); $L$ = the set of all labeled points | `lSet` + in-episode picks |
| $y_u$ | the (unknown) **true label** of point $u$ | — |
| $\tilde{y}_\ell$ | the **observed label** the annotator gave $\ell$ — possibly a noise flip of the true $y_\ell$ | `annotated_labels[g]` |
| $a(\ell)$, $\rho_a$ | the annotator that labeled $\ell$; its accuracy $\rho_a = 1-\text{noise rate}$ | `Annotator.accuracy` |
| $M^a[k,j]$ | the annotator's **confusion matrix** ($C{\times}C$, rows sum to 1): $M^a[k,j] = P(\text{annotator says } j \mid \text{true class is } k)$. The entry $M^a[k,\tilde{y}_\ell]$ used in Step A reads: "if $\ell$'s true class were $k$, how likely was this annotator to answer $\tilde{y}_\ell$?" — the **likelihood** in Bayes' rule. Symmetric annotator: diagonal $=\rho_a$, off-diagonal $=(1-\rho_a)/(C-1)$; an asymmetric annotator just supplies a different matrix | `Annotator.confusion_matrix`, `_confusion_matrix()` |
| $\pi(k)$ | the **class prior** — belief about any point's class before seeing labels; uniform $1/C$ in all runs | `self.pi` |
| $K[u,\ell]$ | the **kernel**: the probability that $u$ shares $\ell$'s true label, from feature distance — ProbCover's "same $\delta$-ball $\Rightarrow$ same label" assumption made explicit. Tophat: $K[u,\ell]=1$ if $\lVert f_u - f_\ell\rVert \le \delta$, else $0$ | `self.K` (base class), $\delta=0.75$ |
| $b_\ell(k)$ | **Step-A belief**: $P(y_\ell = k \mid \tilde{y}_\ell)$ — what we believe about $\ell$'s **own** true label after passing the annotation through the confusion matrix | `_belief()` |
| $q_\ell(u,k)$ | **Step-B message**: the likelihood that neighbor $\ell$ contributes about **point $u$'s** label — $b_\ell$ diluted toward the prior by distance. See the "how to read $q_\ell$" box in Step B | `_neighbor_loglik()` |
| $w_{u\ell}$ | per-neighbor **weight** on that message; default `kernel`: $w_{u\ell}=K[u,\ell]$ | `NEIGHBOR_WEIGHT` |
| $\beta$ | global **tempering exponent** applied to every message (§1.4); default $1.0$ = off | `TEMPER_BETA` |
| $C[u]$ | $u$'s row of the coverage accumulator = the **log unnormalized posterior** over classes (product mode) | `self.C` |
| $P(y_u=\cdot)$ | the usable **posterior** $=\operatorname{softmax}(C[u])$ | `posterior()` |
| $P^{\text{LOO}}_\ell$ | **leave-one-out posterior** of a labeled point — its posterior with its own vote subtracted (§2) | `_loo_posterior_for()` |

Every pool point $u$ carries a distribution $P(y_u = k)$ over all classes instead of a
covered bit. It is built in three steps:

### Step A — annotator noise enters (`_belief`, bayesian_coverage.py:171)

For a labeled point $\ell$ with observed label $\tilde{y}_\ell$ from annotator $a$, the belief over
its **own true label** is Bayes' rule through the annotator's confusion matrix
$M^a[k,j] = P(\text{observed } j \mid \text{true } k)$:

$$
b_\ell(k) \;\propto\; \pi(k)\cdot M^a[k,\, \tilde{y}_\ell]
$$

- $\pi$ = the **class prior** (uniform, $1/C$). Terminology note: the prior is $\pi$, *not* the
  annotator-confidence vector — $b_\ell$ is already a (Step-A) posterior on $\ell$'s own label.
- Symmetric annotator with accuracy $\rho$:

$$
b_\ell(k) =
\begin{cases}
\rho & k = \tilde{y}_\ell \\[2pt]
\dfrac{1-\rho}{C-1} & k \neq \tilde{y}_\ell
\end{cases}
$$

  E.g. a 50%-noise annotator on CIFAR100: $0.5$ on the observed class, $0.5/99$ elsewhere.
- The **full confusion matrix** is used, so asymmetric annotators work for free
  (commit `9b54ce0`; `Annotator.confusion_matrix`).

### Step B — geometry enters (`_neighbor_loglik`, line 189)

The belief propagates to any other point $u$ through the kernel edge, diluted by distance:

$$
q_\ell(u,k) \;=\; K[u,\ell]\cdot b_\ell(k) \;+\; \bigl(1-K[u,\ell]\bigr)\cdot \pi(k)
$$

**How to read $q_\ell(u,k)$.** It answers: *"given only what neighbor $\ell$ knows, how
likely is it that point $u$ belongs to class $k$?"* — one neighbor's "message" about $u$,
a full distribution over the $C$ classes (one $q_\ell(u,\cdot)$ vector per (neighbor, point)
pair). It is a **mixture of two hypotheses**, weighted by the kernel:

- With probability $K[u,\ell]$: "$u$ and $\ell$ share the same true label" (the ProbCover
  ball assumption holds) — then everything we believe about $\ell$'s label, $b_\ell$,
  transfers to $u$ as-is.
- With probability $1-K[u,\ell]$: "$u$ is unrelated to $\ell$" — then $\ell$ tells us
  nothing about $u$, and the message falls back to the prior $\pi$.

Limit cases, with a 60%-accurate annotator on CIFAR100 that said class 7:

| Situation | $K[u,\ell]$ | $q_\ell(u,\cdot)$ |
|---|---|---|
| $u$ right next to $\ell$ | $1$ | $= b_\ell$: $0.6$ on class 7, $0.4/99$ elsewhere |
| $u$ far from $\ell$ | $0$ | $= \pi$: uniform $1/100$ — no information |
| smooth kernel, mid-distance | $0.5$ | halfway: $0.303$ on class 7, ~$0.007$ elsewhere |

Close neighbor + accurate annotator $\Rightarrow$ peaked toward $\tilde{y}_\ell$. Far neighbor **or** noisy
annotator $\Rightarrow$ flat toward the prior. With the tophat kernel (what all our runs use),
$K \in \{0,1\}$ — inside/outside the $\delta$-ball — so distance acts as a gate: in-ball
neighbors send $b_\ell$, out-of-ball neighbors send the (information-free) prior.

### Step C — fuse all labeled neighbors (product of experts)

All labeled neighbors are combined in log-space (naive Bayes over neighbors):

$$
C[u] \;=\; \log \pi \;+\; \sum_{\ell \in L} \beta\, w_{u\ell}\, \log q_\ell(u,\cdot),
\qquad
P(y_u = \cdot) \;=\; \operatorname{softmax}\bigl(C[u]\bigr)
$$

Equivalently in probability space — each neighbor's message enters as a **fractional
vote** with exponent $\beta\, w_{u\ell}$:

$$
P(y_u = k) \;\propto\; \pi(k)\, \prod_{\ell \in L} q_\ell(u,k)^{\,\beta\, w_{u\ell}}
$$

(In the code, `_neighbor_loglik` returns exactly `temper_beta * w * log q`, accumulated
into `self.C` by `_fold_neighbor`.)

- **Product, not addition**: agreeing neighbors *sharpen* the posterior; a single
  confident-but-wrong neighbor is overridable. The **additive** rule
  ($C[:,\text{label}] \mathrel{+}= K_{\text{row}}$ — Itai David's `BAYES_MISP`) is kept behind
  `COMBINE=additive` as the empirical control (design §3 demanded the comparison,
  not an assumption). Result: product wins decisively (see §5.4).

### 1.4 The weights $w_{u\ell}$ and the temper $\beta$ — correlated-neighbor overconfidence

The product above is naive Bayes: it assumes every labeled neighbor is an
**independent** piece of evidence. Neighbors that are close *to each other* violate
this — they describe the same local patch, so their labels are near-duplicates of one
measurement, and a raw product counts the same information several times over.

Worked example (2 classes): five clustered neighbors, each giving $u$ a message that is
$0.7$-confident for class A.

- Naive product ($\beta = 1$, full votes): odds $= (0.7/0.3)^5 \approx 69{:}1
  \Rightarrow P(A) \approx 0.99$.
- But if the five points are effectively one measurement, the honest belief is
  $\approx 0.7$. The raw product is badly **overconfident**.

Overconfidence hurts both planes: an overconfident posterior looks "covered", so the
acquisition score stops sampling that region; and `soft_posterior` training targets
inherit the false peak.

Two knobs shrink the per-neighbor exponent (design §2.4):

1. **Kernel weighting** — `NEIGHBOR_WEIGHT='kernel'` (default): $w_{u\ell} = K[u,\ell]$.
   Each message is scaled by its relevance to $u$: a neighbor on top of $u$ casts a full
   vote, a marginal one a fraction. *Caveat for our runs:* with the tophat kernel
   $K \in \{0,1\}$, so this degenerates to a gate — every in-ball neighbor still casts a
   full vote. It only becomes a real gradient with a smooth kernel (e.g. RBF).
2. **Tempering** — `TEMPER_BETA` $= \beta < 1$: multiply *every* message uniformly, i.e.
   raise every likelihood to the power $\beta$ ("fractional Bayes"). With $N$ agreeing
   neighbors the effective evidence count becomes $\beta N$ instead of $N$: in the example,
   $\beta = 0.2$ turns $(0.7/0.3)^5$ back into $(0.7/0.3)^1 = 0.7$.

The two are complementary and multiply: $w_{u\ell}$ discounts **per neighbor by
relevance** ("this neighbor is far"), $\beta$ discounts **all neighbors uniformly**
("I distrust the independence assumption as such").

**"Isn't distance already inside $q_\ell$ — why multiply by $\beta w$ again?"**
Legitimate question; three-part answer.

1. *The exponent is not needed for correctness.* With a uniform prior,
   $q_\ell(u,k)$ is exactly the naive-Bayes likelihood
   $P(\tilde{y}_\ell \mid y_u = k)$ up to a class-independent constant
   (marginalize over "do $u,\ell$ share a label?": w.p. $K$ the likelihood is
   $M^a[k,\tilde{y}_\ell]$, w.p. $1-K$ it is constant in $k$). So the *exact*
   Bayesian fusion is the plain product: exponent $1$, i.e. `w=equal`, $\beta=1$.
2. *But the two factors do different jobs.* Inside-$K$ sets the **content** of one
   message (how peaked); the exponent sets its **evidence count** (how many votes).
   Dilution alone cannot bound accumulation: $N$ correlated neighbors each sending a
   mild message still accumulate $N\log q$, and as $N$ grows the softmax converges to
   a one-hot — any slightly-peaked message, repeated, multiplies into (false)
   certainty. Only an exponent $<1$ slows the accumulation itself.
3. *Using $w=K$ as that exponent is a heuristic, and it does double-count distance*
   under a smooth kernel (a $K=0.5$ neighbor is diluted *and* half-weighted). It is
   §2.4's cheap proxy — and note it doesn't even measure what causes the correlation
   (neighbor-to-neighbor closeness, not distance to $u$). The principled knob is the
   uniform $\beta$ (standard power/fractional-posterior practice). For smooth-kernel
   runs, `NEIGHBOR_WEIGHT=equal` + tuned $\beta$ is the cleaner configuration.

   With tophat this is all moot: $K\in\{0,1\}$ makes the exponent exactly $1$ in-ball,
   and an out-of-ball neighbor contributes nothing under either weighting (its
   $q=\pi$ is class-constant and cancels in the softmax) — so the experiments in §4
   ran the exact naive-Bayes rule, with no double counting.

**Status in the experiments below:** defaults are $\beta = 1.0$ (tempering off) and
tophat kernel (weights degenerate to 1 in-ball) — so all results in §4 were obtained
with the **raw, untempered product**. The design-§10 check "overconfidence controlled by
`TEMPER_BETA` / `NEIGHBOR_WEIGHT`" is still unverified; a cheap $\beta$-sweep
($\beta \in \{0.25, 0.5, 1.0\}$) is an open follow-up and might specifically help
`soft_posterior` at high noise, where posterior sharpness directly shapes the
training targets.

### Selection criterion — what "maximizing the updated cover" means

Coverage is now **graded**: a point is well-covered when its posterior is peaked,
uncovered when flat. The acquisition score (`_acquisition_scores`, line 211):

$$
\text{uncertainty}(u) \;=\; 1 - \max_k P(y_u = k)
$$

$$
\text{score}(v) \;=\; \sum_u K[v,u]\cdot \text{uncertainty}(u)
\qquad \text{(unresolved mass inside } v\text{'s ball)}
$$

Greedy loop picks $\arg\max_v \text{score}(v)$ — the soft generalization of ProbCover's
"ball covering the most uncovered points". Two load-bearing mechanics:

- **Annotation is in-loop** (`_select_one`, line 250): each pick is annotated
  immediately and folded into $C$, so the *next* pick already sees the updated
  posteriors.
- **Noise-awareness for free:** a label from a noisy annotator barely sharpens its
  ball's posteriors, so the region *stays attractive for re-sampling* — the method
  knows one noisy label didn't really cover it.
- Cross-episode: $C$ resets to the prior each AL episode, then the whole labeled set
  is **re-folded** from annotation records (`_fold_labeled_set`, line 124); without
  this, prior episodes' evidence would be lost and episode 2+ would re-sample regions
  episode 1 already labeled (their neighbors would look uncovered again).
  *Note — this is an engineering choice, not part of the algorithm:* the posterior
  is in fact persisted (`C_general`, global-indexed — training reads it from there),
  but the per-episode $C$ is locally indexed and the index order changes as the
  labeled set grows, so rebuilding is simpler than remapping. Replaying costs one
  vector-add per labeled point (milliseconds) and is correct by construction: the
  posterior is a pure function of the recorded (label, annotator) pairs.

Still `NotImplementedError`: `entropy` (BALD-style) acquisition, empirical prior,
sparse-K, `NoiseAwareBayesianCoverage` (soft per-point reliability $r_\ell$, design §4).

---

## 2. Training plane — `TRAIN_LABEL_MODE` (design §5, commit `61e7557`)

After selection, the same posterior builds the training targets.
Wired in `train_al.py::_bayes_training_loader` (line 744); soft modes go through
`SoftTargetDatasetWrapper` + a soft-cross-entropy branch. **Five arms, all swept:**

| Mode | Classifier trains on | Loss path |
|---|---|---|
| `observed` | raw annotated (noisy) labels — baseline | standard CE |
| `map_posterior` | each labeled point **relabeled to argmax of its posterior** (self-correction) | standard CE |
| `filtered` | only points whose **neighbors agree** with their annotation | standard CE, subset |
| `soft_posterior` | the full posterior vector as a **soft label** | soft CE |
| `reweighted` | one-hot on observed label, **scaled by reliability** $w$ | soft CE (weight = row sum) |

**"Loss path"** = which cross-entropy (CE) branch the mode takes in the training loop
(`train_al.py:1044`) — it *is* the loss:

- **standard CE**: the usual hard cross-entropy against an integer class label —
  equivalently a **one-hot** target vector.
- **soft CE**: the target is a **full probability vector** $t \in [0,1]^C$
  (e.g. the posterior, sums to 1), and the loss is
  $-\sum_k t_k \log \operatorname{softmax}(z)_k$. Mechanically: soft modes ship a
  2-D float label batch $[B, C]$, and the loop switches on `labels.dim() == 2`.
- **`reweighted` twist**: its target row is $w_\ell \times$ one-hot, which sums to
  $w_\ell$, *not* 1 — running it through the same soft-CE formula automatically
  scales that sample's loss by $w_\ell$, so the row sum doubles as the per-sample
  weight (no separate weighting code).

**`filtered` in detail** (common misreading — it is *not* a threshold on the posterior
value): it computes the **leave-one-out (LOO) posterior** $P^{\text{LOO}}_\ell$ — the point's
posterior with its *own vote removed* (`_loo_posterior_for`, line 295). Necessary because
the self-vote ($K[\ell,\ell]=1$) dominates and a point would almost never disagree with itself.

$$
\text{keep } \ell \iff \arg\max_k P^{\text{LOO}}_\ell(k) \;=\; \tilde{y}_\ell
$$

*Your neighbors, judging you without your input, agree with the annotator.*
`reweighted` uses the same signal continuously:

$$
w_\ell \;=\; P^{\text{LOO}}_\ell\bigl(\tilde{y}_\ell\bigr)
$$

— suspect points are down-weighted, not dropped.

**Key design point:** the clean/dirty signal is the **graph posterior itself**
(design §4b) — no trained model, no AUM, available from round one, and non-circular
(AUM judges labels with a network trained on those same labels).

---

## 3. The comparison baseline — Netta's NPC + AUM

Same problem, two *hard* mechanisms (`noise_filters.py`,
`NoiseAwareCoverage` in `coverage_matrix_methods.py`):

- **AUM filter (training plane).** AUM = Area Under the Margin: while training on the
  noisy labeled set, accumulate per-sample over epochs the margin between the
  assigned-label logit and the largest other logit:

$$
\text{AUM}(x,\tilde{y}) \;=\; \frac{1}{T}\sum_{t=1}^{T}
\Bigl( z^{(t)}_{\tilde{y}}(x) \;-\; \max_{k \neq \tilde{y}} z^{(t)}_{k}(x) \Bigr)
$$

  Symbols: $x$ = one training sample; $\tilde{y}$ = its **assigned** (possibly noisy)
  label; $t$ = training-epoch index, $T$ = number of epochs measured over;
  $z^{(t)}_k(x)$ = the network's **logit** (pre-softmax score) for class $k$ on
  sample $x$ at epoch $t$. So $z^{(t)}_{\tilde{y}}(x)$ is the score of the label the
  sample was given, $\max_{k \neq \tilde{y}} z^{(t)}_k(x)$ is the score of the
  strongest *competing* class, and their difference is the **margin** at epoch $t$
  (positive = the network currently agrees with the assigned label). AUM is that
  margin **averaged over the whole training run** — the "area under" the
  margin-vs-epoch curve. Mislabeled samples fight the network $\Rightarrow$ low/negative AUM. Threshold $\Rightarrow$
  binary clean/noisy mask $\Rightarrow$ **train only on the clean subset** (`TRAIN_ON_CLEAN`).
- **NPC (selection plane).** ProbCover, except coverage contributed by
  points flagged noisy is **removed** — a noisy point's ball no longer counts as
  covered, so the region gets re-sampled.

**Relationship:** NPC is exactly the limiting case of Bayesian coverage with
reliability hard-thresholded to $r_\ell \in \{0,1\}$ (design §4) — delete vs. down-weight,
and AUM-as-judge vs. graph-as-judge.

---

## 4. Results — CIFAR100, test accuracy (%), mean±std over seeds {1,2,3}

Like-for-like regime: CIFAR100 SimCLR features, budget 100/episode, tophat $\delta=0.75$,
`from_features` eval. Sources (all complete, per-run `result.json` on disk):

| Sweep | SLURM job | Grid | Status |
|---|---|---|---|
| Bayes comparison | 30978366 | {random, probcover, bayes_additive, bayes_product} × noise {0, .2, .5, .8} × 3 seeds | 48/48 ✔ |
| Bayes train-mode | 30980646 | bayes_product × 5 modes × noise {0, .2, .5, .8} × 3 seeds | 60/60 ✔ |
| Netta NPC repro | 30811243 | {npc, probcover, random} + AUM/clean-train × noise {.2, .5, .8} × 3 seeds | 27/27 ✔ |
| AUM clean ablation | 30863746 | {npc, probcover, random} × clean {on, off} × noise {.2, .5, .8} × 3 seeds | 54/54 ✔ |
| §7.5 pseudo-labeling | 31047129 | bayes_product + pseudo × {all, lin10, agg50} × noise {0, .2, .5, .8} × 3 seeds | 36/36 ✔ |

All `bayes_*` selection rows train on **raw observed labels** unless a train-mode is
named. Netta's sweeps have **no noise-0 arm** (never run) — hence the dashes.
Regenerate these tables with the script noted in §6.

**Netta-side decomposition (rows from the AUM ablation, job 30863746):** the two `npc`
rows separate her two mechanisms. *npc, no clean-train* = NPC **selection only**, trained
on all (noisy) labels — it is $\approx$ probcover everywhere (identical at 100 by
construction: the AUM filter only kicks in after the first training round). *npc + AUM
clean-train* = the full Netta stack. So on her side too, **nearly all the gain over
ProbCover comes from the training plane** (clean-train), not from selection — mirroring
our own selection-vs-training split (§5.1–2). Note clean-train *hurts* at small budgets
under low noise (0.2: 32.0 vs 33.9 at 300 — dropping data you can't spare) and only pays
from ~700 labels on.

**"graph posterior alone" row:** the posterior used as a classifier directly on the
test set, with **no trained network** — each test point is classified by the argmax of
its in-ball posterior (bayes_product trajectories; an uncovered test point keeps the
uniform prior and counts as wrong). Test-set coverage is only 36–46% at 1000 labels,
so this row is bounded by *reach*, not knowledge: on **covered** test points alone it
scores 67.1 / 63.5 / 56.1 / 39.5 (%) at noise 0/.2/.5/.8 @1000 — under heavy noise
that beats every trained classifier in the table. Computed by
`batch_utils/test_posterior_acc.py`; the full posterior-quality analysis is §4.5.

### Noise = 0.0 (clean sanity)

| Method | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| random | 8.5±2.6 | 25.1±0.7 | 33.8±0.2 | 38.4±1.4 | 42.6±1.0 |
| probcover | 23.6±1.8 | **39.9±0.4** | **43.3±0.3** | **44.6±0.2** | **46.5±0.6** |
| bayes_additive (Itai) | 13.4±0.7 | 18.2±1.1 | 20.9±0.5 | 23.2±0.9 | 26.9±0.0 |
| bayes_product | **24.6±1.3** | 38.1±1.3 | 41.6±0.0 | 43.6±0.2 | 45.1±0.1 |
| bayes_product + map_posterior | 22.5±0.5 | 35.9±0.4 | 38.9±0.2 | 40.8±0.4 | 41.8±0.5 |
| bayes_product + soft_posterior | 23.7±1.0 | 36.3±0.7 | 39.2±0.0 | 41.1±0.5 | 42.5±0.1 |
| bayes_product + filtered | 5.6±1.4 | 24.2±0.4 | 31.3±0.9 | 36.1±1.0 | 38.6±0.6 |
| bayes_product + reweighted | 5.8±1.0 | 25.1±1.1 | 32.9±1.0 | 36.9±0.7 | 39.6±0.5 |
| graph posterior alone (no training) | 18.1±0.2 | 23.9±0.4 | 26.8±0.3 | 28.8±0.3 | 30.8±0.2 |
| probcover + AUM clean-train | — | — | — | — | — |
| npc + AUM (Netta) | — | — | — | — | — |

### Noise = 0.2

| Method | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| random | 5.8±2.3 | 19.9±1.6 | 28.2±0.9 | 33.3±1.8 | 37.7±1.6 |
| probcover | 19.7±0.7 | 33.4±1.3 | 37.4±0.4 | 39.5±0.3 | 41.2±0.8 |
| bayes_additive (Itai) | 7.7±1.6 | 12.7±0.8 | 14.3±0.8 | 16.9±1.0 | 19.5±0.9 |
| bayes_product | 21.6±1.5 | 32.8±1.5 | 36.6±0.5 | 37.7±0.7 | 39.2±0.3 |
| bayes_product + map_posterior | 20.6±2.3 | 33.7±2.0 | 36.9±1.1 | 39.2±0.8 | 40.6±0.6 |
| bayes_product + soft_posterior | 19.1±1.7 | 33.4±1.9 | **37.5±1.0** | 40.0±0.5 | 41.5±0.4 |
| bayes_product + filtered | **22.4±2.4** | 28.9±1.5 | 35.8±1.0 | 38.7±1.6 | 40.8±0.7 |
| bayes_product + reweighted | 8.6±1.5 | 26.3±2.6 | 33.5±1.5 | 37.1±0.8 | 40.2±0.5 |
| graph posterior alone (no training) | 14.3±0.8 | 21.4±0.7 | 24.5±0.1 | 26.4±0.4 | 28.6±0.5 |
| probcover + AUM clean-train | 20.2±0.7 | 31.2±1.1 | 36.4±0.5 | 39.5±1.3 | 42.3±0.1 |
| npc, no clean-train (selection only) | 19.7±0.7 | **33.9±1.0** | 37.4±1.7 | 39.8±1.1 | 42.0±0.7 |
| npc + AUM clean-train (Netta) | 20.2±0.7 | 32.0±0.6 | 36.5±0.6 | **40.4±0.8** | **43.3±0.6** |

### Noise = 0.5

| Method | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| random | 4.6±1.6 | 12.7±2.2 | 18.4±1.4 | 23.5±1.0 | 27.1±1.0 |
| probcover | 13.1±1.3 | 21.5±1.8 | 26.0±1.3 | 28.1±0.7 | 30.3±1.2 |
| bayes_additive (Itai) | 5.6±1.1 | 6.0±0.8 | 8.3±1.1 | 9.9±1.5 | 12.4±1.1 |
| bayes_product | 14.9±1.4 | 23.1±0.1 | 26.4±0.6 | 28.1±0.6 | 29.5±0.4 |
| bayes_product + map_posterior | **15.2±1.9** | **27.9±1.7** | 31.9±0.2 | 34.5±1.0 | 37.6±0.4 |
| bayes_product + soft_posterior | 12.3±0.9 | 27.8±1.0 | **33.1±0.5** | **36.2±0.1** | **39.1±0.8** |
| bayes_product + filtered | 4.3±1.0 | 16.5±2.3 | 23.8±2.7 | 29.5±1.3 | 34.3±0.5 |
| bayes_product + reweighted | 5.0±2.6 | 17.6±2.4 | 26.3±1.4 | 32.1±0.6 | 36.5±0.8 |
| graph posterior alone (no training) | 8.7±1.2 | 16.0±0.7 | 19.1±0.4 | 21.4±0.1 | 23.8±0.3 |
| probcover + AUM clean-train | 13.3±1.5 | 20.4±2.6 | 26.0±1.7 | 29.6±1.7 | 33.7±0.8 |
| npc, no clean-train (selection only) | 13.1±1.3 | 22.1±1.3 | 25.8±1.5 | 28.6±1.4 | 30.9±1.0 |
| npc + AUM clean-train (Netta) | 13.3±1.5 | 20.7±1.0 | 26.6±1.3 | 30.4±1.7 | 35.6±1.0 |

### Noise = 0.8

| Method | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| random | 2.6±0.6 | 4.9±2.2 | 7.2±1.7 | 9.4±1.1 | 10.3±1.5 |
| probcover | **5.9±0.9** | 8.2±1.1 | 10.4±0.4 | 11.6±0.3 | 12.7±0.1 |
| bayes_additive (Itai) | 2.0±0.0 | 3.3±0.4 | 3.5±0.4 | 4.8±0.1 | 5.4±0.4 |
| bayes_product | 4.6±0.5 | 9.7±2.0 | 10.9±1.5 | 11.8±1.5 | 12.6±1.6 |
| bayes_product + map_posterior | 5.8±0.2 | **14.2±2.9** | **19.4±2.1** | 22.9±2.2 | 26.2±1.3 |
| bayes_product + soft_posterior | 5.3±0.5 | 13.3±2.0 | 18.9±2.4 | **23.4±2.5** | **27.5±1.9** |
| bayes_product + filtered | 1.3±0.6 | 6.5±0.9 | 10.2±0.2 | 15.7±1.3 | 20.6±2.5 |
| bayes_product + reweighted | 1.3±0.6 | 5.2±0.6 | 11.1±1.1 | 17.0±1.3 | 23.4±1.6 |
| graph posterior alone (no training) | 2.5±0.3 | 7.0±1.9 | 9.5±1.6 | 11.6±1.9 | 14.3±1.4 |
| probcover + AUM clean-train | 5.7±1.6 | 7.1±1.1 | 10.3±1.1 | 11.9±0.7 | 14.7±0.2 |
| npc, no clean-train (selection only) | **5.9±0.9** | 8.1±0.9 | 9.8±1.2 | 10.8±0.4 | 12.5±0.4 |
| npc + AUM clean-train (Netta) | 5.7±1.6 | 7.4±0.6 | 10.3±0.7 | 11.7±1.0 | 15.0±0.8 |

---

### 4.5 The posterior itself, measured directly (offline replay — no classifier)

The posterior is a pure function of (features, labeled ids, annotation records), so it
can be rebuilt offline and scored against ground truth with no classifier in the loop.
Script: `batch_utils/eval_posterior_quality.py` (aggregation:
`aggregate_posterior_quality.py`; raw numbers:
`results/cifar100/posterior_eval/posterior_eval.json`). One trajectory per
(noise, seed) covers **all** train-mode arms — training never feeds back into
selection, so all 5 modes share identical lSets (verified). **Replay is bit-exact**,
validated three ways: regenerated annotator labels match the recorded annotations
(12/12 cells); the LOO `filtered` keep-counts match the counts the real runs printed
to their SLURM logs (**120/120** episode checks — this required replicating the
float32 accumulation order, since isolated points' LOO argmax is decided by rounding
residues); and at noise 0 with no disagreeing neighbors the posterior recovers labels
exactly. All numbers below: CIFAR100, bayes_product trajectories, mean±std over
seeds {1,2,3}.

#### A. Label denoising (labeled points): posterior argmax vs the annotations

| noise | labels | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|
| 0.0 | observed | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 0.0 | posterior argmax | 87.0±2.6 | 75.7±3.3 | 71.9±0.7 | 70.7±1.1 | 68.3±1.3 |
| 0.2 | observed | 81.0±4.4 | 78.8±4.5 | 79.9±3.0 | 80.2±1.7 | 81.0±1.0 |
| 0.2 | posterior argmax | 71.7±2.9 | 70.8±4.4 | 68.5±2.0 | 67.1±2.9 | 66.2±0.5 |
| 0.5 | observed | 48.3±4.7 | 49.9±1.1 | 49.5±1.4 | 49.1±0.5 | 49.5±1.2 |
| 0.5 | posterior argmax | 65.0±9.5 | 70.2±5.3 | 66.7±0.6 | 68.5±0.8 | 67.7±2.6 |
| 0.8 | observed | 19.7±0.6 | 20.1±2.0 | 20.4±0.9 | 20.6±1.9 | 20.6±1.3 |
| 0.8 | posterior argmax | 39.0±10.1 | 51.9±10.8 | 58.0±8.2 | 59.7±6.0 | **61.6±2.8** |

(Both rows: accuracy vs **true** labels, evaluated **on the labeled set** at each
budget — e.g. column 500 = over those 500 annotated points. *observed* = the
annotator's labels as given; *posterior argmax* = each point relabeled to
$\arg\max_k P(y_\ell = k)$ of its own posterior — i.e. exactly the labels
`map_posterior` trains on. The argmax is over the 100 classes, not over points.)

Flip breakdown at budget 1000 (mean counts /1000): noise 0.8 — 811 relabels, **462
fix a wrong label, only 52 break a correct one**; noise 0.5 — 613 relabels, 315 fix,
133 break; noise 0.2 — 435 relabels, only 111 fix, 258 break; noise 0 — 317 relabels,
all by definition breaking correct labels.

**The denoising crossover, isolated.** At noise ≥ 0.5 the posterior argmax is a far
better label than the annotation (61.6% vs 20.6% at 0.8 — **3× cleaner**; 67.7 vs
49.5 at 0.5): the §4 training-plane wins traced to their source. At noise ≤ 0.2 it is
*worse* (66.2 vs 81.0 at 0.2; 68.3 vs 100 clean). The mechanism is **δ-ball impurity**:
in-ball neighbors of *other* classes outvote the point's own annotation — on perfectly
clean data, 31.7% of labeled points sit in balls whose neighbor-vote prefers another
class, so ~32% is the measured impurity of the "same ball ⇒ same label" assumption on
CIFAR100 SimCLR features at δ=0.75. This is the root cause of takeaway 5 (correction
costs on clean data) and of NPC+AUM's remaining edge at noise 0.2. Curious tension:
at 0.2, `map_posterior` matches `observed` in test accuracy (§4) despite training on
15-points-worse labels — the wrong relabels agree with the feature geometry, so they
are *consistent* targets; raw label accuracy understates a target's usefulness.

#### B. Pseudo-label quality (unlabeled pool)

| noise | metric | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|
| 0.0 | covered (% of pool) | 25.6±0.2 | 36.4±1.0 | 41.6±0.7 | 45.6±0.4 | 49.5±0.4 |
| 0.0 | acc on covered | 71.7±0.3 | 69.0±0.6 | 68.1±0.4 | 67.4±0.3 | 66.7±0.4 |
| 0.0 | acc all pool | 18.8±0.2 | 25.4±0.5 | 28.6±0.3 | 30.9±0.3 | 33.2±0.3 |
| 0.0 | acc top-10/class | 76.4±1.2 | 70.1±2.8 | 68.4±1.3 | 68.1±0.7 | 67.9±0.3 |
| 0.2 | covered (% of pool) | 24.4±0.3 | 35.1±0.4 | 40.1±0.3 | 43.9±0.3 | 47.8±0.4 |
| 0.2 | acc on covered | 58.5±3.1 | 61.6±1.7 | 62.8±0.8 | 62.2±0.6 | 62.6±0.3 |
| 0.2 | acc all pool | 14.6±0.8 | 21.9±0.6 | 25.4±0.2 | 27.5±0.2 | 30.1±0.2 |
| 0.2 | acc top-10/class | 64.2±0.4 | 69.1±1.7 | 71.8±1.3 | 74.9±1.5 | **79.3±1.1** |
| 0.5 | covered (% of pool) | 19.9±0.3 | 30.7±0.4 | 36.1±0.6 | 39.9±0.6 | 44.0±0.2 |
| 0.5 | acc on covered | 43.9±6.3 | 52.8±1.5 | 54.1±1.0 | 54.6±1.3 | 55.0±0.6 |
| 0.5 | acc all pool | 9.2±1.4 | 16.5±0.6 | 19.8±0.3 | 22.0±0.2 | 24.4±0.2 |
| 0.5 | acc top-10/class | 35.2±5.7 | 52.2±4.3 | 59.2±2.1 | 65.7±1.3 | **69.4±0.2** |
| 0.8 | covered (% of pool) | 12.6±0.5 | 22.9±0.8 | 28.1±0.7 | 32.1±0.4 | 36.0±0.4 |
| 0.8 | acc on covered | 20.4±1.4 | 30.1±5.9 | 33.3±4.7 | 35.3±4.9 | 37.5±3.2 |
| 0.8 | acc all pool | 3.1±0.3 | 7.3±1.5 | 9.7±1.1 | 11.6±1.5 | 13.7±1.2 |
| 0.8 | acc top-10/class | 11.1±2.4 | 23.0±7.0 | 29.4±2.1 | 35.5±5.0 | **42.8±4.2** |

Row definitions — all on the unlabeled pool (uSet): **covered** = fraction of pool
points with ≥1 labeled point inside their δ-ball (only these carry information — an
uncovered point's posterior is exactly the uniform prior); **acc on covered** =
posterior-argmax accuracy vs true labels over covered points only; **acc all pool** =
the same over *all* pool points including uncovered ones (uncovered argmax ≈ 0.3–0.6%,
sub-chance tie-breaking, so this row ≈ covered × acc-on-covered — it measures
knowledge-times-reach); **acc top-10/class** = accuracy of the 10 most-confident
covered points per *predicted* class (≈1000 points at budget 1000) — what a
pseudo-labeling arm would harvest. Adding the labeled set to "all pool" changes the
numbers by <1pt (e.g. 34.0 vs 33.2 clean @1000; the pool is 45–49× larger than lSet).
The all-pool row also frames the classifier's role sharply: the posterior alone scores
33.2% on the pool (clean, @1000) vs the trained classifier's 46.5% test accuracy — the
network's contribution is precisely *generalizing past the δ-balls* to the uncovered
50–64%; conversely on covered points at noise 0.8 the posterior (37.5%) still beats
the best classifier (27.5%), so under heavy noise the classifier extracts only part
of what the graph already knows.

**Pseudo-labeling (7.5) is justified before running a single training job.** At
budget 1000, the top-10-per-class pseudo-labels are *cleaner than the annotations we
paid for* wherever noise ≥ 0.2: 79.3% vs 81.0 observed at noise 0.2 (a free doubling
of the labeled set at equal quality), **69.4 vs 49.5 at 0.5, 42.8 vs 20.6 at 0.8**.
On clean data they are worse (67.9 vs 100 — ball impurity again), so the arm needs
the same noise gate as the other correction modes. Also note the headroom signal:
at noise 0.8 the posterior alone gets 37.5% on covered pool points while the best
trained classifier reaches 27.5% — the classifier has not yet extracted everything
the posterior knows.

#### C. Calibration (covered unlabeled points) — the §1.4 prediction, confirmed

| noise | metric | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|
| 0.0 | ECE | 0.251±0.003 | 0.264±0.004 | 0.274±0.002 | 0.281±0.001 | 0.286±0.003 |
| 0.0 | mean conf / acc (%) | 96.7 / 71.7 | 95.4 / 69.0 | 95.5 / 68.1 | 95.5 / 67.4 | 95.4 / 66.7 |
| 0.2 | ECE | 0.230±0.032 | 0.228±0.015 | 0.226±0.009 | 0.236±0.006 | 0.239±0.005 |
| 0.2 | mean conf / acc (%) | 81.5 / 58.5 | 84.4 / 61.6 | 85.3 / 62.8 | 85.8 / 62.2 | 86.4 / 62.6 |
| 0.5 | ECE | 0.189±0.062 | 0.149±0.008 | 0.155±0.012 | 0.164±0.013 | 0.168±0.006 |
| 0.5 | mean conf / acc (%) | 62.0 / 43.9 | 67.6 / 52.8 | 69.6 / 54.1 | 70.9 / 54.6 | 71.8 / 55.0 |
| 0.8 | ECE | 0.152±0.018 | 0.110±0.046 | 0.107±0.042 | 0.103±0.041 | 0.102±0.027 |
| 0.8 | mean conf / acc (%) | 33.8 / 20.4 | 39.4 / 30.1 | 43.5 / 33.3 | 45.2 / 35.3 | 47.4 / 37.5 |

The untempered product is **systematically overconfident at every noise level**
(confidence exceeds accuracy everywhere; ECE 0.10–0.29, worst on clean data where
agreeing correlated neighbors multiply hardest) — exactly the §1.4 correlated-neighbor
prediction. This is direct evidence for the β-temper sweep (7.4), and it opens a
shortcut: **β can be tuned offline** by minimizing ECE in this replay, without any
training runs.

#### D. The LOO reliability signal: good score, broken threshold

| noise | metric | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|
| 0.2 | ROC-AUC (detect noisy) | 0.757±0.010 | 0.857±0.022 | 0.864±0.008 | 0.863±0.012 | 0.865±0.013 |
| 0.2 | flag precision (%) | 31.5±10.3 | 37.3±4.1 | 35.7±4.1 | 35.0±2.1 | 34.3±1.7 |
| 0.5 | ROC-AUC | 0.860±0.013 | 0.912±0.026 | 0.899±0.015 | 0.908±0.002 | 0.902±0.010 |
| 0.5 | flag precision (%) | 57.6±4.1 | 62.2±2.0 | 65.2±4.6 | 67.4±1.5 | 68.1±1.2 |
| 0.8 | ROC-AUC | 0.821±0.037 | 0.871±0.044 | 0.887±0.047 | 0.907±0.036 | 0.915±0.022 |
| 0.8 | flag precision (%) | 83.0±1.5 | 83.6±2.1 | 85.4±1.6 | 86.5±1.3 | 87.4±0.7 |
| — | LOO-disagree frac @1000 (%) | noise 0: **52.9** | noise .2: **52.6** | noise .5: 73.9 | noise .8: 90.2 | |

Flag recall is ~99% at noise ≥ 0.5 (nearly every truly-noisy label gets flagged).
Three sharp conclusions:

1. **The score is good, the threshold is broken.** As a *ranking*, LOO reliability
   detects noisy labels well (AUC 0.86–0.92 from 300 labels on). But the `filtered`
   keep-rule ("LOO argmax == observed") flags **53% of perfectly clean labels** at
   noise 0 — ball impurity makes neighbor-argmax a hopeless bar. `filtered`/
   `reweighted` fail because of the *threshold*, not the signal (sharpens 7.6: use a
   percentile / calibrated cut on the continuous $w_\ell$, not argmax agreement).
2. **The naive 7.1 gate signal saturates.** The LOO-disagreement fraction cannot tell
   clean from 20%-noise (52.9 vs 52.6) — the ~53% impurity floor swamps it. The
   noise-gate needs the continuous $w_\ell$ distribution or an impurity-corrected
   baseline, not the raw disagree fraction.
3. **Budget-100 LOO verdicts are numerically arbitrary.** Most labeled points have no
   other labeled neighbor in-ball at 100 labels; their LOO posterior is uniform and
   the keep/drop decision comes down to float32 rounding residues (reproducing the
   runs' exact counts required replicating summation order). The `filtered`@100
   collapse in §4 is exactly this.

---

### 4.6 §7.5 curriculum pseudo-labeling — RESULTS (sweep 31047129, 36/36 ✔)

`TRAIN_LABEL_MODE='pseudo'`: labeled points train on their soft posterior (exactly
`soft_posterior`), **plus** confident *unlabeled* pool points enter the training set
with their posterior as a soft target. Guard (a) is implemented (prior-only points —
flat accumulator row — never pseudo-labeled); guard (b) (per-ball caps) is **not**.
Three arms, selected by `--pseudo_scope` / `--pseudo_k`:

- **all** — every covered pool point (~8.8k pts at 100 labels, noise .5; ~30k at 1000).
- **lin10** — top-k most-confident per *predicted* class, k = training round (1→10).
- **agg50** — same, k grows 1,2,3,5,8,13,20,30,40,50.

Same regime and seeds {1,2,3} as the train-mode sweep, so rows are paired with the
baselines (the `observed` rows below bit-match `bayes_product` in the §4 tables —
deterministic pipeline). Unlike the §4 tables (5 canonical budgets), these show
**all 10 episodes** (labeled size 100…1000, one column per episode); the baseline
rows are re-aggregated at the same resolution. Figure + CSV (also all episodes):
`results/bayes_pseudo/`.

**Noise = 0.0** (all 10 episodes)

| Method | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| observed | 24.6±1.3 | 33.7±0.1 | 38.1±1.3 | 40.2±0.6 | 41.6±0.0 | 42.7±0.4 | 43.6±0.2 | 44.4±0.3 | 44.8±0.4 | 45.1±0.1 |
| map_posterior | 22.5±0.5 | 31.5±0.4 | 35.9±0.4 | 37.9±0.3 | 38.9±0.2 | 40.4±0.3 | 40.8±0.4 | 41.3±0.5 | 41.4±0.6 | 41.8±0.5 |
| soft_posterior | 23.7±1.0 | 32.0±0.8 | 36.3±0.7 | 38.0±0.2 | 39.2±0.0 | 40.9±0.3 | 41.1±0.5 | 41.8±0.6 | 42.6±0.6 | 42.5±0.1 |
| pseudo lin10 | 26.3±1.0 | 37.2±0.4 | 40.6±0.8 | 42.2±0.5 | 43.4±0.0 | 44.6±0.5 | 45.2±0.9 | 45.8±0.7 | 46.0±0.6 | 46.3±0.5 |
| pseudo agg50 | 26.3±1.0 | 37.2±0.4 | 40.6±0.8 | 42.5±0.2 | 43.9±0.2 | 45.1±0.4 | 45.8±0.5 | 46.5±0.6 | 47.2±0.4 | **47.8±0.4** |
| pseudo all | **37.6±0.4** | **40.5±0.4** | **42.6±0.7** | **44.0±0.5** | **44.9±0.1** | **45.8±0.1** | **46.5±0.1** | **46.8±0.2** | **47.3±0.4** | 47.5±0.2 |

**Noise = 0.2** (all 10 episodes)

| Method | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| observed | 21.6±1.5 | 30.1±1.7 | 32.8±1.5 | 35.1±1.0 | 36.6±0.5 | 37.6±1.0 | 37.7±0.7 | 38.2±0.4 | 38.6±0.3 | 39.2±0.3 |
| map_posterior | 20.6±2.3 | 29.6±0.5 | 33.7±2.0 | 35.3±1.5 | 36.9±1.1 | 38.2±0.5 | 39.2±0.8 | 39.8±0.1 | 40.1±0.2 | 40.6±0.6 |
| soft_posterior | 19.1±1.7 | 29.5±0.9 | 33.4±1.9 | 35.6±2.1 | 37.5±1.0 | 39.3±0.4 | 40.0±0.5 | 40.7±0.4 | 41.2±0.3 | 41.5±0.4 |
| pseudo lin10 | 19.7±1.6 | 33.2±1.1 | 37.6±1.6 | 39.7±1.2 | 40.8±0.8 | 41.5±0.3 | 42.0±0.5 | 42.7±0.8 | 43.4±0.8 | 43.8±1.2 |
| pseudo agg50 | 19.7±1.6 | 33.2±1.1 | 37.6±1.6 | **40.1±1.0** | 40.8±0.8 | 41.1±0.4 | 41.4±0.8 | 42.3±1.0 | 43.5±0.9 | 44.1±0.6 |
| pseudo all | **28.6±1.7** | **35.0±2.0** | **37.7±1.7** | 39.9±1.0 | **41.7±0.6** | **43.0±0.4** | **43.6±0.2** | **44.5±0.1** | **45.3±0.1** | **45.8±0.6** |

**Noise = 0.5** (all 10 episodes)

| Method | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| observed | 14.9±1.4 | 20.4±0.7 | 23.1±0.1 | 25.6±0.5 | 26.4±0.6 | 27.3±1.0 | 28.1±0.6 | 29.3±0.1 | 29.6±0.3 | 29.5±0.4 |
| map_posterior | 15.2±1.9 | 23.4±1.6 | 27.9±1.7 | 30.5±1.3 | 31.9±0.2 | 33.1±0.4 | 34.5±1.0 | 35.5±0.9 | 37.0±0.7 | 37.6±0.4 |
| soft_posterior | 12.3±0.9 | 23.0±1.3 | 27.8±1.0 | 31.1±0.9 | 33.1±0.5 | 34.6±0.7 | 36.2±0.1 | 37.8±0.5 | 38.6±0.6 | 39.1±0.8 |
| pseudo lin10 | 12.5±0.9 | 24.7±1.4 | 30.1±0.8 | 34.0±0.7 | 36.0±0.6 | 37.6±1.0 | 38.6±0.4 | 40.0±1.0 | 40.5±0.9 | 40.9±1.0 |
| pseudo agg50 | 12.5±0.9 | 24.7±1.4 | 30.1±0.8 | **34.3±0.6** | **36.5±1.0** | **37.6±1.0** | 38.8±0.9 | 40.0±0.9 | 40.5±0.6 | 41.3±0.7 |
| pseudo all | **18.0±2.9** | **25.0±1.2** | **31.1±1.1** | 33.9±1.0 | 36.1±0.5 | 37.5±0.8 | **39.1±0.6** | **40.5±0.7** | **41.2±0.6** | **41.8±0.7** |

**Noise = 0.8** (all 10 episodes)

| Method | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| observed | 4.6±0.5 | 7.5±0.2 | 9.7±2.0 | 10.9±1.3 | 10.9±1.5 | 11.3±1.4 | 11.8±1.5 | 12.0±2.0 | 12.3±1.9 | 12.6±1.6 |
| map_posterior | 5.8±0.2 | 10.1±1.9 | 14.2±2.9 | 17.4±1.3 | 19.4±2.1 | 21.9±2.2 | 22.9±2.2 | 25.1±2.2 | 24.7±2.4 | 26.2±1.3 |
| soft_posterior | 5.3±0.5 | 9.5±1.2 | 13.4±2.0 | 16.6±1.3 | 18.9±2.4 | 21.3±2.1 | 23.4±2.5 | 25.4±2.4 | 25.7±2.7 | 27.5±1.9 |
| pseudo lin10 | 5.2±0.6 | 10.1±1.1 | 14.3±2.5 | 17.8±1.3 | 20.0±2.1 | **22.2±2.0** | 24.4±2.7 | 26.0±2.7 | 26.1±2.8 | 28.0±2.6 |
| pseudo agg50 | 5.2±0.6 | 10.1±1.1 | 14.3±2.5 | **17.9±1.2** | **20.1±1.9** | 22.1±2.2 | **24.6±2.9** | **26.3±2.9** | **27.0±2.8** | **29.0±2.7** |
| pseudo all | **6.2±0.7** | **10.8±2.1** | **14.4±3.2** | 17.5±1.4 | 19.7±2.5 | 22.2±2.7 | 24.3±3.4 | 25.8±3.1 | 26.4±3.0 | 28.1±2.5 |

Reading: **pseudo wins at every noise level, including 0** — see takeaway 8.
`lin10` ≡ `agg50` through 300 labels by construction (the schedules coincide at
k = 1,2,3; identical numbers = the determinism trust-check again). The §4.5 B
prediction (pseudo-labels cleaner than paid annotations wherever noise ≥ .2, worse
when clean) is confirmed *directionally* by training, with one surprise: the arm
helps on **clean** data too (47.8 vs observed 45.1 @1000; 37.6 vs 24.6 @100 for
`all`) — §4.5 B under-sold it because even 67.9%-clean extra targets are soft
(the wrongness is spread over classes, not committed to a hard wrong label) and
the sheer volume (~9–30k points) dominates the classifier's data diet at these
budgets.

## 5. Takeaways

1. **Selection plane alone ≈ ProbCover.** `bayes_product` wins small at the very-low
   end (1/class: 21.6 vs 19.7 @noise .2; 14.9 vs 13.1 @.5), converges or trails
   slightly by 1000 labels. Annotator-accuracy weighting matters most when labels are
   scarcest, but selection alone is not the win.
2. **The training plane is the win — and it beats Netta's AUM at noise ≥ 0.5.**
   Noise 0.8 @1000: `soft_posterior` **27.5** vs observed 12.6 ($2.2\times$) vs NPC+AUM
   **15.0**. Noise 0.5 @1000: **39.1** vs 29.5 vs 35.6. Soft graph-posterior label
   correction dominates hard AUM filtering exactly where noise is heaviest —
   plausibly because AUM's judge network was itself trained on 50–80% wrong labels,
   while the graph posterior needs no trained model.
3. **At noise 0.2, NPC+AUM still slightly ahead** (43.3 vs 41.5 @1000). Few flips ⇒
   AUM is reliable, and hard-dropping a few bad points beats softening every label.
4. **Product ≫ additive at every cell** — design-§3 gate passes decisively; additive
   collapses on CIFAR100 (26.9 vs 45.1 clean @1000). ⚠ Caveat: at noise 0 on CIFAR100
   the additive arm also fails the "reproduces Itai ≈ probcover" sanity check
   (13.4 vs 23.6 @100; it *passed* on CIFAR10) — so on CIFAR100 our additive branch is
   not a faithful Itai control (likely the Dirichlet($\pi$) init vs. his scalar-$\alpha$; open item).
5. **Clean-data sanity holds:** at noise 0, `observed` is best; every correction mode
   costs a little (soft 42.5 vs 45.1). Correcting labels that aren't wrong removes
   signal ⇒ the mode should be **noise-gated**: observed when clean, soft/map when noisy.
6. **`filtered`/`reweighted` collapse at budget 100** (~5% even at noise 0): with
   ~1 label/class the LOO posterior has almost no neighbors to judge with, so it
   drops/down-weights good points. Respectable only from ~500 on ⇒ needs a
   minimum-neighbor gate if kept.
7. **Trust checks:** NPC ≡ PC at episode 1 in Netta's sweep (filtering starts after the
   first training round); AUM-ablation `clean=off` arms reproduce the no-filter
   baselines bit-for-bit (deterministic pipeline).
8. **Pseudo-labeling (§4.6) is the new best training arm at *every* noise level —
   and it removes the noise gate's necessity.** It beats the previous per-noise
   winners at 1000 labels: 47.8 vs 45.1 (observed) clean, 45.8 vs 41.5 (soft) at .2,
   41.8 vs 39.1 at .5, 29.0 vs 27.5 at .8 — and, unlike soft/map, it also beats
   NPC+AUM at noise 0.2 (45.8 vs 43.3), so the method now leads Netta's stack
   everywhere. Within the arm: `all` dominates at low budget (37.6 vs 26.3 @100
   clean — coverage volume matters most when labels are scarcest) and stays best
   through noise .5; the curriculum arms catch up by 1000 and `agg50` is the safer
   choice at noise .8 (29.0 vs 28.1), consistent with guard (b) being unimplemented —
   `all` amplifies noisy balls hardest exactly where labels are least reliable.
   Takeaway 5's clean-data penalty does *not* apply: pseudo helps clean data too,
   so `pseudo` needs no noise gate (only the scope choice might be gated).

## 6. Reproducing the tables

Per-run data: `episodes[].test_acc` keyed by `labeled_size` in each
`output/CIFAR100/resnet18/<date>/<run>/result.json`, filtered by `sweep_id` ∈
{30978366, 30980646, 30811243, 30863746, 31047129 (§4.6 pseudo)}; group by
`params.al_method` / `params.train_label_mode` (+ `params.pseudo_scope` /
`params.pseudo_k` for the pseudo arms) / noise / seed. Aggregation script (session scratchpad,
copy into `batch_utils/` if wanted long-term): `build_tables.py` — or use the
dashboard Comparison tab (Category = `bayesian_coverage` + `aum_filter`, shared
`noise_rate` filter). The AUM sweep extends to 3000 labels if a longer x-axis is needed.

Note: the CIFAR10 twin sweep (job 30969432, 44/48) exists but is the wrong regime for
coverage methods (10–100 labels/class); read CIFAR100 only.

---

## 7. Next steps

Ordered roughly by value; items 1–3 follow directly from the §5 takeaways.

### 7.1 Noise-gated training policy

The sweep showed a clean trade-off: under noise, posterior-based training wins big
(+15 pts at noise 0.8 @1000), but on clean data it *costs* ~2–3 pts — correcting labels
that aren't wrong throws away signal (takeaway 5). So the right production policy isn't
"always `soft_posterior`," it's a switch. The open question is how the method *knows*
it's in a noisy regime without being told. Candidate gate: for each labeled point we
already compute the LOO posterior (its neighbors' verdict without its own vote, §2);
the fraction of points where that verdict disagrees with the annotation is essentially
a **free noise-rate estimate** — no trained model needed. Clean data → few
disagreements → stay on `observed`; noisy → many → switch to soft/map.
**⚠ Measured (§4.5 D):** the *raw* disagree fraction saturates at a ~53% ball-impurity
floor and cannot distinguish noise 0 from 0.2 (52.9 vs 52.6) — the gate must use the
continuous $w_\ell$ distribution or an impurity-corrected baseline instead. Work items:
design that corrected statistic (calibrate against §4.5's replay across 0/.2/.5/.8),
then wire as `TRAIN_LABEL_MODE=auto`.

### 7.2 Writeup plot + more seeds

The headline claim (soft_posterior 27.5 vs observed 12.6 at noise 0.8) currently rests
on 3 seeds with ±2–3 pt stds at mid-budgets — fine for a direction, thin for a figure
someone will scrutinize. Concretely: a **gain plot** (soft_posterior − observed, per
budget per noise) with error bars, plus ~3 extra seeds at the cells where the std is
widest (noise 0.8, 300–500 labels). Cheap: a re-run of existing configs with new seeds,
~15 GPU-runs.

### 7.3 Head-to-head clean-signal comparison (judge vs judge)

The Bayes-vs-Netta comparison currently confounds two differences at once: selection
rule (bayes_product vs npc) *and* clean-signal (graph posterior vs AUM). The §4
decomposition note showed selection barely matters for either method — so the
interesting fight is **judge-vs-judge**, and the current tables never stage it directly.
The 2×2 to fill in: {bayes_product, npc} × {posterior-based training, AUM clean-train}.
The two missing cells: bayes_product + AUM clean-train (mostly config wiring —
`NOISE_FILTER=aum` on a bayes run) and npc + soft_posterior (needs the posterior
computed on an npc run; slightly more plumbing). If graph-judge wins even on npc
selection, the story is clean: *the posterior is a better label-quality judge than AUM,
independent of selection*.

### 7.4 $\beta$-temper sweep

All results so far use the raw untempered product ($\beta = 1$), which
naive-Bayes-overcounts correlated neighbors (§1.4): five clustered neighbors saying
"class A at 0.7" multiply into ~0.99 certainty when the honest belief is ~0.7. This
false sharpness feeds directly into `soft_posterior` training targets (over-peaked =
closer to a possibly-wrong one-hot) and into selection (overconfident regions stop
being sampled). **Now confirmed empirically (§4.5 C):** the posterior is overconfident
at every noise level (ECE 0.10–0.29, confidence > accuracy everywhere, worst on clean
data). A $\beta \in \{0.25, 0.5, 1.0\}$ sweep on bayes_product + soft_posterior at
noise .5/.8 is ~36 runs — and §4.5's offline replay gives a shortcut: **pre-tune
$\beta$ by minimizing ECE offline**, then confirm the winner with a small training
sweep. Expectation: $\beta < 1$ helps most exactly where the headline result lives.

### 7.5 Semi-supervised training arm (pseudo-labeling) — ✅ DONE (2026-07-09)

Implemented as `TRAIN_LABEL_MODE='pseudo'` (+ `PSEUDO_SCOPE`/`PSEUDO_K_SCHEDULE`,
CLI `--pseudo_scope`/`--pseudo_k`; launcher `batch_utils/sweep_bayes_pseudo.py`) and
swept as job 31047129 — **results in §4.6, takeaway 8: best training arm at every
noise level, no noise gate needed.** Guard (a) (exclude prior-only points) is
implemented — a prior-only point has a flat accumulator row and is never harvested.
Guard (b) (**cap correlated evidence per ball**) is **still open**: one noisy
annotation can be amplified into dozens of agreeing pseudo-targets (the §1.4
double-counting problem moved to the training plane). The results show its
fingerprint — `all` trails the slow-curriculum `agg50` only at noise .8 / high
budget (28.1 vs 29.0) — so a per-ball cap is the natural next refinement if the
high-noise cells matter for the writeup. Remaining follow-ups: per-ball caps, a
confidence floor for `all` (harvest only above a posterior-max threshold), and
scope gating (all→topk as noise grows).

### 7.6 Fix `filtered`/`reweighted` at low budget

Both modes collapse to ~5% at budget 100 *even on clean data* (takeaway 6): with
~1 label/class, a point's LOO posterior is built from almost no neighbors, so the
"judge" fires on noise and drops/down-weights good points — §4.5 D showed the budget-100
verdicts are literally decided by float rounding (no in-ball neighbors → uniform LOO).
§4.5 D also relocated the real defect: the LOO *score* ranks noisy labels well
(AUC 0.86–0.92), but the "argmax == observed" *threshold* flags 53% of clean labels.
Fix accordingly: (a) minimum-neighbor gate (fall back to `observed` below N in-ball
neighbors), and (b) replace the argmax rule with a percentile / calibrated cut on the
continuous $w_\ell$. Worth doing only if these modes stay in the story; if soft/map
remain the winners, this becomes documentation rather than code.

### 7.7 Additive-arm fidelity (Itai control)

On CIFAR100 our additive branch scores 13.4 @100 clean vs probcover's 23.6 — it fails
the "reproduces Itai ≈ probcover" sanity check that it *passed* on CIFAR10 (takeaway 4).
So the CIFAR100 "product ≫ additive" margin, while surely real in direction, is measured
against a possibly-unfaithful Itai control. Suspected cause: our Dirichlet($\pi$)
row-init vs his scalar-$\alpha$ pseudocount (different effective prior strength as
counts accumulate). Fix: expose $\alpha$ as a config knob, match his value, re-run the
additive arm only (12 runs). Matters mainly if the writeup cites the additive
comparison as an Itai reproduction.

### 7.8 Remaining `NotImplementedError` stubs

In rough order of interest:

- **`entropy` acquisition** (BALD-style): select by posterior entropy in the ball
  rather than $1-\max$ — could differ where the posterior is multi-modal.
- **`NoiseAwareBayesianCoverage`** (design §4): per-point soft reliability $r_\ell$ —
  the theoretical umbrella under which NPC is the hard $r_\ell \in \{0,1\}$ limit;
  nice for the paper's framing.
- **Empirical prior**: $\pi$ from observed label frequencies instead of uniform — only
  matters off CIFAR's balanced classes.
- **Sparse-K**: memory scaling for bigger pools — pure engineering; blocks
  ImageNet-scale runs, nothing else.

---

*Rendered versions: `bayesian_coverage_SUMMARY.html` (browser, KaTeX/MathML) and
`bayesian_coverage_SUMMARY.pdf` — regenerate with pandoc after editing this file.*
