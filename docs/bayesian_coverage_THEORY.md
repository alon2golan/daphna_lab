---
title: "Bayesian Coverage — Theory: probabilistic model, training modes, NPC/AUM baseline"
date: 2026-07-12
---

**Part 1/3 of the bayesian-coverage summary** (split 2026-07-12 from the former single
`bayesian_coverage_SUMMARY.md`): **theory (this file)** ·
results: `bayesian_coverage_RESULTS.md` · status, open questions & next steps:
`bayesian_coverage_NEXT_STEPS.md`. Section numbering is preserved from the original
file — §1–§3 live here, §4–§6 in RESULTS, §7+ in NEXT_STEPS; cross-references keep
the original § numbers, so e.g. "§4.5" means RESULTS §4.5 and "7.4" means NEXT_STEPS §7.4.

**Branch:** `bayesian_coverage`
**Companion docs:** design spec in `docs/bayesian_coverage_DESIGN.md`; implementation in
`deep_al/pycls/al/bayesian_coverage.py`; training wiring in `deep_al/tools/train_al.py`
(`_bayes_training_loader`) + `deep_al/pycls/noise/noisy_label_wrapper.py`; classifier-input
normalization study in `docs/bayesian_coverage_NORMALIZATION.md`.
Everything in this file is verified against the code.

## The idea in two sentences

We work with **noisy annotators** and act on **two planes**:

1. **Point selection (AL):** replace ProbCover's binary covered/not-covered with a
   **per-class posterior distribution** per point, built from neighbor annotations,
   each weighted by its **annotator's confusion matrix** and its **kernel distance**,
   fused as a **product of experts**. Select the point whose ball contains the most
   remaining *uncertainty mass*.
2. **Training:** reuse the same posterior to build the training targets — six
   configurable policies (`TRAIN_LABEL_MODE`), from raw noisy labels to full soft
   labels to curriculum pseudo-labeling of unlabeled points (`pseudo`, §4.6).

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
  not an assumption). Result: product wins decisively (RESULTS, takeaway 4).

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

