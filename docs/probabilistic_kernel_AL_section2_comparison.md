# Probabilistic Kernel AL — Section 2, explained + compared

Source: `daphna_lab/papers/probabilistic kernel AL.pdf` (Daphna Weinshall), Section 2:
"A Bayesian framework unifying coverage and uncertainty" (pages 3–5).
Compared against our `deep_al/pycls/al/bayesian_coverage.py` and Itai's
`deep_al/pycls/al/BAYES_MISP.py` (vendored port, verified faithful — 45.0 vs his 45.2 @1000).

---

## 1. What Section 2 says

Section 1's coverage framework tracks a scalar "veracity" per point, `v_j = max_{l∈L} K_lj` —
how well point j is covered by its nearest labeled neighbor. Section 2 replaces that scalar
with a **full class posterior per point**, so the framework can represent label *conflicts*
(two nearby labeled points disagreeing) instead of just "covered or not." That's what's
supposed to make it work beyond the low-budget regime.

**The generative model (§2.1).** Each point's label is categorical, `y_j ~ Cat(θ^j)`, with a
symmetric Dirichlet prior `θ^j ~ Dir(α,…,α)`. By conjugacy, if you observed labels *at point j
itself*, the posterior mean would be `(α + n_m)/(Mα + N)` — prior pseudo-count α plus the
count of each label m.

The twist is that we never observe labels at j; we observe them at neighbors, and a neighbor's
label is only evidence about j to the extent the kernel says they're similar. The paper handles
this with a **powered (tempered) likelihood**: each labeled neighbor l contributes not a full
count of 1 to class `y_l`, but a **discounted soft count** `K_lj` (eq. 4):

    ñ_m^j = Σ_{l∈L} K_lj · 1{y_l = m}

So the whole state of knowledge is a **veracity matrix** `V_mj = α + Σ_l K_lj 1{y_l=m}`
(eq. 5), and the posterior of point j is just its normalized column (eq. 6). Note this is an
*additive* accumulation of kernel-weighted votes on top of a flat prior α.

**Acquisition (§2.2–2.3).** Define a **peakedness score** f(posterior) that measures distance
from one-hot — candidates: `1 − max_k`, entropy, or margin. The ideal query set maximizes the
total peakedness *improvement* over all points (eq. 7), clamped at zero per point. Two
problems, both acknowledged: (i) the objective is **not submodular** (conflicts break
diminishing returns), so greedy has no guarantee; (ii) the greedy step (eq. 8) is **not
computable**, because you'd need to know the label of the candidate before querying it.
Algorithm 1 therefore runs greedy with an *estimated* gain Δ(i), and §2.4 says explicitly that
**Itai's project is finding a surrogate for eq. 8 using the columns of V as a stand-in for the
unknown label distribution**, plus empirically tuning the peakedness score and the prior α
(with Dvir on principled α selection — small α makes the model decisive but fragile to
early/wrong labels).

**Future work (§2.5)** includes injecting *model* predictions as pseudo-labels into V with
confidence weights (eqs. 9–10), and handling incorrect labels — i.e., §2 as written assumes
**all observed labels are correct**; noise is deferred to Section 3 (non-Bayesian) and a fully
integrated model is only Section 5's sketch.

---

## 2. Compared to our `BayesianCoverage`

The headline: **the paper's model is exactly our `additive` branch; our default `product`
branch is a deliberate departure from the paper.**

- **Evidence fusion.** The paper accumulates kernel-weighted votes additively (Dirichlet
  pseudo-counts, eq. 5) — that is literally our `COMBINE='additive'` update
  `C[:, label] += K_row` with posterior `C/ΣC` (bayesian_coverage.py:277-278). Our default is
  `COMBINE='product'`: a product of experts in log space, where each neighbor contributes
  `q_l(u,k) = K·b_l(k) + (1−K)·π(k)` and posteriors multiply. These are genuinely different
  models — additive evidence saturates slowly (confidence n/(n+1) after n votes), product
  saturates after one confident in-ball label. Our sweeps showed product ≫ additive at high
  budget, which is empirical support for departing from the paper's eq. 4.

- **Label noise.** The paper's §2 has no noise model — observed labels enter V raw. Our
  implementation bakes in the annotator confusion matrix at Step A:
  `b_l(k) ∝ π(k)·M[k, ỹ_l]` (bayesian_coverage.py:182-190), so a noisy annotator's vote is
  flattened before propagating. In effect, **our implementation is the paper's §2 + §3 fused
  into one Bayesian model — which the paper itself lists as the not-yet-written Section 5.**
  That's the main thing we have that the document doesn't.

- **Acquisition.** The paper wants expected peakedness reduction (eq. 8) and admits it needs a
  surrogate. Our `coverage_gain` (`score = K @ deficit`) is one specific, cheap surrogate using
  `f = 1 − max_k`: it implicitly assumes a queried label would fully retire its in-ball
  neighbors' deficits. The paper's other peakedness choices (entropy, margin) are our
  `SELECTION='entropy'` NotImplementedError. The `f1` ceiling we added for additive mode
  (bayesian_coverage.py:251-253, the fix for the clumping bug) is exactly the "what posterior
  does one label produce" correction — the same quantity BAYES_MISP uses.

- **The prior α.** The paper treats α as a key open hyperparameter. We pin it:
  `π = uniform 1/C`, and in additive mode the row initializes to π (α = 1/C, not exposed).
  This is a real gap versus the paper's agenda — Dvir's whole sub-project is that knob.

- **Pseudo-labels (eq. 9).** The paper's model-uncertainty extension folds model predictions
  back into V **for selection**. Our `TRAIN_LABEL_MODE='pseudo'` + NW out-of-ball do something
  related but pointedly different: we use the posterior (and NW targets) as **training targets
  only** — selection never sees model output. Also our extras (tempering `TEMPER_BETA`,
  kernel-vs-equal neighbor weighting, the whole TRAIN_LABEL_MODE family, cross-episode
  re-folding) have no counterpart in §2.

---

## 3. Compared to Itai's `BAYES_MISP`

BAYES_MISP **is** the paper's Section 2, implemented — Itai is the person named in §2.4, and
the code is his empirical exploration of exactly the open questions the paper lists:

- **The model is eq. 5/6 verbatim.** `C_general` initializes to a constant α (the Dirichlet
  prior, BAYES_MISP.py:107), updates additively with the kernel row of each selected point
  (`C[:, label] += K_row`, line 515), and posteriors are normalized rows `C/ΣC` — the veracity
  matrix and its normalized columns.

- **The selection is the eq.-8 surrogate.** With `diff_method='max'`, the score is
  `Σ_j max{ (K_ij+α)/(K_ij+αM) − maxP_j, 0 }` (batched_diffs_sparse_max): "if I label point i,
  each neighbor j's posterior max would jump to what one K-weighted vote on a fresh prior
  gives; count the improvement, clamp at 0." That's Algorithm 1's Δ(i) with `f = 1 − max_k`,
  made computable by an optimistic assumption about where the label lands. The general
  weighted path (`batched_diffs_efficient_weighted*`) goes further and does what §2.4
  describes word for word: it computes the gain **for every possible class of the candidate,
  weighted by the candidate's current normalized posterior** (`norm_C` as the surrogate label
  distribution).

- **The knob zoo = the paper's open questions.** `calc_method ∈ {max, entropy, margin}` are
  the three peakedness scores of §2.2. `ALPHA`, `alpha_init_mode` (constant / from_sparsity /
  from_vector), `alpha_decay_gamma`, the `switch_alpha_*` schedules, and `LOCAL_ALPHA`
  per-point priors are all the α-selection question (§2.4, Dvir's item). `cont_method`
  positive/abs is the max{·,0} clamp choice. Sparse K + threshold updates are Itai's sparsity
  contribution from §1.1.

- **NN fusion is eq. 9 implemented.** `_apply_nn_fusion` adds scaled, per-class-accuracy-
  weighted model softmax predictions into a `C_fused` used for selection — that's the paper's
  "future work" veracity update `V += β Σ_u (1−γ_u) K_uj 1{ỹ_u=m}`, with per-class accuracy
  playing the (1−γ) confidence role, already coded.

- **What it doesn't have:** any noise/annotator model — labels are trusted, and a selected
  point's own row is hard-reset to one-hot (line 513-514). Selection-only; training uses
  observed labels.

---

## Three-way summary

The paper's §2 is the spec; BAYES_MISP is a faithful, heavily-instrumented implementation of
that spec (additive Dirichlet + surrogate-greedy + all the open-question knobs); our
BayesianCoverage contains that same model as its `additive` branch (with the acquisition made
exactly equivalent for tophat via the f1 ceiling), but defaults to a different fusion rule
(product of experts) that the paper doesn't contain, adds the annotator-noise correction the
paper defers to Section 5, and spends the posterior on training targets rather than feeding
model predictions back into selection.

**Worth flagging for the group:** our product-vs-additive sweep result (product ≫ additive at
high budget, additive ≈ probcover at low budget after the f1 fix) is direct empirical evidence
on the paper's central §2 design choice — the powered-likelihood *additive* posterior is
exactly the branch that loses at high budget, which is the regime §2 was written to fix.
