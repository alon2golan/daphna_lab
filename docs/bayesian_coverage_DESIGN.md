# Bayesian Coverage for Noisy-Annotator Active Learning — Design

**Status:** design / not yet implemented
**Author:** Alon (HUJI, Daphna lab)
**Date:** 2026-07-01
**Scope:** a probabilistic ("Bayesian") generalization of `CoverageMatrixMethod` /
`NoiseAwareCoverage` (`deep_al/pycls/al/coverage_matrix_methods.py`) for the
noisy-annotator setting.

---

## 0. TL;DR

Coverage in a noisy environment should not be binary. This doc specifies a
model where each point carries a **per-class posterior distribution** over its
label, in which:

1. coverage is soft / per-class (Itai already does this);
2. the strength of coverage depends on the **annotator's accuracy** (new);
3. **all** labeled neighbors of a point are fused Bayesianly, not just the
   nearest (fix the combination rule);
4. (future work) the per-point label reliability is *updated* from a prior
   (= annotator accuracy) to a posterior using AUM once enough points are
   labeled.

Delivered as a **new class `BayesianCoverage(CoverageMatrixMethod)`**, not by
modifying `CoverageMatrixMethod`, because the semantics of the coverage matrix
`C`, its update rule, and the selection score all change.

---

## 1. Baseline: what Itai's `BAYES_MISP` actually does

Reference: <https://github.com/itaidavid1/bayes_al_2025> (fork of TypiClust),
file `deep-al/pycls/al/BAYES_MISP.py`.

Despite the name it is only *partly* Bayesian:

- `C` is `[N, num_classes]`, initialized to a symmetric prior pseudocount
  `alpha` — a **Dirichlet prior**, the one genuinely Bayesian piece.
- On labeling point `ℓ` with observed label `c`, neighbors are updated
  **additively**: `C[u, c] += K[u, ℓ]`, then normalized `norm_C = C / C.sum(1)`
  to a per-class distribution.
- Multiple neighbors reaching the same `(target, class)` are combined by keeping
  the **max** kernel value in one helper; the main path is additive.
- Selection = marginal gain of that normalized coverage. The
  `diff_method` / `cont_method` / `weight_method` flags are variants of
  "sum the positive per-class gain, optionally weighted by the class
  distribution".

**Key limitation:** it **assumes clean labels**. There is no annotator-accuracy
term and no posterior update. So Itai covers point (1) only; points (2) and (4)
are absent and (3) is a crude heuristic.

**Consequence:** our novelty over Itai is precisely points (2), (3)-done-right,
and (4) — the noisy-annotator regime this lab studies.

---

## 2. The probabilistic model (points 1–3)

### Objects and known quantities

- Point `u` has an unknown true label `y_u ∈ {1..C}`.
- A labeled neighbor `ℓ` was annotated by annotator `a(ℓ)`, giving the observed
  (possibly noisy) label `ỹ_ℓ`.
- We know the annotator's **confusion matrix**
  `M^a[k, j] = P(ỹ = j | y = k)`.
  For a symmetric annotator with accuracy `ρ_a`:
  `M^a[k,k] = ρ_a`, off-diagonal `= (1 − ρ_a) / (C − 1)`.
  Both `ρ_a` and the confusion matrix are already available in the codebase:
  `deep_al/pycls/noise/annotator.py` exposes `Annotator.accuracy`
  (`= 1 − noise_rate`), `noise_type` (`sym`/`asym`), and
  `noise_transition_matrix`. We should support the **full confusion matrix**,
  not just a scalar accuracy, so asymmetric annotators work for free.
- `π(k)` = class prior (uniform, or empirical).

### Step A — posterior on the neighbor's own true label (point 2)

Given the observation `ỹ_ℓ`:

```
b_ℓ(k) := P(y_ℓ = k | ỹ_ℓ)  ∝  π(k) · M^a[k, ỹ_ℓ]
```

For a symmetric annotator this is `b_ℓ(ỹ_ℓ) = ρ_a`, and `(1 − ρ_a)/(C − 1)`
elsewhere. **This is where annotator accuracy enters.** A near-oracle annotator
gives a peaked `b_ℓ`; a 90%-noise annotator gives a nearly flat one.

### Step B — propagate the neighbor's belief to `u` through the kernel

Interpret `K[u, ℓ] ∈ [0,1]` as the probability that `u` shares `ℓ`'s true label
(the ProbCover "same ball ⇒ same label" assumption, made soft). A single
neighbor's likelihood for `u`'s class is a mixture:

```
q_ℓ(u, k)  =  K[u, ℓ] · b_ℓ(k)  +  (1 − K[u, ℓ]) · π(k)
```

- Near neighbor + accurate annotator ⇒ peaked toward `ỹ_ℓ`.
- Far neighbor or noisy annotator ⇒ flat toward the prior `π`.

This single line encodes points (1) and (2) together.

### Step C — fuse all labeled neighbors (point 3)

Combine independent evidence from every labeled neighbor `ℓ ∈ L` by a
**product of experts** (naive Bayes over neighbors), computed in **log-space**
for numerical stability:

```
log P̃(y_u = k)  =  log π(k)  +  Σ_{ℓ ∈ L}  w_{uℓ} · log q_ℓ(u, k)
P(y_u = · )      =  softmax_k  log P̃(y_u = ·)
```

- `w_{uℓ}` is a per-neighbor temper/weight (see §2.4).
- **Product, not addition** — see §3 for the full justification and the
  mandatory empirical check.

This changes the meaning of the coverage matrix: instead of
`C[u, c] = max kernel coverage`, `C` now stores the **log unnormalized class
posterior** per point, and the usable distribution is `softmax(C[u])`.

### 2.4 Correlated-neighbor caveat (must be in the model)

Naive product-of-experts assumes neighbors are *independent* evidence. Points
that are close **to each other** are correlated, so the raw product is mildly
**overconfident**. Two mitigations, expose as knobs:

1. **Temper** the likelihood: `w_{uℓ} = β`, a global `β < 1`
   (`log q_ℓ → β · log q_ℓ`).
2. **Kernel-weight** each term: `w_{uℓ} = K[u, ℓ]`, so redundant far neighbors
   contribute less.

Recommended default: kernel-weighting (2) on, `β` exposed as `TEMPER_BETA`.

---

## 3. Product vs. addition — agreement + REQUIRED empirical check

**Mathematically** the product-of-experts (§2.C) is the correct Bayesian
combination of independent evidence:

- More agreeing neighbors should **sharpen** the posterior (entropy ↓); additive
  accumulation saturates and cannot represent "many weak votes ⇒ one confident
  belief".
- A single confident-but-wrong neighbor should be **overridable** by several
  others; the product form allows this, the additive form fights it.
- It degrades gracefully to the prior `π` when a point has no strong neighbor.

**BUT** this replaces Itai's additive rule, which is the rule his reported
results were obtained with. Being mathematically correct is not sufficient
justification to change it.

> **ACTION / open verification (do not skip):** implement both combination rules
> behind a config switch `COMBINE = {"product", "additive"}` and **empirically
> compare** them (same seeds, same noise levels, dashboard Comparison tab).
> Only adopt `product` as the default **if it improves (or at least does not
> harm) downstream accuracy** vs. Itai's `additive`. If product degrades
> results, keep additive as default and document why. Do not assume the math
> settles it.

The `additive` branch should reproduce Itai's numbers as a sanity check.

---

## 4. Reliability: prior now, posterior later (point 4 — FUTURE WORK)

Keep two reliabilities distinct:

- **Annotator reliability** `ρ_a` — the *prior*, known from the annotator
  (used in Step A). **This is what we ship first.**
- **Per-point reliability** `r_ℓ = P(ℓ correctly labeled | evidence)` — starts
  at `ρ_{a(ℓ)}` and gets *updated* as we learn. **Future work.**

When `r_ℓ` is available it replaces the raw `ρ` in Step A:

```
b_ℓ(k)  =  r_ℓ · 1[k = ỹ_ℓ]  +  (1 − r_ℓ) · π(k)
```

Two ways to obtain `r_ℓ` (both deferred; recorded here for completeness):

- **(4a) AUM, model-based** — *this is the SECOND step / future work.* Train on
  the noisy labeled set, get `AUM_ℓ`, fit densities
  `p(AUM | clean)` / `p(AUM | mislabeled)` (threshold-sample / beta-mixture,
  DivideMix-style), then
  `r_ℓ = ρ p(AUM_ℓ|clean) / [ρ p(AUM_ℓ|clean) + (1−ρ) p(AUM_ℓ|mislabeled)]`.
  Caveat: **circularity** — the AUM model was trained on the very noisy labels
  it is judging.
- **(4b) graph-based, no trained model** — leave-one-out posterior on `ℓ` from
  the label graph itself (Steps A–C over `ℓ`'s *other* labeled neighbors). If
  `ℓ`'s neighbors disagree with `ỹ_ℓ`, `r_ℓ` drops automatically. Cheaper,
  non-circular, available from round one. Preferred first approach when we get
  to point (4).

**Unification with existing NPC:** the current `NoiseAwareCoverage` applies a
**hard** `r_ℓ ∈ {0,1}` filter (drop a noisy point's coverage entirely). The
Bayesian model is the **soft** generalization — a suspected-noisy point is
down-weighted by `r_ℓ` rather than deleted. `NoiseAwareCoverage` is the limiting
case `r_ℓ = 1[AUM clean]`.

---

## 5. Training-label policy (CONFIGURABLE — compare all options)

The coverage model produces *soft* per-class beliefs, but a classifier still has
to be trained. **How the training targets are formed must be a parameter**,
analogous to the existing `TRAIN_ON_CLEAN`, so we can compare options head-to-head
on the dashboard.

New config knob `BAYES_COVERAGE.TRAIN_LABEL_MODE` with (at least) these options:

| Value            | Training targets                                                     |
|------------------|----------------------------------------------------------------------|
| `observed`       | the raw annotated (noisy) label `ỹ_ℓ` — current behavior             |
| `soft_posterior` | the soft posterior `softmax(C[ℓ])` as a soft label                   |
| `map_posterior`  | the argmax of the posterior (a possibly self-corrected hard label)   |
| `reweighted`     | observed labels weighted per-sample by reliability `r_ℓ` (needs pt 4) |
| `filtered`       | drop low-reliability points (the `TRAIN_ON_CLEAN` limit; needs pt 4) |

**Requirement:** these are experimental arms to be **compared**, not a single
choice baked in. Wire the knob so a sweep can vary it and the dashboard can
facet on it. Interacts with the existing `TRAIN_ON_CLEAN` / `NOISE_FILTER`
logic — reconcile so they do not silently conflict (e.g. `filtered` should route
through the same clean-mask path).

---

## 6. Selection criterion

Two options; ship the first, offer the second.

- **Soft-coverage gain (drop-in).** Reuse the greedy loop; score each candidate
  by how much annotating it raises total posterior mass toward confident labels
  — essentially Itai's `weighted` diff applied to the softmax posterior. Cheapest,
  closest to the existing `batched_diffs` machinery.
- **Expected uncertainty reduction (fuller Bayesian).** Pick the point whose
  annotation most reduces summed posterior entropy `Σ_u H(y_u)` over the pool
  (BALD-style on the coverage graph). More principled, more compute. Second
  variant, paper-worthy.

Expose as `BAYES_COVERAGE.SELECTION = {"coverage_gain", "entropy"}`.

---

## 7. Architecture — new class, not a modification

Modifying `CoverageMatrixMethod` is rejected: the meaning of `C`
(max-coverage → log-posterior), its update rule (`torch.maximum` → log-space
product), and the selection score all change. Cramming both behind flags makes
`_select_one` unreadable. Subclass and reuse the shared infrastructure (kernel
build, `set_rel_features`, `init_sampling_loop`, `_finalize_selection`). Mirror
the existing hierarchy:

```
CoverageMatrixMethod                          # hard, binary max-coverage (unchanged)
├── NoiseAwareCoverage                        # hard clean/noisy filter (unchanged)
└── BayesianCoverage(CoverageMatrixMethod)             # points 1–3: soft posterior + annotator accuracy
    └── NoiseAwareBayesianCoverage(BayesianCoverage)   # point 4 (FUTURE): soft reliability r_ℓ between batches
```

`NoiseAwareBayesianCoverage` reuses NPC's inner-batch / train-between-batches
loop (`select_samples` + `_rebuild_C_from_clean`), swapping the hard mask for the
soft `r_ℓ`. When no reliability source is set / too few labels exist, `r_ℓ`
defaults to the annotator prior `ρ`, so the base class runs standalone.

### Methods to write in `BayesianCoverage`

- override `init_sampling_loop`: initialize `C` as the broadcast **log-prior**
  `log π`, not max-coverage.
- `_neighbor_loglik(ℓ, ỹ_ℓ, r_ℓ)` → the `log q_ℓ(·, k)` vector (Steps A + B).
- override `_select_one`: accumulate `C += w · log q_ℓ` (or the additive branch
  for the comparison), mask labeled points, argmax the acquisition score,
  annotate in-loop via the existing `annotator_manager.annotate_samples`.
- `posterior()` → `softmax(C)` helper for scoring, training targets, diagnostics.
- pull `ρ_a` / confusion matrix from `annotator_manager`
  (`Annotator.accuracy` + `noise_transition_matrix`).

### New config block `BAYES_COVERAGE`

- `PRIOR` — `uniform` | `empirical` (source of `π`).
- `COMBINE` — `product` | `additive` (see §3; compare!).
- `TEMPER_BETA` — global tempering `β` for the product rule.
- `NEIGHBOR_WEIGHT` — `kernel` | `equal` (the `w_{uℓ}` scheme).
- `SELECTION` — `coverage_gain` | `entropy`.
- `TRAIN_LABEL_MODE` — see §5.
- `RELIABILITY_SOURCE` — `prior` (ship) | `graph` | `aum` (future).
- reuse existing `KERNEL_TYPE`, `INITIAL_DELTA`, `INITIAL_SIGMA`.

---

## 8. Build order

1. **`BayesianCoverage`** with `RELIABILITY_SOURCE=prior`, symmetric annotator,
   `SELECTION=coverage_gain`, `COMBINE` switch present. Validate on a synthetic
   2-cluster toy with known answer; confirm it reduces to ProbCover as `ρ → 1`.
2. Add full **confusion-matrix** annotator support.
3. **Compare `product` vs `additive`** (§3) and pick the default from results.
4. Wire **`TRAIN_LABEL_MODE`** (§5) and compare the arms.
5. (Future) graph-based `r_ℓ` (4b), then AUM `r_ℓ` (4a) as
   `NoiseAwareBayesianCoverage`; confirm it reduces to `NoiseAwareCoverage` as
   `r_ℓ → {0,1}`.
6. (Future) `entropy` acquisition variant.
7. Sweep vs. PC / NPC / Itai-additive across noise levels on the dashboard.

---

## 9. Open verification checklist (carry forward)

- [ ] `product` vs `additive`: does the mathematically-correct rule actually
      improve downstream accuracy? (§3) — **must verify, do not assume.**
- [ ] `additive` branch reproduces Itai's reported numbers (sanity).
- [ ] `ρ → 1` limit reproduces plain ProbCover.
- [ ] `r_ℓ → {0,1}` limit (future) reproduces `NoiseAwareCoverage`.
- [ ] `TRAIN_LABEL_MODE` arms compared head-to-head across noise levels.
- [ ] correlated-neighbor overconfidence controlled by `TEMPER_BETA` /
      `NEIGHBOR_WEIGHT`.
