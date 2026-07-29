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

### Step A — likelihood of the neighbor's annotation (point 2)

> **⚠ CORRECTED 2026-07-29** (prior double-count): the expert is the pure
> **likelihood**, NOT the posterior `b_ℓ(k) ∝ π(k)·M^a[k, ỹ_ℓ]` originally
> written here. A product of N posterior-experts (plus the π out-of-ball
> fallback of the original Step B) multiplies π up to N+1 times and collapses
> to `argmax π` under any non-uniform prior. π enters ONCE, in Step C.
> Identical numerics under uniform π + tophat K, so all runs to date stand.
> Full derivation: bayesian_coverage_THEORY.md Step A fix box.

Given the observation `ỹ_ℓ`:

```
L_ℓ(k) := P(ỹ_ℓ | y_ℓ = k)  ∝  M^a[k, ỹ_ℓ]      (column-normalized)
```

For a symmetric annotator this is `L_ℓ(ỹ_ℓ) = ρ_a`, and `(1 − ρ_a)/(C − 1)`
elsewhere. **This is where annotator accuracy enters.** A near-oracle annotator
gives a peaked `L_ℓ`; a 90%-noise annotator gives a nearly flat one.
(The posterior form `b_ℓ(k) ∝ π(k)·L_ℓ(k)` survives as `_belief`, used only by
the §7.9 NW mixture, where each point enters once with weights summing to 1.)

### Step B — propagate the neighbor's evidence to `u` through the kernel

Interpret `K[u, ℓ] ∈ [0,1]` as the probability that `u` shares `ℓ`'s true label
(the ProbCover "same ball ⇒ same label" assumption, made soft). A single
neighbor's likelihood for `u`'s class is a mixture:

```
q_ℓ(u, k)  =  K[u, ℓ] · L_ℓ(k)  +  (1 − K[u, ℓ]) · (1/C)
```

- Near neighbor + accurate annotator ⇒ peaked toward `ỹ_ℓ`.
- Far neighbor or noisy annotator ⇒ flat toward uniform — constant across `k`,
  so a K=0 neighbor is exactly evidence-free after the softmax (the original
  `π(k)` fallback re-multiplied the prior per far neighbor; see the Step-A box).

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

## 9. Testing & comparison plan

This section is the concrete "how do we know it works, and that we didn't break
anything" plan. Four layers: (A) unit/correctness, (B) regression — protect the
existing `CoverageMatrixMethod` / `NoiseAwareCoverage` path, (C) performance,
(D) the experimental comparison that decides whether we continue and which
version becomes the default.

### A. Unit / correctness tests (fast, CI-style, synthetic)

Put these in `deep_al/tests/test_bayesian_coverage.py`. They must run on CPU on
a tiny synthetic feature set (e.g. 2 well-separated Gaussian clusters, 200
points, 2-4 classes) so they are seconds-fast and deterministic.

1. **Posterior is a valid distribution.** For any state of `C`,
   `posterior()` rows are non-negative and sum to 1 (both `product` and
   `additive`).
2. **`b_l` sanity (Step A).** `_neighbor_true_label_belief(c, r)` returns
   `b[c]=r`, off-diagonal `(1-r)/(C-1)`, sums to 1; `r=1` -> one-hot;
   `r=1/C` -> uniform.
3. **Single-neighbor propagation (Step B).** For one labeled point with `r=1`:
   a point at `K=1` gets posterior -> one-hot on the label; a point at `K=0`
   gets posterior -> prior `pi`. Interpolates monotonically in `K`.
4. **Annotator-accuracy monotonicity (point 2).** Fixing geometry, lowering `r`
   moves the covered point's posterior monotonically toward `pi` (peak
   probability on the label is non-increasing in noise).
5. **Multi-neighbor sharpening (point 3, product).** Two neighbors that agree on
   a label produce a MORE peaked posterior than either alone (entropy strictly
   decreases). Two neighbors that disagree produce HIGHER entropy than a single
   confident one. This is the property additive accumulation fails — assert the
   `product` branch has it.
6. **ProbCover limit.** With `r=1` (oracle), a single class, hard-threshold
   (tophat) kernel, and `SELECTION=coverage_gain`, the sequence of selected
   points matches plain `CoverageMatrixMethod` on the same toy (or covers the
   same set). Documents that Bayesian ⊇ ProbCover.
7. **Determinism.** Same `RNG_SEED` + same config -> identical selected indices
   across two runs.
8. **Numerical safety.** No `nan`/`inf` in `C` or `posterior()` after a full
   selection loop, including `K` values at exactly 0 and 1 and `r` near 1.

### B. Regression tests — DO NOT BREAK the existing coverage path

The whole point is that `BayesianCoverage` is a *new* class; the existing
`CoverageMatrixMethod` and `NoiseAwareCoverage` (NPC) must be byte-for-byte
unaffected. Guard rails:

1. **Import isolation.** `bayesian_coverage.py` only *imports from*
   `coverage_matrix_methods.py`; it must not modify module-level state,
   monkey-patch, or change any base method. Confirm by `git diff` on
   `coverage_matrix_methods.py` being **empty** across this whole effort. (If a
   genuine refactor of the base is ever needed, that is a separate reviewed
   commit with its own regression run.)
2. **Golden-run equivalence for `CoverageMatrixMethod`.** Before any change,
   capture a reference run: one small ProbCover (`SAMPLING_FN` unchanged) AL run
   on CIFAR features, fixed seed, save the selected-index sequence and the
   per-episode test accuracy to `docs/golden/probcover_ref.json`. After the
   Bayesian code lands, re-run the identical config and assert the selected
   indices and final accuracy are **identical** (indices exactly; accuracy to
   float tolerance). Same for one **NPC** config -> `npc_ref.json`.
3. **Routing safety.** After registering `BayesianCoverage` in
   `ActiveLearning.py`, assert existing `SAMPLING_FN` values
   (`probcover`, `npc`, `random`, ...) still resolve to the same classes.
   Adding the new branch must not shadow or reorder existing ones.
4. **Config additivity.** The new `BAYES_COVERAGE` cfg block must have complete
   defaults and must not appear in / alter any existing config key, `config_key`
   hashing, or dashboard `about=<hash>` URLs (cf. run-tagging notes in
   `CLAUDE.md`). Verify an existing sweep's `config_key` is unchanged.
5. **Full existing test suite green.** `deep_al/tests/` (incl.
   `test_annotators.py`) passes unchanged.

### C. Performance / resource tests

`product` accumulates a dense `[M, num_classes]` log-update **per selected
point** and computes `K @ uncertainty` **per greedy step**; on CIFAR-100
(`num_classes=100`) this is heavier than the base method's single-column
`torch.maximum`. Measure before scaling up.

1. **Wall-clock per selection round** vs. `CoverageMatrixMethod` and Itai's
   `additive`, on the real CIFAR-10 and CIFAR-100 feature sets, at budgets
   {100, 300, 1000}. Record as a small table; flag if `product` is > ~2x the
   base per-round cost.
2. **Peak GPU memory** for the dense `[M, C]` accumulator and the `K @ u`
   matmul at the largest `M` we intend to run (full CIFAR pool). Confirm it fits
   the `g0` GPUs; if not, that motivates the (currently unimplemented) sparse
   path.
3. **Scaling knobs behave.** `num_classes` 10 -> 100 and pool size scaling do
   not blow up latency super-linearly beyond the expected `O(M·C)` /
   `O(M^2)` matmul.
4. Perf tests are informational (not pass/fail) but must be recorded in the PR
   so we choose budgets/datasets that are actually runnable on the shared
   cluster (see SLURM etiquette in `CLAUDE.md`).

### D. Experimental comparison — what to compare, and the decision rule

This is the layer that decides **whether we continue** and **which version is
the default**. Run through the existing sweep + dashboard machinery
(`--track_progress`, Comparison tab; see the al-dashboard skill and run-tagging
notes in `CLAUDE.md`). All arms share seeds, budgets, kernel/delta, and dataset.

**Fixed experimental grid (baseline):**
- Datasets: CIFAR-10 first (cheap, `C=10`), then CIFAR-100 (`C=100`, the
  stress case).
- Noise levels (symmetric annotator): {0.0 (clean sanity), 0.2, 0.5, 0.8}.
- Budgets: {100, 300, 1000} selected points.
- Seeds: >= 3 (reuse the lab's seed convention, e.g. 1/2/3) for mean +- std.
- Metric: downstream test accuracy vs. budget (primary); also report mean
  posterior calibration / label-recovery of covered points as a diagnostic.

**Arms to compare (methods):**
1. `random` — floor.
2. `probcover` (`CoverageMatrixMethod`) — the current non-noise-aware baseline.
3. `npc` (`NoiseAwareCoverage`) — current noise-aware baseline.
4. `bayes_additive` — `BayesianCoverage` COMBINE=additive == Itai's rule.
   **This is the Itai-reproduction control.**
5. `bayes_product` — `BayesianCoverage` COMBINE=product == our proposed math.

**Decision rules (must be met to proceed / to flip defaults):**
- **Sanity:** at noise 0.0, `bayes_product`, `bayes_additive`, and `probcover`
  should be within noise of each other. If Bayesian *hurts* on clean data,
  stop and debug — the model must not cost accuracy where there is nothing to
  gain.
- **Itai reproduction:** `bayes_additive` reproduces Itai's reported behaviour
  (and matches `probcover` at noise 0.0). If not, the additive branch is wrong
  and no comparison is valid yet.
- **Core hypothesis (points 1-3):** at noise >= 0.2, `bayes_product` should beat
  `probcover` (uses annotator accuracy) and be competitive with / beat `npc`.
  If it does not beat `probcover` under noise, the annotator-accuracy weighting
  is not paying off — revisit the model before adding complexity.
- **product vs additive (design §3 — the load-bearing check):** adopt
  `COMBINE=product` as the default **only if** it is >= `bayes_additive` on
  downstream accuracy (within std) across the noise grid, with a real win at
  >= one noise level. If `product` is worse, **keep `additive` as default** and
  record the finding; the math being "correct" does not override measured harm.

**TRAIN_LABEL_MODE comparison (design §5), run after a method winner is chosen:**
Fix the winning coverage method, then sweep
`TRAIN_LABEL_MODE ∈ {observed, soft_posterior, map_posterior, reweighted*,
filtered*}` (`*` = needs point-4 reliability, so deferred) over the same
noise/budget/seed grid; facet the dashboard on the `p_train_label_mode` tag.
Report which target policy wins per noise level — this is itself a result, not a
single default to bake in.

**Gate to point 4 (AUM / reliability, future work):** only start point 4 once
`bayes_product` (or `additive`, whichever wins D) has beaten `probcover` under
noise AND a `TRAIN_LABEL_MODE` winner is established — otherwise the soft-`r_l`
work has no validated base to build on.

### Suggested test-artifact layout

- `deep_al/tests/test_bayesian_coverage.py` — layer A + the regression asserts
  in B1/B3/B4.
- `docs/golden/{probcover_ref,npc_ref}.json` — B2 golden runs.
- a launcher (e.g. `batch_utils/sweep_bayes_coverage.py`) for layer D, with
  `--track_progress` and `--tag combine=<...> --tag train_label_mode=<...>` so
  the dashboard Comparison tab can facet the arms.

---

## 10. Open verification checklist (carry forward)

- [ ] `product` vs `additive`: does the mathematically-correct rule actually
      improve downstream accuracy? (§3) — **must verify, do not assume.**
- [ ] `additive` branch reproduces Itai's reported numbers (sanity).
- [ ] `ρ → 1` limit reproduces plain ProbCover.
- [ ] `r_ℓ → {0,1}` limit (future) reproduces `NoiseAwareCoverage`.
- [ ] `TRAIN_LABEL_MODE` arms compared head-to-head across noise levels.
- [ ] correlated-neighbor overconfidence controlled by `TEMPER_BETA` /
      `NEIGHBOR_WEIGHT`.
