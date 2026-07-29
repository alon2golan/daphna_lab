# §9.3 Classifier→posterior feedback loop ("AUM idea, chicken-and-egg") — design options

*Research memo, 2026-07-22. No implementation yet — Alon to pick a direction.*
*Companion to `bayesian_coverage_NEXT_STEPS.md` §9.3 and `papers/probabilistic kernel AL.pdf` §2.5 Eq. 9.*

## The question

Close the loop: classifier predictions → extra evidence in the selection posterior →
better targets → retrain. What are the design options, and is the paper's Eq. 9 the
same idea?

## Grounding fact (§4.5 B) — where the upside lives

The classifier and the posterior hold **non-overlapping knowledge, in opposite regimes**:

| judge | all-pool clean @1000 | covered @ noise .8 |
|---|---|---|
| classifier | **46.5** | 27.5 |
| graph posterior | 33.2 | **37.5** |

The classifier generalizes past the δ-balls (~13 pts of far-field knowledge the
posterior lacks); under heavy noise the posterior knows more than the classifier
extracts. Any good design exploits this asymmetry rather than blending uniformly.

Extra lever: the champion arm (warm-norm linear + `pseudo_all`, 48.4@1000) takes its
training targets **from the posterior**, so improving the posterior moves *both* the
selection plane and the training targets.

## Is the paper's Eq. 9 the same idea?

Eq. 9 (`probabilistic kernel AL.pdf`, §2.5 "Model uncertainty", p.5):

$$V^{\mathcal{L}}_{mj} = \alpha + \sum_{x_l\in\mathcal{L}} K_{lj}\mathbb{1}_{\{y_l=m\}} + \beta \sum_{x_u\in\mathcal{U}} (1-\gamma_u) K_{uj}\mathbb{1}_{\{\tilde y_u=m\}}$$

**Yes in spirit** — it is Daphna's formalization of exactly this loop (model assigns
pseudo-label ỹ_u with confidence 1−γ_u; folded in as extra kernel-weighted, β-discounted
counts in the veracity matrix). Good sign for roadmap alignment. **But four structural
axes differ from what our framework would do, and on each Eq. 9 takes the weaker choice
per our own results:**

1. **Additive/Dirichlet soft-count formalism** (paper Eqs. 4–6). Our gate experiment
   showed additive loses decisively to the product at high budget (takeaway 4). The
   product-form transplant is `C[u] += β_clf · log p_clf(u,·)` — one extra tempered
   expert, literally the §1.4 `TEMPER_BETA` machinery with a classifier message.
2. **Hard pseudo-labels + scalar confidence** (argmax ỹ_u, weight 1−γ_u) vs folding the
   **full softmax as soft evidence**. Argmax discards the distribution exactly where the
   classifier is unsure — the regime where the discount matters most.
3. **Kernel propagation** — Eq. 9 spreads u's pseudo-label to all neighbors j via K_uj.
   The classifier is already a smooth function of the embedding, so kernel-spreading
   double-smooths; under our tophat kernel it means every unlabeled point votes in every
   ball it touches — massive correlated-evidence accumulation, the exact §1.4
   overconfidence failure. Point-local injection is the cleaner comparison.
4. **No noise model on the classifier** — (1−γ_u) is the model's *own* confidence, which
   is the AUM circularity trap in one symbol: a classifier trained on noisy labels is
   confidently wrong and Eq. 9 injects that at high weight.

**Verdict:** treat Eq. 9 as *the idea's statement*, not the design to copy.

## Options, ranked by chance of beating 48.4@1000

### Option 1 — Classifier-as-annotator with an *estimated* confusion matrix ⭐ top pick

Treat the classifier as one more annotator in the existing multi-annotator machinery:

- Its argmax on point u becomes an "observed label" ỹ_u from annotator "clf".
- Its confusion matrix **M_clf is measured externally** — cross-validated on the labeled
  set, or estimated by the §9.4 DS-EM machinery (which estimates confusion matrices
  anyway) — **never taken from the classifier's own softmax confidence**.
- The vote enters through **Step A** (`_belief`, bayesian_coverage.py:171), exactly like
  a human label: `b_u(k) ∝ π(k)·M_clf[k, ỹ_u]`. Step A does the discounting for free:
  a measured 46%-accurate classifier enters as a 0.46-peaked belief; at noise .8 the
  measured M_clf is near-flat → near-uniform belief → the classifier **self-mutes**
  exactly where §4.5 B says it should, with zero special-case code.
- Per-class asymmetry for free (Step A already takes full confusion matrices, commit
  `9b54ce0`) — real for classifiers, which confuse visually similar classes, not
  uniformly.
- **Design note:** fold b_u at point u itself only — skip Step B kernel spreading (the
  classifier is already smooth; kernel-propagating it double-smooths; see objection 3
  to Eq. 9).

Why top: kills circularity honestly (measured reliability, not self-report — the same
epistemology the framework applies to humans, and consistent with our §7.3 story
"the posterior is a better label-quality judge than AUM"). It is also the principled
bridge to the paper's §3 multi-annotator model (classifier = a free annotator with
τ_clf), Tomer's two-annotator threshold rule (paper Eq. 15), and §9.5's expected-gain
acquisition. Paper-grade framing.

### Option 2 — Uncovered-points-only injection

Restrict the classifier expert to points where the graph has literally nothing (§9.3's
own guardrail). Composable with Option 1 (annotator-vote on uncovered points only).

- Surgical fix for the §7.9 failure: far-field NW targets failed because they were
  confidently wrong at distance; the classifier is the object that actually generalizes
  far (46.5 all-pool).
- Feeds **both planes of the champion arm**: `bayes_product` selection sees uncovered
  mass shrink where the classifier is sure, and `pseudo_all` training (currently
  covered-only) gains far-field targets.
- **Risk (acquisition side):** classifier-"covered" regions stop being sampled — if the
  classifier is wrong there the error becomes permanent. Mitigations to test:
  (a) measure selection-trajectory drift in the replay; (b) an ablation cell where the
  classifier evidence affects **training targets only**, leaving acquisition untouched.

### Option 3 — Soft tempered expert everywhere (β_clf < 1), calibrated

The direct product-form transplant of Eq. 9: add `β_clf · log p_clf(u,·)` to every
point's C (§1.4 machinery).

- **Prerequisites it surfaces:** (a) the §7.4 β-sweep was never run — the tempering
  machinery is untested even for labeled evidence; (b) the softmax needs temperature
  calibration first (noisy-trained heads are badly calibrated; β_clf and temperature
  are confounded knobs).
- Lower expected upside than 1–2: on covered points it dilutes the better judge with the
  worse one at high noise — unless β_clf is noise-gated (couples to §7.1 gate + §9.4
  estimator).

### Option 4 — Full EM iteration (posterior ↔ retrain to fixed point)

The maximal loop. Highest variance, classic self-confirmation collapse risk, retrains
inside the loop (expensive). Only after Option 1/2 shows a one-shot gain. Note: pure
self-training adds nothing by construction (a classifier folding back its own argmax
gains no information) — the value is entirely in the **fusion** with independent graph
evidence, so one well-designed injection may capture most of it.

### Not recommended

- **Eq. 9 verbatim** — additive formalism + hard labels + kernel spread +
  self-confidence weighting: each axis is the weaker choice per our own results.
- **Confidence-only gating without a reliability estimate** — AUM's epistemology; would
  undermine the §7.3 judge-vs-judge story.

## Circularity mitigations (the chicken-and-egg core), applicable to any option

- **Measured reliability, not self-report** (Option 1's core move).
- **Cross-fitting / co-training style:** K-fold classifiers so no point is judged by a
  model that trained on labels from its own ball; or only let the classifier vote on
  points outside its pseudo-label training diet.
- **One-shot per episode first**, not iterate-to-fixed-point; track posterior drift.
- **Decouple loops:** round 1, classifier→posterior affects selection only OR training
  targets only (ablation cells), not both.

## Validation plan — replay first, zero GPU (per §9.3's own prescription)

All of Options 1–3 are scoreable on the §4.5 replay: replay the 12 `observed`
trajectories, fold classifier predictions (from the already-trained probes, or cheap
offline linear/MLP probes) under each variant, score posterior quality (argmax
accuracy, purity, NLL) **with vs without**, split by covered/uncovered × noise ×
budget. §4.5 B predicts exactly where each variant should win (clean/low-noise,
uncovered, low–mid budget); if the replay doesn't show the gain in the predicted cell,
no training sweep. Add one replay metric specific to this direction: **selection
overlap / trajectory divergence**, since posterior changes move the acquisition too.

## Literature anchors (if a scan is wanted before deciding)

- Label propagation with pseudo-label injection: Zhu & Ghahramani LP; Iscen et al.,
  "Label Propagation for Deep Semi-supervised Learning", CVPR 2019.
- Confidence-thresholded self-training: FixMatch (its threshold = a crude γ_u gate).
- Co-training / cross-fitting for breaking circularity: Blum & Mitchell.
- Dawid–Skene EM with a model-as-annotator (connects to §9.4's DS-EM winner).

Option 1 composes these into something none of them has: a noise-aware coverage
posterior arbitrating between human and model annotators.
