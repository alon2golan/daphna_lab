---
title: "Bayesian Coverage — Status, intermediate results, open questions, next steps"
date: 2026-07-12
---

**Part 3/3 of the bayesian-coverage summary** (split 2026-07-12 from the former single
`bayesian_coverage_SUMMARY.md`): theory: `bayesian_coverage_THEORY.md` ·
results: `bayesian_coverage_RESULTS.md` · **status & next steps (this file)**.
Section numbering is preserved from the original file — §7 lives here, §1–§3 in THEORY,
§4–§6 in RESULTS; "§4.5", "takeaway 8" etc. refer to RESULTS.

## Status board (updated 2026-07-12, end of day)

**Nothing currently running on SLURM for this project.** The §9.1 w(d) sweep
(job 31058388, 12 runs) COMPLETED overnight 2026-07-13 and was collected the same
night — **verdict ❌ NOT ADOPTED, see §9.1**: clean/.2 negative again (−1.35/−0.43 @100),
so the out-of-ball direction is closed for good; recommended config unchanged.

### ▶ RESUME HERE (updated 2026-07-14)

**9.4 noise-rate bake-off is DONE (measurement only — see §9.4 "Bake-off DONE" block).**
Headline: impurity-corrected LOO most accurate (but needs a clean anchor); Dawid–Skene EM
best assumption-free (separates clean-vs-.2 at every budget, +~0.25 low-noise bias); raw-LOO
saturates; Netta's q̂ under-estimates at low budget. **Nothing wired into production and no
estimator adopted yet — Alon deferred the "which one / how to estimate correctly" decision
to a future session.** Next up per the §9 execution order:

1. ~~**9.2 δ sweep**~~ ✅ CLOSED 2026-07-22 (stage 1 replay + stage 2 paired training,
   job 31108061): **δ=0.75 confirmed at every noise level and budget** — 0.6 loses on
   coverage volume, 0.9 on ball impurity; no retune for CIFAR100/SimCLR (see §9.2).
2. **In-ball RBF-weighted product exponents** (the §7.9 replay's 87→96.7 denoising win)
   — the one remaining *training* change with a measured large effect; 12-run paired sweep.
3. 9.3 classifier-as-tempered-expert (replay first) and the 9.5 posterior-gain
   annotator-selection design doc.

*(superseded plan, kept for context)* the original order had 9.4 first, then 9.2:
1. ~~**9.4 noise-rate estimator bake-off**~~ ✅ DONE 2026-07-14.
2. **9.2 δ replay sweep** (offline): grid {0.5, 0.6, 0.75, 0.9, 1.1} through the §4.5
   replay metrics; the §8.2 Q3 distance data says ±0.15 in δ moves coverage 8%↔49%.
3. **In-ball RBF-weighted product exponents** (the §7.9 replay's 87→96.7 denoising win)
   — the one remaining *training* change with a measured large effect; 12-run paired
   sweep once implemented.
4. 9.3 classifier-as-tempered-expert (replay first) and the 9.5 posterior-gain
   annotator-selection design doc.

| Item | Status |
|---|---|
| Core method (selection + 5 train modes) + comparison sweeps | ✅ DONE (§4, takeaways 1–7) |
| §7.5 pseudo-labeling arm | ✅ DONE — best arm at every noise incl. clean (takeaway 8) |
| Classifier-input normalization rerun | ✅ DONE — convergence-gated; +~1pt for `pseudo all` (takeaway 9, `bayesian_coverage_NORMALIZATION.md`) |
| §7.9 out-of-ball NW pseudo-targets, **constant** weight | ❌ ANSWERED, NOT ADOPTED (takeaway 10) — covered-only stands |
| §8 weekend open questions (thesis / code / distance distro) | ✅ ANSWERED 2026-07-12 (§8; distance data `results/cifar100/nearest_labeled_distance.*`) |
| §8.3 Q2 + 9.1 offline 1-NN/NW evaluator study | ✅ DONE 2026-07-12 — denoise-then-1NN ≈ best trained arm @ noise .8 with zero training (§8.3 Q2) |
| §9.1 w(d) distance-decayed OOB weight | ❌ ANSWERED 2026-07-13, NOT ADOPTED (sweep 31058388): clean −1.1..−1.35 at every budget — even the good shell dilutes; out-of-ball closed for good |
| §9.4 noise-rate estimator bake-off | ✅ MEASURED 2026-07-14 (offline, nothing adopted): DS-EM best assumption-free (separates clean-vs-.2 every budget), impurity-corr LOO most accurate (needs clean anchor), raw-LOO saturates, Netta q̂ under-estimates low-budget — see §9.4 block; **which estimator to adopt deferred** |
| §9.2 δ sweep (replay + paired training) | ✅ CLOSED 2026-07-22 — δ=0.75 confirmed at every noise/budget (job 31108061); 0.6 starves coverage, 0.9 poisons purity |
| §9.7 MLP head | ✅ ANSWERED 2026-07-22 — sklearn cold+best-val MLP adopted for observed labels (beats warm linear at every noise/budget); **pseudo-stack does NOT stack** (torch twin, wdfix sweep 31088628: −1..−5 vs linear+pseudo, deficit grows with noise) |
| §9.8 reverse-KL loss | ❌ ANSWERED 2026-07-22, NOT ADOPTED — ties soft-CE under noise, loses −0.7..−1.7 on clean (both heads); soft-CE stands |
| misp-vs-product 2×2×2 grid (§4.8) | ✅ MEASURED 2026-07-23 (sweeps 31116378/31116406/31116418; misp×pseudo enabled by MispPosteriorEngine) — **regime split: misp + pseudo(norm) + linear = new clean best 49.9@1000 (vs 48.4) and wins noise .2; product keeps noise ≥.5** (29.2 vs 25.2 @1000 n.8); §9.7 "no stack" extends to misp; NEW candidate: noise-adaptive acquisition |
| misp_pure control + cold twins (§4.8 addendum) | ✅ MEASURED 2026-07-26 (sweeps 31130233/31130234/31130235, same lSets as the hybrid seed-for-seed) — **PoE training targets are the denoiser: pure categorical ties hybrid clean (49.6 vs 49.9) then loses −0.7/−2.0/−4.5 @.2/.5/.8**; noise-.8 ladder decomposes additively (product 29.2 > hybrid 25.2 > pure 20.7 ⇒ selection and target family ~4-5 pts each); **cold ≈ warm for all pseudo arms** (warm-start poison is observed-label-only) ⇒ noise-adaptive candidate = switch *acquisition only*, keep PoE targets |
| clean-gap mechanism (§4.8 reading 8) | ✅ MEASURED 2026-07-28 (offline, `eval_clean_selection_geometry.py` + exact product replay on recorded lSets) — **misp ≡ ProbCover ≡ ideal coverage greedy at noise 0** (coverage 55.7% @1000, identical metrics); **product's label-conflict deficit (tied one-hot folds ⇒ deficit ≈ 0.5 on class-boundary overlaps) never retires** → ~47% of pick score by pick 300+, coverage 50.6% (−2,300 pseudo-labels), 76%-revisit anomaly explained ⇒ noise-adaptive candidate is now a single knob: **gate the conflict deficit by q̂ (DS-EM)** — no rule switch, product → ProbCover as q̂ → 0 |
| probcover × PoE-pseudo control (§4.8 reading 7b) | ✅ MEASURED 2026-07-29 (sweep 31150815, `ProbCoverPseudoAdapter` wiring) — clean 49.7 ≈ hybrid 49.9 ✔, but **noise: 46.7/38.5/18.1 vs hybrid 47.5/41.7/25.2 — n.8 below even misp_pure** ⇒ misp's conflict dust is noise-RESPONSIVE and its revisits (57→69% vs probcover's flat 52%) feed the PoE denoiser; pseudo-purity gap tracks test-acc gap ~1:1 (`eval_pc_vs_misp_pseudo_quality.py`); **full n.8 ladder: product 29.2 > hybrid 25.2 > pure 20.7 > pc×PoE 18.1 = 3-point dose-response on the uncertainty-weight dial** — strengthens the q̂-gate candidate |
| **Recommended config** | normalized `pseudo all`, covered-only; **selection: misp on clean/light noise (§4.8), product at noise ≥.5** — training plane CONFIRMED against w(d), MLP-stack and rKL too |
| Best unimplemented spin-off | **in-ball RBF-weighted product exponents** (§7.9 replay: labeled-plane denoising 87→96.7 @100 clean) |
| Open engineering guards | guard (b) per-ball caps (§7.5); `filtered`/`reweighted` threshold fix (§7.6) |

Everything through 2026-07-12 is committed on `bayesian_coverage`: `4a70f92` (docs split
+ §8/§9), `e4249fd` (offline eval scripts), `6e1dfba` (w(d) ramp + launcher `--ramp`).
Note `results/` is gitignored — the eval outputs live on disk only.

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

### 7.7 Additive-arm fidelity (Itai control) — superseded 2026-07-16

> **Status:** the low-budget collapse was the acquisition bug, fixed 2026-07-14
> (RESULTS, note above §4.1) — additive now ≈ probcover @100. More importantly, the
> Itai repro established that his actual "bayes" method is `BAYES_MISP` (diff max),
> **not** our `bayes_additive` — the faithful Itai control is the native `bayes_misp`
> port (RESULTS §4.7). `bayes_additive` stands as our own additive variant, with
> Itai's saturating deficit ceiling adopted for its acquisition. The theoretical
> formalism comparison remains open → §9.6.

On CIFAR100 our additive branch scores 13.4 @100 clean vs probcover + training's 23.6 —
it fails the "reproduces Itai ≈ probcover + training" sanity check that it *passed* on
CIFAR10 (takeaway 4).
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

### 7.9 Out-of-ball posterior propagation (noise-aware soft-kNN) — ❌ ANSWERED, NOT ADOPTED (sweeps 31050702 + 31050847, 2026-07-11)

**Problem (agreed conclusion):** the $\delta$-ball kernel's influence reaches only a
small fraction of the pool at low budget — @100 labels the in-ball evidence touches
~9k of the ~45k pool points (~20%; the `pseudo all` training-set size), and §4.6 shows
`all`'s low-budget dominance comes precisely from training-set *volume*. The other ~80%
sit at posterior = prior and are excluded by guard (a). The delta ball works well
*inside* the balls (purity-calibrated, product fusion); what's missing is that influence
**vanishes outside** — a point far from every labeled point should still be influenced by
its *relatively* nearest labeled evidence.

**Key design insight: the vanishing is caused by the absolute-evidence semantics, not
the kernel shape.** Swapping tophat for RBF only moves the cliff: any decaying kernel
under unnormalized evidence gives far points $\to$ prior (and the sparse-K build truncates
RBF at $\delta$ anyway, `coverage_matrix_methods.py:60`). What makes soft-kNN never vanish
is **competition** — weights normalized *over the labeled set*, so only relative
distances matter:

$$w_\ell(u) = \frac{\exp(-d^2(u,\ell)/T_u)}{\sum_{\ell'} \exp(-d^2(u,\ell')/T_u)},
\qquad P(y_u{=}k) = \sum_\ell w_\ell(u)\, b_\ell(k).$$

**Proposed two-regime posterior** (config `BAYES_COVERAGE.OUT_OF_BALL ∈ {none, nw}`):

- **Covered points (some neighbor within $\delta$): UNCHANGED** — tophat + product of
  experts, so every existing result stands and the comparison to sweep 31047129 is
  per-seed paired (only guard-(a)-excluded rows in `pseudo_label_targets` gain targets).
- **Uncovered points: Nadaraya–Watson mixture over the annotators' beliefs** $b_\ell$
  (the confusion-matrix-corrected per-class beliefs, *not* raw labels) — i.e.
  **noise-aware soft-kNN**: it strictly generalizes the `probcover + 1nn` evaluator
  (38.1 @100 clean — we *know* this signal is strong) into a training-target
  generator that still respects annotator accuracy under noise. Mixture, not product:
  a product of near-prior experts collapses to the prior — exactly the failure being
  fixed; the mixture reads "u carries the label of one of these anchors w.p. $w_\ell$".
- **Temperature: GAP-scaled, replay-validated (2026-07-10 — IMPLEMENTED).** The
  first idea — Zelnik-Manor–Perona local scaling $T_u = d_1 d_m$ — is **wrong in
  512-D SimCLR space**: distances concentrate ($d_1 \approx d_{50}$), the softmax
  smears over dozens of mixed-class anchors and the mixture collapses toward the
  global mean (measured: 4.6% vs 26% uncovered-acc @100 clean). The fix is
  dimensionless *gap* competition, exponent $-\kappa\,(d^2-d_1^2)/(d_m^2-d_1^2)$:
  nearest anchor weight 1, $m$-th nearest $e^{-\kappa}$, concentration-immune.
  $\kappa$ interpolates soft-kNN aggregation ($1$) $\to$ hard 1-NN (large);
  **$\kappa=3$, $m=7$ is the sweep default** (`NW_KAPPA`/`NW_M`). Offline replay
  (`eval_posterior_variants.py`, mean over seeds, uncovered / all-pool argmax acc):
  $\kappa=3$ $\approx$ hard 1-NN on clean (25.7 / 37.4 @100), *beats* it at noise
  .2/.5 (all-pool @1000: 42.3 vs 39.2 at .2; 29.7 vs 25.6 at .5); only fully-soft
  $\kappa=1$ wins at noise .8 (18.7 vs 12.4 @1000 — the aggregation regime;
  follow-up knob). The all-pool target acc @100 clean, 37.4%, $\approx$ the
  probcover+1nn evaluator's 38.1% — the replay reproduces the known 1-NN signal.
- **Confidence for topk/curriculum arms:** absolute $\max_k P$ is meaningless in the
  far regime; use the competition margin $w_1/w_2$ (Lowe-ratio style) or the entropy
  of $w$. `all` needs nothing.

**Expected effect:** `pseudo all`'s training set @100 grows ~9k $\to$ full pool (~45k),
with the far 80% receiving noise-aware-1-NN-quality targets. Combine with normalized
classifier inputs (takeaway 9) — `pseudo all` is the one arm that converges on them.

**Alternatives considered and rejected:** (i) *enlarge $\delta$* — full-weight impure
votes degrade exactly the calibrated in-ball behavior that works, and couple labeling
to selection; (ii) *tophat + absolute RBF tail* — helps a shell around the ball but
still $\to$ prior far away (fails the requirement; keep only as an ablation);
(iii) *graph diffusion / label spreading* (Zhou et al.) — propagates along the manifold
through chains of unlabeled points, strictly more expressive, but changes the covered
regime too; documented follow-up if NW plateaus.

**"Would NW also be better INSIDE the balls?" — answered offline (replay, 2026-07-10):**
mostly no, but one ingredient wins. Pure NW *mixture* over in-ball anchors ≈ product on
accuracy (it loses the count-aggregation that powers the noise-.8 denoising) and only
improves calibration at low noise (ECE 13–22 vs 22–28). But **RBF-distance-weighted
exponents inside the product** fix the §4.5 impurity outvoting essentially for free:
labeled-plane denoising 87.0 $\to$ **96.7** @100 clean, 75.7 $\to$ 87.7 @300,
68.4 $\to$ 81.5 @1000, +7–8 pts at noise .2, ~flat at .8 — with no measurable
unlabeled-plane cost. That is a strong candidate default for the *in-ball* posterior
(config: `KERNEL_TYPE` stays tophat for support; the change is distance-decayed
$w_\ell$ in `_neighbor_loglik`) — worth a paired 12-run confirmation sweep for the
`map/soft_posterior` arms, where labeled-plane denoising is the mechanism.

**RESULTS (sweeps 31050702 w=1.0 12/12 + 31050847 w∈{0.3,0.1} 24/24; per-seed paired
against the pseudo_all rows of the normalized rerun, job 31048960; collected
2026-07-11, `results/bayes_pseudo/nw_oobw_{episodes,deltas}.csv`):**

Paired $\Delta$ = NW arm − baseline, mean over seeds {1,2,3} (std across seeds ≤ 0.3
at clean/.2, ≤ 0.65 at .8 — per-seed signs fully consistent):

| noise | @100 w=1.0 / 0.3 / 0.1 | @300 | @1000 |
|---|---|---|---|
| 0.0 | −2.1 / −2.0 / **−1.6** | −2.0 / −1.8 / −1.4 | −1.4 / −1.3 / −1.1 |
| 0.2 | −1.0 / −1.0 / −0.8 | −1.1 / −0.9 / −0.5 | −0.9 / −1.0 / −0.9 |
| 0.5 | +0.0 / +0.3 / +0.4 | −0.4 / +0.2 / +0.4 | −0.6 / −0.4 / −0.4 |
| 0.8 | +0.8 / +0.9 / +0.5 | +0.3 / +0.4 / +0.5 | **+0.7 / +0.8 / +0.6** |

**Verdict: the down-weighting hypothesis fails — no OOB_WEIGHT setting recovers the
clean/low-noise loss.** The dose-response is monotonic in the right direction but far
too weak: going from full weight to $w=0.1$ (a 10× reduction in the far targets'
loss contribution) claws back only ~0.5 of the −2.1 clean drag @100, and essentially
nothing at noise .2 @1000 (−0.9 at *every* $w$). Extrapolating the trend, the weight
that eliminates the drag is ~0 — i.e. the mechanism is not "far targets are useful
but over-weighted", it is **"far targets are confidently wrong"** (26% uncovered-acc
targets vs the 71%-acc covered signal; the sharp gap-scaled mixture makes them
*look* like clean labels to the CE loss at any row scale). Meanwhile the noise-.8
gain (+0.5..+0.9 everywhere) also barely responds to $w$ — consistent with the
label-smoothing reading (near-uniform NW rows at flat beliefs), which survives
scaling because smoothing is shape-, not magnitude-driven.

**Disposition:** `OUT_OF_BALL=nw` stays in the code as an ablation but is **not**
part of the recommended configuration; the §4.6/takeaway-9 recommendation
(normalized `pseudo_all`, covered-only) stands. If out-of-ball propagation is
revisited, the replay says where the room is: (a) a **confidence floor on the NW
row** (keep only far targets whose top-1 mixture weight clears a margin — the 26%
mean hides a usable high-confidence subset), (b) $\kappa$ **noise-gating** ($\kappa=1$
at .8 was the replay winner — the observed .8 gain is real and might grow), or
(c) skip NW for training targets and harvest the *other* replay finding instead —
**rbf-weighted product exponents inside the balls** (87→96.7 labeled-plane denoising
@100 clean), which is still unimplemented in live code and is the higher-value
follow-up.

---

## 8. Open questions — Alon's weekend notes (2026-07-12), with answers

Questions raised 2026-07-12; answers verified against the thesis PDF / code / features on
disk the same day (sources cited inline).

### 8.1 About Netta's thesis (`oracle-teacher-proj/netta thesis aum.pdf`, TMLR submission version, 24 pp.)

**Q1 — Does the thesis report absolute accuracy numbers?**
**No.** Every results figure (Figs. 4–7, 12–13, 18–19, 23) plots **accuracy *difference*
vs. random selection** (y-axis "Accuracy Difference (%)", §4.2 p. 8), with the x-axis in
E[SPC] units (budget = E[SPC]·C/(1−noise)). Visible deltas: NPC ≈ +4..+7 pts over random
on CIFAR100 at 20–50% noise (Fig. 4), up to ≈ +12–14 pts in Framework 2 at 50% noise
(Figs. 12–13). The only absolute-accuracy remark is qualitative (§4.3 p. 10). **Implication:
every absolute NPC+AUM number we quote (e.g. 43.3 @noise .2 @1000) comes from our own
repro sweeps (jobs 30811243 / 30863746), not from the thesis — our §4 tables are the only
absolute anchor for the comparison.**

**Q2 — Why a linear classifier on the pretrained embedding, given 1-NN ≫ linear at low
budget (38.1 vs 23.6 @100 in our sweep 31048649, matching the ProbCover paper)?**
The thesis **never considers 1-NN as an evaluator** (grep-verified: kNN appears only as a
noise-filter variant, App. G, and a 20-NN cluster demo, App. B). The linear head is
"Framework 2" (§4 p. 7), justified only vs. Framework 1 (full ResNet-18 on images:
"easier to train and usually outperforms framework 1 in the low-budget regime") and
inherited from the ProbCover/TypiClust codebase convention (App. C). So the choice is
lineage, not an argued decision. Note: since the thesis plots only *differences from
random*, a uniformly stronger 1-NN evaluator could shift all curves without necessarily
flipping the NPC-vs-ProbCover ordering — no evidence either way in the thesis; our §8.3 Q2
task measures exactly this on our side.

**Q3 — Noise-rate estimation: how good, and is it in our code?**
In the thesis it is an **emergent byproduct of the LowBudgetAUM filter**, not a module:
q̂ = |L_noisy|/|L| from the filter partition (Alg. 1 l. 5), consumed by *noise dropout*
(η = max(min(q̂, 1−q̂), 0.1), §3.2) and *Weighted NPC* (edge weights 1−q̂, §3.2). Claimed
performance: Fig. 2 ("estimates noise rates more accurately—even with as few as two clean
samples per class") — read off a figure, no numeric error quoted; admitted failures: at
80% noise it can predict "nearly all samples noisy" without dropout (Fig. 19), and on
Clothing1M it failed and the known 38% was injected as a prior (App. E). **In our code:**
`LowBudgetAUMFilter` is implemented (`noise_filters.py:55–76`, thesis hyperparams
AUM_EPOCHS 40 / THR_PERCENTILE 80), and q̂ is implicitly printed every NPC round
(`coverage_matrix_methods.py:278` "filter kept X/Y") **but not stored in result.json**;
noise dropout and Weighted NPC are **not implemented** (the summary doc's "skip for now"
decision). **Cheap check, no GPU:** (a) parse the kept-X/Y lines from the 27 finished
NPC-repro runs (job 30811243) and score 1−X/Y against the realized noise
(`verify_realized_noise.py` reproduces the corruption exactly); (b) reproduce Fig. 2
directly by running `LowBudgetAUMFilter.predict_clean` vs `IdealNoiseFilter` on the same
lSets. → merged into direction 9.4.

### 8.2 About our code

**Q1 — Linear part only, or full ResNet18 on the features?**
**A pure linear probe**: `eval_model_type=from_features` sets `MODEL.LINEAR_FROM_FEATURES=True`
(`core/config.py:453–457`), and `builders.py:86` returns `FeaturesNet(512, 100)` with
`use_mlp=False` — forward pass is a single `nn.Linear(512→100)` (multinomial logistic
regression on frozen features); the ResNet lookup is never reached. The `resnet18` in
`output/CIFAR100/resnet18/` is cosmetic — `cfg.MODEL.TYPE` from the config, naming the
backbone that *produced* the SimCLR features (`train_al.py:497`). Head training: SGD,
BASE_LR 0.025, cosine schedule, momentum 0.9 Nesterov, wd 3e-4, 200 epochs, batch 100,
`SoftCrossEntropyLoss`.

**Q2 — Embedding size?**
**512**, confirmed on disk: `representations_bank/cifar-100_simclr/cifar100_simclr_train.npy`
is (50000, 512) float32, test (10000, 512); `builders.py:87` hardcodes the matching
`num_features=512` for CIFAR100.

**Q3 — Distance distribution from the first 100 labeled points to the rest (normalized features)?**
Computed 2026-07-12 on a **real episode-1 lSet** (bayes_product, noise 0, seed 1, sweep
30978366; recovered from `episode_0/lSet.npy` exactly as the §4.5 replay does), on
**unit-L2-normalized** features (yes — the selection plane always normalizes; note this is
plain unit norm, not the scale-preserving classifier variant). Nearest-labeled-point
Euclidean distance over the 44,900-pt pool:

| quantile | budget 100 | budget 1000 |
|---|---|---|
| min | 0.138 | 0.079 |
| 5% | 0.560 | 0.498 |
| 25% | 0.745 | 0.645 |
| 50% | 0.907 | 0.751 |
| 75% | 1.017 | 0.875 |
| 95% | 1.099 | 0.980 |
| max | 1.204 | 1.110 |

| δ | in-ball @100 | in-ball @1000 |
|---|---|---|
| 0.50 | 2.4% | 5.2% |
| 0.60 | 7.9% | 16.8% |
| **0.75** | **25.8%** (11,575 pts) | **50.0%** |
| 0.90 | 48.8% | 80.3% |
| 1.00 | 70.5% | 97.2% |
| 1.10 | 95.1% | ~100% |

Figure + CSV: `results/cifar100/nearest_labeled_distance.{png,csv}`. Three observations:
(1) **correction to our own shorthand** — coverage @δ=0.75 @100 clean is **25.8%**, not
"~20%/10k" (that figure was the noise-.5 cell; the 25.8% matches the §4.5 replay's
`cov_frac` to 4 decimals). (2) The distribution is *tight*: even at budget 100 no pool
point is farther than ~1.2 from a labeled point (median 0.91) — distance concentration in
512-D, the same effect that broke Zelnik-Manor scaling in §7.9; the informative range for
any RBF/NW weighting is essentially d ∈ [0.5, 1.1], and δ steps as small as ±0.15 move
coverage enormously (0.6→0.75→0.9 ≈ 8%→26%→49%) — directly setting the 9.2 δ-grid.
(3) Both histograms show a notch exactly at d=0.75 — the fingerprint of coverage-greedy
selection depleting the just-covered boundary shell.

### 8.3 About our results

**Q1 — Split theory/results into separate files.** ✅ Done 2026-07-12 — this 3-file split
(THEORY / RESULTS / NEXT_STEPS), section numbers preserved.

**Q2 — Add a 1-NN evaluation for every results row, not just probcover.**
Agreed, and it is **fully offline — no GPU jobs needed**: the 1-NN evaluator needs only
(lSet, labels, features, test set), all recoverable per run — the §4.5 replay already
rebuilds exact trajectories (`eval_posterior_quality.py`). Design: a script
`batch_utils/eval_1nn_all_arms.py` that, per (arm, noise, seed, episode), fits 1-NN on
L2-normalized features and reports three label variants: (a) **observed labels** (the
paper convention — this is what "probcover + 1nn" does; for other selection arms it
isolates the *selection* effect under the 1-NN evaluator), (b) **posterior-argmax labels**
(tests our denoising through the 1-NN evaluator — per §4.5 A this should shine at noise
≥ .5), (c) for pseudo arms, 1-NN fitted on **labeled + harvested pseudo points** with
their targets. Adds an "X + 1nn" column family to every §4 table.
**✅ DONE 2026-07-12** — `batch_utils/eval_1nn_all_arms.py`; full tables (all 4 noise
levels, all 10 budgets): `results/cifar100/eval_1nn_all_arms/tables.md` + `per_episode.csv`
(840 cells). Sanity gate passed: the offline replay reproduces the live sweep-31048649
`probcover + 1nn` numbers to ≤0.03 pts. Headlines:
(i) **selection alone under the 1-NN evaluator: bayes_product ≈ probcover** — slightly
behind on clean (37.6 vs 38.1 @100), ahead only from ~500 labels under noise; takeaway 1
now confirmed under a second evaluator.
(ii) **denoise-then-1NN is the best training-free recipe under noise**: 1-NN fitted on
posterior-argmax labels hits **28.0 @1000 noise .8** (raw-label 1-NN: 10.8; the best
*trained* arm: 29.2 — i.e. within ~1 pt with zero training) and 39.3 at .5; on clean it
costs ~1.5 pts (§4.5 A ball impurity, visible through the evaluator).
(iii) **NW κ=3 is the best clean high-budget training-free evaluator** (46.4 @1000 —
above probcover+1nn 45.6 and above the trained-on-observed head 45.1, still below trained
`pseudo all, norm` 48.4); κ=1 wins among NW at noise .8 (18.7) — but both NW variants are
dominated by the simpler post-argmax 1-NN at noise ≥ .5.

---

## 9. New research directions (weekend notes 2026-07-12) — with assessment

### 9.1 Soft-kNN (Nadaraya–Watson) with RBF distance weights as the *classifier* over all points

**Agree, with one important framing fix:** this is *not* the rejected §7.9 experiment.
§7.9 used NW rows as *CE training targets* for far points and failed because far targets
are confidently wrong **as targets** (26% acc fed to a loss that treats them as truth).
Using NW as the **classifier/evaluator itself** (no training, prediction = distance-weighted
vote) is a different animal — and it is already partially measured in the §7.9 replay:
gap-scaled NW (κ=3, m=7) reaches **37.4% all-pool argmax @100 clean ≈ the 1-NN evaluator's
38.1**, and *beats* hard 1-NN under noise (all-pool @1000: 42.3 vs 39.2 at .2; 29.7 vs 25.6
at .5; κ=1 wins at .8). Expectations: NW-as-classifier should ≈ or beat `probcover + 1nn`
everywhere (it generalizes it), win the low-budget/high-noise cells, and still lose to the
trained `pseudo all, norm` head at high budget on clean (48.4 @1000). Because it is
training-free it slots directly into the §8.3 Q2 offline eval as one more evaluator column
(`nw κ∈{1,3}`) — near-zero extra cost. **Measured 2026-07-12 (see §8.3 Q2):** as predicted,
NW κ=3 is the best training-free evaluator on clean high budget (46.4 @1000) and κ=1 the
best NW at noise .8 — but under noise ≥ .5 both lose to the simpler denoise-then-1NN
(posterior-argmax labels), so the NW evaluator's niche is the clean/low-noise regime.

**Clarified intent (Alon, 2026-07-12): this direction is a *training* scheme, not (only)
an evaluator.** Train the linear classifier on **all** pool points, each with a
posterior/NW soft target and a per-sample loss weight $w_u \in [0,1]$ that **decays
smoothly with distance** to the nearest labeled point (RBF-like), replacing the binary
covered/uncovered split. Unified kernel view (agreed): `pseudo_all` = this scheme with
$w$ = tophat($\delta$); the rejected §7.9 arm = tophat + a *constant* far tail
(OOB_WEIGHT); the proposal = smooth decay. **Assessment: worth one more experiment, and
it is NOT what §7.9 rejected** — §7.9 tested constant far weights {1.0, .3, .1} and
concluded far targets need weight ≈ 0; a smooth decay is precisely the scheme that can
keep the (possibly usable) near shell just outside $\delta$ while zeroing the far field.
Three design constraints from existing measurements:
1. **Keep the product posterior inside the ball** — the §7.9 replay already showed an NW
   *mixture* inside the balls loses the count-aggregation that powers the noise-.8
   denoising. On "soft-kNN inside the ball ≈ pseudo_all at low budget": agreed *where a
   ball holds a single labeled anchor* (then product = mixture = $b_\ell$ exactly — the
   common case @100); they diverge once balls hold several anchors, and the product's
   sharpening is the denoising mechanism, so the smooth kernel should apply to the
   *out-of-ball weight*, not replace the in-ball fusion.
2. **The weight must reach ≈0 by the far field**, else it reduces to the rejected
   constant-weight arm.
3. **A raw RBF over $d$ is ill-conditioned here** — §8.2 Q3 measured hard distance
   concentration (median nearest-distance 0.91 @100, entire pool ≤ ~1.2), so the whole
   dynamic range is $d \in (0.75, 1.2)$; parameterize on the gap $(d-\delta)$, or better,
   **calibrate $w(d)$ = measured NW-target accuracy per distance bin** from the replay —
   weight-as-reliability, no free bandwidth parameter.
**Decisive pre-check (offline, launched 2026-07-12):** bin uncovered points by
nearest-labeled distance and measure NW-target accuracy per bin
(`batch_utils/eval_nw_distance_bins.py` → `results/cifar100/eval_nw_distance_bins/`).
If a shell past $\delta$ has target accuracy near the covered-points level, implement the
smooth-weight scheme; if accuracy cliffs immediately at $\delta$, the §7.9 verdict extends
to smooth weights and the direction closes cheaply. **Result (2026-07-12, sanity-gated vs the §7.9 replay to 0.03 pts): the shell EXISTS and
the scheme is worth implementing, scoped to d ≲ 0.95.** Clean @100, NW κ=3 target accuracy
by nearest-labeled distance: (0.75,0.80] → **55.5%** (vs 25.7% uncovered mean, 71.7%
covered), (0.80,0.85] → 48.7, (0.85,0.90] → 38.7, (0.90,0.95] → 29.8, then 20.7 / 12.8 /
6.8 — a smooth ~8–10 pts per 0.05 of d, **no cliff**; accuracy crosses the uncovered mean
at d ≈ 0.92 and floors past ~1.0. The pattern is monotone in all 12 (noise × budget)
tables: first-bin ≈ 0.72–0.77× covered accuracy at noise ≤ .5, 0.53× at .8.
**Mechanistic confirmation of §7.9's "confidently wrong":** the NW top-1 mixture weight is
flat in d (0.40→0.43 clean @100) — the target is exactly as confident at d>1.05 (6.8%
correct) as at d=0.78 (55.5%), so confidence carries zero distance signal and the gate
must come from d itself, which is precisely what the smooth weight does and a constant
OOB_WEIGHT cannot. κ=3 dominates κ=1 in the shell at noise ≤ .2 (55.5 vs 24.2); κ=1
catches up only at noise ≥ .5, budget ≥ 300. **Payoff expectation (honest):** the
exploitable mass is the two nearest bins, ~6.5k pts ≈ 20% of uncovered @100, at 49–56%
target accuracy (clean) — modest gains concentrated at low noise/low budget; at .8 the
shell is 11–14% accurate, so likely nothing beyond §7.9's label-smoothing +0.5..+0.9
there. Implementation sketch: per-point loss weight w(d) = per-bin measured target
reliability (or a fitted smooth decay dying by d≈0.95), κ=3 targets, product posterior
untouched in-ball; then a paired 12-run confirmation sweep vs `pseudo all, norm`.
Data: `results/cifar100/eval_nw_distance_bins/{bins.csv,tables.md}`
(`batch_utils/eval_nw_distance_bins.py`).

**❌ SWEEP VERDICT (2026-07-13, job 31058388, 12/12 COMPLETED — NOT ADOPTED; the
out-of-ball direction is now closed for good.** Implemented as `OOB_WEIGHT_MODE=ramp`
(W0=0.8, DMAX=0.95, far field dropped; commit `6e1dfba`), per-seed paired vs the
normalized `pseudo all` baseline (job 31048960). Paired Δ (mean over seeds, std ≤ 0.2):

| noise | @100 | @300 | @1000 |
|---|---|---|---|
| 0.0 | **−1.35** | −1.21 | −1.11 |
| 0.2 | −0.43 | −0.34 | −0.72 |
| 0.5 | +0.15 | +0.19 | −0.17 |
| 0.8 | +0.16 | +0.29 | **+0.65** |

The decision rule (above) fires on the close branch: clean/.2 are negative *again*,
recovering only ~0.25 of §7.9's −1.6 clean drag despite keeping only the 49–56%-accurate
shell at ≤0.8 weight and zeroing the confidently-wrong far field. **Mechanism, refined:**
the failure was never only far-field garbage — even the *good* shell dilutes on clean,
because the bar for extra targets is not "better than chance" but "comparable to the
71.7%-accurate covered targets the classifier already trains on"; ~11k targets at 30–55%
accuracy are net-negative next to ~9k at 72%. The noise-.8 gain (+0.65 @1000) matches
the §7.9 label-smoothing reading and is not worth a config fork. **Recommended
configuration unchanged: normalized `pseudo all`, covered-only.** If high-noise cells
ever matter for the writeup, the .8-only smoothing gain is the one thing to revisit.
Collector: `batch_utils/collect_nw_wdist.py` →
`results/bayes_pseudo/nw_wdist_{episodes,deltas}.csv`, `fig_nw_wdist_delta.png`. Also note the *in-ball* RBF-weighted **product
exponents** (§7.9 replay, 87→96.7 denoising @100 clean) remain the highest-value related
follow-up on the *training* side.

### 9.2 δ sweep

**Agree.** δ=0.75 was inherited from the ProbCover paper's purity criterion and never
re-examined for this method — and the mechanisms that dominate our results (ball impurity
≈ 32% at δ=0.75 driving §4.5 A's clean-data relabel damage; coverage volume driving
`pseudo all`) pull in *opposite* directions, so the optimum is plausibly noise-dependent.
Two-stage plan: (1) **offline first** — rerun the §4.5 replay at δ ∈ {0.5, 0.6, 0.75, 0.9,
1.1} (denoising, pseudo-label quality, coverage, calibration per δ — no GPU); the §8.2 Q3
distance distribution tells us which δ values meaningfully change coverage. Caveat: replay
holds the *trajectory* fixed while δ also shapes selection, so (2) confirm the offline
winner with a small paired training sweep (~24 runs) before believing it.

**Stage 1 DONE (2026-07-22)** — `batch_utils/eval_delta_replay.py`, δ ∈ {0.5, 0.6, 0.75,
0.9, 1.1} × 4 noise × 3 seeds × budgets 100–1000, replaying the recorded bayes_product
trajectories. Sanity gate: at δ=0.75 the LOO keep counts match the live runs' logged
counts 10/10 budgets in all 12 cells (exact one-at-a-time fold order preserved — block
matmul deviated on 12/1000 near-tie argmaxes and was rejected). Results in
`results/cifar100/delta_replay/`. Headlines (B=1000 means over seeds):

- **The two mechanisms confirmed to pull opposite ways, so the optimum is
  noise-dependent.** Purity metrics are monotone in −δ everywhere: covered-point
  accuracy on clean 83.9% @δ.5 → 66.7% @δ.75 → 26.2% @δ1.1; ECE degrades monotonically
  with δ at every noise (posterior overconfidence = ball impurity).
- **Denoising and noise-ID want δ≈0.75 once noise ≥ .5.** Labeled-plane post_acc at
  n.8: 24.5% @δ.5 vs **61.6% @δ.75** (tiny balls can't cross-vote, the posterior just
  parrots the noisy label); LOO noise-ID AUC peaks at δ=0.75 at *every* noise level
  (n.8: .915 @.75 vs .684 @.5, .882 @1.1). δ=0.75 was never re-examined but it sits at
  the denoising/noise-ID optimum — inherited luck.
- **Coverage volume collapses below 0.75**: cov 5.1%/16.7%/49.5%/79.9%/100% @1000
  (clean). At δ=0.5 the pseudo-all diet would be ~10× smaller. Correct-covered *mass*
  (cov×acc, clean) is 4.3/13.1/33.0/41.1/26.2% — δ=0.9 adds correct mass but nearly
  doubles the wrong mass (38.8% vs 16.5%), the exact §4.5-A damage regime.
- Top-k-per-class pseudo precision @1000 peaks at δ=0.75 for n.2/.5 (81.9/70.7%), at
  δ≤0.6 on clean, and δ=0.6–0.75 at n.8.

**Reading:** for the noisy regimes (the method's home turf) δ=0.75 is defensible and
plausibly optimal; the interesting bet is **clean/low-noise, where δ≈0.6 trades 3×
less coverage for +12 pts purity** — whether that wins in training depends on the
volume-vs-purity elasticity of the classifier, which replay cannot resolve.
**Stage 2 DONE (job 31108061, 24/24, collected 2026-07-22) — δ=0.75 CONFIRMED, §9.2
CLOSED.** Paired training sweep, champion arm (bayes_product + pseudo all + norm
linear), δ ∈ {0.6, 0.9} × 4 noise × 3 seeds vs the δ=0.75 baseline (31048960, same
seeds). Here δ shapes *selection* too, unlike the replay. Result: **δ=0.75 wins at
every noise level and every budget.** Test acc @1000, δ .6/.75/.9:
clean 47.9/48.4/40.3 · n.2 45.0/46.6/40.0 · n.5 38.9/42.6/36.6 · n.8 24.4/29.2/24.8.
The replay's clean-data purity bet does NOT pay in training — δ=0.6's +12 pts
per-point purity never overcomes its 3× coverage loss (clean −3.7 @100, narrowing to
−0.5 @1000: coverage catches up with budget but never crosses), and δ=0.9's extra
covered mass is poisoned by impurity (−5..−8 clean, −6.6 @1000 n.2), exactly the
§4.5-A damage regime. Reading: end-to-end, training-diet *volume* dominates
per-target purity at low coverage, and impurity dominates once coverage is cheap —
δ=0.75 sits at the crossing at every noise level, matching its replay-side
denoising/noise-ID optimum. The inherited default survives a proper examination;
no δ retune recommended for CIFAR100/SimCLR. (New datasets/embeddings should still
re-run `eval_delta_replay.py` — the crossing is geometry-dependent.)
Rows live in `results_master.csv` under sweep 31108061 (new `delta` column,
default 0.75 for all pre-§9.2 rows; ROW_SPEC renderers unaffected).

### 9.3 Use the classifier to estimate the posterior (feedback loop)

**Agree with guardrails — this is the direction with the biggest upside and the biggest
trap.** Upside: §4.5 B showed the two objects know different things — the classifier
generalizes past the δ-balls (46.5 vs posterior's 33.2 all-pool clean @1000), while under
heavy noise the posterior knows more than the classifier extracts (37.5 vs 27.5 on covered
@ .8). Closing the loop (classifier predictions → extra evidence in the posterior → better
targets → retrain) is EM/co-training. The trap is **circularity** — the exact critique we
levy at AUM (a judge trained on the labels it judges). Recommended shape: fold classifier
predictions in as **one additional tempered expert** (β_clf < 1, §1.4 machinery — never a
full vote), and/or restrict it to **uncovered points only**, where the graph has literally
nothing — that replaces §7.9's failed far-field NW targets with a signal that actually
generalizes far. Validate on the §4.5 replay (posterior quality with/without the classifier
expert) before any training sweep.

### 9.4 Estimate the noise rate instead of assuming it known

**Strongly agree — this is the method's biggest realism gap** (we currently hand the true
annotator accuracy to the confusion matrix). It unifies three existing threads: the §7.1
noise gate (blocked on exactly this statistic), §4.5 D (the raw LOO-disagreement fraction
saturates at the ~53% impurity floor — the estimator must be impurity-corrected), and
Netta's q̂ (§8.1 Q3 — implemented, printed, never scored). Concrete offline bake-off, no
GPU: on the §4.5 replay trajectories, score against the realized noise rate:
(a) impurity-corrected LOO disagreement (calibrate the clean-data floor at δ, subtract);
(b) **Dawid–Skene-style EM**: alternate posterior ↔ confusion-matrix re-estimate — the
natural Bayesian answer, and it estimates the full matrix (asymmetric noise for free);
(c) Netta's q̂ parsed from the 27 NPC-repro logs. Winner gets wired as the §7.1
`TRAIN_LABEL_MODE=auto` gate *and* replaces the oracle confusion matrix in Step A.

#### ✅ Bake-off DONE 2026-07-14 — MEASUREMENT ONLY (production code untouched, nothing wired)

> **Plain-language summary (read this first if you're revisiting cold).** The pipeline is
> currently *told* the annotator's noise rate. This experiment asks: **if we weren't told,
> how well could we guess it?** We took the 12 already-finished `observed` runs
> (noise 0/.2/.5/.8 × 3 seeds), and for each labeled point looked at what its neighbours'
> labels imply, then estimated an overall noise rate four ways. We compared each estimate
> to the *true* fraction of wrong labels in that run. **No training was run and no method
> was plugged into the pipeline — this only measures which estimator is accurate.** Decision
> on which one (if any) to actually adopt is deferred to a future session.

Script `batch_utils/eval_noise_rate_estimators.py` (standalone, replays the same
trajectories as `eval_posterior_quality.py`; sanity: labels regenerate exactly on all 12
cells, realized noise ≈ nominal). Outputs (in gitignored `results/`, on disk only):
`results/cifar100/noise_rate_estimators/{noise_rate_estimators.json,.csv,fig_noise_rate_estimators.png}`.

What the four estimators are, and how each did (mean over seeds; "MAE" = mean absolute
error vs the realized noise rate, over all 4 noise levels × 3 seeds):

| estimator | what it does | result | MAE @B1000 |
|---|---|---|---|
| **raw LOO** | fraction of labeled points whose in-ball neighbour *majority* label ≠ their own label (leave-one-out) | ❌ **saturates** — reads 0.47–0.59 even at true noise 0 (the ball-impurity floor of §4.5 D); can't be used as an absolute rate | ~0.30 |
| **impurity-corrected LOO** | same, then subtract the clean-data floor `f0` and rescale: `q̂=clip((d−f0)/(1−f0))` | ✅ **most accurate** — @B1000 reads 0.01 / 0.19 / 0.49 / 0.81 vs true 0 / 0.19 / 0.50 / 0.79. **Caveat:** `f0` is the impurity floor, which needs a *clean reference set* to calibrate — so this is **not** assumption-free | **0.016** |
| **Dawid–Skene EM** | self-consistent symmetric-noise EM: alternate (posterior over true labels from neighbours) ↔ (re-estimate the noise rate); no clean data used | ✅ **best without any clean anchor** — monotone and cleanly **separates every level including 0-vs-0.2 at every budget** (+0.09–0.10), but carries a roughly constant low-noise over-estimate (~+0.25–0.35: reads ~0.35 at true 0) because impurity leaks in | ~0.19 |
| **soft LOO (true M)** ✱ | reference only — uses the *true* confusion matrix (i.e. is told the rate) | reads 0.53–0.83 at true noise 0 — proves the disagreement family's impurity ceiling persists **even when the rate is known**, so impurity, not ignorance of the rate, is the bottleneck | ~0.33 |
| **Netta's q̂** | `1 − kept/total` from the AUM train-on-clean filter, parsed from the 27 NPC-repro logs (job 30811243) | ❌ **severe low-budget under-estimate** — reads 0.08–0.14 at budget 100 regardless of true noise (AUM needs data); only usable at high budget + mid noise (0.49 vs 0.51 @noise .5, B1000), still low at .8 (0.59 vs 0.79) | n/a (diff. runs) |

**Provisional takeaway (to revisit before adopting anything):**
- For the **§7.1 `auto` gate** (only needs to *tell clean from noisy*, not a precise number):
  **Dawid–Skene EM** is the candidate — self-contained, separates clean from 0.2 at every budget.
- For **replacing the oracle confusion matrix in Step A** (needs an accurate number):
  **impurity-corrected LOO** at higher budgets, *provided* a clean anchor exists to set `f0`;
  otherwise DS-EM with a low-noise bias correction.
- **raw LOO** and **Netta's q̂** are both rejected as absolute estimators (saturation / low-budget
  under-estimate).
- Deeper open question surfaced by the soft-LOO reference: ball **impurity** (~32–53% at δ=0.75),
  not lack of knowledge of the rate, is what caps every neighbour-disagreement estimator — so a
  future attempt should either shrink δ (couples to 9.2) or down-weight far neighbours in the
  disagreement count. **No estimator adopted yet; decision deferred.**

### 9.5 (Idea) Annotator choice by expected posterior gain — unifying NoAA with Bayesian coverage

**Agree — this is a natural and possibly paper-grade unification.** Tomer's NoAA picks the
annotator via a ν̂ threshold binary-searched on proxy data (KMeans clustering on the
embedding — the exact proxy that saturates at B=3000, see the fig5.5 memory). In our
framework the same decision needs no proxy: for candidate point v and annotator a, the
**expected reduction in ball uncertainty mass** is computable *before* annotating —
E over ỹ ~ current predictive of [uncertainty(ball of v) after folding (ỹ, M^a)] — because
folding is one vector-add (§1's Step A–C). The acquisition becomes
argmax_{v,a} expected-gain(v,a) / cost(a): "which point" and "which annotator/oracle" in
one objective, with Tomer's setting the special case a ∈ {noisy, oracle}. First step is a
design section + offline simulation on replay trajectories (compare against NoAA's
threshold rule on the fig5.5 setup); no training runs needed until the simulation says it
wins. Related: with 9.4's estimated confusion matrices this also handles *unknown*
annotator quality — the full "who to ask" problem.

### 9.6 Compare to Itai's formalism (theory side-by-side)

The BAYES_MISP port (RESULTS §4.7) settled the *empirical* comparison: with the shared
linear-head protocol, `bayes_misp` ≈ probcover ≈ `bayes_product` at every noise level.
The open work is theoretical: map his pseudo-count accumulation + diff-max acquisition
onto our posterior/uncertainty language (his $f_1$ ceiling is already identified with our
fixed additive acquisition for tophat kernels, RESULTS note above §4.1), state exactly
when the two selection rules pick different points, and check whether either formalism
*predicts* the high-budget product-vs-additive gap. Deliverable: a THEORY section, no runs.

### 9.7 MLP classifier head — ✅ ANSWERED (head adopted for observed labels; pseudo-stack ❌ does NOT stack — wdfix sweep 31088628, 2026-07-16/22)

Itai's sklearn MLP head (hidden 256) is in the repo (`--eval_model_type from_mlp_sklearn`)
**with best-val epoch selection** (cold clone per episode + per-epoch `partial_fit`,
checkpointed on the clean valSet — RESULTS §4.7, DELTA_ROOTCAUSE pitfall 3). Status
2026-07-16: under this protocol the MLP head **leads the (warm) linear probe at every
noise level and budget** on observed labels (@1000: 46.8/42.5/35.0/17.2 vs
46.7/41.8/31.2/13.6 at noise 0/.2/.5/.8). The earlier "MLP memorizes noise" caveat is
retracted — it was a final-iterate protocol artifact; early stopping recovers +5..+10
under noise. Protocol rules learned: keep the MLP **cold** (warm-start costs it ~4 pts —
plasticity loss) and never compare heads without matching init + iterate-selection.

**What remains open — the actual §9.7 bet:** stack the MLP head with the posterior
training arms. `pseudo all` + linear (norm) is the study's best training arm (49.0 clean
@1000); MLP + observed already does 46.8 with zero posterior machinery. MLP ×
{soft_posterior, pseudo all} needs the torch twin (`from_mlp_torch`, MLPHeadNet in
builders.py) since sklearn's `.fit` can't take soft-target rows — arms are wired but
unswept. If the two gains are even partially additive this sets new bests at every
noise level.

**VERDICT (wdfix sweep 31088628, collected 2026-07-22): the gains do NOT stack.**
First torch-twin sweep (31085854) was invalid — sklearn `alpha=3e-2` was mapped 1:1
onto Adam `weight_decay`, ~200× too strong (sklearn divides the penalty by batch
size), so the head underfit and *degraded* with budget. With `WEIGHT_DECAY=1.5e-4`
the arch-parity gate passes (torch 34.5→44.9 vs sklearn 34.9→46.8 over budgets
100→1000, probcover/clean/observed; residual −1..−1.9 at high budget — torch is a
slightly conservative stand-in). On `pseudo all`, MLP-torch trails the warm-norm
linear at **every** noise and budget: clean −0.9..−1.4, noise .2 ≈ −3, noise .5 ≈ −5,
noise .8 ≈ −5 — the deficit *grows* with noise, so it is not the parity residual.
Reading: the pseudo-all posterior targets already linearize the problem (they are
built from the feature-space ball structure), so the MLP's extra capacity has nothing
left to model and only adds variance under noise. **Recommended config unchanged:
warm-norm linear + `pseudo all` (48.4±0.2 @1000 clean). MLP head (sklearn, cold+
best-val) remains the best *observed-label* head.** Full tables:
`classifier_model_comparison.md`.

**Norm-control (sweep 31116164, 2026-07-23):** the wdfix MLP+pseudo arm trained on
*raw* features while the linear champion trains on *normalized* ones — a possible
confound for the clean-data margin. Rerun of the exact soft-CE arm with
`--normalize_features`: normalization helps the MLP on clean (deficit −0.9..−1.4
→ −0.2..−1.0; 47.4 vs 48.4 @1000) and slightly at noise .2 (≈−2.2 vs ≈−2.6), is
neutral at noise .5 and marginally *negative* at noise .8. Even with matched
normalized inputs the linear head wins every noise×budget cell, so the
"does not stack" verdict stands confound-free.

### 9.8 KL loss instead of CE — ❌ ANSWERED, NOT ADOPTED (sweep 31088628, 2026-07-22)

The soft-target arms (`soft_posterior`, pseudo) train with soft cross-entropy. With
*fixed* targets, forward KL(target‖model) = soft-CE minus the targets' entropy — the
gradients are identical — so the variant with actual content is **reverse**
KL(model‖target): mode-seeking, it lets the classifier commit to a single class inside a
spread-out posterior instead of averaging over it. Possibly a better fit for impure balls
at high noise. Cheap ablation: one loss-flag, re-run the soft/pseudo arms at one noise
level first.

**VERDICT (`--soft_loss rkl`, both heads, all noise levels): no win anywhere.** On the
linear head rKL ties soft-CE under noise (noise .5 ±0.1; noise .8 +0.2, within seed
noise) and **loses on clean** (−0.7 @300 → −1.7 @1000 — mode-seeking commits early
and can't use the calibrated spread). On the MLP head it is uniformly slightly worse
(−0.1..−0.6 below the same head with soft-CE). The impure-ball hypothesis didn't pay:
where posteriors are most spread (noise .8) rKL is exactly neutral. **Soft-CE stands
for all soft-target arms.** Tables: `classifier_model_comparison.md`.

### Proposed execution order (all offline/CPU first)

1. **§8.3 Q2 + 9.1** — `eval_1nn_all_arms.py`: 1-NN + NW evaluator columns for every §4
   row. **✅ DONE 2026-07-12** — see §8.3 Q2 headlines; denoise-then-1NN ≈ best trained
   arm at noise .8 with zero training.
2. **9.4** — noise-rate estimator bake-off on the replay (+ parse Netta's q̂ from logs);
   unblocks §7.1 and the honest "unknown noise" story. **✅ DONE 2026-07-14 (measurement
   only, nothing adopted) — see the §9.4 "Bake-off DONE" block.**
3. **9.2 stage 1** — δ replay sweep (offline).
4. **In-ball RBF product exponents** (from §7.9 replay) — 12-run paired training sweep;
   the one *training* change with a measured large win (87→96.7 denoising).
5. **9.3** — classifier-as-expert on the replay, then 9.5 design doc.
