# ProbCover + 1-NN — why 38.1 @100 (vs paper ~34-35), and the δ pitfall

_Investigation 2026-07-15. Offline replica: `batch_utils/pc1nn_delta_replay.py`
+ `batch_utils/pc1nn_delta_decompose.py` (run: `python batch_utils/pc1nn_delta_decompose.py`).
This is a running log of **result-interpretation pitfalls** in the bayes-coverage study,
so future readers don't re-chase them. Pitfall 1 (the bulk of this doc): **ProbCover's δ
is a per-dataset, per-embedding purity parameter, not a constant** — the source of the
"our number is too high" question. Pitfall 2 (see "Related pitfall" below): **feature
L2-normalization flips sign with training convergence** — helps @1000, hurts @100.
Pitfall 3 (added 2026-07-16): **the two evaluator heads run different training
protocols** — the linear probe warm-starts across episodes while the sklearn MLP
restarts cold and keeps its final (overfit) iterate — which manufactured a fake
"MLP better early, worse late" crossover. Common thread: a single knob whose effect
depends on regime (or differs silently between arms), so a number read at the wrong
regime misleads._

## The question

The §4 tables report **probcover + 1nn = 38.1 @100 labels, CIFAR-100 clean**
(sweep 31048649, seeds 1/2/3 = 38.44/37.68/38.14). The ProbCover paper reports
~34-35% for the same framework (1-NN over SimCLR features). Why are we ~3-4 pts
higher? Is it the embedding? A seed outlier? A bug?

## Method — a replica you can trust

`pc1nn_replay.py` reproduces the *exact* daphna_lab pipeline offline:

1. **Split** (exact `makeLUVSets`): `np.random.seed(seed)` → `shuffle(range(50000))`
   → `uSet = first 45000`, `valSet = last 5000` (INIT_L_RATIO=0.0 ⇒ empty lSet).
2. **Select**: ProbCover greedy over L2-normalized features, tophat δ, max-degree
   `argmax` (lowest index on ties), 100 samples from uSet.
3. **Eval**: 1-NN fit on the 100 selected (L2-normalized train feats + clean
   labels), predict the **10k CIFAR-100 test set** (L2-normalized). test_acc.

**Validation:** the replica reproduces the sweep to the decimal —
38.44 / 37.68 / 38.14. So the single-factor toggles below are trustworthy.

## Result — the gap is ENTIRELY δ. Nothing else.

Controlled decomposition, one factor changed at a time (mean ± std over seeds 1-3):

| factor | variant | acc @100 |
|---|---|---|
| **baseline (ours)** | itai, δ=0.75, argmax | **38.09 ± 0.31** |
| selection rule | argmax (ours) | 38.09 ± 0.31 |
| | random top-5 (paper-comment) | 38.08 ± 0.28 |
| embedding | itai (the one used) | 38.09 ± 0.31 |
| | alon | 38.91 ± 0.21 |
| | typiclust | *(no valid 10k test embedding — see note)* |

**δ sweep (itai, argmax, mean over seeds 1-3):**

| δ | acc | | δ | acc |
|---|---|---|---|---|
| 0.40 | 23.0 | | 0.70 | 37.8 |
| 0.45 | 26.3 | | **0.75** | **38.1 ← ours (peak)** |
| **0.50** | **28.6 ← purity-correct δ** | | 0.80 | 37.5 |
| 0.55 | 32.3 | | 0.85 | 36.2 |
| **0.60** | **34.7 ← paper's ~34-35 lands here** | | 0.90 | 30.8 |
| 0.65 | 36.1 | | | |

## Root cause

**The `BAYESCOV_*.yaml` configs hardcode `initial_delta: 0.75` — the value tuned
for CIFAR-10 — and apply it to CIFAR-100.** For this CIFAR-100 SimCLR embedding the
accuracy-vs-δ curve *peaks exactly at 0.75* (38.1). The paper uses a smaller,
purity-derived δ for CIFAR-100 and lands on the rising flank (~34-35 ≈ δ 0.60).

The repo's own purity criterion confirms 0.75 is too big.
`select_delta.py` (ProbCover's α=0.95 method + knee-point) returns **δ = 0.50** for
this embedding. Ball purity vs δ (KMeans-100 pseudo-labels):

| δ | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | **0.75** | 0.80 |
|---|---|---|---|---|---|---|---|
| purity | 0.962 | 0.930 | 0.867 | 0.765 | 0.606 | **0.391** | 0.178 |

At δ=0.75 the balls are only **39% pure** — ProbCover's core coverage assumption
(each δ-ball ≈ one class) is *violated*.

## Ruled out (with evidence)

- **Embedding.** `features_tag: None` ⇒ Itai's default SimCLR
  (`representations_bank/cifar-100_simclr/cifar100_simclr_train.npy`). At δ=0.75,
  itai 38.09 ≈ alon 38.91. Full-label 1-NN quality (40k→10k, true labels) is
  equal across all three candidates: **itai 56.13, TypiclustRepo 56.99, alon 56.52**
  — TypiClust's own embedding is if anything marginally *best*, so "we had a
  stronger embedding" is false.
- **Selection determinism.** Our `argmax` (38.09) ≈ the commented-out
  paper-style `np.random.choice(top-5)` (38.08). No effect.
- **Seed / outlier.** 38.44 / 37.68 / 38.14 — spread 0.76 pt, no outlier.

## The pitfall to remember (why this matters beyond this one number)

1. **δ is not a constant.** It is a per-dataset *and* per-embedding purity radius.
   The CIFAR-10 value (0.75) is ~1.5× too large for CIFAR-100 SimCLR. **Always
   derive δ from the purity criterion (`select_delta`, α≈0.95) per embedding**, or
   at least verify ball purity before quoting a fixed δ.
2. **1-NN accuracy can *peak beyond* the purity-correct δ.** Counter-intuitively,
   larger δ selects points in the *densest* regions (highest out-degree = most
   *typical* points), making ProbCover behave like TypiClust. At a 1-label-per-class
   budget, typical/central points are much better 1-NN prototypes than the boundary
   points a purity-correct δ=0.50 spreads to (28.6). So a *higher* accuracy at a
   *wrong* δ is expected, not a bug — but the coverage guarantee that motivates
   ProbCover no longer holds there. **A good number at δ=0.75 is the right answer
   for a lucky reason.**
3. **This is study-wide.** δ=0.75 is used for *every* CIFAR-100 ProbCover arm in
   §4 (not just 1-NN). Any absolute ProbCover number in the bayes study is on the
   0.75 curve, past the purity radius. Cross-paper comparison needs this caveat.
4. **When a result beats the literature, decompose before celebrating.** Replicate
   the pipeline offline, confirm it matches to the decimal, then toggle one knob at
   a time. Here it took four toggles to prove three suspects innocent and convict δ.

## Related pitfall: feature L2-normalization flips sign with convergence

_(Companion evidence, sweep 31048960 — full tables in `bayesian_coverage_NORMALIZATION.md`.
Same lesson as above: a single knob whose effect depends on regime, so a number read at
the wrong regime lies.)_

**Setup.** The `from_features` linear head historically trained on **raw** SimCLR features
while the 1-NN eval arm L2-normalizes. Adding `--normalize_features` (scale-preserving
`x/‖x‖ × mean_train_norm`, train + test) isolates the training-input change; the AL
selection side was always normalized, so lSet trajectories pair per-seed.

**Finding — the effect of normalization is not fixed; it flips sign with the training
budget.** Δ = (normalized − raw) test acc, positive = normalization helps:

| budget | regime | Δ (typical arms) | reading |
|---|---|---|---|
| **@100** | far from convergence | **−5 to −15** (bayes_product 24.6→11.7 = −12.9; additive −10.9; probcover −10.6) | normalization **hurts badly** |
| @300 | crossover | small, mixed (−2 to +2) | roughly neutral |
| **@1000** | converged | **+2 to +7** (observed +6.8 @noise.5; product +6.8; +2.4 clean) | normalization **helps almost everywhere** |

**The tell that this is a convergence artifact, not a real "normalization is bad at low
budget":** the one arm that trains on ~10k pseudo-labeled points *every* round —
`pseudo_all` — is **flat-to-positive at ALL budgets** (+0.7 @100 clean, +0.9 @1000). Only
the arms that train on the tiny ~100-point labeled set are hurt at @100.

**Mechanism.** At fixed `BASE_LR`/200 epochs, tiny-budget training **underfits** isotropic
(unit-norm) inputs — smoke test: train acc ~50% (norm) vs ~95% (raw) @100 — while high-norm
raw features produce larger logits and fit fast. Per-sample feature norms act as an implicit
"train me first" weight that a fixed-LR schedule leans on when data is scarce. Once there's
enough data to converge (@1000), that crutch is unnecessary and normalization's true benefit
(isotropic geometry, no norm-as-sample-weight bias) dominates. The gross collapse from plain
unit-L2 (3.3 @100) was fixed by the scale-preserving rescale; this residual @100 penalty was
not, and it is a **training/optimization** artifact, not a property of the representation.

**Lessons to carry forward:**
1. **Judge an intervention only in the converged regime** (here @1000), or after an
   LR/epoch retune. A pre-convergence number measures the optimizer, not the method.
2. **A regime-invariant control arm is the cleanest diagnostic.** `pseudo_all` (always
   ~10k training points) stayed flat and pinned the @100 losses on underfitting — without
   it you might wrongly conclude "normalization degrades low-budget accuracy."
3. **The bayes §4 conclusions were re-checked @1000 and survive** (product ≫ additive;
   pseudo best training arm; posterior-training still beats observed under noise, though the
   margin shrinks — part of what posteriors "fixed" was feature-norm pathology, not only
   label noise). Absolute @1000 numbers are understated by +2-7 under a raw-feature convention.

## Related pitfall 3: warm-start protocol asymmetry between evaluator heads

_(Investigation 2026-07-16, triggered by the BAYES_MISP port's MLP-vs-linear rows
in R2. Diagnostics: session replays on the exact seed-1 lSets — the finding is
log-verified against the production runs, see below. §4.7 of
`bayesian_coverage_RESULTS.md` carries the same decomposition table.)_

**The question.** In R2 (clean), the sklearn-MLP head (`from_mlp_sklearn`, Itai's
evaluator) beats the linear probe by ~+10 @100 labels but *loses* to it @1000
(45.4 vs 46.7). "High-capacity model wins with FEW examples, loses with many" is
backwards — bug?

**Root cause — the two heads never ran the same training protocol.**
`ACTIVE_LEARNING.FINE_TUNE` defaults to **True** (in our repo *and* in Itai's
bayes_al_2025), so:

- the **linear probe is warm-started**: each episode resumes from the previous
  episode's weights. By 1000 labels it has accumulated ~2000 epochs. Log-verified:
  in the production run, episode 2 (300 labels) sits at val 37.2% after just 10
  epochs — a cold model needs ~200 epochs to reach that.
- the **sklearn MLP is rebuilt cold every episode** (explicit in the
  `from_mlp_sklearn` code path, ours and Itai's) and `.fit()` returns the **final**
  iterate — but a cold MLP peaks at ~30-60 adam epochs and then decays
  (overfitting; under label noise, memorization).

**Decomposition** (offline replay, identical seed-1 lSets, the run's real 5000-pt
valSet, clean):

| protocol | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| linear, cold, final epoch | 25.2 | 35.8 | 40.7 | 41.8 | 44.9 |
| linear, **warm** ← what the tables report | 25.6 | 40.0 | 43.6 | 44.4 | 47.1 |
| MLP, cold, **final** ← what the tables report | 36.1 | 39.9 | 41.8 | 43.4 | 44.9 |
| MLP, cold, best-val epoch | 36.0 | 40.0 | 43.3 | 43.8 | 46.7 |
| MLP, warm, best-val | 36.0 | 39.9 | 41.3 | 41.3 | 42.4 |

**Findings.**

1. **The "MLP worse late" half of the crossover is protocol, not architecture.**
   Same protocol (both cold, final iterate): MLP ≥ linear at *every* budget clean.
   The published gap @1000 = linear's warm-start gain (+2.2) plus the MLP's missing
   best-iterate selection (−1.8). Checkpoint selection itself adds ≈0 for the
   linear head on clean data (val is monotone) — warm-start is the active knob.
2. **Warm-start is a real gain for the linear probe (+2-4 pts, biggest @300)** —
   it is effectively extra optimization time; the cold probe's fixed 200-epoch cos
   schedule is under-converged at mid budgets (@300 trajectory 27.5/33.9/35.9 at
   epochs 50/100/200, still rising). This compounds with Pitfall 2: both are
   "the probe measures the optimizer, not the method" artifacts.
3. **Warm-start HURTS the MLP** (42.4 vs 46.7 @1000, even with best-val selection)
   — the known warm-start plasticity-loss effect in neural nets. So "run everything
   warm" is not the fix; the heads want different protocols.
4. **The "MLP better early" half is REAL.** Episode 0 is cold for both heads, and
   the MLP's +10 @100 survives every control: converged linear models (lbfgs
   logistic regression 27.5, adam-linear 28.9) stay ~8 below MLP 36.1; a frozen
   random 512→256 relu lift collapses to 11.4 (so it's *learned* nonlinearity, not
   random features); 1-NN's 38.1 @100 shows the same low-n local structure. At 100
   pts / 100 classes the SimCLR classes are locally, nonlinearly structured, and no
   linear boundary from 100 points captures it.
5. **The noise tables (R3-R5) carry the same confound.** The large linear-over-MLP
   margins there (e.g. 31.2 vs 24.6 @noise .5 @1000) mix real noise-robustness with
   the protocol asymmetry; they should not be read as "MLP memorizes noise" until
   the protocol-matched rerun lands.

**Resolution (protocol decision, 2026-07-16, with Alon):** keep the linear head
warm (established study protocol, genuine gain); keep the MLP cold but add
**best-val epoch selection** against the same clean valSet the probe checkpoints
on — implemented in `train_al.py` (`from_mlp_sklearn` now trains via per-epoch
`partial_fit` + best-val snapshot; deliberately departs from Itai's final-iterate
protocol). MLP noise rows rerun as sweep **31085802**. No cold-start production
runs needed — the decomposition above already quantifies them.

**Postscript — the pitfall immediately bit its own fix.** The first best-val
rerun (sweep 31085570, superseded) came back *worse* than final-iterate @1000
clean (42.9 vs 45.4) — matching the warm-MLP replay row above (42.4) almost
exactly. Cause: the main loop keeps the same model object across episodes when
`FINE_TUNE=True`; sklearn's `.fit()` masked that by re-initializing on every
call, but `partial_fit` **continues from existing weights**, so switching the
implementation silently switched the MLP from cold to warm. Fixed with
`sklearn.base.clone(model)` at the top of the branch (cold clone per episode;
regression test: re-feeding a fitted model reproduces identical results).
Corollary to lesson 1: *the warm/cold property can hide in the fitting API, not
just in the loop* — `.fit()` is cold-by-contract, `partial_fit` is warm-by-contract.

**Outcome (rerun v2, sweep 31085802, 12/12):** the fixed cold + best-val MLP
lands exactly where the replay predicted (46.8 vs predicted 46.7 @1000 clean)
and **erases the "MLP collapses under noise" reading** — recovery vs
final-iterate grows with noise (+1.4 clean, +5.4 @.2, +10.4 @.5, +7.3 @.8 at
1000 labels), i.e. the collapse was the final-iterate protocol measuring
memorization. With each head at its own best protocol the best-val MLP now leads
the warm linear probe at every noise level (42.5 vs 41.8 @.2; 35.0 vs 31.2 @.5;
17.2 vs 13.6 @.8 @1000). Early stopping on a clean valSet = textbook
noise-robustness, quantified.

**Lessons to carry forward.**

1. **Before comparing two "heads", diff their full training protocols**, not just
   architectures: init policy (warm/cold), iterate selection (final/best-val),
   schedule. Here two silent defaults manufactured a crossover that survived two
   wrong explanations (checkpointing; "MLP memorizes noise") before the log
   trajectory exposed the warm start.
2. **`FINE_TUNE=True` is a study-wide property**: every linear-head row in R2-R5
   (all sweeps, all arms) is warm-started cumulative training. Fine — but any new
   evaluator added to the tables must either match it or be flagged.
3. **A trajectory in the production log beats a replica.** The decisive evidence
   was episode 2 starting at val 37.2 after 10 epochs — visible in the sweep's
   own stdout, no rerun needed.

## Note on the TypiClust embedding

`CIFAR100_TypiclustRepo` (`.../downloaded_from_typiclust/features_seed1.npy`) is
train-only (50000×512). The sibling `scan/.../cifar-100_dowloaded/pretext/
test_features_seed1.npy` is *also* 50000 rows — not the 10k CIFAR test set — so it
cannot be pushed through the full pipeline. The full-label 1-NN parity above
(TypiClust 56.99 ≈ itai 56.13) covers the embedding-quality question without it.

## Open decision

- **(a)** Rerun the probcover+1nn (and ideally all probcover) arm at the
  purity-correct **δ=0.50** (and 0.60) so the tables can quote a principled δ; or
- **(b)** Keep δ=0.75 as a deliberate study-wide constant, add a provenance
  footnote (δ=0.75 is the CIFAR-10 radius; CIFAR-100 purity radius is 0.50;
  accuracy peaks at 0.75), and report the δ-curve above.

Recommendation: (b) + publish the δ-curve — 38.1 is a legitimate best-case, but it
must be reported honestly as a δ-peak, not a coverage result.
