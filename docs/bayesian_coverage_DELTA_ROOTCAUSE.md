# ProbCover + 1-NN — why 38.1 @100 (vs paper ~34-35), and the δ pitfall

_Investigation 2026-07-15. Offline replica: `batch_utils/pc1nn_delta_replay.py`
+ `batch_utils/pc1nn_delta_decompose.py` (run: `python batch_utils/pc1nn_delta_decompose.py`).
This is a running log of **result-interpretation pitfalls** in the bayes-coverage study,
so future readers don't re-chase them. Pitfall 1 (the bulk of this doc): **ProbCover's δ
is a per-dataset, per-embedding purity parameter, not a constant** — the source of the
"our number is too high" question. Pitfall 2 (see "Related pitfall" below): **feature
L2-normalization flips sign with training convergence** — helps @1000, hurts @100. Common
thread: a single knob whose effect depends on regime, so a number read at the wrong regime
misleads._

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
