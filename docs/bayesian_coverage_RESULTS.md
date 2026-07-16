---
title: "Bayesian Coverage — Results vs. Netta's NPC/AUM (CIFAR100)"
date: 2026-07-12
---

**Part 2/3 of the bayesian-coverage summary** (split 2026-07-12 from the former single
`bayesian_coverage_SUMMARY.md`): theory: `bayesian_coverage_THEORY.md` ·
**results (this file)** · status, open questions & next steps:
`bayesian_coverage_NEXT_STEPS.md`. Section numbering is preserved from the original
file — §4–§6 live here, §1–§3 in THEORY, §7+ in NEXT_STEPS; cross-references keep the
original § numbers, so e.g. "§1.4" means THEORY §1.4 and "7.4" means NEXT_STEPS §7.4.

**Branch:** `bayesian_coverage`
Everything in this file is verified against the actual `result.json` files on disk.

**How to cite this document.** Section numbers keep the original §4–§6 (see the note above),
but **every table has a stable short label `R1`…`R9`** (R = Results) so it can be called out
without ambiguity — these are independent of the § numbers and do *not* start at 4. The
single most-cited block, the per-noise test-accuracy tables, is **§4.1 "Results by noise"
(Tables R2–R5)**. Full map in the index directly below.

### Table index

| Label | Table | § |
|---|---|---|
| **R1** | Data sources / sweeps | §4 |
| **R2** | Test accuracy by method — **noise 0.0 (clean)** | §4.1 *(Results by noise)* |
| **R3** | Test accuracy by method — **noise 0.2** | §4.1 *(Results by noise)* |
| **R4** | Test accuracy by method — **noise 0.5** | §4.1 *(Results by noise)* |
| **R5** | Test accuracy by method — **noise 0.8** | §4.1 *(Results by noise)* |
| **R6** | Posterior quality vs ground truth (labeled=denoising, pool=pseudo-labels) — **noise 0.0** | §4.5 |
| **R7** | Posterior quality vs ground truth — **noise 0.2** | §4.5 |
| **R8** | Posterior quality vs ground truth — **noise 0.5** | §4.5 |
| **R9** | Posterior quality vs ground truth — **noise 0.8** | §4.5 |

The curriculum pseudo-labeling arms (`lin10`/`agg50`/`all`) are rows inside R2–R5, not
separate tables (see §4.6). The BAYES_MISP port rows (linear + MLP heads) are also
inside R2–R5 (see §4.7).

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

**Robustness (2026-07-10):** all classifier-trained numbers were re-run with L2-normalized
classifier inputs (`docs/bayesian_coverage_NORMALIZATION.md`, takeaway 9) — the qualitative
conclusions survive; normalization is a small consistent extra win for `pseudo all`
(+~1 pt at every budget/noise) and is convergence-gated for everything else.

### Recommended configuration — normalized `pseudo all`, covered-only

The single best-performing setup across the whole study (best arm at *every* noise
level including clean; see §4.6 + takeaways 8 & 9):

- **Selection:** `bayes_product` (per-class posterior, product-of-experts fusion), tophat
  kernel δ = 0.75, `from_features`. The selection plane always uses normalized features.
- **Training:** `TRAIN_LABEL_MODE='pseudo'` with **`--pseudo_scope all`** — labeled points
  train on their soft posterior, **and every *covered* unlabeled pool point** (in-ball,
  passing guard (a): not a flat/prior-only accumulator row) gets its posterior as a soft
  training target (~10k targets/round at budget 100). Covered-only — out-of-ball NW
  targets were tested and rejected (§7.9, takeaway 10).
- **Classifier input:** **`--normalize_features`** = *scale-preserving* L2,
  $x \to \tfrac{x}{\lVert x\rVert}\cdot\overline{\lVert x_{\text{train}}\rVert}$
  ($\overline{\lVert x_{\text{train}}\rVert}=2.2761$ on CIFAR100). **Not** plain unit-L2
  (that collapses low-budget training, 13.3 → 3.3 @100). `pseudo all` trains on enough
  points to always converge, so this is a clean +~1 pt at every budget/noise (unlike the
  labeled-only arms, where normalization is convergence-gated and harms at ≤300 labels).

Caveats: guard (b) (per-ball caps on correlated pseudo-evidence) is unimplemented, so
`agg50` is the safer scope choice at noise 0.8; below ~400 labels `probcover + 1nn` still
beats every pseudo arm (§4.6).


---

## 4. Results — CIFAR100, test accuracy (%), mean±std over seeds {1,2,3}

Like-for-like regime: CIFAR100 SimCLR features, budget 100/episode, tophat $\delta=0.75$,
`from_features` eval. Sources (all complete, per-run `result.json` on disk):

**Table R1 — Data sources / sweeps.**

| Sweep | SLURM job | Grid | Status |
|---|---|---|---|
| Bayes comparison | 30978366 | {random, probcover, bayes_additive, bayes_product} × noise {0, .2, .5, .8} × 3 seeds | 48/48 ✔ |
| Bayes additive re-run | 31066809 | bayes_additive (fixed acquisition) × noise {0, .2, .5, .8} × 3 seeds | 12/12 ✔ |
| Bayes train-mode | 30980646 | bayes_product × 5 modes × noise {0, .2, .5, .8} × 3 seeds | 60/60 ✔ |
| Netta NPC repro | 30811243 | {npc, probcover, random} + AUM/clean-train × noise {.2, .5, .8} × 3 seeds | 27/27 ✔ |
| AUM clean ablation | 30863746 | {npc, probcover, random} × clean {on, off} × noise {.2, .5, .8} × 3 seeds | 54/54 ✔ |
| §7.5 pseudo-labeling | 31047129 | bayes_product + pseudo × {all, lin10, agg50} × noise {0, .2, .5, .8} × 3 seeds | 36/36 ✔ |
| ProbCover + 1-NN eval | 31048649 | probcover × `--eval_model_type NN1` × noise {0, .2, .5, .8} × 3 seeds | 12/12 ✔ |
| BAYES_MISP parity | 31078514 | {probcover, bayes_misp, bayes_additive} × MLP head, clean, seed 1 (port verification vs Itai's repo) | 3/3 ✔ |
| BAYES_MISP noise | 31084429 | bayes_misp × noise {0, .2, .5, .8} × 3 seeds × head {MLP, linear} | 24/24 ✔ |

All `bayes_*` selection rows train on **raw observed labels** unless a train-mode is
named. Netta's sweeps have **no noise-0 arm** (never run) — hence the dashes.
Regenerate these tables with the script noted in §6.

**Netta-side decomposition (rows from the AUM ablation, job 30863746):** the two `npc`
rows separate her two mechanisms. *npc, no clean-train* = NPC **selection only**, trained
on all (noisy) labels — it is $\approx$ probcover + training everywhere (identical at 100
by construction: the AUM filter only kicks in after the first training round). *npc + AUM
clean-train* = the full Netta stack. So on her side too, **nearly all the gain over
probcover + training comes from the training plane** (clean-train), not from selection — mirroring
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

**"probcover + 1nn" row:** ProbCover's paper evaluation framework (ii) — no trained
network at all; a `KNeighborsClassifier(n_neighbors=1)` is fitted each episode on the
labeled set's **L2-normalized** SimCLR features and evaluated on the test set
(`--eval_model_type NN1`, `MODEL.USE_1NN`; launcher `batch_utils/sweep_probcover_1nn.py`,
sweep 31048649). Selection is identical to `probcover + training` — only the evaluator
differs. **Normalization is load-bearing**: the features datasets serve raw
(`normalized=False`) SimCLR features, and Euclidean 1-NN on raw features loses ~18 pts
@100 (19.9 vs 38.1; buggy first sweep 31048609, fixed in `train_al.py`'s USE_1NN branch —
verified by offline replay on the identical lSets). With the paper convention, 1-NN
**beats training a linear head at every clean budget except 1000** — reproducing the
paper's low-budget claim (~35% @100 on CIFAR100). Seeds still matter without training:
the seed shifts the train/val split (hence ProbCover's candidate pool) and, under noise,
the noise realization the 1-NN fits on. The arm exists at all noise levels (0.2/0.5/0.8
@100: 30.9 / 19.9 / 7.6, @1000: 36.9 / 23.8 / 9.9): even fitted on noisy labels, 1-NN
beats the trained head at 100–300 labels at *every* noise level and loses only from
~500–700 on. The `probcover + 1nn` row appears in all four §4.1 noise tables (R2–R5).

> **δ caveat (see `bayesian_coverage_DELTA_ROOTCAUSE.md`).** The 38.1 @100 clean is
> ~3-4 pts above the paper's ~34-35 for the *same* framework, and a controlled
> decomposition shows the gap is **entirely δ**: these arms use `initial_delta=0.75`
> (the CIFAR-10 radius), which sits exactly at the peak of the accuracy-vs-δ curve
> for this embedding, whereas the CIFAR-100 purity-correct δ is **0.50** (balls only
> 39% pure at 0.75). Embedding, selection rule, and seeds are all ruled out. Large δ
> makes ProbCover pick the *most typical* points (TypiClust-like) — better 1-NN
> prototypes at 1-label/class, but the coverage guarantee no longer holds. δ=0.75 is
> study-wide, so this applies to every CIFAR-100 ProbCover row, not just 1-NN.

**"pseudo all, norm" row:** the same `pseudo all` arm but with the classifier trained on
**L2-normalized** SimCLR features (`--normalize_features`, scale-preserving; §7.9/takeaway 9,
sweep 31048960, paired seeds {1,2,3}) — the [[recommended configuration]]. All other rows use
raw features; because normalization is a consistent gain for this arm it holds the column-best
(bold) in most cells (compare directly only against the raw `bayes_product + pseudo all` row for the
like-for-like effect). `pseudo all` is the one arm that trains on enough points (~10–30k) to
always converge, so normalization is a small consistent gain at every budget and noise
(+0.4..+1.1); notably it reaches **38.4 @100 clean**, edging past the training-free
`probcover + 1nn` baseline (38.1). Full raw-vs-norm deltas: `bayesian_coverage_NORMALIZATION.md`.

**Training-free evaluator study (2026-07-12):** every selection arm was additionally
scored offline under 1-NN and gap-scaled NW evaluators — raw observed labels, and for
bayes_product also posterior-argmax labels (`batch_utils/eval_1nn_all_arms.py`; full
tables in `results/cifar100/eval_1nn_all_arms/tables.md`; headline reading in
NEXT_STEPS §8.3 Q2). Standouts: 1-NN on posterior-**argmax** labels reaches 28.0 @1000
noise .8 — within ~1 pt of the best *trained* arm, with no training at all — and the NW
κ=3 evaluator is the best training-free clean high-budget cell (46.4 @1000).

**`bayes_additive` acquisition fix (2026-07-14):** the `bayes_additive` rows below were
**re-run** (sweep 31066809, 12 runs, non-norm) after fixing a selection bug. The original
numbers collapsed at low budget on CIFAR100 (13.4 @100 clean, far below probcover's 23.6)
because the shared `coverage_gain` acquisition used the *product's* uncertainty
`1 - max_k P` on the *additive* posterior `C/ΣC`. That posterior only reaches
$n/(n+1)$ confidence after $n$ in-ball labels, so one clean label left `max_k P ≈ 0.5`
(never ~0), covered points never retired, and — since the score sums neighbor
uncertainties — dense balls kept out-scoring sparse/uncovered ones, so the greedy clumped
and starved sparse classes. The fix (`bayesian_coverage.py` `_acquisition_scores`) gives
the additive branch Itai David's saturating ceiling
$f_1 = (\max_k \pi + 1)/(\sum_k \pi + 1)$ — the dominant-class mass after exactly one
tophat in-ball label — and scores `max(f1 - max_k P, 0)`. For tophat $K\in\{0,1\}$ this is
BAYES\_MISP's `diff_method='max'` score *exactly*, so one clean label now retires a point
and clean/low-budget additive lands on ProbCover (24.8 vs 23.6 @100). Attribution note:
the additive *combine* itself is our own variant — the saturating-ceiling **acquisition**
is what comes from Itai; his actual method is the native `bayes_misp` port (§4.7).

### 4.1 Results by noise (Tables R2–R5)

**The headline block of the whole study** — CIFAR100 test accuracy for every method at
each noise level, one table per noise rate. Cite it as **§4.1 "Results by noise"** and
call the individual tables **R2** (noise 0.0), **R3** (0.2), **R4** (0.5), **R5** (0.8).

**Every arm is factored into three key columns** so the sampling method and the
classifier are never conflated:

- **Sampling** — how points are *selected* for annotation: `random`, `probcover`,
  `bayes_additive`, `bayes_product`, `npc`.
- **Classifier model** — how a test point is *labeled*: `linear head` = a linear probe
  trained on the sampled set; `1-NN` (and future `kNN` / `soft-kNN`) = a non-parametric
  classifier fitted on the sampled features; `posterior-argmax` = the graph posterior used
  directly, with **no trained model**. `(norm)` marks L2-normalized (scale-preserving)
  classifier inputs.
- **Train labels** — the *targets* the classifier is trained on: `observed` (raw noisy
  labels), the label-correction modes `map_posterior` / `soft_posterior` / `filtered` /
  `reweighted`, the curriculum pseudo-labels `pseudo (all/lin10/agg50)`, `AUM clean-train`
  (Netta's stack), or `—` for the training-free `posterior-argmax`.

So the user's examples read directly off the row: "random sampling + linear classifier on
sampled" = (`random`, `linear head`, `observed`); "probcover + 1nn" = (`probcover`, `1-NN`,
`observed`); "probcover + training on sampled" = (`probcover`, `linear head`, `observed`).
A **new evaluator** (e.g. `soft-kNN`) is just a new *Classifier model* value crossed with an
existing Sampling × Train-labels pair — add rows, never columns; §6 describes the canonical
long-form store that keeps every combination organized.

Rows are grouped by method family, set off by italic **separator rows** (baselines →
Bayesian-coverage selection → + label-correction training → + pseudo-labeling →
posterior-as-classifier → Netta NPC/AUM). The `pseudo …` arm definitions and reading are in
§4.6, and finer all-10-episode trajectories live in `results/bayes_pseudo/`.

**Table R2 — Test accuracy by method, noise 0.0 (clean sanity).**

<!-- BEGIN:R2 -->
<!-- generated by batch_utils/render_results_tables.py from results_master.csv — do not edit by hand -->
| Sampling | Classifier model | Train labels | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|---|
| *Baselines* | | | | | | | |
| random | linear head | observed | 8.5±2.6 | 25.1±0.7 | 33.8±0.2 | 38.4±1.4 | 42.6±1.0 |
| probcover | linear head | observed | 23.6±1.8 | 39.9±0.4 | 43.3±0.3 | 44.6±0.2 | 46.5±0.6 |
| probcover | 1-NN | observed | 38.1±0.4 | 42.9±0.2 | 44.5±0.2 | 44.9±0.3 | 45.6±0.6 |
| *Bayesian coverage — selection only (train on observed labels)* | | | | | | | |
| bayes_additive | linear head | observed | 24.8±0.6 | 30.5±0.7 | 33.1±0.1 | 35.9±0.4 | 39.2±0.3 |
| bayes_product | linear head | observed | 24.6±1.3 | 38.1±1.3 | 41.6±0.0 | 43.6±0.2 | 45.1±0.1 |
| *Bayesian coverage + label-correction training* | | | | | | | |
| bayes_product | linear head | map_posterior | 22.5±0.5 | 35.9±0.4 | 38.9±0.2 | 40.8±0.4 | 41.8±0.5 |
| bayes_product | linear head | soft_posterior | 23.7±1.0 | 36.3±0.7 | 39.2±0.0 | 41.1±0.5 | 42.5±0.1 |
| bayes_product | linear head | filtered | 5.6±1.4 | 24.2±0.4 | 31.3±0.9 | 36.1±1.0 | 38.6±0.6 |
| bayes_product | linear head | reweighted | 5.8±1.0 | 25.1±1.1 | 32.9±1.0 | 36.9±0.7 | 39.6±0.5 |
| *Bayesian coverage + pseudo-labeling (§4.6)* | | | | | | | |
| bayes_product | linear head | pseudo (lin10) | 26.3±1.0 | 40.6±0.8 | 43.4±0.0 | 45.2±0.9 | 46.3±0.5 |
| bayes_product | linear head | pseudo (agg50) | 26.3±1.0 | 40.6±0.8 | 43.9±0.2 | 45.8±0.5 | 47.8±0.4 |
| bayes_product | linear head | pseudo (all) | 37.6±0.4 | 42.6±0.7 | 44.9±0.1 | 46.5±0.1 | 47.5±0.2 |
| bayes_product | linear head (norm) | pseudo (all) | **38.4±0.3** | **43.3±0.8** | **45.5±0.3** | **47.1±0.0** | **48.4±0.2** |
| *Posterior as classifier* | | | | | | | |
| bayes_product | posterior-argmax | — | 18.1±0.2 | 23.9±0.4 | 26.8±0.3 | 28.8±0.3 | 30.8±0.2 |
| *Netta NPC / AUM* | | | | | | | |
| probcover | linear head | AUM clean-train | — | — | — | — | — |
| npc | linear head | observed (sel. only) | — | — | — | — | — |
| npc | linear head | AUM clean-train | — | — | — | — | — |
| *BAYES_MISP port (Itai's bayes_al_2025 — diff max, α=0.5, tophat δ=0.75; §4.7)* | | | | | | | |
| bayes_misp (Itai) | linear head | observed | 24.5±2.0 | 39.7±0.8 | 43.4±0.1 | 44.5±0.3 | 46.7±0.4 |
| bayes_misp (Itai) | MLP head | observed | 34.9±1.1 | **40.3±0.7** | **42.5±0.4** | **43.7±0.3** | 45.4±0.5 |
| probcover | MLP head (seed 1) | observed | **35.4±0.0** | 39.8±0.0 | 42.3±0.0 | 43.5±0.0 | **45.7±0.0** |
| bayes_additive | MLP head (seed 1) | observed | 35.0±0.0 | 32.8±0.0 | 33.5±0.0 | 33.6±0.0 | 37.6±0.0 |
<!-- END:R2 -->

**Table R3 — Test accuracy by method, noise 0.2.**

<!-- BEGIN:R3 -->
<!-- generated by batch_utils/render_results_tables.py from results_master.csv — do not edit by hand -->
| Sampling | Classifier model | Train labels | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|---|
| *Baselines* | | | | | | | |
| random | linear head | observed | 5.8±2.3 | 19.9±1.6 | 28.2±0.9 | 33.3±1.8 | 37.7±1.6 |
| probcover | linear head | observed | 19.7±0.7 | 33.4±1.3 | 37.4±0.4 | 39.5±0.3 | 41.2±0.8 |
| probcover | 1-NN | observed | **30.9±1.1** | 34.5±0.3 | 36.2±0.6 | 36.7±0.3 | 36.9±0.5 |
| *Bayesian coverage — selection only (train on observed labels)* | | | | | | | |
| bayes_additive | linear head | observed | 20.7±0.9 | 24.8±1.9 | 22.9±0.7 | 23.7±0.5 | 27.3±1.1 |
| bayes_product | linear head | observed | 21.6±1.5 | 32.8±1.5 | 36.6±0.5 | 37.7±0.7 | 39.2±0.3 |
| *Bayesian coverage + label-correction training* | | | | | | | |
| bayes_product | linear head | map_posterior | 20.6±2.3 | 33.7±2.0 | 36.9±1.1 | 39.2±0.8 | 40.6±0.6 |
| bayes_product | linear head | soft_posterior | 19.1±1.7 | 33.4±1.9 | 37.5±1.0 | 40.0±0.5 | 41.5±0.4 |
| bayes_product | linear head | filtered | 22.4±2.4 | 28.9±1.5 | 35.8±1.0 | 38.7±1.6 | 40.8±0.7 |
| bayes_product | linear head | reweighted | 8.6±1.5 | 26.3±2.6 | 33.5±1.5 | 37.1±0.8 | 40.2±0.5 |
| *Bayesian coverage + pseudo-labeling (§4.6)* | | | | | | | |
| bayes_product | linear head | pseudo (lin10) | 19.7±1.6 | 37.6±1.6 | 40.8±0.8 | 42.0±0.5 | 43.8±1.2 |
| bayes_product | linear head | pseudo (agg50) | 19.7±1.6 | 37.6±1.6 | 40.8±0.8 | 41.4±0.8 | 44.1±0.6 |
| bayes_product | linear head | pseudo (all) | 28.6±1.7 | 37.7±1.7 | 41.7±0.6 | 43.6±0.2 | 45.8±0.6 |
| bayes_product | linear head (norm) | pseudo (all) | 29.7±1.5 | **38.7±1.9** | **42.7±0.6** | **44.5±0.2** | **46.6±0.6** |
| *Posterior as classifier* | | | | | | | |
| bayes_product | posterior-argmax | — | 14.3±0.8 | 21.4±0.7 | 24.5±0.1 | 26.4±0.4 | 28.6±0.5 |
| *Netta NPC / AUM* | | | | | | | |
| probcover | linear head | AUM clean-train | 20.2±0.7 | 31.2±1.1 | 36.4±0.5 | 39.5±1.3 | 42.3±0.1 |
| npc | linear head | observed (sel. only) | 19.7±0.7 | 33.9±1.0 | 37.4±1.7 | 39.8±1.1 | 42.0±0.7 |
| npc | linear head | AUM clean-train | 20.2±0.7 | 32.0±0.6 | 36.5±0.6 | 40.4±0.8 | 43.3±0.6 |
| *BAYES_MISP port (Itai's bayes_al_2025 — diff max, α=0.5, tophat δ=0.75; §4.7)* | | | | | | | |
| bayes_misp (Itai) | linear head | observed | 20.6±0.8 | 33.9±1.3 | 37.6±0.3 | 39.7±0.5 | 41.8±0.4 |
| bayes_misp (Itai) | MLP head | observed | **28.3±1.3** | **32.1±0.9** | **34.1±0.7** | **35.1±0.1** | **37.1±0.2** |
| probcover | MLP head (seed 1) | observed | — | — | — | — | — |
| bayes_additive | MLP head (seed 1) | observed | — | — | — | — | — |
<!-- END:R3 -->

**Table R4 — Test accuracy by method, noise 0.5.**

<!-- BEGIN:R4 -->
<!-- generated by batch_utils/render_results_tables.py from results_master.csv — do not edit by hand -->
| Sampling | Classifier model | Train labels | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|---|
| *Baselines* | | | | | | | |
| random | linear head | observed | 4.6±1.6 | 12.7±2.2 | 18.4±1.4 | 23.5±1.0 | 27.1±1.0 |
| probcover | linear head | observed | 13.1±1.3 | 21.5±1.8 | 26.0±1.3 | 28.1±0.7 | 30.3±1.2 |
| probcover | 1-NN | observed | **19.9±1.4** | 22.5±1.6 | 23.7±1.2 | 23.7±1.1 | 23.8±1.3 |
| *Bayesian coverage — selection only (train on observed labels)* | | | | | | | |
| bayes_additive | linear head | observed | 13.5±1.3 | 14.8±3.4 | 14.6±2.8 | 14.8±2.4 | 15.1±2.3 |
| bayes_product | linear head | observed | 14.9±1.4 | 23.1±0.1 | 26.4±0.6 | 28.1±0.6 | 29.5±0.4 |
| *Bayesian coverage + label-correction training* | | | | | | | |
| bayes_product | linear head | map_posterior | 15.2±1.9 | 27.9±1.7 | 31.9±0.2 | 34.5±1.0 | 37.6±0.4 |
| bayes_product | linear head | soft_posterior | 12.3±0.9 | 27.8±1.0 | 33.1±0.5 | 36.2±0.1 | 39.1±0.8 |
| bayes_product | linear head | filtered | 4.3±1.0 | 16.5±2.3 | 23.8±2.7 | 29.5±1.3 | 34.3±0.5 |
| bayes_product | linear head | reweighted | 5.0±2.6 | 17.6±2.4 | 26.3±1.4 | 32.1±0.6 | 36.5±0.8 |
| *Bayesian coverage + pseudo-labeling (§4.6)* | | | | | | | |
| bayes_product | linear head | pseudo (lin10) | 12.5±0.9 | 30.1±0.8 | 36.0±0.6 | 38.6±0.4 | 40.9±1.0 |
| bayes_product | linear head | pseudo (agg50) | 12.5±0.9 | 30.1±0.8 | 36.5±1.0 | 38.8±0.9 | 41.3±0.7 |
| bayes_product | linear head | pseudo (all) | 18.0±2.9 | 31.1±1.1 | 36.1±0.5 | 39.1±0.6 | 41.8±0.7 |
| bayes_product | linear head (norm) | pseudo (all) | 18.4±2.9 | **32.2±0.9** | **37.4±0.5** | **40.1±0.5** | **42.6±0.5** |
| *Posterior as classifier* | | | | | | | |
| bayes_product | posterior-argmax | — | 8.7±1.2 | 16.0±0.7 | 19.1±0.4 | 21.4±0.1 | 23.8±0.3 |
| *Netta NPC / AUM* | | | | | | | |
| probcover | linear head | AUM clean-train | 13.3±1.5 | 20.4±2.6 | 26.0±1.7 | 29.6±1.7 | 33.7±0.8 |
| npc | linear head | observed (sel. only) | 13.1±1.3 | 22.1±1.3 | 25.8±1.5 | 28.6±1.4 | 30.9±1.0 |
| npc | linear head | AUM clean-train | 13.3±1.5 | 20.7±1.0 | 26.6±1.3 | 30.4±1.7 | 35.6±1.0 |
| *BAYES_MISP port (Itai's bayes_al_2025 — diff max, α=0.5, tophat δ=0.75; §4.7)* | | | | | | | |
| bayes_misp (Itai) | linear head | observed | 13.7±0.8 | 21.5±0.9 | 25.8±1.2 | 28.7±0.9 | 31.2±1.1 |
| bayes_misp (Itai) | MLP head | observed | **17.4±1.4** | **20.0±0.9** | **21.7±1.4** | **22.6±1.2** | **24.6±0.8** |
| probcover | MLP head (seed 1) | observed | — | — | — | — | — |
| bayes_additive | MLP head (seed 1) | observed | — | — | — | — | — |
<!-- END:R4 -->

**Table R5 — Test accuracy by method, noise 0.8.**

<!-- BEGIN:R5 -->
<!-- generated by batch_utils/render_results_tables.py from results_master.csv — do not edit by hand -->
| Sampling | Classifier model | Train labels | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|---|
| *Baselines* | | | | | | | |
| random | linear head | observed | 2.6±0.6 | 4.9±2.2 | 7.2±1.7 | 9.4±1.1 | 10.3±1.5 |
| probcover | linear head | observed | 5.9±0.9 | 8.2±1.1 | 10.4±0.4 | 11.6±0.3 | 12.7±0.1 |
| probcover | 1-NN | observed | **7.6±0.5** | 9.3±0.1 | 9.7±0.5 | 9.7±0.4 | 9.9±0.9 |
| *Bayesian coverage — selection only (train on observed labels)* | | | | | | | |
| bayes_additive | linear head | observed | 5.5±2.0 | 6.0±2.3 | 5.7±2.1 | 5.3±1.4 | 5.4±0.9 |
| bayes_product | linear head | observed | 4.6±0.5 | 9.7±2.0 | 10.9±1.5 | 11.8±1.5 | 12.6±1.6 |
| *Bayesian coverage + label-correction training* | | | | | | | |
| bayes_product | linear head | map_posterior | 5.8±0.2 | 14.2±2.9 | 19.4±2.1 | 22.9±2.2 | 26.2±1.3 |
| bayes_product | linear head | soft_posterior | 5.3±0.5 | 13.3±2.0 | 18.9±2.4 | 23.4±2.5 | 27.5±1.9 |
| bayes_product | linear head | filtered | 1.3±0.6 | 6.5±0.9 | 10.2±0.2 | 15.7±1.3 | 20.6±2.5 |
| bayes_product | linear head | reweighted | 1.3±0.6 | 5.2±0.6 | 11.1±1.1 | 17.0±1.3 | 23.4±1.6 |
| *Bayesian coverage + pseudo-labeling (§4.6)* | | | | | | | |
| bayes_product | linear head | pseudo (lin10) | 5.2±0.6 | 14.3±2.5 | 20.1±2.1 | 24.4±2.7 | 28.0±2.6 |
| bayes_product | linear head | pseudo (agg50) | 5.2±0.6 | 14.3±2.5 | 20.1±1.9 | 24.6±2.9 | 29.0±2.7 |
| bayes_product | linear head | pseudo (all) | 6.2±0.7 | 14.4±3.2 | 19.7±2.5 | 24.3±3.4 | 28.1±2.5 |
| bayes_product | linear head (norm) | pseudo (all) | 6.2±0.6 | **15.0±3.7** | **20.4±2.4** | **25.1±3.2** | **29.2±2.3** |
| *Posterior as classifier* | | | | | | | |
| bayes_product | posterior-argmax | — | 2.5±0.3 | 7.0±1.9 | 9.5±1.6 | 11.6±1.9 | 14.3±1.4 |
| *Netta NPC / AUM* | | | | | | | |
| probcover | linear head | AUM clean-train | 5.7±1.6 | 7.1±1.1 | 10.3±1.1 | 11.9±0.7 | 14.7±0.2 |
| npc | linear head | observed (sel. only) | 5.9±0.9 | 8.1±0.9 | 9.8±1.2 | 10.8±0.4 | 12.5±0.4 |
| npc | linear head | AUM clean-train | 5.7±1.6 | 7.4±0.6 | 10.3±0.7 | 11.7±1.0 | 15.0±0.8 |
| *BAYES_MISP port (Itai's bayes_al_2025 — diff max, α=0.5, tophat δ=0.75; §4.7)* | | | | | | | |
| bayes_misp (Itai) | linear head | observed | 5.9±0.6 | 8.6±0.9 | 11.0±1.3 | 12.1±0.5 | 13.6±0.7 |
| bayes_misp (Itai) | MLP head | observed | **7.0±0.9** | **7.5±0.9** | **8.5±0.8** | **9.4±0.8** | **9.9±0.4** |
| probcover | MLP head (seed 1) | observed | — | — | — | — | — |
| bayes_additive | MLP head (seed 1) | observed | — | — | — | — | — |
<!-- END:R5 -->

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

**Posterior label quality vs ground truth (labeled set + unlabeled pool).** Both halves
below are the **same measurement** — posterior-argmax accuracy against the
**true** labels, from the offline replay — on the two different point sets, so they share
one layout. **Tables R6–R9 give one table per noise level** (0.0/0.2/0.5/0.8). The
`labeled:` rows relabel each *labeled* point to the argmax of its own posterior (exactly
the `map_posterior` targets); the `pool:` rows score the *unlabeled* pool points (the
pseudo-label supply). Read the labeled rows as **denoising** quality and the pool rows as
**pseudo-label** quality.

**Table R6 — Posterior quality vs ground truth, noise 0.0** (labeled = denoising, pool = pseudo-labels).

| measure | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| labeled: observed acc | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| labeled: posterior argmax | 87.0±2.6 | 75.7±3.3 | 71.9±0.7 | 70.7±1.1 | 68.3±1.3 |
| pool: covered (% of pool) | 25.6±0.2 | 36.4±1.0 | 41.6±0.7 | 45.6±0.4 | 49.5±0.4 |
| pool: acc on covered | 71.7±0.3 | 69.0±0.6 | 68.1±0.4 | 67.4±0.3 | 66.7±0.4 |
| pool: acc all pool | 18.8±0.2 | 25.4±0.5 | 28.6±0.3 | 30.9±0.3 | 33.2±0.3 |
| pool: acc top-10/class | 76.4±1.2 | 70.1±2.8 | 68.4±1.3 | 68.1±0.7 | 67.9±0.3 |

**Table R7 — Posterior quality vs ground truth, noise 0.2** (labeled = denoising, pool = pseudo-labels).

| measure | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| labeled: observed acc | 81.0±4.4 | 78.8±4.5 | 79.9±3.0 | 80.2±1.7 | 81.0±1.0 |
| labeled: posterior argmax | 71.7±2.9 | 70.8±4.4 | 68.5±2.0 | 67.1±2.9 | 66.2±0.5 |
| pool: covered (% of pool) | 24.4±0.3 | 35.1±0.4 | 40.1±0.3 | 43.9±0.3 | 47.8±0.4 |
| pool: acc on covered | 58.5±3.1 | 61.6±1.7 | 62.8±0.8 | 62.2±0.6 | 62.6±0.3 |
| pool: acc all pool | 14.6±0.8 | 21.9±0.6 | 25.4±0.2 | 27.5±0.2 | 30.1±0.2 |
| pool: acc top-10/class | 64.2±0.4 | 69.1±1.7 | 71.8±1.3 | 74.9±1.5 | **79.3±1.1** |

**Table R8 — Posterior quality vs ground truth, noise 0.5** (labeled = denoising, pool = pseudo-labels).

| measure | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| labeled: observed acc | 48.3±4.7 | 49.9±1.1 | 49.5±1.4 | 49.1±0.5 | 49.5±1.2 |
| labeled: posterior argmax | 65.0±9.5 | 70.2±5.3 | 66.7±0.6 | 68.5±0.8 | 67.7±2.6 |
| pool: covered (% of pool) | 19.9±0.3 | 30.7±0.4 | 36.1±0.6 | 39.9±0.6 | 44.0±0.2 |
| pool: acc on covered | 43.9±6.3 | 52.8±1.5 | 54.1±1.0 | 54.6±1.3 | 55.0±0.6 |
| pool: acc all pool | 9.2±1.4 | 16.5±0.6 | 19.8±0.3 | 22.0±0.2 | 24.4±0.2 |
| pool: acc top-10/class | 35.2±5.7 | 52.2±4.3 | 59.2±2.1 | 65.7±1.3 | **69.4±0.2** |

**Table R9 — Posterior quality vs ground truth, noise 0.8** (labeled = denoising, pool = pseudo-labels).

| measure | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| labeled: observed acc | 19.7±0.6 | 20.1±2.0 | 20.4±0.9 | 20.6±1.9 | 20.6±1.3 |
| labeled: posterior argmax | 39.0±10.1 | 51.9±10.8 | 58.0±8.2 | 59.7±6.0 | **61.6±2.8** |
| pool: covered (% of pool) | 12.6±0.5 | 22.9±0.8 | 28.1±0.7 | 32.1±0.4 | 36.0±0.4 |
| pool: acc on covered | 20.4±1.4 | 30.1±5.9 | 33.3±4.7 | 35.3±4.9 | 37.5±3.2 |
| pool: acc all pool | 3.1±0.3 | 7.3±1.5 | 9.7±1.1 | 11.6±1.5 | 13.7±1.2 |
| pool: acc top-10/class | 11.1±2.4 | 23.0±7.0 | 29.4±2.1 | 35.5±5.0 | **42.8±4.2** |

**Labeled set (`labeled:` rows) — denoising.** Both rows: accuracy vs **true** labels,
evaluated **on the labeled set** at each budget — e.g. column 500 = over those 500
annotated points. *observed acc* = the annotator's labels as given; *posterior argmax* =
each point relabeled to $\arg\max_k P(y_\ell = k)$ of its own posterior — i.e. exactly the
labels `map_posterior` trains on. The argmax is over the 100 classes, not over points.

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

**Unlabeled pool (`pool:` rows) — pseudo-label supply.** Row definitions — all on the
unlabeled pool (uSet): **covered** = fraction of pool
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

---

### 4.6 §7.5 curriculum pseudo-labeling — arm definitions & reading (sweep 31047129, 36/36 ✔)

`TRAIN_LABEL_MODE='pseudo'`: labeled points train on their soft posterior (exactly
`soft_posterior`), **plus** confident *unlabeled* pool points enter the training set
with their posterior as a soft target. Guard (a) is implemented (prior-only points —
flat accumulator row — never pseudo-labeled); guard (b) (per-ball caps) is **not**.
Three arms, selected by `--pseudo_scope` / `--pseudo_k`:

- **all** — every covered pool point (~8.8k pts at 100 labels, noise .5; ~30k at 1000).
- **lin10** — top-k most-confident per *predicted* class, k = training round (1→10).
- **agg50** — same, k grows 1,2,3,5,8,13,20,30,40,50.

Same regime and seeds {1,2,3} as the train-mode sweep, so these rows are paired
seed-for-seed with the other train-modes. **All three arms are reported inside the §4.1
results-by-noise tables (R2–R5)** as the `bayes_product + pseudo …` rows (5 canonical
budgets). The finer **all-10-episode** trajectories (labeled size 100…1000) and the
training-free **probcover + 1nn** baseline curve live in the figure + CSV at
`results/bayes_pseudo/`.

Reading: **by 1000 labels pseudo wins at every noise level, including 0** — see
takeaway 8 (vs the trained-on-observed baseline: 47.8 vs 45.1 @1000 clean; vs
probcover + 1nn: 47.8 vs 45.6). But against the **training-free 1-NN baseline the
picture at low budget changes**: probcover + 1nn beats every pseudo arm at 100–400
labels on clean data and at 100 labels at every noise level — if the budget stops at
~300, skipping training entirely is the best known cell in these tables. `lin10` ≡
`agg50` through 300 labels by construction (the schedules coincide at k = 1,2,3;
identical numbers = the determinism trust-check again). The §4.5 pool-row prediction
(Tables R6–R9: pseudo-labels cleaner than paid annotations wherever noise ≥ .2, worse when clean)
is confirmed *directionally* by training, with one surprise: the arm helps on
**clean** data too (47.8 vs 45.1 trained-on-observed @1000; 37.6 vs 24.6 @100 for
`all`) — the R6–R9 pool rows under-sold it because even 67.9%-clean extra targets are soft
(the wrongness is spread over classes, not committed to a hard wrong label) and
the sheer volume (~9–30k points) dominates the classifier's data diet at these
budgets.

### 4.7 BAYES_MISP port — Itai's method run natively in this repo (sweeps 31084429 + 31078514)

`--al bayes_misp` is a faithful port of Itai David's `BAYES_MISP`
(github `itaidavid1/bayes_al_2025`; vendored as `deep_al/pycls/al/BAYES_MISP.py`
+ `kernel_utils_misp.py` + `prior_selection.py`), run with his benchmark config:
`--diff_method max --alpha 0.5 --sparse_K`, tophat δ=0.75. Port verified against
his own repo on the shared SimCLR bank (clean, seed 1, sklearn-MLP head):
45.0 @1000 here vs 45.2 in his repo vs 44.9 as he reported — and it tracks
probcover within ≤0.7 at every budget, which is its signature (the max-normalized
`f1` ceiling retires covered balls almost exactly like ProbCover's hard removal).

Two evaluator heads are reported in the R2–R5 tables:

- **linear head** — same protocol as every other row in these tables; directly
  comparable. Expect `bayes_misp ≈ probcover` here too.
- **MLP head** (`--eval_model_type from_mlp_sklearn`, new in this repo) — Itai's
  evaluator: sklearn `MLPClassifier(hidden=(256,), relu, adam, α=3e-2,
  max_iter=300)` fit on **raw** (un-normalized) features. It sits ~10 pts above
  the linear protocol at low budget — this is an *evaluator* difference, not a
  selection difference, so MLP rows must only be compared against other MLP rows
  (the tables bold the two head classes separately). The seed-1 `probcover` /
  `bayes_additive` MLP rows (parity sweep 31078514, clean only) provide that
  in-head reference: additive is −7.4 @1000 vs misp under the identical MLP
  evaluator, its known intrinsic deficit (§4 additive gate, takeaway 1) — under
  the MLP head the gap opens from 300 labels already.

Reading (sweep 31084429, 24/24): **under the linear head, `bayes_misp` ≈
`probcover` at every noise level**, with a small consistent edge at 1000 labels
(46.7 vs 46.5 clean; 41.8 vs 41.2 @.2; 31.2 vs 30.3 @.5; 13.6 vs 12.7 @.8) —
Itai's clean-data signature carries over to noise unchanged. The more surprising
result is the **evaluator × noise interaction: the MLP head's low-budget
advantage inverts under label noise**. Clean, it leads the linear head by +10 @100
and stays ahead through ~700; at noise ≥ .2 it *loses* to the linear head from
~300 labels on, and by 1000 the deficit is large (37.1 vs 41.8 @.2; 24.6 vs 31.2
@.5; 9.9 vs 13.6 @.8). The 256-unit adam-trained MLP (300 iterations, weak 3e-2
L2) has the capacity to memorize noisy labels that the underparameterized linear
probe cannot — so **the MLP evaluator is the flattering choice only on clean
data**; under noise it understates every method. Any comparison against Itai's
reported numbers must therefore hold the head fixed, and noise conclusions drawn
on his MLP protocol do not transfer to ours.

## 5. Takeaways

1. **Selection plane alone ≈ ProbCover.** `bayes_product` wins small at the very-low
   end vs probcover + training (1/class: 21.6 vs 19.7 @noise .2; 14.9 vs 13.1 @.5),
   converges or trails slightly by 1000 labels. Annotator-accuracy weighting matters most when labels are
   scarcest, but selection alone is not the win.
2. **The training plane is the win — and it beats Netta's AUM at noise ≥ 0.5.**
   Noise 0.8 @1000: `soft_posterior` **27.5** vs observed 12.6 ($2.2\times$) vs NPC+AUM
   **15.0**. Noise 0.5 @1000: **39.1** vs 29.5 vs 35.6. Soft graph-posterior label
   correction dominates hard AUM filtering exactly where noise is heaviest —
   plausibly because AUM's judge network was itself trained on 50–80% wrong labels,
   while the graph posterior needs no trained model.
3. **At noise 0.2, NPC+AUM still slightly ahead** (43.3 vs 41.5 @1000). Few flips ⇒
   AUM is reliable, and hard-dropping a few bad points beats softening every label.
4. **Product pulls ahead of additive as budget grows; the low-budget sanity now passes**
   (numbers re-run 2026-07-14 with the fixed additive acquisition — see the note above §4).
   At the lowest budget the two arms tie (24.8 additive vs 24.6 product @100 clean; both
   ≈ probcover's 23.6) — so the "additive ≈ probcover + training" sanity check now
   **passes** on CIFAR100, as it always did on CIFAR10. Product then pulls decisively ahead
   at higher budgets and higher noise (45.1 vs 39.2 clean @1000; 29.5 vs 15.1 @1000 noise
   0.5), so the design-§3 product-over-additive gate still passes. The earlier "additive
   collapses (26.9 @1000, 13.4 @100)" reading was an acquisition bug, not a property of the
   additive rule.
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
9. **Classifier-input normalization is convergence-gated — it helps only once
   training has enough points to converge; before that it harms** (full paired
   tables: `bayesian_coverage_NORMALIZATION.md`, sweep 31048960, 144 runs paired
   per-seed against the §4 originals; the posterior/selection plane always used
   normalized features, so only the classifier input changed). At ≤300 labels the
   labeled-only arms lose 2–15 pts (an optimization artifact at the fixed
   BASE_LR/200 epochs: train acc 50% vs 95% @100 — *not* evidence that
   normalization is bad there); the crossover is ~350–400; by 600–1000 every arm
   gains (+2..+8, e.g. probcover+training +8.1 @600 noise .2). Two consequences:
   (a) **`pseudo all` — which trains on ~10k points every round and therefore
   always converges — gains ~+1 pt at *every* budget and noise** (per-seed
   consistent); since it is the headline arm, normalized inputs are now its
   recommended configuration. (b) `observed` gains the most at ≥400, so the
   soft/map-vs-observed noise margin at mid budgets shrinks under normalization
   (noise .5 @1000: +10 → +4.3) — part of what posterior-training fixes on raw
   features is *feature-norm pathology*, not label noise; quote mid-budget
   noise-robustness margins from the normalized runs. ⚠ Plain unit-L2 *collapses*
   low-budget training entirely (13.3 → 3.3 @100); the scheme here is
   scale-preserving: $x \to x/\lVert x\rVert \cdot \overline{\lVert x_{train}\rVert}$.
10. **Out-of-ball NW pseudo-targets don't pay, at any loss weight (§7.9).** Extending
    `pseudo_all` to the ~80% uncovered pool with gap-scaled noise-aware soft-kNN
    targets costs −1..−2 pts at noise ≤ .2 and gains only +0.5..+0.9 at .8; the
    OOB_WEIGHT dose-response {1.0, 0.3, 0.1} is monotonic but ~5× too weak to
    rescue clean — the far targets are *confidently wrong* (26% acc), not merely
    over-weighted, so covered-only training remains the recommended configuration.
    The high-noise gain behaves like label smoothing (weight-insensitive). The
    live follow-up from the same replay is in-ball rbf-weighted product exponents
    (labeled-plane denoising 87→96.7 @100 clean), not more out-of-ball machinery.
    **2026-07-13 addendum — the distance-decayed weight also fails** (NEXT_STEPS §9.1,
    sweep 31058388): keeping only the measured 49–56%-accurate shell just outside δ at
    ramped weight w(d)≤0.8 (far field dropped) still costs −1.1..−1.35 clean at every
    budget. Even good-shell targets (30–55% acc) dilute the ~72%-accurate covered diet,
    so the out-of-ball direction is closed at *any* weighting; covered-only stands.

## 6. Reproducing the tables — single-source-of-truth pipeline

Tables R2–R5 are **not hand-maintained**. They are a *pivot* of one tidy CSV, regenerated
by two scripts. The data flows one direction only:

```
result.json (per run, per seed)  +  eval_1nn per_episode.csv  +  test_posterior_acc.json
        │   batch_utils/build_results_master.py   (collect + map params → canonical arm keys)
        ▼
results/cifar100/results_master.csv     ← SINGLE SOURCE OF TRUTH  (one row per per-seed cell)
   columns:  sampling, model, train_labels, norm, noise, budget, seed, test_acc, sweep_id, source
        │   batch_utils/render_results_tables.py  (pandas-free pivot: mean±std over seed, bold col-max)
        ├──▶ docs/bayesian_coverage_RESULTS.md   (injected between <!-- BEGIN:R2 -->…<!-- END:R2 --> markers)
        └──▶ results/cifar100/results_tables.html
```

**To regenerate everything:** `python3 batch_utils/build_results_master.py && python3
batch_utils/render_results_tables.py` (add `--check` to the renderer to print without writing).
The renderer overwrites *only* the marked table blocks; all prose is left untouched. Re-render is
idempotent.

**Why a CSV, not wider tables.** The three-column key of R2–R5 is a *view*. As `1-NN`, `kNN`,
`soft-kNN`, and posterior-labeled evaluators multiply on top of every sampling × train-labels
pair, the raw CSV grows by **rows, never columns**, so no table is ever restructured again, and
the dashboard Comparison tab can facet by any axis. The training-free evaluator study already
lands in this shape (`results/cifar100/eval_1nn_all_arms/per_episode.csv`), and its `nw_k1`/`nw_k3`
rows are already in the master CSV — the planned `training + kNN` / `soft-kNN` rows just need
(a) a collector branch in `build_results_master.py` and (b) one line in the renderer's `ROW_SPEC`.

**Provenance / disambiguation.** Every master-CSV row carries its `sweep_id`, and the renderer's
`ROW_SPEC` pins each display row to a specific sweep — so e.g. `bayes_additive` is drawn from the
fixed-acquisition rerun **31066809**, never the superseded pre-fix rows still present under
**30978366**; the two Netta arms come from the AUM ablation **30863746**. Sweeps consumed:
{30978366, 31066809, 30980646, 31047129, 31048960 (norm), 31048649 (1-NN), 30811243, 30863746}.
The AUM sweep extends to 3000 labels if a longer x-axis is needed; the dashboard Comparison tab
(Category = `bayesian_coverage` + `aum_filter`, shared `noise_rate` filter) is the interactive view.

Note: the CIFAR10 twin sweep (job 30969432, 44/48) exists but is the wrong regime for
coverage methods (10–100 labels/class); read CIFAR100 only.
