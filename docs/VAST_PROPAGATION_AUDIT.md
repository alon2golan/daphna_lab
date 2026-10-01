# VAST: source and first-round propagation audit

Checked 2026-10-01. The source-backed correction is to forward class evidence
above the prior, reintroduce hard-label evidence as a fixed source term, and
allow the opening rounds with no freezes. The literal-paper diagnostic used for
the previous runs implements a different recurrence and stopping rule. Neither
switching its sender mask nor adding unit initialization self-evidence removes
its mathematical confidence ceiling.

## Sources inspected

- Public author repository: latest `main` is still
  `3d4cbd7d543b1d1290799bdfb02687519af3a264` (2026-07-02).
  [Propagation code](https://github.com/itaidavid1/bayes_al_2025/blob/3d4cbd7d543b1d1290799bdfb02687519af3a264/deep-al/tools/train_al.py#L270).
- Newer lab source:
  `/cs/labs/daphna/itai.david/py_repos/TypiClust/deep-al/tools/train_al.py`,
  mtime 2026-09-29; SHA256
  `349d583c98ce701889b1ddfb74c580538067e9ba08cbd3ce00dadec14f21b434`.
  `propagate_labels`: lines 479-866; initial-plateau guard: lines 746-759;
  excess-evidence update: lines 765-788.
  Locally inspected copy: `/tmp/vast-author-lab-train_al.py`.
- Actual author log:
  `/cs/labs/daphna/itai.david/py_repos/TypiClust/run_outputs/run_exp_31187414_2.txt`,
  first CIFAR100 experiment beginning at line 3 (experiment path dated 2026-07-30).
  Locally inspected copy: `/tmp/vast-author-run-31187414_2.txt`.
- Attached August PDF: sections 3.2-3.4, Algorithm 1, supplement A.2.
  The [September public paper](https://arxiv.org/html/2609.14451v1#S3.SS4)
  still specifies the additive full-evidence update and first no-new-freeze stop.

The July public helpers are now copied into `deep_al/vast/author_reference.py`.
All four copied function ASTs match the original exactly. The newer lab function
was executed separately from its exact source, without importing its full
training application. The lab code is newer than the recorded July experiment;
matching the log is evidence of compatible behavior, not proof that this exact
file revision generated that log or the paper tables.

## What differs in the executable algorithm

Let `alpha = 1/M`, `H` be one-hot known annotations at selected rows and zero
elsewhere, and `B = K @ H`. For sum normalization and symmetric K:

```text
Initialization used in the controlled comparison: V0 = alpha + B

Author recurrence (raw messages, no decay):
  S = eligible UNLABELED senders
  V_next = alpha + B + K @ ((V_current - alpha) * S[:, None])
  restore labeled and already-frozen rows

Our literal-paper diagnostic:
  S = eligible senders, including labeled rows
  V_next[receivers] = V_current[receivers]
                     + (K @ (V_current * S[:, None]))[receivers]
```

1. **The prior is not a message.** An uninformed row contains only `alpha`.
   Its author-code message `V - alpha` is exactly zero. Our literal formula
   forwards the uniform prior repeatedly, increasing non-class-specific mass.
2. **Known labels remain class-specific sources.** `B` is reconstructed from
   known annotations, regardless of what the labeled row's own normalized
   evidence says. Labeled rows are excluded from the soft-message term by
   default. Their confidence does not need to be 1, and no self-loop is required.
3. **Receivers are reconstructed, not added to their old full row.** The fixed
   prior and hard-label source contribution are added once per reconstructed
   row. Soft excess evidence is the propagated component. This recurrence is
   not a convex average of the previous prior-bearing probabilities and can
   raise the global confidence maximum.
4. **No freeze during the opening rounds is allowed.** The lab
   `early_stop_receivers=True` implementation first requires a receiver count
   below its initial value. Only then can equal consecutive receiver counts
   stop the run. A configured maximum prevents an endless opening plateau.
   The July public function instead uses a fixed iteration count.

## Actual author run: the initial plateau is expected

The first experiment in the inspected log uses 100 selected CIFAR100 labels,
seed 1, BYOL features, prior 0.01, `local_cknn` k=13, logarithmic degree
correction, sender mode `high_confidence` with threshold **0.05**, receiver mode
`high_confidence` with threshold **0.75**, decay 1, labeled soft senders excluded,
cap **50**, and receiver early stopping enabled.

These are recorded settings, not parser defaults or a guess at paper settings.
Receiver counts are logged **before** each update:

| Update about to execute | Eligible senders | Receivers before update |
|---:|---:|---:|
| 1 | 2,010 | 49,900 |
| 2 | 8,141 | 49,900 |
| 3 | 17,398 | 49,838 |
| 4 | 29,345 | 48,436 |
| 5 | 39,728 | 44,112 |
| 13 | 49,779 | 35,230 |
| 14, stopped before update | 49,823 | 35,230 |

The run completes **13 updates**. In particular, update 1 produces no freeze,
but update 2 does, as seen in the receiver count before update 3. Applying the
paper's first-no-new-freeze stopping rule would have cut this run off before
its first freezes.

The log also uses entropy-based admission and `batch_mean` weight application;
our paper-based gate and weighting differ. The lab `local_cknn` implementation
supports adaptive k-NN bandwidths, whereas our saved graphs use fixed-bandwidth
RBF weights. The author's log does not establish the full graph configuration
or exact embedding checkpoint. Therefore matching propagation behavior alone
does not justify claiming reproduction of Table 1 or selecting accuracy-based
defaults. The output/config directory named in this author log returned
`Permission denied` to `tomn`, so its full saved configuration could not be read.

## Confirmed bug in our sender mask

For 100 classes, normalizing an exactly uniform float64 row of `.01` produces
maximum `.010000000000000002`. For 200 classes, `.005` similarly becomes
`.005000000000000001`. The former condition `q.max() > 1/M` therefore admitted
prior-only rows; this explains why earlier traces counted the whole training
pool as senders even when most points had no label evidence.

For positive evidence, `max(q) > 1/M` is equivalent to a nonuniform evidence
row. The diagnostic now uses `V.max(axis=1) > V.min(axis=1)`, avoiding an
arbitrary confidence tolerance. Two regression cases verify that uninformed
seed and two-hop rows do not send. Existing outputs are preserved; their sender
counts retain the old rounding error. This fix does not change the literal
recurrence or repair its freezing failure by itself.

The copied author helpers retain their original threshold comparison for exact
source parity. In them, an accidentally eligible prior-only sender contributes
zero because its message subtracts `alpha`. The audit reports these eligible
but uninformed sender counts separately.

## Controlled checks

`deep_al/tools/audit_vast_propagation.py` uses identical cached zero-diagonal
graphs, saved selected labels and sum-plus-prior initialization for both rules.
Its author-reference settings deliberately use the paper's above-chance sender
threshold `1/M`, receiver threshold 0.75, raw messages, decay 1 and five fixed
rounds. This isolates the recurrence; it is **not** the recovered log's 0.05
threshold/50-cap/guarded-stop configuration. It does not train a classifier.

Each sequential single-round reference run is verified to equal one original
multi-round author call, including source restoration and persistent freezing.
The output records source hashes, graph hash, selected-index hash, uninformed
coverage, per-round global and unlabeled confidence, and freeze counts.

On a five-point unit-weight chain with one known label and 100 possible
classes, the author function's maximum confidence progresses approximately
`0.505 -> 0.505 -> 0.670 -> 0.7525 -> 0.802`. Its first freeze arrives in update 3.
The literal diagnostic stops after update 1 with zero freezes. The exact newer
lab function, using receiver early stopping and cap 10, ignores the initial
plateau, completes six updates, then stops on a zero-receiver plateau.
The source reference and sender correction pass all **38** VAST/replay tests.

Synthetic records: `results/vast/author_audit_chain.json` and
`results/vast/author_lab_audit_chain.json`.
Real-data comparison job: `31833366`; result file
`results/vast/author_audit_31833366.json` (five updates, same six settings).
The job completed successfully (exit 0, 2 minutes 50 seconds). All six sequential
comparisons passed parity with the original multi-round function:

| Dataset | Labels/class | Maximum confidence after update 1 | Frozen unlabeled after update 1 | Maximum confidence after update 5 | Frozen unlabeled after update 5 |
|---|---:|---:|---:|---:|---:|
| CIFAR100 | 1 | 80.34% | 27 | 98.13% | 16,553 |
| CIFAR100 | 2 | 85.24% | 84 | 98.67% | 17,323 |
| CIFAR100 | 4 | 87.15% | 297 | 98.10% | 18,828 |
| TinyImageNet | 1 | 55.54% | 0 | 90.10% | 1,535 |
| TinyImageNet | 2 | 59.87% | 0 | 88.42% | 1,111 |
| TinyImageNet | 4 | 62.02% | 0 | 86.22% | 600 |

Thus even with the source update, applying the literal-paper immediate
no-new-freeze stop would wrongly cut off all three TinyImageNet comparisons.
The source-backed startup guard or a specified fixed-round run is essential.
These are propagation-confidence diagnostics, not pseudo-label correctness or
classifier accuracies; unrevealed labels were not used to tune the comparisons.

Re-run on the cluster (the original cached graphs and manifests must be present):

```bash
python deep_al/tools/audit_vast_propagation.py \
  --run-root results/vast/cifar100_seed0_31831247 \
  --run-root results/vast/tinyimagenet_seed0_31831606 \
  --rounds 5 --output results/vast/author_audit_new.json
```

## Precise questions for Itai

1. Does the prior-subtracted source recurrence above, with the startup-plateau
   guard, describe the code used for the paper? Equations 5/A.1 and Algorithm 1
   currently specify a different update and immediate no-new-freeze stop.
2. Which logged experiment/configuration generated each paper table entry?
   Is `run_exp_31187414_2.txt` part of those experiments or a later variant?
3. Was the sender threshold 0.05 for CIFAR100, `1/M`, or another value?
   What cap and stopping flags were used for each dataset and label budget?
4. Were the graph weights adaptive `local_cknn`, or fixed RBF with
   `2*sigma^2=1`? Were `skip_knn_bandwidth`, `fixed_sigma` or other ablation flags
   enabled? Please identify the feature checkpoint and exact row ordering.
5. Were gating and sample weights based on entropy or maximum probability?
   Was KL weighting applied per example before reduction or to batch means?
   Which current source function/config corresponds to the reported procedure?

No message has been sent to Itai. These questions and the verified examples
are prepared for the user to discuss with him.
