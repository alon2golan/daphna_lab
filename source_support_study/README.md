# CIFAR-100 research on noisy annotation sources

## Why test this

The latest fixed-depth VAST study found near parity between mutable and protected propagation (−0.06 pp across 720 comparisons). Keeping a class-balanced top 80% lost 0.68 pp. Ten-step original-broadcast JS gained 0.69 pp over the stronger herding baseline on symmetric 50% noise, with a single acquisition seed. The older lab runs found large gains from distillation in several noisy regimes and frequent clean-label damage from hard filtering. [Prior lab chat](codex://threads/01a0f6cf-f712-79e3-bf47-13f0f6ac63cf) · [Bayesian coverage results](https://github.com/alon2golan/daphna_lab/blob/bayesian_coverage/docs/bayesian_coverage_RESULTS.md)

Confidence can arise when the same observed label reaches a point by several graph paths. This study asks whether the number and strength of distinct annotations supporting the prediction can separate correct pseudo-labels from overconfident errors. A 13-neighbor surrogate replay gave a positive signal under heavy symmetric noise and mixed signals under asymmetric and clean conditions. The exact CkNN replay tests the mechanism in the implementation used for the lab's most recent VAST comparisons.

## Experiment 1 · Pseudo-label admission

Use the saved Itai protected five-update checkpoints over CIFAR-100 BYOL. Replay 20 cells spanning 200, 500, 1,000 and 2,000 labels, with clean, symmetric 20/50%, and asymmetric 20/50% noise. The original checkpoint uses seed 0 and draws the annotations by clean class. The graph contains 50,000 training images, with the fixed 5,000 validation images excluded from admission. This is a paired SSL experiment conditional on its historical acquisitions.

- Recover the exact teacher's sender and receiver masks, and verify replay against the archived probability matrix. Recompute evidence per original annotation in column blocks through those same masks. For unprotected rows, the sum of the columns plus the uniform prior must reconstruct the teacher.
- For the winning and runner-up probabilities p1 and p2, let f be the fraction of total evidence from the strongest single winning annotation. Rank by (p1 − p2 − f)/(1 − f). This is the smallest margin obtainable by removing any one annotation while holding the masks fixed.
- In a secondary arm, multiply p1 by (1 − 0.25s), where s is the largest share of the winner's above-prior evidence supplied by one annotation. The 0.25 discount was chosen before the exact CkNN results, based on the exploratory equal-edge replay.
- Use the saved Itai gate, confidence, and raw top-two margin as controls. All replacements preserve the saved count within every predicted class. Eligibility uses teacher scores and the saved split. Clean pool labels first enter to measure precision after membership is fixed.
- The primary endpoint is pooled pseudo-label precision at the original gate's coverage. Mean precision across predicted classes and the fraction of retained pseudo-labels stable to a single deletion are secondary. Look for at least +1 pp over Itai and the raw margin on average across sym50 budgets and examine clean and asymmetric use separately. Budget checkpoints share one seed and support descriptive comparisons.

Fixed masks make the deletion calculation exact for the resulting linear operator. A full rerun could change the sender and receiver gates. The interpretive claim is about the dependence of the saved predictions on their sources.

After the pilot showed that the full gate can admit the entire pool, we added one equal-count diagnostic before the full sweep. Use the most confident 10% of the unlabeled pool to set per-predicted-class quotas, then rerank against those quotas. This measures precision when only a small subset can be distilled.

## Related work

- VAST propagates class evidence in a frozen representation and distills the result into an inductive student. [David and Weinshall, 2026](https://arxiv.org/abs/2609.14451)
- SENTPROP uses repeated draws of seed sets to measure the uncertainty of graph predictions. This experiment evaluates the worst single supporting annotation under the saved VAST masks. [Hamilton et al., 2016](https://aclanthology.org/D16-1057/)
- The transductive leave-one-out label filter estimates whether observed labels disagree with the graph after self-influence is excluded. This experiment scores the reliability of unlabeled training targets. [Afonso and Berton, 2020](https://arxiv.org/abs/2009.11811)

## Reproduce the replay

Set the Python import path to this branch, the archived fixed-depth study's source, and its runtime dependencies. Use the archived Slurm runtime with SciPy, NumPy and Torch. The archived study folders are read as inputs.

```bash
python -m unittest source_support_study.test_stability
python -m source_support_study.replay --archive "$fixed_depth_study" \
  --output "$new_results" --index 2
```

Indices 0 through 19 follow budget-major order. Each case produces an aggregate JSON file and a separate NPZ of admitted CIFAR row IDs. The checksums and teacher reconstruction errors are recorded per case.

## Experiment 2 · Matched student with a 10% shortlist

The exact replay motivated a downstream test because the shortlist improved under both symmetric noise rates and declined at asymmetric 50%. Compare the confidence and deletion-margin shortlists with identical predicted-class counts. Fit the original VAST MLP for 300 epochs in both arms. Targets, observed-label cross-entropy, loss weights, optimizer steps and the clean validation schedule are shared. The choice of admitted unlabeled rows is the intervention.

Run symmetric 50% at 500 and 1,000 annotations, plus clean and asymmetric 50% controls at 1,000. These cells were selected after reviewing pseudo-label precision. Both arms are compared on the same 10,000 held-out test images, with one seed. The archived full-gate MLP has a different training set and exposure and serves as context for the shortlist comparison.

For local execution in the pinned CUDA environment, use `python -m source_support_study.student --archive "$fixed_depth_study" --output "$new_results" --index 0`. See `student.sbatch` for the cluster command. The code regenerates the shortlist and uses the archived student and verification routines.

| Experiment | Slurm job | Scope |
|---|---:|---|
| Replay pilot | 32667663 | Two symmetric-50 cells |
| Full replay | 32667824 | Four budgets × five conditions |
| Student pilot | 32667947 | Two-epoch smoke fits |
| Matched student | 32668024 | Four cells × two 300-epoch fits |
