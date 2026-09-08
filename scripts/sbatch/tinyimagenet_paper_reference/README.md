# Tiny ImageNet Paper Reference

This directory contains paper-style Tiny ImageNet MaxHerding reference jobs.

The key difference from `tinyimagenet_degree_study` is that these jobs are meant
to reproduce the MaxHerding paper baseline, not the cutoff/locality experiment.

The paper-style reference uses:

- MaxHerding
- global RBF sigma
- no ProbCover-style cutoff
- no local sigma
- no density normalization
- 1NN evaluation on fixed SSL features
- budget 200 per round
- 10 labeled-set sizes from 200 to 2000

Run:

```bash
sbatch scripts/sbatch/tinyimagenet_paper_reference/maxherding_tinyimagenet_seed3_nn1_global_paper_maor_features.sh
sbatch scripts/sbatch/tinyimagenet_paper_reference/maxherding_tinyimagenet_seed3_nn1_global_paper_maor_features_no_val.sh
sbatch scripts/sbatch/tinyimagenet_paper_reference/maxherding_tinyimagenet_seed3_nn1_no_cutoff_sparse.sh
sbatch scripts/sbatch/tinyimagenet_paper_reference/maxherding_tinyimagenet_seed3_nn1_global_paper.sh
```

The recommended first run is
`maxherding_tinyimagenet_seed3_nn1_global_paper_maor_features.sh`.
It uses the stronger Maor Tiny ImageNet feature pair, which gives about
`29.94%` full-train 1NN accuracy with the correct validation-label order.

If the Maor-feature run is still below the paper, run the no-validation variant:
`maxherding_tinyimagenet_seed3_nn1_global_paper_maor_features_no_val.sh`.
It uses `DATASET.VAL_RATIO=0.0`, so MaxHerding selects from the full 100000-image
training pool. This is closer to the 1NN paper protocol because no validation
split is needed for model selection.

The earlier diagnostic run is `maxherding_tinyimagenet_seed3_nn1_no_cutoff_sparse.sh`.
It keeps the sparse RBF implementation but removes the ProbCover-style cutoff,
so it isolates the cutoff as the first suspected cause of the low accuracy.

The exact dense paper-style job requests `gg:g4:1` instead of `gg:g0:1` because
exact no-cutoff Tiny ImageNet builds a dense `100000 x 100000` RBF kernel. In
`float16`, the kernel alone is about 20 GB before overhead.
