# Professor Notes 2026-05-28 Experiments

These scripts address the PDF note asking whether the early density-normalization
gain is sensitive to the ProbCover cutoff delta.

They run MaxHerding with:

- 1NN evaluation
- ProbCover-style RBF cutoff
- weighted symmetric density normalization
- alpha values `0` and `0.5`
- global sigma and the best local K from the previous CIFAR sweeps
- seeds `3, 4, 5`

Each array is limited to 4 concurrent GPUs using `%4`.

Submit with:

```bash
sbatch scripts/sbatch/professor_notes_2026_05_28/maxherding_cifar10_delta_sensitivity.sh
sbatch scripts/sbatch/professor_notes_2026_05_28/maxherding_cifar100_delta_sensitivity.sh
```

Outputs are written under:

```text
outputs/professor_notes_2026_05_28/
```

After the jobs finish, aggregate them with:

```bash
python3 scripts/aggregate_professor_delta_sensitivity.py
```

The aggregation writes CSVs under:

```text
results_summaries/professor_notes_2026_05_28/
```

and plots under:

```text
visual_outputs/professor_notes_2026_05_28/delta_sensitivity/
```
