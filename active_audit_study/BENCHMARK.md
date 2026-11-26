# Signed audit verification

## Matched comparison

Keep the signed score, BYOL bank, 13-neighbor graph and three-hop voting protocol from the original study. Every setting shares an initial noisy label set. Each equal-cost arm spends an additional 10% of that budget on clean answers, either by auditing old labels or buying new labels. Leave-one-out auditing is the primary reference. Buying the next coverage or MaxHerding selections clean is the acquisition reference. Random audits and unsigned impact remain controls.

| Acquisition order | Seeds | Initial labels | Noise |
|---|---|---|---|
| Seeded greedy 13-neighbor coverage | 10–19 | 400, 600, 800, 1,200 | Clean, symmetric and asymmetric 30/40/50%, CIFAR-100N |
| Archived MaxHerding | 20–24 | 800, 1,200 | Symmetric and asymmetric 30/40/50% |

The MaxHerding replay uses its exact previously exported order. Each seed redraws noise on that shared order. The acquisition reference takes the next 80 or 120 points and buys clean answers. All policies use the same test classifier within a comparison. Graph test accuracy is primary. Cosine 1NN tests transfer to a second predictor.

Look for gains over both equal-cost references across multiple rates in both noise families and multiple budgets. Count each seed pair once within a cell and show the full grid. A negative cell is evidence of a boundary. Adjacent budgets reuse prefixes, and the test set is shared across seeds. Perfect clean verification is simulated from the canonical labels.

## Verified results

Signed auditing improved mean held-out graph accuracy over both equal-cost controls in every synthetic cell. Every cell had a majority of paired wins over leave-one-out.

| Acquisition order | Synthetic cells ahead of both | Seed pairs ahead of leave-one-out | Gain vs leave-one-out | Gain vs clean expansion |
|---|---:|---:|---:|---:|
| Seeded 13-neighbor coverage | 24/24 | 228/240 | +0.21 to +4.61 pp | +2.21 to +8.50 pp |
| Archived MaxHerding | 12/12 | 60/60 | +1.01 to +3.63 pp | +3.36 to +8.08 pp |

Ranges are over cell means. Signed auditing beat clean expansion in all 300 synthetic seed pairs. Its cosine 1NN accuracy improved over leave-one-out in 20/24 coverage cells and 9/12 archived MaxHerding cells. On CIFAR-100N, graph gains over leave-one-out ranged from +0.24 to +1.56 pp over the four budgets. The clean control tied leave-one-out, and clean expansion improved graph accuracy.

Slurm array 32669092 and retry 32669138 supplied 15 verified seed artifacts. The collector verified 2,660 policy evaluations. The dashboard shows absolute accuracy, paired deltas and seed wins for each reference. [Per-seed scores](benchmark_metrics.csv) · [Provenance](benchmark_provenance.json)

## Reproduce

Copy the benchmark source and the archived acquisition order to an isolated experiment directory. Use the tested lab Python environment and the pinned prior graph.

```bash
sbatch --array=10-24%4 run_benchmark.slurm.sh "$job_dir" "$geometry_npz"
python benchmark.py collect "$job_dir"
streamlit run active_audit_study/streamlit_app.py
```

Place the verified `benchmark_metrics.csv` beside the Streamlit file. The dashboard was tested with Streamlit 1.65, pandas 3.0 and Plotly 7.1. The collector checks the full study grid, request counts and order. The archived MaxHerding order was identical across the seven runs in its source snapshot.
