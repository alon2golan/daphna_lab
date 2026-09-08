# Tiny ImageNet Maor Degree Study

These scripts are the reliable Tiny ImageNet continuation of the CIFAR10 and
CIFAR100 MaxHerding density/locality study.

They use:

- Tiny ImageNet
- Maor SSL features
- `features_tag=maor`
- `VAL_RATIO=0.0`
- MaxHerding
- 1NN evaluation
- budget 200 per round
- labeled-set sizes 200 to 2000

The Maor feature pair is used because it gives about `29.94%` full-train 1NN
accuracy, while the previous default Tiny feature pair gave only about `13.13%`.

Run in this order:

```bash
sbatch scripts/sbatch/tinyimagenet_maor_degree_study/tinyimagenet_maor_delta_probe_seed3.sh
sbatch scripts/sbatch/tinyimagenet_maor_degree_study/maxherding_tinyimagenet_maor_seed3_nn1_cutoff_global.sh
sbatch scripts/sbatch/tinyimagenet_maor_degree_study/maxherding_tinyimagenet_maor_seed3_nn1_cutoff_k_sweep.sh
```

After all MaxHerding runs finish:

```bash
sbatch scripts/sbatch/tinyimagenet_maor_degree_study/maxherding_tinyimagenet_maor_density_selection_seed3.sh
```

The K sweep uses:

- 10
- 60
- 250
- 500
- 1000
- 1500
- 3000
- 5000
- 10000

The diagnostics write CSV files to:

```text
results_summaries/maxherding_tinyimagenet_maor_density_selection/
```
