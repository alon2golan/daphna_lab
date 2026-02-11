# Running CIFAR-10 with Kernels on Phoenix Cluster

## Quick Start

### 1. SSH to Phoenix
```bash
ssh ido.dotan@phoenix.cs.huji.ac.il
```

### 2. Navigate to project
```bash
cd /cs/labs/daphna/ido.dotan/projects/daphna_lab
```

### 3. Check available GPUs (recommended)
```bash
sinfo -o "%N %G %t" | awk '{if ($3=="idle") {print "\033[1;32m" $0 "\033[0m"} else if ($3=="mix") {print "\033[1;33m" $0 "\033[0m"} else if ($3=="alloc") {print "\033[1;34m" $0 "\033[0m"} else if ($3=="drain") {print "\033[1;31m" $0 "\033[0m"} else {print $0} }'
```
**Color codes:** 🟢 Green=idle (best), 🟡 Yellow=mix (available), 🔵 Blue=allocated, 🔴 Red=drain (avoid)

**Available GPU types:**
- `gpu:rtx2080:1,vmem:10g,gg:g0:1` - creek, dumfries (most available, default in script)
- `gpu:a5000:1,vmem:24g,gg:g0:1` - binky, drape (more VRAM)
- `gpu:a10:1,vmem:22g,gg:g0:1` - arion, ampere
- `gpu:l4:1,vmem:23g,gg:g0:1` - hasufel, incitatus
- `gpu:rtx6000:1,vmem:24g,gg:g0:1` - firth (most VRAM)

### 4. Submit job with default settings (RBF kernel, ProbCover, RTX2080)
```bash
sbatch sbatch_cifar10_kernels.sh
```

## Custom Configurations

### Change GPU type
Edit line 5 in `sbatch_cifar10_kernels.sh` or use a different GPU:
```bash
# For more VRAM (A5000)
sbatch --gres=gpu:a5000:1,vmem:24g,gg:g0:1 sbatch_cifar10_kernels.sh

# For faster GPU (L4)
sbatch --gres=gpu:l4:1,vmem:23g,gg:g0:1 sbatch_cifar10_kernels.sh
```

### Run with TopHat kernel
```bash
sbatch --export=KERNEL_TYPE=tophat sbatch_cifar10_kernels.sh
```

### Run with DCoM method
```bash
sbatch --export=AL_METHOD=dcom,KERNEL_TYPE=rbf sbatch_cifar10_kernels.sh
```

### Run with custom budget and iterations
```bash
sbatch --export=BUDGET=50,MAX_ITER=20,INITIAL_DELTA=0.6 sbatch_cifar10_kernels.sh
```

### Run multiple experiments with different kernels
```bash
# RBF kernel experiment
sbatch --export=KERNEL_TYPE=rbf,EXP_NAME=rbf_exp sbatch_cifar10_kernels.sh

# TopHat kernel experiment
sbatch --export=KERNEL_TYPE=tophat,EXP_NAME=tophat_exp sbatch_cifar10_kernels.sh
```

## Available Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `KERNEL_TYPE` | `rbf` | `rbf`, `tophat` | Kernel function from kernels.py |
| `AL_METHOD` | `probcover` | `probcover`, `dcom`, `typiclust_rp`, `typiclust_dc` | Active learning strategy |
| `EVAL_MODEL_TYPE` | `from_images` | `from_images`, `from_features`, `NN1` | Model training mode |
| `BUDGET` | `100` | any int | Samples to label per round |
| `INITIAL_SIZE` | `0` | any int | Initial labeled set size |
| `INITIAL_DELTA` | `0.75` | float | Kernel bandwidth parameter |
| `MAX_ITER` | `10` | any int | Number of AL rounds |
| `SEED` | `1` | any int | Random seed |
| `EXP_NAME` | `auto` | string | Experiment name (auto = timestamp) |

## Monitoring Jobs

### Check job status
```bash
squeue -u ido.dotan
```

### Detailed job info
```bash
squeue -u ido.dotan -o "%.18i %.20j %.15u %.12T %.10M %.10l %.10L"
```

### View available GPUs
```bash
sinfo -o "%N %G %t" | awk '{if ($3=="idle") {print "\033[1;32m" $0 "\033[0m"} else if ($3=="mix") {print "\033[1;33m" $0 "\033[0m"} else if ($3=="alloc") {print "\033[1;34m" $0 "\033[0m"} else if ($3=="drain") {print "\033[1;31m" $0 "\033[0m"} else {print $0} }'
```

### View job output (live)
```bash
tail -f outputs/cifar10_kernels-<JOB_ID>.txt
```

### Cancel job
```bash
scancel <JOB_ID>
```

## Output Location

Results are saved to:
```
output/CIFAR10/resnet18/YYYY_M_D/<experiment_name>/
```

Each episode directory contains:
- Model checkpoints
- Training logs
- Selected sample indices (lSet, uSet, activeSet)
- Accuracy metrics

## Example Workflows

### Compare RBF vs TopHat kernels
```bash
# Submit both experiments
sbatch --export=KERNEL_TYPE=rbf,EXP_NAME=compare_rbf,MAX_ITER=15 sbatch_cifar10_kernels.sh
sbatch --export=KERNEL_TYPE=tophat,EXP_NAME=compare_tophat,MAX_ITER=15 sbatch_cifar10_kernels.sh

# Monitor
watch -n 5 'squeue -u ido.dotan'
```

### Test different delta values
```bash
for delta in 0.5 0.75 1.0; do
  sbatch --export=INITIAL_DELTA=$delta,EXP_NAME=delta_${delta},MAX_ITER=10 sbatch_cifar10_kernels.sh
done
```

### Array job for multiple seeds
Modify `sbatch_cifar10_kernels.sh` to add:
```bash
#SBATCH --array=1-5
SEED=${SLURM_ARRAY_TASK_ID}
```

Then submit:
```bash
sbatch sbatch_cifar10_kernels.sh
```

## Troubleshooting

### Check if features exist
```bash
ls -lh results/cifar-10/pretext/features_seed1.npy
```

If missing, generate features first:
```bash
cd scan
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
```

### Check virtual environment
```bash
source venv/bin/activate
python -c "import torch; import faiss; print('torch:', torch.__version__, 'faiss: OK')"
```

### Memory issues
If job fails with OOM, increase memory:
```bash
sbatch --mem=64g sbatch_cifar10_kernels.sh
```

## Notes

- First run may take longer (downloading CIFAR-10 dataset)
- RBF kernel is generally more computationally expensive than TopHat
- For large budgets (>1000), consider increasing memory allocation
- Results are reproducible with fixed `SEED` parameter
