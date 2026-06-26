#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 6
#SBATCH --time=6:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_grid_noise_al_%j.out

# Move to repo and activate venv
repo_dir=/cs/labs/daphna/$USER/projects/daphna_lab
cd $repo_dir
source $repo_dir/venv/bin/activate
cd $repo_dir/deep_al/tools

# Config and run parameters
CFG_PATH="../configs/cifar10/al/RESNET18.yaml"
SEED=0
BUDGET=100
INITIAL_SIZE=100
MAX_ITER=4
EXP_NAME="grid_NAL_full"
INITIAL_DELTA=0.6
KERNEL_TYPE="rbf"
EVAL_MODEL_TYPE="from_images"
MAX_EPOCH=50
WARMUP_EPISODE=3
WARMUP_MIN_ACC=30.0

# Full grid (noise types, AL methods, noise handling methods)
# Noise types: symmetric, asymmetric, CIFAR-10N, CIFAR-100N
GRID_NOISE_TYPES="sym,asym,cifar10n,cifar100n"
# AL methods: typiclust, probcover, margin, dcom, maxhearding
GRID_AL_METHODS="typiclust,probcover,margin,dcom,maxhearding"
# Noise handling methods: label_cleaning, aum, crossval, dividemix
GRID_NOISE_METHODS="label_cleaning,aum,crossval,dividemix"

# CIFAR-N variant to use when CIFAR-N noise selected (auto/aggre_label/worst_label/clean_label/random_label1/...)
CIFAR_N_VARIANT="auto"

# Output directory for grid results
GRID_OUT_DIR="$repo_dir/grid_outputs/full_NAL_grid"
mkdir -p "$GRID_OUT_DIR"

# Run grid (one python call will iterate over Cartesian product)
python3 run_noise_al_pipeline.py \
  --cfg "$CFG_PATH" \
  --grid_noise_types "$GRID_NOISE_TYPES" \
  --grid_al_methods "$GRID_AL_METHODS" \
  --grid_noise_methods "$GRID_NOISE_METHODS" \
  --grid_out_dir "$GRID_OUT_DIR" \
  --noise_rate 0.2 \
  --cifar_n_variant "$CIFAR_N_VARIANT" \
  --seed ${SEED} \
  --budget ${BUDGET} \
  --initial_size ${INITIAL_SIZE} \
  --max_iter ${MAX_ITER} \
  --exp_name ${EXP_NAME} \
  --initial_delta ${INITIAL_DELTA} \
  --kernel_type ${KERNEL_TYPE} \
  --eval_model_type ${EVAL_MODEL_TYPE} \
  --max_epoch ${MAX_EPOCH} \
  --warmup_episode ${WARMUP_EPISODE} \
  --warmup_min_acc ${WARMUP_MIN_ACC}
