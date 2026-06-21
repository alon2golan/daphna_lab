#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --gres=gg:g0:1,vmem:15g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_train_al_%j.out


dir=/cs/labs/daphna/$USER/projects/daphna_lab/
cd $dir
source $dir/venv/bin/activate
cd $dir/deep_al/tools

# Example config file (adjust path as needed)
CFG_PATH="../configs/cifar10/al/RESNET18.yaml"

# Run parameters (adjust as needed)
SEED=0
BUDGET=100
INITIAL_SIZE=100
MAX_ITER=5
EXP_NAME="first_NAL"
INITIAL_DELTA=0.6
KERNEL_TYPE="rbf"
EVAL_MODEL_TYPE="from_images"
MAX_EPOCH=20


# Run the noisy AL pipeline similar to the original training command style
python3 run_noise_al_pipeline.py \
  --cfg $CFG_PATH \
  --noise_type sym \
  --noise_rate 0.2 \
  --al_method typiclust \
  --noise_method label_cleaning \
  --seed ${SEED} \
  --budget ${BUDGET} \
  --initial_size ${INITIAL_SIZE} \
  --max_iter ${MAX_ITER} \
  --exp_name ${EXP_NAME} \
  --initial_delta ${INITIAL_DELTA} \
  --kernel_type ${KERNEL_TYPE} \
  --eval_model_type ${EVAL_MODEL_TYPE} \
  --max_epoch ${MAX_EPOCH} \

