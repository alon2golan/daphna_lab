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
# working: #python train_al.py  --cfg ../configs/cifar100/al/RESNET18.yaml   --exp-name auto   --al "typiclust_rp" --kernel_type rbf --max_iter 3 --initial_delta 0.6 --seed 0 --eval_model_type from_images --initial_size 0 --budget 100
python train_al.py  --cfg ../configs/cifar100/al/RESNET18.yaml   --exp-name auto   --al "maxherding_orig" --kernel_type rbf --max_iter 3 --initial_delta 0.6 --seed 0 --eval_model_type from_images --initial_size 0 --budget 100
