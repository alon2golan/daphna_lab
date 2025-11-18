#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --gres=gg:g0:1,vmem:15g
#SBATCH --mail-user=alon.golan@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_train_al_%j.out

dir=/cs/labs/daphna/$USER/projects/TypiClust/
cd $dir
source $dir/venv/bin/activate
cd $dir/deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_rp --exp-name auto --initial_size 0 --budget 100 