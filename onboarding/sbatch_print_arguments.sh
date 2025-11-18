#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 2
#SBATCH --time=2:00:00
#SBATCH --array=1-3
#SBATCH --output=example_print_arguments_%j.out

dir=/cs/labs/daphna/$USER/projects/daphna_lab/
cd $dir
source $dir/venv/bin/activate
cd $dir/onboarding/
python example_print_arguments.py $SLURM_ARRAY_TASK_ID