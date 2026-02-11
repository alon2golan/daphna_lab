#!/bin/bash
## Memory- doesnt really matter(take extra)
#SBATCH --mem=1g
## Amount of CPU(has to be even) doesn't really matter there are a lot
#SBATCH -c 8
## Time - take extra but not too much because it lowers it's priority
#SBATCH --time=1:00:00
## What GPU to take: can choose groups and not specific ones(like g4), more info with the sinfo command
#SBATCH --gres=gg:g0:1
## You can choose specific nodes if you want(different memory sizes for example)(Currently commented out)
##SBATCH --nodelist=arion-[01-02],binky-[01-05],drape-[01,03],firth-02,hasufel-[01-02],incitatus-[01-02]
## How much runs to do(each one gets different ID)
#SBATCH --array=0-3
## If not killable it runs only on daphna lab(Make sure to select the correct GPU)
##SBATCH --killable
## The path to the output file
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/output-%a-%A.txt
## the lab has gpu:rtx2080:1 (creek, dumfries), gpu:a10:1 (arion), gpu:rtx6000:1 (firth), use vmem:10g
## you can use --requeue for killable tasks
## ssh ido.dotan@phoenix.cs.huji.ac.il
## slurm information https://wiki.rcs.huji.ac.il/en/hurcs/guides/slurm
## squeue -u ido.dotan -o  "%.18i %.20j %.15u %.12T %.10M %.10l %.10L"
## sinfo -o "%N %G %t" | awk '{if ($3=="idle") {print "\033[1;32m" $0 "\033[0m"} else if ($3=="mix") {print "\033[1;33m" $0 "\033[0m"} else if ($3=="alloc") {print "\033[1;34m" $0 "\033[0m"} else if ($3=="drain") {print "\033[1;31m" $0 "\033[0m"} else {print $0} }'

dir=/cs/labs/daphna/ido.dotan/projects/daphna_lab/run_dir

cd $dir

source /cs/labs/daphna/ido.dotan/projects/daphna_lab/venv/bin/activate

## NOTE: THE RESULTS MAY BE A LITTLE LOWER BECAUSE WE DON'T CHOOSE THE BEST MODEL FROM THE VALIDATION SET

## NOTE 2: THE RESULTS MAY BE A LITTLE LOWER BECAUSE OF TRAIN_CE TO CONVERGENCE

python hey.py \
--job_id ${SLURM_ARRAY_TASK_ID} \
--job_name ${SLURM_ARRAY_JOB_ID}

