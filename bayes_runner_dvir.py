import sys
import numpy as np
import itertools
import math

# ==============================================================================
# 1. EXPERIMENT CONFIGURATION
# ==============================================================================

# --- Global Settings ---
NUMBER_OF_REPS = 1 # Number of times each unique config is run (for seeding)
MAX_CMDS_PER_JOB = 1 # Fixed: One command executed per SLURM array job instance
EXP_NAME = "adaptive_prior_true"

# --- Hyperparameter Space ---
DATASETS = ["cifar10"]  # Options: "cifar10", "cifar100", "half_moon", "scenario_a"
AL_METHODS = ["bayes_misp"]
EVAL_MODELS = ["from_images"] # Use "from_images", "from_features", "NN1"

# ALPHA_VALUES = [0.001, 0.01, 0.1, 0.2, 0.5]
ALPHA_VALUES = [0.00001, 0.000001]
alpha_pairs = [(0.00, 0.00)]
# alpha_pairs = [(0.0001, 0.001),(0.0001, 0.005)]
# DIFF_METHODS options: "full_weighted_max", "top2_weighted_max", "max", "weighted_sum"
DIFF_METHODS = ["full_weighted_max"]
CONT_METHODS = ["positive"]

# --- Kernel Settings ---
KERNEL_TYPE = "rbf"
RBF_DELTA = [0.75]  # Initial delta/sigma value for RBF kernel

# --- Utility Maps ---
EVAL_FLAG_MAP = {
    "from_features": "from_features",
    "from_images": "from_images",
    "NN1": "NN1"
}

MAX_ITER = 20

BUDGET_MAP = {
    "cifar10": 20,
    "cifar100": 100,
    "half_moon": 2,
    "scenario_a": 10,
}

TEMPLATE = (
    "python ./train_al.py "
    "--cfg ../configs/{dataset}/al/RESNET18.yaml "
    "--al {al_method} "
    "--exp-name {exp_name} "
    "--initial_size 0 "
    "--budget {budget} "
    "--initial_delta {initial_delta} "
    "--eval_model {eval_model_flag} "
    "--kernel_type {kernel_type} "
    "--diff_method {diff_method} "
    "--alpha {alpha} "
    "--alpha_lower {alpha_lower} "
    "--alpha_upper {alpha_upper} "
    "--cont_method {cont_method} "
    "--seed {seed} "  # Seed is added dynamically in the execution logic
    "--max_iter {max_iter}"
)

# ==============================================================================
# 2. GENERATE UNIQUE COMMANDS (The core hyperparameter sweep)
# ==============================================================================

unique_commands = []

# Iteratively build the combination of parameters
param_combinations = itertools.product(
    DATASETS, EVAL_MODELS, AL_METHODS, ALPHA_VALUES, RBF_DELTA, alpha_pairs, DIFF_METHODS, CONT_METHODS
)

for dataset, eval_model, al_method, alpha, delta, (alpha_lower, alpha_upper), diff_method, cont in param_combinations:

    # --- Exclusion Logic (Rule 1) ---
    # Skip invalid combinations: top2/max difference method cannot be used with 
    # regularization-sum contribution methods.
    is_diff_max = diff_method in ["top2_weighted_max", "max"]
    is_cont_reg_sum = cont.startswith("reg_sum")
    if is_diff_max and is_cont_reg_sum:
        continue

    # --- Parameter Setting ---
    flag = EVAL_FLAG_MAP[eval_model]
    budget = BUDGET_MAP[dataset]


    # --- Format and Save Command (without final seed) ---
    cmd = TEMPLATE.format(
        dataset=dataset,
        exp_name=EXP_NAME,
        al_method=al_method,
        eval_model_flag=flag,
        initial_delta=delta,
        diff_method=diff_method,
        alpha=alpha,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        budget=budget,
        kernel_type=KERNEL_TYPE,
        cont_method=cont,
        seed="{seed}", # Placeholder for the final seed
        max_iter=MAX_ITER
    )
    unique_commands.append(cmd)


# ==============================================================================
# 3. JOB COUNTING AND EXECUTION LOGIC
# ==============================================================================

NUM_UNIQUE_CMDS = len(unique_commands)
TOTAL_COMMANDS_COUNT = NUM_UNIQUE_CMDS * NUMBER_OF_REPS
TOTAL_JOBS = TOTAL_COMMANDS_COUNT # Since MAX_CMDS_PER_JOB = 1

# Check for the special 'TOTAL' argument used by the sbatch script to size the array
if len(sys.argv) > 1 and sys.argv[1] == "TOTAL":
    # Print the total number of jobs required and exit cleanly.
    print(TOTAL_JOBS)
    sys.exit(0)

# The SLURM Array Job Index is passed as the first argument, starting from 1.
try:
    # job_id is 0-indexed version of the SLURM array index ($SLURM_ARRAY_TASK_ID)
    job_id = int(sys.argv[1]) - 1
except IndexError:
    sys.stderr.write("Error: Please provide the job array index (e.g., sys.argv[1])\n")
    sys.exit(1)
except ValueError:
    sys.stderr.write("Error: Job array index must be an integer.\n")
    sys.exit(1)


# Check if the current job index is within the valid range
if 0 <= job_id < TOTAL_JOBS:
    # 1. Determine which unique command configuration this job should run (wraparound)
    command_index = job_id % NUM_UNIQUE_CMDS
    base_command = unique_commands[command_index]

    # 2. Determine which repetition (seed) this job should run
    # Seed is the integer division of job_id by the number of unique commands
    ordered_seed = job_id // NUM_UNIQUE_CMDS
    
    # 3. Final command creation and printing
    final_command = base_command.format(seed=ordered_seed)
    print(final_command)
# If job_id is out of range, the script quietly exits, as intended for array jobs.