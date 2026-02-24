"""
Submit a sweep of train_al.py jobs as a single SLURM array job.

Design
------
AL_CONFIGS   – list of verbatim al-specific arg strings.
               Each entry is one "experiment variant"; they are NEVER
               crossed with each other.
               Example:
                   AL_CONFIGS = [
                       "--al maxherding --sigma 0.5",
                       "--al probcover --initial_delta 1.0",
                   ]

GRID         – dict of params that ARE fully crossed with every AL_CONFIG
               (and with each other).
               Example:
                   GRID = dict(seed=[1, 2, 3])

The final job list = itertools.product(AL_CONFIGS, *GRID.values()),
Each job runs one AL_CONFIG with one combination of GRID values.

FIXED_ARGS   – args added verbatim to every job (cfg, budget, etc.).

Usage
-----
    python batch_utils/sweep_train_al.py              # dry-run: print script
    python batch_utils/sweep_train_al.py --submit     # write .sh + sbatch it
"""

import itertools
import subprocess
import argparse
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Repo / environment paths
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parents[1]
VENV_ACTIVATE = REPO_DIR / "venv" / "bin" / "activate"
TRAIN_SCRIPT = REPO_DIR / "deep_al" / "tools" / "train_al.py"
LOG_DIR = REPO_DIR / "logs" / "sweep"
GENERATED_SCRIPT = REPO_DIR / "batch_utils" / "_generated_sweep.sh"

# ---------------------------------------------------------------------------
# SLURM resource settings
# ---------------------------------------------------------------------------
def _git_user_email() -> str:
    """Read the email from git config user.email, falling back to USER@mail.huji.ac.il."""
    try:
        result = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        user = os.environ.get("USER", "unknown")
        return f"{user}@mail.huji.ac.il"


SLURM_OPTS = {
    "mem": "32g",
    "cpus-per-task": "4",
    "time": "24:00:00",
    "gres": "gg:g0:1,vmem:20g",
     "mail-user": _git_user_email(),
    "mail-type": "FAIL,TIME_LIMIT",
    "output": str(LOG_DIR / "sweep_%A_%a.out"),
}

# ---------------------------------------------------------------------------
# Fixed args added to EVERY job  (edit as needed)
# NOTE: --exp-name is auto-generated from the job parameters; do not set it here.
# ---------------------------------------------------------------------------
FIXED_ARGS = (
    f"--cfg deep_al/configs/cifar100/al/RESNET18.yaml "
    f"--budget 100 "
    f"--eval_model_type from_features "
    f"--max_iter 5 "
)

# ---------------------------------------------------------------------------
# AL_CONFIGS  – each entry is one experiment variant (NOT crossed internally)
# ---------------------------------------------------------------------------
AL_CONFIGS = [
    "--al maxherding",
    "--al maxherding --sparse_K",
    "--al probcover",
]

# ---------------------------------------------------------------------------
# GRID  – params fully crossed with every AL_CONFIG
# ---------------------------------------------------------------------------
GRID = dict(
    initial_delta=[0.75, 1],
    seed=[1, 2, 3],
)

# ---------------------------------------------------------------------------


def args_to_exp_name(al_args: str, grid_keys: list, grid_vals: tuple) -> str:
    """
    Build a readable exp-name from the al_args string and grid values.

    "--al maxherding --sigma 0.5"  +  seed=2
        →  "maxherding_sigma0.5_seed2"

    Rules:
    - Flag names starting with '--' become the key; the next token is the value.
    - Single-token flags (no value, e.g. --finetune) become just the flag name.
    - Leading '--' and internal '-' / '.' are stripped/replaced for readability.
    """
    tokens = al_args.split()
    parts = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok.lstrip("-").replace("-", "_")
            # peek at next token: is it a value or another flag?
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                val = tokens[i + 1].replace(".", "p")  # 0.5 -> 0p5
                parts.append(f"{key}{val}")
                i += 2
            else:
                parts.append(key)
                i += 1
        else:
            i += 1

    for k, v in zip(grid_keys, grid_vals):
        str_v = str(v).replace(".", "p")
        parts.append(f"{k}{str_v}")

    return "_".join(parts)


def build_jobs() -> list[str]:
    """Return list of complete python command strings, one per job."""
    grid_keys = list(GRID.keys())
    grid_combos = list(itertools.product(*GRID.values()))

    jobs = []
    for al_args in AL_CONFIGS:
        for grid_vals in grid_combos:
            grid_str = " ".join(
                f"--{k} {v}" for k, v in zip(grid_keys, grid_vals)
            )
            exp_name = args_to_exp_name(al_args, grid_keys, grid_vals)
            cmd = f"python {TRAIN_SCRIPT} {FIXED_ARGS} {al_args} {grid_str} --exp-name {exp_name}"
            jobs.append(cmd)
    return jobs


def build_array_script(jobs: list[str]) -> str:
    """Generate the bash array script content."""
    n = len(jobs)
    job_lines = "\n".join(f'JOBS[{i}]="{jobs[i]}"' for i in range(n))

    header_lines = "\n".join(
        f"#SBATCH --{k}={v}" for k, v in SLURM_OPTS.items()
    )

    return (
        f"#!/bin/bash\n"
        f"#SBATCH --array=0-{n - 1}\n"
        f"{header_lines}\n"
        f"\n"
        f"# ---- job table (auto-generated by sweep_train_al.py) ----\n"
        f"declare -A JOBS\n"
        f"{job_lines}\n"
        f"\n"
        f"CMD=${{JOBS[$SLURM_ARRAY_TASK_ID]}}\n"
        f"echo \"Task $SLURM_ARRAY_TASK_ID: $CMD\"\n"
        f"\n"
        f"cd {REPO_DIR}\n"
        f"source {VENV_ACTIVATE}\n"
        f"eval \"$CMD\"\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Write the generated .sh script and submit it (default: dry-run)",
    )
    args = parser.parse_args()

    jobs = build_jobs()
    script = build_array_script(jobs)

    print(f"AL_CONFIGS ({len(AL_CONFIGS)} variants):")
    for c in AL_CONFIGS:
        print(f"  {c}")
    print(f"\nGRID: {GRID}")
    print(f"\nTotal jobs: {len(jobs)}\n")
    print("--- Generated script ---")
    print(script)
    print("------------------------")

    if args.submit:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        GENERATED_SCRIPT.write_text(script)
        print(f"Written: {GENERATED_SCRIPT}")
        result = subprocess.run(["sbatch", str(GENERATED_SCRIPT)], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Submitted: {result.stdout.strip()}")
        else:
            print(f"ERROR: {result.stderr.strip()}")
    else:
        print("Dry-run complete. Pass --submit to write + submit the script.")


if __name__ == "__main__":
    main()
