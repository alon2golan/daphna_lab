"""
Submit a sweep of generate_lsets_parquet_worker.py jobs as a single SLURM
array job.

Design mirrors sweep_train_al.py exactly:
  AL_CONFIGS × GRID  →  one job per combination.

Grid:
  al     = [probcover, maxherding]          2 methods
  budget = [100, 200, ..., 1000]           10 budgets
  tau    = [0.1, 0.2, ..., 0.9]            9 tau values
  ─────────────────────────────────────────
  Total  = 2 × 10 × 9 = 180 array jobs

Each job calls generate_lsets_parquet_worker.py for one (al, budget, tau),
generates --n_runs independent oracle + noisy lsets and saves a partial
parquet.  After all jobs finish, run merge_lsets_parquets.py to assemble
the final *_all.parquet files consumed by get_train_idx().

Usage
-----
    python batch_utils/sweep_generate_lsets.py              # dry-run: print script
    python batch_utils/sweep_generate_lsets.py --submit     # write .sh + sbatch it
"""

import itertools
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Repo / environment paths
# ---------------------------------------------------------------------------
REPO_DIR      = Path(__file__).resolve().parents[1]
VENV_ACTIVATE = REPO_DIR / "venv" / "bin" / "activate"
WORKER_SCRIPT = REPO_DIR / "batch_utils" / "generate_lsets_parquet_worker.py"
LOG_DIR       = REPO_DIR / "logs" / "sweep"
GENERATED_SCRIPT = REPO_DIR / "batch_utils" / "_generated_sweep_lsets.sh"

# ---------------------------------------------------------------------------
# SLURM resource settings  (same as sweep_train_al.py)
# ---------------------------------------------------------------------------
def _git_user_email() -> str:
    try:
        result = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        user = os.environ.get("USER", "unknown")
        return f"{user}@mail.huji.ac.il"


SLURM_LOGIN_NODE = "phoenix-gw"
SBATCH_BIN  = "/vol/slurm/phoenix/bindir/bin/sbatch"
SLURM_CONF  = "/vol/slurm/phoenix/slurm.conf"

SLURM_OPTS = {
    "mem":            "32g",
    "cpus-per-task":  "4",
    "time":           "24:00:00",
    "gres":           "gg:g0:1,vmem:20g",
    "mail-user":      _git_user_email(),
    "mail-type":      "FAIL,TIME_LIMIT",
    "output":         str(LOG_DIR / "sweep_lsets_%A_%a.out"),
}

# ---------------------------------------------------------------------------
# Fixed args added to EVERY job
# ---------------------------------------------------------------------------
FIXED_ARGS = (
    "--cfg deep_al/configs/cifar100/al/RESNET18_noisy.yaml "
    "--initial_delta 0.75 "
    "--eval_model_type from_features "
    "--max_iter 10 "
    "--n_runs 5 "
    "--seed 42 "
)

# ---------------------------------------------------------------------------
# AL_CONFIGS  — one entry: just the cfg (already in FIXED_ARGS above so we
# use a single placeholder entry; the actual AL method is in GRID below).
# ---------------------------------------------------------------------------
# We keep a single "empty" AL_CONFIG so the cross-product machinery gives
# exactly 1 × |GRID| = 180 jobs.
AL_CONFIGS = [""]   # no extra per-variant args; everything is in GRID / FIXED_ARGS

# ---------------------------------------------------------------------------
# GRID — fully crossed with AL_CONFIGS
# ---------------------------------------------------------------------------
GRID = dict(
    al=["probcover", "maxherding"],
    budget=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    tau=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
)

# ---------------------------------------------------------------------------
# Helpers  (copied / adapted from sweep_train_al.py)
# ---------------------------------------------------------------------------

def build_sweep_name_from(al_configs: list, grid: dict) -> str:
    al_methods = grid.get("al", [])
    al_part    = "_".join(dict.fromkeys(str(v) for v in al_methods)) or "lsets"

    cfg_stems = []
    for al_args in al_configs:
        tokens = al_args.split()
        for i, tok in enumerate(tokens):
            if tok == "--cfg" and i + 1 < len(tokens):
                cfg_stems.append(Path(tokens[i + 1]).stem)
                break
    cfg_part = "_".join(dict.fromkeys(cfg_stems))

    grid_parts = []
    for k, vals in grid.items():
        if k == "al":
            continue
        if k == "seed":
            grid_parts.append(f"seed{min(vals)}to{max(vals)}")
        else:
            sorted_vals = sorted(set(vals))
            vals_str = "to".join(str(v).replace(".", "p") for v in sorted_vals)
            grid_parts.append(f"{k}{vals_str}")
    grid_part = "_".join(grid_parts)

    name_parts = [p for p in [al_part, cfg_part, grid_part] if p]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "__".join(name_parts) + f"__{ts}"


def args_to_exp_name(al_args: str, grid_keys: list, grid_vals: tuple) -> str:
    tokens = al_args.split()
    parts  = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok.lstrip("-").replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                raw_val = tokens[i + 1]
                if "/" in raw_val:
                    raw_val = Path(raw_val).stem
                val = raw_val.replace(".", "p").replace("-", "_")
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

    return "_".join(parts) if parts else "gen_parquet"


def build_jobs_from(
    al_configs: list,
    grid: dict,
    fixed_args: str,
    sweep_name: str | None = None,
) -> list[str]:
    if sweep_name is None:
        sweep_name = build_sweep_name_from(al_configs, grid)
    grid_keys   = list(grid.keys())
    grid_combos = list(itertools.product(*grid.values()))
    jobs = []
    for al_args in al_configs:
        for grid_vals in grid_combos:
            grid_str = " ".join(
                f"--{k} {v}" for k, v in zip(grid_keys, grid_vals)
            )
            exp_name = args_to_exp_name(al_args, grid_keys, grid_vals)
            # Strip leading/trailing whitespace from al_args (may be empty string)
            al_part  = al_args.strip()
            cmd_parts = [
                f"python {WORKER_SCRIPT}",
                fixed_args.strip(),
            ]
            if al_part:
                cmd_parts.append(al_part)
            cmd_parts += [
                grid_str,
                f"--exp-name {exp_name}",
                f"--sweep-name {sweep_name}",
            ]
            jobs.append(" ".join(cmd_parts))
    return jobs


def build_jobs() -> list[str]:
    return build_jobs_from(AL_CONFIGS, GRID, FIXED_ARGS)


def build_array_script(jobs: list[str]) -> str:
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
        f"# ---- job table (auto-generated by sweep_generate_lsets.py) ----\n"
        f"declare -A JOBS\n"
        f"{job_lines}\n"
        f"\n"
        f"CMD=${{JOBS[$SLURM_ARRAY_TASK_ID]}}\n"
        f'echo "Task $SLURM_ARRAY_TASK_ID: $CMD"\n'
        f"\n"
        f"cd {REPO_DIR}\n"
        f"source {VENV_ACTIVATE}\n"
        f'eval "$CMD"\n'
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_missing_jobs() -> list[str]:
    """Return job command strings only for (al, budget, tau) combos missing a partial file."""
    import glob as _glob

    partial_dir = REPO_DIR / "output" / "CIFAR100" / "partial"
    existing = set(f.name for f in partial_dir.glob("*.parquet")) if partial_dir.exists() else set()

    # Parse FIXED_ARGS once so we can extract --seed
    seed = 42
    tokens = FIXED_ARGS.split()
    for i, tok in enumerate(tokens):
        if tok == "--seed" and i + 1 < len(tokens):
            seed = int(tokens[i + 1])
            break

    missing_jobs = []
    for al in GRID["al"]:
        for budget in GRID["budget"]:
            for tau in GRID["tau"]:
                tau_str = f"{tau:.2f}".replace(".", "p")
                fname = (
                    f"cifar100_resnet18_{al}_sym"
                    f"_tau{tau_str}_budget{budget}_seed{seed}.parquet"
                )
                if fname not in existing:
                    exp_name = f"al{al}_budget{budget}_tau{str(tau).replace('.','p')}"
                    cmd = (
                        f"python {WORKER_SCRIPT} {FIXED_ARGS.strip()}"
                        f" --al {al} --budget {budget} --tau {tau}"
                        f" --exp-name {exp_name}"
                    )
                    missing_jobs.append(cmd)

    return missing_jobs


def main():
    ap = argparse.ArgumentParser(description="Generate lsets parquet SLURM sweep")
    ap.add_argument("--submit", action="store_true",
                    help="Write the script and submit via sbatch (default: dry-run)")
    ap.add_argument("--resubmit-missing", action="store_true",
                    help="Submit only combos whose partial file doesn't exist yet")
    cli = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if cli.resubmit_missing:
        jobs = _find_missing_jobs()
        if not jobs:
            print("All 180 partial files already exist — nothing to resubmit.")
            return
        print(f"# Resubmitting {len(jobs)} missing jobs")
    else:
        jobs = build_jobs()

    script = build_array_script(jobs)

    print(script)
    print(f"\n# Total jobs: {len(jobs)}")

    if cli.submit:
        GENERATED_SCRIPT.write_text(script)
        print(f"\nWrote {GENERATED_SCRIPT}")

        result = subprocess.run(
            ["ssh", SLURM_LOGIN_NODE,
             f"SLURM_CONF={SLURM_CONF} {SBATCH_BIN} {GENERATED_SCRIPT}"],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("sbatch stderr:", result.stderr)
    else:
        print("\n# (dry-run — pass --submit to actually submit)")


if __name__ == "__main__":
    main()
