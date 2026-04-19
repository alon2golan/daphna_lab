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
from datetime import datetime
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


SLURM_LOGIN_NODE = "phoenix-gw"
SBATCH_BIN = "/vol/slurm/phoenix/bindir/bin/sbatch"
SLURM_CONF = "/vol/slurm/phoenix/slurm.conf"

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
    f"--initial_delta 0.75 "
    f"--budget 100 "
    f"--eval_model_type from_features "
    f"--max_iter 10 "
)

# ---------------------------------------------------------------------------
# AL_CONFIGS  – each entry is one experiment variant (NOT crossed internally)
# target_nu=0.0   → oracle-equivalent (tau=1.0, 0% noise via cache lookup)
# target_nu=0.1-0.5 → noisy annotator at the named noise level; tau looked up from cache
# ---------------------------------------------------------------------------
AL_CONFIGS = [
    "--cfg deep_al/configs/cifar100/al/RESNET18.yaml --target_nu 0.0",   # oracle (tau=1)
    "--cfg deep_al/configs/cifar100/al/RESNET18.yaml --target_nu 0.1",   # low noise
    "--cfg deep_al/configs/cifar100/al/RESNET18.yaml --target_nu 0.2",
    "--cfg deep_al/configs/cifar100/al/RESNET18.yaml --target_nu 0.3",
    "--cfg deep_al/configs/cifar100/al/RESNET18.yaml --target_nu 0.5",   # high noise
]

# ---------------------------------------------------------------------------
# GRID  – params fully crossed with every AL_CONFIG
# ---------------------------------------------------------------------------
GRID = dict(
    al=["probcover", "random"],
    seed=[1, 2, 3],
)

# ---------------------------------------------------------------------------


def build_sweep_name_from(al_configs: list, grid: dict) -> str:
    """
    Build a human-readable sweep identifier from arbitrary al_configs + grid.
    Used both by build_sweep_name() and by the dashboard Launch tab.
    """
    # Collect al methods from AL_CONFIGS entries
    al_methods = []
    for al_args in al_configs:
        tokens = al_args.split()
        for i, tok in enumerate(tokens):
            if tok == "--al" and i + 1 < len(tokens):
                al_methods.append(tokens[i + 1])
                break
    # Also check if 'al' is a grid key (e.g. al=["probcover","random"])
    if not al_methods and "al" in grid:
        al_methods = [str(v) for v in grid["al"]]
    al_part = "_".join(dict.fromkeys(al_methods)) or "sweep"

    # cfg variant: use stem of --cfg values to distinguish configs
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
            continue  # already captured in al_part
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


def build_sweep_name() -> str:
    """
    Generate a human-readable sweep identifier that encodes the AL methods, GRID
    ranges, and a wall-clock timestamp so every submission is unique.

    Example: "maxherding_probcover__delta0p75to1_seed1to3__20260406_143000"
    """
    return build_sweep_name_from(AL_CONFIGS, GRID)


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
                raw_val = tokens[i + 1]
                # If value looks like a file path, use only the stem (no dirs, no ext)
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

    return "_".join(parts)


def build_jobs_from(
    al_configs: list,
    grid: dict,
    fixed_args: str,
    sweep_name: str | None = None,
) -> list[str]:
    """
    Parametrised job builder — used by build_jobs() and the dashboard Launch tab.
    If sweep_name is None, one is auto-generated from al_configs + grid.
    """
    if sweep_name is None:
        sweep_name = build_sweep_name_from(al_configs, grid)
    grid_keys = list(grid.keys())
    grid_combos = list(itertools.product(*grid.values()))
    jobs = []
    for al_args in al_configs:
        for grid_vals in grid_combos:
            grid_str = " ".join(
                f"--{k} {v}" for k, v in zip(grid_keys, grid_vals)
            )
            exp_name = args_to_exp_name(al_args, grid_keys, grid_vals)
            cmd = (
                f"python {TRAIN_SCRIPT} {fixed_args.strip()} {al_args} {grid_str}"
                f" --exp-name {exp_name} --sweep-name {sweep_name}"
            )
            jobs.append(cmd)
    return jobs


def build_jobs() -> list[str]:
    """Return list of complete python command strings, one per job."""
    return build_jobs_from(AL_CONFIGS, GRID, FIXED_ARGS)


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


def launch_local_bg(jobs: list[str], sweep_name: str) -> dict:
    """
    Spawn all jobs as a single detached background bash script (non-blocking).
    Used by the dashboard Launch tab so the UI stays responsive.

    Returns a dict with: pid, log_file, script_file, sweep_id, sweep_name,
    total_jobs, started_at, mode='local'.
    """
    sweep_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{sweep_id}.out"
    script_path = LOG_DIR / f"{sweep_id}.sh"

    lines = [
        "#!/bin/bash",
        f"export SLURM_ARRAY_JOB_ID={sweep_id}",
        f"cd {REPO_DIR}",
        f"source {VENV_ACTIVATE}",
        "",
    ]
    for i, cmd in enumerate(jobs):
        lines += [
            f"export SLURM_ARRAY_TASK_ID={i}",
            f'echo "### Task {i + 1}/{len(jobs)}: {cmd}"',
            cmd,
            f'echo "### Task {i + 1} exit: $?"',
            "",
        ]
    lines.append(f'echo "### All {len(jobs)} jobs finished."')

    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            ["bash", str(script_path)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach from the Streamlit process
        )

    return {
        "pid": proc.pid,
        "log_file": str(log_file),
        "script_file": str(script_path),
        "sweep_id": sweep_id,
        "sweep_name": sweep_name,
        "total_jobs": len(jobs),
        "started_at": datetime.now().isoformat(),
        "mode": "local",
    }


def submit_sbatch_jobs(jobs: list[str]) -> dict:
    """
    Write the array script and submit via sbatch (blocking, returns quickly).
    Returns a dict with: returncode, stdout, stderr, script_file.
    Note: sbatch must be available — run from phoenix-gw or any SLURM login node.
    """
    script = build_array_script(jobs)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_SCRIPT.write_text(script)
    try:
        result = subprocess.run(
            ["sbatch", str(GENERATED_SCRIPT)], capture_output=True, text=True
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "script_file": str(GENERATED_SCRIPT),
        }
    except FileNotFoundError:
        # sbatch not available locally — try via SSH to the SLURM login node.
        # Requires passwordless SSH (key-based) to SLURM_LOGIN_NODE.
        try:
            result = subprocess.run(
                [
                    "ssh", "-o", "BatchMode=yes",
                    "-o", "ConnectTimeout=10",
                    SLURM_LOGIN_NODE,
                    f"SLURM_CONF={SLURM_CONF}", SBATCH_BIN, str(GENERATED_SCRIPT),
                ],
                capture_output=True, text=True,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "script_file": str(GENERATED_SCRIPT),
            }
        except FileNotFoundError:
            return {
                "returncode": 127,
                "stdout": "",
                "stderr": (
                    "'ssh' not found. Cannot reach SLURM login node automatically."
                ),
                "script_file": str(GENERATED_SCRIPT),
            }
        except Exception as exc:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": (
                    f"SSH to {SLURM_LOGIN_NODE} failed: {exc}\n"
                    f"Make sure passwordless SSH is set up, then run manually:\n"
                    f"  ssh {SLURM_LOGIN_NODE} sbatch {GENERATED_SCRIPT}"
                ),
                "script_file": str(GENERATED_SCRIPT),
            }


def run_local(jobs: list[str]) -> None:
    """
    Execute all jobs sequentially on the current machine (no SLURM required).

    A synthetic SLURM_ARRAY_JOB_ID is injected into each process environment so
    that ResultWriter groups all runs into the same sweep, identical to a real
    sbatch array job.  SLURM_ARRAY_TASK_ID is set to the job index.
    """
    sweep_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env = os.environ.copy()
    env["SLURM_ARRAY_JOB_ID"] = sweep_id

    print(f"\nRunning {len(jobs)} jobs locally (sweep_id={sweep_id})")
    print("Jobs will run sequentially in this terminal.\n")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for task_id, cmd in enumerate(jobs):
        env["SLURM_ARRAY_TASK_ID"] = str(task_id)
        log_file = LOG_DIR / f"local_{sweep_id}_{task_id}.out"
        print(f"\n{'='*60}")
        print(f"Task {task_id + 1}/{len(jobs)}: {cmd}")
        print(f"Log: {log_file}")
        print(f"{'='*60}")
        with open(log_file, "w") as lf:
            result = subprocess.run(
                f"cd {REPO_DIR} && source {VENV_ACTIVATE} && {cmd}",
                shell=True,
                env=env,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            lf.write(result.stdout.decode(errors="replace"))
        status = "✓ OK" if result.returncode == 0 else f"✗ FAILED (exit {result.returncode})"
        print(f"Task {task_id}: {status}")

    print(f"\n{'='*60}")
    print(f"All {len(jobs)} local jobs finished.")
    print(f"Logs: {LOG_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Submit or run a sweep of train_al.py jobs.\n"
            "\n"
            "Modes:\n"
            "  (default)      Dry-run: print the generated script\n"
            "  --submit       Write _generated_sweep.sh and sbatch it  (run from phoenix-gw)\n"
            "  --local        Run all jobs sequentially on THIS machine (run from takara)\n"
            "\n"
            "Sweep variants (override the hardcoded AL_CONFIGS):\n"
            "  --tau 0.2 0.5 0.8 1.0        → one config per tau value\n"
            "  --target_nu 0.0 0.1 0.2 0.5  → one config per target_nu value\n"
            "If neither is given, the hardcoded AL_CONFIGS list is used.\n"
            "\n"
            "Grid overrides:\n"
            "  --al probcover random maxherding\n"
            "  --seed 1 2 3\n"
            "  --budget 100\n"
            "  --cfg deep_al/configs/cifar100/al/RESNET18.yaml\n"
            "\n"
            "Examples:\n"
            "  # tau sweep, probcover + random, 3 seeds\n"
            "  python batch_utils/sweep_train_al.py --tau 0.2 0.5 0.8 1.0 --submit\n"
            "\n"
            "  # target_nu sweep, only maxherding\n"
            "  python batch_utils/sweep_train_al.py --target_nu 0.0 0.1 0.2 0.3 0.5 --al maxherding --submit\n"
            "\n"
            "  # larger budget\n"
            "  python batch_utils/sweep_train_al.py --target_nu 0.0 0.3 --budget 200 --submit\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--submit", action="store_true",
                        help="Write the generated .sh script and submit via sbatch (from phoenix-gw)")
    parser.add_argument("--local",  action="store_true",
                        help="Run all jobs sequentially on the current machine (from takara)")

    # ── sweep variant (mutually exclusive) ────────────────────────────────
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--tau",       nargs="+", type=float, metavar="TAU",
                         help="Run one config per tau value (e.g. --tau 0.2 0.5 0.8 1.0)")
    variant.add_argument("--target_nu", nargs="+", type=float, metavar="NU",
                         help="Run one config per target_nu value (e.g. --target_nu 0.0 0.1 0.2 0.5)")

    # ── grid overrides ─────────────────────────────────────────────────────
    parser.add_argument("--al",     nargs="+", default=None,
                        metavar="METHOD",
                        help=f"AL methods to sweep (default: {GRID['al']})")
    parser.add_argument("--seed",   nargs="+", type=int, default=None,
                        metavar="SEED",
                        help=f"Seeds to sweep (default: {GRID['seed']})")
    parser.add_argument("--budget", type=int, default=None,
                        help="Budget (default: taken from FIXED_ARGS)")
    parser.add_argument("--cfg",    default=None,
                        help="Config yaml path (default: deep_al/configs/cifar100/al/RESNET18.yaml)")

    args = parser.parse_args()

    if args.submit and args.local:
        parser.error("--submit and --local are mutually exclusive.")

    # ── Build AL_CONFIGS from CLI or fall back to hardcoded ────────────────
    cfg_path = args.cfg or "deep_al/configs/cifar100/al/RESNET18.yaml"

    if args.tau is not None:
        al_configs = [
            f"--cfg {cfg_path} --tau {t}"
            for t in args.tau
        ]
    elif args.target_nu is not None:
        al_configs = [
            f"--cfg {cfg_path} --target_nu {nu}"
            for nu in args.target_nu
        ]
    else:
        al_configs = AL_CONFIGS  # hardcoded default

    # ── Build GRID from CLI or fall back to hardcoded ──────────────────────
    grid = dict(GRID)  # copy
    if args.al is not None:
        grid["al"] = args.al
    if args.seed is not None:
        grid["seed"] = args.seed

    # ── Build FIXED_ARGS with optional budget override ─────────────────────
    fixed_args = FIXED_ARGS
    if args.budget is not None:
        # replace --budget N in fixed_args
        import re
        fixed_args = re.sub(r"--budget\s+\S+", f"--budget {args.budget}", fixed_args)
        if "--budget" not in fixed_args:
            fixed_args += f" --budget {args.budget}"

    # ── Print plan ─────────────────────────────────────────────────────────
    jobs = build_jobs_from(al_configs, grid, fixed_args)
    script = build_array_script(jobs)

    print(f"\nAL_CONFIGS ({len(al_configs)} variants):")
    for c in al_configs:
        print(f"  {c}")
    print(f"\nGRID: {grid}")
    print(f"FIXED_ARGS: {fixed_args.strip()}")
    print(f"\nTotal jobs: {len(jobs)}\n")

    if args.local:
        run_local(jobs)
    elif args.submit:
        print("--- Generated script ---")
        print(script)
        print("------------------------")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        GENERATED_SCRIPT.write_text(script)
        print(f"Written: {GENERATED_SCRIPT}")
        submit_cmd = f"SLURM_CONF={SLURM_CONF} {SBATCH_BIN} {GENERATED_SCRIPT}"
        ssh_cmd = ["ssh", SLURM_LOGIN_NODE, submit_cmd]
        print(f"Submitting: {' '.join(ssh_cmd)}")
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Submitted: {result.stdout.strip()}")
        else:
            print(f"ERROR: {result.stderr.strip()}")
    else:
        print("--- Generated script ---")
        print(script)
        print("------------------------")
        print("Dry-run complete.  Pass --submit (phoenix-gw) or --local (takara) to run.")


if __name__ == "__main__":
    main()

