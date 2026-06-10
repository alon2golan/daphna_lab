"""
sweep_repro_4_15_full.py
------------------------
Full reproduction of the original 4_15 sweep

    probcover_random__RESNET18__seed1to3__20260415_153802

AFTER the noise seed-decoupling fix (NOISE_SEED_OFFSET in
deep_al/pycls/noise/annotator.py), for BOTH al strategies.

Why this exists
---------------
The earlier repro (batch_utils/sweep_repro_oracle_vs_noisy.py, job 30726743)
only covered `probcover`. The `random` half of the 4_15 figure was never
re-run post-fix. This launcher mirrors the ORIGINAL 24-job grid exactly:

    al        ∈ {probcover, random}
    tau       ∈ {0.2, 0.5, 0.8, 1.0}        (1.0 == oracle: nu'=0 from cache)
    seed      ∈ {1, 2, 3}
    budget=100, max_iter=10, initial_delta=0.75, from_features

= 4 tau × 2 al × 3 seed = 24 jobs, identical shape to the original.

nu' source ("NoAA-Adam values")
--------------------------------
Each noisy job (tau<1) passes only --tau; train_al.configure_nu_prime_annotator
looks up nu'(al, B=100, tau) from results/cifar100/nu_prime_cache.json. That
cache is the ADAM-estimator output (estimating_delta_nu.py hardcodes Adam),
covering probcover/random/maxherding at B in {100,200,500}, all tau — i.e.
exactly the "NoAA-Adam" values. The cache is unchanged by the annotator fix
(the estimator never used the buggy Annotator). NOTE: no SGD nu' exists for
`random` — the sgd/adam optcompare work was probcover-only.

Usage
-----
    python batch_utils/sweep_repro_4_15_full.py            # dry-run
    python batch_utils/sweep_repro_4_15_full.py --submit   # sbatch (phoenix-gw), %6
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sweep_train_al", _HERE / "sweep_train_al.py")
swt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(swt)

CFG = "deep_al/configs/cifar100/al/RESNET18.yaml"

# Mirror the original 4_15 grid: tau 0.2/0.5/0.8 noisy + tau 1.0 oracle.
AL_CONFIGS = [
    f"--cfg {CFG} --tau 0.2",
    f"--cfg {CFG} --tau 0.5",
    f"--cfg {CFG} --tau 0.8",
    f"--cfg {CFG} --tau 1.0",   # oracle (nu'(tau=1)=0 from cache)
]
GRID = dict(al=["probcover", "random"], seed=[1, 2, 3])
FIXED_ARGS = (
    "--initial_delta 0.75 "
    "--budget 100 "
    "--eval_model_type from_features "
    "--max_iter 10 "
)
THROTTLE = 6  # %6 concurrency cap (lab GPU etiquette, CLAUDE.md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true",
                    help="Write the array script and sbatch it (via phoenix-gw), capped at %%6.")
    args = ap.parse_args()

    sweep_name = f"repro4_15_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jobs = swt.build_jobs_from(AL_CONFIGS, GRID, FIXED_ARGS, sweep_name=sweep_name)

    print(f"\nsweep_name: {sweep_name}")
    print(f"{len(jobs)} jobs ({len(AL_CONFIGS)} tau x {len(GRID['al'])} al x "
          f"{len(GRID['seed'])} seeds):\n")
    for i, j in enumerate(jobs):
        print(f"  [{i:02d}] {j}")

    # Build the array script, then inject the %THROTTLE concurrency cap.
    script = swt.build_array_script(jobs)
    n = len(jobs)
    script = script.replace(f"--array=0-{n - 1}\n", f"--array=0-{n - 1}%{THROTTLE}\n")
    assert f"%{THROTTLE}" in script, "throttle injection failed"

    out_path = _HERE / "_generated_sweep_repro_4_15_full.sh"
    out_path.write_text(script)
    print(f"\nGenerated (array=0-{n-1}%{THROTTLE}): {out_path}")

    if not args.submit:
        print("\nDry-run. Add --submit to send to SLURM.")
        return

    submit_cmd = f"SLURM_CONF={swt.SLURM_CONF} {swt.SBATCH_BIN} {out_path}"
    res = subprocess.run(["ssh", swt.SLURM_LOGIN_NODE, submit_cmd],
                         capture_output=True, text=True)
    if res.returncode == 0:
        print(f"\nSubmitted: {res.stdout.strip()}")
    else:
        print(f"\nsbatch failed (rc={res.returncode}):\n{res.stderr.strip()}")


if __name__ == "__main__":
    main()
