"""
sweep_repro_4_15_B300nu.py
--------------------------
Same 4_15 AL grid as sweep_repro_4_15_full.py, but the noisy annotator uses the
ν′ estimated at **B=300** (passed explicitly via --nu_prime), while train_al
STILL RUNS at --budget 100. This tests whether a larger-budget ν′ (which is
lower → less injected noise) better fits the multi-episode run, whose labeled
set accumulates well past 100 by ep10.

ν′ source: results/cifar100/nu_prime_cache.json, budget key "300", written by
    sweep_nu_prime.py --methods probcover random --budgets 300 \
        --taus 0.2 0.5 0.8 --seeds 0 1 2 --average
(Adam estimator, same family as the existing B=100/200/500 cache. probcover uses
exact parquet lsets at B=300; random uses the np.random.choice fallback.)

Grid: al∈{probcover,random} × τ∈{0.2,0.5,0.8} (noisy) + τ=1.0 (oracle) × seed∈{1,2,3}
    = 24 jobs. Oracle has no noise → --tau 1.0 (nu'→0 via cache), unaffected by B.

Usage:
    python batch_utils/sweep_repro_4_15_B300nu.py            # dry-run
    python batch_utils/sweep_repro_4_15_B300nu.py --submit   # sbatch (phoenix-gw), %6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent
_spec = importlib.util.spec_from_file_location("sweep_train_al", _HERE / "sweep_train_al.py")
swt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(swt)

CFG = "deep_al/configs/cifar100/al/RESNET18.yaml"
CACHE = REPO / "results" / "cifar100" / "nu_prime_cache.json"
NOISY_TAUS = [0.2, 0.5, 0.8]
SEEDS = [1, 2, 3]
STRATS = ["probcover", "random"]
THROTTLE = 6
FIXED = "--initial_delta 0.75 --budget 100 --eval_model_type from_features --max_iter 10"


def b300_nu(strat: str, tau: float) -> float:
    c = json.load(open(CACHE))
    if "300" not in c.get(strat, {}):
        sys.exit(f"ERROR: no B=300 entry for {strat} in {CACHE}. Run the estimator "
                 f"+ `sweep_nu_prime.py --average --budgets 300 ...` first.")
    key = f"{tau:.4f}"
    val = c[strat]["300"].get(key)
    if val is None:
        sys.exit(f"ERROR: no τ={key} under {strat} B=300 in cache.")
    return float(val)


def build_jobs(sweep_name: str) -> list[str]:
    jobs = []
    for strat in STRATS:
        for tau in NOISY_TAUS:
            nu = b300_nu(strat, tau)
            for seed in SEEDS:
                exp = f"cfgRESNET18_tau{str(tau).replace('.', 'p')}_al{strat}_seed{seed}"
                jobs.append(
                    f"python {swt.TRAIN_SCRIPT} {FIXED} --cfg {CFG} "
                    f"--tau {tau} --nu_prime {nu:.6f} --al {strat} --seed {seed} "
                    f"--exp-name {exp} --sweep-name {sweep_name}")
        # oracle (tau=1.0): no noise; nu'→0 via cache lookup
        for seed in SEEDS:
            exp = f"cfgRESNET18_tau1p0_al{strat}_seed{seed}"
            jobs.append(
                f"python {swt.TRAIN_SCRIPT} {FIXED} --cfg {CFG} "
                f"--tau 1.0 --al {strat} --seed {seed} "
                f"--exp-name {exp} --sweep-name {sweep_name}")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    sweep_name = f"repro4_15_B300nu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jobs = build_jobs(sweep_name)

    print(f"\nsweep_name: {sweep_name}")
    print("B=300 ν′ used (train still at B=100):")
    for strat in STRATS:
        print(f"  {strat:10s}: " + ", ".join(f"τ{t}={b300_nu(strat, t):.4f}" for t in NOISY_TAUS))
    print(f"\n{len(jobs)} jobs:")
    for i, j in enumerate(jobs):
        print(f"  [{i:02d}] {j.split('train_al.py')[1].strip()}")

    script = swt.build_array_script(jobs)
    n = len(jobs)
    script = script.replace(f"--array=0-{n - 1}\n", f"--array=0-{n - 1}%{THROTTLE}\n")
    assert f"%{THROTTLE}" in script
    out = _HERE / "_generated_sweep_repro_4_15_B300nu.sh"
    out.write_text(script)
    print(f"\nGenerated (array=0-{n-1}%{THROTTLE}): {out}")

    if not args.submit:
        print("\nDry-run. Add --submit to send to SLURM.")
        return
    res = subprocess.run(
        ["ssh", swt.SLURM_LOGIN_NODE, f"SLURM_CONF={swt.SLURM_CONF} {swt.SBATCH_BIN} {out}"],
        capture_output=True, text=True)
    print(f"\n{res.stdout.strip() or res.stderr.strip()}")


if __name__ == "__main__":
    main()
