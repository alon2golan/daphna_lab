#!/usr/bin/env python
"""
Verify realized labeled-set noise matches nominal nu' for a set of train_al runs.

Reproduces the annotator's corruption set EXACTLY:
    corrupted = RandomState(seed + OFFSET).choice(N, int(rate*N), replace=False)
and measures the fraction of the actually-selected lSet that landed in it.

The 2026-06-03 bug: OFFSET=0 made the noise RNG collide with cfg.RNG_SEED (the
data-split / selection seed), inflating realized lset noise far above nominal.
The fix set OFFSET=10_000. This script reports realized noise BOTH ways so the
fix is visible: 'fixed' should ~= nominal; 'buggy(offset=0)' shows the old inflation.

Usage:
    python batch_utils/verify_realized_noise.py <exp_dir> [<exp_dir> ...]
Each exp_dir must contain config.yaml + episode_*/lSet.npy + stdout.log.
"""
from __future__ import annotations
import sys, re, glob, os
import numpy as np

N_TRAIN = 50000          # cifar100 full train set (annotator builds is_noisy over all of it)
OFFSET  = 10_000

def final_lset(exp_dir: str) -> np.ndarray:
    eps = []
    for p in glob.glob(os.path.join(exp_dir, "episode_*", "lSet.npy")):
        m = re.search(r"episode_(\d+)", p)
        if m:
            eps.append((int(m.group(1)), p))
    if not eps:
        raise FileNotFoundError(f"no episode_*/lSet.npy in {exp_dir}")
    _, p = max(eps)
    return np.asarray(np.load(p, allow_pickle=True)).astype(int)

def nominal_rate(exp_dir: str) -> float:
    """nu' actually used = 1 - annotator accuracy (printed every round in stdout.log)."""
    txt = open(os.path.join(exp_dir, "stdout.log")).read()
    accs = re.findall(r"'annotator_accuracies':\s*\[([0-9.]+)\]", txt)
    if not accs:
        raise ValueError(f"no annotator_accuracies in {exp_dir}/stdout.log")
    return 1.0 - float(accs[-1])

def rng_seed(exp_dir: str) -> int:
    txt = open(os.path.join(exp_dir, "config.yaml")).read()
    m = re.search(r"RNG_SEED:\s*(\d+)", txt)
    return int(m.group(1))

def corrupted_set(seed: int, rate: float, offset: int) -> set:
    rng = np.random.RandomState(seed + offset)
    idx = rng.choice(N_TRAIN, int(rate * N_TRAIN), replace=False)
    return set(idx.tolist())

def main():
    dirs = sys.argv[1:]
    print(f"{'seed':>4} {'nom':>6} {'n_lset':>7} {'realized(fixed)':>15} {'buggy(off=0)':>13}  exp")
    for d in dirs:
        try:
            seed = rng_seed(d); rate = nominal_rate(d); lset = final_lset(d)
        except Exception as e:
            print(f"  SKIP {d}: {e}"); continue
        if rate <= 0:
            print(f"{seed:>4} {rate:>6.3f} {len(lset):>7}  (oracle, no noise) -> realized 0  {os.path.basename(d)[:50]}")
            continue
        cf = corrupted_set(seed, rate, OFFSET)
        cb = corrupted_set(seed, rate, 0)
        rf = np.mean([i in cf for i in lset])
        rb = np.mean([i in cb for i in lset])
        flag = "  <-- OFF by >5pp" if abs(rf - rate) > 0.05 else ""
        print(f"{seed:>4} {rate:>6.3f} {len(lset):>7} {rf:>15.3f} {rb:>13.3f}  {os.path.basename(d)[:46]}{flag}")

if __name__ == "__main__":
    main()
