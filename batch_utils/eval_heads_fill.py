#!/usr/bin/env python3
"""Offline 1-NN evaluator fills for the classifier-model comparison (2026-07-16).
CPU-only replay from persisted run artifacts — no SLURM. Two fills:

  1nn_observed  [bayes_misp]  KNeighborsClassifier(1) on unit-L2 SimCLR features
                + the run's OBSERVED labels, from the BAYES_MISP noise-sweep
                linear-head runs (31084429; selection is head-independent).
                Method identical to eval_1nn_all_arms.py / the live USE_1NN path.

  1nn_pseudoall [bayes_product]  the §8.3 Q2 variant (c) that was designed but
                never computed: 1-NN fit on the pseudo-all TRAINING SET —
                lSet relabeled to the labeled-plane product-posterior argmax
                PLUS every covered unlabeled pool point with its full-pool
                posterior argmax (guard (a): prior-only rows are never
                harvested; tophat δ=0.75; same accumulator scheme as
                eval_posterior_quality.py). Answers "what does the pseudo-all
                target set alone buy, with zero training?".

CORRECTNESS GATES (abort before writing anything):
  G1  recompute probcover/bayes_product 1nn_observed cells through this code
      path and require == per_episode.csv (±0.05) — same gate as eval_1nn_fill.
  G2  recompute bayes_product 1nn_postargmax cells (labeled-plane argmax path,
      shared with 1nn_pseudoall) and require == per_episode.csv (±0.05).

Writes: results/cifar100/eval_heads_fill/per_episode.csv
        columns: arm, evaluator, noise, seed, budget, test_acc, src_sweep
Run:    python3 batch_utils/eval_heads_fill.py
Then:   python3 batch_utils/build_results_master.py   (ingests the file)
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

BASE = "/cs/labs/daphna/alongol/projects/daphna_lab"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "batch_utils"))

from eval_1nn_all_arms import (BUDGETS, NOISES, SEEDS, acc_1nn, load_lset_and_observed,
                               load_test, noise_aware_beliefs, posterior_argmax_labels)
from eval_posterior_variants import load_clean_labels, load_features

OUT_DIR = os.path.join(BASE, "results/cifar100/eval_heads_fill")
PER_EPISODE_REF = os.path.join(BASE, "results/cifar100/eval_1nn_all_arms/per_episode.csv")
DELTA = 0.75
C = 100
LOG_EPS = 1e-12
GATE_TOL = 0.05
MISP_SWEEP = "31084429"
PRODUCT_SWEEP = "30978366"


def find_run(pattern, sweep_id):
    hits = []
    for r in glob.glob(os.path.join(BASE, pattern)):
        rj = os.path.join(r, "result.json")
        if not os.path.exists(rj):
            continue
        try:
            if json.load(open(rj)).get("sweep_id") == sweep_id:
                hits.append(r)
        except Exception:
            continue
    assert len(hits) == 1, f"{pattern} @ sweep {sweep_id} -> {len(hits)} matches"
    return hits[0]


def find_product_run(ntag, arm, seed):
    pat = f"output/CIFAR100/resnet18/2026_7_2/cfgBAYESCOV_{ntag}_*_al{arm}_seed{seed}_*"
    runs = sorted(glob.glob(os.path.join(BASE, pat)))
    assert len(runs) == 1, f"{pat} -> {runs}"
    return runs[0]


def pool_indices(run):
    """The 45k train pool (initial lSet ∪ uSet at episode 0) — excludes valSet."""
    ld = lambda p: np.load(os.path.join(run, p), allow_pickle=True).astype(int)
    pool = np.union1d(ld("episode_0/lSet.npy"), ld("episode_0/uSet.npy"))
    assert len(pool) == 45000, f"pool size {len(pool)} != 45000 ({run})"
    return pool


def pseudoall_trajectory(run, ntag, seed, feats_np, test_x, test_y, clean):
    """{B: test_acc} for 1-NN fit on the pseudo-all training set at budget B."""
    lset_full, observed_full, ann = load_lset_and_observed(run, ntag, seed, clean)
    lf = torch.as_tensor(feats_np[lset_full])                      # [1000, 512]
    lab_argmax = posterior_argmax_labels(lf, observed_full, ann)
    _, logb, _ = noise_aware_beliefs(observed_full, ann)           # [1000, C]

    pool = pool_indices(run)
    pos_in_pool = {int(g): i for i, g in enumerate(pool)}
    pf_np = feats_np[pool]                                         # [45000, 512]
    inball = (torch.cdist(torch.as_tensor(pf_np), lf) < DELTA)     # [45000, 1000] bool

    out = {}
    Cpool = torch.full((len(pool), C), float(np.log(1.0 / C)))
    next_fold = 0
    for B in BUDGETS:
        Cpool += inball[:, next_fold:B].float() @ logb[next_fold:B]
        next_fold = B
        covered = inball[:, :B].any(dim=1).numpy()
        covered[[pos_in_pool[int(g)] for g in lset_full[:B]]] = False  # exclude labeled rows
        pseudo_rows = np.where(covered)[0]
        fit_x = np.concatenate([feats_np[lset_full[:B]], pf_np[pseudo_rows]], axis=0)
        fit_y = np.concatenate([lab_argmax[B],
                                Cpool[pseudo_rows].argmax(dim=1).numpy().astype(np.int64)])
        out[B] = acc_1nn(fit_x, fit_y, test_x, test_y)
    return out


def gate(feats, test_x, test_y, clean):
    ref = {}
    for r in csv.DictReader(open(PER_EPISODE_REF)):
        ref[(r["arm"], r["evaluator"], r["noise"], r["seed"], r["budget"])] = float(r["test_acc"])

    print("=== G1: 1nn_observed recompute vs per_episode.csv ===")
    for arm, noise, seed, B in [("probcover", 0.0, 1, 100), ("probcover", 0.2, 2, 500),
                                ("bayes_product", 0.5, 1, 300), ("bayes_product", 0.8, 3, 1000)]:
        ntag = {0.0: "n00", 0.2: "n20", 0.5: "n50", 0.8: "n80"}[noise]
        run = find_product_run(ntag, arm, seed)
        lset_full, observed_full, _ = load_lset_and_observed(run, ntag, seed, clean)
        got = acc_1nn(feats[lset_full[:B]], observed_full[:B], test_x, test_y)
        want = ref[(arm, "1nn_observed", f"{noise:.1f}", str(seed), str(B))]
        ok = abs(got - want) <= GATE_TOL
        print(f"  {arm:14s} n{noise} s{seed} B{B}: got {got:.2f} vs ref {want:.2f}  {'OK' if ok else 'MISMATCH'}")
        assert ok, "G1 FAILED — aborting, nothing written"

    print("=== G2: 1nn_postargmax recompute (labeled-plane argmax path) ===")
    for noise, seed, B in [(0.0, 1, 100), (0.8, 3, 1000)]:
        ntag = {0.0: "n00", 0.8: "n80"}[noise]
        run = find_product_run(ntag, "bayes_product", seed)
        lset_full, observed_full, ann = load_lset_and_observed(run, ntag, seed, clean)
        lab = posterior_argmax_labels(torch.as_tensor(feats[lset_full]), observed_full, ann)
        got = acc_1nn(feats[lset_full[:B]], lab[B], test_x, test_y)
        want = ref[("bayes_product", "1nn_postargmax", f"{noise:.1f}", str(seed), str(B))]
        ok = abs(got - want) <= GATE_TOL
        print(f"  bayes_product  n{noise} s{seed} B{B}: got {got:.2f} vs ref {want:.2f}  {'OK' if ok else 'MISMATCH'}")
        assert ok, "G2 FAILED — aborting, nothing written"
    print("  gates passed.\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clean = load_clean_labels()
    feats_np = load_features()                                    # unit-L2, np.float32
    test_x, test_y = load_test()
    gate(feats_np, test_x, test_y, clean)

    rows = []
    # ---- fill 1: bayes_misp 1-NN on observed labels ----
    for ntag, noise in NOISES.items():
        for seed in SEEDS:
            run = find_run(f"output/CIFAR100/resnet18/*/cfgBAYESCOV_{ntag}_*_albayes_misp_linear_seed{seed}_*",
                           MISP_SWEEP)
            lset_full, observed_full, _ = load_lset_and_observed(run, ntag, seed, clean)
            traj = {B: acc_1nn(feats_np[lset_full[:B]], observed_full[:B], test_x, test_y)
                    for B in BUDGETS}
            rows += [("bayes_misp", "1nn_observed", noise, seed, B, round(a, 3), MISP_SWEEP)
                     for B, a in traj.items()]
            print(f"  bayes_misp    1nn n{noise} s{seed}: @100={traj[100]:.1f} @1000={traj[1000]:.1f}")

    # ---- fill 2: bayes_product pseudo-all training-set 1-NN ----
    for ntag, noise in NOISES.items():
        for seed in SEEDS:
            run = find_product_run(ntag, "bayes_product", seed)
            traj = pseudoall_trajectory(run, ntag, seed, feats_np, test_x, test_y, clean)
            rows += [("bayes_product_pseudoall", "1nn_pseudoall", noise, seed, B, round(a, 3), PRODUCT_SWEEP)
                     for B, a in traj.items()]
            print(f"  pseudoall-1nn     n{noise} s{seed}: @100={traj[100]:.1f} @1000={traj[1000]:.1f}")

    out = os.path.join(OUT_DIR, "per_episode.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "evaluator", "noise", "seed", "budget", "test_acc", "src_sweep"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {os.path.relpath(out, BASE)}")


if __name__ == "__main__":
    main()
