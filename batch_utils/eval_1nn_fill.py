#!/usr/bin/env python3
"""Fill / correct 1-NN evaluator columns the original eval_1nn_all_arms study
either missed or computed on superseded selections. CPU-only offline replay.

The §8.3 study (per_episode.csv, 2026-07-12) replayed selection sweep 30978366.
Two problems it left:
  1. `bayes_additive` was replayed from the PRE-FIX additive acquisition (30978366),
     which clumped points at low budget (the bug fixed 2026-07-14, sweep 31066809).
     1-NN fit on those badly-covered points reads absurdly low (18.0 @100 clean,
     below random). We recompute it from the FIXED sweep 31066809.
  2. `npc` had no 1-NN at all. npc selection depends on AUM feedback, so clean-train
     on/off give DIFFERENT lSets (verified) → two lineages, one per npc table row.

Method is identical to eval_1nn_all_arms.py: KNeighborsClassifier(1) on unit-L2
SimCLR features + the run's OBSERVED (noisy) labels, evaluated on the 10k test set.
A CORRECTNESS GATE recomputes probcover/bayes_product (unaffected arms) through this
code path and checks they reproduce per_episode.csv before anything is written.

Runs are found by (config-name pattern AND result.json sweep_id) so the buggy and
fixed bayes_additive runs — same cfg name, different sweep — are never confused.

Writes: results/cifar100/eval_1nn_all_arms/per_episode_fill.csv
        columns: arm, evaluator, noise, seed, budget, test_acc, src_sweep
Run:    python3 batch_utils/eval_1nn_fill.py
Then:   python3 batch_utils/build_results_master.py   (ingests the fill file)
"""
import csv
import glob
import json
import os
import pickle

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

BASE = "/cs/labs/daphna/alongol/projects/daphna_lab"
EVAL_DIR = os.path.join(BASE, "results/cifar100/eval_1nn_all_arms")
PER_EPISODE = os.path.join(EVAL_DIR, "per_episode.csv")
OUT = os.path.join(EVAL_DIR, "per_episode_fill.csv")
TRAIN_FEAT = "/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy"
TEST_FEAT = "/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_test.npy"

BUDGETS = list(range(100, 1001, 100))
NTAG = {0.0: "n00", 0.2: "n20", 0.5: "n50", 0.8: "n80"}
SEEDS = [1, 2, 3]
GATE_TOL = 0.05

# (arm_out, src_sweep, cfg-name glob template, noises). {ntag}/{seed} filled per cell.
FILL = [
    ("bayes_additive", "31066809",
     "output/CIFAR100/resnet18/*/cfgBAYESCOV_{ntag}_*_albayes_additive_seed{seed}_*",
     [0.0, 0.2, 0.5, 0.8]),
    ("npc_selonly", "30863746",
     "output/CIFAR100/resnet18/*/cfgAUMFILTER_{ntag}_cleanoff_alnpc_seed{seed}_*",
     [0.2, 0.5, 0.8]),  # npc has no clean (noise 0) arm
    ("npc_aum", "30863746",
     "output/CIFAR100/resnet18/*/cfgAUMFILTER_{ntag}_cleanon_alnpc_seed{seed}_*",
     [0.2, 0.5, 0.8]),
]


def load_norm(path):
    f = np.load(path)
    return (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float32)


def load_test_labels():
    with open(os.path.join(BASE, "data/cifar-100-python/test"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return np.array(d[b"fine_labels"], dtype=np.int64)


def find_run(pattern, sweep_id):
    """Glob then filter by result.json sweep_id — disambiguates same-cfg-name runs
    (e.g. buggy 30978366 vs fixed 31066809 bayes_additive)."""
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


def lset_observed(run, B):
    lset = np.load(os.path.join(run, "episode_9/lSet.npy"), allow_pickle=True).astype(int)
    recs = {int(k): int(v) for k, v in json.load(open(os.path.join(run, "episode_8/annotated_labels.json"))).items()}
    lset = lset[:B]
    return lset, np.array([recs[int(g)] for g in lset], dtype=np.int64)


def acc_1nn(fit_x, fit_y, test_x, test_y):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(fit_x, fit_y)
    return 100.0 * float((knn.predict(test_x) == test_y).mean())


def trajectory(run, feats, test_x, test_y):
    out = {}
    for B in BUDGETS:
        lset, obs = lset_observed(run, B)
        out[B] = acc_1nn(feats[lset], obs, test_x, test_y)
    return out


def run_gate(feats, test_x, test_y):
    ref = {}
    for r in csv.DictReader(open(PER_EPISODE)):
        if r["evaluator"] == "1nn_observed" and r["arm"] in ("probcover", "bayes_product"):
            ref[(r["arm"], r["noise"], r["seed"], r["budget"])] = float(r["test_acc"])
    checks = [("probcover", 0.0, 1, 100), ("probcover", 0.2, 2, 500),
              ("bayes_product", 0.5, 1, 300), ("bayes_product", 0.8, 3, 1000)]
    print("=== correctness gate (recompute vs per_episode.csv) ===")
    for arm, noise, seed, B in checks:
        run = find_run(f"output/CIFAR100/resnet18/2026_7_2/cfgBAYESCOV_{NTAG[noise]}_*_al{arm}_seed{seed}_*", "30978366")
        lset, obs = lset_observed(run, B)
        got = acc_1nn(feats[lset], obs, test_x, test_y)
        want = ref[(arm, f"{noise:.1f}", str(seed), str(B))]
        ok = abs(got - want) <= GATE_TOL
        print(f"  {arm:14s} n{noise} s{seed} B{B}: got {got:.2f} vs ref {want:.2f}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"GATE FAILED for {arm} n{noise} s{seed} B{B} — aborting, nothing written")
    print("  gate passed.\n")


def main():
    feats = load_norm(TRAIN_FEAT)
    test_x = load_norm(TEST_FEAT)
    test_y = load_test_labels()
    run_gate(feats, test_x, test_y)

    rows = []
    for arm_out, sweep, tmpl, noises in FILL:
        for noise in noises:
            for seed in SEEDS:
                run = find_run(tmpl.format(ntag=NTAG[noise], seed=seed), sweep)
                traj = trajectory(run, feats, test_x, test_y)
                for B, acc in traj.items():
                    rows.append((arm_out, "1nn_observed", noise, seed, B, round(acc, 3), sweep))
                print(f"  {arm_out:14s} n{noise} s{seed}: @100={traj[100]:.1f} @1000={traj[1000]:.1f}")

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "evaluator", "noise", "seed", "budget", "test_acc", "src_sweep"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {os.path.relpath(OUT, BASE)}")


if __name__ == "__main__":
    main()
