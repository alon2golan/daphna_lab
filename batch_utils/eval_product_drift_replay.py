#!/usr/bin/env python3
"""Exact replay of the product acquisition on the recorded clean lSets (2026-07-28).

Hypothesis (clean_selection_geometry follow-up): at noise 0 the product rule
does NOT reduce to the uncovered-count greedy because points covered by anchors
of k DIFFERENT classes keep deficit 1-1/k (the one-hot log-folds tie), so
class-boundary overlap regions retain acquisition mass forever — an
uncertainty-sampling term that misp's diff-max rule shares only in a
prior-damped form: misp's selection posterior is the NORMALIZED additive
count matrix (BAYES_MISP.py `norm_C = C / C.sum`), so a 2-class-conflicted
point keeps deficit f1 − maxP = 1.5/51 − 1.5/52 ≈ 0.00057 vs 0.0194 for an
uncovered in-ball point (2.9% relative weight; ~34 conflicted ≈ 1 uncovered).
In product the same point keeps 0.5 vs 0.99 (50.5%; 2 conflicted ≈ 1
uncovered) — the Dirichlet prior mass αL=50 is what crushes the term in misp.

Replays C (fp32 log-space, exact _fold_neighbor algebra for tophat + clean
annotator) along the RECORDED product/misp clean lSets and, per pick:
  - fidelity: is the recorded pick the argmax of the replayed product score?
  - count-greedy regret: (max #new-covered over candidates) - (#new-covered of pick)
  - decomposition of the pick's score: uncovered mass vs conflicted-covered mass
  - conflict census: #covered points with deficit > 0.3 (label-conflicted)

Writes results/cifar100/clean_selection_geometry/product_drift.csv
Run:   python3 batch_utils/eval_product_drift_replay.py [seed]
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
os.chdir(BASE)

OUT = "results/cifar100/clean_selection_geometry/product_drift.csv"
FEAT_TRAIN = "/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy"
DELTA, C_CLS = 0.75, 100
COS_THR = 1.0 - DELTA ** 2 / 2.0
LOG_EPS_TERM = float(np.log(1e-12))          # in-ball off-class log-fold increment
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1

try:
    DEV = "cuda" if torch.cuda.is_available() and torch.zeros(1).to("cuda") is not None else "cpu"
except Exception:
    DEV = "cpu"


def load_clean_labels():
    import pickle
    with open("/cs/dataset/CIFAR/cifar-100-python/train", "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    return np.array(d[b"fine_labels"], dtype=np.int64)


def find_run(sweep, alm, tlm, ps, seed):
    for f in glob.glob("output/CIFAR100/resnet18/*/*/result.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        p = d.get("params", {})
        if (str(d.get("sweep_id")), p.get("al_method"), p.get("train_label_mode"),
                p.get("pseudo_scope"), int(p.get("seed") or 0),
                float(p.get("noise_rate") or 0)) == (sweep, alm, tlm, ps, seed, 0.0):
            return os.path.dirname(f)
    raise SystemExit(f"run not found: {sweep}/{alm}/seed{seed}")


def main():
    clean = load_clean_labels()
    feats = np.load(FEAT_TRAIN).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)

    run_p = find_run("31048960", "bayes_product", "pseudo", "all", SEED)
    lset_p = np.load(run_p + "/episode_9/lSet.npy", allow_pickle=True).astype(int)
    l0 = np.load(run_p + "/episode_0/lSet.npy", allow_pickle=True).astype(int)
    u0 = np.load(run_p + "/episode_0/uSet.npy", allow_pickle=True).astype(int)
    pool = np.sort(np.concatenate([l0, u0]))
    g2l = {g: i for i, g in enumerate(pool)}
    N = len(pool)
    print(f"seed {SEED}: pool={N}, device={DEV}")

    # adjacency over the pool (strict-< tophat, symmetric)
    pf = torch.from_numpy(feats[pool]).to(DEV)
    nbrs = []
    for s in range(0, N, 2048):
        sims = pf[s:s + 2048] @ pf.T
        mask = sims > COS_THR
        for r in range(mask.shape[0]):
            nbrs.append(torch.nonzero(mask[r]).squeeze(1).cpu().numpy())
        del sims, mask
    deg = np.array([len(x) for x in nbrs], dtype=np.int64)

    # replayed product state
    Cmat = np.full((N, C_CLS), np.log(1.0 / C_CLS), dtype=np.float32)
    deficit = np.full(N, 1.0 - 1.0 / C_CLS, dtype=np.float32)
    score = deficit[0] * deg.astype(np.float32)             # uniform init
    covered = np.zeros(N, dtype=bool)
    uncov_cnt = deg.copy()                                  # #uncovered in ball(v)
    selected = np.zeros(N, dtype=bool)

    wr = csv.DictWriter(open(OUT if SEED == 1 else OUT.replace(".csv", f"_s{SEED}.csv"),
                             "w", newline=""),
                        fieldnames=["i", "is_argmax", "argmax_gap", "count_regret",
                                    "pick_uncov", "max_uncov", "pick_score",
                                    "pick_uncov_mass", "pick_conflict_mass",
                                    "n_conflicted_total"])
    wr.writeheader()
    match = 0
    for i, g in enumerate(lset_p):
        v = g2l[g]
        sm = np.where(selected, -np.inf, score)
        am = int(sm.argmax())
        is_am = (am == v)
        match += is_am
        cm = np.where(selected, -1, uncov_cnt)
        ball = nbrs[v]
        bd = deficit[ball]
        bcov = covered[ball]
        row = dict(
            i=i, is_argmax=int(is_am),
            argmax_gap=round(float(sm[am] - sm[v]), 4),
            count_regret=int(cm.max() - cm[v]),
            pick_uncov=int(cm[v]), max_uncov=int(cm.max()),
            pick_score=round(float(score[v]), 3),
            pick_uncov_mass=round(float(bd[~bcov].sum()), 3),
            pick_conflict_mass=round(float(bd[bcov].sum()), 3),
            n_conflicted_total=int((deficit > 0.3)[covered].sum()) if covered.any() else 0,
        )
        wr.writerow(row)

        # fold pick: in-ball rows, off-class channels += log_eps
        y = clean[g]
        mask_cls = np.ones(C_CLS, dtype=bool)
        mask_cls[y] = False
        Cmat[np.ix_(ball, np.nonzero(mask_cls)[0])] += LOG_EPS_TERM
        # recompute deficits on ball rows (stable softmax)
        rows = Cmat[ball]
        rows = rows - rows.max(axis=1, keepdims=True)
        p = np.exp(rows)
        newdef = (1.0 - p.max(axis=1) / p.sum(axis=1)).astype(np.float32)
        delta_def = newdef - deficit[ball]
        deficit[ball] = newdef
        for u, dd in zip(ball, delta_def):
            if dd != 0.0:
                score[nbrs[u]] += dd
        newly = ball[~covered[ball]]
        covered[newly] = True
        for u in newly:
            uncov_cnt[nbrs[u]] -= 1
        selected[v] = True
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/1000 replayed, argmax-match so far {match}/{i+1}")

    print(f"fidelity: recorded pick == replay argmax in {match}/1000 steps")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
