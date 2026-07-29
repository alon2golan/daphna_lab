#!/usr/bin/env python3
"""Why does misp selection beat product selection on CLEAN data? (2026-07-28)

The e18 control isolated the clean gap to *selection* (same PoE training
targets: misp-hybrid 49.9 vs product 48.4 @1000). By the acquisition algebra
all three rules (probcover / product / misp diff-max) reduce to the SAME
uncovered-count greedy at noise 0 under tophat — yet the recorded lSets say
product ∩ probcover ≈ 490/1000 while misp ∩ probcover ≈ 750/1000: product is
the one that drifts off the coverage greedy on clean data, not misp.

This script measures WHERE it drifts and what the drift costs:
  per arm (product / probcover / misp / random, clean, seeds 1-3) and per
  budget prefix B ∈ {100..1000}:
    - coverage of the seed's 45k pool (δ=0.75 tophat, unit-L2 features — the
      space every selector actually operates in: ds_utils.load_features
      normalized=True regardless of --normalize_features)
    - pseudo-label mass (= covered count; covered-only pseudo_all trains on
      exactly these points) and plurality pseudo accuracy over covered points
    - revisit rate (pick within δ of an earlier pick), mean pick ball size
    - class balance of the lSet prefix (#distinct classes, entropy)
    - 1-NN test accuracy on the prefix (training-free head proxy)
  plus an offline UNCOVERED-COUNT GREEDY simulation ("ideal ProbCover") on the
  same pool for seed-matched reference: overlap with each arm + own metrics.

Writes results/cifar100/clean_selection_geometry/{per_budget.csv,sim_overlap.csv}
Run:   python3 batch_utils/eval_clean_selection_geometry.py
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np
import torch

BASE = "/cs/labs/daphna/alongol/projects/daphna_lab"
sys.path.insert(0, BASE)
os.chdir(BASE)

OUT_DIR = "results/cifar100/clean_selection_geometry"
FEAT_TRAIN = "/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy"
FEAT_TEST = "/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_test.npy"
DELTA = 0.75
COS_THR = 1.0 - DELTA ** 2 / 2.0          # d < δ  ⟺  cos > 1 − δ²/2 (unit-L2)
BUDGETS = list(range(100, 1001, 100))
SEEDS = [1, 2, 3]
C = 100

# arm -> (sweep_id, al_method, train_label_mode, pseudo_scope) in the run-dir map
ARMS = {
    "product": ("31048960", "bayes_product", "pseudo", "all"),
    "probcover": ("31048960", "probcover", "observed", None),
    "misp": ("31116406", "bayes_misp", "pseudo", "all"),
    "random": ("31048960", "random", "observed", None),
}

try:
    DEV = "cuda" if torch.cuda.is_available() and torch.zeros(1).to("cuda") is not None else "cpu"
except Exception:
    DEV = "cpu"


def find_runs():
    m = {}
    for f in glob.glob("output/CIFAR100/resnet18/*/*/result.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        p = d.get("params", {})
        if float(p.get("noise_rate") or 0) != 0.0:
            continue
        key = (str(d.get("sweep_id")), p.get("al_method"), p.get("train_label_mode"),
               p.get("pseudo_scope"), int(p.get("seed") or 0))
        m[key] = os.path.dirname(f)
    return m


def load_clean_labels():
    import pickle
    with open("/cs/dataset/CIFAR/cifar-100-python/train", "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    return np.array(d[b"fine_labels"], dtype=np.int64)


def load_test_labels():
    import pickle
    with open("/cs/dataset/CIFAR/cifar-100-python/test", "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    return np.array(d[b"fine_labels"], dtype=np.int64)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def balls_of(picks_feats, pool_feats_t):
    """List of int32 arrays: pool-local indices within δ of each pick."""
    out = []
    pf = torch.from_numpy(picks_feats).to(DEV)
    for i in range(pf.shape[0]):
        sims = pool_feats_t @ pf[i]
        out.append(torch.nonzero(sims > COS_THR).squeeze(1).cpu().numpy().astype(np.int32))
    return out


def acc_1nn(fit_x, fit_y, test_x_t, test_y):
    fx = torch.from_numpy(fit_x).to(DEV)
    sims = test_x_t @ fx.T                                    # [10000, B]
    pred = torch.as_tensor(fit_y, device=DEV)[sims.argmax(dim=1)].cpu().numpy()
    return 100.0 * float((pred == test_y).mean())


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def arm_metrics(tag, lset_global, pool_global, feats_u, clean, test_x_t, test_y, writer):
    """Per-budget metrics for one selection sequence (global indices)."""
    g2l = {g: i for i, g in enumerate(pool_global)}
    pool_feats = feats_u[pool_global]
    pool_feats_t = torch.from_numpy(pool_feats).to(DEV)
    picks_local = np.array([g2l[g] for g in lset_global], dtype=np.int64)
    balls = balls_of(pool_feats[picks_local], pool_feats_t)

    # revisit: pick i within δ of any earlier pick (cos > thr)
    pick_f = torch.from_numpy(pool_feats[picks_local]).to(DEV)
    sim_pp = (pick_f @ pick_f.T).cpu().numpy()
    revisit = np.array([bool((sim_pp[i, :i] > COS_THR).any()) for i in range(len(picks_local))])

    covered = np.zeros(len(pool_global), dtype=bool)
    # plurality pseudo-label vote counts over covering anchors
    votes = np.zeros((len(pool_global), C), dtype=np.int16)
    rows, marg = [], []
    b_prev = 0
    for B in BUDGETS:
        for i in range(b_prev, B):
            nb = balls[i]
            marg.append(int((~covered[nb]).sum()))
            covered[nb] = True
            votes[nb, clean[lset_global[i]]] += 1
        b_prev = B
        cov_idx = np.nonzero(covered)[0]
        plur = votes[cov_idx].argmax(axis=1)
        pseudo_acc = 100.0 * float((plur == clean[pool_global[cov_idx]]).mean())
        cls_counts = np.bincount(clean[lset_global[:B]], minlength=C)
        rows.append(dict(
            arm=tag[0], seed=tag[1], budget=B,
            coverage=round(100.0 * covered.mean(), 2),
            covered_n=int(covered.sum()),
            pseudo_acc=round(pseudo_acc, 2),
            revisit=round(100.0 * revisit[:B].mean(), 1),
            ball_mean=round(float(np.mean([len(balls[i]) for i in range(B)])), 1),
            marg_new_mean=round(float(np.mean(marg[:B])), 1),
            n_classes=int((cls_counts > 0).sum()),
            cls_entropy=round(entropy(cls_counts), 3),
            max_cls=int(cls_counts.max()),
            acc_1nn=round(acc_1nn(feats_u[lset_global[:B]], clean[lset_global[:B]],
                                  test_x_t, test_y), 2),
        ))
    for r in rows:
        writer.writerow(r)
    return rows


def greedy_sim(pool_global, feats_u, n_pick=1000):
    """Uncovered-count greedy (ideal ProbCover) on the pool; returns global picks."""
    pool_feats = torch.from_numpy(feats_u[pool_global]).to(DEV)
    N = pool_feats.shape[0]
    # adjacency as list of int32 arrays (chunked build)
    nbrs = []
    CH = 2048
    for s in range(0, N, CH):
        sims = pool_feats[s:s + CH] @ pool_feats.T            # [ch, N]
        mask = sims > COS_THR
        for r in range(mask.shape[0]):
            nbrs.append(torch.nonzero(mask[r]).squeeze(1).cpu().numpy().astype(np.int32))
        del sims, mask
    deg = np.array([len(x) for x in nbrs], dtype=np.int64)    # all pool uncovered at start
    covered = np.zeros(N, dtype=bool)
    selected = np.zeros(N, dtype=bool)
    picks = []
    for _ in range(n_pick):
        deg_masked = np.where(selected, -1, deg)
        v = int(deg_masked.argmax())
        picks.append(v)
        selected[v] = True
        newly = nbrs[v][~covered[nbrs[v]]]
        covered[newly] = True
        for u in newly:                                       # deg drops for u's neighbors
            deg[nbrs[u]] -= 1
    return np.array([pool_global[v] for v in picks], dtype=np.int64)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device={DEV}")
    runs = find_runs()
    clean = load_clean_labels()
    test_y = load_test_labels()
    feats_u = unit(np.load(FEAT_TRAIN).astype(np.float32))
    test_x_t = torch.from_numpy(unit(np.load(FEAT_TEST).astype(np.float32))).to(DEV)

    fh = open(os.path.join(OUT_DIR, "per_budget.csv"), "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=[
        "arm", "seed", "budget", "coverage", "covered_n", "pseudo_acc", "revisit",
        "ball_mean", "marg_new_mean", "n_classes", "cls_entropy", "max_cls", "acc_1nn"])
    writer.writeheader()

    fh2 = open(os.path.join(OUT_DIR, "sim_overlap.csv"), "w", newline="")
    w2 = csv.DictWriter(fh2, fieldnames=["seed", "arm", "budget", "overlap_with_sim"])
    w2.writeheader()

    for seed in SEEDS:
        # seed's candidate pool from any run of that seed (all arms share the split)
        run0 = runs[("31048960", "bayes_product", "pseudo", "all", seed)]
        l0 = np.load(run0 + "/episode_0/lSet.npy", allow_pickle=True).astype(int)
        u0 = np.load(run0 + "/episode_0/uSet.npy", allow_pickle=True).astype(int)
        pool_global = np.sort(np.concatenate([l0, u0]))
        print(f"seed {seed}: pool={len(pool_global)}")

        lsets = {}
        for arm, (sid, alm, tlm, ps) in ARMS.items():
            run = runs[(sid, alm, tlm, ps, seed)]
            lsets[arm] = np.load(run + "/episode_9/lSet.npy", allow_pickle=True).astype(int)
            arm_metrics((arm, seed), lsets[arm], pool_global, feats_u, clean,
                        test_x_t, test_y, writer)
            fh.flush()
            print(f"  {arm} done")

        sim = greedy_sim(pool_global, feats_u)
        lsets["sim"] = sim
        arm_metrics(("sim", seed), sim, pool_global, feats_u, clean,
                    test_x_t, test_y, writer)
        fh.flush()
        print(f"  sim done")
        for arm in ("product", "probcover", "misp"):
            for B in BUDGETS:
                w2.writerow(dict(seed=seed, arm=arm, budget=B,
                                 overlap_with_sim=len(set(lsets[arm][:B].tolist())
                                                      & set(sim[:B].tolist()))))
        fh2.flush()
    fh.close()
    fh2.close()
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
