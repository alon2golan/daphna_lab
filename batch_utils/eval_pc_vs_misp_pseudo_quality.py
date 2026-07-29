#!/usr/bin/env python3
"""probcover-vs-misp pseudo-label quality under noise (2026-07-29, e18 control).

The probcover × PoE-pseudo sweep (31150815) landed FAR below the misp-hybrid
under noise (n.5 38.5 vs 41.7; n.8 18.1 vs 25.2 @1000) despite matching it on
clean data — so misp's selection contributes real noise-robustness beyond
ProbCover. Hypothesis: misp's noise-minted conflict residuals raise its
revisit rate (56→69% with noise, vs probcover's flat 52%), and those repeated
in-ball anchors are the raw material PoE denoising needs — without them the
engine propagates confidently-wrong single-anchor pseudo-labels.

Per noise {0.2,0.5,0.8} × arm {probcover 31150815, misp 31116406} × seed:
replay the exact product fold on the run's REGEN-VALIDATED observed labels
along the recorded lSet and measure @1000: pool coverage, PoE-argmax pseudo
accuracy over covered points, revisit rate, and the multi-anchor census
(fraction of covered points reached by ≥2 anchors).

Writes results/cifar100/clean_selection_geometry/pc_vs_misp_noise.csv
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
from deep_al.pycls.noise.annotator import Annotator  # noqa: E402

OUT = "results/cifar100/clean_selection_geometry/pc_vs_misp_noise.csv"
FEAT_TRAIN = "/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy"
DELTA, C_CLS = 0.75, 100
COS_THR = 1.0 - DELTA ** 2 / 2.0
LOG_EPS = 1e-12
NOISES = [0.2, 0.5, 0.8]
SEEDS = [1, 2, 3]
ARMS = {"probcover": ("31150815", "probcover"), "misp": ("31116406", "bayes_misp")}

try:
    DEV = "cuda" if torch.cuda.is_available() and torch.zeros(1).to("cuda") is not None else "cpu"
except Exception:
    DEV = "cpu"


def load_clean_labels():
    import pickle
    with open("/cs/dataset/CIFAR/cifar-100-python/train", "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    return np.array(d[b"fine_labels"], dtype=np.int64)


def find_run(sweep, alm, noise, seed):
    for f in glob.glob("output/CIFAR100/resnet18/*/*/result.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        p = d.get("params", {})
        if (str(d.get("sweep_id")), p.get("al_method"), int(p.get("seed") or 0),
                float(p.get("noise_rate") or 0)) == (sweep, alm, seed, noise):
            return os.path.dirname(f)
    raise SystemExit(f"run not found: {sweep}/{alm}/n{noise}/s{seed}")


def observed_labels(run, clean, noise, seed):
    lset = np.load(run + "/episode_9/lSet.npy", allow_pickle=True).astype(int)
    rec = {int(k): int(v) for k, v in
           json.load(open(run + "/episode_8/annotated_labels.json")).items()}
    obs = np.array([rec[g] for g in lset], dtype=np.int64)

    class _Stub:
        pass
    stub = _Stub()
    stub.targets = clean.tolist()
    ann = Annotator(dataset=stub, noise_type="sym", noise_rate=noise, cost=1,
                    seed=seed, num_classes=C_CLS)
    assert np.array_equal(ann.noisy_labels[lset], obs), f"regen mismatch: {run}"
    return lset, obs


def main():
    clean = load_clean_labels()
    feats = np.load(FEAT_TRAIN).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    wr = csv.DictWriter(open(OUT, "w", newline=""), fieldnames=[
        "arm", "noise", "seed", "coverage", "covered_n", "pseudo_acc",
        "multi_anchor_frac", "revisit"])
    wr.writeheader()

    for noise in NOISES:
        for arm, (sweep, alm) in ARMS.items():
            for seed in SEEDS:
                run = find_run(sweep, alm, noise, seed)
                lset, obs = observed_labels(run, clean, noise, seed)
                l0 = np.load(run + "/episode_0/lSet.npy", allow_pickle=True).astype(int)
                u0 = np.load(run + "/episode_0/uSet.npy", allow_pickle=True).astype(int)
                pool = np.sort(np.concatenate([l0, u0]))
                pf = torch.from_numpy(feats[pool]).to(DEV)
                pick_f = torch.from_numpy(feats[lset]).to(DEV)

                # Step-A belief per anchor (sym annotator, accuracy r = 1 - noise)
                r = 1.0 - noise
                logb = np.log(np.full((len(lset), C_CLS), (1.0 - r) / (C_CLS - 1),
                                      dtype=np.float32).clip(LOG_EPS))
                logb[np.arange(len(lset)), obs] = np.log(max(r, LOG_EPS))

                Cmat = np.full((len(pool), C_CLS), np.log(1.0 / C_CLS), dtype=np.float32)
                anchors = np.zeros(len(pool), dtype=np.int32)
                for i in range(len(lset)):
                    ball = torch.nonzero(pf @ pick_f[i] > COS_THR).squeeze(1).cpu().numpy()
                    Cmat[ball] += logb[i]
                    anchors[ball] += 1

                covered = anchors > 0
                pseudo = Cmat[covered].argmax(axis=1)
                pseudo_acc = 100.0 * float((pseudo == clean[pool[covered]]).mean())
                sim_pp = (pick_f @ pick_f.T).cpu().numpy()
                revisit = 100.0 * float(np.mean(
                    [(sim_pp[i, :i] > COS_THR).any() for i in range(len(lset))]))
                wr.writerow(dict(
                    arm=arm, noise=noise, seed=seed,
                    coverage=round(100.0 * covered.mean(), 2),
                    covered_n=int(covered.sum()),
                    pseudo_acc=round(pseudo_acc, 2),
                    multi_anchor_frac=round(100.0 * float((anchors[covered] >= 2).mean()), 2),
                    revisit=round(revisit, 1)))
                print(f"{arm} n{noise} s{seed} done")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
