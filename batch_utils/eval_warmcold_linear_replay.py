#!/usr/bin/env python3
"""Warm-vs-cold linear-head replay under label noise (2026-07-26).

Question (§4.7 open confound): the pipeline's linear head is WARM-started
across episodes (FINE_TUNE=True) while the sklearn-MLP head is cold. On clean
data warm-starting was measured as +2.2 @1000 for the linear head, but it was
never measured under noise — where entering an episode with weights that
already spent ~2000 epochs fitting noisy labels may HURT, and per-episode
best-val checkpointing cannot undo it. If cold+best-val linear beats the warm
pipeline linear under noise, part of the MLP's noise lead over the linear rows
(R3-R5) is this protocol asymmetry, not architecture.

Replays the BAYES_MISP noise-sweep runs (31084429, linear rows of §4.7):
exact training protocol replica — FeaturesNet linear head on RAW SimCLR
features, SGD(BASE_LR .025, momentum .9, nesterov, wd 3e-4), per-episode
cosine over MAX_EPOCH 200, batch 100, val eval every 2 epochs on the run's
clean 5k valSet, strict-> best-val checkpoint tested on the 10k test set.
Budgets are lSet prefixes; observed labels regen-validated via Annotator
(same as eval_1nn_all_arms.load_lset_and_observed).

Protocols:
  warm  — model+optimizer persist across episodes, continuing from the FINAL
          iterate of the previous episode (pipeline replica; fidelity gate
          against the run's recorded per-episode test accs, warn > 1.0).
  cold  — fresh model+optimizer every episode (the missing cell).

Writes results/cifar100/warmcold_linear_replay/per_episode.csv
Run:   python3 batch_utils/eval_warmcold_linear_replay.py
"""
import csv
import glob
import json
import math
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "/cs/labs/daphna/alongol/projects/daphna_lab"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "batch_utils"))

from deep_al.pycls.noise.annotator import Annotator

OUT_DIR = os.path.join(BASE, "results/cifar100/warmcold_linear_replay")
FEAT_TRAIN = "/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy"
FEAT_TEST = "/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_test.npy"
# arm -> (run-dir glob with {ntag}/{seed} slots, sweep_id filter or None).
# bayes_product = the observed-label production runs (2026_7_2 family) whose
# lSets eval_1nn_all_arms already replays; same OPTIM block (gate verifies).
ARMS = {
    "bayes_misp": (
        "output/CIFAR100/resnet18/2026_7_16/cfgBAYESCOV_{ntag}_*_albayes_misp_linear_seed{seed}_*",
        "31084429"),
    "bayes_product": (
        "output/CIFAR100/resnet18/2026_7_2/cfgBAYESCOV_{ntag}_*_albayes_product_seed{seed}_*",
        None),
}
NOISES = {"n00": 0.0, "n20": 0.2, "n50": 0.5, "n80": 0.8}
SEEDS = [1, 2, 3]
BUDGETS = list(range(100, 1001, 100))
C = 100
# OPTIM/TRAIN block of the 31084429 configs
BASE_LR, MOMENTUM, WD, MAX_EPOCH, BATCH, EVAL_PERIOD = 0.025, 0.9, 3e-4, 200, 100, 2
GATE_WARN = 1.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class FeaturesNet(nn.Module):
    """Verbatim replica of builders.FeaturesNet (use_mlp=False): lin1/lin2 are
    constructed (matching init-RNG consumption and optimizer param groups) but
    unused in forward."""

    def __init__(self, in_layers, out_layers):
        super().__init__()
        self.lin1 = nn.Linear(in_layers, in_layers)
        self.lin2 = nn.Linear(in_layers, in_layers)
        self.final = nn.Linear(in_layers, out_layers)

    def forward(self, x):
        return self.final(x)


def find_run(arm, ntag, seed):
    pattern, sweep = ARMS[arm]
    pat = os.path.join(BASE, pattern.format(ntag=ntag, seed=seed))
    hits = [r for r in glob.glob(pat)
            if os.path.exists(os.path.join(r, "result.json"))
            and (sweep is None
                 or json.load(open(os.path.join(r, "result.json"))).get("sweep_id") == sweep)]
    assert len(hits) == 1, f"{pat} @ {sweep} -> {hits}"
    return hits[0]


def load_clean_labels():
    with open(os.path.join(BASE, "data/cifar-100-python/train"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return np.array(d[b"fine_labels"], dtype=np.int64)


def load_test_labels():
    with open(os.path.join(BASE, "data/cifar-100-python/test"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return np.array(d[b"fine_labels"], dtype=np.int64)


def load_lset_and_observed(run, ntag, seed, clean):
    ld = lambda p: np.load(p, allow_pickle=True).astype(int)
    lset_full = ld(os.path.join(run, "episode_9/lSet.npy"))
    assert len(lset_full) == 1000
    records = {int(k): int(v)
               for k, v in json.load(open(os.path.join(run, "episode_8/annotated_labels.json"))).items()}
    observed_full = np.array([records[g] for g in lset_full], dtype=np.int64)

    class _Stub:
        pass

    stub = _Stub()
    stub.targets = clean.tolist()
    ann = Annotator(dataset=stub, noise_type="sym", noise_rate=NOISES[ntag], cost=1,
                    seed=seed, num_classes=C)
    assert np.array_equal(ann.noisy_labels[lset_full], observed_full), f"annotation regen mismatch: {run}"
    return lset_full, observed_full


def val_indices(run):
    ld = lambda p: np.load(p, allow_pickle=True).astype(int)
    pool = np.concatenate([ld(os.path.join(run, "episode_0/lSet.npy")),
                           ld(os.path.join(run, "episode_0/uSet.npy"))])
    val = np.setdiff1d(np.arange(50000), pool)
    assert len(val) == 5000, len(val)
    return val


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    preds = model(X).argmax(dim=1)
    return 100.0 * float((preds == y).float().mean())


def train_episode(model, opt, Xtr, ytr, Xval, yval, gen):
    """One pipeline episode: MAX_EPOCH epochs, cosine LR, best-val state kept
    (strict >, eval every EVAL_PERIOD epochs and at the last epoch)."""
    n = len(Xtr)
    best_val, best_state = -1.0, None
    for ep in range(MAX_EPOCH):
        lr = BASE_LR * 0.5 * (1.0 + math.cos(math.pi * ep / MAX_EPOCH))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % EVAL_PERIOD == 0 or (ep + 1) == MAX_EPOCH:
            va = accuracy(model, Xval, yval)
            if va > best_val:
                best_val = va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return best_val, best_state


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    feats = torch.as_tensor(np.load(FEAT_TRAIN).astype(np.float32), device=DEV)  # RAW, no L2 norm
    test_X = torch.as_tensor(np.load(FEAT_TEST).astype(np.float32), device=DEV)
    test_y = torch.as_tensor(load_test_labels(), device=DEV)
    clean = load_clean_labels()

    rows = []
    for arm in ARMS:
      for ntag in NOISES:
        for seed in SEEDS:
            run = find_run(arm, ntag, seed)
            lset_full, observed_full = load_lset_and_observed(run, ntag, seed, clean)
            val = val_indices(run)
            Xval = feats[torch.as_tensor(val, device=DEV)]
            yval = torch.as_tensor(clean[val], device=DEV)  # valSet always true labels
            pipeline = {e["labeled_size"]: e["test_acc"]
                        for e in json.load(open(os.path.join(run, "result.json")))["episodes"]}

            for proto in ("warm", "cold"):
                torch.manual_seed(seed)
                gen = torch.Generator().manual_seed(10_000 * seed + 7)
                model = FeaturesNet(512, C).to(DEV)
                opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                                      weight_decay=WD, nesterov=True)
                for B in BUDGETS:
                    if proto == "cold":
                        model = FeaturesNet(512, C).to(DEV)
                        opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                                              weight_decay=WD, nesterov=True)
                    idx = torch.as_tensor(lset_full[:B], device=DEV)
                    Xtr, ytr = feats[idx], torch.as_tensor(observed_full[:B], device=DEV)
                    best_val, best_state = train_episode(model, opt, Xtr, ytr, Xval, yval, gen)
                    tester = FeaturesNet(512, C).to(DEV)
                    tester.load_state_dict(best_state)
                    tacc = accuracy(tester, test_X, test_y)
                    rows.append(dict(arm=arm, ntag=ntag, noise=NOISES[ntag], seed=seed,
                                     protocol=proto, budget=B, test_acc=round(tacc, 2),
                                     best_val=round(best_val, 2),
                                     pipeline_test_acc=pipeline.get(B, "")))
                    gate = ""
                    if proto == "warm" and B in pipeline:
                        diff = tacc - pipeline[B]
                        gate = f" pipeline {pipeline[B]:.2f} (diff {diff:+.2f})" + \
                               ("  ⚠ GATE" if abs(diff) > GATE_WARN else "")
                    print(f"{arm} {ntag} seed{seed} {proto} B={B}: test {tacc:.2f}{gate}", flush=True)

    out = os.path.join(OUT_DIR, "per_episode.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
