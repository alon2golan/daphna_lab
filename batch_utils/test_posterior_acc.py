#!/usr/bin/env python3
"""
Graph-posterior-as-classifier on the CIFAR100 TEST set — the §4-comparable number.

For each (noise, seed, budget): fold the labeled train points (same records the runs
used), classify each test point by argmax of its in-ball posterior. Uncovered test
points keep the uniform prior (counted as wrong, like any classifier that must answer).
Writes results/cifar100/posterior_eval/test_posterior_acc.json.
"""
import numpy as np, torch, json, glob, os, pickle

BASE = '/cs/labs/daphna/alongol/projects/daphna_lab'
OUT = os.path.join(BASE, 'results/cifar100/posterior_eval/test_posterior_acc.json')
DEVICE = 'cuda'
DELTA = 0.75
C = 100
LOG_EPS = 1e-12
BUDGETS = list(range(100, 1001, 100))
NOISES = {'n00': 0.0, 'n20': 0.2, 'n50': 0.5, 'n80': 0.8}


def norm_feats(path):
    f = np.load(path)
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    return torch.as_tensor(f.astype(np.float32), device=DEVICE)


def labels(split):
    with open(os.path.join(BASE, f'data/cifar-100-python/{split}'), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return np.array(d[b'fine_labels'], dtype=np.int64)


train_f = norm_feats('/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy')
test_f = norm_feats('/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_test.npy')
test_y = torch.as_tensor(labels('test'), device=DEVICE)

results = []
for ntag, nr in NOISES.items():
    rho = 1.0 - nr
    off = nr / (C - 1)
    for seed in (1, 2, 3):
        run = sorted(glob.glob(os.path.join(
            BASE, f'output/CIFAR100/resnet18/2026_7_2/cfgBAYESCOV_{ntag}_*_albayes_product_train_label_modeobserved_seed{seed}_*')))[0]
        lset = np.load(os.path.join(run, 'episode_9/lSet.npy'), allow_pickle=True).astype(int)
        rec = {int(k): int(v) for k, v in json.load(open(os.path.join(run, 'episode_8/annotated_labels.json'))).items()}
        obs = np.array([rec[g] for g in lset])

        lf = train_f[torch.as_tensor(lset, device=DEVICE)]
        d = torch.cdist(test_f.unsqueeze(0), lf.unsqueeze(0)).squeeze(0)   # [10000, 1000]
        inball = (d < DELTA).float()
        logb = torch.full((1000, C), float(np.log(max(off, LOG_EPS))), device=DEVICE)
        logb[torch.arange(1000), torch.as_tensor(obs, device=DEVICE)] = float(np.log(max(rho, LOG_EPS)))

        row = {'noise': nr, 'seed': seed, 'budgets': {}}
        for B in BUDGETS:
            Cm = inball[:, :B] @ logb[:B]
            covered = inball[:, :B].sum(dim=1) > 0
            pred = Cm.argmax(dim=1)
            correct = pred == test_y
            row['budgets'][B] = dict(
                acc_all=float(correct[covered].sum() / len(test_y)),  # uncovered = wrong
                acc_covered=float(correct[covered].float().mean()),
                cov_frac=float(covered.float().mean()),
            )
        results.append(row)
        print(ntag, seed, 'done')

json.dump(results, open(OUT, 'w'), indent=1)
print('written', OUT)
