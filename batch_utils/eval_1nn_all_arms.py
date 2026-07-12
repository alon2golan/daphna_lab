#!/usr/bin/env python3
"""
Offline 1-NN + Nadaraya-Watson evaluator columns for every §4 selection arm
(docs/bayesian_coverage_NEXT_STEPS.md §8.3 Q2 + §9.1) — CPU only, no SLURM.

Replays from the persisted artifacts of the SELECTION sweep 30978366
(output/CIFAR100/resnet18/2026_7_2, arms {random, probcover, bayes_additive,
bayes_product} x noise {0,.2,.5,.8} x seeds {1,2,3}); the train-mode arms
(sweep 30980646) share bayes_product's exact lSets, so these 4 arms cover
every §4 row. Per (arm, noise, seed, budget in 100..1000) computes test-set
accuracy (10k CIFAR100 test points; unit-L2-normalized SimCLR features for
BOTH fit and predict — the load-bearing convention of the live USE_1NN branch
in train_al.py) for:

  1nn_observed    KNeighborsClassifier(n_neighbors=1) fit on the lSet's
                  OBSERVED (noisy) labels — all 4 arms. The probcover rows
                  must reproduce the live "probcover + 1nn" numbers of sweep
                  31048649 (SANITY GATE, checked below before anything else
                  is written).
  1nn_postargmax  1-NN fit on the lSet relabeled to the bayes posterior
                  argmax (product-of-experts replay, bit-style identical to
                  eval_posterior_quality.py's labeled plane) — bayes_product.
  nw_k3 / nw_k1   gap-scaled NW mixture (eval_posterior_variants.posterior_
                  variant 'nw_gap_k3'/'nw_gap_all'; kappa=3/1, m=7) predicting
                  each TEST point directly from the labeled set's noise-aware
                  beliefs b_l, argmax — bayes_product.

Writes results/cifar100/eval_1nn_all_arms/per_episode.csv (long format) +
tables.md (one §4-style markdown table per noise level, mean±std over seeds).
Run:  python batch_utils/eval_1nn_all_arms.py          (CPU, ~15 min)
"""
import numpy as np, torch, json, glob, os, csv, pickle, sys
from sklearn.neighbors import KNeighborsClassifier

BASE = '/cs/labs/daphna/alongol/projects/daphna_lab'
OUT_DIR = os.path.join(BASE, 'results/cifar100/eval_1nn_all_arms')
os.makedirs(OUT_DIR, exist_ok=True)
DELTA = 0.75
C = 100
LOG_EPS = 1e-12
BUDGETS = list(range(100, 1001, 100))
NOISES = {'n00': 0.0, 'n20': 0.2, 'n50': 0.5, 'n80': 0.8}
SEEDS = [1, 2, 3]
ARMS = ['probcover', 'random', 'bayes_additive', 'bayes_product']  # probcover first: sanity gate

# sanity gate: live "probcover + 1nn" numbers (sweep 31048649, §4 tables)
GATE = {('n00', 100): 38.1, ('n00', 1000): 45.6, ('n20', 100): 30.9}
GATE_TOL = 0.35

sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'batch_utils'))
from deep_al.pycls.noise.annotator import Annotator
from eval_posterior_variants import posterior_variant, load_clean_labels, load_features

NW_VARIANTS = {'nw_k3': 'nw_gap_k3', 'nw_k1': 'nw_gap_all'}  # kappa=3 / kappa=1, m=7


def load_test():
    with open(os.path.join(BASE, 'data/cifar-100-python/test'), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    labels = np.array(d[b'fine_labels'], dtype=np.int64)
    f = np.load('/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_test.npy')
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    return f.astype(np.float32), labels


def find_run(ntag, arm, seed):
    pat = os.path.join(BASE, f'output/CIFAR100/resnet18/2026_7_2/cfgBAYESCOV_{ntag}_*_al{arm}_seed{seed}_*')
    runs = sorted(glob.glob(pat))
    assert len(runs) == 1, f'{pat} -> {runs}'
    return runs[0]


def load_lset_and_observed(run, ntag, seed, clean):
    """(lset_full[1000], observed_full[1000], Annotator) with the same regen
    validation as eval_posterior_quality.py; budgets are prefixes of lSet."""
    ld = lambda p: np.load(p, allow_pickle=True).astype(int)
    lset_full = ld(os.path.join(run, 'episode_9/lSet.npy'))
    assert len(lset_full) == 1000
    records = {int(k): int(v) for k, v in json.load(open(os.path.join(run, 'episode_8/annotated_labels.json'))).items()}
    observed_full = np.array([records[g] for g in lset_full], dtype=np.int64)

    class _Stub: pass
    stub = _Stub(); stub.targets = clean.tolist()
    ann = Annotator(dataset=stub, noise_type='sym', noise_rate=NOISES[ntag], cost=1, seed=seed, num_classes=C)
    assert np.array_equal(ann.noisy_labels[lset_full], observed_full), f'annotation regen mismatch: {run}'
    return lset_full, observed_full, ann


def acc_1nn(fit_x, fit_y, test_x, test_y):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(fit_x, fit_y)
    return 100.0 * float((knn.predict(test_x) == test_y).mean())


def posterior_argmax_labels(lf, observed_full, ann):
    """Labeled-plane product posterior argmax per budget, replaying the run's
    accumulator (init log(pi), fold anchors one at a time in lSet order —
    same scheme as eval_posterior_quality.py, restricted to labeled rows).
    Returns {B: relabels[B]}."""
    pi = torch.full((C,), 1.0 / C, dtype=torch.float32)
    M_t = torch.as_tensor(ann.confusion_matrix, dtype=torch.float32)
    logb_by_label = torch.empty((C, C), dtype=torch.float32)
    for y in range(C):
        b = pi * M_t[:, y]
        b = b / b.sum().clamp_min(LOG_EPS)
        logb_by_label[y] = torch.log(b.clamp_min(LOG_EPS))
    logb = logb_by_label[torch.as_tensor(observed_full)]                 # [1000, C]

    d_ll = torch.cdist(lf, lf)                                           # [1000, 1000]
    inball = (d_ll < DELTA).float()
    Cmat = torch.log(pi.clamp_min(LOG_EPS)).repeat(len(lf), 1)           # [1000, C]
    out, next_fold = {}, 0
    for B in BUDGETS:
        while next_fold < B:
            Cmat += inball[:, next_fold:next_fold + 1] * logb[next_fold]
            next_fold += 1
        out[B] = Cmat[:B].argmax(dim=1).numpy().astype(np.int64)
    return out


def noise_aware_beliefs(observed_full, ann):
    """b_l rows [1000, C] (pi * M[:,y] normalized) + pi row — as in
    eval_posterior_variants.eval_cell."""
    pi_row = torch.full((1, C), 1.0 / C, dtype=torch.float32)
    M_t = torch.as_tensor(ann.confusion_matrix, dtype=torch.float32)
    bel_by_label = torch.empty((C, C), dtype=torch.float32)
    for y in range(C):
        b = pi_row.squeeze(0) * M_t[:, y]
        bel_by_label[y] = b / b.sum().clamp_min(LOG_EPS)
    beliefs = bel_by_label[torch.as_tensor(observed_full)]
    return beliefs, torch.log(beliefs.clamp_min(LOG_EPS)), pi_row


def eval_cell(arm, ntag, seed, feats, test_x, test_y, clean):
    """rows: (arm, evaluator, noise, seed, budget, test_acc)"""
    run = find_run(ntag, arm, seed)
    lset_full, observed_full, ann = load_lset_and_observed(run, ntag, seed, clean)
    rows = []
    for B in BUDGETS:
        acc = acc_1nn(feats[lset_full[:B]], observed_full[:B], test_x, test_y)
        rows.append((arm, '1nn_observed', NOISES[ntag], seed, B, acc))

    if arm == 'bayes_product':
        lf = torch.as_tensor(feats[lset_full])                           # [1000, 512]
        # (b) 1-NN on posterior-argmax relabels
        relabels = posterior_argmax_labels(lf, observed_full, ann)
        for B in BUDGETS:
            acc = acc_1nn(feats[lset_full[:B]], relabels[B], test_x, test_y)
            rows.append((arm, '1nn_postargmax', NOISES[ntag], seed, B, acc))
        # (c) gap-scaled NW mixture over the labeled anchors, on test points
        beliefs, log_beliefs, pi_row = noise_aware_beliefs(observed_full, ann)
        d_test = torch.cdist(torch.as_tensor(test_x), lf)                # [10000, 1000]
        test_y_t = torch.as_tensor(test_y)
        for B in BUDGETS:
            for name, variant in NW_VARIANTS.items():
                P = posterior_variant(variant, d_test[:, :B], beliefs[:B], log_beliefs[:B], pi_row)
                acc = 100.0 * float((P.argmax(dim=1) == test_y_t).float().mean())
                rows.append((arm, name, NOISES[ntag], seed, B, acc))
    return rows


ROW_LABELS = [  # (arm, evaluator, table row label) in §4 row order
    ('random', '1nn_observed', 'random + 1nn'),
    ('probcover', '1nn_observed', 'probcover + 1nn'),
    ('bayes_additive', '1nn_observed', 'bayes_additive + 1nn'),
    ('bayes_product', '1nn_observed', 'bayes_product + 1nn'),
    ('bayes_product', '1nn_postargmax', 'bayes_product + 1nn post-argmax'),
    ('bayes_product', 'nw_k1', 'bayes_product + nw κ=1'),
    ('bayes_product', 'nw_k3', 'bayes_product + nw κ=3'),
]


def table_md(rows, ntag):
    nr = NOISES[ntag]
    lines = [f'### Noise = {nr}', '',
             '| Method | ' + ' | '.join(str(b) for b in BUDGETS) + ' |',
             '|---|' + '---|' * len(BUDGETS)]
    for arm, ev, label in ROW_LABELS:
        cells = []
        for B in BUDGETS:
            v = [r[5] for r in rows if r[0] == arm and r[1] == ev and r[2] == nr and r[4] == B]
            cells.append(f'{np.mean(v):.1f}±{np.std(v):.1f}' if v else '—')
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    clean = load_clean_labels()
    feats = load_features()                                              # train, unit-L2, [50000, 512]
    test_x, test_y = load_test()                                         # unit-L2, [10000, 512]

    rows = []
    for arm in ARMS:
        for ntag in NOISES:
            for seed in SEEDS:
                cell = eval_cell(arm, ntag, seed, feats, test_x, test_y, clean)
                rows.extend(cell)
                a100 = next(r[5] for r in cell if r[1] == '1nn_observed' and r[4] == 100)
                print(f'{arm:<15}{ntag} seed{seed}: 1nn_observed@100={a100:.1f}')
        if arm == 'probcover':
            print('\n=== SANITY GATE: probcover 1nn_observed vs live sweep 31048649 ===')
            ok = True
            for (ntag, B), target in GATE.items():
                v = [r[5] for r in rows if r[0] == 'probcover' and r[1] == '1nn_observed'
                     and r[2] == NOISES[ntag] and r[4] == B]
                mean = np.mean(v)
                good = abs(mean - target) <= GATE_TOL
                ok &= good
                print(f'  {ntag} @{B}: replay {mean:.2f} vs live {target}  '
                      f'(|Δ|={abs(mean - target):.2f}) {"OK" if good else "FAIL"}')
            if not ok:
                sys.exit('SANITY GATE FAILED — check normalization / label source before trusting anything.')
            print('  GATE PASSED\n')

    with open(os.path.join(OUT_DIR, 'per_episode.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'evaluator', 'noise', 'seed', 'budget', 'test_acc'])
        w.writerows(rows)

    tables = ['# 1-NN + NW evaluator columns for the §4 selection arms',
              '', 'Offline replay (`batch_utils/eval_1nn_all_arms.py`) of sweep 30978366;',
              'mean±std over seeds {1,2,3}; sanity gate vs sweep 31048649 passed.', '']
    for ntag in NOISES:
        tables.append(table_md(rows, ntag))
        tables.append('')
    md = '\n'.join(tables)
    with open(os.path.join(OUT_DIR, 'tables.md'), 'w') as f:
        f.write(md)
    print(md)
    print('written', os.path.join(OUT_DIR, 'per_episode.csv'))
    print('written', os.path.join(OUT_DIR, 'tables.md'))


if __name__ == '__main__':
    main()
