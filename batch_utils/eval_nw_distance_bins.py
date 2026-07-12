#!/usr/bin/env python3
"""
Offline NW pseudo-target accuracy vs DISTANCE from the labeled set on UNCOVERED
pool points (docs/bayesian_coverage_NEXT_STEPS.md §7.9 follow-up) — CPU only.

§7.9 killed constant out-of-ball NW loss weights: far targets are "confidently
wrong" (~26% mean uncovered accuracy at 100 clean). Open question: does that
mean hide a usable high-accuracy SHELL just outside the delta ball (d slightly
> 0.75)? If yes, a smooth distance-decayed weight w(d) has room; if accuracy
collapses immediately past delta, it doesn't.

Replays the bayes_product runs of SELECTION sweep 30978366 (noise {0,.2,.5,.8}
x seeds {1,2,3}), budgets {100,300,1000}. Per uncovered pool point (nearest
labeled distance d > 0.75) computes the gap-scaled NW target from the
noise-aware beliefs (eval_posterior_variants.posterior_variant, kappa=3 and
kappa=1, m=7), bins by d:
  (0.75,0.80] (0.80,0.85] (0.85,0.90] (0.90,0.95] (0.95,1.00] (1.00,1.05] (1.05,inf)
and reports per bin: point count, NW-argmax accuracy vs clean labels, and the
target's mean top-1 mixture weight (confidence proxy). Reference rows per
(noise, budget): covered-points product-posterior argmax accuracy (what
pseudo_all trains on) and overall uncovered NW accuracy (SANITY GATE: ~26% at
100 clean, vs §7.9 / posterior_variants.json).

Writes results/cifar100/eval_nw_distance_bins/bins.csv (long format) +
tables.md (one table per (noise, budget), bins as rows).
Run:  python batch_utils/eval_nw_distance_bins.py          (CPU, ~10 min)
"""
import numpy as np, torch, os, csv, sys

BASE = '/cs/labs/daphna/alongol/projects/daphna_lab'
OUT_DIR = os.path.join(BASE, 'results/cifar100/eval_nw_distance_bins')
os.makedirs(OUT_DIR, exist_ok=True)
DELTA = 0.75
BUDGETS = [100, 300, 1000]
NOISES = {'n00': 0.0, 'n20': 0.2, 'n50': 0.5, 'n80': 0.8}
SEEDS = [1, 2, 3]
KAPPAS = {'k3': 'nw_gap_k3', 'k1': 'nw_gap_all'}     # kappa=3 / kappa=1, m=7
BIN_EDGES = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05]
BIN_LABELS = ['(0.75,0.80]', '(0.80,0.85]', '(0.85,0.90]', '(0.90,0.95]',
              '(0.95,1.00]', '(1.00,1.05]', '(1.05,inf)']
ALL_UNCOV, COVERED = 'ALL uncovered', 'COVERED (product)'   # reference "bins"

# sanity gate: §7.9 / posterior_variants.json nw_gap_k3 acc_uncov @ n00 B=100
GATE_TARGET, GATE_TOL = 25.7, 1.5

sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'batch_utils'))
from eval_posterior_variants import posterior_variant, load_clean_labels, load_features
from eval_1nn_all_arms import find_run, load_lset_and_observed, noise_aware_beliefs


def eval_cell(ntag, seed, feats, clean_t):
    """rows: (noise, seed, budget, kappa, bin, count, acc, mean_top1_weight)"""
    run = find_run(ntag, 'bayes_product', seed)
    lset_full, observed_full, ann = load_lset_and_observed(run, ntag, seed,
                                                           clean_t.numpy())
    beliefs, log_beliefs, pi_row = noise_aware_beliefs(observed_full, ann)
    lf = feats[torch.as_tensor(lset_full)]                   # [1000, 512]
    d_full = torch.empty((feats.shape[0], len(lset_full)), dtype=torch.float32)
    bs = 8192
    for i in range(0, feats.shape[0], bs):
        d_full[i:i + bs] = torch.cdist(feats[i:i + bs], lf)

    ld = lambda p: np.load(p, allow_pickle=True).astype(int)
    nr, rows = NOISES[ntag], []
    for B in BUDGETS:
        uset = ld(os.path.join(run, f'episode_{B // 100 - 1}/uSet.npy'))
        uidx = torch.as_tensor(uset)
        d_u = d_full[uidx, :B]                               # [U, B]
        dmin = d_u.min(dim=1).values
        uncov = ~((d_u < DELTA).sum(dim=1) > 0)              # same test as live
        cu = clean_t[uidx]

        # reference: covered-points product-posterior argmax (pseudo_all's diet)
        Pc = posterior_variant('product', d_u[~uncov], beliefs[:B],
                               log_beliefs[:B], pi_row)
        conf_c, amax_c = Pc.max(dim=1)
        acc_c = 100.0 * float((amax_c == cu[~uncov]).float().mean())
        rows.append((nr, seed, B, 'product', COVERED, int((~uncov).sum()),
                     acc_c, float(conf_c.mean())))

        du, cu_unc = dmin[uncov].numpy(), cu[uncov]
        binidx = np.digitize(du, BIN_EDGES, right=True)      # left-open bins
        for kname, variant in KAPPAS.items():
            P = posterior_variant(variant, d_u[uncov], beliefs[:B],
                                  log_beliefs[:B], pi_row)
            conf, amax = P.max(dim=1)
            corr = (amax == cu_unc).numpy()
            conf = conf.numpy()
            rows.append((nr, seed, B, kname, ALL_UNCOV, len(du),
                         100.0 * float(corr.mean()), float(conf.mean())))
            for b, label in enumerate(BIN_LABELS):
                m = binidx == b
                rows.append((nr, seed, B, kname, label, int(m.sum()),
                             100.0 * float(corr[m].mean()) if m.sum() else float('nan'),
                             float(conf[m].mean()) if m.sum() else float('nan')))
    return rows


def agg(rows, nr, B, kappa, label, col):
    v = [r[col] for r in rows if r[0] == nr and r[2] == B and r[3] == kappa
         and r[4] == label and not (isinstance(r[col], float) and np.isnan(r[col]))]
    return v


def table_md(rows, ntag, B):
    nr = NOISES[ntag]
    lines = [f'### Noise = {nr}, budget = {B}', '',
             '| d bin | count | acc κ=3 | top1 wt κ=3 | acc κ=1 |',
             '|---|---|---|---|---|']
    for label in BIN_LABELS + [ALL_UNCOV, COVERED]:
        k3 = 'product' if label == COVERED else 'k3'
        cnt = agg(rows, nr, B, k3, label, 5)
        a3 = agg(rows, nr, B, k3, label, 6)
        w3 = agg(rows, nr, B, k3, label, 7)
        c_cnt = f'{np.mean(cnt):.0f}' if cnt else '0'
        c_a3 = f'{np.mean(a3):.1f}±{np.std(a3):.1f}' if a3 else '—'
        c_w3 = f'{np.mean(w3):.2f}' if w3 else '—'
        if label == COVERED:
            c_a1 = '—'
        else:
            a1 = agg(rows, nr, B, 'k1', label, 6)
            c_a1 = f'{np.mean(a1):.1f}±{np.std(a1):.1f}' if a1 else '—'
        lines.append(f'| {label} | {c_cnt} | {c_a3} | {c_w3} | {c_a1} |')
    return '\n'.join(lines)


def main():
    clean_t = torch.as_tensor(load_clean_labels())
    feats = torch.as_tensor(load_features())                 # unit-L2, [50000, 512]

    rows = []
    for ntag in NOISES:
        for seed in SEEDS:
            cell = eval_cell(ntag, seed, feats, clean_t)
            rows.extend(cell)
            a = next(r[6] for r in cell if r[2] == 100 and r[3] == 'k3'
                     and r[4] == ALL_UNCOV)
            print(f'{ntag} seed{seed}: uncovered nw_k3 acc@100 = {a:.1f}')

    gate = np.mean(agg(rows, 0.0, 100, 'k3', ALL_UNCOV, 6))
    print(f'\n=== SANITY GATE: n00 @100 uncovered nw_k3 mean {gate:.2f} vs '
          f'§7.9 {GATE_TARGET} (|Δ|={abs(gate - GATE_TARGET):.2f}) ===')
    if abs(gate - GATE_TARGET) > GATE_TOL:
        sys.exit('SANITY GATE FAILED — check lSets / beliefs / kernel before trusting bins.')
    print('GATE PASSED\n')

    with open(os.path.join(OUT_DIR, 'bins.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['noise', 'seed', 'budget', 'kappa', 'bin', 'count', 'acc',
                    'mean_top1_weight'])
        w.writerows(rows)

    tables = ['# NW pseudo-target accuracy vs distance from the labeled set '
              '(uncovered pool points)', '',
              'Offline replay (`batch_utils/eval_nw_distance_bins.py`) of the '
              'bayes_product runs of sweep 30978366;',
              'gap-scaled NW (m=7) from noise-aware beliefs; bins over nearest-'
              'labeled distance d (delta=0.75); count = mean over seeds {1,2,3}, '
              'acc = mean±std; top1 wt = target\'s top-class probability.',
              f'Sanity gate vs §7.9 passed ({gate:.1f} vs {GATE_TARGET} at '
              'noise 0 @100).', '']
    for ntag in NOISES:
        for B in BUDGETS:
            tables += [table_md(rows, ntag, B), '']
    md = '\n'.join(tables)
    with open(os.path.join(OUT_DIR, 'tables.md'), 'w') as f:
        f.write(md)
    print(md)
    print('written', os.path.join(OUT_DIR, 'bins.csv'))
    print('written', os.path.join(OUT_DIR, 'tables.md'))


if __name__ == '__main__':
    main()
