#!/usr/bin/env python3
"""
Offline comparison of in-ball posterior FUSION VARIANTS (§7.9 follow-up: "would
NW also be better INSIDE the balls?") — CPU only, replays from the same persisted
artifacts as eval_posterior_quality.py (lSet trajectories + annotation records of
the 2026_7_2 trainmode runs; identical across arms).

Variants (all share the same anchors/beliefs; only the fusion/weighting differs):
  product      tophat + product of experts, beta=1        (the live method)
  rbf_product  product with distance-decayed exponents inside the ball,
               w(d)=exp(-(d/sigma)^2), sigma=delta        (NW ingredient #1)
  geo_product  product with exponents normalized per point (1/n_u) — the
               geometric-mean / "NW-limit" of the product family (ingredient #2,
               beta = 1/n_u; caps sharpness like a mixture, keeps multiplicative)
  nw_inball    NW MIXTURE over in-ball anchors only (fusion change, same range)
  nw_all       NW mixture over ALL anchors, self-tuning T_u = d1*dm (m=7) —
               §7.9 semantics applied everywhere (fusion + range change)

Metrics (UNLABELED plane — what pseudo-labels actually use; labeled points are
excluded from pseudo targets anyway):
  acc_cov / acc_uncov / acc_all : argmax-vs-clean accuracy on covered /
                                  uncovered / all unlabeled pool points
  ece_cov                       : calibration on covered points
Plus labeled-plane denoising acc (argmax vs clean) for the product family
(NW-all on labeled points is dominated by the self-anchor; skipped).

Writes results/cifar100/posterior_eval/posterior_variants.json + prints a table.
Run:  python batch_utils/eval_posterior_variants.py          (CPU, ~15 min)
"""
import numpy as np, torch, json, glob, os, pickle, sys

BASE = '/cs/labs/daphna/alongol/projects/daphna_lab'
OUT_DIR = os.path.join(BASE, 'results/cifar100/posterior_eval')
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = 'cpu'
DELTA = 0.75
SIGMA = DELTA          # rbf_product: w(delta)=e^-1~0.37, w(0)=1
NW_M = 7
C = 100
LOG_EPS = 1e-12
BUDGETS = [100, 300, 1000]
NOISES = {'n00': 0.0, 'n20': 0.2, 'n50': 0.5, 'n80': 0.8}
SEEDS = [1, 2, 3]
VARIANTS = ['product', 'rbf_product', 'geo_product', 'nw_inball', 'nw_all',
            'nn1_all', 'nw_gap_all', 'nw_gap_k3', 'nw_gap_k8']

sys.path.insert(0, BASE)
from deep_al.pycls.noise.annotator import Annotator


def load_clean_labels():
    with open(os.path.join(BASE, 'data/cifar-100-python/train'), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return np.array(d[b'fine_labels'], dtype=np.int64)


def load_features():
    f = np.load('/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy')
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    return f.astype(np.float32)


def ece(conf, correct, n_bins=10):
    conf = np.asarray(conf); correct = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return float('nan')
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def posterior_variant(variant, d, beliefs, log_beliefs, pi_row):
    """
    d           [N, B]  distances to the B anchors
    beliefs     [B, C]  b_l rows;  log_beliefs = log(b_l.clamp_min(eps))
    returns P   [N, C]
    """
    inball = (d < DELTA).float()
    n_u = inball.sum(dim=1, keepdim=True)                       # [N,1]
    logpi = torch.log(pi_row.clamp_min(LOG_EPS))
    if variant == 'product':
        Cmat = logpi + inball @ log_beliefs
        return torch.softmax(Cmat, dim=1)
    if variant == 'rbf_product':
        w = inball * torch.exp(-(d / SIGMA) ** 2)
        Cmat = logpi + w @ log_beliefs
        return torch.softmax(Cmat, dim=1)
    if variant == 'geo_product':
        w = inball / n_u.clamp_min(1.0)                          # exponents sum to 1
        Cmat = logpi + w @ log_beliefs
        P = torch.softmax(Cmat, dim=1)
        # uncovered rows: exponents all zero -> prior (same as product)
        return P
    if variant == 'nw_inball':
        # mixture over in-ball anchors, kernel-flat (tophat) weights;
        # uncovered rows -> prior
        w = inball / n_u.clamp_min(1.0)
        P = w @ beliefs
        P = torch.where(n_u > 0, P, pi_row.expand_as(P))
        return P
    if variant == 'nw_all':
        d_sorted = d.sort(dim=1).values
        m = min(NW_M, d.shape[1]) - 1
        T = (d_sorted[:, 0] * d_sorted[:, m]).clamp_min(LOG_EPS).unsqueeze(1)
        w = torch.softmax(-(d ** 2) / T, dim=1)
        return w @ beliefs
    if variant == 'nn1_all':
        # hard nearest-anchor belief — the T->0 limit; upper-bounds how sharp
        # any competitive rule can commit, and measures the raw 1-NN signal.
        nn = d.argmin(dim=1)
        return beliefs[nn]
    if variant.startswith('nw_gap'):
        # gap-scaled competition: exponent = -kappa*(d^2 - d_1^2)/(d_m^2 - d_1^2).
        # Dimensionless; immune to high-D distance concentration (the raw
        # d^2/T softmax is too flat when d_1 ~ d_50): the nearest anchor gets
        # weight 1, the m-th nearest e^-kappa. kappa interpolates soft-kNN
        # aggregation (1) -> hard 1-NN commitment (large).
        kappa = {'nw_gap_all': 1.0, 'nw_gap_k3': 3.0, 'nw_gap_k8': 8.0}[variant]
        d2 = d ** 2
        d_sorted2 = d2.sort(dim=1).values
        m = min(NW_M, d.shape[1]) - 1
        gap = (d_sorted2[:, m] - d_sorted2[:, 0]).clamp_min(1e-6).unsqueeze(1)
        w = torch.softmax(-kappa * (d2 - d_sorted2[:, :1]) / gap, dim=1)
        return w @ beliefs
    raise ValueError(variant)


def eval_cell(ntag, seed, feats, clean):
    nr = NOISES[ntag]
    pat = os.path.join(BASE, f'output/CIFAR100/resnet18/2026_7_2/cfgBAYESCOV_{ntag}_*_albayes_product_train_label_modeobserved_seed{seed}_*')
    run = sorted(glob.glob(pat))[0]
    ld = lambda p: np.load(p, allow_pickle=True).astype(int)
    lset_full = ld(os.path.join(run, 'episode_9/lSet.npy'))
    records = {int(k): int(v) for k, v in json.load(open(os.path.join(run, 'episode_8/annotated_labels.json'))).items()}
    observed_full = np.array([records[g] for g in lset_full], dtype=np.int64)

    class _Stub: pass
    stub = _Stub(); stub.targets = clean.tolist()
    ann = Annotator(dataset=stub, noise_type='sym', noise_rate=nr, cost=1, seed=seed, num_classes=C)
    assert np.array_equal(ann.noisy_labels[lset_full], observed_full), 'annotation regen mismatch'

    pi_row = torch.full((1, C), 1.0 / C, dtype=torch.float32)
    M_t = torch.as_tensor(ann.confusion_matrix, dtype=torch.float32)
    bel_by_label = torch.empty((C, C), dtype=torch.float32)
    for y in range(C):
        b = (pi_row.squeeze(0) * M_t[:, y])
        bel_by_label[y] = b / b.sum().clamp_min(LOG_EPS)
    beliefs_full = bel_by_label[torch.as_tensor(observed_full)]          # [1000, C]
    log_beliefs_full = torch.log(beliefs_full.clamp_min(LOG_EPS))

    lf = feats[torch.as_tensor(lset_full)]
    # distances of ALL 50k points to the 1000 anchors, chunked
    N = feats.shape[0]
    d_full = torch.empty((N, len(lset_full)), dtype=torch.float32)
    bs = 8192
    for i in range(0, N, bs):
        d_full[i:i + bs] = torch.cdist(feats[i:i + bs], lf)

    clean_t = torch.as_tensor(clean)
    out = {'ntag': ntag, 'noise': nr, 'seed': seed, 'budgets': {}}
    for B in BUDGETS:
        ep = B // 100 - 1
        uset = ld(os.path.join(run, f'episode_{ep}/uSet.npy'))
        uidx = torch.as_tensor(uset)
        d_B = d_full[:, :B]
        covered_u = (d_B[uidx] < DELTA).sum(dim=1) > 0
        cu = clean_t[uidx]
        lidx = torch.as_tensor(lset_full[:B])
        cl = clean_t[lidx]
        cell = {'cov_frac': float(covered_u.float().mean())}
        for v in VARIANTS:
            P = posterior_variant(v, d_B, beliefs_full[:B], log_beliefs_full[:B], pi_row)
            Pu = P[uidx]
            conf, amax = Pu.max(dim=1)
            corr = (amax == cu)
            r = dict(
                acc_cov=float(corr[covered_u].float().mean()) if covered_u.sum() else float('nan'),
                acc_uncov=float(corr[~covered_u].float().mean()) if (~covered_u).sum() else float('nan'),
                acc_all=float(corr.float().mean()),
                ece_cov=ece(conf[covered_u].numpy(), corr[covered_u].numpy()),
            )
            if not v.endswith('_all') and not v.startswith('nw_gap'):   # self-anchor dominates the *_all variants
                Pl = P[lidx]
                r['denoise_acc'] = float((Pl.argmax(dim=1) == cl).float().mean())
            cell[v] = r
        out['budgets'][B] = cell
    return out


def main():
    clean = load_clean_labels()
    feats = torch.as_tensor(load_features())
    results = []
    for ntag in NOISES:
        for seed in SEEDS:
            r = eval_cell(ntag, seed, feats, clean)
            results.append(r)
            print(f'{ntag} seed{seed} done')
    with open(os.path.join(OUT_DIR, 'posterior_variants.json'), 'w') as f:
        json.dump(results, f, indent=1)

    # summary: mean over seeds
    print('\n=== unlabeled-plane pseudo-target accuracy (mean over seeds) ===')
    hdr = f'{"noise":<7}{"B":<6}{"cov%":<7}' + ''.join(f'{v:>13}' for v in VARIANTS)
    for metric in ('acc_cov', 'acc_uncov', 'acc_all', 'ece_cov'):
        print(f'\n--- {metric} ---\n{hdr}')
        for ntag, nr in NOISES.items():
            for B in BUDGETS:
                cells = [r['budgets'][str(B)] if str(B) in r['budgets'] else r['budgets'][B]
                         for r in results if r['ntag'] == ntag]
                cov = np.mean([c['cov_frac'] for c in cells])
                row = f'{nr:<7}{B:<6}{cov*100:<7.1f}'
                for v in VARIANTS:
                    vals = [c[v][metric] for c in cells if not np.isnan(c[v].get(metric, np.nan))]
                    row += f'{np.mean(vals)*100:>13.1f}' if vals else f'{"—":>13}'
                print(row)
    _den = [v for v in VARIANTS if not v.endswith('_all') and not v.startswith('nw_gap')]
    print('\n--- labeled-plane denoise_acc (argmax vs clean) ---')
    print(f'{"noise":<7}{"B":<6}' + ''.join(f'{v:>13}' for v in _den))
    for ntag, nr in NOISES.items():
        for B in BUDGETS:
            cells = [r['budgets'][B] for r in results if r['ntag'] == ntag]
            row = f'{nr:<7}{B:<6}'
            for v in _den:
                row += f'{np.mean([c[v]["denoise_acc"] for c in cells])*100:>13.1f}'
            print(row)
    print('\nwritten', os.path.join(OUT_DIR, 'posterior_variants.json'))


if __name__ == '__main__':
    main()
