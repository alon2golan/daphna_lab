#!/usr/bin/env python3
"""
NEXT_STEPS §9.2 stage 1: offline δ sweep of the §4.5 posterior replay.

Replays the recorded bayes_product runs (lSet trajectory + annotation records
held FIXED — δ shapes only the posterior here, not selection; stage 2 is a
small paired training sweep to confirm the winner) and rebuilds the posterior
at δ ∈ {0.5, 0.6, 0.75, 0.9, 1.1}. Same metrics as eval_posterior_quality.py:
denoising (post_acc/flips), noise-ID (LOO prec/rec/AUC), coverage, pseudo-label
quality (cov_acc/topk), calibration (ECE).

Differences from eval_posterior_quality.py (which this imports helpers from):
  - CPU-capable: the [N,1000] distance matrix is computed once per run and
    thresholded at every δ (the expensive part is δ-independent).
  - The accumulator folds one labeled neighbor at a time IN lSet ORDER, like
    the live code — float32 summation order decides LOO argmax near-ties
    (block matmul deviated on 12/1000 kept decisions in testing), but only the
    rows inside the new ball are touched, so CPU stays fast.
  - Sanity gate: at δ=0.75 the LOO keep counts are compared to the counts the
    real runs printed to their SLURM logs (kept_match; report, don't hide).

Output: results/cifar100/delta_replay/delta_replay.{json,csv}
"""
import numpy as np, torch, json, glob, os, sys, csv

BASE = '/cs/labs/daphna/alongol/projects/daphna_lab'
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'batch_utils'))
from eval_posterior_quality import (roc_auc, ece, load_clean_labels,
                                    load_features, parse_filtered_log_counts,
                                    NOISES, SEEDS, BUDGETS, C, LOG_EPS)
from deep_al.pycls.noise.annotator import Annotator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DELTAS = [0.5, 0.6, 0.75, 0.9, 1.1]
OUT_DIR = os.path.join(BASE, 'results/cifar100/delta_replay')
os.makedirs(OUT_DIR, exist_ok=True)


def load_run(ntag, seed, clean):
    nr = NOISES[ntag]
    pat = os.path.join(BASE, f'output/CIFAR100/resnet18/2026_7_2/cfgBAYESCOV_{ntag}_*_albayes_product_train_label_modeobserved_seed{seed}_*')
    run = sorted(glob.glob(pat))[0]
    ld = lambda p: np.load(p, allow_pickle=True).astype(int)
    lset_full = ld(os.path.join(run, 'episode_9/lSet.npy'))
    assert len(lset_full) == 1000
    records = {int(k): int(v) for k, v in json.load(open(os.path.join(run, 'episode_8/annotated_labels.json'))).items()}
    observed_full = np.array([records[g] for g in lset_full], dtype=np.int64)
    usets = {B: ld(os.path.join(run, f'episode_{B // 100 - 1}/uSet.npy')) for B in BUDGETS}

    class _Stub: pass
    stub = _Stub(); stub.targets = clean.tolist()
    ann = Annotator(dataset=stub, noise_type='sym', noise_rate=nr, cost=1, seed=seed, num_classes=C)
    regen_ok = bool(np.array_equal(ann.noisy_labels[lset_full], observed_full))
    return lset_full, observed_full, usets, ann, regen_ok


def dist_matrix(feats, lset_full):
    """[N, 1000] pairwise distances to the full labeled trajectory (δ-free)."""
    lf = feats[torch.as_tensor(lset_full, device=DEVICE)]
    N = feats.shape[0]
    D = torch.empty((N, 1000), dtype=torch.float32, device=DEVICE)
    bs = 8192
    for i in range(0, N, bs):
        D[i:i + bs] = torch.cdist(feats[i:i + bs].unsqueeze(0), lf.unsqueeze(0)).squeeze(0)
    return D


def replay_delta(delta, D, lset_full, observed_full, usets, ann, clean_gpu,
                 is_noisy_full, log_kept_by_B):
    inball = (D < delta).float()
    pi_t = torch.full((C,), 1.0 / C, device=DEVICE, dtype=torch.float32)
    M_t = torch.as_tensor(ann.confusion_matrix, device=DEVICE, dtype=torch.float32)
    logb_by_label = torch.empty((C, C), dtype=torch.float32, device=DEVICE)
    for y in range(C):
        b = pi_t * M_t[:, y]
        b = b / b.sum().clamp_min(LOG_EPS)
        logb_by_label[y] = torch.log(b.clamp_min(LOG_EPS))
    logb = logb_by_label[torch.as_tensor(observed_full, device=DEVICE)]     # [1000, C]

    N = D.shape[0]
    Cmat = torch.log(pi_t.clamp_min(LOG_EPS)).repeat(N, 1)
    folded = 0
    out = {}
    for B in BUDGETS:
        while folded < B:
            rows = inball[:, folded].nonzero(as_tuple=True)[0]
            Cmat[rows] += logb[folded]
            folded += 1
        P = torch.softmax(Cmat, dim=1)
        lidx = torch.as_tensor(lset_full[:B], device=DEVICE)
        uidx = torch.as_tensor(usets[B], device=DEVICE)
        obs = torch.as_tensor(observed_full[:B], device=DEVICE)

        # ---- labeled plane (denoising)
        Pl = P[lidx]
        amax_l = Pl.argmax(dim=1)
        cl = clean_gpu[lidx]
        observed_acc = float((obs == cl).float().mean())
        post_acc = float((amax_l == cl).float().mean())
        flip = amax_l != obs
        n_flip = int(flip.sum())
        flips_right = int(((amax_l == cl) & flip).sum())
        good_overturned = int(((obs == cl) & flip).sum())

        # LOO noise-ID
        Cl_loo = Cmat[lidx] - logb[:B]
        Ploo = torch.softmax(Cl_loo, dim=1)
        keep = Ploo.argmax(dim=1) == obs
        kept = int(keep.sum())
        w = Ploo[torch.arange(B, device=DEVICE), obs].cpu().numpy()
        noisy_mask = is_noisy_full[:B]
        flagged = (~keep).cpu().numpy()
        prec = float(noisy_mask[flagged].mean()) if flagged.sum() else float('nan')
        rec = float(flagged[noisy_mask].mean()) if noisy_mask.sum() else float('nan')
        auc = roc_auc(1.0 - w[noisy_mask], 1.0 - w[~noisy_mask]) if noisy_mask.sum() and (~noisy_mask).sum() else float('nan')

        # ---- unlabeled plane (coverage / pseudo quality / calibration)
        Pu = P[uidx]
        conf_u, amax_u = Pu.max(dim=1)
        covered = inball[uidx, :B].sum(dim=1) > 0
        cov_frac = float(covered.float().mean())
        cu = clean_gpu[uidx]
        correct_u = (amax_u == cu)
        cov_acc = float(correct_u[covered].float().mean()) if covered.sum() else float('nan')
        all_acc_u = float(correct_u.float().mean())
        uncov_acc = float(correct_u[~covered].float().mean()) if (~covered).sum() else float('nan')
        conf_cov = conf_u[covered].cpu().numpy()
        corr_cov = correct_u[covered].cpu().numpy()
        ece_cov = ece(conf_cov, corr_cov) if covered.sum() else float('nan')
        mean_conf = float(conf_cov.mean()) if covered.sum() else float('nan')
        topk_acc = {}
        conf_np = conf_u.cpu().numpy(); amax_np = amax_u.cpu().numpy()
        corr_np = correct_u.cpu().numpy(); cov_np = covered.cpu().numpy()
        for k in (1, 5, 10):
            sel = []
            for c in range(C):
                m = np.where(cov_np & (amax_np == c))[0]
                if len(m) == 0:
                    continue
                sel.extend(m[np.argsort(-conf_np[m])[:k]])
            topk_acc[k] = {'n': len(sel), 'acc': float(corr_np[sel].mean()) if sel else float('nan')}

        log_kept = log_kept_by_B.get(B)
        out[B] = dict(
            observed_acc=observed_acc, post_acc=post_acc,
            n_flip=n_flip, flips_right=flips_right, good_overturned=good_overturned,
            kept=kept, kept_logged=log_kept,
            kept_match=(log_kept == kept if (log_kept is not None and delta == 0.75) else None),
            noise_prec=prec, noise_rec=rec, noise_auc=auc,
            loo_disagree_frac=1.0 - kept / B,
            cov_frac=cov_frac, cov_acc=cov_acc, all_acc_u=all_acc_u, uncov_acc=uncov_acc,
            ece=ece_cov, mean_conf=mean_conf, topk=topk_acc,
        )
    return out


def main():
    clean = load_clean_labels()
    feats = torch.as_tensor(load_features(), device=DEVICE)
    clean_gpu = torch.as_tensor(clean, device=DEVICE)
    filtered_counts = parse_filtered_log_counts()
    torch.set_num_threads(os.cpu_count() or 8)

    results = []
    for ntag in NOISES:
        for seed in SEEDS:
            lset_full, observed_full, usets, ann, regen_ok = load_run(ntag, seed, clean)
            if not regen_ok:
                print(f'!! {ntag} seed{seed}: annotator regen MISMATCH — skipping', flush=True)
                continue
            D = dist_matrix(feats, lset_full)
            is_noisy_full = observed_full != clean[lset_full]
            log_kept_by_B = filtered_counts.get((ntag, seed), {})
            for delta in DELTAS:
                cell = dict(ntag=ntag, noise=NOISES[ntag], seed=seed, delta=delta,
                            regen_ok=regen_ok,
                            budgets=replay_delta(delta, D, lset_full, observed_full,
                                                 usets, ann, clean_gpu, is_noisy_full,
                                                 log_kept_by_B))
                results.append(cell)
                gate = [v['kept_match'] for v in cell['budgets'].values() if v['kept_match'] is not None]
                gate_s = f" gate {sum(gate)}/{len(gate)}" if gate else ""
                b1000 = cell['budgets'][1000]
                print(f'{ntag} seed{seed} d={delta:.2f}: cov {b1000["cov_frac"]:.3f} '
                      f'post_acc {b1000["post_acc"]:.3f} cov_acc {b1000["cov_acc"]:.3f} '
                      f'ece {b1000["ece"]:.3f}{gate_s}', flush=True)
            del D

    with open(os.path.join(OUT_DIR, 'delta_replay.json'), 'w') as f:
        json.dump(results, f)

    # flat CSV, one row per (noise, seed, delta, budget)
    with open(os.path.join(OUT_DIR, 'delta_replay.csv'), 'w', newline='') as f:
        wcsv = None
        for cell in results:
            for B, m in cell['budgets'].items():
                row = dict(ntag=cell['ntag'], noise=cell['noise'], seed=cell['seed'],
                           delta=cell['delta'], budget=B,
                           **{k: v for k, v in m.items() if k != 'topk'},
                           top1_acc=m['topk'][1]['acc'], top5_acc=m['topk'][5]['acc'],
                           top10_acc=m['topk'][10]['acc'])
                if wcsv is None:
                    wcsv = csv.DictWriter(f, fieldnames=list(row))
                    wcsv.writeheader()
                wcsv.writerow(row)
    print(f'wrote {OUT_DIR}/delta_replay.json + .csv  ({len(results)} cells)')


if __name__ == '__main__':
    main()
