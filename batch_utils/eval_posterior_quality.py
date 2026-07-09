#!/usr/bin/env python3
"""
Offline evaluation of the BayesianCoverage posterior against ground truth.

Replays the exact posterior each bayes_product run saw (tophat delta=0.75,
product-of-experts, beta=1, kernel weights) from persisted artifacts:
  - lSet trajectory: episode_*/lSet.npy   (identical across all 5 train-mode arms)
  - annotation records: episode_8/annotated_labels.json (covers all 1000 picks)
  - features: the run's own SimCLR CIFAR100 train features, L2-normalized
Validation targets:
  (a) regenerated Annotator noisy labels == recorded annotations
  (b) LOO 'filtered' keep counts == the counts the real runs printed to SLURM logs
  (c) noise=0 posterior argmax on labeled points == clean labels
"""
import numpy as np, torch, json, glob, os, pickle, re, sys

BASE = '/cs/labs/daphna/alongol/projects/daphna_lab'
OUT_DIR = os.path.join(BASE, 'results/cifar100/posterior_eval')
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = 'cuda'
DELTA = 0.75
C = 100
LOG_EPS = 1e-12
BUDGETS = list(range(100, 1001, 100))
NOISES = {'n00': 0.0, 'n20': 0.2, 'n50': 0.5, 'n80': 0.8}
SEEDS = [1, 2, 3]

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


def parse_filtered_log_counts():
    """{(ntag, seed): {budget: kept}} from the real runs' SLURM stdout."""
    out = {}
    for path in glob.glob(os.path.join(BASE, 'logs/sweep/sweep_30980646_*.out')):
        txt = open(path, errors='replace').read()
        m = re.search(r'cfgBAYESCOV_(n\d\d)_tagnoise=[\dp]+_albayes_product_train_label_modefiltered_seed(\d)', txt)
        if not m:
            continue
        key = (m.group(1), int(m.group(2)))
        kept = {}
        for km in re.finditer(r'\[bayes:filtered\] kept (\d+)/(\d+) LOO-clean', txt):
            kept[int(km.group(2))] = int(km.group(1))
        if kept:
            out[key] = kept
    return out


def roc_auc(scores_pos, scores_neg):
    """AUC = P(score_pos > score_neg) + 0.5*P(equal); rank-based (Mann-Whitney)."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float('nan')
    all_s = np.concatenate([scores_pos, scores_neg])
    order = np.argsort(all_s, kind='mergesort')
    ranks = np.empty(len(all_s))
    # average ranks for ties
    sorted_s = all_s[order]
    ranks_sorted = np.arange(1, len(all_s) + 1, dtype=np.float64)
    i = 0
    while i < len(all_s):
        j = i
        while j + 1 < len(all_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks_sorted[i:j + 1] = ranks_sorted[i:j + 1].mean()
        i = j + 1
    ranks[order] = ranks_sorted
    r_pos = ranks[:len(scores_pos)].sum()
    n_p, n_n = len(scores_pos), len(scores_neg)
    return float((r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n))


def ece(conf, correct, n_bins=10):
    conf = np.asarray(conf); correct = np.asarray(correct, dtype=np.float64)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def eval_cell(ntag, seed, feats_gpu, clean, filtered_counts):
    nr = NOISES[ntag]
    pat = os.path.join(BASE, f'output/CIFAR100/resnet18/2026_7_2/cfgBAYESCOV_{ntag}_*_albayes_product_train_label_modeobserved_seed{seed}_*')
    run = sorted(glob.glob(pat))[0]
    ld = lambda p: np.load(p, allow_pickle=True).astype(int)

    lset_full = ld(os.path.join(run, 'episode_9/lSet.npy'))
    assert len(lset_full) == 1000
    records = {int(k): int(v) for k, v in json.load(open(os.path.join(run, 'episode_8/annotated_labels.json'))).items()}
    observed_full = np.array([records[g] for g in lset_full], dtype=np.int64)

    # --- validation (a): regenerate the annotator's noisy labels
    class _Stub: pass
    stub = _Stub(); stub.targets = clean.tolist()
    ann = Annotator(dataset=stub, noise_type='sym', noise_rate=nr, cost=1, seed=seed, num_classes=C)
    regen_ok = bool(np.array_equal(ann.noisy_labels[lset_full], observed_full))

    # --- in-ball matrix vs the full 1000 labeled points (budgets are prefixes)
    lf = feats_gpu[torch.as_tensor(lset_full, device=DEVICE)]
    inball = torch.zeros((feats_gpu.shape[0], 1000), dtype=torch.float32, device=DEVICE)
    bs = 8192
    for i in range(0, feats_gpu.shape[0], bs):
        d = torch.cdist(feats_gpu[i:i + bs].unsqueeze(0), lf.unsqueeze(0)).squeeze(0)
        inball[i:i + bs] = (d < DELTA).float()
    # --- per-labeled-point log-belief rows [1000, C], bit-identical to
    # BayesianCoverage._belief + _neighbor_loglik (k=1): b = pi*M[:,y] / sum,
    # logq = log(b.clamp_min(1e-12)), all in float32 torch ops.
    pi_t = torch.full((C,), 1.0 / C, device=DEVICE, dtype=torch.float32)
    M_t = torch.as_tensor(ann.confusion_matrix, device=DEVICE, dtype=torch.float32)
    logb_by_label = torch.empty((C, C), dtype=torch.float32, device=DEVICE)
    for y in range(C):
        b = pi_t * M_t[:, y]
        b = b / b.sum().clamp_min(LOG_EPS)
        logb_by_label[y] = torch.log(b.clamp_min(LOG_EPS))
    logb = logb_by_label[torch.as_tensor(observed_full, device=DEVICE)]     # [1000, C]

    clean_gpu = torch.as_tensor(clean, device=DEVICE)
    is_noisy_full = observed_full != clean[lset_full]
    out = {'ntag': ntag, 'noise': nr, 'seed': seed, 'regen_ok': regen_ok, 'budgets': {}}

    # Replicate the run's accumulator bit-for-bit: init C = log(pi), then fold
    # one labeled neighbor at a time in lSet order (float32 rounding residues on
    # no-neighbor rows decide LOO argmax ties, so summation order matters).
    N = feats_gpu.shape[0]
    logpi_row = torch.log(pi_t.clamp_min(LOG_EPS))
    Cmat = logpi_row.repeat(N, 1)
    next_fold = 0

    for B in BUDGETS:
        ep = B // 100 - 1
        uset = ld(os.path.join(run, f'episode_{ep}/uSet.npy'))
        while next_fold < B:
            Cmat += inball[:, next_fold:next_fold + 1] * logb[next_fold]
            next_fold += 1
        P = torch.softmax(Cmat, dim=1)
        lidx = torch.as_tensor(lset_full[:B], device=DEVICE)
        uidx = torch.as_tensor(uset, device=DEVICE)
        obs = torch.as_tensor(observed_full[:B], device=DEVICE)

        # ---- labeled plane
        Pl = P[lidx]
        amax_l = Pl.argmax(dim=1)
        cl = clean_gpu[lidx]
        observed_acc = float((obs == cl).float().mean())
        post_acc = float((amax_l == cl).float().mean())
        flip = amax_l != obs
        n_flip = int(flip.sum())
        flips_right = int(((amax_l == cl) & flip).sum())
        good_overturned = int(((obs == cl) & flip).sum())

        # LOO
        Cl_loo = Cmat[lidx] - logb[:B]
        Ploo = torch.softmax(Cl_loo, dim=1)
        amax_loo = Ploo.argmax(dim=1)
        keep = amax_loo == obs
        kept = int(keep.sum())
        w = Ploo[torch.arange(B, device=DEVICE), obs].cpu().numpy()
        noisy_mask = is_noisy_full[:B]
        flagged = (~keep).cpu().numpy()
        prec = float(noisy_mask[flagged].mean()) if flagged.sum() else float('nan')
        rec = float(flagged[noisy_mask].mean()) if noisy_mask.sum() else float('nan')
        auc = roc_auc(1.0 - w[noisy_mask], 1.0 - w[~noisy_mask]) if noisy_mask.sum() and (~noisy_mask).sum() else float('nan')

        # ---- unlabeled plane
        Pu = P[uidx]
        conf_u, amax_u = Pu.max(dim=1)
        covered = inball[uidx, :B].sum(dim=1) > 0
        cov_frac = float(covered.float().mean())
        cu = clean_gpu[uidx]
        correct_u = (amax_u == cu)
        cov_acc = float(correct_u[covered].float().mean()) if covered.sum() else float('nan')
        all_acc_u = float(correct_u.float().mean())
        uncov_acc = float(correct_u[~covered].float().mean()) if (~covered).sum() else float('nan')
        correct_l = (amax_l == cl)
        global_acc = float((correct_l.sum() + correct_u.sum()) / (B + len(uset)))
        # calibration on covered unlabeled points
        conf_cov = conf_u[covered].cpu().numpy()
        corr_cov = correct_u[covered].cpu().numpy()
        ece_cov = ece(conf_cov, corr_cov) if covered.sum() else float('nan')
        mean_conf = float(conf_cov.mean()) if covered.sum() else float('nan')
        # pseudo-label precision: top-k most confident per predicted class (covered only)
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

        log_kept = filtered_counts.get((ntag, seed), {}).get(B)
        out['budgets'][B] = dict(
            observed_acc=observed_acc, post_acc=post_acc,
            n_flip=n_flip, flips_right=flips_right, good_overturned=good_overturned,
            kept=kept, kept_logged=log_kept, kept_match=(log_kept == kept if log_kept is not None else None),
            noise_prec=prec, noise_rec=rec, noise_auc=auc,
            loo_disagree_frac=1.0 - kept / B,
            cov_frac=cov_frac, cov_acc=cov_acc, all_acc_u=all_acc_u, uncov_acc=uncov_acc,
            global_acc=global_acc, ece=ece_cov, mean_conf=mean_conf,
            topk=topk_acc,
        )
    return out


def main():
    clean = load_clean_labels()
    feats = load_features()
    feats_gpu = torch.as_tensor(feats, device=DEVICE)
    filtered_counts = parse_filtered_log_counts()
    print(f'filtered-log validation targets: {sorted(filtered_counts.keys())}')
    results = []
    for ntag in NOISES:
        for seed in SEEDS:
            r = eval_cell(ntag, seed, feats_gpu, clean, filtered_counts)
            results.append(r)
            mm = [v['kept_match'] for v in r['budgets'].values()]
            print(f"{ntag} seed{seed}: regen_ok={r['regen_ok']} kept_match={mm}")
    with open(os.path.join(OUT_DIR, 'posterior_eval.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print('written', os.path.join(OUT_DIR, 'posterior_eval.json'))


if __name__ == '__main__':
    main()
