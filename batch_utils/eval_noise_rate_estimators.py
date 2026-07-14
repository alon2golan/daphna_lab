#!/usr/bin/env python3
"""
§9.4 noise-rate estimator bake-off (OFFLINE, no GPU-training, no production-code change).

The production BayesianCoverage pipeline is *handed* the true annotator accuracy
(the confusion matrix M is built from the known noise rate).  This script does NOT
touch that pipeline; it only measures how well several *self-contained* estimators
recover the realized labeled-set noise rate from the same persisted trajectories the
`observed` bayes_product runs produced.

Ground truth (per run, per budget B):
    realized noise = mean( observed_label[:B] != clean_label[:B] )

Estimators scored (none is told the noise rate, except the explicitly-labelled
oracle-M reference):

  raw_loo   Raw LOO ball-majority disagreement.  For each labeled point, take the
            majority OBSERVED label among its in-ball labeled neighbours (leave-one-out,
            self excluded); disagreement fraction = q̂.  Known to saturate at the ball
            impurity floor (§4.5 D) -> cannot separate noise 0 from 0.2.

  loo_corr  Impurity-corrected raw_loo.  Calibrate the clean-data floor f0(B) from the
            noise=0 cells, then invert the mixture  d = q*1 + (1-q)*f0  ->
            q̂ = clip((d - f0)/(1 - f0), 0, 1).  (f0 is a noise-independent property of
            the embedding's ball impurity; here we read it off the n00 cells.)

  ds_em     Dawid-Skene-style symmetric-noise EM.  Latent true class z_i per labeled
            point; product-of-experts belief over in-ball labeled neighbours (self
            INCLUDED, exactly the run's accumulator) but with a SELF-ESTIMATED symmetric
            confusion M(q); alternate  E: posterior γ_i(c)  /  M: q = mean_i(1-γ_i(y_i))
            to a fixed point.  The natural Bayesian answer; corrects impurity because
            impure balls yield mixed evidence.

  softloo*  REFERENCE ONLY (uses the TRUE M, i.e. is *told* the rate): mean_i (1 - w_i)
            with w_i = LOO posterior mass on the observed label.  Shows the impurity
            ceiling of the disagreement family when M is known.  Marked with '*' — not a
            real detector.

  netta_q̂   Parsed from the 27 NPC-repro SLURM logs (job 30811243): the AUM train-on-clean
            partition ('[train-on-clean] training on X/Y') and the NPC-round kept count
            ('NPC | round N: filter kept X/Y'); q̂ = 1 - X/Y.  Only exists for noise
            {0.2,0.5,0.8}. Different runs/seeds than the replay, aligned by (noise,budget).

Outputs -> results/cifar100/noise_rate_estimators/
    noise_rate_estimators.json   full per-cell numbers
    noise_rate_estimators.csv    tidy long table
    fig_noise_rate_estimators.png  q̂ vs realized, per method
"""
import numpy as np, torch, json, glob, os, pickle, re, sys
from collections import defaultdict

BASE = '/cs/labs/daphna/alongol/projects/daphna_lab'
OUT_DIR = os.path.join(BASE, 'results/cifar100/noise_rate_estimators')
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DELTA = 0.75
C = 100
LOG_EPS = 1e-12
BUDGETS = list(range(100, 1001, 100))
NOISES = {'n00': 0.0, 'n20': 0.2, 'n50': 0.5, 'n80': 0.8}
SEEDS = [1, 2, 3]
DS_ITERS = 50

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


# ----------------------------------------------------------------------------- estimators
def raw_loo_disagree(A_self, obs_oh, obs):
    """A_self: [B,B] float in-ball adjacency incl. diagonal. Returns (q_raw, frac_isolated).
    LOO majority vote of OBSERVED neighbour labels, self excluded."""
    A = A_self.clone()
    A.fill_diagonal_(0.0)                        # leave-one-out: drop own vote
    counts = A @ obs_oh                          # [B,C] neighbour label histogram
    n_nb = A.sum(dim=1)                          # neighbours per point
    has_nb = n_nb > 0
    maj = counts.argmax(dim=1)                   # majority neighbour label
    disagree = (maj != obs) & has_nb
    denom = int(has_nb.sum())
    q_raw = float(disagree.sum()) / denom if denom else float('nan')
    frac_iso = float((~has_nb).float().mean())
    return q_raw, frac_iso


def ds_em_symmetric(A_self, obs, obs_oh, n_iter=DS_ITERS, q0=0.3):
    """Symmetric-noise Dawid-Skene EM on the in-ball neighbour graph (self included).
    logγ_i(c) ∝ cnt_i(c)*log(1-q) + (n_i-cnt_i(c))*log(q/(C-1)) + logπ(c)
    With uniform π the class-independent term cancels in the softmax, so
    γ_i(c) ∝ exp(cnt_i(c)*logr),  logr = log(1-q) - log(q/(C-1)).
    M-step:  q = mean_i (1 - γ_i(y_i))."""
    counts = A_self @ obs_oh                      # [B,C] incl. self
    B = counts.shape[0]
    idx = torch.arange(B, device=counts.device)
    cnt_obs = counts[idx, obs]                    # cnt_i(y_i)
    q = q0
    for _ in range(n_iter):
        q = min(max(q, 1e-4), 1 - 1.0 / C - 1e-6)
        logr = float(np.log(1 - q) - np.log(q / (C - 1)))
        logits = counts * logr                    # [B,C]
        logZ = torch.logsumexp(logits, dim=1)     # [B]
        log_g_obs = cnt_obs * logr - logZ         # log γ_i(y_i)
        g_obs = torch.exp(log_g_obs)
        q_new = float((1.0 - g_obs).mean())
        if abs(q_new - q) < 1e-6:
            q = q_new
            break
        q = q_new
    return q


def eval_cell(ntag, seed, feats_gpu, clean):
    nr = NOISES[ntag]
    pat = os.path.join(BASE, f'output/CIFAR100/resnet18/2026_7_2/cfgBAYESCOV_{ntag}_*_albayes_product_train_label_modeobserved_seed{seed}_*')
    run = sorted(glob.glob(pat))[0]
    ld = lambda p: np.load(p, allow_pickle=True).astype(int)

    lset_full = ld(os.path.join(run, 'episode_9/lSet.npy'))
    assert len(lset_full) == 1000
    records = {int(k): int(v) for k, v in json.load(open(os.path.join(run, 'episode_8/annotated_labels.json'))).items()}
    observed_full = np.array([records[g] for g in lset_full], dtype=np.int64)

    # sanity: regenerated annotator labels == recorded annotations
    class _Stub: pass
    stub = _Stub(); stub.targets = clean.tolist()
    ann = Annotator(dataset=stub, noise_type='sym', noise_rate=nr, cost=1, seed=seed, num_classes=C)
    regen_ok = bool(np.array_equal(ann.noisy_labels[lset_full], observed_full))

    # labeled-vs-labeled ball adjacency (self included), full 1000x1000
    lf = feats_gpu[torch.as_tensor(lset_full, device=DEVICE)]
    D = torch.cdist(lf.unsqueeze(0), lf.unsqueeze(0)).squeeze(0)
    A_full = (D < DELTA).float()

    obs_full_t = torch.as_tensor(observed_full, device=DEVICE)
    # true-M LOO belief rows (for the oracle-M reference only) -----------------
    pi_t = torch.full((C,), 1.0 / C, device=DEVICE, dtype=torch.float32)
    M_t = torch.as_tensor(ann.confusion_matrix, device=DEVICE, dtype=torch.float32)
    logb_by_label = torch.empty((C, C), dtype=torch.float32, device=DEVICE)
    for y in range(C):
        b = pi_t * M_t[:, y]
        b = b / b.sum().clamp_min(LOG_EPS)
        logb_by_label[y] = torch.log(b.clamp_min(LOG_EPS))
    logb_full = logb_by_label[obs_full_t]         # [1000, C]
    logpi_row = torch.log(pi_t.clamp_min(LOG_EPS))

    is_noisy_full = observed_full != clean[lset_full]
    out = {'ntag': ntag, 'noise': nr, 'seed': seed, 'regen_ok': regen_ok, 'run': os.path.basename(run), 'budgets': {}}

    for B in BUDGETS:
        obs = obs_full_t[:B]
        obs_oh = torch.nn.functional.one_hot(obs, C).float()
        A_self = A_full[:B, :B]

        realized = float(is_noisy_full[:B].mean())
        q_raw, frac_iso = raw_loo_disagree(A_self, obs_oh, obs)
        q_ds = ds_em_symmetric(A_self, obs, obs_oh)

        # oracle-M soft LOO disagreement (reference; uses TRUE M) --------------
        # accumulator over first B labeled points, LOO posterior mass on observed
        Cmat = logpi_row.repeat(B, 1) + (A_self @ logb_full[:B])
        Cl_loo = Cmat - logb_full[:B]
        Ploo = torch.softmax(Cl_loo, dim=1)
        w = Ploo[torch.arange(B, device=DEVICE), obs]
        q_softloo_trueM = float((1.0 - w).mean())

        out['budgets'][B] = dict(
            realized=realized, frac_isolated=frac_iso,
            raw_loo=q_raw, ds_em=q_ds, softloo_trueM=q_softloo_trueM,
        )
    return out


# ----------------------------------------------------------------------------- Netta q̂
def parse_netta_qhat():
    """{('n20'|'n50'|'n80', seed): {budget: {'aum': q, 'npc_round': q}}} from job 30811243.
    Only the npc sampling arm runs the AUM filter (pc/random have no partition)."""
    out = defaultdict(lambda: defaultdict(dict))
    for path in glob.glob(os.path.join(BASE, 'logs/sweep/sweep_30811243_*.out')):
        txt = open(path, errors='replace').read()
        m = re.search(r'cfgNETTA_(n\d\d)_al(\w+?)_seed(\d)', txt)
        if not m:
            continue
        ntag, alg, seed = m.group(1), m.group(2), int(m.group(3))
        if alg != 'npc':
            continue
        key = (ntag, seed)
        for km in re.finditer(r'\[train-on-clean\] training on (\d+)/(\d+) labeled points \(filter=aum\)', txt):
            kept, tot = int(km.group(1)), int(km.group(2))
            out[key][tot]['aum'] = 1.0 - kept / tot
        for km in re.finditer(r'NPC \| round \d+: filter kept (\d+)/(\d+) labeled points clean', txt):
            kept, tot = int(km.group(1)), int(km.group(2))
            out[key][tot]['npc_round'] = 1.0 - kept / tot
    return out


# ----------------------------------------------------------------------------- scoring / io
def main():
    clean = load_clean_labels()
    feats_gpu = torch.as_tensor(load_features(), device=DEVICE)

    cells = []
    for ntag in NOISES:
        for seed in SEEDS:
            r = eval_cell(ntag, seed, feats_gpu, clean)
            cells.append(r)
            print(f"{ntag} seed{seed}: regen_ok={r['regen_ok']}  "
                  f"B100 realized={r['budgets'][100]['realized']:.3f} "
                  f"raw={r['budgets'][100]['raw_loo']:.3f} ds={r['budgets'][100]['ds_em']:.3f}")

    # impurity floor f0(B) = mean raw_loo over the n00 cells
    f0 = {}
    for B in BUDGETS:
        vals = [c['budgets'][B]['raw_loo'] for c in cells if c['ntag'] == 'n00']
        f0[B] = float(np.mean(vals))
    for c in cells:
        for B in BUDGETS:
            d = c['budgets'][B]['raw_loo']
            f = f0[B]
            c['budgets'][B]['loo_corr'] = float(np.clip((d - f) / (1 - f), 0.0, 1.0)) if f < 1 else float('nan')

    netta = parse_netta_qhat()
    netta_ser = {f'{k[0]}_seed{k[1]}': dict(v) for k, v in netta.items()}

    # ---- tidy CSV
    est_keys = ['raw_loo', 'loo_corr', 'ds_em', 'softloo_trueM']
    csv_lines = ['ntag,noise,seed,budget,realized,frac_isolated,' + ','.join(est_keys)]
    for c in cells:
        for B in BUDGETS:
            b = c['budgets'][B]
            row = [c['ntag'], f"{c['noise']:.2f}", str(c['seed']), str(B),
                   f"{b['realized']:.4f}", f"{b['frac_isolated']:.4f}"] + [f"{b[k]:.4f}" for k in est_keys]
            csv_lines.append(','.join(row))
    with open(os.path.join(OUT_DIR, 'noise_rate_estimators.csv'), 'w') as fcsv:
        fcsv.write('\n'.join(csv_lines) + '\n')

    with open(os.path.join(OUT_DIR, 'noise_rate_estimators.json'), 'w') as fj:
        json.dump({'delta': DELTA, 'f0_floor': f0, 'cells': cells, 'netta_qhat': netta_ser}, fj, indent=1)

    # ---- scoring tables (mean over seeds), per noise level
    print('\n' + '=' * 78)
    print('MEAN ESTIMATE vs REALIZED  (averaged over 3 seeds)')
    for B in (100, 300, 1000):
        print(f'\n--- budget {B} ---')
        print(f"{'noise':>6} {'realiz':>7} {'raw_loo':>8} {'loo_corr':>9} {'ds_em':>7} {'softloo*':>9} {'netta_aum':>10}")
        for ntag in NOISES:
            cs = [c for c in cells if c['ntag'] == ntag]
            realized = np.mean([c['budgets'][B]['realized'] for c in cs])
            def mean_est(k): return np.mean([c['budgets'][B][k] for c in cs])
            nvals = [netta[(ntag, s)][B]['aum'] for s in SEEDS if (ntag, s) in netta and B in netta[(ntag, s)] and 'aum' in netta[(ntag, s)][B]]
            nstr = f"{np.mean(nvals):>10.3f}" if nvals else f"{'--':>10}"
            print(f"{NOISES[ntag]:>6.2f} {realized:>7.3f} {mean_est('raw_loo'):>8.3f} "
                  f"{mean_est('loo_corr'):>9.3f} {mean_est('ds_em'):>7.3f} {mean_est('softloo_trueM'):>9.3f} {nstr}")

    # ---- headline error metrics across all noise cells (exclude softloo* reference honesty note)
    print('\n' + '=' * 78)
    print('MAE(q̂, realized) over the 4 noise levels x 3 seeds, per budget')
    print(f"{'budget':>6} {'raw_loo':>8} {'loo_corr':>9} {'ds_em':>7} {'softloo*':>9}")
    for B in BUDGETS:
        errs = {k: [] for k in est_keys}
        for c in cells:
            b = c['budgets'][B]
            for k in est_keys:
                errs[k].append(abs(b[k] - b['realized']))
        print(f"{B:>6} " + ' '.join(f"{np.mean(errs[k]):>8.3f}" for k in
              ['raw_loo', 'loo_corr', 'ds_em', 'softloo_trueM']))

    # separation test: can it tell noise 0 from 0.2? (mean over seeds, budget 300)
    print('\n' + '=' * 78)
    print("SEPARATION 0 vs 0.2 (mean over seeds; >0 means it distinguishes):")
    for B in (100, 300, 1000):
        row = []
        for k in est_keys:
            e0 = np.mean([c['budgets'][B][k] for c in cells if c['ntag'] == 'n00'])
            e2 = np.mean([c['budgets'][B][k] for c in cells if c['ntag'] == 'n20'])
            row.append(f"{k}={e2 - e0:+.3f}")
        print(f"  B{B}: " + '  '.join(row))

    make_figure(cells, netta)
    print('\nwritten', OUT_DIR)


def make_figure(cells, netta):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('(no matplotlib, skipping figure)', e)
        return
    budgets = BUDGETS
    methods = [('raw_loo', 'raw LOO'), ('loo_corr', 'impurity-corrected LOO'),
               ('ds_em', 'Dawid-Skene EM'), ('softloo_trueM', 'soft LOO (true M)*')]
    fig, axes = plt.subplots(1, len(NOISES), figsize=(4 * len(NOISES), 4), sharey=True)
    for ax, ntag in zip(axes, NOISES):
        cs = [c for c in cells if c['ntag'] == ntag]
        realized = [np.mean([c['budgets'][B]['realized'] for c in cs]) for B in budgets]
        ax.plot(budgets, realized, 'k--', lw=2, label='realized')
        for k, lbl in methods:
            y = [np.mean([c['budgets'][B][k] for c in cs]) for B in budgets]
            ax.plot(budgets, y, marker='o', ms=3, label=lbl)
        nvals = []
        for B in budgets:
            v = [netta[(ntag, s)][B]['aum'] for s in SEEDS if (ntag, s) in netta and B in netta[(ntag, s)] and 'aum' in netta[(ntag, s)][B]]
            nvals.append(np.mean(v) if v else np.nan)
        if np.any(~np.isnan(nvals)):
            ax.plot(budgets, nvals, marker='s', ms=4, color='tab:brown', label='Netta q̂ (aum)')
        ax.set_title(f'noise {NOISES[ntag]:.1f}')
        ax.set_xlabel('budget')
        ax.grid(alpha=.3)
    axes[0].set_ylabel('estimated / realized noise rate')
    axes[0].legend(fontsize=7, loc='upper right')
    fig.suptitle('§9.4 noise-rate estimator bake-off (offline replay, δ=0.75)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_noise_rate_estimators.png'), dpi=130)
    print('figure ->', os.path.join(OUT_DIR, 'fig_noise_rate_estimators.png'))


if __name__ == '__main__':
    main()
