#!/usr/bin/env python3
"""Aggregate posterior_eval.json -> markdown tables (mean±std over seeds)."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
results = json.load(open(os.path.join(HERE, '../results/cifar100/posterior_eval/posterior_eval.json')))
BUDGETS = [100, 300, 500, 700, 1000]
NOISES = [0.0, 0.2, 0.5, 0.8]


def cells(noise, budget):
    return [r['budgets'][str(budget)] for r in results if r['noise'] == noise]


def ms(vals, pct=True, prec=1):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return '—'
    m, s = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    if pct:
        return f"{100*m:.{prec}f}±{100*s:.{prec}f}"
    return f"{m:.3f}±{s:.3f}"


print("### A. Labeled points — posterior argmax vs observed label accuracy (%)\n")
print("| noise | metric | " + " | ".join(map(str, BUDGETS)) + " |")
print("|---|---|" + "---|" * len(BUDGETS))
for nz in NOISES:
    obs = [ms([c['observed_acc'] for c in cells(nz, b)]) for b in BUDGETS]
    post = [ms([c['post_acc'] for c in cells(nz, b)]) for b in BUDGETS]
    print(f"| {nz} | observed | " + " | ".join(obs) + " |")
    print(f"| {nz} | posterior argmax | " + " | ".join(post) + " |")

print("\n### A2. Flip breakdown at budget 1000 (counts /1000, mean over seeds)\n")
print("| noise | flips | right (fixed a bad label) | overturned a good label |")
print("|---|---|---|---|")
for nz in NOISES:
    cs = cells(nz, 1000)
    print(f"| {nz} | {np.mean([c['n_flip'] for c in cs]):.0f} | "
          f"{np.mean([c['flips_right'] for c in cs]):.0f} | "
          f"{np.mean([c['good_overturned'] for c in cs]):.0f} |")

print("\n### B. Unlabeled pool — coverage and pseudo-label quality (%)\n")
print("| noise | metric | " + " | ".join(map(str, BUDGETS)) + " |")
print("|---|---|" + "---|" * len(BUDGETS))
for nz in NOISES:
    cov = [ms([c['cov_frac'] for c in cells(nz, b)]) for b in BUDGETS]
    acc = [ms([c['cov_acc'] for c in cells(nz, b)]) for b in BUDGETS]
    aacc = [ms([c['all_acc_u'] for c in cells(nz, b)]) for b in BUDGETS]
    t10 = [ms([c['topk']['10']['acc'] for c in cells(nz, b)]) for b in BUDGETS]
    n10 = [f"{np.mean([c['topk']['10']['n'] for c in cells(nz, b)]):.0f}" for b in BUDGETS]
    print(f"| {nz} | covered (% of pool) | " + " | ".join(cov) + " |")
    print(f"| {nz} | acc on covered | " + " | ".join(acc) + " |")
    print(f"| {nz} | acc all pool | " + " | ".join(aacc) + " |")
    print(f"| {nz} | acc top-10/class | " + " | ".join(t10) + " |")
    print(f"| {nz} | n top-10/class | " + " | ".join(n10) + " |")

print("\n### C. Calibration on covered unlabeled points\n")
print("| noise | metric | " + " | ".join(map(str, BUDGETS)) + " |")
print("|---|---|" + "---|" * len(BUDGETS))
for nz in NOISES:
    ec = [ms([c['ece'] for c in cells(nz, b)], pct=False) for b in BUDGETS]
    mc = [ms([c['mean_conf'] for c in cells(nz, b)]) for b in BUDGETS]
    ac = [ms([c['cov_acc'] for c in cells(nz, b)]) for b in BUDGETS]
    print(f"| {nz} | ECE | " + " | ".join(ec) + " |")
    print(f"| {nz} | mean confidence (%) | " + " | ".join(mc) + " |")
    print(f"| {nz} | accuracy (%) | " + " | ".join(ac) + " |")

print("\n### D. LOO reliability as a noise detector (labeled points)\n")
print("| noise | metric | " + " | ".join(map(str, BUDGETS)) + " |")
print("|---|---|" + "---|" * len(BUDGETS))
for nz in NOISES:
    auc = [ms([c['noise_auc'] for c in cells(nz, b)], pct=False) for b in BUDGETS]
    pr = [ms([c['noise_prec'] for c in cells(nz, b)]) for b in BUDGETS]
    rc = [ms([c['noise_rec'] for c in cells(nz, b)]) for b in BUDGETS]
    dis = [ms([c['loo_disagree_frac'] for c in cells(nz, b)]) for b in BUDGETS]
    print(f"| {nz} | ROC-AUC | " + " | ".join(auc) + " |")
    print(f"| {nz} | flag precision (%) | " + " | ".join(pr) + " |")
    print(f"| {nz} | flag recall (%) | " + " | ".join(rc) + " |")
    print(f"| {nz} | LOO-disagree frac (%) | " + " | ".join(dis) + " |")
