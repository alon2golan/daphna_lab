#!/usr/bin/env python
"""
Fig 5.5 in daphna_lab with TWO NoAA variants — NoAA-SGD and NoAA-Adam — using
the daphna-estimator nu_prime values (consolidated in nu_prime_optcompare_ALL.json,
from sweep_nu_prime_optcompare.py jobs 30741466 + 30741961).

Reuses the fig55 accuracy runs (same true-label runs as plot_fig55_daphna.py) for the
5 ν̂-independent strategies, and computes NoAA twice with the two optimizers' ν̂.

  results/fig55_daphna_noaa_optcompare.png / .csv
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/cs/labs/daphna/alongol/projects/daphna_lab")
OUT_ROOT = REPO / "output/CIFAR100/resnet18"
VHAT_JSON = REPO / "results/cifar100/nu_prime_optcompare_ALL.json"   # {opt:{budget:{tau: nu}}}
OUT_DIR = REPO / "results"
BUDGETS = [100, 300, 1000, 3000]
TAUS    = [0.1, 0.25, 0.5]
NOISES  = np.round(np.arange(0.1, 1.0, 0.1), 2)


def load_runs():
    oracle, noisy, sweeps = {}, {}, set()
    for f in glob.glob(str(OUT_ROOT / "**/result.json"), recursive=True):
        try: r = json.load(open(f))
        except Exception: continue
        sn = r.get("sweep_name", "")
        if not sn.startswith("fig55_daphna"): continue
        sweeps.add(sn)
        if r.get("status") != "completed" or not r.get("episodes"): continue
        p = r.get("params", {}); acc = r["episodes"][0]["test_acc"]
        B, tau, nu = p.get("budget"), p.get("tau"), p.get("nu_prime")
        if tau is None: oracle.setdefault(B, []).append(acc)
        else: noisy.setdefault((B, round(tau, 4), round(nu, 4)), []).append(acc)
    latest = sorted(sweeps)[-1] if sweeps else None
    return oracle, noisy, latest


def main():
    oracle, noisy, latest = load_runs()
    # restrict to latest sweep
    o2, n2 = {}, {}
    for f in glob.glob(str(OUT_ROOT / "**/result.json"), recursive=True):
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("sweep_name") != latest or r.get("status") != "completed" or not r.get("episodes"): continue
        p = r.get("params", {}); acc = r["episodes"][0]["test_acc"]
        B, tau, nu = p.get("budget"), p.get("tau"), p.get("nu_prime")
        if tau is None: o2.setdefault(B, []).append(acc)
        else: n2.setdefault((B, round(tau, 4), round(nu, 4)), []).append(acc)
    oracle, noisy = o2, n2
    print(f"sweep: {latest} | oracle Bs {sorted(oracle)} | noisy keys {len(noisy)}")

    vhat = json.load(open(VHAT_JSON))  # {opt:{budget:{tau: nu}}}
    def vhat_of(opt, B, tau):
        try: return float(vhat[opt][str(B)][f"{tau:.4f}"])
        except KeyError: return np.nan

    def oacc(B):
        v = oracle.get(B, []); return float(np.mean(v)) if v else np.nan
    def npn(B, tau):
        return {n: (float(np.mean(noisy[(B, round(tau,4), round(n,4))]))
                    if (B, round(tau,4), round(n,4)) in noisy else np.nan) for n in NOISES}

    rows = []
    for tau in TAUS:
        for B in BUDGETS:
            om = oacc(B); pn = npn(B, tau)
            vals = [v for v in pn.values() if not np.isnan(v)]
            nm = float(np.mean(vals)) if vals else np.nan
            if vals and not np.isnan(om):
                random_m = (om + nm) / 2
                analytical_m = np.nanmean([om if tau > (1 - n) else pn[n] for n in NOISES])
                gt_m = float(np.nanmean([max(om, pn[n]) if not np.isnan(pn[n]) else np.nan for n in NOISES]))
                def noaa(vh): return float(np.nanmean([om if n > vh else pn[n] for n in NOISES])) if not np.isnan(vh) else np.nan
                noaa_sgd  = noaa(vhat_of("sgd",  B, tau))
                noaa_adam = noaa(vhat_of("adam", B, tau))
            else:
                random_m = analytical_m = gt_m = noaa_sgd = noaa_adam = np.nan
            for s, v in [("All-Oracle", om), ("All-Noisy", nm), ("Random", random_m),
                         ("Analytical (τ=1−ν)", analytical_m), ("Ground Truth", gt_m),
                         ("NoAA-SGD", noaa_sgd), ("NoAA-Adam", noaa_adam)]:
                rows.append({"strategy": s, "tau": tau, "budget": B, "acc_mean": v})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "fig55_daphna_noaa_optcompare.csv", index=False)

    colors = {"All-Oracle": "tab:blue", "All-Noisy": "tab:orange", "Random": "tab:green",
              "Analytical (τ=1−ν)": "tab:red", "Ground Truth": "tab:purple",
              "NoAA-SGD": "tab:brown", "NoAA-Adam": "magenta"}
    styles = {"NoAA-SGD": dict(marker="o", ls="-", lw=2), "NoAA-Adam": dict(marker="s", ls="--", lw=2)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, tau in zip(axes, TAUS):
        for s, c in colors.items():
            sub = df[(df.strategy == s) & np.isclose(df.tau, tau)].sort_values("budget")
            if sub.acc_mean.isna().all(): continue
            st = styles.get(s, dict(marker="o", ls="-"))
            ax.plot(sub.budget, sub.acc_mean, color=c, label=s, **st)
        ax.set_title(f"CIFAR100, τ={tau}"); ax.set_xlabel("Total cost (B)"); ax.set_xscale("log"); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean accuracy (%)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Fig 5.5 in daphna_lab — NoAA-SGD vs NoAA-Adam (daphna ν̂ estimator, clean lsets)", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT_DIR / "fig55_daphna_noaa_optcompare.png", dpi=120, bbox_inches="tight")
    print(f"✓ {OUT_DIR/'fig55_daphna_noaa_optcompare.png'}")

if __name__ == "__main__":
    main()
