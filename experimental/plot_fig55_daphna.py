#!/usr/bin/env python
"""
Fig 5.5 reproduction plot, native to daphna_lab (post seed-decoupling fix).

Reads the per-run result.json files produced by sweep_fig55.py, reconstructs the
six strategy curves with the SAME math as the oracle-teacher-proj plotter, pulls
the NoAA v-hat threshold from the existing oracle-teacher-proj table, and writes
a 3-panel figure.

Usage:
    python batch_utils/plot_fig55_daphna.py [--sweep fig55_daphna_YYYYMMDD_HHMMSS]
If --sweep is omitted, the most recent fig55_daphna_* sweep is used.

Outputs:
    results/tomer_embedding_probcover_original.png
    results/tomer_embedding_probcover_original.csv
"""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO    = Path("/cs/labs/daphna/alongol/projects/daphna_lab")
OUT_ROOT = REPO / "output/CIFAR100/resnet18"
V_HAT   = Path("/cs/labs/daphna/alongol/projects/oracle-teacher-proj/results/noaa_v_hat_wide.csv")
OUT_DIR = REPO / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = [100, 300, 1000, 3000]
TAUS    = [0.1, 0.25, 0.5]
NOISES  = np.round(np.arange(0.1, 1.0, 0.1), 2)


def load_runs(sweep: str | None):
    """Return (oracle, noisy) dicts: oracle[B]=[acc...], noisy[(B,tau,nu)]=[acc...]."""
    oracle, noisy = {}, {}
    sweeps_seen = set()
    for f in glob.glob(str(OUT_ROOT / "**/result.json"), recursive=True):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        sn = r.get("sweep_name", "")
        if not sn.startswith("fig55_daphna"):
            continue
        sweeps_seen.add(sn)
        if sweep and sn != sweep:
            continue
        if r.get("status") != "completed" or not r.get("episodes"):
            continue
        p = r.get("params", {})
        acc = r["episodes"][0]["test_acc"]
        B = p.get("budget")
        tau = p.get("tau")
        nu = p.get("nu_prime")
        if tau is None:                      # oracle run
            oracle.setdefault(B, []).append(acc)
        else:                                # noisy run
            noisy.setdefault((B, round(tau, 4), round(nu, 4)), []).append(acc)
    return oracle, noisy, sweeps_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=None, help="sweep_name to plot (default: latest fig55_daphna_*)")
    args = ap.parse_args()

    oracle, noisy, sweeps_seen = load_runs(args.sweep)
    if args.sweep is None and sweeps_seen:
        latest = sorted(sweeps_seen)[-1]
        print(f"sweeps found: {sorted(sweeps_seen)}\nusing latest: {latest}")
        oracle, noisy, _ = load_runs(latest)
    print(f"oracle Bs: {sorted(oracle)} | noisy keys: {len(noisy)}")

    v_hat_df = pd.read_csv(V_HAT) if V_HAT.exists() else None
    if v_hat_df is None:
        print(f"NOTE: {V_HAT} missing — NoAA omitted")

    def oracle_acc(B):
        v = oracle.get(B, [])
        return (float(np.mean(v)) if v else np.nan)

    def noisy_per_nu(B, tau):
        return {n: (float(np.mean(noisy[(B, round(tau, 4), round(n, 4))]))
                    if (B, round(tau, 4), round(n, 4)) in noisy else np.nan)
                for n in NOISES}

    def lookup_v_hat(B, tau):
        if v_hat_df is None:
            return np.nan
        r = v_hat_df[(v_hat_df["budget"] == B) & np.isclose(v_hat_df["tau"], tau)]
        return float(r["mean"].iloc[0]) if not r.empty else np.nan

    # ---- compute curves (identical math to oracle-teacher-proj/Tools/plot_fig55_final.py) ----
    records = []
    for tau in TAUS:
        for B in BUDGETS:
            oracle_m = oracle_acc(B)
            nz_pn = noisy_per_nu(B, tau)
            nz_vals = [v for v in nz_pn.values() if not np.isnan(v)]
            noisy_m = float(np.mean(nz_vals)) if nz_vals else np.nan
            if nz_vals and not np.isnan(oracle_m):
                random_m     = (oracle_m + noisy_m) / 2
                analytical_m = np.nanmean([oracle_m if tau > (1 - n) else nz_pn[n] for n in NOISES])
                gt_m         = float(np.nanmean([max(oracle_m, nz_pn[n]) if not np.isnan(nz_pn[n]) else np.nan for n in NOISES]))
            else:
                random_m = analytical_m = gt_m = np.nan
            v_hat = lookup_v_hat(B, tau)
            if not np.isnan(v_hat) and nz_vals and not np.isnan(oracle_m):
                noaa_m = float(np.nanmean([oracle_m if n > v_hat else nz_pn[n] for n in NOISES]))
            else:
                noaa_m = np.nan
            for strat, val in [
                ("All-Oracle", oracle_m), ("All-Noisy", noisy_m), ("Random", random_m),
                ("Analytical (τ=1−ν)", analytical_m), ("Ground Truth", gt_m), ("NoAA", noaa_m),
            ]:
                records.append({"strategy": strat, "tau": tau, "budget": B, "acc_mean": val,
                                "v_hat": v_hat if strat == "NoAA" else np.nan})

    result_df = pd.DataFrame(records)
    result_df.to_csv(OUT_DIR / "tomer_embedding_probcover_original.csv", index=False)
    print(f"✓ {OUT_DIR / 'tomer_embedding_probcover_original.csv'}")

    # ---- plot ----
    colors = {"All-Oracle": "tab:blue", "All-Noisy": "tab:orange", "Random": "tab:green",
              "Analytical (τ=1−ν)": "tab:red", "Ground Truth": "tab:purple", "NoAA": "tab:brown"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, tau in zip(axes, TAUS):
        for strat, color in colors.items():
            sub = result_df[(result_df["strategy"] == strat) & np.isclose(result_df["tau"], tau)].sort_values("budget")
            vals = sub["acc_mean"].values
            if all(np.isnan(v) for v in vals):
                continue
            ax.plot(sub["budget"], vals, "o-", label=strat, color=color)
        ax.set_title(f"CIFAR100, τ={tau}")
        ax.set_xlabel("Total cost (B)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean accuracy (%)")
    axes[-1].legend(loc="lower right", fontsize=8)
    suffix = " — all six strategies" if v_hat_df is not None else " — five strategies (NoAA pending)"
    fig.suptitle(f"Fig 5.5 reproduction in daphna_lab (CIFAR100 · linear-on-SimCLR · ProbCover, post seed-fix){suffix}", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tomer_embedding_probcover_original.png", dpi=120, bbox_inches="tight")
    print(f"✓ {OUT_DIR / 'tomer_embedding_probcover_original.png'}")


if __name__ == "__main__":
    main()
