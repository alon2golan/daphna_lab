#!/usr/bin/env python3
"""
Reproduce Netta's thesis Figure 5 (Framework 2: linear model on SSL features,
CIFAR100, symmetric noise) from the NETTA_NPC sweep results.

Netta's plot conventions (from "netta thesis aum.pdf", Sec 4.2 / Fig 5):
  * y-axis = mean ACCURACY DIFFERENCE FROM RANDOM (method_acc - random_acc),
    with Standard Error (STE) over repetitions.
  * x-axis = annotation budget in E[SPC] = expected CLEAN samples per class
             = labeled_size * (1 - %noise) / C   (C = 100 for CIFAR100).
  * curves: random (blue, = 0 baseline), ProbCover (green),
            NPC w/ LowBudgetAUM (orange).
  * Framework-2 / CIFAR100 noise filter = LowBudgetAUM  == our `aum` filter.

The differences are computed PAIRED BY SEED (diff per seed, then mean & STE
across seeds), matching the "difference from random" semantics.

NOTE: our sweep budget tops out at 1000 labels (E[SPC] <= ~8), so we only
reproduce the LOW-BUDGET LEFT REGION of Netta's x-axis (she goes to ~70).
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reporting.scanner import load_all_results  # noqa: E402

SWEEP = "NETTA_NPC_cifar100_n20_50_80__20260611_121539"
C = 100  # CIFAR100 classes
NOISES = [20, 50, 80]
# Netta's colour scheme
STYLE = {
    "random":    dict(color="#1f77b4", label="random"),
    "probcover": dict(color="#2ca02c", label="ProbCover"),
    "npc":       dict(color="#ff7f0e", label="NPC w/ LowBudgetAUM (ours)"),
}
ORDER = ["random", "probcover", "npc"]


def _noise(cfg: str):
    m = re.search(r"NETTA_n(\d+)", cfg or "")
    return int(m.group(1)) if m else None


def load_episodes() -> pd.DataFrame:
    df = load_all_results()
    sub = df[df.sweep_name == SWEEP].copy()
    sub["noise"] = sub.p_cfg_file.apply(_noise)
    rows = []
    for _, r in sub.iterrows():
        for ep in r.episodes:
            rows.append(dict(noise=r.noise, method=r.p_al_method,
                             seed=r.p_seed, L=ep["labeled_size"],
                             acc=ep["test_acc"]))
    return pd.DataFrame(rows)


def diff_from_random(e: pd.DataFrame, noise: int):
    """Return dict method -> (espc, mean_diff, ste_diff) over seeds."""
    g = e[e.noise == noise]
    # wide: index (seed, L), columns method -> acc
    piv = g.pivot_table(index=["seed", "L"], columns="method", values="acc")
    out = {}
    for method in ("probcover", "npc"):
        d = (piv[method] - piv["random"]).reset_index()
        d.columns = ["seed", "L", "diff"]
        agg = d.groupby("L")["diff"].agg(["mean", "std", "count"]).reset_index()
        agg["ste"] = agg["std"].fillna(0.0) / np.sqrt(agg["count"])
        agg["espc"] = agg["L"] * (1 - noise / 100) / C
        out[method] = agg
    # random baseline (zero by construction); keep its x for the flat line
    Ls = sorted(g.L.unique())
    out["random"] = pd.DataFrame({
        "L": Ls,
        "espc": [L * (1 - noise / 100) / C for L in Ls],
        "mean": [0.0] * len(Ls), "ste": [0.0] * len(Ls),
    })
    return out


def main():
    e = load_episodes()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, noise in zip(axes, NOISES):
        data = diff_from_random(e, noise)
        for method in ORDER:
            d = data[method]
            st = STYLE[method]
            ax.plot(d["espc"], d["mean"], marker="o", ms=5,
                    color=st["color"], label=st["label"], lw=2)
            if method != "random":
                ax.fill_between(d["espc"], d["mean"] - d["ste"],
                                d["mean"] + d["ste"], color=st["color"],
                                alpha=0.18, lw=0)
        ax.axhline(0, color="#1f77b4", lw=1, ls="-", zorder=1)
        ax.set_title(f"CIFAR100, {noise}% Noise")
        ax.set_xlabel("Budget (E[SPC])")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy Difference (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(
        "Repro of Netta Fig.5 (Framework 2, CIFAR100, LowBudgetAUM) — "
        "acc difference from random vs E[SPC]  (3 seeds, ±STE; "
        "low-budget left region only, our max E[SPC]≈8)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = REPO / "results" / "netta_repro_fig5.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved:", out)

    # Print a small numeric table for the comparison write-up.
    print("\n=== peak advantage over random (max mean diff, low-budget) ===")
    for noise in NOISES:
        data = diff_from_random(e, noise)
        for m in ("npc", "probcover"):
            d = data[m]
            i = d["mean"].idxmax()
            print(f"  n{noise:>2} {m:<10} peak {d['mean'][i]:5.1f}% "
                  f"@E[SPC]={d['espc'][i]:.2f} (L={int(d['L'][i])}); "
                  f"final {d['mean'].iloc[-1]:5.1f}%")


if __name__ == "__main__":
    main()
