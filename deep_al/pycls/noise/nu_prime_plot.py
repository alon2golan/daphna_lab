"""
nu_prime_plot.py
----------------
Plot nu_prime curves from the per-dataset cache.

Two plot types
--------------
  curves   – one subplot per al_method, lines = budgets, x=tau, y=nu_prime
             (what's in the cache, straight from the grid / adaptive sweep)

  iso_nu   – one subplot per al_method, lines = budgets, x=target_nu, y=tau
             (useful to read off: "for this noise level, what tau is needed?")

Usage
-----
  python -m deep_al.pycls.noise.nu_prime_plot --dataset cifar100 [--type curves|iso_nu|both]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless – saves to file
import matplotlib.pyplot as plt

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from deep_al.pycls.noise.nu_prime_cache import load_cache

# ── configuration ─────────────────────────────────────────────────────────────

HIGHLIGHT_BUDGETS = [100, 200, 500, 1000]
AL_METHODS        = ["probcover", "maxherding", "random"]

# nu grid used for iso_nu curves
ISO_NU_GRID = np.linspace(0.05, 0.95, 200)


# ── helpers ───────────────────────────────────────────────────────────────────

def _budget_color(budget: int, all_budgets: list[int]):
    cmap = plt.cm.viridis
    idx  = all_budgets.index(budget)
    return cmap(idx / max(len(all_budgets) - 1, 1))


def _with_oracle_anchor(taus: np.ndarray, nus: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Inject (tau=1.0, nu_prime=0.0) oracle anchor if not already present."""
    if 1.0 not in taus:
        taus = np.append(taus, 1.0)
        nus  = np.append(nus,  0.0)
        idx  = np.argsort(taus)
        taus, nus = taus[idx], nus[idx]
    return taus, nus


def _interp_tau_for_nu(taus: np.ndarray, nus: np.ndarray,
                        nu_query: float) -> Optional[float]:
    """Inverse interpolation: given nu_prime, return tau. nus is decreasing."""
    if len(taus) < 2:
        return None
    # flip so xp is increasing for np.interp
    return float(np.interp(nu_query, nus[::-1], taus[::-1],
                            left=np.nan, right=np.nan))


# ── plot: nu_prime vs tau ─────────────────────────────────────────────────────

def plot_curves(dataset: str, cache: dict, out_dir: Path) -> None:
    """
    One subplot per AL method.
    x-axis = tau, y-axis = nu_prime.
    Each budget is a separate line.
    """
    methods = [m for m in AL_METHODS if m in cache]
    if not methods:
        print("No data in cache – nothing to plot.")
        return

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4),
                              sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        budget_data = cache[method]
        all_budgets = sorted(int(b) for b in budget_data.keys()
                             if int(b) in HIGHLIGHT_BUDGETS)
        if not all_budgets:
            all_budgets = sorted(int(b) for b in budget_data.keys())

        for budget in all_budgets:
            b_key = str(budget)
            tau_dict = budget_data[b_key]
            taus = np.array(sorted(float(t) for t in tau_dict.keys()))
            nus  = np.array([tau_dict[f"{t:.4f}"] for t in taus])
            taus, nus = _with_oracle_anchor(taus, nus)

            color = _budget_color(budget, all_budgets)
            ax.plot(taus, nus, "o-", color=color, label=f"B={budget}", linewidth=1.8)

        ax.set_xlabel(r"$\tau$ (cost ratio)", fontsize=11)
        ax.set_ylabel(r"$\nu'$ (equilibrium noise)", fontsize=11)
        ax.set_title(method, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{dataset.upper()}  –  $\\nu'$ vs $\\tau$", fontsize=13)
    fig.tight_layout()

    out_path = out_dir / f"{dataset}_nu_prime_vs_tau.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── plot: tau vs target_nu ────────────────────────────────────────────────────

def plot_iso_nu(dataset: str, cache: dict, out_dir: Path) -> None:
    """
    One subplot per AL method.
    x-axis = target nu_prime (noise level), y-axis = tau required.
    Each budget is a separate line.
    Reads: "if my noisy annotator has noise=X, what cost ratio makes it worthwhile?"
    """
    methods = [m for m in AL_METHODS if m in cache]
    if not methods:
        print("No data in cache – nothing to plot.")
        return

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4),
                              sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        budget_data = cache[method]
        all_budgets = sorted(int(b) for b in budget_data.keys()
                             if int(b) in HIGHLIGHT_BUDGETS)
        if not all_budgets:
            all_budgets = sorted(int(b) for b in budget_data.keys())

        for budget in all_budgets:
            b_key = str(budget)
            tau_dict = budget_data[b_key]
            taus = np.array(sorted(float(t) for t in tau_dict.keys()))
            nus  = np.array([tau_dict[f"{t:.4f}"] for t in taus])
            taus, nus = _with_oracle_anchor(taus, nus)

            # Only interpolate within the observed nu range
            nu_min, nu_max = nus.min(), nus.max()
            nu_grid = ISO_NU_GRID[(ISO_NU_GRID >= nu_min) & (ISO_NU_GRID <= nu_max)]
            if len(nu_grid) < 2:
                continue

            tau_grid = np.array([_interp_tau_for_nu(taus, nus, nu)
                                  for nu in nu_grid])
            valid = ~np.isnan(tau_grid)

            color = _budget_color(budget, all_budgets)
            ax.plot(nu_grid[valid], tau_grid[valid], "o-", color=color,
                    label=f"B={budget}", linewidth=1.8, markersize=3)

        ax.set_xlabel(r"$\nu'$ (noise level of annotator)", fontsize=11)
        ax.set_ylabel(r"$\tau$ (cost ratio threshold)", fontsize=11)
        ax.set_title(method, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{dataset.upper()}  –  Required $\\tau$ to use noisy annotator", fontsize=13)
    fig.tight_layout()

    out_path = out_dir / f"{dataset}_tau_vs_nu.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── plot: nu_prime heatmap (budget × tau per method) ─────────────────────────

def plot_heatmap(dataset: str, cache: dict, out_dir: Path) -> None:
    """
    One heatmap per AL method: rows=budget, cols=tau, cell=nu_prime value.
    """
    methods = [m for m in AL_METHODS if m in cache]
    if not methods:
        return

    for method in methods:
        budget_data = cache[method]
        budgets = sorted(int(b) for b in budget_data.keys()
                         if int(b) in HIGHLIGHT_BUDGETS)
        if not budgets:
            budgets = sorted(int(b) for b in budget_data.keys())

        # Collect all taus across chosen budgets; always include oracle anchor
        all_taus = sorted({
            float(t)
            for b in budgets
            for t in budget_data[str(b)].keys()
        } | {1.0})

        mat = np.full((len(budgets), len(all_taus)), np.nan)
        for i, budget in enumerate(budgets):
            for j, tau in enumerate(all_taus):
                if tau == 1.0:
                    mat[i, j] = 0.0  # oracle anchor
                    continue
                t_key = f"{tau:.4f}"
                val = budget_data[str(budget)].get(t_key)
                if val is not None:
                    mat[i, j] = val

        fig, ax = plt.subplots(figsize=(max(6, len(all_taus) * 0.7),
                                        max(3, len(budgets) * 0.6) + 0.5))
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn_r",
                       origin="upper")
        plt.colorbar(im, ax=ax, label=r"$\nu'$")

        ax.set_xticks(range(len(all_taus)))
        ax.set_xticklabels([f"{t:.2f}" for t in all_taus], rotation=45, ha="right")
        ax.set_yticks(range(len(budgets)))
        ax.set_yticklabels(budgets)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel("Budget")
        ax.set_title(f"{dataset.upper()} {method}  –  $\\nu'$ heatmap")

        # annotate cells
        for i in range(len(budgets)):
            for j in range(len(all_taus)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                             fontsize=7, color="black")

        fig.tight_layout()
        out_path = out_dir / f"{dataset}_{method}_heatmap.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar100",
                        choices=["cifar10", "cifar100", "imagenet50"])
    parser.add_argument("--type", default="both",
                        choices=["curves", "iso_nu", "heatmap", "both", "all"])
    parser.add_argument("--out_dir", default=None,
                        help="Directory for output PNGs (default: results/<dataset>/)")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], required=True,
                        help="Which optimizer column of the ν' cache to plot "
                             "(sgd = Tomer-style, adam = daphna_lab default).")
    args = parser.parse_args()

    from deep_al.pycls.noise.nu_prime_cache import _method_cache
    full = load_cache(args.dataset)
    if not full:
        print(f"No cache found for dataset '{args.dataset}'. "
              f"Run nu_prime_sweeper.py first.")
        return
    # Slice to the requested optimizer so downstream plot helpers keep their
    # legacy {method: {budget: {tau}}} view.
    cache = {m: _method_cache(full, m, args.optimizer) for m in full}
    cache = {m: d for m, d in cache.items() if d}
    if not cache:
        print(f"No '{args.optimizer}' ν' data for dataset '{args.dataset}'.")
        return

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(_repo_root) / "results" / args.dataset.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_type = args.type
    if plot_type in ("curves", "both", "all"):
        plot_curves(args.dataset, cache, out_dir)
    if plot_type in ("iso_nu", "both", "all"):
        plot_iso_nu(args.dataset, cache, out_dir)
    if plot_type in ("heatmap", "all"):
        plot_heatmap(args.dataset, cache, out_dir)


if __name__ == "__main__":
    main()
