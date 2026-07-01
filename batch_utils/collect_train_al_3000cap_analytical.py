#!/usr/bin/env python
"""
Collect + plot the ANALYTICAL-RULE arm of the train_al 3000-oracle-cost-cap sweep
(job 30911718, sweep_train_al_3000cap_analytical.py, submitted 2026-06-23).

Analytical rule (Tomer's thesis / Fig 5.5 "τ = 1 - ν" boundary): nu' = 1 - tau
    tau=0.1 -> nu'=0.9 ,  tau=0.25 -> nu'=0.75 ,  tau=0.5 -> nu'=0.5
This differs from the original sweep, which set nu' from the NoAA-fit nu_prime_cache.json.

Grid: strategy {probcover, random} x tau {0.1, 0.25, 0.5} x seed {2,3,4}, batch=100
      only, max_iter=29 -> 3000 oracle-cost cap.

Sources every run's result.json under
  output/CIFAR100/resnet18/2026_6_23/<strat>_B100_analytical_tau<tau>_seed<seed>_*/
parses params + per-episode test_acc / labeled_size, derives:
  oracle_cost  = (episode + 2) * batch   (SAME convention as collect_train_al_3000cap.py,
                                          so analytical curves overlay on the cache-ν' curves)
  labeled_size = ep["labeled_size"]      (taken straight from result.json)

Writes (results/train_al_3000cap_analytical/):
  runs_long.csv                     one row per (run, episode)
  summary.csv                       mean/std test_acc over seeds per (strat,tau,episode)
  fig_analytical_pc_vs_random.png   probcover vs random, 3 tau facets
  fig_analytical_vs_cache.png       analytical (ν'=1-τ) vs cache-ν'@100, pc & random, 3 tau facets

Usage:  venv/bin/python batch_utils/collect_train_al_3000cap_analytical.py
"""
from __future__ import annotations
import json, re, glob
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/cs/labs/daphna/alongol/projects/daphna_lab")
RUN_GLOB = str(REPO / "output/CIFAR100/resnet18/2026_6_23/*_B*_analytical_tau*_seed*")
OUT = REPO / "results" / "train_al_3000cap_analytical"
CACHE_SUMMARY = REPO / "results" / "train_al_3000cap" / "summary.csv"  # original cache-ν' arm
SWEEP_ID = "30911718"

TAUS = [0.1, 0.25, 0.5]
STRATS = ["probcover", "random"]
DIRNAME_RE = re.compile(
    r"(?P<strat>probcover|random)_B(?P<batch>\d+)_analytical_tau(?P<tau>[\dp]+)_seed(?P<seed>\d+)_"
)

COLORS = {"probcover": "#1f77b4", "random": "#d62728"}


def collect() -> pd.DataFrame:
    rows = []
    for d in sorted(glob.glob(RUN_GLOB)):
        name = Path(d).name
        m = DIRNAME_RE.search(name)
        if not m:
            continue
        rj = Path(d) / "result.json"
        if not rj.exists():
            print(f"  [skip] no result.json: {name}")
            continue
        data = json.load(open(rj))
        if data.get("status") != "completed":
            print(f"  [warn] status={data.get('status')}: {name}")
        strat = m["strat"]
        batch = int(m["batch"])
        tau = float(m["tau"].replace("p", "."))
        seed = int(m["seed"])
        nu_prime = data["params"].get("nu_prime")
        for ep in data["episodes"]:
            e = ep["episode"]
            rows.append({
                "strat": strat, "batch": batch, "tau": tau, "seed": seed,
                "nu_prime": nu_prime, "episode": e,
                "test_acc": ep["test_acc"], "val_acc": ep.get("val_acc"),
                "labeled_size": ep["labeled_size"],
                "oracle_cost": (e + 2) * batch,
            })
    df = pd.DataFrame(rows)
    print(f"collected {len(df)} episode-rows from "
          f"{df.groupby(['strat','tau','seed']).ngroups} runs "
          f"(nu' per tau: {sorted(df.groupby('tau').nu_prime.first().items())})")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["strat", "tau", "episode"], as_index=False)
    s = g.agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        oracle_cost=("oracle_cost", "first"),
        labeled_size=("labeled_size", "first"),
        nu_prime=("nu_prime", "first"),
        n_seeds=("seed", "nunique"),
    )
    s["test_acc_std"] = s["test_acc_std"].fillna(0.0)
    return s


def _facet_axes():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    return fig, axes


def _plot_curve(ax, sub, color, label, ls="-"):
    sub = sub.sort_values("oracle_cost")
    x = sub["oracle_cost"].to_numpy()
    y = sub["test_acc_mean"].to_numpy()
    e = sub["test_acc_std"].to_numpy()
    ax.plot(x, y, ls, color=color, label=label, lw=2, marker="o", ms=3)
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.15)


def fig_pc_vs_random(s: pd.DataFrame):
    fig, axes = _facet_axes()
    for ax, tau in zip(axes, TAUS):
        nu = s[s.tau == tau].nu_prime.iloc[0]
        for strat in STRATS:
            sub = s[(s.strat == strat) & (s.tau == tau)]
            if len(sub):
                _plot_curve(ax, sub, COLORS[strat], strat)
        ax.set_title(f"τ = {tau}   (ν' = {nu:g})")
        ax.set_xlabel("oracle cost (#queries)")
        ax.grid(alpha=0.3)
        ax.legend(title="strategy", fontsize=9)
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle("Analytical rule (ν' = 1−τ) — ProbCover vs Random, batch=100 "
                 "(mean±std over seeds 2,3,4)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "fig_analytical_pc_vs_random.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}")


def fig_vs_cache(s: pd.DataFrame):
    """Overlay analytical (ν'=1-τ, solid) vs the original cache-ν'@100 arm (dashed)."""
    if not CACHE_SUMMARY.exists():
        print(f"  [skip] no cache summary at {CACHE_SUMMARY}; run collect_train_al_3000cap.py first")
        return
    c = pd.read_csv(CACHE_SUMMARY)
    c = c[(c.batch == 100) & (c.nucache == 100)]
    fig, axes = _facet_axes()
    for ax, tau in zip(axes, TAUS):
        nu_a = s[s.tau == tau].nu_prime.iloc[0]
        for strat in STRATS:
            sa = s[(s.strat == strat) & (s.tau == tau)]
            if len(sa):
                _plot_curve(ax, sa, COLORS[strat], f"{strat} analytical (ν'={nu_a:g})", ls="-")
            sc = c[(c.strat == strat) & (c.tau == tau)]
            if len(sc):
                nu_c = sc.sort_values("oracle_cost").nu_prime.iloc[0] if "nu_prime" in sc else None
                lbl = f"{strat} cache-ν'" + (f" ({nu_c:g})" if nu_c is not None else "")
                _plot_curve(ax, sc, COLORS[strat], lbl, ls="--")
        ax.set_title(f"τ = {tau}")
        ax.set_xlabel("oracle cost (#queries)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle("Analytical rule (ν'=1−τ, solid) vs NoAA cache-ν'@100 (dashed) — batch=100",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "fig_analytical_vs_cache.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}")


def print_final_table(s: pd.DataFrame):
    last = s.sort_values("oracle_cost").groupby(["strat", "tau"], as_index=False).last()
    print("\nfinal-episode test accuracy (mean±std over seeds), analytical rule (ν'=1−τ):")
    for tau in TAUS:
        pc = last[(last.strat == "probcover") & (last.tau == tau)]
        rd = last[(last.strat == "random") & (last.tau == tau)]
        pc_s = f"{pc.test_acc_mean.iloc[0]:.2f}±{pc.test_acc_std.iloc[0]:.2f}" if len(pc) else "  -  "
        rd_s = f"{rd.test_acc_mean.iloc[0]:.2f}±{rd.test_acc_std.iloc[0]:.2f}" if len(rd) else "  -  "
        nu = last[last.tau == tau].nu_prime.iloc[0]
        print(f"  τ={tau} (ν'={nu:g}): probcover {pc_s}   random {rd_s}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "runs_long.csv", index=False)
    print(f"wrote {OUT/'runs_long.csv'}")
    s = summarize(df)
    s.to_csv(OUT / "summary.csv", index=False)
    print(f"wrote {OUT/'summary.csv'}")
    fig_pc_vs_random(s)
    fig_vs_cache(s)
    print_final_table(s)


if __name__ == "__main__":
    main()
