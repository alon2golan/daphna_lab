#!/usr/bin/env python
"""
Collect + plot the train_al 3000-oracle-cost-cap sweep (job 30813471,
sweep_train_al_3000cap.py, submitted 2026-06-11).

Grid: strategy {probcover, random} x tau {0.1, 0.25, 0.5} x seed {2,3,4} x
      (batch, nu'-cache budget): (100,{100,300,1000}), (300,{300}), (1000,{1000}).
All runs capped at ~3000 oracle-cost examples (max_iter 29/9/2 for batch 100/300/1000).

Sources every run's result.json under
  output/CIFAR100/resnet18/2026_6_11/<strat>_B<batch>_nucache<nu_b>_tau<tau>_seed<seed>_*/
parses params + per-episode test_acc / labeled_size, derives:
  oracle_cost  = (episode + 2) * batch      (verified vs cumulative_cost.txt: ep0=2*batch ... ep_last)
  labeled_size = (episode + 1) * batch/tau  (taken straight from result.json)

Writes:
  results/train_al_3000cap/runs_long.csv      one row per (run, episode)
  results/train_al_3000cap/summary.csv        mean/std test_acc over seeds per (strat,batch,nucache,tau,episode)
  results/train_al_3000cap/fig_main_pc_vs_random.png        probcover vs random, batch=100 nucache=100, 3 tau facets
  results/train_al_3000cap/fig_nucache_sensitivity.png      batch=100, nu'-cache budget {100,300,1000}, 3 tau facets
  results/train_al_3000cap/fig_batchsize_effect.png         batch {100,300,1000} (matched nucache), 3 tau facets

Usage:  venv/bin/python batch_utils/collect_train_al_3000cap.py
"""
from __future__ import annotations
import json, re, glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/cs/labs/daphna/alongol/projects/daphna_lab")
RUN_GLOB = str(REPO / "output/CIFAR100/resnet18/2026_6_11/*_B*_nucache*_tau*_seed*")
OUT = REPO / "results" / "train_al_3000cap"
SWEEP_ID = "30813471"

TAUS = [0.1, 0.25, 0.5]
STRATS = ["probcover", "random"]
NUCACHES = [100, 300, 1000]
DIRNAME_RE = re.compile(
    r"(?P<strat>probcover|random)_B(?P<batch>\d+)_nucache(?P<nucache>\d+)_tau(?P<tau>[\dp]+)_seed(?P<seed>\d+)_"
)

COLORS = {"probcover": "#1f77b4", "random": "#d62728"}
NU_COLORS = {100: "#1b9e77", 300: "#d95f02", 1000: "#7570b3"}


def collect() -> pd.DataFrame:
    rows = []
    dirs = sorted(glob.glob(RUN_GLOB))
    for d in dirs:
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
        nucache = int(m["nucache"])
        tau = float(m["tau"].replace("p", "."))
        seed = int(m["seed"])
        nu_prime = data["params"].get("nu_prime")
        for ep in data["episodes"]:
            e = ep["episode"]
            rows.append({
                "strat": strat, "batch": batch, "nucache": nucache,
                "tau": tau, "seed": seed, "nu_prime": nu_prime,
                "episode": e,
                "test_acc": ep["test_acc"], "val_acc": ep.get("val_acc"),
                "labeled_size": ep["labeled_size"],
                "oracle_cost": (e + 2) * batch,
            })
    df = pd.DataFrame(rows)
    print(f"collected {df['seed'].count()} episode-rows from "
          f"{df.groupby(['strat','batch','nucache','tau','seed']).ngroups} runs")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["strat", "batch", "nucache", "tau", "episode"], as_index=False)
    s = g.agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        oracle_cost=("oracle_cost", "first"),
        labeled_size=("labeled_size", "first"),
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


def fig_main(s: pd.DataFrame):
    """probcover vs random, batch=100 nucache=100 (matched), faceted by tau."""
    fig, axes = _facet_axes()
    for ax, tau in zip(axes, TAUS):
        for strat in STRATS:
            sub = s[(s.strat == strat) & (s.batch == 100) & (s.nucache == 100) & (s.tau == tau)]
            if len(sub):
                _plot_curve(ax, sub, COLORS[strat], strat)
        ax.set_title(f"τ = {tau}")
        ax.set_xlabel("oracle cost (#queries)")
        ax.grid(alpha=0.3)
        ax.legend(title="strategy", fontsize=9)
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle("ProbCover vs Random under noisy annotator — batch=100, ν' @ B=100 (mean±std over seeds 2,3,4)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "fig_main_pc_vs_random.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}")


def fig_nucache(s: pd.DataFrame):
    """batch=100: does the budget at which ν' was estimated matter? facet by tau, panels per strategy via linestyle."""
    fig, axes = _facet_axes()
    for ax, tau in zip(axes, TAUS):
        for strat in STRATS:
            ls = "-" if strat == "probcover" else "--"
            for nb in NUCACHES:
                sub = s[(s.strat == strat) & (s.batch == 100) & (s.nucache == nb) & (s.tau == tau)]
                if len(sub):
                    _plot_curve(ax, sub, NU_COLORS[nb], f"{strat} ν'@{nb}", ls=ls)
        ax.set_title(f"τ = {tau}")
        ax.set_xlabel("oracle cost (#queries)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle("ν'-cache-budget sensitivity — batch=100, ν' estimated at B∈{100,300,1000} "
                 "(solid=probcover, dashed=random)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "fig_nucache_sensitivity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}")


def fig_batchsize(s: pd.DataFrame):
    """batch {100,300,1000} at matched nucache (==batch), probcover only solid / random dashed, facet by tau."""
    fig, axes = _facet_axes()
    bcolors = {100: "#1f77b4", 300: "#ff7f0e", 1000: "#2ca02c"}
    for ax, tau in zip(axes, TAUS):
        for strat in STRATS:
            ls = "-" if strat == "probcover" else "--"
            for batch in (100, 300, 1000):
                sub = s[(s.strat == strat) & (s.batch == batch) & (s.nucache == batch) & (s.tau == tau)]
                if len(sub):
                    _plot_curve(ax, sub, bcolors[batch], f"{strat} B={batch}", ls=ls)
        ax.set_title(f"τ = {tau}")
        ax.set_xlabel("oracle cost (#queries)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle("AL batch-size effect at matched ν' (nucache==batch) — solid=probcover, dashed=random",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "fig_batchsize_effect.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}")


def print_final_table(s: pd.DataFrame):
    """Final-episode test_acc per config, for a quick text summary."""
    last = s.sort_values("oracle_cost").groupby(
        ["strat", "batch", "nucache", "tau"], as_index=False).last()
    print("\nfinal-episode test accuracy (mean±std over seeds), by config:")
    for batch in (100, 300, 1000):
        for nb in NUCACHES:
            sub = last[(last.batch == batch) & (last.nucache == nb)]
            if not len(sub):
                continue
            print(f"  batch={batch} ν'@{nb}:")
            for tau in TAUS:
                pc = sub[(sub.strat == "probcover") & (sub.tau == tau)]
                rd = sub[(sub.strat == "random") & (sub.tau == tau)]
                pc_s = f"{pc.test_acc_mean.iloc[0]:.2f}±{pc.test_acc_std.iloc[0]:.2f}" if len(pc) else "  -  "
                rd_s = f"{rd.test_acc_mean.iloc[0]:.2f}±{rd.test_acc_std.iloc[0]:.2f}" if len(rd) else "  -  "
                print(f"    τ={tau}: probcover {pc_s}   random {rd_s}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "runs_long.csv", index=False)
    print(f"wrote {OUT/'runs_long.csv'}")
    s = summarize(df)
    s.to_csv(OUT / "summary.csv", index=False)
    print(f"wrote {OUT/'summary.csv'}")
    fig_main(s)
    fig_nucache(s)
    fig_batchsize(s)
    print_final_table(s)


if __name__ == "__main__":
    main()
