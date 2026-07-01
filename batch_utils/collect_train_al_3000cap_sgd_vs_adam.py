#!/usr/bin/env python
"""
Collect + compare the SGD-nu' vs Adam-nu' 3000-oracle-cost-cap sweeps.

Two arms, identical grid (strategy {probcover,random} x tau {0.1,0.25,0.5} x
seed {2,3,4} x (batch, nu'-cache budget) combos), differing ONLY in which
optimizer estimated nu':
  - ADAM arm: original sweep 30813471, output/.../2026_6_11/<name>_*
  - SGD  arm: sweep 30925479 (+ rerun 30943660), output/.../{2026_6_25,2026_6_28}/<name>_sgd_*

The SGD arm has 2 stale `status:running` dirs (the original timeouts) sitting
next to their real reruns; we dedupe per-config and keep the completed dir
(falling back to max-episodes), so the stale ones drop out.

Writes (under results/train_al_3000cap_sgd_vs_adam/):
  runs_long.csv          one row per (opt, run, episode)
  summary.csv            mean/std test_acc over seeds per (opt,strat,batch,nucache,tau,episode)
  final_compare.csv      final-episode test_acc: adam vs sgd vs delta, per config
  fig_sgd_vs_adam_main.png   batch=100 nucache=100, faceted by tau, sgd vs adam x {pc,random}

Usage:  venv/bin/python batch_utils/collect_train_al_3000cap_sgd_vs_adam.py
"""
from __future__ import annotations
import json, re, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/cs/labs/daphna/alongol/projects/daphna_lab")
OUT = REPO / "results" / "train_al_3000cap_sgd_vs_adam"

ADAM_GLOBS = [str(REPO / "output/CIFAR100/resnet18/2026_6_11/*_B*_nucache*_tau*_seed*")]
SGD_GLOBS = [
    str(REPO / "output/CIFAR100/resnet18/2026_6_25/*_B*_nucache*_tau*_seed*_sgd_*"),
    str(REPO / "output/CIFAR100/resnet18/2026_6_28/*_B*_nucache*_tau*_seed*_sgd_*"),
]

TAUS = [0.1, 0.25, 0.5]
STRATS = ["probcover", "random"]
NUCACHES = [100, 300, 1000]
DIRNAME_RE = re.compile(
    r"(?P<strat>probcover|random)_B(?P<batch>\d+)_nucache(?P<nucache>\d+)_tau(?P<tau>[\dp]+)_seed(?P<seed>\d+)_"
)
COLORS = {"probcover": "#1f77b4", "random": "#d62728"}


def _scan(globs, opt):
    """Return {config_key: best_result_dict} keeping the completed (else max-episode) dir."""
    best = {}
    for g in globs:
        for d in sorted(glob.glob(g)):
            name = Path(d).name
            m = DIRNAME_RE.search(name)
            if not m:
                continue
            rj = Path(d) / "result.json"
            if not rj.exists():
                continue
            data = json.load(open(rj))
            key = (m["strat"], int(m["batch"]), int(m["nucache"]),
                   m["tau"], int(m["seed"]))
            n_ep = len(data.get("episodes", []))
            completed = data.get("status") == "completed"
            cur = best.get(key)
            score = (1 if completed else 0, n_ep)
            if cur is None or score > cur[0]:
                best[key] = (score, data, m, name)
    rows = []
    for key, (score, data, m, name) in best.items():
        if score[1] == 0:
            print(f"  [drop] no episodes for {opt} {name}")
            continue
        strat = m["strat"]; batch = int(m["batch"]); nucache = int(m["nucache"])
        tau = float(m["tau"].replace("p", ".")); seed = int(m["seed"])
        nu_prime = data["params"].get("nu_prime")
        for ep in data["episodes"]:
            e = ep["episode"]
            rows.append({
                "opt": opt, "strat": strat, "batch": batch, "nucache": nucache,
                "tau": tau, "seed": seed, "nu_prime": nu_prime, "episode": e,
                "test_acc": ep["test_acc"], "val_acc": ep.get("val_acc"),
                "labeled_size": ep["labeled_size"],
                "oracle_cost": (e + 2) * batch,
            })
    return rows


def collect() -> pd.DataFrame:
    rows = _scan(ADAM_GLOBS, "adam") + _scan(SGD_GLOBS, "sgd")
    df = pd.DataFrame(rows)
    for opt in ("adam", "sgd"):
        n = df[df.opt == opt].groupby(["strat", "batch", "nucache", "tau", "seed"]).ngroups
        print(f"  {opt}: {n} runs")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["opt", "strat", "batch", "nucache", "tau", "episode"], as_index=False)
    s = g.agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        nu_prime=("nu_prime", "first"),
        oracle_cost=("oracle_cost", "first"),
        labeled_size=("labeled_size", "first"),
        n_seeds=("seed", "nunique"),
    )
    s["test_acc_std"] = s["test_acc_std"].fillna(0.0)
    return s


def final_compare(s: pd.DataFrame) -> pd.DataFrame:
    last = s.sort_values("oracle_cost").groupby(
        ["opt", "strat", "batch", "nucache", "tau"], as_index=False).last()
    keys = ["strat", "batch", "nucache", "tau"]
    adam = last[last.opt == "adam"].set_index(keys)
    sgd = last[last.opt == "sgd"].set_index(keys)
    out = []
    for k in sorted(set(adam.index) | set(sgd.index)):
        a = adam.loc[k] if k in adam.index else None
        g = sgd.loc[k] if k in sgd.index else None
        row = {"strat": k[0], "batch": k[1], "nucache": k[2], "tau": k[3],
               "adam_acc": None if a is None else round(float(a.test_acc_mean), 2),
               "adam_std": None if a is None else round(float(a.test_acc_std), 2),
               "adam_nu": None if a is None else round(float(a.nu_prime), 3),
               "sgd_acc": None if g is None else round(float(g.test_acc_mean), 2),
               "sgd_std": None if g is None else round(float(g.test_acc_std), 2),
               "sgd_nu": None if g is None else round(float(g.nu_prime), 3)}
        if a is not None and g is not None:
            row["delta_sgd_minus_adam"] = round(row["sgd_acc"] - row["adam_acc"], 2)
        else:
            row["delta_sgd_minus_adam"] = None
        out.append(row)
    return pd.DataFrame(out)


def _plot_curve(ax, sub, color, label, ls):
    sub = sub.sort_values("oracle_cost")
    x = sub["oracle_cost"].to_numpy(); y = sub["test_acc_mean"].to_numpy()
    e = sub["test_acc_std"].to_numpy()
    ax.plot(x, y, ls, color=color, label=label, lw=2, marker="o", ms=3)
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.12)


def fig_main(s: pd.DataFrame):
    """batch=100 nucache=100, sgd (solid) vs adam (dashed), pc/random by color, facet tau."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, tau in zip(axes, TAUS):
        for strat in STRATS:
            for opt, ls in (("sgd", "-"), ("adam", "--")):
                sub = s[(s.opt == opt) & (s.strat == strat) &
                        (s.batch == 100) & (s.nucache == 100) & (s.tau == tau)]
                if len(sub):
                    _plot_curve(ax, sub, COLORS[strat], f"{strat} {opt}", ls)
        ax.set_title(f"τ = {tau}")
        ax.set_xlabel("oracle cost (#queries)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle("SGD-ν' vs Adam-ν' under noisy annotator — batch=100, ν'@B=100 "
                 "(solid=SGD, dashed=Adam; mean±std over seeds 2,3,4)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "fig_sgd_vs_adam_main.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "runs_long.csv", index=False)
    print(f"wrote {OUT/'runs_long.csv'}  ({len(df)} episode-rows)")
    s = summarize(df)
    s.to_csv(OUT / "summary.csv", index=False)
    print(f"wrote {OUT/'summary.csv'}")
    fc = final_compare(s)
    fc.to_csv(OUT / "final_compare.csv", index=False)
    print(f"wrote {OUT/'final_compare.csv'}")
    fig_main(s)
    print("\nfinal-episode test-acc  (SGD-ν' vs Adam-ν'), Δ = sgd-adam:")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(fc.to_string(index=False))


if __name__ == "__main__":
    main()
