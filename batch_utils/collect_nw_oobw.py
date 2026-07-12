"""
Collect the §7.9 out-of-ball NW sweeps and pair them against the normalized
pseudo_all baseline (per-seed, per-episode):

  - job 31050702  sweep BAYESCOV_nw_pseudo_cifar100_n0_20_50_80__20260710_164246
                  (OOB_WEIGHT = 1.0, 12 runs)
  - job 31050847  sweep BAYESCOV_nw_oobw_pseudo_cifar100_n0_20_50_80__20260710_165538
                  (OOB_WEIGHT ∈ {0.3, 0.1}, 24 runs)
  - baseline      pseudo_all rows of job 31048960 (normalized rerun)

Writes results/bayes_pseudo/nw_oobw_{episodes,deltas}.csv and
fig_nw_oobw_delta.png. See bayesian_coverage_NEXT_STEPS.md §7.9 for the verdict.

Usage:
    python batch_utils/collect_nw_oobw.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "output" / "CIFAR100" / "resnet18"
RES_DIR = REPO / "results" / "bayes_pseudo"
DAYS = ["2026_7_9", "2026_7_10"]
BASELINE_SWEEP_ID = "31048960"  # keep only tags pseudo=all
NW_SWEEP_IDS = {"31050702", "31050847"}


def collect() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        for rj in (OUT_ROOT / day).glob("*/result.json"):
            try:
                j = json.load(open(rj))
            except Exception:
                continue
            sid = str(j.get("sweep_id"))
            tags = j.get("tags", {})
            if sid == BASELINE_SWEEP_ID:
                if tags.get("pseudo") != "all":
                    continue
                arm = "baseline"
            elif sid in NW_SWEEP_IDS:
                arm = f"nw_w{tags.get('oob_w', '1.0')}"
            else:
                continue
            if j.get("status") != "completed":
                print("SKIP not completed:", rj.parent.name)
                continue
            for ep in j["episodes"]:
                rows.append(dict(arm=arm, noise=float(tags["noise"]),
                                 seed=j["params"]["seed"],
                                 labeled=ep["labeled_size"], acc=ep["test_acc"]))
    return pd.DataFrame(rows)


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.arm == "baseline"].set_index(["noise", "seed", "labeled"])["acc"]
    out = []
    for arm in sorted(a for a in df.arm.unique() if a != "baseline"):
        a = df[df.arm == arm].set_index(["noise", "seed", "labeled"])["acc"]
        d = (a - base).dropna().rename("delta").reset_index()
        d["arm"] = arm
        out.append(d)
    return pd.concat(out)


def plot(deltas: pd.DataFrame, path: Path) -> None:
    arms = [("nw_w1.0", "#08519c", "w = 1.0"),
            ("nw_w0.3", "#4292c6", "w = 0.3"),
            ("nw_w0.1", "#9ecae1", "w = 0.1")]
    ink, muted = "#1a1a2e", "#6b6b7b"
    noises = sorted(deltas.noise.unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(13, 3.4),
                             sharey=True, sharex=True)
    for ax, noise in zip(axes, noises):
        ax.axhline(0, color=muted, lw=1, ls=(0, (4, 3)), zorder=1)
        for arm, color, label in arms:
            g = (deltas[(deltas.arm == arm) & (deltas.noise == noise)]
                 .groupby("labeled")["delta"].agg(["mean", "std"]))
            ax.plot(g.index, g["mean"], color=color, lw=2, marker="o", ms=4,
                    label=label, zorder=3)
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=color, alpha=0.12, lw=0, zorder=2)
        ax.set_title(f"noise = {noise}", fontsize=11, color=ink)
        ax.set_xlabel("labeled-set size", fontsize=9, color=muted)
        ax.tick_params(colors=muted, labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(muted)
        ax.grid(axis="y", color="#e8e8ee", lw=0.7, zorder=0)
        ax.set_xticks([100, 400, 700, 1000])
    axes[0].set_ylabel("Δ test acc  (NW − baseline)", fontsize=10, color=ink)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right",
                   title="OOB_WEIGHT", title_fontsize=9)
    fig.suptitle("§7.9  out-of-ball NW pseudo-targets: paired Δ vs normalized "
                 "pseudo_all  (mean ± std over seeds 1–3, CIFAR100)",
                 fontsize=11.5, color=ink, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")


def main() -> None:
    df = collect()
    counts = df.groupby("arm").size()
    print("run-episodes per arm:\n", counts)
    assert (counts == 120).all(), "expected 12 runs x 10 episodes per arm"
    deltas = paired_deltas(df)
    for lab in [100, 300, 1000]:
        t = (deltas[deltas.labeled == lab]
             .pivot_table(index="noise", columns="arm", values="delta",
                          aggfunc="mean").round(2))
        print(f"\npaired delta vs baseline @ {lab}:\n{t}")
    RES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES_DIR / "nw_oobw_episodes.csv", index=False)
    deltas.to_csv(RES_DIR / "nw_oobw_deltas.csv", index=False)
    plot(deltas, RES_DIR / "fig_nw_oobw_delta.png")
    print("\nwrote", RES_DIR / "nw_oobw_{episodes,deltas}.csv",
          "and fig_nw_oobw_delta.png")


if __name__ == "__main__":
    main()
