"""Inspect paired trained-model comparisons from the matched research runs."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st


HERE = Path(__file__).parent
RESULTS = Path(os.environ.get("MATCHED_RESULTS_DIR", HERE / "results"))
CONDITIONS = {
    "clean": "Clean", "sym0": "Clean",
    "sym30": "Symmetric 30%", "sym40": "Symmetric 40%", "sym50": "Symmetric 50%",
    "asym30": "Asymmetric 30%", "asym40": "Asymmetric 40%", "asym50": "Asymmetric 50%",
}

st.set_page_config(page_title="CIFAR-100 paired comparisons", layout="wide")
st.title("CIFAR-100 paired comparisons")
npc, vast = st.tabs(["NPC · final AUM filter", "Protected VAST · pseudo-label ranks"])

with npc:
    st.caption("Reference: released Netta NPC + LowBudgetAUM + supervised linear classifier on BYOL.")
    st.markdown(
        "Peer-ranked AUM changes the final training mask. Each observed class keeps "
        "its original number of training images."
    )
    grid_file = RESULTS / "npc" / "by_setting.csv"
    seeds_file = RESULTS / "npc" / "per_seed.csv"
    if not grid_file.exists() or not seeds_file.exists():
        st.info("Waiting for verified NPC pairs.")
    else:
        grid = pd.read_csv(grid_file)
        seeds = pd.read_csv(seeds_file)
        complete = grid[grid["seeds"] == 3]
        counts, coverage, gain = st.columns(3)
        counts.metric("Verified seed pairs", f"{len(seeds)} / 105")
        coverage.metric(
            "Wins in complete settings",
            f"{int((complete['delta_pp'] > 1e-8).sum())} / {len(complete)}",
        )
        gain.metric("Mean gain across complete settings",
                    f"{complete['delta_pp'].mean():+.2f} pp" if len(complete) else "Pending")
        family_results = []
        for prefix, label in (("clean", "Clean"), ("sym", "Symmetric"), ("asym", "Cyclic asymmetric")):
            deltas = complete.loc[complete["condition"].str.startswith(prefix), "delta_pp"]
            if not deltas.empty:
                wins = int((deltas > 1e-8).sum())
                family_results.append(f"{label} {deltas.mean():+.2f} pp · {wins}/{len(deltas)} wins")
        if family_results:
            st.caption(" | ".join(family_results))
        grid["Noise"] = grid["condition"].map(CONDITIONS)
        st.subheader("Accuracy change by budget (pp)")
        table = grid.pivot(index="Noise", columns="budget", values="delta_pp")
        table = table.reindex(grid["Noise"].unique())
        st.dataframe(table.style.format("{:+.2f}", na_rep=""), width="stretch")
        options = grid["condition"].drop_duplicates().tolist()
        chosen = st.selectbox("Noise setting", options, index=options.index("sym50") if "sym50" in options else 0,
                              format_func=CONDITIONS.__getitem__, key="npc_noise")
        rows = grid[grid["condition"] == chosen].rename(columns={
            "base_accuracy_pct": "Netta NPC", "peer_accuracy_pct": "Peer-ranked AUM",
        })
        st.line_chart(rows.set_index("budget")[["Netta NPC", "Peer-ranked AUM"]],
                      x_label="Labels", y_label="Test accuracy (%)")
        paired = seeds.loc[seeds["condition"] == chosen, [
            "budget", "seed", "base_accuracy_pct", "peer_accuracy_pct", "delta_pp",
            "retained", "replaced", "base_clean_retained", "peer_clean_retained",
        ]].rename(columns={
            "budget": "Labels", "seed": "Seed", "base_accuracy_pct": "NPC (%)",
            "peer_accuracy_pct": "Peer-ranked AUM (%)", "delta_pp": "Δ pp",
            "retained": "Kept", "replaced": "Swaps", "base_clean_retained": "NPC clean kept",
            "peer_clean_retained": "Peer clean kept",
        })
        st.dataframe(paired, hide_index=True, width="stretch")
        st.download_button("Download paired NPC results", seeds_file.read_bytes(), "npc_per_seed.csv")
    with st.expander("NPC formulas and pseudocode"):
        st.markdown((HERE / "netta" / "PROTOCOL.md").read_text())

with vast:
    st.caption("Reference: the archived protected VAST student and its original pseudo-label targets.")
    st.markdown(
        "Cross-fitted rank alignment chooses a permutation on acquired labels, then fits the "
        "same student with the resulting targets."
    )
    results_file = RESULTS / "vast" / "per_seed.csv"
    if not results_file.exists():
        st.info("Waiting for verified VAST pairs.")
    else:
        results = pd.read_csv(results_file)
        results["condition"] = results["condition"].replace(CONDITIONS)
        displayed = results[[
            "condition", "budget", "ranks_aligned", "reference_pct", "candidate_pct",
            "delta_pp", "prior_winner_swap_pct",
        ]].rename(columns={
            "condition": "Noise", "budget": "Labels", "ranks_aligned": "Ranks aligned",
            "reference_pct": "Protected VAST (%)", "candidate_pct": "Rank alignment (%)",
            "delta_pp": "Δ pp", "prior_winner_swap_pct": "Prior winner swap (%)",
        })
        st.caption("Eight paired settings from archived seed 0. Zero aligned ranks reuses the reference target.")
        st.dataframe(displayed, hide_index=True, width="stretch")
        st.download_button("Download paired VAST results", results_file.read_bytes(), "vast_per_seed.csv")
    with st.expander("VAST formulas and protocol"):
        st.markdown((HERE / "vast_rank" / "PROTOCOL.md").read_text())
