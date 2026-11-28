"""Consolidated results, matched references, and method walkthroughs."""

import pandas as pd
import streamlit as st

from streamlit_app import CONDITIONS, HERE, RESULTS, show_npc, show_vast


NOISE = {**CONDITIONS, "cifar100n": "CIFAR-100N"}
AUDIT_ARMS = {
    "loo_audit": "Leave-one-out", "aum_rank": "Released LowBudgetAUM rank",
    "fresh_clean": "Buy new clean labels", "single_footprint": "Unsigned footprint",
    "diverse_footprint": "Greedy footprint", "random_audit": "Random audit",
    "no_audit": "No verified requests", "best_equal_cost_mean": "Best equal-cost arm per setting",
    "signed_peer": "Signed audit",
}


def grid_table(grid, change):
    ordered = grid["condition"].drop_duplicates().map(NOISE)
    matrix = grid.assign(Noise=grid["condition"].map(NOISE)).pivot(
        index="Noise", columns="budget", values=change,
    ).reindex(ordered)
    st.dataframe(matrix.style.format("{:+.2f}", na_rep=""), width="stretch")


def comparison_caption(grid, change):
    values = grid[change]
    wins = int((values > 1e-8).sum())
    ties = int((values.abs() <= 1e-8).sum())
    st.caption(f"{wins}/{len(values)} wins · {ties} ties · {values.mean():+.2f} pp on average")


def show_peer_only():
    root = HERE / "peer_only" / "results"
    with st.expander("Does the AUM term help?"):
        st.markdown("Peer-only replaces the rank sum with the agreement score. Both fits keep NPC’s class quotas.")
        if not (root / "by_setting.csv").exists():
            st.info("The matched peer-only fits are pending.")
            return
        grid = pd.read_csv(root / "by_setting.csv")
        st.caption("Peer-only gains +0.51 pp under symmetric noise and loses 1.00 pp under cyclic noise. "
                   "Adding AUM recovers +0.93 pp of the cyclic loss.")
        refs = {
            "Fusion vs peer-only": "fusion_minus_peer_only_pp",
            "Peer-only vs released NPC": "peer_only_minus_base_pp",
        }
        selected = st.selectbox("Rank comparison", tuple(refs))
        comparison_caption(grid, refs[selected])
        grid_table(grid, refs[selected])
        st.download_button("Download peer-only and fused fits", (root / "per_seed.csv").read_bytes(),
                           "peer_only_per_seed.csv")


def show_rank_confirmation():
    root = RESULTS / "rank_confirm"
    with st.expander("Fresh-seed AUM-rank confirmation", expanded=True):
        if not (root / "by_setting.csv").exists():
            st.info("Official NPC reruns for seeds 74–76 are pending.")
            return
        grid = pd.read_csv(root / "by_setting.csv")
        pairs = pd.read_csv(root / "per_seed.csv")
        st.caption("Seeds 74–76 · same released NPC fits and AUM-retained labels")
        comparisons = {
            "Raw six-hop graph": "delta_rank_pp",
            "Released NPC classifier": "delta_rank_official_pp",
        }
        choice = st.radio("Confirmation reference", tuple(comparisons), horizontal=True)
        delta_column = comparisons[choice]
        comparison_caption(grid, delta_column)
        grid_table(grid, delta_column)
        depths = [name for name in ("raw3_pct", "raw4_pct", "raw5_pct", "raw6_pct", "raw7_pct", "raw8_pct")
                  if name in grid.columns]
        if depths:
            best = grid[["raw_pct", *depths]].max(axis=1)
            delta = grid["rank_pct"] - best
            st.caption(f"Vs the best raw depth by setting after scoring three-seed means · "
                       f"{int((delta > 1e-8).sum())}/{len(delta)} wins · {delta.mean():+.2f} pp mean")
        options = grid["condition"].drop_duplicates().tolist()
        condition = st.selectbox("Confirmation noise", options, index=options.index("sym50"),
                                 format_func=NOISE.__getitem__)
        rows = pairs.loc[pairs["condition"] == condition, [
            "budget", "seed", "official_pct", "raw_pct", "rank_pct", delta_column,
        ]].rename(columns={
            "budget": "Labels", "seed": "Seed", "official_pct": "Released NPC (%)",
            "raw_pct": "Raw six-hop (%)", "rank_pct": "AUM-rank votes (%)", delta_column: "Δ pp",
        })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button("Download fresh graph pairs", (root / "per_seed.csv").read_bytes(),
                           "npc_rank_confirmation_per_seed.csv")


def show_graph():
    st.markdown("All predictors use the same official NPC acquisition order and final AUM-retained labels.")
    show_rank_confirmation()
    st.subheader("Graph comparisons on seeds 71–73")
    root = RESULTS / "npc_cap"
    weights_root = RESULTS / "npc_graph_weights"
    if not (root / "by_setting.csv").exists():
        st.info("The matched graph replay is pending.")
    else:
        grid = pd.read_csv(root / "by_setting.csv")
        seeds = pd.read_csv(root / "per_seed.csv")
        refs = {
            "Runner-up cap vs raw six-hop": ("delta_cap_pp", "cap_pct", "raw_pct"),
            "Raw six-hop vs released NPC": ("delta_raw_pp", "raw_pct", "official_pct"),
            "Runner-up cap vs released NPC": ("delta_vs_official_pp", "cap_pct", "official_pct"),
        }
        if (weights_root / "by_setting.csv").exists():
            extra = pd.read_csv(weights_root / "by_setting.csv")
            fields = ["sqrt_pct", "rank_pct", "delta_sqrt_pp", "delta_rank_pp"]
            grid = grid.merge(extra[["condition", "budget", *fields]], validate="one_to_one")
            extra_pairs = pd.read_csv(weights_root / "per_seed.csv")
            seeds = seeds.merge(extra_pairs[["condition", "budget", "seed", *fields]], validate="one_to_one")
            grid["delta_rank_official_pp"] = grid["rank_pct"] - grid["official_pct"]
            seeds["delta_rank_official_pp"] = seeds["rank_pct"] - seeds["official_pct"]
            refs.update({
                "Square-root votes vs raw six-hop": ("delta_sqrt_pp", "sqrt_pct", "raw_pct"),
                "AUM-rank votes vs raw six-hop": ("delta_rank_pp", "rank_pct", "raw_pct"),
                "AUM-rank votes vs released NPC": ("delta_rank_official_pp", "rank_pct", "official_pct"),
            })
        choice = st.selectbox("Graph comparison", tuple(refs))
        delta, method, reference = refs[choice]
        comparison_caption(grid, delta)
        grid_table(grid, delta)
        options = grid["condition"].drop_duplicates().tolist()
        condition = st.selectbox("Graph noise", options, index=options.index("sym50"), format_func=NOISE.__getitem__)
        names = {
            "official_pct": "Released NPC (%)", "raw_pct": "Raw six-hop (%)", "cap_pct": "Capped six-hop (%)",
            "sqrt_pct": "Square-root votes (%)", "rank_pct": "AUM-rank votes (%)",
        }
        paired = seeds.loc[seeds["condition"] == condition, ["budget", "seed", reference, method, delta]]
        st.dataframe(paired.rename(columns={"budget": "Labels", "seed": "Seed", delta: "Δ pp", **names}),
                     hide_index=True, width="stretch")
        st.download_button("Download all graph pairs", seeds.to_csv(index=False).encode(),
                           "npc_graph_per_seed.csv")
    with st.expander("Graph formulas and pseudocode"):
        st.markdown((HERE / "npc_cap" / "PROTOCOL.md").read_text())


def show_audits():
    st.markdown("Each audit and clean-expansion arm spends the same extra 10% budget on verified clean answers. "
                "The released LowBudgetAUM detector ranks requests in one control. Its acquisition and "
                "graph predictor are shared with signed auditing.")
    root = HERE / "signed_audit" / "results"
    if (root / "comparisons.csv").exists():
        data = pd.read_csv(root / "comparisons.csv")
        options = data["baseline"].drop_duplicates().tolist()
        reference = st.selectbox("Audit reference", options, index=options.index("best_equal_cost_mean"),
                                 format_func=AUDIT_ARMS.__getitem__)
        metric = st.radio("Audit predictor", ["graph", "cosine_1nn"], horizontal=True,
                          format_func={"graph": "Three-hop graph", "cosine_1nn": "Cosine 1NN"}.__getitem__)
        rows = data[(data["baseline"] == reference) & (data["metric"] == metric)]
        comparison_caption(rows, "delta_pp")
        grid_table(rows, "delta_pp")
        condition = st.selectbox("Audit noise", rows["condition"].drop_duplicates().tolist(),
                                 index=rows["condition"].drop_duplicates().tolist().index("sym50"),
                                 format_func=NOISE.__getitem__)
        details = rows[rows["condition"] == condition][[
            "budget", "reference_arm", "baseline_pct", "signed_pct", "delta_pp", "wins", "ties", "losses",
        ]].copy()
        details["reference_arm"] = details["reference_arm"].map(AUDIT_ARMS)
        st.dataframe(details.rename(columns={
            "budget": "Labels", "reference_arm": "Reference", "baseline_pct": "Reference (%)",
            "signed_pct": "Signed audit (%)", "delta_pp": "Δ pp", "wins": "Seed wins",
            "ties": "Seed ties", "losses": "Seed losses",
        }), hide_index=True, width="stretch")
        metrics = pd.read_csv(root / "metrics.csv")
        budget = st.selectbox("Audit seed detail at labels", details["budget"].tolist())
        arm = rows.loc[(rows["condition"] == condition) & (rows["budget"] == budget), "reference_arm"].iloc[0]
        subset = metrics[(metrics["condition"] == condition) & (metrics["budget"] == budget)
                         & metrics["policy"].isin((arm, "signed_peer"))]
        value = "test_correct" if metric == "graph" else "test_1nn_correct"
        pairs = subset.assign(Accuracy=100 * subset[value] / subset["test_count"]).pivot(
            index="seed", columns="policy", values="Accuracy",
        )
        pairs["Δ pp"] = pairs["signed_peer"] - pairs[arm]
        st.dataframe(pairs.rename(columns=AUDIT_ARMS).style.format("{:.2f}"), width="stretch")
        st.download_button("Download all audit scores", (root / "metrics.csv").read_bytes(),
                           "signed_audit_per_seed.csv")
    else:
        st.info("Fresh comparison with the AUM ranking control is pending.")
    st.caption("Archived audits won 36/36 synthetic settings against leave-one-out and equal-cost expansion.")
    with st.expander("Signed audit formulas and pseudocode"):
        st.markdown((HERE / "signed_audit" / "PROTOCOL.md").read_text())


def main():
    st.set_page_config(page_title="CIFAR-100 research review", layout="wide")
    st.title("CIFAR-100 research review")
    overview, npc, audits, graphs, vast, pilots = st.tabs([
        "What held up", "NPC · AUM filter", "Signed audits", "Graph votes", "Protected VAST", "Pilot inventory",
    ])
    with overview:
        st.markdown((HERE / "REVIEW.md").read_text())
    with npc:
        show_npc()
        show_peer_only()
    with audits:
        show_audits()
    with graphs:
        show_graph()
    with vast:
        show_vast()
    with pilots:
        st.markdown(
            "The original cap and square-root results below come from the lab’s 45,000-image NPC "
            "approximation. The official-run reruns in **Graph votes** supersede those conclusions."
        )
        st.markdown((HERE.parent / "graph_methods_study" / "METHODS.md").read_text())
        st.markdown("[Open the archived pilot dashboard](http://127.0.0.1:18527)")


if __name__ == "__main__":
    main()
