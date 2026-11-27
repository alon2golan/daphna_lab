"""Explore paired CIFAR-100 results for backup coverage and square-root votes."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyze import CHANNELS, CONDITIONS, collision_scores, compare, confirmation_scores, sparse_scores, summarize, vast_scores
from sota_scores import ACQUISITIONS, acquisition_scores, grouped_scores, match

STUDIES = {
    "methods": "Research scorecard",
    "sota": "Strong-baseline comparisons",
    "exact": "Backup coverage · exact MaxHerding",
    "sparse": "Backup coverage · sparse affinities",
    "grouped": "Backup coverage · distinct groups",
    "votes": "Square-root votes · fresh acquisitions",
    "fixed": "Square-root votes · archived MaxHerding",
    "channels": "Square-root votes · fixed corrupted images",
    "vast": "Square-root targets · protected VAST",
}
NOISE_LABELS = {
    "sym20": "Symmetric 20%", "asym20": "Asymmetric 20%", **CONDITIONS, **CHANNELS,
    "iid50": "IID 50%", "cyclic50": "Class-cycle 50%", "cluster50": "Clustered 50%",
    "iid_rate_matched": "IID · matched flip count",
}
HEADS = {
    "raw6": "6-hop graph", "raw3": "3-hop graph", "raw4": "4-hop graph", "raw8": "8-hop graph",
    "sqrt3": "3-hop square root", "sqrt6": "6-hop square root", "1nn": "Cosine 1NN",
}
STAGES = {"student": "Student · square-root targets", "winner_student": "Student · winner swap",
          "unlabeled": "Teacher · full unlabeled pool",
          "admitted": "Teacher · admitted pseudo-labels"}


def strong_baselines():
    scores = acquisition_scores()
    available = ["Graph", "Shared 300-epoch MLP"] if scores.training_filter.eq("mlp").any() else ["Graph"]
    protocol = st.selectbox("Protocol", available)
    scores = scores.loc[scores.training_filter.eq("mlp") == (protocol != "Graph")].copy()
    hard = scores.training_filter.eq("aum")
    scores.loc[hard, "head"] = "aum_" + scores.loc[hard, "head"]
    cohort = st.selectbox("Noise draws", ["Fresh NPC seeds", "Archived seed 0"]
                          if scores.seed.ne(0).any() else ["Archived seed 0"])
    scores = scores.loc[scores.seed.eq(0) if cohort == "Archived seed 0" else scores.seed.ne(0)]
    methods = [key for key in ACQUISITIONS if key in set(scores.method)]
    reference_labels = {**ACQUISITIONS, "envelope": "Best baseline in each cell"}
    head_labels = {
        **HEADS, "cap6": "6-hop · cap at second source", "cap3": "3-hop · cap at second source",
        **{f"aum_{head}": f"{label} · AUM filter" for head, label in HEADS.items()},
        "aum_cap6": "6-hop · cap with AUM filter", "aum_cap3": "3-hop · cap with AUM filter",
        "conditional6": "6-hop · compress flagged votes", "conditional3": "3-hop · compress flagged votes",
        "half6": "6-hop · half-weight flagged votes", "half3": "3-hop · half-weight flagged votes",
        "mlp": "300-epoch MLP",
    }
    columns = st.columns(2)
    candidate = columns[0].selectbox("Candidate acquisition", methods,
                                    index=methods.index("npc") if cohort == "Fresh NPC seeds" else 0,
                                    format_func=ACQUISITIONS.get)
    reference = columns[1].selectbox("Reference acquisition", [*methods, "envelope"],
                                     index=methods.index("npc") if "npc" in methods else 1,
                                     format_func=reference_labels.get)
    candidate_heads = [h for h in head_labels if h in set(scores.loc[scores.method.eq(candidate), "head"])]
    preferred = "cap6" if cohort == "Fresh NPC seeds" else candidate_heads[0]
    head = columns[0].selectbox("Candidate classifier", candidate_heads,
                                index=candidate_heads.index(preferred) if preferred in candidate_heads else 0,
                                format_func=head_labels.get)
    if reference == "envelope":
        policies = ("maxherding", "uherding", "js", "observed_10", "soft_10",
                    "probcover", "npc", "weighted_npc", "maxherding_nas")
        raw_heads = ("raw3", "raw6", "aum_raw3", "aum_raw6") if protocol == "Graph" else ("mlp",)
        comparisons = scores.loc[scores.method.isin(policies) & scores["head"].isin(raw_heads)]
        greatest = comparisons.groupby(["seed", "condition", "budget"], as_index=False).correct.max()
        scores = pd.concat([scores, greatest.assign(method="envelope", head="best")], ignore_index=True)
        baseline_head = "best"
        reference_label = reference_labels[reference]
        st.caption("The strongest reference is selected separately after scoring each noise, budget and seed.")
    else:
        reference_heads = [h for h in head_labels if h in set(scores.loc[scores.method.eq(reference), "head"])]
        default_head = "raw6" if head == "cap6" else head
        baseline_head = columns[1].selectbox("Reference classifier", reference_heads,
                                             index=reference_heads.index(default_head) if default_head in reference_heads else 0,
                                             format_func=head_labels.get)
        reference_label = f"{reference_labels[reference]} · {head_labels[baseline_head]}"
    st.caption(
        "CIFAR-100 BYOL · 10,000 held-out images · 100 shared first queries. "
        "Netta's acquisition rules use the lab's pinned AUM detector. Lab U-Herding uses fixed bandwidth."
    )
    pairs = match(scores, candidate, head, reference, baseline_head)
    if pairs.empty:
        st.info("These settings have no shared evaluation cells.")
        st.stop()
    return pairs, f"{ACQUISITIONS[candidate]} · {head_labels[head]}", reference_label


def group_comparison():
    labels = {"backup": "Ordinary backup", "shuffled": "Shuffled groups", "max": "Max-only coverage"}
    reference = st.selectbox("Reference", labels, format_func=labels.get)
    head = st.selectbox("Classifier", ("raw6", "raw3", "1nn"), format_func=HEADS.get)
    st.caption("Three paired seeds. Grouped backup gives second-coverage credit across different frozen feature groups.")
    return match(grouped_scores(), "grouped", head, reference, head), "Grouped backup", labels[reference]


def heatmap(summary):
    conditions = [key for key in NOISE_LABELS if key in summary.index.get_level_values("condition")]
    pp = summary["delta_pp"].unstack().reindex(index=conditions)
    wins = summary["wins"].unstack().reindex_like(pp)
    n = summary["n"].unstack().reindex_like(pp)
    span = max(1., float(np.nanmax(np.abs(pp))))
    fig = go.Figure(go.Heatmap(
        z=pp.to_numpy(), x=[f"{b:,}" for b in pp.columns], y=[NOISE_LABELS[c] for c in conditions],
        zmin=-span, zmax=span, xgap=3, ygap=3,
        colorscale=[[0, "#dc8c70"], [.5, "#f5f3ed"], [1, "#60b7a4"]],
        texttemplate="%{z:+.2f}", textfont=dict(color="#14262e", size=14),
        customdata=np.stack((wins.to_numpy(), n.to_numpy()), axis=-1),
        colorbar=dict(title="pp", thickness=13), hoverongaps=False,
        hovertemplate="%{y} · %{x} labels<br>%{z:+.2f} pp"
                      "<br>%{customdata[0]:.0f}/%{customdata[1]:.0f} paired wins<extra></extra>",
    ))
    fig.update_layout(
        height=100 + 43 * len(conditions), margin=dict(t=12, b=40, l=4, r=4),
        xaxis=dict(title="Label budget", type="category", side="bottom", automargin=True),
        yaxis=dict(autorange="reversed", automargin=True),
    )
    st.plotly_chart(fig, config={"displaylogo": False, "displayModeBar": False},
                    alt="Mean paired accuracy differences in percentage points by noise and budget.")


def select_comparison(study):
    if study == "sota":
        return strong_baselines()
    if study == "grouped":
        return group_comparison()
    if study in ("exact", "sparse"):
        heads = list(HEADS) if study == "exact" else ["raw6", "raw3", "1nn"]
        head = st.selectbox("Classifier", heads, format_func=HEADS.get)
        if study == "exact":
            data = confirmation_scores().query("acquisition in ['maxherding', 'backup_coverage']")
            arms = ("backup_coverage", "maxherding")
            note = ("Shared first 100 queries and full Gaussian affinities. The two acquisition orders are fixed. "
                    "Five seeds redraw synthetic noise. CIFAR-100N and clean each have one paired order.")
        else:
            data = sparse_scores()
            arms = ("backup", "max")
            note = ("Five fresh starts and paired noise draws. Both arms use the same sparse Gaussian affinities "
                    "and share their first 100 queries within each seed.")
        st.caption(note)
        st.markdown("Coverage uses **closest similarity + ½ × second-closest similarity** for each pool image.")
        reference = "MaxHerding" if study == "exact" else "Max-only coverage"
        return compare(data, *arms, head, head), "Backup coverage", reference

    if study in ("votes", "fixed", "channels"):
        cols = st.columns(2)
        head = cols[0].selectbox("Vote", ["sqrt6", "sqrt3"], format_func=HEADS.get)
        option = cols[1].selectbox("Reference", ["Best raw depth in each cell", "Raw vote at same depth"])
        reference = "best_raw" if option.startswith("Best") else f"raw{head[-1]}"
        arm = "maxherding" if study == "fixed" else "seeded_coverage"
        notes = {
            "votes": "Ten fresh starts and paired synthetic draws. CIFAR-100N uses fixed human annotations.",
            "fixed": "One archived order and five paired synthetic draws. CIFAR-100N uses fixed human annotations.",
            "channels": "Within each seed, the three 50% noise channels corrupt exactly the same images.",
        }
        st.caption(notes[study] + " The best raw depth is selected from 3, 4, 6 and 8 after scoring.")
        st.markdown("For each original annotation, **sum its path weights, then take their square root**.")
        data = collision_scores() if study == "channels" else confirmation_scores().query("acquisition == @arm")
        return compare(data, arm, arm, head, reference), HEADS[head], option

    stage = st.selectbox("Stage", STAGES, format_func=STAGES.get)
    student = stage in ("student", "winner_student")
    data = vast_scores(stage) if student else vast_scores("teacher", stage)
    st.caption("One archived seed. Protected five-update propagation, admission masks and confidence weights are shared.")
    if stage == "winner_student":
        st.markdown("Swap the original winner’s probability with the square-root winner’s probability.")
    else:
        st.markdown("Square-root targets preserve **each image’s total evidence** and change its distribution across classes.")
    if student:
        st.caption("Both students use the same membership and 300 epochs. The reference is the verified archived fit.")
        if 0 < len(data) < 6:
            st.caption(f"{len(data)} of 6 student cases verified.")
    if data.empty:
        st.info("Verified student scores appear here when the fits finish.")
        st.stop()
    method = "Winner swap" if stage == "winner_student" else "Square-root targets"
    return compare(data, "protected", "protected", "sqrt5", "raw5"), method, "Protected VAST"


def render():
    st.set_page_config(page_title="CIFAR-100 graph methods", layout="wide")
    st.title("CIFAR-100 graph methods")
    study = st.selectbox("Experiment", STUDIES, format_func=STUDIES.get)
    if study == "methods":
        st.markdown(Path(__file__).with_name("METHODS.md").read_text())
        return
    pairs, method, reference = select_comparison(study)
    summary = summarize(pairs)
    st.header("Paired accuracy change")
    heatmap(summary)
    available = [key for key in NOISE_LABELS if key in summary.index.get_level_values("condition")]
    preferred = "cluster50" if study == "grouped" else "sym50"
    default_condition = available.index(preferred) if preferred in available else 0
    condition = st.selectbox("Noise setting", available, index=default_condition, format_func=NOISE_LABELS.get)
    detail = summary.loc[condition].reset_index().round(3)
    detail["paired_wins"] = detail["wins"].astype(str) + " / " + detail["n"].astype(str)
    st.dataframe(
        detail[["budget", "reference_pct", "method_pct", "delta_pp", "paired_wins"]], hide_index=True,
        column_config={
            "budget": st.column_config.NumberColumn("Labels", format="%d"),
            "reference_pct": st.column_config.NumberColumn(reference + " (%)", format="%.2f"),
            "method_pct": st.column_config.NumberColumn(method + " (%)", format="%.2f"),
            "delta_pp": st.column_config.NumberColumn("Change (pp)", format="%+.2f"),
            "paired_wins": "Paired wins",
        },
    )
    with st.expander("Scores by seed"):
        st.dataframe(
            pairs.loc[pairs["condition"].eq(condition)].drop(columns=["condition", "win", "tie"]).round(3),
            hide_index=True,
        )
    st.download_button("Download paired scores", pairs.to_csv(index=False), f"{study}_pairs.csv", "text/csv")


render()
