"""Explore paired CIFAR-100 results for backup coverage and square-root votes."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from analyze import CHANNELS, CONDITIONS, collision_scores, compare, confirmation_scores, sparse_scores, summarize, vast_scores

STUDIES = {
    "exact": "Backup coverage · exact MaxHerding",
    "sparse": "Backup coverage · sparse affinities",
    "votes": "Square-root votes · fresh acquisitions",
    "fixed": "Square-root votes · archived MaxHerding",
    "channels": "Square-root votes · fixed corrupted images",
    "vast": "Square-root targets · protected VAST",
}
NOISE_LABELS = {**CONDITIONS, **CHANNELS}
HEADS = {
    "raw6": "6-hop graph", "raw3": "3-hop graph", "raw4": "4-hop graph", "raw8": "8-hop graph",
    "sqrt3": "3-hop square root", "sqrt6": "6-hop square root", "1nn": "Cosine 1NN",
}
STAGES = {"student": "Trained student · test set", "unlabeled": "Teacher · full unlabeled pool",
          "admitted": "Teacher · admitted pseudo-labels"}


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
    data = vast_scores("student") if stage == "student" else vast_scores("teacher", stage)
    st.caption("One archived seed. Protected five-update propagation, admission masks and confidence weights are shared.")
    st.markdown("Square-root targets preserve **each image’s total evidence** and change its distribution across classes.")
    if stage == "student":
        st.caption("Both students use the same membership and 300 epochs. The reference is the verified archived fit.")
        if 0 < len(data) < 6:
            st.caption(f"{len(data)} of 6 student cases verified.")
    if data.empty:
        st.info("Verified student scores appear here when the fits finish.")
        st.stop()
    return compare(data, "protected", "protected", "sqrt5", "raw5"), "Square-root targets", "Protected VAST"


def render():
    st.set_page_config(page_title="CIFAR-100 graph methods", layout="wide")
    st.title("CIFAR-100 graph methods")
    study = st.selectbox("Experiment", STUDIES, format_func=STUDIES.get)
    pairs, method, reference = select_comparison(study)
    summary = summarize(pairs)
    st.header("Paired accuracy change")
    heatmap(summary)
    available = [key for key in NOISE_LABELS if key in summary.index.get_level_values("condition")]
    default_condition = available.index("sym50") if "sym50" in available else 0
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
