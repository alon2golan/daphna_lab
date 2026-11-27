"""Explain the verified 26–27 November research results and their comparisons."""

import pandas as pd
import streamlit as st

from analyze import CONDITIONS, ROOT, compare, confirmation_scores, read, summarize, vast_scores
from sota_scores import acquisition_scores, best_reference, match

AUDIT_COHORTS = {"coverage": "Seeded graph coverage", "maxherding": "Archived MaxHerding"}
AUDIT_REFERENCES = {
    "loo_audit": "Leave-one-out audits", "fresh_clean": "Buy new clean labels",
    "single_footprint": "Unsigned impact", "diverse_footprint": "Diverse unsigned impact",
    "random_audit": "Random audits",
}
AUDIT_HEADS = {"test_correct": "3-hop graph", "test_1nn_correct": "Cosine 1NN"}


def pp_range(values):
    return f"{values.min():+.2f} to {values.max():+.2f} pp"


def audit_pairs(scores, cohort, reference, head):
    selected = scores.loc[
        scores["acquisition"].eq(cohort) & scores["policy"].isin(("signed_peer", reference))
    ]
    if not (10 * selected["clean_requests"]).eq(selected["budget"]).all():
        raise ValueError("Audit comparison has unmatched clean request budgets")
    rows = selected.rename(columns={"acquisition": "order", "policy": "acquisition", "test_count": "samples"})
    return compare(rows, "signed_peer", reference, head, head)


def audit_detail():
    controls = st.columns(3)
    cohort = controls[0].selectbox("Acquisition order", AUDIT_COHORTS, format_func=AUDIT_COHORTS.get)
    reference = controls[1].selectbox("Reference", AUDIT_REFERENCES, format_func=AUDIT_REFERENCES.get)
    head = controls[2].selectbox("Test classifier", AUDIT_HEADS, format_func=AUDIT_HEADS.get)
    st.caption("The x-axis counts starting noisy labels. Each arm then spends 10% extra on simulated clean answers.")
    scores = pd.read_csv(ROOT.parent / "active_audit_study/benchmark_metrics.csv")
    return audit_pairs(scores, cohort, reference, head), "Signed audits", AUDIT_REFERENCES[reference]


def runner_up(open_view):
    st.latex(r"w_i(x)=\min\!\big(h_i(x),h_{(2)}(x)\big)")
    st.markdown(
        "First sum all paths from image $x$ to annotation $i$ to get $h_i$. "
        "Cap the strongest source at the runner-up. For example, 16 and 4 become 4 and 4. "
        "Keep a lone source at full weight."
    )

    graph = acquisition_scores().query("seed > 0 and budget >= 800 and training_filter != 'mlp'")
    all_labels = graph.loc[graph.training_filter.eq("all")]
    strongest = pd.concat([all_labels, best_reference(graph, ("raw3", "raw6"))], ignore_index=True)
    comparisons = (
        ("Raw six-hop NPC", match(all_labels, "npc", "cap6", "npc", "raw6")),
        ("Raw six-hop Weighted NPC", match(all_labels, "weighted_npc", "cap6", "weighted_npc", "raw6")),
        ("Best raw NPC or Weighted NPC", match(strongest, "npc", "cap6", "envelope", "best")),
    )
    rows = []
    for label, pairs in comparisons:
        cells = summarize(pairs)["delta_pp"]
        rows.append({"Reference": label, "Mean change": f"{cells.mean():+.2f} pp",
                     "Positive cells": f"{cells.gt(0).sum()} / {len(cells)}"})
    st.dataframe(rows, hide_index=True)
    st.caption("Three fresh noise draws per cell. Symmetric and asymmetric noise at 30%, 40% and 50%, with 800–2,000 labels.")
    st.markdown(
        "Look at **symmetric 50%** for the largest gains. Check the best-raw reference for a tougher comparison "
        "and the AUM-filtered classifiers for the near-zero effect after hard filtering."
    )
    left, right = st.columns(2)
    left.button("Open NPC cap grid", on_click=open_view, args=("sota",), key="guide_cap_npc")
    right.button("Compare with best raw", on_click=open_view, args=("sota",),
                 kwargs={"reference": "envelope"}, key="guide_cap_best")


def square_root(open_view):
    st.latex(r"w_i(x)=\sqrt{h_i(x)}")
    st.markdown("Sum paths per original annotation first. The square root softens repeated paths from one label.")
    fresh = confirmation_scores().query("acquisition == 'seeded_coverage' and budget >= 800")
    cells = summarize(compare(fresh, "seeded_coverage", "seeded_coverage", "sqrt6", "best_raw"))["delta_pp"]
    rows = []
    for label, conditions in (
        ("Symmetric 30–50%", ("sym30", "sym40", "sym50")),
        ("Asymmetric 50%", ("asym50",)), ("Clean", ("clean",)),
    ):
        values = cells.loc[list(conditions)]
        rows.append({"Noise": label, "Change vs best raw depth": pp_range(values)})
    st.dataframe(rows, hide_index=True)
    st.caption("Ten fresh starts, 800–1,600 labels. The raw depth is selected after scoring each cell.")
    st.markdown(
        "Look at the symmetric rows against **clean** and **asymmetric 50%**. "
        "The fixed-corruption view holds the wrong images constant and changes which wrong classes they receive."
    )
    left, right = st.columns(2)
    left.button("Open fresh-start grid", on_click=open_view, args=("votes",), key="guide_vote_fresh")
    right.button("Open fixed-corruption test", on_click=open_view, args=("channels",), key="guide_vote_channels")


def winner_swap(open_view):
    st.latex(r"a=\arg\max p,\quad b=\arg\max q,\quad (p'_a,p'_b)=(p_b,p_a)")
    st.markdown(
        "With original probabilities $p$ and square-root probabilities $q$, swap the two winners’ entries in $p$. "
        "The new winner gets the original confidence, and the distribution keeps its entropy."
    )
    full = vast_scores("student")
    swap = vast_scores("winner_student")
    keys = ["condition", "budget", "seed", "samples", "raw5"]
    joined = full.merge(swap, on=keys, validate="one_to_one", suffixes=("_full", "_swap"))
    rows = [{
        "Noise": CONDITIONS[row.condition], "Labels": row.budget,
        "Protected VAST": f"{100 * row.raw5 / row.samples:.2f}%",
        "Full square root": f"{100 * row.sqrt5_full / row.samples:.2f}%",
        "Winner swap": f"{100 * row.sqrt5_swap / row.samples:.2f}%",
    } for row in joined.sort_values(["budget", "condition"]).itertuples(index=False)]
    st.dataframe(rows, hide_index=True)
    st.caption("One archived seed. The same membership, confidence weights, optimizer steps and 300 epochs in each fit.")
    st.markdown("Look at **symmetric 50%, 1,200 labels** and compare all three trained scores on that row.")
    left, right = st.columns(2)
    left.button("Open winner-swap grid", on_click=open_view, args=("vast",),
                kwargs={"stage": "winner_student"}, key="guide_vast_swap")
    right.button("Open full compression", on_click=open_view, args=("vast",),
                 kwargs={"stage": "student"}, key="guide_vast_full")


def signed_audit(open_view):
    st.latex(r"A_i=[g_i]_+\sum_x\left[p_{-i}(c_{\rm after}\mid x)-p_{-i}(c_{\rm before}\mid x)\right]")
    st.markdown(
        "Let $g_i$ be the leave-one-out gap favoring a replacement label. "
        "Simulate the switch and add the signed change in normalized votes from the other annotations. "
        "Spend clean verification calls on the highest scores."
    )
    scores = pd.read_csv(ROOT.parent / "active_audit_study/benchmark_metrics.csv")
    rows = []
    for cohort, label in AUDIT_COHORTS.items():
        deltas = []
        for reference in ("loo_audit", "fresh_clean"):
            paired = audit_pairs(scores, cohort, reference, "test_correct")
            synthetic = paired.loc[paired.condition.str.startswith(("sym", "asym"))]
            deltas.append(summarize(synthetic)["delta_pp"])
        both = (deltas[0].gt(0) & deltas[1].gt(0)).sum()
        rows.append({"Acquisition": label, "Positive against both": f"{both} / {len(deltas[0])}",
                     "vs leave-one-out": pp_range(deltas[0]), "vs clean expansion": pp_range(deltas[1])})
    st.dataframe(rows, hide_index=True)
    st.caption("Starting budgets of 400–1,200. Each arm buys 10% extra clean answers, simulated from canonical labels.")
    st.markdown(
        "Look at the symmetric and asymmetric rows under both references. "
        "In the clean control, leave-one-out ties and buying new clean labels helps more."
    )
    st.button("Open signed audit grid", on_click=open_view, args=("audit",), key="guide_audit")


def backup_coverage(open_view):
    st.latex(r"F(S)=\sum_x\left(s_1(x)+0.5s_2(x)\right)")
    st.markdown("Score the closest and second-closest acquired images. A backup gives evidence when the closest label is wrong.")
    fixed = confirmation_scores().query("budget >= 800 and acquisition in ['maxherding', 'backup_coverage']")
    cells = summarize(compare(fixed, "backup_coverage", "maxherding", "raw6", "raw6"))["delta_pp"]
    synthetic = cells.loc[cells.index.get_level_values("condition").str.startswith(("sym", "asym"))]
    scores = acquisition_scores()
    heavy = scores.query("seed == 0 and budget >= 800 and condition == 'sym50' and training_filter == 'all'")
    netta = heavy.loc[heavy.method.isin(("npc", "weighted_npc", "maxherding_nas")) & heavy["head"].eq("raw6")]
    netta = netta.groupby(["seed", "condition", "budget"], as_index=False).correct.max()
    against_netta = match(pd.concat([heavy, netta.assign(method="netta", head="raw6")], ignore_index=True),
                          "backup_coverage", "raw6", "netta", "raw6")
    mlp = scores.loc[scores.training_filter.eq("mlp")]
    against_mlp = match(pd.concat([mlp, best_reference(mlp, ("mlp",))], ignore_index=True),
                        "backup_coverage", "mlp", "envelope", "best")
    st.dataframe([
        {"Comparison": "Exact MaxHerding · same graph", "Result": f"{synthetic.gt(0).sum()}/{len(synthetic)} wins"},
        {"Comparison": "Best Netta arm · symmetric 50%, seed 0", "Result": f"{against_netta.delta_pp.mean():+.1f} pp"},
        {"Comparison": "Best archived shared MLP · seed 0",
         "Result": f"{against_mlp.win.sum()}/{len(against_mlp)} wins, {against_mlp.delta_pp.mean():+.2f} pp"},
    ], hide_index=True)
    st.markdown(
        "Look at the MaxHerding gains at **800–2,000 labels**, then open the NPC check. "
        "Choose **Shared 300-epoch MLP** for the trained-classifier transfer."
    )
    left, right = st.columns(2)
    left.button("Open MaxHerding grid", on_click=open_view, args=("exact",), key="guide_backup_max")
    right.button("Open NPC comparison", on_click=open_view, args=("sota",),
                 kwargs={"cohort": "Archived seed 0"}, key="guide_backup_npc")


def source_stability(_open_view):
    st.latex(r"s(x)=\frac{p_{\rm win}-p_{\rm rival}-f}{1-f}")
    st.markdown(
        "Use the margin after removing the strongest supporting annotation. "
        "$f$ is that source’s share of total evidence. Rerank the 10% shortlist with the same predicted-class counts."
    )
    folder = ROOT.parent / "source_support_study"
    replay = read(folder / "replay_results.json")
    precision = []
    for condition in ("sym20", "sym50", "asym50"):
        gains = [row["top10"]["source_stability"]["precision"] - row["top10"]["confidence"]["precision"]
                 for row in replay if row["condition"] == condition]
        label = "Symmetric 20%" if condition == "sym20" else CONDITIONS[condition]
        precision.append({"Noise": label, "Shortlist precision change": f"{100 * sum(gains) / len(gains):+.2f} pp"})
    st.dataframe(precision, hide_index=True)
    students = read(folder / "student_results.json")
    table = []
    for row in sorted(students, key=lambda entry: (entry["budget"], entry["condition"])):
        reference = row["arms"]["confidence"]["test_accuracy_percent"]
        candidate = row["arms"]["source_stability"]["test_accuracy_percent"]
        label = "Clean" if row["condition"] == "sym0" else CONDITIONS[row["condition"]]
        table.append({"Noise": label, "Labels": row["budget"], "Confidence shortlist": f"{reference:.2f}%",
                      "Source-stability shortlist": f"{candidate:.2f}%", "Change": f"{candidate - reference:+.2f} pp"})
    st.dataframe(table, hide_index=True)
    st.markdown(
        "Compare shortlist **precision** with student **test accuracy**. "
        "The 1,000-label symmetric-50 student gain is one seed, and applying the ranking to the full gate reduced precision."
    )


def render_guide(open_view):
    st.header("Research guide")
    st.caption("26–27 November 2026 · planned runs completed and verified · one frozen CIFAR-100 BYOL test split")
    sections = {
        "Runner-up cap": runner_up,
        "Square-root votes": square_root,
        "VAST winner swap": winner_swap,
        "Signed graph audits": signed_audit,
        "Backup coverage": backup_coverage,
        "VAST source-stability shortlist": source_stability,
    }
    method = st.selectbox("Method", sections)
    sections[method](open_view)
    with st.expander("What we tried and what failed"):
        st.markdown((ROOT / "METHODS.md").read_text())
