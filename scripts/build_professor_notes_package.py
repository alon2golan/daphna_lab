#!/usr/bin/env python3
"""Build response artifacts for the 2026-05-28 professor PDF notes.

The script only uses existing experiment summaries. It creates:

- a note-by-note response checklist
- compact CSVs for the new diagnostics
- plots that can be inserted into the project summary or shown in the meeting
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/cs/labs/daphna/ido.dotan/projects/daphna_lab")
RESULTS_DIR = ROOT / "results_summaries" / "professor_notes_2026_05_28"
PLOTS_DIR = ROOT / "visual_outputs" / "professor_notes_2026_05_28"
NOTES_DIR = ROOT / "session_notes"

BEST_LOCAL = {
    "CIFAR10": "k=3500",
    "CIFAR100": "k=1440",
}
LOCAL_TO_KEEP = {
    "CIFAR10": {"global", "K=60", "K=3500", "K=10000"},
    "CIFAR100": {"global", "K=60", "K=1440", "K=5000"},
}


def _ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> Path:
    path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _alpha_gain_table() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results_summaries/maxherding_density_norm/alpha_gain/alpha_gain_mean.csv")


def _probcover_table(dataset: str) -> pd.DataFrame:
    path = ROOT / f"results_summaries/reference_validation_tables/{dataset.lower()}_probcover_global_reference_combined.csv"
    df = pd.read_csv(path)
    nn1_cols = [col for col in df.columns if col.startswith("nn1_seed_")]
    image_cols = [col for col in df.columns if col.startswith("from_images_seed_")]
    logistic_cols = [col for col in df.columns if col.startswith("logistic_seed_")]
    out = pd.DataFrame({"labels": df["labeled_set_size"]})
    out["probcover_nn1_mean"] = df[nn1_cols].mean(axis=1, skipna=True)
    out["probcover_from_images_mean"] = df[image_cols].mean(axis=1, skipna=True)
    out["probcover_logistic_mean"] = df[logistic_cols].mean(axis=1, skipna=True)
    out["probcover_paper_result"] = df["paper_result"]
    return out


def build_probcover_comparison() -> pd.DataFrame:
    alpha = _alpha_gain_table()
    rows: list[dict[str, object]] = []
    for dataset in ["CIFAR10", "CIFAR100"]:
        prob = _probcover_table(dataset)
        for setting in ["global", BEST_LOCAL[dataset]]:
            part = alpha[(alpha["dataset"] == dataset) & (alpha["setting"] == setting)]
            for labels in sorted(part["labels"].unique()):
                label_part = part[part["labels"] == labels]
                row: dict[str, object] = {
                    "dataset": dataset,
                    "setting": setting,
                    "labels": int(labels),
                    "maxherding_alpha0_mean": float(label_part["mean_baseline_accuracy"].iloc[0]),
                }
                for alpha_value in sorted(label_part["alpha"].unique()):
                    value = float(label_part[label_part["alpha"] == alpha_value]["mean_accuracy"].iloc[0])
                    row[f"maxherding_alpha{alpha_value:g}_mean"] = value
                prob_row = prob[prob["labels"] == labels]
                if not prob_row.empty:
                    for col in prob.columns:
                        if col != "labels":
                            row[col] = float(prob_row[col].iloc[0])
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "probcover_vs_maxherding_alpha.csv", index=False)
    return out


def plot_probcover_comparison(comp: pd.DataFrame) -> list[Path]:
    paths = []
    alpha_cols = [
        "maxherding_alpha0_mean",
        "maxherding_alpha0.25_mean",
        "maxherding_alpha0.5_mean",
        "maxherding_alpha0.75_mean",
    ]
    colors = {
        "maxherding_alpha0_mean": "#222222",
        "maxherding_alpha0.25_mean": "#4C78A8",
        "maxherding_alpha0.5_mean": "#F58518",
        "maxherding_alpha0.75_mean": "#54A24B",
    }
    labels = {
        "maxherding_alpha0_mean": "MaxHerding alpha=0",
        "maxherding_alpha0.25_mean": "MaxHerding alpha=0.25",
        "maxherding_alpha0.5_mean": "MaxHerding alpha=0.5",
        "maxherding_alpha0.75_mean": "MaxHerding alpha=0.75",
    }
    for dataset in ["CIFAR10", "CIFAR100"]:
        for setting in ["global", BEST_LOCAL[dataset]]:
            part = comp[(comp["dataset"] == dataset) & (comp["setting"] == setting)].sort_values("labels")
            fig, ax = plt.subplots(figsize=(9.0, 5.4))
            for col in alpha_cols:
                if col in part:
                    ax.plot(part["labels"], part[col], marker="o", linewidth=1.9, label=labels[col], color=colors[col])
            ax.plot(
                part["labels"],
                part["probcover_nn1_mean"],
                marker="s",
                linewidth=2.2,
                linestyle="--",
                color="#D62728",
                label="ProbCover NN1",
            )
            ax.set_title(f"{dataset}: MaxHerding density normalization vs ProbCover ({setting})")
            ax.set_xlabel("Labeled points")
            ax.set_ylabel("Test accuracy (%)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            paths.append(_save(fig, f"{dataset.lower()}_{setting.replace('=', '')}_probcover_vs_alpha.png"))
    return paths


def _setting_name(row: pd.Series) -> str:
    if not bool(row["local_sigma"]):
        return "global"
    return f"K={int(row['local_k'])}"


def build_selected_degree_budget_tables() -> pd.DataFrame:
    frames = []
    for path in [
        ROOT / "results_summaries/maxherding_density_selection/selection_episode_summary.csv",
        ROOT / "results_summaries/maxherding_tinyimagenet_maor_density_selection/selection_episode_summary.csv",
    ]:
        df = pd.read_csv(path)
        df["setting"] = df.apply(_setting_name, axis=1)
        df["labels"] = (df["episode"] + 1) * df["num_selected"]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    density_frames = []
    for path in [
        ROOT / "results_summaries/maxherding_density_selection/density_summary.csv",
        ROOT / "results_summaries/maxherding_tinyimagenet_maor_density_selection/density_summary.csv",
    ]:
        df = pd.read_csv(path)
        df["setting"] = df.apply(_setting_name, axis=1)
        density_frames.append(df)
    density = pd.concat(density_frames, ignore_index=True)[
        ["dataset", "seed", "setting", "effective_degree_mean", "effective_degree_std"]
    ].rename(
        columns={
            "effective_degree_mean": "dataset_effective_degree_mean",
            "effective_degree_std": "dataset_effective_degree_std",
        }
    )

    merged = data.merge(density, on=["dataset", "seed", "setting"], how="left")
    merged["selected_to_dataset_degree_ratio"] = (
        merged["selected_effective_degree_mean"] / merged["dataset_effective_degree_mean"]
    )
    merged["degree_rank_proxy"] = merged["selected_to_dataset_degree_ratio"]

    cols = [
        "dataset",
        "seed",
        "setting",
        "labels",
        "selected_effective_degree_mean",
        "selected_effective_degree_std",
        "dataset_effective_degree_mean",
        "dataset_effective_degree_std",
        "selected_to_dataset_degree_ratio",
        "degree_rank_proxy",
        "test_accuracy_after_episode",
    ]
    out = merged[cols].sort_values(["dataset", "seed", "setting", "labels"])
    out.to_csv(RESULTS_DIR / "selected_degree_by_budget.csv", index=False)

    grouped = (
        out.groupby(["dataset", "setting", "labels"], as_index=False)
        .agg(
            selected_degree_mean=("selected_effective_degree_mean", "mean"),
            selected_degree_std_mean=("selected_effective_degree_std", "mean"),
            dataset_degree_mean=("dataset_effective_degree_mean", "mean"),
            selected_to_dataset_degree_ratio_mean=("selected_to_dataset_degree_ratio", "mean"),
            selected_to_dataset_degree_ratio_std=("selected_to_dataset_degree_ratio", "std"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "setting", "labels"])
    )
    grouped.to_csv(RESULTS_DIR / "selected_degree_by_budget_mean.csv", index=False)
    return grouped


def plot_selected_degree_budget(grouped: pd.DataFrame) -> list[Path]:
    paths = []
    for dataset in ["CIFAR10", "CIFAR100", "TINYIMAGENET"]:
        part = grouped[grouped["dataset"] == dataset]
        if part.empty:
            continue
        keep = LOCAL_TO_KEEP.get(dataset, set(part["setting"].unique()))
        part = part[part["setting"].isin(keep)]

        fig, ax = plt.subplots(figsize=(9.4, 5.5))
        for setting, group in part.groupby("setting"):
            group = group.sort_values("labels")
            width = 2.7 if setting == "global" else 1.8
            ax.plot(group["labels"], group["selected_degree_mean"], marker="o", linewidth=width, label=setting)
        ax.set_title(f"{dataset}: selected-batch effective degree by budget")
        ax.set_xlabel("Labeled points selected so far")
        ax.set_ylabel("Selected-batch effective degree")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        paths.append(_save(fig, f"{dataset.lower()}_selected_degree_mean_by_budget.png"))

        fig, ax = plt.subplots(figsize=(9.4, 5.5))
        for setting, group in part.groupby("setting"):
            group = group.sort_values("labels")
            width = 2.7 if setting == "global" else 1.8
            ax.plot(
                group["labels"],
                group["selected_to_dataset_degree_ratio_mean"],
                marker="o",
                linewidth=width,
                label=setting,
            )
        ax.axhline(1.0, color="black", linewidth=1.1, linestyle="--", label="Dataset average")
        ax.set_title(f"{dataset}: selected degree normalized by dataset mean")
        ax.set_xlabel("Labeled points selected so far")
        ax.set_ylabel("Selected / dataset effective degree")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        paths.append(_save(fig, f"{dataset.lower()}_selected_degree_ratio_by_budget.png"))
    return paths


def plot_existing_tiny_degree_accuracy() -> Path:
    src = ROOT / "visual_outputs/tinyimagenet_maor_degree_study/cifar_style/tinyimagenet_maor_effective_degree_vs_final_accuracy.png"
    # Keep the source in place; the response package points to the canonical Tiny plot.
    return src


def write_response_checklist() -> Path:
    path = NOTES_DIR / "professor_response_checklist_2026_05_28.md"
    text = """# Professor Response Checklist - 2026-05-28

This note maps every PDF comment to the concrete response before the next meeting.

## 1. Degree Definition

Comment: "is the degree" and "remind me how the degree is computed?"

What she is asking: define the degree used in density normalization and say whether it is the clipped graph degree or `sum_j K_ij`.

Status: partially addressed in the code, needs clearer text in the summary.

Response:

- We used the graph after the ProbCover-style cutoff.
- `effective` degree is the number of nonzero neighbors after removing the diagonal.
- `weighted` degree is `sum_j K_ij` after removing the diagonal.
- The later alpha-wide experiments used `weighted` degree.
- The density diagnostics often report `effective` degree because it is easier to interpret as graph connectivity.

Needed action: update the summary text and cite the new selected-degree plots.

## 2. Alpha Definition

Comment: "is this the alpha you use in the results graphs below?"

What she is asking: confirm that the formula alpha is the same as the plotted alpha.

Status: clarification only.

Response:

- Yes. The plotted alpha is the same total exponent in `D^{-alpha/2} K D^{-alpha/2}`.
- The `/2` appears because symmetric normalization splits the total exponent equally between the two endpoints.

Needed action: update the formula explanation.

## 3. Alpha And Geometric Mean

Comment: "do I understand correctly that alpha is the actual power, and therefore alpha=0.5 is the geometric mean of the rank of the two edge-nodes?"

What she is asking: verify the interpretation of symmetric normalization.

Status: clarification plus one correction.

Response:

- It is degree, not rank, in the current implementation.
- For `alpha=1`, the denominator is the geometric mean of the two endpoint degrees.
- For `alpha=0.5`, the penalty is weaker: `(degree_i degree_j)^0.25`.

Needed action: use this exact explanation in the meeting.

## 4. Average Rank Per Budget

Comment: "can you compute the average rank for each budget?"

What she is asking: check whether selected points become less extreme in degree over time.

Status: partially addressed.

Existing output:

- `results_summaries/professor_notes_2026_05_28/selected_degree_by_budget.csv`
- `results_summaries/professor_notes_2026_05_28/selected_degree_by_budget_mean.csv`
- `visual_outputs/professor_notes_2026_05_28/*selected_degree_ratio_by_budget.png`

Important limitation: existing CSVs contain selected-point degrees and dataset degree summaries, but not the full per-point degree distribution. Therefore the current plot is a degree-ratio/rank-proxy, not an exact percentile rank.

Needed action: if she specifically wants percentile rank, run an additional offline rank computation over sampled anchor degrees or exact full degrees.

## 5. Delta Sensitivity And ProbCover Comparison

Comment: "in 100-400, check sensitivity to delta; compare to ProbCover."

What she is asking: test whether alpha gains depend on optimal delta, and add ProbCover to the same plots.

Status: addressed. The delta-sensitivity runs completed successfully and were aggregated.

Existing output:

- `results_summaries/professor_notes_2026_05_28/probcover_vs_maxherding_alpha.csv`
- `visual_outputs/professor_notes_2026_05_28/*probcover_vs_alpha.png`
- `results_summaries/professor_notes_2026_05_28/delta_sensitivity_by_episode.csv`
- `results_summaries/professor_notes_2026_05_28/delta_sensitivity_mean_by_episode.csv`
- `results_summaries/professor_notes_2026_05_28/delta_sensitivity_gain_vs_alpha0.csv`
- `visual_outputs/professor_notes_2026_05_28/delta_sensitivity/*`

Run verification:

- Expected runs: 120
- Completed runs with result arrays: 120
- Seeds per dataset/setting/delta/alpha: 3
- No OOM, timeout, cancellation, or traceback was found in the logs.

Main result:

- CIFAR10: alpha=0.5 often helps early, but the effect is unstable across delta values. The best final result is still alpha=0 with delta around 0.75 to 0.80.
- CIFAR100: alpha=0.5 helps at low budgets, especially 100 to 300 labels. Smaller deltas, 0.55 and 0.60, preserve the gain longer; larger deltas, 0.70 and 0.75, make alpha harmful later.
- The delta sensitivity supports the conclusion that density normalization is useful only in a budget- and graph-scale-dependent regime.

Needed action: discuss these results in the meeting and decide whether the next adaptive rule should control alpha, delta, or both.

## 6. "Meaning?" In Dynamic Alpha Section

What she is asking: a sentence was too vague.

Status: phrasing change.

Response: say that fixed alpha helped some early budgets, but fixed hand-written schedules did not beat alpha=0 at final budget.

Needed action: update the summary text.

## 7. Single Sigma / Global K

Comment: "can you repeat this experiment with a single sigma?"

What she is asking: separate density normalization from local sigma.

Status: already addressed by existing global-sigma alpha experiments.

Evidence:

- The alpha-wide and alpha-gain plots include `global`.
- The new ProbCover comparison plots include global-sigma MaxHerding with alpha normalization.

Needed action: make this explicit in the summary.

## 8. Greedy Max Coverage Definition

Comment: "is this the definition of greedy max coverage?"

What she is asking: avoid an algorithmic overclaim.

Status: phrasing change.

Response: call it a greedy coverage-style selection rule, not exactly classical greedy max coverage.

Needed action: update summary wording.

## 9. "Does Not Persist" Versus "Degrades Later Performance"

What she is asking: make the result more precise.

Status: phrasing change.

Response: say "the early gain often does not persist, and in several settings becomes negative at later budgets."

Needed action: update summary wording.

## 10. Degree Versus Local Kernel

Comment: "These scores are not totally different..."

What she is asking: avoid separating degree and locality too strongly.

Status: interpretation change.

Response: degree and locality are related. Local sigma affects the kernel graph, and degree is the measurable graph statistic that best explains the behavior we observed.

Needed action: update main conclusion.

## 11. Average Degree And Std As Function Of Budget

Comment: asks for graph of average degree and standard deviation as a function of budget.

Status: selected-batch version done from existing data; remaining-pool/full-graph version still requires recomputation if requested.

Existing output:

- `visual_outputs/professor_notes_2026_05_28/*selected_degree_mean_by_budget.png`
- `visual_outputs/professor_notes_2026_05_28/*selected_degree_ratio_by_budget.png`

Needed action: clarify in the meeting whether the selected-batch version answers her question or whether she wants average degree/std over the entire remaining graph after each episode.

## 12. Switch To Random Or Mixture Strategy

What she is asking: connect degree collapse to strategy switching.

Status: conceptual addition.

Response: when average degree becomes very small, the coverage signal may be weak. This suggests switching to random or mixing strategies, related to Hacohen and Weinshall, NeurIPS 2023, SelectAL.

Needed action: add this to future work.

## 13. Three-Phase Experiment Clarity

Comment: "isn't this what you did in the 3 phase experiments?"

What she is asking: distinguish fixed schedules from adaptive rules.

Status: clarification.

Response: the three-phase schedule was fixed by episode number. The proposed next step is adaptive because it would use measured graph statistics such as degree/rank.

Needed action: update dynamic-alpha explanation.

## 14. ProbCover Figure 9 / Delta Decreases With Budget

What she is asking: reconcile our degree heuristic with the observation that optimal delta decreases as budget increases.

Status: partially addressed by the delta-sensitivity experiment.

Response: decreasing delta can reduce redundant coverage at higher budgets and change which points remain useful after already-covered regions are removed. In the delta sweep, smaller deltas preserved the alpha benefit longer on CIFAR100, while larger deltas made alpha harmful later.

Needed action: discuss whether this motivates an adaptive delta/alpha rule rather than a fixed-alpha decay.

## 15. Positive Feedback

Comment: "Very thoughtful and well put."

Status: no action.

Response: keep the thesis-style takeaway, but update it with the refined degree/locality wording.
"""
    path.write_text(text, encoding="ascii")
    return path


def main() -> None:
    _ensure_dirs()
    checklist = write_response_checklist()
    comp = build_probcover_comparison()
    degree = build_selected_degree_budget_tables()
    paths = []
    paths.extend(plot_probcover_comparison(comp))
    paths.extend(plot_selected_degree_budget(degree))
    paths.append(plot_existing_tiny_degree_accuracy())
    print(f"Wrote checklist: {checklist}")
    print(f"Wrote CSVs under: {RESULTS_DIR}")
    print(f"Wrote plots under: {PLOTS_DIR}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
