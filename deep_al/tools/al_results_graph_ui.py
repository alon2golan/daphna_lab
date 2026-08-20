#!/usr/bin/env python3
"""Interactive and batch graph generator for the active-learning results table.

UI:
    python al_results_graph_ui.py \
        --input /full/path/original_features_456_runs.csv \
        --ui \
        --output-dir /full/path/graphs

Requested initial examination:
    python al_results_graph_ui.py \
        --input /full/path/original_features_456_runs.csv \
        --preset initial-examination \
        --output-dir /full/path/graphs

The script saves publication-ready PDF and SVG files for LaTeX, plus PNG
files for quick inspection.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "final_labeled_size",
    "linear_train_acc",
    "linear_val_acc",
    "linear_test_acc",
    "knn_train_acc",
    "knn_val_acc",
    "knn_test_acc",
    "runtime_sec",
    "fixed_labels_total",
    "correctly_fixed_labels_total",
    "clean_corrupted_total",
    "wrong_to_wrong_total",
]

STANDARD_AL_ORDER = [
    "probcover",
    "maxherding",
    "typiclust",
    "coreset",
    "random",
    "margin",
]

NOISE_ORDER = ["clean", "sym", "asym", "cifar100n"]

FRIENDLY = {
    "probcover": "ProbCover",
    "maxherding": "MaxHerding",
    "typiclust": "TypiClust",
    "coreset": "CoreSet",
    "random": "Random",
    "margin": "Margin",
    "none": "None",
    "label_cleaning": "Label cleaning",
    "crossval": "Cross-validation",
    "temporal": "Temporal",
    "aum": "AUM",
    "cross_entropy": "Cross-entropy",
    "gce": "GCE",
    "bootstrap_soft": "Bootstrap-soft",
    "bootstrap_hard": "Bootstrap-hard",
    "clean": "Clean",
    "sym": "Symmetric noise",
    "asym": "Asymmetric noise",
    "cifar100n": "CIFAR-100N",
}


def friendly(value: object) -> str:
    text = str(value)
    return FRIENDLY.get(text, text.replace("_", " ").title())


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_") or "graph"


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
    })


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    df = pd.read_csv(path)

    required = {
        "status",
        "noise",
        "al_method",
        "noise_method",
        "loss",
        "final_labeled_size",
        "linear_test_acc",
        "knn_test_acc",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["status", "noise", "al_method", "noise_method", "loss"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[df["status"].str.lower() == "completed"].copy()

    # The standard MaxHerding baseline appears in both the general and NPC grids.
    # Keep one identical configuration per budget/noise.
    dedup_keys = [
        "final_labeled_size",
        "noise",
        "al_method",
        "noise_method",
        "loss",
    ]
    df = df.sort_values(dedup_keys + ["source_file"], kind="stable")
    df = df.drop_duplicates(dedup_keys, keep="first").reset_index(drop=True)

    df["config_label"] = df.apply(make_config_label, axis=1)
    df["method_family"] = df.apply(method_family, axis=1)
    return df


def npc_base_method(al_method: str) -> str:
    if "probcover" in al_method:
        return "probcover"
    if "maxherding" in al_method:
        return "maxherding"
    return al_method


def npc_detector(al_method: str) -> str:
    for detector in ("crossval", "knn", "aum", "oracle"):
        if al_method.endswith("_" + detector):
            return detector
    return "npc"


def method_family(row: pd.Series) -> str:
    al = str(row["al_method"])
    noise_method = str(row["noise_method"])
    loss = str(row["loss"])

    if al.startswith("npc_"):
        return "NPC selection"
    if loss != "cross_entropy":
        return "Robust loss"
    if noise_method != "none":
        return "Direct cleaning"
    return "Baseline"


def make_config_label(row: pd.Series) -> str:
    al = str(row["al_method"])
    noise_method = str(row["noise_method"])
    loss = str(row["loss"])

    if al.startswith("npc_"):
        base = friendly(npc_base_method(al))
        detector = friendly(npc_detector(al))
        return f"NPC-{base} | {detector}"

    base = friendly(al)
    if loss != "cross_entropy":
        return f"{base} | {friendly(loss)}"
    if noise_method != "none":
        return f"{base} | {friendly(noise_method)}"
    return f"{base} | Baseline"


def filter_rows(
    df: pd.DataFrame,
    *,
    budgets: Iterable[int] | None = None,
    noises: Iterable[str] | None = None,
    al_methods: Iterable[str] | None = None,
    noise_methods: Iterable[str] | None = None,
    losses: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
) -> pd.DataFrame:
    result = df.copy()

    if budgets:
        result = result[result["final_labeled_size"].isin(list(budgets))]
    if noises:
        result = result[result["noise"].isin(list(noises))]
    if al_methods:
        result = result[result["al_method"].isin(list(al_methods))]
    if noise_methods:
        result = result[result["noise_method"].isin(list(noise_methods))]
    if losses:
        result = result[result["loss"].isin(list(losses))]
    if families:
        result = result[result["method_family"].isin(list(families))]

    return result.copy()


def baseline_rows(df: pd.DataFrame, budget: int, noise: str) -> pd.DataFrame:
    result = filter_rows(
        df,
        budgets=[budget],
        noises=[noise],
        noise_methods=["none"],
        losses=["cross_entropy"],
    )
    result = result[~result["al_method"].str.startswith("npc_")]
    result = result[result["al_method"].isin(STANDARD_AL_ORDER)].copy()

    rank = {method: index for index, method in enumerate(STANDARD_AL_ORDER)}
    result["_order"] = result["al_method"].map(rank).fillna(999)
    return result.sort_values("_order")


def prepare_ranked_rows(
    rows: pd.DataFrame,
    *,
    sort_metric: str = "knn_test_acc",
    top_n: int = 0,
) -> pd.DataFrame:
    result = rows.dropna(subset=[sort_metric]).copy()
    result = result.sort_values(sort_metric, ascending=True)
    if top_n and top_n > 0:
        result = result.tail(int(top_n))
    return result


def plot_accuracy_bars(
    rows: pd.DataFrame,
    *,
    title: str,
    horizontal: bool,
    sort_metric: str,
    top_n: int = 0,
    show_values: bool = False,
) -> plt.Figure:
    if rows.empty:
        raise ValueError("No rows match the selected filters.")

    data = prepare_ranked_rows(rows, sort_metric=sort_metric, top_n=top_n)
    labels = data["config_label"].tolist()
    linear = data["linear_test_acc"].to_numpy(float)
    knn = data["knn_test_acc"].to_numpy(float)

    if horizontal:
        height = max(5.0, 0.34 * len(data) + 1.8)
        fig, ax = plt.subplots(figsize=(11.5, height))
        y = np.arange(len(data))
        bar_height = 0.38

        bars1 = ax.barh(y - bar_height / 2, linear, height=bar_height, label="Linear test")
        bars2 = ax.barh(y + bar_height / 2, knn, height=bar_height, label="1-NN test")
        ax.set_yticks(y, labels)
        ax.set_xlabel("Test accuracy (%)")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right")

        if show_values:
            ax.bar_label(bars1, fmt="%.1f", padding=2, fontsize=7)
            ax.bar_label(bars2, fmt="%.1f", padding=2, fontsize=7)
    else:
        width = max(8.0, 1.05 * len(data) + 2.2)
        fig, ax = plt.subplots(figsize=(width, 5.8))
        x = np.arange(len(data))
        bar_width = 0.38

        bars1 = ax.bar(x - bar_width / 2, linear, width=bar_width, label="Linear test")
        bars2 = ax.bar(x + bar_width / 2, knn, width=bar_width, label="1-NN test")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

        if show_values:
            ax.bar_label(bars1, fmt="%.1f", padding=2, fontsize=8)
            ax.bar_label(bars2, fmt="%.1f", padding=2, fontsize=8)

    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_single_metric_bars(
    rows: pd.DataFrame,
    *,
    metric: str,
    title: str,
    horizontal: bool,
    top_n: int = 0,
    show_values: bool = False,
) -> plt.Figure:
    if rows.empty:
        raise ValueError("No rows match the selected filters.")

    data = prepare_ranked_rows(rows, sort_metric=metric, top_n=top_n)
    labels = data["config_label"].tolist()
    values = data[metric].to_numpy(float)

    metric_names = {
        "linear_test_acc": "Linear test accuracy (%)",
        "knn_test_acc": "1-NN test accuracy (%)",
        "runtime_sec": "Runtime (seconds)",
        "fixed_labels_total": "Label-change operations",
    }
    axis_label = metric_names.get(metric, friendly(metric))

    if horizontal:
        height = max(4.5, 0.30 * len(data) + 1.8)
        fig, ax = plt.subplots(figsize=(11.5, height))
        y = np.arange(len(data))
        bars = ax.barh(y, values)
        ax.set_yticks(y, labels)
        ax.set_xlabel(axis_label)
        ax.grid(axis="x", alpha=0.25)
    else:
        width = max(8.0, 0.75 * len(data) + 2.0)
        fig, ax = plt.subplots(figsize=(width, 5.8))
        x = np.arange(len(data))
        bars = ax.bar(x, values)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylabel(axis_label)
        ax.grid(axis="y", alpha=0.25)

    if show_values:
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)

    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cleaning_outcomes(
    rows: pd.DataFrame,
    *,
    title: str,
    top_n: int = 0,
) -> plt.Figure:
    required = [
        "fixed_labels_total",
        "correctly_fixed_labels_total",
        "clean_corrupted_total",
        "wrong_to_wrong_total",
    ]
    missing = [column for column in required if column not in rows]
    if missing:
        raise ValueError(f"Missing cleaning columns: {missing}")

    data = rows.copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)

    data = data[data["fixed_labels_total"] > 0].copy()
    if data.empty:
        raise ValueError("No selected runs performed any label changes.")

    classified = (
        data["correctly_fixed_labels_total"]
        + data["clean_corrupted_total"]
        + data["wrong_to_wrong_total"]
    )
    data["unclassified_changes"] = (
        data["fixed_labels_total"] - classified
    ).clip(lower=0)

    data = data.sort_values("fixed_labels_total", ascending=True)
    if top_n and top_n > 0:
        data = data.tail(int(top_n))

    height = max(4.5, 0.34 * len(data) + 2.0)
    fig, ax = plt.subplots(figsize=(11.5, height))
    y = np.arange(len(data))
    left = np.zeros(len(data))

    components = [
        ("correctly_fixed_labels_total", "Correctly fixed"),
        ("clean_corrupted_total", "Clean → wrong"),
        ("wrong_to_wrong_total", "Wrong → wrong"),
        ("unclassified_changes", "Unclassified"),
    ]

    for column, label in components:
        values = data[column].to_numpy(float)
        ax.barh(y, values, left=left, label=label)
        left += values

    ax.set_yticks(y, data["config_label"])
    ax.set_xlabel("Label-change operations")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def add_accuracy_delta(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    baseline = df[
        (~df["al_method"].str.startswith("npc_"))
        & (df["noise_method"] == "none")
        & (df["loss"] == "cross_entropy")
    ][
        ["final_labeled_size", "noise", "al_method", metric]
    ].copy()

    baseline = baseline.rename(
        columns={
            "al_method": "_base_al_method",
            metric: "_baseline_metric",
        }
    )

    result = df.copy()
    result["_base_al_method"] = result["al_method"].map(npc_base_method)

    result = result.merge(
        baseline,
        on=["final_labeled_size", "noise", "_base_al_method"],
        how="left",
    )
    result["accuracy_delta"] = (
        result[metric] - result["_baseline_metric"]
    )
    return result


def plot_fixed_vs_accuracy_delta(
    rows: pd.DataFrame,
    *,
    metric: str,
    title: str,
    annotate: bool = True,
) -> plt.Figure:
    data = add_accuracy_delta(rows, metric)
    data["fixed_labels_total"] = pd.to_numeric(
        data["fixed_labels_total"], errors="coerce"
    )
    data = data.dropna(
        subset=["fixed_labels_total", "accuracy_delta"]
    )
    data = data[data["fixed_labels_total"] > 0]

    if data.empty:
        raise ValueError(
            "No selected runs have both fixed-label counts and a matching baseline."
        )

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.scatter(data["fixed_labels_total"], data["accuracy_delta"])
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Label-change operations")
    metric_name = (
        "1-NN test accuracy"
        if metric == "knn_test_acc"
        else "Linear test accuracy"
    )
    ax.set_ylabel(f"Δ {metric_name} vs matching baseline (points)")
    ax.set_title(title)
    ax.grid(alpha=0.25)

    if annotate:
        for _, row in data.iterrows():
            ax.annotate(
                row["config_label"],
                (row["fixed_labels_total"], row["accuracy_delta"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )

    fig.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    output_dir: str | Path,
    filename: str,
    formats: Iterable[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(filename)

    saved: list[Path] = []
    for extension in formats:
        extension = extension.lower().lstrip(".")
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    return saved


def generate_initial_examination(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Generate the 10 requested graphs, each as PDF, SVG and PNG."""
    saved: list[Path] = []

    for budget in (400, 800):
        baseline = baseline_rows(df, budget=budget, noise="clean")
        fig = plot_accuracy_bars(
            baseline,
            title=(
                f"Clean data baseline — {budget} labeled samples\n"
                "No label cleaning, cross-entropy loss"
            ),
            horizontal=False,
            sort_metric="knn_test_acc",
            show_values=True,
        )
        saved.extend(
            save_figure(
                fig,
                output_dir,
                f"baseline_clean_{budget}",
            )
        )
        plt.close(fig)

        for noise in NOISE_ORDER:
            rows = filter_rows(
                df,
                budgets=[budget],
                noises=[noise],
            )
            fig = plot_accuracy_bars(
                rows,
                title=(
                    f"All configurations — {friendly(noise)} — "
                    f"{budget} labeled samples\n"
                    "Sorted by 1-NN test accuracy"
                ),
                horizontal=True,
                sort_metric="knn_test_acc",
                top_n=0,
                show_values=False,
            )
            saved.extend(
                save_figure(
                    fig,
                    output_dir,
                    f"all_configurations_{noise}_{budget}",
                )
            )
            plt.close(fig)

    return saved


def build_custom_figure(
    df: pd.DataFrame,
    *,
    budget: int,
    noise: str,
    selected_al: list[str],
    selected_noise_methods: list[str],
    selected_losses: list[str],
    selected_families: list[str],
    graph_kind: str,
    accuracy_metric: str,
    top_n: int,
    horizontal: bool,
    show_values: bool,
    title: str,
) -> plt.Figure:
    rows = filter_rows(
        df,
        budgets=[int(budget)],
        noises=[noise],
        al_methods=selected_al,
        noise_methods=selected_noise_methods,
        losses=selected_losses,
        families=selected_families,
    )

    default_title = (
        f"{friendly(noise)} — {budget} labeled samples"
    )
    title = title.strip() or default_title

    if graph_kind == "Linear + 1-NN accuracy bars":
        return plot_accuracy_bars(
            rows,
            title=title,
            horizontal=horizontal,
            sort_metric=accuracy_metric,
            top_n=top_n,
            show_values=show_values,
        )
    if graph_kind == "Single accuracy metric":
        return plot_single_metric_bars(
            rows,
            metric=accuracy_metric,
            title=title,
            horizontal=horizontal,
            top_n=top_n,
            show_values=show_values,
        )
    if graph_kind == "Cleaning outcomes":
        return plot_cleaning_outcomes(
            rows,
            title=title,
            top_n=top_n,
        )
    if graph_kind == "Fixed labels vs accuracy delta":
        return plot_fixed_vs_accuracy_delta(
            rows,
            metric=accuracy_metric,
            title=title,
            annotate=True,
        )
    if graph_kind == "Runtime":
        return plot_single_metric_bars(
            rows,
            metric="runtime_sec",
            title=title,
            horizontal=horizontal,
            top_n=top_n,
            show_values=show_values,
        )

    raise ValueError(f"Unsupported graph kind: {graph_kind}")


def launch_ui(
    df: pd.DataFrame,
    output_dir: Path,
    server_name: str,
    port: int,
) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the UI. Install it with:\n"
            "python -m pip install gradio pandas matplotlib"
        ) from exc

    budgets = sorted(
        int(value)
        for value in df["final_labeled_size"].dropna().unique()
    )
    noises = [
        value for value in NOISE_ORDER
        if value in set(df["noise"])
    ]
    al_methods = sorted(df["al_method"].dropna().unique())
    noise_methods = sorted(df["noise_method"].dropna().unique())
    losses = sorted(df["loss"].dropna().unique())
    families = sorted(df["method_family"].dropna().unique())

    temp_dir = Path(tempfile.mkdtemp(prefix="al_graph_preview_"))

    def preview(
        budget,
        noise,
        selected_al,
        selected_noise_methods,
        selected_losses,
        selected_families,
        graph_kind,
        accuracy_metric,
        top_n,
        horizontal,
        show_values,
        title,
    ):
        fig = build_custom_figure(
            df,
            budget=int(budget),
            noise=noise,
            selected_al=selected_al,
            selected_noise_methods=selected_noise_methods,
            selected_losses=selected_losses,
            selected_families=selected_families,
            graph_kind=graph_kind,
            accuracy_metric=accuracy_metric,
            top_n=int(top_n),
            horizontal=bool(horizontal),
            show_values=bool(show_values),
            title=title,
        )
        preview_path = temp_dir / "preview.png"
        fig.savefig(preview_path, bbox_inches="tight")
        plt.close(fig)
        return str(preview_path)

    def save_selected(
        budget,
        noise,
        selected_al,
        selected_noise_methods,
        selected_losses,
        selected_families,
        graph_kind,
        accuracy_metric,
        top_n,
        horizontal,
        show_values,
        title,
        filename,
    ):
        fig = build_custom_figure(
            df,
            budget=int(budget),
            noise=noise,
            selected_al=selected_al,
            selected_noise_methods=selected_noise_methods,
            selected_losses=selected_losses,
            selected_families=selected_families,
            graph_kind=graph_kind,
            accuracy_metric=accuracy_metric,
            top_n=int(top_n),
            horizontal=bool(horizontal),
            show_values=bool(show_values),
            title=title,
        )
        paths = save_figure(
            fig,
            output_dir,
            filename or f"{noise}_{budget}_{graph_kind}",
        )
        plt.close(fig)
        return [str(path) for path in paths]

    with gr.Blocks(title="Active-learning result graphs") as app:
        gr.Markdown(
            "# Active-learning result graphs\n"
            "Filter runs, preview the graph, then save PDF/SVG/PNG files."
        )

        with gr.Row():
            with gr.Column(scale=1):
                budget = gr.Dropdown(
                    budgets,
                    value=budgets[-1],
                    label="Final labeled size",
                )
                noise = gr.Dropdown(
                    noises,
                    value=noises[0],
                    label="Noise type",
                )
                graph_kind = gr.Dropdown(
                    [
                        "Linear + 1-NN accuracy bars",
                        "Single accuracy metric",
                        "Cleaning outcomes",
                        "Fixed labels vs accuracy delta",
                        "Runtime",
                    ],
                    value="Linear + 1-NN accuracy bars",
                    label="Graph",
                )
                accuracy_metric = gr.Dropdown(
                    ["knn_test_acc", "linear_test_acc"],
                    value="knn_test_acc",
                    label="Metric / sorting metric",
                )
                top_n = gr.Slider(
                    0,
                    60,
                    value=0,
                    step=1,
                    label="Top N (0 = all)",
                )
                horizontal = gr.Checkbox(
                    value=True,
                    label="Horizontal bars",
                )
                show_values = gr.Checkbox(
                    value=False,
                    label="Show values on bars",
                )

            with gr.Column(scale=2):
                selected_al = gr.CheckboxGroup(
                    al_methods,
                    value=al_methods,
                    label="AL methods",
                )
                selected_noise_methods = gr.CheckboxGroup(
                    noise_methods,
                    value=noise_methods,
                    label="Noise-handling methods",
                )
                selected_losses = gr.CheckboxGroup(
                    losses,
                    value=losses,
                    label="Losses",
                )
                selected_families = gr.CheckboxGroup(
                    families,
                    value=families,
                    label="Method families",
                )
                title = gr.Textbox(
                    label="Custom title",
                    placeholder="Leave empty for an automatic title",
                )
                filename = gr.Textbox(
                    label="Output filename without extension",
                    value="selected_graph",
                )

        with gr.Row():
            preview_button = gr.Button("Generate preview", variant="primary")
            save_button = gr.Button("Save for LaTeX")

        preview_image = gr.Image(
            label="Preview",
            type="filepath",
        )
        saved_files = gr.File(
            label="Saved PDF, SVG and PNG",
            file_count="multiple",
        )

        common_inputs = [
            budget,
            noise,
            selected_al,
            selected_noise_methods,
            selected_losses,
            selected_families,
            graph_kind,
            accuracy_metric,
            top_n,
            horizontal,
            show_values,
            title,
        ]

        preview_button.click(
            preview,
            inputs=common_inputs,
            outputs=preview_image,
        )
        save_button.click(
            save_selected,
            inputs=common_inputs + [filename],
            outputs=saved_files,
        )

    app.launch(
        server_name=server_name,
        server_port=port,
        share=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to original_features_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="al_result_graphs",
        help="Directory for PDF/SVG/PNG outputs",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive Gradio UI",
    )
    parser.add_argument(
        "--preset",
        choices=["initial-examination"],
        help="Generate a predefined graph set without launching the UI",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        default=7860,
        type=int,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    df = load_table(args.input)
    output_dir = Path(args.output_dir)

    print(
        f"Loaded {len(df)} unique completed configurations "
        f"from {args.input}"
    )

    if args.preset == "initial-examination":
        saved = generate_initial_examination(df, output_dir)
        print(f"Generated {len(saved)} files in {output_dir.resolve()}")
        for path in saved:
            print(path)
        return 0

    if args.ui:
        launch_ui(
            df,
            output_dir=output_dir,
            server_name=args.server_name,
            port=args.port,
        )
        return 0

    raise SystemExit(
        "Choose either --ui or --preset initial-examination."
    )


if __name__ == "__main__":
    raise SystemExit(main())
