#!/usr/bin/env python3
"""Validate the final experiment CSV and regenerate report tables/figures.

The script intentionally depends only on Python's standard library and
ReportLab. Figures are written as vector PDFs and converted to PNG previews
with pdftoppm when it is available.
"""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "data" / "original_features_456_runs.csv"
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"

KEY_COLUMNS = [
    "final_labeled_size",
    "noise",
    "al_method",
    "noise_method",
    "loss",
]
STANDARD_METHODS = [
    "probcover",
    "maxherding",
    "typiclust",
    "coreset",
    "random",
    "margin",
]
NOISES = ["clean", "sym", "asym", "cifar100n"]
DETECTORS = ["crossval", "knn", "aum", "oracle"]

NAMES = {
    "probcover": "ProbCover",
    "maxherding": "MaxHerding",
    "typiclust": "TypiClust",
    "coreset": "CoreSet",
    "random": "Random",
    "margin": "Margin",
    "clean": "Clean",
    "sym": "Symmetric",
    "asym": "Asymmetric",
    "cifar100n": "CIFAR-100N",
    "crossval": "Cross-validation",
    "knn": "kNN",
    "aum": "AUM",
    "oracle": "Oracle",
    "label_cleaning": "Checkpoint",
    "temporal": "Temporal",
    "bootstrap_soft": "Bootstrap-soft",
    "bootstrap_hard": "Bootstrap-hard",
    "gce": "GCE",
}


def number(row: dict[str, str], column: str) -> float:
    return float(row[column])


def avg(values) -> float:
    values = list(values)
    if not values:
        raise AssertionError("Attempted to average an empty set")
    return mean(values)


def load_and_validate() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))

    assert len(raw) == 456, f"Expected 456 raw rows, found {len(raw)}"
    assert all(row["status"].strip().lower() == "completed" for row in raw)

    required = {
        *KEY_COLUMNS,
        "run_key",
        "linear_test_acc",
        "knn_test_acc",
        "fixed_labels_total",
        "correctly_fixed_labels_total",
        "clean_corrupted_total",
        "wrong_to_wrong_total",
    }
    missing = required - set(raw[0])
    assert not missing, f"Missing required CSV columns: {sorted(missing)}"

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[tuple(row[column] for column in KEY_COLUMNS)].append(row)

    duplicate_groups = {key: rows for key, rows in grouped.items() if len(rows) > 1}
    assert len(duplicate_groups) == 8, (
        f"Expected eight duplicated configurations, found {len(duplicate_groups)}"
    )
    for key, rows in duplicate_groups.items():
        budget, _noise, method, noise_method, loss = key
        assert budget in {"400", "800"}
        assert method == "maxherding"
        assert noise_method == "none"
        assert loss == "cross_entropy"
        assert len(rows) == 2
        for metric in ("knn_test_acc", "linear_test_acc"):
            assert math.isclose(number(rows[0], metric), number(rows[1], metric))

    # Prefer the general-grid copy of each repeated ordinary MaxHerding row.
    unique: list[dict[str, str]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: row["experiment_family"] != "many_cleaning_methods")
        unique.append(rows[0])

    assert len(unique) == 448, f"Expected 448 unique runs, found {len(unique)}"
    budgets = Counter(int(row["final_labeled_size"]) for row in unique)
    assert budgets == Counter({400: 224, 800: 224}), budgets
    for row in unique:
        assert row["knn_test_acc"] and row["linear_test_acc"]

    unique.sort(
        key=lambda row: (
            int(row["final_labeled_size"]),
            NOISES.index(row["noise"]),
            row["al_method"],
            row["noise_method"],
            row["loss"],
        )
    )
    return raw, unique


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_index(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    return {
        tuple(row[column] for column in KEY_COLUMNS): row
        for row in rows
    }


def analyze(rows: list[dict[str, str]]) -> dict[str, list[dict]]:
    baseline = [
        row for row in rows
        if row["al_method"] in STANDARD_METHODS
        and row["noise_method"] == "none"
        and row["loss"] == "cross_entropy"
    ]

    standard_summary: list[dict] = []
    for budget in (400, 800):
        for noise in NOISES:
            for method in STANDARD_METHODS:
                match = [
                    row for row in baseline
                    if int(row["final_labeled_size"]) == budget
                    and row["noise"] == noise
                    and row["al_method"] == method
                ]
                assert len(match) == 1
                standard_summary.append({
                    "budget": budget,
                    "noise": noise,
                    "al_method": method,
                    "knn_test_acc": number(match[0], "knn_test_acc"),
                    "linear_test_acc": number(match[0], "linear_test_acc"),
                })

    standard_means: list[dict] = []
    for budget in (400, 800):
        for method in STANDARD_METHODS:
            selected = [
                row for row in standard_summary
                if row["budget"] == budget and row["al_method"] == method
            ]
            standard_means.append({
                "budget": budget,
                "al_method": method,
                "mean_knn_test_acc": avg(row["knn_test_acc"] for row in selected),
                "mean_linear_test_acc": avg(row["linear_test_acc"] for row in selected),
            })

    lookup = row_index(rows)
    loss_rows: list[dict] = []
    loss_selectors = [method for method in STANDARD_METHODS if method != "margin"]
    for budget in (400, 800):
        for loss in ("bootstrap_soft", "bootstrap_hard", "gce"):
            deltas, absolute = [], []
            for noise in NOISES:
                for method in loss_selectors:
                    base = lookup[(str(budget), noise, method, "none", "cross_entropy")]
                    variant = lookup[(str(budget), noise, method, "none", loss)]
                    deltas.append(number(variant, "linear_test_acc") - number(base, "linear_test_acc"))
                    absolute.append(number(variant, "linear_test_acc"))
            assert len(deltas) == 20
            loss_rows.append({
                "budget": budget,
                "loss": loss,
                "mean_linear_delta_vs_ce": avg(deltas),
                "mean_linear_test_acc": avg(absolute),
                "matched_comparisons": len(deltas),
            })

    cleaning_rows: list[dict] = []
    for method in ("label_cleaning", "crossval", "temporal", "aum"):
        d_knn, d_linear, selected = [], [], []
        for noise in NOISES:
            for al_method in STANDARD_METHODS:
                base = lookup[("800", noise, al_method, "none", "cross_entropy")]
                variant = lookup[("800", noise, al_method, method, "cross_entropy")]
                d_knn.append(number(variant, "knn_test_acc") - number(base, "knn_test_acc"))
                d_linear.append(number(variant, "linear_test_acc") - number(base, "linear_test_acc"))
                selected.append(variant)
        assert len(selected) == 24
        cleaning_rows.append({
            "noise_method": method,
            "mean_knn_delta": avg(d_knn),
            "mean_linear_delta": avg(d_linear),
            "label_change_operations": round(sum(number(row, "fixed_labels_total") for row in selected)),
            "correctly_fixed": round(sum(number(row, "correctly_fixed_labels_total") for row in selected)),
            "clean_corrupted": round(sum(number(row, "clean_corrupted_total") for row in selected)),
            "wrong_to_wrong": round(sum(number(row, "wrong_to_wrong_total") for row in selected)),
            "matched_comparisons": len(selected),
        })

    npc_rows: list[dict] = []
    for selector in ("probcover", "maxherding"):
        prefix = f"npc_{selector}_"
        for budget in (400, 800):
            for noise in NOISES:
                base = lookup[(str(budget), noise, selector, "none", "cross_entropy")]
                for detector in DETECTORS:
                    variant = lookup[(
                        str(budget), noise, f"{prefix}{detector}", "none", "cross_entropy"
                    )]
                    value = number(variant, "knn_test_acc")
                    npc_rows.append({
                        "selector": selector,
                        "budget": budget,
                        "noise": noise,
                        "detector": detector,
                        "baseline_knn_test_acc": number(base, "knn_test_acc"),
                        "npc_knn_test_acc": value,
                        "knn_delta": value - number(base, "knn_test_acc"),
                    })

    npc_means: list[dict] = []
    for selector in ("probcover", "maxherding"):
        for budget in (400, 800):
            for detector in DETECTORS:
                selected = [
                    row for row in npc_rows
                    if row["selector"] == selector
                    and row["budget"] == budget
                    and row["detector"] == detector
                ]
                npc_means.append({
                    "selector": selector,
                    "budget": budget,
                    "detector": detector,
                    "mean_knn_delta": avg(row["knn_delta"] for row in selected),
                })

    return {
        "standard": standard_summary,
        "standard_means": standard_means,
        "losses": loss_rows,
        "cleaning": cleaning_rows,
        "npc": npc_rows,
        "npc_means": npc_means,
    }


def save_summaries(results: dict[str, list[dict]], unique: list[dict[str, str]]) -> None:
    write_csv(
        DATA_DIR / "deduplicated_448_runs.csv",
        unique,
        list(unique[0].keys()),
    )
    for name, rows in results.items():
        write_csv(DATA_DIR / f"{name}_summary.csv", rows, list(rows[0].keys()))


def escape_tex(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def tex_tables(results: dict[str, list[dict]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    std = results["standard"]
    means = results["standard_means"]
    mean_lookup = {(row["budget"], row["al_method"]): row for row in means}
    std_lookup = {
        (row["budget"], row["noise"], row["al_method"]): row for row in std
    }
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method & Mean 400 & Mean 800 & Sym. 800 & Asym. 800 \\",
        r"\midrule",
    ]
    for method in STANDARD_METHODS:
        lines.append(
            f"{NAMES[method]} & "
            f"{mean_lookup[(400, method)]['mean_knn_test_acc']:.2f} & "
            f"{mean_lookup[(800, method)]['mean_knn_test_acc']:.2f} & "
            f"{std_lookup[(800, 'sym', method)]['knn_test_acc']:.2f} & "
            f"{std_lookup[(800, 'asym', method)]['knn_test_acc']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "standard_al.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Loss & $\Delta$ linear, 400 & $\Delta$ linear, 800 & Linear, 800 \\",
        r"\midrule",
    ]
    loss_lookup = {(row["budget"], row["loss"]): row for row in results["losses"]}
    for loss in ("bootstrap_soft", "bootstrap_hard", "gce"):
        lines.append(
            f"{NAMES[loss]} & "
            f"{loss_lookup[(400, loss)]['mean_linear_delta_vs_ce']:+.2f} & "
            f"{loss_lookup[(800, loss)]['mean_linear_delta_vs_ce']:+.2f} & "
            f"{loss_lookup[(800, loss)]['mean_linear_test_acc']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "losses.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & $\Delta$1-NN & $\Delta$linear & Changes & Correct & Clean$\to$wrong \\",
        r"\midrule",
    ]
    for row in results["cleaning"]:
        lines.append(
            f"{NAMES[row['noise_method']]} & {row['mean_knn_delta']:+.2f} & "
            f"{row['mean_linear_delta']:+.2f} & {row['label_change_operations']} & "
            f"{row['correctly_fixed']} & {row['clean_corrupted']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "cleaning.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Selector & Budget & Cross-val & kNN & AUM & Oracle \\", r"\midrule",
    ]
    npc_lookup = {
        (row["selector"], row["budget"], row["detector"]): row
        for row in results["npc_means"]
    }
    for selector in ("probcover", "maxherding"):
        for budget in (400, 800):
            values = [npc_lookup[(selector, budget, detector)]["mean_knn_delta"] for detector in DETECTORS]
            lines.append(
                f"{NAMES[selector]} & {budget} & "
                + " & ".join(f"{value:+.2f}" for value in values)
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "npc_means.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Frequently cited values are macros so prose cannot drift from the CSV.
    npc = {
        (row["selector"], row["budget"], row["noise"], row["detector"]): row
        for row in results["npc"]
    }
    stdv = {
        (row["budget"], row["noise"], row["al_method"]): row
        for row in results["standard"]
    }
    macros = {
        "RawRuns": "456",
        "DuplicateRuns": "8",
        "UniqueRuns": "448",
        "RunsPerBudget": "224",
        "PCSymKNNDelta": f"{npc[('probcover', 800, 'sym', 'knn')]['knn_delta']:+.2f}",
        "PCSymCVDelta": f"{npc[('probcover', 800, 'sym', 'crossval')]['knn_delta']:+.2f}",
        "PCSymAUMDelta": f"{npc[('probcover', 800, 'sym', 'aum')]['knn_delta']:+.2f}",
        "MHSymLead": f"{stdv[(800, 'sym', 'maxherding')]['knn_test_acc'] - stdv[(800, 'sym', 'probcover')]['knn_test_acc']:.2f}",
        "PCAsymLead": f"{stdv[(800, 'asym', 'probcover')]['knn_test_acc'] - stdv[(800, 'asym', 'maxherding')]['knn_test_acc']:.2f}",
        "GCEFourDelta": f"{loss_lookup[(400, 'gce')]['mean_linear_delta_vs_ce']:+.2f}",
        "GCEEightDelta": f"{loss_lookup[(800, 'gce')]['mean_linear_delta_vs_ce']:+.2f}",
    }
    text = "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()) + "\n"
    (TABLE_DIR / "analysis_values.tex").write_text(text, encoding="utf-8")


def draw_text(c: canvas.Canvas, x: float, y: float, text: str, size=8, bold=False, align="left") -> None:
    c.setFillColor(black)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    if align == "center":
        c.drawCentredString(x, y, text)
    elif align == "right":
        c.drawRightString(x, y, text)
    else:
        c.drawString(x, y, text)


def standard_figure(rows: list[dict]) -> Path:
    path = FIG_DIR / "standard_al_by_noise.pdf"
    width, height = 820, 470
    c = canvas.Canvas(str(path), pagesize=(width, height))
    draw_text(c, width / 2, height - 25, "Standard AL performance by noise type", 15, True, "center")
    draw_text(c, width / 2, height - 41, "1-NN test accuracy; paired bars show 400 and 800 labeled samples", 9, False, "center")
    colors = {400: HexColor("#6B9AC4"), 800: HexColor("#E07A5F")}
    lookup = {(r["budget"], r["noise"], r["al_method"]): r["knn_test_acc"] for r in rows}
    panel_w, panel_h = 380, 160
    for panel, noise in enumerate(NOISES):
        col, row = panel % 2, panel // 2
        x0 = 55 + col * 400
        y0 = 245 - row * 195
        draw_text(c, x0 + panel_w / 2, y0 + panel_h + 3, NAMES[noise], 10, True, "center")
        c.setStrokeColor(HexColor("#BBBBBB")); c.line(x0, y0, x0, y0 + panel_h); c.line(x0, y0, x0 + panel_w, y0)
        max_y = 50
        for tick in range(0, 51, 10):
            yy = y0 + panel_h * tick / max_y
            c.setStrokeColor(HexColor("#E5E5E5")); c.line(x0, yy, x0 + panel_w, yy)
            draw_text(c, x0 - 6, yy - 3, str(tick), 7, False, "right")
        group_w = panel_w / len(STANDARD_METHODS)
        for i, method in enumerate(STANDARD_METHODS):
            center = x0 + (i + 0.5) * group_w
            for budget, offset in ((400, -8), (800, 8)):
                value = lookup[(budget, noise, method)]
                bar_h = panel_h * value / max_y
                c.setFillColor(colors[budget]); c.rect(center + offset - 7, y0, 13, bar_h, fill=1, stroke=0)
            draw_text(c, center, y0 - 13, NAMES[method].replace("Herding", "Herd."), 6.3, False, "center")
    c.setFillColor(colors[400]); c.rect(330, 8, 12, 8, fill=1, stroke=0); draw_text(c, 347, 8, "400 labels", 8)
    c.setFillColor(colors[800]); c.rect(420, 8, 12, 8, fill=1, stroke=0); draw_text(c, 437, 8, "800 labels", 8)
    c.save()
    return path


def loss_figure(rows: list[dict]) -> Path:
    path = FIG_DIR / "robust_loss_deltas.pdf"
    width, height = 620, 330
    c = canvas.Canvas(str(path), pagesize=(width, height))
    draw_text(c, width / 2, height - 25, "Robust losses compared with cross-entropy", 15, True, "center")
    draw_text(c, width / 2, height - 42, "Mean change in linear test accuracy over 20 matched settings", 9, False, "center")
    lookup = {(r["budget"], r["loss"]): r["mean_linear_delta_vs_ce"] for r in rows}
    x0, y0, plot_w, plot_h = 70, 65, 500, 210
    min_y, max_y = -12, 2
    for tick in range(-12, 3, 2):
        yy = y0 + plot_h * (tick - min_y) / (max_y - min_y)
        c.setStrokeColor(HexColor("#E2E2E2")); c.line(x0, yy, x0 + plot_w, yy)
        draw_text(c, x0 - 8, yy - 3, str(tick), 8, False, "right")
    zero_y = y0 + plot_h * (0 - min_y) / (max_y - min_y)
    c.setStrokeColor(black); c.line(x0, zero_y, x0 + plot_w, zero_y)
    colors = {400: HexColor("#6B9AC4"), 800: HexColor("#E07A5F")}
    losses = ["bootstrap_soft", "bootstrap_hard", "gce"]
    for i, loss in enumerate(losses):
        center = x0 + (i + 0.5) * plot_w / len(losses)
        for budget, offset in ((400, -17), (800, 17)):
            value = lookup[(budget, loss)]
            yy = y0 + plot_h * (value - min_y) / (max_y - min_y)
            bottom, bar_h = min(zero_y, yy), abs(yy - zero_y)
            c.setFillColor(colors[budget]); c.rect(center + offset - 14, bottom, 27, bar_h, fill=1, stroke=0)
            draw_text(c, center + offset, yy + (4 if value >= 0 else -12), f"{value:+.2f}", 8, True, "center")
        draw_text(c, center, y0 - 18, NAMES[loss], 8.5, False, "center")
    draw_text(c, 25, y0 + plot_h / 2, "Accuracy-point change", 8, False, "center")
    c.setFillColor(colors[400]); c.rect(225, 20, 12, 8, fill=1, stroke=0); draw_text(c, 242, 20, "400 labels", 8)
    c.setFillColor(colors[800]); c.rect(330, 20, 12, 8, fill=1, stroke=0); draw_text(c, 347, 20, "800 labels", 8)
    c.save()
    return path


def heat_color(value: float) -> Color:
    limit = 4.5
    t = min(abs(value) / limit, 1.0)
    if value >= 0:
        return Color(1 - 0.58 * t, 1 - 0.18 * t, 1 - 0.55 * t)
    return Color(1 - 0.10 * t, 1 - 0.45 * t, 1 - 0.45 * t)


def npc_figure(rows: list[dict]) -> Path:
    path = FIG_DIR / "npc_deltas_by_noise.pdf"
    width, height = 780, 520
    c = canvas.Canvas(str(path), pagesize=(width, height))
    draw_text(c, width / 2, height - 24, "Noise-aware coverage: change from the ordinary selector", 14, True, "center")
    draw_text(c, width / 2, height - 41, "1-NN test accuracy points; blue is better and red is worse", 9, False, "center")
    lookup = {
        (r["selector"], r["budget"], r["noise"], r["detector"]): r["knn_delta"]
        for r in rows
    }
    panel_w, panel_h = 350, 205
    for panel, (selector, budget) in enumerate([
        ("probcover", 400), ("probcover", 800),
        ("maxherding", 400), ("maxherding", 800),
    ]):
        col, row = panel % 2, panel // 2
        x0 = 55 + col * 375
        y0 = 255 - row * 225
        draw_text(c, x0 + panel_w / 2, y0 + panel_h + 3, f"{NAMES[selector]}, {budget} labels", 10, True, "center")
        label_w, header_h = 90, 24
        cell_w = (panel_w - label_w) / 4
        cell_h = (panel_h - header_h) / 4
        for j, noise in enumerate(NOISES):
            draw_text(c, x0 + label_w + (j + 0.5) * cell_w, y0 + panel_h - 16, NAMES[noise], 7.2, True, "center")
        for i, detector in enumerate(DETECTORS):
            yy = y0 + panel_h - header_h - (i + 1) * cell_h
            draw_text(c, x0 + label_w - 7, yy + cell_h / 2 - 3, NAMES[detector], 7.5, False, "right")
            for j, noise in enumerate(NOISES):
                value = lookup[(selector, budget, noise, detector)]
                xx = x0 + label_w + j * cell_w
                c.setFillColor(heat_color(value)); c.setStrokeColor(white)
                c.rect(xx, yy, cell_w, cell_h, fill=1, stroke=1)
                draw_text(c, xx + cell_w / 2, yy + cell_h / 2 - 3, f"{value:+.2f}", 8.5, True, "center")
    c.save()
    return path


def make_png_previews(pdf_paths: list[Path]) -> None:
    candidates = [
        shutil.which("pdftoppm"),
        "/Users/tomno/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm",
    ]
    executable = next((path for path in candidates if path and Path(path).exists()), None)
    if not executable:
        print("pdftoppm not found; PDF figures were generated without PNG previews")
        return
    for pdf in pdf_paths:
        prefix = pdf.with_suffix("")
        subprocess.run(
            [executable, "-png", "-singlefile", "-r", "180", str(pdf), str(prefix)],
            check=True,
        )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw, unique = load_and_validate()
    results = analyze(unique)
    save_summaries(results, unique)
    tex_tables(results)
    figures = [
        standard_figure(results["standard"]),
        loss_figure(results["losses"]),
        npc_figure(results["npc"]),
    ]
    make_png_previews(figures)
    print(f"Validated {len(raw)} rows -> {len(unique)} unique completed runs")
    print("Budget counts: 400=224, 800=224")
    print("Generated summary CSVs, LaTeX tables, and three vector figures")


if __name__ == "__main__":
    main()
