#!/usr/bin/env python3
import ast
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/cs/labs/daphna/ido.dotan/projects/daphna_lab")
VIS_DIR = ROOT / "visual_outputs"
SUMMARY_DIR = ROOT / "results_summaries"
BASE = ROOT / "outputs" / "maxherding" / "low_budget_k_sweep"


@dataclass
class RunRecord:
    dataset: str
    seed: int
    mode: str
    k: int | None
    curve: list[float]
    final_test: float
    log_path: Path


def extract_curve(log_path: Path):
    text = log_path.read_text()
    current_matches = re.findall(r"Current accuracy values:\s*(\[[^\]]*\])", text)
    if not current_matches:
        raise ValueError(f"Could not find episode accuracy list in {log_path}")
    curve = list(ast.literal_eval(current_matches[-1]))

    final_matches = re.findall(r"Final Test Accuracy:\s*([0-9.]+)%", text)
    if not final_matches:
        raise ValueError(f"Could not find final test accuracy in {log_path}")
    final_test = float(final_matches[-1])
    curve.append(final_test)
    return curve, final_test


def parse_run(log_path: Path):
    text = log_path.read_text()
    cmd_match = re.search(r"Command:\s+python train_al\.py (.*)", text)
    if not cmd_match:
        raise ValueError(f"Could not find command in {log_path}")
    cmd = cmd_match.group(1)

    dataset_match = re.search(r"--cfg \.\./configs/(cifar10|cifar100)/", cmd)
    seed_match = re.search(r"--seed (\d+)", cmd)
    if not dataset_match or not seed_match:
        raise ValueError(f"Could not parse dataset/seed from {log_path}")

    dataset = dataset_match.group(1)
    seed = int(seed_match.group(1))
    k_match = re.search(r"--local_sigma_k (\d+)", cmd)
    mode = "local_sigma" if k_match else "global_sigma"
    k = int(k_match.group(1)) if k_match else None

    curve, final_test = extract_curve(log_path)
    return RunRecord(
        dataset=dataset,
        seed=seed,
        mode=mode,
        k=k,
        curve=curve,
        final_test=final_test,
        log_path=log_path,
    )


def latest_runs():
    records: dict[tuple[str, int, str, int | None], RunRecord] = {}
    for log_path in BASE.glob("**/*.txt"):
        try:
            record = parse_run(log_path)
        except Exception:
            continue
        key = (record.dataset, record.seed, record.mode, record.k)
        prev = records.get(key)
        if prev is None or log_path.stat().st_mtime > prev.log_path.stat().st_mtime:
            records[key] = record
    return records


def xvals_for(dataset: str):
    if dataset == "cifar10":
        return list(range(10, 101, 10))
    if dataset == "cifar100":
        return list(range(200, 3001, 200))
    raise ValueError(dataset)


def plot_series(dataset: str, seed: int, selected_ks: list[int], out_name: str, title: str, tight: bool):
    records = latest_runs()
    global_record = records[(dataset, seed, "global_sigma", None)]
    series = [("global sigma", global_record.curve)]
    for k in selected_ks:
        key = (dataset, seed, "local_sigma", k)
        if key in records:
            series.append((f"local sigma K={k}", records[key].curve))

    xvals = xvals_for(dataset)
    all_vals = [y for _, curve in series for y in curve]
    ymin = min(all_vals)
    ymax = max(all_vals)
    margin = max(0.08, (ymax - ymin) * (0.05 if tight else 0.12))

    plt.figure(figsize=(10, 6))
    for label, curve in series:
        plt.plot(xvals, curve, marker="o", linewidth=2, markersize=5, label=label)

    plt.title(title)
    plt.xlabel("Labeled set size")
    plt.ylabel("Test accuracy (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9)
    plt.ylim(ymin - margin, ymax + margin)
    plt.tight_layout()

    VIS_DIR.mkdir(exist_ok=True)
    out_path = VIS_DIR / out_name
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def write_dataset_summary(dataset: str, out_name: str):
    records = latest_runs()
    selected = [r for r in records.values() if r.dataset == dataset]
    selected.sort(key=lambda r: (r.seed, 0 if r.mode == "global_sigma" else 1, -1 if r.k is None else r.k))

    SUMMARY_DIR.mkdir(exist_ok=True)
    out_path = SUMMARY_DIR / out_name
    xvals = xvals_for(dataset)

    lines = []
    lines.append(f"MaxHerding low-budget results summary for {dataset.upper()}")
    lines.append("")
    lines.append("Episode x-axis:")
    lines.append(str(xvals))
    lines.append("")

    current_seed = None
    for record in selected:
        if record.seed != current_seed:
            current_seed = record.seed
            lines.append(f"=== Seed {current_seed} ===")
        if record.mode == "global_sigma":
            lines.append("global_sigma")
        else:
            lines.append(f"local_sigma | K={record.k}")
        lines.append(f"final_test={record.final_test:.2f}")
        lines.append(f"curve={record.curve}")
        lines.append(f"log={record.log_path}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path


def main():
    generated = []
    generated.append(
        plot_series(
            dataset="cifar10",
            seed=1,
            selected_ks=[3, 15, 30, 480, 1440, 3840],
            out_name="maxherding_cifar10_k_comparison_seed1.png",
            title="MaxHerding on CIFAR10: Representative K Values vs Global Sigma",
            tight=False,
        )
    )
    generated.append(
        plot_series(
            dataset="cifar100",
            seed=1,
            selected_ks=[7, 15, 120, 960, 3840, 7680],
            out_name="maxherding_cifar100_k_comparison_seed1.png",
            title="MaxHerding on CIFAR100: Representative K Values vs Global Sigma",
            tight=False,
        )
    )
    generated.append(
        plot_series(
            dataset="cifar10",
            seed=1,
            selected_ks=[480, 960, 1440, 1920, 2880, 3840],
            out_name="maxherding_cifar10_top_k_comparison_seed1.png",
            title="MaxHerding on CIFAR10: Top Large-K Curves vs Global Sigma",
            tight=True,
        )
    )
    generated.append(
        plot_series(
            dataset="cifar100",
            seed=1,
            selected_ks=[120, 960, 2880, 3840, 5760, 7680],
            out_name="maxherding_cifar100_top_k_comparison_seed1.png",
            title="MaxHerding on CIFAR100: Top Large-K Curves vs Global Sigma",
            tight=True,
        )
    )
    generated.append(write_dataset_summary("cifar10", "maxherding_cifar10_all_results.txt"))
    generated.append(write_dataset_summary("cifar100", "maxherding_cifar100_all_results.txt"))

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
