"""Dry-run and submit independent audio AL configurations as a SLURM array."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import shlex
import subprocess
from pathlib import Path

from .cli import _load_yaml


def _set_path(config: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def build_sweep(config_path: Path, job_dir: Path, dependency: str | None = None) -> tuple[list[Path], str]:
    config = _load_yaml(config_path)
    sweep = config.pop("sweep", {})
    if not isinstance(sweep, dict):
        raise ValueError("sweep must be a YAML mapping")
    grid = sweep.get("grid", {})
    jobs = sweep.get("jobs")
    if jobs is not None:
        if grid:
            raise ValueError("configure either sweep.grid or sweep.jobs, not both")
        if not isinstance(jobs, list) or not jobs or any(not isinstance(item, dict) or not item for item in jobs):
            raise ValueError("sweep.jobs must be a non-empty list of dotted-key mappings")
        variants = [dict(item) for item in jobs]
    else:
        if not isinstance(grid, dict) or any(not isinstance(values, list) or not values for values in grid.values()):
            raise ValueError("sweep.grid values must be non-empty lists")
        keys = list(grid)
        combinations = itertools.product(*(grid[key] for key in keys)) if keys else [()]
        variants = [dict(zip(keys, values)) for values in combinations]
    job_dir = job_dir.expanduser().resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, resolved in enumerate(variants):
        variant = copy.deepcopy(config)
        for key, value in resolved.items():
            _set_path(variant, key, value)
        variant["sweep_job"] = {"index": index, "parameters": resolved, "scheduler": "slurm"}
        path = job_dir / f"job-{index:04d}.yaml"
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("sweep generation requires PyYAML") from exc
        path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        paths.append(path)
    slurm = sweep.get("slurm", {})
    slurm_options = {key: value for key, value in slurm.items()
                     if key not in {"activate", "working_directory"}}
    options = {"time": "24:00:00", "mem": "32G", "cpus-per-task": 4,
               "output": str(job_dir / "slurm-%A_%a.out"), **slurm_options}
    directives = [f"#SBATCH --array=0-{len(paths) - 1}"]
    if dependency:
        if not dependency.startswith(("afterok:", "afterany:", "afterfail:")):
            raise ValueError("dependency must use afterok:, afterany:, or afterfail:")
        directives.append(f"#SBATCH --dependency={dependency}")
    directives.extend(f"#SBATCH --{key}={value}" for key, value in options.items())
    case_lines = [f"  {index}) CONFIG={shlex.quote(str(path))} ;;" for index, path in enumerate(paths)]
    activate = slurm.get("activate")
    working_directory = slurm.get("working_directory")
    body = ["#!/bin/bash", *directives, "set -euo pipefail", "case \"$SLURM_ARRAY_TASK_ID\" in",
            *case_lines, "  *) echo 'Invalid array index' >&2; exit 2 ;;", "esac"]
    if working_directory:
        body.append(f"cd {shlex.quote(str(Path(working_directory).expanduser().resolve()))}")
    if activate:
        body.append(f"source {shlex.quote(str(Path(activate).expanduser().resolve()))}")
    body.extend(["echo \"Starting audio AL config: $CONFIG\"",
                 "python -m audio_embeddings run-al --config \"$CONFIG\""])
    return paths, "\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or submit an audio AL SLURM array")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--job-dir", required=True, type=Path,
                        help="external directory for generated configs and SLURM script")
    parser.add_argument("--submit", action="store_true", help="write the script and submit with sbatch")
    parser.add_argument("--dependency", help="SLURM dependency, for example afterok:12345")
    args = parser.parse_args()
    paths, script = build_sweep(args.config, args.job_dir, args.dependency)
    print(f"Planned {len(paths)} independent jobs:")
    for path in paths:
        print(f"  python -m audio_embeddings run-al --config {path}")
    print("--- SLURM array script ---")
    print(script, end="")
    if args.submit:
        script_path = args.job_dir.expanduser().resolve() / "audio-al-array.sh"
        script_path.write_text(script, encoding="utf-8", newline="\n")
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
