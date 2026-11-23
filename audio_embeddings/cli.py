"""Command line entry point: python -m audio_embeddings.cli ..."""

from __future__ import annotations

import argparse
from pathlib import Path

from .extraction.extractor import extract, finalize_compact_extraction
from .data.compact_utils import pack_audio_batches
from .data.manifest import build_manifest
from .training.trainer_al import run_experiment


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("running configured experiments requires PyYAML; install audio_embeddings/requirements.txt") from exc
    path = path.expanduser().resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration file must contain a YAML mapping")
    base_name = value.pop("base_config", None)
    if base_name is None:
        return value
    base_path = (path.parent / base_name).resolve()
    if path == base_path:
        raise ValueError("base_config cannot refer to the config itself")
    base = _load_yaml(base_path)

    def merge(left: dict, right: dict) -> dict:
        merged = dict(left)
        for key, item in right.items():
            if isinstance(item, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge(merged[key], item)
            else:
                merged[key] = item
        return merged

    return merge(base, value)


def main() -> None:
    parser = argparse.ArgumentParser(prog="audio-embeddings")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("manifest", help="build a dataset JSONL manifest")
    build.add_argument("--dataset", required=True, choices=["fsd50k", "urbansound8k", "esc50", "as20k", "dcase", "dcase2016", "dcase-2016", "dcase2016_1", "dcase20161", "dcase-2016-1", "dcase2016_4", "dcase20164", "dcase-2016-4", "ks1", "ks2", "er", "iemocap"])
    build.add_argument("--root", type=Path, required=True, help="dataset files root")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--metadata", type=Path, help="metadata directory/file (ER requires a CSV)")
    build.add_argument("--version")
    build.add_argument("--corpus", help="ER corpus name, such as IEMOCAP")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--folds", type=int, default=5)

    run = commands.add_parser("extract", help="extract frozen embeddings")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--model", choices=["beats", "clap", "panns", "ssast"], required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="parent directory for immutable run output")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--device", default="cpu", help="torch device, such as cpu or cuda:0")
    run.add_argument("--model-source", type=Path, help="official source directory; BEATs expects unilm/beats, SSAST expects ssast/src")
    run.add_argument("--checkpoint-metadata", type=Path, help="provenance JSON; required for SSAST")
    run.add_argument("--audio-index", type=Path, help="checksummed compact-audio index for one-shard extraction")
    run.add_argument("--archive-id", help="archive ID to extract when --audio-index is provided")
    run.add_argument("--temporary-root", type=Path, help="scratch directory for expanded audio")

    pack = commands.add_parser("pack-audio", help="package manifest audio into compressed batches")
    pack.add_argument("--manifest", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--batch-size", type=int, default=500)
    pack.add_argument("--max-uncompressed-bytes", type=int)

    finalize = commands.add_parser("finalize-extract", help="validate and combine compact-audio extraction shards")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--audio-index", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True, help="extraction output parent directory")

    extract_array = commands.add_parser("extract-array", help="generate or submit a SLURM extraction array")
    extract_array.add_argument("--manifest", type=Path, required=True)
    extract_array.add_argument("--audio-index", type=Path, required=True)
    extract_array.add_argument("--model", choices=["beats", "ssast"], required=True)
    extract_array.add_argument("--checkpoint", type=Path, required=True)
    extract_array.add_argument("--checkpoint-metadata", type=Path)
    extract_array.add_argument("--model-source", type=Path)
    extract_array.add_argument("--output", type=Path, required=True)
    extract_array.add_argument("--job-dir", type=Path, required=True)
    extract_array.add_argument("--batch-size", type=int, default=16)
    extract_array.add_argument("--device", default="cpu")
    extract_array.add_argument("--temporary-root", type=Path)
    extract_array.add_argument("--activate", type=Path)
    extract_array.add_argument("--working-directory", type=Path)
    extract_array.add_argument("--time", default="24:00:00")
    extract_array.add_argument("--mem", default="32G")
    extract_array.add_argument("--cpus-per-task", type=int, default=4)
    extract_array.add_argument("--partition")
    extract_array.add_argument("--gres", help="SLURM generic resources, for example gpu:1")
    extract_array.add_argument("--dependency")
    extract_array.add_argument("--submit", action="store_true")

    al = commands.add_parser("run-al", help="run a YAML-configured audio active-learning experiment")
    al.add_argument("--config", type=Path, required=True)
    al.add_argument("--device", help="override config device (cpu, cuda, or cuda:<index>)")

    sweep = commands.add_parser("sweep", help="dry-run or submit an independent SLURM job array")
    sweep.add_argument("--config", type=Path, required=True)
    sweep.add_argument("--job-dir", type=Path, required=True, help="external directory for generated sweep files")
    sweep.add_argument("--submit", action="store_true")
    sweep.add_argument("--dependency", help="SLURM dependency, for example afterok:12345")

    args = parser.parse_args()
    if args.command == "manifest":
        count = build_manifest(args.dataset, args.root, args.output, version=args.version,
                               metadata=args.metadata, corpus=args.corpus, seed=args.seed, folds=args.folds)
        print(f"Wrote {count} samples to {args.output}")
    elif args.command == "extract":
        print(extract(args.manifest, args.model, args.checkpoint, args.output, batch_size=args.batch_size,
                      device=args.device, model_source=args.model_source,
                      checkpoint_metadata=args.checkpoint_metadata, audio_index=args.audio_index,
                      archive_id=args.archive_id, temporary_root=args.temporary_root))
    elif args.command == "pack-audio":
        print(pack_audio_batches(args.manifest, args.output, batch_size=args.batch_size,
                                 max_uncompressed_bytes=args.max_uncompressed_bytes))
    elif args.command == "finalize-extract":
        print(finalize_compact_extraction(args.manifest, args.audio_index, args.output))
    elif args.command == "extract-array":
        from .extraction.slurm import build_extraction_array
        import subprocess
        archive_ids, script = build_extraction_array(
            args.manifest, args.audio_index, args.model, args.checkpoint, args.output, args.job_dir,
            batch_size=args.batch_size, device=args.device, checkpoint_metadata=args.checkpoint_metadata,
            model_source=args.model_source, temporary_root=args.temporary_root, dependency=args.dependency,
            activate=args.activate, working_directory=args.working_directory,
            time_limit=args.time, memory=args.mem, cpus=args.cpus_per_task,
            partition=args.partition, gres=args.gres)
        script_path = args.job_dir.expanduser().resolve() / "audio-extract-array.sh"
        script_path.write_text(script, encoding="utf-8", newline="\n")
        print(f"Prepared {len(archive_ids)} extraction tasks; script: {script_path}")
        if args.submit:
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)
            print(result.stdout.strip())
    elif args.command == "run-al":
        config = _load_yaml(args.config)
        if args.device is not None:
            config["device"] = args.device
        print(run_experiment(config))
    else:
        from .sweep import build_sweep
        paths, script = build_sweep(args.config, args.job_dir, args.dependency)
        print(f"Planned {len(paths)} independent jobs:")
        for path in paths:
            print(f"  python -m audio_embeddings run-al --config {path}")
        print(script, end="")
        if args.submit:
            import subprocess
            script_path = args.job_dir.expanduser().resolve() / "audio-al-array.sh"
            script_path.write_text(script, encoding="utf-8", newline="\n")
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)
            print(result.stdout.strip())


if __name__ == "__main__":
    main()
