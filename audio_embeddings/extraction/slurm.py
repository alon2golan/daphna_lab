"""Generate SLURM array scripts for independent compact-audio shards."""
from __future__ import annotations

import json
import shlex
from pathlib import Path


def build_extraction_array(manifest: Path, audio_index: Path, model: str, checkpoint: Path,
                           output: Path, job_dir: Path, *, batch_size: int = 16,
                           device: str = "cpu", checkpoint_metadata: Path | None = None,
                           model_source: Path | None = None, temporary_root: Path | None = None,
                           dependency: str | None = None, activate: Path | None = None,
                           working_directory: Path | None = None, time_limit: str = "24:00:00",
                           memory: str = "32G", cpus: int = 4, partition: str | None = None,
                           gres: str | None = None) -> tuple[list[str], str]:
    audio_index = Path(audio_index).expanduser().resolve()
    index = json.loads(audio_index.read_text(encoding="utf-8"))
    archive_ids = [item["archive_id"] for item in index.get("archives", [])]
    if not archive_ids:
        raise ValueError("audio index contains no archives")
    job_dir = Path(job_dir).expanduser().resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    directives = [f"#SBATCH --array=0-{len(archive_ids) - 1}",
                  f"#SBATCH --time={time_limit}", f"#SBATCH --mem={memory}",
                  f"#SBATCH --cpus-per-task={int(cpus)}",
                  f"#SBATCH --output={job_dir / 'extract-%A_%a.out'}"]
    if partition:
        directives.append(f"#SBATCH --partition={partition}")
    if gres:
        directives.append(f"#SBATCH --gres={gres}")
    if dependency:
        if not dependency.startswith(("afterok:", "afterany:", "afterfail:")):
            raise ValueError("dependency must use afterok:, afterany:, or afterfail:")
        directives.append(f"#SBATCH --dependency={dependency}")
    cases = [f"  {i}) ARCHIVE_ID={shlex.quote(archive_id)} ;;" for i, archive_id in enumerate(archive_ids)]
    body = ["#!/bin/bash", *directives, "set -euo pipefail",
            'case "${SLURM_ARRAY_TASK_ID:?}" in', *cases,
            "  *) echo 'Invalid SLURM array index' >&2; exit 2 ;;", "esac"]
    if working_directory:
        body.append(f"cd {shlex.quote(str(Path(working_directory).expanduser().resolve()))}")
    if activate:
        body.append(f"source {shlex.quote(str(Path(activate).expanduser().resolve()))}")
    command = ["python", "-m", "audio_embeddings", "extract", "--manifest", str(Path(manifest).expanduser().resolve()),
               "--audio-index", str(audio_index), "--archive-id", '"$ARCHIVE_ID"', "--model", model,
               "--checkpoint", str(Path(checkpoint).expanduser().resolve()), "--output", str(Path(output).expanduser().resolve()),
               "--batch-size", str(int(batch_size)), "--device", device]
    for flag, value in (("--checkpoint-metadata", checkpoint_metadata), ("--model-source", model_source),
                        ("--temporary-root", temporary_root)):
        if value is not None:
            command.extend([flag, str(Path(value).expanduser().resolve())])
    shell_command = " ".join(shlex.quote(item) if item != '"$ARCHIVE_ID"' else item for item in command)
    body.extend(['echo "Extracting archive $ARCHIVE_ID"', shell_command])
    return archive_ids, "\n".join(body) + "\n"
