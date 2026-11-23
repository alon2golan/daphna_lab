"""Batched, resumable extraction with immutable run configuration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..data.manifest import manifest_digest, read_manifest
from ..models.adapters import create_model
from ..data.audio_io import load_audio, slice_audio, window_audio
from ..data.device import resolve_device
from ..data.compact_utils import cleanup_expanded_batch, extract_audio_batch


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(run_dir: Path, vectors: dict[str, np.ndarray], rows: dict[str, dict[str, Any]], failures: list[dict[str, str]]) -> None:
    ids = list(vectors)
    matrix = np.stack([vectors[key] for key in ids]).astype(np.float32) if ids else np.empty((0, 0), dtype=np.float32)
    fd, temp_name = tempfile.mkstemp(prefix=".embeddings.", suffix=".npz", dir=run_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, sample_ids=np.asarray(ids), embeddings=matrix)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, run_dir / "embeddings.npz")
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    sample_text = "".join(json.dumps(rows[key], ensure_ascii=False) + "\n" for key in ids)
    _atomic_text(run_dir / "samples.jsonl", sample_text)
    _atomic_text(run_dir / "failures.json", json.dumps(failures, indent=2))


def _atomic_text(path: Path, value: str) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def extract(manifest: Path, model_name: str, checkpoint: Path, output: Path, batch_size: int = 16,
            device: str = "cpu", model_source: Path | None = None,
            checkpoint_metadata: Path | None = None, preprocessing: dict[str, Any] | None = None,
            audio_index: Path | None = None, archive_id: str | None = None,
            temporary_root: Path | None = None) -> Path:
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = str(resolve_device(device))
    records = read_manifest(manifest)
    expanded_paths: list[str] = []
    expansion_root: Path | None = None
    index_sha256 = None
    archive_members = None
    if audio_index is not None:
        if not archive_id:
            raise ValueError("archive_id is required with audio_index")
        audio_index = Path(audio_index).expanduser().resolve()
        index_sha256 = _checksum(audio_index)
        index_metadata = json.loads(audio_index.read_text(encoding="utf-8"))
        if index_metadata.get("manifest_sha256") != manifest_digest(manifest):
            raise ValueError("audio index was created for a different manifest")
        archive_entry = next((item for item in index_metadata.get("archives", [])
                              if item["archive_id"] == archive_id), None)
        if archive_entry is None:
            raise ValueError(f"archive {archive_id!r} is not listed in the audio index")
        archive_members = archive_entry["members"]
        member_ids = {str(item["sample_id"]) for item in archive_members}
        records = [row for row in records if str(row["sample_id"]) in member_ids]
        if {str(row["sample_id"]) for row in records} != member_ids:
            raise ValueError(f"archive {archive_id} contains sample IDs missing from the manifest")
        temp_parent = Path(temporary_root).expanduser().resolve() if temporary_root else Path(tempfile.gettempdir())
        temp_parent.mkdir(parents=True, exist_ok=True)
        expansion_root = temp_parent / f"audio-{archive_id}-{uuid.uuid4().hex}"
    elif archive_id is not None:
        raise ValueError("audio_index is required with archive_id")
    model = create_model(model_name, checkpoint, device, model_source)
    preprocessing = dict(preprocessing or {})
    window_seconds = float(preprocessing.get("window_seconds", model.window_seconds))
    if model_name == "ssast" and window_seconds != model.window_seconds:
        raise ValueError("SSAST requires its fixed 10.24-second frontend window")
    channel_mode = preprocessing.get("channel_mode", "mean")
    pad_mode = preprocessing.get("pad_mode", "constant")
    if channel_mode not in {"mean", "first"}:
        raise ValueError("channel_mode must be 'mean' or 'first'")
    if pad_mode not in {"constant", "edge", "reflect"}:
        raise ValueError("pad_mode must be constant, edge, or reflect")
    provenance = None
    if checkpoint_metadata:
        provenance = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict):
            raise ValueError("checkpoint metadata must be a JSON object")
    if model_name == "ssast":
        required = {"source", "pretraining_data", "license", "evaluation_audio_excluded", "supervised_finetuning"}
        if provenance is None or not required.issubset(provenance):
            raise ValueError(f"SSAST requires --checkpoint-metadata JSON with fields: {', '.join(sorted(required))}")
        if provenance["evaluation_audio_excluded"] is not True or provenance["supervised_finetuning"] is not False:
            raise ValueError("SSAST checkpoint must exclude held-out evaluation audio and must not be supervised-finetuned")
    if audio_index is not None:
        expanded = extract_audio_batch(audio_index, archive_id, expansion_root)
        expanded_paths = [item["audio_path"] for item in expanded]
        path_by_id = {item["sample_id"]: item["audio_path"] for item in expanded}
        for row in records:
            row["_original_audio_path"] = row.get("audio_path")
            row["audio_path"] = path_by_id[str(row["sample_id"])]
    config = {"model": model_name, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _checksum(checkpoint),
              "manifest_sha256": manifest_digest(manifest), "batch_size": batch_size, "device": device,
              "sample_rate": model.sample_rate, "channels": channel_mode,
              "windowing": model.pooling, "preprocessing_id": hashlib.sha256(json.dumps({**preprocessing, "window_seconds": window_seconds,
                                  "channel_mode": channel_mode, "pad_mode": pad_mode}, sort_keys=True).encode()).hexdigest()[:12],
              "preprocessing": {**preprocessing, "window_seconds": window_seconds,
                                  "channel_mode": channel_mode, "pad_mode": pad_mode},
              "embedding_pooling": "arithmetic mean over windows",
              "python": platform.python_version(), "numpy": np.__version__,
              "adapter_sha256": _checksum(Path(__file__).parents[1] / "models" / "adapters.py")}
    if provenance is not None:
        config["checkpoint_provenance"] = provenance
    if audio_index is not None:
        config.update({"audio_index_sha256": index_sha256, "archive_id": archive_id})
    if model_name == "ssast":
        config["checkpoint_provenance"] = provenance
        config["preprocessing"] = {**config["preprocessing"],
                                   "frontend": "torchaudio.compliance.kaldi.fbank",
                                   "num_mel_bins": 128, "frame_shift_ms": 10,
                                   "window_type": "hanning", "input_frames": 1024,
                                   "normalization_mean": -4.2677393,
                                   "normalization_std": 4.5689974,
                                   "normalization_divisor": 2,
                                   "feature_pooling": "mean CLS and distillation tokens before task head",
                                   "crop_policy": "fixed 10.24-second window; truncate excess frames",
                                   "padding": pad_mode, "channel_handling": channel_mode}
    if model_name == "beats":
        config["checkpoint_encoder_config"] = getattr(model, "checkpoint_config", None)
        config["preprocessing"] = {**config["preprocessing"],
                                   "frontend": "official BEATs waveform encoder",
                                   "sample_rate": model.sample_rate,
                                   "window_seconds": window_seconds,
                                   "padding": pad_mode,
                                   "padding_mask": "all-false for fixed-length padded input windows",
                                   "feature_pooling": "mean over official encoder frame/token dimension",
                                   "channel_handling": channel_mode}
    if model_name == "beats" and model.source:
        source_files = [model.source / name for name in ("BEATs.py", "modules.py", "backbone.py")]
        source_digest = hashlib.sha256()
        for source_file in source_files:
            if not source_file.is_file():
                raise FileNotFoundError(f"Missing official BEATs source file: {source_file}")
            source_digest.update(source_file.name.encode())
            source_digest.update(bytes.fromhex(_checksum(source_file)))
        config["official_source_sha256"] = source_digest.hexdigest()
    if model_name == "ssast" and model.source:
        source_file = model.source / "models" / "ast_models.py"
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing official SSAST source file: {source_file}")
        config["official_source_sha256"] = _checksum(source_file)
    try:
        import torch
        config["torch"] = torch.__version__
    except ImportError:
        pass
    for distribution in ("torchaudio", "tqdm", "laion-clap", "panns-inference"):
        try:
            config[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    config["created_utc"] = datetime.now(timezone.utc).isoformat()
    config_key = hashlib.sha256(json.dumps({k: v for k, v in config.items() if k != "created_utc"}, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = output / f"{model_name}-{config_key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        for key in config:
            if key != "created_utc" and old.get(key) != config[key]:
                raise ValueError(f"Existing run metadata conflicts at {key}; refusing to mix runs")
        config = old
    else:
        _atomic_text(config_path, json.dumps(config, indent=2))

    vectors: dict[str, np.ndarray] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    emb_file = run_dir / "embeddings.npz"
    samples_file = run_dir / "samples.jsonl"
    if emb_file.exists() and samples_file.exists():
        with np.load(emb_file, allow_pickle=False) as stored:
            vectors = {str(key): value for key, value in zip(stored["sample_ids"], stored["embeddings"])}
        rows_by_id = {row["sample_id"]: row for row in (json.loads(line) for line in samples_file.read_text(encoding="utf-8").splitlines())}
    failures_path = run_dir / "failures.json"
    if failures_path.exists():
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
    pending = [r for r in records if r["sample_id"] not in vectors]
    pending_ids = {r["sample_id"] for r in pending}
    failures = [failure for failure in failures if failure["sample_id"] not in pending_ids]
    try:
        for begin in tqdm(range(0, len(pending), batch_size), desc=f"Extracting {model_name}"):
            rows_batch = pending[begin:begin + batch_size]
            window_batch, window_ids, row_lookup = [], [], {}
            for row in rows_batch:
                try:
                    if not row["audio_path"]:
                        raise ValueError("audio_path is null; acquire audio before extraction")
                    audio_path = Path(row["audio_path"])
                    if not audio_path.is_file():
                        raise FileNotFoundError(str(audio_path))
                    waveform = load_audio(str(audio_path), model.sample_rate, channel_mode=channel_mode)
                    if "start_seconds" in row and "end_seconds" in row:
                        waveform = slice_audio(waveform, model.sample_rate,
                                               float(row["start_seconds"]), float(row["end_seconds"]))
                    chunks = window_audio(waveform, model.sample_rate, window_seconds, pad_mode=pad_mode)
                    window_batch.extend(chunks)
                    window_ids.extend([row["sample_id"]] * len(chunks))
                    row_lookup[row["sample_id"]] = row
                except Exception as exc:
                    failures.append({"sample_id": row["sample_id"], "audio_path": str(row.get("_original_audio_path", row.get("audio_path"))), "error": f"{type(exc).__name__}: {exc}"})
            ids = list(row_lookup)
            if not ids:
                if audio_index is None:
                    _save(run_dir, vectors, rows_by_id, failures)
                continue
            def encode_windows(windows):
                if model_name in {"ssast", "beats"}:
                    import torch
                    batch = torch.stack(windows).to(device)
                    return model.encode(batch)
                return model.encode(np.stack([window.detach().cpu().numpy() for window in windows]))

            try:
                encoded = encode_windows(window_batch)
                if len(encoded) != len(window_ids):
                    raise ValueError(f"encoder returned {len(encoded)} vectors for {len(window_ids)} windows")
                vectors_by_clip: dict[str, list[np.ndarray]] = {sample_id: [] for sample_id in ids}
                for sample_id, vector in zip(window_ids, encoded):
                    vectors_by_clip[sample_id].append(vector)
                batch_vectors = {}
                batch_rows = {}
                for sample_id in ids:
                    vector = np.mean(vectors_by_clip[sample_id], axis=0).astype(np.float32)
                    if not np.isfinite(vector).all():
                        raise ValueError(f"encoder returned non-finite values for {sample_id}")
                    batch_vectors[sample_id] = vector
                    output_row = dict(row_lookup[sample_id])
                    output_row["audio_path"] = output_row.pop("_original_audio_path", output_row.get("audio_path"))
                    batch_rows[sample_id] = output_row
                vectors.update(batch_vectors)
                rows_by_id.update(batch_rows)
            except Exception as exc:
                # Retry per clip so one malformed sample does not discard the whole batch.
                for sample_id in ids:
                    try:
                        clip_windows = [win for win, owner in zip(window_batch, window_ids) if owner == sample_id]
                        encoded = encode_windows(clip_windows)
                        if len(encoded) != len(clip_windows):
                            raise ValueError(f"encoder returned {len(encoded)} vectors for {len(clip_windows)} windows")
                        vector = encoded.mean(axis=0).astype(np.float32)
                        if not np.isfinite(vector).all():
                            raise ValueError(f"encoder returned non-finite values for {sample_id}")
                        vectors[sample_id] = vector
                        output_row = dict(row_lookup[sample_id])
                        output_row["audio_path"] = output_row.pop("_original_audio_path", output_row.get("audio_path"))
                        rows_by_id[sample_id] = output_row
                    except Exception as single_exc:
                        failures.append({"sample_id": sample_id, "audio_path": row_lookup[sample_id].get("_original_audio_path", row_lookup[sample_id]["audio_path"]),
                                         "error": f"{type(single_exc).__name__}: {single_exc} (batch error: {exc})"})
            if audio_index is None:
                _save(run_dir, vectors, rows_by_id, failures)
        if audio_index is not None or not pending:
            _save(run_dir, vectors, rows_by_id, failures)
    finally:
        if expansion_root is not None:
            cleanup_expanded_batch(expansion_root, expanded_paths)
    if audio_index is not None:
        expected_ids = {str(row["sample_id"]) for row in records}
        failed_ids = [str(row["sample_id"]) for row in failures]
        if len(failed_ids) != len(set(failed_ids)) or set(vectors) & set(failed_ids) or set(vectors) | set(failed_ids) != expected_ids:
            raise ValueError(f"archive {archive_id} extraction did not account for every sample exactly once")
        artifacts = {name: _checksum(run_dir / name) for name in ("embeddings.npz", "samples.jsonl", "failures.json")}
        completion = {"archive_id": archive_id, "audio_index_sha256": index_sha256,
                      "expected_sample_ids": sorted(expected_ids), "embedded_count": len(vectors),
                      "failed_count": len(failures), "artifact_sha256": artifacts}
        _atomic_text(run_dir / "archive_complete.json", json.dumps(completion, indent=2))
    if vectors:
        config["embedding_dimension"] = int(next(iter(vectors.values())).shape[-1])
        _atomic_text(config_path, json.dumps(config, indent=2))
    return run_dir


def finalize_compact_extraction(manifest: Path, audio_index: Path, output: Path) -> Path:
    """Validate archive extraction runs and assemble the standard feature cache."""
    manifest = Path(manifest).expanduser().resolve()
    audio_index = Path(audio_index).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    records = read_manifest(manifest)
    index_value = json.loads(audio_index.read_text(encoding="utf-8"))
    index_sha = _checksum(audio_index)
    if index_value.get("manifest_sha256") != manifest_digest(manifest):
        raise ValueError("audio index was created for a different manifest")
    expected = {str(row["sample_id"]) for row in records}
    vectors: dict[str, np.ndarray] = {}
    failures: dict[str, dict[str, Any]] = {}
    common_keys = ("model", "checkpoint_sha256", "checkpoint_provenance", "manifest_sha256",
                   "audio_index_sha256", "preprocessing_id", "preprocessing", "adapter_sha256",
                   "official_source_sha256", "checkpoint_encoder_config", "sample_rate", "channels",
                   "windowing", "device", "batch_size", "python", "numpy", "torch", "torchaudio")
    common_config = None
    found_archives: set[str] = set()
    archives = {item["archive_id"]: item for item in index_value.get("archives", [])}
    for metadata_path in output.rglob("run.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("feature_cache_complete") is True:
            continue
        if metadata.get("audio_index_sha256") != index_sha:
            continue
        archive_id = metadata.get("archive_id")
        if archive_id not in archives:
            raise ValueError(f"extraction output refers to unknown archive {archive_id!r}")
        if archive_id in found_archives:
            raise ValueError(f"multiple extraction outputs found for archive {archive_id}")
        found_archives.add(archive_id)
        if common_config is None:
            common_config = metadata
        elif any(metadata.get(key) != common_config.get(key) for key in common_keys):
            raise ValueError(f"extraction run identity mismatch in archive {archive_id}")
        run_dir = metadata_path.parent
        embedding_path = run_dir / "embeddings.npz"
        samples_path = run_dir / "samples.jsonl"
        failure_path = run_dir / "failures.json"
        completion_path = run_dir / "archive_complete.json"
        if not embedding_path.is_file() or not samples_path.is_file() or not failure_path.is_file() or not completion_path.is_file():
            raise ValueError(f"incomplete extraction artifacts for archive {archive_id}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("audio_index_sha256") != index_sha or completion.get("archive_id") != archive_id:
            raise ValueError(f"archive completion record conflicts for {archive_id}")
        artifact_paths = {"embeddings.npz": embedding_path, "samples.jsonl": samples_path, "failures.json": failure_path}
        if any(_checksum(path) != completion.get("artifact_sha256", {}).get(name)
               for name, path in artifact_paths.items()):
            raise ValueError(f"archive output checksum mismatch for {archive_id}")
        with np.load(embedding_path, allow_pickle=False) as stored:
            ids = [str(value) for value in stored["sample_ids"]]
            matrix = stored["embeddings"]
        if len(ids) != len(matrix) or matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError(f"invalid embedding matrix for archive {archive_id}")
        expected_archive_ids = set(archives[archive_id]["sample_ids"])
        if set(completion.get("expected_sample_ids", [])) != expected_archive_ids:
            raise ValueError(f"archive completion sample IDs mismatch for {archive_id}")
        if not set(ids).issubset(expected_archive_ids):
            raise ValueError(f"unexpected sample IDs in archive output {archive_id}")
        for sample_id, vector in zip(ids, matrix):
            if sample_id in vectors:
                raise ValueError(f"duplicate embedding for sample {sample_id}")
            vectors[sample_id] = np.asarray(vector, dtype=np.float32)
        for failure in json.loads(failure_path.read_text(encoding="utf-8")):
            sample_id = str(failure["sample_id"])
            if sample_id not in expected_archive_ids or sample_id in failures or sample_id in vectors:
                raise ValueError(f"unexpected or duplicate failure for sample {sample_id}")
            failures[sample_id] = failure
        accounted = set(ids) | {sid for sid in failures if sid in expected_archive_ids}
        if accounted != expected_archive_ids:
            raise ValueError(f"archive {archive_id} is incomplete; unaccounted IDs: {sorted(expected_archive_ids - accounted)}")
    packaging_failures = index_value.get("packaging_failures", [])
    for failure in packaging_failures:
        sample_id = str(failure["sample_id"])
        if sample_id not in expected or sample_id in vectors or sample_id in failures:
            raise ValueError(f"unexpected or duplicate packaging failure for {sample_id}")
        failures[sample_id] = failure
    if found_archives != set(archives):
        raise ValueError(f"missing completed archive outputs: {sorted(set(archives) - found_archives)}")
    if set(vectors) | set(failures) != expected or set(vectors) & set(failures):
        missing = expected - set(vectors) - set(failures)
        raise ValueError(f"feature cache does not account for manifest IDs; missing: {sorted(missing)}")
    dimensions = {vector.shape for vector in vectors.values()}
    if len(dimensions) > 1:
        raise ValueError("embedding dimensions differ between archive outputs")
    if common_config is None:
        raise ValueError("no extraction outputs found")
    if not vectors:
        raise ValueError("no embeddings were produced; refusing to finalize an empty feature cache")

    identity = hashlib.sha256(json.dumps({key: common_config.get(key) for key in common_keys},
                                         sort_keys=True).encode()).hexdigest()[:12]
    cache_dir = output / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_run_json = cache_dir / "run.json"
    if final_run_json.exists():
        old = json.loads(final_run_json.read_text(encoding="utf-8"))
        if old.get("feature_cache_identity") != identity:
            raise ValueError(f"feature cache already contains a different run: {cache_dir}")
        final_run_json.unlink()
    ordered_ids = [str(row["sample_id"]) for row in records if str(row["sample_id"]) in vectors]
    matrix = np.stack([vectors[sample_id] for sample_id in ordered_ids]).astype(np.float32) if ordered_ids else np.empty((0, 0), dtype=np.float32)
    fd, temporary = tempfile.mkstemp(prefix=".embeddings.", suffix=".npz", dir=cache_dir)
    os.close(fd)
    try:
        np.savez_compressed(temporary, sample_ids=np.asarray(ordered_ids), embeddings=matrix)
        os.replace(temporary, cache_dir / "embeddings.npz")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    row_by_id = {str(row["sample_id"]): row for row in records}
    sample_text = "".join(json.dumps(row_by_id[sample_id], ensure_ascii=False) + "\n" for sample_id in ordered_ids)
    _atomic_text(cache_dir / "samples.jsonl", sample_text)
    _atomic_text(cache_dir / "failures.json", json.dumps(list(failures.values()), indent=2))
    final_metadata = {key: value for key, value in common_config.items() if key != "archive_id"}
    final_metadata.update({"audio_index": str(audio_index), "feature_cache_identity": identity,
                      "feature_cache_complete": True,
                      "expected_samples": len(expected), "embedded_samples": len(vectors),
                      "failed_samples": len(failures), "embedding_dimension": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[1] else 0})
    final_metadata["artifact_sha256"] = {name: _checksum(cache_dir / name)
                                         for name in ("embeddings.npz", "samples.jsonl", "failures.json")}
    _atomic_text(cache_dir / "run.json", json.dumps(final_metadata, indent=2))
    return cache_dir
