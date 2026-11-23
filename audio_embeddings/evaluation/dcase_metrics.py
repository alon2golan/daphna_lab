"""Official one-second DCASE-style segment metrics for window predictions."""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def evaluate_segment_metrics(records: list[dict], indices: np.ndarray,
                             probabilities: np.ndarray, class_names: list[str],
                             thresholds: np.ndarray) -> dict[str, float]:
    """Expand window tags to timed events and score with DCASE's sed_eval API.

    Window predictions are active over the whole 10-second candidate interval;
    this supports segment metrics only and must not be presented as event timing.
    """
    try:
        import sed_eval
    except ImportError as exc:
        raise RuntimeError("DCASE segment metrics require sed_eval; install audio_embeddings requirements") from exc

    reference_by_file: dict[str, list[dict]] = defaultdict(list)
    estimated_by_file: dict[str, list[dict]] = defaultdict(list)
    for index, row_probs in zip(indices, probabilities):
        record = records[int(index)]
        recording = str(record.get("recording_id", record["sample_id"]))
        scene = str(record.get("scene", ""))
        reference_by_file.setdefault(recording, [])
        estimated_by_file.setdefault(recording, [])
        start = float(record.get("start_seconds", 0.0))
        end = float(record.get("end_seconds", start + 10.0))
        for event in record.get("events", []):
            reference_by_file[recording].append({
                "file": recording, "scene_label": scene,
                "event_label": str(event["event_class"]),
                "event_onset": start + float(event.get("window_onset", event["onset"])),
                "event_offset": start + float(event.get("window_offset", event["offset"])),
            })
        active = np.asarray(row_probs) >= thresholds
        for class_index in np.flatnonzero(active):
            estimated_by_file[recording].append({
                "file": recording, "scene_label": scene,
                "event_label": str(class_names[int(class_index)]),
                "event_onset": start, "event_offset": end,
            })

    metric = sed_eval.sound_event.SegmentBasedMetrics(class_names, time_resolution=1.0)
    for filename in sorted(reference_by_file):
        metric.evaluate(reference_by_file[filename], estimated_by_file[filename])
    overall = metric.results_overall_metrics()
    return {"segment_er": float(overall["error_rate"]["error_rate"]),
            "segment_f_score": float(overall["f_measure"]["f_measure"])}
