"""
ResultWriter: writes and atomically updates result.json for a single train_al.py run.

The file is created at job start (status="running"), updated after every episode,
and finalised with status="completed" or "failed" at exit.

Failure detection uses atexit so status is set to "failed" automatically whenever
the process exits without an explicit call to finish() — covering Python exceptions,
SIGTERM (SLURM time-limit), and normal exit without cleanup.  SIGKILL cannot be
caught, but SLURM sends SIGTERM first.

Atomic write strategy: write to <file>.tmp, then os.replace() → a reader always
sees a complete, valid JSON file.
"""

import atexit
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class ResultWriter:
    """Creates and maintains result.json for one active-learning run."""

    RESULT_FILE = "result.json"

    def __init__(self, exp_dir: str, args, cfg) -> None:
        self.exp_dir = exp_dir
        self._result_path = os.path.join(exp_dir, self.RESULT_FILE)
        self._tmp_path = self._result_path + ".tmp"

        # Sweep identity from SLURM environment --------------------------------
        slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_array_job_id:
            sweep_id = slurm_array_job_id
        elif slurm_job_id:
            sweep_id = slurm_job_id
        else:
            sweep_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

        # Git commit hash -------------------------------------------------------
        commit_hash = _get_commit_hash()

        # Noise / noise-filter knobs (from cfg, not CLI) -----------------------
        # Recorded so the dashboard can facet curves by noise level and by
        # whether predicted-noisy points are dropped before training. TRAIN_ON_CLEAN
        # is tri-state (None => auto: on iff a filter is configured); resolve it to
        # the effective bool the same way train_al._clean_training_indices does.
        _nf = getattr(cfg, "NOISE_FILTER", None)
        _nf_method = str(getattr(_nf, "METHOD", "none")).lower() if _nf is not None else "none"
        _toc = getattr(_nf, "TRAIN_ON_CLEAN", None) if _nf is not None else None
        if _toc is None:
            _toc = _nf_method not in ("none", "noop", "")
        _annot = getattr(cfg, "ANNOTATORS", None)
        _noise_rates = list(getattr(_annot, "NOISE_RATES", []) or []) if _annot is not None else []
        _noise_rate = _noise_rates[0] if _noise_rates else None

        # Free-form run tags (from --tag KEY=VALUE, repeatable) ----------------
        # Kept in a dedicated dict (NOT merged into params) so a tag can never
        # shadow a real parameter. The scanner expands each into a p_<key>
        # column so it rides the dashboard's existing facet/filter machinery.
        tags: dict = {}
        for _t in (getattr(args, "tag", None) or []):
            if "=" not in _t:
                print(f"[ResultWriter] ignoring malformed --tag (need KEY=VALUE): {_t!r}")
                continue
            _k, _v = _t.split("=", 1)
            _k = _k.strip()
            if _k:
                tags[_k] = _v.strip()

        # Params from args + cfg -----------------------------------------------
        params = {
            "al_method": getattr(args, "al", None),
            "eval_model_type": getattr(args, "eval_model_type", None),
            "dataset": cfg.DATASET.NAME,
            "model": cfg.MODEL.TYPE,
            "budget": getattr(args, "budget", None),
            "max_iter": getattr(args, "max_iter", None),
            "seed": getattr(args, "seed", None),
            "initial_delta": getattr(args, "initial_delta", None),
            "sparse_K": getattr(args, "sparse_K", False),
            "initial_size": getattr(args, "initial_size", 0),
            "cfg_file": getattr(args, "cfg_file", None),
            "kernel_type": getattr(args, "kernel_type", None),
            "diff_method": getattr(args, "diff_method", None),
            "alpha": getattr(args, "alpha", None),
            "tau": getattr(args, "tau", None),
            "target_nu": getattr(args, "target_nu", None),
            "features_tag": getattr(args, "features_tag", None),
            "nu_prime": getattr(args, "nu_prime", None),
            "nu_optimizer": getattr(args, "nu_optimizer", None),
            "matrix_dtype": getattr(args, "matrix_dtype", None),
            "noise_rate": _noise_rate,
            "noise_filter": _nf_method,
            "train_on_clean": bool(_toc),
            "train_label_mode": getattr(getattr(cfg, "BAYES_COVERAGE", None),
                                        "TRAIN_LABEL_MODE", "observed"),
            "track_progress": bool(getattr(args, "track_progress", False)),
        }
        # §7.5 pseudo-labeling knobs — recorded ONLY for pseudo runs so every
        # other config's params dict (and hence its config_key hash / dashboard
        # about= URL) is byte-identical to before. They must be params (not just
        # tags): tags are excluded from config_key, so two pseudo arms differing
        # only by scope/schedule would otherwise collapse into one seed-group.
        if params["train_label_mode"] == "pseudo":
            _bc = getattr(cfg, "BAYES_COVERAGE", None)
            params["pseudo_scope"] = getattr(_bc, "PSEUDO_SCOPE", None)
            _sched = getattr(_bc, "PSEUDO_K_SCHEDULE", None) or []
            params["pseudo_k"] = ",".join(str(k) for k in _sched) or None
        # Classifier-input L2 normalization — recorded ONLY when on, for the same
        # config_key-stability reason as the pseudo knobs above. Must be a param
        # (not just a tag) so normalized reruns get distinct config_keys from the
        # historical unnormalized runs instead of collapsing into one seed-group.
        if bool(getattr(args, "normalize_features", False)):
            params["normalize_features"] = True

        sweep_name = getattr(args, "sweep_name", None) or sweep_id

        self._data: dict = {
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "finished_at": None,
            "exp_dir": exp_dir,
            "command": " ".join(sys.argv),
            "commit_hash": commit_hash,
            "sweep_id": sweep_id,
            "sweep_name": sweep_name,
            "slurm_task_id": slurm_task_id,
            "params": params,
            "tags": tags,
            "episodes": [],
            "final_test_acc": None,
            "total_episodes_planned": getattr(args, "max_iter", 0) + 1,
            "total_episodes_done": 0,
        }

        self._write()
        atexit.register(self._atexit_handler)
        print(f"[ResultWriter] Tracking → {self._result_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def record_episode(
        self,
        episode: int,
        test_acc: float,
        val_acc: float,
        train_acc: float,
        labeled_size: int,
        duration_s: float,
    ) -> None:
        """Append one episode's results and persist atomically."""
        self._data["episodes"].append(
            {
                "episode": int(episode),
                "test_acc": round(float(test_acc), 4),
                "val_acc": round(float(val_acc), 4),
                "train_acc": round(float(train_acc), 4),
                "labeled_size": int(labeled_size),
                "duration_s": round(float(duration_s), 2),
            }
        )
        self._data["total_episodes_done"] = len(self._data["episodes"])
        self._data["final_test_acc"] = round(float(test_acc), 4)
        self._write()

    def finish(self) -> None:
        """Mark run as completed and persist."""
        atexit.unregister(self._atexit_handler)
        self._data["status"] = "completed"
        self._data["finished_at"] = datetime.now().isoformat()
        self._write()
        print(f"[ResultWriter] Run completed → {self._result_path}")

    def fail(self, exc: Exception = None) -> None:
        """Mark run as failed (call explicitly when catching an exception)."""
        atexit.unregister(self._atexit_handler)
        self._data["status"] = "failed"
        self._data["finished_at"] = datetime.now().isoformat()
        if exc is not None:
            self._data["error"] = str(exc)
        self._write()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _atexit_handler(self) -> None:
        """Fallback: mark as failed if process exits without finish()/fail()."""
        if self._data.get("status") == "running":
            self._data["status"] = "failed"
            self._data["finished_at"] = datetime.now().isoformat()
            self._data["error"] = (
                "Process exited without calling finish() — likely killed or crashed."
            )
            self._write()

    def _write(self) -> None:
        """Atomic write: tmp file → os.replace to avoid partial reads."""
        with open(self._tmp_path, "w") as fh:
            json.dump(self._data, fh, indent=2, default=str)
        os.replace(self._tmp_path, self._result_path)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_commit_hash() -> str:
    """Return current git HEAD SHA, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
