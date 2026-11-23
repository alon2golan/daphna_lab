"""Task-head and prediction wrapper kept separate from backbone adapters."""
from __future__ import annotations

import numpy as np
import torch

from ..classifier import build_head, calc_probs, fit_head, predict


class AudioALModel:
    """Wrap a backbone adapter with a task head and prediction interface.

    ``frozen_representation`` and ``linearprob`` operate on supplied embeddings;
    ``fine_tune`` is delegated to the waveform-level training functions.
    """

    def __init__(self, adapter, head: torch.nn.Module, objective: dict, device: str = "cpu"):
        self.adapter = adapter
        self.head = head
        self.objective = dict(objective)
        self.device = str(device)
        self.long_shadow = None
        self.short_shadow = None

    @classmethod
    def from_embeddings(cls, adapter, input_dim: int, output_dim: int, objective: dict,
                        config: dict, device: str = "cpu"):
        mode = config.get("adaptation", "frozen_representation")
        if mode == "linearprob" and config.get("hidden_sizes"):
            raise ValueError("linearprob uses one affine layer and cannot configure hidden_sizes")
        if mode not in {"frozen_representation", "linearprob", "fine_tune"}:
            raise ValueError(f"unsupported adaptation mode {mode!r}")
        head_config = {} if mode == "linearprob" else config
        return cls(adapter, build_head(input_dim, output_dim, head_config, device), objective, device)

    def fit(self, features: np.ndarray, targets: np.ndarray, output_dim: int, config: dict, seed: int,
            shadow: bool = False):
        self.head, self.long_shadow, self.short_shadow = fit_head(
            features, targets, output_dim, self.objective, config, self.device, seed, shadow=shadow
        )
        return self

    def predict(self, features: np.ndarray, mc_passes: int = 1):
        return predict(self.head, features, self.objective, self.device, mc_passes=mc_passes)

    def predict_shadows(self, features: np.ndarray):
        if self.long_shadow is None or self.short_shadow is None:
            raise RuntimeError("EMA shadows are not initialized")
        return torch.cat((predict(self.long_shadow, features, self.objective, self.device),
                          predict(self.short_shadow, features, self.objective, self.device)), dim=0)

    @staticmethod
    def probabilities(logits: torch.Tensor, task: str):
        return calc_probs(logits, task)
