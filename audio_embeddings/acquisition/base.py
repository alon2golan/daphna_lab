"""Label-free input and output contracts used by :class:`Sampler`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..task import Task


def _vector(value: Any, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.long, device="cpu").flatten()
    if result.unique().numel() != result.numel():
        raise ValueError(f"{name} must contain unique global IDs")
    return result


@dataclass(frozen=True)
class AcquisitionContext:
    """Immutable input; feature/probability rows align with ``pool_ids``.

    Public inputs are converted to CPU int64 IDs and float32 arrays so
    sampling has no hidden device dependency. Returned IDs are global IDs.
    """

    pool_ids: Any
    labeled_ids: Any
    candidate_ids: Any
    features: Any
    probabilities: Any | None = None
    mc_probabilities: Any | None = None
    seed: int = 0
    task: Task | str = Task.multiclass
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pool = _vector(self.pool_ids, "pool_ids")
        labeled = _vector(self.labeled_ids, "labeled_ids")
        candidates = _vector(self.candidate_ids, "candidate_ids")
        pool_set = set(pool.tolist())
        labeled_set, candidate_set = set(labeled.tolist()), set(candidates.tolist())
        if not labeled_set.issubset(pool_set) or not candidate_set.issubset(pool_set):
            raise ValueError("labeled_ids and candidate_ids must be drawn from pool_ids")
        if labeled_set & candidate_set:
            raise ValueError("labeled_ids and candidate_ids must be disjoint")
        features = torch.as_tensor(self.features, dtype=torch.float32, device="cpu")
        if features.ndim != 2 or features.shape[0] != pool.numel():
            raise ValueError("features must be a matrix with one row per pool_id")
        if not torch.isfinite(features).all():
            raise ValueError("features must contain finite values")
        object.__setattr__(self, "pool_ids", pool)
        object.__setattr__(self, "labeled_ids", labeled)
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "features", features)
        for name, ndim in (("probabilities", 2), ("mc_probabilities", 3)):
            value = getattr(self, name)
            if value is None:
                continue
            prob_tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
            if prob_tensor.ndim != ndim or prob_tensor.shape[-2] != pool.numel():
                raise ValueError(f"{name} must have {ndim} dimensions and one sample row per pool ID")
            if prob_tensor.shape[-1] == 0 or (name == "mc_probabilities" and prob_tensor.shape[0] == 0):
                raise ValueError(f"{name} must contain at least one class and, for MC values, one draw")
            if not torch.isfinite(prob_tensor).all():
                raise ValueError(f"{name} must contain finite values")
            if (prob_tensor < 0).any() or (prob_tensor > 1).any():
                raise ValueError(f"Invalid {name} values: probabilities must lie in [0, 1]")
            object.__setattr__(self, name, prob_tensor)
        try:
            task = Task(self.task)
        except ValueError as exc:
            raise ValueError("task must be Task.multiclass or Task.multilabel") from exc
        object.__setattr__(self, "task", task)
        if self.probabilities is not None and self.mc_probabilities is not None:
            if self.probabilities.shape[1] != self.mc_probabilities.shape[2]:
                raise ValueError("probabilities and mc_probabilities must have the same class count")
            mean = self.mc_probabilities.mean(dim=0)
            if not torch.allclose(self.probabilities, mean, atol=1e-4, rtol=1e-3):
                raise ValueError("probabilities must approximately equal mc_probabilities.mean(dim=0)")
        if task == Task.multiclass:
            for name in ("probabilities", "mc_probabilities"):
                values = getattr(self, name)
                if values is None:
                    continue
                sums = values.sum(dim=-1)
                if not torch.allclose(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3):
                    raise ValueError(f"{name} categorical probability rows must sum to one")

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(self.pool_ids.device)
        mask = torch.isin(self.pool_ids, ids)
        return torch.where(mask)[0]

@dataclass(frozen=True)
class QueryResult:
    selected_ids: torch.Tensor
    scores: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
