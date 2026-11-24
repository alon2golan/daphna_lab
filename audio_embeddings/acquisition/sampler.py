from __future__ import annotations

import math
import time

import torch

from .base import AcquisitionContext, QueryResult
from .numeric import rbf_kernel, squared_euclidean
from ..task import Task


class Sampler:
    """Select unique global IDs from the unlabeled candidates.

    accepts legacy config aliases; direct methods are available when a caller
    wants to invoke one algorithm explicitly.
    """

    _ALIASES = {
        "random": "random", "uncertainty": "least_confidence",
        "least_confidence": "least_confidence", "min_margin": "margin", "margin": "margin",
        "max_entropy": "entropy", "entropy": "entropy", "bald": "bald",
        "core_set": "core_set", "kcenter": "core_set", "k_center": "core_set",
        "probcover": "probcover", "max_herding": "max_herding",
        "maxherding": "max_herding", "typiclust": "typiclust", "cls_al": "cls_al"
    }

    @staticmethod
    def _validate_budget(context: AcquisitionContext, budget: int) -> int:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("Budget must be a positive integer")
        if budget > len(context.candidate_ids):
            raise ValueError("Budget cannot exceed the number of candidates")
        return budget


    def select(self, method: str, context: AcquisitionContext, 
               budget: int, delta: float | None = None, sigma: float | None = None) -> QueryResult:
        refined_key = method.lower().replace("-", "_")
        matched_method = self._ALIASES.get(refined_key)
        if matched_method is None:
            raise ValueError(f"Unknown acquisition method {method}; choose one of {sorted(self._ALIASES)}")
        if matched_method == "random":
            return self.random(context, budget)
        if matched_method == "least_confidence":
            return self.least_confidence(context, budget)
        if matched_method == "margin":
            return self.margin(context, budget)
        if matched_method == "entropy":
            return self.entropy(context, budget)
        if matched_method == "bald":
            return self.bald(context, budget)
        if matched_method == "cls_al":
            return self.cls_al(context, budget)
        if matched_method == "core_set":
            return self.core_set(context, budget)
        if matched_method == "probcover":
            return self.probcover(context, budget, delta)
        if matched_method == "max_herding":
            return self.max_herding(context, budget, sigma=sigma)
        return self.typiclust(context, budget)

    def random(self, context: AcquisitionContext, budget: int) -> QueryResult:
        """Sample a uniform size-B subset; O(N) for permutation (Settles, 2009)."""
        budget = self._validate_budget(context, budget)
        generator = torch.Generator().manual_seed(int(context.seed))
        rand_selected_ids = torch.randperm(len(context.candidate_ids), generator=generator)[:budget].to(context.candidate_ids.device)
        return QueryResult(context.candidate_ids[rand_selected_ids], metadata={"seed": int(context.seed)})

    def _candidate_probabilities(self, context: AcquisitionContext) -> torch.Tensor:
        if context.probabilities is None:
            raise ValueError("this strategy requires precomputed probabilities")
        p = context.probabilities[context.rows(context.candidate_ids)].float()
        return self._stable_probabilities(p)

    @staticmethod
    def _stable_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
        """Clamp a calculation copy for stable logarithms; retain source values."""
        return probabilities.clamp(1e-8, 1 - 1e-8)

    @staticmethod
    def _rank(context: AcquisitionContext, scores: torch.Tensor, budget: int, descending: bool, name: str) -> QueryResult:
        ordered_candidates = torch.argsort(scores, descending=descending, stable=True)[:budget]
        return QueryResult(context.candidate_ids[ordered_candidates], scores[ordered_candidates], {"strategy": name})

    def least_confidence(self, context: AcquisitionContext, budget: int) -> QueryResult:
        """Choose smallest max_c p(c|x), or mean_c max(p,1-p) for multilabel.

        Cost O(N*C). Uncertainty sampling baseline: Settles (2009),
        *Active Learning Literature Survey*.
        """
        budget = self._validate_budget(context, budget)
        probs = self._candidate_probabilities(context)

        if context.task == Task.multiclass:
            scores = probs.max(dim=1).values
        else:
            scores = torch.maximum(probs, 1 - probs).mean(dim=1)

        return self._rank(context, scores, budget, descending=False, name="least confidence")

    def margin(self, context: AcquisitionContext, budget: int) -> QueryResult:
        """Choose smallest p_(1)-p_(2), or mean_c |2p_c-1| for multilabel.

        Cost O(N*C). Tong and Koller (2001), *Support Vector Machine Active
        Learning with Applications to Text Classification*.
        """
        budget = self._validate_budget(context, budget)
        probs = self._candidate_probabilities(context)

        if context.task == Task.multiclass:
            top = probs.topk(2, dim=1).values
            scores = top[:, 0] - top[:, 1]
        else:
            scores = (2 * probs - 1).abs().mean(dim=1)

        return self._rank(context, scores, budget, descending=False, name="margin")

    def entropy(self, context: AcquisitionContext, budget: int) -> QueryResult:
        """Choose largest -sum_c p_c log p_c (or Bernoulli entropy sum).

        Cost O(N*C). Lewis and Gale (1994), *A Sequential Algorithm for
        Training Text Classifiers*.
        """
        budget = self._validate_budget(context, budget)
        probs = self._candidate_probabilities(context)
        if context.task == Task.multiclass:
            scores = -(probs * probs.log()).sum(dim=1)
        else:
            scores = -(probs * probs.log() + (1 - probs) * (1 - probs).log()).sum(dim=1)
        return self._rank(context, scores, budget, descending=True, name="entropy")

    def bald(self, context: AcquisitionContext, budget: int) -> QueryResult:
        """Choose largest H(mean_m p_m)-mean_m H(p_m); O(M*N*C).

        Houlsby et al. (2011), *Bayesian Active Learning for Classification
        and Preference Learning*.
        """
        budget = self._validate_budget(context, budget)
        draws = context.mc_probabilities
        if draws is None or draws.shape[0] < 2:
            raise ValueError("BALD requires at least two MC probability draws")
        draws = draws[:, context.rows(context.candidate_ids)]
        if (draws < 0).any() or (draws > 1).any():
            raise ValueError("MC probabilities must lie in [0, 1]")
        draws = self._stable_probabilities(draws)
        mean = draws.mean(dim=0)
        if context.task == Task.multiclass:
            if not torch.allclose(draws.sum(dim=2), torch.ones(draws.shape[:2]), atol=1e-3):
                raise ValueError("categorical MC probability rows must sum to one")
            predictive = -(mean * mean.log()).sum(dim=1)
            expected = -(draws * draws.log()).sum(dim=2).mean(dim=0)
        else:
            predictive = -(mean * mean.log() + (1 - mean) * (1 - mean).log()).sum(dim=1)
            expected = -(draws * draws.log() + (1 - draws) * (1 - draws).log()).sum(dim=2).mean(dim=0)
        return self._rank(context, predictive - expected, budget, descending=True, name="bald")

    def cls_al(self, context: AcquisitionContext, budget: int) -> QueryResult:
        """Choose largest max_c |p_long(c|x)-p_short(c|x)|; O(N*C).

        Geng et al. (2025), *Complementary Learning System Theory-based
        Active Learning for Audio Classification*.
        """
        budget = self._validate_budget(context, budget)
        draws = context.mc_probabilities
        if draws is None or draws.shape[0] != 2:
            raise ValueError("CLS-AL requires long- and short-term EMA probability predictions")
        rows = context.rows(context.candidate_ids)
        scores = (draws[0, rows] - draws[1, rows]).abs().amax(dim=1)
        return self._rank(context, scores, budget, descending=True, name="cls_al")

    def core_set(self, context: AcquisitionContext, budget: int) -> QueryResult:
        """Farthest-first k-center (Sener & Savarese, 2018), O((N+L)D+BND).

        At each step choose argmax_x min_{z in L union S} ||x-z||^2.
        Ties follow candidate order; a cold start uses the context seed.
        """
        budget = self._validate_budget(context, budget)
        candidate_rows = context.rows(context.candidate_ids)
        labeled_rows = context.rows(context.labeled_ids)
        candidates = context.features[candidate_rows]
        if len(labeled_rows):
            nearest = squared_euclidean(candidates, context.features[labeled_rows]).min(dim=1).values
        else:
            generator = torch.Generator(device=context.candidate_ids.device)
            generator.manual_seed(int(context.seed))
            first = int(torch.randint(len(candidates), (1,), generator=generator,
                                      device=context.candidate_ids.device))
            nearest = squared_euclidean(candidates, candidates[first:first + 1]).squeeze(1)
            nearest[first] = -1
        selected, scores = [], []
        for step in range(budget):
            if not len(labeled_rows) and step == 0:
                pos = first
                score = -1.0
            else:
                pos = int(torch.argmax(nearest))
                score = float(nearest[pos])
            selected.append(int(context.candidate_ids[pos]))
            scores.append(score)
            nearest = torch.minimum(nearest, squared_euclidean(candidates, candidates[pos:pos + 1]).squeeze(1))
            nearest[pos] = -float("inf")
        return QueryResult(torch.tensor(selected, device=context.candidate_ids.device),
                           torch.tensor(scores, device=candidates.device), {"metric": "squared_euclidean"})

    def probcover(self, context: AcquisitionContext, budget: int, delta: float | None) -> QueryResult:
        """Choose argmax_x |{u: d(x,u)<=delta and u not covered}|.

        Cost O(N^2*D+B*N^2). Yehuda et al. (2022), *Active Learning
        Through a Covering Lens*.
        """
        budget = self._validate_budget(context, budget)
        delta_metadata = {"delta_mode": "configured"}
        if delta is None:
            num_classes = context.metadata.get("num_classes")
            if not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError("automatic ProbCover delta requires metadata.num_classes >= 2")
            from deep_al.pycls.datasets.utils.select_delta import select_delta
            started = time.perf_counter()
            delta = float(select_delta(context.features.detach().cpu().numpy(), num_classes,
                                       classifier_on_raw_images=False,
                                       prefer_cuda=context.features.device.type == "cuda"))
            delta_metadata = {"delta_mode": "automatic", "delta_heuristic": "deep_al.select_delta representation-space inflection",
                              "delta_selection_seconds": time.perf_counter() - started,
                              "delta_num_classes": num_classes}
        if not math.isfinite(float(delta)) or delta <= 0:
            raise ValueError("delta must be a finite positive radius; delta=0 is not allowed as it does not allow selection to progress")
        candidate_rows = context.rows(context.candidate_ids)
        labeled_rows = context.rows(context.labeled_ids)
        candidates = context.features[candidate_rows]
        covered = torch.zeros(len(candidates), dtype=torch.bool, device=candidates.device)
        if len(labeled_rows):
            covered = (squared_euclidean(candidates, context.features[labeled_rows]) <= delta ** 2).any(dim=1)
        distances = squared_euclidean(candidates, candidates)
        selected, scores = [], []
        chosen = torch.zeros(len(candidates), dtype=torch.bool, device=candidates.device)
        for _ in range(budget):
            gains = ((distances <= delta ** 2) & ~covered.unsqueeze(0)).sum(dim=1)
            gains[chosen] = -1
            pos = int(torch.argmax(gains))
            selected.append(int(context.candidate_ids[pos]))
            scores.append(float(gains[pos]))
            covered |= distances[pos] <= delta ** 2
            chosen[pos] = True
        return QueryResult(torch.tensor(selected, device=context.candidate_ids.device),
                           torch.tensor(scores, device=candidates.device),
                           {"delta": delta, "metric": "euclidean", **delta_metadata})

    def max_herding(self, context: AcquisitionContext, budget: int, sigma: float = 1.0) -> QueryResult:
        """Choose argmax_x mean_u max(0,K(x,u)-covered(u)); O(N^2*D+B*N^2).

        Sutherland, Noh, and Bae (2024), *Generalized Coverage for More
        Robust Low-Budget Active Learning*.
        """
        budget = self._validate_budget(context, budget)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("max_herding requires finite positive sigma")
        candidate_rows = context.rows(context.candidate_ids)
        labeled_rows = context.rows(context.labeled_ids)
        candidates = context.features[candidate_rows]
        kernel = rbf_kernel(candidates, candidates, sigma)
        if len(labeled_rows):
            covered = rbf_kernel(context.features[labeled_rows], candidates, sigma).max(dim=0).values
        else:
            covered = torch.zeros(len(candidates), device=candidates.device)
        chosen = torch.zeros(len(candidates), dtype=torch.bool, device=candidates.device)
        selected, scores = [], []
        for _ in range(budget):
            gains = (kernel - covered.unsqueeze(0)).clamp_min(0).mean(dim=1)
            gains[chosen] = -1
            pos = int(torch.argmax(gains))
            selected.append(int(context.candidate_ids[pos]))
            scores.append(float(gains[pos]))
            covered = torch.maximum(covered, kernel[pos])
            chosen[pos] = True
        return QueryResult(torch.tensor(selected, device=context.candidate_ids.device),
                           torch.tensor(scores, device=candidates.device), {"sigma": sigma, "kernel": "rbf"})

    def typiclust(self, context: AcquisitionContext, budget: int) -> QueryResult:
        """Cluster candidates and pick locally dense exemplars; O(K-means + N^2*D).

        Uses k-means initialization from TypiClust (Hacohen et al., 2022);
        local density is mean distance to up to 20 nearest cluster neighbors.
        """
        budget = self._validate_budget(context, budget)
        from sklearn.cluster import KMeans

        candidate_rows = context.rows(context.candidate_ids)
        candidates = context.features[candidate_rows]
        candidates_array = candidates.detach().cpu().numpy()
        clusters = KMeans(n_clusters=budget, random_state=int(context.seed), n_init=10).fit_predict(candidates_array)
        selected, densities = [], []
        observed = sorted(set(int(v) for v in clusters))
        for cluster_id in observed:
            positions = torch.where(torch.as_tensor(clusters) == cluster_id)[0]
            members = candidates[positions]
            if len(positions) == 1:
                local, density = 0, float("inf")
            else:
                distances = squared_euclidean(members, members).sqrt()
                distances.fill_diagonal_(float("inf"))
                k = min(20, len(members) - 1)
                mean_distance = distances.topk(k, largest=False).values.mean(dim=1)
                local = int(torch.argmin(mean_distance))
                density = float(-mean_distance[local])
            selected.append(int(context.candidate_ids[positions[local]]))
            densities.append(density)
        selected_set = set(selected)
        for sample_id in context.candidate_ids.tolist():
            if len(selected) == budget:
                break
            if int(sample_id) not in selected_set:
                selected.append(int(sample_id))
                densities.append(float("nan"))
                selected_set.add(int(sample_id))
        return QueryResult(torch.tensor(selected), torch.tensor(densities),
                           {"clusters_requested": budget, "clusters_observed": len(observed),
                            "density_neighbors": 20, "empty_cluster_policy": "fill with lowest global candidate IDs"})
