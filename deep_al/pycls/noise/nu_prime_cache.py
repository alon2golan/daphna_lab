"""
nu_prime_cache.py
-----------------
Read / write / query the per-dataset nu_prime cache.

Cache layout (one JSON file per dataset, e.g. results/cifar100/nu_prime_cache.json):
{
  "probcover": {
    "sgd":  { "100": { "0.1000": 0.79, ... }, "300": { ... }, ... },
    "adam": { "100": { "0.1000": 0.57, ... }, "200": { ... }, ... }
  },
  "maxherding": { "adam": { ... } },   # some methods only have one optimizer
  "random":     { "sgd": { ... }, "adam": { ... } }
}

The `optimizer` level (sgd / adam) distinguishes the NoAA fit that produced the
nu' value: "sgd" is the Tomer-style SGD+Nesterov optimizer, "adam" is the
daphna_lab default. There is NO default optimizer anywhere in this module — every
query and write MUST name one explicitly, so a run can never silently mix the two
(see train_al's --nu_optimizer). A missing (al_method, optimizer, budget) returns
None rather than falling back to the other optimizer.

Keys at every level are strings so they round-trip through JSON cleanly.
Floats are stored with 6 decimal places.

Older cache files were nested {al_method: {budget: {tau}}} (implicitly adam).
_method_cache() detects that legacy shape and reads it as the adam column so a
half-migrated file never crashes; new writes always use the nested schema.

Public API
----------
  load_cache(dataset)                              -> dict
  save_cache(dataset, cache)
  upsert(dataset, al_method, optimizer, budget, tau, nu_prime)     # write one point
  lookup_nu_for_tau(dataset, al_method, optimizer, budget, tau)    -> float (interp)
  lookup_tau_for_nu(dataset, al_method, optimizer, budget, nu)     -> float (interp)
  available_optimizers(dataset, al_method)         -> list[str]
  available_budgets(dataset, al_method, optimizer) -> list[int]
  available_taus(dataset, al_method, optimizer, budget) -> list[float]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
import numpy as np

# ── repo root ─────────────────────────────────────────────────────────────────
_repo_root = str(Path(__file__).resolve().parents[3])

_DATASETS_WITH_OWN_DIR = {"cifar10", "cifar100", "imagenet50"}


def _cache_path(dataset: str) -> Path:
    ds = dataset.lower()
    results_dir = Path(_repo_root) / "results" / ds
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "nu_prime_cache.json"


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_cache(dataset: str) -> dict:
    """Return the full cache dict for *dataset*; empty dict if not found."""
    p = _cache_path(dataset)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def save_cache(dataset: str, cache: dict) -> None:
    """Overwrite the cache file for *dataset* with *cache*."""
    p = _cache_path(dataset)
    with open(p, "w") as f:
        json.dump(cache, f, indent=2)


# ── schema helpers ──────────────────────────────────────────────────────────

_OPTIMIZERS = ("sgd", "adam")


def _is_legacy_method(method_dict: dict) -> bool:
    """True if *method_dict* is the pre-optimizer shape {budget: {tau}}.

    Legacy keys are budget strings (all digits); the new shape's top keys are
    optimizer names (sgd/adam). An empty dict is treated as new-schema.
    """
    if not method_dict:
        return False
    return all(str(k).isdigit() for k in method_dict.keys())


def _method_cache(cache: dict, al_method: str, optimizer: str) -> dict:
    """Return the {budget: {tau: nu'}} sub-dict for (al_method, optimizer).

    Empty dict if that method/optimizer combination has no data. A legacy
    (un-migrated) method dict is served only when *optimizer* == "adam".
    """
    md = cache.get(al_method.lower(), {})
    if _is_legacy_method(md):
        return md if optimizer.lower() == "adam" else {}
    return md.get(optimizer.lower(), {})


# ── single-point write ────────────────────────────────────────────────────────

def upsert(dataset: str, al_method: str, optimizer: str, budget: int,
           tau: float, nu_prime: float) -> None:
    """Insert or update one (al_method, optimizer, budget, tau) → nu_prime entry."""
    cache = load_cache(dataset)
    m = al_method.lower()
    o = optimizer.lower()
    b = str(int(budget))
    t = f"{tau:.4f}"
    md = cache.setdefault(m, {})
    if _is_legacy_method(md):
        # Migrate this method in place: its existing budgets are adam values.
        cache[m] = {"adam": md}
        md = cache[m]
    md.setdefault(o, {}).setdefault(b, {})[t] = round(float(nu_prime), 6)
    save_cache(dataset, cache)


# ── helpers ───────────────────────────────────────────────────────────────────

def available_optimizers(dataset: str, al_method: str) -> list[str]:
    """Optimizers with data for *al_method* (e.g. ['adam', 'sgd'])."""
    md = load_cache(dataset).get(al_method.lower(), {})
    if _is_legacy_method(md):
        return ["adam"] if md else []
    return sorted(o for o in md if md.get(o))


def available_budgets(dataset: str, al_method: str, optimizer: str) -> list[int]:
    md = _method_cache(load_cache(dataset), al_method, optimizer)
    return sorted(int(b) for b in md.keys())


def available_taus(dataset: str, al_method: str, optimizer: str,
                   budget: int) -> list[float]:
    md = _method_cache(load_cache(dataset), al_method, optimizer)
    b = str(int(budget))
    return sorted(float(t) for t in md.get(b, {}).keys())


# ── interpolated lookups ─────────────────────────────────────────────────────

def lookup_nu_for_tau(dataset: str, al_method: str, optimizer: str,
                      budget: int, tau: float) -> Optional[float]:
    """
    Return nu_prime for the given (al_method, optimizer, budget, tau).

    - Exact match → returned as-is.
    - tau in range → 1-D linear interpolation over stored tau points.
    - tau outside range → nearest endpoint (no extrapolation).
    - budget not found → snap to nearest available budget first.
    Returns None if no data exists for this (al_method, optimizer). There is
    NO fallback to the other optimizer — a miss stays a miss.
    """
    md = _method_cache(load_cache(dataset), al_method, optimizer)
    if not md:
        return None

    # Snap budget
    available = sorted(int(b) for b in md.keys())
    if not available:
        return None
    budget_key = str(min(available, key=lambda b: abs(b - int(budget))))

    tau_dict = md[budget_key]
    if not tau_dict:
        return None

    taus = np.array(sorted(float(t) for t in tau_dict.keys()))
    nus  = np.array([tau_dict[f"{t:.4f}"] for t in taus])

    # Always include oracle anchor (tau=1 → nu_prime=0)
    if 1.0 not in taus:
        taus = np.append(taus, 1.0)
        nus  = np.append(nus,  0.0)
        idx  = np.argsort(taus)
        taus, nus = taus[idx], nus[idx]

    if len(taus) == 1:
        return float(nus[0])

    return float(np.interp(tau, taus, nus))


def lookup_tau_for_nu(dataset: str, al_method: str, optimizer: str,
                      budget: int, nu: float) -> Optional[float]:
    """
    Inverse of lookup_nu_for_tau: given a desired nu_prime, find the
    corresponding tau via linear interpolation.

    nu_prime is monotonically *decreasing* in tau, so we flip the arrays.
    Returns None if no data exists for this (al_method, optimizer). No
    fallback to the other optimizer.
    """
    md = _method_cache(load_cache(dataset), al_method, optimizer)
    if not md:
        return None

    available = sorted(int(b) for b in md.keys())
    if not available:
        return None
    budget_key = str(min(available, key=lambda b: abs(b - int(budget))))

    tau_dict = md[budget_key]
    if not tau_dict:
        return None

    taus = np.array(sorted(float(t) for t in tau_dict.keys()))
    nus  = np.array([tau_dict[f"{t:.4f}"] for t in taus])

    # Always include oracle anchor (tau=1 → nu_prime=0)
    if 1.0 not in taus:
        taus = np.append(taus, 1.0)
        nus  = np.append(nus,  0.0)
        idx  = np.argsort(taus)
        taus, nus = taus[idx], nus[idx]

    if len(taus) == 1:
        return float(taus[0])

    # nus is decreasing in taus; flip so np.interp gets an increasing xp
    return float(np.interp(nu, nus[::-1], taus[::-1]))
