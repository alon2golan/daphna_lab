"""
nu_prime_cache.py
-----------------
Read / write / query the per-dataset nu_prime cache.

Cache layout (one JSON file per dataset, e.g. results/cifar100/nu_prime_cache.json):
{
  "probcover": {
    "100": { "0.10": 0.57, "0.20": 0.39, ... },
    "200": { ... },
    ...
  },
  "maxherding": { ... },
  "random":     { ... }
}

Keys at every level are strings so they round-trip through JSON cleanly.
Floats are stored with 6 decimal places.

Public API
----------
  load_cache(dataset)                       -> dict
  save_cache(dataset, cache)
  upsert(dataset, al_method, budget, tau, nu_prime)   # write one point
  lookup_nu_for_tau(dataset, al_method, budget, tau)  -> float  (interpolated)
  lookup_tau_for_nu(dataset, al_method, budget, nu)   -> float  (interpolated)
  available_budgets(dataset, al_method)     -> list[int]
  available_taus(dataset, al_method, budget)-> list[float]
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


# ── single-point write ────────────────────────────────────────────────────────

def upsert(dataset: str, al_method: str, budget: int, tau: float,
           nu_prime: float) -> None:
    """Insert or update one (al_method, budget, tau) → nu_prime entry."""
    cache = load_cache(dataset)
    m = al_method.lower()
    b = str(int(budget))
    t = f"{tau:.4f}"
    cache.setdefault(m, {}).setdefault(b, {})[t] = round(float(nu_prime), 6)
    save_cache(dataset, cache)


# ── helpers ───────────────────────────────────────────────────────────────────

def available_budgets(dataset: str, al_method: str) -> list[int]:
    cache = load_cache(dataset)
    m = al_method.lower()
    return sorted(int(b) for b in cache.get(m, {}).keys())


def available_taus(dataset: str, al_method: str, budget: int) -> list[float]:
    cache = load_cache(dataset)
    m = al_method.lower()
    b = str(int(budget))
    return sorted(float(t) for t in cache.get(m, {}).get(b, {}).keys())


# ── interpolated lookups ─────────────────────────────────────────────────────

def lookup_nu_for_tau(dataset: str, al_method: str, budget: int,
                      tau: float) -> Optional[float]:
    """
    Return nu_prime for the given (al_method, budget, tau).

    - Exact match → returned as-is.
    - tau in range → 1-D linear interpolation over stored tau points.
    - tau outside range → nearest endpoint (no extrapolation).
    - budget not found → snap to nearest available budget first.
    Returns None if no data exists at all.
    """
    cache = load_cache(dataset)
    m = al_method.lower()
    if m not in cache:
        return None

    # Snap budget
    available = sorted(int(b) for b in cache[m].keys())
    if not available:
        return None
    budget_key = str(min(available, key=lambda b: abs(b - int(budget))))

    tau_dict = cache[m][budget_key]
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


def lookup_tau_for_nu(dataset: str, al_method: str, budget: int,
                      nu: float) -> Optional[float]:
    """
    Inverse of lookup_nu_for_tau: given a desired nu_prime, find the
    corresponding tau via linear interpolation.

    nu_prime is monotonically *decreasing* in tau, so we flip the arrays.
    Returns None if no data exists.
    """
    cache = load_cache(dataset)
    m = al_method.lower()
    if m not in cache:
        return None

    available = sorted(int(b) for b in cache[m].keys())
    if not available:
        return None
    budget_key = str(min(available, key=lambda b: abs(b - int(budget))))

    tau_dict = cache[m][budget_key]
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
