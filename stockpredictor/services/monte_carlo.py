"""Monte Carlo simulation service (Feature: Monte Carlo).

Simulates thousands of forward price paths from historical log-returns and
answers questions like "how likely is the price to hit my target before my
stop" and "what does the 95% confidence band look like".
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _close_prices(rows) -> np.ndarray:
    if isinstance(rows, np.ndarray):
        return np.asarray(rows, dtype=float)
    if isinstance(rows, list) or isinstance(rows, tuple):
        if rows and isinstance(rows[0], dict):
            key = "Close" if "Close" in rows[0] else "close"
            out = []
            for row in rows:
                try:
                    out.append(float(row.get(key)))
                except (TypeError, ValueError):
                    continue
            return np.asarray(out, dtype=float)
        return np.asarray(list(rows), dtype=float)
    try:
        col = "Close" if "Close" in rows.columns else "close"
        return rows[col].to_numpy(dtype=float)
    except AttributeError:
        raise ValueError("rows must be a sequence of prices, dicts, or a DataFrame")


def _daily_params(closes: np.ndarray, lookback: int = 120) -> Dict[str, float]:
    clean = closes[np.isfinite(closes)]
    if len(clean) < 10:
        return {"mu": 0.0, "sigma": 0.02, "last": float(clean[-1]) if len(clean) else 1.0}
    returns = np.diff(np.log(np.maximum(clean, 1e-9)))
    use = returns[-lookback:] if lookback else returns
    mu = float(use.mean())
    sigma = float(use.std(ddof=0))
    return {"mu": mu, "sigma": max(sigma, 1e-4), "last": float(clean[-1])}


def simulate_paths(
    closes: Sequence[float],
    horizon: int = 30,
    n_paths: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return an array shaped ``(n_paths, horizon + 1)`` of simulated paths."""
    prices = _close_prices(closes)
    horizon = int(max(1, min(horizon, 500)))
    n_paths = int(max(1, min(n_paths, 20000)))
    params = _daily_params(prices)
    last = params["last"]
    dt = 1.0
    rng = np.random.default_rng(seed)
    shocks = rng.normal(
        (params["mu"] - 0.5 * params["sigma"] ** 2) * dt,
        params["sigma"] * math.sqrt(dt),
        size=(n_paths, horizon),
    )
    log_paths = np.zeros((n_paths, horizon + 1))
    log_paths[:, 0] = math.log(max(last, 1e-9))
    np.cumsum(shocks, axis=1, out=log_paths[:, 1:])
    return np.exp(log_paths)


def simulate(
    closes: Sequence[float],
    horizon: int = 30,
    n_paths: int = 1000,
    seed: Optional[int] = None,
    target: Optional[float] = None,
    stop: Optional[float] = None,
) -> Dict[str, Any]:
    """Full Monte Carlo summary: percentile bands + target/stop probabilities.

    When ``target`` and/or ``stop`` are given, the paths are checked for first
    touch so "probability of hitting target before stop" is meaningful.
    """
    prices = _close_prices(closes)
    if len(prices) < 10:
        return {"success": False, "message": "Not enough price history to simulate"}
    paths = simulate_paths(prices, horizon, n_paths, seed=seed)
    last = float(prices[-1])
    bands = {"lower5": [], "median": [], "upper95": []}
    for step in range(paths.shape[1]):
        col = np.sort(paths[:, step])
        bands["lower5"].append(round(float(col[int(len(col) * 0.05)]), 2))
        bands["median"].append(round(float(col[int(len(col) * 0.50)]), 2))
        bands["upper95"].append(round(float(col[int(len(col) * 0.95)]), 2))

    result: Dict[str, Any] = {
        "success": True,
        "last_price": round(last, 2),
        "horizon": horizon,
        "n_paths": n_paths,
        "bands": bands,
        "mean_final": round(float(paths[:, -1].mean()), 2),
        "prob_above_current": round(float((paths[:, -1] > last).mean() * 100), 2),
        "prob_below_current": round(float((paths[:, -1] < last).mean() * 100), 2),
    }

    if target is not None or stop is not None:
        hit_target = 0
        hit_stop = 0
        reached = 0
        for path in paths:
            touched_target = touched_stop = False
            if target is not None:
                touched_target = bool((path >= target).any())
            if stop is not None:
                touched_stop = bool((path <= stop).any())
            if touched_target or touched_stop:
                reached += 1
            if touched_target:
                hit_target += 1
            if touched_stop:
                hit_stop += 1
        result["target"] = round(float(target), 2) if target is not None else None
        result["stop"] = round(float(stop), 2) if stop is not None else None
        result["prob_hit_target"] = round(hit_target / paths.shape[0] * 100, 2)
        result["prob_hit_stop"] = round(hit_stop / paths.shape[0] * 100, 2)
        result["prob_hit_either"] = round(reached / paths.shape[0] * 100, 2)
    return result
