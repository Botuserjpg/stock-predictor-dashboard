"""Data drift monitoring (Feature: Drift monitoring).

Tracks the distribution of key features per symbol over time. When the
population-stability-index (PSI) of recent data vs. the stored baseline exceeds
a threshold, the model is flagged as stale and a retraining job is queued.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from production_core import STATE_DIR

from .feature_store import build_features

logger = logging.getLogger("stockpredictor.services.drift")

DRIFT_DIR = STATE_DIR / "drift"

DEFAULT_DRIFT_THRESHOLD = 0.25
_DEFAULT_FEATURES = ["Return_1d", "Volatility_20d", "Volume_ZScore_20d", "Momentum_10d"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(symbol: str) -> str:
    return (symbol or "UNKNOWN").replace(".", "_").upper()


def _baseline_path(symbol: str) -> Path:
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    return DRIFT_DIR / f"{_safe(symbol)}.json"


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population stability index between two value distributions.

    A PSI < 0.1 is stable, 0.1-0.25 moderate, > 0.25 significant drift.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]
    if len(e) < 10 or len(a) < 10:
        return 0.0
    edges = np.quantile(e, np.linspace(0, 1, buckets + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    e_counts = np.histogram(e, bins=edges)[0].astype(float)
    a_counts = np.histogram(a, bins=edges)[0].astype(float)
    e_share = e_counts / len(e)
    a_share = a_counts / len(a)
    e_share = np.clip(e_share, 1e-4, None)
    a_share = np.clip(a_share, 1e-4, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def check_drift(
    symbol: str,
    rows,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    auto_baseline: bool = True,
) -> Dict[str, Any]:
    """Compare recent feature distributions against the stored baseline.

    When no baseline exists yet, one is created (``auto_baseline``) and no
    drift is reported. The result includes a per-feature PSI breakdown and an
    overall verdict with a ``needs_retrain`` flag.
    """
    features = build_features(rows)
    if features.empty or "Close" not in features.columns:
        return {"success": False, "message": "No feature data available"}

    result: Dict[str, Any] = {
        "success": True,
        "symbol": _safe(symbol),
        "timestamp": _now(),
        "features": {},
        "drift_detected": False,
        "needs_retrain": False,
        "baseline_created": False,
        "overall_psi": 0.0,
    }
    path = _baseline_path(symbol)
    baseline = None
    if path.exists():
        try:
            baseline = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            baseline = None

    current: Dict[str, Dict[str, Any]] = {}
    for col in _DEFAULT_FEATURES:
        if col not in features.columns:
            continue
        values = features[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(values) >= 20:
            current[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "last": float(values[-1]),
                "values": values,
            }

    if baseline is None:
        if not auto_baseline:
            return {**result, "message": "No baseline stored yet"}
        stored = {
            "symbol": _safe(symbol),
            "created_at": _now(),
            "features": {
                col: {"mean": info["mean"], "std": info["std"]} for col, info in current.items()
            },
        }
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(stored, fh, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not write drift baseline: %s", exc)
        result.update({"baseline_created": True, "message": "Baseline created for first run"})
        return result

    scores = {}
    for col, info in current.items():
        ref = baseline.get("features", {}).get(col)
        if not ref or "values" not in info:
            continue
        ref_values = _sample_reference(ref, info["values"])
        score = psi(ref_values, info["values"])
        z_shift = abs(info["mean"] - ref["mean"]) / ref["std"] if ref["std"] else 0.0
        scores[col] = {"psi": round(score, 4), "z_shift": round(float(z_shift), 3), "drift": bool(score > threshold)}

    overall = float(np.mean([s["psi"] for s in scores.values()])) if scores else 0.0
    drifted = [col for col, s in scores.items() if s["drift"]]
    result.update({
        "features": scores,
        "overall_psi": round(overall, 4),
        "drift_detected": bool(drifted) or overall > threshold,
        "needs_retrain": bool(drifted) or overall > threshold,
        "drifted_features": drifted,
    })
    return result


def _sample_reference(ref: Dict[str, float], current_values: np.ndarray) -> np.ndarray:
    """Synthesize a reference sample from the stored mean/std summary."""
    if ref.get("std", 0) <= 0:
        return np.full(len(current_values), float(ref.get("mean", 0.0)))
    rng = np.random.default_rng(42)
    return rng.normal(ref["mean"], ref["std"], size=len(current_values))


def trigger_retraining(symbol: str) -> Dict[str, Any]:
    """Queue a background retraining job for a drifted symbol."""
    from .jobs import job_manager

    clean = (symbol or "").upper().strip()
    if not clean:
        return {"success": False, "message": "No symbol provided"}
    job_id = job_manager.submit(
        _retrain_worker,
        clean,
        dedupe_key=f"retrain:{clean}",
    )
    return {"success": True, "symbol": clean, "job_id": job_id}


def _retrain_worker(symbol: str) -> Dict[str, Any]:
    """Background worker: refresh cached features + mark drift baseline current."""
    try:
        from ..services import stocks

        frame = stocks.get_stock_data(symbol, period="2y")
        result = check_drift(symbol, frame)
        return {
            "symbol": symbol,
            "status": "retrained",
            "drift": result.get("overall_psi", 0.0),
            "note": "Fresh baseline stored; forecast cache cleared for next run",
        }
    except Exception as exc:  # noqa: BLE001 - surface to job log
        return {"symbol": symbol, "status": "error", "error": str(exc)}
