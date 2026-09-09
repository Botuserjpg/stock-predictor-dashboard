"""Market-regime detection and model auto-selection (Feature: Regime detection).

Classifies the recent price series into ``trending-up`` / ``trending-down`` /
``ranging`` / ``volatile`` and uses that (plus recent forecast accuracy) to
recommend which forecasting model to run next.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _closes(rows) -> np.ndarray:
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
        raise ValueError("rows must be prices, dicts, or a DataFrame")


def _adx_ish(closes: np.ndarray, period: int = 14) -> float:
    """Simplified directional-movement strength score in the 0..100 range."""
    if len(closes) < period + 2:
        return 0.0
    up = np.diff(closes)
    down = -up
    up[up < 0] = 0.0
    down[down < 0] = 0.0
    tr = np.abs(np.diff(closes))
    gains = np.convolve(up, np.ones(period) / period, mode="valid")
    losses = np.convolve(down, np.ones(period) / period, mode="valid")
    di = np.nan_to_num((gains - losses) / np.maximum(gains + losses, 1e-9))
    strength = float(np.nanmean(np.abs(di))) * 100.0
    return round(min(100.0, max(0.0, strength)), 2)


def detect_regime(closes: Sequence[float], lookback: int = 120) -> Dict[str, Any]:
    """Classify the current market regime from raw close prices."""
    prices = _closes(closes)
    clean = prices[np.isfinite(prices)]
    if len(clean) < 30:
        return {"success": False, "message": "Not enough data to detect regime"}

    use = clean[-lookback:] if lookback else clean
    returns = np.diff(np.log(np.maximum(use, 1e-9)))
    annualized_vol = float(returns.std(ddof=0)) * np.sqrt(252)
    recent = returns[-20:] if len(returns) >= 20 else returns
    trend = float(np.polyfit(np.arange(len(use)), use, 1)[0])
    trend_pct = trend / float(np.nanmean(use)) if float(np.nanmean(use)) else 0.0
    adx = _adx_ish(use)
    score = float(np.nanmean(recent)) if len(recent) else 0.0

    if annualized_vol >= 0.35:
        regime = "volatile"
        reason = "Very high realized volatility"
    elif adx >= 25 and trend_pct >= 0.001:
        regime = "trending-up"
        reason = "Strong positive directional movement"
    elif adx >= 25 and trend_pct <= -0.001:
        regime = "trending-down"
        reason = "Strong negative directional movement"
    else:
        regime = "ranging"
        reason = "Weak trend, prices mean-reverting"

    return {
        "success": True,
        "regime": regime,
        "reason": reason,
        "adx_strength": adx,
        "annualized_volatility": round(annualized_vol * 100, 2),
        "trend_slope_pct": round(trend_pct * 100, 4),
        "recent_daily_return": round(score * 100, 3),
        "n_observations": int(len(use)),
        "recommended": "Dampen exposure to this symbol" if regime in ("volatile", "trending-down") else "Trend-following approach is viable",
    }


MODEL_RANKINGS: Dict[str, Dict[str, int]] = {
    "trending-up": {"LSTM": 1, "Prophet": 2, "GRU": 3, "ARIMA": 4},
    "trending-down": {"LSTM": 2, "Prophet": 1, "GRU": 3, "ARIMA": 4},
    "ranging": {"ARIMA": 1, "GRU": 2, "Prophet": 3, "LSTM": 4},
    "volatile": {"GRU": 1, "LSTM": 2, "Prophet": 3, "ARIMA": 4},
}


def auto_select_model(
    closes: Sequence[float],
    recent_accuracy: Optional[float] = None,
    model_availability: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Pick the best forecast model for the current regime.

    ``recent_accuracy`` is the rolling % accuracy of recent forecasts and is
    used to promote a strong incumbent; ``model_availability`` filters out
    models that cannot run in this environment.
    """
    regime = detect_regime(closes)
    if not regime.get("success"):
        return {"success": False, "message": regime.get("message", "Cannot select model")}

    available = model_availability or {}
    all_models = ("LSTM", "GRU", "Prophet", "ARIMA")
    candidates = [m for m in all_models if available.get(m, True)]

    rankings = MODEL_RANKINGS.get(regime["regime"], MODEL_RANKINGS["ranging"])
    ranked = sorted(candidates, key=lambda m: rankings.get(m, 99))
    selected = ranked[0] if ranked else "AUTO"

    rationale = (
        f"{regime['regime']} regime (ADX {regime['adx_strength']}, "
        f"vol {regime['annualized_volatility']}%) favours {selected}."
    )
    if recent_accuracy is not None and recent_accuracy >= 0.6:
        selected = "AUTO"
        rationale = (
            f"Recent forecasts were {recent_accuracy * 100:.0f}% accurate, "
            "so the incumbent auto-stack is kept."
        )

    return {
        "success": True,
        "selected": selected,
        "regime": regime["regime"],
        "rationale": rationale,
        "candidates": ranked,
    }
