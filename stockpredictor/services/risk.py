"""Portfolio risk engine (Feature: Risk engine).

Position sizing (Kelly / volatility targeting), portfolio VaR, correlation
matrix and rebalancing suggestions. All functions are pure so they work offline
on any return/price history.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    cap: float = 0.25,
) -> Dict[str, Any]:
    """Optimal fraction of capital to risk per trade using the Kelly criterion.

    ``avg_win``/``avg_loss`` are positive numbers expressed as fractions of
    capital (e.g. 0.10 = 10%). A fraction of the full Kelly is recommended.
    """
    if not (0.0 <= win_rate <= 1.0):
        raise ValueError("win_rate must be between 0 and 1")
    if avg_win <= 0 or avg_loss <= 0:
        return {"full_kelly": 0.0, "half_kelly": 0.0, "message": "Need positive win/loss payoffs"}

    b = avg_win / avg_loss
    full = (win_rate * (b + 1) - 1) / b
    full = max(0.0, min(float(full), float(cap)))
    return {
        "full_kelly": round(full, 4),
        "half_kelly": round(full * 0.5, 4),
        "quarter_kelly": round(full * 0.25, 4),
        "payoff_ratio": round(b, 3),
    }


def volatility_target_position_size(
    cash: float,
    price: float,
    annual_volatility: float,
    target_annual_vol: float = 0.15,
    max_fraction: float = 0.25,
) -> Dict[str, Any]:
    """Shares to buy so the position contributes ``target_annual_vol`` to a portfolio."""
    if cash <= 0 or price <= 0:
        raise ValueError("cash and price must be positive")
    if annual_volatility <= 0:
        annual_volatility = 0.02
    capital = min(cash, cash * max_fraction)
    position_value = capital * (target_annual_vol / annual_volatility)
    position_value = min(position_value, cash)
    shares = position_value / price
    return {
        "position_value": round(position_value, 2),
        "shares": round(math.floor(shares * 100) / 100, 2),
        "allocation_pct": round(position_value / cash * 100, 2),
    }


def portfolio_var(
    returns_by_symbol: Dict[str, Sequence[float]],
    weights: Optional[Dict[str, float]] = None,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> Dict[str, Any]:
    """Historical/parametric VaR for a portfolio of return series."""
    if not returns_by_symbol:
        return {"success": False, "message": "No return series provided"}

    aligned = {}
    min_len = min(len(v) for v in returns_by_symbol.values())
    if min_len < 5:
        return {"success": False, "message": "Not enough history for VaR"}
    for symbol, series in returns_by_symbol.items():
        aligned[symbol] = np.asarray(list(series)[-min_len:], dtype=float)

    if weights is None:
        weights = {s: 1.0 / len(aligned) for s in aligned}
    w = np.asarray([weights.get(s, 0.0) for s in aligned], dtype=float)
    w = w / w.sum() if w.sum() else w

    matrix = np.column_stack(list(aligned.values()))
    portfolio_returns = matrix @ w
    quantile = (1 - confidence) if confidence >= 0.5 else confidence
    historical_var = float(np.quantile(portfolio_returns, quantile))
    mean = float(portfolio_returns.mean())
    std = float(portfolio_returns.std(ddof=0))
    z = 1.645 if confidence >= 0.95 else 1.28
    parametric_var = float(mean - z * std)

    scale = math.sqrt(horizon_days)
    return {
        "success": True,
        "confidence": confidence,
        "horizon_days": horizon_days,
        "historical_var_1d": round(historical_var * 100, 2),
        "parametric_var_1d": round(parametric_var * 100, 2),
        "var_95_5d": round(historical_var * scale * 100, 2),
        "expected_return_1d": round(mean * 100, 3),
        "volatility_1d": round(std * 100, 3),
    }


def correlation_matrix(
    returns_by_symbol: Dict[str, Sequence[float]],
) -> Dict[str, Any]:
    """Pearson correlation matrix across symbols (JSON-safe)."""
    symbols = list(returns_by_symbol.keys())
    if len(symbols) < 2:
        return {"success": False, "message": "Need at least two symbols"}

    min_len = min(len(v) for v in returns_by_symbol.values())
    if min_len < 5:
        return {"success": False, "message": "Not enough history for correlation"}

    matrix = np.column_stack([
        np.asarray(list(returns_by_symbol[s])[-min_len:], dtype=float) for s in symbols
    ])
    corr = np.corrcoef(matrix, rowvar=False)
    return {
        "success": True,
        "symbols": symbols,
        "matrix": [[round(float(x), 4) for x in row] for row in corr],
    }


def rebalance_suggestions(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    tolerance_pct: float = 5.0,
) -> List[Dict[str, Any]]:
    """Difference between current and target allocations, flagged when > tolerance."""
    suggestions = []
    all_symbols = set(current_weights) | set(target_weights)
    for symbol in sorted(all_symbols):
        cur = current_weights.get(symbol, 0.0) * 100
        tgt = target_weights.get(symbol, 0.0) * 100
        diff = tgt - cur
        if abs(diff) > tolerance_pct:
            suggestions.append({
                "symbol": symbol,
                "current_pct": round(cur, 2),
                "target_pct": round(tgt, 2),
                "diff_pct": round(diff, 2),
                "action": "BUY" if diff > 0 else "SELL",
            })
    return suggestions
