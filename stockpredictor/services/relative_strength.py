"""Relative strength (Feature: Relative strength).

Compares a symbol's momentum against a benchmark (default SPY) across multiple
windows. The composite RS score ranks symbols, e.g. "AA > 1 means the symbol
outperformed the benchmark over the window".
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_BENCHMARK = "SPY"
DEFAULT_PERIODS = (20, 60, 120)
# Longer windows describe more durable momentum, so they carry more weight in
# the composite RS score.
_WINDOW_WEIGHTS = {20: 0.2, 60: 0.3, 120: 0.5}

logger = logging.getLogger("stockpredictor.services.relative_strength")


def _closes(rows) -> Optional[np.ndarray]:
    if isinstance(rows, (list, tuple)):
        if rows and isinstance(rows[0], dict):
            key = "Close" if "Close" in rows[0] else "close"
            return np.asarray([float(r[key]) for r in rows if r.get(key) is not None], dtype=float)
        return np.asarray(list(rows), dtype=float)
    try:
        col = "Close" if "Close" in rows.columns else "close"
        return rows[col].to_numpy(dtype=float)
    except AttributeError:
        return None


def _return_over_window(closes: np.ndarray, window: int) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    start, end = closes[-window - 1], closes[-1]
    if not start:
        return None
    return float(end / start - 1.0)


def relative_strength(
    symbol_rows,
    benchmark_rows,
    periods: Sequence[int] = DEFAULT_PERIODS,
) -> Dict[str, Any]:
    """Per-window relative strength ratio (symbol return / benchmark return)."""
    symbol = _closes(symbol_rows)
    benchmark = _closes(benchmark_rows)
    if symbol is None or benchmark is None or len(symbol) < 10 or len(benchmark) < 10:
        return {"success": False, "message": "Not enough price history for relative strength"}

    windows = {}
    for window in periods:
        sym_ret = _return_over_window(symbol, window)
        bench_ret = _return_over_window(benchmark, window)
        if sym_ret is None or bench_ret is None:
            continue
        ratio = (1 + sym_ret) / (1 + bench_ret) - 1 if bench_ret > -1 else 0.0
        windows[window] = {
            "symbol_return_pct": round(sym_ret * 100, 2),
            "benchmark_return_pct": round(bench_ret * 100, 2),
            "excess_return_pct": round((sym_ret - bench_ret) * 100, 2),
            "rs": round(ratio, 4),
        }
    if not windows:
        return {"success": False, "message": "Windows shorter than available history"}

    # Composite RS score: weighted average excess return across windows. Being
    # an absolute figure (not normalized per symbol) keeps scores comparable
    # across symbols, which is what the sortable ranking table relies on.
    total_weight = sum(_WINDOW_WEIGHTS.get(w, 1.0) for w in windows) or 1.0
    composite = sum(
        windows[w]["excess_return_pct"] * _WINDOW_WEIGHTS.get(w, 1.0) for w in windows
    ) / total_weight
    return {
        "success": True,
        "windows": {str(w): v for w, v in windows.items()},
        "composite": round(float(composite), 2),
    }


def _history_for(get_history, symbol: str):
    """Fetch history for one symbol (pool target); returns (symbol, rows|None)."""
    try:
        return symbol, get_history(symbol, period="1y")
    except Exception:
        logger.debug("Relative strength: history failed for %s", symbol)
        return symbol, None


def _score_one(symbol: str, history, benchmark_rows) -> Optional[Dict[str, Any]]:
    """Relative-strength row for one symbol; None when data is insufficient."""
    try:
        result = relative_strength(history, benchmark_rows)
    except Exception:
        logger.debug("Relative strength: scoring failed for %s", symbol)
        return None
    if not result.get("success"):
        return None
    return {"symbol": symbol, **result}


def compute_rs_table(
    symbols: Sequence[str],
    benchmark: str = DEFAULT_BENCHMARK,
    get_history: callable = None,
) -> Dict[str, Any]:
    """Build a relative-strength ranking table for a list of symbols.

    ``get_history`` maps symbol -> OHLCV rows; defaults to the live stock data
    service. Symbols with no data are skipped, never fatal.
    """
    from production_core import cached

    from ..services import stocks

    get_history = get_history or stocks.get_price_history
    benchmark_rows = None
    try:
        benchmark_rows = get_history(benchmark, period="1y")
    except Exception:
        logger.warning("Relative strength: benchmark %s unavailable", benchmark)
        benchmark_rows = None
    if benchmark_rows is None:
        return {"success": False, "message": "Could not fetch benchmark data"}

    def _produce():
        rows = []
        with ThreadPoolExecutor(max_workers=min(6, len(symbols))) as pool:
            for symbol, history in pool.map(
                lambda s: _history_for(get_history, s), list(symbols)
            ):
                if history is None:
                    continue
                scored = _score_one(symbol, history, benchmark_rows)
                if scored is not None:
                    rows.append(scored)
        rows.sort(key=lambda r: r.get("composite", 0), reverse=True)
        return {
            "success": True,
            "benchmark": benchmark,
            "periods": list(DEFAULT_PERIODS),
            "rows": rows,
        }

    key = "rs:" + benchmark + ":" + ",".join(sorted(symbols))
    return cached(key, ttl=600, producer=_produce)()
