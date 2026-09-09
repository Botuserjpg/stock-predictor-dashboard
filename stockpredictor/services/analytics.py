"""Market analytics service: indices, sectors, and market-wide signals."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("stockpredictor.services.analytics")


def market_overview() -> Dict[str, Any]:
    try:
        import model_utils as mu

        return {
            "indices": mu.get_market_indices(),
            "sectors": _normalize_sectors(mu.get_sector_performance()),
            "sentiment": mu.get_real_market_sentiment() if hasattr(mu, "get_real_market_sentiment") else {},
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Market overview failed: %s", exc)
        return {"indices": {}, "sectors": {}, "sentiment": {}}


def _normalize_sectors(sectors: Dict[str, Any]) -> Dict[str, float]:
    """Reduce each sector's metrics dict down to its headline change percent.

    ``get_sector_performance`` returns nested dicts per sector; the analytics
    template only needs a single float to render the change column.
    """
    normalized: Dict[str, float] = {}
    for name, value in sectors.items():
        if isinstance(value, dict):
            change = value.get("recent_return")
            if change is None:
                change = value.get("avg_return")
            if change is None:
                change = value.get("momentum", 0.0)
            try:
                normalized[name] = float(change)
            except (TypeError, ValueError):
                normalized[name] = 0.0
        else:
            try:
                normalized[name] = float(value)
            except (TypeError, ValueError):
                normalized[name] = 0.0
    return normalized


def technical_overview(symbols: List[str]) -> Dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    from .stocks import get_technical_analysis

    def _analyze(symbol: str):
        try:
            return symbol, get_technical_analysis(symbol, period="6mo")
        except Exception:
            return symbol, {}

    results = {}
    if symbols:
        with ThreadPoolExecutor(max_workers=min(6, len(symbols))) as pool:
            for symbol, data in pool.map(_analyze, symbols):
                results[symbol] = data
    return results
