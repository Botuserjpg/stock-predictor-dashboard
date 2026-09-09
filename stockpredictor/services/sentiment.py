"""Market sentiment service (news / social signals).

The underlying legacy pipeline hits Twitter, NewsAPI, Finnhub and Yahoo
sequentially and is expensive, so results are TTL-cached (memory + disk) for
30 minutes. The same symbol is analysed up to three times per request in the
monolith pipeline; the cache collapses those duplicate calls into one.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("stockpredictor.services.sentiment")

SENTIMENT_TTL_SECONDS = 30 * 60


def _json_safe(value: Any) -> Any:
    """Recursively coerce tuples/sets to lists so results cache to disk cleanly."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def get_sentiment(symbol: str) -> Dict[str, Any]:
    from production_core import cached

    clean = (symbol or "").strip().upper()

    def _produce():
        import model_utils as mu

        result = mu.get_enhanced_sentiment(clean)
        if not isinstance(result, tuple) or len(result) < 2:
            raise ValueError("Unexpected sentiment result shape")
        score, sources = result
        score = float(score)
        if score > 0.05:
            label = "BULLISH"
        elif score < -0.05:
            label = "BEARISH"
        else:
            label = "NEUTRAL"
        return {
            "score": score,
            "label": label,
            "sources": _json_safe(list(sources)),
        }

    try:
        return cached(f"sentiment:{clean}", ttl=SENTIMENT_TTL_SECONDS, producer=_produce)()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Sentiment analysis failed for %s: %s", clean, exc)
        return {
            "error": "Sentiment analysis unavailable",
            "score": 0.0,
            "label": "NEUTRAL",
        }
