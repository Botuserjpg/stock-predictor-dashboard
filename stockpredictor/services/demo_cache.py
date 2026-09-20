"""Precomputed analysis results for the public ``/demo`` route.

The public demo is served exclusively from JSON files written offline by
``scripts/precompute_demo_cache.py``. Live training on the free-tier worker
runs out of memory and crashes the process mid-request, so the web app never
trains here: ``/demo`` only lists a fixed ticker set and reads their cached
JSON. Out-of-set tickers are rejected with a 400 and are never computed live.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from production_core import CACHE_DIR

logger = logging.getLogger("stockpredictor.services.demo_cache")

# Fixed set of tickers that ship with precomputed analysis. Anything outside
# this set is rejected with a 400 on /demo/analyze — never computed live.
DEMO_TICKERS: List[str] = [
    "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA",
    "META", "NFLX", "AMD", "JPM", "V", "DIS", "SPY", "QQQ",
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
]

# Overridable so tests (and the precompute script) can point elsewhere.
DEMO_CACHE_DIR = Path(os.getenv("DEMO_CACHE_DIR", str(CACHE_DIR / "demo_precomputed")))

DEFAULT_DEMO_PARAMS = {
    "period": "1y",
    "days": 30,
    "risk": "medium",
    "model_type": "AUTO",
}


def list_demo_tickers() -> List[str]:
    """Return the fixed list of tickers that have cached demo results."""
    return list(DEMO_TICKERS)


def demo_cache_path(symbol: str) -> Path:
    clean = (symbol or "").strip().upper()
    return DEMO_CACHE_DIR / f"{clean}.json"


def get_demo_result(symbol: str) -> Dict[str, Any]:
    """Load the precomputed analysis for one demo ticker.

    Raises ``KeyError`` when the ticker is not part of the fixed demo set, and
    ``FileNotFoundError`` when the cached JSON has not been produced yet.
    """
    return demo_read(symbol)


def demo_read(symbol: str) -> Dict[str, Any]:
    clean = (symbol or "").strip().upper()
    if clean not in list_demo_tickers():
        raise KeyError(f"{clean} is not in the precomputed demo set")
    path = demo_cache_path(clean)
    if not path.exists():
        raise FileNotFoundError(f"No precomputed analysis cached for {clean}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        logger.error("Failed to read demo cache %s: %s", path, exc)
        raise FileNotFoundError(f"No precomputed analysis cached for {clean}") from exc


def _coerce_keys(value):
    """Recursively make every dict key JSON-safe.

    pandas Timestamps (and any other non-basestring) used as dict keys crash
    ``json.dump`` (``keys must be str, int, float, bool or None``) even with
    ``default=str``, because the value hook never applies to keys. Financial
    statement periods arrive keyed by ``Timestamp``; normalize them to ISO
    date strings, which the ``ymd`` template filter renders identically.
    """
    if isinstance(value, dict):
        out: Dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, (str, int, float, bool)) or key is None:
                safe_key = key
            elif hasattr(key, "isoformat"):
                safe_key = str(key.isoformat())
            else:
                safe_key = str(key)
            out[safe_key] = _coerce_keys(item)
        return out
    if isinstance(value, list):
        return [_coerce_keys(item) for item in value]
    return value


def demo_write(symbol: str, result: Dict[str, Any]) -> Path:
    """Atomically persist one precomputed demo result (overwrites, never appends)."""
    clean = (symbol or "").strip().upper()
    DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = demo_cache_path(clean)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, default=str, ensure_ascii=False)
    tmp.replace(path)
    return path


def build_demo_result(
    symbol: str,
    period: str = "1y",
    days: int = 30,
    risk: str = "medium",
    model_type: str = "AUTO",
) -> Dict[str, Any]:
    """Run the full prediction pipeline for one symbol.

    Intended for offline use only (``scripts/precompute_demo_cache.py``) — this
    executes live model training and should never run from the web app.
    """
    from ..services import sentiment as sentiment_service
    from ..services import stocks

    report = stocks.predict_forecast(
        symbol, period, days, risk, model_type=model_type,
    )
    if not isinstance(report, dict) or report.get("error"):
        return _coerce_keys({
            "symbol": symbol,
            "error": report.get("error") if isinstance(report, dict) else "Prediction failed",
            "message": report.get("message") if isinstance(report, dict) else "",
            "forecast_rows": report.get("forecast_rows", []) if isinstance(report, dict) else [],
        })

    technical: Dict[str, Any] = {}
    sentiment: Dict[str, Any] = {}
    fundamentals: Dict[str, Any] = {}
    price_history: List[Dict[str, Any]] = []

    def _fetch_technical():
        return stocks.get_technical_analysis(symbol, period=period)

    def _fetch_sentiment():
        return sentiment_service.get_sentiment(symbol)

    def _fetch_fundamentals():
        return stocks.get_company_fundamentals(symbol)

    def _fetch_history():
        return stocks.get_price_history(symbol, period=period)

    futures = {
        "technical": _fetch_technical,
        "sentiment": _fetch_sentiment,
        "fundamentals": _fetch_fundamentals,
        "price_history": _fetch_history,
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        mapping = {pool.submit(fn): key for key, fn in futures.items()}
        for future in mapping:
            key = mapping[future]
            try:
                value = future.result()
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("%s fetch failed for %s: %s", key, symbol, exc)
                continue
            if key == "technical" and isinstance(value, dict):
                technical = value
            elif key == "sentiment" and isinstance(value, dict):
                sentiment = value
            elif key == "fundamentals" and isinstance(value, dict):
                fundamentals = value
            elif key == "price_history" and isinstance(value, list):
                price_history = value

    regime, model_selection, analyst_report = _demo_enrich(
        symbol, price_history, report, technical, sentiment,
    )

    return _coerce_keys({
        "symbol": symbol,
        "params": {"period": period, "days": days, "risk": risk, "model_type": (model_type or "AUTO").upper()},
        "report": report,
        "technical": technical,
        "sentiment": sentiment,
        "fundamentals": fundamentals,
        "price_history": price_history,
        "regime": regime,
        "model_selection": model_selection,
        "analyst_report": analyst_report,
        "source": "precomputed",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })


def _demo_enrich(symbol, price_history, report, technical, sentiment):
    """Mirror the dashboard's regime/model/analyst enrichment with safe fallbacks."""
    from ..services.llm_report import rule_based_summary
    from ..services.regime import auto_select_model, detect_regime
    closes: List[float] = []
    for row in price_history or []:
        try:
            value = float(row.get("Close"))
            closes.append(value)
        except (TypeError, ValueError, AttributeError):
            continue

    regime: Dict[str, Any] = {}
    model_selection: Dict[str, Any] = {}
    if len(closes) >= 30:
        try:
            detail = detect_regime(closes)
            if detail.get("success"):
                detection = auto_select_model(closes)
                model_selection = {**detail, **(detection if detection.get("success") else {})}
                regime = detail.get("regime") or {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Regime detection failed for %s: %s", symbol, exc)

    forecast_summary = report.get("executive_summary") if isinstance(report, dict) else {}
    analyst_report = {"success": True, "provider": "rule_based"}
    try:
        analyst_report["report"] = rule_based_summary(symbol, forecast_summary, technical, sentiment)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Analyst summary failed for %s: %s", symbol, exc)
        analyst_report["report"] = "Summary unavailable at this time."
    return regime, model_selection, analyst_report