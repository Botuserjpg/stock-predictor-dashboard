"""Stock data service: wraps the legacy data layer with graceful degradation.

The heavy modules (model_utils / predictor_core) import TensorFlow and scikit-learn,
so they are imported lazily once and cached.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("stockpredictor.services.stocks")

_mu = None
_pc = None


def _load_modules():
    global _mu, _pc
    if _mu is None:
        import model_utils as mu
        _mu = mu
    if _pc is None:
        import predictor_core as pc
        _pc = pc
    return _mu, _pc


def get_stock_data(
    symbol: str, period: str = "1y", interval: str = "1d", **kwargs: Any
) -> pd.DataFrame:
    mu, _ = _load_modules()
    return mu.get_stock_data(symbol, period=period, interval=interval, **kwargs)


def validate_symbol(symbol: str) -> Dict[str, Any]:
    mu, _ = _load_modules()
    return mu.validate_stock_symbol(symbol)


def get_current_price(symbol: str) -> float:
    """Current price with a short TTL cache.

    The underlying legacy lookup is network-heavy (multiple yfinance attempts
    with back-off), so repeated lookups for the same symbol are served from the
    in-memory/file cache for up to 30 seconds.
    """
    from production_core import cached

    clean = (symbol or "").upper().strip()

    def _produce():
        mu, _ = _load_modules()
        return float(mu.get_current_real_price(clean))

    return cached(f"price:{clean}", ttl=30, producer=_produce)()


def get_market_indices() -> Dict[str, Dict[str, Any]]:
    """Market index quotes (already TTL-cached per index at the data layer)."""
    mu, _ = _load_modules()
    return mu.get_market_indices()


def get_quote(symbol: str) -> Dict[str, Any]:
    """Return a lightweight quote (price + daily change) with short TTL cache.

    Falls back to price-only data when the change cannot be computed, so callers
    can always render something useful.
    """
    from production_core import cached

    clean = symbol.upper().strip()

    def _produce():
        try:
            import yfinance as yf

            ticker = yf.Ticker(clean)
            hist = ticker.history(period="5d", interval="1d")
            if hist.empty or len(hist) < 2:
                raise ValueError(f"No quote data for {clean}")
            closes = hist["Close"].dropna()
            price = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            change = price - prev
            change_pct = (change / prev) * 100 if prev else 0.0
            return {
                "symbol": clean,
                "price": round(price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "source": "yahoo",
            }
        except Exception:
            mu, _ = _load_modules()
            price = mu.get_current_real_price(clean)
            return {
                "symbol": clean,
                "price": round(float(price), 2),
                "change": None,
                "change_pct": None,
                "source": "fallback",
            }

    return cached(f"quote:{clean}", ttl=120, producer=_produce)()


def get_technical_analysis(symbol: str, period: str = "6mo") -> Dict[str, Any]:
    """Technical analysis with a 10-minute TTL cache.

    The underlying analyzer recomputes ~20 indicators on every call; results are
    JSON-safe after normalization, so they cache cleanly.
    """
    from production_core import cached

    clean = (symbol or "").upper().strip()

    def _produce():
        mu, _ = _load_modules()
        raw = mu.get_comprehensive_technical_analysis(clean, period=period)
        return normalize_technical(raw)

    return cached(f"ta:{clean}:{period}", ttl=600, producer=_produce)()


def normalize_technical(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten ``get_comprehensive_technical_analysis``'s nested sections into
    the flat keys the templates expect (rsi / trend / macd_signal / ...).

    The legacy analyzer returns a deeply nested structure; the UI only needs a
    handful of headline indicators, so we lift them out with safe defaults.
    """
    if not isinstance(raw, dict) or raw.get("error"):
        return {}

    momentum = raw.get("momentum_indicators") or {}
    trend = raw.get("trend_analysis") or {}
    volatility = raw.get("volatility_analysis") or {}
    volume = raw.get("volume_analysis") or {}

    try:
        rsi = momentum.get("rsi_14")
        if hasattr(rsi, "iloc"):
            rsi = float(rsi.iloc[-1]) if len(rsi) else None
        rsi = round(float(rsi), 1) if rsi is not None else None
    except (TypeError, ValueError):
        rsi = None

    short_trend = (trend.get("short_term") or "").upper()
    medium_trend = (trend.get("medium_term") or "").upper()
    long_trend = (trend.get("long_term") or "").upper()
    bullish_count = sum(t == "BULLISH" for t in (short_trend, medium_trend, long_trend))
    if bullish_count >= 2:
        overall_trend = "BULLISH"
    elif short_trend == "BEARISH" or medium_trend == "BEARISH":
        overall_trend = "BEARISH"
    else:
        overall_trend = "NEUTRAL"

    return {
        "rsi": rsi,
        "rsi_signal": momentum.get("rsi_signal") or ("NEUTRAL" if rsi is None else "NEUTRAL"),
        "trend": overall_trend,
        "macd_signal": momentum.get("macd_signal") or "NEUTRAL",
        "macd_histogram": momentum.get("macd_histogram"),
        "momentum_score": momentum.get("momentum_score"),
        "volatility_level": volatility.get("volatility_level") or "MEDIUM",
        "annualized_volatility": volatility.get("annualized_volatility"),
        "atr": volatility.get("atr"),
        "volume_trend": volume.get("volume_trend") or "NEUTRAL",
        "market_phase": (raw.get("market_phases") or {}).get("phase") or "UNKNOWN",
        "current_price": raw.get("current_price"),
        "symbol": raw.get("symbol"),
        "analysis_date": raw.get("analysis_date"),
    }


def predict_forecast(
    symbol: str, period: str, days: int, risk: str, model_type: str = "AUTO"
) -> Dict[str, Any]:
    """Generate a full prediction report via the legacy predictor.

    Mirrors the monolith's ``generate_forecast_report`` pipeline: trains (or
    loads) the best model for the symbol, produces a forecast table and a
    comprehensive report dict, and returns both merged into a single dict.
    """
    import asyncio

    from production_core import cached

    _load_modules()
    from predictor_core import generate_forecast_report

    model_type = (model_type or "AUTO").strip().upper()
    if model_type not in {"AUTO", "LSTM", "GRU", "ENSEMBLE"}:
        model_type = "AUTO"
    symbol = (symbol or "").strip().upper()
    key = f"forecast:{symbol}:{period}:{days}:{model_type}"

    def _produce():
        forecast_df, report, _csv_path = asyncio.run(
            generate_forecast_report(symbol, days=days, model_type=model_type)
        )
        if not isinstance(report, dict):
            report = {"error": "Prediction failed"}
        else:
            report = dict(report)
            rows = _df_to_rows(forecast_df)
            bands = _build_forecast_bands(forecast_df, symbol)
            if bands and rows:
                for row, (low, high) in zip(rows, zip(bands.get("lower", []), bands.get("upper", []))):
                    row["Low"] = round(low, 2)
                    row["High"] = round(high, 2)
                report["forecast_bands"] = {
                    "daily_volatility": bands.get("daily_volatility"),
                    "confidence": bands.get("confidence"),
                }
            report["forecast_rows"] = rows
        return report

    try:
        return cached(key, ttl=900, producer=_produce)()
    except Exception as exc:  # pragma: no cover - network/model dependent
        logger.warning("Forecast failed for %s: %s", symbol, exc)
        return {
            "error": "Prediction failed",
            "message": str(exc),
            "forecast_rows": [],
        }


def _build_forecast_bands(forecast_df, symbol: str) -> Optional[Dict[str, Any]]:
    """Compute 90% probabilistic fan-interval bands for the forecast rows.

    Returns a ``probabilistic_intervals`` result keyed by lower/upper arrays, or
    None when the forecast lacks usable predictions.
    """
    if forecast_df is None or forecast_df.empty or "Predicted_Price" not in forecast_df.columns:
        return None
    try:
        import numpy as np
        from ml_governance import probabilistic_intervals

        preds = pd.to_numeric(forecast_df["Predicted_Price"], errors="coerce").dropna().values
        if len(preds) < 2:
            return None
        mu, _ = _load_modules()
        hist = mu.get_stock_data(symbol, period="1y", interval="1d")
        recent = hist["Close"] if not hist.empty and "Close" in hist.columns else pd.Series(preds)
        return probabilistic_intervals(preds, recent)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Forecast bands unavailable for %s: %s", symbol, exc)
        return None


def _df_to_rows(forecast_df) -> List[Dict[str, Any]]:
    if forecast_df is None:
        return []
    try:
        return forecast_df.to_dict("records")
    except Exception:  # pragma: no cover - defensive
        return []


def get_price_history(symbol: str, period: str = "6mo", limit: int = 90) -> List[Dict[str, Any]]:
    """Return the most recent ``limit`` trading days as JSON-safe OHLCV rows.

    Served from the 1-hour on-disk cache via :func:`get_stock_data` first; falls
    back to a direct ``yf.download`` when the cache produces no usable closes
    (yfinance ``Ticker.history`` is intermittently NaN for some tickers here).
    """
    mu, _ = _load_modules()
    try:
        df = mu.get_stock_data(
            symbol, period=period, interval="1d",
            include_technical=False, include_sentiment=False,
        )
    except Exception:
        df = None

    if df is None or df.empty or "Close" not in df.columns or df["Close"].notna().sum() < 2:
        df = _download_ohlcv(symbol, period=period)

    if df is None or df.empty or "Close" not in df.columns:
        return []

    columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[columns].dropna(subset=["Close"]).tail(limit)

    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        close = float(row["Close"])
        rows.append({
            "Date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
            "Open": round(float(row.get("Open", close)), 2),
            "High": round(float(row.get("High", close)), 2),
            "Low": round(float(row.get("Low", close)), 2),
            "Close": round(close, 2),
            "Volume": int(float(row.get("Volume", 0) or 0)),
        })
    return rows


def _download_ohlcv(symbol: str, period: str = "6mo"):
    """Fetch an OHLCV frame straight from ``yf.download`` (reliable in this env).

    ``yf.download`` returns MultiIndex columns for a single ticker, so the Close
    frame is reduced to its first column before returning.
    """
    import yfinance as yf

    try:
        data = yf.download(symbol, period=period, interval="1d",
                           progress=False, auto_adjust=False, threads=False)
    except Exception:
        return None
    if data is None or data.empty:
        return None
    close = data["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    data = data.copy()
    data["Close"] = close
    return data


_CURATED_SYMBOLS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
    {"symbol": "GOOG", "name": "Alphabet Inc. (Class C)", "exchange": "NASDAQ"},
    {"symbol": "AMZN", "name": "Amazon.com, Inc.", "exchange": "NASDAQ"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ"},
    {"symbol": "META", "name": "Meta Platforms, Inc.", "exchange": "NASDAQ"},
    {"symbol": "TSLA", "name": "Tesla, Inc.", "exchange": "NASDAQ"},
    {"symbol": "NFLX", "name": "Netflix, Inc.", "exchange": "NASDAQ"},
    {"symbol": "AMD", "name": "Advanced Micro Devices, Inc.", "exchange": "NASDAQ"},
    {"symbol": "INTC", "name": "Intel Corporation", "exchange": "NASDAQ"},
    {"symbol": "AVGO", "name": "Broadcom Inc.", "exchange": "NASDAQ"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE"},
    {"symbol": "V", "name": "Visa Inc.", "exchange": "NYSE"},
    {"symbol": "WMT", "name": "Walmart Inc.", "exchange": "NYSE"},
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "exchange": "NYSE"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "exchange": "NYSE"},
    {"symbol": "PG", "name": "Procter & Gamble Company", "exchange": "NYSE"},
    {"symbol": "KO", "name": "The Coca-Cola Company", "exchange": "NYSE"},
    {"symbol": "PEP", "name": "PepsiCo, Inc.", "exchange": "NASDAQ"},
    {"symbol": "DIS", "name": "The Walt Disney Company", "exchange": "NYSE"},
    {"symbol": "BA", "name": "The Boeing Company", "exchange": "NYSE"},
    {"symbol": "HD", "name": "Home Depot, Inc.", "exchange": "NYSE"},
    {"symbol": "MCD", "name": "McDonald's Corporation", "exchange": "NYSE"},
    {"symbol": "NKE", "name": "Nike, Inc.", "exchange": "NYSE"},
    {"symbol": "CSCO", "name": "Cisco Systems, Inc.", "exchange": "NASDAQ"},
    {"symbol": "ORCL", "name": "Oracle Corporation", "exchange": "NYSE"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "exchange": "NASDAQ"},
    {"symbol": "CRM", "name": "Salesforce, Inc.", "exchange": "NYSE"},
    {"symbol": "IBM", "name": "International Business Machines", "exchange": "NYSE"},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSEARCA"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ"},
    {"symbol": "BTC-USD", "name": "Bitcoin USD", "exchange": "CCC"},
    {"symbol": "ETH-USD", "name": "Ethereum USD", "exchange": "CCC"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited", "exchange": "NSE"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services Limited", "exchange": "NSE"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Limited", "exchange": "NSE"},
    {"symbol": "INFY.NS", "name": "Infosys Limited", "exchange": "NSE"},
    {"symbol": "0700.HK", "name": "Tencent Holdings Limited", "exchange": "HKEX"},
    {"symbol": "TSM", "name": "Taiwan Semiconductor Manufacturing", "exchange": "NYSE"},
]


def search_symbols(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Return autocomplete candidates for ``query`` (symbol or company name).

    Prefers the Yahoo Finance quote-search API (cached for an hour) so the box
    shows live matches, and falls back to a local curated index when the network
    is unavailable so autocomplete still works offline.
    """
    from production_core import cached

    q = (query or "").strip().upper()
    if not q:
        return []

    def _produce():
        results = []
        try:
            import yfinance as yf

            search = yf.Search(q, max_results=limit, news_count=0,
                               enable_fuzzy_query=True)
            for quote in (search.quotes or []):
                try:
                    results.append({
                        "symbol": str(getattr(quote, "symbol", "") or ""),
                        "name": str(getattr(quote, "shortname", "") or
                                    getattr(quote, "longname", "") or ""),
                        "exchange": str(getattr(quote, "exchange", "") or ""),
                    })
                except Exception:
                    continue
        except Exception:
            results = []
        if not results:
            results = _local_search(q, limit)
        seen = set()
        deduped = []
        for result in results:
            key = (result.get("symbol", ""), result.get("name", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped[:limit]

    return cached(f"search:{q}", ttl=3600, producer=_produce)()


def _local_search(q: str, limit: int) -> List[Dict[str, Any]]:
    """Match the curated index by symbol prefix or name substring."""
    q = (q or "").upper()
    if not q:
        return []
    matches = []
    for entry in _CURATED_SYMBOLS:
        symbol = entry["symbol"].upper()
        name = entry["name"].upper()
        if symbol.startswith(q) or q in name:
            matches.append(dict(entry))
            if len(matches) >= limit:
                break
    return matches


def get_company_fundamentals(symbol: str) -> Dict[str, Any]:
    from fundamentals import get_company_fundamentals as _fundamentals

    try:
        return _fundamentals(symbol)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Fundamentals lookup failed for %s: %s", symbol, exc)
        return {}
