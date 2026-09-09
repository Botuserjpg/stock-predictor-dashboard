"""Multi-symbol comparison blueprint.

Compares quotes, technical indicators and sentiment across 2-5 symbols and
renders a normalized overlay chart so relative momentum is easy to eyeball.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from flask import Blueprint, render_template, request
from flask_login import login_required

from production_core import sanitize_symbol

from ..services import sentiment as sentiment_service
from ..services import stocks

bp = Blueprint("compare", __name__)

MAX_SYMBOLS = 5
_MIN_SYMBOLS = 2

# Distinct line colours per series (theme-independent hex values).
SERIES_COLORS = ["#38bdf8", "#a78bfa", "#f5c542", "#2dd4bf", "#fb7185"]


def _normalize(series: List[Dict[str, Any]], base: float = 100.0) -> List[float]:
    """Rebase a price history to ``base`` so different tickers are comparable."""
    closes = []
    for row in series:
        try:
            closes.append(float(row.get("Close")))
        except (TypeError, ValueError):
            continue
    if not closes:
        return []
    first = closes[0] or 1.0
    return [round(base * (c / first), 2) for c in closes]


def _collect(symbol: str) -> Dict[str, Any]:
    """Gather quote, technicals and sentiment for one symbol (pool target)."""
    quote: Dict[str, Any] = {}
    technical: Dict[str, Any] = {}
    sentiment: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    try:
        quote = stocks.get_quote(symbol)
    except Exception:
        pass
    try:
        technical = stocks.get_technical_analysis(symbol, period="6mo")
    except Exception:
        pass
    try:
        sentiment = sentiment_service.get_sentiment(symbol)
    except Exception:
        pass
    try:
        history = stocks.get_price_history(symbol, period="3mo", limit=60)
    except Exception:
        pass

    closes = _normalize(history)
    return {
        "symbol": symbol,
        "quote": quote,
        "technical": technical,
        "sentiment": sentiment,
        "history": history,
        "closes": closes,
    }


def _flatten(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bundle per-symbol data into the structures the template consumes."""
    symbols = [item["symbol"] for item in items]
    row = {
        "symbols": symbols,
        "colors": {item["symbol"]: SERIES_COLORS[i % len(SERIES_COLORS)]
                   for i, item in enumerate(items)},
        "quotes": {item["symbol"]: item["quote"] for item in items},
        "technical": {item["symbol"]: item["technical"] for item in items},
        "sentiment": {item["symbol"]: item["sentiment"] for item in items},
        "series": [
            {
                "symbol": item["symbol"],
                "color": SERIES_COLORS[i % len(SERIES_COLORS)],
                "labels": [row.get("Date", "") for row in item["history"]],
                "values": item["closes"],
            }
            for i, item in enumerate(items)
        ],
        "errors": [item["symbol"] for item in items
                   if not item["quote"] and not item["technical"]],
    }

    # The overlay chart needs every series on the same X axis; keep only the
    # union of labels that all series share (aligned by index from 0).
    min_len = min((len(s["values"]) for s in row["series"] if s["values"]),
                  default=0)
    for s in row["series"]:
        s["values"] = s["values"][:min_len] if min_len else []
        s["labels"] = s["labels"][:min_len] if min_len else []
    return row


def _parse_symbols(raw: str) -> List[str]:
    """Split and sanitize a comma/space separated symbol string."""
    seen: List[str] = []
    for token in str(raw or "").replace(",", " ").split():
        try:
            symbol = sanitize_symbol(token)
        except ValueError:
            continue
        if symbol and symbol not in seen:
            seen.append(symbol)
    return seen


@bp.route("/compare", methods=["GET", "POST"])
@login_required
def page():
    """Render the compare form, or the comparison when symbols are submitted."""
    if request.method == "POST":
        raw = request.form.get("symbols", "")
        symbols = _parse_symbols(raw)
        if len(symbols) < _MIN_SYMBOLS:
            return render_template(
                "compare.html",
                error=f"Enter at least {_MIN_SYMBOLS} symbols to compare "
                      "(separated by commas, e.g. AAPL, MSFT).",
                compare=None,
            )
        if len(symbols) > MAX_SYMBOLS:
            symbols = symbols[:MAX_SYMBOLS]

        with ThreadPoolExecutor(max_workers=min(5, len(symbols))) as pool:
            items = list(pool.map(_collect, symbols))

        return render_template(
            "compare.html",
            compare=_flatten(items),
            error=None,
        )

    return render_template("compare.html", compare=None, error=None)
