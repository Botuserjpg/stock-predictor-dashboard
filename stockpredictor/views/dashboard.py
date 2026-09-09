"""Dashboard blueprint: post-login landing page with market overview."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from flask import Blueprint, render_template
from flask_login import current_user, login_required

from ..services import alerts as alerts_service
from ..services import stocks
from ..services.auth import user_store

bp = Blueprint("dashboard", __name__)

_MAX_WORKERS = 6


def _fetch_live(symbol: str) -> Dict[str, Any]:
    """One watchlist symbol's live quote + recent closes for a sparkline."""
    quote: Dict[str, Any] = {}
    try:
        quote = stocks.get_quote(symbol)
    except Exception:
        pass
    closes: List[float] = []
    try:
        history = stocks.get_price_history(symbol, period="1mo", limit=22)
        closes = [round(float(row["Close"]), 2) for row in history if row.get("Close")]
    except Exception:
        pass
    return {"symbol": symbol, "quote": quote, "closes": closes}


@bp.get("/dashboard")
@login_required
def home():
    """Render the dashboard with indices, watchlist and cached predictions."""
    email = current_user.email
    return _render_dashboard(email)


@bp.get("/demo")
def demo():
    """Read-only public dashboard preview for portfolio and résumé deployments."""
    return _render_dashboard("Guest", demo_mode=True)


def _render_dashboard(email: str, demo_mode: bool = False):
    """Build the dashboard data for an authenticated user or the public demo."""

    try:
        indices = stocks.get_market_indices()
    except Exception:
        indices = {}

    watchlist = ["AAPL", "MSFT", "NVDA", "SPY"] if demo_mode else user_store.watchlists().get(email, [])
    live_watchlist: List[Dict[str, Any]] = []
    movers: Dict[str, List[Dict[str, Any]]] = {"gainers": [], "losers": []}
    if watchlist:
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(watchlist))) as pool:
            live_watchlist = list(pool.map(_fetch_live, watchlist))

        with_changes = [
            item for item in live_watchlist
            if item["quote"].get("change_pct") is not None
        ]
        with_changes.sort(key=lambda item: item["quote"]["change_pct"], reverse=True)
        movers["gainers"] = with_changes[:4]
        movers["losers"] = sorted(with_changes, key=lambda item: item["quote"]["change_pct"])[:4]

    predictions = (
        {
            "AAPL": {"predicted_price": 232.40, "confidence": 0.74, "signal": "BUY", "sentiment": "Positive"},
            "MSFT": {"predicted_price": 468.10, "confidence": 0.68, "signal": "BUY", "sentiment": "Neutral"},
        }
        if demo_mode
        else user_store._state.get("prediction_cache", {})
    )
    history = [] if demo_mode else user_store.analysis_history().get(email, [])
    alerts = [] if demo_mode else alerts_service.list_alerts(email)

    return render_template(
        "dashboard.html",
        indices=indices,
        watchlist=watchlist,
        live_watchlist=live_watchlist,
        movers=movers,
        predictions=predictions,
        history=history,
        alerts=alerts,
        email=email,
        demo_mode=demo_mode,
    )
