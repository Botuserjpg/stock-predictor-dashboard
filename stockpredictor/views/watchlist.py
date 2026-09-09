"""Watchlist blueprint: page + JSON API for per-user symbol lists."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, current_app, jsonify, render_template, request
from flask_login import current_user, login_required

from production_core import audit_log, json_endpoint, sanitize_symbol

from ..services import alerts as alerts_service
from ..services import stocks
from ..services.auth import user_store

bp = Blueprint("watchlist", __name__)

_MAX_WORKERS = 6


def _list_for_user() -> list:
    watchlists = user_store.watchlists()
    return watchlists.setdefault(current_user.email, [])


def _fetch_quote(symbol: str) -> dict:
    quote = {}
    try:
        quote = stocks.get_quote(symbol)
    except Exception:
        pass
    return {
        "symbol": symbol,
        "price": quote.get("price"),
        "change_pct": quote.get("change_pct"),
    }


@bp.get("/watchlist")
@login_required
def page():
    """Render the watchlist page with live prices for each symbol.

    Quotes are fetched concurrently (each is a network call) so the page stays
    responsive even with many symbols.
    """
    symbols = list(_list_for_user())
    if symbols:
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(symbols))) as pool:
            items = list(pool.map(_fetch_quote, symbols))
    else:
        items = []

    prices = {item["symbol"]: item["price"] for item in items}
    prior_triggered = {a.get("id") for a in alerts_service.list_alerts(current_user.email) if a.get("last_triggered")}
    triggered = alerts_service.triggered_alerts(current_user.email, prices)
    new_fired = [a for a in triggered if a.get("id") not in prior_triggered]
    if new_fired:
        try:
            from ..services.push_alerts import notify_triggered

            notify_triggered(new_fired, recipient=current_user.email)
        except Exception:  # pragma: no cover - push channels must not break the page
            pass

    return render_template(
        "watchlist.html",
        watchlist=items,
        symbols=symbols,
        alerts=alerts_service.list_alerts(current_user.email),
        triggered=triggered,
    )


@bp.post("/api/watchlist/add")
@login_required
@json_endpoint
def api_add():
    data = request.get_json(silent=True) or request.form
    symbol = sanitize_symbol(data.get("symbol", ""))
    if not symbol:
        return jsonify({"success": False, "message": "No symbol provided"})

    items = _list_for_user()
    if symbol in items:
        return jsonify({"success": False, "message": f"{symbol} is already in your watchlist"})

    items.append(symbol)
    user_store.persist()
    audit_log("watchlist.add", current_user.email, {"symbol": symbol})
    return jsonify({"success": True, "message": f"{symbol} added to watchlist",
                    "watchlist": items})


@bp.post("/api/watchlist/remove")
@login_required
@json_endpoint
def api_remove():
    data = request.get_json(silent=True) or request.form
    symbol = sanitize_symbol(data.get("symbol", ""))
    items = _list_for_user()
    if symbol not in items:
        return jsonify({"success": False, "message": "Symbol not found in watchlist"})

    items.remove(symbol)
    user_store.persist()
    audit_log("watchlist.remove", current_user.email, {"symbol": symbol})
    return jsonify({"success": True, "message": f"{symbol} removed from watchlist",
                    "watchlist": items})


@bp.get("/api/watchlist/get")
@login_required
def api_get():
    items = _list_for_user()
    return jsonify({"success": True, "watchlist": items, "count": len(items)})


@bp.get("/api/watchlist/quotes")
@login_required
@json_endpoint
def api_watchlist_quotes():
    """Live quotes for the current user's watchlist (REST fallback for realtime).

    Mirrors the Socket.IO ``watchlist_quotes`` payload so clients can degrade
    from websockets to polling without changing their rendering logic.
    """
    from production_core import utc_now

    from ..services.realtime import collect_quotes

    quotes = collect_quotes(
        current_app, current_user.email,
        int(current_app.config.get("REALTIME_MAX_SYMBOLS", 20)),
    )
    return jsonify({"success": True, "quotes": quotes, "updated_at": utc_now()})


@bp.post("/api/alerts/add")
@login_required
@json_endpoint
def api_alert_add():
    data = request.get_json(silent=True) or request.form
    result = alerts_service.add_alert(
        current_user.email,
        data.get("symbol", ""),
        data.get("condition", "above"),
        data.get("threshold", 0),
    )
    if result.get("success"):
        audit_log("alert.add", current_user.email, {
            "symbol": result.get("alert", {}).get("symbol"),
            "condition": result.get("alert", {}).get("condition"),
        })
    return jsonify(result)


@bp.post("/api/alerts/remove")
@login_required
@json_endpoint
def api_alert_remove():
    data = request.get_json(silent=True) or request.form
    result = alerts_service.remove_alert(current_user.email, data.get("alert_id", ""))
    if result.get("success"):
        audit_log("alert.remove", current_user.email, {})
    return jsonify(result)
