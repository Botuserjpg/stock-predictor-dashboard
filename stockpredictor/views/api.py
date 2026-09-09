"""Public JSON API blueprint: current price, portfolio export and CSV export."""
from __future__ import annotations

import csv
import io
from datetime import datetime

from flask import Blueprint, Response, current_app, jsonify, request
from flask_login import current_user, login_required

from production_core import json_endpoint, peek_cached, sanitize_symbol

from ..services import stocks
from ..services.portfolio import get_portfolio

bp = Blueprint("api", __name__)


def _int_arg(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _history(symbol: str, period: str = "2y", limit: int = 500):
    return stocks.get_price_history(symbol, period=period, limit=limit)


@bp.get("/api/price/<symbol>")
@login_required
@json_endpoint
def api_price(symbol: str):
    clean_symbol = sanitize_symbol(symbol)
    price = float(stocks.get_current_price(clean_symbol))
    return jsonify({"price": price, "symbol": clean_symbol})


@bp.get("/api/quote/<symbol>")
@login_required
@json_endpoint
def api_quote(symbol: str):
    """Full lightweight quote (price, daily change) used by the UI widgets."""
    clean_symbol = sanitize_symbol(symbol)
    if not clean_symbol:
        return jsonify({"success": False, "message": "No symbol provided"}), 400
    return jsonify({"success": True, **stocks.get_quote(clean_symbol)})


@bp.get("/api/search")
@login_required
@json_endpoint
def api_search():
    """Symbol autocomplete: match by ticker prefix or company name."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"success": True, "results": []})
    try:
        limit = _int_arg("limit", 8, 1, 15)
        results = stocks.search_symbols(query, limit=limit)
    except Exception:
        results = []
    return jsonify({"success": True, "results": results})


@bp.get("/api/market/ticker")
@login_required
@json_endpoint
def api_market_ticker():
    """A small slice of global indices for the scrolling market ticker bar."""
    try:
        indices = stocks.get_market_indices()
    except Exception:
        indices = {}
    limited = dict(list(indices.items())[:8])
    return jsonify({"success": True, "indices": limited})


@bp.get("/api/portfolio/export")
@login_required
@json_endpoint
def api_portfolio_export():
    """Export a user's portfolio as JSON (mirrors the CSV export semantics)."""
    portfolio = get_portfolio(current_user.email)

    exported = {
        "email": current_user.email,
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cash": portfolio.cash,
        "positions": portfolio.positions,
        "history": portfolio.history,
    }
    return jsonify(exported)


@bp.get("/api/portfolio/export/csv")
@login_required
def api_portfolio_export_csv():
    """Download a user's transaction history as CSV."""
    portfolio = get_portfolio(current_user.email)

    history = portfolio.history
    if not history:
        return jsonify({"success": False, "message": "No transactions to export"}), 404

    output = io.StringIO()
    fieldnames = ["date", "type", "symbol", "quantity", "price"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for entry in history:
        writer.writerow({key: entry.get(key, "") for key in fieldnames})

    filename = f"{current_user.email.split('@')[0]}_transactions.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@bp.get("/api/analysis/csv")
@login_required
def api_analysis_csv():
    """Download a forecast as CSV straight from the (fresh) forecast cache."""
    try:
        symbol = sanitize_symbol(request.args.get("symbol", ""))
    except ValueError:
        return jsonify({"success": False, "message": "Invalid symbol"}), 400
    if not symbol:
        return jsonify({"success": False, "message": "No symbol provided"}), 400

    period = (request.args.get("period") or "1y").strip() or "1y"
    try:
        days = int(request.args.get("days", 30))
    except (TypeError, ValueError):
        days = 30
    model_type = (request.args.get("model_type") or "AUTO").strip().upper() or "AUTO"

    forecast_key = f"forecast:{symbol}:{period}:{days}:{model_type}"
    report = peek_cached(forecast_key, ttl=900)
    rows = (report or {}).get("forecast_rows") or []
    if not rows:
        return jsonify({"success": False, "message": "No cached forecast to export"}), 404

    output = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

    filename = f"{symbol}_forecast_{period}_{days}d_{model_type}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@bp.get("/api/backtest/<symbol>")
@login_required
@json_endpoint
def api_backtest(symbol: str):
    """Walk-forward backtest of the chosen strategy against historical data."""
    from ..services.backtest import run_backtest

    clean_symbol = sanitize_symbol(symbol)
    strategy = (request.args.get("strategy") or "MA_CROSS").strip().upper()
    fast = _int_arg("fast", 20, 2, 120)
    slow = _int_arg("slow", 50, 2, 250)
    if fast >= slow:
        slow = fast + 1
    history = _history(clean_symbol, period="2y", limit=500)
    if not history:
        return jsonify({"success": False, "message": "No price history to backtest"}), 400

    cfg = current_app.config
    result = run_backtest(
        history,
        initial_cash=float(cfg.get("BACKTEST_DEFAULT_CASH", 10000.0)),
        commission=float(cfg.get("BACKTEST_COMMISSION", 0.001)),
        slippage=float(cfg.get("BACKTEST_SLIPPAGE", 0.0005)),
        strategy=strategy,
        fast=fast,
        slow=slow,
    )
    result["symbol"] = clean_symbol
    return jsonify(result)


@bp.post("/api/scenario/<symbol>")
@login_required
@json_endpoint
def api_scenario(symbol: str):
    """Monte Carlo price simulation for target/stop probabilities."""
    from ..services.monte_carlo import simulate

    clean_symbol = sanitize_symbol(symbol)
    data = request.get_json(silent=True) or {}
    try:
        horizon = int(data.get("horizon", 30))
    except (TypeError, ValueError):
        horizon = 30
    horizon = max(1, min(horizon, 250))
    try:
        n_paths = int(data.get("n_paths", current_app.config.get("MONTE_CARLO_PATHS", 1000)))
    except (TypeError, ValueError):
        n_paths = current_app.config.get("MONTE_CARLO_PATHS", 1000)
    n_paths = max(100, min(n_paths, 5000))

    def _float_or_none(key: str):
        try:
            value = data.get(key)
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    target = _float_or_none("target")
    stop = _float_or_none("stop")

    history = _history(clean_symbol, period="1y", limit=250)
    closes = []
    for row in history:
        try:
            closes.append(float(row["Close"]))
        except (TypeError, ValueError, KeyError):
            continue
    result = simulate(closes, horizon=horizon, n_paths=n_paths,
                      target=target, stop=stop, seed=42)
    result["symbol"] = clean_symbol
    return jsonify(result)


@bp.get("/api/explain/<symbol>")
@login_required
@json_endpoint
def api_explain(symbol: str):
    """Feature attribution for the most recent forecast (SHAP or proxy)."""
    from ..services.explain import explain_forecast

    clean_symbol = sanitize_symbol(symbol)
    history = _history(clean_symbol, period="2y", limit=500)
    if not history:
        return jsonify({"success": False, "message": "No price history to explain"}), 400
    return jsonify(explain_forecast(history, persist=True, symbol=clean_symbol))


@bp.get("/api/drift/<symbol>")
@login_required
@json_endpoint
def api_drift(symbol: str):
    """Feature-drift status for a symbol (PSI vs stored baseline)."""
    from ..services.drift import check_drift

    clean_symbol = sanitize_symbol(symbol)
    history = _history(clean_symbol, period="2y", limit=500)
    if not history:
        return jsonify({"success": False, "message": "No price history for drift check"}), 400
    threshold = float(current_app.config.get("DRIFT_THRESHOLD", 0.25))
    return jsonify(check_drift(clean_symbol, history, threshold=threshold))


@bp.post("/api/drift/<symbol>/retrain")
@login_required
@json_endpoint
def api_drift_retrain(symbol: str):
    """Queue a background retraining job for a drifted symbol."""
    from ..services.drift import trigger_retraining

    clean_symbol = sanitize_symbol(symbol)
    return jsonify(trigger_retraining(clean_symbol))


@bp.get("/api/relative-strength")
@login_required
@json_endpoint
def api_relative_strength():
    """Relative-strength ranking of the user's watchlist vs a benchmark."""
    from ..services.auth import user_store
    from ..services.relative_strength import compute_rs_table

    email = current_user.email
    watchlist = user_store.watchlists().get(email, [])
    if not watchlist:
        return jsonify({"success": True, "benchmark": "SPY", "rows": []})
    return jsonify(compute_rs_table(watchlist[:20], benchmark=request.args.get("benchmark", "SPY") or "SPY"))


@bp.post("/api/report/<symbol>")
@login_required
@json_endpoint
def api_llm_report(symbol: str):
    """Generate an AI analyst report for a symbol (LLM when configured)."""
    from ..services.llm_report import generate_analyst_report

    clean_symbol = sanitize_symbol(symbol)
    body = request.get_json(silent=True) or {}
    forecast = body.get("forecast") or {}
    technical = body.get("technical") or {}
    sentiment = body.get("sentiment") or {}
    return jsonify(generate_analyst_report(clean_symbol, forecast, technical, sentiment))


@bp.post("/api/alerts/test-notification")
@login_required
@json_endpoint
def api_test_notification():
    """Send a test push notification to configured channels."""
    from ..services.push_alerts import send_test_notification

    body = request.get_json(silent=True) or {}
    channel = (body.get("channel") or "all").strip().lower()
    return jsonify(send_test_notification(channel))
