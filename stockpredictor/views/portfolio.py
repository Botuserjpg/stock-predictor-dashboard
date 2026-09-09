"""Portfolio blueprint: paper-trading page + JSON API."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, current_app, jsonify, render_template, request
from flask_login import current_user, login_required

from production_core import json_endpoint, sanitize_symbol

from ..services import sentiment as sentiment_service
from ..services import stocks
from ..services.portfolio import DEFAULT_CASH, equity_curve, get_portfolio, save_portfolio, score_position_signal

bp = Blueprint("portfolio", __name__)

_MAX_WORKERS = 6


def _request_data() -> dict:
    return request.get_json(silent=True) or dict(request.form)


def _resolve_price(symbol: str, requested: float) -> float:
    if requested and requested > 0:
        return requested
    try:
        return float(stocks.get_current_price(symbol))
    except Exception:
        return 0.0


def _fetch_position(symbol: str) -> dict:
    """Fetch everything the page needs for one held symbol, degrading to None."""
    result = {"symbol": symbol, "price": None, "ta": None,
              "sentiment": None, "closes": []}
    try:
        result["price"] = float(stocks.get_current_price(symbol))
    except Exception:
        pass
    try:
        result["ta"] = stocks.get_technical_analysis(symbol)
    except Exception:
        pass
    try:
        result["sentiment"] = sentiment_service.get_sentiment(symbol)
    except Exception:
        pass
    try:
        history = stocks.get_price_history(symbol, period="3mo", limit=30) or []
        closes = []
        for row in history:
            try:
                close = float(row.get("Close"))
            except (TypeError, ValueError):
                continue
            if close > 0:
                closes.append(close)
        result["closes"] = closes[-30:]
    except Exception:
        pass
    return result


@bp.get("/portfolio")
@login_required
def page():
    """Render the paper-trading dashboard with live valuations + AI insights."""
    portfolio = get_portfolio(current_user.email)
    positions = portfolio.positions

    fetched = {}
    symbols = list(positions.keys())
    if symbols:
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(symbols))) as pool:
            for info in pool.map(_fetch_position, symbols):
                fetched[info["symbol"]] = info

    rows = []
    cost_basis = 0.0
    market_value = 0.0
    for symbol, pos in positions.items():
        qty = float(pos.get("quantity", 0))
        avg = float(pos.get("avg_price", 0))
        info = fetched.get(symbol) or {}
        price = info.get("price")
        value = (price or avg) * qty
        cost_basis += avg * qty
        market_value += value
        rows.append({
            "symbol": symbol,
            "quantity": qty,
            "avg_price": avg,
            "price": price,
            "value": round(value, 2),
            "closes": info.get("closes", []),
            "unrealized": round((price or avg) - avg, 2) if price else None,
            "unrealized_pct": round(((price or avg) - avg) / avg * 100, 2) if price and avg else None,
            "weight": 0.0,
        })

    total = portfolio.cash + market_value
    for row in rows:
        row["weight"] = round(row["value"] / total * 100, 1) if total else 0.0

    # Net capital contributed: initial virtual cash plus deposits minus
    # withdrawals. This is the correct baseline for measuring total P&L.
    net_contributions = DEFAULT_CASH
    for entry in portfolio.history:
        if entry.get("type") == "DEPOSIT":
            net_contributions += float(entry.get("price", 0) or 0)
        elif entry.get("type") == "WITHDRAW":
            net_contributions -= float(entry.get("price", 0) or 0)

    unrealized_pnl = market_value - cost_basis
    unrealized_pct = unrealized_pnl / cost_basis * 100 if cost_basis else 0.0
    total_return = total - net_contributions
    total_return_pct = total_return / net_contributions * 100 if net_contributions else 0.0

    # AI insights: composite score per held position from technicals +
    # sentiment + price momentum, ranked best-first.
    insights = []
    for row in rows:
        info = fetched.get(row["symbol"]) or {}
        scored = score_position_signal(info.get("ta"), info.get("sentiment"),
                                       row["closes"])
        signal = scored["signal"]
        pill = ("pill-red" if signal in ("SELL", "STRONG_SELL")
                else "pill-neutral" if signal == "HOLD" else "pill-green")
        hint = ("Consider selling — price-drop risk"
                if signal in ("SELL", "STRONG_SELL")
                else "Strong outlook — hold or add"
                if signal in ("BUY", "STRONG_BUY")
                else "Hold — no strong signal")
        insights.append({
            "symbol": row["symbol"],
            "score": scored["score"],
            "signal": signal,
            "pill": pill,
            "reasons": scored["reasons"],
            "hint": hint,
            "avg_price": row["avg_price"],
            "price": row["price"],
            "quantity": row["quantity"],
            "stop_loss": round(row["avg_price"] * 0.92, 2),
            "take_profit": round(row["avg_price"] * 1.20, 2),
        })
    insights.sort(key=lambda i: i["score"], reverse=True)

    sell_signals = [i for i in insights if i["signal"] in ("SELL", "STRONG_SELL")]
    top_picks = insights[:3]

    allocation = [{"label": "Cash", "value": round(portfolio.cash, 2)}]
    pnl_rows = []
    for row in rows:
        allocation.append({"label": row["symbol"], "value": round(row["value"], 2)})
        pnl_rows.append({
            "symbol": row["symbol"],
            "pnl": round((row["unrealized"] or 0) * row["quantity"], 2)
            if row["unrealized"] is not None else None,
        })
    equity = equity_curve(portfolio.history, portfolio.cash, portfolio.positions, {
        sym: (fetched.get(sym) or {}).get("price") for sym in symbols
    })
    if equity:
        from datetime import datetime

        equity.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "value": round(total, 2),
        })

    risk = _risk_engine_context(rows, fetched, portfolio.cash, total)

    return render_template(
        "portfolio.html",
        rows=rows,
        cash=portfolio.cash,
        invested=round(cost_basis, 2),
        market_value=round(market_value, 2),
        total=round(total, 2),
        unrealized_pnl=round(unrealized_pnl, 2),
        unrealized_pct=round(unrealized_pct, 2),
        total_return=round(total_return, 2),
        total_return_pct=round(total_return_pct, 2),
        insights=insights,
        sell_signals=sell_signals,
        top_picks=top_picks,
        allocation=allocation,
        pnl_rows=pnl_rows,
        history=list(reversed(portfolio.history[-40:])),
        equity=equity,
        risk=risk,
    )


def _risk_engine_context(rows: list, fetched: dict, cash: float, total: float) -> dict:
    """Compute the portfolio risk dashboard (VaR, correlation, sizing)."""
    import numpy as np

    from ..services.risk import correlation_matrix, portfolio_var, rebalance_suggestions, volatility_target_position_size

    returns_by_symbol = {}
    annual_vols = {}
    for row in rows:
        closes = [float(c) for c in (fetched.get(row["symbol"]) or {}).get("closes", []) if c and c > 0]
        if len(closes) >= 10:
            arr = np.asarray(closes, dtype=float)
            rets = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
            returns_by_symbol[row["symbol"]] = rets.tolist()
            annual_vols[row["symbol"]] = float(np.std(rets, ddof=0)) * np.sqrt(252)

    risk = {"success": False, "message": "Add at least two positions to enable the risk engine."}
    if len(returns_by_symbol) >= 1:
        try:
            var = portfolio_var(returns_by_symbol)
            risk = {"var": var} if var.get("success") else {}
        except Exception:
            risk = {}
        if len(returns_by_symbol) >= 2:
            try:
                corr = correlation_matrix(returns_by_symbol)
                if corr.get("success"):
                    risk["correlation"] = corr
            except Exception:
                pass

    sizing = []
    for row in rows:
        symbol = row["symbol"]
        price = row.get("price") or row.get("avg_price") or 0
        if not price or symbol not in annual_vols:
            continue
        try:
            size = volatility_target_position_size(cash, price, annual_vols[symbol])
            sizing.append({**size, "symbol": symbol, "annual_vol_pct": round(annual_vols[symbol] * 100, 1)})
        except Exception:
            continue
    risk["sizing"] = sizing

    try:
        current_weights = {r["symbol"]: (r["weight"] / 100.0) for r in rows}
        equal = 1.0 / len(current_weights) if current_weights else 0.0
        risk["rebalance"] = rebalance_suggestions(
            current_weights, {s: equal for s in current_weights}
        )
    except Exception:
        risk["rebalance"] = []

    risk["n_positions"] = len(rows)
    risk["total"] = round(total, 2)
    return risk


@bp.post("/api/portfolio/trade")
@login_required
@json_endpoint
def api_trade():
    data = _request_data()
    symbol = sanitize_symbol(data.get("symbol", ""))
    trade_type = (data.get("type") or "BUY").upper()
    if trade_type not in {"BUY", "SELL"}:
        return jsonify({"success": False, "message": "Invalid trade type"})

    try:
        quantity = float(data.get("quantity", 0))
    except (TypeError, ValueError):
        quantity = 0
    try:
        price = float(data.get("price", 0) or 0)
    except (TypeError, ValueError):
        price = 0.0

    if not symbol or quantity <= 0:
        return jsonify({"success": False, "message": "Invalid trade parameters"})

    price = _resolve_price(symbol, price)
    if price <= 0:
        return jsonify({"success": False, "message": "Unable to determine current price"})

    portfolio = get_portfolio(current_user.email)
    result = portfolio.buy(symbol, price, quantity) if trade_type == "BUY" \
        else portfolio.sell(symbol, price, quantity)
    if result.get("success"):
        save_portfolio(current_user.email, portfolio)
    return jsonify(result)


@bp.post("/api/portfolio/remove")
@login_required
@json_endpoint
def api_remove():
    data = _request_data()
    symbol = sanitize_symbol(data.get("symbol", ""))
    portfolio = get_portfolio(current_user.email)
    result = portfolio.remove(symbol)
    if result.get("success"):
        save_portfolio(current_user.email, portfolio)
    return jsonify(result)


@bp.post("/api/portfolio/cash")
@login_required
@json_endpoint
def api_cash():
    data = _request_data()
    action = (data.get("action") or "deposit").strip().lower()
    if action not in {"deposit", "withdraw"}:
        return jsonify({"success": False, "message": "Action must be deposit or withdraw"})
    try:
        amount = float(data.get("amount", 0))
    except (TypeError, ValueError):
        amount = 0.0
    portfolio = get_portfolio(current_user.email)
    result = portfolio.deposit(amount) if action == "deposit" else portfolio.withdraw(amount)
    if result.get("success"):
        save_portfolio(current_user.email, portfolio)
    result["cash"] = portfolio.cash
    return jsonify(result)


@bp.post("/api/portfolio/optimize")
@login_required
@json_endpoint
def api_optimize():
    data = _request_data()
    risk = (data.get("risk_tolerance") or "medium").strip().lower()
    budget = float(data.get("budget") or current_app.config.get("DEFAULT_INITIAL_CASH", 10000.0))
    portfolio = get_portfolio(current_user.email)
    result = portfolio.optimize(risk_tolerance=risk, budget=budget)
    return jsonify(result)
