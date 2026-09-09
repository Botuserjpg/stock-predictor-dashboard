"""Portfolio service: positions, trades and optimization per user.

State is shared with the legacy app through production_core persistence.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from production_core import audit_log

from .auth import user_store

logger = logging.getLogger("stockpredictor.services.portfolio")

DEFAULT_CASH = 10000.0
TRANSACTION_FEE = 0.001


class Portfolio:
    """A single user's paper-trading portfolio."""

    def __init__(self, email: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.email = email
        data = data or {}
        self.cash: float = float(data.get("cash", DEFAULT_CASH))
        self.positions: Dict[str, Dict[str, Any]] = data.get("positions", {}) or {}
        self.history: List[Dict[str, Any]] = data.get("history", []) or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cash": round(self.cash, 2),
            "positions": self.positions,
            "history": self.history,
        }

    def buy(self, symbol: str, price: float, quantity: float) -> Dict[str, Any]:
        if quantity <= 0:
            return {"success": False, "message": "Quantity must be positive"}
        cost = quantity * price * (1 + TRANSACTION_FEE)
        if cost > self.cash:
            return {"success": False, "message": "Insufficient cash"}
        self.cash -= cost
        pos = self.positions.get(symbol, {"quantity": 0.0, "avg_price": 0.0})
        new_qty = float(pos["quantity"]) + quantity
        pos["avg_price"] = (
            float(pos["avg_price"]) * float(pos["quantity"]) + quantity * price
        ) / new_qty
        pos["quantity"] = new_qty
        self.positions[symbol] = pos
        self.history.append({
            "type": "BUY", "symbol": symbol, "quantity": quantity,
            "price": price, "date": _now(),
        })
        audit_log("portfolio.buy", self.email, {"symbol": symbol, "quantity": quantity})
        return {"success": True, "message": f"Bought {quantity} shares of {symbol}"}

    def sell(self, symbol: str, price: float, quantity: float) -> Dict[str, Any]:
        pos = self.positions.get(symbol)
        if not pos or float(pos["quantity"]) < quantity:
            return {"success": False, "message": "Not enough shares to sell"}
        revenue = quantity * price * (1 - TRANSACTION_FEE)
        self.cash += revenue
        pos["quantity"] = float(pos["quantity"]) - quantity
        if pos["quantity"] <= 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = pos
        self.history.append({
            "type": "SELL", "symbol": symbol, "quantity": quantity,
            "price": price, "date": _now(),
        })
        audit_log("portfolio.sell", self.email, {"symbol": symbol, "quantity": quantity})
        return {"success": True, "message": f"Sold {quantity} shares of {symbol}"}

    def remove(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self.positions:
            return {"success": False, "message": "Symbol not in portfolio"}
        del self.positions[symbol]
        audit_log("portfolio.remove", self.email, {"symbol": symbol})
        return {"success": True, "message": f"Removed {symbol} from portfolio"}

    def deposit(self, amount: float) -> Dict[str, Any]:
        if amount <= 0:
            return {"success": False, "message": "Amount must be positive"}
        self.cash += amount
        self.history.append({
            "type": "DEPOSIT", "symbol": "CASH", "quantity": None,
            "price": round(amount, 2), "date": _now(),
        })
        audit_log("portfolio.deposit", self.email, {"amount": amount})
        return {"success": True, "message": f"Deposited ${amount:,.2f}"}

    def withdraw(self, amount: float) -> Dict[str, Any]:
        if amount <= 0:
            return {"success": False, "message": "Amount must be positive"}
        if amount > self.cash:
            return {"success": False, "message": "Insufficient cash"}
        self.cash -= amount
        self.history.append({
            "type": "WITHDRAW", "symbol": "CASH", "quantity": None,
            "price": round(amount, 2), "date": _now(),
        })
        audit_log("portfolio.withdraw", self.email, {"amount": amount})
        return {"success": True, "message": f"Withdrew ${amount:,.2f}"}

    def optimize(self, risk_tolerance: str = "medium", budget: float = DEFAULT_CASH) -> Dict[str, Any]:
        try:
            from predict import get_enhanced_portfolio_recommendations

            recs = get_enhanced_portfolio_recommendations(
                budget=budget, risk_tolerance=risk_tolerance
            )
            return {"success": True, "recommendations": recs}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Portfolio optimize failed: %s", exc)
            return {"success": False, "message": "Optimization unavailable"}


def _now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def equity_curve(
    history: List[Dict[str, Any]],
    current_cash: float,
    positions: Optional[Dict[str, Any]] = None,
    live_prices: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Reconstruct an approximate portfolio value timeline from transactions.

    Walks the history newest-to-oldest, undoing each trade from the current
    cash/position state, so the balance and holdings at every market event are
    rebuilt without any price lookups. Each position is valued at the most
    recent transaction price for that symbol (falling back to ``live_prices``
    for positions that were never traded in the window).

    Cash events (DEPOSIT / WITHDRAW) are not plotted individually; their
    cumulative effect is folded into the reconstructed balance.

    Returns a list of ``{"date", "value"}`` ordered oldest-to-newest, or ``[]``
    when the history contains no trades.
    """
    live_prices = live_prices or {}
    events = [e for e in history if e.get("type") in ("BUY", "SELL")]
    if not events:
        return []

    cash = float(current_cash)
    quantities: Dict[str, float] = {
        str(symbol): float(data.get("quantity", 0))
        for symbol, data in (positions or {}).items()
    }
    last_price: Dict[str, float] = {}
    curve: List[Dict[str, Any]] = []

    for entry in reversed(events):
        symbol = entry.get("symbol") or ""
        trade_type = entry.get("type")
        try:
            qty = float(entry.get("quantity") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        try:
            price = float(entry.get("price") or 0)
        except (TypeError, ValueError):
            price = 0.0

        if trade_type == "BUY":
            cash += qty * price * (1 + TRANSACTION_FEE)
            quantities[symbol] = quantities.get(symbol, 0.0) - qty
        else:  # SELL
            cash -= qty * price * (1 - TRANSACTION_FEE)
            quantities[symbol] = quantities.get(symbol, 0.0) + qty

        if qty:
            last_price[symbol] = price

        value = cash
        for sym, q in quantities.items():
            if q > 0:
                value += q * last_price.get(sym, live_prices.get(sym, 0.0))
        curve.append({"date": entry.get("date", ""), "value": round(value, 2)})

    curve.reverse()
    return curve


def get_portfolio(email: str) -> Portfolio:
    portfolios = user_store.portfolios()
    return Portfolio(email, portfolios.get(email, {}))


def score_position_signal(
    technical: Optional[Dict[str, Any]] = None,
    sentiment: Optional[Dict[str, Any]] = None,
    closes: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Combine technicals + sentiment (+ price momentum) into a 0-100 score.

    Returns a dict with ``score``, ``signal`` (STRONG_SELL .. STRONG_BUY) and a
    short human-readable ``reasons`` list. Pure and offline-friendly: any of the
    inputs may be missing, in which case the remaining signals carry the score.
    """
    score = 50.0
    reasons: List[str] = []

    if technical:
        rsi = technical.get("rsi")
        if rsi is not None:
            try:
                rsi = float(rsi)
            except (TypeError, ValueError):
                rsi = None
        if rsi is not None:
            if rsi >= 70:
                score -= 15
                reasons.append(f"RSI {rsi:.0f} overbought")
            elif rsi <= 30:
                score += 15
                reasons.append(f"RSI {rsi:.0f} oversold")
            elif rsi > 55:
                score += 5
            elif rsi < 45:
                score -= 5

        trend = (technical.get("trend") or "NEUTRAL").upper()
        if trend == "BULLISH":
            score += 12
            reasons.append("Uptrend")
        elif trend == "BEARISH":
            score -= 12
            reasons.append("Downtrend")

        macd = (technical.get("macd_signal") or "NEUTRAL").upper()
        if macd == "BULLISH":
            score += 10
            reasons.append("MACD bullish")
        elif macd == "BEARISH":
            score -= 10
            reasons.append("MACD bearish")

        volatility = (technical.get("volatility_level") or "MEDIUM").upper()
        if volatility == "HIGH":
            score -= 6
            reasons.append("High volatility")

        momentum = technical.get("momentum_score")
        if momentum is not None:
            try:
                score += max(-10.0, min(10.0, float(momentum) / 5.0))
            except (TypeError, ValueError):
                pass

    if sentiment:
        label = (sentiment.get("label") or "NEUTRAL").upper()
        if label == "BULLISH":
            score += 8
            reasons.append("Positive sentiment")
        elif label == "BEARISH":
            score -= 8
            reasons.append("Negative sentiment")

    if not technical and closes and len(closes) >= 2:
        first, last = float(closes[0]), float(closes[-1])
        if first > 0:
            momentum_pct = (last - first) / first * 100
            if momentum_pct >= 5:
                score += 8
                reasons.append(f"Positive momentum ({momentum_pct:+.1f}%)")
            elif momentum_pct <= -5:
                score -= 8
                reasons.append(f"Negative momentum ({momentum_pct:+.1f}%)")

    score = max(0.0, min(100.0, round(score, 1)))
    if score >= 75:
        signal = "STRONG_BUY"
    elif score >= 60:
        signal = "BUY"
    elif score >= 45:
        signal = "HOLD"
    elif score >= 30:
        signal = "SELL"
    else:
        signal = "STRONG_SELL"

    return {"score": score, "signal": signal, "reasons": reasons[:5]}


def save_portfolio(email: str, portfolio: Portfolio) -> None:
    portfolios = user_store.portfolios()
    portfolios[email] = portfolio.to_dict()
    user_store.persist()
