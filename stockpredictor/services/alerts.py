"""Price-alert service: per-user threshold alerts evaluated against live quotes."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .auth import user_store

logger = logging.getLogger("stockpredictor.services.alerts")

MAX_ALERTS_PER_USER = 20


def list_alerts(email: str) -> List[Dict[str, Any]]:
    return list(user_store.alerts().get((email or "").strip().lower(), []))


def add_alert(
    email: str, symbol: str, condition: str, threshold: float
) -> Dict[str, Any]:
    """Create an alert; returns ``{success, message, alert?}``."""
    email = (email or "").strip().lower()
    symbol = (symbol or "").strip().upper()
    condition = (condition or "above").strip().lower()
    if condition not in {"above", "below"}:
        return {"success": False, "message": "Condition must be 'above' or 'below'"}
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return {"success": False, "message": "Threshold must be a number"}
    if not symbol or threshold <= 0:
        return {"success": False, "message": "A valid symbol and threshold are required"}

    alerts = user_store.alerts()
    items = alerts.setdefault(email, [])
    if len(items) >= MAX_ALERTS_PER_USER:
        return {"success": False, "message": f"Limit of {MAX_ALERTS_PER_USER} alerts reached"}

    alert = {
        "id": uuid.uuid4().hex[:10],
        "symbol": symbol,
        "condition": condition,
        "threshold": round(threshold, 4),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_triggered": None,
    }
    items.append(alert)
    user_store.persist()
    return {"success": True, "message": f"Alert set: {symbol} {condition} {threshold:.2f}", "alert": alert}


def remove_alert(email: str, alert_id: str) -> Dict[str, Any]:
    email = (email or "").strip().lower()
    alerts = user_store.alerts()
    items = alerts.get(email, [])
    remaining = [a for a in items if a.get("id") != alert_id]
    if len(remaining) == len(items):
        return {"success": False, "message": "Alert not found"}
    alerts[email] = remaining
    user_store.persist()
    return {"success": True, "message": "Alert removed"}


def triggered_alerts(email: str, prices: Dict[str, Optional[float]]) -> List[Dict[str, Any]]:
    """Return alerts whose condition is met by the given current prices.

    Triggered alerts are NOT auto-removed (users dismiss them); the timestamp
    is updated so the UI can show when it fired.
    """
    email = (email or "").strip().lower()
    items = user_store.alerts().get(email, [])
    fired = []
    changed = False
    for alert in items:
        price = prices.get(alert.get("symbol"))
        if price is None:
            continue
        threshold = float(alert.get("threshold", 0))
        hit = price >= threshold if alert.get("condition") == "above" else price <= threshold
        if hit:
            fired.append({**alert, "current_price": round(price, 2)})
            if alert.get("last_triggered") is None:
                alert["last_triggered"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
    if changed:
        user_store.persist()
    return fired
