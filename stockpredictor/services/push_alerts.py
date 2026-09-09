"""Push alert dispatch (Feature: Push alerts).

When a price alert triggers, fan the message out to every configured channel:
Telegram (bot), Discord (webhook) and email (existing SMTP service). Every
channel degrades gracefully so a missing/invalid credential never crashes the
alert pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("stockpredictor.services.push_alerts")


def _settings():
    from flask import current_app

    return current_app.config


def send_telegram(message: str) -> Dict[str, Any]:
    """Send a message via the Telegram Bot API."""
    import requests

    cfg = _settings()
    token = cfg.get("TELEGRAM_BOT_TOKEN")
    chat_id = cfg.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return {"success": False, "channel": "telegram", "message": "Telegram not configured"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
    if resp.status_code == 200:
        return {"success": True, "channel": "telegram"}
    return {"success": False, "channel": "telegram", "message": f"HTTP {resp.status_code}"}


def send_discord(message: str) -> Dict[str, Any]:
    """Send a message to a Discord channel via its webhook URL."""
    import requests

    webhook = _settings().get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return {"success": False, "channel": "discord", "message": "Discord not configured"}
    resp = requests.post(webhook, json={"content": message[:1900]}, timeout=10)
    if resp.status_code in (200, 204):
        return {"success": True, "channel": "discord"}
    return {"success": False, "channel": "discord", "message": f"HTTP {resp.status_code}"}


def send_email(subject: str, body: str, recipient: str) -> Dict[str, Any]:
    """Send an alert email through the existing SMTP service."""
    try:
        from .mail import send_alert_email

        ok = send_alert_email(subject, body, recipient)
        return {"success": bool(ok), "channel": "email"}
    except Exception as exc:  # pragma: no cover - SMTP env dependent
        logger.warning("Email alert failed: %s", exc)
        return {"success": False, "channel": "email", "message": str(exc)}


def notify_triggered(fired: List[Dict[str, Any]], recipient: str = "") -> Dict[str, Any]:
    """Fan out a batch of triggered alerts to every configured channel."""
    if not fired:
        return {"success": True, "delivered": []}

    lines = []
    for alert in fired:
        symbol = alert.get("symbol", "?")
        condition = alert.get("condition", "")
        threshold = alert.get("threshold", 0)
        price = alert.get("current_price", 0)
        lines.append(f"🔔 {symbol} {condition} {threshold} — now {price}")
    text = "Stock Predictor Pro\n" + "\n".join(lines)

    delivered = []
    results = []
    for sender in (send_telegram, send_discord):
        try:
            result = sender(text)
            results.append(result)
            if result.get("success"):
                delivered.append(result["channel"])
        except Exception as exc:  # noqa: BLE001 - channel errors must not break others
            logger.warning("Push channel failed: %s", exc)
            results.append({"success": False, "channel": getattr(sender, "__name__", "?"), "message": str(exc)})

    if recipient:
        try:
            email_result = send_email("Stock alert triggered", text, recipient)
            results.append(email_result)
            if email_result.get("success"):
                delivered.append("email")
        except Exception as exc:  # noqa: BLE001
            results.append({"success": False, "channel": "email", "message": str(exc)})

    return {"success": bool(delivered), "delivered": delivered, "results": results}


def send_test_notification(channel: str = "all") -> Dict[str, Any]:
    """Send a test message to a specific channel (used by the settings UI)."""
    text = "🧪 Test notification from Stock Predictor Pro — push alerts configured correctly."
    channel = (channel or "all").lower()
    if channel == "telegram":
        return send_telegram(text)
    if channel == "discord":
        return send_discord(text)
    return notify_triggered([{
        "symbol": "TEST",
        "condition": "above",
        "threshold": 1.0,
        "current_price": 1.0,
    }])
