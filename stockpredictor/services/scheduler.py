"""Background price-alert worker (Feature: Scheduled alerts).

A periodic job evaluates every user's alerts against live quotes, updates the
persisted ``last_triggered`` / ``last_notified`` timestamps and fans
notifications out through the push channels (Telegram/Discord/email) with a
per-alert cooldown so a persistent condition does not re-notify on every tick.

Runs on APScheduler when available (default), otherwise falls back to a daemon
thread. Controlled by the ``ENABLE_ALERT_SCHEDULER`` / ``ALERT_POLL_SECONDS`` /
``ALERT_COOLDOWN_SECONDS`` settings.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from . import alerts as alerts_service
from .auth import user_store
from . import push_alerts

logger = logging.getLogger("stockpredictor.services.scheduler")

DEFAULT_COOLDOWN_SECONDS = 3600

_TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def _parse_timestamp(text: str) -> Optional[datetime]:
    """Best-effort parse of the persisted timestamp strings."""
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except (TypeError, ValueError):
            continue
    try:
        parsed = datetime.fromisoformat(str(text))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except (TypeError, ValueError):
        return None


def _default_price_fetcher(symbol: str) -> Optional[float]:
    """Live price for a symbol via the TTL-cached quote service."""
    from .stocks import get_quote

    try:
        quote = get_quote(symbol) or {}
    except Exception:
        return None
    price = quote.get("price")
    if price is None:
        return None
    try:
        return float(price)
    except (TypeError, ValueError):
        return None


def evaluate_alerts(
    now: Optional[datetime] = None,
    price_fetcher: Optional[Callable[[str], Optional[float]]] = None,
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
    notify: Optional[Callable[[List[Dict[str, Any]], str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Evaluate all users' alerts against live prices and dispatch notifications.

    Pure and side-effect-friendly: ``price_fetcher`` and ``notify`` are
    injectable so tests can avoid the network. Persisted state (``last_triggered``
    / ``last_notified``) is updated through :class:`UserStore`.

    Returns a summary dict ``{symbols, fired, notified, delivered, errors}``.
    """
    price_fetcher = price_fetcher or _default_price_fetcher
    notify = notify or push_alerts.notify_triggered
    now = now or datetime.now()
    now_iso = now.strftime("%Y-%m-%d %H:%M:%S")

    alerts_map = user_store.alerts()
    symbols = sorted({
        str(alert.get("symbol"))
        for items in alerts_map.values()
        for alert in items
        if alert.get("symbol")
    })

    prices: Dict[str, float] = {}
    error_count = 0
    for symbol in symbols:
        try:
            price = price_fetcher(symbol)
        except Exception:
            price = None
        if price is None:
            error_count += 1
            continue
        prices[symbol] = float(price)

    fired_total = 0
    notified: List[Dict[str, Any]] = []
    changed = False

    for email, items in list(alerts_map.items()):
        fired = alerts_service.triggered_alerts(email, prices)
        if not fired:
            continue
        fired_by_id = {str(alert.get("id")): alert for alert in fired}
        for alert in items:
            if str(alert.get("id")) not in fired_by_id:
                continue
            fired_total += 1
            last_notified = _parse_timestamp(alert.get("last_notified"))
            if last_notified is not None and (now - last_notified).total_seconds() < cooldown_seconds:
                continue

            fired_alert = fired_by_id[str(alert.get("id"))]
            try:
                result = notify([fired_alert], recipient=email) or {}
            except Exception as exc:  # noqa: BLE001 - push must not kill the job
                logger.warning("Alert notification failed for %s: %s", email, exc)
                result = {"delivered": []}
            alert["last_notified"] = now_iso
            changed = True
            notified.append({
                "email": email,
                "symbol": alert.get("symbol"),
                "delivered": list(result.get("delivered", [])),
            })

    if changed:
        user_store.persist()

    return {
        "symbols": len(symbols),
        "fired": fired_total,
        "notified": len(notified),
        "delivered": sum(1 for n in notified if n["delivered"]),
        "errors": error_count,
    }


def _run_evaluation(app) -> Dict[str, Any]:
    """Run one evaluation inside the app context (used by the job)."""
    cooldown = int(app.config.get("ALERT_COOLDOWN_SECONDS", DEFAULT_COOLDOWN_SECONDS))
    with app.app_context():
        try:
            return evaluate_alerts(cooldown_seconds=cooldown)
        except Exception as exc:  # noqa: BLE001 - never kill the scheduler
            app.logger.warning("Alert evaluation failed: %s", exc)
            return {"symbols": 0, "fired": 0, "notified": 0, "delivered": 0, "errors": 1}


def _thread_loop(app, interval: float) -> None:
    while True:
        time.sleep(max(1.0, interval))
        try:
            _run_evaluation(app)
        except Exception:  # noqa: BLE001 - keep the loop alive
            logger.warning("Scheduled alert run crashed; continuing", exc_info=True)


def start(app) -> None:
    """Start the periodic alert worker for ``app`` (idempotent, config-gated)."""
    if not app.config.get("ENABLE_ALERT_SCHEDULER"):
        return
    if getattr(app, "_alert_scheduler_started", False):
        return

    interval = int(app.config.get("ALERT_POLL_SECONDS", 300))
    backend = (app.config.get("ALERT_SCHEDULER_BACKEND") or "apscheduler").strip().lower()
    use_apscheduler = backend == "apscheduler"

    if use_apscheduler:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError:
            use_apscheduler = False

    if use_apscheduler:
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            _run_evaluation,
            IntervalTrigger(seconds=max(1, interval)),
            args=[app],
            id="alert_evaluation",
            name="evaluate_price_alerts",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        scheduler.start()
        app._alert_scheduler = scheduler
        app.logger.info("Alert scheduler started (APScheduler, %ss interval)", interval)
    else:
        thread = threading.Thread(
            target=_thread_loop, args=(app, interval), daemon=True, name="alert-scheduler"
        )
        thread.start()
        app._alert_scheduler = thread
        app.logger.info("Alert scheduler started (thread, %ss interval)", interval)

    app._alert_scheduler_started = True


def stop(app) -> None:
    """Stop the alert worker for ``app`` (no-op when never started)."""
    scheduler = getattr(app, "_alert_scheduler", None)
    if scheduler is None:
        return
    try:
        if hasattr(scheduler, "shutdown"):
            scheduler.shutdown(wait=False)
    except Exception:  # noqa: BLE001 - best effort
        pass
    app._alert_scheduler_started = False
