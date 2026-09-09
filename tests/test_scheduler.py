"""Tests for the background alert scheduler (evaluation + cooldown + dispatch).

State persistence is redirected to a throwaway directory BEFORE importing the
package (same pattern as test_stockpredictor.py). No network is used: the
price fetcher and notifier are injected.
"""
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import production_core as pc

_TMP_DIR = tempfile.mkdtemp(prefix="stockpredictor_scheduler_")
pc.STATE_DIR = Path(_TMP_DIR)
pc.STATE_FILE = pc.STATE_DIR / "app_state.json"

from stockpredictor.services.auth import user_store  # noqa: E402
from stockpredictor.services.scheduler import evaluate_alerts, start, stop  # noqa: E402

for _SMTP_KEY in ("SMTP_HOST", "SMTP_USER"):
    os.environ.pop(_SMTP_KEY, None)

_NOW = datetime(2026, 8, 11, 12, 0, 0)


def _alert(alert_id="1", symbol="AAPL", condition="above", threshold=150.0, **overrides):
    alert = {
        "id": alert_id,
        "symbol": symbol,
        "condition": condition,
        "threshold": threshold,
        "created": "2026-01-01 00:00:00",
        "last_triggered": None,
    }
    alert.update(overrides)
    return alert


class AlertEvaluationTests(unittest.TestCase):
    def setUp(self):
        user_store.alerts().clear()
        user_store.persist()

    def _notify_recorder(self, calls):
        def notify(fired, recipient):
            calls.append({"fired": fired, "recipient": recipient})
            return {"delivered": ["email"]}
        return notify

    def test_above_threshold_fires_and_notifies(self):
        user_store.alerts()["a@x.com"] = [_alert()]
        user_store.persist()
        calls = []
        result = evaluate_alerts(
            now=_NOW, price_fetcher=lambda s: 160.0,
            notify=self._notify_recorder(calls),
        )
        self.assertEqual(result["fired"], 1)
        self.assertEqual(result["notified"], 1)
        self.assertEqual(result["delivered"], 1)
        self.assertEqual(calls[0]["recipient"], "a@x.com")
        self.assertEqual(calls[0]["fired"][0]["symbol"], "AAPL")
        self.assertEqual(calls[0]["fired"][0]["current_price"], 160.0)
        stored = user_store.alerts()["a@x.com"][0]
        self.assertTrue(stored["last_triggered"])
        self.assertEqual(stored["last_notified"], "2026-08-11 12:00:00")

    def test_below_condition_uses_le(self):
        user_store.alerts()["a@x.com"] = [_alert(condition="below")]
        user_store.persist()
        calls = []
        result = evaluate_alerts(
            now=_NOW, price_fetcher=lambda s: 140.0,
            notify=self._notify_recorder(calls),
        )
        self.assertEqual(result["fired"], 1)
        self.assertEqual(len(calls), 1)

    def test_unmet_condition_does_not_notify(self):
        user_store.alerts()["a@x.com"] = [_alert()]
        user_store.persist()
        calls = []
        result = evaluate_alerts(
            now=_NOW, price_fetcher=lambda s: 100.0,
            notify=self._notify_recorder(calls),
        )
        self.assertEqual(result["fired"], 0)
        self.assertEqual(result["notified"], 0)
        self.assertEqual(calls, [])

    def test_cooldown_suppresses_repeat_notification(self):
        user_store.alerts()["a@x.com"] = [_alert(
            last_triggered="2026-08-11 11:00:00",
            last_notified="2026-08-11 11:50:00",
        )]
        user_store.persist()
        calls = []
        result = evaluate_alerts(
            now=_NOW, price_fetcher=lambda s: 160.0,
            cooldown_seconds=3600,
            notify=self._notify_recorder(calls),
        )
        self.assertEqual(result["fired"], 1)
        self.assertEqual(result["notified"], 0)
        self.assertEqual(calls, [])
        self.assertEqual(user_store.alerts()["a@x.com"][0]["last_notified"],
                         "2026-08-11 11:50:00")

    def test_cooldown_expired_notifies_again(self):
        user_store.alerts()["a@x.com"] = [_alert(
            last_triggered="2026-08-11 10:00:00",
            last_notified="2026-08-11 10:00:00",
        )]
        user_store.persist()
        calls = []
        result = evaluate_alerts(
            now=_NOW, price_fetcher=lambda s: 160.0,
            cooldown_seconds=3600,
            notify=self._notify_recorder(calls),
        )
        self.assertEqual(result["notified"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(user_store.alerts()["a@x.com"][0]["last_notified"],
                         "2026-08-11 12:00:00")

    def test_missing_price_skips_alert_and_counts_error(self):
        user_store.alerts()["a@x.com"] = [_alert()]
        user_store.persist()
        calls = []
        result = evaluate_alerts(
            now=_NOW, price_fetcher=lambda s: None,
            notify=self._notify_recorder(calls),
        )
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["fired"], 0)
        self.assertEqual(calls, [])

    def test_notify_failure_does_not_abort_run(self):
        user_store.alerts()["a@x.com"] = [_alert()]
        user_store.persist()

        def boom(fired, recipient):
            raise RuntimeError("channel down")

        result = evaluate_alerts(
            now=_NOW, price_fetcher=lambda s: 160.0, notify=boom,
        )
        self.assertEqual(result["notified"], 1)
        self.assertEqual(result["delivered"], 0)
        self.assertEqual(user_store.alerts()["a@x.com"][0]["last_notified"],
                         "2026-08-11 12:00:00")

    def test_multiple_users_and_symbols(self):
        user_store.alerts()["a@x.com"] = [_alert(alert_id="1", symbol="AAPL")]
        user_store.alerts()["b@x.com"] = [
            _alert(alert_id="2", symbol="MSFT", threshold=50.0),
            _alert(alert_id="3", symbol="NVDA", threshold=500.0),
        ]
        user_store.persist()
        prices = {"AAPL": 160.0, "MSFT": 60.0, "NVDA": 400.0}
        calls = []
        result = evaluate_alerts(
            now=_NOW, price_fetcher=lambda s: prices[s],
            notify=self._notify_recorder(calls),
        )
        self.assertEqual(result["symbols"], 3)
        self.assertEqual(result["fired"], 2)  # NVDA below 500 → not hit
        self.assertEqual(len(calls), 2)
        recipients = sorted(c["recipient"] for c in calls)
        self.assertEqual(recipients, ["a@x.com", "b@x.com"])


class SchedulerStartTests(unittest.TestCase):
    """start() is config-gated and idempotent (never runs in tests)."""

    def test_start_is_noop_when_disabled(self):
        from stockpredictor import create_app

        app = create_app({"TESTING": True, "SECRET_KEY": "test-secret",
                          "ENABLE_ALERT_SCHEDULER": False})
        start(app)
        self.assertFalse(getattr(app, "_alert_scheduler_started", False))

    def test_start_is_idempotent(self):
        from stockpredictor import create_app

        app = create_app({"TESTING": True, "SECRET_KEY": "test-secret",
                          "ENABLE_ALERT_SCHEDULER": True,
                          "ALERT_SCHEDULER_BACKEND": "thread",
                          "ALERT_POLL_SECONDS": 999})
        start(app)
        self.assertTrue(app._alert_scheduler_started)
        thread1 = app._alert_scheduler
        start(app)
        self.assertIs(app._alert_scheduler, thread1)
        stop(app)
        self.assertFalse(app._alert_scheduler_started)


if __name__ == "__main__":
    unittest.main()
