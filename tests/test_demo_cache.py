"""Regression tests for the /demo lockdown and /analyze memory guards.

The public demo must never trigger live training: ``/demo`` and
``/demo/analyze`` only list a fixed ticker set and read precomputed JSON from
``data_cache/demo_precomputed``. ``/analyze`` jobs are capped to one at a time
so concurrent requests get a clean "busy" response instead of OOM-killing the
single worker.
"""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import production_core as pc

_TMP_DIR = tempfile.mkdtemp(prefix="stockpredictor_demo_test_")
pc.STATE_DIR = Path(_TMP_DIR)
pc.STATE_FILE = pc.STATE_DIR / "app_state.json"

from stockpredictor import create_app  # noqa: E402
from stockpredictor.services import demo_cache  # noqa: E402
from stockpredictor.services import jobs  # noqa: E402


def _sample_result(symbol: str = "AAPL") -> dict:
    return {
        "symbol": symbol,
        "params": {"period": "1y", "days": 30, "risk": "medium", "model_type": "AUTO"},
        "source": "precomputed",
        "generated_at": "2026-01-01T00:00:00Z",
        "report": {
            "error": None,
            "executive_summary": {
                "company_name": f"{symbol} Corp",
                "exchange": "NASDAQ",
                "model_used": "AUTO",
                "currency_symbol": "$",
                "current_price": 100.0,
                "predicted_price": 105.5,
                "expected_return": 5.5,
                "confidence_score": 0.72,
                "risk_level": "MEDIUM",
                "investment_recommendation": "BUY",
                "analysis_date": "2026-01-01",
            },
            "forecast_rows": [
                {"Date": "2026-01-02", "Predicted_Price": 101.0, "Low": 99.0, "High": 103.0},
                {"Date": "2026-01-03", "Predicted_Price": 105.5, "Low": 102.0, "High": 109.0},
            ],
            "forecast_bands": {"confidence": 0.9, "daily_volatility": 1.2},
        },
        "technical": {"rsi": 55.0, "trend": "BULLISH", "macd_signal": "BUY", "volatility_level": "MEDIUM"},
        "sentiment": {"score": 0.4, "label": "BULLISH", "sources": ["NewsAPI"]},
        "fundamentals": {},
        "price_history": [
            {"Date": "2025-12-01", "Open": 98.0, "High": 99.0, "Low": 97.0, "Close": 98.5, "Volume": 1000},
        ],
        "regime": "TRENDING",
        "model_selection": {"regime": "TRENDING", "adx_strength": 28.0, "annualized_volatility": 18.0, "selected": "LSTM", "rationale": "Trending market favors LSTM."},
        "analyst_report": {"success": True, "provider": "rule_based", "report": "Buy the dip."},
    }


class DemoCacheServiceTests(unittest.TestCase):
    def test_list_demo_tickers_is_fixed_and_servable(self):
        tickers = demo_cache.list_demo_tickers()
        self.assertGreaterEqual(len(tickers), 15)
        self.assertIn("AAPL", tickers)
        self.assertIn("RELIANCE.NS", tickers)
        self.assertEqual(tickers, demo_cache.DEMO_TICKERS)

    def test_get_demo_result_rejects_out_of_set_ticker(self):
        with patch.object(demo_cache, "DEMO_CACHE_DIR", Path(_TMP_DIR) / "demo_precomputed"):
            with self.assertRaises(KeyError):
                demo_cache.get_demo_result("FAKE")

    def test_get_demo_result_requires_cached_json(self):
        tmp = Path(_TMP_DIR) / "missing_cache"
        with patch.object(demo_cache, "DEMO_CACHE_DIR", tmp):
            with self.assertRaises(FileNotFoundError):
                demo_cache.get_demo_result("AAPL")

    def test_write_then_read_roundtrip_overwrites(self):
        tmp = Path(_TMP_DIR) / "demo_write"
        with patch.object(demo_cache, "DEMO_CACHE_DIR", tmp):
            path = demo_cache.demo_write("AAPL", _sample_result("AAPL"))
            self.assertTrue(path.exists())
            # Second write must overwrite the same file, never append.
            first_size = path.stat().st_size
            time.sleep(0.05)
            demo_cache.demo_write("AAPL", _sample_result("AAPL"))
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload["symbol"], "AAPL")
            self.assertEqual(path.stat().st_size, first_size)
            self.assertEqual(demo_cache.get_demo_result("AAPL")["symbol"], "AAPL")


class DemoLockdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        demo_cache.DEMO_CACHE_DIR = Path(_TMP_DIR) / "demo_precomputed"
        demo_cache.DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        demo_cache.demo_write("AAPL", _sample_result("AAPL"))
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "PUBLIC_DEMO": True})
        cls.client = cls.app.test_client()

    def test_demo_page_lists_fixed_tickers_in_dropdown(self):
        with patch("stockpredictor.views.dashboard.stocks.get_market_indices", return_value={}), \
             patch("stockpredictor.views.dashboard._fetch_live", return_value={"symbol": "AAPL", "quote": {}, "closes": []}):
            response = self.client.get("/demo")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("demoAnalyzeForm", body)
        self.assertIn('value="RELIANCE.NS"', body)
        self.assertIn('value="AAPL"', body)

    def test_demo_analyze_serves_precomputed_json(self):
        with patch("stockpredictor.views.dashboard.stocks.get_market_indices", return_value={}):
            response = self.client.get("/demo/analyze?symbol=AAPL")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Demo Analysis: AAPL", body)
        self.assertIn('id="forecastChart"', body)
        self.assertIn("Buy the dip.", body)

    def test_demo_analyze_does_not_call_live_pipeline(self):
        # If the demo route ever tried to train/predict live it would call the
        # pipeline; make it raise so this test proves the route only reads JSON.
        with patch("stockpredictor.views.dashboard.stocks.get_market_indices", return_value={}), \
             patch("stockpredictor.services.demo_cache.build_demo_result",
                   side_effect=AssertionError("demo must never build results live")), \
             patch("stockpredictor.services.stocks.predict_forecast",
                   side_effect=AssertionError("demo must never predict live")):
            response = self.client.get("/demo/analyze?symbol=AAPL")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Demo Analysis: AAPL", response.get_data(as_text=True))

    def test_demo_analyze_rejects_out_of_set_ticker_get(self):
        response = self.client.get("/demo/analyze?symbol=FAKE")
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"not part of the public demo set", response.data)

    def test_demo_analyze_rejects_out_of_set_ticker_post(self):
        import re

        with patch("stockpredictor.views.dashboard.stocks.get_market_indices", return_value={}), \
             patch("stockpredictor.views.dashboard._fetch_live", return_value={"symbol": "AAPL", "quote": {}, "closes": []}):
            page = self.client.get("/demo")
        match = re.search(r'name="csrf-token" content="([^"]+)"', page.get_data(as_text=True))
        token = match.group(1) if match else ""
        response = self.client.post(
            "/demo/analyze",
            data={"symbol": "DOGE-USD", "csrf_token": token},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"not part of the public demo set", response.data)

    def test_demo_analyze_404_when_cache_file_missing(self):
        with patch.object(demo_cache, "DEMO_CACHE_DIR", Path(_TMP_DIR) / "empty_demo"):
            response = self.client.get("/demo/analyze?symbol=MSFT")
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"not available yet", response.data)


class AnalyzeGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
        cls.client = cls.app.test_client()

    def test_training_gate_gives_clean_busy_error_to_second_job(self):
        started = threading.Event()
        release = threading.Event()
        # Dedicated manager keeps the busy-wait tiny and deterministic (the
        # shared singleton uses the production 30s default).
        manager = jobs.JobManager(gate_wait_seconds=1)

        def slow_fn(*args, job_marker=None, **kwargs):
            started.set()
            release.wait(10)
            return {"result": job_marker or "ok"}

        first = manager.submit(slow_fn, job_marker="first", dedupe_key="gate-test-1", owner="a@test.dev")
        self.assertTrue(started.wait(10), "first job should start")

        second = manager.submit(slow_fn, job_marker="second", dedupe_key="gate-test-2", owner="b@test.dev")

        deadline = time.time() + 10
        status = None
        while time.time() < deadline:
            status = manager.status(second, requester="b@test.dev") or {}
            if status.get("status") in ("complete", "error"):
                break
            time.sleep(0.05)
        self.assertEqual(status.get("status"), "error")
        self.assertIn("in progress", (status.get("message") or "").lower())

        release.set()
        deadline = time.time() + 10
        first_status = None
        while time.time() < deadline:
            first_status = manager.status(first, requester="a@test.dev") or {}
            if first_status.get("status") == "complete":
                break
            time.sleep(0.05)
        self.assertEqual(first_status.get("status"), "complete")
        self.assertEqual(first_status.get("result", {}).get("result"), "first")
        release.clear()
        started.clear()

    def test_concurrent_jobs_never_run_parallel(self):
        in_flight = threading.Lock()
        max_in_flight = [0]
        started = threading.Event()
        release = threading.Event()
        manager = jobs.JobManager(gate_wait_seconds=1)

        def probe_fn(*args, **kwargs):
            in_flight.acquire()
            current = max_in_flight[0] + 1
            max_in_flight[0] = max(max_in_flight[0], current)
            in_flight.release()
            started.set()
            release.wait(10)
            in_flight.acquire()
            max_in_flight[0] = max_in_flight[0] - 1
            in_flight.release()
            return {"result": "ok"}

        first = manager.submit(probe_fn, dedupe_key="probe-1", owner="a@test.dev")
        self.assertTrue(started.wait(10))
        manager.submit(probe_fn, dedupe_key="probe-2", owner="b@test.dev")
        # Give the second job time to queue against the gate while the first
        # still runs. It must never overlap the first job's body.
        time.sleep(1.5)
        self.assertEqual(max_in_flight[0], 1, "two training jobs must never overlap")

        status = manager.status(first, requester="a@test.dev") or {}
        self.assertEqual(status.get("status"), "running")
        release.set()
        deadline = time.time() + 10
        while time.time() < deadline:
            status = manager.status(first, requester="a@test.dev") or {}
            if status.get("status") == "complete":
                break
            time.sleep(0.05)
        self.assertEqual(status.get("status"), "complete")
        release.clear()
        started.clear()


class SettingsKnobTests(unittest.TestCase):
    def test_train_epochs_reads_default_and_override(self):
        from stockpredictor.config import Settings

        self.assertEqual(Settings().TRAIN_EPOCHS, 20)
        self.assertEqual(Settings().EARLY_STOPPING_PATIENCE, 5)
        with patch.dict(os.environ, {"TRAIN_EPOCHS": "6", "TRAIN_BATCH_SIZE": "16"}):
            self.assertEqual(Settings().TRAIN_EPOCHS, 6)
            self.assertEqual(Settings().TRAIN_BATCH_SIZE, 16)
        with patch.dict(os.environ, {"TRAIN_EPOCHS": "junk", "TRAIN_BATCH_SIZE": "-3"}):
            self.assertEqual(Settings().TRAIN_EPOCHS, 20)
            self.assertEqual(Settings().TRAIN_BATCH_SIZE, 1)


class _IsoKey:
    """Stand-in for a pandas Timestamp: json can't serialize it as a dict key."""

    def __init__(self, text):
        self._text = text

    def isoformat(self):
        return self._text

    def __repr__(self):
        return f"Timestamp('{self._text}')"


class JsonKeyCoercionTests(unittest.TestCase):
    def test_coerce_keys_flattens_timestamp_keys_to_iso_strings(self):
        payload = {
            "report": {
                "forecast_rows": [{"Date": "2026-01-02", "Predicted_Price": 101.0}],
            },
            "fundamentals": {
                "financial_statements": {
                    "income_statement": {
                        _IsoKey("2025-09-30"): {"Revenue": 100000},
                        _IsoKey("2024-09-30"): {"Revenue": 90000},
                    }
                }
            },
        }
        coerced = demo_cache._coerce_keys(payload)
        income = coerced["fundamentals"]["financial_statements"]["income_statement"]
        self.assertIn("2025-09-30", income)
        self.assertNotIn(_IsoKey("2025-09-30"), income)
        # Round trip through the real writer produce JSON-safe keys.
        tmp = Path(_TMP_DIR) / "coerce_write"
        with patch.object(demo_cache, "DEMO_CACHE_DIR", tmp):
            demo_cache.demo_write("AAPL", coerced)
            with (tmp / "AAPL.json").open("r", encoding="utf-8") as fh:
                reloaded = json.load(fh)
        self.assertEqual(reloaded["fundamentals"]["financial_statements"]["income_statement"]["2025-09-30"]["Revenue"], 100000)

    def test_coerce_keys_is_json_dump_safe(self):
        payload = {"nested": {_IsoKey("2026-01-02"): 3}}
        with self.assertRaises(TypeError):
            json.dumps(payload, default=str)
        coerced = demo_cache._coerce_keys(payload)
        dumped = json.dumps(coerced, default=str)
        self.assertIn("2026-01-02", dumped)


if __name__ == "__main__":
    unittest.main()