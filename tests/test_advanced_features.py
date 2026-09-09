"""Tests for the ten advanced features added to Stock Predictor Pro:
backtesting, Monte Carlo, regime detection, SHAP explainability, drift
monitoring, risk engine, push alerts, LLM analyst report, relative strength
and the feature store.

State is redirected to a throwaway directory BEFORE importing the package and
all data is synthetic — nothing touches the network.
"""
import os
import re
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

import production_core as pc

_TMP_DIR = tempfile.mkdtemp(prefix="stockpredictor_advanced_")
pc.STATE_DIR = Path(_TMP_DIR)
pc.STATE_FILE = pc.STATE_DIR / "app_state.json"

from stockpredictor import create_app  # noqa: E402
from stockpredictor.services.auth import user_store  # noqa: E402

for _SMTP_KEY in ("SMTP_HOST", "SMTP_USER", "TELEGRAM_BOT_TOKEN",
                  "TELEGRAM_CHAT_ID", "DISCORD_WEBHOOK_URL",
                  "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_SMTP_KEY, None)

_VALID_PASSWORD = "Xk9!mnQp4"
_COUNTER = {"n": 0}


def _unique_email(prefix):
    _COUNTER["n"] += 1
    return f"{prefix}{_COUNTER['n']}@example.com"


def _csrf_token(client, url):
    response = client.get(url)
    html = response.get_data(as_text=True)
    match = re.search(r'name="csrf-token" content="([^"]+)"', html)
    if match:
        return match.group(1)
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    return match.group(1) if match else None


def _register(client, email, password=_VALID_PASSWORD):
    token = _csrf_token(client, "/register")
    response = client.post(
        "/register",
        data={"email": email, "password": password,
              "confirm_password": password, "csrf_token": token},
    )
    pending = user_store.pending_registrations.get(email.lower())
    if pending is None:
        return response
    token = _csrf_token(client, f"/verify_otp?email={email}")
    return client.post(
        "/verify_otp",
        data={"email": email, "otp": pending["otp"], "csrf_token": token},
    )


def _synthetic_rows(n=300, seed=1, drift_mult=1.0, vol=0.015, start=100.0):
    """Build OHLCV rows from a geometric random walk."""
    rng = np.random.default_rng(seed)
    closes = [start]
    for i in range(n - 1):
        step = float(rng.normal(0.0005, vol)) * drift_mult if i >= n // 2 else float(rng.normal(0.0005, vol))
        closes.append(closes[-1] * (1 + step))
    idx = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return [{
        "Date": d.strftime("%Y-%m-%d"),
        "Open": round(float(c * 0.999), 2),
        "High": round(float(c * 1.01), 2),
        "Low": round(float(c * 0.99), 2),
        "Close": round(float(c), 2),
        "Volume": 1_000_000 + i,
    } for d, c, i in zip(idx, closes, range(n))]


def _closes(rows):
    return [r["Close"] for r in rows]


def _trending_rows(n=200, daily_return=0.002, vol=0.01, seed=1, start=100.0):
    """Build an OHLCV series with a consistent positive drift."""
    rng = np.random.default_rng(seed)
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + float(rng.normal(daily_return, vol))))
    idx = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return [{
        "Date": d.strftime("%Y-%m-%d"),
        "Open": round(float(c * 0.999), 2),
        "High": round(float(c * 1.01), 2),
        "Low": round(float(c * 0.99), 2),
        "Close": round(float(c), 2),
        "Volume": 1_000_000 + i,
    } for d, c, i in zip(idx, closes, range(n))]


class BacktestTests(unittest.TestCase):
    def test_run_backtest_produces_metrics_and_curves(self):
        from stockpredictor.services.backtest import run_backtest

        result = run_backtest(_synthetic_rows(300), strategy="MA_CROSS")
        self.assertTrue(result["success"])
        metrics = result["metrics"]
        for key in ("total_return_pct", "sharpe", "sortino",
                    "max_drawdown_pct", "win_rate", "total_trades",
                    "buy_and_hold_return_pct", "final_equity"):
            self.assertIn(key, metrics)
        self.assertGreater(len(result["equity_curve"]), 50)
        self.assertEqual(len(result["equity_curve"]), len(result["buy_hold_curve"]))
        self.assertEqual(result["initial_cash"], 10000.0)

    def test_run_backtest_rejects_short_history(self):
        from stockpredictor.services.backtest import run_backtest

        result = run_backtest(_synthetic_rows(15))
        self.assertFalse(result["success"])

    def test_compute_metrics_returns_expected_keys(self):
        from stockpredictor.services.backtest import compute_metrics

        metrics = compute_metrics([100, 101, 102, 103])
        self.assertIn("sharpe", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        self.assertIn("cagr", metrics)

    def test_signal_series_macd_and_rsi(self):
        from stockpredictor.services.backtest import compute_signal_series

        rows = _synthetic_rows(250, seed=3)
        for strategy in ("MA_CROSS", "MACD", "RSI"):
            signals = compute_signal_series(rows, strategy=strategy)
            self.assertGreater(len(signals), 50)
            self.assertTrue(all(s["signal"] in ("BUY", "SELL", "HOLD") for s in signals))


class MonteCarloTests(unittest.TestCase):
    def test_simulate_returns_bands_and_probabilities(self):
        from stockpredictor.services.monte_carlo import simulate

        closes = _closes(_synthetic_rows(250))
        result = simulate(closes, horizon=30, n_paths=1000, seed=42,
                          target=max(closes) * 1.2, stop=min(closes) * 0.8)
        self.assertTrue(result["success"])
        self.assertEqual(len(result["bands"]["median"]), 31)
        self.assertEqual(result["n_paths"], 1000)
        self.assertGreaterEqual(result["prob_hit_target"], 0.0)
        self.assertLessEqual(result["prob_hit_target"], 100.0)
        self.assertGreaterEqual(result["prob_hit_stop"], 0.0)
        self.assertLessEqual(result["prob_hit_stop"], 100.0)

    def test_simulate_rejects_short_history(self):
        from stockpredictor.services.monte_carlo import simulate

        self.assertFalse(simulate([1.0, 2.0])["success"])

    def test_seed_is_deterministic(self):
        from stockpredictor.services.monte_carlo import simulate

        closes = _closes(_synthetic_rows(200, seed=5))
        a = simulate(closes, horizon=20, n_paths=500, seed=11)
        b = simulate(closes, horizon=20, n_paths=500, seed=11)
        self.assertEqual(a["mean_final"], b["mean_final"])


class RegimeTests(unittest.TestCase):
    def test_detect_trending_up(self):
        from stockpredictor.services.regime import detect_regime

        up = _synthetic_rows(250, seed=2, vol=0.008)
        result = detect_regime(_closes(up))
        self.assertTrue(result["success"])
        self.assertIn(result["regime"], ("trending-up", "ranging"))
        self.assertIn("adx_strength", result)
        self.assertIn("annualized_volatility", result)

    def test_auto_select_model_returns_recommendation(self):
        from stockpredictor.services.regime import auto_select_model

        result = auto_select_model(_closes(_synthetic_rows(250)))
        self.assertTrue(result["success"])
        self.assertIn(result["selected"], ("LSTM", "GRU", "Prophet", "ARIMA", "AUTO"))
        self.assertTrue(result["rationale"])

    def test_auto_select_keeps_auto_when_accurate(self):
        from stockpredictor.services.regime import auto_select_model

        result = auto_select_model(_closes(_synthetic_rows(250)), recent_accuracy=0.9)
        self.assertEqual(result["selected"], "AUTO")

    def test_insufficient_data(self):
        from stockpredictor.services.regime import detect_regime

        self.assertFalse(detect_regime([1.0, 2.0, 3.0])["success"])


class ExplainTests(unittest.TestCase):
    def test_explain_forecast_returns_top_features(self):
        from stockpredictor.services.explain import explain_forecast

        result = explain_forecast(_synthetic_rows(300), persist=False)
        self.assertTrue(result["success"])
        self.assertIn(result["method"], ("shap_tree_explainer",
                                        "random_forest_permutation_proxy",
                                        "correlation"))
        self.assertGreaterEqual(len(result["top"]), 3)
        self.assertEqual(len(result["top"]), len(result["importance"]))


class DriftTests(unittest.TestCase):
    def test_psi_small_for_similar_arrays(self):
        from stockpredictor.services.drift import psi

        rng = np.random.default_rng(1)
        a = rng.normal(0.0, 0.01, 500)
        b = rng.normal(0.0, 0.01, 500)
        self.assertLess(psi(a, b), 0.1)

    def test_psi_large_for_different_arrays(self):
        from stockpredictor.services.drift import psi

        rng = np.random.default_rng(1)
        a = rng.normal(0.0, 0.01, 500)
        b = rng.normal(0.4, 0.05, 500)
        self.assertGreater(psi(a, b), 0.5)

    def test_check_drift_creates_baseline_then_detects_shift(self):
        from stockpredictor.services.drift import check_drift

        low = _synthetic_rows(220, seed=8, vol=0.004)
        first = check_drift("DRIFT1", low, auto_baseline=True)
        self.assertTrue(first.get("baseline_created"))
        self.assertFalse(first.get("drift_detected"))

        high = _synthetic_rows(220, seed=9, vol=0.05)
        second = check_drift("DRIFT1", high)
        self.assertTrue(second["drift_detected"])
        self.assertTrue(second["needs_retrain"])
        self.assertGreater(second["overall_psi"], 0.25)

    def test_trigger_retraining_queues_job(self):
        from stockpredictor.services.drift import trigger_retraining

        result = trigger_retraining("AAPL")
        self.assertTrue(result["success"])
        self.assertTrue(result["job_id"])


class RiskEngineTests(unittest.TestCase):
    def test_kelly_fraction(self):
        from stockpredictor.services.risk import kelly_fraction

        result = kelly_fraction(0.55, 0.20, 0.10)
        self.assertGreater(result["half_kelly"], 0.0)
        self.assertLessEqual(result["full_kelly"], 0.25)

    def test_kelly_negative_edge(self):
        from stockpredictor.services.risk import kelly_fraction

        result = kelly_fraction(0.30, 0.10, 0.10)
        self.assertEqual(result["full_kelly"], 0.0)

    def test_volatility_target_sizing(self):
        from stockpredictor.services.risk import volatility_target_position_size

        result = volatility_target_position_size(10000, 50.0, 0.30)
        self.assertGreater(result["shares"], 0)
        self.assertLessEqual(result["allocation_pct"], 25.0)

    def test_portfolio_var(self):
        from stockpredictor.services.risk import portfolio_var

        rng = np.random.default_rng(3)
        returns = {"A": rng.normal(0.0, 0.01, 200).tolist(),
                   "B": rng.normal(0.0, 0.02, 200).tolist()}
        result = portfolio_var(returns)
        self.assertTrue(result["success"])
        self.assertLess(result["historical_var_1d"], 0.0)
        self.assertEqual(result["horizon_days"], 1)

    def test_correlation_matrix(self):
        from stockpredictor.services.risk import correlation_matrix

        rng = np.random.default_rng(4)
        base = rng.normal(0.0, 0.01, 200)
        result = correlation_matrix({"A": base.tolist(), "B": (base * 2).tolist()})
        self.assertTrue(result["success"])
        self.assertEqual(len(result["matrix"]), 2)
        self.assertGreater(result["matrix"][0][1], 0.9)

    def test_rebalance_suggestions(self):
        from stockpredictor.services.risk import rebalance_suggestions

        suggestions = rebalance_suggestions({"A": 0.9, "B": 0.1}, {"A": 0.5, "B": 0.5})
        self.assertEqual(len(suggestions), 2)
        actions = {s["symbol"]: s["action"] for s in suggestions}
        self.assertEqual(actions["A"], "SELL")
        self.assertEqual(actions["B"], "BUY")


class RelativeStrengthTests(unittest.TestCase):
    def test_outperforming_symbol_scores_higher(self):
        from stockpredictor.services.relative_strength import relative_strength

        bench = _synthetic_rows(200, seed=1)
        # Multiply by a slowly growing factor so the outperformer genuinely has
        # higher *returns* than the benchmark (scaling by a constant would leave
        # returns unchanged and defeat the purpose of the test).
        good = [dict(r, Close=round(r["Close"] * (1 + 0.0006 * i), 2))
                for i, r in enumerate(bench)]
        bad = [dict(r, Close=round(r["Close"] * (1 - 0.0006 * i), 2))
               for i, r in enumerate(bench)]
        good_result = relative_strength(good, bench)
        bad_result = relative_strength(bad, bench)
        self.assertTrue(good_result["success"])
        self.assertTrue(bad_result["success"])
        self.assertGreater(good_result["composite"], bad_result["composite"])
        self.assertIn("20", good_result["windows"])

    def test_insufficient_data(self):
        from stockpredictor.services.relative_strength import relative_strength

        result = relative_strength([{"Close": 10.0}], [{"Close": 10.0}])
        self.assertFalse(result["success"])


class FeatureStoreTests(unittest.TestCase):
    def test_log_and_read_importance(self):
        from stockpredictor.services.feature_store import (
            build_features, feature_importance_history, latest_feature_importance,
            list_stored_symbols, log_feature_importance,
        )

        rows = _synthetic_rows(120)
        frame = build_features(rows)
        self.assertIn("RSI_14", frame.columns)
        self.assertEqual(len(frame), len(rows))

        saved = log_feature_importance("FSX", {"Return_1d": 0.5, "RSI_14": 0.2})
        self.assertTrue(saved["success"])
        latest = latest_feature_importance("FSX")
        self.assertIsNotNone(latest)
        self.assertEqual(latest["importance"]["RSI_14"], 0.2)
        self.assertIn("FSX", list_stored_symbols())
        self.assertGreaterEqual(len(feature_importance_history("FSX")), 1)


class LlmReportTests(unittest.TestCase):
    def test_rule_based_summary_mentions_symbol(self):
        from stockpredictor.services.llm_report import rule_based_summary

        text = rule_based_summary("TSLA",
                                  {"current_price": 200.0, "predicted_price": 220.0, "days": 30},
                                  {"overall_rating": "BUY", "trend": "BULLISH"},
                                  {"polarity": 0.4, "count": 12})
        self.assertIn("TSLA", text)
        self.assertIn("220", text)

    def test_generate_falls_back_without_keys(self):
        from stockpredictor.services.llm_report import generate_analyst_report

        result = generate_analyst_report("AAPL", {"current_price": 100.0},
                                         {"overall_rating": "HOLD"})
        self.assertTrue(result["success"])
        self.assertEqual(result["provider"], "rule_based")
        self.assertTrue(result["report"])


class PushAlertTests(unittest.TestCase):
    def test_telegram_unconfigured_degrades(self):
        from stockpredictor.services.push_alerts import send_telegram

        app = create_app({"TESTING": True})
        with app.app_context():
            result = send_telegram("hello")
        self.assertFalse(result["success"])
        self.assertEqual(result["channel"], "telegram")

    def test_discord_unconfigured_degrades(self):
        from stockpredictor.services.push_alerts import send_discord

        app = create_app({"TESTING": True})
        with app.app_context():
            result = send_discord("hello")
        self.assertFalse(result["success"])

    def test_notify_triggered_without_channels(self):
        from stockpredictor.services.push_alerts import notify_triggered

        app = create_app({"TESTING": True})
        with app.app_context():
            result = notify_triggered([{"symbol": "AAPL", "condition": "above",
                                        "threshold": 100, "current_price": 110}],
                                      recipient="")
        self.assertFalse(result["success"])

    def test_notify_triggered_dispatches_telegram(self):
        from stockpredictor.services.push_alerts import notify_triggered

        app = create_app({"TESTING": True, "TELEGRAM_BOT_TOKEN": "tok",
                          "TELEGRAM_CHAT_ID": "123"})
        with app.app_context(), \
             mock.patch("requests.post") as post:
            post.return_value = mock.Mock(status_code=200)
            result = notify_triggered([{"symbol": "AAPL", "condition": "above",
                                        "threshold": 100, "current_price": 110}])
        self.assertIn("telegram", result["delivered"])


class AdvancedApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()
        self.email = _unique_email("adv")
        _register(self.client, self.email)

    def _mock_history(self):
        return mock.patch("stockpredictor.services.stocks.get_price_history",
                          return_value=_synthetic_rows(300))

    def test_backtest_endpoint(self):
        with self._mock_history():
            response = self.client.get("/api/backtest/AAPL?strategy=MA_CROSS&fast=20&slow=50")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("metrics", data)

    def test_scenario_endpoint(self):
        with self._mock_history():
            response = self.client.post(
                "/api/scenario/AAPL",
                json={"horizon": 30, "target": 150.0, "stop": 90.0},
                headers={"X-CSRFToken": _csrf_token(self.client, "/dashboard")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["bands"]["median"]), 31)

    def test_explain_endpoint(self):
        with self._mock_history():
            response = self.client.get("/api/explain/AAPL")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])

    def test_drift_endpoint(self):
        with self._mock_history():
            response = self.client.get("/api/drift/AAPL")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("drift_detected", data)
        self.assertIn("overall_psi", data)

    def test_relative_strength_endpoint(self):
        add = self.client.post(
            "/api/watchlist/add",
            json={"symbol": "AAPL"},
            headers={"X-CSRFToken": _csrf_token(self.client, "/dashboard")},
        )
        self.assertEqual(add.status_code, 200)
        with mock.patch("stockpredictor.services.stocks.get_price_history",
                        side_effect=lambda symbol, *a, **k: _synthetic_rows(200, seed=1)):
            response = self.client.get("/api/relative-strength")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["benchmark"], "SPY")

    def test_test_notification_endpoint(self):
        response = self.client.post(
            "/api/alerts/test-notification",
            json={"channel": "all"},
            headers={"X-CSRFToken": _csrf_token(self.client, "/dashboard")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("delivered", data)

    def test_llm_report_endpoint(self):
        response = self.client.post(
            "/api/report/AAPL",
            json={"forecast": {"current_price": 100}, "technical": {}, "sentiment": {}},
            headers={"X-CSRFToken": _csrf_token(self.client, "/dashboard")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["provider"], "rule_based")


if __name__ == "__main__":
    unittest.main()
