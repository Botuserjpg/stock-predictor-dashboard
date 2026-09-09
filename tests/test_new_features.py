"""Tests for the new features: compare page, equity curve, exports, accent UI.

State persistence is redirected to a throwaway directory BEFORE importing the
package (same pattern as test_stockpredictor.py).
"""
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import production_core as pc

_TMP_DIR = tempfile.mkdtemp(prefix="stockpredictor_new_features_")
pc.STATE_DIR = Path(_TMP_DIR)
pc.STATE_FILE = pc.STATE_DIR / "app_state.json"

from stockpredictor import create_app  # noqa: E402
from stockpredictor.services.auth import user_store  # noqa: E402
from stockpredictor.services.portfolio import score_position_signal  # noqa: E402

for _SMTP_KEY in ("SMTP_HOST", "SMTP_USER"):
    os.environ.pop(_SMTP_KEY, None)

_VALID_PASSWORD = "Xk9!mnQp4"


def _csrf_token(client, url):
    response = client.get(url)
    match = re.search(r'name="csrf_token" value="([^"]+)"', response.get_data(as_text=True))
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


def _fake_quote(symbol):
    return {"symbol": symbol, "price": 100.0, "change": 1.0,
            "change_pct": 1.0, "source": "test"}


def _fake_history(symbol, *args, **kwargs):
    return [
        {"Date": "2026-01-01", "Open": 99.0, "High": 101.0, "Low": 98.0,
         "Close": 100.0, "Volume": 1000},
        {"Date": "2026-01-02", "Open": 100.0, "High": 102.0, "Low": 99.0,
         "Close": 101.0, "Volume": 1100},
    ]


_COUNTER = {"n": 0}


def _unique_email(prefix):
    _COUNTER["n"] += 1
    return f"{prefix}{_COUNTER['n']}@example.com"


class CompareViewTests(unittest.TestCase):
    """Multi-symbol comparison page."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()
        self.email = _unique_email("compare")
        _register(self.client, self.email)

    def test_compare_requires_login(self):
        response = self.app.test_client().get("/compare")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_form_renders(self):
        response = self.client.get("/compare")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Compare Stocks", response.data)
        self.assertIn(b"compareForm", response.data)

    def test_compare_renders_with_two_symbols(self):
        with mock.patch("stockpredictor.services.stocks.get_quote",
                        side_effect=_fake_quote), \
             mock.patch("stockpredictor.services.stocks.get_technical_analysis",
                        return_value={"rsi": 55.0, "trend": "BULLISH",
                                      "macd_signal": "BULLISH",
                                      "volatility_level": "MEDIUM"}), \
             mock.patch("stockpredictor.services.sentiment.get_sentiment",
                        return_value={"score": 0.4, "label": "BULLISH",
                                      "sources": ["test"]}), \
             mock.patch("stockpredictor.services.stocks.get_price_history",
                        side_effect=_fake_history):
            response = self.client.post(
                "/compare",
                data={"symbols": "AAPL, MSFT",
                      "csrf_token": _csrf_token(self.client, "/compare")},
            )
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("AAPL", body)
        self.assertIn("MSFT", body)
        self.assertIn("compareChart", body)

    def test_compare_rejects_single_symbol(self):
        response = self.client.post(
            "/compare",
            data={"symbols": "AAPL", "csrf_token": _csrf_token(self.client, "/compare")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"at least 2 symbols", response.data)

    def test_compare_caps_at_five_symbols(self):
        with mock.patch("stockpredictor.services.stocks.get_quote",
                        side_effect=_fake_quote), \
             mock.patch("stockpredictor.services.stocks.get_technical_analysis",
                        return_value={}), \
             mock.patch("stockpredictor.services.sentiment.get_sentiment",
                        return_value={}), \
             mock.patch("stockpredictor.services.stocks.get_price_history",
                        side_effect=_fake_history):
            response = self.client.post(
                "/compare",
                data={"symbols": "A,B,C,D,E,F,G",
                      "csrf_token": _csrf_token(self.client, "/compare")},
            )
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertEqual(body.count("compareChart"), 1)
        # Only the first five symbols are compared.
        for symbol in ("A", "B", "C", "D", "E"):
            self.assertIn(symbol, body)
        self.assertNotIn('"symbol": "G"', body)


class EquityCurveTests(unittest.TestCase):
    """Pure offline reconstruction of a portfolio value timeline."""

    @classmethod
    def setUpClass(cls):
        from stockpredictor.services.portfolio import equity_curve

        cls.equity_curve = staticmethod(equity_curve)

    def test_empty_history_returns_empty(self):
        curve = self.equity_curve([], 10000.0, {}, {})
        self.assertEqual(curve, [])

    def test_only_cash_events_returns_empty(self):
        history = [
            {"type": "DEPOSIT", "symbol": "CASH", "quantity": None,
             "price": 500, "date": "2026-01-01"},
        ]
        self.assertEqual(self.equity_curve(history, 10500.0, {}, {}), [])

    def test_basic_buy_sell_curve(self):
        history = [
            {"type": "BUY", "symbol": "AAPL", "quantity": 10,
             "price": 100, "date": "2026-01-01"},
            {"type": "SELL", "symbol": "AAPL", "quantity": 4,
             "price": 120, "date": "2026-02-01"},
        ]
        # cash after BUY 10@100 (0.1% fee) then SELL 4@120
        cash = 10000.0 - 10 * 100 * 1.001 + 4 * 120 * 0.999
        positions = {"AAPL": {"quantity": 6}}
        curve = self.equity_curve(history, cash, positions, {})

        self.assertEqual(len(curve), 2)
        first, second = curve
        # Initial state: 10k cash, no shares.
        self.assertEqual(first["value"], 10000.0)
        # Undoing the SELL removes its fee-discounted proceeds (4*120*0.999),
        # leaving reconstructed cash 8999.00 plus 10 shares at the last price.
        self.assertAlmostEqual(second["value"], 8999.0 + 10 * 120, places=2)
        self.assertEqual(first["date"], "2026-01-01")
        self.assertEqual(second["date"], "2026-02-01")

    def test_curve_falls_back_to_live_prices(self):
        history = [
            {"type": "BUY", "symbol": "AAPL", "quantity": 2,
             "price": 50, "date": "2026-01-01"},
        ]
        cash = 10000.0 - 2 * 50 * 1.001
        positions = {"AAPL": {"quantity": 2}}
        curve = self.equity_curve(history, cash, positions, {"AAPL": 70.0})
        # Undoing the BUY adds back price + fee, landing exactly on the
        # starting cash (10000.0).
        self.assertAlmostEqual(curve[0]["value"], 10000.0, places=2)

    def test_curve_accounts_for_deposits_in_cash(self):
        history = [
            {"type": "BUY", "symbol": "AAPL", "quantity": 10,
             "price": 100, "date": "2026-01-01"},
            {"type": "DEPOSIT", "symbol": "CASH", "quantity": None,
             "price": 500, "date": "2026-01-15"},
        ]
        cash = 10000.0 - 10 * 100 * 1.001 + 500
        positions = {"AAPL": {"quantity": 10}}
        curve = self.equity_curve(history, cash, positions, {})
        # Initial reconstructed state includes the (unplotted) deposit offset.
        self.assertAlmostEqual(curve[0]["value"], 10500.0, places=2)


class PortfolioExportTests(unittest.TestCase):
    """Portfolio CSV export endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()
        self.email = _unique_email("portfolio")
        _register(self.client, self.email)

    def test_csv_export_requires_login(self):
        response = self.app.test_client().get("/api/portfolio/export/csv")
        self.assertEqual(response.status_code, 302)

    def test_csv_export_empty_history_returns_404(self):
        response = self.client.get("/api/portfolio/export/csv")
        self.assertEqual(response.status_code, 404)

    def test_csv_export_with_transactions(self):
        email = self.email
        user_store.portfolios()[email] = {
            "cash": 9000.0,
            "positions": {"AAPL": {"quantity": 10, "avg_price": 100.0}},
            "history": [
                {"type": "BUY", "symbol": "AAPL", "quantity": 10,
                 "price": 100, "date": "2026-01-01"},
                {"type": "DEPOSIT", "symbol": "CASH", "quantity": None,
                 "price": 500, "date": "2026-01-02"},
            ],
        }
        user_store.persist()
        response = self.client.get("/api/portfolio/export/csv")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        body = response.get_data(as_text=True)
        self.assertIn("date,type,symbol,quantity,price", body)
        self.assertIn("BUY,AAPL,10,100", body)
        self.assertIn("DEPOSIT,CASH,,500", body)


class UiThemeTests(unittest.TestCase):
    """Accent/theme chrome is wired into every page."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def test_accent_swatches_rendered_on_login_page(self):
        response = self.app.test_client().get("/login")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("accentPicker", body)
        self.assertIn("data-accent=\"teal\"", body)

    def test_css_defines_accent_palettes(self):
        response = self.app.test_client().get("/static/css/app.css")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("[data-accent=\"rose\"]", body)
        self.assertIn("--accent-rgb", body)


class DashboardWatchlistTests(unittest.TestCase):
    """Dashboard renders live watchlist data (services mocked)."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()
        self.email = _unique_email("live")
        _register(self.client, self.email)

    def test_dashboard_renders_live_watchlist(self):
        user_store.watchlists()[self.email] = ["AAPL", "MSFT"]
        user_store.persist()

        with mock.patch("stockpredictor.services.stocks.get_market_indices",
                        return_value={}), \
             mock.patch("stockpredictor.services.stocks.get_quote",
                        side_effect=_fake_quote), \
             mock.patch("stockpredictor.services.stocks.get_price_history",
                        side_effect=_fake_history):
            response = self.client.get("/dashboard")

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Live Watchlist", body)
        self.assertIn("data-sparkline", body)
        self.assertIn("Top Movers", body)


class PortfolioMetricsTests(unittest.TestCase):
    """Portfolio page P&L math is consistent with contributions."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()
        self.email = _unique_email("metrics")
        _register(self.client, self.email)

    def test_fresh_portfolio_shows_zero_pnl(self):
        with mock.patch("stockpredictor.services.stocks.get_current_price",
                        return_value=100.0):
            response = self.client.get("/portfolio")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        # 10k cash, no positions -> total == contributions -> $0.00 P&L
        self.assertIn("▲ $0.00", body)
        self.assertIn("(+0.00%)", body)

    def test_pnl_accounts_for_deposits_and_fees(self):
        # Deposit $500, buy 10 AAPL @ 100 (0.1% fee), live price $110.
        user_store.portfolios()[self.email] = {
            "cash": 9499.0,
            "positions": {"AAPL": {"quantity": 10, "avg_price": 100.0}},
            "history": [
                {"type": "DEPOSIT", "symbol": "CASH", "quantity": None,
                 "price": 500, "date": "2026-01-01"},
                {"type": "BUY", "symbol": "AAPL", "quantity": 10,
                 "price": 100, "date": "2026-01-02"},
            ],
        }
        user_store.persist()
        with mock.patch("stockpredictor.services.stocks.get_current_price",
                        return_value=110.0), \
             mock.patch("stockpredictor.services.stocks.get_technical_analysis",
                        return_value={}), \
             mock.patch("stockpredictor.services.sentiment.get_sentiment",
                        return_value={"score": 0.0, "label": "NEUTRAL"}), \
             mock.patch("stockpredictor.services.stocks.get_price_history",
                        return_value=_fake_history):
            response = self.client.get("/portfolio")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        # contributions = 10000 + 500 = 10500; total = 9499 + 10*110 = 10599
        # total P&L = +99 (0.94%), i.e. the buy fee is a real loss;
        # unrealized = 1100 - 1000 = +100 (10.00%)
        self.assertIn("$99.00", body)
        self.assertIn("(+0.94%)", body)
        self.assertIn("$100.00", body)
        self.assertIn("(+10.00%)", body)


class SignalScoringTests(unittest.TestCase):
    """Pure composite scoring behind the AI insights."""

    def test_neutral_inputs_are_hold(self):
        result = score_position_signal({}, {"score": 0.0, "label": "NEUTRAL"})
        self.assertEqual(result["signal"], "HOLD")
        self.assertTrue(45 <= result["score"] <= 59)

    def test_overbought_bearish_is_sell(self):
        ta = {"rsi": 78.0, "trend": "BEARISH", "macd_signal": "BEARISH",
              "volatility_level": "HIGH"}
        result = score_position_signal(ta, {"label": "BEARISH"})
        self.assertIn(result["signal"], ("SELL", "STRONG_SELL"))
        self.assertLess(result["score"], 45)

    def test_oversold_bullish_is_buy(self):
        ta = {"rsi": 28.0, "trend": "BULLISH", "macd_signal": "BULLISH",
              "volatility_level": "LOW"}
        result = score_position_signal(ta, {"label": "BULLISH"})
        self.assertIn(result["signal"], ("BUY", "STRONG_BUY"))
        self.assertGreater(result["score"], 60)

    def test_momentum_fallback_without_technical(self):
        result = score_position_signal(None, None, [100.0, 108.0, 105.0, 110.0])
        self.assertTrue(any("Positive momentum" in r for r in result["reasons"]))

    def test_score_stays_clamped_and_sell_bound(self):
        ta = {"rsi": 100.0, "trend": "BEARISH", "macd_signal": "BEARISH",
              "volatility_level": "HIGH", "momentum_score": -100}
        result = score_position_signal(ta, {"label": "BEARISH"})
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)
        self.assertIn(result["signal"], ("SELL", "STRONG_SELL"))


class PortfolioInsightsTests(unittest.TestCase):
    """AI insights render on the portfolio page (services mocked)."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()
        self.email = _unique_email("insights")
        _register(self.client, self.email)
        user_store.portfolios()[self.email] = {
            "cash": 8000.0,
            "positions": {"AAPL": {"quantity": 10, "avg_price": 110.0}},
            "history": [
                {"type": "BUY", "symbol": "AAPL", "quantity": 10,
                 "price": 110, "date": "2026-01-01"},
            ],
        }
        user_store.persist()

    def _get(self, price, ta, sentiment):
        with mock.patch("stockpredictor.services.stocks.get_current_price",
                        return_value=price), \
             mock.patch("stockpredictor.services.stocks.get_technical_analysis",
                        return_value=ta), \
             mock.patch("stockpredictor.services.sentiment.get_sentiment",
                        return_value=sentiment), \
             mock.patch("stockpredictor.services.stocks.get_price_history",
                        side_effect=_fake_history):
            return self.client.get("/portfolio")

    def test_sell_signal_renders_banner_and_charts(self):
        response = self._get(
            90.0,
            {"rsi": 80.0, "trend": "BEARISH", "macd_signal": "BEARISH",
             "volatility_level": "HIGH"},
            {"score": -0.4, "label": "BEARISH"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("AI Insights", body)
        self.assertIn("Price-drop risk", body)
        self.assertIn("allocationChart", body)
        self.assertIn("positionPnlChart", body)
        self.assertIn("data-sparkline", body)
        self.assertIn("Most profitable outlook", body)

    def test_bullish_position_gets_strong_outlook(self):
        response = self._get(
            125.0,
            {"rsi": 30.0, "trend": "BULLISH", "macd_signal": "BULLISH",
             "volatility_level": "LOW"},
            {"score": 0.4, "label": "BULLISH"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertNotIn("Price-drop risk", body)
        self.assertIn("Strong outlook", body)
        self.assertIn("STRONG_BUY", body)


class SearchEndpointTests(unittest.TestCase):
    """Symbol autocomplete endpoint (no network: search service is mocked)."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()
        self.email = _unique_email("search")
        _register(self.client, self.email)

    def test_search_requires_login(self):
        response = self.app.test_client().get("/api/search?q=AAPL")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_search_returns_results(self):
        with mock.patch("stockpredictor.services.stocks.search_symbols",
                        return_value=[{"symbol": "AAPL", "name": "Apple Inc.",
                                       "exchange": "NASDAQ"}]):
            response = self.client.get("/api/search?q=AAP")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["symbol"], "AAPL")

    def test_search_empty_query_returns_empty(self):
        with mock.patch("stockpredictor.services.stocks.search_symbols") as mock_search:
            response = self.client.get("/api/search?q=")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["results"], [])
        mock_search.assert_not_called()

    def test_search_degrades_gracefully_on_error(self):
        with mock.patch("stockpredictor.services.stocks.search_symbols",
                        side_effect=RuntimeError("boom")):
            response = self.client.get("/api/search?q=AAPL")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["results"], [])


class SearchServiceTests(unittest.TestCase):
    """Offline fallback of the search service itself."""

    def test_local_search_matches_symbol_prefix(self):
        from stockpredictor.services.stocks import _local_search

        results = _local_search("AAP", 5)
        self.assertTrue(results)
        self.assertTrue(any(r["symbol"] == "AAPL" for r in results))

    def test_local_search_matches_name_substring(self):
        from stockpredictor.services.stocks import _local_search

        results = _local_search("nvidia", 5)
        self.assertTrue(results)
        self.assertTrue(any(r["symbol"] == "NVDA" for r in results))

    def test_local_search_respects_limit(self):
        from stockpredictor.services.stocks import _local_search

        results = _local_search("A", 3)
        self.assertLessEqual(len(results), 3)

    def test_local_search_empty_query(self):
        from stockpredictor.services.stocks import _local_search

        self.assertEqual(_local_search("", 5), [])


class WatchlistQuotesEndpointTests(unittest.TestCase):
    """REST fallback for the realtime quote stream."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()
        self.email = _unique_email("quotes")
        _register(self.client, self.email)

    def test_quotes_endpoint_requires_login(self):
        response = self.app.test_client().get("/api/watchlist/quotes")
        self.assertEqual(response.status_code, 302)

    def test_quotes_endpoint_returns_watchlist_quotes(self):
        user_store.watchlists()[self.email] = ["AAPL", "MSFT"]
        user_store.persist()
        with mock.patch("stockpredictor.services.stocks.get_quote",
                        side_effect=_fake_quote):
            response = self.client.get("/api/watchlist/quotes")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual([q["symbol"] for q in payload["quotes"]], ["AAPL", "MSFT"])
        self.assertIn("updated_at", payload)

    def test_quotes_endpoint_empty_watchlist(self):
        with mock.patch("stockpredictor.services.stocks.get_quote",
                        side_effect=_fake_quote):
            response = self.client.get("/api/watchlist/quotes")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["quotes"], [])


if __name__ == "__main__":
    unittest.main()
