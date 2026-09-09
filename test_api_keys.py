"""
Optional live API credential checks.

These checks call third-party services and require real keys in the environment,
so they are skipped during normal test discovery. Run them explicitly with:

    RUN_EXTERNAL_TESTS=1 python -m unittest test_api_keys -v
"""

import os
import unittest

import requests
from dotenv import load_dotenv

try:
    import praw
except ImportError:  # pragma: no cover - optional dependency
    praw = None


load_dotenv()


def _external_tests_enabled() -> bool:
    return os.getenv("RUN_EXTERNAL_TESTS") == "1"


@unittest.skipUnless(_external_tests_enabled(), "set RUN_EXTERNAL_TESTS=1 to run live API checks")
class APIKeyIntegrationTests(unittest.TestCase):
    timeout = 10

    def test_newsapi_key(self):
        key = os.getenv("NEWSAPI_KEY")
        self.assertTrue(key, "NEWSAPI_KEY is not configured")
        response = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "apiKey": key},
            timeout=self.timeout,
        )
        data = response.json()
        self.assertIn("articles", data)

    def test_alpha_vantage_key(self):
        key = os.getenv("ALPHAVANTAGE_KEY")
        self.assertTrue(key, "ALPHAVANTAGE_KEY is not configured")
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "TIME_SERIES_DAILY", "symbol": "AAPL", "apikey": key},
            timeout=self.timeout,
        )
        self.assertIn("Time Series (Daily)", response.json())

    def test_finnhub_key(self):
        key = os.getenv("FINNHUB_KEY")
        self.assertTrue(key, "FINNHUB_KEY is not configured")
        response = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": "AAPL", "token": key},
            timeout=self.timeout,
        )
        self.assertIn("c", response.json())

    def test_fmp_key(self):
        key = os.getenv("FMP_KEY")
        self.assertTrue(key, "FMP_KEY is not configured")
        response = requests.get(
            "https://financialmodelingprep.com/api/v3/quote/AAPL",
            params={"apikey": key},
            timeout=self.timeout,
        )
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("price", data[0])

    def test_polygon_key(self):
        key = os.getenv("POLYGON_KEY")
        self.assertTrue(key, "POLYGON_KEY is not configured")
        response = requests.get(
            "https://api.polygon.io/v3/reference/tickers",
            params={"limit": 1, "apiKey": key},
            timeout=self.timeout,
        )
        self.assertIn("results", response.json())

    def test_tiingo_key(self):
        key = os.getenv("TIINGO_KEY")
        self.assertTrue(key, "TIINGO_KEY is not configured")
        response = requests.get(
            "https://api.tiingo.com/tiingo/daily/AAPL/prices",
            params={"token": key},
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    @unittest.skipIf(praw is None, "praw is not installed")
    def test_reddit_credentials(self):
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.assertTrue(client_id and client_secret, "Reddit credentials are not configured")
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="Stock Predictor App",
        )
        post = next(reddit.subreddit("stocks").hot(limit=1))
        self.assertTrue(post.title)

    def test_twitter_bearer_token(self):
        token = os.getenv("TWITTER_BEARER_TOKEN")
        self.assertTrue(token, "TWITTER_BEARER_TOKEN is not configured")
        response = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            params={"query": "Apple", "max_results": 10},
            headers={"Authorization": f"Bearer {token}"},
            timeout=self.timeout,
        )
        self.assertEqual(response.status_code, 200, response.text[:200])
        self.assertIn("data", response.json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
