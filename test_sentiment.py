"""
Optional live sentiment integration checks.

The sentiment pipeline may call network APIs and load heavy ML dependencies.
Normal unit test discovery skips these checks. Run explicitly with:

    RUN_EXTERNAL_TESTS=1 python -m unittest test_sentiment -v
"""

import os
import unittest

import pandas as pd


@unittest.skipUnless(os.getenv("RUN_EXTERNAL_TESTS") == "1", "set RUN_EXTERNAL_TESTS=1 to run live checks")
class SentimentIntegrationTests(unittest.TestCase):
    def test_enhanced_sentiment_contract(self):
        from model_utils import add_sentiment_features, get_enhanced_sentiment

        sentiment_score, sources = get_enhanced_sentiment("AAPL")

        self.assertIsInstance(sentiment_score, (float, int))
        self.assertGreaterEqual(sentiment_score, -1.0)
        self.assertLessEqual(sentiment_score, 1.0)
        self.assertIsInstance(sources, list)

        data = pd.DataFrame(
            {
                "Close": [150, 151, 152, 153, 154],
                "Volume": [1_000_000, 1_200_000, 1_100_000, 1_300_000, 1_400_000],
            }
        )
        data_with_sentiment = add_sentiment_features(data, "AAPL")

        self.assertIn("News_Sentiment", data_with_sentiment.columns)
        self.assertIn("Social_Buzz", data_with_sentiment.columns)
        self.assertIn("Sentiment_Strength", data_with_sentiment.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
