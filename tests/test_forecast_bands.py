"""Unit tests for the probabilistic forecast fan-interval bands."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import production_core as pc

_TMP_DIR = tempfile.mkdtemp(prefix="forecast_bands_")
pc.STATE_DIR = Path(_TMP_DIR)
pc.STATE_FILE = pc.STATE_DIR / "app_state.json"

from stockpredictor.services import stocks  # noqa: E402


class _FakeMu:
    def get_stock_data(self, *args, **kwargs):
        index = pd.date_range("2024-01-01", periods=260, freq="D")
        return pd.DataFrame({"Close": np.linspace(100, 130, 260)}, index=index)


class ForecastBandsTests(unittest.TestCase):
    def test_bands_are_generated_for_forecast(self):
        forecast_df = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=10),
                "Predicted_Price": np.linspace(120, 150, 10),
            }
        )
        with patch.object(stocks, "_load_modules", return_value=(_FakeMu(), None)):
            bands = stocks._build_forecast_bands(forecast_df, "TEST")
        self.assertIsNotNone(bands)
        self.assertEqual(len(bands["lower"]), 10)
        self.assertEqual(len(bands["upper"]), 10)
        for low, high in zip(bands["lower"], bands["upper"]):
            self.assertLessEqual(low, high)

    def test_bands_attached_to_report_rows(self):
        forecast_df = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=5),
                "Predicted_Price": np.linspace(100, 110, 5),
            }
        )
        with patch.object(stocks, "_load_modules", return_value=(_FakeMu(), None)):
            report = {"executive_summary": {}}
            rows = stocks._df_to_rows(forecast_df)
            bands = stocks._build_forecast_bands(forecast_df, "TEST")
            if bands and rows:
                for row, (low, high) in zip(rows, zip(bands["lower"], bands["upper"])):
                    row["Low"] = round(low, 2)
                    row["High"] = round(high, 2)
                report["forecast_bands"] = {
                    "daily_volatility": bands.get("daily_volatility"),
                    "confidence": bands.get("confidence"),
                }
            report["forecast_rows"] = rows

        self.assertIn("forecast_bands", report)
        self.assertIn("Low", report["forecast_rows"][0])
        self.assertIn("High", report["forecast_rows"][0])

    def test_bands_none_for_empty_forecast(self):
        self.assertIsNone(stocks._build_forecast_bands(pd.DataFrame(), "TEST"))


if __name__ == "__main__":
    unittest.main()
