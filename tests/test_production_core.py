import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_governance import (
    add_advanced_features,
    clean_market_data,
    probabilistic_intervals,
    select_features_by_correlation,
    validate_market_data,
)
from production_core import RequestIdFilter, audit_log, sanitize_symbol


class ProductionCoreTests(unittest.TestCase):
    def test_symbol_validation_accepts_market_suffixes(self):
        self.assertEqual(sanitize_symbol("aapl"), "AAPL")
        self.assertEqual(sanitize_symbol("reliance.ns"), "RELIANCE.NS")

    def test_symbol_validation_rejects_injection_like_values(self):
        with self.assertRaises(ValueError):
            sanitize_symbol("AAPL;DROP")

    def test_request_id_filter_works_without_request_context(self):
        import logging

        record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)
        self.assertTrue(RequestIdFilter().filter(record))
        self.assertEqual(record.request_id, "-")

    def test_audit_log_works_without_request_context(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_path = Path(tmp_dir) / "audit"
            with patch("production_core.AUDIT_DIR", audit_path):
                audit_log("unit.test", "user@example.com", {"ok": True})

            files = list(audit_path.glob("audit_*.jsonl"))
            self.assertEqual(len(files), 1)
            self.assertIn('"event": "unit.test"', files[0].read_text(encoding="utf-8"))


class MLGovernanceTests(unittest.TestCase):
    def _sample_data(self):
        index = pd.date_range("2024-01-01", periods=90, freq="D")
        close = pd.Series(np.linspace(100, 130, 90), index=index)
        return pd.DataFrame(
            {
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": np.linspace(1000, 2000, 90),
            },
            index=index,
        )

    def test_market_data_validation_and_cleaning(self):
        data = self._sample_data()
        data.iloc[5, data.columns.get_loc("Close")] = np.nan
        report = validate_market_data(data, "AAPL")
        self.assertTrue(report["is_valid"])
        cleaned = clean_market_data(data)
        self.assertFalse(cleaned["Close"].isna().any())

    def test_advanced_features_and_intervals(self):
        data = add_advanced_features(self._sample_data())
        self.assertIn("Momentum_10d", data.columns)
        selected = select_features_by_correlation(data)
        self.assertIn("Close", selected)
        intervals = probabilistic_intervals(np.array([101.0, 102.0]), data["Close"])
        self.assertEqual(len(intervals["lower"]), 2)
        self.assertEqual(len(intervals["upper"]), 2)

    def test_feature_selection_handles_constant_target(self):
        data = pd.DataFrame(
            {
                "Close": [100.0] * 30,
                "Volume": np.linspace(1000, 2000, 30),
                "Momentum": [1.0] * 30,
            }
        )

        self.assertEqual(select_features_by_correlation(data), ["Close"])


if __name__ == "__main__":
    unittest.main()
