"""Unit tests for the shared directional-accuracy metric (metrics_eval)."""
import math
import unittest

import numpy as np

from metrics_eval import directional_accuracy


class DirectionalAccuracyTests(unittest.TestCase):
    def test_hand_made_example_without_base(self):
        # actual = [100, 102, 102, 99, 105]
        # predicted = [101, 100, 103, 100, 104]
        # Direction of point i>=1 compares pred_i vs actual_{i-1} with actual_i vs actual_{i-1}:
        #   i=1: base 100, actual +2 (up), pred 100 (flat) -> miss
        #   i=2: base 102, actual 0 (flat), pred 103 (up)   -> miss
        #   i=3: base 102, actual -3 (down), pred -2 (down) -> hit
        #   i=4: base 99, actual +6 (up), pred +5 (up)      -> hit
        # => 2 hits / 4 points = 50.0%
        actual = np.array([100, 102, 102, 99, 105], dtype=float)
        predicted = np.array([101, 100, 103, 100, 104], dtype=float)
        pct, correct, total = directional_accuracy(predicted, actual)
        self.assertEqual(total, 4)
        self.assertEqual(correct, 2)
        self.assertAlmostEqual(pct, 50.0)

    def test_hand_made_example_with_explicit_base(self):
        # base = [98, 100, 102, 102, 99] (previous closes known at prediction time)
        #   i=0: base 98, actual 100 (up), pred 101 (up)   -> hit
        #   i=1: base 100, actual 102 (up), pred 100 (flat) -> miss
        #   i=2: base 102, actual 102 (flat), pred 103 (up) -> miss
        #   i=3: base 102, actual 99 (down), pred 100 (down) -> hit
        #   i=4: base 99, actual 105 (up), pred 104 (up)    -> hit
        # => 3 hits / 5 points = 60.0%
        actual = np.array([100, 102, 102, 99, 105], dtype=float)
        predicted = np.array([101, 100, 103, 100, 104], dtype=float)
        base = np.array([98, 100, 102, 102, 99], dtype=float)
        pct, correct, total = directional_accuracy(predicted, actual, base=base)
        self.assertEqual(total, 5)
        self.assertEqual(correct, 3)
        self.assertAlmostEqual(pct, 60.0)

    def test_zero_move_returns_flat(self):
        # Flat move (actual == base) means direction is 0; it matches pred 0 only.
        actual = np.array([100, 100], dtype=float)
        predicted = np.array([101, 100], dtype=float)
        pct, correct, total = directional_accuracy(predicted, actual)
        self.assertEqual(total, 1)
        self.assertEqual(correct, 1)
        self.assertAlmostEqual(pct, 100.0)

    def test_too_short_is_nan(self):
        pct, correct, total = directional_accuracy(np.array([1.0]), np.array([1.0]))
        self.assertEqual(total, 0)
        self.assertEqual(correct, 0)
        self.assertTrue(math.isnan(pct))


if __name__ == "__main__":
    unittest.main()