"""Shared directional-accuracy metric.

Definition (matches the held-out backtest/audit in ``backtest_ml``):
    a hit at forecast point ``t``  <=>  sign(pred_t - close_{t-1}) == sign(close_t - close_{t-1})

i.e. the predicted move direction equals the realised move direction, where
``close_{t-1}`` is the previous day's ACTUAL close (the "base" price that was
known at prediction time). Comparing the fraction of hits against 50% is the
correct null hypothesis, and this is the only definition that is meaningful for
a one-step-ahead price forecast.

Usage:
    accuracy_percent, correct, total = directional_accuracy(predicted, actual)

    If the caller knows the previous-close (base) for every point -- e.g. the
    first point of a held-out window whose base was the last in-sample close --
    pass it explicitly so NO point is wasted:

    accuracy_percent, correct, total = directional_accuracy(predicted, actual, base=base)

When ``base`` is not supplied it is derived as ``actual`` shifted by one row and
the first point (which has no previous close) is skipped. Points with no
previous close therefore never count as either a hit or a miss.
"""

import numpy as np


def directional_accuracy(predicted, actual, base=None):
    """Return ``(accuracy_percent, correct, total)`` for one-step forecasts.

    ``predicted`` and ``actual`` must be same-length 1-D arrays of prices.
    ``base`` (optional) must be same length: the actual close known when each
    prediction was made (``actual`` shifted by one). Returns ``(nan, 0, 0)``
    when there is nothing to evaluate.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if predicted.ndim != 1 or actual.ndim != 1:
        raise ValueError("predicted and actual must be 1-D arrays")
    if predicted.shape[0] != actual.shape[0]:
        raise ValueError("predicted and actual must have the same length")

    if base is not None:
        base = np.asarray(base, dtype=float)
        if base.shape[0] != predicted.shape[0]:
            raise ValueError("base must have the same length as predicted/actual")
        pred_dir = np.sign(predicted - base)
        act_dir = np.sign(actual - base)
        total = int(actual.shape[0])
    else:
        if actual.shape[0] < 2:
            return float("nan"), 0, 0
        prev = actual[:-1]
        pred_dir = np.sign(predicted[1:] - prev)
        act_dir = np.sign(actual[1:] - prev)
        total = int(actual.shape[0] - 1)

    correct = int(np.count_nonzero(pred_dir == act_dir))
    return float(correct / total * 100), correct, total