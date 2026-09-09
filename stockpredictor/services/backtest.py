"""Walk-forward backtest engine (Feature: Backtesting).

Strategies are evaluated against historical OHLCV rows with configurable
commission and slippage. Everything here is pure Python/pandas so it can run
offline and be unit-tested without network access.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

DataFrameLike = Union[pd.DataFrame, List[Dict[str, Any]], Sequence[Dict[str, Any]]]

_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def _to_frame(rows: DataFrameLike) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            if col == "Volume":
                df[col] = 0.0
            elif col in ("Open", "High"):
                df[col] = df["Close"]
            else:
                df[col] = df["Close"]
    for col in _OHLCV:
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill()
    return df.sort_values("date" if "date" in df.columns else "Date")


def _dates(df: pd.DataFrame) -> List[str]:
    col = "date" if "date" in df.columns else "Date"
    if col in df.columns:
        return [str(d) for d in df[col].tolist()]
    return [str(i) for i in df.index.tolist()]


def _sma(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(window, min_periods=1).mean().to_numpy()


def _rsi(values: np.ndarray, window: int = 14) -> np.ndarray:
    delta = pd.Series(values).diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = (100 - 100 / (1 + rs)).fillna(50)
    return out.to_numpy()


def _macd(values: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    ema_fast = pd.Series(values).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(values).ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow).to_numpy()


def compute_signal_series(
    rows: DataFrameLike,
    strategy: str = "MA_CROSS",
    fast: int = 20,
    slow: int = 50,
    rsi_window: int = 14,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
) -> List[Dict[str, Any]]:
    """Return per-bar signal rows (``BUY`` / ``SELL`` / ``HOLD``) for the chart."""
    df = _to_frame(rows)
    if df.empty:
        return []
    closes = df["Close"].to_numpy(dtype=float)
    strategy = (strategy or "MA_CROSS").upper()
    dates = _dates(df)
    out: List[Dict[str, Any]] = []

    if strategy == "RSI":
        rsi = _rsi(closes, rsi_window)
        signals = ["HOLD"] * len(df)
        for i in range(1, len(df)):
            if rsi[i] < rsi_low and rsi[i - 1] >= rsi_low:
                signals[i] = "BUY"
            elif rsi[i] > rsi_high and rsi[i - 1] <= rsi_high:
                signals[i] = "SELL"
        for i in range(len(df)):
            out.append({"date": dates[i], "price": round(float(closes[i]), 4), "signal": signals[i], "rsi": round(float(rsi[i]), 2)})
        return out

    if strategy == "MACD":
        macd = _macd(closes)
        signals = ["HOLD"] * len(df)
        for i in range(1, len(df)):
            if macd[i] > 0 and macd[i - 1] <= 0:
                signals[i] = "BUY"
            elif macd[i] < 0 and macd[i - 1] >= 0:
                signals[i] = "SELL"
        for i in range(len(df)):
            out.append({"date": dates[i], "price": round(float(closes[i]), 4), "signal": signals[i], "macd": round(float(macd[i]), 4)})
        return out

    fast_sma = _sma(closes, max(1, int(fast)))
    slow_sma = _sma(closes, max(2, int(slow)))
    signals = ["HOLD"] * len(df)
    for i in range(1, len(df)):
        if fast_sma[i] > slow_sma[i] and fast_sma[i - 1] <= slow_sma[i - 1]:
            signals[i] = "BUY"
        elif fast_sma[i] < slow_sma[i] and fast_sma[i - 1] >= slow_sma[i - 1]:
            signals[i] = "SELL"
    for i in range(len(df)):
        out.append({
            "date": dates[i],
            "price": round(float(closes[i]), 4),
            "signal": signals[i],
            "sma_fast": round(float(fast_sma[i]), 4),
            "sma_slow": round(float(slow_sma[i]), 4),
        })
    return out


def compute_metrics(equity_values: Sequence[float], periods_per_year: float = 252, risk_free: float = 0.02) -> Dict[str, Any]:
    values = np.asarray([float(v) for v in equity_values])
    if len(values) < 2:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_volatility": 0.0,
        }
    returns = np.diff(values) / np.maximum(values[:-1], 1e-9)
    n = len(returns)
    years = max(n / periods_per_year, 1e-9)
    total_growth = values[-1] / values[0] if values[0] else 1.0
    cagr = float(total_growth ** (1.0 / years) - 1.0)
    vol = float(returns.std(ddof=0)) * math.sqrt(periods_per_year)
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0)) * math.sqrt(periods_per_year) if len(downside) else 0.0
    mean_ret = float(returns.mean()) * periods_per_year
    sharpe = float((mean_ret - risk_free) / vol) if vol > 0 else 0.0
    sortino = float((mean_ret - risk_free) / downside_vol) if downside_vol > 0 else 0.0

    running_max = np.maximum.accumulate(values)
    drawdowns = values / np.maximum(running_max, 1e-9) - 1.0
    max_dd = float(drawdowns.min())
    return {
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "annual_volatility": round(vol * 100, 2),
    }


def run_backtest(
    rows: DataFrameLike,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    strategy: str = "MA_CROSS",
    fast: int = 20,
    slow: int = 50,
) -> Dict[str, Any]:
    """Simulate a long-only strategy against historical rows.

    Returns a JSON-safe dict with the equity curve, benchmark (buy & hold)
    curve, trade log and performance metrics.
    """
    df = _to_frame(rows)
    if df.empty or len(df) < 30:
        return {"success": False, "message": "Not enough historical data to backtest"}

    closes = df["Close"].to_numpy(dtype=float)
    dates = _dates(df)
    signals = compute_signal_series(df, strategy=strategy, fast=fast, slow=slow)
    signal_map = [s["signal"] for s in signals]

    cash = float(initial_cash)
    shares = 0.0
    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []
    prev_signal = "HOLD"

    for i in range(len(df)):
        price = float(closes[i])
        signal = signal_map[i]
        if signal == "BUY" and prev_signal != "BUY" and shares == 0 and cash > price:
            fill = price * (1 + slippage)
            shares = (cash * (1 - commission)) / fill
            cost = shares * fill
            cash -= cost
            trades.append({"date": dates[i], "action": "BUY", "price": round(fill, 4), "shares": round(shares, 4), "value": round(cost, 2)})
        elif signal == "SELL" and prev_signal != "SELL" and shares > 0:
            fill = price * (1 - slippage)
            proceeds = shares * fill * (1 - commission)
            trades.append({"date": dates[i], "action": "SELL", "price": round(fill, 4), "shares": round(shares, 4), "value": round(proceeds, 2)})
            cash += proceeds
            shares = 0.0
        prev_signal = signal if signal != "HOLD" else prev_signal
        equity_curve.append({"date": dates[i], "equity": round(cash + shares * price, 2)})

    final_equity = cash + shares * closes[-1]
    initial_price = closes[0]
    buy_hold_end = float(initial_cash * closes[-1] / initial_price) if initial_price else float(initial_cash)
    buy_hold_curve = [{"date": dates[i], "equity": round(float(initial_cash * closes[i] / initial_price), 2) if initial_price else float(initial_cash)} for i in range(len(df))]

    metrics = compute_metrics([e["equity"] for e in equity_curve])
    win_loss: List[float] = []
    buys = [t for t in trades if t["action"] == "BUY"]
    sells = [t for t in trades if t["action"] == "SELL"]
    for buy, sell in zip(buys, sells):
        pnl_pct = (sell["price"] - buy["price"]) / buy["price"] * 100 if buy["price"] else 0.0
        win_loss.append(pnl_pct)
    wins = sum(1 for p in win_loss if p > 0)

    total_return = (final_equity / initial_cash - 1) * 100 if initial_cash else 0.0
    metrics.update({
        "total_return_pct": round(total_return, 2),
        "buy_and_hold_return_pct": round((buy_hold_end / initial_cash - 1) * 100, 2) if initial_cash else 0.0,
        "final_equity": round(final_equity, 2),
        "total_trades": len(trades),
        "round_trips": len(win_loss),
        "win_rate": round(wins / len(win_loss) * 100, 2) if win_loss else 0.0,
        "avg_trade_pct": round(float(np.mean(win_loss)), 2) if win_loss else 0.0,
    })

    return {
        "success": True,
        "symbol": df.get("symbol", "?").iloc[0] if "symbol" in df.columns and len(df) else None,
        "strategy": strategy,
        "parameters": {"fast": int(fast), "slow": int(slow)},
        "initial_cash": round(float(initial_cash), 2),
        "metrics": metrics,
        "equity_curve": equity_curve,
        "buy_hold_curve": buy_hold_curve,
        "trades": trades,
        "signals": signals,
    }
