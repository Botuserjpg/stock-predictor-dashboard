"""Feature store (Feature: Feature store).

Centralizes the OHLCV -> feature transform used by the explainability, drift
and backtest modules, and persists feature-importance snapshots to disk so the
UI can show how each model "sees" a symbol over time.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from production_core import STATE_DIR

logger = logging.getLogger("stockpredictor.services.feature_store")

FEATURE_DIR = STATE_DIR / "feature_store"
IMPORTANCE_DIR = STATE_DIR / "feature_importance"

FEATURE_NAMES = [
    "Return_1d",
    "Return_5d",
    "Return_20d",
    "Volatility_5d",
    "Volatility_20d",
    "Volatility_Ratio",
    "Momentum_10d",
    "Momentum_30d",
    "Volume_ZScore_20d",
    "Price_ZScore_20d",
    "Drawdown_60d",
    "Trend_Strength_20d",
    "RSI_14",
    "RSI_7",
]

DataFrameLike = Union[pd.DataFrame, List[Dict[str, Any]], Sequence[Dict[str, Any]]]


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
    return df


def build_features(rows: DataFrameLike) -> pd.DataFrame:
    """Build indicator features from OHLCV rows; one row per input row."""
    df = _to_frame(rows)
    if df.empty or "Close" not in df.columns:
        return df
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df.get("Volume", pd.Series(index=df.index, data=0)), errors="coerce").fillna(0)
    returns = close.pct_change()

    df["Return_1d"] = returns
    df["Return_5d"] = close.pct_change(5)
    df["Return_20d"] = close.pct_change(20)
    df["Log_Return_1d"] = np.log(close / close.shift(1))
    df["Volatility_5d"] = returns.rolling(5).std()
    df["Volatility_20d"] = returns.rolling(20).std()
    df["Volatility_Ratio"] = df["Volatility_5d"] / df["Volatility_20d"].replace(0, np.nan)
    df["Momentum_10d"] = close / close.shift(10) - 1
    df["Momentum_30d"] = close / close.shift(30) - 1
    df["Volume_ZScore_20d"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0, np.nan)
    df["Price_ZScore_20d"] = (close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan)
    df["Drawdown_60d"] = close / close.rolling(60).max() - 1
    df["Trend_Strength_20d"] = close.rolling(20).apply(_trend_strength, raw=True)
    df["RSI_14"] = _rsi(close.to_numpy(dtype=float), 14)
    df["RSI_7"] = _rsi(close.to_numpy(dtype=float), 7)
    return df


def _trend_strength(values: np.ndarray) -> float:
    if len(values) < 2 or np.isnan(values).any():
        return np.nan
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    mean = float(np.nanmean(values))
    return float(slope / mean) if mean else 0.0


def _rsi(values: np.ndarray, window: int = 14) -> pd.Series:
    delta = pd.Series(values).diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_feature_importance(symbol: str, importance: Dict[str, float]) -> Dict[str, Any]:
    """Persist a feature-importance snapshot under runtime_state/feature_importance/."""
    safe = (symbol or "UNKNOWN").replace(".", "_").upper()
    try:
        IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "symbol": safe,
            "timestamp": _now(),
            "importance": {name: round(float(value), 6) for name, value in importance.items()},
        }
        path = IMPORTANCE_DIR / f"{safe}.json"
        history = []
        if path.exists():
            try:
                history = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                history = []
        history.append(record)
        history = history[-50:]
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        tmp.replace(path)
        return {"success": True, "record": record}
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        logger.warning("Could not persist feature importance for %s: %s", safe, exc)
        return {"success": False, "message": str(exc)}


def feature_importance_history(symbol: str) -> List[Dict[str, Any]]:
    safe = (symbol or "UNKNOWN").replace(".", "_").upper()
    path = IMPORTANCE_DIR / f"{safe}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def latest_feature_importance(symbol: str) -> Optional[Dict[str, Any]]:
    history = feature_importance_history(symbol)
    return history[-1] if history else None


def list_stored_symbols() -> List[str]:
    if not IMPORTANCE_DIR.exists():
        return []
    return sorted(p.stem for p in IMPORTANCE_DIR.glob("*.json"))


def _safe_dir() -> Path:
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    return IMPORTANCE_DIR
