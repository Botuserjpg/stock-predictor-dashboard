import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REGISTRY_DIR = BASE_DIR / "model_registry"
QUALITY_DIR = BASE_DIR / "data_quality"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_fingerprint(data: pd.DataFrame) -> str:
    if data.empty:
        return "empty"
    sample = data.tail(min(len(data), 512)).to_csv(index=True).encode("utf-8")
    return hashlib.sha256(sample).hexdigest()[:16]


def validate_market_data(data: pd.DataFrame, symbol: str = "") -> Dict[str, Any]:
    required = ["Open", "High", "Low", "Close", "Volume"]
    report = {
        "symbol": symbol,
        "timestamp": _now(),
        "rows": int(len(data)),
        "columns": list(data.columns),
        "missing_required_columns": [col for col in required if col not in data.columns],
        "missing_values": {},
        "outlier_counts": {},
        "duplicate_index_count": 0,
        "is_valid": True,
        "warnings": [],
        "fingerprint": dataset_fingerprint(data),
    }

    if data.empty:
        report["is_valid"] = False
        report["warnings"].append("Dataset is empty")
        return report

    report["duplicate_index_count"] = int(data.index.duplicated().sum())
    for col in [c for c in required if c in data.columns]:
        series = pd.to_numeric(data[col], errors="coerce")
        report["missing_values"][col] = int(series.isna().sum())
        if col != "Volume" and (series <= 0).any():
            report["warnings"].append(f"{col} contains non-positive values")

        clean = series.dropna()
        if len(clean) > 10:
            q1, q3 = clean.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                low, high = q1 - 3 * iqr, q3 + 3 * iqr
                report["outlier_counts"][col] = int(((clean < low) | (clean > high)).sum())
            else:
                report["outlier_counts"][col] = 0

    if report["missing_required_columns"]:
        report["is_valid"] = False
    if sum(report["missing_values"].values()) > max(3, len(data) * 0.05):
        report["warnings"].append("Dataset has elevated missing-value rate")

    return report


def clean_market_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.sort_index()
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in cleaned.columns]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned[numeric_cols] = cleaned[numeric_cols].ffill().bfill()

    for col in [c for c in ["Open", "High", "Low", "Close"] if c in cleaned.columns]:
        returns = cleaned[col].pct_change()
        extreme = returns.abs() > 0.35
        if extreme.any():
            cleaned.loc[extreme, col] = np.nan
            cleaned[col] = cleaned[col].ffill().bfill()
    if "Volume" in cleaned.columns:
        cleaned["Volume"] = cleaned["Volume"].clip(lower=0)
    return cleaned


def add_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if "Close" not in df.columns:
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
    df["Anomaly_Score"] = df[["Price_ZScore_20d", "Volume_ZScore_20d"]].abs().mean(axis=1)
    return df


def _trend_strength(values: np.ndarray) -> float:
    if len(values) < 2 or np.isnan(values).any():
        return np.nan
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    return float(slope / np.nanmean(values)) if np.nanmean(values) else 0.0


def select_features_by_correlation(data: pd.DataFrame, target: str = "Close", max_features: int = 14) -> List[str]:
    if target not in data.columns:
        return [target]
    numeric = data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if target not in numeric.columns:
        return [target]
    returns_target = numeric[target].pct_change().shift(-1)
    if returns_target.dropna().nunique() < 2:
        return [target]

    scores: Dict[str, float] = {}
    for col in numeric.columns:
        if col == target:
            continue
        series = numeric[col]
        if series.dropna().nunique() < 2:
            continue
        corr = numeric[col].corr(returns_target)
        if pd.notna(corr):
            scores[col] = abs(float(corr))
    ranked = [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
    selected = [target] + ranked[: max(0, max_features - 1)]
    return selected


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    columns = columns or [c for c in ["Close", "Volume", "Return_1d", "Volatility_20d"] if c in current.columns]
    drift = {"timestamp": _now(), "columns": {}, "drift_detected": False}
    for col in columns:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = pd.to_numeric(reference[col], errors="coerce").dropna()
        cur = pd.to_numeric(current[col], errors="coerce").dropna()
        if len(ref) < 20 or len(cur) < 20:
            continue
        ref_mean, ref_std = ref.mean(), ref.std()
        cur_mean = cur.mean()
        z_shift = abs((cur_mean - ref_mean) / ref_std) if ref_std else 0.0
        drift["columns"][col] = {
            "reference_mean": float(ref_mean),
            "current_mean": float(cur_mean),
            "z_shift": float(z_shift),
            "drift": bool(z_shift >= 2.0),
        }
        drift["drift_detected"] = drift["drift_detected"] or z_shift >= 2.0
    return drift


def probabilistic_intervals(predictions: np.ndarray, recent_prices: pd.Series, confidence: float = 0.9) -> Dict[str, List[float]]:
    preds = np.asarray(predictions, dtype=float)
    returns = pd.to_numeric(recent_prices, errors="coerce").pct_change().dropna()
    daily_vol = float(returns.tail(60).std()) if len(returns) else 0.02
    z = 1.64 if confidence >= 0.9 else 1.28
    horizon = np.sqrt(np.arange(1, len(preds) + 1))
    spread = preds * daily_vol * z * horizon
    return {
        "lower": np.maximum(preds - spread, 0).tolist(),
        "upper": (preds + spread).tolist(),
        "daily_volatility": daily_vol,
        "confidence": confidence,
    }


@dataclass
class ModelRegistryRecord:
    symbol: str
    model_type: str
    version: str
    created_at: str
    metrics: Dict[str, Any]
    feature_columns: List[str]
    data_fingerprint: str
    artifact_path: Optional[str] = None


def register_model(
    symbol: str,
    model_type: str,
    metrics: Dict[str, Any],
    feature_columns: List[str],
    data: pd.DataFrame,
    artifact_path: Optional[str] = None,
) -> Dict[str, Any]:
    REGISTRY_DIR.mkdir(exist_ok=True)
    version = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{dataset_fingerprint(data)}"
    record = ModelRegistryRecord(
        symbol=symbol,
        model_type=model_type,
        version=version,
        created_at=_now(),
        metrics=metrics,
        feature_columns=feature_columns,
        data_fingerprint=dataset_fingerprint(data),
        artifact_path=artifact_path,
    )
    safe_symbol = symbol.replace(".", "_").upper() or "UNKNOWN"
    path = REGISTRY_DIR / f"{safe_symbol}_{model_type}_{version}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2, sort_keys=True)
    return asdict(record)


def save_quality_report(report: Dict[str, Any]) -> None:
    QUALITY_DIR.mkdir(exist_ok=True)
    safe_symbol = (report.get("symbol") or "UNKNOWN").replace(".", "_").upper()
    path = QUALITY_DIR / f"{safe_symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
