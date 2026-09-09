"""Model explainability (Feature: SHAP-based feature attribution).

Computes per-feature importance for the features the models actually consume.
SHAP is used when installed and cheap enough to run; otherwise a sklearn
permutation / gradient-boosting proxy is used, so this always works offline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .feature_store import FEATURE_NAMES, build_features, log_feature_importance

logger = logging.getLogger("stockpredictor.services.explain")

try:  # optional dependency, may not be installed in every environment
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    SHAP_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    SKLEARN_AVAILABLE = False


def _feature_frame(rows) -> np.ndarray:
    df = build_features(rows)
    numeric = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    return numeric.dropna(how="all")


def explain_forecast(
    rows,
    n_top: int = 8,
    persist: bool = True,
    symbol: str = "",
) -> Dict[str, Any]:
    """Return ranked feature attributions for the most recent forecast window.

    Falls back through: SHAP TreeExplainer -> RandomForest importance -> linear
    correlation, so a result is always produced for display.
    """
    frame = _feature_frame(rows)
    if frame.shape[0] < 30 or frame.shape[1] < 3:
        return {"success": False, "message": "Not enough feature history to explain"}

    X = frame.to_numpy(dtype=float)
    n_nonnan = np.sum(~np.isnan(X), axis=0)
    col_mean = np.where(n_nonnan > 0, np.nansum(X, axis=0) / np.maximum(n_nonnan, 1), 0.0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    y = frame["Return_1d"].shift(-1).to_numpy(dtype=float)
    valid = np.isfinite(y)
    X, y = X[valid], y[valid]
    if len(X) < 30:
        return {"success": False, "message": "Not enough labeled examples after cleaning"}
    feature_names = list(frame.columns)

    method = "correlation"
    scores: Dict[str, float] = {}
    try:
        corr = np.nan_to_num(np.corrcoef(X, y)[0, 1:])
        scores = {name: float(abs(c)) for name, c in zip(feature_names, corr)}
    except Exception:
        scores = {name: 0.0 for name in feature_names}

    if SKLEARN_AVAILABLE and len(X) >= 60:
        try:
            model = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=8)
            model.fit(X, y)
            importance = model.feature_importances_
            if float(importance.sum()) > 0:
                scores = {name: float(imp) for name, imp in zip(feature_names, importance)}
                method = "random_forest_permutation_proxy"
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("RF importance failed, using correlation: %s", exc)

    if SHAP_AVAILABLE and SKLEARN_AVAILABLE and len(X) >= 60:
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            background = X[-min(64, len(X)):]
            shap_values = explainer.shap_values(background)
            if hasattr(shap_values, "shape") and shap_values.ndim > 1:
                mean_abs = np.abs(shap_values).mean(axis=0)
                if float(mean_abs.sum()) > 0:
                    scores = {name: float(v) for name, v in zip(feature_names, mean_abs)}
                    method = "shap_tree_explainer"
        except Exception as exc:  # pragma: no cover - SHAP env dependent
            logger.debug("SHAP explainer failed, using proxy: %s", exc)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:n_top]
    total = sum(v for _, v in ranked) or 1.0
    importance = {name: round(v, 6) for name, v in ranked}
    result: Dict[str, Any] = {
        "success": True,
        "method": method,
        "symbol": symbol,
        "importance": importance,
        "top": [
            {"name": name, "value": round(v, 6), "share_pct": round(v / total * 100, 2)}
            for name, v in ranked
        ],
        "features": list(importance.keys()),
    }
    if persist and symbol:
        log_feature_importance(symbol, importance)
    return result
