"""Stock analysis blueprint: forecast forms and results rendering.

Long-running predictions run on a background worker thread so the browser gets
an instant response and polls :func:`job_status`; cached forecasts are rendered
immediately without a retrain.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from production_core import audit_log, peek_cached

from ..services import sentiment as sentiment_service
from ..services import stocks
from ..services.auth import user_store
from ..services.jobs import job_manager

logger = logging.getLogger("stockpredictor.views.analyze")

bp = Blueprint("analyze", __name__)

_ALLOWED_MODELS = {"AUTO", "LSTM", "GRU", "ENSEMBLE"}
_ALLOWED_RISK = {"low", "medium", "high"}
FORECAST_TTL = 900


def _parse_form(form, default_symbol: str = "AAPL") -> Tuple[str, str, int, str, str]:
    """Validate and normalize shared analysis form inputs.

    Returns (symbol, period, days, risk, model_type). Raises ValueError on bad input.
    """
    from production_core import sanitize_symbol

    raw_symbol = (form.get("symbol") or default_symbol).strip().upper()
    if not raw_symbol:
        raise ValueError("Please enter a stock symbol")
    symbol = sanitize_symbol(raw_symbol)

    period = (form.get("period") or "1y").strip() or "1y"
    try:
        days = int(form.get("days", 30))
    except (TypeError, ValueError):
        raise ValueError("Forecast period must be a number")

    min_days = current_app.config.get("MIN_FORECAST_DAYS", 1)
    max_days = current_app.config.get("MAX_FORECAST_DAYS", 365)
    if days < min_days or days > max_days:
        raise ValueError(f"Forecast period must be between {min_days} and {max_days} days")

    model_type = (form.get("model_type") or "AUTO").strip().upper()
    if model_type not in _ALLOWED_MODELS:
        raise ValueError("Unsupported model type")

    risk = (form.get("risk") or "medium").strip().lower()
    if risk not in _ALLOWED_RISK:
        risk = "medium"

    return symbol, period, days, risk, model_type


def _forecast_key(symbol: str, period: str, days: int, model_type: str) -> str:
    return f"forecast:{symbol}:{period}:{days}:{model_type}"


def _save_prediction_to_cache(symbol: str, report: Dict[str, Any]) -> None:
    """Write a prediction into the shared prediction cache and persist."""
    summary = report.get("executive_summary") or {}
    current = summary.get("current_price")
    predicted = summary.get("predicted_price")
    if current is None or predicted is None:
        return
    signal = "BUY" if predicted > current else "SELL"
    cache = user_store._state.setdefault("prediction_cache", {})
    cache[symbol] = {
        "predicted_price": float(predicted),
        "confidence": float(summary.get("confidence_score", 0.5)),
        "signal": signal,
        "sentiment": "BULLISH" if signal == "BUY" else "BEARISH",
        "recommendation": summary.get("investment_recommendation", signal),
        "timestamp": summary.get("analysis_date", ""),
    }
    user_store.persist(write_cache=True)


def _record_history(symbol: str, report: Dict[str, Any], days: int,
                    model_type: str, user_email: Optional[str]) -> None:
    if not user_email:
        return
    summary = report.get("executive_summary") or {}
    current = summary.get("current_price")
    predicted = summary.get("predicted_price")
    if current is None or predicted is None:
        return
    signal = "BUY" if predicted > current else "SELL"
    user_store.add_analysis(user_email, {
        "symbol": symbol,
        "signal": signal,
        "confidence": float(summary.get("confidence_score", 0.5)),
        "predicted_price": float(predicted),
        "current_price": float(current),
        "model_type": model_type,
        "days": days,
        "timestamp": summary.get("analysis_date") or "",
        "run_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


def _assemble_result(symbol: str, period: str, days: int, risk: str,
                     model_type: str, report: Dict[str, Any],
                     user_email: Optional[str] = None) -> Dict[str, Any]:
    """Bundle a forecast report with auxiliary analysis sections."""
    _save_prediction_to_cache(symbol, report)
    _record_history(symbol, report, days, model_type, user_email)
    audit_log("analysis.ran", user_email, {"symbol": symbol, "days": days})

    technical = {}
    sentiment = {}
    fundamentals = {}
    price_history = []
    try:
        technical = stocks.get_technical_analysis(symbol, period=period)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Technical analysis failed for %s: %s", symbol, exc)
    try:
        sentiment = sentiment_service.get_sentiment(symbol)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Sentiment failed for %s: %s", symbol, exc)
    try:
        fundamentals = stocks.get_company_fundamentals(symbol)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Fundamentals failed for %s: %s", symbol, exc)
    try:
        price_history = stocks.get_price_history(symbol, period=period)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Price history failed for %s: %s", symbol, exc)

    regime, model_selection, analyst_report = _enrich_advanced(
        symbol, price_history, report, technical, sentiment
    )

    return {
        "symbol": symbol,
        "params": {
            "period": period,
            "days": days,
            "risk": risk,
            "model_type": model_type,
        },
        "report": report,
        "technical": technical,
        "sentiment": sentiment,
        "fundamentals": fundamentals,
        "price_history": price_history,
        "regime": regime,
        "model_selection": model_selection,
        "analyst_report": analyst_report,
        "in_watchlist": bool(
            user_email and symbol in user_store.watchlists().get(user_email, [])
        ),
    }


def _enrich_advanced(symbol: str, price_history: list, report: Dict[str, Any],
                     technical: Dict[str, Any],
                     sentiment: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Compute the cheap advanced sections (regime, model pick, summary text).

    These are pure/pandas-only so they render instantly on the results page;
    the heavier features (backtest, Monte Carlo, SHAP, LLM report) are loaded
    on demand through the API endpoints.
    """
    from ..services.llm_report import rule_based_summary
    from ..services.regime import auto_select_model

    closes = []
    for row in price_history or []:
        try:
            closes.append(float(row["Close"]))
        except (TypeError, ValueError, KeyError):
            continue

    regime = {}
    model_selection = {}
    if len(closes) >= 30:
        try:
            from ..services.regime import auto_select_model, detect_regime

            detail = detect_regime(closes)
            if detail.get("success"):
                detection = auto_select_model(closes)
                model_selection = {**detail, **(detection if detection.get("success") else {})}
                regime = detail.get("regime")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Regime detection failed for %s: %s", symbol, exc)

    forecast_summary = report.get("executive_summary") or {}
    analyst_report = {"success": True, "provider": "rule_based"}
    try:
        analyst_report["report"] = rule_based_summary(
            symbol, forecast_summary, technical, sentiment
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Analyst summary failed for %s: %s", symbol, exc)
        analyst_report["report"] = "Summary unavailable."
    return regime, model_selection, analyst_report


def _run_analysis(symbol: str, period: str, days: int, risk: str,
                  model_type: str = "AUTO",
                  user_email: Optional[str] = None) -> Dict[str, Any]:
    """Execute the full analysis pipeline for one symbol (background job target)."""
    validation = stocks.validate_symbol(symbol)
    if not validation.get("valid"):
        return {
            "error": validation.get("error", "Symbol not found"),
            "symbol": symbol,
            "validation": validation,
        }

    report = stocks.predict_forecast(symbol, period, days, risk, model_type=model_type)
    if report.get("error"):
        return {"error": report["error"], "symbol": symbol}

    return _assemble_result(symbol, period, days, risk, model_type,
                            report, user_email=user_email)


@bp.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze_stock():
    return _analyze_endpoint(
        "analyze.html", "AI Stock Analysis",
        "AI-Powered Price Predictions & Insights",
        default_symbol="AAPL",
        action_url=url_for("analyze.analyze_stock"),
        error_endpoint="analyze.analyze_stock",
        error_back="analyze.analyze_stock",
    )


@bp.route("/analyze_any", methods=["GET", "POST"])
@login_required
def analyze_any_stock():
    return _analyze_endpoint(
        "analyze.html", "Global Stock Analysis",
        "Analyze any stock from markets worldwide",
        default_symbol="",
        action_url=url_for("analyze.analyze_any_stock"),
        error_endpoint="analyze.analyze_any_stock",
        error_back="analyze.analyze_any_stock",
    )


def _analyze_endpoint(template: str, title: str, subtitle: str,
                      default_symbol: str, action_url: str,
                      error_endpoint: str, error_back: str):
    if request.method == "POST":
        try:
            symbol, period, days, risk, model_type = _parse_form(request.form, default_symbol=default_symbol)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for(error_endpoint))

        if not symbol:
            flash("Please enter a stock symbol to analyze.", "error")
            return redirect(url_for(error_endpoint))

        forecast_key = _forecast_key(symbol, period, days, model_type)
        cached_report = peek_cached(forecast_key, ttl=FORECAST_TTL)
        if cached_report is not None:
            try:
                result = _assemble_result(symbol, period, days, risk, model_type,
                                          cached_report, user_email=current_user.email)
                return render_template("results.html", **result)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Instant render failed for %s: %s", symbol, exc)

        job_id = job_manager.submit(
            _run_analysis, symbol, period, days, risk, model_type,
            user_email=current_user.email,
            dedupe_key=forecast_key,
            owner=current_user.email,
        )
        return render_template(
            "analyzing.html",
            job_id=job_id,
            symbol=symbol,
            days=days,
            model_type=model_type,
            error_endpoint=error_endpoint,
        )

    symbol = request.args.get("symbol", default_symbol)
    return render_template(template, title=title, subtitle=subtitle,
                           action_url=action_url, symbol=symbol)


@bp.get("/results/<job_id>")
@login_required
def results_page(job_id: str):
    """Render stored analysis results, or the progress page while running."""
    status = job_manager.status(job_id, requester=current_user.email)
    if status is None:
        return render_template(
            "error.html",
            title="Analysis Not Found",
            message="This analysis is no longer available. Please run it again.",
            back_url=url_for("analyze.analyze_stock"),
        ), 404

    if status["status"] == "complete":
        result = status["result"] or {}
        if result.get("error"):
            return render_template(
                "error.html",
                title="Analysis Error",
                message=result["error"],
                back_url=url_for("analyze.analyze_stock"),
            ), 400
        return render_template("results.html", **result)

    if status["status"] == "error":
        return render_template(
            "error.html",
            title="Analysis Error",
            message=status["message"],
            back_url=url_for("analyze.analyze_stock"),
        ), 400

    return render_template(
        "analyzing.html",
        job_id=job_id,
        symbol=request.args.get("symbol", ""),
        days=request.args.get("days", ""),
        model_type=request.args.get("model_type", ""),
        error_endpoint="analyze.analyze_stock",
    )


@bp.get("/api/analysis/status/<job_id>")
@login_required
def job_status(job_id: str):
    """Lightweight JSON poll used by the analyzing page."""
    status = job_manager.status(job_id, requester=current_user.email)
    if status is None:
        return jsonify({"status": "missing"})
    return jsonify({
        "status": status["status"],
        "message": status["message"],
    })
