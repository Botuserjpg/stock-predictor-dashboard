"""Market analytics blueprint."""
from __future__ import annotations

from flask import Blueprint, render_template
from flask_login import login_required

from ..services import analytics as analytics_service

bp = Blueprint("analytics", __name__)

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "RELIANCE.NS"]


@bp.get("/analytics")
@login_required
def page():
    overview = analytics_service.market_overview()
    technical = analytics_service.technical_overview(DEFAULT_SYMBOLS)
    return render_template(
        "analytics.html",
        indices=overview.get("indices", {}),
        sectors=overview.get("sectors", {}),
        sentiment=overview.get("sentiment", {}),
        technical=technical,
        symbols=DEFAULT_SYMBOLS,
    )
