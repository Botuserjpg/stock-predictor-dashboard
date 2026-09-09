"""Blueprint registration."""
from __future__ import annotations

from flask import Flask


def register_blueprints(app: Flask) -> None:
    from .main import bp as main_bp
    from .auth import bp as auth_bp
    from .dashboard import bp as dashboard_bp
    from .analyze import bp as analyze_bp
    from .compare import bp as compare_bp
    from .watchlist import bp as watchlist_bp
    from .analytics import bp as analytics_bp
    from .portfolio import bp as portfolio_bp
    from .api import bp as api_bp
    from .admin import bp as admin_bp

    for bp in (main_bp, auth_bp, dashboard_bp, analyze_bp, compare_bp,
               watchlist_bp, analytics_bp, portfolio_bp, api_bp, admin_bp):
        app.register_blueprint(bp)
