"""Stock Predictor Pro — refactored application package.

Use :func:`create_app` as the WSGI factory (also referenced by ``wsgi.py`` and
the Docker image).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from flask import Flask, render_template

from .config import Settings
from .extensions import init_extensions
from .views import register_blueprints


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )

    app.config.update(Settings().to_flask_config())
    if test_config:
        app.config.update(test_config)

    logger = app.logger
    if not logger.handlers:
        from production_core import configure_logging

        logger = configure_logging()

    init_extensions(app)
    register_blueprints(app)

    _register_error_handlers(app)
    _register_context_processors(app)
    _register_template_filters(app)

    return app


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(401)
    def unauthorized(error):
        return render_template(
            "error.html",
            title="Unauthorized",
            message="Please log in to access this page.",
            back_url="/",
        ), 401

    @app.errorhandler(403)
    def forbidden(error):
        return render_template(
            "error.html",
            title="Forbidden",
            message="You do not have permission to access this page.",
            back_url="/dashboard",
        ), 403

    @app.errorhandler(404)
    def not_found(error):
        return render_template(
            "error.html",
            title="Page Not Found",
            message="The page you requested could not be found.",
            back_url="/",
        ), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.exception("Unhandled error: %s", error)
        return render_template(
            "error.html",
            title="Internal Server Error",
            message="Something went wrong on our end. Please try again.",
            back_url="/",
        ), 500


def _register_context_processors(app: Flask) -> None:
    from flask_login import current_user

    @app.context_processor
    def inject_globals():
        return {
            "current_user": current_user,
            "app_name": "Stock Predictor Pro",
        }


def _is_number(value) -> Optional[float]:
    """Return a finite float for usable numeric input, else None.

    ``None``, NaN, infinity and non-numeric values all yield ``None`` so the
    template filters can render a clean em-dash instead of ``nan``.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _register_template_filters(app: Flask) -> None:
    @app.template_filter("fmt")
    def fmt_filter(value, nd: int = 2):
        number = _is_number(value)
        if number is None:
            return "—"
        return f"{number:.{nd}f}"

    @app.template_filter("pct")
    def pct_filter(value, nd: int = 1):
        number = _is_number(value)
        if number is None:
            return "—"
        return f"{number * 100:.{nd}f}%"

    @app.template_filter("money")
    def money_filter(value):
        number = _is_number(value)
        if number is None:
            return "—"
        sign = "-" if number < 0 else ""
        abs_number = abs(number)
        if abs_number >= 1e12:
            return f"{sign}${abs_number / 1e12:.2f}T"
        if abs_number >= 1e9:
            return f"{sign}${abs_number / 1e9:.2f}B"
        if abs_number >= 1e6:
            return f"{sign}${abs_number / 1e6:.2f}M"
        if abs_number >= 1e3:
            return f"{sign}${abs_number / 1e3:.1f}K"
        return f"{number:,.2f}"

    @app.template_filter("ymd")
    def ymd_filter(value):
        text = str(value)
        return text[:10]

    @app.template_filter("host")
    def host_filter(value):
        text = str(value or "").replace("https://", "").replace("http://", "").rstrip("/")
        return text or "—"
