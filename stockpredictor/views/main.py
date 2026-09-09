"""Main / health / metrics blueprint."""
from __future__ import annotations

from flask import Blueprint, Response, abort, current_app, jsonify, redirect, request, url_for
from flask_login import current_user

from production_core import request_metrics, runtime_health

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.home"))
    if current_app.config.get("PUBLIC_DEMO"):
        return redirect(url_for("dashboard.demo"))
    return redirect(url_for("auth.login"))


@bp.get("/health")
def health():
    return jsonify(runtime_health())


@bp.get("/ready")
def ready():
    health_info = runtime_health()
    status_code = 200 if health_info.get("status") == "ok" else 503
    return jsonify(health_info), status_code


@bp.get("/metrics")
def metrics():
    """Prometheus text exposition (opt-in via METRICS_ENABLED=true).

    When METRICS_TOKEN is set, requests must present it in the
    ``X-Metrics-Token`` header (a scrape-config `authorization` header works too).
    """
    from flask import current_app

    if not current_app.config.get("METRICS_ENABLED"):
        abort(404)
    token = current_app.config.get("METRICS_TOKEN")
    if token:
        presented = (
            request.headers.get("X-Metrics-Token")
            or request.headers.get("Authorization", "").removeprefix("Bearer ")
        )
        if presented != token:
            abort(401)
    return Response(
        request_metrics.render(),
        mimetype="text/plain; version=0.0.4; charset=utf-8",
    )
