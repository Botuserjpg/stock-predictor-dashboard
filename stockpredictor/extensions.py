"""Flask extension initialization (auth, session, socketio) for Stock Predictor Pro."""
from __future__ import annotations

import os

from flask import Flask
from flask_login import LoginManager, UserMixin
from flask_socketio import SocketIO

from production_core import get_secret_key

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "error"

# wsgi.py sets WSGI_ASYNC_MODE when eventlet's monkey_patch() is unusable
# (Python 3.13+); fall back to the threading async mode so dev servers start.
_socketio_async_mode = os.environ.get("WSGI_ASYNC_MODE") or None
socketio = SocketIO(async_mode=_socketio_async_mode) if _socketio_async_mode else SocketIO()


class User(UserMixin):
    """Minimal Flask-Login user backed by the persisted users store."""

    def __init__(self, email: str) -> None:
        self.id = email

    @property
    def email(self) -> str:
        return self.id

    @property
    def role(self) -> str:
        from stockpredictor.services.auth import user_store

        return user_store.role(self.id)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def init_extensions(app: Flask) -> None:
    """Register extensions with the application instance."""
    from production_core import install_request_guards
    from stockpredictor.services.auth import user_store

    app.secret_key = app.config.get("SECRET_KEY") or get_secret_key()

    @login_manager.user_loader
    def load_user(user_id: str):
        if user_id in user_store.users:
            return User(user_id)
        return None

    login_manager.init_app(app)

    from stockpredictor.services.realtime import register_realtime

    register_realtime(socketio, app)
    socketio.init_app(app)

    logger = app.logger
    install_request_guards(app, logger)

    if app.config.get("ENABLE_ALERT_SCHEDULER"):
        from stockpredictor.services.scheduler import start as start_alert_scheduler

        start_alert_scheduler(app)

    @app.template_global("csrf_token")
    def csrf_token():
        from production_core import generate_csrf_token

        return generate_csrf_token()
