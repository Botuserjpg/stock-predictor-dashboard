"""Application configuration for the Stock Predictor Pro web application."""
from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Runtime settings loaded from environment variables with safe defaults."""

    def __init__(self) -> None:
        self.FLASK_ENV = self._env_str("FLASK_ENV", "production")
        self.SECRET_KEY = self._env_str("SECRET_KEY")
        self.DEBUG = self._env_bool("FLASK_DEBUG", False)
        self.LOG_LEVEL = self._env_str("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = self._env_str("LOG_FORMAT", "text")
        # Bind to loopback unless explicitly overridden; the Docker image sets
        # HOST=0.0.0.0 so containers keep working unchanged.
        default_host = "0.0.0.0" if self.FLASK_ENV == "production" else "127.0.0.1"
        self.HOST = self._env_str("HOST", default_host)
        self.PORT = self._env_int("PORT", 5000)
        # Public deployments can expose a read-only recruiter-friendly preview
        # without weakening authentication for user accounts or write actions.
        self.PUBLIC_DEMO = self._env_bool("PUBLIC_DEMO", False)

        # Sessions / security
        self.REMEMBER_COOKIE_DURATION = timedelta(
            days=self._env_int("REMEMBER_DAYS", 30)
        )
        self.SESSION_COOKIE_HTTPONLY = True
        self.SESSION_COOKIE_SAMESITE = self._env_str("SESSION_COOKIE_SAMESITE", "Lax")
        # Secure cookies by default in production so sessions never travel over
        # plain HTTP; local/dev environments stay usable without TLS.
        secure_default = self.FLASK_ENV == "production"
        self.SESSION_COOKIE_SECURE = self._env_bool(
            "SESSION_COOKIE_SECURE", secure_default
        )
        self.MAX_CONTENT_LENGTH = self._env_int(
            "MAX_CONTENT_LENGTH", 2 * 1024 * 1024
        )
        self.PERMANENT_SESSION_LIFETIME = timedelta(
            hours=self._env_int("SESSION_HOURS", 12)
        )

        # Content-Security-Policy: report-only by default (pages still carry a
        # few legacy inline scripts); flip CSP_ENFORCE=true once externalized.
        self.CSP_POLICY = self._env_str(
            "CSP_POLICY",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'self'; "
            "base-uri 'self'; "
            "form-action 'self'",
        )
        self.CSP_ENFORCE = self._env_bool("CSP_ENFORCE", False)

        # Observability
        self.METRICS_ENABLED = self._env_bool("METRICS_ENABLED", False)
        self.METRICS_TOKEN = self._env_str("METRICS_TOKEN")

        # Predictions
        self.DEFAULT_FORECAST_DAYS = self._env_int("DEFAULT_FORECAST_DAYS", 30)
        self.MAX_FORECAST_DAYS = self._env_int("MAX_FORECAST_DAYS", 365)
        self.MIN_FORECAST_DAYS = self._env_int("MIN_FORECAST_DAYS", 1)
        self.DEFAULT_INITIAL_CASH = float(self._env_str("DEFAULT_INITIAL_CASH", "10000.0"))
        self.RISK_FREE_RATE = float(self._env_str("RISK_FREE_RATE", "0.02"))

        # Social/sentiment APIs
        self.TWITTER_BEARER_TOKEN = self._env_str("TWITTER_BEARER_TOKEN")

        # Advanced features: push alerts, LLM analyst report, drift monitoring
        self.TELEGRAM_BOT_TOKEN = self._env_str("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID = self._env_str("TELEGRAM_CHAT_ID")
        self.DISCORD_WEBHOOK_URL = self._env_str("DISCORD_WEBHOOK_URL")
        self.OPENAI_API_KEY = self._env_str("OPENAI_API_KEY")
        self.ANTHROPIC_API_KEY = self._env_str("ANTHROPIC_API_KEY")
        self.DRIFT_THRESHOLD = float(self._env_str("DRIFT_THRESHOLD", "0.25"))
        self.BACKTEST_DEFAULT_CASH = float(self._env_str("BACKTEST_DEFAULT_CASH", "10000.0"))
        self.BACKTEST_COMMISSION = float(self._env_str("BACKTEST_COMMISSION", "0.001"))
        self.BACKTEST_SLIPPAGE = float(self._env_str("BACKTEST_SLIPPAGE", "0.0005"))
        self.MONTE_CARLO_PATHS = self._env_int("MONTE_CARLO_PATHS", 1000)

        # Background price-alert worker + realtime quote streaming
        self.ENABLE_ALERT_SCHEDULER = self._env_bool("ENABLE_ALERT_SCHEDULER", False)
        self.ALERT_POLL_SECONDS = self._env_int("ALERT_POLL_SECONDS", 300)
        self.ALERT_COOLDOWN_SECONDS = self._env_int("ALERT_COOLDOWN_SECONDS", 3600)
        self.ALERT_SCHEDULER_BACKEND = self._env_str("ALERT_SCHEDULER_BACKEND", "apscheduler")
        self.ENABLE_REALTIME_PUSH = self._env_bool("ENABLE_REALTIME_PUSH", True)
        self.REALTIME_PUSH_SECONDS = self._env_int("REALTIME_PUSH_SECONDS", 10)
        self.REALTIME_MAX_SYMBOLS = self._env_int("REALTIME_MAX_SYMBOLS", 20)

    @staticmethod
    def _env_str(name: str, default: str = "") -> str:
        value = os.getenv(name)
        return default if value is None else value

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def to_flask_config(self) -> Dict[str, Any]:
        return {
            "SECRET_KEY": self.SECRET_KEY,
            "DEBUG": self.DEBUG,
            "LOG_LEVEL": self.LOG_LEVEL,
            "LOG_FORMAT": self.LOG_FORMAT,
            "HOST": self.HOST,
            "PORT": self.PORT,
            "PUBLIC_DEMO": self.PUBLIC_DEMO,
            "FLASK_ENV": self.FLASK_ENV,
            "REMEMBER_COOKIE_DURATION": self.REMEMBER_COOKIE_DURATION,
            "SESSION_COOKIE_HTTPONLY": self.SESSION_COOKIE_HTTPONLY,
            "SESSION_COOKIE_SAMESITE": self.SESSION_COOKIE_SAMESITE,
            "SESSION_COOKIE_SECURE": self.SESSION_COOKIE_SECURE,
            "MAX_CONTENT_LENGTH": self.MAX_CONTENT_LENGTH,
            "PERMANENT_SESSION_LIFETIME": self.PERMANENT_SESSION_LIFETIME,
            "CSP_POLICY": self.CSP_POLICY,
            "CSP_ENFORCE": self.CSP_ENFORCE,
            "METRICS_ENABLED": self.METRICS_ENABLED,
            "METRICS_TOKEN": self.METRICS_TOKEN,
            "DEFAULT_FORECAST_DAYS": self.DEFAULT_FORECAST_DAYS,
            "MAX_FORECAST_DAYS": self.MAX_FORECAST_DAYS,
            "MIN_FORECAST_DAYS": self.MIN_FORECAST_DAYS,
            "DEFAULT_INITIAL_CASH": self.DEFAULT_INITIAL_CASH,
            "RISK_FREE_RATE": self.RISK_FREE_RATE,
            "TELEGRAM_BOT_TOKEN": self.TELEGRAM_BOT_TOKEN,
            "TELEGRAM_CHAT_ID": self.TELEGRAM_CHAT_ID,
            "DISCORD_WEBHOOK_URL": self.DISCORD_WEBHOOK_URL,
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "ANTHROPIC_API_KEY": self.ANTHROPIC_API_KEY,
            "DRIFT_THRESHOLD": self.DRIFT_THRESHOLD,
            "BACKTEST_DEFAULT_CASH": self.BACKTEST_DEFAULT_CASH,
            "BACKTEST_COMMISSION": self.BACKTEST_COMMISSION,
            "BACKTEST_SLIPPAGE": self.BACKTEST_SLIPPAGE,
            "MONTE_CARLO_PATHS": self.MONTE_CARLO_PATHS,
            "ENABLE_ALERT_SCHEDULER": self.ENABLE_ALERT_SCHEDULER,
            "ALERT_POLL_SECONDS": self.ALERT_POLL_SECONDS,
            "ALERT_COOLDOWN_SECONDS": self.ALERT_COOLDOWN_SECONDS,
            "ALERT_SCHEDULER_BACKEND": self.ALERT_SCHEDULER_BACKEND,
            "ENABLE_REALTIME_PUSH": self.ENABLE_REALTIME_PUSH,
            "REALTIME_PUSH_SECONDS": self.REALTIME_PUSH_SECONDS,
            "REALTIME_MAX_SYMBOLS": self.REALTIME_MAX_SYMBOLS,
        }
