"""SQLite persistence layer for Stock Predictor Pro state.

Routes ``production_core.load_app_state`` / ``save_app_state`` to a relational
SQLite database (``runtime_state/app_state.db``) instead of the legacy JSON
blob, while preserving the dict-based contract used by ``UserStore`` and the
monolith ``app.py``. The database path derives from ``STATE_DIR`` so tests can
redirect state exactly as before.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import create_engine, delete, event, select
from sqlalchemy.orm import sessionmaker

from production_core import STATE_DIR, STATE_VERSION, state_write_lock, utc_now

from .models import (
    Alert,
    AnalysisEntry,
    Base,
    Meta,
    PendingRegistration,
    PasswordResetToken,
    Portfolio,
    PredictionCacheEntry,
    User,
    WatchlistEntry,
)

_engines: Dict[str, tuple] = {}
_lock = threading.Lock()


def db_path() -> Path:
    """Absolute path of the SQLite state database (inside STATE_DIR)."""
    return Path(STATE_DIR) / "app_state.db"


def _ensure_engine():
    """Return (engine, sessionmaker) for the current STATE_DIR, cached by path.

    State is per-test/directory, so engines are keyed by the resolved database
    path rather than cached globally; a fresh ``STATE_DIR`` simply binds a new
    engine.
    """
    path = db_path()
    key = str(path)
    with _lock:
        entry = _engines.get(key)
        if entry is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(
                f"sqlite:///{path.as_posix()}",
                connect_args={"check_same_thread": False},
                pool_recycle=3600,
            )

            @event.listens_for(engine, "connect")
            def _tune_sqlite(dbapi_connection, _record):
                # WAL: readers never block the writer; busy_timeout: concurrent
                # writers queue instead of failing with "database is locked".
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA busy_timeout=5000")
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                finally:
                    cursor.close()

            entry = (
                engine,
                sessionmaker(bind=engine, autoflush=False, expire_on_commit=False),
            )
            _engines[key] = entry
        return entry


def init_db() -> None:
    """Create tables when they do not exist yet (idempotent).

    When the schema is created directly (bypassing ``alembic upgrade head``),
    stamp the database at the current migration head so future upgrades and
    ``alembic upgrade`` stay coherent. Alembic must be present for the stamp;
    if it is not, the schema still works, it just lacks a version marker.
    """
    engine, _ = _ensure_engine()
    Base.metadata.create_all(engine)
    _stamp_alembic_head_if_needed(engine)


def _stamp_alembic_head_if_needed(engine) -> None:
    from sqlalchemy import inspect

    if "alembic_version" in inspect(engine).get_table_names():
        return
    try:
        from alembic.config import Config
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory

        from production_core import BASE_DIR

        cfg = Config(str(BASE_DIR / "alembic.ini"))
        cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
        script = ScriptDirectory.from_config(cfg)
        head = script.get_current_head()
        if not head:
            return
        with engine.begin() as connection:
            MigrationContext.configure(connection).stamp(script, head)
    except Exception:
        pass


def _loads(raw: str, fallback: Any):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def hydrate_state() -> Dict[str, Any]:
    """Rebuild the full legacy state dict from the relational tables."""
    _engine, session_factory = _ensure_engine()
    init_db()

    state: Dict[str, Any] = {
        "version": STATE_VERSION,
        "users_db": {},
        "user_watchlists": {},
        "prediction_cache": {},
        "user_portfolios": {},
        "password_reset_tokens": {},
        "pending_registrations": {},
        "user_alerts": {},
        "analysis_history": {},
    }

    with session_factory() as session:
        version = session.get(Meta, "version")
        if version is not None:
            try:
                state["version"] = int(version.value)
            except (TypeError, ValueError):
                state["version"] = STATE_VERSION

        for user in session.scalars(select(User).order_by(User.email)):
            state["users_db"][user.email] = {
                "password": user.password,
                "created_at": user.created_at,
                "last_login": user.last_login,
                "login_count": user.login_count,
                "password_changed": user.password_changed,
                "failed_attempts": user.failed_attempts,
                "locked_until": user.locked_until,
                "role": user.role,
                "email_verified": user.email_verified,
                "verified_at": user.verified_at,
            }

        for entry in session.scalars(
            select(WatchlistEntry).order_by(WatchlistEntry.email, WatchlistEntry.position, WatchlistEntry.row_id)
        ):
            state["user_watchlists"].setdefault(entry.email, []).append(entry.symbol)

        for row in session.scalars(select(Portfolio).order_by(Portfolio.email)):
            state["user_portfolios"][row.email] = {
                "cash": row.cash,
                "positions": _loads(row.positions, {}),
                "history": _loads(row.history, []),
                "created_at": row.created_at,
            }

        for row in session.scalars(
            select(Alert).order_by(Alert.email, Alert.created, Alert.row_id)
        ):
            state["user_alerts"].setdefault(row.email, []).append({
                "id": row.alert_id,
                "symbol": row.symbol,
                "condition": row.condition,
                "threshold": row.threshold,
                "created": row.created,
                "last_triggered": row.last_triggered,
            })

        for row in session.scalars(select(PendingRegistration).order_by(PendingRegistration.email)):
            state["pending_registrations"][row.email] = {
                "otp": row.otp,
                "expires": row.expires,
                "password": row.password,
                "attempts": row.attempts,
            }

        for row in session.scalars(select(PasswordResetToken).order_by(PasswordResetToken.token)):
            state["password_reset_tokens"][row.token] = {
                "email": row.email,
                "expires": row.expires,
            }

        for row in session.scalars(
            select(AnalysisEntry).order_by(AnalysisEntry.email, AnalysisEntry.position, AnalysisEntry.row_id)
        ):
            state["analysis_history"].setdefault(row.email, []).append(_loads(row.payload, {}))

        for row in session.scalars(select(PredictionCacheEntry).order_by(PredictionCacheEntry.key)):
            state["prediction_cache"][row.key] = _loads(row.value, None)

    return state


def persist_state(state: Dict[str, Any], write_cache: bool = True) -> None:
    """Write the full state dict to the relational tables (replace semantics).

    Mirrors the legacy whole-file rewrite of ``app_state.json`` inside a single
    transaction so the persisted shape always matches the in-memory dict.

    ``write_cache=False`` skips the shared ``prediction_cache`` table — used by
    hot paths that cannot have touched it (auth/watchlist/portfolio flows), so
    a login no longer rewrites the entire (potentially large) cache. Callers
    that DO mutate the cache keep the default ``True``.
    """
    _engine, session_factory = _ensure_engine()
    init_db()

    users_db: Dict[str, Any] = state.get("users_db") or {}
    watchlists: Dict[str, Any] = state.get("user_watchlists") or {}
    portfolios: Dict[str, Any] = state.get("user_portfolios") or {}
    alerts_map: Dict[str, Any] = state.get("user_alerts") or {}
    pending: Dict[str, Any] = state.get("pending_registrations") or {}
    tokens: Dict[str, Any] = state.get("password_reset_tokens") or {}
    analysis: Dict[str, Any] = state.get("analysis_history") or {}
    cache: Dict[str, Any] = state.get("prediction_cache") or {}

    try:
        version = int(state.get("version", STATE_VERSION))
    except (TypeError, ValueError):
        version = STATE_VERSION

    with state_write_lock():
        with session_factory() as session:
            with session.begin():
                tables = [
                    Meta,
                    User,
                    WatchlistEntry,
                    Portfolio,
                    Alert,
                    PendingRegistration,
                    PasswordResetToken,
                    AnalysisEntry,
                ]
                if write_cache:
                    tables.append(PredictionCacheEntry)
                for table in tables:
                    session.execute(delete(table))

                session.add(Meta(key="version", value=str(version)))

                for email, user in users_db.items():
                    session.add(User(
                        email=email,
                        password=str(user.get("password", "")),
                        created_at=str(user.get("created_at") or utc_now()),
                        last_login=user.get("last_login"),
                        login_count=int(user.get("login_count", 0)),
                        password_changed=user.get("password_changed"),
                        failed_attempts=int(user.get("failed_attempts", 0)),
                        locked_until=user.get("locked_until"),
                        role=str(user.get("role", "user")),
                        email_verified=bool(user.get("email_verified", True)),
                        verified_at=user.get("verified_at"),
                    ))

                for email, symbols in watchlists.items():
                    for position, symbol in enumerate(symbols or []):
                        session.add(WatchlistEntry(email=email, symbol=str(symbol), position=position))

                for email, portfolio in portfolios.items():
                    session.add(Portfolio(
                        email=email,
                        cash=float(portfolio.get("cash", 10000.0)),
                        positions=json.dumps(portfolio.get("positions", {}) or {}),
                        history=json.dumps(portfolio.get("history", []) or []),
                        created_at=portfolio.get("created_at"),
                    ))

                for email, items in alerts_map.items():
                    for position, alert in enumerate(items or []):
                        session.add(Alert(
                            alert_id=str(alert.get("id")),
                            email=email,
                            symbol=str(alert.get("symbol", "")),
                            condition=str(alert.get("condition", "above")),
                            threshold=float(alert.get("threshold", 0)),
                            created=str(alert.get("created") or utc_now()),
                            last_triggered=alert.get("last_triggered"),
                        ))

                for email, entry in pending.items():
                    session.add(PendingRegistration(
                        email=email,
                        otp=str(entry.get("otp", "")),
                        expires=str(entry.get("expires", "")),
                        password=str(entry.get("password", "")),
                        attempts=int(entry.get("attempts", 0)),
                    ))

                for token, entry in tokens.items():
                    session.add(PasswordResetToken(
                        token=token,
                        email=str(entry.get("email", "")),
                        expires=str(entry.get("expires", "")),
                    ))

                for email, entries in analysis.items():
                    for position, entry in enumerate(entries or []):
                        session.add(AnalysisEntry(
                            email=email,
                            position=position,
                            payload=json.dumps(entry),
                        ))

                if write_cache:
                    for key, value in cache.items():
                        session.add(PredictionCacheEntry(
                            key=str(key),
                            value=json.dumps(value),
                            updated_at=utc_now(),
                        ))
