"""Relational (SQLite) models backing the Stock Predictor Pro state.

Replaces the legacy ``runtime_state/app_state.json`` blob with a typed
schema. Table columns mirror the persisted JSON shape so ``production_core``
can hydrate the in-memory dict API consumed by :mod:`stockpredictor.services`
and the legacy ``app.py`` without behaviour changes.
"""
from __future__ import annotations

from sqlalchemy import Boolean, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Meta(Base):
    """Key/value metadata (schema version, last write)."""

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class User(Base):
    """Registered accounts (``users_db`` in the legacy state)."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), primary_key=True)
    password: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    last_login: Mapped[str | None] = mapped_column(String(64), nullable=True)
    login_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    password_changed: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failed_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[str | None] = mapped_column(String(64), nullable=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    email_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    verified_at: Mapped[str | None] = mapped_column(String(64), nullable=True)


class WatchlistEntry(Base):
    """One symbol in a user's watchlist (ordered)."""

    __tablename__ = "watchlist_entries"

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (UniqueConstraint("email", "symbol", name="uq_watchlist_email_symbol"),)


class Portfolio(Base):
    """A user's paper-trading portfolio (cash + positions + history)."""

    __tablename__ = "portfolios"

    email: Mapped[str] = mapped_column(String(320), primary_key=True)
    cash: Mapped[float] = mapped_column(Float, nullable=False, default=10000.0)
    positions: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    history: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[str | None] = mapped_column(String(64), nullable=True)


class Alert(Base):
    """A per-user price alert (``above``/``below`` threshold)."""

    __tablename__ = "alerts"

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[str] = mapped_column(String(16), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    condition: Mapped[str] = mapped_column(String(8), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created: Mapped[str] = mapped_column(String(64), nullable=False)
    last_triggered: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (UniqueConstraint("email", "alert_id", name="uq_alert_email_id"),)


class PendingRegistration(Base):
    """Signups awaiting email OTP verification."""

    __tablename__ = "pending_registrations"

    email: Mapped[str] = mapped_column(String(320), primary_key=True)
    otp: Mapped[str] = mapped_column(String(8), nullable=False)
    expires: Mapped[str] = mapped_column(String(64), nullable=False)
    password: Mapped[str] = mapped_column(Text, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class PasswordResetToken(Base):
    """Issued password-reset tokens."""

    __tablename__ = "password_reset_tokens"

    token: Mapped[str] = mapped_column(String(128), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    expires: Mapped[str] = mapped_column(String(64), nullable=False)


class AnalysisEntry(Base):
    """One recorded analysis in a user's history (ordered)."""

    __tablename__ = "analysis_history"

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class PredictionCacheEntry(Base):
    """Shared prediction cache (key -> JSON value)."""

    __tablename__ = "prediction_cache"

    key: Mapped[str] = mapped_column(String(1024), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False, default="null")
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)
