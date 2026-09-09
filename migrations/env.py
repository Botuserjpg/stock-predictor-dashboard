"""Alembic migration environment for Stock Predictor Pro.

The target database resolves to the application's SQLite state file
(``runtime_state/app_state.db``) unless ``STOCKPREDICTOR_DB`` overrides the URL
(e.g. ``alembic revision --autogenerate`` against a throwaway database).
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from stockpredictor.database import db_path
from stockpredictor.models import Base
import stockpredictor.models  # noqa: F401  (register every mapped table)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _db_url() -> str:
    override = os.getenv("STOCKPREDICTOR_DB")
    if override:
        return override
    return f"sqlite:///{db_path().as_posix()}"


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _db_url()
    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
