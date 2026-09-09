"""Socket.IO realtime quote streaming (Feature: Streaming quotes).

Authenticated clients join a per-user room (``user:<email>``) on connect. A
single background loop, started lazily on the first connection, pushes that
user's watchlist quotes into the room every ``REALTIME_PUSH_SECONDS``. The REST
endpoint ``/api/watchlist/quotes`` mirrors the same payload so the UI can fall
back to HTTP polling when websockets are unavailable.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from flask import session
from flask_socketio import join_room, leave_room

from production_core import utc_now

from .auth import user_store
from . import stocks

logger = logging.getLogger("stockpredictor.services.realtime")

_ROOM_PREFIX = "user:"
_EVENT_QUOTES = "watchlist_quotes"
_registered = False
_started = False


def _user_email() -> str:
    user_id = session.get("_user_id")
    return user_id if isinstance(user_id, str) and user_id else ""


def collect_quotes(app, email: str, max_symbols: int) -> List[Dict[str, Any]]:
    """Live quote payload for a user's watchlist (TTL-cached per symbol)."""
    symbols = user_store.watchlists().get(email, [])[:max_symbols]
    quotes: List[Dict[str, Any]] = []
    for symbol in symbols:
        try:
            quote = stocks.get_quote(symbol) or {}
        except Exception:
            quote = {}
        if quote.get("price") is not None:
            quotes.append(quote)
    return quotes


def stream_once(sio, app) -> None:
    """Push fresh quotes to every connected per-user room (best effort)."""
    manager = getattr(getattr(sio, "server", None), "manager", None)
    if manager is None:
        return
    namespace_rooms = getattr(manager, "rooms", {}).get("/", {}) or {}
    max_symbols = int(app.config.get("REALTIME_MAX_SYMBOLS", 20))
    for room in namespace_rooms:
        if not isinstance(room, str) or not room.startswith(_ROOM_PREFIX):
            continue
        email = room[len(_ROOM_PREFIX):]
        quotes = collect_quotes(app, email, max_symbols)
        if quotes:
            sio.emit(
                _EVENT_QUOTES,
                {"quotes": quotes, "updated_at": utc_now()},
                room=room,
                namespace="/",
            )


def _stream_loop(sio, app) -> None:
    interval = max(1.0, float(app.config.get("REALTIME_PUSH_SECONDS", 10)))
    while True:
        time.sleep(interval)
        try:
            with app.app_context():
                stream_once(sio, app)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            logger.warning("Realtime stream iteration failed: %s", exc)


def register_realtime(sio, app) -> None:
    """Register Socket.IO handlers and the background quote stream.

    Must run before ``sio.init_app(app)`` so the handlers are attached to the
    underlying server. Registration is idempotent (guarded per process).
    """
    global _registered, _started
    if _registered:
        return
    _registered = True

    @sio.on("connect")
    def _on_connect(auth=None):
        if not app.config.get("ENABLE_REALTIME_PUSH"):
            return False
        email = _user_email()
        if not email:
            return False
        join_room(_ROOM_PREFIX + email)
        if not _started:
            _started = True
            sio.start_background_task(_stream_loop, sio, app)
        return True

    @sio.on("disconnect")
    def _on_disconnect():
        email = _user_email()
        if email:
            leave_room(_ROOM_PREFIX + email)
