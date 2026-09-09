"""Route decorators for role-based access control."""
from __future__ import annotations

from functools import wraps
from typing import Callable

from flask import abort
from flask_login import current_user


def admin_required(fn: Callable):
    """Require an authenticated user with the ``admin`` role."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            abort(401)
        if not current_user.is_admin:
            abort(403)
        return fn(*args, **kwargs)

    return wrapper
