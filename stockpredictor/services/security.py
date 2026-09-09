"""Security helpers: password hashing, strength validation, lockout."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

_PBKDF2_ITERATIONS = 100_000
MAX_FAILED_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 900
_COMMON_PATTERNS = (
    "123456", "password", "qwerty", "abc123", "letmein",
    "admin", "welcome", "monkey", "password1",
)


def hash_password(password: str) -> str:
    """Hash a password with a random salt (scheme: salt$hex)."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("ascii"), _PBKDF2_ITERATIONS
    )
    return f"{salt}${digest.hex()}"


def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a stored password hash against a plaintext candidate."""
    try:
        salt, stored_hash = stored_password.split("$", 1)
    except (AttributeError, ValueError):
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", provided_password.encode("utf-8"), salt.encode("ascii"), _PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(digest.hex(), stored_hash)


# ---------------------------------------------------------------------------
# Lightweight at-rest hashing for short-lived secrets (reset tokens, OTPs).
# Not a password hash — these secrets are high-entropy and short-lived, so a
# fast keyed digest removes DB-read exposure without adding meaningful cost.
# Legacy plaintext values still verify (transparent migration; old rows die
# naturally with their TTL).
# ---------------------------------------------------------------------------
_SECRET_PREFIX = "sha256$"


def hash_secret(secret: str) -> str:
    """Digest a short-lived secret for storage (``sha256$<hex>``)."""
    return _SECRET_PREFIX + hashlib.sha256(str(secret).encode("utf-8")).hexdigest()


def verify_secret(stored: str, candidate: str) -> bool:
    """Constant-time check of ``candidate`` against a ``hash_secret`` value."""
    if not stored or candidate is None:
        return False
    if str(stored).startswith(_SECRET_PREFIX):
        expected = str(stored)[len(_SECRET_PREFIX):]
        digest = hashlib.sha256(str(candidate).encode("utf-8")).hexdigest()
        return hmac.compare_digest(expected, digest)
    # Legacy plaintext row (written before hardening or by demo mode).
    return hmac.compare_digest(str(stored), str(candidate))


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Validate password strength; returns (is_valid, error_message)."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if len(password) > 128:
        return False, "Password must be less than 128 characters"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    lowered = password.lower()
    if any(p in lowered for p in _COMMON_PATTERNS):
        return False, "Password contains common weak patterns"
    if re.search(r"(.)\1{3,}", password):
        return False, "Password contains too many repeated characters"
    for i in range(len(password) - 2):
        if (
            ord(password[i]) + 1 == ord(password[i + 1])
            and ord(password[i + 1]) + 1 == ord(password[i + 2])
        ):
            return False, "Password contains sequential characters"
    return True, "Password is strong"


EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

LOCKOUT_ATTRS = {"failed_attempts", "locked_until"}


def is_account_locked(user: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """Return True when the account is currently locked out."""
    locked_until = user.get("locked_until")
    if not locked_until:
        return False
    try:
        if isinstance(locked_until, datetime):
            until = locked_until
        else:
            until = datetime.fromisoformat(locked_until)
    except (TypeError, ValueError):
        return False
    now = now or datetime.now()
    return now < until


def lock_account(user: Dict[str, Any], seconds: int = LOCKOUT_SECONDS) -> str:
    """Lock an account for ``seconds``; returns the ISO lock expiry timestamp."""
    until = datetime.now() + timedelta(seconds=seconds)
    user["locked_until"] = until.isoformat()
    return user["locked_until"]


def remaining_lockout_seconds(user: Dict[str, Any], now: Optional[datetime] = None) -> int:
    """Return how many seconds remain until the account unlocks (0 if unlocked)."""
    locked_until = user.get("locked_until")
    if not locked_until:
        return 0
    try:
        if isinstance(locked_until, datetime):
            until = locked_until
        else:
            until = datetime.fromisoformat(locked_until)
    except (TypeError, ValueError):
        return 0
    remaining = (until - (now or datetime.now())).total_seconds()
    return max(0, int(remaining))
