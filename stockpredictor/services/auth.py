"""User store: in-memory registry persisted through production_core.

Shares the same on-disk state file as the legacy application so accounts,
watchlists and portfolios remain compatible across both entry points.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from production_core import audit_log, load_app_state, save_app_state

from .security import (
    EMAIL_PATTERN,
    MAX_FAILED_LOGIN_ATTEMPTS,
    hash_secret,
    hash_password,
    is_account_locked,
    lock_account,
    remaining_lockout_seconds,
    validate_password_strength,
    verify_password,
    verify_secret,
)

ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLES = (ROLE_USER, ROLE_ADMIN)

OTP_TTL_MINUTES = 10
OTP_MAX_ATTEMPTS = 5


def _generate_otp() -> str:
    """Return a new 6-digit one-time password."""
    return f"{secrets.randbelow(1_000_000):06d}"


class UserStore:
    """Persistent user registry with in-memory cache."""

    def __init__(self) -> None:
        self._state: Dict[str, Any] = load_app_state()
        self.users: Dict[str, Any] = self._state.setdefault("users_db", {})

    def persist(self, write_cache: bool = False) -> None:
        """Persist mutable stores; skips the (large) shared prediction cache.

        Auth/watchlist/portfolio/alert/history flows never touch the cache, so
        the default here is ``write_cache=False`` — a login no longer rewrites
        the entire cache table. Callers that DO mutate the cache (the analyze
        flow) pass ``write_cache=True`` explicitly.
        """
        save_app_state(
            self.users,
            self._state.get("user_watchlists", {}),
            self._state.get("prediction_cache", {}),
            self._state.get("user_portfolios", {}),
            extra_state={
                "password_reset_tokens": self._state.get("password_reset_tokens", {}),
                "pending_registrations": self._state.get("pending_registrations", {}),
                "user_alerts": self._state.get("user_alerts", {}),
                "analysis_history": self._state.get("analysis_history", {}),
            },
            write_cache=write_cache,
        )

    def _store_otp(self, email: str, otp: str) -> None:
        """Persist an OTP for a pending signup.

        Hashed at rest whenever real email delivery is configured. In demo mode
        (SMTP unset) the plaintext code stays readable because the UI must show
        it on the verify page — demo mode must never be enabled in production.
        """
        from .mail import smtp_configured

        stored = hash_secret(otp) if smtp_configured() else otp
        entry = self.pending_registrations.get(email)
        if entry is not None:
            entry["otp"] = stored

    def _otp_matches(self, entry: Dict[str, Any], submitted: str) -> bool:
        return verify_secret(str(entry.get("otp", "")), submitted)

    @property
    def pending_registrations(self) -> Dict[str, Any]:
        """Persisted signups awaiting email OTP verification."""
        pending = self._state.setdefault("pending_registrations", {})
        self._purge_expired_pending(pending)
        return pending

    def _purge_expired_pending(self, pending: Dict[str, Any]) -> None:
        now = datetime.now().isoformat()
        expired = [
            email
            for email, entry in pending.items()
            if (entry.get("expires") or "") <= now
        ]
        for email in expired:
            pending.pop(email, None)
        if expired:
            self.persist()

    def start_otp_registration(self, email: str, password: str) -> Optional[str]:
        """Validate credentials and stash a pending signup with a fresh OTP.

        Returns an error message, or None on success (callers read the code
        from :attr:`pending_registrations` to deliver it).
        """
        email = (email or "").strip().lower()
        if not email or not password:
            return "Please fill in all fields"
        if not EMAIL_PATTERN.match(email):
            return "Please enter a valid email address"
        if email in self.users:
            return "Email already registered. Please login instead."
        valid, password_error = validate_password_strength(password)
        if not valid:
            return password_error

        self.pending_registrations[email] = {
            "otp": "",
            "expires": (
                datetime.now() + timedelta(minutes=OTP_TTL_MINUTES)
            ).isoformat(),
            "password": hash_password(password),
            "attempts": 0,
        }
        self._store_otp(email, _generate_otp())
        self.persist()
        return None

    def pending_otp(self, email: str) -> Optional[str]:
        """Return the current OTP for a pending signup (None when absent)."""
        entry = self.pending_registrations.get((email or "").strip().lower())
        return (entry or {}).get("otp")

    def cancel_otp_registration(self, email: str) -> None:
        """Drop a pending signup (e.g. when email delivery failed)."""
        self.pending_registrations.pop((email or "").strip().lower(), None)
        self.persist()

    def resend_otp(self, email: str) -> Optional[str]:
        """Issue a fresh OTP for an existing pending signup.

        Returns the plaintext code (for delivery); only the hash is stored when
        SMTP is configured.
        """
        email = (email or "").strip().lower()
        entry = self.pending_registrations.get(email)
        if entry is None or email in self.users:
            return None
        otp = _generate_otp()
        entry["expires"] = (
            datetime.now() + timedelta(minutes=OTP_TTL_MINUTES)
        ).isoformat()
        entry["attempts"] = 0
        self._store_otp(email, otp)
        self.persist()
        return otp

    def complete_otp_registration(self, email: str, otp: str) -> Optional[str]:
        """Verify the OTP and create the account; returns an error or None."""
        email = (email or "").strip().lower()
        entry = self.pending_registrations.get(email)
        if entry is None:
            return "No pending registration found for this email. Please register again."
        if (entry.get("expires") or "") < datetime.now().isoformat():
            del self.pending_registrations[email]
            self.persist()
            return "Verification code has expired. Please register again."
        if entry.get("attempts", 0) >= OTP_MAX_ATTEMPTS:
            del self.pending_registrations[email]
            self.persist()
            return "Too many incorrect attempts. Please register again."

        submitted = str(otp or "").strip()
        if not submitted or not self._otp_matches(entry, submitted):
            entry["attempts"] = entry.get("attempts", 0) + 1
            self.persist()
            return "Invalid verification code. Please try again."

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.users[email] = {
            "password": entry["password"],
            "created_at": now,
            "last_login": None,
            "login_count": 0,
            "password_changed": now,
            "failed_attempts": 0,
            "locked_until": None,
            "role": ROLE_USER,
            "email_verified": True,
            "verified_at": now,
        }
        del self.pending_registrations[email]
        self.persist()
        return None

    def exists(self, email: str) -> bool:
        return email in self.users

    def register(self, email: str, password: str) -> Optional[str]:
        """Register a new user; returns an error message or None on success."""
        email = email.strip().lower()
        if not email or not password:
            return "Please fill in all fields"
        if not EMAIL_PATTERN.match(email):
            return "Please enter a valid email address"
        if email in self.users:
            return "Email already registered. Please login instead."
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.users[email] = {
            "password": hash_password(password),
            "created_at": now,
            "last_login": None,
            "login_count": 0,
            "password_changed": now,
            "failed_attempts": 0,
            "locked_until": None,
            "role": ROLE_USER,
        }
        self.persist()
        return None

    def get_user(self, email: str) -> Optional[Dict[str, Any]]:
        return self.users.get((email or "").strip().lower())

    def authenticate(self, email: str, password: str) -> Optional[str]:
        """Authenticate; returns an error message or None on success.

        Enforces brute-force lockout: after ``MAX_FAILED_LOGIN_ATTEMPTS`` failed
        attempts the account is locked until the expiry set by ``lock_account``.
        """
        email = email.strip().lower()
        if not email or not password:
            return "Please enter both email and password"
        if not EMAIL_PATTERN.match(email):
            return "Please enter a valid email address"
        user = self.get_user(email)
        if user is None:
            return "Account not found. Please register first."

        if is_account_locked(user):
            remaining = remaining_lockout_seconds(user)
            return f"Account temporarily locked. Try again in {remaining} seconds."

        if not verify_password(user.get("password", ""), password):
            user["failed_attempts"] = user.get("failed_attempts", 0) + 1
            if user["failed_attempts"] >= MAX_FAILED_LOGIN_ATTEMPTS:
                lock_account(user)
                self.persist()
                audit_log("user.account_locked", email, {
                    "failed_attempts": user["failed_attempts"],
                })
                return "Account temporarily locked due to too many failed attempts."
            self.persist()
            return f"Invalid password. {MAX_FAILED_LOGIN_ATTEMPTS - user['failed_attempts']} attempts remaining."

        user["failed_attempts"] = 0
        user["locked_until"] = None
        user["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user["login_count"] = user.get("login_count", 0) + 1
        self.persist()
        return None

    def is_locked(self, email: str) -> bool:
        user = self.get_user(email)
        return bool(user and is_account_locked(user))

    def unlock(self, email: str) -> bool:
        """Clear lockout state; returns True when a user was updated."""
        user = self.get_user(email)
        if user is None:
            return False
        user["failed_attempts"] = 0
        user["locked_until"] = None
        self.persist()
        return True

    def role(self, email: str) -> str:
        user = self.get_user(email)
        return (user or {}).get("role", ROLE_USER)

    def is_admin(self, email: str) -> bool:
        return self.role(email) == ROLE_ADMIN

    def set_role(self, email: str, role: str) -> bool:
        """Promote/demote a user's role; returns True on success."""
        if role not in ROLES:
            return False
        user = self.get_user(email)
        if user is None:
            return False
        user["role"] = role
        self.persist()
        return True

    def set_password(self, email: str, password: str) -> bool:
        user = self.get_user(email)
        if user is None:
            return False
        user["password"] = hash_password(password)
        user["password_changed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user["failed_attempts"] = 0
        user["locked_until"] = None
        self.persist()
        return True

    def reset_tokens(self) -> Dict[str, Any]:
        """Persisted password-reset token store (survives restarts)."""
        tokens = self._state.setdefault("password_reset_tokens", {})
        self._purge_expired_tokens(tokens)
        return tokens

    def _purge_expired_tokens(self, tokens: Dict[str, Any]) -> None:
        now = datetime.now()
        expired = [
            token for token, entry in tokens.items()
            if (entry.get("expires") or "") <= now.isoformat()
        ]
        for token in expired:
            tokens.pop(token, None)
        if expired:
            self.persist()

    def watchlists(self) -> Dict[str, List[str]]:
        return self._state.setdefault("user_watchlists", {})

    def portfolios(self) -> Dict[str, Any]:
        return self._state.setdefault("user_portfolios", {})

    def alerts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Per-user price-alert lists (persisted)."""
        return self._state.setdefault("user_alerts", {})

    def analysis_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Per-user list of past analyses (persisted, newest first)."""
        return self._state.setdefault("analysis_history", {})

    def add_analysis(self, email: str, entry: Dict[str, Any], limit: int = 25) -> None:
        """Record a completed analysis for a user, capped at ``limit``."""
        email = (email or "").strip().lower()
        history = self.analysis_history()
        items = history.setdefault(email, [])
        items.insert(0, entry)
        del items[limit:]
        self.persist()

    def delete_account(self, email: str) -> bool:
        """Permanently remove a user and all associated per-user data.

        ``prediction_cache`` is shared across users and is left untouched.
        Returns True when an account was removed.
        """
        email = (email or "").strip().lower()
        if self.users.pop(email, None) is None:
            return False

        self._state.get("user_watchlists", {}).pop(email, None)
        self._state.get("user_portfolios", {}).pop(email, None)
        self._state.get("pending_registrations", {}).pop(email, None)
        self._state.get("user_alerts", {}).pop(email, None)
        self._state.get("analysis_history", {}).pop(email, None)

        tokens = self._state.get("password_reset_tokens", {})
        for token in [
            token for token, entry in tokens.items() if entry.get("email") == email
        ]:
            tokens.pop(token, None)

        self.persist()
        return True


user_store = UserStore()
