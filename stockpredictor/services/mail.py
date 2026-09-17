"""Email delivery helpers for OTP verification and alerts.

Sends real mail via either provider:

* **Resend** (HTTP API) — used when ``RESEND_API_KEY`` is set. Many free
  hosting tiers block outbound SMTP ports but allow HTTPS, so Resend works
  from sandboxed environments.
* **SMTP** (stdlib ``smtplib``) — used when ``SMTP_HOST`` + ``SMTP_USER`` are
  set and no Resend key is configured (see ``.env.example`` for the TLS/SSL/
  port flags).

When neither is configured the app runs in demo mode: the OTP/reset link is
logged instead of emailed so the flows stay exercisable locally.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

try:  # resend is an optional delivery provider; its absence must not break the app
    import resend

    RESEND_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure degrades to the SMTP/demo path
    resend = None  # type: ignore[assignment]
    RESEND_AVAILABLE = False

logger = logging.getLogger("stock_predictor")

APP_NAME = "Stock Predictor Pro"

_DEFAULT_FROM = "onboarding@resend.dev"


def smtp_configured() -> bool:
    """Return True when real email delivery is configured.

    Name kept as ``smtp_configured`` so existing call sites don't need to
    change; True when a Resend key *or* legacy SMTP credentials are present.
    """
    if os.getenv("RESEND_API_KEY"):
        return True
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_USER"))


def _sender_address() -> str:
    """Build a RFC-5322 sender (``APP_NAME <addr>``) from configuration."""
    sender = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER") or _DEFAULT_FROM
    try:
        return formataddr((APP_NAME, sender))
    except Exception:  # noqa: BLE001 - keep a plain address in any edge case
        return sender


def _resend_configured() -> bool:
    return RESEND_AVAILABLE and bool(os.getenv("RESEND_API_KEY"))


def send_otp_email(email: str, otp: str) -> bool:
    """Email a one-time verification code.

    Returns True when the code was delivered (real send) or surfaced in demo
    mode (logged). Returns False when real delivery was requested but failed;
    the caller should surface an error to the user.
    """
    recipient = (email or "").strip().lower()
    if not smtp_configured():
        logger.warning("OTP for %s (demo mode, mail not configured): %s", recipient, otp)
        return True

    subject = f"{APP_NAME} - Your verification code"
    body = (
        f"Hello,\n\n"
        f"Your {APP_NAME} verification code is: {otp}\n\n"
        f"Enter this code on the signup page to confirm your email address. "
        f"It expires in a few minutes.\n\n"
        f"If you did not request this code, you can safely ignore this email.\n"
    )

    try:
        _send_mail(recipient, subject, body)
        logger.info("OTP emailed to %s", recipient)
        return True
    except Exception as exc:  # noqa: BLE001 - surface as send failure
        logger.error("Failed to email OTP to %s: %s", recipient, exc)
        return False


def send_password_reset_email(email: str, token: str, base_url: str = "") -> bool:
    """Email a password-reset link.

    ``token`` is the raw (unhashed) one-time token; only its digest is stored
    server-side. In demo mode the link is logged instead of emailed so the
    reset flow stays testable locally.
    """
    recipient = (email or "").strip().lower()
    link = f"{base_url.rstrip('/')}/reset_password/{token}" if base_url else \
        f"/reset_password/{token}"
    if not smtp_configured():
        logger.warning("Password reset for %s (demo mode, mail not configured): %s", recipient, link)
        return True

    subject = f"{APP_NAME} - Password reset request"
    body = (
        f"Hello,\n\n"
        f"A password reset was requested for your {APP_NAME} account.\n\n"
        f"Open this link to choose a new password (valid for 1 hour):\n"
        f"{link}\n\n"
        f"If you did not request this, you can safely ignore this email — "
        f"your current password keeps working.\n"
    )

    try:
        _send_mail(recipient, subject, body)
        logger.info("Password reset emailed to %s", recipient)
        return True
    except Exception as exc:  # noqa: BLE001 - surface as send failure
        logger.error("Failed to email password reset to %s: %s", recipient, exc)
        return False


def send_alert_email(subject: str, body: str, recipient: str) -> bool:
    """Send a plain-text alert email (price alerts, push notifications).

    Uses the same delivery configuration as OTP delivery. In demo mode (no
    mail credentials) the message is logged so behaviour is still observable.
    """
    recipient = (recipient or "").strip().lower()
    if not smtp_configured():
        logger.warning("Alert email (demo mode, mail not configured) to %s: %s | %s", recipient, subject, body)
        return True

    try:
        _send_mail(recipient, subject, body)
        logger.info("Alert email sent to %s", recipient)
        return True
    except Exception as exc:  # noqa: BLE001 - surface as send failure
        logger.error("Failed to email alert to %s: %s", recipient, exc)
        return False


def _send_mail(recipient: str, subject: str, body: str) -> None:
    """Deliver via Resend when configured, otherwise via SMTP.

    Raises on any failure so callers' existing try/except-and-log pattern
    keeps working unchanged.
    """
    if _resend_configured():
        _send_resend(recipient, subject, body)
        return
    _send_smtp(recipient, subject, body)


def _send_resend(recipient: str, subject: str, body: str) -> None:
    """Send one plain-text email via the Resend HTTP API."""
    if not RESEND_AVAILABLE:
        raise RuntimeError("resend package is not installed; email delivery unavailable")
    resend.api_key = os.getenv("RESEND_API_KEY")
    sender = os.getenv("SMTP_FROM") or _DEFAULT_FROM

    resend.Emails.send({
        "from": f"{APP_NAME} <{sender}>",
        "to": [recipient],
        "subject": subject,
        "text": body,
    })


def _send_smtp(recipient: str, subject: str, body: str) -> None:
    """Send one plain-text email via SMTP (stdlib ``smtplib``).

    Supports implicit TLS (``SMTP_USE_SSL``) and STARTTLS (``SMTP_USE_TLS``)
    matching the variables documented in ``.env.example``.
    """
    host = os.getenv("SMTP_HOST") or ""
    port = int(os.getenv("SMTP_PORT", "587") or 587)
    user = os.getenv("SMTP_USER") or ""
    password = os.getenv("SMTP_PASSWORD") or ""
    use_ssl = (os.getenv("SMTP_USE_SSL") or "").strip().lower() in ("1", "true", "yes", "on")
    use_tls = (os.getenv("SMTP_USE_TLS") or "").strip().lower() in ("1", "true", "yes", "on")

    if not host:
        raise RuntimeError("SMTP_HOST is not configured")

    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = subject
    message["From"] = _sender_address()
    message["To"] = recipient

    if use_ssl:
        client = smtplib.SMTP_SSL(host, port, timeout=30)
    else:
        client = smtplib.SMTP(host, port, timeout=30)
    try:
        if use_tls and not use_ssl:
            client.starttls()
        if user:
            client.login(user, password)
        client.send_message(message)
    finally:
        try:
            client.quit()
        except Exception:  # noqa: BLE001 - teardown best-effort
            pass