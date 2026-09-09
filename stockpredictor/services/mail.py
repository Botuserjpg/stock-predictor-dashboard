"""Email delivery helpers for OTP verification.

Sends real mail via the Resend HTTP API using environment configuration.
When Resend is not configured the app runs in demo mode: the OTP is logged
instead of emailed so the signup flow can still be exercised locally.

Configuration (see ``.env.example``):
    RESEND_API_KEY   - Resend API key (required for real delivery)
    SMTP_FROM        - sender address (defaults to onboarding@resend.dev)

Why HTTP instead of SMTP: many free hosting tiers (Render's free plan
included) block outbound SMTP ports (25/465/587) but allow normal outbound
HTTPS. Resend sends mail over a plain HTTPS POST request, so it works from
environments where raw SMTP connections time out.
"""
from __future__ import annotations

import logging
import os

import resend

logger = logging.getLogger("stock_predictor")

APP_NAME = "Stock Predictor Pro"

_DEFAULT_FROM = "onboarding@resend.dev"


def smtp_configured() -> bool:
    """Return True when real email delivery is configured.

    Name kept as ``smtp_configured`` so existing call sites don't need to
    change; it now checks for the Resend API key instead of SMTP host/user.
    """
    return bool(os.getenv("RESEND_API_KEY"))


def send_otp_email(email: str, otp: str) -> bool:
    """Email a one-time verification code.

    Returns True when the code was delivered (real send) or surfaced in demo
    mode (logged). Returns False when real delivery was requested but failed;
    the caller should surface an error to the user.
    """
    recipient = (email or "").strip().lower()
    if not smtp_configured():
        logger.warning("OTP for %s (demo mode, RESEND_API_KEY not configured): %s", recipient, otp)
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
        _send_resend(recipient, subject, body)
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
        logger.warning("Password reset for %s (demo mode, RESEND_API_KEY not configured): %s", recipient, link)
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
        _send_resend(recipient, subject, body)
        logger.info("Password reset emailed to %s", recipient)
        return True
    except Exception as exc:  # noqa: BLE001 - surface as send failure
        logger.error("Failed to email password reset to %s: %s", recipient, exc)
        return False


def send_alert_email(subject: str, body: str, recipient: str) -> bool:
    """Send a plain-text alert email (price alerts, push notifications).

    Uses the same delivery configuration as OTP delivery. In demo mode (no
    Resend key configured) the message is logged so behaviour is still
    observable locally.
    """
    recipient = (recipient or "").strip().lower()
    if not smtp_configured():
        logger.warning("Alert email (demo mode, RESEND_API_KEY not configured) to %s: %s | %s", recipient, subject, body)
        return True

    try:
        _send_resend(recipient, subject, body)
        logger.info("Alert email sent to %s", recipient)
        return True
    except Exception as exc:  # noqa: BLE001 - surface as send failure
        logger.error("Failed to email alert to %s: %s", recipient, exc)
        return False


def _send_resend(recipient: str, subject: str, body: str) -> None:
    """Send one plain-text email via the Resend HTTP API.

    Raises on any failure (bad key, invalid recipient, API error) so callers'
    existing try/except-and-log pattern keeps working unchanged.
    """
    resend.api_key = os.getenv("RESEND_API_KEY")
    sender = os.getenv("SMTP_FROM") or _DEFAULT_FROM

    resend.Emails.send({
        "from": f"{APP_NAME} <{sender}>",
        "to": [recipient],
        "subject": subject,
        "text": body,
    })