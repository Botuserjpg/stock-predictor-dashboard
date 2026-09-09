"""Email delivery helpers for OTP verification.

Sends real mail over SMTP using environment configuration. When SMTP is not
configured the app runs in demo mode: the OTP is logged instead of emailed so
the signup flow can still be exercised locally.

Configuration (see ``.env.example``):
    SMTP_HOST        - SMTP server hostname (required for real delivery)
    SMTP_PORT        - port (587 default; 465 for implicit SSL)
    SMTP_USER        - login username (required for real delivery)
    SMTP_PASSWORD    - login password
    SMTP_FROM        - sender address (defaults to SMTP_USER)
    SMTP_USE_TLS     - STARTTLS on the plaintext port (default true)
    SMTP_USE_SSL     - implicit TLS (e.g. port 465); overrides SMTP_USE_TLS
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

logger = logging.getLogger("stock_predictor")

APP_NAME = "Stock Predictor Pro"

_TRUTHY = {"1", "true", "yes", "on"}


def smtp_configured() -> bool:
    """Return True when a real SMTP server can be used for delivery."""
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_USER"))


def send_otp_email(email: str, otp: str) -> bool:
    """Email a one-time verification code.

    Returns True when the code was delivered (real SMTP) or surfaced in demo
    mode (logged). Returns False when real delivery was requested but failed;
    the caller should surface an error to the user.
    """
    recipient = (email or "").strip().lower()
    if not smtp_configured():
        logger.warning("OTP for %s (demo mode, SMTP not configured): %s", recipient, otp)
        return True

    sender = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER")
    body = (
        f"Hello,\n\n"
        f"Your {APP_NAME} verification code is: {otp}\n\n"
        f"Enter this code on the signup page to confirm your email address. "
        f"It expires in a few minutes.\n\n"
        f"If you did not request this code, you can safely ignore this email.\n"
    )
    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = f"{APP_NAME} - Your verification code"
    message["From"] = formataddr((APP_NAME, sender))
    message["To"] = recipient

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD", "")
    use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() in _TRUTHY

    try:
        _send_smtp(sender, recipient, message.as_string(), use_ssl, host, port, username, password)
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
        logger.warning("Password reset for %s (demo mode, SMTP not configured): %s", recipient, link)
        return True

    sender = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER")
    body = (
        f"Hello,\n\n"
        f"A password reset was requested for your {APP_NAME} account.\n\n"
        f"Open this link to choose a new password (valid for 1 hour):\n"
        f"{link}\n\n"
        f"If you did not request this, you can safely ignore this email — "
        f"your current password keeps working.\n"
    )
    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = f"{APP_NAME} - Password reset request"
    message["From"] = formataddr((APP_NAME, sender))
    message["To"] = recipient

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD", "")
    use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() in _TRUTHY

    try:
        _send_smtp(sender, recipient, message.as_string(), use_ssl, host, port, username, password)
        logger.info("Password reset emailed to %s", recipient)
        return True
    except Exception as exc:  # noqa: BLE001 - surface as send failure
        logger.error("Failed to email password reset to %s: %s", recipient, exc)
        return False


def send_alert_email(subject: str, body: str, recipient: str) -> bool:
    """Send a plain-text alert email (price alerts, push notifications).

    Uses the same SMTP configuration as OTP delivery. In demo mode (no SMTP
    configured) the message is logged so behaviour is still observable locally.
    """
    recipient = (recipient or "").strip().lower()
    if not smtp_configured():
        logger.warning("Alert email (demo mode, SMTP not configured) to %s: %s | %s", recipient, subject, body)
        return True

    sender = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER")
    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = subject
    message["From"] = formataddr((APP_NAME, sender))
    message["To"] = recipient

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD", "")
    use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() in _TRUTHY

    try:
        _send_smtp(sender, recipient, message.as_string(), use_ssl, host, port, username, password)
        logger.info("Alert email sent to %s", recipient)
        return True
    except Exception as exc:  # noqa: BLE001 - surface as send failure
        logger.error("Failed to email alert to %s: %s", recipient, exc)
        return False


def _send_smtp(sender, recipient, payload, use_ssl, host, port, username, password) -> None:
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=30) as server:
            if username:
                server.login(username, password)
            server.sendmail(sender, [recipient], payload)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in _TRUTHY
            if use_tls:
                server.starttls()
            if username:
                server.login(username, password)
            server.sendmail(sender, [recipient], payload)
