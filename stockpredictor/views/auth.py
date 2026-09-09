"""Authentication blueprint: register (email OTP), login, logout, reset."""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from production_core import audit_log

from ..services.auth import user_store
from ..services.mail import send_otp_email, send_password_reset_email, smtp_configured
from ..services.security import hash_secret, validate_password_strength, verify_secret

bp = Blueprint("auth", __name__)

RESET_TOKEN_TTL_HOURS = 1


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.home"))
    error = None
    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if password != confirm:
            error = "Passwords do not match"
        else:
            error, otp = user_store.start_otp_registration(email, password)
            if error is None:
                email = email.strip().lower()
                audit_log("user.otp_sent", email, {"email": email})
                if not send_otp_email(email, otp):
                    user_store.cancel_otp_registration(email)
                    error = (
                        "We couldn't send the verification email. "
                        "Please check the SMTP settings and try again."
                    )
                elif smtp_configured():
                    flash(
                        "A verification code has been sent to your email. "
                        "It expires in a few minutes.",
                        "info",
                    )
                    return redirect(url_for("auth.verify_otp", email=email))
                else:
                    # Demo mode (SMTP not configured): show the code on the verify page.
                    return render_template(
                        "auth/verify_otp.html", email=email, error=None, demo_otp=otp
                    )
    return render_template("auth/register.html", error=error, email=request.form.get("email", ""))


@bp.route("/verify_otp", methods=["GET", "POST"])
def verify_otp():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.home"))
    email = (request.form.get("email") or request.args.get("email") or "").strip().lower()
    error = None
    demo_otp = None

    if not email:
        error = "Please enter your email address to continue."
    else:
        entry = user_store.pending_registrations.get(email)
        if entry is None:
            error = "No pending registration found for this email. Please register again."
        elif (entry.get("expires") or "") < datetime.now().isoformat():
            error = "Verification code has expired. Please register again."
        else:
            if not smtp_configured():
                demo_otp = entry.get("otp")
            if request.method == "POST":
                otp = request.form.get("otp", "")
                error = user_store.complete_otp_registration(email, otp)
                if error is None:
                    audit_log("user.registered", email, {"email": email, "verified": True})
                    login_user(_user(email))
                    flash("Account created. Welcome!", "success")
                    return redirect(url_for("dashboard.home"))

    return render_template("auth/verify_otp.html", email=email, error=error, demo_otp=demo_otp)


@bp.post("/resend_otp")
def resend_otp():
    email = (request.form.get("email") or "").strip().lower()
    otp = user_store.resend_otp(email)
    if otp is None:
        flash("No pending registration found for this email. Please register again.", "error")
        return redirect(url_for("auth.register"))
    audit_log("user.otp_resent", email, {"email": email})
    if not send_otp_email(email, otp):
        flash("We couldn't send the verification email. Please try again.", "error")
    else:
        flash("A new verification code has been sent to your email.", "info")
    return redirect(url_for("auth.verify_otp", email=email))


@bp.get("/account")
@login_required
def account():
    """Account settings page with the option to delete the account."""
    user = user_store.get_user(current_user.email) or {}
    return render_template("auth/account.html", account=user)


@bp.post("/account/delete")
@login_required
def delete_account():
    email = current_user.email
    confirm = (request.form.get("confirm_email") or "").strip().lower()
    if confirm != email:
        flash("Account not deleted: the confirmation email did not match.", "error")
        return redirect(url_for("auth.account"))
    if not user_store.delete_account(email):
        flash("Could not delete the account. Please try again.", "error")
        return redirect(url_for("auth.account"))
    audit_log("user.account_deleted", email, {"email": email})
    logout_user()
    flash("Your account and associated data have been deleted.", "info")
    return redirect(url_for("auth.register"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.home"))
    error = None
    if request.method == "POST":
        email = request.form.get("username", "")
        password = request.form.get("password", "")
        remember_me = request.form.get("remember_me") == "on"
        error = user_store.authenticate(email, password)
        if error is None:
            audit_log("user.login", email, {"remember_me": remember_me})
            login_user(_user(email), remember=remember_me)
            return redirect(url_for("dashboard.home"))
        audit_log("user.login_failed", email, {})
    return render_template("auth/login.html", error=error)


@bp.get("/logout")
@login_required
def logout():
    audit_log("user.logout", current_user.email)
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


@bp.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if email in user_store.users:
            token = secrets.token_urlsafe(32)
            tokens = user_store.reset_tokens()
            # Only the digest is persisted; the raw token exists solely inside
            # the one-time email link (demo mode: server log).
            tokens[hash_secret(token)] = {
                "email": email,
                "expires": (datetime.now() + timedelta(hours=RESET_TOKEN_TTL_HOURS)).isoformat(),
            }
            user_store.persist()
            audit_log("user.reset_requested", email, {"email": email})
            send_password_reset_email(email, token, base_url=request.url_root)
            flash(
                "If that account exists, a reset link has been sent "
                "(in demo mode the link appears in the server logs).",
                "info",
            )
            return redirect(url_for("auth.forgot_password"))
        # Same generic response for unknown accounts so the endpoint cannot be
        # used to enumerate registered emails.
        flash(
            "If that account exists, a reset link has been sent "
            "(in demo mode the link appears in the server logs).",
            "info",
        )
        return redirect(url_for("auth.forgot_password"))
    return render_template("auth/forgot_password.html", error=error)


@bp.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token: str):
    tokens = user_store.reset_tokens()
    entry = tokens.get(hash_secret(token))
    if entry is None or entry["expires"] < datetime.now().isoformat():
        return render_template("auth/forgot_password.html",
                               error="Reset link is invalid or has expired."), 400
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if password != confirm:
            return render_template("auth/reset_password.html", error="Passwords do not match", token=token)
        ok, msg = validate_password_strength(password)
        if not ok:
            return render_template("auth/reset_password.html", error=msg, token=token)
        if user_store.set_password(entry["email"], password):
            tokens.pop(hash_secret(token), None)
            user_store.persist()
            audit_log("user.password_reset", entry["email"], {"email": entry["email"]})
        flash("Password updated. Please log in.", "success")
        return redirect(url_for("auth.login"))
    return render_template("auth/reset_password.html", token=token)


def _user(email: str):
    from ..extensions import User

    return User(email)
