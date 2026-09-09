"""Admin blueprint: user management, audit log viewer, runtime stats."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, render_template, request
from flask_login import current_user, login_required

from production_core import AUDIT_DIR, audit_log, json_endpoint, state_db_path

from ..decorators import admin_required
from ..services.auth import ROLE_ADMIN, ROLE_USER, user_store
from ..services.security import validate_password_strength

bp = Blueprint("admin", __name__)

AUDIT_PAGE_SIZE = 50


def _read_audit_entries() -> List[Dict[str, Any]]:
    """Read all JSONL audit files, newest first."""
    entries: List[Dict[str, Any]] = []
    if not AUDIT_DIR.exists():
        return entries
    for path in sorted(AUDIT_DIR.glob("audit_*.jsonl"), reverse=True):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    entry.setdefault("file", path.name)
                    entries.append(entry)
        except (OSError, json.JSONDecodeError):
            continue
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries


def _paginate(items: List[Any], page: int, per_page: int) -> Dict[str, Any]:
    total = len(items)
    pages = max(1, -(-total // per_page))
    page = max(1, min(page, pages))
    start = (page - 1) * per_page
    return {
        "entries": items[start:start + per_page],
        "page": page,
        "pages": pages,
        "total": total,
    }


def _user_summary(email: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "email": email,
        "role": data.get("role", ROLE_USER),
        "created_at": data.get("created_at"),
        "last_login": data.get("last_login"),
        "login_count": data.get("login_count", 0),
        "failed_attempts": data.get("failed_attempts", 0),
        "locked_until": data.get("locked_until"),
    }


@bp.get("/admin")
@login_required
@admin_required
def index():
    users = user_store.users
    entries = _read_audit_entries()
    now = datetime.now().date().isoformat()
    today_events = [e for e in entries if (e.get("timestamp") or "").startswith(now)]

    locked = [
        email for email, data in users.items() if data.get("locked_until")
    ]
    return render_template(
        "admin/index.html",
        stats={
            "users": len(users),
            "admins": sum(1 for d in users.values() if d.get("role") == ROLE_ADMIN),
            "locked": len(locked),
            "audit_events": len(entries),
            "audit_today": len(today_events),
            "state_file_exists": state_db_path().exists(),
        },
        events_24h=[e for e in entries[:40]],
        events_top_event=Counter(e.get("event") for e in entries).most_common(10),
        current_user=current_user,
    )


@bp.get("/admin/users")
@login_required
@admin_required
def users():
    rows = [_user_summary(email, data) for email, data in user_store.users.items()]
    rows.sort(key=lambda r: r["email"])
    return render_template("admin/users.html", users=rows, current_user=current_user)


@bp.post("/admin/users/<email>/unlock")
@login_required
@admin_required
@json_endpoint
def unlock_user(email: str):
    clean = email.strip().lower()
    if user_store.unlock(clean):
        audit_log("admin.user_unlocked", current_user.email, {"target": clean})
        return jsonify({"success": True, "message": f"Unlocked {clean}"})
    return jsonify({"success": False, "message": "User not found"})


@bp.post("/admin/users/<email>/role")
@login_required
@admin_required
@json_endpoint
def set_role(email: str):
    clean = email.strip().lower()
    role = (request.form.get("role") or "").strip().lower()
    if clean == current_user.email and role != ROLE_ADMIN:
        return jsonify({"success": False, "message": "You cannot demote your own account"})
    if user_store.set_role(clean, role):
        audit_log("admin.role_changed", current_user.email, {"target": clean, "role": role})
        return jsonify({"success": True, "message": f"{clean} role set to {role}"})
    return jsonify({"success": False, "message": "User or role not found"})


@bp.post("/admin/users/<email>/password")
@login_required
@admin_required
@json_endpoint
def set_password(email: str):
    clean = email.strip().lower()
    password = request.form.get("password") or ""
    ok, message = validate_password_strength(password)
    if not ok:
        return jsonify({"success": False, "message": message})
    if user_store.set_password(clean, password):
        audit_log("admin.password_reset", current_user.email, {"target": clean})
        return jsonify({"success": True, "message": f"Password reset for {clean}"})
    return jsonify({"success": False, "message": "User not found"})


@bp.get("/admin/audit")
@login_required
@admin_required
def audit():
    event = (request.args.get("event") or "").strip()
    user_filter = (request.args.get("user") or "").strip().lower()
    try:
        page = max(1, int(request.args.get("page", 1)))
    except (TypeError, ValueError):
        page = 1

    entries = _read_audit_entries()
    if event:
        entries = [e for e in entries if e.get("event") == event]
    if user_filter:
        entries = [e for e in entries if (e.get("user_id") or "").lower() == user_filter]

    unique_events = sorted({e.get("event", "unknown") for e in _read_audit_entries()})
    return render_template(
        "admin/audit.html",
        events=_paginate(entries, page, AUDIT_PAGE_SIZE),
        event_options=unique_events,
        selected_event=event,
        user_filter=user_filter,
        current_user=current_user,
    )
