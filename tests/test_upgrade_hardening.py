"""Regression coverage for the production-hardening upgrade set.

Mirrors the conventions of ``test_stockpredictor.py``: runtime state is
redirected to a throwaway directory BEFORE any project import so the suite
never touches developer state.
"""
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

_TMP_DIR = Path(tempfile.mkdtemp(prefix="spp_upgrade_"))

import sys

import production_core as pc  # noqa: E402

# Redirect runtime state BEFORE the package loads — but only when this module
# is the FIRST project importer (standalone run). Under ``unittest discover``
# an earlier suite has already bound ``stockpredictor.database`` to its
# sandbox; re-pointing shared globals here would desync static bindings from
# dynamic readers and break every suite in the process.
if "stockpredictor" not in sys.modules:
    pc.STATE_DIR = _TMP_DIR / "state"
    pc.AUDIT_DIR = _TMP_DIR / "audit"
    pc.STATE_DIR.mkdir(parents=True, exist_ok=True)
    pc.AUDIT_DIR.mkdir(parents=True, exist_ok=True)

import stockpredictor.services.jobs as jobs_mod  # noqa: E402

from stockpredictor import create_app  # noqa: E402
from stockpredictor.database import init_db  # noqa: E402
from stockpredictor.services.auth import user_store  # noqa: E402
from stockpredictor.services.jobs import JobManager  # noqa: E402
from stockpredictor.services.security import (  # noqa: E402
    MAX_FAILED_LOGIN_ATTEMPTS,
    hash_secret,
)

# The developer's .env may configure real SMTP; force demo mode (same as the
# main suite) so OTP handling is deterministic and offline.
for _SMTP_KEY in ("SMTP_HOST", "SMTP_USER"):
    os.environ.pop(_SMTP_KEY, None)

VALID_PASSWORD = "Str0ng!Passw0rd#"


def _csrf_token(client, url):
    page = client.get(url).get_data(as_text=True)
    match = re.search(r'name="csrf-token" content="([^"]+)"', page)
    return match.group(1) if match else ""


def _audit_events():
    path = pc.AUDIT_DIR / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class ClientAddrTrustTests(unittest.TestCase):
    """client_addr honours X-Forwarded-For only from trusted peers."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def _addr(self, remote="10.0.0.9", xff=None):
        headers = {"X-Forwarded-For": xff} if xff else {}
        with self.app.test_request_context(
            "/", environ_base={"REMOTE_ADDR": remote}, headers=headers
        ):
            return pc.client_addr()

    def test_no_trusted_proxies_header_is_ignored(self):
        self.assertEqual(self._addr(remote="10.0.0.9", xff="203.0.113.7"), "10.0.0.9")

    def test_trusted_proxy_rightmost_untrusted_hop_wins(self):
        with mock.patch.object(pc, "TRUSTED_PROXIES", (__import__("ipaddress").ip_network("10.0.0.0/8"),)):
            self.assertEqual(
                self._addr(remote="10.0.0.2", xff="203.0.113.7, 10.0.0.2"),
                "203.0.113.7",
            )

    def test_untrusted_peer_cannot_use_header_even_when_proxies_configured(self):
        with mock.patch.object(pc, "TRUSTED_PROXIES", (__import__("ipaddress").ip_network("10.0.0.0/8"),)):
            self.assertEqual(
                self._addr(remote="192.168.5.5", xff="203.0.113.7"), "192.168.5.5"
            )

    def test_malformed_chain_falls_back_to_peer(self):
        with mock.patch.object(pc, "TRUSTED_PROXIES", (__import__("ipaddress").ip_network("10.0.0.0/8"),)):
            self.assertEqual(
                self._addr(remote="10.0.0.2", xff="not-an-ip"), "10.0.0.2"
            )


class ForgedHeaderRateLimitTests(unittest.TestCase):
    """Rotating X-Forwarded-For must not rotate the rate-limit bucket."""

    def setUp(self):
        self._snapshot = dict(pc.rate_limiter._events)
        pc.rate_limiter._events.clear()

    def tearDown(self):
        pc.rate_limiter._events.clear()
        pc.rate_limiter._events.update(self._snapshot)

    def test_spoofed_xff_shares_one_auth_bucket(self):
        # TESTING mode disables rate limiting (unit suites exceed budgets);
        # this test needs the real guard, so it runs with TESTING off.
        app = create_app({"TESTING": False, "SECRET_KEY": "test-secret"})
        client = app.test_client()
        statuses = []
        for i in range(21):
            response = client.get(
                "/login", headers={"X-Forwarded-For": f"203.0.{i // 256}.{i % 256}"}
            )
            statuses.append(response.status_code)
        self.assertTrue(all(code == 200 for code in statuses[:20]))
        self.assertEqual(statuses[20], 429)


class SecurityHeaderTests(unittest.TestCase):
    def _headers(self, extra_config=None, path="/health"):
        config = {"TESTING": True, "SECRET_KEY": "test-secret"}
        config.update(extra_config or {})
        app = create_app(config)
        response = app.test_client().get(path)
        return response.headers

    def test_csp_is_report_only_by_default(self):
        headers = self._headers()
        self.assertIn("Content-Security-Policy-Report-Only", headers)
        self.assertNotIn("Content-Security-Policy", headers)

    def test_csp_enforce_flag_switches_header(self):
        headers = self._headers({"CSP_ENFORCE": True})
        self.assertIn("Content-Security-Policy", headers)
        self.assertNotIn("Content-Security-Policy-Report-Only", headers)


class ProductionDefaultsTests(unittest.TestCase):
    def test_production_secure_defaults(self):
        from stockpredictor.config import Settings

        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(
                ("SESSION_COOKIE_", "FLASK_ENV", "HOST", "CSP_", "METRICS_", "LOG_FORMAT")
            )
        }
        with mock.patch.dict(os.environ, {**env, "FLASK_ENV": "production"}, clear=True):
            settings = Settings()
        self.assertTrue(settings.SESSION_COOKIE_SECURE)
        self.assertEqual(settings.HOST, "0.0.0.0")

    def test_development_stays_local_and_insecure_for_http(self):
        from stockpredictor.config import Settings

        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("SESSION_COOKIE_", "FLASK_ENV", "HOST"))
        }
        with mock.patch.dict(os.environ, {**env, "FLASK_ENV": "development"}, clear=True):
            settings = Settings()
        self.assertFalse(settings.SESSION_COOKIE_SECURE)
        self.assertEqual(settings.HOST, "127.0.0.1")


class JobOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.manager = JobManager()

    @classmethod
    def setUpClass(cls):
        # Job artifacts live under the jobs subsystem only; no other suite
        # touches them, so repointing mid-run is safe.
        jobs_mod.JOBS_DIR = _TMP_DIR / "jobs"
        jobs_mod.JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def _wait_complete(self, job_id, requester, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.manager.status(job_id, requester=requester)
            if status and status["status"] == "complete":
                return status
            time.sleep(0.02)
        self.fail(f"job {job_id} did not complete in time")

    def test_same_owner_dedupes_different_owner_gets_own_job(self):
        first = self.manager.submit(lambda: {"v": 1}, dedupe_key="dup", owner="a@x.com")
        again = self.manager.submit(lambda: {"v": 1}, dedupe_key="dup", owner="a@x.com")
        other = self.manager.submit(lambda: {"v": 2}, dedupe_key="dup", owner="b@x.com")
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_owner_scoping_blocks_other_requesters(self):
        job_id = self.manager.submit(lambda: {"price": 42}, owner="a@x.com")
        status = self.manager.status(job_id, requester="b@x.com")
        self.assertIsNone(status)
        completed = self._wait_complete(job_id, requester="a@x.com")
        self.assertEqual(completed["result"], {"price": 42})

    def test_disk_artifacts_are_owner_scoped(self):
        owned = {"owner": "a@x.com", "result": {"v": 9}}
        (jobs_mod.JOBS_DIR / "ownedjob01.json").write_text(json.dumps(owned), encoding="utf-8")
        legacy = {"v": 7}
        (jobs_mod.JOBS_DIR / "legacyjob01.json").write_text(json.dumps(legacy), encoding="utf-8")

        self.assertIsNone(self.manager.status("ownedjob01", requester="b@x.com"))
        self.assertEqual(
            self.manager.status("ownedjob01", requester="a@x.com")["result"], {"v": 9}
        )
        # Pre-ownership artifacts remain readable by any authenticated caller.
        self.assertEqual(
            self.manager.status("legacyjob01", requester="whoever@x.com")["result"], {"v": 7}
        )


class OtpHashingTests(unittest.TestCase):
    def setUp(self):
        user_store.pending_registrations.pop("hashedotp@example.com", None)
        user_store.pending_registrations.pop("plainotp@example.com", None)

    def test_otp_hashed_when_smtp_configured(self):
        email = "hashedotp@example.com"
        with mock.patch("stockpredictor.services.mail.smtp_configured", return_value=True):
            self.assertIsNone(user_store.start_otp_registration(email, VALID_PASSWORD))
            otp = user_store.resend_otp(email)
            stored = user_store.pending_otp(email)
            self.assertIsNotNone(otp)
            self.assertTrue(str(stored).startswith("sha256$"))
            self.assertNotEqual(stored, otp)
            error = user_store.complete_otp_registration(email, otp)
            self.assertIsNone(error)
        self.assertIn(email, user_store.users)
        self.assertNotIn(email, user_store.pending_registrations)

    def test_demo_mode_keeps_displayable_plaintext(self):
        email = "plainotp@example.com"
        self.assertIsNone(user_store.start_otp_registration(email, VALID_PASSWORD))
        otp = user_store.resend_otp(email)
        stored = user_store.pending_otp(email)
        self.assertEqual(stored, otp)  # demo UI shows the code; no SMTP to deliver it
        self.assertFalse(str(stored).startswith("sha256$"))


class AuditEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._audit_backup = pc.AUDIT_DIR
        pc.AUDIT_DIR = _TMP_DIR / "audit"
        pc.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        pc.AUDIT_DIR = cls._audit_backup

    def _register(self, email):
        """Create the account via the service layer (no template coupling),
        then make sure no session is logged in before exercising login."""
        self.client.get("/logout")
        error = user_store.start_otp_registration(email, VALID_PASSWORD)
        self.assertIsNone(error)
        error = user_store.complete_otp_registration(
            email, user_store.pending_otp(email)
        )
        self.assertIsNone(error)

    def test_login_failure_logout_and_lockout_are_audited(self):
        email = "auditevents@example.com"
        self._register(email)

        token = _csrf_token(self.client, "/login")
        self.client.post("/login", data={"username": email, "password": "Nope!12345",
                                         "csrf_token": token})
        events = [e["event"] for e in _audit_events()]
        self.assertIn("user.login_failed", events)

        for _ in range(MAX_FAILED_LOGIN_ATTEMPTS):
            token = _csrf_token(self.client, "/login")
            self.client.post("/login", data={"username": email,
                                             "password": "Nope!12345",
                                             "csrf_token": token})
        events = [e["event"] for e in _audit_events()]
        self.assertIn("user.account_locked", events)

        user_store.unlock(email)
        token = _csrf_token(self.client, "/login")
        self.client.post("/login", data={"username": email,
                                         "password": VALID_PASSWORD,
                                         "csrf_token": token})
        self.client.get("/logout")
        events = [e["event"] for e in _audit_events()]
        self.assertIn("user.logout", events)


class WalPragmaTests(unittest.TestCase):
    def test_journal_mode_is_wal_after_init_db(self):
        from stockpredictor.database import db_path

        init_db()
        conn = sqlite3.connect(db_path())
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(mode.lower(), "wal")


class ConcurrentRegistrationTests(unittest.TestCase):
    def test_parallel_registrations_all_persist(self):
        emails = [f"conc{i}@example.com" for i in range(8)]
        errors = []

        def worker(email):
            try:
                error = user_store.start_otp_registration(email, VALID_PASSWORD)
                if error:
                    errors.append((email, error))
                    return
                otp = user_store.pending_otp(email)
                error = user_store.complete_otp_registration(email, otp)
                if error:
                    errors.append((email, error))
            except Exception as exc:  # pragma: no cover - surfaced via assertion
                errors.append((email, repr(exc)))

        threads = [threading.Thread(target=worker, args=(e,)) for e in emails]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(errors, [])
        for email in emails:
            self.assertIn(email, user_store.users)


class MetricsEndpointTests(unittest.TestCase):
    def test_disabled_returns_404(self):
        app = create_app({"TESTING": True, "SECRET_KEY": "k"})
        self.assertEqual(app.test_client().get("/metrics").status_code, 404)

    def test_enabled_exposes_prometheus_payload(self):
        app = create_app({"TESTING": True, "SECRET_KEY": "k", "METRICS_ENABLED": True})
        client = app.test_client()
        client.get("/health")  # generate one observation
        response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("spp_requests_total", body)
        self.assertIn("spp_request_latency_seconds_count", body)

    def test_token_protects_metrics(self):
        app = create_app({
            "TESTING": True, "SECRET_KEY": "k",
            "METRICS_ENABLED": True, "METRICS_TOKEN": "sekrit-token",
        })
        client = app.test_client()
        self.assertEqual(client.get("/metrics").status_code, 401)
        ok = client.get("/metrics", headers={"X-Metrics-Token": "sekrit-token"})
        self.assertEqual(ok.status_code, 200)


class JsonLogFormatterTests(unittest.TestCase):
    def test_record_serializes_to_json_line(self):
        record = logging.LogRecord(
            name="stock_predictor", level=20, pathname=__file__, lineno=1,
            msg="hello %s", args=("world",), exc_info=None,
        )
        record.request_id = "req-123"
        payload = json.loads(pc.JsonLogFormatter().format(record))
        self.assertEqual(payload["message"], "hello world")
        self.assertEqual(payload["request_id"], "req-123")
        self.assertEqual(payload["level"], "INFO")


if __name__ == "__main__":
    unittest.main()
