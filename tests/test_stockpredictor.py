"""Smoke tests for the refactored stockpredictor package.

State persistence is redirected to a throwaway directory BEFORE importing the
package so account/watchlist data written during tests never touches the real
``runtime_state/app_state.json`` used by the monolith.
"""
import os
import re
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote
from unittest.mock import patch

import production_core as pc

_TMP_DIR = tempfile.mkdtemp(prefix="stockpredictor_test_")
pc.STATE_DIR = Path(_TMP_DIR)
pc.STATE_FILE = pc.STATE_DIR / "app_state.json"

from stockpredictor import create_app  # noqa: E402
from stockpredictor.services.auth import user_store  # noqa: E402

# The developer's .env may configure real SMTP; the suite must run in demo mode
# (deterministic, offline, no real emails). Cleared after import so config.py's
# load_dotenv has already run.
for _SMTP_KEY in ("SMTP_HOST", "SMTP_USER"):
    os.environ.pop(_SMTP_KEY, None)

_VALID_PASSWORD = "Xk9!mnQp4"


def _csrf_token(client, url):
    response = client.get(url)
    match = re.search(r'name="csrf_token" value="([^"]+)"', response.get_data(as_text=True))
    return match.group(1) if match else None


def _register(client, email, password=_VALID_PASSWORD):
    """Register a new user through the full email-OTP flow.

    Demo mode (no SMTP configured) shows the code on the verify page and the
    code is also readable from the pending-registration store, so the OTP is
    pulled from ``user_store`` to complete the signup.
    """
    token = _csrf_token(client, "/register")
    response = client.post(
        "/register",
        data={"email": email, "password": password,
              "confirm_password": password, "csrf_token": token},
    )
    pending = user_store.pending_registrations.get(email.lower())
    if pending is None:
        return response  # validation failed before an OTP was issued
    otp = pending["otp"]
    token = _csrf_token(client, f"/verify_otp?email={quote(email)}")
    return client.post(
        "/verify_otp",
        data={"email": email, "otp": otp, "csrf_token": token},
    )


class AppFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
        cls.client = cls.app.test_client()

    def test_health_returns_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")

    def test_ready_returns_ok(self):
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)

    def test_register_page_renders(self):
        response = self.client.get("/register")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Create Account", response.data)

    def test_static_assets_served(self):
        css = self.client.get("/static/css/app.css")
        self.assertEqual(css.status_code, 200)
        js = self.client.get("/static/js/app.js")
        self.assertEqual(js.status_code, 200)

    def test_missing_route_returns_404_page(self):
        response = self.client.get("/definitely/not/a/route")
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"Page Not Found", response.data)

    def test_public_demo_is_read_only_and_root_links_to_it(self):
        app = create_app({
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "PUBLIC_DEMO": True,
        })
        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 302)
        self.assertTrue(client.get("/").location.endswith("/demo"))
        with patch("stockpredictor.views.dashboard.stocks.get_market_indices", return_value={}), \
             patch("stockpredictor.views.dashboard._fetch_live", return_value={"symbol": "AAPL", "quote": {}, "closes": []}):
            response = client.get("/demo")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Public read-only demo", response.data)


class CsrfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
        cls.client = cls.app.test_client()

    def test_post_without_csrf_token_rejected(self):
        response = self.client.post(
            "/register",
            data={"email": "csrf@example.com", "password": _VALID_PASSWORD,
                  "confirm_password": _VALID_PASSWORD},
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertIn("CSRF", payload["message"])

    def test_post_with_invalid_csrf_token_rejected(self):
        response = self.client.post(
            "/register",
            data={"email": "csrf@example.com", "password": _VALID_PASSWORD,
                  "confirm_password": _VALID_PASSWORD, "csrf_token": "bogus-token"},
        )
        self.assertEqual(response.status_code, 400)


class AuthFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()

    def _register(self, email, password=_VALID_PASSWORD):
        return _register(self.client, email, password)

    def test_register_logs_user_in_and_reaches_dashboard(self):
        response = self._register("alice@example.com")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/dashboard"))

        watchlist = self.client.get("/api/watchlist/get")
        self.assertEqual(watchlist.status_code, 200)

    def test_register_rejects_weak_password(self):
        response = self._register("bob@example.com", password="weak")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"at least 8 characters", response.data)

    def test_login_flow_and_watchlist_api(self):
        email = "carol@example.com"
        self._register(email)
        self.client.get("/logout")

        token = _csrf_token(self.client, "/login")
        response = self.client.post(
            "/login",
            data={"username": email, "password": _VALID_PASSWORD,
                  "csrf_token": token},
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/dashboard"))

        watchlist = self.client.get("/api/watchlist/get")
        self.assertEqual(watchlist.status_code, 200)
        self.assertEqual(watchlist.get_json()["watchlist"], [])

    def test_dashboard_requires_login(self):
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])


class SecurityHardeningTests(unittest.TestCase):
    """Account lockout, roles and persisted password-reset tokens."""

    @classmethod
    def setUpClass(cls):
        from stockpredictor.services.auth import user_store

        cls.user_store = user_store
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()

    def _register(self, email, password=_VALID_PASSWORD):
        return _register(self.client, email, password)

    def _login(self, email, password=_VALID_PASSWORD):
        token = _csrf_token(self.client, "/login")
        return self.client.post(
            "/login",
            data={"username": email, "password": password, "csrf_token": token},
        )

    def test_account_locks_after_max_failed_attempts(self):
        from stockpredictor.services.security import MAX_FAILED_LOGIN_ATTEMPTS

        email = "lockme@example.com"
        self._register(email)
        self.client.get("/logout")

        for _ in range(MAX_FAILED_LOGIN_ATTEMPTS):
            self._login(email, password="WrongPass1!")

        self.assertTrue(self.user_store.is_locked(email))
        user = self.user_store.get_user(email)
        self.assertEqual(user["failed_attempts"], MAX_FAILED_LOGIN_ATTEMPTS)
        self.assertIsNotNone(user["locked_until"])

        response = self._login(email, password=_VALID_PASSWORD)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"temporarily locked", response.data)

    def test_unlock_clears_lockout(self):
        from stockpredictor.services.security import MAX_FAILED_LOGIN_ATTEMPTS

        email = "unlockme@example.com"
        self._register(email)
        self.client.get("/logout")
        for _ in range(MAX_FAILED_LOGIN_ATTEMPTS):
            self._login(email, password="WrongPass1!")
        self.assertTrue(self.user_store.is_locked(email))

        self.assertTrue(self.user_store.unlock(email))
        self.assertFalse(self.user_store.is_locked(email))
        response = self._login(email)
        self.assertEqual(response.status_code, 302)

    def test_authenticate_reports_remaining_attempts(self):
        email = "attempts@example.com"
        self._register(email)
        self.client.get("/logout")
        response = self._login(email, password="WrongPass1!")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"attempts remaining", response.data)

    def test_password_reset_token_is_persisted_and_usable(self):
        from unittest import mock

        from stockpredictor.services.security import hash_secret

        email = "resetme@example.com"
        self._register(email)
        self.client.get("/logout")

        with mock.patch(
            "stockpredictor.views.auth.send_password_reset_email"
        ) as fake_send:
            token = _csrf_token(self.client, "/forgot_password")
            response = self.client.post(
                "/forgot_password",
                data={"email": email, "csrf_token": token},
            )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(fake_send.called, "reset email should be dispatched")
        reset_token = fake_send.call_args[0][1]

        # Only the digest is stored at rest; the raw token must never land
        # in the persisted user store.
        tokens = self.user_store.reset_tokens()
        self.assertIn(hash_secret(reset_token), tokens,
                      "hashed reset token should be persisted")
        self.assertNotIn(reset_token, tokens,
                         "raw reset token must not be stored verbatim")

        new_password = "Tr3s!kL9vQ#"
        token = _csrf_token(self.client, f"/reset_password/{reset_token}")
        response = self.client.post(
            f"/reset_password/{reset_token}",
            data={"password": new_password, "confirm_password": new_password,
                  "csrf_token": token},
        )
        self.assertEqual(response.status_code, 302)
        self.assertNotIn(reset_token, self.user_store.reset_tokens())

        login = self._login(email, password=new_password)
        self.assertEqual(login.status_code, 302)


class AdminRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stockpredictor.services.auth import user_store

        cls.user_store = user_store
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()

    def _register(self, email):
        self.client.get("/logout")
        return _register(self.client, email)

    def _login(self, email):
        token = _csrf_token(self.client, "/login")
        return self.client.post(
            "/login",
            data={"username": email, "password": _VALID_PASSWORD, "csrf_token": token},
        )

    def _admin_token(self):
        response = self.client.get("/admin/users")
        match = re.search(r'name="csrf-token" content="([^"]+)"', response.get_data(as_text=True))
        return match.group(1) if match else None

    def test_admin_requires_authentication(self):
        response = self.client.get("/admin")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_non_admin_gets_forbidden(self):
        email = "dave@example.com"
        self._register(email)
        response = self.client.get("/admin")
        self.assertEqual(response.status_code, 403)

    def test_admin_can_view_console_and_users(self):
        email = "admin1@example.com"
        self._register(email)
        self.assertTrue(self.user_store.set_role(email, "admin"))

        response = self.client.get("/admin")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Admin Console", response.data)

        response = self.client.get("/admin/users")
        self.assertEqual(response.status_code, 200)
        self.assertIn(email.encode(), response.data)

        response = self.client.get("/admin/audit")
        self.assertEqual(response.status_code, 200)

    def test_admin_can_unlock_user_via_api(self):
        from stockpredictor.services.security import MAX_FAILED_LOGIN_ATTEMPTS

        victim = "victim@example.com"
        self._register(victim)
        victim_client = self.app.test_client()
        token = _csrf_token(victim_client, "/login")
        for _ in range(MAX_FAILED_LOGIN_ATTEMPTS):
            victim_client.post(
                "/login",
                data={"username": victim, "password": "WrongPass1!", "csrf_token": token},
            )
        self.assertTrue(self.user_store.is_locked(victim))

        admin_email = "admin2@example.com"
        self._register(admin_email)
        self.assertTrue(self.user_store.set_role(admin_email, "admin"))

        response = self.client.post(
            f"/admin/users/{victim}/unlock",
            data={"csrf_token": self._admin_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])
        self.assertFalse(self.user_store.is_locked(victim))

    def test_admin_cannot_demote_self(self):
        email = "admin3@example.com"
        self._register(email)
        self.assertTrue(self.user_store.set_role(email, "admin"))
        response = self.client.post(
            f"/admin/users/{email}/role",
            data={"role": "user", "csrf_token": self._admin_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["success"])
        self.assertTrue(self.user_store.is_admin(email))


class OtpFlowTests(unittest.TestCase):
    """Email OTP verification during signup."""

    @classmethod
    def setUpClass(cls):
        cls.user_store = user_store
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()

    def _submit_registration(self, email, password=_VALID_PASSWORD):
        token = _csrf_token(self.client, "/register")
        return self.client.post(
            "/register",
            data={"email": email, "password": password,
                  "confirm_password": password, "csrf_token": token},
        )

    def test_register_requires_otp_before_account_created(self):
        email = "otp1@example.com"
        response = self._submit_registration(email)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Verify Your Email", response.data)
        self.assertNotIn(email, self.user_store.users)
        self.assertIn(email, self.user_store.pending_registrations)

    def test_verify_page_shows_demo_code_when_smtp_unconfigured(self):
        email = "otp2@example.com"
        self._submit_registration(email)
        response = self.client.get(f"/verify_otp?email={quote(email)}")
        self.assertEqual(response.status_code, 200)
        expected = self.user_store.pending_registrations[email]["otp"].encode()
        self.assertIn(expected, response.data)

    def test_wrong_otp_is_rejected_and_tracks_attempts(self):
        email = "otp3@example.com"
        self._submit_registration(email)
        token = _csrf_token(self.client, f"/verify_otp?email={quote(email)}")
        response = self.client.post(
            "/verify_otp",
            data={"email": email, "otp": "000000", "csrf_token": token},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid verification code", response.data)
        self.assertNotIn(email, self.user_store.users)
        self.assertEqual(self.user_store.pending_registrations[email]["attempts"], 1)

    def test_resend_otp_issues_a_new_code(self):
        email = "otp4@example.com"
        self._submit_registration(email)
        first = self.user_store.pending_registrations[email]["otp"]
        token = _csrf_token(self.client, f"/verify_otp?email={quote(email)}")
        response = self.client.post(
            "/resend_otp",
            data={"email": email, "csrf_token": token},
        )
        self.assertEqual(response.status_code, 302)
        second = self.user_store.pending_registrations[email]["otp"]
        self.assertNotEqual(first, second)

    def test_completed_otp_creates_account_and_logs_in(self):
        email = "otp5@example.com"
        response = _register(self.client, email)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/dashboard"))
        user = self.user_store.users[email]
        self.assertTrue(user.get("email_verified"))
        self.assertNotIn(email, self.user_store.pending_registrations)


class AccountDeletionTests(unittest.TestCase):
    """Account settings page and permanent account deletion."""

    @classmethod
    def setUpClass(cls):
        cls.user_store = user_store
        cls.app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})

    def setUp(self):
        self.client = self.app.test_client()

    def _register(self, email):
        return _register(self.client, email)

    def _delete(self, email, confirm=None):
        token = _csrf_token(self.client, "/account")
        return self.client.post(
            "/account/delete",
            data={"confirm_email": confirm if confirm is not None else email,
                  "csrf_token": token},
        )

    def test_account_page_requires_login(self):
        response = self.client.get("/account")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_delete_account_removes_user_and_associated_data(self):
        email = "delete1@example.com"
        self._register(email)
        self.user_store.watchlists()[email] = ["AAPL", "MSFT"]
        self.user_store.portfolios()[email] = {"cash": 10000.0}
        self.user_store.persist()

        response = self._delete(email)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/register"))
        self.assertNotIn(email, self.user_store.users)
        self.assertNotIn(email, self.user_store.watchlists())
        self.assertNotIn(email, self.user_store.portfolios())

        # session cleared -> dashboard requires login again
        dashboard = self.client.get("/dashboard")
        self.assertEqual(dashboard.status_code, 302)
        self.assertIn("/login", dashboard.headers["Location"])

    def test_delete_account_requires_matching_confirmation_email(self):
        email = "delete2@example.com"
        self._register(email)
        response = self._delete(email, confirm="someone-else@example.com")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/account"))
        self.assertIn(email, self.user_store.users)

    def test_delete_account_requires_csrf(self):
        email = "delete3@example.com"
        self._register(email)
        response = self.client.post("/account/delete", data={"confirm_email": email})
        self.assertEqual(response.status_code, 400)
        self.assertIn(email, self.user_store.users)

    def test_delete_account_rejects_unauthenticated(self):
        token = _csrf_token(self.client, "/login")
        response = self.client.post(
            "/account/delete",
            data={"confirm_email": "x@example.com", "csrf_token": token},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])


class MailServiceTests(unittest.TestCase):
    """Demo-mode OTP delivery when SMTP is not configured."""

    _ENV_KEYS = ("SMTP_HOST", "SMTP_USER")

    def setUp(self):
        self._saved = {key: os.getenv(key) for key in self._ENV_KEYS}

    def tearDown(self):
        for key in self._ENV_KEYS:
            if self._saved[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = self._saved[key]

    def test_demo_mode_when_smtp_unconfigured(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)
        from stockpredictor.services.mail import send_otp_email, smtp_configured

        self.assertFalse(smtp_configured())
        self.assertTrue(send_otp_email("demo@example.com", "123456"))

    def test_smtp_configured_when_host_and_user_set(self):
        os.environ["SMTP_HOST"] = "smtp.example.com"
        os.environ["SMTP_USER"] = "user@example.com"
        from stockpredictor.services.mail import smtp_configured

        self.assertTrue(smtp_configured())


if __name__ == "__main__":
    unittest.main()
