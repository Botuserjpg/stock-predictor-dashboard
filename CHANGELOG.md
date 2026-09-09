# Changelog

All notable changes to Stock Predictor Pro are documented here.
Format follows Keep a Changelog; versions are date-stamped.

## [2.1.0] - 2026-08-25 — Production-hardening upgrade

### Security
- **Client IP trust chain** (`production_core.client_addr`): `X-Forwarded-For`
  is honoured only when the direct peer is in `TRUSTED_PROXIES`; rightmost
  non-trusted hop wins. Rate limiting and audit records can no longer be
  spoofed via header injection.
- **Auth rate limiting actually fires**: endpoint matching now understands
  blueprint namespaces (`auth.login`); previously the limiter never matched
  any route. Disabled under `TESTING` so unit suites aren't throttled.
- **Reset tokens hashed at rest** (SHA-256, `sha256$` prefix): a database or
  state-dump leak no longer yields usable reset links. Legacy plaintext rows
  still verify (transparent migration).
- **OTP codes hashed at rest when SMTP is configured**; demo mode keeps them
  plaintext because the UI displays the code directly.
- **Password reset enumeration fixed**: unknown emails receive the same
  generic response as known ones.
- **CSP shipped** as `Content-Security-Policy-Report-Only` by default;
  `CSP_ENFORCE=true` switches to enforced. Theme bootstrap script externalized
  (`static/js/theme-init.js`).
- **Session cookies Secure-by-default in production**; `HOST` defaults to
  loopback outside production instead of binding all interfaces.
- **New audit events**: `user.login_failed`, `user.account_locked`,
  `user.logout`, `user.password_reset`.

### Reliability
- **SQLite WAL mode** + `busy_timeout=5000`, `foreign_keys=ON`,
  `synchronous=NORMAL` on every connection: readers no longer block the
  writer and concurrent writes queue instead of erroring.
- **Process-wide write lock** serializes whole-state transactions; an
  8-thread concurrent-registration smoke test is part of the suite.
- **Auth/watchlist/portfolio writes skip the prediction-cache table**
  (`persist(write_cache=False)`), so logins no longer rewrite the largest
  shared table.

### Access control
- **Job ownership**: background analyses are scoped to the submitting user
  across memory and on-disk artifacts (`{"owner", "result"}` payload with
  backward-compatible reads of legacy files). Deduplication is per-owner.

### Observability
- **`/metrics` Prometheus endpoint** (opt-in `METRICS_ENABLED`, optional
  `METRICS_TOKEN`): request counters + latency sum/count.
- **JSON logging** option (`LOG_FORMAT=json`) for log shippers.

### Packaging / DevOps
- Pinned `requirements.txt` to the tested environment; added
  `requirements-dev.txt`.
- Hardened multi-stage Dockerfile (non-root, healthcheck, libgomp1) +
  `.dockerignore`; `wsgi.py` eventlet-safe bootstrap; `gunicorn.conf.py`
  refuses >1 worker without Redis (in-process state would diverge).
- CI workflow (lint → tests → pip-audit → docker build/smoke).
- `.gitignore` covers venvs/backtest artifacts/DBs.

### Accessibility / UI
- Skip-to-content link, `<main id="main">`, polite live regions for flashes
  and toasts, visible focus outlines, reduced-motion support.

### Tests
- New suite `tests/test_upgrade_hardening.py` (21 tests): IP trust matrix,
  forged-XFF bucket sharing, CSP modes, production defaults, job scoping,
  hashed tokens/OTPs, audit events, WAL, concurrency smoke, metrics,
  JSON logs. Full suite: **147 tests green**.

### Known limitations
- Monolith shim (`app.py` → package services, ticket H-01) intentionally
  deferred: XL risk vs. value now that both entry points share one runtime.
- Multi-worker scaling requires the Redis-backed rate-limit/cache layer
  (backlog H-04); `gunicorn.conf.py` guards against unsafe fan-out today.
- CSP enforcement stays report-only until the four remaining templates'
  inline scripts (compare/portfolio/results/watchlist) are externalized.
