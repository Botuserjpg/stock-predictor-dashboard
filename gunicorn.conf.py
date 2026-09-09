"""Gunicorn configuration for Stock Predictor Pro.

Socket.IO requires an async worker (eventlet) and, until the Redis-backed
rate-limit/cache layer lands (backlog H-04), exactly ONE worker process:
in-process rate limiting, TTL caches and job state are per-process. The guard
below fails fast instead of silently degrading security under >1 worker.
"""
import os

# Render and Railway inject PORT at runtime. Respect it unless BIND is explicit.
bind = os.getenv("BIND", f"0.0.0.0:{os.getenv('PORT', '5000')}")
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "eventlet")

_workers = int(os.getenv("WEB_CONCURRENCY", "1"))
if _workers > 1 and not os.getenv("REDIS_URL"):
    raise SystemExit(
        "Refusing to start with WEB_CONCURRENCY>1 without REDIS_URL: "
        "in-process rate limits/caches/jobs are not shared across workers. "
        "Set WEB_CONCURRENCY=1 for now or configure Redis (see README)."
    )
workers = max(1, _workers)

accesslog = "-"
errorlog = "-"
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
loglevel = os.getenv("LOG_LEVEL", "info").lower()
forwarded_allow_ips = os.getenv("TRUSTED_PROXIES", "") or None
