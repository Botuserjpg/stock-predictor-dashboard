import ipaddress
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from flask import g, has_request_context, jsonify, request


BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "runtime_state"
AUDIT_DIR = BASE_DIR / "audit_logs"
SECRET_FILE = STATE_DIR / "flask_secret.key"
STATE_FILE = STATE_DIR / "app_state.json"

STATE_VERSION = 1
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


def _parse_trusted_proxies(raw: str) -> tuple:
    """Parse a comma-separated list of trusted proxy IPs/CIDRs (empty by default)."""
    networks = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            networks.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            continue
    return tuple(networks)


# Only requests whose *direct peer* is listed here may influence the client IP
# via X-Forwarded-For. Empty by default so a client can never spoof its address.
TRUSTED_PROXIES = _parse_trusted_proxies(os.getenv("TRUSTED_PROXIES", ""))


def client_addr() -> str:
    """Best-effort real client IP for rate limiting and audit records.

    ``X-Forwarded-For`` is honoured only when the direct peer is a configured
    trusted proxy; otherwise the peer address is used verbatim so a client
    cannot forge its IP (and rotate rate-limit buckets) by setting the header.
    The rightmost non-trusted hop wins (standard proxy-chain semantics).
    """
    remote = request.remote_addr or "unknown"
    if not TRUSTED_PROXIES:
        return remote
    try:
        peer = ipaddress.ip_address(remote)
    except ValueError:
        return remote
    if not any(peer in network for network in TRUSTED_PROXIES):
        return remote

    forwarded = request.headers.get("X-Forwarded-For", "")
    hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
    for hop in reversed(hops):
        try:
            hop_ip = ipaddress.ip_address(hop)
        except ValueError:
            return remote  # malformed chain: fall back to the peer we trust
        if any(hop_ip in network for network in TRUSTED_PROXIES):
            continue
        return str(hop_ip)
    # Every hop was trusted (or the header is empty) — use what the proxy gave us.
    return str(hops[0]) if hops else remote


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_logging() -> logging.Logger:
    """Configure structured-ish application logging without replacing existing handlers.

    Set ``LOG_FORMAT=json`` for one-JSON-object-per-line output (log shippers);
    the default stays the human-readable request-id text format.
    """
    AUDIT_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("stock_predictor")
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    if not logger.handlers:
        if os.getenv("LOG_FORMAT", "").strip().lower() == "json":
            formatter: logging.Formatter = JsonLogFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"
            )
        handler = logging.FileHandler(BASE_DIR / "stock_predictor_app.log", encoding="utf-8")
        handler.setFormatter(formatter)
        handler.addFilter(RequestIdFilter())
        logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.addFilter(RequestIdFilter())
        logger.addHandler(console)

    return logger


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(g, "request_id", "-") if has_request_context() else "-"
        return True


class JsonLogFormatter(logging.Formatter):
    """Minimal JSON-lines formatter (no extra dependency)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_secret_key() -> str:
    """Load a stable secret from env or local runtime state."""
    env_secret = os.getenv("SECRET_KEY") or os.getenv("FLASK_SECRET_KEY")
    if env_secret:
        return env_secret

    STATE_DIR.mkdir(exist_ok=True)
    if SECRET_FILE.exists():
        return SECRET_FILE.read_text(encoding="utf-8").strip()

    import secrets
    import stat

    secret = secrets.token_urlsafe(48)
    SECRET_FILE.write_text(secret, encoding="utf-8")
    try:
        SECRET_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return secret


def state_db_path() -> Path:
    """Absolute path of the SQLite state database (inside STATE_DIR)."""
    return STATE_DIR / "app_state.db"


def load_app_state() -> Dict[str, Any]:
    """Load the application state, hydrating it from the SQLite database.

    The legacy ``runtime_state/app_state.json`` blob is imported once into the
    relational store on the first run, then kept as a read-only artifact. State
    resolution order: existing database, then legacy JSON, then a fresh default.
    """
    STATE_DIR.mkdir(exist_ok=True)
    try:
        from stockpredictor.database import db_path, hydrate_state, init_db

        if db_path().exists():
            init_db()
            return hydrate_state()
    except Exception:
        pass

    if STATE_FILE.exists():
        state = _load_legacy_state_file()
        try:
            from stockpredictor.database import persist_state

            persist_state(state)
        except Exception:
            pass
        return state

    try:
        from stockpredictor.database import hydrate_state

        return hydrate_state()
    except Exception:
        return _default_state()


def _default_state() -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "users_db": {},
        "user_watchlists": {},
        "prediction_cache": {},
        "updated_at": utc_now(),
    }


def _load_legacy_state_file() -> Dict[str, Any]:
    """Parse the legacy JSON state, backing up corrupt files like before."""
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("users_db", {})
        state.setdefault("user_watchlists", {})
        state.setdefault("prediction_cache", {})
        state.setdefault("version", STATE_VERSION)
        return state
    except Exception:
        backup = STATE_FILE.with_suffix(f".corrupt.{int(time.time())}.json")
        STATE_FILE.replace(backup)
        return _default_state()


def save_app_state(
    users_db: Dict[str, Any],
    user_watchlists: Dict[str, Any],
    prediction_cache: Dict[str, Any],
    user_portfolios: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    write_cache: bool = True,
) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    payload = {
        "version": STATE_VERSION,
        "users_db": users_db,
        "user_watchlists": user_watchlists,
        "prediction_cache": prediction_cache,
        "user_portfolios": _serialize_portfolios(user_portfolios or {}),
        "updated_at": utc_now(),
    }
    for key, value in (extra_state or {}).items():
        payload[key] = value

    from stockpredictor.database import persist_state

    persist_state(payload, write_cache=write_cache)


# Serializes whole-state writes within this process so two threads can never
# interleave the delete+insert transaction (SQLite handles cross-process
# contention via WAL + busy_timeout). Combined with the single-writer policy
# documented for production deployments this removes lost-update corruption.
_STATE_WRITE_LOCK = threading.RLock()


def state_write_lock() -> threading.RLock:
    """Process-wide lock guarding read-modify-persist critical sections."""
    return _STATE_WRITE_LOCK


def _serialize_portfolios(user_portfolios: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize portfolio payloads (dict or object form) for persistence.

    The refactored package stores ``Portfolio.to_dict()`` maps (keys
    ``cash``/``positions``/``history``); the legacy app used object attributes
    (``cash_balance``/``holdings``/``transactions``). Both are accepted so state
    written by either entry point survives.
    """
    serialized = {}
    for user_id, portfolio in user_portfolios.items():
        if isinstance(portfolio, dict):
            serialized[user_id] = {
                "cash": portfolio.get("cash", portfolio.get("cash_balance", 10000.0)),
                "positions": portfolio.get("positions", portfolio.get("holdings", {})),
                "history": portfolio.get("history", portfolio.get("transactions", [])),
                "created_at": portfolio.get("created_at", utc_now()),
            }
        else:
            serialized[user_id] = {
                "cash": getattr(portfolio, "cash", getattr(portfolio, "cash_balance", 10000.0)),
                "positions": getattr(portfolio, "positions", getattr(portfolio, "holdings", {})),
                "history": getattr(portfolio, "history", getattr(portfolio, "transactions", [])),
                "created_at": getattr(portfolio, "created_at", utc_now()),
            }
    return serialized


def sanitize_symbol(raw_symbol: str) -> str:
    symbol = (raw_symbol or "").strip().upper()
    if not SYMBOL_PATTERN.match(symbol):
        raise ValueError("Invalid symbol format")
    return symbol


def validate_positive_number(value: Any, field_name: str, min_value: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be numeric")
    if parsed <= min_value:
        raise ValueError(f"{field_name} must be greater than {min_value}")
    return parsed


def audit_log(event: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    AUDIT_DIR.mkdir(exist_ok=True)
    client_ip = None
    request_id = None
    if has_request_context():
        client_ip = client_addr()
        request_id = getattr(g, "request_id", None)

    entry = {
        "timestamp": utc_now(),
        "event": event,
        "user_id": user_id,
        "request_id": request_id,
        "ip": client_ip,
        "metadata": metadata or {},
    }
    path = AUDIT_DIR / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: Dict[str, deque] = defaultdict(deque)

    def check(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.time()
        events = self._events[key]
        while events and events[0] <= now - window_seconds:
            events.popleft()
        if len(events) >= limit:
            return False
        events.append(now)
        return True


rate_limiter = InMemoryRateLimiter()


# ---------------------------------------------------------------------------
# Lightweight Prometheus-style metrics (opt-in via METRICS_ENABLED=true).
# Counters are process-local; the multi-process story is documented in README.
# ---------------------------------------------------------------------------
class RequestMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total: Counter = Counter()
        self._latency_sum = 0.0
        self._latency_count = 0

    def observe(self, endpoint: str, method: str, status: int, duration: float) -> None:
        with self._lock:
            self._total[(endpoint, method, status)] += 1
            self._latency_sum += duration
            self._latency_count += 1

    def render(self) -> str:
        lines = [
            "# HELP spp_requests_total Total HTTP requests.",
            "# TYPE spp_requests_total counter",
        ]
        with self._lock:
            rows = sorted(self._total.items())
            total_sum = self._latency_sum
            total_count = self._latency_count
        for (endpoint, method, status), count in rows:
            safe_endpoint = re.sub(r"[^A-Za-z0-9_]", "_", str(endpoint or "unknown"))
            lines.append(
                f'spp_requests_total{{endpoint="{safe_endpoint}",method="{method}",status="{status}"}} {count}'
            )
        lines.append("# HELP spp_request_latency_seconds_sum Cumulative request latency.")
        lines.append("# TYPE spp_request_latency_seconds_sum counter")
        lines.append(f"spp_request_latency_seconds_sum {total_sum:.6f}")
        lines.append("# TYPE spp_request_latency_seconds_count counter")
        lines.append(f"spp_request_latency_seconds_count {total_count}")
        return "\n".join(lines) + "\n"


request_metrics = RequestMetrics()


class TTLCache:
    """Thread-safe in-memory cache with time-based expiry.

    Falls back to a cross-process file cache for values that outlive the
    process (used by :func:`cached`).
    """

    def __init__(self) -> None:
        self._store: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._lock:
            self._store[key] = (value, time.time() + ttl_seconds)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)


ttl_cache = TTLCache()

# File-cache directory (survives restarts)
CACHE_DIR = BASE_DIR / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def cached(key: str, ttl: int = 300, producer: Optional[Callable] = None):
    """Decorator-ish helper that caches a callable's result for ``ttl`` seconds.

    Usage::

        value = cached("my:key", ttl=300, producer=my_fn)(*args, **kwargs)

    or as a plain function::

        @cached("my:key", ttl=60)
        def compute(...):
            return expensive_work(...)

    Memory-first; serializes JSON-safe payloads to disk so results survive
    restarts and are shared between worker processes.
    """
    file_key = re.sub(r"[^A-Za-z0-9_.-]", "_", key)

    def _wrap(fn: Callable, *args, **kwargs):
        memory_hit = ttl_cache.get(key)
        if memory_hit is not None:
            return memory_hit

        file_path = CACHE_DIR / f"{file_key}.json"
        if file_path.exists():
            try:
                age = time.time() - file_path.stat().st_mtime
                if age <= ttl:
                    with file_path.open("r", encoding="utf-8") as fh:
                        return json.load(fh)
            except Exception:
                pass

        value = fn(*args, **kwargs)
        if value is not None:
            ttl_cache.set(key, value, ttl)
            try:
                tmp = file_path.with_suffix(".tmp")
                with tmp.open("w", encoding="utf-8") as fh:
                    json.dump(value, fh, default=str)
                tmp.replace(file_path)
            except Exception:
                pass
        return value

    if producer is not None:
        return lambda *a, **kw: _wrap(producer, *a, **kw)
    return _wrap


def peek_cached(key: str, ttl: Optional[int] = None):
    """Return a cached value without running its producer, else None.

    ``ttl`` overrides the default TTL used when the entry was stored (used to
    check "is there a fresh enough forecast I can serve instantly?").
    """
    memory_hit = ttl_cache.get(key)
    if memory_hit is not None:
        return memory_hit

    file_key = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    file_path = CACHE_DIR / f"{file_key}.json"
    if file_path.exists():
        try:
            age = time.time() - file_path.stat().st_mtime
            if age <= (ttl if ttl is not None else 10 ** 9):
                with file_path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception:
            return None
    return None


def invalidate_cache(key: str) -> None:
    """Drop a single cache entry (memory + file)."""
    ttl_cache.delete(key)
    file_key = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    file_path = CACHE_DIR / f"{file_key}.json"
    try:
        file_path.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CSRF protection (lightweight, dependency-free via itsdangerous)
# ---------------------------------------------------------------------------
CSRF_MAX_AGE = 60 * 60  # 1 hour
CSRF_SKIP_PATHS = {"/health", "/ready"}


def generate_csrf_token() -> str:
    """Return (and lazily create) a signed CSRF token for the session."""
    from flask import session as flask_session

    token = flask_session.get("_csrf_token")
    if token is None:
        from itsdangerous import URLSafeTimedSerializer

        serializer = URLSafeTimedSerializer(get_secret_key(), salt="csrf")
        token = serializer.dumps({"sid": uuid.uuid4().hex})
        flask_session["_csrf_token"] = token
    return token


def validate_csrf_token(token: Optional[str]) -> bool:
    if not token:
        return False
    try:
        from flask import session as flask_session
        from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

        expected = flask_session.get("_csrf_token")
        if not expected or not hmac_compare(expected, token):
            return False
        serializer = URLSafeTimedSerializer(get_secret_key(), salt="csrf")
        serializer.loads(token, max_age=CSRF_MAX_AGE)
        return True
    except (BadSignature, SignatureExpired, Exception):
        return False


def _inject_csrf_into_html_response(response) -> "flask.Response":
    """Embed the session CSRF token into HTML pages that lack a form field.

    Legacy inline-HTML views (``app.py``) don't render the ``csrf_token()``
    template helper, so splash a meta tag plus a tiny auto-submitting helper:
    the meta value feeds forms and ``fetch()`` calls so the CSRF guard accepts
    their POST/PUT/DELETE requests. Views that already render their own token
    submit the same session-bound value, so this is harmless there too.
    """
    ct = response.content_type or ""
    charset = getattr(response, "charset", None) or "utf-8"
    if response.status_code != 200 or not ct.startswith("text/html"):
        return response
    try:
        text = response.get_data(as_text=True)
    except (UnicodeDecodeError, TypeError):
        return response
    lower = text.lower()
    if "</body>" not in lower or "</head>" not in lower:
        return response

    token = generate_csrf_token()
    meta = '<meta name="csrf-token" content="%s">' % token
    script = (
        "<script>(function(){"
        "var m=document.querySelector('meta[name=csrf-token]');"
        "var token=m?m.content:'';"
        "document.addEventListener('submit',function(e){var f=e.target;"
        "if(f&&f.tagName==='FORM'&&!f.querySelector('[name=csrf_token]')){"
        "var i=document.createElement('input');i.type='hidden';i.name='csrf_token';i.value=token;"
        "f.appendChild(i);}});"
        "if(window.fetch){var f=window.fetch;window.fetch=function(u,o){o=o||{};"
        "var meth=(o.method||(u&&u.method)||'GET').toUpperCase();"
        "if(meth!=='GET'&&meth!=='HEAD'){o.headers=new Headers(o.headers||{});"
        "if(!o.headers.has('X-CSRFToken')){o.headers.set('X-CSRFToken',token);}}"
        "return f.call(this,u,o);};}});</script>"
    )

    at = lower.index("</head>")
    head = text[:at]
    tail = text[at:].strip()
    tail = tail.replace("</head>", meta + "</head>", 1)
    tail = tail.replace("</body>", script + "</body>", 1)
    response.set_data((head + tail).encode(charset))
    return response


def hmac_compare(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def install_request_guards(app, logger: logging.Logger) -> None:
    from flask import session as flask_session

    for handler in logger.handlers:
        handler.addFilter(RequestIdFilter())

    @app.before_request
    def _request_guard():
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        remote_addr = client_addr()
        g.request_started_at = time.perf_counter()
        route = request.endpoint or request.path
        # Blueprint endpoints are namespaced ("auth.login"); the legacy monolith
        # exposes bare names ("login"). Match on the trailing segment.
        route_name = route.rsplit(".", 1)[-1]

        # Rate limiting is a production concern; unit-test suites legitimately
        # hammer auth endpoints far beyond sane per-IP budgets.
        if not app.config.get("TESTING"):
            if route_name in {"login", "register", "forgot_password", "reset_password",
                              "verify_otp", "resend_otp"}:
                if not rate_limiter.check(f"auth:{remote_addr}", limit=20, window_seconds=300):
                    audit_log("rate_limit.auth", metadata={"route": route})
                    return jsonify({"success": False, "message": "Too many authentication attempts"}), 429

            if request.path.startswith("/api/"):
                if not rate_limiter.check(f"api:{remote_addr}", limit=240, window_seconds=60):
                    audit_log("rate_limit.api", metadata={"path": request.path})
                    return jsonify({"success": False, "message": "Rate limit exceeded"}), 429

        # Enforce CSRF on state-changing requests (except whitelisted paths).
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.path not in CSRF_SKIP_PATHS:
            submitted = request.headers.get("X-CSRFToken")
            if submitted is None and request.is_json:
                submitted = (request.get_json(silent=True) or {}).get("csrf_token")
            if submitted is None:
                submitted = request.form.get("csrf_token")
            if not validate_csrf_token(submitted):
                audit_log("security.csrf_rejected", metadata={"route": route, "method": request.method})
                return jsonify({"success": False, "message": "Invalid or missing CSRF token"}), 400

    @app.after_request
    def _security_headers(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", "-")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        response.headers.setdefault("Cache-Control", "no-store" if request.path.startswith("/api/") else "private, max-age=300")

        csp_policy = app.config.get("CSP_POLICY")
        if csp_policy:
            if app.config.get("CSP_ENFORCE"):
                response.headers.setdefault("Content-Security-Policy", csp_policy)
            else:
                # Report-only by default: pages still carry legacy inline scripts,
                # so violations are observable without breaking anything. Flip
                # CSP_ENFORCE=true once all inline scripts are externalized.
                response.headers.setdefault("Content-Security-Policy-Report-Only", csp_policy)

        started_at = getattr(g, "request_started_at", None)
        if app.config.get("METRICS_ENABLED") and started_at is not None and request.path != "/metrics":
            try:
                request_metrics.observe(
                    request.endpoint or "unknown",
                    request.method,
                    response.status_code,
                    time.perf_counter() - started_at,
                )
            except Exception:
                pass
        return _inject_csrf_into_html_response(response)


def json_endpoint(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            return jsonify({"success": False, "message": str(exc)}), 400
        except Exception as exc:
            audit_log("api.error", metadata={"endpoint": request.endpoint, "error": str(exc)})
            return jsonify({"success": False, "message": "Internal server error"}), 500

    return wrapper


def runtime_health() -> Dict[str, Any]:
    models_dir = BASE_DIR / "models"
    cache_dir = BASE_DIR / "data_cache"
    return {
        "status": "ok",
        "timestamp": utc_now(),
        "state_file_exists": state_db_path().exists(),
        "model_count": len(list(models_dir.glob("*.h5"))) if models_dir.exists() else 0,
        "cached_dataset_count": len(list(cache_dir.glob("*.csv"))) if cache_dir.exists() else 0,
        "python_env": os.getenv("FLASK_ENV", "production"),
    }
