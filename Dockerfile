# ---------- build stage: compile/install wheels ----------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------- runtime stage: no compilers, non-root, healthchecked ----------
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    HOST=0.0.0.0 \
    PORT=5000 \
    PATH="/opt/venv/bin:$PATH"

# libgomp1: runtime dep of the tensorflow/scipy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .
# Runtime caches and the local SQLite state are intentionally git-ignored, but
# must be creatable by the unprivileged process at startup.
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Refactored WSGI entry point (stockpredictor package) behind an async worker.
CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
