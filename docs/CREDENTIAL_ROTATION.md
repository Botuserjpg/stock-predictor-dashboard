# Credential Rotation Runbook

The `.env` file in this deployment has existed in working directories that were
copied, zipped and shared during development. **Assume every credential inside
it has been exposed** and rotate all of them. None of the application code
needs to change; only `.env` values (and provider dashboards) are touched.

## Rotation order (highest blast radius first)

| # | Credential | Where it lives in `.env` | Rotate at |
|---|------------|--------------------------|-----------|
| 1 | Flask secret key | `SECRET_KEY` | Generated value — see note below |
| 2 | Gmail app password | `SMTP_PASSWORD` | Google Account → Security → App passwords (revoke old) |
| 3 | Reddit OAuth | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD` | reddit.com/prefs/apps + account password |
| 4 | Twitter/X bearer | `TWITTER_BEARER_TOKEN` | developer.twitter.com → regenerate |
| 5 | Market data keys | `NEWSAPI_KEY`, `ALPHAVANTAGE_KEY`, `FINNHUB_KEY`, `FMP_KEY`, `POLYGON_KEY`, `TIINGO_KEY` | Each provider's dashboard |

## `SECRET_KEY` specifics

Rotating it invalidates: session cookies, remember-me cookies, signed CSRF
tokens. Users are simply logged out; nothing else breaks.

1. Generate a fresh value:
   ```
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```
2. Replace `SECRET_KEY=` in `.env`.
3. Restart the app. If no `SECRET_KEY` is set at all, the app auto-generates
   one into `runtime_state/flask_secret.key` — delete that file too when
   rotating so a new one is minted.

## Verification after rotation

1. `python wsgi.py` starts cleanly; `/health` returns `{"status": "ok"}`.
2. Register a throwaway account with real SMTP configured — the OTP email arrives.
3. News/sentiment features load without provider 401/403s.
4. Old cookies are rejected (fresh login required).

## Prevention

- `.env` is git-ignored and docker-ignored; never bake it into images.
- Distribute deployments as images + a secret store (or at minimum a fresh
  `.env` per host), never as directory copies.
