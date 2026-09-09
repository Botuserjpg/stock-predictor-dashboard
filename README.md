# Stock Predictor Pro

Machine-learning stock analysis application with Streamlit and Flask interfaces:
multi-model forecasting (LSTM/GRU/Prophet/ARIMA), technical indicators, sentiment
analysis, portfolio tracking, watchlists, and model-artifact caching.

> **Disclaimer:** This project is intended for research and education only.
> Forecasts and trading signals are **not financial advice**.

**Public recruiter demo:** `/demo` when `PUBLIC_DEMO=true`. It is read-only;
accounts and all write features require authentication.

## Tech Stack

| Layer | Technology |
|---|---|
| Forecasting | TensorFlow/Keras LSTM, GRU or hybrid; Prophet; ARIMA (pmdarima) |
| Data | yfinance (market data), NewsAPI/Finnhub/Alpha Vantage/FMP/Polygon/Tiingo (quotes & news) |
| Sentiment | Twitter/X API, Reddit OAuth, textblob VADER-style scoring |
| UI | Streamlit (`app_streamlit.py`, `predict.py`) and Flask (`app.py` / `stockpredictor`) |
| Web analytics | pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, plotly, seaborn |
| Ops | gunicorn + eventlet, APScheduler, SQLAlchemy + alembic, Docker |

## Verified Results

Leakage-audited, walk-forward backtest on **5 instruments** (AAPL, MSFT, TSLA,
GOOGL, RY.TO) with a 60-day lookback, 186 strictly out-of-sample trading days per
symbol (930 total), next-day-close horizon:

| Metric | Value |
|---|---|
| RMSE | 81.69 |
| MAPE | 22.99% |
| Directional accuracy | 47.63% |

The harness (`backtest_ml/run_backtest.py`) fingerprints the source modules, splits
chronologically (70/15/15), and refuses to scale on all rows — the test window is
never seen during training/validation. A determinism audit (repeat runs under a
fixed seed) produced identical metrics, confirming the results are reproducible.

## Quick Start (Local)

Requires **Python 3.10**.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in only the API keys you need. Every
provider is optional — blank keys disable that provider and the app degrades
gracefully where possible.

Streamlit app:

```powershell
streamlit run app_streamlit.py
```

Alternate Streamlit analysis app:

```powershell
streamlit run predict.py
```

Flask app (refactored package):

```powershell
python wsgi.py        # equivalent to python app.py
python run_app.py     # launcher menu
```

## Legacy Streamlit dashboards (not the live deployment)

Streamlit Cloud reads secrets from its dashboard, not local files. Configure them
under **Settings → Secrets** as key/value pairs. A ready-to-paste template lives at
[`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example).

1. Push this repo to GitHub (make sure `~/.streamlit/secrets.toml` copied as
   `.streamlit/secrets.toml` locally is **NOT** committed — it is git-ignored).
2. Go to <https://share.streamlit.io/> and create a new app from the repository,
   with **Main file path = `app_streamlit.py`** and **Python version = 3.10**.
3. Open **Settings → Secrets** and paste the contents of `.streamlit/secrets.toml`
   (from the example template) with your real key values.
4. Deploy. The entry point (`app_streamlit.py`) reads keys through
   `config.py`, which checks `st.secrets` first and falls back to environment
   variables, so the same code runs locally and on the cloud.

Notes for the cloud run:

- `tensorflow==2.15.0` is a large dependency; the first deploy can take several
  minutes to install. Keeping model training epochs short keeps cold starts fast.
- Runtime state (`runtime_state/`, `model_cache/`, etc.) is local to each app
  session — Streamlit Cloud is not persistent storage. The app recreates missing
  directories on demand.
- Live API keys (financial data, Twitter, Reddit, SMTP) must be entered in the
  Streamlit Cloud dashboard — never in source code. See the
  [security section](#security--credentials) below.

## Deploy the Flask app (Render or Railway)

```bash
docker build -t stock-predictor-pro .
docker run -d -p 5000:5000 --env-file .env stock-predictor-pro
```

The image runs `gunicorn` with the eventlet worker (required by Socket.IO) as a
non-root user with a `/health` healthcheck. Behind a reverse proxy set
`TRUSTED_PROXIES=<proxy-ip>` so rate limiting sees real client IPs.

> **Single worker today:** in-process rate limits, TTL caches and job state are
> per-process. `gunicorn.conf.py` refuses `WEB_CONCURRENCY > 1` unless `REDIS_URL`
> is configured.

Use the included `render.yaml` or `railway.json` when deploying from GitHub.
Configure these variables in the provider dashboard:

| Variable | Value |
|---|---|
| `FLASK_ENV` | `production` |
| `PUBLIC_DEMO` | `true` |
| `SECRET_KEY` | a long random value, stored as a provider secret |
| `SESSION_COOKIE_SECURE` | `true` |
| `WEB_CONCURRENCY` | `1` |
| `ENABLE_ALERT_SCHEDULER` | `false` |

The Docker command starts the Flask WSGI app (`gunicorn ... wsgi:app`), never a
Streamlit entry point. Render/Railway's injected `PORT` is honored automatically.
Keep API, SMTP, webhook, and metrics credentials out of Git and add them only as
provider secrets when a feature needs them.

## Security & Credentials

Never hardcode API keys, passwords, or personal accounts in source files.
Credentials are resolved, in order, from `st.secrets` (Streamlit Cloud), then
environment variables / `.env`.

- `config.py` centralizes API-key access (`APIConfig`) with `st.secrets` +
  `os.getenv` fallback.
- `.env.example` documents every variable; git-ignore `.env` and
  `.streamlit/secrets.toml` (both are already ignored).
- If a `.env` or secrets file was ever shared, rotate the affected credentials
  immediately (see `docs/CREDENTIAL_ROTATION.md`).

## Project Layout

- `app_streamlit.py` — primary Streamlit dashboard (deploy entry point).
- `predict.py` — alternate Streamlit analysis app.
- `predictor_core.py` — forecasting/model orchestration.
- `model_utils.py` — market data, indicators, sentiment, portfolio helpers.
- `ml_governance.py` — data validation, feature selection, model registry.
- `production_core.py` — runtime state, audit logging, request guards.
- `config.py` — centralized API-key configuration (`APIConfig`).
- `sentiment_analysis.py` — real-time news/social sentiment via APIs.
- `stockpredictor/` — refactored Flask application package (config, services, views).
- `app.py`, `wsgi.py` — Flask entry points.
- `backtest_ml/run_backtest.py` — walk-forward, leakage-audited backtest harness.
- `tests/` — fast unit tests.
- `docs/PRODUCTION_ROADMAP.md` — production hardening roadmap.

Runtime files such as logs, caches, generated reports, and trained models are
intentionally git-ignored.

## Feature Highlights

- **Multi-model forecasting:** LSTM, GRU, hybrid, Prophet, ARIMA, with market
  regime detection (ADX + realized volatility) driving automatic model selection.
- **Sentiment pipeline:** NewsAPI + Twitter/X + Reddit headlines scored with
  textblob, exposing News_Sentiment / Social_Buzz / Sentiment_Strength features.
- **Uncertainty quantification:** 90% probabilistic fan-interval bands and a Monte
  Carlo price simulator.
- **Risk engine:** position sizing (volatility targeting + Kelly), portfolio VaR,
  correlation matrix, and rebalance suggestions.
- **Backtesting + explainability:** walk-forward strategy backtests, SHAP feature
  importance, drift monitoring (PSI) with auto-retrain.
- **Streamlit UI:** dark/light theme, CSV export, analysis history, and portfolio
  simulator tools.
- **Flask app (secondary):** accounts with email OTP verification, watchlists,
  price alerts with Telegram/Discord push, admin console, Prometheus metrics, and
  a CSRF-hardened JSON API.

## Test

Fast local tests:

```powershell
python -m unittest discover -v
```

Live API and sentiment checks are skipped unless credentials are present:

```powershell
$env:RUN_EXTERNAL_TESTS = "1"
python -m unittest test_api_keys test_sentiment -v
```

## Known Limitations

- Streamlit Cloud is ephemeral; runtime state resets on redeploy.
- Single-process backtesting is CPU-bound on the free cloud tier.
- Template CSP stays report-only until all inline scripts are externalized
  (see `docs/PRODUCTION_ROADMAP.md`).

## License & Attribution

For research and educational use. No affiliation with, or endorsement by, any
data provider or exchange.
