# Production Upgrade Roadmap

## Immediate
- ✅ Stabilize application configuration, secret-key handling, request IDs, security headers, and rate limiting.
- ✅ Persist users, watchlists, and prediction cache outside process memory.
- ✅ Add strict symbol and portfolio input validation to public APIs.
- ✅ Add data-quality reports, missing-value cleanup, outlier handling, and advanced feature generation.
- ✅ Add model registry records for trained models and attach feature/data fingerprints.

## High Impact
- ✅ Split the monolithic Flask app into blueprints: auth, analysis, analytics, watchlist, portfolio, API, admin.
- ⏳ Replace JSON state with relational storage using SQLAlchemy migrations.
- ⏳ Move slow yfinance/model training work to background jobs with durable task state.
- ⏳ Add experiment tracking, scheduled retraining, drift-triggered retraining, and model promotion rules.
- ✅ Add a tested service layer for forecasting, portfolio risk, alerts, and report generation.

## Medium Impact
- ⏳ Add SHAP explanations for tree/tabular baselines and integrated-gradient style explanations for neural forecasts.
- ✅ Add probabilistic forecasts to UI charts with fan intervals (Low/High bands + shaded chart range).
- ✅ Add admin role, audit-log viewer, and user lockout controls.
- ⏳ API key rotation.
- Add Redis-backed caching and rate limiting for multi-process deployment.
- Add OpenTelemetry traces, Prometheus metrics, and SLO dashboards.

## Future Enhancements
- Add broker-paper-trading integrations, webhooks, and notification channels.
- Add multi-asset support, factor models, portfolio optimization constraints, and tax-aware simulations.
- Add CI/CD deployment to cloud app services with managed database, object storage, and scheduled retraining workers.
- Add a model evaluation leaderboard and automated challenger/champion deployment.
