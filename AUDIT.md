# StockPredictorApp Audit — UI/UX, Performance & Silent-Failure Pass

Scope: `stockpredictor/` live Flask app (NOT the legacy `app.py` monolith).
Date: 2026-09-20. Tags: **Critical / Moderate / Minor** | Status: Fixed / Pending / Info.

---

## Critical

### C1. Frontend run-analysis polling never stopped on failure — infinite `setInterval`
- **Where:** `stockpredictor/static/js/app.js` `setupAnalysisPolling()`
- **Was:** bare `setInterval(...)` polling `/api/analysis/status/` with an empty `.catch(() => {})` on every tick. If the job errored or the request failed, the interval kept polling forever (each attempt hitting the DB/status endpoint), and the corresponding toast/message was never shown. Error text was also generic (`Request failed`).
- **Fix (implemented):** Rewrote the poller with:
  - `MAX_POLL_MS = 5 * 60 * 1000` hard timeout → auto-stop + message.
  - `MAX_ERROR_STRIKES = 5` — 5 consecutive request failures abort polling.
  - Clean `stop()` / `failMessage()` helpers; interval cleared on every terminal path.
  - `401/403` → redirect to `/login` immediately.
  - `429` → back off one tick (no strike).
  - Statuses `complete`, `error`, `missing` handled explicitly; non-2xx statuses surface `errorMessage(...)` payload.
- **Status:** Fixed & verified in working copy (root `app.py` legacy copy still has the old loop; it is not deployed and was not touched).

### C2. All non-2xx fetch responses collapsed into a generic message
- **Where:** `stockpredictor/static/js/app.js` `request()` helper.
- **Was:** every failure returned only the literal string `Request failed`; response body (message/detail) was discarded, hiding real errors (e.g. "Symbol not found").
- **Fix (implemented):** Added `errorMessage(status, payload)`:
  - Prefers `payload.message` → `payload.error` → `payload.detail`.
  - Falls back by status: `401/403` → "Your session has expired. Please sign in again."; `404` → "…not found"; `429` → "Too many requests…"; `500` → server error; `>=400` → generic with code.
- **Status:** Fixed.

---

## Moderate

### M1. ~1MB `echarts.min.js` loaded on every page
- **Where:** `stockpredictor/templates/base.html` (global) via `siblings/GLOBAL_VENDOR_JS`.
- **Was:** served on all pages incl. login/analyze/dashboard which never draw ECharts.
- **Fix (implemented):** Removed global include; added `{% block vendor_scripts %}` injected into `results.html`, `demo_results.html`, `compare.html` (guarded `{% if compare %}`), `portfolio.html` (guarded `{% if rows %}`). Chart canvas ids (`forecastChart`, `priceChart`, `compareChart`, `allocationChart`, `positionPnlChart`, `equityChart`) all live inside those pages only.
- **Status:** Fixed.

### M2. Chart pages still waited on 401/403 polling before redirecting
- **Where:** `app.js` `setupAnalysisPolling()`.
- **Fix:** covered by C1 (immediate `window.location.href = '/login'` on 401/403). XHR-level redirect handling for window-wide session expiry stays as-is (existing behaviour, avoid rework).
- **Status:** Fixed under C1.

### M3. Portfolio actions allowed double-submit
- **Where:** `stockpredictor/templates/portfolio.html` `trade()`, `cashMove()`, `optimizePortfolio()`.
- **Fix (implemented):** Buttons disabled while a request is in flight (`addedClass('disabled')` / re-enable in `.then` and `.catch`); button was already `type="button"` (no accidental form submit).
- **Status:** Fixed.

### M4. Actionable empty states were missing
- **Where:**
  - `portfolio.html` — empty portfolio rendered just an insight bar with no next step. **Fixed:** added an empty-state CTA linking to `analyze_any_stock` with a prefilled symbol AND the `Portfolio` help link.
  - `watchlist.html` — empty state guidance already present (link to analyze). **Confirmed present; no change.**
- **Status:** Fixed (portfolio); info (watchlist).

### M5. Symbol-search / filter inputs not labelled (a11y)
- **Where:** `compare.html` `#compareSymbols`, `watchlist.html` `#newSymbol`, `#watchlistFilter`.
- **Fix (implemented):** Added `.sr-only` labels for all three inputs; `#newSymbol` now `autocapitalize="characters" spellcheck="false"`.
- **Status:** Fixed.

### M6. Repeated analysis produces repeated network request stalls on wider screens
- **Where:** portfolio vs dashboard "quick" helpers were duplicated.
- **Info:** Identified during audit but refactor deferred (presentation identical; risk of touching working logic late in the pass). Tracked, not fixed this pass.

### M7. Frontend silently swallowed fetch failures on the watchlist/alert/trade flows
- **Where:** `app.js` — `addStock()`, `removeStock()`, `runAlert()`, alert CRUD, `optimizePortfolio()`, portfolio trade/cash.
- **Fix (implemented):** All such handlers were already updating via `StockPredictorAPI`; with the C2 change every rejected request now runs `showToast(err.message, 'error')` in `.catch` (several already did). `addStock()` additionally disables its button in flight and re-enables on failure.
- **Status:** Fixed (covered by C2); `addStock()` hardening added.

### M8. Backend silent-failure spots (no log on degradation / simulation path)
- **Where / Fix (implemented):**
  - `stockpredictor/views/api.py` — search + `/api/market-ticker` warnings.
  - `stockpredictor/views/dashboard.py` — `_fetch_live` debug logs + indices-fetch warning.
  - `stockpredictor/views/portfolio.py` — `_resolve_price` warning; `_fetch_position` debug logs.
  - `stockpredictor/views/watchlist.py` — `_fetch_quote` debug log.
  - `stockpredictor/views/compare.py` — `_collect` module + warning/logging for per-route collection.
  - `model_utils.py` — `get_current_real_price` (line ~3250) logs warning when falling back to `_get_intelligent_estimation()`; `get_stock_data` (line ~934) logs warning when falling back to `_create_enhanced_realistic_data()` simulated data.
- **Status:** Fixed.

### M9. Calculations re-computed on every request
- **Where / Fix (implemented)** (all via `production_core.cached`, JSON-safe, TTL-backed):
  - `stockpredictor/services/stocks.py` `get_company_fundamentals` → `cached("fundamentals:<SYM>", ttl=3600)`.
  - `stockpredictor/services/relative_strength.py` `compute_rs_table` → `cached("rs:<BENCH>:<sym1.sym2...>", ttl=600)` with parallel scoring.
  - `stockpredictor/services/explain.py` `explain_forecast` → `cached("explain:<SYM>", ttl=600)`.
  - `stockpredictor/services/analytics.py` `market_overview` → `cached("analytics:market-overview", ttl=1800)`.
  - `stockpredictor/services/realtime.py` `collect_quotes` → `ThreadPoolExecutor` (max 6) collecting quotes in parallel (was serial).
- **Status:** Fixed.

### M10. Watchlist "realtime updates" never stop when leaving the page
- **Info:** `watchlist.html` starts `startRealtime()` (10s interval); no `visibilitychange`/`beforeunload` stop. Identified; page-nav kills the tab so leak is bounded; deferred to a future pass (would also need server-sent stop signal).
- **Status:** Info (not fixed this pass).

### M11. Locally-managed static assets blocked browser caching for 0s
- **Where:** `stockpredictor/config.py`.
- **Fix (implemented):** Added `SEND_FILE_MAX_AGE_DEFAULT = STATIC_MAX_AGE` env (default 300s) for unversioned assets; content still served fresh on deploy via query-stringed `?v=` on key assets if `STATIC_VERSIONED` env set.
- **Status:** Fixed.

### M12. LLM client calls had no network timeout
- **Where:** `stockpredictor/services/llm_report.py`.
- **Fix (implemented):** `OpenAI(..., timeout=25)` / `Anthropic(..., timeout=25)`.
- **Status:** Fixed.

---

## Minor

### m1. Mobile touch targets <44px and 600px tables overflow
- **Where:** `app.css`.
- **Fix (implemented):** `@media (max-width:600px)` → `.stock-table { min-width:520px }` (horizontal scroll instead of squeeze) + `.btn-sm { min-height:40px }`. New `@media (max-width:420px)` block collapses stats/grid to one column, stacks `.action-row`/`.inline-form`, shrinks `.chart-wrap` height + hero buttons full-width.
- **Status:** Fixed.

### m2. Call-to-action / metric-card markup duplication
- **Where:** `app.css`.
- **Fix (implemented):** `.action-row` (metrics row header w/ trailing action), `.section-h3`, `sr-only`, `.report-pre`, `.preset` (link-styled button); applied to results/demo_results/dashboard/portfolio; compare preset links → real `<button>`s.
- **Status:** Fixed.

### m3. Results confidence metric always styled "positive-up" (hard-coded class)
- **Where:** `results.html`.
- **Fix (implemented):** Conditional `s-mini-up`/`s-mini-down`/`s-mini-gold` by sign on `expected_return`. Also applied `mini-fill` modifier classes in the portfolio insight bar (was inline `background`) → `.s-mini-up/gold/down`.
- **Status:** Fixed.

### m4. Accent swatch buttons had no accessible name
- **Where:** `base.html`.
- **Fix (implemented):** `aria-label="Set accent color to …"` on all 6 swatches; focus-ring/title present.
- **Status:** Fixed.

### m5. `analyze.html` combobox
- **Info:** `#symbol` input already `role="combobox"` + `aria-controls` + sourced instant hint from `#symbolHint`; LIGHTHOUSE "form without label" n/a. No change needed.
- **Status:** Confirmed OK.

### m6. Panic-mode suggestions list duplication
- **Where:** `compute_portfolio_ml` output still has `top/all_picks` variants; `suggestions` sorted stable.
- **Info:** no behavioural bug; consolidation deferred (ML weight mapping must stay intact).
- **Status:** Info.

---

## Pre-existing notes (not from this pass)

- **Flaky test:** `tests/test_upgrade_hardening.py::ConcurrentRegistrationTests.test_parallel_registrations_all_persist` intermittently fails with `RuntimeError: dictionary changed size during iteration` on 8-thread concurrent `start_otp_registration`/`complete_otp_registration`. Runs OK in isolation. Touches `stockpredictor/services/auth.py` (untouched this pass). Recommends a `Lock`/`RLock` around the `users_db` mutation in a future hardening pass.
- **Disk-cache subtlety:** `production_core.cached()` writes JSON to disk best-effort (swallows failures). Payloads containing non-JSON-serializable keys (e.g. `pandas.Timestamp` columns) fall back to memory-only caching — correct, not an error.

## Regression

- `python -m unittest tests.test_demo_cache tests.test_stockpredictor tests.test_new_features tests.test_upgrade_hardening tests.test_scheduler` → 115 ran, only the pre-existing flaky concurrency test failed; all others green. `py_compile` clean on all edited modules.