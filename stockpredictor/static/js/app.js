/* Stock Predictor Pro — UI behaviour.
 * fetch wrappers attach the CSRF token; every state-changing API call must
 * carry it (the server rejects POST/PUT/PATCH/DELETE without it). */
(function () {
    'use strict';

    /* ---------------- CSRF-aware fetch wrapper ---------------- */
    function csrfToken() {
        const meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? meta.getAttribute('content') : '';
    }

    async function request(url, options) {
        options = options || {};
        options.headers = options.headers || {};
        options.headers['X-CSRFToken'] = csrfToken();
        if (options.json) {
            options.headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(options.json);
            delete options.json;
        }
        if (options.form) {
            options.headers['Content-Type'] = 'application/x-www-form-urlencoded';
            const params = new URLSearchParams();
            Object.keys(options.form).forEach(function (key) {
                params.append(key, options.form[key]);
            });
            options.body = params.toString();
            delete options.form;
        }
        const response = await fetch(url, options);
        const contentType = response.headers.get('Content-Type') || '';
        let payload;
        if (contentType.indexOf('application/json') !== -1) {
            payload = await response.json();
        } else {
            payload = await response.text();
        }
        if (!response.ok) {
            const message = (payload && payload.message) || 'Request failed';
            const error = new Error(message);
            error.response = response;
            throw error;
        }
        return payload;
    }

    window.StockPredictorAPI = {
        get: function (url) { return request(url, { method: 'GET' }); },
        post: function (url, data) { return request(url, { method: 'POST', json: data }); },
        postForm: function (url, data) { return request(url, { method: 'POST', form: data }); },
        csrfToken: csrfToken,
        runBacktest: runBacktest,
        runMonteCarlo: runMonteCarlo,
        runExplain: runExplain,
        regenerateReport: regenerateReport,
        loadRelativeStrength: loadRelativeStrength,
        loadDrift: loadDrift,
        retrainSymbol: retrainSymbol,
    };

    /* ---------------- Toast notifications ---------------- */
    function showToast(message, type) {
        type = type || 'info';
        const stack = document.getElementById('toastStack');
        if (!stack) return;
        const toast = document.createElement('div');
        toast.className = 'toast toast-' + type;
        toast.textContent = message;
        stack.appendChild(toast);
        setTimeout(function () {
            toast.classList.add('leaving');
            setTimeout(function () { toast.remove(); }, 320);
        }, 4200);
    }
    window.showToast = showToast;

    /* ---------------- Modal system ---------------- */
    var __modalCloseHandler = null;

    function modalOpen(title, bodyHtml, actions) {
        var backdrop = document.getElementById('modalBackdrop');
        if (!backdrop) return null;
        var titleEl = document.getElementById('modalTitle');
        var bodyEl = document.getElementById('modalBody');
        var actionsEl = document.getElementById('modalActions');
        if (titleEl) titleEl.textContent = title;
        if (bodyEl) bodyEl.innerHTML = bodyHtml;
        if (actionsEl) {
            actionsEl.innerHTML = '';
            actions.forEach(function (a) {
                var btn = document.createElement('button');
                btn.type = 'button';
                btn.className = a.className || 'btn';
                btn.textContent = a.label;
                btn.addEventListener('click', function () { if (a.onClick) a.onClick(btn); });
                actionsEl.appendChild(btn);
            });
        }
        backdrop.hidden = false;
        requestAnimationFrame(function () { backdrop.classList.add('open'); });
        document.body.classList.add('modal-open');
        var first = bodyEl ? bodyEl.querySelector('input, select, button') : null;
        if (first) { setTimeout(function () { try { first.focus(); } catch (e) {} }, 60); }
        return backdrop;
    }

    function modalClose() {
        var backdrop = document.getElementById('modalBackdrop');
        if (!backdrop || backdrop.hidden) return;
        backdrop.classList.remove('open');
        backdrop.hidden = true;
        document.body.classList.remove('modal-open');
        __modalCloseHandler = null;
    }

    function setupModal() {
        var backdrop = document.getElementById('modalBackdrop');
        if (!backdrop) return;
        var closeBtn = document.getElementById('modalClose');
        function close() { if (__modalCloseHandler) __modalCloseHandler(); }
        if (closeBtn) closeBtn.addEventListener('click', close);
        backdrop.addEventListener('click', function (e) { if (e.target === backdrop) close(); });
        document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
    }

    function modalConfirm(opts) {
        return new Promise(function (resolve) {
            var settled = false;
            function done(val) {
                if (settled) return;
                settled = true;
                modalClose();
                resolve(val);
            }
            var backdrop = modalOpen(
                opts.title || 'Please confirm',
                '<p class="modal-message">' + opts.message + '</p>',
                [
                    { label: opts.cancelLabel || 'Cancel', className: 'btn btn-outline', onClick: function () { done(false); } },
                    { label: opts.confirmLabel || 'Confirm', className: opts.confirmClass || 'btn btn-primary', onClick: function () { done(true); } }
                ]
            );
            if (!backdrop) { resolve(false); return; }
            __modalCloseHandler = function () { done(false); };
        });
    }

    function modalPrompt(opts) {
        return new Promise(function (resolve) {
            var settled = false;
            function done(val) {
                if (settled) return;
                settled = true;
                modalClose();
                resolve(val);
            }
            var fields = opts.fields || [];
            var html = opts.message ? '<p class="modal-message">' + opts.message + '</p>' : '';
            html += fields.map(function (f, i) {
                var label = f.label
                    ? '<label class="modal-label" for="modalField' + i + '">' + f.label + '</label>'
                    : '';
                var control;
                if (f.type === 'select') {
                    control = '<select id="modalField' + i + '" class="modal-input">' +
                        (f.options || []).map(function (o) {
                            var val = o.value !== undefined ? o.value : o;
                            var text = o.label !== undefined ? o.label : o;
                            var sel = String(val) === String(f.value) ? ' selected' : '';
                            return '<option value="' + escapeHtml(val) + '"' + sel + '>' + escapeHtml(text) + '</option>';
                        }).join('') + '</select>';
                } else {
                    var t = f.type === 'number' ? 'number' : 'text';
                    control = '<input type="' + t + '" id="modalField' + i + '" class="modal-input" value="' +
                        escapeHtml(f.value || '') + '" placeholder="' + escapeHtml(f.placeholder || '') + '" step="' +
                        escapeHtml(f.step || 'any') + '">';
                }
                return '<div class="modal-field">' + label + control + '</div>';
            }).join('');
            var backdrop = modalOpen(
                opts.title || '',
                html,
                [
                    { label: opts.cancelLabel || 'Cancel', className: 'btn btn-outline', onClick: function () { done(null); } },
                    { label: opts.confirmLabel || 'OK', className: opts.confirmClass || 'btn btn-primary', onClick: function () {
                        var values = {};
                        fields.forEach(function (f, i) {
                            var el = document.getElementById('modalField' + i);
                            values[f.name] = el ? el.value : '';
                        });
                        done(values);
                    } }
                ]
            );
            if (!backdrop) { resolve(null); return; }
            __modalCloseHandler = function () { done(null); };
        });
    }

    window.SppModal = {
        confirm: modalConfirm,
        prompt: modalPrompt,
        close: modalClose
    };

    /* ---------------- Nav hamburger ---------------- */
    function setupNav() {
        const toggle = document.getElementById('navToggle');
        const links = document.getElementById('navLinks');
        if (!toggle || !links) return;
        toggle.addEventListener('click', function () {
            const open = links.classList.toggle('open');
            toggle.classList.toggle('open', open);
            toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        });
        links.addEventListener('click', function (e) {
            if (e.target.tagName === 'A' && links.classList.contains('open')) {
                links.classList.remove('open');
                toggle.classList.remove('open');
            }
        });
    }

    /* ---------------- Back to top ---------------- */
    function setupBackToTop() {
        const btn = document.getElementById('backToTop');
        if (!btn) return;
        window.addEventListener('scroll', function () {
            btn.classList.toggle('visible', window.scrollY > 420);
        }, { passive: true });
        btn.addEventListener('click', function () {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    /* ---------------- Auto-dismiss flash messages ---------------- */
    function setupFlashes() {
        document.querySelectorAll('.flash').forEach(function (el) {
            setTimeout(function () {
                el.style.transition = 'opacity 0.4s, transform 0.4s';
                el.style.opacity = '0';
                el.style.transform = 'translateX(12px)';
                setTimeout(function () { el.remove(); }, 420);
            }, 6000);
        });
    }

    /* ---------------- Market ticker ---------------- */
    function setupTicker() {
        const track = document.getElementById('tickerTrack');
        const bar = document.getElementById('tickerBar');
        if (!track || !bar) return;
        if (document.body.dataset.authenticated !== 'true') return;

        window.StockPredictorAPI.get('/api/market/ticker')
            .then(function (data) {
                const items = data.indices || {};
                const names = Object.keys(items);
                if (!names.length) { bar.remove(); return; }
                track.innerHTML = names.map(function (sym) {
                    const idx = items[sym];
                    const pct = (idx.change_pct !== null && idx.change_pct !== undefined) ? idx.change_pct : 0;
                    const cls = pct >= 0 ? 'trend-up' : 'trend-down';
                    const arrow = pct >= 0 ? '▲' : '▼';
                    return '<span class="ticker-item">' +
                        '<span class="t-symbol">' + escapeHtml(idx.name || sym) + '</span>' +
                        '<span class="t-price">' + escapeHtml(idx.price) + '</span>' +
                        '<span class="' + cls + '">' + arrow + ' ' + escapeHtml(pct.toFixed(2)) + '%</span>' +
                        '</span>';
                }).join('');
            })
            .catch(function () { bar.remove(); });
    }

    /* ---------------- Password reveal + strength ---------------- */
    function setupPasswordFields() {
        document.querySelectorAll('.password-wrap').forEach(function (wrap) {
            const input = wrap.querySelector('input[type="password"]');
            const btn = wrap.querySelector('.password-toggle');
            if (!input || !btn) return;
            btn.addEventListener('click', function () {
                const show = input.type === 'password';
                input.type = show ? 'text' : 'password';
                btn.textContent = show ? '🙈' : '👁️';
            });
        });

        const strength = document.getElementById('passwordStrength');
        const password = document.getElementById('password');
        if (strength && password) {
            password.addEventListener('input', function () {
                const bars = strength.querySelectorAll('.bar');
                let score = 0;
                if (password.value.length >= 8) score++;
                if (password.value.length >= 12) score++;
                if (/[A-Z]/.test(password.value) && /[a-z]/.test(password.value)) score++;
                if (/\d/.test(password.value) && /[^A-Za-z0-9]/.test(password.value)) score++;
                bars.forEach(function (bar, i) {
                    bar.classList.toggle('f' + (i + 1), i < score);
                });
            });
        }
    }

    /* ---------------- Analyze form loading + quote hint ---------------- */
    function setupAnalyzeForm() {
        const form = document.getElementById('stockForm');
        if (!form) return;

        form.addEventListener('submit', function () {
            const btn = form.querySelector('button[type="submit"]');
            if (btn) {
                btn.classList.add('loading');
                btn.innerHTML = '<span class="spinner"></span> Analyzing… (can take a minute)';
                btn.disabled = true;
            }
        });

        const symbolInput = document.getElementById('symbol');
        const hint = document.getElementById('symbolHint');
        if (symbolInput && hint) {
            let timer = null;
            symbolInput.addEventListener('input', function () {
                clearTimeout(timer);
                const sym = symbolInput.value.trim();
                if (!sym) { hint.textContent = ''; hint.className = 'field-hint'; return; }
                timer = setTimeout(function () {
                    hint.textContent = 'Checking…';
                    hint.className = 'field-hint';
                    window.StockPredictorAPI.get('/api/quote/' + encodeURIComponent(sym))
                        .then(function (data) {
                            if (data.success) {
                                const pct = (data.change_pct !== null && data.change_pct !== undefined) ? data.change_pct : null;
                                const cls = pct === null ? 'trend-neutral' : (pct >= 0 ? 'trend-up' : 'trend-down');
                                hint.className = 'field-hint ' + cls;
                                hint.textContent = pct === null
                                    ? '✓ ' + data.symbol + ' — live price ' + data.price
                                    : '✓ ' + data.symbol + ' — ' + data.price + ' (' + pct.toFixed(2) + '%)';
                            } else {
                                hint.textContent = 'Could not fetch a live quote — analysis may still work.';
                                hint.className = 'field-hint muted';
                            }
                        })
                        .catch(function () {
                            hint.textContent = 'Symbol not found yet — analysis will verify it.';
                            hint.className = 'field-hint muted';
                        });
                }, 450);
            });
        }

        const suggestions = document.getElementById('symbolSuggestions');
        if (symbolInput && suggestions) {
            let acTimer = null;
            let activeIndex = -1;
            let items = [];

            function hideList() {
                suggestions.hidden = true;
                activeIndex = -1;
            }

            function renderItems(results) {
                items = results || [];
                suggestions.innerHTML = items.map(function (r, i) {
                    return '<li class="autocomplete-item" role="option" data-index="' + i + '" aria-selected="false">' +
                        '<span class="ac-symbol">' + escapeHtml(r.symbol) + '</span>' +
                        '<span class="ac-name">' + escapeHtml(r.name || '') +
                        (r.exchange ? ' · ' + escapeHtml(r.exchange) : '') + '</span></li>';
                }).join('');
                suggestions.hidden = items.length === 0;
            }

            function selectItem(i) {
                if (!items[i]) return;
                symbolInput.value = items[i].symbol;
                hideList();
                symbolInput.dispatchEvent(new Event('input', { bubbles: true }));
                symbolInput.focus();
            }

            function updateActive() {
                Array.prototype.forEach.call(suggestions.children, function (li, i) {
                    li.classList.toggle('active', i === activeIndex);
                    li.setAttribute('aria-selected', i === activeIndex ? 'true' : 'false');
                });
            }

            symbolInput.addEventListener('input', function () {
                clearTimeout(acTimer);
                const sym = symbolInput.value.trim();
                if (!sym) { hideList(); return; }
                acTimer = setTimeout(function () {
                    window.StockPredictorAPI.get('/api/search?q=' + encodeURIComponent(sym) + '&limit=8')
                        .then(function (data) { renderItems((data && data.results) || []); })
                        .catch(function () { hideList(); });
                }, 200);
            });

            symbolInput.addEventListener('keydown', function (e) {
                if (suggestions.hidden || !items.length) return;
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    activeIndex = (activeIndex + 1) % items.length;
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    activeIndex = (activeIndex - 1 + items.length) % items.length;
                } else if (e.key === 'Enter') {
                    if (activeIndex >= 0) { e.preventDefault(); selectItem(activeIndex); return; }
                } else if (e.key === 'Escape') {
                    hideList();
                } else {
                    return;
                }
                updateActive();
            });

            suggestions.addEventListener('mousedown', function (e) {
                e.preventDefault();
                const li = e.target.closest('.autocomplete-item');
                if (li) selectItem(Number(li.getAttribute('data-index')));
            });

            document.addEventListener('click', function (e) {
                if (!e.target.closest('.autocomplete')) hideList();
            });
        }
    }

    /* ---------------- Forecast chart (results page) ---------------- */
    function setupForecastChart() {
        const node = document.getElementById('forecastChart');
        if (!node) return;

        let rows = [];
        try {
            rows = JSON.parse(node.getAttribute('data-rows') || '[]');
        } catch (e) {
            rows = [];
        }
        if (!rows.length) { node.innerHTML = ''; return; }

        const dateKey = 'Date';
        const valueCandidates = ['Predicted_Price', 'Predicted Price', 'Forecast', 'Close', 'Price'];
        const valueKey = valueCandidates.filter(function (k) { return rows[0][k] !== undefined; })[0];
        if (!valueKey) { node.innerHTML = ''; return; }

        const labels = rows.map(function (r) { return String(r[dateKey] || r.index || ''); });
        const values = rows.map(function (r) { return Number(r[valueKey]); }).filter(function (v) { return isFinite(v); });
        if (values.length < 2) { node.innerHTML = ''; return; }

        const lows = rows.map(function (r) { return Number(r.Low); }).filter(function (v) { return isFinite(v); });
        const highs = rows.map(function (r) { return Number(r.High); }).filter(function (v) { return isFinite(v); });
        const hasBand = lows.length === values.length && highs.length === values.length;

        const colors = chartColors();
        const up = values[values.length - 1] >= values[0];
        const lineColor = up ? colors.up : colors.down;
        const fillColor = 'rgba(' + (up ? colors.upRgb : colors.downRgb) + ',0.14)';
        let showBand = hasBand;
        let chart = null;

        function build() {
            const option = {
                animationDuration: 450,
                color: [lineColor],
                tooltip: chartTooltip(colors),
                legend: { show: false },
                grid: { left: 58, right: 20, top: 30, bottom: 30 },
                xAxis: {
                    type: 'category', boundaryGap: false, data: labels,
                    axisLabel: { color: colors.muted }, axisTick: { show: false },
                    axisLine: { lineStyle: { color: colors.grid } }
                },
                yAxis: {
                    type: 'value', scale: true,
                    axisLabel: { color: colors.muted },
                    splitLine: { lineStyle: { color: colors.grid } }
                },
                series: []
            };
            if (hasBand && showBand) {
                option.series.push(
                    { name: 'Band', type: 'line', data: lows, stack: 'band', symbol: 'none', lineStyle: { opacity: 0 }, emphasis: { disabled: true }, silent: true },
                    { name: 'bandfill', type: 'line', data: highs.map(function (h, i) { return h - lows[i]; }), stack: 'band', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: 'rgba(' + colors.accentRgb + ',0.16)' }, emphasis: { disabled: true }, silent: true }
                );
            }
            option.series.push({
                name: 'Forecast', type: 'line', data: values, showSymbol: false,
                lineStyle: { width: 2.4, color: lineColor },
                areaStyle: { color: fillColor },
                emphasis: { focus: 'series' },
                markPoint: {
                    symbol: 'circle', symbolSize: 6,
                    label: { show: true, color: lineColor, fontSize: 12, fontWeight: 700, position: 'top' },
                    data: [{ coord: [labels.length - 1, values[values.length - 1]], value: values[values.length - 1].toFixed(2) }]
                }
            });
            return option;
        }

        const toggles = [];
        if (hasBand) {
            toggles.push({ label: 'Band', value: 'band', active: true, onSelect: function () { showBand = !showBand; chart.setOption(build(), true); } });
        }
        chart = mountChart(node, build(), { height: 320, toggles: toggles });
    }

    /* ---------------- Price history chart (results page, line ⇄ candlestick) ---------------- */
    function setupPriceChart() {
        const node = document.getElementById('priceChart');
        if (!node) return;

        let rows = [];
        try {
            rows = JSON.parse(node.getAttribute('data-rows') || '[]');
        } catch (e) {
            rows = [];
        }
        if (!rows.length) { node.innerHTML = ''; return; }

        const labels = rows.map(function (r) { return String(r.Date || r.index || ''); });
        const closes = rows.map(function (r) { return Number(r.Close); }).filter(function (v) { return isFinite(v); });
        if (closes.length < 2) { node.innerHTML = ''; return; }

        const hasOhlc = rows.every(function (r) {
            return ['Open', 'High', 'Low', 'Close'].every(function (k) { return isFinite(Number(r[k])); });
        });
        const volumes = rows.map(function (r) { return isFinite(Number(r.Volume)) ? Number(r.Volume) : 0; });
        const hasVol = volumes.some(function (v) { return v > 0; });

        const colors = chartColors();
        const up = closes[closes.length - 1] >= closes[0];
        let mode = 'line';
        let chart = null;

        function build() {
            const option = {
                animationDuration: 450,
                tooltip: (function () { const t = chartTooltip(colors); t.formatter = priceTooltip; return t; })(),
                axisPointer: { link: [{ xAxisIndex: 'all' }], label: { backgroundColor: colors.muted } },
                legend: { show: mode === 'line', top: 0, right: 8, textStyle: { color: colors.muted }, data: ['Close'] },
                grid: [
                    { left: 58, right: 20, top: 34, height: '58%' },
                    { left: 58, right: 20, top: '72%', height: '16%' }
                ],
                xAxis: [
                    { type: 'category', data: labels, boundaryGap: true, axisLabel: { color: colors.muted }, axisTick: { show: false }, axisLine: { lineStyle: { color: colors.grid } } },
                    { type: 'category', data: labels, boundaryGap: true, axisLabel: { show: false }, axisTick: { show: false }, axisLine: { lineStyle: { color: colors.grid } } }
                ],
                yAxis: [
                    { type: 'value', scale: true, axisLabel: { color: colors.muted }, splitLine: { lineStyle: { color: colors.grid } } },
                    { type: 'value', axisLabel: { show: false }, splitLine: { show: false }, min: 0 }
                ],
                series: []
            };
            if (mode === 'candlestick' && hasOhlc) {
                option.series.push({
                    name: 'Price', type: 'candlestick',
                    data: rows.map(function (r) { return [Number(r.Open), Number(r.Close), Number(r.Low), Number(r.High)]; }),
                    itemStyle: { color: colors.up, color0: colors.down, borderColor: colors.up, borderColor0: colors.down },
                    emphasis: { itemStyle: { borderWidth: 2 } }
                });
            } else {
                option.series.push({
                    name: 'Close', type: 'line', data: closes, showSymbol: false,
                    lineStyle: { width: 2.2, color: up ? colors.up : colors.down },
                    areaStyle: { color: 'rgba(' + colors.accentRgb + ',0.16)' },
                    emphasis: { focus: 'series' },
                    markPoint: {
                        symbol: 'circle', symbolSize: 6,
                        label: { show: true, position: 'top', color: colors.text, fontSize: 11, fontWeight: 700 },
                        data: [{ coord: [labels.length - 1, closes[closes.length - 1]], value: closes[closes.length - 1].toFixed(2) }]
                    }
                });
            }
            if (hasVol) {
                option.series.push({
                    name: 'Volume', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: volumes,
                    barWidth: '55%',
                    itemStyle: { color: function (p) {
                        return closes[p.dataIndex] < closes[p.dataIndex - 1]
                            ? 'rgba(' + colors.downRgb + ',0.65)'
                            : 'rgba(' + colors.upRgb + ',0.65)';
                    } }
                });
            }
            return option;
        }

        const toggles = [];
        if (hasOhlc) {
            toggles.push(
                { label: 'Line', value: 'line', active: true, onSelect: function () { mode = 'line'; chart.setOption(build(), true); } },
                { label: 'Candles', value: 'candlestick', active: false, onSelect: function () { mode = 'candlestick'; chart.setOption(build(), true); } }
            );
        }
        chart = mountChart(node, build(), { height: 340, toggles: toggles });
    }

    /* ---------------- Escape helper ---------------- */
    function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    /* ---------------- CSS variable helper (theme-aware charts) ---------------- */
    function cssVar(name, fallback) {
        var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        return value || fallback;
    }

    /* ---------------- ECharts helpers (theme-aware, resizable) ---------------- */
    var __chartRegistry = [];

    function chartColors() {
        return {
            accent: cssVar('--accent', '#38bdf8'),
            accentRgb: cssVar('--accent-rgb', '56, 189, 248'),
            up: cssVar('--up', '#22d3a6'),
            upRgb: cssVar('--up-rgb', '34, 211, 166'),
            down: cssVar('--down', '#fb7185'),
            downRgb: cssVar('--down-rgb', '251, 113, 133'),
            text: cssVar('--text', '#e9eef6'),
            muted: cssVar('--chart-label', 'rgba(255,255,255,0.45)'),
            grid: cssVar('--chart-grid', 'rgba(255,255,255,0.08)')
        };
    }

    function chartTooltip(colors) {
        return {
            backgroundColor: 'rgba(15, 23, 42, 0.92)',
            borderColor: colors.grid,
            borderWidth: 1,
            textStyle: { color: colors.text, fontFamily: 'Inter, sans-serif' },
            axisPointer: { type: 'line', lineStyle: { color: colors.muted, type: 'dashed' } }
        };
    }

    function mountChart(node, option, opts) {
        if (!window.echarts) {
            node.innerHTML = '<p class="muted">Interactive charts unavailable (ECharts failed to load).</p>';
            return null;
        }
        node.innerHTML = '';
        node.style.display = 'flex';
        node.style.flexDirection = 'column';
        node.style.height = ((opts && opts.height) || 300) + 'px';

        if (opts && opts.toggles && opts.toggles.length) {
            var bar = document.createElement('div');
            bar.className = 'chart-toolbar';
            opts.toggles.forEach(function (t) {
                var b = document.createElement('button');
                b.type = 'button';
                b.className = 'chart-toggle' + (t.active ? ' active' : '');
                b.textContent = t.label;
                b.addEventListener('click', function () {
                    Array.prototype.forEach.call(bar.querySelectorAll('.chart-toggle'), function (x) {
                        x.classList.toggle('active', x === b);
                    });
                    if (t.onSelect) t.onSelect(t.value);
                });
                bar.appendChild(b);
            });
            node.appendChild(bar);
        }

        var canvas = document.createElement('div');
        canvas.className = 'chart-canvas';
        node.appendChild(canvas);

        var chart = echarts.init(canvas, null, { renderer: 'canvas' });
        chart.setOption(option, true);
        __chartRegistry.push(chart);
        return chart;
    }

    function disposeAllCharts() {
        __chartRegistry.forEach(function (c) {
            if (c) { try { c.dispose(); } catch (e) {} }
        });
        __chartRegistry = [];
    }

    function priceTooltip(params) {
        var p = params[0];
        if (p && p.seriesType === 'candlestick' && Array.isArray(p.data)) {
            return p.name + '<br/>Open: <b>' + p.data[0] + '</b><br/>' +
                'Close: <b>' + p.data[1] + '</b><br/>' +
                'Low: <b>' + p.data[2] + '</b><br/>' +
                'High: <b>' + p.data[3] + '</b>';
        }
        var lines = params.map(function (x) {
            return x.marker + ' ' + x.seriesName + ': <b>' +
                (x.seriesName === 'Volume' ? Number(x.value).toLocaleString() : x.value) + '</b>';
        });
        return p.name + '<br/>' + lines.join('<br/>');
    }

    /* ---------------- Theme toggle ---------------- */
    function setupTheme() {
        var btn = document.getElementById('themeToggle');
        if (!btn) return;
        function refresh() {
            var theme = document.documentElement.getAttribute('data-theme');
            btn.textContent = theme === 'light' ? '☀️' : '🌙';
        }
        refresh();
        btn.addEventListener('click', function () {
            var theme = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', theme);
            try { localStorage.setItem('spp-theme', theme); } catch (e) {}
            refresh();
            refreshCharts();
        });
    }

    /* ---------------- Accent color picker ---------------- */
    var ACCENT_NAMES = ['blue', 'violet', 'indigo', 'teal', 'gold', 'rose'];

    function setupAccent() {
        var toggle = document.getElementById('accentToggle');
        var swatches = document.getElementById('accentSwatches');
        if (!toggle || !swatches) return;

        function current() {
            var accent = document.documentElement.getAttribute('data-accent') || 'blue';
            return ACCENT_NAMES.indexOf(accent) !== -1 ? accent : 'blue';
        }
        function highlight() {
            swatches.querySelectorAll('.accent-swatch').forEach(function (s) {
                s.classList.toggle('active', s.getAttribute('data-accent') === current());
            });
        }
        function apply(accent) {
            document.documentElement.setAttribute('data-accent', accent);
            try { localStorage.setItem('spp-accent', accent); } catch (e) {}
            highlight();
            refreshCharts();
        }

        toggle.addEventListener('click', function () {
            var hidden = swatches.hasAttribute('hidden');
            if (hidden) {
                swatches.removeAttribute('hidden');
                highlight();
            } else {
                swatches.setAttribute('hidden', '');
            }
        });
        swatches.querySelectorAll('.accent-swatch').forEach(function (swatch) {
            swatch.addEventListener('click', function () {
                apply(swatch.getAttribute('data-accent'));
                swatches.setAttribute('hidden', '');
            });
        });
        document.addEventListener('click', function (e) {
            if (!e.target.closest('#accentPicker')) {
                swatches.setAttribute('hidden', '');
            }
        });
        highlight();
    }

    /* ---------------- Re-render all charts after theme/accent change ---------------- */
    function refreshCharts() {
        disposeAllCharts();
        setupForecastChart();
        setupPriceChart();
        setupCompareChart();
        setupEquityChart();
        setupAllocationChart();
        setupPositionPnlChart();
        setupSparklines();
        if (window.__advancedData && window.__advancedData.backtestChart) {
            var bt = document.getElementById('backtestChart');
            if (bt) renderBacktestChart(bt, window.__advancedData.backtestChart);
        }
        if (window.__advancedData && window.__advancedData.mcChart) {
            var mc = document.getElementById('mcChart');
            if (mc) renderMonteCarloChart(mc, window.__advancedData.mcChart);
        }
    }

    /* ---------------- Compare overlay chart ---------------- */
    function setupCompareChart() {
        var node = document.getElementById('compareChart');
        if (!node) return;
        var series = [];
        try {
            series = JSON.parse(node.getAttribute('data-series') || '[]');
        } catch (e) {
            series = [];
        }
        series = series.filter(function (s) { return s.values && s.values.length >= 2; });
        if (!series.length) { node.innerHTML = ''; return; }

        var colors = chartColors();
        var palette = ['#38bdf8', '#a78bfa', '#f5c542', '#2dd4bf', '#fb7185', '#fb923c'];
        var longest = series.slice().sort(function (a, b) { return b.labels.length - a.labels.length; })[0];
        var labels = (longest && longest.labels) || [];

        var option = {
            animationDuration: 450,
            color: palette,
            tooltip: chartTooltip(colors),
            legend: { top: 0, right: 8, type: 'scroll', textStyle: { color: colors.muted } },
            grid: { left: 58, right: 24, top: 40, bottom: 30 },
            xAxis: {
                type: 'category', boundaryGap: false, data: labels,
                axisLabel: { color: colors.muted }, axisTick: { show: false },
                axisLine: { lineStyle: { color: colors.grid } }
            },
            yAxis: {
                type: 'value', scale: true,
                axisLabel: { color: colors.muted },
                splitLine: { lineStyle: { color: colors.grid } }
            },
            series: series.map(function (s, i) {
                var c = s.color || palette[i % palette.length];
                return {
                    name: s.symbol || s.name || ('Series ' + (i + 1)),
                    type: 'line', data: s.values, showSymbol: false,
                    lineStyle: { width: 2.2, color: c }, itemStyle: { color: c },
                    emphasis: { focus: 'series' }
                };
            })
        };
        mountChart(node, option, { height: 340 });
    }

    /* ---------------- Sparkline helper (dashboard watchlist) ---------------- */
    function sparklineSVG(values, up) {
        if (!values || values.length < 2) return '';
        var color = up ? cssVar('--up', '#22d3a6') : cssVar('--down', '#fb7185');
        var rgb = up ? cssVar('--up-rgb', '34, 211, 166') : cssVar('--down-rgb', '251, 113, 133');
        var w = 92, h = 34, padT = 3, padB = 3;
        var min = Math.min.apply(null, values), max = Math.max.apply(null, values);
        var range = (max - min) || 1;
        var x = function (i) { return (i / (values.length - 1)) * (w - 4) + 2; };
        var y = function (v) { return padT + ((max - v) / range) * (h - padT - padB); };
        var d = values.map(function (v, i) { return (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(v).toFixed(1); }).join(' ');
        var area = d + ' L' + x(values.length - 1).toFixed(1) + ' ' + h + ' L' + x(0).toFixed(1) + ' ' + h + ' Z';
        return '<svg class="sparkline" viewBox="0 0 ' + w + ' ' + h + '" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
            '<path d="' + area + '" fill="rgba(' + rgb + ',0.16)" />' +
            '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" />' +
            '</svg>';
    }
    window.sparklineSVG = sparklineSVG;

    function setupSparklines() {
        document.querySelectorAll('[data-sparkline]').forEach(function (el) {
            var raw = [];
            try { raw = JSON.parse(el.getAttribute('data-sparkline') || '[]'); } catch (e) { raw = []; }
            if (raw.length < 2) return;
            var up = raw[raw.length - 1] >= raw[0];
            el.innerHTML = sparklineSVG(raw, up);
        });
    }

    /* ---------------- Equity curve chart ---------------- */
    function setupEquityChart() {
        var node = document.getElementById('equityChart');
        if (!node) return;
        var rows = [];
        try { rows = JSON.parse(node.getAttribute('data-rows') || '[]'); } catch (e) { rows = []; }
        var values = rows.map(function (r) { return Number(r.value); }).filter(function (v) { return isFinite(v); });
        if (values.length < 2) { node.innerHTML = ''; return; }

        var labels = rows.map(function (r) { return String(r.date || ''); });
        var colors = chartColors();
        var up = values[values.length - 1] >= values[0];
        var lineColor = up ? colors.up : colors.down;

        var option = {
            animationDuration: 450,
            tooltip: chartTooltip(colors),
            grid: { left: 70, right: 20, top: 24, bottom: 28 },
            xAxis: {
                type: 'category', boundaryGap: false, data: labels,
                axisLabel: { color: colors.muted }, axisTick: { show: false },
                axisLine: { lineStyle: { color: colors.grid } }
            },
            yAxis: {
                type: 'value', scale: true,
                axisLabel: { color: colors.muted },
                splitLine: { lineStyle: { color: colors.grid } }
            },
            series: [{
                name: 'Equity', type: 'line', data: values, showSymbol: false,
                lineStyle: { width: 2.2, color: lineColor },
                areaStyle: { color: up ? 'rgba(' + colors.upRgb + ',0.14)' : 'rgba(' + colors.downRgb + ',0.14)' },
                emphasis: { focus: 'series' }
            }]
        };
        mountChart(node, option, { height: 300 });
    }

    /* ---------------- Allocation donut chart ---------------- */
    function setupAllocationChart() {
        var node = document.getElementById('allocationChart');
        if (!node) return;
        var slices = [];
        try { slices = JSON.parse(node.getAttribute('data-slices') || '[]'); } catch (e) { slices = []; }
        slices = slices.filter(function (s) { return Number(s.value) > 0; });
        if (!slices.length) { node.innerHTML = ''; return; }
        var total = slices.reduce(function (a, s) { return a + Number(s.value); }, 0);
        if (!total) { node.innerHTML = ''; return; }

        var colors = chartColors();
        var option = {
            animationDuration: 450,
            color: ['#38bdf8', '#a78bfa', '#f5c542', '#2dd4bf', '#fb7185', '#fb923c', '#818cf8', '#c084fc', '#34d399', '#fbbf24', '#60a5fa', '#f472b6'],
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(15, 23, 42, 0.92)', borderColor: colors.grid, borderWidth: 1,
                textStyle: { color: colors.text },
                formatter: '{b}: <b>${c}</b> ({d}%)'
            },
            legend: { orient: 'horizontal', bottom: 0, type: 'scroll', textStyle: { color: colors.muted } },
            title: {
                text: '$' + total.toFixed(0), subtext: 'Total', left: 'center', top: '38%',
                textStyle: { color: colors.text, fontSize: 18, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace' },
                subtextStyle: { color: colors.muted, fontSize: 11 }
            },
            series: [{
                name: 'Allocation', type: 'pie', radius: ['58%', '80%'], center: ['50%', '44%'],
                avoidLabelOverlap: true,
                itemStyle: { borderWidth: 2, borderColor: 'rgba(255,255,255,0)' },
                label: { color: colors.muted, formatter: '{b} {d}%', fontSize: 11 },
                labelLine: { lineStyle: { color: colors.muted } },
                emphasis: {
                    label: { color: colors.text, fontWeight: 700 },
                    itemStyle: { shadowBlur: 12, shadowColor: 'rgba(0,0,0,0.4)' }
                },
                data: slices.map(function (s) { return { name: s.label, value: Number(s.value) }; })
            }]
        };
        mountChart(node, option, { height: 300 });
    }

    /* ---------------- Unrealized P&L by position (bars) ---------------- */
    function setupPositionPnlChart() {
        var node = document.getElementById('positionPnlChart');
        if (!node) return;
        var rows = [];
        try { rows = JSON.parse(node.getAttribute('data-rows') || '[]'); } catch (e) { rows = []; }
        rows = rows.filter(function (r) { return r.pnl !== null && r.pnl !== undefined; });
        if (!rows.length) { node.innerHTML = ''; return; }
        rows = rows.slice(0, 8);

        var colors = chartColors();
        var data = rows.map(function (r) { return Number(r.pnl); });
        var fmt = function (v) { return (v >= 0 ? '+' : '-') + '$' + Math.abs(Number(v)).toFixed(0); };
        var option = {
            animationDuration: 450,
            tooltip: {
                trigger: 'axis', axisPointer: { type: 'shadow' },
                backgroundColor: 'rgba(15, 23, 42, 0.92)', borderColor: colors.grid, borderWidth: 1,
                textStyle: { color: colors.text },
                formatter: function (params) { var p = params[0]; return p.name + ': <b>' + fmt(p.value) + '</b>'; }
            },
            grid: { left: 64, right: 74, top: 8, bottom: 10 },
            xAxis: { type: 'value', axisLabel: { color: colors.muted, formatter: fmt }, splitLine: { lineStyle: { color: colors.grid } } },
            yAxis: {
                type: 'category', data: rows.map(function (r) { return r.symbol; }),
                axisLabel: { color: colors.muted }, axisTick: { show: false },
                axisLine: { lineStyle: { color: colors.grid } }
            },
            visualMap: { show: false, dimension: 0, pieces: [{ gte: 0, color: colors.up }, { lt: 0, color: colors.down }] },
            series: [{
                name: 'P&L', type: 'bar', data: data, barMaxWidth: 18,
                label: { show: true, position: 'right', color: colors.muted, fontSize: 11, formatter: function (p) { return fmt(p.value); } },
                emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.35)' } }
            }]
        };
        mountChart(node, option, { height: Math.min(430, 60 + data.length * 42) });
    }

    /* ---------------- Advanced features (results page) ---------------- */
    /* API-driven charts cache their last payload so theme/accent changes can
     * re-render them without another network call. */
    window.__advancedData = window.__advancedData || {};

    function fetchText(nodeId) {
        return document.getElementById(nodeId) ? document.getElementById(nodeId).value : '';
    }

    function runBacktest() {
        var card = document.getElementById('backtestCard');
        if (!card) return;
        var symbol = card.getAttribute('data-symbol');
        var strategy = fetchText('btStrategy') || 'MA_CROSS';
        var fast = fetchText('btFast') || '20';
        var slow = fetchText('btSlow') || '50';
        var panel = document.getElementById('backtestPanel');
        panel.innerHTML = '<p class="muted">Running backtest…</p>';
        window.StockPredictorAPI.get('/api/backtest/' + encodeURIComponent(symbol) +
            '?strategy=' + encodeURIComponent(strategy) + '&fast=' + encodeURIComponent(fast) + '&slow=' + encodeURIComponent(slow))
            .then(function (data) {
                if (!data.success) { panel.innerHTML = '<p class="muted">' + escapeHtml(data.message || 'Backtest failed') + '</p>'; return; }
                window.__advancedData.backtest = { symbol: symbol, data: data };
                renderBacktest(panel, data);
            })
            .catch(function (err) { panel.innerHTML = '<p class="muted">Backtest failed: ' + escapeHtml(err.message) + '</p>'; });
    }

    function renderBacktest(panel, data) {
        var m = data.metrics || {};
        var html = '<div class="grid">' +
            metricCell('Total Return', (m.total_return_pct !== undefined ? (m.total_return_pct >= 0 ? '+' : '') + m.total_return_pct + '%' : '—')) +
            metricCell('Buy &amp; Hold', (m.buy_and_hold_return_pct !== undefined ? (m.buy_and_hold_return_pct >= 0 ? '+' : '') + m.buy_and_hold_return_pct + '%' : '—')) +
            metricCell('CAGR', (m.cagr !== undefined ? m.cagr + '%' : '—')) +
            metricCell('Sharpe', m.sharpe !== undefined ? m.sharpe : '—') +
            metricCell('Max Drawdown', m.max_drawdown_pct !== undefined ? m.max_drawdown_pct + '%' : '—') +
            metricCell('Win Rate', m.win_rate !== undefined ? m.win_rate + '%' : '—') +
            metricCell('Trades', m.total_trades !== undefined ? m.total_trades : '—') +
            metricCell('Final Equity', m.final_equity !== undefined ? '$' + Number(m.final_equity).toFixed(0) : '—') +
            '</div>';
        var equity = (data.equity_curve || []).map(function (r) { return { date: r.date, value: r.equity, series: 'strategy' }; })
            .concat((data.buy_hold_curve || []).map(function (r) { return { date: r.date, value: r.equity, series: 'buyhold' }; }));
        var nodeId = 'backtestChart';
        var chartWrap = document.createElement('div');
        chartWrap.className = 'chart-wrap';
        chartWrap.id = nodeId;
        panel.innerHTML = html;
        panel.appendChild(chartWrap);
        window.__advancedData.backtestChart = equity;
        renderBacktestChart(chartWrap, equity);

        var trades = data.trades || [];
        if (trades.length) {
            var table = document.createElement('div');
            table.className = 'table-wrap';
            table.style.marginTop = '16px';
            table.innerHTML = '<table class="stock-table"><thead><tr><th>Date</th><th>Action</th><th>Price</th><th>Shares</th><th>Value</th></tr></thead><tbody>' +
                trades.map(function (t) {
                    return '<tr><td>' + escapeHtml(t.date) + '</td><td>' + escapeHtml(t.action) + '</td><td class="mono">$' + t.price + '</td><td class="mono">' + t.shares + '</td><td class="mono">$' + Number(t.value).toFixed(2) + '</td></tr>';
                }).join('') + '</tbody></table>';
            panel.appendChild(table);
        }
    }

    function metricCell(label, value) {
        return '<div class="metric-card"><div class="metric-label">' + label + '</div><div class="metric-value">' + value + '</div></div>';
    }

    function renderBacktestChart(node, series) {
        var strategy = series.filter(function (s) { return s.series === 'strategy'; });
        var buyhold = series.filter(function (s) { return s.series === 'buyhold'; });
        if (strategy.length < 2) { node.innerHTML = ''; return; }

        var colors = chartColors();
        var option = {
            animationDuration: 450,
            tooltip: chartTooltip(colors),
            legend: { top: 0, right: 8, textStyle: { color: colors.muted }, data: ['Strategy', 'Buy & Hold'] },
            grid: { left: 70, right: 20, top: 36, bottom: 28 },
            xAxis: {
                type: 'category', boundaryGap: false,
                data: strategy.map(function (s) { return s.date; }),
                axisLabel: { color: colors.muted }, axisTick: { show: false },
                axisLine: { lineStyle: { color: colors.grid } }
            },
            yAxis: {
                type: 'value', scale: true,
                axisLabel: { color: colors.muted },
                splitLine: { lineStyle: { color: colors.grid } }
            },
            series: [
                { name: 'Strategy', type: 'line', data: strategy.map(function (s) { return s.value; }), showSymbol: false, lineStyle: { width: 2.4, color: colors.accent }, itemStyle: { color: colors.accent }, emphasis: { focus: 'series' } },
                { name: 'Buy & Hold', type: 'line', data: buyhold.map(function (s) { return s.value; }), showSymbol: false, lineStyle: { width: 1.6, color: colors.muted, type: 'dashed' }, itemStyle: { color: colors.muted } }
            ]
        };
        mountChart(node, option, { height: 300 });
    }

    function runMonteCarlo() {
        var card = document.getElementById('monteCarloCard');
        if (!card) return;
        var symbol = card.getAttribute('data-symbol');
        var payload = {
            horizon: parseInt(fetchText('mcHorizon') || '30', 10) || 30,
            n_paths: 1000
        };
        var target = fetchText('mcTarget');
        var stop = fetchText('mcStop');
        if (target) payload.target = parseFloat(target);
        if (stop) payload.stop = parseFloat(stop);

        var panel = document.getElementById('mcPanel');
        panel.innerHTML = '<p class="muted">Simulating 1,000 paths…</p>';
        window.StockPredictorAPI.post('/api/scenario/' + encodeURIComponent(symbol), payload)
            .then(function (data) {
                if (!data.success) { panel.innerHTML = '<p class="muted">' + escapeHtml(data.message || 'Simulation failed') + '</p>'; return; }
                window.__advancedData.mc = { symbol: symbol, data: data };
                renderMonteCarlo(panel, data);
            })
            .catch(function (err) { panel.innerHTML = '<p class="muted">Simulation failed: ' + escapeHtml(err.message) + '</p>'; });
    }

    function renderMonteCarlo(panel, data) {
        var b = data.bands || {};
        var html = '<div class="grid">' +
            metricCell('Last Price', data.last_price !== undefined ? '$' + data.last_price : '—') +
            metricCell('Mean Final', data.mean_final !== undefined ? '$' + data.mean_final : '—') +
            metricCell('95th %', (b.upper95 && b.upper95.length) ? '$' + b.upper95[b.upper95.length - 1] : '—') +
            metricCell('5th %', (b.lower5 && b.lower5.length) ? '$' + b.lower5[b.lower5.length - 1] : '—') +
            metricCell('Above Today', data.prob_above_current !== undefined ? data.prob_above_current + '%' : '—') +
            metricCell('Below Today', data.prob_below_current !== undefined ? data.prob_below_current + '%' : '—') +
            '</div>';
        if (data.prob_hit_target !== undefined) {
            html += '<div class="grid">' +
                metricCell('P(hit target)', data.prob_hit_target + '%') +
                metricCell('P(hit stop)', data.prob_hit_stop + '%') +
                metricCell('P(hit either)', data.prob_hit_either + '%') +
                '</div>';
        }
        var nodeId = 'mcChart';
        var chartWrap = document.createElement('div');
        chartWrap.className = 'chart-wrap';
        chartWrap.id = nodeId;
        panel.innerHTML = html;
        panel.appendChild(chartWrap);
        window.__advancedData.mcChart = { bands: b, labels: (b.median || []).map(function (_, i) { return i; }) };
        renderMonteCarloChart(chartWrap, window.__advancedData.mcChart);
    }

    function renderMonteCarloChart(node, payload) {
        var median = payload.bands.median || [];
        var lower = payload.bands.lower5 || [];
        var upper = payload.bands.upper95 || [];
        if (median.length < 2) { node.innerHTML = ''; return; }

        var colors = chartColors();
        var labels = median.map(function (_, i) { return 'D' + i; });
        var option = {
            animationDuration: 450,
            tooltip: chartTooltip(colors),
            grid: { left: 58, right: 20, top: 30, bottom: 28 },
            xAxis: {
                type: 'category', boundaryGap: false, data: labels,
                axisLabel: { color: colors.muted }, axisTick: { show: false },
                axisLine: { lineStyle: { color: colors.grid } }
            },
            yAxis: {
                type: 'value', scale: true,
                axisLabel: { color: colors.muted },
                splitLine: { lineStyle: { color: colors.grid } }
            },
            series: [
                { name: 'bandfill', type: 'line', data: lower, stack: 'mc', symbol: 'none', lineStyle: { opacity: 0 }, emphasis: { disabled: true }, silent: true },
                { name: 'band', type: 'line', data: upper.map(function (u, i) { return u - lower[i]; }), stack: 'mc', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: 'rgba(' + colors.accentRgb + ',0.16)' }, emphasis: { disabled: true }, silent: true },
                { name: 'Median', type: 'line', data: median, showSymbol: false, lineStyle: { width: 2.2, color: colors.accent }, itemStyle: { color: colors.accent }, emphasis: { focus: 'series' } }
            ]
        };
        mountChart(node, option, { height: 300 });
    }

    function runExplain() {
        var card = document.getElementById('explainCard');
        if (!card) return;
        var symbol = card.getAttribute('data-symbol');
        var panel = document.getElementById('explainPanel');
        panel.innerHTML = '<p class="muted">Computing feature attributions…</p>';
        window.StockPredictorAPI.get('/api/explain/' + encodeURIComponent(symbol))
            .then(function (data) {
                if (!data.success) { panel.innerHTML = '<p class="muted">' + escapeHtml(data.message || 'Explainability unavailable') + '</p>'; return; }
                window.__advancedData.explain = data;
                renderExplain(panel, data);
            })
            .catch(function (err) { panel.innerHTML = '<p class="muted">Explainability failed: ' + escapeHtml(err.message) + '</p>'; });
    }

    function renderExplain(panel, data) {
        var top = data.top || [];
        if (!top.length) { panel.innerHTML = '<p class="muted">No importance data.</p>'; return; }
        var maxVal = Math.max.apply(null, top.map(function (t) { return t.value; })) || 1;
        var bars = top.map(function (t) {
            var pct = (t.value / maxVal * 100).toFixed(1);
            return '<div class="score-item">' +
                '<span>' + escapeHtml(t.name) + '</span>' +
                '<div class="score-bar"><div style="width:' + pct + '%;"></div></div>' +
                '<b>' + t.value.toFixed(4) + '</b></div>';
        }).join('');
        panel.innerHTML =
            '<p class="field-hint">Method: <span class="chip">' + escapeHtml(data.method || '—') + '</span></p>' +
            '<div class="score-grid">' + bars + '</div>';
    }

    function regenerateReport() {
        var card = document.getElementById('analystReportCard');
        if (!card) return;
        var symbol = card.getAttribute('data-symbol');
        var payload = {};
        try { payload = JSON.parse(card.getAttribute('data-payload') || '{}'); } catch (e) { payload = {}; }
        var body = document.getElementById('analystReportBody');
        var btn = document.getElementById('regenerateReportBtn');
        if (btn) { btn.disabled = true; btn.textContent = 'Generating…'; }
        body.innerHTML = '<p class="muted">Generating AI report…</p>';
        window.StockPredictorAPI.post('/api/report/' + encodeURIComponent(symbol), payload)
            .then(function (data) {
                body.innerHTML = '<pre class="report-pre">' + escapeHtml(data.report || 'No report generated.') + '</pre>';
            })
            .catch(function (err) {
                body.innerHTML = '<p class="muted">Report generation failed: ' + escapeHtml(err.message) + '</p>';
            })
            .then(function () {
                if (btn) { btn.disabled = false; btn.textContent = '✨ Regenerate with AI'; }
            });
    }


    /* ---------------- Dashboard: relative strength + drift (JS-driven) ---------------- */
    function loadRelativeStrength() {
        var panel = document.getElementById('rsPanel');
        var btn = document.getElementById('loadRsBtn');
        if (!panel) return;
        if (btn) btn.textContent = 'Loading…';
        window.StockPredictorAPI.get('/api/relative-strength')
            .then(function (data) {
                if (!data.success) { panel.innerHTML = '<p class="muted">' + escapeHtml(data.message || 'Unavailable') + '</p>'; return; }
                var rows = data.rows || [];
                if (!rows.length) { panel.innerHTML = '<p class="muted">No watchlist symbols to rank. Add symbols to your watchlist first.</p>'; return; }
                var html = '<div class="table-wrap"><table class="stock-table" id="rsTable"><thead><tr>' +
                    '<th data-sort="symbol">Symbol</th>' +
                    '<th data-sort="20d">20d</th>' +
                    '<th data-sort="60d">60d</th>' +
                    '<th data-sort="120d">120d</th>' +
                    '<th data-sort="composite">RS Score</th></tr></thead><tbody>' +
                    rows.map(function (r) {
                        var w20 = r.windows['20'] || {}, w60 = r.windows['60'] || {}, w120 = r.windows['120'] || {};
                        var cls = r.composite >= 55 ? 'trend-up' : (r.composite <= 45 ? 'trend-down' : '');
                        return '<tr data-composite="' + r.composite + '" data-symbol="' + escapeHtml(r.symbol) + '">' +
                            '<td><strong>' + escapeHtml(r.symbol) + '</strong></td>' +
                            '<td class="mono ' + (w20.excess_return_pct >= 0 ? 'trend-up' : 'trend-down') + '">' + w20.excess_return_pct + '%</td>' +
                            '<td class="mono ' + (w60.excess_return_pct >= 0 ? 'trend-up' : 'trend-down') + '">' + w60.excess_return_pct + '%</td>' +
                            '<td class="mono ' + (w120.excess_return_pct >= 0 ? 'trend-up' : 'trend-down') + '">' + w120.excess_return_pct + '%</td>' +
                            '<td><strong class="' + cls + '">' + r.composite + '</strong></td></tr>';
                    }).join('') + '</tbody></table></div>' +
                    '<p class="field-hint muted">Excess return = symbol return − SPY return over each window. RS Score ranks momentum vs the benchmark.</p>';
                panel.innerHTML = html;
                sortableTable(document.getElementById('rsTable'));
            })
            .catch(function (err) { panel.innerHTML = '<p class="muted">Relative strength unavailable: ' + escapeHtml(err.message) + '</p>'; })
            .then(function () { if (btn) btn.textContent = '↻ Load'; });
    }

    function sortableTable(table) {
        if (!table) return;
        Array.prototype.forEach.call(table.querySelectorAll('th'), function (th) {
            th.style.cursor = 'pointer';
            th.addEventListener('click', function () {
                var key = th.getAttribute('data-sort') || 'symbol';
                var tbody = table.querySelector('tbody');
                var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
                var numeric = key !== 'symbol';
                rows.sort(function (a, b) {
                    var av = a.getAttribute('data-' + key);
                    var bv = b.getAttribute('data-' + key);
                    if (numeric) { av = parseFloat(av) || 0; bv = parseFloat(bv) || 0; return av - bv; }
                    return String(av).localeCompare(String(bv));
                });
                rows.forEach(function (r) { tbody.appendChild(r); });
            });
        });
    }

    function loadDrift() {
        var panel = document.getElementById('driftPanel');
        var btn = document.getElementById('loadDriftBtn');
        var card = document.getElementById('driftCard');
        if (!panel || !card) return;
        if (btn) btn.textContent = 'Checking…';
        var symbols = [];
        try { symbols = JSON.parse(card.getAttribute('data-watchlist') || '[]'); } catch (e) { symbols = []; }
        if (!symbols.length) { panel.innerHTML = '<p class="muted">Add symbols to your watchlist to monitor drift.</p>'; return; }

        Promise.all(symbols.map(function (s) {
            return window.StockPredictorAPI.get('/api/drift/' + encodeURIComponent(s))
                .then(function (d) { return { symbol: s, data: d }; })
                .catch(function (err) { return { symbol: s, data: { success: false, message: err.message } }; });
        })).then(function (results) {
            var html = '<div class="table-wrap"><table class="stock-table"><thead><tr>' +
                '<th>Symbol</th><th>Overall PSI</th><th>Drift</th><th>Drifted Features</th><th></th></tr></thead><tbody>' +
                results.map(function (r) {
                    var d = r.data;
                    if (!d.success) return '<tr><td><strong>' + escapeHtml(r.symbol) + '</strong></td><td colspan="4" class="muted">' + escapeHtml(d.message || 'n/a') + '</td></tr>';
                    var drifted = (d.drifted_features || []).map(function (f) { return '<span class="chip">' + escapeHtml(f) + '</span>'; }).join(' ') || '<span class="muted">none</span>';
                    var pill = d.drift_detected ? 'pill-red' : 'pill-green';
                    var retrain = d.drift_detected
                        ? '<button class="btn btn-outline btn-sm" onclick="StockPredictorAPI.retrainSymbol(\'' + escapeHtml(r.symbol) + '\', this)">Retrain</button>'
                        : '';
                    return '<tr><td><strong>' + escapeHtml(r.symbol) + '</strong></td>' +
                        '<td class="mono">' + d.overall_psi + '</td>' +
                        '<td><span class="pill ' + pill + '">' + (d.drift_detected ? 'DRIFTED' : 'OK') + '</span></td>' +
                        '<td>' + drifted + '</td><td>' + retrain + '</td></tr>';
                }).join('') + '</tbody></table></div>';
            panel.innerHTML = html;
            if (btn) btn.textContent = '↻ Check';
        });
    }

    function retrainSymbol(symbol, btn) {
        if (!btn) return;
        var original = btn.textContent;
        btn.disabled = true;
        btn.textContent = 'Queued…';
        window.StockPredictorAPI.post('/api/drift/' + encodeURIComponent(symbol) + '/retrain', {})
            .then(function (data) {
                showToast(data.success ? 'Retraining queued for ' + symbol + ' (job ' + data.job_id + ')' : (data.message || 'Failed'), data.success ? 'success' : 'warning');
            })
            .catch(function (err) { showToast(err.message, 'error'); })
            .then(function () { btn.disabled = false; btn.textContent = original; });
    }


    function setupAnalysisPolling() {
        var card = document.querySelector('.analyzing-card');
        if (!card) return;
        var jobId = card.getAttribute('data-job-id');
        var hint = document.getElementById('jobMessage');
        var fill = document.getElementById('jobProgressFill');
        var endpoint = hint ? hint.getAttribute('data-job-endpoint') : null;
        var resultUrl = hint ? hint.getAttribute('data-result-url') : null;
        if (!jobId || !endpoint) return;

        var elapsed = 0;
        var timer = setInterval(function () {
            elapsed += 1500;
            window.StockPredictorAPI.get(endpoint).then(function (data) {
                if (!hint) return;
                hint.textContent = data.message || hint.textContent;
                if (fill) {
                    var pct = data.status === 'queued' ? 15
                        : (data.status === 'running' ? Math.min(80, 25 + elapsed / 600) : 100);
                    fill.style.width = pct + '%';
                }
                if (data.status === 'complete') {
                    clearInterval(timer);
                    if (fill) fill.style.width = '100%';
                    setTimeout(function () { window.location.href = resultUrl; }, 600);
                } else if (data.status === 'error') {
                    clearInterval(timer);
                    hint.textContent = 'Analysis failed: ' + (data.message || 'unknown error');
                } else if (data.status === 'missing') {
                    clearInterval(timer);
                    hint.textContent = 'Analysis is no longer available. Please try again.';
                }
            }).catch(function () {
                /* transient network error — keep polling */
            });
        }, 1500);
    }

    /* ---------------- Init ---------------- */
    document.addEventListener('DOMContentLoaded', function () {
        setupNav();
        setupBackToTop();
        setupFlashes();
        setupTicker();
        setupPasswordFields();
        setupModal();
        setupAnalyzeForm();
        setupForecastChart();
        setupPriceChart();
        setupCompareChart();
        setupEquityChart();
        setupAllocationChart();
        setupPositionPnlChart();
        setupSparklines();
        setupTheme();
        setupAccent();
        setupAnalysisPolling();
        window.addEventListener('resize', function () {
            __chartRegistry.forEach(function (c) { if (c && c.resize) c.resize(); });
        });
    });
})();
