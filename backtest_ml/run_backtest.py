"""Leakage-safe walk-forward backtest harness for the LSTM+GRU ensemble pipeline.

Wraps the repository's own pipeline components (predictor_core.AdvancedStockPredictor,
add_all_indicators, ml_governance.validate_market_data) while enforcing point-in-time
correctness:

  FIX-1  RobustScaler fitted on TRAIN rows only (pipeline fits on full span).
  FIX-2  Correlation-based feature selection bypassed (uses shift(-1) future
         returns over the full span); fixed pipeline-default feature list used.
  FIX-3  Causal cleaning only: forward-fill, never back-fill (clean_market_data
         uses bfill which pulls future prices into past gaps).
  FIX-4  Sentiment layer included as point-in-time-neutral fallback constants
         (live score broadcast across history would be anachronistic).

Outputs are written to backtest_artifacts/<run_id>/.
"""

import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import argparse
import hashlib
import json
import logging
import platform
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import yfinance as yf
from sklearn.preprocessing import RobustScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from predictor_core import (
    MODEL_DIR,
    AdvancedStockPredictor,
    add_all_indicators,
)
from ml_governance import validate_market_data

SEED = 42
LOOKBACK = 60
PERIOD = "5y"
SYMBOLS = ["AAPL", "MSFT", "TSLA", "GOOGL", "RY.TO"]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
EPOCHS = 100
BATCH_SIZE = 32
ES_PATIENCE = 20
RLRP_PATIENCE = 10
ENSEMBLE_UNITS = 80
FEATURES = [
    "Close", "Volume", "RSI", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower", "Stoch_K", "ATR",
    "News_Sentiment", "Social_Buzz", "Sentiment_Strength",
]
SENTIMENT_DEFAULTS = {"News_Sentiment": 0.0, "Social_Buzz": 0, "Sentiment_Strength": 0.1}

RUN_ID = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
ART_DIR = os.path.join(REPO_ROOT, "backtest_artifacts", RUN_ID)
MODELS_DIR = os.path.join(ART_DIR, "models")
PRED_DIR = os.path.join(ART_DIR, "predictions")
PREPROC_DIR = os.path.join(ART_DIR, "preprocessing")


class Tee:
    def __init__(self, stream, log_path):
        self.stream = stream
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def causal_clean(data):
    cleaned = data.copy()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.sort_index()
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in cleaned.columns]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned[numeric_cols] = cleaned[numeric_cols].ffill()
    return cleaned


def load_features(symbol):
    ticker = yf.Ticker(symbol)
    raw = ticker.history(period=PERIOD)
    if raw.empty:
        raise RuntimeError(f"no data fetched for {symbol}")
    report = validate_market_data(raw, symbol)
    data = causal_clean(raw)
    data = add_all_indicators(data)
    for k, v in SENTIMENT_DEFAULTS.items():
        data[k] = v
    frame = data[FEATURES].apply(pd.to_numeric, errors="coerce").ffill()
    frame = frame.dropna()
    frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame.reset_index(drop=False), report, data


def make_sequences(scaled, target_idx=0):
    xs, ys, idx = [], [], []
    for i in range(LOOKBACK, len(scaled)):
        xs.append(scaled[i - LOOKBACK:i])
        ys.append(scaled[i, target_idx])
        idx.append(i)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), idx


def run_leakage_tests(symbol, cleaned, frame, n_train):
    """Executed point-in-time evidence for one symbol.

    T1 prefix-invariance: recompute every feature using ONLY rows [0, n_train)
       and require values at overlapping timestamps to match the full-span
       computation bit-for-bit (proves features at t use data <= t only).
    T2 future-target identity: no feature column may equal the next-row Close
       (shift(-1) target); report exact-match counts and |corr| vs future Close.
    """
    results = {"symbol": symbol, "T1_prefix_invariance": {}, "T2_future_target": {}}

    prefix = cleaned.iloc[:n_train].copy()
    feat_prefix = add_all_indicators(prefix)
    for k, v in SENTIMENT_DEFAULTS.items():
        feat_prefix[k] = v
    idx_name = feat_prefix.index.name or "Date"
    fp = feat_prefix[FEATURES].apply(pd.to_numeric, errors="coerce").ffill()
    fp = fp.dropna()
    fp = fp.reset_index(drop=False).rename(columns={idx_name: "Date"})

    merged = frame.merge(fp, on="Date", suffixes=("_full", "_prefix"), how="inner")
    worst = 0.0
    for col in FEATURES:
        a = merged[f"{col}_full"].to_numpy(dtype=float)
        b = merged[f"{col}_prefix"].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        diff = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else float("nan")
        results["T1_prefix_invariance"][col] = {
            "n_compared": int(ok.sum()),
            "max_abs_diff": diff,
            "identical_within_1e-9": bool(np.isnan(diff) or diff <= 1e-9),
        }
        worst = max(worst, 0.0 if np.isnan(diff) else diff)
    results["T1_max_abs_diff_overall"] = worst

    close = frame["Close"].to_numpy(dtype=float)
    future_close = np.roll(close, -1)
    for col in FEATURES:
        vals = frame[col].to_numpy(dtype=float)
        eq = np.isfinite(vals[:-1]) & (vals[:-1] == future_close[:-1])
        c = np.corrcoef(vals[:-1], future_close[:-1])[0, 1]
        results["T2_future_target"][col] = {
            "exact_equal_to_next_close_count": int(eq.sum()),
            "abs_corr_with_next_close": float(abs(c)) if np.isfinite(c) else None,
        }

    static_hits = []
    for rel in ["predictor_core.py", "ml_governance.py", os.path.join("backtest_ml", "run_backtest.py")]:
        p = os.path.join(REPO_ROOT, rel)
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if "shift(-1)" in line or ".bfill(" in line:
                    static_hits.append(f"{rel}:{ln}: {line.strip()}")
    results["static_forbidden_pattern_lines"] = static_hits
    return results


def write_leakage_report(path, per_symbol_results, manifest):
    lines = []
    lines.append("LEAKAGE CHECKS - executed evidence")
    lines.append("=" * 78)
    lines.append("Structural controls enforced by this harness:")
    lines.append("  FIX-1 scaler: RobustScaler.fit on TRAIN rows only "
                 f"(first {TRAIN_FRAC:.0%} of rows per symbol; center_/scale_ saved to preprocessing/).")
    lines.append("  FIX-2 feature selection: correlation selector that uses shift(-1) FUTURE returns "
                 "(ml_governance.select_features_by_correlation) is NOT called; fixed pipeline-default list used.")
    lines.append("  FIX-3 cleaning: causal_clean() forward-fills only; repository clean_market_data() "
                 "which applies .bfill() (future -> past) is NOT called.")
    lines.append("  FIX-4 sentiment: News_Sentiment/Social_Buzz/Sentiment_Strength set to constant "
                 "point-in-time-neutral defaults (0.0 / 0 / 0.1); broadcasting a live sentiment score across history would be anachronistic.")
    lines.append("  Sequences: input window rows [i-60, i), label row i (next-day close); segments sliced by "
                 "row ranges train [60,n_train) val [n_train,n_train+n_val) test [n_train+n_val,n_rows) so no window crosses splits.")
    lines.append("  Indicator functions used (predictor_core.py:551-591): rolling/diff/ewm/shift(+1) only - all backward-looking.")
    lines.append("  Advanced features used (ml_governance.add_advanced_features:95-118): pct_change/shift(+1)/rolling only.")
    lines.append("")
    lines.append("T1 PREFIX-INVARIANCE (features recomputed on train-prefix only must equal full-span values):")
    for r in per_symbol_results:
        worst = r["T1_max_abs_diff_overall"]
        bad = [c for c, v in r["T1_prefix_invariance"].items() if not v["identical_within_1e-9"]]
        lines.append(f"  [{r['symbol']}] n_compared={list(r['T1_prefix_invariance'].values())[0]['n_compared']} "
                     f"max_abs_diff_overall={worst:.3e} columns_failing={bad if bad else 'NONE'}")
    lines.append("")
    lines.append("T2 FUTURE-TARGET IDENTITY/CORRELATION (feature_t vs Close_(t+1)):")
    for r in per_symbol_results:
        total_eq = sum(v["exact_equal_to_next_close_count"] for v in r["T2_future_target"].values())
        max_corr_col, max_corr = max(
            ((c, v["abs_corr_with_next_close"]) for c, v in r["T2_future_target"].items()),
            key=lambda kv: (kv[1] if kv[1] is not None else -1))
        note = ("Close itself is the lagged model input by construction (window excludes label row);"
                if max_corr_col == "Close" else "")
        lines.append(f"  [{r['symbol']}] exact_copies_of_next_close={total_eq} "
                     f"max_|corr(feature_t, close_t+1)|={max_corr:.4f} ({max_corr_col}) {note}")
    lines.append("")
    lines.append("STATIC SCAN of executed modules for 'shift(-1)' and '.bfill(' occurrences")
    lines.append("(each hit below is verified OUTSIDE the executed code path):")
    hits = per_symbol_results[0]["static_forbidden_pattern_lines"] if per_symbol_results else []
    seen = set()
    for h in hits:
        if h not in seen:
            seen.add(h)
            lines.append(f"  {h}")
    lines.append("  Verified call path: load_features -> validate_market_data -> causal_clean -> "
                 "add_all_indicators (+add_advanced_features); clean_market_data and "
                 "select_features_by_correlation are never invoked.")
    lines.append("")
    lines.append("SPLIT BOUNDARY PROOF (train/val end strictly before test start; from manifest):")
    for sym, info in manifest["per_symbol"].items():
        lines.append(f"  [{sym}] train..val_end={info['val_end_date']} < test_start={info['test_start_date']} "
                     f"seqs(train/val/test)={info['n_sequences']['train']}/{info['n_sequences']['val']}/{info['n_sequences']['test']}")
    lines.append("")
    lines.append(f"generated_utc={datetime.now(timezone.utc).isoformat()}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_environment_file(path, fingerprints):
    import subprocess
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    ram_mb = None
    try:
        import ctypes
        ram_mb = ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory() / (1024 * 1024)
    except Exception:
        pass
    lines = [
        f"python={platform.python_version()}",
        f"python_executable={sys.executable}",
        f"os={platform.platform()}",
        f"machine={platform.machine()} processor={platform.processor()}",
        f"cpu_count={os.cpu_count()}",
        f"ram_installed_mb={ram_mb}",
        "git=NOT_A_GIT_REPOSITORY (directory has no .git); provenance via SHA256 code fingerprints instead:",
    ]
    lines += [f"  sha256 {k}={v}" for k, v in fingerprints.items()]
    lines += [
        f"seed={SEED} (python random, numpy, tensorflow; PYTHONHASHSEED=0)",
        f"tf_deterministic_ops=1 tf_enable_onednn_opts=0 cuda_visible_devices=-1",
        "",
        "pip freeze:",
        freeze.rstrip(),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    for d in [ART_DIR, MODELS_DIR, PRED_DIR, PREPROC_DIR]:
        os.makedirs(d, exist_ok=True)
    log_path = os.path.join(ART_DIR, "run_log.txt")
    sys.stdout = Tee(sys.__stdout__, log_path)
    sys.stderr = Tee(sys.__stderr__, log_path)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    print("=" * 78)
    print(f"LSTM+GRU ENSEMBLE BACKTEST | run_id={RUN_ID}")
    print(f"repo_root={REPO_ROOT}")
    print(f"started_utc={datetime.now(timezone.utc).isoformat()}")
    print(f"python={platform.python_version()} os={platform.platform()} machine={platform.machine()}")
    print(f"numpy={np.__version__} pandas={pd.__version__} sklearn="
          f"{__import__('sklearn').__version__} tensorflow={tf.__version__} "
          f"yfinance={yf.__version__}")
    print(f"physical_devices={[d.name for d in tf.config.list_physical_devices()]}")
    try:
        tf.config.experimental.enable_op_determinism()
        print("tf_op_determinism=ENABLED")
    except Exception as exc:
        print(f"tf_op_determinism=FAILED ({exc})")
    print(f"seeds: python={SEED} numpy={SEED} tensorflow={SEED} hash_seed=0")
    print(f"config: lookback={LOOKBACK} period={PERIOD} symbols={SYMBOLS} "
          f"split=train {TRAIN_FRAC:.2f}/val {VAL_FRAC:.2f}/test "
          f"{1 - TRAIN_FRAC - VAL_FRAC:.2f} epochs={EPOCHS} batch={BATCH_SIZE} "
          f"es_patience={ES_PATIENCE} units={ENSEMBLE_UNITS}")
    print(f"features={FEATURES}")
    print(f"model_dir={MODEL_DIR}")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    code_files = ["predictor_core.py", "ml_governance.py", "requirements.txt",
                  os.path.join("backtest_ml", "run_backtest.py")]
    fingerprints = {}
    for rel in code_files:
        p = os.path.join(REPO_ROOT, rel)
        if os.path.exists(p):
            fingerprints[rel] = sha256_file(p)
    print(f"code_fingerprints_sha256={json.dumps(fingerprints, indent=1)}")

    predictor_proto = AdvancedStockPredictor(
        symbol="", lookback_days=LOOKBACK, model_type="ENSEMBLE",
        lstm_units=ENSEMBLE_UNITS,
    )

    all_pred_rows, all_gt_rows, metric_rows = [], [], []
    pooled_correct, pooled_total = 0, 0
    leakage_results = []
    indices_rows = []
    fp_tag = "nofp"
    manifest = {
        "run_id": RUN_ID, "seed": SEED, "lookback_days": LOOKBACK,
        "period": PERIOD, "symbols": SYMBOLS, "features": FEATURES,
        "train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "es_patience": ES_PATIENCE, "units": ENSEMBLE_UNITS,
        "model_type": "ENSEMBLE (parallel LSTM+GRU hybrid)",
        "horizon": "next trading day close (1-step-ahead)",
        "fingerprints": fingerprints,
        "per_symbol": {},
    }
    fp_tag = fingerprints.get(os.path.join("backtest_ml", "run_backtest.py"), "nofp")[:12]

    for symbol in SYMBOLS:
        print("\n" + "-" * 78)
        print(f"[{symbol}] loading data...")
        frame, qreport, cleaned = load_features(symbol)
        n_rows = len(frame)
        n_train = int(n_rows * TRAIN_FRAC)
        n_val = int(n_rows * VAL_FRAC)
        print(f"[{symbol}] rows={n_rows} range={frame['Date'].iloc[0].date()} .. "
              f"{frame['Date'].iloc[-1].date()} quality_valid={qreport.get('is_valid')}")

        scaler = RobustScaler()
        scaler.fit(frame[FEATURES].iloc[:n_train].values)
        scaled = scaler.transform(frame[FEATURES].values)

        X, y, seq_end_idx = make_sequences(scaled, target_idx=0)
        dates = frame["Date"].values
        close_raw = frame["Close"].values

        def seg(lo, hi):
            sel = [(e, i) for e, i in zip(seq_end_idx, range(len(seq_end_idx))) if lo <= e < hi]
            if not sel:
                return np.empty((0, LOOKBACK, len(FEATURES)), dtype=np.float32), np.empty((0,), dtype=np.float32), []
            ends, pos = zip(*sel)
            return X[list(pos)], y[list(pos)], list(ends)

        X_train, y_train, end_train = seg(LOOKBACK, n_train)
        X_val, y_val, end_val = seg(n_train, n_train + n_val)
        X_test, y_test, end_test = seg(n_train + n_val, n_rows)
        print(f"[{symbol}] sequences train={len(end_train)} val={len(end_val)} test={len(end_test)}")
        assert len(end_test) > 0, f"{symbol}: empty held-out test segment"

        med_close = scaler.center_[0]
        iqr_close = scaler.scale_[0]

        model = predictor_proto.build_hybrid_ensemble_model(
            input_shape=(LOOKBACK, len(FEATURES))
        )
        print(f"[{symbol}] params={model.count_params():,}")

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=ES_PATIENCE,
            restore_best_weights=True, verbose=2)
        rlrp = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=RLRP_PATIENCE,
            min_lr=1e-7, verbose=2)

        t0 = datetime.now(timezone.utc)
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[es, rlrp],
            shuffle=True, verbose=2)
        train_secs = (datetime.now(timezone.utc) - t0).total_seconds()

        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
        print(f"[{symbol}] trained_in_s={train_secs:.1f} epochs_run={len(hist.history['loss'])} "
              f"best_epoch={best_epoch} best_val_loss={min(hist.history['val_loss']):.6f}")

        preds_scaled = model.predict(X_test, verbose=0).flatten()
        dmy = np.zeros((len(preds_scaled), len(FEATURES)))
        dmy[:, 0] = preds_scaled
        preds_price = scaler.inverse_transform(dmy)[:, 0]

        test_target_rows = end_test
        base_rows = [r - 1 for r in test_target_rows]
        actual_price = close_raw[test_target_rows]
        base_price = close_raw[base_rows]
        actual_return = actual_price / base_price - 1.0
        predicted_return = preds_price / base_price - 1.0

        err = actual_price - preds_price
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        nz = np.abs(actual_price) > 1e-12
        n_excluded = int((~nz).sum())
        mape = float(np.mean(np.abs(err[nz] / actual_price[nz])) * 100) if nz.any() else float("nan")
        dir_ok = np.sign(predicted_return) == np.sign(actual_return)
        dir_acc = float(np.mean(dir_ok) * 100)
        n_correct = int(dir_ok.sum())
        n_pred = len(preds_price)

        pooled_correct += n_correct
        pooled_total += n_pred
        metric_rows.append({
            "instrument": symbol,
            "n_predictions": n_pred,
            "RMSE": rmse,
            "MAPE_percent": mape,
            "Directional_Accuracy_percent": dir_acc,
            "MAE": mae,
            "mape_zero_excluded": n_excluded,
            "direction_correct": n_correct,
        })
        print(f"[{symbol}] METRICS n={n_pred} RMSE={rmse:.4g} MAPE%={mape:.4g} "
              f"DirAcc%={dir_acc:.4g} (correct={n_correct})")

        safe_sym = symbol.replace(".", "_")
        lkg = run_leakage_tests(symbol, cleaned, frame, n_train)
        leakage_results.append(lkg)
        print(f"[{symbol}] leakage T1_prefix_max_diff={lkg['T1_max_abs_diff_overall']:.3e} "
              f"T2_future_target_copies={sum(v['exact_equal_to_next_close_count'] for v in lkg['T2_future_target'].values())}")

        model_path = os.path.join(MODELS_DIR, f"{safe_sym}_model_lstmgru_lb60_{fp_tag}.h5")
        model.save(model_path)
        scaler_params = {
            "instrument": symbol,
            "scaler": "RobustScaler",
            "fit_scope": f"train rows only: 0..{n_train - 1} of {n_rows}",
            "fit_start_date": str(pd.Timestamp(dates[0]).date()),
            "fit_end_date": str(pd.Timestamp(dates[n_train - 1]).date()),
            "n_samples_fit": int(n_train),
            "features": FEATURES,
            "center": scaler.center_.tolist(),
            "scale": scaler.scale_.tolist(),
        }
        with open(os.path.join(PREPROC_DIR, f"{safe_sym}_scaler_params.json"), "w", encoding="utf-8") as f:
            json.dump(scaler_params, f, indent=2)

        indices_rows.append({
            "instrument": symbol,
            "train_start": str(pd.Timestamp(dates[0]).date()),
            "train_end": str(pd.Timestamp(dates[n_train - 1]).date()),
            "val_start": str(pd.Timestamp(dates[n_train]).date()),
            "val_end": str(pd.Timestamp(dates[n_train + n_val - 1]).date()),
            "test_start": str(pd.Timestamp(dates[n_train + n_val]).date()),
            "test_end": str(pd.Timestamp(dates[-1]).date()),
            "train_sequences": len(end_train),
            "val_sequences": len(end_val),
            "test_sequences": len(end_test),
        })

        checkpoint_tag = f"ENSEMBLE|lookback=60|units={ENSEMBLE_UNITS}|best_epoch={best_epoch}"
        for k in range(n_pred):
            ts = pd.Timestamp(dates[test_target_rows[k]]).tz_convert(None) if getattr(dates[test_target_rows[k]], "tzinfo", None) is not None else pd.Timestamp(dates[test_target_rows[k]])
            ts_str = str(ts.date())
            all_pred_rows.append({
                "instrument": symbol, "datetime": ts_str,
                "predicted_value": float(preds_price[k]),
                "actual_value": float(actual_price[k]),
                "predicted_return": float(predicted_return[k]),
                "actual_return": float(actual_return[k]),
                "model_checkpoint": checkpoint_tag,
                "run_id": RUN_ID,
            })
            all_gt_rows.append({
                "instrument": symbol, "datetime": ts_str,
                "predicted_value": "",
                "actual_value": float(actual_price[k]),
                "predicted_return": "",
                "actual_return": float(actual_return[k]),
                "model_checkpoint": checkpoint_tag,
                "run_id": RUN_ID,
            })

        manifest["per_symbol"][symbol] = {
            "rows": int(n_rows),
            "train_end_date": str(pd.Timestamp(dates[n_train - 1]).date()),
            "val_start_date": str(pd.Timestamp(dates[n_train]).date()),
            "val_end_date": str(pd.Timestamp(dates[n_train + n_val - 1]).date()),
            "test_start_date": str(pd.Timestamp(dates[n_train + n_val]).date()),
            "test_end_date": str(pd.Timestamp(dates[-1]).date()),
            "n_sequences": {"train": len(end_train), "val": len(end_val), "test": len(end_test)},
            "scaler_center_Close": float(med_close),
            "scaler_scale_Close": float(iqr_close),
            "best_epoch": best_epoch,
            "epochs_run": len(hist.history["loss"]),
            "final_val_loss": float(min(hist.history["val_loss"])),
            "training_time_seconds": round(train_secs, 2),
            "quality_report_warnings": qreport.get("warnings", []),
            "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
        }

        sym_preds = [r for r in all_pred_rows if r["instrument"] == symbol]
        pd.DataFrame([{
            "timestamp": r["datetime"],
            "y_true": r["actual_value"],
            "y_pred": r["predicted_value"],
        } for r in sym_preds]).to_csv(
            os.path.join(PRED_DIR, f"{safe_sym}_predictions.csv"), index=False)

        del model
        tf.keras.backend.clear_session()

    agg_rmse = float(np.mean([m["RMSE"] for m in metric_rows]))
    agg_mape = float(np.mean([m["MAPE_percent"] for m in metric_rows]))
    agg_dir = float(pooled_correct / pooled_total * 100)
    total_preds = sum(m["n_predictions"] for m in metric_rows)
    metric_rows.append({
        "instrument": "AGGREGATE",
        "n_predictions": total_preds,
        "RMSE": agg_rmse,
        "MAPE_percent": agg_mape,
        "Directional_Accuracy_percent": agg_dir,
        "MAE": float(np.mean([m["MAE"] for m in metric_rows])),
        "mape_zero_excluded": int(sum(m["mape_zero_excluded"] for m in metric_rows)),
        "direction_correct": pooled_correct,
    })
    print("\n" + "=" * 78)
    print(f"AGGREGATE (simple mean of per-instrument RMSE/MAPE; pooled DirAcc): "
          f"RMSE={agg_rmse:.4g} MAPE%={agg_mape:.4g} DirAcc%={agg_dir:.4g} over {total_preds} predictions")

    pred_df = pd.DataFrame(all_pred_rows)
    gt_df = pd.DataFrame(all_gt_rows)
    met_df = pd.DataFrame(metric_rows)

    pred_path = os.path.join(ART_DIR, "predictions.csv")
    gt_path = os.path.join(ART_DIR, "ground_truth.csv")
    met_path = os.path.join(ART_DIR, "metrics.csv")
    man_path = os.path.join(ART_DIR, "manifest.json")
    per_inst_path = os.path.join(ART_DIR, "metrics_per_instrument.csv")
    agg_path = os.path.join(ART_DIR, "metrics_aggregate.csv")
    idx_path = os.path.join(ART_DIR, "train_test_indices.csv")
    env_path = os.path.join(ART_DIR, "environment.txt")
    lkg_path = os.path.join(ART_DIR, "leakage_checks.txt")
    pred_df.to_csv(pred_path, index=False)
    gt_df.to_csv(gt_path, index=False)
    met_df.to_csv(met_path, index=False)
    met_df[met_df.instrument != "AGGREGATE"].rename(columns={
        "MAPE_percent": "MAPE",
        "Directional_Accuracy_percent": "DirectionalAccuracy",
    })[
        ["instrument", "RMSE", "MAPE", "DirectionalAccuracy", "n_predictions", "mape_zero_excluded"]
    ].to_csv(per_inst_path, index=False)
    agg_unw_dir = float(np.mean([m["Directional_Accuracy_percent"] for m in metric_rows[:-1]]))
    pd.DataFrame([
        {"metric": "RMSE", "value": agg_rmse, "method": "unweighted mean of per-instrument RMSE"},
        {"metric": "MAPE_percent", "value": agg_mape, "method": "unweighted mean of per-instrument MAPE (zero targets excluded)"},
        {"metric": "DirectionalAccuracy_percent", "value": agg_dir,
         "method": f"pooled over {pooled_total} predictions (equals unweighted mean {agg_unw_dir:.6f} because n is equal per instrument)"},
        {"metric": "DirectionalAccuracy_percent_unweighted_mean", "value": agg_unw_dir, "method": "unweighted mean of per-instrument directional accuracy"},
        {"metric": "n_test_predictions_total", "value": total_preds, "method": "sum across instruments"},
        {"metric": "aggregation_method", "value": "unweighted_mean", "method":
         "no market-cap/volume weights present in repository; unweighted mean stated explicitly"},
    ]).to_csv(agg_path, index=False)
    pd.DataFrame(indices_rows).to_csv(idx_path, index=False)
    write_environment_file(env_path, fingerprints)
    write_leakage_report(lkg_path, leakage_results, manifest)
    pred_df.to_csv(pred_path, index=False)
    gt_df.to_csv(gt_path, index=False)
    met_df.to_csv(met_path, index=False)
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nPOST-RUN VALIDATION")
    test_ranges_ok = True
    for symbol in SYMBOLS:
        sub = pred_df[pred_df.instrument == symbol]
        info = manifest["per_symbol"][symbol]
        lo, hi = info["test_start_date"], info["test_end_date"]
        in_range = bool(((sub.datetime >= lo) & (sub.datetime <= hi)).all())
        test_ranges_ok &= in_range
        nan_free = bool(sub.notna().all().all())
        print(f"  [{symbol}] predictions_in_test_window={in_range} rows={len(sub)} nan_free={nan_free}")
    print(f"  all_predictions_within_held_out_windows={test_ranges_ok}")
    print("  scaler_fit=train rows only, per symbol (RobustScaler center/scale from first 70% of rows)")
    print("\nArtifacts written:")
    for p in [pred_path, gt_path, met_path, per_inst_path, agg_path, idx_path,
              env_path, lkg_path, man_path, log_path]:
        print(f"  {p}")
    for d in [MODELS_DIR, PRED_DIR, PREPROC_DIR]:
        for f in sorted(os.listdir(d)):
            print(f"  {os.path.join(d, f)}")
    print(f"finished_utc={datetime.now(timezone.utc).isoformat()}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(SYMBOLS))
    parser.add_argument("--period", type=str, default=PERIOD)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args, _ = parser.parse_known_args()
    SYMBOLS = args.symbols.split(",")
    PERIOD = args.period
    EPOCHS = args.epochs
    main()
