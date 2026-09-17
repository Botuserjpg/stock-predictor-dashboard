"""Multi-lookback comparison experiment for the LSTM+GRU ensemble pipeline.

Runs the repository's EXISTING single-lookback backtest methodology
(backtest_ml/run_backtest.py) over a grid of LOOKBACK values. The ONLY
experimental variable changed is the lookback window length. Everything else is
imported/reused unmodified from the existing harness:

  - features (12 fixed), sentiment neutral constants,
  - model: AdvancedStockPredictor.build_hybrid_ensemble_model (LSTM+GRU hybrid),
  - Scaling: RobustScaler fitted on TRAIN rows only (leakage FIX-1),
  - Cleaning: causal_clean() forward-fill only (no bfill; FIX-3),
  - Correlation feature selector bypassed (FIX-2), sentiment constants (FIX-4),
  - Splits: chronological 70/15/15 (train/val/test), sequences sliced by row
    ranges so no window crosses a split,
  - Optimizer Adam(lr=5e-4, clipnorm=1.0), loss=huber_loss,
    EarlyStopping(patience=20, restore_best_weights), ReduceLROnPlateau(0.5, patience=10),
    batch=32, max epochs=100,
  - Determinism settings identical to run_backtest.py (PYTHONHASHSEED=0,
    TF_DETERMINISTIC_OPS=1, CPU-only, single-thread, seed 42,
    tf.config.experimental.enable_op_determinism()).

run_backtest.py is NOT modified. Results are written to
backtest_artifacts/lookback_experiment/lb<lookback>/... plus a master
comparison_summary.csv / comparison_results.json.

Usage:
  python backtest_ml/run_lookback_experiment.py [--lookback 60]
                                               [--symbols AAPL,MSFT,...]
                                               [--epochs 100]
                                               [--out-suffix]
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
import json
import logging
import platform
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the existing single-lookback harness pieces (run_backtest.py is NOT run,
# only imported; its __main__ guard prevents execution).
from run_backtest import (  # noqa: E402
    REPO_ROOT,
    SYMBOLS as DEFAULT_SYMBOLS,
    PERIOD,
    TRAIN_FRAC,
    VAL_FRAC,
    EPOCHS as DEFAULT_EPOCHS,
    BATCH_SIZE,
    ES_PATIENCE,
    RLRP_PATIENCE,
    ENSEMBLE_UNITS,
    FEATURES,
    SENTIMENT_DEFAULTS,
    SEED,
    load_features,
    run_leakage_tests,
    sha256_file,
    Tee,
    write_environment_file,
)
from predictor_core import MODEL_DIR, AdvancedStockPredictor, add_all_indicators  # noqa: E402
from ml_governance import validate_market_data  # noqa: E402

LOOKBACKS = [20, 40, 60, 90, 120, 252, 365]
EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "backtest_artifacts", "lookback_experiment")


def make_sequences_lb(scaled, lookback, target_idx=0):
    """Identical logic to run_backtest.make_sequences, but lookback is a parameter."""
    xs, ys, idx = [], [], []
    for i in range(lookback, len(scaled)):
        xs.append(scaled[i - lookback:i])
        ys.append(scaled[i, target_idx])
        idx.append(i)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), idx


def seg_fn(X, y, seq_end_idx, lo, hi, lookback):
    """Identical split logic to run_backtest.main()'s seg(), lookback parameterized."""
    sel = [(e, i) for e, i in zip(seq_end_idx, range(len(seq_end_idx))) if lo <= e < hi]
    if not sel:
        return np.empty((0, lookback, len(FEATURES)), dtype=np.float32), np.empty((0,), dtype=np.float32), []
    ends, pos = zip(*sel)
    return X[list(pos)], y[list(pos)], list(ends)


def write_leakage_report_lb(path, per_symbol_results, lookback, manifest):
    lines = [
        "LEAKAGE CHECKS - executed evidence (multi-lookback experiment)",
        "=" * 78,
        f"lookback_days={lookback}",
        "Structural controls enforced by this harness (unchanged from run_backtest.py):",
        f"  FIX-1 scaler: RobustScaler.fit on TRAIN rows only (first {TRAIN_FRAC:.0%} of rows per symbol).",
        "  FIX-2 feature selection: correlation selector using shift(-1) FUTURE returns is NOT called.",
        "  FIX-3 cleaning: causal_clean() forward-fills only; clean_market_data() .bfill() never invoked.",
        "  FIX-4 sentiment: constant point-in-time-neutral defaults (0.0 / 0 / 0.1).",
        f"  Sequences: input window rows [i-{lookback}, i), label row i (next-day close); "
        f"segments sliced by row ranges so no window crosses splits.",
        "",
        "T1 PREFIX-INVARIANCE (features recomputed on train-prefix only must equal full-span values):",
    ]
    for r in per_symbol_results:
        worst = r["T1_max_abs_diff_overall"]
        bad = [c for c, v in r["T1_prefix_invariance"].items() if not v["identical_within_1e-9"]]
        lines.append(
            f"  [{r['symbol']}] n_compared={list(r['T1_prefix_invariance'].values())[0]['n_compared']} "
            f"max_abs_diff_overall={worst:.3e} columns_failing={bad if bad else 'NONE'}"
        )
    lines.append("")
    lines.append("T2 FUTURE-TARGET IDENTITY/CORRELATION (feature_t vs Close_(t+1)):")
    for r in per_symbol_results:
        total_eq = sum(v["exact_equal_to_next_close_count"] for v in r["T2_future_target"].values())
        max_corr_col, max_corr = max(
            ((c, v["abs_corr_with_next_close"]) for c, v in r["T2_future_target"].items()),
            key=lambda kv: (kv[1] if kv[1] is not None else -1),
        )
        note = ("Close itself is the lagged model input by construction (window excludes label row);"
                if max_corr_col == "Close" else "")
        lines.append(
            f"  [{r['symbol']}] exact_copies_of_next_close={total_eq} "
            f"max_|corr(feature_t, close_t+1)|={max_corr:.4f} ({max_corr_col}) {note}"
        )
    lines.append("")
    lines.append("SPLIT BOUNDARY PROOF (train/val end strictly before test start; from manifest):")
    for sym, info in manifest["per_symbol"].items():
        lines.append(
            f"  [{sym}] train..val_end={info['val_end_date']} < test_start={info['test_start_date']} "
            f"seqs(train/val/test)={info['n_sequences']['train']}/{info['n_sequences']['val']}/{info['n_sequences']['test']}"
        )
    lines.append("")
    lines.append(f"generated_utc={datetime.now(timezone.utc).isoformat()}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_fingerprints():
    code_files = [
        "predictor_core.py",
        "ml_governance.py",
        "requirements.txt",
        os.path.join("backtest_ml", "run_backtest.py"),
        os.path.join("backtest_ml", "run_lookback_experiment.py"),
    ]
    fingerprints = {}
    for rel in code_files:
        p = os.path.join(REPO_ROOT, rel)
        if os.path.exists(p):
            fingerprints[rel] = sha256_file(p)
    return fingerprints


def complete_flag_path(run_dir):
    return os.path.join(run_dir, "COMPLETE")


def run_one_lookback(lookback, symbols, epochs, out_suffix=""):
    run_id = f"lb{lookback}{out_suffix}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    art_dir = os.path.join(EXPERIMENT_ROOT, run_id)
    models_dir = os.path.join(art_dir, "models")
    pred_dir = os.path.join(art_dir, "predictions")
    preproc_dir = os.path.join(art_dir, "preprocessing")
    for d in [art_dir, models_dir, pred_dir, preproc_dir]:
        os.makedirs(d, exist_ok=True)

    if os.path.exists(complete_flag_path(art_dir)):
        print(f"[{lookback}] already complete at {art_dir} - skipping", flush=True)
        return art_dir

    log_path = os.path.join(art_dir, "run_log.txt")
    sys.stdout = Tee(sys.__stdout__, log_path)
    sys.stderr = Tee(sys.__stderr__, log_path)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    print("=" * 78)
    print(f"LOOKBACK EXPERIMENT | lookback={lookback} | run_id={run_id}")
    print(f"repo_root={REPO_ROOT}")
    print(f"started_utc={datetime.now(timezone.utc).isoformat()}")
    print(f"python={platform.python_version()} os={platform.platform()} machine={platform.machine()}")
    print(f"numpy={np.__version__} pandas={pd.__version__} sklearn={sklearn.__version__} "
          f"tensorflow={tf.__version__}")
    print(f"seeds: python={SEED} numpy={SEED} tensorflow={SEED} hash_seed=0")
    print(f"config: lookback={lookback} period={PERIOD} symbols={symbols} "
          f"split=train {TRAIN_FRAC:.2f}/val {VAL_FRAC:.2f}/test {1 - TRAIN_FRAC - VAL_FRAC:.2f} "
          f"epochs={epochs} batch={BATCH_SIZE} es_patience={ES_PATIENCE} units={ENSEMBLE_UNITS}")
    print(f"features={FEATURES}")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
        print("tf_op_determinism=ENABLED")
    except Exception as exc:
        print(f"tf_op_determinism=FAILED ({exc})")

    fingerprints = build_fingerprints()
    print(f"code_fingerprints_sha256={json.dumps(fingerprints, indent=1)}")

    if lookback >= int(0.70 * 1256):  # practical guard: window can't exceed train rows
        print(f"[{lookback}] WARNING: lookback is large relative to typical train rows")

    all_pred_rows, all_gt_rows, metric_rows = [], [], []
    pooled_correct, pooled_total = 0, 0
    leakage_results = []
    indices_rows = []
    manifest = {
        "run_id": run_id,
        "seed": SEED,
        "lookback_days": lookback,
        "period": PERIOD,
        "symbols": symbols,
        "features": FEATURES,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "es_patience": ES_PATIENCE,
        "units": ENSEMBLE_UNITS,
        "model_type": "ENSEMBLE (parallel LSTM+GRU hybrid)",
        "horizon": "next trading day close (1-step-ahead)",
        "fingerprints": fingerprints,
        "per_symbol": {},
    }

    for symbol in symbols:
        print("\n" + "-" * 78, flush=True)
        print(f"[{symbol}] lookback={lookback} loading data...", flush=True)
        frame, qreport, cleaned = load_features(symbol)
        n_rows = len(frame)
        n_train = int(n_rows * TRAIN_FRAC)
        n_val = int(n_rows * VAL_FRAC)
        print(f"[{symbol}] rows={n_rows} range={frame['Date'].iloc[0].date()} .. "
              f"{frame['Date'].iloc[-1].date()} quality_valid={qreport.get('is_valid')}", flush=True)

        scaler = RobustScalerHolder()(frame, FEATURES, n_train)
        scaled = scaler.transform(frame[FEATURES].values)

        X, y, seq_end_idx = make_sequences_lb(scaled, lookback, target_idx=0)
        dates = frame["Date"].values
        close_raw = frame["Close"].values

        X_train, y_train, end_train = seg_fn(X, y, seq_end_idx, lookback, n_train, lookback)
        X_val, y_val, end_val = seg_fn(X, y, seq_end_idx, n_train, n_train + n_val, lookback)
        X_test, y_test, end_test = seg_fn(X, y, seq_end_idx, n_train + n_val, n_rows, lookback)
        print(f"[{symbol}] sequences train={len(end_train)} val={len(end_val)} test={len(end_test)}", flush=True)
        assert len(end_test) > 0, f"{symbol}: empty held-out test segment for lookback {lookback}"
        assert len(end_train) > 0, f"{symbol}: empty train segment for lookback {lookback}"

        med_close = scaler.center_[0]
        iqr_close = scaler.scale_[0]

        predictor = AdvancedStockPredictor(
            symbol="", lookback_days=lookback, model_type="ENSEMBLE", lstm_units=ENSEMBLE_UNITS,
        )
        model = predictor.build_hybrid_ensemble_model(input_shape=(lookback, len(FEATURES)))
        print(f"[{symbol}] params={model.count_params():,}", flush=True)
        print(f"[{symbol}] train_samples={len(end_train)} X_train.shape={X_train.shape}", flush=True)

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True, verbose=2)
        rlrp = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=RLRP_PATIENCE, min_lr=1e-7, verbose=2)

        t0 = datetime.now(timezone.utc)
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=BATCH_SIZE,
            callbacks=[es, rlrp],
            shuffle=True, verbose=2, )
        train_secs = (datetime.now(timezone.utc) - t0).total_seconds()

        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
        print(f"[{symbol}] trained_in_s={train_secs:.1f} epochs_run={len(hist.history['loss'])} "
              f"best_epoch={best_epoch} best_val_loss={min(hist.history['val_loss']):.6f}", flush=True)

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
        print(f"[{symbol}] METRICS n={n_pred} RMSE={rmse:.4g} MAE={mae:.4g} MAPE%={mape:.4g} "
              f"DirAcc%={dir_acc:.4g} (correct={n_correct})", flush=True)

        safe_sym = symbol.replace(".", "_")
        lkg = run_leakage_tests(symbol, cleaned, frame, n_train)
        leakage_results.append(lkg)
        print(f"[{symbol}] leakage T1_prefix_max_diff={lkg['T1_max_abs_diff_overall']:.3e} "
              f"T2_future_target_copies={sum(v['exact_equal_to_next_close_count'] for v in lkg['T2_future_target'].values())}", flush=True)

        model_path = os.path.join(models_dir, f"{safe_sym}_model_lstmgru_lb{lookback}_{run_id}.h5")
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
        with open(os.path.join(preproc_dir, f"{safe_sym}_scaler_params.json"), "w", encoding="utf-8") as f:
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
            "window_size_days": lookback,
        })

        checkpoint_tag = f"ENSEMBLE|lookback={lookback}|units={ENSEMBLE_UNITS}|best_epoch={best_epoch}"
        for k in range(n_pred):
            ts = pd.Timestamp(dates[test_target_rows[k]])
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert(None)
            ts_str = str(ts.date())
            all_pred_rows.append({
                "instrument": symbol, "datetime": ts_str,
                "predicted_value": float(preds_price[k]),
                "actual_value": float(actual_price[k]),
                "predicted_return": float(predicted_return[k]),
                "actual_return": float(actual_return[k]),
                "model_checkpoint": checkpoint_tag,
                "run_id": run_id,
            })
            all_gt_rows.append({
                "instrument": symbol, "datetime": ts_str,
                "predicted_value": "",
                "actual_value": float(actual_price[k]),
                "predicted_return": "",
                "actual_return": float(actual_return[k]),
                "model_checkpoint": checkpoint_tag,
                "run_id": run_id,
            })

        manifest["per_symbol"][symbol] = {
            "rows": int(n_rows),
            "train_end_date": str(pd.Timestamp(dates[n_train - 1]).date()),
            "val_start_date": str(pd.Timestamp(dates[n_train]).date()),
            "val_end_date": str(pd.Timestamp(dates[n_train + n_val - 1]).date()),
            "test_start_date": str(pd.Timestamp(dates[n_train + n_val]).date()),
            "test_end_date": str(pd.Timestamp(dates[-1]).date()),
            "n_sequences": {"train": len(end_train), "val": len(end_val), "test": len(end_test)},
            "window_size_days": lookback,
            "scaler_center_Close": float(med_close),
            "scaler_scale_Close": float(iqr_close),
            "best_epoch": best_epoch,
            "epochs_run": len(hist.history["loss"]),
            "final_val_loss": float(min(hist.history["val_loss"])),
            "training_time_seconds": round(train_secs, 2),
            "quality_report_warnings": qreport.get("warnings", []),
        }

        del model
        tf.keras.backend.clear_session()

    agg_rmse = float(np.mean([m["RMSE"] for m in metric_rows]))
    agg_mape = float(np.mean([m["MAPE_percent"] for m in metric_rows]))
    agg_mae = float(np.mean([m["MAE"] for m in metric_rows]))
    agg_dir = float(pooled_correct / pooled_total * 100)
    total_preds = sum(m["n_predictions"] for m in metric_rows)
    metric_rows.append({
        "instrument": "AGGREGATE",
        "n_predictions": total_preds,
        "RMSE": agg_rmse,
        "MAPE_percent": agg_mape,
        "Directional_Accuracy_percent": agg_dir,
        "MAE": agg_mae,
        "mape_zero_excluded": int(sum(m["mape_zero_excluded"] for m in metric_rows)),
        "direction_correct": pooled_correct,
    })
    total_train_s = sum(m["training_time_seconds"] for m in manifest["per_symbol"].values())
    print("\n" + "=" * 78)
    print(f"AGGREGATE (simple mean of per-instrument RMSE/MAPE/MAE; pooled DirAcc): "
          f"RMSE={agg_rmse:.4g} MAPE%={agg_mape:.4g} DirAcc%={agg_dir:.4g} over {total_preds} predictions", flush=True)

    pred_df = pd.DataFrame(all_pred_rows)
    gt_df = pd.DataFrame(all_gt_rows)
    met_df = pd.DataFrame(metric_rows)

    pred_path = os.path.join(art_dir, "predictions.csv")
    gt_path = os.path.join(art_dir, "ground_truth.csv")
    met_path = os.path.join(art_dir, "metrics.csv")
    per_inst_path = os.path.join(art_dir, "metrics_per_instrument.csv")
    agg_path = os.path.join(art_dir, "metrics_aggregate.csv")
    idx_path = os.path.join(art_dir, "train_test_indices.csv")
    env_path = os.path.join(art_dir, "environment.txt")
    lkg_path = os.path.join(art_dir, "leakage_checks.txt")
    man_path = os.path.join(art_dir, "manifest.json")
    pred_df.to_csv(pred_path, index=False)
    gt_df.to_csv(gt_path, index=False)
    met_df.to_csv(met_path, index=False)
    met_df[met_df.instrument != "AGGREGATE"].rename(columns={
        "MAPE_percent": "MAPE",
        "Directional_Accuracy_percent": "DirectionalAccuracy",
    })[
        ["instrument", "RMSE", "MAPE", "DirectionalAccuracy", "n_predictions", "mape_zero_excluded"]
    ].to_csv(per_inst_path, index=False)
    pd.DataFrame([
        {"metric": "RMSE", "value": agg_rmse, "method": "unweighted mean of per-instrument RMSE"},
        {"metric": "MAE", "value": agg_mae, "method": "unweighted mean of per-instrument MAE"},
        {"metric": "MAPE_percent", "value": agg_mape, "method": "unweighted mean of per-instrument MAPE (zero targets excluded)"},
        {"metric": "DirectionalAccuracy_percent", "value": agg_dir,
         "method": f"pooled over {pooled_total} predictions ({pooled_correct} correct)"},
        {"metric": "n_test_predictions_total", "value": total_preds, "method": "sum across instruments"},
        {"metric": "lookback_days", "value": lookback, "method": "experiment variable"},
        {"metric": "total_training_time_seconds", "value": round(total_train_s, 2), "method": "sum across instruments"},
        {"metric": "aggregation_method", "value": "unweighted_mean", "method": "no market-cap/volume weights present in repository"},
    ]).to_csv(agg_path, index=False)
    pd.DataFrame(indices_rows).to_csv(idx_path, index=False)
    write_environment_file(env_path, fingerprints)
    write_leakage_report_lb(lkg_path, leakage_results, lookback, manifest)
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nPOST-RUN VALIDATION")
    test_ranges_ok = True
    for symbol in symbols:
        sub = pred_df[pred_df.instrument == symbol]
        info = manifest["per_symbol"][symbol]
        lo, hi = info["test_start_date"], info["test_end_date"]
        in_range = bool(((sub.datetime >= lo) & (sub.datetime <= hi)).all())
        test_ranges_ok &= in_range
        nan_free = bool(sub.notna().all().all())
        print(f"  [{symbol}] predictions_in_test_window={in_range} rows={len(sub)} nan_free={nan_free}")
    print(f"  all_predictions_within_held_out_windows={test_ranges_ok}")
    print(f"  scaler_fit=train rows only, per symbol (RobustScaler center/scale from first {TRAIN_FRAC:.0%} of rows)")
    print("Artifacts written:")
    for p in [pred_path, gt_path, met_path, per_inst_path, agg_path, idx_path,
              env_path, lkg_path, man_path, log_path]:
        print(f"  {p}")

    with open(complete_flag_path(art_dir), "w", encoding="utf-8") as f:
        f.write(f"lookback={lookback} completed\n")
    print(f"finished_utc={datetime.now(timezone.utc).isoformat()}")
    print("=" * 78, flush=True)
    return art_dir


class RobustScalerHolder:
    """Simple helper preserving the exact RobustScaler train-only fit used by run_backtest.py."""

    def __call__(self, frame, features, n_train):
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(frame[features].iloc[:n_train].values)
        return scaler


def save_comparison_files(results):
    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
    per_inst = []
    for r in results:
        for row in r["per_instrument"]:
            per_inst.append(row)
    per_lb = []
    for r in results:
        per_lb.append(r["aggregate"])
    per_inst_df = pd.DataFrame(per_inst)
    per_lb_df = pd.DataFrame(per_lb)
    per_inst_csv = os.path.join(EXPERIMENT_ROOT, "comparison_summary.csv")
    per_lb_csv = os.path.join(EXPERIMENT_ROOT, "comparison_aggregate.csv")
    json_path = os.path.join(EXPERIMENT_ROOT, "comparison_results.json")
    per_inst_df.to_csv(per_inst_csv, index=False)
    per_lb_df.to_csv(per_lb_csv, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"per_instrument": per_inst, "aggregate_by_lookback": per_lb}, f, indent=2, sort_keys=True)
    print(f"\nComparison artifacts written:\n  {per_inst_csv}\n  {per_lb_csv}\n  {json_path}", flush=True)
    print("\nAGGREGATE COMPARISON (by lookback):")
    print(per_lb_df.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=None, help="Run a single lookback only (for resuming).")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--out-suffix", type=str, default="")
    args, _ = parser.parse_known_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    lookbacks = [args.lookback] if args.lookback is not None else LOOKBACKS

    if args.epochs < 100 and args.lookback is None:
        print("WARNING: reduced epochs is only for smoke testing; experiment should use 100.", flush=True)

    results = []
    for lb in lookbacks:
        existing = _find_complete_run(lb, args.out_suffix)
        if existing:
            print(f"[{lb}] ALREADY COMPLETE - reading existing artifacts: {existing}", flush=True)
            results.append(read_existing_run(existing, lb))
            continue
        art_dir = run_one_lookback(lb, symbols, args.epochs, args.out_suffix)
        results.append(read_existing_run(art_dir, lb))

    save_comparison_files(results)


def _find_complete_run(lookback, out_suffix=""):
    import glob
    pat = os.path.join(EXPERIMENT_ROOT, f"lb{lookback}{out_suffix}_*", "COMPLETE")
    hits = glob.glob(pat)
    if hits:
        return os.path.dirname(sorted(hits)[-1])
    return None


def read_existing_run(art_dir, lookback):
    import glob as _glob
    met = pd.read_csv(os.path.join(art_dir, "metrics.csv"))
    idx = pd.read_csv(os.path.join(art_dir, "train_test_indices.csv"))
    man = json.load(open(os.path.join(art_dir, "manifest.json"), encoding="utf-8"))
    agg_row = met[met.instrument == "AGGREGATE"].iloc[0].to_dict()
    total_train_s = sum(p["training_time_seconds"] for p in man["per_symbol"].values())
    per_instrument = []
    for _, row in met[met.instrument != "AGGREGATE"].iterrows():
        per_instrument.append({
            "lookback": lookback,
            "instrument": row["instrument"],
            "n_predictions": int(row["n_predictions"]),
            "RMSE": float(row["RMSE"]),
            "MAE": float(row["MAE"]),
            "MAPE_percent": float(row["MAPE_percent"]),
            "Directional_Accuracy_percent": float(row["Directional_Accuracy_percent"]),
            "direction_correct": int(row["direction_correct"]),
        })
    aggregate = {
        "lookback": lookback,
        "RMSE": float(agg_row["RMSE"]),
        "MAE": float(agg_row["MAE"]),
        "MAPE_percent": float(agg_row["MAPE_percent"]),
        "Directional_Accuracy_percent": float(agg_row["Directional_Accuracy_percent"]),
        "n_predictions": int(agg_row["n_predictions"]),
        "direction_correct": int(agg_row["direction_correct"]),
        "total_training_time_seconds": round(total_train_s, 2),
        "train_sequences": int(idx["train_sequences"].sum()),
        "val_sequences": int(idx["val_sequences"].sum()),
        "test_sequences": int(idx["test_sequences"].sum()),
    }
    return {"per_instrument": per_instrument, "aggregate": aggregate}


if __name__ == "__main__":
    main()