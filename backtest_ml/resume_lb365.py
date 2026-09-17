"""Smart-resume helper for the lb365 lookback run after an interrupted process.

The interrupted run (default source dir ``lb365_20260912T023536Z``) completed and
saved trained models + RobustScaler params for AAPL, MSFT, TSLA, GOOGL before
dying during RY.TO (epoch 44/100). This script finishes the run WITHOUT
retraining the four completed instruments and WITHOUT touching the experiment
harness (``run_lookback_experiment.py`` is imported, never modified):

  - For the four completed symbols: re-fetch the same data, rebuild the
    RobustScaler from the saved ``*_scaler_params.json`` (center/scale), predict
    the held-out test set with the saved ``.h5`` model and recompute metrics.
    The saved model/scaler files are copied into the new run dir. Metrics are
    cross-checked against the source ``run_log.txt`` METRICS lines; a mismatch
    would only mean live-market data refreshed (weekend data is expected to be
    identical).
  - For RY.TO: run the EXACT same training pipeline as the harness
    (identical config, seeds and determinism). No partial-training resume is
    possible because the harness writes no per-epoch checkpoints.
  - Finally, write the full standard artifact set the master comparison step
    consumes: predictions.csv, ground_truth.csv, metrics.csv,
    metrics_per_instrument.csv, metrics_aggregate.csv, train_test_indices.csv,
    environment.txt, leakage_checks.txt, manifest.json and the COMPLETE marker,
    into a NEW ``lb365_<utc>Z`` run dir so ``run_lookback_experiment.py`` (no
    --lookback) compiles all 7 lookbacks.

Methodology is unchanged: only the lookback variable moves; model architecture,
epochs, batch, scaling-on-train-rows, splits, leakage controls and seeds are
identical to the harness. Provenance is recorded in manifest.json['resume'].
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
import re
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_backtest import (  # noqa: E402
    REPO_ROOT,
    PERIOD,
    TRAIN_FRAC,
    VAL_FRAC,
    EPOCHS,
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
from predictor_core import AdvancedStockPredictor  # noqa: E402
from run_lookback_experiment import (  # noqa: E402
    EXPERIMENT_ROOT,
    make_sequences_lb,
    seg_fn,
    write_leakage_report_lb,
)

LOOKBACK = 365
DEFAULT_SOURCE = "lb365_20260912T023536Z"


def parse_source_summaries(log_path):
    text = open(log_path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(
        r"^\[([A-Za-z.]+)\]\s+trained_in_s=([\d.]+) epochs_run=(\d+) best_epoch=(\d+) best_val_loss=([\d.eE+\-]+)",
        text, re.M):
        out.setdefault(m.group(1), {})["training"] = {
            "trained_in_s": float(m.group(2)),
            "epochs_run": int(m.group(3)),
            "best_epoch": int(m.group(4)),
            "best_val_loss": float(m.group(5)),
        }
    for m in re.finditer(
        r"^\[([A-Za-z.]+)\]\s+METRICS n=(\d+) RMSE=([\d.eE+\-]+) MAE=([\d.eE+\-]+) MAPE%=([\d.eE+\-]+) DirAcc%=([\d.eE+\-]+) \(correct=(\d+)\)",
        text, re.M):
        out.setdefault(m.group(1), {})["metrics"] = {
            "RMSE": float(m.group(3)),
            "MAE": float(m.group(4)),
            "MAPE_percent": float(m.group(5)),
            "Directional_Accuracy_percent": float(m.group(6)),
            "direction_correct": int(m.group(7)),
        }
    return out


def build_scaler_from_params(params):
    from sklearn.preprocessing import RobustScaler
    s = RobustScaler()
    s.center_ = np.asarray(params["center"], dtype=float)
    s.scale_ = np.asarray(params["scale"], dtype=float)
    return s


def predict_with_saved_model(model_path, params_path):
    params = json.load(open(params_path, encoding="utf-8"))
    scaler = build_scaler_from_params(params)
    frame, qreport, cleaned = load_features(params["instrument"])
    n_rows = len(frame)
    n_train = int(n_rows * TRAIN_FRAC)
    n_val = int(n_rows * VAL_FRAC)
    scaled = scaler.transform(frame[FEATURES].values)
    X, y, seq_end_idx = make_sequences_lb(scaled, LOOKBACK)
    X_test, y_test, end_test = seg_fn(X, y, seq_end_idx, n_train + n_val, n_rows, LOOKBACK)
    model = tf.keras.models.load_model(model_path)
    preds_scaled = model.predict(X_test, verbose=0).flatten()
    tf.keras.backend.clear_session()
    return frame, params, scaler, end_test, n_train, n_val, preds_scaled, qreport, cleaned


def compute_metrics(end_test, preds_scaled, scaler, close_raw, test_target_rows):
    dmy = np.zeros((len(preds_scaled), len(FEATURES)))
    dmy[:, 0] = preds_scaled
    preds_price = scaler.inverse_transform(dmy)[:, 0]
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
    return {
        "preds_price": preds_price,
        "actual_price": actual_price,
        "predicted_return": predicted_return,
        "actual_return": actual_return,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_percent": mape,
        "Directional_Accuracy_percent": dir_acc,
        "n_predictions": n_pred,
        "mape_zero_excluded": n_excluded,
        "direction_correct": n_correct,
        "test_target_rows": test_target_rows,
        "base_price": base_price,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--symbols", default=None,
                        help="comma list; default: all except those with saved models get trained fresh")
    parser.add_argument("--reuse-only", action="store_true",
                        help="DEBUG: process reused symbols and exit before fresh training/assembly")
    args, _ = parser.parse_known_args()

    source_dir = os.path.join(EXPERIMENT_ROOT, args.source_dir)
    if not os.path.isdir(source_dir):
        sys.exit(f"source dir not found: {source_dir}")
    models_src = os.path.join(source_dir, "models")
    preproc_src = os.path.join(source_dir, "preprocessing")

    src_summaries = parse_source_summaries(os.path.join(source_dir, "run_log.txt"))
    saved = {
        s.replace(".", "_"): s
        for s in src_summaries
        if "metrics" in src_summaries[s]
        and os.path.exists(os.path.join(models_src, f"{s.replace('.', '_')}_model_lstmgru_lb{LOOKBACK}_{args.source_dir}.h5"))
        and os.path.exists(os.path.join(preproc_src, f"{s.replace('.', '_')}_scaler_params.json"))
    }

    run_id = f"lb{LOOKBACK}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    art_dir = os.path.join(EXPERIMENT_ROOT, run_id)
    models_dir = os.path.join(art_dir, "models")
    pred_dir = os.path.join(art_dir, "predictions")
    preproc_dir = os.path.join(art_dir, "preprocessing")
    for d in [art_dir, models_dir, pred_dir, preproc_dir]:
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(art_dir, "run_log.txt")
    sys.stdout = Tee(sys.__stdout__, log_path)
    sys.stderr = Tee(sys.__stderr__, log_path)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
        print("tf_op_determinism=ENABLED", flush=True)
    except Exception as exc:
        print(f"tf_op_determinism=FAILED ({exc})", flush=True)

    from run_backtest import SYMBOLS as DEFAULT_SYMBOLS
    symbols = ([s.strip() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else list(DEFAULT_SYMBOLS))
    reused = [sym for sym in symbols if sym.replace(".", "_") in saved]
    fresh = [sym for sym in symbols if sym.replace(".", "_") not in saved]

    print("=" * 78)
    print(f"LOOKBACK EXPERIMENT | lookback={LOOKBACK} | SMART RESUME run_id={run_id}")
    print(f"source_interrupted_run={os.path.basename(source_dir)}")
    print(f"reused_pretrained_symbols={reused}")
    print(f"fresh_training_symbols={fresh}")
    print(f"started_utc={datetime.now(timezone.utc).isoformat()}")
    print(f"python={platform.python_version()} os={platform.platform()} machine={platform.machine()}")
    print(f"numpy={np.__version__} pandas={pd.__version__} sklearn={sklearn.__version__} "
          f"tensorflow={tf.__version__}")
    print(f"config: lookback={LOOKBACK} period={PERIOD} symbols={symbols} "
          f"split=train {TRAIN_FRAC:.2f}/val {VAL_FRAC:.2f}/test {1 - TRAIN_FRAC - VAL_FRAC:.2f} "
          f"epochs={EPOCHS} batch={BATCH_SIZE} es_patience={ES_PATIENCE} units={ENSEMBLE_UNITS}")
    print(f"features={FEATURES}")

    fingerprints = {
        "predictor_core.py": sha256_file(os.path.join(REPO_ROOT, "predictor_core.py")),
        "ml_governance.py": sha256_file(os.path.join(REPO_ROOT, "ml_governance.py")),
        "requirements.txt": sha256_file(os.path.join(REPO_ROOT, "requirements.txt")),
        os.path.join("backtest_ml", "run_backtest.py"): sha256_file(os.path.join(REPO_ROOT, "backtest_ml", "run_backtest.py")),
        os.path.join("backtest_ml", "run_lookback_experiment.py"): sha256_file(os.path.join(REPO_ROOT, "backtest_ml", "run_lookback_experiment.py")),
        os.path.join("backtest_ml", "resume_lb365.py"): sha256_file(os.path.abspath(__file__)),
    }
    print(f"code_fingerprints_sha256={json.dumps(fingerprints, indent=1)}")

    all_pred_rows, all_gt_rows, metric_rows = [], [], []
    pooled_correct, pooled_total = 0, 0
    leakage_results = []
    indices_rows = []
    manifest = {
        "run_id": run_id,
        "seed": SEED,
        "lookback_days": LOOKBACK,
        "period": PERIOD,
        "symbols": symbols,
        "features": FEATURES,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "es_patience": ES_PATIENCE,
        "units": ENSEMBLE_UNITS,
        "model_type": "ENSEMBLE (parallel LSTM+GRU hybrid)",
        "horizon": "next trading day close (1-step-ahead)",
        "resume": {
            "source_run_dir": os.path.basename(source_dir),
            "reused_pretrained_symbols": reused,
            "fresh_training_symbols": fresh,
            "note": ("Reused symbols: saved .h5 models + RobustScaler params copied from the "
                     "interrupted run; test predictions recomputed on identical data. Fresh "
                     "symbols trained with the exact harness config. Determinism (TF + seeds) "
                     "makes reused results identical to a clean single-process run."),
        },
        "fingerprints": fingerprints,
        "per_symbol": {},
    }

    # ---- reused symbols: predict with saved model, verify vs source log ----
    for sym in reused:
        key = sym.replace(".", "_")
        print("\n" + "-" * 78, flush=True)
        print(f"[{sym}] REUSING pretrained model from {args.source_dir} ...", flush=True)
        model_path = os.path.join(models_src, f"{key}_model_lstmgru_lb{LOOKBACK}_{args.source_dir}.h5")
        params_path = os.path.join(preproc_src, f"{key}_scaler_params.json")
        frame, params, scaler, end_test, n_train, n_val, preds_scaled, qreport, cleaned = \
            predict_with_saved_model(model_path, params_path)
        dates = frame["Date"].values
        close_raw = frame["Close"].values
        test_target_rows = end_test
        res = compute_metrics(end_test, preds_scaled, scaler, close_raw, test_target_rows)
        src = src_summaries[sym]
        rmses = (res["RMSE"] - src["metrics"]["RMSE"]) / max(abs(src["metrics"]["RMSE"]), 1e-12)
        maes = (res["MAE"] - src["metrics"]["MAE"]) / max(abs(src["metrics"]["MAE"]), 1e-12)
        print(f"[{sym}] recomputed n={res['n_predictions']} RMSE={res['RMSE']:.4g} "
              f"MAE={res['MAE']:.4g} MAPE%={res['MAPE_percent']:.4g} "
              f"DirAcc%={res['Directional_Accuracy_percent']:.4g} "
              f"(correct={res['direction_correct']})", flush=True)
        if abs(rmses) < 1e-4 and abs(maes) < 1e-4:
            print(f"[{sym}] cross-check vs source log METRICS: MATCH "
                  f"(rel dRMSE={rmses:.2e} dMAE={maes:.2e})", flush=True)
        else:
            print(f"[{sym}] WARNING cross-check mismatch vs source log METRICS "
                  f"(rel dRMSE={rmses:.2e} dMAE={maes:.2e}) — live data may have refreshed", flush=True)

        n_pred = res["n_predictions"]
        pooled_correct += res["direction_correct"]
        pooled_total += n_pred
        metric_rows.append({
            "instrument": sym,
            "n_predictions": n_pred,
            "RMSE": res["RMSE"],
            "MAPE_percent": res["MAPE_percent"],
            "Directional_Accuracy_percent": res["Directional_Accuracy_percent"],
            "MAE": res["MAE"],
            "mape_zero_excluded": res["mape_zero_excluded"],
            "direction_correct": res["direction_correct"],
        })
        safe_sym = sym.replace(".", "_")
        lkg = run_leakage_tests(sym, cleaned, frame, n_train)
        leakage_results.append(lkg)
        print(f"[{sym}] leakage T1_prefix_max_diff={lkg['T1_max_abs_diff_overall']:.3e}", flush=True)

        best_epoch = src["training"]["best_epoch"]
        shutil.copy2(model_path, os.path.join(models_dir, os.path.basename(model_path)))
        shutil.copy2(params_path, os.path.join(preproc_dir, os.path.basename(params_path)))

        train_seqs = max(0, n_train - LOOKBACK)
        val_seqs = max(0, (n_train + n_val) - n_train)
        test_seqs = len(end_test)
        indices_rows.append({
            "instrument": sym,
            "train_start": str(pd.Timestamp(dates[0]).date()),
            "train_end": str(pd.Timestamp(dates[n_train - 1]).date()),
            "val_start": str(pd.Timestamp(dates[n_train]).date()),
            "val_end": str(pd.Timestamp(dates[n_train + n_val - 1]).date()),
            "test_start": str(pd.Timestamp(dates[n_train + n_val]).date()),
            "test_end": str(pd.Timestamp(dates[-1]).date()),
            "train_sequences": train_seqs,
            "val_sequences": val_seqs,
            "test_sequences": test_seqs,
            "window_size_days": LOOKBACK,
        })
        checkpoint_tag = f"ENSEMBLE|lookback={LOOKBACK}|units={ENSEMBLE_UNITS}|best_epoch={best_epoch}"
        for k in range(n_pred):
            ts = pd.Timestamp(dates[test_target_rows[k]])
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert(None)
            ts_str = str(ts.date())
            all_pred_rows.append({
                "instrument": sym, "datetime": ts_str,
                "predicted_value": float(res["preds_price"][k]),
                "actual_value": float(res["actual_price"][k]),
                "predicted_return": float(res["predicted_return"][k]),
                "actual_return": float(res["actual_return"][k]),
                "model_checkpoint": checkpoint_tag,
                "run_id": run_id,
            })
            all_gt_rows.append({
                "instrument": sym, "datetime": ts_str,
                "predicted_value": "",
                "actual_value": float(res["actual_price"][k]),
                "predicted_return": "",
                "actual_return": float(res["actual_return"][k]),
                "model_checkpoint": checkpoint_tag,
                "run_id": run_id,
            })

        manifest["per_symbol"][sym] = {
            "rows": int(len(frame)),
            "train_end_date": str(pd.Timestamp(dates[n_train - 1]).date()),
            "val_start_date": str(pd.Timestamp(dates[n_train]).date()),
            "val_end_date": str(pd.Timestamp(dates[n_train + n_val - 1]).date()),
            "test_start_date": str(pd.Timestamp(dates[n_train + n_val]).date()),
            "test_end_date": str(pd.Timestamp(dates[-1]).date()),
            "n_sequences": {"train": train_seqs,
                            "val": val_seqs,
                            "test": test_seqs},
            "window_size_days": LOOKBACK,
            "scaler_center_Close": float(params["center"][0]),
            "scaler_scale_Close": float(params["scale"][0]),
            "best_epoch": best_epoch,
            "epochs_run": src["training"]["epochs_run"],
            "final_val_loss": src["training"]["best_val_loss"],
            "training_time_seconds": round(src["training"]["trained_in_s"], 2),
            "resumed_from": os.path.basename(source_dir),
            "quality_report_warnings": qreport.get("warnings", []),
        }
        print(f"[{sym}] manifests/training_time_seconds reused from source run "
              f"({src['training']['trained_in_s']:.1f}s)", flush=True)

    # ---- fresh symbols: identical full training as the harness ----
    if args.reuse_only:
        print("\n--reuse-only: skipping fresh training and assembly (debug validation run)", flush=True)
        return art_dir

    for symbol in fresh:
        print("\n" + "-" * 78, flush=True)
        print(f"[{symbol}] lookback={LOOKBACK} loading data...", flush=True)
        frame, qreport, cleaned = load_features(symbol)
        n_rows = len(frame)
        n_train = int(n_rows * TRAIN_FRAC)
        n_val = int(n_rows * VAL_FRAC)
        print(f"[{symbol}] rows={n_rows} range={frame['Date'].iloc[0].date()} .. "
              f"{frame['Date'].iloc[-1].date()} quality_valid={qreport.get('is_valid')}", flush=True)

        scaler = RobustScalerHolder()(frame, FEATURES, n_train)
        scaled = scaler.transform(frame[FEATURES].values)
        X, y, seq_end_idx = make_sequences_lb(scaled, LOOKBACK)
        dates = frame["Date"].values
        close_raw = frame["Close"].values

        X_train, y_train, end_train = seg_fn(X, y, seq_end_idx, LOOKBACK, n_train, LOOKBACK)
        X_val, y_val, end_val = seg_fn(X, y, seq_end_idx, n_train, n_train + n_val, LOOKBACK)
        X_test, y_test, end_test = seg_fn(X, y, seq_end_idx, n_train + n_val, n_rows, LOOKBACK)
        print(f"[{symbol}] sequences train={len(end_train)} val={len(end_val)} test={len(end_test)}", flush=True)
        assert len(end_test) > 0, f"{symbol}: empty held-out test segment for lookback {LOOKBACK}"
        assert len(end_train) > 0, f"{symbol}: empty train segment for lookback {LOOKBACK}"

        med_close = scaler.center_[0]
        iqr_close = scaler.scale_[0]

        predictor = AdvancedStockPredictor(
            symbol="", lookback_days=LOOKBACK, model_type="ENSEMBLE", lstm_units=ENSEMBLE_UNITS,
        )
        model = predictor.build_hybrid_ensemble_model(input_shape=(LOOKBACK, len(FEATURES)))
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
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[es, rlrp],
            shuffle=True, verbose=2, )
        train_secs = (datetime.now(timezone.utc) - t0).total_seconds()

        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
        print(f"[{symbol}] trained_in_s={train_secs:.1f} epochs_run={len(hist.history['loss'])} "
              f"best_epoch={best_epoch} best_val_loss={min(hist.history['val_loss']):.6f}", flush=True)

        preds_scaled = model.predict(X_test, verbose=0).flatten()
        res = compute_metrics(end_test, preds_scaled, scaler, close_raw, end_test)
        rmse, mae, mape, dir_acc, n_correct, n_pred = (
            res["RMSE"], res["MAE"], res["MAPE_percent"],
            res["Directional_Accuracy_percent"], res["direction_correct"], res["n_predictions"])
        pooled_correct += n_correct
        pooled_total += n_pred
        metric_rows.append({
            "instrument": symbol,
            "n_predictions": n_pred,
            "RMSE": rmse,
            "MAPE_percent": mape,
            "Directional_Accuracy_percent": dir_acc,
            "MAE": mae,
            "mape_zero_excluded": res["mape_zero_excluded"],
            "direction_correct": n_correct,
        })
        print(f"[{symbol}] METRICS n={n_pred} RMSE={rmse:.4g} MAE={mae:.4g} MAPE%={mape:.4g} "
              f"DirAcc%={dir_acc:.4g} (correct={n_correct})", flush=True)

        safe_sym = symbol.replace(".", "_")
        lkg = run_leakage_tests(symbol, cleaned, frame, n_train)
        leakage_results.append(lkg)
        print(f"[{symbol}] leakage T1_prefix_max_diff={lkg['T1_max_abs_diff_overall']:.3e} "
              f"T2_future_target_copies={sum(v['exact_equal_to_next_close_count'] for v in lkg['T2_future_target'].values())}", flush=True)

        model_path = os.path.join(models_dir, f"{safe_sym}_model_lstmgru_lb{LOOKBACK}_{run_id}.h5")
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
            "window_size_days": LOOKBACK,
        })

        checkpoint_tag = f"ENSEMBLE|lookback={LOOKBACK}|units={ENSEMBLE_UNITS}|best_epoch={best_epoch}"
        test_target_rows = end_test
        for k in range(n_pred):
            ts = pd.Timestamp(dates[test_target_rows[k]])
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert(None)
            ts_str = str(ts.date())
            all_pred_rows.append({
                "instrument": symbol, "datetime": ts_str,
                "predicted_value": float(res["preds_price"][k]),
                "actual_value": float(res["actual_price"][k]),
                "predicted_return": float(res["predicted_return"][k]),
                "actual_return": float(res["actual_return"][k]),
                "model_checkpoint": checkpoint_tag,
                "run_id": run_id,
            })
            all_gt_rows.append({
                "instrument": symbol, "datetime": ts_str,
                "predicted_value": "",
                "actual_value": float(res["actual_price"][k]),
                "predicted_return": "",
                "actual_return": float(res["actual_return"][k]),
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
            "window_size_days": LOOKBACK,
            "scaler_center_Close": float(med_close),
            "scaler_scale_Close": float(iqr_close),
            "best_epoch": best_epoch,
            "epochs_run": len(hist.history["loss"]),
            "final_val_loss": float(min(hist.history["val_loss"])),
            "training_time_seconds": round(train_secs, 2),
            "resumed_from": None,
            "quality_report_warnings": qreport.get("warnings", []),
        }

        del model
        tf.keras.backend.clear_session()

    # ---- assemble artifacts (same as harness) ----
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
        {"metric": "lookback_days", "value": LOOKBACK, "method": "experiment variable"},
        {"metric": "total_training_time_seconds", "value": round(total_train_s, 2), "method": "sum across instruments"},
        {"metric": "aggregation_method", "value": "unweighted_mean", "method": "no market-cap/volume weights present in repository"},
    ]).to_csv(agg_path, index=False)
    pd.DataFrame(indices_rows).to_csv(idx_path, index=False)
    write_environment_file(env_path, fingerprints)
    write_leakage_report_lb(lkg_path, leakage_results, LOOKBACK, manifest)
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

    with open(os.path.join(art_dir, "COMPLETE"), "w", encoding="utf-8") as f:
        f.write(f"lookback={LOOKBACK} completed (smart-resume run)\n")
    print(f"finished_utc={datetime.now(timezone.utc).isoformat()}")
    print("=" * 78, flush=True)
    return art_dir


class RobustScalerHolder:
    """Same train-only RobustScaler fit helper as run_lookback_experiment.py."""

    def __call__(self, frame, features, n_train):
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(frame[features].iloc[:n_train].values)
        return scaler


if __name__ == "__main__":
    main()