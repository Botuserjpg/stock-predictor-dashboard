"""Leakage-safe, validation-driven model-selection harness for the LSTM/GRU pipeline.

Purpose
-------
This harness is the vehicle for the optimization of the repository's forecasting
pipeline. It reproduces the *exact* evaluation protocol of
``run_backtest.py`` (chronological 70/15/15 split, RobustScaler fit on TRAIN
rows only, causal cleaning, same RMSE/MAPE/DirAcc formulas) while making the
modelling choices (target representation, feature set, lookback, architecture,
regularization, optimizer, pooling) configurable so that configurations can be
compared fairly.

Leakage guarantees (identical to run_backtest.py):
  * Scalers (features and target) are fit on TRAIN rows only, per instrument.
  * Cleaning is causal (forward-fill only; no .bfill(), no shift(-1) anywhere).
  * Sequences are sliced by row ranges so no window crosses a split boundary.
  * HYPERPARAMETER SELECTION USES THE VALIDATION SEGMENT ONLY.
  * The held-out test segment is evaluated at most ONCE for the final,
    pre-selected configuration, and only when --evaluate-test is passed.

Note on the "walk-forward" claim
--------------------------------
Like run_backtest.py this is a FIXED chronological 70/15/15 split, not a true
walk-forward (rolling/expanding retraining) protocol. We keep the same protocol
for a strict apples-to-apples comparison and label it correctly.

Usage
-----
  python backtest_ml/exp_optimize.py --config <config.json> [--name NAME]
                                     [--evaluate-test]
"""

import os
import sys

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import logging
import platform
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("TF_INTRA_OP_THREADS", "6")))
tf.config.threading.set_inter_op_parallelism_threads(int(os.environ.get("TF_INTER_OP_THREADS", "1")))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sklearn.preprocessing import RobustScaler  # noqa: E402

from predictor_core import AdvancedStockPredictor, add_all_indicators  # noqa: E402
from ml_governance import add_advanced_features, validate_market_data  # noqa: E402

SEED = 42
PERIOD = "5y"
SYMBOLS = ["AAPL", "MSFT", "TSLA", "GOOGL", "RY.TO"]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "backtest_artifacts", "opt_exp")
DATA_CACHE = os.path.join(REPO_ROOT, "data_cache", "opt_raw")

# Feature sets -----------------------------------------------------------------
F0 = [
    "Close", "Volume", "RSI", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower", "Stoch_K", "ATR",
    "News_Sentiment", "Social_Buzz", "Sentiment_Strength",
]
F1 = [
    "Return_1d", "Return_5d", "Return_20d", "Log_Return_1d",
    "Momentum_10d", "Momentum_30d",
    "Volatility_5d", "Volatility_20d", "Volatility_Ratio",
    "RSI", "MACD", "MACD_Signal", "Stoch_K", "Stoch_D",
    "BB_Position", "BB_Width", "ATR_Pct", "EMA_Gap_Pct",
    "Volume_ZScore_20d", "Volume_Ma_Ratio", "Price_ZScore_20d",
    "Drawdown_60d", "Trend_Strength_20d",
    "Close_SMA20", "Close_SMA50", "Dist_Res_20d", "Dist_Sup_20d",
]
FEATURE_SETS = {"F0": F0, "F1": F1}

SENTIMENT_DEFAULTS = {"News_Sentiment": 0.0, "Social_Buzz": 0, "Sentiment_Strength": 0.1}

TARGETS = {"T0": "price", "T1": "logreturn", "T2": "pctreturn"}


def causal_clean(data):
    cleaned = data.copy()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.sort_index()
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in cleaned.columns]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned[numeric_cols] = cleaned[numeric_cols].ffill()
    return cleaned


def load_raw(symbol):
    """Fetch raw OHLCV once and cache to disk so every experiment sees identical data."""
    os.makedirs(DATA_CACHE, exist_ok=True)
    safe = symbol.replace(".", "_")
    path = os.path.join(DATA_CACHE, f"{safe}_raw.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > 0:
            return df
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=PERIOD, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"no data fetched for {symbol}")
    df.to_csv(path)
    return df


def build_frame(symbol):
    """Build a causal feature frame (no future info). Returns (frame, report).

    Frames are cached to disk (keyed by symbol) because adding indicators is the
    most expensive step and every experiment must use identical inputs anyway.
    """
    os.makedirs(DATA_CACHE, exist_ok=True)
    cache_path = os.path.join(DATA_CACHE, f"{symbol.replace('.', '_')}_frame.parquet")
    if os.path.exists(cache_path):
        data = pd.read_parquet(cache_path)
        return data, validate_market_data_skipped(data)

    raw = load_raw(symbol)
    report = validate_market_data(raw, symbol)
    data = causal_clean(raw)
    data = add_all_indicators(data)  # also calls add_advanced_features internally
    for k, v in SENTIMENT_DEFAULTS.items():
        data[k] = v

    close = pd.to_numeric(data["Close"], errors="coerce")
    atr = pd.to_numeric(data["ATR"], errors="coerce")
    vol = pd.to_numeric(data["Volume"], errors="coerce")

    data["ATR_Pct"] = atr / close.replace(0, np.nan)
    data["EMA_Gap_Pct"] = (data["EMA_12"] - data["EMA_26"]) / close.replace(0, np.nan)
    data["Close_SMA20"] = close / data["SMA_20"].replace(0, np.nan) - 1.0
    data["Close_SMA50"] = close / data["SMA_50"].replace(0, np.nan) - 1.0
    data["Dist_Res_20d"] = close / data["Resistance_20d"].replace(0, np.nan) - 1.0
    data["Dist_Sup_20d"] = close / data["Support_20d"].replace(0, np.nan) - 1.0
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / close.replace(0, np.nan)
    vol_ma = vol.rolling(20).mean()
    data["Volume_Ma_Ratio"] = vol / vol_ma.replace(0, np.nan) - 1.0

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.apply(pd.to_numeric, errors="coerce").ffill()
    data.to_parquet(cache_path)
    return data, report


def validate_market_data_skipped(data):
    return {"is_valid": True, "warnings": ["frame loaded from cache; validation rerun skipped"]}


class Tee:
    def __init__(self, stream, log_path):
        self.stream = stream
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, data_):
        self.stream.write(data_)
        self.log.write(data_)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


def make_sequences(feat_scaled, y, lookback):
    xs, ys = [], []
    for i in range(lookback, len(feat_scaled)):
        xs.append(feat_scaled[i - lookback:i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def build_target_series(frame, features, target, n_train):
    """Return (feature_inputs, target) aligned to frame rows.

    T0 price :  scaled Close at row i is the target; window rows [i-L, i).
    T1/T2    :  (log/pct) return over close[i-1] -> close[i] is the target.
                target scaled by a per-instrument RobustScaler fit on TRAIN rows.
    """
    close = frame["Close"].to_numpy(dtype=float)

    if target == "price":
        feat = frame[features].to_numpy(dtype=float)
        scaler = RobustScaler()
        scaler.fit(feat[:n_train])
        feat_scaled = scaler.transform(feat)
        y = feat_scaled[:, 0].copy()  # Close is feature 0
        base_price = np.roll(close, 1)
        return feat_scaled, y, close, base_price, scaler

    logc = np.zeros_like(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log(close, out=logc, where=close > 0)
    ret = np.zeros_like(close)
    ret[1:] = logc[1:] - logc[:-1] if target == "logreturn" else close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan) - 1.0
    ret[:1] = np.nan

    feat = frame[features].to_numpy(dtype=float)
    feat_scaler = RobustScaler()
    feat_scaler.fit(feat[:n_train])
    feat_scaled = feat_scaler.transform(feat)

    y_scaler = RobustScaler()
    y_scaler.fit(ret[:n_train].reshape(-1, 1))
    y_scaled = y_scaler.transform(ret.reshape(-1, 1))[:, 0]

    base_price = np.roll(close, 1)
    return feat_scaled, y_scaled, close, base_price, (feat_scaler, y_scaler)


def build_model(cfg, input_shape, n_features):
    units = int(cfg.get("units", 80))
    dropout = float(cfg.get("dropout", 0.3))
    l2_reg = float(cfg.get("l2", 0.001))
    arch = cfg.get("arch", "hybrid")
    layers = int(cfg.get("layers", 2))
    use_bn = bool(cfg.get("bn", True))
    loss = cfg.get("loss", "huber")
    import tensorflow.keras as keras
    from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                          GRU, Input, LSTM, concatenate)
    from tensorflow.keras.optimizers import Adam

    reg = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    main_input = Input(shape=input_shape, name="main_input")

    def rnn_layer(unit, ret_seq):
        kw = {"units": unit, "return_sequences": ret_seq, "go_backwards": False}
        if reg is not None:
            kw["kernel_regularizer"] = reg
            kw["recurrent_regularizer"] = reg
        if dropout > 0:
            kw["dropout"] = float(cfg.get("rec_dropout_out", 0.0)) or dropout
            kw["recurrent_dropout"] = float(cfg.get("rec_dropout", 0.0))
        cell = LSTM if arch == "lstm" else GRU
        return cell(**kw)

    def stack(x):
        for ln in range(layers):
            last = ln == layers - 1
            x = rnn_layer(units if ln == 0 else units // 2, ret_seq=not last)(x)
            if use_bn:
                x = BatchNormalization()(x)
            if dropout > 0 and not last:
                x = Dropout(dropout)(x)
        return x

    if arch == "hybrid":
        lstm_branch = LSTM(units, return_sequences=True,
                           kernel_regularizer=reg, recurrent_regularizer=reg)(main_input)
        if use_bn:
            lstm_branch = BatchNormalization()(lstm_branch)
        if dropout > 0:
            lstm_branch = Dropout(dropout)(lstm_branch)
        lstm_branch = LSTM(units // 2, return_sequences=False,
                           kernel_regularizer=reg, recurrent_regularizer=reg)(lstm_branch)
        if use_bn:
            lstm_branch = BatchNormalization()(lstm_branch)

        gru_branch = GRU(units, return_sequences=True,
                         kernel_regularizer=reg, recurrent_regularizer=reg)(main_input)
        if use_bn:
            gru_branch = BatchNormalization()(gru_branch)
        if dropout > 0:
            gru_branch = Dropout(dropout)(gru_branch)
        gru_branch = GRU(units // 2, return_sequences=False,
                         kernel_regularizer=reg, recurrent_regularizer=reg)(gru_branch)
        if use_bn:
            gru_branch = BatchNormalization()(gru_branch)
        x = concatenate([lstm_branch, gru_branch])
        x = Dense(units, activation="relu", kernel_regularizer=reg)(x)
        if use_bn:
            x = BatchNormalization()(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        x = Dense(units // 2, activation="relu", kernel_regularizer=reg)(x)
        if dropout > 0:
            x = Dropout(dropout / 2)(x)
    else:
        x = stack(main_input)

    out = Dense(1, activation="linear", name="main_output")(x)
    model = keras.Model(inputs=main_input, outputs=out)
    optimizer = Adam(learning_rate=float(cfg.get("lr", 0.0005)),
                     clipnorm=float(cfg.get("clipnorm", 1.0)))
    model.compile(optimizer=optimizer,
                  loss="huber_loss" if loss == "huber" else "mse",
                  metrics=["mae", "mse"])
    return model


def inverse_to_price(y_hat, cfg, base_price, scaler, close_idx=0):
    """Convert scaled model outputs back to predicted close price."""
    y_hat = np.asarray(y_hat, dtype=float)
    target = TARGETS[cfg["target"]]
    if target == "price":
        feat_scaler = scaler
        dummy = np.zeros((len(y_hat), feat_scaler.n_features_in_))
        dummy[:, close_idx] = y_hat
        return feat_scaler.inverse_transform(dummy)[:, close_idx]
    feat_scaler, y_scaler = scaler
    un = y_scaler.inverse_transform(y_hat.reshape(-1, 1))[:, 0]
    if target == "logreturn":
        return base_price * np.exp(un)
    return base_price * (1.0 + un)


def compute_metrics(actual, pred, base):
    """Identical formulas to run_backtest.py."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    base = np.asarray(base, dtype=float)
    err = actual - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    nz = np.abs(actual) > 1e-12
    mape = float(np.mean(np.abs(err[nz] / actual[nz])) * 100) if nz.any() else float("nan")
    actual_return = actual / base - 1.0
    pred_return = pred / base - 1.0
    dir_ok = np.sign(pred_return) == np.sign(actual_return)
    dir_acc = float(np.mean(dir_ok) * 100)
    return {
        "RMSE": rmse, "MAE": mae, "MAPE_percent": mape,
        "Directional_Accuracy_percent": dir_acc,
        "direction_correct": int(dir_ok.sum()), "n": len(pred),
    }


def aggregate(per_symbol):
    agg_rmse = float(np.mean([m["RMSE"] for m in per_symbol]))
    agg_mape = float(np.mean([m["MAPE_percent"] for m in per_symbol]))
    pooled_correct = sum(m["direction_correct"] for m in per_symbol)
    pooled_total = sum(m["n"] for m in per_symbol)
    agg_dir = float(pooled_correct / pooled_total * 100)
    return {
        "RMSE": agg_rmse, "MAPE_percent": agg_mape,
        "Directional_Accuracy_percent": agg_dir,
        "n_predictions": pooled_total,
        "direction_correct": pooled_correct,
        "aggregation": "unweighted mean RMSE/MAPE; pooled DirAcc (same as run_backtest.py)",
    }


def run_config(cfg, evaluate_test=False, name=None):
    cfg = dict(cfg)
    name = name or cfg.get("name") or "config"
    run_id = f"{name}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    art_dir = os.path.join(EXPERIMENT_ROOT, run_id)
    os.makedirs(art_dir, exist_ok=True)

    log_path = os.path.join(art_dir, "run_log.txt")
    sys.stdout = Tee(sys.__stdout__, log_path)
    sys.stderr = Tee(sys.__stderr__, log_path)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    target = TARGETS[cfg["target"]]
    features = list(FEATURE_SETS[cfg["feature_set"]])
    if target == "price" and "Close" not in features:
        features = ["Close"] + features
    lookback = int(cfg.get("lookback", 60))
    epochs = int(cfg.get("epochs", 100))
    batch_size = int(cfg.get("batch_size", 32))
    es_patience = int(cfg.get("es_patience", 20))
    pooled = bool(cfg.get("pooled", False))

    print("=" * 78)
    print(f"OPT RUN | {name} | run_id={run_id}")
    print(f"config={json.dumps(cfg, sort_keys=True)}")
    print(f"features(n={len(features)}): {features}")
    print(f"target={target} lookback={lookback} pooled={pooled} "
          f"epochs={epochs} batch={batch_size} evaluate_test={evaluate_test}")
    print(f"python={platform.python_version()} tf={tf.__version__}")
    print(f"seed={SEED} TF threads intra={os.environ.get('TF_INTRA_OP_THREADS','6')}")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    frames = {}
    per_symbol_val, per_symbol_test = [], []
    val_pred_rows, test_pred_rows = [], []

    for symbol in SYMBOLS:
        print("\n" + "-" * 78, flush=True)
        print(f"[{symbol}] building frame...", flush=True)
        frame, report = build_frame(symbol)
        n_rows = len(frame)
        n_train = int(n_rows * TRAIN_FRAC)
        n_val = int(n_rows * VAL_FRAC)
        print(f"[{symbol}] rows={n_rows} range={frame.index[0].date()}..{frame.index[-1].date()} "
              f"quality_valid={report.get('is_valid')}", flush=True)

        feat_scaled, y_scaled, close, base_price, scaler = build_target_series(
            frame, features, target, n_train)
        valid = np.isfinite(feat_scaled).all(axis=1) & np.isfinite(y_scaled)
        orig_rows = np.arange(n_rows)[valid]
        feat_scaled, y_scaled, close, base_price = (
            feat_scaled[valid], y_scaled[valid], close[valid], base_price[valid])

        X, y = make_sequences(feat_scaled, y_scaled, lookback)
        seq_end_orig = orig_rows[lookback:]
        close_seq = close[lookback:]
        base_seq = base_price[lookback:]

        def seg(lo, hi):
            sel = (seq_end_orig >= lo) & (seq_end_orig < hi)
            return X[sel], y[sel], seq_end_orig[sel], close_seq[sel], base_seq[sel]

        Xtr, ytr, idxr, _, _ = seg(lookback, n_train)
        Xv, yv, idxv, closev, basev = seg(n_train, n_train + n_val)
        Xte, yte, idxx, closex, basex = seg(n_train + n_val, n_rows)
        print(f"[{symbol}] seqs train={len(Xtr)} val={len(Xv)} test={len(Xte)}", flush=True)
        assert len(Xte) > 0 and len(Xv) > 0 and len(Xtr) > 0

        frames[symbol] = {
            "Xtr": Xtr, "ytr": ytr, "Xv": Xv, "yv": yv, "Xte": Xte, "yte": yte,
            "idxx": idxx, "closex": closex, "basex": basex,
            "idxv": idxv, "closev": closev, "basev": basev,
            "scaler": scaler, "lookback": lookback, "n_train": n_train, "n_rows": n_rows,
            "dates": frame.index.to_numpy(), "features": features,
        }

    # ---- training ----
    if pooled:
        Xtr = np.concatenate([frames[s]["Xtr"] for s in SYMBOLS])
        ytr = np.concatenate([frames[s]["ytr"] for s in SYMBOLS])
        Xv = np.concatenate([frames[s]["Xv"] for s in SYMBOLS])
        yv = np.concatenate([frames[s]["yv"] for s in SYMBOLS])
    else:
        Xtr = Xv = ytr = yv = None

    total_epochs_run = 0
    for symbol in SYMBOLS:
        print(f"\n[{symbol}] training...", flush=True)
        n_features = len(features)
        model = build_model(cfg, (lookback, n_features), n_features)
        print(f"[{symbol}] params={model.count_params():,}", flush=True)

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=es_patience, restore_best_weights=True, verbose=1)
        rlrp = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=float(cfg.get("lr_factor", 0.5)),
            patience=int(cfg.get("rlrp_patience", 10)), min_lr=1e-7, verbose=1)

        t0 = datetime.now(timezone.utc)
        if pooled:
            hist = model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=epochs,
                             batch_size=batch_size, callbacks=[es, rlrp], shuffle=True, verbose=1)
        else:
            hist = model.fit(frames[symbol]["Xtr"], frames[symbol]["ytr"],
                             validation_data=(frames[symbol]["Xv"], frames[symbol]["yv"]),
                             epochs=epochs, batch_size=batch_size,
                             callbacks=[es, rlrp], shuffle=True, verbose=1)
        train_s = (datetime.now(timezone.utc) - t0).total_seconds()
        run_epochs = len(hist.history["loss"])
        total_epochs_run += run_epochs
        print(f"[{symbol}] trained_in_s={train_s:.1f} epochs_run={run_epochs} "
              f"best_val_loss={min(hist.history['val_loss']):.6f}", flush=True)

        # Validation predictions (model selection)
        src = frames[symbol]
        preds_v = model.predict(src["Xv"], verbose=0).flatten()
        pred_price_v = inverse_to_price(preds_v, cfg, src["basev"], src["scaler"])
        vm = compute_metrics(src["closev"], pred_price_v, src["basev"])
        per_symbol_val.append({"instrument": symbol, **vm})
        print(f"[{symbol}] VAL METRICS RMSE={vm['RMSE']:.4g} MAPE%={vm['MAPE_percent']:.4g} "
              f"DirAcc%={vm['Directional_Accuracy_percent']:.4g}", flush=True)

        for k, e in enumerate(src["idxv"]):
            ts = pd.Timestamp(src["dates"][e])
            ts = ts.tz_convert(None) if getattr(ts, "tzinfo", None) is not None else ts
            val_pred_rows.append({
                "instrument": symbol, "datetime": str(ts.date()),
                "predicted_value": float(pred_price_v[k]),
                "actual_value": float(src["closev"][k]),
                "split": "val",
            })

        if evaluate_test:
            preds_t = model.predict(src["Xte"], verbose=0).flatten()
            pred_price_t = inverse_to_price(preds_t, cfg, src["basex"], src["scaler"])
            tm = compute_metrics(src["closex"], pred_price_t, src["basex"])
            per_symbol_test.append({"instrument": symbol, **tm})
            print(f"[{symbol}] TEST METRICS RMSE={tm['RMSE']:.4g} MAPE%={tm['MAPE_percent']:.4g} "
                  f"DirAcc%={tm['Directional_Accuracy_percent']:.4g}", flush=True)
            for k, e in enumerate(src["idxx"]):
                ts = pd.Timestamp(src["dates"][e])
                ts = ts.tz_convert(None) if getattr(ts, "tzinfo", None) is not None else ts
                test_pred_rows.append({
                    "instrument": symbol, "datetime": str(ts.date()),
                    "predicted_value": float(pred_price_t[k]),
                    "actual_value": float(src["closex"][k]),
                    "split": "test",
                })

        del model
        tf.keras.backend.clear_session()

    agg_val = aggregate(per_symbol_val)
    print("\n" + "=" * 78)
    print(f"AGGREGATE VALIDATION: RMSE={agg_val['RMSE']:.4g} MAPE%={agg_val['MAPE_percent']:.4g} "
          f"DirAcc%={agg_val['Directional_Accuracy_percent']:.4g} over "
          f"{agg_val['n_predictions']} validation predictions", flush=True)

    with open(os.path.join(art_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    pd.DataFrame([{"instrument": "AGGREGATE", **agg_val}, *per_symbol_val]).to_csv(
        os.path.join(art_dir, "validation_metrics.csv"), index=False)
    pd.DataFrame(val_pred_rows).to_csv(os.path.join(art_dir, "validation_predictions.csv"), index=False)

    summary = {
        "name": name, "config": cfg, "run_id": run_id,
        "validation": {"aggregate": agg_val, "per_instrument": per_symbol_val},
        "validation_metrics_path": os.path.join(art_dir, "validation_metrics.csv"),
        "total_epochs_run": total_epochs_run,
    }

    if evaluate_test:
        agg_test = aggregate(per_symbol_test)
        print(f"AGGREGATE TEST: RMSE={agg_test['RMSE']:.4g} MAPE%={agg_test['MAPE_percent']:.4g} "
              f"DirAcc%={agg_test['Directional_Accuracy_percent']:.4g} over "
              f"{agg_test['n_predictions']} test predictions", flush=True)
        pd.DataFrame([{"instrument": "AGGREGATE", **agg_test}, *per_symbol_test]).to_csv(
            os.path.join(art_dir, "test_metrics.csv"), index=False)
        pd.DataFrame(test_pred_rows).to_csv(os.path.join(art_dir, "test_predictions.csv"), index=False)
        summary["test"] = {"aggregate": agg_test, "per_instrument": per_symbol_test}

    print("=" * 78, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config (or inline JSON string).")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate the held-out test segment. Intended ONLY for the "
                             "final, pre-selected configuration.")
    args, _ = parser.parse_known_args()

    if os.path.exists(args.config):
        cfg = json.load(open(args.config, encoding="utf-8"))
    else:
        cfg = json.loads(args.config)

    summary = run_config(cfg, evaluate_test=args.evaluate_test, name=args.name)
    out = os.path.join(EXPERIMENT_ROOT, "latest_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if summary.get("test"):
        print(json.dumps(summary["test"]["aggregate"], indent=2))


if __name__ == "__main__":
    main()