"""Independent held-out audit of the LSTM+GRU log-return ensemble (final_dirpen).

Reads the freshly produced per-seed test/validation predictions from
backtest_artifacts/opt_exp/<name>_*/, averages them in PRICE space (same method as
backtest_ml/ensemble_opt.py), then recomputes every metric from scratch and adds the
persistence and always-up baselines on the SAME test points.

Nothing here trusts any stored metric: all numbers come from (actual, pred) pairs and
the causal cached close series.

Outputs:
  results/audit_<date>.csv      (all per-ticker + aggregate + baseline rows)
  results/audit_report.md       (short human report)

Usage:
  python backtest_ml/audit_analyze.py [--date YYYYMMDD]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPT_EXP = os.path.join(REPO_ROOT, "backtest_artifacts", "opt_exp")
FRAME_CACHE = os.path.join(REPO_ROOT, "data_cache", "opt_raw")
RESULTS = os.path.join(REPO_ROOT, "results")

SYMBOLS = ["AAPL", "MSFT", "TSLA", "GOOGL", "RY.TO"]
SEED_NAMES = ["final_dirpen_s42", "final_dirpen_s7", "final_dirpen_s2026"]


def newest_run(name: str) -> str:
    hits = [p for p in glob.glob(os.path.join(OPT_EXP, f"{name}_*")) if os.path.isdir(p)]
    if not hits:
        raise FileNotFoundError(f"no run dir for {name}")
    return max(hits, key=os.path.getmtime)


def base_prices() -> dict:
    out = {}
    for s in SYMBOLS:
        df = pd.read_parquet(os.path.join(FRAME_CACHE, f"{s.replace('.', '_')}_frame.parquet"))
        close = pd.to_numeric(df["Close"], errors="coerce")
        prev = close.shift(1)
        out[s] = {pd.Timestamp(idx).date().isoformat(): float(prev.iloc[i])
                  for i, idx in enumerate(df.index) if np.isfinite(prev.iloc[i])}
    return out


def core_metrics(actual: np.ndarray, pred: np.ndarray, base: np.ndarray) -> dict:
    """Identical formulas to exp_optimize.compute_metrics / run_backtest.py."""
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    base = np.asarray(base, float)
    err = actual - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    nz = np.abs(actual) > 1e-12
    mape = float(np.mean(np.abs(err[nz] / actual[nz])) * 100) if nz.any() else float("nan")
    dir_ok = np.sign(pred / base - 1.0) == np.sign(actual / base - 1.0)
    return {"RMSE": rmse, "MAE": mae, "MAPE_percent": mape,
            "Directional_Accuracy_percent": float(np.mean(dir_ok) * 100),
            "n": int(len(pred)), "direction_correct": int(dir_ok.sum())}


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (c - half) / d, 100 * (c + half) / d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    args = ap.parse_args()

    runs = {name: newest_run(name) for name in SEED_NAMES}
    base_map = base_prices()

    # ---- per-seed sanity + collect predictions ----
    per_seed_agg = []
    seed_dfs = []
    for name, run in runs.items():
        tp = pd.read_csv(os.path.join(run, "test_predictions.csv"))
        tm = pd.read_csv(os.path.join(run, "test_metrics.csv"))
        agg = tm[tm.instrument == "AGGREGATE"].iloc[0]
        per_seed_agg.append({"seed": name, "art_dir": os.path.basename(run),
                             "seed_RMSE": agg.RMSE, "seed_MAPE": agg.MAPE_percent,
                             "seed_DirAcc": agg.Directional_Accuracy_percent,
                             "seed_n": int(agg.n_predictions)})
        tp = tp.copy()
        tp["run"] = name
        seed_dfs.append(tp)

    combined = pd.concat(seed_dfs, ignore_index=True)
    dup = combined.groupby(["instrument", "datetime"])["actual_value"].nunique()
    assert (dup == 1).all(), "actual prices differ across seeds (misaligned test sets)"

    # ---- ensemble: mean predicted price across seeds per (instrument, datetime) ----
    ens = (combined.groupby(["instrument", "datetime"])["predicted_value"].mean().reset_index())
    actual = combined[["instrument", "datetime", "actual_value"]].drop_duplicates()
    ens = ens.merge(actual, on=["instrument", "datetime"], how="inner")
    ens["base"] = [base_map[r["instrument"]].get(r["datetime"], np.nan) for _, r in ens.iterrows()]
    n_missing_base = int(ens["base"].isna().sum())
    ens = ens.dropna(subset=["base"])

    rows = []
    model_rows, pers_rows, up_rows = [], [], []
    for s in SYMBOLS:
        sub = ens[ens.instrument == s]
        a = sub["actual_value"].to_numpy(float)
        p = sub["predicted_value"].to_numpy(float)
        b = sub["base"].to_numpy(float)
        m = core_metrics(a, p, b)
        pm = core_metrics(a, b, b)
        act_ret = a / b - 1.0
        up = float(np.mean(act_ret > 0) * 100)
        model_rows.append({"instrument": s, **m})
        pers_rows.append({"instrument": s, "RMSE": pm["RMSE"], "MAE": pm["MAE"],
                          "MAPE_percent": pm["MAPE_percent"],
                          "Directional_Accuracy_percent": pm["Directional_Accuracy_percent"],
                          "n": pm["n"], "direction_correct": pm["direction_correct"]})
        up_rows.append({"instrument": s, "always_up_acc_percent": up,
                        "up_share_percent": up, "n": int(len(act_ret)),
                        "direction_correct": int((act_ret > 0).sum())})
        rows.append({"section": "model", "instrument": s, "RMSE": m["RMSE"], "MAE": m["MAE"],
                     "MAPE_percent": m["MAPE_percent"],
                     "Directional_Accuracy_percent": m["Directional_Accuracy_percent"],
                     "n_test_points": m["n"], "direction_correct": m["direction_correct"]})
        rows.append({"section": "persistence", "instrument": s, "RMSE": pm["RMSE"], "MAE": pm["MAE"],
                     "MAPE_percent": pm["MAPE_percent"],
                     "Directional_Accuracy_percent": pm["Directional_Accuracy_percent"],
                     "n_test_points": pm["n"], "direction_correct": pm["direction_correct"]})
        rows.append({"section": "always_up", "instrument": s, "RMSE": np.nan, "MAE": np.nan,
                     "MAPE_percent": np.nan, "Directional_Accuracy_percent": up,
                     "n_test_points": int(len(act_ret)), "direction_correct": int((act_ret > 0).sum())})

    def agg(rs):
        c = sum(r["direction_correct"] for r in rs)
        n = sum(r["n"] for r in rs)
        return {"RMSE": float(np.mean([r["RMSE"] for r in rs])),
                "MAE": float(np.mean([r["MAE"] for r in rs])),
                "MAPE_percent": float(np.mean([r["MAPE_percent"] for r in rs])),
                "Directional_Accuracy_percent": float(c / n * 100),
                "Directional_Accuracy_unweighted_mean": float(np.mean([r["Directional_Accuracy_percent"] for r in rs])),
                "n": int(n), "direction_correct": int(c)}

    agg_model = agg(model_rows)
    agg_pers = agg(pers_rows)
    up_total = int(sum(r["direction_correct"] for r in up_rows))
    up_n = int(sum(r["n"] for r in up_rows))
    agg_up = {"Directional_Accuracy_percent": float(up_total / up_n * 100),
              "n": up_n, "direction_correct": up_total}

    k, n = agg_model["direction_correct"], agg_model["n"]
    wlo, whi = wilson_ci(k, n)
    p_hat = k / n
    se = np.sqrt(max(p_hat * (1 - p_hat), 1e-12) / n)
    nlo, nhi = 100 * (p_hat - 1.96 * se), 100 * (p_hat + 1.96 * se)
    bt = stats.binomtest(k, n, 0.5, alternative="two-sided")
    kp, np_ = agg_pers["direction_correct"], agg_pers["n"]
    p_hat_p = kp / np_
    pers_bt = stats.binomtest(kp, np_, 0.5, alternative="two-sided")

    for label, a in [("model", agg_model), ("persistence", agg_pers)]:
        rows.append({"section": label + "_AGGREGATE", "instrument": "AGGREGATE",
                     "RMSE": a["RMSE"], "MAE": a["MAE"], "MAPE_percent": a["MAPE_percent"],
                     "Directional_Accuracy_percent": a["Directional_Accuracy_percent"],
                     "n_test_points": a["n"], "direction_correct": a["direction_correct"],
                     "note": "RMSE/MAE/MAPE=unweighted mean over tickers; DirAcc=pooled"})
    rows.append({"section": "always_up_AGGREGATE", "instrument": "AGGREGATE",
                 "RMSE": np.nan, "MAE": np.nan, "MAPE_percent": np.nan,
                 "Directional_Accuracy_percent": agg_up["Directional_Accuracy_percent"],
                 "n_test_points": agg_up["n"], "direction_correct": agg_up["direction_correct"]})

    os.makedirs(RESULTS, exist_ok=True)
    out_csv = os.path.join(RESULTS, f"audit_{args.date}.csv")
    df = pd.DataFrame(rows)
    extra = pd.DataFrame({
        "section": ["meta"] * 6,
        "instrument": ["model_diracc_wilson_lo95", "model_diracc_wilson_hi95",
                       "model_diracc_normal_lo95", "model_diracc_normal_hi95",
                       "model_binom_pvalue_vs_0.5", "n_missing_base_dropped"],
        "value": [wlo, whi, nlo, nhi, bt.pvalue, n_missing_base],
    })
    pd.concat([df, extra.rename(columns={"value": "note"})], ignore_index=True).to_csv(out_csv, index=False)

    # ---- report ----
    def fmt(x, d=4):
        return "nan" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{d}f}"

    lines = []
    lines.append("# Audit — LSTM+GRU log-return ensemble vs baselines")
    lines.append("")
    lines.append(f"Generated {datetime.now().isoformat(timespec='seconds')} from freshly retrained models "
                 "(no stored metrics trusted).")
    lines.append("")
    lines.append("## Configuration re-run (exact repo config, 3 seeds)")
    lines.append("- pipeline: `backtest_ml/exp_optimize.py` (target=T1 log-return, feature_set=F1, "
                 "lookback=20, hybrid LSTM+GRU 48u, dirpen=1.0, epochs=60, seed 42/7/2026)")
    lines.append("- split: chronological 70/15/15; scaler fit on train rows only; ensemble averages "
                 "price-space predictions across seeds (`ensemble_opt` method).")
    lines.append("")
    lines.append("## Seed runs used")
    lines.append("| seed | art_dir | RMSE | MAPE% | DirAcc% | n |")
    lines.append("|---|---|---|---|---|---|")
    for s in per_seed_agg:
        lines.append(f"| {s['seed']} | {s['art_dir']} | {fmt(s['seed_RMSE'])} | {fmt(s['seed_MAPE'])} "
                     f"| {fmt(s['seed_DirAcc'],2)} | {s['seed_n']} |")
    lines.append("")
    lines.append("## Model vs persistence vs always-up (same held-out test points)")
    lines.append("| ticker | model RMSE | persist RMSE | model MAPE% | persist MAPE% | "
                 "model DirAcc% | persist DirAcc% | always-up DirAcc% | n |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(SYMBOLS):
        m, pm, up = model_rows[i], pers_rows[i], up_rows[i]
        lines.append(f"| {s} | {fmt(m['RMSE'])} | {fmt(pm['RMSE'])} | {fmt(m['MAPE_percent'])} "
                     f"| {fmt(pm['MAPE_percent'])} | {fmt(m['Directional_Accuracy_percent'],2)} "
                     f"| {fmt(pm['Directional_Accuracy_percent'],2)} "
                     f"| {fmt(up['always_up_acc_percent'],2)} | {m['n']} |")
    lines.append(f"| **AVG (unweighted mean)** | **{fmt(agg_model['RMSE'])}** | "
                 f"**{fmt(agg_pers['RMSE'])}** | **{fmt(agg_model['MAPE_percent'])}** | "
                 f"**{fmt(agg_pers['MAPE_percent'])}** | "
                 f"**pooled {fmt(agg_model['Directional_Accuracy_percent'],2)}** | "
                 f"**pooled {fmt(agg_pers['Directional_Accuracy_percent'],2)}** | "
                 f"**pooled {fmt(agg_up['Directional_Accuracy_percent'],2)}** | "
                 f"**{agg_model['n']}** |")
    lines.append("")
    lines.append("## Directional accuracy — 95% CI and test vs 50%")
    lines.append(f"- pooled model DirAcc = {fmt(agg_model['Directional_Accuracy_percent'],2)}% "
                 f"({k}/{n}); Wilson 95% CI = [{fmt(wlo,2)}, {fmt(whi,2)}]; "
                 f"normal 95% CI = [{fmt(nlo,2)}, {fmt(nhi,2)}]")
    lines.append(f"- exact binomial test vs 0.5: p = {bt.pvalue:.4g} -> "
                 f"{'DIFFERS from 50%' if bt.pvalue < 0.05 else 'NOT statistically different from 50%'}")
    lines.append(f"- pooled persistence 'DirAcc' = {fmt(agg_pers['Directional_Accuracy_percent'],2)}% "
                 f"(by convention sign(0) never matches ±1, so this is ~0; persistence forecasts no move)")
    lines.append(f"- pooled always-up DirAcc = {fmt(agg_up['Directional_Accuracy_percent'],2)}% "
                 f"(= share of up days over {up_n} points)")
    lines.append("")
    lines.append("## Verdict")
    beat_rmse = agg_model["RMSE"] < agg_pers["RMSE"]
    beat_mape = agg_model["MAPE_percent"] < agg_pers["MAPE_percent"]
    lines.append(f"- Model beats persistence on RMSE: {beat_rmse} "
                 f"({fmt(agg_model['RMSE'])} vs {fmt(agg_pers['RMSE'])})")
    lines.append(f"- Model beats persistence on MAPE: {beat_mape} "
                 f"({fmt(agg_model['MAPE_percent'])}% vs {fmt(agg_pers['MAPE_percent'])}%)")
    lines.append(f"- Model DirAcc above always-up baseline: "
                 f"{agg_model['Directional_Accuracy_percent'] > agg_up['Directional_Accuracy_percent']} "
                 f"({fmt(agg_model['Directional_Accuracy_percent'],2)}% vs {fmt(agg_up['Directional_Accuracy_percent'],2)}%)")
    lines.append(f"- Model DirAcc above 50%: "
                 f"{agg_model['Directional_Accuracy_percent'] > 50} "
                 f"(statistically vs 50%: p={bt.pvalue:.3g})")
    lines.append("")
    with open(os.path.join(RESULTS, "audit_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("WROTE", out_csv)
    print("WROTE", os.path.join(RESULTS, "audit_report.md"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
