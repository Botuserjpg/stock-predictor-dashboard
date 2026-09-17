# SESSION RESUME — StockPredictorApp lookback experiment

Paste the whole file back as the first prompt to fully re-contextualize. All commands run in
`C:\Users\DELL\Desktop\StockPredictorApp - Copy` via the bash tool (PowerShell 5.1), interpreter:
`.venv310\Scripts\python.exe` (Python 3.10.11, TF 2.15.0, yfinance 1.5.2, sklearn, pandas, numpy).
Do NOT use `.venv_backtest` (no TensorFlow).

## Objective
Controlled experiment: compare LOOKBACK windows [20, 40, 60, 90, 120, 252, 365] for the existing
LSTM+GRU hybrid stock-daily-forecast pipeline. The ONLY change across runs is the lookback value.
Record RMSE/MAE/MAPE/DirAcc per instrument + aggregate, test sequence counts, training time.
Rank lookbacks, discuss trade-offs honestly (incl. DirAcc ~50%), save CSV/JSON artifacts, end with a
beginner-friendly explanation. Do NOT modify `backtest_ml/run_backtest.py` or `predictor_core.py`.

## Harness (already built — do not rewrite)
`backtest_ml/run_lookback_experiment.py` imports everything unchanged from `run_backtest.py`:
load_features, causal_clean (ffill only), run_leakage_tests, Tee, sha256_file,
write_environment_file, FEATURES, SENTIMENT_DEFAULTS, TRAIN_FRAC=0.70, VAL_FRAC=0.15, EPOCHS=100,
BATCH_SIZE=32, ES_PATIENCE=20, RLRP_PATIENCE=10, ENSEMBLE_UNITS=80, SEED=42, PERIOD=5y, SYMBOLS
[AAPL, MSFT, TSLA, GOOGL, RY.TO]; plus `predictor_core.AdvancedStockPredictor(..., model_type="ENSEMBLE",
lstm_units=80).build_hybrid_ensemble_model((lookback, 12))` and `ml_governance.validate_market_data`.
Only local copies of `make_sequences`/`seg` are parameterized by lookback.
Flags: `--lookback N` (single/resume), `--symbols`, `--epochs`, `--out-suffix`.
Resume: `_find_complete_run(lb, suffix)` globs `backtest_artifacts/lookback_experiment/lb<lb>_*/COMPLETE`;
runs that are COMPLETE are read, not retrained. Running with NO `--lookback` compiles the master
comparison from all COMPLETE dirs (do this ONCE at the end — each per-lookback run overwrote them).

Pipeline details (must stay as-is): RobustScaler fit on TRAIN rows only (rows 0..n_train-1);
sequences input (lb, 12 features) label = scaled close at row i (next-day close), windows never cross
splits; splits n_train=int(rows*0.70), n_val=int(rows*0.15), test=n_train+n_val..end; Adam lr 5e-4
clipnorm 1, huber_loss; EarlyStopping(val_loss, patience 20, restore_best); ReduceLROnPlateau(0.5,
patience 10, min_lr 1e-7); shuffle=True. Determinism: PYTHONHASHSEED=0, TF_DETERMINISTIC_OPS=1,
TF_ENABLE_ONEDNN_OPTS=0, TF_CUDNN_DETERMINISTIC=1, CUDA_VISIBLE_DEVICES=-1 (CPU-only), OMP/NUM_INTRAOP/
INTEROP threads=1, tf.config 1-thread, `tf.config.experimental.enable_op_determinism()`, seed 42.
Metrics: RMSE, MAE, MAPE% (zero targets excluded), DirAcc% = sign(pred_ret)==sign(actual_ret), base
price = close[r-1] (row r-1 in the FULL dataset). Aggregate = unweighted mean per-instrument
RMSE/MAE/MAPE, pooled DirAcc. Features (12): Close, Volume, RSI, MACD, MACD_Signal, BB_Upper,
BB_Lower, Stoch_K, ATR, News_Sentiment, Social_Buzz, Sentiment_Strength (sentiment 0.0/0/0.1).

## Data note
Live yfinance 5y fetch now yields ~1,235-1,236 usable rows per symbol after indicator dropna/ffill
(range ~2021-10-08 .. 2026-09-11). Original benchmark `backtest_artifacts/run_20260824T112911Z` used
1,236 rows (AGGREGATE RMSE=81.6852, MAPE=22.9906, DirAcc=47.6344, n=930). Minor differences across
runs are due to data refresh, not methodology changes.

## STATUS — COMPLETED (7 of 7 lookbacks)

All 7 lookbacks are COMPLETE and the master comparison has been compiled. lb365 finished
2026-09-14 11:00 UTC via the smart-resume relaunch (PID 3884, run `lb365_20260913T053525Z`).
The final report `backtest_artifacts/lookback_experiment/LOOKBACK_REPORT.md` is fully written
with all 7 rows, rankings, trade-off discussion, and the beginner-friendly section.

- Leakage T1 prefix diff = 0.000e+00 in all runs. All dirs have COMPLETE marker.
- n_test = 930 for every lookback (5 symbols x 186 predictions each).

| lb   | RMSE    | MAE     | MAPE%   | DirAcc% | n   | train_s  | dir                           |
|------|---------|---------|---------|---------|-----|----------|-------------------------------|
| 20   | 79.2200 | 73.346  | 22.5435 | 49.5699 | 930 | 3065.8   | lb20_20260910T095509Z         |
| 40   | 80.1797 | 74.307  | 22.5837 | 49.5699 | 930 | 4591.6   | lb40_20260910T095509Z         |
| 60   | 82.8350 | 77.255  | 23.4529 | 49.3548 | 930 | 4787.2   | lb60_20260910T112715Z         |
| 90   | 80.1465 | 74.151  | 22.4320 | 50.4301 | 930 | 6058.3   | lb90_20260910T112715Z         |
| 120  | 88.5787 | 82.102  | 24.6404 | 49.4624 | 930 | 13046.0  | lb120_20260910T131424Z        |
| 252  | 80.1219 | 74.300  | 22.6986 | 50.0000 | 930 | 14378.8  | lb252_20260911T083104Z        |
| 365  | 86.2495 | 79.691  | 24.3122 | 49.1398 | 930 | 122940.3 | lb365_20260913T053525Z        |

Final rankings (by RMSE): 20 < 252 < 90 < 40 < 60 < 365 < 120. Conclusion: years of history
(lb365 ~34 h training, second-worst accuracy) add nothing; 20–90 days is the defensible range.
DirAcc band across all lookbacks: 49.14–50.43% (random-coin-flip consistency, honestly reported).

## DIRS TO IGNORE (all INCOMPLETE, no COMPLETE marker — safe to delete)
- lb252_20260910T131424Z — interrupted during MSFT on 2026-09-10
- lb252_20260911T034104Z — interrupted at AAPL epoch 23 on 2026-09-11 morning
- lb365_20260911T034104Z — interrupted at AAPL epoch 46 on 2026-09-11 morning
- lb365_20260911T123356Z — killed when opencode session ended
- lb365_20260911T131646Z — died overnight at MSFT epoch 26/100 (PID 17384)
- lb365_20260912T023536Z — reagent source used by the winning run; model+scaler files were copied
  into lb365_20260913T053525Z. Now redundant — safe to delete (do NOT delete while a future
  smart-resume references it as --source-dir).
- lb365_20260912T134925Z — interrupted at RY.TO epoch 73/100 (PID 7056 died); no COMPLETE.
- lb20smoke_20260910T095152Z — 2-epoch smoke test (AAPL only); safe to delete later

## REPORT
`backtest_artifacts/lookback_experiment/LOOKBACK_REPORT.md` is complete with all 7 lookbacks
(aggregate + per-instrument detail + rankings + trade-offs + beginner section).
Master comparison files exist: `comparison_summary.csv`, `comparison_aggregate.csv`,
`comparison_results.json` (compiled once, from all 7 COMPLETE dirs).

## WHAT HAPPENED THIS SESSION (2026-09-13/14, resumed from this file)
1. Checked lb365: PID 7056 dead, no COMPLETE in lb365_20260912T134925Z (stopped at RY.TO epoch 73/100).
2. Relaunched smart-resume via WMI (PID 3884, run dir lb365_20260913T053525Z). 4 reused symbols
   (AAPL/MSFT/TSLA/GOOGL) re-derived in ~4 min with MATCHING metrics; RY.TO trained fresh from scratch.
3. RY.TO training ran ~11:13 UTC 09-13 → 10:55 UTC 09-14 (~23 h; several ~45 min slow epochs).
4. COMPLETE marker written 2026-09-14 11:00 UTC. AGGREGATE RMSE=86.25, MAE=79.69, MAPE=24.31,
   DirAcc=49.14 (n=930; RY.TO fresh = 91.14/33.20/45.16).
5. Compiled master comparison over all 7 lookbacks (no --lookback). All 6 prior numbers match the table.
6. Filled lb365 into LOOKBACK_REPORT.md (aggregate + per-instrument + rankings + trade-off sections)
   and added the lb365 smart-resume provenance note in section 9.

## NEXT STEPS
The experiment is COMPLETE. Remaining (all optional):
1. Optional cleanup of the incomplete dirs + smoke test listed above.
2. Review LOOKBACK_REPORT.md for any edits you want.
3. No retraining needed — all numbers are final and reproducible (deterministic, seed 42).

## WHAT HAPPENED THIS SESSION (2026-09-11 ~18:00-22:46 local)
1. Resumed from SESSION_RESUME.md. Both old PIDs (18604, 10076) dead, no COMPLETE for lb252/lb365.
2. Attempted Start-Process relaunches for lb252 and lb365 — both processes died (shell cleanup).
3. Discovered lb252_20260911T083104Z was already COMPLETE (a separate relaunch at 08:31 UTC / 14:01
   local ran it fully: 14379s training, finished 12:32 UTC / 18:32 local). My duplicate Start-Process
   found the COMPLETE dir and just recompiled the comparison.
4. Relaunched lb365 via WMI Win32_Process.Create (PID 17384, detached) — this process is STILL
   RUNNING. WMI detaches the process from the shell so it survives session termination.
5. Verified all 6 completed lookback numbers match prior session values.
6. Read per-instrument CSVs for all 6 complete lookbacks.
7. Created LOOKBACK_REPORT.md skeleton with aggregate + per-instrument tables.
8. Monitored lb365 progress: AAPL completed, MSFT at epoch 19/100 as of 22:46.
9. Updated this SESSION_RESUME.md for tomorrow.

## WHAT HAPPENED THIS SESSION (2026-09-12 19:20-22:10 local)
1. Resumed from this file at 15:22. PID 13228 was alive, RY.TO at epoch 36/100 → 4 symbols done.
2. 15:31 — PID 13228 DIED mid-epoch during RY.TO epoch 44/100 (log stopped, no COMPLETE). Machine
   went down or process killed ~15:30.
3. Decided NOT to retrain the 4 finished symbols. Wrote `backtest_ml/resume_lb365.py`
   (smart-resume; does NOT modify the harness). Tested it with `--reuse-only`: all 4 recomputed
   metrics MATCH the old run_log (rel diff ~5e-5, float32 rounding only). Data identical (weekend).
4. Deleted the debug partial dir lb365_20260912T134803Z.
5. Launched the real smart-resume via WMI (PID 7056, run dir lb365_20260912T134925Z) at 19:19 local.
   RY.TO training started (epoch 1/100, 500 train samples). 4 reused symbols re-derived in ~4 min.
6. Monitored 19:34-22:10: RY.TO epochs 1-24 fast (~13-24s), epoch 25 slow (~12 min), epochs 26-51
   fast, epoch 53 was a 5185s (~86 min) slow epoch, epochs 52/54+ fast again; at epoch 59/100 at 22:10.
7. Updated this file with the current state so tomorrow can resume from here.

## WHAT HAPPENED THIS SESSION (2026-09-13/14 19:05 local → COMPLETE)
1. Resumed from this file. PID 7056 dead, no COMPLETE; log stopped at RY.TO epoch 73/100 (last write
   2026-09-12 22:36 local).
2. Relaunched smart-resume via WMI (PID 3884) at 11:04 local 09-13 → run dir lb365_20260913T053525Z.
3. Verified progress: 4 reused symbols done in ~4 min; RY.TO epoch 1/100 with 500 train samples.
4. Monitored 09-13 ~11:00-~23:00: RY.TO climbed steadily; several ~45 min slow epochs (e.g. epoch 34
   took 2678s); reached epoch 61 before a final check gap.
5. 09-14 ~19:00 local: COMPLETE marker present, process exited cleanly. AGGREGATE RMSE=86.25
   MAE=79.69 MAPE%=24.31 DirAcc%=49.14 over n=930 (457 correct). All predictions within held-out
   windows, T1 leakage = 0. RY.TO fresh = 91.14/33.20/45.16 (worst per-instrument of the 5).
6. Compiled master comparison (all 7 lookbacks). Prior 6 rows all match this file's table.
7. Filled lb365 into LOOKBACK_REPORT.md (sections 3/4/5/6/7/8/9) + provenance note in section 9;
   corrected a wrong "~2.4x" factor to "~9.4x" (lb365 train 122,940 s vs lb120 13,046 s).
8. Marked this file COMPLETE (7/7).

## Integrity rules
Only lookback may vary; never retrain with changed epochs/features/scaling for the reported numbers;
state the aggregate method; preserve leakage audit (scaler train-only, ffill only, no future-looking
features, sequences row-sliced so no window crosses splits, deterministic seeds).
