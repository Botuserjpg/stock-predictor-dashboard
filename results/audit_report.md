# Audit — LSTM+GRU log-return ensemble vs baselines

Generated 2026-10-13T10:38:35 from freshly retrained models (no stored metrics trusted).

## Configuration re-run (exact repo config, 3 seeds)
- pipeline: `backtest_ml/exp_optimize.py` (target=T1 log-return, feature_set=F1, lookback=20, hybrid LSTM+GRU 48u, dirpen=1.0, epochs=60, seed 42/7/2026)
- split: chronological 70/15/15; scaler fit on train rows only; ensemble averages price-space predictions across seeds (`ensemble_opt` method).

## Seed runs used
| seed | art_dir | RMSE | MAPE% | DirAcc% | n |
|---|---|---|---|---|---|
| final_dirpen_s42 | final_dirpen_s42_20261013T042607Z | 7.0934 | 1.4486 | 51.01 | 945 |
| final_dirpen_s7 | final_dirpen_s7_20261013T043819Z | 7.2156 | 1.4870 | 47.09 | 945 |
| final_dirpen_s2026 | final_dirpen_s2026_20261013T045316Z | 7.1104 | 1.4514 | 51.43 | 945 |

## Model vs persistence vs always-up (same held-out test points)
| ticker | model RMSE | persist RMSE | model MAPE% | persist MAPE% | model DirAcc% | persist DirAcc% | always-up DirAcc% | n |
|---|---|---|---|---|---|---|---|---|
| AAPL | 4.8094 | 4.8457 | 1.1855 | 1.1973 | 50.79 | 0.00 | 52.91 | 189 |
| MSFT | 9.4790 | 9.4628 | 1.5221 | 1.5136 | 51.85 | 0.00 | 48.68 | 189 |
| TSLA | 11.5790 | 11.4490 | 2.2619 | 2.2263 | 48.68 | 0.00 | 49.74 | 189 |
| GOOGL | 6.8352 | 6.8485 | 1.4402 | 1.4287 | 53.97 | 0.00 | 47.62 | 189 |
| RY.TO | 2.6121 | 2.5842 | 0.7949 | 0.7820 | 51.85 | 0.00 | 55.03 | 189 |
| **AVG (unweighted mean)** | **7.0629** | **7.0381** | **1.4409** | **1.4296** | **pooled 51.43** | **pooled 0.00** | **pooled 50.79** | **945** |

## Directional accuracy — 95% CI and test vs 50%
- pooled model DirAcc = 51.43% (486/945); Wilson 95% CI = [48.24, 54.60]; normal 95% CI = [48.24, 54.62]
- exact binomial test vs 0.5: p = 0.3977 -> NOT statistically different from 50%
- pooled persistence 'DirAcc' = 0.00% (by convention sign(0) never matches ±1, so this is ~0; persistence forecasts no move)
- pooled always-up DirAcc = 50.79% (= share of up days over 945 points)

## Verdict
- Model beats persistence on RMSE: False (7.0629 vs 7.0381)
- Model beats persistence on MAPE: False (1.4409% vs 1.4296%)
- Model DirAcc above always-up baseline: True (51.43% vs 50.79%)
- Model DirAcc above 50%: True (statistically vs 50%: p=0.398)

