"""Sequential runner over a list of opt configs; aggregates validation metrics.

Usage:
  python backtest_ml/run_opt_experiments.py --configs w1_*.json [--comparison w1]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_ml.exp_optimize import EXPERIMENT_ROOT, run_config  # noqa: E402


def run_all(config_paths, comparison):
    import pandas as pd
    rows = []
    for cp in config_paths:
        cfg = json.load(open(cp, encoding="utf-8"))
        name = cfg.get("name") or os.path.basename(cp)[:-5]
        summary = run_config(cfg, evaluate_test=False, name=name)
        agg = summary["validation"]["aggregate"]
        row = {
            "name": name,
            "config": json.dumps(cfg, sort_keys=True),
            "val_RMSE": agg["RMSE"],
            "val_MAPE": agg["MAPE_percent"],
            "val_DirAcc": agg["Directional_Accuracy_percent"],
            "val_n": agg["n_predictions"],
            "arch": cfg.get("arch"), "units": cfg.get("units"),
            "dropout": cfg.get("dropout"), "l2": cfg.get("l2"),
            "lr": cfg.get("lr"), "lookback": cfg.get("lookback"),
            "batch_size": cfg.get("batch_size"), "epochs": cfg.get("epochs"),
            "target": cfg.get("target"), "feature_set": cfg.get("feature_set"),
            "pooled": cfg.get("pooled"),
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(EXPERIMENT_ROOT, f"comparison_{comparison}.csv")
    df.to_csv(out, index=False)
    print("\nWROTE", out)
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--comparison", type=str, default="grid")
    args, _ = parser.parse_known_args()

    paths = []
    for arg in args.configs:
        if any(ch in arg for ch in "*?"):
            paths += sorted(glob.glob(os.path.join(EXPERIMENT_ROOT, "configs", arg)))
        else:
            p = arg if os.path.exists(arg) else os.path.join(EXPERIMENT_ROOT, "configs", arg)
            paths.append(p)
    run_all(paths, args.comparison)


if __name__ == "__main__":
    main()