import os
import json
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model  # noqa: E402

BASE = r"backtest_artifacts/run_20260824T112911Z"
TAG = "46972d20e59e"

for s in ["AAPL", "MSFT", "TSLA", "GOOGL", "RY_TO"]:
    m = load_model(os.path.join(BASE, "models", f"{s}_model_lstmgru_lb60_{TAG}.h5"))
    p = pd.read_csv(os.path.join(BASE, "predictions", f"{s}_predictions.csv"))
    j = json.load(open(os.path.join(BASE, "preprocessing", f"{s}_scaler_params.json")))
    print(s,
          "h5_ok params=%d" % m.count_params(),
          "pred_rows=%d" % len(p),
          "cols=%s" % list(p.columns),
          "scaler_fit_rows=%d" % j["n_samples_fit"],
          "fit_end=%s" % j["fit_end_date"],
          "nan_free=%s" % bool(p.notna().all().all()))

a = pd.read_csv(r"backtest_artifacts/run_20260824T064105Z/predictions.csv")
b = pd.read_csv(os.path.join(BASE, "predictions.csv"))
mm = a.merge(b, on=["instrument", "datetime"], suffixes=("_am", "_canon"))
mm["pdiff"] = (mm.predicted_value_am - mm.predicted_value_canon).abs()
print()
print("canonical(ST) vs morning(MT, original script) max|pred diff| by instrument:")
print(mm.groupby("instrument").pdiff.max().round(6))
print()
print("identical-prediction share:",
      float((mm.pdiff == 0).mean()))
