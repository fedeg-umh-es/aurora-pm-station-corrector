from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
INPUT_PATH = ROOT / "data" / "processed" / "TBD_001_pm_observations_prepared.csv"
PRED_PATH = ROOT / "results" / "runs" / "smoke_test_v0" / "persistence_predictions.csv"
METRICS_PATH = ROOT / "results" / "runs" / "smoke_test_v0" / "persistence_metrics.csv"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing processed input: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
df = df.sort_values(["station_id", "pollutant", "datetime"]).reset_index(drop=True)

lead_times = cfg["evaluation"]["lead_times"]
rows = []

for lead in lead_times:
    tmp = df.copy()
    tmp["y_true"] = tmp["value"]
    tmp["y_pred"] = tmp.groupby(["station_id", "pollutant"])["value"].shift(lead)
    tmp["lead_time"] = lead
    tmp["model"] = "persistence"
    tmp = tmp.dropna(subset=["y_pred"]).copy()
    if len(tmp) == 0:
        continue
    rows.append(tmp[["datetime", "station_id", "pollutant", "lead_time", "model", "y_true", "y_pred"]])

if not rows:
    raise ValueError("No valid persistence predictions were generated.")

pred = pd.concat(rows, ignore_index=True)
pred["error"] = pred["y_true"] - pred["y_pred"]
pred["abs_error"] = pred["error"].abs()
pred["sq_error"] = pred["error"] ** 2

metrics = (
    pred.groupby(["station_id", "pollutant", "lead_time", "model"], as_index=False)
    .agg(
        n=("y_true", "size"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda s: (s.mean()) ** 0.5),
    )
)

PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
pred.to_csv(PRED_PATH, index=False)
metrics.to_csv(METRICS_PATH, index=False)

print("Predictions file exists:", PRED_PATH.exists())
print("Metrics file exists:", METRICS_PATH.exists())
print("Prediction rows:", len(pred))
print("Metric rows:", len(metrics))
print("Leads evaluated:", sorted(pred["lead_time"].unique().tolist()))
print("Pollutants:", sorted(pred["pollutant"].unique().tolist()))
print("Persistence baseline status: OK")
