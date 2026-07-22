from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "results" / "runs" / "smoke_test_v0" / "persistence_predictions.csv"
METRICS_PATH = ROOT / "results" / "runs" / "smoke_test_v0" / "persistence_metrics.csv"
REPORT_PATH = ROOT / "results" / "runs" / "smoke_test_v0" / "README.md"

if not PRED_PATH.exists():
    raise FileNotFoundError(f"Missing predictions file: {PRED_PATH}")

if not METRICS_PATH.exists():
    raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")

pred = pd.read_csv(PRED_PATH)
metrics = pd.read_csv(METRICS_PATH)

lines = []
lines.append("# Smoke test v0 report")
lines.append("")
lines.append("## Inputs")
lines.append(f"- Predictions file: `{PRED_PATH.relative_to(ROOT)}`")
lines.append(f"- Metrics file: `{METRICS_PATH.relative_to(ROOT)}`")
lines.append("")
lines.append("## Basic counts")
lines.append(f"- Prediction rows: {len(pred)}")
lines.append(f"- Metric rows: {len(metrics)}")
lines.append(f"- Leads evaluated: {sorted(pred['lead_time'].unique().tolist())}")
lines.append(f"- Pollutants: {sorted(pred['pollutant'].unique().tolist())}")
lines.append("")
lines.append("## Persistence metrics")
lines.append("")
lines.append("| station_id | pollutant | lead_time | model | n | mae | rmse |")
lines.append("|---|---|---:|---|---:|---:|---:|")

for _, row in metrics.iterrows():
    lines.append(
        f"| {row['station_id']} | {row['pollutant']} | {int(row['lead_time'])} | "
        f"{row['model']} | {int(row['n'])} | {row['mae']:.4f} | {row['rmse']:.4f} |"
    )

lines.append("")
lines.append("## Status")
lines.append("- Smoke test persistence baseline completed successfully.")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

print("Report file exists:", REPORT_PATH.exists())
print("Report path:", REPORT_PATH)
print("Build smoke test report status: OK")
