from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "stations" / "TBD_001_pm_observations.csv"
OUT_PATH = ROOT / "data" / "processed" / "TBD_001_pm_observations_prepared.csv"

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Missing raw file: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)

required_columns = ["datetime", "station_id", "pollutant", "value"]
missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
df["value"] = pd.to_numeric(df["value"], errors="raise")

df = df.sort_values(["station_id", "pollutant", "datetime"]).reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("Prepared file exists:", OUT_PATH.exists())
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("Min datetime:", df["datetime"].min())
print("Max datetime:", df["datetime"].max())
print("Pollutants:", sorted(df["pollutant"].unique().tolist()))
print("Prepare station series status: OK")
