from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
OBS_PATH = ROOT / "data" / "raw" / "stations" / "TBD_001_pm_observations.csv"

required_columns = ["datetime", "station_id", "pollutant", "value"]
allowed_pollutants = {"PM10", "PM25"}

print("Observation file exists:", OBS_PATH.exists())

if not OBS_PATH.exists():
    raise FileNotFoundError(f"Missing observation file: {OBS_PATH}")

with open(OBS_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    rows = list(reader)

print("Columns:", columns)
print("Rows:", len(rows))

missing = [c for c in required_columns if c not in columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for i, row in enumerate(rows, start=1):
    if row["pollutant"] not in allowed_pollutants:
        raise ValueError(f"Invalid pollutant at row {i}: {row['pollutant']}")
    try:
        float(row["value"])
    except ValueError:
        raise ValueError(f"Non-numeric value at row {i}: {row['value']}")

print("Station observation contract status: OK")
