from pathlib import Path
import csv
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
MANIFEST_PATH = ROOT / "data" / "raw" / "stations" / "stations_manifest.csv"
DATASETS_DOC = ROOT / "docs" / "datasets.md"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

required_manifest_columns = [
    "station_id",
    "station_name",
    "city",
    "country",
    "latitude",
    "longitude",
    "pollutants",
    "frequency",
    "status",
]

print("Project:", cfg["project"])
print("Phase:", cfg["phase"])
print("Manifest exists:", MANIFEST_PATH.exists())
print("Datasets doc exists:", DATASETS_DOC.exists())

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    rows = list(reader)

print("Manifest columns:", columns)
print("Manifest rows:", len(rows))

missing = [c for c in required_manifest_columns if c not in columns]
if missing:
    raise ValueError(f"Missing manifest columns: {missing}")

print("Input contract status: OK")
