from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("Project:", cfg["project"])
print("Phase:", cfg["phase"])
print("Scientific question:", cfg["scientific_question"])
print("Study type:", cfg["study_type"])
print("Frequency:", cfg["dataset"]["frequency"])
print("Pollutants:", cfg["dataset"]["pollutants"])
print("Lead times:", cfg["evaluation"]["lead_times"])
print("Baselines:", cfg["baselines"])
print("Models:", cfg["models"])
print("Primary metrics:", cfg["metrics"]["primary"])
print("Smoke test status: OK")
