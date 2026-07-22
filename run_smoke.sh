#!/usr/bin/env bash
set -euo pipefail

python3 scripts/00_smoke_aurora_point.py
python3 scripts/01_check_inputs.py
python3 scripts/02_validate_station_observations.py
python3 scripts/03_prepare_station_series.py
python3 scripts/04_run_persistence_baseline.py
python3 scripts/05_build_smoke_test_report.py

echo "Smoke test pipeline status: OK"
