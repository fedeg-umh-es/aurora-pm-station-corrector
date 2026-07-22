PYTHON=python3

smoke:
$(PYTHON) scripts/00_smoke_aurora_point.py
$(PYTHON) scripts/01_check_inputs.py
$(PYTHON) scripts/02_validate_station_observations.py
$(PYTHON) scripts/03_prepare_station_series.py
$(PYTHON) scripts/04_run_persistence_baseline.py
$(PYTHON) scripts/05_build_smoke_test_report.py

status:
git status

tree:
find . -maxdepth 3 | sort
