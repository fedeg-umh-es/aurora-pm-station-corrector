"""
P1 — Deterministic pilot-station rule for Phase C.

Rule (fixed before looking at any forecast skill): a station is a pilot
candidate iff it is eligible under the PRIMARY criteria AND its catalogue
coordinates are not flagged (coord_status == "ok"). No skill information is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def select_pilot_stations(summary: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    m = summary.merge(manifest, on="station_id", validate="one_to_one")
    keep = m[m["eligible_primary"].astype(bool) & (m["coord_status"] == "ok")]
    cols = ["station_id", "station_name", "city", "latitude", "longitude", "coverage_joint_mean"]
    return keep[cols].sort_values("station_id").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--meta-dir", type=Path, default=ROOT / "data" / "metadata")
    a = p.parse_args()
    s = pd.read_csv(a.meta_dir / "station_summary.csv", dtype={"station_id": str})
    m = pd.read_csv(a.meta_dir / "stations_manifest_gva_2023_2025.csv", dtype={"station_id": str})
    out = select_pilot_stations(s, m)
    out.to_csv(a.meta_dir / "phase_c_pilot_stations.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
