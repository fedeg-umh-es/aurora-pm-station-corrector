from __future__ import annotations

import pandas as pd

from scripts.p1_02_select_pilot_stations import select_pilot_stations


def _inputs():
    summary = pd.DataFrame(
        {
            "station_id": ["A", "B", "C", "D"],
            "eligible_primary": [True, True, False, True],
            "coverage_joint_mean": [0.95, 0.93, 0.99, 0.92],
        }
    )
    manifest = pd.DataFrame(
        {
            "station_id": ["A", "B", "C", "D"],
            "station_name": list("abcd"),
            "city": list("wxyz"),
            "latitude": [39.0] * 4,
            "longitude": [-0.4] * 4,
            "coord_status": ["ok", "duplicated_in_catalogue", "ok", "ok"],
        }
    )
    return summary, manifest


def test_requires_primary_eligibility_and_ok_coordinates():
    out = select_pilot_stations(*_inputs())
    assert list(out["station_id"]) == ["A", "D"]


def test_selection_ignores_everything_but_the_two_flags():
    summary, manifest = _inputs()
    summary.loc[summary.station_id == "D", "coverage_joint_mean"] = 0.10
    assert list(select_pilot_stations(summary, manifest)["station_id"]) == ["A", "D"]
