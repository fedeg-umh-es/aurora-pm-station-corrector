"""
Construct a Microsoft Aurora `Batch` object from CAMS initial-condition files.

Aurora Air Pollution requires two atmospheric states separated by 6 h:
  state at (cycle_time - 6 h)
  state at cycle_time

This module handles the variable mapping, unit conversion, coordinate ordering,
and dimension checks required before calling the Aurora model.

Reference: microsoft/aurora model card and source repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Expected variable names in CAMS GRIB/NetCDF and their Aurora counterparts
# ---------------------------------------------------------------------------

CAMS_TO_AURORA_SURFACE: dict[str, str] = {
    "pm10": "pm10",
    "pm2p5": "pm2p5",
    "u10": "10u",
    "v10": "10v",
    "t2m": "2t",
    "sp": "sp",
    "msl": "msl",
}

CAMS_TO_AURORA_ATMO: dict[str, str] = {
    "u": "u",
    "v": "v",
    "w": "w",
    "t": "t",
    "q": "q",
    "z": "z",
}

# Pressure levels expected by Aurora Air Pollution (hPa)
AURORA_PRESSURE_LEVELS: tuple[int, ...] = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
)


@dataclass
class BatchSpec:
    """Specification for one Aurora Batch (two-state input)."""
    cycle_time: Any          # datetime
    lat: np.ndarray          # 1-D, degrees north, descending
    lon: np.ndarray          # 1-D, degrees east, ascending 0–360
    surf_vars: dict[str, np.ndarray]   # name → (2, H, W) (two states)
    atmo_vars: dict[str, np.ndarray]   # name → (2, len(levels), H, W)
    static_vars: dict[str, np.ndarray] # name → (H, W)
    levels: tuple[int, ...] = AURORA_PRESSURE_LEVELS
    metadata: dict[str, Any] | None = None


def validate_batch_spec(spec: BatchSpec) -> list[str]:
    """
    Return a list of validation errors (empty list = valid).
    Does not raise; caller decides whether to abort.
    """
    errors: list[str] = []
    H, W = len(spec.lat), len(spec.lon)

    # Coordinate monotonicity
    if not np.all(np.diff(spec.lat) < 0):
        errors.append("lat must be strictly descending")
    if not np.all(np.diff(spec.lon) >= 0):
        errors.append("lon must be non-decreasing")

    # Surface variable dimensions
    for name, arr in spec.surf_vars.items():
        if arr.shape != (2, H, W):
            errors.append(
                f"surf_var '{name}' shape {arr.shape} != expected (2, {H}, {W})"
            )

    # Atmospheric variable dimensions
    n_levels = len(spec.levels)
    for name, arr in spec.atmo_vars.items():
        if arr.shape != (2, n_levels, H, W):
            errors.append(
                f"atmo_var '{name}' shape {arr.shape} != "
                f"expected (2, {n_levels}, {H}, {W})"
            )

    # Static variable dimensions
    for name, arr in spec.static_vars.items():
        if arr.shape != (H, W):
            errors.append(
                f"static_var '{name}' shape {arr.shape} != expected ({H}, {W})"
            )

    return errors


def unit_check_pm(arr: np.ndarray, name: str) -> list[str]:
    """
    Heuristic unit check: Aurora expects PM in kg m⁻³.
    CAMS may deliver in µg m⁻³ (typical surface forecast output).
    Flag if values are suspiciously large (likely µg m⁻³ not converted).
    """
    issues: list[str] = []
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        issues.append(f"{name}: all NaN/Inf — check source file")
        return issues
    if finite.max() > 1e-3:
        issues.append(
            f"{name}: max={finite.max():.2e} — likely µg m⁻³ (Aurora expects kg m⁻³). "
            "Apply factor 1e-9 to convert µg m⁻³ → kg m⁻³."
        )
    if np.any(finite < 0):
        issues.append(f"{name}: {(finite < 0).sum()} negative values found")
    return issues
