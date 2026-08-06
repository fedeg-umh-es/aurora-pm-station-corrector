"""
Grid-to-station spatial extraction operators.

Two operators are implemented:
  bilinear        — 4-point bilinear interpolation (primary)
  nearest         — nearest grid point (sensitivity)

Both operators must produce results using the same coordinate system so
that bilinear vs nearest comparisons are meaningful.

Constraint from the registered protocol:
  The same spatial_operator must be applied to both Aurora and CAMS output.
  Any mismatch between the operators used for the two sources is a fatal
  test failure.
"""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import numpy as np


class SpatialOperator(str, Enum):
    BILINEAR = "bilinear"
    NEAREST = "nearest_neighbour"


class GridInfo(NamedTuple):
    lat: np.ndarray   # 1-D, any order
    lon: np.ndarray   # 1-D, 0–360 or -180–180 (must match station convention)


def _normalise_lon(lon: float) -> float:
    """Wrap longitude to [0, 360)."""
    return lon % 360.0


def nearest_index(grid: GridInfo, station_lat: float, station_lon: float) -> tuple[int, int]:
    """Return (i_lat, i_lon) of the nearest grid point."""
    station_lon_norm = _normalise_lon(station_lon)
    grid_lon_norm = np.array([_normalise_lon(x) for x in grid.lon])
    i_lat = int(np.argmin(np.abs(grid.lat - station_lat)))
    i_lon = int(np.argmin(np.abs(grid_lon_norm - station_lon_norm)))
    return i_lat, i_lon


def bilinear_weights(
    grid: GridInfo, station_lat: float, station_lon: float
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """
    Compute bilinear interpolation indices and weights for one station.

    Returns
    -------
    (i0, i1, j0, j1):
        Row and column indices of the four surrounding grid points.
    weights:
        1-D array of four weights summing to 1.0.
    """
    station_lon_norm = _normalise_lon(station_lon)
    grid_lon_norm = np.array([_normalise_lon(x) for x in grid.lon])

    lat_sorted = np.argsort(grid.lat)
    lon_sorted = np.argsort(grid_lon_norm)

    # Find bracketing latitude indices
    i_above = int(np.searchsorted(grid.lat[lat_sorted], station_lat, side="right"))
    i_below = max(0, i_above - 1)
    i_above = min(i_above, len(grid.lat) - 1)
    i0 = lat_sorted[i_below]
    i1 = lat_sorted[i_above]

    # Find bracketing longitude indices
    j_right = int(np.searchsorted(grid_lon_norm[lon_sorted], station_lon_norm, side="right"))
    j_left = max(0, j_right - 1)
    j_right = min(j_right, len(grid.lon) - 1)
    j0 = lon_sorted[j_left]
    j1 = lon_sorted[j_right]

    lat0, lat1 = grid.lat[i0], grid.lat[i1]
    lon0, lon1 = grid_lon_norm[j0], grid_lon_norm[j1]

    dlat = lat1 - lat0 if lat1 != lat0 else 1.0
    dlon = lon1 - lon0 if lon1 != lon0 else 1.0

    # Normalised coordinates of the station in the cell
    t = (station_lat - lat0) / dlat
    u = (station_lon_norm - lon0) / dlon
    t = float(np.clip(t, 0.0, 1.0))
    u = float(np.clip(u, 0.0, 1.0))

    # Bilinear weights: (1-t)(1-u), t(1-u), (1-t)u, tu
    weights = np.array([(1 - t) * (1 - u), t * (1 - u), (1 - t) * u, t * u])
    return (i0, i1, j0, j1), weights


def extract_point(
    field: np.ndarray,
    grid: GridInfo,
    station_lat: float,
    station_lon: float,
    operator: SpatialOperator = SpatialOperator.BILINEAR,
) -> float:
    """
    Extract a scalar value from a 2-D field (H × W) at one station location.

    Parameters
    ----------
    field:
        2-D array with shape (H, W) matching grid.lat and grid.lon.
    grid:
        GridInfo describing the latitude and longitude coordinates.
    station_lat, station_lon:
        Station coordinates in degrees.
    operator:
        Interpolation method.
    """
    if operator == SpatialOperator.NEAREST:
        i_lat, i_lon = nearest_index(grid, station_lat, station_lon)
        return float(field[i_lat, i_lon])

    # Bilinear
    (i0, i1, j0, j1), weights = bilinear_weights(grid, station_lat, station_lon)
    corners = np.array([
        field[i0, j0], field[i1, j0], field[i0, j1], field[i1, j1]
    ])
    if np.any(np.isnan(corners)):
        return float("nan")
    return float(np.dot(weights, corners))


def extract_grid_coords(
    grid: GridInfo,
    station_lat: float,
    station_lon: float,
    operator: SpatialOperator = SpatialOperator.BILINEAR,
) -> tuple[float, float]:
    """
    Return the effective grid coordinates used for extraction.

    For nearest: the coordinates of the nearest grid point.
    For bilinear: the coordinates of the lower-left corner of the bounding cell.
    """
    if operator == SpatialOperator.NEAREST:
        i_lat, i_lon = nearest_index(grid, station_lat, station_lon)
        return float(grid.lat[i_lat]), float(grid.lon[i_lon])
    (i0, i1, j0, j1), _ = bilinear_weights(grid, station_lat, station_lon)
    return float(grid.lat[i0]), float(grid.lon[j0])
