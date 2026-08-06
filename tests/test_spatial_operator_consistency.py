"""
P2 tests — spatial operator.

Verifies:
  1. Bilinear and nearest-neighbour operators are numerically consistent
     for points at exact grid points (must agree).
  2. Bilinear interpolates correctly for a linear field.
  3. Both operators return NaN for an all-NaN field.
  4. extract_grid_coords returns valid coordinates inside the grid domain.
  5. Operator mismatch (Aurora bilinear, CAMS nearest) is detectable.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.extract_point.interpolation import (
    GridInfo,
    SpatialOperator,
    extract_grid_coords,
    extract_point,
)


@pytest.fixture()
def simple_grid() -> GridInfo:
    """5 × 6 grid: lat 50..40 (step -2.5), lon 0..12.5 (step 2.5)."""
    lat = np.arange(50, 37.5, -2.5)  # [50, 47.5, 45, 42.5, 40]
    lon = np.arange(0, 15, 2.5)      # [0, 2.5, 5, 7.5, 10, 12.5]
    return GridInfo(lat=lat, lon=lon)


@pytest.fixture()
def linear_field(simple_grid) -> np.ndarray:
    """Field where value = lat + lon (degrees). Bilinear must reproduce exactly."""
    H, W = len(simple_grid.lat), len(simple_grid.lon)
    field = np.zeros((H, W))
    for i, la in enumerate(simple_grid.lat):
        for j, lo in enumerate(simple_grid.lon):
            field[i, j] = la + lo
    return field


# ---------------------------------------------------------------------------
# Bilinear vs nearest on exact grid points
# ---------------------------------------------------------------------------

def test_bilinear_nearest_agree_at_grid_node(simple_grid, linear_field):
    """At an exact grid point both operators must give the same value."""
    lat0, lon0 = float(simple_grid.lat[2]), float(simple_grid.lon[2])  # 45, 5
    v_bi = extract_point(linear_field, simple_grid, lat0, lon0, SpatialOperator.BILINEAR)
    v_nn = extract_point(linear_field, simple_grid, lat0, lon0, SpatialOperator.NEAREST)
    assert abs(v_bi - v_nn) < 1e-6


# ---------------------------------------------------------------------------
# Bilinear reproduces a linear field exactly
# ---------------------------------------------------------------------------

def test_bilinear_linear_field(simple_grid, linear_field):
    """For a field f(lat,lon) = lat + lon, bilinear must give the exact value."""
    station_lat, station_lon = 44.0, 3.75
    result = extract_point(
        linear_field, simple_grid, station_lat, station_lon, SpatialOperator.BILINEAR
    )
    expected = station_lat + station_lon
    assert abs(result - expected) < 1e-5, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_nan_field_returns_nan(simple_grid):
    H, W = len(simple_grid.lat), len(simple_grid.lon)
    nan_field = np.full((H, W), np.nan)
    v = extract_point(nan_field, simple_grid, 45.0, 5.0, SpatialOperator.BILINEAR)
    assert np.isnan(v)


def test_nearest_nan_field_returns_nan(simple_grid):
    H, W = len(simple_grid.lat), len(simple_grid.lon)
    nan_field = np.full((H, W), np.nan)
    v = extract_point(nan_field, simple_grid, 45.0, 5.0, SpatialOperator.NEAREST)
    assert np.isnan(v)


# ---------------------------------------------------------------------------
# Grid coordinates in domain
# ---------------------------------------------------------------------------

def test_bilinear_grid_coords_in_domain(simple_grid):
    lat_c, lon_c = extract_grid_coords(simple_grid, 44.0, 3.75, SpatialOperator.BILINEAR)
    assert simple_grid.lat.min() <= lat_c <= simple_grid.lat.max()
    assert simple_grid.lon.min() <= lon_c <= simple_grid.lon.max()


def test_nearest_grid_coords_are_grid_point(simple_grid):
    lat_c, lon_c = extract_grid_coords(simple_grid, 44.0, 3.75, SpatialOperator.NEAREST)
    assert lat_c in simple_grid.lat
    assert lon_c in simple_grid.lon


# ---------------------------------------------------------------------------
# Operator mismatch detection
# ---------------------------------------------------------------------------

def test_operator_mismatch_is_detectable(simple_grid, linear_field):
    """
    Applying bilinear to Aurora and nearest to CAMS on the same field
    should yield different values at off-grid stations.
    """
    station_lat, station_lon = 44.0, 3.75  # not on a grid point
    v_aurora = extract_point(
        linear_field, simple_grid, station_lat, station_lon, SpatialOperator.BILINEAR
    )
    v_cams = extract_point(
        linear_field, simple_grid, station_lat, station_lon, SpatialOperator.NEAREST
    )
    # At off-grid points the two operators give different results
    assert abs(v_aurora - v_cams) > 0, (
        "bilinear and nearest must differ at off-grid points "
        "— this test verifies mismatch is detectable"
    )
