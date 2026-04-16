"""
Unit tests for geostats/domain_mask.py

Tests the three public functions:
- compute_distance_mask (isotropic, anisotropic, empty data)
- compute_convex_hull_mask (inside, outside, degenerate)
- apply_mask_to_results (NaN setting, flag column, value preservation)
- Coverage label threshold logic
"""

from __future__ import annotations

import numpy as np
import pytest

from block_model_viewer.geostats.domain_mask import (
    compute_convex_hull_mask,
    compute_distance_mask,
    apply_mask_to_results,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _grid_centroids(nx=10, ny=10, nz=1, spacing=10.0, origin=(0, 0, 0)):
    """Generate a regular grid of block centroids."""
    xs = origin[0] + (np.arange(nx) + 0.5) * spacing
    ys = origin[1] + (np.arange(ny) + 0.5) * spacing
    zs = origin[2] + (np.arange(nz) + 0.5) * spacing
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


# -----------------------------------------------------------------------
# 1. test_distance_mask_isotropic
# -----------------------------------------------------------------------

def test_distance_mask_isotropic():
    """Sphere search: verify correct block count inside/outside."""
    # Data at origin
    data = np.array([[50.0, 50.0, 5.0]])
    # Grid: 10x10x1, spacing=10, so centroids at 5,15,...,95
    blocks = _grid_centroids(10, 10, 1, spacing=10.0)
    assert blocks.shape == (100, 3)

    # Search radius 30m isotropic — should capture blocks within 30m of data point
    mask = compute_distance_mask(
        blocks, data,
        search_radii=(30.0, 30.0, 30.0),
        azimuth_deg=0.0, dip_deg=0.0,
        min_neighbours=1,
    )
    assert mask.dtype == bool
    assert mask.shape == (100,)

    # Count how many centroids are within 30m of (50, 50, 5)
    dists = np.sqrt(np.sum((blocks - data[0]) ** 2, axis=1))
    expected_count = int(np.sum(dists <= 30.0))
    assert int(mask.sum()) == expected_count
    assert expected_count > 0
    assert expected_count < 100  # Not all blocks should be inside


# -----------------------------------------------------------------------
# 2. test_distance_mask_anisotropic
# -----------------------------------------------------------------------

def test_distance_mask_anisotropic():
    """Ellipsoid with azimuth 90deg: verify elongation along East (X)."""
    data = np.array([[50.0, 50.0, 5.0]])
    blocks = _grid_centroids(10, 10, 1, spacing=10.0)

    # Major axis = 60m along azimuth 90 (East = X direction)
    # Minor axis = 20m (N-S direction)
    # Vertical = 20m
    mask = compute_distance_mask(
        blocks, data,
        search_radii=(60.0, 20.0, 20.0),
        azimuth_deg=90.0, dip_deg=0.0,
        min_neighbours=1,
    )

    # Block at (85, 50, 5) = 35m East of data → should be inside (major=60m)
    idx_east = np.argmin(np.sum((blocks - [85, 50, 5]) ** 2, axis=1))
    assert mask[idx_east], "Block 35m East should be inside (major=60m)"

    # Block at (50, 85, 5) = 35m North of data → should be OUTSIDE (minor=20m)
    idx_north = np.argmin(np.sum((blocks - [50, 85, 5]) ** 2, axis=1))
    assert not mask[idx_north], "Block 35m North should be outside (minor=20m)"


# -----------------------------------------------------------------------
# 3. test_distance_mask_empty_data
# -----------------------------------------------------------------------

def test_distance_mask_empty_data():
    """Empty data_coords → returns all-False, no crash."""
    blocks = _grid_centroids(5, 5, 1)
    empty = np.empty((0, 3))

    mask = compute_distance_mask(
        blocks, empty,
        search_radii=(100, 100, 100),
    )
    assert mask.shape == (25,)
    assert mask.sum() == 0

    # Also test None
    mask2 = compute_distance_mask(
        blocks, None,
        search_radii=(100, 100, 100),
    )
    assert mask2.sum() == 0


# -----------------------------------------------------------------------
# 4. test_convex_hull_mask_inside
# -----------------------------------------------------------------------

def test_convex_hull_mask_inside():
    """Simple 8-point cube hull: interior point included."""
    # Cube corners from (0,0,0) to (100,100,100)
    corners = np.array([
        [0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100],
        [100, 100, 0], [100, 0, 100], [0, 100, 100], [100, 100, 100],
    ], dtype=float)

    # Block at center should be inside
    blocks = np.array([[50.0, 50.0, 50.0]])
    mask = compute_convex_hull_mask(blocks, corners, buffer_m=0.0)
    assert mask[0] is np.True_ or mask[0] == True


# -----------------------------------------------------------------------
# 5. test_convex_hull_mask_outside
# -----------------------------------------------------------------------

def test_convex_hull_mask_outside():
    """Same hull: exterior point excluded."""
    corners = np.array([
        [0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100],
        [100, 100, 0], [100, 0, 100], [0, 100, 100], [100, 100, 100],
    ], dtype=float)

    blocks = np.array([[200.0, 200.0, 200.0]])
    mask = compute_convex_hull_mask(blocks, corners, buffer_m=0.0)
    assert not mask[0]


# -----------------------------------------------------------------------
# 6. test_convex_hull_mask_degenerate
# -----------------------------------------------------------------------

def test_convex_hull_mask_degenerate():
    """Coplanar points trigger bounding box fallback, no crash."""
    # All points on z=0 plane → cannot form 3D hull
    coplanar = np.array([
        [0, 0, 0], [100, 0, 0], [0, 100, 0], [100, 100, 0],
    ], dtype=float)

    blocks = np.array([[50.0, 50.0, 0.0], [200.0, 200.0, 0.0]])
    mask = compute_convex_hull_mask(blocks, coplanar, buffer_m=0.0)
    assert mask.shape == (2,)
    assert mask[0]       # Inside bounding box
    assert not mask[1]   # Outside bounding box


# -----------------------------------------------------------------------
# 7. test_apply_mask_sets_nan
# -----------------------------------------------------------------------

def test_apply_mask_sets_nan():
    """Grade values outside mask become NaN."""
    mask = np.array([True, True, False, False, True])
    results = {
        "grade": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "variance": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    }

    out = apply_mask_to_results(
        results, mask,
        grade_keys=["grade"],
        variance_keys=["variance"],
    )

    assert np.isnan(out["grade"][2])
    assert np.isnan(out["grade"][3])
    assert np.isnan(out["variance"][2])
    assert np.isnan(out["variance"][3])


# -----------------------------------------------------------------------
# 8. test_apply_mask_flag_column
# -----------------------------------------------------------------------

def test_apply_mask_flag_column():
    """domain_mask flag column is added correctly."""
    mask = np.array([True, False, True])
    results = {"grade": np.array([1.0, 2.0, 3.0])}

    out = apply_mask_to_results(
        results, mask,
        grade_keys=["grade"],
        variance_keys=[],
    )

    assert "domain_mask" in out
    assert out["domain_mask"].dtype == np.uint8
    np.testing.assert_array_equal(out["domain_mask"], [1, 0, 1])


# -----------------------------------------------------------------------
# 9. test_apply_mask_preserves_inside
# -----------------------------------------------------------------------

def test_apply_mask_preserves_inside():
    """Values inside mask are unchanged."""
    mask = np.array([True, True, False])
    results = {
        "grade": np.array([1.5, 2.5, 3.5]),
        "variance": np.array([0.1, 0.2, 0.3]),
    }

    out = apply_mask_to_results(
        results, mask,
        grade_keys=["grade"],
        variance_keys=["variance"],
    )

    assert out["grade"][0] == 1.5
    assert out["grade"][1] == 2.5
    assert out["variance"][0] == 0.1
    assert out["variance"][1] == 0.2

    # Original not modified
    assert results["grade"][2] == 3.5


# -----------------------------------------------------------------------
# 10. test_coverage_label_thresholds
# -----------------------------------------------------------------------

def test_coverage_label_thresholds():
    """Verify label colour thresholds (green/amber/red)."""
    def _coverage_colour(pct: float) -> str:
        if pct >= 50:
            return "#81c784"  # green
        elif pct >= 20:
            return "#ff9800"  # amber
        else:
            return "#ef5350"  # red

    assert _coverage_colour(75.0) == "#81c784"
    assert _coverage_colour(50.0) == "#81c784"
    assert _coverage_colour(49.9) == "#ff9800"
    assert _coverage_colour(20.0) == "#ff9800"
    assert _coverage_colour(19.9) == "#ef5350"
    assert _coverage_colour(0.0) == "#ef5350"
