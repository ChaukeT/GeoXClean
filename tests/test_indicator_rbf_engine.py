"""
Tests for the Indicator RBF Interpolant Engine.

Pure computation tests — no Qt, no DataRegistry.
"""

import numpy as np
import pytest

from block_model_viewer.geostats.indicator_rbf_engine import (
    IndicatorRBFResult,
    run_indicator_rbf,
    resample_mask_to_grid,
    _volume_filter,
    _build_cell_centre_coords,
)
from block_model_viewer.geostats.rbf_interpolation import create_rbf_anisotropy_from_ranges


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_data(n=200, seed=42):
    """Create synthetic drillhole data with a known high-grade zone.

    High-grade zone: sphere of radius 30 centered at (50, 50, 25).
    Inside: values ~ N(5, 1)
    Outside: values ~ N(1, 0.5)
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform([0, 0, 0], [100, 100, 50], size=(n, 3))
    center = np.array([50.0, 50.0, 25.0])
    dists = np.linalg.norm(coords - center, axis=1)
    inside = dists < 30.0

    values = np.where(inside, rng.normal(5.0, 1.0, n), rng.normal(1.0, 0.5, n))
    return coords, values, inside


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBinarization:
    """Test that binarization at cut-off is correct."""

    def test_simple_cutoff(self):
        coords, values, _ = _make_synthetic_data(n=100)
        cutoff = 3.0
        result = run_indicator_rbf(
            coords, values, cutoff=cutoff,
            interp_nx=10, interp_ny=10, interp_nz=5,
            interp_dx=10.0, interp_dy=10.0, interp_dz=10.0,
            interp_x0=0.0, interp_y0=0.0, interp_z0=0.0,
        )
        assert result.sample_indicators is not None
        valid = np.isfinite(result.sample_indicators)
        # Check that indicators are 0 or 1
        unique = np.unique(result.sample_indicators[valid])
        assert set(unique).issubset({0.0, 1.0})
        # Check count matches
        expected_above = (values >= cutoff).sum()
        actual_above = (result.sample_indicators[valid] == 1.0).sum()
        assert actual_above == expected_above

    def test_all_above_cutoff(self):
        """When all values are above cutoff, all indicators should be 1."""
        coords = np.random.default_rng(0).uniform(0, 100, (50, 3))
        values = np.ones(50) * 10.0
        result = run_indicator_rbf(
            coords, values, cutoff=5.0,
            interp_nx=5, interp_ny=5, interp_nz=5,
            interp_dx=20.0, interp_dy=20.0, interp_dz=20.0,
        )
        valid = np.isfinite(result.sample_indicators)
        assert np.all(result.sample_indicators[valid] == 1.0)

    def test_all_below_cutoff(self):
        """When all values are below cutoff, all indicators should be 0."""
        coords = np.random.default_rng(0).uniform(0, 100, (50, 3))
        values = np.ones(50) * 1.0
        result = run_indicator_rbf(
            coords, values, cutoff=5.0,
            interp_nx=5, interp_ny=5, interp_nz=5,
            interp_dx=20.0, interp_dy=20.0, interp_dz=20.0,
        )
        valid = np.isfinite(result.sample_indicators)
        assert np.all(result.sample_indicators[valid] == 0.0)


class TestProbabilityField:
    """Test that the probability field is well-behaved."""

    def test_probability_range(self):
        """Probability field should be clipped to [0, 1]."""
        coords, values, _ = _make_synthetic_data(n=100)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=10, interp_ny=10, interp_nz=5,
            interp_dx=10.0, interp_dy=10.0, interp_dz=10.0,
        )
        assert result.probability_field.min() >= 0.0
        assert result.probability_field.max() <= 1.0

    def test_probability_field_shape(self):
        """Probability field should have shape (nz, ny, nx)."""
        coords, values, _ = _make_synthetic_data(n=100)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=15, interp_ny=12, interp_nz=8,
            interp_dx=7.0, interp_dy=9.0, interp_dz=6.5,
        )
        assert result.probability_field.shape == (8, 12, 15)

    def test_high_grade_zone_higher_probability(self):
        """Center of high-grade zone should have higher probability than edges."""
        coords, values, _ = _make_synthetic_data(n=300, seed=99)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=20, interp_ny=20, interp_nz=10,
            interp_dx=5.0, interp_dy=5.0, interp_dz=5.0,
            interp_x0=0.0, interp_y0=0.0, interp_z0=0.0,
        )
        # Center of grid (approximately where high-grade zone is)
        nz, ny, nx = result.probability_field.shape
        center_prob = result.probability_field[nz // 2, ny // 2, nx // 2]
        # Corner of grid (far from high-grade zone)
        corner_prob = result.probability_field[0, 0, 0]
        assert center_prob > corner_prob


class TestInsideMask:
    """Test inside/outside mask generation."""

    def test_mask_shape(self):
        coords, values, _ = _make_synthetic_data(n=100)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=10, interp_ny=10, interp_nz=5,
            interp_dx=10.0, interp_dy=10.0, interp_dz=10.0,
        )
        assert result.inside_mask_grid is not None
        assert result.inside_mask_grid.shape == (10 * 10 * 5,)
        assert result.inside_mask_grid.dtype == bool

    def test_mask_consistent_with_probability(self):
        """Inside mask should match probability >= iso_value (no extrapolation clipping)."""
        coords, values, _ = _make_synthetic_data(n=100)
        iso = 0.5
        result = run_indicator_rbf(
            coords, values, cutoff=3.0, iso_value=iso,
            interp_nx=10, interp_ny=10, interp_nz=5,
            interp_dx=10.0, interp_dy=10.0, interp_dz=10.0,
            min_volume_m3=0.0,   # no volume filter
            max_extrapolation_distance=1e9,  # disable extrapolation clipping
        )
        expected = result.probability_field.ravel() >= iso
        np.testing.assert_array_equal(result.inside_mask_grid, expected)


class TestVolumeFilter:
    """Test the minimum volume filter."""

    def test_no_filter(self):
        mask = np.array([True, False, True, True, False, True, True, True])
        shape = (2, 2, 2)
        filtered, n_before, n_after = _volume_filter(mask, shape, 1.0, 0.0)
        np.testing.assert_array_equal(filtered, mask)

    def test_removes_small_component(self):
        # Create a 4x4x1 grid with a small 1-cell component and a large 6-cell one
        mask_3d = np.zeros((1, 4, 4), dtype=bool)
        mask_3d[0, 0, 0] = True  # small component (1 cell)
        mask_3d[0, 1:3, 1:4] = True  # large component (6 cells)
        mask = mask_3d.ravel()

        cell_vol = 10.0
        min_vol = 50.0  # 5 cells minimum

        filtered, n_before, n_after = _volume_filter(mask, (1, 4, 4), cell_vol, min_vol)
        assert n_before == 2
        assert n_after == 1
        # The small component should be removed
        filtered_3d = filtered.reshape(1, 4, 4)
        assert not filtered_3d[0, 0, 0]
        assert filtered_3d[0, 1, 1]


class TestSharedGridResampling:
    """Test resampling onto shared grid."""

    def test_resample_mask(self):
        coords, values, _ = _make_synthetic_data(n=150)
        # Create a shared grid with different resolution
        shared_centroids = np.column_stack([
            np.repeat(np.arange(5, 100, 20), 25),
            np.tile(np.repeat(np.arange(5, 100, 20), 5), 5),
            np.tile(np.arange(5, 50, 10), 25),
        ])

        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=10, interp_ny=10, interp_nz=5,
            interp_dx=10.0, interp_dy=10.0, interp_dz=10.0,
            shared_grid_centroids=shared_centroids,
        )
        assert result.inside_mask_shared is not None
        assert result.inside_mask_shared.shape == (len(shared_centroids),)
        assert result.inside_mask_shared.dtype == bool


class TestSampleLabeling:
    """Test that sample domain labels are correct."""

    def test_labels_are_inside_outside(self):
        coords, values, _ = _make_synthetic_data(n=100)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=10, interp_ny=10, interp_nz=5,
            interp_dx=10.0, interp_dy=10.0, interp_dz=10.0,
        )
        assert result.sample_domain_labels is not None
        unique = set(np.unique(result.sample_domain_labels))
        assert unique.issubset({"Inside", "Outside"})

    def test_labels_length_matches_input(self):
        coords, values, _ = _make_synthetic_data(n=100)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=10, interp_ny=10, interp_nz=5,
            interp_dx=10.0, interp_dy=10.0, interp_dz=10.0,
        )
        assert len(result.sample_domain_labels) == len(values)


class TestStatistics:
    """Test that statistics are computed correctly."""

    def test_statistics_keys(self):
        coords, values, _ = _make_synthetic_data(n=100)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=10, interp_ny=10, interp_nz=5,
            interp_dx=10.0, interp_dy=10.0, interp_dz=10.0,
        )
        stats = result.statistics
        assert "n_samples_total" in stats
        assert "n_samples_above_cutoff" in stats
        assert "n_samples_below_cutoff" in stats
        assert "n_grid_cells_total" in stats
        assert "n_grid_cells_inside" in stats
        assert "volume_inside_m3" in stats

    def test_volume_consistency(self):
        coords, values, _ = _make_synthetic_data(n=100)
        dx, dy, dz = 10.0, 10.0, 10.0
        nx, ny, nz = 10, 10, 5
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=nx, interp_ny=ny, interp_nz=nz,
            interp_dx=dx, interp_dy=dy, interp_dz=dz,
        )
        cell_vol = dx * dy * dz
        total_vol = nx * ny * nz * cell_vol
        assert abs(result.volume_inside_m3 + result.volume_outside_m3 - total_vol) < 1e-6


class TestResultDataclass:
    """Test the IndicatorRBFResult dataclass."""

    def test_to_dict(self):
        coords, values, _ = _make_synthetic_data(n=50)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=5, interp_ny=5, interp_nz=5,
            interp_dx=20.0, interp_dy=20.0, interp_dz=10.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "probability_field" in d
        assert "statistics" in d
        assert "cutoff" in d
        assert d["cutoff"] == 3.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nan_values_handled(self):
        """NaN values in input should be filtered out."""
        coords = np.random.default_rng(0).uniform(0, 100, (60, 3))
        values = np.random.default_rng(0).normal(3, 2, 60)
        values[::5] = np.nan  # 12 NaN values

        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=5, interp_ny=5, interp_nz=5,
            interp_dx=20.0, interp_dy=20.0, interp_dz=10.0,
        )
        assert result.statistics["n_samples_valid"] == 48  # 60 - 12

    def test_iso_value_clipping(self):
        """Iso values outside [0.1, 0.9] should be clipped."""
        coords = np.random.default_rng(0).uniform(0, 100, (50, 3))
        values = np.random.default_rng(0).normal(3, 2, 50)

        result = run_indicator_rbf(
            coords, values, cutoff=3.0, iso_value=0.0,
            interp_nx=5, interp_ny=5, interp_nz=5,
            interp_dx=20.0, interp_dy=20.0, interp_dz=10.0,
        )
        assert result.iso_value == 0.1  # clipped to minimum

        result2 = run_indicator_rbf(
            coords, values, cutoff=3.0, iso_value=1.0,
            interp_nx=5, interp_ny=5, interp_nz=5,
            interp_dx=20.0, interp_dy=20.0, interp_dz=10.0,
        )
        assert result2.iso_value == 0.9  # clipped to maximum

    def test_mismatched_coords_values_raises(self):
        """Mismatched coords and values should raise ValueError."""
        coords = np.random.default_rng(0).uniform(0, 100, (50, 3))
        values = np.ones(30)
        with pytest.raises(ValueError, match="same number of rows"):
            run_indicator_rbf(
                coords, values, cutoff=3.0,
                interp_nx=5, interp_ny=5, interp_nz=5,
                interp_dx=20.0, interp_dy=20.0, interp_dz=10.0,
            )


class TestHelpers:
    """Test helper functions."""

    def test_build_cell_centre_coords(self):
        coords = _build_cell_centre_coords(5, 0.0, 10.0)
        np.testing.assert_allclose(coords, [5.0, 15.0, 25.0, 35.0, 45.0])

    def test_resample_mask_to_grid_basic(self):
        """Test basic resampling from one grid to another."""
        prob_field = np.zeros((3, 4, 5))
        prob_field[1, 2, 2] = 1.0  # one cell is definitely inside

        x = _build_cell_centre_coords(5, 0.0, 10.0)
        y = _build_cell_centre_coords(4, 0.0, 10.0)
        z = _build_cell_centre_coords(3, 0.0, 10.0)

        # Query at the center of the "inside" cell
        target = np.array([[25.0, 25.0, 15.0]])
        mask = resample_mask_to_grid(prob_field, x, y, z, target, 0.5)
        assert mask[0] == True

        # Query at a corner (prob = 0)
        target2 = np.array([[5.0, 5.0, 5.0]])
        mask2 = resample_mask_to_grid(prob_field, x, y, z, target2, 0.5)
        assert mask2[0] == False


# ---------------------------------------------------------------------------
# Accuracy and geometry validation
# ---------------------------------------------------------------------------


class TestAccuracyAndGeometry:
    """Accuracy, anisotropy rotation, face winding, mesh integrity, and
    extrapolation containment tests (covers BUG 1–4 fixes)."""

    # ------------------------------------------------------------------
    # BUG 2 coverage — smoother probability field produces correct direction
    # ------------------------------------------------------------------

    def test_sphere_center_is_inside_domain(self):
        """Probability field should be highest near the high-grade sphere center."""
        rng = np.random.default_rng(77)
        n = 400
        coords = rng.uniform([0, 0, 0], [100, 100, 50], size=(n, 3))
        center = np.array([50.0, 50.0, 25.0])
        inside = np.linalg.norm(coords - center, axis=1) < 30.0
        # Tight distributions so binarization at cutoff=3 cleanly separates
        values = np.where(inside, rng.normal(5.0, 0.3, n), rng.normal(1.0, 0.3, n))

        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=20, interp_ny=20, interp_nz=10,
            interp_dx=5.0, interp_dy=5.0, interp_dz=5.0,
            interp_x0=0.0, interp_y0=0.0, interp_z0=0.0,
        )
        nz, ny, nx = result.probability_field.shape
        # Grid center ≈ (50, 50, 25) — inside the sphere
        center_prob = result.probability_field[nz // 2, ny // 2, nx // 2]
        # Grid corner (0,0,0) — outside the sphere
        corner_prob = result.probability_field[0, 0, 0]
        assert center_prob > corner_prob, (
            f"Center probability ({center_prob:.3f}) should exceed corner "
            f"probability ({corner_prob:.3f})"
        )

    # ------------------------------------------------------------------
    # BUG 1 coverage — anisotropy rotation changes domain elongation axis
    # ------------------------------------------------------------------

    def test_anisotropy_rotation_changes_domain_shape(self):
        """azimuth=0 should elongate domain along X; azimuth=90 along Y.

        Designed with:
        - Moderate 3:1 anisotropy (60 vs 20 m ranges) so the RBF is numerically stable
        - max_extrapolation_distance=1e9 to disable auto-clipping that would mask the effect
        - Dense sampling (500 pts) so the probability field is reliable
        - Large high-grade sphere (R=30) to produce clearly visible inside cells
        """
        rng = np.random.default_rng(55)
        n = 500
        coords = rng.uniform([0, 0, 0], [100, 100, 50], size=(n, 3))
        center = np.array([50.0, 50.0, 25.0])
        inside = np.linalg.norm(coords - center, axis=1) < 30.0
        values = np.where(inside, rng.normal(5.0, 0.3, n), rng.normal(1.0, 0.3, n))

        # azimuth=0: major range along X, compressed in Y
        aniso_az0 = create_rbf_anisotropy_from_ranges(
            60.0, 20.0, 20.0, azimuth=0.0, dip=0.0, plunge=0.0
        )
        result_az0 = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=20, interp_ny=20, interp_nz=10,
            interp_dx=5.0, interp_dy=5.0, interp_dz=5.0,
            interp_x0=0.0, interp_y0=0.0, interp_z0=0.0,
            anisotropy=aniso_az0,
            min_volume_m3=0.0,
            max_extrapolation_distance=1e9,  # disable clipping
        )
        # azimuth=90: rotate 90° → major range now along Y
        aniso_az90 = create_rbf_anisotropy_from_ranges(
            60.0, 20.0, 20.0, azimuth=90.0, dip=0.0, plunge=0.0
        )
        result_az90 = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=20, interp_ny=20, interp_nz=10,
            interp_dx=5.0, interp_dy=5.0, interp_dz=5.0,
            interp_x0=0.0, interp_y0=0.0, interp_z0=0.0,
            anisotropy=aniso_az90,
            min_volume_m3=0.0,
            max_extrapolation_distance=1e9,
        )

        def _x_y_extents(inside_mask_flat, nx, ny, nz):
            """Return (x_span, y_span) in number of cells for the inside domain."""
            vol = inside_mask_flat.reshape(nz, ny, nx)
            any_zx = vol.any(axis=0)  # (ny, nx)
            x_cols = any_zx.any(axis=0)  # (nx,)
            y_rows = any_zx.any(axis=1)  # (ny,)
            return x_cols.sum(), y_rows.sum()

        nz, ny, nx = result_az0.probability_field.shape
        x0, y0 = _x_y_extents(result_az0.inside_mask_grid, nx, ny, nz)
        x90, y90 = _x_y_extents(result_az90.inside_mask_grid, nx, ny, nz)

        # azimuth=0 → elongated in X: x_span > y_span
        assert x0 > y0, f"azimuth=0 should be X-elongated: x_span={x0}, y_span={y0}"
        # azimuth=90 → elongated in Y: y_span > x_span
        assert y90 > x90, f"azimuth=90 should be Y-elongated: x_span={x90}, y_span={y90}"

    # ------------------------------------------------------------------
    # BUG 3 coverage — face winding (inside vs outward normals)
    # ------------------------------------------------------------------

    def test_iso_surface_face_winding_opposite(self):
        """iso_surface_faces and iso_surface_faces_outward should have opposite winding."""
        coords, values, _ = _make_synthetic_data(n=200)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=15, interp_ny=15, interp_nz=8,
            interp_dx=7.0, interp_dy=7.0, interp_dz=7.0,
            interp_x0=0.0, interp_y0=0.0, interp_z0=0.0,
        )
        d = result.to_dict()
        faces_in = d.get("iso_surface_faces")
        faces_out = d.get("iso_surface_faces_outward")

        if faces_in is None:
            pytest.skip("No iso-surface generated for this dataset/parameters")

        assert faces_out is not None, "outward faces must be provided when inward faces exist"
        assert faces_in.shape == faces_out.shape

        # First vertex is shared (winding: v0, v1, v2 → v0, v2, v1)
        np.testing.assert_array_equal(faces_in[:, 0], faces_out[:, 0])
        np.testing.assert_array_equal(faces_in[:, 1], faces_out[:, 2])
        np.testing.assert_array_equal(faces_in[:, 2], faces_out[:, 1])

        # Double flip must be identity
        faces_reflipped = faces_out.copy()
        faces_reflipped[:, [1, 2]] = faces_reflipped[:, [2, 1]]
        np.testing.assert_array_equal(faces_in, faces_reflipped)

    def test_iso_surface_normals_outward_flag(self):
        """iso_surface_normals_outward should always be False (inward = default)."""
        coords, values, _ = _make_synthetic_data(n=100)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=10, interp_ny=10, interp_nz=5,
            interp_dx=10.0, interp_dy=10.0, interp_dz=10.0,
        )
        assert result.iso_surface_normals_outward is False

    # ------------------------------------------------------------------
    # Mesh integrity
    # ------------------------------------------------------------------

    def test_isosurface_mesh_topology(self):
        """Generated mesh should have finite vertices and valid face indices."""
        coords, values, _ = _make_synthetic_data(n=300)
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=18, interp_ny=18, interp_nz=9,
            interp_dx=6.0, interp_dy=6.0, interp_dz=6.0,
            interp_x0=0.0, interp_y0=0.0, interp_z0=0.0,
        )
        verts = result.iso_surface_verts
        faces = result.iso_surface_faces

        if verts is None:
            pytest.skip("No iso-surface generated")

        # All vertex coordinates must be finite
        assert np.all(np.isfinite(verts)), "Iso-surface vertices contain non-finite values"

        # Face indices must be in range
        assert faces.min() >= 0
        assert faces.max() < len(verts)

        # No degenerate faces (all three vertices must be distinct)
        degenerate = (faces[:, 0] == faces[:, 1]) | \
                     (faces[:, 1] == faces[:, 2]) | \
                     (faces[:, 0] == faces[:, 2])
        assert not degenerate.any(), f"{degenerate.sum()} degenerate faces found"

        # All face areas must be positive
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        assert np.all(areas > 1e-12), f"{(areas <= 1e-12).sum()} zero-area faces found"

    # ------------------------------------------------------------------
    # BUG 4 coverage — extrapolation containment
    # ------------------------------------------------------------------

    def test_isosurface_contained_within_data_hull(self):
        """All iso-surface face centroids must be within max_extrapolation_distance
        of the nearest data point (with 10% tolerance)."""
        from scipy.spatial import KDTree

        coords, values, _ = _make_synthetic_data(n=200)
        max_extrap = 20.0
        result = run_indicator_rbf(
            coords, values, cutoff=3.0,
            interp_nx=20, interp_ny=20, interp_nz=10,
            interp_dx=5.0, interp_dy=5.0, interp_dz=5.0,
            interp_x0=0.0, interp_y0=0.0, interp_z0=0.0,
            max_extrapolation_distance=max_extrap,
        )
        verts = result.iso_surface_verts
        faces = result.iso_surface_faces

        if verts is None:
            pytest.skip("No iso-surface generated")

        centroids = verts[faces].mean(axis=1)
        dists, _ = KDTree(coords).query(centroids, k=1)

        tolerance = 1.1  # 10% margin for numerical edge cases
        violations = dists > max_extrap * tolerance
        assert not violations.any(), (
            f"{violations.sum()} face centroids are >{max_extrap * tolerance:.1f}m "
            f"from nearest data (max observed: {dists.max():.2f}m)"
        )

    def test_clip_mesh_to_hull_removes_far_faces(self):
        """_clip_mesh_to_hull should remove faces whose centroids exceed max_dist."""
        from block_model_viewer.geostats.indicator_rbf_engine import _clip_mesh_to_hull

        # Two non-overlapping triangles
        verts = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],  # near origin
            [100.0, 100.0, 100.0], [101.0, 100.0, 100.0], [100.5, 101.0, 100.0],  # far
        ])
        faces = np.array([[0, 1, 2], [3, 4, 5]])

        data_coords = np.array([[0.5, 0.3, 0.0]])  # single point near first triangle
        max_dist = 5.0

        new_verts, new_faces = _clip_mesh_to_hull(verts, faces, data_coords, max_dist)

        assert new_faces is not None
        assert len(new_faces) == 1, f"Expected 1 face after clipping, got {len(new_faces)}"

        # The kept face should index into the first triangle's vertices
        centroid = new_verts[new_faces[0]].mean(axis=0)
        assert np.linalg.norm(centroid - data_coords[0]) < max_dist

    def test_clip_mesh_to_hull_keeps_all_if_all_near(self):
        """_clip_mesh_to_hull should return unchanged mesh when all faces are near data."""
        from block_model_viewer.geostats.indicator_rbf_engine import _clip_mesh_to_hull

        verts = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.5, 1.0, 0.0],
        ])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        data_coords = np.array([[1.5, 0.5, 0.0]])

        new_verts, new_faces = _clip_mesh_to_hull(verts, faces, data_coords, 10.0)
        assert new_faces is not None
        assert len(new_faces) == 2
