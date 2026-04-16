"""
BM-01 through BM-06: Block Model domain checks.

Tests verify block model creation, validation, rotation matrices,
property management, coordinate systems, and export fidelity.
"""
import pytest
import numpy as np
import pandas as pd

pytestmark = [pytest.mark.block_model, pytest.mark.smoke]


# ── BM-01: Block model geometry validation ───────────────────────────────────

@pytest.mark.blocker
class TestBM01GeometryValidation:
    def test_valid_model_passes(self):
        """A well-formed block model should have zero validation errors."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        positions = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
        dimensions = np.array([[10, 10, 5], [10, 10, 5], [10, 10, 5]], dtype=float)
        bm.set_geometry(positions, dimensions)
        errors = bm.validate()
        assert len(errors) == 0, f"BM-01 FAIL: Valid model has errors: {errors}"

    def test_nan_coordinates_detected(self):
        """NaN in positions should be caught by validate()."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        positions = np.array([[np.nan, 0, 0], [10, 0, 0]], dtype=float)
        dimensions = np.array([[10, 10, 5], [10, 10, 5]], dtype=float)
        bm.set_geometry(positions, dimensions)
        errors = bm.validate()
        assert any("NaN" in e or "nan" in e.lower() for e in errors), \
            "BM-01 FAIL: NaN coordinates not detected"

    def test_non_positive_dimensions_detected(self):
        """Zero or negative block dimensions should be caught."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        positions = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        dimensions = np.array([[10, 10, 5], [0, 10, 5]], dtype=float)
        bm.set_geometry(positions, dimensions)
        errors = bm.validate()
        assert any("non-positive" in e.lower() or "dimension" in e.lower() for e in errors), \
            "BM-01 FAIL: Non-positive dimensions not detected"

    def test_duplicate_positions_detected(self):
        """Duplicate block positions should be flagged."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        positions = np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0]], dtype=float)
        dimensions = np.array([[10, 10, 5]] * 3, dtype=float)
        bm.set_geometry(positions, dimensions)
        errors = bm.validate()
        assert any("duplicate" in e.lower() for e in errors), \
            "BM-01 FAIL: Duplicate positions not detected"


# ── BM-02: Rotation matrix orthogonality ─────────────────────────────────────

@pytest.mark.blocker
class TestBM02RotationOrthogonality:
    def test_identity_rotation_accepted(self):
        """Identity matrix should be accepted as a valid rotation."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        positions = np.array([[0, 0, 0]], dtype=float)
        dimensions = np.array([[10, 10, 5]], dtype=float)
        bm.set_geometry(positions, dimensions)
        bm.set_rotation_matrix(np.eye(3))

    def test_non_orthogonal_matrix_rejected(self):
        """A non-orthogonal matrix should raise an error."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        positions = np.array([[0, 0, 0]], dtype=float)
        dimensions = np.array([[10, 10, 5]], dtype=float)
        bm.set_geometry(positions, dimensions)
        bad_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        with pytest.raises(Exception):
            bm.set_rotation_matrix(bad_matrix)

    def test_mining_convention_rotation(self):
        """GSLIB rotation from azimuth/dip/plunge should produce orthogonal matrix."""
        try:
            from block_model_viewer.models.blockmodel_advanced import get_rotation_matrix
            R = get_rotation_matrix(azimuth=45.0, dip=30.0, plunge=0.0)
            # Check orthogonality: R^T R ≈ I
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10,
                                       err_msg="BM-02 FAIL: Rotation matrix not orthogonal")
            # Check determinant ≈ +1
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10,
                                       err_msg="BM-02 FAIL: Rotation determinant ≠ 1")
        except ImportError:
            pytest.skip("blockmodel_advanced not available")


# ── BM-03: Property management integrity ────────────────────────────────────

@pytest.mark.critical
class TestBM03PropertyManagement:
    def test_add_and_retrieve_property(self):
        """Properties should round-trip correctly."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        n = 5
        positions = np.arange(n * 3, dtype=float).reshape(n, 3)
        dimensions = np.ones((n, 3), dtype=float) * 10
        bm.set_geometry(positions, dimensions)
        grades = np.array([55.0, 60.0, 58.0, 52.0, 63.0])
        bm.add_property("Fe", grades)
        retrieved = bm.get_property("Fe")
        assert retrieved is not None, "BM-03 FAIL: Property not found after add"
        np.testing.assert_array_equal(retrieved, grades,
                                       err_msg="BM-03 FAIL: Property values changed")

    def test_property_length_mismatch_detected(self):
        """Adding a property with wrong length should raise ValueError."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        positions = np.zeros((3, 3), dtype=float)
        dimensions = np.ones((3, 3), dtype=float) * 10
        bm.set_geometry(positions, dimensions)
        with pytest.raises(ValueError, match="length"):
            bm.add_property("Fe", np.array([1.0, 2.0]))  # wrong length

    def test_property_statistics(self):
        """get_property_statistics should return correct mean/min/max."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        n = 4
        positions = np.arange(n * 3, dtype=float).reshape(n, 3)
        dimensions = np.ones((n, 3), dtype=float) * 10
        bm.set_geometry(positions, dimensions)
        vals = np.array([10.0, 20.0, 30.0, 40.0])
        bm.add_property("grade", vals)
        stats = bm.get_property_statistics("grade")
        if stats is not None:
            assert abs(stats["mean"] - 25.0) < 0.01, "BM-03 FAIL: Mean incorrect"
            assert abs(stats["min"] - 10.0) < 0.01, "BM-03 FAIL: Min incorrect"
            assert abs(stats["max"] - 40.0) < 0.01, "BM-03 FAIL: Max incorrect"


# ── BM-04: DataFrame round-trip ──────────────────────────────────────────────

@pytest.mark.critical
class TestBM04DataFrameRoundTrip:
    def test_to_dataframe_preserves_data(self):
        """to_dataframe() should include coordinates and properties."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        positions = np.array([[100, 200, 300], [110, 200, 300]], dtype=float)
        dimensions = np.array([[10, 10, 5], [10, 10, 5]], dtype=float)
        bm.set_geometry(positions, dimensions)
        bm.add_property("Fe", np.array([55.0, 60.0]))
        df = bm.to_dataframe()
        assert "Fe" in df.columns, "BM-04 FAIL: Property missing from DataFrame"
        assert len(df) == 2, "BM-04 FAIL: Wrong row count"
        assert abs(df["Fe"].iloc[0] - 55.0) < 0.01, "BM-04 FAIL: Property value changed"


# ── BM-05: Orthogonality detection ──────────────────────────────────────────

@pytest.mark.major
class TestBM05OrthogonalityDetection:
    def test_regular_grid_is_orthogonal(self):
        """A regular axis-aligned grid should be detected as orthogonal."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        # Create a 2x2x1 regular grid
        positions = np.array([
            [5, 5, 2.5], [15, 5, 2.5],
            [5, 15, 2.5], [15, 15, 2.5],
        ], dtype=float)
        dimensions = np.array([[10, 10, 5]] * 4, dtype=float)
        bm.set_geometry(positions, dimensions)
        if hasattr(bm, "is_orthogonal"):
            result = bm.is_orthogonal()
            is_ortho = result[0] if isinstance(result, tuple) else result
            assert is_ortho, "BM-05 FAIL: Regular grid not detected as orthogonal"


# ── BM-06: Block model definition from grid spec ────────────────────────────

@pytest.mark.major
class TestBM06GridSpecCreation:
    def test_from_grid_spec(self):
        """BlockModelDefinition.from_grid_spec should produce correct block count."""
        from block_model_viewer.models.block_model_definition import BlockModelDefinition
        spec = {
            "nx": 10, "ny": 10, "nz": 5,
            "dx": 25.0, "dy": 25.0, "dz": 10.0,
            "x0": 0.0, "y0": 0.0, "z0": 0.0,
        }
        bmd = BlockModelDefinition.from_grid_spec(spec)
        assert bmd.n_blocks == 500, \
            f"BM-06 FAIL: Expected 500 blocks, got {bmd.n_blocks}"
        assert bmd.centres.shape == (500, 3), \
            f"BM-06 FAIL: Centres shape wrong: {bmd.centres.shape}"
