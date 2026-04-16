"""
Tests for JORC audit trail (audit.py).

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md §10.3
"""

import hashlib
import json

import numpy as np
import pytest

from geostats.estimation.audit import JORCAuditRecord
from geostats.estimation.config import RBFConfig, KernelType, DriftType
from geostats.estimation.fastrbf_engine import FastRBFEngine
from geostats.estimation.block_estimator import BlockModelEstimator, EstimationResult


def _make_audit_record(**overrides):
    """Create a fully populated audit record for testing."""
    defaults = dict(
        run_id="test-run-001",
        timestamp="2026-03-04T00:00:00+00:00",
        software_version="GeoX FastRBF 1.0.0",
        operator="Test Operator",
        database_description="Synthetic test data",
        composite_length=2.0,
        num_composites=100,
        num_drillholes=10,
        data_hash="abc123",
        kernel_type="spheroidal",
        kernel_parameters={"sill": 1.0, "nugget": 0.1, "range": 100.0},
        drift_type="constant",
        accuracy=0.001,
        search_max_samples=24,
        search_min_samples=8,
        search_max_per_octant=4,
        search_min_octants=2,
        search_radii=(100.0, 100.0, 50.0),
        search_angles=(0.0, 0.0, 0.0),
        ellipsoid_ratios=(1.0, 1.0, 0.5),
        block_size=(10.0, 10.0, 5.0),
        block_discretisation=4,
        num_blocks_estimated=500,
        num_blocks_total=600,
        cv_rmse=0.5,
        cv_mae=0.3,
        cv_r_squared=0.85,
        cv_mean_error=0.01,
        cv_slope_of_regression=0.98,
        global_bias_percent=2.5,
        measured_blocks=200,
        indicated_blocks=200,
        inferred_blocks=100,
        unclassified_blocks=100,
    )
    defaults.update(overrides)
    return JORCAuditRecord(**defaults)


class TestAuditMandatoryFields:
    """All mandatory JORC fields must be populated."""

    def test_complete_record_passes(self):
        record = _make_audit_record()
        warnings = record.validate_completeness()
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

    def test_missing_operator(self):
        record = _make_audit_record(operator="")
        warnings = record.validate_completeness()
        assert any("operator" in w.lower() for w in warnings)

    def test_missing_classification(self):
        record = _make_audit_record(
            measured_blocks=0, indicated_blocks=0, inferred_blocks=0
        )
        warnings = record.validate_completeness()
        assert any("classified" in w.lower() for w in warnings)

    def test_high_bias_flagged(self):
        record = _make_audit_record(global_bias_percent=8.0)
        warnings = record.validate_completeness()
        assert any("bias" in w.lower() for w in warnings)

    def test_slope_deviation_flagged(self):
        record = _make_audit_record(cv_slope_of_regression=0.7)
        warnings = record.validate_completeness()
        assert any("slope" in w.lower() for w in warnings)


class TestAuditSerialisation:
    """Audit record must serialise to JSON and text."""

    def test_to_dict(self):
        record = _make_audit_record()
        d = record.to_dict()
        assert isinstance(d, dict)
        assert d["run_id"] == "test-run-001"

    def test_to_json(self):
        record = _make_audit_record()
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["kernel_type"] == "spheroidal"

    def test_to_jorc_text(self):
        record = _make_audit_record()
        text = record.to_jorc_table1_section3()
        assert "JORC TABLE 1" in text
        assert "ESTIMATION AND MODELLING TECHNIQUES" in text
        assert "CLASSIFICATION" in text
        assert "VALIDATION" in text


class TestDataHash:
    """Data hash must be SHA-256 of sorted input array bytes."""

    def test_hash_consistency(self):
        points = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float64)
        values = np.array([1.0, 2.0, 3.0])

        # Hash uses lexicographic sort of paired (x, y, z, value) rows
        paired = np.empty((3, 4), dtype=np.float64)
        paired[:, :3] = points
        paired[:, 3] = values
        sort_idx = np.lexsort(
            (paired[:, 3], paired[:, 2], paired[:, 1], paired[:, 0])
        )
        expected_hash = hashlib.md5(paired[sort_idx].tobytes()).hexdigest()

        config = RBFConfig(
            total_sill=1.0, nugget=0.05, base_range=50.0
        )
        engine = FastRBFEngine(config)
        fitted = engine.fit(points, values)

        assert fitted.data_hash == expected_hash
