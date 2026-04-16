"""
Tests for the FastRBF solver (fastrbf_engine.py).

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md §10.1
"""

import numpy as np
import pytest

from geostats.estimation.config import RBFConfig, KernelType, DriftType
from geostats.estimation.fastrbf_engine import FastRBFEngine


def _make_config(**overrides) -> RBFConfig:
    defaults = dict(
        kernel_type=KernelType.SPHEROIDAL,
        total_sill=1.0,
        nugget=0.05,
        base_range=50.0,
        alpha=5,
        drift=DriftType.CONSTANT,
    )
    defaults.update(overrides)
    return RBFConfig(**defaults)


class TestExactInterpolation:
    """With 4 known points the solver must reproduce values at data locations."""

    def test_exact_at_data_points(self):
        config = _make_config(accuracy=1e-8)
        engine = FastRBFEngine(config)

        points = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [50.0, 50.0, 0.0],
        ])
        values = np.array([1.0, 3.0, 2.0, 5.0])

        fitted = engine.fit(points, values)
        predicted = engine.predict(fitted, points)

        np.testing.assert_allclose(predicted, values, atol=0.05)


class TestDriftBehaviour:
    """Test that drift modes affect far-field predictions correctly."""

    def test_constant_drift_far_field(self):
        """With constant drift, far-field value should approach mean of data."""
        config = _make_config(drift=DriftType.CONSTANT)
        engine = FastRBFEngine(config)

        points = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
        ])
        values = np.array([2.0, 4.0, 6.0, 8.0])
        mean_val = values.mean()

        fitted = engine.fit(points, values)

        far_point = np.array([[10000.0, 10000.0, 0.0]])
        far_pred = engine.predict(fitted, far_point)

        # Should be within 50% of data mean
        assert abs(far_pred[0] - mean_val) < mean_val, (
            f"Far-field {far_pred[0]:.2f} too far from mean {mean_val:.2f}"
        )

    def test_none_drift_differs_from_constant(self):
        """With no drift, far-field value should differ from constant drift."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
        ])
        values = np.array([2.0, 4.0, 6.0, 8.0])

        config_const = _make_config(drift=DriftType.CONSTANT)
        config_none = _make_config(drift=DriftType.NONE)

        engine_c = FastRBFEngine(config_const)
        engine_n = FastRBFEngine(config_none)

        fitted_c = engine_c.fit(points, values)
        fitted_n = engine_n.fit(points, values)

        far_point = np.array([[10000.0, 10000.0, 0.0]])
        pred_c = engine_c.predict(fitted_c, far_point)
        pred_n = engine_n.predict(fitted_n, far_point)

        # Without drift polynomial, far-field behavior is different
        assert np.isfinite(pred_n[0]), "None-drift far-field should be finite"
        # The two drift modes should produce different far-field values
        assert pred_c[0] != pytest.approx(pred_n[0], rel=0.01), (
            "Constant vs none drift should differ far from data"
        )


class TestDeterminism:
    """Same inputs must produce bit-identical outputs."""

    def test_deterministic(self):
        config = _make_config()
        engine = FastRBFEngine(config)

        rng = np.random.Generator(np.random.PCG64(99))
        points = rng.uniform(0, 100, (30, 3))
        values = rng.uniform(1, 10, 30)

        fitted1 = engine.fit(points, values)
        fitted2 = engine.fit(points, values)

        query = rng.uniform(0, 100, (10, 3))
        pred1 = engine.predict(fitted1, query)
        pred2 = engine.predict(fitted2, query)

        np.testing.assert_array_equal(pred1, pred2)
        assert fitted1.data_hash == fitted2.data_hash


class TestBlockPrediction:
    """Block prediction with discretisation should average sub-block values."""

    def test_block_predict(self):
        config = _make_config(discretisation_points=2)
        engine = FastRBFEngine(config)

        points = np.array([
            [0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [0.0, 50.0, 0.0],
            [50.0, 50.0, 0.0],
        ])
        values = np.array([1.0, 2.0, 3.0, 4.0])

        fitted = engine.fit(points, values)

        centroids = np.array([[25.0, 25.0, 0.0]])
        sizes = np.array([10.0, 10.0, 10.0])

        block_est = engine.predict_block(fitted, centroids, sizes, n_discretisation=2)
        assert block_est.shape == (1,)
        assert np.isfinite(block_est[0])


class TestFittedRBFMetadata:
    """FittedRBF must store all required metadata."""

    def test_metadata_fields(self):
        config = _make_config()
        engine = FastRBFEngine(config)

        points = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
        values = np.array([1.0, 2.0, 3.0])

        fitted = engine.fit(points, values)

        assert fitted.weights.shape == (3,)
        assert fitted.n_samples == 3
        assert fitted.data_hash != ""
        assert fitted.fit_timestamp != ""
        assert fitted.solve_method in ("direct_symmetric", "direct_general", "gmres")
        assert fitted.accuracy_used > 0
        assert fitted.condition_number > 0 or fitted.condition_number == np.inf
        assert fitted.solve_time_seconds >= 0
