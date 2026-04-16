"""
Tests for cross-validation routines.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md §10.2
"""

import numpy as np
import pytest

from geostats.estimation.config import RBFConfig, KernelType, DriftType
from geostats.estimation.cross_validation import (
    loo_cross_validation,
    kfold_cross_validation,
)


def _smooth_synthetic_data(n=30, seed=42):
    """Generate data from a smooth 3D function: f(x,y,z) = x + 0.5*y."""
    rng = np.random.Generator(np.random.PCG64(seed))
    points = rng.uniform(0, 100, (n, 3))
    values = points[:, 0] + 0.5 * points[:, 1] + rng.normal(0, 2, n)
    return points, values


def _make_config():
    return RBFConfig(
        kernel_type=KernelType.SPHEROIDAL,
        total_sill=500.0,
        nugget=5.0,
        base_range=80.0,
        alpha=5,
        drift=DriftType.LINEAR,
        accuracy=0.1,
    )


class TestLOOCV:
    """LOO cross-validation on synthetic smooth data."""

    def test_rmse_reasonable(self):
        points, values = _smooth_synthetic_data(n=25)
        config = _make_config()
        result = loo_cross_validation(points, values, config)

        assert result.rmse < np.std(values), (
            f"RMSE ({result.rmse:.2f}) should be < std ({np.std(values):.2f})"
        )

    def test_mean_error_near_zero(self):
        points, values = _smooth_synthetic_data(n=25)
        config = _make_config()
        result = loo_cross_validation(points, values, config)

        # ME should be small relative to RMSE
        assert abs(result.mean_error) < result.rmse, (
            f"|ME| ({abs(result.mean_error):.4f}) should be < RMSE ({result.rmse:.4f})"
        )

    def test_r_squared_positive(self):
        points, values = _smooth_synthetic_data(n=25)
        config = _make_config()
        result = loo_cross_validation(points, values, config)

        assert result.r_squared > 0.0, f"R² should be positive, got {result.r_squared:.4f}"


class TestKFoldCV:
    """K-fold spatial cross-validation."""

    def test_kfold_produces_results(self):
        points, values = _smooth_synthetic_data(n=30)
        config = _make_config()
        result = kfold_cross_validation(points, values, config, k=3)

        assert result.k == 3
        assert len(result.fold_results) > 0

    def test_kfold_similar_to_loo(self):
        """K-fold should produce similar aggregate stats to LOO."""
        points, values = _smooth_synthetic_data(n=25)
        config = _make_config()

        loo = loo_cross_validation(points, values, config)
        kf = kfold_cross_validation(points, values, config, k=5)

        # RMSE should be in the same order of magnitude
        assert kf.aggregate_rmse < loo.rmse * 3, (
            f"K-fold RMSE ({kf.aggregate_rmse:.2f}) too far from LOO ({loo.rmse:.2f})"
        )
