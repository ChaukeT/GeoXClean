"""Tests for ARBF data transforms."""

import numpy as np
import pytest

from geostats.arbf.transforms import (
    NormalScoreTable,
    ilr_forward,
    ilr_inverse,
    normal_score_backtransform,
    normal_score_transform,
)


class TestNormalScoreTransform:
    """Test normal-score transform and back-transform."""

    def test_round_trip_preserves_values(self):
        """Back-transforming normal scores recovers original values."""
        rng = np.random.RandomState(42)
        values = rng.lognormal(mean=2.0, sigma=0.5, size=200)

        ns_values, table = normal_score_transform(values, seed=42)
        recovered = normal_score_backtransform(ns_values, table)

        np.testing.assert_allclose(
            np.sort(recovered), np.sort(values), rtol=0.05,
        )

    def test_output_is_standard_normal(self):
        """Normal scores should be approximately N(0,1)."""
        rng = np.random.RandomState(42)
        values = rng.lognormal(mean=2.0, sigma=0.5, size=1000)

        ns_values, _ = normal_score_transform(values, seed=42)

        assert abs(np.mean(ns_values)) < 0.1
        assert abs(np.std(ns_values) - 1.0) < 0.15

    def test_clipping(self):
        """Normal scores are clipped to [-6, 6]."""
        rng = np.random.RandomState(42)
        values = rng.lognormal(mean=2.0, sigma=0.5, size=100)

        ns_values, _ = normal_score_transform(values, seed=42)

        assert np.all(ns_values >= -6.0)
        assert np.all(ns_values <= 6.0)

    def test_deterministic(self):
        """Same seed produces identical results."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ns1, _ = normal_score_transform(values, seed=123)
        ns2, _ = normal_score_transform(values, seed=123)
        np.testing.assert_array_equal(ns1, ns2)

    def test_backtransform_clamps(self):
        """Back-transform clamps to observed data range."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, table = normal_score_transform(values, seed=42)

        # Extreme normal scores
        extreme_ns = np.array([-10.0, 10.0])
        recovered = normal_score_backtransform(extreme_ns, table)

        assert recovered[0] >= 1.0 - 1e-10
        assert recovered[1] <= 5.0 + 1e-10


class TestILRTransform:
    """Test Isometric Log-Ratio transform."""

    def test_round_trip(self):
        """ilr_inverse(ilr_forward(x)) ~= x."""
        rng = np.random.RandomState(42)
        # Generate 3-part compositions summing to 100
        raw = rng.exponential(size=(50, 3))
        compositions = 100.0 * raw / raw.sum(axis=1, keepdims=True)

        ilr_vals = ilr_forward(compositions, kappa=100.0)
        recovered = ilr_inverse(ilr_vals, kappa=100.0)

        np.testing.assert_allclose(recovered, compositions, rtol=1e-6)

    def test_output_dimension(self):
        """ILR reduces D components to D-1."""
        D = 5
        compositions = np.ones((10, D)) * (100.0 / D)
        ilr_vals = ilr_forward(compositions, kappa=100.0)
        assert ilr_vals.shape == (10, D - 1)

    def test_inverse_guarantees_positive(self):
        """Inverse ILR produces strictly positive values."""
        rng = np.random.RandomState(42)
        ilr_vals = rng.randn(20, 3)
        compositions = ilr_inverse(ilr_vals, kappa=100.0)

        assert np.all(compositions > 0)

    def test_inverse_guarantees_closure(self):
        """Inverse ILR rows sum to kappa."""
        rng = np.random.RandomState(42)
        ilr_vals = rng.randn(20, 3)
        compositions = ilr_inverse(ilr_vals, kappa=100.0)

        np.testing.assert_allclose(
            compositions.sum(axis=1), 100.0, rtol=1e-10,
        )

    def test_handles_zeros_with_replacement(self):
        """ILR forward applies multiplicative zero replacement for zeros."""
        compositions = np.array([[50.0, 50.0, 0.0]])
        # Should not raise; zeros are replaced via multiplicative replacement
        result = ilr_forward(compositions)
        assert result.shape == (1, 2)
        assert np.all(np.isfinite(result))

    def test_single_sample(self):
        """Works with single-sample input."""
        comp = np.array([40.0, 30.0, 30.0])
        ilr_vals = ilr_forward(comp, kappa=100.0)
        recovered = ilr_inverse(ilr_vals, kappa=100.0)
        np.testing.assert_allclose(recovered.ravel(), comp, rtol=1e-6)
