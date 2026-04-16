"""Tests for ARBF GPR (posterior mean and variance)."""

import numpy as np
import pytest

from geostats.arbf.gpr import (
    assemble_kernel_matrix,
    build_polynomial_matrix,
    factorise_and_solve,
    predict_mean,
    predict_mean_and_variance,
    predict_variance,
)


def _make_synthetic_data(n=30, seed=42):
    """Generate synthetic 3D data for testing."""
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0, 100, (n, 3))
    # Simple linear trend with noise
    values = 2.0 * coords[:, 0] / 100.0 + rng.normal(0, 0.1, n)
    return coords, values


class TestPolynomialMatrix:
    """Test polynomial basis matrix construction."""

    def test_none_drift(self):
        coords = np.random.randn(10, 3)
        P = build_polynomial_matrix(coords, "none")
        assert P.shape == (10, 0)

    def test_constant_drift(self):
        coords = np.random.randn(10, 3)
        P = build_polynomial_matrix(coords, "constant")
        assert P.shape == (10, 1)
        np.testing.assert_array_equal(P[:, 0], 1.0)

    def test_linear_drift(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        P = build_polynomial_matrix(coords, "linear")
        assert P.shape == (2, 4)
        np.testing.assert_array_equal(P[:, 0], 1.0)
        np.testing.assert_array_equal(P[:, 1], coords[:, 0])


class TestKernelMatrixAssembly:
    """Test augmented kernel matrix assembly."""

    def test_symmetric(self):
        """Kernel matrix must be symmetric."""
        coords, values = _make_synthetic_data(20)
        K_aug, P = assemble_kernel_matrix(coords, sill=1.0, range_=50.0)
        np.testing.assert_allclose(K_aug, K_aug.T, atol=1e-12)

    def test_diagonal_includes_nugget(self):
        """Diagonal should include sill + nugget + accuracy."""
        coords, values = _make_synthetic_data(10)
        sill, nugget, accuracy = 2.0, 0.5, 1e-6
        K_aug, P = assemble_kernel_matrix(
            coords, sill=sill, nugget=nugget, accuracy=accuracy,
        )
        N = coords.shape[0]
        expected_diag = sill + nugget + accuracy
        np.testing.assert_allclose(
            np.diag(K_aug)[:N], expected_diag, rtol=1e-10,
        )

    def test_augmented_shape(self):
        """K_aug shape = (N+M, N+M)."""
        coords, values = _make_synthetic_data(15)
        K_aug, P = assemble_kernel_matrix(coords, drift_type="linear")
        assert K_aug.shape == (15 + 4, 15 + 4)


class TestFactoriseAndSolve:
    """Test Cholesky factorisation and solve."""

    def test_produces_valid_factorisation(self):
        """Factorisation: Cholesky L for M==0, (lu, piv) for M>0."""
        coords, values = _make_synthetic_data(20)
        N = len(values)
        # With drift (default constant, M=1) → LU factorisation
        K_aug, P = assemble_kernel_matrix(coords, sill=1.0, range_=50.0)
        fact, weights, poly_coeffs = factorise_and_solve(K_aug, values)
        M = P.shape[1]
        assert isinstance(fact, tuple), "M>0 should return (lu, piv) tuple"
        lu, piv = fact
        assert lu.shape == (N + M, N + M)

        # Without drift (M=0) → Cholesky L
        K_aug_nd, P_nd = assemble_kernel_matrix(
            coords, sill=1.0, range_=50.0, drift_type="none",
        )
        fact_nd, w_nd, pc_nd = factorise_and_solve(K_aug_nd, values, drift_type="none")
        assert isinstance(fact_nd, np.ndarray), "M==0 should return L (ndarray)"
        assert fact_nd.shape == (N, N)
        # Verify lower triangular
        np.testing.assert_allclose(fact_nd, np.tril(fact_nd), atol=1e-15)

    def test_solution_satisfies_system(self):
        """L L^T [w; c] should equal [z; 0]."""
        coords, values = _make_synthetic_data(15)
        K_aug, P = assemble_kernel_matrix(
            coords, sill=1.0, range_=50.0, drift_type="constant",
        )
        L, weights, poly_coeffs = factorise_and_solve(
            K_aug, values, drift_type="constant",
        )
        N = len(values)
        M = 1  # constant drift
        x = np.concatenate([weights, poly_coeffs])
        rhs = np.zeros(N + M)
        rhs[:N] = values
        reconstructed = K_aug @ x
        np.testing.assert_allclose(reconstructed, rhs, atol=1e-6)


class TestPosteriorVariance:
    """Test the key innovation: posterior variance from Cholesky factor."""

    def test_variance_nonnegative(self):
        """Posterior variance must be >= 0 everywhere."""
        coords, values = _make_synthetic_data(30)
        K_aug, P = assemble_kernel_matrix(
            coords, sill=1.0, range_=50.0, nugget=0.1,
        )
        L, weights, poly_coeffs = factorise_and_solve(K_aug, values)

        query = np.random.RandomState(99).uniform(0, 100, (50, 3))
        variances = predict_variance(
            query, coords, L, sill=1.0, range_=50.0, nugget=0.1,
        )
        assert np.all(variances >= -1e-10)

    def test_variance_near_zero_at_data(self):
        """With nugget=0, variance should be ~0 at data points."""
        coords, values = _make_synthetic_data(20)
        K_aug, P = assemble_kernel_matrix(
            coords, sill=1.0, range_=50.0, nugget=0.0, accuracy=1e-8,
        )
        L, weights, poly_coeffs = factorise_and_solve(K_aug, values)

        variances = predict_variance(
            coords, coords, L, sill=1.0, range_=50.0, nugget=0.0,
        )
        # Should be close to zero at data locations
        assert np.mean(variances) < 0.05

    def test_variance_approaches_sill_far_away(self):
        """Far from data, variance should approach the sill."""
        coords = np.array([[50.0, 50.0, 50.0]])
        values = np.array([1.0])
        sill = 2.0
        K_aug, P = assemble_kernel_matrix(
            coords, sill=sill, range_=10.0, nugget=0.0, accuracy=1e-8,
        )
        L, weights, poly_coeffs = factorise_and_solve(K_aug, values)

        far_point = np.array([[5000.0, 5000.0, 5000.0]])
        var_far = predict_variance(
            far_point, coords, L, sill=sill, range_=10.0,
        )
        # Should be close to sill
        assert var_far[0] > sill * 0.8

    def test_mean_and_variance_consistent(self):
        """predict_mean_and_variance should match separate calls."""
        coords, values = _make_synthetic_data(25)
        K_aug, P = assemble_kernel_matrix(
            coords, sill=1.0, range_=50.0, nugget=0.1,
        )
        L, weights, poly_coeffs = factorise_and_solve(K_aug, values)

        query = np.random.RandomState(99).uniform(0, 100, (20, 3))

        mean_sep = predict_mean(
            query, coords, weights, poly_coeffs,
            sill=1.0, range_=50.0,
        )
        var_sep = predict_variance(
            query, coords, L, sill=1.0, range_=50.0, nugget=0.1,
        )
        mean_joint, var_joint = predict_mean_and_variance(
            query, coords, weights, poly_coeffs, L,
            sill=1.0, range_=50.0, nugget=0.1,
        )

        np.testing.assert_allclose(mean_sep, mean_joint, rtol=1e-10)
        np.testing.assert_allclose(var_sep, var_joint, rtol=1e-10)
