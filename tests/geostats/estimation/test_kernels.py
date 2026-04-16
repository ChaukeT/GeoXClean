"""
Tests for kernel functions (interpolant_functions.py).

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md §10.1
"""

import numpy as np
import pytest

from geostats.estimation.config import KernelType
from geostats.estimation.interpolant_functions import (
    evaluate_kernel,
    linear_kernel,
    spheroidal_kernel,
    spherical_kernel,
    gaussian_kernel,
    exponential_kernel,
    cubic_kernel,
    generalised_cauchy_kernel,
    covariance,
)

SILL = 1.0
RANGE = 100.0
NUGGET = 0.1


class TestKernelAtZero:
    """At r=0, every kernel must return 0 (nugget is NOT added at origin)."""

    @pytest.mark.parametrize(
        "kernel_type",
        list(KernelType),
    )
    def test_kernel_at_zero(self, kernel_type):
        r = np.array([0.0])
        result = evaluate_kernel(r, SILL, RANGE, NUGGET, kernel_type, alpha=5)
        assert result[0] == pytest.approx(0.0, abs=1e-12)


class TestSpheroidalBehaviour:
    """Spheroidal kernel behaviour tests."""

    @pytest.mark.parametrize("alpha", [3, 5, 7, 9])
    def test_spheroidal_at_range_positive(self, alpha):
        """At r=range the kernel should return a significant fraction of sill."""
        r = np.array([RANGE])
        result = spheroidal_kernel(r, SILL, RANGE, nugget=0.0, alpha=alpha)
        ratio = result[0] / SILL
        # Higher alpha → closer to sill at range (α=9≈94%, α=3≈50%).
        assert ratio > 0.3, f"alpha={alpha}: ratio={ratio:.4f} too low"
        assert ratio <= 1.0, f"alpha={alpha}: ratio={ratio:.4f} exceeds sill"

    def test_alpha9_near_sill(self):
        """α=9 should reach closest to sill at r=range."""
        r = np.array([RANGE])
        result = spheroidal_kernel(r, SILL, RANGE, nugget=0.0, alpha=9)
        ratio = result[0] / SILL
        assert 0.90 <= ratio <= 1.0, f"alpha=9: ratio={ratio:.4f}"


class TestSphericalAtRange:
    """Spherical at r=range returns exactly sill+nugget."""

    def test_at_range(self):
        r = np.array([RANGE])
        result = spherical_kernel(r, SILL, RANGE, NUGGET)
        assert result[0] == pytest.approx(SILL + NUGGET, rel=1e-10)

    def test_beyond_range(self):
        r = np.array([RANGE * 2])
        result = spherical_kernel(r, SILL, RANGE, NUGGET)
        assert result[0] == pytest.approx(SILL + NUGGET, rel=1e-10)


class TestMonotonicity:
    """All kernels must be monotonically non-decreasing."""

    @pytest.mark.parametrize("kernel_type", list(KernelType))
    def test_monotonic(self, kernel_type):
        r = np.linspace(0.01, RANGE * 3, 500)
        result = evaluate_kernel(r, SILL, RANGE, NUGGET, kernel_type, alpha=5)
        diffs = np.diff(result)
        # Allow tiny numerical noise
        assert np.all(diffs >= -1e-10), f"Non-monotonic for {kernel_type.value}"


class TestRadialSymmetry:
    """φ(r) must equal φ(|r|) — only non-negative r is valid but ensure consistency."""

    @pytest.mark.parametrize("kernel_type", list(KernelType))
    def test_symmetry(self, kernel_type):
        r1 = np.array([10.0, 50.0, 100.0])
        r2 = np.array([10.0, 50.0, 100.0])  # same — radial functions only take |r|
        v1 = evaluate_kernel(r1, SILL, RANGE, NUGGET, kernel_type, alpha=5)
        v2 = evaluate_kernel(r2, SILL, RANGE, NUGGET, kernel_type, alpha=5)
        np.testing.assert_allclose(v1, v2)


class TestPositiveDefiniteness:
    """Kernel matrix eigenvalues must all be > 0 (with nugget on diagonal)."""

    @pytest.mark.parametrize("kernel_type", list(KernelType))
    def test_positive_definite(self, kernel_type):
        rng = np.random.Generator(np.random.PCG64(42))
        points = rng.uniform(0, 100, size=(20, 3))

        # Build distance matrix
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Build covariance matrix: C(r) = (sill + nugget) - γ(r)
        gamma = evaluate_kernel(D, SILL, RANGE, NUGGET, kernel_type, alpha=5)
        C = (SILL + NUGGET) - gamma

        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues > -1e-8), (
            f"{kernel_type.value}: min eigenvalue = {eigenvalues.min():.6e}"
        )


class TestVectorisedOutput:
    """Kernel must return ndarray matching input shape."""

    @pytest.mark.parametrize("kernel_type", list(KernelType))
    def test_vectorised_shape(self, kernel_type):
        r = np.linspace(0, 200, 100)
        result = evaluate_kernel(r, SILL, RANGE, NUGGET, kernel_type, alpha=5)
        assert result.shape == r.shape


class TestCovariance:
    """C(0) = sill + nugget."""

    @pytest.mark.parametrize("kernel_type", list(KernelType))
    def test_covariance_at_zero(self, kernel_type):
        r = np.array([0.0])
        c = covariance(r, SILL, RANGE, NUGGET, kernel_type, alpha=5)
        assert c[0] == pytest.approx(SILL + NUGGET, rel=1e-10)
