"""Tests for ARBF kernel functions."""

import numpy as np
import pytest

from geostats.arbf.kernels import (
    cubic,
    evaluate_kernel,
    gaussian_kernel,
    kernel_at_zero,
    matern_32,
    matern_52,
    spheroidal,
    supported_kernels,
    wendland_c2,
)


class TestKernelProperties:
    """Test fundamental properties all kernels must satisfy."""

    KERNELS = [
        ("spheroidal", lambda r: spheroidal(r, alpha=1.0)),
        ("gaussian", gaussian_kernel),
        ("matern_32", matern_32),
        ("matern_52", matern_52),
        ("cubic", cubic),
        ("wendland_c2", wendland_c2),
    ]

    @pytest.mark.parametrize("name,fn", KERNELS)
    def test_phi_zero_equals_one(self, name, fn):
        """phi(0) = 1 for all kernels."""
        assert fn(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-15)

    @pytest.mark.parametrize("name,fn", KERNELS)
    def test_positive_values(self, name, fn):
        """phi(r) >= 0 for all r >= 0."""
        r = np.linspace(0, 10, 1000)
        assert np.all(fn(r) >= -1e-15)

    @pytest.mark.parametrize("name,fn", KERNELS)
    def test_monotonically_decreasing(self, name, fn):
        """phi(r) is monotonically non-increasing for r > 0."""
        r = np.linspace(0.001, 5, 500)
        vals = fn(r)
        diffs = np.diff(vals)
        assert np.all(diffs <= 1e-10), f"{name}: not monotonically decreasing"

    @pytest.mark.parametrize("name,fn", KERNELS)
    def test_bounded_zero_one(self, name, fn):
        """phi(r) in [0, 1] for all r >= 0."""
        r = np.linspace(0, 100, 10000)
        vals = fn(r)
        assert np.all(vals >= -1e-15)
        assert np.all(vals <= 1.0 + 1e-15)


class TestSpheroidalKernel:
    """Test spheroidal kernel specifically."""

    def test_alpha_parameter(self):
        """Higher alpha = faster decay."""
        r = np.array([1.0])
        v_low = spheroidal(r, alpha=0.5)
        v_high = spheroidal(r, alpha=5.0)
        assert v_low > v_high, "Higher alpha should produce faster decay"

    def test_approaches_gaussian(self):
        """Large alpha approaches Gaussian kernel."""
        r = np.array([0.5])
        v_large_alpha = spheroidal(r, alpha=50.0)
        v_gaussian = gaussian_kernel(r)
        # Not exact, but should be close in character
        assert abs(v_large_alpha) < 0.1

    def test_formula(self):
        """phi(r) = (1 + r^2)^{-alpha}."""
        r = np.array([2.0])
        alpha = 3.0
        expected = (1.0 + 4.0) ** (-3.0)
        assert spheroidal(r, alpha)[0] == pytest.approx(expected)


class TestCompactKernels:
    """Test compact support kernels."""

    def test_cubic_zero_outside(self):
        """Cubic kernel is zero for r >= 1."""
        r = np.array([1.0, 1.5, 2.0, 10.0])
        assert np.all(cubic(r) == 0.0)

    def test_wendland_zero_outside(self):
        """Wendland C2 is zero for r >= 1."""
        r = np.array([1.0, 1.5, 2.0, 10.0])
        assert np.all(wendland_c2(r) == 0.0)

    def test_cubic_inside(self):
        """Cubic kernel positive for r < 1."""
        r = np.array([0.0, 0.25, 0.5, 0.75, 0.99])
        assert np.all(cubic(r) > 0)

    def test_wendland_inside(self):
        """Wendland C2 positive for r < 1."""
        r = np.array([0.0, 0.25, 0.5, 0.75, 0.99])
        assert np.all(wendland_c2(r) > 0)


class TestDispatcher:
    """Test kernel dispatcher."""

    def test_all_kernels_accessible(self):
        """All supported kernels can be evaluated via dispatcher."""
        r = np.array([0.0, 0.5, 1.0])
        for name in supported_kernels():
            vals = evaluate_kernel(r, kernel_type=name)
            assert vals[0] == pytest.approx(1.0, abs=1e-12)

    def test_unknown_kernel_raises(self):
        """Unknown kernel type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            evaluate_kernel(np.array([0.0]), kernel_type="foobar")

    def test_kernel_at_zero(self):
        """kernel_at_zero always returns 1.0."""
        for name in supported_kernels():
            assert kernel_at_zero(name) == 1.0
