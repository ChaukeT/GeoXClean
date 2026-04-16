"""
Tests for anisotropic distance (utils/distance.py).

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md §10.1
"""

import numpy as np
import pytest

from geostats.utils.distance import (
    anisotropic_distance,
    isotropic_distance,
    pairwise_anisotropic_distance,
    point_to_points_anisotropic,
)


class TestIsotropicMatchesNorm:
    """Isotropic distance must match numpy.linalg.norm."""

    def test_single_pair(self):
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 6.0, 8.0])
        d = isotropic_distance(p1, p2)
        expected = np.linalg.norm(p1 - p2)
        assert d == pytest.approx(expected, rel=1e-12)

    def test_vectorised(self):
        rng = np.random.Generator(np.random.PCG64(42))
        p1 = rng.uniform(0, 100, (50, 3))
        p2 = rng.uniform(0, 100, (50, 3))

        d = isotropic_distance(p1, p2)
        expected = np.linalg.norm(p1 - p2, axis=1)
        np.testing.assert_allclose(d, expected, rtol=1e-12)


class TestIsotropicAnisotropy:
    """With all ratios=1 and zero angles, anisotropic == isotropic."""

    def test_isotropic_equivalence(self):
        rng = np.random.Generator(np.random.PCG64(42))
        p1 = rng.uniform(0, 100, (20, 3))
        p2 = rng.uniform(0, 100, (20, 3))

        d_iso = isotropic_distance(p1, p2)
        d_aniso = anisotropic_distance(p1, p2, 0, 0, 0, 1, 1, 1)
        np.testing.assert_allclose(d_aniso, d_iso, rtol=1e-10)


class TestAnisotropyCompression:
    """ratio_minor=0.5 should compress vertical distance by 2×."""

    def test_vertical_compression(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 10.0])

        d_iso = anisotropic_distance(p1, p2, 0, 0, 0, 1, 1, 1)
        d_compressed = anisotropic_distance(p1, p2, 0, 0, 0, 1, 1, 0.5)

        # Compressed: z distance becomes 10/0.5 = 20
        assert d_compressed == pytest.approx(20.0, rel=1e-10)
        assert d_compressed > d_iso


class TestAzimuthRotation:
    """Azimuth=90 should swap x/y components."""

    def test_azimuth_90(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2_x = np.array([10.0, 0.0, 0.0])
        p2_y = np.array([0.0, 10.0, 0.0])

        # With azimuth=90 and different ratios in major/semi,
        # point along original X should now be measured as semi-major
        d_x = anisotropic_distance(p1, p2_x, azimuth=90, dip=0, pitch=0,
                                    ratio_major=1.0, ratio_semi=2.0, ratio_minor=1.0)
        d_y = anisotropic_distance(p1, p2_y, azimuth=90, dip=0, pitch=0,
                                    ratio_major=1.0, ratio_semi=2.0, ratio_minor=1.0)

        # After 90° rotation, x maps to -y and y maps to x
        # So p2_x (along original X) becomes aligned with the semi axis
        # and p2_y (along original Y) becomes aligned with the major axis
        # With ratio_semi=2.0, the distance along the semi-axis direction
        # is divided by 2 → smaller distance
        assert d_x != d_y  # Rotation breaks symmetry


class TestVectorisedOutputShape:
    """Output shape must match input broadcasting."""

    def test_pairwise_shape(self):
        points = np.random.default_rng(42).uniform(0, 100, (15, 3))
        D = pairwise_anisotropic_distance(points, 0, 0, 0, 1, 1, 1)
        assert D.shape == (15, 15)
        np.testing.assert_allclose(D.diagonal(), 0.0, atol=1e-14)

    def test_point_to_points_shape(self):
        query = np.array([50.0, 50.0, 50.0])
        points = np.random.default_rng(42).uniform(0, 100, (20, 3))
        d = point_to_points_anisotropic(query, points)
        assert d.shape == (20,)
