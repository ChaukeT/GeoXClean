"""
Variogram Determinism Tests for JORC/SAMREC Compliance.

These tests verify that variogram calculations are 100% deterministic:
- Same inputs + same seed = bit-for-bit identical outputs
- Different seeds produce different results
- Determinism holds across all variogram functions

CRITICAL: These tests must pass on every commit touching geostats code.
Failure indicates a regression in reproducibility which violates audit requirements.

Run with: pytest tests/test_variogram_determinism.py -v
"""

import unittest
import numpy as np
from typing import Tuple


def generate_test_data(n_points: int = 500, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate reproducible test data for variogram testing.

    Parameters
    ----------
    n_points : int
        Number of test points
    seed : int
        Random seed for data generation

    Returns
    -------
    coords : np.ndarray
        (N, 3) array of coordinates
    values : np.ndarray
        (N,) array of values
    """
    rng = np.random.default_rng(seed)

    # Generate clustered point cloud (mimics drillhole data)
    coords = np.zeros((n_points, 3))
    coords[:, 0] = rng.uniform(0, 1000, n_points)  # X
    coords[:, 1] = rng.uniform(0, 1000, n_points)  # Y
    coords[:, 2] = rng.uniform(0, 200, n_points)   # Z

    # Generate spatially correlated values (mimics grade distribution)
    # Base trend + noise
    values = (
        0.5 * coords[:, 0] / 1000 +  # X trend
        0.3 * coords[:, 1] / 1000 +  # Y trend
        rng.normal(0, 0.2, n_points)  # Random noise
    )

    return coords, values


class TestUtilsVariogramDeterminism(unittest.TestCase):
    """Test determinism of utils/variogram_functions.py"""

    def setUp(self):
        """Generate test data."""
        self.coords, self.values = generate_test_data(n_points=500, seed=123)

    def test_calculate_experimental_variogram_determinism(self):
        """
        Same inputs + same seed = identical outputs.

        This is the PRIMARY determinism test.
        """
        from block_model_viewer.utils.variogram_functions import calculate_experimental_variogram

        # Run twice with same inputs and seed
        result1 = calculate_experimental_variogram(
            self.coords, self.values,
            n_lags=15, lag_tolerance=0.5, max_samples=500,
            random_state=42
        )

        result2 = calculate_experimental_variogram(
            self.coords, self.values,
            n_lags=15, lag_tolerance=0.5, max_samples=500,
            random_state=42
        )

        # Unpack tuples
        lag_dist1, semivars1, counts1 = result1
        lag_dist2, semivars2, counts2 = result2

        # CRITICAL: Byte-for-byte equality, NOT np.allclose()
        self.assertTrue(
            np.array_equal(lag_dist1, lag_dist2),
            f"Lag distances differ between runs!\n"
            f"Run 1: {lag_dist1[:5]}...\n"
            f"Run 2: {lag_dist2[:5]}..."
        )

        self.assertTrue(
            np.array_equal(semivars1, semivars2),
            f"Semivariances differ between runs!\n"
            f"Run 1: {semivars1[:5]}...\n"
            f"Run 2: {semivars2[:5]}..."
        )

        self.assertTrue(
            np.array_equal(counts1, counts2),
            f"Pair counts differ between runs!\n"
            f"Run 1: {counts1[:5]}...\n"
            f"Run 2: {counts2[:5]}..."
        )

    def test_different_seeds_produce_different_results(self):
        """
        Different seeds should produce different subsampling.

        This verifies the seed is actually being used.
        """
        from block_model_viewer.utils.variogram_functions import calculate_experimental_variogram

        # Use small max_samples to force subsampling
        result1 = calculate_experimental_variogram(
            self.coords, self.values,
            n_lags=15, max_samples=100,
            random_state=42
        )

        result2 = calculate_experimental_variogram(
            self.coords, self.values,
            n_lags=15, max_samples=100,
            random_state=99  # Different seed
        )

        _, semivars1, _ = result1
        _, semivars2, _ = result2

        # Results should differ (subsampling is different)
        # Note: With very large datasets, results could coincidentally match,
        # but with 500 points subsampled to 100, they should differ
        self.assertFalse(
            np.array_equal(semivars1, semivars2),
            "Different seeds should produce different subsampling results!"
        )

    def test_compute_cross_variogram_determinism(self):
        """Cross-variogram must also be deterministic."""
        from block_model_viewer.utils.variogram_functions import compute_cross_variogram

        # Create second variable
        values2 = self.values + np.random.default_rng(456).normal(0, 0.1, len(self.values))

        params = {
            'n_lags': 15,
            'max_samples': 500,
            'random_state': 42
        }

        result1 = compute_cross_variogram(self.values, values2, self.coords, params)
        result2 = compute_cross_variogram(self.values, values2, self.coords, params)

        self.assertTrue(
            np.array_equal(result1['lag_distances'], result2['lag_distances']),
            "Cross-variogram lag distances are not deterministic!"
        )

        self.assertTrue(
            np.array_equal(result1['semivariances'], result2['semivariances']),
            "Cross-variogram semivariances are not deterministic!"
        )


class TestModelsVariogramDeterminism(unittest.TestCase):
    """Test determinism of models/variogram_functions.py"""

    def setUp(self):
        """Generate test data."""
        self.coords, self.values = generate_test_data(n_points=500, seed=123)

    def test_pairwise_variogram_determinism(self):
        """_pairwise_variogram must be deterministic."""
        from block_model_viewer.models.variogram_functions import _pairwise_variogram

        result1 = _pairwise_variogram(
            self.values, self.coords,
            max_pairs=1000, max_samples=200,
            random_state=42
        )

        result2 = _pairwise_variogram(
            self.values, self.coords,
            max_pairs=1000, max_samples=200,
            random_state=42
        )

        dists1, semis1 = result1
        dists2, semis2 = result2

        self.assertTrue(
            np.array_equal(dists1, dists2),
            "Pairwise distances are not deterministic!"
        )

        self.assertTrue(
            np.array_equal(semis1, semis2),
            "Pairwise semivariances are not deterministic!"
        )

    def test_sorted_pairs_array_consistency(self):
        """_sorted_pairs_array must produce consistent ordering."""
        from block_model_viewer.models.variogram_functions import _sorted_pairs_array

        # Create test set (order undefined)
        pairs_set = {(1, 5), (0, 3), (2, 4), (0, 1), (3, 5)}

        result1 = _sorted_pairs_array(pairs_set)
        result2 = _sorted_pairs_array(pairs_set)

        self.assertTrue(
            np.array_equal(result1, result2),
            "_sorted_pairs_array is not deterministic!"
        )

        # Verify lexicographic ordering
        expected = np.array([[0, 1], [0, 3], [1, 5], [2, 4], [3, 5]])
        self.assertTrue(
            np.array_equal(result1, expected),
            f"Expected lexicographic order:\n{expected}\nGot:\n{result1}"
        )


class TestVariogram3DDeterminism(unittest.TestCase):
    """Test determinism of models/variogram3d.py"""

    def setUp(self):
        """Generate test data."""
        self.coords, self.values = generate_test_data(n_points=500, seed=123)

    def test_variogram3d_omnidirectional_determinism(self):
        """Variogram3D omnidirectional calculation must be deterministic."""
        from block_model_viewer.models.variogram3d import Variogram3D

        vgm = Variogram3D(
            n_lags=12,
            lag_distance=50.0,
            pair_cap=10000,
            random_state=42
        )

        result1 = vgm.calculate_omnidirectional(self.coords, self.values)
        result2 = vgm.calculate_omnidirectional(self.coords, self.values)

        self.assertTrue(
            np.array_equal(result1['distance'].values, result2['distance'].values),
            "Variogram3D omni distances are not deterministic!"
        )

        self.assertTrue(
            np.array_equal(result1['gamma'].values, result2['gamma'].values),
            "Variogram3D omni gamma values are not deterministic!"
        )

    def test_variogram3d_directional_determinism(self):
        """Variogram3D directional calculation must be deterministic."""
        from block_model_viewer.models.variogram3d import Variogram3D

        vgm = Variogram3D(
            n_lags=12,
            lag_distance=50.0,
            max_directional_samples=300,
            random_state=42
        )

        # Test major direction
        result1 = vgm.calculate_directional(
            self.coords, self.values, None,
            azimuth_deg=45.0, dip_deg=0.0, cone_tolerance=22.5,
            n_lags=12, max_range=600.0
        )

        result2 = vgm.calculate_directional(
            self.coords, self.values, None,
            azimuth_deg=45.0, dip_deg=0.0, cone_tolerance=22.5,
            n_lags=12, max_range=600.0
        )

        self.assertTrue(
            np.array_equal(result1['distance'].values, result2['distance'].values),
            "Variogram3D directional distances are not deterministic!"
        )

        self.assertTrue(
            np.array_equal(result1['gamma'].values, result2['gamma'].values),
            "Variogram3D directional gamma values are not deterministic!"
        )


class TestVariogramAssistantDeterminism(unittest.TestCase):
    """Test determinism of geostats/variogram_assistant.py"""

    def setUp(self):
        """Generate test data."""
        self.coords, self.values = generate_test_data(n_points=300, seed=123)

    def test_cross_validate_variogram_determinism(self):
        """Cross-validation must be deterministic for reproducible model selection."""
        try:
            from block_model_viewer.geostats.variogram_assistant import cross_validate_variogram
            from block_model_viewer.geostats.variogram_model import VariogramCandidateModel
        except ImportError:
            self.skipTest("variogram_assistant not available")

        # Create simple candidate model
        candidate = VariogramCandidateModel(
            model_type='spherical',
            ranges=[200.0],
            sills=[0.5],
            nugget=0.1
        )

        # Run cross-validation twice
        result1 = cross_validate_variogram(
            self.coords, self.values, candidate,
            n_folds=5, random_state=42
        )

        result2 = cross_validate_variogram(
            self.coords, self.values, candidate,
            n_folds=5, random_state=42
        )

        # CV scores should be identical
        self.assertEqual(
            result1['mse'], result2['mse'],
            f"Cross-validation MSE is not deterministic!\n"
            f"Run 1: {result1['mse']}\n"
            f"Run 2: {result2['mse']}"
        )


class TestMetadataDeterminismTracking(unittest.TestCase):
    """Test that metadata correctly tracks determinism settings."""

    def test_metadata_includes_random_state(self):
        """Metadata must track random_state for audit trail."""
        try:
            import pandas as pd
            from block_model_viewer.models.variogram3d import calculate_3d_variogram
        except ImportError:
            self.skipTest("variogram3d not available")

        coords, values = generate_test_data(n_points=200, seed=123)

        # Create DataFrame
        df = pd.DataFrame({
            'X': coords[:, 0],
            'Y': coords[:, 1],
            'Z': coords[:, 2],
            'GRADE': values
        })

        results = calculate_3d_variogram(
            df, 'X', 'Y', 'Z', 'GRADE',
            nlag=10, lag_distance=50.0,
            random_state=42
        )

        metadata = results.get('metadata', {})

        self.assertIn('random_state', metadata,
            "Metadata must include random_state for audit trail!")

        self.assertEqual(metadata['random_state'], 42,
            "Metadata random_state should match input!")

        self.assertIn('is_deterministic', metadata,
            "Metadata must include is_deterministic flag!")

        self.assertTrue(metadata['is_deterministic'],
            "is_deterministic should be True when random_state is set!")


if __name__ == '__main__':
    unittest.main(verbosity=2)
