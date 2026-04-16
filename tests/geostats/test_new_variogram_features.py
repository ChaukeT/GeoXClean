"""
Tests for newly implemented variogram features:
1. Hole effect model
2. Madogram estimator
3. Pairwise relative variogram
4. Correlogram
5. Non-ergodic correction
6. Spatial block CV
"""

import numpy as np
import pandas as pd
import pytest

import block_model_viewer.geostats  # noqa: F401 — resolve circular imports


def _synthetic_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0, 300, size=(n, 3))
    coords[:, 2] *= 0.3
    values = rng.randn(n) * 5 + 20  # mean=20, std=5
    return coords, values


# ── 1. Hole effect model ─────────────────────────────────────────────

class TestHoleEffectModel:
    def test_in_model_map(self):
        from block_model_viewer.geostats.variogram_model import MODEL_MAP
        assert "hole_effect" in MODEL_MAP

    def test_correct_formula(self):
        from block_model_viewer.geostats.variogram_model import hole_effect_model
        h = np.array([0.0, 50.0, 100.0, 200.0])
        nugget, sill, range_ = 0.5, 2.0, 100.0
        gamma = hole_effect_model(h, range_, sill, nugget)

        # At h=0: gamma should be nugget (model convention)
        # At large h: should oscillate around sill
        assert gamma.shape == (4,)
        assert np.all(np.isfinite(gamma))
        # At h=range: cosine term cos(2*pi*1) = 1, exp(-3) ≈ 0.05
        # gamma ≈ nugget + c * (1 - 0.05*1) ≈ nugget + c*0.95
        assert gamma[2] > nugget  # At h=100, well above nugget

    def test_oscillation(self):
        """Hole effect model should oscillate (gamma can decrease then increase)."""
        from block_model_viewer.geostats.variogram_model import hole_effect_model
        h = np.linspace(1, 300, 500)
        gamma = hole_effect_model(h, 100.0, 2.0, 0.0)
        # Check for at least one local minimum after the first peak
        dg = np.diff(gamma)
        sign_changes = np.sum(np.diff(np.sign(dg)) != 0)
        assert sign_changes >= 2, f"Hole effect should oscillate, got {sign_changes} sign changes"


# ── 2. Madogram estimator ────────────────────────────────────────────

class TestMadogram:
    def test_madogram_estimator(self):
        from block_model_viewer.geostats.variogram_engine_v2 import (
            VariogramEngine, VariogramSearchConfig, DirectionSpec,
        )
        coords, values = _synthetic_data()
        engine = VariogramEngine(coords=coords, values=values)
        omni = DirectionSpec("omni", np.array([1.0, 0.0, 0.0]), tolerance_deg=180.0)

        search = VariogramSearchConfig(
            lag_size=25.0, n_lags=8, estimator="madogram",
        )
        exp = engine.compute_experimental_variogram(direction=omni, search=search)

        assert len(exp.lag_centres) > 0
        assert np.all(np.isfinite(exp.gamma[exp.n_pairs > 0]))
        # Madogram values should be positive
        valid = exp.gamma[exp.n_pairs > 0]
        assert np.all(valid >= 0), f"Madogram produced negative values: {valid}"


# ── 3. Pairwise relative variogram — OBSOLETE ────────────────────────
# ``compute_relative_variogram`` lived only in the legacy
# ``variogram_functions.py`` module and had no production callers.
# Removed with the variogram consolidation (D3). If we want relative
# variograms in the future, reimplement them as a shared helper in
# geostats/experimental_variogram.py.


@pytest.mark.skip(reason="Legacy-only function removed in variogram consolidation (D3)")
class TestRelativeVariogram:
    def test_compute_relative_variogram(self):
        pass

    def test_rejects_negative_values_gracefully(self):
        pass


# ── 4. Correlogram — OBSOLETE ────────────────────────────────────────
# Same story: ``compute_correlogram`` was a legacy-only helper with no
# production callers. Removed with D3.


@pytest.mark.skip(reason="Legacy-only function removed in variogram consolidation (D3)")
class TestCorrelogram:
    def test_compute_correlogram(self):
        pass


# ── 5. Non-ergodic correction ────────────────────────────────────────

class TestNonErgodicCorrection:
    def test_correction_increases_gamma(self):
        from block_model_viewer.geostats.variogram_engine_v2 import (
            VariogramEngine, VariogramSearchConfig, DirectionSpec,
        )
        coords, values = _synthetic_data(n=100)
        engine = VariogramEngine(coords=coords, values=values)
        omni = DirectionSpec("omni", np.array([1.0, 0.0, 0.0]), tolerance_deg=180.0)
        search = VariogramSearchConfig(lag_size=25.0, n_lags=8)

        exp = engine.compute_experimental_variogram(direction=omni, search=search)
        corrected = engine.apply_non_ergodic_correction(exp)

        # Corrected gamma should be >= original gamma (correction factor >= 1)
        valid = exp.n_pairs > 0
        if np.any(valid):
            assert np.all(corrected.gamma[valid] >= exp.gamma[valid] - 1e-10)


# ── 6. Spatial block CV ─────────────────────────────────────────────

class TestSpatialBlockCV:
    def test_spatial_block_folds(self):
        from block_model_viewer.geostats.variogram_assistant import _spatial_block_folds
        coords, _ = _synthetic_data(n=100)
        folds = _spatial_block_folds(coords, n_folds=5, random_state=42)

        assert folds.shape == (100,)
        assert len(np.unique(folds)) == 5
        # Each fold should have at least some points
        for k in range(5):
            assert np.sum(folds == k) >= 5

    def test_cv_mode_in_result(self):
        from block_model_viewer.geostats.variogram_assistant import (
            cross_validate_variogram, VariogramCandidateModel,
        )
        coords, values = _synthetic_data(n=100)
        model = VariogramCandidateModel(
            model_type="spherical",
            nugget=1.0, sills=[4.0], ranges=[80.0],
        )

        result = cross_validate_variogram(
            coords, values, model,
            n_folds=5, cv_mode="spatial_block", random_state=42,
        )
        assert result["cv_mode"] == "spatial_block"
        assert np.isfinite(result["rmse"])
        assert np.isfinite(result["mae"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
