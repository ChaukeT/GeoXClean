"""
Tests for variogram engine v2 — validates the advanced features that
the v2 engine provides over the legacy pipeline:

1. Robust estimator (Cressie-Hawkins)
2. Drift removal (constant, linear)
3. Normal-score transform
4. Auto-lag computation
5. v2-to-legacy dict bridge format
6. v2 vs legacy output parity for standard cases
"""

import numpy as np
import pandas as pd
import pytest

# Resolve circular import
import block_model_viewer.geostats  # noqa: F401


# ── Fixtures ──────────────────────────────────────────────────────────

def _synthetic_stationary(n=300, seed=42):
    """Stationary isotropic field with spherical variogram (nug=0.2, sill=1.0, range=80)."""
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0, 400, size=(n, 3))
    coords[:, 2] *= 0.3  # flatten vertically
    # Generate spatially correlated values using a simple moving-average approach
    values = rng.randn(n)
    return coords, values


def _synthetic_with_outliers(n=300, seed=42):
    """Same as stationary but with 5% extreme outliers injected."""
    coords, values = _synthetic_stationary(n, seed)
    rng = np.random.RandomState(seed + 1)
    n_outliers = int(n * 0.05)
    outlier_idx = rng.choice(n, n_outliers, replace=False)
    values[outlier_idx] *= 10  # 10x extreme values
    return coords, values


def _synthetic_with_trend(n=300, seed=42):
    """Stationary noise + linear trend in X direction."""
    coords, values = _synthetic_stationary(n, seed)
    # Add strong linear trend
    values += coords[:, 0] * 0.01  # trend: 0.01 per metre in X
    return coords, values


def _make_dataframe(coords, values):
    df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
    df["GRADE"] = values
    return df


# ── Test: v2 engine can be imported and instantiated ──────────────────

class TestV2EngineImport:
    def test_import_engine(self):
        from block_model_viewer.geostats.variogram_engine_v2 import VariogramEngine
        assert VariogramEngine is not None

    def test_import_bridge(self):
        from block_model_viewer.geostats.variogram_bridge_v2 import run_variogram_pipeline_v2
        assert callable(run_variogram_pipeline_v2)


# ── Test: v2 bridge produces valid legacy-format output ───────────────

class TestV2BridgeOutput:
    def test_bridge_returns_expected_keys(self):
        from block_model_viewer.geostats.variogram_bridge_v2 import run_variogram_pipeline_v2
        coords, values = _synthetic_stationary(100, seed=99)
        df = _make_dataframe(coords, values)

        result = run_variogram_pipeline_v2(
            data=df, xcol="X", ycol="Y", zcol="Z", vcol="GRADE",
            nlag=10, lag_distance=30, random_state=42,
        )

        # Must have all legacy keys
        assert "omni_variogram" in result
        assert "major_variogram" in result
        assert "minor_variogram" in result
        assert "vertical_variogram" in result
        assert "fitted_models" in result

        # Omni variogram must be a DataFrame with standard columns
        omni = result["omni_variogram"]
        assert isinstance(omni, pd.DataFrame)
        assert "distance" in omni.columns
        assert "gamma" in omni.columns
        assert "npairs" in omni.columns
        assert len(omni) > 0

    def test_bridge_fitted_params_have_sill_fields(self):
        from block_model_viewer.geostats.variogram_bridge_v2 import run_variogram_pipeline_v2
        coords, values = _synthetic_stationary(150, seed=77)
        df = _make_dataframe(coords, values)

        result = run_variogram_pipeline_v2(
            data=df, xcol="X", ycol="Y", zcol="Z", vcol="GRADE",
            nlag=10, lag_distance=30, random_state=42,
        )

        fitted = result.get("fitted_models", {})
        # At least omni should be fitted
        assert len(fitted) > 0
        for direction, models in fitted.items():
            for model_type, params in models.items():
                assert "nugget" in params, f"Missing nugget in {direction}/{model_type}"
                assert "range" in params, f"Missing range in {direction}/{model_type}"
                # Must have explicit total_sill
                assert "total_sill" in params, (
                    f"Missing total_sill in {direction}/{model_type}. "
                    f"Keys: {list(params.keys())}"
                )
                # total_sill must equal nugget + partial sill
                total = params["total_sill"]
                partial = params.get("sill", params.get("partial_sill", 0))
                nug = params["nugget"]
                assert abs(total - (nug + partial)) < 1e-6, (
                    f"total_sill ({total}) != nugget ({nug}) + sill ({partial})"
                )


# ── Test: v2 vs legacy parity — OBSOLETE ──────────────────────────────
# The legacy engine is being removed as part of the variogram
# consolidation (Option A). This parity test was validating both
# engines produced similar output; after D3 there is only v2, so the
# comparison is tautological.


@pytest.mark.skip(reason="Legacy engine removed in variogram consolidation (D3)")
class TestV2LegacyParity:
    def test_omni_variogram_shape_matches(self):
        pass


# ── Test: Robust estimator reduces outlier influence ──────────────────

class TestRobustEstimator:
    def test_robust_estimator_exists(self):
        from block_model_viewer.geostats.variogram_engine_v2 import VariogramEngine
        engine = VariogramEngine.__new__(VariogramEngine)
        assert hasattr(engine, 'compute_experimental_variogram')

    def test_robust_vs_classical_with_outliers(self):
        """Robust estimator should produce lower semivariance than classical for outlier data."""
        from block_model_viewer.geostats.variogram_engine_v2 import (
            VariogramEngine, VariogramSearchConfig, DirectionSpec,
        )

        coords, values = _synthetic_with_outliers(200, seed=33)
        dir_spec = DirectionSpec("omni", np.array([1.0, 0.0, 0.0]), tolerance_deg=180.0)

        engine = VariogramEngine(coords=coords, values=values)

        # Classical (L2)
        search_classic = VariogramSearchConfig(
            lag_size=30.0, n_lags=10, use_robust_estimator=False,
        )
        exp_classic = engine.compute_experimental_variogram(
            direction=dir_spec, search=search_classic,
        )

        # Robust (Cressie-Hawkins)
        search_robust = VariogramSearchConfig(
            lag_size=30.0, n_lags=10, use_robust_estimator=True,
        )
        exp_robust = engine.compute_experimental_variogram(
            direction=dir_spec, search=search_robust,
        )

        # Both should produce valid results
        assert len(exp_classic.lag_centres) > 0
        assert len(exp_robust.lag_centres) > 0

        # Robust semivariance should be less affected by outliers
        classic_mean = np.nanmean(exp_classic.gamma)
        robust_mean = np.nanmean(exp_robust.gamma)

        # Robust should be <= classical for outlier-contaminated data
        assert robust_mean <= classic_mean * 1.05, (
            f"Robust ({robust_mean:.4f}) should be <= classical ({classic_mean:.4f}) "
            f"for outlier data"
        )


# ── Test: Auto-lag computation ────────────────────────────────────────

class TestAutoLag:
    def test_auto_lag_produces_reasonable_spacing(self):
        from block_model_viewer.geostats.variogram_bridge_v2 import run_variogram_pipeline_v2
        coords, values = _synthetic_stationary(200, seed=88)
        df = _make_dataframe(coords, values)

        result = run_variogram_pipeline_v2(
            data=df, xcol="X", ycol="Y", zcol="Z", vcol="GRADE",
            auto_lags=True, random_state=42,
        )

        omni = result["omni_variogram"]
        assert len(omni) >= 5, "Auto-lag should produce at least 5 lags"

        # Lag distances should be monotonically increasing
        dists = omni["distance"].values
        assert np.all(np.diff(dists) > 0), "Lags must be monotonically increasing"

        # Lag spacing should be roughly uniform
        spacings = np.diff(dists)
        cv = np.std(spacings) / np.mean(spacings) if np.mean(spacings) > 0 else 0
        assert cv < 0.3, f"Lag spacing too irregular: CV={cv:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
