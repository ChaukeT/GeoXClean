"""
End-to-end integration test for the FastRBF estimation pipeline.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md §10.4
"""

import hashlib

import numpy as np
import pytest

from geostats.estimation.config import RBFConfig, KernelType, DriftType
from geostats.estimation.fastrbf_engine import FastRBFEngine
from geostats.estimation.block_estimator import BlockModelEstimator
from geostats.estimation.cross_validation import loo_cross_validation
from geostats.estimation.diagnostics import EstimationDiagnostics
from geostats.estimation.audit import JORCAuditRecord
from geostats.variography.experimental import compute_experimental_variogram
from geostats.variography.models import fit_variogram_model
from geostats.utils.transforms import normal_score_transform, normal_score_backtransform, top_cut
from geostats.utils.declustering import cell_declustering


def _generate_synthetic_deposit(seed=42):
    """
    Generate a synthetic deposit with a known grade function.

    f(x, y, z) = 2.0 + 0.02*x + 0.01*y + noise

    Simulates drillhole composites sampling a simple linear grade model.
    """
    rng = np.random.Generator(np.random.PCG64(seed))

    # "Drillhole" composites — clustered in some areas
    n_samples = 80
    # Cluster 1
    pts1 = rng.normal([30, 30, 0], [15, 15, 10], (40, 3))
    # Cluster 2
    pts2 = rng.normal([70, 70, 0], [15, 15, 10], (40, 3))
    points = np.vstack([pts1, pts2])

    # Grade function + noise
    true_grade = 2.0 + 0.02 * points[:, 0] + 0.01 * points[:, 1]
    noise = rng.normal(0, 0.3, n_samples)
    values = true_grade + noise

    # Block model grid
    nx, ny, nz = 10, 10, 5
    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 100, ny)
    z = np.linspace(-25, 25, nz)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    centroids = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    block_size = np.array([10.0, 10.0, 10.0])

    return points, values, centroids, block_size


class TestFullPipeline:
    """
    End-to-end:
    1. Generate synthetic deposit
    2. Top-cut
    3. Decluster
    4. Compute experimental variogram
    5. Fit variogram model
    6. Configure RBF estimation
    7. Estimate block model
    8. Cross-validate
    9. Check diagnostics
    10. Generate JORC audit trail
    11. Verify completeness
    12. Check determinism
    """

    def test_full_estimation_pipeline(self):
        points, values, centroids, block_size = _generate_synthetic_deposit()

        # 1. Top-cut at 99th percentile
        cut_val = float(np.percentile(values, 99))
        values_capped, n_capped = top_cut(values, cut_val)

        # 2. Declustering
        cell_sizes = np.linspace(10, 60, 6)
        declust = cell_declustering(points, values_capped, cell_sizes)
        assert declust.optimal_weights.shape == values.shape

        # 3. Experimental variogram
        exp_vario = compute_experimental_variogram(points, values_capped, n_lags=10)
        assert exp_vario.lags.shape == (10,)
        assert np.any(exp_vario.pair_counts > 0)

        # 4. Fit variogram model
        vario_model = fit_variogram_model(
            exp_vario,
            model_type=KernelType.SPHEROIDAL,
            alpha=5,
        )
        assert vario_model.sill > 0
        assert vario_model.range_ > 0

        # 5. Configure RBF
        config = RBFConfig(
            kernel_type=KernelType.SPHEROIDAL,
            total_sill=vario_model.sill,
            nugget=vario_model.nugget,
            base_range=vario_model.range_,
            alpha=5,
            drift=DriftType.CONSTANT,
            discretisation_points=2,  # faster for test
            search_max_samples=24,
            search_min_samples=4,
            search_min_octants=1,
        )

        # 6. Block model estimation
        estimator = BlockModelEstimator(config)
        result = estimator.estimate(centroids, block_size, points, values_capped)

        assert result.n_blocks_estimated > 0
        assert result.estimated_values.shape == (len(centroids),)

        # Check classifications exist
        classes = set(result.classification)
        assert "Unclassified" in classes or len(classes) > 0

        # 7. Cross-validation
        cv = loo_cross_validation(points, values_capped, config)
        assert cv.n_samples > 0
        assert cv.rmse > 0

        # 8. Diagnostics
        diag = EstimationDiagnostics()

        slope = diag.slope_of_regression(cv.actual, cv.estimated)
        assert 0.0 < slope.slope < 3.0  # reasonable range

        composite_mean = float(np.mean(values_capped))
        bias = diag.global_bias_check(result.estimated_values, composite_mean)
        assert isinstance(bias.bias_percent, float)

        # Grade-tonnage
        tonnages = np.ones(len(centroids)) * 1000.0  # dummy tonnages
        gt = diag.grade_tonnage_curve(result.estimated_values, tonnages)
        assert len(gt.cutoffs) > 0

        # QQ
        qq = diag.qq_plot_data(result.estimated_values, values_capped)
        assert len(qq.estimate_quantiles) == len(qq.composite_quantiles)

        # Swath
        swath = diag.swath_plot_data(
            centroids, result.estimated_values,
            composite_points=points, composite_values=values_capped,
            axis="x",
        )
        assert len(swath.bin_centers) > 0

        # 9. JORC audit trail
        audit = JORCAuditRecord.from_estimation(
            config=config,
            fitted=FastRBFEngine(config).fit(points, values_capped),
            result=result,
            cv_result=cv,
            bias_result=bias,
            slope_result=slope,
            operator="Integration Test",
            database_description="Synthetic deposit",
            composite_length=2.0,
            block_size=(10.0, 10.0, 10.0),
        )

        warnings = audit.validate_completeness()
        # Allow operator and database_description warnings to pass
        critical = [w for w in warnings if "Missing:" in w and "operator" not in w.lower() and "database" not in w.lower()]
        assert len(critical) == 0, f"Critical audit warnings: {critical}"

        text = audit.to_jorc_table1_section3()
        assert "JORC TABLE 1" in text

        # 10. Determinism — run estimation twice
        result2 = estimator.estimate(centroids, block_size, points, values_capped)
        np.testing.assert_array_equal(
            result.estimated_values, result2.estimated_values
        )


class TestNormalScoreRoundTrip:
    """Normal score transform → back-transform should recover original values."""

    def test_roundtrip(self):
        rng = np.random.Generator(np.random.PCG64(42))
        values = rng.lognormal(0, 1, 100)

        ns, table = normal_score_transform(values)
        recovered = normal_score_backtransform(ns, table)

        np.testing.assert_allclose(recovered, values, atol=0.1)
