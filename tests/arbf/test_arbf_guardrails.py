from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

import geostats.arbf.engine as arbf_engine
from geostats.arbf.cross_validation import CVResult
from geostats.arbf.engine import ARBFEstimator, ARBFResult
from geostats.arbf.classification import (
    GeometricCriteria,
    VarianceThresholds,
    classify_blocks,
)
from geostats.arbf.variogram import LocalVariogramResult


def make_grid(shape=(9, 9, 5), domain=(40.0, 40.0, 20.0)):
    axes = [np.linspace(0.0, float(domain[i]), int(shape[i])) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    spacing = np.array(
        [
            float(domain[0]) / max(shape[0] - 1, 1),
            float(domain[1]) / max(shape[1] - 1, 1),
            float(domain[2]) / max(shape[2] - 1, 1),
        ],
        dtype=np.float64,
    )
    return coords.astype(np.float64), spacing


def base_config(**overrides):
    cfg = {
        "kernel_type": "spheroidal",
        "alpha": 1.5,
        "sill": 0.5,
        "nugget": 0.0,
        "range_max": 15.0,
        "range_mid": 15.0,
        "range_min": 10.0,
        "drift_type": "auto",
        "n_subdomains": 0,
        "max_samples": 20,
        "min_samples": 4,
        "change_of_support": False,
        "run_cv": False,
        "parallel": False,
        "verbose": False,
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


def rmse(actual, estimated):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(estimated)) ** 2)))


def test_domain_policy_require_rejects_missing_domains():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    values = np.array([1.0, 1.1, 0.9, 1.05], dtype=np.float64)
    centroids = np.array([[2.5, 0.0, 0.0], [12.5, 0.0, 0.0]], dtype=np.float64)

    estimator = ARBFEstimator(base_config(domain_policy="require"))
    estimator.set_composites(coords, values)
    estimator.set_block_model(centroids, np.array([5.0, 5.0, 5.0], dtype=np.float64))

    with pytest.raises(ValueError, match="domain_policy='require'"):
        estimator.estimate()


def test_auto_drift_selects_linear_for_planar_trend():
    rng = np.random.default_rng(7)
    coords = rng.uniform([0.0, 0.0, 0.0], [40.0, 40.0, 20.0], size=(48, 3))
    values = (
        1.0
        + 0.08 * coords[:, 0]
        - 0.06 * coords[:, 1]
        + 0.03 * coords[:, 2]
        + rng.normal(0.0, 0.01, size=len(coords))
    )
    centroids = np.array([[10.0, 10.0, 10.0], [30.0, 30.0, 10.0]], dtype=np.float64)

    estimator = ARBFEstimator(
        base_config(
            sill=1.0,
            nugget=1e-4,
            range_max=30.0,
            range_mid=30.0,
            range_min=20.0,
            auto_drift_cv_max_samples=40,
        ),
    )
    estimator.set_composites(coords, values)
    estimator.set_block_model(centroids, np.array([5.0, 5.0, 5.0], dtype=np.float64))
    result = estimator.estimate()

    assert result.diagnostics["effective_drift_type"] == "linear"
    assert (
        result.diagnostics["trend_linear_cv_rmse"]
        < result.diagnostics["trend_constant_cv_rmse"]
    )


def test_hard_domains_reduce_contact_smearing():
    grid_coords, spacing = make_grid()
    block_sizes = spacing
    block_domains = np.where(grid_coords[:, 0] < 20.0, 1, 2)
    truth = np.where(block_domains == 1, 1.0, 3.0).astype(np.float64)

    rng = np.random.default_rng(21)
    obs_idx = rng.choice(len(grid_coords), size=90, replace=False)
    obs_coords = grid_coords[obs_idx]
    obs_values = truth[obs_idx]
    obs_domains = block_domains[obs_idx]
    contact_mask = np.abs(grid_coords[:, 0] - 20.0) <= spacing[0] * 1.5

    domain_cfg = base_config(
        drift_type="constant",
        sill=1.0,
        nugget=1e-3,
        range_max=12.0,
        range_mid=12.0,
        range_min=8.0,
        max_samples=24,
    )

    no_domain = ARBFEstimator(domain_cfg)
    no_domain.set_composites(obs_coords, obs_values)
    no_domain.set_block_model(grid_coords, block_sizes)
    no_domain_result = no_domain.estimate()

    hard_domain = ARBFEstimator(
        {**domain_cfg, "domain_policy": "require"},
    )
    hard_domain.set_composites(obs_coords, obs_values)
    hard_domain.set_block_model(grid_coords, block_sizes)
    hard_domain.set_domains(obs_domains, block_domains)
    hard_domain_result = hard_domain.estimate()

    no_domain_contact_rmse = rmse(truth[contact_mask], no_domain_result.grades[contact_mask])
    hard_domain_contact_rmse = rmse(
        truth[contact_mask], hard_domain_result.grades[contact_mask],
    )

    assert hard_domain_contact_rmse < no_domain_contact_rmse
    assert hard_domain_result.diagnostics["domains_enforced"] is True


def test_balanced_local_neighbourhood_preserves_octant_coverage():
    coords = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.12, 0.09, 0.11],
            [0.11, 0.13, 0.09],
            [0.09, 0.11, 0.12],
            [-0.60, -0.60, -0.60],
            [-0.60, -0.60, 0.60],
            [-0.60, 0.60, -0.60],
            [0.60, -0.60, -0.60],
        ],
        dtype=np.float64,
    )
    estimator = ARBFEstimator(
        base_config(
            drift_type="constant",
            range_max=1.0,
            range_mid=1.0,
            range_min=1.0,
            max_samples=4,
            min_samples=4,
            local_search_radii=(1.2, 2.0, 3.0),
            search_min_octants=4,
            balanced_neighbourhood_selection=True,
        ),
    )
    tree = cKDTree(coords)
    idx, pass_idx = estimator._select_local_neighbourhood(
        tree,
        coords,
        np.zeros(3, dtype=np.float64),
    )
    diffs = coords[idx]
    octant_count = int(np.unique(estimator._octant_codes(diffs)).size)

    assert pass_idx == 1
    assert len(idx) == 4
    assert octant_count >= 4


def test_local_neighbourhood_prefers_richer_multioctant_pass():
    coords = np.array(
        [
            [0.40, 0.40, 0.40],
            [-0.40, 0.40, 0.40],
            [0.40, -0.40, 0.40],
            [0.40, 0.40, -0.40],
            [-1.20, -1.20, -1.20],
            [1.20, -1.20, -1.20],
            [-1.20, 1.20, -1.20],
            [-1.20, -1.20, 1.20],
        ],
        dtype=np.float64,
    )
    estimator = ARBFEstimator(
        base_config(
            drift_type="constant",
            range_max=1.0,
            range_mid=1.0,
            range_min=1.0,
            max_samples=32,
            min_samples=4,
            local_search_radii=(0.5, 2.5, 3.5),
            search_min_octants=3,
            balanced_neighbourhood_selection=True,
        ),
    )
    tree = cKDTree(coords)
    idx, pass_idx = estimator._select_local_neighbourhood(
        tree,
        coords,
        np.zeros(3, dtype=np.float64),
    )
    octant_count = int(np.unique(estimator._octant_codes(coords[idx])).size)

    assert pass_idx == 2
    assert len(idx) >= 8
    assert octant_count >= 6


def test_local_neighbourhood_regresses_far_blocks_instead_of_dropping_them():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 5.0, 0.0],
            [5.0, 0.0, 5.0],
        ],
        dtype=np.float64,
    )
    values = np.array([2.4, 2.2, 1.8, 1.9, 1.7, 2.0], dtype=np.float64)
    centroids = np.array(
        [
            [1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0],
            [2.5, 1.5, 2.0],
            [1.5, 2.5, 2.0],
            [3.0, 2.0, 1.5],
            [2.0, 3.0, 1.5],
            [1.0, 2.0, 2.5],
            [2.5, 2.5, 1.0],
            [3.0, 1.0, 2.5],
            [1.0, 3.0, 2.5],
            [60.0, 60.0, 30.0],
        ],
        dtype=np.float64,
    )

    estimator = ARBFEstimator(
        base_config(
            drift_type="constant",
            estimation_mode="local_neighbourhood_gpr",
            max_samples=6,
            min_samples=4,
            local_search_radii=(0.20, 0.35, 0.50),
            range_max=15.0,
            range_mid=15.0,
            range_min=10.0,
            discretisation_density=8,
        ),
    )
    estimator.set_composites(coords, values)
    estimator.set_block_model(
        centroids,
        np.array([5.0, 5.0, 5.0], dtype=np.float64),
    )

    result = estimator.estimate()

    assert np.isfinite(result.grades[:-1]).all()
    assert np.isfinite(result.variances).all()
    assert np.isnan(result.grades[-1])


def test_local_neighbourhood_search_can_vary_within_same_tile(monkeypatch):
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.4],
            [6.0, 0.0, 0.0],
            [6.4, 0.0, 0.0],
            [6.0, 0.4, 0.0],
            [6.0, 0.0, 0.4],
        ],
        dtype=np.float64,
    )
    values = np.array([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    centroids = np.array(
        [
            [0.2, 0.1, 0.1],
            [4.2, 0.1, 0.1],
        ],
        dtype=np.float64,
    )

    estimator = ARBFEstimator(
        base_config(
            drift_type="constant",
            estimation_mode="local_neighbourhood_gpr",
            range_max=10.0,
            range_mid=10.0,
            range_min=10.0,
            max_samples=16,
            min_samples=4,
            local_search_radii=(1.0, 3.0, 5.0),
            search_min_octants=1,
            discretisation_density=8,
        ),
    )
    estimator.set_composites(coords, values)
    estimator.set_block_model(
        centroids,
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )

    seen_query_points = []
    original_select = estimator._select_local_neighbourhood

    def spy_select_local_neighbourhood(
        tree,
        transformed_coords,
        query_point,
        precomputed_candidates=None,
    ):
        seen_query_points.append(np.array(query_point, copy=True))
        return original_select(
            tree,
            transformed_coords,
            query_point,
            precomputed_candidates=precomputed_candidates,
        )

    monkeypatch.setattr(
        estimator,
        "_select_local_neighbourhood",
        spy_select_local_neighbourhood,
    )

    result = estimator.estimate()

    assert np.isfinite(result.grades).all()
    assert float(result.grades[0]) > float(result.grades[1])
    assert len(seen_query_points) >= 1


def test_geometric_stats_respect_configured_search_radii():
    coords = np.array([[0.60, 0.00, 0.00]], dtype=np.float64)
    estimator = ARBFEstimator(
        base_config(
            drift_type="constant",
            range_max=1.0,
            range_mid=1.0,
            range_min=1.0,
            max_samples=1,
            min_samples=1,
            local_search_radii=(0.75, 1.50, 3.0),
        ),
    )
    estimator.set_block_model(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )

    sample_counts, octant_counts, search_passes = estimator._compute_geometric_stats(
        coords,
        np.eye(3, dtype=np.float64),
    )

    assert int(sample_counts[0]) == 1
    assert int(octant_counts[0]) == 1
    assert int(search_passes[0]) == 1


def test_octant_setting_seeds_balance_without_hard_capping_local_neighbourhood():
    coords = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                for scale in (0.2, 0.4, 0.6):
                    coords.append([sx * scale, sy * scale, sz * scale])
    coords = np.asarray(coords, dtype=np.float64)

    estimator = ARBFEstimator(
        base_config(
            drift_type="constant",
            range_max=1.0,
            range_mid=1.0,
            range_min=1.0,
            max_samples=300,
            min_samples=4,
            max_samples_per_octant=2,
            balanced_neighbourhood_selection=True,
        ),
    )

    trimmed = estimator._trim_local_candidates(
        np.arange(len(coords), dtype=np.intp),
        coords,
        np.zeros(3, dtype=np.float64),
    )
    octant_counts = np.bincount(
        estimator._octant_codes(coords[trimmed]),
        minlength=8,
    )
    seed_octant_counts = np.bincount(
        estimator._octant_codes(coords[trimmed[:16]]),
        minlength=8,
    )

    assert len(trimmed) == len(coords)
    assert int(np.max(seed_octant_counts)) <= 2
    assert int(np.max(octant_counts)) == 3


def test_legacy_four_per_octant_default_no_longer_caps_local_neighbourhood():
    coords = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                for scale in (0.2, 0.4, 0.6, 0.8, 1.0):
                    coords.append([sx * scale, sy * scale, sz * scale])
    coords = np.asarray(coords, dtype=np.float64)

    estimator = ARBFEstimator(
        base_config(
            drift_type="constant",
            range_max=1.0,
            range_mid=1.0,
            range_min=1.0,
            max_samples=40,
            min_samples=4,
            balanced_neighbourhood_selection=True,
        ),
    )

    trimmed = estimator._trim_local_candidates(
        np.arange(len(coords), dtype=np.intp),
        coords,
        np.zeros(3, dtype=np.float64),
    )

    assert len(trimmed) == 40


def test_factorise_kernel_system_reuses_base_kernel_across_retries(monkeypatch):
    estimator = ARBFEstimator(base_config(drift_type="constant", accuracy=1e-7))
    local_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    local_values = np.array([1.0, 0.9, 1.1, 1.05], dtype=np.float64)
    vp = LocalVariogramResult(
        sill=1.0,
        nugget=0.05,
        range_=10.0,
        alpha=1.0,
        kernel_type="spheroidal",
        fit_residual=0.0,
        n_pairs=0,
        n_lags=0,
    )
    assemble_calls = {"count": 0}
    solve_calls = {"count": 0}

    def fake_assemble(*args, **kwargs):
        assemble_calls["count"] += 1
        n = local_coords.shape[0]
        return np.eye(n + 1, dtype=np.float64), np.ones((n, 1), dtype=np.float64)

    def fake_factorise(k_aug, values, drift_type="constant"):
        solve_calls["count"] += 1
        n = len(values)
        if solve_calls["count"] < 3:
            weights = np.full(n, 150.0, dtype=np.float64)
        else:
            weights = np.full(n, 10.0, dtype=np.float64)
        return np.eye(n + 1, dtype=np.float64), weights, np.zeros(1, dtype=np.float64)

    monkeypatch.setattr(arbf_engine, "assemble_kernel_matrix", fake_assemble)
    monkeypatch.setattr(arbf_engine, "factorise_and_solve", fake_factorise)

    estimator._factorise_kernel_system(
        local_coords,
        local_values,
        vp,
        np.eye(3, dtype=np.float64),
        np.eye(3, dtype=np.float64),
        context="test neighbourhood",
        drift_type="constant",
    )

    assert assemble_calls["count"] == 1
    assert solve_calls["count"] == 3


def test_classification_handles_zero_variance_percentile_collapse():
    posterior = np.zeros(6, dtype=np.float64)
    sample_counts = np.full(6, 12, dtype=np.int32)
    octant_counts = np.full(6, 4, dtype=np.int32)
    search_passes = np.ones(6, dtype=np.int32)

    result = classify_blocks(
        posterior,
        sample_counts,
        octant_counts,
        search_passes,
        variance_thresholds=VarianceThresholds.from_percentiles(posterior),
        geometric_criteria=GeometricCriteria(),
    )

    assert np.all(result.classes == 3)


def test_spatial_kfold_cv_uses_training_fold_values_only(monkeypatch):
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    values = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    estimator = ARBFEstimator(
        base_config(
            drift_type="constant",
            run_cv=True,
            cv_mode="spatial_kfold",
            cv_folds=2,
            cv_max_samples=4,
            min_samples=2,
        ),
    )
    estimator.set_composites(coords, values)
    estimator.set_block_model(coords.copy(), np.zeros(3, dtype=np.float64))

    monkeypatch.setattr(
        estimator,
        "_auto_cv_folds",
        lambda coords_arg, user_folds: 2,
    )
    monkeypatch.setattr(
        estimator,
        "_assign_spatial_folds",
        lambda coords_arg, n_folds: np.array([0, 0, 1, 1], dtype=np.int32),
    )

    def fake_estimate(self):
        n_blocks = len(self._block_centroids)
        train_mean = float(np.mean(self._composite_values))
        return ARBFResult(
            grades=np.full(n_blocks, train_mean, dtype=np.float64),
            variances=np.zeros(n_blocks, dtype=np.float64),
            classifications=np.zeros(n_blocks, dtype=np.int32),
            classification_names=np.full(n_blocks, "Unclassified", dtype=object),
            diagnostics={},
        )

    monkeypatch.setattr(ARBFEstimator, "estimate", fake_estimate)

    cv_result = estimator._run_spatial_kfold_cv()

    assert cv_result is not None
    np.testing.assert_allclose(
        cv_result.estimated,
        np.array([10.0, 10.0, 0.0, 0.0], dtype=np.float64),
    )


def test_spatial_kfold_negative_r_squared_is_reported_not_downgraded(monkeypatch):
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    estimator = ARBFEstimator(
        base_config(
            run_cv=True,
            cv_mode="spatial_kfold",
            allow_cv_fallback=True,
        ),
    )
    estimator.set_composites(coords, values)
    estimator.set_block_model(coords.copy(), np.ones(3, dtype=np.float64))

    bad_cv = CVResult(
        actual=np.array([1.0, 2.0], dtype=np.float64),
        estimated=np.array([3.0, 3.0], dtype=np.float64),
        errors=np.array([-2.0, -1.0], dtype=np.float64),
        mean_error=-1.5,
        mae=1.5,
        rmse=np.sqrt(2.5),
        r_squared=-3.0,
        correlation=0.0,
        normalised_rmse=1.0,
        slope_of_regression=0.0,
        intercept=0.0,
        n_samples=2,
    )
    monkeypatch.setattr(estimator, "_run_spatial_kfold_cv", lambda: bad_cv)

    result = estimator._run_cross_validation(coords, values)

    assert result is bad_cv
    assert estimator._cv_execution_mode == "spatial_kfold"


def test_spatial_kfold_failure_stays_failed_without_opt_in_fallback(monkeypatch):
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    estimator = ARBFEstimator(
        base_config(
            run_cv=True,
            cv_mode="spatial_kfold",
            allow_cv_fallback=False,
        ),
    )
    estimator.set_composites(coords, values)
    estimator.set_block_model(coords.copy(), np.ones(3, dtype=np.float64))

    monkeypatch.setattr(estimator, "_run_spatial_kfold_cv", lambda: None)

    result = estimator._run_cross_validation(coords, values)

    assert result is None
    assert estimator._cv_execution_mode == "spatial_kfold_failed"


def test_unestimated_domain_blocks_remain_nan_not_zero():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    values = np.array([1.0, 1.1, 0.9, 1.05], dtype=np.float64)
    composite_domains = np.array([1, 1, 1, 1], dtype=np.int32)
    block_centroids = np.array([[2.5, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=np.float64)
    block_domains = np.array([1, 2], dtype=np.int32)

    estimator = ARBFEstimator(base_config(domain_policy="require", drift_type="constant"))
    estimator.set_composites(coords, values)
    estimator.set_block_model(block_centroids, np.array([5.0, 5.0, 5.0], dtype=np.float64))
    estimator.set_domains(composite_domains, block_domains)
    result = estimator.estimate()

    assert np.isfinite(result.grades[0])
    assert np.isnan(result.grades[1])
    assert result.classification_names[1] == "Unclassified"


def test_user_classification_thresholds_are_not_autoscaled(monkeypatch):
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 5.0, 0.0],
            [5.0, 0.0, 5.0],
        ],
        dtype=np.float64,
    )
    values = np.array([2.4, 2.2, 1.8, 1.9, 1.7, 2.0], dtype=np.float64)
    centroids = np.array([[2.0, 2.0, 2.0]], dtype=np.float64)
    captured = {}

    original_classify = arbf_engine.classify_blocks

    def spy_classify_blocks(*args, **kwargs):
        vt = kwargs.get("variance_thresholds")
        captured["thresholds"] = (
            vt.t1_measured,
            vt.t2_indicated,
            vt.t3_inferred,
        )
        return original_classify(*args, **kwargs)

    monkeypatch.setattr(arbf_engine, "classify_blocks", spy_classify_blocks)

    estimator = ARBFEstimator(
        base_config(
            drift_type="constant",
            estimation_mode="local_neighbourhood_gpr",
            classification_thresholds={
                "measured": 0.01,
                "indicated": 0.02,
                "inferred": 0.03,
            },
        ),
    )
    estimator.set_composites(coords, values)
    estimator.set_block_model(centroids, np.array([5.0, 5.0, 5.0], dtype=np.float64))
    result = estimator.estimate()

    assert captured["thresholds"] == (0.01, 0.02, 0.03)
    assert result.diagnostics["classification_threshold_source"] == "user"


def test_auto_sill_is_used_in_global_variogram_diagnostics():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
            [10.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    values = np.array([1.0, 2.0, 1.5, 2.5, 3.0, 2.2], dtype=np.float64)
    expected_sill = float(np.var(values))

    estimator = ARBFEstimator(
        base_config(
            drift_type="constant",
            sill=0.0,
            nugget=0.0,
            estimation_mode="local_neighbourhood_gpr",
        ),
    )
    estimator.set_composites(coords, values)
    estimator.set_block_model(
        np.array([[5.0, 5.0, 5.0]], dtype=np.float64),
        np.array([5.0, 5.0, 5.0], dtype=np.float64),
    )
    result = estimator.estimate()

    assert np.isclose(result.audit_record.sill, expected_sill, rtol=1e-6)
    np.testing.assert_allclose(
        result.diagnostics["classification_thresholds"],
        np.array([0.1, 0.3, 0.6], dtype=np.float64) * expected_sill,
        rtol=1e-6,
    )
