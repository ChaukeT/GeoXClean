from __future__ import annotations

import numpy as np

from geostats.arbf import ARBFEstimator, ARBFSequentialSimulation
from geostats.arbf.cross_validation import support_swath_plots


def make_regular_case(seed: int = 11):
    rng = np.random.default_rng(seed)
    nx, ny, nz = 6, 6, 3
    dx, dy, dz = 10.0, 10.0, 10.0
    centroids = np.array(
        [
            [ix * dx + dx / 2, iy * dy + dy / 2, iz * dz + dz / 2]
            for ix in range(nx)
            for iy in range(ny)
            for iz in range(nz)
        ],
        dtype=np.float64,
    )
    block_sizes = np.array([dx, dy, dz], dtype=np.float64)
    truth = (
        1.5
        + 0.4 * np.sin(centroids[:, 0] / 15.0)
        + 0.3 * np.cos(centroids[:, 1] / 18.0)
        + 0.15 * centroids[:, 2] / 10.0
    )
    obs_idx = rng.choice(len(centroids), size=28, replace=False)
    obs_coords = centroids[obs_idx].copy()
    obs_values = truth[obs_idx].copy()
    return obs_coords, obs_values, centroids, block_sizes, truth, obs_idx


def make_irregular_case(seed: int = 23):
    rng = np.random.default_rng(seed)
    nx, ny, nz = 5, 5, 2
    base = np.array(
        [
            [ix * 12.0 + 6.0, iy * 11.0 + 5.5, iz * 9.0 + 4.5]
            for ix in range(nx)
            for iy in range(ny)
            for iz in range(nz)
        ],
        dtype=np.float64,
    )
    jitter = rng.uniform([-1.2, -1.0, -0.8], [1.2, 1.0, 0.8], size=base.shape)
    centroids = base + jitter
    block_sizes = rng.uniform([8.0, 7.5, 6.0], [12.0, 11.5, 9.0], size=centroids.shape)
    truth = (
        2.0
        + 0.35 * np.sin(centroids[:, 0] / 14.0)
        + 0.25 * np.cos(centroids[:, 1] / 16.0)
        + 0.10 * centroids[:, 2] / 8.0
    )
    obs_idx = rng.choice(len(centroids), size=22, replace=False)
    obs_coords = centroids[obs_idx].copy()
    obs_values = truth[obs_idx].copy()
    return obs_coords, obs_values, centroids, block_sizes


def base_config(**overrides):
    cfg = {
        "kernel_type": "spheroidal",
        "alpha": 1.5,
        "sill": 0.6,
        "nugget": 1e-4,
        "range_max": 28.0,
        "range_mid": 28.0,
        "range_min": 18.0,
        "drift_type": "constant",
        "n_subdomains": 0,
        "max_samples": 18,
        "min_samples": 4,
        "parallel": False,
        "verbose": False,
        "change_of_support": False,
        "seed": 17,
    }
    cfg.update(overrides)
    return cfg


def test_estimator_returns_conditional_bias_and_support_swath():
    obs_coords, obs_values, centroids, block_sizes, _, _ = make_regular_case()

    estimator = ARBFEstimator(base_config(run_cv=True, cv_max_samples=28, cv_folds=4))
    estimator.set_composites(obs_coords, obs_values)
    estimator.set_block_model(centroids, block_sizes)
    result = estimator.estimate()

    assert result.cv_result is not None
    assert result.conditional_bias_result is not None
    assert result.support_swath_data is not None
    assert result.support_swath_data.n_panels_total > 0
    assert "x" in result.support_swath_data.axes
    assert np.isfinite(result.conditional_bias_result.global_slope)
    assert np.isfinite(result.conditional_bias_result.max_abs_bin_bias)
    assert result.audit_record.support_swath_panels_total > 0
    assert np.isfinite(result.audit_record.conditional_bias_binned_slope)
    assert np.isfinite(result.diagnostics["support_swath_mean_rmse"])


def test_spatial_kfold_cv_with_normal_score_keeps_valid_heldout_pairs():
    obs_coords, obs_values, centroids, block_sizes, _, _ = make_regular_case()
    obs_values = np.exp(obs_values)

    estimator = ARBFEstimator(
        base_config(
            run_cv=True,
            cv_mode="spatial_kfold",
            cv_max_samples=28,
            cv_folds=4,
            use_normal_score=True,
            max_samples=18,
        ),
    )
    estimator.set_composites(obs_coords, obs_values)
    estimator.set_block_model(centroids, block_sizes)
    result = estimator.estimate()

    assert result.cv_result is not None
    assert result.cv_result.n_samples >= 14
    assert result.conditional_bias_result is not None
    assert result.diagnostics["cv_execution_mode"] == "spatial_kfold"
    assert np.isfinite(result.cv_result.rmse)


def test_sequential_simulation_is_reproducible_and_honours_exact_data():
    obs_coords, obs_values, centroids, block_sizes, _, obs_idx = make_regular_case()
    cfg = base_config(
        n_realizations=6,
        simulation_seed=123,
        simulation_use_normal_score=True,
    )

    sim_a = ARBFSequentialSimulation(cfg)
    sim_a.set_composites(obs_coords, obs_values)
    sim_a.set_block_model(centroids, block_sizes)
    result_a = sim_a.simulate()

    sim_b = ARBFSequentialSimulation(cfg)
    sim_b.set_composites(obs_coords, obs_values)
    sim_b.set_block_model(centroids, block_sizes)
    result_b = sim_b.simulate()

    assert result_a.realizations.shape == (6, len(centroids))
    assert np.all(np.isfinite(result_a.realizations))
    assert np.allclose(result_a.realizations, result_b.realizations)
    assert float(np.mean(result_a.variance)) > 0.0
    assert not np.allclose(result_a.realizations[0], result_a.realizations[1])

    hard_node_var = np.var(result_a.realizations[:, obs_idx], axis=0)
    assert float(np.max(hard_node_var)) < 1e-10
    assert result_a.diagnostics["hard_data_honoured_nodes"] >= len(obs_idx)


def test_support_swath_handles_irregular_block_geometry():
    obs_coords, obs_values, centroids, block_sizes = make_irregular_case()

    estimator = ARBFEstimator(base_config(run_cv=False, max_samples=16))
    estimator.set_composites(obs_coords, obs_values)
    estimator.set_block_model(centroids, block_sizes)
    result = estimator.estimate()

    assert result.support_swath_data is not None
    assert result.support_swath_data.n_panels_total > 0
    assert result.support_swath_data.n_panels_with_data > 0
    assert any(
        np.any(swath.n_data_panels_per_slice > 0)
        for swath in result.support_swath_data.axes.values()
    )
    assert result.audit_record.support_swath_panels_total > 0


def test_support_swath_splits_block_volume_across_overlapping_panels():
    block_centroids = np.array(
        [
            [2.0, 2.0, 2.0],
            [8.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    block_sizes = np.array(
        [
            [4.0, 4.0, 4.0],
            [8.0, 4.0, 4.0],
        ],
        dtype=np.float64,
    )
    block_estimates = np.array([1.0, 3.0], dtype=np.float64)
    composite_coords = np.array([[1.0, 2.0, 2.0], [9.0, 2.0, 2.0]], dtype=np.float64)
    composite_values = np.array([1.0, 3.0], dtype=np.float64)

    swath = support_swath_plots(
        block_centroids,
        block_estimates,
        composite_coords,
        composite_values,
        block_sizes=block_sizes,
        panel_factors=(1, 1, 1),
    )

    assert swath is not None
    assert swath.panel_shape == (2, 1, 1)
    assert np.isclose(swath.panel_estimated[0], 5.0 / 3.0, atol=1e-8)
    assert np.isclose(swath.panel_estimated[1], 3.0, atol=1e-8)
    assert np.isclose(swath.axes["x"].mean_estimated[0], 5.0 / 3.0, atol=1e-8)
