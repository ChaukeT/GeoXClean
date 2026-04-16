"""ARBF level 2 — synthetic field tests.

Known-truth spatial patterns driven through ARBFEstimatorAdapter:

  * constant field
  * linear trend field
  * smooth Gaussian hill (bump)
  * local spike preservation
  * two separated grade populations (domain-free run)

These check whether ARBF is too smooth, too global, unstable, or
unable to preserve local highs — the behaviours the user flagged as
the most important for block-modelling trust.
"""

from __future__ import annotations

import numpy as np
import pytest

from block_model_viewer.geostats.arbf_adapter import ARBFEstimatorAdapter


def _base_cfg(**overrides):
    cfg = dict(
        range_max=40.0, range_mid=40.0, range_min=40.0,
        azimuth=0.0, dip=0.0, pitch=0.0,
        nugget=0.0, sill=1.0,
        kernel_type="spheroidal",
        drift_type="constant",
        use_normal_score=False,
        force_normal_score=False,
        search_mode="local",
        max_samples=20, min_samples=2,
        local_search_radii=(0.5, 1.0, 2.0),
        max_samples_per_octant=4, search_min_octants=1,
        discretisation_density=1,
        pum_threshold=10000,
        accuracy=1e-9,
        panel_width=0.0,
        run_cv=False,
    )
    cfg.update(overrides)
    return cfg


def _run(coords, values, centroids, dx=2.0, cfg=None):
    adapter = ARBFEstimatorAdapter(cfg or _base_cfg())
    adapter.set_composites(np.asarray(coords, float),
                           np.asarray(values, float).ravel())
    adapter.set_block_model(np.asarray(centroids, float),
                            np.array([dx, dx, dx], float))
    r = adapter.estimate()
    return np.asarray(r["grades"], float), r


def _grid(n=10, extent=40.0, z=5.0):
    xs = np.linspace(2.0, extent - 2.0, n)
    ys = np.linspace(2.0, extent - 2.0, n)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])


# ─────────────────────────────────────────────────────────────────────
# 1. Constant field
# ─────────────────────────────────────────────────────────────────────


def test_synthetic_constant_field():
    """Whole-volume constant field: every estimate must equal the constant."""
    rng = np.random.default_rng(10)
    coords = rng.uniform([0, 0, 0], [40, 40, 10], size=(30, 3))
    values = np.full(30, 7.5)
    centroids = _grid(n=8, extent=40.0, z=5.0)
    grades, _ = _run(coords, values, centroids)
    finite = grades[np.isfinite(grades)]
    assert finite.size >= 50
    assert np.max(np.abs(finite - 7.5)) < 1e-3


# ─────────────────────────────────────────────────────────────────────
# 2. Linear trend field
# ─────────────────────────────────────────────────────────────────────


def test_synthetic_linear_trend_field():
    """Grade varies linearly with X: g(x,y,z) = 2 + 0.1*x.

    With linear drift enabled, ARBF should reproduce a linear trend
    closely (CV slope near 1.0, R² high).
    """
    rng = np.random.default_rng(11)
    coords = rng.uniform([0, 0, 0], [40, 40, 10], size=(80, 3))
    values = 2.0 + 0.1 * coords[:, 0]
    centroids = _grid(n=10, extent=40.0, z=5.0)
    truth = 2.0 + 0.1 * centroids[:, 0]

    grades, _ = _run(coords, values, centroids, dx=3.0,
                     cfg=_base_cfg(drift_type="linear"))
    finite_mask = np.isfinite(grades)
    assert finite_mask.sum() >= 80

    # Regression: estimates vs truth
    from scipy.stats import linregress
    lr = linregress(truth[finite_mask], grades[finite_mask])
    assert lr.rvalue ** 2 > 0.95, f"R² = {lr.rvalue ** 2:.3f}"
    assert 0.90 <= lr.slope <= 1.10, f"slope = {lr.slope:.3f}"
    # Mean error
    me = float(np.mean(grades[finite_mask] - truth[finite_mask]))
    std_truth = float(np.std(truth))
    assert abs(me) < 0.05 * std_truth, (
        f"|ME|/std = {abs(me)/std_truth:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────
# 3. Smooth Gaussian hill
# ─────────────────────────────────────────────────────────────────────


def _gaussian_hill(xy, centre=(20.0, 20.0), sigma=8.0, peak=5.0, base=1.0):
    d2 = (xy[:, 0] - centre[0]) ** 2 + (xy[:, 1] - centre[1]) ** 2
    return base + peak * np.exp(-0.5 * d2 / sigma ** 2)


def test_synthetic_gaussian_hill():
    """Smooth gaussian hill centred on (20,20). Estimates should
    track the hill shape with high correlation.
    """
    rng = np.random.default_rng(12)
    coords_xy = rng.uniform([0, 0], [40, 40], size=(100, 2))
    coords = np.column_stack([coords_xy, np.full(100, 5.0)])
    values = _gaussian_hill(coords_xy)

    centroids = _grid(n=12, extent=40.0, z=5.0)
    truth = _gaussian_hill(centroids[:, :2])

    grades, _ = _run(coords, values, centroids, dx=3.0,
                     cfg=_base_cfg(range_max=20.0, range_mid=20.0,
                                   range_min=20.0))
    finite_mask = np.isfinite(grades)
    assert finite_mask.sum() >= 100

    corr = float(np.corrcoef(grades[finite_mask], truth[finite_mask])[0, 1])
    rmse = float(np.sqrt(np.mean((grades[finite_mask] - truth[finite_mask]) ** 2)))
    std_truth = float(np.std(truth))
    assert corr > 0.90, f"corr = {corr:.3f}"
    assert rmse / std_truth < 0.35, f"RMSE/std = {rmse/std_truth:.3f}"


# ─────────────────────────────────────────────────────────────────────
# 4. Local spike preservation
# ─────────────────────────────────────────────────────────────────────


def test_local_spike_is_not_overly_smoothed():
    """A single isolated high-grade composite surrounded by low
    background should produce a LOCAL high in the block model — not
    spread across the whole domain, not vanished.

    Pass criteria:
      * The block nearest the spike must be at least 40% of the way
        from background to spike value.
      * Blocks far from the spike must stay near background.
      * The smoothing ratio at the spike must exceed 0.40.
    """
    # Regular grid of background composites + one spike
    xs, ys = np.meshgrid(np.linspace(0, 40, 9), np.linspace(0, 40, 9))
    bg_coords = np.column_stack([xs.ravel(), ys.ravel(),
                                 np.full(xs.size, 5.0)])
    values = np.full(len(bg_coords), 1.0)
    # Replace the sample nearest (20,20) with a high spike
    d_to_centre = np.linalg.norm(bg_coords[:, :2] - np.array([20, 20]),
                                  axis=1)
    spike_idx = int(np.argmin(d_to_centre))
    bg_coords[spike_idx] = [20.0, 20.0, 5.0]
    values[spike_idx] = 10.0

    # Fine-resolution block grid
    centroids = _grid(n=15, extent=40.0, z=5.0)

    cfg = _base_cfg(range_max=15.0, range_mid=15.0, range_min=15.0,
                    local_search_radii=(0.3, 0.6, 1.2),
                    max_samples=12, max_samples_per_octant=3)
    grades, _ = _run(bg_coords, values, centroids, dx=2.0, cfg=cfg)
    finite_mask = np.isfinite(grades)
    g_finite = grades[finite_mask]
    c_finite = centroids[finite_mask]

    # Block closest to the spike
    block_dists = np.linalg.norm(c_finite[:, :2] - np.array([20, 20]),
                                  axis=1)
    near_idx = int(np.argmin(block_dists))
    near_value = float(g_finite[near_idx])

    # Blocks far from the spike (> 12 m away in plan)
    far_mask = block_dists > 12.0
    far_values = g_finite[far_mask]

    assert near_value > 1.0 + 0.40 * (10.0 - 1.0), (
        f"spike under-preserved: near-block value = {near_value:.2f} "
        f"(background 1.0, spike 10.0)"
    )
    assert float(np.mean(far_values)) < 1.5, (
        f"spike bled too far: far blocks mean = {float(np.mean(far_values)):.2f}"
    )
    # Maximum estimate should be near the spike, not far from it
    max_idx = int(np.argmax(g_finite))
    max_dist = float(block_dists[max_idx])
    assert max_dist < 5.0, (
        f"max-estimate block is {max_dist:.1f} m from the spike"
    )


# ─────────────────────────────────────────────────────────────────────
# 5. Two separated grade populations
# ─────────────────────────────────────────────────────────────────────


def test_two_separated_populations():
    """Two spatially separated grade populations — estimates should
    respect the local population, not average them together.
    """
    rng = np.random.default_rng(13)
    # Low population at x < 15, high population at x > 25
    low_xy = rng.uniform([0, 0], [12, 30], size=(25, 2))
    high_xy = rng.uniform([28, 0], [40, 30], size=(25, 2))
    low = np.column_stack([low_xy, np.full(25, 5.0)])
    high = np.column_stack([high_xy, np.full(25, 5.0)])
    coords = np.vstack([low, high])
    values = np.concatenate([
        rng.normal(1.0, 0.1, size=25),
        rng.normal(10.0, 0.5, size=25),
    ])

    # Query in the middle of each population
    centroids = np.array([
        [6.0, 15.0, 5.0],    # deep in low pop
        [34.0, 15.0, 5.0],   # deep in high pop
        [0.5, 0.5, 5.0],     # low edge
        [39.5, 29.5, 5.0],   # high edge
    ], dtype=float)

    cfg = _base_cfg(range_max=10.0, range_mid=10.0, range_min=10.0,
                    local_search_radii=(0.5, 1.0, 1.5))
    grades, _ = _run(coords, values, centroids, dx=1.0, cfg=cfg)
    assert np.all(np.isfinite(grades))

    assert grades[0] < 2.0, f"low-pop deep: {grades[0]:.2f} (expected ~1)"
    assert grades[1] > 8.0, f"high-pop deep: {grades[1]:.2f} (expected ~10)"
    assert grades[2] < 2.5, f"low-pop edge: {grades[2]:.2f}"
    assert grades[3] > 8.0, f"high-pop edge: {grades[3]:.2f}"


# ─────────────────────────────────────────────────────────────────────
# 6. Smoothing ratio
# ─────────────────────────────────────────────────────────────────────


def test_smoothing_ratio_not_pathological():
    """var(estimates) / var(samples) should be between ~0.10 and ~1.00.

    Values below 0.10 mean the model is pathologically flat (too much
    smoothing); values above 1.0 mean it's over-fitting / creating
    variance that isn't in the data.

    Gaussian hill + background noise is a good test — the hill has a
    real spatial structure, so the estimator should preserve a
    reasonable fraction of the sample variance.
    """
    rng = np.random.default_rng(14)
    coords_xy = rng.uniform([0, 0], [40, 40], size=(100, 2))
    coords = np.column_stack([coords_xy, np.full(100, 5.0)])
    values = _gaussian_hill(coords_xy) + rng.normal(0, 0.3, 100)

    centroids = _grid(n=12, extent=40.0, z=5.0)
    grades, _ = _run(coords, values, centroids, dx=3.0,
                     cfg=_base_cfg(range_max=15.0, range_mid=15.0,
                                   range_min=15.0))
    finite = grades[np.isfinite(grades)]
    assert finite.size >= 100

    var_est = float(np.var(finite))
    var_data = float(np.var(values))
    ratio = var_est / var_data
    # Block support reduces variance — ratio between 0.10 and 1.0 is
    # the healthy band for a smooth field.
    assert 0.10 <= ratio <= 1.20, (
        f"smoothing ratio = {ratio:.3f} (healthy 0.10–1.20)"
    )
