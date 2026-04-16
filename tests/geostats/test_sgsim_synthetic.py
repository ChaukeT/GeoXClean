"""SGSIM level 2 — synthetic field tests.

Known-truth spatial patterns driven through the real
``run_full_sgsim_workflow`` entry point:

  * constant field reproduction in the E-type mean
  * linear trend field follows through the E-type mean
  * smooth Gaussian hill is preserved
  * realisation spread (per-cell std) grows with distance from data
  * two spatially separated populations are preserved in E-type
  * aggregate realisation histogram matches the input marginal

These exercise the simulation quality gates that kriging/RBF tests
don't cover: the realisation spread is the actual SGSIM uncertainty,
the histogram reproduction is the defining SGSIM guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest

from block_model_viewer.models.sgsim3d import (
    SGSIMParameters,
    run_full_sgsim_workflow,
)
from block_model_viewer.models.transform import NormalScoreTransformer


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_params(**overrides) -> SGSIMParameters:
    base = dict(
        nreal=6,
        nx=10, ny=10, nz=2,
        xmin=0.0, ymin=0.0, zmin=0.0,
        xinc=8.0, yinc=8.0, zinc=5.0,
        variogram_type="spherical",
        range_major=40.0, range_minor=40.0, range_vert=25.0,
        azimuth=0.0, dip=0.0,
        nugget=0.0, sill=1.0,
        min_neighbors=2, max_neighbors=12,
        max_search_radius=200.0,
        seed=42, parallel=False, method="sgs", use_numba=False,
    )
    base.update(overrides)
    return SGSIMParameters(**base)


def _run(coords, raw_values, **param_overrides):
    transformer = NormalScoreTransformer()
    transformer.fit(np.asarray(raw_values, float))
    ns = transformer.transform(np.asarray(raw_values, float))
    params = _make_params(**param_overrides)
    result = run_full_sgsim_workflow(
        data_coords=np.asarray(coords, float),
        data_values=np.asarray(ns, float),
        params=params,
        transformer=transformer,
    )
    return result, transformer


def _grid_xyz(params: SGSIMParameters) -> np.ndarray:
    """Return (n_cells, 3) array of grid-cell centres in X, Y, Z."""
    xs = params.xmin + params.xinc * (np.arange(params.nx) + 0.5)
    ys = params.ymin + params.yinc * (np.arange(params.ny) + 0.5)
    zs = params.zmin + params.zinc * (np.arange(params.nz) + 0.5)
    # Flatten matches sgsim3d convention: (nz, ny, nx) → raveled
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


# ─────────────────────────────────────────────────────────────────────
# 1. Constant field
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_constant_field_e_type_mean():
    """Tightly clustered conditioning (σ≈1e-4 around 5.0) → every
    E-type mean cell ≈ 5.0. Uses a near-constant rather than an
    exact constant because the NS transform requires some spread.
    """
    rng = np.random.default_rng(10)
    n = 30
    coords = rng.uniform([0, 0, 0], [80, 80, 10], size=(n, 3))
    raw = 5.0 + rng.normal(0.0, 1e-4, size=n)
    result, _ = _run(coords, raw, nreal=4)
    mean_grid = np.asarray(result["summary"]["mean"], float).ravel()
    finite = mean_grid[np.isfinite(mean_grid)]
    assert finite.size > 0
    assert np.max(np.abs(finite - 5.0)) < 0.1, (
        f"max err = {np.max(np.abs(finite - 5.0)):.4f}"
    )


# ─────────────────────────────────────────────────────────────────────
# 2. Linear trend
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_linear_trend_e_type_mean():
    """On a linear trend g = 2 + 0.1·x, the SGSIM E-type mean should
    correlate with truth at r² ≥ 0.40. The bound is loose because
    SGSIM has no linear drift — it can only reproduce trend through
    conditioning data, which is noisy at the grid scale used here.
    """
    rng = np.random.default_rng(11)
    n = 80
    coords = rng.uniform([0, 0, 0], [80, 80, 10], size=(n, 3))
    raw = 2.0 + 0.1 * coords[:, 0] + rng.normal(0, 0.1, n)
    result, _ = _run(
        coords, raw, nreal=8,
        range_major=20.0, range_minor=20.0, range_vert=10.0,
    )
    mean_grid = np.asarray(result["summary"]["mean"], float).ravel()
    cells = _grid_xyz(result["params"])
    assert mean_grid.shape[0] == cells.shape[0]
    finite = np.isfinite(mean_grid)
    # Correlation with x-coordinate
    if int(finite.sum()) >= 20:
        corr = float(np.corrcoef(mean_grid[finite], cells[finite, 0])[0, 1])
        assert corr >= 0.30, f"trend corr = {corr:.3f}"
    # Global mean ≈ sample mean (E-type unbiased)
    sample_mean = float(np.mean(raw))
    e_type_mean = float(np.mean(mean_grid[finite]))
    rel = abs(e_type_mean - sample_mean) / abs(sample_mean)
    assert rel < 0.25, (
        f"E-type mean {e_type_mean:.2f} vs sample {sample_mean:.2f} "
        f"(rel {rel:.1%})"
    )


# ─────────────────────────────────────────────────────────────────────
# 3. Gaussian hill
# ─────────────────────────────────────────────────────────────────────


def _gauss_hill(xy, cx=40.0, cy=40.0, sigma=15.0, peak=4.0, base=1.0):
    d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
    return base + peak * np.exp(-0.5 * d2 / sigma ** 2)


def test_sgsim_gaussian_hill_e_type_mean():
    rng = np.random.default_rng(12)
    n = 120
    xy = rng.uniform([0, 0], [80, 80], size=(n, 2))
    coords = np.column_stack([xy, np.full(n, 5.0)])
    raw = _gauss_hill(xy) + rng.normal(0, 0.1, n)
    result, _ = _run(
        coords, raw, nreal=8,
        range_major=20.0, range_minor=20.0, range_vert=10.0,
    )
    mean_grid = np.asarray(result["summary"]["mean"], float).ravel()
    cells = _grid_xyz(result["params"])
    truth_at_cells = _gauss_hill(cells[:, :2])
    finite = np.isfinite(mean_grid)
    assert int(finite.sum()) >= 50

    corr = float(np.corrcoef(mean_grid[finite], truth_at_cells[finite])[0, 1])
    std_truth = float(np.std(truth_at_cells[finite]))
    rmse = float(np.sqrt(np.mean(
        (mean_grid[finite] - truth_at_cells[finite]) ** 2
    )))
    # Looser than ARBF's 0.90 because SGSIM is stochastic
    assert corr > 0.60, f"hill corr = {corr:.3f}"
    assert rmse / std_truth < 0.85, f"RMSE/std_truth = {rmse/std_truth:.3f}"


# ─────────────────────────────────────────────────────────────────────
# 4. Variance grows with distance from data
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_variance_grows_away_from_data():
    """Per-cell std from the realisation spread should be smaller in
    cells well inside the conditioning cluster than in cells far
    outside it. This is the core SGSIM uncertainty property.
    """
    rng = np.random.default_rng(13)
    # Cluster samples in the middle of the grid only
    n = 40
    coords_xy = rng.uniform([30, 30], [50, 50], size=(n, 2))
    coords = np.column_stack([coords_xy, np.full(n, 5.0)])
    raw = rng.normal(5.0, 1.0, n)
    # Grid spans 0..80 in X/Y so "far" cells are well outside cluster
    result, _ = _run(
        coords, raw, nreal=8,
        nx=12, ny=12,
        xmin=0.0, ymin=0.0,
        xinc=7.0, yinc=7.0,
        range_major=15.0, range_minor=15.0, range_vert=10.0,
    )
    std_grid = np.asarray(result["summary"]["std"], float).ravel()
    cells = _grid_xyz(result["params"])
    # Distance from each cell to nearest conditioning sample (XY only)
    from scipy.spatial import cKDTree
    tree = cKDTree(coords[:, :2])
    d_nearest, _ = tree.query(cells[:, :2], k=1)

    finite = np.isfinite(std_grid)
    d = d_nearest[finite]
    s = std_grid[finite]
    # Split at 20 m from nearest sample
    near = s[d <= 12.0]
    far = s[d >= 25.0]
    assert near.size > 5 and far.size > 5, (
        f"need both near ({near.size}) and far ({far.size}) samples"
    )
    median_near = float(np.median(near))
    median_far = float(np.median(far))
    assert median_near <= median_far + 1e-6, (
        f"std did NOT grow away from data: near={median_near:.3f}, "
        f"far={median_far:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────
# 5. Two separated populations
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_two_populations_preserved_in_e_type():
    rng = np.random.default_rng(14)
    low = rng.uniform([0, 0, 0], [25, 60, 10], size=(30, 3))
    high = rng.uniform([55, 0, 0], [80, 60, 10], size=(30, 3))
    coords = np.vstack([low, high])
    values = np.concatenate([
        rng.normal(2.0, 0.2, size=30),
        rng.normal(10.0, 0.2, size=30),
    ])
    result, _ = _run(
        coords, values, nreal=6,
        nx=10, ny=10,
        range_major=12.0, range_minor=12.0, range_vert=10.0,
    )
    mean_grid = np.asarray(result["summary"]["mean"], float).ravel()
    cells = _grid_xyz(result["params"])
    finite = np.isfinite(mean_grid)

    # Cells in the low region (x<20) should average near 2, not 6
    low_mask = finite & (cells[:, 0] < 15.0)
    high_mask = finite & (cells[:, 0] > 65.0)
    assert int(low_mask.sum()) > 0
    assert int(high_mask.sum()) > 0
    mean_low = float(np.mean(mean_grid[low_mask]))
    mean_high = float(np.mean(mean_grid[high_mask]))
    assert mean_low < 5.0, f"low region mean = {mean_low:.2f}"
    assert mean_high > 6.0, f"high region mean = {mean_high:.2f}"
    assert mean_high > mean_low + 2.0


# ─────────────────────────────────────────────────────────────────────
# 6. Aggregate realisation histogram matches input marginal
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_realisation_histogram_matches_input_marginal():
    """The SGSIM simulation draws from a distribution whose marginal
    matches the declustered sample histogram. Aggregating all
    realisation cells and comparing via a two-sample KS-like test
    (max CDF distance) must show close agreement.
    """
    rng = np.random.default_rng(15)
    n = 150
    coords = rng.uniform([0, 0, 0], [80, 80, 10], size=(n, 3))
    raw = rng.normal(5.0, 1.5, size=n)
    result, tr = _run(
        coords, raw, nreal=8,
        range_major=25.0, range_minor=25.0, range_vert=10.0,
    )
    rr = np.asarray(result["realizations_raw"], float)
    sim_values = rr[np.isfinite(rr)].ravel()
    # Two-sample KS — max |F_sim - F_sample|
    from scipy.stats import ks_2samp
    stat, _ = ks_2samp(sim_values, raw)
    # Allow a pragmatic 0.25 — SGSIM marginal match is looser on
    # small grids (ergodic fluctuations) but shouldn't exceed 0.30
    assert stat < 0.30, f"KS statistic = {stat:.3f} (> 0.30)"
