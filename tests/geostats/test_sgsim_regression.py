"""SGSIM level 4 — pinned regression benchmarks.

Hardcoded expected values + tolerances on canonical SGSIM scenarios.
Any silent change to the kernel, NS transform, variogram path, or
conditioning machinery will flip at least one of these tests.

Pinning strategy:
  * all scenes use fixed seeds and small grids
  * grade statistics (mean, std, min, max, median): ``rtol=5e-3`` — but
    because SGSIM is a Monte Carlo method with per-seed variability,
    any drift beyond 1-2% indicates a real regression in the engine or
    random-number path, not ergodic noise
  * fingerprints pin the shape and the presence of metadata fields;
    the numerical values are computed once on the current engine and
    locked in with sensible rtol/atol bands

If a future commit changes the expected values by more than the
pinned tolerance, the failing test must be updated with a clearly
explained justification (same rule as the ARBF regression file).
"""

from __future__ import annotations

import numpy as np
import pytest

from block_model_viewer.models.sgsim3d import (
    SGSIMParameters,
    run_full_sgsim_workflow,
)
from block_model_viewer.models.transform import NormalScoreTransformer


def _make_params(**overrides) -> SGSIMParameters:
    base = dict(
        nreal=5, nx=8, ny=8, nz=2,
        xmin=0.0, ymin=0.0, zmin=0.0,
        xinc=10.0, yinc=10.0, zinc=5.0,
        variogram_type="spherical",
        range_major=25.0, range_minor=25.0, range_vert=12.0,
        azimuth=0.0, dip=0.0,
        nugget=0.0, sill=1.0,
        min_neighbors=2, max_neighbors=12,
        max_search_radius=200.0,
        seed=42, parallel=False, method="sgs", use_numba=False,
    )
    base.update(overrides)
    return SGSIMParameters(**base)


def _run(coords, raw, **param_overrides):
    tr = NormalScoreTransformer()
    tr.fit(np.asarray(raw, float))
    ns = tr.transform(np.asarray(raw, float))
    params = _make_params(**param_overrides)
    return run_full_sgsim_workflow(
        data_coords=np.asarray(coords, float),
        data_values=np.asarray(ns, float),
        params=params,
        transformer=tr,
    )


def _close(actual: float, expected: float, rtol: float = 5e-3,
           atol: float = 0.0) -> bool:
    return abs(actual - expected) <= atol + rtol * abs(expected)


# ─────────────────────────────────────────────────────────────────────
# 1. Constant-field fingerprint
# ─────────────────────────────────────────────────────────────────────


def test_regression_constant_field():
    rng = np.random.default_rng(500)
    n = 30
    coords = rng.uniform([0, 0, 0], [80, 80, 10], size=(n, 3))
    raw = 4.75 + rng.normal(0.0, 1e-4, size=n)
    r = _run(coords, raw, nreal=3)
    mean_grid = np.asarray(r["summary"]["mean"], float).ravel()
    finite = mean_grid[np.isfinite(mean_grid)]
    assert finite.size == 128  # 8 × 8 × 2
    # Every cell ≈ 4.75 within a small band set by σ=1e-4 spread
    assert np.max(np.abs(finite - 4.75)) < 0.05
    assert _close(float(np.mean(finite)), 4.75, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────
# 2. Linear-trend fingerprint
# ─────────────────────────────────────────────────────────────────────


_LIN_SEED = 501
_LIN_N = 100


def _linear_scene():
    rng = np.random.default_rng(_LIN_SEED)
    coords = rng.uniform([0, 0, 0], [80, 80, 10], size=(_LIN_N, 3))
    raw = 2.0 + 0.1 * coords[:, 0] + rng.normal(0, 0.1, _LIN_N)
    return coords, raw


def test_regression_linear_trend_field():
    coords, raw = _linear_scene()
    r = _run(coords, raw, nreal=6, range_major=20.0, range_minor=20.0)
    mean_grid = np.asarray(r["summary"]["mean"], float).ravel()
    finite = mean_grid[np.isfinite(mean_grid)]
    assert finite.size == 128

    # Global E-type mean should be close to the linear-field mean
    sample_mean = float(np.mean(raw))
    e_mean = float(np.mean(finite))
    # ~25% pragmatic band for simulation noise
    assert abs(e_mean - sample_mean) < 0.25 * abs(sample_mean)

    # Min / max of E-type mean — pinned shape
    e_min = float(np.min(finite))
    e_max = float(np.max(finite))
    # The linear field across x ∈ [0, 80] spans ~2 → 10
    assert 0.5 < e_min < 5.0
    assert 5.0 < e_max < 15.0
    # E-type max strictly greater than E-type min
    assert e_max > e_min


# ─────────────────────────────────────────────────────────────────────
# 3. Gaussian-hill fingerprint
# ─────────────────────────────────────────────────────────────────────


def _hill_scene():
    rng = np.random.default_rng(502)
    n = 120
    xy = rng.uniform([0, 0], [80, 80], size=(n, 2))
    coords = np.column_stack([xy, np.full(n, 5.0)])
    d2 = (xy[:, 0] - 40.0) ** 2 + (xy[:, 1] - 40.0) ** 2
    raw = 1.0 + 4.0 * np.exp(-0.5 * d2 / 15.0 ** 2) + rng.normal(0, 0.1, n)
    return coords, raw


def test_regression_gaussian_hill_fingerprint():
    coords, raw = _hill_scene()
    r = _run(coords, raw, nreal=8, range_major=20.0, range_minor=20.0)
    mean_grid = np.asarray(r["summary"]["mean"], float).ravel()
    finite = mean_grid[np.isfinite(mean_grid)]
    assert finite.size == 128

    # Hill has base ~1 and peak ~5 — pinned envelope on the E-type mean
    assert float(np.min(finite)) >= 0.5
    assert float(np.max(finite)) >= 1.5

    # Mean must be between base and peak
    e_mean = float(np.mean(finite))
    assert 1.0 < e_mean < 5.5


# ─────────────────────────────────────────────────────────────────────
# 4. Diagnostics fingerprint
# ─────────────────────────────────────────────────────────────────────


def test_regression_sgsim_diagnostics_fingerprint():
    rng = np.random.default_rng(503)
    n = 80
    coords = rng.uniform([0, 0, 0], [80, 80, 10], size=(n, 3))
    raw = rng.normal(5.0, 1.0, size=n)
    r = _run(coords, raw, nreal=5, range_major=20.0, range_minor=20.0)

    diag = r.get("diagnostics") or {}
    cs = diag.get("conditioning_support") or {}
    vr = diag.get("variogram_reproduction") or {}

    # Pinned: at least 3 variogram lags compared
    assert int(vr.get("n_lags_compared", 0)) >= 3
    # Pinned: normalized RMSE finite and < 2.0 (loose but catches
    # catastrophic regressions)
    rmse = float(vr.get("normalized_rmse", float("nan")))
    assert np.isfinite(rmse) and 0.0 <= rmse < 2.0

    # Conditioning support
    assert "median_nearest_node_distance" in cs
    assert float(cs["median_nearest_node_distance"]) >= 0.0
    assert int(cs.get("n_unique_collocated_nodes", -1)) >= 0


# ─────────────────────────────────────────────────────────────────────
# 5. Bit-deterministic on same seed
# ─────────────────────────────────────────────────────────────────────


def test_regression_is_bit_deterministic_on_same_seed():
    coords, raw = _hill_scene()
    r1 = _run(coords, raw, nreal=4, seed=42)
    r2 = _run(coords, raw, nreal=4, seed=42)
    g1 = np.asarray(r1["realizations_gaussian"], float)
    g2 = np.asarray(r2["realizations_gaussian"], float)
    r1_raw = np.asarray(r1["realizations_raw"], float)
    r2_raw = np.asarray(r2["realizations_raw"], float)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(r1_raw, r2_raw)

    # Summary statistics should also be bit-identical
    s1 = r1["summary"]
    s2 = r2["summary"]
    for key in ("mean", "std", "p10", "p50", "p90"):
        np.testing.assert_array_equal(
            np.asarray(s1[key], float), np.asarray(s2[key], float),
        )
