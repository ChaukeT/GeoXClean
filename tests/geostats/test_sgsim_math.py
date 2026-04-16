"""SGSIM level 1 — unit tests for mathematical properties.

These tests exercise the real production ``run_full_sgsim_workflow``
entry point with minimal configurations that isolate specific
mathematical behaviours of Sequential Gaussian Simulation:

  * correct realisation count and shape
  * deterministic runs on fixed seed
  * different seeds produce different realisations
  * missing transformer is rejected
  * method-alias normalisation
  * constant-field reproduction
  * Gaussian-space prior statistics (mean ≈ 0, std ≈ √sill)
  * back-transform is monotone at every grid node

All tests use the same production entry point the panel drives through
the controller, so they exercise shipped code. Grids are ≤ 10×10×2 so
the whole file runs in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from block_model_viewer.models.sgsim3d import (
    SGSIMParameters,
    _normalize_simulation_method,
    run_full_sgsim_workflow,
)
from block_model_viewer.models.transform import NormalScoreTransformer


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _small_grid_params(**overrides) -> SGSIMParameters:
    base = dict(
        nreal=5,
        nx=8, ny=8, nz=2,
        xmin=0.0, ymin=0.0, zmin=0.0,
        xinc=10.0, yinc=10.0, zinc=5.0,
        variogram_type="spherical",
        range_major=50.0, range_minor=50.0, range_vert=25.0,
        azimuth=0.0, dip=0.0,
        nugget=0.0, sill=1.0,
        min_neighbors=2, max_neighbors=12,
        max_search_radius=200.0,
        seed=42,
        parallel=False,
        method="sgs",
        use_numba=False,
    )
    base.update(overrides)
    return SGSIMParameters(**base)


def _fit_transformer(values: np.ndarray) -> NormalScoreTransformer:
    t = NormalScoreTransformer()
    t.fit(np.asarray(values, dtype=float))
    return t


def _conditioning(n: int = 30, seed: int = 0,
                   extent=(80.0, 80.0, 10.0),
                   mean: float = 5.0, std: float = 1.0):
    """Generate (coords, raw_values, ns_values, transformer)."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform([0, 0, 0], list(extent), size=(n, 3))
    raw_values = rng.normal(mean, std, size=n)
    # Ensure positive for NS transform robustness
    raw_values = np.maximum(raw_values, 0.01)
    transformer = _fit_transformer(raw_values)
    ns_values = transformer.transform(raw_values)
    return coords, raw_values, ns_values, transformer


def _run(coords, ns_values, transformer, **param_overrides) -> dict:
    params = _small_grid_params(**param_overrides)
    return run_full_sgsim_workflow(
        data_coords=np.asarray(coords, float),
        data_values=np.asarray(ns_values, float),
        params=params,
        transformer=transformer,
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Realisation count + shape + dtype
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_returns_requested_number_of_realisations():
    coords, raw, ns, tr = _conditioning(n=30, seed=1)
    r = _run(coords, ns, tr, nreal=5)
    assert r["realizations_gaussian"].shape[0] == 5
    assert r["realizations_raw"].shape[0] == 5


def test_sgsim_realisation_shape_and_dtype():
    coords, raw, ns, tr = _conditioning(n=30, seed=2)
    r = _run(coords, ns, tr, nreal=3, nx=8, ny=7, nz=2)
    rg = r["realizations_gaussian"]
    rr = r["realizations_raw"]
    assert rg.shape == (3, 2, 7, 8)     # (nreal, nz, ny, nx)
    assert rr.shape == (3, 2, 7, 8)
    assert rg.dtype == np.float64
    assert rr.dtype == np.float64
    # Interior cells should all be finite (no NaNs when no domain mask)
    assert np.all(np.isfinite(rg))
    assert np.all(np.isfinite(rr))


# ─────────────────────────────────────────────────────────────────────
# 2. Determinism + seed sensitivity
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_deterministic_with_fixed_seed():
    coords, raw, ns, tr = _conditioning(n=30, seed=3)
    r1 = _run(coords, ns, tr, seed=123)
    r2 = _run(coords, ns, tr, seed=123)
    np.testing.assert_array_equal(
        r1["realizations_gaussian"], r2["realizations_gaussian"]
    )
    np.testing.assert_array_equal(
        r1["realizations_raw"], r2["realizations_raw"]
    )


def test_sgsim_different_seeds_produce_different_realisations():
    coords, raw, ns, tr = _conditioning(n=30, seed=4)
    r1 = _run(coords, ns, tr, seed=101)
    r2 = _run(coords, ns, tr, seed=202)
    diff = np.abs(r1["realizations_gaussian"] - r2["realizations_gaussian"])
    # RMS difference across all cells and all realisations
    rms = float(np.sqrt(np.mean(diff ** 2)))
    assert rms > 0.1, f"different seeds gave identical realisations (rms={rms})"


# ─────────────────────────────────────────────────────────────────────
# 3. Transformer-required gate
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_requires_transformer():
    coords, raw, ns, tr = _conditioning(n=20, seed=5)
    params = _small_grid_params(nreal=2)
    with pytest.raises(ValueError, match="transformer"):
        run_full_sgsim_workflow(
            data_coords=coords,
            data_values=ns,
            params=params,
            transformer=None,
        )


# ─────────────────────────────────────────────────────────────────────
# 4. Method alias normalisation
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_method_alias_normalisation():
    assert _normalize_simulation_method("sgs") == "sgs"
    assert _normalize_simulation_method("sgsim") == "sgs"
    assert _normalize_simulation_method("sequential") == "sgs"
    assert _normalize_simulation_method("fft_ma") == "fft_ma"
    assert _normalize_simulation_method("fftma") == "fft_ma"
    assert _normalize_simulation_method("fft-ma") == "fft_ma"
    assert _normalize_simulation_method("SGS") == "sgs"
    with pytest.raises(ValueError):
        _normalize_simulation_method("nonsense")


# ─────────────────────────────────────────────────────────────────────
# 5. Gaussian-space prior statistics
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_gaussian_space_mean_near_zero():
    """With spatially random conditioning data (NS values ≈ N(0,1)), the
    Gaussian-space simulation mean should be close to 0 across the grid.
    """
    coords, raw, ns, tr = _conditioning(n=40, seed=6)
    r = _run(coords, ns, tr, nreal=5)
    g = r["realizations_gaussian"]
    # Aggregate Gaussian mean across all realisations + cells
    mean_g = float(np.mean(g))
    assert abs(mean_g) < 0.5, f"Gaussian-space mean = {mean_g:.3f}"


def test_sgsim_gaussian_space_std_near_sqrt_sill():
    """With sill=1.0 and enough realisations, the Gaussian-space std
    should be close to 1.0.
    """
    coords, raw, ns, tr = _conditioning(n=40, seed=7)
    r = _run(coords, ns, tr, nreal=8, sill=1.0)
    g = r["realizations_gaussian"]
    std_g = float(np.std(g))
    assert 0.5 <= std_g <= 1.5, f"Gaussian-space std = {std_g:.3f}"


# ─────────────────────────────────────────────────────────────────────
# 6. Back-transform monotonicity
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_back_transform_is_monotone():
    """At every grid node, the rank of realisations in Gaussian space
    must equal the rank of realisations in raw space — the back
    transform is strictly monotone at each node.
    """
    coords, raw, ns, tr = _conditioning(n=30, seed=8)
    r = _run(coords, ns, tr, nreal=6)
    g = r["realizations_gaussian"]    # (nreal, nz, ny, nx)
    rr = r["realizations_raw"]

    # Flatten spatial dims → (nreal, n_cells)
    g_flat = g.reshape(g.shape[0], -1)
    r_flat = rr.reshape(rr.shape[0], -1)

    # For each cell, check rank order matches
    n_cells = g_flat.shape[1]
    # Sample a handful of cells to keep the test quick
    rng_idx = np.arange(0, n_cells, max(n_cells // 20, 1))
    mismatches = 0
    for ci in rng_idx:
        g_col = g_flat[:, ci]
        r_col = r_flat[:, ci]
        if np.any(~np.isfinite(g_col)) or np.any(~np.isfinite(r_col)):
            continue
        order_g = np.argsort(g_col)
        order_r = np.argsort(r_col)
        if not np.array_equal(order_g, order_r):
            mismatches += 1
    assert mismatches == 0, (
        f"back-transform not monotone at {mismatches} cells"
    )


# ─────────────────────────────────────────────────────────────────────
# 7. Constant-field reproduction
# ─────────────────────────────────────────────────────────────────────


def test_sgsim_constant_field_reproduction():
    """If every conditioning sample equals the same value, every
    realisation cell must equal that value (in raw space) because the
    NS transform collapses to a single point and exact-sgs conditioning
    reproduces it at every node.

    Note: the NS transform of a strictly-constant input is degenerate,
    so we use a tightly-clustered distribution (σ = 1e-4) instead of
    an exact constant to keep the transformer well-defined.
    """
    rng = np.random.default_rng(9)
    n = 30
    coords = rng.uniform([0, 0, 0], [80, 80, 10], size=(n, 3))
    raw_values = 5.0 + rng.normal(0.0, 1e-4, size=n)
    transformer = _fit_transformer(raw_values)
    ns_values = transformer.transform(raw_values)
    r = _run(coords, ns_values, transformer, nreal=3)
    rr = r["realizations_raw"]
    finite = rr[np.isfinite(rr)]
    # Every raw realisation cell should be ≈ 5.0 within a small
    # tolerance set by the σ=1e-4 spread in the conditioning set.
    assert np.max(np.abs(finite - 5.0)) < 0.1, (
        f"constant-field reproduction failed: max err "
        f"{np.max(np.abs(finite - 5.0))}"
    )
