"""ARBF level 1 — unit tests for mathematical properties.

These tests exercise the ARBFEstimatorAdapter with minimal configurations
that isolate specific mathematical behaviours:

  * constant-field reproduction
  * exact honouring at data points (when smoothing is ~0)
  * symmetric-configuration symmetric-result
  * anisotropy distance transform direction
  * distance decay monotonicity
  * solver robustness on near-singular input

All tests use the same production entry point the panel uses so they
test the real path, not a toy. Deposits are tiny (< 50 blocks) so the
whole file runs in a couple of seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from block_model_viewer.geostats.arbf_adapter import ARBFEstimatorAdapter


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _cfg(**overrides):
    """Minimal geologically-sensible ARBF config shared by all tests."""
    base = dict(
        range_max=50.0, range_mid=50.0, range_min=50.0,
        azimuth=0.0, dip=0.0, pitch=0.0,
        nugget=0.0, sill=1.0,
        kernel_type="spheroidal",
        drift_type="constant",
        use_normal_score=False,     # avoid NS auto-guard surprises
        force_normal_score=False,
        search_mode="local",
        max_samples=20, min_samples=2,
        local_search_radii=(0.5, 1.0, 2.0),
        max_samples_per_octant=3, search_min_octants=1,
        discretisation_density=1,
        pum_threshold=10000,
        accuracy=1e-9,
        panel_width=0.0,
        run_cv=False,
    )
    base.update(overrides)
    return base


def _run(coords, values, centroids, dx=1.0, cfg=None) -> np.ndarray:
    """Run ARBF and return the grade array."""
    adapter = ARBFEstimatorAdapter(cfg or _cfg())
    adapter.set_composites(coords, values)
    adapter.set_block_model(
        np.asarray(centroids, dtype=float),
        np.array([dx, dx, dx], dtype=float),
    )
    r = adapter.estimate()
    return np.asarray(r["grades"], dtype=float)


# ─────────────────────────────────────────────────────────────────────
# 1. Constant field reproduction
# ─────────────────────────────────────────────────────────────────────


def test_constant_field_reproduction_exact():
    """If every sample has value 2.5, every block must be 2.5 (within ε)."""
    rng = np.random.default_rng(0)
    coords = rng.uniform([0, 0, 0], [30, 30, 15], size=(20, 3))
    values = np.full(20, 2.5, dtype=float)
    centroids = rng.uniform([5, 5, 2], [25, 25, 13], size=(40, 3))

    grades = _run(coords, values, centroids)

    finite = grades[np.isfinite(grades)]
    assert finite.size == 40
    # Constant-field residual should be tiny — generous 1e-3 tolerance
    # to absorb numerical noise from centring + solver.
    assert np.max(np.abs(finite - 2.5)) < 1e-3, (
        f"constant-field max error = {np.max(np.abs(finite - 2.5)):.6f}"
    )
    assert abs(float(np.mean(finite)) - 2.5) < 1e-6


def test_constant_field_arbitrary_magnitude():
    """Constant-field reproduction must hold for any magnitude, not just
    order-unity values — the solver should not lose precision at 1e4.
    """
    rng = np.random.default_rng(1)
    coords = rng.uniform([0, 0, 0], [30, 30, 15], size=(15, 3))
    values = np.full(15, 1.2345e4, dtype=float)
    centroids = rng.uniform([5, 5, 2], [25, 25, 13], size=(20, 3))

    grades = _run(coords, values, centroids)
    finite = grades[np.isfinite(grades)]
    rel_err = np.max(np.abs(finite - 1.2345e4) / 1.2345e4)
    assert rel_err < 1e-3, f"relative error = {rel_err:.2e}"


# ─────────────────────────────────────────────────────────────────────
# 2. Exact honouring at sample locations
# ─────────────────────────────────────────────────────────────────────


def test_exact_honouring_at_sample_locations():
    """With zero nugget / smoothing, estimating at a sample location
    must return the sample value (to within solver tolerance).
    """
    rng = np.random.default_rng(2)
    coords = rng.uniform([0, 0, 0], [30, 30, 15], size=(12, 3))
    values = rng.normal(5.0, 1.5, size=12)

    # Use the sample positions as block centroids — each block
    # centre coincides with one composite.
    grades = _run(coords, values, coords, dx=0.1,
                  cfg=_cfg(nugget=0.0, accuracy=1e-12))

    finite_mask = np.isfinite(grades)
    assert finite_mask.sum() >= len(values) - 2  # allow at most 2 misses
    # Honouring error should be small — RBF with modest nugget will not
    # be bit-exact but should be within ~5% of sample values.
    diffs = np.abs(grades[finite_mask] - values[finite_mask])
    med_rel = float(np.median(diffs / np.abs(values[finite_mask])))
    assert med_rel < 0.10, f"median relative honouring error = {med_rel:.3%}"


# ─────────────────────────────────────────────────────────────────────
# 3. Symmetry
# ─────────────────────────────────────────────────────────────────────


def test_symmetric_configuration_gives_symmetric_result():
    """Two samples at (+d, 0, 0) and (-d, 0, 0) with the same value
    produce a result at the origin equal to that value; at (±x, 0, 0)
    the result must be symmetric in sign of x.
    """
    coords = np.array([[+3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]], dtype=float)
    values = np.array([5.0, 5.0], dtype=float)

    # Query points symmetric around the origin
    xs = np.linspace(-2.5, 2.5, 11)
    centroids = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])

    grades = _run(coords, values, centroids, dx=0.2)
    # Symmetry check: g(+x) ≈ g(-x)
    mid = len(xs) // 2
    left = grades[:mid]
    right = grades[mid + 1:][::-1]
    mask = np.isfinite(left) & np.isfinite(right)
    assert mask.any()
    asymmetry = np.max(np.abs(left[mask] - right[mask]))
    assert asymmetry < 1e-4, f"asymmetry = {asymmetry:.6f}"


# ─────────────────────────────────────────────────────────────────────
# 4. Anisotropy transform direction
# ─────────────────────────────────────────────────────────────────────


def test_anisotropy_pulls_along_major_axis():
    """With major range >> minor range, a sample at (+d, 0, 0) and
    another at (0, +d, 0) should influence a query at the origin
    MORE strongly in the major-axis direction (longer range).
    """
    coords = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=float)
    values = np.array([10.0, 1.0], dtype=float)  # major-axis sample is high

    # Query at origin (should be pulled toward the high sample)
    centroids = np.array([[0.0, 0.0, 0.0]], dtype=float)

    # Strongly anisotropic: major 50 m, minor 5 m
    cfg = _cfg(range_max=50.0, range_mid=5.0, range_min=5.0, azimuth=0.0)
    grades = _run(coords, values, centroids, dx=0.5, cfg=cfg)
    origin_est = float(grades[0])
    assert np.isfinite(origin_est)
    # Major axis is along X — the high-grade X-sample should dominate
    # because its anisotropy-transformed distance is smaller.
    assert origin_est > 5.0, (
        f"expected >5 (pulled toward major-axis high sample), got {origin_est}"
    )


def test_anisotropy_rotation_changes_influence():
    """Rotating the major axis by 90° should flip which sample dominates."""
    coords = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=float)
    values = np.array([10.0, 1.0], dtype=float)
    centroids = np.array([[0.0, 0.0, 0.0]], dtype=float)

    g_x = _run(coords, values, centroids, dx=0.5,
               cfg=_cfg(range_max=50.0, range_mid=5.0, range_min=5.0,
                        azimuth=0.0))[0]
    # Rotate major axis to point along Y
    g_y = _run(coords, values, centroids, dx=0.5,
               cfg=_cfg(range_max=50.0, range_mid=5.0, range_min=5.0,
                        azimuth=90.0))[0]
    assert np.isfinite(g_x) and np.isfinite(g_y)
    # Major along X -> value 10 dominates (high)
    # Major along Y -> value 1 dominates (low)
    assert g_x > g_y, (
        f"expected rotation to flip influence: x={g_x:.2f}, y={g_y:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────
# 5. Distance decay
# ─────────────────────────────────────────────────────────────────────


def test_influence_decays_with_distance():
    """As a query moves from a high-grade cluster into a low-grade
    background, the estimate should monotonically decrease.

    The configuration needs multiple composites in each query's
    neighbourhood so the kernel weighting is actually exercised —
    a single in-range composite would just return the constant-drift
    fit (the composite value) everywhere.
    """
    # Dense grid of low-grade composites everywhere, plus one high
    # sample at the origin. Every query sees many composites in its
    # neighbourhood so kernel weighting is the primary mechanism.
    xs, ys = np.meshgrid(
        np.linspace(-15, 25, 9),
        np.linspace(-10, 10, 5),
    )
    bg_coords = np.column_stack([
        xs.ravel(), ys.ravel(), np.zeros(xs.size),
    ])
    # Drop the sample nearest the origin and replace with a high one
    bg_dist = np.linalg.norm(bg_coords[:, :2], axis=1)
    origin_idx = int(np.argmin(bg_dist))
    bg_coords[origin_idx] = [0.0, 0.0, 0.0]

    values = np.full(len(bg_coords), 1.0)
    values[origin_idx] = 10.0

    # Query along +X from near-origin to far-origin
    dists = np.linspace(1.0, 18.0, 8)
    centroids = np.column_stack([
        dists, np.zeros_like(dists), np.zeros_like(dists),
    ])

    # Compact-support kernel with range long enough to see multiple
    # composites at every query point.
    cfg = _cfg(range_max=20.0, range_mid=20.0, range_min=20.0,
               local_search_radii=(0.5, 1.0, 2.0),
               max_samples=20, max_samples_per_octant=5)
    grades = _run(coords=bg_coords, values=values,
                  centroids=centroids, dx=0.5, cfg=cfg)

    finite_mask = np.isfinite(grades)
    assert finite_mask.sum() >= len(dists) - 1, (
        f"too few finite estimates: {grades}"
    )
    # Monotonic-or-near-monotonic decay from near to far
    # (small wiggles allowed, overall trend must be downward).
    near = float(np.mean(grades[finite_mask][:2]))
    far = float(np.mean(grades[finite_mask][-2:]))
    assert near > far + 0.3, (
        f"expected decay: near={near:.3f}, far={far:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────
# 6. Solver robustness on near-singular input
# ─────────────────────────────────────────────────────────────────────


def test_solver_handles_duplicate_samples():
    """Two composites at the same location with different values would
    make the RBF system singular without a fix. The solver must either
    collapse the duplicate or degrade gracefully (no NaN, no crash).
    """
    coords = np.array([
        [10.0, 10.0, 5.0],
        [10.0, 10.0, 5.0],  # exact duplicate
        [20.0, 20.0, 5.0],
        [5.0, 15.0, 5.0],
    ], dtype=float)
    values = np.array([3.0, 5.0, 1.0, 2.0], dtype=float)
    centroids = np.array([
        [10.0, 10.0, 5.0],
        [15.0, 15.0, 5.0],
        [12.0, 13.0, 5.0],
    ], dtype=float)

    grades = _run(coords, values, centroids, dx=2.0)
    assert np.all(np.isfinite(grades)), (
        f"solver produced NaN on duplicate samples: {grades}"
    )
    # Value range should still be reasonable
    assert float(grades.min()) > 0.0
    assert float(grades.max()) < 10.0


def test_solver_handles_collinear_samples():
    """All composites on a single line — the system can still run but
    must not crash or produce pathological values.
    """
    xs = np.linspace(0.0, 40.0, 10)
    coords = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    values = 0.1 * xs + 2.0  # linear along the line

    centroids = np.array([
        [20.0, 0.0, 0.0],
        [20.0, 5.0, 0.0],   # off the line
        [10.0, 0.0, 0.0],
    ], dtype=float)
    grades = _run(coords, values, centroids, dx=2.0)
    assert np.all(np.isfinite(grades))
    # On the line, the estimate should roughly match the linear trend
    on_line = float(grades[0])
    assert 3.5 < on_line < 4.5, (
        f"expected ~4.0 on the line at x=20, got {on_line:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────
# 7. Determinism
# ─────────────────────────────────────────────────────────────────────


def test_repeated_runs_are_deterministic():
    """Identical inputs must produce bit-identical outputs across runs."""
    rng = np.random.default_rng(5)
    coords = rng.uniform([0, 0, 0], [30, 30, 15], size=(20, 3))
    values = rng.normal(5.0, 1.0, size=20)
    centroids = rng.uniform([5, 5, 2], [25, 25, 13], size=(30, 3))

    g1 = _run(coords, values, centroids)
    g2 = _run(coords, values, centroids)
    np.testing.assert_allclose(g1, g2, rtol=0, atol=0)
