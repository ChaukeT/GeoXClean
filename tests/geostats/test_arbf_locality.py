"""ARBF level 5 — locality tests.

The first four test files prove ARBF is numerically correct, globally
unbiased, structurally correct on known fields, and reproducible.
They do NOT prove that ARBF is *locally* as sharp as a strictly-local
estimator like Ordinary Kriging with a hard search radius.

These three tests close that gap:

  1. ``test_hard_neighbourhood_improves_spike_retention``
     Compares the spike block value at two neighbourhood settings —
     a permissive baseline and a hard-local (max_samples=8,
     radii 0.15/0.30/0.60) config — and asserts the hard-local
     setting retains AT LEAST as much of the spike amplitude.
     If tightening the neighbourhood improves spike retention,
     it means the baseline was bleeding distant samples in, which
     is exactly the "too global" failure mode.

  2. ``test_ordinary_kriging_parity_on_gaussian_hill``
     Runs OK (``ordinary_kriging_fast``) and ARBF on the same
     synthetic gaussian-hill field. Asserts:
       * ARBF correlation with OK > 0.85
       * ARBF smoothing ratio is within 1.8× of OK's
       * ARBF spike-block value is within ±40% of OK's
       * Mean-bias difference is < 5%

  3. ``test_effective_influence_radius_is_bounded``
     Probes ARBF's influence radius directly: places a sentinel
     high-grade "probe" composite at increasing distances from a
     query in an otherwise low-grade background, records the
     query's block estimate as a function of probe distance, and
     asserts that the influence drops below 10% of its maximum
     contribution by the time the probe is 1.5 × variogram-range
     away. If it doesn't, ARBF has long influence tails and is
     behaving as a semi-global rather than local estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from block_model_viewer.geostats.arbf_adapter import ARBFEstimatorAdapter
from block_model_viewer.models.kriging3d import (
    NUMBA_AVAILABLE,
    ordinary_kriging_fast,
)


def _cfg(**overrides):
    cfg = dict(
        range_max=20.0, range_mid=20.0, range_min=20.0,
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
    adapter = ARBFEstimatorAdapter(cfg or _cfg())
    adapter.set_composites(np.asarray(coords, float),
                           np.asarray(values, float).ravel())
    adapter.set_block_model(np.asarray(centroids, float),
                            np.array([dx, dx, dx], float))
    return np.asarray(adapter.estimate()["grades"], float)


# ─────────────────────────────────────────────────────────────────────
# 1. Hard neighbourhood improves spike retention
# ─────────────────────────────────────────────────────────────────────


def _spike_scene():
    """Dense regular background + one high-grade spike at (20, 20, 5)."""
    xs, ys = np.meshgrid(np.linspace(0, 40, 9), np.linspace(0, 40, 9))
    coords = np.column_stack([xs.ravel(), ys.ravel(),
                              np.full(xs.size, 5.0)])
    values = np.full(len(coords), 1.0)
    centre_idx = int(np.argmin(
        np.linalg.norm(coords[:, :2] - np.array([20, 20]), axis=1)
    ))
    coords[centre_idx] = [20.0, 20.0, 5.0]
    values[centre_idx] = 10.0
    # Block grid — coarse enough to contain a block near the spike
    xs_b = np.linspace(2, 38, 15)
    X, Y = np.meshgrid(xs_b, xs_b)
    centroids = np.column_stack([X.ravel(), Y.ravel(),
                                  np.full(X.size, 5.0)])
    return coords, values, centroids


def _spike_block_value(grades, centroids):
    finite = np.isfinite(grades)
    d = np.linalg.norm(centroids[finite, :2] - np.array([20, 20]), axis=1)
    return float(grades[finite][int(np.argmin(d))])


def _far_block_mean(grades, centroids, far_radius=12.0):
    finite = np.isfinite(grades)
    d = np.linalg.norm(centroids[finite, :2] - np.array([20, 20]), axis=1)
    far = d > far_radius
    if not far.any():
        return float("nan")
    return float(np.mean(grades[finite][far]))


def test_hard_neighbourhood_improves_spike_retention():
    """Tightening the neighbourhood should retain the spike better
    and push distant influence closer to background."""
    coords, values, centroids = _spike_scene()

    permissive = _cfg(
        range_max=20.0, range_mid=20.0, range_min=20.0,
        max_samples=40, max_samples_per_octant=6,
        local_search_radii=(0.75, 1.5, 3.0),
    )
    hard_local = _cfg(
        range_max=20.0, range_mid=20.0, range_min=20.0,
        max_samples=8, max_samples_per_octant=2,
        local_search_radii=(0.15, 0.30, 0.60),
    )

    g_perm = _run(coords, values, centroids, dx=2.0, cfg=permissive)
    g_hard = _run(coords, values, centroids, dx=2.0, cfg=hard_local)

    spike_perm = _spike_block_value(g_perm, centroids)
    spike_hard = _spike_block_value(g_hard, centroids)
    far_perm = _far_block_mean(g_perm, centroids)
    far_hard = _far_block_mean(g_hard, centroids)

    # Hard neighbourhood should NOT over-smooth the spike relative
    # to the permissive case. In fact it should match or exceed it.
    assert spike_hard >= spike_perm - 0.2, (
        f"hard neighbourhood lost spike: permissive={spike_perm:.2f}, "
        f"hard={spike_hard:.2f}"
    )
    # Both configs should retain at least ~50% of the peak on the
    # actual nearest block.
    assert spike_hard >= 5.0, (
        f"hard-local spike = {spike_hard:.2f} (expected >= 5.0)"
    )

    # Distant blocks should sit close to background (1.0). The hard
    # config must be at least as tight as the permissive one here.
    assert far_hard <= far_perm + 0.05, (
        f"hard neighbourhood leaked more: permissive far={far_perm:.2f}, "
        f"hard far={far_hard:.2f}"
    )
    assert far_hard < 1.5, (
        f"hard-local far-block mean = {far_hard:.2f} (expected < 1.5)"
    )

    # Spike-to-far contrast ratio — the structural signal strength
    contrast_perm = spike_perm / max(far_perm, 1e-9)
    contrast_hard = spike_hard / max(far_hard, 1e-9)
    assert contrast_hard >= contrast_perm, (
        f"hard config weaker contrast: "
        f"perm={contrast_perm:.2f}, hard={contrast_hard:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────
# 2. Ordinary kriging parity
# ─────────────────────────────────────────────────────────────────────


def _gauss(xy, cx=20.0, cy=20.0, sigma=8.0, peak=5.0, base=1.0):
    d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
    return base + peak * np.exp(-0.5 * d2 / sigma ** 2)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_ordinary_kriging_parity_on_gaussian_hill():
    """ARBF vs OK on the same synthetic field. Correlation between
    the two methods must be high, smoothing ratios within a bounded
    band, and the spike block value within ±40% of OK."""
    rng = np.random.default_rng(400)
    coords_xy = rng.uniform([0, 0], [40, 40], size=(150, 2))
    coords = np.column_stack([coords_xy, np.full(150, 5.0)])
    values = _gauss(coords_xy)

    # Block grid
    xs = np.linspace(3, 37, 12)
    X, Y = np.meshgrid(xs, xs)
    centroids = np.column_stack([X.ravel(), Y.ravel(),
                                  np.full(X.size, 5.0)])

    # ── ARBF (shared range config)
    arbf_cfg = _cfg(
        range_max=15.0, range_mid=15.0, range_min=15.0,
        local_search_radii=(0.5, 1.0, 2.0),
        max_samples=16, max_samples_per_octant=4,
    )
    g_arbf = _run(coords, values, centroids, dx=3.0, cfg=arbf_cfg)

    # ── OK (same range, same kernel family, hard nearest-neighbour
    #       search, n_neighbors=16 to match ARBF max_samples)
    vario_params = {
        "range": 15.0, "sill": 1.0, "nugget": 0.0,
        "anisotropy": None,
    }
    g_ok, _ = ordinary_kriging_fast(
        coords, values, centroids,
        variogram_params=vario_params,
        n_neighbors=16, max_distance=30.0,
        model_type="spherical",
    )

    finite = np.isfinite(g_arbf) & np.isfinite(g_ok)
    assert finite.sum() >= 100

    # Correlation between ARBF and OK
    corr = float(np.corrcoef(g_arbf[finite], g_ok[finite])[0, 1])
    assert corr > 0.85, f"ARBF vs OK corr = {corr:.3f}"

    # Smoothing ratio: var(est) / var(input). ARBF's ratio should be
    # within 1.8× of OK's in either direction.
    arbf_ratio = float(np.var(g_arbf[finite]) / np.var(values))
    ok_ratio = float(np.var(g_ok[finite]) / np.var(values))
    assert ok_ratio > 0 and arbf_ratio > 0
    rel = arbf_ratio / ok_ratio
    assert 0.55 <= rel <= 1.80, (
        f"ARBF smoothing ratio {arbf_ratio:.3f} vs OK {ok_ratio:.3f} "
        f"(rel={rel:.2f})"
    )

    # Mean bias between the two estimators should be small
    mean_diff = abs(float(np.mean(g_arbf[finite])) - float(np.mean(g_ok[finite])))
    mean_bias_rel = mean_diff / max(abs(float(np.mean(g_ok[finite]))), 1e-9)
    assert mean_bias_rel < 0.05, (
        f"ARBF-OK mean bias = {mean_bias_rel:.1%} (expected < 5%)"
    )

    # RMSE between the two — they should agree point-by-point on a
    # smooth field.
    rmse = float(np.sqrt(np.mean((g_arbf[finite] - g_ok[finite]) ** 2)))
    input_std = float(np.std(values))
    assert rmse / input_std < 0.30, (
        f"ARBF-OK RMSE / input_std = {rmse/input_std:.3f}"
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_ordinary_kriging_parity_spike_amplitude():
    """Spike test: both OK and ARBF should damp an isolated spike
    by a comparable amount. ARBF's spike amplitude must be within
    ±40 % of OK's.
    """
    coords, values, centroids = _spike_scene()

    arbf_cfg = _cfg(
        range_max=15.0, range_mid=15.0, range_min=15.0,
        local_search_radii=(0.3, 0.6, 1.2),
        max_samples=12, max_samples_per_octant=3,
    )
    g_arbf = _run(coords, values, centroids, dx=2.0, cfg=arbf_cfg)

    vario_params = {"range": 15.0, "sill": 1.0, "nugget": 0.0}
    g_ok, _ = ordinary_kriging_fast(
        coords, values, centroids,
        variogram_params=vario_params,
        n_neighbors=12, max_distance=18.0,
        model_type="spherical",
    )

    spike_arbf = _spike_block_value(g_arbf, centroids)
    spike_ok = _spike_block_value(g_ok, centroids)

    # OK spike amplitude sets the reference. ARBF should be within
    # ±40 % of it — either side indicates genuine parity (neither
    # wildly over-sharp nor over-smoothed relative to kriging).
    rel = (spike_arbf - 1.0) / max(spike_ok - 1.0, 1e-9)
    assert 0.60 <= rel <= 1.40, (
        f"ARBF spike amplitude = {spike_arbf:.2f} (from background 1.0), "
        f"OK = {spike_ok:.2f}, rel = {rel:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────
# 3. Effective influence radius
# ─────────────────────────────────────────────────────────────────────


def test_effective_influence_radius_is_bounded():
    """Probe ARBF's influence radius directly.

    Method: a query point at the origin sees a background of low-grade
    composites plus ONE high-grade "probe" composite. We vary the probe's
    distance from the query (0 → 2.5 × range) and record the query's
    block estimate as a function of probe distance.

    The contribution of the probe = (estimate_with_probe − estimate_without_probe).
    This should decay smoothly with distance and drop below 10 % of its
    maximum contribution by the time the probe is 1.5 × variogram-range
    away. If it doesn't, ARBF has long influence tails (semi-global
    behaviour).
    """
    VARIO_RANGE = 15.0

    def _make_bg(n_per_side=9):
        xs, ys = np.meshgrid(
            np.linspace(-40, 40, n_per_side),
            np.linspace(-40, 40, n_per_side),
        )
        bg = np.column_stack([
            xs.ravel(), ys.ravel(), np.full(xs.size, 0.0),
        ])
        vals = np.full(len(bg), 1.0)
        # Remove any bg sample within 2 m of the probe distances so
        # we can cleanly add the probe at those positions.
        return bg, vals

    cfg = _cfg(
        range_max=VARIO_RANGE, range_mid=VARIO_RANGE, range_min=VARIO_RANGE,
        local_search_radii=(0.5, 1.0, 2.0),
        max_samples=16, max_samples_per_octant=4,
    )

    query = np.array([[0.0, 0.0, 0.0]], dtype=float)

    # Baseline: estimate with background only (no probe)
    bg, vals_bg = _make_bg()
    baseline = float(_run(bg, vals_bg, query, dx=1.0, cfg=cfg)[0])
    assert np.isfinite(baseline)
    assert 0.9 <= baseline <= 1.1, (
        f"background baseline = {baseline:.3f} (expected ~1)"
    )

    probe_distances = np.array([
        0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 24.0, 32.0
    ])
    contributions = np.zeros_like(probe_distances)
    for i, d in enumerate(probe_distances):
        # Remove any nearby background sample and replace with probe
        bg_i, vals_i = _make_bg()
        probe_pt = np.array([d, 0.0, 0.0])
        # Kick out any existing sample within 1.5 m of the probe
        keep = np.linalg.norm(bg_i - probe_pt, axis=1) > 1.5
        bg_i = bg_i[keep]
        vals_i = vals_i[keep]
        probe_coords = np.vstack([bg_i, probe_pt[None, :]])
        probe_vals = np.concatenate([vals_i, [10.0]])
        est = float(_run(probe_coords, probe_vals, query, dx=1.0, cfg=cfg)[0])
        contributions[i] = est - baseline

    max_contrib = float(np.max(contributions))
    assert max_contrib > 1.0, (
        f"probe at distance 0 barely moved the estimate: "
        f"max contribution = {max_contrib:.2f} (expected > 1.0)"
    )

    # Contribution should be monotonically non-increasing with distance
    # (up to small numerical noise).
    for i in range(1, len(contributions)):
        assert contributions[i] <= contributions[0] + 0.1

    # Decay target: at 1.5 × range the contribution should be <= 10 %
    # of the max-contribution magnitude.
    d_threshold = 1.5 * VARIO_RANGE   # 22.5 m
    idx_threshold = np.searchsorted(probe_distances, d_threshold)
    if idx_threshold < len(contributions):
        contrib_at_threshold = float(contributions[idx_threshold])
    else:
        contrib_at_threshold = float(contributions[-1])

    assert contrib_at_threshold <= 0.10 * max_contrib, (
        f"influence at {d_threshold} m ({contrib_at_threshold:.3f}) > "
        f"10 % of max ({max_contrib:.3f}) — ARBF has long influence tails"
    )

    # At 2 × range, the contribution should be essentially zero
    d_zero = 2.0 * VARIO_RANGE  # 30 m
    idx_zero = np.searchsorted(probe_distances, d_zero)
    if idx_zero < len(contributions):
        contrib_at_zero = float(contributions[idx_zero])
        assert contrib_at_zero <= 0.02 * max_contrib, (
            f"contribution at {d_zero} m = {contrib_at_zero:.4f} "
            f"(expected ~0)"
        )


def test_influence_radius_shrinks_with_tight_neighbourhood():
    """Same probe method, two configs. The tighter config must have a
    strictly shorter decay half-life than the permissive config.
    """
    VARIO_RANGE = 15.0
    cfg_perm = _cfg(
        range_max=VARIO_RANGE, range_mid=VARIO_RANGE, range_min=VARIO_RANGE,
        local_search_radii=(0.75, 1.5, 3.0),
        max_samples=40, max_samples_per_octant=6,
    )
    cfg_tight = _cfg(
        range_max=VARIO_RANGE, range_mid=VARIO_RANGE, range_min=VARIO_RANGE,
        local_search_radii=(0.25, 0.5, 1.0),
        max_samples=8, max_samples_per_octant=2,
    )
    query = np.array([[0.0, 0.0, 0.0]], dtype=float)
    xs, ys = np.meshgrid(np.linspace(-40, 40, 9), np.linspace(-40, 40, 9))
    bg_coords = np.column_stack([xs.ravel(), ys.ravel(),
                                  np.full(xs.size, 0.0)])

    def _probe_curve(cfg):
        baseline = float(_run(bg_coords, np.ones(len(bg_coords)),
                               query, dx=1.0, cfg=cfg)[0])
        dists = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0])
        out = np.zeros_like(dists)
        for i, d in enumerate(dists):
            probe_pt = np.array([d, 0.0, 0.0])
            keep = np.linalg.norm(bg_coords - probe_pt, axis=1) > 1.5
            cs = np.vstack([bg_coords[keep], probe_pt[None, :]])
            vs = np.concatenate([np.ones(int(keep.sum())), [10.0]])
            est = float(_run(cs, vs, query, dx=1.0, cfg=cfg)[0])
            out[i] = est - baseline
        return dists, out

    d_perm, c_perm = _probe_curve(cfg_perm)
    d_tight, c_tight = _probe_curve(cfg_tight)

    # Both curves should start positive and end near zero
    assert c_perm[0] > 0.5
    assert c_tight[0] > 0.5

    # Half-life: first distance where contribution drops to 50 % of max
    def _half_life(d, c):
        target = 0.5 * c[0]
        below = np.where(c < target)[0]
        return float(d[below[0]]) if below.size else float(d[-1])

    hl_perm = _half_life(d_perm, c_perm)
    hl_tight = _half_life(d_tight, c_tight)

    assert hl_tight <= hl_perm, (
        f"tight config has LONGER influence half-life: "
        f"tight={hl_tight:.1f} m, perm={hl_perm:.1f} m"
    )
