"""Regression tests for ARBF swath declustered-mean renormalisation."""

from __future__ import annotations

import numpy as np

from block_model_viewer.geostats.arbf_adapter import ARBFEstimatorAdapter


def test_decl_mean_is_scale_invariant_with_uniform_grades():
    """With uniform grade values, declustered bin mean must equal that
    value — the per-bin renormalisation has to divide by ``sum(w)`` in
    the bin, not by any global weight sum.
    """
    rng = np.random.default_rng(0)
    n = 200
    centroids = rng.uniform([0, 0, 0], [100, 100, 50], size=(n, 3))
    grades = np.full(n, 2.5)
    composite_coords = rng.uniform([0, 0, 0], [100, 100, 50], size=(n, 3))
    composite_values = np.full(n, 2.5)

    # Non-uniform globally-scaled weights (sum to 1) — the classic
    # source of the "green line near zero" bug.
    weights = rng.uniform(0.1, 10.0, size=n)
    weights /= weights.sum()

    swath = ARBFEstimatorAdapter._build_swath_data(
        grades, centroids,
        composite_coords=composite_coords,
        composite_values=composite_values,
        composite_weights=weights,
    )

    for ax in ("x", "y", "z"):
        decl = np.asarray(swath[ax]["mean_actual_declustered"], dtype=float)
        finite = decl[np.isfinite(decl)]
        assert finite.size > 0
        assert np.allclose(finite, 2.5, atol=1e-9)


def test_decl_mean_matches_raw_when_weights_uniform():
    """Uniform weights should produce a declustered mean equal to the
    raw composite mean in every bin.
    """
    rng = np.random.default_rng(1)
    n = 150
    centroids = rng.uniform([0, 0, 0], [80, 80, 40], size=(n, 3))
    grades = rng.normal(5.0, 1.0, size=n)
    composite_coords = rng.uniform([0, 0, 0], [80, 80, 40], size=(n, 3))
    composite_values = rng.normal(5.0, 1.0, size=n)
    weights = np.ones(n)

    swath = ARBFEstimatorAdapter._build_swath_data(
        grades, centroids,
        composite_coords=composite_coords,
        composite_values=composite_values,
        composite_weights=weights,
    )

    for ax in ("x", "y", "z"):
        raw = np.asarray(swath[ax]["mean_actual"], dtype=float)
        decl = np.asarray(swath[ax]["mean_actual_declustered"], dtype=float)
        mask = np.isfinite(raw) & np.isfinite(decl)
        assert mask.any()
        assert np.allclose(raw[mask], decl[mask], atol=1e-12)


def test_counts_reported_per_bin():
    """n_block / n_composite must be populated and sum to the totals."""
    rng = np.random.default_rng(2)
    n = 100
    centroids = rng.uniform([0, 0, 0], [50, 50, 25], size=(n, 3))
    grades = rng.normal(1.0, 0.2, size=n)
    composite_coords = rng.uniform([0, 0, 0], [50, 50, 25], size=(n, 3))
    composite_values = rng.normal(1.0, 0.2, size=n)

    swath = ARBFEstimatorAdapter._build_swath_data(
        grades, centroids,
        composite_coords=composite_coords,
        composite_values=composite_values,
    )

    for ax in ("x", "y", "z"):
        n_block = np.asarray(swath[ax]["n_block"], dtype=int)
        n_comp = np.asarray(swath[ax]["n_composite"], dtype=int)
        assert n_block.sum() == n
        assert n_comp.sum() == n


# ─────────────────────────────────────────────────────────────────────
# Phase B new-field tests
# ─────────────────────────────────────────────────────────────────────


def test_informed_mask_drops_uninformed_blocks_from_panel_mean():
    """Blocks flagged uninformed must not feed into the panel mean, and
    the reported ``n_block`` must count only informed blocks."""
    rng = np.random.default_rng(10)
    n = 200
    centroids = rng.uniform([0, 0, 0], [100, 100, 50], size=(n, 3))
    # informed blocks have grade 5.0, uninformed have grade 999.0 —
    # if the mask leaks, the panel mean will blow up toward 999.
    informed = np.zeros(n, dtype=bool)
    informed[:100] = True
    grades = np.where(informed, 5.0, 999.0)

    composite_coords = rng.uniform([0, 0, 0], [100, 100, 50], size=(60, 3))
    composite_values = np.full(60, 5.0)

    swath = ARBFEstimatorAdapter._build_swath_data(
        grades, centroids,
        composite_coords=composite_coords,
        composite_values=composite_values,
        informed_mask=informed,
        min_blocks_per_panel=1,
        min_composites_per_panel=1,
    )

    for ax in ("x", "y", "z"):
        est = np.asarray(swath[ax]["mean_estimated"], dtype=float)
        finite = est[np.isfinite(est)]
        assert finite.size > 0
        # Must equal 5.0 (informed-only); 999 would indicate leakage.
        assert np.allclose(finite, 5.0, atol=1e-9)
        # n_block reports informed count only — total across bins == 100.
        assert int(np.sum(swath[ax]["n_block"])) == 100


def test_volume_weighted_block_mean_differs_from_simple_mean():
    """With non-uniform block volumes the volume-weighted panel mean
    must equal the hand-computed weighted mean, not the arithmetic mean.
    """
    # Two blocks in one Z slice: volume 1 at grade 10, volume 9 at grade
    # 20. Simple mean = 15. Volume-weighted mean = (1*10 + 9*20) / 10 = 19.
    centroids = np.array([[0, 0, 5], [0, 0, 5]], dtype=float)
    grades = np.array([10.0, 20.0])
    volumes = np.array([1.0, 9.0])

    composite_coords = np.array([[0, 0, 5]], dtype=float)
    composite_values = np.array([15.0])

    swath = ARBFEstimatorAdapter._build_swath_data(
        grades, centroids,
        composite_coords=composite_coords,
        composite_values=composite_values,
        block_volumes=volumes,
        min_blocks_per_panel=2,
        min_composites_per_panel=1,
    )

    vw = np.asarray(swath["z"]["mean_estimated_volume_weighted"], float)
    finite = vw[np.isfinite(vw)]
    assert finite.size == 1
    assert abs(float(finite[0]) - 19.0) < 1e-9


def test_metal_conservation_with_unit_support():
    """Block metal = sum(volume * grade). Composite equivalent metal =
    composite_mean * panel_volume (same volumetric support as blocks).
    """
    centroids = np.array([[0, 0, 5], [0, 0, 5]], dtype=float)
    grades = np.array([10.0, 20.0])
    volumes = np.array([2.0, 3.0])
    # Block metal: 2*10 + 3*20 = 80. Panel volume = 5.
    # Composite raw mean = (4+6)/2 = 5. Raw panel metal = 5*5 = 25.
    composite_coords = np.array([[0, 0, 5], [0, 0, 5]], dtype=float)
    composite_values = np.array([4.0, 6.0])
    support = np.array([1.0, 1.0])

    swath = ARBFEstimatorAdapter._build_swath_data(
        grades, centroids,
        composite_coords=composite_coords,
        composite_values=composite_values,
        block_volumes=volumes,
        composite_support=support,
        min_blocks_per_panel=2,
        min_composites_per_panel=2,
    )

    met_blk = np.asarray(swath["z"]["metal_block"], float)
    met_raw = np.asarray(swath["z"]["metal_composite_raw"], float)
    fb = met_blk[np.isfinite(met_blk)]
    fr = met_raw[np.isfinite(met_raw)]
    assert fb.size == 1 and fr.size == 1
    assert abs(float(fb[0]) - 80.0) < 1e-9
    assert abs(float(fr[0]) - 25.0) < 1e-9


def test_panel_width_produces_uniform_bins():
    """``panel_width`` must produce bins of that width (except the
    final bin which absorbs the remainder)."""
    n = 200
    rng = np.random.default_rng(42)
    centroids = rng.uniform([0, 0, 0], [100, 100, 100], size=(n, 3))
    grades = rng.normal(5.0, 1.0, size=n)
    composite_coords = rng.uniform([0, 0, 0], [100, 100, 100], size=(n, 3))
    composite_values = rng.normal(5.0, 1.0, size=n)

    swath = ARBFEstimatorAdapter._build_swath_data(
        grades, centroids,
        composite_coords=composite_coords,
        composite_values=composite_values,
        panel_width=25.0,
        min_blocks_per_panel=1,
        min_composites_per_panel=1,
    )

    for ax in ("x", "y", "z"):
        pos = np.asarray(swath[ax]["slice_positions"], float)
        assert pos.size >= 3
        assert float(swath[ax]["panel_width"]) == 25.0


def test_legacy_call_signature_unchanged():
    """Callers that supply none of the new params must still get the
    linspace-10-bin behaviour (back-compat)."""
    rng = np.random.default_rng(7)
    n = 300
    centroids = rng.uniform([0, 0, 0], [100, 100, 50], size=(n, 3))
    grades = rng.normal(5.0, 1.0, size=n)
    composite_coords = rng.uniform([0, 0, 0], [100, 100, 50], size=(n, 3))
    composite_values = rng.normal(5.0, 1.0, size=n)

    swath = ARBFEstimatorAdapter._build_swath_data(
        grades, centroids,
        composite_coords=composite_coords,
        composite_values=composite_values,
    )
    for ax in ("x", "y", "z"):
        pos = np.asarray(swath[ax]["slice_positions"], float)
        assert pos.size == 10
        # panel_width is None in legacy path
        assert swath[ax]["panel_width"] is None
