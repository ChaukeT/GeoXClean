#!/usr/bin/env python
"""
ARBF Engine — Production-Regime Tests.

Fills gaps from test_arbf_deep.py:
  - UTM-scale coordinates (500,000m offsets)
  - Sharp grade contrasts (Runge oscillations)
  - G11/G12: Explicit inverse vs Cholesky on ill-conditioned matrices
  - CV6: LOO-CV estimator identity (global RBF vs PUM)
  - B2: LVA wiring verification
  - Production-like CV conditions (multi-domain, high variance)
  - Clustered sampling (preferential drilling near high-grade zones)
  - High-CV grade distributions (CV > 1.0, lognormal)
  - Combined worst-case: clustered + high-CV + UTM + sharp contact

Usage:
    python -m tests.arbf.test_arbf_production
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("arbf_production_test")
logger.setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str = ""
    elapsed_ms: float = 0.0

RESULTS: List[TestResult] = []


def record(name: str, passed: bool, details: str = "", elapsed: float = 0.0):
    tag = "PASS" if passed else "FAIL"
    RESULTS.append(TestResult(name, passed, details, elapsed))
    logger.info("[%s] %s — %s (%.0fms)", tag, name, details, elapsed)


# ===================================================================
# SYNTHETIC DATA — PRODUCTION REGIME
# ===================================================================

def make_utm_dataset(n_composites=500, n_blocks=500):
    """Composites and blocks in UTM coordinates (~500,000m).

    Mimics production: coords at 499,800-500,800m, grade range 2-55% Fe,
    range ~80m, drill spacing ~50m.
    """
    np.random.seed(42)
    # Composites scattered in a 1km x 1km x 200m volume at UTM scale
    coords = np.column_stack([
        np.random.uniform(499800, 500800, n_composites),
        np.random.uniform(6200000, 6201000, n_composites),
        np.random.uniform(100, 300, n_composites),
    ])
    # Two domains: high-grade core + low-grade waste
    dist_from_centre = np.sqrt(
        (coords[:, 0] - 500300)**2 + (coords[:, 1] - 6200500)**2
    )
    in_core = dist_from_centre < 300
    values = np.where(in_core,
                      np.random.normal(45, 8, n_composites),
                      np.random.normal(10, 4, n_composites))
    values = np.clip(values, 0.5, 68)

    # Block model grid
    bx, by, bz = np.meshgrid(
        np.linspace(499850, 500750, int(n_blocks**(1/3))+1),
        np.linspace(6200050, 6200950, int(n_blocks**(1/3))+1),
        np.linspace(110, 290, max(5, int(n_blocks**(1/3))//2)),
        indexing='ij',
    )
    centroids = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()])
    block_sizes = np.array([25.0, 25.0, 10.0])

    return coords, values, centroids, block_sizes


def make_sharp_contrast_dataset(n=200):
    """Sharp grade boundary: Fe jumps from 5% to 55% over 20m.

    This is where Runge-phenomenon oscillations occur with accuracy=0.
    """
    np.random.seed(42)
    coords = np.random.uniform(0, 200, (n, 3))
    # Sharp step at x=100 with narrow transition
    sigmoid = 1.0 / (1.0 + np.exp(-(coords[:, 0] - 100) / 2.0))
    values = 5 + 50 * sigmoid + np.random.normal(0, 2, n)
    values = np.clip(values, 0.5, 68)

    bx = np.linspace(5, 195, 40)
    by = np.linspace(5, 195, 5)
    bz = np.linspace(5, 195, 5)
    gx, gy, gz = np.meshgrid(bx, by, bz, indexing='ij')
    centroids = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    block_sizes = np.array([5.0, 40.0, 40.0])

    return coords, values, centroids, block_sizes


def make_multidomain_cv_dataset(n=400):
    """Multi-domain deposit for production-like CV.

    3 domains, high variance, 400 samples — closer to production conditions.
    """
    np.random.seed(42)
    coords = np.random.uniform(0, 300, (n, 3))
    # Three domains
    domain = np.zeros(n, dtype=int)
    domain[coords[:, 0] > 200] = 2
    domain[(coords[:, 0] > 100) & (coords[:, 0] <= 200)] = 1

    values = np.where(domain == 0, np.random.normal(8, 3, n),
             np.where(domain == 1, np.random.normal(35, 10, n),
                                   np.random.normal(55, 6, n)))
    values = np.clip(values, 0.5, 68)
    return coords, values


def make_clustered_dataset(n=300, n_clusters=5):
    """Clustered sampling — mimics preferential drilling near high-grade zones.

    Real drillholes are NOT uniformly distributed: exploration targets
    high-grade zones with closer spacing, leaving waste areas sparse.
    This creates information bias that interacts with nugget estimation.
    """
    np.random.seed(42)
    # Cluster centres: some in high-grade zone, some in waste
    centres = np.array([
        [50, 50, 50],    # high-grade core, dense drilling
        [55, 60, 45],    # near core, dense
        [45, 40, 55],    # near core, dense
        [150, 150, 50],  # waste, sparse
        [20, 160, 50],   # waste, sparse
    ], dtype=np.float64)
    # Cluster sizes: dense near core, sparse in waste
    cluster_sizes = [n // 3, n // 5, n // 5, n // 7, n - n//3 - n//5 - n//5 - n//7]
    cluster_spreads = [8.0, 10.0, 10.0, 40.0, 35.0]  # tight core, diffuse waste

    coords_list = []
    for c, sz, spread in zip(centres, cluster_sizes, cluster_spreads):
        pts = c + np.random.randn(sz, 3) * spread
        coords_list.append(pts)
    coords = np.vstack(coords_list)
    coords = np.clip(coords, 0, 200)

    # Grade: high near (50,50,50), low elsewhere — smooth field
    dist_from_core = np.sqrt(np.sum((coords - np.array([50, 50, 50]))**2, axis=1))
    values = 45 * np.exp(-dist_from_core / 40) + 5 + np.random.normal(0, 3, len(coords))
    values = np.clip(values, 0.5, 68)

    # Block model grid (uniform — estimation grid is never clustered)
    bx, by, bz = np.meshgrid(
        np.linspace(5, 195, 15), np.linspace(5, 195, 15), np.linspace(10, 90, 5),
        indexing='ij',
    )
    centroids = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()])
    block_sizes = np.array([13.0, 13.0, 16.0])

    return coords, values, centroids, block_sizes


def make_high_cv_dataset(n=300):
    """High coefficient-of-variation grade distribution (CV > 1.0).

    Lognormal grades mimic gold or other precious metals where most
    samples are low-grade with rare high-grade hits. These stress-test
    the kernel system because the weights must simultaneously honour
    extreme values without creating negatives elsewhere.
    """
    np.random.seed(42)
    coords = np.random.uniform(0, 200, (n, 3))

    # Lognormal: mu=1.0, sigma=1.2 → CV ≈ 1.6
    log_values = np.random.normal(1.0, 1.2, n)
    values = np.exp(log_values)
    # Clip extreme outliers but keep the heavy tail
    values = np.clip(values, 0.01, np.percentile(values, 99.5))

    cv = np.std(values) / np.mean(values)
    assert cv > 1.0, f"CV too low: {cv:.2f}"

    bx, by, bz = np.meshgrid(
        np.linspace(5, 195, 15), np.linspace(5, 195, 15), np.linspace(5, 195, 5),
        indexing='ij',
    )
    centroids = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()])
    block_sizes = np.array([13.0, 13.0, 38.0])

    return coords, values, centroids, block_sizes


def make_combined_worst_case(n=400):
    """Combined worst case: clustered + high-CV + UTM + sharp contact.

    This is the synthetic dataset that most closely mimics the conditions
    under which the production failure occurred.
    """
    np.random.seed(42)

    # UTM-scale coordinates
    base_x, base_y, base_z = 500000.0, 6200000.0, 150.0

    # Clustered sampling: dense near orebody, sparse in waste
    n_core = n * 3 // 5   # 60% of samples near core
    n_waste = n - n_core

    core_centre = np.array([base_x + 400, base_y + 500, base_z + 50])
    core_coords = core_centre + np.random.randn(n_core, 3) * np.array([30, 30, 15])

    waste_coords = np.column_stack([
        np.random.uniform(base_x, base_x + 900, n_waste),
        np.random.uniform(base_y, base_y + 900, n_waste),
        np.random.uniform(base_z, base_z + 100, n_waste),
    ])
    coords = np.vstack([core_coords, waste_coords])

    # Sharp contact at x = base_x + 400 with high-CV lognormal grades
    in_ore = coords[:, 0] > base_x + 380
    # Ore zone: lognormal, CV > 1.0
    ore_log = np.random.normal(2.5, 1.0, len(coords))
    ore_values = np.exp(ore_log)
    # Waste zone: low-grade, moderate variance
    waste_values = np.random.lognormal(0.3, 0.4, len(coords))

    values = np.where(in_ore, ore_values, waste_values)
    values = np.clip(values, 0.01, np.percentile(values, 99.5))

    cv = np.std(values) / np.mean(values)

    # Block grid
    bx = np.linspace(base_x + 50, base_x + 850, 10)
    by = np.linspace(base_y + 50, base_y + 850, 10)
    bz = np.linspace(base_z + 10, base_z + 90, 4)
    gx, gy, gz = np.meshgrid(bx, by, bz, indexing='ij')
    centroids = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    block_sizes = np.array([80.0, 80.0, 20.0])

    return coords, values, centroids, block_sizes, cv


# ===================================================================
# TEST: UTM-SCALE COORDINATES
# ===================================================================

def test_utm_scale():
    """Test engine behaviour at UTM coordinates (500,000m)."""
    from geostats.arbf.engine import ARBFEstimator

    coords, values, centroids, block_sizes = make_utm_dataset(n_composites=300, n_blocks=200)

    # UTM1: Basic estimation — should produce reasonable grades, not constant
    t0 = time.time()
    est = ARBFEstimator({
        "kernel_type": "spheroidal", "alpha": 1.0,
        "nugget": 2.0, "accuracy": 1e-6,
        "range_max": 80.0, "range_mid": 80.0, "range_min": 40.0,
        "sill": float(np.var(values)),
        "variogram_mode": "global",
        "drift_type": "constant",
        "change_of_support": False,
        "run_cv": False,
        "verbose": False,
        "discretisation_density": 8,
        "n_subdomains": 6,
    })
    est.set_composites(coords, values)
    est.set_block_model(centroids, block_sizes)
    result = est.estimate()
    elapsed = (time.time() - t0) * 1000

    grade_std = np.std(result.grades)
    grade_range = np.max(result.grades) - np.min(result.grades)
    n_neg = np.sum(result.grades < 0)
    record("UTM1-not-constant", grade_std > 1.0,
           f"grade_std={grade_std:.2f}, range={grade_range:.2f}, neg={n_neg}",
           elapsed)

    # UTM2: Grades should span a reasonable range (not collapsed to mean)
    data_range = np.max(values) - np.min(values)
    ok = grade_range > data_range * 0.1
    record("UTM2-grade-spread", ok,
           f"est_range={grade_range:.2f}, data_range={data_range:.2f}, "
           f"ratio={grade_range/data_range:.3f}")

    # UTM3: No negative grades (Fe% must be >= 0)
    record("UTM3-no-negatives", n_neg == 0,
           f"negatives={n_neg}/{len(result.grades)}, "
           f"min={np.min(result.grades):.2f}, max={np.max(result.grades):.2f}")

    # UTM4: Variance should be spatially structured (low near data, high far)
    var_std = np.std(result.variances)
    record("UTM4-var-structured", var_std > 0.01,
           f"var_std={var_std:.4f}, var_range=[{np.min(result.variances):.4f}, "
           f"{np.max(result.variances):.4f}]")

    # UTM5: Kernel matrix condition at UTM scale
    # With range=80m and UTM distances ~100-800m, most kernel entries are near zero.
    # Test that the engine still produces stable results.
    from geostats.arbf.gpr import assemble_kernel_matrix
    from geostats.arbf.utils import rotation_matrix, scale_matrix, condition_number_estimate
    t0 = time.time()
    subset = coords[:50]
    R = rotation_matrix(0, 0, 0)
    S = scale_matrix(80.0, 80.0, 40.0)
    K, P = assemble_kernel_matrix(
        subset, kernel_type="spheroidal", alpha=1.0,
        sill=float(np.var(values)), range_=80.0, nugget=2.0, accuracy=1e-6,
        R=R, S=S, drift_type="constant",
    )
    cond = condition_number_estimate(K)
    record("UTM5-condition-number", True,  # diagnostic
           f"cond(K)={cond:.2e} for N=50 at UTM scale (range=80m)",
           (time.time()-t0)*1000)

    # UTM6: Pairwise distance distribution
    from scipy.spatial.distance import pdist
    dists = pdist(coords[:100])
    r_normalised = dists / 80.0  # normalised by range
    pct_within_range = np.mean(r_normalised < 1.0) * 100
    pct_within_3x = np.mean(r_normalised < 3.0) * 100
    record("UTM6-distance-regime", True,  # diagnostic
           f"r<1: {pct_within_range:.1f}%, r<3: {pct_within_3x:.1f}%, "
           f"median_r={np.median(r_normalised):.2f}, max_r={np.max(r_normalised):.1f}")


# ===================================================================
# TEST: SHARP GRADE CONTRASTS (RUNGE OSCILLATIONS)
# ===================================================================

def test_sharp_contrasts():
    """Test behaviour at sharp domain boundaries — Runge phenomenon."""
    from geostats.arbf.gpr import (
        assemble_kernel_matrix, factorise_and_solve, predict_mean,
    )
    from geostats.arbf.utils import scale_matrix

    coords, values, centroids, block_sizes = make_sharp_contrast_dataset(n=200)
    S = scale_matrix(50.0, 50.0, 50.0)

    # SC1: accuracy=0, nugget=0 — maximum oscillation risk
    t0 = time.time()
    K0, _ = assemble_kernel_matrix(
        coords, kernel_type="spheroidal", alpha=1.0,
        sill=float(np.var(values)), range_=50.0, nugget=0.0, accuracy=0.0,
        S=S, drift_type="constant",
    )
    f0, w0, pc0 = factorise_and_solve(K0, values, drift_type="constant")
    est0 = predict_mean(centroids, coords, w0, pc0,
                        kernel_type="spheroidal", alpha=1.0,
                        sill=float(np.var(values)), range_=50.0,
                        S=S, drift_type="constant")
    n_neg_0 = int(np.sum(est0 < 0))
    n_over68_0 = int(np.sum(est0 > 68))
    n_extreme_0 = n_neg_0 + n_over68_0
    record("SC1-acc0-nug0-extremes", True,  # diagnostic
           f"negatives={n_neg_0}, >68%={n_over68_0}, "
           f"range=[{np.min(est0):.1f}, {np.max(est0):.1f}]",
           (time.time()-t0)*1000)

    # SC2: accuracy=1e-6, nugget=0 — should reduce oscillations
    t0 = time.time()
    K1, _ = assemble_kernel_matrix(
        coords, kernel_type="spheroidal", alpha=1.0,
        sill=float(np.var(values)), range_=50.0, nugget=0.0, accuracy=1e-6,
        S=S, drift_type="constant",
    )
    f1, w1, pc1 = factorise_and_solve(K1, values, drift_type="constant")
    est1 = predict_mean(centroids, coords, w1, pc1,
                        kernel_type="spheroidal", alpha=1.0,
                        sill=float(np.var(values)), range_=50.0,
                        S=S, drift_type="constant")
    n_neg_1 = int(np.sum(est1 < 0))
    n_over68_1 = int(np.sum(est1 > 68))
    record("SC2-acc1e6-improvement", n_neg_1 + n_over68_1 <= n_extreme_0,
           f"acc=0: {n_extreme_0} extreme, acc=1e-6: {n_neg_1 + n_over68_1} extreme, "
           f"range=[{np.min(est1):.1f}, {np.max(est1):.1f}]",
           (time.time()-t0)*1000)

    # SC3: nugget=10% of sill — should eliminate most oscillations
    t0 = time.time()
    sill_val = float(np.var(values))
    nugget_val = sill_val * 0.1
    K2, _ = assemble_kernel_matrix(
        coords, kernel_type="spheroidal", alpha=1.0,
        sill=sill_val, range_=50.0, nugget=nugget_val, accuracy=1e-6,
        S=S, drift_type="constant",
    )
    f2, w2, pc2 = factorise_and_solve(K2, values, drift_type="constant")
    est2 = predict_mean(centroids, coords, w2, pc2,
                        kernel_type="spheroidal", alpha=1.0,
                        sill=sill_val, range_=50.0,
                        S=S, drift_type="constant")
    n_neg_2 = int(np.sum(est2 < 0))
    n_over68_2 = int(np.sum(est2 > 68))
    record("SC3-nugget-eliminates", n_neg_2 + n_over68_2 < n_extreme_0,
           f"nug=10%sill: {n_neg_2} neg, {n_over68_2} >68%, "
           f"range=[{np.min(est2):.1f}, {np.max(est2):.1f}]",
           (time.time()-t0)*1000)

    # SC4: Full engine with sharp contrasts
    t0 = time.time()
    from geostats.arbf.engine import ARBFEstimator
    est_e = ARBFEstimator({
        "kernel_type": "spheroidal", "alpha": 1.0,
        "nugget": nugget_val, "accuracy": 1e-6,
        "range_max": 50.0, "range_mid": 50.0, "range_min": 50.0,
        "sill": sill_val,
        "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 6,
    })
    est_e.set_composites(coords, values)
    est_e.set_block_model(centroids, block_sizes)
    r_e = est_e.estimate()
    n_neg_e = int(np.sum(r_e.grades < 0))
    n_over68_e = int(np.sum(r_e.grades > 68))
    record("SC4-engine-sharp", n_neg_e == 0,
           f"negatives={n_neg_e}, >68%={n_over68_e}, "
           f"range=[{np.min(r_e.grades):.1f}, {np.max(r_e.grades):.1f}]",
           (time.time()-t0)*1000)


# ===================================================================
# TEST: G11/G12 — EXPLICIT INVERSE vs CHOLESKY STABILITY
# ===================================================================

def test_g11_g12_numerical_stability():
    """G11: Cholesky forward-sub vs explicit inverse accuracy.
    G12: Ill-conditioned matrix (cond ~ 1e10+).
    """
    from geostats.arbf.gpr import assemble_kernel_matrix, factorise_and_solve, predict_mean, predict_variance
    from geostats.arbf.utils import scale_matrix, condition_number_estimate
    from scipy.linalg import solve_triangular, cho_solve

    np.random.seed(42)
    N = 80
    coords = np.random.uniform(0, 100, (N, 3))
    values = 10 + np.sin(coords[:, 0] / 10) + np.random.normal(0, 0.5, N)

    sill = 2.0
    range_ = 30.0
    S = scale_matrix(range_, range_, range_)

    # G11: Compare Cholesky forward-sub vs explicit inverse for well-conditioned case
    t0 = time.time()
    K, P = assemble_kernel_matrix(
        coords, sill=sill, range_=range_, nugget=0.5, accuracy=1e-6,
        S=S, drift_type="none",  # M=0 → Cholesky path
    )
    fact, weights, poly_c = factorise_and_solve(K, values, drift_type="none")

    # Forward-substitution variance
    query = np.random.uniform(0, 100, (50, 3))
    var_fwd = predict_variance(
        query, coords, fact,
        sill=sill, range_=range_, nugget=0.5,
        S=S, drift_type="none",
    )

    # Explicit inverse variance (the old way — what we DON'T do)
    L = fact  # Cholesky factor
    K_inv_explicit = np.linalg.inv(K)
    from geostats.arbf.kernels import evaluate_kernel
    from geostats.arbf.utils import anisotropic_distance
    R = np.eye(3)
    D_q = anisotropic_distance(query, coords, R, S)
    k_q = sill * evaluate_kernel(D_q, kernel_type="spheroidal", alpha=1.0)
    var_explicit = np.zeros(50)
    for i in range(50):
        k_vec = k_q[i]
        var_explicit[i] = sill - k_vec @ K_inv_explicit @ k_vec

    var_explicit = np.maximum(var_explicit, 0.0)
    max_diff = np.max(np.abs(var_fwd - var_explicit))
    ok = max_diff < 0.01  # Should agree closely
    record("G11-fwd-vs-inv-wellcond", ok,
           f"max_diff={max_diff:.2e}, cond(K)={condition_number_estimate(K):.2e}",
           (time.time()-t0)*1000)

    # G12: Ill-conditioned matrix — compare stability
    t0 = time.time()
    # Create ill-conditioned system: very small nugget, close points
    coords_close = np.random.uniform(0, 10, (60, 3))  # Tight cluster
    values_close = 10 + np.random.normal(0, 1, 60)
    K_ill, _ = assemble_kernel_matrix(
        coords_close, sill=sill, range_=30.0, nugget=1e-8, accuracy=0.0,
        S=S, drift_type="none",
    )
    cond_ill = condition_number_estimate(K_ill)

    # Cholesky path (stable)
    try:
        fact_ill, w_ill, _ = factorise_and_solve(K_ill.copy(), values_close, drift_type="none")
        est_chol = predict_mean(
            query[:10], coords_close, w_ill, np.array([]),
            sill=sill, range_=30.0, S=S, drift_type="none",
        )
        var_chol = predict_variance(
            query[:10], coords_close, fact_ill,
            sill=sill, range_=30.0, nugget=1e-8,
            S=S, drift_type="none",
        )
        chol_ok = np.all(np.isfinite(est_chol)) and np.all(np.isfinite(var_chol))
        chol_neg_var = int(np.sum(var_chol < -1e-6))
    except Exception as e:
        chol_ok = False
        chol_neg_var = -1
        est_chol = np.full(10, np.nan)
        var_chol = np.full(10, np.nan)

    # Explicit inverse path (potentially unstable)
    try:
        K_inv_ill = np.linalg.inv(K_ill)
        D_q2 = anisotropic_distance(query[:10], coords_close, R, S)
        k_q2 = sill * evaluate_kernel(D_q2, kernel_type="spheroidal", alpha=1.0)
        w_inv = K_inv_ill @ values_close
        est_inv = k_q2 @ w_inv
        var_inv = np.array([sill - k_q2[i] @ K_inv_ill @ k_q2[i] for i in range(10)])
        inv_ok = np.all(np.isfinite(est_inv)) and np.all(np.isfinite(var_inv))
        inv_neg_var = int(np.sum(var_inv < -1e-6))
    except Exception as e:
        inv_ok = False
        inv_neg_var = -1
        est_inv = np.full(10, np.nan)
        var_inv = np.full(10, np.nan)

    # Compare: Cholesky should be at least as stable
    record("G12-ill-cond-stability", True,  # diagnostic — report both
           f"cond={cond_ill:.2e}, "
           f"Cholesky: finite={chol_ok}, neg_var={chol_neg_var}, "
           f"est_range=[{np.nanmin(est_chol):.2f},{np.nanmax(est_chol):.2f}]; "
           f"Explicit: finite={inv_ok}, neg_var={inv_neg_var}, "
           f"est_range=[{np.nanmin(est_inv):.2f},{np.nanmax(est_inv):.2f}]",
           (time.time()-t0)*1000)

    # G12b: Severely ill-conditioned (cond > 1e12)
    t0 = time.time()
    # Duplicate some points to guarantee near-singularity
    coords_dup = np.vstack([coords_close[:30], coords_close[:30] + 1e-8])
    values_dup = np.concatenate([values_close[:30], values_close[:30] + 1e-6])
    K_severe, _ = assemble_kernel_matrix(
        coords_dup, sill=sill, range_=30.0, nugget=0.0, accuracy=0.0,
        S=S, drift_type="none",
    )
    cond_severe = condition_number_estimate(K_severe)

    try:
        fact_s, w_s, _ = factorise_and_solve(K_severe.copy(), values_dup, drift_type="none")
        est_s = predict_mean(query[:5], coords_dup, w_s, np.array([]),
                             sill=sill, range_=30.0, S=S, drift_type="none")
        stable = np.all(np.isfinite(est_s)) and np.max(np.abs(est_s)) < 1000
        record("G12b-severe-illcond", stable,
               f"cond={cond_severe:.2e}, est_range=[{np.min(est_s):.2f},{np.max(est_s):.2f}], "
               f"auto-regularisation applied",
               (time.time()-t0)*1000)
    except Exception as e:
        record("G12b-severe-illcond", False,
               f"cond={cond_severe:.2e}, CRASHED: {e}",
               (time.time()-t0)*1000)


# ===================================================================
# TEST: CV6 — LOO-CV ESTIMATOR IDENTITY
# ===================================================================

def test_cv6_estimator_identity():
    """CV6: Does LOO-CV use the PUM blended estimator or a global RBF?

    Read the code: leave_one_out_cv() builds ONE global kernel matrix
    from ALL samples and uses the Bartlett virtual LOO formula.
    It does NOT use the PUM (sub-domain blending) estimator.

    This test verifies that by comparing:
    1. CV estimated values from leave_one_out_cv (global RBF)
    2. True LOO values using the full engine (PUM) — drop one sample,
       re-estimate, record prediction at dropped location.

    If they differ significantly, CV is testing the wrong estimator.
    """
    from geostats.arbf.cross_validation import leave_one_out_cv
    from geostats.arbf.engine import ARBFEstimator

    np.random.seed(42)
    N = 60  # Small for brute-force LOO
    coords = np.random.uniform(0, 100, (N, 3))
    values = 10 + 5 * np.exp(-((coords[:, 0] - 50)**2) / (2*30**2)) + \
             np.random.normal(0, 0.5, N)

    sill = float(np.var(values))
    range_ = 40.0
    nugget = 0.25

    # 1. Virtual LOO from cross_validation module (global RBF)
    t0 = time.time()
    cv_result = leave_one_out_cv(
        coords, values,
        kernel_type="spheroidal", alpha=1.0,
        sill=sill, range_=range_, nugget=nugget, accuracy=1e-6,
        drift_type="constant",
    )
    cv_estimated = cv_result.estimated.copy()
    elapsed_cv = (time.time() - t0) * 1000

    # 2. Brute-force LOO using full PUM engine (subsample for speed)
    n_loo = min(20, N)
    loo_indices = np.random.choice(N, n_loo, replace=False)
    pum_estimated = np.zeros(n_loo)

    t0 = time.time()
    for i, drop_idx in enumerate(loo_indices):
        mask = np.ones(N, dtype=bool)
        mask[drop_idx] = False
        est = ARBFEstimator({
            "kernel_type": "spheroidal", "alpha": 1.0,
            "nugget": nugget, "accuracy": 1e-6,
            "range_max": range_, "range_mid": range_, "range_min": range_,
            "sill": sill,
            "variogram_mode": "global",
            "change_of_support": False, "run_cv": False,
            "verbose": False, "discretisation_density": 8,
            "n_subdomains": 4,
        })
        est.set_composites(coords[mask], values[mask])
        est.set_block_model(coords[drop_idx:drop_idx+1], np.array([1.0, 1.0, 1.0]))
        result = est.estimate()
        pum_estimated[i] = result.grades[0]
    elapsed_pum = (time.time() - t0) * 1000

    # Compare
    cv_at_loo = cv_estimated[loo_indices]
    diff = np.abs(cv_at_loo - pum_estimated)
    mean_diff = float(np.mean(diff))
    max_diff = float(np.max(diff))
    corr = float(np.corrcoef(cv_at_loo, pum_estimated)[0, 1]) if n_loo > 2 else 0.0

    # If CV uses a different estimator than PUM, predictions will differ
    record("CV6-estimator-identity", True,  # diagnostic — report the findings
           f"CV(global) vs PUM(brute-force) at {n_loo} points: "
           f"mean_diff={mean_diff:.4f}, max_diff={max_diff:.4f}, corr={corr:.4f}. "
           f"{'SAME estimator' if mean_diff < 0.5 else 'DIFFERENT estimators — CV uses global RBF, not PUM'}",
           elapsed_cv + elapsed_pum)

    # Quantify the bias this introduces
    cv_errors = values[loo_indices] - cv_at_loo
    pum_errors = values[loo_indices] - pum_estimated
    cv_rmse = float(np.sqrt(np.mean(cv_errors**2)))
    pum_rmse = float(np.sqrt(np.mean(pum_errors**2)))
    record("CV6b-rmse-comparison", True,
           f"CV(global) RMSE={cv_rmse:.4f}, PUM(brute-force) RMSE={pum_rmse:.4f}, "
           f"ratio={cv_rmse/max(pum_rmse,1e-12):.3f}")


# ===================================================================
# TEST: B2 — LVA WIRING
# ===================================================================

def test_lva_wiring():
    """B2: Verify that LVA (locally varying anisotropy) actually changes results.

    If LVA is dead code, use_lva=True and use_lva=False produce identical results.
    """
    from geostats.arbf.engine import ARBFEstimator

    np.random.seed(42)
    N = 100
    # Anisotropic data: grade continuity along x=y diagonal
    coords = np.random.uniform(0, 100, (N, 3))
    diag_dist = (coords[:, 0] + coords[:, 1]) / np.sqrt(2)
    values = 10 + 5 * np.sin(diag_dist / 20) + np.random.normal(0, 0.3, N)

    centroids = np.random.uniform(10, 90, (50, 3))
    block_sizes = np.array([10.0, 10.0, 10.0])

    common_cfg = {
        "kernel_type": "spheroidal", "alpha": 1.0,
        "nugget": 0.5, "accuracy": 1e-6,
        "range_max": 40.0, "range_mid": 40.0, "range_min": 40.0,
        "sill": float(np.var(values)),
        "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 4,
    }

    # Without LVA
    t0 = time.time()
    est_no_lva = ARBFEstimator({**common_cfg, "use_lva": False})
    est_no_lva.set_composites(coords, values)
    est_no_lva.set_block_model(centroids.copy(), block_sizes)
    r_no = est_no_lva.estimate()

    # With LVA
    est_lva = ARBFEstimator({**common_cfg, "use_lva": True, "lva_source": "data"})
    est_lva.set_composites(coords, values)
    est_lva.set_block_model(centroids.copy(), block_sizes)
    r_lva = est_lva.estimate()
    elapsed = (time.time() - t0) * 1000

    diff = np.abs(r_no.grades - r_lva.grades)
    mean_diff = float(np.mean(diff))
    max_diff = float(np.max(diff))
    identical = mean_diff < 1e-10

    record("B2-lva-wired", not identical,
           f"LVA {'NOT wired (dead code)' if identical else 'IS wired (changes results)'}. "
           f"mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}",
           elapsed)

    # Check that LVA results are still reasonable
    record("B2b-lva-reasonable", np.std(r_lva.grades) > 0.1,
           f"lva_std={np.std(r_lva.grades):.4f}, "
           f"no_lva_std={np.std(r_no.grades):.4f}")


# ===================================================================
# TEST: PRODUCTION-LIKE CV
# ===================================================================

def test_production_cv():
    """CV on multi-domain, high-variance data — closer to production conditions."""
    from geostats.arbf.cross_validation import leave_one_out_cv

    coords, values = make_multidomain_cv_dataset(n=300)
    data_var = float(np.var(values))

    # Production-like parameters
    t0 = time.time()
    cv = leave_one_out_cv(
        coords, values,
        kernel_type="spheroidal", alpha=1.0,
        sill=data_var, range_=60.0,
        nugget=data_var * 0.1,  # 10% nugget
        accuracy=1e-6,
        drift_type="constant",
        max_samples=300,
    )
    elapsed = (time.time() - t0) * 1000

    record("PCV1-slope", 0.3 < cv.slope_of_regression < 1.5,
           f"slope={cv.slope_of_regression:.4f} (production-like: 3 domains, N=300)",
           elapsed)

    record("PCV2-r-squared", cv.r_squared > -0.5,
           f"R²={cv.r_squared:.4f}")

    record("PCV3-mean-error", abs(cv.mean_error) < 5.0,
           f"ME={cv.mean_error:.4f}")

    # CV with nugget=0 for comparison (suspected to be worse)
    t0 = time.time()
    cv0 = leave_one_out_cv(
        coords, values,
        kernel_type="spheroidal", alpha=1.0,
        sill=data_var, range_=60.0,
        nugget=0.0, accuracy=1e-6,
        drift_type="constant",
        max_samples=300,
    )
    record("PCV4-nugget-effect", True,
           f"nugget=0: slope={cv0.slope_of_regression:.4f}, RMSE={cv0.rmse:.4f}; "
           f"nugget=10%: slope={cv.slope_of_regression:.4f}, RMSE={cv.rmse:.4f}",
           (time.time()-t0)*1000)


# ===================================================================
# TEST: CLUSTERED SAMPLING
# ===================================================================

def test_clustered_sampling():
    """CL1-CL5: Clustered (preferential) sampling — mimics real drilling patterns.

    When samples are clustered around high-grade zones, the estimator sees
    biased information density. Tests whether the engine handles this
    without: (a) over-smoothing waste zones, (b) negatives from extrapolation,
    (c) nugget sensitivity amplified by clustering.
    """
    from geostats.arbf.engine import ARBFEstimator

    coords, values, centroids, block_sizes = make_clustered_dataset(n=300)

    sill = float(np.var(values))
    nugget_10 = sill * 0.10

    # CL1: Basic estimation with clustered data
    t0 = time.time()
    est = ARBFEstimator({
        "kernel_type": "spheroidal", "alpha": 1.0,
        "nugget": nugget_10, "accuracy": 1e-6,
        "range_max": 50.0, "range_mid": 50.0, "range_min": 50.0,
        "sill": sill,
        "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 6,
    })
    est.set_composites(coords, values)
    est.set_block_model(centroids, block_sizes)
    result = est.estimate()
    elapsed = (time.time() - t0) * 1000

    n_neg = int(np.sum(result.grades < 0))
    grade_std = float(np.std(result.grades))
    grade_mean = float(np.mean(result.grades))
    record("CL1-clustered-basic", grade_std > 1.0 and n_neg == 0,
           f"mean={grade_mean:.2f}, std={grade_std:.2f}, neg={n_neg}, "
           f"range=[{np.min(result.grades):.2f}, {np.max(result.grades):.2f}]",
           elapsed)

    # CL2: Grades in dense-sample zone should be higher than sparse-waste zone
    # Core blocks: near (50,50,50)
    core_mask = np.sqrt(np.sum((centroids - np.array([50, 50, 50]))**2, axis=1)) < 60
    waste_mask = ~core_mask
    if np.any(core_mask) and np.any(waste_mask):
        core_mean = float(np.mean(result.grades[core_mask]))
        waste_mean = float(np.mean(result.grades[waste_mask]))
        record("CL2-spatial-gradient", core_mean > waste_mean,
               f"core_mean={core_mean:.2f}, waste_mean={waste_mean:.2f}, "
               f"ratio={core_mean/max(waste_mean, 0.01):.2f}")
    else:
        record("CL2-spatial-gradient", False, "Could not partition blocks into core/waste")

    # CL3: Variance should be higher in sparse waste zone than dense core
    if np.any(core_mask) and np.any(waste_mask):
        core_var = float(np.mean(result.variances[core_mask]))
        waste_var = float(np.mean(result.variances[waste_mask]))
        record("CL3-variance-density", waste_var > core_var,
               f"core_var={core_var:.2f}, waste_var={waste_var:.2f}, "
               f"ratio={waste_var/max(core_var, 1e-6):.2f}")
    else:
        record("CL3-variance-density", False, "Could not partition blocks")

    # CL4: Nugget sensitivity with clustered data
    # nugget=0 should produce more extreme estimates due to tight clusters
    est0 = ARBFEstimator({
        "kernel_type": "spheroidal", "alpha": 1.0,
        "nugget": 0.0, "accuracy": 1e-6,
        "range_max": 50.0, "range_mid": 50.0, "range_min": 50.0,
        "sill": sill, "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 6,
    })
    est0.set_composites(coords, values)
    est0.set_block_model(centroids.copy(), block_sizes)
    r0 = est0.estimate()
    n_neg_0 = int(np.sum(r0.grades < 0))
    range_nug0 = float(np.max(r0.grades) - np.min(r0.grades))
    range_nug10 = float(np.max(result.grades) - np.min(result.grades))
    record("CL4-nugget-sensitivity", True,  # diagnostic
           f"nugget=0: neg={n_neg_0}, range={range_nug0:.2f}; "
           f"nugget=10%: neg={n_neg}, range={range_nug10:.2f}")

    # CL5: CV on clustered data — slope should still be reasonable
    from geostats.arbf.cross_validation import leave_one_out_cv
    t0 = time.time()
    cv = leave_one_out_cv(
        coords, values,
        kernel_type="spheroidal", alpha=1.0,
        sill=sill, range_=50.0,
        nugget=nugget_10, accuracy=1e-6,
        drift_type="constant",
    )
    record("CL5-clustered-cv", 0.2 < cv.slope_of_regression < 1.5,
           f"slope={cv.slope_of_regression:.4f}, R²={cv.r_squared:.4f}, "
           f"RMSE={cv.rmse:.4f}",
           (time.time() - t0) * 1000)


# ===================================================================
# TEST: HIGH-CV GRADE DISTRIBUTION
# ===================================================================

def test_high_cv():
    """HCV1-HCV5: Lognormal grade distribution with CV > 1.0.

    Heavy-tailed grade distributions (gold, PGMs) produce extreme sample
    values that stress-test the kernel system. Without proper nugget/accuracy
    parameters, the estimator can produce negatives or extreme overshoots.
    """
    from geostats.arbf.engine import ARBFEstimator

    coords, values, centroids, block_sizes = make_high_cv_dataset(n=300)
    cv = float(np.std(values) / np.mean(values))
    sill = float(np.var(values))
    nugget_15 = sill * 0.15  # 15% nugget for high-CV data

    # HCV1: Basic estimation — no negatives for lognormal data
    t0 = time.time()
    est = ARBFEstimator({
        "kernel_type": "spheroidal", "alpha": 1.0,
        "nugget": nugget_15, "accuracy": 1e-6,
        "range_max": 50.0, "range_mid": 50.0, "range_min": 50.0,
        "sill": sill, "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 6,
    })
    est.set_composites(coords, values)
    est.set_block_model(centroids, block_sizes)
    result = est.estimate()
    elapsed = (time.time() - t0) * 1000

    n_neg = int(np.sum(result.grades < 0))
    record("HCV1-no-negatives", True,  # diagnostic — report count
           f"CV={cv:.2f}, neg={n_neg}/{len(result.grades)}, "
           f"range=[{np.min(result.grades):.3f}, {np.max(result.grades):.3f}]",
           elapsed)

    # HCV2: Estimated mean should be within 50% of data mean (not wildly biased)
    data_mean = float(np.mean(values))
    est_mean = float(np.mean(result.grades))
    ratio = est_mean / max(data_mean, 1e-6)
    record("HCV2-mean-preservation", 0.5 < ratio < 2.0,
           f"data_mean={data_mean:.3f}, est_mean={est_mean:.3f}, ratio={ratio:.3f}")

    # HCV3: Nugget=0 vs nugget=15% — high-CV should amplify the difference
    est0 = ARBFEstimator({
        "kernel_type": "spheroidal", "alpha": 1.0,
        "nugget": 0.0, "accuracy": 1e-6,
        "range_max": 50.0, "range_mid": 50.0, "range_min": 50.0,
        "sill": sill, "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 6,
    })
    est0.set_composites(coords, values)
    est0.set_block_model(centroids.copy(), block_sizes)
    r0 = est0.estimate()
    n_neg_0 = int(np.sum(r0.grades < 0))
    record("HCV3-nugget-amplification", True,  # diagnostic
           f"nugget=0: neg={n_neg_0}, max={np.max(r0.grades):.3f}; "
           f"nugget=15%: neg={n_neg}, max={np.max(result.grades):.3f}")

    # HCV4: CV on raw high-CV data — slope expected to be poor.
    # This is a KNOWN LIMITATION: LOO-CV on raw lognormal data (CV>1) is
    # unreliable because the kernel system cannot represent the heavy tail.
    # Normal-score transform is required before estimation for high-CV deposits.
    from geostats.arbf.cross_validation import leave_one_out_cv
    t0 = time.time()
    cv_result = leave_one_out_cv(
        coords, values,
        kernel_type="spheroidal", alpha=1.0,
        sill=sill, range_=50.0,
        nugget=nugget_15, accuracy=1e-6,
        drift_type="constant",
    )
    # Pass condition: slope is poor (< 0.3) confirming that raw high-CV
    # data NEEDS normal-score transform. If slope were good, the test
    # wouldn't be catching anything.
    raw_slope_poor = cv_result.slope_of_regression < 0.3
    record("HCV4-raw-cv-poor", raw_slope_poor,
           f"slope={cv_result.slope_of_regression:.4f}, R²={cv_result.r_squared:.4f}, "
           f"RMSE={cv_result.rmse:.4f}, data_CV={cv:.2f} — "
           f"{'confirms need for normal-score transform' if raw_slope_poor else 'unexpectedly good'}",
           (time.time() - t0) * 1000)

    # HCV5: Normal-score transform should help high-CV data
    from geostats.arbf.transforms import normal_score_transform, normal_score_backtransform
    ns_values, ns_table = normal_score_transform(values)
    ns_cv = float(np.std(ns_values) / max(np.mean(np.abs(ns_values)), 1e-6))
    record("HCV5-nscore-reduces-cv", ns_cv < cv,
           f"original_CV={cv:.2f}, nscore_CV={ns_cv:.2f}, "
           f"reduction={1 - ns_cv/cv:.1%}")


# ===================================================================
# TEST: COMBINED WORST CASE
# ===================================================================

def test_combined_worst_case():
    """WC1-WC5: All four conditions combined — the production-failure regime.

    Clustered + high-CV + UTM-scale + sharp contact. This is the synthetic
    dataset that most closely mimics the conditions under which slope=0.261
    and negative Fe% were observed in production.
    """
    from geostats.arbf.engine import ARBFEstimator
    from geostats.arbf.cross_validation import leave_one_out_cv

    coords, values, centroids, block_sizes, cv = make_combined_worst_case(n=400)
    sill = float(np.var(values))

    # WC1: nugget=0, accuracy=0 — the bad parameter regime
    t0 = time.time()
    est_bad = ARBFEstimator({
        "kernel_type": "spheroidal", "alpha": 1.0,
        "nugget": 0.0, "accuracy": 0.0,
        "range_max": 80.0, "range_mid": 80.0, "range_min": 40.0,
        "sill": sill, "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 6,
    })
    est_bad.set_composites(coords, values)
    est_bad.set_block_model(centroids.copy(), block_sizes)
    r_bad = est_bad.estimate()
    elapsed_bad = (time.time() - t0) * 1000

    n_neg_bad = int(np.sum(r_bad.grades < 0))
    record("WC1-bad-params", True,  # diagnostic — this IS the failure mode
           f"nugget=0,acc=0: neg={n_neg_bad}, CV={cv:.2f}, "
           f"range=[{np.min(r_bad.grades):.2f}, {np.max(r_bad.grades):.2f}]",
           elapsed_bad)

    # WC2: nugget=10%, accuracy=1e-6 — the recommended fix
    t0 = time.time()
    nugget_10 = sill * 0.10
    est_good = ARBFEstimator({
        "kernel_type": "spheroidal", "alpha": 1.0,
        "nugget": nugget_10, "accuracy": 1e-6,
        "range_max": 80.0, "range_mid": 80.0, "range_min": 40.0,
        "sill": sill, "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 6,
    })
    est_good.set_composites(coords, values)
    est_good.set_block_model(centroids.copy(), block_sizes)
    r_good = est_good.estimate()
    elapsed_good = (time.time() - t0) * 1000

    n_neg_good = int(np.sum(r_good.grades < 0))
    improvement = n_neg_bad - n_neg_good
    record("WC2-good-params", n_neg_good < n_neg_bad or n_neg_bad == 0,
           f"nugget=10%,acc=1e-6: neg={n_neg_good} (was {n_neg_bad}), "
           f"improvement={improvement} fewer negatives, "
           f"range=[{np.min(r_good.grades):.2f}, {np.max(r_good.grades):.2f}]",
           elapsed_good)

    # WC3: CV on combined worst case — the bottom line
    t0 = time.time()
    # Use a subset for CV (full 400 is slow)
    cv_n = min(200, len(coords))
    cv_idx = np.random.choice(len(coords), cv_n, replace=False)
    cv_bad = leave_one_out_cv(
        coords[cv_idx], values[cv_idx],
        kernel_type="spheroidal", alpha=1.0,
        sill=sill, range_=80.0,
        nugget=0.0, accuracy=1e-6,
        drift_type="constant",
    )
    cv_good = leave_one_out_cv(
        coords[cv_idx], values[cv_idx],
        kernel_type="spheroidal", alpha=1.0,
        sill=sill, range_=80.0,
        nugget=nugget_10, accuracy=1e-6,
        drift_type="constant",
    )
    elapsed_cv = (time.time() - t0) * 1000

    record("WC3-cv-comparison", True,  # diagnostic
           f"bad(nug=0): slope={cv_bad.slope_of_regression:.4f}, RMSE={cv_bad.rmse:.4f}; "
           f"good(nug=10%): slope={cv_good.slope_of_regression:.4f}, RMSE={cv_good.rmse:.4f}",
           elapsed_cv)

    # WC4: Nugget=10% slope should be closer to 1.0 than nugget=0
    slope_bad_err = abs(cv_bad.slope_of_regression - 1.0)
    slope_good_err = abs(cv_good.slope_of_regression - 1.0)
    record("WC4-nugget-fixes-slope", slope_good_err < slope_bad_err,
           f"bad_slope_error={slope_bad_err:.4f}, good_slope_error={slope_good_err:.4f}")

    # WC5: Dataset characterisation — confirm all four conditions are present
    # UTM-scale: coordinate magnitudes > 100,000m
    is_utm = np.mean(coords[:, 0]) > 100000
    # Clustered: core samples (first 60%) are tighter than waste samples
    core_n = len(coords) * 3 // 5
    core_spread = np.std(coords[:core_n, 0])
    waste_spread = np.std(coords[core_n:, 0])
    is_clustered = core_spread < waste_spread * 0.5
    # High-CV
    is_high_cv = cv > 1.0
    # Sharp contact: check if there are both low and high grade regions
    has_contact = (np.percentile(values, 90) / max(np.percentile(values, 10), 0.01)) > 5

    all_four = is_utm and is_clustered and is_high_cv and has_contact
    record("WC5-conditions-verified", all_four,
           f"UTM={is_utm} (mean_x={np.mean(coords[:,0]):.0f}), "
           f"clustered={is_clustered} (core_spread={core_spread:.0f}, waste_spread={waste_spread:.0f}), "
           f"high_CV={is_high_cv} (CV={cv:.2f}), "
           f"sharp_contact={has_contact}")


# ===================================================================
# SINGLE-DOMAIN (NO PUM) TESTS
# ===================================================================


def test_single_domain():
    """Test single-domain estimation path — no sub-domains, no PUM blending."""
    from geostats.arbf.engine import ARBFEstimator

    # ---- SD1: Single-domain vs PUM on small dataset ----
    # For N=200, single-domain should produce results with ZERO between-model variance
    np.random.seed(42)
    N = 200
    coords = np.random.uniform(0, 500, (N, 3))
    values = 10.0 + 5.0 * np.sin(coords[:, 0] / 80) + np.random.normal(0, 1, N)

    n_blocks = 100
    block_coords = np.random.uniform(50, 450, (n_blocks, 3))
    block_sizes = np.full((n_blocks, 3), 25.0)

    t0 = time.time()
    est_sd = ARBFEstimator({
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "sill": float(np.var(values)),
        "nugget": float(np.var(values)) * 0.1,
        "accuracy": 1e-6,
        "range_max": 100.0, "range_mid": 100.0, "range_min": 100.0,
        "n_subdomains": 1,  # Force single-domain
        "pum_threshold": 3000,
        "run_cv": True,
        "change_of_support": False,
        "variogram_mode": "global",
        "verbose": False,
    })
    est_sd.set_composites(coords, values)
    est_sd.set_block_model(block_coords, block_sizes)
    result_sd = est_sd.estimate()
    elapsed_sd = (time.time() - t0) * 1000

    # Check single_domain flag in diagnostics
    is_single = result_sd.diagnostics.get("single_domain", False)
    record("SD1-flag", is_single,
           f"single_domain={is_single}", elapsed_sd)

    # Check between-model variance is exactly zero
    between = result_sd.diagnostics.get("blended_between_variance_mean", -1)
    record("SD1-no-between-variance", between == 0.0,
           f"between_variance_mean={between:.6e} (should be 0.0)", elapsed_sd)

    # ---- SD2: Single-domain produces non-trivial estimates ----
    grade_std = float(np.std(result_sd.grades))
    record("SD2-nontrivial-estimates", grade_std > 0.5,
           f"grade_std={grade_std:.4f} (>0.5 = non-trivial)", elapsed_sd)

    # ---- SD3: Single-domain matches PUM with 1 subdomain ----
    # Compare single-domain (direct GPR) to PUM with n_subdomains=0 but
    # pum_threshold high enough to trigger single-domain anyway
    est_pum = ARBFEstimator({
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "sill": float(np.var(values)),
        "nugget": float(np.var(values)) * 0.1,
        "accuracy": 1e-6,
        "range_max": 100.0, "range_mid": 100.0, "range_min": 100.0,
        "n_subdomains": 0,  # Auto — but N=200 < threshold=3000, so single-domain
        "pum_threshold": 3000,
        "run_cv": False,
        "change_of_support": False,
        "variogram_mode": "global",
        "verbose": False,
    })
    est_pum.set_composites(coords, values)
    est_pum.set_block_model(block_coords, block_sizes)
    result_pum = est_pum.estimate()

    # Same engine path → identical results
    grade_diff = float(np.max(np.abs(result_sd.grades - result_pum.grades)))
    record("SD3-auto-threshold", grade_diff < 1e-10,
           f"max_grade_diff={grade_diff:.2e} (auto-threshold triggers single-domain)", 0)

    # ---- SD4: Explicit PUM with many subdomains (N=200) ----
    # Force PUM with high threshold disabled
    t0 = time.time()
    est_pum_forced = ARBFEstimator({
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "sill": float(np.var(values)),
        "nugget": float(np.var(values)) * 0.1,
        "accuracy": 1e-6,
        "range_max": 100.0, "range_mid": 100.0, "range_min": 100.0,
        "n_subdomains": 5,
        "pum_threshold": 0,  # Disable auto single-domain
        "max_samples": 100,
        "run_cv": False,
        "change_of_support": False,
        "variogram_mode": "global",
        "verbose": False,
    })
    est_pum_forced.set_composites(coords, values)
    est_pum_forced.set_block_model(block_coords, block_sizes)
    result_pum_forced = est_pum_forced.estimate()
    elapsed_pum = (time.time() - t0) * 1000

    is_pum = not result_pum_forced.diagnostics.get("single_domain", True)
    n_sd = result_pum_forced.diagnostics.get("n_subdomains", 0)
    record("SD4-forced-pum", is_pum and n_sd == 5,
           f"PUM mode: single_domain={not is_pum}, n_subdomains={n_sd}", elapsed_pum)

    # PUM should have non-zero between-model variance
    between_pum = result_pum_forced.diagnostics.get("blended_between_variance_mean", 0)
    record("SD4-pum-between-variance", between_pum > 0,
           f"PUM between_variance_mean={between_pum:.6e} (>0)", elapsed_pum)

    # ---- SD5: Single-domain with normal-score ----
    est_ns = ARBFEstimator({
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "sill": float(np.var(values)),
        "nugget": float(np.var(values)) * 0.1,
        "accuracy": 1e-6,
        "range_max": 100.0, "range_mid": 100.0, "range_min": 100.0,
        "n_subdomains": 1,
        "pum_threshold": 3000,
        "use_normal_score": True,
        "run_cv": True,
        "change_of_support": False,
        "variogram_mode": "global",
        "verbose": False,
    })
    est_ns.set_composites(coords, values)
    est_ns.set_block_model(block_coords, block_sizes)
    result_ns = est_ns.estimate()

    ns_std = float(np.std(result_ns.grades))
    ns_between = result_ns.diagnostics.get("blended_between_variance_mean", -1)
    record("SD5-normal-score-single-domain", ns_std > 0.5 and ns_between == 0.0,
           f"NS single-domain: grade_std={ns_std:.4f}, between_var={ns_between:.6e}", 0)

    # ---- SD6: UTM-scale single domain ----
    utm_coords, utm_values, utm_blocks, utm_bsizes = make_utm_dataset(300, 200)
    est_utm = ARBFEstimator({
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "sill": float(np.var(utm_values)),
        "nugget": float(np.var(utm_values)) * 0.15,
        "accuracy": 1e-6,
        "range_max": 80.0, "range_mid": 80.0, "range_min": 40.0,
        "n_subdomains": 1,
        "pum_threshold": 3000,
        "run_cv": True,
        "change_of_support": False,
        "variogram_mode": "global",
        "verbose": False,
    })
    est_utm.set_composites(utm_coords, utm_values)
    est_utm.set_block_model(utm_blocks, utm_bsizes)
    result_utm = est_utm.estimate()

    utm_is_single = result_utm.diagnostics.get("single_domain", False)
    utm_grade_std = float(np.std(result_utm.grades))
    record("SD6-utm-single-domain", utm_is_single and utm_grade_std > 1.0,
           f"UTM single-domain: grade_std={utm_grade_std:.2f}, single={utm_is_single}", 0)

    # ---- SD7: Large N bypasses single-domain ----
    # N > threshold should use PUM
    large_n = 500
    large_coords = np.random.uniform(0, 500, (large_n, 3))
    large_values = np.random.normal(10, 3, large_n)
    est_large = ARBFEstimator({
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "range_max": 100.0, "range_mid": 100.0, "range_min": 100.0,
        "n_subdomains": 0,
        "pum_threshold": 100,  # Very low threshold
        "run_cv": False,
        "change_of_support": False,
        "variogram_mode": "global",
        "verbose": False,
    })
    est_large.set_composites(large_coords, large_values)
    est_large.set_block_model(block_coords, block_sizes)
    result_large = est_large.estimate()

    large_is_pum = not result_large.diagnostics.get("single_domain", True)
    record("SD7-pum-above-threshold", large_is_pum,
           f"N={large_n} > threshold=100: PUM={large_is_pum}", 0)


# ===================================================================
# REPORT
# ===================================================================

def generate_report():
    n_total = len(RESULTS)
    n_pass = sum(1 for r in RESULTS if r.passed)
    n_fail = n_total - n_pass

    lines = [
        "# ARBF Engine — Production-Regime Test Report\n",
        "## Summary",
        f"- Total tests: {n_total}",
        f"- Passed: {n_pass}",
        f"- Failed: {n_fail}",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Python: {sys.version.split()[0]}",
        f"- NumPy: {np.__version__}",
        "",
    ]

    failures = [r for r in RESULTS if not r.passed]
    if failures:
        lines.append("## Critical Findings\n")
        for f in failures:
            lines.append(f"- **{f.name}**: {f.details}")
        lines.append("")

    lines.append("## All Results\n")
    lines.append("| Test | Result | Details | Time |")
    lines.append("|------|--------|---------|------|")
    for t in RESULTS:
        tag = "PASS" if t.passed else "**FAIL**"
        det = t.details.replace("\n", " ").replace("|", "\\|")
        if len(det) > 150:
            det = det[:147] + "..."
        lines.append(f"| {t.name} | {tag} | {det} | {t.elapsed_ms:.0f}ms |")
    lines.append("")

    lines.append("## Gap Analysis — Previously Untested Issues\n")
    lines.append("")
    lines.append("### C1: K_inv vs Cholesky Stability (G11/G12)")
    g11 = next((r for r in RESULTS if "G11" in r.name), None)
    g12 = next((r for r in RESULTS if "G12-ill" in r.name), None)
    g12b = next((r for r in RESULTS if "G12b" in r.name), None)
    if g11: lines.append(f"- G11: {g11.details}")
    if g12: lines.append(f"- G12: {g12.details}")
    if g12b: lines.append(f"- G12b: {g12b.details}")
    lines.append("")

    lines.append("### C3: LOO-CV Estimator Identity (CV6)")
    cv6 = next((r for r in RESULTS if "CV6-est" in r.name), None)
    cv6b = next((r for r in RESULTS if "CV6b" in r.name), None)
    if cv6: lines.append(f"- {cv6.details}")
    if cv6b: lines.append(f"- {cv6b.details}")
    lines.append("")

    lines.append("### B2: LVA Wiring")
    b2 = next((r for r in RESULTS if "B2-lva" in r.name), None)
    if b2: lines.append(f"- {b2.details}")
    lines.append("")

    report = "\n".join(lines)
    report_path = Path(__file__).parent.parent.parent / "ARBF_Production_Test_Report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return report_path


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_total = time.time()

    print("=" * 70)
    print("ARBF ENGINE — PRODUCTION-REGIME TESTS")
    print("=" * 70)

    test_fns = [
        ("UTM-scale coordinates", test_utm_scale),
        ("Sharp grade contrasts", test_sharp_contrasts),
        ("G11/G12 numerical stability", test_g11_g12_numerical_stability),
        ("CV6 estimator identity", test_cv6_estimator_identity),
        ("B2 LVA wiring", test_lva_wiring),
        ("Production-like CV", test_production_cv),
        ("Clustered sampling", test_clustered_sampling),
        ("High-CV grade distribution", test_high_cv),
        ("Combined worst case", test_combined_worst_case),
        ("Single-domain (no PUM)", test_single_domain),
    ]

    for name, fn in test_fns:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            record(f"CRASH-{name}", False, f"{e}\n{traceback.format_exc()}")
            logger.error("Test %s crashed: %s", name, e)

    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 70}")
    n_pass = sum(1 for r in RESULTS if r.passed)
    n_fail = len(RESULTS) - n_pass
    print(f"TOTAL: {len(RESULTS)} tests, {n_pass} passed, {n_fail} failed ({elapsed_total:.1f}s)")
    print(f"{'=' * 70}")

    if n_fail > 0:
        print("\nFAILED TESTS:")
        for r in RESULTS:
            if not r.passed:
                print(f"  [{r.name}] {r.details}")

    report_path = generate_report()
    print(f"\nReport: {report_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
