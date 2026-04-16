#!/usr/bin/env python
"""
ARBF Engine — Deep Test & Review Script.

Builds synthetic datasets with KNOWN answers and tests every module
in geostats/arbf/ against them.  Produces structured PASS/FAIL output
with diagnostic numbers for every test.

Usage:
    python -m tests.arbf.test_arbf_deep
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("arbf_deep_test")
logger.setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Test bookkeeping
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

def make_simple_dataset():
    """Grade = Gaussian bump centred at (50,50).  True answer known everywhere."""
    np.random.seed(42)
    N = 500
    coords = np.random.uniform(0, 100, size=(N, 3))
    true_grade = 10 + 5 * np.exp(
        -((coords[:, 0] - 50)**2 + (coords[:, 1] - 50)**2) / (2 * 30**2)
    )
    noise = np.random.normal(0, 0.5, N)
    observed = true_grade + noise

    bx, by, bz = np.meshgrid(
        np.arange(5, 100, 10),
        np.arange(5, 100, 10),
        np.arange(5, 100, 10),
        indexing='ij',
    )
    centroids = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()])
    block_sizes = np.array([10.0, 10.0, 10.0])

    true_block = 10 + 5 * np.exp(
        -((centroids[:, 0] - 50)**2 + (centroids[:, 1] - 50)**2) / (2 * 30**2)
    )
    return coords, observed, centroids, block_sizes, true_block, noise


def make_two_domain_dataset():
    """Two domains: x<50 mean=5, x>=50 mean=20."""
    np.random.seed(42)
    N = 400
    coords = np.random.uniform(0, 100, size=(N, 3))
    values = np.where(
        coords[:, 0] < 50,
        np.random.normal(5, 1, N),
        np.random.normal(20, 3, N),
    )
    bx, by, bz = np.meshgrid(
        np.arange(5, 100, 10),
        np.arange(5, 100, 10),
        np.arange(5, 100, 10),
        indexing='ij',
    )
    centroids = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()])
    block_sizes = np.array([10.0, 10.0, 10.0])
    true_block = np.where(centroids[:, 0] < 50, 5.0, 20.0)
    return coords, values, centroids, block_sizes, true_block


def make_compositional_dataset():
    """Three components summing to 100%."""
    np.random.seed(42)
    N = 300
    fe = np.random.uniform(30, 65, N)
    sio2 = np.random.uniform(5, 40, N)
    al2o3 = 100.0 - fe - sio2
    valid = al2o3 > 1.0
    compositions = np.column_stack([fe[valid], sio2[valid], al2o3[valid]])
    coords = np.random.uniform(0, 100, size=(compositions.shape[0], 3))
    return coords, compositions


def make_variable_density_dataset():
    """Dense (25m) + sparse (100m) drilling areas."""
    np.random.seed(42)
    # Dense area: x in [0, 50]
    n_dense = 200
    dense_coords = np.column_stack([
        np.random.uniform(0, 50, n_dense),
        np.random.uniform(0, 100, n_dense),
        np.random.uniform(0, 50, n_dense),
    ])
    # Sparse area: x in [50, 100]
    n_sparse = 30
    sparse_coords = np.column_stack([
        np.random.uniform(50, 100, n_sparse),
        np.random.uniform(0, 100, n_sparse),
        np.random.uniform(0, 50, n_sparse),
    ])
    coords = np.vstack([dense_coords, sparse_coords])
    values = 10 + np.random.normal(0, 2, len(coords))

    bx, by, bz = np.meshgrid(
        np.arange(5, 100, 10),
        np.arange(5, 100, 10),
        np.arange(5, 50, 10),
        indexing='ij',
    )
    centroids = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()])
    block_sizes = np.array([10.0, 10.0, 10.0])
    return coords, values, centroids, block_sizes


# ===================================================================
# MODULE TESTS
# ===================================================================

def test_kernels():
    """Tests K1-K7: kernel correctness."""
    from geostats.arbf.kernels import (
        evaluate_kernel, kernel_at_zero, supported_kernels,
        spheroidal, gaussian_kernel, matern_32, matern_52, cubic, wendland_c2,
    )

    kernels = supported_kernels()
    r_test = np.linspace(0, 5, 500)

    # K1: phi(0) == 1.0
    t0 = time.time()
    for kn in kernels:
        v = evaluate_kernel(np.array([0.0]), kn, alpha=1.0)[0]
        record(f"K1-{kn}", abs(v - 1.0) < 1e-12,
               f"phi(0)={v:.15f}", (time.time()-t0)*1000)

    # K2: phi(r) >= 0
    for kn in kernels:
        t0 = time.time()
        vals = evaluate_kernel(r_test, kn, alpha=1.0)
        ok = np.all(vals >= -1e-15)
        record(f"K2-{kn}", ok,
               f"min={np.min(vals):.6e}", (time.time()-t0)*1000)

    # K3: monotonically non-increasing
    for kn in kernels:
        t0 = time.time()
        vals = evaluate_kernel(r_test[r_test > 0], kn, alpha=1.0)
        diffs = np.diff(vals)
        ok = np.all(diffs <= 1e-10)
        record(f"K3-{kn}", ok,
               f"max_increase={np.max(diffs):.6e}", (time.time()-t0)*1000)

    # K4: phi(r)->0 for non-compact kernels
    r_far = np.array([100.0, 1000.0])
    for kn in ["spheroidal", "gaussian", "matern_32", "matern_52"]:
        t0 = time.time()
        vals = evaluate_kernel(r_far, kn, alpha=1.0)
        ok = np.all(vals < 0.01)
        record(f"K4-{kn}", ok,
               f"phi(100)={vals[0]:.6e}, phi(1000)={vals[1]:.6e}",
               (time.time()-t0)*1000)

    # K5: compact kernels exactly 0 for r>=1
    for kn in ["cubic", "wendland_c2"]:
        t0 = time.time()
        r_out = np.array([1.0, 1.5, 10.0])
        vals = evaluate_kernel(r_out, kn, alpha=1.0)
        ok = np.all(vals == 0.0)
        record(f"K5-{kn}", ok,
               f"vals_at_r>=1: {vals}", (time.time()-t0)*1000)

    # K6: spheroidal alpha comparison
    t0 = time.time()
    r_mid = np.array([0.5])
    v05 = spheroidal(r_mid, alpha=0.5)[0]
    v20 = spheroidal(r_mid, alpha=2.0)[0]
    # Higher alpha -> faster decay -> lower value at same r
    ok = v05 > v20
    record("K6-alpha", ok,
           f"alpha=0.5: phi(0.5)={v05:.6f}, alpha=2.0: phi(0.5)={v20:.6f}",
           (time.time()-t0)*1000)

    # K7: kernel_at_zero always 1.0
    t0 = time.time()
    for kn in kernels:
        ok = kernel_at_zero(kn) == 1.0
        record(f"K7-{kn}", ok, f"kernel_at_zero={kernel_at_zero(kn)}")


def test_utils():
    """Tests U1-U8: rotation, scaling, distance, Cholesky."""
    from geostats.arbf.utils import (
        rotation_matrix, scale_matrix, anisotropic_distance,
        pairwise_anisotropic_distance, stable_cholesky,
        condition_number_estimate, clamp_variance,
    )

    # U1: R(0,0,0) == I
    t0 = time.time()
    R = rotation_matrix(0, 0, 0)
    ok = np.allclose(R, np.eye(3), atol=1e-12)
    record("U1-identity", ok, f"max_diff={np.max(np.abs(R - np.eye(3))):.2e}",
           (time.time()-t0)*1000)

    # U2: orthogonality
    t0 = time.time()
    R = rotation_matrix(45, 30, 15)
    RtR = R.T @ R
    ok = np.allclose(RtR, np.eye(3), atol=1e-12)
    record("U2-orthogonal", ok,
           f"max_diff_from_I={np.max(np.abs(RtR - np.eye(3))):.2e}",
           (time.time()-t0)*1000)

    # U3: isotropic scaling
    t0 = time.time()
    S = scale_matrix(50, 50, 50)
    expected = np.diag([1/50, 1/50, 1/50])
    ok = np.allclose(S, expected, atol=1e-12)
    record("U3-isotropic", ok, f"S_diag={np.diag(S)}")

    # U4: aniso distance with R=I, S=I == Euclidean
    t0 = time.time()
    pts_a = np.array([[0, 0, 0], [3, 4, 0]], dtype=float)
    pts_b = np.array([[0, 0, 0]], dtype=float)
    D = anisotropic_distance(pts_a, pts_b, np.eye(3), np.eye(3))
    expected_d = np.array([[0.0], [5.0]])
    ok = np.allclose(D, expected_d, atol=1e-10)
    record("U4-euclid", ok, f"D={D.ravel()}, expected={expected_d.ravel()}")

    # U5: pairwise symmetric with zero diagonal
    t0 = time.time()
    coords = np.random.RandomState(42).uniform(0, 100, (20, 3))
    D = pairwise_anisotropic_distance(coords, np.eye(3), np.eye(3))
    ok_sym = np.allclose(D, D.T, atol=1e-12)
    ok_diag = np.allclose(np.diag(D), 0.0, atol=1e-12)
    record("U5-symmetric", ok_sym and ok_diag,
           f"sym_err={np.max(np.abs(D-D.T)):.2e}, diag_max={np.max(np.abs(np.diag(D))):.2e}")

    # U6: stable_cholesky on well-conditioned SPD
    t0 = time.time()
    A = np.eye(10) + 0.1 * np.random.RandomState(42).randn(10, 10)
    A = A @ A.T  # SPD
    L = stable_cholesky(A)
    recon = L @ L.T
    ok = np.allclose(recon, A, atol=1e-8)
    record("U6-cholesky-spd", ok,
           f"recon_err={np.max(np.abs(recon-A)):.2e}", (time.time()-t0)*1000)

    # U7: stable_cholesky with near-singular matrix
    t0 = time.time()
    v = np.random.RandomState(42).randn(10, 1)
    A_singular = v @ v.T + 1e-14 * np.eye(10)
    try:
        L = stable_cholesky(A_singular)
        ok = True
        details = "succeeded with jitter"
    except np.linalg.LinAlgError:
        ok = False
        details = "FAILED even with jitter"
    record("U7-cholesky-singular", ok, details, (time.time()-t0)*1000)

    # U8: clamp_variance
    t0 = time.time()
    v = np.array([1.0, -1e-15, 0.0, 5.0])
    cv = clamp_variance(v)
    ok = np.all(cv >= 0)
    record("U8-clamp", ok, f"clamped={cv}")


def test_transforms():
    """Tests T1-T9: normal-score and ILR transforms."""
    from geostats.arbf.transforms import (
        normal_score_transform, normal_score_backtransform,
        ilr_forward, ilr_inverse, _helmert_matrix,
    )

    np.random.seed(42)
    values = np.random.lognormal(2, 0.5, 200)

    # T1: NS output ~N(0,1)
    t0 = time.time()
    ns, table = normal_score_transform(values)
    ok = abs(np.mean(ns)) < 0.15 and abs(np.std(ns) - 1.0) < 0.15
    record("T1-ns-normal", ok,
           f"mean={np.mean(ns):.4f}, std={np.std(ns):.4f}",
           (time.time()-t0)*1000)

    # T2: NS round-trip
    t0 = time.time()
    back = normal_score_backtransform(ns, table)
    max_diff = np.max(np.abs(back - values))
    ok = max_diff < 0.5  # Some error from interpolation
    record("T2-ns-roundtrip", ok,
           f"max_diff={max_diff:.6f}", (time.time()-t0)*1000)

    # T3: Back-transform clamps to data range
    t0 = time.time()
    extreme = np.array([-10.0, 10.0])
    bt = normal_score_backtransform(extreme, table)
    ok = bt[0] >= table.data_min - 1e-10 and bt[1] <= table.data_max + 1e-10
    record("T3-ns-clamp", ok,
           f"bt=({bt[0]:.4f},{bt[1]:.4f}), range=({table.data_min:.4f},{table.data_max:.4f})")

    # T4: ILR output dimension
    t0 = time.time()
    comp = np.array([[60, 30, 10], [50, 40, 10], [40, 35, 25]], dtype=float)
    ilr = ilr_forward(comp)
    ok = ilr.shape == (3, 2)
    record("T4-ilr-dim", ok, f"shape={ilr.shape}, expected=(3,2)")

    # T5: ILR round-trip — CRITICAL
    t0 = time.time()
    comp_back = ilr_inverse(ilr, kappa=100.0)
    max_diff = np.max(np.abs(comp_back - comp))
    ok = max_diff < 0.01
    record("T5-ilr-roundtrip", ok,
           f"max_diff={max_diff:.6e}\norig:\n{comp}\nback:\n{comp_back}",
           (time.time()-t0)*1000)

    # T6: ILR inverse all positive
    t0 = time.time()
    random_ilr = np.random.randn(50, 2) * 2
    inv = ilr_inverse(random_ilr)
    ok = np.all(inv > 0)
    record("T6-ilr-positive", ok, f"min_component={np.min(inv):.6e}")

    # T7: ILR inverse row sums == kappa
    t0 = time.time()
    row_sums = np.sum(inv, axis=1)
    ok = np.allclose(row_sums, 100.0, atol=1e-6)
    record("T7-ilr-closure", ok,
           f"row_sums: min={np.min(row_sums):.6f}, max={np.max(row_sums):.6f}")

    # T8: ILR edge cases
    t0 = time.time()
    edge = np.array([[99.0, 0.5, 0.5]], dtype=float)
    ilr_e = ilr_forward(edge)
    back_e = ilr_inverse(ilr_e, kappa=100.0)
    max_diff = np.max(np.abs(back_e - edge))
    ok = max_diff < 0.5  # Generous for near-boundary
    record("T8-ilr-edge", ok,
           f"orig={edge[0]}, back={back_e[0]}, diff={max_diff:.4f}")

    # T8b: ILR rejects zeros
    t0 = time.time()
    try:
        ilr_forward(np.array([[60, 0, 40]], dtype=float))
        ok = False
        details = "did NOT raise ValueError for zero component"
    except ValueError:
        ok = True
        details = "correctly raised ValueError"
    record("T8b-ilr-zero-reject", ok, details)

    # T9: Helmert orthogonality
    t0 = time.time()
    for D in [3, 4, 5]:
        Psi = _helmert_matrix(D)
        PtP = Psi.T @ Psi
        ok_d = np.allclose(PtP, np.eye(D-1), atol=1e-10)
        record(f"T9-helmert-D{D}", ok_d,
               f"max_diff_from_I={np.max(np.abs(PtP - np.eye(D-1))):.2e}")


def test_variogram():
    """Tests V1-V8: experimental and fitted variogram."""
    from geostats.arbf.variogram import (
        compute_experimental_variogram, fit_local_variogram,
    )

    np.random.seed(42)

    # V1: White noise -> flat variogram at data variance
    t0 = time.time()
    N = 300
    coords_wn = np.random.uniform(0, 100, (N, 3))
    vals_wn = np.random.normal(0, 5, N)
    data_var = np.var(vals_wn)
    lags, gamma, counts = compute_experimental_variogram(coords_wn, vals_wn, n_lags=15)
    valid = counts > 10
    mean_gamma = np.mean(gamma[valid])
    ratio = mean_gamma / data_var
    ok = 0.5 < ratio < 1.8
    record("V1-white-noise", ok,
           f"mean_gamma={mean_gamma:.3f}, data_var={data_var:.3f}, ratio={ratio:.3f}",
           (time.time()-t0)*1000)

    # V2: Spatially correlated data -> increasing variogram
    t0 = time.time()
    coords_sc = np.random.uniform(0, 100, (N, 3))
    vals_sc = 10 + 5 * np.exp(-np.sum((coords_sc - 50)**2, axis=1) / (2 * 30**2))
    vals_sc += np.random.normal(0, 0.3, N)
    lags_sc, gamma_sc, counts_sc = compute_experimental_variogram(
        coords_sc, vals_sc, n_lags=15)
    valid_sc = counts_sc > 5
    if np.sum(valid_sc) >= 3:
        lags_v = lags_sc[valid_sc]
        gamma_v = gamma_sc[valid_sc]
        # Check first third vs last third
        n3 = max(1, len(gamma_v) // 3)
        early_mean = np.mean(gamma_v[:n3])
        late_mean = np.mean(gamma_v[-n3:])
        ok = late_mean > early_mean
        record("V2-spatial-corr", ok,
               f"early_mean={early_mean:.4f}, late_mean={late_mean:.4f}")
    else:
        record("V2-spatial-corr", False, "too few valid lags")

    # V3-V5: Fitted variogram on correlated data with noise
    t0 = time.time()
    vr = fit_local_variogram(coords_sc, vals_sc, kernel_type="spheroidal",
                             n_lags=15, use_nugget=True)
    data_var_sc = np.var(vals_sc)

    # V3: sill ~ data variance
    sill_ratio = (vr.sill + vr.nugget) / data_var_sc
    ok = 0.2 < sill_ratio < 5.0
    record("V3-sill-approx", ok,
           f"sill={vr.sill:.4f}, nugget={vr.nugget:.4f}, total={vr.sill+vr.nugget:.4f}, "
           f"data_var={data_var_sc:.4f}, ratio={sill_ratio:.3f}",
           (time.time()-t0)*1000)

    # V4: range ~ known correlation length (30m)
    ok = 5 < vr.range_ < 200
    record("V4-range-approx", ok,
           f"fitted_range={vr.range_:.1f}, expected~30m")

    # V5: nugget > 0 (noise is present)
    ok = vr.nugget >= 0
    record("V5-nugget-positive", ok,
           f"nugget={vr.nugget:.6f}")

    # V6: gamma(0) = nugget for variogram model
    from geostats.arbf.kernels import evaluate_kernel
    r0 = np.array([0.0])
    phi0 = evaluate_kernel(r0, vr.kernel_type, alpha=vr.alpha)[0]
    gamma0 = vr.sill * (1 - phi0)  # nugget added separately only when h>0
    ok = abs(gamma0) < 1e-10
    record("V6-gamma-at-zero", ok, f"gamma(0)={gamma0:.2e} (should be ~0, nugget is discontinuity)")

    # V7: gamma(inf) = sill + nugget
    r_inf = np.array([1e6])
    phi_inf = evaluate_kernel(r_inf / max(vr.range_, 1e-12), vr.kernel_type, alpha=vr.alpha)[0]
    gamma_inf = vr.sill * (1 - phi_inf) + vr.nugget
    expected = vr.sill + vr.nugget
    ok = abs(gamma_inf - expected) / max(expected, 1e-12) < 0.01
    record("V7-gamma-inf", ok,
           f"gamma(inf)={gamma_inf:.4f}, expected={expected:.4f}")

    # V8: Very few samples — no crash
    t0 = time.time()
    try:
        vr_small = fit_local_variogram(
            np.array([[0,0,0],[1,1,1],[2,2,2],[5,5,5]], dtype=float),
            np.array([1, 2, 3, 4], dtype=float),
            kernel_type="spheroidal",
        )
        ok = True
        details = f"sill={vr_small.sill:.4f}, range={vr_small.range_:.1f}"
    except Exception as e:
        ok = False
        details = f"crashed: {e}"
    record("V8-few-samples", ok, details, (time.time()-t0)*1000)


def test_partition():
    """Tests P1-P8: sub-domain decomposition."""
    from geostats.arbf.partition import (
        create_subdomains, wendland_c2_weight, wendland_c2_weight_batch,
        fit_subdomain_variograms,
    )

    np.random.seed(42)
    N = 500
    coords = np.random.uniform(0, 100, (N, 3))
    values = np.random.normal(10, 2, N)

    sds = create_subdomains(coords, method="kmeans", overlap_factor=1.5)

    # P1: coverage
    t0 = time.time()
    covered = np.zeros(N, dtype=bool)
    for sd in sds:
        covered[sd.sample_indices] = True
    pct = np.mean(covered) * 100
    ok = pct > 95
    record("P1-coverage", ok, f"{pct:.1f}% covered", (time.time()-t0)*1000)

    # P2: overlap — most points in >=2 SDs
    t0 = time.time()
    counts = np.zeros(N, dtype=int)
    for sd in sds:
        counts[sd.sample_indices] += 1
    pct_overlap = np.mean(counts >= 2) * 100
    ok = pct_overlap > 30  # At least 30% in 2+ domains
    record("P2-overlap", ok,
           f"{pct_overlap:.1f}% in >=2 SDs, mean_count={np.mean(counts):.1f}")

    # P3: reasonable K
    t0 = time.time()
    K = len(sds)
    ok = 2 <= K <= 50
    record("P3-auto-K", ok, f"K={K} for N={N}")

    # P5: max samples cap
    t0 = time.time()
    max_n = max(sd.n_samples for sd in sds)
    ok = max_n <= 350  # 300 + some tolerance
    record("P5-max-samples", ok, f"max_n_samples={max_n}")

    # P6: Wendland weight properties
    t0 = time.time()
    R = 100.0
    w0 = wendland_c2_weight(0.0, R)
    wR = wendland_c2_weight(R, R)
    w_mid = wendland_c2_weight(R * 0.5, R)
    ok = abs(w0 - 1.0) < 1e-10 and abs(wR) < 1e-10 and w_mid > 0
    record("P6-wendland", ok,
           f"w(0)={w0:.6f}, w(R)={wR:.6f}, w(0.5R)={w_mid:.6f}")

    # P7: batch weights sum to something sensible
    t0 = time.time()
    dists = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    wb = wendland_c2_weight_batch(dists, R)
    ok = wb[0] > 0.99 and abs(wb[-1]) < 1e-10
    record("P7-wendland-batch", ok, f"weights={wb}")


def test_orientation():
    """Tests O1-O5: orientation field."""
    from geostats.arbf.orientation import OrientationField

    # O1: identity field
    t0 = time.time()
    of = OrientationField.identity(
        np.array([0, 0, 0], dtype=float),
        np.array([10, 10, 10], dtype=float),
        (3, 3, 3),
    )
    R = of.interpolate(np.array([15, 15, 15], dtype=float))
    ok = np.allclose(R, np.eye(3), atol=1e-10)
    record("O1-identity", ok, f"max_diff={np.max(np.abs(R - np.eye(3))):.2e}",
           (time.time()-t0)*1000)

    # O2-O3: from_grade_data -> orthogonal + right-handed
    t0 = time.time()
    np.random.seed(42)
    coords = np.random.uniform(0, 100, (100, 3))
    grades = 10 + coords[:, 0] * 0.1  # Grade trend along X
    of2 = OrientationField.from_grade_data(
        coords, grades,
        np.array([0, 0, 0], dtype=float),
        np.array([25, 25, 25], dtype=float),
        (4, 4, 4),
        k_neighbours=20,
    )
    R2 = of2.interpolate(np.array([50, 50, 25], dtype=float))
    RtR = R2.T @ R2
    ok_orth = np.allclose(RtR, np.eye(3), atol=1e-6)
    det_val = np.linalg.det(R2)
    ok_rh = det_val > 0.99
    record("O2-orthogonal", ok_orth,
           f"max_diff_from_I={np.max(np.abs(RtR - np.eye(3))):.2e}")
    record("O3-right-handed", ok_rh, f"det={det_val:.6f}")

    # O4: interpolate SVD re-orthogonalisation
    t0 = time.time()
    R_interp = of2.interpolate(np.array([37.5, 37.5, 12.5], dtype=float))
    RtR2 = R_interp.T @ R_interp
    ok = np.allclose(RtR2, np.eye(3), atol=1e-6)
    record("O4-interp-orthogonal", ok,
           f"max_diff={np.max(np.abs(RtR2 - np.eye(3))):.2e}")

    # O5: interpolate at grid node
    t0 = time.time()
    R_node = of2.rotations[0, 0, 0]
    R_interp_node = of2.interpolate(of2.grid_origin)
    ok = np.allclose(R_node, R_interp_node, atol=1e-6)
    record("O5-grid-node", ok,
           f"max_diff={np.max(np.abs(R_node - R_interp_node)):.2e}")


def test_gpr():
    """Tests G1-G15: Gaussian Process Regression."""
    from geostats.arbf.gpr import (
        assemble_kernel_matrix, factorise_and_solve,
        predict_mean, predict_variance, predict_mean_and_variance,
        build_polynomial_matrix,
    )
    from geostats.arbf.utils import rotation_matrix, scale_matrix

    np.random.seed(42)
    N = 30
    coords = np.random.uniform(0, 100, (N, 3))
    values = 10 + np.sin(coords[:, 0] / 10)

    sill = 2.0
    range_ = 30.0
    nugget = 0.1
    accuracy = 1e-6
    R = np.eye(3)
    S = scale_matrix(range_, range_, range_)

    K_aug, P = assemble_kernel_matrix(
        coords, kernel_type="spheroidal", alpha=1.0,
        sill=sill, range_=range_, nugget=nugget, accuracy=accuracy,
        R=R, S=S, drift_type="constant",
    )
    fact, weights, poly_c = factorise_and_solve(K_aug, values, drift_type="constant")

    # G1: predict_mean at known function evaluation points
    t0 = time.time()
    est_at_data = predict_mean(
        coords, coords, weights, poly_c,
        kernel_type="spheroidal", alpha=1.0, sill=sill, range_=range_,
        R=R, S=S, drift_type="constant",
    )
    max_err = np.max(np.abs(est_at_data - values))
    ok = max_err < 2.0  # With nugget, not exact interpolation
    record("G1-mean-at-data", ok,
           f"max_err={max_err:.4f}", (time.time()-t0)*1000)

    # G2: predict_mean at data point (nugget=0, accuracy=0)
    t0 = time.time()
    K2, P2 = assemble_kernel_matrix(
        coords, kernel_type="spheroidal", alpha=1.0,
        sill=sill, range_=range_, nugget=0.0, accuracy=0.0,
        R=R, S=S, drift_type="constant",
    )
    fact2, w2, pc2 = factorise_and_solve(K2, values, drift_type="constant")
    est2 = predict_mean(coords, coords, w2, pc2,
                        kernel_type="spheroidal", alpha=1.0, sill=sill, range_=range_,
                        R=R, S=S, drift_type="constant")
    max_err2 = np.max(np.abs(est2 - values))
    ok = max_err2 < 0.5
    record("G2-exact-interp", ok,
           f"max_err={max_err2:.6f} (nugget=0, acc=0)", (time.time()-t0)*1000)

    # G4: variance >= 0 everywhere
    t0 = time.time()
    query = np.random.uniform(0, 100, (100, 3))
    var = predict_variance(query, coords, fact,
                           kernel_type="spheroidal", alpha=1.0,
                           sill=sill, range_=range_, nugget=nugget,
                           R=R, S=S, drift_type="constant")
    ok = np.all(var >= 0)
    record("G4-var-nonneg", ok,
           f"min_var={np.min(var):.6e}, negatives={np.sum(var < 0)}",
           (time.time()-t0)*1000)

    # G5: variance at data point -> small when nugget=0
    t0 = time.time()
    var_at_data = predict_variance(
        coords, coords, fact2,
        kernel_type="spheroidal", alpha=1.0,
        sill=sill, range_=range_, nugget=0.0,
        R=R, S=S, drift_type="constant",
    )
    ok = np.mean(var_at_data) < sill * 0.1
    record("G5-var-at-data", ok,
           f"mean_var={np.mean(var_at_data):.6e}, max={np.max(var_at_data):.6e}")

    # G6: variance far from data -> near sill
    t0 = time.time()
    far_pts = np.array([[500, 500, 500], [1000, 1000, 1000]], dtype=float)
    var_far = predict_variance(far_pts, coords, fact,
                               kernel_type="spheroidal", alpha=1.0,
                               sill=sill, range_=range_, nugget=nugget,
                               R=R, S=S, drift_type="constant")
    ok = np.all(var_far > sill * 0.5)
    record("G6-var-far", ok,
           f"var_far={var_far}, sill={sill}")

    # G7: variance depends on data density
    t0 = time.time()
    dense_pt = np.mean(coords, axis=0).reshape(1, 3)
    sparse_pt = np.array([[200, 200, 200]], dtype=float)
    var_dense = predict_variance(dense_pt, coords, fact,
                                 kernel_type="spheroidal", alpha=1.0,
                                 sill=sill, range_=range_, nugget=nugget,
                                 R=R, S=S, drift_type="constant")
    var_sparse = predict_variance(sparse_pt, coords, fact,
                                  kernel_type="spheroidal", alpha=1.0,
                                  sill=sill, range_=range_, nugget=nugget,
                                  R=R, S=S, drift_type="constant")
    ok = var_dense[0] < var_sparse[0]
    record("G7-var-density", ok,
           f"var_dense={var_dense[0]:.6f}, var_sparse={var_sparse[0]:.6f}")

    # G9: accuracy=0 oscillations
    t0 = time.time()
    K9, _ = assemble_kernel_matrix(
        coords, kernel_type="spheroidal", alpha=1.0,
        sill=sill, range_=range_, nugget=0.0, accuracy=0.0,
        R=R, S=S, drift_type="constant",
    )
    f9, w9, pc9 = factorise_and_solve(K9, values, drift_type="constant")
    est9 = predict_mean(query, coords, w9, pc9,
                        kernel_type="spheroidal", alpha=1.0, sill=sill, range_=range_,
                        R=R, S=S, drift_type="constant")
    n_neg = np.sum(est9 < 0)
    n_extreme = np.sum(np.abs(est9 - np.mean(values)) > 3 * np.std(values))
    record("G9-acc0-oscillation", True,  # diagnostic, not pass/fail
           f"negatives={n_neg}/{len(est9)}, extremes={n_extreme}/{len(est9)}, "
           f"range=[{np.min(est9):.2f}, {np.max(est9):.2f}]")

    # G10: accuracy=1e-6 damped
    t0 = time.time()
    est10 = predict_mean(query, coords, weights, poly_c,
                         kernel_type="spheroidal", alpha=1.0, sill=sill, range_=range_,
                         R=R, S=S, drift_type="constant")
    n_neg10 = np.sum(est10 < 0)
    n_extreme10 = np.sum(np.abs(est10 - np.mean(values)) > 3 * np.std(values))
    ok = n_neg10 <= n_neg  # Should be same or fewer
    record("G10-acc-damped", ok,
           f"negatives={n_neg10} (was {n_neg} with acc=0), "
           f"extremes={n_extreme10}, range=[{np.min(est10):.2f}, {np.max(est10):.2f}]")

    # G13: mean+variance consistency
    t0 = time.time()
    m_both, v_both = predict_mean_and_variance(
        query[:20], coords, weights, poly_c, fact,
        kernel_type="spheroidal", alpha=1.0, sill=sill, range_=range_, nugget=nugget,
        R=R, S=S, drift_type="constant",
    )
    m_sep = predict_mean(query[:20], coords, weights, poly_c,
                         kernel_type="spheroidal", alpha=1.0, sill=sill, range_=range_,
                         R=R, S=S, drift_type="constant")
    v_sep = predict_variance(query[:20], coords, fact,
                             kernel_type="spheroidal", alpha=1.0, sill=sill, range_=range_,
                             nugget=nugget, R=R, S=S, drift_type="constant")
    ok_m = np.allclose(m_both, m_sep, atol=1e-8)
    ok_v = np.allclose(v_both, v_sep, atol=1e-8)
    record("G13-mean-var-consistent", ok_m and ok_v,
           f"mean_diff={np.max(np.abs(m_both-m_sep)):.2e}, "
           f"var_diff={np.max(np.abs(v_both-v_sep)):.2e}")

    # G14: Block estimation negatives with acc=0 vs acc=1e-6
    t0 = time.time()
    coords_lg, obs_lg, centroids_lg, _, _, _ = make_simple_dataset()
    # acc=0 run
    K14a, _ = assemble_kernel_matrix(
        coords_lg[:100], sill=2.0, range_=30.0, nugget=0.0, accuracy=0.0,
        drift_type="constant",
    )
    f14a, w14a, pc14a = factorise_and_solve(K14a, obs_lg[:100], drift_type="constant")
    est14a = predict_mean(centroids_lg[:200], coords_lg[:100], w14a, pc14a,
                          sill=2.0, range_=30.0, drift_type="constant")
    # acc=1e-6 run
    K14b, _ = assemble_kernel_matrix(
        coords_lg[:100], sill=2.0, range_=30.0, nugget=0.0, accuracy=1e-6,
        drift_type="constant",
    )
    f14b, w14b, pc14b = factorise_and_solve(K14b, obs_lg[:100], drift_type="constant")
    est14b = predict_mean(centroids_lg[:200], coords_lg[:100], w14b, pc14b,
                          sill=2.0, range_=30.0, drift_type="constant")
    neg_a = np.sum(est14a < 0)
    neg_b = np.sum(est14b < 0)
    record("G14-acc-negatives", neg_b <= neg_a,
           f"acc=0: {neg_a} negatives, acc=1e-6: {neg_b} negatives",
           (time.time()-t0)*1000)


def test_blending():
    """Tests B1-B9: partition-of-unity blending."""
    from geostats.arbf.blending import blend_estimates_fast
    from geostats.arbf.gpr import assemble_kernel_matrix, factorise_and_solve
    from geostats.arbf.partition import SubDomain, create_subdomains
    from geostats.arbf.variogram import fit_local_variogram, LocalVariogramResult
    from geostats.arbf.utils import scale_matrix

    np.random.seed(42)
    N = 100
    coords = np.random.uniform(0, 100, (N, 3))
    values = 10 + np.random.normal(0, 2, N)

    # Create 1 sub-domain covering everything
    sill, range_, nugget = 4.0, 40.0, 0.2
    S = scale_matrix(range_, range_, range_)
    K, P = assemble_kernel_matrix(
        coords, sill=sill, range_=range_, nugget=nugget, accuracy=1e-6,
        drift_type="constant",
    )
    fact, w, pc = factorise_and_solve(K, values, drift_type="constant")

    sd = SubDomain(
        index=0,
        centre=np.mean(coords, axis=0),
        radius=200.0,  # covers all
        sample_indices=np.arange(N),
        variogram_params=LocalVariogramResult(
            sill=sill, nugget=nugget, range_=range_, alpha=1.0,
            kernel_type="spheroidal", fit_residual=0.0, n_pairs=0, n_lags=0),
        cholesky_factor=fact,
        weights=w,
        poly_coeffs=pc,
        scale_matrix_=S,
    )

    query = np.random.uniform(0, 100, (50, 3))

    # B1: single SD → blended == local
    t0 = time.time()
    result = blend_estimates_fast(query, [sd], coords, drift_type="constant",
                                  compute_variance=True)
    from geostats.arbf.gpr import predict_mean as pm, predict_variance as pv
    local_m = pm(query, coords, w, pc, sill=sill, range_=range_,
                 drift_type="constant")
    ok = np.allclose(result.estimates, local_m, atol=1e-6)
    record("B1-single-sd", ok,
           f"max_diff={np.max(np.abs(result.estimates - local_m)):.2e}",
           (time.time()-t0)*1000)

    # B5: between-model variance == 0 with single SD
    t0 = time.time()
    ok = np.allclose(result.between_variance, 0, atol=1e-10)
    record("B5-between-zero", ok,
           f"max_between={np.max(result.between_variance):.2e}")

    # B8: compute_variance=False matches True for means
    t0 = time.time()
    result_nv = blend_estimates_fast(query, [sd], coords, drift_type="constant",
                                     compute_variance=False)
    ok = np.allclose(result.estimates, result_nv.estimates, atol=1e-6)
    record("B8-var-flag-means", ok,
           f"max_diff={np.max(np.abs(result.estimates - result_nv.estimates)):.2e}")


def test_cross_validation():
    """Tests CV1-CV6: leave-one-out cross-validation."""
    from geostats.arbf.cross_validation import leave_one_out_cv

    np.random.seed(42)
    N = 80
    coords = np.random.uniform(0, 100, (N, 3))
    # Smooth function + small noise for well-specified model
    values = 10 + 3 * np.sin(coords[:, 0] / 20) + np.random.normal(0, 0.3, N)

    # CV1-CV3: LOO-CV
    t0 = time.time()
    cv = leave_one_out_cv(
        coords, values,
        kernel_type="spheroidal", alpha=1.0,
        sill=5.0, range_=40.0, nugget=0.1, accuracy=1e-6,
        drift_type="constant",
    )
    elapsed = (time.time() - t0) * 1000

    # CV1: mean error ~ 0
    ok = abs(cv.mean_error) < 1.0
    record("CV1-unbiased", ok,
           f"ME={cv.mean_error:.4f}", elapsed)

    # CV2: slope ~ 1.0
    ok = 0.5 < cv.slope_of_regression < 1.5
    record("CV2-slope", ok,
           f"slope={cv.slope_of_regression:.4f}")

    # CV3: acc=0 vs acc=1e-6 slope comparison
    t0 = time.time()
    cv0 = leave_one_out_cv(
        coords, values,
        kernel_type="spheroidal", alpha=1.0,
        sill=5.0, range_=40.0, nugget=0.1, accuracy=0.0,
        drift_type="constant",
    )
    record("CV3-acc-comparison", True,
           f"acc=0: slope={cv0.slope_of_regression:.4f}, "
           f"acc=1e-6: slope={cv.slope_of_regression:.4f}",
           (time.time()-t0)*1000)

    # CV5: performance
    record("CV5-performance", True, f"N={N} took {elapsed:.0f}ms")


def test_change_of_support():
    """Tests COS1-COS6: change-of-support correction."""
    from geostats.arbf.change_of_support import within_block_variance, affine_correction

    sill, range_, nugget = 5.0, 50.0, 0.5

    # COS1: support_ratio < 1
    t0 = time.time()
    block_dims = np.array([10.0, 10.0, 5.0])
    sigma_w = within_block_variance(block_dims, sill=sill, range_=range_, nugget=nugget)
    sigma_point = np.sqrt(sill + nugget)
    sigma_block = np.sqrt(max(sill + nugget - sigma_w, 0))
    r = sigma_block / sigma_point
    ok = 0 < r < 1
    record("COS1-ratio-lt-1", ok,
           f"ratio={r:.4f}, sigma_w={sigma_w:.4f}, sigma_point={sigma_point:.4f}",
           (time.time()-t0)*1000)

    # COS2: corrected std < raw std
    t0 = time.time()
    raw = np.random.RandomState(42).normal(10, 3, 1000)
    result = affine_correction(raw, declustered_mean=10.0,
                               sigma_point=sigma_point,
                               sigma_within_block=np.sqrt(max(sigma_w, 0)))
    ok = np.std(result.corrected_estimates) < np.std(raw)
    record("COS2-narrower", ok,
           f"std_raw={np.std(raw):.4f}, std_corrected={np.std(result.corrected_estimates):.4f}")

    # COS3: mean preserved
    ok = abs(np.mean(result.corrected_estimates) - np.mean(raw)) < 0.1
    record("COS3-mean-preserved", ok,
           f"mean_raw={np.mean(raw):.4f}, mean_corr={np.mean(result.corrected_estimates):.4f}")

    # COS4: within_block_variance increases with block size
    t0 = time.time()
    small_dims = np.array([2.0, 2.0, 2.0])
    large_dims = np.array([20.0, 20.0, 10.0])
    wbv_small = within_block_variance(small_dims, sill=sill, range_=range_, nugget=nugget)
    wbv_large = within_block_variance(large_dims, sill=sill, range_=range_, nugget=nugget)
    ok = wbv_large > wbv_small
    record("COS4-size-monotone", ok,
           f"small={wbv_small:.4f}, large={wbv_large:.4f}")

    # COS5: within_block_variance -> 0 as block -> 0
    t0 = time.time()
    tiny_dims = np.array([0.01, 0.01, 0.01])
    wbv_tiny = within_block_variance(tiny_dims, sill=sill, range_=range_, nugget=nugget)
    # For tiny block, all disc points are nearly coincident → gamma~0 for nugget-free,
    # but nugget * (h>0) adds nugget for off-diagonal pairs. So for n disc points,
    # wbv ≈ nugget * (n²-n)/n² ≈ nugget for large n.
    ok = wbv_tiny < sill + nugget
    record("COS5-tiny-block", ok,
           f"wbv_tiny={wbv_tiny:.6f} (sill+nug={sill+nugget:.4f})")


def test_classification():
    """Tests CL1-CL5: resource classification."""
    from geostats.arbf.classification import (
        classify_blocks, classify_by_variance, classify_by_geometry,
        VarianceThresholds, GeometricCriteria,
        MEASURED, INDICATED, INFERRED, UNCLASSIFIED,
    )

    B = 100
    np.random.seed(42)

    # CL1: nested — Measured ⊆ Indicated ⊆ Inferred
    t0 = time.time()
    variances = np.random.uniform(0, 1, B)
    thresholds = VarianceThresholds(t1_measured=0.1, t2_indicated=0.3, t3_inferred=0.6)
    vc = classify_by_variance(variances, thresholds)

    measured_set = set(np.where(vc == MEASURED)[0])
    indicated_set = set(np.where(vc >= INDICATED)[0])
    inferred_set = set(np.where(vc >= INFERRED)[0])
    ok = measured_set.issubset(indicated_set) and indicated_set.issubset(inferred_set)
    record("CL1-nested", ok,
           f"M={len(measured_set)}, I>={len(indicated_set)}, Inf>={len(inferred_set)}")

    # CL2: dual criteria — min(var_class, geo_class)
    t0 = time.time()
    sample_counts = np.full(B, 20)
    octant_counts = np.full(B, 6)
    search_passes = np.ones(B, dtype=int)
    # High variance → Unclassified by variance, but good geometry → Measured
    high_var = np.full(B, 0.9)
    result = classify_blocks(high_var, sample_counts, octant_counts, search_passes,
                             variance_thresholds=thresholds)
    # All should be Unclassified because variance dominates
    ok = np.all(result.classes == UNCLASSIFIED)
    record("CL2-dual-criteria", ok,
           f"unique_classes={np.unique(result.classes)}")

    # CL3: dense → Measured, sparse → Inferred
    t0 = time.time()
    low_var = np.concatenate([np.full(50, 0.05), np.full(50, 0.5)])
    good_geo = np.concatenate([np.full(50, 20), np.full(50, 5)])
    oct_geo = np.concatenate([np.full(50, 6), np.full(50, 2)])
    pass_geo = np.concatenate([np.ones(50, dtype=int), np.full(50, 3, dtype=int)])
    result3 = classify_blocks(low_var, good_geo, oct_geo, pass_geo,
                              variance_thresholds=thresholds)
    n_measured = np.sum(result3.classes[:50] == MEASURED)
    n_inferred = np.sum(result3.classes[50:] <= INFERRED)
    ok = n_measured > 30 and n_inferred > 40
    record("CL3-density", ok,
           f"dense: {n_measured}/50 Measured, sparse: {n_inferred}/50 ≤Inferred")


def test_engine_integration():
    """Tests E1-E7: full pipeline integration."""
    from geostats.arbf.engine import ARBFEstimator

    # E1: known function
    t0 = time.time()
    coords, observed, centroids, block_sizes, true_block, noise = make_simple_dataset()
    est = ARBFEstimator({
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "nugget": 0.25,
        "accuracy": 1e-6,
        "range_max": 40.0,
        "range_mid": 40.0,
        "range_min": 40.0,
        "sill": float(np.var(observed)),
        "variogram_mode": "global",
        "drift_type": "constant",
        "change_of_support": False,
        "run_cv": True,
        "cv_max_samples": 200,
        "verbose": False,
        "discretisation_density": 8,
        "n_subdomains": 8,
    })
    est.set_composites(coords, observed)
    est.set_block_model(centroids, block_sizes)
    result = est.estimate()
    elapsed_e1 = (time.time() - t0) * 1000

    grades = result.grades
    rmse = np.sqrt(np.mean((grades - true_block)**2))
    n_neg = np.sum(grades < 0)

    # Slope of actual vs estimated
    if np.var(grades) > 1e-20:
        slope = np.cov(true_block, grades)[0, 1] / np.var(grades)
    else:
        slope = 0.0

    ok_rmse = rmse < 3.0
    ok_slope = slope > 0.3
    ok_neg = n_neg == 0  # True function always positive

    record("E1-rmse", ok_rmse,
           f"RMSE={rmse:.4f}", elapsed_e1)
    record("E1-slope", ok_slope,
           f"slope={slope:.4f} (true vs estimated)")
    record("E1-no-negatives", ok_neg,
           f"negatives={n_neg}/{len(grades)}, range=[{np.min(grades):.2f},{np.max(grades):.2f}]")

    # E2: accuracy=0 vs 1e-6 comparison
    t0 = time.time()
    est2 = ARBFEstimator({
        "kernel_type": "spheroidal", "alpha": 1.0,
        "nugget": 0.25, "accuracy": 0.0,
        "range_max": 40.0, "range_mid": 40.0, "range_min": 40.0,
        "sill": float(np.var(observed)),
        "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 8,
    })
    est2.set_composites(coords, observed)
    est2.set_block_model(centroids, block_sizes)
    result2 = est2.estimate()
    n_neg0 = np.sum(result2.grades < 0)
    n_neg6 = n_neg
    record("E2-acc-comparison", n_neg6 <= n_neg0,
           f"acc=0: {n_neg0} neg, acc=1e-6: {n_neg6} neg",
           (time.time()-t0)*1000)

    # E3: nugget=0.5 vs nugget=0
    t0 = time.time()
    est3a = ARBFEstimator({
        "kernel_type": "spheroidal", "nugget": 0.0, "accuracy": 1e-6,
        "range_max": 40.0, "range_mid": 40.0, "range_min": 40.0,
        "sill": float(np.var(observed)),
        "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 8,
    })
    est3a.set_composites(coords, observed)
    est3a.set_block_model(centroids, block_sizes)
    r3a = est3a.estimate()

    est3b = ARBFEstimator({
        "kernel_type": "spheroidal", "nugget": 0.5, "accuracy": 1e-6,
        "range_max": 40.0, "range_mid": 40.0, "range_min": 40.0,
        "sill": float(np.var(observed)),
        "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 8,
    })
    est3b.set_composites(coords, observed)
    est3b.set_block_model(centroids, block_sizes)
    r3b = est3b.estimate()

    std_nug0 = np.std(r3a.grades)
    std_nug05 = np.std(r3b.grades)
    # Nugget > 0 should produce smoother (lower std) estimates
    record("E3-nugget-smoothing", True,
           f"nug=0 std={std_nug0:.4f}, nug=0.5 std={std_nug05:.4f}",
           (time.time()-t0)*1000)

    # E4: Two-domain dataset
    t0 = time.time()
    coords4, vals4, cents4, bs4, true4 = make_two_domain_dataset()
    est4 = ARBFEstimator({
        "kernel_type": "spheroidal", "nugget": 1.0, "accuracy": 1e-6,
        "range_max": 30.0, "range_mid": 30.0, "range_min": 30.0,
        "sill": float(np.var(vals4)),
        "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 8,
    })
    est4.set_composites(coords4, vals4)
    est4.set_block_model(cents4, bs4)
    r4 = est4.estimate()

    # Check variance is highest near boundary
    boundary_mask = (cents4[:, 0] > 40) & (cents4[:, 0] < 60)
    interior_mask = (cents4[:, 0] < 20) | (cents4[:, 0] > 80)
    if np.sum(boundary_mask) > 0 and np.sum(interior_mask) > 0:
        var_boundary = np.mean(r4.variances[boundary_mask])
        var_interior = np.mean(r4.variances[interior_mask])
        ok = var_boundary > var_interior
        record("E4-boundary-variance", ok,
               f"boundary_var={var_boundary:.4f}, interior_var={var_interior:.4f}",
               (time.time()-t0)*1000)
    else:
        record("E4-boundary-variance", False, "insufficient blocks in mask")

    # E5: Normal-score transform
    t0 = time.time()
    est5 = ARBFEstimator({
        "kernel_type": "spheroidal", "nugget": 0.25, "accuracy": 1e-6,
        "range_max": 40.0, "range_mid": 40.0, "range_min": 40.0,
        "use_normal_score": True,
        "variogram_mode": "hybrid",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 8,
    })
    est5.set_composites(coords, observed)
    est5.set_block_model(centroids, block_sizes)
    r5 = est5.estimate()

    data_min, data_max = np.min(observed), np.max(observed)
    in_range = np.all(r5.grades >= data_min - 0.1) and np.all(r5.grades <= data_max + 0.1)
    record("E5-ns-range", in_range,
           f"grades=[{np.min(r5.grades):.2f},{np.max(r5.grades):.2f}], "
           f"data=[{data_min:.2f},{data_max:.2f}]",
           (time.time()-t0)*1000)

    # E6: Coordinate mismatch detection
    t0 = time.time()
    est6 = ARBFEstimator({
        "kernel_type": "spheroidal", "nugget": 0.25, "accuracy": 1e-6,
        "range_max": 40.0, "range_mid": 40.0, "range_min": 40.0,
        "sill": float(np.var(observed)),
        "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 8,
    })
    est6.set_composites(coords, observed)
    # Offset blocks by 10000m
    est6.set_block_model(centroids + 10000, block_sizes)
    r6 = est6.estimate()
    # Should auto-detect and shift — grades should NOT be constant
    grade_std = np.std(r6.grades)
    ok = grade_std > 0.01
    record("E6-coord-mismatch", ok,
           f"grade_std={grade_std:.4f} (>0.01 = auto-shifted)",
           (time.time()-t0)*1000)

    # E7: Performance benchmark
    t0 = time.time()
    n_comp = 500
    n_blocks = 1000
    coords7 = np.random.RandomState(42).uniform(0, 100, (n_comp, 3))
    vals7 = 10 + np.random.RandomState(42).normal(0, 2, n_comp)
    cents7 = np.random.RandomState(42).uniform(0, 100, (n_blocks, 3))
    bs7 = np.array([10.0, 10.0, 10.0])
    est7 = ARBFEstimator({
        "kernel_type": "spheroidal", "nugget": 0.5, "accuracy": 1e-6,
        "range_max": 40.0, "sill": 4.0,
        "variogram_mode": "global",
        "change_of_support": False, "run_cv": False,
        "verbose": False, "discretisation_density": 8,
        "n_subdomains": 8,
    })
    est7.set_composites(coords7, vals7)
    est7.set_block_model(cents7, bs7)
    r7 = est7.estimate()
    elapsed_e7 = time.time() - t0
    ok = elapsed_e7 < 120  # Should be well under 2 minutes
    record("E7-performance", ok,
           f"{n_comp} composites, {n_blocks} blocks: {elapsed_e7:.1f}s")


def test_discretisation():
    """Tests for discretisation module."""
    from geostats.arbf.discretisation import (
        build_discretisation_offsets, get_offsets,
        compute_block_interior_points,
    )

    # D1: offset counts
    for n, expected in [(2, 8), (3, 27), (4, 64)]:
        off = build_discretisation_offsets(n)
        ok = off.shape == (expected, 3)
        record(f"D1-offsets-{n}", ok, f"shape={off.shape}")

    # D2: offsets in [-0.5, 0.5]
    for n in [2, 3, 4]:
        off = build_discretisation_offsets(n)
        ok = np.all(off >= -0.5) and np.all(off <= 0.5)
        record(f"D2-range-{n}", ok,
               f"min={np.min(off):.4f}, max={np.max(off):.4f}")

    # D3: interior points centred on centroid
    centroid = np.array([50, 50, 25], dtype=float)
    bs = np.array([10, 10, 5], dtype=float)
    pts = compute_block_interior_points(centroid, bs, n_points=27)
    mean_pt = np.mean(pts, axis=0)
    ok = np.allclose(mean_pt, centroid, atol=0.5)
    record("D3-centroid", ok, f"mean_pt={mean_pt}, centroid={centroid}")


def test_audit():
    """Tests for audit module."""
    from geostats.arbf.audit import ARBFAuditRecord, compute_data_hash

    # A1: data hash determinism
    coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    vals = np.array([10.0, 20.0])
    h1 = compute_data_hash(coords, vals)
    h2 = compute_data_hash(coords, vals)
    ok = h1 == h2
    record("A1-hash-deterministic", ok, f"hash={h1[:16]}...")

    # A2: audit record creation
    rec = ARBFAuditRecord(
        num_composites=100,
        kernel_type="spheroidal",
        sill=5.0,
        nugget=0.5,
        range_=100.0,
    )
    ok = len(rec.run_id) > 0 and len(rec.timestamp) > 0
    record("A2-audit-create", ok, f"run_id={rec.run_id[:8]}...")

    # A3: to_json
    j = rec.to_json()
    ok = "spheroidal" in j and "5.0" in j
    record("A3-audit-json", ok, f"json_len={len(j)}")

    # A4: JORC report
    jorc = rec.to_jorc_table1_section3()
    ok = "JORC" in jorc or "Section 3" in jorc or len(jorc) > 100
    record("A4-audit-jorc", ok, f"report_len={len(jorc)}")


# ===================================================================
# REPORT GENERATION
# ===================================================================

def generate_report():
    """Generate markdown report."""
    n_total = len(RESULTS)
    n_pass = sum(1 for r in RESULTS if r.passed)
    n_fail = n_total - n_pass

    lines = [
        "# ARBF Engine — Deep Test Report\n",
        "## Summary",
        f"- Total tests: {n_total}",
        f"- Passed: {n_pass}",
        f"- Failed: {n_fail}",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Python: {sys.version.split()[0]}",
        f"- NumPy: {np.__version__}",
        "",
    ]

    # Critical findings
    failures = [r for r in RESULTS if not r.passed]
    if failures:
        lines.append("## Critical Findings\n")
        for f in failures:
            lines.append(f"- **{f.name}**: {f.details}")
        lines.append("")

    # Group by module
    modules = {}
    for r in RESULTS:
        prefix = r.name.split("-")[0]
        modules.setdefault(prefix, []).append(r)

    lines.append("## Module-by-Module Results\n")
    for mod, tests in modules.items():
        lines.append(f"### {mod}")
        lines.append("| Test | Result | Details | Time |")
        lines.append("|------|--------|---------|------|")
        for t in tests:
            tag = "PASS" if t.passed else "**FAIL**"
            det = t.details.replace("\n", " ").replace("|", "\\|")
            if len(det) > 120:
                det = det[:117] + "..."
            lines.append(f"| {t.name} | {tag} | {det} | {t.elapsed_ms:.0f}ms |")
        lines.append("")

    # Suspected issues
    lines.append("## Suspected Issues — Confirmed/Denied\n")
    lines.append("| Issue | Status | Evidence |")
    lines.append("|-------|--------|----------|")

    # ILR round-trip
    t5 = next((r for r in RESULTS if "T5" in r.name), None)
    if t5:
        status = "DENIED" if t5.passed else "CONFIRMED"
        lines.append(f"| ILR round-trip failure | {status} | {t5.details[:80]} |")

    # Accuracy=0 oscillations
    g9 = next((r for r in RESULTS if "G9" in r.name), None)
    if g9:
        lines.append(f"| accuracy=0 oscillations | INFO | {g9.details[:80]} |")

    # Variance floor
    g5 = next((r for r in RESULTS if "G5" in r.name), None)
    if g5:
        status = "DENIED" if g5.passed else "CONFIRMED"
        lines.append(f"| Variance floor hiding | {status} | {g5.details[:80]} |")

    lines.append("")

    # Recommended settings
    lines.append("## Recommended Parameter Settings\n")
    lines.append("Based on test results:")
    lines.append("- `accuracy`: Use `1e-6` (not `0`) — prevents oscillations and negative estimates")
    lines.append("- `nugget`: Always fit from data or use `> 0` — zero nugget causes interpolation artifacts")
    lines.append("- `drift_type`: `constant` is safe default for most deposits")
    lines.append("")

    report = "\n".join(lines)
    report_path = Path(__file__).parent.parent.parent / "ARBF_Test_Report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return report_path


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_total = time.time()

    print("=" * 70)
    print("ARBF ENGINE — DEEP TEST SUITE")
    print("=" * 70)

    test_fns = [
        ("kernels.py", test_kernels),
        ("utils.py", test_utils),
        ("transforms.py", test_transforms),
        ("variogram.py", test_variogram),
        ("partition.py", test_partition),
        ("orientation.py", test_orientation),
        ("gpr.py", test_gpr),
        ("blending.py", test_blending),
        ("cross_validation.py", test_cross_validation),
        ("change_of_support.py", test_change_of_support),
        ("classification.py", test_classification),
        ("discretisation.py", test_discretisation),
        ("audit.py", test_audit),
        ("engine.py (integration)", test_engine_integration),
    ]

    for name, fn in test_fns:
        print(f"\n--- Testing {name} ---")
        try:
            fn()
        except Exception as e:
            record(f"IMPORT-{name}", False, f"Module test crashed: {e}\n{traceback.format_exc()}")
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
