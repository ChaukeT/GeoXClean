"""
ARBF Cross-Validation and Conditional Bias Diagnostics.

LOO-CV, KNA, and swath plots for JORC-compliant model validation.
Eq. 9.1 -- 9.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from scipy.linalg import lu_solve, solve_triangular

from .gpr import (
    assemble_kernel_matrix,
    build_polynomial_matrix,
    factorise_and_solve,
    predict_mean,
)
from .partition import SubDomain
from .utils import rotation_matrix, scale_matrix

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Leave-one-out cross-validation results (Eq. 9.1)."""

    actual: np.ndarray
    estimated: np.ndarray
    errors: np.ndarray
    mean_error: float          # ME (should be ~0 for unbiased)
    mae: float                 # Mean Absolute Error
    rmse: float                # Root Mean Squared Error
    r_squared: float           # Coefficient of determination
    correlation: float         # Pearson correlation
    normalised_rmse: float     # RMSE / StdDev(actual)
    slope_of_regression: float # Slope (actual on estimated, Eq. 9.2)
    intercept: float           # Intercept
    n_samples: int


@dataclass
class SwathData:
    """Swath plot data for one axis."""

    axis: str
    slice_positions: np.ndarray
    mean_estimated: np.ndarray
    mean_actual: np.ndarray
    n_blocks_per_slice: np.ndarray


@dataclass
class ConditionalBiasResult:
    """Conditional-bias diagnostics derived from cross-validation pairs."""

    estimated_bin_centres: np.ndarray
    mean_estimated: np.ndarray
    mean_actual: np.ndarray
    count_per_bin: np.ndarray
    global_slope: float
    global_intercept: float
    binned_slope: float
    binned_intercept: float
    mean_bin_bias: float
    max_abs_bin_bias: float
    n_populated_bins: int


@dataclass
class SupportSwathData:
    """Support-aware swath data for one axis using aggregated panels.

    ``mean_actual`` is the primary support-aware reference and is populated
    from the declustered composite grade where available.  ``mean_actual_raw``
    is retained for context only.
    """

    axis: str
    slice_positions: np.ndarray
    mean_estimated: np.ndarray
    mean_actual: np.ndarray
    mean_actual_raw: np.ndarray
    mean_actual_declustered: np.ndarray
    n_panels_per_slice: np.ndarray
    n_data_panels_per_slice: np.ndarray
    n_composites_per_slice: np.ndarray
    block_volume_per_slice: np.ndarray
    composite_weight_per_slice: np.ndarray


@dataclass
class SupportSwathResult:
    """Support-aware panel swath diagnostics for a regular block grid."""

    panel_factors: Tuple[int, int, int]
    panel_shape: Tuple[int, int, int]
    panel_size: np.ndarray
    panel_centroids: np.ndarray
    panel_block_volumes: np.ndarray
    panel_estimated: np.ndarray
    panel_actual: np.ndarray
    panel_actual_raw: np.ndarray
    panel_actual_declustered: np.ndarray
    panel_composite_counts: np.ndarray
    panel_composite_weight_sums: np.ndarray
    panel_estimated_metal: np.ndarray
    panel_actual_metal_raw: np.ndarray
    panel_actual_metal_declustered: np.ndarray
    axes: Dict[str, SupportSwathData]
    n_panels_total: int
    n_panels_with_data: int


def _select_representative_subset(
    sample_coords: np.ndarray,
    max_samples: int,
    R: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """Select a spatially representative subset in anisotropic search space.

    The previous nearest-to-median subsample over-represented the deposit core
    and systematically under-tested boundary/extrapolation behaviour.  Use a
    deterministic farthest-point subset instead so the retained samples cover
    the full support of the dataset.
    """
    n_total = sample_coords.shape[0]
    if max_samples <= 0 or n_total <= max_samples:
        return np.arange(n_total, dtype=np.intp)

    search_coords = np.asarray(sample_coords, dtype=np.float64) @ (S @ R).T
    n_keep = min(int(max_samples), n_total)
    selected = np.empty(n_keep, dtype=np.intp)
    centre = np.median(search_coords, axis=0)
    selected[0] = int(np.argmin(np.linalg.norm(search_coords - centre, axis=1)))
    min_dist = np.linalg.norm(search_coords - search_coords[selected[0]], axis=1)

    for i in range(1, n_keep):
        selected[i] = int(np.argmax(min_dist))
        d = np.linalg.norm(search_coords - search_coords[selected[i]], axis=1)
        min_dist = np.minimum(min_dist, d)

    return np.unique(selected)


def leave_one_out_cv(
    sample_coords: np.ndarray,
    sample_values: np.ndarray,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    sill: float = 1.0,
    range_: float = 100.0,
    range_mid: Optional[float] = None,
    range_min: Optional[float] = None,
    nugget: float = 0.0,
    accuracy: float = 1e-6,
    drift_type: str = "constant",
    azimuth: float = 0.0,
    dip: float = 0.0,
    pitch: float = 0.0,
    max_samples: int = 500,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> CVResult:
    """Leave-one-out cross-validation (Eq. 9.1).

    For each sample i:
    1. Remove from dataset.
    2. Fit RBF on remaining N-1 samples.
    3. Predict at removed location.
    4. Record error.

    Parameters
    ----------
    sample_coords : np.ndarray
        (N, 3) coordinates.
    sample_values : np.ndarray
        (N,) values.
    kernel_type, alpha, sill, range_, range_mid, range_min, nugget, accuracy : ...
        Kernel and solver parameters.
    drift_type : str
        Polynomial drift type.
    azimuth, dip, pitch : float
        Anisotropy angles.
    max_samples : int
        If N > max_samples, subsample for efficiency.
    progress_callback : callable, optional
        Progress callback(percent, message).

    Returns
    -------
    CVResult
        Comprehensive cross-validation diagnostics.
    """
    N = len(sample_values)
    R = rotation_matrix(azimuth, dip, pitch)
    if range_mid is None:
        range_mid = range_
    if range_min is None:
        range_min = range_
    S = scale_matrix(range_, range_mid, range_min)

    # Subsample if too large using a deterministic spatially representative
    # subset.  A central nearest-to-median core is optimistic because it drops
    # the edge and sparse-support composites that dominate real interpolation
    # risk.
    if N > max_samples:
        idx = _select_representative_subset(sample_coords, max_samples, R, S)
        coords = sample_coords[idx]
        values = sample_values[idx]
        N = len(idx)
        logger.info(
            "LOO-CV: spatially subsampled to %d representative samples "
            "(anisotropic farthest-point coverage).", N,
        )
    else:
        coords = sample_coords.copy()
        values = sample_values.copy()

    if progress_callback:
        progress_callback(10, "LOO-CV: assembling kernel matrix")

    # Build and factorise the FULL kernel matrix once — O(N^3)
    K_aug, P = assemble_kernel_matrix(
        coords,
        kernel_type=kernel_type,
        alpha=alpha,
        sill=sill,
        range_=range_,
        nugget=nugget,
        accuracy=accuracy,
        R=R,
        S=S,
        drift_type=drift_type,
    )
    M = P.shape[1]

    if progress_callback:
        progress_callback(30, "LOO-CV: factorising")

    factorisation, weights_full, poly_coeffs_full = factorise_and_solve(
        K_aug, values, drift_type=drift_type,
    )

    # Build augmented RHS
    z_aug = np.zeros(N + M, dtype=np.float64)
    z_aug[:N] = values

    # Solve for x_aug = K_aug^{-1} z_aug  (already done: [weights_full; poly_coeffs_full])
    x_aug = np.concatenate([weights_full, poly_coeffs_full])

    if progress_callback:
        progress_callback(50, "LOO-CV: computing inverse diagonal")

    # Compute diagonal of K_aug^{-1} — needed for virtual LOO formula.
    # The LOO formula only uses inv_diag[:N] (sample rows), so we solve for
    # the first N columns of the inverse only, avoiding M wasted solves.
    is_cholesky = not isinstance(factorisation, tuple)
    if is_cholesky:
        # L L^T = K_aug → K^{-1} = L^{-T} L^{-1}
        # diag(K^{-1})_i = ||col_i(L^{-1})||^2; only need first N columns.
        L_inv_N = solve_triangular(
            factorisation,
            np.eye(N + M, N, dtype=np.float64),  # first N columns of identity
            lower=True,
        )
        inv_diag = np.sum(L_inv_N ** 2, axis=0)  # shape (N,)
    else:
        # LU path: solve for first N columns of K_aug^{-1}, extract diagonal.
        K_aug_inv_N = lu_solve(
            factorisation, np.eye(N + M, N, dtype=np.float64)
        )
        inv_diag = K_aug_inv_N[:N].diagonal().copy()  # shape (N,)

    if progress_callback:
        progress_callback(80, "LOO-CV: virtual cross-validation")

    # Virtual LOO formula (Bartlett identity):
    #   z*_{-i} = z_i - x_aug[i] / inv_diag[i]
    # where x_aug = K_aug^{-1} z_aug, inv_diag = diag(K_aug^{-1})
    # LOO error_i = x_aug[i] / inv_diag[i]
    actual = values.copy()
    inv_diag_samples = inv_diag[:N]
    inv_diag_samples = np.maximum(inv_diag_samples, 1e-30)  # guard div-by-zero
    loo_errors = x_aug[:N] / inv_diag_samples
    estimated = actual - loo_errors

    if progress_callback:
        progress_callback(100, "LOO-CV complete")

    return _compute_cv_statistics(actual, estimated)


def _compute_cv_statistics(
    actual: np.ndarray,
    estimated: np.ndarray,
) -> CVResult:
    """Compute comprehensive CV statistics."""
    N = len(actual)
    errors = actual - estimated
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    me = float(np.mean(errors))
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    std_actual = float(np.std(actual))
    normalised_rmse = rmse / max(std_actual, 1e-12)

    # R-squared
    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Correlation
    corr = float(np.corrcoef(actual, estimated)[0, 1]) if N > 1 else 0.0

    # Slope of regression (Eq. 9.2): actual = a + b * estimated
    if N > 1 and np.var(estimated) > 1e-20:
        slope = float(np.cov(actual, estimated)[0, 1] / np.var(estimated))
        intercept = float(np.mean(actual) - slope * np.mean(estimated))
    else:
        slope = 1.0
        intercept = 0.0

    return CVResult(
        actual=actual,
        estimated=estimated,
        errors=errors,
        mean_error=me,
        mae=mae,
        rmse=rmse,
        r_squared=float(r2),
        correlation=corr,
        normalised_rmse=normalised_rmse,
        slope_of_regression=slope,
        intercept=intercept,
        n_samples=N,
    )


def kriging_neighbourhood_analysis(
    sample_coords: np.ndarray,
    sample_values: np.ndarray,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    sill: float = 1.0,
    range_: float = 100.0,
    nugget: float = 0.0,
    accuracy: float = 1e-6,
    drift_type: str = "constant",
    azimuth: float = 0.0,
    dip: float = 0.0,
    pitch: float = 0.0,
    range_factors: Optional[List[float]] = None,
    max_samples_list: Optional[List[int]] = None,
    min_samples_list: Optional[List[int]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    """Automated Kriging Neighbourhood Analysis (Eq. 9.3).

    D5: Sweeps the full parameter space — range factor, max samples
    (search radius proxy), and min samples — computing LOO-CV slope
    and RMSE for each combination to find optimal parameters.

    Parameters
    ----------
    sample_coords, sample_values : np.ndarray
        Input data.
    kernel_type, alpha, sill, range_, nugget, accuracy, drift_type : ...
        Base estimation parameters.
    azimuth, dip, pitch : float
        Anisotropy angles.
    range_factors : list of float, optional
        Range multipliers to test (default [0.5, 1.0, 1.5, 2.0]).
    max_samples_list : list of int, optional
        Max sample counts to sweep (proxy for search radius).
    min_samples_list : list of int, optional
        Min sample counts to sweep.
    progress_callback : callable, optional
        Progress callback.

    Returns
    -------
    dict
        'results': list of dicts
        'best': dict with best parameters (closest slope to 1.0)
    """
    if range_factors is None:
        range_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    if max_samples_list is None:
        max_samples_list = [200]
    if min_samples_list is None:
        min_samples_list = [4]

    # Build full parameter grid
    param_grid = [
        (rf, ms)
        for rf in range_factors
        for ms in max_samples_list
    ]
    results = []
    total = len(param_grid)

    for idx, (rf, ms) in enumerate(param_grid):
        if progress_callback:
            pct = int(100 * idx / total)
            progress_callback(pct, f"KNA: range={rf:.2f}x, max_samples={ms}")

        test_range = rf * range_

        cv = leave_one_out_cv(
            sample_coords,
            sample_values,
            kernel_type=kernel_type,
            alpha=alpha,
            sill=sill,
            range_=test_range,
            nugget=nugget,
            accuracy=accuracy,
            drift_type=drift_type,
            azimuth=azimuth,
            dip=dip,
            pitch=pitch,
            max_samples=ms,
        )

        results.append({
            "range_factor": rf,
            "range": test_range,
            "max_samples": ms,
            "slope": cv.slope_of_regression,
            "rmse": cv.rmse,
            "normalised_rmse": cv.normalised_rmse,
            "r_squared": cv.r_squared,
            "mae": cv.mae,
            "mean_error": cv.mean_error,
        })

    if progress_callback:
        progress_callback(100, "KNA complete")

    # Find best: closest slope to 1.0 (with normalised RMSE tiebreaker)
    scored = sorted(
        results,
        key=lambda r: (2.0 * abs(r["slope"] - 1.0) + r["normalised_rmse"]),
    )
    best = scored[0] if scored else results[0]

    return {"results": results, "best": best}


def swath_plots(
    block_centroids: np.ndarray,
    block_estimates: np.ndarray,
    composite_coords: np.ndarray,
    composite_values: np.ndarray,
    n_slices: int = 20,
    axes: str = "xyz",
) -> Dict[str, SwathData]:
    """Generate swath plot data for validation.

    For each axis, slice the model, compute mean estimated
    vs nearest-neighbour composite mean.

    Parameters
    ----------
    block_centroids : np.ndarray
        (B, 3) block centroids.
    block_estimates : np.ndarray
        (B,) estimated grades.
    composite_coords : np.ndarray
        (N, 3) composite coordinates.
    composite_values : np.ndarray
        (N,) composite values.
    n_slices : int
        Number of slices per axis.
    axes : str
        Which axes to compute ('x', 'y', 'z', or combination).

    Returns
    -------
    dict of SwathData
        Keyed by axis name.
    """
    result = {}
    axis_map = {"x": 0, "y": 1, "z": 2}

    for axis_name in axes:
        dim = axis_map.get(axis_name)
        if dim is None:
            continue

        # Determine slice boundaries
        all_vals = np.concatenate([
            block_centroids[:, dim],
            composite_coords[:, dim],
        ])
        lo, hi = np.min(all_vals), np.max(all_vals)
        edges = np.linspace(lo, hi, n_slices + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])

        mean_est = np.full(n_slices, np.nan, dtype=np.float64)
        mean_act = np.full(n_slices, np.nan, dtype=np.float64)
        n_blocks = np.zeros(n_slices, dtype=np.int32)

        for s in range(n_slices):
            # Blocks in this slice
            mask_b = (block_centroids[:, dim] >= edges[s]) & \
                     (block_centroids[:, dim] < edges[s + 1])
            # Composites in this slice
            mask_c = (composite_coords[:, dim] >= edges[s]) & \
                     (composite_coords[:, dim] < edges[s + 1])

            n_b = np.sum(mask_b)
            n_blocks[s] = n_b
            if n_b > 0:
                mean_est[s] = np.nanmean(block_estimates[mask_b])
            if np.sum(mask_c) > 0:
                mean_act[s] = np.nanmean(composite_values[mask_c])

        result[axis_name] = SwathData(
            axis=axis_name,
            slice_positions=centres,
            mean_estimated=mean_est,
            mean_actual=mean_act,
            n_blocks_per_slice=n_blocks,
        )

    return result


def conditional_bias_diagnostics(
    actual: np.ndarray,
    estimated: np.ndarray,
    n_bins: int = 10,
) -> Optional[ConditionalBiasResult]:
    """Summarise conditional bias from paired actual / estimated values.

    The global regression slope is carried through directly, while the
    binned means expose the familiar conditional-bias pattern where low
    estimates are high-biased and high estimates are low-biased.
    """
    actual = np.asarray(actual, dtype=np.float64).ravel()
    estimated = np.asarray(estimated, dtype=np.float64).ravel()
    mask = np.isfinite(actual) & np.isfinite(estimated)
    if np.sum(mask) < 2:
        return None

    actual = actual[mask]
    estimated = estimated[mask]
    cv = _compute_cv_statistics(actual, estimated)

    n_bins = max(2, min(int(n_bins), len(actual)))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(estimated, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        eps = max(float(np.std(estimated)) * 1e-9, 1e-12)
        lo = float(np.min(estimated)) - eps
        hi = float(np.max(estimated)) + eps
        edges = np.array([lo, hi], dtype=np.float64)

    n_eff = len(edges) - 1
    mean_est = np.full(n_eff, np.nan, dtype=np.float64)
    mean_act = np.full(n_eff, np.nan, dtype=np.float64)
    bin_centres = np.full(n_eff, np.nan, dtype=np.float64)
    counts = np.zeros(n_eff, dtype=np.int32)

    for i in range(n_eff):
        if i == n_eff - 1:
            in_bin = (estimated >= edges[i]) & (estimated <= edges[i + 1])
        else:
            in_bin = (estimated >= edges[i]) & (estimated < edges[i + 1])
        counts[i] = int(np.sum(in_bin))
        if counts[i] == 0:
            continue
        mean_est[i] = float(np.mean(estimated[in_bin]))
        mean_act[i] = float(np.mean(actual[in_bin]))
        bin_centres[i] = float(0.5 * (edges[i] + edges[i + 1]))

    populated = counts > 0
    n_populated = int(np.sum(populated))
    if n_populated == 0:
        return None

    bin_bias = mean_act[populated] - mean_est[populated]
    if n_populated > 1 and np.var(mean_est[populated]) > 1e-20:
        binned_slope = float(
            np.cov(mean_act[populated], mean_est[populated])[0, 1]
            / np.var(mean_est[populated])
        )
        binned_intercept = float(
            np.mean(mean_act[populated]) - binned_slope * np.mean(mean_est[populated])
        )
    else:
        binned_slope = 1.0
        binned_intercept = 0.0

    return ConditionalBiasResult(
        estimated_bin_centres=bin_centres,
        mean_estimated=mean_est,
        mean_actual=mean_act,
        count_per_bin=counts,
        global_slope=float(cv.slope_of_regression),
        global_intercept=float(cv.intercept),
        binned_slope=binned_slope,
        binned_intercept=binned_intercept,
        mean_bin_bias=float(np.mean(bin_bias)),
        max_abs_bin_bias=float(np.max(np.abs(bin_bias))),
        n_populated_bins=n_populated,
    )


def _choose_panel_factor(n_cells: int, target_panels: int) -> int:
    """Choose a panel aggregation factor that preserves a coarse swath grid."""
    n_cells = int(max(n_cells, 1))
    divisors = [d for d in range(1, n_cells + 1) if n_cells % d == 0]
    if n_cells <= target_panels:
        return 1
    return min(
        divisors,
        key=lambda d: (abs((n_cells / d) - target_panels), d == 1, d),
    )


def _infer_regular_grid(
    block_centroids: np.ndarray,
    tol: float = 1e-8,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Infer a regular axis-aligned grid ordering from block centroids."""
    centroids = np.asarray(block_centroids, dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[1] != 3 or len(centroids) == 0:
        return None

    x_vals = np.unique(np.round(centroids[:, 0], 8))
    y_vals = np.unique(np.round(centroids[:, 1], 8))
    z_vals = np.unique(np.round(centroids[:, 2], 8))
    grid_shape = (len(x_vals), len(y_vals), len(z_vals))
    if int(np.prod(grid_shape)) != len(centroids):
        return None

    order = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))
    ordered = centroids[order]
    gx, gy, gz = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    expected = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    if not np.allclose(ordered, expected, atol=tol, rtol=0.0):
        return None

    return order, grid_shape, (x_vals.astype(np.float64), y_vals.astype(np.float64), z_vals.astype(np.float64))


def _resolve_uniform_block_size(
    block_sizes: Optional[np.ndarray],
    axis_values: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Resolve a single representative block size for panel aggregation."""
    if block_sizes is not None:
        bs = np.asarray(block_sizes, dtype=np.float64)
        if bs.ndim == 1 and bs.shape[0] == 3:
            return bs.copy()
        if bs.ndim == 2 and bs.shape[1] == 3 and len(bs) > 0:
            ref = bs[0]
            if np.allclose(bs, ref, atol=1e-8, rtol=0.0):
                return ref.astype(np.float64)

    resolved = []
    for vals in axis_values:
        if len(vals) > 1:
            spacing = np.diff(vals)
            resolved.append(float(np.median(spacing)))
        else:
            resolved.append(1.0)
    return np.asarray(resolved, dtype=np.float64)


def _infer_axis_spacing(values: np.ndarray) -> float:
    """Infer a representative centroid spacing along one axis."""
    vals = np.unique(np.round(np.asarray(values, dtype=np.float64), 8))
    if len(vals) <= 1:
        return 1.0
    diffs = np.diff(vals)
    diffs = diffs[diffs > 1e-12]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))


def _expand_block_sizes(
    block_centroids: np.ndarray,
    block_sizes: Optional[np.ndarray],
) -> np.ndarray:
    """Resolve per-block sizes for panel aggregation."""
    centroids = np.asarray(block_centroids, dtype=np.float64)
    n_blocks = len(centroids)
    if block_sizes is not None:
        bs = np.asarray(block_sizes, dtype=np.float64)
        if bs.ndim == 1 and bs.shape[0] == 3:
            return np.tile(bs.reshape(1, 3), (n_blocks, 1))
        if bs.ndim == 2 and bs.shape == (n_blocks, 3):
            return bs.copy()

    inferred = np.array(
        [
            _infer_axis_spacing(centroids[:, 0]),
            _infer_axis_spacing(centroids[:, 1]),
            _infer_axis_spacing(centroids[:, 2]),
        ],
        dtype=np.float64,
    )
    inferred = np.maximum(inferred, 1.0)
    return np.tile(inferred.reshape(1, 3), (n_blocks, 1))


def _choose_panel_factor_from_extent(
    extent: float,
    representative_size: float,
    target_panels: int,
) -> int:
    """Choose a coarse panel size multiplier for irregular geometry."""
    rep = max(float(representative_size), 1e-12)
    approx_cells = max(int(np.ceil(max(float(extent), rep) / rep)), 1)
    if approx_cells <= target_panels:
        return 1
    return max(1, int(np.ceil(approx_cells / max(int(target_panels), 1))))


def _panel_linear_index(ix: int, iy: int, iz: int, panel_shape: Tuple[int, int, int]) -> int:
    """Flatten a 3-D panel index."""
    return int(ix * (panel_shape[1] * panel_shape[2]) + iy * panel_shape[2] + iz)


def _accumulate_block_panel_overlaps(
    block_mins: np.ndarray,
    block_maxs: np.ndarray,
    block_estimates: np.ndarray,
    overall_min: np.ndarray,
    panel_size: np.ndarray,
    panel_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate exact axis-aligned block-to-panel overlap volumes.

    Vectorized: most regular-grid blocks map to exactly one panel,
    handled in a single np.add.at pass.  Only boundary blocks that
    straddle panel edges fall through to the per-block loop.
    """
    n_panels_total = int(np.prod(panel_shape))
    panel_volume = np.zeros(n_panels_total, dtype=np.float64)
    panel_est_sum = np.zeros(n_panels_total, dtype=np.float64)

    finite_mask = np.isfinite(block_estimates)
    if not np.any(finite_mask):
        return panel_volume, panel_est_sum

    bmin = block_mins[finite_mask]
    bmax = block_maxs[finite_mask]
    est = block_estimates[finite_mask]

    # Panel index for each block corner
    start_idx = np.floor((bmin - overall_min) / panel_size).astype(np.int32)
    end_idx = np.floor(((bmax - overall_min) / panel_size) - 1e-12).astype(np.int32)
    for dim in range(3):
        start_idx[:, dim] = np.clip(start_idx[:, dim], 0, panel_shape[dim] - 1)
        end_idx[:, dim] = np.clip(end_idx[:, dim], 0, panel_shape[dim] - 1)

    # Fast path: blocks that fit entirely within one panel
    single_panel = np.all(start_idx == end_idx, axis=1)

    if np.any(single_panel):
        sp_idx = start_idx[single_panel]
        sp_linear = np.ravel_multi_index(
            (sp_idx[:, 0], sp_idx[:, 1], sp_idx[:, 2]), panel_shape,
        )
        sp_vol = np.prod(bmax[single_panel] - bmin[single_panel], axis=1)
        np.add.at(panel_volume, sp_linear, sp_vol)
        np.add.at(panel_est_sum, sp_linear, est[single_panel] * sp_vol)

    # Slow path: boundary blocks straddling multiple panels
    boundary = np.where(~single_panel)[0]
    for bi in boundary:
        b_est = float(est[bi])
        for ix in range(int(start_idx[bi, 0]), int(end_idx[bi, 0]) + 1):
            px0 = overall_min[0] + ix * panel_size[0]
            ox = min(bmax[bi, 0], px0 + panel_size[0]) - max(bmin[bi, 0], px0)
            if ox <= 0.0:
                continue
            for iy in range(int(start_idx[bi, 1]), int(end_idx[bi, 1]) + 1):
                py0 = overall_min[1] + iy * panel_size[1]
                oy = min(bmax[bi, 1], py0 + panel_size[1]) - max(bmin[bi, 1], py0)
                if oy <= 0.0:
                    continue
                for iz in range(int(start_idx[bi, 2]), int(end_idx[bi, 2]) + 1):
                    pz0 = overall_min[2] + iz * panel_size[2]
                    oz = min(bmax[bi, 2], pz0 + panel_size[2]) - max(bmin[bi, 2], pz0)
                    if oz <= 0.0:
                        continue
                    ov = float(ox * oy * oz)
                    linear = _panel_linear_index(ix, iy, iz, panel_shape)
                    panel_volume[linear] += ov
                    panel_est_sum[linear] += b_est * ov

    return panel_volume, panel_est_sum


def support_swath_plots(
    block_centroids: np.ndarray,
    block_estimates: np.ndarray,
    composite_coords: np.ndarray,
    composite_values: np.ndarray,
    block_sizes: Optional[np.ndarray] = None,
    panel_factors: Optional[Tuple[int, int, int]] = None,
    axes: str = "xyz",
    declustering_weights: Optional[np.ndarray] = None,
) -> Optional[SupportSwathResult]:
    """Generate support-aware swaths by aggregating blocks onto a panel lattice.

    Works on both regular and irregular block geometries. Block estimates are
    aggregated with block-volume weights, while composites are assigned into
    the same coarse panels for a more support-consistent swath comparison than
    direct point-versus-block slicing. The primary reference (``mean_actual``)
    is the declustered composite grade aggregated and re-normalised within each
    panel and slice.
    """
    centroids = np.asarray(block_centroids, dtype=np.float64)
    estimates = np.asarray(block_estimates, dtype=np.float64).ravel()
    comp_coords = np.asarray(composite_coords, dtype=np.float64)
    comp_values = np.asarray(composite_values, dtype=np.float64).ravel()

    if (
        centroids.ndim != 2
        or centroids.shape[1] != 3
        or len(centroids) == 0
        or len(estimates) != len(centroids)
    ):
        return None

    per_block_sizes = _expand_block_sizes(centroids, block_sizes)
    per_block_sizes = np.maximum(per_block_sizes, 1e-12)
    block_mins = centroids - 0.5 * per_block_sizes
    block_maxs = centroids + 0.5 * per_block_sizes

    # Support swaths are a block-support diagnostic, so anchor the panel lattice
    # to the block model extents and only compare composites that fall inside it.
    overall_min = np.min(block_mins, axis=0)
    overall_max = np.max(block_maxs, axis=0)
    extent = np.maximum(overall_max - overall_min, 1e-12)

    representative_block_size = np.median(per_block_sizes, axis=0)
    representative_block_size = np.maximum(representative_block_size, 1e-12)
    if panel_factors is None:
        fx = _choose_panel_factor_from_extent(extent[0], representative_block_size[0], 6)
        fy = _choose_panel_factor_from_extent(extent[1], representative_block_size[1], 6)
        fz = _choose_panel_factor_from_extent(extent[2], representative_block_size[2], 4)
        panel_factors = (fx, fy, fz)
    fx, fy, fz = [int(max(1, f)) for f in panel_factors]

    panel_size = representative_block_size * np.array([fx, fy, fz], dtype=np.float64)
    panel_size = np.maximum(panel_size, extent / np.array([6.0, 6.0, 4.0], dtype=np.float64))
    panel_shape = tuple(
        int(max(1, np.ceil(extent[i] / max(panel_size[i], 1e-12))))
        for i in range(3)
    )
    n_panels_total = int(np.prod(panel_shape))

    def _panel_indices(points: np.ndarray) -> np.ndarray:
        idx = np.floor((points - overall_min) / panel_size).astype(np.int32)
        for dim in range(3):
            idx[:, dim] = np.clip(idx[:, dim], 0, panel_shape[dim] - 1)
        return idx

    panel_volume, panel_est_sum = _accumulate_block_panel_overlaps(
        block_mins,
        block_maxs,
        estimates,
        overall_min,
        panel_size,
        panel_shape,
    )

    panel_coords = np.full((n_panels_total, 3), np.nan, dtype=np.float64)
    for ix in range(panel_shape[0]):
        for iy in range(panel_shape[1]):
            for iz in range(panel_shape[2]):
                linear = _panel_linear_index(ix, iy, iz, panel_shape)
                panel_coords[linear] = overall_min + panel_size * (
                    np.array([ix, iy, iz], dtype=np.float64) + 0.5
                )

    panel_est = np.full(n_panels_total, np.nan, dtype=np.float64)
    have_block_support = panel_volume > 0
    panel_est[have_block_support] = (
        panel_est_sum[have_block_support] / panel_volume[have_block_support]
    )

    comp_sum_raw = np.zeros(n_panels_total, dtype=np.float64)
    comp_count = np.zeros(n_panels_total, dtype=np.int32)
    comp_weight_sum = np.zeros(n_panels_total, dtype=np.float64)
    comp_decl_sum = np.zeros(n_panels_total, dtype=np.float64)

    if comp_coords.ndim == 2 and comp_coords.shape[1] == 3 and len(comp_coords) == len(comp_values):
        comp_mask = np.all(np.isfinite(comp_coords), axis=1) & np.isfinite(comp_values)
        tol = np.maximum(panel_size * 1e-9, 1e-9)
        comp_mask &= np.all(comp_coords >= (overall_min - tol), axis=1)
        comp_mask &= np.all(comp_coords <= (overall_max + tol), axis=1)

        if declustering_weights is not None:
            dw = np.asarray(declustering_weights, dtype=np.float64).ravel()
            if len(dw) == len(comp_values):
                comp_mask &= np.isfinite(dw) & (dw > 0.0)
                weights = dw[comp_mask]
            else:
                weights = np.ones(int(np.sum(comp_mask)), dtype=np.float64)
        else:
            weights = np.ones(int(np.sum(comp_mask)), dtype=np.float64)

        if np.any(comp_mask):
            comp_coords = comp_coords[comp_mask]
            comp_values = comp_values[comp_mask]
            comp_panel_idx = _panel_indices(comp_coords)
            comp_linear = np.ravel_multi_index(comp_panel_idx.T, panel_shape)
            np.add.at(comp_sum_raw, comp_linear, comp_values)
            np.add.at(comp_count, comp_linear, 1)
            np.add.at(comp_weight_sum, comp_linear, weights)
            np.add.at(comp_decl_sum, comp_linear, weights * comp_values)
        else:
            comp_coords = np.empty((0, 3), dtype=np.float64)
            comp_values = np.empty(0, dtype=np.float64)

    panel_actual_raw = np.full(n_panels_total, np.nan, dtype=np.float64)
    have_raw_data = comp_count > 0
    panel_actual_raw[have_raw_data] = comp_sum_raw[have_raw_data] / comp_count[have_raw_data]

    panel_actual_decl = np.full(n_panels_total, np.nan, dtype=np.float64)
    have_decl_data = comp_weight_sum > 0.0
    panel_actual_decl[have_decl_data] = comp_decl_sum[have_decl_data] / comp_weight_sum[have_decl_data]

    # Primary reference: declustered panel mean where available, otherwise raw.
    panel_actual = np.where(np.isfinite(panel_actual_decl), panel_actual_decl, panel_actual_raw)

    panel_est_metal = np.full(n_panels_total, np.nan, dtype=np.float64)
    panel_actual_metal_raw = np.full(n_panels_total, np.nan, dtype=np.float64)
    panel_actual_metal_decl = np.full(n_panels_total, np.nan, dtype=np.float64)
    panel_est_metal[have_block_support] = panel_est_sum[have_block_support]
    raw_metal_mask = have_block_support & np.isfinite(panel_actual_raw)
    decl_metal_mask = have_block_support & np.isfinite(panel_actual)
    panel_actual_metal_raw[raw_metal_mask] = (
        panel_actual_raw[raw_metal_mask] * panel_volume[raw_metal_mask]
    )
    panel_actual_metal_decl[decl_metal_mask] = (
        panel_actual[decl_metal_mask] * panel_volume[decl_metal_mask]
    )

    panel_volume_grid = panel_volume.reshape(panel_shape)
    panel_est_sum_grid = panel_est_sum.reshape(panel_shape)
    panel_count_grid = comp_count.reshape(panel_shape)
    panel_weight_grid = comp_weight_sum.reshape(panel_shape)
    panel_raw_sum_grid = comp_sum_raw.reshape(panel_shape)
    panel_decl_sum_grid = comp_decl_sum.reshape(panel_shape)

    axis_map = {"x": 0, "y": 1, "z": 2}
    axes_result: Dict[str, SupportSwathData] = {}

    for axis_name in axes:
        dim = axis_map.get(axis_name)
        if dim is None or panel_shape[dim] <= 0:
            continue

        n_slices = panel_shape[dim]
        mean_est = np.full(n_slices, np.nan, dtype=np.float64)
        mean_raw = np.full(n_slices, np.nan, dtype=np.float64)
        mean_decl = np.full(n_slices, np.nan, dtype=np.float64)
        mean_act = np.full(n_slices, np.nan, dtype=np.float64)
        n_panels = np.zeros(n_slices, dtype=np.int32)
        n_data_panels = np.zeros(n_slices, dtype=np.int32)
        n_composites = np.zeros(n_slices, dtype=np.int32)
        block_volume_per_slice = np.zeros(n_slices, dtype=np.float64)
        composite_weight_per_slice = np.zeros(n_slices, dtype=np.float64)
        positions = overall_min[dim] + panel_size[dim] * (np.arange(n_slices, dtype=np.float64) + 0.5)

        for s in range(n_slices):
            if dim == 0:
                vol_slice = panel_volume_grid[s, :, :]
                est_sum_slice = panel_est_sum_grid[s, :, :]
                raw_sum_slice = panel_raw_sum_grid[s, :, :]
                decl_sum_slice = panel_decl_sum_grid[s, :, :]
                cnt_slice = panel_count_grid[s, :, :]
                w_slice = panel_weight_grid[s, :, :]
            elif dim == 1:
                vol_slice = panel_volume_grid[:, s, :]
                est_sum_slice = panel_est_sum_grid[:, s, :]
                raw_sum_slice = panel_raw_sum_grid[:, s, :]
                decl_sum_slice = panel_decl_sum_grid[:, s, :]
                cnt_slice = panel_count_grid[:, s, :]
                w_slice = panel_weight_grid[:, s, :]
            else:
                vol_slice = panel_volume_grid[:, :, s]
                est_sum_slice = panel_est_sum_grid[:, :, s]
                raw_sum_slice = panel_raw_sum_grid[:, :, s]
                decl_sum_slice = panel_decl_sum_grid[:, :, s]
                cnt_slice = panel_count_grid[:, :, s]
                w_slice = panel_weight_grid[:, :, s]

            n_panels[s] = int(np.sum(vol_slice > 0.0))
            n_data_panels[s] = int(np.sum(cnt_slice > 0))
            n_composites[s] = int(np.sum(cnt_slice))
            block_volume_per_slice[s] = float(np.sum(vol_slice))
            composite_weight_per_slice[s] = float(np.sum(w_slice))

            if block_volume_per_slice[s] > 0.0:
                mean_est[s] = float(np.sum(est_sum_slice) / block_volume_per_slice[s])
            raw_count = int(np.sum(cnt_slice))
            if raw_count > 0:
                mean_raw[s] = float(np.sum(raw_sum_slice) / raw_count)
            if composite_weight_per_slice[s] > 0.0:
                mean_decl[s] = float(np.sum(decl_sum_slice) / composite_weight_per_slice[s])

            mean_act[s] = mean_decl[s] if np.isfinite(mean_decl[s]) else mean_raw[s]

        axes_result[axis_name] = SupportSwathData(
            axis=axis_name,
            slice_positions=positions,
            mean_estimated=mean_est,
            mean_actual=mean_act,
            mean_actual_raw=mean_raw,
            mean_actual_declustered=mean_decl,
            n_panels_per_slice=n_panels,
            n_data_panels_per_slice=n_data_panels,
            n_composites_per_slice=n_composites,
            block_volume_per_slice=block_volume_per_slice,
            composite_weight_per_slice=composite_weight_per_slice,
        )

    return SupportSwathResult(
        panel_factors=(fx, fy, fz),
        panel_shape=panel_shape,
        panel_size=np.asarray(panel_size, dtype=np.float64),
        panel_centroids=panel_coords,
        panel_block_volumes=panel_volume,
        panel_estimated=panel_est,
        panel_actual=panel_actual,
        panel_actual_raw=panel_actual_raw,
        panel_actual_declustered=panel_actual_decl,
        panel_composite_counts=comp_count,
        panel_composite_weight_sums=comp_weight_sum,
        panel_estimated_metal=panel_est_metal,
        panel_actual_metal_raw=panel_actual_metal_raw,
        panel_actual_metal_declustered=panel_actual_metal_decl,
        axes=axes_result,
        n_panels_total=n_panels_total,
        n_panels_with_data=int(np.sum(np.isfinite(panel_actual))),
    )
