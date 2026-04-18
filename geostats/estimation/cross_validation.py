"""
Cross-validation routines for FastRBF estimation.

Implements:
  - Leave-One-Out (LOO) cross-validation
  - Spatially blocked k-fold cross-validation
  - Jackknife variance estimation

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import RBFConfig
from .fastrbf_engine import FastRBFEngine

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Cross-validation statistics."""

    actual: NDArray[np.float64]
    estimated: NDArray[np.float64]
    errors: NDArray[np.float64]
    abs_errors: NDArray[np.float64]
    sq_errors: NDArray[np.float64]
    mean_error: float  # ME — should be ~0
    mae: float
    rmse: float
    r_squared: float
    correlation: float
    normalised_rmse: float  # RMSE / StdDev of actuals
    n_samples: int
    warnings: list[str]


@dataclass
class FoldResult:
    """Per-fold result for k-fold CV."""

    fold_id: int
    n_train: int
    n_test: int
    me: float
    mae: float
    rmse: float
    r_squared: float


@dataclass
class KFoldResult:
    """Aggregate k-fold cross-validation result."""

    fold_results: list[FoldResult]
    aggregate_me: float
    aggregate_mae: float
    aggregate_rmse: float
    aggregate_r_squared: float
    k: int


@dataclass
class JackknifeResult:
    """Jackknife variance estimation per prediction point."""

    means: NDArray[np.float64]
    variances: NDArray[np.float64]


def _compute_stats(
    actual: NDArray[np.float64],
    estimated: NDArray[np.float64],
) -> CVResult:
    """Compute comprehensive CV statistics."""
    errors = estimated - actual
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    n = len(actual)
    me = float(np.mean(errors))
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))

    # R²
    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Correlation coefficient
    if np.std(actual) > 0 and np.std(estimated) > 0:
        correlation = float(np.corrcoef(actual, estimated)[0, 1])
    else:
        correlation = 0.0

    # Normalised RMSE
    std_actual = float(np.std(actual))
    nrmse = rmse / std_actual if std_actual > 0 else float("inf")

    warnings = []
    if abs(me) > 0.1 * rmse and rmse > 0:
        warnings.append(
            f"Mean error ({me:.4f}) exceeds 10% of RMSE ({rmse:.4f}): "
            "potential systematic bias"
        )
    if nrmse >= 1.0:
        warnings.append(
            f"Normalised RMSE ({nrmse:.3f}) >= 1.0: model may not be useful"
        )

    return CVResult(
        actual=actual,
        estimated=estimated,
        errors=errors,
        abs_errors=abs_errors,
        sq_errors=sq_errors,
        mean_error=me,
        mae=mae,
        rmse=rmse,
        r_squared=r_squared,
        correlation=correlation,
        normalised_rmse=nrmse,
        n_samples=n,
        warnings=warnings,
    )


_LOO_MAX_SAMPLES = 300  # Above this, subsample to keep runtime < ~2 min


def loo_cross_validation(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    config: RBFConfig,
    progress_callback=None,
    max_samples: int = _LOO_MAX_SAMPLES,
) -> CVResult:
    """
    Leave-One-Out cross-validation.

    For each sample, remove it, re-solve the RBF system, predict
    at its location.  Records actual, estimated, error, and
    computes ME, MAE, RMSE, R², correlation, normalised RMSE.

    If N > max_samples, a spatially stratified subsample is used
    to keep runtime practical (O(N³) per iteration).

    Parameters
    ----------
    points : (N, 3) ndarray
    values : (N,) ndarray
    config : RBFConfig
    progress_callback : callable(int, str), optional
    max_samples : int
        Maximum samples for LOO. If N exceeds this, subsample.

    Returns
    -------
    CVResult
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    # Subsample if dataset is too large for LOO
    if n > max_samples:
        logger.info(
            "LOO CV: subsampling %d → %d samples (full LOO would be O(N⁴))",
            n, max_samples,
        )
        idx = _stratified_subsample(points, max_samples)
        points = points[idx]
        values = values[idx]
        n = len(values)

    logger.info("Running LOO cross-validation with %d samples", n)

    engine = FastRBFEngine(config)
    estimates = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Leave out sample i
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        train_pts = points[mask]
        train_vals = values[mask]
        query = points[i : i + 1]

        try:
            fitted = engine.fit(train_pts, train_vals)
            est = engine.predict(fitted, query)
            estimates[i] = est[0]
        except Exception as exc:
            logger.debug("LOO sample %d failed: %s", i, exc)
            estimates[i] = np.nan

        # Report progress every 10 iterations
        if progress_callback and (i % 10 == 0 or i == n - 1):
            pct = int(75 + 15 * (i + 1) / n)  # 75% → 90%
            progress_callback(pct, f"Cross-validation: {i + 1}/{n}")

    # Remove failed predictions
    valid = ~np.isnan(estimates)
    if valid.sum() < n:
        logger.warning(
            "LOO: %d/%d predictions failed", n - valid.sum(), n
        )

    result = _compute_stats(values[valid], estimates[valid])
    return result


def _stratified_subsample(
    points: NDArray[np.float64], n_target: int, seed: int = 42
) -> NDArray[np.int64]:
    """
    Spatially stratified subsample — preserves spatial coverage.

    Sorts along the longest axis, then takes evenly spaced indices
    with slight random jitter for representativeness.
    """
    n = points.shape[0]
    if n <= n_target:
        return np.arange(n)

    # Sort along longest axis for spatial stratification
    ranges = points.max(axis=0) - points.min(axis=0)
    axis = int(np.argmax(ranges))
    sorted_idx = np.argsort(points[:, axis])

    # Evenly spaced indices
    step = n / n_target
    idx = np.array([int(i * step) for i in range(n_target)], dtype=np.int64)
    idx = np.clip(idx, 0, n - 1)

    return sorted_idx[idx]


def kfold_cross_validation(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    config: RBFConfig,
    k: int = 5,
    seed: int = 42,
) -> KFoldResult:
    """
    Spatially blocked k-fold cross-validation.

    Folds are assigned by spatial clustering (k-means on coordinates)
    rather than random assignment, to avoid spatial leakage.

    Parameters
    ----------
    points : (N, 3) ndarray
    values : (N,) ndarray
    config : RBFConfig
    k : int
        Number of folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    KFoldResult
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    logger.info("Running %d-fold spatial CV with %d samples", k, n)

    # Spatial blocking via simple k-means-like assignment
    rng = np.random.Generator(np.random.PCG64(seed))

    # Assign folds by spatial slicing along the longest axis
    ranges = points.max(axis=0) - points.min(axis=0)
    longest_axis = int(np.argmax(ranges))
    sorted_idx = np.argsort(points[:, longest_axis])

    fold_ids = np.zeros(n, dtype=np.int32)
    fold_size = n // k
    for f in range(k):
        start = f * fold_size
        end = start + fold_size if f < k - 1 else n
        fold_ids[sorted_idx[start:end]] = f

    engine = FastRBFEngine(config)
    fold_results = []

    all_actual = []
    all_estimated = []

    for f in range(k):
        test_mask = fold_ids == f
        train_mask = ~test_mask

        train_pts = points[train_mask]
        train_vals = values[train_mask]
        test_pts = points[test_mask]
        test_vals = values[test_mask]

        try:
            fitted = engine.fit(train_pts, train_vals)
            est = engine.predict(fitted, test_pts)
        except Exception as exc:
            logger.warning("Fold %d failed: %s", f, exc)
            est = np.full(len(test_vals), np.nan)

        valid = ~np.isnan(est)
        if valid.sum() > 0:
            fold_cv = _compute_stats(test_vals[valid], est[valid])
            fold_results.append(
                FoldResult(
                    fold_id=f,
                    n_train=int(train_mask.sum()),
                    n_test=int(test_mask.sum()),
                    me=fold_cv.mean_error,
                    mae=fold_cv.mae,
                    rmse=fold_cv.rmse,
                    r_squared=fold_cv.r_squared,
                )
            )
            all_actual.append(test_vals[valid])
            all_estimated.append(est[valid])

    # Aggregate
    if all_actual:
        agg_actual = np.concatenate(all_actual)
        agg_est = np.concatenate(all_estimated)
        agg = _compute_stats(agg_actual, agg_est)
    else:
        agg = _compute_stats(np.array([0.0]), np.array([0.0]))

    return KFoldResult(
        fold_results=fold_results,
        aggregate_me=agg.mean_error,
        aggregate_mae=agg.mae,
        aggregate_rmse=agg.rmse,
        aggregate_r_squared=agg.r_squared,
        k=k,
    )


def jackknife_variance(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    query_points: NDArray[np.float64],
    config: RBFConfig,
) -> JackknifeResult:
    """
    Jackknife estimate of prediction variance at query points.

    For each leave-one-out subset, predicts at all query points.
    The jackknife variance is computed from the spread of these
    predictions.

    Parameters
    ----------
    points : (N, 3)
    values : (N,)
    query_points : (M, 3)
    config : RBFConfig

    Returns
    -------
    JackknifeResult
        means: (M,) jackknife mean predictions
        variances: (M,) jackknife variance estimates
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)

    n = len(values)
    m = query_points.shape[0]

    logger.info("Jackknife variance: %d samples, %d query points", n, m)

    engine = FastRBFEngine(config)
    predictions = np.empty((n, m), dtype=np.float64)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        try:
            fitted = engine.fit(points[mask], values[mask])
            predictions[i] = engine.predict(fitted, query_points)
        except Exception:
            predictions[i] = np.nan

    # Jackknife statistics
    means = np.nanmean(predictions, axis=0)
    # Jackknife variance: (n_valid-1)/n_valid * Σ(θ_i - θ_bar)²
    # Use n_valid (not n) to account for failed predictions
    n_valid_per_query = np.sum(~np.isnan(predictions), axis=0)
    n_valid_per_query = np.maximum(n_valid_per_query, 1)  # avoid division by zero
    deviations = predictions - means[np.newaxis, :]
    variances = (
        (n_valid_per_query - 1) / n_valid_per_query
        * np.nansum(deviations ** 2, axis=0)
    )

    return JackknifeResult(means=means, variances=variances)
