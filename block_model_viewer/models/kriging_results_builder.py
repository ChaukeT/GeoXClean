"""
Kriging Results Builder
=======================

Converts raw kriging outputs to professional OrdinaryKrigingResults objects.

This bridges the gap between the current implementation (dict-based QA metrics)
and the professional dataclass-based results structure.

Author: GeoX Development Team
Date: 2026-02-07
"""

import numpy as np
from typing import Dict, Optional
from .geostat_results import OrdinaryKrigingResults


def build_ordinary_kriging_results(
    estimates: np.ndarray,
    variances: np.ndarray,
    qa_metrics: Optional[Dict[str, np.ndarray]],
    metadata: Optional[Dict] = None
) -> OrdinaryKrigingResults:
    """
    Build professional OrdinaryKrigingResults object from raw outputs.

    Parameters
    ----------
    estimates : np.ndarray
        (M,) array of kriging estimates
    variances : np.ndarray
        (M,) array of kriging variances
    qa_metrics : dict, optional
        QA metrics dict with keys: kriging_efficiency, slope_of_regression,
        n_samples, pass_number, distance_to_nearest, pct_negative_weights
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    OrdinaryKrigingResults
        Professional results object matching industry standards
    """
    m = len(estimates)

    # Status: 0 = unestimated, 1 = estimated
    status = np.where(~np.isnan(estimates), 1, 0).astype(int)

    if qa_metrics is not None:
        # Professional mode: extract QA metrics
        kriging_efficiency = qa_metrics.get('kriging_efficiency', np.full(m, np.nan))
        slope_of_regression = qa_metrics.get('slope_of_regression', np.full(m, np.nan))
        num_samples = qa_metrics.get('n_samples', np.zeros(m, dtype=int))
        search_pass = qa_metrics.get('pass_number', np.zeros(m, dtype=int))
        min_distance = qa_metrics.get('distance_to_nearest', np.full(m, np.nan))
        pct_neg_weights = qa_metrics.get('pct_negative_weights', np.full(m, np.nan))

        # Compute sum_negative_weights from percentage
        sum_negative_weights = pct_neg_weights / 100.0 * num_samples

        # Derive other attributes
        # kriging_mean: for OK, this is the estimate itself (local mean)
        kriging_mean = estimates.copy()

        # sum_weights: for OK, sum(w) = 1.0 by constraint (slope_of_regression ≈ 1)
        sum_weights = slope_of_regression.copy()

        # lagrange_multiplier: not directly computed, set to NaN
        lagrange_multiplier = np.full(m, np.nan)

        # avg_distance: not computed, set to NaN (could compute if needed)
        avg_distance = np.full(m, np.nan)

        # nearest_sample_id: not tracked, set to -1
        nearest_sample_id = np.full(m, -1, dtype=int)

        # num_duplicates_removed: not tracked, set to 0
        num_duplicates_removed = np.zeros(m, dtype=int)

        # search_volume: not tracked, set to NaN
        search_volume = np.full(m, np.nan)

    else:
        # Legacy mode: minimal attributes
        kriging_efficiency = np.full(m, np.nan)
        slope_of_regression = np.full(m, 1.0)  # OK constraint
        num_samples = np.zeros(m, dtype=int)
        search_pass = np.ones(m, dtype=int)  # All Pass 1 (legacy single-pass)
        min_distance = np.full(m, np.nan)
        sum_negative_weights = np.zeros(m)
        kriging_mean = estimates.copy()
        sum_weights = np.ones(m)
        lagrange_multiplier = np.full(m, np.nan)
        avg_distance = np.full(m, np.nan)
        nearest_sample_id = np.full(m, -1, dtype=int)
        num_duplicates_removed = np.zeros(m, dtype=int)
        search_volume = np.full(m, np.nan)

    return OrdinaryKrigingResults(
        estimates=estimates,
        status=status,
        kriging_mean=kriging_mean,
        kriging_variance=variances,
        kriging_efficiency=kriging_efficiency,
        slope_of_regression=slope_of_regression,
        lagrange_multiplier=lagrange_multiplier,
        num_samples=num_samples,
        sum_weights=sum_weights,
        sum_negative_weights=sum_negative_weights,
        min_distance=min_distance,
        avg_distance=avg_distance,
        nearest_sample_id=nearest_sample_id,
        num_duplicates_removed=num_duplicates_removed,
        search_pass=search_pass,
        search_volume=search_volume,
        metadata=metadata or {}
    )


def extract_qa_summary_from_results(results: OrdinaryKrigingResults) -> Dict:
    """
    Extract QA summary statistics from OrdinaryKrigingResults.

    Parameters
    ----------
    results : OrdinaryKrigingResults
        Professional results object

    Returns
    -------
    dict
        QA summary matching controller metadata format
    """
    valid = results.status == 1
    if not np.any(valid):
        return {'n_valid': 0}

    ke_valid = results.kriging_efficiency[valid]
    sor_valid = results.slope_of_regression[valid]
    neg_wt_valid = results.sum_negative_weights[valid] / results.num_samples[valid] * 100.0
    pass_nums = results.search_pass[valid]

    return {
        'kriging_efficiency_mean': float(np.nanmean(ke_valid)),
        'kriging_efficiency_min': float(np.nanmin(ke_valid)),
        'slope_of_regression_mean': float(np.nanmean(sor_valid)),
        'pct_negative_weights_max': float(np.nanmax(neg_wt_valid)),
        'pass_1_count': int(np.sum(pass_nums == 1)),
        'pass_2_count': int(np.sum(pass_nums == 2)),
        'pass_3_count': int(np.sum(pass_nums == 3)),
        'unestimated_count': int(np.sum(results.status == 0)),
        'n_valid': int(np.sum(valid))
    }


# Convenience function for backward compatibility
def ordinary_kriging_3d_with_results(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    target_coords: np.ndarray,
    variogram_params: Dict,
    n_neighbors: int = 12,
    max_distance: Optional[float] = None,
    model_type: str = "spherical",
    progress_callback=None,
    search_passes: Optional[list] = None,
    compute_qa_metrics: bool = True,
) -> OrdinaryKrigingResults:
    """
    Professional-mode ordinary kriging that returns OrdinaryKrigingResults object.

    This is a wrapper around ordinary_kriging_3d() that automatically converts
    the output to the professional OrdinaryKrigingResults dataclass.

    Parameters
    ----------
    Same as ordinary_kriging_3d()

    Returns
    -------
    OrdinaryKrigingResults
        Professional results object with all QA metrics populated

    Example
    -------
    >>> from block_model_viewer.models.kriging_results_builder import ordinary_kriging_3d_with_results
    >>> results = ordinary_kriging_3d_with_results(
    ...     data_coords, data_values, target_coords, variogram_params,
    ...     search_passes=[...],
    ...     compute_qa_metrics=True
    ... )
    >>> print(f"Mean KE: {np.nanmean(results.kriging_efficiency):.3f}")
    >>> print(f"Pass 1: {np.sum(results.search_pass == 1)} blocks")
    """
    from .kriging3d import ordinary_kriging_3d

    # Call underlying function
    estimates, variances, qa_metrics = ordinary_kriging_3d(
        data_coords=data_coords,
        data_values=data_values,
        target_coords=target_coords,
        variogram_params=variogram_params,
        n_neighbors=n_neighbors,
        max_distance=max_distance,
        model_type=model_type,
        progress_callback=progress_callback,
        search_passes=search_passes,
        compute_qa_metrics=compute_qa_metrics
    )

    # Build professional results object
    metadata = {
        'variogram_params': variogram_params,
        'n_neighbors': n_neighbors,
        'max_distance': max_distance,
        'model_type': model_type,
        'multi_pass_enabled': search_passes is not None,
        'n_passes': len(search_passes) if search_passes else 1
    }

    return build_ordinary_kriging_results(
        estimates=estimates,
        variances=variances,
        qa_metrics=qa_metrics,
        metadata=metadata
    )
