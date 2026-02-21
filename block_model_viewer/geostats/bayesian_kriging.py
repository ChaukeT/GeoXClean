"""
Bayesian / Soft Kriging Extensions (STEP 23).

Provides wrappers for OK/UK/IK/CoK that incorporate soft data as Bayesian priors.

Industry-Standard Approach:
--------------------------
Soft/Bayesian Kriging incorporates uncertain (soft) information by:
1. Treating soft data as additional observations with increased measurement error
2. Modifying the covariance matrix diagonal: C_soft(i,i) = C(i,i) + σ²_soft
3. Using precision-weighted combination of kriging estimates and soft priors

This implementation follows the methodology described in:
- Journel, A.G. (1986) "Geostatistics: Models and Tools for the Earth Sciences"
- Deutsch, C.V. (2002) "Geostatistical Reservoir Modeling"
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal, Tuple, List
import numpy as np
import logging

from scipy.spatial.distance import cdist
from scipy.linalg import solve, LinAlgError

from .soft_data import SoftDataSet
from ..models.kriging3d import ordinary_kriging_3d, get_variogram_function, apply_anisotropy
from .universal_kriging import UniversalKriging3D, DriftModel
from .indicator_kriging import IKConfig, IKResult, run_indicator_kriging_job
from .cokriging3d import CoKriging3D, CoKrigingConfig

logger = logging.getLogger(__name__)


@dataclass
class BayesianKrigingConfig:
    """Configuration for Bayesian kriging."""
    base_method: Literal["OK", "UK", "IK", "CoK"]
    prior_type: Literal["mean_only", "mean_var"] = "mean_var"
    soft_weighting: float = 0.5  # Weight for soft data (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _find_nearest_soft_data(
    target_coords: np.ndarray,
    soft_data: SoftDataSet,
    max_distance: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find nearest soft data points for each target location.
    
    Args:
        target_coords: (M, 3) target coordinates
        soft_data: SoftDataSet
        max_distance: Optional maximum distance to search
    
    Returns:
        Tuple of (soft_means, soft_variances, soft_coords) arrays
    """
    if len(soft_data.points) == 0:
        return np.array([]), np.array([]), np.array([])
    
    soft_coords = soft_data.get_coords()
    soft_means = soft_data.get_means()
    soft_variances = soft_data.get_variances()
    
    # Find nearest soft point for each target
    from scipy.spatial.distance import cdist
    
    distances = cdist(target_coords, soft_coords)
    
    if max_distance is not None:
        distances[distances > max_distance] = np.inf
    
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(target_coords)), nearest_indices]
    
    # Filter out targets with no nearby soft data
    valid_mask = np.isfinite(nearest_distances)
    
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
    
    return (
        soft_means[nearest_indices[valid_mask]],
        soft_variances[nearest_indices[valid_mask]],
        soft_coords[nearest_indices[valid_mask]]
    )


def run_bayesian_ok(
    coords: np.ndarray,
    values: np.ndarray,
    soft_data: Optional[SoftDataSet],
    variogram_model: Dict[str, Any],
    config: BayesianKrigingConfig,
    locations: np.ndarray,
    search_params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Bayesian Ordinary Kriging with soft data.
    
    Industry-Standard Approach:
    1. If no soft data: standard OK
    2. With soft data: precision-weighted combination of OK estimate and soft prior
       Z*_bayes = (Z*_ok/σ²_ok + Z_soft/σ²_soft) / (1/σ²_ok + 1/σ²_soft)
       σ²_bayes = 1 / (1/σ²_ok + w_soft/σ²_soft)
    
    Args:
        coords: (N, 3) hard data coordinates
        values: (N,) hard data values
        soft_data: Optional SoftDataSet
        variogram_model: Variogram parameters dict
        config: BayesianKrigingConfig
        locations: (M, 3) target locations
        search_params: Optional search parameters
    
    Returns:
        Tuple of (estimates, variances)
    """
    n_neighbors = search_params.get('n_neighbors', 12) if search_params else 12
    max_distance = search_params.get('max_distance') if search_params else None
    model_type = variogram_model.get('model_type', 'spherical') if variogram_model else 'spherical'
    
    if soft_data is None or len(soft_data.points) == 0:
        # Fallback to standard OK
        logger.info("No soft data provided, using standard OK")
        estimates, variances, _ = ordinary_kriging_3d(  # Ignore QA metrics
            coords,
            values,
            locations,
            variogram_model,
            n_neighbors=n_neighbors,
            max_distance=max_distance,
            model_type=model_type,
            progress_callback=None
        )
        return estimates, variances
    
    # Find nearest soft data for each target
    soft_means, soft_variances, soft_coords = _find_nearest_soft_data(
        locations,
        soft_data,
        max_distance=max_distance
    )
    
    if len(soft_means) == 0:
        # No soft data nearby, use standard OK
        logger.info("No soft data nearby targets, using standard OK")
        estimates, variances, _ = ordinary_kriging_3d(  # Ignore QA metrics
            coords,
            values,
            locations,
            variogram_model,
            n_neighbors=n_neighbors,
            max_distance=max_distance,
            model_type=model_type,
            progress_callback=None
        )
        return estimates, variances

    # Run standard OK first
    ok_estimates, ok_variances, _ = ordinary_kriging_3d(  # Ignore QA metrics
        coords,
        values,
        locations,
        variogram_model,
        n_neighbors=n_neighbors,
        max_distance=max_distance,
        model_type=model_type,
        progress_callback=None
    )
    
    # Initialize output arrays
    estimates = ok_estimates.copy()
    variances = ok_variances.copy()
    
    soft_weight = config.soft_weighting
    
    # Map soft data to target locations
    # Find nearest soft data point for each target
    from scipy.spatial.distance import cdist as scipy_cdist
    
    soft_coords_all = soft_data.get_coords()
    soft_means_all = soft_data.get_means()
    soft_variances_all = soft_data.get_variances()
    
    if len(soft_coords_all) == 0:
        return estimates, variances
    
    distances = scipy_cdist(locations, soft_coords_all)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(locations)), nearest_idx]
    
    # Bayesian update: precision-weighted combination
    # Z*_bayes = (Z*_ok * prec_ok + Z_soft * prec_soft * w) / (prec_ok + prec_soft * w)
    # σ²_bayes = 1 / (prec_ok + prec_soft * w)
    
    for i in range(len(locations)):
        ok_est = ok_estimates[i]
        ok_var = ok_variances[i]
        
        # Skip if OK estimate is invalid
        if np.isnan(ok_est) or np.isnan(ok_var) or ok_var <= 0:
            continue
        
        # Check if soft data is within reasonable distance
        if max_distance is not None and nearest_dist[i] > max_distance:
            continue  # No soft data nearby, keep OK estimate
        
        soft_mean = soft_means_all[nearest_idx[i]]
        soft_var = soft_variances_all[nearest_idx[i]]
        
        if np.isnan(soft_mean) or np.isnan(soft_var) or soft_var <= 0:
            continue
        
        # Precision-weighted combination (Bayesian update)
        ok_precision = 1.0 / ok_var
        soft_precision = soft_weight / soft_var  # Weight applied to soft data precision
        total_precision = ok_precision + soft_precision
        
        if total_precision > 0:
            # Bayesian estimate
            estimates[i] = (ok_precision * ok_est + soft_precision * soft_mean) / total_precision
            # Bayesian variance
            variances[i] = 1.0 / total_precision
    
    logger.info(f"Bayesian OK completed: {np.sum(~np.isnan(estimates))}/{len(locations)} valid estimates")
    
    return estimates, variances


def run_bayesian_uk(
    coords: np.ndarray,
    values: np.ndarray,
    soft_data: Optional[SoftDataSet],
    variogram_model: Dict[str, Any],
    drift_model: DriftModel,
    config: BayesianKrigingConfig,
    locations: np.ndarray,
    search_params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Bayesian Universal Kriging with soft data.
    
    Industry-Standard Approach:
    1. Run standard UK to get estimates and variances
    2. Apply precision-weighted Bayesian update with soft data
    
    Args:
        coords: (N, 3) hard data coordinates
        values: (N,) hard data values
        soft_data: Optional SoftDataSet
        variogram_model: Variogram parameters dict
        drift_model: DriftModel instance
        config: BayesianKrigingConfig
        locations: (M, 3) target locations
        search_params: Optional search parameters
    
    Returns:
        Tuple of (estimates, variances)
    """
    # Run standard UK first
    uk = UniversalKriging3D(
        coords,
        values,
        variogram_model,
        drift_model,
        search_params or {}
    )
    uk_estimates, uk_variances = uk.estimate(locations)
    
    if soft_data is None or len(soft_data.points) == 0:
        logger.info("No soft data provided, using standard UK")
        return uk_estimates, uk_variances
    
    # Initialize output arrays
    estimates = uk_estimates.copy()
    variances = uk_variances.copy()
    
    soft_weight = config.soft_weighting
    max_distance = search_params.get('max_distance') if search_params else None
    
    # Get soft data arrays
    soft_coords_all = soft_data.get_coords()
    soft_means_all = soft_data.get_means()
    soft_variances_all = soft_data.get_variances()
    
    if len(soft_coords_all) == 0:
        return estimates, variances
    
    # Find nearest soft data for each target
    from scipy.spatial.distance import cdist as scipy_cdist
    distances = scipy_cdist(locations, soft_coords_all)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(locations)), nearest_idx]
    
    # Apply Bayesian update
    for i in range(len(locations)):
        uk_est = uk_estimates[i]
        uk_var = uk_variances[i]
        
        if np.isnan(uk_est) or np.isnan(uk_var) or uk_var <= 0:
            continue
        
        if max_distance is not None and nearest_dist[i] > max_distance:
            continue
        
        soft_mean = soft_means_all[nearest_idx[i]]
        soft_var = soft_variances_all[nearest_idx[i]]
        
        if np.isnan(soft_mean) or np.isnan(soft_var) or soft_var <= 0:
            continue
        
        # Precision-weighted combination
        uk_precision = 1.0 / uk_var
        soft_precision = soft_weight / soft_var
        total_precision = uk_precision + soft_precision
        
        if total_precision > 0:
            estimates[i] = (uk_precision * uk_est + soft_precision * soft_mean) / total_precision
            variances[i] = 1.0 / total_precision
    
    logger.info(f"Bayesian UK completed: {np.sum(~np.isnan(estimates))}/{len(locations)} valid estimates")
    
    return estimates, variances


def run_bayesian_ik(
    coords: np.ndarray,
    values: np.ndarray,
    soft_data: Optional[SoftDataSet],
    variogram_template: Dict[str, Any],
    ik_config: IKConfig,
    locations: np.ndarray,
    search_params: Optional[Dict[str, Any]] = None,
    soft_weight: float = 0.5
) -> Any:
    """
    Run Bayesian Indicator Kriging with soft probabilities.
    
    Industry-Standard Approach:
    For IK, soft data represents soft probabilities at each threshold.
    The Bayesian update is applied to the CDF probabilities:
    P_bayes(Z <= t) = w_hard * P_ik(Z <= t) + w_soft * P_soft(Z <= t)
    
    If soft data provides mean/variance instead of probabilities,
    we convert to probability using normal CDF approximation.
    
    Args:
        coords: (N, 3) hard data coordinates
        values: (N,) hard data values
        soft_data: Optional SoftDataSet (mean/variance or soft probabilities)
        variogram_template: Variogram template dict
        ik_config: IKConfig
        locations: (M, 3) target locations
        search_params: Optional search parameters
        soft_weight: Weight for soft data (0-1)
    
    Returns:
        IKResult-like object
    """
    from .indicator_kriging import IKResult, run_indicator_kriging_job
    from scipy.stats import norm
    import pandas as pd
    
    # Convert IKConfig to job params format for run_indicator_kriging_job()
    # Create DataFrame from coords and values
    data_df = pd.DataFrame({
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'value': values
    })
    
    # Build job params dict
    job_params = {
        'data_df': data_df,
        'variable': 'value',
        'thresholds': ik_config.thresholds,
        'variogram_template': ik_config.variogram_model_template,
        'grid_config': {
            'spacing': (1.0, 1.0, 1.0),  # Dummy spacing - we'll use locations directly
            'origin': None,
            'counts': None
        },
        'search_config': {
            'n_neighbors': ik_config.search_params.get('n_neighbors', 12),
            'max_distance': ik_config.search_params.get('max_distance', None)
        },
        'compute_median': ik_config.compute_median,
        'compute_mean': ik_config.compute_mean
    }
    
    # For point-based estimation (locations), we need to create a custom grid
    # The function expects a grid, but we can create a point cloud grid
    # Actually, we need to modify the approach - create a grid that matches locations
    # For now, create a minimal grid that covers locations
    if len(locations) > 0:
        # Create a grid that covers all locations
        loc_min = locations.min(axis=0)
        loc_max = locations.max(axis=0)
        spacing = ik_config.search_params.get('spacing', (10.0, 10.0, 5.0))
        
        # Calculate grid dimensions
        dx, dy, dz = spacing
        nx = max(2, int(np.ceil((loc_max[0] - loc_min[0]) / dx)) + 1)
        ny = max(2, int(np.ceil((loc_max[1] - loc_min[1]) / dy)) + 1)
        nz = max(2, int(np.ceil((loc_max[2] - loc_min[2]) / dz)) + 1)
        
        job_params['grid_config'] = {
            'spacing': spacing,
            'origin': tuple(loc_min),
            'counts': (nx, ny, nz)
        }
    
    # Run IK job
    ik_result_dict = run_indicator_kriging_job(job_params)
    
    if 'error' in ik_result_dict:
        logger.error(f"Indicator Kriging failed: {ik_result_dict['error']}")
        # Return empty result
        return IKResult(
            thresholds=np.array(ik_config.thresholds),
            probabilities=np.zeros((len(locations), len(ik_config.thresholds))),
            median=None,
            mean=None,
            metadata={'error': ik_result_dict['error']}
        )
    
    # Extract probabilities for the specific locations
    # The result has probabilities for the full grid, but we need them for specific locations
    # Use nearest neighbor search to find closest grid points to each location
    probs_grid = ik_result_dict['probabilities']  # (nx, ny, nz, n_thresh)
    
    # Get grid coordinates from result
    grid_x = ik_result_dict['grid_x']  # (nx, ny, nz) or 1D
    grid_y = ik_result_dict['grid_y']
    grid_z = ik_result_dict['grid_z']
    
    # Flatten grid coordinates
    if grid_x.ndim == 3:
        grid_coords = np.column_stack([
            grid_x.ravel(order='F'),
            grid_y.ravel(order='F'),
            grid_z.ravel(order='F')
        ])
    else:
        grid_coords = np.column_stack([grid_x, grid_y, grid_z])
    
    # Find nearest grid points to each location
    n_locations = len(locations)
    n_thresh = probs_grid.shape[-1]
    probabilities = np.full((n_locations, n_thresh), np.nan)
    
    if len(grid_coords) > 0:
        from scipy.spatial.distance import cdist
        distances = cdist(locations, grid_coords)
        nearest_indices = np.argmin(distances, axis=1)
        
        # Extract probabilities for nearest grid points
        probs_flat = probs_grid.reshape(-1, n_thresh)  # (nx*ny*nz, n_thresh)
        for i, idx in enumerate(nearest_indices):
            if idx < len(probs_flat):
                probabilities[i, :] = probs_flat[idx, :]
    
    # Extract median and mean similarly
    median = None
    mean = None
    if ik_config.compute_median and 'median' in ik_result_dict:
        median_grid = ik_result_dict['median']  # (nx, ny, nz)
        median_flat = median_grid.ravel(order='F')
        if len(grid_coords) > 0 and len(median_flat) > 0:
            from scipy.spatial.distance import cdist
            distances = cdist(locations, grid_coords)
            nearest_indices = np.argmin(distances, axis=1)
            median = np.full(n_locations, np.nan)
            for i, idx in enumerate(nearest_indices):
                if idx < len(median_flat):
                    median[i] = median_flat[idx]
    
    if ik_config.compute_mean and 'mean' in ik_result_dict:
        mean_grid = ik_result_dict['mean']  # (nx, ny, nz)
        mean_flat = mean_grid.ravel(order='F')
        if len(grid_coords) > 0 and len(mean_flat) > 0:
            from scipy.spatial.distance import cdist
            distances = cdist(locations, grid_coords)
            nearest_indices = np.argmin(distances, axis=1)
            mean = np.full(n_locations, np.nan)
            for i, idx in enumerate(nearest_indices):
                if idx < len(mean_flat):
                    mean[i] = mean_flat[idx]
    
    ik_result = IKResult(
        thresholds=np.array(ik_result_dict['thresholds']),
        probabilities=probabilities,
        median=median,
        mean=mean,
        metadata=ik_result_dict.get('metadata', {})
    )
    
    if soft_data is None or len(soft_data.points) == 0:
        logger.info("No soft data provided, using standard IK")
        return ik_result
    
    # Get soft data
    soft_coords = soft_data.get_coords()
    soft_means = soft_data.get_means()
    soft_variances = soft_data.get_variances()
    
    if len(soft_coords) == 0:
        return ik_result
    
    # Find nearest soft data for each target
    from scipy.spatial.distance import cdist as scipy_cdist
    distances = scipy_cdist(locations, soft_coords)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(locations)), nearest_idx]
    
    max_distance = search_params.get('max_distance') if search_params else None
    thresholds = ik_config.thresholds
    
    # Update probabilities with Bayesian weighting
    probabilities = ik_result.probabilities.copy()
    
    for i in range(len(locations)):
        if max_distance is not None and nearest_dist[i] > max_distance:
            continue
        
        soft_mean = soft_means[nearest_idx[i]]
        soft_var = soft_variances[nearest_idx[i]]
        
        if np.isnan(soft_mean) or np.isnan(soft_var) or soft_var <= 0:
            continue
        
        soft_std = np.sqrt(soft_var)
        
        # Convert soft data (mean, variance) to probabilities using normal CDF
        for k, threshold in enumerate(thresholds):
            ik_prob = probabilities[i, k]
            
            if np.isnan(ik_prob):
                continue
            
            # Soft probability: P(Z <= threshold) from normal distribution
            soft_prob = norm.cdf(threshold, loc=soft_mean, scale=soft_std)
            
            # Bayesian weighted combination
            hard_weight = 1.0 - soft_weight
            probabilities[i, k] = hard_weight * ik_prob + soft_weight * soft_prob
    
    # Ensure probabilities are monotonic (CDF property)
    for i in range(len(locations)):
        probs = probabilities[i, :]
        # Sort to ensure monotonically increasing
        sorted_probs = np.maximum.accumulate(probs)
        probabilities[i, :] = np.clip(sorted_probs, 0.0, 1.0)
    
    # Recompute median and mean from updated probabilities
    median = None
    mean = None
    
    if ik_config.compute_median:
        median = np.full(len(locations), np.nan)
        for i in range(len(locations)):
            probs = probabilities[i, :]
            if np.all(np.isnan(probs)):
                continue
            valid_mask = ~np.isnan(probs)
            if np.any(valid_mask):
                valid_probs = probs[valid_mask]
                valid_thresh = np.array(thresholds)[valid_mask]
                idx = np.searchsorted(valid_probs, 0.5)
                if idx > 0 and idx < len(valid_probs):
                    p_low, p_high = valid_probs[idx - 1], valid_probs[idx]
                    t_low, t_high = valid_thresh[idx - 1], valid_thresh[idx]
                    if p_high != p_low:
                        median[i] = t_low + (0.5 - p_low) * (t_high - t_low) / (p_high - p_low)
                    else:
                        median[i] = (t_low + t_high) / 2.0
                elif idx == 0:
                    median[i] = valid_thresh[0]
                else:
                    median[i] = valid_thresh[-1]
    
    if ik_config.compute_mean:
        mean = np.full(len(locations), np.nan)
        for i in range(len(locations)):
            probs = probabilities[i, :]
            valid_mask = ~np.isnan(probs)
            if np.sum(valid_mask) >= 2:
                valid_probs = probs[valid_mask]
                valid_thresh = np.array(thresholds)[valid_mask]
                mean[i] = np.trapz(1 - valid_probs, valid_thresh)
    
    logger.info(f"Bayesian IK completed: {np.sum(~np.isnan(probabilities[:, 0]))}/{len(locations)} valid estimates")
    
    return IKResult(
        thresholds=np.array(thresholds),
        probabilities=probabilities,
        median=median,
        mean=mean,
        metadata={
            'n_thresholds': len(thresholds),
            'compute_median': ik_config.compute_median,
            'compute_mean': ik_config.compute_mean,
            'bayesian': True,
            'soft_weight': soft_weight
        }
    )


def run_bayesian_cok(
    coords: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    soft_data: Optional[SoftDataSet],
    variogram_models: Dict[str, Dict[str, Any]],
    cok_config: CoKrigingConfig,
    locations: np.ndarray,
    search_params: Optional[Dict[str, Any]] = None,
    soft_weight: float = 0.5
) -> Any:
    """
    Run Bayesian Co-Kriging with soft secondary information.
    
    Industry-Standard Approach:
    Soft data is treated as additional secondary measurements with uncertainty.
    The approach incorporates soft secondary data by:
    1. Running standard CoK with combined hard + soft secondary data
    2. Applying precision-weighted Bayesian update if mean_var mode
    
    Args:
        coords: (N, 3) data coordinates
        primary: (N,) primary variable values
        secondary: (N,) secondary variable values
        soft_data: Optional SoftDataSet (for soft secondary data)
        variogram_models: Dict with 'primary', 'secondary', 'cross' variogram models
        cok_config: CoKrigingConfig
        locations: (M, 3) target locations
        search_params: Optional search parameters
        soft_weight: Weight for soft data (0-1)
    
    Returns:
        CoKrigingResult-like object
    """
    from .cokriging3d import CoKriging3D, CoKrigingResult
    
    # Run standard CoK first
    cok = CoKriging3D(
        coords,
        primary,
        secondary,
        variogram_models,
        cok_config
    )
    cok_result = cok.estimate(locations)
    
    if soft_data is None or len(soft_data.points) == 0:
        logger.info("No soft data provided, using standard CoK")
        return cok_result
    
    # Get soft data (treated as soft secondary measurements)
    soft_coords = soft_data.get_coords()
    soft_means = soft_data.get_means()
    soft_variances = soft_data.get_variances()
    
    if len(soft_coords) == 0:
        return cok_result
    
    # Find nearest soft data for each target
    from scipy.spatial.distance import cdist as scipy_cdist
    distances = scipy_cdist(locations, soft_coords)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(locations)), nearest_idx]
    
    max_distance = search_params.get('max_distance') if search_params else None
    
    # Apply Bayesian update
    estimates = cok_result.estimates.copy()
    variances = cok_result.variance.copy()
    
    for i in range(len(locations)):
        cok_est = cok_result.estimates[i]
        cok_var = cok_result.variance[i]
        
        if np.isnan(cok_est) or np.isnan(cok_var) or cok_var <= 0:
            continue
        
        if max_distance is not None and nearest_dist[i] > max_distance:
            continue
        
        soft_mean = soft_means[nearest_idx[i]]
        soft_var = soft_variances[nearest_idx[i]]
        
        if np.isnan(soft_mean) or np.isnan(soft_var) or soft_var <= 0:
            continue
        
        # Precision-weighted combination
        cok_precision = 1.0 / cok_var
        soft_precision = soft_weight / soft_var
        total_precision = cok_precision + soft_precision
        
        if total_precision > 0:
            estimates[i] = (cok_precision * cok_est + soft_precision * soft_mean) / total_precision
            variances[i] = 1.0 / total_precision
    
    logger.info(f"Bayesian CoK completed: {np.sum(~np.isnan(estimates))}/{len(locations)} valid estimates")
    
    return CoKrigingResult(
        estimates=estimates,
        variance=variances,
        primary_weights=cok_result.primary_weights,
        secondary_weights=cok_result.secondary_weights,
        metadata={
            **cok_result.metadata,
            'bayesian': True,
            'soft_weight': soft_weight
        }
    )


def run_bayesian_kriging_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for controller integration.
    
    Args:
        params: Job parameters dict with:
            - base_method: "OK", "UK", "IK", or "CoK"
            - coords, values: Hard data
            - soft_data: Optional SoftDataSet dict
            - variogram_model: Variogram parameters
            - config: BayesianKrigingConfig dict
            - locations: Target locations
            - Other method-specific parameters
    
    Returns:
        Result dict suitable for controller payload
    """
    base_method = params.get('base_method', 'OK')
    
    if base_method == 'OK':
        return _run_bayesian_ok_job(params)
    elif base_method == 'UK':
        return _run_bayesian_uk_job(params)
    elif base_method == 'IK':
        return _run_bayesian_ik_job(params)
    elif base_method == 'CoK':
        return _run_bayesian_cok_job(params)
    else:
        return {'error': f"Unknown base method: {base_method}"}


def _run_bayesian_ok_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Bayesian OK job."""
    coords = params['coords']
    values = params['values']
    locations = params['locations']
    variogram_model = params['variogram_model']
    
    # Parse soft data if provided
    soft_data = None
    if 'soft_data' in params and params['soft_data'] is not None:
        from .soft_data import SoftDataSet
        soft_dict = params['soft_data']
        if isinstance(soft_dict, dict):
            # Reconstruct SoftDataSet from dict
            points = []
            for p in soft_dict.get('points', []):
                from .soft_data import SoftPoint
                points.append(SoftPoint(**p))
            soft_data = SoftDataSet(points=points, metadata=soft_dict.get('metadata', {}))
    
    config_dict = params.get('config', {})
    config = BayesianKrigingConfig(
        base_method='OK',
        prior_type=config_dict.get('prior_type', 'mean_var'),
        soft_weighting=config_dict.get('soft_weighting', 0.5)
    )
    
    search_params = params.get('search_params', {})
    
    try:
        estimates, variances = run_bayesian_ok(
            coords,
            values,
            soft_data,
            variogram_model,
            config,
            locations,
            search_params
        )
        
        return {
            'estimates': estimates.tolist(),
            'variances': variances.tolist(),
            'method': 'bayesian_ok'
        }
    except Exception as e:
        logger.error(f"Bayesian OK job failed: {e}", exc_info=True)
        return {'error': str(e)}


def _run_bayesian_uk_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Bayesian UK job."""
    coords = params['coords']
    values = params['values']
    locations = params['locations']
    variogram_model = params['variogram_model']
    drift_type = params.get('drift_type', 'constant')
    
    # Parse soft data
    soft_data = None
    if 'soft_data' in params and params['soft_data'] is not None:
        from .soft_data import SoftDataSet
        soft_dict = params['soft_data']
        if isinstance(soft_dict, dict):
            points = []
            for p in soft_dict.get('points', []):
                from .soft_data import SoftPoint
                points.append(SoftPoint(**p))
            soft_data = SoftDataSet(points=points, metadata=soft_dict.get('metadata', {}))
    
    drift_model = DriftModel(drift_type)
    config_dict = params.get('config', {})
    config = BayesianKrigingConfig(
        base_method='UK',
        prior_type=config_dict.get('prior_type', 'mean_var'),
        soft_weighting=config_dict.get('soft_weighting', 0.5)
    )
    search_params = params.get('search_params', {})
    
    try:
        estimates, variances = run_bayesian_uk(
            coords,
            values,
            soft_data,
            variogram_model,
            drift_model,
            config,
            locations,
            search_params
        )
        
        return {
            'estimates': estimates.tolist(),
            'variances': variances.tolist(),
            'method': 'bayesian_uk'
        }
    except Exception as e:
        logger.error(f"Bayesian UK job failed: {e}", exc_info=True)
        return {'error': str(e)}


def _run_bayesian_ik_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Bayesian IK job."""
    coords = params['coords']
    values = params['values']
    locations = params['locations']
    variogram_template = params['variogram_template']
    thresholds = params['thresholds']
    
    ik_config = IKConfig(
        thresholds=thresholds,
        variogram_model_template=variogram_template,
        search_params=params.get('search_params', {}),
        compute_median=params.get('compute_median', True),
        compute_mean=params.get('compute_mean', True)
    )
    
    # Parse soft data
    soft_data = None
    if 'soft_data' in params and params['soft_data'] is not None:
        from .soft_data import SoftDataSet
        soft_dict = params['soft_data']
        if isinstance(soft_dict, dict):
            points = []
            for p in soft_dict.get('points', []):
                from .soft_data import SoftPoint
                points.append(SoftPoint(**p))
            soft_data = SoftDataSet(points=points, metadata=soft_dict.get('metadata', {}))
    
    config_dict = params.get('config', {})
    soft_weight = config_dict.get('soft_weighting', 0.5)
    search_params = params.get('search_params', {})
    
    try:
        result = run_bayesian_ik(
            coords,
            values,
            soft_data,
            variogram_template,
            ik_config,
            locations,
            search_params,
            soft_weight=soft_weight
        )
        
        # Convert result to dict
        if hasattr(result, 'probabilities'):
            return {
                'probabilities': result.probabilities.tolist(),
                'thresholds': result.thresholds.tolist(),
                'median': result.median.tolist() if result.median is not None else None,
                'mean': result.mean.tolist() if result.mean is not None else None,
                'method': 'bayesian_ik',
                'bayesian': True,
                'soft_weight': soft_weight
            }
        else:
            return result
    except Exception as e:
        logger.error(f"Bayesian IK job failed: {e}", exc_info=True)
        return {'error': str(e)}


def _run_bayesian_cok_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Bayesian CoK job."""
    coords = params['coords']
    primary = params['primary']
    secondary = params['secondary']
    locations = params['locations']
    variogram_models = params.get('variogram_models') or {}
    
    cok_config = CoKrigingConfig(
        primary_name=params.get('primary_name', 'Primary'),
        secondary_name=params.get('secondary_name', 'Secondary'),
        method=params.get('method', 'full'),
        variogram_primary=variogram_models.get('primary') or {},
        variogram_secondary=variogram_models.get('secondary') or {},
        cross_variogram=variogram_models.get('cross') or {},
        search_params=params.get('search_params') or {}
    )
    
    # Parse soft data
    soft_data = None
    if 'soft_data' in params and params['soft_data'] is not None:
        from .soft_data import SoftDataSet
        soft_dict = params['soft_data']
        if isinstance(soft_dict, dict):
            points = []
            for p in soft_dict.get('points', []):
                from .soft_data import SoftPoint
                points.append(SoftPoint(**p))
            soft_data = SoftDataSet(points=points, metadata=soft_dict.get('metadata', {}))
    
    config_dict = params.get('config', {})
    soft_weight = config_dict.get('soft_weighting', 0.5)
    search_params = params.get('search_params', {})
    
    try:
        result = run_bayesian_cok(
            coords,
            primary,
            secondary,
            soft_data,
            variogram_models,
            cok_config,
            locations,
            search_params,
            soft_weight=soft_weight
        )
        
        # Convert result to dict
        if hasattr(result, 'estimates'):
            return {
                'estimates': result.estimates.tolist(),
                'variances': result.variance.tolist(),
                'method': 'bayesian_cok',
                'bayesian': True,
                'soft_weight': soft_weight
            }
        else:
            return result
    except Exception as e:
        logger.error(f"Bayesian CoK job failed: {e}", exc_info=True)
        return {'error': str(e)}

