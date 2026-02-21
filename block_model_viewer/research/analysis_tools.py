"""
Research Analysis Tools
=======================

Professional geostatistical analysis tools for research and validation.
Implements industry-standard diagnostics and metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ESTIMATION ANALYSIS
# ============================================================================

def compute_kriging_summary_table(
    composites: pd.DataFrame,
    estimates: np.ndarray,
    variable: str,
    domain_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute kriging summary table per domain.
    
    Metrics:
    - Global mean of composites (declustered + raw)
    - Global mean of estimates
    - Bias % (estimate/composite)
    - Regression slope (composite vs estimate)
    - Intercept
    - R²
    - Mean absolute error (MAE)
    - RMSE
    
    Args:
        composites: DataFrame with composite data
        estimates: Array of kriging estimates
        variable: Variable name
        domain_col: Optional domain column name
        
    Returns:
        DataFrame with summary statistics per domain
    """
    results = []
    
    if domain_col and domain_col in composites.columns:
        domains = composites[domain_col].unique()
    else:
        domains = ['All']
    
    for domain in domains:
        if domain == 'All':
            comp_data = composites[variable].dropna()
            est_data = estimates[~np.isnan(estimates)]
        else:
            mask = composites[domain_col] == domain
            comp_data = composites.loc[mask, variable].dropna()
            est_data = estimates[mask][~np.isnan(estimates[mask])]
        
        if len(comp_data) == 0 or len(est_data) == 0:
            continue
        
        # Match lengths (use minimum)
        n = min(len(comp_data), len(est_data))
        comp_vals = comp_data.values[:n]
        est_vals = est_data[:n]
        
        # Basic statistics
        comp_mean = comp_vals.mean()
        est_mean = est_vals.mean()
        bias_pct = ((est_mean - comp_mean) / comp_mean * 100) if comp_mean != 0 else np.nan
        
        # Regression statistics
        if len(comp_vals) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(comp_vals, est_vals)
            r_squared = r_value ** 2
        else:
            slope = intercept = r_squared = np.nan
        
        # Error metrics
        mae = mean_absolute_error(comp_vals, est_vals)
        rmse = np.sqrt(mean_squared_error(comp_vals, est_vals))
        
        results.append({
            'Domain': domain,
            'N_Samples': len(comp_vals),
            'Composite_Mean': comp_mean,
            'Estimate_Mean': est_mean,
            'Bias_Pct': bias_pct,
            'Slope': slope,
            'Intercept': intercept,
            'R_Squared': r_squared,
            'MAE': mae,
            'RMSE': rmse
        })
    
    return pd.DataFrame(results)


def compute_slope_of_regression(
    composites: pd.DataFrame,
    estimates: np.ndarray,
    variable: str,
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute slope of regression diagnostics.
    
    Slope ≈ 1 = good
    Slope < 1 = smoothing
    Slope > 1 = instability or clustering
    
    Args:
        composites: DataFrame with composite data
        estimates: Array of estimates
        variable: Variable name
        group_by: Optional grouping column (domain, bench, lithology)
        
    Returns:
        DataFrame with slope statistics
    """
    results = []
    
    if group_by and group_by in composites.columns:
        groups = composites[group_by].unique()
    else:
        groups = ['All']
    
    for group in groups:
        if group == 'All':
            comp_data = composites[variable].dropna()
            est_data = estimates[~np.isnan(estimates)]
        else:
            mask = composites[group_by] == group
            comp_data = composites.loc[mask, variable].dropna()
            est_data = estimates[mask][~np.isnan(estimates[mask])]
        
        if len(comp_data) == 0 or len(est_data) == 0:
            continue
        
        n = min(len(comp_data), len(est_data))
        comp_vals = comp_data.values[:n]
        est_vals = est_data[:n]
        
        if len(comp_vals) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(comp_vals, est_vals)
            r_squared = r_value ** 2
        else:
            slope = intercept = r_squared = std_err = np.nan
        
        results.append({
            'Group': group,
            'N': len(comp_vals),
            'Slope': slope,
            'Intercept': intercept,
            'R_Squared': r_squared,
            'Std_Error': std_err,
            'P_Value': p_value if len(comp_vals) > 1 else np.nan
        })
    
    return pd.DataFrame(results)


def leave_one_out_cross_validation(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    kriging_func,
    kriging_params: Dict[str, Any],
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Perform Leave-One-Out Cross-Validation (LOOCV).
    
    Args:
        data_coords: (N, 3) array of data coordinates
        data_values: (N,) array of data values
        kriging_func: Kriging function to use
        kriging_params: Parameters for kriging function
        progress_callback: Optional progress callback
        
    Returns:
        Dictionary with:
        - errors: Array of errors (actual - estimated)
        - estimates: Array of LOOCV estimates
        - actuals: Array of actual values
        - mae: Mean absolute error
        - rmse: Root mean squared error
        - bias: Mean bias
        - r_squared: R²
    """
    n = len(data_values)
    errors = np.zeros(n)
    estimates_loocv = np.zeros(n)
    actuals = data_values.copy()
    
    for i in range(n):
        if progress_callback:
            progress_callback(i + 1, f"LOOCV {i+1}/{n}")
        
        # Remove point i
        mask = np.arange(n) != i
        train_coords = data_coords[mask]
        train_values = data_values[mask]
        test_coord = data_coords[i:i+1]
        
        try:
            # Estimate at point i using remaining data
            est, var = kriging_func(
                train_coords,
                train_values,
                test_coord,
                **kriging_params
            )
            estimates_loocv[i] = est[0] if len(est) > 0 else np.nan
        except Exception as e:
            logger.warning(f"LOOCV failed for point {i}: {e}")
            estimates_loocv[i] = np.nan
        
        errors[i] = actuals[i] - estimates_loocv[i]
    
    # Compute statistics
    valid_mask = ~np.isnan(estimates_loocv)
    if valid_mask.sum() > 0:
        valid_errors = errors[valid_mask]
        valid_actuals = actuals[valid_mask]
        valid_estimates = estimates_loocv[valid_mask]
        
        mae = np.mean(np.abs(valid_errors))
        rmse = np.sqrt(np.mean(valid_errors ** 2))
        bias = np.mean(valid_errors)
        
        if len(valid_actuals) > 1:
            r_squared = r2_score(valid_actuals, valid_estimates)
        else:
            r_squared = np.nan
    else:
        mae = rmse = bias = r_squared = np.nan
    
    return {
        'errors': errors,
        'estimates': estimates_loocv,
        'actuals': actuals,
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'r_squared': r_squared
    }


def compute_swath_plot_data(
    composites: pd.DataFrame,
    estimates: np.ndarray,
    grid_coords: np.ndarray,
    direction: str = 'X',
    n_bins: int = 50
) -> pd.DataFrame:
    """
    Compute swath plot data for composites and estimates.
    
    Args:
        composites: DataFrame with composite data
        estimates: Array of estimates
        grid_coords: (N, 3) array of grid coordinates
        direction: 'X', 'Y', or 'Z'
        n_bins: Number of bins
        
    Returns:
        DataFrame with swath statistics per bin
    """
    if direction.upper() == 'X':
        comp_coords = composites['X'].values
        grid_coords_1d = grid_coords[:, 0]
    elif direction.upper() == 'Y':
        comp_coords = composites['Y'].values
        grid_coords_1d = grid_coords[:, 1]
    else:  # Z
        comp_coords = composites['Z'].values
        grid_coords_1d = grid_coords[:, 2]
    
    # Create bins
    all_coords = np.concatenate([comp_coords, grid_coords_1d])
    min_coord = all_coords.min()
    max_coord = all_coords.max()
    bins = np.linspace(min_coord, max_coord, n_bins + 1)
    
    results = []
    for i in range(n_bins):
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_center = (bin_min + bin_max) / 2
        
        # Composite statistics
        comp_mask = (comp_coords >= bin_min) & (comp_coords < bin_max)
        comp_in_bin = composites.loc[comp_mask, composites.columns[composites.dtypes == np.number].tolist()[0]]
        comp_mean = comp_in_bin.mean() if len(comp_in_bin) > 0 else np.nan
        comp_std = comp_in_bin.std() if len(comp_in_bin) > 0 else np.nan
        comp_count = len(comp_in_bin)
        
        # Estimate statistics
        est_mask = (grid_coords_1d >= bin_min) & (grid_coords_1d < bin_max)
        est_in_bin = estimates[est_mask]
        est_mean = np.nanmean(est_in_bin) if len(est_in_bin) > 0 else np.nan
        est_std = np.nanstd(est_in_bin) if len(est_in_bin) > 0 else np.nan
        est_count = len(est_in_bin[~np.isnan(est_in_bin)])
        
        results.append({
            'Bin_Center': bin_center,
            'Comp_Mean': comp_mean,
            'Comp_Std': comp_std,
            'Comp_Count': comp_count,
            'Est_Mean': est_mean,
            'Est_Std': est_std,
            'Est_Count': est_count
        })
    
    return pd.DataFrame(results)


# ============================================================================
# SIMULATION ANALYSIS
# ============================================================================

def compute_simulation_reproduction_stats(
    composites: np.ndarray,
    realizations: np.ndarray,
    n_samples: int = 1000
) -> Dict[str, Any]:
    """
    Compute histogram and variogram reproduction statistics.
    
    Args:
        composites: Array of composite values
        realizations: (nreal, nz, ny, nx) array of realizations
        n_samples: Number of samples to use for comparison
        
    Returns:
        Dictionary with reproduction statistics
    """
    # Sample composites
    if len(composites) > n_samples:
        comp_sample = np.random.choice(composites, n_samples, replace=False)
    else:
        comp_sample = composites
    
    # Sample from realizations
    nreal = realizations.shape[0]
    real_samples = []
    for i in range(nreal):
        real_flat = realizations[i].ravel()
        valid = real_flat[~np.isnan(real_flat)]
        if len(valid) > n_samples:
            real_samples.append(np.random.choice(valid, n_samples, replace=False))
        else:
            real_samples.append(valid)
    
    # Histogram statistics
    comp_mean = np.mean(comp_sample)
    comp_var = np.var(comp_sample)
    comp_std = np.std(comp_sample)
    comp_skew = stats.skew(comp_sample) if len(comp_sample) > 2 else np.nan
    
    real_means = [np.mean(s) for s in real_samples]
    real_vars = [np.var(s) for s in real_samples]
    real_stds = [np.std(s) for s in real_samples]
    real_skews = [stats.skew(s) if len(s) > 2 else np.nan for s in real_samples]
    
    return {
        'composite_mean': comp_mean,
        'composite_variance': comp_var,
        'composite_std': comp_std,
        'composite_skewness': comp_skew,
        'realization_mean_mean': np.mean(real_means),
        'realization_mean_std': np.std(real_means),
        'realization_var_mean': np.mean(real_vars),
        'realization_var_std': np.std(real_vars),
        'realization_std_mean': np.mean(real_stds),
        'realization_std_std': np.std(real_stds),
        'realization_skew_mean': np.nanmean(real_skews),
        'realization_skew_std': np.nanstd(real_skews)
    }


def compute_uncertainty_grids(
    realizations: np.ndarray,
    cutoffs: List[float]
) -> Dict[str, np.ndarray]:
    """
    Compute uncertainty grids from realizations.
    
    Returns:
        Dictionary with:
        - mean: E[Z]
        - variance: Var(Z)
        - std: Standard deviation
        - p10, p50, p90: Percentiles
        - prob_above_cutoff: P(Z > cutoff) for each cutoff
    """
    mean = np.nanmean(realizations, axis=0)
    variance = np.nanvar(realizations, axis=0)
    std = np.sqrt(variance)
    
    p10 = np.nanpercentile(realizations, 10, axis=0)
    p50 = np.nanpercentile(realizations, 50, axis=0)
    p90 = np.nanpercentile(realizations, 90, axis=0)
    
    prob_above_cutoff = {}
    for cutoff in cutoffs:
        prob_above_cutoff[f'prob_above_{cutoff}'] = np.nanmean(realizations > cutoff, axis=0)
    
    return {
        'mean': mean,
        'variance': variance,
        'std': std,
        'p10': p10,
        'p50': p50,
        'p90': p90,
        **prob_above_cutoff
    }

