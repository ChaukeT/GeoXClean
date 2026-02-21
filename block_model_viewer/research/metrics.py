"""
Metrics Library

Compute research metrics from existing outputs for experiment evaluation.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def rmse_cv(samples: np.ndarray, estimates: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error for cross-validation.
    
    Args:
        samples: Actual values
        estimates: Estimated/predicted values
    
    Returns:
        RMSE value
    """
    valid_mask = ~(np.isnan(samples) | np.isnan(estimates))
    if np.sum(valid_mask) == 0:
        return np.nan
    
    errors = samples[valid_mask] - estimates[valid_mask]
    rmse = np.sqrt(np.mean(errors ** 2))
    
    return float(rmse)


def mae_cv(samples: np.ndarray, estimates: np.ndarray) -> float:
    """
    Compute Mean Absolute Error for cross-validation.
    
    Args:
        samples: Actual values
        estimates: Estimated/predicted values
    
    Returns:
        MAE value
    """
    valid_mask = ~(np.isnan(samples) | np.isnan(estimates))
    if np.sum(valid_mask) == 0:
        return np.nan
    
    errors = np.abs(samples[valid_mask] - estimates[valid_mask])
    mae = np.mean(errors)
    
    return float(mae)


def smoothing_index(original: np.ndarray, estimate: np.ndarray) -> float:
    """
    Compute smoothing index (ratio of variances).
    
    Smoothing index = var(estimate) / var(original)
    Values < 1 indicate smoothing (variance reduction).
    
    Args:
        original: Original data values
        estimate: Estimated values
    
    Returns:
        Smoothing index
    """
    valid_mask = ~(np.isnan(original) | np.isnan(estimate))
    if np.sum(valid_mask) < 2:
        return np.nan
    
    var_original = np.var(original[valid_mask])
    var_estimate = np.var(estimate[valid_mask])
    
    if var_original == 0:
        return np.nan
    
    smoothing = var_estimate / var_original
    
    return float(smoothing)


def nugget_to_sill_ratio(variogram_model: Dict[str, Any]) -> float:
    """
    Compute nugget-to-sill ratio from variogram model.
    
    Args:
        variogram_model: Variogram model dict with 'nugget' and 'sill' keys
    
    Returns:
        Nugget-to-sill ratio
    """
    nugget = variogram_model.get('nugget', 0.0)
    sill = variogram_model.get('sill', 1.0)
    
    if sill == 0:
        return np.nan
    
    ratio = nugget / sill
    
    return float(ratio)


def grade_tonnage_loss(gt_curve_ref: np.ndarray, gt_curve_est: np.ndarray) -> float:
    """
    Compute grade-tonnage curve loss (integrated absolute difference).
    
    Args:
        gt_curve_ref: Reference grade-tonnage curve (shape: [n_cutoffs, 2] or [n_cutoffs])
        gt_curve_est: Estimated grade-tonnage curve (same shape)
    
    Returns:
        Integrated absolute difference
    """
    if gt_curve_ref.shape != gt_curve_est.shape:
        logger.warning("GT curve shapes don't match")
        return np.nan
    
    # Handle 2D arrays (cutoff, tonnage pairs)
    if gt_curve_ref.ndim == 2:
        if gt_curve_ref.shape[1] == 2:
            # Extract tonnage column
            ref_tonnage = gt_curve_ref[:, 1]
            est_tonnage = gt_curve_est[:, 1]
        else:
            ref_tonnage = gt_curve_ref.ravel()
            est_tonnage = gt_curve_est.ravel()
    else:
        ref_tonnage = gt_curve_ref
        est_tonnage = gt_curve_est
    
    # Compute integrated absolute difference
    diff = np.abs(ref_tonnage - est_tonnage)
    loss = np.trapz(diff)
    
    return float(loss)


def npv_stat(npv_samples: np.ndarray) -> Dict[str, float]:
    """
    Compute NPV statistics from samples.
    
    Args:
        npv_samples: Array of NPV values
    
    Returns:
        Dict with mean, std, p10, p50, p90, min, max
    """
    valid_samples = npv_samples[~np.isnan(npv_samples)]
    
    if len(valid_samples) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'p10': np.nan,
            'p50': np.nan,
            'p90': np.nan,
            'min': np.nan,
            'max': np.nan
        }
    
    return {
        'mean': float(np.mean(valid_samples)),
        'std': float(np.std(valid_samples)),
        'p10': float(np.percentile(valid_samples, 10)),
        'p50': float(np.percentile(valid_samples, 50)),
        'p90': float(np.percentile(valid_samples, 90)),
        'min': float(np.min(valid_samples)),
        'max': float(np.max(valid_samples))
    }


def irr_stat(irr_samples: np.ndarray) -> Dict[str, float]:
    """
    Compute IRR statistics from samples.
    
    Args:
        irr_samples: Array of IRR values
    
    Returns:
        Dict with mean, std, p10, p50, p90, min, max
    """
    valid_samples = irr_samples[~np.isnan(irr_samples)]
    
    if len(valid_samples) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'p10': np.nan,
            'p50': np.nan,
            'p90': np.nan,
            'min': np.nan,
            'max': np.nan
        }
    
    return {
        'mean': float(np.mean(valid_samples)),
        'std': float(np.std(valid_samples)),
        'p10': float(np.percentile(valid_samples, 10)),
        'p50': float(np.percentile(valid_samples, 50)),
        'p90': float(np.percentile(valid_samples, 90)),
        'min': float(np.min(valid_samples)),
        'max': float(np.max(valid_samples))
    }


def risk_adjusted_npv(npv_samples: np.ndarray, risk_aversion: float = 0.5) -> float:
    """
    Compute risk-adjusted NPV using utility function.
    
    Args:
        npv_samples: Array of NPV values
        risk_aversion: Risk aversion coefficient (0 = risk neutral, >0 = risk averse)
    
    Returns:
        Risk-adjusted NPV
    """
    valid_samples = npv_samples[~np.isnan(npv_samples)]
    
    if len(valid_samples) == 0:
        return np.nan
    
    if risk_aversion == 0:
        return float(np.mean(valid_samples))
    
    # Exponential utility: U(x) = -exp(-risk_aversion * x)
    # Risk-adjusted NPV = -log(mean(exp(-risk_aversion * x))) / risk_aversion
    utilities = -np.exp(-risk_aversion * valid_samples)
    mean_utility = np.mean(utilities)
    
    if mean_utility >= 0:
        return np.nan
    
    risk_adj_npv = -np.log(-mean_utility) / risk_aversion
    
    return float(risk_adj_npv)


def schedule_exposure(profile: Any, threshold: float) -> float:
    """
    Compute schedule exposure (fraction of periods above risk threshold).
    
    Args:
        profile: ScheduleRiskProfile instance
        threshold: Risk threshold value
    
    Returns:
        Fraction of periods above threshold
    """
    if not hasattr(profile, 'periods'):
        return np.nan
    
    periods = profile.periods
    if not periods:
        return np.nan
    
    # Count periods with combined_risk_score above threshold
    above_threshold = sum(
        1 for p in periods
        if hasattr(p, 'combined_risk_score') and p.combined_risk_score is not None
        and p.combined_risk_score > threshold
    )
    
    exposure = above_threshold / len(periods)
    
    return float(exposure)


def risk_percentile(profile: Any, p: float = 0.9) -> float:
    """
    Compute risk percentile from schedule profile.
    
    Args:
        profile: ScheduleRiskProfile instance
        p: Percentile (0-1)
    
    Returns:
        Risk value at percentile
    """
    if not hasattr(profile, 'periods'):
        return np.nan
    
    periods = profile.periods
    if not periods:
        return np.nan
    
    # Extract risk scores
    risk_scores = [
        p.combined_risk_score
        for p in periods
        if hasattr(p, 'combined_risk_score') and p.combined_risk_score is not None
    ]
    
    if not risk_scores:
        return np.nan
    
    percentile_value = np.percentile(risk_scores, p * 100)
    
    return float(percentile_value)


def compute_metrics(metric_names: List[str], context: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute multiple metrics from context.
    
    Args:
        metric_names: List of metric names to compute
        context: Dict containing:
            - 'samples', 'estimates' for geostats metrics
            - 'npv_samples', 'irr_samples' for economic metrics
            - 'economic_result' (EconomicUncertaintyResult) for economic metrics
            - 'risk_profile' (ScheduleRiskProfile) for risk metrics
            - 'variogram_model' for variogram metrics
            - 'gt_curve_ref', 'gt_curve_est' for GT loss
    
    Returns:
        Dict mapping metric names to values
    """
    results = {}
    
    for metric_name in metric_names:
        try:
            if metric_name == 'rmse_cv':
                if 'samples' in context and 'estimates' in context:
                    results[metric_name] = rmse_cv(
                        np.asarray(context['samples']),
                        np.asarray(context['estimates'])
                    )
            
            elif metric_name == 'mae_cv':
                if 'samples' in context and 'estimates' in context:
                    results[metric_name] = mae_cv(
                        np.asarray(context['samples']),
                        np.asarray(context['estimates'])
                    )
            
            elif metric_name == 'smoothing_index':
                if 'original' in context and 'estimate' in context:
                    results[metric_name] = smoothing_index(
                        np.asarray(context['original']),
                        np.asarray(context['estimate'])
                    )
            
            elif metric_name == 'nugget_to_sill_ratio':
                if 'variogram_model' in context:
                    results[metric_name] = nugget_to_sill_ratio(context['variogram_model'])
            
            elif metric_name == 'gt_loss':
                if 'gt_curve_ref' in context and 'gt_curve_est' in context:
                    results[metric_name] = grade_tonnage_loss(
                        np.asarray(context['gt_curve_ref']),
                        np.asarray(context['gt_curve_est'])
                    )
            
            elif metric_name.startswith('npv_'):
                if 'npv_samples' in context:
                    npv_stats = npv_stat(np.asarray(context['npv_samples']))
                    stat_name = metric_name.replace('npv_', '')
                    if stat_name in npv_stats:
                        results[metric_name] = npv_stats[stat_name]
                elif 'economic_result' in context:
                    econ_result = context['economic_result']
                    if hasattr(econ_result, 'npv_samples'):
                        npv_stats = npv_stat(econ_result.npv_samples)
                        stat_name = metric_name.replace('npv_', '')
                        if stat_name in npv_stats:
                            results[metric_name] = npv_stats[stat_name]
            
            elif metric_name.startswith('irr_'):
                if 'irr_samples' in context:
                    irr_stats = irr_stat(np.asarray(context['irr_samples']))
                    stat_name = metric_name.replace('irr_', '')
                    if stat_name in irr_stats:
                        results[metric_name] = irr_stats[stat_name]
                elif 'economic_result' in context:
                    econ_result = context['economic_result']
                    if hasattr(econ_result, 'irr_samples'):
                        irr_stats = irr_stat(econ_result.irr_samples)
                        stat_name = metric_name.replace('irr_', '')
                        if stat_name in irr_stats:
                            results[metric_name] = irr_stats[stat_name]
            
            elif metric_name == 'risk_adjusted_npv':
                if 'npv_samples' in context:
                    risk_aversion = context.get('risk_aversion', 0.5)
                    results[metric_name] = risk_adjusted_npv(
                        np.asarray(context['npv_samples']),
                        risk_aversion
                    )
            
            elif metric_name.startswith('schedule_exposure_'):
                if 'risk_profile' in context:
                    threshold = float(metric_name.split('_')[-1])
                    results[metric_name] = schedule_exposure(
                        context['risk_profile'],
                        threshold
                    )
            
            elif metric_name.startswith('risk_percentile_'):
                if 'risk_profile' in context:
                    p = float(metric_name.split('_')[-1]) / 100.0
                    results[metric_name] = risk_percentile(
                        context['risk_profile'],
                        p
                    )
            
        except Exception as e:
            logger.warning(f"Failed to compute metric '{metric_name}': {e}")
            results[metric_name] = np.nan
    
    return results

