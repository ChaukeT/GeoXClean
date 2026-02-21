"""
Probabilistic Geotechnical Analysis Wrapper.

Provides Monte Carlo and LHS analysis for stope stability and slope risk
with uncertainty quantification.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

from .dataclasses import (
    StopeStabilityInput, StopeStabilityResult,
    SlopeRiskInput, SlopeRiskResult,
    GeotechMCResult
)
from .stope_stability import evaluate_stope_probabilistic
from .slope_risk import evaluate_slope

logger = logging.getLogger(__name__)


def run_stope_stability_monte_carlo(
    input: StopeStabilityInput,
    n_realizations: int,
    sampler: Optional[Any] = None,
    **kwargs
) -> GeotechMCResult:
    """
    Run Monte Carlo analysis for stope stability.
    
    Args:
        input: Base StopeStabilityInput
        n_realizations: Number of realizations
        sampler: Optional sampler instance (LHS or Monte Carlo)
        **kwargs: Additional parameters for evaluate_stope_probabilistic
    
    Returns:
        GeotechMCResult with distributions and statistics
    """
    logger.info(f"Running stope stability Monte Carlo: {n_realizations} realizations")
    
    # Prepare parameters
    params = {
        'n_realizations': n_realizations,
        'sampler': 'monte_carlo' if sampler is None else 'lhs',
        **kwargs
    }
    
    # Run probabilistic evaluation
    results = evaluate_stope_probabilistic(input, params)
    
    # Extract arrays
    stability_numbers = np.array([r.stability_number for r in results])
    stability_classes = [r.stability_class for r in results]
    
    # Create result object
    mc_result = GeotechMCResult(
        input_params=input.__dict__,
        n_realizations=n_realizations,
        stability_numbers=stability_numbers,
        stability_classes=stability_classes
    )
    
    # Compute summary statistics
    mc_result.compute_summary_stats()
    
    # Compute exceedance curves
    sorted_n = np.sort(stability_numbers)
    exceedance_probs = np.linspace(1.0, 0.0, len(sorted_n))
    mc_result.exceedance_curves['stability_number'] = np.column_stack([
        sorted_n, exceedance_probs
    ])
    
    logger.info(f"Monte Carlo complete: Mean N={np.mean(stability_numbers):.2f}")
    
    return mc_result


def run_slope_risk_monte_carlo(
    input: SlopeRiskInput,
    n_realizations: int,
    sampler: Optional[Any] = None,
    **kwargs
) -> GeotechMCResult:
    """
    Run Monte Carlo analysis for slope risk.
    
    Args:
        input: Base SlopeRiskInput
        n_realizations: Number of realizations
        sampler: Optional sampler instance
        **kwargs: Parameter distributions:
            - rmr_dist: Distribution for RMR
            - slope_angle_dist: Distribution for slope angle
            - bench_height_dist: Distribution for bench height
    
    Returns:
        GeotechMCResult with distributions and statistics
    """
    logger.info(f"Running slope risk Monte Carlo: {n_realizations} realizations")
    
    # Sample uncertain parameters
    rmr_dist = kwargs.get('rmr_dist', {
        'type': 'normal',
        'mean': input.rock_mass_properties.get('RMR', 50.0),
        'std': 10.0
    })
    
    slope_angle_dist = kwargs.get('slope_angle_dist', {
        'type': 'normal',
        'mean': input.overall_slope_angle,
        'std': 2.0
    })
    
    bench_height_dist = kwargs.get('bench_height_dist', {
        'type': 'normal',
        'mean': input.bench_height,
        'std': 1.0
    })
    
    # Sample parameters
    sampler_type = 'monte_carlo' if sampler is None else 'lhs'
    
    rmr_samples = _sample_parameter(rmr_dist, n_realizations, sampler_type)
    slope_angle_samples = _sample_parameter(slope_angle_dist, n_realizations, sampler_type)
    bench_height_samples = _sample_parameter(bench_height_dist, n_realizations, sampler_type)
    
    # Evaluate each realization
    risk_indices = []
    risk_classes = []
    
    for i in range(n_realizations):
        # Create modified input
        modified_props = input.rock_mass_properties.copy()
        modified_props['RMR'] = rmr_samples[i]
        
        modified_input = SlopeRiskInput(
            bench_height=bench_height_samples[i],
            overall_slope_angle=slope_angle_samples[i],
            pit_wall_orientation=input.pit_wall_orientation,
            rock_mass_properties=modified_props,
            water_present=input.water_present,
            structural_features=input.structural_features
        )
        
        result = evaluate_slope(modified_input)
        risk_indices.append(result.risk_index)
        risk_classes.append(result.qualitative_class)
    
    # Create result object
    mc_result = GeotechMCResult(
        input_params=input.__dict__,
        n_realizations=n_realizations,
        risk_indices=np.array(risk_indices),
        risk_classes=risk_classes
    )
    
    # Compute summary statistics
    mc_result.compute_summary_stats()
    
    # Compute exceedance curves
    sorted_risk = np.sort(risk_indices)
    exceedance_probs = np.linspace(1.0, 0.0, len(sorted_risk))
    mc_result.exceedance_curves['risk_index'] = np.column_stack([
        sorted_risk, exceedance_probs
    ])
    
    logger.info(f"Monte Carlo complete: Mean risk index={np.mean(risk_indices):.2f}")
    
    return mc_result


def _sample_parameter(
    dist_params: Dict[str, Any],
    n_samples: int,
    sampler_type: str
) -> np.ndarray:
    """Sample parameter from distribution (same as in stope_stability.py)."""
    dist_type = dist_params.get('type', 'normal').lower()
    mean = dist_params.get('mean', 1.0)
    std = dist_params.get('std', 0.1)
    min_val = dist_params.get('min', None)
    max_val = dist_params.get('max', None)
    
    if sampler_type == 'lhs':
        try:
            from ..uncertainty_engine.lhs_sampler import LHSSampler, LHSConfig
            config = LHSConfig(n_samples=n_samples, n_dimensions=1)
            lhs = LHSSampler(config)
            uniform_samples = lhs.sample_uniform()[:, 0]
        except ImportError:
            uniform_samples = np.random.uniform(0, 1, n_samples)
    else:
        uniform_samples = np.random.uniform(0, 1, n_samples)
    
    # Transform to target distribution
    if dist_type == 'normal':
        from scipy.stats import norm
        samples = norm.ppf(uniform_samples, loc=mean, scale=std)
    elif dist_type == 'uniform':
        if min_val is None or max_val is None:
            min_val = mean - std
            max_val = mean + std
        samples = min_val + uniform_samples * (max_val - min_val)
    elif dist_type == 'lognormal':
        from scipy.stats import lognorm
        sigma = np.sqrt(np.log(1 + (std/mean)**2))
        mu = np.log(mean) - 0.5 * sigma**2
        samples = lognorm.ppf(uniform_samples, s=sigma, scale=np.exp(mu))
    else:
        from scipy.stats import norm
        samples = norm.ppf(uniform_samples, loc=mean, scale=std)
    
    if min_val is not None:
        samples = np.maximum(samples, min_val)
    if max_val is not None:
        samples = np.minimum(samples, max_val)
    
    return samples

