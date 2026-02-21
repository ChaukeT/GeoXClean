"""
Mathews Stability Graph Engine for Stope Stability Analysis.

Implements the Mathews Stability Graph method for evaluating underground
stope stability and support requirements.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any

from .dataclasses import StopeStabilityInput, StopeStabilityResult

logger = logging.getLogger(__name__)


def evaluate_stope(input: StopeStabilityInput) -> StopeStabilityResult:
    """
    Evaluate stope stability using Mathews Stability Graph method.
    
    The stability number N is computed as:
        N = Q' * A * B * C
    
    Where:
        Q' = Modified Q-value
        A = Stress factor
        B = Joint orientation factor
        C = Gravity factor
    
    Args:
        input: StopeStabilityInput with geometry and rock mass parameters
    
    Returns:
        StopeStabilityResult with stability classification
    """
    # Compute stability number
    stability_number = (
        input.q_prime *
        input.stress_factor *
        input.joint_orientation_factor *
        input.gravity_factor
    )
    
    # Determine stability class based on position on stability graph
    # Simplified classification based on N and hydraulic radius
    hr = input.hydraulic_radius
    
    # Stability boundaries (simplified Mathews graph)
    # These are approximate and should be calibrated for specific mines
    if stability_number >= 10 and hr <= 10:
        stability_class = StopeStabilityResult.STABILITY_CLASSES['STABLE']
        fos = stability_number / 5.0  # Approximate FOS
        prob_instability = 0.05
        support_class = StopeStabilityResult.SUPPORT_CLASSES['NONE']
        notes = "Stable - no support required"
    elif stability_number >= 5 and hr <= 15:
        stability_class = StopeStabilityResult.STABILITY_CLASSES['TRANSITION']
        fos = stability_number / 4.0
        prob_instability = 0.20
        support_class = StopeStabilityResult.SUPPORT_CLASSES['LIGHT']
        notes = "Transition zone - light support recommended"
    elif stability_number >= 2 and hr <= 20:
        stability_class = StopeStabilityResult.STABILITY_CLASSES['TRANSITION']
        fos = stability_number / 3.0
        prob_instability = 0.40
        support_class = StopeStabilityResult.SUPPORT_CLASSES['MODERATE']
        notes = "Transition zone - moderate support required"
    else:
        stability_class = StopeStabilityResult.STABILITY_CLASSES['CAVING']
        fos = stability_number / 2.0
        prob_instability = 0.80
        support_class = StopeStabilityResult.SUPPORT_CLASSES['HEAVY']
        notes = "Caving/overbreak expected - heavy support required"
    
    # Adjust probability based on factor of safety
    if fos < 1.0:
        prob_instability = min(0.95, prob_instability + (1.0 - fos) * 0.3)
    elif fos > 2.0:
        prob_instability = max(0.05, prob_instability - (fos - 2.0) * 0.1)
    
    result = StopeStabilityResult(
        stability_number=stability_number,
        factor_of_safety=fos,
        probability_of_instability=prob_instability,
        stability_class=stability_class,
        recommended_support_class=support_class,
        notes=notes
    )
    
    logger.info(f"Stope stability evaluated: N={stability_number:.2f}, Class={stability_class}")
    
    return result


def evaluate_stope_probabilistic(
    input: StopeStabilityInput,
    params: Dict[str, Any]
) -> List[StopeStabilityResult]:
    """
    Evaluate stope stability with probabilistic parameter uncertainty.
    
    Uses Monte Carlo or LHS sampling over uncertain parameters:
    - Q' (modified Q-value)
    - Stress factor
    - Span uncertainty
    
    Args:
        input: Base StopeStabilityInput
        params: Parameters dict:
            - n_realizations: Number of realizations
            - q_prime_dist: Distribution for Q' (dict with 'type', 'mean', 'std')
            - stress_factor_dist: Distribution for stress factor
            - span_dist: Distribution for span (optional)
            - sampler: Optional sampler ('monte_carlo' or 'lhs')
    
    Returns:
        List of StopeStabilityResult for each realization
    """
    n_realizations = params.get('n_realizations', 100)
    sampler_type = params.get('sampler', 'monte_carlo')
    
    # Sample uncertain parameters
    q_prime_samples = _sample_parameter(
        params.get('q_prime_dist', {'type': 'normal', 'mean': input.q_prime, 'std': input.q_prime * 0.2}),
        n_realizations,
        sampler_type
    )
    
    stress_factor_samples = _sample_parameter(
        params.get('stress_factor_dist', {'type': 'normal', 'mean': input.stress_factor, 'std': input.stress_factor * 0.15}),
        n_realizations,
        sampler_type
    )
    
    span_samples = _sample_parameter(
        params.get('span_dist', {'type': 'normal', 'mean': input.span, 'std': input.span * 0.1}),
        n_realizations,
        sampler_type
    ) if 'span_dist' in params else np.full(n_realizations, input.span)
    
    # Evaluate each realization
    results = []
    for i in range(n_realizations):
        # Create modified input
        modified_input = StopeStabilityInput(
            span=span_samples[i],
            height=input.height,
            q_prime=q_prime_samples[i],
            stress_factor=stress_factor_samples[i],
            joint_orientation_factor=input.joint_orientation_factor,
            gravity_factor=input.gravity_factor,
            dilution_allowance=input.dilution_allowance,
            rock_mass_properties=input.rock_mass_properties
        )
        
        result = evaluate_stope(modified_input)
        results.append(result)
    
    logger.info(f"Probabilistic evaluation complete: {n_realizations} realizations")
    
    return results


def _sample_parameter(
    dist_params: Dict[str, Any],
    n_samples: int,
    sampler_type: str
) -> np.ndarray:
    """
    Sample parameter from distribution.
    
    Args:
        dist_params: Distribution parameters dict
        n_samples: Number of samples
        sampler_type: 'monte_carlo' or 'lhs'
    
    Returns:
        Array of sampled values
    """
    dist_type = dist_params.get('type', 'normal').lower()
    mean = dist_params.get('mean', 1.0)
    std = dist_params.get('std', 0.1)
    min_val = dist_params.get('min', None)
    max_val = dist_params.get('max', None)
    
    if sampler_type == 'lhs':
        # Use LHS for better coverage
        try:
            from ..uncertainty_engine.lhs_sampler import LHSSampler, LHSConfig
            config = LHSConfig(n_samples=n_samples, n_dimensions=1)
            lhs = LHSSampler(config)
            uniform_samples = lhs.sample_uniform()[:, 0]
        except ImportError:
            uniform_samples = np.random.uniform(0, 1, n_samples)
    else:
        # Simple Monte Carlo
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
        # Convert mean/std to lognormal parameters
        sigma = np.sqrt(np.log(1 + (std/mean)**2))
        mu = np.log(mean) - 0.5 * sigma**2
        samples = lognorm.ppf(uniform_samples, s=sigma, scale=np.exp(mu))
    else:
        # Default to normal
        from scipy.stats import norm
        samples = norm.ppf(uniform_samples, loc=mean, scale=std)
    
    # Clip to bounds if specified
    if min_val is not None:
        samples = np.maximum(samples, min_val)
    if max_val is not None:
        samples = np.minimum(samples, max_val)
    
    return samples

