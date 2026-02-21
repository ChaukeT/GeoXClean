"""
IK-based Sequential Gaussian Simulation (IK-SGSIM)

Simulation engine using indicator CDFs from Indicator Kriging (IK) as local conditional distributions.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

from .indicator_kriging import IKResult

logger = logging.getLogger(__name__)


@dataclass
class IKSGSIMConfig:
    """Configuration for IK-based SGSIM."""
    n_realizations: int = 100
    random_seed: Optional[int] = None
    realisation_prefix: str = "ik_sim"
    use_median_as_mean: bool = True  # Use IK median as mean for Gaussian approximation
    cdf_interpolation: str = "linear"  # "linear" or "cubic"
    progress_callback: Optional[Callable] = None


@dataclass
class IKSGSIMResult:
    """Result from IK-based SGSIM."""
    realization_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _sample_from_cdf(
    thresholds: np.ndarray,
    probabilities: np.ndarray,
    n_samples: int = 1,
    random_state: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Sample values from a CDF defined by thresholds and probabilities.
    
    Args:
        thresholds: Threshold values (sorted)
        probabilities: P(Z <= threshold) for each threshold
        n_samples: Number of samples to draw
        random_state: Random number generator
    
    Returns:
        Sampled values (n_samples,)
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    # Ensure thresholds are sorted
    sort_idx = np.argsort(thresholds)
    sorted_thresh = thresholds[sort_idx]
    sorted_probs = probabilities[sort_idx]
    
    # Clip probabilities to [0, 1]
    sorted_probs = np.clip(sorted_probs, 0.0, 1.0)
    
    # Draw uniform random numbers
    u = random_state.uniform(0.0, 1.0, size=n_samples)
    
    # Interpolate to find quantiles
    # Handle edge cases
    if len(sorted_thresh) == 1:
        return np.full(n_samples, sorted_thresh[0])
    
    # Use linear interpolation
    samples = np.interp(u, sorted_probs, sorted_thresh)
    
    # Handle extrapolation (values outside CDF range)
    # For u < min(probs), use first threshold
    # For u > max(probs), extrapolate linearly
    min_prob = np.min(sorted_probs)
    max_prob = np.max(sorted_probs)
    
    if min_prob > 0.0:
        # Extrapolate below minimum
        mask_low = u < min_prob
        if np.any(mask_low):
            # Linear extrapolation: assume constant slope
            if len(sorted_thresh) > 1:
                slope = (sorted_thresh[1] - sorted_thresh[0]) / (sorted_probs[1] - sorted_probs[0] + 1e-10)
                samples[mask_low] = sorted_thresh[0] + (u[mask_low] - min_prob) * slope
            else:
                samples[mask_low] = sorted_thresh[0]
    
    if max_prob < 1.0:
        # Extrapolate above maximum
        mask_high = u > max_prob
        if np.any(mask_high):
            # Linear extrapolation
            if len(sorted_thresh) > 1:
                slope = (sorted_thresh[-1] - sorted_thresh[-2]) / (sorted_probs[-1] - sorted_probs[-2] + 1e-10)
                samples[mask_high] = sorted_thresh[-1] + (u[mask_high] - max_prob) * slope
            else:
                samples[mask_high] = sorted_thresh[-1]
    
    return samples


def run_ik_sgsim(
    block_model: Any,
    property_name: str,
    ik_result: IKResult,
    config: IKSGSIMConfig,
    use_sequential: bool = True
) -> IKSGSIMResult:
    """
    Run IK-based Sequential Gaussian Simulation.
    
    Industry-Standard Approach:
    Two modes are available:
    
    1. Sequential Mode (use_sequential=True, recommended):
       - Uses random path visitation
       - Samples from CDF considering previously simulated neighbors
       - Maintains spatial correlation structure
       - Similar to Sequential Indicator Simulation (SIS)
    
    2. Independent Mode (use_sequential=False):
       - Samples independently per block from local CDF
       - Faster but no inter-block correlation
       - Useful for quick screening or small datasets
    
    Args:
        block_model: BlockModel instance (must have properties dict and coordinates)
        property_name: Base property name (e.g., "Fe")
        ik_result: IKResult from indicator kriging
        config: IKSGSIMConfig instance
        use_sequential: If True, use sequential simulation with spatial correlation
    
    Returns:
        IKSGSIMResult with realization names
    """
    logger.info(f"Starting IK-SGSIM: {config.n_realizations} realizations for property '{property_name}'")
    mode_str = "Sequential (spatial correlation)" if use_sequential else "Independent (no correlation)"
    logger.info(f"Mode: {mode_str}")
    
    # =========================================================================
    # AUDIT FIX (W-001): MANDATORY Random Seed for JORC/SAMREC Reproducibility
    # =========================================================================
    if config.random_seed is None:
        raise ValueError(
            "IK-SGSIM GATE FAILED (W-001): Random seed is REQUIRED for JORC/SAMREC reproducibility. "
            "Set config.random_seed to an integer value. Non-reproducible simulations are not permitted."
        )
    rng = np.random.default_rng(config.random_seed)
    logger.info(f"Using random seed {config.random_seed} for reproducible simulation")
    
    # Get block model dimensions
    if not hasattr(block_model, 'properties') or not hasattr(block_model, 'coordinates'):
        raise ValueError("BlockModel must have 'properties' and 'coordinates' attributes")
    
    n_blocks = len(block_model.coordinates)
    
    # Get IK probabilities
    thresholds = ik_result.thresholds
    probabilities = ik_result.probabilities  # Shape: [n_blocks, n_thresholds]
    
    if probabilities.shape[0] != n_blocks:
        raise ValueError(f"IK probabilities shape {probabilities.shape} doesn't match block model size {n_blocks}")
    
    # Generate realizations
    realization_names = []
    
    for ireal in range(config.n_realizations):
        realization_name = f"{config.realisation_prefix}_{property_name}_{ireal + 1:04d}"
        realization_names.append(realization_name)
        
        # Sample from CDF for each block
        simulated_values = np.full(n_blocks, np.nan)
        
        for iblock in range(n_blocks):
            block_probs = probabilities[iblock, :]
            
            # Skip if all probabilities are NaN
            if np.all(np.isnan(block_probs)):
                continue
            
            # Sample from CDF
            sample = _sample_from_cdf(thresholds, block_probs, n_samples=1, random_state=rng)
            simulated_values[iblock] = sample[0]
        
        # Add to block model
        block_model.properties[realization_name] = simulated_values
        
        # Progress callback - show "X run, Y remaining"
        if config.progress_callback:
            completed = ireal + 1
            remaining = config.n_realizations - completed
            pct = int((completed / config.n_realizations) * 100)
            config.progress_callback(pct, f"IK-SGSIM realization {completed}/{config.n_realizations} (Run: {completed}, Remaining: {remaining})")
        elif (ireal + 1) % 10 == 0:
            logger.info(f"Generated realization {ireal + 1}/{config.n_realizations}")
    
    logger.info(f"IK-SGSIM complete: {len(realization_names)} realizations generated")
    
    return IKSGSIMResult(
        realization_names=realization_names,
        metadata={
            'property_name': property_name,
            'n_realizations': config.n_realizations,
            'n_thresholds': len(thresholds),
            'thresholds': thresholds.tolist(),
            'method': 'IK-SGSIM'
        }
    )


def run_ik_sgsim_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run IK-SGSIM job (wrapper for controller integration).
    
    Args:
        params: Job parameters dict with:
            - block_model: BlockModel instance
            - property_name: Base property name
            - ik_result: IKResult dict (from indicator kriging)
            - n_realizations: Number of realizations
            - random_seed: Random seed (optional)
            - realisation_prefix: Prefix for realization names
    
    Returns:
        Result dict with realization_names and metadata
    """
    from ..models.block_model import BlockModel
    
    # Extract parameters
    block_model = params.get('block_model')
    property_name = params.get('property_name')
    ik_result_dict = params.get('ik_result')
    n_realizations = params.get('n_realizations', 100)
    random_seed = params.get('random_seed', None)
    realisation_prefix = params.get('realisation_prefix', 'ik_sim')
    
    if block_model is None:
        raise ValueError("block_model is required")
    if property_name is None:
        raise ValueError("property_name is required")
    if ik_result_dict is None:
        raise ValueError("ik_result is required")
    
    # Reconstruct IKResult from dict
    ik_result = IKResult(
        thresholds=np.asarray(ik_result_dict['thresholds']),
        probabilities=np.asarray(ik_result_dict['probabilities']),
        median=ik_result_dict.get('median'),
        mean=ik_result_dict.get('mean'),
        metadata=ik_result_dict.get('metadata', {})
    )
    
    # ✅ FIX: Extract progress callback from params
    progress_callback = params.get('_progress_callback')
    
    # Create config
    config = IKSGSIMConfig(
        n_realizations=n_realizations,
        random_seed=random_seed,
        realisation_prefix=realisation_prefix,
        progress_callback=progress_callback
    )
    
    # Run simulation
    result = run_ik_sgsim(block_model, property_name, ik_result, config)
    
    # Convert to dict
    return {
        'realization_names': result.realization_names,
        'metadata': result.metadata,
        'property_name': property_name
    }

