"""
Economic Propagation Engine

Propagate grade realisations through:
- Block value calculation
- IRR / NPV evaluation
- Pit shell generation
- Scheduling
- Risk metrics
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .grade_realisations import GradeRealisationSet

logger = logging.getLogger(__name__)


@dataclass
class EconomicUncertaintyConfig:
    """Configuration for economic uncertainty propagation."""
    property_name: str
    realisations: GradeRealisationSet
    economic_params: Dict[str, Any]
    use_existing_schedule: bool = False
    reoptimise_per_realisation: bool = True
    n_realizations_to_sample: Optional[int] = None  # If None, use all
    pit_optimization_params: Dict[str, Any] = field(default_factory=dict)
    schedule_optimization_params: Dict[str, Any] = field(default_factory=dict)
    risk_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EconomicUncertaintyResult:
    """Result from economic uncertainty propagation."""
    npv_samples: np.ndarray
    irr_samples: np.ndarray
    pit_shell_ids: List[str]  # per realisation
    schedule_profiles: List[Any]  # ScheduleRiskProfile instances
    summary_stats: Dict[str, Any] = field(default_factory=dict)


def _calculate_block_values_for_realisation(
    block_model: Any,
    grade_values: np.ndarray,
    economic_params: Dict[str, Any]
) -> np.ndarray:
    """
    Calculate block values for a realisation.
    
    Args:
        block_model: BlockModel instance
        grade_values: Grade values for this realisation
        economic_params: Economic parameters
    
    Returns:
        Block values array
    """
    from ..irr_engine.npv_calc import calculate_block_value
    
    # Get tonnage (assume it's in block model)
    if 'TONNAGE' not in block_model.properties:
        raise ValueError("BlockModel must have 'TONNAGE' property")
    
    tonnage = np.asarray(block_model.properties['TONNAGE'])
    
    # Economic parameters
    metal_price = economic_params.get('metal_price', 60.0)
    mining_cost = economic_params.get('mining_cost', 2.5)
    processing_cost = economic_params.get('processing_cost', 8.0)
    recovery = economic_params.get('recovery', 0.85)
    
    # Calculate values
    n_blocks = len(tonnage)
    block_values = np.zeros(n_blocks)
    
    for i in range(n_blocks):
        if np.isnan(grade_values[i]) or np.isnan(tonnage[i]):
            block_values[i] = -tonnage[i] * mining_cost if not np.isnan(tonnage[i]) else 0.0
            continue
        
        value = calculate_block_value(
            tonnage=tonnage[i],
            grade=grade_values[i],
            metal_price=metal_price,
            mining_cost=mining_cost,
            processing_cost=processing_cost,
            recovery=recovery
        )
        block_values[i] = value
    
    return block_values


def _optimize_pit_for_realisation(
    block_model: Any,
    block_values: np.ndarray,
    params: Dict[str, Any]
) -> List[int]:
    """
    Optimize pit shell for a realisation.
    
    Args:
        block_model: BlockModel instance
        block_values: Block values array
        params: Pit optimization parameters
    
    Returns:
        List of block IDs in optimal pit
    """
    from ..irr_engine.lerchs_grossmann import LerchsGrossmann
    
    # Create temporary block model DataFrame
    coords = block_model.coordinates
    tonnage = block_model.properties.get('TONNAGE', np.ones(len(coords)))
    
    # Create DataFrame for LerchsGrossmann
    block_df = pd.DataFrame({
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'TONNAGE': tonnage,
        'GRADE': np.zeros(len(coords)),  # Not used, values are in block_values
        'VALUE': block_values
    })
    
    # Economic params
    economic_params = params.get('economic_params', {})
    
    # Create optimizer
    lg = LerchsGrossmann(
        block_df,
        economic_params,
        slope_angle=params.get('slope_angle', 45.0)
    )
    
    # Optimize
    optimal_pit = lg.optimize()
    
    return optimal_pit


def _optimize_schedule_for_realisation(
    block_model: Any,
    pit_blocks: List[int],
    block_values: np.ndarray,
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Optimize schedule for a realisation.
    
    Args:
        block_model: BlockModel instance
        pit_blocks: List of block IDs in pit
        block_values: Block values array
        params: Schedule optimization parameters
    
    Returns:
        Schedule DataFrame
    """
    from ..irr_engine.milp_optimizer import MiningScheduleOptimizer
    
    # Create DataFrame with pit blocks
    coords = block_model.coordinates
    tonnage = block_model.properties.get('TONNAGE', np.ones(len(coords)))
    
    # Filter to pit blocks
    pit_mask = np.zeros(len(coords), dtype=bool)
    pit_mask[pit_blocks] = True
    
    pit_df = pd.DataFrame({
        'BLOCK_ID': np.arange(len(coords))[pit_mask],
        'X': coords[pit_mask, 0],
        'Y': coords[pit_mask, 1],
        'Z': coords[pit_mask, 2],
        'TONNAGE': tonnage[pit_mask],
        'GRADE': np.zeros(np.sum(pit_mask)),  # Not used
        'VALUE': block_values[pit_mask]
    })
    
    # Create optimizer
    optimizer = MiningScheduleOptimizer(
        pit_df,
        economic_params=params.get('economic_params', {}),
        max_periods=params.get('max_periods', 20),
        period_tonnage_target=params.get('period_tonnage_target', 100000.0)
    )
    
    # Optimize schedule
    schedule = optimizer.optimize()
    
    return schedule


def _compute_risk_profile(
    schedule: pd.DataFrame,
    block_model: Any,
    params: Dict[str, Any]
) -> Any:
    """
    Compute risk profile for a schedule.
    
    Args:
        schedule: Schedule DataFrame
        block_model: BlockModel instance
        params: Risk parameters
    
    Returns:
        ScheduleRiskProfile instance
    """
    from ..risk.schedule_risk_linker import build_period_risk_profile
    
    # Build risk profile
    profile = build_period_risk_profile(
        schedule=schedule,
        hazard_volume=None,  # Would need to be passed in params
        rockburst_results=None,
        slope_results=None,
        params={
            'schedule_id': params.get('schedule_id', 'realisation_schedule'),
            'aggregation_method': params.get('aggregation_method', 'mean'),
            'risk_weights': params.get('risk_weights', {}),
            'period_days': params.get('period_days', 30.0)
        }
    )
    
    return profile


def run_economic_uncertainty(
    block_model: Any,
    config: EconomicUncertaintyConfig
) -> EconomicUncertaintyResult:
    """
    Run economic uncertainty propagation.
    
    For each realisation:
    1. Calculate block values
    2. Compute NPV/IRR (optionally optimize pit/schedule)
    3. Compute risk profiles
    
    Args:
        block_model: BlockModel instance
        config: EconomicUncertaintyConfig instance
    
    Returns:
        EconomicUncertaintyResult instance
    """
    logger.info(f"Starting economic uncertainty propagation for '{config.property_name}'")
    logger.info(f"Realizations: {config.realisations.n_realizations}")
    
    # Determine how many realizations to process
    n_to_process = config.n_realizations_to_sample
    if n_to_process is None:
        n_to_process = config.realisations.n_realizations
    else:
        n_to_process = min(n_to_process, config.realisations.n_realizations)
    
    # Sample indices if needed
    if n_to_process < config.realisations.n_realizations:
        indices = np.random.choice(
            config.realisations.n_realizations,
            size=n_to_process,
            replace=False
        )
        indices = sorted(indices)
    else:
        indices = list(range(config.realisations.n_realizations))
    
    # Storage for results
    npv_samples = []
    irr_samples = []
    pit_shell_ids = []
    schedule_profiles = []
    
    # Process each realisation
    for idx, realisation_idx in enumerate(indices):
        logger.info(f"Processing realisation {idx + 1}/{n_to_process} (index {realisation_idx})")
        
        # Get grade values
        from .grade_realisations import get_realisation_values
        grade_values = get_realisation_values(block_model, config.realisations, realisation_idx)
        
        # Calculate block values
        block_values = _calculate_block_values_for_realisation(
            block_model,
            grade_values,
            config.economic_params
        )
        
        # Add block values to block model temporarily
        temp_value_prop = f"_temp_value_{realisation_idx}"
        block_model.properties[temp_value_prop] = block_values
        
        # Optimize pit if requested
        pit_blocks = None
        if config.reoptimise_per_realisation:
            try:
                pit_blocks = _optimize_pit_for_realisation(
                    block_model,
                    block_values,
                    {
                        'economic_params': config.economic_params,
                        **config.pit_optimization_params
                    }
                )
                pit_shell_ids.append(f"pit_real_{realisation_idx}")
            except Exception as e:
                logger.warning(f"Pit optimization failed for realisation {realisation_idx}: {e}")
                pit_shell_ids.append(None)
        else:
            pit_shell_ids.append("existing_pit")
            # Use existing pit if available
            if 'PIT_ID' in block_model.properties:
                pit_mask = block_model.properties['PIT_ID'] > 0
                pit_blocks = np.where(pit_mask)[0].tolist()
            else:
                pit_blocks = list(range(len(block_values)))
        
        # Optimize schedule if requested
        schedule = None
        if config.reoptimise_per_realisation and pit_blocks:
            try:
                schedule = _optimize_schedule_for_realisation(
                    block_model,
                    pit_blocks,
                    block_values,
                    {
                        'economic_params': config.economic_params,
                        **config.schedule_optimization_params
                    }
                )
            except Exception as e:
                logger.warning(f"Schedule optimization failed for realisation {realisation_idx}: {e}")
        
        # Use existing schedule if available and not reoptimizing
        if schedule is None and config.use_existing_schedule:
            if 'PERIOD' in block_model.properties:
                # Create schedule from existing PERIOD property
                schedule = pd.DataFrame({
                    'BLOCK_ID': np.arange(len(block_model.coordinates)),
                    'PERIOD': block_model.properties['PERIOD'],
                    'MINED': np.ones(len(block_model.coordinates))
                })
        
        # Compute NPV/IRR
        npv = None
        irr = None
        
        if schedule is not None:
            try:
                from ..irr_engine.npv_calc import calculate_npv, calculate_irr
                
                # Create block model DataFrame
                block_df = pd.DataFrame({
                    'BLOCK_ID': np.arange(len(block_model.coordinates)),
                    'TONNAGE': block_model.properties.get('TONNAGE', np.ones(len(block_model.coordinates))),
                    'GRADE': grade_values,
                    'VALUE': block_values
                })
                
                # Calculate NPV
                discount_rate = config.economic_params.get('discount_rate', 0.10)
                npv_result = calculate_npv(
                    schedule,
                    block_df,
                    config.economic_params,
                    discount_rate
                )
                npv = npv_result.get('npv', 0.0)
                
                # Calculate IRR
                cashflows = npv_result.get('cashflows', np.array([0.0]))
                if len(cashflows) > 0 and np.sum(cashflows) > 0:
                    irr = calculate_irr(cashflows)
                else:
                    irr = 0.0
                
            except Exception as e:
                logger.warning(f"NPV/IRR calculation failed for realisation {realisation_idx}: {e}")
        
        npv_samples.append(npv if npv is not None else np.nan)
        irr_samples.append(irr if irr is not None else np.nan)
        
        # Compute risk profile
        if schedule is not None:
            try:
                profile = _compute_risk_profile(
                    schedule,
                    block_model,
                    {
                        'schedule_id': f"realisation_{realisation_idx}",
                        **config.risk_params
                    }
                )
                schedule_profiles.append(profile)
            except Exception as e:
                logger.warning(f"Risk profile computation failed for realisation {realisation_idx}: {e}")
                schedule_profiles.append(None)
        else:
            schedule_profiles.append(None)
        
        # Clean up temporary property
        if temp_value_prop in block_model.properties:
            del block_model.properties[temp_value_prop]
    
    # Convert to arrays
    npv_samples = np.asarray(npv_samples)
    irr_samples = np.asarray(irr_samples)
    
    # Compute summary statistics
    summary_stats = {
        'n_realizations': n_to_process,
        'npv': {
            'mean': float(np.nanmean(npv_samples)),
            'std': float(np.nanstd(npv_samples)),
            'p10': float(np.nanpercentile(npv_samples, 10)),
            'p50': float(np.nanpercentile(npv_samples, 50)),
            'p90': float(np.nanpercentile(npv_samples, 90)),
            'min': float(np.nanmin(npv_samples)),
            'max': float(np.nanmax(npv_samples))
        },
        'irr': {
            'mean': float(np.nanmean(irr_samples)),
            'std': float(np.nanstd(irr_samples)),
            'p10': float(np.nanpercentile(irr_samples, 10)),
            'p50': float(np.nanpercentile(irr_samples, 50)),
            'p90': float(np.nanpercentile(irr_samples, 90)),
            'min': float(np.nanmin(irr_samples)),
            'max': float(np.nanmax(irr_samples))
        }
    }
    
    logger.info(f"Economic uncertainty propagation complete")
    logger.info(f"NPV: mean={summary_stats['npv']['mean']:.2f}, std={summary_stats['npv']['std']:.2f}")
    logger.info(f"IRR: mean={summary_stats['irr']['mean']:.4f}, std={summary_stats['irr']['std']:.4f}")
    
    return EconomicUncertaintyResult(
        npv_samples=npv_samples,
        irr_samples=irr_samples,
        pit_shell_ids=pit_shell_ids,
        schedule_profiles=schedule_profiles,
        summary_stats=summary_stats
    )


def run_economic_uncertainty_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run economic uncertainty job (wrapper for controller integration).
    
    Args:
        params: Job parameters dict
    
    Returns:
        Result dict
    """
    from .grade_realisations import GradeRealisationSet
    
    # Extract parameters
    block_model = params.get('block_model')
    property_name = params.get('property_name')
    realisation_names = params.get('realisation_names', [])
    economic_params = params.get('economic_params', {})
    use_existing_schedule = params.get('use_existing_schedule', False)
    reoptimise_per_realisation = params.get('reoptimise_per_realisation', True)
    n_realizations_to_sample = params.get('n_realizations_to_sample', None)
    
    if block_model is None:
        raise ValueError("block_model is required")
    if property_name is None:
        raise ValueError("property_name is required")
    if not realisation_names:
        raise ValueError("realisation_names is required")
    
    # Create GradeRealisationSet
    realisations = GradeRealisationSet(
        property_name=property_name,
        realisation_names=realisation_names,
        n_realizations=len(realisation_names)
    )
    
    # Create config
    config = EconomicUncertaintyConfig(
        property_name=property_name,
        realisations=realisations,
        economic_params=economic_params,
        use_existing_schedule=use_existing_schedule,
        reoptimise_per_realisation=reoptimise_per_realisation,
        n_realizations_to_sample=n_realizations_to_sample,
        pit_optimization_params=params.get('pit_optimization_params', {}),
        schedule_optimization_params=params.get('schedule_optimization_params', {}),
        risk_params=params.get('risk_params', {})
    )
    
    # Run propagation
    result = run_economic_uncertainty(block_model, config)
    
    # Convert to dict
    return {
        'npv_samples': result.npv_samples.tolist(),
        'irr_samples': result.irr_samples.tolist(),
        'pit_shell_ids': result.pit_shell_ids,
        'schedule_profiles': [
            {
                'schedule_id': p.schedule_id if p else None,
                'summary_stats': p.summary_stats if p else {}
            }
            for p in result.schedule_profiles
        ],
        'summary_stats': result.summary_stats
    }

