"""
Unified API for IRR/NPV/Pit operations.

Provides thin wrapper functions that take typed configs and return typed results.

Updated 2025-12:
- Integrated validation module for audit compliance
- Added provenance tracking
- Added classification filter support
"""

import logging
import pandas as pd
from typing import Optional, Callable, Any, Dict, List

from .config_loader import IRRConfig, PitConfig, ScheduleConfig
from .scenario_generator import ScenarioGenerator, ScenarioConfig
from .irr_bisection import find_irr_alpha
from .npv_calc import calculate_npv
from .lerchs_grossmann import LerchsGrossmann
from .results_model import IRRResult, build_irr_result
from .validation import validate_economic_params, EconomicParameterError

logger = logging.getLogger(__name__)


def run_irr(
    config: IRRConfig, 
    progress_callback: Optional[Callable] = None,
    classification_filter: Optional[List[str]] = None,
    strict_classification: bool = True,  # MP-015 FIX: Default to True for JORC/SAMREC compliance
    store_all_cashflows: bool = False,
    validate_inputs: bool = True
) -> IRRResult:
    """
    Run IRR analysis with typed configuration.
    
    Args:
        config: IRR configuration
        progress_callback: Optional callback for progress updates
        classification_filter: List of classifications to include (default: Measured, Indicated)
        strict_classification: If True, raise error if classification column missing
        store_all_cashflows: If True, store cash flows for all scenarios (audit trail)
        validate_inputs: If True, validate all inputs (recommended)
        
    Returns:
        IRRResult with analysis results
        
    Raises:
        EconomicParameterError: If required economic parameters are missing
        ValueError: If strict_classification=True and classification column missing
        
    Note (2025-12):
        If config.nested_shells is provided, the analysis will use dynamic pit
        selection: for each price scenario, the optimal pit shell is selected
        based on the scenario's price level. This ensures rational pit boundaries
        that respond to price changes, rather than mining a fixed pit at all prices.
        
    Audit Compliance (2025-12):
        - Economic parameters are validated (no silent defaults)
        - Classification filtering is applied for JORC/SAMREC compliance
        - Provenance record is created for reproducibility
        - Multiple IRR warnings are tracked
    """
    # Convert BlockModel to DataFrame if needed
    if hasattr(config.block_model, 'to_dataframe'):
        block_model_df = config.block_model.to_dataframe()
    elif isinstance(config.block_model, pd.DataFrame):
        block_model_df = config.block_model.copy()
    else:
        raise ValueError(f"Unsupported block_model type: {type(config.block_model)}")
    
    # Validate economic parameters BEFORE proceeding (Violation #5 fix)
    if validate_inputs:
        try:
            config.economic_params = validate_economic_params(config.economic_params)
            logger.info("Economic parameters validated successfully")
        except EconomicParameterError as e:
            logger.error(f"Economic parameter validation failed: {e}")
            raise
    
    # Generate scenarios
    scenario_config = ScenarioConfig(**config.scenario_config)
    generator = ScenarioGenerator(scenario_config)
    scenarios = generator.generate_all_scenarios(block_model_df)
    
    # Log if dynamic pit selection is enabled
    if config.nested_shells is not None:
        logger.info("Dynamic pit selection ENABLED: nested shells provided")
    else:
        logger.info("Dynamic pit selection DISABLED: using fixed pit for all scenarios")
    
    # Run IRR bisection with all audit compliance features
    raw_result = find_irr_alpha(
        block_model=block_model_df,
        scenarios=scenarios,
        economic_params=config.economic_params,
        alpha=config.irr_search.get('alpha', 0.80),
        r_low=config.irr_search.get('r_low', 0.0),
        r_high=config.irr_search.get('r_high', 0.50),
        tolerance=config.tolerance,
        max_iterations=config.max_iterations,
        num_periods=config.num_periods,
        production_capacity=config.production_capacity,
        parallel=config.parallel,
        progress_callback=progress_callback,
        nested_shells=config.nested_shells,
        # Audit compliance parameters
        classification_filter=classification_filter,
        strict_classification=strict_classification,
        store_all_cashflows=store_all_cashflows,
        validate_inputs=False  # Already validated above
    )
    
    # Build normalized result with comprehensive metadata
    metadata = {
        'config': config,
        'dynamic_pit_enabled': config.nested_shells is not None,
        'classification_filter': classification_filter,
        'validation_performed': validate_inputs
    }
    
    if config.nested_shells is not None:
        metadata['shell_selection_stats'] = raw_result.get('shell_selection_stats')
    
    # Include provenance in metadata
    if 'provenance' in raw_result:
        metadata['provenance'] = raw_result['provenance']
    
    # Include audit compliance info
    metadata['audit_compliance'] = {
        'multiple_irr_warnings': raw_result.get('multiple_irr_warnings', 0),
        'blocks_before_filter': raw_result.get('blocks_before_filter', 0),
        'blocks_after_filter': raw_result.get('blocks_after_filter', 0),
        'classification_filter_applied': raw_result.get('classification_filter_applied', False)
    }
    
    result = build_irr_result(raw_result, metadata=metadata)
    return result


def run_npv(config: ScheduleConfig) -> Dict[str, Any]:
    """
    Run NPV calculation with typed configuration.
    
    Args:
        config: Schedule configuration
        
    Returns:
        Dictionary with NPV results
    """
    # Convert BlockModel to DataFrame if needed
    if hasattr(config.block_model, 'to_dataframe'):
        block_model_df = config.block_model.to_dataframe()
    elif isinstance(config.block_model, pd.DataFrame):
        block_model_df = config.block_model.copy()
    else:
        raise ValueError(f"Unsupported block_model type: {type(config.block_model)}")
    
    # For NPV, we need a schedule - this is a simplified version
    # In practice, you'd optimize the schedule first
    # This is a placeholder that calculates NPV for a simple greedy schedule
    from .milp_optimizer import MiningScheduleOptimizer
    
    optimizer = MiningScheduleOptimizer(
        block_model=block_model_df,
        economic_params=config.economic_params,
        num_periods=config.num_periods,
        production_capacity=config.production_capacity
    )
    
    schedule = optimizer.optimize()
    
    # Calculate NPV
    result = calculate_npv(
        schedule=schedule,
        block_model=block_model_df,
        economic_params=config.economic_params,
        discount_rate=config.discount_rate
    )
    
    return result


def run_pit_optimisation(config: PitConfig) -> Dict[str, Any]:
    """
    Run pit optimization with typed configuration.
    
    Args:
        config: Pit configuration
        
    Returns:
        Dictionary with pit optimization results
    """
    # Convert BlockModel to DataFrame if needed
    if hasattr(config.block_model, 'to_dataframe'):
        block_model_df = config.block_model.to_dataframe()
    elif isinstance(config.block_model, pd.DataFrame):
        block_model_df = config.block_model.copy()
    else:
        raise ValueError(f"Unsupported block_model type: {type(config.block_model)}")
    
    # Run Lerchs-Grossmann
    lg = LerchsGrossmann(
        block_model=block_model_df,
        slope_angles=config.slope_angles,
        economic_params=config.economic_params
    )
    
    if config.nested_shells and config.shell_factors:
        # Generate nested shells
        shells = []
        for factor in config.shell_factors:
            # Adjust economic parameters by factor
            adjusted_params = config.economic_params.copy()
            adjusted_params['metal_price'] = adjusted_params.get('metal_price', 60.0) * factor
            
            lg_adjusted = LerchsGrossmann(
                block_model=block_model_df,
                slope_angles=config.slope_angles,
                economic_params=adjusted_params
            )
            pit_blocks = lg_adjusted.optimize()
            shells.append({
                'factor': factor,
                'blocks': pit_blocks,
                'tonnage': block_model_df.loc[pit_blocks, 'TONNAGE'].sum() if 'TONNAGE' in block_model_df.columns else 0
            })
        
        return {
            'operation': 'nested_shells',
            'shells': shells,
            'block_model': block_model_df
        }
    else:
        # Single pit
        pit_blocks = lg.optimize()
        
        return {
            'operation': 'single_pit',
            'selected_blocks': pit_blocks,
            'block_model': block_model_df,
            'values': block_model_df.loc[pit_blocks, 'VALUE'].values if 'VALUE' in block_model_df.columns else None
        }


def run_schedule_optimisation(config: ScheduleConfig) -> Dict[str, Any]:
    """
    Run production schedule optimization with typed configuration.
    
    Args:
        config: Schedule configuration
        
    Returns:
        Dictionary with schedule results
    """
    # Convert BlockModel to DataFrame if needed
    if hasattr(config.block_model, 'to_dataframe'):
        block_model_df = config.block_model.to_dataframe()
    elif isinstance(config.block_model, pd.DataFrame):
        block_model_df = config.block_model.copy()
    else:
        raise ValueError(f"Unsupported block_model type: {type(config.block_model)}")
    
    from .milp_optimizer import MiningScheduleOptimizer
    
    optimizer = MiningScheduleOptimizer(
        block_model=block_model_df,
        economic_params=config.economic_params,
        num_periods=config.num_periods,
        production_capacity=config.production_capacity
    )
    
    schedule = optimizer.optimize()
    
    # Calculate NPV for the optimized schedule
    npv_result = calculate_npv(
        schedule=schedule,
        block_model=block_model_df,
        economic_params=config.economic_params,
        discount_rate=config.discount_rate
    )
    
    return {
        'schedule': schedule,
        'npv': npv_result['npv'],
        'cashflows': npv_result['cashflows'],
        'revenue': npv_result.get('revenue', 0),
        'operating_cost': npv_result.get('operating_cost', 0),
        'capital_cost': npv_result.get('capital_cost', 0)
    }

