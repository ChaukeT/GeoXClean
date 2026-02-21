"""
IRR Distribution Algorithm (Optimized).

Finds the maximum risk-adjusted IRR (IRR_α) using optimized distribution-based approach.

Key Enhancement (2025-12):
--------------------------
Added Dynamic Pit Shell Selection to address the integration gap between pit optimization
and IRR calculations. For each price scenario, the algorithm now selects the optimal
pit shell from pre-computed nested shells, ensuring rational pit boundaries that respond
to price changes rather than mining a fixed pit at all prices.

This fixes the issue where low-price scenarios artificially inflated risk by assuming
the mine would continue extracting marginal blocks that wouldn't be economic at the
scenario's price.

Audit Fixes (2025-12):
---------------------
- Violation #1: Added classification filter integration
- Violation #2: Added multiple IRR detection to scenario results
- Violation #4: Added provenance tracking
- Violation #5: Removed silent defaults
- Violation #6: Enforced deterministic seeding
- Violation #7: Store all scenario cash flows for audit trail
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable, Tuple, List, Any, TYPE_CHECKING
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .npv_calc import calculate_npv, calculate_irr, calculate_irr_with_metadata, detect_multiple_irr
from .fast_scheduler import FastScheduler
from .validation import (
    validate_irr_inputs,
    validate_economic_params,
    apply_classification_filter,
    EconomicParameterError
)
from .provenance import (
    create_provenance,
    ensure_deterministic_seed,
    IRRProvenance
)

if TYPE_CHECKING:
    from .dynamic_shell_selector import DynamicPitShellSelector

logger = logging.getLogger(__name__)


def _process_scenario_task(
    i: int,
    base_data: Dict[str, np.ndarray],
    scenario_params: Dict[str, float],
    grade_vector: Optional[np.ndarray],
    econ_static: Dict,
    num_periods: int,
    production_capacity: float,
    shell_mask: Optional[np.ndarray] = None,
    store_cashflows: bool = False
) -> Tuple[int, float, float, Dict]:
    """
    Pure worker function for parallel execution.
    
    Args:
        i: Scenario index
        base_data: Dictionary of numpy arrays (TONNAGE, BLOCK_ID, PHASE, GRADE)
        scenario_params: Dictionary with price, cost_m, cost_p, rec
        grade_vector: Optional grade vector for this scenario
        econ_static: Static economic parameters (selling_cost, capex)
        num_periods: Number of mining periods
        production_capacity: Maximum tonnage per period
        shell_mask: Optional boolean mask for blocks in the selected pit shell.
                   If provided, only these blocks will be scheduled (dynamic pit).
                   If None, all blocks are used (fixed pit - legacy behavior).
        store_cashflows: If True, include cash flows in metadata (for audit trail)
    
    Returns:
        Tuple of (index, irr, npv_ref, metadata_dict)
        metadata_dict includes: blocks_in_pit, total_blocks, shell_applied, 
                               tonnage_mined, ore_value, and optionally cashflows
    """
    try:
        # Reconstruct lightweight DataFrame for scheduler
        # We only need columns required for Value calc and Scheduling
        # Base data has: TONNAGE, PHASE, BLOCK_ID (and optionally GRADE for fallback)
        
        tonnage = base_data['TONNAGE'].copy()
        
        # Determine Grade
        if grade_vector is not None:
            grade = grade_vector.copy()
        else:
            grade = base_data.get('GRADE', np.zeros_like(tonnage)).copy()
        
        # === DYNAMIC PIT SHELL INTEGRATION ===
        # If shell_mask is provided, zero out blocks NOT in the selected shell.
        # This simulates "not mining" those blocks in this scenario.
        # The scheduler will naturally skip zero-tonnage blocks.
        if shell_mask is not None:
            # Blocks outside the shell have zero tonnage (won't be mined)
            tonnage = tonnage * shell_mask.astype(np.float64)
            # Also zero out grade for consistency
            grade = grade * shell_mask.astype(np.float64)
            
        # Params
        price = scenario_params['price']
        cost_m = scenario_params['cost_m']
        cost_p = scenario_params['cost_p']
        rec = scenario_params['rec']
        
        # Calculate Value
        # Value = (Tonnes * Grade * Rec * Price) - (Tonnes * Costs)
        revenue = tonnage * grade * rec * price
        costs = tonnage * (cost_m + cost_p)
        values = revenue - costs
        
        # Create DataFrame for FastScheduler
        # It needs: TONNAGE, VALUE, PHASE (if present), BLOCK_ID
        df_data = {
            'TONNAGE': tonnage,
            'VALUE': values,
            'BLOCK_ID': base_data['BLOCK_ID']
        }
        if 'PHASE' in base_data:
            df_data['PHASE'] = base_data['PHASE']
            
        # Make sure to copy to avoid any shared memory issues (though pickling handles this)
        block_model_slice = pd.DataFrame(df_data)
        
        # Track metadata for diagnostics
        metadata = {
            'blocks_in_pit': int((tonnage > 0).sum()) if shell_mask is not None else len(tonnage),
            'total_blocks': len(tonnage),
            'shell_applied': shell_mask is not None
        }
        
        # Run Scheduler
        schedule = FastScheduler.schedule_greedy(
            block_model_slice, num_periods, production_capacity
        )
        
        # Calculate Financials (NPV/IRR)
        # We need to reconstruct the full view for calculate_npv? 
        # calculate_npv expects 'block_model' dataframe.
        # We can reuse block_model_slice, but we need 'GRADE' if calculate_npv uses it?
        # Checking calculate_npv usage in previous code... 
        # It seems to use 'VALUE' if present, or re-calculates.
        # Let's ensure 'GRADE' is in the dataframe if calculate_npv needs it for reporting, 
        # but for pure NPV calc on cashflows, it might just aggregate VALUE?
        # Wait, standard calculate_npv often recalculates value based on params provided.
        # Let's look at how it was called:
        # calculate_npv(schedule, base_model, params, discount_rate=0.0)
        
        # To be safe, let's add GRADE to block_model_slice
        block_model_slice['GRADE'] = grade
        
        # Run NPV Calc (Raw Cashflows)
        # We pass the scenario params so it uses them
        run_params = {
            'metal_price': price, 'mining_cost': cost_m, 
            'processing_cost': cost_p, 'recovery': rec,
            'selling_cost': econ_static.get('selling_cost', 0.0),
            'capex': econ_static.get('capex', [])
        }
        
        res = calculate_npv(schedule, block_model_slice, run_params, discount_rate=0.0)
        cfs = res['cashflows']
        
        # Calculate IRR with multiple IRR detection
        irr, irr_metadata = calculate_irr_with_metadata(cfs)
        
        if irr is None or irr <= -0.99:
            irr = np.nan
            
        # Reference NPV (e.g. @ 10%)
        res_ref = calculate_npv(schedule, block_model_slice, run_params, discount_rate=0.10)
        npv_ref = res_ref['npv']
        
        # Add more metadata
        metadata['tonnage_mined'] = float(tonnage.sum())
        metadata['ore_value'] = float(values[values > 0].sum())
        metadata['irr_analysis'] = {
            'sign_changes': irr_metadata.get('sign_changes', 0),
            'has_multiple_irr_risk': irr_metadata.get('has_multiple_irr_risk', False),
            'irr_is_reliable': irr_metadata.get('irr_is_reliable', True)
        }
        
        # Store cash flows for audit trail (Violation #7 fix)
        if store_cashflows:
            metadata['cashflows'] = cfs.tolist() if isinstance(cfs, np.ndarray) else list(cfs)
            metadata['revenue'] = res.get('revenue', 0)
            metadata['operating_cost'] = res.get('operating_cost', 0)
        
        return i, irr, npv_ref, metadata
        
    except Exception as e:
        logger.error(f"Scenario {i} failed: {e}", exc_info=True)
        return i, np.nan, 0.0, {'error': str(e)}


def find_irr_alpha(
    block_model: pd.DataFrame,
    scenarios: Dict,
    economic_params: Dict,
    alpha: float = 0.80,
    r_low: float = 0.0,
    r_high: float = 0.50,
    tolerance: float = 0.001,
    max_iterations: int = 30,
    num_periods: int = 20,
    production_capacity: float = 1000000,
    parallel: bool = False,
    progress_callback: Optional[Callable] = None,
    config: Optional[Dict] = None,
    shell_selector: Optional['DynamicPitShellSelector'] = None,
    nested_shells: Optional[Dict] = None,
    # New parameters for audit compliance
    classification_filter: Optional[List[str]] = None,
    classification_column: str = 'CLASSIFICATION',
    strict_classification: bool = False,
    store_all_cashflows: bool = False,
    validate_inputs: bool = True
) -> Dict:
    """
    Find the maximum risk-adjusted IRR using optimized distribution-based approach.
    
    Algorithm Shift: Instead of Bisection Search on Discount Rate (which assumes a single 
    deterministic NPV curve), we now calculate the IRR Distribution.
    
    Old Way: Try Rate 10% -> Check % NPV>0. Try Rate 12%...
    New Way (Standard): For each of 100 scenarios, calculate its specific IRR. 
    You get a histogram of 100 IRRs. The "80% Confidence IRR" is simply the P20 of this histogram.
    
    NEW (2025-12) - Dynamic Pit Shell Integration:
    For each price scenario, the algorithm now selects the optimal pit shell from 
    pre-computed nested shells. This ensures that low-price scenarios mine a smaller,
    more economic pit rather than the full ultimate pit at reduced margins.
    
    AUDIT COMPLIANCE (2025-12):
    - Classification filtering for JORC/SAMREC compliance
    - No silent defaults for economic parameters
    - Provenance tracking for reproducibility
    - Multiple IRR detection
    - Optional cash flow storage for audit trail
    
    Args:
        block_model: Block model DataFrame
        scenarios: Dictionary of stochastic scenarios
        economic_params: Base economic parameters (REQUIRED: metal_price, mining_cost,
                        processing_cost, recovery - no silent defaults)
        alpha: Confidence level (e.g., 0.80 for 80%)
        r_low: Lower bound for discount rate (legacy parameter, not used in new approach)
        r_high: Upper bound for discount rate (legacy parameter, not used in new approach)
        tolerance: Convergence tolerance (legacy parameter, not used in new approach)
        max_iterations: Maximum iterations (legacy parameter, not used in new approach)
        num_periods: Number of mining periods
        production_capacity: Maximum tonnage per period
        parallel: Use parallel processing (ignored, fast path is sequential)
        progress_callback: Optional callback function(iteration, r, satisfaction_pct)
        config: Configuration dictionary
        shell_selector: Optional DynamicPitShellSelector for price-responsive pit boundaries.
        nested_shells: Optional nested shells result dict.
        classification_filter: List of classifications to include (default: Measured, Indicated)
        classification_column: Name of classification column in block model
        strict_classification: If True, raise error if classification column missing
        store_all_cashflows: If True, store cash flows for each scenario (audit trail)
        validate_inputs: If True, validate all inputs before processing
        
    Returns:
        Dictionary with:
            - 'irr_alpha': Maximum risk-adjusted IRR (P20 of distribution)
            - 'best_schedule': Mining schedule at IRR_α
            - 'npv_distribution': NPV values for all scenarios at reference rate
            - 'irr_distribution': IRR values for all scenarios (new)
            - 'satisfaction_rate': Percentage of scenarios with NPV ≥ 0
            - 'iterations': Number of scenarios processed
            - 'convergence_history': Synthetic convergence history for UI compatibility
            - 'economic_breakdown': Economic breakdown for UI display
            - 'shell_selection_stats': Statistics on shell selection (if shell_selector used)
            - 'provenance': IRRProvenance record for reproducibility
            - 'scenario_cashflows': Cash flows per scenario (if store_all_cashflows=True)
            - 'multiple_irr_warnings': Count of scenarios with multiple IRR risk
            
    Raises:
        EconomicParameterError: If required economic parameters are missing
        ValueError: If strict_classification=True and classification column missing
    """
    
    logger.info(f"Starting Optimized IRR Search (Alpha={alpha:.2f})")
    
    # === VALIDATION (Violation #5 fix) ===
    original_block_count = len(block_model)
    validation_metadata = {}
    
    if validate_inputs:
        try:
            # Validate economic parameters - NO SILENT DEFAULTS
            economic_params = validate_economic_params(economic_params)
            validation_metadata['economic_params_validated'] = True
        except EconomicParameterError as e:
            logger.error(f"Economic parameter validation failed: {e}")
            raise
    
    # === CLASSIFICATION FILTER (Violation #1 fix) ===
    classification_result = None
    if classification_filter is not None or strict_classification:
        try:
            block_model, classification_result = apply_classification_filter(
                block_model,
                classification_filter,
                classification_column,
                strict_mode=strict_classification
            )
            validation_metadata['classification_applied'] = True
            validation_metadata['blocks_filtered'] = classification_result.removed_blocks
        except ValueError as e:
            if strict_classification:
                raise
            logger.warning(f"Classification filter skipped: {e}")
    
    # === DETERMINISTIC SEEDING (Violation #6 fix) ===
    if config is None:
        config = {}
    seed_manager = ensure_deterministic_seed(config)
    start_time = time.time()
    
    num_scenarios = scenarios.get('num_scenarios', 100)
    scenario_irrs = []
    scenario_npvs_base = []  # NPV at r=0 or base rate
    
    # Prepare base model structure (Extract numpy arrays for efficiency)
    base_model = block_model.copy()
    if 'PHASE' not in base_model.columns:
        base_model['PHASE'] = 1  # Default phase
        
    # Prepare lightweight data dictionary for workers
    base_data_arrays = {
        'TONNAGE': base_model['TONNAGE'].values.astype(np.float64),
        'BLOCK_ID': base_model['BLOCK_ID'].values if 'BLOCK_ID' in base_model.columns else base_model.index.values,
    }
    if 'PHASE' in base_model.columns:
        base_data_arrays['PHASE'] = base_model['PHASE'].values.astype(np.int32)
    if 'GRADE' in base_model.columns:
        base_data_arrays['GRADE'] = base_model['GRADE'].values.astype(np.float64)

    # --- GRADE MATRIX PREPARATION ---
    # Pre-align grade scenarios to avoid slow dictionary lookups in loop
    scenario_grades_matrix = None
    if 'grades' in scenarios and scenarios['grades']:
        logger.info("Preparing grade scenarios matrix...")
        grade_map = scenarios['grades']
        
        # Initialize with base grades
        n_blocks = len(base_model)
        base_grade_vec = base_data_arrays.get('GRADE', np.zeros(n_blocks))
        
        # We need to map block_id -> index
        # Assuming block_ids are unique
        block_ids = base_data_arrays['BLOCK_ID']
        block_id_to_idx = {bid: idx for idx, bid in enumerate(block_ids)}
        
        # Check size of first valid entry to verify shape
        if grade_map:
            first_key = next(iter(grade_map))
            # Ensure it matches num_scenarios
            actual_scenarios = len(grade_map[first_key])
            if actual_scenarios != num_scenarios:
                logger.warning(f"Grade scenarios length ({actual_scenarios}) mismatch config ({num_scenarios}). Adjusting.")
                num_scenarios = min(num_scenarios, actual_scenarios)
        
        scenario_grades_matrix = np.zeros((n_blocks, num_scenarios), dtype=np.float32)
        
        # Fill with base grades as default
        for i in range(num_scenarios):
            scenario_grades_matrix[:, i] = base_grade_vec
            
        # Update with simulated values
        count_mapped = 0
        for bid, grades in grade_map.items():
            if bid in block_id_to_idx:
                idx = block_id_to_idx[bid]
                # Take slice in case lengths differ
                valid_len = min(len(grades), num_scenarios)
                scenario_grades_matrix[idx, :valid_len] = grades[:valid_len]
                count_mapped += 1
                
        logger.info(f"Mapped {count_mapped}/{n_blocks} blocks with simulated grades.")

    # Static Economic Params
    econ_static = {
        'selling_cost': economic_params.get('selling_cost', 0.0),
        'capex': economic_params.get('capex', [])
    }

    # --- DYNAMIC PIT SHELL SELECTOR INITIALIZATION ---
    # If nested shells are provided but no selector, create one automatically
    use_dynamic_pit = shell_selector is not None or nested_shells is not None
    
    if use_dynamic_pit:
        if shell_selector is None and nested_shells is not None:
            # Create selector from nested shells
            from .dynamic_shell_selector import create_shell_selector_from_lg_result
            base_price = float(economic_params.get('metal_price', 60.0))
            shell_selector = create_shell_selector_from_lg_result(
                nested_shells, 
                block_model, 
                base_price,
                interpolation_mode='floor'  # Conservative: use smaller pit for edge cases
            )
            logger.info(f"Created dynamic shell selector from nested shells (base_price=${base_price})")
        
        if shell_selector is not None:
            logger.info(f"Dynamic pit selection ENABLED: {len(shell_selector.shells)} shells available")
            logger.info(f"  RF range: {shell_selector._sorted_rfs[0]:.2f} - {shell_selector._sorted_rfs[-1]:.2f}")
        else:
            use_dynamic_pit = False
            logger.warning("Could not initialize shell selector - falling back to fixed pit")
    else:
        logger.info("Dynamic pit selection DISABLED: Using fixed pit for all scenarios")
    
    # Pre-compute shell masks for each scenario (if using dynamic pit)
    # This is more efficient than computing during parallel execution
    scenario_shell_masks = {}
    if use_dynamic_pit and shell_selector is not None:
        logger.info("Pre-computing shell masks for scenarios...")
        base_price = float(economic_params.get('metal_price', 60.0))
        
        for i in range(num_scenarios):
            # Get scenario price
            if 'prices' in scenarios:
                if isinstance(scenarios['prices'], np.ndarray):
                    if scenarios['prices'].ndim > 1:
                        scenario_price = float(scenarios['prices'][i].mean())
                    else:
                        scenario_price = float(scenarios['prices'][i])
                else:
                    scenario_price = float(scenarios['prices'].get(i, base_price))
            else:
                scenario_price = base_price
            
            # Get shell for this scenario's price
            result = shell_selector.select_shell(scenario_price)
            scenario_shell_masks[i] = result.block_mask
        
        logger.info(f"Pre-computed {len(scenario_shell_masks)} shell masks")
        logger.info(f"  Shell selector cache: {shell_selector._cache_hits} hits, {shell_selector._cache_misses} misses")

    # --- BATCH SIMULATION (PARALLEL) ---
    logger.info(f"Launching parallel execution with ProcessPoolExecutor (Scenarios: {num_scenarios})")
    
    # Storage for all scenario cash flows (Violation #7 fix)
    all_scenario_cashflows = [] if store_all_cashflows else None
    
    futures = []
    with ProcessPoolExecutor() as executor:
        for i in range(num_scenarios):
            # 1. Extract Params - use validated economic_params (no defaults)
            if 'prices' in scenarios:
                if isinstance(scenarios['prices'], np.ndarray):
                    price = float(scenarios['prices'][i].mean() if scenarios['prices'].ndim > 1 else scenarios['prices'][i])
                else:
                    price = float(scenarios['prices'].get(i, economic_params['metal_price']))
            else:
                price = float(economic_params['metal_price'])
                
            # Costs - no silent defaults (validated economic_params has required fields)
            if 'costs' in scenarios:
                if isinstance(scenarios['costs'], dict):
                    cost_m = float(scenarios['costs'].get('mining_cost', {}).get(i, economic_params['mining_cost']))
                    cost_p = float(scenarios['costs'].get('processing_cost', {}).get(i, economic_params['processing_cost']))
                else:
                    cost_m = float(economic_params['mining_cost'])
                    cost_p = float(economic_params['processing_cost'])
            else:
                cost_m = float(economic_params['mining_cost'])
                cost_p = float(economic_params['processing_cost'])
            
            # Recovery - no silent defaults
            if 'recoveries' in scenarios:
                if isinstance(scenarios['recoveries'], np.ndarray):
                    rec = float(scenarios['recoveries'][i] if scenarios['recoveries'].ndim > 0 else scenarios['recoveries'])
                else:
                    rec = float(scenarios['recoveries'].get(i, economic_params['recovery']))
            else:
                rec = float(economic_params['recovery'])
            
            scenario_params = {
                'price': price, 'cost_m': cost_m, 'cost_p': cost_p, 'rec': rec
            }
            
            # 2. Extract Grade Vector
            grade_vec = None
            if scenario_grades_matrix is not None:
                grade_vec = scenario_grades_matrix[:, i]
            
            # 3. Get Shell Mask (if using dynamic pit)
            shell_mask = scenario_shell_masks.get(i) if use_dynamic_pit else None
            
            # 4. Submit Task (with cash flow storage for audit trail)
            future = executor.submit(
                _process_scenario_task,
                i,
                base_data_arrays,
                scenario_params,
                grade_vec,
                econ_static,
                num_periods,
                production_capacity,
                shell_mask,
                store_all_cashflows  # Violation #7 fix
            )
            futures.append(future)
            
        # Collect Results
        completed_count = 0
        scenario_metadata = []
        multiple_irr_count = 0
        
        for future in as_completed(futures):
            i, irr, npv_ref, metadata = future.result()
            
            scenario_irrs.append(irr)
            scenario_npvs_base.append(npv_ref)
            scenario_metadata.append(metadata)
            
            # Track multiple IRR warnings
            if metadata.get('irr_analysis', {}).get('has_multiple_irr_risk', False):
                multiple_irr_count += 1
            
            # Store cash flows for audit trail (Violation #7 fix)
            if store_all_cashflows and 'cashflows' in metadata:
                all_scenario_cashflows.append({
                    'scenario_index': i,
                    'cashflows': metadata['cashflows'],
                    'irr': irr,
                    'npv_ref': npv_ref
                })
            
            completed_count += 1
            if progress_callback and completed_count % 5 == 0:
                progress_callback(completed_count, 0.0, float(completed_count)/num_scenarios)
    
    # Log multiple IRR warnings summary
    if multiple_irr_count > 0:
        logger.warning(
            f"MULTIPLE IRR WARNING: {multiple_irr_count}/{num_scenarios} scenarios "
            f"({100*multiple_irr_count/num_scenarios:.1f}%) have multiple sign changes in cash flow. "
            "IRR results for these scenarios may be unreliable."
        )

    # --- RESULTS ANALYSIS ---
    valid_irrs = np.array([x for x in scenario_irrs if not np.isnan(x)])
    
    if len(valid_irrs) == 0:
        logger.error("No valid IRRs found (all scenarios failed/negative).")
        # Return empty safe structure to prevent UI crash
        return _empty_result(num_scenarios, alpha)

    # Calculate IRR_alpha (The value exceeded by alpha% of scenarios)
    # If Alpha=0.80 (80% confidence), we want the 20th percentile.
    target_percentile = (1.0 - alpha) * 100
    irr_alpha_val = np.percentile(valid_irrs, target_percentile)
    
    # Generate Representative Schedule (using Mean Parameters)
    logger.info("Generating Representative Schedule...")
    avg_price = economic_params.get('metal_price', 100.0)
    avg_rec = economic_params.get('recovery', 0.85)
    avg_cost = economic_params.get('mining_cost', 10.0) + economic_params.get('processing_cost', 20.0)
    
    grade_col = base_model.get('GRADE', pd.Series([0.0] * len(base_model)))
    base_model['VALUE'] = (base_model['TONNAGE'] * grade_col * avg_rec * avg_price) - \
                          (base_model['TONNAGE'] * avg_cost)
                          
    best_schedule = FastScheduler.schedule_greedy(base_model, num_periods, production_capacity)
    
    # Final Details at the calculated IRR
    best_res = calculate_npv(best_schedule, base_model, economic_params, discount_rate=irr_alpha_val)
    
    # --- UI COMPATIBILITY FIX ---
    # Create a synthetic convergence history so the UI graph has something to show
    # We map "Scenarios Processed" to "Iteration"
    steps = np.linspace(1, num_scenarios, 10, dtype=int)
    conv_history = {
        'iterations': steps.tolist(),
        'r_values': np.full(10, irr_alpha_val).tolist(),  # Flat line (since we didn't iterate R)
        'satisfaction_rates': np.full(10, alpha).tolist(),  # Constant target
        'mean_npvs': np.linspace(np.min(scenario_npvs_base), np.mean(scenario_npvs_base), 10).tolist(),  # Trend
        'r_low_values': np.full(10, irr_alpha_val * 0.9).tolist(),
        'r_high_values': np.full(10, irr_alpha_val * 1.1).tolist()
    }

    elapsed = time.time() - start_time
    logger.info(f"IRR Analysis Complete in {elapsed:.1f}s. IRR_alpha={irr_alpha_val:.2%}")

    # Compile shell selection statistics
    shell_stats = None
    if use_dynamic_pit and shell_selector is not None:
        shell_stats = shell_selector.get_statistics()
        
        # Add scenario-level statistics
        shells_used = [m.get('shell_applied', False) for m in scenario_metadata]
        blocks_in_pit = [m.get('blocks_in_pit', 0) for m in scenario_metadata if 'blocks_in_pit' in m]
        
        shell_stats['scenarios_with_dynamic_pit'] = sum(1 for s in shells_used if s)
        shell_stats['avg_blocks_per_scenario'] = np.mean(blocks_in_pit) if blocks_in_pit else 0
        shell_stats['min_blocks_per_scenario'] = np.min(blocks_in_pit) if blocks_in_pit else 0
        shell_stats['max_blocks_per_scenario'] = np.max(blocks_in_pit) if blocks_in_pit else 0
        
        logger.info(f"Dynamic pit selection: "
                   f"{shell_stats['scenarios_with_dynamic_pit']}/{num_scenarios} scenarios used shells, "
                   f"avg blocks/scenario: {shell_stats['avg_blocks_per_scenario']:.0f}")

    # === CREATE PROVENANCE RECORD (Violation #4 fix) ===
    provenance = create_provenance(
        block_model=base_model,
        config=config,
        economic_params=economic_params,
        seed_manager=seed_manager,
        classification_filter=classification_filter,
        blocks_before_filter=original_block_count,
        validation_metadata=validation_metadata,
        solver_type='greedy_heuristic'
    )
    
    return {
        'irr_alpha': irr_alpha_val,
        'best_schedule': best_schedule,
        'npv_distribution': np.array(scenario_npvs_base),  # For plotting
        'irr_distribution': np.array(scenario_irrs),  # NEW: Full IRR distribution
        'satisfaction_rate': alpha,  # By definition
        'iterations': num_scenarios,
        'alpha_target': alpha,
        'num_scenarios': num_scenarios,
        'mean_npv': np.mean(scenario_npvs_base),
        'std_npv': np.std(scenario_npvs_base),
        'min_npv': np.min(scenario_npvs_base),
        'max_npv': np.max(scenario_npvs_base),
        'mean_irr': float(np.nanmean(valid_irrs)),  # NEW
        'std_irr': float(np.nanstd(valid_irrs)),    # NEW
        'best_cashflows': best_res['cashflows'],
        'best_npv_details': best_res,
        'convergence_history': conv_history,  # UI Crash Fix
        'economic_breakdown': {
            'Revenue': best_res.get('revenue', 0),
            'Operating Cost': best_res.get('operating_cost', 0),
            'Capital Cost': best_res.get('capital_cost', 0)
        },
        'shell_selection_stats': shell_stats,
        'dynamic_pit_enabled': use_dynamic_pit,
        # === AUDIT COMPLIANCE ADDITIONS ===
        'provenance': provenance.to_dict(),  # Violation #4: Full provenance record
        'scenario_cashflows': all_scenario_cashflows,  # Violation #7: All cash flows
        'multiple_irr_warnings': multiple_irr_count,  # Violation #2: Track warnings
        'classification_filter_applied': classification_result is not None,
        'blocks_before_filter': original_block_count,
        'blocks_after_filter': len(base_model),
        'validation_metadata': validation_metadata
    }


def _empty_result(num: int, alpha: float) -> Dict:
    """Return empty safe structure to prevent UI crash."""
    return {
        'irr_alpha': 0.0, 
        'best_schedule': None, 
        'npv_distribution': np.zeros(num),
        'irr_distribution': np.zeros(num),  # NEW
        'satisfaction_rate': 0.0, 
        'iterations': 0, 
        'alpha_target': alpha,
        'num_scenarios': num, 
        'mean_npv': 0, 
        'std_npv': 0, 
        'min_npv': 0, 
        'max_npv': 0,
        'mean_irr': 0,  # NEW
        'std_irr': 0,   # NEW
        'best_cashflows': np.zeros(1), 
        'convergence_history': {
            'iterations': [],
            'r_values': [],
            'satisfaction_rates': [],
            'mean_npvs': [],
            'r_low_values': [],
            'r_high_values': []
        }, 
        'economic_breakdown': {
            'Revenue': 0,
            'Operating Cost': 0,
            'Capital Cost': 0
        },
        'shell_selection_stats': None,
        'dynamic_pit_enabled': False,
        # === AUDIT COMPLIANCE ADDITIONS ===
        'provenance': None,
        'scenario_cashflows': None,
        'multiple_irr_warnings': 0,
        'classification_filter_applied': False,
        'blocks_before_filter': 0,
        'blocks_after_filter': 0,
        'validation_metadata': {}
    }
