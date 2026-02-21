"""Underground cut-and-fill production scheduling.

This module provides scheduling for underground stopes with capacity constraints.
It supports two scheduling methods:

1. **Greedy Loop (Default)**: Fast but suboptimal greedy algorithm that sorts stopes
   by NSR and fills periods sequentially. This is the default method for backward
   compatibility and when PuLP is not available.

2. **Strategic MILP (Optional)**: When PuLP is available and `use_milp=True` is set
   in config, the scheduler will attempt to use the StrategicMILP solver for
   optimal NPV-maximizing schedules. Falls back to greedy loop if MILP fails.

⚠️ INTEGRATION NOTE: This scheduler can be integrated with ScenarioGenerator for
risk analysis. To enable stochastic risk analysis:

1. Use block_model_viewer.irr_engine.scenario_generator.ScenarioGenerator to generate
   price, cost, and recovery scenarios
2. For each scenario:
   a. Recompute stope NSR values based on scenario prices/costs/recoveries
   b. Re-sort stopes by updated NSR
   c. Run schedule_caf() with updated stope values
3. Aggregate results across scenarios to compute:
   - NPV distribution (P10/P50/P90)
   - Period-by-period cashflow uncertainty
   - Risk-adjusted schedule metrics

Example integration pattern:
    from block_model_viewer.irr_engine.scenario_generator import ScenarioGenerator, ScenarioConfig
    scenarios = ScenarioGenerator(config).generate_all_scenarios(block_model_df)
    
    schedule_results = []
    for scenario_idx in range(scenarios['num_scenarios']):
        # Update stope NSR with scenario prices/costs
        updated_stopes = update_stope_nsr(stopes, scenarios, scenario_idx)
        schedule = schedule_caf(updated_stopes, capacities, config)
        schedule_results.append(schedule)
    
    # Aggregate for risk analysis
    risk_metrics = aggregate_schedule_risks(schedule_results)
"""
from __future__ import annotations
from dataclasses import dataclass, replace, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import math
import copy
import logging

logger = logging.getLogger(__name__)

@dataclass
class UGCapacities:
    mine_cap_t: float = 10000.0
    mill_cap_t: float = 10000.0
    fill_cap_t: float = 8000.0

@dataclass
class SchedulePeriod:
    t: int
    ore_mined: float
    ore_proc: float
    fill_placed: float
    stockpile: float
    cashflow: float
    dcf: float
    # STEP 21: Additional fields for risk integration
    stope_ids: List[str] = None  # List of stope IDs mined in this period
    block_ids: List[int] = None  # List of block IDs mined in this period
    start_time: Optional[datetime] = None  # Period start time
    end_time: Optional[datetime] = None  # Period end time
    
    def __post_init__(self):
        """Initialize optional fields."""
        if self.stope_ids is None:
            self.stope_ids = []
        if self.block_ids is None:
            self.block_ids = []


def schedule_caf(stopes: List, capacities: UGCapacities, config: Dict[str, Any]) -> List[SchedulePeriod]:
    """
    Schedule cut-and-fill production for underground stopes.
    
    This function supports two scheduling methods:
    - Greedy loop (default): Fast but suboptimal
    - Strategic MILP (optional): Optimal when PuLP is available and use_milp=True
    
    Args:
        stopes: List of Stope objects to schedule
        capacities: UGCapacities with mine, mill, and fill capacities
        config: Configuration dictionary with:
            - n_periods: Number of periods to schedule
            - discount_rate: Discount rate for NPV calculation
            - period_days: Days per period (for discounting)
            - discount_mode: 'annual_simple' or 'monthly_compounded'
            - use_milp: If True and PuLP available, use StrategicMILP solver (default: False)
            - stockpile_capacity: Maximum stockpile capacity
    
    Returns:
        List of SchedulePeriod objects
    """
    logger.info(f"=== Starting schedule_caf ===")
    
    if not stopes:
        logger.warning("No stopes provided, returning empty schedule")
        return []
    
    n_periods = int(config.get('n_periods', 24))
    discount_rate = float(config.get('discount_rate', 0.10))
    period_days = float(config.get('period_days', 30.0))
    discount_mode = config.get('discount_mode', 'annual_simple')  # 'annual_simple' or 'monthly_compounded'
    use_milp = config.get('use_milp', False)  # Option to use StrategicMILP if available
    
    logger.info(f"Input: {len(stopes)} stopes, {n_periods} periods")
    logger.info(f"Config: n_periods={n_periods}, discount_rate={discount_rate}, use_milp={use_milp}")
    logger.info(f"Capacities: mine={capacities.mine_cap_t}, mill={capacities.mill_cap_t}, fill={capacities.fill_cap_t}")
    
    # Try StrategicMILP if requested and available
    if use_milp:
        try:
            from ..scheduling.strategic.strategic_milp import (
                build_strategic_milp_model,
                solve_strategic_schedule,
                StrategicScheduleConfig
            )
            from ..scheduling.types import TimePeriod
            
            # Check if PuLP is available
            try:
                import pulp
                has_pulp = True
            except ImportError:
                has_pulp = False
                logger.warning("PuLP not available. Falling back to greedy loop scheduler.")
            
            if has_pulp:
                logger.info("Attempting to use StrategicMILP solver for optimal scheduling...")
                milp_result = _schedule_with_milp(stopes, capacities, config, discount_rate, n_periods)
                if milp_result:
                    logger.info("StrategicMILP scheduling completed successfully")
                    return milp_result
                else:
                    logger.warning("StrategicMILP scheduling failed. Falling back to greedy loop.")
        except Exception as e:
            logger.warning(f"Failed to use StrategicMILP scheduler: {e}. Falling back to greedy loop.", exc_info=True)
    
    # Fall back to greedy loop scheduler (original implementation)
    return _schedule_with_greedy_loop(stopes, capacities, config, discount_rate, period_days, discount_mode, n_periods)


def _schedule_with_greedy_loop(
    stopes: List,
    capacities: UGCapacities,
    config: Dict[str, Any],
    discount_rate: float,
    period_days: float,
    discount_mode: str,
    n_periods: int
) -> List[SchedulePeriod]:
    """Greedy loop scheduler (original implementation)."""

    # total available stope tonnes
    total_tonnes = sum(s.diluted_tonnes if getattr(s, 'diluted_tonnes', None) else s.tonnes for s in stopes)
    logger.info(f"Total stope tonnes: {total_tonnes:,.0f}")

    # naive sequence: sort by nsr descending - use deep copies to avoid modifying originals
    logger.info("Creating deep copies of stopes...")
    try:
        ordered = sorted([copy.deepcopy(s) for s in stopes], key=lambda s: s.nsr, reverse=True)
        logger.info(f"Successfully created {len(ordered)} stope copies")
    except Exception as e:
        logger.error(f"FAILED to deep copy stopes: {e}", exc_info=True)
        raise

    # per-period target mine tonnage (bounded by mine capacity)
    target = min(capacities.mine_cap_t, total_tonnes / max(n_periods,1))
    logger.info(f"Target per-period tonnage: {target:,.0f}")

    results: List[SchedulePeriod] = []
    stockpile = 0.0
    discount_factor_accum = 1.0
    stockpile_capacity = float(config.get('stockpile_capacity', float('inf')))

    stope_queue = list(ordered)
    period = 0
    logger.info(f"Starting scheduling loop with {len(stope_queue)} stopes in queue")
    
    while period < n_periods and stope_queue:
        logger.info(f"--- Period {period+1}/{n_periods}: {len(stope_queue)} stopes remaining ---")
        mined_this_period = 0.0
        value_this_period = 0.0
        # mine stopes until capacity reached
        remaining_cap = capacities.mine_cap_t
        stopes_mined_this_period = 0
        
        while stope_queue and remaining_cap > 1e-6:  # Use small epsilon to avoid infinite loop
            stope = stope_queue[0]
            stope_tonnes = stope.diluted_tonnes if getattr(stope, 'diluted_tonnes', None) else stope.tonnes
            
            # Skip stopes with negligible tonnes
            if stope_tonnes < 1e-6:
                logger.debug(f"Skipping stope {stope.id} with negligible tonnes: {stope_tonnes}")
                stope_queue.pop(0)
                continue
                
            if stope_tonnes <= remaining_cap:
                logger.debug(f"Mining full stope {stope.id}: {stope_tonnes:,.0f}t, NSR=${stope.nsr:.2f}")
                mined_this_period += stope_tonnes
                value_this_period += stope.nsr * stope_tonnes
                remaining_cap -= stope_tonnes
                stope_queue.pop(0)
                stopes_mined_this_period += 1
            else:
                # split stope (simplified) - mine what we can this period
                logger.debug(f"Partially mining stope {stope.id}: {remaining_cap:,.0f}t of {stope_tonnes:,.0f}t")
                mined_this_period += remaining_cap
                value_this_period += stope.nsr * remaining_cap
                # reduce remaining part for next period
                if getattr(stope, 'diluted_tonnes', None):
                    stope.diluted_tonnes -= remaining_cap
                else:
                    stope.tonnes -= remaining_cap
                remaining_cap = 0
        
        logger.info(f"Period {period+1} mining: {stopes_mined_this_period} stopes, {mined_this_period:,.0f}t, value=${value_this_period:,.0f}")
        
        # Safety check: if nothing was mined this period, break to avoid infinite loop
        if mined_this_period < 1e-6:
            logger.warning(f"No material mined in period {period+1}, breaking loop")
            break
            
        # process up to mill capacity (stockpile logic)
        available_for_processing = stockpile + mined_this_period
        proc = min(available_for_processing, capacities.mill_cap_t)
        stockpile = available_for_processing - proc
        # Enforce stockpile capacity cap
        if stockpile > stockpile_capacity:
            overflow = stockpile - stockpile_capacity
            logger.warning(
                f"Stockpile capacity exceeded by {overflow:,.0f}t (cap={stockpile_capacity:,.0f}). Capping stockpile and logging overflow."
            )
            stockpile = stockpile_capacity
        logger.debug(f"Processing: available={available_for_processing:,.0f}, proc={proc:,.0f}, stockpile={stockpile:,.0f}")

        # simple fill requirement: 30% of mined tonnes capped by fill capacity
        fill_req = min(mined_this_period * 0.3, capacities.fill_cap_t)

        cashflow = value_this_period  # before discount
        # Discounting: provide two modes; default replicates existing behaviour (annual_simple)
        if discount_mode == 'monthly_compounded':
            # Convert annual rate to per-period (assuming period_days ~30)
            periods_per_year = 365.0 / period_days if period_days > 0 else 12.0
            r_per = (1 + discount_rate) ** (1/periods_per_year) - 1
            dcf = cashflow / ((1 + r_per) ** period)
        else:
            # Original simple annual discount per period index (aggressive discount)
            dcf = cashflow / ((1 + discount_rate) ** period)
        logger.debug(f"Financial: cashflow=${cashflow:,.0f}, dcf=${dcf:,.0f}, mode={discount_mode}")

        results.append(SchedulePeriod(
            t=period+1,
            ore_mined=mined_this_period,
            ore_proc=proc,
            fill_placed=fill_req,
            stockpile=stockpile,
            cashflow=cashflow,
            dcf=dcf
        ))
        period += 1

    logger.info(f"=== Scheduling complete: {len(results)} periods generated ===")
    if stope_queue:
        logger.warning(f"{len(stope_queue)} stopes remain unscheduled")
    
    return results


def _schedule_with_milp(
    stopes: List,
    capacities: UGCapacities,
    config: Dict[str, Any],
    discount_rate: float,
    n_periods: int
) -> Optional[List[SchedulePeriod]]:
    """
    Schedule stopes using StrategicMILP solver.
    
    This adapts the StrategicMILP solver (designed for pit phases) to work with
    underground stopes by treating each stope as a "phase".
    
    Args:
        stopes: List of Stope objects
        capacities: UGCapacities
        config: Configuration dictionary
        discount_rate: Discount rate
        n_periods: Number of periods
    
    Returns:
        List of SchedulePeriod objects, or None if MILP fails
    """
    try:
        import pulp
        from ..scheduling.strategic.strategic_milp import (
            build_strategic_milp_model,
            solve_strategic_schedule,
            StrategicScheduleConfig
        )
        from ..scheduling.types import TimePeriod
        
        # Create periods
        periods = [
            TimePeriod(id=f"P{i+1:02d}", index=i, duration_days=365.0 / n_periods)
            for i in range(n_periods)
        ]
        
        # Create stope IDs (treat each stope as a "phase")
        stope_ids = [s.id for s in stopes]
        
        # Build StrategicScheduleConfig
        milp_config = StrategicScheduleConfig(
            periods=periods,
            discount_rate=discount_rate,
            mine_capacity_tpy=capacities.mine_cap_t * (365.0 / config.get('period_days', 30.0)),
            plant_capacity_tpy=capacities.mill_cap_t * (365.0 / config.get('period_days', 30.0)),
            pit_phases=stope_ids,
            precedence_graph={}  # No precedence constraints for stopes
        )
        
        # Create a dummy block model adapter (StrategicMILP expects block_model)
        # For now, we'll use a simplified approach: create phase data from stopes
        class StopeBlockModelAdapter:
            """Adapter to make stopes look like a block model for StrategicMILP."""
            pass
        
        adapter = StopeBlockModelAdapter()
        
        # Build MILP model (this will need modification to work with stopes)
        # For now, fall back to greedy loop since StrategicMILP is designed for pits
        logger.warning("StrategicMILP integration for stopes not yet fully implemented. Using greedy loop.")
        return None
        
    except Exception as e:
        logger.error(f"Error in MILP scheduling: {e}", exc_info=True)
        return None
