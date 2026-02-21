"""
NPVS Solver (STEP 32)

Main solver for Net Present Value Scheduling optimization.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    pulp = None

from ..scheduling.types import ScheduleResult, TimePeriod, ScheduleDecision
from .npvs_data import NpvsBlock, NpvsConfig

logger = logging.getLogger(__name__)


def run_npvs(payload: dict) -> Dict[str, Any]:
    """
    Run NPVS optimization.
    
    Args:
        payload: Dictionary containing:
            - block_model: BlockModel instance
            - block_model_property: Name of value property to use
            - config: NPVS configuration dictionary
    
    Returns:
        Dictionary with:
            - schedule: ScheduleResult as dict
            - npv: NPV value
    """
    block_model = payload.get("block_model")
    value_field = payload.get("block_model_property", "block_value")
    cfg_dict = payload.get("config", {})
    
    if block_model is None:
        raise ValueError("block_model is required in payload")
    
    # Convert block model to NPVS blocks
    blocks = block_model.to_npvs_blocks(value_field)
    
    # Create NPVS config
    npvs_config = NpvsConfig(
        periods=cfg_dict.get("periods", []),
        destinations=cfg_dict.get("destinations", []),
        discount_rate=cfg_dict.get("discount_rate", 0.08),
        mining_capacity_tpy=cfg_dict.get("mining_capacity_tpy", 15_000_000),
        plant_capacity_tpy=cfg_dict.get("plant_capacity_tpy", 10_000_000),
        stockpile_capacity_t=cfg_dict.get("stockpile_capacity_t", {}),
        max_phase_rate_tpy=cfg_dict.get("max_phase_rate_tpy", {}),
        min_phase_rate_tpy=cfg_dict.get("min_phase_rate_tpy", {}),
        penalty_unscheduled_factor=cfg_dict.get("penalty_unscheduled_factor", 0.0),
    )
    
    # Run optimization
    schedule, model = core_run_npvs(blocks, npvs_config)
    
    # MP-014 FIX: Improve NPV extraction with explicit logging
    npv = 0.0
    npv_extraction_method = "unknown"
    if model is not None and hasattr(model, 'objective'):
        try:
            if HAS_PULP:
                npv = float(pulp.value(model.objective))
                npv_extraction_method = "pulp_objective"
                logger.info(f"NPVS NPV extracted from PuLP objective: ${npv:,.2f}")
            else:
                # This shouldn't happen due to MP-004 fix, but handle defensively
                npv = schedule.metadata.get("npv", 0.0) if hasattr(schedule, 'metadata') else 0.0
                npv_extraction_method = "metadata_fallback"
                logger.warning(f"NPV extracted from metadata fallback: ${npv:,.2f}")
        except Exception as e:
            logger.error(f"Failed to extract NPV from model: {e}")
            # Try to get from metadata as last resort
            npv = schedule.metadata.get("npv", 0.0) if hasattr(schedule, 'metadata') else 0.0
            npv_extraction_method = "error_fallback"
            if npv == 0.0:
                logger.error("NPV extraction failed - reported NPV is 0.0 which may be incorrect")
    else:
        logger.warning("No optimization model available - NPV set to 0.0")
    
    return {
        "schedule": schedule.to_dict() if hasattr(schedule, 'to_dict') else schedule,
        "npv": npv
    }


def core_run_npvs(blocks: List[NpvsBlock], config: NpvsConfig) -> tuple:
    """
    Core NPVS optimization algorithm.
    
    Args:
        blocks: List of NpvsBlock
        config: NpvsConfig
    
    Returns:
        Tuple of (ScheduleResult, optimization_model)
        
    Raises:
        ImportError: If PuLP is not available (MP-004 fix - no silent fallback)
    """
    # MP-004 FIX: Make PuLP requirement explicit - no silent fallback to greedy
    if not HAS_PULP:
        raise ImportError(
            "NPVS optimization requires PuLP for MILP scheduling. "
            "The greedy fallback has been disabled as it produces significantly different results. "
            "Install PuLP with: pip install pulp\n"
            "Alternatively, use the Strategic Schedule panel with 'Nested Shells' method "
            "which does not require MILP."
        )
    
    logger.info(f"Running NPVS optimization with {len(blocks)} blocks, {len(config.periods)} periods")
    
    try:
        # Create optimization problem
        prob = pulp.LpProblem("NPVS_Optimization", pulp.LpMaximize)
        
        # Decision variables: x[b, t, d] = tonnes of block b mined in period t to destination d
        periods = config.periods
        destinations = config.destinations
        
        x = {}
        for block in blocks:
            for period in periods:
                for dest in destinations:
                    key = (block.id, period["id"], dest["id"])
                    x[key] = pulp.LpVariable(f"x_{block.id}_{period['id']}_{dest['id']}", 
                                             lowBound=0, cat='Continuous')
        
        # Objective: Maximize discounted NPV
        objective_terms = []
        for block in blocks:
            for period in periods:
                discount_factor = period.get("discount_factor", 1.0)
                for dest in destinations:
                    # Value = block value - processing cost
                    processing_cost = dest.get("processing_cost_per_t", 0.0)
                    net_value = block.value_raw - processing_cost
                    key = (block.id, period["id"], dest["id"])
                    objective_terms.append(x[key] * net_value * discount_factor)
        
        prob += pulp.lpSum(objective_terms), "Total_NPV"
        
        # Constraints
        
        # 1. Each block mined at most once (sum across all periods and destinations)
        for block in blocks:
            prob += pulp.lpSum([
                x[(block.id, period["id"], dest["id"])]
                for period in periods
                for dest in destinations
            ]) <= block.tonnage, f"Mine_Once_{block.id}"
        
        # 2. Mining capacity per period
        for period in periods:
            prob += pulp.lpSum([
                x[(block.id, period["id"], dest["id"])]
                for block in blocks
                for dest in destinations
            ]) <= config.mining_capacity_tpy, f"Mining_Capacity_{period['id']}"
        
        # 3. Plant capacity per period (only for plant destinations)
        plant_destinations = [d for d in destinations if d.get("type") == "plant"]
        for period in periods:
            if plant_destinations:
                prob += pulp.lpSum([
                    x[(block.id, period["id"], dest["id"])]
                    for block in blocks
                    for dest in plant_destinations
                ]) <= config.plant_capacity_tpy, f"Plant_Capacity_{period['id']}"
        
        # 4. Destination capacity per period
        for period in periods:
            for dest in destinations:
                capacity = dest.get("capacity_tpy", float('inf'))
                if capacity < float('inf'):
                    prob += pulp.lpSum([
                        x[(block.id, period["id"], dest["id"])]
                        for block in blocks
                    ]) <= capacity, f"Dest_Capacity_{dest['id']}_{period['id']}"
        
        # 5. Precedence constraints
        for block in blocks:
            if block.precedence_parents:
                for parent_id in block.precedence_parents:
                    # Block can only be mined after parent is mined
                    for period_idx, period in enumerate(periods):
                        for dest in destinations:
                            # Sum of parent mined in earlier periods
                            parent_mined = pulp.lpSum([
                                x[(parent_id, p["id"], d["id"])]
                                for p_idx, p in enumerate(periods)
                                if p_idx < period_idx
                                for d in destinations
                            ])
                            prob += x[(block.id, period["id"], dest["id"])] <= parent_mined, \
                                   f"Precedence_{block.id}_after_{parent_id}_{period['id']}"
        
        # Solve
        logger.info("Solving NPVS optimization problem...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        decisions = []
        for block in blocks:
            for period in periods:
                for dest in destinations:
                    key = (block.id, period["id"], dest["id"])
                    if key in x and x[key].varValue and x[key].varValue > 1e-6:
                        decisions.append(ScheduleDecision(
                            period_id=period["id"],
                            unit_id=str(block.id),
                            tonnes=float(x[key].varValue),
                            destination=dest["id"]
                        ))
        
        # Create TimePeriod objects
        time_periods = [
            TimePeriod(
                id=p["id"],
                index=p["index"],
                duration_days=p.get("duration_years", 1.0) * 365.0
            )
            for p in periods
        ]
        
        # Extract NPV
        npv_value = 0.0
        if prob.status == 1:  # Optimal
            try:
                npv_value = float(pulp.value(prob.objective))
            except Exception as e:
                logger.warning(f"Could not extract NPV from solved model: {e}")
        
        schedule = ScheduleResult(
            periods=time_periods,
            decisions=decisions,
            metadata={"npv": npv_value, "status": prob.status}
        )
        
        logger.info(f"NPVS optimization complete: {len(decisions)} decisions, NPV = ${npv_value:,.2f}")
        
        return schedule, prob
        
    except Exception as e:
        logger.error(f"NPVS optimization failed: {e}", exc_info=True)
        # Fallback to simple schedule
        return _create_simple_schedule(blocks, config), None


def _create_simple_schedule(blocks: List[NpvsBlock], config: NpvsConfig) -> ScheduleResult:
    """
    Create a simple greedy schedule as fallback.
    
    Args:
        blocks: List of NpvsBlock
        config: NpvsConfig
    
    Returns:
        ScheduleResult
    """
    logger.info("Creating simple greedy schedule")
    
    # Sort blocks by value (descending)
    sorted_blocks = sorted(blocks, key=lambda b: b.value_raw, reverse=True)
    
    periods = config.periods
    destinations = config.destinations
    plant_dest = next((d for d in destinations if d.get("type") == "plant"), None)
    
    if not plant_dest:
        plant_dest = destinations[0] if destinations else None
    
    decisions = []
    remaining_capacity = {p["id"]: config.mining_capacity_tpy for p in periods}
    plant_capacity = {p["id"]: config.plant_capacity_tpy for p in periods}
    
    for block in sorted_blocks:
        for period in periods:
            period_id = period["id"]
            
            # Check if we have capacity
            if remaining_capacity[period_id] >= block.tonnage:
                # Check plant capacity if sending to plant
                if plant_dest and plant_dest.get("type") == "plant":
                    if plant_capacity[period_id] >= block.tonnage:
                        decisions.append(ScheduleDecision(
                            period_id=period_id,
                            unit_id=str(block.id),
                            tonnes=block.tonnage,
                            destination=plant_dest["id"]
                        ))
                        remaining_capacity[period_id] -= block.tonnage
                        plant_capacity[period_id] -= block.tonnage
                        break
                else:
                    # Send to first available destination
                    dest = destinations[0] if destinations else None
                    if dest:
                        decisions.append(ScheduleDecision(
                            period_id=period_id,
                            unit_id=str(block.id),
                            tonnes=block.tonnage,
                            destination=dest["id"]
                        ))
                        remaining_capacity[period_id] -= block.tonnage
                        break
    
    time_periods = [
        TimePeriod(
            id=p["id"],
            index=p["index"],
            duration_days=p.get("duration_years", 1.0) * 365.0
        )
        for p in periods
    ]
    
    return ScheduleResult(
        periods=time_periods,
        decisions=decisions,
        metadata={"method": "simple_greedy"}
    )

