"""
Strategic MILP Scheduler (STEP 30)

NPV-maximizing annual schedule using MILP.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import logging

from ..types import TimePeriod, ScheduleDecision, ScheduleResult

logger = logging.getLogger(__name__)

# Try to import PuLP
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    logger.warning("PuLP not installed. Strategic MILP scheduling will not be available.")


@dataclass
class StrategicScheduleConfig:
    """
    Configuration for strategic MILP scheduling.
    
    Attributes:
        periods: List of TimePeriod (annual)
        discount_rate: Discount rate for NPV
        mine_capacity_tpy: Mining capacity (tonnes per year)
        plant_capacity_tpy: Plant capacity (tonnes per year)
        stockpile_capacity: Dictionary mapping stockpile name -> capacity
        pit_phases: List of phase IDs from pit_phases / nested shells
        block_source: "long_model" or "geomet"
        block_value_field: Chosen value property name
        precedence_graph: Dictionary mapping phase_id -> list of predecessor phase IDs
    """
    periods: List[TimePeriod]
    discount_rate: float
    mine_capacity_tpy: float
    plant_capacity_tpy: float
    stockpile_capacity: Dict[str, float] = field(default_factory=dict)
    pit_phases: List[str] = field(default_factory=list)
    block_source: str = "long_model"
    block_value_field: str = "value"
    precedence_graph: Dict[str, List[str]] = field(default_factory=dict)


def build_strategic_milp_model(
    block_model: Any,
    config: StrategicScheduleConfig,
    solver_config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Build strategic MILP model for scheduling.
    
    Args:
        block_model: BlockModel instance
        config: StrategicScheduleConfig
        solver_config: Optional solver configuration
        
    Returns:
        PuLP problem instance (or None if PuLP unavailable)
    """
    if not HAS_PULP:
        logger.error("PuLP not available. Cannot build MILP model.")
        return None
    
    solver_config = solver_config or {}
    
    # Create problem
    prob = pulp.LpProblem("Strategic_Schedule", pulp.LpMaximize)
    
    # Get phase tonnage and value data
    phase_data = {}
    for phase_id in config.pit_phases:
        # Extract blocks for this phase (simplified - would need actual phase mapping)
        phase_data[phase_id] = {
            "tonnes": 0.0,
            "ore_tonnes": 0.0,
            "value": 0.0
        }
    
    # Decision variables: x[p, t] = tonnes of phase p mined in period t
    x = {}
    for phase_id in config.pit_phases:
        x[phase_id] = {}
        for period in config.periods:
            x[phase_id][period.id] = pulp.LpVariable(
                f"x_{phase_id}_{period.id}",
                lowBound=0.0,
                upBound=phase_data.get(phase_id, {}).get("tonnes", 1e6),
                cat='Continuous'
            )
    
    # Objective: Maximize discounted NPV
    objective = 0.0
    for phase_id in config.pit_phases:
        phase_value = phase_data.get(phase_id, {}).get("value", 0.0)
        phase_tonnes = phase_data.get(phase_id, {}).get("tonnes", 1.0)
        if phase_tonnes > 0:
            value_per_tonne = phase_value / phase_tonnes
        else:
            value_per_tonne = 0.0
        
        for period in config.periods:
            discount_factor = 1.0 / ((1.0 + config.discount_rate) ** period.index)
            objective += value_per_tonne * x[phase_id][period.id] * discount_factor
    
    prob += objective
    
    # Constraints
    # 1. Mining capacity per period
    for period in config.periods:
        prob += pulp.lpSum([x[p][period.id] for p in config.pit_phases]) <= config.mine_capacity_tpy, \
                f"MineCapacity_{period.id}"
    
    # 2. Plant capacity per period (ore only)
    for period in config.periods:
        ore_tonnes = pulp.lpSum([
            (phase_data.get(p, {}).get("ore_tonnes", 0.0) / max(phase_data.get(p, {}).get("tonnes", 1.0), 1.0)) * x[p][period.id]
            for p in config.pit_phases
        ])
        prob += ore_tonnes <= config.plant_capacity_tpy, f"PlantCapacity_{period.id}"
    
    # 3. Precedence constraints
    for phase_id, predecessors in config.precedence_graph.items():
        if phase_id not in config.pit_phases:
            continue
        
        for pred_id in predecessors:
            if pred_id not in config.pit_phases:
                continue
            
            # Cumulative precedence: sum of phase_id mined up to period t <= sum of pred_id mined up to period t
            for period_idx, period in enumerate(config.periods):
                cumulative_phase = pulp.lpSum([x[phase_id][p.id] for p in config.periods[:period_idx+1]])
                cumulative_pred = pulp.lpSum([x[pred_id][p.id] for p in config.periods[:period_idx+1]])
                prob += cumulative_phase <= cumulative_pred, f"Precedence_{phase_id}_{pred_id}_{period.id}"
    
    # 4. Phase tonnage limits
    for phase_id in config.pit_phases:
        total_tonnes = pulp.lpSum([x[phase_id][p.id] for p in config.periods])
        phase_max_tonnes = phase_data.get(phase_id, {}).get("tonnes", 1e6)
        prob += total_tonnes <= phase_max_tonnes, f"PhaseLimit_{phase_id}"
    
    logger.info(f"Built strategic MILP model: {len(config.pit_phases)} phases, {len(config.periods)} periods")
    return prob


def solve_strategic_schedule(model: Any, config: Optional[StrategicScheduleConfig] = None) -> ScheduleResult:
    """
    Solve strategic MILP model and return schedule result.
    
    Args:
        model: PuLP problem instance
        config: Optional StrategicScheduleConfig (to extract periods)
        
    Returns:
        ScheduleResult
    """
    if model is None:
        logger.error("Model is None. Cannot solve.")
        return ScheduleResult()
    
    if not HAS_PULP:
        logger.error("PuLP not available. Cannot solve MILP model.")
        return ScheduleResult()
    
    # Solve
    try:
        solver = pulp.PULP_CBC_CMD(timeLimit=300, msg=0)
        model.solve(solver)
        
        status = pulp.LpStatus[model.status]
        logger.info(f"Strategic MILP solution status: {status}")
        
        if status not in ['Optimal', 'Feasible']:
            logger.warning(f"MILP did not find optimal solution: {status}")
            return ScheduleResult()
        
        # Extract solution
        decisions = []
        periods = config.periods if config else []
        
        # Extract decisions from solved variables
        for var in model.variables():
            if var.varValue is not None and var.varValue > 1e-6:
                # Parse variable name: x_phaseId_periodId
                parts = var.name.split('_')
                if len(parts) >= 3 and parts[0] == 'x':
                    phase_id = parts[1]
                    period_id = '_'.join(parts[2:])
                    
                    decisions.append(ScheduleDecision(
                        period_id=period_id,
                        unit_id=phase_id,
                        tonnes=var.varValue,
                        destination="plant"
                    ))
        
        objective_value = pulp.value(model.objective) if model.objective else 0.0
        logger.info(f"Optimal NPV: ${objective_value:,.2f}")
        logger.info(f"Extracted {len(decisions)} schedule decisions")
        
        return ScheduleResult(
            periods=periods,
            decisions=decisions,
            metadata={
                "status": status,
                "objective_value": objective_value,
                "solver": "CBC"
            }
        )
    
    except Exception as e:
        logger.error(f"Error solving strategic MILP: {e}", exc_info=True)
        return ScheduleResult()

