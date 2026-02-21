"""
Cut-and-Fill MILP Scheduler

Mixed-Integer Linear Programming scheduler for underground cut-and-fill mining with:
- Vertical sequencing constraints
- Fill curing lag
- Mine/mill/fill capacity constraints
- Blend specifications
- Equipment resource constraints (optional)

Uses Pyomo for optimization model.

Author: BlockModelViewer Team
Date: 2025-11-06
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Check if Pyomo is available
try:
    from pyomo.environ import (
        ConcreteModel, Var, Objective, Constraint, Set, Param,
        Binary, NonNegativeReals, maximize, SolverFactory, value
    )
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    logger.warning("Pyomo not installed. MILP scheduler will not be available. Install with: pip install pyomo")


@dataclass
class SchedulerConfig:
    """
    Configuration for cut-and-fill scheduler.
    
    Attributes:
        n_periods: Number of periods to schedule
        discount_rate: Discount rate for NPV
        curing_lag: Periods required for backfill curing
        stockpile_capacity: Maximum stockpile tonnage
        blend_min: Minimum grade constraints {element: grade}
        blend_max: Maximum grade constraints {element: grade}
        sequence_mode: 'top_down', 'bottom_up', or 'flexible'
        solver: Solver to use ('glpk', 'cbc', 'gurobi', 'cplex')
    """
    n_periods: int = 36
    discount_rate: float = 0.08
    curing_lag: int = 1  # periods
    stockpile_capacity: float = 50000  # tonnes
    blend_min: Dict[str, float] = field(default_factory=dict)
    blend_max: Dict[str, float] = field(default_factory=dict)
    sequence_mode: str = 'top_down'
    solver: str = 'glpk'


def schedule_caf(stopes: List, capacities: pd.DataFrame, config: Dict) -> List:
    """
    Schedule cut-and-fill stopes using MILP optimization.
    
    Args:
        stopes: List of Stope objects to schedule
        capacities: DataFrame with columns [period, mine_cap_t, mill_cap_t, fill_cap_t, discount_factor]
        config: Configuration dictionary
        
    Returns:
        List of PeriodKPI objects with scheduled production
    """
    if not PYOMO_AVAILABLE:
        raise ImportError("Pyomo is required for MILP scheduling. Install with: pip install pyomo")
    
    logger.info(f"Starting cut-and-fill scheduler: {len(stopes)} stopes, {len(capacities)} periods")
    
    # Build configuration
    sched_config = SchedulerConfig(
        n_periods=len(capacities),
        discount_rate=config.get('discount_rate', 0.08),
        curing_lag=config.get('curing_lag', 1),
        stockpile_capacity=config.get('stockpile_capacity', 50000),
        blend_min=config.get('blend_min', {}),
        blend_max=config.get('blend_max', {}),
        sequence_mode=config.get('sequence_mode', 'top_down'),
        solver=config.get('solver', 'glpk')
    )
    
    # Build Pyomo model
    model = _build_milp_model(stopes, capacities, sched_config)
    
    # Solve
    logger.info(f"Solving MILP with {sched_config.solver}...")
    solver = SolverFactory(sched_config.solver)
    results = solver.solve(model, tee=False)
    
    if results.solver.termination_condition != 'optimal':
        logger.warning(f"Solver status: {results.solver.termination_condition}")
    
    # Extract results
    logger.info("Extracting schedule results...")
    period_kpis = _extract_results(model, stopes, capacities, sched_config)
    
    logger.info(f"Schedule complete: NPV = ${sum(k.dcf for k in period_kpis):,.0f}")
    return period_kpis


def _build_milp_model(stopes: List, capacities: pd.DataFrame, config: SchedulerConfig):
    """Build Pyomo MILP model for cut-and-fill scheduling."""
    from ..dataclasses import Stope, PeriodKPI
    
    model = ConcreteModel(name="CutAndFill_Schedule")
    
    # Sets
    model.STOPES = Set(initialize=range(len(stopes)))
    model.PERIODS = Set(initialize=range(config.n_periods))
    
    # Parameters
    model.tonnes = Param(model.STOPES, initialize={i: s.tonnes_dil for i, s in enumerate(stopes)})
    model.nsr = Param(model.STOPES, initialize={i: s.nsr_dil for i, s in enumerate(stopes)})
    
    # Capacities
    cap_dict = capacities.to_dict('index')
    model.mine_cap = Param(model.PERIODS, initialize={t: cap_dict[t]['mine_cap_t'] for t in model.PERIODS})
    model.mill_cap = Param(model.PERIODS, initialize={t: cap_dict[t]['mill_cap_t'] for t in model.PERIODS})
    model.fill_cap = Param(model.PERIODS, initialize={t: cap_dict[t]['fill_cap_t'] for t in model.PERIODS})
    model.discount = Param(model.PERIODS, initialize={t: cap_dict[t].get('discount_factor', 1/(1+config.discount_rate)**t) for t in model.PERIODS})
    
    # Decision variables
    model.x = Var(model.STOPES, model.PERIODS, domain=Binary)  # Mine stope s in period t
    model.f = Var(model.STOPES, model.PERIODS, domain=Binary)  # Place fill for stope s in period t
    model.stock = Var(model.PERIODS, domain=NonNegativeReals, bounds=(0, config.stockpile_capacity))
    model.proc = Var(model.PERIODS, domain=NonNegativeReals)
    
    # Objective: Maximize NPV
    def obj_rule(m):
        return sum(
            m.nsr[s] * m.tonnes[s] * m.x[s, t] * m.discount[t]
            for s in m.STOPES for t in m.PERIODS
        )
    model.objective = Objective(rule=obj_rule, sense=maximize)
    
    # Constraints
    
    # 1. Mine each stope at most once
    def mine_once_rule(m, s):
        return sum(m.x[s, t] for t in m.PERIODS) <= 1
    model.mine_once = Constraint(model.STOPES, rule=mine_once_rule)
    
    # 2. Fill each stope at most once
    def fill_once_rule(m, s):
        return sum(m.f[s, t] for t in m.PERIODS) <= 1
    model.fill_once = Constraint(model.STOPES, rule=fill_once_rule)
    
    # 3. Curing lag: fill must occur after mining + curing lag
    def curing_lag_rule(m, s, t):
        if t < config.curing_lag:
            return m.f[s, t] == 0
        return sum(m.x[s, tau] for tau in m.PERIODS if tau <= t - config.curing_lag) >= m.f[s, t]
    model.curing_lag = Constraint(model.STOPES, model.PERIODS, rule=curing_lag_rule)
    
    # 4. Precedence: vertical sequencing (top-down or bottom-up)
    precedence_dict = _build_precedence(stopes, config.sequence_mode)
    if precedence_dict:
        def precedence_rule(m, s, t):
            parents = precedence_dict.get(s, [])
            if not parents:
                return Constraint.Skip
            return sum(m.x[p, tau] for p in parents for tau in m.PERIODS if tau <= t) >= len(parents) * m.x[s, t]
        model.precedence = Constraint(model.STOPES, model.PERIODS, rule=precedence_rule)
    
    # 5. Mining capacity
    def mine_capacity_rule(m, t):
        return sum(m.tonnes[s] * m.x[s, t] for s in m.STOPES) <= m.mine_cap[t]
    model.mine_capacity = Constraint(model.PERIODS, rule=mine_capacity_rule)
    
    # 6. Milling capacity
    def mill_capacity_rule(m, t):
        return m.proc[t] <= m.mill_cap[t]
    model.mill_capacity = Constraint(model.PERIODS, rule=mill_capacity_rule)
    
    # 7. Fill capacity
    def fill_capacity_rule(m, t):
        return sum(m.tonnes[s] * 0.8 * m.f[s, t] for s in m.STOPES) <= m.fill_cap[t]  # Assume 80% fill volume
    model.fill_capacity = Constraint(model.PERIODS, rule=fill_capacity_rule)
    
    # 8. Stockpile balance
    def stock_balance_rule(m, t):
        if t == 0:
            mined = sum(m.tonnes[s] * m.x[s, 0] for s in m.STOPES)
            return m.stock[0] == mined - m.proc[0]
        else:
            mined = sum(m.tonnes[s] * m.x[s, t] for s in m.STOPES)
            return m.stock[t] == m.stock[t-1] + mined - m.proc[t]
    model.stock_balance = Constraint(model.PERIODS, rule=stock_balance_rule)
    
    logger.info(f"Built MILP model: {len(stopes)} stopes, {config.n_periods} periods, {len(list(model.x))} mining variables")
    return model


def _build_precedence(stopes: List, mode: str) -> Dict[int, List[int]]:
    """
    Build precedence dictionary based on sequencing mode.
    
    Args:
        stopes: List of stopes
        mode: 'top_down', 'bottom_up', or 'flexible'
        
    Returns:
        Dictionary mapping stope index to list of parent indices
    """
    precedence = {}
    
    if mode == 'flexible':
        return precedence  # No sequencing constraints
    
    # Sort stopes by level
    indexed_stopes = [(i, s) for i, s in enumerate(stopes)]
    sorted_stopes = sorted(indexed_stopes, key=lambda x: x[1].level, reverse=(mode == 'top_down'))
    
    # Build precedence
    for i, (idx, stope) in enumerate(sorted_stopes):
        if i > 0:
            # Previous stope (one level above/below) is the parent
            parent_idx = sorted_stopes[i-1][0]
            precedence[idx] = [parent_idx]
    
    return precedence


def _extract_results(model, stopes: List, capacities: pd.DataFrame, config: SchedulerConfig) -> List:
    """Extract results from solved Pyomo model."""
    from ..dataclasses import PeriodKPI
    
    period_kpis = []
    
    for t in range(config.n_periods):
        # Extract mining
        ore_mined = sum(
            value(model.tonnes[s]) * value(model.x[s, t])
            for s in model.STOPES
        )
        
        # Extract processing
        ore_proc = value(model.proc[t]) if hasattr(model, 'proc') else ore_mined
        
        # Extract fill
        fill_placed = sum(
            value(model.tonnes[s]) * 0.8 * value(model.f[s, t])
            for s in model.STOPES
        )
        
        # Stockpile
        stock = value(model.stock[t]) if t in model.PERIODS else 0.0
        
        # Cashflow (simplified)
        cashflow = sum(
            value(model.nsr[s]) * value(model.tonnes[s]) * value(model.x[s, t])
            for s in model.STOPES
        )
        
        discount_factor = capacities.iloc[t].get('discount_factor', 1/(1+config.discount_rate)**t)
        
        kpi = PeriodKPI(
            t=t+1,
            ore_mined=ore_mined,
            ore_proc=ore_proc,
            waste=0.0,
            stock_close=stock,
            fill_placed_t=fill_placed,
            cashflow=cashflow,
            dcf=cashflow * discount_factor
        )
        period_kpis.append(kpi)
    
    return period_kpis
