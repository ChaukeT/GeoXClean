"""
SLOS Optimizer (STEP 37)

Optimize SLOS stope sequence and scheduling.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

from .slos_geometry import StopeInstance
from ...mine_planning.scheduling.types import TimePeriod, ScheduleDecision, ScheduleResult

logger = logging.getLogger(__name__)


@dataclass
class SlosOptimiserConfig:
    """
    Configuration for SLOS schedule optimization.
    
    Attributes:
        periods: List of TimePeriod objects
        discount_rate: Discount rate for NPV calculation
        target_tonnes_per_period: Target tonnes per period
        max_concurrent_stopes: Maximum concurrent stopes
        development_lag_periods: Development lag in periods
        geotech_factors: Optional geotechnical factors
    """
    periods: List[TimePeriod]
    discount_rate: float = 0.10
    target_tonnes_per_period: float = 100_000.0
    max_concurrent_stopes: int = 5
    development_lag_periods: int = 2
    geotech_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlosScheduleResult:
    """
    Result from SLOS schedule optimization.
    
    Attributes:
        periods: List of TimePeriod objects
        decisions: List of ScheduleDecision objects
        metadata: Additional metadata
    """
    periods: List[TimePeriod]
    decisions: List[ScheduleDecision]
    metadata: Dict[str, Any] = field(default_factory=dict)


def optimise_slos_schedule(
    stopes: List[StopeInstance],
    config: SlosOptimiserConfig
) -> SlosScheduleResult:
    """
    Optimize SLOS schedule to maximize NPV.
    
    Args:
        stopes: List of StopeInstance objects
        config: SlosOptimiserConfig
    
    Returns:
        SlosScheduleResult
    """
    logger.info(f"Optimizing SLOS schedule for {len(stopes)} stopes over {len(config.periods)} periods")
    
    if not stopes:
        logger.warning("No stopes provided for optimization")
        return SlosScheduleResult(
            periods=config.periods,
            decisions=[],
            metadata={'error': 'No stopes provided'}
        )
    
    if not config.periods:
        logger.warning("No periods provided for optimization")
        return SlosScheduleResult(
            periods=[],
            decisions=[],
            metadata={'error': 'No periods provided'}
        )
    
    # Calculate stope values (simplified - would use actual prices/costs)
    stope_values = {}
    for stope in stopes:
        # Simple value calculation: tonnes * average grade * price - costs
        # In practice, would use actual prices and costs
        value = stope.tonnes * 100.0  # Placeholder
        stope_values[stope.id] = value
    
    # Simple greedy scheduling algorithm
    # In production, would use MILP or max-closure variant
    decisions = []
    scheduled_stopes = set()
    
    for period_idx, period in enumerate(config.periods):
        period_tonnes = 0.0
        period_stopes = []
        
        # Select stopes for this period
        for stope in stopes:
            if stope.id in scheduled_stopes:
                continue
            
            # Check precedence
            if any(pred not in scheduled_stopes for pred in stope.predecessors):
                continue
            
            # Check if adding this stope would exceed target
            if period_tonnes + stope.tonnes > config.target_tonnes_per_period * 1.1:  # 10% tolerance
                continue
            
            # Check max concurrent stopes
            if len(period_stopes) >= config.max_concurrent_stopes:
                continue
            
            period_stopes.append(stope)
            period_tonnes += stope.tonnes
            scheduled_stopes.add(stope.id)
            
            # Create decision
            decision = ScheduleDecision(
                period_id=period.id,
                unit_id=stope.id,
                tonnes=stope.tonnes,
                destination="plant"  # Default destination
            )
            decisions.append(decision)
        
        logger.debug(f"Period {period.id}: Scheduled {len(period_stopes)} stopes, {period_tonnes:,.0f} tonnes")
    
    # Calculate NPV
    npv = 0.0
    for period_idx, period in enumerate(config.periods):
        period_decisions = [d for d in decisions if d.period_id == period.id]
        period_value = sum(stope_values.get(d.unit_id, 0.0) for d in period_decisions)
        discount_factor = 1.0 / ((1.0 + config.discount_rate) ** period_idx)
        npv += period_value * discount_factor
    
    metadata = {
        'npv': npv,
        'total_stopes': len(stopes),
        'scheduled_stopes': len(scheduled_stopes),
        'total_tonnes': sum(d.tonnes for d in decisions),
        'algorithm': 'greedy'  # Would be 'milp' in production
    }
    
    logger.info(f"SLOS schedule optimization complete: NPV = ${npv:,.0f}")
    
    return SlosScheduleResult(
        periods=config.periods,
        decisions=decisions,
        metadata=metadata
    )

