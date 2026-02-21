"""
Tactical Pushback Scheduler (STEP 30)

Turn strategic annual tonnes into monthly per-pushback/phase targets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

from ..types import TimePeriod, ScheduleDecision, ScheduleResult

logger = logging.getLogger(__name__)


@dataclass
class TacticalScheduleConfig:
    """
    Configuration for tactical scheduling.
    
    Attributes:
        periods: List of TimePeriod (months/quarters)
        strategic_schedule: Strategic ScheduleResult
        max_active_pushbacks: Maximum number of active pushbacks
        bench_increment: Bench height increment
        smoothing_window: Smoothing window for tonnage distribution
    """
    periods: List[TimePeriod]
    strategic_schedule: ScheduleResult
    max_active_pushbacks: int = 3
    bench_increment: float = 15.0
    smoothing_window: int = 3


def derive_pushback_schedule(
    pit_phases: Any,
    strategic_schedule: ScheduleResult,
    config: TacticalScheduleConfig
) -> ScheduleResult:
    """
    Derive tactical pushback schedule from strategic schedule.
    
    Args:
        pit_phases: Pit phases data
        strategic_schedule: Strategic ScheduleResult
        config: TacticalScheduleConfig
        
    Returns:
        ScheduleResult with tactical decisions
    """
    # Get strategic tonnes by period
    strategic_tonnes = strategic_schedule.get_tonnes_by_period()
    
    # Distribute strategic tonnes across tactical periods
    decisions = []
    
    # Simple distribution: divide annual tonnes evenly across months
    for strategic_period_id, annual_tonnes in strategic_tonnes.items():
        # Find tactical periods that fall within this strategic period
        # (simplified - would need actual date matching)
        tactical_periods_in_year = [p for p in config.periods if p.id.startswith(strategic_period_id[:2])]
        
        if not tactical_periods_in_year:
            # Fallback: use all tactical periods
            tactical_periods_in_year = config.periods
        
        tonnes_per_tactical_period = annual_tonnes / len(tactical_periods_in_year)
        
        for tactical_period in tactical_periods_in_year:
            # Distribute across pushbacks (simplified)
            pushback_ids = ["PB1", "PB2", "PB3"]  # Would get from pit_phases
            tonnes_per_pushback = tonnes_per_tactical_period / len(pushback_ids)
            
            for pushback_id in pushback_ids[:config.max_active_pushbacks]:
                decisions.append(ScheduleDecision(
                    period_id=tactical_period.id,
                    unit_id=pushback_id,
                    tonnes=tonnes_per_pushback,
                    destination="plant"
                ))
    
    logger.info(f"Derived tactical schedule: {len(decisions)} decisions across {len(config.periods)} periods")
    
    return ScheduleResult(
        periods=config.periods,
        decisions=decisions,
        metadata={
            "method": "pushback_distribution",
            "source": "strategic_schedule"
        }
    )

