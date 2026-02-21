"""
Nested Shell Scheduler (STEP 30)

Map nested shells/phases to periods with simple heuristics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

from ..types import TimePeriod, ScheduleDecision, ScheduleResult

logger = logging.getLogger(__name__)


@dataclass
class NestedShellScheduleConfig:
    """
    Configuration for nested shell scheduling.
    
    Attributes:
        shell_ids: List of shell IDs
        target_years: Target number of years
        tonnes_per_year_target: Target tonnes per year
        allow_phase_overlap: Whether to allow overlapping phases
    """
    shell_ids: List[str]
    target_years: int
    tonnes_per_year_target: float
    allow_phase_overlap: bool = False


def allocate_shells_to_periods(
    shell_tonnage: Dict[str, float],
    config: NestedShellScheduleConfig
) -> ScheduleResult:
    """
    Allocate nested shells to periods using simple heuristics.
    
    Args:
        shell_tonnage: Dictionary mapping shell_id -> tonnes
        config: NestedShellScheduleConfig
        
    Returns:
        ScheduleResult
    """
    # Create periods
    periods = []
    for year in range(config.target_years):
        period = TimePeriod(
            id=f"Y{year+1:02d}",
            index=year,
            duration_days=365.0
        )
        periods.append(period)
    
    # Simple allocation: assign shells sequentially to years
    decisions = []
    remaining_capacity = config.tonnes_per_year_target
    
    current_period_idx = 0
    current_period = periods[current_period_idx] if periods else None
    
    for shell_id in config.shell_ids:
        shell_tonnes = shell_tonnage.get(shell_id, 0.0)
        
        if shell_tonnes <= 0:
            continue
        
        # Allocate shell to current period
        if current_period and remaining_capacity >= shell_tonnes:
            decisions.append(ScheduleDecision(
                period_id=current_period.id,
                unit_id=shell_id,
                tonnes=shell_tonnes,
                destination="plant"
            ))
            remaining_capacity -= shell_tonnes
        else:
            # Move to next period
            if current_period_idx < len(periods) - 1:
                current_period_idx += 1
                current_period = periods[current_period_idx]
                remaining_capacity = config.tonnes_per_year_target
                
                decisions.append(ScheduleDecision(
                    period_id=current_period.id,
                    unit_id=shell_id,
                    tonnes=min(shell_tonnes, remaining_capacity),
                    destination="plant"
                ))
                remaining_capacity -= min(shell_tonnes, remaining_capacity)
    
    logger.info(f"Allocated {len(config.shell_ids)} shells to {len(periods)} periods")
    
    return ScheduleResult(
        periods=periods,
        decisions=decisions,
        metadata={
            "method": "nested_shell_heuristic",
            "total_shells": len(config.shell_ids)
        }
    )

