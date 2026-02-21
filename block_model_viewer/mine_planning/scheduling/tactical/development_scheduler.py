"""
Development Scheduler (STEP 30)

Schedule development meters (drifts, raises, crosscuts) to meet tactical production targets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

from ..types import TimePeriod, ScheduleDecision, ScheduleResult

logger = logging.getLogger(__name__)


@dataclass
class DevelopmentTask:
    """
    Development task definition.
    
    Attributes:
        id: Task identifier
        type: Task type ("drift", "crosscut", "raise")
        length_m: Length in meters
        predecessor_ids: List of predecessor task IDs
        location: Location dictionary (level, x, y, z)
    """
    id: str
    type: str
    length_m: float
    predecessor_ids: List[str] = field(default_factory=list)
    location: Dict[str, float] = field(default_factory=dict)


@dataclass
class DevelopmentScheduleConfig:
    """
    Configuration for development scheduling.
    
    Attributes:
        periods: List of TimePeriod
        development_rate_m_per_period: Development rate (meters per period)
        max_concurrent_faces: Maximum concurrent development faces
    """
    periods: List[TimePeriod]
    development_rate_m_per_period: float
    max_concurrent_faces: int = 3


def schedule_development(
    tasks: List[DevelopmentTask],
    config: DevelopmentScheduleConfig
) -> ScheduleResult:
    """
    Schedule development tasks.
    
    Args:
        tasks: List of DevelopmentTask
        config: DevelopmentScheduleConfig
        
    Returns:
        ScheduleResult with development decisions
    """
    decisions = []
    
    # Simple scheduling: assign tasks sequentially respecting precedence
    scheduled_tasks = set()
    current_period_idx = 0
    current_period_meters = 0.0
    
    # Build precedence graph
    predecessors_map = {task.id: set(task.predecessor_ids) for task in tasks}
    
    # Schedule tasks
    while len(scheduled_tasks) < len(tasks):
        # Find tasks ready to schedule (all predecessors done)
        ready_tasks = [
            task for task in tasks
            if task.id not in scheduled_tasks
            and all(pred_id in scheduled_tasks for pred_id in predecessors_map.get(task.id, set()))
        ]
        
        if not ready_tasks:
            # No ready tasks - schedule remaining tasks ignoring precedence
            ready_tasks = [task for task in tasks if task.id not in scheduled_tasks]
        
        if not ready_tasks:
            break
        
        # Schedule ready tasks up to capacity
        for task in ready_tasks[:config.max_concurrent_faces]:
            if current_period_meters + task.length_m > config.development_rate_m_per_period:
                # Move to next period
                current_period_idx += 1
                if current_period_idx >= len(config.periods):
                    break
                current_period_meters = 0.0
            
            if current_period_idx >= len(config.periods):
                break
            
            period = config.periods[current_period_idx]
            decisions.append(ScheduleDecision(
                period_id=period.id,
                unit_id=task.id,
                tonnes=task.length_m,  # Using tonnes field for meters
                destination=task.type
            ))
            
            scheduled_tasks.add(task.id)
            current_period_meters += task.length_m
    
    logger.info(f"Scheduled {len(decisions)} development tasks across {len(config.periods)} periods")
    
    return ScheduleResult(
        periods=config.periods,
        decisions=decisions,
        metadata={
            "method": "development_sequential",
            "total_tasks": len(tasks),
            "total_meters": sum(task.length_m for task in tasks)
        }
    )

