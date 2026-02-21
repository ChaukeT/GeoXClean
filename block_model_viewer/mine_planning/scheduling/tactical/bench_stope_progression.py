"""
Bench/Stope Progression Scheduler (STEP 30)

Translate pushback/stope schedule into bench-by-bench or level-by-level advancement.
"""

from typing import Any, Optional
import logging

from ..types import ScheduleResult

logger = logging.getLogger(__name__)


def build_bench_schedule_from_pushbacks(
    block_model: Any,
    pushback_schedule: ScheduleResult,
    bench_height: float
) -> ScheduleResult:
    """
    Build bench schedule from pushback schedule.
    
    Args:
        block_model: BlockModel instance
        pushback_schedule: Pushback ScheduleResult
        bench_height: Bench height
        
    Returns:
        ScheduleResult with bench-level decisions
    """
    # Convert pushback decisions to bench decisions
    decisions = []
    
    for decision in pushback_schedule.decisions:
        # Map pushback to benches (simplified)
        bench_ids = [f"{decision.unit_id}_B{i}" for i in range(1, 6)]  # Would get from block_model
        
        tonnes_per_bench = decision.tonnes / len(bench_ids)
        
        for bench_id in bench_ids:
            decisions.append(type(decision)(
                period_id=decision.period_id,
                unit_id=bench_id,
                tonnes=tonnes_per_bench,
                destination=decision.destination
            ))
    
    logger.info(f"Built bench schedule: {len(decisions)} bench decisions")
    
    return ScheduleResult(
        periods=pushback_schedule.periods,
        decisions=decisions,
        metadata={
            "method": "bench_progression",
            "bench_height": bench_height,
            "source": "pushback_schedule"
        }
    )


def build_stope_schedule_from_ug_phases(
    stope_groups: Any,
    strategic_schedule: ScheduleResult
) -> ScheduleResult:
    """
    Build stope schedule from UG phases.
    
    Args:
        stope_groups: Stope groups data
        strategic_schedule: Strategic ScheduleResult
        
    Returns:
        ScheduleResult with stope-level decisions
    """
    # Convert strategic decisions to stope decisions
    decisions = []
    
    for decision in strategic_schedule.decisions:
        # Map phase to stopes (simplified)
        stope_ids = [f"STOPE_{i}" for i in range(1, 10)]  # Would get from stope_groups
        
        tonnes_per_stope = decision.tonnes / len(stope_ids)
        
        for stope_id in stope_ids:
            decisions.append(type(decision)(
                period_id=decision.period_id,
                unit_id=stope_id,
                tonnes=tonnes_per_stope,
                destination=decision.destination
            ))
    
    logger.info(f"Built stope schedule: {len(decisions)} stope decisions")
    
    return ScheduleResult(
        periods=strategic_schedule.periods,
        decisions=decisions,
        metadata={
            "method": "stope_progression",
            "source": "strategic_schedule"
        }
    )

