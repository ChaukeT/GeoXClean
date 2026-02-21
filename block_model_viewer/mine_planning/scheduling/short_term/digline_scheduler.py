"""
Short-Term Digline Scheduler (STEP 30)

Create weekly/daily schedule based on GC model, diglines, and fleet capacity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..types import TimePeriod, ScheduleDecision, ScheduleResult

logger = logging.getLogger(__name__)


@dataclass
class ShortTermScheduleConfig:
    """
    Configuration for short-term digline scheduling.
    
    Attributes:
        periods: List of TimePeriod (days/weeks)
        fleet_config: FleetConfig reference
        routes: List of Route references
        gc_model_ref: GC model instance reference
        diglines_ref: DiglineSet instance reference
        plant_daily_target_t: Plant daily target (tonnes)
        grade_targets: Dictionary mapping element -> (min, max) grade range
    """
    periods: List[TimePeriod]
    fleet_config: Any  # FleetConfig
    routes: List[Any]  # List[Route]
    gc_model_ref: Any  # GCModel
    diglines_ref: Any  # DiglineSet
    plant_daily_target_t: float
    grade_targets: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def build_short_term_schedule(
    gc_model: Any,
    diglines: Any,
    config: ShortTermScheduleConfig
) -> ScheduleResult:
    """
    Build short-term schedule from GC model and diglines.
    
    Args:
        gc_model: GCModel instance
        diglines: DiglineSet instance
        config: ShortTermScheduleConfig
        
    Returns:
        ScheduleResult
    """
    # Get digline summaries
    # Note: Would need ore_waste_result from classify_gc_blocks first
    # Simplified for now - would call summarise_by_digpolygon with proper inputs
    digline_summaries = {}
    
    # If diglines have polygons, create placeholder summaries
    if hasattr(diglines, 'polygons') and diglines.polygons:
        for polygon in diglines.polygons:
            digline_summaries[polygon.id] = {
                "tonnes_ore": 0.0,
                "tonnes_waste": 0.0,
                "grade": {}
            }
    
    decisions = []
    
    # Allocate diglines to periods based on fleet capacity and targets
    remaining_target = config.plant_daily_target_t
    
    for period in config.periods:
        period_tonnes = 0.0
        
        # Select diglines for this period
        for polygon in diglines.polygons:
            if period_tonnes >= remaining_target:
                break
            
            # Get tonnes available from this digline
            summary = digline_summaries.get(polygon.id, {})
            available_tonnes = summary.get("tonnes_ore", 0.0)
            
            if available_tonnes > 0:
                tonnes_to_schedule = min(available_tonnes, remaining_target - period_tonnes)
                
                decisions.append(ScheduleDecision(
                    period_id=period.id,
                    unit_id=polygon.id,
                    tonnes=tonnes_to_schedule,
                    destination="plant"
                ))
                
                period_tonnes += tonnes_to_schedule
        
        remaining_target = config.plant_daily_target_t  # Reset for next period
    
    logger.info(f"Built short-term schedule: {len(decisions)} decisions across {len(config.periods)} periods")
    
    return ScheduleResult(
        periods=config.periods,
        decisions=decisions,
        metadata={
            "method": "digline_scheduler",
            "plant_target": config.plant_daily_target_t
        }
    )

