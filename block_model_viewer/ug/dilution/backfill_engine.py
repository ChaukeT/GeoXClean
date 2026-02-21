"""
Backfill Engine (STEP 37)

Calculate backfill requirements and track voids.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

from ..slos.slos_geometry import StopeInstance
from ...mine_planning.scheduling.types import TimePeriod

logger = logging.getLogger(__name__)


@dataclass
class BackfillRecipe:
    """
    Backfill recipe specification.
    
    Attributes:
        id: Recipe identifier
        density_t_m3: Density in tonnes per cubic meter
        cement_content_pct: Cement content percentage
        cost_per_m3: Cost per cubic meter
    """
    id: str
    density_t_m3: float = 1.8
    cement_content_pct: float = 5.0
    cost_per_m3: float = 50.0


@dataclass
class BackfillResult:
    """
    Result from backfill calculation.
    
    Attributes:
        stope_id: Stope identifier
        volume_m3: Backfill volume in cubic meters
        cost: Total backfill cost
    """
    stope_id: str
    volume_m3: float
    cost: float


def compute_backfill_for_stope(
    stope: StopeInstance,
    recipe: BackfillRecipe
) -> BackfillResult:
    """
    Compute backfill requirements for a stope.
    
    Args:
        stope: StopeInstance
        recipe: BackfillRecipe
    
    Returns:
        BackfillResult
    """
    logger.debug(f"Computing backfill for stope {stope.id}")
    
    # Calculate stope volume (simplified)
    # Assume density 2.7 t/m³ for ore
    stope_volume_m3 = stope.tonnes / 2.7
    
    # Backfill volume equals stope volume (simplified)
    # In practice, would account for shrinkage, voids, etc.
    backfill_volume_m3 = stope_volume_m3
    
    # Calculate cost
    cost = backfill_volume_m3 * recipe.cost_per_m3
    
    logger.debug(f"Stope {stope.id}: {backfill_volume_m3:.0f} m³ backfill, cost ${cost:,.0f}")
    
    return BackfillResult(
        stope_id=stope.id,
        volume_m3=backfill_volume_m3,
        cost=cost
    )


def accumulate_void_time_series(
    stopes: List[StopeInstance],
    backfill_results: List[BackfillResult],
    periods: List[TimePeriod],
    schedule_decisions: Optional[List[Any]] = None
) -> Dict[str, List[float]]:
    """
    Accumulate void and backfill volumes over time.
    
    Args:
        stopes: List of StopeInstance objects
        backfill_results: List of BackfillResult objects
        periods: List of TimePeriod objects
        schedule_decisions: Optional list of ScheduleDecision objects
    
    Returns:
        Dictionary with 'void_volume_m3', 'backfilled_volume_m3', 'unfilled_volume_m3' per period
    """
    logger.info("Accumulating void time series")
    
    # Create backfill lookup
    backfill_by_stope = {bf.stope_id: bf for bf in backfill_results}
    
    # Create stope lookup
    stope_by_id = {s.id: s for s in stopes}
    
    # Initialize time series
    void_volume_m3 = []
    backfilled_volume_m3 = []
    unfilled_volume_m3 = []
    
    cumulative_void = 0.0
    cumulative_backfilled = 0.0
    
    # Track which stopes have been mined and backfilled
    mined_stopes = set()
    backfilled_stopes = set()
    
    # If schedule decisions provided, use them to determine when stopes are mined
    if schedule_decisions:
        for period in periods:
            period_decisions = [d for d in schedule_decisions if d.period_id == period.id]
            
            # Add newly mined stopes
            for decision in period_decisions:
                if decision.unit_id in stope_by_id:
                    mined_stopes.add(decision.unit_id)
            
            # Calculate void volume (mined but not backfilled)
            period_void = 0.0
            period_backfilled = 0.0
            
            for stope_id in mined_stopes:
                stope = stope_by_id[stope_id]
                stope_volume = stope.tonnes / 2.7  # Assume density
                
                if stope_id in backfilled_stopes:
                    period_backfilled += stope_volume
                else:
                    # Assume backfill happens 1 period after mining (simplified)
                    if stope_id in backfill_by_stope:
                        backfilled_stopes.add(stope_id)
                        period_backfilled += stope_volume
                    else:
                        period_void += stope_volume
            
            cumulative_void += period_void
            cumulative_backfilled += period_backfilled
            
            void_volume_m3.append(cumulative_void)
            backfilled_volume_m3.append(cumulative_backfilled)
            unfilled_volume_m3.append(cumulative_void - cumulative_backfilled)
    else:
        # Simplified: assume all stopes mined in first period, backfilled gradually
        total_void = sum(s.tonnes / 2.7 for s in stopes)
        
        for period_idx, period in enumerate(periods):
            # Gradually backfill over time
            backfill_rate = 0.2  # 20% per period
            backfilled_fraction = min(1.0, (period_idx + 1) * backfill_rate)
            
            backfilled = total_void * backfilled_fraction
            unfilled = total_void - backfilled
            
            void_volume_m3.append(total_void)
            backfilled_volume_m3.append(backfilled)
            unfilled_volume_m3.append(unfilled)
    
    return {
        'void_volume_m3': void_volume_m3,
        'backfilled_volume_m3': backfilled_volume_m3,
        'unfilled_volume_m3': unfilled_volume_m3
    }

