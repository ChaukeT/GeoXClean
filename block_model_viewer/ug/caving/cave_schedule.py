"""
Cave Schedule Simulator (STEP 37)

Simulate cave draw over time.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

from .cave_footprint import CaveFootprint
from ...mine_planning.scheduling.types import TimePeriod, ScheduleDecision, ScheduleResult

logger = logging.getLogger(__name__)


@dataclass
class CaveDrawRule:
    """
    Rules for cave draw simulation.
    
    Attributes:
        max_draw_rate_tpy: Maximum draw rate tonnes per year
        max_draw_height_m: Maximum draw height in meters
        dilution_entry_height_ratio: Height ratio when dilution enters
        secondary_break_fraction: Fraction requiring secondary breakage
    """
    max_draw_rate_tpy: float = 10_000_000.0
    max_draw_height_m: float = 100.0
    dilution_entry_height_ratio: float = 0.7
    secondary_break_fraction: float = 0.1


@dataclass
class CaveScheduleConfig:
    """
    Configuration for cave draw simulation.
    
    Attributes:
        periods: List of TimePeriod objects
        rule: CaveDrawRule
        target_tonnes_per_period: Target tonnes per period
    """
    periods: List[TimePeriod]
    rule: CaveDrawRule
    target_tonnes_per_period: float = 100_000.0


def simulate_cave_draw(
    footprint: CaveFootprint,
    config: CaveScheduleConfig
) -> ScheduleResult:
    """
    Simulate cave draw over time.
    
    Args:
        footprint: CaveFootprint
        config: CaveScheduleConfig
    
    Returns:
        ScheduleResult
    """
    logger.info(f"Simulating cave draw for footprint with {len(footprint.cells)} cells")
    
    if not footprint.cells:
        logger.warning("Footprint has no cells")
        return ScheduleResult(
            periods=config.periods,
            decisions=[],
            metadata={'error': 'No cells in footprint'}
        )
    
    if not config.periods:
        logger.warning("No periods provided")
        return ScheduleResult(
            periods=[],
            decisions=[],
            metadata={'error': 'No periods provided'}
        )
    
    # Group cells by level
    cells_by_level = {}
    for cell in footprint.cells:
        level_key = f"{cell.level:.1f}"
        if level_key not in cells_by_level:
            cells_by_level[level_key] = []
        cells_by_level[level_key].append(cell)
    
    # Sort levels from top to bottom
    sorted_levels = sorted(cells_by_level.keys(), key=lambda x: float(x), reverse=True)
    
    # Simulate draw from top down
    decisions = []
    drawn_cells = set()
    current_level_idx = 0
    
    for period_idx, period in enumerate(config.periods):
        period_tonnes = 0.0
        
        # Calculate available draw capacity
        period_duration_years = period.duration_days / 365.0 if hasattr(period, 'duration_days') else 1.0
        max_draw_tonnes = config.rule.max_draw_rate_tpy * period_duration_years
        
        # Draw from current level
        while period_tonnes < config.target_tonnes_per_period and current_level_idx < len(sorted_levels):
            level_key = sorted_levels[current_level_idx]
            level_cells = cells_by_level[level_key]
            
            # Draw from cells in this level
            for cell in level_cells:
                if cell.id in drawn_cells:
                    continue
                
                if period_tonnes + cell.tonnage > max_draw_tonnes:
                    break
                
                if period_tonnes + cell.tonnage > config.target_tonnes_per_period * 1.1:
                    break
                
                drawn_cells.add(cell.id)
                period_tonnes += cell.tonnage
                
                # Create decision
                decision = ScheduleDecision(
                    period_id=period.id,
                    unit_id=cell.id,
                    tonnes=cell.tonnage,
                    destination="plant"
                )
                decisions.append(decision)
            
            # Move to next level if current level is exhausted
            if all(c.id in drawn_cells for c in level_cells):
                current_level_idx += 1
            else:
                break
        
        logger.debug(f"Period {period.id}: Drew {period_tonnes:,.0f} tonnes from {len([c for c in drawn_cells if c in [cell.id for cell in footprint.cells]])} cells")
    
    total_tonnes = sum(d.tonnes for d in decisions)
    
    metadata = {
        'total_cells': len(footprint.cells),
        'drawn_cells': len(drawn_cells),
        'total_tonnes': total_tonnes,
        'algorithm': 'top_down_draw'
    }
    
    logger.info(f"Cave draw simulation complete: {total_tonnes:,.0f} tonnes over {len(config.periods)} periods")
    
    return ScheduleResult(
        periods=config.periods,
        decisions=decisions,
        metadata=metadata
    )

