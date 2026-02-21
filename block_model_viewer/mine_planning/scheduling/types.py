"""
Shared Scheduling Types (STEP 30)

Common dataclasses for all scheduling layers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import date
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimePeriod:
    """
    Represents a time period in the schedule.
    
    Attributes:
        id: Period identifier (e.g., "Y01", "M01_2027", "W_2027_03")
        index: Period index (0-based)
        start_date: Start date (optional)
        end_date: End date (optional)
        duration_days: Duration in days
    """
    id: str
    index: int
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    duration_days: float = 365.0
    
    def __post_init__(self):
        """Validate period."""
        if self.duration_days <= 0:
            raise ValueError("duration_days must be positive")


@dataclass
class ScheduleBlock:
    """
    Represents a block in the schedule.
    
    Attributes:
        block_id: Block identifier
        bench: Bench identifier (optional)
        pit: Pit/phase identifier (optional)
        stope_id: Stope identifier for UG (optional)
        ore_type: Ore type code (optional)
        tonnes: Tonnes
        grade_by_element: Dictionary mapping element name -> grade value
        value: Block value
        geomet_value: Geomet-adjusted value (optional)
    """
    block_id: int
    bench: Optional[str] = None
    pit: Optional[str] = None
    stope_id: Optional[str] = None
    ore_type: Optional[str] = None
    tonnes: float = 0.0
    grade_by_element: Dict[str, float] = field(default_factory=dict)
    value: float = 0.0
    geomet_value: Optional[float] = None
    
    def __post_init__(self):
        """Validate block."""
        if self.tonnes < 0:
            raise ValueError("tonnes must be non-negative")


@dataclass
class ScheduleDecision:
    """
    Represents a scheduling decision (what to mine where and when).
    
    Attributes:
        period_id: Period identifier
        unit_id: Unit identifier (pit bench, stope, digline, etc.)
        tonnes: Tonnes to mine/extract
        destination: Destination (plant, stockpile_x, waste_dump_y)
    """
    period_id: str
    unit_id: str
    tonnes: float
    destination: str = "plant"
    
    def __post_init__(self):
        """Validate decision."""
        if self.tonnes < 0:
            raise ValueError("tonnes must be non-negative")


@dataclass
class ScheduleResult:
    """
    Result from scheduling optimization.
    
    Attributes:
        periods: List of TimePeriod
        decisions: List of ScheduleDecision
        metadata: Additional metadata (NPV, objective value, etc.)
    """
    periods: List[TimePeriod] = field(default_factory=list)
    decisions: List[ScheduleDecision] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_tonnes_by_period(self) -> Dict[str, float]:
        """Get total tonnes per period."""
        tonnes_by_period = {}
        for decision in self.decisions:
            tonnes_by_period[decision.period_id] = tonnes_by_period.get(decision.period_id, 0.0) + decision.tonnes
        return tonnes_by_period
    
    def get_decisions_for_period(self, period_id: str) -> List[ScheduleDecision]:
        """Get all decisions for a specific period."""
        return [d for d in self.decisions if d.period_id == period_id]
    
    def get_total_tonnes(self) -> float:
        """Get total tonnes across all periods."""
        return sum(d.tonnes for d in self.decisions)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ScheduleResult to dictionary for serialization."""
        return {
            "periods": [
                {
                    "id": p.id,
                    "index": p.index,
                    "start_date": p.start_date.isoformat() if p.start_date else None,
                    "end_date": p.end_date.isoformat() if p.end_date else None,
                    "duration_days": p.duration_days
                }
                for p in self.periods
            ],
            "decisions": [
                {
                    "period_id": d.period_id,
                    "unit_id": d.unit_id,
                    "tonnes": d.tonnes,
                    "destination": d.destination
                }
                for d in self.decisions
            ],
            "metadata": self.metadata
        }

