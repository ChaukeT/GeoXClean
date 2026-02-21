"""
NPVS Data Structures (STEP 32)

Data classes for NPVS optimization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class NpvsBlock:
    """
    Block representation for NPVS optimization.
    
    Attributes:
        id: Block identifier
        tonnage: Block tonnage
        bench: Bench identifier (optional)
        pit: Pit/phase identifier (optional)
        phase: Phase identifier (optional)
        ore_type: Ore type code (optional)
        grade_by_element: Dictionary mapping element name -> grade value
        value_raw: Raw block value
        precedence_parents: List of block IDs that must be mined before this block
    """
    id: int
    tonnage: float
    bench: Optional[str] = None
    pit: Optional[str] = None
    phase: Optional[str] = None
    ore_type: Optional[str] = None
    grade_by_element: Dict[str, float] = None
    value_raw: float = 0.0
    precedence_parents: List[int] = None
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.grade_by_element is None:
            self.grade_by_element = {}
        if self.precedence_parents is None:
            self.precedence_parents = []


@dataclass
class NpvsConfig:
    """
    Configuration for NPVS optimization.
    
    Attributes:
        periods: List of period dictionaries with id, index, duration_years, discount_factor
        destinations: List of destination dictionaries with id, type, capacity_tpy, recovery_by_element, processing_cost_per_t
        discount_rate: Discount rate
        mining_capacity_tpy: Mining capacity tonnes per year
        plant_capacity_tpy: Plant capacity tonnes per year
        stockpile_capacity_t: Stockpile capacity by destination (optional)
        max_phase_rate_tpy: Maximum phase rate tonnes per year (optional)
        min_phase_rate_tpy: Minimum phase rate tonnes per year (optional)
        penalty_unscheduled_factor: Penalty factor for unscheduled blocks (optional)
    """
    periods: List[Dict[str, Any]]
    destinations: List[Dict[str, Any]]
    discount_rate: float
    mining_capacity_tpy: float
    plant_capacity_tpy: float
    stockpile_capacity_t: Dict[str, float] = None
    max_phase_rate_tpy: Dict[str, float] = None
    min_phase_rate_tpy: Dict[str, float] = None
    penalty_unscheduled_factor: float = 0.0
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.stockpile_capacity_t is None:
            self.stockpile_capacity_t = {}
        if self.max_phase_rate_tpy is None:
            self.max_phase_rate_tpy = {}
        if self.min_phase_rate_tpy is None:
            self.min_phase_rate_tpy = {}

