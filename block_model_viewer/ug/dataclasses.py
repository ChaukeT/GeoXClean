"""
Core dataclasses for Underground Mining Module

Defines standard contracts for stopes, periods, and capacities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class Stope:
    """
    Represents an optimized underground stope.
    
    Attributes:
        id: Unique stope identifier
        level: Mining level (elevation or index)
        voxels: List of block IDs comprising this stope
        tonnes_raw: Ore tonnage before dilution
        grade_raw: Grade dictionary before dilution {"Fe": 0.62, ...}
        tonnes_dil: Ore tonnage after dilution
        grade_dil: Grade dictionary after dilution
        nsr_dil: Net smelter return per tonne (after dilution)
        geom: Geometry parameters (length, width, height, pillars)
        parents: List of parent stope IDs (precedence)
        risk_score: Ground control risk index (probability × consequence)
    """
    id: str
    level: int
    voxels: List[int]
    tonnes_raw: float
    grade_raw: Dict[str, float]
    tonnes_dil: float
    grade_dil: Dict[str, float]
    nsr_dil: float
    geom: Dict[str, float] = field(default_factory=dict)  # length, width, height, crown_pillar, rib_pillar
    parents: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            'stope_id': self.id,
            'level': self.level,
            'voxels': len(self.voxels),
            'tonnes_raw': self.tonnes_raw,
            'tonnes_dil': self.tonnes_dil,
            'nsr_dil': self.nsr_dil,
            'length': self.geom.get('length', 0),
            'width': self.geom.get('width', 0),
            'height': self.geom.get('height', 0),
            'crown_pillar': self.geom.get('crown_pillar', 0),
            'rib_pillar': self.geom.get('rib_pillar', 0),
            'risk_score': self.risk_score,
            **{f'grade_raw_{k}': v for k, v in self.grade_raw.items()},
            **{f'grade_dil_{k}': v for k, v in self.grade_dil.items()},
            'parents': '|'.join(self.parents) if self.parents else ''
        }


@dataclass
class PeriodKPI:
    """
    Key performance indicators for a mining period.
    
    Used for cut-and-fill scheduling results and ESG tracking.
    
    Attributes:
        t: Period number (e.g., month 1, 2, 3...)
        ore_mined: Tonnes of ore extracted
        ore_proc: Tonnes of ore processed/milled
        waste: Tonnes of waste/development material
        head_grade: Grade to mill {"Fe": 0.60, ...}
        metal: Metal produced/recovered {"Fe": 45000, ...}
        stock_open: Opening stockpile tonnes
        stack: Tonnes added to stockpile
        reclaim: Tonnes reclaimed from stockpile
        stock_close: Closing stockpile tonnes
        cashflow: Period cashflow (USD)
        dcf: Discounted cashflow (NPV contribution)
        co2e_t: CO2 equivalent emissions (tonnes)
        water_m3: Water consumed (cubic meters)
        tailings_t: Tailings produced (tonnes)
        fill_placed_t: Backfill placed (tonnes)
        energy_mwh: Energy consumed (MWh)
    """
    t: int
    ore_mined: float = 0.0
    ore_proc: float = 0.0
    waste: float = 0.0
    head_grade: Dict[str, float] = field(default_factory=dict)
    metal: Dict[str, float] = field(default_factory=dict)
    stock_open: float = 0.0
    stack: float = 0.0
    reclaim: float = 0.0
    stock_close: float = 0.0
    cashflow: float = 0.0
    dcf: float = 0.0
    co2e_t: float = 0.0
    water_m3: float = 0.0
    tailings_t: float = 0.0
    fill_placed_t: float = 0.0
    energy_mwh: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame/CSV export."""
        result = {
            'period': self.t,
            'ore_mined': self.ore_mined,
            'ore_proc': self.ore_proc,
            'waste': self.waste,
            'stock_open': self.stock_open,
            'stack': self.stack,
            'reclaim': self.reclaim,
            'stock_close': self.stock_close,
            'cashflow': self.cashflow,
            'dcf': self.dcf,
            'co2e_t': self.co2e_t,
            'water_m3': self.water_m3,
            'tailings_t': self.tailings_t,
            'fill_placed_t': self.fill_placed_t,
            'energy_mwh': self.energy_mwh
        }
        
        # Add grade fields
        result.update({f'grade_{k}': v for k, v in self.head_grade.items()})
        
        # Add metal fields
        result.update({f'metal_{k}': v for k, v in self.metal.items()})
        
        return result


@dataclass
class UGCapacities:
    """
    Underground mine capacities and constraints by period.
    
    Attributes:
        period: Period number
        mine_cap_t: Mining capacity (tonnes/period)
        mill_cap_t: Milling capacity (tonnes/period)
        fill_cap_t: Backfill capacity (tonnes/period)
        vent_power_mw: Available ventilation power (MW)
        discount_factor: Discount factor for NPV (1/(1+r)^t)
    """
    period: int
    mine_cap_t: float
    mill_cap_t: float
    fill_cap_t: float
    vent_power_mw: float = 0.0
    discount_factor: float = 1.0
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List['UGCapacities']:
        """Create list of UGCapacities from DataFrame."""
        capacities = []
        for _, row in df.iterrows():
            capacities.append(cls(
                period=int(row.get('period', row.name)),
                mine_cap_t=float(row['mine_cap_t']),
                mill_cap_t=float(row['mill_cap_t']),
                fill_cap_t=float(row['fill_cap_t']),
                vent_power_mw=float(row.get('vent_power_mw', 0)),
                discount_factor=float(row.get('discount_factor', 1.0))
            ))
        return capacities
