"""
Core dataclasses for ESG Module

Defines standard contracts for emissions, water, waste, and reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class EmissionSource(Enum):
    """Categories of emission sources."""
    DIESEL = "diesel"
    ELECTRICITY = "electricity"
    EXPLOSIVES = "explosives"
    CEMENT = "cement"
    LIME = "lime"
    REAGENTS = "reagents"
    OTHER = "other"


@dataclass
class EmissionFactor:
    """
    Emission factor for CO2e calculation.
    
    Attributes:
        source: Emission source category
        fuel_or_energy: Specific fuel/energy type (e.g., "Diesel", "Grid Electricity")
        unit: Unit of consumption (L, MWh, kg, t)
        ef_kgco2e_per_unit: Emission factor (kg CO2e per unit)
        scope: Scope 1, 2, or 3
    """
    source: EmissionSource
    fuel_or_energy: str
    unit: str
    ef_kgco2e_per_unit: float
    scope: int = 1
    
    def calc_emissions(self, activity: float) -> float:
        """Calculate CO2e emissions for given activity level."""
        return activity * self.ef_kgco2e_per_unit


@dataclass
class WaterNode:
    """
    Water balance node (pond, tank, process).
    
    Attributes:
        node_id: Unique identifier
        node_type: Type (pond, tank, process, tailings)
        volume_m3: Current volume (m3)
        capacity_m3: Maximum capacity (m3)
        area_m2: Surface area for evaporation
        evap_coeff_mm_per_day: Evaporation rate
        permit_freeboard_m: Required freeboard (regulatory)
        current_freeboard_m: Actual freeboard
        salinity_ppm: Total dissolved solids
    """
    node_id: str
    node_type: str  # pond, tank, process, tailings
    volume_m3: float = 0.0
    capacity_m3: float = 0.0
    area_m2: float = 0.0
    evap_coeff_mm_per_day: float = 5.0
    permit_freeboard_m: float = 1.0
    current_freeboard_m: float = 0.0
    salinity_ppm: float = 0.0
    
    def is_freeboard_compliant(self) -> bool:
        """Check if freeboard meets regulatory requirement."""
        return self.current_freeboard_m >= self.permit_freeboard_m
    
    def evaporation_m3_per_day(self) -> float:
        """Calculate daily evaporation."""
        return self.area_m2 * (self.evap_coeff_mm_per_day / 1000.0)


@dataclass
class ESGReport:
    """
    Consolidated ESG report for a period or project.
    
    Attributes:
        period: Reporting period
        total_co2e_t: Total CO2 equivalent (tonnes)
        co2e_per_t_ore: CO2e intensity (kg/t ore)
        co2e_per_t_product: CO2e intensity (kg/t concentrate)
        co2e_by_source: Breakdown by emission source
        water_consumed_m3: Total water consumed
        water_per_t_ore: Water intensity (m3/t ore)
        water_recycled_pct: Percentage of water recycled
        tailings_t: Tailings produced (tonnes)
        waste_rock_t: Waste rock produced (tonnes)
        disturbed_area_ha: Land disturbed (hectares)
        rehab_area_ha: Land rehabilitated (hectares)
        compliance_flags: Dictionary of compliance checks
    """
    period: int
    total_co2e_t: float = 0.0
    co2e_per_t_ore: float = 0.0
    co2e_per_t_product: float = 0.0
    co2e_by_source: Dict[str, float] = field(default_factory=dict)
    water_consumed_m3: float = 0.0
    water_per_t_ore: float = 0.0
    water_recycled_pct: float = 0.0
    tailings_t: float = 0.0
    waste_rock_t: float = 0.0
    disturbed_area_ha: float = 0.0
    rehab_area_ha: float = 0.0
    compliance_flags: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        result = {
            'period': self.period,
            'total_co2e_t': self.total_co2e_t,
            'co2e_per_t_ore': self.co2e_per_t_ore,
            'co2e_per_t_product': self.co2e_per_t_product,
            'water_consumed_m3': self.water_consumed_m3,
            'water_per_t_ore': self.water_per_t_ore,
            'water_recycled_pct': self.water_recycled_pct,
            'tailings_t': self.tailings_t,
            'waste_rock_t': self.waste_rock_t,
            'disturbed_area_ha': self.disturbed_area_ha,
            'rehab_area_ha': self.rehab_area_ha
        }
        
        # Add CO2e breakdown
        result.update({f'co2e_{k}': v for k, v in self.co2e_by_source.items()})
        
        # Add compliance flags
        result.update({f'compliant_{k}': v for k, v in self.compliance_flags.items()})
        
        return result
