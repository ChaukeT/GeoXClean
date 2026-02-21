"""
Waste & Land Management Module

Tracks waste rock generation, land disturbance, and rehabilitation progress.
Supports ESG reporting and mine closure planning.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WasteType(Enum):
    """Types of waste rock"""
    INERT = "inert"  # Non-acid generating
    PAG = "pag"  # Potentially acid generating
    NAG = "nag"  # Non-acid generating (tested)
    CONTAMINATED = "contaminated"


class LandUseType(Enum):
    """Land use classifications"""
    UNDISTURBED = "undisturbed"
    MINING = "mining"
    WASTE_DUMP = "waste_dump"
    TAILINGS = "tailings"
    INFRASTRUCTURE = "infrastructure"
    REHABILITATED = "rehabilitated"
    CLOSURE_READY = "closure_ready"


class RehabilitationStage(Enum):
    """Stages of rehabilitation"""
    PLANNED = "planned"
    EARTHWORKS = "earthworks"
    SEEDED = "seeded"
    ESTABLISHED = "established"
    MATURE = "mature"
    MONITORING = "monitoring"


@dataclass
class WasteRockDump:
    """
    Represents a waste rock dump.
    
    Attributes:
        dump_id: Unique identifier
        waste_type: Type of waste rock
        volume: Total volume (m³)
        tonnage: Total tonnage (t)
        footprint_area: Area of footprint (m²)
        height: Maximum height (m)
        stability_factor: Factor of safety (>1.3 typically required)
        lined: Whether dump has liner
        drainage: Whether dump has drainage system
        monitoring: Whether geochemical monitoring active
    """
    dump_id: str
    waste_type: WasteType
    volume: float = 0.0  # m³
    tonnage: float = 0.0  # t
    footprint_area: float = 0.0  # m²
    height: float = 0.0  # m
    stability_factor: float = 1.5
    lined: bool = False
    drainage: bool = False
    monitoring: bool = False


@dataclass
class LandParcel:
    """
    Represents a parcel of land affected by mining.
    
    Attributes:
        parcel_id: Unique identifier
        area: Area in hectares
        land_use: Current land use
        disturbance_date: Period when disturbance began
        rehabilitation_stage: Current rehabilitation stage
        vegetation_cover: Vegetation cover percentage (0-100)
        soil_stability: Soil stability rating (0-10)
        biodiversity_score: Biodiversity score (0-100)
        target_closure_date: Target period for closure
    """
    parcel_id: str
    area: float  # hectares
    land_use: LandUseType
    disturbance_date: Optional[int] = None
    rehabilitation_stage: RehabilitationStage = RehabilitationStage.PLANNED
    vegetation_cover: float = 0.0  # %
    soil_stability: float = 0.0  # 0-10
    biodiversity_score: float = 0.0  # 0-100
    target_closure_date: Optional[int] = None


@dataclass
class WasteLandReport:
    """
    Report on waste and land management for a period.
    
    Attributes:
        period: Time period
        waste_generated_t: Waste rock generated (t)
        waste_by_type: Breakdown by waste type
        total_waste_volume: Total waste volume (m³)
        disturbed_area_ha: Total disturbed area (hectares)
        rehabilitated_area_ha: Area rehabilitated this period (hectares)
        cumulative_rehabilitated_ha: Cumulative rehabilitated area (hectares)
        vegetation_success_rate: % of rehab areas meeting vegetation targets
        biodiversity_index: Overall biodiversity index (0-100)
        compliance_issues: List of compliance issues identified
    """
    period: int
    waste_generated_t: float = 0.0
    waste_by_type: Dict[str, float] = field(default_factory=dict)
    total_waste_volume: float = 0.0
    disturbed_area_ha: float = 0.0
    rehabilitated_area_ha: float = 0.0
    cumulative_rehabilitated_ha: float = 0.0
    vegetation_success_rate: float = 0.0
    biodiversity_index: float = 0.0
    compliance_issues: List[str] = field(default_factory=list)


def track_waste_rock(
    mining_schedule: List[Dict],
    strip_ratio: float = 3.0,  # waste:ore ratio
    waste_density: float = 2.2,  # t/m³
    pag_percentage: float = 0.15  # % potentially acid generating
) -> List[WasteLandReport]:
    """
    Track waste rock generation over mining schedule.
    
    Args:
        mining_schedule: Mining schedule with ore tonnages per period
        strip_ratio: Waste to ore ratio
        waste_density: Waste rock density (t/m³)
        pag_percentage: Percentage of waste that is PAG
    
    Returns:
        List of WasteLandReports per period
    """
    logger.info(f"Tracking waste rock for {len(mining_schedule)} periods")
    
    reports = []
    
    for schedule_item in mining_schedule:
        period = schedule_item.get('period', 0)
        ore_mined = schedule_item.get('ore_mined', 0.0)
        
        # Calculate waste
        waste_generated = ore_mined * strip_ratio
        waste_volume = waste_generated / waste_density
        
        # Classify waste
        pag_waste = waste_generated * pag_percentage
        nag_waste = waste_generated * (1 - pag_percentage)
        
        report = WasteLandReport(
            period=period,
            waste_generated_t=waste_generated,
            waste_by_type={
                WasteType.PAG.value: pag_waste,
                WasteType.NAG.value: nag_waste
            },
            total_waste_volume=waste_volume
        )
        
        reports.append(report)
    
    logger.info(f"Waste tracking complete: {len(reports)} reports")
    return reports


def calculate_disturbance(
    mining_schedule: List[Dict],
    pit_area_per_kt: float = 0.1,  # ha per 1000t ore
    waste_dump_area_per_kt: float = 0.3,  # ha per 1000t waste
    strip_ratio: float = 3.0
) -> Dict[int, float]:
    """
    Calculate land disturbance by period.
    
    Args:
        mining_schedule: Mining schedule
        pit_area_per_kt: Pit area per 1000 tonnes ore (ha/kt)
        waste_dump_area_per_kt: Waste dump area per 1000 tonnes waste (ha/kt)
        strip_ratio: Waste to ore ratio
    
    Returns:
        Dict mapping period to disturbed area (ha)
    """
    disturbance = {}
    
    for schedule_item in mining_schedule:
        period = schedule_item.get('period', 0)
        ore_mined = schedule_item.get('ore_mined', 0.0)
        
        # Calculate areas
        pit_area = (ore_mined / 1000) * pit_area_per_kt
        waste_area = (ore_mined * strip_ratio / 1000) * waste_dump_area_per_kt
        
        disturbance[period] = pit_area + waste_area
    
    logger.info(f"Disturbance calculated for {len(disturbance)} periods")
    return disturbance


def plan_rehabilitation(
    land_parcels: List[LandParcel],
    rehab_capacity_ha_per_period: float = 10.0,
    min_age_for_rehab: int = 12  # periods (e.g., 1 year for monthly periods)
) -> List[Dict]:
    """
    Plan progressive rehabilitation schedule.
    
    Algorithm:
    1. Identify parcels ready for rehabilitation (disturbed > min_age, not yet rehabbed)
    2. Prioritize by age (oldest first)
    3. Allocate capacity
    
    Args:
        land_parcels: List of land parcels
        rehab_capacity_ha_per_period: Rehabilitation capacity per period (ha)
        min_age_for_rehab: Minimum age before rehab can start (periods)
    
    Returns:
        List of rehabilitation schedule items
    """
    logger.info(f"Planning rehabilitation for {len(land_parcels)} parcels")
    
    # Find parcels needing rehabilitation
    eligible = []
    for parcel in land_parcels:
        if parcel.land_use in [LandUseType.MINING, LandUseType.WASTE_DUMP]:
            if parcel.rehabilitation_stage == RehabilitationStage.PLANNED:
                if parcel.disturbance_date and parcel.disturbance_date <= min_age_for_rehab:
                    eligible.append(parcel)
    
    # Sort by age (oldest first)
    eligible.sort(key=lambda p: p.disturbance_date or 0)
    
    # Allocate capacity
    schedule = []
    current_period = min_age_for_rehab + 1
    remaining_capacity = rehab_capacity_ha_per_period
    
    for parcel in eligible:
        if remaining_capacity >= parcel.area:
            # Can rehabilitate in this period
            schedule.append({
                'parcel_id': parcel.parcel_id,
                'period': current_period,
                'area_ha': parcel.area,
                'stage': RehabilitationStage.EARTHWORKS.value
            })
            remaining_capacity -= parcel.area
        else:
            # Need new period
            current_period += 1
            remaining_capacity = rehab_capacity_ha_per_period
            schedule.append({
                'parcel_id': parcel.parcel_id,
                'period': current_period,
                'area_ha': parcel.area,
                'stage': RehabilitationStage.EARTHWORKS.value
            })
            remaining_capacity -= parcel.area
    
    logger.info(f"Rehabilitation schedule: {len(schedule)} activities over {current_period - min_age_for_rehab} periods")
    return schedule


def calculate_biodiversity_index(
    land_parcels: List[LandParcel],
    baseline_score: float = 75.0
) -> float:
    """
    Calculate overall biodiversity index for mine site.
    
    Index = Σ(parcel_area × parcel_biodiversity_score) / Total_area
    
    Normalized to baseline (pre-mining) score of 75.
    
    Args:
        land_parcels: List of land parcels
        baseline_score: Baseline biodiversity score (pre-mining)
    
    Returns:
        Biodiversity index (0-100)
    """
    if not land_parcels:
        return baseline_score
    
    total_area = sum(p.area for p in land_parcels)
    if total_area == 0:
        return baseline_score
    
    weighted_score = sum(p.area * p.biodiversity_score for p in land_parcels)
    index = weighted_score / total_area
    
    return min(100.0, max(0.0, index))


def estimate_rehabilitation_cost(
    area_ha: float,
    land_use: LandUseType,
    unit_cost_per_ha: Dict[str, float] = None
) -> float:
    """
    Estimate rehabilitation cost.
    
    Default costs ($/ha):
    - Pit: $50,000 (backfilling, contouring)
    - Waste dump: $30,000 (contouring, capping)
    - Tailings: $80,000 (capping, cover system)
    - Infrastructure: $20,000 (demolition, cleanup)
    
    Args:
        area_ha: Area in hectares
        land_use: Type of land use
        unit_cost_per_ha: Custom unit costs
    
    Returns:
        Total rehabilitation cost ($)
    """
    if unit_cost_per_ha is None:
        unit_cost_per_ha = {
            LandUseType.MINING.value: 50000,
            LandUseType.WASTE_DUMP.value: 30000,
            LandUseType.TAILINGS.value: 80000,
            LandUseType.INFRASTRUCTURE.value: 20000
        }
    
    unit_cost = unit_cost_per_ha.get(land_use.value, 30000)
    total_cost = area_ha * unit_cost
    
    return total_cost


def design_waste_dump(
    total_waste_t: float,
    waste_density: float = 2.2,  # t/m³
    max_height: float = 50.0,  # m
    slope_angle: float = 37.0,  # degrees (2H:1V ≈ 27°, 3H:1V ≈ 18°)
    bench_width: float = 10.0  # m
) -> Dict[str, float]:
    """
    Design waste rock dump geometry.
    
    Args:
        total_waste_t: Total waste tonnage (t)
        waste_density: Waste density (t/m³)
        max_height: Maximum dump height (m)
        slope_angle: Overall slope angle (degrees)
        bench_width: Bench width (m)
    
    Returns:
        Dict with volume_m3, footprint_area_m2, height_m, stability_factor
    """
    import math
    
    # Calculate volume
    volume_m3 = total_waste_t / waste_density
    
    # Calculate footprint (assuming conical shape)
    # V = (1/3) × π × r² × h
    # r = sqrt(3V / (π × h))
    radius = math.sqrt(3 * volume_m3 / (math.pi * max_height))
    footprint_area = math.pi * (radius ** 2)
    
    # Estimate stability factor (simplified Bishop method)
    # FoS ≈ 1.5 for well-designed dumps
    # Decreases with steeper slopes
    base_fos = 1.8
    slope_factor = 1.0 - (slope_angle - 18) / 100  # Penalty for steeper slopes
    stability_factor = base_fos * slope_factor
    
    return {
        'volume_m3': round(volume_m3, 0),
        'footprint_area_m2': round(footprint_area, 0),
        'height_m': max_height,
        'slope_angle_deg': slope_angle,
        'stability_factor': round(stability_factor, 2),
        'bench_width_m': bench_width
    }
