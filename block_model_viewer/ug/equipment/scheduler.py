"""
Equipment Scheduler

Resource-constrained equipment scheduling for underground mining operations.
Handles fleet management, availability calendars, and maintenance scheduling.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EquipmentType(Enum):
    """Types of underground mining equipment"""
    JUMBO_DRILL = "jumbo_drill"
    PRODUCTION_DRILL = "production_drill"
    LHD = "lhd"  # Load-Haul-Dump
    TRUCK = "truck"
    BOLTER = "bolter"
    SHOTCRETE = "shotcrete"
    CHARGER = "charger"
    GRADER = "grader"
    UTILITY = "utility"


@dataclass
class Equipment:
    """
    Represents a piece of mining equipment.
    
    Attributes:
        equipment_id: Unique identifier
        equipment_type: Type of equipment
        capacity: Capacity (t, m³, or appropriate unit)
        operating_cost: Operating cost per hour ($/hr)
        availability: Physical availability (0-1, accounts for downtime)
        utilization_target: Target utilization (0-1)
        fuel_consumption: Fuel consumption (L/hr)
        power_consumption: Power consumption (kW)
        crew_required: Number of operators required
    """
    equipment_id: str
    equipment_type: EquipmentType
    capacity: float
    operating_cost: float  # $/hr
    availability: float = 0.85  # 85% physical availability
    utilization_target: float = 0.75  # 75% target utilization
    fuel_consumption: float = 0.0  # L/hr
    power_consumption: float = 0.0  # kW
    crew_required: int = 1


@dataclass
class MaintenanceSchedule:
    """
    Maintenance schedule for equipment.
    
    Attributes:
        equipment_id: Equipment identifier
        maintenance_type: Type (preventive, repair, inspection)
        start_period: Period when maintenance starts
        duration_periods: Number of periods equipment unavailable
        cost: Maintenance cost ($)
    """
    equipment_id: str
    maintenance_type: str
    start_period: int
    duration_periods: int
    cost: float = 0.0


@dataclass
class EquipmentAssignment:
    """
    Assignment of equipment to stope/activity in a period.
    
    Attributes:
        equipment_id: Equipment identifier
        stope_id: Stope identifier
        period: Time period
        hours: Hours assigned
        activity: Activity (drilling, loading, hauling, support)
        tonnes_moved: Tonnes moved (if applicable)
        operating_cost: Operating cost for this assignment ($)
        fuel_used: Fuel consumed (L)
        energy_used: Energy consumed (kWh)
    """
    equipment_id: str
    stope_id: str
    period: int
    hours: float
    activity: str
    tonnes_moved: float = 0.0
    operating_cost: float = 0.0
    fuel_used: float = 0.0
    energy_used: float = 0.0


def schedule_equipment(
    stope_schedule: List[Dict],  # From MILP scheduler
    equipment_fleet: List[Equipment],
    maintenance_schedules: List[MaintenanceSchedule],
    hours_per_period: int = 720,  # 30 days × 24 hrs
    shifts_per_day: int = 3
) -> List[EquipmentAssignment]:
    """
    Schedule equipment to stopes considering availability and maintenance.
    
    Algorithm:
    1. For each period, identify stopes being mined
    2. Calculate equipment requirements per stope
    3. Assign equipment considering:
       - Availability (physical + maintenance)
       - Utilization targets
       - Crew availability
    4. Track operating costs, fuel, energy
    
    Args:
        stope_schedule: List of stope mining periods
        equipment_fleet: Available equipment
        maintenance_schedules: Planned maintenance
        hours_per_period: Working hours per period
        shifts_per_day: Number of shifts per day
    
    Returns:
        List of EquipmentAssignments
    """
    logger.info(f"Scheduling {len(equipment_fleet)} equipment units across {len(stope_schedule)} stope periods")
    
    assignments = []
    
    # Build maintenance calendar
    maintenance_calendar = {}
    for maint in maintenance_schedules:
        for p in range(maint.start_period, maint.start_period + maint.duration_periods):
            if p not in maintenance_calendar:
                maintenance_calendar[p] = set()
            maintenance_calendar[p].add(maint.equipment_id)
    
    # Group stopes by period
    periods = {}
    for stope in stope_schedule:
        period = stope.get('period', 0)
        if period not in periods:
            periods[period] = []
        periods[period].append(stope)
    
    # Schedule each period
    for period, stopes in sorted(periods.items()):
        logger.debug(f"Period {period}: scheduling {len(stopes)} stopes")
        
        # Calculate available hours per equipment type
        available_hours = {}
        for equip in equipment_fleet:
            # Check if under maintenance
            if period in maintenance_calendar and equip.equipment_id in maintenance_calendar[period]:
                continue
            
            # Available hours = hours_per_period × availability × utilization
            avail_hrs = hours_per_period * equip.availability * equip.utilization_target
            
            if equip.equipment_type not in available_hours:
                available_hours[equip.equipment_type] = {}
            available_hours[equip.equipment_type][equip.equipment_id] = avail_hrs
        
        # Assign equipment to each stope
        for stope in stopes:
            stope_id = stope.get('stope_id', 'unknown')
            tonnes = stope.get('tonnes', 0)
            
            if tonnes <= 0:
                continue
            
            # Estimate equipment requirements
            requirements = _estimate_equipment_needs(tonnes, stope)
            
            # Assign each equipment type
            for equip_type, hours_needed in requirements.items():
                if equip_type not in available_hours:
                    logger.warning(f"No {equip_type.value} available in period {period}")
                    continue
                
                # Find available equipment of this type
                for equip_id, avail_hrs in available_hours[equip_type].items():
                    if avail_hrs <= 0:
                        continue
                    
                    # Get equipment details
                    equip = next((e for e in equipment_fleet if e.equipment_id == equip_id), None)
                    if not equip:
                        continue
                    
                    # Assign hours (limited by availability)
                    assigned_hours = min(hours_needed, avail_hrs)
                    
                    # Calculate costs and consumption
                    assignment = EquipmentAssignment(
                        equipment_id=equip_id,
                        stope_id=stope_id,
                        period=period,
                        hours=assigned_hours,
                        activity=_activity_for_equipment(equip_type),
                        tonnes_moved=tonnes * (assigned_hours / hours_needed) if hours_needed > 0 else 0,
                        operating_cost=assigned_hours * equip.operating_cost,
                        fuel_used=assigned_hours * equip.fuel_consumption,
                        energy_used=assigned_hours * equip.power_consumption
                    )
                    assignments.append(assignment)
                    
                    # Reduce available hours
                    available_hours[equip_type][equip_id] -= assigned_hours
                    hours_needed -= assigned_hours
                    
                    if hours_needed <= 0:
                        break
    
    logger.info(f"Created {len(assignments)} equipment assignments")
    return assignments


def _estimate_equipment_needs(
    tonnes: float,
    stope: Dict
) -> Dict[EquipmentType, float]:
    """
    Estimate equipment hours needed for a stope.
    
    Simplified model based on industry benchmarks:
    - Drilling: 0.02 hrs/t (jumbo for development, production drill for stoping)
    - Loading: 0.01 hrs/t (LHD)
    - Hauling: 0.015 hrs/t (truck)
    - Support: 0.005 hrs/t (bolter)
    
    Args:
        tonnes: Tonnes to mine
        stope: Stope dictionary
    
    Returns:
        Dict mapping EquipmentType to hours required
    """
    requirements = {}
    
    # Production drilling
    requirements[EquipmentType.PRODUCTION_DRILL] = tonnes * 0.02
    
    # Loading (LHD)
    requirements[EquipmentType.LHD] = tonnes * 0.01
    
    # Hauling (Truck)
    requirements[EquipmentType.TRUCK] = tonnes * 0.015
    
    # Support installation
    requirements[EquipmentType.BOLTER] = tonnes * 0.005
    
    return requirements


def _activity_for_equipment(equip_type: EquipmentType) -> str:
    """Map equipment type to activity name"""
    mapping = {
        EquipmentType.JUMBO_DRILL: "development_drilling",
        EquipmentType.PRODUCTION_DRILL: "production_drilling",
        EquipmentType.LHD: "loading",
        EquipmentType.TRUCK: "hauling",
        EquipmentType.BOLTER: "ground_support",
        EquipmentType.SHOTCRETE: "shotcrete_application",
        EquipmentType.CHARGER: "charging",
        EquipmentType.GRADER: "maintenance",
        EquipmentType.UTILITY: "utility"
    }
    return mapping.get(equip_type, "unknown")


def calculate_equipment_requirements(
    annual_production: float,  # t/year
    mine_depth: float = 500.0,  # m
    haul_distance: float = 2000.0  # m
) -> Dict[EquipmentType, int]:
    """
    Calculate required equipment fleet size.
    
    Uses industry benchmarks for productivity:
    - LHD: 100-150 t/hr
    - Truck: 80-120 t/hr
    - Drill: 50-80 t/day per rig
    - Bolter: 200-300 t/day
    
    Args:
        annual_production: Annual production target (t/year)
        mine_depth: Average mining depth (m)
        haul_distance: Average haul distance (m)
    
    Returns:
        Dict mapping EquipmentType to quantity required
    """
    # Operating hours per year (accounting for availability)
    operating_hours = 8760 * 0.85 * 0.75  # hrs/year × availability × utilization
    
    # Calculate fleet sizes
    fleet = {}
    
    # LHD (assume 120 t/hr productivity)
    lhd_productivity = 120  # t/hr
    fleet[EquipmentType.LHD] = int(np.ceil(annual_production / (lhd_productivity * operating_hours)))
    
    # Trucks (adjust for haul distance, assume 100 t/hr baseline)
    truck_productivity = 100 * (2000 / max(haul_distance, 1000))  # Adjust for distance
    fleet[EquipmentType.TRUCK] = int(np.ceil(annual_production / (truck_productivity * operating_hours)))
    
    # Production drills (assume 60 t/day)
    drill_productivity = 60 * 365  # t/year per rig
    fleet[EquipmentType.PRODUCTION_DRILL] = int(np.ceil(annual_production / drill_productivity))
    
    # Bolters (assume 250 t/day)
    bolter_productivity = 250 * 365  # t/year
    fleet[EquipmentType.BOLTER] = int(np.ceil(annual_production / bolter_productivity))
    
    # Utility equipment (rule of thumb: 1 per 3 production units)
    fleet[EquipmentType.UTILITY] = max(1, (fleet[EquipmentType.LHD] + fleet[EquipmentType.TRUCK]) // 3)
    
    logger.info(f"Equipment requirements for {annual_production:,.0f} t/year:")
    for equip_type, qty in fleet.items():
        logger.info(f"  {equip_type.value}: {qty} units")
    
    return fleet


def optimize_fleet_size(
    production_schedule: List[Dict],
    equipment_costs: Dict[EquipmentType, float],  # $/unit
    operating_costs: Dict[EquipmentType, float],  # $/hr
    planning_horizon: int = 36  # months
) -> Dict[str, any]:
    """
    Optimize equipment fleet size to minimize total cost.
    
    Balances:
    - Capital cost (purchasing equipment)
    - Operating cost (usage)
    - Opportunity cost (underutilization)
    
    Args:
        production_schedule: Production schedule by period
        equipment_costs: Capital cost per equipment type
        operating_costs: Operating cost per hour per type
        planning_horizon: Planning horizon in months
    
    Returns:
        Dict with optimal_fleet, total_capital, total_operating, npv
    """
    logger.info("Optimizing fleet size...")
    
    # Calculate peak requirements per period
    peak_requirements = {}
    for period_data in production_schedule:
        period = period_data.get('period', 0)
        tonnes = period_data.get('tonnes', 0)
        
        requirements = _estimate_equipment_needs(tonnes, period_data)
        
        for equip_type, hours in requirements.items():
            if equip_type not in peak_requirements:
                peak_requirements[equip_type] = 0
            peak_requirements[equip_type] = max(peak_requirements[equip_type], hours)
    
    # Convert hours to units (assume 720 hrs/month, 85% availability, 75% utilization)
    hours_per_unit = 720 * 0.85 * 0.75
    
    optimal_fleet = {}
    total_capital = 0.0
    total_operating = 0.0
    
    for equip_type, peak_hours in peak_requirements.items():
        # Calculate units needed
        units_needed = int(np.ceil(peak_hours / hours_per_unit))
        optimal_fleet[equip_type] = units_needed
        
        # Calculate costs
        capital = units_needed * equipment_costs.get(equip_type, 0)
        operating = peak_hours * planning_horizon * operating_costs.get(equip_type, 0)
        
        total_capital += capital
        total_operating += operating
    
    # Calculate NPV (simplified, assume 10% discount rate)
    discount_rate = 0.10
    npv = -total_capital + sum(
        total_operating / planning_horizon / ((1 + discount_rate) ** (i / 12))
        for i in range(planning_horizon)
    )
    
    result = {
        'optimal_fleet': optimal_fleet,
        'total_capital': total_capital,
        'total_operating': total_operating,
        'npv': npv
    }
    
    logger.info(f"Optimal fleet: {optimal_fleet}")
    logger.info(f"Total capital: ${total_capital:,.0f}")
    logger.info(f"Total operating (over {planning_horizon} months): ${total_operating:,.0f}")
    
    return result


# Import numpy if available, otherwise use basic math
try:
    import numpy as np
except ImportError:
    import math
    class np:
        @staticmethod
        def ceil(x):
            return math.ceil(x)
