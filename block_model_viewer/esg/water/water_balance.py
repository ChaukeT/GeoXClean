"""
Water Balance Module

Node-link water balance model for mine water management.
Tracks water flows, pond levels, freeboard, salinity, and compliance.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of water nodes in the network"""
    SOURCE = "source"  # External source (river, aquifer, supply)
    SINK = "sink"  # External discharge point
    POND = "pond"  # Storage pond
    TAILINGS_DAM = "tailings_dam"
    PIT = "pit"  # Open pit collecting water
    UNDERGROUND = "underground"  # UG mine dewatering
    PROCESS_PLANT = "process_plant"
    POTABLE = "potable"  # Potable water system


@dataclass
class WaterNode:
    """
    Represents a node in the water balance network.
    
    Attributes:
        node_id: Unique identifier
        node_type: Type of node
        volume: Current volume (m³)
        capacity: Maximum capacity (m³)
        area: Surface area (m² for ponds, for evaporation)
        min_volume: Minimum operating volume (m³)
        evaporation_coeff: Evaporation coefficient (m/day)
        salinity: Water salinity (mg/L TDS)
        regulated: Whether node is regulated (requires compliance monitoring)
        freeboard_required: Required freeboard (m)
    """
    node_id: str
    node_type: NodeType
    volume: float = 0.0  # m³
    capacity: float = 1000000.0  # m³
    area: float = 0.0  # m²
    min_volume: float = 0.0  # m³
    evaporation_coeff: float = 0.005  # m/day (typical arid region)
    salinity: float = 500.0  # mg/L TDS
    regulated: bool = False
    freeboard_required: float = 1.0  # m


@dataclass
class WaterLink:
    """
    Represents a flow connection between nodes.
    
    Attributes:
        link_id: Unique identifier
        from_node: Source node ID
        to_node: Destination node ID
        capacity: Maximum flow capacity (m³/day)
        flow: Current flow rate (m³/day)
        pump_power: Pump power if applicable (kW)
        treatment: Whether water is treated in this link
        treatment_efficiency: Treatment efficiency (0-1) for contaminant removal
    """
    link_id: str
    from_node: str
    to_node: str
    capacity: float = 10000.0  # m³/day
    flow: float = 0.0  # m³/day
    pump_power: float = 0.0  # kW
    treatment: bool = False
    treatment_efficiency: float = 0.0


@dataclass
class WaterBalance:
    """
    Water balance results for a period.
    
    Attributes:
        period: Time period
        node_volumes: Dict mapping node_id to volume (m³)
        link_flows: Dict mapping link_id to flow (m³/day)
        inflows: Dict mapping node_id to total inflow (m³)
        outflows: Dict mapping node_id to total outflow (m³)
        evaporation: Dict mapping node_id to evaporation loss (m³)
        precipitation: Dict mapping node_id to precipitation gain (m³)
        freeboard: Dict mapping node_id to freeboard (m)
        compliance_status: Dict mapping node_id to compliance (True/False)
        total_water_use: Total water consumed (m³)
        recycled_water: Total water recycled (m³)
        discharge: Total water discharged (m³)
    """
    period: int
    node_volumes: Dict[str, float] = field(default_factory=dict)
    link_flows: Dict[str, float] = field(default_factory=dict)
    inflows: Dict[str, float] = field(default_factory=dict)
    outflows: Dict[str, float] = field(default_factory=dict)
    evaporation: Dict[str, float] = field(default_factory=dict)
    precipitation: Dict[str, float] = field(default_factory=dict)
    freeboard: Dict[str, float] = field(default_factory=dict)
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    total_water_use: float = 0.0
    recycled_water: float = 0.0
    discharge: float = 0.0
    water_use: float = 0.0  # For ESG Dashboard compatibility
    recycled: float = 0.0  # For ESG Dashboard compatibility


def simulate_water_balance(
    nodes: List[WaterNode],
    links: List[WaterLink],
    process_water_demand: Dict[int, float],  # period → m³/day
    precipitation: Dict[int, float],  # period → mm/day
    n_periods: int = 12,
    days_per_period: int = 30
) -> List[WaterBalance]:
    """
    Simulate water balance over multiple periods.
    
    Algorithm:
    1. For each period:
       - Add inflows (precipitation, supply, recycling)
       - Calculate evaporation losses
       - Route flows through network (satisfy demands)
       - Update node volumes
       - Check freeboard compliance
    
    Args:
        nodes: List of water nodes
        links: List of water links
        process_water_demand: Water demand by period (m³/day)
        precipitation: Precipitation by period (mm/day)
        n_periods: Number of periods to simulate
        days_per_period: Days per period
    
    Returns:
        List of WaterBalance results per period
    """
    logger.info(f"Simulating water balance for {n_periods} periods")
    
    # Initialize node states
    results = []
    
    for period in range(1, n_periods + 1):
        logger.debug(f"Period {period}: simulating water balance")
        
        balance = WaterBalance(period=period)
        # For ESG Dashboard compatibility
        balance.water_use = 0.0
        balance.recycled = 0.0

        # Get period parameters
        precip_mm = precipitation.get(period, 0.0)
        demand_m3_day = process_water_demand.get(period, 0.0)
        
        # Calculate precipitation inflow (mm/day → m³)
        for node in nodes:
            if node.area > 0:
                precip_m3 = (precip_mm / 1000) * node.area * days_per_period
                balance.precipitation[node.node_id] = precip_m3
                node.volume += precip_m3
        
        # Calculate evaporation losses
        for node in nodes:
            if node.area > 0:
                evap_m3 = estimate_evaporation(
                    area=node.area,
                    evaporation_coeff=node.evaporation_coeff,
                    days=days_per_period
                )
                balance.evaporation[node.node_id] = evap_m3
                node.volume = max(0, node.volume - evap_m3)
        
        # Satisfy process water demand (simplified routing)
        # Assume process plant draws from recycling ponds first, then fresh water
        water_needed = demand_m3_day * days_per_period
        water_supplied = 0.0
        
        # Try to supply from ponds/recycling
        for node in nodes:
            if node.node_type in [NodeType.POND, NodeType.TAILINGS_DAM]:
                available = node.volume - node.min_volume
                if available > 0:
                    supply = min(available, water_needed - water_supplied)
                    node.volume -= supply
                    water_supplied += supply
                    balance.recycled_water += supply
                    balance.recycled += supply  # For ESG Dashboard compatibility
                    if water_supplied >= water_needed:
                        break
        
        # If still need water, draw from source
        if water_supplied < water_needed:
            shortage = water_needed - water_supplied
            for node in nodes:
                if node.node_type == NodeType.SOURCE:
                    node.volume += shortage  # Assume infinite source
                    balance.total_water_use += shortage
                    balance.water_use += shortage  # For ESG Dashboard compatibility
                    break
        
        # Return water to ponds (assume 80% returns as tailings)
        water_returned = water_needed * 0.80
        for node in nodes:
            if node.node_type == NodeType.TAILINGS_DAM:
                returned = min(water_returned, node.capacity - node.volume)
                node.volume += returned
                water_returned -= returned
                if water_returned <= 0:
                    break
        
        # Calculate freeboard and compliance
        for node in nodes:
            if node.node_type in [NodeType.POND, NodeType.TAILINGS_DAM]:
                freeboard = calculate_pond_freeboard(
                    volume=node.volume,
                    capacity=node.capacity,
                    area=node.area
                )
                balance.freeboard[node.node_id] = freeboard
                
                # Check compliance
                if node.regulated:
                    compliant = freeboard >= node.freeboard_required
                    balance.compliance_status[node.node_id] = compliant
                    if not compliant:
                        logger.warning(f"Period {period}: {node.node_id} freeboard {freeboard:.2f}m < required {node.freeboard_required:.2f}m")
        
        # Record node volumes
        balance.node_volumes = {n.node_id: n.volume for n in nodes}
        
        # Record link flows (simplified)
        balance.link_flows = {l.link_id: l.flow for l in links}
        
        # Summary metrics
        balance.total_water_use += water_needed
        balance.water_use += water_needed  # For ESG Dashboard compatibility
        balance.discharge = 0.0  # No discharge in this simplified model

        results.append(balance)
    
    logger.info(f"Water balance simulation complete: {len(results)} periods")
    return results


def calculate_pond_freeboard(
    volume: float,  # m³
    capacity: float,  # m³
    area: float  # m²
) -> float:
    """
    Calculate freeboard (distance from water surface to dam crest).
    
    Freeboard = (Capacity - Volume) / Area
    
    Args:
        volume: Current water volume (m³)
        capacity: Maximum capacity (m³)
        area: Surface area (m²)
    
    Returns:
        Freeboard in meters
    """
    if area <= 0:
        return 0.0
    
    available_volume = capacity - volume
    freeboard = available_volume / area
    return max(0.0, freeboard)


def estimate_evaporation(
    area: float,  # m²
    evaporation_coeff: float = 0.005,  # m/day
    days: int = 30
) -> float:
    """
    Estimate evaporation losses from a pond.
    
    Evaporation = Area × Coefficient × Days
    
    Typical coefficients:
    - Arid regions: 0.005-0.010 m/day
    - Temperate: 0.002-0.005 m/day
    - Humid: 0.001-0.003 m/day
    
    Args:
        area: Surface area (m²)
        evaporation_coeff: Evaporation coefficient (m/day)
        days: Number of days
    
    Returns:
        Evaporation volume (m³)
    """
    evaporation_m3 = area * evaporation_coeff * days
    return evaporation_m3


def calculate_water_footprint(
    total_water_use: float,  # m³
    recycled_water: float,  # m³
    production: float  # tonnes
) -> Dict[str, float]:
    """
    Calculate water footprint metrics.
    
    Args:
        total_water_use: Total water consumed (m³)
        recycled_water: Water recycled (m³)
        production: Production (tonnes)
    
    Returns:
        Dict with intensity, recycling_rate, fresh_water_use
    """
    fresh_water = total_water_use - recycled_water
    
    intensity = total_water_use / production if production > 0 else 0.0
    recycling_rate = recycled_water / total_water_use if total_water_use > 0 else 0.0
    
    return {
        'total_water_use_m3': total_water_use,
        'fresh_water_use_m3': fresh_water,
        'recycled_water_m3': recycled_water,
        'water_intensity_m3_per_t': intensity,
        'recycling_rate': recycling_rate
    }


def design_tailings_dam(
    annual_production: float,  # t/year
    water_content: float = 0.40,  # m³ water / m³ tailings
    dry_density: float = 1.8,  # t/m³
    mine_life: int = 20,  # years
    freeboard: float = 2.0  # m
) -> Dict[str, float]:
    """
    Design tailings storage facility.
    
    Args:
        annual_production: Annual ore production (t/year)
        water_content: Water content of tailings slurry
        dry_density: Dry density of tailings (t/m³)
        mine_life: Mine life (years)
        freeboard: Required freeboard (m)
    
    Returns:
        Dict with capacity_m3, area_m2, height_m, volume_earth_m3
    """
    # Calculate total tailings volume
    total_tailings_t = annual_production * mine_life
    total_tailings_m3 = total_tailings_t / dry_density
    
    # Add water content
    total_volume = total_tailings_m3 * (1 + water_content)
    
    # Add freeboard (assume 10% of total height)
    capacity_m3 = total_volume * 1.15  # 15% safety factor
    
    # Estimate dimensions (assume square footprint, 3:1 slope)
    # For 3:1 slope, height ≈ sqrt(capacity / 10)
    import math
    height = math.pow(capacity_m3 / 10, 1/3)
    area_base = capacity_m3 / (height / 2)  # Average depth = height/2
    
    # Earthworks volume (assume 3:1 slope, embankment)
    volume_earth = area_base * height * 0.3  # Simplified
    
    return {
        'capacity_m3': round(capacity_m3, 0),
        'area_m2': round(area_base, 0),
        'height_m': round(height, 1),
        'volume_earth_m3': round(volume_earth, 0),
        'freeboard_m': freeboard
    }
