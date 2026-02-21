"""
Cycle Time Model (STEP 30)

Compute cycle times and effective capacity for routes.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Route:
    """
    Route definition.
    
    Attributes:
        id: Route identifier
        source: Source location (digline, pit area, stope)
        destination: Destination location (plant, stockpile, dump)
        distance_km: Distance in kilometers
        vertical_change_m: Vertical change in meters
        congestion_factor: Congestion factor (1.0 = no congestion, >1.0 = slower)
    """
    id: str
    source: str
    destination: str
    distance_km: float
    vertical_change_m: float = 0.0
    congestion_factor: float = 1.0
    
    def __post_init__(self):
        """Validate route."""
        if self.distance_km < 0:
            raise ValueError("distance_km must be non-negative")
        if self.congestion_factor < 0:
            raise ValueError("congestion_factor must be non-negative")


@dataclass
class CycleTimeResult:
    """
    Result from cycle time calculation.
    
    Attributes:
        route_id: Route identifier
        truck_cycle_minutes: Truck cycle time in minutes
        tonnes_per_hour: Effective tonnes per hour
    """
    route_id: str
    truck_cycle_minutes: float
    tonnes_per_hour: float


def compute_cycle_time(
    truck: Any,  # Truck
    route: Route,
    parameters: Optional[Dict[str, float]] = None
) -> CycleTimeResult:
    """
    Compute cycle time for a truck on a route.
    
    Args:
        truck: Truck instance
        route: Route instance
        parameters: Optional parameters (load_time, dump_time, etc.)
        
    Returns:
        CycleTimeResult
    """
    parameters = parameters or {}
    
    load_time_min = parameters.get("load_time_min", 3.0)
    dump_time_min = parameters.get("dump_time_min", 2.0)
    
    # Travel time
    # Loaded travel
    loaded_speed_kmh = truck.speed_loaded_kmh / route.congestion_factor
    loaded_time_min = (route.distance_km / loaded_speed_kmh) * 60.0 if loaded_speed_kmh > 0 else 0.0
    
    # Empty travel
    empty_speed_kmh = truck.speed_empty_kmh / route.congestion_factor
    empty_time_min = (route.distance_km / empty_speed_kmh) * 60.0 if empty_speed_kmh > 0 else 0.0
    
    # Vertical change penalty (simplified)
    vertical_penalty_min = abs(route.vertical_change_m) * 0.1  # 0.1 min per meter
    
    # Total cycle time
    cycle_time_min = load_time_min + loaded_time_min + dump_time_min + empty_time_min + vertical_penalty_min
    
    # Effective tonnes per hour
    cycles_per_hour = 60.0 / cycle_time_min if cycle_time_min > 0 else 0.0
    tonnes_per_hour = cycles_per_hour * truck.payload_tonnes * truck.availability * truck.utilisation
    
    return CycleTimeResult(
        route_id=route.id,
        truck_cycle_minutes=cycle_time_min,
        tonnes_per_hour=tonnes_per_hour
    )

