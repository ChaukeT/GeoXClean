"""
Fleet Model (STEP 30)

Represent trucks, shovels, loaders, and their capacities.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Truck:
    """
    Truck definition.
    
    Attributes:
        id: Truck identifier
        payload_tonnes: Payload capacity (tonnes)
        speed_loaded_kmh: Speed when loaded (km/h)
        speed_empty_kmh: Speed when empty (km/h)
        availability: Availability fraction (0-1)
        utilisation: Utilisation fraction (0-1)
    """
    id: str
    payload_tonnes: float
    speed_loaded_kmh: float
    speed_empty_kmh: float
    availability: float = 0.9
    utilisation: float = 0.85
    
    def __post_init__(self):
        """Validate truck."""
        if not (0 <= self.availability <= 1):
            raise ValueError("availability must be between 0 and 1")
        if not (0 <= self.utilisation <= 1):
            raise ValueError("utilisation must be between 0 and 1")


@dataclass
class Shovel:
    """
    Shovel definition.
    
    Attributes:
        id: Shovel identifier
        capacity_tonnes: Bucket capacity (tonnes)
        cycle_time_sec: Cycle time (seconds)
        availability: Availability fraction (0-1)
        utilisation: Utilisation fraction (0-1)
    """
    id: str
    capacity_tonnes: float
    cycle_time_sec: float
    availability: float = 0.9
    utilisation: float = 0.85
    
    def __post_init__(self):
        """Validate shovel."""
        if not (0 <= self.availability <= 1):
            raise ValueError("availability must be between 0 and 1")
        if not (0 <= self.utilisation <= 1):
            raise ValueError("utilisation must be between 0 and 1")
    
    def get_productivity_tph(self) -> float:
        """Get productivity in tonnes per hour."""
        if self.cycle_time_sec <= 0:
            return 0.0
        cycles_per_hour = 3600.0 / self.cycle_time_sec
        return cycles_per_hour * self.capacity_tonnes * self.availability * self.utilisation


@dataclass
class FleetConfig:
    """
    Fleet configuration.
    
    Attributes:
        trucks: List of Truck
        shovels: List of Shovel
        shift_hours: Shift duration in hours
    """
    trucks: List[Truck] = field(default_factory=list)
    shovels: List[Shovel] = field(default_factory=list)
    shift_hours: float = 12.0
    
    def get_total_truck_capacity_tph(self) -> float:
        """Get total truck capacity in tonnes per hour."""
        total = 0.0
        for truck in self.trucks:
            # Simplified: assume trucks can make trips based on average speed
            # In practice, would need cycle time calculation
            total += truck.payload_tonnes * truck.availability * truck.utilisation * 2.0  # 2 trips per hour estimate
        return total
    
    def get_total_shovel_capacity_tph(self) -> float:
        """Get total shovel capacity in tonnes per hour."""
        return sum(shovel.get_productivity_tph() for shovel in self.shovels)

