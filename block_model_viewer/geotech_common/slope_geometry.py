"""
Slope Geometry - Define slope sectors and geometry for pit walls.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class SlopeSector:
    """Single slope sector (wall segment)."""
    id: str
    toe_point: Tuple[float, float, float]  # (x, y, z)
    crest_point: Tuple[float, float, float]  # (x, y, z)
    height: float  # Vertical height (m)
    dip: float  # Dip angle (degrees from horizontal)
    dip_direction: float  # Dip direction (degrees, 0-360)
    bench_height: float  # Height of individual benches (m)
    berm_width: float  # Width of berms between benches (m)
    overall_slope_angle: float  # Overall slope angle (degrees from horizontal)
    domain_code: Optional[str] = None  # Linked to geological domain
    material_name: Optional[str] = None  # Linked to GeotechMaterial
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_geometry(self) -> Dict[str, float]:
        """Compute derived geometry parameters."""
        import math
        
        # Compute horizontal distance
        dx = self.crest_point[0] - self.toe_point[0]
        dy = self.crest_point[1] - self.toe_point[1]
        dz = self.crest_point[2] - self.toe_point[2]
        
        horizontal_distance = math.sqrt(dx**2 + dy**2)
        vertical_height = abs(dz)
        
        # Compute actual slope angle
        if horizontal_distance > 0:
            actual_slope_angle = math.degrees(math.atan(vertical_height / horizontal_distance))
        else:
            actual_slope_angle = 90.0
        
        # Compute number of benches
        n_benches = int(vertical_height / self.bench_height) if self.bench_height > 0 else 1
        
        return {
            "horizontal_distance": horizontal_distance,
            "vertical_height": vertical_height,
            "actual_slope_angle": actual_slope_angle,
            "n_benches": n_benches,
            "slope_length": math.sqrt(horizontal_distance**2 + vertical_height**2)
        }


class SlopeSet:
    """Collection of slope sectors."""
    
    def __init__(self, name: str = "default"):
        """Initialize slope set."""
        self.name = name
        self.sectors: List[SlopeSector] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_sector(self, sector: SlopeSector):
        """Add a slope sector."""
        self.sectors.append(sector)
    
    def get_sector(self, sector_id: str) -> Optional[SlopeSector]:
        """Get sector by ID."""
        for sector in self.sectors:
            if sector.id == sector_id:
                return sector
        return None
    
    def get_sectors_by_domain(self, domain_code: str) -> List[SlopeSector]:
        """Get all sectors for a given domain."""
        return [s for s in self.sectors if s.domain_code == domain_code]
    
    def list_sectors(self) -> List[str]:
        """List all sector IDs."""
        return [s.id for s in self.sectors]

