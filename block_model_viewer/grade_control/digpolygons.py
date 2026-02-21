"""
Diglines / Ore-Waste Polygons Engine (STEP 29)

Represent diglines (ore/waste polygons) at bench level.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlanPolygon:
    """
    Plan polygon representing digline at bench level.
    
    Attributes:
        id: Unique polygon identifier
        bench_code: Bench code/elevation
        vertices_xy: Array of 2D vertices (N, 2) in plan view
        elevation: Elevation (z-coordinate)
        ore_flag: True if ore polygon, False if waste
        material_type: Material type ("ore", "waste", "low-grade", "stockpile", etc.)
        target_destination: Target destination ("plant", stockpile name, etc.)
    """
    id: str
    bench_code: str
    vertices_xy: np.ndarray  # (N, 2) array
    elevation: float
    ore_flag: bool = True
    material_type: Optional[str] = None
    target_destination: Optional[str] = None
    
    def __post_init__(self):
        """Validate vertices."""
        if self.vertices_xy.shape[1] != 2:
            raise ValueError("vertices_xy must be (N, 2) array")
        if len(self.vertices_xy) < 3:
            raise ValueError("Polygon must have at least 3 vertices")


@dataclass
class DiglineSet:
    """
    Collection of digline polygons.
    
    Attributes:
        polygons: List of PlanPolygon
        metadata: Additional metadata
    """
    polygons: List[PlanPolygon] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_polygon(self, polygon: PlanPolygon) -> None:
        """Add polygon to set."""
        self.polygons.append(polygon)
    
    def get_polygons_for_bench(self, bench_code: str) -> List[PlanPolygon]:
        """Get all polygons for a specific bench."""
        return [p for p in self.polygons if p.bench_code == bench_code]
    
    def get_ore_polygons(self) -> List[PlanPolygon]:
        """Get all ore polygons."""
        return [p for p in self.polygons if p.ore_flag]
    
    def get_waste_polygons(self) -> List[PlanPolygon]:
        """Get all waste polygons."""
        return [p for p in self.polygons if not p.ore_flag]


def point_in_polygon(point: np.ndarray, polygon_vertices: np.ndarray) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm.
    
    Args:
        point: Point coordinates (x, y)
        polygon_vertices: Polygon vertices (N, 2) array
        
    Returns:
        True if point is inside polygon
    """
    x, y = point[0], point[1]
    n = len(polygon_vertices)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon_vertices[i]
        xj, yj = polygon_vertices[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def blocks_within_polygon(
    gc_grid: Any,
    polygon: PlanPolygon
) -> np.ndarray:
    """
    Find GC blocks within a polygon.
    
    Args:
        gc_grid: GCGridDefinition
        polygon: PlanPolygon
        
    Returns:
        Boolean array indicating which blocks are inside polygon
    """
    block_centers = gc_grid.get_block_centers()
    n_blocks = gc_grid.get_block_count()
    
    # Filter blocks at same elevation (within tolerance)
    elevation_tolerance = gc_grid.dz / 2.0
    z_mask = np.abs(block_centers[:, 2] - polygon.elevation) <= elevation_tolerance
    
    # Check which blocks are inside polygon (plan view)
    inside_mask = np.zeros(n_blocks, dtype=bool)
    
    for i in range(n_blocks):
        if z_mask[i]:
            point_xy = block_centers[i, :2]
            if point_in_polygon(point_xy, polygon.vertices_xy):
                inside_mask[i] = True
    
    return inside_mask


def assign_ore_waste_flags(
    gc_model: Any,
    diglines: DiglineSet,
    cutoff_rules: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Assign ore/waste flags based on diglines and cutoff rules.
    
    Args:
        gc_model: GCModel
        diglines: DiglineSet
        cutoff_rules: Dictionary with cutoff rules (e.g., {"Fe": {"cutoff": 50.0, "direction": ">="}})
        
    Returns:
        Dictionary mapping polygon_id -> boolean mask of ore blocks
    """
    n_blocks = gc_model.grid.get_block_count()
    results = {}
    
    for polygon in diglines.polygons:
        # Find blocks within polygon
        block_mask = blocks_within_polygon(gc_model.grid, polygon)
        
        if not np.any(block_mask):
            results[polygon.id] = np.zeros(n_blocks, dtype=bool)
            continue
        
        # Classify blocks based on cutoff rules
        ore_mask = np.zeros(n_blocks, dtype=bool)
        
        # Check each cutoff rule
        for prop_name, rule in cutoff_rules.items():
            prop_values = gc_model.get_property(prop_name)
            if prop_values is None:
                continue
            
            cutoff = rule.get("cutoff", 0.0)
            direction = rule.get("direction", ">=")
            
            if direction == ">=":
                meets_cutoff = prop_values >= cutoff
            elif direction == "<=":
                meets_cutoff = prop_values <= cutoff
            elif direction == ">":
                meets_cutoff = prop_values > cutoff
            elif direction == "<":
                meets_cutoff = prop_values < cutoff
            else:
                meets_cutoff = np.zeros(n_blocks, dtype=bool)
            
            # Combine with polygon mask
            ore_mask |= (block_mask & meets_cutoff)
        
        # If polygon is explicitly marked as waste, override
        if not polygon.ore_flag:
            ore_mask[block_mask] = False
        
        results[polygon.id] = ore_mask
    
    return results

