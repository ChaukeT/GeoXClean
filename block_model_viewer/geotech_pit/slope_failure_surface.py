"""
Slope Failure Surface - Generate candidate failure surfaces for 2D/3D LEM.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import math
import logging

from ..geotech_common.slope_geometry import SlopeSector

logger = logging.getLogger(__name__)


@dataclass
class FailureSurface2D:
    """2D failure surface (cross-section view)."""
    x_coords: np.ndarray  # Horizontal coordinates
    z_coords: np.ndarray  # Vertical coordinates
    surface_type: str  # "circular", "log-spiral", "polygonal"
    center: Optional[tuple[float, float]] = None  # Center point for circular surfaces
    radius: Optional[float] = None  # Radius for circular surfaces
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def get_length(self) -> float:
        """Compute surface length."""
        if len(self.x_coords) < 2:
            return 0.0
        dx = np.diff(self.x_coords)
        dz = np.diff(self.z_coords)
        lengths = np.sqrt(dx**2 + dz**2)
        return float(np.sum(lengths))


@dataclass
class FailureSurface3D:
    """3D failure surface."""
    vertices: np.ndarray  # (N, 3) array of vertices
    faces: np.ndarray  # (M, 3) array of face indices
    surface_type: str  # "ellipsoidal", "spherical", "polyhedral"
    center: Optional[tuple[float, float, float]] = None
    radii: Optional[tuple[float, float, float]] = None  # For ellipsoidal surfaces
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def get_area(self) -> float:
        """Compute approximate surface area."""
        if len(self.faces) == 0:
            return 0.0
        
        # Compute area of each triangular face
        areas = []
        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            
            # Cross product for triangle area
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            areas.append(area)
        
        return float(np.sum(areas))


def generate_circular_surfaces(
    slope_sector: SlopeSector,
    search_params: Dict[str, Any]
) -> List[FailureSurface2D]:
    """
    Generate circular failure surfaces for 2D analysis.
    
    Args:
        slope_sector: Slope sector to analyze
        search_params: Dictionary with:
            - n_surfaces: Number of surfaces to generate
            - center_x_range: (min, max) for center X coordinate
            - center_z_range: (min, max) for center Z coordinate
            - radius_range: (min, max) for radius
    
    Returns:
        List of FailureSurface2D objects
    """
    n_surfaces = search_params.get("n_surfaces", 50)
    center_x_range = search_params.get("center_x_range", (-100, 100))
    center_z_range = search_params.get("center_z_range", (-50, 50))
    radius_range = search_params.get("radius_range", (10, 200))
    
    # Get slope geometry
    toe_x, toe_z = slope_sector.toe_point[0], slope_sector.toe_point[2]
    crest_x, crest_z = slope_sector.crest_point[0], slope_sector.crest_point[2]
    
    surfaces = []
    
    for i in range(n_surfaces):
        # Random center within search window
        center_x = np.random.uniform(center_x_range[0], center_x_range[1]) + (toe_x + crest_x) / 2
        center_z = np.random.uniform(center_z_range[0], center_z_range[1]) + min(toe_z, crest_z)
        
        # Random radius
        radius = np.random.uniform(radius_range[0], radius_range[1])
        
        # Generate circular arc
        # Find intersection points with slope
        # Simplified: assume arc starts at toe and ends at crest
        angle_start = math.atan2(toe_z - center_z, toe_x - center_x)
        angle_end = math.atan2(crest_z - center_z, crest_x - center_x)
        
        # Ensure proper angle range
        if angle_end < angle_start:
            angle_end += 2 * math.pi
        
        # Generate points along arc
        n_points = 50
        angles = np.linspace(angle_start, angle_end, n_points)
        
        x_coords = center_x + radius * np.cos(angles)
        z_coords = center_z + radius * np.sin(angles)
        
        surface = FailureSurface2D(
            x_coords=x_coords,
            z_coords=z_coords,
            surface_type="circular",
            center=(center_x, center_z),
            radius=radius,
            metadata={"surface_index": i}
        )
        surfaces.append(surface)
    
    logger.info(f"Generated {len(surfaces)} circular failure surfaces")
    return surfaces


def generate_3d_surfaces(
    slope_sector: SlopeSector,
    search_params: Dict[str, Any]
) -> List[FailureSurface3D]:
    """
    Generate 3D ellipsoidal failure surfaces.
    
    Args:
        slope_sector: Slope sector to analyze
        search_params: Dictionary with:
            - n_surfaces: Number of surfaces to generate
            - center_range: (x_range, y_range, z_range) tuples
            - radius_range: (rx_range, ry_range, rz_range) tuples
    
    Returns:
        List of FailureSurface3D objects
    """
    n_surfaces = search_params.get("n_surfaces", 20)
    
    # Get slope geometry
    toe = np.array(slope_sector.toe_point)
    crest = np.array(slope_sector.crest_point)
    center_base = (toe + crest) / 2
    
    # Default ranges
    center_ranges = search_params.get("center_range", [
        (-50, 50),  # x
        (-50, 50),  # y
        (-30, 30)   # z
    ])
    
    radius_ranges = search_params.get("radius_range", [
        (20, 100),  # rx
        (20, 100),  # ry
        (10, 50)    # rz
    ])
    
    surfaces = []
    
    for i in range(n_surfaces):
        # Random center
        center = np.array([
            center_base[0] + np.random.uniform(center_ranges[0][0], center_ranges[0][1]),
            center_base[1] + np.random.uniform(center_ranges[1][0], center_ranges[1][1]),
            center_base[2] + np.random.uniform(center_ranges[2][0], center_ranges[2][1])
        ])
        
        # Random radii
        radii = (
            np.random.uniform(radius_ranges[0][0], radius_ranges[0][1]),
            np.random.uniform(radius_ranges[1][0], radius_ranges[1][1]),
            np.random.uniform(radius_ranges[2][0], radius_ranges[2][1])
        )
        
        # Generate ellipsoid mesh
        # Use spherical coordinates and transform to ellipsoid
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        vertices = []
        for vi in v:
            for ui in u:
                # Spherical coordinates
                x = radii[0] * np.sin(vi) * np.cos(ui)
                y = radii[1] * np.sin(vi) * np.sin(ui)
                z = radii[2] * np.cos(vi)
                
                # Translate to center
                vertex = center + np.array([x, y, z])
                vertices.append(vertex)
        
        vertices = np.array(vertices)
        
        # Generate faces (simplified triangulation)
        faces = []
        n_u, n_v = 20, 20
        for vi in range(n_v - 1):
            for ui in range(n_u - 1):
                idx = vi * n_u + ui
                
                # Two triangles per quad
                faces.append([idx, idx + 1, idx + n_u])
                faces.append([idx + 1, idx + n_u + 1, idx + n_u])
        
        faces = np.array(faces)
        
        surface = FailureSurface3D(
            vertices=vertices,
            faces=faces,
            surface_type="ellipsoidal",
            center=tuple(center),
            radii=radii,
            metadata={"surface_index": i}
        )
        surfaces.append(surface)
    
    logger.info(f"Generated {len(surfaces)} 3D ellipsoidal failure surfaces")
    return surfaces

