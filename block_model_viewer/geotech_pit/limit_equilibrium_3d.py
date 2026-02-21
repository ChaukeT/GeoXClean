"""
3D Limit Equilibrium Method - Simplified 3D LEM wrapper.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import math
import logging

from ..geotech_common.slope_geometry import SlopeSector
from ..geotech_common.material_properties import GeotechMaterial
from .slope_failure_surface import FailureSurface3D

logger = logging.getLogger(__name__)


@dataclass
class SlopeLEM3DConfig:
    """Configuration for 3D LEM analysis."""
    method: str = "ellipsoid"  # "ellipsoid", "column-based"
    n_columns: int = 10  # Number of columns for column-based method
    pore_pressure_mode: str = "none"  # "none", "ru_factor"
    ru: Optional[float] = None  # Pore pressure ratio


@dataclass
class LEM3DResult:
    """Result from 3D LEM analysis."""
    surface: FailureSurface3D
    fos: float  # Factor of Safety
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


def compute_fos_3d(
    surface: FailureSurface3D,
    slope_sector: SlopeSector,
    material: GeotechMaterial,
    config: SlopeLEM3DConfig
) -> LEM3DResult:
    """
    Compute Factor of Safety using simplified 3D Limit Equilibrium Method.
    
    This is a simplified implementation using column-based approach.
    More advanced 3D methods can be added later.
    
    Args:
        surface: 3D failure surface
        slope_sector: Slope sector geometry
        material: Geotechnical material properties
        config: LEM configuration
    
    Returns:
        LEM3DResult with FOS
    """
    if config.method == "ellipsoid":
        return _compute_ellipsoid_3d(surface, slope_sector, material, config)
    elif config.method == "column-based":
        return _compute_column_based_3d(surface, slope_sector, material, config)
    else:
        raise ValueError(f"Unknown 3D LEM method: {config.method}")


def _compute_ellipsoid_3d(
    surface: FailureSurface3D,
    slope_sector: SlopeSector,
    material: GeotechMaterial,
    config: SlopeLEM3DConfig
) -> LEM3DResult:
    """
    Simplified ellipsoidal 3D FOS computation.
    
    Uses simplified approach: approximate as 2D sections and average.
    """
    # Get surface center and radii
    if surface.center is None or surface.radii is None:
        # Compute from vertices
        center = np.mean(surface.vertices, axis=0)
        radii = (
            np.max(surface.vertices[:, 0]) - np.min(surface.vertices[:, 0]),
            np.max(surface.vertices[:, 1]) - np.min(surface.vertices[:, 1]),
            np.max(surface.vertices[:, 2]) - np.min(surface.vertices[:, 2])
        )
    else:
        center = np.array(surface.center)
        radii = surface.radii
    
    # Material properties
    phi_rad = math.radians(material.friction_angle)
    c = material.cohesion
    gamma = material.unit_weight
    
    # Simplified: approximate volume and driving/resisting forces
    # Volume of ellipsoid
    volume = (4/3) * math.pi * radii[0] * radii[1] * radii[2]
    
    # Weight
    W = gamma * volume
    
    # Approximate failure surface area
    # Using approximate formula for ellipsoid surface area
    a, b, c_ell = radii
    if a == b == c_ell:
        # Sphere
        area = 4 * math.pi * a**2
    else:
        # Approximate ellipsoid area
        area = 4 * math.pi * ((a*b + a*c_ell + b*c_ell) / 3)**(2/3)
    
    # Driving moment (simplified)
    # Assume center of mass is at ellipsoid center
    toe = np.array(slope_sector.toe_point)
    crest = np.array(slope_sector.crest_point)
    
    # Horizontal distance from toe to center
    dx = center[0] - toe[0]
    dy = center[1] - toe[1]
    horizontal_dist = math.sqrt(dx**2 + dy**2)
    
    # Driving force (simplified)
    driving_force = W * math.sin(math.radians(slope_sector.dip))
    
    # Resisting force
    # Normal force on surface
    normal_force = W * math.cos(math.radians(slope_sector.dip))
    
    # Pore pressure
    ru = config.ru if config.pore_pressure_mode == "ru_factor" else 0.0
    u = ru * gamma * radii[2] if ru else 0.0
    
    # Effective normal force
    N_prime = normal_force - u * area
    
    # Resisting force
    resisting_force = c * area + N_prime * math.tan(phi_rad)
    
    # FOS
    if driving_force > 0:
        fos = resisting_force / driving_force
    else:
        fos = 999.0  # Very stable
    
    return LEM3DResult(
        surface=surface,
        fos=float(fos),
        metadata={
            "method": "ellipsoid",
            "volume": float(volume),
            "area": float(area),
            "driving_force": float(driving_force),
            "resisting_force": float(resisting_force)
        }
    )


def _compute_column_based_3d(
    surface: FailureSurface3D,
    slope_sector: SlopeSector,
    material: GeotechMaterial,
    config: SlopeLEM3DConfig
) -> LEM3DResult:
    """
    Column-based 3D LEM (simplified).
    
    Divides failure mass into columns and computes FOS.
    """
    # For now, delegate to ellipsoid method
    # Full column-based implementation would divide into columns
    result = _compute_ellipsoid_3d(surface, slope_sector, material, config)
    result.metadata["method"] = "column-based"
    result.metadata["n_columns"] = config.n_columns
    
    return result


def search_critical_surface_3d(
    slope_sector: SlopeSector,
    material: GeotechMaterial,
    search_params: Dict[str, Any],
    config: SlopeLEM3DConfig
) -> List[LEM3DResult]:
    """
    Search for critical failure surface using 3D LEM.
    
    Args:
        slope_sector: Slope sector to analyze
        material: Geotechnical material properties
        search_params: Parameters for surface generation
        config: LEM configuration
    
    Returns:
        List of LEM3DResult objects, ranked by FOS (lowest first)
    """
    from .slope_failure_surface import generate_3d_surfaces
    
    # Generate candidate surfaces
    surfaces = generate_3d_surfaces(slope_sector, search_params)
    
    # Compute FOS for each surface
    results = []
    for surface in surfaces:
        try:
            result = compute_fos_3d(surface, slope_sector, material, config)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to compute FOS for 3D surface: {e}")
            continue
    
    # Sort by FOS (lowest = most critical)
    results.sort(key=lambda r: r.fos)
    
    logger.info(f"Found {len(results)} valid 3D surfaces, critical FOS: {results[0].fos:.3f}" if results else "No valid 3D surfaces found")
    
    return results

