"""
2D Limit Equilibrium Method - Bishop simplified, Janbu simplified.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import math
import logging

from ..geotech_common.slope_geometry import SlopeSector
from ..geotech_common.material_properties import GeotechMaterial
from .slope_failure_surface import FailureSurface2D

logger = logging.getLogger(__name__)


@dataclass
class SlopeLEM2DConfig:
    """Configuration for 2D LEM analysis."""
    method: str = "Bishop"  # "Bishop", "Janbu"
    pore_pressure_mode: str = "none"  # "none", "ru_factor", "pwp_profile"
    ru: Optional[float] = None  # Pore pressure ratio (u / (γ * H))
    n_slices: int = 20  # Number of slices for analysis


@dataclass
class LEM2DResult:
    """Result from 2D LEM analysis."""
    surface: FailureSurface2D
    fos: float  # Factor of Safety
    normal_stress: np.ndarray  # Normal stress on each slice
    shear_stress: np.ndarray  # Shear stress on each slice
    slice_forces: Dict[str, Any]  # Detailed slice forces
    converged: bool = True
    iterations: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


def compute_fos_2d(
    surface: FailureSurface2D,
    slope_sector: SlopeSector,
    material: GeotechMaterial,
    config: SlopeLEM2DConfig
) -> LEM2DResult:
    """
    Compute Factor of Safety using 2D Limit Equilibrium Method.
    
    Args:
        surface: Failure surface to analyze
        slope_sector: Slope sector geometry
        material: Geotechnical material properties
        config: LEM configuration
    
    Returns:
        LEM2DResult with FOS and detailed results
    """
    if config.method == "Bishop":
        return _compute_bishop_simplified(surface, slope_sector, material, config)
    elif config.method == "Janbu":
        return _compute_janbu_simplified(surface, slope_sector, material, config)
    else:
        raise ValueError(f"Unknown LEM method: {config.method}")


def _compute_bishop_simplified(
    surface: FailureSurface2D,
    slope_sector: SlopeSector,
    material: GeotechMaterial,
    config: SlopeLEM2DConfig
) -> LEM2DResult:
    """
    Bishop Simplified Method.
    
    Iterative solution for FOS using Bishop's simplified method.
    """
    # Get surface coordinates
    x_coords = surface.x_coords
    z_coords = surface.z_coords
    
    if len(x_coords) < 2:
        return LEM2DResult(
            surface=surface,
            fos=0.0,
            normal_stress=np.array([]),
            shear_stress=np.array([]),
            slice_forces={},
            converged=False
        )
    
    # Divide into slices
    n_slices = config.n_slices
    slice_width = (x_coords[-1] - x_coords[0]) / n_slices
    
    # Material properties
    phi_rad = math.radians(material.friction_angle)
    c = material.cohesion
    gamma = material.unit_weight
    
    # Pore pressure
    ru = config.ru if config.pore_pressure_mode == "ru_factor" else 0.0
    
    # Initial FOS guess
    fos = 1.5
    max_iterations = 50
    tolerance = 1e-4
    
    for iteration in range(max_iterations):
        fos_old = fos
        
        # Compute FOS for this iteration
        numerator = 0.0
        denominator = 0.0
        
        for i in range(n_slices):
            # Slice center
            x_center = x_coords[0] + (i + 0.5) * slice_width
            
            # Find corresponding z on surface and slope
            # Simplified: interpolate z on surface
            z_surface = np.interp(x_center, x_coords, z_coords)
            
            # Find z on slope (simplified linear interpolation)
            toe_x, toe_z = slope_sector.toe_point[0], slope_sector.toe_point[2]
            crest_x, crest_z = slope_sector.crest_point[0], slope_sector.crest_point[2]
            
            if abs(crest_x - toe_x) < 1e-6:
                z_slope = toe_z
            else:
                slope_gradient = (crest_z - toe_z) / (crest_x - toe_x)
                z_slope = toe_z + slope_gradient * (x_center - toe_x)
            
            # Slice height
            h = max(0, z_slope - z_surface)
            
            # Slice base angle (simplified)
            if i < len(x_coords) - 1:
                dx = x_coords[i+1] - x_coords[i]
                dz = z_coords[i+1] - z_coords[i]
                alpha = math.atan2(dz, dx)
            else:
                alpha = 0.0
            
            # Weight
            W = gamma * h * slice_width
            
            # Pore pressure
            u = ru * gamma * h if ru else 0.0
            
            # Effective normal force
            N_prime = (W - u * slice_width / math.cos(alpha)) * math.cos(alpha)
            
            # Bishop's simplified method
            m_alpha = math.cos(alpha) + math.sin(alpha) * math.tan(phi_rad) / fos
            
            if m_alpha > 0:
                numerator += (c * slice_width / math.cos(alpha) + N_prime * math.tan(phi_rad)) / m_alpha
                denominator += W * math.sin(alpha) / m_alpha
        
        if denominator > 0:
            fos = numerator / denominator
        else:
            fos = 0.0
        
        # Check convergence
        if abs(fos - fos_old) < tolerance:
            break
    
    # Compute normal and shear stresses (simplified)
    normal_stress = np.zeros(n_slices)
    shear_stress = np.zeros(n_slices)
    
    for i in range(n_slices):
        x_center = x_coords[0] + (i + 0.5) * slice_width
        z_surface = np.interp(x_center, x_coords, z_coords)
        toe_x, toe_z = slope_sector.toe_point[0], slope_sector.toe_point[2]
        crest_x, crest_z = slope_sector.crest_point[0], slope_sector.crest_point[2]
        
        if abs(crest_x - toe_x) < 1e-6:
            z_slope = toe_z
        else:
            slope_gradient = (crest_z - toe_z) / (crest_x - toe_x)
            z_slope = toe_z + slope_gradient * (x_center - toe_x)
        
        h = max(0, z_slope - z_surface)
        W = gamma * h * slice_width
        
        if i < len(x_coords) - 1:
            dx = x_coords[i+1] - x_coords[i]
            dz = z_coords[i+1] - z_coords[i]
            alpha = math.atan2(dz, dx)
        else:
            alpha = 0.0
        
        u = ru * gamma * h if ru else 0.0
        N_prime = (W - u * slice_width / math.cos(alpha)) * math.cos(alpha)
        
        normal_stress[i] = N_prime / (slice_width / math.cos(alpha))
        shear_stress[i] = (c + normal_stress[i] * math.tan(phi_rad)) / fos
    
    converged = iteration < max_iterations - 1
    
    return LEM2DResult(
        surface=surface,
        fos=float(fos),
        normal_stress=normal_stress,
        shear_stress=shear_stress,
        slice_forces={"method": "Bishop", "n_slices": n_slices},
        converged=converged,
        iterations=iteration + 1
    )


def _compute_janbu_simplified(
    surface: FailureSurface2D,
    slope_sector: SlopeSector,
    material: GeotechMaterial,
    config: SlopeLEM2DConfig
) -> LEM2DResult:
    """
    Janbu Simplified Method.
    
    Similar to Bishop but with different assumptions about interslice forces.
    """
    # For now, use Bishop as approximation (full Janbu requires more complex iteration)
    # In a full implementation, this would use Janbu's correction factor
    result = _compute_bishop_simplified(surface, slope_sector, material, config)
    result.slice_forces["method"] = "Janbu"
    
    # Apply Janbu correction factor (simplified)
    # Full Janbu would require iterative correction
    correction_factor = 0.95  # Simplified correction
    result.fos *= correction_factor
    
    return result


def search_critical_surface_2d(
    slope_sector: SlopeSector,
    material: GeotechMaterial,
    search_params: Dict[str, Any],
    config: SlopeLEM2DConfig
) -> List[LEM2DResult]:
    """
    Search for critical failure surface using 2D LEM.
    
    Args:
        slope_sector: Slope sector to analyze
        material: Geotechnical material properties
        search_params: Parameters for surface generation
        config: LEM configuration
    
    Returns:
        List of LEM2DResult objects, ranked by FOS (lowest first)
    """
    from .slope_failure_surface import generate_circular_surfaces
    
    # Generate candidate surfaces
    surfaces = generate_circular_surfaces(slope_sector, search_params)
    
    # Compute FOS for each surface
    results = []
    for surface in surfaces:
        try:
            result = compute_fos_2d(surface, slope_sector, material, config)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to compute FOS for surface: {e}")
            continue
    
    # Sort by FOS (lowest = most critical)
    results.sort(key=lambda r: r.fos)
    
    logger.info(f"Found {len(results)} valid surfaces, critical FOS: {results[0].fos:.3f}" if results else "No valid surfaces found")
    
    return results

