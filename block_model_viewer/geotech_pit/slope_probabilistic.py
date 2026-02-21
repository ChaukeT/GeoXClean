"""
Probabilistic Slope Stability - Monte Carlo / LHS analysis.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import logging

from ..geotech_common.slope_geometry import SlopeSector
from ..geotech_common.material_properties import GeotechMaterial
from ..geotech_common.probability_utils import sample_material_properties, probability_of_failure, compute_fos_statistics
from .limit_equilibrium_2d import compute_fos_2d, SlopeLEM2DConfig, LEM2DResult
from .limit_equilibrium_3d import compute_fos_3d, SlopeLEM3DConfig, LEM3DResult
from .slope_failure_surface import FailureSurface2D, FailureSurface3D

logger = logging.getLogger(__name__)


@dataclass
class ProbSlopeConfig:
    """Configuration for probabilistic slope analysis."""
    slope_sector: SlopeSector
    method_2d_or_3d: str  # "2D" | "3D"
    base_material: GeotechMaterial
    cov: Dict[str, float]  # Coefficient of variation per parameter
    n_realizations: int
    surface: Optional[FailureSurface2D | FailureSurface3D] = None  # Fixed surface or None for critical
    lem_config_2d: Optional[SlopeLEM2DConfig] = None
    lem_config_3d: Optional[SlopeLEM3DConfig] = None
    search_params: Optional[Dict[str, Any]] = None  # If surface is None, use these to search


@dataclass
class ProbSlopeResult:
    """Result from probabilistic slope analysis."""
    fos_samples: np.ndarray  # Array of FOS values
    probability_of_failure: float
    fos_stats: Dict[str, float]
    critical_surface_stats: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


def run_probabilistic_slope(config: ProbSlopeConfig) -> ProbSlopeResult:
    """
    Run probabilistic slope stability analysis.
    
    Samples material properties and computes FOS distribution.
    
    Args:
        config: Probabilistic analysis configuration
    
    Returns:
        ProbSlopeResult with FOS samples and statistics
    """
    # Sample material properties
    material_samples = sample_material_properties(
        config.base_material,
        config.n_realizations,
        config.cov
    )
    
    fos_samples = []
    critical_surfaces = []
    
    if config.method_2d_or_3d == "2D":
        # 2D analysis
        lem_config = config.lem_config_2d or SlopeLEM2DConfig()
        
        for i, material in enumerate(material_samples):
            try:
                if config.surface is not None:
                    # Use fixed surface
                    surface = config.surface
                else:
                    # Find critical surface for this material
                    from .limit_equilibrium_2d import search_critical_surface_2d
                    search_params = config.search_params or {"n_surfaces": 20}
                    results = search_critical_surface_2d(
                        config.slope_sector,
                        material,
                        search_params,
                        lem_config
                    )
                    
                    if not results:
                        continue
                    
                    # Use most critical surface
                    result = results[0]
                    fos_samples.append(result.fos)
                    critical_surfaces.append(result.surface)
                    continue
                
                # Compute FOS for fixed surface
                from .limit_equilibrium_2d import compute_fos_2d
                result = compute_fos_2d(
                    surface,
                    config.slope_sector,
                    material,
                    lem_config
                )
                
                fos_samples.append(result.fos)
                critical_surfaces.append(result.surface)
                
            except Exception as e:
                logger.warning(f"Failed to compute FOS for realization {i}: {e}")
                continue
    
    else:  # 3D
        lem_config = config.lem_config_3d or SlopeLEM3DConfig()
        
        for i, material in enumerate(material_samples):
            try:
                if config.surface is not None:
                    # Use fixed surface
                    surface = config.surface
                else:
                    # Find critical surface for this material
                    from .limit_equilibrium_3d import search_critical_surface_3d
                    search_params = config.search_params or {"n_surfaces": 10}
                    results = search_critical_surface_3d(
                        config.slope_sector,
                        material,
                        search_params,
                        lem_config
                    )
                    
                    if not results:
                        continue
                    
                    # Use most critical surface
                    result = results[0]
                    fos_samples.append(result.fos)
                    critical_surfaces.append(result.surface)
                    continue
                
                # Compute FOS for fixed surface
                from .limit_equilibrium_3d import compute_fos_3d
                result = compute_fos_3d(
                    surface,
                    config.slope_sector,
                    material,
                    lem_config
                )
                
                fos_samples.append(result.fos)
                critical_surfaces.append(result.surface)
                
            except Exception as e:
                logger.warning(f"Failed to compute FOS for 3D realization {i}: {e}")
                continue
    
    fos_samples = np.array(fos_samples)
    
    if len(fos_samples) == 0:
        logger.warning("No valid FOS samples generated")
        return ProbSlopeResult(
            fos_samples=np.array([]),
            probability_of_failure=0.0,
            fos_stats={},
            critical_surface_stats=None
        )
    
    # Compute statistics
    pof = probability_of_failure(fos_samples)
    fos_stats = compute_fos_statistics(fos_samples)
    
    # Critical surface statistics
    critical_surface_stats = None
    if critical_surfaces:
        # Compute statistics on critical surface parameters
        if config.method_2d_or_3d == "2D":
            radii = [s.radius for s in critical_surfaces if s.radius is not None]
            if radii:
                critical_surface_stats = {
                    "mean_radius": float(np.mean(radii)),
                    "std_radius": float(np.std(radii)),
                    "min_radius": float(np.min(radii)),
                    "max_radius": float(np.max(radii))
                }
        else:
            # 3D surface stats
            areas = [s.get_area() for s in critical_surfaces]
            if areas:
                critical_surface_stats = {
                    "mean_area": float(np.mean(areas)),
                    "std_area": float(np.std(areas)),
                    "min_area": float(np.min(areas)),
                    "max_area": float(np.max(areas))
                }
    
    logger.info(f"Probabilistic analysis complete: {len(fos_samples)} realizations, POF={pof:.3f}, mean FOS={fos_stats['mean']:.3f}")
    
    return ProbSlopeResult(
        fos_samples=fos_samples,
        probability_of_failure=pof,
        fos_stats=fos_stats,
        critical_surface_stats=critical_surface_stats,
        metadata={
            "n_realizations": config.n_realizations,
            "method": config.method_2d_or_3d,
            "n_valid": len(fos_samples)
        }
    )

