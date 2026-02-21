"""
GC Kriging Engine (STEP 29)

Specialized kriging for Grade Control support using blast-hole and GC hole data.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class GCKrigingConfig:
    """
    Configuration for GC kriging.
    
    Attributes:
        property_name: Property to estimate (e.g., "Fe", "SiO2")
        variogram_model: Variogram model parameters (dict with "type", "range", "sill", "nugget")
        search_ellipse: Search ellipse parameters (dict with "major", "minor", "vertical", "azimuth", "dip")
        min_samples: Minimum number of samples required
        max_samples: Maximum number of samples to use
        bench_code: Optional bench code to filter samples
        use_dh: Use drillhole data
        use_bh: Use blast-hole data
    """
    property_name: str
    variogram_model: Dict[str, Any]
    search_ellipse: Dict[str, float] = field(default_factory=lambda: {
        "major": 50.0,
        "minor": 50.0,
        "vertical": 10.0,
        "azimuth": 0.0,
        "dip": 0.0
    })
    min_samples: int = 3
    max_samples: int = 16
    bench_code: Optional[str] = None
    use_dh: bool = True
    use_bh: bool = True


@dataclass
class GCKrigingResult:
    """
    Result from GC kriging.
    
    Attributes:
        estimates: Array of estimated values
        variance: Array of kriging variances
        gc_grid: GCGridDefinition
    """
    estimates: np.ndarray
    variance: np.ndarray
    gc_grid: Any  # GCGridDefinition
    
    def __post_init__(self):
        """Validate arrays."""
        if len(self.estimates) != len(self.variance):
            raise ValueError("estimates and variance must have same length")
        if len(self.estimates) != self.gc_grid.get_block_count():
            raise ValueError("estimates length must match GC grid block count")


def run_gc_ok(
    blast_or_gc_samples: Any,
    gc_grid: Any,
    config: GCKrigingConfig
) -> GCKrigingResult:
    """
    Run Ordinary Kriging for Grade Control.
    
    Args:
        blast_or_gc_samples: Sample data (DrillholeDatabase subset or dict with coords/values)
        gc_grid: GCGridDefinition
        config: GCKrigingConfig
        
    Returns:
        GCKrigingResult
    """
    from ..models.kriging3d import ordinary_kriging_3d
    
    # Extract sample coordinates and values
    if hasattr(blast_or_gc_samples, 'collars') and hasattr(blast_or_gc_samples, 'assays'):
        # DrillholeDatabase-like structure
        coords_list = []
        values_list = []
        
        for collar in blast_or_gc_samples.collars:
            # Filter by bench if specified
            if config.bench_code:
                # Check if collar is in specified bench (simplified)
                bench_elevation = float(config.bench_code) if config.bench_code.replace('.', '').replace('-', '').isdigit() else None
                if bench_elevation and abs(collar.z - bench_elevation) > 2.0:
                    continue
            
            # Get assays for this hole
            assays = blast_or_gc_samples.get_assays_for(collar.hole_id)
            for assay in assays:
                if config.property_name in assay.values:
                    # Calculate 3D position (simplified: assume vertical)
                    x = collar.x
                    y = collar.y
                    z = collar.z - (assay.depth_from + assay.depth_to) / 2.0
                    
                    coords_list.append([x, y, z])
                    values_list.append(assay.values[config.property_name])
        
        if not coords_list:
            raise ValueError("No samples found matching criteria")
        
        data_coords = np.array(coords_list)
        data_values = np.array(values_list)
    
    elif isinstance(blast_or_gc_samples, dict):
        # Dict with 'coords' and 'values' keys
        data_coords = np.array(blast_or_gc_samples['coords'])
        data_values = np.array(blast_or_gc_samples['values'])
    
    else:
        raise ValueError("Unsupported sample data format")
    
    # Get target coordinates (GC block centers)
    target_coords = gc_grid.get_block_centers()
    
    # Prepare variogram parameters
    variogram_params = config.variogram_model.copy()
    
    # Add anisotropy if search ellipse provided
    if config.search_ellipse:
        variogram_params['anisotropy'] = {
            "azimuth": config.search_ellipse.get("azimuth", 0.0),
            "dip": config.search_ellipse.get("dip", 0.0),
            "major_range": config.search_ellipse.get("major", 50.0),
            "minor_range": config.search_ellipse.get("minor", 50.0),
            "vert_range": config.search_ellipse.get("vertical", 10.0)
        }
    
    # Run kriging
    max_distance = max(
        config.search_ellipse.get("major", 50.0),
        config.search_ellipse.get("minor", 50.0),
        config.search_ellipse.get("vertical", 10.0)
    )
    
    estimates, variances, _ = ordinary_kriging_3d(  # Ignore QA metrics for grade control
        data_coords=data_coords,
        data_values=data_values,
        target_coords=target_coords,
        variogram_params=variogram_params,
        n_neighbors=config.max_samples,
        max_distance=max_distance,
        model_type=variogram_params.get("type", "spherical")
    )
    
    logger.info(f"GC kriging complete: {len(estimates)} blocks estimated")
    
    return GCKrigingResult(
        estimates=estimates,
        variance=variances,
        gc_grid=gc_grid
    )

