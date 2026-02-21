"""
GC Simulation Engine (STEP 29)

Conditional simulation at GC support for dilution/ore loss risk quantification.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class GCSimulationConfig:
    """
    Configuration for GC simulation.
    
    Attributes:
        property_name: Property to simulate
        variogram_model: Variogram model parameters
        n_realizations: Number of realizations
        random_seed: Random seed for reproducibility
        gc_grid: GCGridDefinition
        min_samples: Minimum samples for kriging
        max_samples: Maximum samples for kriging
    """
    property_name: str
    variogram_model: Dict[str, Any]
    n_realizations: int = 10
    random_seed: Optional[int] = None
    gc_grid: Any = None  # GCGridDefinition
    min_samples: int = 4
    max_samples: int = 16


@dataclass
class GCSimulationResult:
    """
    Result from GC simulation.
    
    Attributes:
        realization_names: List of property keys added to GCModel
        gc_grid: GCGridDefinition
        metadata: Additional metadata
    """
    realization_names: list[str] = field(default_factory=list)
    gc_grid: Any = None  # GCGridDefinition
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_gc_sgsim(
    samples: Any,
    gc_grid: Any,
    config: GCSimulationConfig
) -> GCSimulationResult:
    """
    Run Sequential Gaussian Simulation for Grade Control.
    
    Args:
        samples: Sample data (same format as run_gc_ok)
        gc_grid: GCGridDefinition
        config: GCSimulationConfig
        
    Returns:
        GCSimulationResult
    """
    from ..models.sgsim3d import run_sgsim_simulation, SGSIMParameters
    
    # Extract sample coordinates and values (same as GC kriging)
    if hasattr(samples, 'collars') and hasattr(samples, 'assays'):
        coords_list = []
        values_list = []
        
        for collar in samples.collars:
            assays = samples.get_assays_for(collar.hole_id)
            for assay in assays:
                if config.property_name in assay.values:
                    x = collar.x
                    y = collar.y
                    z = collar.z - (assay.depth_from + assay.depth_to) / 2.0
                    
                    coords_list.append([x, y, z])
                    values_list.append(assay.values[config.property_name])
        
        data_coords = np.array(coords_list)
        data_values = np.array(values_list)
    
    elif isinstance(samples, dict):
        data_coords = np.array(samples['coords'])
        data_values = np.array(samples['values'])
    
    else:
        raise ValueError("Unsupported sample data format")
    
    # Get target coordinates
    target_coords = gc_grid.get_block_centers()
    
    # Prepare SGSIM parameters
    variogram_params = config.variogram_model
    
    # Create grid specification for SGSIM
    bounds = gc_grid.get_bounds()
    
    # SGSIMParameters requires specific fields (note: uses 'nugget' and 'sill', not 'variogram_nugget')
    anisotropy = variogram_params.get("anisotropy", {})
    if not isinstance(anisotropy, dict):
        anisotropy = {}
    
    sgsim_params = SGSIMParameters(
        nx=gc_grid.nx,
        ny=gc_grid.ny,
        nz=gc_grid.nz,
        xmin=bounds[0],
        ymin=bounds[2],
        zmin=bounds[4],
        xinc=gc_grid.dx,
        yinc=gc_grid.dy,
        zinc=gc_grid.dz,
        variogram_type=variogram_params.get("type", "spherical"),
        range_major=anisotropy.get("major_range", variogram_params.get("range", 50.0)),
        range_minor=anisotropy.get("minor_range", variogram_params.get("range", 50.0)),
        range_vert=anisotropy.get("vert_range", variogram_params.get("range", 50.0)),
        azimuth=anisotropy.get("azimuth", 0.0),
        dip=anisotropy.get("dip", 0.0),
        nugget=variogram_params.get("nugget", 0.0),
        sill=variogram_params.get("sill", 1.0),
        min_neighbors=config.min_samples if hasattr(config, 'min_samples') else 4,
        max_neighbors=config.max_samples if hasattr(config, 'max_samples') else 16,
        max_search_radius=100.0,
        nreal=config.n_realizations,
        seed=config.random_seed
    )
    
    # Run SGSIM (returns all realizations at once)
    # SGSIM returns shape (nreal, nz, ny, nx)
    realizations = run_sgsim_simulation(
        data_coords=data_coords,
        data_values=data_values,
        params=sgsim_params
    )
    
    # Extract realization names
    realization_names = []
    for i in range(config.n_realizations):
        prop_name = f"{config.property_name}_real_{i+1}"
        realization_names.append(prop_name)
    
    logger.info(f"GC simulation complete: {config.n_realizations} realizations")
    
    return GCSimulationResult(
        realization_names=realization_names,
        gc_grid=gc_grid,
        metadata={
            "n_realizations": config.n_realizations,
            "property_name": config.property_name
        }
    )

