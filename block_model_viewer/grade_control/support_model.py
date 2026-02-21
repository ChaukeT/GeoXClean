"""
Grade Control Support Model (STEP 29)

Define short-term / GC support (SMU blocks) and resample from long-term model.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class GCGridDefinition:
    """
    Definition of Grade Control grid at SMU support.
    
    Attributes:
        origin: Grid origin (x, y, z)
        dx: Block size in X direction
        dy: Block size in Y direction
        dz: Block size in Z direction (typically bench height)
        nx: Number of blocks in X
        ny: Number of blocks in Y
        nz: Number of blocks in Z
        bench_codes: Optional array of bench IDs per z-layer
    """
    origin: Tuple[float, float, float]
    dx: float
    dy: float
    dz: float
    nx: int
    ny: int
    nz: int
    bench_codes: Optional[np.ndarray] = None
    
    def get_block_centers(self) -> np.ndarray:
        """Get array of block center coordinates."""
        x_centers = np.linspace(
            self.origin[0] + self.dx / 2,
            self.origin[0] + (self.nx - 0.5) * self.dx,
            self.nx
        )
        y_centers = np.linspace(
            self.origin[1] + self.dy / 2,
            self.origin[1] + (self.ny - 0.5) * self.dy,
            self.ny
        )
        z_centers = np.linspace(
            self.origin[2] + self.dz / 2,
            self.origin[2] + (self.nz - 0.5) * self.dz,
            self.nz
        )
        
        # Create meshgrid
        xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
        
        # Flatten to (N, 3)
        centers = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        return centers
    
    def get_block_count(self) -> int:
        """Get total number of blocks in grid."""
        return self.nx * self.ny * self.nz
    
    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Get grid bounds (xmin, xmax, ymin, ymax, zmin, zmax)."""
        xmin = self.origin[0]
        xmax = self.origin[0] + self.nx * self.dx
        ymin = self.origin[1]
        ymax = self.origin[1] + self.ny * self.dy
        zmin = self.origin[2]
        zmax = self.origin[2] + self.nz * self.dz
        return (xmin, xmax, ymin, ymax, zmin, zmax)


@dataclass
class GCModel:
    """
    Grade Control model at SMU support.
    
    Attributes:
        grid: GCGridDefinition
        properties: Dictionary mapping property name -> values array
        metadata: Additional metadata
    """
    grid: GCGridDefinition
    properties: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_property(self, name: str) -> Optional[np.ndarray]:
        """Get property by name."""
        return self.properties.get(name)
    
    def add_property(self, name: str, values: np.ndarray) -> None:
        """Add property to model."""
        n_blocks = self.grid.get_block_count()
        if len(values) != n_blocks:
            raise ValueError(f"Property values length ({len(values)}) must match block count ({n_blocks})")
        self.properties[name] = values
    
    def get_property_names(self) -> list[str]:
        """Get list of property names."""
        return list(self.properties.keys())


def derive_gc_grid_from_long_term(
    long_model: Any,
    smu_size: Tuple[float, float, float]
) -> GCGridDefinition:
    """
    Derive GC grid from long-term block model.
    
    Args:
        long_model: BlockModel instance (long-term model)
        smu_size: SMU block size (dx, dy, dz)
        
    Returns:
        GCGridDefinition
    """
    if long_model.positions is None or long_model.dimensions is None:
        raise ValueError("Long-term model must have geometry set")
    
    # Get bounds from long-term model
    bounds = long_model.bounds
    if bounds is None:
        raise ValueError("Long-term model bounds not available")
    
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    # Calculate number of blocks
    dx, dy, dz = smu_size
    nx = int(np.ceil((xmax - xmin) / dx))
    ny = int(np.ceil((ymax - ymin) / dy))
    nz = int(np.ceil((zmax - zmin) / dz))
    
    # Adjust origin to align with grid
    origin_x = xmin
    origin_y = ymin
    origin_z = zmin
    
    return GCGridDefinition(
        origin=(origin_x, origin_y, origin_z),
        dx=dx,
        dy=dy,
        dz=dz,
        nx=nx,
        ny=ny,
        nz=nz
    )


def resample_long_term_to_gc(
    long_model: Any,
    gc_grid: GCGridDefinition,
    method: str = "volume_weighted"
) -> GCModel:
    """
    Resample long-term model to GC grid (initial GC model).
    
    Args:
        long_model: BlockModel instance
        gc_grid: GCGridDefinition
        method: Resampling method ("volume_weighted", "nearest", "average")
        
    Returns:
        GCModel with resampled properties
    """
    gc_model = GCModel(grid=gc_grid)
    
    if long_model.positions is None or long_model.dimensions is None:
        logger.warning("Long-term model has no geometry, returning empty GC model")
        return gc_model
    
    gc_centers = gc_grid.get_block_centers()
    n_gc_blocks = gc_grid.get_block_count()
    
    long_positions = long_model.positions
    long_dimensions = long_model.dimensions
    
    # For each GC block, find overlapping long-term blocks
    if method == "volume_weighted":
        # Volume-weighted average
        for prop_name in long_model.get_property_names():
            prop_values = long_model.get_property(prop_name)
            if prop_values is None:
                continue
            
            gc_values = np.zeros(n_gc_blocks)
            gc_volumes = np.zeros(n_gc_blocks)
            
            for i in range(n_gc_blocks):
                gc_center = gc_centers[i]
                gc_half_dims = np.array([gc_grid.dx, gc_grid.dy, gc_grid.dz]) / 2.0
                
                # Find overlapping long-term blocks
                for j in range(len(long_positions)):
                    long_center = long_positions[j]
                    long_half_dims = long_dimensions[j] / 2.0
                    
                    # Check overlap
                    overlap = True
                    for k in range(3):
                        dist = abs(gc_center[k] - long_center[k])
                        max_dist = gc_half_dims[k] + long_half_dims[k]
                        if dist > max_dist:
                            overlap = False
                            break
                    
                    if overlap:
                        # Calculate overlap volume (simplified: use smaller volume)
                        overlap_vol = min(
                            gc_grid.dx * gc_grid.dy * gc_grid.dz,
                            long_dimensions[j, 0] * long_dimensions[j, 1] * long_dimensions[j, 2]
                        )
                        
                        if not np.isnan(prop_values[j]):
                            gc_values[i] += prop_values[j] * overlap_vol
                            gc_volumes[i] += overlap_vol
                
                if gc_volumes[i] > 0:
                    gc_values[i] /= gc_volumes[i]
                else:
                    gc_values[i] = np.nan
            
            gc_model.add_property(prop_name, gc_values)
    
    elif method == "nearest":
        # Nearest neighbor
        from scipy.spatial import cKDTree
        
        tree = cKDTree(long_positions)
        
        for prop_name in long_model.get_property_names():
            prop_values = long_model.get_property(prop_name)
            if prop_values is None:
                continue
            
            _, indices = tree.query(gc_centers, k=1)
            gc_values = prop_values[indices]
            gc_model.add_property(prop_name, gc_values)
    
    else:  # average
        # Simple average of nearby blocks
        from scipy.spatial import cKDTree
        
        tree = cKDTree(long_positions)
        search_radius = max(gc_grid.dx, gc_grid.dy, gc_grid.dz) * 1.5
        
        for prop_name in long_model.get_property_names():
            prop_values = long_model.get_property(prop_name)
            if prop_values is None:
                continue
            
            gc_values = np.zeros(n_gc_blocks)
            
            for i in range(n_gc_blocks):
                neighbors = tree.query_ball_point(gc_centers[i], search_radius)
                if neighbors:
                    neighbor_values = prop_values[neighbors]
                    valid_values = neighbor_values[~np.isnan(neighbor_values)]
                    if len(valid_values) > 0:
                        gc_values[i] = np.mean(valid_values)
                    else:
                        gc_values[i] = np.nan
                else:
                    gc_values[i] = np.nan
            
            gc_model.add_property(prop_name, gc_values)
    
    logger.info(f"Resampled long-term model to GC grid: {n_gc_blocks} blocks")
    return gc_model

