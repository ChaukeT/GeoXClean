"""
Unified Simulation Interface
============================

All simulation methods must implement:
    simulate(df, variogram, grid_def, params) -> StructuredGrid

with:
    - grid using pv.StructuredGrid
    - grid.cell_data[property_name] = values
    - metadata: {"method": "SGSIM"} etc.
"""

import logging
import numpy as np
import pyvista as pv
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GridDefinition:
    """Grid definition for simulation."""
    origin: Tuple[float, float, float]  # (xmin, ymin, zmin)
    spacing: Tuple[float, float, float]  # (dx, dy, dz)
    counts: Tuple[int, int, int]  # (nx, ny, nz)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid shape (nz, ny, nx) - standard geostatistics convention."""
        return (self.counts[2], self.counts[1], self.counts[0])
    
    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return self.counts[0] * self.counts[1] * self.counts[2]


@dataclass
class VariogramModel:
    """Unified variogram model structure."""
    model_type: str  # 'spherical', 'exponential', 'gaussian'
    range_: float  # Major range
    sill: float  # Total sill
    range_minor: Optional[float] = None  # Minor range (for anisotropy)
    range_vert: Optional[float] = None  # Vertical range (for anisotropy)
    nugget: float = 0.0  # Nugget effect
    azimuth: float = 0.0  # Azimuth angle (degrees)
    dip: float = 0.0  # Dip angle (degrees)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'model_type': self.model_type,
            'range': self.range_,
            'range_minor': self.range_minor or self.range_,
            'range_vert': self.range_vert or self.range_,
            'sill': self.sill,
            'nugget': self.nugget,
            'azimuth': self.azimuth,
            'dip': self.dip,
            'anisotropy': {
                'azimuth': self.azimuth,
                'dip': self.dip,
                'major_range': self.range_,
                'minor_range': self.range_minor or self.range_,
                'vert_range': self.range_vert or self.range_
            } if self.range_minor is not None or self.range_vert is not None else None
        }


def create_structured_grid(
    values: np.ndarray,
    grid_def: GridDefinition,
    property_name: str = "Grade",
    metadata: Optional[Dict[str, Any]] = None
) -> pv.StructuredGrid:
    """
    Create PyVista StructuredGrid from simulation results.
    
    This is the unified output format for all simulation methods.
    
    Args:
        values: 3D array (nz, ny, nx) or flattened array (n_cells,)
        grid_def: Grid definition
        property_name: Name of the property
        metadata: Optional metadata dict
    
    Returns:
        PyVista StructuredGrid with cell_data[property_name] = values
    """
    # Ensure values are in correct shape
    if values.ndim == 1:
        # Flattened array - reshape to grid
        if len(values) != grid_def.n_cells:
            raise ValueError(
                f"Values length {len(values)} doesn't match grid size {grid_def.n_cells}"
            )
        values_3d = values.reshape(grid_def.shape, order='F')
    elif values.ndim == 3:
        # Already 3D - check shape
        if values.shape != grid_def.shape:
            # Try to transpose if needed
            if values.shape == tuple(reversed(grid_def.shape)):
                values_3d = np.transpose(values, (2, 1, 0))
            else:
                raise ValueError(
                    f"Values shape {values.shape} doesn't match grid shape {grid_def.shape}"
                )
        else:
            values_3d = values
    else:
        raise ValueError(f"Values must be 1D or 3D, got {values.ndim}D")
    
    # Create edge coordinates for RectilinearGrid
    nx, ny, nz = grid_def.counts
    xmin, ymin, zmin = grid_def.origin
    dx, dy, dz = grid_def.spacing
    
    x_edges = np.linspace(xmin, xmin + nx * dx, nx + 1)
    y_edges = np.linspace(ymin, ymin + ny * dy, ny + 1)
    z_edges = np.linspace(zmin, zmin + nz * dz, nz + 1)
    
    # Create RectilinearGrid (industry standard for block models)
    grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
    
    # Add property as cell data (Fortran order for consistency)
    grid.cell_data[property_name] = values_3d.ravel(order='F')
    
    # Store metadata in grid field data
    if metadata:
        for key, value in metadata.items():
            # Store simple values that can be serialized
            if isinstance(value, (str, int, float, bool)):
                grid.field_data[key] = [value]
            elif isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float, bool)) for v in value):
                grid.field_data[key] = value
    
    logger.info(
        f"Created StructuredGrid: {grid.n_cells} cells, "
        f"property '{property_name}', shape={grid_def.shape}"
    )
    
    return grid


def create_grid_from_coords(
    grid_coords: np.ndarray,
    values: np.ndarray,
    grid_def: GridDefinition,
    property_name: str = "Grade",
    metadata: Optional[Dict[str, Any]] = None
) -> pv.StructuredGrid:
    """
    Create StructuredGrid from grid coordinates and values.
    
    Useful when simulation methods return flattened coordinate arrays.
    
    Args:
        grid_coords: (n_cells, 3) array of grid cell centroids
        values: (n_cells,) array of values
        grid_def: Grid definition
        property_name: Name of the property
        metadata: Optional metadata dict
    
    Returns:
        PyVista StructuredGrid
    """
    if len(grid_coords) != len(values):
        raise ValueError(
            f"Grid coords length {len(grid_coords)} doesn't match values length {len(values)}"
        )
    
    if len(grid_coords) != grid_def.n_cells:
        raise ValueError(
            f"Grid coords length {len(grid_coords)} doesn't match grid size {grid_def.n_cells}"
        )
    
    # Reshape values to grid
    values_3d = values.reshape(grid_def.shape, order='F')
    
    return create_structured_grid(values_3d, grid_def, property_name, metadata)


def add_simulation_to_renderer(
    grid: pv.StructuredGrid,
    property_name: str,
    renderer: Any,
    layer_name: Optional[str] = None,
    update_legend: bool = True
) -> None:
    """
    Add simulation result to renderer and update legend.
    
    This is the unified way to visualize simulation results.
    
    Args:
        grid: PyVista StructuredGrid with simulation results
        property_name: Property name to visualize
        renderer: Renderer instance (must have add_block_model_layer and legend_manager)
        layer_name: Optional custom layer name
        update_legend: Whether to update legend (default True)
    """
    if layer_name is None:
        method = grid.field_data.get('method', ['Simulation'])[0]
        layer_name = f"{method}: {property_name}"
    
    # Add grid to renderer
    renderer.add_block_model_layer(grid, property_name, layer_name=layer_name)
    
    # Update legend
    if update_legend and hasattr(renderer, 'legend_manager') and renderer.legend_manager is not None:
        try:
            # Get data range
            if property_name in grid.cell_data:
                data = grid.cell_data[property_name]
            elif property_name in grid.point_data:
                data = grid.point_data[property_name]
            else:
                logger.warning(f"Property '{property_name}' not found in grid")
                return
            
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                vmin = float(np.nanmin(valid_data))
                vmax = float(np.nanmax(valid_data))
                
                # Get colormap from metadata or use default
                colormap = 'viridis'
                if 'colormap' in grid.field_data:
                    colormap = grid.field_data['colormap'][0]
                
                logger.info(
                    f"Updating legend for '{property_name}': "
                    f"range=[{vmin:.3f}, {vmax:.3f}], colormap={colormap}"
                )
                
                renderer.legend_manager.set_continuous(
                    field=property_name.upper(),
                    vmin=vmin,
                    vmax=vmax,
                    cmap_name=colormap
                )
        except Exception as e:
            logger.warning(f"Could not update legend: {e}", exc_info=True)

