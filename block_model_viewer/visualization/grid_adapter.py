"""
Grid Adapter - Converts GridPayload to PyVista grids.

This adapter handles all PyVista structured grid creation logic.
"""

import logging
import numpy as np
import pyvista as pv
from typing import Tuple, Optional

from .render_payloads import GridPayload

logger = logging.getLogger(__name__)


class GridAdapter:
    """Adapter for converting GridPayload to PyVista structured grids."""
    
    def to_pv_grid(self, payload: GridPayload) -> Tuple[pv.DataSet, Optional[str]]:
        """
        Convert GridPayload to PyVista grid.
        
        Args:
            payload: Grid payload
            
        Returns:
            Tuple of (pv.DataSet grid, scalar field name)
        """
        try:
            grid_data = payload.grid
            
            # Handle different grid input formats
            if isinstance(grid_data, tuple) and len(grid_data) == 3:
                # Tuple of (x, y, z) coordinate arrays
                x, y, z = grid_data
                grid = pv.StructuredGrid(x, y, z)
            elif isinstance(grid_data, np.ndarray):
                if payload.origin is not None and payload.spacing is not None and payload.dimensions is not None:
                    # Uniform grid
                    grid = pv.UniformGrid()
                    grid.origin = payload.origin
                    grid.spacing = payload.spacing
                    grid.dimensions = payload.dimensions
                else:
                    # Try to infer from array shape
                    if grid_data.ndim == 3:
                        # Assume it's a 3D array - create structured grid
                        nx, ny, nz = grid_data.shape
                        x = np.arange(nx)
                        y = np.arange(ny)
                        z = np.arange(nz)
                        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                        grid = pv.StructuredGrid(X, Y, Z)
                    else:
                        raise ValueError(f"Cannot infer grid structure from array shape {grid_data.shape}")
            else:
                raise ValueError(f"Unsupported grid data type: {type(grid_data)}")
            
            # Add scalars
            scalar_name = None
            if payload.scalars is not None:
                scalars = np.asarray(payload.scalars)
                scalar_name = payload.name
                grid[scalar_name] = scalars.ravel(order='F')  # Fortran order for structured grids
            
            return grid, scalar_name
            
        except Exception as e:
            logger.error(f"Error converting grid payload to PyVista: {e}", exc_info=True)
            raise
    
    def create_actor(self, payload: GridPayload, plotter: pv.Plotter) -> pv.Actor:
        """
        Create and add actor to plotter from payload.
        
        Args:
            payload: Grid payload
            plotter: PyVista plotter
            
        Returns:
            PyVista actor
        """
        grid, scalar_name = self.to_pv_grid(payload)
        
        # Add to plotter
        actor = plotter.add_mesh(
            grid,
            scalars=scalar_name,
            opacity=payload.metadata.get('opacity', 1.0),
            show_edges=False,
            cmap=payload.metadata.get('colormap', 'viridis'),
            show_scalar_bar=False  # LegendManager handles scalar bars
        )
        
        # Set visibility
        actor.SetVisibility(1 if payload.visible else 0)
        
        return actor

