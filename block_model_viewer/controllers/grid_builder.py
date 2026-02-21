"""
Grid Builder - Creates PyVista grids from primitive data in main thread.

This module provides functions to create PyVista/VTK objects from primitive
data structures returned by worker threads. All PyVista operations MUST happen
in the main thread to prevent deadlocks and freezes.
"""

import logging
import numpy as np
import pyvista as pv
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def create_grid_from_result(result: Dict[str, Any]) -> Optional[pv.StructuredGrid]:
    """
    Create PyVista StructuredGrid from worker result (primitive data only).
    
    This function runs in the MAIN THREAD and creates PyVista objects safely.
    
    Supports:
    - Kriging results (estimates, variances)
    - Block model builder results (grid_def with edges)
    - Indicator kriging results (probabilities, thresholds)
    
    Args:
        result: Result dict from worker with primitive data (numpy arrays, dicts)
    
    Returns:
        PyVista StructuredGrid/RectilinearGrid or None if creation fails
    """
    if not result.get('_create_grid_in_main_thread', False):
        # Result doesn't need grid creation
        return None
    
    # Check if this is a block model builder result (uses RectilinearGrid)
    if 'grid_def' in result and 'x_edges' in result.get('grid_def', {}):
        return _create_rectilinear_grid_from_builder(result)
    
    try:
        grid_def = result.get('grid_def')
        if not grid_def:
            logger.warning("Cannot create grid: missing grid_def in result")
            return None
        
        origin = grid_def['origin']
        spacing = grid_def['spacing']
        counts = grid_def['counts']
        
        x0, y0, z0 = origin
        dx, dy, dz = spacing
        nx, ny, nz = counts
        
        # Create edge coordinates
        x_edges = np.linspace(x0, x0 + nx * dx, nx + 1)
        y_edges = np.linspace(y0, y0 + ny * dy, ny + 1)
        z_edges = np.linspace(z0, z0 + nz * dz, nz + 1)
        
        # Create RectilinearGrid (main thread - safe)
        grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
        
        # Add primary property
        property_name = result.get('property_name', 'Grade')
        if 'estimates' in result:
            estimates = result['estimates']
            if isinstance(estimates, np.ndarray):
                if estimates.ndim == 3:
                    grid.cell_data[property_name] = estimates.ravel(order='F')
                else:
                    grid.cell_data[property_name] = estimates
        elif 'primary_values' in result:
            grid.cell_data[property_name] = result['primary_values']
        
        # Add variance property if available
        variance_property = result.get('variance_property')
        if variance_property and 'variances' in result:
            variances = result['variances']
            if isinstance(variances, np.ndarray):
                if variances.ndim == 3:
                    grid.cell_data[variance_property] = variances.ravel(order='F')
                else:
                    grid.cell_data[variance_property] = variances
        
        # Add secondary data if available
        secondary_property = result.get('secondary_property')
        if secondary_property and 'secondary_data' in result:
            secondary_data = result['secondary_data']
            if isinstance(secondary_data, np.ndarray):
                grid.cell_data[secondary_property] = secondary_data
        
        # Add additional properties (for Indicator Kriging)
        additional_properties = result.get('additional_properties', {})
        for prop_name, prop_data in additional_properties.items():
            if isinstance(prop_data, np.ndarray):
                grid.cell_data[prop_name] = prop_data
        
        threshold_properties = result.get('threshold_properties', {})
        for prop_name, prop_data in threshold_properties.items():
            if isinstance(prop_data, np.ndarray):
                grid.cell_data[prop_name] = prop_data
        
        # Add metadata
        metadata = result.get('metadata', {})
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                grid.field_data[key] = [value]
            elif isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float, bool)) for v in value):
                grid.field_data[key] = value
        
        logger.info(f"Created PyVista grid in main thread: {grid.n_cells} cells, property '{property_name}'")
        return grid
        
    except Exception as e:
        logger.error(f"Failed to create grid from result: {e}", exc_info=True)
        return None


def _create_rectilinear_grid_from_builder(result: Dict[str, Any]) -> Optional[pv.RectilinearGrid]:
    """
    Create PyVista RectilinearGrid from block model builder result.
    
    This handles the special case where block model builder returns edge coordinates
    instead of origin/spacing/counts.
    
    Args:
        result: Result dict with grid_def containing x_edges, y_edges, z_edges
    
    Returns:
        PyVista RectilinearGrid or None if creation fails
    """
    try:
        grid_def = result.get('grid_def', {})
        info = result.get('info', {})
        
        x_edges = grid_def.get('x_edges')
        y_edges = grid_def.get('y_edges')
        z_edges = grid_def.get('z_edges')
        
        if x_edges is None or y_edges is None or z_edges is None:
            logger.warning("Cannot create RectilinearGrid: missing edge coordinates")
            return None
        
        # Create RectilinearGrid (main thread - safe)
        grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
        
        # Add cell data from info dict
        cell_data = info.get('cell_data', {})
        for prop_name, prop_data in cell_data.items():
            if isinstance(prop_data, np.ndarray):
                grid.cell_data[prop_name] = prop_data
        
        # Add metadata
        grade_col = info.get('grade_col', 'Grade')
        var_col = info.get('var_col')
        
        logger.info(f"Created PyVista RectilinearGrid in main thread: {grid.n_cells} cells")
        return grid
        
    except Exception as e:
        logger.error(f"Failed to create RectilinearGrid from builder result: {e}", exc_info=True)
        return None

