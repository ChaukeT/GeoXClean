"""
Geometry decimation and downsampling utilities for LOD rendering.

Provides functions to reduce mesh complexity and block model resolution
for performance optimization.
"""

import logging
from typing import Optional, Tuple
import numpy as np
import pyvista as pv

from ..models.block_model import BlockModel

logger = logging.getLogger(__name__)


def decimate_block_grid(
    block_model: BlockModel,
    factor: int,
    preserve_properties: bool = True
) -> BlockModel:
    """
    Downsample block model by sampling every Nth block in each direction.
    
    Args:
        block_model: Source BlockModel
        factor: Downsampling factor (e.g., 2 = every 2nd block, 3 = every 3rd block)
        preserve_properties: Whether to preserve property values
        
    Returns:
        New BlockModel with reduced resolution
    """
    if factor <= 1:
        return block_model
    
    try:
        positions = block_model.positions
        if positions is None or len(positions) == 0:
            return block_model
        
        # Create index mask for downsampling
        # Sample every Nth block based on position indices
        # This is a simple approach - more sophisticated methods could use spatial hashing
        
        # Get unique X, Y, Z coordinates
        x_coords = np.unique(positions[:, 0])
        y_coords = np.unique(positions[:, 1])
        z_coords = np.unique(positions[:, 2])
        
        # Downsample coordinates
        x_sampled = x_coords[::factor]
        y_sampled = y_coords[::factor]
        z_sampled = z_coords[::factor]
        
        # Create mask for blocks that match sampled coordinates
        mask = np.zeros(len(positions), dtype=bool)
        for x in x_sampled:
            for y in y_sampled:
                for z in z_sampled:
                    # Find blocks at this coordinate
                    coord_mask = (
                        (np.abs(positions[:, 0] - x) < 1e-6) &
                        (np.abs(positions[:, 1] - y) < 1e-6) &
                        (np.abs(positions[:, 2] - z) < 1e-6)
                    )
                    mask |= coord_mask
        
        # Create downsampled model
        from ..models.block_model import BlockMetadata
        downsampled = BlockModel(metadata=block_model.metadata)
        
        # Set geometry using the correct API
        downsampled_positions = positions[mask]
        if block_model.dimensions is not None:
            downsampled_dimensions = block_model.dimensions[mask]
        else:
            # If no dimensions, create default dimensions
            downsampled_dimensions = np.ones((len(downsampled_positions), 3), dtype=np.float32)
        
        downsampled.set_geometry(downsampled_positions, downsampled_dimensions)
        
        # Copy properties
        if preserve_properties:
            for prop_name in block_model.get_property_names():
                prop_values = block_model.get_property(prop_name)
                if prop_values is not None:
                    downsampled.add_property(prop_name, prop_values[mask])
        
        logger.info(f"Downsampled block model: {len(positions)} -> {downsampled.block_count} blocks (factor={factor})")
        return downsampled
        
    except Exception as e:
        logger.error(f"Error downsampling block model: {e}", exc_info=True)
        return block_model


def decimate_mesh(
    mesh: pv.PolyData,
    target_reduction: float,
    method: str = "quadric"
) -> pv.PolyData:
    """
    Decimate a PyVista mesh using VTK algorithms.
    
    Args:
        mesh: Input mesh (PolyData)
        target_reduction: Target reduction ratio (0.0 = no reduction, 0.9 = 90% reduction)
        method: Decimation method ("quadric" or "pro")
        
    Returns:
        Decimated mesh
    """
    if target_reduction <= 0.0 or mesh.n_cells == 0:
        return mesh
    
    try:
        target_reduction = min(0.95, max(0.0, target_reduction))
        target_cells = max(1, int(mesh.n_cells * (1.0 - target_reduction)))
        
        if target_cells >= mesh.n_cells:
            return mesh
        
        if method == "quadric":
            # Use QuadricDecimation (faster, good for general use)
            decimated = mesh.decimate(target_reduction)
        elif method == "pro":
            # Use ProDecimation (slower, better quality)
            decimated = mesh.decimate_pro(target_reduction)
        else:
            logger.warning(f"Unknown decimation method: {method}, using quadric")
            decimated = mesh.decimate(target_reduction)
        
        logger.debug(f"Decimated mesh: {mesh.n_cells} -> {decimated.n_cells} cells ({target_reduction*100:.1f}% reduction)")
        return decimated
        
    except Exception as e:
        logger.warning(f"Error decimating mesh: {e}", exc_info=True)
        return mesh


def decimate_unstructured_grid(
    grid: pv.UnstructuredGrid,
    target_reduction: float,
    method: str = "quadric"
) -> pv.UnstructuredGrid:
    """
    Decimate an unstructured grid.
    
    Args:
        grid: Input UnstructuredGrid
        target_reduction: Target reduction ratio
        method: Decimation method
        
    Returns:
        Decimated grid
    """
    try:
        # Convert to PolyData for decimation
        poly = grid.extract_surface()
        decimated_poly = decimate_mesh(poly, target_reduction, method)
        
        # Convert back to UnstructuredGrid
        # Note: This is a simplified approach - full conversion may require
        # more sophisticated handling of cell types
        decimated_grid = decimated_poly.cast_to_unstructured_grid()
        
        return decimated_grid
        
    except Exception as e:
        logger.warning(f"Error decimating unstructured grid: {e}", exc_info=True)
        return grid


def get_mesh_complexity(mesh) -> dict:
    """
    Get complexity metrics for a mesh.
    
    Args:
        mesh: PyVista mesh
        
    Returns:
        Dict with complexity metrics
    """
    try:
        return {
            'n_points': mesh.n_points if hasattr(mesh, 'n_points') else 0,
            'n_cells': mesh.n_cells if hasattr(mesh, 'n_cells') else 0,
            'memory_mb': (
                (mesh.n_points * 3 * 4 if hasattr(mesh, 'n_points') else 0) +
                (mesh.n_cells * 8 if hasattr(mesh, 'n_cells') else 0)
            ) / (1024 * 1024) if hasattr(mesh, 'n_points') else 0
        }
    except Exception:
        return {'n_points': 0, 'n_cells': 0, 'memory_mb': 0.0}

