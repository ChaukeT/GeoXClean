"""
Block Model Mesh Builder - Auto-detection for ImageData vs UnstructuredGrid

This module provides centralized detection logic for choosing between:
- pv.ImageData (UniformGrid) for orthogonal block models
- pv.UnstructuredGrid for rotated, sub-blocked, or irregular models

The detection happens in the renderer, not the controller, ensuring proper
separation of concerns.
"""

import logging
import numpy as np
import pyvista as pv
from typing import Optional, Tuple, Dict, Any

from ..models.block_model import BlockModel

logger = logging.getLogger(__name__)


def is_uniform_grid(block_model: BlockModel, tolerance: float = 1e-6) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Detect if block model is uniform (orthogonal, constant spacing, axis-aligned, full grid).
    
    This is the industry-standard detection algorithm for choosing ImageData vs UnstructuredGrid.
    
    Detection Rules:
    1. Constant spacing: All blocks have same dx, dy, dz
    2. Axis-aligned: No rotation (rotation_matrix is None or identity)
    3. Full grid: No missing cells (len(df) == nx * ny * nz)
    
    Args:
        block_model: BlockModel instance
        tolerance: Tolerance for floating point comparisons
    
    Returns:
        Tuple of (is_uniform: bool, grid_info: dict or None)
        grid_info contains: origin, spacing, dims if uniform, None otherwise
    """
    if block_model.block_count == 0:
        return False, None
    
    positions = block_model.positions
    dimensions = block_model.dimensions
    
    if positions is None or dimensions is None:
        return False, None
    
    # Check 1: Constant spacing
    # Get unique X, Y, Z coordinates
    xs = np.sort(np.unique(positions[:, 0]))
    ys = np.sort(np.unique(positions[:, 1]))
    zs = np.sort(np.unique(positions[:, 2]))
    
    if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
        logger.debug("Not enough unique coordinates for uniform grid detection")
        return False, None
    
    # Calculate spacing differences
    dx_unique = np.unique(np.diff(xs))
    dy_unique = np.unique(np.diff(ys))
    dz_unique = np.unique(np.diff(zs))
    
    # Check if spacing is constant (within tolerance)
    constant_spacing = (
        len(dx_unique) == 1 and
        len(dy_unique) == 1 and
        len(dz_unique) == 1
    )
    
    if not constant_spacing:
        logger.debug(f"Non-constant spacing detected: dx={len(dx_unique)} unique, dy={len(dy_unique)} unique, dz={len(dz_unique)} unique")
        return False, None
    
    # Get spacing values
    dx = float(dx_unique[0])
    dy = float(dy_unique[0])
    dz = float(dz_unique[0])
    
    # Check 2: Axis-aligned (no rotation)
    rotation_matrix = getattr(block_model, '_rotation_matrix', None)
    axis_aligned = True
    
    if rotation_matrix is not None:
        # Check if rotation matrix is identity (within tolerance)
        identity = np.eye(3)
        if not np.allclose(rotation_matrix, identity, atol=tolerance):
            logger.debug(f"Rotation detected: rotation_matrix is not identity")
            axis_aligned = False
    
    if not axis_aligned:
        return False, None
    
    # Check 3: Full grid (no missing cells)
    nx = len(xs)
    ny = len(ys)
    nz = len(zs)
    expected_n = nx * ny * nz
    actual_n = len(positions)
    
    complete_grid = (actual_n == expected_n)
    
    if not complete_grid:
        logger.debug(f"Incomplete grid: expected {expected_n} blocks, got {actual_n} blocks")
        return False, None
    
    # Check 4: Uniform block dimensions (all blocks same size)
    if len(dimensions) > 0:
        unique_dims = np.unique(dimensions, axis=0)
        if len(unique_dims) > 1:
            # Check if dimensions are close enough (within tolerance)
            dim_std = np.std(dimensions, axis=0)
            if np.any(dim_std > tolerance):
                logger.debug(f"Non-uniform block dimensions detected (std: {dim_std})")
                return False, None
    
    # All checks passed - this is a uniform grid
    # Calculate origin (minimum block center positions)
    origin = (
        float(xs[0]),
        float(ys[0]),
        float(zs[0])
    )
    
    spacing = (dx, dy, dz)
    dims = (nx, ny, nz)
    
    grid_info = {
        'origin': origin,
        'spacing': spacing,
        'dims': dims,
        'nx': nx,
        'ny': ny,
        'nz': nz
    }
    
    logger.info(f"✅ UNIFORM GRID DETECTED: {nx}×{ny}×{nz} = {actual_n:,} blocks, spacing=({dx:.2f}, {dy:.2f}, {dz:.2f})")
    
    return True, grid_info


def build_uniform_grid(block_model: BlockModel, grid_info: Dict[str, Any]) -> pv.ImageData:
    """
    Build PyVista ImageData (UniformGrid) from uniform block model.
    
    This is the memory-efficient path for orthogonal block models.
    
    Args:
        block_model: BlockModel instance (must be uniform)
        grid_info: Grid information dict from is_uniform_grid()
    
    Returns:
        PyVista ImageData grid with all properties assigned
    """
    origin = grid_info['origin']
    spacing = grid_info['spacing']
    nx, ny, nz = grid_info['dims']
    
    # Create ImageData grid
    grid = pv.ImageData()
    grid.origin = origin
    grid.spacing = spacing
    # VTK ImageData dimensions are number of points (cells + 1)
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    
    positions = block_model.positions
    properties = block_model.properties
    
    # Calculate grid indices for all positions
    x_indices = np.round((positions[:, 0] - origin[0]) / spacing[0]).astype(np.int32)
    y_indices = np.round((positions[:, 1] - origin[1]) / spacing[1]).astype(np.int32)
    z_indices = np.round((positions[:, 2] - origin[2]) / spacing[2]).astype(np.int32)
    
    # Validate indices
    valid_mask = (
        (x_indices >= 0) & (x_indices < nx) &
        (y_indices >= 0) & (y_indices < ny) &
        (z_indices >= 0) & (z_indices < nz)
    )
    
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        logger.warning(f"{invalid_count} blocks have invalid grid indices (out of bounds)")
        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        z_indices = z_indices[valid_mask]
    
    # Assign properties to grid cells
    for prop_name, prop_values in properties.items():
        if len(prop_values) != len(positions):
            logger.warning(f"Property '{prop_name}' has {len(prop_values)} values, expected {len(positions)}")
            continue
        
        # Filter property values if needed
        if not np.all(valid_mask):
            prop_values = prop_values[valid_mask]
        
        # Initialize array with NaN for missing blocks
        prop_array = np.full((nz, ny, nx), np.nan, dtype=prop_values.dtype)
        
        # Vectorized assignment using advanced indexing
        prop_array[z_indices, y_indices, x_indices] = prop_values
        
        # Add to grid (VTK uses C-order)
        grid.cell_data[prop_name] = prop_array.ravel(order='C')
    
    # CRITICAL FOR PICKING: Store Original_ID mapping for efficient block lookup
    # This enables O(1) picking instead of O(N) search
    # Create sequential IDs matching the original block order
    original_id_array = np.full((nz, ny, nx), -1, dtype=np.int64)
    
    # Create mapping: for each position, store its index in the original positions array
    # We need to map from grid indices back to original block indices
    if np.all(valid_mask):
        # All positions are valid - use sequential indices
        sequential_ids = np.arange(len(positions), dtype=np.int64)
        original_id_array[z_indices, y_indices, x_indices] = sequential_ids
    else:
        # Some positions filtered - need to map valid positions
        valid_positions = positions[valid_mask]
        sequential_ids = np.arange(len(valid_positions), dtype=np.int64)
        # Map back to original indices
        original_indices = np.where(valid_mask)[0]
        original_id_array[z_indices, y_indices, x_indices] = original_indices
    
    grid.cell_data['Original_ID'] = original_id_array.ravel(order='C')
    logger.debug(f"Added Original_ID to ImageData grid for picking ({len(positions)} mappings)")
    
    logger.info(f"Created ImageData grid: {grid.n_cells:,} cells, {len(properties)} properties")
    
    return grid


def build_unstructured_grid(block_model: BlockModel) -> pv.UnstructuredGrid:
    """
    Build PyVista UnstructuredGrid from block model.
    
    This is the fallback path for rotated, sub-blocked, or irregular models.
    
    Args:
        block_model: BlockModel instance
    
    Returns:
        PyVista UnstructuredGrid with all properties assigned
    """
    positions = block_model.positions
    dimensions = block_model.dimensions
    properties = block_model.properties
    
    if positions is None or dimensions is None:
        raise ValueError("BlockModel must have positions and dimensions")
    
    n_blocks = len(positions)
    
    # Calculate half-dimensions for corner computation
    half_dims = dimensions / 2.0
    
    # Generate block corners (8 vertices per block)
    # Using vectorized operations for performance
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=np.float64)
    
    # Expand positions and half_dims for broadcasting
    positions_expanded = positions[:, None, :]  # (n_blocks, 1, 3)
    half_dims_expanded = half_dims[:, None, :]  # (n_blocks, 1, 3)
    
    # Compute all corners at once: (n_blocks, 8, 3)
    corners = positions_expanded + half_dims_expanded * signs[None, :, :]
    
    # Flatten corners for VTK: (n_blocks * 8, 3)
    all_points = corners.reshape(-1, 3)
    
    # Create VTK cells (hexahedrons)
    # Each hexahedron needs: [8, v0, v1, v2, v3, v4, v5, v6, v7]
    cells = []
    cell_types = []
    
    for i in range(n_blocks):
        base_idx = i * 8
        cell = [8] + [base_idx + j for j in range(8)]
        cells.extend(cell)
        cell_types.append(12)  # VTK_HEXAHEDRON
    
    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    
    # Create UnstructuredGrid
    grid = pv.UnstructuredGrid(cells, cell_types, all_points)
    
    # Assign properties
    for prop_name, prop_values in properties.items():
        if len(prop_values) == n_blocks:
            grid.cell_data[prop_name] = prop_values
        else:
            logger.warning(f"Property '{prop_name}' has {len(prop_values)} values, expected {n_blocks}")
    
    logger.info(f"Created UnstructuredGrid: {grid.n_cells:,} cells, {len(properties)} properties")
    
    return grid


def generate_block_model_mesh(block_model: BlockModel, tolerance: float = 1e-6) -> pv.DataSet:
    """
    Generate PyVista mesh from block model with automatic grid type selection.
    
    This is the main entry point for mesh generation. It automatically chooses
    between ImageData and UnstructuredGrid based on the model characteristics.
    
    Args:
        block_model: BlockModel instance
        tolerance: Tolerance for floating point comparisons
    
    Returns:
        PyVista DataSet (either ImageData or UnstructuredGrid)
    
    Raises:
        Exception: If mesh generation fails (with fallback attempt)
    """
    try:
        # Detect if model is uniform
        is_uniform, grid_info = is_uniform_grid(block_model, tolerance)
        
        if is_uniform and grid_info is not None:
            # Use ImageData for uniform grids (memory efficient)
            return build_uniform_grid(block_model, grid_info)
        else:
            # Use UnstructuredGrid for rotated/irregular models
            return build_unstructured_grid(block_model)
    
    except Exception as e:
        logger.exception("Failed to build mesh with auto-detection, falling back to UnstructuredGrid")
        # Fallback: Always use UnstructuredGrid if detection fails
        try:
            return build_unstructured_grid(block_model)
        except Exception as e2:
            logger.exception("Failed to build UnstructuredGrid fallback")
            raise RuntimeError(f"Failed to generate block model mesh: {e2}") from e2

