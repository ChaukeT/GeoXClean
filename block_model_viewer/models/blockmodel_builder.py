"""
Block Model Builder Module

Constructs regular 3D block models from kriging or estimation results.
Supports user-defined block sizes, automatic grid detection, and export to various formats.

Author: Block Model Viewer Development Team
Date: 2025-10-25
"""

import logging
import numpy as np
import pandas as pd
# NOTE: PyVista imports removed - grid creation happens in main thread to prevent worker freezes
from typing import Tuple, Optional, Dict, TYPE_CHECKING, Any
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    import pyvista as pv

logger = logging.getLogger(__name__)


def _convert_density_to_t_m3(density_value: float, unit: str) -> float:
    """
    Convert density value to tonnes per cubic meter.

    Parameters
    ----------
    density_value : float
        Density value in specified unit
    unit : str
        Unit of density value ('t/m³', 'g/cm³', 'kg/m³')

    Returns
    -------
    float
        Density in t/m³
    """
    if unit == 'kg/m³' or unit == 'kg/m' or unit == 'kg/m3':
        # Convert kg/m³ to t/m³ (divide by 1000)
        return density_value / 1000.0
    elif unit == 'g/cm³':
        # g/cm³ is numerically equal to t/m³ for most mining applications
        # (this is a common approximation)
        return density_value
    else:  # 't/m³' or any other unit
        return density_value


def build_block_model(
    df: pd.DataFrame,
    xcol: str = 'X',
    ycol: str = 'Y',
    zcol: str = 'Z',
    grade_col: str = 'Fe_est',
    var_col: Optional[str] = 'Variance',
    xinc: float = 25.0,
    yinc: float = 25.0,
    zinc: float = 10.0,
    origin: Optional[Tuple[float, float, float]] = None,
    extents: Optional[Tuple[float, float, float, float, float, float]] = None,
    max_blocks: int = 100000,
    volume_tonnage_config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Build a regular 3D block model from kriging/estimation results.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with estimation results (must have X, Y, Z coordinates and grade/variance)
    xcol, ycol, zcol : str
        Column names for coordinates
    grade_col : str
        Column name for estimated grade
    var_col : str, optional
        Column name for kriging variance (can be None)
    xinc, yinc, zinc : float
        Block dimensions in metres
    origin : tuple, optional
        (xmin, ymin, zmin) for grid origin. Auto-detected if None.
    extents : tuple, optional
        (xmin, xmax, ymin, ymax, zmin, zmax). Auto-detected if None.
    max_blocks : int
        Maximum number of blocks to prevent memory issues
    volume_tonnage_config : dict, optional
        Configuration for volume and tonnage calculations:
        - 'volume_method': 'From Block Dimensions (DX*DY*DZ)', 'From Column', or 'Calculate from Formula'
        - 'volume_column': column name for volume data (if using 'From Column')
        - 'density_source': 'Constant Value', 'From Column', or 'Calculate from SG'
        - 'density_column': column name for density data (if using 'From Column')
        - 'density_value': constant density value
        - 'density_unit': 't/m³', 'g/cm³', or 'kg/m³'

    Returns
    -------
    block_df : pd.DataFrame
        DataFrame of block centroids with estimated values
    block_grid : pv.StructuredGrid
        PyVista StructuredGrid for visualization
    info : dict
        Information about the block model (dimensions, extents, etc.)
    """
    logger.info(f"Building block model: xinc={xinc}, yinc={yinc}, zinc={zinc}")
    
    # Validate input
    required_cols = [xcol, ycol, zcol, grade_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove NaN values
    valid_mask = ~(df[[xcol, ycol, zcol, grade_col]].isna().any(axis=1))
    df_clean = df[valid_mask].copy()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data points after removing NaN values")
    
    logger.info(f"Using {len(df_clean)}/{len(df)} valid estimation points")
    
    # Determine extents
    if extents is None:
        xmin, xmax = df_clean[xcol].min(), df_clean[xcol].max()
        ymin, ymax = df_clean[ycol].min(), df_clean[ycol].max()
        zmin, zmax = df_clean[zcol].min(), df_clean[zcol].max()
        
        # Add small buffer
        x_buffer = xinc * 0.5
        y_buffer = yinc * 0.5
        z_buffer = zinc * 0.5
        
        xmin -= x_buffer
        xmax += x_buffer
        ymin -= y_buffer
        ymax += y_buffer
        zmin -= z_buffer
        zmax += z_buffer
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = extents
    
    # Calculate number of blocks
    nx = int(np.ceil((xmax - xmin) / xinc))
    ny = int(np.ceil((ymax - ymin) / yinc))
    nz = int(np.ceil((zmax - zmin) / zinc))
    
    total_blocks = nx * ny * nz
    
    logger.info(f"Block model dimensions: {nx} x {ny} x {nz} = {total_blocks} blocks")
    logger.info(f"Extents: X[{xmin:.1f}, {xmax:.1f}], Y[{ymin:.1f}, {ymax:.1f}], Z[{zmin:.1f}, {zmax:.1f}]")
    
    # Note: Block count limit check removed - no limit on block model size
    if total_blocks > max_blocks:
        logger.info(f"Building large block model: {total_blocks} blocks (max_blocks parameter: {max_blocks} - limit check disabled)")
    
    # Determine origin
    if origin is None:
        origin = (xmin + xinc/2, ymin + yinc/2, zmin + zinc/2)
    
    # Generate block centroids
    xs = np.linspace(xmin + xinc/2, xmin + nx*xinc - xinc/2, nx)
    ys = np.linspace(ymin + yinc/2, ymin + ny*yinc - yinc/2, ny)
    zs = np.linspace(zmin + zinc/2, zmin + nz*zinc - zinc/2, nz)
    
    logger.info(f"Generating {nx} x {ny} x {nz} block centroids...")
    # CRITICAL: Use indexing='ij' for (nx, ny, nz) shape
    # PyVista RectilinearGrid expects cell data in Fortran order (x varies fastest)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    
    # Flatten to centroids using Fortran order to match PyVista RectilinearGrid cell ordering
    # RectilinearGrid cell order: x varies fastest, then y, then z
    centroids = np.column_stack([gx.ravel(order='F'), gy.ravel(order='F'), gz.ravel(order='F')])
    
    logger.info(f"Assigning grades to {len(centroids)} blocks using nearest-neighbor...")
    
    # Build KDTree from estimation points
    xyz_est = df_clean[[xcol, ycol, zcol]].values
    vals_est = df_clean[grade_col].values
    
    tree = cKDTree(xyz_est)
    distances, idx = tree.query(centroids, k=1)
    
    # Assign grades
    est_grades = vals_est[idx]
    
    # AUDIT GATE: Validate block count matches after grade assignment
    if len(est_grades) != len(centroids):
        raise ValueError(
            f"CRITICAL: Block count mismatch after grade assignment. "
            f"Expected {len(centroids)} grades, got {len(est_grades)}. "
            f"This indicates a data integrity issue."
        )
    
    # Assign variances if available
    if var_col is not None and var_col in df_clean.columns:
        variances_est = df_clean[var_col].values
        est_variances = variances_est[idx]
        # Validate variance array size
        if len(est_variances) != len(centroids):
            raise ValueError(
                f"CRITICAL: Variance array size mismatch. "
                f"Expected {len(centroids)} variances, got {len(est_variances)}."
            )
    else:
        est_variances = np.full(len(centroids), np.nan)
    
    # Log assignment statistics
    logger.info(f"Grade assignment complete:")
    logger.info(f"  Mean distance to nearest point: {distances.mean():.2f} m")
    logger.info(f"  Max distance to nearest point: {distances.max():.2f} m")
    logger.info(f"  Grade range: [{np.nanmin(est_grades):.2f}, {np.nanmax(est_grades):.2f}]")
    logger.info(f"  Block count validation: PASSED ({len(centroids)} blocks)")
    
    # Create block DataFrame
    block_df = pd.DataFrame({
        'XC': centroids[:, 0],
        'YC': centroids[:, 1],
        'ZC': centroids[:, 2],
        'XINC': xinc,
        'YINC': yinc,
        'ZINC': zinc,
        grade_col: est_grades,
        'DISTANCE': distances  # Distance to nearest kriging point
    })

    # Add canonical X/Y/Z aliases for registry validation and downstream panels
    # (DataRegistrySimple currently requires X/Y/Z to accept DataFrame block models)
    if 'X' not in block_df.columns:
        block_df['X'] = block_df['XC']
    if 'Y' not in block_df.columns:
        block_df['Y'] = block_df['YC']
    if 'Z' not in block_df.columns:
        block_df['Z'] = block_df['ZC']
    
    if var_col is not None and var_col in df_clean.columns:
        block_df[var_col] = est_variances
    
    # Calculate block volumes and tonnage based on configuration
    volume_tonnage_config = volume_tonnage_config or {}

    # Calculate volume
    volume_method = volume_tonnage_config.get('volume_method', 'From Block Dimensions (DX*DY*DZ)')

    if volume_method == 'From Column':
        volume_column = volume_tonnage_config.get('volume_column')
        if volume_column and volume_column in df.columns:
            # Use volume from column - need to assign to each block
            logger.warning("Volume from column not yet implemented - using block dimensions")
            block_df['VOLUME'] = xinc * yinc * zinc
        else:
            logger.warning(f"Volume column '{volume_column}' not found - using block dimensions")
            block_df['VOLUME'] = xinc * yinc * zinc
    elif volume_method == 'Calculate from Formula':
        # Placeholder for future formula-based volume calculation
        logger.warning("Formula-based volume calculation not yet implemented - using block dimensions")
        block_df['VOLUME'] = xinc * yinc * zinc
    else:  # 'From Block Dimensions (DX*DY*DZ)'
        block_df['VOLUME'] = xinc * yinc * zinc

    # Calculate tonnage based on density configuration
    density_source = volume_tonnage_config.get('density_source', 'Constant Value')
    density_unit = volume_tonnage_config.get('density_unit', 't/m³')

    if density_source == 'From Column':
        density_column = volume_tonnage_config.get('density_column')
        if density_column and density_column in df.columns:
            # Use density from column - assign to blocks using nearest neighbor
            density_values = df[density_column].values
            block_df['DENSITY'] = density_values[idx]  # Use same indices as grade assignment
        else:
            logger.warning(f"Density column '{density_column}' not found - using constant density")
            density_value = volume_tonnage_config.get('density_value', 2.7)
            block_df['DENSITY'] = _convert_density_to_t_m3(density_value, density_unit)
    elif density_source == 'Calculate from SG':
        # Placeholder for specific gravity calculations
        logger.warning("SG-based density calculation not yet implemented - using constant density")
        density_value = volume_tonnage_config.get('density_value', 2.7)
        block_df['DENSITY'] = _convert_density_to_t_m3(density_value, density_unit)
    else:  # 'Constant Value'
        density_value = volume_tonnage_config.get('density_value', 2.7)
        block_df['DENSITY'] = _convert_density_to_t_m3(density_value, density_unit)

    # Calculate tonnage
    block_df['TONNAGE'] = block_df['VOLUME'] * block_df['DENSITY']
    
    logger.info("Creating PyVista RectilinearGrid (proper block model)...")
    
    # Create EDGE coordinates for blocks (not centroids)
    # Industry standard: blocks are defined by their edges/corners
    x_edges = np.linspace(xmin, xmin + nx*xinc, nx + 1)
    y_edges = np.linspace(ymin, ymin + ny*yinc, ny + 1)
    z_edges = np.linspace(zmin, zmin + nz*zinc, nz + 1)
    
    # Prepare grid definition for PyVista grid creation (happens in main thread)
    # CRITICAL: Don't create PyVista objects in worker thread - causes freezes
    grid_def = {
        'x_edges': x_edges,
        'y_edges': y_edges,
        'z_edges': z_edges,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'total_blocks': total_blocks
    }
    
    # Prepare cell data (properties) - primitive numpy arrays only
    cell_data = {
        grade_col: est_grades,
        'DISTANCE': distances
    }
    
    if var_col is not None and var_col in df_clean.columns:
        cell_data[var_col] = est_variances
    
    logger.info(f"Prepared grid data: {total_blocks} blocks (PyVista grid creation deferred to main thread)")
    
    # Store metadata
    info = {
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'total_blocks': total_blocks,
        'xinc': xinc,
        'yinc': yinc,
        'zinc': zinc,
        'origin': origin,
        'extents': (xmin, xmax, ymin, ymax, zmin, zmax),
        'grade_col': grade_col,
        'var_col': var_col,
        'mean_distance': distances.mean(),
        'max_distance': distances.max(),
        'grade_range': (np.nanmin(est_grades), np.nanmax(est_grades)),
        'n_estimation_points': len(df_clean),
        'grid_def': grid_def,  # Grid definition for PyVista creation
        'cell_data': cell_data,  # Cell data arrays
        '_create_grid_in_main_thread': True  # Flag to create PyVista grid in main thread
    }
    
    logger.info(f"Block model built successfully: {total_blocks} blocks")
    
    return block_df, grid_def, info


def suggest_block_sizes(
    df: pd.DataFrame,
    xcol: str = 'X',
    ycol: str = 'Y',
    zcol: str = 'Z',
    max_blocks: int = 100000
) -> Tuple[float, float, float]:
    """
    Suggest reasonable block sizes based on data extents and target block count.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with coordinate columns
    xcol, ycol, zcol : str
        Column names for coordinates
    max_blocks : int
        Target maximum number of blocks
    
    Returns
    -------
    xinc, yinc, zinc : float
        Suggested block dimensions
    """
    # Get data extents
    xmin, xmax = df[xcol].min(), df[xcol].max()
    ymin, ymax = df[ycol].min(), df[ycol].max()
    zmin, zmax = df[zcol].min(), df[zcol].max()
    
    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin
    
    # Calculate aspect ratios
    aspect_xy = y_range / x_range if x_range > 0 else 1.0
    aspect_xz = z_range / x_range if x_range > 0 else 0.5
    
    # Target blocks per dimension (cube root of max_blocks)
    target_n = max_blocks ** (1/3)
    
    # Distribute blocks proportionally
    nx = target_n
    ny = target_n * aspect_xy
    nz = target_n * aspect_xz
    
    # Calculate block sizes
    xinc = x_range / nx if nx > 0 else 25.0
    yinc = y_range / ny if ny > 0 else 25.0
    zinc = z_range / nz if nz > 0 else 10.0
    
    # Round to nice numbers
    xinc = round(xinc / 5) * 5  # Round to nearest 5
    yinc = round(yinc / 5) * 5
    zinc = round(zinc / 2.5) * 2.5  # Round to nearest 2.5
    
    # Ensure minimum sizes
    xinc = max(xinc, 5.0)
    yinc = max(yinc, 5.0)
    zinc = max(zinc, 2.5)
    
    logger.info(f"Suggested block sizes: xinc={xinc:.1f}, yinc={yinc:.1f}, zinc={zinc:.1f}")
    
    return xinc, yinc, zinc


def export_block_model(
    block_df: pd.DataFrame,
    output_path: str,
    format: str = 'csv'
) -> None:
    """
    Export block model to file.
    
    Parameters
    ----------
    block_df : pd.DataFrame
        Block model DataFrame
    output_path : str
        Output file path
    format : str
        Export format: 'csv' or 'excel'
    """
    if format.lower() == 'csv':
        block_df.to_csv(output_path, index=False)
        logger.info(f"Exported block model to CSV: {output_path}")
    elif format.lower() == 'excel':
        block_df.to_excel(output_path, index=False, engine='openpyxl')
        logger.info(f"Exported block model to Excel: {output_path}")
    else:
        raise ValueError(f"Unsupported export format: {format}")


def export_block_grid(
    block_grid: "pv.StructuredGrid",
    output_path: str
) -> None:
    """
    Export PyVista block grid to VTK format.
    
    Parameters
    ----------
    block_grid : pv.StructuredGrid
        PyVista grid
    output_path : str
        Output file path (should end with .vtk or .vts)
    """
    block_grid.save(output_path)
    logger.info(f"Exported block grid to VTK: {output_path}")
