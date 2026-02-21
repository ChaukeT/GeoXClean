"""
Advanced Block Model Builder
============================

Professional-grade 3D block model generation with rotation, sub-blocking, 
solid masking, anisotropic search, and resource classification.

Similar to Leapfrog, Datamine, and Surpac capabilities.

Author: Block Model Viewer Development Team
"""

import numpy as np
import pandas as pd
import pyvista as pv
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
from scipy.spatial import cKDTree
from scipy.ndimage import zoom
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# BLOCK MODEL DEFINITION
# ============================================================================

@dataclass
class BlockModelDefinition:
    """
    Defines a 3D block model with optional rotation.
    
    Attributes:
        origin_x, origin_y, origin_z: Model origin in global coordinates
        nx, ny, nz: Number of blocks in each direction
        xinc, yinc, zinc: Block dimensions in each direction
        rotation: Azimuth rotation in degrees (clockwise from North)
        dip: Dip rotation in degrees (from horizontal)
        plunge: Plunge rotation in degrees
    """
    origin_x: float
    origin_y: float
    origin_z: float
    nx: int
    ny: int
    nz: int
    xinc: float
    yinc: float
    zinc: float
    rotation: float = 0.0  # Azimuth (degrees clockwise from North)
    dip: float = 0.0       # Dip (degrees from horizontal)
    plunge: float = 0.0    # Plunge (degrees)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_extents(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get model extents in global coordinates.
        
        Returns:
            (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        # Calculate corner points in local coordinates
        corners_local = np.array([
            [0, 0, 0],
            [self.nx * self.xinc, 0, 0],
            [0, self.ny * self.yinc, 0],
            [self.nx * self.xinc, self.ny * self.yinc, 0],
            [0, 0, self.nz * self.zinc],
            [self.nx * self.xinc, 0, self.nz * self.zinc],
            [0, self.ny * self.yinc, self.nz * self.zinc],
            [self.nx * self.xinc, self.ny * self.yinc, self.nz * self.zinc]
        ])
        
        # Rotate to global coordinates
        corners_global = inverse_rotate_points(corners_local, self)
        
        # Add origin
        corners_global += np.array([self.origin_x, self.origin_y, self.origin_z])
        
        # Get extents
        xmin, ymin, zmin = corners_global.min(axis=0)
        xmax, ymax, zmax = corners_global.max(axis=0)
        
        return xmin, xmax, ymin, ymax, zmin, zmax


# ============================================================================
# ROTATION UTILITIES (Mining Convention)
# ============================================================================

def get_rotation_matrix(azimuth: float, dip: float, plunge: float) -> np.ndarray:
    """
    Create 3D rotation matrix using mining convention.
    
    Mining convention:
    - Azimuth: Clockwise from North (0° = North, 90° = East)
    - Dip: Angle from horizontal (positive down)
    - Plunge: Rotation about trend axis
    
    Args:
        azimuth: Azimuth rotation in degrees (clockwise from North)
        dip: Dip angle in degrees (from horizontal)
        plunge: Plunge angle in degrees
    
    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    az_rad = np.deg2rad(azimuth)
    dip_rad = np.deg2rad(dip)
    plunge_rad = np.deg2rad(plunge)
    
    # Rotation about Z-axis (azimuth)
    Rz = np.array([
        [np.cos(az_rad), -np.sin(az_rad), 0],
        [np.sin(az_rad), np.cos(az_rad), 0],
        [0, 0, 1]
    ])
    
    # Rotation about Y-axis (dip)
    Ry = np.array([
        [np.cos(dip_rad), 0, np.sin(dip_rad)],
        [0, 1, 0],
        [-np.sin(dip_rad), 0, np.cos(dip_rad)]
    ])
    
    # Rotation about X-axis (plunge)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(plunge_rad), -np.sin(plunge_rad)],
        [0, np.sin(plunge_rad), np.cos(plunge_rad)]
    ])
    
    # Combined rotation: Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    
    return R


def rotate_points(points: np.ndarray, definition: BlockModelDefinition) -> np.ndarray:
    """
    Rotate points from global to local (model) coordinates.
    
    Args:
        points: (N, 3) array of global coordinates
        definition: Block model definition with rotation parameters
    
    Returns:
        (N, 3) array of local coordinates
    """
    # Get rotation matrix
    R = get_rotation_matrix(definition.rotation, definition.dip, definition.plunge)
    
    # Translate to origin
    origin = np.array([definition.origin_x, definition.origin_y, definition.origin_z])
    translated = points - origin
    
    # Rotate
    rotated = translated @ R.T
    
    return rotated


def inverse_rotate_points(points: np.ndarray, definition: BlockModelDefinition) -> np.ndarray:
    """
    Rotate points from local (model) to global coordinates.
    
    Args:
        points: (N, 3) array of local coordinates
        definition: Block model definition with rotation parameters
    
    Returns:
        (N, 3) array of global coordinates
    """
    # Get rotation matrix
    R = get_rotation_matrix(definition.rotation, definition.dip, definition.plunge)
    
    # Rotate back (inverse rotation)
    rotated = points @ R
    
    # Translate to global origin
    origin = np.array([definition.origin_x, definition.origin_y, definition.origin_z])
    global_coords = rotated + origin
    
    return global_coords


# ============================================================================
# BLOCK MODEL GENERATION
# ============================================================================

def generate_blocks(defn: BlockModelDefinition) -> np.ndarray:
    """
    Generate block centroids in global coordinates.
    
    Args:
        defn: Block model definition
    
    Returns:
        (N, 3) array of block centroids in global coordinates
    """
    logger.info(f"Generating block model: {defn.nx}x{defn.ny}x{defn.nz} = {defn.nx*defn.ny*defn.nz} blocks")
    
    # Generate block centroids in local coordinates
    x = np.arange(defn.nx) * defn.xinc + defn.xinc / 2
    y = np.arange(defn.ny) * defn.yinc + defn.yinc / 2
    z = np.arange(defn.nz) * defn.zinc + defn.zinc / 2
    
    # Create 3D grid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten to get all centroids
    centroids_local = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Transform to global coordinates
    centroids_global = inverse_rotate_points(centroids_local, defn)
    
    logger.info(f"Generated {len(centroids_global)} block centroids")
    
    return centroids_global


def create_block_grid(defn: BlockModelDefinition) -> pv.RectilinearGrid:
    """
    Create a PyVista RectilinearGrid for the block model.
    
    Note: RectilinearGrid cannot be rotated. For rotated models,
    use StructuredGrid or UnstructuredGrid.
    
    Args:
        defn: Block model definition
    
    Returns:
        PyVista RectilinearGrid (unrotated)
    """
    # Create edge coordinates (for RectilinearGrid, blocks are defined by edges)
    x_edges = np.linspace(defn.origin_x, defn.origin_x + defn.nx * defn.xinc, defn.nx + 1)
    y_edges = np.linspace(defn.origin_y, defn.origin_y + defn.ny * defn.yinc, defn.ny + 1)
    z_edges = np.linspace(defn.origin_z, defn.origin_z + defn.nz * defn.zinc, defn.nz + 1)
    
    # Create RectilinearGrid
    grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
    
    logger.info(f"Created PyVista grid with {grid.n_cells} cells")
    
    return grid


# ============================================================================
# SOLID MASKING & CLIPPING
# ============================================================================

def mask_blocks_by_solid(block_centroids: np.ndarray, solid_mesh: pv.PolyData) -> np.ndarray:
    """
    Mask blocks by a solid boundary (keep only blocks inside).
    
    Args:
        block_centroids: (N, 3) array of block centroid coordinates
        solid_mesh: PyVista mesh defining the solid boundary
    
    Returns:
        Boolean mask (N,) indicating which blocks are inside
    """
    logger.info(f"Masking {len(block_centroids)} blocks by solid mesh")
    
    # Use PyVista's select_enclosed_points
    points_cloud = pv.PolyData(block_centroids)
    selected = points_cloud.select_enclosed_points(solid_mesh, check_surface=False)
    
    # Get the mask
    mask = selected['SelectedPoints'].astype(bool)
    
    n_inside = mask.sum()
    n_outside = (~mask).sum()
    
    logger.info(f"Masking result: {n_inside} blocks inside, {n_outside} blocks outside")
    
    return mask


# ============================================================================
# ANISOTROPIC SEARCH
# ============================================================================

def apply_anisotropy_transform(points: np.ndarray, 
                               azm: float, dip: float, plunge: float,
                               major: float, semi_major: float, minor: float) -> np.ndarray:
    """
    Apply anisotropic search ellipsoid transformation.
    
    Transforms points so that distance calculations in the transformed
    space correspond to anisotropic distances in the original space.
    
    Args:
        points: (N, 3) array of points
        azm: Azimuth of major axis (degrees)
        dip: Dip of major axis (degrees)
        plunge: Plunge angle (degrees)
        major: Major axis radius
        semi_major: Semi-major axis radius
        minor: Minor axis radius
    
    Returns:
        (N, 3) transformed points
    """
    # Create rotation matrix for search ellipsoid orientation
    R = get_rotation_matrix(azm, dip, plunge)
    
    # Rotate points to ellipsoid coordinates
    rotated = points @ R.T
    
    # Scale by axis ratios to transform ellipsoid to sphere
    scale = np.array([1.0 / major, 1.0 / semi_major, 1.0 / minor])
    transformed = rotated * scale
    
    return transformed


# ============================================================================
# PROPERTY ASSIGNMENT
# ============================================================================

def assign_properties_nearest_neighbor(block_centroids: np.ndarray,
                                      estimation_df: pd.DataFrame,
                                      value_col: str = 'Fe_est',
                                      var_col: str = 'Variance') -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign properties to blocks using nearest neighbor interpolation.
    
    Args:
        block_centroids: (N, 3) array of block centroids
        estimation_df: DataFrame with X, Y, Z and value columns
        value_col: Column name for values (e.g., grade)
        var_col: Column name for variance
    
    Returns:
        Tuple of (values, variances) arrays
    """
    logger.info(f"Assigning properties to {len(block_centroids)} blocks using nearest neighbor")
    
    # Extract estimation points
    est_points = estimation_df[['X', 'Y', 'Z']].values
    est_values = estimation_df[value_col].values
    est_variances = estimation_df[var_col].values if var_col in estimation_df.columns else np.zeros(len(est_values))
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(est_points)
    
    # Find nearest neighbor for each block
    distances, indices = tree.query(block_centroids, k=1)
    
    # Assign values
    values = est_values[indices]
    variances = est_variances[indices]
    
    logger.info(f"Property assignment complete. Mean value: {np.nanmean(values):.3f}")
    
    return values, variances


def assign_properties_idw(block_centroids: np.ndarray,
                         estimation_df: pd.DataFrame,
                         value_col: str = 'Fe_est',
                         var_col: str = 'Variance',
                         max_distance: float = 100.0,
                         n_neighbors: int = 8,
                         power: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign properties using Inverse Distance Weighting (IDW).
    
    Args:
        block_centroids: (N, 3) array of block centroids
        estimation_df: DataFrame with X, Y, Z and value columns
        value_col: Column name for values
        var_col: Column name for variance
        max_distance: Maximum search distance
        n_neighbors: Number of neighbors to use
        power: IDW power parameter (typically 2.0)
    
    Returns:
        Tuple of (values, variances) arrays
    """
    logger.info(f"Assigning properties using IDW (n={n_neighbors}, power={power})")
    
    # Extract estimation points
    est_points = estimation_df[['X', 'Y', 'Z']].values
    est_values = estimation_df[value_col].values
    est_variances = estimation_df[var_col].values if var_col in estimation_df.columns else np.zeros(len(est_values))
    
    # Build KD-tree
    tree = cKDTree(est_points)
    
    # Find nearest neighbors
    distances, indices = tree.query(block_centroids, k=n_neighbors, distance_upper_bound=max_distance)
    
    # Calculate IDW weights
    values = np.zeros(len(block_centroids))
    variances = np.zeros(len(block_centroids))
    
    for i in range(len(block_centroids)):
        # Get valid neighbors (within max_distance)
        valid_mask = distances[i] < np.inf
        valid_distances = distances[i][valid_mask]
        valid_indices = indices[i][valid_mask]
        
        if len(valid_distances) == 0:
            # No neighbors found - assign NaN
            values[i] = np.nan
            variances[i] = np.nan
            continue
        
        # Avoid division by zero for exact matches
        valid_distances[valid_distances == 0] = 1e-10
        
        # Calculate weights
        weights = 1.0 / (valid_distances ** power)
        weights /= weights.sum()
        
        # Weighted average
        values[i] = np.sum(weights * est_values[valid_indices])
        variances[i] = np.sum(weights * est_variances[valid_indices])
    
    logger.info(f"IDW assignment complete. Mean value: {np.nanmean(values):.3f}")
    
    return values, variances


# ============================================================================
# RESOURCE CLASSIFICATION
# ============================================================================

def classify_blocks(block_df: pd.DataFrame,
                   variance_col: str = 'Variance',
                   measured_threshold: float = 0.05,
                   indicated_threshold: float = 0.15) -> pd.DataFrame:
    """
    Classify blocks into resource categories based on variance.
    
    Categories:
    - Measured: Variance < measured_threshold
    - Indicated: Variance < indicated_threshold
    - Inferred: Variance >= indicated_threshold
    
    Args:
        block_df: DataFrame with block data
        variance_col: Column name for variance
        measured_threshold: Threshold for Measured category
        indicated_threshold: Threshold for Indicated category
    
    Returns:
        DataFrame with 'Category' column added
    """
    logger.info("Classifying blocks by variance")
    
    if variance_col not in block_df.columns:
        logger.warning(f"Variance column '{variance_col}' not found. Using default classification.")
        block_df['Category'] = 'Unclassified'
        return block_df
    
    # Define conditions
    conditions = [
        (block_df[variance_col] < measured_threshold),
        (block_df[variance_col] < indicated_threshold),
        (block_df[variance_col] >= indicated_threshold)
    ]
    
    categories = ['Measured', 'Indicated', 'Inferred']
    
    # Apply classification
    block_df['Category'] = np.select(conditions, categories, default='Unclassified')
    
    # Log statistics
    category_counts = block_df['Category'].value_counts()
    logger.info(f"Classification results:\n{category_counts}")
    
    return block_df


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_block_model(block_df: pd.DataFrame, path: str = 'output/block_model.csv'):
    """
    Export block model to CSV.
    
    Args:
        block_df: DataFrame with block data
        path: Output file path
    """
    logger.info(f"Exporting block model to {path}")
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Export
    block_df.to_csv(path, index=False)
    
    logger.info(f"Exported {len(block_df)} blocks to {path}")


def export_to_vtk(grid: pv.RectilinearGrid, path: str = 'output/block_model.vtk'):
    """
    Export block model to VTK format.
    
    Args:
        grid: PyVista grid with block data
        path: Output file path
    """
    logger.info(f"Exporting block model to VTK: {path}")
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save
    grid.save(path)
    
    logger.info(f"Exported grid with {grid.n_cells} cells to {path}")


def export_to_datamine(block_df: pd.DataFrame, path: str = 'output/block_model.dm'):
    """
    Export block model to Datamine-compatible format.
    
    Args:
        block_df: DataFrame with block data
        path: Output file path
    """
    logger.info(f"Exporting block model to Datamine format: {path}")
    
    # Datamine format is essentially CSV with specific column naming
    # XC, YC, ZC for centroids
    # Other columns as needed
    
    dm_df = block_df.copy()
    
    # Rename coordinate columns if needed
    if 'X' in dm_df.columns:
        dm_df.rename(columns={'X': 'XC', 'Y': 'YC', 'Z': 'ZC'}, inplace=True)
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Export
    dm_df.to_csv(path, index=False)
    
    logger.info(f"Exported {len(dm_df)} blocks to Datamine format")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_block_model(grid: pv.RectilinearGrid,
                         property_name: str = 'Fe_est',
                         cmap: str = 'viridis',
                         opacity: float = 1.0,
                         show_edges: bool = True) -> pv.Plotter:
    """
    Visualize block model in 3D using PyVista.
    
    Args:
        grid: PyVista grid with block data
        property_name: Property to visualize
        cmap: Colormap name
        opacity: Transparency (0=transparent, 1=opaque)
        show_edges: Whether to show block edges
    
    Returns:
        PyVista Plotter instance
    """
    logger.info(f"Visualizing block model: property='{property_name}'")
    
    # VIOLATION NOTE: This function creates a standalone plotter for export/utility purposes.
    # It is NOT part of the main viewer rendering system. For main viewer integration,
    # visualization should use the unified Renderer/LegendManager/OverlayManager system.
    # This function is kept for backward compatibility with standalone export utilities.
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add mesh without scalar bar (use custom LegendWidget instead)
    plotter.add_mesh(
        grid,
        scalars=property_name,
        preference='cell',
        cmap=cmap,
        opacity=opacity,
        show_edges=show_edges,
        edge_color='white',
        line_width=0.5,
        show_scalar_bar=False  # LegendManager handles scalar bars in main viewer
    )
    
    # VIOLATION NOTE: Direct axes calls are allowed here as this is a standalone utility,
    # not part of the unified rendering system. For main viewer, use OverlayManager.
    plotter.show_axes()
    plotter.add_axes_at_origin(labels_off=False)
    
    # Set background
    plotter.set_background('white')
    
    logger.info("Visualization ready")
    
    return plotter


















