"""
3D interpolation of geotechnical parameters.

Provides functions to interpolate rock-mass properties from point measurements
to block model or 3D grid using IDW or kriging.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from scipy.spatial.distance import cdist

from .dataclasses import RockMassPoint, RockMassGrid
from .rock_mass_model import RockMassModel
from ..models.block_model import BlockModel

logger = logging.getLogger(__name__)


def interpolate_to_block_model(
    drill_data: RockMassModel,
    block_model: BlockModel,
    params: Dict[str, Any]
) -> RockMassGrid:
    """
    Interpolate geotechnical parameters to block model.
    
    Args:
        drill_data: RockMassModel with point measurements
        block_model: Target BlockModel
        params: Interpolation parameters:
            - variable: Property to interpolate ('RQD', 'Q', 'RMR', 'GSI')
            - method: 'IDW' or 'OK' (ordinary kriging)
            - power: Power parameter for IDW (default 2.0)
            - max_neighbors: Maximum neighbors for IDW (default 20)
            - search_radius: Search radius (optional)
    
    Returns:
        RockMassGrid with interpolated values
    """
    variable = params.get('variable', 'RMR').upper()
    method = params.get('method', 'IDW').upper()
    
    if block_model.positions is None:
        raise ValueError("Block model must have positions set")
    
    block_positions = block_model.positions
    n_blocks = len(block_positions)
    
    # Get source data
    source_coords = drill_data.get_coordinates()
    source_values = drill_data.get_property_array(variable)
    
    # Filter out NaN values
    valid_mask = ~np.isnan(source_values)
    if not np.any(valid_mask):
        raise ValueError(f"No valid {variable} measurements found")
    
    source_coords = source_coords[valid_mask]
    source_values = source_values[valid_mask]
    
    logger.info(f"Interpolating {variable} using {method}: {len(source_values)} points -> {n_blocks} blocks")
    
    if method == 'IDW':
        interpolated = _interpolate_idw(
            source_coords, source_values, block_positions, params
        )
    elif method == 'OK':
        # Try to use existing kriging if available
        interpolated = _interpolate_kriging(
            source_coords, source_values, block_positions, params
        )
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Create RockMassGrid
    grid_definition = {
        'n_blocks': n_blocks,
        'bounds': block_model.bounds,
        'block_count': block_model.block_count
    }
    
    grid = RockMassGrid(grid_definition=grid_definition)
    
    # Set the interpolated property
    if variable == 'RQD':
        grid.rqd = interpolated
    elif variable == 'Q':
        grid.q = interpolated
    elif variable == 'RMR':
        grid.rmr = interpolated
    elif variable == 'GSI':
        grid.gsi = interpolated
    
    # Compute quality categories
    grid.compute_quality_categories()
    
    logger.info(f"Interpolation complete: {variable} mean={np.nanmean(interpolated):.2f}")
    
    return grid


def _interpolate_idw(
    source_coords: np.ndarray,
    source_values: np.ndarray,
    target_coords: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Inverse Distance Weighting interpolation.
    
    Args:
        source_coords: Source point coordinates (n_source, 3)
        source_values: Source values (n_source,)
        target_coords: Target coordinates (n_target, 3)
        params: Parameters dict with 'power' and 'max_neighbors'
    
    Returns:
        Interpolated values (n_target,)
    """
    power = params.get('power', 2.0)
    max_neighbors = params.get('max_neighbors', 20)
    search_radius = params.get('search_radius', None)
    
    n_target = len(target_coords)
    interpolated = np.full(n_target, np.nan)
    
    # Compute distances
    distances = cdist(target_coords, source_coords)
    
    for i in range(n_target):
        dists = distances[i, :]
        
        # Apply search radius if specified
        if search_radius is not None:
            mask = dists <= search_radius
            if not np.any(mask):
                continue
            dists = dists[mask]
            values = source_values[mask]
        else:
            values = source_values
        
        # Use nearest neighbors
        if len(dists) > max_neighbors:
            nearest_idx = np.argsort(dists)[:max_neighbors]
            dists = dists[nearest_idx]
            values = values[nearest_idx]
        
        # Avoid division by zero
        dists = np.maximum(dists, 1e-6)
        
        # Compute weights
        weights = 1.0 / (dists ** power)
        weights_sum = np.sum(weights)
        
        if weights_sum > 0:
            interpolated[i] = np.sum(weights * values) / weights_sum
    
    return interpolated


def _interpolate_kriging(
    source_coords: np.ndarray,
    source_values: np.ndarray,
    target_coords: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Ordinary kriging interpolation (fallback to IDW if kriging not available).
    
    Args:
        source_coords: Source point coordinates
        source_values: Source values
        target_coords: Target coordinates
        params: Kriging parameters
    
    Returns:
        Interpolated values
    """
    # Try to use existing kriging module
    try:
        from ..models.simple_kriging3d import simple_kriging_3d, SKParameters
        
        # Create kriging parameters with correct field names
        sk_params = SKParameters(
            global_mean=np.nanmean(source_values),  # Use data mean
            variogram_type='spherical',  # Correct: variogram_type not variogram_model
            sill=params.get('sill', 1.0),  # Correct: sill not variogram_sill
            nugget=params.get('nugget', 0.1),  # Correct: nugget not variogram_nugget
            range_major=params.get('range', 100.0),  # Correct: range_major not variogram_range
            range_minor=params.get('range', 100.0),  # Default to isotropic
            range_vert=params.get('range', 100.0),
            max_search_radius=params.get('search_radius', 200.0),  # Correct: max_search_radius not search_radius
            ndmax=params.get('max_neighbors', 20)  # Correct: ndmax not max_neighbors
        )
        
        # Run kriging
        estimates, variances, nn, diagnostics = simple_kriging_3d(
            data_coords=source_coords,
            data_values=source_values,
            target_coords=target_coords,
            params=sk_params
        )
        
        return estimates
        
    except ImportError:
        logger.warning("Kriging module not available, falling back to IDW")
        return _interpolate_idw(source_coords, source_values, target_coords, params)

