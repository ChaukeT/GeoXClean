"""
Seismic Hazard Volume Construction

Build 3D hazard grids from seismic events using kernel density or distance-based metrics.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from scipy.spatial.distance import cdist

from .dataclasses import SeismicCatalogue, HazardVolume

logger = logging.getLogger(__name__)


def build_hazard_volume(
    catalog: SeismicCatalogue,
    grid_definition: Dict[str, Any],
    params: Dict[str, Any]
) -> HazardVolume:
    """
    Build seismic hazard volume from catalogue.
    
    Computes hazard index per cell using kernel or distance-based metric:
    - Count of events within radius
    - Weighted by magnitude or energy
    - Optional time decay factor
    
    Args:
        catalog: SeismicCatalogue with events
        grid_definition: Grid definition dict with:
            - 'coordinates': (n_points, 3) array of grid point coordinates
            - 'n_cells' or 'n_points': Number of grid cells
        params: Parameters dict:
            - method: 'distance' or 'kernel' (default: 'distance')
            - radius: Influence radius for distance method (default: 100.0)
            - power: Power law exponent for distance weighting (default: 2.0)
            - magnitude_weight: Weight by magnitude (default: True)
            - energy_weight: Weight by energy if available (default: False)
            - time_decay: Time decay factor (days, default: None for no decay)
            - reference_time: Reference time for decay (default: latest event)
    
    Returns:
        HazardVolume instance
    """
    if not catalog.events:
        raise ValueError("Catalogue is empty")
    
    method = params.get('method', 'distance')
    radius = params.get('radius', 100.0)
    power = params.get('power', 2.0)
    magnitude_weight = params.get('magnitude_weight', True)
    energy_weight = params.get('energy_weight', False)
    time_decay = params.get('time_decay', None)
    reference_time = params.get('reference_time', None)
    
    # Get grid coordinates
    if 'coordinates' in grid_definition:
        grid_coords = np.array(grid_definition['coordinates'])
    else:
        raise ValueError("grid_definition must contain 'coordinates'")
    
    n_cells = len(grid_coords)
    
    # Get event coordinates and attributes
    event_coords = catalog.get_coordinates()
    magnitudes = catalog.get_magnitudes()
    
    # Compute weights
    weights = np.ones(len(catalog.events))
    
    if magnitude_weight:
        # Weight by magnitude (exponential scaling)
        weights *= 10.0 ** (magnitudes - np.min(magnitudes))
    
    if energy_weight:
        # Weight by energy if available
        energies = np.array([e.energy if e.energy else 0.0 for e in catalog.events])
        if np.any(energies > 0):
            weights *= energies / np.max(energies)
    
    # Apply time decay if specified
    if time_decay is not None:
        if reference_time is None:
            reference_time = max(catalog.get_times())
        
        times = catalog.get_times()
        for i, event_time in enumerate(times):
            days_diff = (reference_time - event_time).total_seconds() / 86400.0
            decay_factor = np.exp(-days_diff / time_decay)
            weights[i] *= decay_factor
    
    # Normalize weights
    weights = weights / np.max(weights) if np.max(weights) > 0 else weights
    
    # Compute hazard index
    hazard_index = np.zeros(n_cells)
    
    if method == 'distance':
        # Distance-based method: sum of weighted inverse distance
        for i, grid_point in enumerate(grid_coords):
            distances = cdist([grid_point], event_coords)[0]
            
            # Only consider events within radius
            within_radius = distances <= radius
            if not np.any(within_radius):
                continue
            
            # Compute weighted inverse distance
            inv_distances = 1.0 / (distances[within_radius] + 1e-6)  # Add small epsilon
            inv_distances = inv_distances ** power
            weighted_sum = np.sum(weights[within_radius] * inv_distances)
            
            hazard_index[i] = weighted_sum
    
    elif method == 'kernel':
        # Kernel density estimation (simplified Gaussian kernel)
        kernel_radius = params.get('kernel_radius', radius)
        kernel_sigma = params.get('kernel_sigma', kernel_radius / 3.0)
        
        for i, grid_point in enumerate(grid_coords):
            distances = cdist([grid_point], event_coords)[0]
            
            # Gaussian kernel
            kernel_values = np.exp(-0.5 * (distances / kernel_sigma) ** 2)
            kernel_values[distances > kernel_radius] = 0.0
            
            weighted_sum = np.sum(weights * kernel_values)
            hazard_index[i] = weighted_sum
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize hazard index (0-1 scale)
    if np.max(hazard_index) > 0:
        hazard_index = hazard_index / np.max(hazard_index)
    
    # Get time window and magnitude range
    time_window = catalog.get_time_range()
    mag_range = (float(np.min(magnitudes)), float(np.max(magnitudes)))
    
    # Create hazard volume
    volume = HazardVolume(
        grid_definition=grid_definition,
        hazard_index=hazard_index,
        time_window=time_window,
        magnitude_range=mag_range,
        metadata={
            'method': method,
            'n_events': len(catalog.events),
            'radius': radius,
            'power': power
        }
    )
    
    logger.info(f"Built hazard volume: {n_cells} cells, mean index={np.mean(hazard_index):.4f}")
    
    return volume

