"""
Rockburst Index Engine

Compute rockburst potential index near excavations, stopes, and drives.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from scipy.spatial.distance import cdist

from .dataclasses import HazardVolume, RockburstIndexResult
from ..geotech.dataclasses import RockMassGrid

logger = logging.getLogger(__name__)


def compute_rockburst_index_at_points(
    hazard_volume: HazardVolume,
    rock_mass_grid: Optional[RockMassGrid],
    points: np.ndarray,
    params: Dict[str, Any]
) -> List[RockburstIndexResult]:
    """
    Compute rockburst index at specified points.
    
    Combines:
    - Seismic hazard from hazard_volume
    - Rock mass properties from rock_mass_grid (optional)
    - Distance to excavation surfaces (optional)
    - Stress surrogate (optional)
    
    Args:
        hazard_volume: HazardVolume with seismic hazard indices
        rock_mass_grid: Optional RockMassGrid with rock mass properties
        points: Array of points (n_points, 3) to evaluate
        params: Parameters dict:
            - hazard_weight: Weight for seismic hazard (default: 0.6)
            - rock_mass_weight: Weight for rock mass factor (default: 0.3)
            - stress_weight: Weight for stress factor (default: 0.1)
            - stress_surrogate: Optional stress values array (n_points,)
            - excavation_distance: Optional distance to excavation (n_points,)
            - rmr_threshold: RMR threshold for high rockburst risk (default: 60.0)
    
    Returns:
        List of RockburstIndexResult instances
    """
    n_points = len(points)
    
    # Get weights
    hazard_weight = params.get('hazard_weight', 0.6)
    rock_mass_weight = params.get('rock_mass_weight', 0.3)
    stress_weight = params.get('stress_weight', 0.1)
    
    # Normalize weights
    total_weight = hazard_weight + rock_mass_weight + stress_weight
    hazard_weight /= total_weight
    rock_mass_weight /= total_weight
    stress_weight /= total_weight
    
    # Get hazard indices at points
    hazard_indices = np.zeros(n_points)
    for i, point in enumerate(points):
        hazard_indices[i] = hazard_volume.get_hazard_at_point(point[0], point[1], point[2])
    
    # Get rock mass factors
    rock_mass_factors = np.ones(n_points)
    if rock_mass_grid is not None:
        rmr_values = rock_mass_grid.get_property('RMR')
        if rmr_values is not None:
            # Get RMR at points (simplified: use nearest grid point)
            grid_coords = hazard_volume.grid_definition.get('coordinates')
            if grid_coords is not None:
                grid_coords = np.array(grid_coords)
                distances = cdist(points, grid_coords)
                nearest_indices = np.argmin(distances, axis=1)
                
                # Map RMR to rockburst factor (lower RMR = higher risk)
                rmr_threshold = params.get('rmr_threshold', 60.0)
                for i, nearest_idx in enumerate(nearest_indices):
                    if nearest_idx < len(rmr_values):
                        rmr = rmr_values[nearest_idx]
                        # Normalize: RMR < threshold increases risk
                        if rmr < rmr_threshold:
                            rock_mass_factors[i] = 1.0 - (rmr_threshold - rmr) / rmr_threshold
                        else:
                            rock_mass_factors[i] = 0.5  # Moderate risk
    else:
        # No rock mass data: use default moderate factor
        rock_mass_factors.fill(0.5)
    
    # Get stress factors
    stress_factors = np.ones(n_points)
    stress_surrogate = params.get('stress_surrogate')
    if stress_surrogate is not None:
        stress_values = np.array(stress_surrogate)
        if len(stress_values) == n_points:
            # Normalize stress (higher stress = higher risk)
            if np.max(stress_values) > 0:
                stress_factors = stress_values / np.max(stress_values)
    
    # Compute combined rockburst index
    rockburst_indices = (
        hazard_weight * hazard_indices +
        rock_mass_weight * rock_mass_factors +
        stress_weight * stress_factors
    )
    
    # Clamp to [0, 1]
    rockburst_indices = np.clip(rockburst_indices, 0.0, 1.0)
    
    # Count contributing events (simplified: use hazard index as proxy)
    # In practice, would count events within influence radius
    contributing_events = (hazard_indices * 100).astype(int)  # Rough estimate
    
    # Create results
    results = []
    for i, point in enumerate(points):
        index_value = float(rockburst_indices[i])
        index_class = RockburstIndexResult.classify_index(index_value)
        
        notes = f"Hazard: {hazard_indices[i]:.3f}, Rock mass factor: {rock_mass_factors[i]:.3f}"
        if stress_surrogate is not None:
            notes += f", Stress factor: {stress_factors[i]:.3f}"
        
        result = RockburstIndexResult(
            location=tuple(point),
            index_value=index_value,
            index_class=index_class,
            contributing_events=int(contributing_events[i]),
            notes=notes
        )
        results.append(result)
    
    logger.info(f"Computed rockburst index for {n_points} points")
    
    return results

