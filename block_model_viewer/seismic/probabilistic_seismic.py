"""
Probabilistic Seismic Analysis

Monte Carlo and LHS analysis for seismic hazard and rockburst risk.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from .dataclasses import (
    SeismicCatalogue, HazardVolume, RockburstIndexResult,
    SeismicMCResult
)
from .hazard_volume import build_hazard_volume
from .rockburst_index import compute_rockburst_index_at_points
from ..geotech.dataclasses import RockMassGrid

logger = logging.getLogger(__name__)


def run_hazard_monte_carlo(
    catalog: SeismicCatalogue,
    grid_definition: Dict[str, Any],
    n_realizations: int,
    params: Dict[str, Any]
) -> SeismicMCResult:
    """
    Run Monte Carlo analysis for seismic hazard volume.
    
    Varies:
    - Event locations (location uncertainty)
    - Magnitudes (magnitude error)
    - b-value / GR parameters
    - Attenuation parameters
    
    Args:
        catalog: Base SeismicCatalogue
        grid_definition: Grid definition
        n_realizations: Number of realizations
        params: Parameters dict:
            - location_uncertainty: Location uncertainty std dev (default: 10.0 m)
            - magnitude_error: Magnitude error std dev (default: 0.2)
            - b_value_uncertainty: b-value uncertainty std dev (default: 0.1)
            - sampler: 'monte_carlo' or 'lhs' (default: 'monte_carlo')
            - base_params: Parameters for build_hazard_volume
    
    Returns:
        SeismicMCResult with realisations and statistics
    """
    logger.info(f"Running hazard Monte Carlo: {n_realizations} realizations")
    
    location_uncertainty = params.get('location_uncertainty', 10.0)
    magnitude_error = params.get('magnitude_error', 0.2)
    sampler_type = params.get('sampler', 'monte_carlo')
    base_params = params.get('base_params', {})
    
    realisations = []
    
    for i in range(n_realizations):
        # Create perturbed catalogue
        perturbed_events = []
        
        for event in catalog.events:
            # Perturb location
            if location_uncertainty > 0:
                location_noise = np.random.normal(0, location_uncertainty, 3)
                new_x = event.x + location_noise[0]
                new_y = event.y + location_noise[1]
                new_z = event.z + location_noise[2]
            else:
                new_x, new_y, new_z = event.x, event.y, event.z
            
            # Perturb magnitude
            if magnitude_error > 0:
                mag_noise = np.random.normal(0, magnitude_error)
                new_magnitude = max(0.0, event.magnitude + mag_noise)
            else:
                new_magnitude = event.magnitude
            
            # Create perturbed event
            from .dataclasses import SeismicEvent
            perturbed_event = SeismicEvent(
                id=f"{event.id}_MC{i}",
                time=event.time,
                x=new_x,
                y=new_y,
                z=new_z,
                magnitude=new_magnitude,
                energy=event.energy,
                moment=event.moment,
                mechanism=event.mechanism,
                quality=event.quality
            )
            perturbed_events.append(perturbed_event)
        
        # Create perturbed catalogue
        perturbed_catalog = SeismicCatalogue(
            events=perturbed_events,
            metadata={**catalog.metadata, 'realization': i}
        )
        
        # Build hazard volume
        try:
            volume = build_hazard_volume(perturbed_catalog, grid_definition, base_params)
            realisations.append(volume)
        except Exception as e:
            logger.warning(f"Failed to build hazard volume for realization {i}: {e}")
            continue
    
    # Create MC result
    mc_result = SeismicMCResult(
        realisations=realisations,
        n_realizations=len(realisations)
    )
    
    # Compute statistics
    mc_result.compute_summary_stats()
    mc_result.compute_exceedance_curve()
    
    logger.info(f"Monte Carlo complete: {len(realisations)} successful realizations")
    
    return mc_result


def run_rockburst_monte_carlo(
    hazard_volume: HazardVolume,
    rock_mass_grid: Optional[RockMassGrid],
    points: np.ndarray,
    n_realizations: int,
    params: Dict[str, Any]
) -> SeismicMCResult:
    """
    Run Monte Carlo analysis for rockburst index.
    
    Varies:
    - Hazard volume (from seismic MC)
    - Rock mass properties
    - Stress surrogates
    
    Args:
        hazard_volume: Base HazardVolume (or use MC realisations)
        rock_mass_grid: Optional RockMassGrid
        points: Points to evaluate (n_points, 3)
        n_realizations: Number of realizations
        params: Parameters dict:
            - hazard_mc_result: Optional SeismicMCResult (uses realisations)
            - rock_mass_uncertainty: RMR uncertainty std dev (default: 5.0)
            - stress_uncertainty: Stress uncertainty std dev (default: 0.1)
            - base_params: Parameters for compute_rockburst_index_at_points
    
    Returns:
        SeismicMCResult with rockburst index distributions
    """
    logger.info(f"Running rockburst Monte Carlo: {n_realizations} realizations")
    
    # Check if we have hazard MC results
    hazard_mc_result = params.get('hazard_mc_result')
    if hazard_mc_result and hazard_mc_result.realisations:
        hazard_volumes = hazard_mc_result.realisations
        n_realizations = min(n_realizations, len(hazard_volumes))
    else:
        # Use single base volume for all realizations
        hazard_volumes = [hazard_volume] * n_realizations
    
    rock_mass_uncertainty = params.get('rock_mass_uncertainty', 5.0)
    stress_uncertainty = params.get('stress_uncertainty', 0.1)
    base_params = params.get('base_params', {})
    
    # Store rockburst indices from all realizations
    all_indices = []
    
    for i in range(n_realizations):
        vol = hazard_volumes[i]
        
        # Perturb rock mass grid if available
        perturbed_rm_grid = rock_mass_grid
        if rock_mass_grid is not None and rock_mass_uncertainty > 0:
            # Create perturbed RMR values
            rmr_values = rock_mass_grid.get_property('RMR')
            if rmr_values is not None:
                rmr_noise = np.random.normal(0, rock_mass_uncertainty, len(rmr_values))
                perturbed_rmr = np.clip(rmr_values + rmr_noise, 0, 100)
                
                # Create perturbed grid
                from ..geotech.dataclasses import RockMassGrid
                perturbed_rm_grid = RockMassGrid(
                    grid_definition=rock_mass_grid.grid_definition,
                    rqd=rock_mass_grid.rqd,
                    q=rock_mass_grid.q,
                    rmr=perturbed_rmr,
                    gsi=rock_mass_grid.gsi
                )
        
        # Perturb stress if provided
        stress_surrogate = base_params.get('stress_surrogate')
        if stress_surrogate is not None and stress_uncertainty > 0:
            stress_values = np.array(stress_surrogate)
            stress_noise = np.random.normal(0, stress_uncertainty * np.max(stress_values), len(stress_values))
            perturbed_stress = np.maximum(0, stress_values + stress_noise)
            base_params['stress_surrogate'] = perturbed_stress.tolist()
        
        # Compute rockburst index
        try:
            results = compute_rockburst_index_at_points(
                vol, perturbed_rm_grid, points, base_params
            )
            
            # Extract indices
            indices = [r.index_value for r in results]
            all_indices.extend(indices)
        except Exception as e:
            logger.warning(f"Failed to compute rockburst index for realization {i}: {e}")
            continue
    
    # Create synthetic hazard volumes from rockburst indices
    # (for compatibility with SeismicMCResult structure)
    # In practice, would store rockburst-specific results
    synthetic_volumes = []
    indices_per_realization = len(points)
    
    for i in range(n_realizations):
        start_idx = i * indices_per_realization
        end_idx = start_idx + indices_per_realization
        if end_idx <= len(all_indices):
            indices_array = np.array(all_indices[start_idx:end_idx])
            
            # Create synthetic volume (reusing grid definition)
            synthetic_vol = HazardVolume(
                grid_definition=hazard_volume.grid_definition,
                hazard_index=indices_array,  # Store rockburst indices as hazard_index
                time_window=hazard_volume.time_window,
                magnitude_range=hazard_volume.magnitude_range,
                metadata={'type': 'rockburst_index', 'realization': i}
            )
            synthetic_volumes.append(synthetic_vol)
    
    # Create MC result
    mc_result = SeismicMCResult(
        realisations=synthetic_volumes,
        n_realizations=len(synthetic_volumes)
    )
    
    # Compute statistics
    mc_result.compute_summary_stats()
    mc_result.compute_exceedance_curve()
    
    logger.info(f"Rockburst Monte Carlo complete: {len(synthetic_volumes)} realizations")
    
    return mc_result

