"""
Seismic and Rockburst Hazard Analysis Module

Provides seismic event handling, hazard volume construction, rockburst index
computation, and probabilistic seismic analysis.
"""

from .dataclasses import (
    SeismicEvent,
    SeismicCatalogue,
    HazardVolume,
    RockburstIndexResult,
    SeismicMCResult
)

from .catalog import (
    load_catalog,
    filter_catalog,
    compute_b_value,
    compute_event_rate
)

from .hazard_volume import build_hazard_volume

from .rockburst_index import compute_rockburst_index_at_points

from .probabilistic_seismic import (
    run_hazard_monte_carlo,
    run_rockburst_monte_carlo
)

__all__ = [
    'SeismicEvent',
    'SeismicCatalogue',
    'HazardVolume',
    'RockburstIndexResult',
    'SeismicMCResult',
    'load_catalog',
    'filter_catalog',
    'compute_b_value',
    'compute_event_rate',
    'build_hazard_volume',
    'compute_rockburst_index_at_points',
    'run_hazard_monte_carlo',
    'run_rockburst_monte_carlo',
]

