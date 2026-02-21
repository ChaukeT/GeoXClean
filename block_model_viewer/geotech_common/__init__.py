"""
Geotechnical Common - Shared types and utilities for geotechnical analysis.
"""

from .material_properties import GeotechMaterial, GeotechMaterialLibrary, estimate_material_from_rock_mass
from .slope_geometry import SlopeSector, SlopeSet
from .probability_utils import sample_material_properties, probability_of_failure

__all__ = [
    'GeotechMaterial',
    'GeotechMaterialLibrary',
    'estimate_material_from_rock_mass',
    'SlopeSector',
    'SlopeSet',
    'sample_material_properties',
    'probability_of_failure',
]

