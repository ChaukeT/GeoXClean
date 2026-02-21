"""
Geometallurgical Chain Module (STEP 38)

Full geomet chain from ore types to plant response to NPV.
"""

from .ore_type_model import OreDomain, OreTypeModel, infer_ore_domain_for_block, attach_ore_type_to_block_model
from .plant_response import RecoverySurface, ThroughputSurface, PlantRoute, compute_recovery, compute_throughput
from .route_selector import RouteSelectorConfig, choose_route_for_block
from .geomet_value_engine import GeometValueConfig, compute_geomet_block_values

__all__ = [
    'OreDomain',
    'OreTypeModel',
    'infer_ore_domain_for_block',
    'attach_ore_type_to_block_model',
    'RecoverySurface',
    'ThroughputSurface',
    'PlantRoute',
    'compute_recovery',
    'compute_throughput',
    'RouteSelectorConfig',
    'choose_route_for_block',
    'GeometValueConfig',
    'compute_geomet_block_values',
]

