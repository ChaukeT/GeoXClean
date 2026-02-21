"""
ESG Water & Tailings Module

Node-link water balance modeling for mine water management.
"""

from .water_balance import (
    WaterNode,
    NodeType,
    WaterLink,
    WaterBalance,
    simulate_water_balance,
    calculate_pond_freeboard,
    estimate_evaporation,
    calculate_water_footprint
)

__all__ = [
    'WaterNode',
    'NodeType',
    'WaterLink',
    'WaterBalance',
    'simulate_water_balance',
    'calculate_pond_freeboard',
    'estimate_evaporation',
    'calculate_water_footprint'
]
