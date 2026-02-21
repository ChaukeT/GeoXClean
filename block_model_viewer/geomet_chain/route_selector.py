"""
Route Selector (STEP 38)

Select optimal plant route for each block/ore type.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

from .plant_response import PlantRoute, compute_recovery, compute_throughput

logger = logging.getLogger(__name__)


@dataclass
class RouteSelectorConfig:
    """
    Configuration for route selection.
    
    Attributes:
        routes: List of PlantRoute objects
        rule: Selection rule ("highest_value", "fixed_route", "blend_target")
        blend_targets: Optional grade constraints for blending
        energy_cost_per_kWh: Energy cost per kWh
        plant_nominal_tph: Nominal plant throughput (for throughput calculation)
        prices: Prices by element
    """
    routes: List[PlantRoute] = field(default_factory=list)
    rule: str = "highest_value"
    blend_targets: Optional[Dict[str, float]] = None
    energy_cost_per_kWh: float = 0.10
    plant_nominal_tph: float = 1000.0
    prices: Dict[str, float] = field(default_factory=dict)


def choose_route_for_block(
    ore_domain_id: str,
    head_grades: Dict[str, float],
    hardness_index: float,
    config: RouteSelectorConfig
) -> str:
    """
    Choose optimal plant route for a block.
    
    Args:
        ore_domain_id: Ore domain ID
        head_grades: Head grades by element
        hardness_index: Hardness index
        config: RouteSelectorConfig
    
    Returns:
        PlantRoute.id
    """
    if not config.routes:
        logger.warning("No routes available")
        return ""
    
    if config.rule == "fixed_route":
        # Use first route as fixed
        return config.routes[0].id
    
    if config.rule == "highest_value":
        # Compute value per tonne for each route
        best_route_id = ""
        best_value = float('-inf')
        
        for route in config.routes:
            # Compute recoveries
            recoveries = compute_recovery(route, ore_domain_id, head_grades)
            
            # Compute throughput factor
            effective_tph = compute_throughput(
                route, ore_domain_id, hardness_index, config.plant_nominal_tph
            )
            throughput_factor = effective_tph / config.plant_nominal_tph if config.plant_nominal_tph > 0 else 1.0
            
            # Compute revenue
            revenue = 0.0
            for element, grade in head_grades.items():
                recovery = recoveries.get(element, 0.0)
                price = config.prices.get(element, 0.0)
                revenue += grade * recovery * price
            
            # Compute costs
            # Base cost
            cost = route.base_cost_per_t
            
            # Energy cost (simplified: assume specific energy proportional to hardness)
            specific_energy = 15.0 + (hardness_index - 5.0) * 2.0  # kWh/t
            energy_cost = specific_energy * route.variable_cost_per_kWh
            
            # Total cost
            total_cost = cost + energy_cost
            
            # Net value per tonne (adjusted for throughput)
            net_value = (revenue - total_cost) * throughput_factor
            
            if net_value > best_value:
                best_value = net_value
                best_route_id = route.id
        
        return best_route_id
    
    elif config.rule == "blend_target":
        # Select route that best matches blend targets
        if not config.blend_targets:
            # Fall back to highest value
            return choose_route_for_block(
                ore_domain_id, head_grades, hardness_index,
                RouteSelectorConfig(
                    routes=config.routes,
                    rule="highest_value",
                    prices=config.prices,
                    plant_nominal_tph=config.plant_nominal_tph
                )
            )
        
        best_route_id = ""
        best_match_score = float('inf')
        
        for route in config.routes:
            # Compute recoveries
            recoveries = compute_recovery(route, ore_domain_id, head_grades)
            
            # Compute concentrate grades
            concentrate_grades = {}
            for element, grade in head_grades.items():
                recovery = recoveries.get(element, 0.0)
                # Simplified: assume mass pull = recovery (for single element)
                concentrate_grades[element] = grade * recovery / max(recovery, 0.01)
            
            # Compute match score (sum of squared differences from targets)
            match_score = 0.0
            for element, target in config.blend_targets.items():
                actual = concentrate_grades.get(element, 0.0)
                match_score += (actual - target) ** 2
            
            if match_score < best_match_score:
                best_match_score = match_score
                best_route_id = route.id
        
        return best_route_id
    
    else:
        logger.warning(f"Unknown route selection rule: {config.rule}")
        return config.routes[0].id if config.routes else ""

