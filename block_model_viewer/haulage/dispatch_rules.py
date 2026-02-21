"""
Dispatch Rules (STEP 30)

Simple dispatch rules (greedy and priority-based).
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def allocate_trucks_to_routes(
    fleet: Any,  # FleetConfig
    routes: List[Any],  # List[Route]
    production_targets: Dict[str, float]  # tonnes per route per period
) -> Dict[str, Any]:
    """
    Allocate trucks to routes based on production targets.
    
    Args:
        fleet: FleetConfig instance
        routes: List of Route instances
        production_targets: Dictionary mapping route_id -> tonnes per period
        
    Returns:
        Dictionary with allocation of truck-hours per route
    """
    from .cycle_time_model import compute_cycle_time
    
    allocation = {}
    
    # Calculate required truck-hours per route
    route_requirements = {}
    for route in routes:
        route_id = route.id
        target_tonnes = production_targets.get(route_id, 0.0)
        
        if target_tonnes <= 0:
            continue
        
        # Use first truck for capacity estimate (simplified)
        if fleet.trucks:
            truck = fleet.trucks[0]
            cycle_result = compute_cycle_time(truck, route)
            
            if cycle_result.tonnes_per_hour > 0:
                required_hours = target_tonnes / cycle_result.tonnes_per_hour
                route_requirements[route_id] = required_hours
            else:
                route_requirements[route_id] = 0.0
        else:
            route_requirements[route_id] = 0.0
    
    # Allocate trucks (greedy: assign to routes with highest requirements first)
    total_truck_hours = sum(truck.availability * truck.utilisation * fleet.shift_hours for truck in fleet.trucks)
    
    sorted_routes = sorted(route_requirements.items(), key=lambda x: x[1], reverse=True)
    
    allocation = {}
    remaining_hours = total_truck_hours
    
    for route_id, required_hours in sorted_routes:
        allocated_hours = min(required_hours, remaining_hours)
        allocation[route_id] = {
            "truck_hours": allocated_hours,
            "estimated_tonnes": allocated_hours * (production_targets.get(route_id, 0.0) / max(required_hours, 1.0))
        }
        remaining_hours -= allocated_hours
    
    logger.info(f"Allocated trucks to {len(allocation)} routes")
    
    return allocation

