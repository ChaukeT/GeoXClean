"""
Plant Response Surfaces (STEP 38)

Recovery and throughput response surfaces by ore type/plant route.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecoverySurface:
    """
    Recovery surface for an element.
    
    Attributes:
        id: Surface identifier
        element: Element name (e.g., "Fe", "Cu")
        parameters: Parameters dict (e.g., polynomial coefficients or lookup table)
        ore_domain_ids: List of ore domain IDs this applies to
    """
    id: str
    element: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    ore_domain_ids: List[str] = field(default_factory=list)


@dataclass
class ThroughputSurface:
    """
    Throughput response surface.
    
    Attributes:
        id: Surface identifier
        parameters: Parameters dict (e.g., k(specific_energy, hardness))
        ore_domain_ids: List of ore domain IDs this applies to
    """
    id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    ore_domain_ids: List[str] = field(default_factory=list)


@dataclass
class PlantRoute:
    """
    Plant route definition.
    
    Attributes:
        id: Route identifier (e.g., "PlantA", "PlantB", "Bypass")
        name: Display name
        recovery_surfaces: List of RecoverySurface objects
        throughput_surfaces: List of ThroughputSurface objects
        base_cost_per_t: Base cost per tonne
        variable_cost_per_kWh: Variable cost per kWh
    """
    id: str
    name: str = ""
    recovery_surfaces: List[RecoverySurface] = field(default_factory=list)
    throughput_surfaces: List[ThroughputSurface] = field(default_factory=list)
    base_cost_per_t: float = 10.0
    variable_cost_per_kWh: float = 0.10


def compute_recovery(
    route: PlantRoute,
    ore_domain_id: str,
    head_grades: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute recovery for each element using recovery surfaces.
    
    Args:
        route: PlantRoute
        ore_domain_id: Ore domain ID
        head_grades: Head grades by element
    
    Returns:
        Dictionary of recoveries by element
    """
    recoveries = {}
    
    for surface in route.recovery_surfaces:
        # Check if surface applies to this ore domain
        if surface.ore_domain_ids and ore_domain_id not in surface.ore_domain_ids:
            continue
        
        element = surface.element
        head_grade = head_grades.get(element, 0.0)
        
        # Simple recovery model: linear or polynomial
        params = surface.parameters
        
        if params.get("type") == "linear":
            # Linear: recovery = a + b * head_grade
            a = params.get("a", 0.8)
            b = params.get("b", 0.0)
            recovery = a + b * head_grade
        elif params.get("type") == "polynomial":
            # Polynomial: recovery = a + b*x + c*x^2
            a = params.get("a", 0.8)
            b = params.get("b", 0.0)
            c = params.get("c", 0.0)
            recovery = a + b * head_grade + c * head_grade ** 2
        elif params.get("type") == "lookup":
            # Lookup table
            lookup = params.get("lookup", {})
            recovery = lookup.get(head_grade, params.get("default", 0.8))
        else:
            # Default: constant recovery
            recovery = params.get("base_recovery", 0.8)
        
        # Clamp to [0, 1]
        recovery = max(0.0, min(1.0, recovery))
        recoveries[element] = recovery
    
    return recoveries


def compute_throughput(
    route: PlantRoute,
    ore_domain_id: str,
    hardness_index: float,
    plant_nominal_tph: float
) -> float:
    """
    Compute effective throughput based on hardness and ore domain.
    
    Args:
        route: PlantRoute
        ore_domain_id: Ore domain ID
        hardness_index: Hardness index
        plant_nominal_tph: Nominal plant throughput (tonnes per hour)
    
    Returns:
        Effective throughput in tph
    """
    # Find applicable throughput surface
    applicable_surface = None
    for surface in route.throughput_surfaces:
        if not surface.ore_domain_ids or ore_domain_id in surface.ore_domain_ids:
            applicable_surface = surface
            break
    
    if not applicable_surface:
        # Default: no reduction
        return plant_nominal_tph
    
    params = applicable_surface.parameters
    
    if params.get("type") == "hardness_factor":
        # Throughput = nominal * factor(hardness)
        factor = params.get("base_factor", 1.0)
        hardness_penalty = params.get("hardness_penalty", 0.0)
        factor = factor - hardness_penalty * (hardness_index - 5.0) / 5.0
        factor = max(0.1, min(1.0, factor))  # Clamp to [0.1, 1.0]
        return plant_nominal_tph * factor
    elif params.get("type") == "energy_limit":
        # Throughput limited by specific energy
        specific_energy = params.get("specific_energy_kWh_t", 20.0)
        # Harder ore requires more energy, reducing throughput
        energy_factor = 1.0 / (1.0 + (hardness_index - 5.0) * 0.1)
        return plant_nominal_tph * energy_factor
    else:
        # Default: no reduction
        return plant_nominal_tph

