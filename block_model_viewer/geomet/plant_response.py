"""
Plant Response Model (STEP 28)

Combine comminution + liberation + separation into plant response
at block / ore type scale.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from .comminution_model import ComminutionCircuitConfig, ComminutionOreProperties, predict_comminution_response
from .separation_model import SeparationConfig, predict_separation_response
from .liberation_model import LiberationModelConfig, predict_liberation
from .domains_links import GeometDomainMap


@dataclass
class PlantConfig:
    """
    Plant configuration combining comminution and separation.
    
    Attributes:
        name: Plant name/identifier
        comminution_config: ComminutionCircuitConfig
        separation_configs: List of separation stages (cascading)
        ore_type_map: GeometDomainMap
        constraints: Dictionary with plant constraints (capacity, water, etc.)
    """
    name: str
    comminution_config: ComminutionCircuitConfig
    separation_configs: List[SeparationConfig] = field(default_factory=list)
    ore_type_map: Optional[GeometDomainMap] = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlantResponse:
    """
    Plant response for an ore type or block.
    
    Attributes:
        ore_type_code: Ore type code
        recovery_by_element: Dictionary of recoveries by element
        concentrate_grade_by_element: Dictionary of concentrate grades by element
        mass_pull: Mass pull to concentrate (0-1)
        throughput: Throughput (t/h)
        specific_energy: Specific energy (kWh/t)
        water_consumption: Water consumption (m³/t)
        notes: Additional notes
    """
    ore_type_code: str
    recovery_by_element: Dict[str, float] = field(default_factory=dict)
    concentrate_grade_by_element: Dict[str, float] = field(default_factory=dict)
    mass_pull: float = 0.0
    throughput: float = 0.0
    specific_energy: float = 0.0
    water_consumption: Optional[float] = None
    notes: str = ""


def evaluate_ore_type_response(
    ore_type_code: str,
    chemistry: Dict[str, float],
    plant_config: PlantConfig,
    liberation_models: Dict[tuple, LiberationModelConfig],
    comminution_props: Dict[str, ComminutionOreProperties]
) -> PlantResponse:
    """
    Evaluate plant response for a given ore type.
    
    Args:
        ore_type_code: Ore type code
        chemistry: Dictionary with element grades
        plant_config: PlantConfig
        liberation_models: Dictionary mapping (ore_type, mineral) -> LiberationModelConfig
        comminution_props: Dictionary mapping ore_type_code -> ComminutionOreProperties
        
    Returns:
        PlantResponse
    """
    # Get comminution properties
    ore_props = comminution_props.get(ore_type_code)
    if ore_props is None:
        # Use default properties
        ore_props = ComminutionOreProperties(
            ore_type_code=ore_type_code,
            work_index_bond=12.0  # Default
        )
    
    # Predict comminution response
    comminution_result = predict_comminution_response(
        ore_props,
        plant_config.comminution_config
    )
    
    # Apply throughput constraint
    throughput = comminution_result.throughput
    if "max_throughput" in plant_config.constraints:
        throughput = min(throughput, plant_config.constraints["max_throughput"])
    
    # Predict liberation for primary element
    primary_element = list(chemistry.keys())[0] if chemistry else "Fe"
    liberation_key = (ore_type_code, primary_element)
    
    if liberation_key in liberation_models:
        liberation_config = liberation_models[liberation_key]
        liberation_fractions = predict_liberation(
            comminution_result.size_distribution,
            liberation_config
        )
        avg_liberation = np.mean(liberation_fractions)
    else:
        # Default liberation estimate
        avg_liberation = 0.7
    
    # Evaluate separation stages
    recovery_by_element = {}
    concentrate_grade_by_element = {}
    overall_mass_pull = 1.0
    
    for sep_config in plant_config.separation_configs:
        # Create liberation curve from average
        from .liberation_model import LiberationCurve
        lib_curve = LiberationCurve(
            size_classes=comminution_result.size_classes,
            liberation_fraction=np.full_like(
                comminution_result.size_classes,
                avg_liberation
            ),
            mineral_name=primary_element,
            ore_type_code=ore_type_code
        )
        
        # Predict separation
        sep_response = predict_separation_response(
            lib_curve,
            chemistry,
            sep_config
        )
        
        # Accumulate recoveries (simplified: use last stage)
        recovery_by_element[primary_element] = sep_response.recovery
        concentrate_grade_by_element[primary_element] = sep_response.concentrate_grade
        overall_mass_pull = sep_response.mass_pull
    
    # Estimate water consumption (typical: 0.5-2.0 m³/t)
    water_consumption = plant_config.constraints.get("water_per_tonne", 1.0)
    
    return PlantResponse(
        ore_type_code=ore_type_code,
        recovery_by_element=recovery_by_element,
        concentrate_grade_by_element=concentrate_grade_by_element,
        mass_pull=overall_mass_pull,
        throughput=throughput,
        specific_energy=comminution_result.specific_energy,
        water_consumption=water_consumption,
        notes=f"Plant: {plant_config.name}"
    )


def evaluate_block_response(
    block_record: Dict[str, Any],
    plant_config: PlantConfig,
    helpers: Dict[str, Any]
) -> PlantResponse:
    """
    Evaluate plant response for a single block.
    
    Args:
        block_record: Dictionary with block data (ore_type_code, grades, etc.)
        plant_config: PlantConfig
        helpers: Dictionary with helper functions/models
        
    Returns:
        PlantResponse
    """
    ore_type_code = block_record.get("ore_type_code", "UNKNOWN")
    
    # Extract chemistry from block record
    chemistry = {}
    for key, value in block_record.items():
        if key not in ["ore_type_code", "x", "y", "z", "tonnage"]:
            try:
                float_val = float(value)
                if float_val > 0:  # Only include positive grades
                    chemistry[key] = float_val
            except (ValueError, TypeError):
                pass
    
    # Get models from helpers
    liberation_models = helpers.get("liberation_models", {})
    comminution_props = helpers.get("comminution_props", {})
    
    return evaluate_ore_type_response(
        ore_type_code,
        chemistry,
        plant_config,
        liberation_models,
        comminution_props
    )

