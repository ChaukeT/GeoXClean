"""
Geometallurgical Block Model (STEP 28)

Build and manage geomet attributes on the block model.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from .domains_links import GeometDomainMap
from .plant_response import PlantConfig, evaluate_block_response


@dataclass
class GeometBlockAttributes:
    """
    Geometallurgical attributes for blocks.
    
    Attributes:
        ore_type_code: Array of ore type codes per block
        plant_name: Plant name
        recovery_by_element: Dictionary mapping element -> recovery array
        concentrate_grade_by_element: Dictionary mapping element -> grade array
        plant_tonnage_factor: Array of tonnage factors (throughput/capacity proxy)
        plant_specific_energy: Array of specific energy values (kWh/t)
    """
    ore_type_code: np.ndarray  # per block
    plant_name: str
    recovery_by_element: Dict[str, np.ndarray] = field(default_factory=dict)
    concentrate_grade_by_element: Dict[str, np.ndarray] = field(default_factory=dict)
    plant_tonnage_factor: np.ndarray = field(default_factory=lambda: np.array([]))
    plant_specific_energy: np.ndarray = field(default_factory=lambda: np.array([]))


def assign_ore_types_to_blocks(
    block_model: Any,
    geomet_domain_map: GeometDomainMap,
    rules: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Assign ore types to blocks based on domain mapping.
    
    Args:
        block_model: BlockModel instance
        geomet_domain_map: GeometDomainMap
        rules: Optional rules dictionary
        
    Returns:
        Array of ore type codes (strings) per block
    """
    n_blocks = block_model.block_count
    ore_types = np.full(n_blocks, "UNKNOWN", dtype=object)
    
    # Get domain property if available
    domain_property = rules.get("domain_property", "domain") if rules else "domain"
    
    if domain_property in block_model.get_property_names():
        domain_codes = block_model.get_property(domain_property)
        
        # Map domains to ore types
        for i, domain_code in enumerate(domain_codes):
            if domain_code is None or domain_code == "":
                continue
            
            # Try to infer ore type
            ore_type_code = None
            for ot_code, ot_def in geomet_domain_map.ore_types.items():
                if str(domain_code) in ot_def.geology_domains:
                    ore_type_code = ot_code
                    break
            
            if ore_type_code:
                ore_types[i] = ore_type_code
    
    return ore_types


def compute_geomet_attributes_for_blocks(
    block_model: Any,
    plant_config: PlantConfig,
    domain_map: GeometDomainMap,
    liberation_models: Dict[tuple, Any],
    comminution_props: Dict[str, Any]
) -> GeometBlockAttributes:
    """
    Compute geometallurgical attributes for all blocks.
    
    Args:
        block_model: BlockModel instance
        plant_config: PlantConfig
        domain_map: GeometDomainMap
        liberation_models: Dictionary mapping (ore_type, mineral) -> LiberationModelConfig
        comminution_props: Dictionary mapping ore_type_code -> ComminutionOreProperties
        
    Returns:
        GeometBlockAttributes
    """
    n_blocks = block_model.block_count
    
    # Assign ore types
    rules = {"domain_property": "domain"}
    ore_type_codes = assign_ore_types_to_blocks(
        block_model,
        domain_map,
        rules
    )
    
    # Initialize arrays
    recovery_by_element = {}
    concentrate_grade_by_element = {}
    tonnage_factors = np.zeros(n_blocks)
    specific_energies = np.zeros(n_blocks)
    
    # Get property names for chemistry
    property_names = block_model.get_property_names()
    element_names = [p for p in property_names if p not in ["x", "y", "z", "domain", "ore_type"]]
    
    # Initialize recovery/grade arrays for each element
    for element in element_names:
        recovery_by_element[element] = np.zeros(n_blocks)
        concentrate_grade_by_element[element] = np.zeros(n_blocks)
    
    # Process each block
    helpers = {
        "liberation_models": liberation_models,
        "comminution_props": comminution_props
    }
    
    for i in range(n_blocks):
        # Build block record
        block_record = {
            "ore_type_code": ore_type_codes[i]
        }
        
        # Add chemistry
        for element in element_names:
            values = block_model.get_property(element)
            if values is not None and i < len(values):
                block_record[element] = float(values[i])
        
        # Evaluate plant response
        try:
            response = evaluate_block_response(
                block_record,
                plant_config,
                helpers
            )
            
            # Store recoveries and grades
            for element, recovery in response.recovery_by_element.items():
                if element in recovery_by_element:
                    recovery_by_element[element][i] = recovery
            
            for element, grade in response.concentrate_grade_by_element.items():
                if element in concentrate_grade_by_element:
                    concentrate_grade_by_element[element][i] = grade
            
            tonnage_factors[i] = response.throughput / 1000.0  # Normalize
            specific_energies[i] = response.specific_energy
            
        except Exception:
            # Skip blocks with errors
            continue
    
    return GeometBlockAttributes(
        ore_type_code=ore_type_codes,
        plant_name=plant_config.name,
        recovery_by_element=recovery_by_element,
        concentrate_grade_by_element=concentrate_grade_by_element,
        plant_tonnage_factor=tonnage_factors,
        plant_specific_energy=specific_energies
    )


def attach_geomet_to_block_model(
    block_model: Any,
    geomet_attrs: GeometBlockAttributes
) -> None:
    """
    Attach geometallurgical properties to block model.
    
    Adds properties like:
      - ore_type
      - rec_Fe_<plant_name>
      - rec_SiO2_<plant_name>
      - gvalue_<plant_name>  (geomet-adjusted block value)
    
    Args:
        block_model: BlockModel instance
        geomet_attrs: GeometBlockAttributes
    """
    plant_name = geomet_attrs.plant_name
    
    # Add ore type
    block_model.add_property("ore_type", geomet_attrs.ore_type_code)
    
    # Add recovery properties
    for element, recovery_array in geomet_attrs.recovery_by_element.items():
        prop_name = f"rec_{element}_{plant_name}"
        block_model.add_property(prop_name, recovery_array)
    
    # Add concentrate grade properties
    for element, grade_array in geomet_attrs.concentrate_grade_by_element.items():
        prop_name = f"conc_{element}_{plant_name}"
        block_model.add_property(prop_name, grade_array)
    
    # Add tonnage factor
    block_model.add_property(
        f"tonnage_factor_{plant_name}",
        geomet_attrs.plant_tonnage_factor
    )
    
    # Add specific energy
    block_model.add_property(
        f"specific_energy_{plant_name}",
        geomet_attrs.plant_specific_energy
    )
    
    # Calculate and add geomet-adjusted block value (gvalue)
    # This would typically use prices and costs from economic parameters
    # For now, we'll create a placeholder that can be calculated later
    # The actual calculation should use calculate_block_value_geomet from irr_engine.npv_calc
    n_blocks = len(geomet_attrs.ore_type_code)
    gvalue = np.zeros(n_blocks)
    
    # Simple placeholder: recovery-weighted value
    # In practice, this should use actual prices and costs
    for element, recovery_array in geomet_attrs.recovery_by_element.items():
        grade_array = block_model.get_property(element)
        if grade_array is not None:
            # Placeholder calculation - should be replaced with actual economic model
            gvalue += recovery_array * grade_array
    
    block_model.add_property(
        f"gvalue_{plant_name}",
        gvalue
    )

