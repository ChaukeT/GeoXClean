"""
Geometallurgy Module (STEP 28)

Links geological domains, block model grades, liberation/mineralogy,
comminution/separation performance, and plant recovery/throughput
to produce geometallurgical block models and plant-linked resource metrics.
"""

from .domains_links import (
    GeometOreType,
    GeometDomainMap,
    infer_ore_type
)

from .liberation_model import (
    LiberationCurve,
    LiberationModelConfig,
    predict_liberation,
    build_liberation_curve_from_testwork
)

from .comminution_model import (
    ComminutionOreProperties,
    ComminutionCircuitConfig,
    ComminutionResult,
    predict_comminution_response,
    estimate_work_index_from_ucs,
    estimate_comminution_from_geotech
)

from .separation_model import (
    SeparationConfig,
    SeparationResponse,
    predict_separation_response
)

from .plant_response import (
    PlantConfig,
    PlantResponse,
    evaluate_ore_type_response,
    evaluate_block_response
)

from .geomet_block_model import (
    GeometBlockAttributes,
    assign_ore_types_to_blocks,
    compute_geomet_attributes_for_blocks,
    attach_geomet_to_block_model
)

__all__ = [
    # Domain linking
    "GeometOreType",
    "GeometDomainMap",
    "infer_ore_type",
    # Liberation
    "LiberationCurve",
    "LiberationModelConfig",
    "predict_liberation",
    "build_liberation_curve_from_testwork",
    # Comminution
    "ComminutionOreProperties",
    "ComminutionCircuitConfig",
    "ComminutionResult",
    "predict_comminution_response",
    "estimate_work_index_from_ucs",
    "estimate_comminution_from_geotech",
    # Separation
    "SeparationConfig",
    "SeparationResponse",
    "predict_separation_response",
    # Plant response
    "PlantConfig",
    "PlantResponse",
    "evaluate_ore_type_response",
    "evaluate_block_response",
    # Block model
    "GeometBlockAttributes",
    "assign_ore_types_to_blocks",
    "compute_geomet_attributes_for_blocks",
    "attach_geomet_to_block_model",
]

