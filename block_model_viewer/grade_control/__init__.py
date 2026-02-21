"""
Grade Control Module (STEP 29)

Short-term grade control models at SMU support, diglines, and ore/waste classification.
"""

from .support_model import (
    GCGridDefinition,
    GCModel,
    derive_gc_grid_from_long_term,
    resample_long_term_to_gc
)

from .gc_kriging import (
    GCKrigingConfig,
    GCKrigingResult,
    run_gc_ok
)

from .gc_simulation import (
    GCSimulationConfig,
    GCSimulationResult,
    run_gc_sgsim
)

from .digpolygons import (
    PlanPolygon,
    DiglineSet,
    blocks_within_polygon,
    assign_ore_waste_flags
)

from .ore_waste_marking import (
    OreWasteCutoffRule,
    OreWasteClassificationResult,
    classify_gc_blocks,
    summarise_by_digpolygon
)

__all__ = [
    # Support model
    "GCGridDefinition",
    "GCModel",
    "derive_gc_grid_from_long_term",
    "resample_long_term_to_gc",
    # GC Kriging
    "GCKrigingConfig",
    "GCKrigingResult",
    "run_gc_ok",
    # GC Simulation
    "GCSimulationConfig",
    "GCSimulationResult",
    "run_gc_sgsim",
    # Diglines
    "PlanPolygon",
    "DiglineSet",
    "blocks_within_polygon",
    "assign_ore_waste_flags",
    # Ore/Waste marking
    "OreWasteCutoffRule",
    "OreWasteClassificationResult",
    "classify_gc_blocks",
    "summarise_by_digpolygon",
]

