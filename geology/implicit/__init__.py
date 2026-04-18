"""
Implicit Geological Modelling Engine
=====================================

Gradient-augmented RBF interpolation of signed distance functions
with marching cubes surface extraction.  Reuses ARBF kernel
infrastructure for all kernel evaluations.

For datasets exceeding PUM_THRESHOLD (default 1500) constraints the
engine automatically uses Partition-of-Unity Method (PUM) domain
decomposition with Wendland C2 blending.

Modules
-------
gradient_kernels : Derivative kernel entries for the augmented matrix.
contact_data     : Data model for contacts and orientations.
signed_distance  : SDF construction from drillhole contacts.
scalar_field     : Augmented matrix assembly, solve, evaluate; PUM engine.
surface_extraction : Marching cubes surface extraction.
domain_model     : Domain assignment to block models.
geological_model : Main orchestrator.
validation       : Contact honouring QC.
audit            : Geological model audit trail.
"""

from .geological_model import GeologicalModelBuilder
from .stratigraphy import StratigraphicModelColumn, build_stratigraphic_model
from .vein_model import build_vein_model
from .fault_model import FaultDefinition, build_faulted_geological_model
from .fold_frame import FoldFrame, FoldFrameConfig
from .domain_model import (
    assign_domains_potential_field,
    assign_domains_independent_surfaces,
    assign_domains_from_lithology,
)
from .scalar_field import (
    PUM_THRESHOLD,
    PUMScalarField,
    solve_augmented_system_pum,
    evaluate_scalar_field_pum,
    make_evaluate_fn_pum,
)

__all__ = [
    "GeologicalModelBuilder",
    "StratigraphicModelColumn",
    "build_stratigraphic_model",
    "build_vein_model",
    "FaultDefinition",
    "build_faulted_geological_model",
    "FoldFrame",
    "FoldFrameConfig",
    "assign_domains_potential_field",
    "assign_domains_independent_surfaces",
    "assign_domains_from_lithology",
    # PUM API
    "PUM_THRESHOLD",
    "PUMScalarField",
    "solve_augmented_system_pum",
    "evaluate_scalar_field_pum",
    "make_evaluate_fn_pum",
]
