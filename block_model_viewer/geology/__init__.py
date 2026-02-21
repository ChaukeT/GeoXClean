"""
GeoX Geology Package.

LoopStructural-based geological modeling engine with JORC/SAMREC compliance.

This package provides:
- ChronosEngine: Coordinate normalization and event-stacking
- GeoXIndustryModeler: CP-grade geological modeling (DEPRECATED - use GeologicalModelRunner)
- GeologicalModelRunner: Unified CP-grade pipeline (THE AUTHORITATIVE ENTRY POINT)
- GradientEstimator: PCA-based gradient computation from contact geometry
- ComplianceManager: Spatial misfit calculation and audit reporting
- FaultDetectionEngine: Automatic fault plane suggestion from model errors

Architecture:
- block_model_viewer.geology: Engine modules (pure computation)
- block_model_viewer.ui.loopstructural_panel: UI adapters
"""

__version__ = "1.1.0"

from .chronos_engine import ChronosEngine
from .industry_modeler import GeoXIndustryModeler  # Deprecated
from .compliance_manager import ComplianceManager, AuditReport
from .fault_detection import FaultDetectionEngine, SuggestedFault
from .mesh_validator import verify_mesh_integrity
from .model_runner import (
    GeologicalModelRunner,
    ModelResult,
    validate_and_filter_drillhole_data,
)
from .gradient_estimator import (
    compute_contact_gradients,
    ContactGradient,
    GradientEstimationReport,
)

__all__ = [
    "__version__",
    # Core engines
    "ChronosEngine",
    "GeoXIndustryModeler",  # Deprecated - kept for backward compatibility
    # Unified pipeline (RECOMMENDED)
    "GeologicalModelRunner",
    "ModelResult",
    # Data validation
    "validate_and_filter_drillhole_data",
    # Gradient estimation (NEW)
    "compute_contact_gradients",
    "ContactGradient",
    "GradientEstimationReport",
    # Compliance
    "ComplianceManager",
    "AuditReport",
    # Fault detection
    "FaultDetectionEngine",
    "SuggestedFault",
    # Utilities
    "verify_mesh_integrity",
]

