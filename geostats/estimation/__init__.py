"""
FastRBF Estimation sub-package.

Core RBF solver, kernel library, search neighbourhood, block estimator,
cross-validation, diagnostics, and JORC audit trail.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from .config import RBFConfig, KernelType, DriftType
from .fastrbf_engine import FastRBFEngine, FittedRBF
from .block_estimator import BlockModelEstimator, EstimationResult
from .interpolant_functions import evaluate_kernel
from .search_neighbourhood import SearchNeighbourhood
from .cross_validation import loo_cross_validation, kfold_cross_validation
from .diagnostics import EstimationDiagnostics
from .audit import JORCAuditRecord

__all__ = [
    "RBFConfig",
    "KernelType",
    "DriftType",
    "FastRBFEngine",
    "FittedRBF",
    "BlockModelEstimator",
    "EstimationResult",
    "evaluate_kernel",
    "SearchNeighbourhood",
    "loo_cross_validation",
    "kfold_cross_validation",
    "EstimationDiagnostics",
    "JORCAuditRecord",
]
