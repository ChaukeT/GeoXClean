"""
Adaptive RBF (ARBF) Estimation Engine.

Partition-of-unity RBF interpolation with Gaussian Process Regression
equivalence for JORC-compliant mineral resource estimation with
uncertainty quantification.

Public API
----------
ARBFEstimator : Main orchestrator class
ARBFConfig : Pydantic configuration model
"""

from .engine import ARBFEstimator, ARBFResult
from .quality_gate import QualityCheck, QualityGateResult, evaluate_geostatistical_gate
from .simulation import ARBFSequentialSimulation, ARBFSimulationResult

__all__ = [
    "ARBFEstimator",
    "ARBFResult",
    "QualityCheck",
    "QualityGateResult",
    "evaluate_geostatistical_gate",
    "ARBFSequentialSimulation",
    "ARBFSimulationResult",
]
