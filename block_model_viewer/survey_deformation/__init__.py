"""
Survey deformation & subsidence analysis package.

Modules here are intentionally deterministic and audit-friendly. They ingest
survey and groundwater datasets from DataRegistrySimple, derive time-series
metrics, and push derived results back to the registry with full provenance.
"""

from .data_models import (
    PointObservation,
    PointTimeSeries,
    WellObservation,
    WellTimeSeries,
    DatasetProvenance,
    DerivedResult,
)
from .subsidence_engine import SubsidenceTimeSeriesEngine
from .control_stability_engine import ControlStabilityEngine
from .groundwater_engine import GroundwaterTimeSeriesEngine
from .coupling_engine import CouplingEngine
from .deformation_index import DeformationIndexEngine
from .filtering import SurveyNoiseFilter
from .qc import SurveyDataValidator

__all__ = [
    "PointObservation",
    "PointTimeSeries",
    "WellObservation",
    "WellTimeSeries",
    "DatasetProvenance",
    "DerivedResult",
    "SubsidenceTimeSeriesEngine",
    "ControlStabilityEngine",
    "GroundwaterTimeSeriesEngine",
    "CouplingEngine",
    "DeformationIndexEngine",
    "SurveyNoiseFilter",
    "SurveyDataValidator",
]
