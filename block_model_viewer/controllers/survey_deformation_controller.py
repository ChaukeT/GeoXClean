"""
Survey Deformation & Subsidence Controller.

Orchestrates ingestion, QC, time-series analysis, stability classification,
groundwater interpretation, coupling, and deformation index computation.
All derived outputs are written back to DataRegistrySimple with provenance
metadata so UI/reporting layers can remain audit-ready.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import pandas as pd

from ..core.data_registry_simple import DataRegistrySimple
from ..survey_deformation import (
    SubsidenceTimeSeriesEngine,
    ControlStabilityEngine,
    GroundwaterTimeSeriesEngine,
    CouplingEngine,
    DeformationIndexEngine,
    SurveyNoiseFilter,
)
from ..survey_deformation.reporting import ReportingEngine
from ..survey_deformation.data_models import PointTimeSeries, WellTimeSeries
from ..survey_deformation.qc import SurveyDataValidator, ValidationIssue

logger = logging.getLogger(__name__)


class SurveyDeformationController:
    """Controller for survey-based deformation and subsidence workflows."""

    SUBSIDENCE_RAW_KEY = "subsidence_survey_raw"
    SUBSIDENCE_TIMESERIES_KEY = "subsidence_timeseries"
    SUBSIDENCE_METRICS_KEY = "subsidence_metrics"
    CONTROL_STABILITY_KEY = "control_stability"
    GROUNDWATER_RAW_KEY = "groundwater_raw"
    GROUNDWATER_TIMESERIES_KEY = "groundwater_timeseries"
    GROUNDWATER_METRICS_KEY = "groundwater_metrics"
    COUPLING_KEY = "subsidence_groundwater_coupling"
    DEFORMATION_INDEX_KEY = "deformation_index"

    def __init__(self, app_controller: Any):
        self._app = app_controller
        self.registry: DataRegistrySimple = app_controller.registry

        # Engines
        self.subsidence_engine = SubsidenceTimeSeriesEngine(self.registry, software_version="geox")
        self.control_engine = ControlStabilityEngine(self.registry, software_version="geox")
        self.groundwater_engine = GroundwaterTimeSeriesEngine(self.registry, software_version="geox")
        self.coupling_engine = CouplingEngine(self.registry, software_version="geox")
        self.index_engine = DeformationIndexEngine(self.registry, software_version="geox")
        self.filter = SurveyNoiseFilter()
        self.reporting = ReportingEngine()
        self.validator = SurveyDataValidator()

    # ------------------------------------------------------------------ #
    # Ingestion helpers
    # ------------------------------------------------------------------ #

    def ingest_subsidence_dataframe(
        self,
        df: pd.DataFrame,
        source_file: Optional[Path] = None,
        parameters: Optional[Dict[str, Any]] = None,
        registry_key: str = SUBSIDENCE_RAW_KEY,
        source_panel: str = "survey_deformation",
    ) -> str:
        """Validate, normalise, and register subsidence survey data as RAW."""
        normalized = PointTimeSeries(df).df
        issues = self.validator.validate_survey(normalized)
        self._raise_on_errors(issues)

        meta = {
            "labels": ["RAW"],
            "parameters": parameters or {},
            "source_file": str(source_file) if source_file else None,
            "method": "subsidence_ingest",
        }
        ok = self.registry.register_model(registry_key, normalized, metadata=meta, source_panel=source_panel)
        if not ok:
            raise RuntimeError("Failed to register subsidence survey data")
        logger.info(f"Registered subsidence survey data under key '{registry_key}'")
        return registry_key

    def ingest_groundwater_dataframe(
        self,
        df: pd.DataFrame,
        source_file: Optional[Path] = None,
        parameters: Optional[Dict[str, Any]] = None,
        registry_key: str = GROUNDWATER_RAW_KEY,
        source_panel: str = "survey_deformation",
    ) -> str:
        """Validate, normalise, and register groundwater well data as RAW."""
        normalized = WellTimeSeries(df).df
        issues = self.validator.validate_wells(normalized)
        self._raise_on_errors(issues)

        meta = {
            "labels": ["RAW"],
            "parameters": parameters or {},
            "source_file": str(source_file) if source_file else None,
            "method": "groundwater_ingest",
        }
        ok = self.registry.register_model(registry_key, normalized, metadata=meta, source_panel=source_panel)
        if not ok:
            raise RuntimeError("Failed to register groundwater data")
        logger.info(f"Registered groundwater data under key '{registry_key}'")
        return registry_key

    # ------------------------------------------------------------------ #
    # Engine orchestration
    # ------------------------------------------------------------------ #

    def run_subsidence_time_series(
        self,
        survey_key: str = SUBSIDENCE_RAW_KEY,
        parameters: Optional[Dict[str, Any]] = None,
        registry_key_prefix: str = "subsidence",
        source_panel: str = "survey_deformation",
    ):
        """Compute displacement, velocity, acceleration and register results."""
        survey_df = self.registry.get_data(survey_key)
        if survey_df is None:
            raise KeyError(f"Survey data not found in registry key '{survey_key}'")

        result = self.subsidence_engine.build_timeseries(survey_df, parameters=parameters, input_dataset_id=survey_key)
        self.subsidence_engine.register_results(
            result,
            registry_key_prefix=registry_key_prefix,
            source_panel=source_panel,
        )
        # Store pointers for downstream stages
        return result

    def run_control_stability(
        self,
        subsidence_timeseries_key: str = SUBSIDENCE_TIMESERIES_KEY,
        subsidence_metrics_key: str = SUBSIDENCE_METRICS_KEY,
        survey_precision_mm: float = 5.0,
        significance_sigma: float = 2.0,
        parameters: Optional[Dict[str, Any]] = None,
        registry_key: str = CONTROL_STABILITY_KEY,
        source_panel: str = "survey_deformation",
    ):
        """Classify control stability and register derived table."""
        ts_df = self.registry.get_data(subsidence_timeseries_key)
        metrics_df = self.registry.get_data(subsidence_metrics_key)
        if ts_df is None or metrics_df is None:
            raise KeyError("Subsidence timeseries or metrics missing from registry")

        bundle = self.control_engine.evaluate(
            ts_df,
            metrics_df,
            survey_precision_mm=survey_precision_mm,
            significance_sigma=significance_sigma,
            parameters=parameters,
            input_dataset_ids=[subsidence_timeseries_key, subsidence_metrics_key],
        )
        self.control_engine.register_results(bundle, registry_key=registry_key, source_panel=source_panel)
        return bundle

    def run_groundwater_time_series(
        self,
        groundwater_key: str = GROUNDWATER_RAW_KEY,
        parameters: Optional[Dict[str, Any]] = None,
        registry_key_prefix: str = "groundwater",
        source_panel: str = "survey_deformation",
    ):
        """Compute descriptive groundwater rates and register results."""
        well_df = self.registry.get_data(groundwater_key)
        if well_df is None:
            raise KeyError(f"Groundwater data not found in registry key '{groundwater_key}'")

        bundle = self.groundwater_engine.build_timeseries(
            well_df, parameters=parameters, input_dataset_id=groundwater_key
        )
        self.groundwater_engine.register_results(
            bundle,
            registry_key_prefix=registry_key_prefix,
            source_panel=source_panel,
        )
        return bundle

    def run_coupling(
        self,
        subsidence_timeseries_key: str = SUBSIDENCE_TIMESERIES_KEY,
        groundwater_timeseries_key: str = GROUNDWATER_TIMESERIES_KEY,
        max_distance: float = 1500.0,
        parameters: Optional[Dict[str, float]] = None,
        registry_key: str = COUPLING_KEY,
        source_panel: str = "survey_deformation",
    ):
        """Compute spatial/temporal correlation between subsidence and groundwater."""
        subsidence_ts = self.registry.get_data(subsidence_timeseries_key)
        groundwater_ts = self.registry.get_data(groundwater_timeseries_key)
        if subsidence_ts is None or groundwater_ts is None:
            raise KeyError("Subsidence or groundwater timeseries missing from registry")

        bundle = self.coupling_engine.correlate(
            subsidence_ts,
            groundwater_ts,
            max_distance=max_distance,
            parameters=parameters,
            input_dataset_ids=[subsidence_timeseries_key, groundwater_timeseries_key],
        )
        self.coupling_engine.register_results(bundle, registry_key=registry_key, source_panel=source_panel)
        return bundle

    def run_deformation_index(
        self,
        subsidence_metrics_key: str = SUBSIDENCE_METRICS_KEY,
        stability_key: str = CONTROL_STABILITY_KEY,
        coupling_key: str = COUPLING_KEY,
        weights: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        registry_key: str = DEFORMATION_INDEX_KEY,
        source_panel: str = "survey_deformation",
    ):
        """Combine metrics into a deformation index and register results."""
        subsidence_metrics = self.registry.get_data(subsidence_metrics_key)
        stability_df = self.registry.get_data(stability_key)
        coupling_df = self.registry.get_data(coupling_key)
        if subsidence_metrics is None or stability_df is None or coupling_df is None:
            raise KeyError("Required inputs for deformation index missing from registry")

        bundle = self.index_engine.compute_index(
            subsidence_metrics,
            stability_df,
            coupling_df,
            weights=weights,
            parameters=parameters,
            input_dataset_ids=[subsidence_metrics_key, stability_key, coupling_key],
        )
        self.index_engine.register_results(bundle, registry_key=registry_key, source_panel=source_panel)
        return bundle

    # ------------------------------------------------------------------ #
    # Job registry payload wrappers
    # ------------------------------------------------------------------ #

    def _prepare_subsidence_time_series_payload(
        self, params: Dict[str, Any], progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        return self.run_subsidence_time_series(
            survey_key=params.get("survey_key", self.SUBSIDENCE_RAW_KEY),
            parameters=params.get("parameters") or {},
            registry_key_prefix=params.get("registry_key_prefix", "subsidence"),
            source_panel=params.get("source_panel", "survey_deformation"),
        )

    def _prepare_control_stability_payload(
        self, params: Dict[str, Any], progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        return self.run_control_stability(
            subsidence_timeseries_key=params.get("subsidence_timeseries_key", self.SUBSIDENCE_TIMESERIES_KEY),
            subsidence_metrics_key=params.get("subsidence_metrics_key", self.SUBSIDENCE_METRICS_KEY),
            survey_precision_mm=params.get("survey_precision_mm", 5.0),
            significance_sigma=params.get("significance_sigma", 2.0),
            parameters=params.get("parameters") or {},
            registry_key=params.get("registry_key", self.CONTROL_STABILITY_KEY),
            source_panel=params.get("source_panel", "survey_deformation"),
        )

    def _prepare_groundwater_time_series_payload(
        self, params: Dict[str, Any], progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        return self.run_groundwater_time_series(
            groundwater_key=params.get("groundwater_key", self.GROUNDWATER_RAW_KEY),
            parameters=params.get("parameters") or {},
            registry_key_prefix=params.get("registry_key_prefix", "groundwater"),
            source_panel=params.get("source_panel", "survey_deformation"),
        )

    def _prepare_coupling_payload(
        self, params: Dict[str, Any], progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        return self.run_coupling(
            subsidence_timeseries_key=params.get("subsidence_timeseries_key", self.SUBSIDENCE_TIMESERIES_KEY),
            groundwater_timeseries_key=params.get("groundwater_timeseries_key", self.GROUNDWATER_TIMESERIES_KEY),
            max_distance=params.get("max_distance", 1500.0),
            parameters=params.get("parameters") or {},
            registry_key=params.get("registry_key", self.COUPLING_KEY),
            source_panel=params.get("source_panel", "survey_deformation"),
        )

    def _prepare_deformation_index_payload(
        self, params: Dict[str, Any], progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        return self.run_deformation_index(
            subsidence_metrics_key=params.get("subsidence_metrics_key", self.SUBSIDENCE_METRICS_KEY),
            stability_key=params.get("stability_key", self.CONTROL_STABILITY_KEY),
            coupling_key=params.get("coupling_key", self.COUPLING_KEY),
            weights=params.get("weights"),
            parameters=params.get("parameters") or {},
            registry_key=params.get("registry_key", self.DEFORMATION_INDEX_KEY),
            source_panel=params.get("source_panel", "survey_deformation"),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _raise_on_errors(issues: Optional[list[ValidationIssue]]) -> None:
        if not issues:
            return
        errors = [i for i in issues if i.level.upper() == "ERROR"]
        if errors:
            msg = "; ".join(i.message for i in errors)
            raise ValueError(msg)
