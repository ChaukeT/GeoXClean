"""
Subsidence time-series engine.

Responsibilities:
- Build time series per survey point
- Compute displacement, velocity, acceleration (deterministic fits)
- Return audit-friendly metadata for registry storage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.data_registry_simple import DataRegistrySimple
from .data_models import DatasetProvenance, PointTimeSeries, DerivedResult

logger = logging.getLogger(__name__)


@dataclass
class SubsidenceFitResult:
    timeseries: pd.DataFrame
    per_point_metrics: pd.DataFrame
    provenance: DatasetProvenance


class SubsidenceTimeSeriesEngine:
    """Deterministic subsidence calculator."""

    def __init__(self, registry: Optional[DataRegistrySimple] = None, software_version: str = "unknown"):
        self.registry = registry
        self.software_version = software_version

    @staticmethod
    def _compute_delta_velocity(group: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative displacement and per-interval velocity."""
        work = group.copy()
        work["delta_z_mm"] = (work["elevation"] - work["elevation"].iloc[0]) * 1000.0
        time_years = work["survey_date"].astype("int64") / 1e9 / (60 * 60 * 24 * 365.25)
        work["velocity_mm_per_year"] = 0.0
        work.loc[1:, "velocity_mm_per_year"] = (
            work["delta_z_mm"].diff().iloc[1:] / time_years.diff().iloc[1:]
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return work

    @staticmethod
    def _fit_trends(group: pd.DataFrame) -> Tuple[float, float, float, float]:
        """
        Fit linear and quadratic trends.

        Returns:
            velocity_mm_per_year, acceleration_mm_per_year2, r2_linear, r2_quadratic
        """
        times_years = (group["survey_date"] - group["survey_date"].min()).dt.total_seconds() / (
            60 * 60 * 24 * 365.25
        )
        y = group["elevation"].values
        if len(group) < 2 or np.all(times_years == times_years.iloc[0]):
            return 0.0, 0.0, 1.0, 1.0

        # Linear fit
        lin_coef = np.polyfit(times_years, y, deg=1)
        y_pred_lin = np.polyval(lin_coef, times_years)

        # Quadratic fit
        quad_coef = np.polyfit(times_years, y, deg=2)
        y_pred_quad = np.polyval(quad_coef, times_years)

        def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            return 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot

        velocity_mm_per_year = lin_coef[0] * 1000.0
        acceleration_mm_per_year2 = 2 * quad_coef[0] * 1000.0
        return (
            float(velocity_mm_per_year),
            float(acceleration_mm_per_year2),
            float(r2(y, y_pred_lin)),
            float(r2(y, y_pred_quad)),
        )

    def build_timeseries(
        self,
        survey_df: pd.DataFrame,
        parameters: Optional[Dict[str, Any]] = None,
        input_dataset_id: Optional[str] = None,
    ) -> SubsidenceFitResult:
        """
        Build per-point time series with displacement, velocity, and acceleration.
        """
        parameters = parameters or {}
        base = PointTimeSeries(survey_df).df
        grouped = base.groupby("point_id", sort=False)

        timeseries_list = []
        metrics_rows = []

        for point_id, group in grouped:
            enriched = self._compute_delta_velocity(group)
            velocity, acceleration, r2_lin, r2_quad = self._fit_trends(enriched)
            total_delta = float(enriched["delta_z_mm"].iloc[-1]) if not enriched.empty else 0.0
            timeseries_list.append(enriched)

            metrics_rows.append(
                {
                    "point_id": point_id,
                    "total_delta_mm": total_delta,
                    "velocity_mm_per_year": velocity,
                    "acceleration_mm_per_year2": acceleration,
                    "r2_linear": r2_lin,
                    "r2_quadratic": r2_quad,
                    "epoch_count": len(enriched),
                }
            )

        timeseries_df = pd.concat(timeseries_list, ignore_index=True) if timeseries_list else pd.DataFrame()
        metrics_df = pd.DataFrame(metrics_rows)

        provenance = DatasetProvenance(
            input_dataset_ids=[input_dataset_id] if input_dataset_id else [],
            parameters=parameters,
            method="subsidence_time_series",
            software_version=self.software_version,
        )
        return SubsidenceFitResult(timeseries_df, metrics_df, provenance)

    def register_results(
        self,
        result: SubsidenceFitResult,
        registry_key_prefix: str = "subsidence",
        source_panel: str = "survey_deformation",
    ) -> bool:
        """
        Register both timeseries and per-point metrics in the registry if available.
        """
        if self.registry is None:
            logger.warning("No registry attached; skipping registration")
            return False

        ts_key = f"{registry_key_prefix}_timeseries"
        metrics_key = f"{registry_key_prefix}_metrics"

        meta = {
            "provenance": result.provenance.__dict__,
            "timestamp": datetime.utcnow().isoformat(),
            "labels": ["DERIVED"],
        }

        ts_ok = self.registry.register_model(
            ts_key,
            result.timeseries,
            metadata=meta,
            source_panel=source_panel,
        )
        metrics_ok = self.registry.register_model(
            metrics_key,
            result.per_point_metrics,
            metadata=meta,
            source_panel=source_panel,
        )
        return ts_ok and metrics_ok

    def compute_and_register(
        self,
        survey_df: pd.DataFrame,
        parameters: Optional[Dict[str, Any]] = None,
        input_dataset_id: Optional[str] = None,
        registry_key_prefix: str = "subsidence",
        source_panel: str = "survey_deformation",
    ) -> SubsidenceFitResult:
        """Convenience wrapper to compute and push results into the registry."""
        result = self.build_timeseries(
            survey_df=survey_df,
            parameters=parameters,
            input_dataset_id=input_dataset_id,
        )
        self.register_results(result, registry_key_prefix=registry_key_prefix, source_panel=source_panel)
        return result
