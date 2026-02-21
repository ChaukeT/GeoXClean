"""
Groundwater time-series engine.

Responsible for standardising well data and computing descriptive drawdown rates.
Deliberately avoids flow equations or transmissivity assumptions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..core.data_registry_simple import DataRegistrySimple
from .data_models import DatasetProvenance, WellTimeSeries

logger = logging.getLogger(__name__)


class GroundwaterTimeSeriesEngine:
    """Deterministic groundwater statistics (survey-centric)."""

    def __init__(self, registry: Optional[DataRegistrySimple] = None, software_version: str = "unknown"):
        self.registry = registry
        self.software_version = software_version

    @staticmethod
    def _fit_drawdown(group: pd.DataFrame) -> Dict[str, float]:
        times_years = (group["date"] - group["date"].min()).dt.total_seconds() / (60 * 60 * 24 * 365.25)
        levels = group["water_level"].values
        if len(group) < 2 or np.allclose(times_years, times_years[0]):
            return {"rate_per_year": 0.0, "r2_linear": 1.0}

        coef = np.polyfit(times_years, levels, deg=1)
        pred = np.polyval(coef, times_years)
        ss_res = np.sum((levels - pred) ** 2)
        ss_tot = np.sum((levels - np.mean(levels)) ** 2)
        r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
        return {
            "rate_per_year": float(coef[0]),
            "r2_linear": float(r2),
        }

    def build_timeseries(
        self,
        well_df: pd.DataFrame,
        parameters: Optional[Dict[str, Any]] = None,
        input_dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        parameters = parameters or {}
        base = WellTimeSeries(well_df).df
        grouped = base.groupby("well_id", sort=False)

        metrics_rows = []
        for well_id, group in grouped:
            stats = self._fit_drawdown(group)
            metrics_rows.append(
                {
                    "well_id": well_id,
                    "rate_per_year": stats["rate_per_year"],
                    "r2_linear": stats["r2_linear"],
                    "epoch_count": len(group),
                }
            )

        metrics_df = pd.DataFrame(metrics_rows)
        provenance = DatasetProvenance(
            input_dataset_ids=[input_dataset_id] if input_dataset_id else [],
            parameters=parameters,
            method="groundwater_time_series",
            software_version=self.software_version,
        )

        return {"timeseries": base, "metrics": metrics_df, "provenance": provenance}

    def register_results(
        self,
        result_bundle: Dict[str, Any],
        registry_key_prefix: str = "groundwater",
        source_panel: str = "survey_deformation",
    ) -> bool:
        if self.registry is None:
            logger.warning("No registry attached; skipping registration")
            return False

        meta = {
            "provenance": result_bundle["provenance"].__dict__,
            "timestamp": datetime.utcnow().isoformat(),
            "labels": ["DERIVED"],
        }

        ts_ok = self.registry.register_model(
            f"{registry_key_prefix}_timeseries",
            result_bundle["timeseries"],
            metadata=meta,
            source_panel=source_panel,
        )
        metrics_ok = self.registry.register_model(
            f"{registry_key_prefix}_metrics",
            result_bundle["metrics"],
            metadata=meta,
            source_panel=source_panel,
        )
        return ts_ok and metrics_ok
