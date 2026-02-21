"""
Control stability engine (vertical datum integrity).

Classifies benchmarks as Stable / Suspect / Failed based on observed movement
relative to survey precision. Uses deterministic thresholds for auditability.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..core.data_registry_simple import DataRegistrySimple
from .data_models import DatasetProvenance

logger = logging.getLogger(__name__)


class ControlStabilityEngine:
    """Vertical control stability classifier."""

    def __init__(self, registry: Optional[DataRegistrySimple] = None, software_version: str = "unknown"):
        self.registry = registry
        self.software_version = software_version

    @staticmethod
    def _classify(movement_mm: float, sigma_threshold_mm: float) -> str:
        abs_move = abs(movement_mm)
        if abs_move <= sigma_threshold_mm:
            return "Stable"
        if abs_move <= sigma_threshold_mm * 2:
            return "Suspect"
        return "Failed"

    def evaluate(
        self,
        subsidence_timeseries: pd.DataFrame,
        subsidence_metrics: pd.DataFrame,
        survey_precision_mm: float = 5.0,
        significance_sigma: float = 2.0,
        parameters: Optional[Dict[str, Any]] = None,
        input_dataset_ids: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate control stability using displacement magnitude vs survey precision.
        """
        parameters = parameters or {}
        input_dataset_ids = input_dataset_ids or []

        ts_required = {"point_id", "delta_z_mm"}
        missing_ts = ts_required - set(subsidence_timeseries.columns)
        if missing_ts:
            raise ValueError(f"Timeseries missing columns: {', '.join(sorted(missing_ts))}")

        metrics_required = {"point_id", "total_delta_mm"}
        missing_m = metrics_required - set(subsidence_metrics.columns)
        if missing_m:
            raise ValueError(f"Metrics missing columns: {', '.join(sorted(missing_m))}")

        sigma_threshold_mm = survey_precision_mm * significance_sigma
        results = []
        for pid, group in subsidence_timeseries.groupby("point_id"):
            metrics_row = subsidence_metrics[subsidence_metrics["point_id"] == pid]
            total_delta = float(metrics_row["total_delta_mm"].iloc[0]) if not metrics_row.empty else 0.0
            residual_std = float(group["delta_z_mm"].std(ddof=0)) if len(group) > 1 else 0.0
            classification = self._classify(total_delta, sigma_threshold_mm)
            stability_score = {"Stable": 1.0, "Suspect": 0.5, "Failed": 0.0}[classification]
            results.append(
                {
                    "point_id": pid,
                    "movement_mm": total_delta,
                    "residual_std_mm": residual_std,
                    "sigma_threshold_mm": sigma_threshold_mm,
                    "classification": classification,
                    "stability_score": stability_score,
                    "epoch_count": int(len(group)),
                }
            )

        result_df = pd.DataFrame(results)
        provenance = DatasetProvenance(
            input_dataset_ids=input_dataset_ids,
            parameters={
                **parameters,
                "survey_precision_mm": survey_precision_mm,
                "significance_sigma": significance_sigma,
            },
            method="control_stability",
            software_version=self.software_version,
        )
        return {"control_stability": result_df, "provenance": provenance}

    def register_results(
        self,
        result_bundle: Dict[str, Any],
        registry_key: str = "control_stability",
        source_panel: str = "survey_deformation",
    ) -> bool:
        if self.registry is None:
            logger.warning("No registry attached; skipping registration")
            return False

        df = result_bundle["control_stability"]
        provenance = result_bundle["provenance"]
        meta = {
            "provenance": provenance.__dict__,
            "timestamp": datetime.utcnow().isoformat(),
            "labels": ["DERIVED"],
        }
        return self.registry.register_model(registry_key, df, metadata=meta, source_panel=source_panel)
