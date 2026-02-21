"""
Deformation index engine.

Combines subsidence rate, acceleration, control stability, and groundwater
correlation strength into a single interpretable metric (0–1 and 0–100).
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


class DeformationIndexEngine:
    """Computes a weighted deformation index per survey point."""

    def __init__(self, registry: Optional[DataRegistrySimple] = None, software_version: str = "unknown"):
        self.registry = registry
        self.software_version = software_version

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        """Normalize to 0-1 using absolute values to capture magnitude."""
        abs_vals = series.abs()
        max_val = abs_vals.max()
        if max_val == 0 or np.isnan(max_val):
            return pd.Series(0.0, index=series.index)
        return abs_vals / max_val

    def compute_index(
        self,
        subsidence_metrics: pd.DataFrame,
        stability_df: pd.DataFrame,
        coupling_df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        input_dataset_ids: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Compute the deformation index.

        weights keys:
            velocity, acceleration, stability, coupling
        """
        weights = weights or {"velocity": 0.35, "acceleration": 0.25, "stability": 0.25, "coupling": 0.15}
        parameters = parameters or {}
        input_dataset_ids = input_dataset_ids or []

        required_sub = {"point_id", "velocity_mm_per_year", "acceleration_mm_per_year2"}
        required_stab = {"point_id", "stability_score"}
        required_cpl = {"point_id", "pearson_corr"}

        if missing := required_sub - set(subsidence_metrics.columns):
            raise ValueError(f"Subsidence metrics missing: {', '.join(sorted(missing))}")
        if missing := required_stab - set(stability_df.columns):
            raise ValueError(f"Stability results missing: {', '.join(sorted(missing))}")
        if missing := required_cpl - set(coupling_df.columns):
            raise ValueError(f"Coupling results missing: {', '.join(sorted(missing))}")

        merged = subsidence_metrics.merge(stability_df, on="point_id", how="left").merge(
            coupling_df[["point_id", "pearson_corr"]], on="point_id", how="left"
        )

        merged["velocity_norm"] = self._normalize_series(merged["velocity_mm_per_year"])
        merged["acceleration_norm"] = self._normalize_series(merged["acceleration_mm_per_year2"])
        merged["stability_norm"] = merged["stability_score"].clip(lower=0.0, upper=1.0)
        merged["coupling_norm"] = merged["pearson_corr"].abs().fillna(0.0).clip(0.0, 1.0)

        merged["deformation_index_0_1"] = (
            merged["velocity_norm"] * weights.get("velocity", 0.0)
            + merged["acceleration_norm"] * weights.get("acceleration", 0.0)
            + merged["stability_norm"] * weights.get("stability", 0.0)
            + merged["coupling_norm"] * weights.get("coupling", 0.0)
        )
        # Normalize by sum of weights to keep bounded
        weight_sum = sum(weights.values()) or 1.0
        merged["deformation_index_0_1"] = merged["deformation_index_0_1"] / weight_sum
        merged["deformation_index_0_100"] = merged["deformation_index_0_1"] * 100.0
        merged = merged.sort_values("deformation_index_0_1", ascending=False).reset_index(drop=True)
        merged["rank"] = merged.index + 1

        provenance = DatasetProvenance(
            input_dataset_ids=input_dataset_ids,
            parameters={**parameters, "weights": weights},
            method="deformation_index",
            software_version=self.software_version,
        )
        return {"deformation_index": merged, "provenance": provenance}

    def register_results(
        self,
        result_bundle: Dict[str, Any],
        registry_key: str = "deformation_index",
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
        return self.registry.register_model(
            registry_key,
            result_bundle["deformation_index"],
            metadata=meta,
            source_panel=source_panel,
        )
