"""
Coupling and interpretation engine.

Relates subsidence behaviour to groundwater behaviour using descriptive
correlations (no causation implied).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..core.data_registry_simple import DataRegistrySimple
from .data_models import DatasetProvenance

logger = logging.getLogger(__name__)


class CouplingEngine:
    """Spatial and temporal correlation between subsidence and groundwater."""

    def __init__(self, registry: Optional[DataRegistrySimple] = None, software_version: str = "unknown"):
        self.registry = registry
        self.software_version = software_version

    @staticmethod
    def _nearest_well_lookup(well_df: pd.DataFrame) -> cKDTree:
        coords = well_df[["easting", "northing"]].values
        return cKDTree(coords)

    @staticmethod
    def _classify_influence(corr: float, distance: float, max_distance: float) -> str:
        if distance > max_distance:
            return "independent"
        if corr >= 0.6:
            return "groundwater-associated"
        if corr >= 0.3:
            return "unclear"
        return "independent"

    def correlate(
        self,
        subsidence_timeseries: pd.DataFrame,
        groundwater_timeseries: pd.DataFrame,
        max_distance: float = 1500.0,
        parameters: Optional[Dict[str, float]] = None,
        input_dataset_ids: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Compute spatially nearest well for each subsidence point and a simple
        Pearson correlation on co-dated observations.
        """
        parameters = parameters or {}
        input_dataset_ids = input_dataset_ids or []

        required_sub = {"point_id", "easting", "northing", "survey_date", "delta_z_mm"}
        required_well = {"well_id", "easting", "northing", "date", "water_level"}
        missing_sub = required_sub - set(subsidence_timeseries.columns)
        missing_well = required_well - set(groundwater_timeseries.columns)
        if missing_sub:
            raise ValueError(f"Subsidence timeseries missing: {', '.join(sorted(missing_sub))}")
        if missing_well:
            raise ValueError(f"Groundwater timeseries missing: {', '.join(sorted(missing_well))}")

        wells_first = groundwater_timeseries.groupby("well_id").first().reset_index()
        tree = self._nearest_well_lookup(wells_first)

        results = []
        sub_points = subsidence_timeseries.groupby("point_id")
        well_coords = wells_first[["easting", "northing"]].values

        for pid, group in sub_points:
            point_coord = group[["easting", "northing"]].iloc[0].values
            dist, idx = tree.query(point_coord)
            nearest_well = wells_first.iloc[idx]
            well_id = nearest_well["well_id"]

            # Align on date to compute correlation (same-day only, conservative)
            point_ts = group[["survey_date", "delta_z_mm"]].copy()
            point_ts["date"] = point_ts["survey_date"].dt.date
            well_ts = groundwater_timeseries[groundwater_timeseries["well_id"] == well_id][
                ["date", "water_level"]
            ].copy()
            well_ts["date"] = well_ts["date"].dt.date

            merged = pd.merge(point_ts, well_ts, on="date", how="inner")
            corr = float(merged["delta_z_mm"].corr(merged["water_level"])) if len(merged) > 1 else 0.0
            classification = self._classify_influence(corr, dist, max_distance)

            results.append(
                {
                    "point_id": pid,
                    "well_id": well_id,
                    "distance": float(dist),
                    "max_distance": float(max_distance),
                    "pearson_corr": corr,
                    "classification": classification,
                    "aligned_epochs": len(merged),
                }
            )

        result_df = pd.DataFrame(results)
        provenance = DatasetProvenance(
            input_dataset_ids=input_dataset_ids,
            parameters={**parameters, "max_distance": max_distance},
            method="subsidence_groundwater_coupling",
            software_version=self.software_version,
        )
        return {"coupling": result_df, "provenance": provenance}

    def register_results(
        self,
        result_bundle: Dict[str, Any],
        registry_key: str = "subsidence_groundwater_coupling",
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
            result_bundle["coupling"],
            metadata=meta,
            source_panel=source_panel,
        )
