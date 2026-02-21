"""
Reporting helpers for survey deformation module.

Produces audit-friendly structures (DataFrames, metadata) ready for export to
PDF/Excel by the UI layer. No file I/O here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


class ReportingEngine:
    """Assemble derived outputs into report-ready structures."""

    def build_report_payload(
        self,
        subsidence_metrics: pd.DataFrame,
        stability_df: pd.DataFrame,
        groundwater_metrics: Optional[pd.DataFrame] = None,
        coupling_df: Optional[pd.DataFrame] = None,
        deformation_index_df: Optional[pd.DataFrame] = None,
        provenance: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Bundle results for downstream export layers.
        """
        payload = {
            "generated_at": datetime.utcnow().isoformat(),
            "provenance": provenance or {},
            "tables": {
                "subsidence_metrics": subsidence_metrics,
                "control_stability": stability_df,
            },
        }
        if groundwater_metrics is not None:
            payload["tables"]["groundwater_metrics"] = groundwater_metrics
        if coupling_df is not None:
            payload["tables"]["coupling"] = coupling_df
        if deformation_index_df is not None:
            payload["tables"]["deformation_index"] = deformation_index_df
        return payload
