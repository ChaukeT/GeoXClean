"""
Survey noise filtering helpers.

These functions never mutate the caller's DataFrame. They return filtered
series alongside flags so callers can store RAW vs FILTERED separately.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SurveyNoiseFilter:
    """Deterministic filters for survey elevation time series."""

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def moving_average(
        self, df: pd.DataFrame, value_col: str = "elevation", window: int = 3
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Apply centred moving average.

        Returns a DataFrame with a new column '{value_col}_filtered'.
        """
        work = df.copy()
        work = work.sort_values("survey_date")
        work[f"{value_col}_filtered"] = (
            work[value_col].rolling(window=window, center=True, min_periods=1).mean()
        )
        meta = {"method": "moving_average", "window": window}
        return work, meta

    def zscore_outlier_flags(
        self, df: pd.DataFrame, value_col: str = "elevation", z_threshold: float = 3.5
    ) -> pd.Series:
        """Flag outliers using Z-score without altering values."""
        vals = df[value_col].astype(float)
        mean = vals.mean()
        std = vals.std(ddof=0)
        if std == 0:
            return pd.Series(False, index=df.index)
        zscores = (vals - mean) / std
        return zscores.abs() > z_threshold

    def mad_outlier_flags(
        self, df: pd.DataFrame, value_col: str = "elevation", threshold: float = 3.5
    ) -> pd.Series:
        """Flag outliers using Median Absolute Deviation."""
        vals = df[value_col].astype(float)
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        if mad == 0:
            return pd.Series(False, index=df.index)
        modified_z = 0.6745 * (vals - median) / mad
        return np.abs(modified_z) > threshold

    def apply_filter(
        self,
        df: pd.DataFrame,
        method: str = "moving_average",
        value_col: str = "elevation",
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Dispatch filter by method name.

        Supported methods:
        - moving_average
        - zscore (flags only)
        - mad (flags only)
        """
        method = method.lower()
        if method == "moving_average":
            return self.moving_average(df, value_col=value_col, **kwargs)
        if method == "zscore":
            flags = self.zscore_outlier_flags(df, value_col=value_col, **kwargs)
            work = df.copy()
            work[f"{value_col}_outlier_zscore"] = flags
            meta = {"method": "zscore", **kwargs}
            return work, meta
        if method == "mad":
            flags = self.mad_outlier_flags(df, value_col=value_col, **kwargs)
            work = df.copy()
            work[f"{value_col}_outlier_mad"] = flags
            meta = {"method": "mad", **kwargs}
            return work, meta

        raise ValueError(f"Unsupported filter method: {method}")
