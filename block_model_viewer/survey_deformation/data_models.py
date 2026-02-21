"""
Lightweight data models for survey-based deformation analysis.

These classes standardise inputs from survey and groundwater tables, enforce
required fields, and carry provenance so derived outputs remain traceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Provenance containers
# ---------------------------------------------------------------------------


@dataclass
class DatasetProvenance:
    """Provenance metadata captured for every derived dataset."""

    input_dataset_ids: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    method: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    software_version: str = "unknown"
    random_state: Optional[int] = None
    notes: Optional[str] = None


@dataclass
class DerivedResult:
    """Wrapper linking a result payload to its provenance."""

    name: str
    payload: Any
    provenance: DatasetProvenance


# ---------------------------------------------------------------------------
# Survey observations
# ---------------------------------------------------------------------------


@dataclass
class PointObservation:
    """Single survey observation."""

    point_id: str
    easting: float
    northing: float
    elevation: float
    survey_date: datetime
    survey_method: Optional[str] = None
    instrument_id: Optional[str] = None
    surveyor: Optional[str] = None


class PointTimeSeries:
    """
    Time series for a survey point.

    Internally stored as a pandas DataFrame with canonical columns:
    point_id, easting, northing, elevation, survey_date, survey_method,
    instrument_id, surveyor.
    """

    REQUIRED_COLUMNS = {"point_id", "easting", "northing", "elevation", "survey_date"}

    def __init__(self, df: pd.DataFrame, source_name: str = "subsidence_raw"):
        self.source_name = source_name
        self.df = self._normalise(df)

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce expected columns and dtypes without mutating caller data."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("PointTimeSeries expects a pandas DataFrame")

        rename_map = {
            "Point ID": "point_id",
            "POINT_ID": "point_id",
            "EASTING": "easting",
            "NORTHING": "northing",
            "ELEVATION": "elevation",
            "Survey date": "survey_date",
            "SURVEY_DATE": "survey_date",
        }

        work = df.rename(columns=rename_map).copy()
        missing = PointTimeSeries.REQUIRED_COLUMNS - set(work.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        work["survey_date"] = pd.to_datetime(work["survey_date"], errors="coerce")
        if work["survey_date"].isna().any():
            bad = work[work["survey_date"].isna()]
            raise ValueError(f"Unparseable survey_date for rows: {bad.index.tolist()}")

        work = work.sort_values(["point_id", "survey_date"]).reset_index(drop=True)
        return work

    def observation_count(self) -> int:
        return len(self.df)

    def split_by_point(self) -> Dict[str, pd.DataFrame]:
        """Return dict of DataFrames keyed by point_id."""
        return {pid: g.copy() for pid, g in self.df.groupby("point_id")}


# ---------------------------------------------------------------------------
# Groundwater observations
# ---------------------------------------------------------------------------


@dataclass
class WellObservation:
    """Single groundwater observation."""

    well_id: str
    easting: float
    northing: float
    water_level: float
    date: datetime
    well_type: Optional[str] = None


class WellTimeSeries:
    """
    Time series for a groundwater well.

    Canonical columns: well_id, easting, northing, water_level, date, well_type
    """

    REQUIRED_COLUMNS = {"well_id", "easting", "northing", "water_level", "date"}

    def __init__(self, df: pd.DataFrame, source_name: str = "groundwater_raw"):
        self.source_name = source_name
        self.df = self._normalise(df)

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("WellTimeSeries expects a pandas DataFrame")

        rename_map = {
            "Well ID": "well_id",
            "WELL_ID": "well_id",
            "EASTING": "easting",
            "NORTHING": "northing",
            "Water level": "water_level",
            "WATER_LEVEL": "water_level",
            "Date": "date",
            "DATE": "date",
        }

        work = df.rename(columns=rename_map).copy()
        missing = WellTimeSeries.REQUIRED_COLUMNS - set(work.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        if work["date"].isna().any():
            bad = work[work["date"].isna()]
            raise ValueError(f"Unparseable dates for rows: {bad.index.tolist()}")

        work = work.sort_values(["well_id", "date"]).reset_index(drop=True)
        return work

    def observation_count(self) -> int:
        return len(self.df)

    def split_by_well(self) -> Dict[str, pd.DataFrame]:
        return {wid: g.copy() for wid, g in self.df.groupby("well_id")}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def ensure_time_monotonic(group: pd.DataFrame, time_col: str) -> Tuple[bool, Optional[int]]:
    """
    Check monotonicity of a time column within a grouped series.

    Returns:
        (is_monotonic, first_bad_index)
    """
    diffs = group[time_col].diff().dt.total_seconds()
    bad_idx = diffs[diffs < 0].index
    if len(bad_idx) > 0:
        return False, int(bad_idx[0])
    return True, None
