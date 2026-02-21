"""
DRILLHOLE VALIDATION ENGINE (Validation Only, No Auto-Fixing)

Author: GeoX

Purpose: Perform complete collar, survey, assay, lithology, QAQC and cross-table QC.

DESIGN PRINCIPLES:
- Validation NEVER raises exceptions - all errors are returned as ValidationViolation objects
- Schema mismatches are reported as violations, not crashes
- All column accesses are guarded
- Supports multiple column naming conventions (e.g., depth vs depth_from)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set, Callable
from datetime import datetime
import hashlib
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class ValidationViolation:
    table: str
    rule_code: str
    severity: str      # ERROR / WARNING / INFO
    hole_id: str
    row_index: int
    message: str


@dataclass
class ValidationConfig:
    # Interval tolerances
    max_interval_gap: float = 0.10
    max_small_overlap: float = 0.02
    standard_sample_length: Optional[float] = None
    allow_nonstandard_samples: bool = True

    # Survey limits
    max_dip_change_deg: float = 10.0
    max_az_change_deg: float = 20.0
    dip_min: float = -90.0
    dip_max: float = 90.0
    az_min: float = 0.0
    az_max: float = 360.0

    # Survey extra QC
    survey_start_tolerance: float = 2.0
    survey_td_tolerance: float = 5.0
    max_survey_spacing: float = 30.0
    dip_reversal_deg: float = 45.0
    az_reversal_deg: float = 90.0

    # Topography check
    topo_elevation_col: Optional[str] = None
    max_elev_diff: float = 5.0

    # QAQC
    primary_grade_col: Optional[str] = None
    blank_max_value: Optional[float] = None
    duplicate_max_rel_diff: float = 0.1
    min_qaqc_ratio: float = 0.05
    crm_specs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Business rules
    hole_id_prefixes: Optional[List[str]] = None
    
    # Configurable column names (allows schema flexibility)
    survey_depth_col: Optional[str] = None  # Auto-detected if None
    lithology_code_col: str = "lith_code"   # Default, but can be configured


@dataclass
class ValidationResult:
    violations: List[ValidationViolation]
    # Metadata about validation run
    tables_validated: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)
    # New fields for registry integration
    timestamp: Optional[str] = None  # ISO8601 timestamp
    config_hash: Optional[str] = None  # SHA256 of config for reproducibility
    
    @property
    def status(self) -> str:
        """
        Compute validation status from violations.
        
        Returns:
            "PASS" if no violations
            "WARN" if only warnings/info
            "FAIL" if any errors
        """
        if not self.violations:
            return "PASS"
        
        has_errors = any(v.severity == "ERROR" for v in self.violations)
        if has_errors:
            return "FAIL"
        
        has_warnings = any(v.severity == "WARNING" for v in self.violations)
        if has_warnings:
            return "WARN"
        
        # Only INFO level violations
        return "PASS"
    
    @property
    def fatal_count(self) -> int:
        """Count of ERROR-level violations."""
        return sum(1 for v in self.violations if v.severity == "ERROR")
    
    @property
    def warn_count(self) -> int:
        """Count of WARNING-level violations."""
        return sum(1 for v in self.violations if v.severity == "WARNING")
    
    @property
    def info_count(self) -> int:
        """Count of INFO-level violations."""
        return sum(1 for v in self.violations if v.severity == "INFO")
    
    @property
    def ok(self) -> bool:
        """Check if validation passed (no errors, warnings allowed)."""
        return self.status in ("PASS", "WARN")
    
    def get_violations_summary(self) -> Dict[str, Any]:
        """
        Get a summary breakdown of violations by table and type.
        
        Returns:
            Dictionary with counts by table and by rule_code
        """
        by_table: Dict[str, int] = {}
        by_rule: Dict[str, int] = {}
        by_severity: Dict[str, int] = {"ERROR": 0, "WARNING": 0, "INFO": 0}
        
        for v in self.violations:
            by_table[v.table] = by_table.get(v.table, 0) + 1
            by_rule[v.rule_code] = by_rule.get(v.rule_code, 0) + 1
            by_severity[v.severity] = by_severity.get(v.severity, 0) + 1
        
        return {
            "by_table": by_table,
            "by_rule": by_rule,
            "by_severity": by_severity,
            "total": len(self.violations),
        }


# =========================================================
# SCHEMA DETECTION UTILITIES
# =========================================================

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find a column in DataFrame by checking multiple candidate names.
    Returns the first matching column name, or None if not found.
    """
    if df is None or df.empty:
        return None
    for col in candidates:
        if col in df.columns:
            return col
        # Case-insensitive check
        for actual_col in df.columns:
            if actual_col.lower() == col.lower():
                return actual_col
    return None


def _check_required_columns(
    df: pd.DataFrame, 
    table_name: str, 
    required: List[str],
    violations: List[ValidationViolation]
) -> bool:
    """
    Check if required columns exist. Returns True if all exist.
    Adds SCHEMA_ERROR violations if columns are missing.
    """
    if df is None:
        violations.append(ValidationViolation(
            table_name, "SCHEMA_ERROR", "ERROR",
            "", -1,
            f"{table_name.upper()} table is None (not provided)."
        ))
        return False
    
    missing = [col for col in required if col not in df.columns]
    if missing:
        violations.append(ValidationViolation(
            table_name, "SCHEMA_ERROR", "ERROR",
            "", -1,
            f"{table_name.upper()} missing required columns: {', '.join(missing)}. "
            f"Available columns: {', '.join(df.columns.tolist())}"
        ))
        return False
    return True


# =========================================================
# UTILITIES
# =========================================================

def _angular_diff_deg(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    diff = np.abs(a2 - a1)
    return np.where(diff > 180, 360 - diff, diff)


def _sort_intervals(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["hole_id", "from_depth", "to_depth"], kind="mergesort")


def _group(df):
    if df.empty:
        return []
    return df.groupby("hole_id", sort=False)


def _safe_get_td(collars: pd.DataFrame, hole_id: Any) -> Optional[float]:
    """
    Safely get total_depth for a hole_id. Returns None if not found.
    Handles type mismatches between hole_id types.
    """
    if collars.empty or "hole_id" not in collars.columns or "total_depth" not in collars.columns:
        return None
    
    # Try exact match first
    match = collars.loc[collars["hole_id"] == hole_id, "total_depth"]
    if not match.empty:
        return float(match.values[0])
    
    # Try string conversion match
    hid_str = str(hole_id)
    match = collars.loc[collars["hole_id"].astype(str) == hid_str, "total_depth"]
    if not match.empty:
        return float(match.values[0])
    
    return None


# =========================================================
# COLLAR VALIDATION
# =========================================================

def validate_collars(collars: pd.DataFrame, cfg: ValidationConfig) -> List[ValidationViolation]:
    """
    Validate collar data. Returns violations list, never raises exceptions.
    """
    v = []

    # Handle None or empty DataFrame
    if collars is None:
        v.append(ValidationViolation(
            "collars", "SCHEMA_ERROR", "ERROR",
            "", -1,
            "Collars table is None (not provided)."
        ))
        return v
    
    if collars.empty:
        # Empty collars is a warning, not an error (may be intentional)
        v.append(ValidationViolation(
            "collars", "COLLAR_EMPTY", "WARNING",
            "", -1,
            "Collars table is empty."
        ))
        return v

    # Check required columns - support common aliases
    hole_id_col = _find_column(collars, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    easting_col = _find_column(collars, ["easting", "x", "east", "EASTING", "X"])
    northing_col = _find_column(collars, ["northing", "y", "north", "NORTHING", "Y"])
    elevation_col = _find_column(collars, ["elevation", "z", "elev", "rl", "ELEVATION", "Z", "RL"])
    td_col = _find_column(collars, ["total_depth", "depth", "max_depth", "length", "TOTAL_DEPTH", "DEPTH"])

    missing = []
    if not hole_id_col:
        missing.append("hole_id")
    if not easting_col:
        missing.append("easting/x")
    if not northing_col:
        missing.append("northing/y")
    if not elevation_col:
        missing.append("elevation/z")
    if not td_col:
        missing.append("total_depth")
    
    if missing:
        v.append(ValidationViolation(
            "collars", "SCHEMA_ERROR", "ERROR",
            "", -1,
            f"Collars missing required columns: {', '.join(missing)}. "
            f"Available: {', '.join(collars.columns.tolist())}"
        ))
        return v

    # Use detected column names for validation
    required_cols = [hole_id_col, easting_col, northing_col, elevation_col, td_col]
    
    # Missing values - report which specific fields are missing
    for idx, row in collars[collars[required_cols].isna().any(axis=1)].iterrows():
        missing_fields = [col for col in required_cols if pd.isna(row[col])]
        missing_str = ", ".join(missing_fields)
        v.append(ValidationViolation(
            "collars", "COLLAR_MISSING_FIELDS", "ERROR",
            str(row.get(hole_id_col, "Unknown")), idx,
            f"Collar missing required fields: {missing_str}."
        ))

    # Duplicate hole IDs
    for idx, row in collars[collars[hole_id_col].duplicated(keep=False)].iterrows():
        v.append(ValidationViolation(
            "collars", "COLLAR_DUPLICATE_ID", "ERROR",
            str(row[hole_id_col]), idx,
            "Duplicate hole_id detected."
        ))

    # Topo elevation mismatch
    if cfg.topo_elevation_col and cfg.topo_elevation_col in collars.columns:
        diff = (collars[elevation_col] - collars[cfg.topo_elevation_col]).abs()
        for idx, row in collars[diff > cfg.max_elev_diff].iterrows():
            v.append(ValidationViolation(
                "collars", "COLLAR_TOPO_MISMATCH", "WARNING",
                str(row[hole_id_col]), idx,
                f"Elevation differs from topo by {diff.loc[idx]:.2f} m."
            ))

    return v


# =========================================================
# SURVEY VALIDATION
# =========================================================

def validate_surveys(surveys: pd.DataFrame, collars: pd.DataFrame, cfg: ValidationConfig) -> List[ValidationViolation]:
    """
    Validate survey data. Returns violations list, never raises exceptions.
    Supports both 'depth' (single column) and 'depth_from'/'depth_to' (interval) schemas.
    """
    v = []

    # Handle None DataFrame
    if surveys is None:
        v.append(ValidationViolation(
            "surveys", "SCHEMA_ERROR", "ERROR",
            "", -1,
            "Surveys table is None (not provided)."
        ))
        return v

    if surveys.empty:
        return v  # Empty surveys is OK (some projects don't have survey data)

    # Detect column names - support multiple schemas
    hole_id_col = _find_column(surveys, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    azimuth_col = _find_column(surveys, ["azimuth", "azi", "bearing", "AZIMUTH", "AZI"])
    dip_col = _find_column(surveys, ["dip", "inclination", "incl", "DIP", "INCL"])
    
    # Survey depth can be single column OR interval - detect schema
    depth_col = cfg.survey_depth_col  # Use config if specified
    if not depth_col:
        # Try single depth column first (most common for surveys)
        depth_col = _find_column(surveys, ["depth", "DEPTH", "md", "MD", "measured_depth"])
        
    # If single depth not found, try interval columns
    depth_from_col = None
    depth_to_col = None
    use_interval_schema = False
    
    if not depth_col:
        depth_from_col = _find_column(surveys, ["depth_from", "from_depth", "from", "FROM", "DEPTH_FROM"])
        depth_to_col = _find_column(surveys, ["depth_to", "to_depth", "to", "TO", "DEPTH_TO"])
        if depth_from_col:
            depth_col = depth_from_col  # Use from_depth as the depth reference
            use_interval_schema = True
    
    # Check required columns
    missing = []
    if not hole_id_col:
        missing.append("hole_id")
    if not depth_col:
        missing.append("depth (or depth_from)")
    if not azimuth_col:
        missing.append("azimuth")
    if not dip_col:
        missing.append("dip")
    
    if missing:
        v.append(ValidationViolation(
            "surveys", "SCHEMA_ERROR", "ERROR",
            "", -1,
            f"Surveys missing required columns: {', '.join(missing)}. "
            f"Available: {', '.join(surveys.columns.tolist())}"
        ))
        return v

    required_survey = [hole_id_col, depth_col, azimuth_col, dip_col]
    
    # Check for missing values in required fields
    for idx, row in surveys[surveys[required_survey].isna().any(axis=1)].iterrows():
        missing_fields = [col for col in required_survey if pd.isna(row[col])]
        missing_str = ", ".join(missing_fields)
        v.append(ValidationViolation(
            "surveys", "SURVEY_MISSING_FIELDS", "ERROR",
            str(row.get(hole_id_col, "Unknown")), idx,
            f"Survey missing required fields: {missing_str}."
        ))

    # Build collar_ids set safely
    collar_ids: Set[str] = set()
    if collars is not None and not collars.empty:
        collar_hole_col = _find_column(collars, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
        if collar_hole_col:
            collar_ids = set(collars[collar_hole_col].astype(str))

    for hid, g in _group(surveys):
        hid_str = str(hid)
        g = g.sort_values(depth_col)

        # Missing collar
        if collar_ids and hid_str not in collar_ids:
            for idx in g.index:
                v.append(ValidationViolation(
                    "surveys", "SURVEY_NO_COLLAR", "ERROR",
                    hid_str, idx,
                    "Survey exists for a hole with no collar entry."
                ))

        try:
            depths = g[depth_col].to_numpy(dtype=float)
            dips = g[dip_col].to_numpy(dtype=float)
            azs = g[azimuth_col].to_numpy(dtype=float)
        except (ValueError, TypeError) as e:
            v.append(ValidationViolation(
                "surveys", "SURVEY_INVALID_DATA", "ERROR",
                hid_str, -1,
                f"Survey data contains non-numeric values: {str(e)}"
            ))
            continue

        if len(depths) == 0:
            continue

        # Depth monotonic
        bad = np.where(np.diff(depths) <= 0)[0]
        for i in bad:
            v.append(ValidationViolation(
                "surveys", "SURVEY_NON_MONOTONIC", "ERROR",
                hid_str, g.index[i+1],
                "Survey depths are not strictly increasing."
            ))

        # Dip range
        for idx, row in g[(g[dip_col] < cfg.dip_min) | (g[dip_col] > cfg.dip_max)].iterrows():
            v.append(ValidationViolation(
                "surveys", "SURVEY_DIP_RANGE", "ERROR",
                hid_str, idx,
                f"Dip {row[dip_col]} outside [{cfg.dip_min}°, {cfg.dip_max}°]"
            ))

        # Azimuth range
        for idx, row in g[(g[azimuth_col] < cfg.az_min) | (g[azimuth_col] > cfg.az_max)].iterrows():
            v.append(ValidationViolation(
                "surveys", "SURVEY_AZ_RANGE", "ERROR",
                hid_str, idx,
                f"Azimuth {row[azimuth_col]} outside [{cfg.az_min}°, {cfg.az_max}°]"
            ))

        # Curvature
        d_depth = np.diff(depths)
        with np.errstate(divide='ignore', invalid='ignore'):
            dip_rate = np.where(d_depth > 0, np.abs(np.diff(dips)) / d_depth * 10, 0)
            az_rate = np.where(d_depth > 0, _angular_diff_deg(azs[:-1], azs[1:]) / d_depth * 10, 0)

        # Curvature violations
        for i, rate in enumerate(dip_rate):
            if np.isfinite(rate) and rate > cfg.max_dip_change_deg:
                v.append(ValidationViolation(
                    "surveys", "SURVEY_DIP_CURVATURE", "WARNING",
                    hid_str, g.index[i+1],
                    f"High dip change: {rate:.1f}° per 10 m."
                ))

        for i, rate in enumerate(az_rate):
            if np.isfinite(rate) and rate > cfg.max_az_change_deg:
                v.append(ValidationViolation(
                    "surveys", "SURVEY_AZ_CURVATURE", "WARNING",
                    hid_str, g.index[i+1],
                    f"High azimuth change: {rate:.1f}° per 10 m."
                ))

        # Missing survey at start
        if depths[0] > cfg.survey_start_tolerance:
            v.append(ValidationViolation(
                "surveys", "SURVEY_START_MISSING", "WARNING",
                hid_str, g.index[0],
                f"First survey at {depths[0]} m; expected near 0–{cfg.survey_start_tolerance} m."
            ))

        # Missing survey at TD - SAFE access
        td = _safe_get_td(collars, hid)
        if td is not None and len(depths) > 0:
            if td - depths[-1] > cfg.survey_td_tolerance:
                v.append(ValidationViolation(
                    "surveys", "SURVEY_NOT_TO_TD", "WARNING",
                    hid_str, g.index[-1],
                    f"Survey ends {td - depths[-1]:.1f} m above TD ({td} m)."
                ))

        # Survey spacing
        spacing = np.diff(depths)
        for i, sp in enumerate(spacing):
            if sp > cfg.max_survey_spacing:
                v.append(ValidationViolation(
                    "surveys", "SURVEY_SPACING_LARGE", "WARNING",
                    hid_str, g.index[i+1],
                    f"Survey spacing {sp:.1f} m > max {cfg.max_survey_spacing:.1f} m."
                ))

    return v


# =========================================================
# INTERVAL VALIDATION (ASSAYS + LITHOLOGIES)
# =========================================================

def validate_intervals(
    df: pd.DataFrame, 
    collars: pd.DataFrame, 
    table: str, 
    cfg: ValidationConfig, 
    require_code: Optional[str] = None
) -> List[ValidationViolation]:
    """
    Validate interval data (assays or lithology). Returns violations list, never raises exceptions.
    """
    v = []

    # Handle None DataFrame
    if df is None:
        v.append(ValidationViolation(
            table, "SCHEMA_ERROR", "ERROR",
            "", -1,
            f"{table.upper()} table is None (not provided)."
        ))
        return v

    if df.empty:
        return v  # Empty table is OK

    # Detect column names
    hole_id_col = _find_column(df, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    from_col = _find_column(df, ["from_depth", "depth_from", "from", "FROM", "DEPTH_FROM", "MFROM"])
    to_col = _find_column(df, ["to_depth", "depth_to", "to", "TO", "DEPTH_TO", "MTO"])
    
    # Check required columns
    missing = []
    if not hole_id_col:
        missing.append("hole_id")
    if not from_col:
        missing.append("from_depth/from")
    if not to_col:
        missing.append("to_depth/to")
    
    if missing:
        v.append(ValidationViolation(
            table, "SCHEMA_ERROR", "ERROR",
            "", -1,
            f"{table.upper()} missing required columns: {', '.join(missing)}. "
            f"Available: {', '.join(df.columns.tolist())}"
        ))
        return v

    # Check require_code column if specified
    code_col = None
    if require_code:
        code_col = _find_column(df, [require_code, require_code.upper(), require_code.lower()])
        if not code_col:
            v.append(ValidationViolation(
                table, "SCHEMA_WARNING", "WARNING",
                "", -1,
                f"{table.upper()} missing optional column '{require_code}'. "
                f"Code validation will be skipped."
            ))

    required_interval = [hole_id_col, from_col, to_col]

    # Check for missing values in required fields
    for idx, row in df[df[required_interval].isna().any(axis=1)].iterrows():
        missing_fields = [col for col in required_interval if pd.isna(row[col])]
        missing_str = ", ".join(missing_fields)
        v.append(ValidationViolation(
            table, f"{table.upper()}_MISSING_FIELDS", "ERROR",
            str(row.get(hole_id_col, "Unknown")), idx,
            f"{table.capitalize()} missing required fields: {missing_str}."
        ))

    # Build collar TD map safely
    collar_td: Dict[Any, float] = {}
    if collars is not None and not collars.empty:
        collar_hole_col = _find_column(collars, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
        collar_td_col = _find_column(collars, ["total_depth", "depth", "max_depth", "length", "TOTAL_DEPTH"])
        if collar_hole_col and collar_td_col:
            try:
                collar_td = collars.set_index(collar_hole_col)[collar_td_col].to_dict()
            except Exception:
                pass  # Proceed without TD validation

    # Sort intervals
    try:
        df_sorted = df.sort_values([hole_id_col, from_col, to_col], kind="mergesort")
    except Exception:
        df_sorted = df  # Proceed with unsorted if sorting fails

    for hid, g in _group(df_sorted):
        hid_str = str(hid)
        try:
            g_d = g.sort_values(from_col)
        except Exception:
            g_d = g
        prev_to = None

        for idx, row in g_d.iterrows():

            f = row[from_col]
            t = row[to_col]

            # Missing code check
            if code_col and pd.isna(row.get(code_col)):
                v.append(ValidationViolation(
                    table, f"{table.upper()}_MISSING_CODE", "ERROR",
                    hid_str, idx,
                    f"{table} interval missing required '{require_code}'."
                ))

            # Non-numeric depths - specify which one is missing
            if pd.isna(f) or pd.isna(t):
                missing_depths = []
                if pd.isna(f):
                    missing_depths.append("from_depth")
                if pd.isna(t):
                    missing_depths.append("to_depth")
                missing_str = " and ".join(missing_depths)
                v.append(ValidationViolation(
                    table, f"{table.upper()}_NON_NUMERIC_DEPTH", "ERROR",
                    hid_str, idx,
                    f"Non-numeric {missing_str}."
                ))
                continue

            # Convert to float safely
            try:
                f = float(f)
                t = float(t)
            except (ValueError, TypeError):
                v.append(ValidationViolation(
                    table, f"{table.upper()}_NON_NUMERIC_DEPTH", "ERROR",
                    hid_str, idx,
                    f"Cannot convert depths to numbers: from={row[from_col]}, to={row[to_col]}"
                ))
                continue

            # Negative / zero length
            if t <= f:
                v.append(ValidationViolation(
                    table, f"{table.upper()}_NEGATIVE_LENGTH", "ERROR",
                    hid_str, idx,
                    f"Interval length invalid: {f}–{t}"
                ))

            # Gaps / overlaps
            if prev_to is not None:
                if f - prev_to > cfg.max_interval_gap:
                    v.append(ValidationViolation(
                        table, f"{table.upper()}_GAP", "WARNING",
                        hid_str, idx,
                        f"Gap of {f - prev_to:.3f} m before this interval."
                    ))
                if f < prev_to:
                    v.append(ValidationViolation(
                        table, f"{table.upper()}_OVERLAP", "ERROR",
                        hid_str, idx,
                        f"Overlap of {prev_to - f:.3f} m."
                    ))

            prev_to = t

            # Exceeds TD
            td = collar_td.get(hid) or collar_td.get(hid_str)
            if td is not None and t > td + 1e-3:
                v.append(ValidationViolation(
                    table, f"{table.upper()}_BEYOND_TD", "ERROR",
                    hid_str, idx,
                    f"Interval to_depth {t} exceeds TD {td}"
                ))

    return v


# =========================================================
# QAQC VALIDATION
# =========================================================

def validate_qaqc(assays: pd.DataFrame, cfg: ValidationConfig) -> List[ValidationViolation]:
    """
    Validate QAQC data. Returns violations list, never raises exceptions.
    All column accesses are guarded - missing QAQC columns simply skip those checks.
    """
    v = []
    
    # Handle None or empty DataFrame
    if assays is None or assays.empty:
        return v
    
    if cfg.primary_grade_col is None:
        return v  # No grade column configured, skip QAQC validation

    col = cfg.primary_grade_col
    if col not in assays.columns:
        # Try to find the column case-insensitively
        col = _find_column(assays, [cfg.primary_grade_col, cfg.primary_grade_col.upper(), cfg.primary_grade_col.lower()])
        if not col:
            return v  # Grade column not found, skip QAQC validation

    # Check for qaqc_type column - GUARDED ACCESS
    qaqc_type_col = _find_column(assays, ["qaqc_type", "QAQC_TYPE", "qaqc", "QAQC", "sample_type", "SAMPLE_TYPE"])
    if not qaqc_type_col:
        # No QAQC type column - skip QAQC validation entirely (this is OK)
        return v
    
    # Detect hole_id column
    hole_id_col = _find_column(assays, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    if not hole_id_col:
        hole_id_col = "hole_id"  # Use default, will return "Unknown" if missing

    # Blank samples check
    if cfg.blank_max_value is not None:
        try:
            blank_mask = assays[qaqc_type_col].astype(str).str.upper() == "BLANK"
            for idx, row in assays[blank_mask].iterrows():
                val = row[col]
                if pd.notna(val) and val > cfg.blank_max_value:
                    v.append(ValidationViolation(
                        "assays", "QAQC_BLANK_FAIL", "ERROR",
                        str(row.get(hole_id_col, "Unknown")), idx,
                        f"Blank value {val} > allowed {cfg.blank_max_value}"
                    ))
        except Exception as e:
            logger.warning(f"Error in blank validation: {e}")

    # CRM check - GUARDED ACCESS
    qaqc_ref_col = _find_column(assays, ["qaqc_reference_id", "QAQC_REFERENCE_ID", "crm_code", "CRM_CODE", "standard_id"])
    if qaqc_ref_col and cfg.crm_specs:
        try:
            std_mask = assays[qaqc_type_col].astype(str).str.upper() == "STANDARD"
            stds = assays[std_mask]
            for idx, row in stds.iterrows():
                code = str(row[qaqc_ref_col])
                if code not in cfg.crm_specs:
                    v.append(ValidationViolation(
                        "assays", "QAQC_STD_UNKNOWN", "WARNING",
                        str(row.get(hole_id_col, "Unknown")), idx,
                        f"Unknown CRM {code}"
                    ))
                    continue

                spec = cfg.crm_specs[code]
                val = row[col]

                if pd.notna(val):
                    mean_val = spec.get("mean", 0)
                    if mean_val != 0:
                        diff = abs(val - mean_val)
                        if diff > spec.get("tol_abs", 999) or diff / mean_val > spec.get("tol_pct", 999):
                            v.append(ValidationViolation(
                                "assays", "QAQC_STD_FAIL", "ERROR",
                                str(row.get(hole_id_col, "Unknown")), idx,
                                f"Standard {code} failed tolerance check."
                            ))
        except Exception as e:
            logger.warning(f"Error in CRM validation: {e}")

    # Duplicate check - GUARDED ACCESS
    parent_col = _find_column(assays, ["parent_sample_id", "PARENT_SAMPLE_ID", "parent_id", "original_id"])
    sample_id_col = _find_column(assays, ["sample_id", "SAMPLE_ID", "sampleid"])
    
    if parent_col and sample_id_col:
        try:
            dup_mask = assays[qaqc_type_col].astype(str).str.upper().isin(["DUPLICATE", "CRD", "DUP"])
            dups = assays[dup_mask]

            parent_map = assays.set_index(sample_id_col)

            for idx, row in dups.iterrows():
                pid = row[parent_col]
                if pid in parent_map.index:
                    parent = parent_map.loc[pid]
                    parent_val = parent[col] if isinstance(parent, pd.Series) else parent[col].iloc[0]
                    if pd.notna(parent_val) and pd.notna(row[col]) and parent_val != 0:
                        rel = abs(parent_val - row[col]) / parent_val
                        if rel > cfg.duplicate_max_rel_diff:
                            v.append(ValidationViolation(
                                "assays", "QAQC_DUP_FAIL", "WARNING",
                                str(row.get(hole_id_col, "Unknown")), idx,
                                f"Duplicate difference {rel*100:.1f}% exceeds limit."
                            ))
        except Exception as e:
            logger.warning(f"Error in duplicate validation: {e}")

    # QAQC insertion rate
    try:
        qaqc_types = assays[qaqc_type_col].astype(str).str.upper()
        cnt_original = len(assays[qaqc_types == "ORIGINAL"])
        cnt_qaqc = len(assays[qaqc_types != "ORIGINAL"])
        if cnt_original > 0:
            ratio = cnt_qaqc / cnt_original
            if ratio < cfg.min_qaqc_ratio:
                v.append(ValidationViolation(
                    "assays", "QAQC_RATE_LOW", "WARNING",
                    "", -1,
                    f"QAQC insertion ratio {ratio:.3f} below minimum {cfg.min_qaqc_ratio:.3f}"
                ))
    except Exception as e:
        logger.warning(f"Error in QAQC rate validation: {e}")

    return v


# =========================================================
# CROSS TABLE CONSISTENCY
# =========================================================

def validate_cross_table(collars, surveys, assays, lith) -> List[ValidationViolation]:
    """
    Validate cross-table consistency. Returns violations list, never raises exceptions.
    """
    v = []

    # Handle None collars
    if collars is None or collars.empty:
        return v  # Can't do cross-table validation without collars

    # Detect hole_id column in collars
    collar_hole_col = _find_column(collars, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    if not collar_hole_col:
        v.append(ValidationViolation(
            "collars", "SCHEMA_ERROR", "ERROR",
            "", -1,
            "Collars missing 'hole_id' column - cannot perform cross-table validation."
        ))
        return v
    
    collar_ids = set(collars[collar_hole_col].astype(str))

    # Assays/Lith with no collar
    for tname, df in [("assays", assays), ("lithology", lith)]:
        if df is None or df.empty:
            continue
        
        hole_col = _find_column(df, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
        if not hole_col:
            continue  # Skip if no hole_id column
            
        for idx, row in df[~df[hole_col].astype(str).isin(collar_ids)].iterrows():
            v.append(ValidationViolation(
                tname, f"{tname.upper()}_NO_COLLAR", "ERROR",
                str(row[hole_col]), idx,
                "Record exists for missing collar."
            ))

    # TD consistency
    td_col = _find_column(collars, ["total_depth", "depth", "max_depth", "length", "TOTAL_DEPTH"])
    if not td_col:
        return v  # Can't do TD validation without total_depth

    try:
        td_map = collars.set_index(collar_hole_col)[td_col].to_dict()
    except Exception:
        return v

    def last_to(df):
        if df is None or df.empty:
            return {}
        hole_col = _find_column(df, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
        to_col = _find_column(df, ["to_depth", "depth_to", "to", "TO", "DEPTH_TO"])
        if not hole_col or not to_col:
            return {}
        try:
            g = df.groupby(hole_col)[to_col].max()
            return g.to_dict()
        except Exception:
            return {}

    a_last = last_to(assays)
    l_last = last_to(lith)

    for hid, td in td_map.items():
        hid_str = str(hid)
        
        # Check assays
        a_max = a_last.get(hid) or a_last.get(hid_str)
        if a_max is not None and td is not None:
            try:
                if float(td) - float(a_max) > 1.0:
                    v.append(ValidationViolation(
                        "assays", "ASSAYS_NOT_TO_TD", "WARNING",
                        hid_str, -1,
                        f"Assays end {float(td) - float(a_max):.2f} m above TD."
                    ))
            except (ValueError, TypeError):
                pass

        # Check lithology
        l_max = l_last.get(hid) or l_last.get(hid_str)
        if l_max is not None and td is not None:
            try:
                if float(td) - float(l_max) > 1.0:
                    v.append(ValidationViolation(
                        "lithology", "LITH_NOT_TO_TD", "WARNING",
                        hid_str, -1,
                        f"Lithology ends {float(td) - float(l_max):.2f} m above TD."
                    ))
            except (ValueError, TypeError):
                pass

    return v


# =========================================================
# BUSINESS RULES
# =========================================================

def validate_business_rules(collars, cfg) -> List[ValidationViolation]:
    """
    Validate business rules. Returns violations list, never raises exceptions.
    """
    v = []

    if collars is None or collars.empty:
        return v

    # Detect column names
    hole_col = _find_column(collars, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    elev_col = _find_column(collars, ["elevation", "z", "elev", "rl", "ELEVATION", "Z", "RL"])
    td_col = _find_column(collars, ["total_depth", "depth", "max_depth", "length", "TOTAL_DEPTH"])
    
    if not hole_col:
        return v

    if cfg.hole_id_prefixes:
        allowed = tuple(cfg.hole_id_prefixes)
        for idx, row in collars.iterrows():
            hid = str(row[hole_col])
            if not hid.startswith(allowed):
                v.append(ValidationViolation(
                    "collars", "BUS_BAD_HOLEID", "INFO",
                    hid, idx,
                    "HoleID prefix not in allowed list."
                ))

    for idx, row in collars.iterrows():
        hid = str(row.get(hole_col, "Unknown"))
        
        if elev_col and pd.notna(row.get(elev_col)) and row[elev_col] < -1000:
            v.append(ValidationViolation(
                "collars", "BUS_ELEVATION_LOW", "WARNING",
                hid, idx,
                "Elevation value looks abnormally low."
            ))

        if td_col and pd.notna(row.get(td_col)) and row[td_col] < 0:
            v.append(ValidationViolation(
                "collars", "BUS_NEGATIVE_TD", "ERROR",
                hid, idx,
                "Total depth is negative."
            ))

    return v


# =========================================================
# CONFIG HASHING (for reproducibility)
# =========================================================

def _hash_validation_config(cfg: ValidationConfig) -> str:
    """
    Compute SHA256 hash of ValidationConfig for reproducibility tracking.
    
    This allows downstream systems to verify that the same configuration
    was used across validation runs.
    """
    # Convert config to a deterministic string representation
    config_dict = {
        "max_interval_gap": cfg.max_interval_gap,
        "max_small_overlap": cfg.max_small_overlap,
        "standard_sample_length": cfg.standard_sample_length,
        "allow_nonstandard_samples": cfg.allow_nonstandard_samples,
        "max_dip_change_deg": cfg.max_dip_change_deg,
        "max_az_change_deg": cfg.max_az_change_deg,
        "dip_min": cfg.dip_min,
        "dip_max": cfg.dip_max,
        "az_min": cfg.az_min,
        "az_max": cfg.az_max,
        "survey_start_tolerance": cfg.survey_start_tolerance,
        "survey_td_tolerance": cfg.survey_td_tolerance,
        "max_survey_spacing": cfg.max_survey_spacing,
        "primary_grade_col": cfg.primary_grade_col,
        "blank_max_value": cfg.blank_max_value,
        "duplicate_max_rel_diff": cfg.duplicate_max_rel_diff,
        "min_qaqc_ratio": cfg.min_qaqc_ratio,
        "lithology_code_col": cfg.lithology_code_col,
    }
    
    # Sort keys for deterministic ordering
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# =========================================================
# MASTER VALIDATOR
# =========================================================

def run_drillhole_validation(
    collars: pd.DataFrame,
    surveys: pd.DataFrame,
    assays: pd.DataFrame,
    lithology: pd.DataFrame,
    cfg: Optional[ValidationConfig] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> ValidationResult:
    """
    Master validation function. NEVER raises exceptions.
    All errors are returned as ValidationViolation objects.
    
    Args:
        collars: Collar data DataFrame
        surveys: Survey data DataFrame
        assays: Assay data DataFrame
        lithology: Lithology data DataFrame
        cfg: Validation configuration (optional)
        progress_callback: Optional callback for progress reporting (percent, message)
    
    Returns:
        ValidationResult with all violations and metadata
    """
    if cfg is None:
        cfg = ValidationConfig()

    violations: List[ValidationViolation] = []
    tables_validated: List[str] = []
    schema_errors: List[str] = []
    
    def report_progress(percent: int, message: str):
        if progress_callback:
            try:
                progress_callback(percent, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    report_progress(0, "Starting drillhole validation")

    # Validate collars (required for other validations)
    try:
        report_progress(10, "Validating collar data")
        collar_violations = validate_collars(collars, cfg)
        violations.extend(collar_violations)
        tables_validated.append("collars")
    except Exception as e:
        logger.error(f"Unexpected error in collar validation: {e}", exc_info=True)
        violations.append(ValidationViolation(
            "collars", "VALIDATION_ERROR", "ERROR",
            "", -1,
            f"Internal validation error: {str(e)}"
        ))
        schema_errors.append(f"collars: {str(e)}")

    # Validate surveys
    try:
        report_progress(30, "Validating survey data")
        survey_violations = validate_surveys(surveys, collars, cfg)
        violations.extend(survey_violations)
        tables_validated.append("surveys")
    except Exception as e:
        logger.error(f"Unexpected error in survey validation: {e}", exc_info=True)
        violations.append(ValidationViolation(
            "surveys", "VALIDATION_ERROR", "ERROR",
            "", -1,
            f"Internal validation error: {str(e)}"
        ))
        schema_errors.append(f"surveys: {str(e)}")

    # Validate assays
    try:
        report_progress(50, "Validating assay intervals")
        assay_violations = validate_intervals(assays, collars, "assays", cfg)
        violations.extend(assay_violations)
        tables_validated.append("assays")
    except Exception as e:
        logger.error(f"Unexpected error in assay validation: {e}", exc_info=True)
        violations.append(ValidationViolation(
            "assays", "VALIDATION_ERROR", "ERROR",
            "", -1,
            f"Internal validation error: {str(e)}"
        ))
        schema_errors.append(f"assays: {str(e)}")

    # Validate lithology
    try:
        report_progress(70, "Validating lithology intervals")
        # Use configured lithology code column
        lith_code_col = cfg.lithology_code_col if cfg.lithology_code_col else "lith_code"
        lith_violations = validate_intervals(lithology, collars, "lithology", cfg, require_code=lith_code_col)
        violations.extend(lith_violations)
        tables_validated.append("lithology")
    except Exception as e:
        logger.error(f"Unexpected error in lithology validation: {e}", exc_info=True)
        violations.append(ValidationViolation(
            "lithology", "VALIDATION_ERROR", "ERROR",
            "", -1,
            f"Internal validation error: {str(e)}"
        ))
        schema_errors.append(f"lithology: {str(e)}")

    # Validate QAQC
    try:
        report_progress(80, "Validating QAQC samples")
        qaqc_violations = validate_qaqc(assays, cfg)
        violations.extend(qaqc_violations)
    except Exception as e:
        logger.error(f"Unexpected error in QAQC validation: {e}", exc_info=True)
        violations.append(ValidationViolation(
            "assays", "VALIDATION_ERROR", "ERROR",
            "", -1,
            f"Internal QAQC validation error: {str(e)}"
        ))

    # Validate cross-table consistency
    try:
        report_progress(90, "Validating cross-table consistency")
        cross_violations = validate_cross_table(collars, surveys, assays, lithology)
        violations.extend(cross_violations)
    except Exception as e:
        logger.error(f"Unexpected error in cross-table validation: {e}", exc_info=True)
        violations.append(ValidationViolation(
            "cross_table", "VALIDATION_ERROR", "ERROR",
            "", -1,
            f"Internal cross-table validation error: {str(e)}"
        ))

    # Validate business rules
    try:
        business_violations = validate_business_rules(collars, cfg)
        violations.extend(business_violations)
    except Exception as e:
        logger.error(f"Unexpected error in business rule validation: {e}", exc_info=True)
        violations.append(ValidationViolation(
            "collars", "VALIDATION_ERROR", "ERROR",
            "", -1,
            f"Internal business rule validation error: {str(e)}"
        ))

    # Compute timestamp and config hash for reproducibility
    timestamp = datetime.now().isoformat()
    config_hash = _hash_validation_config(cfg)
    
    report_progress(100, f"Validation complete: {len(violations)} issues found")
    
    return ValidationResult(
        violations=violations,
        tables_validated=tables_validated,
        schema_errors=schema_errors,
        timestamp=timestamp,
        config_hash=config_hash,
    )
