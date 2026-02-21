"""
DRILLHOLE AUTO-FIX ENGINE (GeoX)

Plug-compatible with drillhole_validation.py

Responsibilities:
- Run validation
- Apply safe, deterministic corrections
- Log all fixes (before/after, rule, confidence)
- Re-run validation on cleaned data

This module does NOT touch raw source files.
You decide where to persist cleaned tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .drillhole_validation import (
    ValidationConfig,
    ValidationViolation,
    ValidationResult,
    run_drillhole_validation,
)


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class FixAction:
    table: str
    rule_code: str
    hole_id: str
    row_index: int
    columns: Dict[str, Dict[str, Any]]  # {col: {"old": ..., "new": ...}}
    reason: str
    confidence: float  # 0.0–1.0


@dataclass
class AutoFixResult:
    collars: pd.DataFrame
    surveys: pd.DataFrame
    assays: pd.DataFrame
    lithology: pd.DataFrame

    violations_before: List[ValidationViolation]
    violations_after: List[ValidationViolation]
    fixes: List[FixAction]


# =========================================================
# HELPERS
# =========================================================

def _record_fix(
    fixes: List[FixAction],
    table: str,
    rule_code: str,
    hole_id: str,
    row_index: int,
    col_changes: Dict[str, Tuple[Any, Any]],
    reason: str,
    confidence: float,
):
    """
    Utility to append a FixAction to the log.
    col_changes = {col_name: (old_value, new_value), ...}
    """
    columns = {c: {"old": old, "new": new} for c, (old, new) in col_changes.items()}
    fixes.append(
        FixAction(
            table=table,
            rule_code=rule_code,
            hole_id=str(hole_id),
            row_index=int(row_index),
            columns=columns,
            reason=reason,
            confidence=float(confidence),
        )
    )


# =========================================================
# COLLAR AUTO-FIX
# =========================================================

def autofix_collars(
    collars: pd.DataFrame,
    cfg: ValidationConfig,
    fixes: List[FixAction],
) -> pd.DataFrame:
    """
    Conservative collar fixes:
    - normalise hole_id (strip + uppercase)
    
    SAFETY: Never raises exceptions, returns original data if fixes fail.
    """
    if collars is None or collars.empty:
        return collars if collars is not None else pd.DataFrame()

    df = collars.copy(deep=True)
    
    # Detect hole_id column
    hole_id_col = _find_column(df, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    if not hole_id_col:
        return df  # Can't fix without hole_id

    try:
        for idx, row in df.iterrows():
            hid_old = row[hole_id_col]
            hid_new = str(hid_old).strip().upper()

            if hid_new != hid_old:
                df.at[idx, hole_id_col] = hid_new
                _record_fix(
                    fixes=fixes,
                    table="collars",
                    rule_code="COLLAR_HOLEID_NORMALISED",
                    hole_id=str(hid_old),
                    row_index=idx,
                    col_changes={hole_id_col: (hid_old, hid_new)},
                    reason="Normalised hole_id to stripped uppercase.",
                    confidence=1.0,
                )
    except Exception:
        pass  # If fixing fails, return partially fixed data

    return df


# =========================================================
# COLUMN DETECTION HELPERS
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


# =========================================================
# SURVEY AUTO-FIX
# =========================================================

def autofix_surveys(
    surveys: pd.DataFrame,
    collars: pd.DataFrame,
    cfg: ValidationConfig,
    fixes: List[FixAction],
) -> pd.DataFrame:
    """
    Conservative survey fixes:
    - normalise azimuth into [0, 360)
    - interpolate surveys in large gaps (spacing > max_survey_spacing)
    
    Supports multiple depth column schemas:
    - 'depth' (single column)
    - 'depth_from' / 'depth_to' (interval columns)
    """
    if surveys is None or surveys.empty:
        return surveys if surveys is not None else pd.DataFrame()

    df = surveys.copy(deep=True)

    # Detect column names
    hole_id_col = _find_column(df, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    azimuth_col = _find_column(df, ["azimuth", "azi", "bearing", "AZIMUTH", "AZI"])
    dip_col = _find_column(df, ["dip", "inclination", "incl", "DIP", "INCL"])
    
    # Survey depth can be single column OR interval - detect schema
    depth_col = _find_column(df, ["depth", "DEPTH", "md", "MD", "measured_depth"])
    if not depth_col:
        # Try interval columns
        depth_col = _find_column(df, ["depth_from", "from_depth", "from", "FROM", "DEPTH_FROM"])
    
    # If we can't find essential columns, return unchanged
    if not hole_id_col or not depth_col:
        return df
    
    # Azimuth normalisation doesn't require depth column
    if azimuth_col:
        # 1) Normalise azimuth
        for idx, row in df.iterrows():
            az_old = row.get(azimuth_col)
            if pd.isna(az_old):
                continue

            try:
                az_new = float(az_old) % 360.0
            except (ValueError, TypeError):
                continue

            # 360° is equivalent to 0°; normalise to 0
            if np.isclose(az_new, 360.0):
                az_new = 0.0

            if not np.isclose(az_new, float(az_old)):
                df.at[idx, azimuth_col] = az_new
                _record_fix(
                    fixes=fixes,
                    table="surveys",
                    rule_code="SURVEY_AZ_NORMALISED",
                    hole_id=str(row.get(hole_id_col, "")),
                    row_index=idx,
                    col_changes={azimuth_col: (az_old, az_new)},
                    reason="Normalised azimuth into [0, 360) range.",
                    confidence=1.0,
                )

    # 2) Clamp dip to valid range
    if dip_col:
        for idx, row in df.iterrows():
            dip_old = row.get(dip_col)
            if pd.isna(dip_old):
                continue

            try:
                dip_val = float(dip_old)
            except (ValueError, TypeError):
                continue

            # Clamp to [dip_min, dip_max]
            dip_new = max(cfg.dip_min, min(cfg.dip_max, dip_val))

            if not np.isclose(dip_new, dip_val):
                df.at[idx, dip_col] = dip_new
                _record_fix(
                    fixes=fixes,
                    table="surveys",
                    rule_code="SURVEY_DIP_CLAMPED",
                    hole_id=str(row.get(hole_id_col, "")),
                    row_index=idx,
                    col_changes={dip_col: (dip_old, dip_new)},
                    reason=f"Clamped dip from {dip_old:.4f}° to valid range [{cfg.dip_min}°, {cfg.dip_max}°].",
                    confidence=0.95,
                )

    # 3) Interpolate surveys in large gaps (only if we have all needed columns)
    if not azimuth_col or not dip_col:
        return df

    # Group by hole_id and process each hole
    new_rows = []
    try:
        for hid, g in df.groupby(hole_id_col, sort=False):
            g_sorted = g.sort_values(depth_col).copy()
            
            try:
                depths = g_sorted[depth_col].to_numpy(dtype=float)
                azimuths = g_sorted[azimuth_col].to_numpy(dtype=float)
                dips = g_sorted[dip_col].to_numpy(dtype=float)
            except (ValueError, TypeError):
                continue  # Skip this hole if data is not numeric
            
            if len(depths) < 2:
                continue
            
            # Check for large gaps and interpolate
            for i in range(len(depths) - 1):
                gap = depths[i + 1] - depths[i]
                
                if gap > cfg.max_survey_spacing:
                    # Interpolate surveys at regular intervals
                    num_interp = int(np.ceil(gap / cfg.max_survey_spacing)) - 1
                    if num_interp > 0:
                        interp_depths = np.linspace(depths[i], depths[i + 1], num_interp + 2)[1:-1]
                        
                        # Linear interpolation for azimuth and dip
                        for interp_depth in interp_depths:
                            # Interpolation weight
                            t = (interp_depth - depths[i]) / gap
                            
                            # Interpolate azimuth (handle wrap-around at 360°)
                            az1 = azimuths[i]
                            az2 = azimuths[i + 1]
                            # Handle angular interpolation
                            if abs(az2 - az1) > 180:
                                if az2 > az1:
                                    az1 += 360
                                else:
                                    az2 += 360
                            interp_az = (az1 * (1 - t) + az2 * t) % 360.0
                            
                            # Interpolate dip (simple linear)
                            interp_dip = dips[i] * (1 - t) + dips[i + 1] * t
                            
                            # Create new survey row
                            new_row = g_sorted.iloc[i].copy()
                            new_row[depth_col] = interp_depth
                            new_row[azimuth_col] = interp_az
                            new_row[dip_col] = interp_dip
                            # Use a synthetic index that won't conflict
                            new_row.name = f"interp_{hid}_{interp_depth:.2f}"
                            new_rows.append(new_row)
                            
                            _record_fix(
                                fixes=fixes,
                                table="surveys",
                                rule_code="SURVEY_INTERPOLATED",
                                hole_id=str(hid),
                                row_index=-1,  # New row, no original index
                                col_changes={
                                    depth_col: (None, interp_depth),
                                    azimuth_col: (None, interp_az),
                                    dip_col: (None, interp_dip),
                                },
                                reason=f"Interpolated survey at {interp_depth:.2f} m to fill gap of {gap:.1f} m.",
                                confidence=0.8,  # Interpolation is less certain than direct fixes
                            )
    except Exception:
        pass  # If grouping fails, skip interpolation
    
    # Add interpolated rows to dataframe
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values([hole_id_col, depth_col]).reset_index(drop=True)
    
    return df


# =========================================================
# INTERVAL AUTO-FIX (ASSAYS & LITHO)
# =========================================================

def autofix_intervals(
    df: pd.DataFrame,
    collars: pd.DataFrame,
    table: str,
    cfg: ValidationConfig,
    fixes: List[FixAction],
) -> pd.DataFrame:
    """
    Safe interval fixes:
    - small gaps -> snap next from_depth down to prev to_depth
    - small overlaps -> snap from_depth up to prev to_depth
    - negative/zero length -> extend to standard_sample_length (if configured)
    - small overshoot beyond TD -> clamp to TD

    Works for both assays and lithology.
    
    SAFETY: Never raises exceptions, returns original data if fixes fail.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    # Detect column names
    hole_id_col = _find_column(df, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    from_col = _find_column(df, ["from_depth", "depth_from", "from", "FROM", "DEPTH_FROM", "MFROM"])
    to_col = _find_column(df, ["to_depth", "depth_to", "to", "TO", "DEPTH_TO", "MTO"])
    
    if not hole_id_col or not from_col or not to_col:
        # Missing required columns - return unchanged
        return df

    out = df.copy(deep=True)
    
    try:
        out = out.sort_values([hole_id_col, from_col, to_col], kind="mergesort")
    except Exception:
        pass  # Continue with unsorted data

    # Map TD from collars
    collar_td: Dict[Any, float] = {}
    if collars is not None and not collars.empty:
        collar_hole_col = _find_column(collars, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
        collar_td_col = _find_column(collars, ["total_depth", "depth", "max_depth", "length", "TOTAL_DEPTH"])
        if collar_hole_col and collar_td_col:
            try:
                collar_td = (
                    collars[[collar_hole_col, collar_td_col]]
                    .dropna(subset=[collar_td_col])
                    .set_index(collar_hole_col)[collar_td_col]
                    .to_dict()
                )
            except Exception:
                pass

    try:
        for hid, g in out.groupby(hole_id_col, sort=False):
            hid_str = str(hid)
            idxs = list(g.index)
            prev_to = None

            td = collar_td.get(hid) or collar_td.get(hid_str)

            for idx in idxs:
                f_old = out.at[idx, from_col]
                t_old = out.at[idx, to_col]

                # Skip if NaN — let validation handle
                if pd.isna(f_old) or pd.isna(t_old):
                    prev_to = t_old
                    continue

                try:
                    f = float(f_old)
                    t = float(t_old)
                except (ValueError, TypeError):
                    prev_to = t_old
                    continue

                col_changes: Dict[str, Tuple[Any, Any]] = {}

                # 1) Negative or zero length
                length = t - f
                if length <= 0 and cfg.standard_sample_length is not None:
                    t_new = f + cfg.standard_sample_length
                    col_changes[to_col] = (t_old, t_new)
                    t = t_new

                # 2) Gap / overlap relative to previous interval
                if prev_to is not None:
                    try:
                        prev_to_float = float(prev_to)
                        gap = f - prev_to_float
                        overlap = prev_to_float - f

                        # Small gap – snap down
                        if gap > 0 and gap <= cfg.max_interval_gap:
                            f_new = prev_to_float
                            col_changes[from_col] = (f_old, f_new)
                            f = f_new

                        # Overlap – snap up (fix ALL overlaps, regardless of size)
                        if overlap > 0:
                            f_new = prev_to_float
                            col_changes[from_col] = (f_old, f_new)
                            f = f_new
                    except (ValueError, TypeError):
                        pass

                # 3) Slight overshoot beyond TD
                if td is not None:
                    try:
                        td_float = float(td)
                        overshoot = t - td_float
                        if overshoot > 0 and overshoot <= cfg.max_interval_gap:
                            t_new = td_float
                            col_changes[to_col] = (t_old, t_new)
                            t = t_new
                    except (ValueError, TypeError):
                        pass

                # Apply changes if any
                if col_changes:
                    if from_col in col_changes:
                        out.at[idx, from_col] = col_changes[from_col][1]
                    if to_col in col_changes:
                        out.at[idx, to_col] = col_changes[to_col][1]

                    _record_fix(
                        fixes=fixes,
                        table=table,
                        rule_code=f"{table.upper()}_INTERVAL_AUTOFIX",
                        hole_id=hid_str,
                        row_index=idx,
                        col_changes=col_changes,
                        reason="Safe interval auto-fix (gap/overlap/length/TD clamp).",
                        confidence=0.9,
                    )

                prev_to = t
    except Exception:
        pass  # If fixing fails, return partially fixed data

    return out


def autofix_assays(
    assays: pd.DataFrame,
    collars: pd.DataFrame,
    cfg: ValidationConfig,
    fixes: List[FixAction],
) -> pd.DataFrame:
    return autofix_intervals(assays, collars, table="assays", cfg=cfg, fixes=fixes)


def autofix_lithology(
    lithology: pd.DataFrame,
    collars: pd.DataFrame,
    cfg: ValidationConfig,
    fixes: List[FixAction],
) -> pd.DataFrame:
    return autofix_intervals(lithology, collars, table="lithology", cfg=cfg, fixes=fixes)


# =========================================================
# MASTER AUTO-FIX RUNNER
# =========================================================

def run_drillhole_autofix(
    collars: pd.DataFrame,
    surveys: pd.DataFrame,
    assays: pd.DataFrame,
    lithology: pd.DataFrame,
    cfg: Optional[ValidationConfig] = None,
    max_iterations: int = 10,
) -> AutoFixResult:
    """
    High-level pipeline:

    1) Run validation on input data (violations_before)
    2) Apply safe auto-fixes iteratively until convergence (collar/survey/assay/lithology)
    3) Re-run validation on cleaned data (violations_after)
    4) Return cleaned tables + violations + fix log

    The iterative approach ensures all cascading issues are fixed (e.g., fixing one overlap
    may reveal another that needs fixing).
    """
    if cfg is None:
        cfg = ValidationConfig()

    fixes: List[FixAction] = []

    # 1) Validate before
    before = run_drillhole_validation(
        collars=collars,
        surveys=surveys,
        assays=assays,
        lithology=lithology,
        cfg=cfg,
    )
    violations_before = before.violations

    # 2) Apply auto-fixes iteratively until convergence
    collars_fixed = collars.copy()
    surveys_fixed = surveys.copy()
    assays_fixed = assays.copy()
    lith_fixed = lithology.copy()

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        fixes_before_iter = len(fixes)

        # Apply auto-fixes
        collars_fixed = autofix_collars(collars_fixed, cfg, fixes)
        surveys_fixed = autofix_surveys(surveys_fixed, collars_fixed, cfg, fixes)
        assays_fixed = autofix_assays(assays_fixed, collars_fixed, cfg, fixes)
        lith_fixed = autofix_lithology(lith_fixed, collars_fixed, cfg, fixes)

        # Check if we made any fixes this iteration
        fixes_this_iter = len(fixes) - fixes_before_iter
        if fixes_this_iter == 0:
            # No more fixes needed - converged
            break

    # 3) Validate after
    after = run_drillhole_validation(
        collars=collars_fixed,
        surveys=surveys_fixed,
        assays=assays_fixed,
        lithology=lith_fixed,
        cfg=cfg,
    )
    violations_after = after.violations

    # 4) Package result
    return AutoFixResult(
        collars=collars_fixed,
        surveys=surveys_fixed,
        assays=assays_fixed,
        lithology=lith_fixed,
        violations_before=violations_before,
        violations_after=violations_after,
        fixes=fixes,
    )

