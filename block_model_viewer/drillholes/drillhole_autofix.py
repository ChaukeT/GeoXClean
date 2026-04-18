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

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .drillhole_validation import (
    ValidationConfig,
    ValidationViolation,
    ValidationResult,
    run_drillhole_validation,
    _find_column,
)

# Detection limit pattern: "<0.01", "<=0.5", ">10", "≤0.01", "BDL", "ND", "TR"
_DL_RE = re.compile(r'^[<≤]\s*([\d.]+)', re.UNICODE)
_DL_KEYWORDS = {"BDL", "ND", "TRACE", "TR"}


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


def _clamp_confidence(deviation: float) -> float:
    """Confidence score for a range-clamping fix based on deviation size.

    - ≤5 units  → 0.95  (rounding / instrument drift)
    - ≤15 units → 0.85  (data-entry typo)
    - >15 units → 0.70  (systemic error, less certain)
    """
    abs_dev = abs(deviation)
    if abs_dev <= 5.0:
        return 0.95
    elif abs_dev <= 15.0:
        return 0.85
    return 0.70


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
    - clamp negative total_depth to 0

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

    # 2) Clamp negative total_depth to 0
    td_col = _find_column(df, ["total_depth", "max_depth", "length", "TOTAL_DEPTH", "MAX_DEPTH", "EOH"])
    if td_col:
        try:
            for idx, row in df.iterrows():
                td_old = row.get(td_col)
                if pd.isna(td_old):
                    continue
                try:
                    td_val = float(td_old)
                except (ValueError, TypeError):
                    continue

                if td_val < 0:
                    df.at[idx, td_col] = 0.0
                    _record_fix(
                        fixes=fixes,
                        table="collars",
                        rule_code="COLLAR_TD_CLAMPED",
                        hole_id=str(row.get(hole_id_col, "")),
                        row_index=idx,
                        col_changes={td_col: (td_old, 0.0)},
                        reason=f"Clamped negative total_depth {td_val:.2f} to 0.",
                        confidence=_clamp_confidence(td_val),
                    )
        except Exception:
            pass

    return df


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
    - clamp dip to [cfg.dip_min, cfg.dip_max]
    - clamp azimuth to [cfg.az_min, cfg.az_max] after normalisation
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

    # 1b) Clamp dip to valid range [cfg.dip_min, cfg.dip_max]
    if dip_col:
        for idx, row in df.iterrows():
            dip_old = row.get(dip_col)
            if pd.isna(dip_old):
                continue
            try:
                dip_val = float(dip_old)
            except (ValueError, TypeError):
                continue

            if dip_val < cfg.dip_min:
                dip_new = cfg.dip_min
            elif dip_val > cfg.dip_max:
                dip_new = cfg.dip_max
            else:
                continue  # In range — nothing to do

            deviation = dip_val - dip_new
            df.at[idx, dip_col] = dip_new
            _record_fix(
                fixes=fixes,
                table="surveys",
                rule_code="SURVEY_DIP_CLAMPED",
                hole_id=str(row.get(hole_id_col, "")),
                row_index=idx,
                col_changes={dip_col: (dip_old, dip_new)},
                reason=f"Clamped dip {dip_val:.2f} to [{cfg.dip_min}, {cfg.dip_max}] (deviation: {deviation:+.2f}).",
                confidence=_clamp_confidence(deviation),
            )

    # 1c) Clamp azimuth to [cfg.az_min, cfg.az_max] after normalisation
    if azimuth_col:
        for idx, row in df.iterrows():
            az_raw = row.get(azimuth_col)
            if pd.isna(az_raw):
                continue
            try:
                az_val = float(az_raw)
            except (ValueError, TypeError):
                continue

            if az_val < cfg.az_min:
                az_new = cfg.az_min
            elif az_val > cfg.az_max:
                az_new = cfg.az_max
            else:
                continue

            deviation = az_val - az_new
            df.at[idx, azimuth_col] = az_new
            _record_fix(
                fixes=fixes,
                table="surveys",
                rule_code="SURVEY_AZ_CLAMPED",
                hole_id=str(row.get(hole_id_col, "")),
                row_index=idx,
                col_changes={azimuth_col: (az_raw, az_new)},
                reason=f"Clamped azimuth {az_val:.2f} to [{cfg.az_min}, {cfg.az_max}] (deviation: {deviation:+.2f}).",
                confidence=_clamp_confidence(deviation),
            )

    # 2) Insert missing survey at depth 0 (collar orientation)
    if dip_col and azimuth_col:
        collar_rows: List[pd.Series] = []
        try:
            for hid, g in df.groupby(hole_id_col, sort=False):
                g_sorted = g.sort_values(depth_col)
                first_depth = g_sorted[depth_col].iloc[0]
                try:
                    first_depth_f = float(first_depth)
                except (ValueError, TypeError):
                    continue
                if first_depth_f <= cfg.survey_start_tolerance:
                    continue  # Already has a near-surface survey

                # Create depth-0 survey: assume vertical (dip=-90), copy azimuth from first survey
                first_row = g_sorted.iloc[0].copy()
                first_az = first_row.get(azimuth_col, 0.0)
                try:
                    first_az = float(first_az)
                except (ValueError, TypeError):
                    first_az = 0.0

                new_row = first_row.copy()
                new_row[depth_col] = 0.0
                new_row[dip_col] = -90.0
                new_row[azimuth_col] = first_az
                collar_rows.append(new_row)

                _record_fix(
                    fixes=fixes,
                    table="surveys",
                    rule_code="SURVEY_START_INSERTED",
                    hole_id=str(hid),
                    row_index=-1,
                    col_changes={
                        depth_col: (None, 0.0),
                        dip_col: (None, -90.0),
                        azimuth_col: (None, first_az),
                    },
                    reason=f"Inserted collar survey at depth 0 (vertical, az={first_az:.1f}°). "
                           f"First survey was at {first_depth_f:.1f} m.",
                    confidence=0.8,
                )
        except Exception:
            pass

        if collar_rows:
            collar_df = pd.DataFrame(collar_rows)
            df = pd.concat([df, collar_df], ignore_index=True)
            df = df.sort_values([hole_id_col, depth_col]).reset_index(drop=True)

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

    # 4) Fix SURVEY_AZ_CURVATURE — detect back-bearing (±180°) azimuth errors
    #    A 176° change over 10 m is almost certainly a back-bearing / front-bearing
    #    transcription error, not a real dogleg.  If the angular diff between
    #    consecutive azimuths is > az_reversal_deg (default 90°), check whether
    #    flipping the suspect reading by ±180° produces a smooth trajectory.
    if azimuth_col and dip_col:
        try:
            for hid, g in df.groupby(hole_id_col, sort=False):
                g_sorted = g.sort_values(depth_col)
                idxs = list(g_sorted.index)
                if len(idxs) < 2:
                    continue

                azs = g_sorted[azimuth_col].to_numpy(dtype=float)
                deps = g_sorted[depth_col].to_numpy(dtype=float)

                for i in range(1, len(idxs)):
                    d_depth = deps[i] - deps[i - 1]
                    if d_depth <= 0:
                        continue

                    diff = abs(azs[i] - azs[i - 1])
                    if diff > 180:
                        diff = 360 - diff

                    rate = diff / d_depth * 10.0  # deg per 10 m
                    if rate <= cfg.max_az_change_deg:
                        continue  # within tolerance

                    # Candidate: flip by 180°
                    az_old = float(azs[i])
                    az_flipped = (az_old + 180.0) % 360.0

                    # Check if flipped value produces a smaller curvature
                    diff_flipped = abs(az_flipped - azs[i - 1])
                    if diff_flipped > 180:
                        diff_flipped = 360 - diff_flipped

                    # Also check against the NEXT survey if available
                    forward_ok = True
                    if i + 1 < len(azs):
                        diff_fwd_old = abs(azs[i] - azs[i + 1])
                        if diff_fwd_old > 180:
                            diff_fwd_old = 360 - diff_fwd_old
                        diff_fwd_flip = abs(az_flipped - azs[i + 1])
                        if diff_fwd_flip > 180:
                            diff_fwd_flip = 360 - diff_fwd_flip
                        # Flipped value should not make the forward curvature worse
                        if diff_fwd_flip > diff_fwd_old + 10:
                            forward_ok = False

                    if diff_flipped < diff * 0.5 and forward_ok:
                        # Apply the fix
                        idx = idxs[i]
                        df.at[idx, azimuth_col] = az_flipped
                        azs[i] = az_flipped  # update local array for next iteration

                        _record_fix(
                            fixes=fixes,
                            table="surveys",
                            rule_code="SURVEY_AZ_BACK_BEARING",
                            hole_id=str(hid),
                            row_index=idx,
                            col_changes={azimuth_col: (az_old, az_flipped)},
                            reason=(
                                f"Back-bearing correction: azimuth {az_old:.1f}° → "
                                f"{az_flipped:.1f}° (curvature {rate:.1f}°/10 m → "
                                f"{diff_flipped / d_depth * 10:.1f}°/10 m)."
                            ),
                            confidence=0.85 if forward_ok and diff_flipped < 5 else 0.70,
                        )
        except Exception:
            pass

    # 4b) Fix remaining SURVEY_DIP_CURVATURE and SURVEY_AZ_CURVATURE
    #     After back-bearing correction (step 4), any remaining high-curvature
    #     stations are likely transcription errors or instrument glitches.
    #     Safe fix: replace the outlier dip/azimuth with a weighted average
    #     of its immediate neighbours (linear interpolation), which smooths
    #     the spike while preserving the overall trajectory.
    if dip_col and azimuth_col:
        try:
            for hid, g in df.groupby(hole_id_col, sort=False):
                g_sorted = g.sort_values(depth_col)
                idxs = list(g_sorted.index)
                if len(idxs) < 3:
                    continue  # need at least 3 stations to smooth

                dips = g_sorted[dip_col].to_numpy(dtype=float)
                azs = g_sorted[azimuth_col].to_numpy(dtype=float)
                deps = g_sorted[depth_col].to_numpy(dtype=float)

                for i in range(1, len(idxs) - 1):
                    d_prev = deps[i] - deps[i - 1]
                    d_next = deps[i + 1] - deps[i]
                    if d_prev <= 0 or d_next <= 0:
                        continue

                    # --- Dip curvature ---
                    dip_rate = abs(dips[i] - dips[i - 1]) / d_prev * 10.0
                    if np.isfinite(dip_rate) and dip_rate > cfg.max_dip_change_deg:
                        # Smooth: weighted average of neighbours by inverse distance
                        w_prev = 1.0 / d_prev
                        w_next = 1.0 / d_next
                        dip_new = (dips[i - 1] * w_prev + dips[i + 1] * w_next) / (w_prev + w_next)
                        dip_new = round(dip_new, 2)
                        new_rate = abs(dip_new - dips[i - 1]) / d_prev * 10.0
                        if new_rate < dip_rate:
                            dip_old = float(dips[i])
                            idx = idxs[i]
                            df.at[idx, dip_col] = dip_new
                            dips[i] = dip_new
                            _record_fix(
                                fixes=fixes,
                                table="surveys",
                                rule_code="SURVEY_DIP_CURVATURE_SMOOTHED",
                                hole_id=str(hid),
                                row_index=idx,
                                col_changes={dip_col: (dip_old, dip_new)},
                                reason=(
                                    f"Smoothed dip: {dip_old:.1f}° → {dip_new:.1f}° "
                                    f"(curvature {dip_rate:.1f}°/10 m → {new_rate:.1f}°/10 m)."
                                ),
                                confidence=0.75,
                            )

                    # --- Azimuth curvature ---
                    az_diff = abs(azs[i] - azs[i - 1])
                    if az_diff > 180:
                        az_diff = 360 - az_diff
                    az_rate = az_diff / d_prev * 10.0
                    if np.isfinite(az_rate) and az_rate > cfg.max_az_change_deg:
                        # Circular weighted average for azimuth
                        w_prev = 1.0 / d_prev
                        w_next = 1.0 / d_next
                        # Convert to unit vectors for proper circular mean
                        sin_avg = (np.sin(np.radians(azs[i - 1])) * w_prev
                                   + np.sin(np.radians(azs[i + 1])) * w_next)
                        cos_avg = (np.cos(np.radians(azs[i - 1])) * w_prev
                                   + np.cos(np.radians(azs[i + 1])) * w_next)
                        az_new = float(np.degrees(np.arctan2(sin_avg, cos_avg))) % 360.0
                        az_new = round(az_new, 2)
                        # Verify improvement
                        new_diff = abs(az_new - azs[i - 1])
                        if new_diff > 180:
                            new_diff = 360 - new_diff
                        new_rate = new_diff / d_prev * 10.0
                        if new_rate < az_rate:
                            az_old = float(azs[i])
                            idx = idxs[i]
                            df.at[idx, azimuth_col] = az_new
                            azs[i] = az_new
                            _record_fix(
                                fixes=fixes,
                                table="surveys",
                                rule_code="SURVEY_AZ_CURVATURE_SMOOTHED",
                                hole_id=str(hid),
                                row_index=idx,
                                col_changes={azimuth_col: (az_old, az_new)},
                                reason=(
                                    f"Smoothed azimuth: {az_old:.1f}° → {az_new:.1f}° "
                                    f"(curvature {az_rate:.1f}°/10 m → {new_rate:.1f}°/10 m)."
                                ),
                                confidence=0.70,
                            )
        except Exception:
            pass

    # 5) Fix SURVEY_NOT_TO_TD — extrapolate last survey to total depth
    #    Standard industry practice: assume the hole continues with the same
    #    dip and azimuth as the last measurement.  This is safe because survey
    #    tools measure at discrete stations; the driller doesn't stop drilling
    #    at the last survey depth.
    if dip_col and azimuth_col:
        collar_hole_col = _find_column(
            collars, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"]
        ) if collars is not None and not collars.empty else None
        collar_td_col = _find_column(
            collars, ["total_depth", "depth", "max_depth", "length", "TOTAL_DEPTH", "MAX_DEPTH", "EOH"]
        ) if collars is not None and not collars.empty else None

        td_rows: List[pd.Series] = []
        if collar_hole_col and collar_td_col:
            try:
                collar_td_map: Dict[str, float] = {}
                for _, crow in collars.iterrows():
                    chid = str(crow.get(collar_hole_col, ""))
                    ctd = crow.get(collar_td_col)
                    if pd.notna(ctd):
                        try:
                            collar_td_map[chid] = float(ctd)
                        except (ValueError, TypeError):
                            pass

                for hid, g in df.groupby(hole_id_col, sort=False):
                    hid_str = str(hid)
                    td_val = collar_td_map.get(hid_str)
                    if td_val is None:
                        continue

                    g_sorted = g.sort_values(depth_col)
                    last_depth = float(g_sorted[depth_col].iloc[-1])
                    shortfall = td_val - last_depth

                    if shortfall <= cfg.survey_td_tolerance:
                        continue  # within tolerance — no fix needed

                    # Extrapolate: copy last survey's dip/azimuth at TD depth
                    last_row = g_sorted.iloc[-1].copy()
                    last_az = float(last_row.get(azimuth_col, 0.0))
                    last_dip = float(last_row.get(dip_col, -90.0))

                    new_row = last_row.copy()
                    new_row[depth_col] = td_val
                    new_row[dip_col] = last_dip
                    new_row[azimuth_col] = last_az
                    td_rows.append(new_row)

                    _record_fix(
                        fixes=fixes,
                        table="surveys",
                        rule_code="SURVEY_EXTRAPOLATED_TO_TD",
                        hole_id=hid_str,
                        row_index=-1,
                        col_changes={
                            depth_col: (last_depth, td_val),
                            dip_col: (None, last_dip),
                            azimuth_col: (None, last_az),
                        },
                        reason=(
                            f"Extrapolated survey to TD: inserted measurement at "
                            f"{td_val:.1f} m (dip={last_dip:.1f}°, az={last_az:.1f}°). "
                            f"Last survey was at {last_depth:.1f} m, {shortfall:.1f} m above TD."
                        ),
                        confidence=0.85,
                    )
            except Exception:
                pass

        if td_rows:
            td_df = pd.DataFrame(td_rows)
            df = pd.concat([df, td_df], ignore_index=True)
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
    - negative from_depth / to_depth -> clamp to 0
    - small gaps -> snap next from_depth down to prev to_depth
    - small overlaps (within max_small_overlap) -> snap from_depth up to prev to_depth
    - small overshoot beyond TD -> clamp to TD
    - large gaps -> insert no-sample filler interval (NaN grades)
    - assays not to TD -> extend with no-sample interval to collar TD

    NOT auto-fixed (left for manual review):
    - large overlaps (may indicate duplicate or misassigned data)

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

    rows_to_drop: List[int] = []

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

                # 0) Clamp negative depths to 0
                if f < 0:
                    f_clamped = 0.0
                    _record_fix(
                        fixes=fixes,
                        table=table,
                        rule_code=f"{table.upper()}_DEPTH_CLAMPED",
                        hole_id=hid_str,
                        row_index=idx,
                        col_changes={from_col: (f_old, f_clamped)},
                        reason=f"Clamped negative from_depth {f:.2f} to 0.",
                        confidence=_clamp_confidence(f),
                    )
                    out.at[idx, from_col] = f_clamped
                    f_old = f_clamped
                    f = f_clamped

                if t < 0:
                    t_clamped = 0.0
                    _record_fix(
                        fixes=fixes,
                        table=table,
                        rule_code=f"{table.upper()}_DEPTH_CLAMPED",
                        hole_id=hid_str,
                        row_index=idx,
                        col_changes={to_col: (t_old, t_clamped)},
                        reason=f"Clamped negative to_depth {t:.2f} to 0.",
                        confidence=_clamp_confidence(t),
                    )
                    out.at[idx, to_col] = t_clamped
                    t_old = t_clamped
                    t = t_clamped

                col_changes: Dict[str, Tuple[Any, Any]] = {}

                # 1) Negative length — swap from/to (obvious data-entry error)
                #    Zero length is left for manual review (ambiguous).
                length = t - f
                if length < 0:
                    # from > to → swap them
                    f_new, t_new = t, f
                    _record_fix(
                        fixes=fixes,
                        table=table,
                        rule_code=f"{table.upper()}_FROM_TO_SWAPPED",
                        hole_id=hid_str,
                        row_index=idx,
                        col_changes={
                            from_col: (f_old, f_new),
                            to_col: (t_old, t_new),
                        },
                        reason=f"Swapped from_depth ({f:.2f}) and to_depth ({t:.2f}) — negative length.",
                        confidence=0.90,
                    )
                    out.at[idx, from_col] = f_new
                    out.at[idx, to_col] = t_new
                    f_old, t_old = f_new, t_new
                    f, t = f_new, t_new
                    length = t - f
                if length == 0:
                    # Zero-length interval (FROM == TO) — no sample material.
                    # Mark for removal.
                    _record_fix(
                        fixes=fixes,
                        table=table,
                        rule_code=f"{table.upper()}_ZERO_LENGTH_REMOVED",
                        hole_id=hid_str,
                        row_index=idx,
                        col_changes={},
                        reason=f"Removed zero-length interval ({f:.2f}–{t:.2f}).",
                        confidence=0.95,
                    )
                    rows_to_drop.append(idx)
                    prev_to = t
                    continue

                # 2) Gap / overlap relative to previous interval
                if prev_to is not None:
                    try:
                        prev_to_float = float(prev_to)
                        gap = f - prev_to_float
                        overlap = prev_to_float - f

                        # Small gap – snap down (only if within tolerance)
                        if gap > 0 and gap <= cfg.max_interval_gap:
                            f_new = prev_to_float
                            col_changes[from_col] = (f_old, f_new)
                            f = f_new

                        # Small overlap – snap current from_depth up
                        if overlap > 0 and overlap <= cfg.max_small_overlap:
                            f_new = prev_to_float
                            col_changes[from_col] = (f_old, f_new)
                            f = f_new

                        # Large overlap – truncate PREVIOUS interval's to_depth
                        # to current from_depth. This is standard industry practice:
                        # the later (deeper) sample is authoritative.
                        elif overlap > cfg.max_small_overlap:
                            prev_idx = idxs[idxs.index(idx) - 1] if idxs.index(idx) > 0 else None
                            if prev_idx is not None:
                                prev_to_old = out.at[prev_idx, to_col]
                                out.at[prev_idx, to_col] = f
                                _record_fix(
                                    fixes=fixes,
                                    table=table,
                                    rule_code=f"{table.upper()}_OVERLAP_TRUNCATED",
                                    hole_id=hid_str,
                                    row_index=prev_idx,
                                    col_changes={to_col: (prev_to_old, f)},
                                    reason=f"Truncated overlapping interval to_depth "
                                           f"{float(prev_to_old):.2f} -> {f:.2f} "
                                           f"(overlap {overlap:.2f} m).",
                                    confidence=0.80,
                                )
                    except (ValueError, TypeError):
                        pass

                # 3) Cap to_depth that exceeds TD
                if td is not None:
                    try:
                        td_float = float(td)
                        overshoot = t - td_float
                        if overshoot > 1e-3:
                            t_new = td_float
                            col_changes[to_col] = (t_old, t_new)
                            t = t_new
                            # Also cap from_depth if it's at or beyond TD
                            if f >= td_float:
                                # Entire interval beyond TD — mark for drop
                                col_changes[from_col] = (f_old, td_float)
                                f = td_float
                    except (ValueError, TypeError):
                        pass

                # Apply changes if any
                if col_changes:
                    if from_col in col_changes:
                        out.at[idx, from_col] = col_changes[from_col][1]
                    if to_col in col_changes:
                        out.at[idx, to_col] = col_changes[to_col][1]

                    # Scale confidence by magnitude of change
                    max_change = max(
                        abs(float(v[1]) - float(v[0]))
                        for v in col_changes.values()
                        if v[0] is not None and v[1] is not None
                    )
                    if max_change <= 0.01:
                        confidence = 1.0   # Sub-centimetre: rounding artifact
                    elif max_change <= cfg.max_interval_gap:
                        confidence = 0.95  # Within configured tolerance
                    elif max_change <= cfg.max_small_overlap:
                        confidence = 0.85  # Small overlap fix
                    else:
                        confidence = 0.7   # Larger fix (e.g. TD cap beyond tolerance)

                    _record_fix(
                        fixes=fixes,
                        table=table,
                        rule_code=f"{table.upper()}_INTERVAL_AUTOFIX",
                        hole_id=hid_str,
                        row_index=idx,
                        col_changes=col_changes,
                        reason="Safe interval auto-fix (gap/overlap/TD clamp).",
                        confidence=confidence,
                    )

                prev_to = t
    except Exception:
        pass  # If fixing fails, return partially fixed data

    # Drop zero-length intervals collected during pass 1
    if rows_to_drop:
        out = out.drop(index=rows_to_drop, errors='ignore').reset_index(drop=True)

    # ── Pass 2: Fill large gaps and extend to TD with no-sample rows ──
    # Identify grade/value columns (everything except hole_id, from, to)
    metadata_cols = {hole_id_col, from_col, to_col}
    grade_cols = [c for c in out.columns if c not in metadata_cols]

    filler_rows: List[Dict[str, Any]] = []

    try:
        for hid, g in out.groupby(hole_id_col, sort=False):
            hid_str = str(hid)
            g_sorted = g.sort_values(from_col)

            td = collar_td.get(hid) or collar_td.get(hid_str)

            # Collect valid intervals as (from, to) pairs
            intervals = []
            for idx in g_sorted.index:
                f_val = g_sorted.at[idx, from_col]
                t_val = g_sorted.at[idx, to_col]
                if pd.isna(f_val) or pd.isna(t_val):
                    continue
                try:
                    intervals.append((float(f_val), float(t_val)))
                except (ValueError, TypeError):
                    continue

            if not intervals:
                continue

            # (a) Fill gaps between consecutive intervals
            for i in range(len(intervals) - 1):
                _, prev_end = intervals[i]
                next_start, _ = intervals[i + 1]
                gap = next_start - prev_end

                if gap > cfg.max_interval_gap:
                    row_data = {hole_id_col: hid, from_col: prev_end, to_col: next_start}
                    for gc in grade_cols:
                        row_data[gc] = np.nan
                    filler_rows.append(row_data)

                    _record_fix(
                        fixes=fixes,
                        table=table,
                        rule_code=f"{table.upper()}_GAP_FILLED",
                        hole_id=hid_str,
                        row_index=-1,
                        col_changes={
                            from_col: (None, prev_end),
                            to_col: (None, next_start),
                        },
                        reason=f"Inserted no-sample interval [{prev_end:.2f}, {next_start:.2f}] to fill {gap:.2f} m gap.",
                        confidence=0.9,
                    )

            # (b) Extend to TD if assays end above total depth
            if td is not None:
                try:
                    td_float = float(td)
                    _, last_to = intervals[-1]
                    shortfall = td_float - last_to

                    if shortfall > 1.0:
                        row_data = {hole_id_col: hid, from_col: last_to, to_col: td_float}
                        for gc in grade_cols:
                            row_data[gc] = np.nan
                        filler_rows.append(row_data)

                        _record_fix(
                            fixes=fixes,
                            table=table,
                            rule_code=f"{table.upper()}_EXTENDED_TO_TD",
                            hole_id=hid_str,
                            row_index=-1,
                            col_changes={
                                from_col: (None, last_to),
                                to_col: (None, td_float),
                            },
                            reason=f"Extended {table} to TD: inserted no-sample interval [{last_to:.2f}, {td_float:.2f}] ({shortfall:.2f} m).",
                            confidence=0.85,
                        )
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass  # If gap-fill fails, return data with existing fixes only

    # Append filler rows
    if filler_rows:
        filler_df = pd.DataFrame(filler_rows)
        out = pd.concat([out, filler_df], ignore_index=True)
        try:
            out = out.sort_values([hole_id_col, from_col, to_col], kind="mergesort").reset_index(drop=True)
        except Exception:
            pass

    return out


def autofix_assay_grades(
    assays: pd.DataFrame,
    fixes: List[FixAction],
) -> pd.DataFrame:
    """
    Fix assay grade values:
    - Detection limit markers ("<0.01", "<=0.5", "BDL", "ND", "TR") → half-DL numeric
    - Negative grades → clamp to 0

    SAFETY: Never raises exceptions, returns original data if fixes fail.
    """
    if assays is None or assays.empty:
        return assays if assays is not None else pd.DataFrame()

    out = assays.copy(deep=True)
    hole_id_col = _find_column(out, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    from_col = _find_column(out, ["from_depth", "depth_from", "from", "FROM", "DEPTH_FROM", "MFROM"])
    to_col = _find_column(out, ["to_depth", "depth_to", "to", "TO", "DEPTH_TO", "MTO"])

    # Identify grade columns (everything except structural columns)
    skip = {hole_id_col, from_col, to_col}
    # Also skip QAQC metadata columns
    qaqc_cols = {"qaqc_type", "QAQC_TYPE", "qaqc", "sample_type", "SAMPLE_TYPE",
                 "qaqc_reference_id", "QAQC_REFERENCE_ID", "crm_code", "CRM_CODE",
                 "parent_sample_id", "PARENT_SAMPLE_ID", "sample_id", "SAMPLE_ID"}
    skip.update(qaqc_cols)
    grade_cols = [c for c in out.columns if c not in skip and c is not None]

    try:
        for col in grade_cols:
            # ── 1) Detection limit markers → half-DL numeric ──
            for idx in out.index:
                val = out.at[idx, col]
                if pd.isna(val):
                    continue

                val_str = str(val).strip().upper()
                numeric_replacement = None

                # Check regex pattern: "<0.01", "<=0.5", "≤0.01"
                m = _DL_RE.match(str(val).strip())
                if m:
                    try:
                        dl_value = float(m.group(1))
                        numeric_replacement = dl_value / 2.0
                    except (ValueError, TypeError):
                        pass

                # Check keyword markers: "BDL", "ND", "TR", "TRACE"
                if numeric_replacement is None and val_str in _DL_KEYWORDS:
                    numeric_replacement = 0.0  # No DL value available, use 0

                if numeric_replacement is not None:
                    hid = str(out.at[idx, hole_id_col]) if hole_id_col else "Unknown"
                    out.at[idx, col] = numeric_replacement
                    _record_fix(
                        fixes=fixes,
                        table="assays",
                        rule_code="ASSAY_DL_REPLACED",
                        hole_id=hid,
                        row_index=idx,
                        col_changes={col: (val, numeric_replacement)},
                        reason=f"Detection limit '{val}' in '{col}' replaced with {numeric_replacement:.6g}.",
                        confidence=0.9,
                    )

            # ── 2) Negative grades → clamp to 0 ──
            try:
                numeric_vals = pd.to_numeric(out[col], errors="coerce")
                neg_mask = numeric_vals < 0
                for idx in out.index[neg_mask]:
                    old_val = out.at[idx, col]
                    hid = str(out.at[idx, hole_id_col]) if hole_id_col else "Unknown"
                    out.at[idx, col] = 0.0
                    _record_fix(
                        fixes=fixes,
                        table="assays",
                        rule_code="ASSAY_NEG_CLAMPED",
                        hole_id=hid,
                        row_index=idx,
                        col_changes={col: (old_val, 0.0)},
                        reason=f"Negative grade {old_val} in '{col}' clamped to 0.",
                        confidence=0.85,
                    )
            except Exception:
                pass
    except Exception:
        pass  # If grade fixing fails, return partially fixed data

    return out


def autofix_assays(
    assays: pd.DataFrame,
    collars: pd.DataFrame,
    cfg: ValidationConfig,
    fixes: List[FixAction],
) -> pd.DataFrame:
    out = autofix_intervals(assays, collars, table="assays", cfg=cfg, fixes=fixes)
    out = autofix_assay_grades(out, fixes)
    return out


def autofix_lithology(
    lithology: pd.DataFrame,
    collars: pd.DataFrame,
    cfg: ValidationConfig,
    fixes: List[FixAction],
) -> pd.DataFrame:
    return autofix_intervals(lithology, collars, table="lithology", cfg=cfg, fixes=fixes)


# =========================================================
# ORPHAN REMOVAL (NO_COLLAR)
# =========================================================

def _remove_orphans(
    collars: pd.DataFrame,
    surveys: pd.DataFrame,
    assays: pd.DataFrame,
    lithology: pd.DataFrame,
    fixes: List[FixAction],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove records from surveys/assays/lithology whose hole_id has no collar.

    Returns (surveys_clean, assays_clean, lithology_clean).
    """
    if collars is None or collars.empty:
        return surveys, assays, lithology

    collar_hid_col = _find_column(collars, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
    if not collar_hid_col:
        return surveys, assays, lithology

    valid_ids = set(
        collars[collar_hid_col].dropna().astype(str).str.strip().str.upper()
    )

    def _clean(df: pd.DataFrame, table: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df if df is not None else pd.DataFrame()
        hid_col = _find_column(df, ["hole_id", "holeid", "HOLE_ID", "HoleID", "hole"])
        if not hid_col:
            return df
        ids = df[hid_col].fillna("").astype(str).str.strip().str.upper()
        orphan_mask = ~ids.isin(valid_ids) | (ids == "")
        orphan_count = int(orphan_mask.sum())
        if orphan_count == 0:
            return df

        # Record one fix per orphan hole_id (not per row, to keep log compact)
        orphan_ids = set(ids[orphan_mask].unique()) - {""}
        for oid in sorted(orphan_ids):
            n = int((ids == oid).sum())
            _record_fix(
                fixes=fixes,
                table=table,
                rule_code=f"{table.upper()}_NO_COLLAR_REMOVED",
                hole_id=oid,
                row_index=-1,
                col_changes={},
                reason=f"Removed {n} {table} record(s) for hole '{oid}' — no matching collar.",
                confidence=0.95,
            )

        return df[~orphan_mask].reset_index(drop=True)

    return (
        _clean(surveys, "surveys"),
        _clean(assays, "assays"),
        _clean(lithology, "lithology"),
    )


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

        # Remove orphan records (NO_COLLAR) — records whose hole_id
        # has no matching collar.  Run after collar normalisation so
        # the uppercase hole_ids match.
        surveys_fixed, assays_fixed, lith_fixed = _remove_orphans(
            collars_fixed, surveys_fixed, assays_fixed, lith_fixed, fixes,
        )

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

