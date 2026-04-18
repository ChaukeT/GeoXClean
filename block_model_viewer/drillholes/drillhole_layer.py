from __future__ import annotations

import copy
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pyvista as pv
from PyQt6.QtGui import QColor

from .datamodel import Collar, DrillholeDatabase, LithologyInterval, AssayInterval, SurveyInterval
from ..utils.desurvey import minimum_curvature_path_from_surveys

logger = logging.getLogger(__name__)


@dataclass
class DrillholeLayerConfig:
    database: DrillholeDatabase
    composite_df: Optional[pd.DataFrame] = None
    radius: float = 1.0
    color_mode: str = "Lithology"  # or "Assay"


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_drillhole_polylines(
    database: DrillholeDatabase,
    composite_df: Optional[pd.DataFrame] = None,
    assay_field_name: Optional[str] = None,
    registry: Any = None,
):
    """
    Build drillhole polylines for visualization.
    
    Args:
        database: DrillholeDatabase with collar, survey, assay, lithology data
        composite_df: Optional composite DataFrame
        assay_field_name: Optional assay field to visualize
        registry: DataRegistry instance for persistent interval IDs (critical for GPU picking)
    
    Returns:
        Dictionary with polyline data and _registry reference for GPU picking
    """
    hole_ids = []
    collar_coords = {}
    hole_depths = {}
    surveys = {}
    lithology_intervals = {}
    assays_by_hole = {}

    # DataFrame iteration (high-performance)
    if not database.collars.empty:
        for _, row in database.collars.iterrows():
            hid = str(row['hole_id'])
            collar_coords[hid] = (float(row['x']), float(row['y']), float(row['z']))
            hole_depths[hid] = float(row['length']) if pd.notna(row.get('length')) else None
            hole_ids.append(hid)

    if not database.surveys.empty:
        survey_df = database.surveys
        # Detect schema: single 'depth' column (point measurements) or
        # 'depth_from'/'depth_to' (interval schema)
        has_depth = 'depth' in survey_df.columns
        has_interval = 'depth_from' in survey_df.columns and 'depth_to' in survey_df.columns
        for _, row in survey_df.iterrows():
            hid = str(row['hole_id'])
            if has_interval:
                d_from = float(row['depth_from'])
                d_to = float(row['depth_to'])
            elif has_depth:
                d_from = float(row['depth'])
                d_to = float(row['depth'])  # Point measurement
            else:
                continue
            surveys.setdefault(hid, []).append(
                {
                    "depth_from": d_from,
                    "depth_to": d_to,
                    "azimuth": float(row['azimuth']),
                    "dip": float(row['dip']),
                }
            )

    if not database.lithology.empty:
        for _, row in database.lithology.iterrows():
            hid = str(row['hole_id'])
            lithology_intervals.setdefault(hid, []).append(
                {
                    "from": float(row['depth_from']),
                    "to": float(row['depth_to']),
                    "code": str(row.get('lith_code', 'Unknown')),
                }
            )

    if not database.assays.empty:
        for _, row in database.assays.iterrows():
            hid = str(row['hole_id'])
            values: Dict[str, float] = {}
            # Extract element columns (everything except hole_id, depth_from, depth_to)
            meta_cols = {'hole_id', 'depth_from', 'depth_to'}
            for key, value in row.items():
                if key not in meta_cols and pd.notna(value):
                    try:
                        values[key] = float(value)
                    except (TypeError, ValueError):
                        continue
            assays_by_hole.setdefault(hid, []).append(
                {"from": float(row['depth_from']), "to": float(row['depth_to']), "values": values}
            )

    for holes in (surveys, lithology_intervals, assays_by_hole):
        for hole_list in holes.values():
            hole_list.sort(key=lambda interval: interval.get("from", 0.0))

    # ── Composite DataFrame integration ──────────────────────────────
    # When composite_df is provided (user selected Composites as source),
    # merge it INTO assays_by_hole so coloured tubes use composite grades
    # instead of raw assays.
    if composite_df is not None and not composite_df.empty:
        # Normalise the hole-ID column name (composites may use HoleID, HOLEID, etc.)
        hole_col = None
        for candidate in composite_df.columns:
            if candidate.lower().replace('_', '') in ('holeid', 'hole_id', 'bhid', 'hole', 'dhid'):
                hole_col = candidate
                break
        if hole_col is None:
            logger.warning(
                "[DRILLHOLE DIAG] composite_df has no recognisable hole-ID column. "
                f"Columns: {list(composite_df.columns)}"
            )
        else:
            logger.info(
                f"[DRILLHOLE DIAG] Using composite_df ({len(composite_df)} rows) "
                f"with hole-ID column '{hole_col}' — overriding raw assays"
            )
            # Detect depth columns
            from_col = next((c for c in composite_df.columns if c.lower() in ('from', 'depth_from', 'from_depth')), None)
            to_col = next((c for c in composite_df.columns if c.lower() in ('to', 'depth_to', 'to_depth')), None)
            if from_col is None or to_col is None:
                logger.warning(
                    f"[DRILLHOLE DIAG] composite_df missing depth columns. "
                    f"Columns: {list(composite_df.columns)}"
                )
            else:
                # Replace assays_by_hole with composite data
                composite_assays: Dict[str, list] = {}
                meta_cols_comp = {hole_col, from_col, to_col}
                for _, row in composite_df.iterrows():
                    hid = str(row[hole_col])
                    values: Dict[str, float] = {}
                    for key, value in row.items():
                        if key not in meta_cols_comp and pd.notna(value):
                            try:
                                values[key] = float(value)
                            except (TypeError, ValueError):
                                continue
                    composite_assays.setdefault(hid, []).append(
                        {"from": float(row[from_col]), "to": float(row[to_col]), "values": values}
                    )
                for hole_list in composite_assays.values():
                    hole_list.sort(key=lambda interval: interval.get("from", 0.0))

                # Log holes that exist in composites but NOT in collars (case mismatch detection)
                composite_hole_ids = set(composite_assays.keys())
                collar_hole_ids = set(hole_ids)
                in_comp_not_collar = composite_hole_ids - collar_hole_ids
                in_collar_not_comp = collar_hole_ids - composite_hole_ids
                if in_comp_not_collar:
                    logger.warning(
                        f"[DRILLHOLE DIAG] {len(in_comp_not_collar)} holes in composites but NOT in collars "
                        f"(possible case mismatch): {sorted(in_comp_not_collar)[:10]}"
                    )
                if in_collar_not_comp:
                    logger.warning(
                        f"[DRILLHOLE DIAG] {len(in_collar_not_comp)} holes in collars but NOT in composites: "
                        f"{sorted(in_collar_not_comp)[:10]}"
                    )

                assays_by_hole = composite_assays
                logger.info(
                    f"[DRILLHOLE DIAG] Replaced assays_by_hole with {len(composite_assays)} "
                    f"composite holes ({sum(len(v) for v in composite_assays.values())} intervals)"
                )

    element_names = {
        element
        for assays in assays_by_hole.values()
        for entry in assays
        for element in entry.get("values", {}).keys()
    }
    # Case-insensitive match attempt
    original_field_name = assay_field_name  # Track original request for warning

    # If no assay field provided, choose a sensible default
    if not assay_field_name:
        assay_field_name = _choose_assay_field(element_names)
        logger.info(f"No assay field specified - auto-selected '{assay_field_name}' from available elements: {sorted(element_names)[:10]}")
    elif assay_field_name not in element_names:
        # Try to find case-insensitive match
        found = False
        for name in element_names:
            if name.lower() == assay_field_name.lower():
                assay_field_name = name
                found = True
                break
        if not found:
            fallback_field = _choose_assay_field(element_names)
            logger.warning(
                f"Property '{original_field_name}' not found in assay data. "
                f"Available assay elements: {sorted(element_names) if element_names else 'none'}. "
                f"Falling back to '{fallback_field}'."
            )
            assay_field_name = fallback_field

    hole_polys = {}
    hole_segment_lith = {}
    hole_segment_assay = {}
    hole_segment_from_depth = {}
    hole_segment_to_depth = {}
    lith_colors = {}
    lith_to_index = {}
    radii = {}

    for hid in hole_ids:
        collar = collar_coords.get(hid)
        if collar is None:
            continue
        collar_x, collar_y, collar_z = collar

        hole_surveys = surveys.get(hid, [])
        
        # Determine total depth
        total_depth = hole_depths.get(hid)
        if total_depth is None:
            candidates: List[float] = []
            candidates.extend(survey["depth_to"] for survey in hole_surveys if survey["depth_to"] is not None)
            candidates.extend(lith["to"] for lith in lithology_intervals.get(hid, []))
            candidates.extend(assay["to"] for assay in assays_by_hole.get(hid, []))
            if candidates:
                total_depth = float(max(candidates))
            else:
                total_depth = 0.0
            hole_depths[hid] = total_depth
        
        # Use shared Minimum Curvature algorithm for consistency
        # Convert surveys to the format expected by minimum_curvature_path_from_surveys
        survey_list = []
        for survey in hole_surveys:
            survey_list.append({
                'depth_from': survey["depth_from"],
                'depth_to': survey["depth_to"],
                'azimuth': survey["azimuth"],
                'dip': survey["dip"]
            })
        
        # Calculate 3D path using shared Minimum Curvature algorithm
        # Pass total_depth to ensure path extends to full depth even if no surveys
        coord_depths, station_coords = minimum_curvature_path_from_surveys(
            collar_x, collar_y, collar_z, survey_list,
            default_azimuth=0.0,
            default_dip=-90.0,
            total_depth=total_depth if total_depth > 0 else None
        )

        if len(coord_depths) < 2:
            hole_polys[hid] = pv.PolyData()
            hole_segment_lith[hid] = []
            hole_segment_assay[hid] = []
            hole_segment_from_depth[hid] = []
            hole_segment_to_depth[hid] = []
            continue

        depth_to_point = {depth: tuple(coord) for depth, coord in zip(coord_depths, station_coords)}

        # Break depths: only use data boundaries (assay/lith), collar, and TD.
        # Survey stations are used for desurvey path shape but NOT as
        # segment break points — including them creates many tiny segments
        # between survey stations that have no assay data and render gray.
        break_depths = set()
        break_depths.update(depth for lith in lithology_intervals.get(hid, []) for depth in (lith["from"], lith["to"]))
        break_depths.update(depth for assay in assays_by_hole.get(hid, []) for depth in (assay["from"], assay["to"]))
        break_depths.add(0.0)
        break_depths.add(float(hole_depths.get(hid, 0.0)))
        sorted_breaks = sorted(d for d in break_depths if d >= 0.0)

        for depth in sorted_breaks:
            if depth in depth_to_point:
                continue
            for seg_idx in range(len(coord_depths) - 1):
                d_start = coord_depths[seg_idx]
                d_end = coord_depths[seg_idx + 1]
                if d_start <= depth <= d_end and d_end != d_start:
                    t = (depth - d_start) / (d_end - d_start)
                    start_pt = station_coords[seg_idx]
                    end_pt = station_coords[seg_idx + 1]
                    interp = start_pt + t * (end_pt - start_pt)
                    depth_to_point[depth] = tuple(interp)
                    break

        sorted_depth_points = sorted(depth_to_point.items())

        # ── FIX: Filter NaN/Inf coordinates and near-duplicate points ──
        # Root cause 1 & 2: NaN/Inf from corrupt desurvey and spline
        # overshoot on near-zero segments produce scattered noise artifacts.
        filtered_depth_points = []
        for depth, coord in sorted_depth_points:
            c = np.asarray(coord, dtype=float)
            # Skip NaN/Inf coordinates (root cause: desurvey corruption)
            if not np.all(np.isfinite(c)):
                logger.debug("Hole %s: skipping NaN/Inf point at depth %.2f", hid, depth)
                continue
            # Skip near-duplicate points < 1mm apart (prevents spline overshoot)
            if filtered_depth_points:
                prev_c = np.asarray(filtered_depth_points[-1][1], dtype=float)
                if np.linalg.norm(c - prev_c) < 0.001:
                    logger.debug("Hole %s: merging near-duplicate at depth %.2f", hid, depth)
                    continue
            filtered_depth_points.append((depth, coord))

        points = []
        index_map = {}
        for depth, coord in filtered_depth_points:
            index_map[depth] = len(points)
            points.append(coord)

        lines = []
        lith_list = []
        assay_list = []
        from_depth_list = []
        to_depth_list = []
        for idx in range(len(filtered_depth_points) - 1):
            start_depth = filtered_depth_points[idx][0]
            end_depth = filtered_depth_points[idx + 1][0]
            if abs(end_depth - start_depth) < 1e-6:
                continue
            i0 = index_map[start_depth]
            i1 = index_map[end_depth]
            lines.extend([2, i0, i1])
            mid_depth = 0.5 * (start_depth + end_depth)
            lith_list.append(_get_lith_code(lithology_intervals.get(hid, []), mid_depth))
            assay_list.append(_get_assay_value(assays_by_hole.get(hid, []), mid_depth, assay_field_name))
            from_depth_list.append(start_depth)
            to_depth_list.append(end_depth)

        if not lines:
            hole_polys[hid] = pv.PolyData()
            hole_segment_lith[hid] = []
            hole_segment_assay[hid] = []
            hole_segment_from_depth[hid] = []
            hole_segment_to_depth[hid] = []
            continue

        poly = pv.PolyData(np.array(points, dtype=float))
        poly.lines = np.array(lines, dtype=np.int64)
        hole_polys[hid] = poly
        hole_segment_lith[hid] = lith_list
        hole_segment_assay[hid] = assay_list
        hole_segment_from_depth[hid] = from_depth_list
        hole_segment_to_depth[hid] = to_depth_list

    # ── Per-hole diagnostic logging ─────────────────────────────────
    # Logs which holes are SKIPPED and WHY (no assay data, missing field, all NaN)
    total_collars = len(hole_ids)
    holes_with_assay_records = 0
    holes_with_valid_field = 0
    holes_with_coloured_segments = 0
    holes_skipped_no_assay = []
    holes_skipped_no_field = []
    holes_skipped_all_nan = []

    for hid in hole_ids:
        assay_intervals = assays_by_hole.get(hid, [])
        seg_assays = hole_segment_assay.get(hid, [])

        if not assay_intervals:
            holes_skipped_no_assay.append(hid)
            continue
        holes_with_assay_records += 1

        # Check if the target assay field exists in ANY interval for this hole
        field_found = any(
            assay_field_name in interval.get("values", {})
            for interval in assay_intervals
        )
        if not field_found:
            available_fields = sorted({
                k for interval in assay_intervals
                for k in interval.get("values", {}).keys()
            })
            holes_skipped_no_field.append((hid, available_fields))
            continue
        holes_with_valid_field += 1

        # Check if any segment produced a non-NaN value
        valid_values = [v for v in seg_assays if v is not None and not np.isnan(v) and v > 0]
        if not valid_values:
            holes_skipped_all_nan.append(hid)
            continue
        holes_with_coloured_segments += 1

    logger.info(
        f"[DRILLHOLE DIAG] Hole count pipeline:\n"
        f"  Total holes from collars:          {total_collars}\n"
        f"  Holes with assay/composite records: {holes_with_assay_records}\n"
        f"  Holes with '{assay_field_name}' field:      {holes_with_valid_field}\n"
        f"  Holes with valid (>0) values:      {holes_with_coloured_segments}\n"
        f"  ── Gap analysis ──\n"
        f"  No assay data at all:              {len(holes_skipped_no_assay)}\n"
        f"  Missing '{assay_field_name}' column:        {len(holes_skipped_no_field)}\n"
        f"  All values NaN or zero:            {len(holes_skipped_all_nan)}"
    )
    if holes_skipped_no_assay:
        logger.warning(
            f"[DRILLHOLE DIAG] Holes with NO assay/composite data "
            f"({len(holes_skipped_no_assay)}): {holes_skipped_no_assay[:20]}"
        )
    if holes_skipped_no_field:
        for hid, avail in holes_skipped_no_field[:10]:
            logger.warning(
                f"[DRILLHOLE DIAG] Hole {hid}: '{assay_field_name}' column not found. "
                f"Available: {avail}"
            )
    if holes_skipped_all_nan:
        logger.warning(
            f"[DRILLHOLE DIAG] Holes with all NaN/zero '{assay_field_name}' values "
            f"({len(holes_skipped_all_nan)}): {holes_skipped_all_nan[:20]}"
        )

    # Compute assay min/max AFTER loop completes (must be outside loop)
    all_assay_values = [
        value for values in hole_segment_assay.values() for value in values
        if value is not None and not np.isnan(value)
    ]
    # DH-03 FIX: Use data-driven lower bound instead of hardcoded 0.0.
    # Geophysical logs (magnetic susceptibility, density deviation, etc.)
    # can have legitimate non-zero or negative minima.  Hardcoding 0.0
    # compresses the useful colour range for such properties.
    assay_min = float(np.min(all_assay_values)) if all_assay_values else 0.0
    # For concentration assays (most common case), floor at 0.0 since
    # negative concentrations are physically impossible.
    if assay_min > 0:
        assay_min = 0.0
    assay_max = float(np.max(all_assay_values)) if all_assay_values else 1.0
    if assay_max == assay_min:
        assay_max = assay_min + 1.0
    # 98th percentile of POSITIVE values for auto-clim
    # Using max instead of p98 compresses 90%+ of grades into the dark end
    # of the colour ramp when extreme outliers are present.
    positive_values = [v for v in all_assay_values if v > 0 and np.isfinite(v)]
    assay_p98 = float(np.percentile(positive_values, 98)) if positive_values else assay_max

    unique_codes = sorted({code for codes in hole_segment_lith.values() for code in codes if code})
    if not unique_codes:
        unique_codes = ["Unknown"]
    
    # Professional geological color palette (industry-standard inspired)
    # These colors are distinguishable and commonly used in mining software
    PROFESSIONAL_COLORS = [
        "#E6194B",  # Red (sandstone, ore)
        "#3CB44B",  # Green (shale, dolomite)
        "#FFE119",  # Yellow (limestone)
        "#4363D8",  # Blue (basalt, mudstone)
        "#F58231",  # Orange (siltstone)
        "#911EB4",  # Purple (granite)
        "#46F0F0",  # Cyan (diorite)
        "#F032E6",  # Magenta (schist)
        "#BCF60C",  # Lime (conglomerate)
        "#FABEBE",  # Pink (clay)
        "#008080",  # Teal (gabbro)
        "#E6BEFF",  # Lavender (gneiss)
        "#9A6324",  # Brown (overburden)
        "#FFFAC8",  # Cream (chalk)
        "#800000",  # Maroon (iron formation)
        "#AAFFC3",  # Mint (serpentinite)
        "#808000",  # Olive (amphibolite)
        "#FFD8B1",  # Apricot (quartzite)
        "#000075",  # Navy (diabase)
        "#808080",  # Gray (unknown/waste)
    ]
    
    color_list = []
    for idx in range(len(unique_codes)):
        if idx < len(PROFESSIONAL_COLORS):
            color_list.append(PROFESSIONAL_COLORS[idx])
        else:
            # Fallback to HSV cycling for additional codes
            color = QColor()
            color.setHsvF(((idx - len(PROFESSIONAL_COLORS)) / max(len(unique_codes) - len(PROFESSIONAL_COLORS), 1)), 0.75, 0.90)
            color_list.append(color.name())
    
    lith_to_index = {code: idx for idx, code in enumerate(unique_codes)}
    lith_colors = {code: color_list[idx % len(color_list)] for code, idx in lith_to_index.items()}

    return {
        "hole_polys": hole_polys,
        "hole_segment_lith": hole_segment_lith,
        "hole_segment_assay": hole_segment_assay,
        "hole_segment_from_depth": hole_segment_from_depth,
        "hole_segment_to_depth": hole_segment_to_depth,
        "lith_colors": lith_colors,
        "lith_to_index": lith_to_index,
        "assay_field": assay_field_name,
        "assay_min": assay_min,
        "assay_max": assay_max,
        "assay_p98": assay_p98,
        "hole_ids": hole_ids,
        "collar_coords": collar_coords,
        "_registry": registry,  # CRITICAL: Pass registry for persistent interval IDs
    }


def _get_lith_code(intervals: List[Dict[str, float]], depth: float) -> str:
    # DH-07 FIX: Use closed interval [from, to] for the last interval so
    # the exact TD boundary doesn't fall through to "Unknown".  The half-open
    # [from, to) convention is correct for interior intervals (prevents double-
    # counting at shared boundaries) but the last interval needs to include
    # its endpoint.
    for i, interval in enumerate(intervals):
        is_last = (i == len(intervals) - 1)
        if is_last:
            if interval["from"] <= depth <= interval["to"]:
                return interval["code"]
        else:
            if interval["from"] <= depth < interval["to"]:
                return interval["code"]
    return "Unknown"


# Sentinel values used in GSQ and other drillhole databases for below-detection
_NULL_SENTINELS = frozenset({-999, -99, -1, -5, -100})


def _get_assay_value(intervals: List[Dict[str, Dict[str, float]]], depth: float, field: str) -> float:
    for interval in intervals:
        if interval["from"] <= depth < interval["to"]:
            val = interval["values"].get(field)
            if val is not None:
                fval = float(val)
                # DH-02 FIX: Only reject known sentinel values, NOT all negatives.
                # Half-detection-limit proxies (e.g. -0.5) and geophysical logs
                # (magnetic susceptibility) can have legitimate negative values.
                # The blanket `fval < 0` was destroying valid grade data.
                if fval in _NULL_SENTINELS:
                    return np.nan
                return fval
    return np.nan


def _choose_assay_field(element_names: set) -> str:
    if not element_names:
        return "Grade"
    for candidate in element_names:
        if candidate.lower() == "grade":
            return candidate
    return sorted(element_names)[0]

