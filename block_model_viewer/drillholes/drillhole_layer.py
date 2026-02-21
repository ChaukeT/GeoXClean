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
        for _, row in database.surveys.iterrows():
            hid = str(row['hole_id'])
            surveys.setdefault(hid, []).append(
                {
                    "depth_from": float(row['depth_from']),
                    "depth_to": float(row['depth_to']),
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
            continue

        depth_to_point = {depth: tuple(coord) for depth, coord in zip(coord_depths, station_coords)}
        break_depths = set(depth_to_point.keys())
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
        points = []
        index_map = {}
        for depth, coord in sorted_depth_points:
            index_map[depth] = len(points)
            points.append(coord)

        lines = []
        lith_list = []
        assay_list = []
        for idx in range(len(sorted_depth_points) - 1):
            start_depth = sorted_depth_points[idx][0]
            end_depth = sorted_depth_points[idx + 1][0]
            if abs(end_depth - start_depth) < 1e-6:
                continue
            i0 = index_map[start_depth]
            i1 = index_map[end_depth]
            lines.extend([2, i0, i1])
            mid_depth = 0.5 * (start_depth + end_depth)
            lith_list.append(_get_lith_code(lithology_intervals.get(hid, []), mid_depth))
            assay_list.append(_get_assay_value(assays_by_hole.get(hid, []), mid_depth, assay_field_name))

        if not lines:
            hole_polys[hid] = pv.PolyData()
            hole_segment_lith[hid] = []
            hole_segment_assay[hid] = []
            continue

        poly = pv.PolyData(np.array(points, dtype=float))
        poly.lines = np.array(lines, dtype=np.int64)
        hole_polys[hid] = poly
        hole_segment_lith[hid] = lith_list
        hole_segment_assay[hid] = assay_list

    # Compute assay min/max AFTER loop completes (must be outside loop)
    all_assay_values = [
        value for values in hole_segment_assay.values() for value in values
        if value is not None and not np.isnan(value)
    ]
    assay_min = float(np.min(all_assay_values)) if all_assay_values else 0.0
    assay_max = float(np.max(all_assay_values)) if all_assay_values else 1.0
    if assay_max == assay_min:
        assay_max = assay_min + 1.0

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
        "lith_colors": lith_colors,
        "lith_to_index": lith_to_index,
        "assay_field": assay_field_name,
        "assay_min": assay_min,
        "assay_max": assay_max,
        "hole_ids": hole_ids,
        "collar_coords": collar_coords,
        "_registry": registry,  # CRITICAL: Pass registry for persistent interval IDs
    }


def _get_lith_code(intervals: List[Dict[str, float]], depth: float) -> str:
    for interval in intervals:
        if interval["from"] <= depth < interval["to"]:
            return interval["code"]
    return "Unknown"


def _get_assay_value(intervals: List[Dict[str, Dict[str, float]]], depth: float, field: str) -> float:
    for interval in intervals:
        if interval["from"] <= depth < interval["to"]:
            val = interval["values"].get(field)
            if val is not None:
                return float(val)
    return np.nan


def _choose_assay_field(element_names: set) -> str:
    if not element_names:
        return "Grade"
    for candidate in element_names:
        if candidate.lower() == "grade":
            return candidate
    return sorted(element_names)[0]

