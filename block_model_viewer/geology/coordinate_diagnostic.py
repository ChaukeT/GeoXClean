"""
COORDINATE SYSTEM DIAGNOSTIC TOOL

This tool diagnoses the coordinate mismatch between:
1. BIF surfaces (geological model surfaces)
2. Drillhole intervals (BIF lithology segments)

The issue manifests as surfaces and drillholes appearing in completely different
Z frames, despite both using the same source data.

Usage:
    from block_model_viewer.geology.coordinate_diagnostic import DiagnosticReport
    report = DiagnosticReport()
    report.compare_drillhole_to_surface(drillhole_df, surface_mesh)
    report.print_summary()
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CoordinateMismatchInfo:
    """Details of a coordinate system mismatch."""
    drillhole_z_range: Tuple[float, float]
    surface_z_range: Tuple[float, float]
    vertical_offset: float
    is_sign_inverted: bool
    is_datum_mismatch: bool
    suggested_cause: str


@dataclass
class DiagnosticResult:
    """Complete diagnostic result for coordinate systems."""
    drillhole_bounds: Dict[str, Tuple[float, float]]
    surface_bounds: Dict[str, Tuple[float, float]]
    global_shift_applied: Optional[np.ndarray]
    scaler_transform_applied: bool
    mismatches: List[CoordinateMismatchInfo] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def diagnose_coordinate_systems(
    drillhole_df: pd.DataFrame,
    surface_vertices: np.ndarray,
    scaler: Optional[Any] = None,
    global_shift: Optional[np.ndarray] = None,
) -> DiagnosticResult:
    """
    Diagnose coordinate system mismatches between drillholes and surfaces.
    
    This function performs the "30-second proof" described in the user's analysis:
    - Pick a drillhole with a clear BIF intercept
    - Compare its midpoint (X, Y, Z) to the nearest surface vertex
    - If separation is tens to hundreds of meters everywhere, the model never
      intersected the data in the same space.
    
    Args:
        drillhole_df: DataFrame with drillhole data (must have X, Y, Z columns)
        surface_vertices: Numpy array of surface vertices (N x 3)
        scaler: Optional sklearn scaler used for coordinate transformation
        global_shift: Optional global shift applied to renderer coordinates
        
    Returns:
        DiagnosticResult with complete analysis
    """
    result = DiagnosticResult(
        drillhole_bounds={},
        surface_bounds={},
        global_shift_applied=global_shift,
        scaler_transform_applied=scaler is not None,
    )
    
    # 1. Extract coordinate bounds from drillholes
    if drillhole_df is not None and not drillhole_df.empty:
        for axis in ['X', 'Y', 'Z']:
            if axis in drillhole_df.columns:
                result.drillhole_bounds[axis] = (
                    float(drillhole_df[axis].min()),
                    float(drillhole_df[axis].max())
                )
    
    # 2. Extract coordinate bounds from surface
    if surface_vertices is not None and len(surface_vertices) > 0:
        result.surface_bounds['X'] = (float(np.min(surface_vertices[:, 0])), float(np.max(surface_vertices[:, 0])))
        result.surface_bounds['Y'] = (float(np.min(surface_vertices[:, 1])), float(np.max(surface_vertices[:, 1])))
        result.surface_bounds['Z'] = (float(np.min(surface_vertices[:, 2])), float(np.max(surface_vertices[:, 2])))
    
    # 3. Analyze Z coordinate relationship (the most common mismatch)
    if 'Z' in result.drillhole_bounds and 'Z' in result.surface_bounds:
        dh_z_min, dh_z_max = result.drillhole_bounds['Z']
        sf_z_min, sf_z_max = result.surface_bounds['Z']
        
        dh_z_center = (dh_z_min + dh_z_max) / 2
        sf_z_center = (sf_z_min + sf_z_max) / 2
        
        vertical_offset = abs(dh_z_center - sf_z_center)
        dh_z_span = dh_z_max - dh_z_min
        sf_z_span = sf_z_max - sf_z_min
        
        # Detect sign inversion (one positive, one negative Z range)
        is_sign_inverted = (dh_z_min * sf_z_min < 0) or (dh_z_max * sf_z_max < 0)
        
        # Detect datum mismatch (large systematic offset)
        is_datum_mismatch = vertical_offset > max(dh_z_span, sf_z_span) * 0.5
        
        # Determine likely cause
        if is_sign_inverted:
            cause = "SIGN INVERSION: Surface Z uses positive-up, drillholes use positive-down (or vice versa)"
        elif is_datum_mismatch:
            if vertical_offset > 100:
                cause = "DATUM MISMATCH: Surface and drillholes reference different Z datums (e.g., RL vs depth-from-collar)"
            else:
                cause = "MINOR OFFSET: Small vertical offset detected, likely a padding or boundary issue"
        elif sf_z_span < 10 and dh_z_span > 100:
            cause = "SCALE FAILURE: Surface Z range is tiny (likely still in normalized [0,1] space)"
        elif sf_z_span > 1000000:
            cause = "INVERSE TRANSFORM FAILED: Surface Z range is astronomical (double-transform or missing inverse)"
        else:
            cause = "UNKNOWN: Coordinate systems appear compatible but may have subtle issues"
        
        mismatch = CoordinateMismatchInfo(
            drillhole_z_range=(dh_z_min, dh_z_max),
            surface_z_range=(sf_z_min, sf_z_max),
            vertical_offset=vertical_offset,
            is_sign_inverted=is_sign_inverted,
            is_datum_mismatch=is_datum_mismatch,
            suggested_cause=cause,
        )
        result.mismatches.append(mismatch)
        
        # Generate recommendations
        if is_sign_inverted:
            result.recommendations.append(
                "FIX: Check desurvey.py line 158 - Z is computed as collar_z + cumsum(dz). "
                "Dip convention may differ between survey data and model expectation."
            )
        
        if is_datum_mismatch:
            result.recommendations.append(
                "FIX: Check if surface was built in normalized space [0,1] and not transformed "
                "back to world coordinates. See industry_modeler.py _model_to_world()."
            )
        
        if sf_z_span < 10:
            result.recommendations.append(
                "CRITICAL: Surface appears to be in normalized space (Z range < 10m). "
                "The inverse_transform in industry_modeler.py may not have been applied. "
                "Check extract_unified_geology_mesh() method."
            )
    
    return result


def find_nearest_surface_point(
    query_point: np.ndarray,
    surface_vertices: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Find the nearest surface vertex to a query point.
    
    Args:
        query_point: (3,) array with X, Y, Z
        surface_vertices: (N, 3) array of surface vertices
        
    Returns:
        Tuple of (nearest_vertex, distance)
    """
    if surface_vertices is None or len(surface_vertices) == 0:
        return None, float('inf')
    
    distances = np.linalg.norm(surface_vertices - query_point, axis=1)
    min_idx = np.argmin(distances)
    return surface_vertices[min_idx], distances[min_idx]


def spot_check_drillhole_to_surface(
    drillhole_df: pd.DataFrame,
    surface_vertices: np.ndarray,
    lithology_filter: Optional[str] = None,
    n_samples: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform spot checks comparing drillhole interval midpoints to nearest surface points.
    
    This is the "30-second proof" that can definitively show if coordinates are misaligned.
    
    Args:
        drillhole_df: DataFrame with drillhole intervals (needs X, Y, Z, and optionally lithology)
        surface_vertices: Surface vertex array (N x 3)
        lithology_filter: Optional lithology code to filter for (e.g., 'BIF')
        n_samples: Number of random samples to check
        
    Returns:
        List of spot check results with drillhole point, surface point, and distance
    """
    results = []
    
    if drillhole_df is None or drillhole_df.empty or surface_vertices is None:
        return results
    
    # Filter by lithology if specified
    df = drillhole_df
    if lithology_filter:
        lith_col = None
        for col in ['lithology', 'lith_code', 'formation', 'rock_type']:
            if col in df.columns:
                lith_col = col
                break
        
        if lith_col:
            mask = df[lith_col].astype(str).str.upper().str.contains(lithology_filter.upper(), na=False)
            df = df[mask]
    
    if df.empty:
        logger.warning(f"No drillhole intervals found for lithology filter '{lithology_filter}'")
        return results
    
    # Sample random rows
    sample_size = min(n_samples, len(df))
    sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    
    for _, row in sample_df.iterrows():
        try:
            query_point = np.array([float(row['X']), float(row['Y']), float(row['Z'])])
            nearest_vertex, distance = find_nearest_surface_point(query_point, surface_vertices)
            
            result = {
                'hole_id': row.get('hole_id', row.get('HOLEID', 'Unknown')),
                'drillhole_x': query_point[0],
                'drillhole_y': query_point[1],
                'drillhole_z': query_point[2],
                'surface_x': nearest_vertex[0] if nearest_vertex is not None else None,
                'surface_y': nearest_vertex[1] if nearest_vertex is not None else None,
                'surface_z': nearest_vertex[2] if nearest_vertex is not None else None,
                'distance_3d': distance,
                'vertical_offset': abs(query_point[2] - nearest_vertex[2]) if nearest_vertex is not None else None,
                'horizontal_offset': np.linalg.norm([query_point[0] - nearest_vertex[0], query_point[1] - nearest_vertex[1]]) if nearest_vertex is not None else None,
            }
            results.append(result)
            
            logger.info(
                f"Spot check: Hole {result['hole_id']} at ({query_point[0]:.1f}, {query_point[1]:.1f}, {query_point[2]:.1f}) "
                f"-> Nearest surface at ({nearest_vertex[0]:.1f}, {nearest_vertex[1]:.1f}, {nearest_vertex[2]:.1f}) "
                f"Distance: {distance:.1f}m (Vertical: {result['vertical_offset']:.1f}m)"
            )
            
        except Exception as e:
            logger.warning(f"Failed spot check for row: {e}")
            continue
    
    return results


def print_diagnostic_report(result: DiagnosticResult, spot_checks: Optional[List[Dict]] = None):
    """
    Print a formatted diagnostic report.
    """
    print("=" * 80)
    print("COORDINATE SYSTEM DIAGNOSTIC REPORT")
    print("=" * 80)
    
    print("\n--- DRILLHOLE BOUNDS (World Coordinates) ---")
    for axis, bounds in result.drillhole_bounds.items():
        print(f"  {axis}: [{bounds[0]:.2f}, {bounds[1]:.2f}]  (span: {bounds[1] - bounds[0]:.2f}m)")
    
    print("\n--- SURFACE BOUNDS (After Rendering Pipeline) ---")
    for axis, bounds in result.surface_bounds.items():
        print(f"  {axis}: [{bounds[0]:.2f}, {bounds[1]:.2f}]  (span: {bounds[1] - bounds[0]:.2f}m)")
    
    print(f"\n--- TRANSFORM STATE ---")
    print(f"  Global Shift Applied: {result.global_shift_applied is not None}")
    if result.global_shift_applied is not None:
        print(f"    Shift: [{result.global_shift_applied[0]:.2f}, {result.global_shift_applied[1]:.2f}, {result.global_shift_applied[2]:.2f}]")
    print(f"  Scaler Transform Applied: {result.scaler_transform_applied}")
    
    if result.mismatches:
        print("\n--- COORDINATE MISMATCHES DETECTED ---")
        for i, mismatch in enumerate(result.mismatches):
            print(f"\n  Mismatch #{i+1}:")
            print(f"    Drillhole Z range: [{mismatch.drillhole_z_range[0]:.2f}, {mismatch.drillhole_z_range[1]:.2f}]")
            print(f"    Surface Z range:   [{mismatch.surface_z_range[0]:.2f}, {mismatch.surface_z_range[1]:.2f}]")
            print(f"    Vertical Offset:   {mismatch.vertical_offset:.2f}m")
            print(f"    Sign Inverted:     {mismatch.is_sign_inverted}")
            print(f"    Datum Mismatch:    {mismatch.is_datum_mismatch}")
            print(f"    CAUSE: {mismatch.suggested_cause}")
    
    if spot_checks:
        print("\n--- SPOT CHECKS (Drillhole → Surface Distance) ---")
        for check in spot_checks:
            status = "✓ OK" if check['distance_3d'] < 50 else "✗ MISMATCH"
            print(f"  {check['hole_id']}: {status}")
            print(f"    Drillhole: ({check['drillhole_x']:.1f}, {check['drillhole_y']:.1f}, {check['drillhole_z']:.1f})")
            if check['surface_x'] is not None:
                print(f"    Surface:   ({check['surface_x']:.1f}, {check['surface_y']:.1f}, {check['surface_z']:.1f})")
                print(f"    Distance:  {check['distance_3d']:.1f}m (Vertical: {check['vertical_offset']:.1f}m)")
    
    if result.recommendations:
        print("\n--- RECOMMENDATIONS ---")
        for i, rec in enumerate(result.recommendations):
            print(f"  {i+1}. {rec}")
    
    print("\n" + "=" * 80)


def analyze_transform_pipeline(
    raw_contacts_df: pd.DataFrame,
    normalized_contacts: Optional[pd.DataFrame] = None,
    surface_world_coords: Optional[np.ndarray] = None,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Analyze the complete transform pipeline from raw data to rendered surface.
    
    Pipeline stages:
    1. Raw drillhole data (World/UTM coordinates)
    2. Normalized coordinates for LoopStructural (StandardScaler -> [~-2, +2])
    3. LoopStructural internal space ([0, 1] or local grid)
    4. Surface extraction (LoopStructural local space)
    5. Inverse transform to world coordinates (scaler.inverse_transform)
    6. Renderer local precision shift (_to_local_precision)
    
    Returns diagnostic info for each stage.
    """
    analysis = {
        'stages': [],
        'errors': [],
        'warnings': [],
    }
    
    # Stage 1: Raw data
    if raw_contacts_df is not None:
        stage1 = {
            'name': 'Raw Drillhole Data',
            'space': 'World (UTM)',
            'x_range': (float(raw_contacts_df['X'].min()), float(raw_contacts_df['X'].max())),
            'y_range': (float(raw_contacts_df['Y'].min()), float(raw_contacts_df['Y'].max())),
            'z_range': (float(raw_contacts_df['Z'].min()), float(raw_contacts_df['Z'].max())),
        }
        analysis['stages'].append(stage1)
        
        # Check for suspicious raw data
        if stage1['x_range'][1] < 1000 and stage1['y_range'][1] < 1000:
            analysis['warnings'].append(
                "Raw data has small X/Y values (<1000). Is this already normalized?"
            )
    
    # Stage 2: Normalized
    if normalized_contacts is not None and 'X_n' in normalized_contacts.columns:
        stage2 = {
            'name': 'Normalized (StandardScaler)',
            'space': 'Standardized [~-2, +2]',
            'x_range': (float(normalized_contacts['X_n'].min()), float(normalized_contacts['X_n'].max())),
            'y_range': (float(normalized_contacts['Y_n'].min()), float(normalized_contacts['Y_n'].max())),
            'z_range': (float(normalized_contacts['Z_n'].min()), float(normalized_contacts['Z_n'].max())),
        }
        analysis['stages'].append(stage2)
    
    # Stage 5: Surface after inverse transform
    if surface_world_coords is not None:
        stage5 = {
            'name': 'Surface After Inverse Transform',
            'space': 'World (UTM) expected',
            'x_range': (float(np.min(surface_world_coords[:, 0])), float(np.max(surface_world_coords[:, 0]))),
            'y_range': (float(np.min(surface_world_coords[:, 1])), float(np.max(surface_world_coords[:, 1]))),
            'z_range': (float(np.min(surface_world_coords[:, 2])), float(np.max(surface_world_coords[:, 2]))),
        }
        analysis['stages'].append(stage5)
        
        # Check if inverse transform was applied correctly
        if stage5['x_range'][1] < 10:
            analysis['errors'].append(
                "CRITICAL: Surface X is still in normalized space (< 10). "
                "scaler.inverse_transform() was NOT applied!"
            )
        
        if len(analysis['stages']) > 0:
            stage1 = analysis['stages'][0]
            # Compare to raw data
            x_diff = abs(stage5['x_range'][0] - stage1['x_range'][0])
            y_diff = abs(stage5['y_range'][0] - stage1['y_range'][0])
            z_diff = abs(stage5['z_range'][0] - stage1['z_range'][0])
            
            if x_diff > 10000 or y_diff > 10000:
                analysis['warnings'].append(
                    f"Large X/Y offset between raw data and surface: X={x_diff:.0f}m, Y={y_diff:.0f}m. "
                    "This could indicate different coordinate reference systems."
                )
            
            if z_diff > 100:
                analysis['warnings'].append(
                    f"Large Z offset between raw data and surface: {z_diff:.0f}m. "
                    "Check if surface was built in depth space vs RL space."
                )
    
    return analysis


# Quick command-line interface
if __name__ == "__main__":
    print("Run this module from within GeoX to diagnose coordinate issues.")
    print("Usage: from block_model_viewer.geology.coordinate_diagnostic import *")
