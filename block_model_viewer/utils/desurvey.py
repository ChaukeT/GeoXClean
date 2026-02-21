"""
Unified Drillhole Desurvey Utility - Minimum Curvature Algorithm
================================================================

This module provides the single source of truth for drillhole desurveying
calculations. All UI and backend code should use these functions to ensure
consistent 3D coordinate calculation across the application.

The Minimum Curvature method is the industry standard for calculating
drillhole trajectories from survey data.

Algorithm Reference:
    Saari, V.K. (1977). "Minimum curvature method for survey calculations"
    SPE Paper 6876.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union, Dict

logger = logging.getLogger(__name__)


def minimum_curvature_desurvey(
    collar_x: float,
    collar_y: float,
    collar_z: float,
    survey_df: pd.DataFrame,
    depth_col: str = None,
    azimuth_col: str = None,
    dip_col: str = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculate 3D drillhole trajectory using the Minimum Curvature method.
    
    This is the industry-standard algorithm for converting survey data
    (depth, azimuth, dip) into 3D coordinates (X, Y, Z).
    
    Args:
        collar_x: Collar X coordinate (Easting)
        collar_y: Collar Y coordinate (Northing)
        collar_z: Collar Z coordinate (Elevation/RL)
        survey_df: DataFrame with survey data (DEPTH, AZIMUTH, DIP)
        depth_col: Optional column name for depth (auto-detected if None)
        azimuth_col: Optional column name for azimuth (auto-detected if None)
        dip_col: Optional column name for dip (auto-detected if None)
        
    Returns:
        Tuple of (depths, xs, ys, zs) arrays, or (None, None, None, None) if invalid
        
    Notes:
        - Azimuth is measured clockwise from North (0-360 degrees)
        - Dip is measured from horizontal (-90 = straight down, 0 = horizontal)
        - Z coordinates decrease downward (positive Z is up)
    """
    try:
        logger.debug(f"minimum_curvature_desurvey called: collar=({collar_x}, {collar_y}, {collar_z}), "
                    f"survey_df={'None' if survey_df is None else f'{len(survey_df)} rows'}")
        
        if survey_df is None or survey_df.empty:
            logger.debug("  Survey DataFrame is None or empty")
            return None, None, None, None
    
        # Auto-detect column names if not provided
        logger.debug(f"  Detecting columns from: {list(survey_df.columns)}")
        depth_col = depth_col or _detect_column(survey_df, ["DEPTH", "MD", "MEASURED_DEPTH"])
        azimuth_col = azimuth_col or _detect_column(survey_df, ["AZI", "AZIMUTH", "AZ", "BEARING"])
        dip_col = dip_col or _detect_column(survey_df, ["DIP", "INCLINATION", "INC"])
        
        logger.debug(f"  Detected columns: depth={depth_col}, azimuth={azimuth_col}, dip={dip_col}")
        
        if not depth_col or not azimuth_col or not dip_col:
            # BUG FIX #11: Log as ERROR (not warning) for missing required columns
            missing = []
            if not depth_col:
                missing.append("DEPTH/MD/MEASURED_DEPTH")
            if not azimuth_col:
                missing.append("AZI/AZIMUTH/AZ/BEARING")
            if not dip_col:
                missing.append("DIP/INCLINATION/INC")
            logger.error(f"DESURVEY FAILED: Missing required columns: {missing}. "
                        f"Available columns: {list(survey_df.columns)}. "
                        f"Hole will use vertical fallback coordinates.")
            return None, None, None, None
    
        # Sort by depth and extract values
        logger.debug(f"  Sorting by {depth_col}")
        surveys = survey_df.sort_values(depth_col)

        # Drop rows with NaN values in required columns
        initial_count = len(surveys)
        surveys = surveys.dropna(subset=[depth_col, azimuth_col, dip_col])
        dropped_count = initial_count - len(surveys)
        if dropped_count > 0:
            logger.debug(f"  Dropped {dropped_count} rows with NaN values")

        if surveys.empty:
            logger.warning("No valid survey data after removing NaN values")
            return None, None, None, None

        # BUG FIX #20: Check for and remove duplicate depths
        depths_check = surveys[depth_col].values
        if len(depths_check) != len(np.unique(depths_check)):
            dup_count = len(depths_check) - len(np.unique(depths_check))
            logger.warning(f"  Found {dup_count} duplicate depth values - removing duplicates (keeping first)")
            surveys = surveys.drop_duplicates(subset=[depth_col], keep='first')
        
        logger.debug(f"  Converting {len(surveys)} survey points to numeric")
        depths = pd.to_numeric(surveys[depth_col], errors='coerce').values
        azimuths_deg = pd.to_numeric(surveys[azimuth_col], errors='coerce').values
        dips_deg = pd.to_numeric(surveys[dip_col], errors='coerce').values
        
        # Check for any remaining NaN values
        if np.any(np.isnan(depths)) or np.any(np.isnan(azimuths_deg)) or np.any(np.isnan(dips_deg)):
            logger.warning("NaN values found in survey data after conversion")
            return None, None, None, None
        
        # Ensure we have at least one valid survey point
        if len(depths) == 0:
            logger.warning("No valid survey depths found")
            return None, None, None, None
        
        logger.debug(f"  Processing {len(depths)} valid survey points")
        
        # Force start at depth 0 if missing
        if depths[0] > 0:
            logger.debug(f"  Adding depth 0 point (first depth was {depths[0]})")
            depths = np.insert(depths, 0, 0.0)
            azimuths_deg = np.insert(azimuths_deg, 0, azimuths_deg[0])
            dips_deg = np.insert(dips_deg, 0, dips_deg[0])
    
        # Convert to radians
        logger.debug("  Converting angles to radians")
        azimuths = np.radians(azimuths_deg)
        dips = np.radians(dips_deg)
        
        # Interval lengths
        dr = np.diff(depths)
        logger.debug(f"  Calculated {len(dr)} intervals, depth range: {depths[0]:.2f} to {depths[-1]:.2f}")
        
        # Angles at start and end of each interval
        i1, i2 = dips[:-1], dips[1:]  # Dip angles
        a1, a2 = azimuths[:-1], azimuths[1:]  # Azimuth angles
        
        # Dogleg Angle (Alpha) using standard formula:
        # cos(alpha) = cos(I2-I1) - sin(I1)*sin(I2)*(1 - cos(A2-A1))
        logger.debug("  Calculating dogleg angles")
        c_i1, s_i1 = np.cos(i1), np.sin(i1)
        c_i2, s_i2 = np.cos(i2), np.sin(i2)
        d_a = a2 - a1
        
        cos_alpha = np.cos(i2 - i1) - (s_i1 * s_i2 * (1 - np.cos(d_a)))
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)  # Numerical stability
        alpha = np.arccos(cos_alpha)
        
        # Ratio Factor: F = tan(alpha/2) / (alpha/2)
        # This approaches 1.0 as alpha approaches 0 (straight hole)
        # BUG FIX #10: Compute limit BEFORE division to avoid NaN/inf
        logger.debug("  Calculating ratio factors")
        F = np.ones_like(alpha)  # Initialize all to 1.0 (the limit value)
        # Only compute for non-straight sections (alpha >= 1e-6)
        non_straight = alpha >= 1e-6
        if np.any(non_straight):
            half_alpha = alpha[non_straight] / 2.0
            F[non_straight] = np.tan(half_alpha) / half_alpha
        
        # Minimum Curvature displacement calculations
        # dx = East, dy = North, dz = Vertical (positive up)
        logger.debug("  Calculating displacements")
        dx = (dr / 2) * (c_i1 * np.sin(a1) + c_i2 * np.sin(a2)) * F
        dy = (dr / 2) * (c_i1 * np.cos(a1) + c_i2 * np.cos(a2)) * F
        dz = (dr / 2) * (s_i1 + s_i2) * F  # Note: dip sign handles direction
        
        # Cumulative sum to get coordinates
        logger.debug("  Computing cumulative coordinates")
        xs = np.concatenate(([collar_x], np.cumsum(dx) + collar_x))
        ys = np.concatenate(([collar_y], np.cumsum(dy) + collar_y))
        zs = np.concatenate(([collar_z], np.cumsum(dz) + collar_z))
        
        logger.debug(f"  Desurvey complete: {len(depths)} points, final position: ({xs[-1]:.2f}, {ys[-1]:.2f}, {zs[-1]:.2f})")
        return depths, xs, ys, zs
    except Exception as e:
        logger.error(f"ERROR in minimum_curvature_desurvey: {e}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, None


def interpolate_at_depth(
    depths: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    target_depth: float,
) -> Tuple[float, float, float]:
    """
    Interpolate 3D coordinates at a specific depth along the drillhole.
    
    Uses linear interpolation between survey points (the points are already
    on the Minimum Curvature curve, so linear interpolation between them
    is appropriate for short intervals).
    
    Args:
        depths: Array of depths from minimum_curvature_desurvey()
        xs: Array of X coordinates from minimum_curvature_desurvey()
        ys: Array of Y coordinates from minimum_curvature_desurvey()
        zs: Array of Z coordinates from minimum_curvature_desurvey()
        target_depth: The depth at which to interpolate coordinates
        
    Returns:
        Tuple of (x, y, z) coordinates at the target depth
    """
    x = float(np.interp(target_depth, depths, xs))
    y = float(np.interp(target_depth, depths, ys))
    z = float(np.interp(target_depth, depths, zs))
    return x, y, z


def interpolate_at_depths(
    depths: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    target_depths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate 3D coordinates at multiple depths (vectorized).

    Args:
        depths: Array of depths from minimum_curvature_desurvey()
        xs: Array of X coordinates from minimum_curvature_desurvey()
        ys: Array of Y coordinates from minimum_curvature_desurvey()
        zs: Array of Z coordinates from minimum_curvature_desurvey()
        target_depths: Array of depths at which to interpolate coordinates

    Returns:
        Tuple of (xs, ys, zs) arrays at the target depths
    """
    # BUG FIX #6: Warn about extrapolation beyond survey coverage
    min_survey_depth = depths.min()
    max_survey_depth = depths.max()
    extrapolated_below = target_depths > max_survey_depth
    extrapolated_above = target_depths < min_survey_depth

    if np.any(extrapolated_below):
        count = np.sum(extrapolated_below)
        max_target = target_depths[extrapolated_below].max()
        logger.warning(f"  {count} sample(s) extend beyond survey depth ({max_survey_depth:.1f}m). "
                      f"Max sample depth: {max_target:.1f}m. Coordinates will be extrapolated.")

    if np.any(extrapolated_above):
        count = np.sum(extrapolated_above)
        min_target = target_depths[extrapolated_above].min()
        logger.warning(f"  {count} sample(s) are above survey start ({min_survey_depth:.1f}m). "
                      f"Min sample depth: {min_target:.1f}m. Coordinates will be extrapolated.")

    interp_xs = np.interp(target_depths, depths, xs)
    interp_ys = np.interp(target_depths, depths, ys)
    interp_zs = np.interp(target_depths, depths, zs)
    return interp_xs, interp_ys, interp_zs


def desurvey_hole_dataframe(
    collar_x: float,
    collar_y: float,
    collar_z: float,
    survey_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    Convenience function that returns survey DataFrame with added X, Y, Z columns.
    
    Args:
        collar_x, collar_y, collar_z: Collar coordinates
        survey_df: Survey DataFrame
        
    Returns:
        Survey DataFrame with X, Y, Z columns added, or None if desurvey failed
    """
    depths, xs, ys, zs = minimum_curvature_desurvey(collar_x, collar_y, collar_z, survey_df)
    
    if depths is None:
        return None
    
    result = survey_df.copy()
    
    # Detect depth column
    depth_col = _detect_column(survey_df, ["DEPTH", "MD", "MEASURED_DEPTH"])
    if not depth_col:
        return None
    
    # Interpolate coordinates for each survey point
    survey_depths = result[depth_col].values
    result['X'], result['Y'], result['Z'] = interpolate_at_depths(
        depths, xs, ys, zs, survey_depths
    )
    
    return result


def add_coordinates_to_intervals(
    intervals_df: pd.DataFrame,
    collar_df: pd.DataFrame,
    survey_df: Optional[pd.DataFrame],
    hole_id_col: str = "HOLEID",
    from_col: str = "FROM",
    to_col: str = "TO",
) -> pd.DataFrame:
    """
    Add X, Y, Z coordinates to interval data (assays, composites, lithology).
    
    Calculates the 3D midpoint coordinates for each interval using
    Minimum Curvature desurveying.
    
    Args:
        intervals_df: DataFrame with interval data (FROM, TO, grade columns)
        collar_df: DataFrame with collar coordinates (HOLEID, X, Y, Z)
        survey_df: DataFrame with survey data (HOLEID, DEPTH, AZI, DIP) or None
        hole_id_col: Column name for hole ID
        from_col: Column name for interval start depth
        to_col: Column name for interval end depth
        
    Returns:
        DataFrame with X, Y, Z columns added (coordinates at interval midpoint)
    """
    result = intervals_df.copy()
    result['X'] = 0.0
    result['Y'] = 0.0
    result['Z'] = 0.0
    
    # Calculate midpoint depths
    result['_MID'] = (result[from_col] + result[to_col]) / 2.0
    
    # Detect collar coordinate column names (case-insensitive)
    collar_x_col = _detect_column(collar_df, ['X', 'EASTING', 'EAST'])
    collar_y_col = _detect_column(collar_df, ['Y', 'NORTHING', 'NORTH'])
    collar_z_col = _detect_column(collar_df, ['Z', 'ELEVATION', 'RL', 'ELEV'])
    collar_id_col = _detect_column(collar_df, [hole_id_col, 'HOLEID', 'HOLE_ID', 'HoleID'])
    
    if not collar_x_col or not collar_y_col or not collar_z_col or not collar_id_col:
        logger.warning(f"Could not detect collar coordinate columns. Found: x={collar_x_col}, y={collar_y_col}, z={collar_z_col}, id={collar_id_col}")
        return result
    
    # Index collars for fast lookup
    collar_idx = collar_df.set_index(collar_id_col)
    
    # Process each hole
    for hid, group in result.groupby(hole_id_col):
        if hid not in collar_idx.index:
            continue
        
        collar = collar_idx.loc[hid]
        x0 = float(collar[collar_x_col])
        y0 = float(collar[collar_y_col])
        z0 = float(collar[collar_z_col])
        
        # Get desurvey path
        survey_id_col = _detect_column(survey_df, [hole_id_col, 'HOLEID', 'HOLE_ID', 'HoleID']) if survey_df is not None else None
        if survey_df is not None and survey_id_col and hid in survey_df[survey_id_col].values:
            hole_survey = survey_df[survey_df[survey_id_col] == hid]
            depths, xs, ys, zs = minimum_curvature_desurvey(x0, y0, z0, hole_survey)
            
            if depths is not None:
                # Interpolate coordinates at midpoints
                mids = group['_MID'].values
                interp_xs, interp_ys, interp_zs = interpolate_at_depths(
                    depths, xs, ys, zs, mids
                )
                result.loc[group.index, 'X'] = interp_xs
                result.loc[group.index, 'Y'] = interp_ys
                result.loc[group.index, 'Z'] = interp_zs
            else:
                # Fallback to vertical hole
                _apply_vertical_coords(result, group, x0, y0, z0, from_col, to_col)
        else:
            # No survey data - assume vertical hole
            _apply_vertical_coords(result, group, x0, y0, z0, from_col, to_col)
    
    # Clean up temporary column
    result.drop('_MID', axis=1, inplace=True)
    
    return result


def _apply_vertical_coords(
    result: pd.DataFrame,
    group: pd.DataFrame,
    x0: float,
    y0: float,
    z0: float,
    from_col: str = 'FROM',
    to_col: str = 'TO',
) -> None:
    """Apply vertical hole assumption for missing survey data."""
    # Use provided column names, falling back to defaults
    if from_col in group.columns and to_col in group.columns:
        mids = (group[from_col] + group[to_col]) / 2.0
    elif 'FROM' in group.columns and 'TO' in group.columns:
        mids = (group['FROM'] + group['TO']) / 2.0
    elif '_MID' in group.columns:
        mids = group['_MID']
    else:
        mids = group.iloc[:, 0]  # Last resort fallback
    result.loc[group.index, 'X'] = x0
    result.loc[group.index, 'Y'] = y0
    result.loc[group.index, 'Z'] = z0 - mids.values  # Z decreases with depth


def _detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Detect column name from a list of candidates (case-insensitive).
    
    Args:
        df: DataFrame to search
        candidates: List of possible column names
        
    Returns:
        Matched column name or None
    """
    upper_cols = {c.upper(): c for c in df.columns}
    for candidate in candidates:
        if candidate.upper() in upper_cols:
            return upper_cols[candidate.upper()]
    return None


def get_drillhole_path(
    collar_x: float,
    collar_y: float,
    collar_z: float,
    survey_df: pd.DataFrame,
    num_points: int = 100,
    max_depth: Optional[float] = None,
) -> np.ndarray:
    """
    Get smooth drillhole path as array of 3D points for visualization.
    
    Args:
        collar_x, collar_y, collar_z: Collar coordinates
        survey_df: Survey DataFrame
        num_points: Number of points along the path
        max_depth: Maximum depth (uses survey max if None)
        
    Returns:
        Array of shape (num_points, 3) with [X, Y, Z] coordinates
    """
    depths, xs, ys, zs = minimum_curvature_desurvey(collar_x, collar_y, collar_z, survey_df)
    
    if depths is None:
        return np.array([[collar_x, collar_y, collar_z]])
    
    if max_depth is None:
        max_depth = depths[-1]
    
    # Generate evenly spaced depths
    target_depths = np.linspace(0, max_depth, num_points)
    
    # Interpolate coordinates
    interp_xs, interp_ys, interp_zs = interpolate_at_depths(
        depths, xs, ys, zs, target_depths
    )
    
    return np.column_stack([interp_xs, interp_ys, interp_zs])


def minimum_curvature_segment(
    prev_x: float,
    prev_y: float,
    prev_z: float,
    prev_depth: float,
    curr_depth: float,
    prev_azimuth: float,
    prev_dip: float,
    curr_azimuth: float,
    curr_dip: float,
) -> Tuple[float, float, float]:
    """
    Calculate the next 3D coordinate using Minimum Curvature algorithm for a single segment.
    
    This function matches the exact algorithm used in drillhole_layer.py to ensure
    mathematical consistency across the application.
    
    Args:
        prev_x, prev_y, prev_z: Previous 3D coordinates
        prev_depth: Previous depth (measured depth)
        curr_depth: Current depth (measured depth)
        prev_azimuth: Azimuth at previous depth (degrees, 0-360)
        prev_dip: Dip at previous depth (degrees, -90 = straight down)
        curr_azimuth: Azimuth at current depth (degrees, 0-360)
        curr_dip: Dip at current depth (degrees, -90 = straight down)
        
    Returns:
        Tuple of (x, y, z) coordinates at curr_depth
        
    Notes:
        - Uses inclination (I = 90° + dip) as per industry standard Minimum Curvature
        - Azimuth is measured clockwise from North (0-360 degrees)
        - Dip is measured from horizontal (-90 = straight down, 0 = horizontal)
        - Z coordinates decrease downward (positive Z is up)
    """
    import math
    
    dmd = curr_depth - prev_depth
    if dmd <= 0:
        return prev_x, prev_y, prev_z
    
    # Convert dip to inclination (I = 90° + dip)
    # This is the standard convention for Minimum Curvature
    I1 = math.radians(90.0 + prev_dip)
    I2 = math.radians(90.0 + curr_dip)
    A1 = math.radians(prev_azimuth)
    A2 = math.radians(curr_azimuth)
    
    # Calculate dogleg angle (B) using standard Minimum Curvature formula:
    # cos(B) = cos(I2-I1) - sin(I1)*sin(I2)*(1 - cos(A2-A1))
    cos_dI = math.cos(I2 - I1) - (math.sin(I1) * math.sin(I2) * (1 - math.cos(A2 - A1)))
    cos_dI = max(-1.0, min(1.0, cos_dI))  # Numerical stability
    B = math.acos(cos_dI)
    
    # Ratio Factor (rf): rf = (2/B) * tan(B/2)
    # This approaches 1.0 as B approaches 0 (straight hole)
    rf = 1.0 if abs(B) < 1e-9 else (2.0 / B) * math.tan(B / 2.0)
    
    # Minimum Curvature displacement calculations
    # Note: Z decreases downward, so we subtract dZ
    dX = 0.5 * dmd * (math.sin(I1) * math.sin(A1) + math.sin(I2) * math.sin(A2)) * rf
    dY = 0.5 * dmd * (math.sin(I1) * math.cos(A1) + math.sin(I2) * math.cos(A2)) * rf
    dZ = 0.5 * dmd * (math.cos(I1) + math.cos(I2)) * rf
    
    x = prev_x + dX
    y = prev_y + dY
    z = prev_z - dZ  # Z decreases downward
    
    return x, y, z


def minimum_curvature_path_from_surveys(
    collar_x: float,
    collar_y: float,
    collar_z: float,
    surveys: List[Dict[str, float]],
    default_azimuth: float = 0.0,
    default_dip: float = -90.0,
    total_depth: Optional[float] = None,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Calculate 3D drillhole path from survey intervals using Minimum Curvature.
    
    This function processes survey intervals (depth_from, depth_to, azimuth, dip)
    and returns the 3D coordinates at each survey station, matching the exact
    algorithm used in drillhole_layer.py.
    
    Args:
        collar_x, collar_y, collar_z: Collar coordinates
        surveys: List of survey dictionaries, each with keys:
            - 'depth_from': Start depth
            - 'depth_to': End depth  
            - 'azimuth': Azimuth in degrees
            - 'dip': Dip in degrees
        default_azimuth: Default azimuth to use if no surveys (degrees)
        default_dip: Default dip to use if no surveys (degrees, -90 = straight down)
        total_depth: Optional total depth to extend path to if no surveys
            
    Returns:
        Tuple of (depths, coordinates) where:
            - depths: List of depths at each station
            - coordinates: List of numpy arrays [x, y, z] at each station
            
    Notes:
        - Uses the same Minimum Curvature algorithm as drillhole_layer.py
        - Ensures mathematical consistency across the application
    """
    if not surveys:
        # No surveys - create simple vertical path if total_depth provided
        if total_depth and total_depth > 0:
            station_coords = [
                np.array([collar_x, collar_y, collar_z], dtype=float),
                np.array([collar_x, collar_y, collar_z - total_depth], dtype=float)
            ]
            return [0.0, total_depth], station_coords
        else:
            return [0.0], [np.array([collar_x, collar_y, collar_z], dtype=float)]
    
    # Sort surveys by depth
    sorted_surveys = sorted(surveys, key=lambda s: s.get('depth_from', 0.0))
    
    # Build station points dictionary (depth -> (azimuth, dip))
    station_points = {0.0: (sorted_surveys[0].get('azimuth', default_azimuth), sorted_surveys[0].get('dip', default_dip))}
    
    for survey in sorted_surveys:
        depth_from = survey.get('depth_from', 0.0)
        depth_to = survey.get('depth_to', 0.0)
        az = survey.get('azimuth', default_azimuth)
        dip = survey.get('dip', default_dip)
        station_points[depth_from] = (az, dip)
        station_points[depth_to] = (az, dip)
    
    # Add total_depth if provided and beyond last survey
    if total_depth is not None and total_depth > max(station_points.keys()):
        # Use the last survey's azimuth/dip for extension
        last_depth = max(station_points.keys())
        last_az, last_dip = station_points[last_depth]
        station_points[total_depth] = (last_az, last_dip)
    
    sorted_depths = sorted(station_points.keys())
    station_coords = [np.array([collar_x, collar_y, collar_z], dtype=float)]
    coord_depths = [sorted_depths[0]]
    
    # Calculate coordinates using Minimum Curvature for each segment
    for idx in range(1, len(sorted_depths)):
        prev_depth = sorted_depths[idx - 1]
        curr_depth = sorted_depths[idx]
        prev_az, prev_dip = station_points[prev_depth]
        curr_az, curr_dip = station_points[curr_depth]
        
        prev_x, prev_y, prev_z = station_coords[-1]
        x, y, z = minimum_curvature_segment(
            prev_x, prev_y, prev_z,
            prev_depth, curr_depth,
            prev_az, prev_dip,
            curr_az, curr_dip
        )
        
        station_coords.append(np.array([x, y, z], dtype=float))
        coord_depths.append(curr_depth)
    
    return coord_depths, station_coords