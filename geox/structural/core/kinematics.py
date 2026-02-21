"""
Kinematic Feasibility Analysis - Explicit, Parameter-Driven.

Implements Markland-style kinematic feasibility checks in vector form:
- Planar sliding
- Wedge sliding (intersection line)
- Toppling (flexural and block)

All checks use explicit parameters with no magic numbers.
Intermediate quantities are returned for full audit trail.

References:
- Hoek, E. and Bray, J. (1981). Rock Slope Engineering.
- Goodman, R.E. (1989). Introduction to Rock Mechanics.
- Wyllie, D.C. and Mah, C.W. (2004). Rock Slope Engineering.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from .models import (
    OrientationData,
    KinematicResult,
    KinematicFeasibility,
    FailureMode,
)
from .orientation_math import (
    dip_dipdir_to_normal,
    normal_to_dip_dipdir,
    canonicalize_to_lower_hemisphere,
    angle_between_vectors,
    normalize_vectors,
)


# =============================================================================
# SLOPE GEOMETRY
# =============================================================================

def compute_slope_normal(slope_dip: float, slope_dip_direction: float) -> np.ndarray:
    """
    Compute the outward-facing normal of a slope face.
    
    The normal points OUT of the slope (into the excavation).
    
    Args:
        slope_dip: Slope face dip in degrees (0-90)
        slope_dip_direction: Slope face dip direction in degrees (0-360)
    
    Returns:
        3-element unit normal vector
    """
    # The slope normal points in the dip direction (outward from slope)
    normal = dip_dipdir_to_normal(slope_dip, slope_dip_direction)
    return normal.flatten()


def compute_daylight_envelope(
    slope_dip: float,
    slope_dip_direction: float,
    n_points: int = 180
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the daylight envelope for planar sliding.
    
    The daylight envelope is the locus of pole orientations that
    would daylight on the slope face.
    
    Args:
        slope_dip: Slope dip in degrees
        slope_dip_direction: Slope dip direction in degrees
        n_points: Number of points for envelope
    
    Returns:
        Tuple of (x, y) projected coordinates for envelope
    """
    from .stereonet import project_schmidt, Hemisphere
    
    # The daylight envelope is a great circle perpendicular to the slope
    # Poles plotting within this arc could daylight
    
    # Generate points along the daylight envelope
    # The envelope passes through the slope pole position
    slope_normal = compute_slope_normal(slope_dip, slope_dip_direction)
    
    # The daylight boundary is where the plane just touches the slope
    # This is a circle at angle = slope_dip from the center
    
    t = np.linspace(0, 2 * np.pi, n_points)
    
    # Create envelope as a cone around the slope direction
    # Simplified: use the slope dip direction ± 90 degrees
    dd_rad = np.radians(slope_dip_direction)
    
    # Envelope is approximately a great circle through the slope pole
    # For planar sliding, planes must dip toward the slope
    envelope_normals = []
    for angle in t:
        # Points along the envelope
        az = slope_dip_direction + np.degrees(angle)
        envelope_normals.append(dip_dipdir_to_normal(slope_dip, az % 360))
    
    envelope_normals = np.vstack(envelope_normals)
    x, y = project_schmidt(envelope_normals, Hemisphere.LOWER)
    
    return x, y


def compute_friction_cone(
    friction_angle: float,
    n_points: int = 180
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the friction cone (small circle) on stereonet.
    
    Poles plotting within this cone have dips less than the friction angle
    and would not slide.
    
    Args:
        friction_angle: Friction angle in degrees
        n_points: Number of points
    
    Returns:
        Tuple of (x, y) projected coordinates
    """
    from .stereonet import compute_small_circle, Hemisphere, NetType
    
    # Friction cone is centered on vertical (0, 0, -1)
    x, y = compute_small_circle(
        axis=np.array([0, 0, -1]),
        cone_angle=friction_angle,
        n_points=n_points,
        net_type=NetType.SCHMIDT,
        hemisphere=Hemisphere.LOWER
    )
    
    return x, y


def line_of_intersection(
    normal1: np.ndarray,
    normal2: np.ndarray
) -> np.ndarray:
    """
    Compute the line of intersection of two planes.
    
    Args:
        normal1: First plane normal (3-element)
        normal2: Second plane normal (3-element)
    
    Returns:
        Unit vector along line of intersection
        Returns zero vector if planes are parallel
    """
    normal1 = np.asarray(normal1, dtype=np.float64).flatten()
    normal2 = np.asarray(normal2, dtype=np.float64).flatten()
    
    # Line of intersection is perpendicular to both normals
    intersection = np.cross(normal1, normal2)
    
    magnitude = np.linalg.norm(intersection)
    if magnitude < 1e-10:
        return np.zeros(3)
    
    return intersection / magnitude


# =============================================================================
# PLANAR SLIDING
# =============================================================================

def planar_sliding_feasibility(
    normals: np.ndarray,
    slope_dip: float,
    slope_dip_direction: float,
    friction_angle: float,
    lateral_limits: Optional[float] = None,
) -> List[KinematicFeasibility]:
    """
    Check kinematic feasibility for planar sliding.
    
    A plane can slide if:
    1. Daylight condition: plane dips toward the slope face (within ±90° of slope dip direction)
       AND plane dip > 0 AND plane dip < slope dip
    2. Friction condition: plane dip > friction angle
    3. Lateral condition (optional): within lateral limits of slope dip direction
    
    Args:
        normals: Nx3 array of plane pole unit vectors
        slope_dip: Slope face dip in degrees
        slope_dip_direction: Slope face dip direction in degrees
        friction_angle: Friction angle in degrees
        lateral_limits: Optional lateral limit in degrees (default: 20°)
    
    Returns:
        List of KinematicFeasibility results, one per measurement
    """
    normals = np.atleast_2d(normals)
    normals = canonicalize_to_lower_hemisphere(normals)
    
    if lateral_limits is None:
        lateral_limits = 20.0  # Default Markland limit
    
    # Get plane dips and dip directions
    plane_dips, plane_dip_dirs = normal_to_dip_dipdir(normals)
    
    results = []
    
    for i in range(len(normals)):
        plane_dip = plane_dips[i]
        plane_dd = plane_dip_dirs[i]
        
        # 1. Daylight condition
        # Plane must dip less steeply than slope but steeper than friction angle
        # AND dip direction must be within lateral limits of slope dip direction
        
        # Angular difference in dip direction
        dd_diff = abs(plane_dd - slope_dip_direction)
        dd_diff = min(dd_diff, 360 - dd_diff)  # Wrap around
        
        # Daylight: dip < slope dip AND dip direction within 90° of slope
        daylights = (plane_dip > 0 and 
                     plane_dip < slope_dip and 
                     dd_diff < 90)
        
        daylight_margin = slope_dip - plane_dip if daylights else plane_dip - slope_dip
        
        # 2. Friction condition
        friction_ok = plane_dip > friction_angle
        friction_margin = plane_dip - friction_angle
        
        # 3. Lateral condition
        lateral_ok = dd_diff <= lateral_limits
        lateral_margin = lateral_limits - dd_diff
        
        # Overall feasibility
        is_feasible = daylights and friction_ok and lateral_ok
        
        results.append(KinematicFeasibility(
            is_feasible=is_feasible,
            failure_mode=FailureMode.PLANAR,
            daylight_condition=daylights,
            friction_condition=friction_ok,
            lateral_condition=lateral_ok,
            daylight_margin_deg=daylight_margin,
            friction_margin_deg=friction_margin,
            lateral_margin_deg=lateral_margin,
            measurement_index=i,
            measurement_normal=normals[i].copy(),
        ))
    
    return results


# =============================================================================
# WEDGE SLIDING
# =============================================================================

def wedge_sliding_feasibility(
    normals: np.ndarray,
    slope_dip: float,
    slope_dip_direction: float,
    friction_angle: float,
    lateral_limits: Optional[float] = None,
) -> List[KinematicFeasibility]:
    """
    Check kinematic feasibility for wedge failure.
    
    A wedge can slide along the line of intersection of two planes if:
    1. Daylight condition: intersection line plunges toward the slope face
       AND intersection plunge < slope dip
    2. Friction condition: intersection plunge > friction angle
    3. Lateral condition: intersection trend within lateral limits
    
    Args:
        normals: Nx3 array of plane pole unit vectors
        slope_dip: Slope face dip in degrees
        slope_dip_direction: Slope face dip direction in degrees
        friction_angle: Friction angle in degrees
        lateral_limits: Optional lateral limit in degrees
    
    Returns:
        List of KinematicFeasibility results, one per plane pair
    """
    normals = np.atleast_2d(normals)
    normals = canonicalize_to_lower_hemisphere(normals)
    n = len(normals)
    
    if lateral_limits is None:
        lateral_limits = 20.0
    
    if n < 2:
        return []
    
    from .orientation_math import vector_to_plunge_trend
    
    results = []
    
    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Compute line of intersection
            intersection = line_of_intersection(normals[i], normals[j])
            
            if np.linalg.norm(intersection) < 1e-10:
                # Parallel planes - no wedge
                continue
            
            # Ensure intersection points downward
            if intersection[2] > 0:
                intersection = -intersection
            
            # Get plunge and trend
            plunge, trend = vector_to_plunge_trend(intersection.reshape(1, 3))
            plunge = float(plunge[0])
            trend = float(trend[0])
            
            # 1. Daylight condition
            trend_diff = abs(trend - slope_dip_direction)
            trend_diff = min(trend_diff, 360 - trend_diff)
            
            daylights = (plunge > 0 and 
                        plunge < slope_dip and 
                        trend_diff < 90)
            
            daylight_margin = slope_dip - plunge if daylights else plunge - slope_dip
            
            # 2. Friction condition
            friction_ok = plunge > friction_angle
            friction_margin = plunge - friction_angle
            
            # 3. Lateral condition
            lateral_ok = trend_diff <= lateral_limits
            lateral_margin = lateral_limits - trend_diff
            
            is_feasible = daylights and friction_ok and lateral_ok
            
            results.append(KinematicFeasibility(
                is_feasible=is_feasible,
                failure_mode=FailureMode.WEDGE,
                daylight_condition=daylights,
                friction_condition=friction_ok,
                lateral_condition=lateral_ok,
                daylight_margin_deg=daylight_margin,
                friction_margin_deg=friction_margin,
                lateral_margin_deg=lateral_margin,
                measurement_index=i,  # First plane index
                measurement_normal=intersection,  # Store intersection line
                limiting_vectors={
                    "plane1_index": i,
                    "plane2_index": j,
                    "intersection_plunge": plunge,
                    "intersection_trend": trend,
                }
            ))
    
    return results


# =============================================================================
# TOPPLING
# =============================================================================

def toppling_feasibility(
    normals: np.ndarray,
    slope_dip: float,
    slope_dip_direction: float,
    friction_angle: float,
    base_plane_dip: Optional[float] = None,
) -> List[KinematicFeasibility]:
    """
    Check kinematic feasibility for toppling failure.
    
    Toppling can occur when:
    1. Plane strikes roughly parallel to slope (within ±30°)
    2. Plane dips into the slope (opposite to slope dip direction)
    3. Plane dip > (90 - slope_dip - friction_angle) for direct toppling
       OR additional check for flexural toppling
    
    Args:
        normals: Nx3 array of plane pole unit vectors
        slope_dip: Slope face dip in degrees
        slope_dip_direction: Slope face dip direction in degrees
        friction_angle: Friction angle in degrees (usually for base plane)
        base_plane_dip: Optional dip of base plane (default: horizontal)
    
    Returns:
        List of KinematicFeasibility results
    """
    normals = np.atleast_2d(normals)
    normals = canonicalize_to_lower_hemisphere(normals)
    
    from .orientation_math import strike_from_dip_direction
    
    plane_dips, plane_dip_dirs = normal_to_dip_dipdir(normals)
    
    # Slope strike
    slope_strike_arr = strike_from_dip_direction(slope_dip_direction)
    slope_strike = float(slope_strike_arr[0]) if hasattr(slope_strike_arr, '__len__') else float(slope_strike_arr)
    
    # Critical angle for toppling
    # Planes must dip steeper than (90 - slope_dip - friction_angle)
    critical_dip = 90 - slope_dip - friction_angle
    critical_dip = max(0, critical_dip)
    
    results = []
    
    for i in range(len(normals)):
        plane_dip = plane_dips[i]
        plane_dd = plane_dip_dirs[i]
        
        # Plane strike
        plane_strike_arr = strike_from_dip_direction(plane_dd)
        plane_strike = float(plane_strike_arr[0]) if hasattr(plane_strike_arr, '__len__') else float(plane_strike_arr)
        
        # 1. Strike parallel check (within ±30° of slope strike)
        strike_diff = abs(plane_strike - slope_strike)
        strike_diff = min(strike_diff, 360 - strike_diff)
        
        strike_parallel = strike_diff < 30
        
        # 2. Dips into slope (dip direction opposite to slope)
        dd_diff = abs(plane_dd - slope_dip_direction)
        dd_diff = min(dd_diff, 360 - dd_diff)
        
        dips_into_slope = dd_diff > 90  # Dips away from slope face
        
        # 3. Sufficient steepness for toppling
        steep_enough = plane_dip > critical_dip
        
        # Daylight for toppling is the strike parallel condition
        daylights = strike_parallel and dips_into_slope
        daylight_margin = 30 - strike_diff if strike_parallel else strike_diff - 30
        
        # Friction condition for toppling
        friction_ok = steep_enough
        friction_margin = plane_dip - critical_dip
        
        # Lateral isn't typically used for toppling
        lateral_ok = True
        
        is_feasible = daylights and friction_ok
        
        results.append(KinematicFeasibility(
            is_feasible=is_feasible,
            failure_mode=FailureMode.TOPPLING,
            daylight_condition=daylights,
            friction_condition=friction_ok,
            lateral_condition=lateral_ok,
            daylight_margin_deg=daylight_margin,
            friction_margin_deg=friction_margin,
            lateral_margin_deg=None,
            measurement_index=i,
            measurement_normal=normals[i].copy(),
            limiting_vectors={
                "strike_difference": strike_diff,
                "dip_direction_difference": dd_diff,
                "critical_dip": critical_dip,
            }
        ))
    
    return results


# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================

def kinematic_analysis(
    data: OrientationData,
    slope_dip: float,
    slope_dip_direction: float,
    friction_angle: float,
    lateral_limits: Optional[float] = None,
    tension_cutoff: Optional[float] = None,
    analyze_planar: bool = True,
    analyze_wedge: bool = True,
    analyze_toppling: bool = True,
) -> KinematicResult:
    """
    Perform comprehensive kinematic feasibility analysis.
    
    Args:
        data: OrientationData with plane pole normals
        slope_dip: Slope face dip in degrees
        slope_dip_direction: Slope face dip direction in degrees
        friction_angle: Friction angle in degrees
        lateral_limits: Lateral limit in degrees (default: 20°)
        tension_cutoff: Tension crack cutoff angle (not implemented yet)
        analyze_planar: Whether to check planar sliding
        analyze_wedge: Whether to check wedge sliding
        analyze_toppling: Whether to check toppling
    
    Returns:
        KinematicResult with all feasibility results and envelopes
    """
    normals = canonicalize_to_lower_hemisphere(data.normals)
    
    # Compute slope normal for result
    slope_normal = compute_slope_normal(slope_dip, slope_dip_direction)
    
    # Initialize result
    result = KinematicResult(
        slope_dip=slope_dip,
        slope_dip_direction=slope_dip_direction,
        slope_normal=slope_normal,
        friction_angle=friction_angle,
        lateral_limits=lateral_limits or 20.0,
        tension_cutoff=tension_cutoff,
        n_total_measurements=len(normals),
        parameters={
            "analyze_planar": analyze_planar,
            "analyze_wedge": analyze_wedge,
            "analyze_toppling": analyze_toppling,
        }
    )
    
    # Planar analysis
    if analyze_planar:
        planar_results = planar_sliding_feasibility(
            normals, slope_dip, slope_dip_direction, friction_angle, lateral_limits
        )
        result.planar_results = planar_results
        result.n_planar_feasible = sum(1 for r in planar_results if r.is_feasible)
    
    # Wedge analysis
    if analyze_wedge:
        wedge_results = wedge_sliding_feasibility(
            normals, slope_dip, slope_dip_direction, friction_angle, lateral_limits
        )
        result.wedge_results = wedge_results
        result.n_wedge_feasible = sum(1 for r in wedge_results if r.is_feasible)
    
    # Toppling analysis
    if analyze_toppling:
        toppling_results = toppling_feasibility(
            normals, slope_dip, slope_dip_direction, friction_angle
        )
        result.toppling_results = toppling_results
        result.n_toppling_feasible = sum(1 for r in toppling_results if r.is_feasible)
    
    # Compute envelopes for plotting
    try:
        result.daylight_envelope = compute_daylight_envelope(slope_dip, slope_dip_direction)
        result.friction_envelope = compute_friction_cone(friction_angle)
    except Exception:
        pass  # Envelopes are optional
    
    return result


def summarize_kinematic_results(result: KinematicResult) -> Dict[str, Any]:
    """
    Create a summary of kinematic analysis results.
    
    Args:
        result: KinematicResult from kinematic_analysis
    
    Returns:
        Dictionary with summary statistics and risk assessment
    """
    summary = {
        "slope_parameters": {
            "dip": result.slope_dip,
            "dip_direction": result.slope_dip_direction,
            "friction_angle": result.friction_angle,
            "lateral_limits": result.lateral_limits,
        },
        "n_measurements": result.n_total_measurements,
        "planar": {
            "n_feasible": result.n_planar_feasible,
            "fraction": result.planar_feasible_fraction,
        },
        "wedge": {
            "n_feasible": result.n_wedge_feasible,
            "n_pairs_analyzed": len(result.wedge_results),
            "fraction": result.wedge_feasible_fraction,
        },
        "toppling": {
            "n_feasible": result.n_toppling_feasible,
            "fraction": result.toppling_feasible_fraction,
        },
    }
    
    # Risk assessment
    max_fraction = max(
        result.planar_feasible_fraction,
        result.wedge_feasible_fraction,
        result.toppling_feasible_fraction
    )
    
    if max_fraction > 0.5:
        risk_level = "HIGH"
    elif max_fraction > 0.2:
        risk_level = "MODERATE"
    elif max_fraction > 0.05:
        risk_level = "LOW"
    else:
        risk_level = "VERY_LOW"
    
    summary["risk_level"] = risk_level
    
    # Dominant failure mode
    if result.n_planar_feasible > max(result.n_wedge_feasible, result.n_toppling_feasible):
        summary["dominant_mode"] = "planar"
    elif result.n_wedge_feasible > result.n_toppling_feasible:
        summary["dominant_mode"] = "wedge"
    elif result.n_toppling_feasible > 0:
        summary["dominant_mode"] = "toppling"
    else:
        summary["dominant_mode"] = "none"
    
    return summary

