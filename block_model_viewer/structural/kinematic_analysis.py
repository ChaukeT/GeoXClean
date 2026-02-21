"""
Kinematic Analysis - Simple kinematic feasibility checks for slopes.

Supports structural features (faults, folds, unconformities) loaded from CSV.
"""

from typing import List, Dict, Any, Union, TYPE_CHECKING
import numpy as np

from .datasets import PlaneMeasurement

if TYPE_CHECKING:
    from .feature_types import FaultFeature, FoldFeature, UnconformityFeature, StructuralFeatureCollection


def kinematic_plane_slope_feasibility(
    planes: List[PlaneMeasurement],
    slope_dip: float,
    slope_dip_direction: float,
    phi: float
) -> Dict[str, Any]:
    """
    Check kinematic feasibility for plane failure.
    
    Args:
        planes: List of PlaneMeasurement objects
        slope_dip: Slope dip angle in degrees
        slope_dip_direction: Slope dip direction in degrees (azimuth)
        phi: Friction angle in degrees
    
    Returns:
        Dictionary with feasibility results
    """
    feasible_count = 0
    feasible_planes = []
    
    for plane in planes:
        # Check if plane daylight condition is met
        # Plane daylights if its dip direction is within 20 degrees of slope dip direction
        angle_diff = abs(plane.dip_direction - slope_dip_direction)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        if angle_diff > 20:
            continue
        
        # Check if plane dip is steeper than slope dip
        if plane.dip < slope_dip:
            continue
        
        # Check if plane dip is less than friction angle (would slide)
        if plane.dip > phi:
            continue
        
        # Plane is kinematically feasible
        feasible_count += 1
        feasible_planes.append(plane)
    
    return {
        "feasible_count": feasible_count,
        "total_count": len(planes),
        "feasible_fraction": feasible_count / len(planes) if planes else 0.0,
        "feasible_planes": feasible_planes,
    }


def kinematic_wedge_feasibility(
    planes: List[PlaneMeasurement],
    slope_dip: float,
    slope_dip_direction: float,
    phi: float
) -> Dict[str, Any]:
    """
    Check kinematic feasibility for wedge failure (simplified).
    
    Args:
        planes: List of PlaneMeasurement objects (need at least 2)
        slope_dip: Slope dip angle in degrees
        slope_dip_direction: Slope dip direction in degrees
        phi: Friction angle in degrees
    
    Returns:
        Dictionary with feasibility results
    """
    if len(planes) < 2:
        return {
            "feasible_count": 0,
            "total_pairs": 0,
            "feasible_fraction": 0.0,
            "feasible_pairs": [],
        }
    
    feasible_pairs = []
    
    # Check all pairs of planes
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            plane1 = planes[i]
            plane2 = planes[j]
            
            # Simplified check: both planes must daylight
            angle_diff1 = abs(plane1.dip_direction - slope_dip_direction)
            angle_diff1 = min(angle_diff1, 360 - angle_diff1)
            angle_diff2 = abs(plane2.dip_direction - slope_dip_direction)
            angle_diff2 = min(angle_diff2, 360 - angle_diff2)
            
            if angle_diff1 > 20 or angle_diff2 > 20:
                continue
            
            # Both planes must be steeper than slope
            if plane1.dip < slope_dip or plane2.dip < slope_dip:
                continue
            
            # Wedge line of intersection must plunge less than phi
            # Simplified: check if average dip is reasonable
            avg_dip = (plane1.dip + plane2.dip) / 2
            if avg_dip > phi:
                continue
            
            feasible_pairs.append((plane1, plane2))
    
    total_pairs = len(planes) * (len(planes) - 1) // 2
    
    return {
        "feasible_count": len(feasible_pairs),
        "total_pairs": total_pairs,
        "feasible_fraction": len(feasible_pairs) / total_pairs if total_pairs > 0 else 0.0,
        "feasible_pairs": feasible_pairs,
    }


# =============================================================================
# STRUCTURAL FEATURE SUPPORT
# =============================================================================

def fault_plane_stability(
    fault_feature: "FaultFeature",
    slope_dip: float,
    slope_dip_direction: float,
    phi: float,
) -> Dict[str, Any]:
    """
    Analyze kinematic stability of a fault feature against a slope face.
    
    Args:
        fault_feature: FaultFeature from structural CSV import
        slope_dip: Slope dip angle in degrees
        slope_dip_direction: Slope dip direction in degrees
        phi: Friction angle in degrees
        
    Returns:
        Dictionary with stability analysis results
    """
    # Extract plane measurements from fault orientations
    planes = []
    if hasattr(fault_feature, 'orientations'):
        for orient in fault_feature.orientations:
            planes.append(PlaneMeasurement(
                dip=orient.dip,
                dip_direction=orient.azimuth,
                set_id=fault_feature.name,
            ))
    
    if not planes:
        return {
            "fault_name": fault_feature.name,
            "analysis_possible": False,
            "reason": "No orientations available",
            "feasible_count": 0,
            "total_count": 0,
        }
    
    # Run plane slope feasibility
    result = kinematic_plane_slope_feasibility(planes, slope_dip, slope_dip_direction, phi)
    
    # Add fault-specific metadata
    result["fault_name"] = fault_feature.name
    result["analysis_possible"] = True
    result["displacement_type"] = fault_feature.displacement_type.value if hasattr(fault_feature.displacement_type, 'value') else str(fault_feature.displacement_type)
    
    # Risk assessment
    if result["feasible_fraction"] > 0.5:
        result["risk_level"] = "HIGH"
        result["recommendation"] = "Fault orientation highly unfavorable for slope stability"
    elif result["feasible_fraction"] > 0.2:
        result["risk_level"] = "MODERATE"
        result["recommendation"] = "Fault may contribute to slope instability"
    else:
        result["risk_level"] = "LOW"
        result["recommendation"] = "Fault orientation generally favorable"
    
    return result


def analyze_structural_collection_stability(
    collection: "StructuralFeatureCollection",
    slope_dip: float,
    slope_dip_direction: float,
    phi: float,
) -> Dict[str, Any]:
    """
    Analyze kinematic stability of all structural features in a collection.
    
    Args:
        collection: StructuralFeatureCollection from CSV import
        slope_dip: Slope dip angle in degrees
        slope_dip_direction: Slope dip direction in degrees
        phi: Friction angle in degrees
        
    Returns:
        Dictionary with comprehensive stability analysis
    """
    from .stereonet import collection_to_planes
    
    results = {
        "slope_parameters": {
            "dip": slope_dip,
            "dip_direction": slope_dip_direction,
            "friction_angle": phi,
        },
        "fault_analyses": [],
        "unconformity_analyses": [],
        "combined_plane_analysis": None,
        "combined_wedge_analysis": None,
        "overall_risk": "UNKNOWN",
        "summary": {},
    }
    
    # Analyze individual faults
    high_risk_faults = 0
    for fault in collection.faults:
        fault_result = fault_plane_stability(fault, slope_dip, slope_dip_direction, phi)
        results["fault_analyses"].append(fault_result)
        if fault_result.get("risk_level") == "HIGH":
            high_risk_faults += 1
    
    # Analyze unconformities similarly
    for unconformity in collection.unconformities:
        # Extract planes from unconformity
        planes = []
        if hasattr(unconformity, 'orientations'):
            for orient in unconformity.orientations:
                planes.append(PlaneMeasurement(
                    dip=orient.dip,
                    dip_direction=orient.azimuth,
                    set_id=unconformity.name,
                ))
        
        if planes:
            unc_result = kinematic_plane_slope_feasibility(planes, slope_dip, slope_dip_direction, phi)
            unc_result["unconformity_name"] = unconformity.name
            unc_result["unconformity_type"] = unconformity.unconformity_type.value if hasattr(unconformity.unconformity_type, 'value') else str(unconformity.unconformity_type)
            results["unconformity_analyses"].append(unc_result)
    
    # Combined analysis using all planes
    all_planes = collection_to_planes(collection)
    
    if all_planes:
        results["combined_plane_analysis"] = kinematic_plane_slope_feasibility(
            all_planes, slope_dip, slope_dip_direction, phi
        )
        
        results["combined_wedge_analysis"] = kinematic_wedge_feasibility(
            all_planes, slope_dip, slope_dip_direction, phi
        )
    
    # Calculate overall risk
    total_faults = len(collection.faults)
    if total_faults > 0:
        if high_risk_faults / total_faults > 0.5:
            results["overall_risk"] = "HIGH"
        elif high_risk_faults / total_faults > 0.2:
            results["overall_risk"] = "MODERATE"
        else:
            results["overall_risk"] = "LOW"
    elif results.get("combined_plane_analysis"):
        frac = results["combined_plane_analysis"].get("feasible_fraction", 0)
        if frac > 0.5:
            results["overall_risk"] = "HIGH"
        elif frac > 0.2:
            results["overall_risk"] = "MODERATE"
        else:
            results["overall_risk"] = "LOW"
    
    # Summary statistics
    results["summary"] = {
        "n_faults_analyzed": len(results["fault_analyses"]),
        "n_high_risk_faults": high_risk_faults,
        "n_unconformities_analyzed": len(results["unconformity_analyses"]),
        "n_planes_total": len(all_planes),
        "overall_risk_level": results["overall_risk"],
    }
    
    return results

