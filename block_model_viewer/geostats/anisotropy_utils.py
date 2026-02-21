"""
Anisotropy utilities for proper range ordering and naming conventions.

Professional geostatistical practice expects:
- Primary (major) axis: longest range
- Secondary (semi-major): intermediate range  
- Minor (tertiary): shortest range

This module provides utilities to enforce this convention.
"""

from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def enforce_range_ordering(
    major_range: float,
    minor_range: float, 
    vertical_range: float,
    use_descriptive_names: bool = True
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Enforce proper anisotropy range ordering convention.
    
    Industry standard expects Primary >= Secondary >= Tertiary ranges.
    When user-specified "major" and "minor" don't follow this convention,
    this function reorders them and provides appropriate warnings.
    
    Args:
        major_range: User-specified "major" range
        minor_range: User-specified "minor" range  
        vertical_range: Vertical range
        use_descriptive_names: If True, use Primary/Secondary/Tertiary names
                              If False, preserve Major/Minor but reorder values
    
    Returns:
        Tuple of:
        - Dict with properly ordered ranges 
        - Dict with naming information/warnings
    """
    # Collect all ranges with their original names
    ranges = [
        (major_range, "major", "horizontal_1"),
        (minor_range, "minor", "horizontal_2"), 
        (vertical_range, "vertical", "vertical")
    ]
    
    # Sort by range value (descending)
    ranges.sort(key=lambda x: x[0], reverse=True)
    
    # Determine naming strategy
    if use_descriptive_names:
        # Use Primary/Secondary/Tertiary (clearest for users)
        ordered_names = ["primary", "secondary", "tertiary"]
    else:
        # Keep Major/Minor/Vertical but reorder values
        ordered_names = ["major", "minor", "vertical"]
    
    # Build result
    ordered_ranges = {}
    naming_info = {
        "reordered": False,
        "original_names": [],
        "new_names": [],
        "warnings": []
    }
    
    # Check if reordering occurred
    original_order = [major_range, minor_range, vertical_range]
    current_order = [r[0] for r in ranges]
    
    if original_order != current_order:
        naming_info["reordered"] = True
        logger.warning(
            f"Range ordering corrected: original({major_range:.1f}, {minor_range:.1f}, {vertical_range:.1f}) "
            f"→ ordered({current_order[0]:.1f}, {current_order[1]:.1f}, {current_order[2]:.1f})"
        )
        
        # Provide user-friendly explanation
        if major_range < minor_range:
            naming_info["warnings"].append(
                f"Original 'Minor' range ({minor_range:.1f}m) > 'Major' range ({major_range:.1f}m). "
                "Ranges have been reordered to follow geostatistical convention."
            )
    
    # Assign ordered ranges
    for i, (range_val, orig_name, spatial_type) in enumerate(ranges):
        new_name = ordered_names[i]
        ordered_ranges[f"{new_name}_range"] = range_val
        
        # Track naming changes
        naming_info["original_names"].append(orig_name)
        naming_info["new_names"].append(new_name)
        
        # Add directional information
        ordered_ranges[f"{new_name}_direction"] = spatial_type
        
        # Add anisotropy ratios relative to primary
        if i == 0:  # Primary (longest)
            ordered_ranges[f"{new_name}_ratio"] = 1.0
        else:
            ratio = range_val / ranges[0][0] if ranges[0][0] > 0 else 1.0
            ordered_ranges[f"{new_name}_ratio"] = ratio
    
    # Maintain backward compatibility by also providing major/minor/vertical
    if use_descriptive_names:
        ordered_ranges["major_range"] = ranges[0][0]  # Primary becomes major
        ordered_ranges["minor_range"] = ranges[1][0]  # Secondary becomes minor  
        ordered_ranges["vertical_range"] = ranges[2][0]  # Tertiary becomes vertical
    
    return ordered_ranges, naming_info


def create_anisotropy_summary(ranges_info: Dict[str, Any], naming_info: Dict[str, str]) -> str:
    """
    Create user-friendly anisotropy summary with proper naming.
    
    Args:
        ranges_info: Ordered ranges from enforce_range_ordering
        naming_info: Naming information from enforce_range_ordering
    
    Returns:
        HTML formatted summary string
    """
    primary = ranges_info.get("primary_range", ranges_info.get("major_range", 0))
    secondary = ranges_info.get("secondary_range", ranges_info.get("minor_range", 0))
    tertiary = ranges_info.get("tertiary_range", ranges_info.get("vertical_range", 0))
    
    # Ratios relative to primary axis
    sec_ratio = secondary / primary if primary > 0 else 1.0
    tert_ratio = tertiary / primary if primary > 0 else 1.0
    
    summary = f"""
    <b>Anisotropy (Ordered by Range):</b><br>
    Primary: {primary:.1f}m (1.00)<br>
    Secondary: {secondary:.1f}m ({sec_ratio:.2f})<br>
    Tertiary: {tertiary:.1f}m ({tert_ratio:.2f})
    """
    
    # Add warnings if reordering occurred
    if naming_info.get("reordered", False):
        summary += "<br><br><span style='color: #ffb74d'>"
        summary += "<b>⚠ Range Ordering Corrected:</b><br>"
        for warning in naming_info.get("warnings", []):
            summary += f"<i>{warning}</i><br>"
        summary += "</span>"
    
    return summary


def validate_anisotropy_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and standardize anisotropy parameters.
    
    Args:
        params: Dictionary containing anisotropy parameters
    
    Returns:
        Dictionary with validated and standardized parameters
    """
    # Extract ranges with sensible defaults
    major_range = params.get("major_range", params.get("range", 100.0))
    minor_range = params.get("minor_range", major_range * 0.5)
    vertical_range = params.get("vertical_range", params.get("vert_range", major_range * 0.25))
    
    # Enforce ordering
    ordered_ranges, naming_info = enforce_range_ordering(
        major_range, minor_range, vertical_range, use_descriptive_names=False
    )
    
    # Update parameters with ordered ranges
    result = params.copy()
    result.update(ordered_ranges)
    result["_anisotropy_naming_info"] = naming_info
    
    return result