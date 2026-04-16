"""
LoopStructural utility functions.

Moved unchanged from the monolithic loopstructural_panel.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _filter_outliers_for_extent(df: pd.DataFrame, min_valid: float = 100.0) -> pd.DataFrame:
    """
    Filter out outlier coordinates before calculating model extent.

    This prevents the model extent from being stretched by invalid
    coordinates like (0, 0, 0) placeholder data.

    Args:
        df: DataFrame with X, Y, Z columns
        min_valid: Minimum absolute value for valid X/Y coordinates

    Returns:
        Filtered DataFrame with outliers removed
    """
    if df is None or len(df) == 0:
        return df

    # Remove rows where X and Y are both very small (likely placeholder data)
    mask = ~((np.abs(df['X']) < min_valid) & (np.abs(df['Y']) < min_valid))

    # Only filter if we would retain at least 50% of data
    if mask.sum() >= len(df) * 0.5:
        filtered = df[mask].copy()
        if len(filtered) < len(df):
            logger.info(
                f"Filtered {len(df) - len(filtered)} outlier points for extent calculation "
                f"(X and Y both < {min_valid}m)"
            )
        return filtered

    return df


def _calculate_proportional_scalar_spacing(
    df: pd.DataFrame,
    stratigraphy: List[str],
    min_spacing: float = 0.5,
    hole_id_col: str = 'hole_id'
) -> Dict[str, float]:
    """
    Calculate proportional scalar values based on unit thicknesses.

    For geological modeling, thin units should have smaller scalar ranges
    to ensure proper interpolation without numerical artifacts.

    Algorithm:
    1. Calculate average thickness for each unit from drillhole intersections
    2. Normalize thicknesses to total thickness
    3. Apply minimum spacing to prevent collapse of thin units
    4. Return scalar value mapping for each formation

    Args:
        df: DataFrame with formation, Z, and hole_id columns
        stratigraphy: Ordered list of formation names (oldest to youngest)
        min_spacing: Minimum scalar spacing between units (default 0.5)
        hole_id_col: Column name for drillhole identification

    Returns:
        Dict mapping formation names to scalar values
    """
    if df is None or len(df) == 0 or not stratigraphy:
        # Fall back to sequential spacing
        return {form: float(i) for i, form in enumerate(stratigraphy)}

    # Check if we have hole_id column for thickness calculation
    if hole_id_col not in df.columns:
        # Try common alternatives
        for alt_col in ['HOLEID', 'HoleID', 'BHID', 'hole', 'drillhole_id']:
            if alt_col in df.columns:
                hole_id_col = alt_col
                break
        else:
            # Can't calculate thicknesses, use sequential
            logger.info("No hole_id column found - using sequential scalar spacing")
            return {form: float(i) for i, form in enumerate(stratigraphy)}

    # Calculate average thickness for each formation across all holes
    thicknesses = {}

    for hole_id in df[hole_id_col].unique():
        hole_data = df[df[hole_id_col] == hole_id].copy()

        if 'formation' not in hole_data.columns:
            continue

        # Sort by depth (Z, descending for typical drillholes)
        hole_data = hole_data.sort_values('Z', ascending=False)

        # Group consecutive same-formation intervals
        prev_formation = None
        interval_start_z = None

        for _, row in hole_data.iterrows():
            formation = row.get('formation')
            z = row.get('Z', 0)

            if pd.isna(formation):
                continue

            if formation != prev_formation:
                # Close previous interval
                if prev_formation is not None and interval_start_z is not None:
                    thickness = interval_start_z - z
                    if thickness > 0:
                        if prev_formation not in thicknesses:
                            thicknesses[prev_formation] = []
                        thicknesses[prev_formation].append(thickness)

                # Start new interval
                interval_start_z = z
                prev_formation = formation

    # Calculate average thicknesses
    avg_thicknesses = {}
    for form in stratigraphy:
        if form in thicknesses and len(thicknesses[form]) > 0:
            avg_thicknesses[form] = np.mean(thicknesses[form])
        else:
            # Use default minimum thickness for unknown units
            avg_thicknesses[form] = 1.0  # Default 1m

    # Normalize to proportional values
    total_thickness = sum(avg_thicknesses.values())
    if total_thickness < 1e-10:
        total_thickness = len(stratigraphy)

    # Calculate cumulative scalar values
    # Apply minimum spacing to ensure thin units are represented
    cumulative_val = 0.0
    formation_to_val = {}

    for i, form in enumerate(stratigraphy):
        formation_to_val[form] = cumulative_val

        # Calculate proportional spacing
        proportion = avg_thicknesses.get(form, 1.0) / total_thickness * len(stratigraphy)
        spacing = max(min_spacing, proportion)
        cumulative_val += spacing

    logger.info(f"Proportional scalar spacing calculated: {formation_to_val}")
    logger.info(f"Average thicknesses (m): {avg_thicknesses}")

    return formation_to_val
