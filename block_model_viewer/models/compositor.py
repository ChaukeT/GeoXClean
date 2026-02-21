"""
Drillhole Compositor.

Standardizes assay intervals to regular lengths (e.g., 1m) for geostatistics.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def composite_drillholes(
    df: pd.DataFrame,
    interval_length: float = 1.0,
    min_comp_length: float = 0.5,
    hole_id_col: str = 'HOLEID',
    from_col: str = 'FROM',
    to_col: str = 'TO',
    grade_cols: list = None
) -> pd.DataFrame:
    """
    Regularizes drillhole intervals (Downhole Compositing).
    
    Args:
        df: DataFrame with drillhole assay data
        interval_length: Target composite length (default 1.0m)
        min_comp_length: Minimum composite length to keep (default 0.5m)
        hole_id_col: Column name for hole ID
        from_col: Column name for interval start depth
        to_col: Column name for interval end depth
        grade_cols: List of grade column names to composite (default: ['GRADE'])
    
    Returns:
        DataFrame with composited intervals
    """
    if grade_cols is None:
        grade_cols = ['GRADE']  # Default

    composites = []

    # Group by Hole
    grouped = df.groupby(hole_id_col)

    for hole_id, group in grouped:
        group = group.sort_values(from_col)
        
        # Calculate total depth
        max_depth = group[to_col].max()
        
        # Generate new intervals
        # 0 to max_depth with step = interval_length
        new_from = np.arange(0, max_depth, interval_length)
        new_to = new_from + interval_length
        
        # Clip last interval to max_depth
        new_to[-1] = min(new_to[-1], max_depth)
        
        # Filter small residuals at end of hole
        lengths = new_to - new_from
        valid_mask = lengths >= min_comp_length
        
        new_from = new_from[valid_mask]
        new_to = new_to[valid_mask]
        
        # Create Composite DataFrame for this hole
        hole_comps = pd.DataFrame({
            hole_id_col: hole_id,
            from_col: new_from,
            to_col: new_to,
            'LENGTH': new_to - new_from
        })
        
        # Weighted Average for Grades
        # This is a simplified length-weighted average. 
        # Production code usually handles gaps/missing data more robustly.
        for g_col in grade_cols:
            hole_comps[g_col] = np.nan
            
            # Slow loop for clarity (Vectorize this in production if needed)
            for i, row in hole_comps.iterrows():
                f, t = row[from_col], row[to_col]
                
                # Find overlapping original intervals
                # Overlap logic: max(start1, start2) < min(end1, end2)
                overlaps = group[
                    (group[from_col] < t) & (group[to_col] > f)
                ].copy()
                
                if not overlaps.empty:
                    # Calculate intersection length
                    overlaps['seg_from'] = np.maximum(overlaps[from_col], f)
                    overlaps['seg_to'] = np.minimum(overlaps[to_col], t)
                    overlaps['seg_len'] = overlaps['seg_to'] - overlaps['seg_from']
                    
                    # Weighted Avg
                    w_grade = np.sum(overlaps[g_col] * overlaps['seg_len'])
                    total_len = np.sum(overlaps['seg_len'])
                    
                    if total_len > 0:
                        hole_comps.at[i, g_col] = w_grade / total_len

        composites.append(hole_comps)

    if not composites:
        return pd.DataFrame()

    return pd.concat(composites, ignore_index=True)

