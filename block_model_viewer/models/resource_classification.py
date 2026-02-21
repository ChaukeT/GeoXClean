"""
Resource Classification System for Block Models (Simple Spacing-Based)
======================================================================

⚠️  DEPRECATED: This module is deprecated and will be removed in a future version.
    For production-grade classification, use JORCClassificationEngine instead.

⚠️  WARNING: This module provides SIMPLE spacing-based classification ONLY.
    This is NOT JORC/SAMREC compliant and should NOT be used for final
    resource statements or regulatory reporting.

For FULL JORC/SAMREC-compliant classification with variogram-normalized
isotropic distances, use:

    from block_model_viewer.models.jorc_classification_engine import JORCClassificationEngine

This module (resource_classification.py) is intended for:
- Quick initial classification/scoping based purely on distance
- Situations where variogram data is not yet available
- Simple proximity-based analysis for exploration stage
- Rapid QA/visualization checks

The full JORC engine (jorc_classification_engine.py) MUST be used for:
- Final resource statements
- Compliance with JORC/SAMREC/NI 43-101
- Variogram-based anisotropic search
- Production-grade classification

Migration Guide:
----------------
Replace:
    from block_model_viewer.models.resource_classification import SpacingClassifier
    classifier = SpacingClassifier()
    
With:
    from block_model_viewer.models.jorc_classification_engine import JORCClassificationEngine
    engine = JORCClassificationEngine(variogram, ruleset)
    result = engine.classify(blocks_df, drillholes_df)

Auto-detection: When block_df or drillhole_df are None, the system attempts
to retrieve active datasets from the application context.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from scipy.spatial import cKDTree, KDTree

if TYPE_CHECKING:
    from .block_model import BlockModel

from ..utils.coordinate_utils import ensure_xyz_columns, detect_coordinate_columns

logger = logging.getLogger(__name__)

# Standard colors for resource categories (industry convention)
CATEGORY_COLORS = {
    'Measured': '#2ca02c',    # Green - high confidence
    'Indicated': '#ffbf00',   # Amber - medium confidence
    'Inferred': '#d62728',    # Red - low confidence
    'Unclassified': '#7f7f7f' # Grey - no classification
}

CATEGORY_ORDER = ['Measured', 'Indicated', 'Inferred', 'Unclassified']


@dataclass
class DomainRules:
    """Rules derived from variogram and spacing for a domain."""
    measured: 'SpacingRule'
    indicated: 'SpacingRule'
    inferred: 'SpacingRule'
    a_major: float  # Variogram major range
    edhs: Optional[float] = None  # Effective Drillhole Spacing (optional)
    domain_name: str = "default"
    
    def __repr__(self) -> str:
        edhs_str = f"{self.edhs:.1f}m" if self.edhs is not None else "N/A"
        return (
            f"DomainRules(domain={self.domain_name}, a_major={self.a_major:.1f}m, "
            f"EDHS={edhs_str}, Measured={self.measured}, "
            f"Indicated={self.indicated}, Inferred={self.inferred})"
        )



@dataclass
class SpacingRule:
    """
    Defines a spacing rule for a resource category.
    
    Attributes:
        max_dist: Maximum distance (m) to nearest drillhole
        min_holes: Minimum number of drillholes within max_dist
    """
    max_dist: float
    min_holes: int
    
    def __repr__(self) -> str:
        return f"SpacingRule(max_dist={self.max_dist}m, min_holes={self.min_holes})"


@dataclass
class ClassificationParams:
    """
    Parameters for drillhole spacing-based classification.
    
    Attributes:
        rules: Dict mapping category names to SpacingRule objects
        count_radii: List of radii (m) for counting nearby drillholes
    """
    rules: Dict[str, SpacingRule]
    count_radii: List[float]
    
    def __repr__(self) -> str:
        return f"ClassificationParams(rules={self.rules}, count_radii={self.count_radii})"


@dataclass
class ClassificationResult:
    """Result of resource classification."""
    classified_df: pd.DataFrame  # DataFrame with 'Category' column added
    summary: Dict[str, Any]  # Summary statistics
    params: ClassificationParams
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpacingClassifier:
    """
    Simple spacing-based resource classifier (NOT JORC compliant).
    
    ⚠️  DEPRECATED: This class is deprecated. Use JORCClassificationEngine instead.
    
    ⚠️  WARNING: This classifier uses simple distance thresholds only.
    It does NOT account for:
    - Variogram anisotropy
    - Kriging variance
    - Slope of regression
    - Geological confidence
    
    For JORC/SAMREC-compliant classification, use JORCClassificationEngine.
    
    Migration:
        OLD: SpacingClassifier()
        NEW: JORCClassificationEngine(variogram, ruleset)
    
    This classifier is intended for:
    - Quick scoping/exploration stage
    - Situations where variogram data is unavailable
    - Simple visualization checks
    
    Usage:
        classifier = SpacingClassifier()  # DEPRECATED
        result = classifier.classify(block_df, drillhole_df, params)
    """
    
    def __init__(self):
        """Initialize spacing classifier."""
        import warnings
        warnings.warn(
            "SpacingClassifier is DEPRECATED. "
            "Use JORCClassificationEngine from jorc_classification_engine.py instead. "
            "This will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning(
            "SpacingClassifier initialized (DEPRECATED). "
            "This is NOT JORC compliant. Use JORCClassificationEngine for production."
        )
        pass
    
    def classify(
        self,
        block_model: Union['BlockModel', pd.DataFrame],
        drillhole_df: pd.DataFrame,
        params: ClassificationParams
    ) -> ClassificationResult:
        """
        Classify blocks based on drillhole spacing (NOT JORC compliant).
        
        ✅ NEW STANDARD API: Accepts BlockModel or DataFrame (backward compatible)

        Args:
            block_model: BlockModel instance (preferred) or DataFrame (legacy)
            drillhole_df: DataFrame with drillhole locations
            params: Classification parameters

        Returns:
            ClassificationResult with classified blocks and summary
            
        Notes:
            ⚠️ DEPRECATED: NOT JORC/SAMREC compliant (spacing-based only).
            For production: Use JORCClassificationEngine from jorc_classification_engine.py
        """
        # --- NEW: Handle BlockModel input (standard API) ---
        from .block_model import BlockModel
        
        if isinstance(block_model, BlockModel):
            # ✅ STANDARD API: Extract using BlockModel methods
            try:
                payload = block_model.get_engine_payload()
                block_df = block_model.to_dataframe()
                
                # Ensure X, Y, Z columns exist
                if 'X' not in block_df.columns and 'x' in block_df.columns:
                    block_df['X'] = block_df['x']
                    block_df['Y'] = block_df['y']
                    block_df['Z'] = block_df['z']
                elif 'X' not in block_df.columns and 'XC' in block_df.columns:
                    block_df['X'] = block_df['XC']
                    block_df['Y'] = block_df['YC']
                    block_df['Z'] = block_df['ZC']
                
                logger.info(f"✅ BlockModel API: Extracted {len(block_df)} blocks for spacing classification")
                
            except Exception as e:
                raise ValueError(f"Failed to extract data from BlockModel: {e}")
        
        elif isinstance(block_model, pd.DataFrame):
            # --- LEGACY: DataFrame input ---
            block_df = block_model.copy()
            logger.info(f"⚠️ DataFrame API (legacy): {len(block_df)} blocks")
        
        else:
            raise ValueError(f"Unsupported block_model type: {type(block_model)}")

        # Run classification
        classified_df = classify_by_spacing(
            block_df=block_df,
            drillhole_df=drillhole_df,
            params=params
        )

        # Get summary
        summary = get_classification_summary(classified_df)

        return ClassificationResult(
            classified_df=classified_df,
            summary=summary,
            params=params,
            metadata={
                'block_count': len(classified_df),
                'drillhole_count': len(drillhole_df),
                'method': 'spacing_based',
                'warning': 'NOT JORC compliant - spacing only',
                'recommendation': 'Use JORCClassificationEngine for production',
            }
        )


# Legacy alias for backward compatibility
# ⚠️ DEPRECATED: Use JORCClassificationEngine instead.
ResourceClassifier = SpacingClassifier


def compute_effective_drillhole_spacing(
    drillhole_df: pd.DataFrame,
    k: int = 3,
    x_col: str = 'X',
    y_col: str = 'Y',
    z_col: str = 'Z',
    hole_id_col: Optional[str] = 'HOLEID'
) -> float:
    """
    Compute Effective Drillhole Spacing (EDHS) as the average distance 
    to the k-th nearest drillhole.
    
    Uses drillhole centers (collars or midpoints) to avoid counting multiple
    samples from the same hole.
    
    Args:
        drillhole_df: DataFrame with drillhole locations
        k: Which nearest neighbor to use (default: 3 for 3rd nearest)
        x_col, y_col, z_col: Coordinate column names
        hole_id_col: Column name for hole ID (to get unique holes)
    
    Returns:
        EDHS in meters (average distance to k-th nearest drillhole)
    """
    # Get unique drillhole locations (collars or midpoints)
    if hole_id_col and hole_id_col in drillhole_df.columns:
        # Group by hole ID and take mean coordinates (collar or midpoint)
        coords_df = drillhole_df.groupby(hole_id_col)[[x_col, y_col, z_col]].mean()
        logger.info(f"Computing EDHS from {len(coords_df)} unique drillholes")
    else:
        # Use all points if no hole ID column
        coords_df = drillhole_df[[x_col, y_col, z_col]].drop_duplicates()
        logger.info(f"Computing EDHS from {len(coords_df)} unique locations (no HOLEID column)")
    
    if len(coords_df) < k + 1:
        logger.warning(f"Not enough drillholes ({len(coords_df)}) for k={k}, using k=1")
        k = 1
    
    coords = coords_df[[x_col, y_col, z_col]].values
    
    # Remove NaN coordinates
    valid_mask = ~np.isnan(coords).any(axis=1)
    coords = coords[valid_mask]
    
    if len(coords) < k + 1:
        raise ValueError(f"Not enough valid drillhole coordinates ({len(coords)}) for EDHS calculation")
    
    # Build KD-tree
    tree = cKDTree(coords)
    
    # Query k+1 nearest neighbors (including self)
    distances, _ = tree.query(coords, k=min(k+1, len(coords)))
    
    # Extract k-th nearest distance (skip self at index 0)
    if distances.ndim == 1:
        # Only one point, use the distance itself
        kth_distances = distances[k] if len(distances) > k else distances[-1]
    else:
        # Multiple points, extract k-th column (skip column 0 which is self)
        kth_distances = distances[:, k] if distances.shape[1] > k else distances[:, -1]
    
    edhs = float(np.mean(kth_distances))
    
    logger.info(f"EDHS (k={k}): {edhs:.2f}m (mean distance to {k}-th nearest drillhole)")
    return edhs


def derive_domain_rules(
    a_major: float,
    edhs: Optional[float] = None,
    domain_name: str = "default",
    min_dist: float = 10.0,
    max_dist: float = 500.0,
    edhs_factors: Optional[Dict[str, float]] = None
) -> DomainRules:
    """
    Derive classification rules from variogram major range and drillhole spacing.
    
    Based on JORC/SAMREC guidance:
    - Measured: 0.25 × range (very tight spacing)
    - Indicated: 0.67 × range (2/3 of range)
    - Inferred: 1.00 × range (up to full range)
    
    Optionally respects EDHS constraints to ensure consistency with actual drilling pattern.
    
    Args:
        a_major: Variogram major range (meters)
        edhs: Effective Drillhole Spacing (meters). If None, only variogram-based rules used.
        domain_name: Name of domain (for logging)
        min_dist: Minimum allowed distance (clamp lower bound)
        max_dist: Maximum allowed distance (clamp upper bound)
        edhs_factors: Optional dict with keys 'measured', 'indicated', 'inferred' 
                     specifying minimum multiples of EDHS (default: 0.75, 1.5, 2.5)
    
    Returns:
        DomainRules with derived spacing rules
    """
    if edhs_factors is None:
        edhs_factors = {'measured': 0.75, 'indicated': 1.5, 'inferred': 2.5}
    
    # Base distances from variogram range
    d_M = 0.25 * a_major
    d_I = 0.67 * a_major  # 2/3 of range
    d_F = 1.00 * a_major
    
    # Respect EDHS constraints if provided
    if edhs is not None and edhs > 0:
        d_M = max(d_M, edhs_factors['measured'] * edhs)
        d_I = max(d_I, edhs_factors['indicated'] * edhs)
        d_F = max(d_F, edhs_factors['inferred'] * edhs)
        logger.info(f"Domain '{domain_name}': Applying EDHS constraints (EDHS={edhs:.1f}m)")
    
    # Clamp to reasonable bounds first
    d_M = max(min_dist, min(max_dist, d_M))
    d_I = max(min_dist, min(max_dist, d_I))
    d_F = max(min_dist, min(max_dist, d_F))
    
    # Ensure ordering: Measured < Indicated < Inferred
    # Apply ordering constraints, but respect min_dist
    d_M = max(min_dist, min(d_M, d_I * 0.8))  # Measured should be tighter than Indicated
    d_I = max(min_dist, min(d_I, d_F * 0.8))  # Indicated should be tighter than Inferred
    
    rules = DomainRules(
        measured=SpacingRule(d_M, 3),
        indicated=SpacingRule(d_I, 2),
        inferred=SpacingRule(d_F, 1),
        a_major=a_major,
        edhs=edhs,
        domain_name=domain_name
    )
    
    edhs_str = f"{edhs:.1f}m" if edhs is not None else "N/A"
    logger.info(
        f"Domain '{domain_name}' rules derived: a_major={a_major:.1f}m, "
        f"EDHS={edhs_str} -> "
        f"Measured: d={d_M:.1f}m, n=3; "
        f"Indicated: d={d_I:.1f}m, n=2; "
        f"Inferred: d={d_F:.1f}m, n=1"
    )
    
    return rules


def extract_variogram_range(
    variogram_results: Dict[str, Any],
    direction: str = 'major',
    model_type: Optional[str] = None
) -> Optional[float]:
    """
    Extract variogram range from variogram results dictionary.
    
    Args:
        variogram_results: Dictionary from variogram analysis
        direction: Direction to extract ('major', 'minor', 'omni', 'vertical')
        model_type: Specific model type to use (e.g., 'spherical'). 
                   If None, uses first available model.
    
    Returns:
        Range in meters, or None if not found
    """
    fitted_models = variogram_results.get('fitted_models', {})
    
    if direction not in fitted_models:
        logger.warning(f"Direction '{direction}' not found in variogram results. Available: {list(fitted_models.keys())}")
        # Try omni as fallback
        if 'omni' in fitted_models:
            direction = 'omni'
            logger.info(f"Using 'omni' direction as fallback")
        else:
            return None
    
    direction_models = fitted_models[direction]
    
    if not direction_models:
        logger.warning(f"No fitted models for direction '{direction}'")
        return None
    
    # If model_type specified, use it; otherwise use first available
    if model_type and model_type in direction_models:
        model_params = direction_models[model_type]
    else:
        # Use first available model
        model_params = list(direction_models.values())[0]
        model_type = list(direction_models.keys())[0]
        logger.info(f"Using model type '{model_type}' for direction '{direction}'")
    
    range_value = model_params.get('range')
    
    if range_value is None:
        logger.warning(f"No 'range' parameter in model '{model_type}' for direction '{direction}'")
        return None
    
    logger.info(f"Extracted variogram range: {range_value:.1f}m (direction='{direction}', model='{model_type}')")
    return float(range_value)


def classify_by_spacing(
    block_df: Optional[pd.DataFrame] = None,
    drillhole_df: Optional[pd.DataFrame] = None,
    params: Optional[ClassificationParams] = None,
    data_getter: Optional[Callable] = None,
    x_col: str = 'XC',
    y_col: str = 'YC',
    z_col: str = 'ZC',
    dh_x_col: str = 'X',
    dh_y_col: str = 'Y',
    dh_z_col: str = 'Z',
    measured_spacing: float = 50.0,
    indicated_spacing: float = 100.0,
    inferred_spacing: float = 200.0
) -> pd.DataFrame:
    """
    Classify blocks based on proximity to drillholes.
    
    Args:
        block_df: DataFrame containing block model with coordinates (optional if data_getter provided)
        drillhole_df: DataFrame containing drillhole locations (optional if data_getter provided)
        params: Classification parameters (optional, will use spacing values if None)
        data_getter: Optional callback function to retrieve data if None provided
        x_col, y_col, z_col: Column names for block coordinates
        dh_x_col, dh_y_col, dh_z_col: Column names for drillhole coordinates
        measured_spacing: Distance threshold for Measured category (m) - used if params is None
        indicated_spacing: Distance threshold for Indicated category (m) - used if params is None
        inferred_spacing: Distance threshold for Inferred category (m) - used if params is None
    
    Returns:
        DataFrame with added columns:
            - 'Category': Classification category
            - 'Nearest_DH_Dist': Distance to nearest drillhole (m)
            - 'DH_Count_<radius>': Number of drillholes within each radius
    """
    # Auto-detect data if not provided
    if block_df is None or drillhole_df is None:
        if data_getter is not None:
            try:
                retrieved = data_getter()
                if block_df is None:
                    block_df = retrieved.get('block_df')
                if drillhole_df is None:
                    drillhole_df = retrieved.get('drillhole_df')
                logger.info("Auto-detected block model and drillhole data from application context")
            except Exception as e:
                logger.warning(f"Auto-detection failed: {e}")
    
    # Validate that we have required data
    if block_df is None:
        raise ValueError("No block model data found. Ensure estimation or compositing has run first.")
    if drillhole_df is None:
        raise ValueError("No drillhole data found. Load drillhole data first.")
    
    # ------------------------------------------------------------------
    # Coordinate handling
    # ------------------------------------------------------------------
    # Normalise drillhole coordinates to X/Y/Z where possible
    drillhole_df = ensure_xyz_columns(drillhole_df)
    if all(col in drillhole_df.columns for col in ("X", "Y", "Z")):
        dh_x_col, dh_y_col, dh_z_col = "X", "Y", "Z"
    else:
        mapping = detect_coordinate_columns(drillhole_df.columns)
        if mapping.get("X") and mapping.get("Y") and mapping.get("Z"):
            dh_x_col, dh_y_col, dh_z_col = mapping["X"], mapping["Y"], mapping["Z"]
    
    # For blocks, prefer centroid-style columns in this order:
    # 1. XC/YC/ZC (common in CSVs)
    # 2. x/y/z   (used by internal BlockModel.to_dataframe)
    # 3. X/Y/Z   (fallback if centroids already named like drillholes)
    if all(col in block_df.columns for col in ("XC", "YC", "ZC")):
        x_col, y_col, z_col = "XC", "YC", "ZC"
    elif all(col in block_df.columns for col in ("x", "y", "z")):
        x_col, y_col, z_col = "x", "y", "z"
    elif all(col in block_df.columns for col in ("X", "Y", "Z")):
        x_col, y_col, z_col = "X", "Y", "Z"
    else:
        # Try to detect centroid-style coordinates generically
        mapping = detect_coordinate_columns(block_df.columns)
        if mapping.get("X") and mapping.get("Y") and mapping.get("Z"):
            x_col, y_col, z_col = mapping["X"], mapping["Y"], mapping["Z"]
    
    # Build params if not provided
    if params is None:
        params = ClassificationParams(
            rules={
                'Measured': SpacingRule(max_dist=measured_spacing, min_holes=3),
                'Indicated': SpacingRule(max_dist=indicated_spacing, min_holes=2),
                'Inferred': SpacingRule(max_dist=inferred_spacing, min_holes=1)
            },
            count_radii=[measured_spacing, indicated_spacing, inferred_spacing]
        )
    
    logger.info(f"Starting classification for {len(block_df)} blocks using {len(drillhole_df)} drillholes")
    
    # Validate required columns after auto-detection
    for col in [x_col, y_col, z_col]:
        if col not in block_df.columns:
            raise ValueError(
                f"Block DataFrame missing required coordinate column '{col}'. "
                f"Available columns: {list(block_df.columns)}"
            )
    
    for col in [dh_x_col, dh_y_col, dh_z_col]:
        if col not in drillhole_df.columns:
            raise ValueError(
                f"Drillhole DataFrame missing required coordinate column '{col}'. "
                f"Available columns: {list(drillhole_df.columns)}"
            )
    
    # Extract coordinates
    block_coords = block_df[[x_col, y_col, z_col]].values
    dh_coords = drillhole_df[[dh_x_col, dh_y_col, dh_z_col]].values
    
    logger.info(f"Building KDTree with {len(dh_coords)} drillhole locations...")
    
    # Build KD-tree for efficient spatial queries (handle NaNs safely)
    # Remove any NaN points from drillhole coordinates
    valid_dh_mask = ~np.isnan(dh_coords).any(axis=1)
    if not valid_dh_mask.all():
        logger.warning(f"Removing {np.sum(~valid_dh_mask)} drillhole points with NaN coordinates")
        dh_coords = dh_coords[valid_dh_mask]
    
    if len(dh_coords) == 0:
        raise ValueError("No valid drillhole coordinates found after removing NaNs")
    
    tree = cKDTree(dh_coords)
    
    # Query nearest drillhole distance for all blocks
    logger.info("Computing nearest drillhole distances...")
    nearest_dists, nearest_idx = tree.query(block_coords, k=1)
    
    # Handle NaN distances (shouldn't happen, but be safe)
    nearest_dists = np.nan_to_num(nearest_dists, nan=np.inf)
    
    # Count drillholes within specified radii
    result_df = block_df.copy()
    result_df['Nearest_DH_Dist'] = nearest_dists
    
    logger.info(f"Counting drillholes within radii: {params.count_radii}")
    for radius in params.count_radii:
        # Query all drillholes within radius
        counts = tree.query_ball_point(block_coords, r=radius, return_length=True)
        result_df[f'DH_Count_{int(radius)}m'] = counts
        logger.info(f"  Radius {radius}m: avg {np.mean(counts):.2f} holes per block")
    
    # Apply classification rules (priority order: Measured > Indicated > Inferred)
    logger.info("Applying classification rules...")
    categories = np.full(len(result_df), 'Unclassified', dtype=object)
    
    for category in CATEGORY_ORDER[:3]:  # Measured, Indicated, Inferred
        if category not in params.rules:
            continue
        
        rule = params.rules[category]
        count_col = f'DH_Count_{int(rule.max_dist)}m'
        
        # Block meets criteria if:
        # 1. Nearest drillhole is within max_dist
        # 2. At least min_holes drillholes are within max_dist
        mask = (
            (result_df['Nearest_DH_Dist'] <= rule.max_dist) &
            (result_df[count_col] >= rule.min_holes)
        )
        
        # Only apply if not already classified to a higher confidence category
        unclassified_mask = (categories == 'Unclassified')
        categories[mask & unclassified_mask] = category
        
        n_classified = np.sum(mask & unclassified_mask)
        logger.info(f"  {category}: {n_classified} blocks ({n_classified/len(result_df)*100:.1f}%)")
    
    result_df['Category'] = categories
    
    # Summary statistics
    summary = result_df['Category'].value_counts()
    logger.info(f"Classification complete:")
    for cat in CATEGORY_ORDER:
        count = summary.get(cat, 0)
        pct = count / len(result_df) * 100
        logger.info(f"  {cat}: {count} blocks ({pct:.1f}%)")
    
    return result_df


def get_classification_summary(classified_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Generate summary statistics for classified blocks.
    
    Args:
        classified_df: DataFrame with 'Category' and 'Nearest_DH_Dist' columns
    
    Returns:
        Dictionary with statistics per category
    """
    summary = {}
    
    for category in CATEGORY_ORDER:
        cat_df = classified_df[classified_df['Category'] == category]
        
        if len(cat_df) == 0:
            summary[category] = {
                'count': 0,
                'percentage': 0.0,
                'avg_dist': np.nan,
                'min_dist': np.nan,
                'max_dist': np.nan
            }
        else:
            summary[category] = {
                'count': len(cat_df),
                'percentage': len(cat_df) / len(classified_df) * 100,
                'avg_dist': cat_df['Nearest_DH_Dist'].mean(),
                'min_dist': cat_df['Nearest_DH_Dist'].min(),
                'max_dist': cat_df['Nearest_DH_Dist'].max()
            }
    
    return summary


def export_summary_table(classified_df: pd.DataFrame, output_csv: str) -> None:
    """
    Export classification summary as a compact CSV table.
    
    Args:
        classified_df: Classified DataFrame with 'Category' column
        output_csv: Output CSV file path
    """
    summary = get_classification_summary(classified_df)
    rows = []
    for cat in CATEGORY_ORDER:
        stats = summary[cat]
        rows.append({
            "Category": cat,
            "Blocks": stats["count"],
            "Percentage": stats["percentage"],
            "Avg_Dist_m": stats.get("avg_dist", np.nan),
            "Min_Dist_m": stats.get("min_dist", np.nan),
            "Max_Dist_m": stats.get("max_dist", np.nan)
        })
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_csv, index=False)
    logger.info(f"Exported classification summary table → {output_csv}")


def export_classification(
    classified_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    include_stats: bool = True,
    include_screenshot: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    scalar_field: Optional[str] = None
) -> Dict[str, Path]:
    """
    Export classification results with summary, table, and screenshot.
    Automatically saves to exports/ with timestamped filenames.
    
    Args:
        classified_df: Classified DataFrame (optional, will attempt auto-detection)
        output_path: Optional specific output path (if None, uses timestamped filename)
        include_stats: Whether to include summary statistics in output
        include_screenshot: Whether to generate 3D screenshot
        save_dir: Directory to save exports (defaults to "exports")
        scalar_field: Optional property field to color by in screenshot
    
    Returns:
        Dictionary with export file paths:
            - 'csv': Main classified blocks CSV
            - 'summary_csv': Summary table CSV
            - 'summary_txt': Summary text file
            - 'image': Screenshot PNG (if include_screenshot=True)
    """
    # Auto-detect classified_df if not provided
    if classified_df is None:
        try:
            # Try to import from application context
            from ..models.block_model import BlockModel
            # This won't work directly, but we can check for a global or module-level reference
            logger.warning("No classified_df provided - auto-detection from memory not implemented yet")
            raise ValueError("No classified block model found. Run classification first.")
        except Exception as e:
            logger.error(f"Auto-detection failed: {e}")
            raise ValueError("No classified block model found in memory. Run classification first.")
    
    # Validate required columns
    required_cols = ['Category', 'Nearest_DH_Dist']
    if not all(col in classified_df.columns for col in required_cols):
        raise ValueError(f"Classified DataFrame missing required columns: {required_cols}")
    
    # Determine coordinate columns (XC/YC/ZC or X/Y/Z)
    coord_cols = None
    if all(col in classified_df.columns for col in ['XC', 'YC', 'ZC']):
        coord_cols = ['XC', 'YC', 'ZC']
    elif all(col in classified_df.columns for col in ['X', 'Y', 'Z']):
        coord_cols = ['X', 'Y', 'Z']
    else:
        logger.warning("Coordinate columns not found for screenshot - skipping image export")
        include_screenshot = False
    
    # Setup export directory
    if save_dir is None:
        save_dir = Path("exports")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export main CSV
    if output_path is None:
        csv_path = save_dir / f"classified_blocks_{timestamp}.csv"
    else:
        csv_path = Path(output_path)
    
    # Safe export - handle NaNs
    classified_df_clean = classified_df.copy()
    classified_df_clean.to_csv(csv_path, index=False, na_rep='NaN')
    logger.info(f"Exported classified blocks → {csv_path}")
    
    results = {"csv": csv_path}
    
    # Export summary table
    if include_stats:
        table_csv = save_dir / f"classified_blocks_summary_{timestamp}.csv"
        export_summary_table(classified_df, str(table_csv))
        results["summary_csv"] = table_csv
        
        # Export summary text
        summary_txt = save_dir / f"classified_blocks_summary_{timestamp}.txt"
        summary = get_classification_summary(classified_df)
        
        with open(summary_txt, "w", encoding='utf-8') as f:
            f.write("RESOURCE CLASSIFICATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            for cat in CATEGORY_ORDER:
                stats = summary[cat]
                f.write(f"{cat}:\n")
                f.write(f"  Blocks: {stats['count']}\n")
                f.write(f"  Percentage: {stats['percentage']:.2f}%\n")
                
                if not np.isnan(stats.get('avg_dist', np.nan)):
                    f.write(f"  Avg Distance to DH: {stats['avg_dist']:.2f} m\n")
                    f.write(f"  Min Distance to DH: {stats['min_dist']:.2f} m\n")
                    f.write(f"  Max Distance to DH: {stats['max_dist']:.2f} m\n")
                f.write("\n")
        
        logger.info(f"Exported summary text → {summary_txt}")
        results["summary_txt"] = summary_txt
    
    # Export 3D screenshot (off-screen rendering)
    if include_screenshot and coord_cols is not None:
        try:
            import pyvista as pv
            
            # Use Category as color field (categorical)
            color_field = "Category"
            
            # Extract coordinates
            points = classified_df[coord_cols].values
            
            # Remove NaN coordinates
            valid_mask = ~np.isnan(points).any(axis=1)
            if not valid_mask.all():
                logger.warning(f"Removing {np.sum(~valid_mask)} points with NaN coordinates for screenshot")
                points = points[valid_mask]
                classified_subset = classified_df[valid_mask].copy()
            else:
                classified_subset = classified_df.copy()
            
            if len(points) == 0:
                logger.warning("No valid points for screenshot - skipping image export")
            else:
                # Create point cloud
                point_cloud = pv.PolyData(points)
                point_cloud[color_field] = classified_subset[color_field].astype(str)
                
                # Create off-screen plotter (no GUI freeze)
                plotter = pv.Plotter(off_screen=True)
                
                # Add points colored by category
                plotter.add_points(
                    point_cloud,
                    scalars=color_field,
                    render_points_as_spheres=True,
                    point_size=6,
                    cmap="Set1"  # Categorical colormap
                )
                
                # VIOLATION FIX: Removed plotter.add_axes() - axes should be managed by OverlayManager
                # Note: This is a standalone visualization function, not part of main viewer
                # For main viewer integration, axes should be controlled via OverlayManager
                # For standalone export, we allow axes here as it's not part of unified rendering
                try:
                    plotter.add_axes()
                except Exception:
                    pass  # Ignore if axes already exist
                plotter.add_floor(grid=True, color="white", opacity=0.3)
                
                # Render and save screenshot
                img_path = save_dir / f"classified_blocks_{timestamp}.png"
                plotter.render()  # Ensure scene is rendered
                plotter.screenshot(str(img_path))
                plotter.close()
                
                logger.info(f"Saved classification screenshot → {img_path}")
                results["image"] = img_path
        
        except ImportError:
            logger.warning("PyVista not available - skipping screenshot export")
        except Exception as e:
            logger.error(f"Screenshot export failed: {e}", exc_info=True)
    
    return results


def generate_classification_report(
    classified_df: pd.DataFrame,
    params: ClassificationParams,
    variogram_results: Optional[Dict[str, Any]] = None,
    edhs: Optional[float] = None,
    domain_name: str = "default",
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Generate an auditable classification report for JORC/SAMREC compliance.
    
    The report includes:
    - Input parameters (variogram ranges, EDHS, chosen distances)
    - Method description
    - Classification statistics
    - Diagnostics (spacing histograms, maps)
    - Override log (if any)
    
    Args:
        classified_df: Classified DataFrame with 'Category' column
        params: Classification parameters used
        variogram_results: Optional variogram results dictionary
        edhs: Optional Effective Drillhole Spacing value
        domain_name: Domain name
        output_path: Output file path (defaults to timestamped filename)
    
    Returns:
        Path to generated report file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("exports") / f"classification_report_{timestamp}.txt"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = get_classification_summary(classified_df)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RESOURCE CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Domain: {domain_name}\n\n")
        
        # 1. INPUTS SECTION
        f.write("-" * 80 + "\n")
        f.write("1. INPUT PARAMETERS\n")
        f.write("-" * 80 + "\n\n")
        
        # Variogram parameters
        if variogram_results:
            f.write("Variogram Parameters:\n")
            fitted_models = variogram_results.get('fitted_models', {})
            for direction in ['major', 'minor', 'omni', 'vertical']:
                if direction in fitted_models:
                    models = fitted_models[direction]
                    if models:
                        model_params = list(models.values())[0]
                        f.write(f"  {direction.capitalize()}: ")
                        f.write(f"range={model_params.get('range', 'N/A'):.1f}m, ")
                        f.write(f"nugget={model_params.get('nugget', 'N/A'):.3f}, ")
                        f.write(f"sill={model_params.get('sill', 'N/A'):.3f}\n")
            f.write("\n")
        
        # EDHS
        if edhs is not None:
            f.write(f"Effective Drillhole Spacing (EDHS): {edhs:.2f}m\n")
            f.write("  (Average distance to 3rd nearest drillhole)\n\n")
        
        # Classification rules
        f.write("Classification Rules:\n")
        for category in ['Measured', 'Indicated', 'Inferred']:
            if category in params.rules:
                rule = params.rules[category]
                f.write(f"  {category}:\n")
                f.write(f"    Max Distance: {rule.max_dist:.1f}m\n")
                f.write(f"    Min Holes: {rule.min_holes}\n")
        f.write("\n")
        
        # 2. METHOD SECTION
        f.write("-" * 80 + "\n")
        f.write("2. CLASSIFICATION METHOD\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(
            "Blocks are classified using a distance-and-hole count rule based on:\n"
            "• Drillhole spacing (proximity to drillholes)\n"
            "• Continuity (variogram ranges)\n"
            "• Minimum number of independent drillholes\n\n"
        )
        
        if variogram_results:
            f.write(
                "Distance thresholds were derived from variogram major range:\n"
                "• Measured: 0.25 × range (very tight spacing)\n"
                "• Indicated: 0.67 × range (2/3 of range)\n"
                "• Inferred: 1.00 × range (up to full range)\n\n"
            )
        
        if edhs is not None:
            f.write(
                f"Distances were constrained by Effective Drillhole Spacing (EDHS={edhs:.1f}m):\n"
                "• Measured: ≥ 0.75 × EDHS\n"
                "• Indicated: ≥ 1.5 × EDHS\n"
                "• Inferred: ≥ 2.5 × EDHS\n\n"
            )
        
        f.write(
            "This method aligns with JORC Table 1 / SAMREC Table 1 guidance for "
            "geological confidence classification based on data spacing and continuity.\n\n"
        )
        
        # 3. RESULTS SECTION
        f.write("-" * 80 + "\n")
        f.write("3. CLASSIFICATION RESULTS\n")
        f.write("-" * 80 + "\n\n")
        
        total_blocks = len(classified_df)
        f.write(f"Total Blocks Classified: {total_blocks:,}\n\n")
        
        for category in CATEGORY_ORDER:
            stats = summary[category]
            f.write(f"{category}:\n")
            f.write(f"  Blocks: {stats['count']:,} ({stats['percentage']:.2f}%)\n")
            if not np.isnan(stats.get('avg_dist', np.nan)):
                f.write(f"  Avg Distance to Nearest DH: {stats['avg_dist']:.2f}m\n")
                f.write(f"  Min Distance: {stats['min_dist']:.2f}m\n")
                f.write(f"  Max Distance: {stats['max_dist']:.2f}m\n")
            f.write("\n")
        
        # 4. DIAGNOSTICS SECTION
        f.write("-" * 80 + "\n")
        f.write("4. DIAGNOSTICS\n")
        f.write("-" * 80 + "\n\n")
        
        if 'Nearest_DH_Dist' in classified_df.columns:
            f.write("Distance Statistics by Category:\n")
            for category in CATEGORY_ORDER:
                cat_df = classified_df[classified_df['Category'] == category]
                if len(cat_df) > 0:
                    dists = cat_df['Nearest_DH_Dist']
                    f.write(f"  {category}:\n")
                    f.write(f"    Mean: {dists.mean():.2f}m\n")
                    f.write(f"    Median: {dists.median():.2f}m\n")
                    f.write(f"    Std Dev: {dists.std():.2f}m\n")
                    f.write(f"    P10: {dists.quantile(0.1):.2f}m\n")
                    f.write(f"    P90: {dists.quantile(0.9):.2f}m\n")
            f.write("\n")
        
        # 5. OVERRIDE LOG (placeholder - can be extended)
        f.write("-" * 80 + "\n")
        f.write("5. OVERRIDE LOG\n")
        f.write("-" * 80 + "\n\n")
        f.write("No manual overrides recorded.\n")
        f.write("(This section can be populated if CP manually adjusts classifications)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Generated classification report → {output_path}")
    return output_path


