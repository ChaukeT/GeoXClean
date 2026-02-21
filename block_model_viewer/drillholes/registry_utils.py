from __future__ import annotations

from typing import Dict, Any, Optional, List, Set
import logging

import numpy as np
import pandas as pd

from .datamodel import (
    AssayInterval,
    Collar,
    DrillholeDatabase,
    LithologyInterval,
    SurveyInterval,
)

logger = logging.getLogger(__name__)


def build_database_from_registry(data: Dict[str, Any]) -> DrillholeDatabase:
    """
    Construct a DrillholeDatabase from registry tables.
    
    Args:
        data: Dictionary containing drillhole data with keys:
            - "collars": DataFrame with collar information
            - "surveys": DataFrame with survey information (optional)
            - "assays": DataFrame with assay information (optional)
            - "lithology": DataFrame with lithology information (optional)
    
    Returns:
        DrillholeDatabase instance
        
    Raises:
        ValueError: If data is not a dictionary or if required data is missing
        TypeError: If data values are not DataFrames when present
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary, got {type(data).__name__}")
    
    logger.debug(f"Building database from registry. Available keys: {list(data.keys())}")
    
    db = DrillholeDatabase()

    try:
        collar_df = _extract_dataframe(data, "collars")
        survey_df = _extract_dataframe(data, "surveys")
        assay_df = _extract_dataframe(data, "assays")
        lithology_df = _extract_dataframe(data, "lithology")

        # Collars are required - check if we have any data
        if collar_df is None or collar_df.empty:
            available_keys = [k for k in data.keys() if isinstance(data.get(k), pd.DataFrame)]
            raise ValueError(
                f"No collar data found. Collars are required to build a drillhole database.\n"
                f"Available DataFrame keys: {available_keys}"
            )
        
        logger.debug(f"Found collars: {len(collar_df)} rows, columns: {list(collar_df.columns)}")
        
        _populate_collars(db, collar_df)
        _populate_surveys(db, survey_df)
        _populate_assays(db, assay_df)
        _populate_lithology(db, lithology_df)
        
        # Verify we created at least some collars
        if db.collars.empty:
            raise ValueError(
                f"Failed to populate collars. Check that collar data has required columns.\n"
                f"Found columns: {list(collar_df.columns)}\n"
                f"Expected one of: holeid/hole_id/hole, x/easting, y/northing, z/elevation/rl"
            )
        
        logger.info(
            f"Built database: {len(db.collars)} collars, {len(db.surveys)} surveys, "
            f"{len(db.assays)} assays, {len(db.lithology)} lithology intervals"
        )
    except ValueError:
        # Re-raise ValueError as-is (these are our validation errors)
        raise
    except Exception as e:
        logger.error(f"Error building database from registry: {e}", exc_info=True)
        raise ValueError(f"Failed to build drillhole database: {e}") from e

    return db


def composite_assays_from_df(df: Optional[pd.DataFrame]) -> List[AssayInterval]:
    """Build composited assays list from composites DataFrame."""
    if df is None or df.empty:
        return []

    hole_col = _find_column(df, {"holeid", "hole_id", "hole"})
    from_col = _find_column(df, {"from", "depth_from"})
    to_col = _find_column(df, {"to", "depth_to"})
    if not hole_col or not from_col or not to_col:
        return []

    exclude = {hole_col, from_col, to_col}
    value_cols = [col for col in df.columns if col not in exclude]

    assays = []
    for _, row in df.iterrows():
        hole_id_val = row.get(hole_col)
        if pd.isna(hole_id_val):
            continue
        start = _safe_float(row.get(from_col))
        end = _safe_float(row.get(to_col))
        if start is None or end is None:
            continue

        values = {}
        for col in value_cols:
            val = _safe_float(row.get(col))
            if val is not None:
                values[col] = val

        if not values:
            continue

        assays.append(
            AssayInterval(
                hole_id=str(hole_id_val),
                depth_from=start,
                depth_to=end,
                values=values,
            )
        )

    return assays


def composite_lithology_from_df(df: Optional[pd.DataFrame]) -> List[LithologyInterval]:
    """Build lithology intervals from composites DataFrame."""
    if df is None or df.empty:
        return []

    hole_col = _find_column(df, {"holeid", "hole_id", "hole"})
    from_col = _find_column(df, {"from", "depth_from"})
    to_col = _find_column(df, {"to", "depth_to"})
    code_col = _find_column(df, {"lithology", "lith", "code", "lith_code"})
    if not hole_col or not from_col or not to_col:
        return []

    lithos = []
    for _, row in df.iterrows():
        hole_id_val = row.get(hole_col)
        start = _safe_float(row.get(from_col))
        end = _safe_float(row.get(to_col))
        if pd.isna(hole_id_val) or start is None or end is None:
            continue
        lithos.append(
            LithologyInterval(
                hole_id=str(hole_id_val),
                depth_from=start,
                depth_to=end,
                lith_code=str(row.get(code_col)) if code_col and not pd.isna(row.get(code_col)) else "Composite",
            )
        )
    return lithos


def _extract_dataframe(data: Dict[str, Any], key: str) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame copy if available.
    
    Args:
        data: Dictionary containing the data
        key: Key to look up in the dictionary
        
    Returns:
        DataFrame copy if available, None otherwise
        
    Raises:
        TypeError: If the value exists but is not a DataFrame
    """
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    # Log warning but don't crash - some keys might have non-DataFrame values
    logger.warning(f"Expected DataFrame for key '{key}', got {type(value).__name__}. Skipping.")
    return None


def _find_column(df: pd.DataFrame, candidates: Set[str]) -> Optional[str]:
    """Case-insensitive column lookup."""
    lower_candidates = {name.strip().lower() for name in candidates}
    for col in df.columns:
        if col and col.strip().lower() in lower_candidates:
            return col
    return None


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float safely."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _populate_collars(db: DrillholeDatabase, df: Optional[pd.DataFrame]) -> None:
    if df is None:
        return

    hole_col = _find_column(df, {"holeid", "hole_id", "hole"})
    x_col = _find_column(df, {"x", "easting"})
    y_col = _find_column(df, {"y", "northing"})
    z_col = _find_column(df, {"z", "elevation", "rl"})
    length_col = _find_column(df, {"depth", "totaldepth", "length"})

    if not hole_col:
        return

    # Build list of dictionaries (pandas DataFrame.append() was removed)
    rows = []
    for _, row in df.iterrows():
        hole_id_val = row.get(hole_col)
        if pd.isna(hole_id_val):
            continue
        hole_id = str(hole_id_val)

        rows.append({
            'hole_id': hole_id,
            'x': _safe_float(row.get(x_col)) or 0.0,
            'y': _safe_float(row.get(y_col)) or 0.0,
            'z': _safe_float(row.get(z_col)) or 0.0,
            'azimuth': None,  # Not in source data
            'dip': None,  # Not in source data
            'length': _safe_float(row.get(length_col)),
        })
    
    if rows:
        # Create DataFrame from list and concatenate with existing
        new_df = pd.DataFrame(rows)
        if db.collars.empty:
            db.collars = new_df
        else:
            db.collars = pd.concat([db.collars, new_df], ignore_index=True)


def _populate_surveys(db: DrillholeDatabase, df: Optional[pd.DataFrame]) -> None:
    if df is None:
        return

    hole_col = _find_column(df, {"holeid", "hole_id", "hole"})
    depth_col = _find_column(df, {"depth", "md", "distance", "depth_from"})
    az_col = _find_column(df, {"azimuth", "azi", "az"})
    dip_col = _find_column(df, {"dip", "inclination"})

    if not hole_col or not depth_col:
        return

    # Build list of dictionaries (pandas DataFrame.append() was removed)
    rows = []
    for _, row in df.iterrows():
        hole_id_val = row.get(hole_col)
        depth_val = _safe_float(row.get(depth_col))
        if pd.isna(hole_id_val) or depth_val is None:
            continue

        rows.append({
            'hole_id': str(hole_id_val),
            'depth_from': depth_val,
            'depth_to': depth_val,
            'azimuth': _safe_float(row.get(az_col)) or 0.0,
            'dip': _safe_float(row.get(dip_col)) or -90.0,
        })
    
    if rows:
        # Create DataFrame from list and concatenate with existing
        new_df = pd.DataFrame(rows)
        if db.surveys.empty:
            db.surveys = new_df
        else:
            db.surveys = pd.concat([db.surveys, new_df], ignore_index=True)


def _populate_assays(db: DrillholeDatabase, df: Optional[pd.DataFrame]) -> None:
    if df is None:
        return

    hole_col = _find_column(df, {"holeid", "hole_id", "hole"})
    from_col = _find_column(df, {"from", "depth_from", "start"})
    to_col = _find_column(df, {"to", "depth_to", "end"})

    if not hole_col or not from_col or not to_col:
        return

    # Check for lithology column in assays DataFrame
    lith_col = _find_column(df, {"lithology", "lith", "lith_code", "code"})
    
    exclude = {hole_col, from_col, to_col}
    if lith_col:
        exclude.add(lith_col)  # Exclude lithology from assay values
    
    value_cols = [col for col in df.columns if col not in exclude]

    # Build list of dictionaries (pandas DataFrame.append() was removed)
    rows = []
    lithology_rows = []
    for _, row in df.iterrows():
        hole_id_val = row.get(hole_col)
        depth_from = _safe_float(row.get(from_col))
        depth_to = _safe_float(row.get(to_col))
        if pd.isna(hole_id_val) or depth_from is None or depth_to is None:
            continue

        # Build row dict with required columns and all value columns
        row_dict = {
            'hole_id': str(hole_id_val),
            'depth_from': depth_from,
            'depth_to': depth_to,
        }
        
        # Add all value columns (excluding lithology)
        has_values = False
        for col in value_cols:
            val = _safe_float(row.get(col))
            if val is not None:
                row_dict[col] = val
                has_values = True

        if has_values:
            rows.append(row_dict)
        
        # Extract lithology if present
        if lith_col:
            lith_code = row.get(lith_col)
            if lith_code is not None and not pd.isna(lith_code):
                lithology_rows.append({
                    'hole_id': str(hole_id_val),
                    'depth_from': depth_from,
                    'depth_to': depth_to,
                    'lith_code': str(lith_code),
                })
    
    if rows:
        # Create DataFrame from list and concatenate with existing
        new_df = pd.DataFrame(rows)
        if db.assays.empty:
            db.assays = new_df
        else:
            db.assays = pd.concat([db.assays, new_df], ignore_index=True)
    
    # Populate lithology from assays DataFrame if lithology column exists
    if lithology_rows:
        lith_df = pd.DataFrame(lithology_rows)
        if db.lithology.empty:
            db.lithology = lith_df
        else:
            db.lithology = pd.concat([db.lithology, lith_df], ignore_index=True)


def _populate_lithology(db: DrillholeDatabase, df: Optional[pd.DataFrame]) -> None:
    if df is None:
        return

    hole_col = _find_column(df, {"holeid", "hole_id", "hole"})
    from_col = _find_column(df, {"from", "depth_from", "start"})
    to_col = _find_column(df, {"to", "depth_to", "end"})
    code_col = _find_column(df, {"code", "lithology", "lith_code", "lith"})

    if not hole_col or not from_col or not to_col:
        return

    # Build list of dictionaries (pandas DataFrame.append() was removed)
    rows = []
    for _, row in df.iterrows():
        hole_id_val = row.get(hole_col)
        depth_from = _safe_float(row.get(from_col))
        depth_to = _safe_float(row.get(to_col))
        if pd.isna(hole_id_val) or depth_from is None or depth_to is None:
            continue
        lith_code = str(row.get(code_col)) if code_col and not pd.isna(row.get(code_col)) else "Unknown"
        
        rows.append({
            'hole_id': str(hole_id_val),
            'depth_from': depth_from,
            'depth_to': depth_to,
            'lith_code': lith_code,
        })
    
    if rows:
        # Create DataFrame from list and concatenate with existing
        new_df = pd.DataFrame(rows)
        if db.lithology.empty:
            db.lithology = new_df
        else:
            db.lithology = pd.concat([db.lithology, new_df], ignore_index=True)

