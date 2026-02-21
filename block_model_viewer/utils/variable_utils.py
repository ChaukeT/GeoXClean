"""
Variable Utilities for Geostatistics Panels

Centralized variable validation and selection logic for kriging,
simulation, and other geostatistical workflows.

This module eliminates duplicate variable handling code across panels
and provides a single source of truth for:
- Variable validation
- Auto-selection of grade columns
- Filtering coordinate columns
- Error message generation
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns to exclude from variable selection (case-insensitive)
COORDINATE_COLUMNS = frozenset({
    'X', 'Y', 'Z', 'MIDX', 'MIDY', 'MIDZ',
    'FROM', 'TO', 'LENGTH', 'DEPTH',
    'HOLEID', 'HOLE_ID', 'BHID', 'DH_ID',
    'AZIMUTH', 'DIP', 'INCLINATION',
    'SAMPLE_ID', 'SAMPLEID', 'INDEX'
})


@dataclass
class VariableValidationResult:
    """Result of variable validation."""
    is_valid: bool
    variable: Optional[str]
    error_message: Optional[str]
    available_variables: List[str]


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Get all numeric columns from a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of numeric column names
    """
    if df is None or df.empty:
        return []
    
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_grade_columns(df: pd.DataFrame, exclude_coords: bool = True) -> List[str]:
    """
    Get grade/property columns suitable for geostatistical analysis.
    
    Filters out coordinate columns and other non-grade columns.
    
    Args:
        df: Input DataFrame with drillhole/sample data
        exclude_coords: Whether to exclude coordinate columns (default True)
        
    Returns:
        Sorted list of grade column names
    """
    if df is None or df.empty:
        return []
    
    numeric_cols = get_numeric_columns(df)
    
    if not exclude_coords:
        return sorted(numeric_cols)
    
    # Filter out coordinate and metadata columns
    grade_cols = [
        col for col in numeric_cols
        if col.upper() not in COORDINATE_COLUMNS
    ]
    
    return sorted(grade_cols)


def validate_variable(
    variable: Optional[str],
    df: pd.DataFrame,
    context: str = "analysis"
) -> VariableValidationResult:
    """
    Validate that a variable exists and is suitable for geostatistical analysis.
    
    Args:
        variable: The variable name to validate
        df: DataFrame containing the data
        context: Context string for error messages (e.g., "Ordinary Kriging")
        
    Returns:
        VariableValidationResult with validation status and details
    """
    available = get_grade_columns(df)
    
    # Check if DataFrame is valid
    if df is None or df.empty:
        return VariableValidationResult(
            is_valid=False,
            variable=None,
            error_message=f"No data available for {context}. Please load drillhole/composite data first.",
            available_variables=[]
        )
    
    # Check if any variables are available
    if not available:
        return VariableValidationResult(
            is_valid=False,
            variable=None,
            error_message=f"No numeric grade columns found in data for {context}. "
                         f"Ensure your data contains numeric columns other than coordinates.",
            available_variables=[]
        )
    
    # Check if variable is specified
    if not variable or not variable.strip():
        return VariableValidationResult(
            is_valid=False,
            variable=None,
            error_message=f"No variable selected for {context}. Please select a grade column.",
            available_variables=available
        )
    
    variable = variable.strip()
    
    # Check if variable exists in DataFrame
    if variable not in df.columns:
        return VariableValidationResult(
            is_valid=False,
            variable=variable,
            error_message=f"Variable '{variable}' not found in data. "
                         f"Available variables: {', '.join(available[:10])}",
            available_variables=available
        )
    
    # Check if variable is numeric
    if not np.issubdtype(df[variable].dtype, np.number):
        return VariableValidationResult(
            is_valid=False,
            variable=variable,
            error_message=f"Variable '{variable}' is not numeric. "
                         f"Please select a numeric grade column.",
            available_variables=available
        )
    
    # Check for all-NaN
    if df[variable].isna().all():
        return VariableValidationResult(
            is_valid=False,
            variable=variable,
            error_message=f"Variable '{variable}' contains only missing values.",
            available_variables=available
        )
    
    return VariableValidationResult(
        is_valid=True,
        variable=variable,
        error_message=None,
        available_variables=available
    )


def auto_select_variable(
    df: pd.DataFrame,
    preferred: Optional[str] = None,
    fallback_to_first: bool = True
) -> Tuple[Optional[str], List[str]]:
    """
    Automatically select a suitable variable from the DataFrame.
    
    Priority:
    1. If preferred is specified and valid, use it
    2. Common grade column names (AU, AG, CU, FE, ZN, PB, etc.)
    3. First available grade column (if fallback_to_first=True)
    
    Args:
        df: DataFrame with drillhole/sample data
        preferred: Preferred variable name (optional)
        fallback_to_first: Whether to fall back to first available column
        
    Returns:
        Tuple of (selected_variable, available_variables)
    """
    available = get_grade_columns(df)
    
    if not available:
        return None, []
    
    # If preferred is specified and valid, use it
    if preferred and preferred in available:
        return preferred, available
    
    # Try common grade column names
    common_grades = ['AU', 'AG', 'CU', 'FE', 'ZN', 'PB', 'NI', 'CO', 
                     'GRADE', 'VALUE', 'ASSAY']
    
    for grade in common_grades:
        # Case-insensitive match
        matches = [col for col in available if col.upper() == grade]
        if matches:
            logger.info(f"Auto-selected variable '{matches[0]}' (common grade column)")
            return matches[0], available
    
    # Fallback to first available
    if fallback_to_first and available:
        logger.info(f"Auto-selected first available variable '{available[0]}'")
        return available[0], available
    
    return None, available


def populate_variable_combo(
    combo_box,
    df: pd.DataFrame,
    current_selection: Optional[str] = None,
    block_signals: bool = True
) -> Optional[str]:
    """
    Populate a QComboBox with available grade columns.
    
    Args:
        combo_box: QComboBox widget to populate
        df: DataFrame with data
        current_selection: Current selection to preserve (optional)
        block_signals: Whether to block signals during update
        
    Returns:
        The selected variable after population, or None
    """
    if combo_box is None:
        return None
    
    available = get_grade_columns(df)
    
    if block_signals:
        combo_box.blockSignals(True)
    
    try:
        combo_box.clear()
        
        if not available:
            return None
        
        combo_box.addItems(available)
        
        # Restore or auto-select
        if current_selection and current_selection in available:
            combo_box.setCurrentText(current_selection)
            return current_selection
        elif available:
            # Auto-select
            selected, _ = auto_select_variable(df)
            if selected:
                combo_box.setCurrentText(selected)
                return selected
            else:
                combo_box.setCurrentIndex(0)
                return combo_box.currentText()
    finally:
        if block_signals:
            combo_box.blockSignals(False)
    
    return None


def get_variable_from_combo_or_fallback(
    combo_box,
    df: pd.DataFrame,
    context: str = "analysis"
) -> VariableValidationResult:
    """
    Get and validate variable from a combo box, with automatic fallback.
    
    This is the main entry point for panels to get a validated variable.
    
    Args:
        combo_box: QComboBox widget (can be None)
        df: DataFrame with data
        context: Context for error messages
        
    Returns:
        VariableValidationResult with validation status
    """
    variable = None
    available = get_grade_columns(df) if df is not None else []
    
    # Try to get from combo box
    if combo_box is not None:
        try:
            variable = combo_box.currentText().strip() if combo_box.currentText() else None
        except Exception:
            variable = None
    
    # If combo is empty, try to populate it
    if not variable and combo_box is not None and df is not None and not df.empty:
        variable = populate_variable_combo(combo_box, df)
    
    # If still no variable but we have available columns, auto-select
    if not variable and available:
        selected, _ = auto_select_variable(df, fallback_to_first=True)
        if selected:
            variable = selected
            logger.info(f"Auto-selected variable '{selected}' for {context}")
            # Update combo box if available
            if combo_box is not None:
                try:
                    combo_box.setCurrentText(selected)
                except Exception:
                    pass
    
    # Validate the variable
    result = validate_variable(variable, df, context)
    
    # Final fallback: if invalid but we have available variables, force first one
    if not result.is_valid and available:
        fallback_var = available[0]
        logger.warning(f"Forcing fallback to first variable '{fallback_var}' for {context}")
        return VariableValidationResult(
            is_valid=True,
            variable=fallback_var,
            error_message=None,
            available_variables=available
        )
    
    return result


def ensure_required_columns(
    df: pd.DataFrame,
    variable: str,
    coord_cols: List[str] = None
) -> Tuple[bool, Optional[str], List[str]]:
    """
    Ensure DataFrame has required columns for kriging/simulation.
    
    Args:
        df: Input DataFrame
        variable: Grade variable column name
        coord_cols: Coordinate column names (default: ['X', 'Y', 'Z'])
        
    Returns:
        Tuple of (is_valid, error_message, missing_columns)
    """
    if coord_cols is None:
        coord_cols = ['X', 'Y', 'Z']
    
    required = coord_cols + [variable]
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}", missing
    
    return True, None, []

