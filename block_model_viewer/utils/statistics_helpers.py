"""
Centralized Statistics Helper Module.

STAT-009: Provides consistent statistical calculations across all GeoX components.

This module ensures:
- Consistent statistical formulas across the application
- Proper handling of edge cases (division by zero, empty data, NaN values)
- Support for declustering weights
- Proper normalisation of weighted statistics
- Audit-ready provenance tracking

USAGE:
    from block_model_viewer.utils.statistics_helpers import (
        compute_descriptive_stats,
        compute_weighted_stats,
        compute_cv_safe,
        validate_numeric_data,
    )
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StatisticsError(Exception):
    """Exception raised for statistics computation errors."""
    pass


class EmptyDataError(StatisticsError):
    """Exception raised when data is empty or contains no valid values."""
    pass


class InvalidWeightsError(StatisticsError):
    """Exception raised when weights are invalid (e.g., all zeros, negative)."""
    pass


def validate_numeric_data(
    data: Union[np.ndarray, pd.Series],
    name: str = "data",
    allow_empty: bool = False,
    allow_nan: bool = True,
) -> np.ndarray:
    """
    Validate and clean numeric data for statistical analysis.
    
    Args:
        data: Input data as numpy array or pandas Series
        name: Name of the data for error messages
        allow_empty: If False, raises EmptyDataError for empty arrays
        allow_nan: If True, removes NaN values; if False, raises error on NaN
    
    Returns:
        Cleaned numpy array with valid numeric values
        
    Raises:
        StatisticsError: For invalid data types
        EmptyDataError: For empty data when allow_empty=False
    """
    # Convert to numpy if needed
    if isinstance(data, pd.Series):
        data = data.values
    elif not isinstance(data, np.ndarray):
        try:
            data = np.asarray(data)
        except (TypeError, ValueError) as e:
            raise StatisticsError(f"Cannot convert {name} to numeric array: {e}")
    
    # Check for numeric type
    if not np.issubdtype(data.dtype, np.number):
        raise StatisticsError(f"{name} must be numeric, got dtype {data.dtype}")
    
    # Handle NaN values
    nan_mask = np.isnan(data)
    nan_count = np.sum(nan_mask)
    
    if not allow_nan and nan_count > 0:
        raise StatisticsError(f"{name} contains {nan_count} NaN values")
    
    # Remove NaN values
    valid_data = data[~nan_mask]
    
    # Check for empty
    if len(valid_data) == 0 and not allow_empty:
        raise EmptyDataError(f"{name} contains no valid (non-NaN) values")
    
    return valid_data


def compute_cv_safe(mean: float, std: float) -> float:
    """
    Compute coefficient of variation with division-by-zero protection.
    
    STAT-006: Guards against division by zero.
    
    Args:
        mean: Mean value
        std: Standard deviation
        
    Returns:
        CV as percentage (std/mean * 100), or 0.0 if mean is zero
    """
    if mean == 0 or np.isnan(mean) or np.isnan(std):
        return 0.0
    return (std / abs(mean)) * 100


def compute_descriptive_stats(
    data: Union[np.ndarray, pd.Series],
    name: str = "data",
    include_percentiles: bool = True,
) -> Dict[str, Any]:
    """
    Compute standard descriptive statistics with proper edge case handling.
    
    Args:
        data: Numeric data array
        name: Name of the variable for error messages
        include_percentiles: If True, include P10, P25, P50, P75, P90
        
    Returns:
        Dictionary with computed statistics
        
    Raises:
        EmptyDataError: If data contains no valid values
    """
    valid_data = validate_numeric_data(data, name, allow_empty=False)
    
    n = len(valid_data)
    mean_val = float(np.mean(valid_data))
    std_val = float(np.std(valid_data))
    var_val = float(np.var(valid_data))
    
    stats = {
        "count": n,
        "mean": mean_val,
        "std": std_val,
        "var": var_val,
        "cv": compute_cv_safe(mean_val, std_val),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "median": float(np.median(valid_data)),
    }
    
    if include_percentiles:
        stats.update({
            "p10": float(np.percentile(valid_data, 10)),
            "p25": float(np.percentile(valid_data, 25)),
            "p50": float(np.percentile(valid_data, 50)),
            "p75": float(np.percentile(valid_data, 75)),
            "p90": float(np.percentile(valid_data, 90)),
        })
    
    return stats


def validate_weights(
    weights: Union[np.ndarray, pd.Series],
    data_length: int,
    name: str = "weights",
) -> np.ndarray:
    """
    Validate weights for weighted statistics calculation.
    
    Args:
        weights: Weight values
        data_length: Expected length to match data
        name: Name for error messages
        
    Returns:
        Validated numpy array of weights
        
    Raises:
        InvalidWeightsError: If weights are invalid
    """
    # Convert to numpy
    if isinstance(weights, pd.Series):
        weights = weights.values
    elif not isinstance(weights, np.ndarray):
        try:
            weights = np.asarray(weights)
        except (TypeError, ValueError) as e:
            raise InvalidWeightsError(f"Cannot convert {name} to array: {e}")
    
    # Check length
    if len(weights) != data_length:
        raise InvalidWeightsError(
            f"{name} length ({len(weights)}) must match data length ({data_length})"
        )
    
    # Check for numeric
    if not np.issubdtype(weights.dtype, np.number):
        raise InvalidWeightsError(f"{name} must be numeric")
    
    # Check for non-negative
    if np.any(weights < 0):
        raise InvalidWeightsError(f"{name} contains negative values")
    
    # Check for all zeros
    if np.sum(weights) == 0:
        raise InvalidWeightsError(f"{name} sum is zero (all weights are zero)")
    
    return weights


def compute_weighted_stats(
    data: Union[np.ndarray, pd.Series],
    weights: Union[np.ndarray, pd.Series],
    name: str = "data",
    include_raw: bool = True,
) -> Dict[str, Any]:
    """
    Compute weighted statistics with proper normalisation.
    
    STAT-002, STAT-003: Implements correct weighted statistics for declustering.
    
    Args:
        data: Numeric data array
        weights: Weights array (e.g., declustering weights)
        name: Name of the variable
        include_raw: If True, also include unweighted (raw) statistics
        
    Returns:
        Dictionary with weighted statistics and optional raw statistics
        
    Note:
        Weighted variance uses the formula:
        var_w = sum(w_i * (x_i - mean_w)^2) / sum(w_i)
        
        This is the "reliability weights" formula appropriate for
        declustering weights where weights indicate sample importance,
        not frequency/replication counts.
    """
    valid_data = validate_numeric_data(data, name, allow_empty=False)
    valid_weights = validate_weights(weights, len(data), "weights")
    
    # Handle NaN alignment between data and weights
    nan_mask = np.isnan(data if isinstance(data, np.ndarray) else data.values)
    if np.any(nan_mask):
        valid_data = valid_data  # Already cleaned by validate_numeric_data
        valid_weights = valid_weights[~nan_mask]
    
    # Normalize weights
    weight_sum = np.sum(valid_weights)
    
    # Weighted mean
    weighted_mean = np.sum(valid_data * valid_weights) / weight_sum
    
    # Weighted variance (reliability weights formula)
    weighted_var = np.sum(valid_weights * (valid_data - weighted_mean) ** 2) / weight_sum
    weighted_std = np.sqrt(weighted_var)
    
    result = {
        "count": len(valid_data),
        "weighted_mean": float(weighted_mean),
        "weighted_std": float(weighted_std),
        "weighted_var": float(weighted_var),
        "weighted_cv": compute_cv_safe(weighted_mean, weighted_std),
        "weight_sum": float(weight_sum),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
    }
    
    if include_raw:
        raw_mean = float(np.mean(valid_data))
        raw_std = float(np.std(valid_data))
        
        result.update({
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "raw_var": float(np.var(valid_data)),
            "raw_cv": compute_cv_safe(raw_mean, raw_std),
            "bias_correction": float(weighted_mean - raw_mean),
            "bias_correction_pct": (
                float((weighted_mean - raw_mean) / raw_mean * 100) 
                if raw_mean != 0 else 0.0
            ),
        })
    
    return result


def compute_grade_tonnage_row(
    grades: np.ndarray,
    lengths: np.ndarray,
    cutoff: float,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute a single row of a grade-tonnage curve.
    
    STAT-003: Implements correct grade-tonnage calculation with declustering.
    STAT-006: Guards against division by zero.
    
    Args:
        grades: Grade values
        lengths: Interval lengths (tonnage proxy)
        cutoff: Cutoff grade
        weights: Optional declustering weights
        
    Returns:
        Dictionary with tonnage, grade, metal for this cutoff
    """
    # Filter by cutoff
    mask = grades >= cutoff
    
    if not np.any(mask):
        return {
            "cutoff": cutoff,
            "tonnes": 0.0,
            "grade": 0.0,
            "metal": 0.0,
            "count": 0,
            "weighted": weights is not None,
        }
    
    filtered_grades = grades[mask]
    filtered_lengths = lengths[mask]
    
    # STAT-006: Guard against zero tonnage
    tonnes = np.sum(filtered_lengths)
    if tonnes <= 0:
        return {
            "cutoff": cutoff,
            "tonnes": 0.0,
            "grade": 0.0,
            "metal": 0.0,
            "count": int(np.sum(mask)),
            "weighted": weights is not None,
        }
    
    # Calculate grade
    if weights is not None:
        filtered_weights = weights[mask]
        combined = filtered_lengths * filtered_weights
        combined_sum = np.sum(combined)
        
        if combined_sum > 0:
            grade = np.sum(filtered_grades * combined) / combined_sum
        else:
            # Fallback to length-weighted
            grade = np.sum(filtered_grades * filtered_lengths) / tonnes
    else:
        grade = np.sum(filtered_grades * filtered_lengths) / tonnes
    
    metal = tonnes * grade
    
    return {
        "cutoff": cutoff,
        "tonnes": float(tonnes),
        "grade": float(grade),
        "metal": float(metal),
        "count": int(np.sum(mask)),
        "weighted": weights is not None,
    }


def attach_provenance(
    result: Dict[str, Any],
    source_type: str,
    operation: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Attach provenance metadata to a statistics result.
    
    STAT-004: Implements provenance tracking for audit trail.
    
    Args:
        result: Statistics result dictionary
        source_type: Data source type ('composites', 'assays', etc.)
        operation: Operation performed ('descriptive_stats', 'grade_tonnage', etc.)
        **kwargs: Additional provenance fields
        
    Returns:
        Result dictionary with provenance attached
    """
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "source_type": source_type,
        "operation": operation,
        "module": "statistics_helpers",
        **kwargs
    }
    
    result["provenance"] = provenance
    return result


# Convenience functions for common patterns

def safe_mean(data: Union[np.ndarray, pd.Series]) -> float:
    """Compute mean, returning 0.0 for empty data."""
    try:
        valid = validate_numeric_data(data, allow_empty=True)
        return float(np.mean(valid)) if len(valid) > 0 else 0.0
    except StatisticsError:
        return 0.0


def safe_std(data: Union[np.ndarray, pd.Series]) -> float:
    """Compute std, returning 0.0 for empty data."""
    try:
        valid = validate_numeric_data(data, allow_empty=True)
        return float(np.std(valid)) if len(valid) > 0 else 0.0
    except StatisticsError:
        return 0.0


def safe_cv(data: Union[np.ndarray, pd.Series]) -> float:
    """Compute CV, returning 0.0 for empty data or zero mean."""
    mean = safe_mean(data)
    std = safe_std(data)
    return compute_cv_safe(mean, std)

