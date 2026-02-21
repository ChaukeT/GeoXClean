"""
Normal Score Transformer for Geostatistical Simulation
======================================================

Handles Forward (Raw -> Gaussian) and Backward (Gaussian -> Raw) transformations
for Sequential Gaussian Simulation (SGSIM) workflows.

Critical for proper metal/tonnage calculations - SGSIM runs in Gaussian space,
but all physical calculations must use back-transformed raw values.

AUDIT COMPLIANCE (TRF-002, TRF-003, TRF-006, TRF-007):
- Implements declustering weight support via weighted CDF
- Uses deterministic tie-breaking via lexsort
- Uses PCHIP interpolation for back-transform (monotonic, smoother)
- Logs warnings for clipped/extrapolated values

Author: Block Model Viewer Team
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.stats import norm
import logging
import pickle
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# TRANSFORM COLUMN SUFFIXES (for filtering in reporting)
# =============================================================================
TRANSFORM_SUFFIXES = ('_NS', '_GAUSS', '_LN', '_LOG', '_BC', '_SQRT', '_TR')


def is_transformed_column(column_name: str) -> bool:
    """
    Check if a column name indicates a transformed variable.
    
    Used by reporting panels to filter out transformed columns.
    
    Parameters
    ----------
    column_name : str
        Column name to check
        
    Returns
    -------
    bool
        True if column appears to be transformed
    """
    return column_name.upper().endswith(TRANSFORM_SUFFIXES)


def filter_transformed_columns(columns: list) -> list:
    """
    Filter out transformed columns from a list.
    
    TRF-012 COMPLIANCE: Prevents transformed values from appearing in
    resource reports and grade-tonnage curves.
    
    Parameters
    ----------
    columns : list
        List of column names
        
    Returns
    -------
    list
        Filtered list without transformed columns
    """
    return [c for c in columns if not is_transformed_column(c)]


class NormalScoreTransformer:
    """
    Handles Forward (Raw -> Gaussian) and Backward (Gaussian -> Raw) transformations.
    Essential for SGSIM workflows.
    
    AUDIT COMPLIANCE:
    - TRF-002: Supports declustering weights via weighted CDF
    - TRF-003: Deterministic tie-breaking using lexsort
    - TRF-006: PCHIP interpolation for smooth back-transform
    - TRF-007: Warnings for clipped/extrapolated values

    Usage:
        transformer = NormalScoreTransformer()
        transformer.fit(raw_grade_data, weights=declustering_weights)
        
        # Forward transform for SGSIM input
        gaussian_data = transformer.transform(raw_grade_data)
        
        # Run SGSIM on gaussian_data...
        
        # Back transform for physical calculations
        raw_realizations = transformer.back_transform(gaussian_realizations)
    """

    def __init__(self):
        self.raw_data_sorted = None
        self.gaussian_data_sorted = None
        self.is_fitted = False
        # Metadata for tail extrapolation (min/max capping)
        self.min_val = 0.0
        self.max_val = 1.0
        # Provenance tracking
        self.fit_timestamp: Optional[str] = None
        self.sample_count: int = 0
        self.weights_used: bool = False
        self.data_hash: Optional[str] = None
        # Clipping statistics (for audit)
        self._clip_count_lower: int = 0
        self._clip_count_upper: int = 0

    def fit(self, data: np.ndarray, weights: np.ndarray = None):
        """
        Learns the distribution of the raw data.
        
        TRF-002 COMPLIANCE: Implements weighted CDF for declustering weights.
        TRF-003 COMPLIANCE: Uses deterministic tie-breaking via lexsort.

        Parameters
        ----------
        data : np.ndarray
            Raw grade values (any shape, will be flattened)
        weights : np.ndarray, optional
            Weights for declustering. If provided, the CDF is computed using
            weighted cumulative sums. Weights should sum to 1 or will be normalized.
        """
        # Flatten data
        flat_data = np.asarray(data).ravel()
        
        # Remove NaNs
        valid_mask = ~np.isnan(flat_data)
        valid_data = flat_data[valid_mask]
        
        if len(valid_data) == 0:
            raise ValueError("No valid data to fit Normal Score Transform")
        
        # Handle weights
        if weights is not None:
            flat_weights = np.asarray(weights).ravel()
            if len(flat_weights) != len(flat_data):
                raise ValueError(
                    f"Weights length ({len(flat_weights)}) must match data length ({len(flat_data)})"
                )
            valid_weights = flat_weights[valid_mask]
            
            # Normalize weights
            weight_sum = np.sum(valid_weights)
            if weight_sum <= 0:
                logger.warning("TRF-002: Weights sum to zero or negative, using equal weights")
                valid_weights = np.ones(len(valid_data)) / len(valid_data)
            else:
                valid_weights = valid_weights / weight_sum
            
            self.weights_used = True
            logger.info(f"TRF-002: Using declustering weights (sum normalized to 1.0)")
        else:
            valid_weights = np.ones(len(valid_data)) / len(valid_data)
            self.weights_used = False

        self.min_val = np.min(valid_data)
        self.max_val = np.max(valid_data)
        self.sample_count = len(valid_data)
        
        # Compute data hash for provenance tracking
        self.data_hash = hashlib.sha256(valid_data.tobytes()).hexdigest()[:16]
        self.fit_timestamp = datetime.now().isoformat()

        # =================================================================
        # TRF-003 COMPLIANCE: Deterministic tie-breaking
        # =================================================================
        # When values are equal (ties), we use the original index as tiebreaker
        # This ensures reproducible results across runs
        n = len(valid_data)
        original_indices = np.arange(n)
        
        # lexsort sorts by LAST key first, so we pass (original_indices, valid_data)
        # Primary sort: by value, Secondary sort (tiebreaker): by original index
        sort_indices = np.lexsort((original_indices, valid_data))
        
        self.raw_data_sorted = valid_data[sort_indices]
        sorted_weights = valid_weights[sort_indices]

        # =================================================================
        # TRF-002 COMPLIANCE: Weighted CDF calculation
        # =================================================================
        # For weighted CDF, cumulative probability at point k is:
        # P(k) = sum(weights[0:k]) + 0.5 * weight[k]  (Hazen-like midpoint)
        cumulative_weights = np.cumsum(sorted_weights)
        
        # Compute plotting positions using weighted CDF
        # pk[i] = (cumsum[i] - 0.5 * w[i]) for midpoint plotting
        # This is equivalent to Hazen formula for equal weights
        pk = cumulative_weights - 0.5 * sorted_weights
        
        # Ensure pk stays in (0, 1) to avoid infinite Gaussian values
        pk = np.clip(pk, 1e-10, 1 - 1e-10)

        # Map to Gaussian (Quantile Function of Standard Normal)
        self.gaussian_data_sorted = norm.ppf(pk)

        self.is_fitted = True
        
        # Count ties for audit reporting
        unique_vals, counts = np.unique(valid_data, return_counts=True)
        n_ties = np.sum(counts > 1)
        if n_ties > 0:
            logger.info(
                f"TRF-003: {n_ties} unique values have ties ({np.sum(counts[counts > 1])} total tied samples). "
                f"Using deterministic tie-breaking by original index."
            )
        
        logger.info(
            f"NST Fitted on {n} samples. Range: [{self.min_val:.4f}, {self.max_val:.4f}]. "
            f"Weights: {'declustered' if self.weights_used else 'equal'}. Hash: {self.data_hash}"
        )

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Forward: Raw Grade -> Gaussian Value

        Parameters
        ----------
        data : np.ndarray
            Raw grade values (any shape)

        Returns
        -------
        np.ndarray
            Gaussian values (same shape as input)
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        # Handle input shape
        original_shape = data.shape
        flat_data = np.asarray(data).ravel()

        # Track values outside training range
        below_min = flat_data < self.min_val
        above_max = flat_data > self.max_val
        n_below = np.sum(below_min & ~np.isnan(flat_data))
        n_above = np.sum(above_max & ~np.isnan(flat_data))
        
        if n_below > 0 or n_above > 0:
            logger.warning(
                f"TRF-007: Transform clipping: {n_below} values below min ({self.min_val:.4f}), "
                f"{n_above} values above max ({self.max_val:.4f}). Values will be clamped."
            )

        # Interpolate: Given Raw X, find Gaussian Y
        # Using linear interpolation for forward transform (standard practice)
        f = interp1d(
            self.raw_data_sorted,
            self.gaussian_data_sorted,
            kind='linear',
            bounds_error=False,
            fill_value=(self.gaussian_data_sorted[0], self.gaussian_data_sorted[-1])
        )

        transformed = f(flat_data)
        return transformed.reshape(original_shape)

    def back_transform(self, gaussian_data: np.ndarray) -> np.ndarray:
        """
        Backward: Gaussian Value -> Raw Grade
        CRITICAL for SGSIM post-processing.
        
        TRF-006 COMPLIANCE: Uses PCHIP interpolation for smoother, monotonic
        back-transformation that better preserves quantiles.
        
        TRF-007 COMPLIANCE: Logs warnings for clipped values.

        Parameters
        ----------
        gaussian_data : np.ndarray
            Gaussian values (any shape)

        Returns
        -------
        np.ndarray
            Raw grade values (same shape as input)
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        original_shape = gaussian_data.shape
        flat_gauss = np.asarray(gaussian_data).ravel()
        
        # Track clipping for audit
        gauss_min = self.gaussian_data_sorted[0]
        gauss_max = self.gaussian_data_sorted[-1]
        
        valid_mask = ~np.isnan(flat_gauss)
        below_min = (flat_gauss < gauss_min) & valid_mask
        above_max = (flat_gauss > gauss_max) & valid_mask
        
        n_below = np.sum(below_min)
        n_above = np.sum(above_max)
        
        self._clip_count_lower += n_below
        self._clip_count_upper += n_above
        
        if n_below > 0 or n_above > 0:
            logger.warning(
                f"TRF-007: Back-transform clipping: {n_below} values below Gaussian min ({gauss_min:.3f}), "
                f"{n_above} values above Gaussian max ({gauss_max:.3f}). "
                f"Clipping to raw range [{self.min_val:.4f}, {self.max_val:.4f}]."
            )

        # =================================================================
        # TRF-006 COMPLIANCE: Use PCHIP interpolation
        # =================================================================
        # PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is:
        # - Monotonic (preserves order)
        # - Smooth (C1 continuous)
        # - Better for quantile preservation than linear
        try:
            pchip = PchipInterpolator(
                self.gaussian_data_sorted,
                self.raw_data_sorted,
                extrapolate=False  # Don't extrapolate beyond range
            )
            back_transformed = pchip(flat_gauss)
            
            # Handle values outside range (PCHIP returns NaN for extrapolation)
            out_of_range = np.isnan(back_transformed) & valid_mask
            if np.any(out_of_range):
                # Clip to bounds
                back_transformed[below_min] = self.min_val
                back_transformed[above_max] = self.max_val
                
        except Exception as e:
            # Fallback to linear if PCHIP fails
            logger.warning(f"PCHIP interpolation failed ({e}), falling back to linear")
            f = interp1d(
                self.gaussian_data_sorted,
                self.raw_data_sorted,
                kind='linear',
                bounds_error=False,
                fill_value=(self.min_val, self.max_val)
            )
            back_transformed = f(flat_gauss)

        return back_transformed.reshape(original_shape)

    def get_provenance(self) -> Dict[str, Any]:
        """
        Get provenance metadata for audit trail.
        
        Returns
        -------
        dict
            Provenance information including fit parameters and statistics
        """
        return {
            'is_fitted': self.is_fitted,
            'fit_timestamp': self.fit_timestamp,
            'sample_count': self.sample_count,
            'weights_used': self.weights_used,
            'data_hash': self.data_hash,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'clip_count_lower': self._clip_count_lower,
            'clip_count_upper': self._clip_count_upper,
        }

    def validate_data_match(self, data: np.ndarray) -> Tuple[bool, str]:
        """
        Validate that data matches the fitted transformer.
        
        Checks if the data hash matches to detect stale transformer usage.
        
        Parameters
        ----------
        data : np.ndarray
            Data to validate
            
        Returns
        -------
        tuple
            (is_match, message)
        """
        flat_data = np.asarray(data).ravel()
        valid_mask = ~np.isnan(flat_data)
        valid_data = flat_data[valid_mask]
        
        current_hash = hashlib.sha256(valid_data.tobytes()).hexdigest()[:16]
        
        if current_hash == self.data_hash:
            return True, "Data hash matches fitted transformer"
        else:
            return False, (
                f"Data hash mismatch! Fitted on {self.data_hash}, "
                f"current data is {current_hash}. Transformer may be stale."
            )

    def save(self, filepath: str):
        """
        Save transformer to file for later use.

        Parameters
        ----------
        filepath : str
            Path to save the transformer
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Transformer saved to {filepath}")

    def save_json(self, filepath: str):
        """
        Save transformer to JSON format for auditability.
        
        Safer and more portable than pickle.

        Parameters
        ----------
        filepath : str
            Path to save the transformer (should end in .json)
        """
        data = {
            'version': '2.0',  # Version with weighted CDF support
            'raw_data_sorted': self.raw_data_sorted.tolist(),
            'gaussian_data_sorted': self.gaussian_data_sorted.tolist(),
            'min_val': self.min_val,
            'max_val': self.max_val,
            'provenance': self.get_provenance(),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Transformer saved to JSON: {filepath}")

    @staticmethod
    def load(filepath: str) -> 'NormalScoreTransformer':
        """
        Load transformer from file.

        SECURITY: Validates path and file size before loading.
        Prefer using load_json() for safer deserialization.

        Parameters
        ----------
        filepath : str
            Path to load the transformer from

        Returns
        -------
        NormalScoreTransformer
            Loaded transformer instance
            
        Raises
        ------
        SecurityError
            If security validation fails
        """
        from ..utils.security import validate_pickle_file, SecurityError
        
        try:
            validated_path, file_size = validate_pickle_file(Path(filepath))
            logger.debug(f"Loading transformer from {validated_path} ({file_size} bytes)")
            with open(validated_path, 'rb') as f:
                transformer = pickle.load(f)
            logger.info(f"Transformer loaded from {filepath}")
            return transformer
        except SecurityError as e:
            logger.error(f"Security error loading transformer: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading transformer: {e}")
            raise

    @staticmethod
    def load_json(filepath: str) -> 'NormalScoreTransformer':
        """
        Load transformer from JSON file.

        Parameters
        ----------
        filepath : str
            Path to load the transformer from

        Returns
        -------
        NormalScoreTransformer
            Loaded transformer instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        transformer = NormalScoreTransformer()
        transformer.raw_data_sorted = np.array(data['raw_data_sorted'])
        transformer.gaussian_data_sorted = np.array(data['gaussian_data_sorted'])
        transformer.min_val = data['min_val']
        transformer.max_val = data['max_val']
        transformer.is_fitted = True
        
        # Restore provenance if available
        if 'provenance' in data:
            prov = data['provenance']
            transformer.fit_timestamp = prov.get('fit_timestamp')
            transformer.sample_count = prov.get('sample_count', len(transformer.raw_data_sorted))
            transformer.weights_used = prov.get('weights_used', False)
            transformer.data_hash = prov.get('data_hash')
        
        logger.info(f"Transformer loaded from JSON: {filepath}")
        return transformer
