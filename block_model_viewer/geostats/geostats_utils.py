"""
Shared Geostatistics Utilities.

Centralized variogram functions and common geostatistical operations
to reduce duplication across modules.

AUDIT FIX (V-002): This module now uses the canonical variogram functions
from variogram_model.py. The legacy signature wrappers (h, nugget, sill, range_)
are preserved for backward compatibility but delegate to the canonical functions.

See: docs/VARIOGRAM_SUBSYSTEM_AUDIT.md
"""

import numpy as np
from typing import Dict, Callable, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Try to import scipy.spatial.cKDTree
try:
    from scipy.spatial import cKDTree
    KDTree_AVAILABLE = True
except ImportError:
    cKDTree = None
    KDTree_AVAILABLE = False
    logger.warning("scipy.spatial.cKDTree not available. Neighbor search will fail.")

# Try to import numba for acceleration
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.debug("Numba not available, using numpy fallback for geostatistics")


# ----------------------------------------------------------------------
# Variogram Model Functions - CANONICAL SOURCE
# ----------------------------------------------------------------------
# AUDIT FIX: Import canonical functions from unified module to ensure consistency.
# These are the GSLIB standard functions with signature: func(h, range_, sill, nugget)

try:
    from .variogram_model import (
        spherical_model as _canonical_spherical,
        exponential_model as _canonical_exponential,
        gaussian_model as _canonical_gaussian,
        MODEL_MAP as _CANONICAL_MODEL_MAP,
        VariogramModel,
    )
    UNIFIED_VARIOGRAM_AVAILABLE = True
except ImportError:
    UNIFIED_VARIOGRAM_AVAILABLE = False
    logger.warning("AUDIT: Unified variogram_model not available, using local implementations")
    _canonical_spherical = None
    _canonical_exponential = None
    _canonical_gaussian = None


# Legacy wrapper functions - PRESERVE OLD SIGNATURE for backward compatibility
# Old signature: (h, nugget, sill, range_) - used by some existing code
# New canonical signature: (h, range_, sill, nugget) - GSLIB standard

def spherical_variogram(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """
    Spherical variogram model (LEGACY SIGNATURE).
    
    AUDIT NOTE: This function uses the old parameter order (h, nugget, sill, range_)
    for backward compatibility. New code should use variogram_model.spherical_model
    which uses the GSLIB standard order (h, range_, sill, nugget).
    
    Args:
        h: Distance array
        nugget: Nugget effect (variance at h=0)
        sill: Total sill (nugget + partial sill)
        range_: Range parameter (distance where sill is reached)
    
    Returns:
        Variogram values
    """
    if UNIFIED_VARIOGRAM_AVAILABLE and _canonical_spherical is not None:
        # Delegate to canonical function (reorder parameters)
        return _canonical_spherical(h, range_, sill, nugget)
    else:
        # Local fallback implementation
        h = np.asarray(h, dtype=float)
        a = max(range_, 1e-9)
        c = max(sill - nugget, 0.0)
        hr = h / a
        gamma = np.full_like(h, nugget + c, dtype=float)
        mask = hr < 1.0
        gamma[mask] = nugget + c * (1.5 * hr[mask] - 0.5 * hr[mask] ** 3)
        return gamma


def exponential_variogram(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """
    Exponential variogram model (LEGACY SIGNATURE).
    
    AUDIT NOTE: This function uses the old parameter order (h, nugget, sill, range_)
    for backward compatibility. New code should use variogram_model.exponential_model.
    
    Args:
        h: Distance array
        nugget: Nugget effect
        sill: Total sill
        range_: Practical range (distance where 95% of sill is reached)
    
    Returns:
        Variogram values
    """
    if UNIFIED_VARIOGRAM_AVAILABLE and _canonical_exponential is not None:
        return _canonical_exponential(h, range_, sill, nugget)
    else:
        h = np.asarray(h, dtype=float)
        a = max(range_, 1e-9)
        c = max(sill - nugget, 0.0)
        return nugget + c * (1.0 - np.exp(-3.0 * h / a))


def gaussian_variogram(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """
    Gaussian variogram model (LEGACY SIGNATURE).
    
    AUDIT NOTE: This function uses the old parameter order (h, nugget, sill, range_)
    for backward compatibility. New code should use variogram_model.gaussian_model.
    
    Args:
        h: Distance array
        nugget: Nugget effect
        sill: Total sill
        range_: Practical range
    
    Returns:
        Variogram values
    """
    if UNIFIED_VARIOGRAM_AVAILABLE and _canonical_gaussian is not None:
        return _canonical_gaussian(h, range_, sill, nugget)
    else:
        h = np.asarray(h, dtype=float)
        a = max(range_, 1e-9)
        c = max(sill - nugget, 0.0)
        return nugget + c * (1.0 - np.exp(-3.0 * (h / a) ** 2))


# Model map for easy lookup (LEGACY - uses old signature wrappers)
VARIogram_MODELS: Dict[str, Callable] = {
    "spherical": spherical_variogram,
    "exponential": exponential_variogram,
    "gaussian": gaussian_variogram,
}

# CANONICAL model map (uses GSLIB standard signature)
# New code should prefer this over VARIogram_MODELS
if UNIFIED_VARIOGRAM_AVAILABLE:
    CANONICAL_VARIOGRAM_MODELS = _CANONICAL_MODEL_MAP
else:
    # Fallback: create canonical-signature wrappers
    def _make_canonical_wrapper(legacy_func):
        def canonical_wrapper(h, range_, sill, nugget):
            return legacy_func(h, nugget, sill, range_)
        return canonical_wrapper
    
    CANONICAL_VARIOGRAM_MODELS = {
        "spherical": _make_canonical_wrapper(spherical_variogram),
        "exponential": _make_canonical_wrapper(exponential_variogram),
        "gaussian": _make_canonical_wrapper(gaussian_variogram),
    }


# ----------------------------------------------------------------------
# Numba-accelerated distance calculations (if available)
# ----------------------------------------------------------------------

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def euclidean_distance_3d(x1: float, y1: float, z1: float,
                               x2: float, y2: float, z2: float) -> float:
        """Fast 3D Euclidean distance calculation."""
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        return np.sqrt(dx*dx + dy*dy + dz*dz)
else:
    def euclidean_distance_3d(x1: float, y1: float, z1: float,
                               x2: float, y2: float, z2: float) -> float:
        """3D Euclidean distance calculation (numpy fallback)."""
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        return np.sqrt(dx*dx + dy*dy + dz*dz)


# ----------------------------------------------------------------------
# Variogram fitting utilities
# ----------------------------------------------------------------------

def estimate_variogram_parameters(
    lags: np.ndarray,
    semivariances: np.ndarray,
    model_type: str = "spherical"
) -> Tuple[float, float, float]:
    """
    Estimate variogram parameters from experimental data.
    
    Args:
        lags: Lag distances
        semivariances: Experimental semivariances
        model_type: Model type ('spherical', 'exponential', 'gaussian')
    
    Returns:
        Tuple of (nugget, sill, range)
    """
    # Filter valid data
    mask = np.isfinite(lags) & np.isfinite(semivariances) & (lags >= 0) & (semivariances >= 0)
    x = lags[mask]
    y = semivariances[mask]
    
    if len(x) < 3:
        # Not enough data
        sill_est = float(np.nanmax(y)) if len(y) > 0 else 1.0
        range_est = float(np.nanmax(x) * 0.7) if len(x) > 0 else 1.0
        return 0.0, sill_est, max(range_est, 1.0)
    
    # Estimate nugget from first lag or y-intercept
    nugget_est = float(y[0]) if len(y) > 0 else 0.0
    
    # Estimate sill from plateau
    # Use last 30% of data to estimate sill
    n_plateau = max(3, int(len(y) * 0.3))
    sill_est = float(np.nanmean(y[-n_plateau:]))
    
    # Estimate range (distance where variogram reaches ~95% of sill)
    target = nugget_est + 0.95 * (sill_est - nugget_est)
    range_idx = np.where(y >= target)[0]
    if len(range_idx) > 0:
        range_est = float(x[range_idx[0]])
    else:
        range_est = float(np.nanmax(x) * 0.7)
    
    return nugget_est, sill_est, max(range_est, 1.0)


def get_variogram_function(model_type: str) -> Callable:
    """
    Get variogram function by name.
    
    Args:
        model_type: Model type name
    
    Returns:
        Variogram function
    """
    func = VARIogram_MODELS.get(model_type.lower(), spherical_variogram)
    if func is None:
        logger.warning(f"Unknown variogram model '{model_type}', using spherical")
        return spherical_variogram
    return func


# ----------------------------------------------------------------------
# Neighbor Search (Consolidated) - DETERMINISTIC VERSION
# ----------------------------------------------------------------------

def _stable_argsort_with_tiebreak(distances: np.ndarray, original_indices: np.ndarray) -> np.ndarray:
    """
    Perform stable argsort with deterministic tie-breaking.
    
    When two points have identical distances, breaks ties by original index
    to ensure consistent ordering across runs. This is critical for kriging
    reproducibility.
    
    Args:
        distances: Array of distances to sort
        original_indices: Array of original indices (used for tie-breaking)
    
    Returns:
        Array of indices that would sort distances with deterministic ties
    """
    # Use lexsort: sorts by last key first, then by earlier keys for ties
    # Sort by (original_index, distance) so distance is primary, index is tiebreaker
    # lexsort sorts by LAST key first, so we pass (original_indices, distances)
    # to get primary sort by distance, secondary by index
    return np.lexsort((original_indices, distances))


class NeighborSearcher:
    """
    Unified neighbor search for all kriging engines.
    
    DETERMINISTIC DESIGN:
    ---------------------
    This class ensures reproducible results across multiple runs by:
    1. Using stable sorting with deterministic tie-breaking (by original index)
    2. Consistent ordering when multiple points are equidistant from target
    3. Fixed neighbor ranking following professional geostatistics conventions
    
    Professional engines (Datamine, Vulcan, Leapfrog) maintain determinism by
    fixing neighbor ranking and solver ordering. This implementation follows
    the same principles.
    
    Handles:
    - Anisotropy transformation
    - cKDTree construction
    - KNN and radius+KMax queries
    - Padding with -1 for Numba compatibility
    - Single target case handling
    - **Deterministic tie-breaking for equal distances**
    
    This consolidates duplicate neighbor search logic from:
    - universal_kriging.py
    - indicator_kriging.py
    - cokriging3d.py
    - kriging3d.py
    """
    
    def __init__(
        self,
        data_coords: np.ndarray,
        anisotropy_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize neighbor searcher.
        
        Args:
            data_coords: (N, 3) array of data coordinates
            anisotropy_params: Optional dict with keys:
                - 'azimuth': float (degrees)
                - 'dip': float (degrees)
                - 'major_range': float
                - 'minor_range': float
                - 'vert_range': float
        """
        if not KDTree_AVAILABLE:
            raise ImportError("scipy.spatial.cKDTree is required for NeighborSearcher")
        
        self.data_coords_orig = np.asarray(data_coords, dtype=np.float64)
        self.n_data = len(self.data_coords_orig)
        
        # Store original indices for deterministic tie-breaking
        self._original_indices = np.arange(self.n_data, dtype=np.int64)
        
        # Apply anisotropy transformation if provided
        if anisotropy_params:
            from ..models.kriging3d import apply_anisotropy
            self.data_coords_transformed = apply_anisotropy(
                self.data_coords_orig,
                anisotropy_params.get('azimuth', 0.0),
                anisotropy_params.get('dip', 0.0),
                anisotropy_params.get('major_range', 100.0),
                anisotropy_params.get('minor_range', 100.0),
                anisotropy_params.get('vert_range', 100.0)
            )
            self.has_anisotropy = True
            self.anisotropy_params = anisotropy_params
        else:
            self.data_coords_transformed = self.data_coords_orig.copy()
            self.has_anisotropy = False
            self.anisotropy_params = None
        
        # Build KDTree from transformed coordinates
        self.tree = cKDTree(self.data_coords_transformed)
    
    def search(
        self,
        target_coords: np.ndarray,
        n_neighbors: int = 12,
        max_distance: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for neighbors for target coordinates.
        
        DETERMINISTIC: Results are guaranteed to be identical across runs
        when called with the same inputs. Tie-breaking uses original data
        indices to ensure consistent neighbor ordering.
        
        Args:
            target_coords: (M, 3) array of target coordinates
            n_neighbors: Maximum number of neighbors to find
            max_distance: Optional maximum search distance (in original coordinate space)
                         If anisotropy is enabled, this is automatically transformed
        
        Returns:
            Tuple of (neighbor_indices, distances) where:
            - neighbor_indices: (M, n_neighbors) array with -1 padding for missing neighbors
            - distances: (M, n_neighbors) array with np.inf for missing neighbors
        """
        target_coords = np.asarray(target_coords, dtype=np.float64)
        
        # Transform target coordinates if anisotropy is enabled
        if self.has_anisotropy:
            from ..models.kriging3d import apply_anisotropy
            target_transformed = apply_anisotropy(
                target_coords,
                self.anisotropy_params.get('azimuth', 0.0),
                self.anisotropy_params.get('dip', 0.0),
                self.anisotropy_params.get('major_range', 100.0),
                self.anisotropy_params.get('minor_range', 100.0),
                self.anisotropy_params.get('vert_range', 100.0)
            )
            
            # Transform max_distance to anisotropic space
            if max_distance is not None:
                range_geometric_mean = (
                    self.anisotropy_params.get('major_range', 100.0) *
                    self.anisotropy_params.get('minor_range', 100.0) *
                    self.anisotropy_params.get('vert_range', 100.0)
                ) ** (1.0 / 3.0)
                max_distance_transformed = max_distance / range_geometric_mean
            else:
                max_distance_transformed = None
        else:
            target_transformed = target_coords
            max_distance_transformed = max_distance
        
        # Query neighbors
        if max_distance_transformed is not None:
            # Radius + KMax search with DETERMINISTIC tie-breaking
            neighbor_indices_list = []
            distances_list = []
            
            for i in range(len(target_transformed)):
                p = target_transformed[i]
                idxs = self.tree.query_ball_point(p, r=max_distance_transformed)
                
                if len(idxs) == 0:
                    # No neighbors within range - pad with -1
                    neighbor_indices_list.append([-1] * n_neighbors)
                    distances_list.append([np.inf] * n_neighbors)
                    continue
                
                # Convert to numpy array for consistent handling
                idxs = np.array(idxs, dtype=np.int64)
                
                # Get distances
                pts = self.data_coords_transformed[idxs]
                d = np.linalg.norm(pts - p, axis=1)
                
                # DETERMINISTIC: Use stable sort with tie-breaking by original index
                # This ensures that when two points have identical distances,
                # the one with the smaller original index comes first
                order = _stable_argsort_with_tiebreak(d, idxs)[:n_neighbors]
                nbr_idx = idxs[order]
                nbr_dist = d[order]
                
                # Pad to n_neighbors if needed
                if len(nbr_idx) < n_neighbors:
                    padded_idx = np.full(n_neighbors, -1, dtype=np.int64)
                    padded_dist = np.full(n_neighbors, np.inf, dtype=np.float64)
                    padded_idx[:len(nbr_idx)] = nbr_idx
                    padded_dist[:len(nbr_dist)] = nbr_dist
                    neighbor_indices_list.append(padded_idx.tolist())
                    distances_list.append(padded_dist.tolist())
                else:
                    neighbor_indices_list.append(nbr_idx.tolist())
                    distances_list.append(nbr_dist.tolist())
            
            neighbor_indices = np.array(neighbor_indices_list, dtype=np.int64)
            distances = np.array(distances_list, dtype=np.float64)
        else:
            # KNN search (no distance limit) with DETERMINISTIC tie-breaking
            k = min(n_neighbors, len(self.data_coords_transformed))
            dists, indices = self.tree.query(target_transformed, k=k)
            
            # Handle scalar return (when k=1 or single target)
            if dists.ndim == 1:
                dists = dists.reshape(1, -1)
                indices = indices.reshape(1, -1)
            
            # DETERMINISTIC: Re-sort with stable tie-breaking
            # cKDTree.query may not have deterministic tie-breaking behavior
            # across different runs or platforms, so we enforce it here
            m = len(target_transformed)
            sorted_indices = np.zeros_like(indices, dtype=np.int64)
            sorted_dists = np.zeros_like(dists, dtype=np.float64)
            
            for i in range(m):
                row_dists = dists[i]
                row_indices = indices[i].astype(np.int64)
                
                # Apply stable sort with tie-breaking by original index
                order = _stable_argsort_with_tiebreak(row_dists, row_indices)
                sorted_indices[i] = row_indices[order]
                sorted_dists[i] = row_dists[order]
            
            indices = sorted_indices
            dists = sorted_dists
            
            # Ensure we have exactly n_neighbors columns (pad if needed)
            if indices.shape[1] < n_neighbors:
                padded_idx = np.full((m, n_neighbors), -1, dtype=np.int64)
                padded_dist = np.full((m, n_neighbors), np.inf, dtype=np.float64)
                padded_idx[:, :indices.shape[1]] = indices
                padded_dist[:, :dists.shape[1]] = dists
                neighbor_indices = padded_idx
                distances = padded_dist
            else:
                neighbor_indices = indices[:, :n_neighbors]
                distances = dists[:, :n_neighbors]
        
        # Replace np.inf distances with -1 in indices (for Numba safety)
        neighbor_indices[distances == np.inf] = -1
        
        return neighbor_indices, distances
    
    def get_transformed_coords(self) -> np.ndarray:
        """Get transformed coordinates (for covariance calculation)."""
        return self.data_coords_transformed
    
    def get_original_coords(self) -> np.ndarray:
        """Get original coordinates (for drift calculation)."""
        return self.data_coords_orig

