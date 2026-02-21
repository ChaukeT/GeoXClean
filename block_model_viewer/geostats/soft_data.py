"""
Soft Data Structures and Utilities (High-Performance).

OPTIMIZATION: Numba JIT Compilation for Zero-Copy Math.

Handles probabilistic data inputs for Bayesian/Soft Kriging.
Refactored for Numba JIT compilation to replace standard NumPy broadcasting.

Why Numba here?
Standard NumPy vectorization creates intermediate arrays. For 100M blocks x 50 thresholds,
NumPy tries to allocate a 500M * float32 matrix (~20GB RAM), causing a crash.
Numba fuses these loops, calculating Mean/Variance per block without memory allocation
overhead, running at C++ speeds.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

logger = logging.getLogger(__name__)


@dataclass
class SoftDataSet:
    """
    Structure-of-Arrays (SoA) for maximum memory locality.
    
    Coordinates are automatically normalized to local origin (0,0,0) before float32 conversion
    to ensure numerical stability with large coordinate systems (e.g., UTM 6,000,000).
    This prevents precision errors in downstream kriging operations (especially drift matrices).
    """
    coords: np.ndarray        # (N, 3) float32 - normalized to local origin
    means: np.ndarray         # (N,) float32
    variances: np.ndarray     # (N,) float32
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize coordinates to local origin BEFORE float32 conversion for numerical stability
        # Large coordinates (e.g., UTM 6,000,000) lose precision in float32, causing issues
        # in downstream kriging operations (especially drift matrices)
        coords_array = np.asarray(self.coords, dtype=np.float64)  # Use float64 for normalization
        coord_center = np.mean(coords_array, axis=0)
        coords_normalized = coords_array - coord_center
        
        # Store center for potential restoration (though typically not needed)
        self._coord_center = coord_center
        
        # Now convert to float32 (normalized coords are typically small, so float32 is safe)
        self.coords = coords_normalized.astype(np.float32)
        self.means = np.asarray(self.means, dtype=np.float32)
        self.variances = np.asarray(self.variances, dtype=np.float32)
        
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(f"Coords must be (N, 3). Got {self.coords.shape}")
        
        if len(self.means) != len(self.coords) or len(self.variances) != len(self.coords):
            raise ValueError("Length mismatch between coords, means, and variances.")

    @property
    def n_points(self) -> int:
        return len(self.coords)
    
    def get_std_devs(self) -> np.ndarray:
        return np.sqrt(self.variances)
    
    def get_coords(self) -> np.ndarray:
        """
        Get normalized coordinates (centered at local origin).
        
        Returns:
            (N, 3) array of normalized coordinates (float32)
        
        Note: Coordinates are normalized to local origin (0,0,0) for numerical stability.
        This ensures float32 precision is sufficient even for large coordinate systems.
        """
        return self.coords
    
    def get_means(self) -> np.ndarray:
        """Get mean values array."""
        return self.means
    
    def get_variances(self) -> np.ndarray:
        """Get variance values array."""
        return self.variances
    
    def get_coord_center(self) -> np.ndarray:
        """
        Get the coordinate center used for normalization.
        
        Returns:
            (3,) array with original coordinate center (float64)
        """
        return getattr(self, '_coord_center', np.array([0.0, 0.0, 0.0]))
    
    @property
    def points(self) -> list:
        """
        Compatibility property for legacy code that expects a 'points' attribute.
        
        Returns:
            Empty list (SoftDataSet uses SoA structure, not list of points)
        """
        return []


# --- NUMBA KERNELS (The Speed Engine) ---

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _numba_ik_to_moments(probs, thresholds, min_t, max_t):
    """
    Converts CDF probabilities to Mean and Variance using Numba.
    
    Args:
        probs (float32[:, :]): Shape (N_blocks, N_thresholds). CDF values.
        thresholds (float32[:]): Shape (N_thresholds,). Cutoff values.
        min_t (float32): Lower tail bound.
        max_t (float32): Upper tail bound.
        
    Returns:
        means (float32[:]), variances (float32[:])
    """
    n_blocks = probs.shape[0]
    n_thresh = probs.shape[1]
    
    means = np.zeros(n_blocks, dtype=np.float32)
    vars_ = np.zeros(n_blocks, dtype=np.float32)
    
    # Parallelize over blocks (OpenMP style)
    for i in prange(n_blocks):
        m = 0.0
        m2 = 0.0
        
        # 1. Lower Tail: min_t to thresholds[0]
        # P_mass = probs[i, 0] - 0.0
        p_val = probs[i, 0]
        val_mid = (min_t + thresholds[0]) * 0.5
        m += p_val * val_mid
        m2 += p_val * (val_mid * val_mid)
        
        prev_p = p_val
        
        # 2. Middle Intervals
        for t in range(1, n_thresh):
            p_curr = probs[i, t]
            p_mass = p_curr - prev_p
            
            # Midpoint between t-1 and t
            val_mid = (thresholds[t-1] + thresholds[t]) * 0.5
            
            m += p_mass * val_mid
            m2 += p_mass * (val_mid * val_mid)
            
            prev_p = p_curr
            
        # 3. Upper Tail: thresholds[-1] to max_t
        p_mass = 1.0 - prev_p
        val_mid = (thresholds[n_thresh-1] + max_t) * 0.5
        
        m += p_mass * val_mid
        m2 += p_mass * (val_mid * val_mid)
        
        # Store
        means[i] = m
        # Var = E[X^2] - (E[X])^2
        v = m2 - (m * m)
        vars_[i] = v if v > 0 else 0.0
        
    return means, vars_


# Fallback Python implementation if Numba not available
def _python_ik_to_moments(probs, thresholds, min_t, max_t):
    """
    Python fallback for IK to moments conversion (slower but works without Numba).
    """
    n_blocks = probs.shape[0]
    n_thresh = probs.shape[1]
    
    means = np.zeros(n_blocks, dtype=np.float32)
    vars_ = np.zeros(n_blocks, dtype=np.float32)
    
    for i in range(n_blocks):
        m = 0.0
        m2 = 0.0
        
        # Lower Tail
        p_val = probs[i, 0]
        val_mid = (min_t + thresholds[0]) * 0.5
        m += p_val * val_mid
        m2 += p_val * (val_mid * val_mid)
        
        prev_p = p_val
        
        # Middle Intervals
        for t in range(1, n_thresh):
            p_curr = probs[i, t]
            p_mass = p_curr - prev_p
            val_mid = (thresholds[t-1] + thresholds[t]) * 0.5
            m += p_mass * val_mid
            m2 += p_mass * (val_mid * val_mid)
            prev_p = p_curr
            
        # Upper Tail
        p_mass = 1.0 - prev_p
        val_mid = (thresholds[n_thresh-1] + max_t) * 0.5
        m += p_mass * val_mid
        m2 += p_mass * (val_mid * val_mid)
        
        means[i] = m
        v = m2 - (m * m)
        vars_[i] = v if v > 0 else 0.0
        
    return means, vars_


def soft_from_block_model(
    block_model: Any,
    property_name: str,
    mask: Optional[np.ndarray] = None,
    variance_estimate: Optional[float] = None
) -> SoftDataSet:
    """
    Vectorized extraction from Block Model.
    
    Coordinates are automatically normalized to local origin in SoftDataSet.__post_init__
    for numerical stability with large coordinate systems.
    """
    if not hasattr(block_model, 'positions') or block_model.positions is None:
        raise ValueError("Block model has no positions")
    
    if property_name not in block_model.properties:
        raise ValueError(f"Property '{property_name}' not found.")
    
    # zero-copy references if possible
    positions = block_model.positions
    values = block_model.properties[property_name]
    
    # Apply mask via boolean indexing (fast)
    if mask is not None:
        positions = positions[mask]
        values = values[mask]
    
    N = len(values)
    
    # Vectorized variance assignment
    if variance_estimate is None:
        # If no variance provided, assume global variance of the property
        global_var = np.var(values)
        variances = np.full(N, global_var, dtype=np.float32)
    else:
        variances = np.full(N, variance_estimate, dtype=np.float32)
    
    return SoftDataSet(
        coords=positions,
        means=values,
        variances=variances,
        metadata={
            'source': 'block_model',
            'property_name': property_name
        }
    )


def soft_from_ik_result(
    ik_result: Any,
    thresholds: Optional[Union[list, np.ndarray]] = None,
    upper_tail: Optional[float] = None,
    lower_tail: Optional[float] = None
) -> SoftDataSet:
    """
    Optimized converter calling the Numba kernel.
    
    Converts Indicator Kriging CDFs to Soft Means and Variances.
    Uses Numba JIT compilation for high-performance processing.
    """
    # 1. Data Extraction (Handle Dict or Object)
    if isinstance(ik_result, dict):
        probs = ik_result.get('probabilities') 
        grid_x = ik_result.get('grid_x')
        grid_y = ik_result.get('grid_y')
        grid_z = ik_result.get('grid_z')
        ik_thresholds = ik_result.get('thresholds', thresholds)
    else:
        probs = ik_result.probabilities
        grid_x = ik_result.grid_x
        grid_y = ik_result.grid_y
        grid_z = ik_result.grid_z
        ik_thresholds = getattr(ik_result, 'thresholds', thresholds)

    # 2. Validation
    if probs is None or ik_thresholds is None:
        raise ValueError("IK Result missing probabilities or thresholds.")

    # Flatten inputs for Numba
    ik_thresholds = np.asarray(ik_thresholds, dtype=np.float32)
    # Ensure sorted
    sort_idx = np.argsort(ik_thresholds)
    ik_thresholds = ik_thresholds[sort_idx]
    
    # Handle Grids vs Lists
    # Use float64 initially for coordinate normalization (before float32 conversion in SoftDataSet)
    if grid_x.ndim == 3:
        n_total = grid_x.size
        coords = np.column_stack((
            grid_x.ravel().astype(np.float64), 
            grid_y.ravel().astype(np.float64), 
            grid_z.ravel().astype(np.float64)
        ))
        # Reshape probs to (N, n_thresh)
        if probs.ndim == 4:
            probs = probs.reshape(n_total, -1)
    else:
        coords = np.column_stack((
            np.asarray(grid_x, dtype=np.float64),
            np.asarray(grid_y, dtype=np.float64),
            np.asarray(grid_z, dtype=np.float64)
        ))
    
    # Note: SoftDataSet.__post_init__ will normalize coordinates before float32 conversion

    # Reorder probs to match sorted thresholds
    probs = probs[:, sort_idx].astype(np.float32)

    # Define Tails
    min_t = float(lower_tail if lower_tail is not None else ik_thresholds[0])
    max_t = float(upper_tail if upper_tail is not None else ik_thresholds[-1])

    # 3. CALL NUMBA KERNEL (or Python fallback)
    # This runs in C-speed, effectively zero RAM overhead
    if NUMBA_AVAILABLE:
        means, variances = _numba_ik_to_moments(probs, ik_thresholds, min_t, max_t)
    else:
        logger.warning("Numba not available. Using slower Python fallback.")
        means, variances = _python_ik_to_moments(probs, ik_thresholds, min_t, max_t)

    return SoftDataSet(
        coords=coords,
        means=means,
        variances=variances,
        metadata={'source': 'ik_result_numba' if NUMBA_AVAILABLE else 'ik_result'}
    )


def soft_from_csv(filepath: str) -> SoftDataSet:
    """
    Optimized CSV Loader with auto-detection of column names.
    """
    try:
        # engine='c' and low_memory=False for speed
        df = pd.read_csv(filepath, engine='c') 
    except Exception as e:
        raise IOError(f"CSV Read Error: {e}")

    # Auto-detect columns
    cols = df.columns.str.lower()
    
    # Map 'x', 'y', 'z'
    x_col = None
    y_col = None
    z_col = None
    for i, col_lower in enumerate(cols):
        if col_lower == 'x' and x_col is None:
            x_col = df.columns[i]
        elif col_lower == 'y' and y_col is None:
            y_col = df.columns[i]
        elif col_lower == 'z' and z_col is None:
            z_col = df.columns[i]
    
    if x_col is None or y_col is None or z_col is None:
        raise ValueError("Could not auto-detect X, Y, Z columns.")
    
    # Map Mean/Grade
    possible_means = ['mean', 'avg', 'grade', 'value']
    mean_col = next((c for c in df.columns if c.lower() in possible_means), None)
    if not mean_col: 
        raise ValueError("Could not auto-detect Mean column (e.g., 'mean', 'grade').")
    
    # Map Variance
    possible_vars = ['var', 'variance', 'std', 'stddev']
    var_col = next((c for c in df.columns if c.lower() in possible_vars), None)

    # Convert to Numpy Float64 initially (for normalization), then SoftDataSet will convert to float32
    coords = df[[x_col, y_col, z_col]].to_numpy(dtype=np.float64)
    means = df[mean_col].to_numpy(dtype=np.float32)

    if var_col:
        variances = df[var_col].to_numpy(dtype=np.float32)
        # Check if column was actually std dev
        if 'std' in var_col.lower():
            variances = variances ** 2
    else:
        # Default global variance if missing
        glob_var = np.var(means)
        variances = np.full_like(means, glob_var, dtype=np.float32)

    return SoftDataSet(
        coords=coords,
        means=means,
        variances=variances,
        metadata={'source': 'csv', 'path': filepath}
    )
