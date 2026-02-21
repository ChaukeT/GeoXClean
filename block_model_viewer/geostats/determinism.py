"""
Kriging Determinism Module

Ensures reproducible results across multiple runs of kriging algorithms.

BACKGROUND
----------
Ordinary Kriging is fundamentally deterministic - it solves a linear system
of equations, and the same inputs should always produce the same outputs.
Professional geostatistics software (Datamine, Vulcan, Leapfrog) guarantees
reproducibility by fixing neighbor ranking and solver ordering.

SOURCES OF NON-DETERMINISM
--------------------------
If you see run-to-run drift in kriging results, check these common causes:

1. **KD-tree/Ball-tree query ordering**
   - Floating-point tie breaks in neighbor distances
   - Different trees may return equidistant neighbors in different orders
   - FIX: Use stable sort with tie-breaking by original index

2. **Multi-threading execution order**
   - Numba prange / ThreadPoolExecutor may execute in non-deterministic order
   - This can affect which blocks are processed first
   - FIX: Results should be independent of processing order (currently OK)
   
3. **NumPy parallel thread non-determinism**
   - OpenBLAS/MKL may use non-deterministic parallel algorithms
   - FIX: Set OMP_NUM_THREADS=1 for strict reproducibility tests

4. **Pseudo-random sample thinning**
   - If your workflow includes random sampling, fix the random seed
   - FIX: Use np.random.seed() or pass explicit seeds

5. **Variogram spline interpolation**
   - Some interpolation methods may have floating-point instabilities
   - FIX: Use analytical variogram models (spherical, exponential, gaussian)

IMPLEMENTATION IN GeoX
----------------------
This module provides functions to:
1. Set environment variables for deterministic NumPy/OpenBLAS
2. Configure thread counts for reproducibility testing
3. Verify that kriging results are deterministic

USAGE
-----
For normal use, GeoX is deterministic by default (as of this fix).

For strict reproducibility testing:
    
    from block_model_viewer.geostats.determinism import enable_strict_determinism
    
    enable_strict_determinism()  # Sets OMP_NUM_THREADS=1, etc.
    
    # Run kriging...
    result1 = ordinary_kriging_3d(...)
    result2 = ordinary_kriging_3d(...)
    
    assert np.allclose(result1, result2)  # Should pass

For auditors who need to verify reproducibility:
    
    from block_model_viewer.geostats.determinism import verify_kriging_determinism
    
    is_deterministic, report = verify_kriging_determinism(
        data_coords, data_values, target_coords, variogram_params
    )
"""

import os
import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


def enable_strict_determinism():
    """
    Configure environment for strict determinism in kriging.
    
    Sets environment variables to ensure single-threaded execution
    in NumPy/OpenBLAS/MKL. This is useful for reproducibility testing
    but may reduce performance.
    
    Note: Must be called BEFORE importing numpy or kriging modules
    if environment variables are to take effect.
    """
    # Limit OpenMP threads (used by NumPy's OpenBLAS backend)
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Limit MKL threads (if using Intel MKL)
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Limit OpenBLAS threads
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # Limit Numba threading
    os.environ['NUMBA_NUM_THREADS'] = '1'
    
    logger.info("Strict determinism enabled: OMP/MKL/OpenBLAS/Numba threads set to 1")
    logger.warning("Performance may be reduced. Re-import numpy if already imported.")


def get_determinism_settings() -> Dict[str, Any]:
    """
    Get current determinism-related settings.
    
    Returns:
        Dict with current thread counts and settings
    """
    settings = {
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', 'not set'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS', 'not set'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS', 'not set'),
        'NUMBA_NUM_THREADS': os.environ.get('NUMBA_NUM_THREADS', 'not set'),
    }
    
    # Try to get actual thread counts from libraries
    try:
        import numba
        settings['numba_actual_threads'] = numba.config.NUMBA_NUM_THREADS
    except ImportError as e:
        logger.debug(f"Numba not available: {e}")
        settings['numba_actual_threads'] = 'unavailable'
    
    return settings


def verify_kriging_determinism(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    target_coords: np.ndarray,
    variogram_params: Dict,
    n_runs: int = 3,
    n_neighbors: int = 12,
    max_distance: Optional[float] = None,
    model_type: str = "spherical",
    rtol: float = 1e-10,
    atol: float = 1e-14
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that kriging produces deterministic results.
    
    Runs ordinary kriging multiple times and checks that results
    are identical (within floating-point precision).
    
    Args:
        data_coords: (N, 3) array of data coordinates
        data_values: (N,) array of data values
        target_coords: (M, 3) array of target coordinates
        variogram_params: Variogram parameters dict
        n_runs: Number of runs to compare (default: 3)
        n_neighbors: Number of neighbors
        max_distance: Maximum search distance
        model_type: Variogram model type
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    
    Returns:
        Tuple of (is_deterministic, report_dict) where:
        - is_deterministic: True if all runs produce identical results
        - report_dict: Detailed report with statistics
    """
    from ..models.kriging3d import ordinary_kriging_3d
    
    results = []
    
    for run in range(n_runs):
        estimates, variances, _ = ordinary_kriging_3d(  # Ignore QA metrics for determinism test
            data_coords=data_coords,
            data_values=data_values,
            target_coords=target_coords,
            variogram_params=variogram_params,
            n_neighbors=n_neighbors,
            max_distance=max_distance,
            model_type=model_type
        )
        results.append((estimates.copy(), variances.copy()))
    
    # Compare all runs to first run
    base_est, base_var = results[0]
    
    max_est_diff = 0.0
    max_var_diff = 0.0
    all_identical = True
    
    for run_idx, (est, var) in enumerate(results[1:], 2):
        est_diff = np.abs(est - base_est)
        var_diff = np.abs(var - base_var)
        
        # Handle NaN values (should be in same positions)
        nan_match = np.array_equal(np.isnan(est), np.isnan(base_est))
        
        # Compare non-NaN values
        valid_mask = ~np.isnan(est) & ~np.isnan(base_est)
        if np.any(valid_mask):
            max_est_diff = max(max_est_diff, float(np.nanmax(est_diff)))
            max_var_diff = max(max_var_diff, float(np.nanmax(var_diff)))
        
        # Check if results are close enough
        if not nan_match:
            all_identical = False
            logger.warning(f"Run {run_idx}: NaN positions differ from run 1")
        elif np.any(valid_mask):
            est_close = np.allclose(est[valid_mask], base_est[valid_mask], rtol=rtol, atol=atol)
            var_close = np.allclose(var[valid_mask], base_var[valid_mask], rtol=rtol, atol=atol)
            if not (est_close and var_close):
                all_identical = False
                logger.warning(f"Run {run_idx}: Results differ from run 1")
    
    report = {
        'n_runs': n_runs,
        'n_targets': len(target_coords),
        'n_data': len(data_coords),
        'max_estimate_difference': max_est_diff,
        'max_variance_difference': max_var_diff,
        'is_deterministic': all_identical,
        'tolerance_rtol': rtol,
        'tolerance_atol': atol,
        'settings': get_determinism_settings()
    }
    
    if all_identical:
        logger.info(f"Kriging determinism verified: {n_runs} runs produced identical results")
    else:
        logger.error(f"Kriging determinism FAILED: max difference in estimates = {max_est_diff:.2e}")
    
    return all_identical, report


def check_neighbor_ordering_determinism(
    data_coords: np.ndarray,
    target_coords: np.ndarray,
    n_neighbors: int = 12,
    n_runs: int = 3
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that neighbor search produces deterministic ordering.
    
    This is a lower-level check that specifically tests the neighbor
    search component, which is a common source of non-determinism.
    
    Args:
        data_coords: (N, 3) array of data coordinates
        target_coords: (M, 3) array of target coordinates
        n_neighbors: Number of neighbors to find
        n_runs: Number of runs to compare
    
    Returns:
        Tuple of (is_deterministic, report_dict)
    """
    from .geostats_utils import NeighborSearcher
    
    searcher = NeighborSearcher(data_coords)
    
    results = []
    for _ in range(n_runs):
        indices, distances = searcher.search(target_coords, n_neighbors=n_neighbors)
        results.append((indices.copy(), distances.copy()))
    
    # Compare all runs
    base_idx, base_dist = results[0]
    all_identical = True
    
    for run_idx, (idx, dist) in enumerate(results[1:], 2):
        if not np.array_equal(idx, base_idx):
            all_identical = False
            n_diff = np.sum(idx != base_idx)
            logger.warning(f"Run {run_idx}: {n_diff} neighbor indices differ from run 1")
    
    report = {
        'n_runs': n_runs,
        'n_targets': len(target_coords),
        'n_data': len(data_coords),
        'n_neighbors': n_neighbors,
        'is_deterministic': all_identical
    }
    
    return all_identical, report

