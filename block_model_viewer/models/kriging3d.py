# ==============================
# File: kriging3d.py (UPDATED)
# ==============================

author = "Block Model Viewer Development Team"

import logging
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:  # SciPy optional
    KDTree = None

# NeighborSearcher import (preferred over direct cKDTree)
try:
    from ..geostats.geostats_utils import NeighborSearcher
    NEIGHBOR_SEARCHER_AVAILABLE = True
except ImportError:
    NEIGHBOR_SEARCHER_AVAILABLE = False
    NeighborSearcher = None

from scipy.spatial.distance import cdist
from scipy.linalg import solve, LinAlgError

from .geostat_results import OrdinaryKrigingResults

# Try to import Numba-accelerated engine
try:
    from .kriging_engine import run_kriging_kernel, NUMBA_AVAILABLE
except ImportError:
    NUMBA_AVAILABLE = False
    run_kriging_kernel = None

# Import standardized visualization module
try:
    from .visualization import create_block_model, export_to_vti, add_property_to_grid
except ImportError:
    create_block_model = None
    export_to_vti = None
    add_property_to_grid = None

logger = logging.getLogger(__name__)

# -----------------------------
# Variogram kernels (γ(h)) - CANONICAL SOURCE
# -----------------------------
# AUDIT FIX: Use unified variogram_model module as single source of truth
# This eliminates duplicate implementations that caused signature inconsistencies
# See: docs/VARIOGRAM_SUBSYSTEM_AUDIT.md (Issue V-002)

try:
    from ..geostats.variogram_model import (
        spherical_model as spherical_variogram,
        exponential_model as exponential_variogram,
        gaussian_model as gaussian_variogram,
        MODEL_MAP as MODEL_GAMMA,
        get_variogram_function,
        VariogramModel,
        validate_variogram_data_match,
        compute_data_hash,
    )
    UNIFIED_VARIOGRAM_AVAILABLE = True
except ImportError:
    # AUDIT FIX (V-NEW-002): Fallback now uses CANONICAL convention (TOTAL sill)
    # Previously used PARTIAL sill which caused environment-dependent behavior
    UNIFIED_VARIOGRAM_AVAILABLE = False
    logger.warning(
        "AUDIT WARNING: Using local variogram functions - unified module not available. "
        "Ensure variogram_model.py is importable for consistent behavior."
    )
    
    def spherical_variogram(h: np.ndarray, range_: float, sill: float, nugget: float = 0.0) -> np.ndarray:
        """
        Spherical variogram model (FALLBACK - CANONICAL CONVENTION).
        
        AUDIT FIX (V-NEW-002): Uses TOTAL sill (sill = nugget + partial_sill)
        
        Parameters
        ----------
        h : array-like
            Distance values
        range_ : float
            Range parameter
        sill : float
            TOTAL sill (nugget + partial sill) - CANONICAL CONVENTION
        nugget : float
            Nugget effect
        
        Returns
        -------
        np.ndarray
            Variogram values γ(h)
        """
        h = np.asarray(h, dtype=float)
        rr = max(range_, 1e-12)
        partial_sill = max(sill - nugget, 0.0)  # CANONICAL: compute partial sill
        t = h / rr
        gamma = np.where(
            h <= rr,
            nugget + partial_sill * (1.5 * t - 0.5 * t**3),
            sill,  # At h >= range, gamma = total sill
        )
        return gamma

    def exponential_variogram(h: np.ndarray, range_: float, sill: float, nugget: float = 0.0) -> np.ndarray:
        """Exponential variogram (FALLBACK - CANONICAL: sill = TOTAL sill)."""
        h = np.asarray(h, dtype=float)
        rr = max(range_, 1e-12)
        partial_sill = max(sill - nugget, 0.0)  # CANONICAL: compute partial sill
        return nugget + partial_sill * (1.0 - np.exp(-3.0 * h / rr))

    def gaussian_variogram(h: np.ndarray, range_: float, sill: float, nugget: float = 0.0) -> np.ndarray:
        """Gaussian variogram (FALLBACK - CANONICAL: sill = TOTAL sill)."""
        h = np.asarray(h, dtype=float)
        rr = max(range_, 1e-12)
        partial_sill = max(sill - nugget, 0.0)  # CANONICAL: compute partial sill
        return nugget + partial_sill * (1.0 - np.exp(-3.0 * (h / rr) ** 2))

    MODEL_GAMMA = {
        "spherical": spherical_variogram,
        "exponential": exponential_variogram,
        "gaussian": gaussian_variogram,
    }

    def get_variogram_function(model_type: str):
        f = MODEL_GAMMA.get(model_type.lower())
        return f if f is not None else spherical_variogram
    
    VariogramModel = None
    validate_variogram_data_match = None
    compute_data_hash = None


# -----------------------------
# Anisotropy transformation
# -----------------------------

def apply_anisotropy(
    coords: np.ndarray, 
    azimuth_deg: float, 
    dip_deg: float,
    major_range: float, 
    minor_range: float, 
    vert_range: float
) -> np.ndarray:
    """
    Applies anisotropic scaling and rotation to coordinates for directional variogram modelling.
    Converts input XYZ coordinates into anisotropy-scaled space where distances are isotropic.
    
    Parameters
    ----------
    coords : np.ndarray
        (N, 3) array of (X, Y, Z) coordinates
    azimuth_deg : float
        Azimuth angle in degrees (0=North, clockwise)
    dip_deg : float
        Dip angle in degrees (0=horizontal, positive down from horizontal)
    major_range : float
        Range in major direction (along strike)
    minor_range : float
        Range in minor direction (across strike)
    vert_range : float
        Range in vertical direction
    
    Returns
    -------
    np.ndarray
        (N, 3) array of transformed coordinates in anisotropy space
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N, 3) array")
    
    # Convert degrees to radians
    az = np.deg2rad(azimuth_deg)
    dip = np.deg2rad(dip_deg)
    
    # Rotation matrices (standard geological convention)
    # First rotate around Z-axis by azimuth
    Rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az),  np.cos(az), 0],
        [0,           0,          1]
    ])
    
    # Then rotate around X-axis by dip
    Rx = np.array([
        [1, 0,           0          ],
        [0, np.cos(dip), -np.sin(dip)],
        [0, np.sin(dip),  np.cos(dip)]
    ])
    
    # Combined rotation: apply azimuth first, then dip
    R = Rx @ Rz
    
    # Apply rotation to coordinates
    rot = coords @ R.T
    
    # Scale coordinates by range ratios (divide by range to normalize)
    # This transforms anisotropic space to isotropic space
    scaled = np.column_stack([
        rot[:, 0] / max(major_range, 1e-9),
        rot[:, 1] / max(minor_range, 1e-9),
        rot[:, 2] / max(vert_range, 1e-9)
    ])
    
    return scaled


# -----------------------------
# Estimation grid helper
# -----------------------------

def create_estimation_grid(
    data_coords: np.ndarray,
    grid_spacing: Tuple[float, float, float] = (10.0, 10.0, 5.0),
    buffer: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_points: int = 50_000,
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Extents
    xmin, ymin, zmin = data_coords.min(axis=0) - np.array(buffer, dtype=float)
    xmax, ymax, zmax = data_coords.max(axis=0) + np.array(buffer, dtype=float)

    dx, dy, dz = map(float, grid_spacing)

    nx = int((xmax - xmin) / dx) + 1
    ny = int((ymax - ymin) / dy) + 1
    nz = int((zmax - zmin) / dz) + 1

    total = nx * ny * nz

    if total > max_points:
        logger.warning(
            f"Requested grid {nx}×{ny}×{nz}={total} exceeds max_points={max_points}. Autoscaling spacing."
        )
        scale = (total / max_points) ** (1.0 / 3.0)
        dx *= scale; dy *= scale; dz *= scale
        nx = int((xmax - xmin) / dx) + 1
        ny = int((ymax - ymin) / dy) + 1
        nz = int((zmax - zmin) / dz) + 1
        total = nx * ny * nz
        if progress_callback:
            progress_callback(15, f"Grid rescaled to {nx}×{ny}×{nz} = {total} points")

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)

    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")

    target_coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    logger.info(f"Created estimation grid: {nx}×{ny}×{nz} = {total} points; spacing dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")
    return grid_x, grid_y, grid_z, target_coords


# -----------------------------
# Fast Numba-Accelerated Ordinary Kriging (3D)
# -----------------------------

def ordinary_kriging_fast(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    target_coords: np.ndarray,
    variogram_params: Dict,
    n_neighbors: int = 12,
    max_distance: Optional[float] = None,
    model_type: str = "spherical",
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-performance Ordinary Kriging using Numba JIT compilation.
    
    This function uses a Numba-compiled kernel for parallel execution across all CPU cores.
    For large datasets (100K+ blocks), this can be 10-100x faster than the standard implementation.
    
    REQUIRES Numba to be installed. Will raise ImportError if Numba is not available.
    
    Parameters
    ----------
    data_coords : np.ndarray
        (N, 3) array of data coordinates
    data_values : np.ndarray
        (N,) array of data values
    target_coords : np.ndarray
        (M, 3) array of target coordinates
    variogram_params : Dict
        Variogram parameters including 'range', 'sill', 'nugget', and optional 'anisotropy'
    n_neighbors : int
        Maximum number of neighbors to use
    max_distance : float, optional
        Maximum search distance (applied before neighbor selection)
    model_type : str
        Variogram model type ('spherical', 'exponential', 'gaussian')
    progress_callback : callable, optional
        Progress callback function(progress: int, message: str)
        
    Returns
    -------
    estimates : np.ndarray
        (M,) array of kriged estimates
    variances : np.ndarray
        (M,) array of kriging variances
        
    Raises
    ------
    ImportError
        If Numba is not available
    """
    if not NUMBA_AVAILABLE or run_kriging_kernel is None:
        raise ImportError(
            "Numba is required for ordinary_kriging_fast(). "
            "Please install it with: pip install numba\n"
            "Or use ordinary_kriging_3d() for the standard implementation."
        )
    
    # Parse parameters
    rng = float(variogram_params["range"])
    sill_total = float(variogram_params["sill"])  # TOTAL sill (including nugget)
    nug = float(variogram_params.get("nugget", 0.0))
    partial_sill = sill_total - nug  # Partial sill for variogram calculation
    
    # Map model string to int code for Numba
    model_map = {"spherical": 0, "exponential": 1, "gaussian": 2}
    model_code = model_map.get(model_type.lower(), 0)
    
    params = np.array([rng, partial_sill, nug, model_code], dtype=np.float64)
    
    # Handle anisotropy if provided
    anisotropy_params = variogram_params.get("anisotropy", None)
    if anisotropy_params:
        azimuth = float(anisotropy_params.get("azimuth", 0.0))
        dip = float(anisotropy_params.get("dip", 0.0))
        major_range = float(anisotropy_params.get("major_range", rng))
        minor_range = float(anisotropy_params.get("minor_range", rng))
        vert_range = float(anisotropy_params.get("vert_range", rng))
        
        # Transform coordinates to anisotropy space
        data_coords_aniso = apply_anisotropy(
            data_coords, azimuth, dip, major_range, minor_range, vert_range
        )
        target_coords_aniso = apply_anisotropy(
            target_coords, azimuth, dip, major_range, minor_range, vert_range
        )
        
        # In transformed space, use normalized range
        params[0] = 1.0  # effective_range = 1.0 in normalized space
        range_geometric_mean = (major_range * minor_range * vert_range) ** (1.0 / 3.0)
        
        logger.info(
            f"Fast anisotropic kriging: azimuth={azimuth:.1f}°, dip={dip:.1f}°, "
            f"ranges(major={major_range:.1f}, minor={minor_range:.1f}, vert={vert_range:.1f})"
        )
    else:
        data_coords_aniso = data_coords
        target_coords_aniso = target_coords
        range_geometric_mean = rng
    
    # Neighbor Search using unified NeighborSearcher
    from ..geostats.geostats_utils import NeighborSearcher
    
    if progress_callback:
        progress_callback(10, "Querying neighbors...")
    
    # Prepare anisotropy params for NeighborSearcher
    searcher_aniso_params = None
    if anisotropy_params:
        searcher_aniso_params = {
            'azimuth': azimuth,
            'dip': dip,
            'major_range': major_range,
            'minor_range': minor_range,
            'vert_range': vert_range
        }
    
    searcher = NeighborSearcher(data_coords, anisotropy_params=searcher_aniso_params)
    neighbor_indices, _ = searcher.search(
        target_coords=target_coords,
        n_neighbors=n_neighbors,
        max_distance=max_distance
    )
    
    m = target_coords.shape[0]
    
    # Get transformed coordinates for kernel (if anisotropy was applied)
    if anisotropy_params:
        data_coords_for_kernel = searcher.get_transformed_coords()
        target_coords_for_kernel = target_coords_aniso  # Already computed above
    else:
        data_coords_for_kernel = data_coords
        target_coords_for_kernel = target_coords
    
    # Run optimized kernel
    if progress_callback:
        progress_callback(30, f"Running Numba-accelerated kriging on {m:,} blocks...")
    
    estimates, variances = run_kriging_kernel(
        target_coords_for_kernel,
        neighbor_indices,
        data_coords_for_kernel,
        data_values,
        params
    )
    
    if progress_callback:
        progress_callback(100, "Kriging complete")
    
    # Logging
    ok = ~np.isnan(estimates)
    n_ok = int(ok.sum())
    logger.info(f"Fast kriging completed: {n_ok}/{m} valid estimates")
    if n_ok:
        vals = estimates[ok]
        logger.info(f"Estimate range [{vals.min():.3f}, {vals.max():.3f}], mean {vals.mean():.3f}, std {vals.std():.3f}")
    
    return estimates, variances


# -----------------------------
# Ordinary Kriging (3D) - Standard Implementation
# -----------------------------

def ordinary_kriging_3d(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    target_coords: np.ndarray,
    variogram_params: Dict,
    n_neighbors: int = 12,
    max_distance: Optional[float] = None,
    model_type: str = "spherical",
    progress_callback=None,
    search_passes: Optional[list] = None,
    compute_qa_metrics: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Ordinary kriging using covariance form: C(h) = sill_total − γ(h).

    Expect **total** sill in variogram_params['sill'] (i.e., nugget + partial_sill).

    PROFESSIONAL ENHANCEMENT: Supports multi-pass search strategy for JORC/NI 43-101 compliance.

    Parameters
    ----------
    data_coords : np.ndarray
        (N, 3) array of data coordinates (X, Y, Z)
    data_values : np.ndarray
        (N,) array of data values
    target_coords : np.ndarray
        (M, 3) array of target coordinates
    variogram_params : dict
        Variogram parameters: range, sill, nugget, anisotropy
    n_neighbors : int
        Max neighbors (legacy single-pass parameter)
    max_distance : float, optional
        Max search distance
    model_type : str
        Variogram model type
    progress_callback : callable, optional
        Progress callback function
    search_passes : list of dict, optional
        Multi-pass search configuration. Each dict: {min_neighbors, max_neighbors, ellipsoid_multiplier}
        If None, uses single-pass search with n_neighbors
    compute_qa_metrics : bool
        If True, compute QA metrics (kriging efficiency, slope of regression, etc.)

    Returns
    -------
    estimates : np.ndarray
        (M,) array of kriging estimates
    variances : np.ndarray
        (M,) array of kriging variances
    qa_metrics : dict or None
        QA metrics dict if compute_qa_metrics=True, else None.
        Contains: kriging_efficiency, slope_of_regression, n_samples, pass_number,
                  distance_to_nearest, pct_negative_weights
    """
    rng = float(variogram_params["range"])  # practical range convention in γ (used as reference)
    sill_total = float(variogram_params["sill"])  # TOTAL sill (including nugget)
    nug = float(variogram_params.get("nugget", 0.0))

    gamma_fun = get_variogram_function(model_type)
    
    # Handle anisotropy if provided
    anisotropy_params = variogram_params.get("anisotropy", None)
    if anisotropy_params:
        azimuth = float(anisotropy_params.get("azimuth", 0.0))
        dip = float(anisotropy_params.get("dip", 0.0))
        major_range = float(anisotropy_params.get("major_range", rng))
        minor_range = float(anisotropy_params.get("minor_range", rng))
        vert_range = float(anisotropy_params.get("vert_range", rng))
        
        # Transform coordinates to anisotropy space
        data_coords_aniso = apply_anisotropy(
            data_coords, azimuth, dip, major_range, minor_range, vert_range
        )
        target_coords_aniso = apply_anisotropy(
            target_coords, azimuth, dip, major_range, minor_range, vert_range
        )
        
        # In transformed anisotropic space, coordinates are normalized by ranges
        # Use a normalized range value (1.0) for the variogram function
        # This works because after scaling, distances in transformed space are isotropic
        # and a range of 1.0 represents the unit distance in that space
        effective_range = 1.0
        
        # Calculate geometric mean for scaling max_distance from original to transformed space
        range_geometric_mean = (major_range * minor_range * vert_range) ** (1.0 / 3.0)
        
        logger.info(
            f"Anisotropic kriging: azimuth={azimuth:.1f}°, dip={dip:.1f}°, "
            f"ranges(major={major_range:.1f}, minor={minor_range:.1f}, vert={vert_range:.1f})"
        )
    else:
        # Isotropic case
        data_coords_aniso = data_coords
        target_coords_aniso = target_coords
        effective_range = rng
        range_geometric_mean = rng  # For consistency in max_distance scaling

    # CRITICAL FIX: Use NeighborSearcher for unified neighbor search
    # This consolidates duplicate neighbor search logic and handles anisotropy correctly
    use_neighbor_searcher = NEIGHBOR_SEARCHER_AVAILABLE
    use_tree = KDTree is not None and not use_neighbor_searcher  # Fallback to direct KDTree
    
    # Build neighbor search structure
    searcher = None
    tree = None
    
    if use_neighbor_searcher:
        # Use unified NeighborSearcher (handles anisotropy automatically)
        searcher = NeighborSearcher(
            data_coords,  # Original coordinates (anisotropy handled internally)
            anisotropy_params=anisotropy_params if anisotropy_params else None
        )
        logger.debug("Using NeighborSearcher for neighbor search")
    elif use_tree:
        # Fallback to direct KDTree (legacy support)
        tree = KDTree(data_coords_aniso)
        logger.debug("Using direct KDTree for neighbor search (legacy)")
    else:
        logger.warning("No neighbor search available; falling back to full distance scan (very slow).")

    m = target_coords.shape[0]
    estimates = np.full(m, np.nan, dtype=float)
    variances = np.full(m, np.nan, dtype=float)

    # QA metrics arrays (initialized if compute_qa_metrics=True)
    qa_arrays = {}
    if compute_qa_metrics:
        qa_arrays = {
            'kriging_efficiency': np.full(m, np.nan, dtype=float),
            'slope_of_regression': np.full(m, np.nan, dtype=float),
            'n_samples': np.full(m, 0, dtype=int),
            'pass_number': np.full(m, 0, dtype=int),  # 0 = unestimated, 1/2/3 = pass number
            'distance_to_nearest': np.full(m, np.nan, dtype=float),
            'pct_negative_weights': np.full(m, np.nan, dtype=float),
        }

    # Multi-pass configuration
    use_multi_pass = search_passes is not None and len(search_passes) > 0
    if use_multi_pass:
        logger.info(f"Multi-pass search enabled: {len(search_passes)} passes configured")
        for idx, pass_cfg in enumerate(search_passes, 1):
            logger.info(
                f"  Pass {idx}: min={pass_cfg['min_neighbors']}, "
                f"max={pass_cfg['max_neighbors']}, "
                f"ellipsoid_mult={pass_cfg['ellipsoid_multiplier']:.2f}"
            )
    else:
        # Legacy single-pass mode
        search_passes = [{'min_neighbors': 3, 'max_neighbors': n_neighbors, 'ellipsoid_multiplier': 1.0}]
        logger.info(f"Single-pass search (legacy): min=3, max={n_neighbors}")

    # Compute data variance (for kriging efficiency calculation)
    data_variance = float(np.var(data_values))

    report_every = max(1, m // 20)  # Report every 5% instead of 10%

    for i in range(m):
        p_aniso = target_coords_aniso[i]
        p_original = target_coords[i]

        # Multi-pass search loop
        estimation_success = False
        for pass_idx, pass_config in enumerate(search_passes, 1):
            min_neighbors_pass = pass_config['min_neighbors']
            max_neighbors_pass = pass_config['max_neighbors']
            ellipsoid_mult = pass_config['ellipsoid_multiplier']

            # Scale max_distance by ellipsoid multiplier for this pass
            max_distance_pass = max_distance * ellipsoid_mult if max_distance is not None else None

            # Neighbour search (using unified NeighborSearcher or fallback)
            if searcher is not None:
                # Use NeighborSearcher (handles anisotropy and max_distance automatically)
                target_single = target_coords[i:i+1]  # Single target as (1, 3) array
                nbr_indices, nbr_distances = searcher.search(
                    target_single,
                    n_neighbors=max_neighbors_pass,
                    max_distance=max_distance_pass
                )
                # Extract results (first row since we queried one point)
                nbr_idx = nbr_indices[0]
                d = nbr_distances[0]
                # Filter out -1 padding and inf distances
                valid_mask = nbr_idx >= 0
                nbr_idx = nbr_idx[valid_mask]
                d = d[valid_mask]

                if len(nbr_idx) < min_neighbors_pass:
                    # Try next pass
                    continue
            elif use_tree:
                if max_distance_pass is not None:
                    # Transform max_distance to anisotropic space
                    if anisotropy_params:
                        # Scale max_distance by geometric mean of ranges to transform to normalized space
                        max_dist_aniso = float(max_distance_pass) / range_geometric_mean
                    else:
                        max_dist_aniso = float(max_distance_pass)

                    idxs = tree.query_ball_point(p_aniso, r=max_dist_aniso)
                    if not idxs or len(idxs) < min_neighbors_pass:
                        # Try next pass
                        continue
                    # take closest max_neighbors_pass among those
                    # DETERMINISTIC: Use lexsort for stable tie-breaking by original index
                    pts_aniso = data_coords_aniso[idxs]
                    d = np.linalg.norm(pts_aniso - p_aniso, axis=1)
                    idxs_arr = np.array(idxs, dtype=int)
                    order = np.lexsort((idxs_arr, d))[:max_neighbors_pass]  # Sort by distance, then by index
                    nbr_idx = idxs_arr[order]
                else:
                    d, nbr_idx = tree.query(p_aniso, k=min(max_neighbors_pass, len(data_coords)))
                    if np.isscalar(d):  # k==1 returns scalar
                        d = np.array([d]); nbr_idx = np.array([nbr_idx])
                    else:
                        # DETERMINISTIC: Re-sort with stable tie-breaking
                        nbr_idx = np.asarray(nbr_idx, dtype=int)
                        d = np.asarray(d)
                        order = np.lexsort((nbr_idx, d))
                        nbr_idx = nbr_idx[order]
                        d = d[order]

                if len(nbr_idx) < min_neighbors_pass:
                    continue
            else:
                d_all = np.linalg.norm(data_coords_aniso - p_aniso, axis=1)
                all_indices = np.arange(len(d_all), dtype=int)
                if max_distance_pass is not None:
                    # Transform max_distance to anisotropic space
                    if anisotropy_params:
                        # Scale max_distance by geometric mean of ranges to transform to normalized space
                        max_dist_aniso = float(max_distance_pass) / range_geometric_mean
                    else:
                        max_dist_aniso = float(max_distance_pass)

                    mask = d_all <= max_dist_aniso
                    if not np.any(mask) or mask.sum() < min_neighbors_pass:
                        continue
                    # DETERMINISTIC: Use lexsort for stable tie-breaking
                    masked_d = d_all[mask]
                    masked_idx = all_indices[mask]
                    order = np.lexsort((masked_idx, masked_d))[:max_neighbors_pass]
                    nbr_idx = masked_idx[order]
                else:
                    # DETERMINISTIC: Use lexsort for stable tie-breaking by original index
                    order = np.lexsort((all_indices, d_all))[:max_neighbors_pass]
                    nbr_idx = all_indices[order]

                if len(nbr_idx) < min_neighbors_pass:
                    continue

            # Sufficient neighbors found - proceed with kriging
            n = len(nbr_idx)

            P_aniso = data_coords_aniso[nbr_idx]
            v = data_values[nbr_idx]

            # Build covariance matrix K (n+1,n+1)
            # Using covariance C = sill_total − γ
            # Use anisotropic distances and effective range
            D = cdist(P_aniso, P_aniso)
            C = sill_total - gamma_fun(D, effective_range, sill_total - nug, nug)

            K = np.zeros((n + 1, n + 1), dtype=float)
            K[:n, :n] = C
            K[:n, n] = 1.0
            K[n, :n] = 1.0
            # K[n, n] = 0

            # RHS is covariance between target and data, plus unbiasedness term 1
            d0 = np.linalg.norm(P_aniso - p_aniso, axis=1)
            c0 = sill_total - gamma_fun(d0, effective_range, sill_total - nug, nug)
            rhs = np.append(c0, 1.0)

            try:
                # Add small regularization to diagonal for numerical stability
                # This prevents ill-conditioned matrices from producing extreme values
                K_reg = K.copy()
                reg_value = 1e-10 * np.abs(K[:n, :n]).max()  # Scale regularization to matrix values
                np.fill_diagonal(K_reg[:n, :n], K_reg[:n, :n].diagonal() + reg_value)

                w_mu = solve(K_reg, rhs, assume_a='sym')  # [w_1..w_n, mu]
            except LinAlgError:
                # Least-squares fallback with regularization
                w_mu = np.linalg.lstsq(K, rhs, rcond=1e-10)[0]

            w = w_mu[:n]
            mu = w_mu[n]

            # Estimate and variance
            est = float(np.dot(w, v))

            # Sanity check: if estimate is extreme, fall back to simple average
            data_range = v.max() - v.min()
            data_mean = v.mean()
            if data_range > 0:
                # Check if estimate is unreasonably far from data range
                if abs(est - data_mean) > 10 * data_range:
                    est = data_mean  # Fall back to mean of neighbors

            # σ²_K = C(0) − w^T c0 − μ
            var = float(max(0.0, sill_total - float(np.dot(w, c0)) - mu))

            estimates[i] = est
            variances[i] = var

            # Compute QA metrics if requested
            if compute_qa_metrics:
                # Kriging Efficiency: KE = 1 - (kriging_variance / data_variance)
                if data_variance > 0:
                    qa_arrays['kriging_efficiency'][i] = 1.0 - (var / data_variance)
                else:
                    qa_arrays['kriging_efficiency'][i] = 0.0

                # Slope of Regression: SoR ≈ sum(weights) (should be ~1.0 for OK)
                # More precise: SoR = sum(weights * correlation_coefficient)
                # Simplified: use sum of weights (OK constraint ensures sum(w)=1, so SoR ~ 1)
                qa_arrays['slope_of_regression'][i] = float(np.sum(w))

                # Number of samples used
                qa_arrays['n_samples'][i] = n

                # Pass number
                qa_arrays['pass_number'][i] = pass_idx

                # Distance to nearest sample (in original space)
                P_original = data_coords[nbr_idx]
                distances_original = np.linalg.norm(P_original - p_original, axis=1)
                qa_arrays['distance_to_nearest'][i] = float(np.min(distances_original))

                # Percentage of negative weights
                n_negative = np.sum(w < 0)
                qa_arrays['pct_negative_weights'][i] = 100.0 * n_negative / n if n > 0 else 0.0

            # Mark success and break from multi-pass loop
            estimation_success = True
            break  # Exit pass loop

        # If no pass succeeded, log and continue to next block
        if not estimation_success and progress_callback and (i % report_every == 0 or i == m - 1):
            pct = (i + 1) * 100 // m
            progress_callback(int(pct), f"Kriging: {pct:.0f}% ({i+1}/{m})")
            continue

        if progress_callback and (i % report_every == 0 or i == m - 1):
            pct = (i + 1) * 100 // m
            progress_callback(int(pct), f"Kriging: {pct:.0f}% ({i+1}/{m})")

    # Logging
    ok = ~np.isnan(estimates)
    n_ok = int(ok.sum())
    logger.info(f"Kriging completed: {n_ok}/{m} valid estimates ({n_ok*100.0/m:.1f}%)")
    if n_ok:
        vals = estimates[ok]
        logger.info(f"Estimate range [{vals.min():.3f}, {vals.max():.3f}], mean {vals.mean():.3f}, std {vals.std():.3f}")

    # Multi-pass summary
    if use_multi_pass and compute_qa_metrics:
        pass_counts = {p: int(np.sum(qa_arrays['pass_number'] == p)) for p in range(1, len(search_passes) + 1)}
        logger.info("Multi-pass search summary:")
        for pass_num, count in pass_counts.items():
            pct = count * 100.0 / m if m > 0 else 0
            logger.info(f"  Pass {pass_num}: {count} blocks ({pct:.1f}%)")
        unestimated = int(np.sum(qa_arrays['pass_number'] == 0))
        logger.info(f"  Unestimated: {unestimated} blocks ({unestimated*100.0/m:.1f}%)")

    # QA metrics summary
    if compute_qa_metrics and n_ok > 0:
        ke_valid = qa_arrays['kriging_efficiency'][ok]
        sor_valid = qa_arrays['slope_of_regression'][ok]
        neg_wt_valid = qa_arrays['pct_negative_weights'][ok]

        logger.info("QA Metrics Summary:")
        logger.info(f"  Kriging Efficiency: mean={np.nanmean(ke_valid):.3f}, min={np.nanmin(ke_valid):.3f}, max={np.nanmax(ke_valid):.3f}")
        logger.info(f"  Slope of Regression: mean={np.nanmean(sor_valid):.3f}, min={np.nanmin(sor_valid):.3f}, max={np.nanmax(sor_valid):.3f}")
        logger.info(f"  Negative Weights %: mean={np.nanmean(neg_wt_valid):.1f}%, max={np.nanmax(neg_wt_valid):.1f}%")

        # Flag concerns
        n_low_ke = np.sum(ke_valid < 0.3)
        if n_low_ke > 0:
            logger.warning(f"⚠ {n_low_ke} blocks ({n_low_ke*100.0/n_ok:.1f}%) have Kriging Efficiency < 0.3")

        n_bad_sor = np.sum((sor_valid < 0.8) | (sor_valid > 1.2))
        if n_bad_sor > 0:
            logger.warning(f"⚠ {n_bad_sor} blocks ({n_bad_sor*100.0/n_ok:.1f}%) have Slope of Regression outside [0.8, 1.2]")

        n_high_neg = np.sum(neg_wt_valid > 20.0)
        if n_high_neg > 0:
            logger.warning(f"⚠ {n_high_neg} blocks ({n_high_neg*100.0/n_ok:.1f}%) have >20% negative weights")

    # Package QA metrics for return
    qa_metrics = None
    if compute_qa_metrics:
        qa_metrics = {
            'kriging_efficiency': qa_arrays['kriging_efficiency'],
            'slope_of_regression': qa_arrays['slope_of_regression'],
            'n_samples': qa_arrays['n_samples'],
            'pass_number': qa_arrays['pass_number'],
            'distance_to_nearest': qa_arrays['distance_to_nearest'],
            'pct_negative_weights': qa_arrays['pct_negative_weights'],
        }

    return estimates, variances, qa_metrics


# -----------------------------
# Export helpers
# -----------------------------

def export_kriging_results(
    output_path: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    estimates: np.ndarray,
    variances: np.ndarray,
    format: str = 'csv'
):
    fmt = format.lower()
    if fmt == 'csv':
        coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        df = pd.DataFrame({
            'X': coords[:, 0], 'Y': coords[:, 1], 'Z': coords[:, 2],
            'ESTIMATE': estimates.ravel(), 'VARIANCE': variances.ravel()
        }).dropna()
        df.to_csv(output_path, index=False)
        logger.info(f"Exported kriging CSV: {output_path} ({len(df)} rows)")
        return

    if fmt == 'vtk' or fmt == 'vti':
        if create_block_model is None or export_to_vti is None:
            raise ImportError("Visualization module not available. Cannot export to VTK/VTI.")
        
        # Determine grid dimensions from meshgrid
        # grid_x, grid_y, grid_z are from meshgrid with indexing='ij', so shape is (nx, ny, nz)
        nx = grid_x.shape[0]
        ny = grid_x.shape[1]
        nz = grid_x.shape[2]
        
        # Get origin and spacing from grid coordinates
        xmin = float(grid_x[0, 0, 0])
        ymin = float(grid_y[0, 0, 0])
        zmin = float(grid_z[0, 0, 0])
        
        # Calculate spacing (assuming uniform spacing)
        dx = float(grid_x[1, 0, 0] - grid_x[0, 0, 0]) if nx > 1 else 1.0
        dy = float(grid_y[0, 1, 0] - grid_y[0, 0, 0]) if ny > 1 else 1.0
        dz = float(grid_z[0, 0, 1] - grid_z[0, 0, 0]) if nz > 1 else 1.0
        
        # 1. Reshape flat arrays to 3D
        # Kriging grid generation uses indexing='ij' -> (nx, ny, nz)
        est_3d = estimates.reshape((nx, ny, nz), order='C')
        var_3d = variances.reshape((nx, ny, nz), order='C')
        
        # 2. Transpose to Z-Y-X standard for the visualizer
        # (nx, ny, nz) -> (nz, ny, nx)
        est_zyx = est_3d.transpose(2, 1, 0)
        var_zyx = var_3d.transpose(2, 1, 0)
        
        # 3. Create Grid using standardized visualization module
        grid = create_block_model(
            values=est_zyx,
            origin=(xmin, ymin, zmin),
            spacing=(dx, dy, dz),
            dims=(nx, ny, nz),
            name="Estimate"
        )
        
        # Add Variance as a second property
        add_property_to_grid(grid, var_zyx, "Variance")
        
        # Save as VTI (modern XML format) or VTK
        if fmt == 'vti' or output_path.endswith('.vti'):
            output_path_vti = output_path if output_path.endswith('.vti') else output_path.replace('.vtk', '.vti')
            export_to_vti(grid, output_path_vti)
        else:
            # Fallback to VTK if explicitly requested
            grid.save(output_path)
            logger.info(f"Exported kriging VTK: {output_path}")
        
        return

    raise ValueError(f"Unknown export format: {format}")


# -----------------------------
# Enhanced Ordinary Kriging with Full Professional Outputs
# -----------------------------

def ordinary_kriging_3d_full(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    target_coords: np.ndarray,
    variogram_params: Dict,
    n_neighbors: int = 12,
    max_distance: Optional[float] = None,
    model_type: str = "spherical",
    min_neighbors: int = 3,
    block_variance: Optional[float] = None,
    search_pass: int = 1,
    compute_weights: bool = False,
    progress_callback=None,
) -> OrdinaryKrigingResults:
    """
    Enhanced Ordinary Kriging with comprehensive professional output attributes.
    
    Returns all standard attributes produced by professional geostatistical software
    (Leapfrog, Isatis, Datamine, Surpac).
    
    Parameters
    ----------
    data_coords : np.ndarray
        (N, 3) array of data coordinates
    data_values : np.ndarray
        (N,) array of data values
    target_coords : np.ndarray
        (M, 3) array of target coordinates
    variogram_params : Dict
        Variogram parameters including 'range', 'sill', 'nugget', and optional 'anisotropy'
    n_neighbors : int
        Maximum number of neighbors to use
    max_distance : float, optional
        Maximum search distance
    model_type : str
        Variogram model type ('spherical', 'exponential', 'gaussian')
    min_neighbors : int
        Minimum number of neighbors required
    block_variance : float, optional
        Block variance for computing kriging efficiency (if None, uses sill_total)
    search_pass : int
        Search pass number (1, 2, 3, etc.)
    compute_weights : bool
        Whether to store weight vectors (memory intensive)
    progress_callback : callable, optional
        Progress callback function(progress: int, message: str)
    
    Returns
    -------
    OrdinaryKrigingResults
        Comprehensive results with all professional attributes
    """
    rng = float(variogram_params["range"])
    sill_total = float(variogram_params["sill"])
    nug = float(variogram_params.get("nugget", 0.0))
    
    # Use block_variance for KE calculation if provided, otherwise use sill_total
    block_var = block_variance if block_variance is not None else sill_total
    
    gamma_fun = get_variogram_function(model_type)
    
    # Handle anisotropy
    anisotropy_params = variogram_params.get("anisotropy", None)
    if anisotropy_params:
        azimuth = float(anisotropy_params.get("azimuth", 0.0))
        dip = float(anisotropy_params.get("dip", 0.0))
        major_range = float(anisotropy_params.get("major_range", rng))
        minor_range = float(anisotropy_params.get("minor_range", rng))
        vert_range = float(anisotropy_params.get("vert_range", rng))
        
        data_coords_aniso = apply_anisotropy(
            data_coords, azimuth, dip, major_range, minor_range, vert_range
        )
        target_coords_aniso = apply_anisotropy(
            target_coords, azimuth, dip, major_range, minor_range, vert_range
        )
        effective_range = 1.0
        range_geometric_mean = (major_range * minor_range * vert_range) ** (1.0 / 3.0)
    else:
        data_coords_aniso = data_coords
        target_coords_aniso = target_coords
        effective_range = rng
        range_geometric_mean = rng
        azimuth = 0.0
        dip = 0.0
        major_range = rng
        minor_range = rng
        vert_range = rng
    
    # CRITICAL FIX: Use NeighborSearcher for unified neighbor search
    use_neighbor_searcher = NEIGHBOR_SEARCHER_AVAILABLE
    use_tree = KDTree is not None and not use_neighbor_searcher
    
    searcher = None
    tree = None
    
    if use_neighbor_searcher:
        searcher = NeighborSearcher(
            data_coords,
            anisotropy_params=anisotropy_params if anisotropy_params else None
        )
        logger.debug("Using NeighborSearcher for neighbor search")
    elif use_tree:
        tree = KDTree(data_coords_aniso)
        logger.debug("Using direct KDTree for neighbor search (legacy)")
    else:
        logger.warning("No neighbor search available; falling back to full distance scan (very slow).")
    
    m = target_coords.shape[0]
    
    # Initialize output arrays
    estimates = np.full(m, np.nan, dtype=float)
    status = np.full(m, 0, dtype=int)  # 0=OK, 1=no samples, 2=min samples not met
    kriging_mean = np.full(m, np.nan, dtype=float)
    kriging_variance = np.full(m, np.nan, dtype=float)
    kriging_efficiency = np.full(m, np.nan, dtype=float)
    slope_of_regression = np.full(m, np.nan, dtype=float)
    lagrange_multiplier = np.full(m, np.nan, dtype=float)
    num_samples = np.zeros(m, dtype=int)
    sum_weights = np.full(m, np.nan, dtype=float)
    sum_negative_weights = np.zeros(m, dtype=float)
    min_distance = np.full(m, np.nan, dtype=float)
    avg_distance = np.full(m, np.nan, dtype=float)
    nearest_sample_id = np.full(m, -1, dtype=int)
    num_duplicates_removed = np.zeros(m, dtype=int)
    search_pass_array = np.full(m, search_pass, dtype=int)
    
    # Anisotropy arrays
    ellipsoid_rotation = np.full(m, azimuth, dtype=float) if anisotropy_params else None
    anisotropy_x = np.full(m, major_range, dtype=float) if anisotropy_params else None
    anisotropy_y = np.full(m, minor_range, dtype=float) if anisotropy_params else None
    anisotropy_z = np.full(m, vert_range, dtype=float) if anisotropy_params else None
    
    # Optional weight storage
    weight_vectors = [] if compute_weights else None
    sample_coords_matrix = [] if compute_weights else None
    
    # Search volume (approximate as sphere volume)
    search_volume = np.full(m, np.nan, dtype=float)
    if max_distance is not None:
        search_volume[:] = (4.0 / 3.0) * np.pi * (max_distance ** 3)
    
    report_every = max(1, m // 10)
    
    # FREEZE PROTECTION: Log worker start
    logger.debug("WORKER START: ordinary_kriging_3d_full")
    logger.debug(f"Processing {m:,} target blocks")
    
    # Track duplicate samples (same coordinates)
    from collections import defaultdict
    coord_to_indices = defaultdict(list)
    for idx, coord in enumerate(data_coords):
        coord_key = tuple(np.round(coord, 6))
        coord_to_indices[coord_key].append(idx)
    
    for i in range(m):
        p_aniso = target_coords_aniso[i]
        p_orig = target_coords[i]
        
        # Neighbor search (using unified NeighborSearcher or fallback)
        if searcher is not None:
            # Use NeighborSearcher (handles anisotropy and max_distance automatically)
            target_single = target_coords[i:i+1]
            nbr_indices, nbr_distances = searcher.search(
                target_single,
                n_neighbors=n_neighbors,
                max_distance=max_distance
            )
            nbr_idx = nbr_indices[0]
            d = nbr_distances[0]
            valid_mask = nbr_idx >= 0
            nbr_idx = nbr_idx[valid_mask]
            d = d[valid_mask]
            
            if len(nbr_idx) < min_neighbors:
                status[i] = 2  # min samples not met
                continue
        elif use_tree:
            if max_distance is not None:
                max_dist_aniso = float(max_distance) / range_geometric_mean if anisotropy_params else float(max_distance)
                idxs = tree.query_ball_point(p_aniso, r=max_dist_aniso)
                if not idxs:
                    status[i] = 1  # no samples
                    continue
                # DETERMINISTIC: Use lexsort for stable tie-breaking by original index
                pts_aniso = data_coords_aniso[idxs]
                d = np.linalg.norm(pts_aniso - p_aniso, axis=1)
                idxs_arr = np.array(idxs, dtype=int)
                order = np.lexsort((idxs_arr, d))[:n_neighbors]  # Sort by distance, then by index
                nbr_idx = idxs_arr[order]
            else:
                d, nbr_idx = tree.query(p_aniso, k=min(n_neighbors, len(data_coords)))
                if np.isscalar(d):
                    d = np.array([d])
                    nbr_idx = np.array([nbr_idx])
                else:
                    # DETERMINISTIC: Re-sort with stable tie-breaking
                    nbr_idx = np.asarray(nbr_idx, dtype=int)
                    d = np.asarray(d)
                    order = np.lexsort((nbr_idx, d))
                    nbr_idx = nbr_idx[order]
                    d = d[order]
        else:
            d_all = np.linalg.norm(data_coords_aniso - p_aniso, axis=1)
            all_indices = np.arange(len(d_all), dtype=int)
            if max_distance is not None:
                max_dist_aniso = float(max_distance) / range_geometric_mean if anisotropy_params else float(max_distance)
                mask = d_all <= max_dist_aniso
                if not np.any(mask):
                    status[i] = 1
                    continue
                # DETERMINISTIC: Use lexsort for stable tie-breaking
                masked_d = d_all[mask]
                masked_idx = all_indices[mask]
                order = np.lexsort((masked_idx, masked_d))[:n_neighbors]
                nbr_idx = masked_idx[order]
                d = masked_d[order]
            else:
                # DETERMINISTIC: Use lexsort for stable tie-breaking by original index
                order = np.lexsort((all_indices, d_all))[:n_neighbors]
                nbr_idx = all_indices[order]
                d = d_all[order]
        
        n = len(nbr_idx)
        if n < min_neighbors:
            status[i] = 2  # min samples not met
            continue
        
        # Check for duplicates
        nbr_coords_orig = data_coords[nbr_idx]
        unique_coords = []
        unique_indices = []
        seen_coords = set()
        for idx, coord in zip(nbr_idx, nbr_coords_orig):
            coord_key = tuple(np.round(coord, 6))
            if coord_key not in seen_coords:
                unique_coords.append(coord)
                unique_indices.append(idx)
                seen_coords.add(coord_key)
        
        n_duplicates = n - len(unique_indices)
        num_duplicates_removed[i] = n_duplicates
        
        if len(unique_indices) < min_neighbors:
            status[i] = 2
            continue
        
        # Use unique neighbors
        nbr_idx_unique = np.array(unique_indices)
        n_unique = len(nbr_idx_unique)
        P_aniso = data_coords_aniso[nbr_idx_unique]
        v = data_values[nbr_idx_unique]
        
        # Compute distances in original space for MinD, AvgD
        d_orig = np.linalg.norm(data_coords[nbr_idx_unique] - p_orig, axis=1)
        min_distance[i] = float(d_orig.min())
        avg_distance[i] = float(d_orig.mean())
        nearest_sample_id[i] = int(nbr_idx_unique[np.argmin(d_orig)])
        
        # Build covariance matrix
        D = cdist(P_aniso, P_aniso)
        C = sill_total - gamma_fun(D, effective_range, sill_total - nug, nug)
        
        K = np.zeros((n_unique + 1, n_unique + 1), dtype=float)
        K[:n_unique, :n_unique] = C
        K[:n_unique, n_unique] = 1.0
        K[n_unique, :n_unique] = 1.0
        
        d0 = np.linalg.norm(P_aniso - p_aniso, axis=1)
        c0 = sill_total - gamma_fun(d0, effective_range, sill_total - nug, nug)
        rhs = np.append(c0, 1.0)
        
        try:
            # Add small regularization to diagonal for numerical stability
            K_reg = K.copy()
            reg_value = 1e-10 * np.abs(K[:n_unique, :n_unique]).max()
            np.fill_diagonal(K_reg[:n_unique, :n_unique], K_reg[:n_unique, :n_unique].diagonal() + reg_value)
            
            w_mu = solve(K_reg, rhs, assume_a='sym')
        except LinAlgError:
            w_mu = np.linalg.lstsq(K, rhs, rcond=1e-10)[0]
        
        w = w_mu[:n_unique]
        mu = w_mu[n_unique]
        
        # Core outputs
        est = float(np.dot(w, v))
        
        # Sanity check: if estimate is extreme, fall back to simple average
        data_range = v.max() - v.min() if len(v) > 1 else 1.0
        data_mean = v.mean()
        if data_range > 0 and abs(est - data_mean) > 10 * data_range:
            est = data_mean  # Fall back to mean of neighbors
            
        var = float(max(0.0, sill_total - float(np.dot(w, c0)) - mu))
        
        estimates[i] = est
        kriging_variance[i] = var
        lagrange_multiplier[i] = mu
        
        # Kriging mean (local mean implied by OK system)
        # For OK, this is the estimate itself (since weights sum to 1)
        kriging_mean[i] = est
        
        # Kriging efficiency: KE = 1 - (KV / block_variance)
        if block_var > 0:
            kriging_efficiency[i] = 1.0 - (var / block_var)
        else:
            kriging_efficiency[i] = np.nan
        
        # Slope of Regression (SoR) - measure of conditional bias
        # SoR = Cov(Z*, Z) / Var(Z*) where Z* is estimate, Z is true value
        # Approximate using sample variance and covariance
        sample_var = float(np.var(v))
        if sample_var > 0:
            # Approximate covariance as weighted covariance
            weighted_mean = est
            cov_approx = float(np.sum(w * (v - weighted_mean) * (v - np.mean(v))))
            sor = cov_approx / max(var, 1e-10) if var > 0 else 1.0
            slope_of_regression[i] = sor
        else:
            slope_of_regression[i] = 1.0
        
        # Neighbor attributes
        num_samples[i] = n_unique
        sum_weights[i] = float(np.sum(w))
        sum_negative_weights[i] = float(np.sum(w[w < 0]))
        
        # Store weights if requested
        if compute_weights:
            w_full = np.zeros(len(data_coords))
            w_full[nbr_idx_unique] = w
            weight_vectors.append(w_full)
            sample_coords_matrix.append(data_coords[nbr_idx_unique])
        
        if progress_callback and (i % report_every == 0 or i == m - 1):
            pct = (i + 1) * 100 // m
            progress_callback(int(pct), f"Kriging: {pct:.0f}% ({i+1}/{m})")
    
    # Convert weight lists to arrays if computed
    if compute_weights and weight_vectors:
        weight_vectors = np.array(weight_vectors)
        sample_coords_matrix = np.array(sample_coords_matrix)
    
    # Create results object
    results = OrdinaryKrigingResults(
        estimates=estimates,
        status=status,
        kriging_mean=kriging_mean,
        kriging_variance=kriging_variance,
        kriging_efficiency=kriging_efficiency,
        slope_of_regression=slope_of_regression,
        lagrange_multiplier=lagrange_multiplier,
        num_samples=num_samples,
        sum_weights=sum_weights,
        sum_negative_weights=sum_negative_weights,
        min_distance=min_distance,
        avg_distance=avg_distance,
        nearest_sample_id=nearest_sample_id,
        num_duplicates_removed=num_duplicates_removed,
        search_pass=search_pass_array,
        search_volume=search_volume,
        ellipsoid_rotation=ellipsoid_rotation,
        anisotropy_x=anisotropy_x,
        anisotropy_y=anisotropy_y,
        anisotropy_z=anisotropy_z,
        weight_vectors=weight_vectors,
        sample_coords_matrix=sample_coords_matrix,
        metadata={
            'variogram_params': variogram_params,
            'n_neighbors': n_neighbors,
            'max_distance': max_distance,
            'model_type': model_type,
            'min_neighbors': min_neighbors,
        }
    )
    
    # FREEZE PROTECTION: Log worker end
    logger.debug("WORKER END: ordinary_kriging_3d_full")
    logger.info(f"Enhanced OK completed: {np.sum(status == 0)}/{m} valid estimates")
    
    return results
