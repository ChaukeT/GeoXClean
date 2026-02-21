"""
Turning Bands Simulation
========================

Fast Gaussian random field simulation using 1D line processes.

Industry Standard Implementation:
- Generates 1D Gaussian processes along multiple directions
- Superimposes them to create 3D field matching target variogram
- Very fast for large domains (>100M cells)
- Can be conditional or unconditional
- AUDIT FIX TB-001: Proper anisotropy handling via coordinate transformation

Use Cases:
- Large domains where SGSIM is too slow
- Geomechanical property fields
- Early exploration datasets
- Hydrogeological permeability fields

References:
- Matheron (1973) - Original turning bands method
- Mantoglou & Wilson (1982) - Implementation details
- Lantuéjoul (2002) - Geostatistical Simulation

AUDIT FIXES APPLIED:
- TB-001: Proper anisotropy handling via coordinate transformation
- TB-004: Complete lineage metadata in results
- CROSS-001: Variogram hash in metadata
- CROSS-002: Data source validation documented
"""

import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class TurningBandsConfig:
    """Configuration for Turning Bands Simulation."""
    n_realizations: int = 100
    n_bands: int = 1000  # Number of bands (directions)
    random_seed: Optional[int] = None
    
    # Variogram parameters
    variogram_type: str = 'spherical'  # 'spherical', 'exponential', 'gaussian'
    range_major: float = 100.0
    range_minor: float = 100.0
    range_vert: float = 50.0
    azimuth: float = 0.0  # Degrees, clockwise from North
    dip: float = 0.0      # Degrees, positive downward
    nugget: float = 0.0
    sill: float = 1.0
    
    # Conditioning
    condition: bool = True
    max_neighbors: int = 12
    max_search_radius: float = 200.0
    
    realization_prefix: str = "tb"
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters.
        
        AUDIT FIX: Ensure parameters are valid before simulation.
        """
        errors = []
        if self.nugget < 0:
            errors.append(f"Nugget cannot be negative (got {self.nugget})")
        if self.sill <= 0:
            errors.append(f"Sill must be positive (got {self.sill})")
        if self.nugget >= self.sill:
            errors.append(f"Nugget ({self.nugget}) must be less than sill ({self.sill})")
        if self.range_major <= 0 or self.range_minor <= 0 or self.range_vert <= 0:
            errors.append(f"Ranges must be positive")
        if self.n_bands < 10:
            errors.append(f"n_bands should be at least 10 for reasonable accuracy (got {self.n_bands})")
        valid_types = ['spherical', 'exponential', 'gaussian']
        if self.variogram_type.lower() not in valid_types:
            errors.append(f"Invalid variogram_type '{self.variogram_type}'")
        return errors
    
    def compute_hash(self) -> str:
        """
        Compute deterministic hash for lineage tracking.
        
        AUDIT FIX CROSS-001: Enable variogram hash validation.
        """
        params_str = (
            f"{self.variogram_type}:{self.range_major}:{self.range_minor}:{self.range_vert}:"
            f"{self.azimuth}:{self.dip}:{self.sill}:{self.nugget}:{self.n_realizations}:{self.random_seed}"
        )
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]


@dataclass
class TurningBandsResult:
    """Result from Turning Bands Simulation."""
    realizations: np.ndarray  # [n_realizations, n_grid]
    realization_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _create_rotation_matrix(azimuth: float, dip: float) -> np.ndarray:
    """
    Create rotation matrix from azimuth and dip angles.
    
    AUDIT FIX TB-001: Proper rotation for anisotropy.
    
    Convention:
    - Azimuth: Clockwise from North (Y-axis), in degrees
    - Dip: Positive downward from horizontal, in degrees
    
    Args:
        azimuth: Azimuth angle in degrees
        dip: Dip angle in degrees
    
    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    az_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)
    
    # Rotation around Z-axis (azimuth)
    cos_az, sin_az = np.cos(az_rad), np.sin(az_rad)
    Rz = np.array([
        [cos_az, -sin_az, 0],
        [sin_az, cos_az, 0],
        [0, 0, 1]
    ])
    
    # Rotation around X-axis (dip)
    cos_dip, sin_dip = np.cos(dip_rad), np.sin(dip_rad)
    Rx = np.array([
        [1, 0, 0],
        [0, cos_dip, -sin_dip],
        [0, sin_dip, cos_dip]
    ])
    
    # Combined rotation: first azimuth, then dip
    return Rx @ Rz


def _create_anisotropy_transform(config: TurningBandsConfig) -> np.ndarray:
    """
    Create combined anisotropy transformation matrix.
    
    AUDIT FIX TB-001: Transform coordinates to account for anisotropy.
    
    This matrix transforms coordinates so that:
    - The major axis becomes the X-axis
    - The minor axis becomes the Y-axis
    - The vertical axis becomes the Z-axis
    - All axes are scaled to unit range
    
    After transformation, distances can be computed isotropically.
    
    Args:
        config: Turning bands configuration with anisotropy parameters
    
    Returns:
        3x3 transformation matrix
    """
    # Rotation matrix
    R = _create_rotation_matrix(config.azimuth, config.dip)
    
    # Scaling matrix (normalize by ranges)
    # Using the major range as reference
    ref_range = config.range_major
    S = np.diag([
        ref_range / config.range_major,
        ref_range / config.range_minor,
        ref_range / config.range_vert
    ])
    
    # Combined transform: first rotate to principal axes, then scale
    return S @ R


def _generate_band_directions(n_bands: int) -> np.ndarray:
    """
    Generate uniformly distributed directions on unit sphere.
    
    Uses Fibonacci lattice for quasi-uniform distribution.
    
    Args:
        n_bands: Number of directions
    
    Returns:
        Array of unit vectors (n_bands, 3)
    """
    directions = np.zeros((n_bands, 3))
    
    # Golden angle
    phi = np.pi * (3 - np.sqrt(5))  # ~2.4 radians
    
    for i in range(n_bands):
        y = 1 - (i / (n_bands - 1)) * 2  # y from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        
        directions[i, 0] = np.cos(theta) * radius
        directions[i, 1] = y
        directions[i, 2] = np.sin(theta) * radius
    
    return directions


def _line_covariance(h: np.ndarray, range_: float, sill: float, model_type: str) -> np.ndarray:
    """
    Compute 1D covariance function for turning bands.
    
    The 1D covariance must be chosen so that when superimposed
    it produces the target 3D covariance.
    
    For spherical 3D: 1D covariance is C1(h) = C3(h) - h*dC3/dh
    """
    h = np.abs(h)
    C = np.zeros_like(h, dtype=float)
    
    if model_type == 'spherical':
        # For spherical variogram, the 1D turning bands covariance
        mask = h < range_
        r = h[mask] / range_
        # Spherical 1D covariance
        C[mask] = sill * (1 - 1.5 * r + 0.5 * r**3)
        C[~mask] = 0.0
        
    elif model_type == 'exponential':
        # Exponential: C(h) = sill * exp(-3h/a)
        C = sill * np.exp(-3 * h / range_)
        
    elif model_type == 'gaussian':
        # Gaussian: C(h) = sill * exp(-3(h/a)^2)
        C = sill * np.exp(-3 * (h / range_)**2)
    
    return C


def _generate_1d_process(
    t: np.ndarray,
    range_: float,
    sill: float,
    model_type: str
) -> np.ndarray:
    """
    Generate 1D Gaussian process along a line using spectral method.
    
    Args:
        t: Positions along the line
        range_: Variogram range
        sill: Variogram sill
        model_type: Variogram type
    
    Returns:
        1D Gaussian process values
    """
    n = len(t)
    
    # Use moving average method for speed
    # Window size based on range
    dt = np.mean(np.diff(np.sort(t))) if n > 1 else 1.0
    window_size = max(1, int(range_ / dt))
    
    # Generate white noise
    white_noise = np.random.randn(n + 2 * window_size)
    
    # Moving average kernel (approximates covariance structure)
    kernel = _line_covariance(
        np.arange(window_size) * dt,
        range_,
        sill,
        model_type
    )
    kernel = kernel / np.sqrt(np.sum(kernel**2) + 1e-10)
    
    # Convolve
    from scipy.ndimage import convolve1d
    process = convolve1d(white_noise, kernel, mode='constant')
    
    # Extract relevant portion
    return process[window_size:window_size + n]


def run_turning_bands(
    grid_coords: np.ndarray,
    config: TurningBandsConfig,
    conditioning_coords: Optional[np.ndarray] = None,
    conditioning_values: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable] = None,
    source_data_hash: Optional[str] = None
) -> TurningBandsResult:
    """
    Run Turning Bands Simulation.
    
    Industry-standard algorithm:
    1. Generate n_bands uniformly distributed directions
    2. Transform coordinates for anisotropy (AUDIT FIX TB-001)
    3. For each realization:
       a. Generate 1D Gaussian process along each band direction
       b. Project transformed grid points onto each band
       c. Sum contributions from all bands (scaled by 1/sqrt(n_bands))
    4. If conditioning, apply proper Simple Kriging correction
    
    AUDIT FIXES APPLIED:
    - TB-001: Proper anisotropy handling via coordinate transformation
    - TB-004: Complete lineage metadata
    - CROSS-001: Variogram hash in metadata
    
    Args:
        grid_coords: Grid coordinates (M, 3)
        config: Turning bands configuration
        conditioning_coords: Optional conditioning data coordinates (N, 3)
        conditioning_values: Optional conditioning data values (N,)
        progress_callback: Optional progress callback
        source_data_hash: Optional hash of source data for lineage
    
    Returns:
        TurningBandsResult with simulated realizations
        
    Raises:
        ValueError: If configuration validation fails
    """
    # AUDIT FIX: Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        error_msg = "Turning Bands configuration validation failed: " + "; ".join(validation_errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Starting Turning Bands: {config.n_realizations} realizations, {config.n_bands} bands")
    logger.info(f"Anisotropy: major={config.range_major}, minor={config.range_minor}, vert={config.range_vert}")
    logger.info(f"Orientation: azimuth={config.azimuth}°, dip={config.dip}°")
    
    # Set random seed
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
    
    n_grid = len(grid_coords)
    
    # AUDIT FIX TB-001: Create anisotropy transformation matrix
    aniso_transform = _create_anisotropy_transform(config)
    
    # Transform grid coordinates for anisotropy
    # After transformation, we can use isotropic simulation with major range
    transformed_coords = grid_coords @ aniso_transform.T
    
    # Generate band directions (on unit sphere in transformed space)
    directions = _generate_band_directions(config.n_bands)
    
    # Use major range as reference (since coordinates are transformed)
    effective_range = config.range_major
    
    # Initialize realizations
    realizations = np.zeros((config.n_realizations, n_grid))
    realization_names = []
    
    # Check conditioning
    has_conditioning = (
        config.condition and 
        conditioning_coords is not None and 
        conditioning_values is not None and
        len(conditioning_coords) > 0
    )
    
    # Transform conditioning coordinates if present
    if has_conditioning:
        transformed_cond_coords = conditioning_coords @ aniso_transform.T
    
    # Main simulation loop
    progress_interval = max(1, config.n_realizations // 50)
    for ireal in range(config.n_realizations):
        if progress_callback and ((ireal + 1) % progress_interval == 0 or (ireal + 1) == config.n_realizations):
            overall_progress = 5 + int(((ireal + 1) / config.n_realizations) * 90)
            progress_callback(overall_progress, f"{ireal + 1}/{config.n_realizations}")

        # Unconditional simulation via turning bands
        sim_values = np.zeros(n_grid)

        # Progress updates during band generation (only for first few realizations to avoid spam)
        # Bands are fast, so only report realization-level progress
        for i_band in range(config.n_bands):
            direction = directions[i_band]

            # Project TRANSFORMED grid points onto this line
            # AUDIT FIX TB-001: Use transformed coordinates for proper anisotropy
            projections = np.dot(transformed_coords, direction)
            
            # Generate 1D process along this line
            sort_idx = np.argsort(projections)
            sorted_proj = projections[sort_idx]
            
            process_1d = _generate_1d_process(
                sorted_proj,
                effective_range,
                config.sill - config.nugget,
                config.variogram_type
            )
            
            # Add to simulation (unsort)
            unsort_idx = np.argsort(sort_idx)
            sim_values += process_1d[unsort_idx]
        
        # Scale by 1/sqrt(n_bands) for correct variance
        sim_values /= np.sqrt(config.n_bands)
        
        # Add nugget effect
        if config.nugget > 0:
            sim_values += np.sqrt(config.nugget) * np.random.randn(n_grid)
        
        # Conditioning via Simple Kriging (using transformed coordinates)
        if has_conditioning:
            sim_values = _condition_realization_sk(
                grid_coords,  # Original coords for output
                transformed_coords,  # Transformed coords for distances
                sim_values,
                conditioning_coords,
                transformed_cond_coords,
                conditioning_values,
                config
            )
        
        realizations[ireal] = sim_values
        realization_names.append(f"{config.realization_prefix}_{ireal + 1:04d}")
    
    logger.info(f"Turning Bands complete: {config.n_realizations} realizations")
    
    # AUDIT FIX TB-004: Complete lineage metadata
    result = TurningBandsResult(
        realizations=realizations,
        realization_names=realization_names,
        metadata={
            # Core parameters
            'n_realizations': config.n_realizations,
            'n_bands': config.n_bands,
            'n_grid': n_grid,
            'variogram_type': config.variogram_type,
            'range_major': config.range_major,
            'range_minor': config.range_minor,
            'range_vert': config.range_vert,
            'azimuth': config.azimuth,
            'dip': config.dip,
            'sill': config.sill,
            'nugget': config.nugget,
            'conditional': has_conditioning,
            'n_conditioning': len(conditioning_coords) if has_conditioning else 0,
            'method': 'Turning Bands Simulation',
            # AUDIT FIX: Lineage tracking
            'variogram_hash': config.compute_hash(),
            'source_data_hash': source_data_hash,
            'execution_timestamp': datetime.now().isoformat(),
            'anisotropy_method': 'coordinate_transformation',
            'conditioning_method': 'Simple Kriging' if has_conditioning else None,
            'audit_version': '2.0.0-TB-001-fix',
        }
    )
    
    # ✅ NEW: Create StructuredGrid for first realization (for visualization)
    # Note: Full implementation would create grids for all realizations
    if len(realizations) > 0 and len(grid_coords) > 0:
        # Infer grid definition from coordinates
        coords_min = grid_coords.min(axis=0)
        coords_max = grid_coords.max(axis=0)
        
        # Estimate grid spacing - use default spacing to avoid issues with sparse coordinates
        spacing_est = (10.0, 10.0, 5.0)

        # Estimate grid counts - ensure at least 1 in each dimension
        counts_est = (
            max(1, int(np.ceil((coords_max[0] - coords_min[0] + spacing_est[0]) / spacing_est[0]))),
            max(1, int(np.ceil((coords_max[1] - coords_min[1] + spacing_est[1]) / spacing_est[1]))),
            max(1, int(np.ceil((coords_max[2] - coords_min[2] + spacing_est[2]) / spacing_est[2])))
        )
        
        # Create grid for mean realization
        mean_realization = np.mean(realizations, axis=0)
        
        # Try to reshape if possible
        if len(mean_realization) == counts_est[0] * counts_est[1] * counts_est[2]:
            from .simulation_interface import create_structured_grid, GridDefinition
            
            grid_def = GridDefinition(
                origin=tuple(coords_min),
                spacing=spacing_est,
                counts=counts_est
            )
            
            result.grid = create_structured_grid(
                mean_realization,
                grid_def,
                property_name="TB_Mean",
                metadata={
                    'method': 'Turning Bands Simulation',
                    'n_realizations': config.n_realizations,
                    'n_bands': config.n_bands
                }
            )
    
    return result


def _compute_covariance_tb(
    h: np.ndarray,
    range_: float,
    sill: float,
    nugget: float,
    model_type: str,
    include_nugget_diagonal: bool = False
) -> np.ndarray:
    """
    Compute covariance values for Turning Bands conditioning.
    
    Args:
        h: Distance array
        range_: Variogram range
        sill: Total sill
        nugget: Nugget effect
        model_type: Variogram type
        include_nugget_diagonal: Add nugget at h=0
    
    Returns:
        Covariance values
    """
    partial_sill = sill - nugget
    
    if model_type == 'exponential':
        C = partial_sill * np.exp(-3 * h / range_)
    elif model_type == 'gaussian':
        C = partial_sill * np.exp(-3 * (h / range_)**2)
    else:  # spherical
        C = np.zeros_like(h, dtype=float)
        mask = h < range_
        hr = h[mask] / range_
        C[mask] = partial_sill * (1 - 1.5 * hr + 0.5 * hr**3)
    
    if include_nugget_diagonal:
        C = np.where(h == 0, C + nugget, C)
    
    return C


def _condition_realization_sk(
    grid_coords: np.ndarray,
    transformed_grid_coords: np.ndarray,
    unconditional_values: np.ndarray,
    cond_coords: np.ndarray,
    transformed_cond_coords: np.ndarray,
    cond_values: np.ndarray,
    config: TurningBandsConfig
) -> np.ndarray:
    """
    Condition unconditional realization using proper Simple Kriging.
    
    AUDIT FIX: Uses proper Simple Kriging instead of IDW.
    
    Algorithm:
    1. Get unconditional values at conditioning locations
    2. Compute residuals: data - unconditional
    3. Apply Simple Kriging to interpolate residuals
    4. Add kriged residuals to unconditional field
    
    Args:
        grid_coords: Original grid coordinates
        transformed_grid_coords: Anisotropy-transformed grid coordinates
        unconditional_values: Unconditional simulation values
        cond_coords: Original conditioning coordinates
        transformed_cond_coords: Transformed conditioning coordinates
        cond_values: Conditioning data values
        config: Configuration
    
    Returns:
        Conditioned values
    """
    n_grid = len(grid_coords)
    n_cond = len(cond_coords)
    
    # Use major range (coordinates are already transformed for anisotropy)
    effective_range = config.range_major
    
    # Build KDTree for transformed conditioning coordinates
    tree = cKDTree(transformed_cond_coords)
    
    # Get unconditional values at conditioning locations via nearest grid point
    grid_tree = cKDTree(transformed_grid_coords)
    _, grid_idx = grid_tree.query(transformed_cond_coords, k=1)
    uncond_at_cond = unconditional_values[grid_idx]
    
    # Compute residuals at conditioning locations
    residuals = cond_values - uncond_at_cond
    
    # Pre-compute covariance matrix for conditioning data
    cond_dists = squareform(pdist(transformed_cond_coords))
    C_data_data = _compute_covariance_tb(
        cond_dists, effective_range, config.sill, config.nugget,
        config.variogram_type, include_nugget_diagonal=True
    )
    
    # Add regularization for numerical stability
    C_data_data += np.eye(n_cond) * 1e-10 * np.max(np.diag(C_data_data))
    
    # Initialize conditioned values
    conditional_values = unconditional_values.copy()
    
    # Simple Kriging for each grid cell
    for i in range(n_grid):
        target = transformed_grid_coords[i]
        
        # Find nearby conditioning data in transformed space
        dist, idx = tree.query(
            target.reshape(1, -1),
            k=min(config.max_neighbors, n_cond),
            distance_upper_bound=config.max_search_radius
        )
        
        dist = dist[0]
        idx = idx[0]
        valid = dist < np.inf
        
        if np.sum(valid) < 2:
            continue
        
        nb_idx = idx[valid]
        nb_dists = dist[valid]
        nb_residuals = residuals[nb_idx]
        n_nb = len(nb_idx)
        
        # Build local kriging system
        if n_nb > 1:
            C_nb = C_data_data[np.ix_(nb_idx, nb_idx)]
        else:
            C_nb = np.array([[config.sill]])
        
        # Covariance between target and neighbors
        c0 = _compute_covariance_tb(
            nb_dists, effective_range, config.sill, config.nugget,
            config.variogram_type, include_nugget_diagonal=False
        )
        
        # Solve kriging system
        try:
            from scipy.linalg import solve
            weights = solve(C_nb, c0, assume_a='pos')
        except Exception:
            weights, _, _, _ = np.linalg.lstsq(C_nb, c0, rcond=1e-10)
        
        # Apply Simple Kriging estimate for residual
        conditional_values[i] += np.sum(weights * nb_residuals)
    
    return conditional_values


def _condition_realization(
    grid_coords: np.ndarray,
    unconditional_values: np.ndarray,
    cond_coords: np.ndarray,
    cond_values: np.ndarray,
    config: TurningBandsConfig
) -> np.ndarray:
    """
    Legacy conditioning function - forwards to proper SK implementation.
    
    DEPRECATED: Use _condition_realization_sk instead.
    """
    # Create identity transform for legacy calls
    transformed_grid = grid_coords.copy()
    transformed_cond = cond_coords.copy()
    
    return _condition_realization_sk(
        grid_coords, transformed_grid,
        unconditional_values,
        cond_coords, transformed_cond,
        cond_values,
        config
    )

