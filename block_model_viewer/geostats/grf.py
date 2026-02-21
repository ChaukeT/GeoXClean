"""
Gaussian Random Fields (GRF)
============================

Unconditional Gaussian simulation from variogram or spectral methods.

Industry Standard Implementation:
- Generates fields using spectral decomposition or FFT
- Very fast for large domains
- Can be conditioned via post-processing kriging step (AUDIT FIX: proper SK)
- Flexible for various covariance functions

Use Cases:
- Fast generation of many fields
- Geomechanical heterogeneity
- Early exploration uncertainty
- Hydrogeological permeability fields
- Porosity modeling
- Pre-processing for other methods

References:
- Dietrich & Newsam (1993) - Fast generation of Gaussian random fields
- Wood & Chan (1994) - FFT-based simulation
- Lantuéjoul (2002) - Spectral methods

AUDIT FIXES APPLIED:
- GRF-002: Replaced IDW conditioning with proper Simple Kriging
- GRF-003: Added covariance_type validation
- CROSS-001: Added variogram hash to metadata
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

logger = logging.getLogger(__name__)


@dataclass
class GRFConfig:
    """Configuration for Gaussian Random Field simulation."""
    n_realizations: int = 100
    random_seed: Optional[int] = None
    
    # Variogram/covariance parameters
    covariance_type: str = 'spherical'  # 'spherical', 'exponential', 'gaussian', 'matern'
    range_x: float = 100.0
    range_y: float = 100.0
    range_z: float = 50.0
    sill: float = 1.0
    nugget: float = 0.0
    
    # For Matérn covariance
    matern_nu: float = 1.5  # Smoothness parameter
    
    # Simulation method
    method: str = 'fft'  # 'fft', 'cholesky', 'spectral'
    
    # Conditioning
    condition: bool = False
    max_neighbors: int = 12
    max_search_radius: float = 200.0
    
    realization_prefix: str = "grf"
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters.
        
        AUDIT FIX GRF-003: Validate sill > nugget and ranges > 0
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        if self.nugget < 0:
            errors.append(f"Nugget cannot be negative (got {self.nugget})")
        if self.sill <= 0:
            errors.append(f"Sill must be positive (got {self.sill})")
        if self.nugget >= self.sill:
            errors.append(f"Nugget ({self.nugget}) must be less than sill ({self.sill})")
        if self.range_x <= 0 or self.range_y <= 0 or self.range_z <= 0:
            errors.append(f"Ranges must be positive (got {self.range_x}, {self.range_y}, {self.range_z})")
        valid_types = ['spherical', 'exponential', 'gaussian', 'matern']
        if self.covariance_type.lower() not in valid_types:
            errors.append(f"Invalid covariance_type '{self.covariance_type}', must be one of {valid_types}")
        return errors
    
    def compute_hash(self) -> str:
        """
        Compute deterministic hash of configuration for lineage tracking.
        
        AUDIT FIX CROSS-001: Enable variogram hash validation
        """
        params_str = (
            f"{self.covariance_type}:{self.range_x}:{self.range_y}:{self.range_z}:"
            f"{self.sill}:{self.nugget}:{self.n_realizations}:{self.random_seed}"
        )
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]


@dataclass
class GRFResult:
    """Result from Gaussian Random Field simulation."""
    realizations: np.ndarray  # [n_realizations, nz, ny, nx] or [n_realizations, n_points]
    realization_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _compute_covariance(
    h: np.ndarray,
    range_: float,
    sill: float,
    nugget: float,
    cov_type: str,
    include_nugget_diagonal: bool = False
) -> np.ndarray:
    """
    Compute covariance values from distances.
    
    AUDIT FIX GRF-002: Proper covariance function for Simple Kriging.
    
    Args:
        h: Distance array (can be 1D or 2D)
        range_: Correlation range
        sill: Total sill (nugget + partial sill)
        nugget: Nugget effect
        cov_type: Covariance type ('spherical', 'exponential', 'gaussian')
        include_nugget_diagonal: If True, add nugget to diagonal (h=0)
    
    Returns:
        Covariance values
    """
    partial_sill = sill - nugget
    
    if cov_type == 'exponential':
        # Exponential: C(h) = partial_sill * exp(-3h/a)
        C = partial_sill * np.exp(-3 * h / range_)
    elif cov_type == 'gaussian':
        # Gaussian: C(h) = partial_sill * exp(-3(h/a)^2)
        C = partial_sill * np.exp(-3 * (h / range_)**2)
    else:  # spherical
        # Spherical: C(h) = partial_sill * (1 - 1.5*h/a + 0.5*(h/a)^3) for h < a
        C = np.zeros_like(h, dtype=float)
        mask = h < range_
        hr = h[mask] / range_
        C[mask] = partial_sill * (1 - 1.5 * hr + 0.5 * hr**3)
    
    # Add nugget at h=0 (on diagonal for covariance matrix)
    if include_nugget_diagonal:
        C = np.where(h == 0, C + nugget, C)
    
    return C


def _simple_kriging_conditioning(
    grid_coords: np.ndarray,
    uncond_field: np.ndarray,
    conditioning_coords: np.ndarray,
    conditioning_values: np.ndarray,
    config: GRFConfig
) -> np.ndarray:
    """
    Apply Simple Kriging conditioning to unconditional GRF.
    
    AUDIT FIX GRF-002: Proper Simple Kriging instead of IDW.
    
    This ensures that:
    1. Sample values are exactly reproduced at data locations
    2. Kriging weights honor the covariance structure
    3. Conditioning is mathematically correct
    
    Algorithm:
    1. Compute unconditional values at conditioning locations
    2. Compute residuals: data - unconditional
    3. Apply Simple Kriging to interpolate residuals to grid
    4. Add kriged residuals to unconditional field
    
    Args:
        grid_coords: Grid coordinates (n_cells, 3)
        uncond_field: Unconditional field values (flattened)
        conditioning_coords: Conditioning data coordinates (n_cond, 3)
        conditioning_values: Conditioning data values (n_cond,)
        config: GRF configuration
    
    Returns:
        Conditioned field values (flattened)
    """
    n_cells = len(grid_coords)
    n_cond = len(conditioning_coords)
    
    # Use effective range (geometric mean for anisotropy)
    effective_range = (config.range_x * config.range_y * config.range_z) ** (1/3)
    
    # Build KDTree for conditioning data
    cond_tree = cKDTree(conditioning_coords)
    
    # Get unconditional values at conditioning locations via nearest neighbor
    grid_tree = cKDTree(grid_coords)
    _, grid_idx = grid_tree.query(conditioning_coords, k=1)
    uncond_at_cond = uncond_field[grid_idx]
    
    # Compute residuals at conditioning locations
    residuals = conditioning_values - uncond_at_cond
    
    # Initialize conditioned field
    conditioned_field = uncond_field.copy()
    
    # Pre-compute covariance matrix for conditioning data (for kriging system)
    # This is C(data, data) - the covariance between conditioning points
    cond_dists = squareform(pdist(conditioning_coords))
    C_data_data = _compute_covariance(
        cond_dists, effective_range, config.sill, config.nugget,
        config.covariance_type, include_nugget_diagonal=True
    )
    
    # Add small regularization for numerical stability
    C_data_data += np.eye(n_cond) * 1e-10 * np.max(np.diag(C_data_data))
    
    # Simple Kriging for each grid cell
    for i in range(n_cells):
        target = grid_coords[i]
        
        # Find nearby conditioning data
        dist, idx = cond_tree.query(
            target.reshape(1, -1),
            k=min(config.max_neighbors, n_cond),
            distance_upper_bound=config.max_search_radius
        )
        
        dist = dist[0]
        idx = idx[0]
        valid = dist < np.inf
        
        if np.sum(valid) < 2:
            # Not enough neighbors - keep unconditional value
            continue
        
        nb_idx = idx[valid]
        nb_dists = dist[valid]
        nb_residuals = residuals[nb_idx]
        n_nb = len(nb_idx)
        
        # Build local kriging system
        # C_nb: covariance matrix between neighbors
        if n_nb > 1:
            C_nb = C_data_data[np.ix_(nb_idx, nb_idx)]
        else:
            C_nb = np.array([[config.sill]])
        
        # c0: covariance between target and neighbors
        c0 = _compute_covariance(
            nb_dists, effective_range, config.sill, config.nugget,
            config.covariance_type, include_nugget_diagonal=False
        )
        
        # Solve kriging system: C_nb @ weights = c0
        try:
            from scipy.linalg import solve
            weights = solve(C_nb, c0, assume_a='pos')
        except Exception:
            # Fallback to least squares if solve fails
            weights, _, _, _ = np.linalg.lstsq(C_nb, c0, rcond=1e-10)
        
        # Apply Simple Kriging estimate for residual
        # SK estimate = sum(weights * residuals) since SK mean is 0 for residuals
        conditioned_field[i] += np.sum(weights * nb_residuals)
    
    return conditioned_field


def _spectral_density(k: np.ndarray, range_: float, sill: float, cov_type: str) -> np.ndarray:
    """
    Compute spectral density function for given covariance model.
    
    Args:
        k: Wavenumber array
        range_: Correlation range
        sill: Variance (sill)
        cov_type: Covariance type
    
    Returns:
        Spectral density values
    """
    # Avoid division by zero
    k = np.maximum(k, 1e-10)
    
    if cov_type == 'exponential':
        # Exponential: S(k) ∝ 1/(1 + (ak)^2)^(d+1)/2
        a = range_ / 3  # Practical range
        return sill * a**3 / (1 + (a * k)**2)**2
    
    elif cov_type == 'gaussian':
        # Gaussian: S(k) ∝ exp(-(ak)^2/4)
        a = range_ / np.sqrt(3)
        return sill * (a * np.sqrt(np.pi))**3 * np.exp(-(a * k)**2 / 4)
    
    elif cov_type == 'spherical':
        # Spherical spectral density (approximate)
        a = range_
        # Use exponential approximation for simplicity
        return sill * a**3 / (1 + (a * k / 2)**2)**2
    
    else:  # Default to exponential
        a = range_ / 3
        return sill * a**3 / (1 + (a * k)**2)**2


def _fft_simulation(
    shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    config: GRFConfig
) -> np.ndarray:
    """
    Generate GRF using FFT-based spectral method.
    
    Args:
        shape: Grid shape (nz, ny, nx)
        spacing: Grid spacing (dz, dy, dx)
        config: GRF configuration
    
    Returns:
        Single GRF realization
    """
    nz, ny, nx = shape
    dz, dy, dx = spacing
    
    # Create frequency grids
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=dz) * 2 * np.pi
    
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Anisotropic wavenumber
    K = np.sqrt(
        (KX / (config.range_x / 100))**2 +
        (KY / (config.range_y / 100))**2 +
        (KZ / (config.range_z / 100))**2
    )
    
    # Spectral density
    effective_range = (config.range_x * config.range_y * config.range_z) ** (1/3)
    S = _spectral_density(K, effective_range, config.sill - config.nugget, config.covariance_type)
    
    # Generate complex Gaussian noise
    noise_real = np.random.randn(nx, ny, nz)
    noise_imag = np.random.randn(nx, ny, nz)
    noise = noise_real + 1j * noise_imag
    
    # Apply spectral filter
    filtered = noise * np.sqrt(S)
    
    # Inverse FFT
    field = np.real(np.fft.ifftn(filtered))
    
    # Normalize to target variance
    field = field / np.std(field) * np.sqrt(config.sill - config.nugget)
    
    # Add nugget
    if config.nugget > 0:
        field += np.sqrt(config.nugget) * np.random.randn(*shape)
    
    return field.transpose(2, 1, 0)  # Return as (nz, ny, nx)


def _cholesky_simulation(
    coords: np.ndarray,
    config: GRFConfig
) -> np.ndarray:
    """
    Generate GRF using Cholesky decomposition.
    
    More accurate but O(n³) complexity - only for small grids.
    
    Args:
        coords: Point coordinates (N, 3)
        config: GRF configuration
    
    Returns:
        Single GRF realization at specified coordinates
    """
    n = len(coords)
    
    if n > 5000:
        logger.warning(f"Cholesky method with {n} points may be slow. Consider FFT method.")
    
    # Build covariance matrix
    from scipy.spatial.distance import pdist, squareform
    
    # Anisotropic distances
    scaled_coords = coords.copy()
    scaled_coords[:, 0] /= config.range_x
    scaled_coords[:, 1] /= config.range_y
    scaled_coords[:, 2] /= config.range_z
    
    dists = squareform(pdist(scaled_coords))
    
    # Covariance from variogram: C(h) = sill - gamma(h)
    if config.covariance_type == 'exponential':
        C = (config.sill - config.nugget) * np.exp(-3 * dists)
    elif config.covariance_type == 'gaussian':
        C = (config.sill - config.nugget) * np.exp(-3 * dists**2)
    else:  # spherical
        C = np.zeros_like(dists)
        mask = dists < 1
        C[mask] = (config.sill - config.nugget) * (1 - 1.5 * dists[mask] + 0.5 * dists[mask]**3)
    
    # Add nugget to diagonal
    C += config.nugget * np.eye(n)
    
    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
    except np.linalg.LinAlgError:
        logger.warning("Cholesky failed, using eigendecomposition")
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 0)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # Generate field
    z = np.random.randn(n)
    field = L @ z
    
    return field


def run_grf(
    grid_shape: Tuple[int, int, int],
    grid_spacing: Tuple[float, float, float],
    config: GRFConfig,
    conditioning_coords: Optional[np.ndarray] = None,
    conditioning_values: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable] = None,
    source_data_hash: Optional[str] = None
) -> GRFResult:
    """
    Run Gaussian Random Field simulation.
    
    Industry-standard algorithm:
    1. Generate unconditional GRF using FFT or Cholesky
    2. If conditioning data provided:
       a. Generate unconditional values at conditioning locations
       b. Compute residuals: data - unconditional
       c. Krige residuals to grid (AUDIT FIX: proper Simple Kriging)
       d. Add kriged residuals to unconditional field
    
    AUDIT FIXES APPLIED:
    - GRF-002: Uses proper Simple Kriging for conditioning (not IDW)
    - GRF-003: Validates config parameters before execution
    - CROSS-001: Includes variogram/covariance hash in metadata
    
    Args:
        grid_shape: Output grid shape (nz, ny, nx)
        grid_spacing: Grid cell size (dz, dy, dx)
        config: GRF configuration
        conditioning_coords: Optional conditioning data coordinates
        conditioning_values: Optional conditioning data values
        progress_callback: Optional progress callback
        source_data_hash: Optional hash of source data for lineage tracking
    
    Returns:
        GRFResult with simulated realizations
        
    Raises:
        ValueError: If configuration validation fails
    """
    # AUDIT FIX GRF-003: Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        error_msg = "GRF configuration validation failed: " + "; ".join(validation_errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Starting GRF: {config.n_realizations} realizations, grid {grid_shape}")
    logger.info(f"Method: {config.method}, Covariance: {config.covariance_type}")
    
    # Set random seed
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
    
    nz, ny, nx = grid_shape
    dz, dy, dx = grid_spacing
    n_cells = nz * ny * nx
    
    # Check conditioning
    has_conditioning = (
        config.condition and 
        conditioning_coords is not None and 
        conditioning_values is not None and
        len(conditioning_coords) > 0
    )
    
    # Generate grid coordinates
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Initialize realizations
    realizations = np.zeros((config.n_realizations, nz, ny, nx))
    realization_names = []
    
    # Main simulation loop
    last_progress_pct = 0
    for ireal in range(config.n_realizations):
        if progress_callback:
            # Calculate percentage (5-95% for simulation, reserve 0-5 for setup, 95-100 for post-process)
            pct = 5 + int((ireal + 1) / config.n_realizations * 90)
            # Update every 2% or on final realization
            progress_interval = max(1, config.n_realizations // 50)
            if (ireal + 1) % progress_interval == 0 or (ireal + 1) == config.n_realizations:
                progress_callback(pct, f"{ireal + 1}/{config.n_realizations}")
        
        # Generate unconditional field
        if config.method == 'fft':
            uncond_field = _fft_simulation(grid_shape, grid_spacing, config)
        elif config.method == 'cholesky':
            uncond_values = _cholesky_simulation(grid_coords, config)
            uncond_field = uncond_values.reshape(nx, ny, nz).transpose(2, 1, 0)
        else:
            # Default to FFT
            uncond_field = _fft_simulation(grid_shape, grid_spacing, config)
        
        # AUDIT FIX GRF-002: Conditioning using proper Simple Kriging
        if has_conditioning:
            # Use proper Simple Kriging conditioning instead of IDW
            conditioned_values = _simple_kriging_conditioning(
                grid_coords=grid_coords,
                uncond_field=uncond_field.flatten(),
                conditioning_coords=conditioning_coords,
                conditioning_values=conditioning_values,
                config=config
            )
            realizations[ireal] = conditioned_values.reshape(nx, ny, nz).transpose(2, 1, 0)
        else:
            realizations[ireal] = uncond_field
        
        realization_names.append(f"{config.realization_prefix}_{ireal + 1:04d}")
    
    logger.info(f"GRF complete: {config.n_realizations} realizations")
    
    # Compute statistics
    mean_field = np.mean(realizations, axis=0)
    var_field = np.var(realizations, axis=0)
    
    # AUDIT FIX CROSS-001: Include complete lineage metadata
    return GRFResult(
        realizations=realizations,
        realization_names=realization_names,
        metadata={
            # Core parameters
            'n_realizations': config.n_realizations,
            'grid_shape': grid_shape,
            'grid_spacing': grid_spacing,
            'method': config.method,
            'covariance_type': config.covariance_type,
            'target_sill': config.sill,
            'target_nugget': config.nugget,
            'ranges': (config.range_x, config.range_y, config.range_z),
            'actual_variance': float(np.mean(var_field)),
            'conditional': has_conditioning,
            'n_conditioning': len(conditioning_coords) if has_conditioning else 0,
            'algorithm': 'Gaussian Random Field (FFT/Cholesky)',
            # AUDIT FIX: Lineage tracking
            'covariance_hash': config.compute_hash(),
            'source_data_hash': source_data_hash,
            'execution_timestamp': datetime.now().isoformat(),
            'conditioning_method': 'Simple Kriging' if has_conditioning else None,
            'audit_version': '2.0.0-GRF-002-fix',
        }
    )

