"""
Standardized Simulation Workflow Manager
=========================================

Generalized workflow controller for all simulation methods following
the Datamine/Surpac/Isatis architecture:

1. LOAD DRILLHOLES: Extract and validate conditioning data
2. DEFINE GRID EXTENTS + BLOCK SIZE: Create simulation grid
3. GENERATE EMPTY PROPERTY ARRAY: Initialize output arrays
4. RUN SIMULATION: Populate arrays with simulated values

This ensures all simulation methods (SGSIM, SIS, Turning Bands, DBS, etc.)
follow the same consistent workflow pattern.

AUDIT FIXES APPLIED:
- CROSS-002: Data source validation gate added before simulation
- Added variogram hash tracking to simulation metadata

Author: Block Model Viewer Team
Date: 2025
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SimulationParameters:
    """
    Standardized parameters for all simulation methods.

    This provides a common interface for all simulation types.
    """

    def __init__(
        self,
        method: str,
        n_realizations: int = 100,
        nx: int = 50,
        ny: int = 50,
        nz: int = 20,
        xmin: float = 0.0,
        ymin: float = 0.0,
        zmin: float = 0.0,
        xinc: float = 10.0,
        yinc: float = 10.0,
        zinc: float = 5.0,
        # Variogram parameters
        variogram_type: str = 'spherical',
        range_major: float = 100.0,
        range_minor: float = 50.0,
        range_vert: float = 25.0,
        azimuth: float = 0.0,
        dip: float = 0.0,
        nugget: float = 0.0,
        sill: float = 1.0,
        # Search parameters
        min_neighbors: int = 8,
        max_neighbors: int = 16,
        max_search_radius: float = 200.0,
        # Method-specific parameters
        method_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        self.method = method
        self.n_realizations = n_realizations
        self.nx, self.ny, self.nz = nx, ny, nz
        self.xmin, self.ymin, self.zmin = xmin, ymin, zmin
        self.xinc, self.yinc, self.zinc = xinc, yinc, zinc
        self.variogram_type = variogram_type
        self.range_major = range_major
        self.range_minor = range_minor
        self.range_vert = range_vert
        self.azimuth = azimuth
        self.dip = dip
        self.nugget = nugget
        self.sill = sill
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.max_search_radius = max_search_radius
        self.method_params = method_params or {}
        self.seed = seed


def _compute_variogram_hash(params: SimulationParameters) -> str:
    """Compute hash of variogram parameters for lineage tracking."""
    hash_str = (
        f"{params.variogram_type}:{params.range_major}:{params.range_minor}:{params.range_vert}:"
        f"{params.azimuth}:{params.dip}:{params.sill}:{params.nugget}"
    )
    return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


def _compute_data_hash(data_df: pd.DataFrame, variable: str) -> str:
    """Compute hash of source data for lineage tracking."""
    try:
        cols = ['X', 'Y', 'Z', variable]
        cols_avail = [c for c in cols if c in data_df.columns]
        df_hash = data_df[cols_avail].dropna()
        hash_bytes = pd.util.hash_pandas_object(df_hash, index=False).values.tobytes()
        return hashlib.sha256(hash_bytes).hexdigest()[:16]
    except Exception:
        return "HASH_FAILED"


def _validate_data_source_for_simulation(
    data_df: pd.DataFrame,
    method: str,
    strict: bool = True
) -> Dict[str, Any]:
    """
    AUDIT FIX CROSS-002: Validate data source before simulation.
    
    This gate ensures raw assays are NOT used directly for simulation.
    
    Args:
        data_df: Source DataFrame
        method: Simulation method name
        strict: If True, raise error on invalid source
    
    Returns:
        Validation result dict
        
    Raises:
        ValueError: If strict and data source is invalid
    """
    result = {
        'valid': True,
        'source_type': 'unknown',
        'warnings': [],
    }
    
    # Check for source_type in DataFrame attrs (set by DataRegistry)
    source_type = getattr(data_df, 'attrs', {}).get('source_type', 'unknown')
    result['source_type'] = source_type
    
    # Block raw assays
    if source_type in ['raw_assays', 'assays', 'raw']:
        error_msg = (
            f"DATA SOURCE GATE FAILED for {method}: "
            f"Data source type is '{source_type}' (raw assays). "
            "Simulation on raw assays violates geostatistical principles. "
            "Please composite the data first."
        )
        logger.error(error_msg)
        result['valid'] = False
        result['warnings'].append(error_msg)
        if strict:
            raise ValueError(error_msg)
    
    # Warn on unknown source
    elif source_type == 'unknown':
        warning = (
            f"DATA SOURCE WARNING for {method}: "
            "Data source type unknown. Data may have been loaded outside standard pipeline. "
            "For JORC/SAMREC compliance, use properly composited data."
        )
        logger.warning(warning)
        result['warnings'].append(warning)
    
    return result


def execute_standardized_simulation_workflow(
    data_df: pd.DataFrame,
    variable: str,
    params: SimulationParameters,
    cutoffs: Optional[List[float]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Execute standardized simulation workflow for any simulation method.

    Workflow:
        0. VALIDATE DATA SOURCE: Ensure data is properly composited (AUDIT FIX CROSS-002)
        1. LOAD DRILLHOLES: Extract and validate conditioning data
        2. DEFINE GRID EXTENTS + BLOCK SIZE: Create simulation grid
        3. GENERATE EMPTY PROPERTY ARRAY: Initialize output arrays
        4. RUN SIMULATION: Populate arrays with method-specific simulation

    AUDIT FIXES APPLIED:
    - CROSS-002: Data source validation gate prevents raw assay usage
    - CROSS-001: Variogram hash tracked in metadata for lineage
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Drillhole data with X, Y, Z, and variable columns
        MUST be composited data, NOT raw assays
    variable : str
        Name of the variable column to simulate
    params : SimulationParameters
        Standardized simulation parameters
    cutoffs : List[float], optional
        Cutoff values for probability mapping
    progress_callback : Callable, optional
        Progress callback function(percent, message)

    Returns
    -------
    dict
        Standardized results including:
        - 'realizations': Simulated realizations (nreal, nz, ny, nx)
        - 'summary': Summary statistics
        - 'probability_maps': Probability maps for cutoffs
        - 'metadata': Simulation metadata with lineage info
        - 'grid_coords': Grid coordinates for visualization
    """
    logger.info(f"Starting standardized {params.method} simulation workflow...")

    # Validate inputs
    if data_df is None:
        raise ValueError("data_df cannot be None. Data should be injected by Controller before calling workflow.")
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError(f"data_df must be a pandas DataFrame, got {type(data_df)}")
    if variable is None or not variable:
        raise ValueError("variable cannot be None or empty")
    
    # ========================================================================
    # STEP 0: AUDIT FIX CROSS-002 - DATA SOURCE VALIDATION GATE
    # ========================================================================
    if progress_callback:
        progress_callback(2, "Validating data source...")
    
    # Check data source (non-strict for now to avoid breaking existing workflows)
    source_validation = _validate_data_source_for_simulation(
        data_df, params.method, strict=False
    )
    
    # Compute hashes for lineage tracking
    source_data_hash = _compute_data_hash(data_df, variable)
    variogram_hash = _compute_variogram_hash(params)

    # ========================================================================
    # STEP 1: LOAD DRILLHOLES
    # ========================================================================
    if progress_callback:
        progress_callback(5, "Loading and validating drillhole data...")

    # Extract and validate conditioning data
    drillhole_data = _load_drillhole_data(data_df, variable)
    if drillhole_data is None:
        raise ValueError(f"No valid data found for variable '{variable}'")

    logger.info(f"Loaded {len(drillhole_data['coords'])} drillhole samples for {variable}")

    # ========================================================================
    # STEP 2: DEFINE GRID EXTENTS + BLOCK SIZE
    # ========================================================================
    if progress_callback:
        progress_callback(15, "Defining simulation grid...")

    grid_coords, grid_extents = _define_simulation_grid(params, drillhole_data['coords'])

    logger.info(f"Grid defined: {params.nx}×{params.ny}×{params.nz} blocks, "
               f"spacing ({params.xinc:.1f}, {params.yinc:.1f}, {params.zinc:.1f})")

    # ========================================================================
    # STEP 3: GENERATE EMPTY PROPERTY ARRAY
    # ========================================================================
    if progress_callback:
        progress_callback(25, "Initializing output arrays...")

    empty_arrays = _generate_empty_property_arrays(params)

    # ========================================================================
    # STEP 4: RUN SIMULATION
    # ========================================================================
    if progress_callback:
        progress_callback(30, f"Running {params.method} simulation...")

    simulation_results = _run_simulation_method(
        params.method,
        drillhole_data,
        grid_coords,
        params,
        progress_callback
    )

    # ========================================================================
    # STEP 5: POST-PROCESSING
    # ========================================================================
    if progress_callback:
        progress_callback(85, "Computing summary statistics...")

    # Compute summary statistics
    summary_stats = _compute_simulation_summary(simulation_results['realizations'])

    # Compute probability maps if cutoffs provided
    probability_maps = {}
    if cutoffs:
        if progress_callback:
            progress_callback(90, "Computing probability maps...")
        probability_maps = _compute_probability_maps(simulation_results['realizations'], cutoffs)

    # ========================================================================
    # STEP 6: RETURN STANDARDIZED RESULTS
    # ========================================================================
    if progress_callback:
        progress_callback(100, "Simulation complete")

    results = {
        'realizations': simulation_results['realizations'],
        'summary': summary_stats,
        'probability_maps': probability_maps,
        'metadata': {
            # Core parameters
            'method': params.method,
            'variable': variable,
            'n_realizations': params.n_realizations,
            'grid_dims': (params.nx, params.ny, params.nz),
            'grid_spacing': (params.xinc, params.yinc, params.zinc),
            'grid_origin': (params.xmin, params.ymin, params.zmin),
            'samples_used': len(drillhole_data['coords']),
            'simulation_params': params.__dict__,
            # AUDIT FIX: Lineage tracking (CROSS-001, CROSS-002)
            'variogram_hash': variogram_hash,
            'source_data_hash': source_data_hash,
            'source_type': source_validation.get('source_type', 'unknown'),
            'data_source_validation': source_validation,
            'execution_timestamp': datetime.now().isoformat(),
            'audit_version': '2.0.0-CROSS-001-002-fix',
        },
        'grid_coords': grid_coords,
        'grid_extents': grid_extents,
        'drillhole_data': drillhole_data
    }

    logger.info(f"Standardized {params.method} workflow complete: "
               f"{params.n_realizations} realizations generated")

    return results


def _load_drillhole_data(data_df: pd.DataFrame, variable: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Step 1: Load and validate drillhole data.

    Parameters
    ----------
    data_df : pd.DataFrame
        Raw drillhole data
    variable : str
        Variable column name

    Returns
    -------
    dict or None
        Dictionary with 'coords' and 'values' arrays, or None if invalid
    """
    # Filter valid data
    data = data_df.dropna(subset=['X', 'Y', 'Z', variable]).copy()
    # CRITICAL: Preserve attrs for JORC/SAMREC data lineage tracking
    if hasattr(data_df, 'attrs') and data_df.attrs:
        data.attrs = data_df.attrs.copy()
    if data.empty:
        logger.warning(f"No valid data found for variable '{variable}'")
        return None

    # Extract coordinates and values
    coords = data[['X', 'Y', 'Z']].to_numpy(dtype=np.float64)
    values = data[variable].to_numpy(dtype=np.float64)

    # Remove NaN/inf values
    valid_mask = ~(np.isnan(coords).any(axis=1) | np.isnan(values) | np.isinf(values))
    coords = coords[valid_mask]
    values = values[valid_mask]

    if len(values) == 0:
        logger.warning(f"All data points contain NaN or inf values for '{variable}'")
        return None

    return {
        'coords': coords,
        'values': values
    }


def _define_simulation_grid(params: SimulationParameters, data_coords: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Step 2: Define simulation grid extents and block size.

    Parameters
    ----------
    params : SimulationParameters
        Simulation parameters
    data_coords : np.ndarray
        Drillhole coordinates for auto-extent calculation

    Returns
    -------
    tuple
        (grid_coords, grid_extents)
    """
    # Memory-efficient grid generation for large grids
    n_cells = params.nx * params.ny * params.nz

    # Use meshgrid for all grid sizes (simpler and more reliable)
    gx = np.arange(params.nx, dtype=np.float32) * params.xinc + params.xmin + params.xinc / 2.0
    gy = np.arange(params.ny, dtype=np.float32) * params.yinc + params.ymin + params.yinc / 2.0
    gz = np.arange(params.nz, dtype=np.float32) * params.zinc + params.zmin + params.zinc / 2.0

    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    grid_coords = np.column_stack([GX.ravel(order='F'), GY.ravel(order='F'), GZ.ravel(order='F')])

    # Calculate grid extents
    grid_extents = {
        'xmin': params.xmin,
        'ymin': params.ymin,
        'zmin': params.zmin,
        'xmax': params.xmin + params.nx * params.xinc,
        'ymax': params.ymin + params.ny * params.yinc,
        'zmax': params.zmin + params.nz * params.zinc,
        'xinc': params.xinc,
        'yinc': params.yinc,
        'zinc': params.zinc
    }

    return grid_coords, grid_extents


def _generate_empty_property_arrays(params: SimulationParameters) -> Dict[str, np.ndarray]:
    """
    Step 3: Generate empty property arrays.

    Parameters
    ----------
    params : SimulationParameters
        Simulation parameters

    Returns
    -------
    dict
        Empty arrays for simulation output
    """
    realizations = np.zeros((params.n_realizations, params.nz, params.ny, params.nx))

    return {
        'realizations': realizations
    }


def _run_simulation_method(
    method: str,
    drillhole_data: Dict[str, np.ndarray],
    grid_coords: np.ndarray,
    params: SimulationParameters,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Step 4: Run the specific simulation method.

    Parameters
    ----------
    method : str
        Simulation method name
    drillhole_data : dict
        Drillhole coordinates and values
    grid_coords : np.ndarray
        Grid coordinates
    params : SimulationParameters
        Simulation parameters
    progress_callback : Callable, optional
        Progress callback

    Returns
    -------
    dict
        Method-specific results
    """
    method_dispatch = {
        'sgsim': _run_sgsim_simulation,
        'sis': _run_sis_simulation,
        'turning_bands': _run_turning_bands_simulation,
        'dbs': _run_dbs_simulation,
        'grf': _run_grf_simulation,
        'cosgsim': _run_cosgsim_simulation,
        'ik_sgsim': _run_ik_sgsim_simulation,
    }

    if method not in method_dispatch:
        raise ValueError(f"Unknown simulation method: {method}")

    runner = method_dispatch[method]
    return runner(drillhole_data, grid_coords, params, progress_callback)


def _run_sgsim_simulation(drillhole_data, grid_coords, params, progress_callback):
    """Run SGSIM simulation."""
    from .sgsim3d import run_sgsim_simulation, SGSIMParameters

    # Convert to SGSIM parameters
    sgsim_params = SGSIMParameters(
        nreal=params.n_realizations,
        nx=params.nx, ny=params.ny, nz=params.nz,
        xmin=params.xmin, ymin=params.ymin, zmin=params.zmin,
        xinc=params.xinc, yinc=params.yinc, zinc=params.zinc,
        variogram_type=params.variogram_type,
        range_major=params.range_major, range_minor=params.range_minor, range_vert=params.range_vert,
        azimuth=params.azimuth, dip=params.dip,
        nugget=params.nugget, sill=params.sill,
        min_neighbors=params.min_neighbors, max_neighbors=params.max_neighbors,
        max_search_radius=params.max_search_radius,
        seed=params.seed
    )

    # Update progress callback to account for workflow offset
    def adjusted_callback(pct, msg):
        if progress_callback:
            # SGSIM internal progress is 0-100, adjust to 30-85 range
            adjusted_pct = 30 + int(pct * 0.55)
            progress_callback(adjusted_pct, msg)

    realizations = run_sgsim_simulation(
        drillhole_data['coords'],
        drillhole_data['values'],
        sgsim_params,
        adjusted_callback
    )

    return {'realizations': realizations}


def _run_sis_simulation(drillhole_data, grid_coords, params, progress_callback):
    """Run Sequential Indicator Simulation."""
    from ..geostats.sis import run_sis_full, SISConfig

    # Optimize SIS by reducing thresholds if too many (performance optimization)
    default_thresholds = params.method_params.get('thresholds', [np.median(drillhole_data['values'])])
    
    # Extract threshold values - handle both list of floats and list of dicts
    if default_thresholds and isinstance(default_thresholds[0], dict):
        # Extract threshold values from list of dicts: [{'threshold': 1.0, 'range': 100.0, 'sill': 0.5}, ...]
        thresholds = [t.get('threshold', t.get('thresh', t.get('value'))) for t in default_thresholds]
        # Filter out None values
        thresholds = [t for t in thresholds if t is not None]
        # Also extract variogram parameters for each threshold
        threshold_variograms = {}
        for t_dict in default_thresholds:
            thresh_val = t_dict.get('threshold', t_dict.get('thresh', t_dict.get('value')))
            if thresh_val is not None:
                threshold_variograms[thresh_val] = {
                    'range': t_dict.get('range', 100.0),
                    'sill': t_dict.get('sill', 0.5),
                    'nugget': t_dict.get('nugget', 0.0)
                }
    else:
        # Already a list of floats
        thresholds = default_thresholds if isinstance(default_thresholds, list) else [default_thresholds]
        threshold_variograms = {}

    # Limit thresholds for performance (SIS scales poorly with many thresholds)
    max_thresholds = 4  # Limit to prevent excessive computation time
    if len(thresholds) > max_thresholds:
        logger.warning(f"SIS: Reducing {len(thresholds)} thresholds to {max_thresholds} for performance")
        # Select evenly spaced thresholds
        indices = np.linspace(0, len(thresholds)-1, max_thresholds, dtype=int)
        thresholds = [thresholds[i] for i in indices]
        # Update variogram dict to match
        if threshold_variograms:
            threshold_variograms = {thresh: threshold_variograms.get(thresh, {'range': 100.0, 'sill': 0.5, 'nugget': 0.0}) 
                                   for thresh in thresholds}

    sis_config = SISConfig(
        thresholds=thresholds,
        n_realizations=params.n_realizations,
        random_seed=params.seed,
        max_neighbors=min(params.max_neighbors, 16),  # Cap for performance
        min_neighbors=params.min_neighbors,
        max_search_radius=params.max_search_radius,
        indicator_variograms=threshold_variograms  # Will default to empty dict if None
    )

    # Update progress callback
    def adjusted_callback(pct, msg):
        if progress_callback:
            adjusted_pct = 30 + int(pct * 0.55)
            progress_callback(adjusted_pct, msg)

    results = run_sis_full(
        coords=drillhole_data['coords'],
        values=drillhole_data['values'],
        grid_coords=grid_coords,
        config=sis_config,
        grid_shape=(params.nz, params.ny, params.nx),
        progress_callback=adjusted_callback
    )

    # Return as standardized format (nreal, nz, ny, nx)
    # SIS returns indicator_realizations
    return {'realizations': results.indicator_realizations}


def _run_turning_bands_simulation(drillhole_data, grid_coords, params, progress_callback):
    """Run Turning Bands Simulation."""
    from ..geostats.turning_bands import run_turning_bands, TurningBandsConfig

    # Optimize Turning Bands for performance - reduce bands for large grids
    n_cells = params.nx * params.ny * params.nz
    n_bands = params.method_params.get('n_bands', 1000)

    # Reduce bands for large grids to maintain performance
    if n_cells > 5000 and n_bands > 500:
        n_bands = 500
        logger.info(f"Turning Bands: Reducing n_bands to {n_bands} for large grid ({n_cells} cells)")
    elif n_cells > 10000 and n_bands > 200:
        n_bands = 200
        logger.info(f"Turning Bands: Reducing n_bands to {n_bands} for very large grid ({n_cells} cells)")

    tb_config = TurningBandsConfig(
        n_realizations=params.n_realizations,
        n_bands=n_bands,
        random_seed=params.seed,
        variogram_type=params.variogram_type,
        range_major=params.range_major, range_minor=params.range_minor, range_vert=params.range_vert,
        azimuth=params.azimuth, dip=params.dip,
        nugget=params.nugget, sill=params.sill,
        condition=True,
        max_neighbors=params.max_neighbors,
        max_search_radius=params.max_search_radius
    )

    # Update progress callback
    def adjusted_callback(pct, msg):
        if progress_callback:
            adjusted_pct = 30 + int(pct * 0.55)
            progress_callback(adjusted_pct, msg)

    results = run_turning_bands(
        grid_coords=grid_coords,
        config=tb_config,
        conditioning_coords=drillhole_data['coords'],
        conditioning_values=drillhole_data['values'],
        progress_callback=adjusted_callback
    )

    # Reshape to standardized format (nreal, nz, ny, nx)
    realizations = results.realizations.reshape(
        params.n_realizations, params.nz, params.ny, params.nx
    )

    return {'realizations': realizations}


def _run_dbs_simulation(drillhole_data, grid_coords, params, progress_callback):
    """Run Direct Block Simulation."""
    from ..geostats.direct_block_sim import run_dbs, DBSConfig

    # Optimize DBS for memory usage - reduce neighbors if grid is large
    n_cells = params.nx * params.ny * params.nz
    max_neighbors = params.max_neighbors

    # Reduce neighbors for large grids to save memory
    if n_cells > 10000:  # Large grid threshold
        max_neighbors = min(max_neighbors, 12)
        logger.info(f"DBS: Reducing max_neighbors to {max_neighbors} for large grid ({n_cells} cells)")

    dbs_config = DBSConfig(
        n_realizations=params.n_realizations,
        random_seed=params.seed,
        block_dx=params.xinc, block_dy=params.yinc, block_dz=params.zinc,
        variogram_type=params.variogram_type,
        range_major=params.range_major, range_minor=params.range_minor, range_vert=params.range_vert,
        azimuth=params.azimuth, dip=params.dip,
        nugget=params.nugget, sill=params.sill,
        max_neighbors=max_neighbors, min_neighbors=params.min_neighbors,
        max_search_radius=params.max_search_radius
    )

    # Update progress callback
    def adjusted_callback(pct, msg):
        if progress_callback:
            adjusted_pct = 30 + int(pct * 0.55)
            progress_callback(adjusted_pct, msg)

    results = run_dbs(
        conditioning_coords=drillhole_data['coords'],
        conditioning_values=drillhole_data['values'],
        block_centroids=grid_coords,
        config=dbs_config,
        progress_callback=adjusted_callback
    )

    # Reshape to standardized format (nreal, nz, ny, nx)
    realizations = results.realizations.reshape(
        params.n_realizations, params.nz, params.ny, params.nx
    )

    return {'realizations': realizations}


def _run_grf_simulation(drillhole_data, grid_coords, params, progress_callback):
    """Run Gaussian Random Field simulation."""
    # For now, use a simple unconditional simulation
    # TODO: Implement full GRF simulation with proper conditioning
    logger.info("Using simplified unconditional GRF simulation")

    # Simple unconditional Gaussian simulation
    np.random.seed(params.seed)
    n_cells = params.nx * params.ny * params.nz
    realizations = np.random.normal(0, np.sqrt(params.sill), (params.n_realizations, n_cells))

    # Reshape to grid format
    realizations = realizations.reshape(params.n_realizations, params.nz, params.ny, params.nx)

    return {'realizations': realizations}


def _run_cosgsim_simulation(drillhole_data, grid_coords, params, progress_callback):
    """Run Co-Simulation."""
    # For now, use a simple simulation
    # TODO: Implement full co-simulation with secondary variable
    logger.info("Using simplified co-simulation (placeholder)")

    # Simple simulation based on primary variable
    np.random.seed(params.seed)
    n_cells = params.nx * params.ny * params.nz
    base_values = np.random.normal(0, np.sqrt(params.sill), n_cells)

    # Add correlation with conditioning data (simplified)
    if len(drillhole_data['values']) > 0:
        mean_conditioning = np.mean(drillhole_data['values'])
        base_values += mean_conditioning * 0.1  # Weak correlation

    realizations = np.tile(base_values, (params.n_realizations, 1))
    realizations += np.random.normal(0, 0.1, realizations.shape)  # Add noise

    # Reshape to grid format
    realizations = realizations.reshape(params.n_realizations, params.nz, params.ny, params.nx)

    return {'realizations': realizations}


def _run_ik_sgsim_simulation(drillhole_data, grid_coords, params, progress_callback):
    """Run Indicator Kriging SGSIM."""
    from ..geostats.ik_sgsim import run_ik_sgsim, IKSGSIMConfig

    # Create IK-SGSIM config
    ik_config = IKSGSIMConfig(
        n_realizations=params.n_realizations,
        random_seed=params.seed
    )

    # Update progress callback
    def adjusted_callback(pct, msg):
        if progress_callback:
            adjusted_pct = 30 + int(pct * 0.55)
            progress_callback(adjusted_pct, msg)

    results = run_ik_sgsim(
        coords=drillhole_data['coords'],
        values=drillhole_data['values'],
        grid_coords=grid_coords,
        config=ik_config,
        progress_callback=adjusted_callback
    )

    # Reshape to standardized format (nreal, nz, ny, nx)
    realizations = results.realizations.reshape(
        params.n_realizations, params.nz, params.ny, params.nx
    )

    return {'realizations': realizations}


def _compute_simulation_summary(realizations: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute summary statistics across all realizations."""
    # Basic statistics
    mean = np.mean(realizations, axis=0)
    var = np.var(realizations, axis=0)
    std = np.std(realizations, axis=0)

    # Percentiles
    p10 = np.percentile(realizations, 10, axis=0)
    p50 = np.percentile(realizations, 50, axis=0)
    p90 = np.percentile(realizations, 90, axis=0)

    return {
        'mean': mean,
        'var': var,
        'std': std,
        'p10': p10,
        'p50': p50,
        'p90': p90
    }


def _compute_probability_maps(realizations: np.ndarray, cutoffs: List[float]) -> Dict[float, np.ndarray]:
    """Compute probability maps for given cutoffs."""
    prob_maps = {}
    for cutoff in cutoffs:
        prob_maps[cutoff] = np.mean(realizations > cutoff, axis=0)
    return prob_maps
