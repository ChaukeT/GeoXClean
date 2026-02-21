"""
Multiple-Point Simulation (MPS)
===============================

Pattern-based simulation using training images instead of variograms.

Industry Standard Implementation:
- Scans training image for multi-point patterns
- Reproduces complex geological structures
- Handles channels, folds, lenses, faults, anisotropic features

Use Cases:
- Highly complex geology
- Channelized systems (fluvial, turbidite)
- Vein networks
- Facies modeling
- Structural geology with non-stationary features

References:
- Strebelle (2002) - SNESIM algorithm
- Mariethoz & Caers (2015) - Multiple-Point Geostatistics
- Guardiano & Srivastava (1993) - Original MPS concept

AUDIT NOTES:
- MPS does NOT use variograms (uses training image patterns instead)
- No variogram consistency check needed (N/A)
- CROSS-002: Data source validation should be done at controller level

AUDIT FIXES APPLIED:
- MPS-001: Added configuration validation
- MPS-002: Complete lineage metadata in results
"""

import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

# Try to import numba for acceleration
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MPSConfig:
    """Configuration for Multiple-Point Simulation."""
    n_realizations: int = 10
    random_seed: Optional[int] = None
    
    # Template/pattern parameters
    template_size: Tuple[int, int, int] = (5, 5, 3)  # Half-size in each direction
    max_patterns: int = 10000  # Maximum patterns to store
    
    # Search parameters
    max_neighbors: int = 30
    min_replicates: int = 3  # Minimum pattern matches required
    
    # Servo system (for target proportion)
    use_servo: bool = True
    servo_factor: float = 1.0
    
    # Categories (for categorical simulation)
    categories: Optional[List[int]] = None  # Auto-detect if None
    
    realization_prefix: str = "mps"
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters.
        
        AUDIT FIX MPS-001: Ensure parameters are valid before simulation.
        """
        errors = []
        if self.n_realizations <= 0:
            errors.append(f"n_realizations must be positive (got {self.n_realizations})")
        if any(t <= 0 for t in self.template_size):
            errors.append(f"template_size dimensions must be positive (got {self.template_size})")
        if self.max_patterns <= 0:
            errors.append(f"max_patterns must be positive (got {self.max_patterns})")
        if self.max_neighbors <= 0:
            errors.append(f"max_neighbors must be positive (got {self.max_neighbors})")
        if self.min_replicates <= 0:
            errors.append(f"min_replicates must be positive (got {self.min_replicates})")
        if self.servo_factor < 0 or self.servo_factor > 10:
            errors.append(f"servo_factor should be in range [0, 10] (got {self.servo_factor})")
        return errors
    
    def compute_hash(self) -> str:
        """
        Compute deterministic hash for lineage tracking.
        
        AUDIT FIX: Enable configuration hash validation.
        Note: MPS does not use variograms, so this is a configuration hash.
        """
        params_str = (
            f"{self.template_size}:{self.max_patterns}:{self.max_neighbors}:"
            f"{self.min_replicates}:{self.use_servo}:{self.servo_factor}:"
            f"{self.n_realizations}:{self.random_seed}"
        )
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]


@dataclass
class MPSResult:
    """Result from Multiple-Point Simulation."""
    realizations: np.ndarray  # [n_realizations, nz, ny, nx]
    realization_names: List[str]
    category_proportions: Dict[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternDatabase:
    """
    Database for storing and querying multi-point patterns.
    
    Uses a tree structure indexed by data event (neighboring values).
    """
    
    def __init__(self, max_patterns: int = 10000):
        self.max_patterns = max_patterns
        self.patterns = defaultdict(list)  # data_event -> list of center values
        self.n_patterns = 0
    
    def add_pattern(self, data_event: Tuple, center_value: int):
        """Add a pattern to the database."""
        if self.n_patterns < self.max_patterns:
            self.patterns[data_event].append(center_value)
            self.n_patterns += 1
    
    def get_cpdf(self, data_event: Tuple, categories: List[int]) -> Dict[int, float]:
        """
        Get conditional probability distribution for given data event.
        
        Returns probability of each category given the data event.
        """
        matches = self.patterns.get(data_event, [])
        
        if len(matches) < 3:
            # Not enough matches - return uniform
            return {cat: 1.0 / len(categories) for cat in categories}
        
        # Count occurrences
        counts = defaultdict(int)
        for val in matches:
            counts[val] += 1
        
        total = len(matches)
        return {cat: counts.get(cat, 0) / total for cat in categories}


if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def _scan_ti_numba(ti, hz, hy, hx):
        """
        Numba-accelerated Training Image scanner.
        Returns: (patterns_array, centers_array)
        patterns_array: Flattened patterns (n_patterns, pattern_size)
        """
        nz, ny, nx = ti.shape
        
        # Calculate valid range
        z_start, z_end = hz, nz - hz
        y_start, y_end = hy, ny - hy
        x_start, x_end = hx, nx - hx
        
        n_pats = (z_end - z_start) * (y_end - y_start) * (x_end - x_start)
        
        # Template size (excluding center)
        pat_size = (2*hz+1) * (2*hy+1) * (2*hx+1)
        
        # Output arrays
        patterns = np.empty((n_pats, pat_size), dtype=ti.dtype)
        centers = np.empty(n_pats, dtype=ti.dtype)
        
        idx = 0
        center_offset = (hz * (2*hy+1) * (2*hx+1)) + (hy * (2*hx+1)) + hx
        
        for z in range(z_start, z_end):
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    # Extract neighborhood
                    pat = ti[z-hz:z+hz+1, y-hy:y+hy+1, x-hx:x+hx+1]
                    flat_pat = pat.flatten()
                    
                    patterns[idx] = flat_pat
                    centers[idx] = ti[z, y, x]  # Center value
                    
                    idx += 1
                    
        return patterns, centers
else:
    # Fallback if Numba not available
    def _scan_ti_numba(ti, hz, hy, hx):
        # This won't be called if NUMBA_AVAILABLE is False
        raise NotImplementedError("Numba not available")


def _scan_training_image(
    ti: np.ndarray,
    template_half_size: Tuple[int, int, int],
    max_patterns: int
) -> PatternDatabase:
    """
    Scan training image and build pattern database.
    
    Uses Numba-accelerated scanner if available for performance.
    
    Args:
        ti: Training image array (nz, ny, nx)
        template_half_size: Half-size of template in each direction
        max_patterns: Maximum patterns to store
    
    Returns:
        PatternDatabase populated from training image
    """
    db = PatternDatabase(max_patterns)
    
    nz, ny, nx = ti.shape
    hz, hy, hx = template_half_size
    
    # Use Numba scanner if available
    if NUMBA_AVAILABLE:
        try:
            pats, centers = _scan_ti_numba(ti, hz, hy, hx)
            
            # Center index in flattened array
            pat_len = pats.shape[1]
            center_idx = pat_len // 2
            
            # Masking out center for the key
            mask = np.ones(pat_len, dtype=bool)
            mask[center_idx] = False
            
            # Populate database
            for i in range(min(len(pats), max_patterns)):
                # Create key tuple from neighbors only (excluding center)
                key = tuple(int(pats[i, j]) for j in range(pat_len) if mask[j])
                db.add_pattern(key, int(centers[i]))
            
            logger.info(f"Scanned training image (Numba): {db.n_patterns} patterns stored")
            return db
        except Exception as e:
            logger.warning(f"Numba scan failed, falling back to Python: {e}")
            # Fall through to Python implementation
    
    # Python fallback (original implementation)
    for z in range(hz, nz - hz):
        for y in range(hy, ny - hy):
            for x in range(hx, nx - hx):
                if db.n_patterns >= max_patterns:
                    break
                
                # Extract template neighborhood
                neighborhood = ti[
                    z - hz:z + hz + 1,
                    y - hy:y + hy + 1,
                    x - hx:x + hx + 1
                ]
                
                # Create data event (exclude center)
                center_value = int(ti[z, y, x])
                
                # Flatten and create hashable tuple
                flat = neighborhood.flatten()
                center_flat_idx = hz * (2*hy+1) * (2*hx+1) + hy * (2*hx+1) + hx
                data_event = tuple(int(v) for i, v in enumerate(flat) if i != center_flat_idx)
                
                db.add_pattern(data_event, center_value)
    
    logger.info(f"Scanned training image: {db.n_patterns} patterns stored")
    return db


def _get_data_event(
    sim: np.ndarray,
    z: int, y: int, x: int,
    template_half_size: Tuple[int, int, int]
) -> Tuple:
    """
    Extract data event from current simulation state.
    
    Only includes already-simulated values (not NaN).
    """
    nz, ny, nx = sim.shape
    hz, hy, hx = template_half_size
    
    values = []
    
    for dz in range(-hz, hz + 1):
        for dy in range(-hy, hy + 1):
            for dx in range(-hx, hx + 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue  # Skip center
                
                nz_idx = z + dz
                ny_idx = y + dy
                nx_idx = x + dx
                
                if 0 <= nz_idx < nz and 0 <= ny_idx < ny and 0 <= nx_idx < nx:
                    val = sim[nz_idx, ny_idx, nx_idx]
                    if not np.isnan(val):
                        values.append(int(val))
                    else:
                        values.append(-999)  # Unknown
                else:
                    values.append(-999)  # Outside bounds
    
    return tuple(values)


def run_mps(
    grid_shape: Tuple[int, int, int],
    training_image: np.ndarray,
    config: MPSConfig,
    conditioning_coords: Optional[np.ndarray] = None,
    conditioning_values: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable] = None,
    source_data_hash: Optional[str] = None
) -> MPSResult:
    """
    Run Multiple-Point Simulation using SNESIM-like algorithm.
    
    Industry-standard algorithm:
    1. Scan training image to build pattern database
    2. For each realization:
       a. Initialize grid (NaN for unsimulated, values for conditioning)
       b. Visit nodes in random order
       c. At each node:
          - Extract data event from neighbors
          - Query pattern database for CPDF
          - Draw from CPDF
       d. Apply servo system if enabled
    
    AUDIT NOTES:
    - MPS uses training image patterns, NOT variograms
    - Variogram consistency check is N/A for MPS
    - Data source validation should occur at controller level (CROSS-002)
    
    AUDIT FIXES APPLIED:
    - MPS-001: Configuration validation added
    - MPS-002: Complete lineage metadata in results
    
    Args:
        grid_shape: Output grid shape (nz, ny, nx)
        training_image: Training image array (must be same or larger than grid)
        config: MPS configuration
        conditioning_coords: Optional conditioning data coordinates
        conditioning_values: Optional conditioning data values
        progress_callback: Optional progress callback
        source_data_hash: Optional hash of source data for lineage
    
    Returns:
        MPSResult with simulated realizations
        
    Raises:
        ValueError: If configuration validation fails
    """
    # AUDIT FIX MPS-001: Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        error_msg = "MPS configuration validation failed: " + "; ".join(validation_errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Starting MPS: {config.n_realizations} realizations, grid {grid_shape}")
    
    # Set random seed
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
    
    nz, ny, nx = grid_shape
    n_cells = nz * ny * nx
    
    # Detect categories
    if config.categories is None:
        unique_vals = np.unique(training_image[~np.isnan(training_image)])
        categories = [int(v) for v in unique_vals]
    else:
        categories = config.categories
    
    logger.info(f"Categories: {categories}")
    
    # Compute target proportions from training image
    target_proportions = {}
    valid_ti = training_image[~np.isnan(training_image)]
    for cat in categories:
        target_proportions[cat] = np.sum(valid_ti == cat) / len(valid_ti)
    
    # Scan training image
    pattern_db = _scan_training_image(
        training_image,
        config.template_size,
        config.max_patterns
    )
    
    # Initialize realizations
    realizations = np.zeros((config.n_realizations, nz, ny, nx))
    realization_names = []
    
    # Process conditioning data
    has_conditioning = conditioning_coords is not None and conditioning_values is not None
    
    # Main simulation loop
    for ireal in range(config.n_realizations):
        if progress_callback:
            progress_callback(ireal + 1, f"Realization {ireal + 1}/{config.n_realizations}")
        
        # Initialize simulation grid
        sim = np.full((nz, ny, nx), np.nan)
        
        # Apply conditioning data
        if has_conditioning:
            for i, (coord, val) in enumerate(zip(conditioning_coords, conditioning_values)):
                iz = int(np.clip(coord[2], 0, nz - 1))
                iy = int(np.clip(coord[1], 0, ny - 1))
                ix = int(np.clip(coord[0], 0, nx - 1))
                sim[iz, iy, ix] = val
        
        # Random path
        indices = [(z, y, x) for z in range(nz) for y in range(ny) for x in range(nx)]
        np.random.shuffle(indices)
        
        # Current proportions for servo system
        current_counts = {cat: 0 for cat in categories}
        n_simulated = 0
        
        # Simulate each node
        for z, y, x in indices:
            if not np.isnan(sim[z, y, x]):
                # Already conditioned
                current_counts[int(sim[z, y, x])] += 1
                n_simulated += 1
                continue
            
            # Get data event
            data_event = _get_data_event(sim, z, y, x, config.template_size)
            
            # Query pattern database
            cpdf = pattern_db.get_cpdf(data_event, categories)
            
            # Apply servo system
            if config.use_servo and n_simulated > 0:
                for cat in categories:
                    current_prop = current_counts[cat] / n_simulated
                    target_prop = target_proportions[cat]
                    
                    if current_prop < target_prop:
                        cpdf[cat] *= (1 + config.servo_factor * (target_prop - current_prop))
                    else:
                        cpdf[cat] *= (1 - config.servo_factor * (current_prop - target_prop))
                
                # Normalize
                total = sum(cpdf.values())
                if total > 0:
                    cpdf = {cat: p / total for cat, p in cpdf.items()}
            
            # Draw from CPDF
            probs = [cpdf[cat] for cat in categories]
            probs = np.array(probs)
            probs = np.maximum(probs, 0)
            probs /= probs.sum()
            
            sim_value = np.random.choice(categories, p=probs)
            sim[z, y, x] = sim_value
            
            current_counts[sim_value] += 1
            n_simulated += 1
        
        realizations[ireal] = sim
        realization_names.append(f"{config.realization_prefix}_{ireal + 1:04d}")
    
    # Compute final proportions
    final_proportions = {}
    all_reals = realizations.flatten()
    for cat in categories:
        final_proportions[cat] = np.sum(all_reals == cat) / len(all_reals)
    
    logger.info(f"MPS complete: {config.n_realizations} realizations")
    
    # Compute training image hash for lineage
    ti_hash = hashlib.sha256(training_image.tobytes()).hexdigest()[:16]
    
    # AUDIT FIX MPS-002: Complete lineage metadata
    return MPSResult(
        realizations=realizations,
        realization_names=realization_names,
        category_proportions=final_proportions,
        metadata={
            # Core parameters
            'n_realizations': config.n_realizations,
            'grid_shape': grid_shape,
            'template_size': config.template_size,
            'max_patterns': config.max_patterns,
            'max_neighbors': config.max_neighbors,
            'min_replicates': config.min_replicates,
            'use_servo': config.use_servo,
            'servo_factor': config.servo_factor,
            'categories': categories,
            'target_proportions': target_proportions,
            'final_proportions': final_proportions,
            'n_patterns': pattern_db.n_patterns,
            'n_conditioning': len(conditioning_coords) if has_conditioning else 0,
            'method': 'Multiple-Point Simulation (SNESIM)',
            # AUDIT FIX: Lineage tracking
            'config_hash': config.compute_hash(),
            'training_image_hash': ti_hash,
            'source_data_hash': source_data_hash,
            'execution_timestamp': datetime.now().isoformat(),
            'variogram_used': False,  # MPS does NOT use variograms
            'audit_note': 'MPS uses training image patterns, variogram N/A',
            'audit_version': '2.0.0-MPS-001-fix',
        }
    )

