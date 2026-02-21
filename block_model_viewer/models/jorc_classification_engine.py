"""
JORC/SAMREC-Compliant Resource Classification Engine
====================================================

Industry-standard resource classification system implementing the specification:
- Optional domain selection (classifies full model extent if no domain selected)
- Automatic distance calculations using variogram-normalized isotropic space
- Percentage-based thresholds (% of variogram range)
- cKDTree-based spatial search for performance

Compliant with:
- JORC 2012 Code (Table 1, Section 3)
- SAMREC Code (Table 1)
- CIM Definition Standards

Author: GeoX Mining Software Platform
"""

from __future__ import annotations

import logging
import math
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal, Any, Tuple, Callable, Union, TYPE_CHECKING
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from .block_model import BlockModel

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Audit Hashing Functions (JORC Compliance)
# ------------------------------------------------------------------ #

def _hash_variogram(variogram: "VariogramModel") -> str:
    """
    Generate deterministic hash of variogram parameters for audit trail.
    
    This ensures reproducibility verification for JORC/SAMREC compliance.
    
    Returns:
        16-character hex hash of variogram configuration
    """
    params = {
        "azimuth": round(variogram.azimuth, 6),
        "dip": round(variogram.dip, 6),
        "pitch": round(variogram.pitch, 6),
        "range_major": round(variogram.range_major, 6),
        "range_semi": round(variogram.range_semi, 6),
        "range_minor": round(variogram.range_minor, 6),
        "sill": round(variogram.sill, 6),
        "nugget": round(variogram.nugget, 6),
    }
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()[:16]


def _hash_ruleset(ruleset: "ClassificationRuleset") -> str:
    """
    Generate deterministic hash of classification ruleset for audit trail.
    
    This ensures reproducibility verification for JORC/SAMREC compliance.
    
    Returns:
        16-character hex hash of ruleset configuration
    """
    # Extract critical classification parameters
    params = {
        "measured": {
            "max_iso_distance": round(ruleset.measured.max_iso_distance, 6),
            "min_unique_holes": ruleset.measured.min_unique_holes,
            "max_kv_ratio": round(ruleset.measured.max_kv_ratio, 6) if ruleset.measured.max_kv_ratio else None,
            "min_slope": round(ruleset.measured.min_slope, 6) if ruleset.measured.min_slope else None,
        },
        "indicated": {
            "max_iso_distance": round(ruleset.indicated.max_iso_distance, 6),
            "min_unique_holes": ruleset.indicated.min_unique_holes,
            "max_kv_ratio": round(ruleset.indicated.max_kv_ratio, 6) if ruleset.indicated.max_kv_ratio else None,
            "min_slope": round(ruleset.indicated.min_slope, 6) if ruleset.indicated.min_slope else None,
        },
        "inferred": {
            "max_iso_distance": round(ruleset.inferred.max_iso_distance, 6),
            "min_unique_holes": ruleset.inferred.min_unique_holes,
        },
        "geology_confidence": ruleset.geology_confidence,
    }
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()[:16]

# Try to import numba for acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.debug("Numba not available, using numpy fallback for classification")


# ------------------------------------------------------------------ #
# Optimized Unique Hole Counting (Numba-accelerated or Numpy)
# ------------------------------------------------------------------ #

def _encode_hole_ids_vectorized(neighbor_hole_ids: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of hole IDs to numeric codes.
    Much faster than nested loops for large arrays.
    
    Parameters
    ----------
    neighbor_hole_ids : np.ndarray
        (n_blocks, k_neighbors) array of hole IDs (strings or objects)
    
    Returns
    -------
    np.ndarray
        (n_blocks, k_neighbors) array of int32 codes
    """
    # Flatten for unique/mapping
    flat = neighbor_hole_ids.flatten()
    
    # Get unique values and create mapping
    unique_ids, inverse = np.unique(flat, return_inverse=True)
    
    # Reshape back to original shape
    return inverse.reshape(neighbor_hole_ids.shape).astype(np.int32)


def _count_unique_holes_fast(
    neighbor_hole_ids: np.ndarray,
    dists_iso: np.ndarray,
    meas_thresh: float,
    ind_thresh: float,
    inf_thresh: float,
    progress_callback=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Count unique holes within thresholds for each block.
    Uses Numba JIT if available, otherwise optimized numpy.
    
    Returns:
        dist_to_1st, dist_to_2nd, dist_to_3rd,
        n_holes_within_meas, n_holes_within_ind, n_holes_within_inf
    """
    n_blocks = neighbor_hole_ids.shape[0]
    k_neighbors = neighbor_hole_ids.shape[1]
    
    # Initialize output arrays
    dist_to_1st = np.full(n_blocks, np.inf)
    dist_to_2nd = np.full(n_blocks, np.inf)
    dist_to_3rd = np.full(n_blocks, np.inf)
    n_holes_within_meas = np.zeros(n_blocks, dtype=np.int32)
    n_holes_within_ind = np.zeros(n_blocks, dtype=np.int32)
    n_holes_within_inf = np.zeros(n_blocks, dtype=np.int32)
    
    use_numba = False

    # Ensure int32 for Numba (should already be int32 from early encoding)
    if neighbor_hole_ids.dtype != np.int32:
        neighbor_hole_ids_numeric = neighbor_hole_ids.astype(np.int32)
    else:
        neighbor_hole_ids_numeric = neighbor_hole_ids
    
    if NUMBA_AVAILABLE and neighbor_hole_ids_numeric is not None:
        # Use Numba-accelerated version
        try:
            if progress_callback:
                progress_callback(50, "Running Numba-accelerated classification...")
            logger.info(f"Using Numba acceleration: {n_blocks:,} blocks, {k_neighbors} neighbors/block")
            _count_unique_holes_numba(
                neighbor_hole_ids_numeric, dists_iso,
                meas_thresh, ind_thresh, inf_thresh,
                dist_to_1st, dist_to_2nd, dist_to_3rd,
                n_holes_within_meas, n_holes_within_ind, n_holes_within_inf
            )
            use_numba = True  # Success
            logger.info("Numba classification completed successfully")
            if progress_callback:
                progress_callback(70, "Numba classification complete")
        except Exception as e:
            # Fallback to numpy if Numba fails
            logger.warning(f"Numba execution failed, using numpy fallback: {e}", exc_info=True)
            use_numba = False
    
    if not use_numba:
        # Optimized numpy fallback with early termination
        logger.info(f"Using numpy fallback (Numba {'not available' if not NUMBA_AVAILABLE else 'failed'}): {n_blocks:,} blocks, {k_neighbors} neighbors/block")
        if progress_callback:
            progress_callback(50, "Running numpy classification (optimized)...")

        # Use numeric hole IDs for faster processing
        hole_ids_to_use = neighbor_hole_ids_numeric if neighbor_hole_ids_numeric is not None else neighbor_hole_ids

        # Pre-compute max threshold for early exit
        max_thresh = max(meas_thresh, ind_thresh, inf_thresh)

        # Process in chunks for progress updates and memory efficiency
        chunk_size = max(10000, n_blocks // 20)  # Larger chunks with early exit

        for start in range(0, n_blocks, chunk_size):
            end = min(start + chunk_size, n_blocks)
            chunk_slice = slice(start, end)

            # Extract chunk data
            chunk_hole_ids = hole_ids_to_use[chunk_slice]  # (chunk_size, k_neighbors)
            chunk_dists = dists_iso[chunk_slice]  # (chunk_size, k_neighbors)
            chunk_n = end - start

            # Process each block with early termination
            for i in range(chunk_n):
                block_hole_ids = chunk_hole_ids[i]  # (k_neighbors,)
                block_dists = chunk_dists[i]  # (k_neighbors,)

                # Track unique holes with early exit
                seen_holes = set()
                unique_dists_list = []

                for j in range(len(block_hole_ids)):
                    hid = block_hole_ids[j]
                    dist = block_dists[j]

                    # Early exit: found 3 unique holes AND beyond max threshold
                    if len(seen_holes) >= 3 and dist > max_thresh:
                        break

                    if hid not in seen_holes:
                        seen_holes.add(hid)
                        unique_dists_list.append(dist)

                        # Count within thresholds
                        if dist <= meas_thresh:
                            n_holes_within_meas[start + i] += 1
                        if dist <= ind_thresh:
                            n_holes_within_ind[start + i] += 1
                        if dist <= inf_thresh:
                            n_holes_within_inf[start + i] += 1

                # Extract distances to 1st, 2nd, 3rd unique holes
                n_unique = len(unique_dists_list)
                if n_unique >= 1:
                    dist_to_1st[start + i] = unique_dists_list[0]
                if n_unique >= 2:
                    dist_to_2nd[start + i] = unique_dists_list[1]
                if n_unique >= 3:
                    dist_to_3rd[start + i] = unique_dists_list[2]

            # Progress update
            if progress_callback:
                pct = 50 + int(20 * end / n_blocks)
                progress_callback(pct, f"Counting unique holes... {end:,}/{n_blocks:,}")
    
    return (dist_to_1st, dist_to_2nd, dist_to_3rd,
            n_holes_within_meas, n_holes_within_ind, n_holes_within_inf)


if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _count_unique_holes_numba(
        neighbor_hole_ids: np.ndarray,  # int32 array (numeric hole ID codes)
        dists_iso: np.ndarray,
        meas_thresh: float,
        ind_thresh: float,
        inf_thresh: float,
        dist_to_1st: np.ndarray,
        dist_to_2nd: np.ndarray,
        dist_to_3rd: np.ndarray,
        n_holes_within_meas: np.ndarray,
        n_holes_within_ind: np.ndarray,
        n_holes_within_inf: np.ndarray
    ):
        """Numba-accelerated unique hole counting with parallel execution.

        OPTIMIZED: Uses early termination once 3 unique holes found AND distance
        exceeds max threshold. Neighbors are sorted by distance from KDTree query.

        Note: neighbor_hole_ids must be int32 (numeric codes), not strings.
        String hole IDs are converted to numeric codes before calling this function.
        """
        n_blocks = neighbor_hole_ids.shape[0]
        k_neighbors = neighbor_hole_ids.shape[1]

        # Pre-compute max threshold for early exit
        max_thresh = max(meas_thresh, ind_thresh, inf_thresh)

        for i in prange(n_blocks):
            # Track unique holes - use small fixed array (we only need up to ~10-20 unique)
            # Using hash-like approach with modulo for O(1) lookup
            seen_count = 0
            seen_holes = np.empty(64, dtype=np.int32)  # Small fixed buffer
            unique_dists = np.empty(64, dtype=np.float64)

            for j in range(k_neighbors):
                hid = neighbor_hole_ids[i, j]
                dist = dists_iso[i, j]

                # EARLY EXIT: If we've found 3 unique holes AND current distance
                # exceeds max threshold, we're done (neighbors sorted by distance)
                if seen_count >= 3 and dist > max_thresh:
                    break

                # Check if already seen (linear search in small buffer is fast)
                is_new = True
                for s in range(seen_count):
                    if seen_holes[s] == hid:
                        is_new = False
                        break

                if is_new and seen_count < 64:  # Prevent buffer overflow
                    seen_holes[seen_count] = hid
                    unique_dists[seen_count] = dist
                    seen_count += 1

                    # Count within thresholds
                    if dist <= meas_thresh:
                        n_holes_within_meas[i] += 1
                    if dist <= ind_thresh:
                        n_holes_within_ind[i] += 1
                    if dist <= inf_thresh:
                        n_holes_within_inf[i] += 1

            # Extract distances to 1st, 2nd, 3rd unique holes
            if seen_count >= 1:
                dist_to_1st[i] = unique_dists[0]
            if seen_count >= 2:
                dist_to_2nd[i] = unique_dists[1]
            if seen_count >= 3:
                dist_to_3rd[i] = unique_dists[2]


# ------------------------------------------------------------------ #
# Type Aliases & Constants
# ------------------------------------------------------------------ #

ClassificationLevel = Literal["Measured", "Indicated", "Inferred", "Unclassified"]
GeoConfLevel = Literal["high", "medium", "low", "not_specified"]

# JORC-standard classification colors
CLASSIFICATION_COLORS = {
    "Measured": "#2ca02c",      # Green - high confidence
    "Indicated": "#ffbf00",     # Amber - medium confidence  
    "Inferred": "#d62728",      # Red - low confidence
    "Unclassified": "#7f7f7f",  # Grey - insufficient data
}

CLASSIFICATION_ORDER = ["Measured", "Indicated", "Inferred", "Unclassified"]

# ------------------------------------------------------------------ #
# Configuration Dataclasses
# ------------------------------------------------------------------ #

@dataclass
class VariogramModel:
    """
    Variogram model parameters for anisotropic search ellipsoid.
    
    All angles in degrees, ranges in meters.
    Follows mining industry convention (Z-X-Y rotation order).
    """
    azimuth: float = 0.0       # Rotation around Z (0-360, clockwise from North)
    dip: float = 0.0           # Rotation around X (-90 to 90, positive down)
    pitch: float = 0.0         # Rotation around Y (rake angle)
    range_major: float = 100.0 # Primary axis range (meters)
    range_semi: float = 80.0   # Secondary axis range (meters)
    range_minor: float = 40.0  # Tertiary axis range (meters)
    sill: float = 1.0          # Total sill (nugget + partial sills)
    nugget: float = 0.0        # Nugget effect
    
    def __post_init__(self):
        """Validate variogram parameters."""
        if self.range_major <= 0 or self.range_semi <= 0 or self.range_minor <= 0:
            raise ValueError("All variogram ranges must be positive")
        if self.sill <= 0:
            raise ValueError("Sill must be positive")
        # Ensure proper range ordering
        if not (self.range_major >= self.range_semi >= self.range_minor):
            logger.warning(
                f"Range ordering violated: major={self.range_major}, "
                f"semi={self.range_semi}, minor={self.range_minor}. "
                "Consider reordering for proper anisotropy."
            )

    @property
    def mean_range(self) -> float:
        """Geometric mean of ranges for reference."""
        return (self.range_major * self.range_semi * self.range_minor) ** (1/3)
    
    @property
    def anisotropy_ratio(self) -> Tuple[float, float]:
        """Anisotropy ratios (semi:major, minor:major)."""
        return (
            self.range_semi / self.range_major if self.range_major > 0 else 1.0,
            self.range_minor / self.range_major if self.range_major > 0 else 1.0,
        )


@dataclass
class ClassificationThresholds:
    """
    Classification thresholds for a single category (Measured/Indicated/Inferred).
    
    All distance values are expressed as fractions of variogram range (0.0 to 3.0+).
    In isotropic space, 1.0 = full variogram range.
    """
    # Primary criteria (always applied)
    max_iso_distance: float    # Maximum normalized distance (0.25, 0.50, 1.00 typical)
    min_unique_holes: int      # Minimum unique drillholes within threshold
    
    # Optional secondary criteria (set to None to disable)
    max_kv_ratio: Optional[float] = None    # Maximum KV/Sill ratio (e.g., 0.3 for Measured)
    min_slope: Optional[float] = None       # Minimum slope of regression (e.g., 0.9)
    max_pass: Optional[int] = None          # Maximum search pass number
    
    # Simulation-based criteria (optional)
    max_spread_ratio: Optional[float] = None  # Max (P90-P10)/mean spread
    
    def __post_init__(self):
        """Validate threshold parameters."""
        if self.max_iso_distance <= 0:
            raise ValueError("max_iso_distance must be positive")
        if self.min_unique_holes < 1:
            raise ValueError("min_unique_holes must be at least 1")
        if self.max_kv_ratio is not None and self.max_kv_ratio < 0:
            raise ValueError("max_kv_ratio must be non-negative")
        if self.min_slope is not None and not (0 <= self.min_slope <= 1):
            raise ValueError("min_slope must be between 0 and 1")


@dataclass
class ClassificationRuleset:
    """
    Complete ruleset for resource classification.
    
    Contains thresholds for all three classification levels plus metadata.
    """
    measured: ClassificationThresholds
    indicated: ClassificationThresholds
    inferred: ClassificationThresholds
    
    # Optional geology confidence downgrade
    # "not_specified" will be treated as "high" (most permissive) during classification
    geology_confidence: GeoConfLevel = "high"
    
    # Metadata
    domain_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate ruleset consistency."""
        # Ensure proper ordering: Measured < Indicated < Inferred distances
        if self.measured.max_iso_distance >= self.indicated.max_iso_distance:
            logger.warning(
                f"Measured distance ({self.measured.max_iso_distance}) >= "
                f"Indicated ({self.indicated.max_iso_distance}). "
                "This may cause unexpected classification behavior."
            )
        if self.indicated.max_iso_distance >= self.inferred.max_iso_distance:
            logger.warning(
                f"Indicated distance ({self.indicated.max_iso_distance}) >= "
                f"Inferred ({self.inferred.max_iso_distance}). "
                "This may cause unexpected classification behavior."
            )
    
    @classmethod
    def from_ui_params(
        cls,
        meas_dist_pct: float,
        meas_min_holes: int,
        ind_dist_pct: float,
        ind_min_holes: int,
        inf_dist_pct: float,
        inf_min_holes: int,
        meas_kv_enabled: bool = False,
        meas_kv_pct: float = 30.0,
        meas_slope_enabled: bool = False,
        meas_slope: float = 0.9,
        ind_kv_enabled: bool = False,
        ind_kv_pct: float = 60.0,
        ind_slope_enabled: bool = False,
        ind_slope: float = 0.8,
        geology_confidence: GeoConfLevel = "high",
        domain_name: Optional[str] = None,
    ) -> "ClassificationRuleset":
        """
        Create ruleset from UI slider values.
        
        Parameters
        ----------
        meas_dist_pct : float
            Measured max distance as percentage of range (e.g., 25 for 25%)
        meas_min_holes : int
            Minimum holes for Measured
        ind_dist_pct : float
            Indicated max distance as percentage of range
        ind_min_holes : int
            Minimum holes for Indicated
        inf_dist_pct : float
            Inferred max distance as percentage of range
        inf_min_holes : int
            Minimum holes for Inferred
        *_kv_enabled : bool
            Whether to apply KV criterion
        *_kv_pct : float
            Maximum KV as percentage of sill
        *_slope_enabled : bool
            Whether to apply slope criterion
        *_slope : float
            Minimum slope value (0-1)
        geology_confidence : str
            Geology confidence level
        domain_name : str, optional
            Domain identifier
        
        Returns
        -------
        ClassificationRuleset
            Configured ruleset
        """
        return cls(
            measured=ClassificationThresholds(
                max_iso_distance=meas_dist_pct / 100.0,
                min_unique_holes=meas_min_holes,
                max_kv_ratio=meas_kv_pct / 100.0 if meas_kv_enabled else None,
                min_slope=meas_slope if meas_slope_enabled else None,
            ),
            indicated=ClassificationThresholds(
                max_iso_distance=ind_dist_pct / 100.0,
                min_unique_holes=ind_min_holes,
                max_kv_ratio=ind_kv_pct / 100.0 if ind_kv_enabled else None,
                min_slope=ind_slope if ind_slope_enabled else None,
            ),
            inferred=ClassificationThresholds(
                max_iso_distance=inf_dist_pct / 100.0,
                min_unique_holes=inf_min_holes,
            ),
            geology_confidence=geology_confidence,
            domain_name=domain_name,
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ClassificationResult:
    """Result container for classification operation with JORC audit trail."""
    classified_df: pd.DataFrame
    summary: Dict[str, Any]
    ruleset: ClassificationRuleset
    variogram: VariogramModel
    domain_name: Optional[str]
    audit_records: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    # JORC Audit Trail: Deterministic hashes for reproducibility verification
    variogram_hash: str = ""
    ruleset_hash: str = ""
    
    def __post_init__(self):
        """Generate audit hashes if not already set."""
        if not self.variogram_hash and self.variogram:
            self.variogram_hash = _hash_variogram(self.variogram)
        if not self.ruleset_hash and self.ruleset:
            self.ruleset_hash = _hash_ruleset(self.ruleset)


# ------------------------------------------------------------------ #
# Geometry Utilities (Isotropic Transformation)
# ------------------------------------------------------------------ #

class IsotropicTransformer:
    """
    Transforms coordinates from real-world to isotropic space.
    
    In isotropic space:
    - The variogram ellipsoid becomes a unit sphere
    - Distance of 1.0 = full variogram range
    - All directional distances are normalized by respective ranges
    
    This follows industry-standard practice from Leapfrog, Datamine, Surpac.
    """
    
    def __init__(self, variogram: VariogramModel):
        """
        Initialize transformer with variogram model.
        
        Parameters
        ----------
        variogram : VariogramModel
            Variogram configuration with orientation and ranges
        """
        self.variogram = variogram
        self._rotation_matrix = self._compute_rotation_matrix(
            variogram.azimuth, variogram.dip, variogram.pitch
        )
        self._scale_factors = np.array([
            1.0 / max(variogram.range_major, 1e-6),
            1.0 / max(variogram.range_semi, 1e-6),
            1.0 / max(variogram.range_minor, 1e-6),
        ])
    
    @staticmethod
    def _compute_rotation_matrix(azimuth: float, dip: float, pitch: float) -> np.ndarray:
        """
        Compute 3D rotation matrix using Z-X-Y convention (mining standard).
        
        Parameters
        ----------
        azimuth : float
            Rotation around Z axis (degrees, clockwise from North)
        dip : float
            Rotation around X axis (degrees, positive down)
        pitch : float
            Rotation around Y axis (degrees, rake)
        
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        # Convert to radians
        a = math.radians(azimuth)
        d = math.radians(dip)
        p = math.radians(pitch)
        
        # Rotation around Z (azimuth)
        cos_a, sin_a = math.cos(a), math.sin(a)
        Rz = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Rotation around X (dip)
        cos_d, sin_d = math.cos(d), math.sin(d)
        Rx = np.array([
            [1, 0, 0],
            [0, cos_d, -sin_d],
            [0, sin_d, cos_d]
        ])
        
        # Rotation around Y (pitch)
        cos_p, sin_p = math.cos(p), math.sin(p)
        Ry = np.array([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p]
        ])
        
        # Combined: Ry @ Rx @ Rz
        return Ry @ Rx @ Rz
    
    def transform(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates to isotropic space.
        
        Parameters
        ----------
        coords : np.ndarray
            (N, 3) array of real-world coordinates [X, Y, Z]
        
        Returns
        -------
        np.ndarray
            (N, 3) array of isotropic coordinates where distance=1.0 is the range
        """
        if coords.shape[0] == 0:
            return coords
        
        # 1. Rotate to align with variogram principal axes
        rotated = coords @ self._rotation_matrix.T
        
        # 2. Scale by inverse ranges (normalize to unit sphere)
        return rotated * self._scale_factors
    
    def inverse_transform(self, iso_coords: np.ndarray) -> np.ndarray:
        """
        Transform from isotropic back to real-world coordinates.
        
        Parameters
        ----------
        iso_coords : np.ndarray
            (N, 3) array of isotropic coordinates
        
        Returns
        -------
        np.ndarray
            (N, 3) array of real-world coordinates
        """
        if iso_coords.shape[0] == 0:
            return iso_coords
        
        # Reverse scaling
        scaled = iso_coords / self._scale_factors
        
        # Reverse rotation
        return scaled @ self._rotation_matrix


# ------------------------------------------------------------------ #
# Core Classification Engine
# ------------------------------------------------------------------ #

class JORCClassificationEngine:
    """
    JORC/SAMREC-compliant Resource Classification Engine.
    
    This engine implements the full specification:
    - Optional domain selection (full model extent if no domain)
    - Automatic distance calculations in isotropic space
    - Percentage-based thresholds of variogram range
    - Efficient cKDTree spatial queries
    - Comprehensive audit trail
    - Numba-accelerated unique hole counting
    
    This is THE authoritative classification engine. All other classification
    modules should delegate to this engine for JORC/SAMREC compliance.
    
    Usage
    -----
    ```python
    engine = JORCClassificationEngine(
        variogram=VariogramModel(azimuth=45, dip=60, range_major=120, ...),
        ruleset=ClassificationRuleset.from_ui_params(
            meas_dist_pct=25, meas_min_holes=3,
            ind_dist_pct=50, ind_min_holes=2,
            inf_dist_pct=100, inf_min_holes=1,
        ),
    )
    result = engine.classify(blocks_df, drillholes_df)
    ```
    
    Version Information
    -------------------
    METHOD_VERSION: "JORC-v1.0"
    BUILD: "Numba-accelerated"
    """
    
    # Version tags for audit trail and traceability
    METHOD_VERSION = "JORC-v1.0"
    BUILD = "Numba-accelerated"
    
    def __init__(
        self,
        variogram: VariogramModel,
        ruleset: ClassificationRuleset,
        domain_column: Optional[str] = None,
        domain_value: Optional[str] = None,
    ):
        """
        Initialize the classification engine.
        
        Parameters
        ----------
        variogram : VariogramModel
            Variogram model defining the search ellipsoid
        ruleset : ClassificationRuleset
            Classification thresholds and criteria
        domain_column : str, optional
            Column name for domain filtering (e.g., 'DOMAIN', 'ZONE')
            If None, classifies full model extent
        domain_value : str, optional
            Domain value to filter on. Required if domain_column is set.
        """
        self.variogram = variogram
        self.ruleset = ruleset
        self.domain_column = domain_column
        self.domain_value = domain_value
        
        # Initialize transformer
        self.transformer = IsotropicTransformer(variogram)
        
        # Audit records
        self._audit_records: List[Dict[str, Any]] = []
        
        logger.info(
            f"JORCClassificationEngine initialized: "
            f"variogram=(range_major={variogram.range_major}m, sill={variogram.sill}), "
            f"domain={domain_value or 'FULL MODEL'}, "
            f"version={self.METHOD_VERSION}, build={self.BUILD}"
        )
    
    def classify(
        self,
        blocks_df: Union['BlockModel', pd.DataFrame],
        drillholes_df: pd.DataFrame,
        block_coord_cols: Optional[Tuple[str, str, str]] = None,
        hole_coord_cols: Tuple[str, str, str] = ("X", "Y", "Z"),
        hole_id_col: str = "HOLE_ID",
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> ClassificationResult:
        """
        Classify all blocks based on drillhole proximity in isotropic space.
        
        ✅ NEW STANDARD API: Accepts BlockModel or DataFrame (backward compatible)
        
        Parameters
        ----------
        blocks_df : BlockModel or pd.DataFrame
            Block model with coordinate columns
            - **Preferred:** BlockModel instance (uses standard get_engine_payload() API)
            - **Legacy:** pd.DataFrame (for backward compatibility)
        drillholes_df : pd.DataFrame
            Drillhole data with coordinate and hole ID columns
        block_coord_cols : tuple, optional
            Column names for block coordinates (X, Y, Z). Only used for DataFrame input.
            Ignored for BlockModel input (uses standard 'X', 'Y', 'Z' extraction)
        hole_coord_cols : tuple
            Column names for drillhole coordinates (X, Y, Z)
        hole_id_col : str
            Column name for drillhole identifier
        progress_callback : callable, optional
            Callback(percent, message) for progress updates
        
        Returns
        -------
        ClassificationResult
            Container with classified DataFrame and metadata
            
        Notes
        -----
        **Standard API (Recommended):**
        
            engine = JORCClassificationEngine(variogram, ruleset)
            result = engine.classify(block_model, drillholes_df)
        
        **Legacy API (Backward Compatible):**
        
            result = engine.classify(blocks_df, drillholes_df, block_coord_cols=('XC', 'YC', 'ZC'))
        """
        start_time = datetime.now()
        
        if progress_callback:
            progress_callback(0, "Initializing classification...")
        
        # --- NEW: Handle BlockModel input (standard API) ---
        from ..models.block_model import BlockModel
        
        if isinstance(blocks_df, BlockModel):
            # ✅ STANDARD API: Extract using BlockModel methods
            if progress_callback:
                progress_callback(2, "Extracting block coordinates from BlockModel...")
            
            try:
                # Use standard payload extraction
                payload = blocks_df.get_engine_payload(
                    domain_field=self.domain_column if self.domain_column else None
                )
                
                # Create DataFrame for classification (includes all properties for filtering)
                blocks = blocks_df.to_dataframe()
                
                # Extract coordinates directly from payload (validated, float64)
                block_coords_raw = payload['coords']  # N×3, float64
                
                # Normalize coordinate columns to XC, YC, ZC for consistency
                blocks['XC'] = block_coords_raw[:, 0]
                blocks['YC'] = block_coords_raw[:, 1]
                blocks['ZC'] = block_coords_raw[:, 2]
                
                logger.info(
                    f"✅ BlockModel API: Extracted {len(blocks)} blocks using standard payload"
                )
                
            except Exception as e:
                raise ValueError(
                    f"Failed to extract data from BlockModel: {e}\n"
                    "Ensure BlockModel has positions set and required properties."
                )
        
        else:
            # --- LEGACY: Handle DataFrame input (backward compatibility) ---
            if block_coord_cols is None:
                block_coord_cols = ("XC", "YC", "ZC")  # Default for DataFrame
            
            if progress_callback:
                progress_callback(2, "Using DataFrame input (legacy mode)...")
            
            blocks = blocks_df.copy()
            # Normalize coordinates using existing method
            blocks = self._normalize_block_coords(blocks, block_coord_cols)
            
            logger.info(
                f"⚠️ DataFrame API (legacy): Extracted {len(blocks)} blocks. "
                "Consider migrating to BlockModel for better performance."
            )
        
        # --- Continue with drillhole processing (same for both modes) ---
        holes = drillholes_df.copy()
        
        # Ensure coordinate columns exist - try fallbacks
        blocks = self._normalize_block_coords(blocks, block_coord_cols)
        holes = self._normalize_hole_coords(holes, hole_coord_cols, hole_id_col)
        
        # Apply domain filter if specified
        if self.domain_column and self.domain_value:
            if self.domain_column in blocks.columns:
                domain_mask = blocks[self.domain_column] == self.domain_value
                n_before = len(blocks)
                blocks = blocks[domain_mask].copy()
                logger.info(
                    f"Domain filter applied: {len(blocks)}/{n_before} blocks "
                    f"in domain '{self.domain_value}'"
                )
            else:
                logger.warning(
                    f"Domain column '{self.domain_column}' not found in blocks. "
                    "Classifying full model extent."
                )
            
            if self.domain_column in holes.columns:
                holes = holes[holes[self.domain_column] == self.domain_value].copy()
        else:
            logger.info("No domain filter - classifying full model extent")
        
        if len(blocks) == 0:
            logger.warning("No blocks to classify after domain filter")
            return self._empty_result(blocks_df)
        
        if len(holes) == 0:
            logger.warning("No drillholes available for classification")
            blocks["CLASS_AUTO"] = "Unclassified"
            blocks["CLASS_FINAL"] = "Unclassified"
            blocks["CLASS_REASON"] = "No drillholes in domain"
            return self._empty_result(blocks)

        import time
        t_start = time.perf_counter()

        if progress_callback:
            progress_callback(10, "Transforming to isotropic space...")
        
        # Extract coordinates
        xc, yc, zc = "XC", "YC", "ZC"  # Normalized names
        block_coords = blocks[[xc, yc, zc]].values
        hole_coords = holes[["X", "Y", "Z"]].values
        hole_ids_raw = holes[hole_id_col].values if hole_id_col in holes.columns else np.arange(len(holes))

        # CRITICAL OPTIMIZATION: Encode hole IDs to integers EARLY (1,905 samples)
        # instead of after neighbor lookup (65+ million elements).
        # np.unique on 1,905 strings is ~35,000x faster than on 65 million.
        if hole_ids_raw.dtype == object or hole_ids_raw.dtype.kind in ('U', 'S'):
            _, hole_ids = np.unique(hole_ids_raw, return_inverse=True)
            hole_ids = hole_ids.astype(np.int32)
            logger.debug(f"Encoded {len(hole_ids_raw)} string hole IDs to int32")
        else:
            hole_ids = hole_ids_raw.astype(np.int32)
        
        # Transform to isotropic space
        blocks_iso = self.transformer.transform(block_coords)
        holes_iso = self.transformer.transform(hole_coords)
        t_transform = time.perf_counter()
        logger.info(f"[TIMING] Transform to isotropic: {t_transform - t_start:.2f}s")

        if progress_callback:
            progress_callback(20, "Building spatial index...")

        # Build cKDTree for efficient queries
        tree = cKDTree(holes_iso)
        t_tree = time.perf_counter()
        logger.info(f"[TIMING] Build KDTree: {t_tree - t_transform:.2f}s")
        
        # Determine k (number of neighbors to query)
        # Need enough to find min_unique_holes unique drillholes
        # CRITICAL: Each drillhole may have many samples, so we need to query
        # enough neighbors to find multiple unique holes
        max_holes_needed = max(
            self.ruleset.measured.min_unique_holes,
            self.ruleset.indicated.min_unique_holes,
            self.ruleset.inferred.min_unique_holes,
        )
        
        # Calculate average samples per unique drillhole
        n_unique_holes = len(np.unique(hole_ids))
        samples_per_hole = len(holes_iso) / max(n_unique_holes, 1)

        # OPTIMIZED: Query fewer neighbors since we have early termination
        # We need enough to find 3 unique holes + buffer for threshold counting
        # With early exit, we stop once we find 3 unique holes AND exceed max threshold
        # Safety factor of 2x samples_per_hole * 3 holes = 6x samples typically needed
        k_neighbors_estimated = int(max_holes_needed * samples_per_hole * 2) + 20
        k_neighbors_max = min(150, len(holes_iso))  # Reduced from 500 - early exit handles the rest
        k_neighbors = min(k_neighbors_estimated, k_neighbors_max, len(holes_iso))
        
        if k_neighbors_estimated > k_neighbors_max:
            logger.warning(
                f"k_neighbors estimated ({k_neighbors_estimated}) exceeds max ({k_neighbors_max}), "
                f"capping at {k_neighbors} for performance. "
                f"This may affect classification accuracy for blocks with many samples per hole."
            )
        
        logger.info(
            f"Classification query: {n_unique_holes} unique holes, "
            f"~{samples_per_hole:.1f} samples/hole, querying k={k_neighbors} neighbors "
            f"(estimated: {k_neighbors_estimated}, max: {k_neighbors_max})"
        )
        
        if progress_callback:
            progress_callback(30, "Computing spatial relationships...")

        # Query nearest neighbors using parallel processing for speed
        # workers=-1 uses all available CPU cores (significant speedup for large models)
        if k_neighbors > 0:
            dists_iso, indices = tree.query(blocks_iso, k=k_neighbors, workers=-1)

            # Handle single neighbor case
            if dists_iso.ndim == 1:
                dists_iso = dists_iso.reshape(-1, 1)
                indices = indices.reshape(-1, 1)
        else:
            dists_iso = np.full((len(blocks), 1), np.inf)
            indices = np.zeros((len(blocks), 1), dtype=int)

        t_query = time.perf_counter()
        logger.info(f"[TIMING] KDTree query ({len(blocks):,} blocks x {k_neighbors} neighbors): {t_query - t_tree:.2f}s")

        if progress_callback:
            progress_callback(50, "Counting unique drillholes...")
        
        # Calculate distances to nth unique hole
        n_blocks = len(blocks)
        neighbor_hole_ids = hole_ids[indices]
        
        # Thresholds
        meas_thresh = self.ruleset.measured.max_iso_distance
        ind_thresh = self.ruleset.indicated.max_iso_distance
        inf_thresh = self.ruleset.inferred.max_iso_distance
        
        # Use optimized counting function
        (dist_to_1st, dist_to_2nd, dist_to_3rd,
         n_holes_within_meas, n_holes_within_ind, n_holes_within_inf) = \
            _count_unique_holes_fast(
                neighbor_hole_ids, dists_iso,
                meas_thresh, ind_thresh, inf_thresh,
                progress_callback
            )

        t_count = time.perf_counter()
        logger.info(f"[TIMING] Count unique holes: {t_count - t_query:.2f}s")

        if progress_callback:
            progress_callback(70, "Applying classification rules...")
        
        # Apply classification logic
        classifications = np.full(n_blocks, "Unclassified", dtype=object)
        reasons = np.full(n_blocks, "Outside all thresholds", dtype=object)
        
        # Get optional criteria from blocks if available
        kv_values = blocks["KV"].values if "KV" in blocks.columns else np.full(n_blocks, np.nan)
        slope_values = blocks["SLOPE"].values if "SLOPE" in blocks.columns else np.full(n_blocks, np.nan)
        
        # Normalize KV by sill for comparison
        kv_ratio = kv_values / self.variogram.sill if self.variogram.sill > 0 else kv_values
        
        # Apply rules in order of priority (Measured first)
        
        # --- INFERRED ---
        inf_rule = self.ruleset.inferred
        mask_inf_dist = dist_to_1st <= inf_rule.max_iso_distance
        mask_inf_holes = n_holes_within_inf >= inf_rule.min_unique_holes
        mask_inferred = mask_inf_dist & mask_inf_holes
        
        classifications[mask_inferred] = "Inferred"
        reasons[mask_inferred] = f"Dist≤{inf_rule.max_iso_distance:.0%}R, Holes≥{inf_rule.min_unique_holes}"
        
        # --- INDICATED ---
        ind_rule = self.ruleset.indicated
        mask_ind_dist = dist_to_2nd <= ind_rule.max_iso_distance
        mask_ind_holes = n_holes_within_ind >= ind_rule.min_unique_holes
        mask_indicated = mask_ind_dist & mask_ind_holes
        
        # Apply optional KV criterion
        if ind_rule.max_kv_ratio is not None:
            mask_ind_kv = (np.isnan(kv_ratio)) | (kv_ratio <= ind_rule.max_kv_ratio)
            mask_indicated &= mask_ind_kv
        
        # Apply optional slope criterion
        if ind_rule.min_slope is not None:
            mask_ind_slope = (np.isnan(slope_values)) | (slope_values >= ind_rule.min_slope)
            mask_indicated &= mask_ind_slope
        
        # Geology confidence check
        geo_conf = self.ruleset.geology_confidence
        if geo_conf == "not_specified":
            geo_conf = "high"  # Default to high if not specified
        
        if geo_conf not in ("high", "medium"):
            mask_indicated[:] = False  # Can't be Indicated with low geology confidence
        
        classifications[mask_indicated] = "Indicated"
        reasons[mask_indicated] = f"Dist≤{ind_rule.max_iso_distance:.0%}R, Holes≥{ind_rule.min_unique_holes}"
        
        # --- MEASURED ---
        meas_rule = self.ruleset.measured
        mask_meas_dist = dist_to_3rd <= meas_rule.max_iso_distance
        mask_meas_holes = n_holes_within_meas >= meas_rule.min_unique_holes
        mask_measured = mask_meas_dist & mask_meas_holes
        
        # Apply optional KV criterion
        if meas_rule.max_kv_ratio is not None:
            mask_meas_kv = (np.isnan(kv_ratio)) | (kv_ratio <= meas_rule.max_kv_ratio)
            mask_measured &= mask_meas_kv
        
        # Apply optional slope criterion
        if meas_rule.min_slope is not None:
            mask_meas_slope = (np.isnan(slope_values)) | (slope_values >= meas_rule.min_slope)
            mask_measured &= mask_meas_slope
        
        # Geology confidence check
        geo_conf = self.ruleset.geology_confidence
        if geo_conf == "not_specified":
            geo_conf = "high"  # Default to high if not specified
        
        if geo_conf != "high":
            mask_measured[:] = False  # Can't be Measured without high geology confidence
        
        classifications[mask_measured] = "Measured"
        reasons[mask_measured] = f"Dist≤{meas_rule.max_iso_distance:.0%}R, Holes≥{meas_rule.min_unique_holes}"
        
        if progress_callback:
            progress_callback(90, "Writing results...")
        
        # Write back results
        blocks["CLASS_AUTO"] = classifications
        blocks["CLASS_FINAL"] = classifications  # CP overrides applied later
        blocks["CLASS_REASON"] = reasons
        
        # Diagnostic columns
        blocks["DIST_ISO_1ST"] = dist_to_1st
        blocks["DIST_ISO_2ND"] = dist_to_2nd
        blocks["DIST_ISO_3RD"] = dist_to_3rd
        blocks["N_HOLES_MEAS"] = n_holes_within_meas
        blocks["N_HOLES_IND"] = n_holes_within_ind
        blocks["N_HOLES_INF"] = n_holes_within_inf
        
        # Convert isotropic distances back to approximate real-world distances
        blocks["DIST_REAL_1ST"] = dist_to_1st * self.variogram.range_major
        
        # Generate summary
        summary = self._generate_summary(blocks, classifications)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if progress_callback:
            progress_callback(100, "Classification complete")
        
        t_end = time.perf_counter()
        logger.info(f"[TIMING] Apply rules + summary: {t_end - t_count:.2f}s")
        logger.info(f"[TIMING] TOTAL classification: {t_end - t_start:.2f}s")

        logger.info(
            f"Classification complete: "
            f"Measured={summary['Measured']['count']}, "
            f"Indicated={summary['Indicated']['count']}, "
            f"Inferred={summary['Inferred']['count']}, "
            f"Unclassified={summary['Unclassified']['count']} "
            f"({execution_time:.2f}s)"
        )
        
        # Generate deterministic hashes for JORC audit trail
        variogram_hash = _hash_variogram(self.variogram)
        ruleset_hash = _hash_ruleset(self.ruleset)
        
        # Add version info to audit records for traceability
        version_audit = {
            "timestamp": datetime.now().isoformat(),
            "event": "classification_complete",
            "method_version": self.METHOD_VERSION,
            "build": self.BUILD,
            "engine": "JORCClassificationEngine",
            "variogram_hash": variogram_hash,
            "ruleset_hash": ruleset_hash,
            "n_blocks_classified": len(blocks),
            "domain": self.domain_value or "FULL_MODEL",
        }
        self._audit_records.append(version_audit)
        
        logger.info(
            f"JORC Audit Trail: variogram_hash={variogram_hash}, "
            f"ruleset_hash={ruleset_hash}"
        )
        
        return ClassificationResult(
            classified_df=blocks,
            summary=summary,
            ruleset=self.ruleset,
            variogram=self.variogram,
            domain_name=self.domain_value,
            audit_records=self._audit_records,
            execution_time_seconds=execution_time,
            variogram_hash=variogram_hash,
            ruleset_hash=ruleset_hash,
        )
    
    def _normalize_block_coords(
        self, 
        df: pd.DataFrame, 
        preferred_cols: Tuple[str, str, str]
    ) -> pd.DataFrame:
        """Normalize block coordinate columns to XC, YC, ZC."""
        xc, yc, zc = preferred_cols
        
        # Try preferred columns first
        if all(c in df.columns for c in [xc, yc, zc]):
            if xc != "XC":
                df["XC"] = df[xc]
            if yc != "YC":
                df["YC"] = df[yc]
            if zc != "ZC":
                df["ZC"] = df[zc]
            return df
        
        # Try common alternatives
        alternatives = [
            ("XC", "YC", "ZC"),
            ("x", "y", "z"),
            ("X", "Y", "Z"),
            ("xc", "yc", "zc"),
            ("XCENTER", "YCENTER", "ZCENTER"),
            ("X_CENTER", "Y_CENTER", "Z_CENTER"),
        ]
        
        for x, y, z in alternatives:
            if all(c in df.columns for c in [x, y, z]):
                df["XC"] = df[x]
                df["YC"] = df[y]
                df["ZC"] = df[z]
                return df
        
        raise ValueError(
            f"Could not find block coordinate columns. "
            f"Available: {list(df.columns)}"
        )
    
    def _normalize_hole_coords(
        self,
        df: pd.DataFrame,
        preferred_cols: Tuple[str, str, str],
        hole_id_col: str,
    ) -> pd.DataFrame:
        """Normalize drillhole coordinate columns to X, Y, Z."""
        x, y, z = preferred_cols
        
        # Try preferred columns first
        if all(c in df.columns for c in [x, y, z]):
            if x != "X":
                df["X"] = df[x]
            if y != "Y":
                df["Y"] = df[y]
            if z != "Z":
                df["Z"] = df[z]
        else:
            # Try alternatives
            alternatives = [
                ("X", "Y", "Z"),
                ("x", "y", "z"),
                ("EAST", "NORTH", "RL"),
                ("EASTING", "NORTHING", "ELEVATION"),
            ]
            
            found = False
            for ax, ay, az in alternatives:
                if all(c in df.columns for c in [ax, ay, az]):
                    df["X"] = df[ax]
                    df["Y"] = df[ay]
                    df["Z"] = df[az]
                    found = True
                    break
            
            if not found:
                raise ValueError(
                    f"Could not find drillhole coordinate columns. "
                    f"Available: {list(df.columns)}"
                )
        
        # Ensure HOLE_ID exists
        if hole_id_col not in df.columns:
            # Try alternatives
            for alt in ["HOLEID", "HoleID", "hole_id", "BHID", "DH_ID"]:
                if alt in df.columns:
                    df["HOLE_ID"] = df[alt]
                    break
            else:
                # Generate synthetic IDs
                df["HOLE_ID"] = df.index.astype(str)
                logger.warning("No hole ID column found, using index as HOLE_ID")
        elif hole_id_col != "HOLE_ID":
            df["HOLE_ID"] = df[hole_id_col]
        
        return df
    
    def _generate_summary(
        self, 
        df: pd.DataFrame, 
        classifications: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Generate classification summary statistics."""
        summary = {}
        total = len(df)
        
        for category in CLASSIFICATION_ORDER:
            mask = classifications == category
            count = np.sum(mask)
            pct = 100 * count / total if total > 0 else 0
            
            cat_data = {
                "count": int(count),
                "percentage": round(pct, 2),
                "color": CLASSIFICATION_COLORS[category],
            }
            
            if count > 0 and "DIST_REAL_1ST" in df.columns:
                cat_df = df[mask]
                cat_data["avg_distance_m"] = round(cat_df["DIST_REAL_1ST"].mean(), 1)
                cat_data["min_distance_m"] = round(cat_df["DIST_REAL_1ST"].min(), 1)
                cat_data["max_distance_m"] = round(cat_df["DIST_REAL_1ST"].max(), 1)
            
            summary[category] = cat_data
        
        summary["total_blocks"] = total
        summary["variogram_range_major"] = self.variogram.range_major
        summary["domain"] = self.domain_value or "Full Model"
        
        return summary
    
    def compute_distance_diagnostics(
        self,
        blocks_df: Union['BlockModel', pd.DataFrame],
        drillholes_df: pd.DataFrame,
        block_coord_cols: Optional[Tuple[str, str, str]] = None,
        hole_coord_cols: Tuple[str, str, str] = ("X", "Y", "Z"),
        hole_id_col: str = "HOLE_ID",
        progress_callback: Optional[Callable[[int, str], None]] = None,
        sample_size: int = 10_000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute distance diagnostics WITHOUT applying classification.

        This is used by the auto-suggest feature to analyze distance distributions
        before running the actual classification.

        Uses random sampling for large block models to dramatically improve performance.
        A sample of 10,000 blocks provides statistically valid percentile estimates
        with ~1% error, which is sufficient for threshold suggestion.

        Parameters
        ----------
        blocks_df : BlockModel or pd.DataFrame
            Block model data
        drillholes_df : pd.DataFrame
            Drillhole data
        block_coord_cols, hole_coord_cols, hole_id_col : optional
            Column name specifications
        progress_callback : optional
            Progress callback (percent, message)
        sample_size : int, default 10_000
            Maximum number of blocks to sample for distance computation.
            Set to 0 or negative to disable sampling (process all blocks).

        Returns
        -------
        tuple of (dist_to_1st, dist_to_2nd, dist_to_3rd)
            Arrays of isotropic distances to 1st, 2nd, 3rd unique drillholes
            for sampled blocks. These are in normalized units where 1.0 = variogram range.
        """
        if progress_callback:
            progress_callback(0, "Computing distance diagnostics...")
        
        # --- Handle BlockModel vs DataFrame input ---
        from ..models.block_model import BlockModel
        
        if isinstance(blocks_df, BlockModel):
            blocks = blocks_df.to_dataframe()
            if block_coord_cols is None:
                block_coord_cols = ("XC", "YC", "ZC")
        else:
            blocks = blocks_df.copy()
            if block_coord_cols is None:
                block_coord_cols = ("XC", "YC", "ZC")
        
        holes = drillholes_df.copy()
        
        # Normalize coordinates
        blocks = self._normalize_block_coords(blocks, block_coord_cols)
        holes = self._normalize_hole_coords(holes, hole_coord_cols, hole_id_col)
        
        if len(blocks) == 0 or len(holes) == 0:
            n = len(blocks) if len(blocks) > 0 else 1
            return (
                np.full(n, np.inf),
                np.full(n, np.inf),
                np.full(n, np.inf)
            )
        
        if progress_callback:
            progress_callback(10, "Extracting coordinates...")

        # Extract coordinates
        block_coords = blocks[["XC", "YC", "ZC"]].values
        hole_coords = holes[["X", "Y", "Z"]].values
        hole_ids_raw = holes["HOLE_ID"].values if "HOLE_ID" in holes.columns else np.arange(len(holes))

        # CRITICAL OPTIMIZATION: Encode hole IDs to integers EARLY
        if hole_ids_raw.dtype == object or hole_ids_raw.dtype.kind in ('U', 'S'):
            _, hole_ids = np.unique(hole_ids_raw, return_inverse=True)
            hole_ids = hole_ids.astype(np.int32)
        else:
            hole_ids = hole_ids_raw.astype(np.int32)

        n_blocks_total = len(block_coords)

        # Apply sampling for large block models (dramatic performance improvement)
        # A sample of 10,000 blocks provides statistically valid percentile estimates
        # with ~1% error - sufficient for threshold suggestion
        use_sampling = sample_size > 0 and n_blocks_total > sample_size
        if use_sampling:
            np.random.seed(42)  # Reproducible sampling
            sample_indices = np.random.choice(n_blocks_total, sample_size, replace=False)
            block_coords_sample = block_coords[sample_indices]
            logger.info(
                f"Distance diagnostics: Sampling {sample_size:,} of {n_blocks_total:,} blocks "
                f"({100*sample_size/n_blocks_total:.1f}%) for faster computation"
            )
            if progress_callback:
                progress_callback(15, f"Sampling {sample_size:,} of {n_blocks_total:,} blocks...")
        else:
            block_coords_sample = block_coords
            logger.info(f"Distance diagnostics: Processing all {n_blocks_total:,} blocks")

        if progress_callback:
            progress_callback(20, "Transforming to isotropic space...")

        # Transform to isotropic space (only sampled blocks)
        blocks_iso = self.transformer.transform(block_coords_sample)
        holes_iso = self.transformer.transform(hole_coords)

        if progress_callback:
            progress_callback(40, "Building spatial index...")
        
        # Build cKDTree
        tree = cKDTree(holes_iso)
        
        # Determine k (need enough neighbors to find 3 unique holes)
        # OPTIMIZED: Reduced from 500 since _count_unique_holes_fast has early exit
        n_unique_holes = len(np.unique(hole_ids))
        samples_per_hole = len(holes_iso) / max(n_unique_holes, 1)
        k_neighbors_estimated = int(3 * samples_per_hole * 2) + 20
        k_neighbors_max = min(100, len(holes_iso))  # Reduced - early exit handles the rest
        k_neighbors = min(k_neighbors_estimated, k_neighbors_max, len(holes_iso))
        
        if progress_callback:
            progress_callback(60, "Computing spatial relationships...")

        # Query nearest neighbors using parallel processing
        dists_iso, indices = tree.query(blocks_iso, k=k_neighbors, workers=-1)

        if dists_iso.ndim == 1:
            dists_iso = dists_iso.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        
        if progress_callback:
            progress_callback(80, "Counting unique drillholes...")
        
        # Count unique holes (use a very large threshold to get all distances)
        neighbor_hole_ids = hole_ids[indices]
        
        # Use the fast counting function with very large thresholds
        # (we just want distances, not counts based on thresholds)
        (dist_to_1st, dist_to_2nd, dist_to_3rd, _, _, _) = _count_unique_holes_fast(
            neighbor_hole_ids, dists_iso,
            meas_thresh=999.0, ind_thresh=999.0, inf_thresh=999.0,
            progress_callback=None
        )
        
        if progress_callback:
            progress_callback(100, "Distance diagnostics complete")

        n_computed = len(dist_to_1st)
        sample_note = f" (sampled from {n_blocks_total:,})" if use_sampling else ""
        logger.info(
            f"Distance diagnostics: {n_computed:,} blocks{sample_note}, "
            f"dist_1st median={np.nanmedian(dist_to_1st):.3f}, "
            f"dist_2nd median={np.nanmedian(dist_to_2nd):.3f}, "
            f"dist_3rd median={np.nanmedian(dist_to_3rd):.3f}"
        )

        return dist_to_1st, dist_to_2nd, dist_to_3rd
    
    def _empty_result(self, df: pd.DataFrame) -> ClassificationResult:
        """Return empty result for edge cases."""
        return ClassificationResult(
            classified_df=df,
            summary={cat: {"count": 0, "percentage": 0} for cat in CLASSIFICATION_ORDER},
            ruleset=self.ruleset,
            variogram=self.variogram,
            domain_name=self.domain_value,
            audit_records=[],
            execution_time_seconds=0,
        )
    
    def export_audit_excel(self, filepath: str, result: ClassificationResult) -> None:
        """
        Export comprehensive audit report to Excel.
        
        Parameters
        ----------
        filepath : str
            Output file path (.xlsx)
        result : ClassificationResult
            Classification result to export
        """
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Sheet 1: Classified blocks
                result.classified_df.to_excel(
                    writer, sheet_name="Classified_Blocks", index=False
                )
                
                # Sheet 2: Summary
                summary_df = pd.DataFrame([
                    {
                        "Category": cat,
                        "Count": data.get("count", 0),
                        "Percentage": data.get("percentage", 0),
                        "Avg_Distance_m": data.get("avg_distance_m", "N/A"),
                    }
                    for cat, data in result.summary.items()
                    if cat in CLASSIFICATION_ORDER
                ])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Sheet 3: Parameters
                params_data = {
                    "Parameter": [
                        "Variogram Range Major (m)",
                        "Variogram Range Semi (m)",
                        "Variogram Range Minor (m)",
                        "Variogram Azimuth (°)",
                        "Variogram Dip (°)",
                        "Sill",
                        "Measured Max Distance (% Range)",
                        "Measured Min Holes",
                        "Indicated Max Distance (% Range)",
                        "Indicated Min Holes",
                        "Inferred Max Distance (% Range)",
                        "Inferred Min Holes",
                        "Domain",
                        "Geology Confidence",
                    ],
                    "Value": [
                        self.variogram.range_major,
                        self.variogram.range_semi,
                        self.variogram.range_minor,
                        self.variogram.azimuth,
                        self.variogram.dip,
                        self.variogram.sill,
                        f"{self.ruleset.measured.max_iso_distance * 100:.0f}%",
                        self.ruleset.measured.min_unique_holes,
                        f"{self.ruleset.indicated.max_iso_distance * 100:.0f}%",
                        self.ruleset.indicated.min_unique_holes,
                        f"{self.ruleset.inferred.max_iso_distance * 100:.0f}%",
                        self.ruleset.inferred.min_unique_holes,
                        result.domain_name or "Full Model",
                        self.ruleset.geology_confidence,
                    ],
                }
                pd.DataFrame(params_data).to_excel(
                    writer, sheet_name="Parameters", index=False
                )
            
            logger.info(f"Audit report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export audit report: {e}")
            raise


# ------------------------------------------------------------------ #
# Factory Functions
# ------------------------------------------------------------------ #

def create_default_ruleset(
    for_commodity: str = "gold",
    variogram_range: float = 100.0,
) -> ClassificationRuleset:
    """
    Create a sensible default ruleset for common commodities.
    
    Parameters
    ----------
    for_commodity : str
        Commodity type ('gold', 'copper', 'iron', 'coal')
    variogram_range : float
        Reference variogram range for guidance
    
    Returns
    -------
    ClassificationRuleset
        Pre-configured ruleset
    """
    # Industry-standard defaults
    if for_commodity.lower() in ("gold", "au"):
        return ClassificationRuleset.from_ui_params(
            meas_dist_pct=25, meas_min_holes=3,
            ind_dist_pct=50, ind_min_holes=2,
            inf_dist_pct=100, inf_min_holes=1,
            meas_kv_enabled=True, meas_kv_pct=30,
            ind_kv_enabled=True, ind_kv_pct=60,
        )
    elif for_commodity.lower() in ("copper", "cu"):
        return ClassificationRuleset.from_ui_params(
            meas_dist_pct=30, meas_min_holes=3,
            ind_dist_pct=60, ind_min_holes=2,
            inf_dist_pct=120, inf_min_holes=1,
        )
    elif for_commodity.lower() in ("iron", "fe"):
        return ClassificationRuleset.from_ui_params(
            meas_dist_pct=35, meas_min_holes=4,
            ind_dist_pct=70, ind_min_holes=2,
            inf_dist_pct=150, inf_min_holes=1,
        )
    else:
        # General default
        return ClassificationRuleset.from_ui_params(
            meas_dist_pct=25, meas_min_holes=3,
            ind_dist_pct=50, ind_min_holes=2,
            inf_dist_pct=100, inf_min_holes=1,
        )


def suggest_thresholds_from_distances(
    dist_to_1st: np.ndarray,
    dist_to_2nd: np.ndarray,
    dist_to_3rd: np.ndarray,
    target_measured: float = 0.10,
    target_indicated: float = 0.35,
    target_inferred: float = 0.80,
) -> Dict[str, Dict[str, Any]]:
    """
    Suggest classification thresholds based on distance distributions.
    
    This implements data-driven auto-suggestion for CP review.
    
    Target coverages (configurable):
    - Measured: ~10% of blocks with 3rd unique hole inside
    - Indicated: ~35% of blocks with 2nd unique hole inside
    - Inferred: ~80% of blocks with 1st unique hole inside
    
    Parameters
    ----------
    dist_to_1st, dist_to_2nd, dist_to_3rd : np.ndarray
        Isotropic distances to 1st/2nd/3rd unique drillholes (normalized, 1.0 = range)
    target_measured : float
        Target fraction of blocks for Measured (default 0.10 = 10%)
    target_indicated : float
        Target fraction of blocks for Indicated (default 0.35 = 35%)
    target_inferred : float
        Target fraction of blocks for Inferred (default 0.80 = 80%)
    
    Returns
    -------
    dict
        Suggested thresholds for each category:
        {
            "measured": {"dist_pct": 25, "min_holes": 3},
            "indicated": {"dist_pct": 60, "min_holes": 2},
            "inferred": {"dist_pct": 100, "min_holes": 1},
            "diagnostics": {...}
        }
    """
    # === Validate input arrays ===
    if dist_to_1st is None or len(dist_to_1st) == 0:
        logger.error("Auto-suggest FAILED: dist_to_1st is None or empty")
        return {
            "measured": {"dist_pct": 25, "min_holes": 3},
            "indicated": {"dist_pct": 60, "min_holes": 2},
            "inferred": {"dist_pct": 100, "min_holes": 1},
            "diagnostics": {"error": "Empty distance arrays - check coordinate systems"}
        }

    # Get finite values only
    finite1 = dist_to_1st[np.isfinite(dist_to_1st)]
    finite2 = dist_to_2nd[np.isfinite(dist_to_2nd)]
    finite3 = dist_to_3rd[np.isfinite(dist_to_3rd)]

    # === Validate finite arrays have reasonable values ===
    if len(finite1) > 0:
        median1 = np.median(finite1)
        if median1 > 100.0:  # More than 100× variogram range!
            logger.error(
                f"Auto-suggest FAILED: Distances are HUGE (median={median1:.1f} × range). "
                f"This indicates coordinate system mismatch between blocks and drillholes. "
                f"Check that both use the same coordinate system (UTM or Local)."
            )
            return {
                "measured": {"dist_pct": 25, "min_holes": 3},
                "indicated": {"dist_pct": 60, "min_holes": 2},
                "inferred": {"dist_pct": 100, "min_holes": 1},
                "diagnostics": {
                    "error": "COORDINATE MISMATCH DETECTED",
                    "median_distance": float(median1),
                    "hint": "Blocks and drillholes may be in different coordinate systems"
                }
            }

    # Handle edge cases
    if len(finite1) == 0:
        logger.warning("No finite dist_to_1st values, using defaults")
        return {
            "measured": {"dist_pct": 25, "min_holes": 3},
            "indicated": {"dist_pct": 60, "min_holes": 2},
            "inferred": {"dist_pct": 100, "min_holes": 1},
            "diagnostics": {"error": "No finite distance values - no drillholes found"}
        }

    # === Warn if too few samples for reliable statistics ===
    if len(finite1) < 100:
        logger.warning(
            f"Auto-suggest: Only {len(finite1)} valid distance samples. "
            f"Results may be unreliable. Recommend >1000 samples for stable percentiles."
        )

    # Calculate percentile distances in isotropic units (R = 1.0)
    # For Measured: use dist_to_3rd (need 3 holes)
    # For Indicated: use dist_to_2nd (need 2 holes)
    # For Inferred: use dist_to_1st (need 1 hole)
    
    if len(finite3) > 0:
        d_meas_iso = np.quantile(finite3, target_measured)
    else:
        d_meas_iso = 0.25  # Fallback
    
    if len(finite2) > 0:
        d_ind_iso = np.quantile(finite2, target_indicated)
    else:
        d_ind_iso = 0.60  # Fallback
    
    d_inf_iso = np.quantile(finite1, target_inferred)
    
    # Clamp to reasonable bands (industry practice)
    d_meas_iso = np.clip(d_meas_iso, 0.10, 0.80)
    d_ind_iso = np.clip(d_ind_iso, 0.20, 1.00)
    d_inf_iso = np.clip(d_inf_iso, 0.50, 1.50)
    
    # Ensure proper ordering (Measured < Indicated < Inferred)
    if d_ind_iso <= d_meas_iso:
        d_ind_iso = d_meas_iso + 0.15
    if d_inf_iso <= d_ind_iso:
        d_inf_iso = d_ind_iso + 0.20
    
    # Convert to % of range for UI
    meas_pct = int(round(d_meas_iso * 100))
    ind_pct = int(round(d_ind_iso * 100))
    inf_pct = int(round(d_inf_iso * 100))
    
    # Calculate actual coverage at these thresholds (for diagnostics)
    actual_meas_cov = np.sum(finite3 <= d_meas_iso) / len(finite3) if len(finite3) > 0 else 0
    actual_ind_cov = np.sum(finite2 <= d_ind_iso) / len(finite2) if len(finite2) > 0 else 0
    actual_inf_cov = np.sum(finite1 <= d_inf_iso) / len(finite1) if len(finite1) > 0 else 0
    
    logger.info(
        f"Auto-suggested thresholds: "
        f"Measured={meas_pct}% (coverage={actual_meas_cov:.1%}), "
        f"Indicated={ind_pct}% (coverage={actual_ind_cov:.1%}), "
        f"Inferred={inf_pct}% (coverage={actual_inf_cov:.1%})"
    )
    
    return {
        "measured": {"dist_pct": meas_pct, "min_holes": 3},
        "indicated": {"dist_pct": ind_pct, "min_holes": 2},
        "inferred": {"dist_pct": inf_pct, "min_holes": 1},
        "diagnostics": {
            "dist_1st_median": float(np.median(finite1)),
            "dist_2nd_median": float(np.median(finite2)) if len(finite2) > 0 else None,
            "dist_3rd_median": float(np.median(finite3)) if len(finite3) > 0 else None,
            "actual_measured_coverage": float(actual_meas_cov),
            "actual_indicated_coverage": float(actual_ind_cov),
            "actual_inferred_coverage": float(actual_inf_cov),
            "target_measured": target_measured,
            "target_indicated": target_indicated,
            "target_inferred": target_inferred,
        }
    }

