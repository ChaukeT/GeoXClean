"""
3D Variogram Modeling Module

This module provides comprehensive 3D variogram analysis for drillhole data,
including experimental variogram calculation, directional analysis, model fitting,
and visualization.

Professional features:
- Auto num_lags with direction-specific overrides (horizontal vs downhole)
- Nested structure support (short + long range models) 
- Global nugget enforcement across all directions
- Extended downhole variogram to capture long-range vertical structure
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Dict, List, Any
import logging

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - fallback if SciPy not installed
    cKDTree = None

from .variogram_functions import (
    calculate_pair_attributes,
    fit_variogram_model,
    MODEL_MAP,
    _sorted_pairs_array,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURABLE THRESHOLDS (loaded from config, with defaults)
# ============================================================
def _get_variogram_config() -> Dict[str, Any]:
    """
    Load variogram configuration with sensible defaults.

    Advanced users can override these via ~/.geox/config.toml
    """
    defaults = {
        'random_state': 42,
        'max_directional_samples': 1500,
        'pair_cap': 200000,
        'min_pairs_per_lag': 30,
        'weak_threshold': 50,
        'critical_lags_with_pairs': 3,
        'inherit_omni_sill_for_weak': True,
        'sill_cap_multiplier': 1.3,
        'default_n_lags': 12,
        'default_lag_tolerance_fraction': 0.3,
        'default_cone_tolerance': 15.0,
    }

    try:
        from ..config import Config
        config = Config()
        vg_config = config.get('variogram', {})
        # Merge with defaults
        for key in defaults:
            if key in vg_config:
                defaults[key] = vg_config[key]
    except Exception:
        pass  # Use defaults if config not available

    return defaults


# Module-level config (loaded once)
VARIOGRAM_CONFIG = _get_variogram_config()

ModelType = Literal["spherical", "exponential", "gaussian"]


@dataclass
class NestedStructure:
    """Represents a single structure in a nested variogram model."""
    model_type: str
    contribution: float  # Partial sill (C1, C2, etc.)
    range_major: float
    range_minor: float = None
    range_vertical: float = None
    
    def __post_init__(self):
        if self.range_minor is None:
            self.range_minor = self.range_major
        if self.range_vertical is None:
            self.range_vertical = self.range_major


@dataclass
class NestedVariogramModel:
    """
    Professional nested variogram model supporting multiple structures.
    
    Standard geostatistical model form:
    γ(h) = C0 + C1·g1(h/a1) + C2·g2(h/a2) + ...
    
    Where C0 = nugget, C1/C2 = partial sills, a1/a2 = ranges, g = model function.
    """
    nugget: float
    structures: List[NestedStructure] = field(default_factory=list)
    
    @property
    def total_sill(self) -> float:
        """Total sill (C0 + C1 + C2 + ...)"""
        return self.nugget + sum(s.contribution for s in self.structures)
    
    @property
    def n_structures(self) -> int:
        return len(self.structures)
    
    def evaluate(self, h: np.ndarray, direction: str = 'major') -> np.ndarray:
        """
        Evaluate nested model at distances h.
        
        Parameters
        ----------
        h : np.ndarray
            Distances to evaluate
        direction : str
            Direction for anisotropic ranges ('major', 'minor', 'vertical')
        """
        h = np.asarray(h, dtype=float)
        result = np.full_like(h, self.nugget)
        
        for struct in self.structures:
            # Get range for this direction
            if direction == 'major':
                rng = struct.range_major
            elif direction == 'minor':
                rng = struct.range_minor
            elif direction == 'vertical':
                rng = struct.range_vertical
            else:
                rng = struct.range_major
            
            # Evaluate model function
            model_func = MODEL_MAP.get(struct.model_type, MODEL_MAP['spherical'])
            # MODEL_MAP functions: model(h, range, sill, nugget)
            # For nested, we want just the contribution without additional nugget
            contrib = model_func(h, rng, struct.contribution, 0.0)
            result += contrib
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for serialization."""
        return {
            'nugget': self.nugget,
            'total_sill': self.total_sill,
            'n_structures': self.n_structures,
            'structures': [
                {
                    'model_type': s.model_type,
                    'contribution': s.contribution,
                    'range_major': s.range_major,
                    'range_minor': s.range_minor,
                    'range_vertical': s.range_vertical
                }
                for s in self.structures
            ]
        }
    
    @classmethod
    def from_single_fit(cls, nugget: float, psill: float, prange: float, 
                        model_type: str = 'spherical') -> 'NestedVariogramModel':
        """Create nested model from single-structure fit parameters."""
        return cls(
            nugget=nugget,
            structures=[NestedStructure(
                model_type=model_type,
                contribution=psill,
                range_major=prange
            )]
        )


def calculate_auto_lags(
    coords: np.ndarray,
    direction: str = 'horizontal',
    from_depths: Optional[np.ndarray] = None,
    to_depths: Optional[np.ndarray] = None,
    target_coverage: float = 0.5,
    base_n_lags: int = 15
) -> Tuple[int, float, float]:
    """
    Calculate optimal lag parameters based on data geometry and direction.
    
    This implements professional practice:
    - Horizontal: lag ≈ 0.5-1x drill spacing, n_lags to cover ~2-3x expected range
    - Downhole: lag = composite length, n_lags extended to see long-range structure (150m+)
    
    Parameters
    ----------
    coords : np.ndarray
        (N, 3) array of coordinates
    direction : str
        'horizontal', 'downhole', or 'vertical'
    from_depths, to_depths : np.ndarray, optional
        FROM/TO depths for composite length calculation
    target_coverage : float
        Fraction of maximum extent to cover (default 0.5 = half diagonal)
    base_n_lags : int
        Base number of lags for horizontal directions
    
    Returns
    -------
    n_lags, lag_distance, max_range : Tuple[int, float, float]
    """
    coords = np.asarray(coords, float)
    if coords.shape[0] < 2:
        return base_n_lags, 25.0, base_n_lags * 25.0
    
    # Calculate data extent
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents = maxs - mins
    
    if direction == 'downhole':
        # Downhole: use composite length as lag, extend to see 140-150m structure
        # Compute composite/sample length
        composite_length = 2.0  # Default if no FROM/TO
        if from_depths is not None and to_depths is not None:
            sample_lengths = np.abs(np.asarray(to_depths, float) - np.asarray(from_depths, float))
            valid_lengths = sample_lengths[~np.isnan(sample_lengths) & (sample_lengths > 0)]
            if len(valid_lengths) > 0:
                composite_length = float(np.nanmedian(valid_lengths))
        
        # IMPORTANT: Cap lag at reasonable maximum for nugget estimation
        # Typical composite lengths are 1-10m; if longer, data may be over-composited
        MAX_REASONABLE_DOWNHOLE_LAG = 10.0  # metres
        if composite_length > MAX_REASONABLE_DOWNHOLE_LAG:
            logger.warning(
                f"Composite length ({composite_length:.1f}m) exceeds typical range (1-10m). "
                f"This often indicates lithology-based compositing with few samples per hole. "
                f"Using {MAX_REASONABLE_DOWNHOLE_LAG:.1f}m as downhole lag. "
                f"For accurate nugget estimation, consider using raw assays."
            )
            lag_distance = MAX_REASONABLE_DOWNHOLE_LAG
        else:
            # Lag = composite length (industry standard)
            lag_distance = composite_length
        
        # Target max distance: 150m or 0.5 x mean hole length
        # This ensures we capture the long-range downhole structure (your ~140m range)
        vertical_extent = extents[2]  # Z extent
        target_max_dist = max(150.0, vertical_extent * target_coverage)
        
        # Calculate n_lags to reach target
        n_lags = max(15, int(np.ceil(target_max_dist / lag_distance)))
        n_lags = min(n_lags, 40)  # Cap at 40 lags to avoid over-computation
        
        max_range = lag_distance * n_lags
        logger.info(f"Auto-lags (downhole): composite={composite_length:.1f}m, lag={lag_distance:.1f}m, n_lags={n_lags}, max_dist={max_range:.1f}m")
        
    elif direction == 'vertical':
        # Vertical directional: shorter range, based on vertical extent
        vertical_extent = extents[2]
        lag_distance = max(5.0, vertical_extent / base_n_lags)
        n_lags = base_n_lags
        max_range = lag_distance * n_lags
        
    else:  # horizontal (omni, major, minor)
        # Horizontal: use drill spacing as guide
        horizontal_extent = np.sqrt(extents[0]**2 + extents[1]**2)
        
        # Estimate drill spacing using nearest neighbors
        try:
            if cKDTree is not None and coords.shape[0] >= 5:
                tree = cKDTree(coords[:, :2])  # XY only
                dists, _ = tree.query(coords[:, :2], k=2)
                drill_spacing = float(np.percentile(dists[:, 1], 75))
            else:
                drill_spacing = horizontal_extent / np.sqrt(coords.shape[0])
        except Exception:
            drill_spacing = horizontal_extent / np.sqrt(coords.shape[0])
        
        # Lag = 0.5-1x drill spacing
        lag_distance = max(5.0, drill_spacing * 0.5)
        
        # n_lags to cover ~2-3x expected range
        n_lags = base_n_lags
        max_range = lag_distance * n_lags
        
        # Ensure we cover reasonable extent
        target_extent = horizontal_extent * target_coverage
        if max_range < target_extent:
            n_lags = int(np.ceil(target_extent / lag_distance))
            n_lags = min(n_lags, 25)  # Cap
            max_range = lag_distance * n_lags
        
        logger.info(f"Auto-lags (horizontal): spacing~{drill_spacing:.1f}m, lag={lag_distance:.1f}m, n_lags={n_lags}")
    
    return n_lags, lag_distance, max_range


class Variogram3D:
    """
    3D Variogram Calculator and Model Fitter
    
    Calculates experimental variograms in 3D space and fits theoretical models.
    Supports omnidirectional and directional variograms with efficient vectorized computation.
    
    Professional features:
    - Auto num_lags: Automatically calculates optimal lag parameters per direction
    - Nested structures: Support for multi-structure models (short + long range)
    - Global nugget: Lock nugget across all directions for consistency
    """
    
    def __init__(self,
                 n_lags: int = 12,
                 lag_distance: float = 25.0,
                 lag_tolerance: Optional[float] = None,
                 max_range: Optional[float] = None,
                 model: ModelType = "spherical",
                 pair_cap: int = 200_000,
                 max_directional_samples: int = 1500,
                 z_positive_up: bool = True,
                 random_state: Optional[int] = 42,
                 auto_lags: bool = False,
                 n_structures: int = 1,
                 global_nugget: Optional[float] = None):
        """
        Initialize 3D Variogram calculator.
        
        Parameters
        ----------
        n_lags : int
            Default number of lag bins (must be >= 1). Ignored if auto_lags=True.
        lag_distance : float
            Default lag spacing in metres (must be > 0). Ignored if auto_lags=True.
        lag_tolerance : float, optional
            Lag tolerance in metres. Defaults to lag_distance * 0.3 (30% - tighter for better structure)
        max_range : float, optional
            Default maximum range for variogram calculation
        model : ModelType
            Default model type for fitting
        pair_cap : int
            Maximum number of pairs to evaluate for omnidirectional calculations
        max_directional_samples : int
            Maximum number of samples used in directional calculations to avoid O(N^2) blowups
        z_positive_up : bool
            Whether Z increases upward (True) or downward (False); affects dip sign
        random_state : int, optional
            Random seed for reproducible subsampling (default 42 for determinism).
            Set to None for non-reproducible random sampling (not recommended).
        auto_lags : bool
            If True, automatically calculate optimal lags per direction (professional mode)
        n_structures : int
            Number of nested structures (1 = single model, 2 = nested short+long range)
        global_nugget : float, optional
            If set, enforces this nugget value across all directions (addresses nugget inconsistency)
        """
        # Input validation
        if n_lags < 1:
            raise ValueError(f"n_lags must be >= 1, got {n_lags}")
        if lag_distance <= 0:
            raise ValueError(f"lag_distance must be > 0, got {lag_distance}")
        if pair_cap < 1:
            raise ValueError(f"pair_cap must be >= 1, got {pair_cap}")
        if max_directional_samples < 1:
            raise ValueError(f"max_directional_samples must be >= 1, got {max_directional_samples}")
        if n_structures < 1 or n_structures > 3:
            raise ValueError(f"n_structures must be 1-3, got {n_structures}")
        
        self.n_lags = int(n_lags)
        self.lag_distance = float(lag_distance)
        # Tighter default tolerance (30% instead of 50%) for better structure preservation
        self.lag_tolerance = float(lag_tolerance) if lag_tolerance is not None else self.lag_distance * 0.3
        if self.lag_tolerance <= 0:
            raise ValueError(f"lag_tolerance must be > 0, got {self.lag_tolerance}")
        if self.lag_tolerance > self.lag_distance:
            logger.warning(f"lag_tolerance ({self.lag_tolerance:.1f}m) exceeds lag_distance ({self.lag_distance:.1f}m). "
                          f"This may smear variogram structure. Industry norm is 20-40% of lag_distance.")
        
        self.max_range = float(max_range) if max_range is not None else self.n_lags * self.lag_distance
        if self.max_range <= 0:
            raise ValueError(f"max_range must be > 0, got {self.max_range}")
        
        self.model = model
        self.pair_cap = int(pair_cap)
        self.max_directional_samples = int(max_directional_samples)
        self.z_positive_up = bool(z_positive_up)
        self.random_state = random_state
        
        # Professional features
        self.auto_lags = bool(auto_lags)
        self.n_structures = int(n_structures)
        self.global_nugget = float(global_nugget) if global_nugget is not None else None
        
        # Cache for auto-calculated lags per direction
        self._auto_lag_cache: Dict[str, Tuple[int, float, float]] = {}

    # -------------------------
    # Omnidirectional
    # -------------------------
    def calculate_omnidirectional(self,
                                  coords: np.ndarray,
                                  values: np.ndarray,
                                  sample_weights: Optional[np.ndarray] = None,
                                  n_lags: Optional[int] = None,
                                  max_range: Optional[float] = None,
                                  pair_cap: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate omnidirectional variogram using cKDTree pairs within max_range.
        
        Parameters
        ----------
        coords : np.ndarray
            (N, 3) array of (x, y, z) coordinates
        values : np.ndarray
            (N,) array of property values
        sample_weights : np.ndarray, optional
            (N,) array of sample weights for declustering
        n_lags : int, optional
            Number of lag bins (default: self.n_lags)
        max_range : float, optional
            Maximum range for variogram (default: self.max_range)
        pair_cap : int, optional
            Maximum number of pairs to compute (default: self.pair_cap)
            
        Returns
        -------
        pd.DataFrame
            Variogram table with columns: distance, gamma, npairs
        """
        # Input validation
        if coords.shape[0] != len(values):
            raise ValueError(f"coords and values must have same length: coords={coords.shape[0]}, values={len(values)}")
        if coords.shape[1] != 3:
            raise ValueError(f"coords must be (N, 3) array, got shape {coords.shape}")
        if sample_weights is not None and len(sample_weights) != len(values):
            raise ValueError(f"sample_weights must have same length as values: weights={len(sample_weights)}, values={len(values)}")

        # Validate no NaN/Inf values
        if np.any(~np.isfinite(coords)):
            raise ValueError("coords must not contain NaN or Inf values")
        if np.any(~np.isfinite(values)):
            raise ValueError("values must not contain NaN or Inf values")
        if sample_weights is not None and np.any(~np.isfinite(sample_weights)):
            raise ValueError("sample_weights must not contain NaN or Inf values")

        nl = int(n_lags or self.n_lags)
        if nl < 1:
            raise ValueError(f"n_lags must be >= 1, got {nl}")
        mr = float(max_range or self.max_range)
        if mr <= 0:
            raise ValueError(f"max_range must be > 0, got {mr}")
        
        n = coords.shape[0]
        if n < 2:
            logger.warning("Insufficient data for variogram calculation (need at least 2 samples)")
            return pd.DataFrame({"distance": [], "gamma": [], "npairs": []})

        # If pair count would explode, subsample points to respect pair_cap
        cap = pair_cap or self.pair_cap
        total_pairs = n * (n - 1) // 2
        if cap and total_pairs > cap:
            m = int((1 + np.sqrt(1 + 8 * cap)) // 2)
            m = max(2, min(n, m))
            # Use reproducible RNG if random_state provided
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, size=m, replace=False)
            coords = coords[idx]
            values = values[idx]
            if sample_weights is not None:
                sample_weights = np.asarray(sample_weights, float)[idx]
            logger.info(f"Subsampled from {n} to {m} points to respect pair_cap={cap}")

        if cKDTree is not None:
            tree = cKDTree(coords)
            pairs = tree.query_pairs(r=mr)
            # Use _sorted_pairs_array for deterministic ordering
            pairs_arr = _sorted_pairs_array(pairs)
        else:
            # Fallback to dense pairs if SciPy unavailable
            idx_i, idx_j = np.triu_indices(coords.shape[0], k=1)
            pairs_arr = np.vstack((idx_i, idx_j)).T

        if cap and len(pairs_arr) > cap:
            # Use reproducible RNG if random_state provided
            original_count = len(pairs_arr)
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(pairs_arr), size=cap, replace=False)
            pairs_arr = pairs_arr[idx]
            seed_info = f" (seed={self.random_state})" if self.random_state is not None else " (random)"
            logger.info(f"Subsampled pairs from {original_count:,} to {cap:,}{seed_info}")

        if pairs_arr.size == 0:
            return pd.DataFrame({"distance": [], "gamma": [], "npairs": []})

        dists, semis, _ = calculate_pair_attributes(coords, values, pairs_arr[:, 0], pairs_arr[:, 1])
        pair_weights = None
        if sample_weights is not None:
            sw = np.asarray(sample_weights, float)
            pair_weights = sw[pairs_arr[:, 0]] * sw[pairs_arr[:, 1]]
        edges = np.linspace(0.0, mr, nl + 1)
        bins = np.digitize(dists, edges) - 1
        valid = (bins >= 0) & (bins < nl)
        if not np.any(valid):
            return pd.DataFrame({"distance": [], "gamma": [], "npairs": []})
        bins = bins[valid]; dists = dists[valid]; semis = semis[valid]
        if pair_weights is not None:
            pair_weights = pair_weights[valid]
            df = pd.DataFrame({"bin": bins, "d": dists, "g": semis, "w": pair_weights})
            out = df.groupby("bin").apply(
                lambda x: pd.Series({
                    "distance": np.average(x["d"], weights=x["w"]),
                    "gamma": np.average(x["g"], weights=x["w"]),
                    "npairs": float(x["w"].sum())
                })
            ).reset_index()
        else:
            df = pd.DataFrame({"bin": bins, "d": dists, "g": semis})
            out = df.groupby("bin").agg(distance=("d", "mean"),
                                        gamma=("g", "mean"),
                                        npairs=("g", "size")).reset_index(drop=True)
        return out

    # -------------------------
    # Directional by cone tolerance
    # -------------------------
    def calculate_directional(self,
                              coords: np.ndarray,
                              values: np.ndarray,
                              sample_weights: Optional[np.ndarray],
                              azimuth_deg: float,
                              dip_deg: float,
                              cone_tolerance: float = 15.0,
                              n_lags: Optional[int] = None,
                              max_range: Optional[float] = None,
                              max_samples: Optional[int] = None,
                              pair_cap: Optional[int] = None,
                              bandwidth: Optional[float] = None,
                              ) -> pd.DataFrame:
        """
        Calculate directional variogram along a specific orientation using cone tolerance.
        
        Parameters
        ----------
        coords : np.ndarray
            (N, 3) array of (x, y, z) coordinates
        values : np.ndarray
            (N,) array of property values
        azimuth_deg : float
            Direction azimuth in degrees (0=North, clockwise). Will be normalized to [0, 360).
        dip_deg : float
            Direction dip in degrees (0=horizontal, positive down from horizontal)
        cone_tolerance : float
            Angular tolerance in degrees for direction matching (default: 15.0 - tighter for better anisotropy discrimination)
        n_lags : int, optional
            Number of lag bins (default: self.n_lags)
        max_range : float, optional
            Maximum range for variogram (default: auto-calculated or self.max_range)
        max_samples : int, optional
            Maximum number of samples used (subsampled) to limit pair counts
        pair_cap : int, optional
            Maximum number of pairs to compute (currently applied via sample thinning)
        bandwidth : float, optional
            Maximum perpendicular distance from direction vector
            
        Returns
        -------
        pd.DataFrame
            Directional variogram table with columns: distance, gamma, npairs
        """
        # Input validation
        if coords.shape[0] != len(values):
            raise ValueError(f"coords and values must have same length: coords={coords.shape[0]}, values={len(values)}")
        if coords.shape[1] != 3:
            raise ValueError(f"coords must be (N, 3) array, got shape {coords.shape}")
        if sample_weights is not None and len(sample_weights) != len(values):
            raise ValueError(f"sample_weights must have same length as values: weights={len(sample_weights)}, values={len(values)}")
        
        # Normalize azimuth to [0, 360)
        azimuth_deg = azimuth_deg % 360.0
        
        # Warn if cone tolerance is too wide (industry norm is 10-20°)
        if cone_tolerance > 22.5:
            logger.warning(f"Cone tolerance ({cone_tolerance:.1f}°) exceeds industry norm (10-20°). "
                          f"This may smear anisotropic structure. Recommended: 15° for Fe/Cu deposits.")
        
        nl = int(n_lags or self.n_lags)
        if nl < 1:
            raise ValueError(f"n_lags must be >= 1, got {nl}")
        max_samples = int(max_samples or self.max_directional_samples)
        mr = float(max_range or self.max_range)
        if mr <= 0:
            raise ValueError(f"max_range must be > 0, got {mr}")
        bw = float(bandwidth) if bandwidth is not None else np.inf

        # Subsample points to cap pair counts for directional computation
        if coords.shape[0] > max_samples:
            # Use reproducible RNG if random_state provided
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(coords.shape[0], size=max_samples, replace=False)
            coords = coords[idx]
            values = values[idx]
            if sample_weights is not None:
                sample_weights = np.asarray(sample_weights, float)[idx]
            logger.info(f"Subsampled from {coords.shape[0]} to {max_samples} points for directional variogram")

        # Build cKDTree for efficient neighbor search within range
        tree = cKDTree(coords) if cKDTree is not None else None

        # Unit vector of direction from azimuth (clockwise from +Y north) and dip.
        # We assume standard mine-survey convention: azimuth from North rotating to East, dip positive down from horizontal.
        az = np.deg2rad(azimuth_deg)
        dip = np.deg2rad(dip_deg)
        # Direction cosines in right-handed XYZ: X-east, Y-north, Z-up (negative down)
        # For Z-up coordinate systems used in many apps, a downward dip reduces Z.
        ux = np.sin(az) * np.cos(dip)
        uy = np.cos(az) * np.cos(dip)
        uz = -np.sin(dip) if self.z_positive_up else np.sin(dip)
        u = np.array([ux, uy, uz], dtype=float)
        u /= np.linalg.norm(u) + 1e-12

        # Gather pairs with cone + bandwidth filter
        dists_list, semis_list = [], []
        target_pairs = pair_cap or self.pair_cap
        pairs_accumulated = 0

        if tree is not None:
            pairs = tree.query_pairs(r=mr)
            # Use _sorted_pairs_array for deterministic ordering
            pairs_arr = _sorted_pairs_array(pairs)
        else:
            idx_i, idx_j = np.triu_indices(coords.shape[0], k=1)
            pairs_arr = np.vstack((idx_i, idx_j)).T

        if target_pairs and len(pairs_arr) > target_pairs:
            # Use reproducible RNG if random_state provided
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(pairs_arr), size=target_pairs, replace=False)
            pairs_arr = pairs_arr[idx]

        dists, semis, vecs = calculate_pair_attributes(coords, values, pairs_arr[:, 0], pairs_arr[:, 1])
        pair_weights = None
        if sample_weights is not None:
            sw = np.asarray(sample_weights, float)
            pair_weights = sw[pairs_arr[:, 0]] * sw[pairs_arr[:, 1]]
        
        # Improved angle filtering using dot-product threshold (more numerically stable than arccos)
        lens = dists + 1e-12
        dots = np.einsum('ij,j->i', vecs, u)
        # Normalize dot products to get cosine of angle
        cosang = dots / lens
        # Clip to valid range [-1, 1] to avoid numerical issues
        cosang = np.clip(cosang, -1.0, 1.0)
        
        # Use cosine threshold instead of arccos for better numerical stability
        # cos(cone_tolerance) gives the threshold - pairs with |cos(angle)| >= this are within tolerance
        cos_threshold = np.cos(np.deg2rad(cone_tolerance))
        # For directional variogram, we want pairs aligned with direction (positive dot product)
        # and within cone tolerance
        angle_mask = np.abs(cosang) >= cos_threshold
        
        # Perpendicular distance filtering (bandwidth)
        sin_theta = np.sqrt(1.0 - cosang**2)
        perp_dist = dists * sin_theta
        bw_mask = perp_dist <= bw
        mask = angle_mask & bw_mask
        if not np.any(mask):
            return pd.DataFrame({"distance": [], "gamma": [], "npairs": []})
        dists = dists[mask]; semis = semis[mask]
        if pair_weights is not None:
            pair_weights = pair_weights[mask]
        # Bin
        edges = np.linspace(0.0, mr, nl + 1)
        bins = np.digitize(dists, edges) - 1
        valid = (bins >= 0) & (bins < nl)
        if not np.any(valid):
            return pd.DataFrame({"distance": [], "gamma": [], "npairs": []})
        bins = bins[valid]; dists = dists[valid]; semis = semis[valid]
        if pair_weights is not None:
            pair_weights = pair_weights[valid]
            df = pd.DataFrame({"bin": bins, "d": dists, "g": semis, "w": pair_weights})
            out = df.groupby("bin").apply(
                lambda x: pd.Series({
                    "distance": np.average(x["d"], weights=x["w"]),
                    "gamma": np.average(x["g"], weights=x["w"]),
                    "npairs": float(x["w"].sum())
                })
            ).reset_index()
        else:
            df = pd.DataFrame({"bin": bins, "d": dists, "g": semis})
            out = df.groupby("bin").agg(distance=("d", "mean"),
                                        gamma=("g", "mean"),
                                        npairs=("g", "size")).reset_index(drop=True)
        return out

    # -------------------------
    # Downhole variogram (nugget-focused)
    # -------------------------
    def calculate_downhole(self,
                           coords: np.ndarray,
                           values: np.ndarray,
                           hole_ids: np.ndarray,
                           from_depths: Optional[np.ndarray] = None,
                           to_depths: Optional[np.ndarray] = None,
                           sample_weights: Optional[np.ndarray] = None,
                           n_lags: Optional[int] = None,
                           max_range: Optional[float] = None) -> pd.DataFrame:
        """
        Compute downhole variogram by pairing samples within the same hole.
        
        This is CRITICAL for nugget estimation because:
        1. Along-hole distance reflects actual sample spacing
        2. First lags capture short-range variability + measurement error
        3. Adjacent samples (h = composite length) give best nugget estimate
        
        For proper nugget estimation:
        - Use tight first lag bins (ideally < sample length)
        - Include duplicate pairs if available (h ≈ 0)
        - First 2-3 lags are most important for nugget extrapolation
        """
        # CRITICAL: Downhole variogram uses VERTICAL spacing, NOT horizontal!
        # Horizontal lag (25-100m) is for Major/Minor/Omni variograms
        # Downhole lag should be based on composite/sample length (1-5m typically)
        
        # Compute composite/sample length
        sample_length = None
        if from_depths is not None and to_depths is not None:
            sample_lengths = np.abs(np.asarray(to_depths, float) - np.asarray(from_depths, float))
            valid_lengths = sample_lengths[~np.isnan(sample_lengths) & (sample_lengths > 0)]
            if len(valid_lengths) > 0:
                sample_length = float(np.nanmedian(valid_lengths))
        
        # CRITICAL: Downhole variogram uses VERTICAL spacing, NOT horizontal lag_distance!
        # Horizontal lag (self.lag_distance = 25-100m) is for Major/Minor/Omni variograms
        # Downhole lag should be based on composite/sample length (1-5m typically)
        
        # Compute downhole-specific lag = composite length (industry standard)
        if sample_length is not None and sample_length > 0:
            downhole_lag = sample_length * 1.0  # Lag = composite length
        else:
            # Fallback: use small default (2m) if no FROM/TO data
            downhole_lag = 2.0
        
        # Number of lags for downhole (typically 10-20 for nugget estimation)
        nl = int(n_lags or self.n_lags)
        nl = max(10, min(25, nl))  # Clamp to reasonable range
        
        # Simplified max-range calculation: one composite length per lag, capped at 10-15 lags
        # This is sufficient for nugget estimation and avoids over-engineering
        if max_range is None:
            # Standard approach: lag * n_lags, capped at reasonable vertical range
            mr = downhole_lag * nl
            if sample_length is not None and sample_length > 0:
                # Cap at 15x composite length (industry standard for vertical structure)
                max_reasonable = sample_length * 15
                if mr > max_reasonable:
                    nl = min(nl, 15)  # Cap at 15 lags
                    mr = downhole_lag * nl
        else:
            mr = float(max_range)
            if mr <= 0:
                raise ValueError(f"max_range must be > 0, got {mr}")
        
        sample_info = f"{sample_length:.2f}m" if sample_length is not None else "unknown"
        logger.debug(f"Downhole variogram: lag={downhole_lag:.2f}m, n_lags={nl}, max_range={mr:.1f}m (composite={sample_info})")

        all_dists: List[float] = []
        all_semis: List[float] = []

        df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
        df["val"] = values
        df["hole"] = hole_ids
        
        # Calculate mid-depth for sorting samples along hole
        # CRITICAL: Use FROM/TO depths for along-hole distance, NOT 3D coordinates!
        # 3D coordinates may all be at collar location before desurveying
        use_depth_distance = False
        if from_depths is not None and to_depths is not None:
            df["FROM"] = np.asarray(from_depths, float)
            df["TO"] = np.asarray(to_depths, float)
            df["MID"] = 0.5 * (df["FROM"] + df["TO"])
            use_depth_distance = True
        elif from_depths is not None:
            df["MID"] = np.asarray(from_depths, float)
            df["FROM"] = df["MID"]
            use_depth_distance = True
        elif to_depths is not None:
            df["MID"] = np.asarray(to_depths, float)
            df["TO"] = df["MID"]
            use_depth_distance = True
        else:
            # Fall back to Z coordinate (works for roughly vertical holes)
            df["MID"] = df["Z"].astype(float)
            use_depth_distance = False

        total_pairs = 0
        unique_holes = df["hole"].unique()
        logger.info(f"Downhole variogram: {len(unique_holes)} holes, use_depth_distance={use_depth_distance}, max_range={mr:.1f}")
        
        if use_depth_distance:
            depth_range = df["MID"].max() - df["MID"].min()
            logger.info(f"Depth range: {df['MID'].min():.1f} to {df['MID'].max():.1f} (span: {depth_range:.1f})")
        else:
            z_range = df["Z"].max() - df["Z"].min()
            logger.info(f"Z range: {df['Z'].min():.1f} to {df['Z'].max():.1f} (span: {z_range:.1f})")

        holes_processed = 0
        holes_skipped = 0
        
        for hole_name in unique_holes:
            group = df[df["hole"] == hole_name]
            if len(group) < 2:
                holes_skipped += 1
                if holes_skipped == 1:  # Log warning on first skip
                    logger.warning(f"Hole '{hole_name}' has only {len(group)} sample(s) - skipping. "
                                  f"This may indicate data quality issues. Check for missing samples or incorrect HOLEID assignment.")
                continue
            
            # Sort by mid-depth
            g = group.sort_values("MID").reset_index(drop=True)
            g_vals = g["val"].to_numpy(float)
            n = len(g_vals)
            
            # Compute along-hole distance
            # PREFER depth-based distance (FROM/TO) over 3D coordinates
            # because 3D coords may not be computed yet (all at collar)
            if use_depth_distance and "MID" in g.columns:
                # Use mid-depth differences as along-hole distance
                mid_depths = g["MID"].to_numpy(float)
                along = mid_depths - mid_depths[0]  # Distance from first sample
                
                # Log first hole for debugging
                if holes_processed == 0:
                    logger.info(f"First hole '{hole_name}': {n} samples, depth range {mid_depths.min():.1f} to {mid_depths.max():.1f}")
            else:
                # Fallback: use 3D coordinate distance
                g_coords = g[["X", "Y", "Z"]].to_numpy(float)
                segment_lengths = np.linalg.norm(np.diff(g_coords, axis=0), axis=1)
                along = np.concatenate([[0.0], np.cumsum(segment_lengths)])
                
                # Log first hole for debugging
                if holes_processed == 0:
                    logger.info(f"First hole '{hole_name}': {n} samples, 3D distance span {np.max(along):.2f}")
                
                # Check if 3D distances are essentially zero (all at same location)
                if np.max(along) < 0.01:
                    # Try using Z differences as proxy for depth
                    z_vals = g["Z"].to_numpy(float)
                    along = np.abs(z_vals - z_vals[0])
                    if np.max(along) < 0.01:
                        logger.warning(f"Hole {hole_name}: all {n} samples at same location (3D and Z), skipping")
                        holes_skipped += 1
                        continue
            
            holes_processed += 1
            
            # Vectorized pair computation
            idx_i, idx_j = np.triu_indices(n, k=1)
            dh = np.abs(along[idx_j] - along[idx_i])
            
            # Filter by max range
            in_range = dh <= mr
            if not np.any(in_range):
                continue
                
            idx_i = idx_i[in_range]
            idx_j = idx_j[in_range]
            dh = dh[in_range]
            
            # Compute semivariance
            sem = 0.5 * (g_vals[idx_j] - g_vals[idx_i]) ** 2
            
            all_dists.extend(dh.tolist())
            all_semis.extend(sem.tolist())
            total_pairs += len(dh)

        if not all_dists:
            logger.warning(f"No valid downhole pairs found. Processed {holes_processed} holes, skipped {holes_skipped}")
            if holes_skipped == len(unique_holes):
                # All holes have only 1 sample - likely composited data
                logger.warning(
                    "ROOT CAUSE: All holes have only 1 sample each. This commonly occurs after "
                    "lithology-based compositing when each hole intersects only one lithological unit. "
                    "RECOMMENDATION: Use raw assays (not composites) for downhole variogram to estimate nugget effect. "
                    "In the Variogram panel, select 'Raw Assays' as the data source for accurate nugget estimation."
                )
            else:
                logger.warning("Check: 1) HOLEID column detected? 2) FROM/TO columns exist? 3) Multiple samples per hole?")
            return pd.DataFrame({"distance": [], "gamma": [], "npairs": []})

        # Log distance statistics for debugging
        dists_arr_debug = np.array(all_dists)
        logger.info(f"Downhole variogram: {total_pairs} pairs from {holes_processed} holes (skipped {holes_skipped})")
        logger.info(f"Distance range: {dists_arr_debug.min():.2f} to {dists_arr_debug.max():.2f} m (median: {np.median(dists_arr_debug):.2f})")

        edges = np.linspace(0.0, mr, nl + 1)
        dists_arr = np.asarray(all_dists)
        semis_arr = np.asarray(all_semis)
        bins = np.digitize(dists_arr, edges) - 1
        valid = (bins >= 0) & (bins < nl)
        if not np.any(valid):
            return pd.DataFrame({"distance": [], "gamma": [], "npairs": []})
        bins = bins[valid]
        dists_arr = dists_arr[valid]
        semis_arr = semis_arr[valid]
        df_out = pd.DataFrame({"bin": bins, "d": dists_arr, "g": semis_arr})
        return df_out.groupby("bin").agg(distance=("d", "mean"),
                                         gamma=("g", "mean"),
                                         npairs=("g", "size")).reset_index(drop=True)

    # -------------------------
    # Model fitting that reuses shared fitter
    # -------------------------
    def fit_model(self, distances: np.ndarray, gamma: np.ndarray, model_type: ModelType, sill_norm: bool = False, sill_cap: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Fit model returning nugget, partial sill, range.
        
        Requires minimum data density for reliable fitting (at least 3 lags with pairs).
        
        Parameters
        ----------
        distances : array
            Lag distances
        gamma : array
            Semivariances
        model_type : str
            Variogram model type
        sill_norm : bool
            Whether to normalize by sample variance
        sill_cap : float, optional
            Maximum allowable sill value. IMPORTANT for geostatistical soundness:
            should be set to the sample variance or omnidirectional sill to prevent
            unrealistic fits in directions with sparse data.
        """
        # Input validation
        if len(distances) != len(gamma):
            raise ValueError(f"distances and gamma must have same length: distances={len(distances)}, gamma={len(gamma)}")
        
        # Check minimum data requirements
        valid_mask = ~(np.isnan(distances) | np.isnan(gamma))
        n_valid = np.sum(valid_mask)
        if n_valid < 3:
            logger.warning(f"Insufficient data for variogram fitting: {n_valid} valid points (need >= 3). "
                          f"Skipping model fit.")
            raise ValueError(f"Insufficient data for variogram fitting: {n_valid} valid points (need >= 3)")
        
        # Check for minimum lag density (at least 3 lags with pairs)
        unique_lags = len(np.unique(np.round(distances[valid_mask], decimals=1)))
        if unique_lags < 3:
            logger.warning(f"Insufficient lag diversity for variogram fitting: {unique_lags} unique lags (need >= 3). "
                          f"Results may be unreliable.")
        
        try:
            max_lag = float(np.nanmax(distances[valid_mask])) if n_valid > 0 else None
            nugget, sill, prange = fit_variogram_model(
                distances[valid_mask], 
                gamma[valid_mask], 
                model_type, 
                max_lag=max_lag, 
                sill_norm=sill_norm,
                sill_cap=sill_cap
            )
            psill = max(sill - nugget, 0.0)
            
            # Apply global nugget if set
            if self.global_nugget is not None:
                logger.info(f"Enforcing global nugget: {self.global_nugget:.3f} (fitted was {nugget:.3f})")
                # Adjust partial sill to maintain total sill
                psill = max(nugget + psill - self.global_nugget, 0.0)
                nugget = self.global_nugget
            
            return nugget, psill, prange
        except (ValueError, RuntimeError) as e:
            # Numerical optimization failures
            logger.warning(f"Variogram fitting failed (numerical issue): {e}")
            # Use more geologically meaningful fallback
            g = np.asarray(gamma[valid_mask], float)
            d = np.asarray(distances[valid_mask], float)
            if len(g) == 0:
                raise ValueError("No valid data points for fitting")
            nug = max(np.nanmin(g), 0.0)
            tot = max(np.nanmax(g), nug + 1e-6)
            # Use first non-zero lag as range estimate (more meaningful than median)
            non_zero_d = d[d > 0]
            if len(non_zero_d) > 0:
                pr = float(np.nanpercentile(non_zero_d, 50))  # Median of non-zero distances
            else:
                pr = max(np.nanmedian(d), 1.0)
            
            # Apply global nugget if set
            if self.global_nugget is not None:
                nug = self.global_nugget
            
            logger.info(f"Using fallback parameters: nugget={nug:.3f}, sill={tot-nug:.3f}, range={pr:.1f}")
            return nug, tot - nug, pr
        except Exception as e:
            # Unexpected errors should be re-raised
            logger.error(f"Unexpected error in variogram fitting: {e}", exc_info=True)
            raise

    def fit_nested_model(
        self, 
        distances: np.ndarray, 
        gamma: np.ndarray, 
        model_type: ModelType,
        n_structures: Optional[int] = None,
        sill_norm: bool = False
    ) -> NestedVariogramModel:
        """
        Fit nested variogram model with multiple structures.
        
        For n_structures=2, fits:
        γ(h) = C0 + C1·Sph(h/a1) + C2·Sph(h/a2)
        
        where a1 < a2 (short range + long range).
        
        Parameters
        ----------
        distances : np.ndarray
            Lag distances
        gamma : np.ndarray
            Experimental semivariance values
        model_type : ModelType
            Model type for each structure
        n_structures : int, optional
            Number of structures (default: self.n_structures)
        sill_norm : bool
            Whether to normalize by sill during fitting
            
        Returns
        -------
        NestedVariogramModel
            Fitted nested model with nugget and structures
        """
        n_struct = n_structures or self.n_structures
        
        # Input validation
        valid_mask = ~(np.isnan(distances) | np.isnan(gamma))
        d = np.asarray(distances[valid_mask], float)
        g = np.asarray(gamma[valid_mask], float)
        
        if len(d) < 4:
            raise ValueError(f"Insufficient data for nested fitting: {len(d)} points (need >= 4)")
        
        # Sort by distance
        sort_idx = np.argsort(d)
        d = d[sort_idx]
        g = g[sort_idx]
        
        # Single structure - delegate to simple fit
        if n_struct == 1:
            nugget, psill, prange = self.fit_model(distances, gamma, model_type, sill_norm)
            return NestedVariogramModel.from_single_fit(nugget, psill, prange, model_type)
        
        # Two structures: short-range + long-range
        # Industry approach: fit short-range first (early lags), then fit residual
        
        # 1. Estimate nugget from first 2-3 lags
        early_lags = d[:min(3, len(d))]
        early_gamma = g[:min(3, len(g))]
        nugget_est = max(0.0, float(np.min(early_gamma)))
        
        # Apply global nugget if set
        if self.global_nugget is not None:
            nugget_est = self.global_nugget
        
        # 2. Estimate total sill from plateau
        sill_est = float(np.nanpercentile(g, 90))
        
        # 3. Find inflection point for structure separation
        # Look for where the slope changes significantly
        max_d = d.max()
        
        # Short range: first ~30% of max distance
        short_range_threshold = max_d * 0.3
        short_mask = d <= short_range_threshold
        
        if np.sum(short_mask) >= 3:
            # Fit short range structure
            try:
                nug1, sill1, range1 = fit_variogram_model(
                    d[short_mask], g[short_mask], model_type, sill_norm=sill_norm
                )
                # Partial sill for short structure
                c1 = max(sill1 - nug1, 0.0)
            except Exception:
                c1 = (sill_est - nugget_est) * 0.3
                range1 = max_d * 0.2
        else:
            c1 = (sill_est - nugget_est) * 0.3
            range1 = max_d * 0.2
        
        # Long range structure gets remaining variance
        c2 = max(sill_est - nugget_est - c1, 0.0)
        range2 = max_d * 0.7  # Longer range
        
        # Fine-tune with scipy optimization if available
        try:
            from scipy.optimize import minimize
            
            def nested_objective(params):
                """Objective function for nested model fitting."""
                c1_opt, r1_opt, c2_opt, r2_opt = params
                if c1_opt < 0 or c2_opt < 0 or r1_opt <= 0 or r2_opt <= 0:
                    return 1e10
                
                model_func = MODEL_MAP.get(model_type, MODEL_MAP['spherical'])
                y_pred = nugget_est + model_func(d, r1_opt, c1_opt, 0.0) + model_func(d, r2_opt, c2_opt, 0.0)
                
                # Weighted least squares (weight by 1/distance to emphasize early lags)
                weights = 1.0 / (d + 1.0)
                return float(np.sum(weights * (g - y_pred)**2))
            
            # Initial guess
            x0 = [c1, range1, c2, range2]
            bounds = [
                (0, sill_est),  # c1
                (1.0, max_d),   # r1
                (0, sill_est),  # c2
                (1.0, max_d * 1.5)  # r2
            ]
            
            result = minimize(nested_objective, x0, bounds=bounds, method='L-BFGS-B')
            if result.success:
                c1, range1, c2, range2 = result.x
                logger.info(f"Nested fit optimized: C1={c1:.2f}, R1={range1:.1f}, C2={c2:.2f}, R2={range2:.1f}")
        except ImportError:
            logger.warning("scipy not available for nested optimization, using initial estimates")
        except Exception as e:
            logger.warning(f"Nested optimization failed: {e}, using initial estimates")
        
        # Ensure range1 < range2 (short range < long range)
        if range1 > range2:
            range1, range2 = range2, range1
            c1, c2 = c2, c1
        
        # Build nested model
        structures = [
            NestedStructure(model_type=model_type, contribution=c1, range_major=range1),
            NestedStructure(model_type=model_type, contribution=c2, range_major=range2)
        ]
        
        return NestedVariogramModel(nugget=nugget_est, structures=structures)

    def evaluate_model(self, h: np.ndarray, model_type: ModelType, nugget: float, psill: float, prange: float) -> np.ndarray:
        """
        Evaluate variogram model at given distances.
        
        Parameters
        ----------
        h : np.ndarray
            Distances at which to evaluate
        model_type : ModelType
            Model type
        nugget : float
            Nugget effect
        psill : float
            Partial sill
        prange : float
            Practical range
            
        Returns
        -------
        np.ndarray
            Modeled gamma values
        """
        return MODEL_MAP[model_type](h, prange, psill + nugget, nugget)


# Backward compatibility: keep old API structure for existing code
def run_variogram_pipeline(
    data: pd.DataFrame,
    xcol: str = "X",
    ycol: str = "Y",
    zcol: str = "Z",
    vcol: str = "Fe",
    hole_id_col: Optional[str] = None,
    from_col: Optional[str] = None,
    to_col: Optional[str] = None,
    default_azimuth: Optional[float] = None,
    default_dip: Optional[float] = None,
    z_positive_up: bool = True,
    nlag: int = 12,
    lag_distance: float = 25.0,
    lag_tolerance: Optional[float] = None,
    azimuth_tolerance: Optional[float] = None,
    dip_tolerance: Optional[float] = None,
    model_types: Optional[list] = None,
    use_sill_norm: bool = True,
    random_state: Optional[int] = 42,
    # New professional features
    auto_lags: bool = False,
    n_structures: int = 1,
    global_nugget: Optional[float] = None,
    # Progress callback for UI updates
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> dict:
    """
    Run complete 3D variogram analysis pipeline (backward compatibility wrapper).
    
    This function wraps the new Variogram3D class to maintain compatibility with existing code.
    """
    logger.info("Starting 3D variogram pipeline")
    logger.info(f"Data: {len(data)} points, variable: {vcol}")

    # Progress callback for UI
    def update_progress(percent: int, message: str):
        if progress_callback:
            progress_callback(percent, message)

    update_progress(0, "Preparing data...")

    # Build list of required columns (coordinates + value)
    required_cols = [xcol, ycol, zcol, vcol]
    
    # Identify hole ID column (needed for downhole variogram)
    actual_hole_col = None
    if hole_id_col and hole_id_col in data.columns:
        actual_hole_col = hole_id_col
    else:
        # Auto-detect hole ID column - comprehensive list of common variations
        hole_id_candidates = [
            "HOLEID", "HOLE_ID", "BHID", "hole_id", "HoleID", "Hole_ID",
            "DRILLHOLE", "DrillHole", "drillhole", "DRILL_HOLE", "Drill_Hole",
            "DRILL_ID", "DrillID", "DHOLE", "DH_ID", "DHID", "DH",
            "BOREHOLE", "Borehole", "borehole", "BORE_ID", "BoreID",
            "HOLE", "Hole", "hole", "ID", "id", "WELLID", "WELL_ID",
            "COLLAR_ID", "CollarID", "COLLAR", "Collar",
            # With underscores and variations
            "HOLE_NAME", "HoleName", "hole_name",
        ]
        for candidate in hole_id_candidates:
            if candidate in data.columns:
                actual_hole_col = candidate
                break
        
        # If still not found, try case-insensitive search
        if actual_hole_col is None:
            lower_cols = {c.lower(): c for c in data.columns}
            for pattern in ["holeid", "hole_id", "bhid", "drillhole", "drill_id", "dh_id"]:
                if pattern in lower_cols:
                    actual_hole_col = lower_cols[pattern]
                    logger.info(f"Found hole ID column via case-insensitive match: {actual_hole_col}")
                    break
    
    # Identify FROM/TO columns (needed for downhole variogram)
    actual_from_col = None
    actual_to_col = None
    if from_col and from_col in data.columns:
        actual_from_col = from_col
    else:
        # Try many common variations
        for candidate in ["FROM", "DEPTH_FROM", "from", "From", "FROM_", "FROMDEPTH", 
                          "FROM_DEPTH", "SAMPFROM", "SAMP_FROM", "SAMPLE_FROM", "FROMPTH"]:
            if candidate in data.columns:
                actual_from_col = candidate
                break
    if to_col and to_col in data.columns:
        actual_to_col = to_col
    else:
        for candidate in ["TO", "DEPTH_TO", "to", "To", "TO_", "TODEPTH", 
                          "TO_DEPTH", "SAMPTO", "SAMP_TO", "SAMPLE_TO", "TOPTH"]:
            if candidate in data.columns:
                actual_to_col = candidate
                break
    
    # Log column detection results for debugging - AUDIT IMPROVEMENT
    logger.info(f"Column detection: hole_id={actual_hole_col}, from={actual_from_col}, to={actual_to_col}")
    
    # Provide actionable warning if columns not found
    if actual_hole_col is None:
        logger.warning(
            "HOLEID column not found. Downhole variogram will use vertical proxy. "
            f"Available columns: {list(data.columns)[:20]}... "
            "For accurate nugget estimation, ensure data has HOLEID/BHID column."
        )
    if not actual_from_col or not actual_to_col:
        logger.warning(
            "FROM/TO columns not found. Downhole variogram will use Z coordinates as fallback. "
            "For accurate downhole spacing, ensure data has FROM/TO depth columns."
        )
    
    # Add optional columns to keep (but don't require them for dropna)
    cols_to_keep = required_cols.copy()
    if actual_hole_col:
        cols_to_keep.append(actual_hole_col)
    if actual_from_col:
        cols_to_keep.append(actual_from_col)
    if actual_to_col:
        cols_to_keep.append(actual_to_col)
    
    # Only keep columns that exist
    cols_to_keep = [c for c in cols_to_keep if c in data.columns]
    
    # Select columns and drop rows with NaN in required columns only
    clean_data = data[cols_to_keep].dropna(subset=required_cols)
    
    if len(clean_data) == 0:
        raise ValueError("All data contains NaN values. Cannot compute variogram.")
    
    if len(clean_data) < len(data):
        removed = len(data) - len(clean_data)
        logger.warning(f"Removed {removed} rows with NaN values ({removed/len(data)*100:.1f}%)")
    
    # Extract coordinates and values
    coords = clean_data[[xcol, ycol, zcol]].values
    values = clean_data[vcol].values
    
    # Simple declustering weights: inverse count per lag_distance cube
    try:
        cell = max(lag_distance, 1e-6)
        mins = coords.min(axis=0)
        keys = np.floor((coords - mins) / cell).astype(int)
        # encode keys to tuples for hashing
        tuples = [tuple(k) for k in keys]
        counts: Dict[tuple, int] = {}
        for t in tuples:
            counts[t] = counts.get(t, 0) + 1
        sample_weights = np.array([1.0 / counts[t] for t in tuples], dtype=float)
    except Exception:
        sample_weights = None
    
    # Calculate max range (default to nlag * lag_distance)
    max_range = float(lag_distance * nlag)
    
    if model_types is None:
        model_types = ['spherical', 'exponential']
    
    # Use new Variogram3D class with tighter defaults
    # Default lag_tolerance: 30% of lag_distance (was 50%)
    default_lag_tol = lag_distance * 0.3 if lag_tolerance is None else lag_tolerance
    # Default cone tolerances: 15° (was 22.5°) for better anisotropy discrimination
    default_az_tol = 15.0 if azimuth_tolerance is None else azimuth_tolerance
    default_dip_tol = 15.0 if dip_tolerance is None else dip_tolerance
    
    vgm = Variogram3D(
        n_lags=nlag,
        lag_distance=lag_distance,
        lag_tolerance=default_lag_tol,
        max_range=max_range,
        model=model_types[0] if model_types else "spherical",
        z_positive_up=z_positive_up,
        random_state=random_state,  # Enables reproducible results
        auto_lags=auto_lags,
        n_structures=n_structures,
        global_nugget=global_nugget
    )
    
    # Auto-calculate lag parameters if enabled
    if auto_lags:
        logger.info("Auto-lags enabled: calculating optimal parameters per direction")
        
        # Get FROM/TO values for composite length calculation
        from_vals_for_auto = clean_data[actual_from_col].values if actual_from_col and actual_from_col in clean_data.columns else None
        to_vals_for_auto = clean_data[actual_to_col].values if actual_to_col and actual_to_col in clean_data.columns else None
        
        # Calculate auto-lags for horizontal directions
        h_nlags, h_lag_dist, h_max_range = calculate_auto_lags(
            coords, direction='horizontal',
            from_depths=from_vals_for_auto, to_depths=to_vals_for_auto,
            base_n_lags=nlag
        )
        
        # Calculate auto-lags for downhole (extended to ~150m)
        d_nlags, d_lag_dist, d_max_range = calculate_auto_lags(
            coords, direction='downhole',
            from_depths=from_vals_for_auto, to_depths=to_vals_for_auto,
            base_n_lags=nlag
        )
        
        # Cache the auto-calculated values
        vgm._auto_lag_cache['horizontal'] = (h_nlags, h_lag_dist, h_max_range)
        vgm._auto_lag_cache['downhole'] = (d_nlags, d_lag_dist, d_max_range)
        
        # Update the main parameters for horizontal
        max_range = h_max_range
        logger.info(f"Auto-lags: horizontal={h_nlags} lags @ {h_lag_dist:.1f}m, downhole={d_nlags} lags @ {d_lag_dist:.1f}m")

    # Extract hole ids for downhole variogram
    hole_ids = None
    if actual_hole_col and actual_hole_col in clean_data.columns:
        hole_ids = clean_data[actual_hole_col].astype(str).values
        logger.info(f"Found hole ID column: {actual_hole_col} ({len(set(hole_ids))} unique holes)")

    from_vals = clean_data[actual_from_col].values if actual_from_col and actual_from_col in clean_data.columns else None
    to_vals = clean_data[actual_to_col].values if actual_to_col and actual_to_col in clean_data.columns else None
    
    # 1. Calculate omnidirectional variogram
    update_progress(10, "Calculating omnidirectional variogram...")
    omni = vgm.calculate_omnidirectional(coords, values, sample_weights=sample_weights, n_lags=nlag, max_range=max_range)
    logger.info(f"Omnidirectional variogram calculated: {len(omni)} valid lags")
    update_progress(20, "Omnidirectional variogram complete")

    # 1b. Downhole variogram for nugget estimation
    # IMPORTANT: Use auto-calculated lags for downhole to capture long-range structure (~140-150m)
    update_progress(25, "Calculating downhole variogram...")
    downhole = pd.DataFrame({"distance": [], "gamma": [], "npairs": []})
    
    if hole_ids is not None:
        # Use auto-calculated downhole lags if available
        if auto_lags and 'downhole' in vgm._auto_lag_cache:
            d_nlags, d_lag_dist, d_max_range = vgm._auto_lag_cache['downhole']
            logger.info(f"Using auto-calculated downhole lags: {d_nlags} lags, max_range={d_max_range:.1f}m")
            downhole = vgm.calculate_downhole(
                coords, values, hole_ids, from_vals, to_vals, 
                sample_weights=sample_weights, n_lags=d_nlags, max_range=d_max_range
            )
        else:
            # Default: let calculate_downhole compute its own max_range from composite length
            downhole = vgm.calculate_downhole(
                coords, values, hole_ids, from_vals, to_vals,
                sample_weights=sample_weights, n_lags=nlag, max_range=None
            )
        logger.info("Downhole variogram calculated: %d valid lags", len(downhole))
    else:
        # FALLBACK: If no HOLEID found, compute a vertical variogram as proxy
        # This ensures the Downhole tab always shows something useful
        logger.warning("No HOLEID column detected. Using vertical variogram as downhole proxy.")
        logger.warning("For accurate nugget estimation, ensure your data has a HOLEID/BHID column.")
        
        # Compute tight vertical variogram (dip=90°) with smaller lags for nugget estimation
        # Use smaller lag distance for nugget-focused analysis
        vert_lag_dist = lag_distance * 0.5  # Smaller lags for nugget
        vert_max_range = vert_lag_dist * 15  # ~15 lags max
        vert_n_lags = min(15, nlag)
        
        downhole = vgm.calculate_directional(
            coords, values, sample_weights,
            azimuth_deg=0.0, dip_deg=90.0, cone_tolerance=10.0,  # Tight vertical cone
            n_lags=vert_n_lags, max_range=vert_max_range
        )
        logger.info("Vertical proxy variogram calculated: %d valid lags (proxy for downhole)", len(downhole))
    
    update_progress(35, "Downhole variogram complete")
    
    # 2. Calculate directional variograms
    update_progress(40, "Calculating directional variograms...")
    # Major horizontal direction (N-S, azimuth=0)
    major_az = default_azimuth if default_azimuth is not None else 0.0
    major_dip = default_dip if default_dip is not None else 0.0
    major = vgm.calculate_directional(
        coords, values, sample_weights,
        azimuth_deg=major_az, dip_deg=major_dip, cone_tolerance=default_az_tol,
        n_lags=nlag, max_range=max_range
    )
    logger.info(f"Directional variogram 'major' calculated: {len(major)} valid lags (azimuth={major_az:.1f}°, dip={major_dip:.1f}°)")
    
    # Minor horizontal direction (perpendicular to major, 90° clockwise from major azimuth)
    # CRITICAL: Minor must be orthogonal to Major for proper anisotropy characterization
    minor_az = (major_az + 90.0) % 360.0
    minor = vgm.calculate_directional(
        coords, values, sample_weights,
        azimuth_deg=minor_az, dip_deg=0.0, cone_tolerance=default_az_tol,
        n_lags=nlag, max_range=max_range
    )
    logger.info(f"Directional variogram 'minor' calculated: {len(minor)} valid lags (azimuth={minor_az:.1f}°)")
    
    # Vertical direction (dip=90)
    # Use vertical-specific max range if auto-lags enabled, otherwise use standard max_range
    vert_max_range = max_range
    if auto_lags:
        # Vertical range is typically shorter (stratigraphy thickness)
        # We use calculate_auto_lags with 'vertical' to get a better estimate based on Z extent
        _, _, vert_max_range = calculate_auto_lags(
            coords, direction='vertical',
            base_n_lags=nlag
        )
        logger.info(f"Auto-lags (vertical): max_range={vert_max_range:.1f}m")

    vertical = vgm.calculate_directional(
        coords, values, sample_weights,
        azimuth_deg=0.0, dip_deg=90.0, cone_tolerance=default_dip_tol,
        n_lags=nlag, max_range=vert_max_range
    )
    logger.info(f"Directional variogram 'vertical' calculated: {len(vertical)} valid lags (max_range={vert_max_range:.1f}m)")
    update_progress(55, "Directional variograms complete")

    # 3. Fit models to each direction
    update_progress(60, "Fitting variogram models...")
    fitted_models = {}
    nested_models = {}  # Store nested models separately
    
    # GEOSTATISTICAL CONSTRAINT: Calculate sample variance as reference sill
    # The sill should approximately equal the sample variance in any direction.
    # When directional variograms have few pairs, the fit may be unreliable,
    # so we cap the sill to prevent unrealistic values.
    sample_variance = float(np.nanvar(values))
    logger.info(f"Sample variance: {sample_variance:.3f}")
    
    # First fit omnidirectional to get reference sill (most stable with most pairs)
    omni_sill_reference = None
    if len(omni) >= 3:
        try:
            # Quick fit to get reference sill
            omni_nug, omni_psill, _ = vgm.fit_model(
                omni["distance"].to_numpy(float),
                omni["gamma"].to_numpy(float),
                model_types[0] if model_types else "spherical",
                sill_norm=use_sill_norm
            )
            omni_sill_reference = omni_nug + omni_psill
            logger.info(f"Omnidirectional reference sill: {omni_sill_reference:.3f} (will use as cap for directional fits)")
        except Exception as e:
            logger.warning(f"Could not compute omnidirectional reference sill: {e}")
    
    # Use sample variance as fallback if omni fit failed
    sill_cap = omni_sill_reference if omni_sill_reference is not None else sample_variance
    # Add margin for natural variability (configurable, default 30%)
    sill_multiplier = VARIOGRAM_CONFIG['sill_cap_multiplier']
    sill_cap = sill_cap * sill_multiplier
    logger.info(f"Sill cap for directional fits: {sill_cap:.3f} (multiplier={sill_multiplier})")

    # Track weak directions for metadata and UI warnings
    weak_directions = []

    for direction_name, vg_data in [
        ('downhole', downhole),
        ('omni', omni),
        ('major', major),
        ('minor', minor),
        ('vertical', vertical)
    ]:
        # Check minimum data requirements before fitting
        if len(vg_data) < 3:
            logger.warning(f"Insufficient data for {direction_name} variogram fitting: {len(vg_data)} lags (need >= 3). Skipping.")
            continue

        # ============================================================
        # WEAK DIRECTION GUARDS (Professional Geostatistics Standards)
        # ============================================================
        # Thresholds loaded from config (can be overridden in ~/.geox/config.toml)
        MIN_PAIRS_PER_LAG = VARIOGRAM_CONFIG['min_pairs_per_lag']  # Default: 30
        WEAK_THRESHOLD = VARIOGRAM_CONFIG['weak_threshold']  # Default: 50
        CRITICAL_LAGS = VARIOGRAM_CONFIG['critical_lags_with_pairs']  # Default: 3

        pairs_per_lag = vg_data.get("npairs", pd.Series([0] * len(vg_data)))
        lags_with_pairs = (pairs_per_lag > 0).sum()
        total_pairs = int(pairs_per_lag.sum())
        avg_pairs_per_lag = total_pairs / len(vg_data) if len(vg_data) > 0 else 0

        # Track weak direction status
        is_weak_direction = False
        direction_warning = None

        if lags_with_pairs < CRITICAL_LAGS:
            is_weak_direction = True
            direction_warning = (
                f"CRITICAL: {direction_name} has only {lags_with_pairs} lags with pairs (need ≥{CRITICAL_LAGS}). "
                f"Model fit is statistically unreliable."
            )
            logger.warning(direction_warning)
        elif avg_pairs_per_lag < MIN_PAIRS_PER_LAG:
            is_weak_direction = True
            direction_warning = (
                f"WEAK DIRECTION: {direction_name} avg {avg_pairs_per_lag:.0f} pairs/lag "
                f"(need ≥{MIN_PAIRS_PER_LAG}). Total pairs: {total_pairs}. "
                f"Consider: wider cone tolerance, more data, or inherit sill from omni."
            )
            logger.warning(direction_warning)
        elif avg_pairs_per_lag < WEAK_THRESHOLD:
            direction_warning = (
                f"MARGINAL: {direction_name} avg {avg_pairs_per_lag:.0f} pairs/lag "
                f"(recommended ≥{WEAK_THRESHOLD}). Total pairs: {total_pairs}."
            )
            logger.info(direction_warning)

        # Store weak direction info for metadata
        if direction_warning:
            weak_directions.append({
                'direction': direction_name,
                'avg_pairs_per_lag': avg_pairs_per_lag,
                'total_pairs': total_pairs,
                'lags_with_pairs': lags_with_pairs,
                'is_critical': lags_with_pairs < CRITICAL_LAGS or avg_pairs_per_lag < MIN_PAIRS_PER_LAG,
                'warning': direction_warning
            })

        # Don't apply sill_cap to omnidirectional (it's the reference itself)
        # For weak directions, consider forcing sill inheritance from omni
        dir_sill_cap = None if direction_name == 'omni' else sill_cap

        # PROFESSIONAL GUARD: For critically weak directions, inherit sill from omni
        # This prevents wild sill estimates from sparse data (configurable behavior)
        inherit_for_weak = VARIOGRAM_CONFIG['inherit_omni_sill_for_weak']
        if inherit_for_weak and is_weak_direction and direction_name != 'omni' and omni_sill_reference is not None:
            dir_sill_cap = omni_sill_reference
            logger.info(f"Weak direction guard: {direction_name} sill capped to omni reference ({omni_sill_reference:.3f})")
        
        fitted_models[direction_name] = {}
        for model_type in model_types:
            try:
                # Use nested fitting if n_structures > 1
                if n_structures > 1:
                    nested_model = vgm.fit_nested_model(
                        vg_data["distance"].to_numpy(float),
                        vg_data["gamma"].to_numpy(float),
                        model_type,
                        n_structures=n_structures,
                        sill_norm=use_sill_norm
                    )
                    
                    # Store nested model
                    nested_models[direction_name] = nested_model
                    
                    # Extract primary structure for backward compatibility
                    nugget = nested_model.nugget
                    if nested_model.structures:
                        # Use longest-range structure as primary
                        primary = max(nested_model.structures, key=lambda s: s.range_major)
                        psill = nested_model.total_sill - nugget
                        prange = primary.range_major
                    else:
                        psill = 0.0
                        prange = 100.0
                    
                    params = {
                        'model_type': model_type,
                        'nugget': float(nugget),
                        'sill': float(psill),
                        'range': float(prange),
                        'total_sill': float(nested_model.total_sill),
                        'nested': nested_model.to_dict()  # Include full nested model
                    }
                    
                    logger.info(f"Nested model fitted ({direction_name}, {model_type}): "
                               f"nugget={nugget:.3f}, sill={nested_model.total_sill:.3f}, "
                               f"structures={nested_model.n_structures}")
                else:
                    # Single structure fitting with sill_cap for geostatistical soundness
                    nugget, psill, prange = vgm.fit_model(
                        vg_data["distance"].to_numpy(float),
                        vg_data["gamma"].to_numpy(float),
                        model_type,
                        sill_norm=use_sill_norm,
                        sill_cap=dir_sill_cap
                    )
                    total_sill = nugget + psill
                    
                    # GEOSTATISTICAL VALIDATION: Warn if sill is suspiciously different from reference
                    if omni_sill_reference is not None and direction_name != 'omni':
                        sill_ratio = total_sill / omni_sill_reference
                        if sill_ratio > 1.5 or sill_ratio < 0.5:
                            logger.warning(f"GEOSTATISTICS WARNING: {direction_name} sill ({total_sill:.2f}) is "
                                         f"{sill_ratio:.1f}x the omnidirectional sill ({omni_sill_reference:.2f}). "
                                         f"This may indicate fitting issues due to sparse data.")
                    
                    params = {
                        'model_type': model_type,
                        'nugget': float(nugget),
                        'sill': float(psill),
                        'range': float(prange),
                        'total_sill': float(total_sill)
                    }
                    logger.info(f"Model fitted ({direction_name}, {model_type}): nugget={nugget:.3f}, sill={psill:.3f}, range={prange:.1f}")
                
                fitted_models[direction_name][model_type] = params
                
            except ValueError as e:
                # Insufficient data - skip this model
                logger.warning(f"Failed to fit {model_type} model for {direction_name}: {e}")
            except (RuntimeError, Exception) as e:
                # Numerical or unexpected errors
                logger.warning(f"Failed to fit {model_type} model for {direction_name}: {e}")
    
    # 4. Compile results
    update_progress(90, "Compiling results...")
    
    # AUDIT FIX: Add data lineage tracking (V-001)
    # Compute hash of source data for lineage verification at estimation time
    try:
        from ..geostats.variogram_model import compute_data_hash
        from ..geostats.variogram_gates import add_lineage_metadata
        from datetime import datetime
        
        source_data_hash = compute_data_hash(coords, values, vcol)
        fit_timestamp = datetime.now().isoformat()
        
        # Determine source dataset type
        if 'declust_weight' in clean_data.columns:
            source_dataset_type = "declustered"
        elif 'SAMPLE_COUNT' in clean_data.columns or 'SUPPORT' in clean_data.columns:
            source_dataset_type = "composites"
        else:
            source_dataset_type = "raw_assays"
            
    except ImportError:
        logger.warning("AUDIT: Could not import lineage tracking modules")
        source_data_hash = None
        fit_timestamp = None
        source_dataset_type = None
    
    results = {
        'downhole_variogram': downhole,
        'omni_variogram': omni,
        'major_variogram': major,
        'minor_variogram': minor,
        'vertical_variogram': vertical,
        'fitted_models': fitted_models,
        'nested_models': nested_models,  # Full nested model objects
        'variogram_object': vgm,
        # Include direction azimuths for transparency (CRITICAL for QA/QC)
        'major_azimuth': major_az,
        'minor_azimuth': minor_az,
        'major_dip': major_dip,
        # Include lag configuration for transparency
        'lag_config': {
            'auto_lags': auto_lags,
            'n_structures': n_structures,
            'global_nugget': global_nugget,
            'horizontal': vgm._auto_lag_cache.get('horizontal', (nlag, lag_distance, max_range)),
            'downhole': vgm._auto_lag_cache.get('downhole', None)
        },
        # AUDIT: Lineage metadata for JORC/SAMREC compliance
        'metadata': {
            'source_data_hash': source_data_hash,
            'fit_timestamp': fit_timestamp,
            'source_dataset_type': source_dataset_type,
            'source_data_n_samples': len(clean_data),
            'source_data_n_rows': len(data),  # Original input size
            'variable': vcol,
            'sample_variance': sample_variance,
            # Subsampling tracking for transparency
            'subsampled': len(clean_data) > vgm.max_directional_samples,
            'subsample_size': min(len(clean_data), vgm.max_directional_samples),
            'subsample_seed': vgm.random_state,
            'subsample_fraction': min(1.0, vgm.max_directional_samples / len(clean_data)) if len(clean_data) > 0 else 1.0,
            'max_directional_samples': vgm.max_directional_samples,
            'pair_cap': vgm.pair_cap,
            'random_state': vgm.random_state,
            'is_deterministic': vgm.random_state is not None,
            # Weak direction tracking for UI warnings
            'weak_directions': weak_directions,
            'has_weak_directions': len([w for w in weak_directions if w['is_critical']]) > 0,
        }
    }

    # Log summary of professional features used
    if auto_lags:
        logger.info(f"Auto-lags enabled: horizontal max_range={max_range:.1f}m")
    if n_structures > 1:
        logger.info(f"Nested structures enabled: {n_structures} structures per direction")
    if global_nugget is not None:
        logger.info(f"Global nugget enforced: {global_nugget:.3f}")
    
    logger.info("3D variogram pipeline completed successfully")
    update_progress(100, "Analysis complete")

    return results


def plot_variogram_cloud(
    data: pd.DataFrame,
    xcol: str = "X",
    ycol: str = "Y",
    zcol: str = "Z",
    vcol: str = "Fe",
    max_pairs: int = 2000,
    max_dist: Optional[float] = None,
    max_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate variogram cloud data (distance vs semivariance for sampled pairs).
    
    Parameters
    ----------
    data : DataFrame
        Input data with coordinates and values
    xcol, ycol, zcol : str
        Column names for coordinates
    vcol : str
        Column name for variable to analyze
    max_pairs : int
        Maximum number of pairs to return (default 2000)
    max_dist : float, optional
        Maximum distance for pairs. If None, uses 50% of data extent.
    max_samples : int
        Maximum samples to use before subsampling (default 2000).
        This prevents O(N²) explosion for large datasets.
    
    Returns
    -------
    distances, gamma_values : arrays
        Distance and semivariance for each pair
    """
    n_total = len(data)
    logger.info(f"Generating variogram cloud from {n_total} samples (max {max_pairs} pairs)")
    
    # Filter out rows with NaN values
    required_cols = [xcol, ycol, zcol, vcol]
    clean_data = data[required_cols].dropna()
    n_clean = len(clean_data)
    
    if n_clean == 0:
        raise ValueError("All data contains NaN values. Cannot generate variogram cloud.")
    
    logger.info(f"Valid samples: {n_clean} (dropped {n_total - n_clean} with NaN)")
    
    coords = clean_data[[xcol, ycol, zcol]].values
    values = clean_data[vcol].values
    
    # Use optimized function with subsampling
    from .variogram_functions import _pairwise_variogram
    distances, gamma_values = _pairwise_variogram(
        values, coords, 
        max_pairs=max_pairs,
        max_dist=max_dist,
        max_samples=max_samples
    )
    
    logger.info(f"Variogram cloud generated: {len(distances)} pairs")
    
    return distances, gamma_values
