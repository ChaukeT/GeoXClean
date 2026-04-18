"""
ARBF Estimation Engine — Main Orchestrator.

Implements the complete 10-step ARBF workflow:

Step 1:  Data transforms (normal-score, ILR)
Step 2:  Build orientation field (if use_lva=True)
Step 3:  Create sub-domains and fit local variograms
Step 4:  Assemble and factorise local kernel matrices
Step 5:  Estimate blocks with PUM blending + posterior variance
Step 6:  Adaptive block discretisation
Step 7:  Cross-validation and diagnostics
Step 8:  Change-of-support correction
Step 9:  Resource classification
Step 10: Generate JORC audit record

Each step is individually toggleable.  If disabled, the workflow
skips it and continues.

Usage
-----
>>> estimator = ARBFEstimator(config)
>>> estimator.set_composites(coords, values)
>>> estimator.set_block_model(centroids, block_sizes)
>>> results = estimator.estimate()
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .audit import ARBFAuditRecord, compute_data_hash
from .blending import BlendedResult, _build_group_keys, blend_estimates_fast
from .change_of_support import (
    ChangeOfSupportResult,
    affine_correction,
    within_block_variance,
)
from .classification import (
    ClassificationResult,
    GeometricCriteria,
    VarianceThresholds,
    classify_blocks,
)
from .cross_validation import (
    CVResult,
    ConditionalBiasResult,
    SupportSwathResult,
    SwathData,
    _compute_cv_statistics,
    conditional_bias_diagnostics,
    leave_one_out_cv,
    support_swath_plots,
    swath_plots,
)
from .discretisation import (
    adaptive_discretisation_density,
    compute_block_interior_points,
    get_offsets,
)
from .gpr import (
    _compute_kernel_batch,
    assemble_kernel_matrix,
    build_polynomial_matrix,
    compute_l_inv,
    factorise_and_solve,
    predict_mean,
    predict_mean_and_variance,
    predict_variance,
    solve_weights_from_factor,
)
from .orientation import OrientationField
from .partition import SubDomain, create_subdomains, fit_subdomain_variograms, wendland_c2_weight_batch
from .variogram import LocalVariogramResult
from .transforms import (
    NormalScoreTable,
    detect_already_normal_scored,
    ilr_forward,
    ilr_inverse,
    normal_score_backtransform,
    normal_score_backtransform_mean,
    normal_score_transform,
)
from .utils import clamp_variance, rotation_matrix, scale_matrix

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return wrapper

logger = logging.getLogger(__name__)


# ─── Numba-accelerated spread-aware neighbourhood selection ───────────
# This is the single hottest function in the ARBF engine (~50% of total
# runtime).  The inner loop computes min-separation from every candidate
# to every already-selected sample — O(max_total × n_octants × n_selected).
# In pure Python/numpy this is 4M+ small-array operations per estimation.
# Numba JIT compiles the entire nested loop to native code, eliminating
# all Python object/array overhead.

@njit(cache=True)
def _spread_select_jit(
    diffs: np.ndarray,        # (K, 3) sorted candidate diffs
    dists: np.ndarray,        # (K,) sorted candidate distances
    weights: np.ndarray,      # (K,) candidate weights
    oct_codes: np.ndarray,    # (K,) int8 octant codes
    max_total: int,           # max samples to select
    max_per_octant: int,      # per-octant cap
    weight_scale: float,      # normalisation for weight term
) -> np.ndarray:
    """Greedy spread-aware sample selection (numba JIT).

    Returns array of selected position indices into the sorted arrays.
    """
    K = len(diffs)

    # Per-octant position lists (max K entries each)
    # Numba doesn't support lists of variable-length arrays, so use
    # a flat (8, K) array + count array.
    oct_count = np.zeros(8, dtype=np.int32)
    oct_pos = np.empty((8, K), dtype=np.int64)
    for i in range(K):
        o = oct_codes[i]
        oct_pos[o, oct_count[o]] = i
        oct_count[o] += 1

    # Find occupied octants sorted by nearest sample distance
    n_occupied = 0
    occ_octants = np.empty(8, dtype=np.int32)
    occ_sort_key = np.empty(8, dtype=np.float64)
    for o in range(8):
        if oct_count[o] > 0:
            occ_octants[n_occupied] = o
            occ_sort_key[n_occupied] = dists[oct_pos[o, 0]]
            n_occupied += 1

    # Simple insertion sort on occupied octants by distance (max 8)
    for i in range(1, n_occupied):
        j = i
        while j > 0 and occ_sort_key[j - 1] > occ_sort_key[j]:
            occ_sort_key[j], occ_sort_key[j - 1] = occ_sort_key[j - 1], occ_sort_key[j]
            occ_octants[j], occ_octants[j - 1] = occ_octants[j - 1], occ_octants[j]
            j -= 1

    # Selected buffer
    sel = np.empty(max_total, dtype=np.int64)
    sel_diffs = np.empty((max_total, 3), dtype=np.float64)
    n_sel = 0
    selected_counts = np.zeros(8, dtype=np.int32)
    cursors = np.zeros(8, dtype=np.int32)

    # First pass: one sample per occupied octant
    for oi in range(n_occupied):
        if n_sel >= max_total:
            break
        o = occ_octants[oi]
        pos = oct_pos[o, 0]
        sel[n_sel] = pos
        sel_diffs[n_sel, 0] = diffs[pos, 0]
        sel_diffs[n_sel, 1] = diffs[pos, 1]
        sel_diffs[n_sel, 2] = diffs[pos, 2]
        n_sel += 1
        selected_counts[o] += 1
        cursors[o] = 1

    # Fill remaining with spread-aware greedy selection
    while n_sel < max_total:
        best_pos = -1
        best_octant = -1
        best_score = -1e30

        for oi in range(n_occupied):
            o = occ_octants[oi]
            if selected_counts[o] >= max_per_octant:
                continue
            c = cursors[o]
            if c >= oct_count[o]:
                continue
            pos = oct_pos[o, c]

            # Min squared distance to all selected
            min_sq = 1e30
            for s in range(n_sel):
                dx = sel_diffs[s, 0] - diffs[pos, 0]
                dy = sel_diffs[s, 1] - diffs[pos, 1]
                dz = sel_diffs[s, 2] - diffs[pos, 2]
                sq = dx * dx + dy * dy + dz * dz
                if sq < min_sq:
                    min_sq = sq

            min_sep = min_sq ** 0.5 if n_sel > 0 else 0.0

            score = (
                min_sep
                - 0.20 * dists[pos]
                - 0.15 * selected_counts[o]
                + 0.15 * weights[pos] / weight_scale
            )
            if score > best_score:
                best_score = score
                best_pos = pos
                best_octant = o

        if best_pos < 0:
            break

        sel[n_sel] = best_pos
        sel_diffs[n_sel, 0] = diffs[best_pos, 0]
        sel_diffs[n_sel, 1] = diffs[best_pos, 1]
        sel_diffs[n_sel, 2] = diffs[best_pos, 2]
        n_sel += 1
        selected_counts[best_octant] += 1
        cursors[best_octant] += 1

    return sel[:n_sel]


@dataclass
class ARBFResult:
    """Typed result from ARBF estimation (replaces plain dict).

    Prevents typos in key access and enables IDE autocompletion.
    """

    grades: np.ndarray
    variances: np.ndarray
    classifications: np.ndarray
    classification_names: np.ndarray
    stitching_variance: Optional[np.ndarray] = None
    total_blending_variance: Optional[np.ndarray] = None
    cv_result: Optional[Any] = None
    swath_data: Optional[Dict] = None
    support_swath_data: Optional[Any] = None
    conditional_bias_result: Optional[Any] = None
    cos_result: Optional[Any] = None
    classification_result: Optional[Any] = None
    audit_record: Optional[Any] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Dict-like access for backwards compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return [f.name for f in self.__dataclass_fields__.values()]


class ARBFEstimator:
    """Main ARBF orchestrator.

    Parameters
    ----------
    config : dict
        Configuration dictionary with estimation parameters.
        See ``__init__`` for full list of supported keys.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}

        # Kernel parameters
        self.kernel_type: str = cfg.get("kernel_type", "spheroidal")
        self.alpha: float = cfg.get("alpha", 1.0)
        self.drift_type: str = cfg.get("drift_type", "auto")
        self.nugget: float = cfg.get("nugget", 0.0)
        self.accuracy: float = cfg.get("accuracy", 1e-6)
        self.weight_sanity_limit: float = cfg.get("weight_sanity_limit", 100.0)
        self.max_auto_regularization_fraction: float = cfg.get(
            "max_auto_regularization_fraction", 0.05,
        )
        self.auto_drift_cv_max_samples: int = cfg.get("auto_drift_cv_max_samples", 80)
        self.auto_drift_min_rmse_improvement: float = cfg.get(
            "auto_drift_min_rmse_improvement", 0.02,
        )
        self.auto_drift_min_slope_improvement: float = cfg.get(
            "auto_drift_min_slope_improvement", 0.05,
        )
        self.auto_drift_max_slope_deviation: float = cfg.get(
            "auto_drift_max_slope_deviation", 0.20,
        )

        # Sub-domain parameters
        self.n_subdomains: int = cfg.get("n_subdomains", 0)  # 0 = auto
        self.subdomain_method: str = cfg.get("subdomain_method", "kmeans")
        self.subdomain_centres: Optional[np.ndarray] = cfg.get("subdomain_centres")
        self.subdomain_radii: Optional[np.ndarray] = cfg.get("subdomain_radii")
        self.max_samples: int = cfg.get("max_samples", 300)
        self.min_samples: int = cfg.get("min_samples", 4)
        self.overlap_factor: float = cfg.get("overlap_factor", 1.5)
        self.estimation_mode: str = cfg.get(
            "estimation_mode", "local_neighbourhood_gpr",
        ).lower()
        self.local_search_radii: Tuple[float, float, float] = tuple(
            float(x) for x in cfg.get("local_search_radii", (0.75, 1.5, 3.0))
        )
        self.balanced_neighbourhood_selection: bool = bool(
            cfg.get("balanced_neighbourhood_selection", True),
        )
        self.search_min_octants: int = int(cfg.get("search_min_octants", 3))
        self.search_min_octants_linear: int = int(
            cfg.get("search_min_octants_linear", 4),
        )
        self.max_samples_per_octant: int = int(
            cfg.get("max_samples_per_octant", 4),
        )

        # Single-domain (no PUM) threshold.  When N < pum_threshold
        # (or n_subdomains == 1), bypass sub-domain decomposition and
        # build ONE global kernel matrix for all samples.  Eliminates
        # between-model variance, blending artifacts, and sub-domain
        # boundary effects.  A 3000×3000 Cholesky takes < 1 second.
        self.pum_threshold: int = cfg.get("pum_threshold", 3000)

        # Anisotropy
        self.azimuth: float = cfg.get("azimuth", 0.0)
        self.dip: float = cfg.get("dip", 0.0)
        self.pitch: float = cfg.get("pitch", 0.0)
        self.range_max: float = cfg.get("range_max", 100.0)
        self.range_mid: float = cfg.get("range_mid", 100.0)
        self.range_min: float = cfg.get("range_min", 100.0)

        # LVA
        self.use_lva: bool = cfg.get("use_lva", False)
        self.lva_source: str = cfg.get("lva_source", "data")

        # Transforms
        self.use_normal_score: bool = cfg.get("use_normal_score", False)
        self.use_ilr: bool = cfg.get("use_ilr", False)
        self.ilr_components: Optional[List[str]] = cfg.get("ilr_components")

        # Variogram mode: "local" fits per-subdomain, "global" uses user's
        # imported parameters for all subdomains, "hybrid" fits locally but
        # falls back to global when nugget > sill (poor local fit).
        self.variogram_mode: str = cfg.get("variogram_mode", "hybrid")
        # User-supplied sill (partial sill, excluding nugget)
        self.sill: float = cfg.get("sill", 0.0)

        # Change of support
        self.change_of_support: bool = cfg.get("change_of_support", True)
        self.change_of_support_mode: str = cfg.get(
            "change_of_support_mode", "discretized",
        ).lower()
        self.prefilter_blocks: bool = cfg.get("prefilter_blocks", False)
        self.clip_to_drill_footprint: bool = bool(
            cfg.get("clip_to_drill_footprint", False),
        )
        self.footprint_buffer_ranges: float = float(
            cfg.get("footprint_buffer_ranges", 1.5),
        )
        # Estimation should preserve the reporting model by default and let
        # the quality gate / classification logic describe low-confidence
        # areas. Hard-masking uninformed NS blocks is an explicit opt-in.
        self.mask_uninformed_ns_blocks: bool = bool(
            cfg.get("mask_uninformed_ns_blocks", False),
        )

        # Classification
        self.classification_thresholds: Optional[Dict] = cfg.get(
            "classification_thresholds",
        )
        self.geometric_criteria: Optional[Dict] = cfg.get("geometric_criteria")

        # Discretisation
        self.discretisation_mode: str = cfg.get("discretisation", "fixed")
        self.discretisation_density: int = cfg.get("discretisation_density", 27)
        # Radius (in same units as coordinates) for computing per-block local
        # means in the affine CoS correction.  0 = use global declustered mean.
        self.local_mean_radius: float = cfg.get("local_mean_radius", 0.0)

        # Cell size for Deutsch (1989) cell declustering when computing the
        # CoS global mean.  0 = auto-select from median sample spacing.
        # Ignored when set_declustering_weights() has been called.
        self.decluster_cell_size: float = cfg.get("decluster_cell_size", 0.0)

        # Rotation convention used for azimuth/dip/pitch angles.
        # "geox" (default): R = Ry(pitch) @ Rx(dip) @ Rz(-azimuth)
        # "leapfrog":       dip_direction / dip / pitch (converted on load)
        # "datamine":       bearing / dip / plunge (converted on load)
        # This is documentation only — angles are already pre-converted
        # by the UI layer.  Stored for audit record transparency.
        self.rotation_convention: str = cfg.get("rotation_convention", "geox")
        self.domain_policy: str = cfg.get("domain_policy", "warn").lower()

        # When True, use geodesic path integration (Eq. 4.4 in the ARBF
        # paper) for pairs where the local orientation changes > 15°.
        # Required for strongly folded deposits (Bushveld, Wits Basin).
        # Slower than linear LVA (~10× for N=1000) — disable for flat or
        # gently folded deposits.
        self.use_geodesic: bool = cfg.get("use_geodesic", False)

        # Execution — parallel=True by default so sub-domain factorisations
        # run concurrently (BLAS releases the GIL; threads give real speedup).
        self.parallel: bool = cfg.get("parallel", True)
        self.n_workers: int = cfg.get("n_workers", 4)
        self.verbose: bool = cfg.get("verbose", True)

        # Clipping
        self.clip_min: Optional[float] = cfg.get("clip_min")
        self.clip_max: Optional[float] = cfg.get("clip_max")

        # Cross-validation
        self.run_cv: bool = cfg.get("run_cv", True)
        self.cv_max_samples: int = cfg.get("cv_max_samples", 500)
        self.cv_mode: str = cfg.get("cv_mode", "spatial_kfold")
        self.cv_folds: int = cfg.get("cv_folds", 5)
        self.allow_cv_fallback: bool = bool(cfg.get("allow_cv_fallback", False))

        # Operator / audit
        self.operator: str = cfg.get("operator", "")

        # Seed for determinism
        self.seed: int = cfg.get("seed", 42)

        # Internal state
        self._composite_coords: Optional[np.ndarray] = None
        self._composite_values: Optional[np.ndarray] = None
        self._block_centroids: Optional[np.ndarray] = None
        self._block_sizes: Optional[np.ndarray] = None
        self._orientation_field: Optional[OrientationField] = None
        self._subdomains: Optional[List[SubDomain]] = None
        self._progress_callback: Optional[Callable[[int, str], None]] = None

        # Domain support (hard geological boundaries)
        self._composite_domains: Optional[np.ndarray] = None
        self._block_domains: Optional[np.ndarray] = None

        # Spatial declustering weights (one per composite, summing to 1).
        # When set, these override the arithmetic mean in the CoS correction.
        self._declustering_weights: Optional[np.ndarray] = None

        # Cache for within_block_variance — same result for every ILR component
        # since variogram params and block dims don't change between components.
        self._sigma_w_sq_cache: Dict[Tuple[Any, ...], float] = {}
        self._stabilization_events: List[Dict[str, Any]] = []
        self._working_values: Optional[np.ndarray] = None
        self._estimation_geometry_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._effective_drift_type: str = self.drift_type
        self._trend_diagnostics: Dict[str, Any] = {}
        self._domain_diagnostics: Dict[str, Any] = {}
        self._local_drift_fallbacks: int = 0
        self._search_selection_weights: Optional[np.ndarray] = None
        self._cv_execution_mode: str = "not_run"
        self._reporting_block_mask: Optional[np.ndarray] = None
        self._classification_threshold_source: str = "unset"
        self._classification_threshold_values: Optional[Tuple[float, float, float]] = None
        self._blocks_needing_knn_fallback: Optional[np.ndarray] = None

        if self.change_of_support_mode not in {"discretized", "affine_legacy"}:
            raise ValueError(
                "change_of_support_mode must be 'discretized' or 'affine_legacy'",
            )
        if self.estimation_mode not in {"local_neighbourhood_gpr", "pum_legacy"}:
            raise ValueError(
                "estimation_mode must be 'local_neighbourhood_gpr' or 'pum_legacy'",
            )
        if self.domain_policy not in {"ignore", "warn", "require"}:
            raise ValueError("domain_policy must be 'ignore', 'warn', or 'require'")
        if len(self.local_search_radii) != 3:
            raise ValueError("local_search_radii must contain three search radii")
        if any(r <= 0.0 for r in self.local_search_radii):
            raise ValueError("local_search_radii values must be positive")
        if any(
            self.local_search_radii[i] >= self.local_search_radii[i + 1]
            for i in range(len(self.local_search_radii) - 1)
        ):
            raise ValueError("local_search_radii must be strictly increasing")
        if self.search_min_octants < 1 or self.search_min_octants_linear < 1:
            raise ValueError("search_min_octants values must be positive")
        if self.footprint_buffer_ranges <= 0.0:
            raise ValueError("footprint_buffer_ranges must be positive")
        if self.search_min_octants > 8 or self.search_min_octants_linear > 8:
            raise ValueError("search_min_octants values cannot exceed 8")
        if self.max_samples_per_octant < 0:
            raise ValueError("max_samples_per_octant must be >= 0")
        if self.auto_drift_max_slope_deviation < 0.0:
            raise ValueError("auto_drift_max_slope_deviation must be non-negative")

    # ------------------------------------------------------------------
    # Data setters
    # ------------------------------------------------------------------

    def set_composites(
        self,
        coords: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Set composite sample data.

        Parameters
        ----------
        coords : np.ndarray
            (N, 3) sample coordinates.
        values : np.ndarray
            (N,) sample values.
        """
        self._composite_coords = np.asarray(coords, dtype=np.float64)
        self._composite_values = np.asarray(values, dtype=np.float64).ravel()
        if self._composite_coords.shape[0] != len(self._composite_values):
            raise ValueError(
                f"Coordinate rows ({self._composite_coords.shape[0]}) != "
                f"value count ({len(self._composite_values)})"
            )

    def set_block_model(
        self,
        centroids: np.ndarray,
        block_sizes: np.ndarray,
    ) -> None:
        """Set block model geometry.

        Parameters
        ----------
        centroids : np.ndarray
            (B, 3) block centroids.
        block_sizes : np.ndarray
            (3,) uniform block sizes or (B, 3) per-block sizes.
        """
        self._block_centroids = np.asarray(centroids, dtype=np.float64)
        self._block_sizes = np.asarray(block_sizes, dtype=np.float64)

    def set_declustering_weights(self, weights: np.ndarray) -> None:
        """Set pre-computed spatial declustering weights for the CoS mean.

        The Competent Person may supply weights from any accepted declustering
        method (cell declustering, polygonal, kriging-based).  These weights
        override the engine's internal cell-declustering calculation and must
        sum to 1.0.  Weights must align 1-to-1 with the composites passed to
        ``set_composites()``.

        Parameters
        ----------
        weights : np.ndarray
            (N,) positive weights that sum to 1.0.
        """
        weights = np.asarray(weights, dtype=np.float64).ravel()
        w_sum = float(np.sum(weights))
        if abs(w_sum - 1.0) > 1e-4:
            logger.warning(
                "Declustering weights sum to %.6f (expected 1.0). "
                "Normalising automatically.",
                w_sum,
            )
            weights = weights / w_sum
        self._declustering_weights = weights

    def set_domains(
        self,
        composite_domains: np.ndarray,
        block_domains: Optional[np.ndarray] = None,
    ) -> None:
        """Set geological domain labels for hard-boundary estimation.

        When domain labels are set, each block is estimated ONLY from
        composites within the same domain.  High-grade mineralisation in
        one domain cannot smear across a fault or lithological boundary
        into adjacent waste rock.

        Parameters
        ----------
        composite_domains : np.ndarray
            (N,) integer or string domain label for each composite.
        block_domains : np.ndarray, optional
            (B,) integer or string domain label for each block.  If None,
            blocks are assigned to their nearest composite's domain using
            a KD-tree lookup.
        """
        self._composite_domains = np.asarray(composite_domains).ravel()
        if block_domains is not None:
            self._block_domains = np.asarray(block_domains).ravel()
        else:
            self._block_domains = None  # resolved lazily in estimate()

    def set_orientation_field(self, field: OrientationField) -> None:
        """Set a pre-computed orientation field for LVA."""
        self._orientation_field = field

    def set_progress_callback(
        self, callback: Callable[[int, str], None],
    ) -> None:
        """Set progress callback for UI integration."""
        self._progress_callback = callback

    # ------------------------------------------------------------------
    # Main estimation workflow
    # ------------------------------------------------------------------

    def estimate(self) -> ARBFResult:
        """Execute the full 10-step ARBF workflow.

        Returns
        -------
        ARBFResult
            Typed result with grades, variances, classifications, etc.
            Supports dict-like access for backwards compatibility.
        """
        t_start = time.time()

        # ---- Dedicated ARBF run log file ----
        # Writes a standalone log for each run, easy to inspect.
        _arbf_fh = None
        try:
            import os
            from pathlib import Path
            log_dir = Path(os.getenv("LOCALAPPDATA", ".")) / "GeoX"
            log_dir.mkdir(parents=True, exist_ok=True)
            arbf_log_path = log_dir / "arbf_run.log"
            _arbf_fh = logging.FileHandler(
                arbf_log_path, mode="w", encoding="utf-8",
            )
            _arbf_fh.setLevel(logging.DEBUG)
            _arbf_fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            ))
            logger.addHandler(_arbf_fh)
            logger.info("ARBF run log: %s", arbf_log_path)
        except Exception:
            pass  # Non-critical — main logging still works

        self._validate_inputs()
        coords = self._composite_coords.copy()
        values = self._composite_values.copy()
        B = self._block_centroids.shape[0]
        self._working_values = values.copy()
        self._estimation_geometry_stats = None
        self._effective_drift_type = self.drift_type.lower()
        self._trend_diagnostics = {}
        self._domain_diagnostics = {}
        self._local_drift_fallbacks = 0
        self._cv_execution_mode = "not_run"
        self._reporting_block_mask = None
        self._classification_threshold_source = "unset"
        self._classification_threshold_values = None
        self._blocks_needing_knn_fallback = None
        self._search_selection_weights = (
            np.asarray(self._declustering_weights, dtype=np.float64)
            if self._declustering_weights is not None
            and len(self._declustering_weights) == len(values)
            else None
        )
        if self._search_selection_weights is None and len(values) > 0:
            _, self._search_selection_weights = self._cell_decluster(
                coords,
                values,
                self.decluster_cell_size,
            )

        # Reset per-run caches so repeated estimate() calls do not reuse
        # stale support or stabilization diagnostics.
        self._sigma_w_sq_cache = {}
        self._stabilization_events = []

        self._progress(0, "Starting ARBF estimation")

        # ============================================================
        # CONFIGURATION DUMP — log every input parameter
        # ============================================================
        logger.info("=" * 72)
        logger.info("ARBF ESTIMATION — FULL CONFIGURATION LOG")
        logger.info("=" * 72)
        logger.info("INPUT DATA:")
        logger.info("  Composites       : N = %d", len(values))
        logger.info("  Composite range  : min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                     float(np.min(values)), float(np.max(values)),
                     float(np.mean(values)), float(np.std(values)))
        logger.info("  Composite var    : %.6f", float(np.var(values)))
        logger.info("  Coord extent X   : [%.1f, %.1f] (span=%.1f)",
                     float(np.min(coords[:, 0])), float(np.max(coords[:, 0])),
                     float(np.max(coords[:, 0]) - np.min(coords[:, 0])))
        logger.info("  Coord extent Y   : [%.1f, %.1f] (span=%.1f)",
                     float(np.min(coords[:, 1])), float(np.max(coords[:, 1])),
                     float(np.max(coords[:, 1]) - np.min(coords[:, 1])))
        logger.info("  Coord extent Z   : [%.1f, %.1f] (span=%.1f)",
                     float(np.min(coords[:, 2])), float(np.max(coords[:, 2])),
                     float(np.max(coords[:, 2]) - np.min(coords[:, 2])))
        logger.info("  Block model      : B = %d blocks", B)
        block_sizes = self._block_sizes
        if block_sizes.ndim == 1:
            logger.info("  Block sizes      : [%.1f, %.1f, %.1f]",
                         block_sizes[0], block_sizes[1], block_sizes[2])
        else:
            logger.info("  Block sizes      : per-block, first=[%.1f, %.1f, %.1f]",
                         block_sizes[0, 0], block_sizes[0, 1], block_sizes[0, 2])
        logger.info("KERNEL PARAMETERS (user input):")
        logger.info("  kernel_type      : %s", self.kernel_type)
        logger.info("  alpha            : %.4f", self.alpha)
        logger.info("  sill (partial)   : %.6f", self.sill)
        logger.info("  nugget           : %.6f", self.nugget)
        total_input = self.sill + self.nugget
        nug_ratio_input = self.nugget / total_input if total_input > 0 else 0
        logger.info("  total sill (C0+C): %.6f", total_input)
        logger.info("  nugget ratio     : %.1f%% (nugget/total)",
                     nug_ratio_input * 100)
        logger.info("  accuracy         : %.2e", self.accuracy)
        logger.info("  weight_limit     : %.1f", self.weight_sanity_limit)
        logger.info("  max_auto_reg_frac: %.4f", self.max_auto_regularization_fraction)
        logger.info("  drift_type       : %s", self.drift_type)
        logger.info("  auto_drift_cv_n  : %d", self.auto_drift_cv_max_samples)
        logger.info("  auto_drift_slope : ±%.2f", self.auto_drift_max_slope_deviation)
        logger.info("ANISOTROPY:")
        logger.info("  azimuth          : %.1f°", self.azimuth)
        logger.info("  dip              : %.1f°", self.dip)
        logger.info("  pitch            : %.1f°", self.pitch)
        logger.info("  range_max        : %.1f", self.range_max)
        logger.info("  range_mid        : %.1f", self.range_mid)
        logger.info("  range_min        : %.1f", self.range_min)
        logger.info("ESTIMATION:")
        logger.info("  estimation_mode  : %s", self.estimation_mode)
        logger.info("  variogram_mode   : %s", self.variogram_mode)
        logger.info("  n_subdomains     : %d (0=auto)", self.n_subdomains)
        logger.info("  subdomain_method : %s", self.subdomain_method)
        logger.info("  max_samples      : %d", self.max_samples)
        logger.info("  min_samples      : %d", self.min_samples)
        logger.info("  overlap_factor   : %.2f", self.overlap_factor)
        logger.info("  pum_threshold    : %d", self.pum_threshold)
        logger.info("  search_radii     : %.2f / %.2f / %.2f",
                    self.local_search_radii[0],
                    self.local_search_radii[1],
                    self.local_search_radii[2])
        logger.info("  balanced_search  : %s", self.balanced_neighbourhood_selection)
        logger.info("  min_octants      : %d (linear=%d)",
                    self.search_min_octants, self.search_min_octants_linear)
        logger.info("  max/octant       : %d (0=unlimited)", self.max_samples_per_octant)
        logger.info("TRANSFORMS:")
        logger.info("  normal_score     : %s", self.use_normal_score)
        logger.info("  ILR              : %s", self.use_ilr)
        logger.info("CORRECTIONS:")
        logger.info("  change_of_support: %s", self.change_of_support)
        logger.info("  cos_mode         : %s", self.change_of_support_mode)
        logger.info("  prefilter_blocks : %s", self.prefilter_blocks)
        logger.info("  clip_min         : %s", self.clip_min)
        logger.info("  clip_max         : %s", self.clip_max)
        logger.info("VALIDATION:")
        logger.info("  run_cv           : %s", self.run_cv)
        logger.info("  cv_mode          : %s", self.cv_mode)
        logger.info("  cv_folds         : %d", self.cv_folds)
        logger.info("  cv_max_samples   : %d", self.cv_max_samples)
        logger.info("  allow_cv_fallback: %s", self.allow_cv_fallback)
        logger.info("  discretisation   : %s (density=%d)",
                     self.discretisation_mode, self.discretisation_density)
        logger.info("  use_lva          : %s (source=%s)", self.use_lva, self.lva_source)
        logger.info("  use_geodesic     : %s", self.use_geodesic)
        logger.info("  seed             : %d", self.seed)
        logger.info("COMPLIANCE:")
        logger.info("  rotation_convention : %s", self.rotation_convention)
        logger.info("  domain_policy       : %s", self.domain_policy)
        logger.info("  decluster_cell_size : %.1f (0=auto)", self.decluster_cell_size)
        logger.info("  declustering_weights: %s",
                     "user-supplied" if self._declustering_weights is not None
                     else "auto (cell declustering)")
        self._domain_diagnostics = self._summarize_domain_state()
        logger.info(
            "  geological_domains  : %d (enforced=%s, block_assignment=%s)",
            self._domain_diagnostics["n_geological_domains"],
            self._domain_diagnostics["domains_enforced"],
            self._domain_diagnostics["block_domain_assignment"],
        )
        logger.info("=" * 72)

        # ---- Domain pre-processing: hard geological boundaries ----
        # When domain labels are set, the estimator loops over unique domains
        # and estimates each in isolation — composites from domain A cannot
        # influence blocks in domain B, preventing grade smearing across
        # faults, lithological contacts, and oxide/sulphide boundaries.
        if self._composite_domains is not None:
            return self._estimate_with_domains()

        # ---- Step 0: Coordinate overlap check ----
        # Detect coordinate mismatch (e.g. UTM composites vs local block
        # model) that puts ALL blocks outside all subdomain radii.
        block_centroid = np.mean(self._block_centroids, axis=0)
        comp_centroid = np.mean(coords, axis=0)
        shift = block_centroid - comp_centroid
        shift_mag = np.linalg.norm(shift)
        comp_extent = np.max(np.max(coords, axis=0) - np.min(coords, axis=0))

        if shift_mag > comp_extent * 2.0:
            logger.warning(
                "COORDINATE MISMATCH: Block centroid (%.1f, %.1f, %.1f) "
                "is %.1fm from composite centroid (%.1f, %.1f, %.1f). "
                "Composite extent is %.1fm. Auto-shifting blocks to "
                "align with composites.",
                *block_centroid, shift_mag, *comp_centroid, comp_extent,
            )
            self._block_centroids = self._block_centroids - shift
        elif shift_mag > comp_extent * 0.5:
            logger.info(
                "Block-composite centroid offset: %.1fm (extent: %.1fm). "
                "Blocks partially overlap composites.",
                shift_mag, comp_extent,
            )

        # ---- Step 1: Data transforms ----
        ns_table = None
        ilr_kappa = None
        ilr_n_components = None
        _ilr_all: Optional[np.ndarray] = None  # (N, D-1) all ILR coords
        if self.use_ilr and values.ndim == 2 and values.shape[1] > 1:
            self._progress(3, "Step 1: ILR forward transform")
            ilr_kappa = float(np.sum(values[0]))
            ilr_n_components = values.shape[1]
            _ilr_all = ilr_forward(values, kappa=ilr_kappa)   # (N, D-1)
            logger.info(
                "ILR forward: %d components → %d ILR coords (kappa=%.1f)",
                ilr_n_components, _ilr_all.shape[1], ilr_kappa,
            )
            # Start the main pipeline with component 0.  Remaining components
            # are estimated after Step 5 using the stored Cholesky factors
            # (no re-factorisation required — only the RHS changes).
            values = _ilr_all[:, 0].copy()
            if _ilr_all.shape[1] > 1:
                logger.info(
                    "ILR multi-component: will estimate all %d coordinates "
                    "independently post-Step-5 via stored Cholesky factors.",
                    _ilr_all.shape[1],
                )
            else:
                logger.info("ILR: single coordinate (D=2 composition).")
        ns_table = None
        if self.use_normal_score:
            self._progress(5, "Step 1: Normal-score transform")
            # Guard: detect if values are already normal-scored to prevent
            # double-transform.  If the user pre-normalised their data AND
            # set use_normal_score=True the engine would map NS values to
            # new NS values, producing a garbage back-transform table.
            if detect_already_normal_scored(values):
                logger.warning(
                    "DOUBLE-TRANSFORM GUARD: input values appear to already "
                    "be normal-score transformed (mean=%.3f, std=%.3f, "
                    "min=%.3f, max=%.3f).  Skipping forward transform to "
                    "prevent double-normalisation.  Pass raw (original) "
                    "grades when use_normal_score=True.",
                    float(np.mean(values)), float(np.std(values)),
                    float(np.min(values)), float(np.max(values)),
                )
                # Build an identity mapping table so back-transform is a
                # no-op (values are already in NS space).  The table maps
                # each NS value to itself — back-transform returns NS
                # estimates directly, which is correct if the user intended
                # to work entirely in NS space.
                sorted_idx = np.argsort(values)
                ns_table = NormalScoreTable(
                    original_sorted=values[sorted_idx].copy(),
                    normal_scores_sorted=values[sorted_idx].copy(),
                    data_min=float(np.min(values)),
                    data_max=float(np.max(values)),
                )
                logger.warning(
                    "Identity NS table built.  Back-transform will return "
                    "NS-space estimates.  To get original-scale grades, "
                    "pass RAW grades (not pre-normalised) to set_composites()."
                )
            else:
                values, ns_table = normal_score_transform(
                    values, seed=self.seed, coords=coords,
                )
                logger.info("Normal-score transform applied (spatial tie-breaking)")

        # ---- Step 2: Build orientation field ----
        if self.use_lva and self._orientation_field is None:
            self._progress(10, "Step 2: Building orientation field")
            self._orientation_field = self._build_orientation_field(coords, values)
            logger.info("Orientation field constructed from %s", self.lva_source)

        if self.accuracy < 0.0:
            raise ValueError("accuracy must be non-negative")

        # Build user's global variogram parameters for "global" / "hybrid" modes.
        # sill comes from config (user's imported variogram or data variance).
        user_sill = self.sill if self.sill > 0 else float(np.var(values))
        user_nugget = self.nugget
        user_range = self.range_max
        user_alpha = self.alpha

        # B11 fix: if sill was auto-computed (user set 0) but nugget is in
        # original data units, the two are on different scales.  Rescale
        # nugget so the nugget ratio is preserved against the data variance.
        if self.sill <= 0 and self.nugget > 0:
            original_total = self.nugget + float(np.var(self._composite_values))
            if original_total > 0:
                nugget_ratio = self.nugget / original_total
                user_nugget = float(np.var(values)) * nugget_ratio
                logger.info(
                    "B11: Auto-sill active, rescaling nugget to match: "
                    "original_nugget=%.4f → rescaled=%.4f (ratio=%.4f)",
                    self.nugget, user_nugget, nugget_ratio,
                )

        # Safeguard: detect sill << data variance (variogram mis-specification).
        # A properly fitted variogram should have total_sill ≈ data_variance.
        # When sill is orders of magnitude smaller, the kernel matrix cannot
        # explain the data variation, producing ill-conditioned weights and
        # wildly wrong estimates.
        data_var = float(np.var(values))
        total_sill = user_sill + user_nugget
        if total_sill > 0 and data_var > 0 and not self.use_normal_score:
            sill_ratio = total_sill / data_var
            if sill_ratio < 0.01:
                logger.warning(
                    "VARIOGRAM MISMATCH: total sill (%.4f) is only %.2f%% of "
                    "data variance (%.4f).  The kernel cannot explain the data "
                    "variation — estimates will be unreliable.  "
                    "FIX: re-fit the variogram (total sill should ≈ data variance), "
                    "or enable Normal Score transform to auto-rescale.",
                    total_sill, sill_ratio * 100, data_var,
                )
            elif sill_ratio < 0.1:
                logger.warning(
                    "VARIOGRAM WARNING: total sill (%.4f) is %.1f%% of "
                    "data variance (%.4f).  Consider re-fitting the variogram "
                    "or enabling Normal Score transform.",
                    total_sill, sill_ratio * 100, data_var,
                )

        # Safeguard: when nugget > sill the variogram model says most
        # variance is random (not spatially structured).  This produces
        # near-zero CV R² and extreme smoothing.  Warn and optionally
        # swap so spatial structure is preserved.
        if user_nugget > user_sill > 0:
            nugget_ratio = user_nugget / (user_nugget + user_sill)
            # Theoretical max LOO-CV R² ≈ (1 - nugget_ratio)^2
            max_r2 = (1.0 - nugget_ratio) ** 2
            logger.warning(
                "VARIOGRAM WARNING: nugget (%.4f) > sill (%.4f). "
                "Nugget accounts for %.0f%% of total variance → "
                "theoretical max LOO-CV R² ≈ %.2f. "
                "Consider re-fitting the variogram or increasing alpha "
                "to ≥ 1.5 for improved smoothness.",
                user_nugget, user_sill,
                nugget_ratio * 100, max_r2,
            )

        # ---- Normal-score variogram handling ----
        # The NS transform is a VALUE-DOMAIN operation: it changes the grade
        # distribution to standard normal but leaves spatial positions unchanged.
        # The variogram RANGE (a spatial property) is therefore preserved.
        # However, the SHAPE and SILL of the variogram in NS space can differ
        # substantially from the raw variogram — especially for skewed data
        # (Au, Cu) where the raw variogram may reflect outlier-driven variance.
        #
        # Correct approach when use_normal_score is active:
        #   • If variogram_mode == "global": force "hybrid" so variograms are
        #     re-fitted on the NS data.  The user's range_max/azimuth/dip are
        #     preserved as starting parameters; only sill/nugget are re-fitted.
        #   • If variogram_mode == "local" or "hybrid": already re-fits on
        #     whatever values are passed (which are now NS-transformed).
        #   • The user_sill/nugget are set to the NS data variance/0 as a
        #     fallback when local fitting fails (hybrid mode).
        if self.use_normal_score and self.sill > 0:
            ns_var = float(np.var(values))  # ≈ 1.0 after transform
            if self.variogram_mode == "global":
                logger.warning(
                    "Normal-score transform is active but variogram_mode='global'. "
                    "The user's sill=%.4f is in ORIGINAL data units; applying it "
                    "directly to NS data (variance≈%.4f) will mis-scale the GPR. "
                    "Switching variogram_mode to 'hybrid' so sill/nugget are "
                    "re-fitted on the NS-transformed data.  "
                    "Range (%.1f) and anisotropy are preserved.",
                    self.sill, ns_var, self.range_max,
                )
                self.variogram_mode = "hybrid"
            # Set user_sill/nugget to NS-space values as the global fallback
            # used by hybrid mode when local fitting fails.
            original_total = user_sill + user_nugget
            if original_total > 0:
                nugget_ratio = user_nugget / original_total
                user_sill = ns_var * (1.0 - nugget_ratio)
                user_nugget = ns_var * nugget_ratio
                logger.info(
                    "NS global fallback params: sill %.4f → %.4f, "
                    "nugget %.4f → %.4f (ratio %.3f preserved, ns_var=%.4f)",
                    self.sill, user_sill, self.nugget, user_nugget,
                    nugget_ratio, ns_var,
                )

        # Store effective (possibly rescaled) params for CV and other steps
        self._effective_sill = user_sill
        self._effective_nugget = user_nugget
        self._effective_alpha = user_alpha
        self._effective_drift_type = self._resolve_effective_drift_type(
            coords, values,
        )

        # ---- Log effective variogram parameters ----
        eff_total = user_sill + user_nugget
        eff_nug_ratio = user_nugget / eff_total if eff_total > 0 else 0
        logger.info("-" * 72)
        logger.info("EFFECTIVE VARIOGRAM PARAMS (after rescaling):")
        logger.info("  sill (partial)   : %.6f", user_sill)
        logger.info("  nugget           : %.6f", user_nugget)
        logger.info("  total (C0+C)     : %.6f", eff_total)
        logger.info("  nugget ratio     : %.1f%%", eff_nug_ratio * 100)
        logger.info("  range            : %.1f", user_range)
        logger.info("  alpha            : %.4f", user_alpha)
        logger.info(
            "  drift            : %s (requested=%s, selection=%s)",
            self._effective_drift_type,
            self.drift_type,
            self._trend_diagnostics.get("selection_method", "manual"),
        )
        logger.info("  accuracy         : %.2e", self.accuracy)
        if eff_nug_ratio > 0.5:
            max_r2 = (1.0 - eff_nug_ratio) ** 2
            logger.warning(
                "  *** HIGH NUGGET RATIO (%.0f%%) — theoretical max LOO-CV R² ≈ %.2f ***",
                eff_nug_ratio * 100, max_r2,
            )
        logger.info("-" * 72)

        R_global = rotation_matrix(self.azimuth, self.dip, self.pitch)
        ratio_mid = self.range_mid / max(self.range_max, 1e-12)
        ratio_min = self.range_min / max(self.range_max, 1e-12)
        self._working_values = values.copy()

        # ---- Decide: single-domain vs PUM sub-domains ----
        N = len(values)

        use_single_domain = (
            self.n_subdomains == 1
            or (self.n_subdomains == 0 and N <= self.pum_threshold)
        )

        if self.estimation_mode == "local_neighbourhood_gpr":
            self._single_domain = False
            self._subdomains = []
            self._progress(15, "Step 3: Local-neighbourhood GPR")
            if self.n_subdomains not in (0, 1):
                logger.warning(
                    "Local-neighbourhood GPR ignores n_subdomains=%d; "
                    "set estimation_mode='pum_legacy' to use partitioning.",
                    self.n_subdomains,
                )
            logger.info(
                "Local-neighbourhood GPR: global covariance, anisotropic search, "
                "max_samples=%d, min_samples=%d, radii=%.2f/%.2f/%.2f",
                self.max_samples,
                self.min_samples,
                self.local_search_radii[0],
                self.local_search_radii[1],
                self.local_search_radii[2],
            )
            self._progress(25, "Step 4: Preparing local-neighbourhood search")
        elif use_single_domain:
            # ---- Single-domain path (no PUM) ----
            # One global kernel matrix for ALL samples.  No sub-domain
            # decomposition, no Wendland blending, no between-model
            # variance.  Mathematically clean for N ≤ ~3,000.
            self._progress(15, "Step 3: Single-domain mode (no sub-domains)")
            self._single_domain = True
            if self.n_subdomains == 0:
                logger.info(
                    "Auto single-domain: N=%d <= pum_threshold=%d",
                    N, self.pum_threshold,
                )

            # Create one SubDomain containing ALL samples with a radius
            # large enough to cover the entire domain + query points.
            extent = np.max(coords, axis=0) - np.min(coords, axis=0)
            global_radius = float(np.linalg.norm(extent)) * 2.0
            global_centre = np.mean(coords, axis=0)

            sd = SubDomain(
                index=0,
                centre=global_centre,
                radius=global_radius,
                sample_indices=np.arange(N, dtype=np.intp),
            )
            sd.variogram_params = LocalVariogramResult(
                sill=user_sill,
                nugget=user_nugget,
                range_=user_range,
                alpha=user_alpha,
                kernel_type=self.kernel_type,
                fit_residual=0.0,
                n_pairs=0,
                n_lags=0,
            )
            self._subdomains = [sd]

            logger.info(
                "Single-domain: N=%d (threshold=%d), sill=%.4f, "
                "nugget=%.4f, range=%.1f, alpha=%.3f",
                N, self.pum_threshold, user_sill, user_nugget,
                user_range, user_alpha,
            )

            # ---- Step 4: Factorise the single global kernel matrix ----
            self._progress(25, "Step 4: Factorising %d×%d kernel matrix" % (N, N))
            S = scale_matrix(user_range, user_range * ratio_mid, user_range * ratio_min)

            fact, weights, poly_coeffs, applied_accuracy, w_max = \
                self._factorise_kernel_system(
                    coords,
                    values,
                    sd.variogram_params,
                    R_global,
                    S,
                    context="Single-domain",
                )
            logger.info(
                "Single-domain factorisation: requested accuracy=%.2e, "
                "applied accuracy=%.2e, |weights|_max=%.1f",
                self.accuracy, applied_accuracy, w_max,
            )

            sd.cholesky_factor = fact
            sd.l_inv = compute_l_inv(fact)  # cache L^{-1} once for PUM blending
            sd.weights = weights
            sd.poly_coeffs = poly_coeffs
            sd.scale_matrix_ = S

            logger.info(
                "Single-domain factorised: weights_range=[%.4e,%.4e], "
                "poly_c0=%.4f",
                float(np.min(weights)), float(np.max(weights)),
                float(poly_coeffs[0]) if len(poly_coeffs) > 0 else 0.0,
            )

        else:
            # ---- PUM sub-domain path (original) ----
            self._single_domain = False
            logger.warning(
                "estimation_mode='pum_legacy' retains legacy partition-of-unity "
                "blending and is not the preferred geostatistical mode.",
            )

            # ---- Step 3: Create sub-domains and fit local variograms ----
            self._progress(15, "Step 3: Creating sub-domains")
            self._subdomains = create_subdomains(
                coords,
                method=self.subdomain_method,
                k=self.n_subdomains if self.n_subdomains > 0 else None,
                centres=self.subdomain_centres,
                radii=self.subdomain_radii,
                overlap_factor=self.overlap_factor,
                min_samples_per_subdomain=self.min_samples,
                max_samples_per_subdomain=self.max_samples,
                seed=self.seed,
            )

            use_local_fit = (self.variogram_mode == "local")

            if use_local_fit or self.variogram_mode == "hybrid":
                self._progress(20, "Step 3: Fitting local variograms")
                fit_subdomain_variograms(
                    self._subdomains,
                    coords,
                    values,
                    kernel_type=self.kernel_type,
                )

            if self.variogram_mode == "global":
                # Use user's imported variogram for ALL subdomains
                self._progress(20, "Step 3: Using global variogram parameters")
                for sd in self._subdomains:
                    sd.variogram_params = LocalVariogramResult(
                        sill=user_sill,
                        nugget=user_nugget,
                        range_=user_range,
                        alpha=user_alpha,
                        kernel_type=self.kernel_type,
                        fit_residual=0.0,
                        n_pairs=0,
                        n_lags=0,
                    )
                logger.info(
                    "Global variogram: sill=%.4f, nugget=%.4f, range=%.1f, alpha=%.3f",
                    user_sill, user_nugget, user_range, user_alpha,
                )

            # ---- Step 4: Assemble and factorise local kernel matrices ----
            self._progress(25, "Step 4: Factorising kernel matrices")

            # Pre-process: hybrid variogram fallback (must be sequential)
            for sd in self._subdomains:
                if sd.variogram_params is None:
                    continue
                vp = sd.variogram_params
                if self.variogram_mode == "hybrid":
                    local_values_i = values[sd.sample_indices]
                    local_var = float(np.var(local_values_i)) if len(local_values_i) > 1 else 1.0
                    poor_fit = (
                        vp.nugget > vp.sill * 1.5
                        or vp.sill < local_var * 0.01
                        or vp.range_ < 1.0   # range < 1m means optimizer hit lower bound (no structure)
                    )
                    if poor_fit:
                        # Use global range/alpha (spatial structure is deposit-wide)
                        # but scale sill to LOCAL data variance to prevent
                        # variance discontinuities at subdomain boundaries.
                        # Without this, adjacent subdomains can have sill ratios
                        # of 7:1 (e.g. local sill=2 vs global sill=15), creating
                        # massive cliffs in posterior variance that break the
                        # Measured/Indicated/Inferred classification.
                        local_sill = max(local_var * 0.9, user_sill * 0.1)
                        local_sill = min(local_sill, user_sill * 3.0)
                        local_nugget = user_nugget * (local_sill / max(user_sill, 1e-12))
                        logger.warning(
                            "SD %d: poor local variogram (sill=%.3f, nugget=%.3f, "
                            "local_var=%.3f). Hybrid fallback: range=%.1f, "
                            "sill=%.3f (local-scaled, was global=%.3f).",
                            sd.index, vp.sill, vp.nugget, local_var,
                            user_range, local_sill, user_sill,
                        )
                        vp.sill = local_sill
                        vp.nugget = local_nugget
                        vp.range_ = user_range
                        vp.alpha = user_alpha

            # D1: Parallel factorisation (BLAS releases GIL → threads work)
            def _factorise_one(sd):
                if sd.variogram_params is None:
                    return sd
                vp = sd.variogram_params
                local_coords = coords[sd.sample_indices]
                local_values_i = values[sd.sample_indices]
                n_local = len(local_values_i)
                S = scale_matrix(vp.range_, vp.range_ * ratio_mid, vp.range_ * ratio_min)

                fact, weights, poly_coeffs, applied_accuracy, w_max = \
                    self._factorise_kernel_system(
                        local_coords,
                        local_values_i,
                        vp,
                        R_global,
                        S,
                        context=f"SD {sd.index}",
                    )
                logger.info(
                    "SD %d: requested accuracy=%.2e, applied accuracy=%.2e, "
                    "|weights|_max=%.1f",
                    sd.index, self.accuracy, applied_accuracy, w_max,
                )

                sd.cholesky_factor = fact
                sd.l_inv = compute_l_inv(fact)  # cache L^{-1} once for PUM blending
                sd.weights = weights
                sd.poly_coeffs = poly_coeffs
                sd.scale_matrix_ = S
                return sd

            n_sd = len(self._subdomains)
            if self.parallel and n_sd > 1:
                with ThreadPoolExecutor(max_workers=min(self.n_workers, n_sd)) as pool:
                    futures = {pool.submit(_factorise_one, sd): i
                               for i, sd in enumerate(self._subdomains)}
                    done = 0
                    for future in as_completed(futures):
                        future.result()  # propagate exceptions
                        done += 1
                        if done % max(1, n_sd // 5) == 0:
                            pct = 25 + int(20 * done / n_sd)
                            self._progress(pct, f"Step 4: Factorised {done}/{n_sd}")
            else:
                for i, sd in enumerate(self._subdomains):
                    _factorise_one(sd)
                    if (i + 1) % max(1, n_sd // 5) == 0:
                        pct = 25 + int(20 * (i + 1) / n_sd)
                        self._progress(pct, f"Step 4: Factorised {i + 1}/{n_sd}")

            # Diagnostic logging
            for sd in self._subdomains:
                if sd.weights is not None and sd.variogram_params is not None:
                    vp = sd.variogram_params
                    logger.info(
                        "SD %d: n=%d, sill=%.4f, nugget=%.4f, range=%.1f, "
                        "alpha=%.3f, weights_range=[%.4e,%.4e], poly_c0=%.4f",
                        sd.index, sd.n_samples, vp.sill, vp.nugget, vp.range_,
                        vp.alpha,
                        float(np.min(sd.weights)), float(np.max(sd.weights)),
                        float(sd.poly_coeffs[0]) if len(sd.poly_coeffs) > 0 else 0.0,
                    )

        # ---- Step 5: Estimate blocks with PUM blending ----
        self._progress(45, "Step 5: Estimating blocks")
        blended = self._estimate_blocks(coords, R_global)

        # ── Expand back to full block count if pre-filtered ──────────
        _active_mask = getattr(self, '_block_active_mask', None)
        if _active_mask is not None:
            # Restore original full-size centroids/sizes
            self._block_centroids = self._block_centroids_full
            self._block_sizes = self._block_sizes_full
            B_full = len(_active_mask)
            def _expand(arr_active, fill=np.nan):
                full = np.full(B_full, fill, dtype=np.float64)
                full[_active_mask] = arr_active
                return full
            blended_est = _expand(blended.estimates)
            blended_var = _expand(blended.variances)
            blended_within = _expand(blended.within_variance)
            blended_between = _expand(blended.between_variance, fill=0.0)
            blended_total = _expand(
                blended.total_variance
                if blended.total_variance is not None
                else blended.variances + blended.between_variance,
                fill=0.0,
            )
            if self._estimation_geometry_stats is not None:
                sc, oc, sp = self._estimation_geometry_stats
                sc_full = np.zeros(B_full, dtype=np.int32)
                oc_full = np.zeros(B_full, dtype=np.int32)
                sp_full = np.full(B_full, 3, dtype=np.int32)
                sc_full[_active_mask] = sc
                oc_full[_active_mask] = oc
                sp_full[_active_mask] = sp
                self._estimation_geometry_stats = (sc_full, oc_full, sp_full)
        else:
            blended_est = blended.estimates
            blended_var = blended.variances
            blended_within = blended.within_variance
            blended_between = blended.between_variance
            blended_total = (
                blended.total_variance
                if blended.total_variance is not None
                else blended.variances + blended.between_variance
            )

        # Diagnostic: log blending summary
        logger.info(
            "Blending summary: est_range=[%.4f,%.4f], "
            "posterior_range=[%.6e,%.6e], within_mean=%.6e, "
            "stitching_mean=%.6e",
            float(np.nanmin(blended_est)), float(np.nanmax(blended_est)),
            float(np.nanmin(blended_var)), float(np.nanmax(blended_var)),
            float(np.nanmean(blended_within)),
            float(np.nanmean(blended_between)),
        )

        grades = blended_est.copy()
        variances = blended_within.copy()
        stitching_variances = blended_between.copy()
        total_blending_variances = blended_total.copy()

        # jorc_variances tracks the true GPR posterior variance (within_variance
        # only — without PUM blending artifact) for use in JORC classification
        # and the variance-corrected NS back-transform.  It is updated below
        # when CoS is applied so it reflects the correct block-support variance.
        jorc_variances = blended_within.copy()
        # Keep a copy of point-support variance for the NS back-transform.
        # The GH quadrature needs the estimation uncertainty (how well do we
        # know the grade?), NOT the block-support variance (how variable is
        # the grade within the block?).  CoS reduces jorc_variances, which
        # makes blocks appear poorly-informed and forces the back-transform
        # to use the naive median — losing mass balance for skewed data.
        _point_support_variances = blended_within.copy()

        # ---- Step 6: Adaptive two-pass discretisation ----
        # Identify high-gradient blocks from the first-pass grades, then
        # re-estimate only those blocks with 64-point discretisation.
        # This solves the former chicken-and-egg ordering problem: gradients
        # are now computed from Step 5 estimates, not from a separate pass.
        if self.discretisation_mode == "adaptive":
            self._progress(48, "Step 6: Adaptive re-discretisation")
            grades, variances, stitching_variances = \
                self._apply_adaptive_discretisation(
                    grades, variances, stitching_variances, coords, R_global,
                )
            total_blending_variances = variances + stitching_variances

        # ---- Step 7: Cross-validation ----
        cv_result = None
        swath_data = None
        support_swath_result = None
        conditional_bias_result = None
        if self.run_cv:
            self._progress(65, "Step 7: Cross-validation")
            cv_result = self._run_cross_validation(coords, values)
            cv_result_failed = cv_result is None
            if cv_result_failed:
                cv_result = CVResult(
                    actual=np.empty(0, dtype=np.float64),
                    estimated=np.empty(0, dtype=np.float64),
                    errors=np.empty(0, dtype=np.float64),
                    mean_error=float("nan"),
                    mae=float("nan"),
                    rmse=float("nan"),
                    r_squared=float("nan"),
                    correlation=float("nan"),
                    normalised_rmse=float("nan"),
                    slope_of_regression=float("nan"),
                    intercept=float("nan"),
                    n_samples=0,
                )
            logger.info("-" * 72)
            executed_cv_mode = getattr(
                self,
                "_cv_execution_mode",
                (self.cv_mode or "spatial_kfold").lower(),
            )
            logger.info(
                "CROSS-VALIDATION RESULTS (requested=%s, executed=%s):",
                self.cv_mode,
                executed_cv_mode,
            )
            logger.info("  N samples used   : %d", cv_result.n_samples)
            logger.info("  Slope            : %.4f (ideal=1.0)", cv_result.slope_of_regression)
            logger.info("  R^2              : %.4f", cv_result.r_squared)
            logger.info("  RMSE             : %.4f", cv_result.rmse)

            # CV1 fix: also compute CV metrics in original units so JORC
            # reporting reflects the true estimation quality, not the
            # artificially-inflated NS-space metrics.
            if (
                str(executed_cv_mode).startswith("fast_loo")
                and self.use_normal_score
                and ns_table is not None
            ):
                cv_actual_orig = normal_score_backtransform(cv_result.actual, ns_table)
                cv_est_orig = normal_score_backtransform(cv_result.estimated, ns_table)
                from .cross_validation import _compute_cv_statistics
                cv_result_orig = _compute_cv_statistics(cv_actual_orig, cv_est_orig)
                logger.info("  --- Original units ---")
                logger.info("  Slope (original) : %.4f", cv_result_orig.slope_of_regression)
                logger.info("  R² (original)    : %.4f", cv_result_orig.r_squared)
                logger.info("  RMSE (original)  : %.4f", cv_result_orig.rmse)
                logger.info("  MAE (original)   : %.4f", cv_result_orig.mae)
                # Use original-unit metrics for the audit record (JORC compliance)
                cv_result = cv_result_orig
            else:
                logger.info("  Mean Error (ME)  : %.6f (should be ~0)", cv_result.mean_error)
                logger.info("  MAE              : %.4f", cv_result.mae)
                logger.info("  Normalised RMSE  : %.4f", cv_result.normalised_rmse)
                logger.info("  Correlation      : %.4f", cv_result.correlation)
                logger.info("  Intercept        : %.4f", cv_result.intercept)
            logger.info("-" * 72)
            if cv_result_failed:
                cv_result = None
            # B4: swath_plots deferred to after back-transform so values
            # are in original geological units, not normal-score space.

        # ---- Pre-filter: mask blocks with insufficient data support ----
        # Like SGSIM, only retain blocks where the estimation actually
        # carries local information.  Blocks with posterior variance
        # exceeding 50% of the prior (sill + nugget) have more uncertainty
        # than signal — their back-transformed grades would be dominated
        # by the prior distribution, not by conditioning data.
        #
        # Masking BEFORE the back-transform is critical: it prevents
        # data-sparse blocks from entering the Gauss-Hermite quadrature
        # where skewed distributions inflate grades toward the population
        # mean.  This is the geostatistical equivalent of "don't estimate
        # what you can't see" (Isaaks & Srivastava 1989, Ch. 12).
        # ---- Always mask uninformed blocks ----
        # Blocks with posterior variance exceeding the prior (sill + nugget)
        # have no local data support — their estimates are pure RBF
        # extrapolation, not conditioned on nearby composites.  This is
        # geostatistically unsound regardless of whether NS transform is
        # active (Isaaks & Srivastava 1989, Ch. 12).
        _pre_sill = float(
            (self._effective_sill + self._effective_nugget)
            if hasattr(self, '_effective_sill') and self._effective_sill is not None
            else 1.0
        )
        _vr = jorc_variances / max(_pre_sill, 1e-12)
        _needs_fallback = getattr(self, "_blocks_needing_knn_fallback", None)
        if _needs_fallback is not None and len(_needs_fallback) == len(grades):
            _uninformed = np.asarray(_needs_fallback, dtype=bool) & (_vr > 0.95)
        else:
            _uninformed = _vr > 0.95
        _n_masked = int(np.sum(_uninformed))
        if _n_masked > 0:
            grades[_uninformed] = np.nan
            jorc_variances[_uninformed] = np.nan
            _n_retained = int(np.sum(~_uninformed))
            logger.info(
                "PRE-MASK: %d / %d blocks (%.1f%%) have σ²/sill > 0.95 → "
                "set to NaN (no local data support).  %d blocks retained.",
                _n_masked, len(grades), 100.0 * _n_masked / len(grades),
                _n_retained,
            )

        # Stricter threshold for NS-transformed variables: blocks with
        # variance > 50% of sill are more uncertain than informative.
        # Back-transforming them inflates grades via Jensen's inequality.
        if self.mask_uninformed_ns_blocks:
            _uninformed_ns = _vr > 0.50
            _n_masked_ns = int(np.sum(_uninformed_ns & ~_uninformed))
            if _n_masked_ns > 0:
                grades[_uninformed_ns] = np.nan
                jorc_variances[_uninformed_ns] = np.nan
                _n_retained_ns = int(np.sum(np.isfinite(grades)))
                logger.info(
                    "NS-MASK: additional %d blocks (σ²/sill > 0.50) → NaN.  "
                    "%d blocks retained for back-transform.",
                    _n_masked_ns, _n_retained_ns,
                )

        # ---- Step 8: Change-of-support correction ----
        cos_result = None
        grade_std = float(np.nanstd(grades))  # NaN-safe after pre-mask
        if self.change_of_support:
            self._progress(80, "Step 8: Change-of-support correction")
            effective_disc_density = self._estimation_discretisation_density()
            global_vp = self._get_global_variogram_params()
            block_dims = self._block_sizes
            if block_dims.ndim == 2:
                block_dims = block_dims[0]

            sigma_w_sq = self._within_block_variance_cached(
                block_dims,
                global_vp,
                R_global,
                effective_disc_density,
            )
            sigma_point_sq = max(global_vp["sill"] + global_vp["nugget"], 0.0)
            sigma_point = float(np.sqrt(sigma_point_sq))

            if sigma_w_sq >= sigma_point_sq * 0.95 and sigma_point_sq > 0.0:
                logger.warning(
                    "CoS: within-block variance (%.4f) >= 95%% of point "
                    "variance (%.4f). Blocks may be large relative to "
                    "range (%.1f). Capping to preserve grade variability.",
                    sigma_w_sq, sigma_point_sq, global_vp["range_"],
                )
                sigma_w_sq = min(sigma_w_sq, sigma_point_sq * 0.75)

            sigma_block_sq = max(sigma_point_sq - sigma_w_sq, 0.0)
            sigma_block = float(np.sqrt(sigma_block_sq))
            support_ratio = (
                sigma_block / sigma_point
                if sigma_point > 0.0
                else 0.0
            )

            if (
                self.change_of_support_mode != "affine_legacy"
                or effective_disc_density > 1
            ):
                cos_result = ChangeOfSupportResult(
                    corrected_estimates=grades.copy(),
                    support_ratio=float(support_ratio),
                    sigma_point=float(sigma_point),
                    sigma_block=float(sigma_block),
                    sigma_within_block=float(np.sqrt(max(sigma_w_sq, 0.0))),
                    declustered_mean=float(np.nanmean(grades)),
                    method="discretized_block",
                )
                # Rescale variances from point-support to block-support
                jorc_variances *= support_ratio ** 2
                variances *= support_ratio ** 2
                total_blending_variances = variances + stitching_variances
                logger.info(
                    "Change-of-support: discretized block support "
                    "(density=%d, effective=%d). "
                    "Variance rescaled by support_ratio²=%.4f.",
                    self.discretisation_density, effective_disc_density,
                    support_ratio ** 2,
                )
            elif grade_std < 1e-12:
                logger.warning(
                    "Skipping affine change-of-support: all block estimates are "
                    "constant (std=%.2e).",
                    grade_std,
                )
            else:
                logger.warning(
                    "change_of_support_mode='affine_legacy' is deprecated and "
                    "not geostatistically preferred. Use discretized block support.",
                )
                if self._declustering_weights is not None:
                    dw = self._declustering_weights
                    if len(dw) == len(values):
                        declustered_mean = float(np.dot(dw, values))
                        logger.info(
                            "CoS mean: using CP-supplied declustering weights "
                            "(mean=%.4f vs arithmetic=%.4f)",
                            declustered_mean, float(np.mean(values)),
                        )
                    else:
                        logger.warning(
                            "Declustering weights length (%d) != N composites (%d). "
                            "Falling back to cell declustering.",
                            len(dw), len(values),
                        )
                        declustered_mean, _ = self._cell_decluster(
                            coords, values, self.decluster_cell_size,
                        )
                else:
                    declustered_mean, _ = self._cell_decluster(
                        coords, values, self.decluster_cell_size,
                    )

                cos_local_means = None
                if self.local_mean_radius > 0:
                    from scipy.spatial import cKDTree as _cKDTree
                    _tree = _cKDTree(coords)
                    cos_local_means = np.empty(B, dtype=np.float64)
                    for _b in range(B):
                        _nn = _tree.query_ball_point(
                            self._block_centroids[_b], self.local_mean_radius,
                        )
                        cos_local_means[_b] = (
                            float(np.mean(values[_nn]))
                            if len(_nn) >= 4
                            else declustered_mean
                        )
                    logger.info(
                        "CoS local mean (radius=%.1f): min=%.4f, max=%.4f, std=%.4f",
                        self.local_mean_radius,
                        float(np.min(cos_local_means)),
                        float(np.max(cos_local_means)),
                        float(np.std(cos_local_means)),
                    )

                cos_result = affine_correction(
                    grades,
                    declustered_mean=declustered_mean,
                    sigma_point=sigma_point,
                    sigma_within_block=np.sqrt(max(sigma_w_sq, 0.0)),
                    local_means=cos_local_means,
                )
                jorc_variances *= cos_result.support_ratio ** 2
                variances *= cos_result.support_ratio ** 2
                total_blending_variances = variances + stitching_variances
                logger.info("-" * 72)
                logger.info("CHANGE-OF-SUPPORT (CoS):")
                logger.info("  Method           : %s", cos_result.method)
                logger.info("  σ_point          : %.6f", sigma_point)
                logger.info("  σ_block          : %.6f", sigma_block)
                logger.info("  σ_block/σ_point  : %.4f", support_ratio)
                logger.info("  Declustered mean : %.6f", declustered_mean)
                logger.info("  Grade before CoS : min=%.4f, max=%.4f, std=%.4f",
                             float(np.nanmin(grades)), float(np.nanmax(grades)),
                             float(np.nanstd(grades)))
                logger.info("  Grade after CoS  : min=%.4f, max=%.4f, std=%.4f",
                             float(np.nanmin(cos_result.corrected_estimates)),
                             float(np.nanmax(cos_result.corrected_estimates)),
                             float(np.nanstd(cos_result.corrected_estimates)))
                logger.info("-" * 72)
                grades = cos_result.corrected_estimates

        # ---- Back-transform if normal-score was used ----
        if self.use_normal_score and ns_table is not None:
            self._progress(85, "Back-transforming estimates")
            logger.info("BACK-TRANSFORM (normal-score → original, variance-corrected):")
            logger.info("  Before: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                         float(np.nanmin(grades)), float(np.nanmax(grades)),
                         float(np.nanmean(grades)), float(np.nanstd(grades)))
            # Use variance-corrected back-transform (Gauss-Hermite quadrature)
            # to recover E[Z | data] = ∫ φ⁻¹(t) N(t; ŷ, σ²) dt, not the median
            # φ⁻¹(ŷ).  For skewed distributions (lognormal gold, etc.) the
            # difference is material — the naive transform underestimates high
            # grade blocks and distorts the grade-tonnage curve.
            # Use TOTAL sill (sill + nugget) in NS space for the GH
            # information threshold.  The posterior variance includes the
            # nugget component, so comparing against partial sill makes
            # blocks appear poorly-informed → they get the naive median
            # instead of GH mean → systematic underestimation for skewed
            # distributions.  With total sill, the threshold r = sigma²/sill
            # correctly identifies blocks where the data reduces uncertainty
            # beyond the nugget floor.
            _bt_sill = float(
                (self._effective_sill or 0.0) + (self._effective_nugget or 0.0)
                if hasattr(self, '_effective_sill') and self._effective_sill is not None
                else 1.0
            )
            # Use point-support variance (pre-CoS) for the GH information
            # threshold.  CoS-reduced variances would make blocks appear
            # poorly-informed and force naive median back-transform.
            grades = normal_score_backtransform_mean(
                grades, _point_support_variances, ns_table, sill=_bt_sill,
            )
            logger.info("  After : min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                         float(np.nanmin(grades)), float(np.nanmax(grades)),
                         float(np.nanmean(grades)), float(np.nanstd(grades)))

        # ---- Clip ----
        if self.clip_min is not None or self.clip_max is not None:
            pre_clip_min = float(np.nanmin(grades))
            pre_clip_max = float(np.nanmax(grades))
        if self.clip_min is not None:
            n_clipped_lo = int(np.nansum(grades < self.clip_min))
            # JORC AUDIT: clipping negative grades destroys kriging mass balance.
            # Negative block estimates arise from negatively-weighted samples used
            # to account for the screen effect of clustered data.  Setting them
            # to zero removes those negative weights from the unbiasedness
            # constraint and inflates the global metal content of the deposit.
            # For JORC/NI 43-101 compliant estimates, the preferred remedy is
            # to apply a non-linear transform (Normal Score) before estimation
            # so that the kriging system operates on a symmetric distribution
            # and never produces negative back-transformed grades.
            if self.clip_min <= 0.0 and n_clipped_lo > 0:
                logger.warning(
                    "JORC AUDIT WARNING — GRADE CLIPPING DESTROYS MASS BALANCE: "
                    "%d blocks (%.1f%%) have grades below %.4f and are being "
                    "reset to %.4f.  Kriging negative weights guarantee unbiased "
                    "estimation; removing them overstates contained metal.  "
                    "Resolution: set use_normal_score=True so the kriging system "
                    "operates on a Gaussian transform and naturally produces "
                    "non-negative back-transformed grades.",
                    n_clipped_lo,
                    100.0 * n_clipped_lo / max(len(grades), 1),
                    self.clip_min,
                    self.clip_min,
                )
            grades = np.maximum(grades, self.clip_min)
            logger.info("CLIP MIN: %.4f — %d blocks clipped (were as low as %.4f)",
                         self.clip_min, n_clipped_lo, pre_clip_min)
        if self.clip_max is not None:
            n_clipped_hi = int(np.nansum(grades > self.clip_max))
            grades = np.minimum(grades, self.clip_max)
            logger.info("CLIP MAX: %.4f — %d blocks clipped (were as high as %.4f)",
                         self.clip_max, n_clipped_hi, pre_clip_max)

        # ---- ILR multi-component: estimate remaining coordinates + inverse ----
        # Component 0 has been fully processed (CoS, NS back-transform, clip).
        # Now estimate ILR coordinates 1..D-2 using stored Cholesky factors,
        # then apply ilr_inverse to recover the full D-part composition.
        if (
            self.use_ilr
            and _ilr_all is not None
            and ilr_n_components is not None
            and ilr_kappa is not None
            and _ilr_all.shape[1] > 1
        ):
            D_minus_1 = _ilr_all.shape[1]
            ilr_block_estimates = np.zeros((B, D_minus_1), dtype=np.float64)
            ilr_block_estimates[:, 0] = grades   # component 0 fully processed
            for j in range(1, D_minus_1):
                values_j = _ilr_all[:, j].copy()
                ns_table_j = None
                if self.use_normal_score:
                    values_j, ns_table_j = normal_score_transform(
                        values_j, seed=self.seed + j, coords=coords,
                    )
                self._working_values = values_j.copy()
                self._refit_subdomain_weights(values_j)
                blended_j = self._estimate_blocks(coords, R_global)
                grades_j = blended_j.estimates.copy()
                if self.use_normal_score and ns_table_j is not None:
                    grades_j = normal_score_backtransform_mean(
                        grades_j, blended_j.within_variance, ns_table_j,
                        sill=_bt_sill,
                    )
                ilr_block_estimates[:, j] = grades_j
                logger.info(
                    "ILR component %d/%d: min=%.4f, max=%.4f, mean=%.4f",
                    j + 1, D_minus_1,
                    float(np.min(grades_j)), float(np.max(grades_j)),
                    float(np.mean(grades_j)),
                )
            compositions = ilr_inverse(ilr_block_estimates, kappa=ilr_kappa)
            logger.info(
                "ILR inverse: %d blocks × %d components. "
                "Row sums: min=%.4f, max=%.4f (target %.1f).",
                B, compositions.shape[1],
                float(np.min(compositions.sum(axis=1))),
                float(np.max(compositions.sum(axis=1))),
                ilr_kappa,
            )
            grades = compositions[:, 0]
            logger.info(
                "ILR primary (component 0): min=%.4f, max=%.4f, mean=%.4f",
                float(np.nanmin(grades)), float(np.nanmax(grades)),
                float(np.nanmean(grades)),
            )

        # ---- Swath plots (after back-transform so values are in original units) ----
        if self.run_cv:
            swath_data = swath_plots(
                self._block_centroids,
                grades,
                self._composite_coords,
                self._composite_values,
            )
            conditional_bias_result = (
                conditional_bias_diagnostics(
                    cv_result.actual,
                    cv_result.estimated,
                )
                if cv_result is not None
                else None
            )

        support_swath_result = support_swath_plots(
            self._block_centroids,
            grades,
            self._composite_coords,
            self._composite_values,
            block_sizes=self._block_sizes,
            declustering_weights=self._resolve_support_swath_declustering_weights(),
        )

        # ---- Step 9: Resource classification ----
        self._progress(88, "Step 9: Classification")
        if self._estimation_geometry_stats is not None:
            sample_counts, octant_counts, search_passes = \
                self._estimation_geometry_stats
        else:
            sample_counts, octant_counts, search_passes = \
                self._compute_geometric_stats(coords, R_global)

        vt = None
        prior_variance_for_classification = max(
            float(getattr(self, "_effective_sill", user_sill)),
            0.0,
        )
        if self.classification_thresholds:
            t1 = self.classification_thresholds.get("measured", 0.1)
            t2 = self.classification_thresholds.get("indicated", 0.3)
            t3 = self.classification_thresholds.get("inferred", 0.6)
            if not (np.isfinite(t1) and np.isfinite(t2) and np.isfinite(t3)):
                raise ValueError("classification thresholds must be finite values")
            if not (0.0 <= t1 < t2 < t3):
                raise ValueError(
                    "classification thresholds must satisfy 0 <= measured < indicated < inferred",
                )

            vt = VarianceThresholds(
                t1_measured=t1,
                t2_indicated=t2,
                t3_inferred=t3,
            )
            self._classification_threshold_source = "user"
            logger.info(
                "Classification thresholds: using user-supplied values "
                "(Measured<=%.4f, Indicated<=%.4f, Inferred<=%.4f).",
                t1, t2, t3,
            )
        else:
            if prior_variance_for_classification <= 0.0:
                raise ValueError(
                    "Cannot derive default classification thresholds because prior variance is non-positive",
                )
            vt = VarianceThresholds.from_prior_variance(
                prior_variance_for_classification,
            )
            self._classification_threshold_source = "prior_fraction_default"
            logger.warning(
                "No explicit classification thresholds supplied. "
                "Using prior-variance fractions of the effective sill "
                "(Measured<=%.4f, Indicated<=%.4f, Inferred<=%.4f).",
                vt.t1_measured, vt.t2_indicated, vt.t3_inferred,
            )
        self._classification_threshold_values = (
            float(vt.t1_measured),
            float(vt.t2_indicated),
            float(vt.t3_inferred),
        )

        gc = None
        if self.geometric_criteria:
            gc = GeometricCriteria(**self.geometric_criteria)

        # JORC classification uses only the true GPR posterior (within_variance).
        # The PUM between_variance is a Wendland-blending artefact from stitching
        # sub-domain boundaries — it is not a geostatistical estimation variance
        # and has no physical meaning for resource classification.  Using the
        # combined variances wrongly downgrades well-sampled boundary blocks.
        classification_result = classify_blocks(
            jorc_variances,
            sample_counts,
            octant_counts,
            search_passes,
            variance_thresholds=vt,
            geometric_criteria=gc,
            prior_variance=prior_variance_for_classification,
        )

        # ---- Step 10: JORC audit record ----
        self._progress(95, "Step 10: Generating audit record")
        elapsed = time.time() - t_start
        audit = self._build_audit_record(
            coords, values, grades, variances,
            cv_result, conditional_bias_result, support_swath_result,
            cos_result, classification_result, elapsed,
        )
        support_swath_summary = self._summarize_support_swath(support_swath_result)

        self._progress(100, "ARBF estimation complete")

        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        elapsed = time.time() - t_start
        logger.info("=" * 72)
        logger.info("ARBF ESTIMATION — FINAL SUMMARY")
        logger.info("=" * 72)
        logger.info("  Elapsed time     : %.1f seconds", elapsed)
        logger.info("  Blocks estimated : %d / %d", int(np.sum(np.isfinite(grades))), len(grades))
        logger.info("  Grade stats      : min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                     float(np.nanmin(grades)), float(np.nanmax(grades)),
                     float(np.nanmean(grades)), float(np.nanstd(grades)))
        logger.info("  Variance stats   : min=%.6e, max=%.6e, mean=%.6e, median=%.6e",
                     float(np.nanmin(variances)), float(np.nanmax(variances)),
                     float(np.nanmean(variances)), float(np.nanmedian(variances)))
        if classification_result is not None:
            report_mask = self._reporting_block_mask_for_output(
                len(classification_result.class_names),
            )
            scoped_names = np.asarray(classification_result.class_names, dtype=object)[
                report_mask
            ]
            for cls_name in ["Measured", "Indicated", "Inferred", "Unclassified"]:
                count = int(np.sum(scoped_names == cls_name))
                pct = count / len(scoped_names) * 100 if len(scoped_names) > 0 else 0
                logger.info("  %-16s : %d blocks (%.1f%%)", cls_name, count, pct)
        if cv_result is not None:
            logger.info("  CV R²            : %.4f", cv_result.r_squared)
            logger.info("  CV Slope         : %.4f", cv_result.slope_of_regression)
            logger.info("  CV RMSE          : %.4f", cv_result.rmse)
        logger.info("=" * 72)

        # Replace any remaining inf variances with NaN (blocks outside data)
        variances = np.where(np.isinf(variances), np.nan, variances)
        total_blending_variances = np.where(
            np.isinf(total_blending_variances), np.nan, total_blending_variances,
        )

        # Remove the dedicated ARBF log handler
        if _arbf_fh is not None:
            logger.removeHandler(_arbf_fh)
            _arbf_fh.close()

        reporting_mask = self._reporting_block_mask_for_output(len(grades))

        return ARBFResult(
            grades=grades,
            variances=variances,
            classifications=classification_result.classes,
            classification_names=classification_result.class_names,
            stitching_variance=stitching_variances,
            total_blending_variance=total_blending_variances,
            cv_result=cv_result,
            swath_data=swath_data,
            support_swath_data=support_swath_result,
            conditional_bias_result=conditional_bias_result,
            cos_result=cos_result,
            classification_result=classification_result,
            audit_record=audit,
            diagnostics={
                "estimation_mode": self.estimation_mode,
                "requested_drift_type": self.drift_type,
                "effective_drift_type": self._active_drift_type(),
                "drift_selection_method": self._trend_diagnostics.get(
                    "selection_method", "manual",
                ),
                "trend_constant_cv_rmse": float(
                    self._trend_diagnostics.get("constant", {}).get("rmse", 0.0),
                ),
                "trend_linear_cv_rmse": float(
                    self._trend_diagnostics.get("linear", {}).get("rmse", 0.0),
                ),
                "trend_rmse_improvement": float(
                    self._trend_diagnostics.get("rmse_improvement", 0.0),
                ),
                "trend_slope_improvement": float(
                    self._trend_diagnostics.get("slope_improvement", 0.0),
                ),
                "local_drift_fallbacks": int(self._local_drift_fallbacks),
                "domain_policy": self._domain_diagnostics.get(
                    "domain_policy", self.domain_policy,
                ),
                "geological_domains": int(
                    self._domain_diagnostics.get("n_geological_domains", 0),
                ),
                "domains_enforced": bool(
                    self._domain_diagnostics.get("domains_enforced", False),
                ),
                "clip_to_drill_footprint": self.clip_to_drill_footprint,
                "footprint_buffer_ranges": self.footprint_buffer_ranges,
                "cv_requested_mode": self.cv_mode,
                "cv_execution_mode": self._cv_execution_mode,
                "classification_threshold_source": self._classification_threshold_source,
                "classification_thresholds": (
                    list(self._classification_threshold_values)
                    if self._classification_threshold_values is not None
                    else None
                ),
                "block_domain_assignment": self._domain_diagnostics.get(
                    "block_domain_assignment", "none",
                ),
                "single_domain": getattr(self, '_single_domain', False),
                "n_subdomains": len(self._subdomains),
                "avg_samples_per_sd": (
                    float(np.mean([sd.n_samples for sd in self._subdomains]))
                    if self._subdomains
                    else 0.0
                ),
                "n_blocks_estimated": int(np.sum(np.isfinite(grades[reporting_mask]))),
                "n_blocks_total": int(np.sum(reporting_mask)),
                "elapsed_seconds": elapsed,
                "mean_estimation_samples": (
                    float(np.mean(sample_counts))
                    if sample_counts is not None and len(sample_counts) > 0
                    else 0.0
                ),
                "mean_estimation_octants": (
                    float(np.mean(octant_counts))
                    if octant_counts is not None and len(octant_counts) > 0
                    else 0.0
                ),
                "blended_within_variance_mean": float(np.mean(
                    blended.within_variance,
                )),
                "blended_between_variance_mean": float(np.mean(
                    blended.between_variance,
                )),
                "stitching_variance_mean": float(np.nanmean(stitching_variances)),
                "total_blending_variance_mean": float(
                    np.nanmean(total_blending_variances),
                ),
                "conditional_bias_binned_slope": (
                    float(conditional_bias_result.binned_slope)
                    if conditional_bias_result is not None
                    else float("nan")
                ),
                "conditional_bias_max_abs_bin_bias": (
                    float(conditional_bias_result.max_abs_bin_bias)
                    if conditional_bias_result is not None
                    else float("nan")
                ),
                "support_swath_panel_factors": (
                    list(support_swath_result.panel_factors)
                    if support_swath_result is not None
                    else None
                ),
                "support_swath_panels_total": int(
                    support_swath_summary["n_panels_total"],
                ),
                "support_swath_panels_with_data": int(
                    support_swath_summary["n_panels_with_data"],
                ),
                "support_swath_mean_rmse": float(
                    support_swath_summary["mean_rmse"],
                ),
                "support_swath_mean_bias": float(
                    support_swath_summary["mean_bias"],
                ),
                "cv_r_squared": (
                    float(cv_result.r_squared)
                    if cv_result is not None
                    else float("nan")
                ),
                "cv_slope_of_regression": (
                    float(cv_result.slope_of_regression)
                    if cv_result is not None
                    else float("nan")
                ),
                "cv_mean_error": (
                    float(cv_result.mean_error)
                    if cv_result is not None
                    else float("nan")
                ),
                "n_stabilized_systems": len(self._stabilization_events),
                "max_applied_accuracy": float(max(
                    [self.accuracy] + [
                        evt["applied_accuracy"]
                        for evt in self._stabilization_events
                    ],
                )),
            },
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _estimate_with_domains(self) -> "ARBFResult":
        """Run estimation domain-by-domain to enforce hard geological boundaries.

        Each domain is estimated independently using only its own composites.
        Results are merged by filling pre-allocated output arrays at the
        correct block indices.  Blocks that are not covered by any domain
        remain NaN / Unclassified rather than being turned into synthetic
        zero-grade estimates.
        """
        from scipy.spatial import cKDTree as _cKDTree
        t_start = time.time()

        composite_domains = self._composite_domains
        composite_coords = self._composite_coords
        composite_values = self._composite_values
        block_centroids = self._block_centroids
        B = block_centroids.shape[0]

        unique_domains = np.unique(composite_domains)
        logger.info(
            "Hard-boundary estimation: %d domains — %s",
            len(unique_domains), list(unique_domains),
        )

        # Assign blocks to domains via nearest-composite domain lookup
        if self._block_domains is not None:
            block_domain_arr = self._block_domains
        else:
            logger.info(
                "Block domains not set — assigning by nearest composite (KD-tree).",
            )
            tree = _cKDTree(composite_coords)
            _, nn_idx = tree.query(block_centroids, k=1)
            block_domain_arr = composite_domains[nn_idx]

        # Pre-allocate full output arrays (NaN for unestimated blocks)
        grades_out = np.full(B, np.nan, dtype=np.float64)
        variances_out = np.full(B, np.nan, dtype=np.float64)
        stitching_out = np.full(B, np.nan, dtype=np.float64)
        total_var_out = np.full(B, np.nan, dtype=np.float64)
        jorc_var_out = np.full(B, np.nan, dtype=np.float64)
        classifications_out = np.full(B, 0, dtype=np.int32)
        class_names_out = np.full(B, "Unclassified", dtype=object)

        results_by_domain: Dict[Any, "ARBFResult"] = {}

        for dom in unique_domains:
            # --- Filter composites for this domain ---
            comp_mask = composite_domains == dom
            dom_coords = composite_coords[comp_mask]
            dom_values = composite_values[comp_mask]

            if len(dom_values) < self.min_samples:
                logger.warning(
                    "Domain '%s': only %d composites (min_samples=%d). "
                    "Skipping — its blocks will be Unclassified.",
                    dom, len(dom_values), self.min_samples,
                )
                continue

            # --- Filter blocks for this domain ---
            blk_mask = block_domain_arr == dom
            if not np.any(blk_mask):
                logger.info("Domain '%s': no blocks assigned. Skipping.", dom)
                continue

            dom_centroids = block_centroids[blk_mask]
            dom_block_sizes = (
                self._block_sizes[blk_mask]
                if self._block_sizes.ndim == 2
                else self._block_sizes
            )

            # --- Create a child estimator for this domain ---
            child_cfg = self._build_child_config(for_cv=False)
            child_cfg["domain_policy"] = "ignore"
            child = ARBFEstimator(child_cfg)
            child.set_composites(dom_coords, dom_values)
            child.set_block_model(dom_centroids, dom_block_sizes)
            if self._orientation_field is not None and self.lva_source != "data":
                child.set_orientation_field(self._orientation_field)
            if self._declustering_weights is not None:
                child.set_declustering_weights(
                    self._declustering_weights[comp_mask],
                )
            if self._progress_callback:
                child.set_progress_callback(self._progress_callback)

            logger.info(
                "Domain '%s': %d composites, %d blocks",
                dom, len(dom_values), int(np.sum(blk_mask)),
            )
            dom_result = child.estimate()
            results_by_domain[dom] = dom_result

            # Merge back into full arrays
            blk_idx = np.where(blk_mask)[0]
            grades_out[blk_idx] = dom_result.grades
            variances_out[blk_idx] = dom_result.variances
            if dom_result.stitching_variance is not None:
                stitching_out[blk_idx] = dom_result.stitching_variance
            if dom_result.total_blending_variance is not None:
                total_var_out[blk_idx] = dom_result.total_blending_variance
            else:
                total_var_out[blk_idx] = dom_result.variances
            jorc_var_out[blk_idx] = dom_result.variances
            classifications_out[blk_idx] = dom_result.classifications
            class_names_out[blk_idx] = dom_result.classification_names

        covered_mask = np.isfinite(variances_out)
        n_blocks_estimated = int(np.sum(covered_mask))
        logger.info(
            "Domain merge: %d / %d blocks estimated (%.1f%% covered)",
            n_blocks_estimated, B,
            100.0 * np.mean(covered_mask),
        )

        cv_actual_parts: List[np.ndarray] = []
        cv_est_parts: List[np.ndarray] = []
        for dom_result in results_by_domain.values():
            cv_dom = dom_result.cv_result
            if cv_dom is None:
                continue
            actual = np.asarray(cv_dom.actual, dtype=np.float64)
            estimated = np.asarray(cv_dom.estimated, dtype=np.float64)
            finite = np.isfinite(actual) & np.isfinite(estimated)
            if np.any(finite):
                cv_actual_parts.append(actual[finite])
                cv_est_parts.append(estimated[finite])

        cv_result = None
        swath_data = None
        conditional_bias_result = None
        if cv_actual_parts:
            cv_result = _compute_cv_statistics(
                np.concatenate(cv_actual_parts),
                np.concatenate(cv_est_parts),
            )
        if self.run_cv:
            swath_data = swath_plots(
                self._block_centroids,
                grades_out,
                self._composite_coords,
                self._composite_values,
            )
            if cv_result is not None:
                conditional_bias_result = conditional_bias_diagnostics(
                    cv_result.actual,
                    cv_result.estimated,
                )

        support_swath_result = support_swath_plots(
            self._block_centroids,
            grades_out,
            self._composite_coords,
            self._composite_values,
            block_sizes=self._block_sizes,
            declustering_weights=self._resolve_support_swath_declustering_weights(),
        )

        cos_children = [
            (dom_result.cos_result, int(np.sum(block_domain_arr == dom)))
            for dom, dom_result in results_by_domain.items()
            if dom_result.cos_result is not None
        ]
        cos_result = None
        if cos_children:
            weights = np.array([count for _, count in cos_children], dtype=np.float64)
            weights /= max(np.sum(weights), 1.0)
            cos_result = ChangeOfSupportResult(
                corrected_estimates=grades_out.copy(),
                support_ratio=float(np.sum([
                    w * child.support_ratio
                    for w, (child, _) in zip(weights, cos_children)
                ])),
                sigma_point=float(np.sum([
                    w * child.sigma_point
                    for w, (child, _) in zip(weights, cos_children)
                ])),
                sigma_block=float(np.sum([
                    w * child.sigma_block
                    for w, (child, _) in zip(weights, cos_children)
                ])),
                sigma_within_block=float(np.sum([
                    w * child.sigma_within_block
                    for w, (child, _) in zip(weights, cos_children)
                ])),
                declustered_mean=float(np.sum([
                    w * child.declustered_mean
                    for w, (child, _) in zip(weights, cos_children)
                ])),
                method="domain_weighted_merge",
            )

        class_summary = self._reporting_class_summary(class_names_out)
        classification_result = ClassificationResult(
            classes=classifications_out,
            class_names=class_names_out,
            variance_classes=classifications_out.copy(),
            geometric_classes=classifications_out.copy(),
            summary=class_summary,
        )

        # Replace any remaining inf variances with NaN (unestimated blocks)
        variances_out = np.where(np.isinf(variances_out), np.nan, variances_out)
        total_var_out = np.where(np.isinf(total_var_out), np.nan, total_var_out)
        domain_child_drifts = {
            str(dom): results_by_domain[dom].diagnostics.get(
                "effective_drift_type", self._active_drift_type(),
            )
            for dom in results_by_domain
        }
        domain_child_threshold_sources = {
            str(dom): results_by_domain[dom].diagnostics.get(
                "classification_threshold_source", "unset",
            )
            for dom in results_by_domain
        }
        domain_child_thresholds = [
            tuple(result.diagnostics.get("classification_thresholds", []) or [])
            for result in results_by_domain.values()
            if result.diagnostics.get("classification_thresholds") is not None
        ]
        unique_child_drifts = sorted(set(domain_child_drifts.values()))
        merged_drift = (
            unique_child_drifts[0]
            if len(unique_child_drifts) == 1
            else ("mixed" if unique_child_drifts else self._active_drift_type())
        )
        unique_threshold_sources = sorted(set(domain_child_threshold_sources.values()))
        self._classification_threshold_source = (
            unique_threshold_sources[0]
            if len(unique_threshold_sources) == 1
            else ("mixed" if unique_threshold_sources else "unset")
        )
        unique_threshold_sets = sorted(set(domain_child_thresholds))
        self._classification_threshold_values = (
            unique_threshold_sets[0]
            if len(unique_threshold_sets) == 1 and len(unique_threshold_sets[0]) == 3
            else None
        )
        support_swath_summary = self._summarize_support_swath(support_swath_result)
        elapsed = time.time() - t_start
        local_drift_fallbacks = int(sum(
            int(dom_result.diagnostics.get("local_drift_fallbacks", 0))
            for dom_result in results_by_domain.values()
        ))
        audit = self._build_audit_record(
            self._composite_coords,
            self._composite_values,
            grades_out,
            variances_out,
            cv_result,
            conditional_bias_result,
            support_swath_result,
            cos_result,
            classification_result,
            elapsed,
        )

        return ARBFResult(
            grades=grades_out,
            variances=variances_out,
            classifications=classifications_out,
            classification_names=class_names_out,
            stitching_variance=stitching_out,
            total_blending_variance=total_var_out,
            cv_result=cv_result,
            swath_data=swath_data,
            support_swath_data=support_swath_result,
            conditional_bias_result=conditional_bias_result,
            cos_result=cos_result,
            classification_result=classification_result,
            audit_record=audit,
            diagnostics={
                "estimation_mode": self.estimation_mode,
                "requested_drift_type": self.drift_type,
                "effective_drift_type": merged_drift,
                "azimuth": self.azimuth,
                "dip": self.dip,
                "pitch": self.pitch,
                "range_max": self.range_max,
                "range_mid": self.range_mid,
                "range_min": self.range_min,
                "clip_to_drill_footprint": self.clip_to_drill_footprint,
                "footprint_buffer_ranges": self.footprint_buffer_ranges,
                "cv_requested_mode": self.cv_mode,
                "cv_execution_mode": self._cv_execution_mode,
                "classification_threshold_source": self._classification_threshold_source,
                "classification_thresholds": (
                    list(self._classification_threshold_values)
                    if self._classification_threshold_values is not None
                    else None
                ),
                "domain_policy": self._domain_diagnostics.get(
                    "domain_policy", self.domain_policy,
                ),
                "geological_domains": int(len(unique_domains)),
                "domains_enforced": True,
                "block_domain_assignment": (
                    "provided" if self._block_domains is not None else "nearest_composite"
                ),
                "estimated_domains": int(len(results_by_domain)),
                "n_blocks_estimated": n_blocks_estimated,
                "n_blocks_total": int(B),
                "child_effective_drifts": domain_child_drifts,
                "child_classification_threshold_sources": domain_child_threshold_sources,
                "elapsed_seconds": elapsed,
                "local_drift_fallbacks": local_drift_fallbacks,
                "conditional_bias_binned_slope": (
                    float(conditional_bias_result.binned_slope)
                    if conditional_bias_result is not None
                    else float("nan")
                ),
                "conditional_bias_max_abs_bin_bias": (
                    float(conditional_bias_result.max_abs_bin_bias)
                    if conditional_bias_result is not None
                    else float("nan")
                ),
                "support_swath_panel_factors": (
                    list(support_swath_result.panel_factors)
                    if support_swath_result is not None
                    else None
                ),
                "support_swath_panels_total": int(
                    support_swath_summary["n_panels_total"],
                ),
                "support_swath_panels_with_data": int(
                    support_swath_summary["n_panels_with_data"],
                ),
                "support_swath_mean_rmse": float(
                    support_swath_summary["mean_rmse"],
                ),
                "support_swath_mean_bias": float(
                    support_swath_summary["mean_bias"],
                ),
                "cv_r_squared": (
                    float(cv_result.r_squared) if cv_result is not None else float("nan")
                ),
                "cv_slope_of_regression": (
                    float(cv_result.slope_of_regression)
                    if cv_result is not None
                    else float("nan")
                ),
                "cv_mean_error": (
                    float(cv_result.mean_error) if cv_result is not None else float("nan")
                ),
            },
        )

    def _cell_decluster(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        cell_size: float = 0.0,
    ) -> Tuple[float, np.ndarray]:
        """Deutsch (1989) cell declustering for the global mean.

        Assigns each sample to a regular 3-D grid cell of size *cell_size*.
        Every sample is weighted by the inverse of the number of samples in
        its cell, so densely drilled high-grade zones are downweighted and
        sparsely drilled low-grade zones are upweighted.

        Parameters
        ----------
        coords : np.ndarray
            (N, 3) sample coordinates.
        values : np.ndarray
            (N,) grade values.
        cell_size : float
            Declustering cell side length.  ``0`` triggers auto-selection:
            5× the median nearest-neighbour spacing (covers the typical
            tight/wide drill pattern range for Au, Cu, Fe deposits).

        Returns
        -------
        declustered_mean : float
            Weighted mean grade.
        weights : np.ndarray
            (N,) normalised declustering weights (sum = 1).
        """
        from scipy.spatial import cKDTree as _cKDTree

        N = len(values)

        if cell_size <= 0.0:
            if N > 1:
                tree = _cKDTree(coords)
                nn_d, _ = tree.query(coords, k=min(2, N))
                nn1 = nn_d[:, 1] if nn_d.ndim > 1 else nn_d
                median_spacing = float(np.median(nn1[nn1 > 1e-6]))
                cell_size = max(median_spacing * 5.0, 1.0)
            else:
                cell_size = 1.0
            logger.info("Cell declustering: auto cell_size = %.2f m", cell_size)

        # Assign each sample to a cell index
        mins = np.min(coords, axis=0)
        cell_idx = np.floor((coords - mins) / cell_size).astype(np.int64)

        # Count samples per cell — fully vectorised, zero Python loops.
        # Convert (N, 3) cell indices to a flat unique integer key using
        # stride-based linearisation, then use np.unique to count per cell.
        max_ci = cell_idx.max(axis=0)
        strides = np.array([
            (max_ci[1] + 1) * (max_ci[2] + 1),
            (max_ci[2] + 1),
            1,
        ], dtype=np.int64)
        flat_keys = (cell_idx * strides).sum(axis=1)  # (N,) unique cell IDs

        _, cell_labels, cell_counts = np.unique(
            flat_keys, return_inverse=True, return_counts=True,
        )
        # Weight each sample by 1/count_of_its_cell, then normalise
        weights = 1.0 / cell_counts[cell_labels].astype(np.float64)
        weights /= weights.sum()
        declustered_mean = float(np.dot(weights, values))

        n_occupied = int(cell_counts.shape[0])
        logger.info(
            "Cell declustering: %d samples → %d occupied cells "
            "(cell_size=%.1f). Arithmetic mean=%.4f, declustered=%.4f",
            N, n_occupied, cell_size,
            float(np.mean(values)), declustered_mean,
        )
        return declustered_mean, weights

    def _validate_inputs(self) -> None:
        """Validate that required data is set and parameters are reasonable."""
        if self._composite_coords is None or self._composite_values is None:
            raise ValueError("Composite data not set. Call set_composites() first.")
        if self._block_centroids is None or self._block_sizes is None:
            raise ValueError("Block model not set. Call set_block_model() first.")
        if len(self._composite_values) < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} composites, "
                f"got {len(self._composite_values)}."
            )
        if self.drift_type.lower() not in {"none", "constant", "linear", "auto"}:
            raise ValueError(
                "drift_type must be 'none', 'constant', 'linear', or 'auto'",
            )
        if self._composite_domains is not None and (
            len(self._composite_domains) != len(self._composite_values)
        ):
            raise ValueError(
                "Composite domain labels must align 1-to-1 with composites.",
            )
        if self._block_domains is not None and (
            len(self._block_domains) != len(self._block_centroids)
        ):
            raise ValueError(
                "Block domain labels must align 1-to-1 with block centroids.",
            )

        # ── Anisotropy sanity check ──────────────────────────────────────
        # If the three variogram ranges differ by more than 1.5:1 but all
        # orientation angles are exactly zero, the user almost certainly
        # forgot to set the variogram orientation.  An un-rotated search
        # ellipsoid aligned with the grid axes rarely matches geological
        # continuity and will produce biased estimates.
        max_range = max(self.range_max, 1e-12)
        min_range = max(self.range_min, 1e-12)
        aniso_ratio = max_range / min_range
        all_angles_zero = (
            abs(self.azimuth) < 1e-6
            and abs(self.dip) < 1e-6
            and abs(self.pitch) < 1e-6
        )
        if aniso_ratio > 1.5 and all_angles_zero:
            logger.warning(
                "ANISOTROPY CHECK: range ratio is %.1f:1 (%.1f / %.1f / %.1f m) "
                "but azimuth/dip/pitch are all 0°.  This means the search "
                "ellipsoid is aligned with the GRID axes, not the geological "
                "continuity.  Unless the deposit truly trends along X/Y/Z, "
                "this WILL bias estimates.  Fit directional variograms to "
                "determine the correct orientation.",
                aniso_ratio, self.range_max, self.range_mid, self.range_min,
            )

        # ── Block size vs range check ────────────────────────────────────
        # Blocks much larger than the correlation range produce highly
        # smoothed estimates that may not reflect local grade variability.
        bs = self._block_sizes
        if bs.ndim == 2:
            bs = bs[0]
        max_block_dim = float(np.max(bs))
        if max_block_dim > max_range * 0.75:
            logger.warning(
                "SUPPORT CHECK: largest block dimension (%.1f m) is > 75%% of "
                "the variogram range (%.1f m).  Block estimates will be "
                "heavily smoothed.  Consider using smaller blocks or "
                "verifying that the variogram range is correct.",
                max_block_dim, max_range,
            )

    def _progress(self, percent: int, message: str) -> None:
        """Emit progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(percent, message)
        if self.verbose:
            logger.info("[%3d%%] %s", percent, message)

    def _build_orientation_field(
        self,
        coords: np.ndarray,
        values: np.ndarray,
    ) -> OrientationField:
        """Build orientation field based on configuration."""
        # Determine grid parameters from data extent
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        extent = maxs - mins
        grid_dims = (
            max(int(extent[0] / max(self.range_max, 1) * 2), 5),
            max(int(extent[1] / max(self.range_mid, 1) * 2), 5),
            max(int(extent[2] / max(self.range_min, 1) * 2), 5),
        )
        # Cap grid size
        grid_dims = tuple(min(d, 50) for d in grid_dims)
        grid_spacing = extent / np.maximum(np.array(grid_dims) - 1, 1)

        if self.lva_source == "data":
            return OrientationField.from_grade_data(
                coords, values, mins, grid_spacing, grid_dims,
            )
        else:
            return OrientationField.identity(mins, grid_spacing, grid_dims)

    def _support_discretisation_density(self, requested_density: int) -> int:
        """Density used when approximating block-support variance."""
        if requested_density > 1:
            return int(requested_density)
        return 27 if self.change_of_support else int(requested_density)

    def _estimation_discretisation_density(self) -> int:
        """Effective discretisation density used for block estimation."""
        if self.change_of_support and self.change_of_support_mode != "affine_legacy":
            return self._support_discretisation_density(self.discretisation_density)
        return int(self.discretisation_density)

    def _active_drift_type(self) -> str:
        """Drift model selected for the current estimation run."""
        return getattr(self, "_effective_drift_type", self.drift_type)

    def _local_neighbourhood_drift_type(
        self,
        local_coords: np.ndarray,
        octant_count: Optional[int] = None,
    ) -> str:
        """Resolve a stable drift basis for one local neighbourhood."""
        drift_type = self._active_drift_type()
        if drift_type != "linear":
            return drift_type

        min_linear_samples = max(self.min_samples, 8)
        centred = np.asarray(local_coords, dtype=np.float64) - np.mean(
            local_coords, axis=0, keepdims=True,
        )
        if (
            len(local_coords) < min_linear_samples
            or np.linalg.matrix_rank(centred) < 3
            or (
                octant_count is not None
                and int(octant_count) < int(self.search_min_octants_linear)
            )
        ):
            self._local_drift_fallbacks += 1
            return "constant"
        return "linear"

    def _summarize_domain_state(self) -> Dict[str, Any]:
        """Summarise hard-boundary domain state and enforce policy."""
        n_domains = (
            int(len(np.unique(self._composite_domains)))
            if self._composite_domains is not None
            else 0
        )
        block_assignment = (
            "provided"
            if self._block_domains is not None
            else "nearest_composite"
            if self._composite_domains is not None
            else "none"
        )
        diagnostics = {
            "domain_policy": self.domain_policy,
            "n_geological_domains": n_domains,
            "domains_enforced": bool(self._composite_domains is not None),
            "block_domain_assignment": block_assignment,
        }

        if self._composite_domains is None:
            if self.domain_policy == "require":
                raise ValueError(
                    "domain_policy='require' but no geological domains were supplied. "
                    "Call set_domains() before estimate()."
                )
            if self.domain_policy == "warn":
                logger.warning(
                    "DOMAINS: no geological domains supplied. Estimation will run "
                    "as a pooled domain and can smear across contacts or faults.",
                )
            return diagnostics

        if self._block_domains is None:
            logger.warning(
                "DOMAINS: block domains not supplied. Falling back to nearest-composite "
                "domain assignment, which is weaker than a geological solids model.",
            )

        logger.info(
            "DOMAINS: hard boundaries enabled for %d geological domain(s); "
            "block assignment=%s; policy=%s.",
            n_domains,
            block_assignment,
            self.domain_policy,
        )
        return diagnostics

    def _evaluate_drift_candidates(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        *,
        use_estimator_cv: bool = True,
    ) -> Dict[str, Any]:
        """Compare constant vs linear drift by spatially representative LOO-CV."""
        max_samples = min(max(self.auto_drift_cv_max_samples, 0), len(values))
        if max_samples < max(self.min_samples + 1, 8):
            raise ValueError("Not enough samples for automatic drift selection")

        subset_idx = self._select_spatial_cv_indices(coords, max_samples)
        subset_coords = coords[subset_idx]
        subset_values = values[subset_idx]
        candidate_max_samples = min(len(subset_values), max(24, min(max_samples, 48)))
        if len(subset_values) > candidate_max_samples:
            candidate_idx = self._select_spatial_cv_indices(
                subset_coords, candidate_max_samples,
            )
            subset_coords = subset_coords[candidate_idx]
            subset_values = subset_values[candidate_idx]
            subset_idx = subset_idx[candidate_idx]
        common_kwargs = {
            "kernel_type": self.kernel_type,
            "alpha": getattr(self, "_effective_alpha", self.alpha),
            "sill": getattr(self, "_effective_sill", self.sill),
            "range_": self.range_max,
            "range_mid": self.range_mid,
            "range_min": self.range_min,
            "nugget": getattr(self, "_effective_nugget", self.nugget),
            "accuracy": self.accuracy,
            "azimuth": self.azimuth,
            "dip": self.dip,
            "pitch": self.pitch,
            "max_samples": len(subset_values),
        }

        if use_estimator_cv:
            subset_weights = None
            if self._declustering_weights is not None:
                subset_weights = np.asarray(
                    self._declustering_weights[subset_idx], dtype=np.float64,
                )
                w_sum = float(np.sum(subset_weights))
                if w_sum > 0.0:
                    subset_weights = subset_weights / w_sum
                else:
                    subset_weights = None

            common_cfg = self._build_child_config(for_cv=True)
            common_cfg.update(
                {
                    "alpha": getattr(self, "_effective_alpha", self.alpha),
                    "sill": getattr(self, "_effective_sill", self.sill),
                    "nugget": getattr(self, "_effective_nugget", self.nugget),
                    "cv_folds": max(2, min(3, self.cv_folds, len(subset_values))),
                    "cv_max_samples": int(len(subset_values)),
                    "verbose": False,
                }
            )

            def _run_drift_candidate(drift_name: str) -> Tuple[CVResult, str]:
                candidate = ARBFEstimator({**common_cfg, "drift_type": drift_name})
                candidate.set_composites(subset_coords, subset_values)
                if subset_weights is not None:
                    candidate.set_declustering_weights(subset_weights)
                cv_result = candidate._run_spatial_kfold_cv()
                if cv_result is not None:
                    return cv_result, "estimator_spatial_kfold"
                if not self.allow_cv_fallback:
                    raise RuntimeError(
                        "Spatial CV unavailable for drift comparison and allow_cv_fallback=False",
                    )
                return (
                    leave_one_out_cv(
                        subset_coords,
                        subset_values,
                        drift_type=drift_name,
                        **common_kwargs,
                    ),
                    "fallback_fast_loo",
                )
        else:
            def _run_drift_candidate(drift_name: str) -> Tuple[CVResult, str]:
                return (
                    leave_one_out_cv(
                        subset_coords,
                        subset_values,
                        drift_type=drift_name,
                        **common_kwargs,
                    ),
                    "fast_loo",
                )

        cv_constant, method_constant = _run_drift_candidate("constant")
        cv_linear, method_linear = _run_drift_candidate("linear")

        rmse_improvement = (
            (cv_constant.rmse - cv_linear.rmse) / cv_constant.rmse
            if cv_constant.rmse > 0
            else 0.0
        )
        slope_improvement = (
            abs(cv_constant.slope_of_regression - 1.0)
            - abs(cv_linear.slope_of_regression - 1.0)
        )
        linear_slope_deviation = abs(cv_linear.slope_of_regression - 1.0)
        constant_slope_deviation = abs(cv_constant.slope_of_regression - 1.0)
        choose_linear = (
            cv_linear.rmse < cv_constant.rmse
            and linear_slope_deviation <= self.auto_drift_max_slope_deviation
            and linear_slope_deviation <= constant_slope_deviation + 1e-12
            and (
                rmse_improvement >= self.auto_drift_min_rmse_improvement
                or slope_improvement >= self.auto_drift_min_slope_improvement
            )
        )

        if method_constant == method_linear == "estimator_spatial_kfold":
            selection_method = "estimator_spatial_kfold"
        elif method_constant == method_linear == "fast_loo":
            selection_method = "fast_loo"
        else:
            selection_method = "estimator_spatial_kfold_with_fast_loo_fallback"

        return {
            "requested_drift_type": self.drift_type,
            "selection_method": selection_method,
            "subset_size": int(len(subset_values)),
            "constant": {
                "rmse": float(cv_constant.rmse),
                "r2": float(cv_constant.r_squared),
                "slope": float(cv_constant.slope_of_regression),
                "method": method_constant,
            },
            "linear": {
                "rmse": float(cv_linear.rmse),
                "r2": float(cv_linear.r_squared),
                "slope": float(cv_linear.slope_of_regression),
                "method": method_linear,
            },
            "rmse_improvement": float(rmse_improvement),
            "slope_improvement": float(slope_improvement),
            "constant_slope_deviation": float(constant_slope_deviation),
            "linear_slope_deviation": float(linear_slope_deviation),
            "choose_linear": bool(choose_linear),
        }

    def _resolve_effective_drift_type(
        self,
        coords: np.ndarray,
        values: np.ndarray,
    ) -> str:
        """Resolve requested drift configuration to a concrete run-time model."""
        requested = self.drift_type.lower()
        if requested not in {"none", "constant", "linear", "auto"}:
            raise ValueError(
                "drift_type must be 'none', 'constant', 'linear', or 'auto'",
            )
        if requested in {"none", "constant", "linear"}:
            self._trend_diagnostics = {
                "requested_drift_type": requested,
                "effective_drift_type": requested,
                "selection_method": "manual",
                "auto_selected": False,
            }
            return requested

        try:
            trend = self._evaluate_drift_candidates(
                coords,
                values,
                use_estimator_cv=(requested == "auto"),
            )
        except Exception as exc:
            logger.warning(
                "DRIFT: automatic drift comparison failed (%s). Falling back to %s drift.",
                exc,
                "constant" if requested == "auto" else requested,
            )
            effective = "constant" if requested == "auto" else requested
            self._trend_diagnostics = {
                "requested_drift_type": requested,
                "effective_drift_type": effective,
                "selection_method": "fallback",
                "auto_selected": False,
                "failure": str(exc),
            }
            return effective

        suggested = "linear" if trend["choose_linear"] else "constant"
        if requested == "auto":
            effective = suggested
            if effective == "linear":
                logger.warning(
                    "DRIFT: auto-selected linear drift from estimator-faithful CV "
                    "(constant rmse=%.4f, linear rmse=%.4f, improvement=%.1f%%, "
                    "constant slope=%.3f, linear slope=%.3f).",
                    trend["constant"]["rmse"],
                    trend["linear"]["rmse"],
                    100.0 * trend["rmse_improvement"],
                    trend["constant"]["slope"],
                    trend["linear"]["slope"],
                )
            else:
                logger.info(
                    "DRIFT: auto-selected constant drift from estimator-faithful CV "
                    "(linear improvement %.1f%%, slope improvement %.3f).",
                    100.0 * trend["rmse_improvement"],
                    trend["slope_improvement"],
                )
        else:
            effective = requested
            if suggested == "linear":
                logger.warning(
                    "DRIFT: constant drift was requested, but estimator-faithful CV indicates "
                    "linear drift is materially better (constant rmse=%.4f, linear "
                    "rmse=%.4f, improvement=%.1f%%, constant slope=%.3f, linear "
                    "slope=%.3f). Consider drift_type='auto' or 'linear'.",
                    trend["constant"]["rmse"],
                    trend["linear"]["rmse"],
                    100.0 * trend["rmse_improvement"],
                    trend["constant"]["slope"],
                    trend["linear"]["slope"],
                )

        self._trend_diagnostics = {
            **trend,
            "effective_drift_type": effective,
            "auto_selected": requested == "auto",
        }
        return effective

    def _global_anisotropy_transform(self) -> np.ndarray:
        """Transform matrix for global anisotropic search space."""
        R_global = rotation_matrix(self.azimuth, self.dip, self.pitch)
        S_global = scale_matrix(self.range_max, self.range_mid, self.range_min)
        return np.ascontiguousarray(S_global @ R_global, dtype=np.float64)

    def _transform_for_search(self, coords: np.ndarray) -> np.ndarray:
        """Project coordinates into the global anisotropic search space."""
        T = self._global_anisotropy_transform()
        return np.asarray(coords, dtype=np.float64) @ T.T

    def _compute_footprint_clip_mask(
        self,
        sample_coords: np.ndarray,
        block_centroids: np.ndarray,
    ) -> np.ndarray:
        """Mask blocks outside the buffered drillhole footprint."""
        blocks = np.asarray(block_centroids, dtype=np.float64)
        if not self.clip_to_drill_footprint or len(blocks) == 0:
            return np.ones(len(blocks), dtype=bool)

        samples = np.asarray(sample_coords, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) == 0:
            return np.ones(len(blocks), dtype=bool)

        search_samples = self._transform_for_search(samples)
        search_blocks = self._transform_for_search(blocks)
        tree = cKDTree(search_samples)
        nn_count = tree.query_ball_point(
            search_blocks,
            r=max(self.footprint_buffer_ranges, 1e-12),
            return_length=True,
        )
        return np.asarray(nn_count, dtype=np.int32) > 0

    def _reporting_block_mask_for_output(self, n_blocks: int) -> np.ndarray:
        """Resolve the reporting model mask for audit and QA summaries."""
        mask = getattr(self, "_reporting_block_mask", None)
        if mask is None or len(mask) != n_blocks:
            return np.ones(n_blocks, dtype=bool)
        return np.asarray(mask, dtype=bool)

    def _reporting_class_summary(self, class_names: np.ndarray) -> Dict[str, int]:
        """Summarise classification on the reporting model only."""
        names = np.asarray(class_names, dtype=object)
        mask = self._reporting_block_mask_for_output(len(names))
        scoped = names[mask]
        return {
            "Measured": int(np.sum(scoped == "Measured")),
            "Indicated": int(np.sum(scoped == "Indicated")),
            "Inferred": int(np.sum(scoped == "Inferred")),
            "Unclassified": int(np.sum(scoped == "Unclassified")),
        }

    def _within_block_variance_cached(
        self,
        block_dims: np.ndarray,
        variogram_params: Dict[str, Any],
        R_global: np.ndarray,
        n_discretisation: int,
    ) -> float:
        """Cache within-block variance for repeated support calculations."""
        support_density = self._support_discretisation_density(n_discretisation)
        if support_density <= 1:
            return 0.0

        block_dims = np.asarray(block_dims, dtype=np.float64).ravel()
        ratio_mid = self.range_mid / max(self.range_max, 1e-12)
        ratio_min = self.range_min / max(self.range_max, 1e-12)
        S = scale_matrix(
            variogram_params["range_"],
            variogram_params["range_"] * ratio_mid,
            variogram_params["range_"] * ratio_min,
        )
        key = (
            tuple(np.round(block_dims, 8).tolist()),
            int(support_density),
            variogram_params["kernel_type"],
            float(variogram_params["alpha"]),
            float(variogram_params["sill"]),
            float(variogram_params["nugget"]),
            float(variogram_params["range_"]),
            float(self.azimuth),
            float(self.dip),
            float(self.pitch),
        )
        if key not in self._sigma_w_sq_cache:
            self._sigma_w_sq_cache[key] = within_block_variance(
                block_dims,
                kernel_type=variogram_params["kernel_type"],
                alpha=variogram_params["alpha"],
                sill=variogram_params["sill"],
                range_=variogram_params["range_"],
                nugget=variogram_params["nugget"],
                n_discretisation=support_density,
                R=R_global,
                S=S,
            )
        return self._sigma_w_sq_cache[key]

    def _factorise_kernel_system(
        self,
        local_coords: np.ndarray,
        local_values: np.ndarray,
        variogram_params: LocalVariogramResult,
        R_global: np.ndarray,
        S: np.ndarray,
        context: str,
        drift_type: Optional[str] = None,
    ) -> Tuple[object, np.ndarray, np.ndarray, float, float]:
        """Factorise one kernel system with explicit stabilization retries."""
        active_drift = drift_type or self._active_drift_type()
        total_variance = max(
            float(variogram_params.sill + variogram_params.nugget),
            1e-12,
        )
        requested_accuracy = max(float(self.accuracy), 0.0)
        numerical_floor = np.finfo(np.float64).eps * total_variance
        base_accuracy = max(requested_accuracy, numerical_floor)
        max_accuracy = max(
            base_accuracy,
            requested_accuracy,
            total_variance * self.max_auto_regularization_fraction,
        )
        n_local = int(local_coords.shape[0])

        attempts = [base_accuracy]
        while len(attempts) < 6:
            next_accuracy = min(max_accuracy, attempts[-1] * 10.0)
            if next_accuracy <= attempts[-1] * (1.0 + 1e-12):
                break
            attempts.append(next_accuracy)
            if next_accuracy >= max_accuracy * (1.0 - 1e-12):
                break

        # Assemble the geometry-dependent kernel system once.  Stabilization
        # retries only change the diagonal regularization term, so rebuilding
        # pairwise distances and polynomial blocks for every attempt wastes
        # most of the runtime on difficult neighbourhoods.
        K_aug_base, _ = assemble_kernel_matrix(
            local_coords,
            kernel_type=variogram_params.kernel_type,
            alpha=variogram_params.alpha,
            sill=variogram_params.sill,
            range_=variogram_params.range_,
            nugget=variogram_params.nugget,
            accuracy=0.0,
            R=R_global,
            S=S,
            orientation_field=self._orientation_field if self.use_lva else None,
            drift_type=active_drift,
            use_geodesic=self.use_geodesic,
        )
        diag_idx = np.arange(n_local, dtype=np.intp)

        last_error: Optional[Exception] = None
        for attempt_index, attempt_accuracy in enumerate(attempts, start=1):
            K_aug = K_aug_base.copy()
            if attempt_accuracy > 0.0:
                K_aug[diag_idx, diag_idx] += attempt_accuracy
            try:
                fact, weights, poly_coeffs = factorise_and_solve(
                    K_aug, local_values, drift_type=active_drift,
                )
            except np.linalg.LinAlgError as exc:
                last_error = exc
                logger.warning(
                    "%s: factorisation failed with accuracy=%.2e "
                    "(attempt %d/%d).",
                    context, attempt_accuracy, attempt_index, len(attempts),
                )
                continue

            w_max = float(np.max(np.abs(weights))) if len(weights) > 0 else 0.0
            is_stable = w_max <= self.weight_sanity_limit
            if is_stable or attempt_index == len(attempts):
                if attempt_accuracy > requested_accuracy * (1.0 + 1e-12):
                    logger.warning(
                        "%s: stabilized with accuracy=%.2e "
                        "(requested %.2e, |weights|_max=%.1f).",
                        context, attempt_accuracy, requested_accuracy, w_max,
                    )
                    self._stabilization_events.append(
                        {
                            "context": context,
                            "requested_accuracy": requested_accuracy,
                            "applied_accuracy": attempt_accuracy,
                            "weight_max": w_max,
                            "attempt": attempt_index,
                        }
                    )
                return fact, weights, poly_coeffs, attempt_accuracy, w_max

            logger.warning(
                "%s: |weights|_max=%.1f exceeds limit %.1f at accuracy=%.2e. "
                "Retrying with stronger diagonal regularization.",
                context, w_max, self.weight_sanity_limit, attempt_accuracy,
            )

        if last_error is not None:
            raise last_error
        raise np.linalg.LinAlgError(f"{context}: factorisation failed")

    def _build_child_config(self, for_cv: bool = False) -> Dict[str, Any]:
        """Clone the estimator configuration for child runs."""
        keys = [
            "kernel_type", "alpha", "drift_type", "nugget", "accuracy",
            "auto_drift_cv_max_samples", "auto_drift_min_rmse_improvement",
            "auto_drift_min_slope_improvement",
            "auto_drift_max_slope_deviation",
            "weight_sanity_limit", "max_auto_regularization_fraction",
            "n_subdomains", "subdomain_method", "max_samples", "min_samples",
            "overlap_factor", "estimation_mode", "local_search_radii",
            "max_samples_per_octant",
            "pum_threshold", "azimuth", "dip", "pitch",
            "range_max", "range_mid", "range_min", "use_lva", "lva_source",
            "use_normal_score", "use_ilr", "variogram_mode", "sill",
            "change_of_support", "change_of_support_mode",
            "classification_thresholds",
            "geometric_criteria", "discretisation_mode",
            "discretisation_density", "local_mean_radius",
            "decluster_cell_size", "rotation_convention", "use_geodesic",
            "parallel", "n_workers", "verbose", "clip_min", "clip_max",
            "run_cv", "cv_max_samples", "cv_mode", "cv_folds",
            "allow_cv_fallback", "seed",
            "operator", "prefilter_blocks", "domain_policy",
            "clip_to_drill_footprint", "footprint_buffer_ranges",
            "mask_uninformed_ns_blocks",
        ]
        cfg = {k: getattr(self, k) for k in keys}
        if for_cv:
            cfg.update(
                {
                    "drift_type": self._active_drift_type(),
                    "run_cv": False,
                    # CoS is disabled for CV because held-out "blocks" are
                    # sample points with zero volume — block-support
                    # correction would be meaningless.
                    "change_of_support": False,
                    "discretisation_density": 1,
                    "clip_min": None,
                    "clip_max": None,
                    "verbose": False,
                    "prefilter_blocks": False,
                    "clip_to_drill_footprint": False,
                    "footprint_buffer_ranges": 1.0,
                    "mask_uninformed_ns_blocks": False,
                    # Preserve domain policy from parent so that CV
                    # metrics reflect the same population boundaries
                    # used in the actual estimation.  The old
                    # "domain_policy": "ignore" caused CV to pool all
                    # domains, inflating smoothing error on multi-domain
                    # deposits and misaligning CV metrics with estimation.
                    "domain_policy": self.domain_policy,
                }
            )
        return cfg

    def _select_spatial_cv_indices(
        self,
        coords: np.ndarray,
        max_samples: int,
    ) -> np.ndarray:
        """Deterministic farthest-point subset for spatial CV."""
        search_coords = self._transform_for_search(coords)
        n_total = search_coords.shape[0]
        if max_samples <= 0 or n_total <= max_samples:
            return np.arange(n_total, dtype=np.intp)

        n_keep = min(max_samples, n_total)
        selected = np.empty(n_keep, dtype=np.intp)
        centre = np.median(search_coords, axis=0)
        selected[0] = int(np.argmin(np.linalg.norm(search_coords - centre, axis=1)))
        min_dist = np.linalg.norm(search_coords - search_coords[selected[0]], axis=1)

        for i in range(1, n_keep):
            selected[i] = int(np.argmax(min_dist))
            d = np.linalg.norm(search_coords - search_coords[selected[i]], axis=1)
            min_dist = np.minimum(min_dist, d)

        return np.unique(selected)

    def _assign_spatial_folds(
        self,
        coords: np.ndarray,
        n_folds: int,
    ) -> np.ndarray:
        """Assign samples to spatial folds using horizontal (XY) blocking.

        For drillhole data, 3D Voronoi folds remove entire vertical columns
        (drillholes).  With short vertical ranges (common in tabular/vein
        deposits), this forces vertical extrapolation and produces
        misleadingly poor CV metrics.

        Instead, we block horizontally (XY only): each fold removes a
        horizontal neighbourhood of drillholes but every fold retains
        composites at ALL depth levels.  This tests at the between-
        drillhole spacing, which is the relevant scale for block estimation.

        The method detects whether the data is drillhole-like by comparing
        vertical extent to the vertical range.  If vertical extent > 3×
        vertical range (common for drillholes), it uses XY-only blocking.
        Otherwise it falls back to full 3D blocking.
        """
        search_coords = self._transform_for_search(coords)
        n_samples = search_coords.shape[0]
        if n_samples <= 1 or n_folds <= 1:
            return np.zeros(n_samples, dtype=np.int32)

        n_folds = min(n_folds, n_samples)

        # Detect drillhole geometry: large vertical extent relative to
        # vertical range means data is arranged in vertical columns.
        z_extent = float(np.max(coords[:, 2]) - np.min(coords[:, 2]))
        range_min = getattr(self, "range_min", None)
        use_2d = (
            range_min is not None
            and range_min > 0
            and z_extent > 3.0 * range_min
        )

        if use_2d:
            # XY-only blocking in anisotropic space (ignore Z component)
            block_coords = search_coords[:, :2]
        else:
            block_coords = search_coords

        # Farthest-point centre selection
        n_keep = min(n_folds, n_samples)
        centre = np.median(block_coords, axis=0)
        selected = [int(np.argmin(np.linalg.norm(block_coords - centre, axis=1)))]
        min_dist = np.linalg.norm(block_coords - block_coords[selected[0]], axis=1)
        for _ in range(1, n_keep):
            idx = int(np.argmax(min_dist))
            selected.append(idx)
            d = np.linalg.norm(block_coords - block_coords[idx], axis=1)
            min_dist = np.minimum(min_dist, d)
        centres = block_coords[np.array(selected)]

        dists = np.linalg.norm(
            block_coords[:, np.newaxis, :] - centres[np.newaxis, :, :], axis=2,
        )
        labels = np.argmin(dists, axis=1).astype(np.int32)

        # Guarantee that every fold has at least one sample.
        unique = np.unique(labels)
        if len(unique) != n_folds:
            order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
            labels = np.zeros(n_samples, dtype=np.int32)
            labels[order] = np.arange(n_samples, dtype=np.int32) % n_folds

        return labels

    def _auto_cv_folds(self, coords: np.ndarray, user_folds: int) -> int:
        """Choose fold count so that fold diameter ≈ variogram range.

        If the user-requested fold count would produce spatial blocks
        larger than the major range, we increase the fold count so that
        CV tests interpolation ability (not extrapolation).

        ``_transform_for_search`` already normalises coordinates by
        range (via ``scale_matrix = diag(1/a_max, 1/a_mid, 1/a_min)``),
        so distances in transformed space are in "range units".  We want
        the average fold diameter in those units to be ≤ 1.0:

            fold_diam ≈ max_extent_ru / k^(1/3) ≤ 1.0
            → k_min ≈ max_extent_ru^3

        Clamped to [user_folds, 20] to stay practical.
        """
        n_samples = coords.shape[0]
        if n_samples < 10:
            return max(2, min(user_folds, n_samples))

        search_coords = self._transform_for_search(coords)
        extent = np.max(search_coords, axis=0) - np.min(search_coords, axis=0)  # already in range units

        # For drillhole data, use horizontal (XY) extent only.
        # Vertical extent inflates fold count but vertical blocking
        # is handled by _assign_spatial_folds via 2D mode.
        z_extent = float(np.max(coords[:, 2]) - np.min(coords[:, 2]))
        range_min = getattr(self, "range_min", None)
        if range_min and range_min > 0 and z_extent > 3.0 * range_min:
            max_extent_ru = float(np.max(extent[:2]))  # XY only
        else:
            max_extent_ru = float(np.max(extent))

        if max_extent_ru <= 0:
            return max(2, min(user_folds, n_samples))

        # fold_diam ≈ max_extent_ru / k^(1/3) ≤ 1.0
        n_folds_min = int(np.ceil(max_extent_ru ** 3))
        n_folds = max(user_folds, n_folds_min)
        n_folds = min(n_folds, 20, n_samples // max(self.min_samples, 2))
        n_folds = max(2, n_folds)

        if n_folds != user_folds:
            logger.info(
                "Spatial CV: auto-tuned folds from %d to %d "
                "(extent=%.1f range-units, target fold diam ≤ 1 range).",
                user_folds, n_folds, max_extent_ru,
            )
        return n_folds

    def _run_spatial_kfold_cv(self) -> Optional[CVResult]:
        """Estimator-faithful spatial k-fold CV in original units."""
        coords_all = self._composite_coords
        values_all = self._composite_values
        if coords_all is None or values_all is None or len(values_all) < 2:
            return None

        subset_idx = self._select_spatial_cv_indices(coords_all, self.cv_max_samples)
        coords = coords_all[subset_idx]
        values = values_all[subset_idx]

        # Subset domain labels if available — needed for domain-faithful CV
        subset_domains = None
        if self._composite_domains is not None:
            subset_domains = self._composite_domains[subset_idx]

        if len(values) < 2:
            return None

        # Auto-tune fold count so that fold diameter ≈ variogram range.
        # With 5 folds over a 500m deposit and range=80m, each fold covers
        # ~200m (2.5× range), turning CV into an extrapolation test instead
        # of interpolation.  We want fold diameter ≤ range so that the
        # held-out points are within reach of training data.
        n_folds = self._auto_cv_folds(coords, self.cv_folds)
        labels = self._assign_spatial_folds(coords, n_folds)

        actual_parts: List[np.ndarray] = []
        estimate_parts: List[np.ndarray] = []
        folds_failed = 0

        for fold in range(n_folds):
            test_mask = labels == fold
            train_mask = ~test_mask
            if not np.any(test_mask) or np.sum(train_mask) < self.min_samples:
                folds_failed += 1
                continue

            try:
                child = ARBFEstimator(self._build_child_config(for_cv=True))
                child.set_composites(
                    coords[train_mask], values[train_mask],
                )
                child.set_block_model(
                    coords[test_mask],
                    np.zeros(3, dtype=np.float64),
                )

                # Pass domain labels so the child respects geological
                # boundaries during CV, matching actual estimation behavior.
                if subset_domains is not None:
                    child.set_domains(
                        subset_domains[train_mask],
                        subset_domains[test_mask],
                    )

                if (
                    self._orientation_field is not None
                    and self.lva_source != "data"
                ):
                    child.set_orientation_field(self._orientation_field)

                if self._declustering_weights is not None:
                    dw = self._declustering_weights[subset_idx][train_mask]
                    dw_sum = float(np.sum(dw))
                    if dw_sum > 0:
                        child.set_declustering_weights(dw / dw_sum)

                fold_result = child.estimate()
                pred = np.asarray(fold_result.grades, dtype=np.float64)
                actual = values[test_mask]
                finite = np.isfinite(pred)
                if not np.any(finite):
                    logger.warning(
                        "Spatial CV fold %d/%d produced no finite predictions.",
                        fold + 1, n_folds,
                    )
                    folds_failed += 1
                    continue

                actual_parts.append(actual[finite])
                estimate_parts.append(pred[finite])
            except Exception as exc:
                logger.warning(
                    "Spatial CV fold %d/%d raised %s: %s — skipping fold.",
                    fold + 1, n_folds, type(exc).__name__, exc,
                )
                folds_failed += 1
                continue

        if not actual_parts:
            return None

        actual_all = np.concatenate(actual_parts)
        estimate_all = np.concatenate(estimate_parts)
        n_valid = len(actual_all)

        # Accept if at least 2 folds succeeded and we have >= 25% of samples.
        # The old 50% threshold was too aggressive for skewed distributions
        # where 1-2 folds can legitimately produce NaN on tail subsets.
        min_folds_ok = max(2, n_folds - folds_failed)
        min_valid = max(2 * min_folds_ok, int(np.ceil(0.25 * len(values))))
        if n_valid < min_valid:
            logger.warning(
                "Spatial CV retained only %d/%d valid held-out predictions "
                "(%d/%d folds failed). Below acceptance threshold (%d); "
                "falling back to fast_loo.",
                n_valid, len(values), folds_failed, n_folds, min_valid,
            )
            return None
        if folds_failed > 0:
            logger.info(
                "Spatial CV: %d/%d folds succeeded (%d valid predictions).",
                n_folds - folds_failed, n_folds, n_valid,
            )
        return _compute_cv_statistics(actual_all, estimate_all)

    def _run_cross_validation(
        self,
        transformed_coords: np.ndarray,
        transformed_values: np.ndarray,
    ) -> Optional[CVResult]:
        """Run the configured cross-validation workflow."""
        cv_mode = (self.cv_mode or "spatial_kfold").lower()
        self._cv_execution_mode = cv_mode

        if cv_mode == "fast_loo":
            global_vp = self._get_global_variogram_params()
            self._cv_execution_mode = "fast_loo"
            return leave_one_out_cv(
                transformed_coords,
                transformed_values,
                kernel_type=global_vp["kernel_type"],
                alpha=global_vp["alpha"],
                sill=global_vp["sill"],
                range_=global_vp["range_"],
                range_mid=self.range_mid,
                range_min=self.range_min,
                nugget=global_vp["nugget"],
                accuracy=self.accuracy,
                drift_type=self._active_drift_type(),
                azimuth=self.azimuth,
                dip=self.dip,
                pitch=self.pitch,
                max_samples=self.cv_max_samples,
                progress_callback=self._progress_callback,
            )

        cv_result = self._run_spatial_kfold_cv()
        if cv_result is not None:
            self._cv_execution_mode = "spatial_kfold"
            return cv_result

        if not self.allow_cv_fallback:
            self._cv_execution_mode = "spatial_kfold_failed"
            logger.warning(
                "Spatial k-fold CV produced no valid folds and allow_cv_fallback=False. "
                "Leaving CV result unset instead of downgrading to fast_loo.",
            )
            return None

        logger.warning(
            "Spatial k-fold CV produced no valid folds. "
            "allow_cv_fallback=True -> falling back to fast_loo.",
        )

        global_vp = self._get_global_variogram_params()
        self._cv_execution_mode = "fast_loo_fallback"
        return leave_one_out_cv(
            transformed_coords,
            transformed_values,
            kernel_type=global_vp["kernel_type"],
            alpha=global_vp["alpha"],
            sill=global_vp["sill"],
            range_=global_vp["range_"],
            range_mid=self.range_mid,
            range_min=self.range_min,
            nugget=global_vp["nugget"],
            accuracy=self.accuracy,
            drift_type=self._active_drift_type(),
            azimuth=self.azimuth,
            dip=self.dip,
            pitch=self.pitch,
            max_samples=self.cv_max_samples,
            progress_callback=self._progress_callback,
        )

    def _estimate_blocks(
        self,
        coords: np.ndarray,
        R_global: np.ndarray,
    ) -> BlendedResult:
        """Estimate all blocks.

        Single-domain mode: direct GPR prediction (no blending).
        PUM mode: two-pass blending with Wendland weights.

        Both modes use block discretisation for mean and centroid-only
        variance for efficiency.
        """
        centroids = self._block_centroids
        block_sizes = self._block_sizes

        B = centroids.shape[0]
        if block_sizes.ndim == 1:
            bs = block_sizes
        else:
            bs = block_sizes[0]

        # ── SPEED FIX: Pre-filter blocks within geometric-mean range of any
        # composite.  Using max range includes too many blocks when anisotropy
        # is high (e.g. 312m major with 54m minor).  The geometric mean gives
        # a balanced isotropic equivalent.
        footprint_mask = self._compute_footprint_clip_mask(coords, centroids)
        self._reporting_block_mask = (
            footprint_mask.copy() if self.clip_to_drill_footprint else None
        )
        n_within_footprint = int(np.sum(footprint_mask))
        if self.clip_to_drill_footprint:
            logger.info(
                "Footprint clip: %d / %d blocks (%.1f%%) within %.2f anisotropic "
                "range units of the drill footprint.",
                n_within_footprint,
                B,
                100.0 * n_within_footprint / max(B, 1),
                self.footprint_buffer_ranges,
            )

        search_radius = float((self.range_max * self.range_mid * self.range_min) ** (1.0 / 3.0))
        from scipy.spatial import cKDTree as _cKDTree_speed
        _data_tree = _cKDTree_speed(coords)
        _nn_count = _data_tree.query_ball_point(centroids, r=search_radius, return_length=True)
        support_mask = (
            np.asarray(_nn_count) >= max(self.min_samples, 1)
            if self.prefilter_blocks
            else np.ones(B, dtype=bool)
        )
        _active_mask = footprint_mask & support_mask
        _n_active = int(np.sum(_active_mask))
        _n_skipped = B - _n_active

        if _n_active == 0:
            raise ValueError(
                "No blocks remain after applying the current footprint / search filters. "
                "Relax footprint_buffer_ranges or disable footprint clipping.",
            )

        if _n_skipped > 0:
            logger.info(
                "SPEED: Pre-filtered %d / %d blocks (%.1f%%) within %.0f m "
                "of data. Skipping %d blocks with no nearby composites.",
                _n_active, B, 100.0 * _n_active / B,
                search_radius, _n_skipped,
            )
            # Store originals and active mask for post-estimation expansion
            self._block_active_mask = _active_mask
            self._block_centroids_full = self._block_centroids
            self._block_sizes_full = self._block_sizes
            # Swap to filtered versions — all downstream code (PUM blending,
            # single-domain prediction) uses self._block_centroids/sizes
            self._block_centroids = centroids[_active_mask]
            if block_sizes.ndim == 2:
                self._block_sizes = block_sizes[_active_mask]
            centroids = self._block_centroids
            block_sizes = self._block_sizes
            B = centroids.shape[0]  # update B to filtered count
        else:
            self._block_active_mask = None
            self._block_centroids_full = None
            self._block_sizes_full = None

        n_disc = self._estimation_discretisation_density()
        if n_disc != self.discretisation_density:
            logger.info(
                "Change-of-support: estimation discretisation upgraded from %d to %d.",
                self.discretisation_density, n_disc,
            )

        if self.estimation_mode == "local_neighbourhood_gpr":
            return self._estimate_blocks_local_neighbourhood(coords, R_global)

        # ── SPEED: PUM and single-domain paths use centroid-only prediction.
        # Disc-point means ≈ centroid means for smooth RBF fields (difference
        # < 1% for typical block sizes).  Change-of-support variance is
        # computed analytically via _within_block_variance_cached, not from
        # disc points.  Skipping the B×n_pts predict_mean call eliminates
        # ~96% of kernel evaluations (27× fewer queries at n_disc=27).

        if getattr(self, '_single_domain', False):
            # ---- Single-domain: direct GPR prediction (centroid-only) ----
            sd = self._subdomains[0]
            vp = sd.variogram_params
            R = R_global if R_global is not None else np.eye(3)
            S = sd.scale_matrix_
            lva_field = self._orientation_field if self.use_lva else None

            N = coords.shape[0]

            logger.info(
                "Single-domain prediction: using all %d samples per block "
                "(B=%d, centroid-only → %.1fM kernel evaluations)",
                N, B, B * N / 1e6,
            )
            # Centroid-only mean — fused Numba path, O(B×N) with zero
            # allocation.  Block mean ≈ centroid mean for smooth RBF fields.
            block_grades = predict_mean(
                centroids,
                coords,
                sd.weights,
                sd.poly_coeffs,
                kernel_type=vp.kernel_type,
                alpha=vp.alpha,
                sill=vp.sill,
                range_=vp.range_,
                R=R,
                S=S,
                drift_type=self._active_drift_type(),
                orientation_field=lva_field,
            )

            s2_centroids = predict_variance(
                centroids,
                coords,
                sd.cholesky_factor,
                kernel_type=vp.kernel_type,
                alpha=vp.alpha,
                sill=vp.sill,
                range_=vp.range_,
                nugget=vp.nugget,
                R=R,
                S=S,
                drift_type=self._active_drift_type(),
                orientation_field=lva_field,
                l_inv=sd.l_inv,
            )

            # Block kriging variance: GPR posterior at centroid minus the
            # theoretical within-block average variogram γ̄(V,V).
            if n_disc > 1 and self.change_of_support:
                sigma_w_sq = self._within_block_variance_cached(
                    bs,
                    {
                        "kernel_type": vp.kernel_type,
                        "alpha": vp.alpha,
                        "sill": vp.sill,
                        "range_": vp.range_,
                        "nugget": vp.nugget,
                    },
                    R,
                    n_disc,
                )
                block_variances = np.maximum(s2_centroids - sigma_w_sq, 0.0)
            else:
                block_variances = s2_centroids

            return BlendedResult(
                estimates=block_grades,
                variances=block_variances,
                n_active_subdomains=np.ones(B, dtype=np.int32),
                within_variance=block_variances,
                between_variance=np.zeros(B, dtype=np.float64),
                total_variance=block_variances,
            )

        # ---- PUM sub-domain path — centroid-only single-pass ----
        # SPEED: predict_mean_and_variance at centroids only.  Block mean ≈
        # centroid mean for smooth RBF fields (eliminates B×n_disc disc-point
        # predict_mean calls — 27× fewer kernel evaluations at default density).
        lva_field = self._orientation_field if self.use_lva else None

        _sd_centres = np.array([sd.centre for sd in self._subdomains])
        _sd_radii = np.array([sd.radius for sd in self._subdomains])

        # (B, K) Euclidean distance from each centroid to each subdomain centre
        _dists = np.linalg.norm(
            centroids[:, np.newaxis, :] - _sd_centres[np.newaxis, :, :], axis=2,
        )
        _active_mask = (_dists < _sd_radii[np.newaxis, :]).copy()

        # Activate nearest subdomain for blocks outside all radii
        _outside = ~np.any(_active_mask, axis=1)
        if np.any(_outside):
            _nearest = np.argmin(_dists[_outside], axis=1)
            _active_mask[np.where(_outside)[0], _nearest] = True

        _keys = _build_group_keys(_active_mask)  # Numba, O(B × K)
        _unique_keys = np.unique(_keys)

        block_grades = np.zeros(B, dtype=np.float64)
        _total_var = np.zeros(B, dtype=np.float64)
        _n_active = np.zeros(B, dtype=np.int32)
        _within_var = np.zeros(B, dtype=np.float64)
        _between_var = np.zeros(B, dtype=np.float64)

        _R = R_global if R_global is not None else np.eye(3)

        for _key in _unique_keys:
            _bidx = np.where(_keys == _key)[0]      # block indices in this group
            _aids = np.where(_active_mask[_bidx[0]])[0]  # active subdomain indices
            _nb = len(_bidx)

            # Wendland PUM weights at centroids
            _adists = _dists[np.ix_(_bidx, _aids)]  # (_nb, K_active)
            _psi = np.zeros_like(_adists)
            for _ai, _sdx in enumerate(_aids):
                _psi[:, _ai] = wendland_c2_weight_batch(
                    _adists[:, _ai], self._subdomains[_sdx].radius,
                )
            _psi_sum = np.maximum(np.sum(_psi, axis=1, keepdims=True), 1e-15)
            _wk = _psi / _psi_sum
            _zero_wt = (_psi_sum.ravel() < 1e-15)
            if np.any(_zero_wt):
                _wk[_zero_wt] = 1.0 / len(_aids)

            _cent_batch = centroids[_bidx]

            _f_cent = np.zeros((_nb, len(_aids)), dtype=np.float64)
            _s2 = np.zeros((_nb, len(_aids)), dtype=np.float64)

            for _ai, _sdx in enumerate(_aids):
                _sd = self._subdomains[_sdx]
                if _sd.weights is None or _sd.cholesky_factor is None \
                        or _sd.variogram_params is None:
                    continue
                _vp = _sd.variogram_params
                _S = _sd.scale_matrix_ if _sd.scale_matrix_ is not None \
                    else scale_matrix(_vp.range_, _vp.range_, _vp.range_)
                _lc = coords[_sd.sample_indices]

                # Mean + variance at centroids (L_inv cached at fit time → dgemm)
                _fc, _s2c = predict_mean_and_variance(
                    _cent_batch, _lc, _sd.weights, _sd.poly_coeffs,
                    _sd.cholesky_factor,
                    kernel_type=_vp.kernel_type, alpha=_vp.alpha,
                    sill=_vp.sill, range_=_vp.range_, nugget=_vp.nugget,
                    R=_R, S=_S, drift_type=self._active_drift_type(),
                    orientation_field=lva_field,
                    l_inv=_sd.l_inv,
                )
                _f_cent[:, _ai] = _fc
                _s2[:, _ai] = _s2c

            # Blend centroid means → block grade
            _fblend = np.sum(_wk * _f_cent, axis=1)
            block_grades[_bidx] = _fblend
            _n_active[_bidx] = len(_aids)

            # Blend variance (within + between)
            _within = np.sum(_wk * _s2, axis=1)
            _between = np.sum(_wk * (_f_cent - _fblend[:, np.newaxis]) ** 2, axis=1)
            _total_var[_bidx] = _within + _between
            _within_var[_bidx] = _within
            _between_var[_bidx] = _between

        # Wrap results in BlendedResult to reuse the variance/classification logic below
        blended_var = BlendedResult(
            estimates=block_grades,
            variances=clamp_variance(_within_var),
            n_active_subdomains=_n_active,
            within_variance=clamp_variance(_within_var),
            between_variance=_between_var,
            total_variance=clamp_variance(_total_var),
        )

        # Block kriging variance: subtract the theoretical within-block
        # average variogram γ̄(V,V) from the centroid GPR posterior variance.
        # This is the correct Journel & Huijbregts block kriging variance
        # reduction, replacing the statistically incoherent old formula
        # (centroid variance + dispersion of smoothed estimates).
        if n_disc > 1 and self.change_of_support:
            global_vp = self._get_global_variogram_params()
            sigma_w_sq = self._within_block_variance_cached(
                bs, global_vp, _R, n_disc,
            )
            block_variances = np.maximum(blended_var.variances - sigma_w_sq, 0.0)
            within_var = np.maximum(blended_var.within_variance - sigma_w_sq, 0.0)
        else:
            block_variances = blended_var.variances
            within_var = blended_var.within_variance
        total_var = within_var + blended_var.between_variance

        return BlendedResult(
            estimates=block_grades,
            variances=within_var,
            n_active_subdomains=blended_var.n_active_subdomains,
            within_variance=within_var,
            between_variance=blended_var.between_variance,
            total_variance=clamp_variance(total_var),
        )

    @staticmethod
    def _octant_codes(differences: np.ndarray) -> np.ndarray:
        """Encode 3D signs into octant codes in the anisotropic search space."""
        differences = np.asarray(differences, dtype=np.float64)
        if differences.size == 0:
            return np.empty(0, dtype=np.int8)
        return (
            (differences[:, 0] >= 0.0).astype(np.int8) * 4
            + (differences[:, 1] >= 0.0).astype(np.int8) * 2
            + (differences[:, 2] >= 0.0).astype(np.int8)
        )

    def _required_local_octants(self, drift_type: Optional[str] = None) -> int:
        """Minimum octant coverage required for a stable local neighbourhood."""
        active_drift = (drift_type or self._active_drift_type()).lower()
        if active_drift == "linear":
            return int(self.search_min_octants_linear)
        return int(self.search_min_octants)

    def _trim_local_candidates(
        self,
        idx: np.ndarray,
        transformed_coords: np.ndarray,
        query_point: np.ndarray,
    ) -> np.ndarray:
        """Trim local candidates while preserving directional balance."""
        idx = np.asarray(idx, dtype=np.intp)
        if len(idx) == 0:
            return idx

        diffs = transformed_coords[idx] - query_point
        dists = np.linalg.norm(diffs, axis=1)
        if self._search_selection_weights is not None:
            weights = np.asarray(
                self._search_selection_weights[idx], dtype=np.float64,
            )
        else:
            weights = np.ones(len(idx), dtype=np.float64)

        order = np.lexsort((-weights, dists))
        idx = idx[order]
        diffs = diffs[order]
        dists = dists[order]
        weights = weights[order]

        max_total = len(idx)
        if self.max_samples > 0:
            max_total = min(max_total, int(self.max_samples))
        if max_total <= 0:
            return np.empty(0, dtype=np.intp)

        if not self.balanced_neighbourhood_selection:
            return idx[:max_total]

        # Inline octant encoding — avoids function call + asarray overhead
        oct_codes = (
            (diffs[:, 0] >= 0.0).view(np.int8) * 4
            + (diffs[:, 1] >= 0.0).view(np.int8) * 2
            + (diffs[:, 2] >= 0.0).view(np.int8)
        )
        oct_positions = [
            np.flatnonzero(oct_codes == octant).astype(np.intp)
            for octant in range(8)
        ]
        occupied_octants = [
            octant for octant in range(8) if len(oct_positions[octant]) > 0
        ]
        if not occupied_octants:
            return idx[:max_total]

        if self.max_samples_per_octant > 0:
            seed_total = min(
                max_total,
                len(occupied_octants) * int(self.max_samples_per_octant),
            )
        else:
            seed_total = max_total
        if max_total >= len(idx):
            return idx

        weight_scale = max(float(np.max(weights)), 1e-12)
        target_octants = max(1, min(8, len(occupied_octants)))
        max_per_octant = int(np.ceil(seed_total / target_octants))
        if self.max_samples_per_octant > 0:
            max_per_octant = min(max_per_octant, self.max_samples_per_octant)
        max_per_octant = max(max_per_octant, 1)

        # The per-octant setting is used only to seed a directionally
        # balanced neighbourhood. After that, the remaining nearest
        # composites are appended so the local solve can reach max_samples
        # and preserve meaningful multi-sample interaction between holes.
        selected_positions_arr = _spread_select_jit(
            np.ascontiguousarray(diffs, dtype=np.float64),
            np.ascontiguousarray(dists, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(oct_codes, dtype=np.int8),
            seed_total,
            max_per_octant,
            weight_scale,
        )

        selected_positions = selected_positions_arr[
            np.argsort(dists[selected_positions_arr], kind="stable")
        ]

        if len(selected_positions) >= max_total:
            return idx[selected_positions[:max_total]]

        selected_mask = np.zeros(len(idx), dtype=bool)
        selected_mask[selected_positions] = True
        remaining_positions = np.flatnonzero(~selected_mask)
        n_extra = min(max_total - len(selected_positions), len(remaining_positions))
        if n_extra > 0:
            selected_positions = np.concatenate(
                [selected_positions, remaining_positions[:n_extra]],
            )

        return idx[selected_positions]

    def _select_local_neighbourhood(
        self,
        tree: cKDTree,
        transformed_coords: np.ndarray,
        query_point: np.ndarray,
        precomputed_candidates: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """Select the actual estimation neighbourhood for one block.

        If precomputed_candidates is provided (from batch KDTree query at
        max radius), the multi-pass search uses subset filtering instead
        of separate KDTree queries — avoids 3 KDTree calls per block.
        """
        selected = np.empty(0, dtype=np.intp)
        chosen_pass = len(self.local_search_radii)
        required_samples = (
            max(self.min_samples, 8)
            if self._active_drift_type() == "linear"
            else self.min_samples
        )
        required_octants = self._required_local_octants()
        if self.max_samples > 0:
            target_samples = min(
                int(self.max_samples),
                max(int(required_samples) * 4, 16),
            )
        else:
            target_samples = max(int(required_samples) * 4, 16)
        desired_octants = min(8, max(int(required_octants), 6))
        best_score = (-1, -1, -1, -1, -1, -len(self.local_search_radii) - 1)

        if precomputed_candidates is not None and len(precomputed_candidates) > 0:
            # Distances already known — filter by radius thresholds
            cand_dists = np.linalg.norm(
                transformed_coords[precomputed_candidates] - query_point, axis=1,
            )

        for pass_idx, radius in enumerate(self.local_search_radii, start=1):
            if precomputed_candidates is not None and len(precomputed_candidates) > 0:
                mask = cand_dists <= radius
                idx = precomputed_candidates[mask]
            else:
                idx = np.asarray(
                    tree.query_ball_point(query_point, r=radius), dtype=np.intp,
                )
            if len(idx) == 0:
                continue
            idx = self._trim_local_candidates(idx, transformed_coords, query_point)
            diffs = transformed_coords[idx] - query_point
            octant_count = int(np.unique(self._octant_codes(diffs)).size)
            richness = min(int(len(idx)), int(target_samples))
            score = (
                int(len(idx) >= required_samples and octant_count >= required_octants),
                int(octant_count >= desired_octants),
                min(int(octant_count), desired_octants),
                int(richness >= target_samples),
                richness,
                -int(pass_idx),
            )
            if score > best_score:
                best_score = score
                selected = idx
                chosen_pass = pass_idx

        # Blocks outside the configured search radii still need an estimate.
        # Fall back to the nearest composites so the local RBF posterior
        # regresses toward the drift mean instead of creating NaN islands.
        if len(selected) >= required_samples:
            return selected, chosen_pass

        if precomputed_candidates is not None and len(precomputed_candidates) > 0:
            fallback_idx = np.asarray(precomputed_candidates, dtype=np.intp)
        else:
            if self.max_samples > 0:
                # Cap fallback at max_samples — using 3x max_samples caused
                # catastrophic over-smoothing on sparse datasets (e.g. 900
                # out of 1,314 composites → near-constant global-mean estimate).
                k_fallback = min(
                    max(int(self.max_samples), max(required_samples * 4, 32)),
                    len(transformed_coords),
                )
            else:
                k_fallback = len(transformed_coords)
            _, knn_idx = tree.query(query_point, k=k_fallback)
            fallback_idx = np.asarray(np.atleast_1d(knn_idx), dtype=np.intp)
            fallback_idx = fallback_idx[fallback_idx < len(transformed_coords)]

        if len(fallback_idx) == 0:
            return selected, chosen_pass

        fallback_idx = self._trim_local_candidates(
            fallback_idx,
            transformed_coords,
            query_point,
        )
        if len(fallback_idx) == 0:
            return selected, chosen_pass

        return fallback_idx, chosen_pass

    def _estimate_blocks_local_neighbourhood(
        self,
        coords: np.ndarray,
        R_global: np.ndarray,
    ) -> BlendedResult:
        """Estimate blocks with a single global covariance and local search.

        Optimisations for large block models (>100k blocks):
        1. Pre-filter: batch KDTree query skips blocks with no samples
           within the maximum search radius — typically eliminates 30-70%
           of blocks for models extending beyond the drillhole envelope.
        2. Cache: blocks sharing identical sample neighbourhoods reuse
           the factorised kernel system (O(k^3) LU/Cholesky saved).
        3. Batch predict: blocks sharing the same cached neighbourhood
           are predicted together in a single vectorised call.
        4. Progress: logs every 5% for UI responsiveness.
        """
        values = (
            self._working_values
            if self._working_values is not None
            else self._composite_values
        )
        centroids = self._block_centroids
        block_sizes = self._block_sizes
        B = centroids.shape[0]

        transformed_coords = self._transform_for_search(coords)
        transformed_blocks = self._transform_for_search(centroids)
        tree = cKDTree(transformed_coords)

        global_vp = self._get_global_variogram_params()
        vp_local = LocalVariogramResult(
            sill=global_vp["sill"],
            nugget=global_vp["nugget"],
            range_=global_vp["range_"],
            alpha=global_vp["alpha"],
            kernel_type=global_vp["kernel_type"],
            fit_residual=0.0,
            n_pairs=0,
            n_lags=0,
        )
        ratio_mid = self.range_mid / max(self.range_max, 1e-12)
        ratio_min = self.range_min / max(self.range_max, 1e-12)
        S_global = scale_matrix(
            vp_local.range_,
            vp_local.range_ * ratio_mid,
            vp_local.range_ * ratio_min,
        )
        n_disc = self._estimation_discretisation_density()
        offsets = get_offsets(n_disc)
        n_disc = offsets.shape[0]  # actual point count (may differ from requested)

        estimates = np.full(B, np.nan, dtype=np.float64)
        variances = np.full(B, np.nan, dtype=np.float64)
        sample_counts = np.zeros(B, dtype=np.int32)
        octant_counts = np.zeros(B, dtype=np.int32)
        search_passes = np.full(B, len(self.local_search_radii), dtype=np.int32)

        # ── Pre-filter: skip blocks outside data support ──────────────
        # Use return_length=True to avoid allocating candidate lists for
        # every block (which can consume >1 GB for 934k+ block models).
        # The tiling phase does its own k-NN from tile centres.
        # transformed_coords / transformed_blocks already live in the
        # anisotropic search space, so local_search_radii are dimensionless
        # range units here and must not be rescaled by raw ranges again.
        search_radii = np.asarray(self.local_search_radii, dtype=np.float64)
        max_radius = float(search_radii[-1]) if len(search_radii) > 0 else 0.0
        t_prefilter = time.time()

        _data_extent = np.max(transformed_coords, axis=0) - np.min(transformed_coords, axis=0)
        _block_extent = np.max(transformed_blocks, axis=0) - np.min(transformed_blocks, axis=0)
        _volume_ratio = np.prod(np.maximum(_block_extent, 1.0)) / max(np.prod(np.maximum(_data_extent, 1.0)), 1.0)
        _use_prefilter = B > 5_000 or _volume_ratio > 1.5

        required_samples = (
            max(self.min_samples, 8)
            if self._active_drift_type() == "linear"
            else self.min_samples
        )
        if _use_prefilter:
            logger.info(
                "LGPR: support diagnostic for %d blocks at max_radius=%.2f search units...",
                B, max_radius,
            )
        candidate_counts = np.asarray(
            tree.query_ball_point(
                transformed_blocks, r=max_radius, return_length=True,
            ),
            dtype=np.int32,
        )
        needs_knn_fallback = candidate_counts < required_samples
        self._blocks_needing_knn_fallback = needs_knn_fallback
        n_fallback_blocks = int(np.sum(needs_knn_fallback))

        active_indices = np.arange(B, dtype=np.intp)
        n_active = len(active_indices)
        logger.info(
            "LGPR support diagnostic: %d / %d blocks (%.1f%%) need beyond-radius "
            "k-NN fallback; all blocks retained for estimation. Took %.1fs.",
            n_fallback_blocks, B, 100.0 * n_fallback_blocks / max(B, 1),
            time.time() - t_prefilter,
        )

        _orientation_field = self._orientation_field if self.use_lva else None
        _do_cos = n_disc > 1 and self.change_of_support
        _sigma_w_sq = None
        if _do_cos:
            _block_dims = block_sizes[0] if block_sizes.ndim == 2 else block_sizes
            _sigma_w_sq = self._within_block_variance_cached(
                _block_dims, global_vp, R_global, n_disc,
            )

        # ── Phase 1: Super-block tiling ──
        # Group adjacent blocks into spatial tiles. Each tile shares ONE
        # neighbourhood (searched from tile centre) and ONE factorised
        # kernel system. All blocks in the tile are predicted in a single
        # vectorised call. This reduces factorisations by ~20-50× and
        # eliminates the per-block Python loop for search/trim.
        #
        # Tile size = fraction of search radius to ensure blocks at tile
        # edges are still well-served by the tile-centre neighbourhood.
        # The maximum positional error for a block at the tile edge is
        # half the tile diagonal ≈ 0.87 × tile_span.  With tile_span =
        # 0.5 × min_search_radius, worst-case offset ≈ 0.43 × R1 which
        # is acceptable since blocks share ~85%+ of their neighbourhood.
        first_pass_radius = float(search_radii[0]) if len(search_radii) > 0 else 1.0
        # Scale tile size based on block count: for large models, use
        # bigger tiles to reduce total factorisation count.
        _tile_frac = 0.5
        if n_active > 500_000:
            _tile_frac = 0.65  # More aggressive tiling for very large models
        elif n_active > 100_000:
            _tile_frac = 0.55
        if block_sizes.ndim == 2:
            _bs = block_sizes[0]
        else:
            _bs = block_sizes
        _bs_abs = np.abs(np.asarray(_bs, dtype=np.float64))
        _axis_ranges = np.array(
            [self.range_max, self.range_mid, self.range_min], dtype=np.float64,
        )
        _tile_target = np.maximum(
            first_pass_radius * _tile_frac * _axis_ranges,
            np.maximum(_bs_abs, 1.0),
        )
        _supports_have_extent = _bs_abs > 1e-9
        _tile_basis = np.where(_supports_have_extent, _bs_abs, _tile_target)
        _min_tile_cells = np.where(_supports_have_extent, 2, 1).astype(np.int32)
        _tile_cells = np.maximum(
            _min_tile_cells,
            np.round(_tile_target / np.maximum(_tile_basis, 1e-12)).astype(np.int32),
        )
        _tile_span_actual = np.where(
            _supports_have_extent,
            _tile_cells.astype(np.float64) * _tile_basis,
            _tile_target,
        )

        t_tile = time.time()

        # Assign each active block to a tile via integer grid coordinates
        active_centroids = centroids[active_indices]
        _origin = active_centroids.min(axis=0)
        with np.errstate(invalid="ignore"):
            tile_ijk = np.floor(
                (active_centroids - _origin) / np.maximum(_tile_span_actual, 1e-12)
            ).astype(np.int32)
        np.nan_to_num(tile_ijk, copy=False)  # Replace NaN→0 for degenerate grids

        # Vectorised tile grouping: encode (i,j,k) → single int, then use
        # np.unique to find groups (avoids 934k Python dict insertions).
        _nj = int(tile_ijk[:, 1].max()) + 1 if n_active > 0 else 1
        _nk = int(tile_ijk[:, 2].max()) + 1 if n_active > 0 else 1
        tile_flat = tile_ijk[:, 0].astype(np.int64) * (_nj * _nk) + tile_ijk[:, 1].astype(np.int64) * _nk + tile_ijk[:, 2].astype(np.int64)
        unique_tiles, inverse, tile_counts = np.unique(tile_flat, return_inverse=True, return_counts=True)
        n_tiles = len(unique_tiles)

        # Build tile_map as list of arrays for fast iteration
        # Sort by tile assignment for cache-friendly access
        sort_order = np.argsort(inverse, kind='stable')
        tile_boundaries = np.zeros(n_tiles + 1, dtype=np.intp)
        np.cumsum(tile_counts, out=tile_boundaries[1:])
        tile_members_sorted = sort_order  # indices into active_indices
        logger.info(
            "LGPR tiling: %d blocks → %d tiles (avg %.1f blocks/tile, "
            "tile_span=%.1f×%.1f×%.1f m). Tiling took %.2fs.",
            n_active, n_tiles, n_active / max(n_tiles, 1),
            _tile_span_actual[0], _tile_span_actual[1], _tile_span_actual[2],
            time.time() - t_tile,
        )

        # ── Phase 2: Per-tile search + factorise + batch predict ──
        t_est = time.time()
        n_estimated = 0
        n_tiles_skipped = 0

        # k-NN candidate budget for the per-block neighbourhood searches
        # executed inside each tile.
        if self.max_samples > 0:
            k_search = min(
                max(int(self.max_samples), max(self.min_samples * 8, 64)),
                len(transformed_coords),
            )
        else:
            k_search = len(transformed_coords)
        logger.info("LGPR: block-specific neighbourhoods within tiles (k=%d).", k_search)

        # ── Tile worker function ──
        # SPEED: one neighbourhood lookup from tile centre, one factorisation,
        # one vectorised prediction for ALL blocks in the tile.  Eliminates
        # the per-block Python loop (was 40k iterations of neighbourhood
        # selection + dict-key hashing).
        def _process_tile(it: int):
            """Process one tile: centre search → factorise → batch predict."""
            s_t, e_t = tile_boundaries[it], tile_boundaries[it + 1]
            members = tile_members_sorted[s_t:e_t]
            mb_ids = active_indices[members]
            n_batch = len(mb_ids)
            query_points_t = transformed_blocks[mb_ids]

            # Single neighbourhood from tile centre
            tile_centre = query_points_t.mean(axis=0)
            best_idx, best_pass = self._select_local_neighbourhood(
                tree, transformed_coords, tile_centre,
            )
            if len(best_idx) < self.min_samples:
                return None

            local_coords = coords[best_idx]
            local_values = values[best_idx]
            diff = transformed_coords[best_idx] - tile_centre
            oct_code = (
                (diff[:, 0] >= 0).astype(np.int8) * 4
                + (diff[:, 1] >= 0).astype(np.int8) * 2
                + (diff[:, 2] >= 0).astype(np.int8)
            )
            n_oct = int(np.unique(oct_code).size)
            local_drift = self._local_neighbourhood_drift_type(
                local_coords, octant_count=n_oct,
            )

            tile_pass = np.full(n_batch, best_pass, dtype=np.int32)
            tile_samples = np.full(n_batch, len(best_idx), dtype=np.int32)
            tile_octants = np.full(n_batch, n_oct, dtype=np.int32)

            # Single factorisation for entire tile
            fact, wts, pc, _, _ = self._factorise_kernel_system(
                local_coords, local_values, vp_local,
                R_global, S_global,
                context=f"TILE n={len(best_idx)} blocks={n_batch}",
                drift_type=local_drift,
            )
            li = compute_l_inv(fact)

            # Batch predict all blocks — centroid-only, fused Numba
            tile_est = predict_mean(
                centroids[mb_ids], local_coords, wts, pc,
                kernel_type=vp_local.kernel_type,
                alpha=vp_local.alpha, sill=vp_local.sill,
                range_=vp_local.range_,
                R=R_global, S=S_global,
                drift_type=local_drift,
                orientation_field=_orientation_field,
            )

            s2_all = predict_variance(
                centroids[mb_ids], local_coords, fact,
                kernel_type=vp_local.kernel_type,
                alpha=vp_local.alpha, sill=vp_local.sill,
                range_=vp_local.range_, nugget=vp_local.nugget,
                R=R_global, S=S_global,
                drift_type=local_drift,
                orientation_field=_orientation_field,
                l_inv=li,
            )

            if _do_cos and _sigma_w_sq is not None:
                d = s2_all - float(_sigma_w_sq)
                tile_var = np.where(
                    d > 0, d, np.maximum(s2_all * 0.05, 0.0),
                )
            else:
                tile_var = np.maximum(s2_all, 0.0)

            return (mb_ids, tile_pass, tile_samples, tile_octants, tile_est, tile_var)

        # ── Execute tiles serially ──
        _prev_level = logger.level
        if n_tiles > 100:
            logger.setLevel(logging.WARNING)

        log_interval = max(1, n_tiles // 20)

        for it in range(n_tiles):
            result = _process_tile(it)
            if result is None:
                n_tiles_skipped += 1
                continue
            mb_ids, bp, ns, no, te, tv = result
            estimates[mb_ids] = te
            variances[mb_ids] = tv
            search_passes[mb_ids] = bp
            sample_counts[mb_ids] = ns
            octant_counts[mb_ids] = no
            n_estimated += len(mb_ids)
            if it > 0 and it % log_interval == 0 and _prev_level <= logging.INFO:
                logger.setLevel(_prev_level)
                logger.info(
                    "LGPR progress: %d/%d tiles (%.0f%%), %d blocks",
                    it, n_tiles, 100.0 * it / n_tiles, n_estimated,
                )
                if n_tiles > 100:
                    logger.setLevel(logging.WARNING)

        logger.setLevel(_prev_level)

        logger.info(
            "LGPR complete: %d/%d blocks estimated in %d tiles (%.1fs), "
            "%d blocks required beyond-radius fallback, %d tiles skipped (no data).",
            n_estimated, B, n_tiles, time.time() - t_est,
            n_fallback_blocks, n_tiles_skipped,
        )

        self._estimation_geometry_stats = (
            sample_counts,
            octant_counts,
            search_passes,
        )

        return BlendedResult(
            estimates=estimates,
            variances=clamp_variance(variances),
            n_active_subdomains=np.ones(B, dtype=np.int32),
            within_variance=clamp_variance(variances),
            between_variance=np.zeros(B, dtype=np.float64),
            total_variance=clamp_variance(variances),
        )

    def _refit_subdomain_weights(self, new_values: np.ndarray) -> None:
        """Re-solve GPR weights for a new grade variable.

        Reuses the stored Cholesky or LU factorisation from Step 4 to solve
        for new weights without repeating O(N³) refactorisation.  This is
        used by the ILR multi-component pipeline: the kernel matrix K_aug is
        identical for all ILR coordinates (only the RHS changes), so we can
        call ``solve_weights_from_factor`` which costs only O(N²).

        Parameters
        ----------
        new_values : np.ndarray
            (N,) grade values for the new variable (ILR component j).
        """
        if self._subdomains is None:
            return
        for sd in self._subdomains:
            if sd.cholesky_factor is None or sd.weights is None:
                continue
            idx = sd.sample_indices
            vals_local = new_values[idx]
            M = len(sd.poly_coeffs) if sd.poly_coeffs is not None else 0
            sd.weights, sd.poly_coeffs = solve_weights_from_factor(
                sd.cholesky_factor, vals_local, M=M,
            )

    def _apply_adaptive_discretisation(
        self,
        grades: np.ndarray,
        variances: np.ndarray,
        stitching_variances: np.ndarray,
        coords: np.ndarray,
        R_global: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Re-estimate blocks in high-gradient areas with finer discretisation."""
        B = self._block_centroids.shape[0]
        block_sizes = self._block_sizes
        global_vp = self._get_global_variogram_params()

        # Use initial estimates to determine gradients
        def predict_fn(pts: np.ndarray) -> np.ndarray:
            result = blend_estimates_fast(
                pts, self._subdomains, coords,
                drift_type=self._active_drift_type(), R_global=R_global,
            )
            return result.estimates

        densities = adaptive_discretisation_density(
            self._block_centroids,
            block_sizes,
            predict_fn,
        )

        # Re-estimate blocks that need higher discretisation
        need_update = densities > 8  # Only re-estimate medium/high
        update_idx = np.where(need_update)[0]

        if len(update_idx) == 0:
            return grades, variances, stitching_variances

        logger.info(
            "Re-estimating %d blocks with adaptive discretisation", len(update_idx),
        )

        # Batch by discretisation tier so we call blend_estimates_fast once
        # per tier rather than once per block — eliminates the per-block
        # grouping / distance-matrix overhead that dominated runtime here.
        for disc_level in [27, 64]:
            tier_mask = densities[update_idx] == disc_level
            tier_idx = update_idx[tier_mask]
            if len(tier_idx) == 0:
                continue

            offsets = get_offsets(disc_level)
            n_pts = len(offsets)

            # Build all disc points for this tier in one shot
            if block_sizes.ndim == 2:
                bs_arr = block_sizes[tier_idx]  # (n_tier, 3)
                disc_all = (
                    self._block_centroids[tier_idx, np.newaxis, :]
                    + offsets[np.newaxis, :, :] * bs_arr[:, np.newaxis, :]
                ).reshape(-1, 3)
            else:
                disc_all = (
                    self._block_centroids[tier_idx, np.newaxis, :]
                    + offsets[np.newaxis, :, :] * block_sizes[np.newaxis, np.newaxis, :]
                ).reshape(-1, 3)

            # One blending call covers all blocks in the tier
            blended = blend_estimates_fast(
                disc_all, self._subdomains, coords,
                drift_type=self._active_drift_type(), R_global=R_global,
                compute_variance=True,
            )

            f_all = blended.estimates.reshape(len(tier_idx), n_pts)
            s2_all = blended.variances.reshape(len(tier_idx), n_pts)
            between_all = blended.between_variance.reshape(len(tier_idx), n_pts)

            block_grades_tier = f_all.mean(axis=1)
            point_posterior = s2_all.mean(axis=1)
            if n_pts > 1 and self.change_of_support:
                if block_sizes.ndim == 2:
                    sigma_w_sq = np.array(
                        [
                            self._within_block_variance_cached(
                                block_sizes[idx], global_vp, R_global, disc_level,
                            )
                            for idx in tier_idx
                        ],
                        dtype=np.float64,
                    )
                else:
                    sigma_w_sq = self._within_block_variance_cached(
                        block_sizes, global_vp, R_global, disc_level,
                    )
                posterior_tier = np.maximum(point_posterior - sigma_w_sq, 0.0)
            else:
                posterior_tier = point_posterior

            grades[tier_idx] = block_grades_tier
            variances[tier_idx] = posterior_tier
            stitching_variances[tier_idx] = np.mean(between_all, axis=1)

        return grades, variances, stitching_variances

    def _compute_geometric_stats(
        self,
        coords: np.ndarray,
        R_global: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute geometric quality stats for each block.

        Returns sample count, octant count, and search pass per block.
        Uses vectorised KD-tree queries for speed.
        """
        from scipy.spatial import cKDTree

        B = self._block_centroids.shape[0]
        sample_counts = np.zeros(B, dtype=np.int32)
        octant_counts = np.zeros(B, dtype=np.int32)
        radii = tuple(float(radius) for radius in self.local_search_radii)
        search_passes = np.full(B, len(radii), dtype=np.int32)

        if self.use_lva:
            logger.warning(
                "Geometric classification uses global anisotropy only; "
                "local-varying anisotropy is not yet honoured in search geometry.",
            )

        search_coords = self._transform_for_search(coords)
        search_blocks = self._transform_for_search(self._block_centroids)
        tree = cKDTree(search_coords)

        # Chunked ball queries to avoid RAM explosion on large block models.
        # query_ball_point(all_B_centroids) allocates O(B × N) intermediate
        # storage; for B=50M blocks and N=5000 composites this is ~2TB.
        # Chunking keeps peak memory to CHUNK × N × 8 bytes ≈ 2 GB.
        CHUNK = 50_000
        counts_by_pass = [np.empty(B, dtype=np.intp) for _ in radii]
        required_samples = (
            max(self.min_samples, 8)
            if self._active_drift_type() == "linear"
            else self.min_samples
        )

        for start in range(0, B, CHUNK):
            end = min(start + CHUNK, B)
            chunk_pts = search_blocks[start:end]
            for pass_idx, radius in enumerate(radii):
                counts_by_pass[pass_idx][start:end] = tree.query_ball_point(
                    chunk_pts, r=radius, return_length=True,
                )

        sample_counts = counts_by_pass[-1].astype(np.int32)
        for pass_idx in range(len(radii) - 2, -1, -1):
            mask = counts_by_pass[pass_idx] >= required_samples
            search_passes[mask] = pass_idx + 1
            sample_counts[mask] = counts_by_pass[pass_idx][mask].astype(np.int32)

        # Octant counts — use nearest neighbours within the widest search pass.
        if self.max_samples > 0:
            k_nn = min(max(required_samples, self.max_samples), len(coords))
        else:
            k_nn = len(coords)
        nn_dists, nn_idx = tree.query(
            search_blocks,
            k=k_nn,
            distance_upper_bound=radii[-1],
        )
        if k_nn == 1:
            nn_dists = nn_dists.reshape(-1, 1)
            nn_idx = nn_idx.reshape(-1, 1)
        if nn_idx.ndim == 1:
            nn_idx = nn_idx.reshape(-1, 1)
            nn_dists = nn_dists.reshape(-1, 1)
        valid = np.isfinite(nn_dists) & (nn_idx < len(search_coords))
        safe_idx = np.where(valid, nn_idx, 0)
        # Vectorised octant computation
        nn_coords = search_coords[safe_idx]  # (B, k_nn, 3)
        diff = nn_coords - search_blocks[:, np.newaxis, :]  # (B, k_nn, 3)
        oct_code = (
            (diff[:, :, 0] >= 0).astype(np.int8) * 4
            + (diff[:, :, 1] >= 0).astype(np.int8) * 2
            + (diff[:, :, 2] >= 0).astype(np.int8)
        )  # (B, k_nn) values 0-7
        # Vectorised octant counting: encode as bitmask then vectorised popcount.
        oct_bits = np.where(
            valid,
            np.left_shift(np.ones(oct_code.shape, dtype=np.int32), oct_code),
            0,
        )
        oct_mask_per_block = np.bitwise_or.reduce(oct_bits, axis=1)  # (B,) uint8 bitmask

        # Popcount via a 256-entry lookup table — O(B) with no Python loop.
        _POPCOUNT_LUT = np.array(
            [bin(i).count('1') for i in range(256)], dtype=np.int32,
        )
        octant_counts = _POPCOUNT_LUT[oct_mask_per_block & 0xFF]

        return sample_counts, octant_counts, search_passes

    def _get_global_variogram_params(self) -> Dict[str, Any]:
        """Get representative variogram parameters.

        Prefer the effective run-time parameters established during
        ``estimate()``.  This keeps uncertainty scaling consistent when the
        sill was auto-derived from the working data variance rather than
        explicitly supplied in the config.
        """
        effective_sill = getattr(self, "_effective_sill", None)
        effective_nugget = getattr(self, "_effective_nugget", None)
        effective_alpha = getattr(self, "_effective_alpha", None)

        if effective_sill is not None:
            return {
                "kernel_type": self.kernel_type,
                "alpha": effective_alpha if effective_alpha is not None else self.alpha,
                "sill": float(effective_sill),
                "nugget": float(
                    effective_nugget if effective_nugget is not None else self.nugget
                ),
                "range_": self.range_max,
            }

        if not self._subdomains:
            return {
                "kernel_type": self.kernel_type,
                "alpha": self.alpha,
                "sill": 1.0,
                "nugget": self.nugget,
                "range_": self.range_max,
            }

        # Use parameters from the sub-domain with most samples
        best_sd = max(self._subdomains, key=lambda sd: sd.n_samples)
        if best_sd.variogram_params is not None:
            vp = best_sd.variogram_params
            return {
                "kernel_type": vp.kernel_type,
                "alpha": vp.alpha,
                "sill": vp.sill,
                "nugget": vp.nugget,
                "range_": vp.range_,
            }

        return {
            "kernel_type": self.kernel_type,
            "alpha": self.alpha,
            "sill": 1.0,
            "nugget": self.nugget,
            "range_": self.range_max,
        }

    def _resolve_support_swath_declustering_weights(self) -> Optional[np.ndarray]:
        """Return declustering weights for support-aware swath diagnostics.

        Support swaths should compare block means against a declustered sample
        reference. Use Competent Person supplied weights where available,
        otherwise fall back to the engine's cell-declustering scheme.
        """
        if self._composite_coords is None or self._composite_values is None:
            return None

        values = np.asarray(self._composite_values, dtype=np.float64).ravel()
        if len(values) == 0:
            return None

        if (
            self._declustering_weights is not None
            and len(self._declustering_weights) == len(values)
        ):
            weights = np.asarray(self._declustering_weights, dtype=np.float64).ravel()
            mask = np.isfinite(weights) & (weights > 0.0)
            if np.any(mask):
                return weights

        _, weights = self._cell_decluster(
            np.asarray(self._composite_coords, dtype=np.float64),
            values,
            self.decluster_cell_size,
        )
        return np.asarray(weights, dtype=np.float64).ravel()

    def _summarize_support_swath(
        self,
        support_swath_result: Optional[SupportSwathResult],
    ) -> Dict[str, float]:
        """Reduce support swath output to compact audit metrics."""
        summary = {
            "n_panels_total": 0,
            "n_panels_with_data": 0,
            "mean_rmse": 0.0,
            "mean_bias": 0.0,
        }
        if support_swath_result is None:
            return summary

        summary["n_panels_total"] = int(support_swath_result.n_panels_total)
        summary["n_panels_with_data"] = int(support_swath_result.n_panels_with_data)

        rmses: List[float] = []
        biases: List[float] = []
        for swath in support_swath_result.axes.values():
            mask = np.isfinite(swath.mean_estimated) & np.isfinite(swath.mean_actual)
            if not np.any(mask):
                continue
            diff = swath.mean_estimated[mask] - swath.mean_actual[mask]
            rmses.append(float(np.sqrt(np.mean(diff ** 2))))
            biases.append(float(np.mean(diff)))

        if rmses:
            summary["mean_rmse"] = float(np.mean(rmses))
            summary["mean_bias"] = float(np.mean(biases))

        return summary

    def _build_audit_record(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        grades: np.ndarray,
        variances: np.ndarray,
        cv_result: Optional[CVResult],
        conditional_bias_result: Optional[ConditionalBiasResult],
        support_swath_result: Optional[SupportSwathResult],
        cos_result: Optional[ChangeOfSupportResult],
        classification_result: Optional[ClassificationResult],
        elapsed: float,
    ) -> ARBFAuditRecord:
        """Build JORC Table 1 Section 3 audit record."""
        global_vp = self._get_global_variogram_params()
        support_swath_summary = self._summarize_support_swath(support_swath_result)
        bs = self._block_sizes
        if bs.ndim == 2:
            bs = bs[0]

        reporting_mask = self._reporting_block_mask_for_output(len(grades))

        record = ARBFAuditRecord(
            operator=self.operator,
            num_composites=len(self._composite_values),
            data_hash=compute_data_hash(
                self._composite_coords, self._composite_values,
            ),
            estimation_mode=self.estimation_mode,
            kernel_type=self.kernel_type,
            kernel_alpha=self.alpha,
            sill=global_vp["sill"],
            nugget=global_vp["nugget"],
            range_=global_vp["range_"],
            drift_type=self._active_drift_type(),
            requested_drift_type=self.drift_type,
            effective_drift_type=self._active_drift_type(),
            drift_selection_method=self._trend_diagnostics.get(
                "selection_method", "manual",
            ),
            trend_cv_constant_rmse=float(
                self._trend_diagnostics.get("constant", {}).get("rmse", 0.0),
            ),
            trend_cv_linear_rmse=float(
                self._trend_diagnostics.get("linear", {}).get("rmse", 0.0),
            ),
            trend_cv_rmse_improvement=float(
                self._trend_diagnostics.get("rmse_improvement", 0.0),
            ),
            accuracy=self.accuracy,
            n_subdomains=len(self._subdomains) if self._subdomains else 0,
            subdomain_method=self.subdomain_method,
            avg_samples_per_subdomain=float(np.mean([
                sd.n_samples for sd in self._subdomains
            ])) if self._subdomains else 0.0,
            use_lva=self.use_lva,
            lva_source=self.lva_source,
            use_normal_score=self.use_normal_score,
            use_ilr=self.use_ilr,
            use_change_of_support=self.change_of_support,
            max_samples=self.max_samples,
            min_samples=self.min_samples,
            azimuth=self.azimuth,
            dip=self.dip,
            pitch=self.pitch,
            range_max=self.range_max,
            range_mid=self.range_mid,
            range_min=self.range_min,
            domain_policy=self._domain_diagnostics.get("domain_policy", self.domain_policy),
            n_geological_domains=int(
                self._domain_diagnostics.get("n_geological_domains", 0),
            ),
            domains_enforced=bool(
                self._domain_diagnostics.get("domains_enforced", False),
            ),
            block_domain_assignment=str(
                self._domain_diagnostics.get("block_domain_assignment", "none"),
            ),
            block_size=bs.tolist(),
            n_blocks_estimated=int(np.sum(np.isfinite(grades[reporting_mask]))),
            n_blocks_total=int(np.sum(reporting_mask)),
            clip_to_drill_footprint=self.clip_to_drill_footprint,
            footprint_buffer_ranges=self.footprint_buffer_ranges,
            discretisation=self.discretisation_mode,
            discretisation_density=(
                str(self.discretisation_density)
                if self._estimation_discretisation_density() == self.discretisation_density
                else f"{self.discretisation_density}->{self._estimation_discretisation_density()}"
            ),
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            elapsed_seconds=elapsed,
            output_attributes=["ARBF_GRADE", "ARBF_VARIANCE", "ARBF_CLASS"],
        )

        if cv_result is not None:
            record.cv_rmse = cv_result.rmse
            record.cv_mae = cv_result.mae
            record.cv_r_squared = cv_result.r_squared
            record.cv_mean_error = cv_result.mean_error
            record.cv_slope_of_regression = cv_result.slope_of_regression
            record.cv_normalised_rmse = cv_result.normalised_rmse

        if conditional_bias_result is not None:
            record.conditional_bias_global_slope = (
                conditional_bias_result.global_slope
            )
            record.conditional_bias_binned_slope = (
                conditional_bias_result.binned_slope
            )
            record.conditional_bias_mean_bin_bias = (
                conditional_bias_result.mean_bin_bias
            )
            record.conditional_bias_max_abs_bin_bias = (
                conditional_bias_result.max_abs_bin_bias
            )

        if support_swath_result is not None:
            record.support_swath_panel_factors = list(
                support_swath_result.panel_factors,
            )
            record.support_swath_panels_total = int(
                support_swath_summary["n_panels_total"],
            )
            record.support_swath_panels_with_data = int(
                support_swath_summary["n_panels_with_data"],
            )
            record.support_swath_mean_rmse = float(
                support_swath_summary["mean_rmse"],
            )
            record.support_swath_mean_bias = float(
                support_swath_summary["mean_bias"],
            )

        if cos_result is not None:
            record.support_ratio = cos_result.support_ratio
            record.sigma_point = cos_result.sigma_point
            record.sigma_block = cos_result.sigma_block

        if classification_result is not None:
            reporting_class_summary = self._reporting_class_summary(
                classification_result.class_names,
            )
            if self._classification_threshold_values is not None:
                record.variance_thresholds = list(self._classification_threshold_values)
            record.classification_method = (
                "dual-criteria (variance + geometric, "
                f"threshold_source={self._classification_threshold_source})"
            )
            record.measured_blocks = reporting_class_summary.get("Measured", 0)
            record.indicated_blocks = reporting_class_summary.get("Indicated", 0)
            record.inferred_blocks = reporting_class_summary.get("Inferred", 0)
            record.unclassified_blocks = reporting_class_summary.get("Unclassified", 0)

        return record
