"""
ARBF Estimator Adapter — wraps the self-contained FastRBF engine (v2) to
expose the same interface as the external ``geostats.arbf.engine.ARBFEstimator``.

The controller (``geostats_controller._prepare_arbf_payload``) calls:
    estimator = ARBFEstimator(config)          # dict with 65+ keys
    estimator.set_composites(coords, values)
    estimator.set_block_model(centroids, block_sizes)
    estimator.set_domains(composite_domains, block_domains)   # optional
    estimator.set_progress_callback(callback)                  # optional
    results = estimator.estimate()                             # dict-like

This adapter translates that interface into FastRBFEstimator calls.

Engine selection
----------------
There are TWO ARBF engines in the codebase:

1. ``geostats/arbf/engine.py`` — standalone, older, 10-step orchestrator with
   its own sub-modules (blending, classification, cross_validation, etc.).
   NOT used by the GeoX UI.

2. ``block_model_viewer/geostats/fastrbf_engine_v2.py`` — integrated, newer,
   self-contained FastRBF estimator. **This is the AUTHORITATIVE engine** used
   by the GeoX UI via this adapter. The controller imports
   ``ARBFEstimatorAdapter`` (aliased as ``ARBFEstimator``) from this module.

The old engine at ``geostats/arbf/engine.py`` is retained for standalone/
scripting use but is not wired into any UI panel or controller.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .fastrbf_engine_v2 import (
    Anisotropy,
    BlockSettings,
    FastRBFEstimator,
    NeighbourhoodSettings,
    PartitionSettings,
    RBFKernel,
    RBFSettings,
    UncertaintyScore,
    VariogramModel,
    VariogramStructure,
    compute_uncertainty_index,
    cv_metrics,
    leave_one_out_cv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotation matrix from geological angles (GeoX convention)
# ---------------------------------------------------------------------------

def euler_to_rotation_matrix(
    azimuth: float = 0.0,
    dip: float = 0.0,
    pitch: float = 0.0,
) -> np.ndarray:
    """Build a 3x3 rotation matrix from geological angles (degrees).

    GeoX native convention (right-hand rule, geographic azimuth):
        R = Ry(pitch) @ Rx(dip) @ Rz(-azimuth)

    Axis definitions:
        X = East, Y = North, Z = Up
        azimuth: clockwise from North (0=N, 90=E) — negated for Rz
        dip: positive downward from horizontal
        pitch (rake): rotation about the major axis
    """
    az = np.deg2rad(-azimuth)
    dp = np.deg2rad(dip)
    pt = np.deg2rad(pitch)

    cz, sz = np.cos(az), np.sin(az)
    cx, sx = np.cos(dp), np.sin(dp)
    cy, sy = np.cos(pt), np.sin(pt)

    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])

    return Ry @ Rx @ Rz


# ---------------------------------------------------------------------------
# Kernel name mapping
# ---------------------------------------------------------------------------

_KERNEL_MAP = {
    "spheroidal": "spheroidal",
    "spherical": "spheroidal",
    "exponential": "exponential",
    "gaussian": "gaussian",
    "generalised_cauchy": "cauchy",
    "gen_cauchy": "cauchy",
    "cauchy": "cauchy",
    # Panel display names
    "Spheroidal (Leapfrog default)": "spheroidal",
    "Spherical": "spheroidal",
    "Gaussian": "gaussian",
    "Exponential": "exponential",
}

# RBF basis function mapping (panel display name → engine name)
_RBF_BASIS_MAP = {
    "Wendland C2": "wendland_c2",
    "Wendland C4": "wendland_c4",
    "Gaussian RBF": "gaussian",
    "Multiquadric": "multiquadric",
    "Inverse Multiquadric": "inverse_multiquadric",
    "Cubic": "cubic",
    "Thin Plate Spline": "thin_plate_spline",
    # Internal names pass through
    "wendland_c2": "wendland_c2",
    "wendland_c4": "wendland_c4",
    "gaussian": "gaussian",
    "multiquadric": "multiquadric",
    "inverse_multiquadric": "inverse_multiquadric",
    "cubic": "cubic",
    "thin_plate_spline": "thin_plate_spline",
}


# ---------------------------------------------------------------------------
# Result wrapper (dict-like access)
# ---------------------------------------------------------------------------

class _ResultDict(dict):
    """Dict subclass that also supports attribute access for dataclass compat."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

_CLASS_NAMES = np.array(["Measured", "Indicated", "Inferred", "Unclassified"])


# ---------------------------------------------------------------------------
# Direct variogram v2 → FastRBF conversion (zero remapping)
# ---------------------------------------------------------------------------

def variogram_v2_to_engine_params(
    v2_model: Any,
    v2_metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Convert a VariogramModel3D (from variogram_engine_v2) directly into
    FastRBF engine dataclass instances.

    Returns a dict with keys: 'anisotropy', 'variogram', 'rbf_settings'
    that can be passed straight to FastRBFEstimator — no UI round-trip.

    Parameters
    ----------
    v2_model : variogram_engine_v2.VariogramModel3D
        The fitted 3D variogram model.
    v2_metadata : dict, optional
        The model's metadata dict (contains transform_mode, drift_mode, etc.).
    """
    from .variogram_engine_v2 import VariogramModel3D, NestedStructure

    if not isinstance(v2_model, VariogramModel3D):
        raise TypeError(f"Expected VariogramModel3D, got {type(v2_model)}")

    meta = v2_metadata or v2_model.metadata or {}

    # ── Variogram model ──────────────────────────────────────────────
    # Use the primary (largest sill) structure for the kernel type
    if v2_model.structures:
        primary = max(v2_model.structures, key=lambda s: s.sill)
        model_name = _KERNEL_MAP.get(primary.model, primary.model)
        ranges = primary.ranges
        partial_sill = sum(s.sill for s in v2_model.structures)
    else:
        raise ValueError(
            "Variogram model has no fitted structures. "
            "Re-fit the variogram before running estimation."
        )

    variogram = VariogramModel(
        model=model_name,
        nugget_micro=max(v2_model.nugget, 0.0),
        partial_sill=max(partial_sill, 1e-6),
        ranges=ranges,
    )

    # ── Anisotropy (rotation matrix passes through directly) ─────────
    anisotropy = Anisotropy(
        ranges=ranges,
        rotation_matrix=v2_model.rotation_matrix.copy(),
    )

    # ── RBF settings from variogram metadata ─────────────────────────
    transform_mode = meta.get("transform_mode", "raw")
    drift_mode = meta.get("drift_mode", "none")
    if drift_mode == "auto":
        drift_mode = "linear"
    if drift_mode not in ("none", "constant", "linear"):
        drift_mode = "constant"

    use_normal_scores = transform_mode in ("nscore", "normal_score")

    rbf_settings = RBFSettings(
        drift=drift_mode,
        use_normal_scores=use_normal_scores,
    )

    # Build RBF kernel from variogram guidance
    rbf_kernel = RBFKernel.from_variogram_guidance(variogram)

    return {
        "anisotropy": anisotropy,
        "variogram": variogram,
        "rbf_settings": rbf_settings,
        "rbf_kernel": rbf_kernel,
        "transform_mode": transform_mode,
        "drift_mode": drift_mode,
    }


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class ARBFEstimatorAdapter:
    """Drop-in replacement for ``geostats.arbf.engine.ARBFEstimator``."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._coords: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None
        self._block_sizes: Optional[np.ndarray] = None
        self._block_volumes: Optional[np.ndarray] = None
        self._composite_domains: Optional[np.ndarray] = None
        self._block_domains: Optional[np.ndarray] = None
        self._progress_callback: Optional[Callable] = None
        self._declustering_weights: Optional[np.ndarray] = None
        self._ns_auto_disabled: bool = False
        self._ns_auto_disabled_cv: float = float("nan")

    # ── Data setters (same signatures as old engine) ─────────────────────

    def set_composites(self, coords: np.ndarray, values: np.ndarray) -> None:
        self._coords = np.asarray(coords, dtype=float)
        self._values = np.asarray(values, dtype=float).ravel()

    def set_block_model(self, centroids: np.ndarray, block_sizes: np.ndarray) -> None:
        self._centroids = np.asarray(centroids, dtype=float)
        self._block_sizes = np.asarray(block_sizes, dtype=float)

        # Per-block volume — used by the panel QA to compute volume-
        # weighted block means. Handles both single 3-vector (broadcast
        # to all blocks) and per-block 2D arrays.
        n_blocks = self._centroids.shape[0]
        if self._block_sizes.ndim == 1 and self._block_sizes.size == 3:
            _vol = float(np.prod(self._block_sizes))
            self._block_volumes = np.full(n_blocks, _vol, dtype=float)
        elif self._block_sizes.ndim == 2 and self._block_sizes.shape[0] == n_blocks:
            self._block_volumes = np.prod(self._block_sizes, axis=1).astype(float)
        else:
            self._block_volumes = np.ones(n_blocks, dtype=float)

    def set_domains(
        self,
        composite_domains: np.ndarray,
        block_domains: np.ndarray,
    ) -> None:
        self._composite_domains = np.asarray(composite_domains)
        self._block_domains = np.asarray(block_domains)

    def set_declustering_weights(self, weights: np.ndarray) -> None:
        """Attach per-composite declustering weights.

        The weights are forwarded to ``FastRBFEstimator(sample_weights=...)``
        inside :meth:`estimate`, which applies them as a first-order
        weight-dependent diagonal regularisation in the RBF solve:
        samples with higher weight are fit more tightly while samples
        with lower weight are treated as noisier observations that the
        interpolant is allowed to deviate from. This isn't a full
        weighted least-squares reformulation of the augmented system,
        but it makes declustering actually bite the estimates and
        preserves the symmetric structure required by the fast
        Cholesky path.
        """
        self._declustering_weights = np.asarray(weights, dtype=float)
        _n = int(self._declustering_weights.size)
        _s = float(np.nansum(self._declustering_weights))
        logger.info(
            "ARBFEstimatorAdapter: %d declustering weights received "
            "(sum=%.2f) — applied as weight-dependent diagonal "
            "regularisation in the FastRBFEstimator solve.",
            _n, _s,
        )

    def set_orientation_field(self, field: Any) -> None:
        # LVA not supported in v2 engine — silently ignore
        logger.info("ARBFAdapter: LVA orientation field ignored (not supported in v2 engine)")

    def set_progress_callback(self, callback: Callable) -> None:
        self._progress_callback = callback

    # ── Config → engine parameter translation ────────────────────────────

    def _build_anisotropy(self) -> Anisotropy:
        cfg = self._config
        ranges = (
            max(cfg.get("range_max", 100.0), 1e-3),
            max(cfg.get("range_mid", 100.0), 1e-3),
            max(cfg.get("range_min", 100.0), 1e-3),
        )
        rot = euler_to_rotation_matrix(
            azimuth=cfg.get("azimuth", 0.0),
            dip=cfg.get("dip", 0.0),
            pitch=cfg.get("pitch", 0.0),
        )
        return Anisotropy(ranges=ranges, rotation_matrix=rot)

    def _build_variogram(self) -> VariogramModel:
        cfg = self._config
        kernel = cfg.get("kernel_type", "spheroidal")
        model = _KERNEL_MAP.get(kernel, "spheroidal")
        nugget = max(cfg.get("nugget", 0.0), 0.0)
        partial_sill = max(cfg.get("sill", 1.0), 1e-6)
        ranges = (
            max(cfg.get("range_max", 100.0), 1e-3),
            max(cfg.get("range_mid", 100.0), 1e-3),
            max(cfg.get("range_min", 100.0), 1e-3),
        )

        # ── Nested structures (from variogram panel) ─────────────────
        # Each structure has: model_type, contribution, range_major,
        # range_minor, range_vertical.  The primary ranges (above) are
        # used for the Anisotropy transform; per-structure ranges are
        # used for rescaling inside VariogramModel.gamma_from_h.
        #
        # Med5 fix: previously this path required at least 2 structures
        # (``len(raw_structs) >= 2``), silently ignoring single-structure
        # variograms imported from the panel. A 1-structure variogram is
        # perfectly valid (most deposits use one spherical); drop the
        # >= 2 check so any non-empty structures list propagates.
        raw_structs = cfg.get("variogram_structures")
        structures = None
        if raw_structs and isinstance(raw_structs, list) and len(raw_structs) >= 1:
            structures = []
            for s in raw_structs:
                mt = _KERNEL_MAP.get(s.get("model_type", "spherical"), "spheroidal")
                structures.append(VariogramStructure(
                    model_type=mt,
                    contribution=max(float(s.get("contribution", 0.0)), 1e-6),
                    ranges=(
                        max(float(s.get("range_major", ranges[0])), 1e-3),
                        max(float(s.get("range_minor", ranges[1])), 1e-3),
                        max(float(s.get("range_vertical", ranges[2])), 1e-3),
                    ),
                ))
            # Use the longest-range structure's ranges for primary anisotropy.
            # With a single structure, this is simply that structure's ranges.
            longest = max(structures, key=lambda s: s.ranges[0])
            ranges = longest.ranges
            # Override partial_sill (nested sills replace it)
            partial_sill = sum(s.contribution for s in structures)
            logger.info(
                "ARBF: variogram with %d structure%s "
                "(total partial sill=%.4f, primary ranges=%.1f/%.1f/%.1f)",
                len(structures), "" if len(structures) == 1 else "s",
                partial_sill, *ranges,
            )

        return VariogramModel(
            model=model,
            nugget_micro=nugget,
            partial_sill=partial_sill,
            ranges=ranges,
            structures=structures,
        )

    def _build_rbf_kernel(self, variogram: VariogramModel) -> RBFKernel:
        """Build the RBFKernel, using variogram parameters as guidance."""
        cfg = self._config
        basis_name = cfg.get("rbf_basis_type", "wendland_c2")
        basis = _RBF_BASIS_MAP.get(basis_name, "wendland_c2")
        shape_param = float(cfg.get("rbf_shape_parameter", 1.0))
        support_radius = float(cfg.get("rbf_support_radius", 1.0))

        # If user hasn't specified RBF parameters, infer from variogram
        if "rbf_basis_type" not in cfg:
            return RBFKernel.from_variogram_guidance(variogram, basis=basis)

        return RBFKernel(
            basis=basis,
            shape_parameter=shape_param,
            support_radius=support_radius,
            nugget=variogram.nugget_micro,
        )

    def _build_rbf_settings(self) -> RBFSettings:
        cfg = self._config
        drift = cfg.get("drift_type", "constant")
        if drift == "auto":
            drift = "linear"
        if drift not in ("none", "constant", "linear"):
            drift = "constant"
        basis_type = _RBF_BASIS_MAP.get(
            cfg.get("rbf_basis_type", "wendland_c2"), "wendland_c2")
        return RBFSettings(
            drift=drift,
            smoothing=0.0,
            epsilon=max(cfg.get("accuracy", 1e-6), 1e-12),
            use_normal_scores=bool(cfg.get("use_normal_score", True)),
            gh_order=20,
            basis_type=basis_type,
            shape_parameter=float(cfg.get("rbf_shape_parameter", 1.0)),
            support_radius=float(cfg.get("rbf_support_radius", 1.0)),
        )

    def _build_neighbourhood(self) -> NeighbourhoodSettings:
        cfg = self._config
        n_max = max(int(cfg.get("max_samples", 300)), 4)
        n_min = max(int(cfg.get("min_samples", 4)), 2)
        max_per_octant = max(int(cfg.get("max_samples_per_octant", 4)), 1)
        search_min_octants = int(cfg.get("search_min_octants", 3))
        n_start = min(search_min_octants * max_per_octant, n_max)

        search_mode = cfg.get("search_mode", "local")

        if search_mode == "global":
            # Global (Leapfrog-style): no search radius — use ALL composites.
            # RBF acts as Dual Kriging: every data point influences every
            # block.  Extrapolates to drift mean everywhere.  Best for
            # smooth continuous fields from sparse data.
            search_radius = None
            # In global mode, increase n_max to allow the full dataset
            n_max = max(n_max, 10000)
            max_per_octant = max(max_per_octant, 2000)
            logger.info("ARBF: Global search mode — all composites inform every block")
        else:
            # Local (Mining): compact kernel with anisotropic search ellipsoid.
            # Search radius in TRANSFORMED SPACE (range-normalised units).
            # The KD-tree is built in anisotropy-transformed coordinates where
            # 1 unit = 1 variogram range.  The UI spinners already express radii
            # as range-multipliers (e.g. 0.75, 1.5, 2.0), so we use them
            # directly — do NOT multiply by range_max.
            radii = cfg.get("local_search_radii", (0.75, 1.5, 2.0))
            if radii:
                search_radius = float(max(radii))
            else:
                search_radius = None
            logger.info(
                "ARBF: Local search mode — search_radius=%.2f (transformed)",
                search_radius or 0.0,
            )

        return NeighbourhoodSettings(
            n_min=n_min,
            n_start=max(n_start, n_min),
            n_max=n_max,
            max_per_octant=max_per_octant,
            search_radius=search_radius,
            global_mode=(search_mode == "global"),
        )

    def _build_block_settings(self) -> BlockSettings:
        cfg = self._config
        density = int(cfg.get("discretisation_density", 27))
        # Cube root for per-axis subpoint count
        n_per_axis = max(round(density ** (1.0 / 3.0)), 1)

        # Block dimensions from grid_spec or block_sizes
        if self._block_sizes is not None:
            if self._block_sizes.ndim == 1 and self._block_sizes.size == 3:
                dims = tuple(float(d) for d in self._block_sizes)
            elif self._block_sizes.ndim == 2:
                dims = tuple(float(d) for d in self._block_sizes[0])
            else:
                dims = (10.0, 10.0, 10.0)
        else:
            dims = (10.0, 10.0, 10.0)

        return BlockSettings(
            nx=n_per_axis,
            ny=n_per_axis,
            nz=n_per_axis,
            dims=dims,
        )

    def _build_partition_settings(self) -> PartitionSettings:
        cfg = self._config
        n_samples = self._coords.shape[0] if self._coords is not None else 0
        pum_threshold = int(cfg.get("pum_threshold", 3000))
        n_subdomains = int(cfg.get("n_subdomains", 0))

        enabled = (n_subdomains > 1) or (n_samples > pum_threshold)

        # Tile size: use anisotropy ranges as basis
        range_max = cfg.get("range_max", 100.0)
        range_mid = cfg.get("range_mid", 100.0)
        range_min = cfg.get("range_min", 100.0)
        tile_size = (
            max(range_max * 2.0, 50.0),
            max(range_mid * 2.0, 50.0),
            max(range_min * 2.0, 50.0),
        )

        overlap = max(float(cfg.get("overlap_factor", 2.0)), 1.1)

        return PartitionSettings(
            enabled=enabled,
            tile_size=tile_size,
            overlap_factor=overlap,
            stitch_eta=0.50,
            min_coverage_ratio=0.995,
        )

    # ── Classification from uncertainty index ────────────────────────────

    def _classify_blocks(
        self,
        uncertainty_index: np.ndarray,
        fail_flags: np.ndarray,
        sill_total: float = 1.0,
    ) -> np.ndarray:
        """Classification guidance based on uncertainty index.

        This is classification GUIDANCE, not JORC-compliant classification.
        Final JORC classification requires competent person review
        considering drill spacing, geological continuity, domain
        confidence, and reconciliation evidence.

        High confidence:     UI < 0.25
        Moderate confidence: 0.25 ≤ UI < 0.50
        Low confidence:      0.50 ≤ UI < 0.75
        Very low confidence: UI ≥ 0.75 OR fail_flag OR NaN
        """
        thresholds = self._config.get("classification_thresholds") or {}
        high_thresh = thresholds.get("measured", 0.25)
        moderate_thresh = thresholds.get("indicated", 0.50)
        low_thresh = thresholds.get("inferred", 0.75)

        # Start with Unclassified (3), then assign upward
        classifications = np.full(len(uncertainty_index), 3, dtype=int)

        # Only classify finite, non-failed blocks
        valid = np.isfinite(uncertainty_index) & ~fail_flags.astype(bool)
        classifications[valid & (uncertainty_index < low_thresh)] = 2      # Low confidence → Inferred
        classifications[valid & (uncertainty_index < moderate_thresh)] = 1  # Moderate → Indicated
        classifications[valid & (uncertainty_index < high_thresh)] = 0     # High → Measured

        n_total = len(classifications)
        n_meas = int(np.sum(classifications == 0))
        n_ind = int(np.sum(classifications == 1))
        n_inf = int(np.sum(classifications == 2))
        n_unc = int(np.sum(classifications == 3))
        logger.info(
            "Classification guidance: High=%d (%.1f%%), Moderate=%d (%.1f%%), "
            "Low=%d (%.1f%%), VeryLow=%d (%.1f%%)",
            n_meas, 100 * n_meas / max(n_total, 1),
            n_ind, 100 * n_ind / max(n_total, 1),
            n_inf, 100 * n_inf / max(n_total, 1),
            n_unc, 100 * n_unc / max(n_total, 1),
        )
        return classifications

    # ── Direct v2 variogram path ────────────────────────────────────────

    def _try_v2_variogram_direct(
        self, vario_override: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Check if variogram_results contains v2 fitted models and use them
        directly — no UI spinner round-trip, no field renaming.

        Parameters
        ----------
        vario_override : dict, optional
            If provided, use this variogram result dict instead of
            ``self._config["variogram_results"]``.  Used for per-domain
            variogram lookup.
        """
        vario = vario_override if vario_override is not None else self._config.get("variogram_results")
        if vario is None or not isinstance(vario, dict):
            return None

        v2_models = vario.get("_v2_fitted_models")
        if not v2_models:
            return None

        # Pick the best available model (major > omni > first available)
        from .variogram_engine_v2 import VariogramModel3D
        best = v2_models.get("major") or v2_models.get("omni")
        if best is None:
            best = next(iter(v2_models.values()), None)
        if best is None or not isinstance(best, VariogramModel3D):
            return None

        try:
            params = variogram_v2_to_engine_params(best, best.metadata)
            # Preserve user overrides from config for epsilon and normal score
            cfg = self._config
            params["rbf_settings"].epsilon = max(cfg.get("accuracy", 1e-6), 1e-12)
            # Allow panel normal-score checkbox to override transform mode
            if "use_normal_score" in cfg:
                params["rbf_settings"].use_normal_scores = bool(cfg["use_normal_score"])
            # Allow panel drift combo to override if explicitly set
            drift_override = cfg.get("drift_type", "")
            if drift_override == "auto":
                drift_override = "constant"  # safe default for auto
            if drift_override in ("none", "constant", "linear"):
                params["rbf_settings"].drift = drift_override
            # Ensure drift is never "auto" — engine doesn't support it
            if params["rbf_settings"].drift not in ("none", "constant", "linear"):
                params["rbf_settings"].drift = "constant"
            return params
        except Exception as exc:
            logger.error(
                "v2 variogram direct path failed: %s", exc, exc_info=True,
            )
            return None

    # ── Progress helper ──────────────────────────────────────────────────

    def _report(self, pct: int, msg: str) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(pct, msg)
            except Exception:
                pass

    # ── Main estimation ──────────────────────────────────────────────────

    def estimate(self) -> _ResultDict:
        """Run the estimation and return results matching the old ARBFResult interface."""
        t0 = time.time()
        timings = {}  # Track duration of each step

        self._report(5, "ARBF v2: Building engine parameters...")

        if self._coords is None or self._values is None:
            raise ValueError("Composites not set. Call set_composites() first.")
        if self._centroids is None:
            raise ValueError("Block model not set. Call set_block_model() first.")

        # Try direct variogram v2 path (zero remapping)
        t_step = time.time()
        v2_params = self._try_v2_variogram_direct()
        if v2_params is not None:
            anisotropy = v2_params["anisotropy"]
            variogram = v2_params["variogram"]
            rbf_settings = v2_params["rbf_settings"]
            rbf_kernel = v2_params.get("rbf_kernel") or self._build_rbf_kernel(variogram)
            logger.info(
                "ARBF: Using v2 variogram as guidance — model=%s, "
                "nugget=%.4f, sill=%.4f, ranges=%s, drift=%s, nscore=%s",
                variogram.model, variogram.nugget_micro, variogram.partial_sill,
                variogram.ranges, rbf_settings.drift, rbf_settings.use_normal_scores,
            )
        else:
            anisotropy = self._build_anisotropy()
            variogram = self._build_variogram()
            rbf_settings = self._build_rbf_settings()
        # Build the true RBF kernel (variogram is guidance only)
        rbf_kernel = self._build_rbf_kernel(variogram)
        logger.info(
            "ARBF: RBF kernel — basis=%s, shape=%.4f, support_radius=%.2f, nugget=%.6f",
            rbf_kernel.basis, rbf_kernel.shape_parameter,
            rbf_kernel.support_radius, rbf_kernel.nugget,
        )
        # ── NS auto-guard ────────────────────────────────────────────
        # On heavy-lognormal data (CV > 2), the normal-score back-
        # transform via Gauss-Hermite integration attenuates the local
        # signal and pushes CV slope far from 1.0. We auto-disable the
        # NS transform in that case unless the user has explicitly
        # forced it on via ``force_normal_score=True``. See 11/12/14 of
        # the panel-debugger sound-workflow scripts for empirical
        # validation on the real Cu dataset.
        if rbf_settings.use_normal_scores and self._values is not None:
            _v = np.asarray(self._values, dtype=float)
            _v = _v[np.isfinite(_v) & (_v > 0)] if (_v > 0).any() else _v
            if _v.size > 10:
                _mean = float(np.mean(_v))
                _std = float(np.std(_v))
                _cv = _std / max(abs(_mean), 1e-12)
                _force = bool(self._config.get("force_normal_score", False))
                if _cv > 2.0 and not _force:
                    logger.warning(
                        "ARBF NS auto-guard: data CV=%.2f exceeds 2.0 — "
                        "disabling Normal-Score transform for this run. "
                        "Heavy-tail NS back-transform is known to attenuate "
                        "the local signal (observed CV slope drops from "
                        "~0.9 to ~0.4 on multi-population deposits). "
                        "Set force_normal_score=True in the config to "
                        "override.",
                        _cv,
                    )
                    rbf_settings.use_normal_scores = False
                    self._ns_auto_disabled = True
                    self._ns_auto_disabled_cv = _cv
                else:
                    self._ns_auto_disabled = False
                    self._ns_auto_disabled_cv = _cv

        neighbourhood = self._build_neighbourhood()
        block_settings = self._build_block_settings()
        partition = self._build_partition_settings()
        timings["1_parameter_setup"] = time.time() - t_step

        # ── Parameter summary for diagnostics ────────────────────────────
        cfg = self._config
        logger.info(
            "ARBF PARAMETERS: search_mode=%s, basis=%s, support_radius=%.2f, "
            "ranges=(%.1f/%.1f/%.1f), azimuth=%.1f, dip=%.1f, pitch=%.1f, "
            "nugget_ratio=%.2f, n_composites=%d, footprint_clip=%s, "
            "domain_mask=%s",
            "global" if neighbourhood.global_mode else "local",
            rbf_kernel.basis, rbf_kernel.support_radius,
            anisotropy.ranges[0], anisotropy.ranges[1], anisotropy.ranges[2],
            cfg.get("azimuth", 0.0), cfg.get("dip", 0.0), cfg.get("pitch", 0.0),
            variogram.nugget_micro / max(variogram.sill_total, 1e-12),
            self._coords.shape[0] if self._coords is not None else 0,
            cfg.get("clip_to_drill_footprint", False),
            self._composite_domains is not None,
        )

        # ── Domain-separated estimation ──────────────────────────────────
        if self._composite_domains is not None and self._block_domains is not None:
            logger.info(
                "ARBF ADAPTER: domain-separated path active "
                "(composite_domains=%s, block_domains=%s)",
                np.unique(self._composite_domains),
                np.unique(self._block_domains),
            )
            return self._estimate_by_domain(
                anisotropy, variogram, rbf_settings, neighbourhood,
                block_settings, partition, t0, rbf_kernel=rbf_kernel,
            )

        # ── Single-domain estimation ─────────────────────────────────────
        t_step = time.time()
        self._report(10, "ARBF v2: Building estimator (KD-tree, NS transform, support ratio)...")

        estimator = FastRBFEstimator(
            coords=self._coords,
            values=self._values,
            anisotropy=anisotropy,
            variogram=variogram,
            rbf_settings=rbf_settings,
            neighbourhood=neighbourhood,
            block_settings=block_settings,
            partition_settings=partition,
            rbf_kernel=rbf_kernel,
            sample_weights=self._declustering_weights,
        )
        timings["2_estimator_init"] = time.time() - t_step

        # ── Diagnostic: log NS path decision ─────────────────────────
        logger.info(
            "ARBF DIAGNOSTIC — NS path:\n"
            "  rbf_settings.use_normal_scores = %s\n"
            "  estimator.nst is not None      = %s\n"
            "  variogram.sill_total           = %.4f\n"
            "  data mean                      = %.2f\n"
            "  data var                        = %.2f\n"
            "  data var / sill                = %.2f",
            rbf_settings.use_normal_scores,
            estimator.nst is not None,
            variogram.sill_total,
            float(np.mean(self._values)),
            float(np.var(self._values)),
            float(np.var(self._values)) / max(variogram.sill_total, 1e-12),
        )

        n_blocks = self._centroids.shape[0]
        self._report(15, f"ARBF v2: Estimating {n_blocks:,} blocks...")

        # Estimate blocks with progress updates
        t_step = time.time()
        raw = self._estimate_blocks_with_progress(estimator, self._centroids, 15, 70)
        timings["3_block_estimation"] = time.time() - t_step

        grades = raw["ARBF_GRADE"]
        variances = raw["ARBF_VAR_BLOCK"]
        fail_flags = raw["ARBF_FAIL_FLAG"]
        stitch_var = raw["ARBF_STITCH_VAR"]

        # ── POST-ESTIMATION DIAGNOSTIC ─────────────────────────────────
        _n_nan = int(np.sum(np.isnan(grades)))
        _n_zero = int(np.sum(grades == 0))
        _n_neg = int(np.sum(grades < 0))
        _n_finite = int(np.sum(np.isfinite(grades)))
        _fg_range = ""
        if _n_finite > 0:
            _fg = grades[np.isfinite(grades)]
            _fg_range = f", finite range [{_fg.min():.2f}, {_fg.max():.2f}], mean={_fg.mean():.2f}"
        logger.info(
            "ARBF ADAPTER post-estimation: %d blocks, %d NaN (%.1f%%), "
            "%d zero, %d negative, %d finite%s",
            len(grades), _n_nan,
            100.0 * _n_nan / max(len(grades), 1),
            _n_zero, _n_neg, _n_finite, _fg_range,
        )

        # ── Affine mean correction REMOVED ─────────────────────────────
        # A global multiplicative correction to force estimated mean to
        # match data mean is not sound geostatistical practice.  It can
        # distort local grade relationships and mask model problems.
        # If the back-transform is correct and the local model is sound,
        # the mean should be close without forced correction.
        # The correction factor is logged for diagnostics only.
        t_step = time.time()
        _affine_correction_factor = 1.0
        if rbf_settings.use_normal_scores:
            finite_mask = np.isfinite(grades) & (grades > 0)
            if np.any(finite_mask):
                est_mean = float(np.mean(grades[finite_mask]))
                data_mean = float(np.mean(self._values))
                if est_mean > 1e-12 and data_mean > 1e-12:
                    _affine_correction_factor = data_mean / est_mean
                    logger.info(
                        "E-type mean ratio (diagnostic only, NOT applied): "
                        "factor=%.4f (data_mean=%.2f, est_mean=%.2f)",
                        _affine_correction_factor, data_mean, est_mean,
                    )
        timings["3b_affine_correction"] = time.time() - t_step

        # Grade clipping
        t_step = time.time()
        clip_min = self._config.get("clip_min")
        clip_max = self._config.get("clip_max")
        if clip_min is not None:
            n_below = int(np.sum(grades < float(clip_min)))
            if n_below > 0:
                logger.warning(
                    "ARBF grade clipping: %d of %d blocks (%.1f%%) clipped to min=%.4f",
                    n_below, len(grades), 100.0 * n_below / max(len(grades), 1),
                    float(clip_min),
                )
            grades = np.clip(grades, float(clip_min), None)
        if clip_max is not None:
            n_above = int(np.sum(grades > float(clip_max)))
            if n_above > 0:
                logger.warning(
                    "ARBF grade clipping: %d of %d blocks (%.1f%%) clipped to max=%.4f",
                    n_above, len(grades), 100.0 * n_above / max(len(grades), 1),
                    float(clip_max),
                )
            grades = np.clip(grades, None, float(clip_max))

        # Even without user-specified clipping, warn about negative grades
        if clip_min is None:
            n_negative = int(np.sum(grades < 0.0))
            if n_negative > 0:
                logger.warning(
                    "ARBF: %d of %d blocks (%.1f%%) have negative estimated grades. "
                    "Consider enabling 'Grade Clipping' with min=0 in the ARBF panel "
                    "for non-negative variables (e.g., Cu, Zn, Pb).",
                    n_negative, len(grades), 100.0 * n_negative / max(len(grades), 1),
                )
        timings["4_grade_clipping"] = time.time() - t_step

        # Classification (based on uncertainty index, not variance ratio)
        t_step = time.time()
        self._report(72, "ARBF v2: Classifying blocks...")
        sill_total = variogram.sill_total if variogram else 1.0
        uncertainty_idx = raw.get("ARBF_UNCERTAINTY", np.full(len(grades), 0.5))
        classifications = self._classify_blocks(uncertainty_idx, fail_flags, sill_total)
        timings["5_classification"] = time.time() - t_step

        # Cross-validation
        t_step = time.time()
        cv_result = None
        audit_cv = {}
        if self._config.get("run_cv", True):
            self._report(75, "ARBF v2: Running cross-validation...")
            cv_max = min(int(self._config.get("cv_max_samples", 300)), self._coords.shape[0])
            cv_result = leave_one_out_cv(
                coords=self._coords,
                values=self._values,
                anisotropy=anisotropy,
                variogram=variogram,
                rbf_settings=rbf_settings,
                neighbourhood=neighbourhood,
                block_settings=block_settings,
                max_points=min(cv_max, 300),
            )
            audit_cv = {
                "cv_slope_of_regression": cv_result.get("SLOPE"),
                "cv_r_squared": cv_result.get("R2"),
                "cv_rmse": cv_result.get("RMSE"),
                "cv_mean_error": cv_result.get("ME"),
                "cv_mae": cv_result.get("MAE"),
                "cv_std_resid_mean": cv_result.get("STD_RESID_MEAN"),
                "cv_std_resid_var": cv_result.get("STD_RESID_VAR"),
                "cv_cover_50": cv_result.get("COVER_50"),
                "cv_cover_68": cv_result.get("COVER_68"),
                "cv_cover_90": cv_result.get("COVER_90"),
                "cv_cover_95": cv_result.get("COVER_95"),
            }
        timings["6_cross_validation"] = time.time() - t_step

        # ── Variance calibration ───────────────────────────────────
        # Fit a robust non-negative linear map from raw block variance
        # to a calibrated variance using the CV residuals. The ARBF
        # *mean* estimate is untouched — this only adjusts the
        # predictive variance to better match empirical residual spread.
        variances_calibrated, calib_audit = self._calibrate_variance(
            variances, cv_result,
            mode=self._config.get("variance_calibration_mode", "global"),
        )
        audit_cv.update(calib_audit)

        elapsed = time.time() - t0
        t_step = time.time()
        self._report(95, "ARBF v2: Packaging results...")

        # Build audit record
        n_measured = int(np.sum(classifications == 0))
        n_indicated = int(np.sum(classifications == 1))
        n_inferred = int(np.sum(classifications == 2))
        n_unclassified = int(np.sum(classifications == 3))
        fail_ratio = float(np.mean(fail_flags.astype(float)))

        # Compute support metrics properly, guarding against NaN
        finite_vars = variances[np.isfinite(variances)]
        if finite_vars.size > 0 and sill_total > 1e-12:
            mean_var = float(np.mean(finite_vars))
            mean_var_ratio = mean_var / sill_total
            sigma_point = float(np.sqrt(sill_total))
            sigma_block = float(np.sqrt(max(mean_var, 0.0)))
            support_ratio = sigma_block / sigma_point if sigma_point > 1e-12 else float("nan")
        else:
            mean_var = 0.0
            mean_var_ratio = float("nan")
            sigma_point = float(np.sqrt(max(sill_total, 0.0)))
            sigma_block = float("nan")
            support_ratio = float("nan")

        # Compute NS-space variance metrics (for audit)
        var_ns = raw.get("ARBF_VAR_NS", variances)
        finite_var_ns = var_ns[np.isfinite(var_ns)] if var_ns is not None else finite_vars
        mean_var_ns = float(np.mean(finite_var_ns)) if finite_var_ns.size > 0 else float("nan")

        # Mean uncertainty index
        mean_uncertainty_index = float(np.nanmean(uncertainty_idx))

        # Grade statistics
        finite_grades = grades[np.isfinite(grades)]
        grade_stats = {}
        if finite_grades.size > 0:
            grade_stats = {
                "grade_mean": float(np.mean(finite_grades)),
                "grade_median": float(np.median(finite_grades)),
                "grade_std": float(np.std(finite_grades)),
                "grade_min": float(np.min(finite_grades)),
                "grade_max": float(np.max(finite_grades)),
                "grade_p05": float(np.percentile(finite_grades, 5)),
                "grade_p95": float(np.percentile(finite_grades, 95)),
                "max_to_median_ratio": float(
                    np.max(finite_grades) / max(np.median(finite_grades), 1e-12)
                ),
                "n_negative_grades": int(np.sum(finite_grades < 0)),
                "n_bounded_grades": int(np.sum(
                    (finite_grades == np.min(finite_grades)) | (finite_grades == np.max(finite_grades))
                )),
            }

        audit_record = {
            "num_composites": int(self._coords.shape[0]),
            "n_blocks_estimated": int(n_blocks),
            "n_blocks_total": int(n_blocks),
            "n_blocks_active": int(np.sum(np.isfinite(grades))),
            "elapsed_seconds": elapsed,
            "measured_blocks": n_measured,
            "indicated_blocks": n_indicated,
            "inferred_blocks": n_inferred,
            "unclassified_blocks": n_unclassified,
            "fail_ratio": fail_ratio,
            "mean_variance_ratio": mean_var_ratio,
            "mean_variance_ns": mean_var_ns,
            "sigma_point": sigma_point,
            "sigma_block": sigma_block,
            "support_ratio": support_ratio,
            "variance_ratio": mean_var_ratio,
            "mean_uncertainty_index": mean_uncertainty_index,
            "sill_total": float(sill_total),
            "ns_auto_disabled": bool(self._ns_auto_disabled),
            "ns_auto_disabled_cv": float(self._ns_auto_disabled_cv),
            **grade_stats,
            **audit_cv,
            "affine_correction_factor": _affine_correction_factor,
        }

        diagnostics = {
            "estimation_mode": "adaptive_local_rbf",
            "effective_drift_type": rbf_settings.drift,
            "n_subdomains": 1,
            "n_blocks_estimated": int(n_blocks),
            "n_blocks_total": int(n_blocks),
            "elapsed_seconds": elapsed,
            "neff_mean": float(np.nanmean(raw["ARBF_NEFF"])),
            "condnum_mean": float(np.nanmean(raw["ARBF_CONDNUM"])),
            "pum_enabled": partition.enabled,
            "use_normal_score": rbf_settings.use_normal_scores,
            "rbf_basis": rbf_kernel.basis,
            "rbf_shape_parameter": rbf_kernel.shape_parameter,
            "rbf_support_radius": rbf_kernel.support_radius,
            "kernel_type": rbf_kernel.basis,
            "timings": timings,
        }

        # Build swath data — volume-weighted block mean on informed
        # blocks only, with broad panels derived from block size and
        # variogram range. This is the primary ARBF Gate 3 input.
        cfg = self._config
        _panel_width = float(cfg.get("panel_width", 0.0) or 0.0)
        if _panel_width <= 0:
            # Auto: 3 SMU widths or 10% of range_max, whichever is larger.
            _block_dim = 1.0
            if self._block_sizes is not None:
                _bs = np.asarray(self._block_sizes, dtype=float)
                if _bs.ndim == 1 and _bs.size >= 1:
                    _block_dim = float(np.mean(_bs))
                elif _bs.ndim == 2 and _bs.size:
                    _block_dim = float(np.mean(_bs))
            _range_for_panel = float(cfg.get("range_max", 100.0) or 100.0)
            _panel_width = max(3.0 * _block_dim, 0.10 * _range_for_panel)
        _informed_mask = raw.get("ARBF_INFORMED")
        if _informed_mask is not None:
            _informed_mask = np.asarray(_informed_mask, dtype=bool)
            logger.info(
                "ARBF panel QA: %d / %d blocks informed (panel_width=%.1f m)",
                int(_informed_mask.sum()), int(_informed_mask.size),
                _panel_width,
            )
        _composite_support = cfg.get("composite_support")
        swath_data = self._build_swath_data(
            grades, self._centroids,
            composite_coords=self._coords, composite_values=self._values,
            composite_weights=self._declustering_weights,
            block_volumes=self._block_volumes,
            composite_support=_composite_support,
            informed_mask=_informed_mask,
            panel_width=_panel_width,
        )

        # ── Gate 3: support-aware panel reproduction ───────────────────
        # Aggregate block panel mean vs declustered composite panel mean
        # across all three axes. Drops panels with too few blocks /
        # composites (_build_swath_data sets those to NaN). This is the
        # primary ARBF validation gate.
        gate3 = self._compute_gate3(swath_data)
        audit_record.update(gate3)

        # ── Support-aware panel swaths via the CP-patched helper ───────
        # ``support_swath_plots`` aggregates blocks onto a coarse panel
        # lattice, re-normalises declustering weights per panel, and
        # returns per-panel block metal and declustered composite metal.
        # The UI prefers ``support_swath_data`` over the basic
        # ``swath_data`` — populating it switches the primary grade
        # curve to the CP-reviewed support-aware reference.
        support_swath_result = None
        try:
            from geostats.arbf.cross_validation import support_swath_plots
            _block_sizes_arg = self._block_sizes
            if _block_sizes_arg is not None and _block_sizes_arg.ndim == 1:
                _block_sizes_arg = np.broadcast_to(
                    _block_sizes_arg, (self._centroids.shape[0], 3),
                ).copy()
            support_swath_result = support_swath_plots(
                block_centroids=self._centroids,
                block_estimates=grades,
                composite_coords=self._coords,
                composite_values=self._values,
                block_sizes=_block_sizes_arg,
                declustering_weights=self._declustering_weights,
            )
            if support_swath_result is not None:
                logger.info(
                    "Support swath (CP-patched): %d panels, %d with data",
                    int(getattr(support_swath_result, "n_panels_total", 0)),
                    int(getattr(support_swath_result, "n_panels_with_data", 0)),
                )
        except Exception as _exc:
            logger.warning(
                "Support swath computation failed — falling back to "
                "basic swath_data only: %s", _exc,
            )
        timings["7_swath_and_packaging"] = time.time() - t_step
        timings["total"] = elapsed

        # Log timing breakdown
        logger.info("ARBF Timing Breakdown:")
        logger.info("  %-30s %8.2fs", "1. Parameter setup", timings.get("1_parameter_setup", 0))
        logger.info("  %-30s %8.2fs", "2. Estimator init (KD-tree, NS)", timings.get("2_estimator_init", 0))
        logger.info("  %-30s %8.2fs", "3. Block estimation", timings.get("3_block_estimation", 0))
        logger.info("  %-30s %8.2fs", "4. Grade clipping", timings.get("4_grade_clipping", 0))
        logger.info("  %-30s %8.2fs", "5. Classification", timings.get("5_classification", 0))
        logger.info("  %-30s %8.2fs", "6. Cross-validation", timings.get("6_cross_validation", 0))
        logger.info("  %-30s %8.2fs", "7. Swath + packaging", timings.get("7_swath_and_packaging", 0))
        logger.info("  %-30s %8.2fs", "TOTAL", elapsed)
        logger.info("  Blocks: %s, Per block: %.3f ms",
                     f"{n_blocks:,}", elapsed / max(n_blocks, 1) * 1000)

        self._report(100, f"ARBF v2: Complete in {elapsed:.1f}s ({n_blocks:,} blocks)")

        return _ResultDict(
            grades=grades,
            variances=variances,
            variances_raw=variances,
            variances_calibrated=variances_calibrated,
            classifications=classifications,
            classification_names=_CLASS_NAMES[classifications],
            stitching_variance=stitch_var,
            total_blending_variance=variances + stitch_var,
            cv_result=cv_result,
            swath_data=swath_data,
            support_swath_data=support_swath_result,
            conditional_bias_result=None,
            cos_result=None,
            classification_result=None,
            audit_record=audit_record,
            diagnostics=diagnostics,
            fail_flags=fail_flags,
            neff=raw["ARBF_NEFF"],
        )

    # ── Domain-separated estimation ──────────────────────────────────────

    def _estimate_by_domain(
        self,
        anisotropy: Anisotropy,
        variogram: VariogramModel,
        rbf_settings: RBFSettings,
        neighbourhood: NeighbourhoodSettings,
        block_settings: BlockSettings,
        partition: PartitionSettings,
        t0: float,
        rbf_kernel: Optional[RBFKernel] = None,
    ) -> _ResultDict:
        """Run estimation per domain, then merge results.

        Every step runs independently per domain: estimation, CV, swath
        data, grade clipping.  No global fallbacks or mixed-population
        statistics.  Each domain behaves as if it is the only dataset.
        """
        unique_domains = np.unique(self._composite_domains)
        n_domains = len(unique_domains)
        n_blocks = self._centroids.shape[0]

        all_grades = np.full(n_blocks, np.nan, dtype=float)
        all_variances = np.full(n_blocks, np.nan, dtype=float)
        all_variances_ns = np.full(n_blocks, np.nan, dtype=float)
        all_stitch = np.zeros(n_blocks, dtype=float)
        all_fail = np.zeros(n_blocks, dtype=int)
        all_neff = np.full(n_blocks, np.nan, dtype=float)
        all_uncertainty = np.ones(n_blocks, dtype=float)

        # Per-domain CV results and swath data
        per_domain_cv: Dict[str, Any] = {}
        per_domain_swath: Dict[str, Any] = {}
        run_cv = self._config.get("run_cv", True)
        cv_max_cfg = int(self._config.get("cv_max_samples", 300))

        # Per-domain variograms from registry (may be empty)
        domain_variograms = self._config.get("domain_variograms", {})

        # Log domain variogram coverage
        if domain_variograms:
            available = [str(d) for d in unique_domains if str(d) in domain_variograms]
            missing = [str(d) for d in unique_domains if str(d) not in domain_variograms]
            logger.info(
                "ARBF domain variograms: %d/%d domains have per-domain variograms. "
                "Available: %s. Missing (will use global): %s",
                len(available), n_domains, available or "(none)", missing or "(none)",
            )
        else:
            logger.info(
                "ARBF: No per-domain variograms provided — all %d domains "
                "will use the global variogram parameters.",
                n_domains,
            )

        self._report(10, f"ARBF v2: Estimating {n_domains} domains...")

        for di, domain in enumerate(unique_domains):
            domain_key = str(domain)
            pct_base = 10 + int(60 * di / max(n_domains, 1))
            pct_end = 10 + int(60 * (di + 1) / max(n_domains, 1))

            comp_mask = self._composite_domains == domain
            block_mask = self._block_domains == domain

            domain_coords = self._coords[comp_mask]
            domain_values = self._values[comp_mask]
            domain_centroids = self._centroids[block_mask]

            if domain_coords.shape[0] < 2 or domain_centroids.shape[0] == 0:
                logger.warning("Domain %s: skipped (< 2 composites or 0 blocks)", domain)
                continue

            self._report(pct_base, f"ARBF v2: Domain {domain} ({domain_coords.shape[0]} composites)...")

            # ── Resolve per-domain variogram parameters ─────────────
            d_anisotropy = anisotropy
            d_variogram = variogram
            d_rbf_settings = rbf_settings
            d_rbf_kernel = rbf_kernel

            domain_vario_dict = domain_variograms.get(domain_key)
            if domain_vario_dict is not None:
                domain_params = self._try_v2_variogram_direct(
                    vario_override=domain_vario_dict,
                )
                if domain_params is not None:
                    d_anisotropy = domain_params["anisotropy"]
                    d_variogram = domain_params["variogram"]
                    d_rbf_settings = domain_params["rbf_settings"]
                    d_rbf_kernel = (
                        domain_params.get("rbf_kernel")
                        or self._build_rbf_kernel(d_variogram)
                    )
                    logger.info(
                        "Domain %s: Using PER-DOMAIN variogram "
                        "(ranges=%s, nugget=%.4f, sill=%.4f)",
                        domain_key, d_anisotropy.ranges,
                        d_variogram.nugget_micro, d_variogram.partial_sill,
                    )
                else:
                    logger.warning(
                        "Domain %s: Per-domain variogram stored but v2 "
                        "extraction failed — falling back to global params.",
                        domain_key,
                    )
            else:
                logger.info(
                    "Domain %s: No per-domain variogram — using global params.",
                    domain_key,
                )

            try:
                # Slice declustering weights by the domain's composite mask
                _domain_w = (
                    self._declustering_weights[comp_mask]
                    if self._declustering_weights is not None else None
                )
                estimator = FastRBFEstimator(
                    coords=domain_coords,
                    values=domain_values,
                    anisotropy=d_anisotropy,
                    variogram=d_variogram,
                    rbf_settings=d_rbf_settings,
                    neighbourhood=neighbourhood,
                    block_settings=block_settings,
                    partition_settings=partition,
                    rbf_kernel=d_rbf_kernel,
                    sample_weights=_domain_w,
                )
                def _domain_progress(pct, msg, _b=pct_base, _e=pct_end):
                    scaled = _b + int(pct * (_e - _b - 2) / 100)
                    self._report(scaled, msg)
                raw = estimator.estimate_blocks(
                    domain_centroids, progress_callback=_domain_progress,
                )

                block_indices = np.where(block_mask)[0]
                all_grades[block_indices] = raw["ARBF_GRADE"]
                all_variances[block_indices] = raw["ARBF_VAR_BLOCK"]
                all_variances_ns[block_indices] = raw.get("ARBF_VAR_NS", raw["ARBF_VAR_BLOCK"])
                all_stitch[block_indices] = raw["ARBF_STITCH_VAR"]
                all_fail[block_indices] = raw["ARBF_FAIL_FLAG"]
                all_neff[block_indices] = raw["ARBF_NEFF"]
                all_uncertainty[block_indices] = raw.get("ARBF_UNCERTAINTY", 0.5)

                # ── Per-domain CV (using domain data only) ──────────────
                if run_cv and domain_coords.shape[0] >= 4:
                    self._report(
                        pct_end - 2,
                        f"ARBF v2: Cross-validation for domain {domain}...",
                    )
                    cv_max = min(cv_max_cfg, domain_coords.shape[0])
                    try:
                        domain_cv = leave_one_out_cv(
                            coords=domain_coords,
                            values=domain_values,
                            anisotropy=d_anisotropy,
                            variogram=d_variogram,
                            rbf_settings=d_rbf_settings,
                            neighbourhood=neighbourhood,
                            block_settings=block_settings,
                            max_points=min(cv_max, 300),
                            rbf_kernel=d_rbf_kernel,
                        )
                        per_domain_cv[domain_key] = domain_cv
                        logger.info(
                            "Domain %s CV: R²=%.3f, RMSE=%.4f, Slope=%.3f, "
                            "ME=%.4f (%d composites)",
                            domain, domain_cv.get("R2", float("nan")),
                            domain_cv.get("RMSE", float("nan")),
                            domain_cv.get("SLOPE", float("nan")),
                            domain_cv.get("ME", float("nan")),
                            domain_coords.shape[0],
                        )
                    except Exception as cv_exc:
                        logger.warning(
                            "Domain %s CV failed: %s", domain, cv_exc,
                        )
                elif run_cv:
                    logger.warning(
                        "Domain %s: skipping CV (only %d composites, need >= 4)",
                        domain, domain_coords.shape[0],
                    )

                # ── Per-domain swath data ───────────────────────────────
                per_domain_swath[domain_key] = self._build_swath_data(
                    all_grades[block_indices],
                    self._centroids[block_indices],
                    composite_coords=domain_coords,
                    composite_values=domain_values,
                    composite_weights=(
                        self._declustering_weights[comp_mask]
                        if self._declustering_weights is not None else None
                    ),
                )

            except Exception as exc:
                logger.error("Domain %s estimation failed: %s", domain, exc)
                block_indices = np.where(block_mask)[0]
                all_grades[block_indices] = float(np.mean(domain_values))
                all_variances[block_indices] = float(np.var(domain_values))
                all_variances_ns[block_indices] = float(np.var(domain_values))
                all_fail[block_indices] = 1
                all_uncertainty[block_indices] = 1.0  # fully uncertain

        # Fill any unassigned blocks with fallback
        unassigned = np.isnan(all_grades)
        n_unassigned = int(np.sum(unassigned))
        if n_unassigned > 0:
            logger.warning(
                "ARBF (domain mode): %d of %d blocks (%.1f%%) are not assigned "
                "to any domain. These blocks will be filled with the global "
                "composite mean (%.4f) and flagged as failed. Check that all "
                "block centroids have a matching domain label.",
                n_unassigned, n_blocks,
                100.0 * n_unassigned / max(n_blocks, 1),
                float(np.mean(self._values)),
            )
            all_grades[unassigned] = float(np.mean(self._values))
            all_variances[unassigned] = float(np.var(self._values))
            all_variances_ns[unassigned] = float(np.var(self._values))
            all_fail[unassigned] = 1
            all_uncertainty[unassigned] = 1.0

        # Grade clipping (prevent negatives or out-of-range values)
        clip_min = self._config.get("clip_min")
        clip_max = self._config.get("clip_max")
        if clip_min is not None:
            n_below = int(np.sum(all_grades < float(clip_min)))
            if n_below > 0:
                logger.warning(
                    "ARBF grade clipping (domain mode): %d of %d blocks (%.1f%%) "
                    "clipped to min=%.4f",
                    n_below, len(all_grades),
                    100.0 * n_below / max(len(all_grades), 1), float(clip_min),
                )
            all_grades = np.clip(all_grades, float(clip_min), None)
        if clip_max is not None:
            n_above = int(np.sum(all_grades > float(clip_max)))
            if n_above > 0:
                logger.warning(
                    "ARBF grade clipping (domain mode): %d of %d blocks (%.1f%%) "
                    "clipped to max=%.4f",
                    n_above, len(all_grades),
                    100.0 * n_above / max(len(all_grades), 1), float(clip_max),
                )
            all_grades = np.clip(all_grades, None, float(clip_max))

        # Even without user-specified clipping, warn about negative grades
        if clip_min is None:
            n_negative = int(np.sum(all_grades < 0.0))
            if n_negative > 0:
                logger.warning(
                    "ARBF (domain mode): %d of %d blocks (%.1f%%) have negative "
                    "estimated grades. Consider enabling 'Grade Clipping' with "
                    "min=0 in the ARBF panel for non-negative variables.",
                    n_negative, len(all_grades),
                    100.0 * n_negative / max(len(all_grades), 1),
                )

        # Classification (based on uncertainty index)
        self._report(72, "ARBF v2: Classifying blocks...")
        sill_total = variogram.sill_total if variogram else 1.0
        classifications = self._classify_blocks(all_uncertainty, all_fail, sill_total)

        # ── Assemble combined CV from per-domain results ────────────────
        cv_result = None
        audit_cv = {}
        if run_cv and per_domain_cv:
            # Concatenate per-domain actual/estimated/pred_std arrays and
            # recompute combined metrics from the pooled data.  This gives
            # correct coverage and standardised-residual statistics (which
            # cannot be averaged across domains).
            combined_actual: List[float] = []
            combined_estimated: List[float] = []
            combined_pred_std: List[float] = []

            for dcv in per_domain_cv.values():
                combined_actual.extend(dcv.get("actual", []))
                combined_estimated.extend(dcv.get("estimated", []))
                combined_pred_std.extend(dcv.get("pred_std", []))

            total_samples = len(combined_actual)

            if total_samples > 0:
                pooled = cv_metrics(
                    np.asarray(combined_actual, dtype=float),
                    np.asarray(combined_estimated, dtype=float),
                    np.asarray(combined_pred_std, dtype=float),
                )

                cv_result = {
                    "per_domain": per_domain_cv,
                    "combined": {
                        **pooled,
                        "n_samples": total_samples,
                        "n_domains": len(per_domain_cv),
                    },
                    # Top-level keys for backward compatibility
                    **pooled,
                    "slope_of_regression": pooled.get("SLOPE", 1.0),
                    "actual": combined_actual,
                    "estimated": combined_estimated,
                    "pred_std": combined_pred_std,
                }

                logger.info(
                    "ARBF CV combined (pooled across %d domains, "
                    "%d total samples): R²=%.3f, RMSE=%.4f, Slope=%.3f, ME=%.4f",
                    len(per_domain_cv), total_samples,
                    pooled.get("R2", float("nan")),
                    pooled.get("RMSE", float("nan")),
                    pooled.get("SLOPE", float("nan")),
                    pooled.get("ME", float("nan")),
                )

                audit_cv = {
                    "cv_slope_of_regression": pooled.get("SLOPE"),
                    "cv_r_squared": pooled.get("R2"),
                    "cv_rmse": pooled.get("RMSE"),
                    "cv_mean_error": pooled.get("ME"),
                    "cv_std_resid_mean": pooled.get("STD_RESID_MEAN"),
                    "cv_std_resid_var": pooled.get("STD_RESID_VAR"),
                    "cv_cover_50": pooled.get("COVER_50"),
                    "cv_cover_68": pooled.get("COVER_68"),
                    "cv_cover_90": pooled.get("COVER_90"),
                    "cv_cover_95": pooled.get("COVER_95"),
                    "cv_per_domain": {
                        dkey: {
                            "slope": dcv.get("SLOPE"),
                            "r2": dcv.get("R2"),
                            "rmse": dcv.get("RMSE"),
                            "me": dcv.get("ME"),
                            "n_samples": len(dcv.get("actual", [])),
                        }
                        for dkey, dcv in per_domain_cv.items()
                    },
                }

        # ── Variance calibration (domain path) ────────────────────
        # Same post-estimation fit as the single-domain path.
        all_variances_calibrated, _calib_audit_d = self._calibrate_variance(
            all_variances, cv_result,
            mode=self._config.get("variance_calibration_mode", "global"),
        )
        audit_cv.update(_calib_audit_d)

        elapsed = time.time() - t0
        self._report(95, "ARBF v2: Packaging results...")

        n_measured = int(np.sum(classifications == 0))
        n_indicated = int(np.sum(classifications == 1))
        n_inferred = int(np.sum(classifications == 2))
        n_unclassified_cls = int(np.sum(classifications == 3))
        fail_ratio = float(np.mean(all_fail.astype(float)))

        finite_vars = all_variances[np.isfinite(all_variances)]
        if finite_vars.size > 0 and sill_total > 1e-12:
            mean_var = float(np.mean(finite_vars))
            mean_var_ratio = mean_var / sill_total
            sigma_point = float(np.sqrt(sill_total))
            sigma_block = float(np.sqrt(max(mean_var, 0.0)))
            support_ratio = sigma_block / sigma_point if sigma_point > 1e-12 else float("nan")
        else:
            mean_var = 0.0
            mean_var_ratio = float("nan")
            sigma_point = float(np.sqrt(max(sill_total, 0.0)))
            sigma_block = float("nan")
            support_ratio = float("nan")

        finite_var_ns = all_variances_ns[np.isfinite(all_variances_ns)]
        mean_var_ns = float(np.mean(finite_var_ns)) if finite_var_ns.size > 0 else float("nan")
        mean_uncertainty_index = float(np.nanmean(all_uncertainty))

        finite_grades = all_grades[np.isfinite(all_grades)]
        grade_stats = {}
        if finite_grades.size > 0:
            grade_stats = {
                "grade_mean": float(np.mean(finite_grades)),
                "grade_median": float(np.median(finite_grades)),
                "grade_std": float(np.std(finite_grades)),
                "grade_min": float(np.min(finite_grades)),
                "grade_max": float(np.max(finite_grades)),
                "grade_p05": float(np.percentile(finite_grades, 5)),
                "grade_p95": float(np.percentile(finite_grades, 95)),
                "max_to_median_ratio": float(
                    np.max(finite_grades) / max(np.median(finite_grades), 1e-12)
                ),
                "n_negative_grades": int(np.sum(finite_grades < 0)),
                "n_bounded_grades": int(np.sum(
                    (finite_grades == np.min(finite_grades)) | (finite_grades == np.max(finite_grades))
                )),
            }

        audit_record = {
            "num_composites": int(self._coords.shape[0]),
            "n_blocks_estimated": int(n_blocks),
            "n_blocks_total": int(n_blocks),
            "n_blocks_active": int(np.sum(np.isfinite(all_grades))),
            "elapsed_seconds": elapsed,
            "measured_blocks": n_measured,
            "indicated_blocks": n_indicated,
            "inferred_blocks": n_inferred,
            "unclassified_blocks": n_unclassified_cls,
            "fail_ratio": fail_ratio,
            "mean_variance_ratio": mean_var_ratio,
            "mean_variance_ns": mean_var_ns,
            "sigma_point": sigma_point,
            "sigma_block": sigma_block,
            "support_ratio": support_ratio,
            "variance_ratio": mean_var_ratio,
            "mean_uncertainty_index": mean_uncertainty_index,
            "sill_total": float(sill_total),
            "n_domains": int(n_domains),
            "n_unassigned_blocks": n_unassigned,
            **grade_stats,
            **audit_cv,
        }

        diagnostics = {
            "estimation_mode": "adaptive_local_rbf",
            "effective_drift_type": rbf_settings.drift,
            "n_subdomains": int(n_domains),
            "n_blocks_estimated": int(n_blocks),
            "n_blocks_total": int(n_blocks),
            "elapsed_seconds": elapsed,
            "pum_enabled": partition.enabled,
            "use_normal_score": rbf_settings.use_normal_scores,
            "kernel_type": variogram.model,
            "n_unassigned_blocks": n_unassigned,
            "block_domain_assignment": {
                str(d): int(np.sum(self._block_domains == d))
                for d in unique_domains
            },
        }

        # Swath data: per-domain + combined global. Passes through the
        # same panel-QA machinery as the single-domain path so Gate 3
        # is computed on domain-separated runs too.
        cfg_local = self._config
        _panel_width = float(cfg_local.get("panel_width", 0.0) or 0.0)
        if _panel_width <= 0:
            _bd = 1.0
            if self._block_sizes is not None:
                _bs = np.asarray(self._block_sizes, dtype=float)
                if _bs.size:
                    _bd = float(np.mean(_bs))
            _range_for_panel = float(cfg_local.get("range_max", 100.0) or 100.0)
            _panel_width = max(3.0 * _bd, 0.10 * _range_for_panel)

        swath_data = {
            "per_domain": per_domain_swath,
            **self._build_swath_data(
                all_grades, self._centroids,
                composite_coords=self._coords, composite_values=self._values,
                composite_weights=self._declustering_weights,
                block_volumes=self._block_volumes,
                panel_width=_panel_width,
            ),
        }

        # Gate 3 on the domain-merged global swath.
        gate3_domain = self._compute_gate3({
            k: v for k, v in swath_data.items()
            if k in ("x", "y", "z")
        })
        audit_record.update(gate3_domain)

        self._report(100, "ARBF v2: Complete")

        return _ResultDict(
            grades=all_grades,
            variances=all_variances,
            variances_raw=all_variances,
            variances_calibrated=all_variances_calibrated,
            classifications=classifications,
            classification_names=_CLASS_NAMES[classifications],
            stitching_variance=all_stitch,
            total_blending_variance=all_variances + all_stitch,
            cv_result=cv_result,
            swath_data=swath_data,
            support_swath_data=None,
            conditional_bias_result=None,
            cos_result=None,
            classification_result=None,
            audit_record=audit_record,
            diagnostics=diagnostics,
            fail_flags=all_fail,
            neff=all_neff,
        )

    # ── Block estimation with progress ───────────────────────────────────

    def _estimate_blocks_with_progress(
        self,
        estimator: FastRBFEstimator,
        centroids: np.ndarray,
        pct_start: int,
        pct_end: int,
    ) -> Dict[str, np.ndarray]:
        """Estimate blocks using the fast global-solve path with progress."""
        n = centroids.shape[0]

        def _progress(pct: int, msg: str):
            # Map 0-100 to pct_start-pct_end
            scaled = pct_start + int(pct * (pct_end - pct_start) / 100)
            self._report(scaled, msg)

        return estimator.estimate_blocks(centroids, progress_callback=_progress)

    # ── Swath data builder ───────────────────────────────────────────────

    @staticmethod
    def _build_swath_data(
        grades: np.ndarray,
        centroids: np.ndarray,
        n_bins: int = 10,
        composite_coords: Optional[np.ndarray] = None,
        composite_values: Optional[np.ndarray] = None,
        composite_weights: Optional[np.ndarray] = None,
        *,
        block_volumes: Optional[np.ndarray] = None,
        composite_support: Optional[np.ndarray] = None,
        informed_mask: Optional[np.ndarray] = None,
        panel_width: Optional[float] = None,
        min_blocks_per_panel: int = 4,
        min_composites_per_panel: int = 3,
    ) -> Dict[str, Any]:
        """Build swath data binned along X, Y, Z axes.

        Returns per-bin: raw composite mean, declustered composite mean,
        block mean (volume-weighted when ``block_volumes`` supplied),
        composite count, block count, and contained metal on both sides.

        When ``informed_mask`` is supplied, only informed blocks feed the
        block panel mean and ``n_block``. When ``panel_width`` is supplied
        the bin edges are uniform panels of that width (the user's
        3–5 SMU rule) instead of ``n_bins``-linspace slices — undersized
        panels (< ``min_blocks_per_panel`` / ``min_composites_per_panel``)
        are set to NaN so edge-effects don't corrupt Gate 3 metrics.
        """
        swath = {}
        axis_labels = {"x": "X", "y": "Y", "z": "Z"}
        axis_names = ["x", "y", "z"]

        w = None
        if (
            composite_weights is not None
            and composite_values is not None
            and np.asarray(composite_weights).size == np.asarray(composite_values).size
        ):
            w = np.asarray(composite_weights, dtype=float).ravel()
            if not np.any(np.isfinite(w)) or float(np.nansum(w)) <= 0:
                w = None

        # Composite support lengths — optional length-weighting so longer
        # intervals carry more weight in both the declustered mean and
        # the contained-metal calculation.
        supp = None
        if (
            composite_support is not None
            and composite_values is not None
            and np.asarray(composite_support).size == np.asarray(composite_values).size
        ):
            supp = np.asarray(composite_support, dtype=float).ravel()
            supp = np.where(np.isfinite(supp) & (supp > 0), supp, 0.0)
            if float(np.sum(supp)) <= 0:
                supp = None

        # Block volumes — used as weights for the volume-weighted panel
        # mean and for contained-metal on the block side. Defaults to a
        # uniform weight so the refactor stays back-compatible.
        n_blocks_total = int(centroids.shape[0])
        if (
            block_volumes is not None
            and np.asarray(block_volumes).size == n_blocks_total
        ):
            vol = np.asarray(block_volumes, dtype=float).ravel()
            vol = np.where(np.isfinite(vol) & (vol > 0), vol, 0.0)
        else:
            vol = np.ones(n_blocks_total, dtype=float)

        # Informed-blocks mask — drops extrapolated cells from the
        # block panel mean entirely.
        if (
            informed_mask is not None
            and np.asarray(informed_mask).size == n_blocks_total
        ):
            informed = np.asarray(informed_mask, dtype=bool).ravel()
        else:
            informed = np.ones(n_blocks_total, dtype=bool)

        # Scale sanity check — warn if composite values and block grades
        # are on wildly different scales (typically NS vs raw).
        try:
            if composite_values is not None and grades is not None:
                _gm = float(np.nanmean(grades))
                _vm = float(np.nanmean(composite_values))
                _gm_a = abs(_gm) if np.isfinite(_gm) else 0.0
                _vm_a = abs(_vm) if np.isfinite(_vm) else 0.0
                if min(_gm_a, _vm_a) > 1e-6:
                    _ratio = max(_gm_a, _vm_a) / min(_gm_a, _vm_a)
                    if _ratio > 50.0:
                        logger.warning(
                            "Swath scale mismatch: block-mean=%.4g vs "
                            "composite-mean=%.4g (ratio %.1fx) — curves "
                            "may be on different scales (NS vs raw?).",
                            _gm, _vm, _ratio,
                        )
        except Exception:
            pass

        # One-shot negative-weight warning
        _neg_warned = {"done": False}
        def _warn_negative_weights_once() -> None:
            if not _neg_warned["done"]:
                logger.warning(
                    "Swath declustering: some bins contain negative "
                    "weights — skipping declustered mean for those bins.",
                )
                _neg_warned["done"] = True

        _debug = logger.isEnabledFor(logging.DEBUG)

        for ax_i, ax_name in enumerate(axis_names):
            coords_ax = centroids[:, ax_i]
            ax_lo = float(coords_ax.min())
            ax_hi = float(coords_ax.max())

            if panel_width is not None and panel_width > 0:
                # Broad panels of user-specified width. Final edge is
                # snapped to ax_hi so the last panel absorbs any remainder.
                _edges = np.arange(ax_lo, ax_hi, float(panel_width))
                edges = np.concatenate([_edges, [ax_hi]])
                if edges.size < 2:
                    edges = np.array([ax_lo, ax_hi], dtype=float)
                nb = edges.size - 1
            else:
                edges = np.linspace(ax_lo, ax_hi, n_bins + 1)
                nb = n_bins

            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            bin_means_est = np.full(nb, np.nan, dtype=float)
            bin_means_est_vw = np.full(nb, np.nan, dtype=float)
            bin_means_act = np.full(nb, np.nan, dtype=float)
            bin_means_decl = np.full(nb, np.nan, dtype=float)
            bin_metal_block = np.full(nb, np.nan, dtype=float)
            bin_metal_decl = np.full(nb, np.nan, dtype=float)
            bin_metal_raw = np.full(nb, np.nan, dtype=float)
            n_block = np.zeros(nb, dtype=int)
            n_comp = np.zeros(nb, dtype=int)

            # panel_volume: total volume of informed blocks in the panel;
            # used to normalise the composite-side "equivalent metal" so
            # both metal curves share the same volumetric support.
            panel_volumes = np.zeros(nb, dtype=float)

            digitized = np.digitize(coords_ax, edges) - 1
            digitized = np.clip(digitized, 0, nb - 1)
            for bi in range(nb):
                mask_all = digitized == bi
                mask_inf = mask_all & informed
                cnt_inf = int(np.sum(mask_inf))
                n_block[bi] = cnt_inf
                if cnt_inf >= max(min_blocks_per_panel, 1):
                    g_bi = grades[mask_inf]
                    v_bi = vol[mask_inf]
                    finite = np.isfinite(g_bi) & np.isfinite(v_bi) & (v_bi > 0)
                    if finite.any():
                        g_f = g_bi[finite]
                        v_f = v_bi[finite]
                        _vs = float(np.sum(v_f))
                        if _vs > 0:
                            panel_volumes[bi] = _vs
                            _vw = float(np.sum(v_f * g_f) / _vs)
                            bin_means_est_vw[bi] = _vw
                            bin_means_est[bi] = float(np.mean(g_f))
                            bin_metal_block[bi] = float(np.sum(v_f * g_f))
                elif cnt_inf > 0:
                    # Undersupported — still report the simple mean but
                    # no volume-weighted mean (plot picks this up as NaN).
                    g_bi = grades[mask_inf]
                    g_f = g_bi[np.isfinite(g_bi)]
                    if g_f.size:
                        bin_means_est[bi] = float(np.mean(g_f))

            if composite_coords is not None and composite_values is not None:
                comp_ax = composite_coords[:, ax_i]
                comp_dig = np.digitize(comp_ax, edges) - 1
                comp_dig = np.clip(comp_dig, 0, nb - 1)
                for bi in range(nb):
                    mask = comp_dig == bi
                    cnt = int(np.sum(mask))
                    n_comp[bi] = cnt
                    if cnt >= max(min_composites_per_panel, 1):
                        vals_bi = composite_values[mask]
                        finite_m = np.isfinite(vals_bi)
                        vals_bi = vals_bi[finite_m]
                        if not vals_bi.size:
                            continue
                        bin_means_act[bi] = float(np.mean(vals_bi))
                        # Declustered mean uses effective weight = w * support.
                        w_eff = None
                        if w is not None:
                            wb = w[mask][finite_m]
                            if np.any(wb < 0):
                                _warn_negative_weights_once()
                            else:
                                w_eff = wb
                        if supp is not None:
                            sb = supp[mask][finite_m]
                            if w_eff is None:
                                w_eff = sb
                            else:
                                w_eff = w_eff * sb
                        if w_eff is not None and float(np.sum(w_eff)) > 0:
                            _ws = float(np.sum(w_eff))
                            bin_means_decl[bi] = float(
                                np.sum(w_eff * vals_bi) / _ws
                            )
                        else:
                            bin_means_decl[bi] = bin_means_act[bi]

                        # Composite-side "equivalent contained metal":
                        # the declustered / raw composite grade rescaled
                        # to the panel's block volume so it's directly
                        # comparable to sum(v_j * g_j) on the block side.
                        # This converts point-support grades to the same
                        # volumetric support and makes the metal bars
                        # physically meaningful alongside block metal.
                        _pvol = panel_volumes[bi]
                        if _pvol > 0:
                            if np.isfinite(bin_means_decl[bi]):
                                bin_metal_decl[bi] = (
                                    float(bin_means_decl[bi]) * _pvol
                                )
                            if np.isfinite(bin_means_act[bi]):
                                bin_metal_raw[bi] = (
                                    float(bin_means_act[bi]) * _pvol
                                )

                    if _debug:
                        _sw = float(np.sum(w[mask])) if (w is not None and cnt) else float("nan")
                        logger.debug(
                            "swath[%s] bin=%d n_block=%d n_comp=%d sum_w=%.4g "
                            "raw=%.4g decl=%.4g blk=%.4g blk_vw=%.4g",
                            ax_name, bi, n_block[bi], cnt, _sw,
                            bin_means_act[bi], bin_means_decl[bi],
                            bin_means_est[bi], bin_means_est_vw[bi],
                        )
            else:
                bin_means_act = bin_means_est.copy()
                bin_means_decl = bin_means_est.copy()

            # Back-compat: when no block_volumes supplied the old simple
            # mean stays as the "mean_estimated" field so legacy callers
            # get identical output. The new volume-weighted field is
            # separate and only populated when weights are supplied.
            if block_volumes is None and informed_mask is None:
                est_for_output = bin_means_est
            else:
                # When the caller explicitly requested volume weighting
                # or informed filtering, promote the volume-weighted line
                # to "mean_estimated" so the UI picks it up as primary.
                est_for_output = np.where(
                    np.isfinite(bin_means_est_vw), bin_means_est_vw, bin_means_est,
                )

            swath[ax_name] = {
                "slice_positions": bin_centers.tolist(),
                "mean_estimated": est_for_output.tolist(),
                "mean_estimated_volume_weighted": bin_means_est_vw.tolist(),
                "mean_actual": bin_means_act.tolist(),
                "mean_actual_declustered": bin_means_decl.tolist(),
                "metal_block": bin_metal_block.tolist(),
                "metal_composite_declustered": bin_metal_decl.tolist(),
                "metal_composite_raw": bin_metal_raw.tolist(),
                "n_block": n_block.tolist(),
                "n_composite": n_comp.tolist(),
                "panel_width": float(panel_width) if panel_width else None,
                "axis": axis_labels[ax_name],
            }
        return swath

    @staticmethod
    def _calibrate_variance(
        variances_raw: np.ndarray,
        cv_result: Optional[Dict[str, Any]],
        mode: str = "global",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Post-estimation variance calibration.

        Fits a non-negative linear mapping ``residual^2 ≈ a + b * var_raw``
        against the leave-one-out CV residuals, then returns
        ``var_cal = a + b * var_raw`` applied to every block.

        The mean estimate is NOT touched. Only the predictive variance
        is rescaled so its empirical coverage matches the observed
        residual spread.

        Robustness:
          * fit is done on the middle 90 % of squared residuals
            (trimmed non-negative least squares) to avoid heavy-tail
            samples dominating the fit
          * uses scipy.optimize.nnls which is deterministic and has
            no random initialisation
          * falls back to identity mapping (a=0, b=1) if CV is absent,
            trivially perfect, or the fit fails

        Parameters
        ----------
        variances_raw : np.ndarray
            Per-block raw predictive variances.
        cv_result : dict or None
            Output of ``leave_one_out_cv``; must contain ``actual``,
            ``estimated``, and ``pred_std`` arrays for the fit. For
            domain-path merged results this is the pooled concatenation.
        mode : str
            ``"global"`` (default) — one (a, b) for all blocks.
            ``"none"`` — identity mapping, only diagnostics are reported.

        Returns
        -------
        variances_calibrated : np.ndarray
            Same shape as ``variances_raw``. NaN inputs stay NaN.
        audit : dict
            Fields: calibration_enabled, calibration_mode, calibration_model,
            calibration_a, calibration_b, calibration_n_samples,
            calibration_std_z_mean_before/after, calibration_std_z_std_before/after,
            calibration_cover_68_before/after, calibration_cover_95_before/after.
        """
        var_raw_arr = np.asarray(variances_raw, dtype=float)
        audit: Dict[str, Any] = {
            "calibration_enabled": False,
            "calibration_mode": "none",
            "calibration_model": "linear",
            "calibration_a": 0.0,
            "calibration_b": 1.0,
            "calibration_n_samples": 0,
        }

        # Disabled mode → identity
        if mode == "none":
            audit["calibration_mode"] = "none"
            return var_raw_arr.copy(), audit

        # Extract CV residual arrays
        if not isinstance(cv_result, dict):
            audit["calibration_skip_reason"] = "no_cv_result"
            return var_raw_arr.copy(), audit

        act = np.asarray(cv_result.get("actual", []), dtype=float)
        est = np.asarray(cv_result.get("estimated", []), dtype=float)
        pstd = np.asarray(cv_result.get("pred_std", []), dtype=float)

        if act.size == 0 or est.size != act.size or pstd.size != act.size:
            audit["calibration_skip_reason"] = "cv_arrays_missing_or_mismatched"
            return var_raw_arr.copy(), audit

        mask = np.isfinite(act) & np.isfinite(est) & np.isfinite(pstd) & (pstd > 0)
        if int(mask.sum()) < 10:
            audit["calibration_skip_reason"] = "too_few_cv_samples"
            return var_raw_arr.copy(), audit

        residual = act[mask] - est[mask]
        var_cv = pstd[mask] ** 2
        resid_sq = residual ** 2

        # Standardised residual diagnostics BEFORE calibration
        z_before = residual / np.sqrt(np.maximum(var_cv, 1e-18))
        z_mean_before = float(np.mean(z_before))
        z_std_before = float(np.std(z_before))
        cov68_before = float(np.mean(np.abs(z_before) <= 1.0))
        cov95_before = float(np.mean(np.abs(z_before) <= 1.959964))

        # ── Scale calibration: b = mean(z²) on the full sample ───
        # The leave-one-out CV's pred_std comes from the PRESS formula
        # and is nearly constant across samples (it only varies through
        # the leverage term 1 - h_ii). A linear NNLS fit against a
        # near-constant predictor is degenerate; a median-of-ratios
        # estimator is biased low on heavy-tail data.
        #
        # The statistic we actually want is the empirical variance of
        # the standardised residuals on the FULL sample. Setting
        # b = mean(z²) drives std(z_after) on the full sample to
        # exactly 1 by construction, because
        #   z_after_i = z_before_i / sqrt(b)
        #   mean(z_after²) = mean(z_before²) / b = 1.
        #
        # We compute b from all finite CV samples so the calibration
        # target matches the metric the tests (and the user) measure.
        # A heavy-tail sanity cap prevents one pathological outlier
        # from pulling b to ∞: the cap is 4 × robust_MAD² which equals
        # ~8 × (median|z|)², i.e. the tail of z is squashed only when
        # a single sample contributes more than 4× the robust core
        # variance.
        z_values = residual / np.sqrt(np.maximum(var_cv, 1e-18))
        z_sq = z_values ** 2
        b_full = float(np.mean(z_sq)) if z_sq.size else 1.0

        # Heavy-tail sanity: cap b at 4 × Gaussian-equivalent robust
        # variance (MAD-based). When the data is light-tailed the
        # two estimators agree and the cap is inactive; when it is
        # heavy-tailed the cap protects against one outlier
        # dominating the fit.
        if z_values.size >= 20:
            med_abs = float(np.median(np.abs(z_values)))
            mad_var = (med_abs / 0.6744897501960817) ** 2
            b_cap = 4.0 * max(mad_var, 1e-18)
            b_hat = min(b_full, b_cap)
        else:
            b_hat = b_full

        if not np.isfinite(b_hat) or b_hat <= 1e-12:
            b_hat = max(b_full, 1.0)
        a_hat = 0.0

        # Apply calibration: var_cal = a + b * var_raw, clamped to >= 0
        with np.errstate(invalid="ignore"):
            var_cal = a_hat + b_hat * var_raw_arr
        var_cal = np.where(
            np.isfinite(var_raw_arr),
            np.maximum(var_cal, 0.0),
            var_raw_arr,
        )

        # Diagnostics AFTER calibration (on same CV residuals)
        var_cv_cal = a_hat + b_hat * var_cv
        var_cv_cal = np.maximum(var_cv_cal, 1e-18)
        z_after = residual / np.sqrt(var_cv_cal)
        z_mean_after = float(np.mean(z_after))
        z_std_after = float(np.std(z_after))
        cov68_after = float(np.mean(np.abs(z_after) <= 1.0))
        cov95_after = float(np.mean(np.abs(z_after) <= 1.959964))

        audit.update({
            "calibration_enabled": True,
            "calibration_mode": "global",
            "calibration_model": "linear",
            "calibration_a": a_hat,
            "calibration_b": b_hat,
            "calibration_n_samples": int(mask.sum()),
            "calibration_n_trimmed_used": int(z_values.size),
            "calibration_std_z_mean_before": z_mean_before,
            "calibration_std_z_std_before": z_std_before,
            "calibration_std_z_mean_after": z_mean_after,
            "calibration_std_z_std_after": z_std_after,
            "calibration_cover_68_before": cov68_before,
            "calibration_cover_68_after": cov68_after,
            "calibration_cover_95_before": cov95_before,
            "calibration_cover_95_after": cov95_after,
        })
        return var_cal, audit

    @staticmethod
    def _compute_gate3(swath_data: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate Gate 3 panel-reproduction metrics across X/Y/Z.

        The *relative* bias ``(blk - dec) / dec`` is ill-defined when the
        declustered composite mean is close to zero — which happens on
        stationary Gaussian data (FE_PCT, normal-score residuals, demeaned
        assays, etc.), where the correct sign of the diagnostic should
        still be PASS but ``|dec| ≈ 0`` forces the ratio to explode.

        Safeguard: when ``|dec_mean|`` is small relative to ``dec_std``
        (ratio < 0.1), fall back to a normalised-by-spread metric instead
        of the relative bias. Specifically we report
        ``(blk_mean − dec_mean) / dec_std`` as ``gate3_grade_bias`` so
        the thresholds still work (a 15 % shift in mean per-panel-std is
        comparable to a 15 % relative shift on a positive-reference
        distribution). The raw relative bias is still exposed as
        ``gate3_panel_grade_bias_rel`` for backwards compatibility, and
        ``gate3_reference_mode`` records which metric was used.

        Returns keys ready to splice into the audit_record:
        ``gate3_panel_grade_bias``, ``gate3_panel_metal_bias``,
        ``gate3_valid_panels``, ``gate3_total_panels``, ``gate3_status``,
        ``gate3_reference_mode``, ``gate3_panel_grade_bias_rel``.
        """
        blk_all: list = []
        dec_all: list = []
        mblk_tot = 0.0
        mdec_tot = 0.0
        n_valid = 0
        n_total = 0
        for ax in ("x", "y", "z"):
            sd = swath_data.get(ax)
            if not sd:
                continue
            vw = sd.get("mean_estimated_volume_weighted", sd.get("mean_estimated"))
            blk = np.asarray(vw, float)
            dec = np.asarray(sd.get("mean_actual_declustered", []), float)
            mblk = np.asarray(sd.get("metal_block", []), float)
            mdec = np.asarray(sd.get("metal_composite_declustered", []), float)
            valid = np.isfinite(blk) & np.isfinite(dec)
            n_valid += int(valid.sum())
            n_total += int(blk.size)
            if valid.any():
                blk_all.extend(blk[valid].tolist())
                dec_all.extend(dec[valid].tolist())
            mvalid = np.isfinite(mblk) & np.isfinite(mdec)
            if mvalid.any():
                mblk_tot += float(np.sum(mblk[mvalid]))
                mdec_tot += float(np.sum(mdec[mvalid]))

        blk_arr = np.asarray(blk_all, float)
        dec_arr = np.asarray(dec_all, float)

        grade_bias_rel = float("nan")
        grade_bias_used = float("nan")
        ref_mode = "unknown"
        if dec_arr.size > 0:
            blk_mean = float(np.mean(blk_arr))
            dec_mean = float(np.mean(dec_arr))
            dec_std = float(np.std(dec_arr)) if dec_arr.size > 1 else 0.0
            # Classical relative bias (kept for back-compat + logging).
            if abs(dec_mean) > 1e-12:
                grade_bias_rel = (blk_mean - dec_mean) / dec_mean
            # Near-zero reference: |mean| << std → relative bias is unstable.
            # Prefer a spread-normalised metric.
            near_zero = (
                dec_std > 0
                and abs(dec_mean) < 0.10 * max(dec_std, 1e-12)
            )
            if near_zero:
                grade_bias_used = (blk_mean - dec_mean) / dec_std
                ref_mode = "spread_normalised"
            elif np.isfinite(grade_bias_rel):
                grade_bias_used = grade_bias_rel
                ref_mode = "relative"
            else:
                ref_mode = "undefined"

        # Metal bias — same safeguard. Panel metals are
        # ~ decl_mean × panel_volume, so when the grade distribution
        # is zero-centred the panel totals average toward zero and the
        # relative metal ratio is ill-defined in exactly the same way
        # the grade-side relative bias is. When the grade path chose
        # the spread-normalised fallback, force metal to do the same
        # and derive the metric directly from the grade arrays.
        metal_bias_rel = float("nan")
        metal_bias_used = float("nan")
        metal_ref_mode = "unknown"
        if ref_mode == "spread_normalised":
            metal_bias_used = grade_bias_used
            metal_ref_mode = "spread_normalised"
            if mdec_tot > 0:
                metal_bias_rel = (mblk_tot - mdec_tot) / mdec_tot
        elif mdec_tot > 0:
            metal_bias_rel = (mblk_tot - mdec_tot) / mdec_tot
            metal_bias_used = metal_bias_rel
            metal_ref_mode = "relative"
        else:
            metal_bias_used = grade_bias_used
            metal_ref_mode = ref_mode

        def _status(gb: float, mb: float) -> str:
            if not np.isfinite(gb) or not np.isfinite(mb):
                return "unknown"
            if abs(gb) < 0.15 and abs(mb) < 0.20:
                return "PASS"
            if abs(gb) < 0.30 and abs(mb) < 0.40:
                return "WARN"
            return "FAIL"

        return {
            "gate3_panel_grade_bias": grade_bias_used,
            "gate3_panel_metal_bias": metal_bias_used,
            "gate3_panel_grade_bias_rel": grade_bias_rel,
            "gate3_panel_metal_bias_rel": metal_bias_rel,
            "gate3_reference_mode": ref_mode,
            "gate3_metal_reference_mode": metal_ref_mode,
            "gate3_valid_panels": int(n_valid),
            "gate3_total_panels": int(n_total),
            "gate3_status": _status(grade_bias_used, metal_bias_used),
        }


# Alias for drop-in import compatibility
ARBFEstimator = ARBFEstimatorAdapter
