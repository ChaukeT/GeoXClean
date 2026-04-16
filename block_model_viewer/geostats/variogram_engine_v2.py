from __future__ import annotations

"""
Improved variogram engine for ARBF, kriging, and simulation workflows.

This module addresses the main gaps often found in practical mining variogram
implementations:

1. Residual variography with explicit drift removal.
2. Transformation-aware variography: raw, normal-score, log, indicator.
3. Domain-wise and local-window variography.
4. Directional anisotropy in 3D.
5. Nested model fitting with audit metadata.
6. Support metadata tracking.
7. Indicator variograms for simulation workflows.
8. Robust experimental variogram option for skewed and contaminated data.

The module is a coherent reference implementation intended to be readable,
maintainable, and directly usable as a foundation in ARBF- or kriging-based
systems.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Iterable
import hashlib
import json
import math

import logging
import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    from scipy.special import erf, erfinv
except Exception as exc:  # pragma: no cover
    raise ImportError("This script requires scipy. Install with: pip install scipy") from exc


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_2d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    return arr


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def _std_norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def _std_norm_ppf(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return math.sqrt(2.0) * erfinv(2.0 * p - 1.0)


def _rotation_matrix_from_azimuth_dip_plunge(azimuth_deg: float, dip_deg: float, plunge_deg: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    dip = math.radians(dip_deg)
    pl = math.radians(plunge_deg)

    cz, sz = math.cos(az), math.sin(az)
    cy, sy = math.cos(dip), math.sin(dip)
    cx, sx = math.cos(pl), math.sin(pl)

    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    return rz @ ry @ rx


def stable_hash(obj: Dict) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# -----------------------------------------------------------------------------
# Metadata and configuration
# -----------------------------------------------------------------------------

@dataclass
class SupportMetadata:
    input_support: str = "composite"
    input_support_length: Optional[float] = None
    target_support: str = "point"
    target_support_dims: Optional[Tuple[float, float, float]] = None


@dataclass
class DriftConfig:
    mode: str = "none"  # none, constant, linear


@dataclass
class TransformConfig:
    mode: str = "raw"  # raw, nscore, log, indicator
    indicator_threshold: Optional[float] = None


@dataclass
class DirectionSpec:
    name: str
    unit_vector: np.ndarray
    tolerance_deg: float = 22.5
    bandwidth: Optional[float] = None


@dataclass
class VariogramSearchConfig:
    lag_size: float
    n_lags: int
    min_pairs: int = 30
    max_pairs_per_lag: int = 200000
    use_robust_estimator: bool = True
    random_seed: int = 123
    # Estimator type: "classical" (Matheron), "robust" (Cressie-Hawkins),
    # "madogram" (L1 estimator), "rodogram" (L0.5 estimator).
    # When set, this takes precedence over the legacy `use_robust_estimator`
    # boolean.  If left as "classical" the boolean still controls robust mode
    # for backward compatibility.
    estimator: str = "classical"


@dataclass
class Anisotropy3D:
    ranges: Tuple[float, float, float] = (100.0, 60.0, 20.0)
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))

    def transform(self, coords: np.ndarray) -> np.ndarray:
        coords = _as_2d_float(coords, "coords")
        rot = coords @ self.rotation_matrix.T
        scales = np.array([1.0 / max(r, 1e-12) for r in self.ranges], dtype=float)
        return rot * scales


@dataclass
class NestedStructure:
    model: str  # nugget, spherical, exponential, gaussian, cauchy
    sill: float
    ranges: Tuple[float, float, float]
    cauchy_beta: float = 1.5


@dataclass
class VariogramModel3D:
    nugget: float
    structures: List[NestedStructure]
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    metadata: Dict = field(default_factory=dict)

    @property
    def total_sill(self) -> float:
        return self.nugget + sum(s.sill for s in self.structures)

    def covariance(self, h_xyz: np.ndarray) -> np.ndarray:
        h_xyz = _as_2d_float(h_xyz, "h_xyz")
        out = np.full(h_xyz.shape[0], self.total_sill, dtype=float)
        for s in self.structures:
            out -= s.sill * _core_gamma_component(h_xyz, s.model, s.ranges, self.rotation_matrix, s.cauchy_beta)
        zero_mask = np.linalg.norm(h_xyz, axis=1) == 0.0
        out[zero_mask] = self.total_sill
        return out

    def semivariance(self, h_xyz: np.ndarray) -> np.ndarray:
        h_xyz = _as_2d_float(h_xyz, "h_xyz")
        out = np.full(h_xyz.shape[0], self.nugget, dtype=float)
        for s in self.structures:
            out += s.sill * _core_gamma_component(h_xyz, s.model, s.ranges, self.rotation_matrix, s.cauchy_beta)
        zero_mask = np.linalg.norm(h_xyz, axis=1) == 0.0
        out[zero_mask] = 0.0
        return out


@dataclass
class ExperimentalVariogram:
    lag_centres: np.ndarray
    gamma: np.ndarray
    n_pairs: np.ndarray
    direction_name: str
    transform_mode: str
    drift_mode: str
    domain_name: str
    support_metadata: SupportMetadata
    robust: bool
    local_window_center: Optional[np.ndarray] = None
    lineage_hash: str = ""
    # Unit vector of the direction used to compute this variogram.
    # Stored so that fit_nested_model can construct lag vectors along the
    # correct anisotropic axis instead of always defaulting to X.
    direction_unit_vector: Optional[np.ndarray] = None
    # Madogram values (L1 estimator): gamma_mad(h) = 0.5 * mean(|dz|).
    # Populated when estimator="madogram" or always alongside the primary
    # estimator for diagnostic comparison.
    madogram: Optional[np.ndarray] = None
    # Which estimator produced `gamma`: "classical", "robust", "madogram", "rodogram"
    estimator_type: str = "classical"
    # Sum of per-pair weights per lag (declustering ``w_i * w_j`` aggregated
    # over the lag bin). Populated alongside ``gamma`` when sample_weights
    # are attached to the engine. Used by ``fit_nested_model`` to weight
    # the curve_fit residuals — this replaces the unweighted ``n_pairs``-
    # based sigma that was inverse-variance-weighting by raw pair count,
    # effectively ignoring declustering at the fit step.
    pair_weight_sum: Optional[np.ndarray] = None


# -----------------------------------------------------------------------------
# Nested-structure inflection detector (ported from legacy variogram3d so
# v2 can do a real two-structure residual fit without re-importing legacy)
# -----------------------------------------------------------------------------

def _find_nested_inflection(
    d: np.ndarray, g: np.ndarray, max_d: float
) -> float:
    """Detect the inflection point of the experimental variogram for
    nested structure separation using a finite-difference second
    derivative.

    The inflection point marks where the curvature of ``γ(h)`` changes
    from concave-up (short-range structure rising) to concave-down
    (approaching the sill plateau). Data-driven — adapts to the actual
    variogram shape instead of using a fixed fraction of max distance.

    Falls back to ``max_d * 0.3`` when the inflection cannot be
    resolved (too few lags, flat variogram, etc.).
    """
    n = len(d)
    if n < 5:
        return max_d * 0.3

    # Smooth γ(h) with a small centred moving average before taking
    # finite differences so noisy lags don't dominate.
    window = min(5, max(3, n // 4))
    if window % 2 == 0:
        window += 1
    half = window // 2

    g_smooth = np.copy(g)
    for i in range(half, n - half):
        g_smooth[i] = np.mean(g[i - half : i + half + 1])

    d2g = np.zeros(n)
    for i in range(1, n - 1):
        dh_fwd = d[i + 1] - d[i]
        dh_bwd = d[i] - d[i - 1]
        if dh_fwd > 0 and dh_bwd > 0:
            d2g[i] = (
                (g_smooth[i + 1] - g_smooth[i]) / dh_fwd
                - (g_smooth[i] - g_smooth[i - 1]) / dh_bwd
            ) / ((dh_fwd + dh_bwd) * 0.5)

    # Search 10%-70% of max_d for the most-negative second derivative
    # (maximum concave-down curvature) to avoid edge and plateau noise.
    lo = int(n * 0.1)
    hi = int(n * 0.7)
    if hi <= lo:
        hi = min(lo + 3, n - 1)

    search_region = d2g[lo:hi]
    if len(search_region) == 0 or np.all(search_region == 0.0):
        return max_d * 0.3

    inflection_idx = lo + int(np.argmin(search_region))
    inflection_d = float(d[inflection_idx])

    # Hard sanity bounds
    inflection_d = max(inflection_d, max_d * 0.05)
    inflection_d = min(inflection_d, max_d * 0.65)
    return inflection_d


# -----------------------------------------------------------------------------
# Transform handling
# -----------------------------------------------------------------------------

class NormalScoreTransformer:
    def __init__(self) -> None:
        self.values_sorted = np.empty(0, dtype=float)
        self.probs_sorted = np.empty(0, dtype=float)

    def fit(self, values: ArrayLike) -> "NormalScoreTransformer":
        z = np.sort(_as_1d_float(values, "values"))
        n = z.size
        p = (np.arange(1, n + 1, dtype=float) - 0.5) / n
        self.values_sorted = z
        self.probs_sorted = p
        return self

    def transform(self, values: ArrayLike) -> np.ndarray:
        z = _as_1d_float(values, "values")
        p = np.interp(z, self.values_sorted, self.probs_sorted)
        return _std_norm_ppf(p)


# -----------------------------------------------------------------------------
# Drift handling
# -----------------------------------------------------------------------------

def drift_design_matrix(coords: np.ndarray, mode: str) -> np.ndarray:
    coords = _as_2d_float(coords, "coords")
    if mode == "none":
        return np.zeros((coords.shape[0], 0), dtype=float)
    if mode == "constant":
        return np.ones((coords.shape[0], 1), dtype=float)
    if mode == "linear":
        return np.hstack([np.ones((coords.shape[0], 1), dtype=float), coords])
    raise ValueError(f"Unsupported drift mode: {mode}")


def remove_drift(coords: np.ndarray, values: np.ndarray, config: DriftConfig) -> Tuple[np.ndarray, Dict]:
    X = drift_design_matrix(coords, config.mode)
    if X.shape[1] == 0:
        return values.copy(), {"mode": config.mode, "coefficients": []}
    beta, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
    fitted = X @ beta
    resid = values - fitted
    return resid, {"mode": config.mode, "coefficients": beta.tolist()}


# -----------------------------------------------------------------------------
# Variogram core models
# -----------------------------------------------------------------------------

def _anisotropic_radius(h_xyz: np.ndarray, ranges: Tuple[float, float, float], rotation_matrix: np.ndarray) -> np.ndarray:
    h_xyz = _as_2d_float(h_xyz, "h_xyz")
    rot = h_xyz @ rotation_matrix.T
    scale = np.array([1.0 / max(r, 1e-12) for r in ranges], dtype=float)
    u = rot * scale
    return np.linalg.norm(u, axis=1)


def _core_gamma_component(
    h_xyz: np.ndarray,
    model: str,
    ranges: Tuple[float, float, float],
    rotation_matrix: np.ndarray,
    cauchy_beta: float = 1.5,
) -> np.ndarray:
    r = _anisotropic_radius(h_xyz, ranges, rotation_matrix)

    if model == "nugget":
        out = np.ones_like(r)
        out[r == 0.0] = 0.0
        return out
    if model == "spherical":
        out = np.where(r < 1.0, 1.5 * r - 0.5 * r**3, 1.0)
        out[r == 0.0] = 0.0
        return out
    if model == "exponential":
        out = 1.0 - np.exp(-3.0 * r)
        out[r == 0.0] = 0.0
        return out
    if model == "gaussian":
        out = 1.0 - np.exp(-3.0 * r**2)
        out[r == 0.0] = 0.0
        return out
    if model == "cauchy":
        beta = max(cauchy_beta, 1e-8)
        out = 1.0 - (1.0 + r**2) ** (-beta)
        out[r == 0.0] = 0.0
        return out
    raise ValueError(f"Unsupported model: {model}")


# -----------------------------------------------------------------------------
# Direction handling
# -----------------------------------------------------------------------------

def default_directions() -> List[DirectionSpec]:
    return [
        DirectionSpec("omni", np.array([1.0, 0.0, 0.0]), tolerance_deg=180.0),
        DirectionSpec("major_x", np.array([1.0, 0.0, 0.0]), tolerance_deg=22.5),
        DirectionSpec("minor_y", np.array([0.0, 1.0, 0.0]), tolerance_deg=22.5),
        DirectionSpec("vertical_z", np.array([0.0, 0.0, 1.0]), tolerance_deg=22.5),
    ]


def pair_direction_mask(h_xyz: np.ndarray, direction: DirectionSpec) -> np.ndarray:
    h_xyz = _as_2d_float(h_xyz, "h_xyz")
    norms = np.linalg.norm(h_xyz, axis=1)
    mask = norms > 0.0
    if direction.tolerance_deg >= 179.999:
        return mask
    h_unit = np.zeros_like(h_xyz)
    h_unit[mask] = h_xyz[mask] / norms[mask][:, None]
    # VR-02 fix: guard against zero-length direction vector
    dir_norm = np.linalg.norm(direction.unit_vector)
    if dir_norm < 1e-15:
        raise ValueError(
            f"Direction '{direction.name}' has a zero-length unit_vector. "
            "Cannot compute directional variogram with a zero direction."
        )
    d = direction.unit_vector / dir_norm
    cosang = np.abs(np.sum(h_unit * d[None, :], axis=1))
    tol = math.cos(math.radians(direction.tolerance_deg))
    dir_mask = cosang >= tol
    if direction.bandwidth is not None and direction.tolerance_deg < 179.999:
        proj = np.sum(h_xyz * d[None, :], axis=1)
        perp = np.linalg.norm(h_xyz - proj[:, None] * d[None, :], axis=1)
        dir_mask &= perp <= direction.bandwidth
    return mask & dir_mask


# -----------------------------------------------------------------------------
# Experimental variogram computation
# -----------------------------------------------------------------------------

class VariogramEngine:
    def __init__(
        self,
        coords: ArrayLike,
        values: ArrayLike,
        domain_ids: Optional[Sequence[str]] = None,
        support_metadata: Optional[SupportMetadata] = None,
        sample_weights: Optional[ArrayLike] = None,
    ) -> None:
        self.coords = _as_2d_float(coords, "coords")
        self.values_raw = _as_1d_float(values, "values")
        if self.coords.shape[0] != self.values_raw.size:
            raise ValueError("coords and values size mismatch")
        if sample_weights is not None:
            _w = np.asarray(sample_weights, dtype=float).ravel()
            if _w.size != self.values_raw.size:
                raise ValueError(
                    "sample_weights size mismatch (%d vs %d)"
                    % (_w.size, self.values_raw.size)
                )
            # Guard against negative/NaN/zero-only weights.
            _bad = ~np.isfinite(_w) | (_w < 0)
            if _bad.any():
                _w = _w.copy()
                _w[_bad] = 0.0
            if _w.sum() <= 0:
                logger.warning(
                    "v2 variogram: sample_weights sum to zero — ignoring "
                    "weights and falling back to unweighted."
                )
                _w = None
            self.weights_raw = _w
        else:
            self.weights_raw = None
        if self.coords.shape[1] != 3:
            raise ValueError("coords must be Nx3")

        # VR-25 fix: validate input data for NaN/Inf
        bad_coords = ~np.isfinite(self.coords).all(axis=1)
        bad_values = ~np.isfinite(self.values_raw)
        bad_mask = bad_coords | bad_values
        n_bad = bad_mask.sum()
        if n_bad > 0:
            logger.warning(
                "VR-25: Removed %d / %d samples with NaN/Inf in coords or values",
                n_bad, self.coords.shape[0],
            )
            good = ~bad_mask
            self.coords = self.coords[good]
            self.values_raw = self.values_raw[good]
            # Keep weights in sync with filtered rows
            if sample_weights is not None and getattr(self, "weights_raw", None) is not None:
                self.weights_raw = self.weights_raw[good]
            if self.coords.shape[0] < 4:
                raise ValueError(
                    f"After removing {n_bad} NaN/Inf samples, only "
                    f"{self.coords.shape[0]} remain (minimum 4 required)."
                )

        if domain_ids is None:
            self.domain_ids = np.array(["default"] * self.coords.shape[0], dtype=object)
        else:
            self.domain_ids = np.asarray(domain_ids, dtype=object).reshape(-1)
            if self.domain_ids.size != self.coords.shape[0]:
                raise ValueError("domain_ids size mismatch")

        self.support_metadata = support_metadata or SupportMetadata()
        self._nscore_transformer = None

    # -------------------------
    # Transform pipeline
    # -------------------------
    def apply_transform(self, values: np.ndarray, config: TransformConfig) -> Tuple[np.ndarray, Dict]:
        if config.mode == "raw":
            return values.copy(), {"mode": "raw"}
        if config.mode == "normal_score":
            config = TransformConfig(mode="nscore", indicator_threshold=config.indicator_threshold)
        if config.mode == "log":
            shift = 0.0
            minv = float(np.min(values))
            if minv <= 0.0:
                shift = abs(minv) + 1e-6
            out = np.log(values + shift)
            return out, {"mode": "log", "shift": shift}
        if config.mode == "nscore":
            self._nscore_transformer = NormalScoreTransformer().fit(values)
            out = self._nscore_transformer.transform(values)
            return out, {"mode": "nscore"}
        if config.mode == "indicator":
            if config.indicator_threshold is None:
                raise ValueError("indicator_threshold is required for indicator transform")
            out = (values >= config.indicator_threshold).astype(float)
            return out, {"mode": "indicator", "threshold": config.indicator_threshold}
        raise ValueError(f"Unsupported transform mode: {config.mode}")

    # -------------------------
    # Domain and local windows
    # -------------------------
    def _subset_domain(self, domain_name: str) -> Tuple[np.ndarray, np.ndarray]:
        mask = self.domain_ids == domain_name
        return self.coords[mask], self.values_raw[mask]

    def _subset_domain_weights(self, domain_name: str) -> Optional[np.ndarray]:
        """Return weights aligned with `_subset_domain` rows, or None if
        no weights are attached to the engine."""
        if self.weights_raw is None:
            return None
        mask = self.domain_ids == domain_name
        return self.weights_raw[mask]

    def _subset_local_window(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        center: Optional[np.ndarray],
        radius: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if center is None or radius is None:
            return coords, values
        d = np.linalg.norm(coords - center[None, :], axis=1)
        mask = d <= radius
        return coords[mask], values[mask]

    # -------------------------
    # Pair generation
    # -------------------------
    def _pair_arrays(self, coords: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = coords.shape[0]
        iu, ju = np.triu_indices(n, k=1)
        h = coords[ju] - coords[iu]
        dz = values[ju] - values[iu]
        return h, dz

    # -------------------------
    # Experimental variogram
    # -------------------------
    def compute_experimental_variogram(
        self,
        domain_name: str = "default",
        transform: Optional[TransformConfig] = None,
        drift: Optional[DriftConfig] = None,
        direction: Optional[DirectionSpec] = None,
        search: Optional[VariogramSearchConfig] = None,
        local_window_center: Optional[ArrayLike] = None,
        local_window_radius: Optional[float] = None,
    ) -> ExperimentalVariogram:
        transform = transform or TransformConfig(mode="raw")
        drift = drift or DriftConfig(mode="none")
        direction = direction or default_directions()[0]
        search = search or VariogramSearchConfig(lag_size=10.0, n_lags=20)

        coords, values = self._subset_domain(domain_name)
        weights = self._subset_domain_weights(domain_name)
        if coords.shape[0] < 4:
            raise ValueError(f"Domain '{domain_name}' has too few samples")

        lwc = None if local_window_center is None else _as_1d_float(local_window_center, "local_window_center")
        _pre_n = coords.shape[0]
        coords, values = self._subset_local_window(coords, values, lwc, local_window_radius)
        if weights is not None and coords.shape[0] != _pre_n:
            # Re-apply local window mask to weights by recomputing it
            if local_window_center is not None and local_window_radius is not None:
                full_coords, _ = self._subset_domain(domain_name)
                d = np.linalg.norm(full_coords - _as_1d_float(local_window_center, "local_window_center")[None, :], axis=1)
                weights = weights[d <= local_window_radius]
        if coords.shape[0] < 4:
            raise ValueError("Local window has too few samples")

        transformed_values, transform_meta = self.apply_transform(values, transform)
        residual_values, drift_meta = remove_drift(coords, transformed_values, drift)

        # Pre-subsample POINTS before pair generation to avoid O(N^2) memory.
        # For N points, pair count = N*(N-1)/2.  Cap at ~5000 points (~12.5M pairs)
        # to keep memory under ~400 MB.  Use seeded RNG for reproducibility.
        rng = np.random.default_rng(search.random_seed)
        max_points_for_pairs = 5000
        if coords.shape[0] > max_points_for_pairs:
            n_original = coords.shape[0]
            idx = rng.choice(coords.shape[0], size=max_points_for_pairs, replace=False)
            idx.sort()  # preserve spatial ordering
            coords = coords[idx]
            residual_values = residual_values[idx]
            if weights is not None:
                weights = weights[idx]
            logger.info(
                "V2 engine: subsampled %d → %d points before pair generation",
                n_original, max_points_for_pairs,
            )

        # Generate pair indices + arrays; keep iu/ju so we can build
        # per-pair weights w_i * w_j for the weighted pair estimator.
        n_pts = coords.shape[0]
        iu, ju = np.triu_indices(n_pts, k=1)
        h_xyz = coords[ju] - coords[iu]
        dz = residual_values[ju] - residual_values[iu]
        if weights is not None:
            pair_w = weights[iu] * weights[ju]
        else:
            pair_w = None
        h_norm = np.linalg.norm(h_xyz, axis=1)

        dir_mask = pair_direction_mask(h_xyz, direction)
        h_xyz = h_xyz[dir_mask]
        dz = dz[dir_mask]
        h_norm = h_norm[dir_mask]
        if pair_w is not None:
            pair_w = pair_w[dir_mask]

        # Post-subsample pairs if still too many after directional filtering
        if h_norm.size > search.max_pairs_per_lag * search.n_lags:
            keep = rng.choice(h_norm.size, size=search.max_pairs_per_lag * search.n_lags, replace=False)
            h_xyz = h_xyz[keep]
            dz = dz[keep]
            h_norm = h_norm[keep]
            if pair_w is not None:
                pair_w = pair_w[keep]

        lag_centres = (np.arange(search.n_lags, dtype=float) + 0.5) * search.lag_size
        gamma = np.full(search.n_lags, np.nan, dtype=float)
        madogram_arr = np.full(search.n_lags, np.nan, dtype=float)
        n_pairs = np.zeros(search.n_lags, dtype=int)
        # Per-lag sum of pair weights (w_i * w_j). Tracks the effective
        # "weighted pair count" so downstream fitters can inverse-variance
        # weight by declustered support instead of raw pair count.
        pair_weight_sum_arr = (
            np.zeros(search.n_lags, dtype=float) if pair_w is not None else None
        )

        # Resolve estimator mode: the new `estimator` field takes precedence,
        # but if it is left at default ("classical") and the legacy boolean
        # `use_robust_estimator` is True, fall back to "robust" for backward
        # compatibility.
        _est = search.estimator.lower().strip()
        if _est == "classical" and search.use_robust_estimator:
            _est = "robust"

        def _wmean(arr: np.ndarray, w: Optional[np.ndarray]) -> float:
            """Weighted mean with graceful fallback to unweighted."""
            if w is None:
                return float(np.mean(arr))
            _wsum = float(np.sum(w))
            if _wsum <= 0:
                return float(np.mean(arr))
            return float(np.sum(arr * w) / _wsum)

        for i in range(search.n_lags):
            low = i * search.lag_size
            high = (i + 1) * search.lag_size
            mask = (h_norm > low) & (h_norm <= high)
            n_pairs[i] = int(np.sum(mask))
            if n_pairs[i] < search.min_pairs:
                continue
            dz_lag = dz[mask]
            diffs = np.abs(dz_lag)
            pair_w_lag = pair_w[mask] if pair_w is not None else None
            if pair_weight_sum_arr is not None and pair_w_lag is not None:
                pair_weight_sum_arr[i] = float(np.sum(pair_w_lag))

            # --- primary estimator ---
            if _est == "robust":
                # Cressie-Hawkins robust semivariogram estimator.
                m = _wmean(np.sqrt(diffs), pair_w_lag)
                denom = 0.457 + 0.494 / max(n_pairs[i], 1) + 0.045 / max(n_pairs[i] ** 2, 1)
                gamma[i] = 0.5 * (m ** 4) / max(denom, 1e-12)
            elif _est == "madogram":
                # Madogram (L1 estimator): gamma_mad(h) = 0.5 * mean(|dz|)
                # More robust to outliers than classical; useful for skewed
                # distributions.  Reference: Cressie (1993), Ch. 2.4.
                gamma[i] = 0.5 * _wmean(diffs, pair_w_lag)
            elif _est == "rodogram":
                # GS-13 fix: standard rodogram (L0.5 estimator):
                # gamma_rod(h) = 0.5 * mean(|dz|^0.5)
                gamma[i] = 0.5 * _wmean(np.sqrt(diffs), pair_w_lag)
            else:
                # Classical Matheron estimator: gamma(h) = 0.5 * mean(dz^2)
                # With declustering weights w_i, the weighted estimator is
                # gamma(h) = (Σ w_i w_j (z_i - z_j)²) / (2 Σ w_i w_j)
                gamma[i] = 0.5 * _wmean(dz_lag ** 2, pair_w_lag)

            # --- always compute madogram as a companion diagnostic ---
            madogram_arr[i] = 0.5 * _wmean(diffs, pair_w_lag)

        meta = {
            "domain": domain_name,
            "transform": transform_meta,
            "drift": drift_meta,
            "direction": {
                "name": direction.name,
                "unit_vector": direction.unit_vector.tolist(),
                "tolerance_deg": direction.tolerance_deg,
                "bandwidth": direction.bandwidth,
            },
            "search": asdict(search),
            "support": asdict(self.support_metadata),
            "local_window_center": None if lwc is None else lwc.tolist(),
            "local_window_radius": local_window_radius,
        }

        # Normalise the direction vector so fit_nested_model can use it directly.
        # VR-02 fix: guard against zero-length direction vector
        _dir_norm = np.linalg.norm(direction.unit_vector)
        if _dir_norm < 1e-15:
            raise ValueError(
                f"Direction '{direction.name}' has a zero-length unit_vector."
            )
        dir_vec = direction.unit_vector / _dir_norm

        return ExperimentalVariogram(
            lag_centres=lag_centres,
            gamma=gamma,
            n_pairs=n_pairs,
            direction_name=direction.name,
            transform_mode=transform.mode,
            drift_mode=drift.mode,
            domain_name=domain_name,
            support_metadata=self.support_metadata,
            robust=search.use_robust_estimator,
            local_window_center=lwc,
            lineage_hash=stable_hash(meta),
            direction_unit_vector=dir_vec,
            madogram=madogram_arr,
            estimator_type=_est,
            pair_weight_sum=pair_weight_sum_arr,
        )

    # -------------------------
    # Non-ergodic correction
    # -------------------------
    def apply_non_ergodic_correction(self, exp: ExperimentalVariogram) -> ExperimentalVariogram:
        """Apply non-ergodic correction for finite-domain variograms.

        In classical geostatistics the ergodic assumption equates ensemble
        averages with spatial averages.  For finite domains the sample
        variance systematically underestimates the true (population) variance,
        and the experimental variogram inherits this bias — especially at
        large lags where few pairs are available.

        The correction (Deutsch, 2002; Journel & Huijbregts, 1978) rescales
        each lag bin:

            gamma_corrected(h) = gamma(h) * N / (N - n_h)

        where *N* is the total number of data points and *n_h* the number
        of *unique points* involved in pairs at lag *h* (approximated here
        as ``min(2 * n_pairs_h, N)`` since each pair contributes at most
        two distinct points).

        Parameters
        ----------
        exp : ExperimentalVariogram
            Uncorrected experimental variogram produced by
            :meth:`compute_experimental_variogram`.

        Returns
        -------
        ExperimentalVariogram
            A **new** ``ExperimentalVariogram`` instance with corrected
            ``gamma`` values.  The ``lineage_hash`` is updated to reflect
            the correction, and ``madogram`` (if present) is also corrected.

        References
        ----------
        Deutsch, C.V. (2002). *Geostatistical Reservoir Modeling*,
            Oxford University Press, pp. 65-67.
        Journel, A.G. & Huijbregts, Ch.J. (1978). *Mining Geostatistics*,
            Academic Press.
        """
        import copy

        N = int(self.coords.shape[0])
        if N < 2:
            return exp

        gamma_corr = exp.gamma.copy()
        madogram_corr = exp.madogram.copy() if exp.madogram is not None else None

        for i in range(len(exp.lag_centres)):
            n_h = int(exp.n_pairs[i])
            if n_h < 1 or not np.isfinite(exp.gamma[i]):
                continue
            # GS-16 fix: better estimate of unique points in this lag bin.
            # Previous formula min(2*n_h, N) assumed no shared endpoints, which
            # over-estimates unique points for clustered data and amplifies the
            # correction factor at distant lags. Use geometric estimate:
            # n_unique ≈ min(ceil(1 + sqrt(1 + 8*n_h) / 2), N)
            # which accounts for endpoint sharing (n_h ≤ n_unique*(n_unique-1)/2).
            # Also cap correction factor at 3.0 to prevent distortion.
            import math
            n_unique_est = min(int(math.ceil(0.5 + math.sqrt(0.25 + 2.0 * n_h))), N)
            denom = max(N - n_unique_est, 1)
            correction_factor = min(N / denom, 3.0)
            gamma_corr[i] = exp.gamma[i] * correction_factor
            if madogram_corr is not None and np.isfinite(madogram_corr[i]):
                madogram_corr[i] = madogram_corr[i] * correction_factor

        # Guard against NaN / Inf that could arise from very small denominator
        gamma_corr = np.where(np.isfinite(gamma_corr), gamma_corr, np.nan)
        if madogram_corr is not None:
            madogram_corr = np.where(np.isfinite(madogram_corr), madogram_corr, np.nan)

        corrected = copy.copy(exp)
        corrected.gamma = gamma_corr
        corrected.madogram = madogram_corr
        corrected.lineage_hash = stable_hash({
            "parent_hash": exp.lineage_hash,
            "correction": "non_ergodic_deutsch_2002",
            "N": N,
        })
        return corrected

    # -------------------------
    # Local variography map
    # -------------------------
    def compute_local_variograms(
        self,
        centers: ArrayLike,
        radius: float,
        domain_name: str = "default",
        transform: Optional[TransformConfig] = None,
        drift: Optional[DriftConfig] = None,
        direction: Optional[DirectionSpec] = None,
        search: Optional[VariogramSearchConfig] = None,
    ) -> List[ExperimentalVariogram]:
        centers = _as_2d_float(centers, "centers")
        out: List[ExperimentalVariogram] = []
        for c in centers:
            try:
                vg = self.compute_experimental_variogram(
                    domain_name=domain_name,
                    transform=transform,
                    drift=drift,
                    direction=direction,
                    search=search,
                    local_window_center=c,
                    local_window_radius=radius,
                )
                out.append(vg)
            except ValueError:
                continue
        return out

    # -------------------------
    # Model fitting
    # -------------------------
    def fit_nested_model(
        self,
        exp: ExperimentalVariogram,
        model_types: Sequence[str] = ("spherical", "exponential"),
        n_structures: int = 1,
        initial_rotation: Optional[np.ndarray] = None,
        initial_ranges: Optional[Tuple[float, float, float]] = None,
        sample_variance: Optional[float] = None,
        sill_reference: Optional[float] = None,
        use_sill_norm: bool = False,
    ) -> VariogramModel3D:
        lag = exp.lag_centres
        gamma = exp.gamma
        pairs = exp.n_pairs.astype(float)
        mask = np.isfinite(gamma) & (pairs > 0)
        if np.sum(mask) < 3:
            raise ValueError("Too few valid lags for fitting")

        lag = lag[mask]
        gamma = gamma[mask]
        pairs = pairs[mask]

        # Low-priority wiring: when ``use_sill_norm`` is requested, scale
        # gamma by the plateau sill so curve_fit operates on unit-sill
        # data. We rescale the fitted sill back to the original space
        # after the fit so callers see the same units as without
        # normalisation. This mirrors the legacy ``fit_variogram_model``
        # behaviour and exists for backward compatibility.
        _sill_scale = 1.0
        if use_sill_norm:
            _sill_scale = float(np.nanmax(gamma))
            if _sill_scale > 1e-12:
                gamma = gamma / _sill_scale
            else:
                _sill_scale = 1.0

        rotation = np.eye(3, dtype=float) if initial_rotation is None else np.asarray(initial_rotation, dtype=float)
        if initial_ranges is None:
            # NOTE: These are initial guesses for the 1D→3D range mapping only.
            # The actual directional ranges are determined by fitting each direction
            # independently and assembling in _build_combined_3d_model() /
            # _assemble_nested_directional_ranges().  The 0.6/0.25 ratios here
            # affect the intermediate VariogramModel3D representation but are
            # overridden when directional fits are combined.
            initial_ranges = (max(lag[-1], 1.0), max(lag[-1] * 0.6, 1.0), max(lag[-1] * 0.25, 1.0))

        n_structures = max(1, int(n_structures))
        model_types = list(model_types)
        if len(model_types) < n_structures:
            model_types = model_types + [model_types[-1]] * (n_structures - len(model_types))

        def unpack(theta: np.ndarray) -> VariogramModel3D:
            nugget = max(theta[0], 0.0)
            structures: List[NestedStructure] = []
            off = 1
            for i in range(n_structures):
                sill = max(theta[off], 0.0)
                rmaj = max(theta[off + 1], 1e-6)
                rsemi = max(theta[off + 2], 1e-6)
                rmin = max(theta[off + 3], 1e-6)
                off += 4
                beta = 1.5
                if model_types[i] == "cauchy":
                    beta = max(theta[off], 0.2)
                    off += 1
                structures.append(NestedStructure(model=model_types[i], sill=sill, ranges=(rmaj, rsemi, rmin), cauchy_beta=beta))
            return VariogramModel3D(nugget=nugget, structures=structures, rotation_matrix=rotation)

        # ── Reduce to 1D fitting problem ──────────────────────────
        # The experimental variogram is 1D (lag → gamma).  Fitting a 3D
        # anisotropic model (nugget + sill + 3 ranges per structure) to 1D
        # data creates a degenerate surface — L-BFGS-B can't get gradient
        # for the two unused range axes, destabilising the entire fit.
        #
        # SOLUTION: Fit only (nugget, sill, range_along_direction) using
        # scipy.optimize.curve_fit (Levenberg-Marquardt), which is the
        # standard GSLIB approach.  Then map the 1D range back into the
        # 3D model using initial_ranges ratios.
        from scipy.optimize import curve_fit as _curve_fit

        sill_max = float(np.nanmax(gamma))
        range_upper = max(float(lag[-1]) * 1.1, 1.0)

        # Reference "reasonable sill" = sample variance (preferred) or an
        # omni/major reference passed in by the bridge. The total sill of a
        # fitted stationary variogram cannot legitimately exceed the sample
        # variance by much; bounding curve_fit prevents noisy directions
        # from inflating both sill and range.
        _sv = None
        if sample_variance is not None and np.isfinite(sample_variance) and sample_variance > 0:
            _sv = float(sample_variance)
        elif sill_reference is not None and np.isfinite(sill_reference) and sill_reference > 0:
            _sv = float(sill_reference)

        if _sv is not None:
            # Allow 20% headroom so legitimate anisotropic directions can
            # slightly exceed the sample variance without being clamped.
            _psill_cap = min(sill_max * 1.3, _sv * 1.2)
            _nug_cap = min(float(gamma[0]), _sv * 0.9)
        else:
            _psill_cap = sill_max * 1.3
            _nug_cap = float(gamma[0])

        # Keep bounds strictly positive and ordered.
        _psill_cap = max(_psill_cap, 1e-3 * 10)
        _nug_cap = max(_nug_cap, 1e-9)

        # Choose 1D model function
        def _spherical_1d(h, nug, psill, rng):
            r = np.minimum(h / max(rng, 1e-6), 1.0)
            return nug + psill * (1.5 * r - 0.5 * r**3)

        def _exponential_1d(h, nug, psill, rng):
            return nug + psill * (1.0 - np.exp(-3.0 * h / max(rng, 1e-6)))

        def _gaussian_1d(h, nug, psill, rng):
            return nug + psill * (1.0 - np.exp(-3.0 * (h / max(rng, 1e-6))**2))

        _model_funcs_1d = {
            "spherical": _spherical_1d,
            "exponential": _exponential_1d,
            "gaussian": _gaussian_1d,
        }

        # Pick best model_type for the primary structure
        primary_type = model_types[0] if model_types[0] in _model_funcs_1d else "spherical"
        model_func_1d = _model_funcs_1d[primary_type]

        # GSLIB weights: sqrt(N(h) / gamma(h)^2) → sigma for curve_fit
        # Med1 fix: when the experimental variogram carries a per-lag
        # ``pair_weight_sum`` (populated when declustering weights were
        # attached to the engine), use the weighted pair count instead
        # of the raw count. This makes curve_fit's inverse-variance
        # weighting honour declustering — otherwise the fit silently
        # ignores the declustering and matches the unweighted result.
        g_sq = gamma ** 2
        g_sq_floor = max(float(np.median(g_sq)) * 0.1, 1e-6)
        _effective_pairs = pairs
        _raw_pws = getattr(exp, "pair_weight_sum", None)
        if _raw_pws is not None and _raw_pws.size == exp.n_pairs.size:
            _pws = np.asarray(_raw_pws, dtype=float)[mask]
            if _pws.size == pairs.size and np.all(_pws >= 0) and _pws.sum() > 0:
                _effective_pairs = _pws
        w = _effective_pairs / np.maximum(g_sq, g_sq_floor)
        sigma = 1.0 / np.sqrt(np.maximum(w, 1e-12))

        # Estimate range: first lag where gamma reaches 90% of plateau
        _plateau = sill_max * 0.9
        _range_est = float(lag[-1]) * 0.5
        for _k in range(len(gamma)):
            if gamma[_k] >= _plateau:
                _range_est = max(float(lag[_k]), float(lag[0]) * 2)
                break

        # Bounds for curve_fit: (nugget, partial_sill, range)
        _psill0 = min(sill_max, _psill_cap * 0.99)
        _nug0 = min(float(gamma[0]) * 0.2, _nug_cap * 0.5)
        p0 = [_nug0, _psill0, _range_est]
        lower = [0.0, 1e-3, float(lag[0]) * 0.5]
        upper = [_nug_cap, _psill_cap, range_upper]

        _fit_converged = True
        try:
            popt, _ = _curve_fit(
                model_func_1d, lag, gamma,
                p0=p0, sigma=sigma, absolute_sigma=False,
                bounds=(lower, upper), maxfev=5000,
            )
            fit_nug, fit_psill, fit_range = float(popt[0]), float(popt[1]), float(popt[2])
        except Exception:
            # Fallback: simple moment-based estimates
            _fit_converged = False
            fit_nug = max(float(gamma[0]) * 0.5, 0.0)
            fit_psill = max(sill_max - fit_nug, 1e-3)
            fit_range = _range_est

        # Undo the sill normalisation so downstream code sees the
        # original-space nugget and partial sill. Ranges are scale-free.
        if _sill_scale != 1.0:
            gamma = gamma * _sill_scale
            fit_nug = fit_nug * _sill_scale
            fit_psill = fit_psill * _sill_scale

        # Hard post-fit clamp: total sill must not exceed ~1.2x sample
        # variance. curve_fit honours the per-parameter bounds but nug +
        # psill can still sum high; this enforces the stationary-variance
        # constraint as a final safety net.
        if _sv is not None:
            _total_cap = _sv * 1.2
            _total_fit = fit_nug + fit_psill
            if _total_fit > _total_cap and _total_fit > 0:
                _scale = _total_cap / _total_fit
                fit_nug = fit_nug * _scale
                fit_psill = fit_psill * _scale
                logger.warning(
                    "v2 variogram fit: total sill %.3f exceeded 1.2x sample "
                    "variance %.3f — rescaled to nugget=%.3f, psill=%.3f.",
                    _total_fit, _sv, fit_nug, fit_psill,
                )

        # Map 1D range into 3D anisotropic ranges using initial_ranges ratios
        r0 = initial_ranges
        ratio_semi = r0[1] / max(r0[0], 1e-6)
        ratio_min = r0[2] / max(r0[0], 1e-6)

        if n_structures == 1:
            # Single-structure path — unchanged.
            rmaj = fit_range
            rsemi = fit_range * ratio_semi
            rmin = fit_range * ratio_min
            structures = [
                NestedStructure(
                    model=primary_type,
                    sill=fit_psill,
                    ranges=(rmaj, rsemi, rmin),
                )
            ]
        else:
            # Multi-structure path — real residual fit ported from legacy
            # variogram3d.fit_nested_model (see consolidation plan B1).
            # Fits a short-range + long-range model using an inflection
            # detector, then runs constrained least squares so both
            # partial sills sum to no more than the post-clamp total.
            d_sorted_idx = np.argsort(lag)
            d_sorted = lag[d_sorted_idx]
            g_sorted = gamma[d_sorted_idx]

            nugget_est = float(fit_nug)
            max_partial = max(fit_psill, 1e-3)
            max_d = float(d_sorted[-1]) if d_sorted.size else range_upper

            # 1. Inflection-based split between short- and long-range
            short_range_threshold = _find_nested_inflection(
                d_sorted, g_sorted, max_d
            )
            short_mask = d_sorted <= short_range_threshold

            # 2. Short-range seed: fit the 1D kernel on early lags only.
            c1_init: float
            r1_init: float
            if int(np.sum(short_mask)) >= 3:
                try:
                    _d_short = d_sorted[short_mask]
                    _g_short = g_sorted[short_mask]
                    _upper_short = [
                        max(nugget_est * 1.05, 1e-6),
                        max_partial,
                        max(float(_d_short[-1]) * 1.1, 1.0),
                    ]
                    _lower_short = [0.0, 1e-3, float(_d_short[0]) * 0.5]
                    _p0_short = [
                        min(nugget_est, _upper_short[0] * 0.5),
                        min(max_partial * 0.3, _upper_short[1]),
                        float(np.median(_d_short)),
                    ]
                    _popt_short, _ = _curve_fit(
                        model_func_1d, _d_short, _g_short,
                        p0=_p0_short,
                        bounds=(_lower_short, _upper_short),
                        maxfev=3000,
                    )
                    c1_init = max(float(_popt_short[1]), 0.0)
                    r1_init = max(float(_popt_short[2]), float(_d_short[0]))
                except Exception:
                    c1_init = max_partial * 0.3
                    r1_init = max_d * 0.2
            else:
                c1_init = max_partial * 0.3
                r1_init = max_d * 0.2

            # Cap the short-range contribution so the long-range
            # structure always has room to exist (legacy used 90%).
            c1_init = min(c1_init, max_partial * 0.9)
            c2_init = max(max_partial - c1_init, 0.0)
            r2_init = max_d * 0.7

            # 3. Joint constrained refinement
            from scipy.optimize import minimize as _minimize

            def _nested_model(h, c1, r1, c2, r2):
                return (
                    nugget_est
                    + model_func_1d(h, 0.0, c1, r1)
                    + model_func_1d(h, 0.0, c2, r2)
                )

            def _objective(params):
                c1, r1, c2, r2 = params
                if c1 < 0 or c2 < 0 or r1 <= 0 or r2 <= 0:
                    return 1e10
                y_pred = _nested_model(d_sorted, c1, r1, c2, r2)
                w = 1.0 / (d_sorted + 1.0)  # emphasise early lags
                base = float(np.sum(w * (g_sorted - y_pred) ** 2))
                excess = max(0.0, c1 + c2 - max_partial)
                return base + 1e6 * excess ** 2

            x0 = [c1_init, r1_init, c2_init, r2_init]
            bounds = [
                (0.0, max_partial),      # c1
                (1.0, max_d),            # r1
                (0.0, max_partial),      # c2
                (1.0, max_d * 1.5),      # r2
            ]
            try:
                opt = _minimize(_objective, x0, bounds=bounds, method="L-BFGS-B")
                if opt.success:
                    c1, r1, c2, r2 = [float(v) for v in opt.x]
                else:
                    c1, r1, c2, r2 = x0
            except Exception as exc:
                logger.warning(
                    "Nested variogram refinement failed (%s); using initial estimates.",
                    exc,
                )
                c1, r1, c2, r2 = x0

            # Enforce total-sill cap post-optimisation.
            total_c = c1 + c2
            if total_c > max_partial and total_c > 1e-12:
                scale = max_partial / total_c
                c1 *= scale
                c2 *= scale

            # Enforce ordering: short-range first (r1 < r2).
            if r1 > r2:
                r1, r2 = r2, r1
                c1, c2 = c2, c1

            logger.info(
                "Nested fit (v2): nug=%.3f, C1=%.3f@%.1fm, C2=%.3f@%.1fm "
                "(cap=%.3f, inflection=%.1fm)",
                nugget_est, c1, r1, c2, r2, max_partial, short_range_threshold,
            )

            structures = [
                NestedStructure(
                    model=model_types[0] if 0 < len(model_types) else primary_type,
                    sill=float(c1),
                    ranges=(float(r1), float(r1) * ratio_semi, float(r1) * ratio_min),
                ),
                NestedStructure(
                    model=model_types[1] if 1 < len(model_types) else primary_type,
                    sill=float(c2),
                    ranges=(float(r2), float(r2) * ratio_semi, float(r2) * ratio_min),
                ),
            ]
            # Keep `fit_psill` consistent with the rescaled totals so any
            # downstream code that inspects the scalar `fit_psill` sees
            # the real post-clamp partial sill sum.
            fit_psill = float(c1 + c2)

        model = VariogramModel3D(nugget=fit_nug, structures=structures, rotation_matrix=rotation)

        # Compute goodness of fit
        if exp.direction_unit_vector is not None:
            dir_unit = exp.direction_unit_vector / max(np.linalg.norm(exp.direction_unit_vector), 1e-12)
        else:
            dir_unit = np.array([1.0, 0.0, 0.0])
        h_xyz = np.column_stack([lag * dir_unit[0], lag * dir_unit[1], lag * dir_unit[2]])
        pred = model.semivariance(h_xyz)
        obj_val = float(np.sum((gamma - pred) ** 2))

        model.metadata = {
            "fit_success": _fit_converged,
            "fit_message": "curve_fit converged" if _fit_converged else "curve_fit failed — using moment estimates",
            "objective": obj_val,
            "lineage_hash": exp.lineage_hash,
            "direction_name": exp.direction_name,
            "transform_mode": exp.transform_mode,
            "drift_mode": exp.drift_mode,
            "domain_name": exp.domain_name,
            "support_metadata": asdict(exp.support_metadata),
            "robust_estimator": exp.robust,
        }
        return model

    # -------------------------
    # Directional suite and anisotropy inference
    # -------------------------
    def fit_directional_suite(
        self,
        domain_name: str = "default",
        transform: Optional[TransformConfig] = None,
        drift: Optional[DriftConfig] = None,
        search: Optional[VariogramSearchConfig] = None,
        directions: Optional[List[DirectionSpec]] = None,
    ) -> Dict[str, ExperimentalVariogram]:
        directions = directions or default_directions()
        out: Dict[str, ExperimentalVariogram] = {}
        for d in directions:
            try:
                out[d.name] = self.compute_experimental_variogram(
                    domain_name=domain_name,
                    transform=transform,
                    drift=drift,
                    direction=d,
                    search=search,
                )
            except ValueError:
                continue
        return out

    def infer_rotation_from_pca(self, domain_name: str = "default") -> np.ndarray:
        """Return a rotation matrix whose rows are the principal axes of
        the horizontal support cloud, ordered by descending eigenvalue.

        L2 fix: when declustering weights are attached to the engine,
        compute a weighted covariance so the principal direction tracks
        the declustered spatial-support distribution instead of the
        clustering geometry. Unweighted PCA on preferentially sampled
        data picks up the drill-grid orientation; weighted PCA picks
        up the real geological continuity direction.
        """
        coords, _ = self._subset_domain(domain_name)
        weights = self._subset_domain_weights(domain_name)

        if weights is not None and weights.sum() > 0:
            _w = np.asarray(weights, dtype=float).ravel()
            _w = _w / _w.sum()
            mean = np.sum(coords * _w[:, None], axis=0)
            centered = coords - mean
            # Weighted covariance: E[(x - mu)(x - mu)^T] under the weight
            # distribution. Use the bias-corrected factor 1 / (1 - sum(w^2))
            # so a uniform weight vector reduces to the sample covariance.
            denom = 1.0 - float(np.sum(_w ** 2))
            if denom <= 1e-12:
                denom = 1.0
            cov = (centered.T * _w) @ centered / denom
        else:
            centered = coords - np.mean(coords, axis=0, keepdims=True)
            cov = np.cov(centered.T)

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        return eigvecs[:, order].T

    # -------------------------
    # Indicator variograms for simulation
    # -------------------------
    def compute_indicator_variogram_set(
        self,
        thresholds: Sequence[float],
        domain_name: str = "default",
        drift: Optional[DriftConfig] = None,
        direction: Optional[DirectionSpec] = None,
        search: Optional[VariogramSearchConfig] = None,
    ) -> Dict[float, ExperimentalVariogram]:
        out: Dict[float, ExperimentalVariogram] = {}
        for thr in thresholds:
            out[float(thr)] = self.compute_experimental_variogram(
                domain_name=domain_name,
                transform=TransformConfig(mode="indicator", indicator_threshold=float(thr)),
                drift=drift,
                direction=direction,
                search=search,
            )
        return out

    # -------------------------
    # Validation and completeness checks
    # -------------------------
    def validate_model_for_methods(
        self,
        model: VariogramModel3D,
        allow_for_arbf: bool = True,
        allow_for_kriging: bool = True,
        allow_for_simulation: bool = True,
    ) -> Dict[str, object]:
        issues: List[str] = []
        warnings_: List[str] = []

        if model.total_sill <= 0.0:
            issues.append("Total sill must be positive.")
        if len(model.structures) == 0:
            issues.append("At least one nested structure is required.")
        for s in model.structures:
            if any(r <= 0.0 for r in s.ranges):
                issues.append(f"Structure {s.model} has non-positive range.")
            if s.sill < 0.0:
                issues.append(f"Structure {s.model} has negative sill.")

        transform_mode = model.metadata.get("transform_mode", "raw")
        drift_mode = model.metadata.get("drift_mode", "none")
        support_meta = model.metadata.get("support_metadata", {})

        if allow_for_simulation and transform_mode == "raw":
            warnings_.append("Simulation on raw-space variograms may be unstable for strongly skewed variables.")
        if allow_for_simulation and transform_mode == "indicator":
            pass
        if allow_for_simulation and transform_mode not in {"indicator", "nscore", "raw", "log"}:
            warnings_.append("Unknown transform mode for simulation.")

        if allow_for_kriging and drift_mode == "none":
            warnings_.append("No drift removal or drift metadata recorded. Check whether trend is negligible.")

        if support_meta.get("input_support") != support_meta.get("target_support"):
            warnings_.append("Input support differs from target support. Verify support consistency before estimation or simulation.")

        if allow_for_arbf:
            for s in model.structures:
                if s.model not in {"spherical", "exponential", "gaussian", "cauchy"}:
                    warnings_.append(f"Structure {s.model} may require special handling in ARBF kernel mapping.")

        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "warnings": warnings_,
            "metadata": model.metadata,
        }


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

def make_demo_data(seed: int = 7, n: int = 350) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords = rng.uniform([0.0, 0.0, 0.0], [500.0, 300.0, 120.0], size=(n, 3))
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    trend = 0.004 * x - 0.003 * z + 0.0015 * y
    lenses = (
        1.8 * np.exp(-(((x - 120.0) / 90.0) ** 2 + ((y - 60.0) / 40.0) ** 2 + ((z - 50.0) / 20.0) ** 2))
        + 1.2 * np.exp(-(((x - 360.0) / 80.0) ** 2 + ((y - 220.0) / 70.0) ** 2 + ((z - 30.0) / 18.0) ** 2))
    )
    noise = rng.normal(0.0, 0.20, size=n)
    values = np.exp(trend + lenses + noise)

    domains = np.where(x < 250.0, "west", "east")
    return coords, values, domains


def demo() -> None:
    coords, values, domains = make_demo_data()
    support = SupportMetadata(
        input_support="composite",
        input_support_length=2.0,
        target_support="block",
        target_support_dims=(10.0, 10.0, 5.0),
    )
    engine = VariogramEngine(coords, values, domain_ids=domains, support_metadata=support)

    search = VariogramSearchConfig(lag_size=15.0, n_lags=18, min_pairs=25, use_robust_estimator=True)
    drift = DriftConfig(mode="linear")
    transform = TransformConfig(mode="nscore")
    directions = default_directions()

    suite = engine.fit_directional_suite(
        domain_name="west",
        transform=transform,
        drift=drift,
        search=search,
        directions=directions,
    )

    major = suite.get("major_x") or next(iter(suite.values()))
    rotation = engine.infer_rotation_from_pca(domain_name="west")
    model = engine.fit_nested_model(
        major,
        model_types=("spherical", "exponential"),
        n_structures=2,
        initial_rotation=rotation,
    )

    validation = engine.validate_model_for_methods(model)
    print("Directional variograms:", list(suite.keys()))
    print("Fitted total sill:", model.total_sill)
    print("Validation ok:", validation["ok"])
    print("Warnings:", validation["warnings"])

    thresholds = np.quantile(values[domains == "west"], [0.5, 0.75, 0.9]).tolist()
    ind_set = engine.compute_indicator_variogram_set(
        thresholds=thresholds,
        domain_name="west",
        drift=DriftConfig(mode="constant"),
        direction=DirectionSpec("major_x", np.array([1.0, 0.0, 0.0]), 22.5),
        search=search,
    )
    print("Indicator thresholds:", list(ind_set.keys()))

    centres = np.array([[120.0, 70.0, 40.0], [200.0, 100.0, 50.0], [320.0, 220.0, 30.0]], dtype=float)
    local_vgs = engine.compute_local_variograms(
        centers=centres,
        radius=90.0,
        domain_name="west",
        transform=transform,
        drift=drift,
        direction=DirectionSpec("major_x", np.array([1.0, 0.0, 0.0]), 22.5),
        search=search,
    )
    print("Local variograms computed:", len(local_vgs))


if __name__ == "__main__":
    demo()
