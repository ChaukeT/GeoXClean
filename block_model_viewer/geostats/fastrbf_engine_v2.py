from __future__ import annotations

"""
Adaptive local radial basis function (ARBF) estimator for spatial datasets.

This module implements a true RBF-based local estimator for mining block
estimation.  The interpolation matrix Φ is built from radial basis functions
(Gaussian, multiquadric, inverse multiquadric, Wendland C2/C4, cubic, or
thin-plate spline) — NOT from variogram-derived covariance.

Key design principles:
- RBF basis functions drive the interpolation matrix, not variogram covariance
- Anisotropy is applied as distance scaling (rotation + range normalisation),
  not through a covariance ellipsoid
- Polynomial drift augmentation (none / constant / linear) follows standard
  RBF form: [Φ + λI, P; P', 0] @ [w; β] = [z; 0]
- Uncertainty is an empirical index (neighbourhood geometry, conditioning,
  data spacing), not kriging variance
- Variogram parameters may be used as *guidance* for inferring shape
  parameter / support radius, but do not define the system matrix

The VariogramModel class is retained for backward compatibility with the
variogram fitting pipeline.  When passed to FastRBFEstimator it is auto-
converted to an RBFKernel via from_variogram_guidance().

Features:
- anisotropy-aware neighbourhood search (KD-tree, octant-balanced)
- optional normal-score transform with Gauss-Hermite back-transform
- block-support estimation via subpoint discretisation
- partition-of-unity blending for large datasets
- empirical uncertainty index for classification guidance
- cross-validation and calibration diagnostics
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import logging
import math
import warnings

import numpy as np

logger = logging.getLogger(__name__)
from numpy.typing import ArrayLike

try:
    from scipy.linalg import cho_factor, cho_solve, solve, LinAlgError
    from scipy.spatial import cKDTree
    from scipy.special import erf, erfinv
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This script requires scipy. Install with: pip install scipy"
    ) from exc


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _as_2d_float(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array.")
    return arr


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _safe_log10(x: float) -> float:
    return math.log10(max(x, 1.0))


def _std_norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def _std_norm_ppf(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return math.sqrt(2.0) * erfinv(2.0 * p - 1.0)


def merge_near_duplicates(
    coords: np.ndarray,
    values: np.ndarray,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge exact or near-duplicate sample locations.

    Returns:
        merged_coords, merged_values, counts
    """
    coords = _as_2d_float(coords)
    values = _as_1d_float(values, "values")
    if coords.shape[0] != values.size:
        raise ValueError("coords and values size mismatch")

    rounded = np.round(coords / max(tol, 1e-12)).astype(np.int64)
    uniq, inv = np.unique(rounded, axis=0, return_inverse=True)

    merged_coords = np.zeros((uniq.shape[0], coords.shape[1]), dtype=float)
    merged_values = np.zeros(uniq.shape[0], dtype=float)
    counts = np.zeros(uniq.shape[0], dtype=int)

    for i in range(values.size):
        j = inv[i]
        merged_coords[j] += coords[i]
        merged_values[j] += values[i]
        counts[j] += 1

    merged_coords /= counts[:, None]
    merged_values /= counts
    return merged_coords, merged_values, counts


# -----------------------------------------------------------------------------
# Normal-score transform
# -----------------------------------------------------------------------------

# Module-level cache for Gauss-Hermite quadrature nodes/weights.
# hermgauss(n) is expensive (~0.4ms) and produces identical results
# for the same order — caching avoids recomputing it per block.
_GH_CACHE: Dict[int, tuple] = {}


@dataclass
class NormalScoreTransformer:
    values_sorted: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    probs_sorted: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    lower_tail_slope: Optional[float] = None
    upper_tail_slope: Optional[float] = None
    data_min: float = float("-inf")

    def fit(self, values: ArrayLike) -> "NormalScoreTransformer":
        z = _as_1d_float(values, "values")
        order = np.argsort(z)
        z_sorted = z[order]
        n = z_sorted.size
        probs = (np.arange(1, n + 1, dtype=float) - 0.5) / n
        self.values_sorted = z_sorted
        self.probs_sorted = probs
        self.data_min = float(z_sorted[0])

        if n >= 2:
            self.lower_tail_slope = (z_sorted[1] - z_sorted[0]) / max(probs[1] - probs[0], 1e-12)
            self.upper_tail_slope = (z_sorted[-1] - z_sorted[-2]) / max(probs[-1] - probs[-2], 1e-12)
        else:
            self.lower_tail_slope = 0.0
            self.upper_tail_slope = 0.0
        return self

    def transform(self, values: ArrayLike) -> np.ndarray:
        z = _as_1d_float(values, "values")
        p = np.interp(
            z,
            self.values_sorted,
            self.probs_sorted,
            left=self.probs_sorted[0],
            right=self.probs_sorted[-1],
        )
        return _std_norm_ppf(p)

    @staticmethod
    def _gh_nodes_weights(order: int):
        """Cached Gauss-Hermite quadrature nodes and weights.

        hermgauss(n) computes eigenvalues of an n×n companion matrix —
        calling it per block wastes ~37% of total estimation time.
        Module-level cache avoids the dataclass mutable-default restriction.
        """
        if order not in _GH_CACHE:
            _GH_CACHE[order] = np.polynomial.hermite.hermgauss(order)
        return _GH_CACHE[order]

    def inverse_backtransform_batch(self, mus: np.ndarray, vars: np.ndarray,
                                    gh_order: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised back-transform for an entire array of (mu, sigma²) pairs.

        Returns (means_original, vars_original) in original data space.
        This is ~100x faster than calling inverse_expectation() per block.
        """
        nodes, weights = self._gh_nodes_weights(gh_order)
        n = len(mus)
        sigmas = np.sqrt(np.maximum(vars, 0.0))

        # (n, gh_order) — all quadrature arguments at once
        # args[i, j] = sqrt(2) * sigma[i] * nodes[j] + mu[i]
        args = np.sqrt(2.0) * sigmas[:, None] * nodes[None, :] + mus[:, None]

        # Convert to probabilities (vectorised)
        probs = _std_norm_cdf(args)  # (n, gh_order)

        # Inverse quantile for ALL values at once (flatten, interp, reshape)
        probs_flat = probs.ravel()
        vals_flat = self.inverse_quantile(probs_flat)
        vals = vals_flat.reshape(n, -1)  # (n, gh_order)

        # Weighted sums
        inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
        means_out = np.sum(weights[None, :] * vals, axis=1) * inv_sqrt_pi
        vars_out = np.sum(weights[None, :] * vals ** 2, axis=1) * inv_sqrt_pi - means_out ** 2
        vars_out = np.maximum(vars_out, 0.0)

        return means_out, vars_out

    def inverse_expectation(self, mu: float, sigma2: float, gh_order: int = 20) -> float:
        sigma = math.sqrt(max(sigma2, 0.0))
        nodes, weights = self._gh_nodes_weights(gh_order)
        args = np.sqrt(2.0) * sigma * nodes + mu
        probs = _std_norm_cdf(args)
        vals = self.inverse_quantile(probs)
        return float(np.sum(weights * vals) / math.sqrt(math.pi))

    def inverse_second_moment(self, mu: float, sigma2: float, gh_order: int = 20) -> float:
        sigma = math.sqrt(max(sigma2, 0.0))
        nodes, weights = self._gh_nodes_weights(gh_order)
        args = np.sqrt(2.0) * sigma * nodes + mu
        probs = _std_norm_cdf(args)
        vals = self.inverse_quantile(probs)
        return float(np.sum(weights * vals**2) / math.sqrt(math.pi))

    def inverse_quantile(self, probs: ArrayLike) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)

        out = np.interp(
            p,
            self.probs_sorted,
            self.values_sorted,
            left=np.nan,
            right=np.nan,
        )

        # Lower tail linear extension
        lower_mask = p < self.probs_sorted[0]
        if np.any(lower_mask):
            out[lower_mask] = self.values_sorted[0] + self.lower_tail_slope * (p[lower_mask] - self.probs_sorted[0])

        # Upper tail linear extension
        upper_mask = p > self.probs_sorted[-1]
        if np.any(upper_mask):
            out[upper_mask] = self.values_sorted[-1] + self.upper_tail_slope * (p[upper_mask] - self.probs_sorted[-1])

        # Enforce data minimum bound: if all input data was non-negative,
        # do not allow back-transform to produce negative values.
        # This prevents the lower-tail linear extension from extrapolating
        # below the physical minimum (e.g., 0 for grade variables).
        if self.data_min >= 0.0:
            n_clipped = int(np.sum(out < 0.0))
            if n_clipped > 0:
                logger.debug(
                    "NormalScoreTransformer: clipping %d/%d back-transformed "
                    "values to data_min=%.4f (lower tail extrapolation)",
                    n_clipped, out.size, self.data_min,
                )
            np.clip(out, self.data_min, None, out=out)

        return out


# -----------------------------------------------------------------------------
# Spatial model configuration
# -----------------------------------------------------------------------------

@dataclass
class Anisotropy:
    ranges: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))

    def __post_init__(self) -> None:
        self.rotation_matrix = np.asarray(self.rotation_matrix, dtype=float)
        if self.rotation_matrix.shape != (3, 3):
            raise ValueError("rotation_matrix must be 3x3")
        if len(self.ranges) != 3:
            raise ValueError("ranges must have length 3")
        if any(r <= 0 for r in self.ranges):
            raise ValueError("anisotropy ranges must be positive")

    def transform(self, coords: np.ndarray) -> np.ndarray:
        coords = _as_2d_float(coords)
        if coords.shape[1] != 3:
            raise ValueError("coords must be Nx3")
        rotated = coords @ self.rotation_matrix.T
        scales = np.array([1.0 / r for r in self.ranges], dtype=float)
        return rotated * scales


def _model_core(model: str, hr: np.ndarray, cauchy_beta: float = 1.5) -> np.ndarray:
    """Evaluate normalised variogram model core: 0 at h=0, 1 at h→∞."""
    if model in ("spheroidal", "spherical"):
        x = hr  # h is already range-normalised
        return np.where(x < 1.0, 1.5 * x - 0.5 * x ** 3, 1.0)
    elif model == "exponential":
        return 1.0 - np.exp(-3.0 * hr)
    elif model == "gaussian":
        return 1.0 - np.exp(-3.0 * hr ** 2)
    elif model == "cauchy":
        beta = max(cauchy_beta, 1e-6)
        return 1.0 - (1.0 + hr ** 2) ** (-beta)
    else:
        raise ValueError(f"Unsupported variogram model: {model}")


def _covariance_core_inplace(
    model: str, hr: np.ndarray, contribution: float,
    out: np.ndarray, cauchy_beta: float = 1.5,
) -> None:
    """Add C_i · (1 - core_i(h)) to *out* in-place.  Minimises temporaries."""
    if model in ("spheroidal", "spherical"):
        # C_i(h) = C_i * (1 - 1.5x + 0.5x³) for x<1, else 0
        out += np.where(
            hr < 1.0,
            contribution * (1.0 - 1.5 * hr + 0.5 * hr ** 3),
            0.0,
        )
    elif model == "exponential":
        out += contribution * np.exp(-3.0 * hr)
    elif model == "gaussian":
        out += contribution * np.exp(-3.0 * hr ** 2)
    elif model == "cauchy":
        beta = max(cauchy_beta, 1e-6)
        out += contribution * (1.0 + hr ** 2) ** (-beta)


@dataclass
class VariogramStructure:
    """One structure in a nested variogram: model + contribution + ranges."""
    model_type: str = "spheroidal"
    contribution: float = 1.0       # partial sill (Cᵢ) for this structure
    ranges: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    cauchy_beta: float = 1.5


@dataclass
class VariogramModel:
    """Variogram model with optional nested structures.

    Single-structure mode (backward compat):
        γ(h) = C₀ + C₁ · model(h)
    Nested mode (when ``structures`` is non-empty):
        γ(h) = C₀ + Σᵢ Cᵢ · modelᵢ(h × primary_range / structᵢ_range)
    where h is in primary-range-normalised space (from Anisotropy.transform).
    """
    model: str = "spheroidal"
    nugget_micro: float = 0.0
    partial_sill: float = 1.0
    ranges: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    cauchy_beta: float = 1.5
    structures: Optional[List[VariogramStructure]] = None

    # ── Cached per-structure scale factors (set by _cache_structure_scales) ──
    _struct_scales: Optional[List[float]] = field(default=None, repr=False)

    @property
    def sill_total(self) -> float:
        if self.structures:
            return self.nugget_micro + sum(s.contribution for s in self.structures)
        return self.nugget_micro + self.partial_sill

    @staticmethod
    def _effective_range(ranges: Tuple[float, float, float]) -> float:
        """Geometric mean of 3D ranges — used for scalar range scaling."""
        return float((ranges[0] * ranges[1] * ranges[2]) ** (1.0 / 3.0))

    def _cache_structure_scales(self) -> List[float]:
        """Pre-compute per-structure distance scale factors."""
        if self._struct_scales is not None:
            return self._struct_scales
        if not self.structures:
            self._struct_scales = []
            return self._struct_scales
        primary_eff = self._effective_range(self.ranges)
        self._struct_scales = [
            primary_eff / max(self._effective_range(s.ranges), 1e-12)
            for s in self.structures
        ]
        return self._struct_scales

    def gamma_from_h(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        hr = np.clip(h, 0.0, None)

        if self.structures:
            scales = self._cache_structure_scales()
            out = np.zeros_like(hr)
            for struct, scale in zip(self.structures, scales):
                hr_s = hr * scale if abs(scale - 1.0) > 1e-9 else hr
                core = _model_core(struct.model_type, hr_s, struct.cauchy_beta)
                out += struct.contribution * core
            out += self.nugget_micro
            out = np.where(hr == 0.0, 0.0, out)
            return out

        # ── Single structure (original code) ──────────────────────────
        core = _model_core(self.model, hr, self.cauchy_beta)
        out = self.nugget_micro + self.partial_sill * core
        out = np.where(hr == 0.0, 0.0, out)
        return out

    def covariance_from_h(self, h: np.ndarray) -> np.ndarray:
        """Compute covariance C(h) = sill - γ(h).

        For the nested case, computes covariance DIRECTLY (avoiding
        sill - gamma indirection) to reduce temporary array allocations.
        """
        h = np.asarray(h, dtype=float)

        if self.structures:
            # ── Fast direct covariance for nested variograms ──────────
            # C(h) = nugget*(h==0) + Σ_i C_i*(1 - core_i(h*scale_i))
            # We accumulate covariance directly instead of computing
            # gamma first and subtracting from sill.
            scales = self._cache_structure_scales()
            hr = np.clip(h, 0.0, None)
            out = np.zeros_like(hr)
            for struct, scale in zip(self.structures, scales):
                hr_s = hr * scale if abs(scale - 1.0) > 1e-9 else hr
                _covariance_core_inplace(
                    struct.model_type, hr_s, struct.contribution,
                    out, struct.cauchy_beta,
                )
            # Add nugget contribution at h=0
            out[hr == 0.0] += self.nugget_micro
            return out

        return self.sill_total - self.gamma_from_h(h)


@dataclass
class RBFKernel:
    """True radial basis function kernel for RBF interpolation.

    The kernel evaluates φ(r) at anisotropy-transformed distances r.
    This is NOT a variogram covariance — it is the RBF basis function
    that builds the interpolation matrix Φ.

    Parameters
    ----------
    basis : str
        Basis function type.  Positive-definite kernels (gaussian,
        inverse_multiquadric, wendland_c2, wendland_c4) do not strictly
        require polynomial augmentation.  Conditionally positive-definite
        kernels (multiquadric, cubic, thin_plate_spline) require at least
        a constant or linear polynomial term for unique solvability.
    shape_parameter : float
        ε for Gaussian / MQ / IMQ kernels.  Controls smoothness vs
        localisation.  In anisotropy-transformed space where 1 unit = 1
        variogram range, ε = 1.0 is a sensible default.
    support_radius : float
        R for compactly-supported kernels (Wendland C2/C4).  The kernel
        is identically zero for r ≥ R.  In transformed space, R = 1.0
        means support = 1 variogram range.
    nugget : float
        Diagonal regularisation added to Φ (measurement noise / smoothing).
        Equivalent to Tikhonov regularisation λ in [Φ + λI].
    """
    basis: str = "wendland_c2"
    shape_parameter: float = 1.0
    support_radius: float = 1.0
    nugget: float = 0.0

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate φ(r) at distances r (in anisotropy-transformed space)."""
        r = np.asarray(r, dtype=float)
        r = np.maximum(r, 0.0)

        if self.basis == "gaussian":
            return np.exp(-(self.shape_parameter * r) ** 2)

        elif self.basis == "multiquadric":
            return np.sqrt(1.0 + (self.shape_parameter * r) ** 2)

        elif self.basis == "inverse_multiquadric":
            return 1.0 / np.sqrt(1.0 + (self.shape_parameter * r) ** 2)

        elif self.basis == "wendland_c2":
            # Wendland C2 in R³: φ(r) = (1 - r/R)⁴₊ · (4r/R + 1)
            rn = r / max(self.support_radius, 1e-12)
            return np.where(rn < 1.0,
                            (1.0 - rn) ** 4 * (4.0 * rn + 1.0),
                            0.0)

        elif self.basis == "wendland_c4":
            # Wendland C4 in R³: φ(r) = (1 - r/R)⁶₊ · (35r²+18r+3)/3
            rn = r / max(self.support_radius, 1e-12)
            return np.where(rn < 1.0,
                            (1.0 - rn) ** 6 * (35.0 * rn ** 2 + 18.0 * rn + 3.0) / 3.0,
                            0.0)

        elif self.basis == "cubic":
            return r ** 3

        elif self.basis == "thin_plate_spline":
            # TPS in 3D: φ(r) = r
            return r

        else:
            raise ValueError(f"Unknown RBF basis: {self.basis}")

    @property
    def is_positive_definite(self) -> bool:
        """Whether this kernel is strictly positive definite (not just cpd)."""
        return self.basis in ("gaussian", "inverse_multiquadric",
                              "wendland_c2", "wendland_c4")

    @property
    def needs_polynomial(self) -> bool:
        """Whether this basis requires polynomial augmentation."""
        return self.basis in ("multiquadric", "cubic", "thin_plate_spline")

    @property
    def min_polynomial_degree(self) -> int:
        """Minimum polynomial degree for unique solvability.

        Returns -1 if no polynomial is required (positive definite kernels).
        0 = constant, 1 = linear.
        """
        if self.basis in ("cubic",):
            return 1
        if self.basis in ("multiquadric", "thin_plate_spline"):
            return 0
        return -1

    @property
    def effective_support(self) -> Optional[float]:
        """Hard distance cutoff beyond which φ(r) ≈ 0 (transformed space).

        For compactly-supported kernels (Wendland) this is exact.
        For Gaussian, we use the 1% threshold: φ(r) = exp(-(ε·r)²) = 0.01
        → r = sqrt(ln 100) / ε ≈ 2.146 / ε.
        For infinite-range kernels (TPS, cubic, MQ) returns None.
        """
        if self.basis in ("wendland_c2", "wendland_c4"):
            return self.support_radius
        if self.basis == "gaussian":
            # Distance where kernel decays to 1%
            return 2.146 / max(self.shape_parameter, 1e-12)
        if self.basis == "inverse_multiquadric":
            # Distance where kernel decays to ~10% → 1/sqrt(1+(ε·r)²) = 0.1
            return 9.95 / max(self.shape_parameter, 1e-12)
        # TPS, cubic, multiquadric — truly global, no natural cutoff
        return None

    @property
    def phi_at_zero(self) -> float:
        """φ(0) — the diagonal value of the RBF matrix."""
        if self.basis == "gaussian":
            return 1.0
        elif self.basis == "multiquadric":
            return 1.0
        elif self.basis == "inverse_multiquadric":
            return 1.0
        elif self.basis in ("wendland_c2", "wendland_c4"):
            return 1.0
        elif self.basis in ("cubic", "thin_plate_spline"):
            return 0.0
        return 1.0

    @staticmethod
    def from_variogram_guidance(
        variogram: 'VariogramModel',
        basis: str = "wendland_c2",
    ) -> 'RBFKernel':
        """Create an RBFKernel using variogram parameters as guidance.

        Uses the variogram range to infer shape parameter / support radius.
        The variogram is guidance only — it does NOT define the system matrix.
        The nugget from the variogram is used as Tikhonov regularisation.

        All distances are in ANISOTROPY-TRANSFORMED SPACE (1 unit = 1 range).
        """
        nugget = variogram.nugget_micro

        if basis in ("gaussian", "inverse_multiquadric"):
            # ε controls how fast the kernel decays.
            # In transformed space, 1 unit = 1 range.
            # ε = 3.0 → Gaussian decays to exp(-9) ≈ 0.01% at 1 range
            # (effectively compact, matching variogram practical range).
            shape = 3.0
            return RBFKernel(basis=basis, shape_parameter=shape,
                             support_radius=1.0, nugget=nugget)
        elif basis == "multiquadric":
            return RBFKernel(basis=basis, shape_parameter=3.0,
                             support_radius=1.0, nugget=nugget)
        elif basis in ("wendland_c2", "wendland_c4"):
            # In transformed space 1 unit = 1 range.
            # support_radius = 1.0 means the kernel is exactly zero
            # beyond the variogram range — tight, geologically local.
            return RBFKernel(basis=basis, shape_parameter=1.0,
                             support_radius=1.0, nugget=nugget)
        else:
            return RBFKernel(basis=basis, shape_parameter=1.0,
                             support_radius=1.0, nugget=nugget)


# ---------------------------------------------------------------------------
# Uncertainty index — replaces kriging variance
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyScore:
    """Empirical uncertainty index for RBF estimation.

    This is NOT a kriging variance or posterior variance.
    It is a composite score from neighbourhood geometry, system conditioning,
    and data spacing.  The index ranges from 0 (well-informed) to 1
    (poorly informed / unreliable).
    """
    index: float              # composite 0–1
    neighbourhood_score: float  # 0=poor, 1=good
    spacing_score: float        # 0=far from data, 1=close
    condition_score: float      # 0=ill-conditioned, 1=well-conditioned
    octant_score: float         # fraction of octants with data


def compute_uncertainty_index(
    n_used: int,
    n_min: int,
    n_max: int,
    octant_counts: Optional[np.ndarray],
    nearest_distance: float,
    support_radius: float,
    condnum: float,
    condition_warn: float = 1e8,
    condition_fail: float = 1e12,
) -> UncertaintyScore:
    """Compute a composite uncertainty index from neighbourhood properties.

    Parameters
    ----------
    n_used : int
        Number of samples in the local neighbourhood.
    n_min, n_max : int
        Minimum / maximum neighbourhood size.
    octant_counts : (8,) array or None
        Number of samples in each octant.
    nearest_distance : float
        Distance to closest sample (in transformed space).
    support_radius : float
        Kernel support radius (in transformed space).
    condnum : float
        Condition number of the local system.
    condition_warn, condition_fail : float
        Condition number thresholds.

    Returns
    -------
    UncertaintyScore with composite index in [0, 1].
    """
    # 1. Neighbourhood count score (0–1, 1 = good)
    if n_max > n_min:
        nh_score = min(max((n_used - n_min) / (n_max - n_min), 0.0), 1.0)
    elif n_used >= n_min:
        nh_score = 1.0
    else:
        nh_score = max(n_used / max(n_min, 1), 0.0)

    # 2. Octant coverage score (0–1, 1 = all octants have data)
    if octant_counts is not None:
        octant_score = float(np.sum(octant_counts > 0)) / 8.0
    else:
        octant_score = 0.5  # unknown

    # 3. Data spacing score (0–1, 1 = very close to data)
    # Exponential decay: score = exp(-2 * nearest / support_radius)
    if support_radius > 0 and np.isfinite(nearest_distance):
        spacing_score = float(np.exp(-2.0 * nearest_distance / max(support_radius, 1e-12)))
    else:
        spacing_score = 0.0

    # 4. Condition score (0–1, 1 = well-conditioned)
    if not np.isfinite(condnum) or condnum >= condition_fail:
        cond_score = 0.0
    elif condnum <= 1.0:
        cond_score = 1.0
    else:
        # Log-scale: score decreases from 1.0 at cond=1 to 0.0 at cond=fail
        log_cond = _safe_log10(condnum)
        log_fail = _safe_log10(condition_fail)
        cond_score = max(1.0 - log_cond / max(log_fail, 1.0), 0.0)

    # Composite: weighted blend (neighbourhood geometry dominates)
    w_nh = 0.30
    w_oct = 0.20
    w_sp = 0.30
    w_cond = 0.20

    composite = (w_nh * nh_score + w_oct * octant_score
                 + w_sp * spacing_score + w_cond * cond_score)

    # Invert: 0 = certain, 1 = uncertain
    uncertainty = 1.0 - composite

    return UncertaintyScore(
        index=float(np.clip(uncertainty, 0.0, 1.0)),
        neighbourhood_score=nh_score,
        spacing_score=spacing_score,
        condition_score=cond_score,
        octant_score=octant_score,
    )


@dataclass
class RBFSettings:
    drift: str = "constant"  # none, constant, linear
    smoothing: float = 0.0
    epsilon: float = 1e-10
    condition_warn: float = 1e8
    condition_fail: float = 1e12
    use_normal_scores: bool = True
    gh_order: int = 20
    # RBF-specific parameters
    basis_type: str = "wendland_c2"
    shape_parameter: float = 1.0
    support_radius: float = 1.0
    # Legacy fields (kept for backward compatibility with adapter)
    measurement_noise: float = 0.0
    high_variance_mask_ratio: float = 0.50
    stable_gh_ratio: float = 0.30
    cond_inflation_gamma: float = 0.20
    neff_inflation_eta: float = 0.25


@dataclass
class NeighbourhoodSettings:
    n_min: int = 16
    n_start: int = 32
    n_max: int = 64
    max_per_octant: int = 12
    search_radius: Optional[float] = None
    global_mode: bool = False  # True = Leapfrog-style global search (all data)


@dataclass
class BlockSettings:
    nx: int = 3
    ny: int = 3
    nz: int = 3
    dims: Tuple[float, float, float] = (10.0, 10.0, 10.0)

    def subpoint_offsets(self) -> np.ndarray:
        xs = (np.arange(self.nx, dtype=float) + 0.5) / self.nx - 0.5
        ys = (np.arange(self.ny, dtype=float) + 0.5) / self.ny - 0.5
        zs = (np.arange(self.nz, dtype=float) + 0.5) / self.nz - 0.5
        grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij"), dtype=float)
        pts = grid.reshape(3, -1).T
        dims = np.array(self.dims, dtype=float)
        return pts * dims


@dataclass
class PartitionSettings:
    enabled: bool = False
    tile_size: Tuple[float, float, float] = (100.0, 100.0, 100.0)
    overlap_factor: float = 2.0
    stitch_eta: float = 0.50
    min_coverage_ratio: float = 0.995


# -----------------------------------------------------------------------------
# Drift basis
# -----------------------------------------------------------------------------

def drift_basis(coords: np.ndarray, drift: str) -> np.ndarray:
    coords = _as_2d_float(coords)
    if drift == "none":
        return np.zeros((coords.shape[0], 0), dtype=float)
    if drift in ("constant", "auto"):
        # "auto" resolved to "constant" as a safe default
        return np.ones((coords.shape[0], 1), dtype=float)
    if drift == "linear":
        ones = np.ones((coords.shape[0], 1), dtype=float)
        return np.hstack([ones, coords])
    # Unknown drift type — fall back to constant instead of crashing
    logger.warning("Unknown drift type '%s', using 'constant'", drift)
    return np.ones((coords.shape[0], 1), dtype=float)


# -----------------------------------------------------------------------------
# Neighbourhood selection
# -----------------------------------------------------------------------------

def _octant_index(vecs: np.ndarray) -> np.ndarray:
    return ((vecs[:, 0] >= 0).astype(int) << 2) | ((vecs[:, 1] >= 0).astype(int) << 1) | ((vecs[:, 2] >= 0).astype(int) << 0)


def select_local_neighbours(
    tree: cKDTree,
    coords_t: np.ndarray,
    query_t: np.ndarray,
    nh: NeighbourhoodSettings,
    *,
    kernel_support: Optional[float] = None,
) -> np.ndarray:
    """Select local neighbours for a query point.

    Parameters
    ----------
    kernel_support : float, optional
        Hard distance cutoff in transformed space.  Samples beyond this
        distance have zero kernel contribution — including them only adds
        polynomial drift artefacts.  When provided, this overrides
        ``nh.search_radius`` if the latter is larger or absent.
    """
    n_max = min(nh.n_max, coords_t.shape[0])

    # Effective search radius: the tighter of the neighbourhood setting
    # and the kernel's compact support.  This prevents including samples
    # that contribute φ(r) = 0 but still affect the polynomial drift.
    eff_radius = nh.search_radius
    if kernel_support is not None:
        if eff_radius is None or kernel_support < eff_radius:
            eff_radius = kernel_support

    if eff_radius is None:
        # No search radius AND no kernel support — k-nearest fallback.
        # This should rarely happen (Wendland kernels always have support).
        dists, idx = tree.query(query_t, k=n_max)
        dists = np.atleast_1d(dists)
        idx = np.atleast_1d(idx)
    else:
        idx = np.array(tree.query_ball_point(query_t, r=eff_radius), dtype=int)
        if idx.size == 0:
            return np.array([], dtype=np.intp)
        cand = coords_t[idx] - query_t
        dists = np.linalg.norm(cand, axis=1)
        order = np.argsort(dists)
        idx = idx[order][:n_max]
        dists = dists[order][:n_max]

    if idx.size == 0:
        return np.array([], dtype=np.intp)

    vecs = coords_t[idx] - query_t
    oct_idx = _octant_index(vecs)

    selected: List[int] = []
    oct_counts: Dict[int, int] = {i: 0 for i in range(8)}
    for local_i, global_i in enumerate(idx):
        octant = int(oct_idx[local_i])
        if oct_counts[octant] < nh.max_per_octant:
            selected.append(int(global_i))
            oct_counts[octant] += 1
        if len(selected) >= n_max:
            break

    if len(selected) < nh.n_min:
        selected = list(map(int, idx[: max(nh.n_min, len(selected))]))

    return np.array(selected, dtype=int)


# -----------------------------------------------------------------------------
# Local RBF solve
# -----------------------------------------------------------------------------

@dataclass
class LocalSolveResult:
    pred_mean: float
    pred_var: float
    condnum: float
    neff: float
    n_used: int
    fail_flag: bool


class LocalRBFSystem:
    """Local RBF interpolation system.

    Builds and solves the augmented RBF system:
        [Φ + λI   P] [w]   [z]
        [P'       0] [β] = [0]

    where Φ_ij = φ(r_ij) is the RBF basis function evaluated at
    anisotropy-transformed distances, and P is the polynomial drift basis.
    """

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        rbf_kernel: RBFKernel,
        settings: RBFSettings,
        anisotropy: Anisotropy,
        *,
        variogram: Optional[VariogramModel] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        self.coords = _as_2d_float(coords)
        self.values = _as_1d_float(values, "values")
        self.rbf_kernel = rbf_kernel
        self.settings = settings
        self.anisotropy = anisotropy
        # Variogram kept only for legacy back-transform dispersion
        self._variogram = variogram
        # Per-sample weights (declustering). None means uniform.
        self.sample_weights: Optional[np.ndarray] = (
            np.asarray(sample_weights, dtype=float).ravel()
            if sample_weights is not None else None
        )

    def _distance_matrix(self, a_t: np.ndarray, b_t: np.ndarray) -> np.ndarray:
        diff = a_t[:, None, :] - b_t[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def solve_point(self, query: np.ndarray) -> LocalSolveResult:
        # Stability pass: drop near-duplicate samples before the solve.
        # Duplicates within the local neighbourhood make Phi rank-deficient.
        _coords = self.coords
        _values = self.values
        _weights_local = self.sample_weights
        n_orig = _coords.shape[0]
        if n_orig >= 2:
            _ct_pre = self.anisotropy.transform(_coords)
            _ord = np.lexsort(_ct_pre.T)
            _keep = np.zeros(n_orig, dtype=bool)
            _keep[_ord[0]] = True
            _prev = _ct_pre[_ord[0]]
            for _k in range(1, n_orig):
                _idx = _ord[_k]
                if float(np.linalg.norm(_ct_pre[_idx] - _prev)) > 1e-3:
                    _keep[_idx] = True
                    _prev = _ct_pre[_idx]
            if int(_keep.sum()) < n_orig:
                _coords = _coords[_keep]
                _values = _values[_keep]
                if _weights_local is not None:
                    _weights_local = _weights_local[_keep]

        # Local centring so the drift basis has O(1) magnitudes even
        # when coords are in UTM space.
        if _coords.shape[0] > 0:
            centroid = np.mean(_coords, axis=0, keepdims=True)
        else:
            centroid = np.zeros((1, 3), dtype=float)
        coords_centred = _coords - centroid
        query_centred = query.reshape(1, 3) - centroid

        coords_t = self.anisotropy.transform(_coords)
        query_t = self.anisotropy.transform(query.reshape(1, 3))

        # ── RBF interpolation matrix Φ ────────────────────────────────
        r_nn = self._distance_matrix(coords_t, coords_t)
        Phi = self.rbf_kernel.evaluate(r_nn)

        # Tikhonov regularisation: Φ + λI. Floor bumped from 0 to a
        # kernel-scale fraction so the diagonal never sits at exact
        # zero for smooth kernels that decay at large separations.
        diag_add = (self.rbf_kernel.nugget + self.settings.smoothing
                    + self.settings.epsilon)
        _kernel_scale = max(float(self.rbf_kernel.phi_at_zero), 1e-12)
        diag_add = max(diag_add, _kernel_scale * 1e-6)
        Phi += diag_add * np.eye(Phi.shape[0], dtype=float)

        # Per-sample declustering weights: add extra diagonal
        # regularisation inversely proportional to weight.
        if _weights_local is not None and _weights_local.size == Phi.shape[0]:
            _w = _weights_local.astype(float)
            _mean = float(np.mean(_w))
            if _mean > 0 and np.any(_w != _mean):
                _wn = _w / _mean
                _extra = _kernel_scale * np.maximum(
                    1.0 / np.maximum(_wn, 1e-3) - 1.0, 0.0
                )
                Phi[np.diag_indices_from(Phi)] += _extra

        # Polynomial drift basis — evaluate on CENTRED coordinates so
        # linear/quadratic drift terms have O(1) magnitudes.
        P = drift_basis(coords_centred, self.settings.drift)
        p0 = drift_basis(query_centred, self.settings.drift)

        # RBF vector: φ(||x - x_i||) for each sample i
        r_nq = self._distance_matrix(coords_t, query_t)
        phi0 = self.rbf_kernel.evaluate(r_nq).reshape(-1)

        n = Phi.shape[0]
        m = P.shape[1]
        A = np.zeros((n + m, n + m), dtype=float)
        A[:n, :n] = Phi
        if m > 0:
            A[:n, n:] = P
            A[n:, :n] = P.T

        rhs = np.concatenate([_values, np.zeros(m, dtype=float)])
        b0 = np.concatenate([phi0, p0.reshape(-1)])

        # Condition number — compute via full np.linalg.cond() so the
        # fail threshold can actually reject unstable solves. The old
        # diagonal-ratio heuristic missed most ill-conditioning.
        try:
            condnum = float(np.linalg.cond(A))
        except Exception:
            condnum = float("inf")
        fail_flag = False
        if not np.isfinite(condnum):
            condnum = self.settings.condition_fail
            fail_flag = True

        # Progressive regularisation:
        #   (a) Cholesky
        #   (b) Cholesky with boosted diagonal (up to 2 retries)
        #   (c) scipy.linalg.solve (LU on symmetric)
        #   (d) SVD least-squares via scipy.linalg.lstsq (last resort)
        solved = False
        for reg_attempt in range(3):
            try:
                cfac, lower = cho_factor(A, lower=True, check_finite=False)
                sol = cho_solve((cfac, lower), rhs, check_finite=False)
                solved = True
                break
            except LinAlgError:
                if reg_attempt < 2:
                    boost = float(np.mean(np.diag(A[:n, :n]))) * (10 ** (reg_attempt - 4))
                    A[:n, :n] += boost * np.eye(n, dtype=float)
                    continue

        if not solved:
            try:
                sol = solve(A, rhs, assume_a="sym")
                solved = True
            except Exception:
                pass

        if not solved:
            try:
                from scipy.linalg import lstsq as _lstsq
                sol, _res, _rank, _sv = _lstsq(A, rhs, lapack_driver="gelsd")
                _sv_pos = _sv[_sv > 0]
                if _sv_pos.size > 1:
                    condnum = float(_sv_pos.max() / _sv_pos.min())
                solved = True
            except Exception:
                pass

        if not solved:
            return LocalSolveResult(
                pred_mean=float(np.mean(_values)),
                pred_var=float(np.var(_values)),
                condnum=condnum,
                neff=1.0,
                n_used=n,
                fail_flag=True,
            )

        # ── RBF prediction: ẑ(x) = Σ w_i φ(r_i) + p(x)'β ────────────
        pred_mean = float(b0 @ sol)

        # ── Local range clamp ────────────────────────────────────────
        # Clip runaway predictions to the neighbourhood envelope ± 10%
        # tolerance so ill-conditioned solves can't produce absurd
        # extrema. Flag as fail if clamping was required.
        _local_min = float(np.min(_values))
        _local_max = float(np.max(_values))
        _tol = max(abs(_local_max - _local_min) * 0.10, 1e-9)
        _hi, _lo = _local_max + _tol, _local_min - _tol
        if pred_mean > _hi or pred_mean < _lo:
            pred_mean = float(np.clip(pred_mean, _lo, _hi))
            fail_flag = True

        # ── Local residual spread as variance proxy ────────────────────
        weights = sol[:n]
        Phi_raw = self.rbf_kernel.evaluate(r_nn)  # without regularisation
        fitted = Phi_raw @ weights
        if m > 0:
            fitted += P @ sol[n:]
        residuals = _values - fitted
        pred_var = float(np.var(residuals)) if n > 1 else 0.0

        if condnum >= self.settings.condition_fail:
            fail_flag = True

        w_abs = np.abs(weights)
        neff = float((np.sum(w_abs) ** 2) / max(np.sum(w_abs ** 2), 1e-12))

        if condnum >= self.settings.condition_fail:
            fail_flag = True

        return LocalSolveResult(
            pred_mean=pred_mean,
            pred_var=pred_var,
            condnum=condnum,
            neff=neff,
            n_used=n,
            fail_flag=fail_flag,
        )


# -----------------------------------------------------------------------------
# Partition-of-unity tiling
# -----------------------------------------------------------------------------

@dataclass
class Tile:
    tile_id: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    sample_idx: np.ndarray

    def contains_with_overlap(self, x: np.ndarray) -> bool:
        return bool(np.all(x >= self.bounds_min) and np.all(x <= self.bounds_max))

    def weight(self, x: np.ndarray) -> float:
        center = 0.5 * (self.bounds_min + self.bounds_max)
        half = 0.5 * (self.bounds_max - self.bounds_min)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.abs((x - center) / np.maximum(half, 1e-12))
        r = float(np.max(scaled))
        if r >= 1.0:
            return 0.0
        return (1.0 - r) ** 2


class PartitionManager:
    def __init__(self, coords: np.ndarray, settings: PartitionSettings) -> None:
        self.coords = _as_2d_float(coords)
        self.settings = settings
        self.tiles: List[Tile] = []
        if settings.enabled:
            self._build_tiles()

    def _build_tiles(self) -> None:
        mins = np.min(self.coords, axis=0)
        maxs = np.max(self.coords, axis=0)
        tile_size = np.array(self.settings.tile_size, dtype=float)
        overlap = tile_size * (self.settings.overlap_factor - 1.0) / 2.0

        grids = []
        for d in range(3):
            edges = [mins[d]]
            while edges[-1] < maxs[d]:
                edges.append(edges[-1] + tile_size[d])
            grids.append(np.array(edges, dtype=float))

        tile_id = 0
        for ix in range(len(grids[0]) - 1):
            for iy in range(len(grids[1]) - 1):
                for iz in range(len(grids[2]) - 1):
                    bmin = np.array([grids[0][ix], grids[1][iy], grids[2][iz]], dtype=float) - overlap
                    bmax = np.array([grids[0][ix + 1], grids[1][iy + 1], grids[2][iz + 1]], dtype=float) + overlap
                    mask = np.all((self.coords >= bmin) & (self.coords <= bmax), axis=1)
                    idx = np.where(mask)[0]
                    if idx.size > 0:
                        self.tiles.append(Tile(tile_id=tile_id, bounds_min=bmin, bounds_max=bmax, sample_idx=idx))
                        tile_id += 1

    def active_tiles(self, x: np.ndarray) -> List[Tile]:
        if not self.settings.enabled:
            return []
        return [tile for tile in self.tiles if tile.contains_with_overlap(x)]


# -----------------------------------------------------------------------------
# Main FastRBF estimator
# -----------------------------------------------------------------------------

@dataclass
class BlockEstimate:
    mean: float
    variance: float  # kept for interface compat; now = local residual spread
    std: float
    neff: float
    condnum: float
    n_subpoints: int
    stitch_variance: float
    coverage_count: int
    highvar_flag: bool
    backtransform_mode: str
    fail_flag: bool
    variance_ns: float = np.nan
    uncertainty_index: float = 0.5  # 0=well-informed, 1=unreliable


class FastRBFEstimator:
    def __init__(
        self,
        coords: ArrayLike,
        values: ArrayLike,
        anisotropy: Anisotropy,
        variogram: Optional[VariogramModel] = None,
        rbf_settings: Optional[RBFSettings] = None,
        neighbourhood: Optional[NeighbourhoodSettings] = None,
        block_settings: Optional[BlockSettings] = None,
        partition_settings: Optional[PartitionSettings] = None,
        *,
        rbf_kernel: Optional[RBFKernel] = None,
        sample_weights: Optional[ArrayLike] = None,
    ) -> None:
        coords = _as_2d_float(coords)
        values = _as_1d_float(values, "values")
        if coords.shape[1] != 3:
            raise ValueError("coords must be Nx3")
        if coords.shape[0] != values.size:
            raise ValueError("coords and values size mismatch")

        # Sample weights (e.g. declustering). Stored as a 1-D array of
        # length N and sliced by neighbourhood index in the solver
        # path. None means uniform weighting.
        self._sample_weights: Optional[np.ndarray] = None
        if sample_weights is not None:
            _w = np.asarray(sample_weights, dtype=float).ravel()
            if _w.size != values.size:
                raise ValueError(
                    f"sample_weights size {_w.size} does not match "
                    f"values size {values.size}"
                )
            _bad = ~np.isfinite(_w) | (_w < 0)
            if _bad.any():
                _w = _w.copy()
                _w[_bad] = 0.0
            if _w.sum() <= 0:
                logger.warning(
                    "FastRBFEstimator: sample_weights sum to zero — "
                    "ignoring weights."
                )
            else:
                self._sample_weights = _w

        # Detect exact duplicates and jitter them
        from scipy.spatial import cKDTree as _cKDTree
        _tree = _cKDTree(coords)
        pairs = _tree.query_pairs(r=1e-6)
        if pairs:
            coords = coords.copy()  # ensure writable for in-place jitter
            rng_jitter = np.random.RandomState(42)
            jittered = set()
            for i, j in pairs:
                if j not in jittered:
                    coords[j] += rng_jitter.uniform(-0.01, 0.01, size=3)
                    jittered.add(j)
            if jittered:
                logger.debug("Jittered %d collocated samples to break coordinate ties", len(jittered))

        self.coords = coords
        self.values_raw = values
        self.anisotropy = anisotropy
        self.rbf_settings = rbf_settings or RBFSettings()
        self.neighbourhood = neighbourhood or NeighbourhoodSettings()
        self.block_settings = block_settings or BlockSettings()
        self.partition_settings = partition_settings or PartitionSettings(enabled=False)

        # ── Resolve RBF kernel ────────────────────────────────────────────
        # If an explicit RBFKernel is provided, use it.
        # If only a VariogramModel is provided, convert it to an RBFKernel
        # using the variogram ranges as guidance (the variogram does NOT
        # define the system matrix — only the RBF kernel does).
        self.variogram = variogram  # kept for NS back-transform dispersion
        if rbf_kernel is not None:
            self.rbf_kernel = rbf_kernel
        elif variogram is not None:
            self.rbf_kernel = RBFKernel.from_variogram_guidance(
                variogram,
                basis=self.rbf_settings.basis_type,
            )
            logger.info(
                "ARBF: Auto-created RBFKernel from variogram guidance — "
                "basis=%s, shape=%.4f, support_radius=%.2f, nugget=%.6f",
                self.rbf_kernel.basis, self.rbf_kernel.shape_parameter,
                self.rbf_kernel.support_radius, self.rbf_kernel.nugget,
            )
        else:
            # No variogram and no kernel — use defaults from settings
            self.rbf_kernel = RBFKernel(
                basis=self.rbf_settings.basis_type,
                shape_parameter=self.rbf_settings.shape_parameter,
                support_radius=self.rbf_settings.support_radius,
            )

        # Store data statistics for grade bounding
        self._data_min = float(np.min(values))
        self._data_max = float(np.max(values))
        self._data_mean = float(np.mean(values))
        self._data_std = float(np.std(values))

        # Normal-score transform (optional — not required by RBF)
        self.nst = None
        if self.rbf_settings.use_normal_scores:
            self.nst = NormalScoreTransformer().fit(self.values_raw)
            self.values_model = self.nst.transform(self.values_raw)
        else:
            self.values_model = self.values_raw.copy()

        self.coords_t = self.anisotropy.transform(self.coords)
        self.tree = cKDTree(self.coords_t)
        self.partition = PartitionManager(self.coords, self.partition_settings)

        # Precompute block subpoint offsets (same for all blocks)
        self._cached_subpoint_offsets = self.block_settings.subpoint_offsets()

        # Import batch solver ONCE (not per estimate_block call)
        from .arbf_batch_solver import BlockBatchSolver
        self._BlockBatchSolver = BlockBatchSolver

        # Block support ratio: ratio of average within-block kernel value
        # to the kernel diagonal (φ(0)).  Measures how much the kernel
        # decorrelates across the block volume.
        offsets = self._cached_subpoint_offsets
        if offsets.shape[0] > 1:
            off_t = self.anisotropy.transform(offsets)
            diff = off_t[:, None, :] - off_t[None, :, :]
            h_sub = np.linalg.norm(diff, axis=2)
            phi_sub = self.rbf_kernel.evaluate(h_sub)
            phi0 = max(self.rbf_kernel.phi_at_zero, 1e-12)
            self._support_ratio = float(np.clip(np.mean(phi_sub) / phi0, 0.05, 1.0))
        else:
            self._support_ratio = 1.0

        # Cache tile KD-trees for partition mode
        self._tile_trees: Dict[int, cKDTree] = {}
        self._tile_coords_t: Dict[int, np.ndarray] = {}
        if self.partition_settings.enabled:
            for tile in self.partition.tiles:
                tc = self.anisotropy.transform(self.coords[tile.sample_idx])
                self._tile_coords_t[tile.tile_id] = tc
                self._tile_trees[tile.tile_id] = cKDTree(tc)

    def _solve_local_subpoint(self, x: np.ndarray, sample_idx: np.ndarray) -> LocalSolveResult:
        _local_w = (
            self._sample_weights[sample_idx]
            if self._sample_weights is not None else None
        )
        solver = LocalRBFSystem(
            coords=self.coords[sample_idx],
            values=self.values_model[sample_idx],
            rbf_kernel=self.rbf_kernel,
            settings=self.rbf_settings,
            anisotropy=self.anisotropy,
            variogram=self.variogram,
            sample_weights=_local_w,
        )
        return solver.solve_point(x)

    def _predict_subpoint_global(self, x: np.ndarray) -> LocalSolveResult:
        _ks = None if self.neighbourhood.global_mode else self.rbf_kernel.effective_support
        idx = select_local_neighbours(
            self.tree, self.coords_t,
            self.anisotropy.transform(x.reshape(1, 3))[0],
            self.neighbourhood,
            kernel_support=_ks,
        )
        return self._solve_local_subpoint(x, idx)

    def _predict_subpoint_partitioned(self, x: np.ndarray) -> Tuple[float, float, float, int, bool]:
        tiles = self.partition.active_tiles(x)
        if len(tiles) == 0:
            res = self._predict_subpoint_global(x)
            return res.pred_mean, res.pred_var, 0.0, 0, res.fail_flag

        weights = np.array([tile.weight(x) for tile in tiles], dtype=float)
        active = weights > 0.0
        tiles = [t for t, keep in zip(tiles, active) if keep]
        weights = weights[active]
        if weights.size == 0:
            res = self._predict_subpoint_global(x)
            return res.pred_mean, res.pred_var, 0.0, 0, res.fail_flag

        weights = weights / np.sum(weights)
        means = []
        vars_ = []
        fails = []
        for tile in tiles:
            local_coords_t = self.anisotropy.transform(self.coords[tile.sample_idx])
            local_tree = cKDTree(local_coords_t)
            query_t = self.anisotropy.transform(x.reshape(1, 3))[0]
            _ks = None if self.neighbourhood.global_mode else self.rbf_kernel.effective_support
            idx_local = select_local_neighbours(
                local_tree, local_coords_t, query_t, self.neighbourhood,
                kernel_support=_ks,
            )
            idx_global = tile.sample_idx[idx_local]
            res = self._solve_local_subpoint(x, idx_global)
            means.append(res.pred_mean)
            vars_.append(res.pred_var)
            fails.append(res.fail_flag)

        means_arr = np.asarray(means, dtype=float)
        vars_arr = np.asarray(vars_, dtype=float)
        mean_blend = float(np.sum(weights * means_arr))
        # Law of total variance: within + between
        var_within = float(np.sum(weights * vars_arr))
        var_between = float(np.sum(weights * (means_arr - mean_blend) ** 2))
        var_base = var_within + var_between

        stitch = 0.0
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                stitch += weights[i] * weights[j] * (means_arr[i] - means_arr[j]) ** 2
        stitch *= self.partition_settings.stitch_eta
        var_blend = var_base + stitch
        return mean_blend, var_blend, stitch, len(tiles), any(fails)

    def estimate_block(self, center: ArrayLike) -> BlockEstimate:
        """Estimate one block using batched solver (1 factorization per block)."""
        center = _as_1d_float(center, "center")
        if center.size != 3:
            raise ValueError("Block center must have 3 coordinates")
        # Centroid-only — analytical support_ratio handles block-support
        # correction.  Subpoint discretisation adds 27× overhead for <2%
        # improvement on smooth RBF fields.
        subpts = center.reshape(1, 3)
        n_sub = 1

        # ── Search neighbours ONCE at block centre ───────────────────────
        center_t = self.anisotropy.transform(center.reshape(1, 3))[0]
        # In global mode, skip kernel_support cutoff — all data contributes
        _ks = None if self.neighbourhood.global_mode else self.rbf_kernel.effective_support
        idx = select_local_neighbours(
            self.tree, self.coords_t, center_t, self.neighbourhood,
            kernel_support=_ks,
        )

        # No composites within search radius → uninformed block → NaN.
        # Previously this fell back to k-nearest (ignoring distance),
        # which silently returned the prior mean for ~88% of blocks.
        if idx.size == 0:
            return BlockEstimate(
                mean=np.nan,
                variance=np.nan,
                std=np.nan,
                neff=0.0,
                condnum=0.0,
                n_subpoints=n_sub,
                stitch_variance=0.0,
                coverage_count=0,
                highvar_flag=True,
                backtransform_mode="none",
                fail_flag=True,
            )

        if not self.partition_settings.enabled:
            # ── GLOBAL MODE: one solver, batched subpoints ───────────────
            solver = self._BlockBatchSolver(
                local_coords=self.coords[idx],
                local_values=self.values_model[idx],
                rbf_kernel=self.rbf_kernel,
                settings=self.rbf_settings,
                anisotropy=self.anisotropy,
                compute_condition=getattr(self.rbf_settings, '_compute_condition', False),
                sample_weights=(
                    self._sample_weights[idx]
                    if self._sample_weights is not None else None
                ),
            )
            result = solver.predict_subpoints_batch(subpts)

            sub_means_arr = result.sub_means
            sub_vars_arr = result.sub_vars
            stitch_var_total = 0.0
            coverage_count = n_sub
            fail_flag = result.fail_flag
            neff_mean = result.neff
            cond_mean = result.condnum

        else:
            # ── PARTITIONED MODE: blend tile solvers at block centre ─────
            tiles = self.partition.active_tiles(center)
            if len(tiles) == 0:
                # Fallback to global
                solver = self._BlockBatchSolver(
                    local_coords=self.coords[idx],
                    local_values=self.values_model[idx],
                    rbf_kernel=self.rbf_kernel,
                    settings=self.rbf_settings,
                    anisotropy=self.anisotropy,
                    sample_weights=(
                        self._sample_weights[idx]
                        if self._sample_weights is not None else None
                    ),
                )
                result = solver.predict_subpoints_batch(subpts)
                sub_means_arr = result.sub_means
                sub_vars_arr = result.sub_vars
                stitch_var_total = 0.0
                coverage_count = n_sub
                fail_flag = result.fail_flag
                neff_mean = result.neff
                cond_mean = result.condnum
            else:
                tile_weights = np.array([t.weight(center) for t in tiles], dtype=float)
                active = tile_weights > 0
                tiles = [t for t, a in zip(tiles, active) if a]
                tile_weights = tile_weights[active]
                if tile_weights.size == 0:
                    tiles = []
                else:
                    tile_weights /= np.sum(tile_weights)

                if len(tiles) == 0:
                    solver = self._BlockBatchSolver(
                        local_coords=self.coords[idx],
                        local_values=self.values_model[idx],
                        variogram=self.variogram,
                        settings=self.rbf_settings,
                        anisotropy=self.anisotropy,
                        sample_weights=(
                            self._sample_weights[idx]
                            if self._sample_weights is not None else None
                        ),
                    )
                    result = solver.predict_subpoints_batch(subpts)
                    sub_means_arr = result.sub_means
                    sub_vars_arr = result.sub_vars
                    stitch_var_total = 0.0
                    coverage_count = n_sub
                    fail_flag = result.fail_flag
                    neff_mean = result.neff
                    cond_mean = result.condnum
                else:
                    # Build one solver per tile (cached trees)
                    tile_means = []
                    tile_vars = []
                    tile_fails = []
                    for tile in tiles:
                        tc = self._tile_coords_t.get(tile.tile_id)
                        tt = self._tile_trees.get(tile.tile_id)
                        if tc is None or tt is None:
                            continue
                        _ks = None if self.neighbourhood.global_mode else self.rbf_kernel.effective_support
                        tidx = select_local_neighbours(
                            tt, tc, center_t, self.neighbourhood,
                            kernel_support=_ks,
                        )
                        global_idx = tile.sample_idx[tidx]
                        tsolver = BlockBatchSolver(
                            local_coords=self.coords[global_idx],
                            local_values=self.values_model[global_idx],
                            variogram=self.variogram,
                            settings=self.rbf_settings,
                            anisotropy=self.anisotropy,
                            sample_weights=(
                                self._sample_weights[global_idx]
                                if self._sample_weights is not None else None
                            ),
                        )
                        tres = tsolver.predict_subpoints_batch(subpts)
                        tile_means.append(tres.sub_means)
                        tile_vars.append(tres.sub_vars)
                        tile_fails.append(tres.fail_flag)

                    n_tiles = len(tile_means)
                    if n_tiles == 0:
                        sub_means_arr = np.full(n_sub, float(np.mean(self.values_model)))
                        sub_vars_arr = np.full(n_sub, float(np.var(self.values_model)))
                        stitch_var_total = 0.0
                        coverage_count = 0
                        fail_flag = True
                        neff_mean = 1.0
                        cond_mean = float("nan")
                    else:
                        tw = tile_weights[:n_tiles]
                        tw /= np.sum(tw)
                        means_stack = np.array(tile_means)  # (n_tiles, n_sub)
                        vars_stack = np.array(tile_vars)
                        sub_means_arr = np.sum(tw[:, None] * means_stack, axis=0)
                        # Law of total variance:
                        #   Var = E[Var(f|tile)] + Var(E[f|tile])
                        # Within-tile (expected conditional variance):
                        within_var = np.sum(tw[:, None] * vars_stack, axis=0)
                        # Between-tile (variance of conditional means):
                        blend_mean = sub_means_arr  # already computed
                        between_var = np.sum(
                            tw[:, None] * (means_stack - blend_mean[None, :]) ** 2,
                            axis=0,
                        )
                        sub_vars_arr = within_var + between_var
                        # Stitching variance (conservative penalty for tile disagreement)
                        stitch = 0.0
                        for i in range(n_tiles):
                            for j in range(i + 1, n_tiles):
                                diff = float(np.mean((means_stack[i] - means_stack[j]) ** 2))
                                stitch += tw[i] * tw[j] * diff
                        stitch *= self.partition_settings.stitch_eta
                        stitch_var_total = stitch
                        sub_vars_arr += stitch
                        coverage_count = n_sub
                        fail_flag = any(tile_fails)
                        neff_mean = float("nan")
                        cond_mean = float("nan")

        # ── Block-level aggregation ──────────────────────────────────────
        mean_ns = float(np.mean(sub_means_arr))
        # Block-support spread = within-subpoint + between-subpoint
        within_var = float(np.mean(sub_vars_arr))
        between_var = float(np.var(sub_means_arr)) if len(sub_means_arr) > 1 else 0.0
        var_ns_block = max((within_var + between_var) * self._support_ratio, 0.0)

        # Use data variance as reference scale (not variogram sill)
        data_var = float(np.var(self.values_model))
        ref_scale = max(data_var, 1e-12)
        highvar_flag = var_ns_block > (self.rbf_settings.high_variance_mask_ratio * ref_scale)
        stable_gh = var_ns_block <= (self.rbf_settings.stable_gh_ratio * ref_scale)

        # Store NS-space variance before back-transform
        variance_ns_out = var_ns_block

        if self.rbf_settings.use_normal_scores and self.nst is not None:
            # ── SUBPOINT-LEVEL back-transform ──────────────────────────
            # Back-transform each subpoint using its local residual
            # spread as dispersion, then average in raw space.
            gh_order = self.rbf_settings.gh_order
            _nugget_floor = max(self.rbf_kernel.nugget, 1e-6)
            _sill_cap = max(data_var, 1.0)
            n_sub_bt = sub_means_arr.shape[0]
            if n_sub_bt > 1:
                bt_dispersion = np.clip(sub_vars_arr, _nugget_floor, _sill_cap)
                sub_means_raw, sub_vars_raw = self.nst.inverse_backtransform_batch(
                    sub_means_arr, bt_dispersion, gh_order,
                )
                mean_out = float(np.nanmean(sub_means_raw))
                var_out = float(np.nanmean(sub_vars_raw))
            else:
                bt_disp = max(min(float(sub_vars_arr[0]), _sill_cap), _nugget_floor)
                mean_out = self.nst.inverse_expectation(mean_ns, bt_disp, gh_order=gh_order)
                second_orig = self.nst.inverse_second_moment(mean_ns, bt_disp, gh_order=gh_order)
                var_out = max(second_orig - mean_out ** 2, 0.0)
            backtransform_mode = "stable_gh" if stable_gh else ("highvar_gh" if not highvar_flag else "masked_highvar")
        else:
            mean_out = mean_ns
            var_out = var_ns_block
            backtransform_mode = "raw_space"

        # ── Uncertainty index (replaces kriging variance for classification) ──
        _nearest_dist = 0.0
        if hasattr(result, 'neff'):
            # Use result from batch solver
            _nearest_dist_arr = self.tree.query(center_t.reshape(1, -1), k=1)[0]
            _nearest_dist = float(_nearest_dist_arr[0]) if np.isfinite(_nearest_dist_arr[0]) else 999.0
        else:
            _nearest_dist = 999.0

        uncertainty = compute_uncertainty_index(
            n_used=len(idx),
            n_min=self.neighbourhood.n_min,
            n_max=self.neighbourhood.n_max,
            octant_counts=None,  # not tracked per block in batch mode
            nearest_distance=_nearest_dist,
            support_radius=self.rbf_kernel.support_radius,
            condnum=float(cond_mean) if np.isfinite(cond_mean) else 1e12,
            condition_warn=self.rbf_settings.condition_warn,
            condition_fail=self.rbf_settings.condition_fail,
        )

        coverage_ratio = coverage_count / max(n_sub, 1)
        if self.partition_settings.enabled and coverage_ratio < self.partition_settings.min_coverage_ratio:
            fail_flag = True

        # ── Grade bounding: prevent catastrophic extrapolation ────────────
        bound_margin = max(3.0 * self._data_std, abs(self._data_max - self._data_min) * 0.5)
        grade_floor = self._data_min - bound_margin
        grade_ceiling = self._data_max + bound_margin
        if self._data_min >= 0.0:
            grade_floor = max(grade_floor, 0.0)

        if not np.isnan(mean_out):
            if mean_out < grade_floor or mean_out > grade_ceiling:
                logger.debug(
                    "Grade bounded: %.2f clipped to [%.2f, %.2f]",
                    mean_out, grade_floor, grade_ceiling,
                )
                mean_out = float(np.clip(mean_out, grade_floor, grade_ceiling))
                fail_flag = True

        return BlockEstimate(
            mean=float(mean_out),
            variance=float(max(var_out, 0.0)),
            std=float(math.sqrt(max(var_out, 0.0))),
            neff=float(neff_mean) if np.isfinite(neff_mean) else float("nan"),
            condnum=float(cond_mean) if np.isfinite(cond_mean) else float("nan"),
            n_subpoints=n_sub,
            stitch_variance=float(stitch_var_total) if self.partition_settings.enabled else 0.0,
            coverage_count=coverage_count,
            highvar_flag=highvar_flag,
            backtransform_mode=backtransform_mode,
            fail_flag=fail_flag,
            variance_ns=float(variance_ns_out),
            uncertainty_index=uncertainty.index,
        )

    def estimate_blocks(self, centers: ArrayLike, profile: bool = False,
                        n_workers: Optional[int] = None,
                        progress_callback=None) -> Dict[str, np.ndarray]:
        """Estimate all blocks, optionally in parallel.

        Parameters
        ----------
        centers : (N, 3) array
        profile : bool
        n_workers : int or None
            Number of parallel threads.  ``None`` = auto (min(cpu_count, 8)).
            Set to 1 to disable parallelism (e.g. for debugging).
        progress_callback : callable(percent: int, message: str) or None
        """
        from .arbf_profiler import EstimationProfiler
        from scipy.spatial import cKDTree
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        centers = _as_2d_float(centers)
        n = centers.shape[0]
        prof = EstimationProfiler() if profile else None

        # ── Output arrays: NaN by default (uninformed blocks stay NaN) ──
        out = {
            "ARBF_GRADE": np.full(n, np.nan, dtype=float),
            "ARBF_VAR_BLOCK": np.full(n, np.nan, dtype=float),
            "ARBF_VAR_NS": np.full(n, np.nan, dtype=float),
            "ARBF_STD_BLOCK": np.full(n, np.nan, dtype=float),
            "ARBF_NEFF": np.full(n, np.nan, dtype=float),
            "ARBF_CONDNUM": np.full(n, np.nan, dtype=float),
            "ARBF_PUM_COUNT": np.zeros(n, dtype=float),
            "ARBF_STITCH_VAR": np.zeros(n, dtype=float),
            "ARBF_HIGHVAR_FLAG": np.zeros(n, dtype=int),
            "ARBF_FAIL_FLAG": np.ones(n, dtype=int),  # default = failed/outside
            "ARBF_UNCERTAINTY": np.ones(n, dtype=float),  # 0=certain, 1=uncertain
            # ARBF_INFORMED: True if the block has at least one composite
            # within one variogram range (transformed-space distance ≤ 1.0).
            # This is independent of the solver path — used by downstream
            # panel QA to drop extrapolated blocks from the denominator.
            "ARBF_INFORMED": np.zeros(n, dtype=bool),
        }

        # ── Pre-filter: only estimate blocks within search radius ────
        # Build a KDTree of composites in anisotropy-transformed space
        # and query which block centroids have at least one composite
        # within the search radius.  Blocks outside the mask are never
        # sent to the estimator — they stay NaN, saving computation.
        #
        # ── Pre-filter: skip blocks with no nearby composites ─────
        _is_global = self.neighbourhood.global_mode
        centers_t = self.anisotropy.transform(centers)
        _range_max = float(max(self.anisotropy.ranges))

        # ── ARBF_INFORMED mask (independent of solver path) ─────────────
        # A block is "informed" if at least one composite lies within one
        # variogram range in anisotropy-transformed space. This is a
        # tighter definition than the solver's search radius and is used
        # by the panel QA to drop extrapolated blocks.
        try:
            _informed_tree = cKDTree(self.coords_t)
            _informed_d, _ = _informed_tree.query(centers_t, k=1)
            out["ARBF_INFORMED"] = (_informed_d <= 1.0)
            logger.info(
                "ARBF_INFORMED: %d / %d blocks within 1 variogram range "
                "(%.1f%%)",
                int(out["ARBF_INFORMED"].sum()), n,
                100.0 * out["ARBF_INFORMED"].mean(),
            )
        except Exception as _exc:
            logger.warning("ARBF_INFORMED: fallback — %s", _exc)

        if _is_global:
            # Global mode: every block is informed — no pre-filter needed.
            inside_idx = np.arange(n)
            n_inside = n
            n_outside = 0
            logger.info(
                "Block pre-filter: GLOBAL mode — all %d blocks will be estimated "
                "(all composites inform every block)", n,
            )
        else:
            # Local mode: neighbourhood.search_radius is in TRANSFORMED SPACE
            # (1 unit = 1 variogram range).  Also honour the kernel's compact
            # support as a hard cutoff.
            _search_rad_t = getattr(self.neighbourhood, 'search_radius', None)
            _kernel_support = self.rbf_kernel.effective_support
            if _search_rad_t is None or _search_rad_t <= 0:
                if _kernel_support is not None:
                    _search_rad_t = _kernel_support
                else:
                    _search_rad_t = 3.0  # fallback: 3× range
            # Tighten to kernel support if available
            if _kernel_support is not None and _kernel_support < _search_rad_t:
                _search_rad_t = _kernel_support

            comp_tree = cKDTree(self.coords_t)
            try:
                near_counts = comp_tree.query_ball_point(
                    centers_t, r=_search_rad_t, return_length=True)
                inside_mask = np.asarray(near_counts) > 0
            except TypeError:
                neighbours = comp_tree.query_ball_point(centers_t, r=_search_rad_t)
                inside_mask = np.array([len(nb) > 0 for nb in neighbours])
            inside_idx = np.where(inside_mask)[0]
            n_inside = len(inside_idx)
            n_outside = n - n_inside

            logger.info(
                "Block pre-filter: %d / %d blocks inside search radius "
                "(%.1f%%), %d outside (skipped). "
                "search_radius=%.2f transformed (≈%.1f m at range_max=%.1f)",
                n_inside, n, 100.0 * n_inside / max(n, 1),
                n_outside, _search_rad_t, _search_rad_t * _range_max, _range_max,
            )

        if n_inside == 0:
            logger.warning("No blocks within search radius — all NaN")
            return out

        # Only pass the inside blocks to the estimator
        inside_centers = centers[inside_idx]

        # ── Choose estimation strategy based on the user's explicit
        # search-mode selection ONLY. The previous data-size heuristic
        # (``n_samples <= 5000 and n_inside > 500``) silently escalated
        # "Local" runs to a single global solve on virtually every real
        # mining deposit, bypassing the per-block neighbourhood selection
        # and the search radius. That was the root cause of ARBF's
        # long-range leakage and over-smoothing of local highs. We now
        # honour ``neighbourhood.global_mode`` directly: global only if
        # the user explicitly picked Global (Leapfrog-style) in the
        # panel. Everything else goes through the local per-block path
        # (estimate_block → select_local_neighbours → BlockBatchSolver).
        n_samples = self.coords.shape[0]
        use_global = bool(self.neighbourhood.global_mode)

        logger.info(
            "ARBF ENGINE: use_global=%s  partition.enabled=%s  "
            "n_samples=%d  n_inside=%d  n_blocks=%d",
            use_global, self.partition_settings.enabled,
            n_samples, n_inside, n,
        )

        # ── Global solve: estimate ALL blocks, not just those within
        # search radius.  The global RBF system uses every sample in one
        # factorisation, so any block location is valid.  Blocks far from
        # data will naturally get high uncertainty (→ low confidence),
        # which is the correct behaviour.  Restricting to the search
        # radius creates artificial "islands" around drillholes.
        # The local solve still needs the search radius because it
        # builds a per-block neighbourhood.
        if use_global:
            global_n = n  # estimate ALL blocks
            global_centers = centers
            global_idx = np.arange(n)
            logger.info(
                "Global solve selected — estimating ALL %d blocks "
                "(search-radius pre-filter bypassed for global path)",
                global_n,
            )
        else:
            global_n = n_inside
            global_centers = inside_centers
            global_idx = inside_idx

        # Temporary output arrays
        inside_out = {
            "ARBF_GRADE": np.zeros(global_n, dtype=float),
            "ARBF_VAR_BLOCK": np.zeros(global_n, dtype=float),
            "ARBF_VAR_NS": np.zeros(global_n, dtype=float),
            "ARBF_STD_BLOCK": np.zeros(global_n, dtype=float),
            "ARBF_NEFF": np.full(global_n, np.nan, dtype=float),
            "ARBF_CONDNUM": np.full(global_n, np.nan, dtype=float),
            "ARBF_PUM_COUNT": np.zeros(global_n, dtype=float),
            "ARBF_STITCH_VAR": np.zeros(global_n, dtype=float),
            "ARBF_HIGHVAR_FLAG": np.zeros(global_n, dtype=int),
            "ARBF_FAIL_FLAG": np.zeros(global_n, dtype=int),
            "ARBF_UNCERTAINTY": np.full(global_n, 0.5, dtype=float),
        }

        if use_global:
            self._estimate_blocks_global(global_centers, inside_out,
                                         progress_callback)
        else:
            progress_interval = max(global_n // 20, 1)
            for i in range(global_n):
                est = self.estimate_block(global_centers[i])
                inside_out["ARBF_GRADE"][i] = est.mean
                inside_out["ARBF_VAR_BLOCK"][i] = est.variance
                inside_out["ARBF_VAR_NS"][i] = est.variance_ns
                inside_out["ARBF_STD_BLOCK"][i] = est.std
                inside_out["ARBF_NEFF"][i] = est.neff
                inside_out["ARBF_CONDNUM"][i] = est.condnum
                inside_out["ARBF_PUM_COUNT"][i] = est.coverage_count
                inside_out["ARBF_STITCH_VAR"][i] = est.stitch_variance
                inside_out["ARBF_HIGHVAR_FLAG"][i] = int(est.highvar_flag)
                inside_out["ARBF_FAIL_FLAG"][i] = int(est.fail_flag)
                inside_out["ARBF_UNCERTAINTY"][i] = est.uncertainty_index
                if progress_callback and (i + 1) % progress_interval == 0:
                    progress_callback(int((i + 1) / global_n * 100),
                                      f"Estimated {i + 1}/{global_n} blocks")

        # ── Scatter results back into full-size output arrays ──────────
        for key in inside_out:
            out[key][global_idx] = inside_out[key]

        logger.info(
            "Estimation complete: %d blocks estimated, %d skipped (outside mask)",
            global_n, n - global_n,
        )

        if prof is not None:
            prof.set("blocks_estimated", n_inside)
            prof.set("blocks_skipped", n_outside)
            prof.set("subpoints_per_block", self.block_settings.nx * self.block_settings.ny * self.block_settings.nz)
            out["_profile"] = prof.summary()

        return out

    def _estimate_blocks_global(self, centers: np.ndarray,
                                out: Dict[str, np.ndarray],
                                progress_callback=None) -> None:
        """Fast global-solve estimation: ONE factorisation, bulk evaluation.

        Builds the GLOBAL RBF interpolation matrix Φ from the chosen
        basis function, solves the augmented system ONCE, then evaluates
        at all M target points via batched matrix-vector products.

        The system is:
            [Φ + λI   P] [w]   [z]
            [P'       0] [β] = [0]

        Prediction at point x:
            ẑ(x) = Σ w_i φ(||x - x_i||) + p(x)'β
        """
        import warnings
        from scipy.linalg import cho_factor, cho_solve, solve, LinAlgWarning, LinAlgError
        from scipy.spatial.distance import cdist

        n_samples = self.coords.shape[0]
        n_blocks = centers.shape[0]
        logger.info("Global RBF solve: %d samples, %d blocks, basis=%s",
                     n_samples, n_blocks, self.rbf_kernel.basis)

        # ── Local centring for drift basis ─────────────────────────────
        # UTM coordinates are huge (x≈259,000, y≈7,922,000) so a linear
        # drift basis [1, x, y, z] evaluated in raw units gives polynomial
        # columns with magnitudes many orders larger than Φ, driving the
        # augmented system's condition number to 1e18+ and producing
        # ill-conditioning warnings from scipy.linalg.solve. Centre the
        # coordinates to the composite centroid so [1, x_c, y_c, z_c] all
        # sit in O(range_max). Distances (and therefore Φ) are unchanged.
        self._global_centroid = np.mean(self.coords, axis=0, keepdims=True)
        coords_centred = self.coords - self._global_centroid

        # ── 1. Build and factorise the GLOBAL augmented RBF system ─────
        coords_t = self.coords_t  # anisotropy-transformed
        r_nn = cdist(coords_t, coords_t)

        # ── RBF interpolation matrix Φ ─────────────────────────────────
        Phi = self.rbf_kernel.evaluate(r_nn)

        # Log Φ matrix density for diagnostics
        offdiag_mask = ~np.eye(n_samples, dtype=bool)
        pct_nonzero = float((np.abs(Phi[offdiag_mask]) > 1e-12).mean())
        logger.info(
            "Phi matrix: %dx%d, basis=%s, %.1f%% nonzero off-diagonal",
            n_samples, n_samples, self.rbf_kernel.basis, pct_nonzero * 100,
        )

        # ── Sparsity check for compactly-supported kernels ─────────────
        # If too few sample pairs have non-zero kernel values, the system
        # may produce isolated islands.  Warn but do not auto-inject
        # background — that was a variogram-covariance pattern.
        if pct_nonzero < 0.10 and self.rbf_kernel.basis.startswith("wendland"):
            logger.warning(
                "ARBF: Only %.1f%% of sample pairs have non-zero "
                "kernel values.  Consider increasing support_radius "
                "(current: %.2f) or switching to a global kernel "
                "(gaussian, inverse_multiquadric).",
                pct_nonzero * 100, self.rbf_kernel.support_radius,
            )

        # Tikhonov regularisation: Φ + λI
        # Adaptive diagonal floor raised from 1e-8 to 1e-6 so
        # well-conditioned long-range kernels still carry a non-trivial
        # regulariser and don't slide into rcond ≈ 1e-19 territory when
        # samples cluster tightly relative to the kernel support.
        kernel_scale = max(float(np.mean(np.diag(Phi))), 1e-12)
        diag_reg = max(
            self.rbf_kernel.nugget + self.rbf_settings.smoothing
            + self.rbf_settings.epsilon,
            kernel_scale * 1e-6,
        )
        Phi += diag_reg * np.eye(n_samples, dtype=float)

        # Drift basis — evaluate on CENTRED coords so the linear/quadratic
        # drift columns have O(1) magnitude instead of O(UTM).
        drift_mode = self.rbf_settings.drift
        P = drift_basis(coords_centred, drift_mode)
        m = P.shape[1]

        nm = n_samples + m
        A = np.zeros((nm, nm), dtype=float)
        A[:n_samples, :n_samples] = Phi
        if m > 0:
            A[:n_samples, n_samples:] = P
            A[n_samples:, :n_samples] = P.T

        rhs = np.concatenate([self.values_model, np.zeros(m, dtype=float)])

        # ── Progressive solver: Cholesky → boosted Cholesky → LU sym
        # → SVD lstsq → fallback. Same pattern used by BlockBatchSolver
        # so the global path rejects ill-conditioned solves too. Without
        # this, a single sample cluster near the origin produced
        # rcond ≈ 8e-19 and let 97k grade values leak out.
        sol = None
        use_cho = False
        factor = lower = None
        solver_path = "cholesky"
        _A_work = A.copy()
        for _attempt in range(3):
            try:
                factor, lower = cho_factor(_A_work, lower=True, check_finite=False)
                sol = cho_solve((factor, lower), rhs, check_finite=False)
                use_cho = True
                if _attempt > 0:
                    solver_path = f"cholesky_boosted_{_attempt}"
                break
            except LinAlgError:
                _boost = kernel_scale * 10.0 ** (_attempt - 3)
                _A_work[:n_samples, :n_samples] += _boost * np.eye(
                    n_samples, dtype=float,
                )
        if sol is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=LinAlgWarning)
                    sol = solve(_A_work, rhs, assume_a="sym")
                solver_path = "lu_sym"
                factor = lower = None
            except Exception:
                pass
        if sol is None:
            try:
                from scipy.linalg import lstsq as _lstsq
                _sol, _res, _rank, _sv = _lstsq(
                    _A_work, rhs, lapack_driver="gelsd",
                )
                sol = _sol
                solver_path = "svd_lstsq"
                factor = lower = None
            except Exception:
                pass
        if sol is None:
            logger.error(
                "Global RBF solve: all solver paths failed — returning "
                "data-mean fallback",
            )
            sol = np.zeros(nm, dtype=float)
            sol[:n_samples] = float(np.mean(self.values_model)) / max(n_samples, 1)
            solver_path = "fallback_uniform"

        A = _A_work  # keep boosted A for any later users

        # Condition number — compute once for diagnostics.
        try:
            _condnum_global = float(np.linalg.cond(A))
        except Exception:
            _condnum_global = float("inf")

        logger.info(
            "Global system factorised: %dx%d (solver=%s, cond=%.2e)",
            nm, nm, solver_path, _condnum_global,
        )

        # ── 2. Block-support estimation ────────────────────────────────
        # Adaptive subpoint discretisation: only evaluate at multiple
        # subpoints when block size is a significant fraction of the
        # variogram range (ratio > 0.25).  For small blocks relative to
        # ranges (typical in mining: 10m blocks, 200m+ ranges), the
        # centre-point estimate is an excellent approximation of the
        # block mean, and the analytical support_ratio already provides
        # the correct block-support variance correction.
        #
        # When subpoints ARE used: evaluate grades at all subpoints
        # (fast — just sol @ B), compute variance at centres only
        # (expensive — Cholesky back-solve), add between-subpoint
        # variance for proper block-support variance estimation.
        sub_offsets = self._cached_subpoint_offsets  # (n_sub, 3)
        n_sub = sub_offsets.shape[0]
        block_dims = np.array(self.block_settings.dims, dtype=float)
        vranges = np.array(self.anisotropy.ranges, dtype=float)
        # Per-axis ratio: block_dim[i] / range[i].  Mining deposits often
        # have very short vertical ranges (10-20m) relative to horizontal
        # (100-500m).  Using min(ranges) would trigger subpoints on nearly
        # every deposit.  The analytical support_ratio already corrects
        # variance, so subpoints are only needed when a block dimension
        # EXCEEDS its corresponding range (ratio > 1.0).
        per_axis_ratio = block_dims / np.maximum(vranges, 1e-6)
        block_range_ratio = float(np.max(per_axis_ratio))

        # SPEED: always use centroid-only prediction.  The analytical
        # support_ratio already provides the correct block-support variance
        # correction.  Subpoint discretisation adds ~27× cdist/kernel cost
        # for <2% grade improvement on smooth RBF fields.
        use_subpoints = False
        n_sub_eff = 1

        logger.info(
            "Global path: max block/range ratio=%.2f — centre-point "
            "estimation with analytical support correction "
            "(support_ratio=%.4f, per-axis: %.2f / %.2f / %.2f)",
            block_range_ratio, self._support_ratio, *per_axis_ratio,
        )

        chunk_size = min(20000, max(1000, n_blocks // 10))
        data_var = float(np.var(self.values_model))

        # ── Precompute global RBF fit residuals at sample locations ───
        # Phi is already regularised (Phi + λI).  The system solves
        # (Phi+λI) @ w = z, so Phi_reg @ w ≈ z (near-zero residual).
        # To get the ACTUAL smoothing residual, evaluate with the
        # UNREGULARISED kernel: fitted = Phi_raw @ w + P @ beta.
        # The difference z - fitted captures the smoothing effect.
        Phi_raw = self.rbf_kernel.evaluate(r_nn)  # without diagonal reg
        _w = sol[:n_samples]
        _fitted_at_samples = Phi_raw @ _w
        if m > 0:
            _fitted_at_samples += P @ sol[n_samples:]
        global_resid = self.values_model - _fitted_at_samples
        global_resid_sq = global_resid ** 2
        global_resid_var = float(np.mean(global_resid_sq))
        logger.info(
            "Global RBF fit: residual std=%.4f (%.1f%% of data std)",
            np.sqrt(global_resid_var),
            100.0 * np.sqrt(global_resid_var) / max(self._data_std, 1e-12),
        )

        # ── 2a. Subpoint grades (only when needed) ───────────────────
        block_means = np.zeros(n_blocks, dtype=float)
        between_var = np.zeros(n_blocks, dtype=float)

        if use_subpoints:
            sub_grade_chunk = max(min(20000 // max(n_sub, 1), 5000), 100)
            for start in range(0, n_blocks, sub_grade_chunk):
                end = min(start + sub_grade_chunk, n_blocks)
                chunk_centers = centers[start:end]
                n_chunk = end - start

                sub_pts = (chunk_centers[:, None, :]
                           + sub_offsets[None, :, :]).reshape(-1, 3)
                n_pts = sub_pts.shape[0]

                sub_t = self.anisotropy.transform(sub_pts)
                r_sub = cdist(sub_t, coords_t)
                phi_sub = self.rbf_kernel.evaluate(r_sub)

                B_sub = np.zeros((nm, n_pts), dtype=float)
                B_sub[:n_samples, :] = phi_sub.T
                if m > 0:
                    B_sub[n_samples:, :] = drift_basis(
                        sub_pts - self._global_centroid, drift_mode,
                    ).T

                sub_means = sol @ B_sub
                sub_2d = sub_means.reshape(n_chunk, n_sub)

                block_means[start:end] = np.mean(sub_2d, axis=1)
                between_var[start:end] = np.var(sub_2d, axis=1)

                if progress_callback:
                    pct = int(end / n_blocks * 30)
                    progress_callback(pct, f"Subpoint grades {end:,}/{n_blocks:,}")

        # ── 2b. Centre-point grades + uncertainty ─────────────────────
        # RBF prediction: ẑ(x) = Σ w_i φ(||x - x_i||) + p(x)'β
        # Per-block variance from kernel-weighted global residuals.
        # Per-block uncertainty from neighbourhood geometry.
        for start in range(0, n_blocks, chunk_size):
            end = min(start + chunk_size, n_blocks)
            chunk_centers = centers[start:end]
            n_chunk = end - start

            chunk_t = self.anisotropy.transform(chunk_centers)
            r_chunk = cdist(chunk_t, coords_t)
            phi_chunk = self.rbf_kernel.evaluate(r_chunk)

            B = np.zeros((nm, n_chunk), dtype=float)
            B[:n_samples, :] = phi_chunk.T
            if m > 0:
                B[n_samples:, :] = drift_basis(
                    chunk_centers - self._global_centroid, drift_mode,
                ).T

            # Centre-point grades (used as block mean when subpoints skipped)
            if not use_subpoints:
                block_means[start:end] = sol @ B

            means = block_means[start:end].copy()

            # ── Per-block variance proxy from global RBF residuals ────
            # The global RBF fit has residuals at the sample locations.
            # For each block, compute a distance-weighted local residual
            # variance using the kernel values as weights.
            #
            # fitted_at_samples = sol @ Phi_global (precomputed once)
            # residuals = values - fitted_at_samples
            # local_var(block) = Σ φ(r_bi) * residual_i² / Σ φ(r_bi)
            #
            # This gives blocks near high-residual regions higher variance,
            # and blocks in well-fitted regions lower variance.

            # phi_chunk is (n_chunk, n_samples) — kernel values from block to each sample
            phi_weights = phi_chunk  # (n_chunk, n_samples)
            phi_sums = np.sum(phi_weights, axis=1, keepdims=True)  # (n_chunk, 1)
            phi_sums = np.maximum(phi_sums, 1e-12)

            # Weighted local residual variance per block
            # global_resid_sq is (n_samples,) — precomputed before the chunk loop
            vars_chunk = np.sum(phi_weights * global_resid_sq[None, :], axis=1) / phi_sums.ravel()
            vars_chunk = np.maximum(vars_chunk, 0.0)

            # Add between-subpoint spread if subpoints were used
            bv = between_var[start:end]
            vars_chunk_ns = (vars_chunk + bv) * self._support_ratio
            vars_chunk_ns = np.maximum(vars_chunk_ns, 1e-12)

            # ── Uncertainty index per block (vectorized) ──────────────
            nearest_dists = np.min(r_chunk, axis=1)  # (n_chunk,)
            n_nonzero_per_block = np.sum(phi_chunk > 1e-12, axis=1)  # (n_chunk,)
            _n_min = self.neighbourhood.n_min
            _n_max = self.neighbourhood.n_max
            _sup_r = max(self.rbf_kernel.support_radius, 1e-12)
            if _n_max > _n_min:
                nh_scores = np.clip(
                    (n_nonzero_per_block - _n_min) / (_n_max - _n_min), 0.0, 1.0,
                )
            else:
                nh_scores = np.where(
                    n_nonzero_per_block >= _n_min, 1.0,
                    np.maximum(n_nonzero_per_block / max(_n_min, 1), 0.0),
                )
            sp_scores = np.exp(-2.0 * nearest_dists / _sup_r)
            # cond_score=1.0 for global solve, octant_score=0.5 (unknown)
            uncertainty = 1.0 - (0.30 * nh_scores + 0.20 * 0.5
                                 + 0.30 * sp_scores + 0.20 * 1.0)
            out["ARBF_UNCERTAINTY"][start:end] = np.clip(uncertainty, 0.0, 1.0)

            # Back-transform if normal scores
            if self.nst is not None:
                _nugget_floor = max(self.rbf_kernel.nugget, 1e-6)
                _sill_cap = max(data_var, 1.0)
                bt_dispersion = np.clip(vars_chunk_ns, _nugget_floor, _sill_cap)
                means, vars_bt = self.nst.inverse_backtransform_batch(
                    means, bt_dispersion, self.rbf_settings.gh_order,
                )
                vars_chunk = vars_bt
            else:
                vars_chunk = vars_chunk_ns

            # Grade bounding
            bound_margin = max(3.0 * self._data_std,
                               abs(self._data_max - self._data_min) * 0.5)
            grade_floor = self._data_min - bound_margin
            grade_ceiling = self._data_max + bound_margin
            if self._data_min >= 0.0:
                grade_floor = max(grade_floor, 0.0)
            n_clipped = int(np.sum((means < grade_floor) | (means > grade_ceiling)))
            if n_clipped > 0:
                logger.warning(
                    "Global path: %d/%d blocks clipped to [%.2f, %.2f]",
                    n_clipped, n_chunk, grade_floor, grade_ceiling,
                )
                out["ARBF_FAIL_FLAG"][start:end] = np.where(
                    (means < grade_floor) | (means > grade_ceiling),
                    1, out["ARBF_FAIL_FLAG"][start:end],
                )
                means = np.clip(means, grade_floor, grade_ceiling)

            out["ARBF_GRADE"][start:end] = means
            out["ARBF_VAR_BLOCK"][start:end] = vars_chunk
            out["ARBF_VAR_NS"][start:end] = vars_chunk_ns
            out["ARBF_STD_BLOCK"][start:end] = np.sqrt(np.maximum(vars_chunk, 0.0))
            out["ARBF_NEFF"][start:end] = n_samples
            out["ARBF_PUM_COUNT"][start:end] = n_sub_eff

            # High-variance flag uses uncertainty index threshold
            out["ARBF_HIGHVAR_FLAG"][start:end] = (
                out["ARBF_UNCERTAINTY"][start:end] > self.rbf_settings.high_variance_mask_ratio
            ).astype(int)

            if progress_callback:
                base = 30 if use_subpoints else 0
                scale = 70 if use_subpoints else 100
                pct = base + int(end / n_blocks * scale)
                progress_callback(pct, f"Estimated {end:,}/{n_blocks:,} blocks")

        logger.info(
            "Global estimation complete: %d blocks (subpoints=%s, "
            "n_sub=%d, block_range_ratio=%.3f)",
            n_blocks, use_subpoints, n_sub_eff, block_range_ratio,
        )


# -----------------------------------------------------------------------------
# Cross-validation and calibration
# -----------------------------------------------------------------------------

def regression_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """OLS slope of actual vs estimated: actual = slope * estimated + intercept.

    FIXED: Previous implementation computed regression through the origin
    (slope = sum(x*y) / sum(x^2)) which inflates slope when the mean is
    far from zero.  Now uses proper OLS: slope = cov(x,y) / var(x).
    This is the standard conditional bias diagnostic in geostatistics.
    """
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if x.size < 2:
        return float("nan")
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    var_x = float(np.dot(x_centered, x_centered))
    if var_x <= 1e-30:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / var_x)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def cv_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    me = float(np.mean(err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    slope = regression_slope(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    z = err / np.maximum(y_std, 1e-12)
    z_mean = float(np.mean(z))
    z_var = float(np.var(z))

    cover50 = float(np.mean(np.abs(z) <= 0.67448975))
    cover68 = float(np.mean(np.abs(z) <= 1.0))
    cover90 = float(np.mean(np.abs(z) <= 1.64485363))
    cover95 = float(np.mean(np.abs(z) <= 1.95996398))

    return {
        "ME": me,
        "MAE": mae,
        "RMSE": rmse,
        "SLOPE": slope,
        "R2": r2,
        "STD_RESID_MEAN": z_mean,
        "STD_RESID_VAR": z_var,
        "COVER_50": cover50,
        "COVER_68": cover68,
        "COVER_90": cover90,
        "COVER_95": cover95,
    }


def leave_one_out_cv(
    coords: ArrayLike,
    values: ArrayLike,
    anisotropy: Anisotropy,
    variogram: Optional[VariogramModel] = None,
    rbf_settings: Optional[RBFSettings] = None,
    neighbourhood: Optional[NeighbourhoodSettings] = None,
    block_settings: Optional[BlockSettings] = None,
    max_points: Optional[int] = None,
    *,
    rbf_kernel: Optional[RBFKernel] = None,
) -> Dict[str, float]:
    """Leave-one-out cross-validation using the RBF estimator.

    LIMITATION: This function validates on the full dataset globally.
    When domain-separated estimation is used, the CV should ideally
    validate within each domain separately.  For multi-domain deposits,
    rely on per-domain swath plots instead of CV R².
    """
    import warnings
    from scipy.linalg import (
        cho_factor, cho_solve, solve, LinAlgWarning, LinAlgError,
    )
    from scipy.spatial.distance import cdist

    coords = _as_2d_float(coords)
    values = _as_1d_float(values, "values")
    n = coords.shape[0]
    if max_points is not None:
        n = min(n, max_points)

    rbf_settings = rbf_settings or RBFSettings()
    if rbf_kernel is None:
        rbf_kernel = RBFKernel()

    # ── PRESS formula: one factorisation instead of N ──────────
    # LOO prediction: ŷ_{-i}(x_i) = y_i - e_i / (1 - h_ii)
    # where e_i = y_i - ŷ(x_i) is the full-system residual and
    # h_ii is the leverage (diagonal of the hat matrix).
    coords_sub = coords[:n]
    values_sub = values[:n]

    coords_t = anisotropy.transform(coords_sub)
    r_nn = cdist(coords_t, coords_t)
    Phi_raw = rbf_kernel.evaluate(r_nn)

    kernel_scale = max(float(np.mean(np.diag(Phi_raw))), 1e-12)
    diag_reg = max(
        rbf_kernel.nugget + rbf_settings.smoothing + rbf_settings.epsilon,
        kernel_scale * 1e-6,
    )
    Phi = Phi_raw + diag_reg * np.eye(n, dtype=float)

    # Centre drift basis — UTM-scale coords blow up rcond otherwise.
    _cv_centroid = np.mean(coords_sub, axis=0, keepdims=True)
    coords_sub_centred = coords_sub - _cv_centroid
    drift_mode = rbf_settings.drift
    P = drift_basis(coords_sub_centred, drift_mode)
    m = P.shape[1]
    nm = n + m

    A = np.zeros((nm, nm), dtype=float)
    A[:n, :n] = Phi
    if m > 0:
        A[:n, n:] = P
        A[n:, :n] = P.T

    rhs = np.concatenate([values_sub, np.zeros(m, dtype=float)])

    sol = None
    factor = lower = None
    for _attempt in range(3):
        try:
            factor, lower = cho_factor(A, lower=True, check_finite=False)
            sol = cho_solve((factor, lower), rhs, check_finite=False)
            break
        except LinAlgError:
            A[:n, :n] += (kernel_scale * 10.0 ** (_attempt - 3)) * np.eye(
                n, dtype=float,
            )
    if sol is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=LinAlgWarning)
                sol = solve(A, rhs, assume_a="sym")
            factor = lower = None
        except Exception:
            pass
    if sol is None:
        from scipy.linalg import lstsq as _lstsq
        sol, _, _, _ = _lstsq(A, rhs, lapack_driver="gelsd")
        factor = lower = None

    # Full-system predictions at sample points (RAW kernel, not regularized)
    B = np.zeros((nm, n), dtype=float)
    B[:n, :] = Phi_raw
    if m > 0:
        B[n:, :] = P.T
    fitted = sol @ B  # (n,)

    # Hat matrix diagonal: H = B' @ A^{-1} @ B_rhs, h_ii = B[:,i]' @ A^{-1}[:,i]
    # Solve A @ X = B (first n columns only = identity mapped through B)
    I_n = np.eye(nm, n, dtype=float)
    if factor is not None:
        A_inv_cols = cho_solve((factor, lower), I_n, check_finite=False)
    else:
        A_inv_cols = solve(A, I_n, assume_a="sym")
    # h_ii = dot(B[:,i], A_inv_cols[:,i]) for each sample point i
    h_diag = np.sum(B * A_inv_cols, axis=0)  # (n,)

    residuals = values_sub - fitted
    # LOO predictions via PRESS: e_i^{CV} = e_i / (1 - h_ii)
    denom = np.maximum(1.0 - h_diag, 1e-12)
    pred = values_sub - residuals / denom
    # Approximate LOO std from leverage
    resid_var = float(np.mean(residuals ** 2))
    pred_std = np.sqrt(np.maximum(resid_var / denom, 1e-18))

    result = cv_metrics(values_sub, pred, pred_std)
    result["actual"] = values_sub.tolist()
    result["estimated"] = pred.tolist()
    result["pred_std"] = pred_std.tolist()
    result["slope_of_regression"] = result.get("SLOPE", 1.0)
    return result


# -----------------------------------------------------------------------------
# Example synthetic workflow
# -----------------------------------------------------------------------------

def make_synthetic_geology(
    n_samples: int = 400,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords = rng.uniform([0, 0, 0], [400, 300, 120], size=(n_samples, 3))

    # Smooth latent field with anisotropic trend and skewed uplift.
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    trend = 0.003 * x + 0.002 * y - 0.004 * z
    centres = np.array([[100, 80, 60], [300, 210, 40], [220, 120, 90]], dtype=float)
    amps = np.array([2.2, 1.5, 1.8], dtype=float)
    ranges = np.array([[80, 60, 25], [70, 90, 35], [90, 40, 20]], dtype=float)

    field = np.zeros(n_samples, dtype=float)
    for c, a, r in zip(centres, amps, ranges):
        d = ((coords - c) / r) ** 2
        field += a * np.exp(-0.5 * np.sum(d, axis=1))

    noise = rng.normal(0.0, 0.25, size=n_samples)
    raw = np.exp(trend + field + noise)  # strongly skewed

    # Prediction grid centres
    gx, gy, gz = np.meshgrid(
        np.linspace(20, 380, 12),
        np.linspace(20, 280, 10),
        np.linspace(10, 110, 6),
        indexing="ij",
    )
    blocks = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return coords, raw, blocks


def run_example() -> None:
    coords, values, blocks = make_synthetic_geology(n_samples=300, seed=7)

    anis = Anisotropy(
        ranges=(80.0, 45.0, 20.0),
        rotation_matrix=np.eye(3),
    )
    # RBF kernel (true RBF — not variogram covariance)
    kernel = RBFKernel(
        basis="wendland_c2",
        shape_parameter=1.0,
        support_radius=1.5,
        nugget=0.08,
    )
    rbf_settings = RBFSettings(
        drift="linear",
        smoothing=0.02,
        epsilon=1e-9,
        use_normal_scores=True,
        gh_order=20,
        basis_type="wendland_c2",
    )
    neighbourhood = NeighbourhoodSettings(
        n_min=20,
        n_start=32,
        n_max=56,
        max_per_octant=10,
        search_radius=None,
    )
    block_settings = BlockSettings(nx=3, ny=3, nz=3, dims=(20.0, 20.0, 10.0))
    partition = PartitionSettings(
        enabled=True,
        tile_size=(140.0, 120.0, 60.0),
        overlap_factor=2.0,
        stitch_eta=0.50,
        min_coverage_ratio=0.995,
    )

    estimator = FastRBFEstimator(
        coords=coords,
        values=values,
        anisotropy=anis,
        rbf_kernel=kernel,
        rbf_settings=rbf_settings,
        neighbourhood=neighbourhood,
        block_settings=block_settings,
        partition_settings=partition,
    )

    results = estimator.estimate_blocks(blocks)
    print("Estimated blocks:", len(blocks))
    print("Mean grade preview:", results["ARBF_GRADE"][:5])
    print("Block variance preview:", results["ARBF_VAR_BLOCK"][:5])
    print("Fail flags:", np.sum(results["ARBF_FAIL_FLAG"]))

    metrics = leave_one_out_cv(
        coords=coords,
        values=values,
        anisotropy=anis,
        rbf_settings=rbf_settings,
        neighbourhood=neighbourhood,
        block_settings=block_settings,
        max_points=40,
        rbf_kernel=kernel,
    )
    print("\nLOO CV metrics (original units):")
    for k, v in metrics.items():
        print(f"{k:16s}: {v:.6f}")


if __name__ == "__main__":
    run_example()
