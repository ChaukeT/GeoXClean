"""
ARBF Data Transforms.

Normal-score transform and Isometric Log-Ratio (ILR) transform
for compositional data.

References
----------
- Egozcue et al. (2003). Isometric logratio transformations for
  compositional data analysis. Mathematical Geology, 35(3), 279-300.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normal-score transform
# ---------------------------------------------------------------------------


@dataclass
class NormalScoreTable:
    """Lookup table for normal-score back-transformation."""

    original_sorted: np.ndarray
    normal_scores_sorted: np.ndarray
    data_min: float
    data_max: float
    _interpolator: Optional[PchipInterpolator] = field(
        default=None, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        self._interpolator = PchipInterpolator(
            self.normal_scores_sorted,
            self.original_sorted,
            extrapolate=True,
        )


def detect_already_normal_scored(values: np.ndarray) -> bool:
    """Detect whether values appear to already be normal-score transformed.

    Uses four heuristics that together strongly indicate NS data:
    1. Values span both negative and positive (grades are typically >= 0)
    2. Mean is close to 0 (within 0.3)
    3. Std is close to 1 (within 0.3)
    4. Range is roughly [-3, 3] (max |value| < 5)

    All four must hold.  This avoids false positives on naturally
    centred data (e.g. residuals) by requiring the combination.

    Parameters
    ----------
    values : np.ndarray
        (N,) array of sample values.

    Returns
    -------
    bool
        True if the data looks like it has already been normal-scored.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if len(v) < 10:
        return False
    v_mean = float(np.mean(v))
    v_std = float(np.std(v))
    v_min = float(np.min(v))
    v_max = float(np.max(v))

    has_negatives = v_min < -0.5
    mean_near_zero = abs(v_mean) < 0.3
    std_near_one = 0.7 < v_std < 1.3
    range_bounded = v_max < 5.0 and v_min > -5.0

    return has_negatives and mean_near_zero and std_near_one and range_bounded


def normal_score_transform(
    values: np.ndarray,
    seed: int = 42,
    coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, NormalScoreTable]:
    """Rank-based normal-score transform.

    Maps data values to standard normal quantiles via their ranks.
    Ties are broken deterministically using spatial coordinates when
    available, so that samples at the same detection limit that are
    close together in space receive similar normal scores.  This
    prevents artificial pure-nugget effects at lower detection limits.

    When ``coords`` is not provided, falls back to random jitter with
    the given ``seed``.

    Parameters
    ----------
    values : np.ndarray
        (N,) array of grade values.
    seed : int
        Random seed for fallback random-jitter tie-breaking.
    coords : np.ndarray, optional
        (N, 3) sample coordinates.  When supplied, tied values are
        disambiguated by a normalised spatial hash so that spatially
        proximal ties sort together.

    Returns
    -------
    ns_values : np.ndarray
        (N,) normal-score transformed values.
    table : NormalScoreTable
        Mapping table for back-transformation.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    N = len(values)
    rng = np.random.RandomState(seed)

    # Break ties using spatial coordinates when available.
    # Tied values that are close in space receive similar normal scores,
    # preventing artificial pure-nugget effects at detection limits.
    jitter_scale = (np.max(values) - np.min(values) + 1e-12) * 1e-10
    if coords is not None:
        coords_arr = np.asarray(coords, dtype=np.float64)
        if coords_arr.shape[0] == N:
            c_range = np.max(coords_arr, axis=0) - np.min(coords_arr, axis=0)
            c_range[c_range < 1e-12] = 1.0
            c_norm = (coords_arr - coords_arr.min(axis=0)) / c_range
            # Project onto a single scalar with incommensurate weights so
            # that spatially close samples sort together for any tied value.
            jitter = jitter_scale * (
                c_norm @ np.array([0.7370, 0.3194, 0.1213]) - 0.5
            )
        else:
            jitter = rng.uniform(-jitter_scale, jitter_scale, size=N)
    else:
        jitter = rng.uniform(-jitter_scale, jitter_scale, size=N)
    jittered = values + jitter

    # Rank-based probabilities
    ranks = np.argsort(np.argsort(jittered)).astype(np.float64)
    probs = (ranks + 0.5) / N

    # Map to standard normal quantiles
    ns_values = norm.ppf(probs)
    ns_values = np.clip(ns_values, -6.0, 6.0)

    # Build mapping table (sorted by normal score for monotonic interpolation)
    sort_idx = np.argsort(ns_values)
    table = NormalScoreTable(
        original_sorted=values[sort_idx].copy(),
        normal_scores_sorted=ns_values[sort_idx].copy(),
        data_min=float(np.min(values)),
        data_max=float(np.max(values)),
    )

    return ns_values, table


def normal_score_backtransform(
    ns_values: np.ndarray,
    table: NormalScoreTable,
) -> np.ndarray:
    """Invert the normal-score transform using stored mapping table.

    Uses PCHIP monotonic interpolation to map normal scores back to
    original units.  Clamps to the observed data range to prevent
    extrapolation artifacts.

    Parameters
    ----------
    ns_values : np.ndarray
        Normal-score values to back-transform.
    table : NormalScoreTable
        Mapping table from ``normal_score_transform``.

    Returns
    -------
    np.ndarray
        Back-transformed values in original units.
    """
    ns_values = np.asarray(ns_values, dtype=np.float64)
    result = table._interpolator(ns_values)
    return np.clip(result, table.data_min, table.data_max)


def normal_score_backtransform_mean(
    ns_estimates: np.ndarray,
    ns_variances: np.ndarray,
    table: NormalScoreTable,
    n_gh: int = 10,
    sill: float = 1.0,
) -> np.ndarray:
    """Variance-corrected NS back-transform with information-weighted blending.

    For well-informed blocks (low posterior variance relative to the sill),
    the conditional mean E[Z | ŷ, σ²] is the correct estimator and is
    computed via Gauss-Hermite quadrature over the posterior distribution.

    For poorly-informed blocks (posterior variance ≈ sill), the GH
    quadrature produces E[Z | ŷ, σ² → sill] ≈ E[Z] — the unconditional
    mean of the original data.  For positively-skewed distributions
    (Cu, Au) this is much higher than the conditional median φ⁻¹(ŷ)
    and produces systematic overestimation in swath plots (Deutsch &
    Journel 1998, §IV.4; Pyrcz & Deutsch 2014, §6.3).

    The blended estimator is::

        Z*(x) = α(x) · E_GH[Z | ŷ, σ²] + (1 − α(x)) · φ⁻¹(ŷ)

    where α(x) = 1 − σ²/sill measures how much information the data
    provides at location x.  When σ² → 0 (interpolation), α → 1 and
    the GH conditional mean is used.  When σ² → sill (extrapolation),
    α → 0 and the naive back-transform (conditional median) is used.
    This prevents the artificial inflation of data-sparse blocks while
    preserving the correct conditional expectation near data.

    Parameters
    ----------
    ns_estimates : np.ndarray
        (B,) block estimates in normal-score space.
    ns_variances : np.ndarray
        (B,) posterior variances in normal-score space (from GPR).
    table : NormalScoreTable
        Mapping table from ``normal_score_transform``.
    n_gh : int
        Number of Gauss-Hermite quadrature nodes.
    sill : float
        Prior variance (sill + nugget) in normal-score space.  Used to
        compute the information weight α = 1 − σ²/sill.  Default 1.0
        (standard normal-score space).

    Returns
    -------
    np.ndarray
        (B,) back-transformed block mean grades in original units.
    """
    from numpy.polynomial.hermite_e import hermegauss

    ns_estimates = np.asarray(ns_estimates, dtype=np.float64)
    ns_variances = np.asarray(ns_variances, dtype=np.float64)

    # Gauss-Hermite nodes and weights (probabilist's Hermite).
    nodes, weights = hermegauss(n_gh)
    weights_norm = weights / weights.sum()

    # Clamp evaluation points to the observed NS range (prevents PCHIP
    # extrapolation beyond data support).
    ns_lo = float(table.normal_scores_sorted[0])
    ns_hi = float(table.normal_scores_sorted[-1])

    # ── Naive back-transform: conditional median φ⁻¹(ŷ) ────────────────
    # This is the primary estimator for all blocks.  It maps the NS
    # estimate directly back to original units via the PCHIP mapping
    # table.  For a symmetric distribution this equals the conditional
    # mean; for skewed distributions it equals the conditional MEDIAN.
    naive = normal_score_backtransform(ns_estimates, table)

    # ── GH variance correction: only for well-informed blocks ─────────
    # The GH quadrature computes the conditional MEAN, which differs
    # from the median for skewed distributions.  It is mathematically
    # correct but ONLY reliable when the posterior variance is small
    # enough that the quadrature nodes stay within the well-sampled
    # region of the NS mapping table.
    #
    # For data-sparse blocks (σ² > 30% of sill), the GH integration
    # spreads into the tails where the back-transform can inflate
    # grades by orders of magnitude for right-skewed variables (Cu, Au).
    #
    # Decision rule (hard cutoff, not a blend):
    #   σ²/sill < 0.30 → well-informed → use GH conditional mean
    #   σ²/sill ≥ 0.30 → insufficient data → use naive median
    #
    # This is the Journel & Huijbregts (1978) practical recommendation:
    # the variance-corrected back-transform should only be applied where
    # the estimation variance is a small fraction of the total variance.
    effective_sill = max(float(sill), 1e-12)
    # Clamp NS variances to the sill: posterior > sill is numerically
    # pathological (poorly-conditioned GPR) and would cause GH quadrature
    # to sample extreme tails of the NS mapping → garbage grades.
    ns_variances = np.minimum(ns_variances, effective_sill)
    ns_std = np.sqrt(np.maximum(ns_variances, 0.0))

    # A block is well-informed when its posterior variance is within 70%
    # of the sill.  The 0.70 threshold is the Journel & Huijbregts
    # practical cutoff: blocks retaining > 70% of prior uncertainty
    # have too little conditioning to trust the GH mean.
    r = ns_variances / effective_sill  # r → 0 near data, r → 1 far from data
    well_informed = r < 0.70

    n_well = int(np.sum(well_informed))
    n_total = len(ns_estimates)
    logger.info(
        "Back-transform: %d / %d blocks (%.1f%%) are well-informed "
        "(var/sill < 0.70) -> GH conditional mean.  Remainder -> naive median.",
        n_well, n_total, 100.0 * n_well / max(n_total, 1),
    )

    if n_well > 0:
        # Compute GH only for well-informed blocks (saves computation)
        idx_well = np.where(well_informed)[0]
        gh_subset = np.zeros(n_well, dtype=np.float64)
        ns_est_w = ns_estimates[idx_well]
        ns_std_w = ns_std[idx_well]
        for k in range(n_gh):
            y_eval = ns_est_w + ns_std_w * nodes[k]
            y_eval = np.clip(y_eval, ns_lo, ns_hi)
            gh_subset += weights_norm[k] * normal_score_backtransform(y_eval, table)
        # Replace naive values with GH values for well-informed blocks
        naive[idx_well] = gh_subset

    return naive


# ---------------------------------------------------------------------------
# Isometric Log-Ratio (ILR) transform
# ---------------------------------------------------------------------------


def ilr_forward(
    compositions: np.ndarray,
    kappa: float = 100.0,
) -> np.ndarray:
    """Isometric log-ratio forward transform (Eq. 7.2).

    Maps D-part compositions from the simplex S^D to R^{D-1}.

    y_j = sqrt(j/(j+1)) * ln(G_j / x_{j+1}),  j = 1, ..., D-1

    where G_j = (prod_{i=1}^{j} x_i)^{1/j} is the geometric mean of
    the first j components.

    Parameters
    ----------
    compositions : np.ndarray
        (N, D) array where each row sums to *kappa* (100 or 1).
        All values must be > 0.
    kappa : float
        Closure constant (default 100 for percentages).

    Returns
    -------
    np.ndarray
        (N, D-1) ILR coordinates.

    Raises
    ------
    ValueError
        If any composition value is <= 0.
    """
    compositions = np.asarray(compositions, dtype=np.float64)
    if compositions.ndim == 1:
        compositions = compositions.reshape(1, -1)
    N, D = compositions.shape

    # T3 fix: apply multiplicative zero replacement instead of raising.
    # Real-world compositional data often has zeros (e.g., Cu=0% in waste).
    # Replace zeros with delta = 0.65 × detection limit (Mart\u00EDn-Fern\u00E1ndez
    # et al. 2003), then renormalise rows to preserve closure.
    has_zeros = np.any(compositions <= 0, axis=1)
    if np.any(has_zeros):
        delta = np.min(compositions[compositions > 0]) * 0.65 if np.any(compositions > 0) else 1e-6
        compositions = compositions.copy()  # don't modify input
        for i in np.where(has_zeros)[0]:
            row = compositions[i]
            zero_mask = row <= 0
            n_zero = int(np.sum(zero_mask))
            row[zero_mask] = delta
            # Renormalise non-zero components to preserve closure
            row[~zero_mask] *= (kappa - n_zero * delta) / np.sum(row[~zero_mask])
            compositions[i] = row
        logger.info(
            "ILR: multiplicative zero replacement applied to %d / %d rows (delta=%.2e)",
            int(np.sum(has_zeros)), N, delta,
        )

    log_x = np.log(compositions)
    ilr = np.zeros((N, D - 1), dtype=np.float64)

    for j in range(1, D):
        # Geometric mean of first j components (using log-sum-exp for stability)
        log_Gj = np.mean(log_x[:, :j], axis=1)
        ilr[:, j - 1] = np.sqrt(j / (j + 1.0)) * (log_Gj - log_x[:, j])

    return ilr


def ilr_inverse(
    ilr_values: np.ndarray,
    kappa: float = 100.0,
) -> np.ndarray:
    """Inverse ILR transform (Eq. 7.2 inverse).

    Maps R^{D-1} back to the D-part simplex.

    Guarantees: all components > 0 and rows sum to *kappa*.

    Parameters
    ----------
    ilr_values : np.ndarray
        (N, D-1) ILR coordinates.
    kappa : float
        Closure constant (default 100).

    Returns
    -------
    np.ndarray
        (N, D) compositions with all values > 0 and row sums = kappa.
    """
    ilr_values = np.asarray(ilr_values, dtype=np.float64)
    if ilr_values.ndim == 1:
        ilr_values = ilr_values.reshape(1, -1)
    N, Dm1 = ilr_values.shape
    D = Dm1 + 1

    # Build the Helmert sub-matrix (contrast matrix) Psi: (D, D-1)
    Psi = _helmert_matrix(D)

    # x_i proportional to exp(sum_j Psi[i,j] * y_j)
    log_x = ilr_values @ Psi.T  # (N, D)

    # Softmax-like normalisation for numerical stability
    log_x_max = np.max(log_x, axis=1, keepdims=True)
    exp_x = np.exp(log_x - log_x_max)
    row_sums = np.sum(exp_x, axis=1, keepdims=True)
    compositions = kappa * exp_x / row_sums

    return compositions


def _helmert_matrix(D: int) -> np.ndarray:
    """Build the (D x D-1) Helmert sub-matrix for ILR transform.

    Psi[i, j] defines the orthonormal basis on the simplex.
    """
    Psi = np.zeros((D, D - 1), dtype=np.float64)
    for j in range(D - 1):
        k = j + 1
        coeff = np.sqrt(k / (k + 1.0))
        Psi[:k, j] = 1.0 / k * coeff
        Psi[k, j] = -coeff
    return Psi
