"""
Estimation diagnostics for JORC Table 1 Section 3 compliance.

Provides:
  - Slope of regression (conditional bias)
  - Global bias check
  - Swath plot data
  - QQ plot data
  - Grade-tonnage curves
  - Kriging Neighbourhood Analysis (KNA)

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SlopeResult:
    """Slope of regression of actual on estimated."""

    slope: float
    intercept: float
    r_squared: float
    is_conditionally_biased: bool  # True if slope deviates significantly from 1
    message: str


@dataclass
class BiasResult:
    """Global bias check result."""

    block_mean: float
    composite_mean: float
    bias_percent: float
    is_flagged: bool  # True if > 5%
    message: str


@dataclass
class SwathData:
    """Swath plot data for one axis."""

    bin_centers: NDArray[np.float64]
    estimate_means: NDArray[np.float64]
    composite_means: NDArray[np.float64]
    bin_counts: NDArray[np.int32]
    axis: str


@dataclass
class QQData:
    """Quantile-quantile comparison data."""

    estimate_quantiles: NDArray[np.float64]
    composite_quantiles: NDArray[np.float64]
    quantile_levels: NDArray[np.float64]


@dataclass
class GradeTonnageCurve:
    """Grade-tonnage data for a range of cut-off grades."""

    cutoffs: NDArray[np.float64]
    tonnes_above: NDArray[np.float64]
    mean_grade_above: NDArray[np.float64]
    metal_above: NDArray[np.float64]


@dataclass
class KNAResult:
    """Kriging Neighbourhood Analysis result."""

    parameter_sets: list[dict]
    slopes: NDArray[np.float64]
    rmses: NDArray[np.float64]
    best_index: int
    best_params: dict


class EstimationDiagnostics:
    """
    JORC Table 1 Section 3 diagnostic checks for estimation quality.
    """

    def slope_of_regression(
        self,
        actual: NDArray[np.float64],
        estimated: NDArray[np.float64],
    ) -> SlopeResult:
        """
        Slope of regression of actual on estimated.

        A perfect model has slope = 1.0.  Significant deviation
        indicates conditional bias.  JORC requires this check.

        Parameters
        ----------
        actual : (N,) ndarray
        estimated : (N,) ndarray

        Returns
        -------
        SlopeResult
        """
        actual = np.asarray(actual, dtype=np.float64)
        estimated = np.asarray(estimated, dtype=np.float64)

        # Ordinary least squares: actual = slope * estimated + intercept
        A = np.column_stack([estimated, np.ones_like(estimated)])
        result = np.linalg.lstsq(A, actual, rcond=None)
        coeffs = result[0]
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        # R²
        predicted = slope * estimated + intercept
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        biased = abs(slope - 1.0) > 0.1
        msg = (
            f"Slope={slope:.3f} (target=1.0). "
            + ("CONDITIONAL BIAS DETECTED." if biased else "Within acceptable range.")
        )

        return SlopeResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_sq,
            is_conditionally_biased=biased,
            message=msg,
        )

    def global_bias_check(
        self,
        block_estimates: NDArray[np.float64],
        composite_mean: float,
        block_volumes: Optional[NDArray[np.float64]] = None,
    ) -> BiasResult:
        """
        Compare volume-weighted block mean vs declustered composite mean.

        A difference > 5% is a red flag for JORC reporting.

        Parameters
        ----------
        block_estimates : (B,) ndarray
        composite_mean : float
        block_volumes : (B,) ndarray, optional
            For volume-weighted mean.  If None, equal weighting.

        Returns
        -------
        BiasResult
        """
        block_estimates = np.asarray(block_estimates, dtype=np.float64)
        valid = ~np.isnan(block_estimates)

        if block_volumes is not None:
            block_volumes = np.asarray(block_volumes, dtype=np.float64)
            block_mean = float(
                np.average(block_estimates[valid], weights=block_volumes[valid])
            )
        else:
            block_mean = float(np.nanmean(block_estimates))

        if composite_mean != 0:
            bias_pct = abs(block_mean - composite_mean) / abs(composite_mean) * 100.0
        else:
            bias_pct = 0.0 if block_mean == 0 else float("inf")

        flagged = bias_pct > 5.0
        msg = (
            f"Block mean={block_mean:.4f}, Composite mean={composite_mean:.4f}, "
            f"Bias={bias_pct:.1f}%. "
            + ("RED FLAG: bias exceeds 5%." if flagged else "Acceptable.")
        )

        return BiasResult(
            block_mean=block_mean,
            composite_mean=composite_mean,
            bias_percent=float(bias_pct),
            is_flagged=flagged,
            message=msg,
        )

    def swath_plot_data(
        self,
        block_centroids: NDArray[np.float64],
        block_estimates: NDArray[np.float64],
        composite_points: Optional[NDArray[np.float64]] = None,
        composite_values: Optional[NDArray[np.float64]] = None,
        axis: str = "x",
        n_bins: int = 20,
    ) -> SwathData:
        """
        Generate swath plot data (mean grade by spatial slice).

        Shows spatial distribution of bias.  JORC requires visual
        inspection of estimation vs nearest-neighbour.

        Parameters
        ----------
        block_centroids : (B, 3) ndarray
        block_estimates : (B,) ndarray
        composite_points : (N, 3) ndarray, optional
        composite_values : (N,) ndarray, optional
        axis : str
            'x', 'y', or 'z'
        n_bins : int

        Returns
        -------
        SwathData
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        ax_idx = axis_map.get(axis.lower(), 0)

        block_centroids = np.asarray(block_centroids, dtype=np.float64)
        block_estimates = np.asarray(block_estimates, dtype=np.float64)

        valid = ~np.isnan(block_estimates)
        coords = block_centroids[valid, ax_idx]
        ests = block_estimates[valid]

        bins = np.linspace(coords.min(), coords.max(), n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        est_means = np.zeros(n_bins, dtype=np.float64)
        comp_means = np.zeros(n_bins, dtype=np.float64)
        counts = np.zeros(n_bins, dtype=np.int32)

        digitized = np.digitize(coords, bins) - 1
        digitized = np.clip(digitized, 0, n_bins - 1)

        for b in range(n_bins):
            mask = digitized == b
            if mask.sum() > 0:
                est_means[b] = ests[mask].mean()
                counts[b] = mask.sum()

        if composite_points is not None and composite_values is not None:
            comp_coords = np.asarray(composite_points, dtype=np.float64)[:, ax_idx]
            comp_vals = np.asarray(composite_values, dtype=np.float64)
            comp_dig = np.digitize(comp_coords, bins) - 1
            comp_dig = np.clip(comp_dig, 0, n_bins - 1)
            for b in range(n_bins):
                mask = comp_dig == b
                if mask.sum() > 0:
                    comp_means[b] = comp_vals[mask].mean()

        return SwathData(
            bin_centers=bin_centers,
            estimate_means=est_means,
            composite_means=comp_means,
            bin_counts=counts,
            axis=axis,
        )

    def qq_plot_data(
        self,
        estimates: NDArray[np.float64],
        composites: NDArray[np.float64],
        n_quantiles: int = 100,
    ) -> QQData:
        """
        Quantile-quantile comparison of estimate vs composite distributions.

        Parameters
        ----------
        estimates : (B,) ndarray
        composites : (N,) ndarray
        n_quantiles : int

        Returns
        -------
        QQData
        """
        levels = np.linspace(0, 100, n_quantiles)
        est_q = np.percentile(estimates[~np.isnan(estimates)], levels)
        comp_q = np.percentile(composites, levels)

        return QQData(
            estimate_quantiles=est_q,
            composite_quantiles=comp_q,
            quantile_levels=levels / 100.0,
        )

    def grade_tonnage_curve(
        self,
        estimates: NDArray[np.float64],
        tonnages: NDArray[np.float64],
        cutoffs: Optional[NDArray[np.float64]] = None,
        n_cutoffs: int = 50,
    ) -> GradeTonnageCurve:
        """
        Generate grade-tonnage data for a range of cut-off grades.

        Required for JORC resource reporting.

        Parameters
        ----------
        estimates : (B,) ndarray
            Block grade estimates.
        tonnages : (B,) ndarray
            Block tonnages (or volumes × density).
        cutoffs : ndarray, optional
            Explicit cut-off grades.  If None, auto-generated.
        n_cutoffs : int
            Number of auto-generated cutoffs.

        Returns
        -------
        GradeTonnageCurve
        """
        estimates = np.asarray(estimates, dtype=np.float64)
        tonnages = np.asarray(tonnages, dtype=np.float64)

        valid = ~np.isnan(estimates)
        est = estimates[valid]
        ton = tonnages[valid]

        if cutoffs is None:
            cutoffs = np.linspace(est.min(), est.max(), n_cutoffs)
        else:
            cutoffs = np.asarray(cutoffs, dtype=np.float64)

        tonnes_above = np.empty(len(cutoffs))
        mean_above = np.empty(len(cutoffs))
        metal_above = np.empty(len(cutoffs))

        for i, co in enumerate(cutoffs):
            mask = est >= co
            if mask.sum() > 0:
                tonnes_above[i] = ton[mask].sum()
                mean_above[i] = np.average(est[mask], weights=ton[mask])
                metal_above[i] = (est[mask] * ton[mask]).sum()
            else:
                tonnes_above[i] = 0.0
                mean_above[i] = 0.0
                metal_above[i] = 0.0

        return GradeTonnageCurve(
            cutoffs=cutoffs,
            tonnes_above=tonnes_above,
            mean_grade_above=mean_above,
            metal_above=metal_above,
        )

    def kriging_neighbourhood_analysis(
        self,
        points: NDArray[np.float64],
        values: NDArray[np.float64],
        base_config: "RBFConfig",
        min_samples_range: tuple[int, int] = (4, 16),
        max_samples_range: tuple[int, int] = (12, 48),
        discretisation_range: tuple[int, int] = (2, 6),
    ) -> KNAResult:
        """
        KNA: Test combinations of search parameters to find optimal config.

        Varies min/max samples and discretisation points, runs LOO-CV
        for each combination, and reports slope of regression and RMSE.

        Parameters
        ----------
        points : (N, 3)
        values : (N,)
        base_config : RBFConfig
        min_samples_range : (lo, hi)
        max_samples_range : (lo, hi)
        discretisation_range : (lo, hi)

        Returns
        -------
        KNAResult
        """
        from .cross_validation import loo_cross_validation

        param_sets = []
        min_vals = range(min_samples_range[0], min_samples_range[1] + 1, 4)
        max_vals = range(max_samples_range[0], max_samples_range[1] + 1, 12)
        disc_vals = range(discretisation_range[0], discretisation_range[1] + 1, 2)

        for mn in min_vals:
            for mx in max_vals:
                if mn > mx:
                    continue
                for disc in disc_vals:
                    param_sets.append(
                        {
                            "search_min_samples": mn,
                            "search_max_samples": mx,
                            "discretisation_points": disc,
                        }
                    )

        logger.info("KNA: testing %d parameter combinations", len(param_sets))

        slopes = np.empty(len(param_sets))
        rmses = np.empty(len(param_sets))

        for i, ps in enumerate(param_sets):
            cfg = base_config.model_copy(update=ps)
            try:
                cv = loo_cross_validation(points, values, cfg)
                sr = self.slope_of_regression(cv.actual, cv.estimated)
                slopes[i] = sr.slope
                rmses[i] = cv.rmse
            except Exception as exc:
                logger.debug("KNA combo %d failed: %s", i, exc)
                slopes[i] = np.nan
                rmses[i] = np.nan

        # Best = closest slope to 1.0 with lowest RMSE.
        # Slope deviation is weighted 2× more than normalised RMSE
        # because conditional bias (slope != 1) is a more serious
        # JORC concern than global error magnitude.
        valid = ~(np.isnan(slopes) | np.isnan(rmses))
        if valid.any():
            norm_rmse = rmses / (rmses[valid].max() + 1e-10)
            score = 2.0 * np.abs(slopes - 1.0) + norm_rmse
            score[~valid] = np.inf
            best_idx = int(np.argmin(score))
        else:
            best_idx = 0

        return KNAResult(
            parameter_sets=param_sets,
            slopes=slopes,
            rmses=rmses,
            best_index=best_idx,
            best_params=param_sets[best_idx] if param_sets else {},
        )
