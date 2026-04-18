"""
Variogram model fitting.

Fits theoretical variogram models to experimental variogram data
using weighted least squares with scipy.optimize.

Supported models: Spheroidal, Linear, Spherical, Gaussian,
Exponential, Cubic, Generalised Cauchy.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ..estimation.config import KernelType
from ..estimation.interpolant_functions import evaluate_kernel
from .experimental import ExperimentalVariogram

logger = logging.getLogger(__name__)


@dataclass
class VariogramModel:
    """Fitted variogram model parameters."""

    model_type: KernelType
    sill: float
    nugget: float
    range_: float
    alpha: int  # for spheroidal / gen. cauchy
    fit_residual: float
    fit_method: str
    experimental: ExperimentalVariogram


def fit_variogram_model(
    experimental: ExperimentalVariogram,
    model_type: KernelType = KernelType.SPHEROIDAL,
    alpha: int = 5,
    use_nugget: bool = True,
    fit_method: Literal["least_squares", "weighted_least_squares"] = "weighted_least_squares",
) -> VariogramModel:
    """
    Fit a variogram model to experimental data.

    Weighted least squares:  weight = N(h) / h²  (more weight on short lags).
    Uses scipy.optimize.minimize with L-BFGS-B and bounded parameters.

    Parameters
    ----------
    experimental : ExperimentalVariogram
    model_type : KernelType
    alpha : int
        Alpha for spheroidal / generalised Cauchy kernels.
    use_nugget : bool
        If False, nugget is fixed at 0.
    fit_method : str
        'least_squares' or 'weighted_least_squares'

    Returns
    -------
    VariogramModel
    """
    lags = experimental.lags
    gamma_exp = experimental.semivariance
    counts = experimental.pair_counts

    # Filter out empty bins
    valid = counts > 0
    lags_v = lags[valid]
    gamma_v = gamma_exp[valid]
    counts_v = counts[valid]

    if len(lags_v) < 3:
        raise ValueError("Need at least 3 non-empty lag bins for fitting")

    # Weights
    if fit_method == "weighted_least_squares":
        weights = counts_v.astype(np.float64) / (lags_v ** 2 + 1e-10)
    else:
        weights = np.ones_like(lags_v, dtype=np.float64)
    weights /= weights.sum()

    # Initial guesses
    sill_init = float(gamma_v.max())
    range_init = float(lags_v[len(lags_v) // 2])
    nugget_init = float(gamma_v[0] * 0.1) if use_nugget else 0.0

    def objective(params):
        sill_p, range_p = params[0], params[1]
        nugget_p = params[2] if use_nugget else 0.0

        gamma_model = evaluate_kernel(
            lags_v,
            sill=sill_p,
            range_=range_p,
            nugget=nugget_p,
            kernel_type=model_type,
            alpha=alpha,
        )
        residuals = (gamma_v - gamma_model) ** 2
        return float(np.sum(weights * residuals))

    # Bounds
    if use_nugget:
        x0 = [sill_init, range_init, nugget_init]
        bounds = [(1e-10, sill_init * 5), (1e-5, lags_v.max() * 3), (0, sill_init)]
    else:
        x0 = [sill_init, range_init]
        bounds = [(1e-10, sill_init * 5), (1e-5, lags_v.max() * 3)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    fitted_sill = result.x[0]
    fitted_range = result.x[1]
    fitted_nugget = result.x[2] if use_nugget else 0.0

    logger.info(
        "Variogram fit: %s, sill=%.4f, range=%.2f, nugget=%.4f, residual=%.6f",
        model_type.value,
        fitted_sill,
        fitted_range,
        fitted_nugget,
        result.fun,
    )

    return VariogramModel(
        model_type=model_type,
        sill=fitted_sill,
        nugget=fitted_nugget,
        range_=fitted_range,
        alpha=alpha,
        fit_residual=float(result.fun),
        fit_method=fit_method,
        experimental=experimental,
    )
