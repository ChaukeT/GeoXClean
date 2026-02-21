"""
Geotechnical and Stability Analysis Module

Provides rock-mass property modeling, stope stability analysis (Mathews Graph),
and slope risk assessment with probabilistic capabilities.
"""

from .dataclasses import (
    RockMassPoint,
    RockMassGrid,
    StopeStabilityInput,
    StopeStabilityResult,
    SlopeRiskInput,
    SlopeRiskResult,
    GeotechMCResult
)

from .rock_mass_model import RockMassModel
from .interpolation import interpolate_to_block_model
from .stope_stability import evaluate_stope, evaluate_stope_probabilistic
from .slope_risk import evaluate_slope
from .probabilistic_geotech import (
    run_stope_stability_monte_carlo,
    run_slope_risk_monte_carlo
)

__all__ = [
    'RockMassPoint',
    'RockMassGrid',
    'StopeStabilityInput',
    'StopeStabilityResult',
    'SlopeRiskInput',
    'SlopeRiskResult',
    'GeotechMCResult',
    'RockMassModel',
    'interpolate_to_block_model',
    'evaluate_stope',
    'evaluate_stope_probabilistic',
    'evaluate_slope',
    'run_stope_stability_monte_carlo',
    'run_slope_risk_monte_carlo',
]

