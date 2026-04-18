"""
Variography sub-package.

Experimental variogram computation, model fitting, and anisotropy rotation.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from .experimental import compute_experimental_variogram, ExperimentalVariogram
from .models import fit_variogram_model, VariogramModel
from .anisotropy import rotation_matrix_3d, apply_anisotropy

__all__ = [
    "compute_experimental_variogram",
    "ExperimentalVariogram",
    "fit_variogram_model",
    "VariogramModel",
    "rotation_matrix_3d",
    "apply_anisotropy",
]
