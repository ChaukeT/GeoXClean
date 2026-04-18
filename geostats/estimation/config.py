"""
Pydantic v2 configuration schemas for FastRBF estimation.

All estimation parameters are validated before computation begins.
Follows Leapfrog Geo parameter conventions.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class KernelType(str, Enum):
    """Supported radial basis function kernels."""

    LINEAR = "linear"
    SPHEROIDAL = "spheroidal"
    SPHERICAL = "spherical"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    CUBIC = "cubic"
    GENERALISED_CAUCHY = "generalised_cauchy"


class DriftType(str, Enum):
    """Polynomial drift (trend) modes for the RBF system."""

    CONSTANT = "constant"
    LINEAR = "linear"
    NONE = "none"


class RBFConfig(BaseModel):
    """
    Validated configuration for FastRBF estimation.

    All physical parameters carry units in their descriptions.
    Pydantic enforces bounds at construction time so no invalid
    configuration can reach the solver.
    """

    # ── Kernel ────────────────────────────────────────────────
    kernel_type: KernelType = KernelType.SPHEROIDAL
    total_sill: float = Field(gt=0, description="Total sill (variance units)")
    nugget: float = Field(ge=0, description="Nugget effect")
    base_range: float = Field(gt=0, description="Range in coordinate units")
    alpha: Literal[3, 5, 7, 9] = 5  # Only for spheroidal / gen. cauchy

    # ── Drift ─────────────────────────────────────────────────
    drift: DriftType = DriftType.CONSTANT

    # ── Anisotropy ────────────────────────────────────────────
    azimuth: float = Field(default=0.0, ge=0.0, lt=360.0)
    dip: float = Field(default=0.0, ge=-90.0, le=90.0)
    pitch: float = Field(default=0.0, ge=-90.0, le=90.0)
    ratio_major: float = Field(default=1.0, gt=0.0)
    ratio_semi: float = Field(default=1.0, gt=0.0)
    ratio_minor: float = Field(default=1.0, gt=0.0)

    # ── Search neighbourhood ──────────────────────────────────
    search_max_samples: int = Field(default=24, ge=4, le=100)
    search_min_samples: int = Field(default=8, ge=1)
    search_max_per_octant: int = Field(default=4, ge=1)
    search_min_octants: int = Field(default=2, ge=1, le=8)

    # ── Solver ────────────────────────────────────────────────
    accuracy: Optional[float] = Field(default=None, gt=0)
    max_solver_iterations: int = Field(default=1000, ge=100)

    # ── Block discretisation ──────────────────────────────────
    discretisation_points: int = Field(default=4, ge=1, le=10)

    # ── Evaluation limits ─────────────────────────────────────
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    # ── Validators ────────────────────────────────────────────

    @field_validator("nugget")
    @classmethod
    def nugget_less_than_sill(cls, v: float, info) -> float:
        sill = info.data.get("total_sill")
        if sill is not None and v > sill:
            raise ValueError(f"Nugget ({v}) must not exceed total sill ({sill})")
        if sill is not None and v == sill and sill > 0:
            import warnings
            warnings.warn(
                f"Pure nugget model (nugget == total_sill == {v}). "
                "The kernel matrix will have no spatial structure — "
                "interpolation will produce spatially uniform estimates.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @field_validator("search_min_samples")
    @classmethod
    def min_less_than_max(cls, v: int, info) -> int:
        mx = info.data.get("search_max_samples")
        if mx is not None and v > mx:
            raise ValueError(f"min_samples ({v}) > max_samples ({mx})")
        return v

    @field_validator("drift")
    @classmethod
    def linear_kernel_needs_drift(cls, v: "DriftType", info) -> "DriftType":
        """Linear kernel is conditionally negative definite — requires drift."""
        kernel = info.data.get("kernel_type")
        if kernel == KernelType.LINEAR and v == DriftType.NONE:
            raise ValueError(
                "Linear kernel requires drift (constant or linear). "
                "Without polynomial drift terms the system matrix is singular."
            )
        return v


    # ── Derived properties ────────────────────────────────────

    @property
    def nugget_to_sill_ratio(self) -> float:
        """Nugget-to-sill ratio (0 = no nugget, 1 = pure nugget).

        Standard definition: nugget / total_sill.  Since total_sill
        already includes the nugget component (total_sill = partial_sill
        + nugget), the denominator is just total_sill.
        """
        return self.nugget / self.total_sill if self.total_sill > 0 else 0.0

    @property
    def is_isotropic(self) -> bool:
        """True when all ellipsoid ratios are unity."""
        return (
            self.ratio_major == 1.0
            and self.ratio_semi == 1.0
            and self.ratio_minor == 1.0
        )

    @property
    def partial_sill(self) -> float:
        """Partial sill = total_sill - nugget (the structured component)."""
        return self.total_sill - self.nugget

    @property
    def search_radii(self) -> Tuple[float, float, float]:
        """Effective search radii (major, semi, minor) in real units."""
        return (
            self.base_range * self.ratio_major,
            self.base_range * self.ratio_semi,
            self.base_range * self.ratio_minor,
        )

    @property
    def search_angles(self) -> Tuple[float, float, float]:
        """Search ellipsoid rotation angles (azimuth, dip, pitch) in degrees."""
        return (self.azimuth, self.dip, self.pitch)
