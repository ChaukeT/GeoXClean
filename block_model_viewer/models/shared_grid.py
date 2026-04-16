"""
Shared Grid Definition — Single Source of Truth for block model geometry.

All estimation and simulation panels write to and read from this object
via the DataRegistry.  This ensures that ARBF, Kriging, SGSIM, IK, etc.
always operate on the same grid and domain mask.

Usage:
    from ..models.shared_grid import SharedGridDefinition

    grid_def = SharedGridDefinition(
        nx=100, ny=80, nz=25,
        dx=10.0, dy=10.0, dz=5.0,
        x0=257800.0, y0=7922000.0, z0=-300.0,
    )
    registry.register_shared_grid(grid_def, source_panel="Kriging")

    # Any other panel can retrieve the same grid:
    grid_def = registry.get_shared_grid()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SharedGridDefinition:
    """Immutable specification of the block model grid.

    Coordinates follow the GeoX convention:
      x0, y0, z0  = origin (minimum corner of the first block)
      dx, dy, dz  = block dimensions (spacing)
      nx, ny, nz  = number of blocks along each axis
    """

    nx: int = 50
    ny: int = 50
    nz: int = 25
    dx: float = 10.0
    dy: float = 10.0
    dz: float = 5.0
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    # Optional: source panel that last updated this grid
    source_panel: str = ""

    @property
    def n_blocks(self) -> int:
        return self.nx * self.ny * self.nz

    def build_centroids(self) -> np.ndarray:
        """Return (n_blocks, 3) array of block centre coordinates."""
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        zs = self.z0 + (np.arange(self.nz) + 0.5) * self.dz
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    def to_grid_spec(self) -> Dict[str, Any]:
        """Convert to the grid_spec dict that controller methods expect."""
        return {
            "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "dx": self.dx, "dy": self.dy, "dz": self.dz,
            "x0": self.x0, "y0": self.y0, "z0": self.z0,
            "xmin": self.x0, "ymin": self.y0, "zmin": self.z0,
            "xinc": self.dx, "yinc": self.dy, "zinc": self.dz,
        }

    def __eq__(self, other):
        if not isinstance(other, SharedGridDefinition):
            return NotImplemented
        return (
            self.nx == other.nx and self.ny == other.ny and self.nz == other.nz
            and np.isclose(self.dx, other.dx)
            and np.isclose(self.dy, other.dy)
            and np.isclose(self.dz, other.dz)
            and np.isclose(self.x0, other.x0)
            and np.isclose(self.y0, other.y0)
            and np.isclose(self.z0, other.z0)
        )


@dataclass
class SharedDomainMask:
    """Shared domain mask computed once, used by all estimation methods.

    Stores both the boolean mask and the parameters used to compute it,
    so downstream methods can verify consistency.
    """

    mask: np.ndarray          # (n_blocks,) bool — True = informed
    method: str = "distance"  # "distance" or "convex_hull"
    n_informed: int = 0
    n_total: int = 0
    pct_informed: float = 0.0

    # Parameters used to compute the mask (for reproducibility)
    search_radii: tuple = (200.0, 200.0, 200.0)
    azimuth_deg: float = 0.0
    dip_deg: float = 0.0
    min_neighbours: int = 1
    buffer_m: float = 0.0

    source_panel: str = ""

    @property
    def active_indices(self) -> np.ndarray:
        """Integer indices of informed blocks."""
        return np.where(self.mask)[0]

    def to_metadata(self) -> Dict[str, Any]:
        """Return a serialisable metadata dict for audit."""
        return {
            "domain_mask_applied": True,
            "method": self.method,
            "n_informed": self.n_informed,
            "n_total": self.n_total,
            "pct_informed": round(self.pct_informed, 1),
            "search_radii": self.search_radii,
            "azimuth_deg": self.azimuth_deg,
            "source_panel": self.source_panel,
        }
