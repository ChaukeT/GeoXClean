"""
Block Model Definition — geometry-only lattice object.

This stores ONLY the target grid geometry:
  - origin, dimensions, block sizes
  - precomputed block centres
  - IJK indices

It does NOT contain sample data, variograms, estimation results,
or any renderer/UI references. It is reusable across multiple
estimation runs and active cell masks.

This is the SINGLE SOURCE OF TRUTH for block model geometry.
All estimation methods MUST use the shared definition from the
DataRegistry rather than creating their own grids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BlockModelDefinition:
    name: str
    origin: Tuple[float, float, float]        # (x0, y0, z0) — corner of first block
    dims: Tuple[int, int, int]                # (nx, ny, nz) — block counts
    block_size: Tuple[float, float, float]    # (dx, dy, dz) — block sizes
    centres: np.ndarray                       # (N, 3) precomputed
    ijk_indices: Optional[np.ndarray] = None  # (N, 3) integer indices
    rotation_matrix: Optional[np.ndarray] = None  # 3×3 rotation or None (axis-aligned)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return self.centres.shape[0]

    @property
    def nx(self) -> int:
        return self.dims[0]

    @property
    def ny(self) -> int:
        return self.dims[1]

    @property
    def nz(self) -> int:
        return self.dims[2]

    @property
    def is_regular(self) -> bool:
        """True if block count matches nx*ny*nz (unclipped regular grid)."""
        return self.n_blocks == self.dims[0] * self.dims[1] * self.dims[2]

    @property
    def dx(self) -> float:
        return self.block_size[0]

    @property
    def dy(self) -> float:
        return self.block_size[1]

    @property
    def dz(self) -> float:
        return self.block_size[2]

    @property
    def extents(self) -> Tuple[float, float, float, float, float, float]:
        """(xmin, xmax, ymin, ymax, zmin, zmax) — full model extent."""
        x0, y0, z0 = self.origin
        nx, ny, nz = self.dims
        dx, dy, dz = self.block_size
        return (x0, x0 + nx * dx, y0, y0 + ny * dy, z0, z0 + nz * dz)

    @property
    def memory_per_property_mb(self) -> float:
        """Approximate MB per float64 property on this grid."""
        return self.nx * self.ny * self.nz * 8 / (1024 * 1024)

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def centroids(self) -> np.ndarray:
        """Return (N, 3) float64 array of block centres (same as self.centres)."""
        return self.centres

    def to_image_data(self):
        """Create a PyVista ImageData grid from this definition.

        Returns an empty grid (no cell_data) — callers add properties.
        Raises ImportError if PyVista is not available.
        """
        import pyvista as pv

        if self.rotation_matrix is not None:
            raise ValueError(
                "Cannot create ImageData from a rotated block model definition. "
                "Use to_structured_grid() for rotated grids."
            )

        grid = pv.ImageData()
        nx, ny, nz = self.dims
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.origin = self.origin
        grid.spacing = self.block_size
        return grid

    def to_structured_grid(self):
        """Create a PyVista StructuredGrid from this definition.

        Fallback for rotated grids; wastes more memory than ImageData.
        """
        import pyvista as pv

        xs = self.centres[:, 0].reshape(self.dims)
        ys = self.centres[:, 1].reshape(self.dims)
        zs = self.centres[:, 2].reshape(self.dims)
        return pv.StructuredGrid(xs, ys, zs)

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask: which points fall inside the block model extent.

        Parameters
        ----------
        points : (N, 3) array

        Returns
        -------
        np.ndarray of bool, shape (N,)
        """
        points = np.asarray(points, dtype=float)
        xmin, xmax, ymin, ymax, zmin, zmax = self.extents
        return (
            (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
            (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
            (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
        )

    def to_grid_spec(self) -> Dict[str, Any]:
        """Convert to the legacy grid_spec dict used by ARBF and other panels."""
        return {
            "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "dx": self.dx, "dy": self.dy, "dz": self.dz,
            "x0": self.origin[0], "y0": self.origin[1], "z0": self.origin[2],
        }

    def to_sgsim_params(self) -> Dict[str, Any]:
        """Convert to the flat key dict used by SGSIM and simulation panels."""
        return {
            "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "xmin": self.origin[0], "ymin": self.origin[1], "zmin": self.origin[2],
            "xinc": self.dx, "yinc": self.dy, "zinc": self.dz,
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_params(
        cls,
        origin: Tuple[float, float, float],
        spacing: Tuple[float, float, float],
        dimensions: Tuple[int, int, int],
        name: str = "shared_block_model",
    ) -> "BlockModelDefinition":
        """Build from origin, spacing, and dimensions — the standard constructor.

        This is the preferred way to create a definition from the Define Block
        Model panel or from any code that specifies the grid explicitly.
        """
        x0, y0, z0 = map(float, origin)
        dx, dy, dz = map(float, spacing)
        nx, ny, nz = map(int, dimensions)

        xs = x0 + (np.arange(nx, dtype=float) + 0.5) * dx
        ys = y0 + (np.arange(ny, dtype=float) + 0.5) * dy
        zs = z0 + (np.arange(nz, dtype=float) + 0.5) * dz

        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        centres = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        ix, iy, iz = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij",
        )
        ijk = np.column_stack([ix.ravel(), iy.ravel(), iz.ravel()])

        return cls(
            name=name,
            origin=(x0, y0, z0),
            dims=(nx, ny, nz),
            block_size=(dx, dy, dz),
            centres=centres,
            ijk_indices=ijk,
            metadata={"source": "from_params"},
        )

    @classmethod
    def from_grid_spec(cls, grid_spec: Dict[str, Any], name: str = "arbf_grid") -> "BlockModelDefinition":
        """Build from the standard grid_spec dict used by the ARBF panel.

        Expected keys: nx, ny, nz, dx, dy, dz, x0, y0, z0
        """
        nx = int(grid_spec.get("nx", 50))
        ny = int(grid_spec.get("ny", 50))
        nz = int(grid_spec.get("nz", 25))
        dx = float(grid_spec.get("dx", 10.0))
        dy = float(grid_spec.get("dy", 10.0))
        dz = float(grid_spec.get("dz", 10.0))
        x0 = float(grid_spec.get("x0", 0.0))
        y0 = float(grid_spec.get("y0", 0.0))
        z0 = float(grid_spec.get("z0", 0.0))

        # Cell centres
        xs = x0 + (np.arange(nx, dtype=float) + 0.5) * dx
        ys = y0 + (np.arange(ny, dtype=float) + 0.5) * dy
        zs = z0 + (np.arange(nz, dtype=float) + 0.5) * dz

        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        centres = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        # IJK indices
        ix, iy, iz = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij",
        )
        ijk = np.column_stack([ix.ravel(), iy.ravel(), iz.ravel()])

        return cls(
            name=name,
            origin=(x0, y0, z0),
            dims=(nx, ny, nz),
            block_size=(dx, dy, dz),
            centres=centres,
            ijk_indices=ijk,
            metadata={"source": "grid_spec", "grid_spec": grid_spec},
        )

    @classmethod
    def from_centroids(
        cls,
        centroids: np.ndarray,
        block_size: Tuple[float, float, float],
        name: str = "arbf_grid",
    ) -> "BlockModelDefinition":
        """Build from an existing array of centroids (e.g. from block model import)."""
        centroids = np.asarray(centroids, dtype=float)
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError("centroids must be (N, 3)")

        # Infer dims from unique coordinates
        # FP-10 FIX: Use 10-decimal rounding (was 6) to avoid losing
        # precision in models with >6 significant digits (common in UTM).
        ux = np.unique(np.round(centroids[:, 0], 10))
        uy = np.unique(np.round(centroids[:, 1], 10))
        uz = np.unique(np.round(centroids[:, 2], 10))
        nx, ny, nz = len(ux), len(uy), len(uz)

        origin = (float(ux[0] - block_size[0] / 2) if nx > 0 else 0.0,
                  float(uy[0] - block_size[1] / 2) if ny > 0 else 0.0,
                  float(uz[0] - block_size[2] / 2) if nz > 0 else 0.0)

        return cls(
            name=name,
            origin=origin,
            dims=(nx, ny, nz),
            block_size=block_size,
            centres=centroids,
            metadata={"source": "centroids"},
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_blocks": self.n_blocks,
            "dims": self.dims,
            "block_size": self.block_size,
            "origin": self.origin,
            "is_regular": self.is_regular,
            "extents": self.extents,
            "memory_per_property_mb": round(self.memory_per_property_mb, 2),
        }

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty = valid)."""
        errors = []
        nx, ny, nz = self.dims
        dx, dy, dz = self.block_size
        if nx < 1 or ny < 1 or nz < 1:
            errors.append(f"Dimensions must be ≥ 1 (got {nx}×{ny}×{nz})")
        if dx <= 0 or dy <= 0 or dz <= 0:
            errors.append(f"Block sizes must be > 0 (got {dx}×{dy}×{dz})")
        total = nx * ny * nz
        if total > 5_000_000:
            errors.append(
                f"Block model has {total:,} blocks (hard limit: 5,000,000). "
                "Increase block size or reduce extent."
            )
        return errors

    def geometry_matches(self, other: "BlockModelDefinition") -> bool:
        """True if origin, dims, and block_size are identical."""
        return (
            self.origin == other.origin
            and self.dims == other.dims
            and self.block_size == other.block_size
        )

    def __repr__(self) -> str:
        nx, ny, nz = self.dims
        dx, dy, dz = self.block_size
        return (
            f"BlockModelDefinition('{self.name}', "
            f"{nx}×{ny}×{nz} @ {dx}×{dy}×{dz}m, "
            f"origin={self.origin})"
        )
