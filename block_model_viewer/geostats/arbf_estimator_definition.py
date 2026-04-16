"""
ARBF Estimator Definition — reusable estimator configuration.

Stores sample data + all engine parameters WITHOUT any block model geometry,
UI references, or renderer state. Can be reused across multiple block models
and active cell masks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .fastrbf_engine_v2 import (
    Anisotropy,
    BlockSettings,
    NeighbourhoodSettings,
    PartitionSettings,
    RBFSettings,
    VariogramModel,
)
from .arbf_modes import EstimationMode, STANDARD, get_mode

logger = logging.getLogger(__name__)


@dataclass
class ARBFEstimatorDefinition:
    """Immutable estimator specification. Reusable across block models."""

    sample_coords: np.ndarray         # (N, 3)
    sample_values: np.ndarray         # (N,)
    variogram: VariogramModel
    anisotropy: Anisotropy
    rbf_settings: RBFSettings
    neighbourhood: NeighbourhoodSettings
    block_settings: BlockSettings
    partition_settings: PartitionSettings
    mode: EstimationMode = field(default_factory=lambda: STANDARD)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.sample_coords.shape[0]

    @classmethod
    def from_adapter_config(
        cls,
        config: Dict[str, Any],
        coords: np.ndarray,
        values: np.ndarray,
        mode_name: str = "standard",
    ) -> "ARBFEstimatorDefinition":
        """Build from the controller's config dict, reusing arbf_adapter build methods.

        This avoids duplicating the 65-key config → dataclass mapping.
        """
        from .arbf_adapter import (
            ARBFEstimatorAdapter,
            variogram_v2_to_engine_params,
        )

        # Create a temporary adapter just to use its _build_* methods
        adapter = ARBFEstimatorAdapter(config)

        # Try v2 direct path first
        v2_params = adapter._try_v2_variogram_direct()
        if v2_params is not None:
            variogram = v2_params["variogram"]
            anisotropy = v2_params["anisotropy"]
            rbf_settings = v2_params["rbf_settings"]
        else:
            variogram = adapter._build_variogram()
            anisotropy = adapter._build_anisotropy()
            rbf_settings = adapter._build_rbf_settings()

        neighbourhood = adapter._build_neighbourhood()

        # Block settings from config
        adapter._block_sizes = np.array(config.get("block_sizes", [10.0, 10.0, 10.0]))
        block_settings = adapter._build_block_settings()

        # Partition settings
        adapter._coords = coords
        partition = adapter._build_partition_settings()

        mode = get_mode(mode_name)

        # Apply mode overrides
        if mode.max_neighbours < neighbourhood.n_max:
            neighbourhood = NeighbourhoodSettings(
                n_min=neighbourhood.n_min,
                n_start=neighbourhood.n_start,
                n_max=mode.max_neighbours,
                max_per_octant=neighbourhood.max_per_octant,
                search_radius=neighbourhood.search_radius,
            )

        n_per_axis = mode.subpoints_per_axis
        block_settings = BlockSettings(
            nx=n_per_axis,
            ny=n_per_axis,
            nz=n_per_axis,
            dims=block_settings.dims,
        )

        if not mode.partition_stitching:
            partition = PartitionSettings(enabled=False)

        return cls(
            sample_coords=np.asarray(coords, dtype=float),
            sample_values=np.asarray(values, dtype=float).ravel(),
            variogram=variogram,
            anisotropy=anisotropy,
            rbf_settings=rbf_settings,
            neighbourhood=neighbourhood,
            block_settings=block_settings,
            partition_settings=partition,
            mode=mode,
            metadata={
                "variable": config.get("variable", ""),
                "n_samples": len(coords),
                "kernel_type": variogram.model,
                "drift": rbf_settings.drift,
                "use_normal_scores": rbf_settings.use_normal_scores,
                "mode": mode.name,
            },
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "model": self.variogram.model,
            "nugget": self.variogram.nugget_micro,
            "partial_sill": self.variogram.partial_sill,
            "ranges": self.anisotropy.ranges,
            "drift": self.rbf_settings.drift,
            "normal_scores": self.rbf_settings.use_normal_scores,
            "max_neighbours": self.neighbourhood.n_max,
            "subpoints": self.block_settings.nx ** 3,
            "mode": self.mode.name,
        }
