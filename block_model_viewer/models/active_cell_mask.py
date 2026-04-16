"""
Active Cell Mask — defines which blocks are eligible for estimation.

Supports:
  - Footprint clipping (KD-tree distance to nearest sample)
  - Domain filtering (by geological domain code)
  - Elevation range filtering
  - Manual boolean mask
  - Combination of multiple masks (AND logic)

ARBF must estimate ONLY on active cells. Inactive cells receive NaN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Reason codes for inactive cells
REASON_ACTIVE = 0
REASON_OUTSIDE_FOOTPRINT = 1
REASON_OUTSIDE_DOMAIN = 2
REASON_OUTSIDE_ELEVATION = 3
REASON_MANUAL_EXCLUDE = 4


@dataclass
class ActiveCellMask:
    block_model_name: str
    active: np.ndarray                             # (N,) bool
    reason_codes: Optional[np.ndarray] = None      # (N,) int8
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Block centres reference (set by factory methods) ─────────────
    _centres: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_total(self) -> int:
        return self.active.size

    @property
    def n_active(self) -> int:
        return int(np.sum(self.active))

    @property
    def n_inactive(self) -> int:
        return self.n_total - self.n_active

    @property
    def pct_active(self) -> float:
        return (self.n_active / max(self.n_total, 1)) * 100.0

    @property
    def active_centres(self) -> np.ndarray:
        """Return only the active block centres."""
        if self._centres is None:
            raise ValueError("Block centres not set. Use a factory method or set _centres.")
        return self._centres[self.active]

    @property
    def active_indices(self) -> np.ndarray:
        """Integer indices of active blocks."""
        return np.where(self.active)[0]

    # ── Factory: footprint clipping ──────────────────────────────────

    @classmethod
    def from_footprint_clip(
        cls,
        centres: np.ndarray,
        sample_coords: np.ndarray,
        search_radius: float,
        buffer_factor: float = 1.5,
        block_model_name: str = "arbf_grid",
    ) -> "ActiveCellMask":
        """Keep only blocks within buffer_factor * search_radius of nearest sample."""
        centres = np.asarray(centres, dtype=float)
        sample_coords = np.asarray(sample_coords, dtype=float)
        buffer_distance = search_radius * buffer_factor

        tree = cKDTree(sample_coords)
        dists, _ = tree.query(centres, k=1)
        active = dists <= buffer_distance

        n_total = centres.shape[0]
        n_active = int(np.sum(active))
        logger.info(
            "Footprint clip: %d -> %d blocks (removed %d, %.1f%%). Buffer=%.1f m",
            n_total, n_active, n_total - n_active,
            (n_total - n_active) / max(n_total, 1) * 100, buffer_distance,
        )

        reason = np.where(active, REASON_ACTIVE, REASON_OUTSIDE_FOOTPRINT).astype(np.int8)

        mask = cls(
            block_model_name=block_model_name,
            active=active,
            reason_codes=reason,
            metadata={
                "method": "footprint_clip",
                "search_radius": search_radius,
                "buffer_factor": buffer_factor,
                "buffer_distance": buffer_distance,
            },
        )
        mask._centres = centres
        return mask

    # ── Factory: domain filter ───────────────────────────────────────

    @classmethod
    def from_domain_filter(
        cls,
        centres: np.ndarray,
        block_domain_ids: np.ndarray,
        target_domains: Sequence,
        block_model_name: str = "arbf_grid",
    ) -> "ActiveCellMask":
        """Keep only blocks whose domain ID is in the target set."""
        active = np.isin(block_domain_ids, list(target_domains))
        reason = np.where(active, REASON_ACTIVE, REASON_OUTSIDE_DOMAIN).astype(np.int8)

        mask = cls(
            block_model_name=block_model_name,
            active=active,
            reason_codes=reason,
            metadata={"method": "domain_filter", "target_domains": list(target_domains)},
        )
        mask._centres = np.asarray(centres, dtype=float)
        return mask

    # ── Factory: elevation range ─────────────────────────────────────

    @classmethod
    def from_elevation_range(
        cls,
        centres: np.ndarray,
        z_min: float,
        z_max: float,
        block_model_name: str = "arbf_grid",
    ) -> "ActiveCellMask":
        """Keep only blocks within an elevation (Z) range."""
        centres = np.asarray(centres, dtype=float)
        z = centres[:, 2]
        active = (z >= z_min) & (z <= z_max)
        reason = np.where(active, REASON_ACTIVE, REASON_OUTSIDE_ELEVATION).astype(np.int8)

        mask = cls(
            block_model_name=block_model_name,
            active=active,
            reason_codes=reason,
            metadata={"method": "elevation_range", "z_min": z_min, "z_max": z_max},
        )
        mask._centres = centres
        return mask

    # ── Factory: all active (no clipping) ────────────────────────────

    @classmethod
    def all_active(
        cls,
        centres: np.ndarray,
        block_model_name: str = "arbf_grid",
    ) -> "ActiveCellMask":
        """All blocks are active (no clipping)."""
        centres = np.asarray(centres, dtype=float)
        mask = cls(
            block_model_name=block_model_name,
            active=np.ones(centres.shape[0], dtype=bool),
            metadata={"method": "all_active"},
        )
        mask._centres = centres
        return mask

    # ── Combine masks (AND logic) ────────────────────────────────────

    @classmethod
    def combine(cls, *masks: "ActiveCellMask") -> "ActiveCellMask":
        """Combine multiple masks using AND logic. First mask provides centres."""
        if not masks:
            raise ValueError("At least one mask required")

        combined_active = masks[0].active.copy()
        combined_reason = masks[0].reason_codes.copy() if masks[0].reason_codes is not None else np.zeros_like(combined_active, dtype=np.int8)

        for m in masks[1:]:
            if m.active.size != combined_active.size:
                raise ValueError("All masks must have the same number of blocks")
            # Where this mask deactivates a block, record its reason
            newly_inactive = combined_active & ~m.active
            if m.reason_codes is not None:
                combined_reason[newly_inactive] = m.reason_codes[newly_inactive]
            combined_active &= m.active

        result = cls(
            block_model_name=masks[0].block_model_name,
            active=combined_active,
            reason_codes=combined_reason,
            metadata={"method": "combined", "n_masks": len(masks)},
        )
        result._centres = masks[0]._centres
        return result

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        result = {
            "n_total": self.n_total,
            "n_active": self.n_active,
            "n_inactive": self.n_inactive,
            "pct_active": round(self.pct_active, 1),
        }
        if self.reason_codes is not None:
            for code, label in [
                (REASON_OUTSIDE_FOOTPRINT, "outside_footprint"),
                (REASON_OUTSIDE_DOMAIN, "outside_domain"),
                (REASON_OUTSIDE_ELEVATION, "outside_elevation"),
                (REASON_MANUAL_EXCLUDE, "manual_exclude"),
            ]:
                count = int(np.sum(self.reason_codes == code))
                if count > 0:
                    result[f"inactive_{label}"] = count
        return result
