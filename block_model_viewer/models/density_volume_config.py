"""
Shared Density & Volume Configuration — single source of truth.

Both the JORC Classification panel and the Resource Reporting panel read
from this shared config to ensure they produce identical tonnage numbers.
The config is stored in the DataRegistry.

If a shared BlockModelDefinition exists, volume is derived from it
automatically (dx * dy * dz from the definition spacing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DensityVolumeConfig:
    """Unified density and volume configuration for resource reporting.

    Stored in DataRegistry so both Classification and Reporting panels
    share the same settings.  Eliminates conflicting tonnage values.
    """

    # ── Density ──────────────────────────────────────────────────
    density_mode: str = "constant"  # "constant", "per_domain", "per_block"
    constant_density: float = 2.7   # t/m³, used when mode == "constant"
    domain_densities: Dict[str, float] = field(default_factory=dict)
    # e.g. {"Oxide": 2.2, "Sulphide": 2.8, "Transition": 2.5}
    density_column: Optional[str] = None  # block model column, used when mode == "per_block"

    # ── Volume ───────────────────────────────────────────────────
    volume_mode: str = "from_definition"  # "from_definition", "from_column", "constant"
    volume_column: Optional[str] = None   # block model column
    constant_dx: float = 10.0
    constant_dy: float = 10.0
    constant_dz: float = 5.0

    # ── Metadata ─────────────────────────────────────────────────
    source_panel: str = ""  # which panel last modified this config

    def get_block_volume(self, definition=None) -> Optional[float]:
        """Return constant block volume in m³, or None if per-block.

        Parameters
        ----------
        definition : BlockModelDefinition, optional
            If provided and volume_mode == "from_definition", volume is
            computed from definition.spacing.
        """
        if self.volume_mode == "from_definition" and definition is not None:
            dx, dy, dz = definition.block_size
            return dx * dy * dz
        elif self.volume_mode == "constant":
            return self.constant_dx * self.constant_dy * self.constant_dz
        return None  # per-block — caller reads from column

    def get_density_for_domain(self, domain_name: str) -> float:
        """Return density for a specific domain, or constant_density as fallback."""
        if self.density_mode == "per_domain":
            return self.domain_densities.get(domain_name, self.constant_density)
        return self.constant_density

    def to_reporting_configs(self, definition=None):
        """Convert to the (DensityConfig, VolumeConfig) tuple used by ResourceReportingEngine.

        Parameters
        ----------
        definition : BlockModelDefinition, optional
            Used to derive volume from spacing when volume_mode == "from_definition".

        Returns
        -------
        (DensityConfig, VolumeConfig)
        """
        import pandas as pd
        from .resource_reporting_engine import DensityConfig, VolumeConfig

        # ── Density ──
        if self.density_mode == "constant":
            dc = DensityConfig(mode="constant", constant_value=self.constant_density)
        elif self.density_mode == "per_domain":
            # Build domain table DataFrame expected by DensityConfig
            if self.domain_densities:
                rows = [{"DOMAIN": k, "DENSITY": v} for k, v in self.domain_densities.items()]
                dt = pd.DataFrame(rows)
                dc = DensityConfig(mode="domain", domain_table=dt)
            else:
                dc = DensityConfig(mode="constant", constant_value=self.constant_density)
        elif self.density_mode == "per_block":
            dc = DensityConfig(
                mode="block",
                block_density_field=self.density_column or "DENSITY",
            )
        else:
            dc = DensityConfig(mode="constant", constant_value=self.constant_density)

        # ── Volume ──
        if self.volume_mode == "from_definition" and definition is not None:
            dx, dy, dz = definition.block_size
            vc = VolumeConfig(mode="constant", dx=dx, dy=dy, dz=dz)
        elif self.volume_mode == "from_column":
            vc = VolumeConfig(mode="field", field_name=self.volume_column or "BLOCK_VOLUME")
        elif self.volume_mode == "constant":
            vc = VolumeConfig(
                mode="constant",
                dx=self.constant_dx, dy=self.constant_dy, dz=self.constant_dz,
            )
        else:
            # Fallback
            vc = VolumeConfig(mode="constant", dx=self.constant_dx, dy=self.constant_dy, dz=self.constant_dz)

        return dc, vc

    def to_classification_params(self) -> Tuple[Optional[str], float]:
        """Convert to (density_field, default_density) for the classification engine.

        Returns the format expected by JORCClassificationEngine.__init__().
        """
        if self.density_mode == "per_block":
            return self.density_column, self.constant_density
        elif self.density_mode == "constant":
            return None, self.constant_density
        else:
            # per_domain: classification engine doesn't support per-domain directly,
            # pass None and let it auto-detect or use constant fallback
            return None, self.constant_density

    def summary_text(self, definition=None) -> str:
        """Human-readable summary for display in panels."""
        parts = []

        # Density
        if self.density_mode == "constant":
            parts.append(f"Density: {self.constant_density:.2f} t/m\u00b3 (constant)")
        elif self.density_mode == "per_domain":
            n = len(self.domain_densities)
            parts.append(f"Density: per-domain ({n} domains)")
        elif self.density_mode == "per_block":
            parts.append(f"Density: from column '{self.density_column}'")

        # Volume
        vol = self.get_block_volume(definition)
        if vol is not None:
            parts.append(f"Volume: {vol:,.0f} m\u00b3/block")
        elif self.volume_mode == "from_column":
            parts.append(f"Volume: from column '{self.volume_column}'")

        return " | ".join(parts)
