"""
Lightweight container for renderer scene layers.

Provides both attribute-style access and a minimal mapping interface so
existing UI components that expect ``dict``-like objects keep working while we
gradually migrate to a richer API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


# Layer type constants (STEP 19: Geotechnical layers)
LAYER_TYPE_GEOTECH_GRID = "geotech_grid"
LAYER_TYPE_STOPE_STABILITY = "stope_stability"
LAYER_TYPE_SLOPE_RISK = "slope_risk"

# Layer type constants (STEP 20: Seismic layers)
LAYER_TYPE_SEISMIC_EVENTS = "seismic_events"
LAYER_TYPE_SEISMIC_HAZARD = "seismic_hazard"
LAYER_TYPE_ROCKBURST_INDEX = "rockburst_index"

# Layer type constants (STEP 21: Schedule Risk layers)
LAYER_TYPE_SCHEDULE_RISK = "schedule_risk"

# Layer type constants (STEP 26: Drillhole, Geology, Structural layers)
LAYER_TYPE_DRILLHOLE_TRACE = "drillhole_trace"
LAYER_TYPE_DRILLHOLE_ASSAY = "drillhole_assay"
LAYER_TYPE_DRILLHOLE_LITHOLOGY = "drillhole_lithology"
LAYER_TYPE_GEOLOGY_WIREFRAME = "geology_wireframe"
LAYER_TYPE_STRUCTURAL_DATA = "structural_data"

# Layer type constants (STEP 27: Slope Stability layers)
LAYER_TYPE_SLOPE_SECTOR = "slope_sector"
LAYER_TYPE_FAILURE_SURFACE_2D = "failure_surface_2d"
LAYER_TYPE_FAILURE_SURFACE_3D = "failure_surface_3d"
LAYER_TYPE_FOS_HEATMAP = "fos_heatmap"

# Layer type constants (STEP 29: Grade Control & Reconciliation layers)
LAYER_TYPE_GC_BLOCKS = "gc_blocks"
LAYER_TYPE_DIG_POLYGONS = "dig_polygons"
LAYER_TYPE_DIG_BLOCK_FLAGS = "dig_block_flags"

# Layer type constants (STEP 30: Scheduling layers)
LAYER_TYPE_SCHEDULE_PERIOD_HIGHLIGHT = "schedule_period_highlight"
LAYER_TYPE_DIGLINE_SCHEDULE = "digline_schedule"
LAYER_TYPE_FLEET_ROUTE_OVERLAY = "fleet_route_overlay"

# Layer type constants (STEP 33: Pushback layers)
LAYER_TYPE_PUSHBACK_PHASES = "pushback_phases"

# Layer type constants (STEP 37: Advanced Underground layers)
LAYER_TYPE_UG_STOPE_LAYOUT = "ug_stope_layout"
LAYER_TYPE_UG_CAVE_FOOTPRINT = "ug_cave_footprint"
LAYER_TYPE_UG_VOID_STATE = "ug_void_state"


@dataclass
class SceneLayer:
    """Represents a visual layer managed by the renderer."""

    name: str
    layer_type: str = "default"
    actor: Any = None
    data: Any = None
    visible: bool = True
    opacity: float = 1.0
    properties: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    pickable: bool = True

    # ------------------------------------------------------------------
    # Dict-style compatibility helpers
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if key in ("metadata", "properties") or value is not None:
            return value
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Return attribute or metadata entry, keeping legacy callers working."""
        if key == "name":
            return self.name
        if key == "type":
            return self.layer_type
        if key == "actor":
            return self.actor
        if key == "data":
            return self.data
        if key == "visible":
            return self.visible
        if key == "opacity":
            return self.opacity
        if key == "properties":
            return self.properties
        if key == "pickable":
            return self.pickable
        if key == "metadata":
            return self.metadata
        return self.metadata.get(key, default)

    def keys(self) -> Iterable[str]:
        """Expose keys used by existing UI components."""
        return ("name", "type", "actor", "data", "visible", "opacity", "properties", "metadata", "pickable")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def update_properties(self, properties: Optional[Iterable[str]]) -> None:
        if properties is None:
            return
        self.properties = list(properties)

    def to_payload(self) -> Dict[str, Any]:
        """Export a lightweight dict for UI serialization."""
        payload = {
            "name": self.name,
            "type": self.layer_type,
            "visible": self.visible,
            "opacity": self.opacity,
            "pickable": self.pickable,
        }
        if self.properties:
            payload["properties"] = list(self.properties)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


