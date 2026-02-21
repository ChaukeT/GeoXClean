"""
Unified State Manager
=====================

Single source of truth for all visualization and application state.
Provides thread-safe access and change notifications via Qt signals.

Key Features:
- Thread-safe state access with RLock
- Qt signals for state change notifications
- Serializable state for persistence
- Undo/redo support (future)

Usage:
    from block_model_viewer.core.state_manager import get_state_manager
    
    state = get_state_manager()
    state.legend_state_changed.connect(on_legend_changed)
    state.update_legend("Au_ppm", 0.0, 10.0, "viridis")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DATA CLASSES
# =============================================================================

@dataclass
class LegendState:
    """State for legend/colorbar display."""
    property_name: Optional[str] = None
    title: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    colormap: str = "viridis"
    categories: Optional[List[str]] = None
    category_colors: Optional[Dict[str, Any]] = None
    mode: str = "continuous"  # "continuous" or "discrete"
    visible: bool = True
    orientation: str = "vertical"
    font_size: int = 12
    
    def update(
        self,
        property_name: str,
        vmin: Optional[float],
        vmax: Optional[float],
        colormap: str,
        categories: Optional[List[str]] = None,
        category_colors: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update legend state."""
        self.property_name = property_name
        self.title = property_name
        self.vmin = vmin
        self.vmax = vmax
        self.colormap = colormap
        self.categories = categories
        self.category_colors = category_colors
        self.mode = "discrete" if categories else "continuous"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/signals."""
        return {
            "property": self.property_name,
            "title": self.title,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "colormap": self.colormap,
            "categories": self.categories,
            "category_colors": self.category_colors,
            "mode": self.mode,
            "visible": self.visible,
            "orientation": self.orientation,
            "font_size": self.font_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegendState":
        """Create from dictionary."""
        return cls(
            property_name=data.get("property"),
            title=data.get("title"),
            vmin=data.get("vmin"),
            vmax=data.get("vmax"),
            colormap=data.get("colormap", "viridis"),
            categories=data.get("categories"),
            category_colors=data.get("category_colors"),
            mode=data.get("mode", "continuous"),
            visible=data.get("visible", True),
            orientation=data.get("orientation", "vertical"),
            font_size=data.get("font_size", 12),
        )


@dataclass
class CameraState:
    """State for 3D camera."""
    position: Tuple[float, float, float] = (0.0, 0.0, 1000.0)
    focal_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    view_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    parallel_scale: float = 100.0
    is_parallel: bool = False
    clipping_range: Tuple[float, float] = (0.1, 10000.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": list(self.position),
            "focal_point": list(self.focal_point),
            "view_up": list(self.view_up),
            "parallel_scale": self.parallel_scale,
            "is_parallel": self.is_parallel,
            "clipping_range": list(self.clipping_range),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraState":
        return cls(
            position=tuple(data.get("position", [0, 0, 1000])),
            focal_point=tuple(data.get("focal_point", [0, 0, 0])),
            view_up=tuple(data.get("view_up", [0, 1, 0])),
            parallel_scale=data.get("parallel_scale", 100.0),
            is_parallel=data.get("is_parallel", False),
            clipping_range=tuple(data.get("clipping_range", [0.1, 10000])),
        )


@dataclass
class LayerState:
    """State for a single scene layer."""
    name: str
    visible: bool = True
    opacity: float = 1.0
    color: Optional[str] = None
    colormap: Optional[str] = None
    active_property: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "visible": self.visible,
            "opacity": self.opacity,
            "color": self.color,
            "colormap": self.colormap,
            "active_property": self.active_property,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerState":
        return cls(
            name=data.get("name", ""),
            visible=data.get("visible", True),
            opacity=data.get("opacity", 1.0),
            color=data.get("color"),
            colormap=data.get("colormap"),
            active_property=data.get("active_property"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DrillholeState:
    """State for drillhole visualization."""
    visible_holes: Set[str] = field(default_factory=set)
    selected_holes: Set[str] = field(default_factory=set)
    hovered_hole: Optional[str] = None
    color_mode: str = "lithology"  # "lithology" or "assay"
    active_assay_field: Optional[str] = None
    radius: float = 1.0
    colormap: str = "viridis"
    show_labels: bool = False
    show_collars: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "visible_holes": list(self.visible_holes),
            "selected_holes": list(self.selected_holes),
            "hovered_hole": self.hovered_hole,
            "color_mode": self.color_mode,
            "active_assay_field": self.active_assay_field,
            "radius": self.radius,
            "colormap": self.colormap,
            "show_labels": self.show_labels,
            "show_collars": self.show_collars,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DrillholeState":
        return cls(
            visible_holes=set(data.get("visible_holes", [])),
            selected_holes=set(data.get("selected_holes", [])),
            hovered_hole=data.get("hovered_hole"),
            color_mode=data.get("color_mode", "lithology"),
            active_assay_field=data.get("active_assay_field"),
            radius=data.get("radius", 1.0),
            colormap=data.get("colormap", "viridis"),
            show_labels=data.get("show_labels", False),
            show_collars=data.get("show_collars", True),
        )


@dataclass
class GeologyState:
    """State for geological modelling."""
    active_model_name: Optional[str] = None
    visible_surfaces: Set[str] = field(default_factory=set)
    visible_solids: Set[str] = field(default_factory=set)
    surface_opacity: float = 0.7
    solid_opacity: float = 0.5
    show_contacts: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_model_name": self.active_model_name,
            "visible_surfaces": list(self.visible_surfaces),
            "visible_solids": list(self.visible_solids),
            "surface_opacity": self.surface_opacity,
            "solid_opacity": self.solid_opacity,
            "show_contacts": self.show_contacts,
        }


# =============================================================================
# UNIFIED STATE MANAGER
# =============================================================================

class UnifiedStateManager(QObject):
    """
    Single source of truth for all visualization state.
    
    Provides:
    - Thread-safe state access
    - Qt signals for state change notifications
    - Serialization for persistence
    - State history for undo/redo (future)
    """
    
    # Signals for state changes
    legend_state_changed = pyqtSignal(dict)
    camera_state_changed = pyqtSignal(dict)
    layer_state_changed = pyqtSignal(str, dict)  # layer_name, state_dict
    drillhole_state_changed = pyqtSignal(dict)
    geology_state_changed = pyqtSignal(dict)
    
    # General state change signal (for any change)
    state_changed = pyqtSignal(str, dict)  # category, data
    
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        
        # Thread-safe lock
        self._lock = threading.RLock()
        
        # State objects
        self._legend_state = LegendState()
        self._camera_state = CameraState()
        self._layer_states: Dict[str, LayerState] = {}
        self._drillhole_state = DrillholeState()
        self._geology_state = GeologyState()
        
        # History for undo/redo (future)
        self._history: List[Dict[str, Any]] = []
        self._history_index: int = -1
        self._max_history: int = 50
        
        logger.info("UnifiedStateManager initialized")
    
    # =========================================================================
    # Legend State
    # =========================================================================
    
    def update_legend(
        self,
        property_name: str,
        vmin: Optional[float],
        vmax: Optional[float],
        colormap: str,
        categories: Optional[List[str]] = None,
        category_colors: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update legend state and emit signal."""
        with self._lock:
            self._legend_state.update(
                property_name, vmin, vmax, colormap, categories, category_colors
            )
            state_dict = self._legend_state.to_dict()
        
        self.legend_state_changed.emit(state_dict)
        self.state_changed.emit("legend", state_dict)
        logger.debug(f"Legend state updated: {property_name}")
    
    def set_legend_visibility(self, visible: bool) -> None:
        """Set legend visibility."""
        with self._lock:
            self._legend_state.visible = visible
            state_dict = self._legend_state.to_dict()
        
        self.legend_state_changed.emit(state_dict)
    
    def get_legend_state(self) -> Dict[str, Any]:
        """Get current legend state."""
        with self._lock:
            return self._legend_state.to_dict()
    
    # =========================================================================
    # Camera State
    # =========================================================================
    
    def update_camera(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        focal_point: Optional[Tuple[float, float, float]] = None,
        view_up: Optional[Tuple[float, float, float]] = None,
        parallel_scale: Optional[float] = None,
        is_parallel: Optional[bool] = None,
    ) -> None:
        """Update camera state."""
        with self._lock:
            if position is not None:
                self._camera_state.position = position
            if focal_point is not None:
                self._camera_state.focal_point = focal_point
            if view_up is not None:
                self._camera_state.view_up = view_up
            if parallel_scale is not None:
                self._camera_state.parallel_scale = parallel_scale
            if is_parallel is not None:
                self._camera_state.is_parallel = is_parallel
            
            state_dict = self._camera_state.to_dict()
        
        self.camera_state_changed.emit(state_dict)
        self.state_changed.emit("camera", state_dict)
    
    def get_camera_state(self) -> Dict[str, Any]:
        """Get current camera state."""
        with self._lock:
            return self._camera_state.to_dict()
    
    # =========================================================================
    # Layer State
    # =========================================================================
    
    def register_layer(self, name: str, **kwargs) -> None:
        """Register a new layer with optional initial state."""
        with self._lock:
            self._layer_states[name] = LayerState(name=name, **kwargs)
            state_dict = self._layer_states[name].to_dict()
        
        self.layer_state_changed.emit(name, state_dict)
        logger.debug(f"Layer registered: {name}")
    
    def update_layer(
        self,
        name: str,
        visible: Optional[bool] = None,
        opacity: Optional[float] = None,
        colormap: Optional[str] = None,
        active_property: Optional[str] = None,
    ) -> None:
        """Update layer state."""
        with self._lock:
            if name not in self._layer_states:
                self._layer_states[name] = LayerState(name=name)
            
            layer = self._layer_states[name]
            if visible is not None:
                layer.visible = visible
            if opacity is not None:
                layer.opacity = opacity
            if colormap is not None:
                layer.colormap = colormap
            if active_property is not None:
                layer.active_property = active_property
            
            state_dict = layer.to_dict()
        
        self.layer_state_changed.emit(name, state_dict)
    
    def remove_layer(self, name: str) -> None:
        """Remove a layer from state."""
        with self._lock:
            if name in self._layer_states:
                del self._layer_states[name]
        
        self.layer_state_changed.emit(name, {"removed": True})
    
    def get_layer_state(self, name: str) -> Optional[Dict[str, Any]]:
        """Get state for a specific layer."""
        with self._lock:
            if name in self._layer_states:
                return self._layer_states[name].to_dict()
            return None
    
    def get_all_layers(self) -> Dict[str, Dict[str, Any]]:
        """Get all layer states."""
        with self._lock:
            return {name: layer.to_dict() for name, layer in self._layer_states.items()}
    
    # =========================================================================
    # Drillhole State
    # =========================================================================
    
    def set_drillhole_visibility(self, hole_id: str, visible: bool) -> None:
        """Set visibility for a single drillhole."""
        with self._lock:
            if visible:
                self._drillhole_state.visible_holes.add(hole_id)
            else:
                self._drillhole_state.visible_holes.discard(hole_id)
            
            state_dict = self._drillhole_state.to_dict()
        
        self.drillhole_state_changed.emit(state_dict)
    
    def set_all_drillholes_visible(self, hole_ids: Set[str]) -> None:
        """Set visibility for all drillholes at once."""
        with self._lock:
            self._drillhole_state.visible_holes = set(hole_ids)
            state_dict = self._drillhole_state.to_dict()
        
        self.drillhole_state_changed.emit(state_dict)
    
    def get_visible_holes(self) -> Set[str]:
        """Get set of visible hole IDs."""
        with self._lock:
            return set(self._drillhole_state.visible_holes)
    
    def set_drillhole_color_mode(self, mode: str, assay_field: Optional[str] = None) -> None:
        """Set drillhole coloring mode."""
        with self._lock:
            self._drillhole_state.color_mode = mode
            if assay_field is not None:
                self._drillhole_state.active_assay_field = assay_field
            state_dict = self._drillhole_state.to_dict()
        
        self.drillhole_state_changed.emit(state_dict)
    
    def set_drillhole_radius(self, radius: float) -> None:
        """Set drillhole display radius."""
        with self._lock:
            self._drillhole_state.radius = radius
            state_dict = self._drillhole_state.to_dict()
        
        self.drillhole_state_changed.emit(state_dict)
    
    def get_drillhole_state(self) -> Dict[str, Any]:
        """Get current drillhole state."""
        with self._lock:
            return self._drillhole_state.to_dict()
    
    # =========================================================================
    # Geology State
    # =========================================================================
    
    def update_geology(
        self,
        active_model_name: Optional[str] = None,
        visible_surfaces: Optional[Set[str]] = None,
        visible_solids: Optional[Set[str]] = None,
    ) -> None:
        """Update geology state."""
        with self._lock:
            if active_model_name is not None:
                self._geology_state.active_model_name = active_model_name
            if visible_surfaces is not None:
                self._geology_state.visible_surfaces = visible_surfaces
            if visible_solids is not None:
                self._geology_state.visible_solids = visible_solids
            
            state_dict = self._geology_state.to_dict()
        
        self.geology_state_changed.emit(state_dict)
    
    def get_geology_state(self) -> Dict[str, Any]:
        """Get current geology state."""
        with self._lock:
            return self._geology_state.to_dict()
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for serialization."""
        with self._lock:
            return {
                "legend": self._legend_state.to_dict(),
                "camera": self._camera_state.to_dict(),
                "layers": {n: l.to_dict() for n, l in self._layer_states.items()},
                "drillholes": self._drillhole_state.to_dict(),
                "geology": self._geology_state.to_dict(),
                "timestamp": datetime.now().isoformat(),
            }
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        with self._lock:
            if "legend" in state:
                self._legend_state = LegendState.from_dict(state["legend"])
            if "camera" in state:
                self._camera_state = CameraState.from_dict(state["camera"])
            if "layers" in state:
                self._layer_states = {
                    n: LayerState.from_dict(l) for n, l in state["layers"].items()
                }
            if "drillholes" in state:
                self._drillhole_state = DrillholeState.from_dict(state["drillholes"])
        
        # Emit all state changed signals
        self.legend_state_changed.emit(self._legend_state.to_dict())
        self.camera_state_changed.emit(self._camera_state.to_dict())
        self.drillhole_state_changed.emit(self._drillhole_state.to_dict())
        for name, layer in self._layer_states.items():
            self.layer_state_changed.emit(name, layer.to_dict())
        
        logger.info("State restored from serialized data")
    
    def reset(self) -> None:
        """Reset all state to defaults."""
        with self._lock:
            self._legend_state = LegendState()
            self._camera_state = CameraState()
            self._layer_states.clear()
            self._drillhole_state = DrillholeState()
            self._geology_state = GeologyState()
        
        self.state_changed.emit("reset", {})
        logger.info("State manager reset to defaults")


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_state_manager: Optional[UnifiedStateManager] = None


def get_state_manager() -> UnifiedStateManager:
    """
    Get or create the global state manager instance.
    
    Returns:
        UnifiedStateManager singleton instance
    """
    global _state_manager
    if _state_manager is None:
        _state_manager = UnifiedStateManager()
    return _state_manager


def reset_state_manager() -> None:
    """Reset the global state manager (for testing)."""
    global _state_manager
    if _state_manager is not None:
        _state_manager.reset()
        _state_manager = None

