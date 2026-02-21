"""
Drillhole State Manager

Centralized state management for drillhole visualization.
Guarantees consistent state across renderer, UI, and analysis panels.

Integration:
- Renderer calls register_holes(), set_visibility(), set_selection() on state changes
- State persists across visibility toggles and selection changes
- Supports undo/redo via history stack (not yet exposed in UI)

Stores:
- Visibility flags per hole
- Selected intervals
- Hovered interval
- Camera state
- Clipping planes
- Active color property
- Rendering mode
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraState:
    """Camera state for view restoration."""
    position: Tuple[float, float, float] = (0, 0, 1000)
    focal_point: Tuple[float, float, float] = (0, 0, 0)
    view_up: Tuple[float, float, float] = (0, 1, 0)
    parallel_scale: float = 100.0
    is_parallel: bool = False
    clipping_range: Tuple[float, float] = (0.1, 10000)


@dataclass
class ClipPlaneState:
    """Clipping plane configuration."""
    enabled: bool = False
    origin: Tuple[float, float, float] = (0, 0, 0)
    normal: Tuple[float, float, float] = (0, 0, 1)
    position: float = 0.0  # Slider position (0-1)
    orientation: str = "XY"  # XY, XZ, YZ, or Custom


@dataclass
class DrillholeVisualState:
    """Visual state for a single drillhole."""
    hole_id: str
    visible: bool = True
    expanded: bool = True  # UI tree expansion
    intervals_selected: Set[int] = field(default_factory=set)  # color_ids
    custom_color: Optional[str] = None  # Override color


@dataclass
class SceneState:
    """Complete scene state for save/restore."""
    # Visibility
    visible_holes: Set[str] = field(default_factory=set)
    all_visible: bool = True
    
    # Selection
    selected_interval_ids: Set[int] = field(default_factory=set)
    selected_hole_ids: Set[str] = field(default_factory=set)
    selection_mode: str = "interval"  # 'interval' or 'hole'
    
    # Hover
    hovered_interval_id: Optional[int] = None
    
    # Display
    color_property: str = "lithology"
    colormap: str = "tab10"
    tube_radius: float = 1.0
    render_quality: int = 16  # n_sides
    show_collars: bool = True
    show_labels: bool = False
    
    # Camera
    camera: CameraState = field(default_factory=CameraState)
    
    # Clipping
    clip_plane: ClipPlaneState = field(default_factory=ClipPlaneState)
    clip_box: Optional[Tuple[float, float, float, float, float, float]] = None
    
    # Filters
    depth_filter: Optional[Tuple[float, float]] = None  # min, max depth
    lith_filter: Set[str] = field(default_factory=set)  # show only these
    assay_filter: Optional[Tuple[float, float]] = None  # min, max assay
    
    # Metadata
    hole_count: int = 0
    interval_count: int = 0
    data_bounds: Optional[Tuple[float, float, float, float, float, float]] = None


class DrillholeStateManager:
    """
    Manages all drillhole visualization state.
    
    Features:
    - Centralized state storage
    - State change notifications
    - State persistence (save/load)
    - Undo/redo support (TODO)
    - State validation
    """
    
    def __init__(self):
        self._state = SceneState()
        self._hole_states: Dict[str, DrillholeVisualState] = {}
        self._listeners: Dict[str, List[callable]] = {}
        self._history: List[SceneState] = []
        self._redo_stack: List[SceneState] = []
        self._max_history = 50
        
        logger.info("DrillholeStateManager initialized")
    
    @property
    def state(self) -> SceneState:
        return self._state
    
    # -------------------------------------------------------------------------
    # State Accessors
    # -------------------------------------------------------------------------
    
    def get_visible_holes(self) -> Set[str]:
        return self._state.visible_holes.copy()
    
    def is_hole_visible(self, hole_id: str) -> bool:
        if self._state.all_visible:
            return True
        return hole_id in self._state.visible_holes
    
    def get_selected_intervals(self) -> Set[int]:
        return self._state.selected_interval_ids.copy()
    
    def is_interval_selected(self, color_id: int) -> bool:
        return color_id in self._state.selected_interval_ids
    
    def get_hovered_interval(self) -> Optional[int]:
        return self._state.hovered_interval_id
    
    def get_color_property(self) -> str:
        return self._state.color_property
    
    def get_colormap(self) -> str:
        return self._state.colormap
    
    # -------------------------------------------------------------------------
    # State Mutators
    # -------------------------------------------------------------------------
    
    def set_visibility(self, hole_id: str, visible: bool):
        """Set visibility for a single hole."""
        self._push_history()
        
        if visible:
            self._state.visible_holes.add(hole_id)
        else:
            self._state.visible_holes.discard(hole_id)
        
        self._state.all_visible = False
        self._notify("visibility_changed", {"hole_id": hole_id, "visible": visible})
    
    def set_all_visible(self, visible: bool):
        """Set visibility for all holes."""
        self._push_history()
        
        self._state.all_visible = visible
        if visible:
            # Restore all holes
            self._state.visible_holes = set(self._hole_states.keys())
        else:
            self._state.visible_holes.clear()
        
        self._notify("visibility_changed", {"all_visible": visible})
    
    def toggle_visibility(self, hole_ids: Set[str]):
        """Toggle visibility for a set of holes."""
        self._push_history()
        
        for hid in hole_ids:
            if hid in self._state.visible_holes:
                self._state.visible_holes.discard(hid)
            else:
                self._state.visible_holes.add(hid)
        
        self._state.all_visible = False
        self._notify("visibility_changed", {"toggled": hole_ids})
    
    def set_selection(self, interval_ids: Set[int], append: bool = False):
        """Set selected intervals."""
        self._push_history()
        
        if append:
            self._state.selected_interval_ids.update(interval_ids)
        else:
            self._state.selected_interval_ids = interval_ids.copy()
        
        self._notify("selection_changed", {"selected": self._state.selected_interval_ids})
    
    def clear_selection(self):
        """Clear all selection."""
        self._push_history()
        self._state.selected_interval_ids.clear()
        self._state.selected_hole_ids.clear()
        self._notify("selection_changed", {"selected": set()})
    
    def set_hover(self, interval_id: Optional[int]):
        """Set hovered interval (no history push - too frequent)."""
        if self._state.hovered_interval_id != interval_id:
            self._state.hovered_interval_id = interval_id
            self._notify("hover_changed", {"hovered": interval_id})
    
    def set_color_property(self, property_name: str):
        """Set color property (lithology or assay field)."""
        self._push_history()
        self._state.color_property = property_name
        self._notify("color_property_changed", {"property": property_name})
    
    def set_colormap(self, colormap: str):
        """Set colormap."""
        self._push_history()
        self._state.colormap = colormap
        self._notify("colormap_changed", {"colormap": colormap})
    
    def set_camera(self, camera: CameraState):
        """Set camera state."""
        self._state.camera = camera
        self._notify("camera_changed", {"camera": camera})
    
    def set_clip_plane(self, clip: ClipPlaneState):
        """Set clipping plane."""
        self._push_history()
        self._state.clip_plane = clip
        self._notify("clip_plane_changed", {"clip": clip})
    
    def set_tube_radius(self, radius: float):
        """Set tube radius."""
        self._push_history()
        self._state.tube_radius = radius
        self._notify("render_settings_changed", {"tube_radius": radius})
    
    def set_render_quality(self, n_sides: int):
        """Set render quality (tube n_sides)."""
        self._push_history()
        self._state.render_quality = n_sides
        self._notify("render_settings_changed", {"render_quality": n_sides})
    
    # -------------------------------------------------------------------------
    # Hole Registration
    # -------------------------------------------------------------------------
    
    def register_holes(self, hole_ids: List[str]):
        """Register holes for state tracking."""
        for hid in hole_ids:
            if hid not in self._hole_states:
                self._hole_states[hid] = DrillholeVisualState(hole_id=hid)
        
        self._state.visible_holes = set(hole_ids)
        self._state.hole_count = len(hole_ids)
        logger.info(f"Registered {len(hole_ids)} holes")
    
    def set_interval_count(self, count: int):
        """Set total interval count."""
        self._state.interval_count = count
    
    def set_data_bounds(
        self,
        bounds: Tuple[float, float, float, float, float, float]
    ):
        """Set data bounding box (xmin, xmax, ymin, ymax, zmin, zmax)."""
        self._state.data_bounds = bounds
    
    # -------------------------------------------------------------------------
    # Filters
    # -------------------------------------------------------------------------
    
    def set_depth_filter(self, min_depth: Optional[float], max_depth: Optional[float]):
        """Set depth filter range."""
        self._push_history()
        if min_depth is None and max_depth is None:
            self._state.depth_filter = None
        else:
            self._state.depth_filter = (min_depth or 0.0, max_depth or float('inf'))
        self._notify("filter_changed", {"depth_filter": self._state.depth_filter})
    
    def set_lith_filter(self, lith_codes: Set[str]):
        """Set lithology filter (show only these codes)."""
        self._push_history()
        self._state.lith_filter = lith_codes.copy()
        self._notify("filter_changed", {"lith_filter": self._state.lith_filter})
    
    def set_assay_filter(self, min_val: Optional[float], max_val: Optional[float]):
        """Set assay value filter range."""
        self._push_history()
        if min_val is None and max_val is None:
            self._state.assay_filter = None
        else:
            self._state.assay_filter = (min_val or 0.0, max_val or float('inf'))
        self._notify("filter_changed", {"assay_filter": self._state.assay_filter})
    
    def clear_filters(self):
        """Clear all filters."""
        self._push_history()
        self._state.depth_filter = None
        self._state.lith_filter.clear()
        self._state.assay_filter = None
        self._notify("filter_changed", {"cleared": True})
    
    # -------------------------------------------------------------------------
    # Event System
    # -------------------------------------------------------------------------
    
    def on(self, event: str, callback: callable):
        """Register event listener."""
        self._listeners.setdefault(event, []).append(callback)
    
    def off(self, event: str, callback: callable):
        """Unregister event listener."""
        if event in self._listeners:
            try:
                self._listeners[event].remove(callback)
            except ValueError:
                pass
    
    def _notify(self, event: str, data: Dict[str, Any]):
        """Notify all listeners of an event."""
        for callback in self._listeners.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.warning(f"Event listener error for {event}: {e}")
    
    # -------------------------------------------------------------------------
    # History / Undo-Redo
    # -------------------------------------------------------------------------
    
    def _push_history(self):
        """Push current state to history."""
        import copy
        
        # Deep copy current state
        state_copy = SceneState(
            visible_holes=self._state.visible_holes.copy(),
            all_visible=self._state.all_visible,
            selected_interval_ids=self._state.selected_interval_ids.copy(),
            selected_hole_ids=self._state.selected_hole_ids.copy(),
            selection_mode=self._state.selection_mode,
            hovered_interval_id=self._state.hovered_interval_id,
            color_property=self._state.color_property,
            colormap=self._state.colormap,
            tube_radius=self._state.tube_radius,
            render_quality=self._state.render_quality,
            show_collars=self._state.show_collars,
            show_labels=self._state.show_labels,
            clip_plane=copy.copy(self._state.clip_plane),
            clip_box=self._state.clip_box,
            depth_filter=self._state.depth_filter,
            lith_filter=self._state.lith_filter.copy(),
            assay_filter=self._state.assay_filter,
            hole_count=self._state.hole_count,
            interval_count=self._state.interval_count,
            data_bounds=self._state.data_bounds,
        )
        
        self._history.append(state_copy)
        self._redo_stack.clear()
        
        # Limit history size
        if len(self._history) > self._max_history:
            self._history.pop(0)
    
    def undo(self) -> bool:
        """Undo last state change."""
        if not self._history:
            return False
        
        # Save current to redo
        self._redo_stack.append(self._state)
        
        # Restore previous
        self._state = self._history.pop()
        self._notify("state_restored", {"action": "undo"})
        return True
    
    def redo(self) -> bool:
        """Redo last undone change."""
        if not self._redo_stack:
            return False
        
        # Save current to history
        self._history.append(self._state)
        
        # Restore from redo
        self._state = self._redo_stack.pop()
        self._notify("state_restored", {"action": "redo"})
        return True
    
    def can_undo(self) -> bool:
        return len(self._history) > 0
    
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    
    def save_state(self, filepath: Path):
        """Save state to JSON file."""
        try:
            # Convert to serializable format
            data = {
                "visible_holes": list(self._state.visible_holes),
                "all_visible": self._state.all_visible,
                "selected_interval_ids": list(self._state.selected_interval_ids),
                "selected_hole_ids": list(self._state.selected_hole_ids),
                "selection_mode": self._state.selection_mode,
                "color_property": self._state.color_property,
                "colormap": self._state.colormap,
                "tube_radius": self._state.tube_radius,
                "render_quality": self._state.render_quality,
                "show_collars": self._state.show_collars,
                "show_labels": self._state.show_labels,
                "camera": {
                    "position": list(self._state.camera.position),
                    "focal_point": list(self._state.camera.focal_point),
                    "view_up": list(self._state.camera.view_up),
                    "parallel_scale": self._state.camera.parallel_scale,
                    "is_parallel": self._state.camera.is_parallel,
                },
                "clip_plane": {
                    "enabled": self._state.clip_plane.enabled,
                    "origin": list(self._state.clip_plane.origin),
                    "normal": list(self._state.clip_plane.normal),
                    "position": self._state.clip_plane.position,
                    "orientation": self._state.clip_plane.orientation,
                },
                "depth_filter": list(self._state.depth_filter) if self._state.depth_filter else None,
                "lith_filter": list(self._state.lith_filter),
                "assay_filter": list(self._state.assay_filter) if self._state.assay_filter else None,
                "hole_count": self._state.hole_count,
                "interval_count": self._state.interval_count,
                "data_bounds": list(self._state.data_bounds) if self._state.data_bounds else None,
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved drillhole state to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self, filepath: Path) -> bool:
        """Load state from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self._state.visible_holes = set(data.get("visible_holes", []))
            self._state.all_visible = data.get("all_visible", True)
            self._state.selected_interval_ids = set(data.get("selected_interval_ids", []))
            self._state.selected_hole_ids = set(data.get("selected_hole_ids", []))
            self._state.selection_mode = data.get("selection_mode", "interval")
            self._state.color_property = data.get("color_property", "lithology")
            self._state.colormap = data.get("colormap", "tab10")
            self._state.tube_radius = data.get("tube_radius", 1.0)
            self._state.render_quality = data.get("render_quality", 16)
            self._state.show_collars = data.get("show_collars", True)
            self._state.show_labels = data.get("show_labels", False)
            
            cam = data.get("camera", {})
            self._state.camera = CameraState(
                position=tuple(cam.get("position", [0, 0, 1000])),
                focal_point=tuple(cam.get("focal_point", [0, 0, 0])),
                view_up=tuple(cam.get("view_up", [0, 1, 0])),
                parallel_scale=cam.get("parallel_scale", 100.0),
                is_parallel=cam.get("is_parallel", False),
            )
            
            clip = data.get("clip_plane", {})
            self._state.clip_plane = ClipPlaneState(
                enabled=clip.get("enabled", False),
                origin=tuple(clip.get("origin", [0, 0, 0])),
                normal=tuple(clip.get("normal", [0, 0, 1])),
                position=clip.get("position", 0.0),
                orientation=clip.get("orientation", "XY"),
            )
            
            df = data.get("depth_filter")
            self._state.depth_filter = tuple(df) if df else None
            
            self._state.lith_filter = set(data.get("lith_filter", []))
            
            af = data.get("assay_filter")
            self._state.assay_filter = tuple(af) if af else None
            
            self._state.hole_count = data.get("hole_count", 0)
            self._state.interval_count = data.get("interval_count", 0)
            
            db = data.get("data_bounds")
            self._state.data_bounds = tuple(db) if db else None
            
            self._notify("state_restored", {"action": "load"})
            logger.info(f"Loaded drillhole state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    
    def reset(self):
        """Reset to default state."""
        self._state = SceneState()
        self._hole_states.clear()
        self._history.clear()
        self._redo_stack.clear()
        self._notify("state_reset", {})
        logger.info("DrillholeStateManager reset")


# Singleton instance
_state_manager: Optional[DrillholeStateManager] = None

def get_drillhole_state_manager() -> DrillholeStateManager:
    """Get or create the global drillhole state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = DrillholeStateManager()
    return _state_manager

