"""
Unified Overlay Manager - Single source of truth for all overlay logic.

This module consolidates overlay state, axis/coordinate widgets, scale bar,
and overlay lifecycle management. The OverlayManager is the central authority
for all overlay-related operations, while the Renderer acts as a passive
visualization backend that only receives draw commands.

Architecture:
    OverlayManager (this class) -> commands -> Renderer.add/remove/clear_overlay_actor()
"""

from __future__ import annotations

import logging
import weakref
import numpy as np
from typing import Dict, Optional, Tuple, Any, List, TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

# FIX CS-003: Import AppState for visibility gating
try:
    from ..controllers.app_state import AppState
except ImportError:
    # Fallback if import fails
    from enum import IntEnum
    class AppState(IntEnum):
        EMPTY = 0
        DATA_LOADED = 1
        RENDERED = 2
        BUSY = 3

if TYPE_CHECKING:
    from .renderer import Renderer

logger = logging.getLogger(__name__)


class OverlayManager(QObject):
    """
    Unified overlay manager - single source of truth for all overlay logic.
    
    Consolidates functionality from the former AxisManager and OverlayStateManager:
    - Overlay state management (grid, axes, bounds visibility)
    - Widget bindings (elevation, coordinate, scale bar)
    - Camera event handling
    - Tick spacing computation
    - Overlay lifecycle (rebuild triggers)
    
    The Renderer becomes a passive backend that only receives draw commands
    through add_overlay_actor() / remove_overlay_actor() / clear_overlay_actors().
    """

    # Signals for UI synchronization
    overlays_changed = pyqtSignal(dict)   # Overlay visibility state changed
    bounds_changed = pyqtSignal(tuple)    # Scene bounds updated (xmin, xmax, ymin, ymax, zmin, zmax)
    camera_changed = pyqtSignal(dict)     # Camera metadata changed

    def __init__(self, renderer: Optional["Renderer"] = None):
        """
        Initialize the unified overlay manager.
        
        Args:
            renderer: Optional renderer reference for issuing draw commands.
                      Can be set later via attach_renderer().
        """
        super().__init__()
        
        # Weak reference to renderer for issuing draw commands
        self._renderer_ref: Optional[weakref.ref] = None
        if renderer is not None:
            self._renderer_ref = weakref.ref(renderer)
        
        # ============================================================================
        # OVERLAY STATE (from OverlayStateManager)
        # ============================================================================
        self._overlays_enabled: Dict[str, bool] = {}
        self._widgets: Dict[str, Any] = {}
        
        # ============================================================================
        # BOUNDS AND CAMERA STATE (from AxisManager)
        # ============================================================================
        self._bounds: Optional[Tuple[float, float, float, float, float, float]] = None
        self._camera_metadata: Optional[Dict[str, Any]] = None
        
        # ============================================================================
        # WIDGET REFERENCES (from AxisManager)
        # ============================================================================
        self._elevation_widget: Optional[Any] = None
        self._coordinate_widget: Optional[Any] = None
        self._scale_bar_widget: Optional[Any] = None
        
        # ============================================================================
        # OVERLAY ACTORS (managed by this class, drawn by renderer)
        # ============================================================================
        self._overlay_actors: Dict[str, List[Any]] = {
            'axes': [],
            'scale_bar': [],
            'grid': [],
            'bounds': [],
            'other': [],
        }
        
        # ============================================================================
        # TICK SPACING SETTINGS
        # ============================================================================
        self._x_major: Optional[float] = None
        self._x_minor: Optional[float] = None
        self._y_major: Optional[float] = None
        self._y_minor: Optional[float] = None
        self._z_major: Optional[float] = None
        self._z_minor: Optional[float] = None
        
        # ============================================================================
        # UPDATE SUSPENSION (prevents race conditions during batch operations)
        # ============================================================================
        self._updates_suspended: bool = False
        self._pending_rebuild: bool = False
        
        # ============================================================================
        # FIX CS-003: APPLICATION STATE TRACKING
        # Overlays should only be visible when app state is RENDERED
        # ============================================================================
        self._app_state: AppState = AppState.EMPTY

    # ============================================================================
    # RENDERER ATTACHMENT
    # ============================================================================
    
    def attach_renderer(self, renderer: "Renderer") -> None:
        """
        Attach a renderer for issuing draw commands.
        
        Args:
            renderer: Renderer instance to attach.
        """
        self._renderer_ref = weakref.ref(renderer)
        logger.debug("OverlayManager attached to renderer")
    
    @property
    def renderer(self) -> Optional["Renderer"]:
        """Get the attached renderer, or None if not attached or deallocated."""
        if self._renderer_ref is None:
            return None
        return self._renderer_ref()

    # ============================================================================
    # FIX CS-003: APPLICATION STATE HANDLING
    # ============================================================================
    
    def on_app_state_changed(self, state: int) -> None:
        """
        Handle application state changes.
        
        FIX CS-003: Overlays should be hidden in EMPTY state and visible only
        when app state is RENDERED.
        
        Args:
            state: AppState enum value (as int for signal compatibility)
        """
        try:
            new_state = AppState(state)
        except ValueError:
            logger.warning(f"OverlayManager: Invalid app state value: {state}")
            return
        
        if self._app_state == new_state:
            return
        
        old_state = self._app_state
        self._app_state = new_state
        logger.debug(f"OverlayManager: State changed {old_state.name} -> {new_state.name}")
        
        # Apply state-specific visibility rules
        if new_state == AppState.EMPTY:
            # Hide all overlays and clear bounds
            self._hide_all_overlays()
            self.set_bounds(None)
        elif new_state == AppState.DATA_LOADED:
            # Data loaded but not rendered - keep overlays hidden
            self._hide_all_overlays()
        elif new_state == AppState.RENDERED:
            # Scene is rendered - overlays can be shown based on their enabled state
            self._restore_overlay_visibility()
        elif new_state == AppState.BUSY:
            # During engine processing - suspend updates
            self.suspend_updates()
    
    def _hide_all_overlays(self) -> None:
        """Hide all overlay widgets without changing their enabled state."""
        # Hide widgets
        if self._elevation_widget is not None:
            try:
                self._elevation_widget.hide()
            except Exception:
                pass
        
        if self._coordinate_widget is not None:
            try:
                if hasattr(self._coordinate_widget, 'set_visible'):
                    self._coordinate_widget.set_visible(False)
            except Exception:
                pass
        
        if self._scale_bar_widget is not None:
            try:
                if hasattr(self._scale_bar_widget, 'set_visible'):
                    self._scale_bar_widget.set_visible(False)
            except Exception:
                pass
        
        logger.debug("OverlayManager: All overlays hidden (EMPTY/DATA_LOADED state)")
    
    def _restore_overlay_visibility(self) -> None:
        """Restore overlay visibility based on their enabled states."""
        # Resume any suspended updates
        if self._updates_suspended:
            self.resume_updates()
        
        # Restore widget visibility based on bounds and enabled state
        self._update_widgets()
        
        logger.debug("OverlayManager: Overlay visibility restored (RENDERED state)")

    # ============================================================================
    # OVERLAY STATE MANAGEMENT (from OverlayStateManager)
    # ============================================================================

    def toggle_overlay(self, name: str, state: bool) -> None:
        """
        Toggle overlay visibility and notify listeners.
        
        Args:
            name: Overlay name ('axes', 'ground_grid', 'grid', 'bounds', 'scale_bar', etc.)
            state: True to show, False to hide
        """
        normalized = bool(state)
        if self._overlays_enabled.get(name) == normalized:
            return
        
        self._overlays_enabled[name] = normalized
        self.overlays_changed.emit(self._overlays_enabled.copy())
        
        # Issue command to renderer
        renderer = self.renderer
        if renderer is not None:
            # Map overlay names to renderer methods
            if name in ('axes', 'floating_axes'):
                try:
                    renderer.set_floating_axes_enabled(normalized)
                except Exception as e:
                    logger.debug(f"Could not toggle axes: {e}")
            elif name in ('ground_grid', 'grid'):
                try:
                    if hasattr(renderer, 'set_show_ground_grid'):
                        renderer.set_show_ground_grid(normalized)
                except Exception as e:
                    logger.debug(f"Could not toggle grid: {e}")
            elif name in ('bounds', 'bounding_box'):
                try:
                    if hasattr(renderer, 'toggle_bounds'):
                        renderer.toggle_bounds(normalized)
                except Exception as e:
                    logger.debug(f"Could not toggle bounds: {e}")
            elif name == 'scale_bar':
                try:
                    if hasattr(renderer, 'set_scale_bar_enabled'):
                        renderer.set_scale_bar_enabled(normalized)
                except Exception as e:
                    logger.debug(f"Could not toggle scale bar: {e}")
        
        logger.debug(f"Overlay '{name}' toggled to {normalized}")
    
    def is_overlay_enabled(self, name: str) -> bool:
        """Check if an overlay is currently enabled."""
        return self._overlays_enabled.get(name, False)

    def attach_widget(self, kind: str, widget: Any) -> None:
        """
        Register a Qt widget associated with an overlay (e.g., scale bar).
        
        Args:
            kind: Widget type identifier
            widget: Widget instance
        """
        self._widgets[kind] = widget
        logger.debug(f"Attached widget '{kind}'")

    def get_widget(self, kind: str) -> Optional[Any]:
        """Return a previously attached widget."""
        return self._widgets.get(kind)

    def get_state(self) -> Dict[str, Any]:
        """Return a serializable snapshot of overlay settings."""
        return {
            "overlays": dict(self._overlays_enabled),
            "bounds": self._bounds,
            "camera": self._camera_metadata,
        }

    # ============================================================================
    # BOUNDS MANAGEMENT (merged from both classes)
    # ============================================================================

    def set_bounds(self, bounds: Optional[Tuple[float, float, float, float, float, float]]) -> None:
        """
        Update scene bounds and notify widgets/listeners.
        
        This is the single entry point for bounds updates. It:
        1. Caches the bounds
        2. Emits bounds_changed signal
        3. Updates all bound widgets
        4. Triggers overlay rebuilds if needed
        
        Args:
            bounds: Scene bounds tuple (xmin, xmax, ymin, ymax, zmin, zmax) or None
        """
        if bounds is None:
            if self._bounds is not None:
                self._bounds = None
                self.bounds_changed.emit(())
                self._update_widgets()
            return
        
        # Normalize bounds to tuple of floats
        normalized = tuple(float(b) for b in bounds)
        if self._bounds == normalized:
            return
        
        self._bounds = normalized
        self.bounds_changed.emit(normalized)
        self._update_widgets()
        
        # Auto-compute tick spacing if not set
        if self._x_major is None:
            self._auto_compute_tick_spacing()
        
        # Trigger rebuild if not suspended
        if not self._updates_suspended:
            self._rebuild_overlays()
        else:
            self._pending_rebuild = True
        
        logger.debug(f"Bounds updated to {normalized}")

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Get current scene bounds."""
        return self._bounds

    # ============================================================================
    # CAMERA MANAGEMENT (from AxisManager)
    # ============================================================================

    def set_camera_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Update camera metadata (position, orientation, etc.).
        
        Args:
            metadata: Camera metadata dictionary with keys like 'position', 'focal_point', etc.
        """
        if metadata == self._camera_metadata:
            return
        
        self._camera_metadata = metadata
        self.camera_changed.emit(metadata)
        self._update_widgets()
        
        logger.debug("Camera metadata updated")

    @property
    def camera_metadata(self) -> Optional[Dict[str, Any]]:
        """Get current camera metadata."""
        return self._camera_metadata

    def update_on_camera_move(self, camera_info: Dict[str, Any]) -> None:
        """
        Handle camera movement events.
        
        Called by the renderer when the camera moves. Updates overlays that
        depend on camera position/orientation.
        
        Args:
            camera_info: Camera information dictionary
        """
        self.set_camera_metadata(camera_info)
        
        # Scale bar may need updating on camera move (for perspective projection)
        if self._scale_bar_widget is not None:
            try:
                if hasattr(self._scale_bar_widget, 'update'):
                    self._scale_bar_widget.update()
            except Exception:
                pass

    # ============================================================================
    # WIDGET BINDING (from AxisManager)
    # ============================================================================

    def bind_elevation_widget(self, widget: Any) -> None:
        """
        Bind elevation axis widget to this manager.
        
        Args:
            widget: Elevation axis widget instance
        """
        self._elevation_widget = widget
        self._update_widgets()
        logger.debug("Bound elevation widget")

    def bind_coordinate_widget(self, widget: Any) -> None:
        """
        Bind coordinate display widget to this manager.
        
        Args:
            widget: Coordinate display widget instance
        """
        self._coordinate_widget = widget
        self._update_widgets()
        logger.debug("Bound coordinate widget")

    def bind_scale_bar_widget(self, widget: Any) -> None:
        """
        Bind scale bar widget to this manager.
        
        Args:
            widget: Scale bar widget instance
        """
        self._scale_bar_widget = widget
        self._update_widgets()
        logger.debug("Bound scale bar widget")

    def _update_widgets(self) -> None:
        """Update all bound widgets with current state."""
        bounds = self._bounds
        
        # Update elevation widget
        if self._elevation_widget is not None:
            try:
                if bounds:
                    self._elevation_widget.show_for_bounds(bounds)
                else:
                    self._elevation_widget.hide()
            except Exception as e:
                logger.debug(f"Could not update elevation widget: {e}")
        
        # Update coordinate widget
        if self._coordinate_widget is not None:
            try:
                self._coordinate_widget.set_visible(bool(bounds))
                if bounds and hasattr(self._coordinate_widget, 'update_bounds'):
                    self._coordinate_widget.update_bounds(bounds)
            except Exception as e:
                logger.debug(f"Could not update coordinate widget: {e}")
        
        # Update scale bar widget
        if self._scale_bar_widget is not None:
            try:
                if bounds:
                    if hasattr(self._scale_bar_widget, 'update_bounds'):
                        self._scale_bar_widget.update_bounds(bounds)
                    if hasattr(self._scale_bar_widget, 'set_visible'):
                        self._scale_bar_widget.set_visible(True)
                else:
                    if hasattr(self._scale_bar_widget, 'set_visible'):
                        self._scale_bar_widget.set_visible(False)
            except Exception as e:
                logger.debug(f"Could not update scale bar widget: {e}")

    # ============================================================================
    # TICK SPACING COMPUTATION
    # ============================================================================

    def set_tick_spacing(
        self,
        x_major: Optional[float] = None, x_minor: Optional[float] = None,
        y_major: Optional[float] = None, y_minor: Optional[float] = None,
        z_major: Optional[float] = None, z_minor: Optional[float] = None
    ) -> None:
        """
        Set tick spacing for axis overlays.
        
        Args:
            x_major, x_minor: Major/minor tick spacing for X axis
            y_major, y_minor: Major/minor tick spacing for Y axis
            z_major, z_minor: Major/minor tick spacing for Z axis
        """
        self._x_major = x_major
        self._x_minor = x_minor
        self._y_major = y_major
        self._y_minor = y_minor
        self._z_major = z_major
        self._z_minor = z_minor
        
        if not self._updates_suspended:
            self._rebuild_overlays()
        else:
            self._pending_rebuild = True

    def compute_tick_spacing(self, bounds: Tuple[float, float, float, float, float, float]) -> Dict[str, Tuple[float, float]]:
        """
        Compute optimal tick spacing for given bounds.
        
        Args:
            bounds: Scene bounds (xmin, xmax, ymin, ymax, zmin, zmax)
            
        Returns:
            Dictionary with 'x', 'y', 'z' keys, each containing (major, minor) spacing tuple
        """
        result = {}
        for axis, (min_val, max_val) in [
            ('x', (bounds[0], bounds[1])),
            ('y', (bounds[2], bounds[3])),
            ('z', (bounds[4], bounds[5])),
        ]:
            span = max_val - min_val
            major, minor = self._auto_axis_spacing(span)
            result[axis] = (major, minor)
        return result

    def _auto_axis_spacing(self, span: float) -> Tuple[float, float]:
        """Compute a 'nice' major/minor tick spacing for a given span."""
        if not np.isfinite(span) or span <= 0:
            return 1.0, 0.5
        
        rough = max(span / 6.0, 1e-6)
        exponent = np.floor(np.log10(rough))
        base = 10 ** exponent
        candidates = [1.0, 2.0, 5.0, 10.0]
        
        major = base * candidates[-1]
        for c in candidates:
            candidate = base * c
            if rough <= candidate:
                major = candidate
                break
        
        minor = max(major / 5.0, major * 0.2)
        return major, minor

    def _auto_compute_tick_spacing(self) -> None:
        """Auto-compute tick spacing based on current bounds."""
        if self._bounds is None:
            return
        
        spacing = self.compute_tick_spacing(self._bounds)
        self._x_major, self._x_minor = spacing['x']
        self._y_major, self._y_minor = spacing['y']
        self._z_major, self._z_minor = spacing['z']

    # ============================================================================
    # OVERLAY LIFECYCLE
    # ============================================================================

    def suspend_updates(self) -> None:
        """
        Suspend overlay updates during batch operations.
        
        Use this to prevent race conditions when performing multiple
        operations that would each trigger a rebuild.
        """
        self._updates_suspended = True
        logger.debug("Overlay updates suspended")

    def resume_updates(self, force: bool = False) -> None:
        """
        Resume overlay updates after suspension.
        
        Args:
            force: If True, force a rebuild even if no pending changes
        """
        self._updates_suspended = False
        if self._pending_rebuild or force:
            self._pending_rebuild = False
            self._rebuild_overlays()
        logger.debug("Overlay updates resumed")

    def rebuild_axis_overlay(self) -> None:
        """
        Rebuild the axis box overlay.
        
        Clears existing axis actors and creates new ones based on current bounds.
        """
        renderer = self.renderer
        if renderer is None or self._bounds is None:
            return
        
        if not self._overlays_enabled.get('axes', False):
            return
        
        try:
            # Clear existing axis actors
            self._clear_overlay_type('axes')
            
            # Trigger renderer to rebuild floating axes
            if hasattr(renderer, 'set_floating_axes_enabled'):
                spacing = self.compute_tick_spacing(self._bounds)
                renderer.set_floating_axes_enabled(
                    True,
                    x_major=self._x_major or spacing['x'][0],
                    x_minor=self._x_minor or spacing['x'][1],
                    y_major=self._y_major or spacing['y'][0],
                    y_minor=self._y_minor or spacing['y'][1],
                    z_major=self._z_major or spacing['z'][0],
                    z_minor=self._z_minor or spacing['z'][1],
                )
            logger.debug("Rebuilt axis overlay")
        except Exception as e:
            logger.warning(f"Could not rebuild axis overlay: {e}", exc_info=True)

    def rebuild_scale_bar(self) -> None:
        """Rebuild the scale bar overlay."""
        renderer = self.renderer
        if renderer is None or self._bounds is None:
            return
        
        if not self._overlays_enabled.get('scale_bar', False):
            return
        
        try:
            # Trigger scale bar update
            if self._scale_bar_widget is not None:
                if hasattr(self._scale_bar_widget, 'update_bounds'):
                    self._scale_bar_widget.update_bounds(self._bounds)
            logger.debug("Rebuilt scale bar overlay")
        except Exception as e:
            logger.warning(f"Could not rebuild scale bar: {e}", exc_info=True)

    def _rebuild_overlays(self) -> None:
        """Rebuild all active overlays."""
        if self._bounds is None:
            return
        
        # Rebuild active overlays
        if self._overlays_enabled.get('axes', False):
            self.rebuild_axis_overlay()
        
        if self._overlays_enabled.get('scale_bar', False):
            self.rebuild_scale_bar()

    def _clear_overlay_type(self, overlay_type: str) -> None:
        """Clear all actors of a specific overlay type."""
        renderer = self.renderer
        if renderer is None:
            return
        
        actors = self._overlay_actors.get(overlay_type, [])
        for actor in actors:
            try:
                renderer.remove_overlay_actor(actor)
            except Exception as e:
                logger.debug(f"Could not remove overlay actor: {e}")
        
        self._overlay_actors[overlay_type] = []

    def clear_all_overlays(self) -> None:
        """Clear all overlay actors."""
        renderer = self.renderer
        if renderer is None:
            return
        
        try:
            renderer.clear_overlay_actors()
        except Exception as e:
            logger.debug(f"Could not clear overlay actors: {e}")
        
        # Reset internal tracking
        for key in self._overlay_actors:
            self._overlay_actors[key] = []
        
        logger.debug("Cleared all overlays")

    def close(self) -> None:
        """Clean up overlay actors during shutdown."""
        try:
            self.clear_all_overlays()
        except Exception as e:
            logger.debug(f"Error during overlay manager close: {e}")


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

class OverlayStateManager(OverlayManager):
    """
    Backward-compatible alias for OverlayManager.
    
    .. deprecated::
        Use OverlayManager instead. This class exists only for backward
        compatibility with code that imports OverlayStateManager.
    """
    
    def __init__(self, renderer: Optional["Renderer"] = None):
        import warnings
        warnings.warn(
            "OverlayStateManager is deprecated; use OverlayManager instead",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(renderer)
