"""
3D viewer widget integrating a PyVista backend with PyQt6.
"""

from __future__ import annotations

import pyvista as pv
import pyvistaqt as pvqt
import numpy as np
import pandas as pd
import vtk
from pyvista import _vtk as vtkmodules  # Access VTK classes for clipping
from typing import Optional, Dict, Any, Tuple
import logging
import weakref
import types

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QApplication, QMenu, QToolTip
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPoint, QObject, QEvent
from PyQt6.QtGui import QShowEvent, QContextMenuEvent, QAction, QMouseEvent

from ..models.block_model import BlockModel
from ..visualization import Renderer, ColorMapper, Filters
from ..visualization.picking_controller import get_picking_controller, PickingLOD
from .elevation_axis_widget import ElevationAxisWidget
from .scale_bar_widget import ScaleBarWidget
from .north_arrow_widget import NorthArrowWidget
from .utils.hover_debouncer import HoverDebouncer

logger = logging.getLogger(__name__)


class _PlotterEventFilter(QObject):
    """Event filter to enhance interactions on the QtInteractor (double-click finish, single-click picking)."""
    def __init__(self, viewer_widget: 'ViewerWidget'):
        super().__init__(viewer_widget)
        self._viewer = viewer_widget
        self._press_pos = None
        self._press_time = None



    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()
    def eventFilter(self, watched, event):  # type: ignore[override]
        try:
            # DEBUG: Log all mouse button events to trace consumption
            if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
                btn = event.button()
                if btn == Qt.MouseButton.LeftButton:
                    logger.info(f"FILTER: Mouse event {event.type()} detected on {watched}")

            # Double-click handling
            if event.type() == QEvent.Type.MouseButtonDblClick:
                if self._viewer and self._viewer.renderer and getattr(self._viewer.renderer, 'measure_mode', None) == 'area':
                    if hasattr(self._viewer.renderer, 'finish_area_measurement'):
                        self._viewer.renderer.finish_area_measurement()
                        parent = self._viewer.window()
                        if hasattr(parent, 'statusBar'):
                            parent.statusBar().showMessage("Area measurement finished (double-click)", 2000)
                        return True  # consume event
            
            # CLICK HANDLING: Catch events directly from the interactor widget
            elif event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    import time
                    self._press_pos = event.position().toPoint()
                    self._press_time = time.time()
                    logger.info(f"FILTER: Mouse press recorded at {self._press_pos}")
            
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
                    import time
                    release_pos = event.position().toPoint()
                    elapsed = time.time() - self._press_time if self._press_time else 999
                    
                    # Check if this was a click (not drag)
                    dx = abs(release_pos.x() - self._press_pos.x())
                    dy = abs(release_pos.y() - self._press_pos.y())
                    
                    logger.info(f"FILTER: Release at {release_pos}, dx={dx}, dy={dy}, elapsed={elapsed:.3f}")
                    
                    if dx < 5 and dy < 5 and elapsed < 0.5:
                        # This is a VALID click on the interactor
                        logger.debug(f"FILTER: Valid click detected at ({release_pos.x()}, {release_pos.y()})")
                        
                        # Gate on PickingController - single authority for picking state
                        picking_ctrl = get_picking_controller()
                        if self._viewer and picking_ctrl.click_allowed and hasattr(self._viewer, '_handle_block_click'):
                            logger.debug("FILTER: Calling _handle_block_click")
                            self._viewer._handle_block_click(release_pos.x(), release_pos.y())
                        elif self._viewer and not picking_ctrl.click_allowed:
                            logger.debug(f"FILTER: Click not allowed (LOD={picking_ctrl.lod.name})")
                    else:
                        logger.info("FILTER: Ignored (drag or long press)")
                    
                    self._press_pos = None
                    self._press_time = None

            # Wheel event handling for scale bar zoom updates
            elif event.type() == QEvent.Type.Wheel:
                # Update scale bar after wheel zoom
                if self._viewer and self._viewer.renderer:
                    # Use QTimer.singleShot to update after VTK processes the zoom
                    QTimer.singleShot(50, self._viewer.renderer.update_scale_bar_on_zoom)

        except Exception as e:
            logger.error(f"Event filter error: {e}", exc_info=True)

        return False  # Don't consume event, let VTK handle it too


class ViewerWidget(QWidget):
    """
    3D viewer widget using PyVistaQt for integration with PyQt6.
    
    Provides 3D visualization of block models with interactive controls.
    """
    
    # Signals
    block_picked = pyqtSignal(int, dict)  # block_index, properties
    view_changed = pyqtSignal(dict)  # camera_info
    global_pick_event = pyqtSignal(dict)  # Global pick information (for PickInfoPanel)
    box_selection_completed = pyqtSignal(tuple)  # (xmin, xmax, ymin, ymax, zmin, zmax)
    blocks_selected = pyqtSignal(set)  # set of block indices
    plane_position_changed = pyqtSignal(str, float)  # axis, position
    # Emitted when the viewer's high-level mouse/interaction mode changes.
    mouse_mode_changed = pyqtSignal(str)
    # Emitted whenever renderer layers are added/removed/updated.
    layers_changed = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        
        # Set minimum size for the viewer
        self.setMinimumSize(400, 400)
        
        # Visualization components
        self.renderer = Renderer()
        self._layer_callback_refs: list = []
        self._install_layer_change_bridge()
        self.color_mapper = ColorMapper()
        self.filters = Filters()
        
        # Current state
        self.current_model: Optional[BlockModel] = None
        self.current_property: Optional[str] = None
        self.current_colormap = 'viridis'
        self.current_color_mode = 'continuous'  # 'continuous' or 'discrete'
        self.current_transparency = 1.0
        self._override_block_size: Optional[Tuple[float, float, float]] = None
        
        # Backend-specific view handle
        self.plotter = None
        
        # Picking controller - single authority for all picking state
        self._picking_controller = get_picking_controller()
        
        # Legacy flag for backward compatibility - delegates to PickingController
        self._interaction_enabled = False  # Disabled by default - enabled after renderable actors added
        self.last_click_xy: Optional[Tuple[float, float]] = None
        self.display_grid = None
        self.mesh_actor = None
        self.properties_to_show = ['Au', 'Cu', 'LITO', 'DENSITY']  # Customize as needed
        
        # Hover highlighting state
        self._hover_highlight_actor = None  # Actor for hover highlight
        self._hovered_block_id = None  # Currently hovered block ID
        
        self._gesture_state = {
            'pan_start': None,
            'zoom_start': None,
            'rotate_start': None
        }
        self._last_camera_info: Optional[Dict[str, Any]] = None
        self._cached_bounds: Optional[Tuple[float, float, float, float, float, float]] = None
        self._bounds_cache_time: float = 0.0
        try:
            self.renderer.configure_tooltip_properties(self.properties_to_show)
        except Exception:
            logger.debug("Renderer does not support tooltip configuration")
        
        # Interactive slicing state (VTK mapper-based clipping - FAST & NO FREEZING)
        self.slice_active: bool = False
        self._plane_widget = None  # PyVista plane widget handle
        self._clip_plane = None  # VTK clipping plane object
        self._current_normal = "z"  # Current slice plane orientation
        
        # Interactive selection state
        self.selection_mode = None  # None, 'box', 'click', 'plane'
        self._box_widget = None  # PyVista box widget for rubber-band selection
        self._plane_section_widget = None  # PyVista plane widget for cross-section positioning
        self._selected_blocks = set()  # Currently selected block indices
        self._selection_actor = None  # Actor showing selected blocks
        self._plane_actor = None  # Visible plane actor to aid interaction

        # Resize handling state - strict guards to prevent repeated VTK mutations
        # GPU-safe interaction storm protection (v2)
        # Tracks both resize and camera interaction (drag, wheel)
        self._resizing = False  # True during active resize drag
        self._interacting = False  # True during camera manipulation (mouse drag, wheel)
        self._edges_temporarily_hidden = False  # True if edges were hidden for performance
        self._user_edge_preference = True  # User's desired edge state (restored after interaction)

        # Single timer for restoring edges after BOTH resize and interaction settle
        self._edge_restore_timer = QTimer(self)
        self._edge_restore_timer.setSingleShot(True)
        self._edge_restore_timer.timeout.connect(self._on_interaction_complete)

        # Deferred edge suppression timer – avoids suppressing edges on simple clicks.
        # Edge suppression only fires if the interaction lasts longer than this delay.
        self._edge_suppress_timer = QTimer(self)
        self._edge_suppress_timer.setSingleShot(True)
        self._edge_suppress_timer.setInterval(80)  # 80ms – clicks are typically <50ms
        self._edge_suppress_timer.timeout.connect(self._suppress_edges_for_performance)

        # Defer plotter initialization to prevent freeze
        self._plotter_initialized = False
        self._placeholder = None  # Placeholder widget until plotter is created
        self._setup_ui()
        # Don't create QtInteractor here - wait until widget is shown to prevent freeze
        
        self._legend_manager = None
        self._multi_legend_widget = None
        self._overlay_manager = None
        self._axis_manager = None

        logger.info("Initialized viewer widget")
    
    def showEvent(self, event):
        """Override showEvent to create and initialize plotter when widget becomes visible."""
        super().showEvent(event)
        if self.plotter is None and not self._plotter_initialized:
            # Widget is now visible, create plotter and initialize
            # Use a longer delay to ensure main window initialization completes first
            # This prevents blocking during app.processEvents() calls in main.py
            QTimer.singleShot(1000, self._create_and_initialize_plotter)
    
    def resizeEvent(self, event):
        """
        GPU-safe resize handling with strict state guard (v2).

        Strategy:
        - Hide edges ONCE per resize session (not on every event)
        - Never render during drag
        - Debounce and render once after resize settles (180ms)

        Key fix: Prevent repeated VTK property mutations by guarding edge suppression
        """

        super().resizeEvent(event)

        if self.plotter is None or not self._plotter_initialized:
            return

        # If this is the FIRST resize event in this drag session
        if not self._resizing:
            self._resizing = True

            # Suppress edges ONCE (not on every event)
            self._suppress_edges_for_performance()

        # Restart debounce timer (every event extends the wait)
        # v2: Use 180ms for edge restore after quiet period
        self._edge_restore_timer.start(180)

    def _suppress_edges_for_performance(self):
        """
        Suppress edges ONCE per interaction/resize storm.
        Saves user's edge preference and hides edges temporarily.
        """
        if not self._edges_temporarily_hidden:
            if hasattr(self, "renderer") and self.renderer is not None:
                try:
                    # Save user's current edge preference before hiding
                    current_visibility = self.renderer.get_edge_visibility()
                    self._user_edge_preference = current_visibility

                    # Only hide if edges are currently visible
                    if current_visibility:
                        self.renderer.set_edge_visibility(False)
                        self._edges_temporarily_hidden = True
                        logger.debug("[EDGE PERF] Suppressed edges for interaction/resize storm")
                except Exception:
                    pass

    def _on_interaction_complete(self):
        """
        Finalize interaction/resize (v2):
        - Restore edges to USER'S PREFERENCE (only if they were hidden)
        - Render once
        - Reset state flags

        Called after 180ms of quiet (no resize, no camera drag/wheel).
        """

        if self.plotter is None or not self._plotter_initialized:
            self._resizing = False
            self._interacting = False
            return

        try:
            # Restore edges ONLY if they were temporarily hidden
            if self._edges_temporarily_hidden:
                if hasattr(self, "renderer") and self.renderer is not None:
                    try:
                        # Restore to user's preference, not blindly to True
                        self.renderer.set_edge_visibility(self._user_edge_preference)
                        logger.debug(f"[EDGE PERF] Restored edges to user preference: {self._user_edge_preference}")
                    except Exception:
                        pass
                self._edges_temporarily_hidden = False

            # Single render at final size/position
            self.plotter.render()

        except Exception:
            pass
        finally:
            self._resizing = False
            self._interacting = False
    
    def mousePressEvent(self, event: QMouseEvent):
        """
        Handle mouse press events.
        
        FIX CS-007: Consolidated mouse handling - this is a pass-through handler.
        Primary click detection is done by _PlotterEventFilter on the VTK interactor.
        This handler only stores state for fallback click detection on the Qt widget.
        """
        # Store press state for potential click detection (fallback path)
        if event.button() == Qt.MouseButton.LeftButton:
            import time
            self._click_press_pos = event.position().toPoint()
            self._click_press_time = time.time()
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Handle mouse release events.
        
        FIX CS-007: Consolidated mouse handling - uses PickingController as single authority.
        The _PlotterEventFilter handles clicks on the VTK interactor widget.
        This handler serves as a fallback for clicks that reach the Qt widget directly.
        """
        try:
            is_left = event.button() == Qt.MouseButton.LeftButton
            has_press = hasattr(self, '_click_press_pos') and self._click_press_pos is not None
            
            if is_left and has_press:
                import time
                release_pos = event.position().toPoint()
                elapsed = time.time() - self._click_press_time if hasattr(self, '_click_press_time') else 999
                
                # Check if this was a click (not drag)
                dx = abs(release_pos.x() - self._click_press_pos.x())
                dy = abs(release_pos.y() - self._click_press_pos.y())
                
                if dx < 5 and dy < 5 and elapsed < 0.5:
                    # Use PickingController as single authority for click permission
                    if self._picking_controller.click_allowed:
                        logger.debug(f"Qt fallback click at ({release_pos.x()}, {release_pos.y()})")
                        self._handle_block_click(release_pos.x(), release_pos.y())
                
                # Clear press state
                self._click_press_pos = None
                self._click_press_time = None
        except Exception as e:
            logger.debug(f"Error in mouseReleaseEvent: {e}")
            
        super().mouseReleaseEvent(event)
    
    def enable_interaction(self, cell_count: int = 0, has_block_model: bool = False, has_drillholes: bool = False):
        """
        Enable hover and click interactions.
        
        Called after renderable actors (block model or drillholes) are successfully added.
        Must be called explicitly after data is rendered - interaction is disabled by default.
        
        Args:
            cell_count: Number of cells in the block model (for LOD computation)
            has_block_model: Whether a block model is loaded
            has_drillholes: Whether drillholes are loaded
        """
        if not self._interaction_enabled:
            self._interaction_enabled = True
            # Notify PickingController of data state
            self._picking_controller.on_data_loaded(
                cell_count=cell_count,
                has_block_model=has_block_model,
                has_drillholes=has_drillholes
            )
            logger.info(f"Interaction enabled (LOD={self._picking_controller.lod.name}, cells={cell_count})")
    
    def disable_interaction(self):
        """
        Disable hover and click interactions.
        
        Called when returning to EMPTY state (no renderable data).
        """
        if self._interaction_enabled:
            self._interaction_enabled = False
            # Notify PickingController
            self._picking_controller.on_data_cleared()
            logger.info("Interaction disabled (EMPTY state, LOD-P0)")
    
    @property
    def interaction_enabled(self) -> bool:
        """Return whether hover/click interaction is currently enabled."""
        return self._picking_controller.data_loaded
    
    def _setup_ui(self):
        """Setup the UI layout."""
        # Outer horizontal layout: left = elevation axis widget, right = plotter area
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Elevation axis widget (screen-space, left side)
        try:
            self._elevation_widget = ElevationAxisWidget(self)
        except Exception:
            self._elevation_widget = None

        if self._elevation_widget is not None:
            outer.addWidget(self._elevation_widget, stretch=0)

        # Right-side vertical area: placeholder -> plotter will be inserted here
        right_container = QWidget(self)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Create placeholder widget - defer QtInteractor creation to prevent freeze
        # QtInteractor creation can block on OpenGL context initialization
        self._placeholder = QWidget(right_container)
        self._placeholder.setStyleSheet("background-color: #f0f0f0;")
        self._placeholder.setMinimumSize(400, 400)
        right_layout.addWidget(self._placeholder, stretch=1)

        outer.addWidget(right_container, stretch=1)

        # Plotter will be created when widget is shown
        logger.info("UI setup complete, plotter creation deferred")

    def bind_managers(self, legend_manager=None, overlay_manager=None, axis_manager=None):
        """Bind high-level managers exposed by the controller."""
        self._legend_manager = legend_manager
        self._overlay_manager = overlay_manager
        self._axis_manager = axis_manager
        
        # Create legend manager if not provided but renderer exists
        if self.renderer is not None:
            if legend_manager is None:
                # Create legend manager if it doesn't exist
                from .legend_manager import LegendManager
                if not hasattr(self.renderer, 'legend_manager') or self.renderer.legend_manager is None:
                    legend_manager = LegendManager(self.renderer)
                    self.renderer.legend_manager = legend_manager
                    self._legend_manager = legend_manager
                    logger.info("Created LegendManager and attached to renderer")
            
            # Attach legend manager to renderer so it can be accessed
            if legend_manager is not None:
                self.renderer.legend_manager = legend_manager
        
        # Overlay system is already initialized in renderer.__init__
        # Just ensure overlay_engine is initialized when plotter is ready
        if self.renderer is not None:
            if hasattr(self.renderer, 'overlay_engine') and self.renderer.overlay_engine is not None:
                # Overlay engine already exists, just ensure it's initialized
                if self.renderer.plotter is not None and not hasattr(self.renderer.overlay_engine, '_initialized'):
                    try:
                        self.renderer.overlay_engine.initialize(self.renderer.plotter)
                        logger.info("Initialized overlay_engine with plotter in bind_managers")
                    except Exception as e:
                        logger.warning(f"Could not initialize overlay_engine: {e}", exc_info=True)
                # Set legacy overlay_manager reference for backward compatibility
                self.renderer.overlay_manager = self.renderer.overlay_engine
                self._overlay_manager = self.renderer.overlay_engine
                logger.info("Attached overlay_engine to renderer (via overlay_manager)")
        
        if self._axis_manager and self._elevation_widget is not None:
            try:
                self._axis_manager.bind_elevation_widget(self._elevation_widget)
            except Exception:
                pass
        self._attach_overlay_widgets()
    
    def _install_layer_change_bridge(self) -> None:
        """Ensure renderer layer change events are funneled through this widget."""
        renderer = getattr(self, "renderer", None)
        if renderer is None or not hasattr(renderer, "set_layer_change_callback"):
            return

        original_setter = renderer.set_layer_change_callback

        def bridged_setter(r_self, callback):
            if callback is None:
                return
            ref = self._make_weak_callback(callback)
            # Avoid duplicates
            for existing in list(self._layer_callback_refs):
                try:
                    if existing() is callback:
                        return
                except TypeError:
                    continue
            self._layer_callback_refs.append(ref)

        renderer.set_layer_change_callback = types.MethodType(bridged_setter, renderer)
        try:
            original_setter(self._handle_renderer_layers_changed)
            logger.info("Renderer layer callback bridge installed successfully")
        except Exception as exc:
            try:
                exc_msg = str(exc)
                logger.warning(f"Failed to install renderer layer callback bridge via original_setter: {exc_msg}")
            except Exception:
                logger.warning("Failed to install renderer layer callback bridge via original_setter")

            # CRITICAL FALLBACK: Set callback directly if original_setter fails
            # This ensures layer_change_callback is NEVER None
            try:
                renderer.layer_change_callback = self._handle_renderer_layers_changed
                logger.info("Set renderer.layer_change_callback directly as fallback")
            except Exception as fallback_exc:
                logger.error(f"CRITICAL: Could not set layer_change_callback even as fallback: {fallback_exc}")

        # VERIFY: Check that callback was actually set
        if renderer.layer_change_callback is None:
            logger.error(f"CRITICAL: layer_change_callback is STILL None after bridge setup on renderer {id(renderer)}!")
            # Force set it one more time
            renderer.layer_change_callback = self._handle_renderer_layers_changed
            logger.warning("Force-set layer_change_callback as last resort")
        else:
            logger.info(f"Verified: layer_change_callback is set to {renderer.layer_change_callback}")

    def _make_weak_callback(self, callback):
        try:
            if hasattr(callback, "__self__") and callback.__self__ is not None:
                return weakref.WeakMethod(callback)  # type: ignore[arg-type]
            return weakref.ref(callback)  # type: ignore[arg-type]
        except TypeError:
            return lambda: callback  # noqa: E731 - fallback closure

    def _handle_renderer_layers_changed(self):
        """Dispatch renderer layer changes to Qt listeners and legacy callbacks."""
        logger.debug(f"[STATE DEBUG] _handle_renderer_layers_changed called, {len(self._layer_callback_refs)} callbacks registered")
        
        layers_payload = []
        try:
            scene_layers = getattr(self.renderer, "scene_layers", None)
            if isinstance(scene_layers, dict):
                layers_payload = list(scene_layers.values())
                logger.debug(f"[STATE DEBUG] scene_layers has {len(layers_payload)} layers")
            elif scene_layers is not None:
                layers_payload = list(scene_layers)
                logger.debug(f"[STATE DEBUG] scene_layers (non-dict) has {len(layers_payload)} layers")
            else:
                logger.debug("[STATE DEBUG] scene_layers is None")
        except Exception:
            try:
                exc_msg = str(exc) if 'exc' in locals() else str(e) if 'e' in locals() else '<exception>'
                logger.debug(f"Failed to gather scene layers during update: {exc_msg}")
            except Exception:
                logger.debug("Failed to gather scene layers during update: <unprintable error>")

        try:
            self.layers_changed.emit(layers_payload)
        except Exception as exc:
            try:
                exc_msg = str(exc)
                logger.debug(f"layers_changed emission failed: {exc_msg}")
            except Exception:
                logger.debug("layers_changed emission failed: <unprintable error>")

        survivors = []
        callbacks_called = 0
        for ref in list(self._layer_callback_refs):
            try:
                callback = ref()
            except TypeError:
                callback = None
            if callback is None:
                logger.debug("[STATE DEBUG] Weak callback ref returned None (garbage collected)")
                continue
            survivors.append(ref)
            try:
                callback()
                callbacks_called += 1
            except Exception as exc:
                try:
                    exc_msg = str(exc)
                    logger.debug(f"Legacy layer callback raised: {exc_msg}")
                except Exception:
                    logger.debug("Legacy layer callback raised: <unprintable error>")
        self._layer_callback_refs = survivors
        logger.debug(f"[STATE DEBUG] Called {callbacks_called} callbacks, {len(survivors)} surviving refs")
    
    def _create_and_initialize_plotter(self):
        """Create QtInteractor and initialize plotter after widget is visible."""
        if self._plotter_initialized or self.plotter is not None:
            logger.debug(f"Skipping plotter creation: initialized={self._plotter_initialized}, plotter={self.plotter is not None}")
            return
        
        try:
            # Ensure widget and parent window are visible before creating QtInteractor
            if not self.isVisible():
                logger.debug("Widget not visible yet, retrying in 200ms...")
                QTimer.singleShot(200, self._create_and_initialize_plotter)
                return
            
            # Check if parent window is ready
            parent = self.window()
            if parent is None or not parent.isVisible():
                logger.debug("Parent window not visible yet, retrying in 200ms...")
                QTimer.singleShot(200, self._create_and_initialize_plotter)
                return
            
            logger.info("Creating QtInteractor...")
            
            # Don't process events here - let the main event loop handle it
            # Processing events here can cause nested event loops and hangs
            
            # Create QtInteractor in a separate step with error handling
            try:
                logger.info("Initializing PyVista QtInteractor (this may take a moment)...")
                # Use QTimer to create in next event loop iteration to avoid blocking
                # Use a longer delay to ensure main window initialization completes first
                QTimer.singleShot(500, lambda: self._create_qtinteractor_actual())
            except Exception as e:
                logger.error(f"Failed to schedule QtInteractor creation: {e}", exc_info=True)
                raise
            
        except Exception as e:
            logger.error(f"Failed to create and initialize plotter: {e}", exc_info=True)
    
    def _create_qtinteractor_actual(self):
        """Actually create the QtInteractor - called after delay."""
        if self.plotter is not None:
            return
        
        try:
            logger.info("Creating QtInteractor now...")
            # Don't process events here - we're already in an event handler
            # Processing events here can cause nested event loops
            
            # Create QtInteractor - this can block if OpenGL context isn't ready
            self.plotter = pvqt.QtInteractor(self)
            self.plotter.interactor.setSizePolicy(
                QWidget().sizePolicy().Policy.Expanding,
                QWidget().sizePolicy().Policy.Expanding
            )
            
            # FIX: Ensure render window erases background between frames (prevents ghosting)
            try:
                render_window = self.plotter.render_window
                if render_window:
                    render_window.SetEraseBackground(True)
                    render_window.SetErase(True)
                    logger.debug("Enabled render window background erase")
            except Exception as e:
                logger.debug(f"Could not set render window erase: {e}")
            
            # Replace placeholder with plotter in the right-side layout
            if self._placeholder:
                try:
                    placeholder_parent = self._placeholder.parent()
                    parent_layout = None
                    if placeholder_parent is not None:
                        parent_layout = getattr(placeholder_parent, 'layout', lambda: None)()
                    # Fallback to this widget's layout
                    if parent_layout is None:
                        parent_layout = self.layout()
                    if parent_layout is not None:
                        parent_layout.removeWidget(self._placeholder)
                    self._placeholder.deleteLater()
                    self._placeholder = None
                    # Add the plotter interactor into the same parent layout
                    if parent_layout is not None:
                        parent_layout.addWidget(self.plotter.interactor, stretch=1)
                    # Don't process events here - let the main event loop handle it
                except Exception:
                    # Best-effort replacement; if it fails, at least set plotter parent
                    try:
                        self.plotter.interactor.setParent(self)
                    except Exception:
                        pass
            
            logger.info("QtInteractor created successfully")
            
            # Process events again
            QApplication.processEvents()
            
            # Now initialize the renderer with additional delay
            QTimer.singleShot(100, self._initialize_plotter_deferred)
            
            # Create legend widget after a longer delay to ensure everything is ready
            QTimer.singleShot(300, self._create_legend_widget)
            
            # Install interaction event filter (double-click finish for area tool)
            # Note: Single click picking is now handled by ViewerWidget.mouseReleaseEvent directly
            try:
                self._plotter_event_filter = _PlotterEventFilter(self)
                # Only install on main interactor for double-click handling
                try:
                    self.plotter.interactor.installEventFilter(self._plotter_event_filter)
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Could not install plotter event filter: {e}")
            
            # Initialize hover debouncer for high-performance picking
            try:
                self._hover_debouncer = HoverDebouncer(interval_ms=50, parent=self)
                self._hover_debouncer.hover_stable.connect(self._on_hover_stable)
                
                # Attach mouse move observer to VTK interactor
                interactor = self.plotter.interactor
                if interactor is not None:
                    # Store callback reference to prevent garbage collection
                    self._mouse_move_callback = lambda obj, event: self._on_mouse_move(obj, event)
                    interactor.AddObserver("MouseMoveEvent", self._mouse_move_callback)
                    
                    # Attach left button click observer for block clicking
                    self._mouse_click_callback = lambda obj, event: self._on_mouse_click(obj, event)
                    
                    # VTK AddObserver signature: AddObserver(event, callback, priority)
                    # Priority is positional, not keyword. Lower = higher priority.
                    observer_id1 = interactor.AddObserver("LeftButtonPressEvent", self._mouse_click_callback, -1.0)
                    observer_id2 = interactor.AddObserver("LeftButtonReleaseEvent", self._on_mouse_release, -1.0)
                    
                    logger.info(f"Mouse observers attached (observer IDs: {observer_id1}, {observer_id2})")
                    
                    # Test: Log all available events to see what's available
                    logger.debug(f"Interactor type: {type(interactor)}")
                else:
                    logger.warning("Cannot attach hover debouncer: interactor is None")
            except Exception as e:
                logger.warning(f"Could not initialize hover debouncer: {e}", exc_info=True)
            # Install an EndInteractionEvent observer to update screen-space widgets
            # Use throttling to prevent lag during interaction
            try:
                iren = getattr(self.plotter, 'iren', None) or getattr(self.plotter, 'interactor', None)
                if iren is not None:
                    # Throttle updates to prevent lag - only update after interaction stops
                    self._interaction_update_timer = QTimer()
                    self._interaction_update_timer.setSingleShot(True)
                    self._interaction_update_timer.timeout.connect(self._deferred_update_widgets)
                    self._interaction_update_timer.setInterval(100)  # Update 100ms after interaction ends
                    
                    def _on_end_interaction(*args, **kwargs):
                        # Don't update immediately - use timer to debounce/throttle updates
                        # This prevents lag during continuous interaction (dragging, rotating)
                        try:
                            # Restart timer - this ensures updates only happen after interaction stops
                            self._interaction_update_timer.stop()
                            self._interaction_update_timer.start()

                            # v2: Restart edge restore timer - restores edges after 180ms of quiet
                            if self._interacting:
                                # Cancel deferred suppression if it hasn't fired yet (simple click)
                                if self._edge_suppress_timer.isActive():
                                    self._edge_suppress_timer.stop()
                                    logger.debug("[INTERACTION STORM] Quick interaction (click) – edge suppression skipped")
                                    self._interacting = False
                                else:
                                    self._edge_restore_timer.stop()
                                    self._edge_restore_timer.start(180)
                                    logger.debug("[INTERACTION STORM] Camera interaction ended, edge restore timer started")
                        except Exception:
                            pass
                    # store to keep a reference alive
                    self._end_interaction_callback = _on_end_interaction
                    try:
                        # vtk uses AddObserver, pyvista wrappers sometimes expose add_observer
                        if hasattr(iren, 'AddObserver'):
                            iren.AddObserver('EndInteractionEvent', _on_end_interaction)
                        elif hasattr(iren, 'add_observer'):
                            iren.add_observer('EndInteractionEvent', _on_end_interaction)
                    except Exception:
                        pass

                    # v2: Add StartInteractionEvent observer to suppress edges during camera manipulation
                    def _on_start_interaction(*args, **kwargs):
                        """Suppress edges when camera interaction begins (drag, wheel)."""
                        try:
                            if not self._interacting:
                                self._interacting = True
                                # Defer edge suppression – if the interaction ends quickly
                                # (simple click), the timer is cancelled before it fires.
                                self._edge_suppress_timer.start()
                                logger.debug("[INTERACTION STORM] Camera interaction started, edge suppression deferred")
                        except Exception:
                            pass

                    self._start_interaction_callback = _on_start_interaction
                    try:
                        if hasattr(iren, 'AddObserver'):
                            iren.AddObserver('StartInteractionEvent', _on_start_interaction)
                        elif hasattr(iren, 'add_observer'):
                            iren.add_observer('StartInteractionEvent', _on_start_interaction)
                    except Exception:
                        pass
            except Exception:
                pass
            # Save original interactor style so we can restore it later
            try:
                inter = getattr(self.plotter, 'iren', None) or getattr(self.plotter, 'interactor', None)
                if inter is not None:
                    try:
                        self._original_interactor_style = inter.GetInteractorStyle()
                    except Exception:
                        # Some wrappers expose style differently
                        self._original_interactor_style = getattr(inter, 'InteractorStyle', None)
            except Exception:
                self._original_interactor_style = None
            # Trigger an initial elevation widget update after plotter exists
            try:
                QTimer.singleShot(200, lambda: self._update_elevation_widget())
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Failed to create QtInteractor: {e}", exc_info=True)
            # Don't raise - allow window to stay open even if plotter fails
    
    def _initialize_plotter_deferred(self):
        """Initialize plotter after widget is visible to prevent freeze."""
        if self._plotter_initialized or self.plotter is None:
            logger.debug(f"Skipping plotter init: initialized={self._plotter_initialized}, plotter={self.plotter is not None}")
            return
        
        try:
            # Ensure widget is visible before initializing
            if not self.isVisible():
                logger.debug("Widget not visible yet, retrying in 100ms...")
                QTimer.singleShot(100, self._initialize_plotter_deferred)
                return
            
            logger.info("Starting deferred plotter initialization...")
            
            # Process events to keep UI responsive during initialization
            QApplication.processEvents()
            
            logger.info("Initializing renderer plotter...")
            self.renderer.initialize_plotter(self.plotter, parent_window=self.plotter.interactor)
            
            # Ensure overlay manager is attached to renderer
            if hasattr(self.renderer, 'overlay_manager') and self.renderer.overlay_manager is not None:
                try:
                    if self.renderer.overlay_manager._renderer_ref is None:
                        self.renderer.overlay_manager.attach_renderer(self.renderer)
                        logger.info("Attached overlay_manager to renderer in _create_and_initialize_plotter")
                except Exception as e:
                    logger.warning(f"Could not initialize overlay_manager with plotter: {e}", exc_info=True)
            
            # Process events again
            QApplication.processEvents()
            
            # Enable global picking across all scene layers (blocks, drillholes, etc.)
            logger.info("Setting up global picking...")
            try:
                # NOTE: We're using custom event filter + _handle_block_click for clicking
                # The PyVista enable_point_picking causes double-picking and sometimes picks wrong blocks
                # So we disable it and rely on our custom system
                logger.info("Skipping PyVista enable_point_picking (using custom event filter instead)")
                # Still set callback in case other systems need it, but don't enable PyVista picking
                self.renderer.set_pick_callback(self._on_global_pick)
                # Don't call setup_global_picking() - it enables PyVista's picking which conflicts
                logger.info("Global picking setup complete (custom system only)")
            except Exception as e:
                logger.warning(f"Could not setup global picking: {e}")
            
            # Final event processing
            QApplication.processEvents()
            
            logger.info("Setting up plotter...")
            self._setup_plotter()
            self._plotter_initialized = True
            logger.info("Plotter initialized successfully after widget shown")
            
            # Ensure elevation widget reflects initial scene bounds
            try:
                self._update_elevation_widget()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to initialize plotter: {e}", exc_info=True)
    
    def _setup_plotter(self):
        """Setup the PyVista plotter."""
        if self.plotter is None or not self._plotter_initialized:
            return
        try:
            # Use background color from renderer if available, otherwise default to white
            bg_color = 'white'
            if hasattr(self, 'renderer') and hasattr(self.renderer, 'background_color'):
                bg_color = self.renderer.background_color
            
            self.plotter.set_background(bg_color)
            
            # Do not enable anti-aliasing; keep block edges crisp
            if not getattr(self.renderer, "use_overlays", False):
                # Add world axes and bounding box. Capture return value where possible
                try:
                    axes_actor = None
                    try:
                        # PyVista axes removed - using custom overlay system instead
                        axes_actor = None
                    except Exception:
                        # Some pyvista builds may not return the actor; call without capture
                        try:
                            # PyVista axes removed - using custom overlay system instead
                            pass
                        except Exception:
                            axes_actor = None

                    # Attempt to make axis captions upright (defensive)
                    if axes_actor is not None:
                        try:
                            # Common VTK caption actor accessors
                            xa = getattr(axes_actor, 'GetXAxisCaptionActor2D', None)
                            ya = getattr(axes_actor, 'GetYAxisCaptionActor2D', None)
                            za = getattr(axes_actor, 'GetZAxisCaptionActor2D', None)
                            if xa is not None:
                                try:
                                    t = xa()
                                    # Some caption actors expose GetTextActor
                                    get_text = getattr(t, 'GetTextActor', None)
                                    if get_text is not None:
                                        txt = get_text()
                                        if hasattr(txt, 'SetOrientation'):
                                            txt.SetOrientation(0)
                                except Exception:
                                    pass
                            if ya is not None:
                                try:
                                    t = ya()
                                    get_text = getattr(t, 'GetTextActor', None)
                                    if get_text is not None:
                                        txt = get_text()
                                        if hasattr(txt, 'SetOrientation'):
                                            txt.SetOrientation(0)
                                except Exception:
                                    pass
                            if za is not None:
                                try:
                                    t = za()
                                    get_text = getattr(t, 'GetTextActor', None)
                                    if get_text is not None:
                                        txt = get_text()
                                        if hasattr(txt, 'SetOrientation'):
                                            txt.SetOrientation(0)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                except Exception:
                    logger.debug('Could not add axes actor')

                try:
                    # Disable PyVista's built-in cube axes/bounds labels so our
                    # overlay manager provides a single, consistent set of
                    # upright/readable labels. Use a defensive removal so this
                    # works across different pyvista/vtk versions.
                    try:
                        # Preferred API (if available)
                        if hasattr(self.plotter, 'remove_bounds_axes'):
                            try:
                                self.plotter.remove_bounds_axes()
                            except Exception:
                                pass
                        else:
                            # Best-effort: attempt to remove any bounds actor
                            try:
                                # Some pyvista builds expose _bounds_actor on the plotter
                                b_actor = getattr(self.plotter, '_bounds_actor', None)
                                if b_actor is not None:
                                    try:
                                        self.plotter.remove_actor(b_actor)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    pass
            self.plotter.track_mouse_position()
            # Global picking is enabled via renderer; use custom click highlight below
            self._setup_block_picking()
            
            # Note: We'll use VTK observer for clicking instead of enable_cell_picking
            # because enable_cell_picking conflicts with camera rotation
            # The click handler is set up via VTK observer in hover debouncer initialization
            # Try to prevent PyVista/VTK from auto-tilting text by nudging camera/axes
            try:
                cam = getattr(self.plotter, 'camera', None)
                if cam is not None:
                    try:
                        # Set a reasonable view angle and zero roll to keep labels upright
                        if hasattr(cam, 'SetViewAngle'):
                            cam.SetViewAngle(35)
                        if hasattr(cam, 'SetRoll'):
                            cam.SetRoll(0)
                    except Exception:
                        pass
                # Reset camera to apply changes
                try:
                    if hasattr(self.plotter, 'reset_camera'):
                        self.plotter.reset_camera()
                except Exception:
                    pass
            except Exception:
                pass
            logger.info("Setup PyVista plotter")
        except Exception as e:
            logger.error(f"Error setting up plotter: {e}", exc_info=True)

    def _update_elevation_widget(self):
        """Refresh the ElevationAxisWidget from current scene bounds and overlay state."""
        try:
            if getattr(self, '_elevation_widget', None) is None:
                return
            # Cache bounds to avoid expensive ComputeVisiblePropBounds() calls during interaction
            import time
            current_time = time.time()
            bounds = None
            # Use cached bounds if available and recent (cache for 0.5 seconds)
            if self._cached_bounds is not None and (current_time - self._bounds_cache_time) < 0.5:
                bounds = self._cached_bounds
            else:
                # Prefer renderer-derived scene bounds
                try:
                    if self.renderer and hasattr(self.renderer, '_get_scene_bounds'):
                        bounds = self.renderer._get_scene_bounds()
                        # Cache the bounds
                        if bounds:
                            self._cached_bounds = bounds
                            self._bounds_cache_time = current_time
                    else:
                        bounds = None
                except Exception:
                    bounds = None
                # Fallback to current model bounds
                if not bounds and self.current_model is not None:
                    try:
                        bounds = getattr(self.current_model, 'bounds', None) or getattr(self.current_model, 'get_bounds', lambda: None)()
                        # Cache the bounds
                        if bounds:
                            self._cached_bounds = bounds
                            self._bounds_cache_time = current_time
                    except Exception:
                        bounds = None

            # Show/hide based on renderer axes/overlay state
            show_axes = getattr(self.renderer, 'show_axes', True)
            if show_axes and bounds:
                try:
                    self._elevation_widget.show_for_bounds(bounds)
                except Exception:
                    pass
            else:
                try:
                    self._elevation_widget.hide()
                except Exception:
                    pass
        except Exception:
            pass
    
    def _deferred_update_widgets(self):
        """Update elevation widget and HUD overlay after interaction ends (throttled)."""
        try:
            self._update_elevation_widget()
        except Exception:
            pass
    
    def _create_legend_widget(self):
        """Create legend widget after plotter is initialized."""
        try:
            # Ensure legend manager exists
            if self._legend_manager is None:
                if self.renderer is not None:
                    from .legend_manager import LegendManager
                    self._legend_manager = LegendManager(self.renderer)
                    self.renderer.legend_manager = self._legend_manager
                    logger.info("Created LegendManager in _create_legend_widget")
                else:
                    logger.warning("Cannot create legend widget: renderer is None")
                    return

            # Create the legend widget directly as a floating overlay
            if self._legend_manager.widget is None:
                from .legend_widget import LegendWidget

                # Get the main window as parent for the floating legend widget
                parent_window = self.window() if self.window() else None
                if parent_window is None:
                    logger.warning("Cannot create legend widget: no parent window found")
                    return

                # Create the legend widget
                legend_widget = LegendWidget(parent=parent_window)

                # Bind it to the legend manager (this will apply layout and positioning)
                self._legend_manager.bind_widget(legend_widget)

                # Connect legend's colormap_changed signal to update renderer and property panel
                try:
                    legend_widget.colormap_changed.connect(self._on_legend_colormap_changed)
                    logger.info("Connected legend colormap_changed signal")
                except Exception as e:
                    logger.warning(f"Could not connect legend colormap_changed signal: {e}")

                # Ensure widget is visible and raised - must be done after setting parent
                # Note: Window flags are set in LegendWidget.__init__, so we don't need to set them again
                legend_widget.setVisible(True)
                legend_widget.show()
                legend_widget.raise_()

                # Force update to ensure it renders and positioning is applied
                legend_widget.update()
                parent_window.update()

                # Re-apply layout to ensure correct positioning after parent is set
                try:
                    legend_widget.apply_layout(
                        anchor=self._legend_manager._state.anchor,
                        position=self._legend_manager._state.position,
                        margin=self._legend_manager._state.margin,
                        size=self._legend_manager._state.size
                    )
                except Exception as e:
                    logger.debug(f"Could not re-apply layout: {e}")

                logger.info(f"Created and showed legend widget (parent={parent_window}, visible={legend_widget.isVisible()})")

            # Also check if legend manager already has a widget but it's not visible
            elif self._legend_manager.widget is not None:
                widget = self._legend_manager.widget
                if not widget.isVisible():
                    widget.setVisible(True)
                    widget.show()
                    widget.raise_()
                    widget.update()
                    logger.info("Showed existing legend widget")
                else:
                    logger.debug("Legend widget already visible")

            # Create multi-legend widget for interactive add/remove
            self._create_multi_legend_widget()

            self._attach_overlay_widgets()
        except Exception as e:
            logger.error(f"Failed to create legend widget: {e}", exc_info=True)

    def _create_multi_legend_widget(self):
        """Create multi-legend widget for interactive legend management."""
        try:
            if self._multi_legend_widget is not None:
                return  # Already created

            parent_window = self.window() if self.window() else None
            if parent_window is None:
                logger.warning("Cannot create multi-legend widget: no parent window found")
                return

            from .multi_legend_widget import MultiLegendWidget

            # Create the multi-legend widget as a floating window
            # Set window flags FIRST before setting parent to ensure it's a floating overlay
            self._multi_legend_widget = MultiLegendWidget()
            self._multi_legend_widget.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.Tool |
                Qt.WindowType.WindowStaysOnTopHint
            )
            self._multi_legend_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self._multi_legend_widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

            # Now set the parent (for proper window management)
            self._multi_legend_widget.setParent(parent_window)
            # Re-apply window flags after setting parent (PyQt6 requirement)
            self._multi_legend_widget.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.Tool |
                Qt.WindowType.WindowStaysOnTopHint
            )

            # Position in bottom-left corner of the viewer area
            margin = 24
            x = margin
            y = parent_window.height() - self._multi_legend_widget.height() - margin - 100
            self._multi_legend_widget.move(x, y)

            # Bind to legend manager for add/remove functionality
            if self._legend_manager is not None:
                self._legend_manager.bind_multi_widget(self._multi_legend_widget)

            # Initially hide - user can show via menu or keyboard shortcut
            self._multi_legend_widget.hide()

            logger.info("Created multi-legend widget as floating overlay")
        except Exception as e:
            logger.error(f"Failed to create multi-legend widget: {e}", exc_info=True)

    def _ensure_scale_bar_widget(self):
        """Create renderer-managed scale bar widget if it does not yet exist."""
        if self.renderer is None:
            return None
        widget = getattr(self.renderer, 'scale_bar_widget', None)

        # Correct parent is the plotter (QtInteractor) so widget appears on top of VTK
        correct_parent = self.plotter if self.plotter is not None else self

        # If widget exists, ensure it has correct parent
        if widget is not None:
            if widget.parent() != correct_parent and correct_parent is not None:
                widget.setParent(correct_parent)
                logger.info(f"Reparented scale bar widget to {correct_parent.__class__.__name__}")
            return widget

        # Create new widget
        try:
            widget = ScaleBarWidget(parent=correct_parent)
            widget.hide()
            if hasattr(self.renderer, 'attach_scale_bar_widget'):
                self.renderer.attach_scale_bar_widget(widget)
            else:
                self.renderer.scale_bar_widget = widget
            logger.info(f"Created scale bar widget with parent={correct_parent.__class__.__name__}")
        except Exception as exc:
            try:
                logger.warning(f"Could not create scale bar widget: {exc}")
            except Exception:
                logger.warning("Could not create scale bar widget: <unprintable error>")
            return None
        return widget

    def _ensure_north_arrow_widget(self):
        """Create renderer-managed north arrow widget if it does not yet exist."""
        if self.renderer is None:
            return None
        widget = getattr(self.renderer, 'north_arrow_widget', None)

        # Correct parent is the plotter (QtInteractor) so widget appears on top of VTK
        correct_parent = self.plotter if self.plotter is not None else self

        # If widget exists, ensure it has correct parent
        if widget is not None:
            if widget.parent() != correct_parent and correct_parent is not None:
                widget.setParent(correct_parent)
                logger.info(f"Reparented north arrow widget to {correct_parent.__class__.__name__}")
            return widget

        # Create new widget
        try:
            widget = NorthArrowWidget(parent=correct_parent)
            widget.hide()
            if hasattr(self.renderer, 'attach_north_arrow_widget'):
                self.renderer.attach_north_arrow_widget(widget)
            else:
                self.renderer.north_arrow_widget = widget
            logger.info(f"Created north arrow widget with parent={correct_parent.__class__.__name__}")
        except Exception as exc:
            try:
                logger.warning(f"Could not create north arrow widget: {exc}")
            except Exception:
                logger.warning("Could not create north arrow widget: <unprintable error>")
            return None
        return widget

    def toggle_multi_legend(self, visible: Optional[bool] = None) -> bool:
        """
        Toggle or set the legend's multi-element mode.

        In multi-mode, the legend shows a toolbar with [+] button
        and allows adding/removing multiple legend elements.

        Args:
            visible: If provided, set multi-mode to this value. If None, toggle.

        Returns:
            The new multi-mode state.
        """
        # Ensure legend widget exists - create it if needed
        if self._legend_manager is None or self._legend_manager.widget is None:
            logger.info("Creating legend widget for multi-mode")
            self._create_legend_widget()

        # Get reference to the legend widget
        legend_widget = None
        if self._legend_manager is not None and hasattr(self._legend_manager, 'widget'):
            legend_widget = self._legend_manager.widget

        if legend_widget is None:
            logger.warning("No legend widget available for multi-mode after creation attempt")
            return False

        # Toggle or set multi-mode
        if visible is None:
            visible = not legend_widget.is_multi_mode()

        legend_widget.enable_multi_mode(visible)

        # Ensure legend is visible and properly positioned
        if visible:
            legend_widget.show()
            legend_widget.raise_()

            # Connect add_element_requested signal to show dialog
            if not hasattr(self, '_multi_mode_signals_connected'):
                legend_widget.add_element_requested.connect(self._on_add_legend_element_requested)
                self._multi_mode_signals_connected = True

        logger.info(f"Legend multi-mode {'enabled' if visible else 'disabled'}")
        return visible

    def _on_add_legend_element_requested(self):
        """Handle add element request from legend widget."""
        try:
            from .legend_add_dialog import LegendAddDialog

            dialog = LegendAddDialog(self.renderer, self.window())
            dialog.item_selected.connect(self._on_legend_dialog_item_selected)
            dialog.exec()
        except Exception as e:
            logger.error(f"Failed to show add legend dialog: {e}")

    def _on_legend_dialog_item_selected(self, layer_name: str, property_name: Optional[str], is_discrete: bool):
        """Handle selection from add legend dialog."""
        if self._legend_manager is not None:
            self._legend_manager.add_legend_for_layer(layer_name, property_name)

    def is_multi_legend_visible(self) -> bool:
        """Check if legend is in multi-element mode."""
        if self._legend_manager is not None and hasattr(self._legend_manager, 'widget'):
            legend_widget = self._legend_manager.widget
            return legend_widget is not None and legend_widget.is_multi_mode()
        return False

    def _attach_overlay_widgets(self):
        """Register renderer-created overlay widgets with the overlay engine."""
        # Attach scale bar widget - create first, then optionally bind to managers
        try:
            if self.renderer is not None:
                scale_bar = self._ensure_scale_bar_widget()
                if scale_bar is not None:
                    logger.info("Attached scale bar widget")
                    # Optionally bind to overlay managers if they exist
                    try:
                        if hasattr(self.renderer, 'overlay_engine') and self.renderer.overlay_engine is not None:
                            attach_widget = getattr(self.renderer.overlay_engine, 'attach_widget', None)
                            if callable(attach_widget):
                                attach_widget('scale_bar', scale_bar)
                        if hasattr(self.renderer, 'overlay_state') and self.renderer.overlay_state is not None:
                            self.renderer.overlay_state.attach_widget('scale_bar', scale_bar)
                        if self._axis_manager is not None:
                            self._axis_manager.bind_scale_bar_widget(scale_bar)
                    except Exception:
                        pass  # Optional bindings, don't fail widget creation
        except Exception as e:
            logger.warning(f"Could not attach scale bar widget: {e}", exc_info=True)
        # Attach north arrow widget
        try:
            if self.renderer is not None:
                north_arrow = self._ensure_north_arrow_widget()
                if north_arrow is not None:
                    logger.info("Attached north arrow widget")
        except Exception as e:
            logger.warning(f"Could not attach north arrow widget: {e}", exc_info=True)
        try:
            if self._axis_manager is not None and self.renderer is not None:
                coord_widget = getattr(self.renderer, 'coordinate_display_widget', None)
                if coord_widget is not None:
                    self._axis_manager.bind_coordinate_widget(coord_widget)
        except Exception:
            pass
    
    def _setup_block_picking(self):
        """Enable interactive block picking with cursor-based tooltips."""
        if self.plotter is None:
            return
        try:
            self.plotter.enable_trackball_style()
        except Exception:
            pass  # May already be set
        try:
            self.renderer.configure_tooltip_properties(self.properties_to_show)
        except Exception:
            pass
        logger.info("Configured block picking via renderer-managed pipeline")

    def set_interaction_mode(self, mode: str) -> None:
        """
        High-level API to set the viewer's interaction mode.

        Supported modes: 'pan', 'rotate'/'trackball', 'select', 'zoom'/'zoom_box'.
        This method is defensive: it first tries to set a VTK interactor
        style on the available interactor; if no interactor is present it
        will still update the cursor and invoke renderer-level picking
        handlers where appropriate.
        """
        try:
            # Normalize mode
            m = (mode or "").lower()
            plotter = getattr(self, 'plotter', None)
            interactor = None
            if plotter is not None:
                interactor = getattr(plotter, 'iren', None) or getattr(plotter, 'interactor', None)
            # Also accept renderer-held plotter-like attribute
            if interactor is None:
                interactor = getattr(self.renderer, 'interactor', None)

            # Map to VTK interactor styles where available
            style_obj = None
            try:
                # Prefer vtkmodules (packaging-friendly)
                from vtkmodules.vtkInteractionStyle import (
                    vtkInteractorStyleTrackballCamera,
                    vtkInteractorStyleImage,
                    vtkInteractorStyleRubberBandCamera,
                    vtkInteractorStyleTrackballActor,
                )
            except Exception:
                try:
                    import vtk
                    vtkInteractorStyleTrackballCamera = vtk.vtkInteractorStyleTrackballCamera
                    vtkInteractorStyleImage = vtk.vtkInteractorStyleImage
                    vtkInteractorStyleRubberBandCamera = vtk.vtkInteractorStyleRubberBandCamera
                    vtkInteractorStyleTrackballActor = vtk.vtkInteractorStyleTrackballActor
                except Exception:
                    vtkInteractorStyleTrackballCamera = vtkInteractorStyleImage = vtkInteractorStyleRubberBandCamera = vtkInteractorStyleTrackballActor = None

            if m in ("rotate", "trackball"):
                if vtkInteractorStyleTrackballCamera is not None:
                    style_obj = vtkInteractorStyleTrackballCamera()
                cursor = Qt.CursorShape.ArrowCursor
            elif m == "pan":
                if vtkInteractorStyleImage is not None:
                    style_obj = vtkInteractorStyleImage()
                cursor = Qt.CursorShape.OpenHandCursor
            elif m in ("zoom", "zoom_box", "zoombox"):
                if vtkInteractorStyleRubberBandCamera is not None:
                    style_obj = vtkInteractorStyleRubberBandCamera()
                cursor = Qt.CursorShape.SizeAllCursor
            elif m in ("select", "click"):
                # Selection uses trackball camera style but enables picking callbacks
                if vtkInteractorStyleTrackballCamera is not None:
                    style_obj = vtkInteractorStyleTrackballCamera()
                cursor = Qt.CursorShape.PointingHandCursor
            else:
                # Default to trackball
                if vtkInteractorStyleTrackballCamera is not None:
                    style_obj = vtkInteractorStyleTrackballCamera()
                cursor = Qt.CursorShape.ArrowCursor

            # Support restoring original/default interactor style
            if m in ("original", "default", "reset"):
                cursor = Qt.CursorShape.ArrowCursor
                # Try to restore previously saved style object
                try:
                    inter = getattr(plotter, 'iren', None) or getattr(plotter, 'interactor', None) or getattr(self.renderer, 'interactor', None)
                    orig = getattr(self, '_original_interactor_style', None)
                    if inter is not None and orig is not None:
                        try:
                            inter.SetInteractorStyle(orig)
                        except Exception:
                            try:
                                setattr(inter, 'InteractorStyle', orig)
                            except Exception:
                                pass
                        # ensure picking disabled when restoring to original
                        try:
                            if hasattr(self.renderer, 'disable_picking'):
                                self.renderer.disable_picking()
                        except Exception:
                            pass
                except Exception:
                    pass

            # Apply style to interactor if available
            if interactor is not None and style_obj is not None:
                try:
                    interactor.SetInteractorStyle(style_obj)
                except Exception:
                    try:
                        # Some wrappers require setting attribute instead
                        setattr(interactor, 'InteractorStyle', style_obj)
                    except Exception:
                        pass

            # Update cursor on the viewer widget regardless
            try:
                self.setCursor(cursor)
            except Exception:
                pass

            # Selection mode should also ensure renderer picking is active
            if m in ("select", "click"):
                try:
                    if self.renderer is not None:
                        # Ensure pick callback is wired; ViewerWidget provides _on_global_pick
                        if hasattr(self, '_on_global_pick'):
                            try:
                                self.renderer.set_pick_callback(self._on_global_pick)
                            except Exception:
                                pass
                        try:
                            self.renderer.setup_global_picking()
                        except Exception:
                            pass
                except Exception:
                    pass

            # Persist current mouse mode on renderer for other components
            try:
                if self.renderer is not None:
                    self.renderer._current_mouse_mode = m
            except Exception:
                pass

            # Emit a high-level signal so UI can update (menu checks, status bar)
            try:
                # Emit normalized mode string
                self.mouse_mode_changed.emit(m)
            except Exception:
                pass

        except Exception as e:
            logger.debug("set_interaction_mode failed: %s", e)
    
    def _initialize_picking_state(self):
        """Initialize the state needed for block picking after loading a model."""
        import time
        pick_start = time.time()
        if self.renderer is None:
            return
        
        try:
            # Get the display grid from renderer
            if 'unstructured_grid' in self.renderer.block_meshes:
                self.display_grid = self.renderer.block_meshes['unstructured_grid']
                
                if self.display_grid is not None and self.display_grid.GetNumberOfCells() > 0:
                    pick_time = time.time() - pick_start
                    logger.info(f"PERF: _initialize_picking_state took {pick_time:.3f}s")
                    logger.info(f"Initialized picking state: {self.display_grid.GetNumberOfCells()} cells")
                else:
                    logger.warning("Display grid has no cells")
            else:
                logger.warning("No unstructured_grid found in renderer")
                
        except Exception as e:
            logger.error(f"Error initializing picking state: {e}")
    
    def clear_block_selection(self):
        """Clear the selected block highlight and tooltip."""
        try:
            if self.renderer:
                self.renderer.clear_pick_overlays(render=True)
                logger.info("Cleared block selection and tooltip")
        except Exception as e:
            logger.debug(f"Error clearing selection: {e}")
    
    def highlight_swath_interval(self, axis: str, lower: float, upper: float):
        """
        Highlight a 3D interval (swath bin) in the viewer.
        
        Args:
            axis: Axis direction ('X', 'Y', or 'Z')
            lower: Lower bound of interval
            upper: Upper bound of interval
        """
        if self.plotter is None or self.display_grid is None:
            logger.warning("Cannot highlight swath: plotter or grid not initialized")
            return
        
        try:
            # Remove previous swath highlight
            self.plotter.remove_actor('swath_highlight', reset_camera=False, render=False)
            self.plotter.remove_actor('swath_label', reset_camera=False, render=False)
            
            # Get grid bounds
            bounds = self.display_grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            
            # Create a translucent box for the swath interval
            if axis == "X":
                box_bounds = (lower, upper, bounds[2], bounds[3], bounds[4], bounds[5])
            elif axis == "Y":
                box_bounds = (bounds[0], bounds[1], lower, upper, bounds[4], bounds[5])
            else:  # Z
                box_bounds = (bounds[0], bounds[1], bounds[2], bounds[3], lower, upper)
            
            # Create box mesh
            swath_box = pv.Box(bounds=box_bounds)
            
            # Add to plotter with semi-transparent orange
            self.plotter.add_mesh(
                swath_box,
                color='orange',
                opacity=0.3,
                show_edges=True,
                edge_color='darkorange',
                line_width=2.0,
                name='swath_highlight',
                pickable=False,
                reset_camera=False,
                render=False
            )
            
            # Add text label
            label_text = f"Swath: {axis} = [{lower:.1f}, {upper:.1f}] m"
            self.plotter.add_text(
                label_text,
                name='swath_label',
                position='upper_left',
                font_size=11,
                color='darkorange'
            )
            
            self.plotter.render()
            
            logger.info(f"Highlighted swath interval: {axis} [{lower:.2f}, {upper:.2f}]")
            
        except Exception as e:
            logger.error(f"Error highlighting swath interval: {e}")
    
    def clear_swath_highlight(self):
        """Clear swath interval highlight."""
        if self.plotter is None:
            return
        
        try:
            self.plotter.remove_actor('swath_highlight', reset_camera=False, render=False)
            self.plotter.remove_actor('swath_label', reset_camera=False, render=False)
            self.plotter.render()
            logger.debug("Cleared swath highlight")
        except Exception as e:
            logger.debug(f"Error clearing swath highlight: {e}")
    
    def load_block_model(self, block_model: BlockModel):
        """
        Load a block model for visualization.
 
        Args:
            block_model: BlockModel to visualize
        """
        import time
        widget_start = time.time()
        self._prepare_block_model(block_model)
 
        # Load into renderer
        renderer_start = time.time()
        self.renderer.load_block_model(block_model)
        renderer_time = time.time() - renderer_start
        logger.info(f"PERF: Renderer.load_block_model took {renderer_time:.3f}s")
 
        self._post_load_updates(block_model, source="ViewerWidget.load_block_model")
        widget_time = time.time() - widget_start
        logger.info(f"PERF: ViewerWidget.load_block_model total: {widget_time:.3f}s (renderer: {renderer_time:.3f}s)")
 
    def refresh_scene(self, block_model: BlockModel):
        """
        Refresh local state and ensure block model is loaded in renderer.
        Since we disabled blockModelLoaded signal, we need to ensure renderer loads it here.
        """
        import time
        start = time.time()
        self._prepare_block_model(block_model, log_prefix="Refresh")
        
        # FIX: Since blockModelLoaded signal is disabled, we need to load the model here
        # Check if renderer already has this model loaded to avoid double loading
        if self.renderer and self.renderer.current_model != self.current_model:
            renderer_start = time.time()
            self.renderer.load_block_model(self.current_model)
            renderer_time = time.time() - renderer_start
            logger.info(f"PERF: Renderer.load_block_model (via refresh_scene) took {renderer_time:.3f}s")
        else:
            logger.debug("Renderer already has this model loaded, skipping load_block_model")
        
        self._post_load_updates(block_model, source="ViewerWidget.refresh_scene")
        elapsed = time.time() - start
        logger.info(f"PERF: ViewerWidget.refresh_scene total: {elapsed:.3f}s")
 
    def _prepare_block_model(self, block_model, log_prefix: str = "Filter setup") -> None:
        """Common preparation logic before interacting with the renderer."""
        import time

        # Convert DataFrame to BlockModel if needed
        if isinstance(block_model, pd.DataFrame):
            # Import BlockModel here to avoid circular imports
            from ..models.block_model import BlockModel

            # Create BlockModel and update from DataFrame
            bm = BlockModel()
            bm.update_from_dataframe(block_model)
            self.current_model = bm
            logger.info(f"Converted DataFrame to BlockModel: {bm.block_count} blocks")
        else:
            # Already a BlockModel
            self.current_model = block_model

        # Invalidate bounds cache when model changes
        self._cached_bounds = None
        self._bounds_cache_time = 0.0

        filter_start = time.time()
        self.filters.set_block_model(self.current_model)
        filter_time = time.time() - filter_start
        logger.info(f"PERF: {log_prefix} took {filter_time:.3f}s")
 
        # Apply override block size if set (use self.current_model - may be converted from DataFrame)
        if self._override_block_size is not None and getattr(self.current_model, 'positions', None) is not None:
            dims = np.tile(np.array(self._override_block_size, dtype=float), (self.current_model.block_count, 1))
            self.current_model.set_geometry(self.current_model.positions, dims)
 
    def _post_load_updates(self, block_model: BlockModel, source: str) -> None:
        """Shared post-load logic after renderer state is up-to-date."""
        try:
            # Enable interaction now that renderable actors are present
            # Pass cell count for LOD computation
            if block_model is None:
                cell_count = 0
            elif hasattr(block_model, "block_count"):
                cell_count = int(block_model.block_count)
            else:
                try:
                    cell_count = len(block_model)
                except Exception:
                    cell_count = 0
            self.enable_interaction(
                cell_count=cell_count,
                has_block_model=True,
                has_drillholes=hasattr(self.renderer, '_drillhole_hole_actors') and bool(self.renderer._drillhole_hole_actors)
            )

            bounds = None
            # Prefer renderer-derived scene bounds
            try:
                if self.renderer and hasattr(self.renderer, '_get_scene_bounds'):
                    bounds = self.renderer._get_scene_bounds()
                else:
                    bounds = None
            except Exception:
                bounds = None
            # Fallback to model-provided bounds if available
            if not bounds:
                try:
                    bounds = getattr(block_model, 'bounds', None) or getattr(block_model, 'get_bounds', lambda: None)()
                except Exception:
                    bounds = None
 
            if getattr(self, '_elevation_widget', None) is not None:
                # Only show axis when renderer intends axes/bounds to be visible
                if getattr(self.renderer, 'show_axes', True):
                    self._elevation_widget.show_for_bounds(bounds)
                else:
                    try:
                        self._elevation_widget.hide()
                    except Exception:
                        pass
        except Exception:
            pass
 
        # Defer expensive operations to avoid blocking UI
        # Initialize picking state after a short delay to allow mesh to be fully added
        QTimer.singleShot(100, self._initialize_picking_state)
 
        # Fit to view after picking state is initialized (with additional delay for render)
        QTimer.singleShot(200, self.fit_to_view)
 
        block_count = getattr(self.current_model, 'block_count', None) or (len(block_model) if hasattr(block_model, '__len__') else 0)
        logger.info(f"{source}: synchronized {block_count} blocks")

    def override_block_size(self, dx: float, dy: float, dz: float):
        """Set a uniform block size and rebuild the current model view."""
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError("Block sizes must be positive")
        self._override_block_size = (dx, dy, dz)
        if self.current_model is not None and self.current_model.positions is not None:
            dims = np.tile(np.array(self._override_block_size, dtype=float), (self.current_model.block_count, 1))
            self.current_model.set_geometry(self.current_model.positions, dims)
            self.renderer.load_block_model(self.current_model)
            self.fit_to_view()
    
    def set_property_coloring(self, property_name: str):
        """
        Color blocks by a property.
        
        Args:
            property_name: Name of property to use for coloring
        """
        if self.current_model is None:
            return
        
        self.current_property = property_name
        
        # Get property values
        property_values = self.current_model.get_property(property_name)
        if property_values is None:
            logger.warning(f"Property '{property_name}' not found")
            return
        
        # Apply coloring with current color mode
        discrete = (self.current_color_mode == 'discrete')
        self.renderer.set_property_coloring(property_name, self.current_colormap, discrete=discrete)
        
        logger.info(f"Applied property coloring: {property_name} (mode: {self.current_color_mode})")
    
    def set_colormap(self, colormap: str):
        """
        Set the colormap for property visualization.
        
        Args:
            colormap: Name of the colormap
        """
        self.current_colormap = colormap
        self.color_mapper.set_colormap(colormap)
        
        # Reapply coloring if a property is selected
        if self.current_property:
            self.set_property_coloring(self.current_property)
        
        logger.info(f"Set colormap: {colormap}")
    
    def _on_legend_colormap_changed(self, colormap: str):
        """
        Handle colormap change from legend widget.
        Updates the renderer and property panel to stay synchronized.
        
        Args:
            colormap: New colormap name
        """
        try:
            if not colormap or not self.renderer:
                return
            
            # Update renderer's current colormap
            self.renderer.current_colormap = colormap
            self.current_colormap = colormap
            
            # Get active layer and property from renderer or property panel
            active_layer = None
            property_name = None
            
            # Try to get from renderer's active layers
            if hasattr(self.renderer, 'active_layers') and self.renderer.active_layers:
                # Get the first active layer (or could use a "current" layer)
                active_layer = list(self.renderer.active_layers.keys())[0] if self.renderer.active_layers else None
                if active_layer:
                    layer_data = self.renderer.active_layers[active_layer]
                    # Try to get property from layer or renderer
                    property_name = getattr(self.renderer, 'current_property', None)
            
            # If we have both layer and property, update the layer
            if active_layer and property_name:
                # Get color mode from property panel if accessible via main window
                color_mode = 'continuous'  # Default
                custom_colors = None
                
                # Try to get color mode from property panel
                try:
                    main_window = self.window()
                    if main_window and hasattr(main_window, 'property_panel') and main_window.property_panel:
                        prop_panel = main_window.property_panel
                        if hasattr(prop_panel, 'color_mode_combo') and prop_panel.color_mode_combo:
                            color_mode = prop_panel.color_mode_combo.currentText().lower()
                        # Get custom colors if in discrete mode
                        if color_mode == 'discrete':
                            custom_colors = prop_panel._custom_discrete_colors.get((active_layer, property_name))
                except Exception as e:
                    logger.debug(f"Could not get color mode from property panel: {e}")
                
                self.renderer.update_layer_property(
                    active_layer,
                    property_name,
                    colormap,
                    color_mode,
                    custom_colors=custom_colors
                )
                logger.info(f"Updated layer '{active_layer}' colormap to '{colormap}' from legend")
            else:
                # If no active layer/property, just update the current colormap
                logger.debug(f"Legend colormap changed to '{colormap}' but no active layer/property to update")
            
            # Emit signal to notify property panel (if connected)
            # The property panel should listen to this and update its dropdown
            # We'll handle this connection in main_window
            
        except Exception as e:
            logger.error(f"Error handling legend colormap change: {e}", exc_info=True)
    
    def set_color_mode(self, mode: str):
        """
        Set the color mode (continuous or discrete).
        
        Args:
            mode: 'continuous' or 'discrete'
        """
        from PyQt6.QtWidgets import QMessageBox
        
        # Check if discrete mode is appropriate for current property
        if mode == 'discrete' and self.current_property and self.current_model:
            prop_values = self.current_model.get_property(self.current_property)
            if prop_values is not None:
                unique_values = np.unique(prop_values[~np.isnan(prop_values)])
                n_categories = len(unique_values)
                
                # Warn if too many categories
                if n_categories > 100:
                    QMessageBox.warning(
                        self,
                        "Too Many Categories",
                        f"Property '{self.current_property}' has {n_categories} unique values.\n\n"
                        "Discrete mode works best with < 20 categories.\n"
                        "The display will be limited to 100 categories.\n\n"
                        "Consider using Continuous mode instead.",
                        QMessageBox.StandardButton.Ok
                    )
                elif n_categories > 20:
                    QMessageBox.information(
                        self,
                        "Many Categories",
                        f"Property '{self.current_property}' has {n_categories} unique values.\n\n"
                        "Discrete mode works best with < 20 categories.\n"
                        "Performance may be slower.",
                        QMessageBox.StandardButton.Ok
                    )
        
        self.current_color_mode = mode
        
        # NOTE: Do NOT call set_property_coloring here!
        # The property panel's _on_color_mode_changed already handles the update
        # via update_layer_property which properly includes custom colors.
        # Calling set_property_coloring would overwrite those changes without custom colors.
        
        logger.info(f"Set color mode: {mode}")

        if self.renderer is not None:
            try:
                self.renderer._refresh_legend_from_active_layer()
            except Exception:
                pass
    
    def apply_property_filter(self, property_name: str, min_value: float, max_value: float):
        """
        Apply property-based filtering with validation warnings.
        
        Args:
            property_name: Name of property to filter on
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
        """
        if self.current_model is None:
            return
        
        # Apply filter
        filtered_indices = self.filters.apply_property_filter(property_name, min_value, max_value)
        
        # Validation: Check if filter removes ALL blocks
        if len(filtered_indices) == 0:
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Filter Warning")
            msg.setText("Filter Removes All Blocks")
            msg.setInformativeText(
                f"The current filter settings remove all blocks from view.\n\n"
                f"Property: {property_name}\n"
                f"Range: {min_value:.2f} to {max_value:.2f}\n\n"
                "Adjust the filter range to display blocks."
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            logger.warning(f"Filter removes all blocks: {property_name} [{min_value:.2f}, {max_value:.2f}]")
        # Validation: Warn if very few blocks remain
        elif len(filtered_indices) < 10:
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Filter Information")
            msg.setText(f"Only {len(filtered_indices)} blocks visible")
            msg.setInformativeText(
                f"The current filter is very restrictive.\n\n"
                f"Property: {property_name}\n"
                f"Range: {min_value:.2f} to {max_value:.2f}\n\n"
                f"Visible blocks: {len(filtered_indices)}\n"
                f"Total blocks: {len(self.current_model.data)}"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            logger.info(f"Filter shows very few blocks: {len(filtered_indices)}")
        
        # Update visibility
        self.renderer.update_visibility(filtered_indices)
        
        logger.info(f"Applied property filter: {len(filtered_indices)} blocks visible")
    
    def apply_spatial_slice(self, axis: str, position: float):
        """
        Apply spatial slicing (deprecated - use enable_interactive_slice instead).
        
        Args:
            axis: Axis to slice along ('x', 'y', 'z')
            position: Position along the axis
        """
        if self.current_model is None:
            return
        
        # Apply slice
        filtered_indices = self.filters.apply_spatial_slice(axis, position)
        
        # Update visibility
        self.renderer.update_visibility(filtered_indices)
        
        logger.info(f"Applied spatial slice {axis} at {position}: {len(filtered_indices)} blocks visible")
    
    def enable_interactive_slice(self, enable: bool):
        """
        Stable interactive slicing using VTK mapper clipping plane.
        Compatible with PyVista QtInteractor.
        """
        if self.plotter is None or self.display_grid is None:
            logger.warning("No active plotter or mesh.")
            return

        if enable:
            if getattr(self, "slice_active", False):
                logger.info("Slice already active.")
                return

            from pyvista import _vtk

            # Create clipping plane object
            self._clip_plane = _vtk.vtkPlane()
            normal_map = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
            self._current_normal = getattr(self, "_current_normal", "z")
            normal = normal_map.get(self._current_normal, (0, 0, 1))
            self._clip_plane.SetNormal(*normal)
            self._clip_plane.SetOrigin(*self.display_grid.center)

            # Get the actor of the main mesh safely
            try:
                # Try to get mesh_actor from renderer first (it stores it when loading)
                if hasattr(self.renderer, "mesh_actor") and self.renderer.mesh_actor is not None:
                    self.mesh_actor = self.renderer.mesh_actor
                elif not hasattr(self, "mesh_actor") or self.mesh_actor is None:
                    # fallback to first actor if not saved
                    actors = list(self.plotter.renderer.GetActors())
                    if len(actors) == 0:
                        logger.error("No visible actors to slice.")
                        return
                    self.mesh_actor = actors[0]
            except Exception as e:
                logger.error(f"Cannot find mesh actor: {e}")
                return

            # Attach clipping plane to mapper
            mapper = self.mesh_actor.GetMapper()
            mapper.RemoveAllClippingPlanes()
            mapper.AddClippingPlane(self._clip_plane)

            # Add interactive plane widget
            bounds = self.display_grid.bounds
            self._plane_widget = self.plotter.add_plane_widget(
                callback=self._update_clip_plane,
                bounds=bounds,
                normal=self._current_normal,
                origin=self.display_grid.center,
                color="cyan",
                opacity=0.4,
                outline_translation=True,
                implicit=False
            )

            self.slice_active = True
            self.plotter.render()
            logger.info(f"✓ Interactive slice enabled ({self._current_normal.upper()}-normal)")

        else:
            if not getattr(self, "slice_active", False):
                return

            try:
                if hasattr(self, "_clip_plane") and self._clip_plane:
                    mapper = self.mesh_actor.GetMapper()
                    mapper.RemoveAllClippingPlanes()

                if hasattr(self, "_plane_widget") and self._plane_widget:
                    self.plotter.clear_plane_widgets()
                    self._plane_widget = None

                self.plotter.render()
                logger.info("✓ Interactive slice disabled")
            except Exception as e:
                logger.error(f"Error disabling slice: {e}", exc_info=True)

            self.slice_active = False
    
    def _update_clip_plane(self, normal, origin):
        """Update VTK clipping plane instantly (GPU-side)."""
        if not getattr(self, "slice_active", False) or not hasattr(self, "_clip_plane"):
            return

        try:
            self._clip_plane.SetNormal(*normal)
            self._clip_plane.SetOrigin(*origin)
            self.plotter.render()
        except Exception as e:
            logger.error(f"Clip plane update failed: {e}", exc_info=True)

    def set_slice_orientation(self, axis: str):
        """Change slice plane orientation dynamically."""
        axis = axis.lower()
        if not getattr(self, "slice_active", False):
            self._current_normal = axis
            self.enable_interactive_slice(True)
            return

        normal_map = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
        normal = normal_map.get(axis, (0, 0, 1))
        self._current_normal = axis

        try:
            self._clip_plane.SetNormal(*normal)

            if hasattr(self, "_plane_widget") and self._plane_widget:
                self.plotter.clear_plane_widgets()

            bounds = self.display_grid.bounds
            self._plane_widget = self.plotter.add_plane_widget(
                callback=self._update_clip_plane,
                bounds=bounds,
                normal=self._current_normal,
                origin=self.display_grid.center,
                color="cyan",
                opacity=0.4,
                outline_translation=True,
                implicit=False
            )

            self.plotter.render()
            logger.info(f"✓ Slice orientation set to {axis.upper()} plane")

        except Exception as e:
            logger.error(f"Failed to change slice orientation: {e}", exc_info=True)
    
    def set_transparency(self, alpha: float):
        """
        Set transparency for all blocks.
        
        Args:
            alpha: Transparency value (0.0 = transparent, 1.0 = opaque)
        """
        self.current_transparency = alpha
        self.renderer.set_transparency(alpha)
        
        logger.info(f"Set transparency: {alpha}")
    
    def reset_camera(self):
        """Reset camera to default view."""
        self.renderer.reset_camera()
        logger.info("Reset camera")
    
    def fit_to_view(self):
        """Fit the model to the current view."""
        self.renderer.fit_to_view()
        logger.info("Fitted to view")

    def zoom(self, factor: float):
        """Zoom the active camera by factor and re-render.

        Args:
            factor: zoom factor (>1 zooms in, <1 zooms out)
        """
        try:
            # Prefer PyVista plotter render path
            cam = None
            if self.plotter is not None:
                try:
                    cam = self.plotter.renderer.GetActiveCamera()
                except Exception:
                    cam = None
            if cam is None and getattr(self.renderer, 'renderer', None) is not None:
                try:
                    cam = self.renderer.renderer.GetActiveCamera()
                except Exception:
                    cam = None

            if cam is not None:
                cam.Zoom(factor)
            # Trigger a render via PyVista if possible
            try:
                if self.plotter is not None:
                    self.plotter.render()
                else:
                    rw = getattr(self.renderer, 'render_window', None)
                    if rw is not None:
                        rw.Render()
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"zoom failed: {e}")

    def zoom_in(self):
        """Convenience zoom in (20% step)."""
        self.zoom(1.2)

    def zoom_out(self):
        """Convenience zoom out (20% step)."""
        self.zoom(1.0 / 1.2)
    
    def toggle_projection(self):
        """Toggle between perspective and orthographic projection."""
        if self.plotter is None:
            return
        
        # Toggle projection mode (parallel = orthographic)
        if bool(self.plotter.camera.parallel_projection):
            self.plotter.camera.disable_parallel_projection()
        else:
            self.plotter.camera.enable_parallel_projection()
        
        logger.info("Toggled projection mode")

    # View presets
    def view_top(self):
        if self.plotter is not None:
            self.plotter.view_xy()
            logger.info("Set view: Top")

    def view_front(self):
        if self.plotter is not None:
            self.plotter.view_xz()
            logger.info("Set view: Front")

    def view_right(self):
        if self.plotter is not None:
            self.plotter.view_yz()
            logger.info("Set view: Right")

    def view_iso(self):
        if self.plotter is not None:
            try:
                self.plotter.view_isometric()
            except Exception:
                # Fallback: set a generic isometric camera
                self.fit_to_view()
            logger.info("Set view: Isometric")
    
    def toggle_axes(self, show: bool):
        """Toggle coordinate axes visibility."""
        if self.renderer:
            self.renderer.toggle_overlay("axes", show)
        logger.info(f"Toggled axes: {show}")
    
    def toggle_bounds(self, show: bool):
        """Toggle bounding box visibility."""
        if self.renderer:
            self.renderer.toggle_overlay("bounds", show)
        logger.info(f"Toggled bounds: {show}")
    
    def export_screenshot(self, filename: str, resolution: Tuple[int, int] = (1920, 1080)):
        """
        Export current view as screenshot.
        
        Args:
            filename: Output filename
            resolution: Image resolution (width, height)
        """
        self.renderer.export_screenshot(filename, resolution)
        logger.info(f"Exported screenshot: {filename}")
    
    def export_filtered_data(self, filename: str):
        """
        Export filtered data.
        
        Args:
            filename: Output filename
        """
        if self.current_model is None:
            return
        
        # Get current filtered indices
        filtered_indices = self.filters.apply_combined_filters()
        
        # Determine format from extension
        file_path = filename.lower()
        if file_path.endswith('.csv'):
            format_type = 'csv'
        elif file_path.endswith('.vtk'):
            format_type = 'vtk'
        else:
            format_type = 'csv'  # Default
        
        # Export data
        self.filters.export_filtered_data(filtered_indices, filename, format_type)
        logger.info(f"Exported filtered data: {filename}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.current_model is None:
            return {}
        
        info = {
            'block_count': self.current_model.block_count,
            'bounds': self.current_model.bounds,
            'properties': self.current_model.get_property_names(),
            'metadata': {
                'source_file': self.current_model.metadata.source_file,
                'file_format': self.current_model.metadata.file_format,
                'units': self.current_model.metadata.units
            }
        }
        
        return info
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get current camera information."""
        camera_info = self.renderer.get_camera_info()
        if camera_info != self._last_camera_info:
            self._last_camera_info = camera_info
        return camera_info
    
    def _on_point_picked(self, point):
        """Handle point picking events."""
        if self.current_model is None or point is None:
            return
        
        # Find closest block
        positions = self.current_model.positions
        if positions is None:
            return
        
        # Calculate distances to all block centers
        distances = np.linalg.norm(positions - point, axis=1)
        closest_block_idx = np.argmin(distances)
        
        # Get block properties
        properties = {}
        for prop_name in self.current_model.get_property_names():
            prop_values = self.current_model.get_property(prop_name)
            if prop_values is not None:
                properties[prop_name] = prop_values[closest_block_idx]
        
        # Emit signal
        self.block_picked.emit(closest_block_idx, properties)
        
        logger.info(f"Picked block {closest_block_idx}: {properties}")
    
    def _on_global_pick(self, info_dict: Dict[str, Any]):
        """
        Handle global pick events from the renderer.
        
        This is called by the renderer's global picking system and forwards
        the pick information to any registered handlers (e.g., PickInfoPanel).
        
        Args:
            info_dict: Dictionary containing picked object information
        """
        try:
            # Use QTimer to defer the update to avoid VTK re-entry
            QTimer.singleShot(0, lambda: self._update_pick_info(info_dict))
        except Exception as e:
            logger.error(f"Error handling global pick: {e}", exc_info=True)
    
    def _update_pick_info(self, info_dict: Dict[str, Any]):
        """
        Update pick information displays.
        
        This is called on a Qt timer to safely update UI without blocking
        the VTK render thread.
        
        Args:
            info_dict: Dictionary containing picked object information
        """
        # Extract block_id from Original_ID if available
        block_id = None
        cell_id = info_dict.get('Cell ID', -1)
        
        # Check if Original_ID is in the info_dict (from cell_data)
        if 'Original_ID' in info_dict:
            block_id = int(info_dict['Original_ID'])
            logger.info(f"Block clicked via global pick: block_id={block_id}, cell_id={cell_id}")
        else:
            # Fallback: try to get Original_ID from the picked mesh
            try:
                if hasattr(self.renderer, 'plotter') and self.renderer.plotter is not None:
                    picked_mesh = self.renderer.plotter.picked_mesh
                    if picked_mesh is not None and "Original_ID" in picked_mesh.cell_data:
                        if 0 <= cell_id < len(picked_mesh.cell_data["Original_ID"]):
                            block_id = int(picked_mesh.cell_data["Original_ID"][cell_id])
                            logger.info(f"Block clicked via global pick (from mesh): block_id={block_id}, cell_id={cell_id}")
            except Exception as e:
                logger.debug(f"Could not extract Original_ID from mesh: {e}")
        
        # Build properties dict (exclude metadata fields)
        properties = {}
        exclude_keys = {'Layer', 'Layer Type', 'Cell ID', 'Position', 'Original_ID', 'Row Index'}
        for key, value in info_dict.items():
            if key not in exclude_keys:
                try:
                    # Convert to float/int if numeric
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        if not (isinstance(value, float) and np.isnan(value)):
                            properties[key] = float(value) if isinstance(value, (float, np.floating)) else int(value)
                except Exception:
                    pass
        
        # Update info_dict with block_id if found
        if block_id is not None and block_id >= 0:
            info_dict['block_id'] = block_id
            info_dict['event_type'] = 'click'
        
        # Emit the global pick event signal for PickInfoPanel
        self.global_pick_event.emit(info_dict)
        
        logger.debug(f"Global pick: {info_dict.get('Layer', 'Unknown')}, Cell {cell_id}, Block {block_id if block_id is not None else 'N/A'}")
        
        # Emit the block_picked signal with correct block_id
        if block_id is not None and block_id >= 0:
            self.block_picked.emit(block_id, properties)
            
            # Show status message
            prop_summary = ", ".join([f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}={v}" 
                                     for k, v in list(properties.items())[:3]])
            if prop_summary:
                parent = self.window()
                if hasattr(parent, 'statusBar'):
                    parent.statusBar().showMessage(f"Block {block_id}: {prop_summary}", 3000)
        elif cell_id is not None and isinstance(cell_id, (int, np.integer)) and cell_id >= 0:
            # Fallback: emit with cell_id if no block_id found
            self.block_picked.emit(int(cell_id), properties)
    
    def clear_model(self):
        """Clear the current model."""
        self.current_model = None
        self.current_property = None
        
        if self.plotter is not None:
            self.plotter.clear()
            self._setup_plotter()
        
        # Disable interaction if no drillhole data remains
        if self.renderer and not (hasattr(self.renderer, '_drillhole_hole_actors') and 
                                  self.renderer._drillhole_hole_actors and 
                                  len(self.renderer._drillhole_hole_actors) > 0):
            self.disable_interaction()
        
        logger.info("Cleared model")
    
    def get_statistics(self, property_name: str) -> Dict[str, Any]:
        """Get statistics for a property."""
        if self.current_model is None:
            return {}
        
        return self.color_mapper.get_property_statistics(
            self.current_model.get_property(property_name)
        )
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of active filters."""
        return self.filters.get_filter_summary()
    
    def set_uniform_block_size(self, dx: float, dy: float, dz: float):
        """
        Set uniform block size for the model.
        
        Args:
            dx, dy, dz: Uniform dimensions for all blocks.
        """
        if self.current_model is None:
            logger.warning("Cannot set uniform block size: no model loaded.")
            return
        
        self.renderer.set_uniform_block_size(dx, dy, dz)
        logger.info(f"Requested uniform block size: ({dx}, {dy}, {dz})")
    
    def set_view_preset(self, preset: str):
        """
        Set camera to a predefined view preset.
        
        Args:
            preset: 'top', 'front', 'right', 'isometric'
        """
        # View presets work with either block model OR drillholes
        has_model = self.current_model is not None
        has_drillholes = (
            self.renderer is not None and 
            "drillholes" in self.renderer.active_layers
        )
        
        if not self.plotter or (not has_model and not has_drillholes):
            # Silently return - no need to log when user tries view preset without data
            return
        
        self.renderer.set_view_preset(preset)
        logger.info(f"Requested view preset: {preset}")
    
    def set_legend_settings(self, position: str, font_size: int):
        """
        Set legend (scalar bar) settings.
        
        Args:
            position: 'vertical' or 'horizontal'
            font_size: Font size for legend text
        """
        self.renderer.set_legend_settings(position, font_size)
        
        # Re-apply current property coloring to update the legend
        if self.current_property and self.current_model:
            self.renderer.set_property_coloring(self.current_property, self.current_colormap)
        
        logger.info(f"Set legend settings: position={position}, font_size={font_size}")
    
    def update_legend_style(self, style_dict: dict):
        """
        Update legend style with comprehensive parameters.
        
        Args:
            style_dict: Dictionary with legend style parameters from LegendStyleState
                {
                    'count': int,
                    'decimals': int,
                    'shadow': bool,
                    'outline': bool,
                    'background': tuple (r, g, b),
                    'background_opacity': float,
                    'orientation': str,
                    'font_size': int,
                    'mode': str
                }
        """
        if not self.renderer:
            return
        
        try:
            # Update renderer legend parameters
            self.renderer.update_legend_style(style_dict)
            
            # Re-apply current property coloring to update the legend
            if self.current_property and self.current_model:
                self.renderer.set_property_coloring(self.current_property, self.current_colormap)
            
            logger.info(f"Updated legend style: {style_dict}")
        except Exception as e:
            logger.error(f"Error updating legend style: {e}", exc_info=True)
    
    def set_axis_font(self, family: str, size: int):
        """
        Set axis font family and size.
        
        Args:
            family: Font family name
            size: Font size in points
        """
        self.renderer.set_axis_font(family, size)
        logger.info(f"Set axis font: {family}, size={size}")
    
    def set_axis_font_color(self, rgb_tuple: tuple):
        """
        Set axis font color.
        
        Args:
            rgb_tuple: RGB color as tuple (0-1 range)
        """
        self.renderer.set_axis_font_color(rgb_tuple)
        logger.info(f"Set axis color: RGB{rgb_tuple}")
    
    def set_opacity(self, opacity: float):
        """
        Set block opacity with instant update.
        
        Args:
            opacity: Opacity value (0.0-1.0)
        """
        self.renderer.set_opacity(opacity)
        logger.info(f"Set opacity: {opacity}")
    
    def set_edge_color(self, rgb_tuple: tuple):
        """
        Set edge color with instant update.
        
        Args:
            rgb_tuple: RGB color as tuple (0-1 range)
        """
        self.renderer.set_edge_color(rgb_tuple)
        logger.info(f"Set edge color: RGB{rgb_tuple}")
    
    def set_background_color(self, rgb_tuple: tuple):
        """
        Set background color with instant update.
        
        Args:
            rgb_tuple: RGB color as tuple (0-1 range)
        """
        self.renderer.set_background_color(rgb_tuple)
        logger.info(f"Set background color: RGB{rgb_tuple}")
    
    def toggle_lighting(self, enabled: bool):
        """
        Toggle lighting on/off.
        
        Args:
            enabled: True to enable, False to disable
        """
        self.renderer.toggle_lighting(enabled)
        logger.info(f"Lighting: {enabled}")
    
    def update_legend_font_size(self, size: int):
        """
        Update legend font size.
        
        Args:
            size: Font size in points
        """
        self.renderer.update_legend_font_size(size)
        
        # Re-apply current property coloring to update the legend
        if self.current_property and self.current_model:
            self.renderer.set_property_coloring(self.current_property, self.current_colormap)
        
        logger.info(f"Legend font size: {size}")
    
    def toggle_legend_visibility(self, visible: bool):
        """
        Toggle legend bar visibility.
        
        Args:
            visible: True to show, False to hide
        """
        self.renderer.set_legend_visibility(visible)
        
        # Re-apply current property coloring to update display
        if self.current_property and self.current_model:
            self.renderer.set_property_coloring(self.current_property, self.current_colormap)
        
        logger.info(f"Legend visibility: {visible}")
    
    def set_legend_orientation(self, orientation: str):
        """
        Set legend bar orientation.
        
        Args:
            orientation: 'vertical' or 'horizontal'
        """
        self.renderer.set_legend_orientation(orientation)
        
        # Re-apply current property coloring to update the legend
        if self.current_property and self.current_model:
            self.renderer.set_property_coloring(self.current_property, self.current_colormap)
        
        logger.info(f"Legend orientation: {orientation}")
    
    def reset_legend_position(self):
        """Reset legend bar to default position."""
        self.renderer.reset_legend_position()
        
        # Re-apply current property coloring to update the legend
        if self.current_property and self.current_model:
            self.renderer.set_property_coloring(self.current_property, self.current_colormap)
        
        logger.info("Legend position reset")
    
    def toggle_orthographic_projection(self, enabled: bool):
        """
        Toggle between orthographic and perspective projection.
        
        Args:
            enabled: True for orthographic, False for perspective
        """
        if self.plotter is None:
            return
        
        if enabled:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        
        self.plotter.render()
        logger.info(f"Projection: {'Orthographic' if enabled else 'Perspective'}")
    
    def set_trackball_mode(self, enabled: bool):
        """
        Set trackball rotation mode.
        
        Args:
            enabled: True to enable trackball, False for terrain mode
        """
        if self.plotter is None:
            return
        
        if enabled:
            self.plotter.enable_trackball_style()
        else:
            self.plotter.enable_terrain_style()
        
        logger.info(f"Trackball mode: {enabled}")
    
    def export_mesh_to_file(self, filename: str):
        """
        Export the current mesh to a file.
        
        Args:
            filename: Output filename (STL, OBJ, etc.)
        """
        try:
            if 'unstructured_grid' in self.renderer.block_meshes:
                mesh = self.renderer.block_meshes['unstructured_grid']
                mesh.save(filename)
                logger.info(f"Exported mesh to: {filename}")
            else:
                logger.warning("No mesh available to export")
        except Exception as e:
            logger.error(f"Error exporting mesh: {e}")
            raise
    
    def clear_scene(self):
        """
        Clear the scene and remove all actors/meshes.
        
        FIX CS-002: Properly notifies PickingController, clears overlays,
        and resets all control state.
        """
        try:
            # Clear all actors from plotter
            self.plotter.clear()
            
            # Clear block meshes
            self.renderer.block_meshes.clear()
            
            # FIX CS-002: Notify PickingController that scene is empty
            self._interaction_enabled = False
            self._picking_controller.on_data_cleared()
            
            # Clear viewer state
            self.block_centers = None
            self.display_grid = None
            self.current_model = None
            self.current_property = None
            self._hovered_block_id = None
            self._selected_blocks.clear()
            
            # Clear any selected block highlight
            if 'selected_block' in self.renderer.block_meshes:
                del self.renderer.block_meshes['selected_block']
            
            # Clear hover highlight
            self._clear_hover_highlight()
            
            # Clear swath highlight
            self.clear_swath_highlight()
            
            # Clear selection actor if present
            if self._selection_actor is not None:
                try:
                    self.plotter.remove_actor(self._selection_actor)
                except Exception:
                    pass
                self._selection_actor = None
            
            # FIX CS-002: Clear overlay state via overlay manager
            if hasattr(self.renderer, 'overlay_manager') and self.renderer.overlay_manager is not None:
                try:
                    self.renderer.overlay_manager.set_bounds(None)
                except Exception:
                    pass
            
            # Render empty scene
            self.plotter.render()
            
            logger.info("Cleared scene (interaction disabled, LOD-P0)")
            
        except Exception as e:
            logger.error(f"Error clearing scene: {e}")
            raise
    
    # REMOVED: Duplicate get_camera_info method - using version at line 1726 which tracks camera changes
    
    def toggle_scalar_bar_visibility(self, visible: bool):
        """
        DEPRECATED: Use LegendManager.set_visibility() instead.
        
        Toggle legend visibility via LegendManager.
        
        Args:
            visible: True to show, False to hide
        """
        try:
            if self.renderer and self.renderer.legend_manager:
                self.renderer.legend_manager.set_visibility(visible)
            elif self.renderer:
                # Fallback to renderer method for backward compatibility
                self.renderer.toggle_scalar_bar(visible)
            logger.info(f"Legend visibility: {visible}")
        except Exception as e:
            logger.error(f"Error toggling legend visibility: {e}", exc_info=True)
    
    def contextMenuEvent(self, event: QContextMenuEvent):
        """
        Handle right-click context menu in the 3D viewport.
        
        Provides quick access to common viewport operations like camera controls,
        view bookmarks, and rendering options.
        """
        menu = QMenu(self)
        
        # Camera operations
        menu.addAction("Reset View", self.fit_to_view)
        menu.addAction("Top View", lambda: self.set_view_preset('top'))
        menu.addAction("Front View", lambda: self.set_view_preset('front'))
        menu.addAction("Side View", lambda: self.set_view_preset('side'))
        menu.addSeparator()
        
        # Copy/paste camera position
        copy_camera_action = menu.addAction("Copy Camera Position")
        copy_camera_action.triggered.connect(self._copy_camera_position)
        
        paste_camera_action = menu.addAction("Paste Camera Position")
        paste_camera_action.triggered.connect(self._paste_camera_position)
        menu.addSeparator()
        
        # Rendering options
        if self.current_model is not None:
            wireframe_action = menu.addAction("Toggle Wireframe")
            wireframe_action.triggered.connect(self._toggle_wireframe)
            
            edges_action = menu.addAction("Toggle Edge Visibility")
            edges_action.triggered.connect(self._toggle_edges)
            menu.addSeparator()
        
        # Measurement tools
        measure_action = menu.addAction("Measure Distance (Ruler)")
        measure_action.triggered.connect(self._start_measure_distance)

        clear_meas_action = menu.addAction("Clear Measurements")
        clear_meas_action.triggered.connect(self._clear_measurements)
        # Area measurement tools
        area_action = menu.addAction("Measure Area (Polygon)")
        area_action.triggered.connect(self._start_measure_area)
        finish_area_action = menu.addAction("Finish Area Measurement")
        finish_area_action.triggered.connect(self._finish_measure_area)
        cancel_area_action = menu.addAction("Cancel Area Measurement")
        cancel_area_action.triggered.connect(self._cancel_measure_area)
        export_meas_action = menu.addAction("Export Measurements…")
        export_meas_action.triggered.connect(self._export_measurements)
        menu.addSeparator()

        # Screenshot
        screenshot_action = menu.addAction("Save Screenshot...")
        screenshot_action.triggered.connect(self._save_screenshot)
        
        # Show menu at cursor position
        menu.exec(event.globalPos())
    
    def _copy_camera_position(self):
        """Copy current camera position to clipboard."""
        try:
            camera_info = self.renderer.get_camera_info()
            import json
            from PyQt6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(json.dumps(camera_info, indent=2))
            logger.info("Camera position copied to clipboard")
            # Show brief status message if parent has statusBar
            parent = self.window()
            if hasattr(parent, 'statusBar'):
                parent.statusBar().showMessage("Camera position copied to clipboard", 2000)
        except Exception as e:
            logger.error(f"Error copying camera position: {e}")
    
    def _paste_camera_position(self):
        """Paste camera position from clipboard."""
        try:
            import json
            from PyQt6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            text = clipboard.text()
            camera_info = json.loads(text)
            
            # Validate structure
            if 'position' not in camera_info or 'focal_point' not in camera_info:
                QMessageBox.warning(self, "Invalid Data", "Clipboard does not contain valid camera position data.")
                return
            
            # Apply camera position
            self.renderer.set_camera_position(
                camera_info['position'],
                camera_info['focal_point'],
                camera_info.get('view_up', [0, 0, 1])
            )
            logger.info("Camera position restored from clipboard")
            
            # Show brief status message
            parent = self.window()
            if hasattr(parent, 'statusBar'):
                parent.statusBar().showMessage("Camera position restored", 2000)
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Invalid Format", "Clipboard does not contain valid JSON data.")
        except Exception as e:
            logger.error(f"Error pasting camera position: {e}")
            QMessageBox.warning(self, "Error", f"Failed to restore camera position:\n{str(e)}")
    
    def _toggle_wireframe(self):
        """Toggle wireframe rendering mode."""
        try:
            # This would need renderer support - placeholder for now
            logger.info("Wireframe toggle requested (not yet implemented)")
            QMessageBox.information(self, "Not Implemented", "Wireframe mode not yet implemented.")
        except Exception as e:
            logger.error(f"Error toggling wireframe: {e}")
    
    def _toggle_edges(self):
        """Toggle edge visibility on blocks."""
        try:
            # Access the current block mesh actor and toggle edge visibility
            if self.renderer and hasattr(self.renderer, 'active_layers'):
                for layer_name, layer_info in self.renderer.active_layers.items():
                    if layer_info['type'] == 'blocks':
                        actor = layer_info.get('actor')
                        if actor:
                            current_edges = actor.GetProperty().GetEdgeVisibility()
                            actor.GetProperty().SetEdgeVisibility(not current_edges)
                            self.renderer.plotter.render()
                            logger.info(f"Toggled edges for layer '{layer_name}': {not current_edges}")
                            
                            # Status message
                            parent = self.window()
                            if hasattr(parent, 'statusBar'):
                                status = "on" if not current_edges else "off"
                                parent.statusBar().showMessage(f"Block edges {status}", 2000)
                            return
            
            QMessageBox.information(self, "No Blocks", "No block model loaded.")
        except Exception as e:
            logger.error(f"Error toggling edges: {e}")
    
    def _save_screenshot(self):
        """Save a screenshot of the current view."""
        try:
            from PyQt6.QtWidgets import QFileDialog
            from pathlib import Path
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Screenshot",
                str(Path.home() / "screenshot.png"),
                "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*.*)"
            )
            
            if filename:
                self.renderer.plotter.screenshot(filename)
                logger.info(f"Screenshot saved: {filename}")
                QMessageBox.information(self, "Success", f"Screenshot saved to:\n{filename}")
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            QMessageBox.warning(self, "Error", f"Failed to save screenshot:\n{str(e)}")

    def _start_measure_distance(self):
        """Enable distance measurement mode (two clicks)."""
        try:
            if self.renderer and hasattr(self.renderer, 'start_distance_measurement'):
                self.renderer.start_distance_measurement()
                parent = self.window()
                if hasattr(parent, 'statusBar'):
                    parent.statusBar().showMessage("Ruler: click two points to measure distance", 4000)
            else:
                logger.debug("Distance measurement not available (method not implemented)")
        except Exception as e:
            logger.error(f"Error starting distance measurement: {e}")

    def _clear_measurements(self):
        """Clear all measurement overlays from the scene."""
        try:
            if self.renderer:
                self.renderer.clear_measurements()
                parent = self.window()
                if hasattr(parent, 'statusBar'):
                    parent.statusBar().showMessage("Measurements cleared", 2000)
        except Exception as e:
            logger.error(f"Error clearing measurements: {e}")

    def _start_measure_area(self):
        """Enable area measurement (click multiple points, then finish)."""
        try:
            if self.renderer and hasattr(self.renderer, 'start_area_measurement'):
                self.renderer.start_area_measurement()
                parent = self.window()
                if hasattr(parent, 'statusBar'):
                    parent.statusBar().showMessage("Area: click to add vertices; choose 'Finish Area Measurement' to compute", 5000)
            else:
                logger.debug("Area measurement not available (method not implemented)")
        except Exception as e:
            logger.error(f"Error starting area measurement: {e}")

    def _finish_measure_area(self):
        """Finish area polygon and compute area."""
        try:
            if self.renderer and hasattr(self.renderer, 'finish_area_measurement'):
                self.renderer.finish_area_measurement()
                parent = self.window()
                if hasattr(parent, 'statusBar'):
                    parent.statusBar().showMessage("Area computed", 2000)
            else:
                logger.debug("Area measurement finish not available (method not implemented)")
        except Exception as e:
            logger.error(f"Error finishing area measurement: {e}")

    def _cancel_measure_area(self):
        """Cancel current area measurement session without clearing overlays."""
        try:
            if self.renderer:
                self.renderer.cancel_area_measurement()
                parent = self.window()
                if hasattr(parent, 'statusBar'):
                    parent.statusBar().showMessage("Area measurement cancelled", 2000)
        except Exception as e:
            logger.error(f"Error cancelling area measurement: {e}")

    def _export_measurements(self):
        """Export recorded measurements to a CSV file."""
        try:
            if not self.renderer:
                return
            from PyQt6.QtWidgets import QFileDialog
            from pathlib import Path
            default_path = Path.home() / "measurements.csv"
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Measurements",
                str(default_path),
                "CSV Files (*.csv);;All Files (*.*)"
            )
            if filename:
                count = self.renderer.export_measurements(filename, fmt='csv')
                parent = self.window()
                if hasattr(parent, 'statusBar'):
                    msg = f"Exported {count} measurement(s) to {filename}" if count else "No measurements to export"
                    parent.statusBar().showMessage(msg, 4000)
        except Exception as e:
            logger.error(f"Error exporting measurements: {e}")
    
    # ============================================================================
    # Interactive Selection Methods
    # ============================================================================
    
    def start_box_selection_mode(self):
        """Start interactive box selection mode with rubber-band visualization."""
        # Box selection works with either block model OR drillholes
        has_model = self.current_model is not None
        has_drillholes = (
            self.renderer is not None and
            "drillholes" in self.renderer.active_layers
        )

        if not self.plotter:
            error_msg = "Cannot start box selection: 3D viewer not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not has_model and not has_drillholes:
            error_msg = "Cannot start box selection: No block model or drillholes loaded. Please load data first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # End any other selection mode
            self.end_selection_mode()
            
            self.selection_mode = 'box'
            
            # Create box widget for rubber-band selection
            if self._box_widget is None:
                # Get model bounds for initial box
                bounds = self.plotter.bounds
                
                def box_callback(box_bounds):
                    """Called when box is adjusted."""
                    logger.info(f"Box selection bounds: {box_bounds}")
                    # Emit signal with bounds
                    self.box_selection_completed.emit(box_bounds)
                
                self._box_widget = self.plotter.add_box_widget(
                    callback=box_callback,
                    bounds=bounds,
                    factor=1.25,
                    rotation_enabled=False,
                    color='green',
                    use_planes=False
                )
            
            logger.info("Started box selection mode - drag box corners to select region")
            parent = self.window()
            if hasattr(parent, 'statusBar'):
                parent.statusBar().showMessage("Box Selection: Drag corners to define selection region", 5000)
                
        except Exception as e:
            logger.error(f"Error starting box selection: {e}", exc_info=True)
    
    def end_box_selection_mode(self):
        """End box selection mode and return bounds."""
        if self._box_widget is not None:
            try:
                # Get final bounds before removing widget
                bounds = self._box_widget.GetBounds() if hasattr(self._box_widget, 'GetBounds') else self.plotter.bounds
                
                # Remove widget
                self.plotter.clear_box_widgets()
                self._box_widget = None
                
                logger.info(f"Ended box selection mode with bounds: {bounds}")
                return bounds
                
            except Exception as e:
                logger.error(f"Error ending box selection: {e}")
                return None
        return None
    
    def start_click_selection_mode(self):
        """Start click-to-select blocks mode."""
        if not self.plotter:
            error_msg = "Cannot start click selection: 3D viewer not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not self.current_model:
            error_msg = "Cannot start click selection: No block model loaded. Please load a block model first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # End any other selection mode
            self.end_selection_mode()
            
            self.selection_mode = 'click'
            self._selected_blocks = set()
            
            # Enable picking
            self.plotter.enable_cell_picking(
                callback=self._on_block_click_for_selection,
                show=False,  # Don't show default picking visualization
                color='yellow',
                through=False
            )
            
            logger.info("Started click selection mode - click blocks to select (Ctrl+Click for multi-select)")
            parent = self.window()
            if hasattr(parent, 'statusBar'):
                parent.statusBar().showMessage("Click Selection: Click blocks (Ctrl for multi-select, Shift to deselect)", 5000)
                
        except Exception as e:
            logger.error(f"Error starting click selection: {e}", exc_info=True)
    
    def _on_block_clicked(self, cell_id):
        """
        Handle block click - called by PyVista's enable_cell_picking.
        
        This is called for normal clicking (not just selection mode).
        """
        if cell_id is None or cell_id < 0:
            return
        
        try:
            # Get the picked mesh from plotter
            if not hasattr(self.plotter, 'picked_mesh') or self.plotter.picked_mesh is None:
                return
            
            mesh = self.plotter.picked_mesh
            
            # Check if this is a block model
            if "Original_ID" in mesh.cell_data:
                original_ids = mesh.cell_data["Original_ID"]
                if 0 <= cell_id < len(original_ids):
                    block_id = int(original_ids[cell_id])
                    if block_id >= 0:
                        # Get properties from mesh cell_data
                        properties = {}
                        for prop_name in mesh.cell_data.keys():
                            if prop_name == 'Original_ID':
                                continue
                            try:
                                prop_array = mesh.cell_data[prop_name]
                                if 0 <= cell_id < len(prop_array):
                                    prop_value = prop_array[cell_id]
                                    if not (isinstance(prop_value, (int, float)) and np.isnan(prop_value)):
                                        properties[prop_name] = prop_value
                            except Exception:
                                pass
                        
                        logger.info(f"Block clicked: ID={block_id}, properties={properties}")
                        
                        # Emit signals
                        self.block_picked.emit(block_id, properties)
                        self.global_pick_event.emit({
                            'block_id': block_id,
                            'properties': properties,
                            'layer': 'Block Model',
                            'event_type': 'click'
                        })
                        
                        # Show status message
                        prop_summary = ", ".join([f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}={v}" 
                                                 for k, v in list(properties.items())[:3]])
                        if prop_summary:
                            parent = self.window()
                            if hasattr(parent, 'statusBar'):
                                parent.statusBar().showMessage(f"Block {block_id}: {prop_summary}", 3000)
        except Exception as e:
            logger.debug(f"Error in block click handler: {e}", exc_info=True)
    
    def _on_block_click_for_selection(self, cell_id):
        """Handle block click in selection mode."""
        try:
            modifiers = QApplication.keyboardModifiers()
            
            if cell_id is None or cell_id < 0:
                return
            
            # Check modifiers
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                # Shift: deselect
                if cell_id in self._selected_blocks:
                    self._selected_blocks.remove(cell_id)
                    logger.info(f"Deselected block {cell_id}")
            elif modifiers & Qt.KeyboardModifier.ControlModifier:
                # Ctrl: add to selection
                self._selected_blocks.add(cell_id)
                logger.info(f"Added block {cell_id} to selection (total: {len(self._selected_blocks)})")
            else:
                # No modifier: replace selection
                self._selected_blocks = {cell_id}
                logger.info(f"Selected block {cell_id}")
            
            # Update visualization - DISABLED: No visual highlight overlay
            # self._update_selection_visualization()
            
            # Emit signal
            self.blocks_selected.emit(self._selected_blocks)
            
        except Exception as e:
            logger.error(f"Error in block click selection: {e}", exc_info=True)
    
    def _update_selection_visualization(self):
        """Update visual highlighting of selected blocks."""
        try:
            # Remove old selection actor
            if self._selection_actor is not None:
                self.plotter.remove_actor(self._selection_actor)
                self._selection_actor = None
            
            if not self._selected_blocks:
                return
            
            # Get block model data
            if not self.current_model:
                return
            
            # Use BlockModel's positions and dimensions directly (more reliable than DataFrame)
            positions = self.current_model.positions
            dimensions = self.current_model.dimensions
            
            if positions is None or dimensions is None:
                logger.warning("Cannot update selection visualization: positions or dimensions missing")
                return
            
            selected_indices = list(self._selected_blocks)
            
            if len(selected_indices) == 0:
                return
            
            # Create mesh for selected blocks
            import pyvista as pv
            points = []
            cells = []
            cell_offset = 0
            
            for block_id in selected_indices:
                # Validate block_id is within range
                if block_id >= len(positions) or block_id >= len(dimensions):
                    logger.warning(f"Block ID {block_id} out of range (max: {len(positions)})")
                    continue
                
                # Get block center and dimensions directly from arrays
                center = positions[block_id]
                size = dimensions[block_id]
                x, y, z = center[0], center[1], center[2]
                dx, dy, dz = size[0], size[1], size[2]
                
                # Make highlight slightly larger (5% bigger) to ensure it's visible
                scale = 1.05
                dx_scaled = dx * scale
                dy_scaled = dy * scale
                dz_scaled = dz * scale
                
                # Create cube vertices (slightly enlarged)
                vertices = np.array([
                    [x - dx_scaled/2, y - dy_scaled/2, z - dz_scaled/2],
                    [x + dx_scaled/2, y - dy_scaled/2, z - dz_scaled/2],
                    [x + dx_scaled/2, y + dy_scaled/2, z - dz_scaled/2],
                    [x - dx_scaled/2, y + dy_scaled/2, z - dz_scaled/2],
                    [x - dx_scaled/2, y - dy_scaled/2, z + dz_scaled/2],
                    [x + dx_scaled/2, y - dy_scaled/2, z + dz_scaled/2],
                    [x + dx_scaled/2, y + dy_scaled/2, z + dz_scaled/2],
                    [x - dx_scaled/2, y + dy_scaled/2, z + dz_scaled/2],
                ])
                
                points.extend(vertices)
                
                # Define cell (hexahedron = 8 vertices)
                cell = [8] + list(range(cell_offset, cell_offset + 8))
                cells.extend(cell)
                cell_offset += 8
            
            if len(points) > 0:
                # Create unstructured grid
                points = np.array(points)
                cells = np.array(cells)
                
                celltypes = np.full(len(selected_indices), pv.CellType.HEXAHEDRON, dtype=np.uint8)
                grid = pv.UnstructuredGrid(cells, celltypes, points)
                
                # Add to plotter with wireframe style for visibility
                self._selection_actor = self.plotter.add_mesh(
                    grid,
                    style='wireframe',
                    color='yellow',
                    line_width=4,
                    name='_selection_highlight',
                    pickable=False,
                    render_lines_as_tubes=False
                )
                
                # --- Make highlight always visible on top ---
                import vtk
                prop = self._selection_actor.GetProperty()
                
                # Bright yellow emissive color (no lighting)
                prop.SetColor(1.0, 1.0, 0.0)  # Bright yellow RGB
                prop.SetAmbient(1.0)  # Full ambient (emissive)
                prop.SetDiffuse(0.0)  # No diffuse lighting
                prop.SetSpecular(0.0)  # No specular highlights
                prop.SetInterpolationToFlat()
                
                # Wireframe properties
                prop.SetRepresentationToWireframe()
                prop.SetLineWidth(4)
                prop.SetEdgeVisibility(True)
                
                # CRITICAL: Disable depth testing and pull wireframe forward
                prop.SetDepthOffset(-10)  # Negative pulls wireframe forward
                prop.SetRenderLinesAsTubes(True)  # Thicker lines
                
                # Force rendering properties
                self._selection_actor.SetForceOpaque(True)
                prop.SetOpacity(1.0)  # Full opacity
                
                # Disable lighting so color is always bright
                prop.SetLighting(False)
                
                logger.info(f"Updated selection visualization: {len(selected_indices)} blocks (wireframe overlay)")
                
                logger.info(f"Updated selection visualization: {len(selected_indices)} blocks")
            
        except Exception as e:
            logger.error(f"Error updating selection visualization: {e}", exc_info=True)
    
    def end_click_selection_mode(self):
        """End click selection mode."""
        try:
            self.plotter.disable_picking()
            
            # Clear selection visualization
            if self._selection_actor is not None:
                self.plotter.remove_actor(self._selection_actor)
                self._selection_actor = None
            
            selected = self._selected_blocks.copy()
            self._selected_blocks = set()
            
            logger.info(f"Ended click selection mode with {len(selected)} blocks selected")
            return selected
            
        except Exception as e:
            logger.error(f"Error ending click selection: {e}")
            return set()
    
    def start_plane_positioning_mode(self, axis='Z', position=None):
        """Start interactive plane positioning for cross-sections."""
        if not self.plotter:
            error_msg = "Cannot start plane positioning: 3D viewer not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not self.current_model:
            error_msg = "Cannot start plane positioning: No block model loaded"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # End any other selection mode
            self.end_selection_mode()
            
            self.selection_mode = 'plane'
            # NOTE: Picking is gated by PickingController, no need to manually disable
            # The selection_mode='plane' check prevents conflicting interactions
            
            # Get model bounds
            bounds = self.plotter.bounds
            
            # Determine plane orientation and initial position
            # If a position is provided but clearly outside bounds (e.g., 0.0 while model is at 172k),
            # clamp it to the model midpoint on that axis.
            ax = axis.upper() if isinstance(axis, str) else 'Z'
            if ax == 'X':
                min_b, max_b = bounds[0], bounds[1]
            elif ax == 'Y':
                min_b, max_b = bounds[2], bounds[3]
            else:
                min_b, max_b = bounds[4], bounds[5]
            mid_b = (min_b + max_b) / 2.0

            if position is None:
                position = mid_b
            else:
                try:
                    p = float(position)
                except Exception:
                    p = mid_b
                # Consider out-of-bounds if outside by more than 2% of span
                span = max(1e-6, (max_b - min_b))
                margin = 0.02 * span
                if p < (min_b - margin) or p > (max_b + margin):
                    p = mid_b
                position = p
            
            # Set normal based on axis
            if ax == 'X':
                normal = (1, 0, 0)
                origin = (position, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2)
            elif ax == 'Y':
                normal = (0, 1, 0)
                origin = ((bounds[0] + bounds[1]) / 2, position, (bounds[4] + bounds[5]) / 2)
            else:  # Z
                normal = (0, 0, 1)
                origin = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, position)
            
            def plane_callback(normal_vec, origin_point):
                """Called when plane is moved."""
                # Determine which axis and position
                if abs(normal_vec[0]) > 0.9:  # X-plane
                    pos = origin_point[0]
                    ax = 'X'
                elif abs(normal_vec[1]) > 0.9:  # Y-plane
                    pos = origin_point[1]
                    ax = 'Y'
                else:  # Z-plane
                    pos = origin_point[2]
                    ax = 'Z'
                
                logger.info(f"Plane moved: {ax}={pos:.2f}")
                self.plane_position_changed.emit(ax, pos)

                # Update visible plane actor to match widget
                try:
                    import pyvista as pv
                    # Determine plane sizes based on bounds and axis
                    x_len = bounds[1] - bounds[0]
                    y_len = bounds[3] - bounds[2]
                    z_len = bounds[5] - bounds[4]
                    if abs(normal_vec[0]) > 0.9:  # X-plane spans YZ
                        i_size, j_size = y_len * 1.05, z_len * 1.05
                    elif abs(normal_vec[1]) > 0.9:  # Y-plane spans XZ
                        i_size, j_size = x_len * 1.05, z_len * 1.05
                    else:  # Z-plane spans XY
                        i_size, j_size = x_len * 1.05, y_len * 1.05
                    new_plane = pv.Plane(center=origin_point, direction=normal_vec, i_size=i_size, j_size=j_size)

                    # Create or update actor
                    if self._plane_actor is None:
                        self._plane_actor = self.plotter.add_mesh(
                            new_plane,
                            color='cyan',
                            opacity=0.25,
                            name='_interactive_plane',
                            pickable=False,
                            show_edges=True,
                            edge_color='magenta',
                            line_width=2,
                        )
                    else:
                        # Replace the underlying mesh by removing and re-adding for compatibility
                        try:
                            self.plotter.remove_actor(self._plane_actor)
                        except Exception:
                            pass
                        self._plane_actor = self.plotter.add_mesh(
                            new_plane,
                            color='cyan',
                            opacity=0.25,
                            name='_interactive_plane',
                            pickable=False,
                            show_edges=True,
                            edge_color='magenta',
                            line_width=2,
                        )
                except Exception as _e:
                    logger.debug(f"Plane actor update skipped: {_e}")
            
            # Create a visible plane actor initially
            try:
                import pyvista as pv
                x_len = bounds[1] - bounds[0]
                y_len = bounds[3] - bounds[2]
                z_len = bounds[5] - bounds[4]
                if ax == 'X':
                    i_size, j_size = y_len * 1.05, z_len * 1.05
                elif ax == 'Y':
                    i_size, j_size = x_len * 1.05, z_len * 1.05
                else:
                    i_size, j_size = x_len * 1.05, y_len * 1.05
                init_plane = pv.Plane(center=origin, direction=normal, i_size=i_size, j_size=j_size)
                # Remove existing if any
                if self._plane_actor is not None:
                    try:
                        self.plotter.remove_actor(self._plane_actor)
                    except Exception:
                        pass
                    self._plane_actor = None
                self._plane_actor = self.plotter.add_mesh(
                    init_plane,
                    color='cyan',
                    opacity=0.25,
                    name='_interactive_plane',
                    pickable=False,
                    show_edges=True,
                    edge_color='magenta',
                    line_width=2,
                )
            except Exception as _e:
                logger.debug(f"Initial plane actor creation skipped: {_e}")

            # Create plane widget for interaction
            self._plane_section_widget = self.plotter.add_plane_widget(
                callback=plane_callback,
                normal=normal,
                origin=origin,
                bounds=bounds,
                factor=1.5,
                normal_rotation=True,
                implicit=False,
                outline_translation=False,
                origin_translation=True
            )
            
            # Emit initial position to sync the panel spinner
            try:
                self.plane_position_changed.emit(ax, float(position))
            except Exception:
                pass

            logger.info(f"Started plane positioning mode: {ax}={position:.2f}")
            parent = self.window()
            if hasattr(parent, 'statusBar'):
                parent.statusBar().showMessage(f"Plane Positioning: Drag plane to position cross-section ({axis}-axis)", 5000)
                
        except Exception as e:
            logger.error(f"Error starting plane positioning: {e}", exc_info=True)
    
    def end_plane_positioning_mode(self):
        """End plane positioning mode."""
        if self._plane_section_widget is not None:
            try:
                # Get final position
                origin = self._plane_section_widget.GetOrigin()
                normal = self._plane_section_widget.GetNormal()
                
                # Remove widget
                self.plotter.clear_plane_widgets()
                self._plane_section_widget = None

                # Remove visible plane actor
                if self._plane_actor is not None:
                    try:
                        self.plotter.remove_actor(self._plane_actor)
                    except Exception:
                        pass
                    self._plane_actor = None
                
                # NOTE: Picking state is managed by PickingController, no force-enable needed

                logger.info(f"Ended plane positioning mode: origin={origin}, normal={normal}")
                return origin, normal
                
            except Exception as e:
                logger.error(f"Error ending plane positioning: {e}")
                return None, None
        return None, None

    def clear_plane_overlay(self):
        """Remove any interactive plane widget and its visible actor, restoring picking."""
        try:
            # Remove widget if present
            if self._plane_section_widget is not None:
                try:
                    self.plotter.clear_plane_widgets()
                except Exception:
                    pass
                self._plane_section_widget = None
            # Remove visible plane actor if present
            if self._plane_actor is not None:
                try:
                    self.plotter.remove_actor(self._plane_actor)
                except Exception:
                    pass
                self._plane_actor = None
            # NOTE: Picking state is managed by PickingController, no force-enable needed
            # Reset selection mode if it was plane
            if getattr(self, 'selection_mode', None) == 'plane':
                self.selection_mode = None
            logger.info("Cleared plane overlay")
            # Render to update view
            try:
                self.plotter.render()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error clearing plane overlay: {e}")
    
    def _on_mouse_move(self, obj, event):
        """
        Handle mouse move events - forward to hover debouncer.
        
        This is called on every mouse move, but the debouncer ensures
        expensive picking only happens when the mouse stops moving.
        
        PERFORMANCE: Early exit if hover not allowed by PickingController.
        """
        try:
            # Early exit if hover not allowed (LOD-P0 or hover disabled)
            if not self._picking_controller.hover_allowed:
                return
            
            vtk_x, vtk_y = obj.GetEventPosition()
            # Convert VTK coords (bottom-left) to Qt coords (top-left) for tooltip
            render_window = obj.GetRenderWindow()
            if render_window:
                size = render_window.GetSize()
                qt_y = size[1] - vtk_y - 1
            else:
                qt_y = vtk_y
            
            # Forward Qt coordinates to debouncer
            if hasattr(self, '_hover_debouncer') and self._hover_debouncer is not None:
                self._hover_debouncer.mouse_moved((vtk_x, qt_y))
        except Exception as e:
            logger.debug(f"Error in mouse move handler: {e}")
        finally:
            # Always call parent handler to preserve default interaction (rotation, pan, etc.)
            try:
                obj.OnMouseMove()
            except Exception:
                pass
    
    def _on_hover_stable(self, pos: tuple[int, int]) -> None:
        """
        Handle hover when mouse stops moving - ACTOR-LEVEL ONLY.
        
        CRITICAL: Hover NEVER uses vtkCellPicker. Only vtkPropPicker.
        This is O(actors) not O(cells), so it's always fast.
        
        Args:
            pos: (x, y) in Qt coordinates (top-left origin) from debouncer
        """
        # Gate on PickingController
        if not self._picking_controller.hover_allowed:
            QToolTip.hideText()
            return
        
        if not self.plotter:
            QToolTip.hideText()
            return
        
        try:
            qt_x, qt_y = pos
            
            vtk_renderer = self.plotter.renderer
            if vtk_renderer is None:
                QToolTip.hideText()
                return
            
            # Convert Qt coords to VTK coords
            vtk_x = qt_x
            vtk_y = self.height() - qt_y - 1
            
            # CRITICAL: Use PROP picker (actor-level), NEVER cell picker
            # This is O(actors) not O(cells) - always fast
            prop_picker = self._picking_controller.get_prop_picker()
            if prop_picker is None:
                QToolTip.hideText()
                return
            
            # Actor-level pick only
            prop_picker.Pick(vtk_x, vtk_y, 0, vtk_renderer)
            picked_actor = prop_picker.GetActor()
            
            if picked_actor is None:
                QToolTip.hideText()
                self._hovered_block_id = None
                return
            
            # Show actor-level tooltip (no cell data access)
            if picked_actor == self.renderer.mesh_actor:
                self._show_tooltip_qt((qt_x, qt_y), "Block Model")
                return
            
            # Check drillhole actors
            if hasattr(self.renderer, '_drillhole_hole_actors'):
                for hole_id, actor in self.renderer._drillhole_hole_actors.items():
                    if picked_actor == actor:
                        self._show_tooltip_qt((qt_x, qt_y), f"Drillhole: {hole_id}")
                        return
            
            # Generic actor tooltip
            self._show_tooltip_qt((qt_x, qt_y), "Object")
            
        except Exception as e:
            logger.debug(f"Hover error: {e}")
    
    def _handle_block_click(self, x: int, y: int):
        """
        Handle block click from Qt event filter.
        
        CRITICAL LOD SEPARATION:
        - LOD-P1: Use vtkPropPicker ONLY (actor-level, fast)
        - LOD-P2+: Use vtkCellPicker (cell-level, expensive)
        
        RULE: Never call cell picker unless LOD >= P2_CELL.
        """
        # Gate on PickingController
        if not self._picking_controller.click_allowed:
            return
        
        if not self.plotter:
            return
        
        vtk_renderer = self.plotter.renderer
        if vtk_renderer is None:
            return
        
        # Convert Qt to VTK coordinates
        vtk_y = self.height() - y
        current_lod = self._picking_controller.lod
        
        # =====================================================================
        # LOD-P1: ACTOR-LEVEL ONLY (fast path)
        # Uses vtkPropPicker - O(actors) not O(cells)
        # MUST complete in < 5ms. NO cell access. NO property reads.
        # =====================================================================
        if current_lod == PickingLOD.P1_ACTOR:
            import time
            _p1_start = time.perf_counter()
            
            prop_picker = self._picking_controller.get_prop_picker()
            if prop_picker is None:
                logger.info("P1 CLICK EXIT: no prop picker")
                return
            
            prop_picker.Pick(x, vtk_y, 0, vtk_renderer)
            picked_actor = prop_picker.GetActor()
            
            if picked_actor is None:
                logger.info("P1 CLICK EXIT: no actor")
                return
            
            # Quick actor identification (no iteration needed for block model)
            layer_name = "Block Model" if picked_actor == self.renderer.mesh_actor else "Object"
            
            # Lightweight status update only - NO signal emission in P1
            # Signals may have expensive receivers
            parent = self.window()
            if hasattr(parent, 'statusBar'):
                parent.statusBar().showMessage(f"Selected: {layer_name}", 2000)
            
            _p1_elapsed = (time.perf_counter() - _p1_start) * 1000
            logger.info(f"P1 CLICK COMPLETE: {layer_name} in {_p1_elapsed:.2f}ms")
            
            # ABSOLUTE EXIT - nothing after this
            return
        
        # =====================================================================
        # LOD-P2+: CELL-LEVEL (expensive path, with timing)
        # Uses vtkCellPicker - O(cells)
        # CRITICAL: vtkCellPicker freezes on large models (>200k cells)
        # =====================================================================
        if current_lod >= PickingLOD.P2_CELL:
            # SAFETY CHECK: Calculate total scene cells to prevent freeze
            total_cells = 0
            layer_cell_counts = {}
            if hasattr(self, 'renderer') and self.renderer and hasattr(self.renderer, 'active_layers'):
                try:
                    for layer_name, layer_info in self.renderer.active_layers.items():
                        layer_data = layer_info.get('data')
                        layer_type = layer_info.get('type', '').lower()

                        # Handle different layer data structures
                        if layer_type == 'drillholes' and isinstance(layer_data, dict):
                            # Drillholes stored as dict with 'hole_polys'
                            if 'hole_polys' in layer_data:
                                hole_cells = sum(poly.n_cells for poly in layer_data['hole_polys'].values() if hasattr(poly, 'n_cells'))
                                total_cells += hole_cells
                                layer_cell_counts[layer_name] = hole_cells
                            else:
                                layer_cell_counts[layer_name] = "no hole_polys"
                        elif hasattr(layer_data, 'n_cells'):
                            # Block models, geology surfaces (PyVista meshes)
                            layer_cells = layer_data.n_cells
                            total_cells += layer_cells
                            layer_cell_counts[layer_name] = layer_cells
                        else:
                            layer_cell_counts[layer_name] = f"no n_cells (type={type(layer_data).__name__})"
                except Exception as e:
                    logger.warning(f"[PICK] Error calculating cells: {e}")
            
            logger.info(f"[PICK CHECK] Total cells: {total_cells:,}, layers: {layer_cell_counts}")
            
            # BLOCK PICKING FOR EXTREME MODELS: >100k cells causes freeze (lowered from 200k for safety)
            PICK_CELL_LIMIT = 100000
            if total_cells > PICK_CELL_LIMIT:
                logger.warning(f"[PICK BLOCKED] Scene has {total_cells:,} cells (limit={PICK_CELL_LIMIT:,}) - cell picking disabled to prevent freeze")
                parent = self.window()
                if hasattr(parent, 'statusBar'):
                    parent.statusBar().showMessage(f"Picking disabled for large model ({total_cells:,} cells)", 3000)
                return
            
            with self._picking_controller.timed_click():
                try:
                    cell_picker = self._picking_controller.get_cell_picker()
                    if cell_picker is None:
                        logger.info("[PICK] Cell picker not available")
                        return

                    cell_picker.Pick(x, vtk_y, 0, vtk_renderer)
                    picked_actor = cell_picker.GetActor()
                    cell_id = cell_picker.GetCellId()

                    logger.info(f"[PICK] Actor: {picked_actor}, Cell ID: {cell_id}")

                    if picked_actor is None or cell_id < 0:
                        logger.info("[PICK] No valid pick (actor is None or cell_id < 0)")
                        return
                    
                    # Get mesh from actor
                    mesh = None
                    mapper = picked_actor.GetMapper()
                    if mapper is not None:
                        vtk_data = mapper.GetInput()
                        if vtk_data is not None:
                            try:
                                mesh = pv.wrap(vtk_data)
                                logger.info(f"[PICK] Mesh extracted, cell_data keys: {list(mesh.cell_data.keys()) if hasattr(mesh, 'cell_data') else 'no cell_data'}")
                            except Exception as e:
                                logger.info(f"[PICK] Failed to wrap VTK data: {e}")

                    if mesh is None:
                        logger.info("[PICK] Mesh is None, cannot extract properties")
                        return

                    # Extract block data
                    if "Original_ID" in mesh.cell_data:
                        logger.info(f"[PICK] Found Original_ID in mesh.cell_data")
                        original_ids = mesh.cell_data["Original_ID"]
                        if 0 <= cell_id < len(original_ids):
                            block_id = int(original_ids[cell_id])
                            
                            if block_id >= 0:
                                # Get properties
                                properties = {}
                                for prop_name in mesh.cell_data.keys():
                                    if prop_name == 'Original_ID':
                                        continue
                                    try:
                                        prop_array = mesh.cell_data[prop_name]
                                        if 0 <= cell_id < len(prop_array):
                                            prop_value = prop_array[cell_id]
                                            if not (isinstance(prop_value, (int, float)) and np.isnan(prop_value)):
                                                properties[prop_name] = prop_value
                                    except Exception:
                                        pass
                                
                                self._selected_blocks = {block_id}
                                
                                self.block_picked.emit(block_id, properties)
                                self.global_pick_event.emit({
                                    'block_id': block_id,
                                    'properties': properties,
                                    'layer': 'Block Model',
                                    'event_type': 'click',
                                    'lod': current_lod.name
                                })
                                
                                prop_summary = ", ".join([f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                                         for k, v in list(properties.items())[:3]])
                                if prop_summary:
                                    parent = self.window()
                                    if hasattr(parent, 'statusBar'):
                                        parent.statusBar().showMessage(f"Block {block_id}: {prop_summary}", 3000)
                    else:
                        # No Original_ID - might be drillhole or geology
                        logger.info(f"[PICK] No Original_ID in mesh.cell_data, available keys: {list(mesh.cell_data.keys())}")

                    # Check if picked actor is a drillhole
                    if picked_actor is not None and cell_id >= 0:
                        if hasattr(self.renderer, 'drillhole_actors') and picked_actor in self.renderer.drillhole_actors.values():
                            # Find which hole was picked
                            for hole_id, actor in self.renderer.drillhole_actors.items():
                                if actor == picked_actor:
                                    # Get drillhole data from active layers
                                    if 'drillholes' in self.renderer.active_layers:
                                        dh_data = self.renderer.active_layers['drillholes']['data']

                                        # Extract segment info if available
                                        properties = {
                                            'hole_id': hole_id,
                                            'layer': 'Drillhole'
                                        }

                                        if 'collar_coords' in dh_data:
                                            collar = dh_data['collar_coords'].get(hole_id)
                                            if collar:
                                                properties['collar_x'] = collar[0]
                                                properties['collar_y'] = collar[1]
                                                properties['collar_z'] = collar[2]

                                        # Emit signal
                                        self.global_pick_event.emit({
                                            'hole_id': hole_id,
                                            'properties': properties,
                                            'layer': f'Drillhole: {hole_id}',
                                            'event_type': 'click',
                                            'lod': current_lod.name
                                        })

                                        # Status bar update
                                        parent = self.window()
                                        if hasattr(parent, 'statusBar'):
                                            parent.statusBar().showMessage(f"Drillhole: {hole_id}", 3000)
                                    break

                except Exception as e:
                    logger.debug(f"P2 click error: {e}")
    
    def _on_mouse_release(self, obj, event):
        """
        Handle mouse release events - track for drag detection.
        
        PERFORMANCE: Actual picking is handled by _handle_block_click via Qt event filter.
        This handler only clears press state to avoid redundant picking.
        """
        # Clear press position (drag detection state)
        self._mouse_press_pos = None
    
    def _on_mouse_click(self, obj, event):
        """
        Handle mouse click events - store press position for drag detection.
        
        NOTE: Actual picking is done via Qt event filter, not this handler.
        This avoids conflicts with VTK camera controls.
        """
        # Always call parent handler first to preserve default interaction
        try:
            obj.OnLeftButtonDown()
        except Exception:
            pass
        
        # Store press position for drag detection (used by Qt filter)
        if self._picking_controller.click_allowed:
            try:
                x, y = obj.GetEventPosition()
                self._mouse_press_pos = (x, y)
            except Exception:
                pass
    
    def _show_tooltip_qt(self, qt_pos: tuple[int, int], text: str) -> None:
        """
        Display tooltip at the given Qt screen position.
        
        Args:
            qt_pos: (x, y) in Qt coordinates (top-left origin)
            text: Tooltip text to display
        """
        try:
            x, y = qt_pos
            # Convert widget-local Qt coordinates to global screen coordinates
            global_pos = self.mapToGlobal(QPoint(x, y))
            QToolTip.showText(global_pos, text, self)
            logger.debug(f"Tooltip shown at Qt({x}, {y}) -> Global({global_pos.x()}, {global_pos.y()})")
        except Exception as e:
            logger.error(f"Error showing tooltip: {e}", exc_info=True)
    
    def _update_hover_highlight(self, block_id: int, cell_id: int, mesh) -> None:
        """
        Update visual highlight for hovered block.
        
        Creates a subtle highlight overlay on the hovered block.
        """
        try:
            # Clear previous hover highlight
            self._clear_hover_highlight()
            
            if not self.plotter or not self.current_model:
                return
            
            # Get block position and dimensions from model
            positions = self.current_model.positions
            dimensions = self.current_model.dimensions
            
            if positions is None or block_id >= len(positions):
                return
            
            # Get block center and size
            center = positions[block_id]
            if dimensions is not None and block_id < len(dimensions):
                size = dimensions[block_id]
            else:
                # Default size if not available
                size = np.array([10.0, 10.0, 10.0])
            
            # Create a slightly larger box for highlight (5% larger)
            import pyvista as pv
            highlight_box = pv.Box(
                bounds=(
                    center[0] - size[0] * 0.525, center[0] + size[0] * 0.525,
                    center[1] - size[1] * 0.525, center[1] + size[1] * 0.525,
                    center[2] - size[2] * 0.525, center[2] + size[2] * 0.525
                )
            )
            
            # Add highlight actor with wireframe style for visibility
            self._hover_highlight_actor = self.plotter.add_mesh(
                highlight_box,
                style='wireframe',
                color='yellow',
                line_width=3,
                name='_hover_highlight',
                pickable=False,  # Don't interfere with picking
                render_lines_as_tubes=False
            )
            
            # --- Make hover highlight always visible on top ---
            import vtk
            prop = self._hover_highlight_actor.GetProperty()
            
            # Bright yellow emissive color (no lighting)
            prop.SetColor(1.0, 1.0, 0.0)  # Bright yellow RGB
            prop.SetAmbient(1.0)  # Full ambient (emissive)
            prop.SetDiffuse(0.0)  # No diffuse lighting
            prop.SetSpecular(0.0)  # No specular highlights
            prop.SetInterpolationToFlat()
            
            # Wireframe properties
            prop.SetRepresentationToWireframe()
            prop.SetLineWidth(3)
            prop.SetEdgeVisibility(True)
            
            # CRITICAL: Disable depth testing and pull wireframe forward
            prop.SetDepthOffset(-5)  # Negative pulls wireframe forward (less than selection)
            prop.SetRenderLinesAsTubes(True)  # Thicker lines
            
            # Force rendering properties
            self._hover_highlight_actor.SetForceOpaque(True)
            prop.SetOpacity(0.8)  # Slightly transparent for hover
            
            # Disable lighting so color is always bright
            prop.SetLighting(False)
            
            logger.debug(f"Hover highlight updated for block {block_id} (wireframe overlay)")
            
        except Exception as e:
            logger.debug(f"Error updating hover highlight: {e}", exc_info=True)
    
    def _clear_hover_highlight(self) -> None:
        """Clear the hover highlight."""
        try:
            if self._hover_highlight_actor is not None and self.plotter is not None:
                self.plotter.remove_actor(self._hover_highlight_actor)
                self._hover_highlight_actor = None
        except Exception:
            pass
    
    def end_selection_mode(self):
        """End any active selection mode."""
        if self.selection_mode == 'box':
            self.end_box_selection_mode()
        elif self.selection_mode == 'click':
            self.end_click_selection_mode()
        elif self.selection_mode == 'plane':
            self.end_plane_positioning_mode()
        
        self.selection_mode = None
        logger.info("Ended selection mode")
