"""
3D rendering engine using PyVista/VTK.
"""

import pyvista as pv
import vtk
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Iterable, Set, Callable
import logging
import threading

from ..models.block_model import BlockModel
from .lod_manager import LODManager
from .decimation import decimate_block_grid, decimate_mesh, get_mesh_complexity
from ..utils.profiling import profile_section
from .overlay_manager import OverlayManager
from .pyvista_axes_scalebar import FloatingAxes
from .visual_density_controller import VisualDensityController

# REMOVED: PyVistaOverlayManager stub
# All overlay functionality is now handled by the unified OverlayManager.
# This stub has been removed to prevent duplicate actor tracking and coordinate drift.
from ..drillholes.drillhole_layer import build_drillhole_polylines
from ..drillholes.datamodel import DrillholeDatabase
from matplotlib.colors import ListedColormap
from .picking_controller import get_picking_controller

logger = logging.getLogger(__name__)


class _ResizeDebounceFilter:
    """
    Monkey-patches the PyVista QtInteractor's resizeEvent to debounce VTK renders
    when a large model is loaded.

    PROBLEM: When a docked panel is resized, Qt resizes the VTK widget, whose
    resizeEvent calls render_window.SetSize() → ConfigureEvent() → Render().
    With 500K+ cells each Render() is heavy GPU work, and 40+ back-to-back calls
    on the main thread cause a hard UI freeze.

    WHY NOT an event filter? Returning False lets VTK's handler run (still freezes).
    Returning True blocks the base QWidget resize (breaks layout). Neither works.

    SOLUTION: Replace the QtInteractor's resizeEvent with a wrapper that updates
    the render window size (so Qt layout stays correct) but SKIPS the VTK Render().
    A 150 ms single-shot timer fires one deferred render after the user stops dragging.
    """

    def __init__(self, renderer_ref):
        from PyQt6.QtCore import QTimer

        self._renderer = renderer_ref
        self._active = False
        self._original_resizeEvent = None  # Saved original method
        self._interactor = None
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._deferred_render)

    def install(self, interactor_widget):
        """Monkey-patch the interactor's resizeEvent."""
        import types
        self._interactor = interactor_widget
        self._original_resizeEvent = interactor_widget.resizeEvent

        debounce = self  # closure ref

        def _patched_resizeEvent(self_widget, event):
            if debounce._active:
                # Update render window size (keeps Qt layout correct) but SKIP Render()
                debounce._resize_without_render(self_widget, event)
                debounce._timer.start()
            else:
                # Normal path — call original VTK resizeEvent
                debounce._original_resizeEvent(event)

        interactor_widget.resizeEvent = types.MethodType(_patched_resizeEvent, interactor_widget)
        logger.info("[RESIZE DEBOUNCE] Monkey-patched QtInteractor.resizeEvent")

    def activate(self):
        self._active = True
        logger.info("[RESIZE DEBOUNCE] Activated — resize renders will be debounced")

    def deactivate(self):
        self._active = False
        if self._timer.isActive():
            self._timer.stop()
        logger.info("[RESIZE DEBOUNCE] Deactivated — normal resize rendering restored")

    def _resize_without_render(self, widget, event):
        """Update VTK render window size to match Qt widget, but don't render."""
        try:
            from PyQt6.QtWidgets import QWidget
            # Let QWidget base class handle geometry/layout (NOT VTK's override)
            QWidget.resizeEvent(widget, event)

            # Sync VTK render window size so it's ready when we do render
            renderer = self._renderer
            if renderer and renderer.plotter and renderer.plotter.render_window:
                rw = renderer.plotter.render_window
                dpr = widget.devicePixelRatio()
                w = int(widget.width() * dpr)
                h = int(widget.height() * dpr)
                if w > 0 and h > 0:
                    rw.SetSize(w, h)
                    # Also update the VTK interactor's size (without triggering render)
                    iren = rw.GetInteractor()
                    if iren:
                        iren.SetSize(w, h)
                        iren.UpdateSize(w, h)
        except Exception as e:
            logger.debug(f"[RESIZE DEBOUNCE] _resize_without_render: {e}")

    def _deferred_render(self):
        """Execute a single render after resize settles."""
        try:
            renderer = self._renderer
            if renderer is None or renderer.plotter is None:
                return
            plotter = renderer.plotter
            rw = plotter.render_window
            if rw:
                # Fire ConfigureEvent so VTK knows about the new size
                iren = rw.GetInteractor()
                if iren:
                    iren.ConfigureEvent()
                rw.Render()
            plotter.render()
            logger.debug("[RESIZE DEBOUNCE] Deferred render complete")
        except Exception as e:
            logger.warning(f"[RESIZE DEBOUNCE] Deferred render error: {e}")


class Renderer:
    """
    3D rendering engine for block models using PyVista/VTK.
    
    Handles mesh generation, camera controls, and rendering settings.
    """
    
    def __init__(self, registry: Any = None):
        """
        Initialize the renderer.
        
        Args:
            registry: Optional DataRegistry for persistent interval IDs (critical for GPU picking)
        """
        self.plotter: Optional[pv.Plotter] = None
        self._original_show_bounds = None  # Saved before monkey-patching
        self.block_meshes: Dict[str, pv.PolyData] = {}
        self.current_model: Optional[BlockModel] = None
        self.visible_blocks: Optional[np.ndarray] = None
        self._drillhole_hole_actors: Dict[str, Any] = {}
        self._drillhole_collar_actors: Dict[str, Any] = {}

        # Layer management (must be initialized early for all code paths)
        self.active_layers = {}
        self.scene_layers = {}

        # Default settings (must be initialized early)
        self.current_colormap = 'viridis'
        self.default_opacity = {
            'drillhole': 1.0,
            'volume': 0.6,
            'blocks': 0.8,
            'mesh': 0.7,
            'schedule': 0.8,
            'pit': 0.7,
            'default': 0.7
        }

        # Drillhole caching and rendering (must be initialized early)
        self._drillhole_polylines_cache: Optional[Dict[str, Any]] = None
        self._drillhole_label_actor = None
        self._gpu_drillhole_renderer: Optional[Any] = None
        self._use_gpu_drillholes: bool = False
        self.visual_density_controller: Optional[Any] = None
        self.drillhole_visual_density_enabled: bool = True

        # Layer change callback for UI updates
        self.layer_change_callback = None

        # State manager for drillholes
        self._drillhole_state_manager = None

        # Geology bounds for camera calculations (surfaces + solids)
        self._geology_bounds: Optional[Tuple[float, float, float, float, float, float]] = None
        
        # ============================================================================
        # SINGLE SPACE AUTHORITY (CRITICAL FOR GPU FLOATING-POINT PRECISION)
        # ============================================================================
        # Large UTM coordinates (e.g., 500,000m) exceed GPU 32-bit float precision.
        # The GPU ignores decimal places beyond ~10cm at these magnitudes.
        # SOLUTION: Lock a SINGLE global shift on first dataset load.
        # All subsequent data (surfaces, drillholes, blocks) use the SAME shift.
        # This prevents "jitter" and "picking drift" from inconsistent transforms.
        # ============================================================================
        self._global_shift: Optional[np.ndarray] = None  # LOCKED on first dataset
        self._global_shift_lock = threading.Lock()  # Thread-safe initialization
        self._local_origin: Optional[np.ndarray] = None  # Alias for backward compatibility
        self._fixed_bounds: Optional[Tuple[float, float, float, float, float, float]] = None  # Locked scene bounds

        # VTK observer tracking for cleanup (prevents memory leaks)
        self._observer_tags: List[Tuple[str, Any, int]] = []  # (type, object, tag)

        # DataRegistry reference for persistent interval IDs (GPU picking stability)
        self._registry = registry

        # Resize debounce filter — prevents UI freeze when resizing panels with large models
        # Installed in initialize_plotter(), activated when _has_large_model is set
        self._resize_debounce: Optional[_ResizeDebounceFilter] = None
    
    def _to_local_precision(self, points: np.ndarray) -> np.ndarray:
        """
        THE TRANSFORM GATE: Single Space Authority for all coordinate transforms.
        
        Every point entering the renderer MUST pass through this gate.
        The first dataset locks the global shift; all subsequent data uses the SAME shift.
        
        This prevents:
        - Video jitter (inconsistent GPU precision)
        - Picking drift (actor coordinates changing between loads)
        - Overlay instability (HUD actors moving relative to data)
        
        Args:
            points: World coordinates (UTM, lat/lon, etc.)
            
        Returns:
            Local coordinates centered at (0, 0, 0) for GPU precision
        """
        # Thread-safe global shift initialization
        with self._global_shift_lock:
            if self._global_shift is None:
                # LOCK THE SHIFT: First dataset defines the coordinate authority
                self._global_shift = np.mean(points, axis=0)
                logger.info("=" * 80)
                logger.info("SPACE AUTHORITY LOCKED")
                logger.info("=" * 80)
                logger.info(f"Global shift: [{self._global_shift[0]:.2f}, {self._global_shift[1]:.2f}, {self._global_shift[2]:.2f}]")
                logger.info("All subsequent data will use this SAME shift for coordinate stability")
                logger.info("=" * 80)
                # Backward compatibility alias
                self._local_origin = self._global_shift
            shift = self._global_shift
        return points - shift
    
    def attach_registry(self, registry: Any) -> None:
        """
        Attach DataRegistry for persistent interval IDs.
        
        CRITICAL: Registry must be attached before rendering drillholes to ensure
        GPU picking uses persistent GLOBAL_INTERVAL_IDs instead of temporary counters.
        
        Args:
            registry: DataRegistry instance
        """
        self._registry = registry
        logger.info("Attached DataRegistry to renderer for persistent interval IDs")
        
        # Rendering settings
        self.show_axes = False
        self.show_bounds = True
        self.show_grid = False
        
        # Display settings - Professional rendering (Leapfrog-style)
        self.current_opacity = 1.0
        self.edge_color = 'black'  # Black edges for better color balance and less brightness
        self.background_color = 'lightgrey'
        self.lighting_enabled = True
        self.edge_width = 2.0  # Thicker edges for clear block boundaries in mining software style
        
        # Lighting/shading parameters
        self.ambient = 0.4
        self.diffuse = 0.6
        self.specular = 0.2
        self.specular_power = 15
        self.smooth_shading = False
        
        # Legend settings
        self.legend_position = 'vertical'  # 'vertical' or 'horizontal'
        self.legend_location = 'upper_right'  # PyVista location string
        self.legend_font_size = 13
        self.legend_visible = False  # Hide legend bar by default
        self.legend_x = 0.85  # X position for vertical
        self.legend_y = 0.05  # Y position
        self.current_scalar_bar = None
        
        # Axis font settings
        self.axis_font_size = 12
        self.axis_font_color = 'black'
        self.axis_font_family = 'Arial'
        
        # Mesh actor reference for dynamic updates
        self.mesh_actor = None
        self._current_grid_signature: Optional[Tuple[int, int]] = None

        # ============================================================================
        # UNIFIED LAYER MANAGEMENT SYSTEM
        # ============================================================================
        # DYNAMIC layer management - supports ANY visualization type
        # Layers are added on-the-fly as visualizations are created
        self.active_layers = {}  # Empty dict - layers added dynamically
        
        # Default opacity settings for different layer types
        self.default_opacity = {
            'drillhole': 1.0,
            'volume': 0.6,
            'blocks': 0.8,
            'mesh': 0.7,
            'schedule': 0.8,
            'pit': 0.7,
            'default': 0.7
        }
        
        # Current active property for block model
        self.current_property = None
        self.current_colormap = 'viridis'
        self._drillhole_legend_metadata: Optional[Dict[str, Any]] = None

        # Layer change callback for UI updates (already initialized at line 182 - don't overwrite!)
        
        # ============================================================================
        # GLOBAL PICKING SYSTEM
        # ============================================================================
        # Scene layer registry for picking and inspection
        self.scene_layers = {}  # layer_name -> {actor, data, type, properties}
        
        # Picking callbacks
        self.pick_callback = None  # Callback for pick events: func(info_dict)
        self.picking_enabled = True
        self.last_picked_actor = None
        
        # ============================================================================
        # LOD AND PERFORMANCE MANAGEMENT
        # ============================================================================
        # LOD Manager for automatic level-of-detail rendering
        self.lod_manager: Optional[LODManager] = None
        self.lod_config: Dict[str, Any] = {'lod_quality': 0.7}

        # Visual Density Controller for drillhole LOD based on camera distance
        self.visual_density_controller: Optional[VisualDensityController] = None
        self.drillhole_visual_density_enabled: bool = True  # Developer toggle

        # Sampling settings
        self.sampling_enabled = False
        self.sampling_factor = 1  # 1 = no sampling, 2 = every 2nd block, etc.
        
        # ============================================================================
        # OVERLAY SYSTEM - UNIFIED MANAGER
        # ============================================================================
        # Unified overlay manager (single source of truth for all overlay logic)
        self.overlay_state = OverlayManager(renderer=self)
        # Legacy alias for backward compatibility
        self.overlay_manager = self.overlay_state
        # REMOVED: overlay_engine stub (prevents duplicate actor tracking)
        
        # ============================================================================
        # OVERLAY ACTOR TRACKING (managed by OverlayManager, drawn here)
        # ============================================================================
        self._overlay_actors: Dict[str, List[Any]] = {
            'axes': [],
            'scale_bar': [],
            'grid': [],
            'bounds': [],
            'other': [],
        }
        
        # PyVista Floating Axes and HUD scale bar control
        self.floating_axes: Optional[FloatingAxes] = None
        self.scale_bar_widget = None
        self.show_floating_axes = False
        self.show_scale_bar_3d = False
        self._floating_axes_params: Optional[Dict[str, Any]] = None
        self._scale_bar_params: Optional[Dict[str, Any]] = None
        self._scale_bar_callback_registered = False
        self._scale_bar_render_cb = None
        self._scale_bar_camera_cb = None
        self._scale_bar_anchor_name = "bottom_right"

        # North arrow widget control
        self.north_arrow_widget = None
        self.show_north_arrow = False
        self._north_arrow_params: Optional[Dict[str, Any]] = None
        self._north_arrow_callback_registered = False
        self._north_arrow_camera_cb = None
        self._north_arrow_anchor_name = "top_right"
        
        # Fixed scene bounds (locked when model loads)
        self._fixed_scene_bounds: Optional[Tuple[float, float, float, float, float, float]] = None
        self.max_blocks_render = 500_000  # Maximum blocks to render (can be adjusted)
        
        # Drillhole label actor
        self._drillhole_label_actor = None
        self._selected_drillhole_interval: Optional[Any] = None  # Selected interval for info panel
        
        # Drillhole caching for performance
        self._drillhole_hole_actors: Dict[str, Any] = {}  # hole_id -> actor
        self._drillhole_collar_actors: Dict[str, Any] = {}  # hole_id -> collar marker actor
        self._drillhole_polylines_cache: Optional[Dict[str, Any]] = None  # Cached polylines data
        
        # GPU-optimized drillhole renderer (optional, for large datasets)
        self._gpu_drillhole_renderer: Optional[Any] = None  # DrillholeGPURenderer instance
        self._use_gpu_drillholes: bool = False  # Auto-enable for large datasets
        
        # State manager integration
        try:
            from ..visualization.drillhole_state import get_drillhole_state_manager
            self._drillhole_state_manager = get_drillhole_state_manager()
        except Exception as e:
            logger.debug(f"Could not initialize drillhole state manager: {e}")
            self._drillhole_state_manager = None
        
        # Measurement system
        self.measure_mode = None  # 'distance' or 'area' or None
        self.measure_points = []  # List of clicked points
        self.measure_actors = []  # List of actors for measurement lines/shapes
        self.measurements = []  # List of completed measurements
        self._measure_line_actor = None  # Temporary line while measuring
        self._measure_polygon_actor = None  # Temporary polygon while measuring
        self._measure_callback_id = None  # Callback ID for removal
        self._drillhole_hole_actors: Dict[str, Any] = {}
        self._drillhole_collar_actors: Dict[str, Any] = {}
        
    def initialize_plotter(self, plotter: pv.Plotter, parent_window=None) -> None:
        """
        Initialize the plotter for rendering.
        
        Args:
            plotter: PyVista plotter instance
            parent_window: Optional parent window (for compatibility)
        """
        self.plotter = plotter

        # Save original PyVista methods before monkey-patching (for floating axes)
        self._original_show_bounds = getattr(plotter, 'show_bounds', None)

        # Disable ALL PyVista overlays (replaced by HUD system)
        try:
            self.plotter.show_axes = False
            if hasattr(self.plotter, 'hide_axes'):
                self.plotter.hide_axes()
            # show_grid may be a property or method depending on PyVista version
            try:
                self.plotter.show_grid = False
            except (TypeError, AttributeError):
                try:
                    self.plotter.show_grid(False)
                except TypeError:
                    # Newer PyVista: show_grid() takes no arguments
                    self.plotter.show_grid()
            # Remove scalar bar if it exists (non-fatal if it doesn't)
            try:
                self.plotter.remove_scalar_bar()
            except (StopIteration, AttributeError, KeyError):
                # Scalar bar doesn't exist, which is fine
                pass
            # Remove bounds axes
            try:
                self.plotter.remove_bounds_axes()
            except Exception as e:
                logger.debug(f"Could not remove bounds axes: {e}")
            # Remove all lights (HUD doesn't need them)
            try:
                self.plotter.remove_all_lights()
            except Exception as e:
                logger.debug(f"Could not remove lights: {e}")
            logger.info("Disabled all PyVista overlays (using HUD system instead)")
        except Exception as e:
            logger.debug(f"Could not disable PyVista overlays: {e}")
        
        self._setup_plotter()
        
        # Initialize LOD Manager if not already initialized
        if self.lod_manager is None:
            self.lod_manager = LODManager(self.lod_config)

        # Initialize Visual Density Controller for drillhole LOD
        if self.visual_density_controller is None:
            self.visual_density_controller = VisualDensityController(
                plotter=self.plotter,
                enabled=self.drillhole_visual_density_enabled,
                debug_logging=False,  # Can be enabled via developer toggle
            )
        
        # CRITICAL: Add camera callback to maintain clipping range based on model bounds
        # This prevents HUD overlay actors from affecting clipping planes during interactive rotation
        try:
            camera = self.plotter.renderer.GetActiveCamera()
            if camera:
                # Store callback reference to prevent garbage collection
                self._camera_clipping_callback = lambda obj, event: self._maintain_clipping_range()
                tag = camera.AddObserver("ModifiedEvent", self._camera_clipping_callback)
                self._observer_tags.append(('camera', camera, tag))
                logger.debug("Added camera callback to maintain clipping range")
        except Exception as e:
            logger.debug(f"Could not add camera clipping callback: {e}")
        
        # ============================================================================
        # FIX: FREEZE OVERLAYS DURING CAMERA INTERACTION
        # ============================================================================
        # PROBLEM: Rebuilding overlays during camera movement causes:
        #   - Video jitter (actors disappear/reappear mid-frame)
        #   - Picking drift (actor IDs change during interaction)
        #   - LOD thrashing (geometry swaps while user is looking)
        # SOLUTION: Suspend overlay updates when user starts interacting,
        #           resume only after mouse release and camera is still.
        # ============================================================================
        try:
            iren = self.plotter.iren
            if iren:
                # FREEZE: Stop all overlay updates when user starts moving camera
                tag1 = iren.AddObserver("StartInteractionEvent", lambda o, e: self.overlay_state.suspend_updates())
                # RESUME: Restart overlay updates only after interaction ends
                tag2 = iren.AddObserver("EndInteractionEvent", lambda o, e: self.overlay_state.resume_updates(force=True))
                self._observer_tags.extend([('interactor', iren, tag1), ('interactor', iren, tag2)])
                logger.info("Hooked VTK interaction events to freeze overlays during camera movement")
        except Exception as e:
            logger.warning(f"Could not hook VTK interaction events: {e}")
        
        # ============================================================================
        # FIX: DEBOUNCE RESIZE RENDERS FOR LARGE MODELS
        # ============================================================================
        # PROBLEM: Panel resize fires 40+ Qt resize events per second. Each one
        # triggers VTK Render() with 500K+ cells, freezing the UI thread.
        # SOLUTION: Install an event filter that suppresses VTK renders during
        # resize and fires a single deferred render after the user stops dragging.
        # The filter is installed now but only activated when _has_large_model is set.
        # ============================================================================
        try:
            interactor = getattr(self.plotter, 'interactor', None)
            if interactor is not None:
                self._resize_debounce = _ResizeDebounceFilter(self)
                self._resize_debounce.install(interactor)
                # If a large model is already loaded, activate immediately
                if getattr(self, '_has_large_model', False):
                    self._resize_debounce.activate()
                logger.info("[RESIZE DEBOUNCE] Ready (will activate on large model load)")
        except Exception as e:
            logger.warning(f"Could not install resize debounce: {e}")
        
        # Initialize unified overlay manager (attach renderer reference)
        try:
            if hasattr(self.overlay_state, 'attach_renderer'):
                self.overlay_state.attach_renderer(self)
                logger.info("Attached renderer to unified OverlayManager")
            
            # Sync overlay state with renderer settings
            logger.info(f"Syncing overlay manager with renderer state: show_axes={self.show_axes}, show_grid={self.show_grid}, show_bounds={self.show_bounds}")
            
            # Enable overlays based on initial renderer settings
            if self.show_grid:
                self.overlay_state.toggle_overlay("ground_grid", True)
            if self.show_bounds:
                self.overlay_state.toggle_overlay("bounds", True)
        except Exception as e:
            logger.warning(f"Could not initialize overlay manager: {e}", exc_info=True)
        
        logger.info("Initialized renderer plotter")
        
        # If we already have a block model loaded, add it to the plotter now
        if self.current_model is not None and self.block_meshes:
            logger.info("Plotter initialized, adding existing block model meshes")
            self._add_meshes_to_plotter()

            # Register as a layer if mesh_actor was created
            if self.mesh_actor is not None:
                # Support both ImageData and UnstructuredGrid
                mesh_data = (
                    self.block_meshes.get('imagedata') or
                    self.block_meshes.get('unstructured_grid') or
                    self.block_meshes.get('point_cloud')
                )

                # Get unique layer name based on file name
                layer_name = self._get_block_model_layer_name(self.current_model)

                self.add_layer(
                    layer_name=layer_name,
                    actor=self.mesh_actor,
                    data=mesh_data,
                    layer_type='blocks',
                    opacity=self.current_opacity
                )
                logger.info(f"Registered existing block model as active layer '{layer_name}'")
    
    def _setup_plotter(self) -> None:
        """Setup plotter with professional rendering settings."""
        if self.plotter is None:
            return
        
        # Set background
        self.plotter.set_background(self.background_color)
        
        # Enable professional rendering features
        # CRITICAL: Enable anti-aliasing for Text3D labels to render properly
        # Without this, HUD text labels will be invisible
        try:
            # Try different anti-aliasing methods (newer PyVista uses string names)
            if hasattr(self.plotter, 'enable_anti_aliasing'):
                try:
                    self.plotter.enable_anti_aliasing('msaa')  # Multi-sample anti-aliasing
                except (TypeError, ValueError):
                    try:
                        self.plotter.enable_anti_aliasing('fxaa')  # Fast approximate anti-aliasing
                    except (TypeError, ValueError):
                        # Fallback: try numeric value for older PyVista versions
                        self.plotter.enable_anti_aliasing(8)
        except Exception as e:
            logger.warning(f"Could not enable anti-aliasing: {e}")
        
        # LAG FIX: Disable expensive eye dome lighting for better performance
        # try:
        #     self.plotter.enable_eye_dome_lighting()  # Enhanced depth perception - DISABLED
        # except Exception as e:
        #     logger.warning(f"Could not enable eye dome lighting: {e}")
        
        # QUALITY FIX: Enable depth peeling for proper transparency and z-fighting prevention
        # This is critical for overlapping geological surfaces and solids
        try:
            renderer = self.plotter.renderer
            if renderer:
                # BALANCED SETTINGS: Depth peeling disabled to prevent flickering/disappearing
                renderer.SetUseDepthPeeling(0)
                # renderer.SetMaximumNumberOfPeels(6)  # Balanced quality/performance
                # renderer.SetOcclusionRatio(0.05)  # Minimal early termination for sharp edges
                
                # CRITICAL FIX: Enable two-sided lighting globally
                # This ensures back-facing surfaces are properly illuminated
                # Without this, surfaces can appear dark/black when viewed from certain angles
                renderer.SetTwoSidedLighting(True)
                
                logger.debug("Enabled depth peeling and two-sided lighting for quality rendering")
        except Exception as e:
            logger.debug(f"Could not enable depth peeling: {e}")
        
        # DISABLED: PyVista axes/bounds - use unified OverlayManager instead
        # The OverlayManager handles axes and grids via FloatingAxes/ScaleBar3D
        # Do not call show_axes() or add_bounding_box() here
        
        # FULL AXIS WIPEOUT - Remove all VTK/PyVista axes actors (red tick labels, cube axes, etc.)
        try:
            renderer = self.plotter.renderer
            
            # STEP 1: Nuke VTKCubeAxesActor and VTKCubeAxesActor2D (the source of red labels)
            # Iterate through all actors and remove cube axes actors
            actors_to_remove = []
            actors_collection = renderer.GetActors()
            actors_collection.InitTraversal()
            actor = actors_collection.GetNextItem()
            while actor is not None:
                if actor.IsA("vtkCubeAxesActor") or actor.IsA("vtkCubeAxesActor2D"):
                    actors_to_remove.append(actor)
                actor = actors_collection.GetNextItem()
            
            for actor in actors_to_remove:
                try:
                    renderer.RemoveActor(actor)
                except Exception as e:
                    logger.debug(f"Could not remove cube axes actor: {e}")
            
            # Disable all PyVista axes methods
            self.plotter.show_axes = False
            if hasattr(self.plotter, 'hide_axes'):
                self.plotter.hide_axes()
            # show_grid may be a property or method depending on PyVista version
            try:
                self.plotter.show_grid = False
            except (TypeError, AttributeError):
                try:
                    self.plotter.show_grid(False)
                except TypeError:
                    # Newer PyVista: show_grid() takes no arguments
                    self.plotter.show_grid()
            
            # STEP 2: Disable PyVista autoload of cube axes (set internal flags)
            self.plotter._show_bounds = False
            if hasattr(self.plotter, '_cube_axes_actor'):
                self.plotter._cube_axes_actor = None
            if hasattr(self.plotter, '_cube_axes_actor2d'):
                self.plotter._cube_axes_actor2d = None
            
            # Remove bounds axes
            try:
                self.plotter.remove_bounds_axes()
            except Exception as e:
                logger.debug(f"Could not remove bounds axes: {e}")
            
            # Remove bounding box
            try:
                self.plotter.remove_bounding_box()
            except Exception as e:
                logger.debug(f"Could not remove bounding box: {e}")
            
            # Remove axes actor from renderer
            try:
                if hasattr(self.plotter.renderer, 'axes_actor') and self.plotter.renderer.axes_actor is not None:
                    self.plotter.renderer.RemoveActor(self.plotter.renderer.axes_actor)
                    self.plotter.renderer.axes_actor = None
            except Exception as e:
                logger.debug(f"Could not remove axes actor: {e}")
            
            # STEP 3: Monkey-patch PyVista methods to prevent cube axes creation
            def _noop_axes(*args, **kwargs):
                return None
            self.plotter.add_axes = _noop_axes
            self.plotter.show_bounds = _noop_axes
            self.plotter.add_bounding_box = _noop_axes
            
            # Also monkey-patch at module level if possible
            try:
                import pyvista as pv
                if hasattr(pv.plotting, 'widgets') and hasattr(pv.plotting.widgets, 'WidgetHelper'):
                    pv.plotting.widgets.WidgetHelper.add_bounds_axes = lambda *args, **kwargs: None
                if hasattr(pv.plotting, 'plotting') and hasattr(pv.plotting.plotting, 'Plotter'):
                    # Don't override the class method, just the instance method
                    pass
            except Exception as e:
                logger.debug(f"Could not monkey-patch PyVista methods: {e}")
            
            # Remove scalar bars (defensive: PyVista can raise StopIteration if none exist)
            try:
                self.plotter.remove_scalar_bar()
            except Exception as e:
                logger.debug(f"Could not remove scalar bar: {e}")
            
            # Disable camera settings that might trigger axes
            try:
                self.plotter.renderer.GetActiveCamera().SetParallelProjection(False)
                # REMOVED: Depth peeling now controlled centrally in _apply_geology_z_fighting_fixes
                # DO NOT enable depth peeling here as it conflicts with geology rendering
                # self.plotter.renderer.SetUseDepthPeeling(1)
                # self.plotter.renderer.SetMaximumNumberOfPeels(8)  # Balance quality vs performance
                # self.plotter.renderer.SetOcclusionRatio(0.0)  # Always use depth peeling
            except Exception as e:
                logger.debug(f"Could not configure camera/depth peeling: {e}")
        except Exception as e:
            logger.debug(f"Error removing axes actors: {e}", exc_info=True)
    
    def _remove_all_axes_actors(self) -> None:
        """
        Remove all VTK/PyVista axes actors (red tick labels, cube axes, etc.).
        Call this after any operation that might create axes (e.g., add_mesh).
        """
        if self.plotter is None:
            return
        
        try:
            renderer = self.plotter.renderer
            
            # STEP 1: Nuke VTKCubeAxesActor and VTKCubeAxesActor2D (the source of red labels)
            # Iterate through all actors and remove cube axes actors
            actors_to_remove = []
            actors_collection = renderer.GetActors()
            actors_collection.InitTraversal()
            actor = actors_collection.GetNextItem()
            while actor is not None:
                if actor.IsA("vtkCubeAxesActor") or actor.IsA("vtkCubeAxesActor2D"):
                    actors_to_remove.append(actor)
                actor = actors_collection.GetNextItem()
            
            for actor in actors_to_remove:
                try:
                    renderer.RemoveActor(actor)
                except Exception:
                    pass
            
            # Remove axes actor from renderer
            if hasattr(self.plotter.renderer, 'axes_actor') and self.plotter.renderer.axes_actor is not None:
                self.plotter.renderer.RemoveActor(self.plotter.renderer.axes_actor)
                self.plotter.renderer.axes_actor = None
            
            # STEP 2: Disable PyVista autoload of cube axes (set internal flags)
            self.plotter._show_bounds = False
            if hasattr(self.plotter, '_cube_axes_actor'):
                self.plotter._cube_axes_actor = None
            if hasattr(self.plotter, '_cube_axes_actor2d'):
                self.plotter._cube_axes_actor2d = None
            
            # Disable all axes methods
            self.plotter.show_axes = False
            if hasattr(self.plotter, 'hide_axes'):
                self.plotter.hide_axes()
            self.plotter.show_grid(False)
            
            # Remove bounds axes and bounding box
            try:
                self.plotter.remove_bounds_axes()
            except Exception:
                pass
            try:
                self.plotter.remove_bounding_box()
            except Exception:
                pass
        except Exception:
            pass  # Silently fail - axes removal is best-effort
    
    def _add_labeled_axes_with_grid(self) -> None:
        """DISABLED: Use unified OverlayManager instead of PyVista axes/bounds."""
        # This method is disabled - axes and grids are handled by unified OverlayManager
        # via FloatingAxes and ScaleBar3D classes
        pass
    
    def refresh_bounds(self) -> None:
        """Refresh the bounds/grid display with current font settings."""
        if self.plotter is None or self.current_model is None:
            return
        
        # Refresh by re-adding block model mesh (preserves other layers)
        # Note: _add_meshes_to_plotter now only removes block model actors
        self._setup_plotter()
        
        # Re-add meshes
        self._add_meshes_to_plotter()
        
        # Force render
        self.plotter.render()
        
        logger.info("Refreshed bounds with new settings")
    
    def close_overlays(self) -> None:
        """
        Close and clean up overlay actors during shutdown.
        This method is called during application shutdown to cleanly remove overlay actors.
        """
        try:
            # Clear all overlay actors tracked by the renderer
            self.clear_overlay_actors()
            
            # Clear floating axes
            if self.floating_axes is not None and self.plotter is not None:
                try:
                    self.floating_axes.clear(self.plotter)
                except Exception:
                    pass
                self.floating_axes = None
            
            # Notify unified overlay manager to close
            if hasattr(self, 'overlay_state') and self.overlay_state is not None:
                try:
                    self.overlay_state.close()
                except Exception:
                    pass
            
            logger.debug("Closed overlay actors")
        except Exception as e:
            logger.debug(f"Error closing overlays: {e}", exc_info=True)
    
    def _get_block_model_layer_name(self, block_model: BlockModel) -> str:
        """
        Get a unique layer name for a block model based on its source file.

        This allows multiple block models to coexist in the scene with unique names.

        Args:
            block_model: BlockModel to get layer name for

        Returns:
            Unique layer name (e.g., "Block Model: production_2024" or "Block Model")
        """
        from pathlib import Path

        # Try to get file name from block model metadata
        file_name = None

        # Method 1: Check block model metadata
        if hasattr(block_model, 'metadata'):
            metadata = block_model.metadata
            if hasattr(metadata, 'source_file') and metadata.source_file:
                file_name = Path(metadata.source_file).stem
            elif hasattr(metadata, 'file_path') and metadata.file_path:
                file_name = Path(metadata.file_path).stem

        # Method 2: Check DataRegistry metadata
        if not file_name:
            try:
                # Try to get from DataRegistry if available
                from ..core.data_registry import DataRegistry
                registry = DataRegistry.get_instance()
                if registry:
                    bm_metadata = registry.get_block_model_metadata()
                    if bm_metadata and bm_metadata.metadata:
                        # Check for file_path in metadata dict
                        file_path = bm_metadata.metadata.get('file_path') or bm_metadata.metadata.get('source_path')
                        if file_path:
                            file_name = Path(file_path).stem
            except Exception as e:
                logger.debug(f"Could not get file name from registry: {e}")

        # Generate layer name
        if file_name:
            layer_name = f"Block Model: {file_name}"
            logger.info(f"[LAYER_NAME] Using unique layer name: '{layer_name}'")
        else:
            layer_name = "Block Model"
            logger.info(f"[LAYER_NAME] Using default layer name: '{layer_name}'")

        return layer_name

    def load_block_model(self, block_model: BlockModel) -> None:
        """
        Load a block model for rendering.

        Args:
            block_model: BlockModel to render
        """
        import time
        renderer_start = time.time()

        # Clear previous meshes to free GPU memory
        self._clear_block_meshes()

        self.current_model = block_model
        self.visible_blocks = np.arange(block_model.block_count)
        
        # Suspend overlay updates during model load to prevent race conditions
        overlay_suspended = False
        if hasattr(self, 'overlay_state') and self.overlay_state is not None:
            try:
                self.overlay_state.suspend_updates()
                overlay_suspended = True
                logger.debug("Overlay updates suspended during block model load")
            except Exception:
                overlay_suspended = False

        try:
            # Generate meshes for all blocks
            mesh_start = time.time()
            self._generate_block_meshes()
            mesh_time = time.time() - mesh_start
            logger.info(f"PERF: _generate_block_meshes took {mesh_time:.3f}s")

            # ================================================================
            # CRITICAL FIX: Apply coordinate transformation to generated meshes
            # ================================================================
            # Block models must use the SAME coordinate shift as drillholes and geology
            # to render together in the same scene. Without this, block models remain
            # at UTM coordinates (e.g., 500,000m) while other layers are shifted to local
            # coordinates (~0,0,0), causing camera positioning issues.
            # ================================================================
            if self.block_meshes:
                transform_start = time.time()
                self._apply_coordinate_transform_to_meshes()
                transform_time = time.time() - transform_start
                logger.info(f"PERF: Coordinate transformation took {transform_time:.3f}s")

            # Add meshes to plotter (only if plotter is initialized)
            if self.plotter is not None:
                add_start = time.time()
                self._add_meshes_to_plotter()
                add_time = time.time() - add_start
                logger.info(f"PERF: _add_meshes_to_plotter took {add_time:.3f}s")

                # Set basic camera position before render so blocks are visible immediately
                # This prevents blocks from being off-screen when first rendered
                # CRITICAL: Use transformed mesh bounds, not original model bounds, for camera positioning
                try:
                    # Get bounds from transformed mesh
                    bounds = None
                    for mesh_key, mesh in self.block_meshes.items():
                        if mesh_key.startswith('_') or mesh is None:
                            continue
                        if hasattr(mesh, 'bounds'):
                            bounds = mesh.bounds
                            logger.info(f"[CAMERA] Using transformed mesh bounds from '{mesh_key}': {bounds}")
                            break

                    # Fallback to model bounds if no mesh bounds available
                    if bounds is None and self.current_model and self.current_model.bounds is not None:
                        bounds = self.current_model.bounds
                        # Transform to local coordinates if global shift is set
                        if self._global_shift is not None:
                            shift = self._global_shift
                            bounds = (
                                bounds[0] - shift[0], bounds[1] - shift[0],
                                bounds[2] - shift[1], bounds[3] - shift[1],
                                bounds[4] - shift[2], bounds[5] - shift[2],
                            )
                        logger.info(f"[CAMERA] Using model bounds (fallback): {bounds}")

                    if bounds is not None:
                        center = [
                            (bounds[0] + bounds[1]) / 2,
                            (bounds[2] + bounds[3]) / 2,
                            (bounds[4] + bounds[5]) / 2
                        ]
                        size = max(
                            bounds[1] - bounds[0],
                            bounds[3] - bounds[2],
                            bounds[5] - bounds[4]
                        )
                        distance = size * 2.0
                        # Use plotter.camera_position property for efficient camera setting
                        self.plotter.camera_position = [
                            [center[0] + distance * 0.7, center[1] + distance * 0.7, center[2] + distance * 0.7],
                            center,
                            [0, 0, 1]
                        ]

                        # CRITICAL: Set camera clipping range based on model bounds only
                        # This prevents HUD overlay actors from affecting clipping planes during rotation
                        camera = self.plotter.renderer.GetActiveCamera()
                        if camera:
                            # QUALITY FIX: Improved clipping for close-up inspection
                            model_size = size
                            cam_distance = distance  # Initial camera distance
                            # Very small near plane to allow zooming in without geometry disappearing
                            near_plane = max(0.001, min(cam_distance * 0.0001, model_size * 0.00001))
                            far_plane = max(model_size * 50.0, cam_distance * 100.0)
                            camera.SetClippingRange(near_plane, far_plane)
                            logger.info(f"[CAMERA] Set clipping range: near={near_plane:.6f}, far={far_plane:.1f}")

                        logger.info(f"[CAMERA] Set initial camera position: center={center}, distance={distance:.1f}")
                    else:
                        logger.warning("[CAMERA] No bounds available for camera positioning")
                except Exception as e:
                    logger.debug(f"Could not set initial camera position: {e}")

                # Force immediate render with full Qt/VTK synchronization
                render_start = time.time()
                self.force_render_update()
                render_time = time.time() - render_start
                logger.info(f"PERF: Initial render after load took {render_time:.3f}s")
            else:
                logger.warning("PERF: Plotter not initialized, skipping mesh addition")
                add_time = 0
                render_time = 0

            renderer_time = time.time() - renderer_start
            logger.info(f"PERF: Renderer.load_block_model total: {renderer_time:.3f}s (mesh: {mesh_time:.3f}s, add: {add_time:.3f}s, render: {render_time:.3f}s)")

            # Register as a layer in the active layers system
            if self.plotter is not None:
                logger.info(f"LAYER_DEBUG: mesh_actor = {self.mesh_actor}, block_meshes keys = {list(self.block_meshes.keys())}")
                if self.mesh_actor is not None:
                    # Support both ImageData and UnstructuredGrid
                    mesh_data = (
                        self.block_meshes.get('imagedata') or
                        self.block_meshes.get('unstructured_grid') or
                        self.block_meshes.get('point_cloud')
                    )

                    # Get unique layer name based on file name
                    layer_name = self._get_block_model_layer_name(block_model)

                    logger.info(f"LAYER_DEBUG: Calling add_layer with mesh_data type = {type(mesh_data)}")
                    self.add_layer(
                        layer_name=layer_name,
                        actor=self.mesh_actor,
                        data=mesh_data,
                        layer_type='blocks',
                        opacity=self.current_opacity
                    )
                    logger.info(f"Registered block model as active layer '{layer_name}'")
                else:
                    logger.warning("LAYER_DEBUG: mesh_actor is None, cannot register as layer!")
            else:
                logger.info("Plotter not initialized yet, meshes will be added when plotter is ready")

            logger.info(f"Loaded block model with {block_model.block_count} blocks")

            # CRITICAL: Remove any cube axes that might have been created during mesh addition
            # This must be called after all mesh operations to ensure cube axes are removed
            if self.plotter is not None:
                self._remove_all_axes_actors()
            
            # Update floating axes bounds if enabled
            if self.show_floating_axes:
                try:
                    self._update_floating_axes_bounds()
                except Exception as e:
                    logger.error(f"Error updating floating axes bounds: {e}", exc_info=True)
            
            # Update scene bounds (delegates to unified overlay manager)
            self._update_scene_bounds()
        finally:
            # Resume overlay updates after model load
            if overlay_suspended and hasattr(self, 'overlay_state') and self.overlay_state is not None:
                try:
                    self.overlay_state.resume_updates(force=True)
                    logger.debug("Overlay updates resumed after block model load")
                except Exception:
                    pass
    
    def _generate_block_meshes(self) -> None:
        """
        Generate PyVista meshes for all blocks.
        
        ARCHITECTURAL RULE: Regular block models MUST use pv.ImageData (UniformGrid).
        Only rotated or sub-blocked models may use pv.UnstructuredGrid.
        
        This rule ensures:
        - 40-100× memory reduction for regular grids (3-4GB → 40-120MB for 10M blocks)
        - 5-15× performance improvement (40-120 FPS vs 2-8 FPS)
        - Enables large model visualization on standard hardware
        
        Uses centralized detection from block_model_mesh_builder module.
        """
        if self.current_model is None:
            return
        
        # Clear existing meshes
        self.block_meshes.clear()
        
        block_count = self.current_model.block_count
        if block_count > 0:
            # Use centralized mesh builder with auto-detection
            from .block_model_mesh_builder import generate_block_model_mesh
            
            try:
                mesh = generate_block_model_mesh(self.current_model)
                
                # Store mesh based on type
                if isinstance(mesh, pv.ImageData):
                    self.block_meshes['imagedata'] = mesh
                    self.block_meshes['unstructured_grid'] = mesh  # Also store for compatibility
                    self.block_meshes['_is_imagedata'] = True
                    
                    # Calculate memory savings estimate
                    unstructured_memory_mb = (8 * block_count * 3 * 8) / (1024**2)
                    imagedata_memory_mb = (3 * 8 + 3 * 8 + 3 * 4) / (1024**2)
                    savings_mb = unstructured_memory_mb - imagedata_memory_mb
                    
                    logger.info(f"✅ UNIFORM GRID - Using ImageData (UniformGrid) for {block_count:,} blocks")
                    logger.info(f"   Memory savings: ~{savings_mb:.1f}MB geometry memory avoided (40-100× reduction)")
                    logger.info(f"   Performance: Expected 40-120 FPS vs 2-8 FPS with UnstructuredGrid")
                else:
                    self.block_meshes['unstructured_grid'] = mesh
                    self.block_meshes['_is_imagedata'] = False
                    
                    logger.info(f"⚠️  NON-UNIFORM GRID - Using UnstructuredGrid for {block_count:,} blocks")
                    logger.info(f"   Note: UnstructuredGrid uses explicit geometry (higher memory)")
                
            except Exception as e:
                logger.exception("Failed to generate block model mesh, falling back to UnstructuredGrid")
                # Fallback: Always use UnstructuredGrid if detection fails
                from .block_model_mesh_builder import build_unstructured_grid
                try:
                    mesh = build_unstructured_grid(self.current_model)
                    self.block_meshes['unstructured_grid'] = mesh
                    self.block_meshes['_is_imagedata'] = False
                    logger.info("Fallback: Created UnstructuredGrid mesh")
                except Exception as e2:
                    logger.exception("Failed to build UnstructuredGrid fallback")
                    raise RuntimeError(f"Failed to generate block model mesh: {e2}") from e2
            return
        
        # Get block corners
        corners = self.current_model.get_block_corners()
        
        # Create individual block meshes
        for i in range(len(corners)):
            block_corners = corners[i]
            
            # Create box mesh from corners
            # Define faces for a box (6 faces, each with 4 vertices)
            faces = np.array([
                [4, 0, 1, 2, 3],  # Bottom face
                [4, 4, 7, 6, 5],  # Top face
                [4, 0, 4, 5, 1],  # Front face
                [4, 2, 6, 7, 3],  # Back face
                [4, 0, 3, 7, 4],  # Left face
                [4, 1, 5, 6, 2]   # Right face
            ])
            
            # Create mesh
            mesh = pv.PolyData(block_corners, faces)
            self.block_meshes[f'block_{i}'] = mesh

    def _apply_coordinate_transform_to_meshes(self) -> None:
        """
        Apply coordinate transformation to all generated block meshes.

        This ensures block models use the SAME coordinate shift as drillholes and geology
        for proper camera positioning and scene rendering.

        CRITICAL: This must be called AFTER _generate_block_meshes() but BEFORE
        _add_meshes_to_plotter() to ensure meshes are in local coordinates when added to scene.
        """
        if not self.block_meshes:
            return

        # Get a representative point from the block model to lock the global shift
        if self.current_model and self.current_model.bounds is not None:
            bounds = self.current_model.bounds
            center_point = np.array([
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ]).reshape(1, 3)

            # Lock global shift if not already set, or use existing shift
            _ = self._to_local_precision(center_point)

            if self._global_shift is not None:
                shift = self._global_shift
                logger.info(f"[BLOCK LOAD] Applying coordinate shift to block meshes: shift=[{shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f}]")

                # Apply shift to all meshes based on their type
                for mesh_key, mesh in self.block_meshes.items():
                    if mesh_key.startswith('_'):  # Skip metadata keys like '_is_imagedata'
                        continue

                    if mesh is None:
                        continue

                    # Check if mesh has already been shifted
                    if getattr(mesh, '_coordinate_shifted', False):
                        logger.debug(f"[BLOCK LOAD] Mesh '{mesh_key}' already shifted, skipping")
                        continue

                    try:
                        if isinstance(mesh, pv.RectilinearGrid):
                            # RectilinearGrid: shift the x, y, z edge arrays
                            mesh.x = mesh.x - shift[0]
                            mesh.y = mesh.y - shift[1]
                            mesh.z = mesh.z - shift[2]
                            logger.debug(f"[BLOCK LOAD] Shifted RectilinearGrid '{mesh_key}'")

                        elif isinstance(mesh, pv.ImageData):
                            # ImageData: shift the origin
                            origin = np.array(mesh.origin)
                            mesh.origin = tuple(origin - shift)
                            logger.debug(f"[BLOCK LOAD] Shifted ImageData '{mesh_key}' origin")

                        elif hasattr(mesh, 'points'):
                            # StructuredGrid/UnstructuredGrid/PolyData: shift points directly
                            mesh.points = mesh.points - shift
                            logger.debug(f"[BLOCK LOAD] Shifted points for '{mesh_key}' ({len(mesh.points)} points)")

                        # Mark mesh as shifted to prevent double-shifting
                        mesh._coordinate_shifted = True

                    except Exception as e:
                        logger.warning(f"[BLOCK LOAD] Failed to shift mesh '{mesh_key}': {e}")

                # Log final bounds after transformation
                for mesh_key, mesh in self.block_meshes.items():
                    if mesh_key.startswith('_') or mesh is None:
                        continue
                    try:
                        shifted_bounds = mesh.bounds
                        logger.info(
                            f"[BLOCK LOAD] Mesh '{mesh_key}' transformed bounds: "
                            f"({shifted_bounds[0]:.2f}, {shifted_bounds[1]:.2f}, "
                            f"{shifted_bounds[2]:.2f}, {shifted_bounds[3]:.2f}, "
                            f"{shifted_bounds[4]:.2f}, {shifted_bounds[5]:.2f})"
                        )
                        break  # Only log one mesh for brevity
                    except Exception:
                        pass

    def _is_orthogonal_grid(self, positions: np.ndarray, dimensions: np.ndarray, 
                            tolerance: float = 1e-6) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]]:
        """
        Detect if block model is orthogonal (uniform grid, axis-aligned, no rotation).
        
        CRITICAL MEMORY OPTIMIZATION: This detection enables ImageData (UniformGrid) usage,
        which uses ZERO geometry memory vs UnstructuredGrid's ~4GB for 5M blocks.
        
        Detection checks:
        1. Uniform cell sizes (all blocks have same dx, dy, dz)
        2. Regular grid spacing (positions form regular XYZ grid)
        3. Axis-aligned (no rotation - rotated models won't align to regular grid)
        
        If all checks pass → Use ImageData (massive memory savings)
        If any check fails → Use UnstructuredGrid (handles rotated/irregular models)
        
        Args:
            positions: (N, 3) array of block center positions
            dimensions: (N, 3) array of block dimensions
            tolerance: Tolerance for floating point comparisons
            
        Returns:
            If orthogonal: Tuple of (origin, spacing, dimensions) for ImageData
            If not orthogonal: None (indicates rotated/irregular model)
        """
        if len(positions) == 0:
            return None
        
        # Check 1: All blocks must have the same dimensions (uniform grid)
        unique_dims = np.unique(dimensions, axis=0)
        if len(unique_dims) > 1:
            # Check if dimensions are close enough (within tolerance)
            dim_std = np.std(dimensions, axis=0)
            if np.any(dim_std > tolerance):
                logger.debug(f"Non-uniform block dimensions detected (std: {dim_std})")
                return None
        
        # Get uniform block size
        spacing = dimensions[0].copy()
        
        # Check 2: Blocks must form a regular grid pattern
        # Find unique X, Y, Z coordinates
        unique_x = np.unique(positions[:, 0])
        unique_y = np.unique(positions[:, 1])
        unique_z = np.unique(positions[:, 2])
        
        # Check if coordinates are evenly spaced
        if len(unique_x) > 1:
            x_diffs = np.diff(np.sort(unique_x))
            x_spacing = np.median(x_diffs)
            if np.any(np.abs(x_diffs - x_spacing) > tolerance):
                logger.debug(f"X coordinates not evenly spaced")
                return None
        else:
            x_spacing = spacing[0]
        
        if len(unique_y) > 1:
            y_diffs = np.diff(np.sort(unique_y))
            y_spacing = np.median(y_diffs)
            if np.any(np.abs(y_diffs - y_spacing) > tolerance):
                logger.debug(f"Y coordinates not evenly spaced")
                return None
        else:
            y_spacing = spacing[1]
        
        if len(unique_z) > 1:
            z_diffs = np.diff(np.sort(unique_z))
            z_spacing = np.median(z_diffs)
            if np.any(np.abs(z_diffs - z_spacing) > tolerance):
                logger.debug(f"Z coordinates not evenly spaced")
                return None
        else:
            z_spacing = spacing[2]
        
        # Verify spacing matches block dimensions
        if (np.abs(x_spacing - spacing[0]) > tolerance or
            np.abs(y_spacing - spacing[1]) > tolerance or
            np.abs(z_spacing - spacing[2]) > tolerance):
            logger.debug(f"Spacing mismatch: expected {spacing}, got ({x_spacing}, {y_spacing}, {z_spacing})")
            return None
        
        # Check 3: All blocks must be present (complete grid)
        # For a complete grid, we should have nx * ny * nz blocks
        nx, ny, nz = len(unique_x), len(unique_y), len(unique_z)
        expected_blocks = nx * ny * nz
        
        if len(positions) != expected_blocks:
            logger.debug(f"Incomplete grid: expected {expected_blocks} blocks, got {len(positions)}")
            # Still allow if it's a subset (could be masked), but check if positions align to grid
            # Check if all positions align to the grid
            x_min, y_min, z_min = positions.min(axis=0)
            for pos in positions:
                # Check if position aligns to grid
                x_idx = round((pos[0] - x_min) / x_spacing)
                y_idx = round((pos[1] - y_min) / y_spacing)
                z_idx = round((pos[2] - z_min) / z_spacing)
                expected_pos = np.array([
                    x_min + x_idx * x_spacing,
                    y_min + y_idx * y_spacing,
                    z_min + z_idx * z_spacing
                ])
                if np.any(np.abs(pos - expected_pos) > tolerance):
                    logger.debug(f"Position {pos} does not align to grid")
                    return None
        
        # Check 4: Blocks must be axis-aligned (no rotation)
        # Verify that block centers align to grid (simpler check)
        x_min_pos = positions[:, 0].min()
        y_min_pos = positions[:, 1].min()
        z_min_pos = positions[:, 2].min()
        
        # Sample check: verify positions align to grid
        sample_size = min(100, len(positions))
        sample_indices = np.random.choice(len(positions), sample_size, replace=False)
        for idx in sample_indices:
            pos = positions[idx]
            # Check if position aligns to grid
            x_offset = (pos[0] - x_min_pos) % x_spacing
            y_offset = (pos[1] - y_min_pos) % y_spacing
            z_offset = (pos[2] - z_min_pos) % z_spacing
            # Allow for small floating point errors
            if (x_offset > tolerance and abs(x_offset - x_spacing) > tolerance or
                y_offset > tolerance and abs(y_offset - y_spacing) > tolerance or
                z_offset > tolerance and abs(z_offset - z_spacing) > tolerance):
                logger.debug(f"Position {pos} does not align to grid (offsets: {x_offset}, {y_offset}, {z_offset})")
                return None
        
        # All checks passed - this is an orthogonal grid
        # Calculate origin (bottom-left-back corner of first block)
        half_dims = spacing / 2.0
        origin = (positions[:, 0].min() - half_dims[0],
                  positions[:, 1].min() - half_dims[1],
                  positions[:, 2].min() - half_dims[2])
        
        grid_dims = (nx, ny, nz)
        
        logger.info(f"Detected orthogonal grid: origin={origin}, spacing={spacing}, dimensions={grid_dims}")
        return (origin, tuple(spacing), grid_dims)
    
    def _generate_imagedata_meshes(self, grid_info: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]) -> None:
        """
        Generate ImageData (UniformGrid) meshes for orthogonal block models.
        
        This is the memory-efficient path for regular block models (40-100× less memory than UnstructuredGrid).
        
        Args:
            grid_info: Tuple of (origin, spacing, dimensions) from BlockModel.is_orthogonal() or _is_orthogonal_grid()
        """
        import time
        start_time = time.time()
        
        if self.current_model is None:
            return
        
        positions = self.current_model.positions
        dimensions = self.current_model.dimensions
        properties = self.current_model.properties
        
        if positions is None or dimensions is None:
            return
        
        # Apply sampling if enabled
        indices = None
        if self.sampling_enabled and self.sampling_factor > 1:
            logger.info(f"Applying sampling factor {self.sampling_factor} to {len(positions)} blocks")
            from .decimation import decimate_block_grid
            sampled_model = decimate_block_grid(self.current_model, self.sampling_factor, preserve_properties=True)
            positions = sampled_model.positions
            dimensions = sampled_model.dimensions
            if positions is not None:
                original_positions = self.current_model.positions
                indices = []
                for pos in positions:
                    matches = np.where(
                        (np.abs(original_positions[:, 0] - pos[0]) < 1e-6) &
                        (np.abs(original_positions[:, 1] - pos[1]) < 1e-6) &
                        (np.abs(original_positions[:, 2] - pos[2]) < 1e-6)
                    )[0]
                    if len(matches) > 0:
                        indices.append(matches[0])
                indices = np.array(indices) if indices else np.arange(len(positions))
            else:
                indices = np.arange(len(positions))
            filtered_properties = sampled_model.properties
        elif len(positions) > self.max_blocks_render:
            logger.info(f"Large model detected ({len(positions)} blocks), applying sampling to fit {self.max_blocks_render} block limit")
            from .decimation import decimate_block_grid
            sampling_factor = max(1, int(np.ceil(len(positions) / self.max_blocks_render)))
            sampled_model = decimate_block_grid(self.current_model, sampling_factor, preserve_properties=True)
            positions = sampled_model.positions
            dimensions = sampled_model.dimensions
            if positions is not None:
                original_positions = self.current_model.positions
                indices = []
                for pos in positions:
                    matches = np.where(
                        (np.abs(original_positions[:, 0] - pos[0]) < 1e-6) &
                        (np.abs(original_positions[:, 1] - pos[1]) < 1e-6) &
                        (np.abs(original_positions[:, 2] - pos[2]) < 1e-6)
                    )[0]
                    if len(matches) > 0:
                        indices.append(matches[0])
                indices = np.array(indices) if indices else np.arange(len(positions))
            else:
                indices = np.arange(len(positions))
            filtered_properties = sampled_model.properties
        else:
            filtered_properties = properties
            indices = np.arange(len(positions))
        
        # Store filtered data
        self._filtered_positions = positions
        self._filtered_dimensions = dimensions
        self._filtered_properties = filtered_properties
        self._filtered_indices = indices if indices is not None else np.arange(len(positions))
        
        # Recompute grid_info for sampled model if needed
        if len(positions) != self.current_model.block_count:
            # Re-check orthogonality for sampled model
            if hasattr(self.current_model, 'is_orthogonal'):
                try:
                    # Create temporary model for sampling check
                    from ..models.block_model import BlockModel
                    temp_model = BlockModel()
                    temp_model.set_geometry(positions, dimensions)
                    is_orth, grid_info = temp_model.is_orthogonal()
                    if not is_orth:
                        logger.warning("Sampled model is not orthogonal, falling back to UnstructuredGrid")
                        self._generate_optimized_meshes()
                        return
                except Exception:
                    grid_info = self._is_orthogonal_grid(positions, dimensions)
                    if grid_info is None:
                        logger.warning("Sampled model is not orthogonal, falling back to UnstructuredGrid")
                        self._generate_optimized_meshes()
                        return
        
        # Create ImageData grid
        grid = self._create_imagedata_grid(positions, dimensions, filtered_properties, grid_info, indices)
        
        # Store grid
        self.block_meshes['unstructured_grid'] = grid  # Keep name for compatibility
        self.block_meshes['imagedata'] = grid  # Also store as 'imagedata' for explicit access
        self.block_meshes['_is_imagedata'] = True  # Flag to indicate ImageData
        
        # Calculate memory savings
        num_blocks = len(positions)
        unstructured_memory_mb = (8 * num_blocks * 3 * 8) / (1024**2)  # 8 points * blocks * 3 coords * 8 bytes
        imagedata_memory_mb = (3 * 8 + 3 * 8 + 3 * 4) / (1024**2)  # origin + spacing + dims (negligible)
        savings_mb = unstructured_memory_mb - imagedata_memory_mb
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Created ImageData grid: {num_blocks:,} blocks, {len(filtered_properties)} properties in {elapsed:.2f}s")
        logger.info(f"   Memory savings: ~{savings_mb:.1f}MB geometry memory avoided (40-100× reduction)")
        logger.info(f"   Performance: Expected 40-120 FPS vs 2-8 FPS with UnstructuredGrid")
    
        return grid

    def _create_imagedata_grid(self, positions: np.ndarray, dimensions: np.ndarray,
                               properties: Dict[str, np.ndarray],
                               grid_info: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]],
                               indices: Optional[np.ndarray] = None) -> pv.ImageData:
        """
        Create PyVista ImageData (UniformGrid) from orthogonal block model.
        """
        origin, spacing, (nx, ny, nz) = grid_info
        
        # =========================================================
        # CRITICAL: Apply Global Shift via Transform Gate
        # =========================================================
        # We must shift the ImageData ORIGIN to match the Geology local origin.
        # This prevents the 500km spatial split that caused "additional figures".
        origin_shifted = self._to_local_precision(np.array(origin).reshape(1, 3)).flatten()
        
        # Create ImageData grid
        grid = pv.ImageData()
        grid.origin = tuple(origin_shifted)
        grid.spacing = spacing
        # VTK ImageData dimensions are number of points (cells + 1)
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        
        # Get the actual minimum block center positions
        x_min_center = positions[:, 0].min()
        y_min_center = positions[:, 1].min()
        z_min_center = positions[:, 2].min()
        
        # OPTIMIZED: Vectorized grid index calculation
        # Calculate grid indices for all positions at once
        x_indices = np.round((positions[:, 0] - x_min_center) / spacing[0]).astype(np.int32)
        y_indices = np.round((positions[:, 1] - y_min_center) / spacing[1]).astype(np.int32)
        z_indices = np.round((positions[:, 2] - z_min_center) / spacing[2]).astype(np.int32)
        
        # Validate indices (filter out-of-bounds)
        valid_mask = (
            (x_indices >= 0) & (x_indices < nx) &
            (y_indices >= 0) & (y_indices < ny) &
            (z_indices >= 0) & (z_indices < nz)
        )
        
        if not np.all(valid_mask):
            invalid_count = np.sum(~valid_mask)
            logger.warning(f"{invalid_count} blocks have invalid grid indices (out of bounds)")
            # Filter to valid indices only
            x_indices = x_indices[valid_mask]
            y_indices = y_indices[valid_mask]
            z_indices = z_indices[valid_mask]
            valid_positions = positions[valid_mask]
        else:
            valid_positions = positions
        
        # Create 3D arrays for properties (shape: nz, ny, nx) - OPTIMIZED with vectorized assignment
        for prop_name, prop_values in properties.items():
            if len(prop_values) != len(positions):
                logger.warning(f"Property '{prop_name}' has {len(prop_values)} values, expected {len(positions)}")
                continue
            
            # Filter property values to match valid positions
            if not np.all(valid_mask):
                prop_values = prop_values[valid_mask]
            
            # Initialize array with NaN for missing blocks
            prop_array = np.full((nz, ny, nx), np.nan, dtype=prop_values.dtype)
            
            # OPTIMIZED: Vectorized assignment using advanced indexing
            # This is much faster than looping for large models
            prop_array[z_indices, y_indices, x_indices] = prop_values
            
            # Add to grid (VTK uses C-order, so ravel is correct)
            grid.cell_data[prop_name] = prop_array.ravel(order='C')
            logger.debug(f"Added property '{prop_name}' to ImageData grid ({len(prop_values)} values)")
        
        # CRITICAL FOR PICKING: Store Original_ID mapping for efficient block lookup
        # This enables O(1) picking instead of O(N) search
        if indices is not None:
            # Create array matching VTK cell ordering (same as properties)
            original_id_array = np.full((nz, ny, nx), -1, dtype=np.int64)
            original_id_array[z_indices, y_indices, x_indices] = indices
            grid.cell_data['Original_ID'] = original_id_array.ravel(order='C')
            logger.debug(f"Added Original_ID to ImageData grid for picking ({len(indices)} mappings)")
        else:
            # If no indices provided, use sequential IDs
            original_id_array = np.full((nz, ny, nx), -1, dtype=np.int64)
            sequential_ids = np.arange(len(valid_positions), dtype=np.int64)
            original_id_array[z_indices, y_indices, x_indices] = sequential_ids
            grid.cell_data['Original_ID'] = original_id_array.ravel(order='C')
            logger.debug(f"Added sequential Original_ID to ImageData grid for picking")
        
        return grid
    
    def _generate_optimized_meshes(self) -> None:
        """Generate optimized meshes for large models using UnstructuredGrid with LOD and sampling support."""
        if self.current_model is None:
            return
        
        import time
        start_time = time.time()
        with profile_section("mesh_generation"):
            positions = self.current_model.positions
            dimensions = self.current_model.dimensions
            if positions is None or dimensions is None:
                return
            
            # Debug: Print position ranges
            logger.info(f"Position ranges - X: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f}")
            logger.info(f"Position ranges - Y: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f}")
            logger.info(f"Position ranges - Z: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f}")
            
            # Apply sampling if enabled
            indices = None
            if self.sampling_enabled and self.sampling_factor > 1:
                logger.info(f"Applying sampling factor {self.sampling_factor} to {len(positions)} blocks")
                # Use decimation utility for proper spatial sampling
                sampled_model = decimate_block_grid(self.current_model, self.sampling_factor, preserve_properties=True)
                positions = sampled_model.positions
                dimensions = sampled_model.dimensions
                # Map indices back to original model
                if positions is not None:
                    # Create mapping by matching positions
                    original_positions = self.current_model.positions
                    indices = []
                    for pos in positions:
                        matches = np.where(
                            (np.abs(original_positions[:, 0] - pos[0]) < 1e-6) &
                            (np.abs(original_positions[:, 1] - pos[1]) < 1e-6) &
                            (np.abs(original_positions[:, 2] - pos[2]) < 1e-6)
                        )[0]
                        if len(matches) > 0:
                            indices.append(matches[0])
                    indices = np.array(indices) if indices else np.arange(len(positions))
                else:
                    indices = np.arange(len(positions))
                filtered_properties = sampled_model.properties
            elif len(positions) > self.max_blocks_render:
                # For very large models, use intelligent sampling
                logger.info(f"Large model detected ({len(positions)} blocks), applying sampling to fit {self.max_blocks_render} block limit")
                # Calculate sampling factor
                sampling_factor = max(1, int(np.ceil(len(positions) / self.max_blocks_render)))
                sampled_model = decimate_block_grid(self.current_model, sampling_factor, preserve_properties=True)
                positions = sampled_model.positions
                dimensions = sampled_model.dimensions
                if positions is not None:
                    # Map indices
                    original_positions = self.current_model.positions
                    indices = []
                    for pos in positions:
                        matches = np.where(
                            (np.abs(original_positions[:, 0] - pos[0]) < 1e-6) &
                            (np.abs(original_positions[:, 1] - pos[1]) < 1e-6) &
                            (np.abs(original_positions[:, 2] - pos[2]) < 1e-6)
                        )[0]
                        if len(matches) > 0:
                            indices.append(matches[0])
                    indices = np.array(indices) if indices else np.arange(len(positions))
                else:
                    indices = np.arange(len(positions))
                filtered_properties = sampled_model.properties
            else:
                filtered_properties = self.current_model.properties
                indices = np.arange(len(positions))
            
            # Store the filtered data for later use
            self._filtered_positions = positions
            self._filtered_dimensions = dimensions
            self._filtered_properties = filtered_properties
            # Map from filtered cell id -> original block index for filtering
            self._filtered_indices = indices if indices is not None else np.arange(len(positions))
            
            # MEMORY OPTIMIZATION: Check if model is orthogonal (uniform grid, axis-aligned)
            # If orthogonal, use ImageData (UniformGrid) which uses almost zero geometry memory
            # instead of UnstructuredGrid which stores 8 points per block
            grid_info = self._is_orthogonal_grid(positions, dimensions)
            
            if grid_info is not None:
                # Use ImageData for orthogonal models (massive memory savings)
                # This is the same approach used by Leapfrog, Vulcan, Deswik, and Seequent
                num_blocks = len(positions)
                memory_saved_mb = (8 * num_blocks * 3 * 8) / (1024**2)  # 8 points * blocks * 3 coords * 8 bytes
                
                logger.info(f"✅ ORTHOGONAL GRID DETECTED - Using ImageData (UniformGrid)")
                logger.info(f"   Blocks: {num_blocks:,}")
                logger.info(f"   Memory savings: ~{memory_saved_mb:.1f}MB geometry memory avoided")
                logger.info(f"   Performance: Expected 40-120 FPS vs 2-8 FPS with UnstructuredGrid")
                
                grid = self._create_imagedata_grid(positions, dimensions, filtered_properties, grid_info, indices)
                self.block_meshes['unstructured_grid'] = grid  # Store as 'unstructured_grid' for compatibility
                self.block_meshes['imagedata'] = grid  # Also store as 'imagedata' for explicit access
                self.block_meshes['_is_imagedata'] = True  # Flag to indicate ImageData
                elapsed = time.time() - start_time
                logger.info(f"Created ImageData grid with {num_blocks:,} blocks and {len(grid.cell_data)} properties in {elapsed:.2f}s")
                return
            
            # Fall back to UnstructuredGrid for non-orthogonal models (rotated/irregular)
            # Rotated models from Vulcan "Rotated Block Model" or geological transforms
            # must use UnstructuredGrid as they cannot be represented as UniformGrid
            logger.info(f"⚠️  NON-ORTHOGONAL GRID DETECTED - Using UnstructuredGrid")
            logger.info(f"   Reason: Model appears rotated, skewed, or irregular")
            logger.info(f"   Note: UnstructuredGrid uses explicit geometry (higher memory)")
            self.block_meshes['_is_imagedata'] = False
            
            # Create UnstructuredGrid with hexahedrons using vectorized operations
            # This is much faster than Python loops for large models
            num_blocks = len(positions)
            
            # Calculate half dimensions for all blocks at once (vectorized)
            half_dims = dimensions / 2.0  # Shape: (num_blocks, 3)
            
            # Generate all 8 corners for all blocks using vectorized operations
            # Corner offsets relative to center (8 corners x 3 dimensions)
            corner_offsets = np.array([
                [-1, -1, -1],  # 0: bottom-left-back
                [ 1, -1, -1],  # 1: bottom-right-back
                [ 1,  1, -1],  # 2: bottom-right-front
                [-1,  1, -1],  # 3: bottom-left-front
                [-1, -1,  1],  # 4: top-left-back
                [ 1, -1,  1],  # 5: top-right-back
                [ 1,  1,  1],  # 6: top-right-front
                [-1,  1,  1],  # 7: top-left-front
            ], dtype=np.float64)  # Shape: (8, 3)
            
            # Broadcast to compute all corners: (num_blocks, 8, 3)
            # positions[:, None, :] -> (num_blocks, 1, 3)
            # corner_offsets[None, :, :] -> (1, 8, 3)
            # half_dims[:, None, :] -> (num_blocks, 1, 3)
            corners = positions[:, None, :] + corner_offsets[None, :, :] * half_dims[:, None, :]
            # Reshape to (num_blocks * 8, 3) for all points
            all_points = corners.reshape(-1, 3)
            # =========================================================
            # CRITICAL: Apply Global Shift via Transform Gate
            # =========================================================
            # Large UTM coordinates (500km+) crash precision.
            # We subtract the Space Authority's Locked Shift to move to (0,0,0).
            all_points = self._to_local_precision(all_points)
            
            # Create cell connectivity array
            # Each hexahedron uses 8 points, and we need to map from block index to point indices
            num_points_per_block = 8
            point_indices_base = np.arange(num_blocks * num_points_per_block, dtype=np.int64)
            point_indices = point_indices_base.reshape(num_blocks, num_points_per_block)
            
            # Create cells array: [8, p0, p1, p2, p3, p4, p5, p6, p7] for each cell
            # Prepend 8 to each row and flatten
            cells = np.column_stack([
                np.full(num_blocks, 8, dtype=np.int64),  # Cell size
                point_indices
            ]).flatten()
            
            # All cells are hexahedrons (VTK type 12)
            cell_types = np.full(num_blocks, 12, dtype=np.uint8)
            
            # Store property values for each cell (already in correct format)
            cell_data = {}
            for prop_name, prop_values in filtered_properties.items():
                # Ensure values are numpy array and have correct length
                if isinstance(prop_values, (list, tuple)):
                    prop_values = np.array(prop_values)
                if len(prop_values) == num_blocks:
                    cell_data[prop_name] = prop_values
                else:
                    logger.warning(f"Property '{prop_name}' has {len(prop_values)} values, expected {num_blocks}")
            
            # Create the UnstructuredGrid
            grid = pv.UnstructuredGrid(cells, cell_types, all_points)
            
            # Add properties as cell data
            for prop_name, values in cell_data.items():
                if len(values) == len(positions):  # Ensure correct number of values
                    grid.cell_data[prop_name] = values
                    logger.info(f"Added cell data '{prop_name}' with {len(values)} values")
                else:
                    logger.warning(f"Skipping property '{prop_name}': expected {len(positions)} values, got {len(values)}")
            
            self.block_meshes['unstructured_grid'] = grid
            elapsed = time.time() - start_time
            logger.info(f"Created UnstructuredGrid with {len(positions)} blocks and {len(grid.cell_data)} properties in {elapsed:.2f}s")
    
    def _add_meshes_to_plotter(self) -> None:
        """Add all block meshes to the plotter."""
        import time
        add_start = time.time()
        if self.plotter is None:
            logger.warning("Cannot add meshes: plotter is not initialized")
            return
        
        if not self.block_meshes:
            logger.warning("Cannot add meshes: no block meshes available")
            return
        
        # CRITICAL FIX: Remove only block model actor instead of clearing all actors
        # plotter.clear() removes drillholes, geology, and all other layers!
        # We only want to remove the previous block model mesh.
        clear_start = time.time()
        
        # Remove existing block model actor only
        if self.mesh_actor is not None:
            try:
                self.plotter.remove_actor(self.mesh_actor)
                logger.debug("Removed previous block model actor")
            except Exception as e:
                logger.debug(f"Could not remove previous mesh_actor: {e}")
            self.mesh_actor = None
        
        # Also remove from layer tracking
        if "blocks" in self.active_layers:
            try:
                self.clear_layer("blocks")
            except Exception:
                pass
        
        self._current_grid_signature = None
        clear_time = time.time() - clear_start
        logger.info(f"PERF: block model clear took {clear_time:.3f}s")
        
        # Remove any existing PyVista scalar bars (disabled - use custom LegendManager)
        try:
            self.plotter.remove_scalar_bar()
        except Exception:
            pass
        
        setup_start = time.time()
        self._setup_plotter()
        setup_time = time.time() - setup_start
        logger.info(f"PERF: _setup_plotter() took {setup_time:.3f}s")
        
        # Check if we have ImageData or UnstructuredGrid
        # Support both grid types with priority to ImageData
        grid = None
        is_imagedata = False
        
        if 'imagedata' in self.block_meshes:
            grid = self.block_meshes['imagedata']
            is_imagedata = True
        elif 'unstructured_grid' in self.block_meshes:
            grid = self.block_meshes['unstructured_grid']
            is_imagedata = self.block_meshes.get('_is_imagedata', False)
        
        if grid is not None:
            # Apply visibility filtering at cell level if requested
            if self.visible_blocks is not None:
                try:
                    if is_imagedata and isinstance(grid, pv.ImageData):
                        # For ImageData, extract_cells converts to UnstructuredGrid
                        # This is acceptable for visibility filtering, but note the conversion
                        original_to_keep = set(map(int, self.visible_blocks.tolist()))
                        cell_mask = np.isin(self._filtered_indices, list(original_to_keep))
                        ids = np.nonzero(cell_mask)[0].astype(np.int64)
                        if ids.size > 0:
                            grid = grid.extract_cells(ids)
                            # Note: extract_cells on ImageData returns UnstructuredGrid
                            logger.debug(f"Applied visibility filtering to ImageData grid ({ids.size} cells visible)")
                    else:
                        # Standard UnstructuredGrid filtering
                        original_to_keep = set(map(int, self.visible_blocks.tolist()))
                        cell_mask = np.isin(self._filtered_indices, list(original_to_keep))
                        ids = np.nonzero(cell_mask)[0].astype(np.int64)
                        if ids.size > 0:
                            grid = grid.extract_cells(ids)
                except Exception as e:
                    logger.warning(f"Failed to apply cell-level visibility: {e}")
            
            # MINING BLOCK MODEL VISUALIZATION (Leapfrog/Datamine style)
            # Requirements:
            # 1. FLAT shading (no smooth gradients between blocks)
            # 2. NO interpolation (each block has uniform color)
            # 3. Cell-based coloring (not point-based)
            # 4. Visible edges on ALL blocks (even same-colored neighbors)
            # 5. Disable lighting to prevent soft gradients
            
        # Determine which property to use for initial coloring
        # This prevents double render - mesh will have correct color from the start
        initial_property = None
        initial_colormap = self.current_colormap or 'viridis'
        if self.current_model and self.current_model.properties:
            if self.current_property and self.current_property in grid.cell_data:
                initial_property = self.current_property
            else:
                prop_names = list(self.current_model.properties.keys())
                if prop_names and prop_names[0] in grid.cell_data:
                    initial_property = prop_names[0]

            if initial_property:
                logger.info(f"Adding mesh with initial property coloring: {initial_property}")

            # CRITICAL PERFORMANCE FIX: Disable edges for UnstructuredGrid
            # UnstructuredGrid edge extraction is extremely expensive (50-100x slower than ImageData)
            # For regular block models, use ImageData instead of UnstructuredGrid
            show_edges_enabled = True  # Default for ImageData/RectilinearGrid
            if isinstance(grid, pv.UnstructuredGrid):
                show_edges_enabled = False  # Force off for UnstructuredGrid
                logger.warning(f"[MESH TYPE] UnstructuredGrid detected ({grid.n_cells} cells) - disabling edges for performance")
            else:
                logger.info(f"[MESH TYPE] Using {type(grid).__name__} ({grid.n_cells} cells) - edges enabled")

            try:
                add_mesh_start = time.time()
                self.mesh_actor = self.plotter.add_mesh(
                    grid,
                    scalars=initial_property if initial_property else None,  # Add scalars from start to prevent double render
                    cmap=initial_colormap if initial_property else None,
                    show_edges=show_edges_enabled,  # Conditional: False for UnstructuredGrid
                    edge_color=self.edge_color,
                    line_width=self.edge_width,
                    opacity=self.current_opacity,
                    lighting=True,  # Enable lighting for better color balance and depth
                    smooth_shading=False,  # CRITICAL: Flat shading (no Gouraud/Phon)
                    interpolate_before_map=False,  # CRITICAL: No color interpolation
                    preference='cell',  # Use cell data (not point data)
                    style='surface',  # Render as solid surfaces
                    ambient=0.6,  # Reduced ambient for better color balance (was 1.0 - too bright)
                    diffuse=0.4,  # Add diffuse for depth perception (was 0.0)
                    specular=0.1,  # Small specular for subtle highlights (was 0.0)
                    pbr=False,  # Disable physically-based rendering
                    show_scalar_bar=False,  # DISABLED: Use custom LegendManager instead
                    name='block_model',
                    pickable=True  # CRITICAL: Enable picking for hover/click interaction
                )
                add_mesh_time = time.time() - add_mesh_start
                logger.info(f"PERF: plotter.add_mesh() took {add_mesh_time:.3f}s")
                
                # CRITICAL: Remove any axes that might have been auto-created by add_mesh
                # PyVista sometimes enables cube axes automatically when adding meshes
                self._remove_all_axes_actors()
                
                # CRITICAL: Disable axes again after add_mesh (it might re-enable them)
                try:
                    self.plotter.show_axes = False
                    if hasattr(self.plotter, 'hide_axes'):
                        self.plotter.hide_axes()
                    # show_grid may be a property or method depending on PyVista version
                    try:
                        self.plotter.show_grid = False
                    except (TypeError, AttributeError):
                        try:
                            self.plotter.show_grid(False)
                        except TypeError:
                            # Newer PyVista: show_grid() takes no arguments
                            self.plotter.show_grid()
                    self.plotter._show_bounds = False
                except Exception:
                    pass
                logger.info(f"Added mesh to plotter, mesh_actor = {self.mesh_actor}")
                self._current_grid_signature = self._compute_grid_signature(grid)
                
                # Remove any auto-created scalar bars immediately after adding mesh
                try:
                    self.plotter.remove_scalar_bar()
                except Exception:
                    pass
                
                # Note: Mesh is now added with scalars from the start (above), so no need for
                # additional property coloring code here. This prevents double render.
            except Exception as e:
                logger.error(f"Error adding mesh to plotter: {e}", exc_info=True)
                self.mesh_actor = None
                raise
            
            # CRITICAL: Force the mapper to use flat shading at the VTK level
            vtk_setup_start = time.time()
            vtk_setup_time = 0.0
            if self.mesh_actor:
                # CRITICAL: Ensure actor is pickable for hover/click interaction
                self.mesh_actor.SetPickable(1)
                logger.debug(f"Set mesh_actor pickable: {self.mesh_actor.GetPickable()}")
                
                mapper = self.mesh_actor.GetMapper()
                if mapper:
                    # Ensure flat interpolation (VTK_FLAT = 0)
                    prop = self.mesh_actor.GetProperty()
                    prop.SetInterpolationToFlat()  # Force flat shading
                    prop.SetAmbient(0.6)  # Reduced ambient for better color balance (was 1.0 - too bright)
                    prop.SetDiffuse(0.4)  # Add diffuse for depth perception (was 0.0)
                    prop.SetSpecular(0.1)  # Small specular for subtle highlights (was 0.0)
                    prop.SetSpecularPower(15)  # Control specular sharpness
                    
                    # Enable polygon offset to prevent Z-fighting between edges and faces
                    mapper.SetResolveCoincidentTopologyToPolygonOffset()
                    mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1, 1)
                    
                    vtk_setup_time = time.time() - vtk_setup_start
                    logger.info(f"PERF: VTK property/mapper setup took {vtk_setup_time:.3f}s")
                    logger.info("Applied flat shading with balanced lighting at VTK property level")
            
            add_total_time = time.time() - add_start
            logger.info(f"PERF: _add_meshes_to_plotter total: {add_total_time:.3f}s (clear: {clear_time:.3f}s, setup: {setup_time:.3f}s, add_mesh: {add_mesh_time:.3f}s, vtk: {vtk_setup_time:.3f}s)")
            logger.info(f"Added block model with edges ({self.edge_color}, width {self.edge_width})")
            
        elif 'point_cloud' in self.block_meshes:
            mesh = self.block_meshes['point_cloud']
            
            # Apply visibility filtering if needed
            if self.visible_blocks is not None and len(self.visible_blocks) < self.current_model.block_count:
                # Filter the point cloud
                filtered_positions = self.current_model.positions[self.visible_blocks]
                filtered_mesh = pv.PolyData(filtered_positions)
                
                # Add filtered properties
                for prop_name, prop_values in self.current_model.properties.items():
                    filtered_mesh[prop_name] = prop_values[self.visible_blocks]
                
                mesh = filtered_mesh
            
            # Add as point cloud with proper styling
            self.plotter.add_mesh(mesh, point_size=20, render_points_as_spheres=True, 
                                style='points', color='red', opacity=0.8)
            logger.info("Added optimized point cloud to plotter")
        else:
            # Add visible blocks (original method for small models)
            if self.visible_blocks is not None:
                for i in self.visible_blocks:
                    mesh_key = f'block_{i}'
                    if mesh_key in self.block_meshes:
                        mesh = self.block_meshes[mesh_key]
                        
                        # Add scalar data if available
                        if self.current_model and self.current_model.properties:
                            # Use first property for coloring
                            prop_name = list(self.current_model.properties.keys())[0]
                            prop_values = self.current_model.get_property(prop_name)
                            if prop_values is not None:
                                mesh[f'{prop_name}'] = prop_values[i]
                        
                        self.plotter.add_mesh(mesh, show_edges=True, edge_color=self.edge_color, 
                                            line_width=self.edge_width)
    
    def update_visibility(self, visible_indices: np.ndarray) -> None:
        """
        Update which blocks are visible.
        
        Args:
            visible_indices: Array of block indices to show
        """
        self.visible_blocks = visible_indices
        self._add_meshes_to_plotter()
        logger.info(f"Updated visibility: {len(visible_indices)} blocks visible")
    
    def set_property_coloring(self, property_name: str, colormap: str = 'viridis', discrete: bool = False) -> None:
        """
        Color blocks by a property.
        
        CRITICAL: NSR fields are automatically detected and use divergent colormap
        (red-white-blue) with zero-centered normalization for proper visualization.
        
        Args:
            property_name: Name of property to use for coloring
            colormap: PyVista colormap name (ignored for NSR fields)
            discrete: Whether to use discrete coloring
        """
        if self.plotter is None or self.current_model is None:
            return
        
        prop_values = self.current_model.get_property(property_name)
        if prop_values is None:
            logger.warning(f"Property '{property_name}' not found")
            return
        
        # CRITICAL: Detect NSR fields and use ColorMapper for consistent coloring
        is_nsr = property_name.upper() in ['NSR', 'NSR_TOTAL', 'NET_SMELTER_RETURN', 'NSR_TOTAL']
        center_zero = False
        nsr_cmap = None
        
        if is_nsr:
            # Use NSR-specific divergent colormap via ColorMapper
            from .color_mapper import ColorMapper
            color_mapper = ColorMapper()
            nsr_cmap = color_mapper.get_nsr_map()
            
            # Calculate NSR range
            nsr_min = float(np.nanmin(prop_values))
            nsr_max = float(np.nanmax(prop_values))
            
            # Map NSR to colors with zero-centered normalization
            colors, color_metadata = color_mapper.map_property_to_colors(
                values=prop_values,
                property_name=property_name,
                colormap=nsr_cmap,
                vmin=nsr_min,
                vmax=nsr_max,
                center_zero=True  # Zero-centered for divergent map
            )
            
            # Use the NSR colormap name for VTK
            colormap = 'nsr_diverging'
            center_zero = True
            
            logger.info(f"Detected NSR field '{property_name}', using divergent colormap (range: ${nsr_min:.2f} to ${nsr_max:.2f}/t)")
        
        # OPTIMIZATION: Try to update existing mesh actor instead of clearing/re-adding
        # Only re-add if grid signature differs (new mesh) or actor is missing
        if self.mesh_actor is not None:
            mapper = self.mesh_actor.GetMapper()
            actor_dataset = None
            actor_signature = None
            if mapper:
                dataset_getter = getattr(mapper, 'GetInputAsDataSet', None)
                try:
                    if callable(dataset_getter):
                        actor_dataset = dataset_getter()
                    else:
                        actor_dataset = mapper.GetInput()
                except Exception:
                    actor_dataset = None

                actor_signature = self._compute_grid_signature(actor_dataset)

            expected_signature = self._current_grid_signature
            same_grid = expected_signature is None or actor_signature == expected_signature

            if same_grid and mapper:
                try:
                    # Update scalar array
                    mapper.SelectColorArray(property_name)
                    mapper.SetScalarModeToUseCellData()
                    mapper.SetScalarVisibility(1)

                    # Update colormap via lookup table
                    import pyvista as pv
                    lut = mapper.GetLookupTable()
                    if lut is None:
                        # Create new lookup table if needed
                        lut = pv._vtk.vtkLookupTable()
                        mapper.SetLookupTable(lut)

                    # Get scalar range from the property
                    # For NSR with zero-centered normalization, use symmetric range
                    if center_zero and is_nsr:
                        nsr_min = float(np.nanmin(prop_values))
                        nsr_max = float(np.nanmax(prop_values))
                        max_abs = max(abs(nsr_min), abs(nsr_max))
                        scalar_range = (-max_abs, max_abs)
                    else:
                        scalar_range = (float(np.nanmin(prop_values)), float(np.nanmax(prop_values)))
                    
                    lut.SetRange(scalar_range)
                    lut.SetTableRange(scalar_range)

                    # Apply colormap using matplotlib directly
                    try:
                        import matplotlib.cm as cm
                        # For NSR, use the NSR colormap object directly
                        if is_nsr and nsr_cmap is not None:
                            cmap_matplotlib = nsr_cmap
                        else:
                            cmap_matplotlib = cm.get_cmap(colormap)
                        
                        lut.SetNumberOfTableValues(256)
                        lut.Build()
                        for i in range(256):
                            rgba = cmap_matplotlib(i / 255.0)
                            lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], rgba[3] if len(rgba) > 3 else 1.0)
                    except Exception as e:
                        logger.debug(f"Could not apply colormap '{colormap}' to LUT, using default viridis: {e}")
                        # Fallback: use default viridis
                        try:
                            import matplotlib.cm as cm
                            cmap_matplotlib = cm.get_cmap('viridis')
                            lut.SetNumberOfTableValues(256)
                            lut.Build()
                            for i in range(256):
                                rgba = cmap_matplotlib(i / 255.0)
                                lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], rgba[3] if len(rgba) > 3 else 1.0)
                        except Exception:
                            pass

                    # Force mapper and actor update
                    mapper.Modified()
                    self.mesh_actor.Modified()

                    # Trigger render update (PyVista will handle this automatically, but ensure it happens)
                    if hasattr(self.plotter, 'render'):
                        try:
                            self.plotter.render()
                        except Exception:
                            pass

                    logger.info(f"Updated existing mesh actor with property '{property_name}' and colormap '{colormap}' (no clear/re-add)")

                    # Update custom legend widget via LegendManager
                    if hasattr(self, 'legend_manager') and self.legend_manager is not None:
                        try:
                            if discrete:
                                unique_values = np.unique(prop_values[~np.isnan(prop_values)])
                                unique_values = unique_values[unique_values != 0]
                                unique_values = sorted(unique_values)  # Returns a list
                                if len(unique_values) <= 100:
                                    self.legend_manager.update_discrete(
                                        property_name=property_name,
                                        categories=list(unique_values),  # Ensure it's a list (sorted already returns list)
                                        cmap_name=colormap,
                                    )
                            else:
                                # Pass the full property array so the legend manager can
                                # derive vmin/vmax consistently with other code paths.
                                prop_array = np.asarray(prop_values)
                                
                                # For NSR, pass center_zero flag for divergent map display
                                if is_nsr:
                                    # Calculate symmetric range for legend
                                    nsr_min = float(np.nanmin(prop_values))
                                    nsr_max = float(np.nanmax(prop_values))
                                    max_abs = max(abs(nsr_min), abs(nsr_max))
                                    
                                    self.legend_manager.update_continuous(
                                        property_name=property_name,
                                        data=prop_array,
                                        cmap_name='nsr_diverging',
                                        vmin=-max_abs,
                                        vmax=max_abs,
                                        center_zero=True
                                    )
                                else:
                                    self.legend_manager.update_continuous(
                                        property_name=property_name,
                                        data=prop_array,
                                        cmap_name=colormap,
                                    )
                        except Exception as e:
                            logger.debug(f"Could not update legend manager: {e}")

                    # Update layer metadata (find the current block model layer)
                    current_layer_name = self._get_block_model_layer_name(self.current_model)
                    if current_layer_name in self.active_layers:
                        self.active_layers[current_layer_name]['current_property'] = property_name
                        self.active_layers[current_layer_name]['current_colormap'] = colormap
                    else:
                        # Fallback: find any layer that starts with "Block Model"
                        for layer_name in self.active_layers:
                            if layer_name.startswith("Block Model"):
                                self.active_layers[layer_name]['current_property'] = property_name
                                self.active_layers[layer_name]['current_colormap'] = colormap
                                break

                    # Store current property/colormap for future reference
                    self.current_property = property_name
                    self.current_colormap = colormap

                    logger.info(f"Updated layer '{current_layer_name}' to show property '{property_name}' with colormap '{colormap}'")
                    return  # Successfully updated, no need to clear/re-add
                except Exception as e:
                    logger.warning(f"Could not update existing mesh actor, falling back to clear/re-add: {e}")
            else:
                logger.info("Mesh actor grid signature mismatch; rebuilding mesh for property update")
        
        # Fallback: Remove and re-add block model meshes with coloring (only if update failed)
        # CRITICAL: Do NOT use plotter.clear() as it removes drillholes and geology!
        if self.mesh_actor is not None:
            try:
                self.plotter.remove_actor(self.mesh_actor)
            except Exception:
                pass
            self.mesh_actor = None
        # Remove any existing PyVista scalar bars (disabled - use custom LegendManager)
        try:
            self.plotter.remove_scalar_bar()
        except (AttributeError, RuntimeError, StopIteration) as e:
            logger.debug(f"Expected: scalar bar removal failed: {e}")
        self._setup_plotter()
        self._current_grid_signature = None
        
        # Handle discrete coloring for categorical data
        if discrete:
            # Get unique values and create discrete colormap
            unique_values = np.unique(prop_values[~np.isnan(prop_values)])
            unique_values = unique_values[unique_values != 0]  # Exclude 0 (often means "not assigned")
            unique_values = sorted(unique_values)
            n_categories = len(unique_values)
            
            # Limit discrete mode to reasonable number of categories
            if n_categories > 100:
                logger.error(f"Discrete mode with {n_categories} categories - TOO MANY! Limiting to 100 categories. Use continuous mode instead.")
                # Take only first 100 categories
                unique_values = unique_values[:100]
                n_categories = 100
            elif n_categories > 20:
                logger.warning(f"Discrete mode with {n_categories} categories - consider using continuous mode for better performance")
            
            # For discrete mode, use categorical colormap name directly
            # PyVista will handle the discrete rendering
            import matplotlib.pyplot as plt
            from matplotlib import cm
            
            # If not already using a categorical colormap, suggest one
            categorical_maps = ['tab10', 'tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2', 'Pastel1', 'Pastel2']
            if colormap not in categorical_maps:
                # Use a good default categorical colormap
                if n_categories <= 10:
                    colormap = 'tab10'
                elif n_categories <= 20:
                    colormap = 'tab20'
                else:
                    colormap = 'tab20'  # Will repeat colors
                logger.info(f"Switched to categorical colormap '{colormap}' for discrete mode")
            
            # DISABLED: PyVista scalar bar arguments - use custom LegendManager instead
            logger.info(f"Using discrete coloring with {n_categories} categories: {unique_values[:10]}{'...' if n_categories > 10 else ''}")
        else:
            # DISABLED: PyVista scalar bar - use custom LegendManager instead
            pass
        
        # Check if we have ImageData or UnstructuredGrid
        grid = None
        if 'imagedata' in self.block_meshes:
            grid = self.block_meshes['imagedata'].copy()
        elif 'unstructured_grid' in self.block_meshes:
            grid = self.block_meshes['unstructured_grid'].copy()
        
        if grid is not None:
            # Ensure the property exists in the grid's cell data
            if property_name in grid.cell_data:
                # Add as colored UnstructuredGrid with professional rendering
                # Enforced flat, cell-based shading with balanced lighting for better color perception
                self.mesh_actor = self.plotter.add_mesh(
                    grid,
                    scalars=property_name,
                    cmap=colormap,
                    show_edges=False,  # Wireframe layer below forces all edges for blocky look
                    edge_color=self.edge_color,
                    line_width=self.edge_width,
                    opacity=self.current_opacity,
                    lighting=True,  # Enable lighting for better color balance
                    smooth_shading=False,  # Flat shading for distinct block faces
                    interpolate_before_map=False,  # Prevent colour interpolation between cells
                    preference='cell',  # Colour by cell data to keep each block uniform
                    ambient=0.6,  # Reduced ambient for better color balance (was 1.0 - too bright)
                    diffuse=0.4,  # Add diffuse for depth perception (was 0.0)
                    specular=0.1,  # Small specular for subtle highlights (was 0.0)
                    show_scalar_bar=False  # DISABLED: Use custom LegendManager instead
                )
                
                # Force flat shading at the VTK property level as a safety net
                if self.mesh_actor:
                    mapper = self.mesh_actor.GetMapper()
                    if mapper:
                        prop = self.mesh_actor.GetProperty()
                        if prop is not None:
                            prop.SetInterpolationToFlat()
                            prop.SetAmbient(0.6)  # Reduced ambient for better color balance (was 1.0 - too bright)
                            prop.SetDiffuse(0.4)  # Add diffuse for depth perception (was 0.0)
                            prop.SetSpecular(0.1)  # Small specular for subtle highlights (was 0.0)
                            prop.SetSpecularPower(15)  # Control specular sharpness
                        
                        mapper.SetResolveCoincidentTopologyToPolygonOffset()
                        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1, 1)
                
                # Add dedicated wireframe overlay so block boundaries are always visible
                self.plotter.add_mesh(
                    grid,
                    style='wireframe',
                    color=self.edge_color,
                    line_width=self.edge_width,
                    opacity=1.0,
                    render_lines_as_tubes=False,
                    name='block_wireframe_colored'
                )
                grid_type = "ImageData" if isinstance(grid, pv.ImageData) else "UnstructuredGrid"
                logger.info(f"Applied property coloring to {grid_type}: {property_name}")
                
                # Update custom legend widget via LegendManager
                if hasattr(self, 'legend_manager') and self.legend_manager is not None:
                    try:
                        if discrete:
                            # Update discrete legend
                            unique_values = np.unique(prop_values[~np.isnan(prop_values)])
                            unique_values = unique_values[unique_values != 0]
                            unique_values = sorted(unique_values)  # Returns a list
                            if len(unique_values) <= 100:  # Only if reasonable number of categories
                                self.legend_manager.update_discrete(
                                    property_name=property_name,
                                    categories=list(unique_values),  # Ensure it's a list
                                    cmap_name=colormap
                                )
                        else:
                            # Update continuous legend
                            prop_array = np.asarray(prop_values)
                            finite_data = prop_array[np.isfinite(prop_array)]
                            if len(finite_data) > 0:
                                self.legend_manager.update_continuous(
                                    property_name=property_name,
                                    data=prop_array,
                                    cmap_name=colormap
                                )
                    except Exception as e:
                        logger.warning(f"Failed to update legend: {e}", exc_info=True)
            else:
                logger.warning(f"Property '{property_name}' not found in grid cell data")
                # Add without coloring
                self.mesh_actor = self.plotter.add_mesh(
                    grid, 
                    show_edges=False, 
                    edge_color=self.edge_color,
                    line_width=self.edge_width,
                    opacity=self.current_opacity,
                    lighting=True,  # Enable lighting for better color balance
                    smooth_shading=False,
                    interpolate_before_map=False,
                    preference='cell',
                    ambient=0.6,  # Reduced ambient for better color balance (was 1.0 - too bright)
                    diffuse=0.4,  # Add diffuse for depth perception (was 0.0)
                    specular=0.1  # Small specular for subtle highlights (was 0.0)
                )
                
                if self.mesh_actor:
                    mapper = self.mesh_actor.GetMapper()
                    if mapper:
                        prop = self.mesh_actor.GetProperty()
                        if prop is not None:
                            prop.SetInterpolationToFlat()
                            prop.SetAmbient(0.6)  # Reduced ambient for better color balance (was 1.0 - too bright)
                            prop.SetDiffuse(0.4)  # Add diffuse for depth perception (was 0.0)
                            prop.SetSpecular(0.1)  # Small specular for subtle highlights (was 0.0)
                            prop.SetSpecularPower(15)  # Control specular sharpness

                        mapper.SetResolveCoincidentTopologyToPolygonOffset()
                        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1, 1)

                self._current_grid_signature = self._compute_grid_signature(grid)
                self.plotter.add_mesh(
                    grid,
                    style='wireframe',
                    color=self.edge_color,
                    line_width=self.edge_width,
                    opacity=1.0,
                    render_lines_as_tubes=False,
                    name='block_wireframe_default'
                )
            
        elif 'point_cloud' in self.block_meshes:
            mesh = self.block_meshes['point_cloud'].copy()
            
            # Apply visibility filtering if needed
            if self.visible_blocks is not None and len(self.visible_blocks) < self.current_model.block_count:
                # Filter the point cloud
                filtered_positions = self.current_model.positions[self.visible_blocks]
                filtered_mesh = pv.PolyData(filtered_positions)
                
                # Add filtered properties
                for prop_name, prop_values in self.current_model.properties.items():
                    filtered_mesh[prop_name] = prop_values[self.visible_blocks]
                
                mesh = filtered_mesh
            
            # Add as colored point cloud (scalar bar disabled - use custom LegendManager)
            self.plotter.add_mesh(mesh, scalars=property_name, point_size=20, 
                                render_points_as_spheres=True, style='points', 
                                cmap=colormap, opacity=0.8, show_scalar_bar=False)  # DISABLED: Use custom LegendManager instead
            logger.info(f"Applied property coloring to point cloud: {property_name}")
        else:
            # Original block-by-block coloring for small models
            if self.visible_blocks is not None:
                for i in self.visible_blocks:
                    mesh_key = f'block_{i}'
                    if mesh_key in self.block_meshes:
                        mesh = self.block_meshes[mesh_key].copy()
                        mesh[property_name] = prop_values[i]
                        
                        self.plotter.add_mesh(mesh, scalars=property_name, show_edges=True, 
                                            edge_color=self.edge_color, line_width=self.edge_width,
                                            cmap=colormap, show_scalar_bar=False)  # DISABLED: Use custom LegendManager instead
        
        logger.info(f"Applied property coloring: {property_name}")
    
    def reset_camera(self) -> None:
        """
        Reset camera with proper bounds handling for all model types.
        
        Phase 3.1 Fix: Handles thin/vertical models, viewport aspect ratio,
        and projection mode properly. No more hardcoded multipliers.
        """
        if self.plotter is None:
            return
        
        # Get unified scene bounds from all layers
        bounds = self._get_scene_bounds()
        if bounds is None:
            self.plotter.reset_camera()
            logger.info("Reset camera view (default)")
            return
        
        # Calculate center and dimensions
        center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ])
        
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        
        # Prevent division by zero
        dx = max(dx, 1e-6)
        dy = max(dy, 1e-6)
        dz = max(dz, 1e-6)
        
        # Get viewport aspect ratio
        try:
            w, h = self.plotter.window_size
            viewport_aspect = w / max(1, h)
        except Exception:
            viewport_aspect = 1.5  # Default 3:2 aspect
        
        # Calculate model extents
        max_extent = max(dx, dy, dz)
        min_extent = min(dx, dy, dz)
        horizontal_extent = max(dx, dy)
        
        # Detect thin/vertical models (veins, narrow zones, etc.)
        aspect_ratio = max_extent / max(min_extent, 1e-6)
        is_thin_model = aspect_ratio > 10  # Very elongated
        
        if is_thin_model:
            # Use orthographic projection for thin models
            # Distance based on largest dimension
            distance = max_extent * 1.5
            logger.debug(f"Thin model detected (aspect={aspect_ratio:.1f}), using adjusted camera")
        else:
            # Standard perspective projection
            # Use FOV-based distance calculation
            fov_rad = np.radians(30)  # PyVista default FOV ~30 degrees
            # Distance needed to fit model in view
            fit_distance = (max_extent / 2) / np.tan(fov_rad / 2)
            # Add 20% padding
            distance = fit_distance * 1.2
            
            # Adjust for viewport aspect ratio
            if horizontal_extent > dz:
                # Horizontal model - might need more distance for narrow viewports
                if viewport_aspect < 1.0:
                    distance *= 1.0 / viewport_aspect
        
        # Position camera for isometric-ish view (45° azimuth, 35° elevation)
        # This provides good visibility for most geological models
        azimuth = np.radians(45)
        elevation = np.radians(35)
        
        cam_x = center[0] + distance * np.cos(elevation) * np.cos(azimuth)
        cam_y = center[1] + distance * np.cos(elevation) * np.sin(azimuth)
        cam_z = center[2] + distance * np.sin(elevation)
        
        position = [cam_x, cam_y, cam_z]
        
        self.plotter.camera_position = [position, center.tolist(), [0, 0, 1]]
        
        # DIAGNOSTIC: Log camera position and focal point
        logger.info(f"[CAMERA DEBUG] Position: ({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})")
        logger.info(f"[CAMERA DEBUG] Focal point (center): ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
        logger.info(f"[CAMERA DEBUG] Distance: {distance:.1f}")
        
        # Set clipping range based on scene size
        try:
            camera = self.plotter.renderer.GetActiveCamera()
            if camera:
                # Near plane: small enough for close inspection but not too small (z-fighting)
                near = max(1.0, distance * 0.001)
                # Far plane: large enough to see entire scene
                far = distance * 20.0 + max_extent * 5.0
                camera.SetClippingRange(near, far)
                logger.debug(f"Set camera clipping range: near={near:.2f}, far={far:.1f}")
        except Exception as e:
            logger.debug(f"Could not set clipping range: {e}")
        
        logger.info(f"Reset camera view (bounds: {dx:.0f}x{dy:.0f}x{dz:.0f}, distance: {distance:.0f})")
        
        # DIAGNOSTIC: Check actors after camera reset
        try:
            n_actors = len(self.plotter.renderer.actors)
            logger.info(f"[CAMERA DEBUG] Renderer has {n_actors} actors after reset_camera")
            
            # Log actor names/types for debugging
            for i, (name, actor) in enumerate(self.plotter.renderer.actors.items()):
                try:
                    visible = actor.GetVisibility() if hasattr(actor, 'GetVisibility') else 'unknown'
                    bounds = actor.GetBounds() if hasattr(actor, 'GetBounds') else None
                    logger.info(f"[CAMERA DEBUG]   Actor {i}: name='{name}', visible={visible}, bounds={bounds}")
                except Exception:
                    logger.info(f"[CAMERA DEBUG]   Actor {i}: name='{name}' (could not get details)")
            
            # Force render
            self.plotter.render()
            logger.info("[CAMERA DEBUG] Forced render after camera reset")
        except Exception as e:
            logger.debug(f"Camera reset diagnostics failed: {e}")
    
    def _get_scene_bounds(self) -> Optional[List[float]]:
        """
        Get unified bounds from all visible scene layers.

        Returns:
            [xmin, xmax, ymin, ymax, zmin, zmax] or None if no layers
        """
        all_bounds = []

        # CRITICAL FIX: Get bounds from transformed mesh instead of original model bounds
        # This prevents camera positioning issues when block models are in different coordinate systems
        mesh_bounds_added = False
        if self.block_meshes:
            for mesh_key, mesh in self.block_meshes.items():
                if mesh_key.startswith('_') or mesh is None:
                    continue
                if hasattr(mesh, 'bounds'):
                    bounds = mesh.bounds
                    if bounds is not None and len(bounds) >= 6:
                        if not all(b == 0 for b in bounds):
                            all_bounds.append(bounds)
                            mesh_bounds_added = True
                            logger.debug(f"[SCENE_BOUNDS] Using transformed mesh bounds from '{mesh_key}'")
                            break  # One mesh is enough

        # Fallback to model bounds if no mesh bounds available (only if mesh bounds not added)
        if not mesh_bounds_added and self.current_model is not None and hasattr(self.current_model, 'bounds'):
            bounds = self.current_model.bounds
            if bounds is not None:
                # Transform to local coordinates if global shift is set
                if self._global_shift is not None:
                    shift = self._global_shift
                    bounds = (
                        bounds[0] - shift[0], bounds[1] - shift[0],
                        bounds[2] - shift[1], bounds[3] - shift[1],
                        bounds[4] - shift[2], bounds[5] - shift[2],
                    )
                    logger.debug(f"[SCENE_BOUNDS] Using transformed model bounds (fallback)")
                all_bounds.append(bounds)
        
        # Get bounds from active layers
        for layer_name, layer_data in self.active_layers.items():
            actor = layer_data.get("actor")
            if actor is not None:
                try:
                    actor_bounds = actor.GetBounds()
                    if actor_bounds is not None and len(actor_bounds) >= 6:
                        # Filter out invalid bounds (all zeros or inf)
                        if not all(b == 0 for b in actor_bounds):
                            all_bounds.append(actor_bounds)
                except Exception:
                    pass
        
        # Get bounds from drillhole actors
        if hasattr(self, '_drillhole_hole_actors'):
            for hole_id, actor in self._drillhole_hole_actors.items():
                if actor is not None:
                    try:
                        bounds = actor.GetBounds()
                        if bounds is not None and len(bounds) >= 6:
                            if not all(b == 0 for b in bounds):
                                all_bounds.append(bounds)
                                break  # One drillhole is enough to set scale
                    except Exception:
                        pass
        
        if not all_bounds:
            return None
        
        # Compute union of all bounds
        xmin = min(b[0] for b in all_bounds)
        xmax = max(b[1] for b in all_bounds)
        ymin = min(b[2] for b in all_bounds)
        ymax = max(b[3] for b in all_bounds)
        zmin = min(b[4] for b in all_bounds)
        zmax = max(b[5] for b in all_bounds)
        
        return [xmin, xmax, ymin, ymax, zmin, zmax]
    
    def set_camera_position(self, position: Tuple[float, float, float], 
                           focal_point: Tuple[float, float, float],
                           view_up: Tuple[float, float, float]) -> None:
        """
        Set camera position and orientation.
        
        Args:
            position: Camera position (x, y, z)
            focal_point: Point camera is looking at
            view_up: Up vector for camera
        """
        if self.plotter is not None:
            self.plotter.camera_position = position
            self.plotter.camera.focal_point = focal_point
            self.plotter.camera.up = view_up
            logger.info(f"Set camera position: {position}")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get current camera information."""
        if self.plotter is None or self.plotter.camera is None:
            return {}
        
        try:
            # camera_position returns a tuple with position, focal_point, and view_up
            cam_pos = self.plotter.camera_position
            position = cam_pos[0] if isinstance(cam_pos, tuple) else self.plotter.camera.position
            focal_point = cam_pos[1] if isinstance(cam_pos, tuple) and len(cam_pos) > 1 else self.plotter.camera.focal_point
            
            return {
                'position': position,
                'focal_point': focal_point,
                'up': self.plotter.camera.up,
                'view_angle': self.plotter.camera.view_angle
            }
        except Exception as e:
            logger.warning(f"Error getting camera info: {e}")
            return {}
    
    def pan_camera(self, dx: float, dy: float) -> None:
        """
        Pan the camera by the specified amounts.
        
        Args:
            dx: Horizontal pan amount (positive = right, negative = left)
            dy: Vertical pan amount (positive = up, negative = down)
        """
        if self.plotter is None or self.plotter.camera is None:
            return
        
        try:
            camera = self.plotter.camera
            # Get current focal point
            focal_point = np.array(camera.focal_point)
            position = np.array(camera.position)
            
            # Calculate view direction and right vector
            view_dir = focal_point - position
            view_distance = np.linalg.norm(view_dir)
            if view_distance == 0:
                return
            
            view_dir = view_dir / view_distance
            up = np.array(camera.up)
            up = up / np.linalg.norm(up) if np.linalg.norm(up) > 0 else np.array([0, 0, 1])
            
            # Calculate right vector (perpendicular to view direction and up)
            right = np.cross(view_dir, up)
            right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([1, 0, 0])
            
            # Calculate actual up vector (perpendicular to view direction and right)
            actual_up = np.cross(right, view_dir)
            actual_up = actual_up / np.linalg.norm(actual_up) if np.linalg.norm(actual_up) > 0 else up
            
            # Pan amount based on view distance
            pan_scale = view_distance * 0.1  # Scale panning by view distance
            
            # Calculate pan vector
            pan_vector = (right * dx * pan_scale) + (actual_up * dy * pan_scale)
            
            # Update focal point and position
            new_focal_point = focal_point + pan_vector
            new_position = position + pan_vector
            
            camera.focal_point = new_focal_point
            camera.position = new_position
            
            logger.debug(f"Panned camera: dx={dx}, dy={dy}")
        except Exception as e:
            logger.warning(f"Error panning camera: {e}")
    
    def rotate_camera(self, azimuth: float, elevation: float) -> None:
        """
        Rotate the camera by the specified angles.
        
        Args:
            azimuth: Horizontal rotation in degrees (positive = rotate right)
            elevation: Vertical rotation in degrees (positive = rotate up)
        """
        if self.plotter is None or self.plotter.camera is None:
            return
        
        try:
            camera = self.plotter.camera
            # Get current camera state
            focal_point = np.array(camera.focal_point)
            position = np.array(camera.position)
            up = np.array(camera.up)
            
            # Calculate view direction
            view_dir = focal_point - position
            view_distance = np.linalg.norm(view_dir)
            if view_distance == 0:
                return
            
            # Normalize up vector
            up = up / np.linalg.norm(up) if np.linalg.norm(up) > 0 else np.array([0, 0, 1])
            
            # Convert angles to radians
            azimuth_rad = np.radians(azimuth)
            elevation_rad = np.radians(elevation)
            
            # Calculate rotation around up vector (azimuth)
            if abs(azimuth_rad) > 1e-6:
                # Create rotation matrix around up vector
                cos_a = np.cos(azimuth_rad)
                sin_a = np.sin(azimuth_rad)
                # Rodrigues' rotation formula
                view_dir = view_dir * cos_a + np.cross(up, view_dir) * sin_a + up * np.dot(up, view_dir) * (1 - cos_a)
            
            # Calculate rotation around right vector (elevation)
            if abs(elevation_rad) > 1e-6:
                right = np.cross(view_dir, up)
                right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([1, 0, 0])
                # Create rotation matrix around right vector
                cos_e = np.cos(elevation_rad)
                sin_e = np.sin(elevation_rad)
                # Rodrigues' rotation formula
                view_dir = view_dir * cos_e + np.cross(right, view_dir) * sin_e + right * np.dot(right, view_dir) * (1 - cos_e)
            
            # Normalize view direction
            view_dir = view_dir / np.linalg.norm(view_dir)
            
            # Update camera position (keep same distance from focal point)
            new_position = focal_point - view_dir * view_distance
            
            camera.position = new_position
            
            logger.debug(f"Rotated camera: azimuth={azimuth}°, elevation={elevation}°")
        except Exception as e:
            logger.warning(f"Error rotating camera: {e}")
    
    def set_transparency(self, alpha: float) -> None:
        """
        Set transparency for all blocks with instant visual update.
        
        Args:
            alpha: Transparency value (0.0 = transparent, 1.0 = opaque)
        """
        if self.plotter is None:
            return
        
        # Update all mesh actors
        for actor in self.plotter.renderer.actors.values():
            if hasattr(actor, 'GetProperty'):
                prop = actor.GetProperty()
                # Only update if it's a mesh actor (not axes, labels, etc.)
                if prop is not None:
                    prop.SetOpacity(alpha)
        
        # Defer render - PyVista will render on next frame automatically
        # This improves performance during interactive updates
        logger.info(f"Set transparency: {alpha}")
    
    def toggle_axes(self, show: bool) -> None:
        """Toggle axes visibility (uses overlay system, not PyVista axes)."""
        self.show_axes = show
        self.toggle_overlay("axes", show)
    
    def toggle_bounds(self, show: bool) -> None:
        """Toggle bounding box visibility (uses overlay system, not PyVista bounds)."""
        self.show_bounds = show
        self.toggle_overlay("bounds", show)
    
    def set_show_ground_grid(self, show: bool) -> None:
        """Toggle ground grid visibility (uses overlay system, not PyVista grid)."""
        self.show_grid = show
        self.toggle_overlay("ground_grid", show)
    
    def toggle_scalar_bar(self, show: bool) -> None:
        """
        Toggle scalar bar (color legend) visibility.
        
        Args:
            show: True to show, False to hide
        """
        try:
            if self.plotter is not None:
                # Try to access the scalar bar widget
                if hasattr(self.plotter.renderer, '_scalar_bar'):
                    scalar_bar = self.plotter.renderer._scalar_bar
                    if scalar_bar:
                        if show:
                            scalar_bar.SetVisibility(1)
                        else:
                            scalar_bar.SetVisibility(0)
                        # Defer render - PyVista will render on next frame automatically
                        # Note: We're not using PyVista scalar bars, but keeping for compatibility
                        logger.info(f"Toggled scalar bar: {show}")
                    else:
                        logger.warning("No scalar bar found to toggle")
                else:
                    logger.warning("Scalar bar not yet created - toggle will have no effect")
        except Exception as e:
            logger.error(f"Error toggling scalar bar: {e}", exc_info=True)
    
    def export_screenshot(self, filename: str, resolution: Tuple[int, int] = (1920, 1080)) -> None:
        """
        Export current view as screenshot.
        
        Args:
            filename: Output filename
            resolution: Image resolution (width, height)
        """
        if self.plotter is not None:
            self.plotter.screenshot(filename, window_size=resolution)
            logger.info(f"Exported screenshot: {filename}")
    
    def get_model_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Get bounds of the current model."""
        if self.current_model is None:
            return None
        return self.current_model.bounds
    
    def set_floating_axes_enabled(
        self,
        enabled: bool,
        x_major: Optional[float] = None,
        x_minor: Optional[float] = None,
        y_major: Optional[float] = None,
        y_minor: Optional[float] = None,
        z_major: Optional[float] = None,
        z_minor: Optional[float] = None,
        draw_box: bool = True,
        color: str = 'white',
        line_width: float = 2.0,
        font_size: float = 12.0,
        auto_spacing: bool = True,
    ) -> None:
        """
        Enable or disable the floating axes box with tick marks.
        
        Args:
            enabled: Whether to show the floating axes
            x_major, x_minor, y_major, y_minor, z_major, z_minor: Tick spacing for each axis
            draw_box: Whether to draw full bounding box or just the 3 axes
        """
        if self.plotter is None:
            return
        
        # Store parameters
        self._floating_axes_params = {
            'x_major': x_major,
            'x_minor': x_minor,
            'y_major': y_major,
            'y_minor': y_minor,
            'z_major': z_major,
            'z_minor': z_minor,
            'draw_box': draw_box,
            'color': color,
            'line_width': line_width,
            'font_size': font_size,
            'auto_spacing': auto_spacing,
        }
        
        if enabled:
            # Clear existing axes if they exist (in case we're updating parameters)
            if self.floating_axes is not None:
                self.floating_axes.clear(self.plotter)
                self.floating_axes = None
            # Also remove any VTK bounds actor
            try:
                self.plotter.remove_bounds_axes()
            except Exception:
                pass

            # Temporarily disable the flag to allow bounds computation
            # _get_scene_bounds() skips computed bounds when overlays are visible
            was_enabled = self.show_floating_axes
            self.show_floating_axes = False

            # Reset fixed bounds to force fresh computation from current data
            self._fixed_bounds = None

            # Get bounds while flag is temporarily disabled
            bounds = self._get_scene_bounds()

            if bounds is None:
                # Restore previous state if we can't get bounds
                self.show_floating_axes = was_enabled
                logger.warning("Cannot enable floating axes: no scene bounds available")
                return

            # Now set the flag to True after we have bounds
            self.show_floating_axes = True

            params = self._floating_axes_params or {}
            axis_color = params.get('color', 'white')
            font_size = params.get('font_size', 12)
            line_width_val = params.get('line_width', 2.0)

            # Compute world bounds for labels (add global_shift back to local bounds)
            # Compute world bounds for labels
            label_offset = self._global_shift if self._global_shift is not None else np.zeros(3)
            world_bounds = [
                bounds[0] + label_offset[0], bounds[1] + label_offset[0],
                bounds[2] + label_offset[1], bounds[3] + label_offset[1],
                bounds[4] + label_offset[2], bounds[5] + label_offset[2]
            ]

            # Calculate spans
            x_span = bounds[1] - bounds[0]
            y_span = bounds[3] - bounds[2]
            z_span = bounds[5] - bounds[4]

            # Generate world coordinate range strings for axis titles
            x_labels = self._generate_world_axis_labels(bounds[0], bounds[1], label_offset[0], 5)
            y_labels = self._generate_world_axis_labels(bounds[2], bounds[3], label_offset[1], 5)
            z_labels = self._generate_world_axis_labels(bounds[4], bounds[5], label_offset[2], 5)

            # Use PyVista's show_bounds for the nice grid design
            try:
                show_bounds_fn = self._original_show_bounds
                if show_bounds_fn is None:
                    raise RuntimeError("Original show_bounds method not available")

                show_bounds_fn(
                    bounds=bounds,
                    grid='back',
                    location='outer',
                    ticks='both',
                    n_xlabels=5,
                    n_ylabels=5,
                    n_zlabels=5,
                    xlabel=f'X ({x_labels})',
                    ylabel=f'Y ({y_labels})',
                    zlabel=f'Z ({z_labels})',
                    color=axis_color,
                    font_size=font_size,
                    minor_ticks=True,
                    padding=0.0,
                    use_2d=False,
                )

                # Try to modify tick labels to show world coordinates
                self._update_cube_axes_labels(label_offset)

                logger.info(f"Enabled floating axes with local bounds {bounds}, world range X=[{world_bounds[0]:.0f}, {world_bounds[1]:.0f}], Y=[{world_bounds[2]:.0f}, {world_bounds[3]:.0f}], Z=[{world_bounds[4]:.0f}, {world_bounds[5]:.0f}]")
            except Exception as e:
                logger.warning(f"Failed to show bounds: {e}")
        else:
            # Disable floating axes
            self.show_floating_axes = False
            if self.floating_axes is not None:
                self.floating_axes.clear(self.plotter)
                self.floating_axes = None
            # Also remove VTK bounds actor
            try:
                self.plotter.remove_bounds_axes()
            except Exception:
                pass
            logger.info("Disabled floating axes")
    
    def attach_scale_bar_widget(self, widget) -> None:
        """Attach the renderer-managed Qt scale bar widget."""
        self.scale_bar_widget = widget
        if not self.show_scale_bar_3d:
            try:
                widget.hide()
            except Exception:
                pass
        self._apply_scale_bar_anchor()
        self._install_scale_bar_callback()

    def set_scale_bar_3d_enabled(
        self,
        enabled: bool,
        units: str = "m",
        bar_fraction: float = 0.2,
        color: str = "white",
        font_size: float = 12.0,
        position_x: float = 0.8,
        position_y: float = 0.05,
        line_width: float = 2.0,
        anchor: Optional[str] = None
    ) -> None:
        """
        Enable or disable the dynamic scale bar using PyVista's ruler/scale capabilities.

        Creates a professional scale bar that:
        - Updates automatically when zooming in/out
        - Shows distance values based on current view
        - Positioned in screen space (doesn't move with scene)
        """
        self.show_scale_bar_3d = enabled
        self._scale_bar_params = {
            'units': units,
            'bar_fraction': bar_fraction,
            'color': color,
            'font_size': font_size,
            'position_x': position_x,
            'position_y': position_y,
            'line_width': line_width,
            'anchor': anchor
        }

        # Remove existing scale bar
        self._remove_scale_bar_actors()

        if not enabled or self.plotter is None:
            logger.info("Scale bar disabled")
            return

        try:
            logger.info(f"Creating scale bar, plotter={self.plotter}")

            from block_model_viewer.visualization.pyvista_axes_scalebar import ScaleBar3D

            # Position based on anchor (normalized viewport coordinates)
            anchor_name = (anchor or "bottom_right").lower()
            if "left" in anchor_name:
                pos_x = 0.08
            else:
                pos_x = 0.72
            if "top" in anchor_name:
                pos_y = 0.90
            else:
                pos_y = 0.08

            logger.info(f"ScaleBar3D position: ({pos_x}, {pos_y}), units={units}")

            # Create ScaleBar3D instance with proper settings
            self._scale_bar_3d = ScaleBar3D(
                units=units,
                bar_fraction=bar_fraction,
                color=color,
                text_color=color,
                line_width=max(2.0, line_width),
                position_x=pos_x,
                position_y=pos_y
            )

            logger.info(f"ScaleBar3D instance created: {self._scale_bar_3d}")

            # Add to plotter - registers update callback for zoom response
            self._scale_bar_3d.add_to_plotter(self.plotter)

            logger.info("ScaleBar3D added to plotter")

            # Force an immediate render to show the scale bar
            try:
                self.plotter.render()
                logger.info("Render completed")
            except Exception as re:
                logger.warning(f"Render exception: {re}")

        except Exception as e:
            logger.error(f"Failed to create scale bar: {e}", exc_info=True)

    def update_scale_bar_on_zoom(self) -> None:
        """Update the scale bar after a zoom event. Called from Qt wheel event handler."""
        if not self.show_scale_bar_3d or self.plotter is None:
            return
        if hasattr(self, '_scale_bar_3d') and self._scale_bar_3d is not None:
            try:
                self._scale_bar_3d.update(self.plotter)
                # Trigger render to show updated scale bar
                self.plotter.render()
            except Exception as e:
                logger.debug(f"Scale bar update failed: {e}")

    def _remove_scale_bar_actors(self) -> None:
        """Remove existing scale bar VTK actors."""
        if self.plotter is None:
            return
        try:
            # Clear ScaleBar3D instance if it exists
            if hasattr(self, '_scale_bar_3d') and self._scale_bar_3d is not None:
                try:
                    self._scale_bar_3d.clear(self.plotter)
                except Exception:
                    pass
                self._scale_bar_3d = None

            # Remove named actors (legacy cleanup)
            for name in ['_scale_bar_line', '_scale_bar_labels', '_scale_bar_label']:
                try:
                    self.plotter.remove_actor(name, render=False)
                except Exception:
                    pass
            # Clear actors list
            if hasattr(self, '_scale_bar_actors'):
                self._scale_bar_actors = []
            # Legacy cleanup
            if hasattr(self, '_scale_bar_text_actor'):
                self._scale_bar_text_actor = None
            if hasattr(self, '_scale_bar_line_actor'):
                self._scale_bar_line_actor = None
        except Exception:
            pass

    def _round_to_nice_number(self, value: float) -> float:
        """Round value to a 'nice' number (1, 2, 5, 10, 20, 50, 100, etc.)"""
        if value <= 0:
            return 1.0
        import math
        exponent = math.floor(math.log10(value))
        fraction = value / (10 ** exponent)
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10
        return nice_fraction * (10 ** exponent)

    def _derive_anchor_from_position(self, pos_x: float, pos_y: float) -> str:
        """Best-effort anchor inference from normalized coordinates."""
        if pos_y >= 0.5:
            return "top_right" if pos_x >= 0.5 else "top_left"
        return "bottom_right" if pos_x >= 0.5 else "bottom_left"

    def _install_scale_bar_callback(self) -> None:
        """Register a render callback to keep the HUD scale bar in sync."""
        if self._scale_bar_callback_registered or self.plotter is None:
            return
        try:
            self._scale_bar_render_cb = lambda *_: self._update_scale_bar_widget()
            self.plotter.add_on_render_callback(self._scale_bar_render_cb, render_event=True)
            self._scale_bar_callback_registered = True
        except Exception as exc:
            logger.debug(f"Could not install scale bar callback: {exc}")
        # Also watch the camera for modifications (zoom/pan) to force updates
        try:
            renderer = getattr(self.plotter, 'renderer', None)
            camera = renderer.GetActiveCamera() if renderer is not None else None
            if camera is not None and self._scale_bar_camera_cb is None:
                self._scale_bar_camera_cb = lambda *_: self._update_scale_bar_widget()
                camera.AddObserver("ModifiedEvent", self._scale_bar_camera_cb)
        except Exception as exc:
            logger.debug(f"Could not attach scale bar camera observer: {exc}")

    def _apply_scale_bar_anchor(self) -> None:
        if self.scale_bar_widget is None:
            return
        anchor = (self._scale_bar_params or {}).get('anchor', self._scale_bar_anchor_name)
        try:
            self.scale_bar_widget.set_anchor(anchor)
        except Exception:
            pass

    def _update_scale_bar_widget(self) -> None:
        if not self.show_scale_bar_3d or self.scale_bar_widget is None or self.plotter is None:
            return
        try:
            wpp = self._compute_world_per_pixel_camera()
            if wpp is None:
                wpp = self._compute_world_per_pixel()
            if wpp is None or wpp <= 0:
                return
            window_size = getattr(self.plotter, 'window_size', None)
            if not window_size or window_size[0] <= 0:
                return
            params = self._scale_bar_params or {}
            fraction = params.get('bar_fraction', 0.2) or 0.2
            display_width = max(1.0, float(window_size[0]))
            pixel_target = np.clip(display_width * fraction, 80.0, display_width * 0.9)
            world_target = pixel_target * wpp
            nice_length = self._choose_scale_bar_length(world_target)
            if nice_length <= 0:
                return
            px_length = np.clip(nice_length / wpp, 60.0, display_width * 0.9)
            units = params.get('units', 'm').strip()
            display_value, display_units = self._convert_length_for_display(nice_length, units)
            self.scale_bar_widget.set_scale(px_length, display_value, display_units)
            self._apply_scale_bar_anchor()
        except Exception as exc:
            logger.debug(f"Could not update scale bar widget: {exc}")

    def _choose_scale_bar_length(self, target: float) -> float:
        if target <= 0:
            return 0.0
        exp = int(np.floor(np.log10(target))) if target > 0 else 0
        base = 10 ** exp
        candidates = np.array([1.0, 2.0, 5.0, 10.0]) * base
        idx = int(np.abs(candidates - target).argmin())
        length = float(candidates[idx])
        if length <= 0:
            length = target
        return length

    def _convert_length_for_display(self, length_meters: float, units: str) -> Tuple[float, str]:
        """Convert world length (meters) into the panel-selected display units."""
        unit_key = (units or "").strip().lower()
        if not unit_key:
            return length_meters, "m"
        factors = {
            "m": 1.0,
            "meter": 1.0,
            "meters": 1.0,
            "km": 0.001,
            "kilometer": 0.001,
            "kilometers": 0.001,
            "mm": 1000.0,
            "millimeter": 1000.0,
            "millimeters": 1000.0,
            "ft": 3.28084,
            "foot": 3.28084,
            "feet": 3.28084,
        }
        factor = factors.get(unit_key, 1.0)
        converted = length_meters * factor
        label = units if units else "m"
        return converted, label

    def _compute_world_per_pixel_camera(self) -> Optional[float]:
        """Compute world-units per screen pixel based on current camera parameters."""
        if self.plotter is None:
            return None
        camera = getattr(self.plotter, 'camera', None)
        if camera is None:
            return None
        window_size = getattr(self.plotter, 'window_size', None)
        if not window_size or window_size[0] <= 0 or window_size[1] <= 0:
            return None
        aspect = window_size[0] / max(window_size[1], 1)
        try:
            if camera.GetParallelProjection():
                height = camera.GetParallelScale() * 2.0
            else:
                pos = np.array(camera.GetPosition())
                focal = np.array(camera.GetFocalPoint())
                dist = np.linalg.norm(focal - pos)
                if dist <= 1e-6:
                    return None
                fov = camera.GetViewAngle()
                height = 2 * dist * np.tan(np.deg2rad(fov) / 2.0)
            width = height * aspect
            return width / max(window_size[0], 1)
        except Exception:
            return None

    def _update_floating_axes_bounds(self) -> None:
        """Update floating axes bounds when scene bounds change."""
        if not self.show_floating_axes or self.floating_axes is None or self.plotter is None:
            return
        
        bounds = self._get_scene_bounds()
        if bounds is None:
            return
        
        x_range = (bounds[0], bounds[1])
        y_range = (bounds[2], bounds[3])
        z_range = (bounds[4], bounds[5])
        
        # Recreate axes with new bounds
        self.floating_axes.clear(self.plotter)
        
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        z_span = z_range[1] - z_range[0]
        avg_span = (x_span + y_span + z_span) / 3.0
        
        major_tick_length = avg_span * 0.02
        minor_tick_length = avg_span * 0.01
        label_size = avg_span * 0.015
        
        params = self._floating_axes_params or {}
        auto_spacing = params.get('auto_spacing', False)
        if auto_spacing:
            x_major, x_minor = self._auto_axis_spacing(x_span)
            y_major, y_minor = self._auto_axis_spacing(y_span)
            z_major, z_minor = self._auto_axis_spacing(z_span)
        else:
            spacings = self.floating_axes.spacings
            x_major = params.get('x_major', spacings['x']['major'])
            x_minor = params.get('x_minor', spacings['x']['minor'])
            y_major = params.get('y_major', spacings['y']['major'])
            y_minor = params.get('y_minor', spacings['y']['minor'])
            z_major = params.get('z_major', spacings['z']['major'])
            z_minor = params.get('z_minor', spacings['z']['minor'])
        axis_color = params.get('color', 'white')
        line_width = params.get('line_width', 2.0)
        label_size_override = params.get('font_size', label_size)
        # Pass label_offset to display world/UTM coordinates instead of local
        label_offset = self._global_shift if self._global_shift is not None else None
        self.floating_axes = FloatingAxes(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            x_major=x_major,
            x_minor=x_minor,
            y_major=y_major,
            y_minor=y_minor,
            z_major=z_major,
            z_minor=z_minor,
            major_tick_length=major_tick_length,
            minor_tick_length=minor_tick_length,
            axis_color=axis_color,
            tick_color=axis_color,
            label_color=axis_color,
            label_size=label_size_override,
            axis_line_width=line_width,
            tick_line_width=max(line_width * 0.7, 1.0),
            label_offset=label_offset
        )
        self.floating_axes.add_to_plotter(self.plotter, draw_box=params.get('draw_box', True))

    # =========================================================================
    # NORTH ARROW WIDGET SUPPORT
    # =========================================================================

    def attach_north_arrow_widget(self, widget) -> None:
        """Attach the renderer-managed Qt north arrow widget."""
        self.north_arrow_widget = widget
        if not self.show_north_arrow:
            try:
                widget.hide()
            except Exception:
                pass
        self._apply_north_arrow_anchor()
        self._install_north_arrow_callback()

    def set_north_arrow_enabled(
        self,
        enabled: bool,
        color: str = "white",
        font_size: float = 12.0,
        anchor: Optional[str] = None
    ) -> None:
        """
        Enable or disable the orientation/compass widget.

        Uses PyVista's built-in axes widget which:
        - Shows X/Y/Z orientation in a corner
        - Rotates WITH the camera to always show current orientation
        - Professional appearance like Leapfrog/Vulcan/Surpac
        - Y-axis (green) points North in mining convention

        Args:
            enabled: Whether to show the orientation widget
            color: Color theme (white, black, gray) - affects labels
            font_size: Not used (VTK controls size)
            anchor: Position anchor (top_right, top_left, bottom_right, bottom_left)
        """
        self.show_north_arrow = enabled
        anchor_name = (anchor or self._north_arrow_anchor_name or "top_right").lower()
        self._north_arrow_params = {
            'color': color,
            'font_size': font_size,
            'anchor': anchor_name
        }
        self._north_arrow_anchor_name = anchor_name

        # Remove existing orientation widget
        self._remove_north_arrow_actors()

        if not enabled or self.plotter is None:
            logger.info("Orientation widget disabled")
            return

        try:
            # Use VTK's vtkCameraOrientationWidget directly for QtInteractor compatibility
            logger.info(f"Creating orientation widget, plotter={self.plotter}")

            from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget

            # Create the camera orientation widget
            widget = vtkCameraOrientationWidget()

            # Set the parent renderer
            widget.SetParentRenderer(self.plotter.renderer)

            # Get the VTK interactor from the render window
            render_window = self.plotter.render_window if hasattr(self.plotter, 'render_window') else None
            if render_window is None and hasattr(self.plotter, 'ren_win'):
                render_window = self.plotter.ren_win

            if render_window is not None:
                vtk_interactor = render_window.GetInteractor()
                if vtk_interactor is not None:
                    widget.SetInteractor(vtk_interactor)
                    logger.info(f"Set interactor from render window: {vtk_interactor}")

            # Enable the widget
            widget.On()
            self._orientation_widget = widget

            logger.info(f"Orientation widget created: {self._orientation_widget}")

            # Force render
            try:
                self.plotter.render()
            except Exception as re:
                logger.debug(f"Render after axes: {re}")

        except Exception as e:
            logger.error(f"Failed to create orientation widget: {e}", exc_info=True)

    def _remove_north_arrow_actors(self) -> None:
        """Remove existing orientation widget and legacy north arrow actors."""
        if self.plotter is None:
            return
        try:
            # Remove VTK camera orientation widget if it exists
            if hasattr(self, '_orientation_widget') and self._orientation_widget is not None:
                try:
                    # Check if it's a vtkCameraOrientationWidget (has Off method)
                    if hasattr(self._orientation_widget, 'Off'):
                        self._orientation_widget.Off()
                    elif self._orientation_widget is True:
                        # Legacy: Widget exists but we don't have reference
                        if hasattr(self.plotter, 'hide_axes'):
                            self.plotter.hide_axes()
                    else:
                        self.plotter.remove_actor(self._orientation_widget, render=False)
                except Exception:
                    pass
                self._orientation_widget = None

            # Remove legacy named actors
            for name in ['_north_arrow', '_north_arrow_n', '_north_arrow_triangle', '_north_arrow_stem',
                         '_north_arrow_shaft', '_north_arrow_cone', '_north_arrow_label']:
                try:
                    self.plotter.remove_actor(name, render=False)
                except Exception:
                    pass

            # Clear actors list
            if hasattr(self, '_north_arrow_actors'):
                self._north_arrow_actors = []
            if hasattr(self, '_north_arrow_actor'):
                self._north_arrow_actor = None
        except Exception:
            pass

    def _apply_north_arrow_anchor(self) -> None:
        """Apply the anchor position to the north arrow widget."""
        if self.north_arrow_widget is None:
            return
        anchor = (self._north_arrow_params or {}).get('anchor', self._north_arrow_anchor_name)
        try:
            self.north_arrow_widget.set_anchor(anchor)
        except Exception:
            pass

    def _install_north_arrow_callback(self) -> None:
        """Register a camera callback to keep the north arrow rotation in sync."""
        if self._north_arrow_callback_registered or self.plotter is None:
            return
        try:
            renderer = getattr(self.plotter, 'renderer', None)
            camera = renderer.GetActiveCamera() if renderer is not None else None
            if camera is not None and self._north_arrow_camera_cb is None:
                self._north_arrow_camera_cb = lambda *_: self._update_north_arrow_rotation()
                camera.AddObserver("ModifiedEvent", self._north_arrow_camera_cb)
                self._north_arrow_callback_registered = True
        except Exception as exc:
            logger.debug(f"Could not attach north arrow camera observer: {exc}")

    def _update_north_arrow_rotation(self) -> None:
        """Update north arrow rotation based on camera azimuth."""
        if not self.show_north_arrow or self.north_arrow_widget is None or self.plotter is None:
            return
        try:
            camera = getattr(self.plotter, 'camera', None)
            if camera is None:
                return

            # Get camera position and focal point
            pos = np.array(camera.GetPosition())
            focal = np.array(camera.GetFocalPoint())

            # Calculate the view direction in XY plane (azimuth)
            view_dir = focal - pos
            # Project to XY plane
            dx = view_dir[0]
            dy = view_dir[1]

            # Calculate azimuth angle (0 = looking north/+Y, 90 = looking east/+X)
            # atan2 gives angle from +X axis, so we adjust to get angle from +Y axis
            azimuth = np.degrees(np.arctan2(dx, dy))

            # The north arrow should rotate opposite to the camera azimuth
            # so that it always points north in screen space
            self.north_arrow_widget.set_rotation(azimuth)
        except Exception as exc:
            logger.debug(f"Could not update north arrow rotation: {exc}")

    def _get_scene_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        DETERMINISTIC BOUNDS: Ignore HUD and Overlays.
        
        CRITICAL FIX: Compute bounds ONLY from real data layers (blocks, drillholes, geology).
        HUD actors (axes, scale bars, preview meshes) have expanded bounds that cause feedback loops.
        
        Returns (xmin, xmax, ymin, ymax, zmin, zmax) or None.
        """
        # 1. Return locked bounds if available (prevents drift)
        if self._fixed_bounds is not None:
            return self._fixed_bounds
        
        # 2. Use block model bounds if available (transform to local if shift is set)
        if self.current_model is not None and self.current_model.bounds is not None:
            try:
                bounds = self.current_model.bounds
                if self._global_shift is not None:
                    shift = self._global_shift
                    bounds = (
                        bounds[0] - shift[0], bounds[1] - shift[0],
                        bounds[2] - shift[1], bounds[3] - shift[1],
                        bounds[4] - shift[2], bounds[5] - shift[2],
                    )
                return bounds
            except Exception:
                pass
        
        # 3. Legacy fallback: _fixed_scene_bounds (for backward compatibility)
        if self._fixed_scene_bounds is not None:
            return self._fixed_scene_bounds

        # 4. Compute from DATA LAYERS ONLY (ignore overlays)
        data_bounds = []
        for layer_name, layer_data in self.active_layers.items():
            # Check both 'type' and 'layer_type' keys (different registration methods use different keys)
            layer_type = layer_data.get('layer_type', layer_data.get('type', ''))
            # ONLY include real data layers, NOT overlays or previews
            if layer_type in ('blocks', 'drillhole', 'drillholes', 'geology_surface', 'mesh', 'classification', 'volume'):
                actor = layer_data.get('actor')
                if actor is not None:
                    try:
                        bounds = actor.GetBounds()
                        if bounds and len(bounds) >= 6 and not all(b == 0 for b in bounds):
                            data_bounds.append(bounds)
                    except Exception:
                        pass

        # 5. Also check drillhole actors stored separately in _drillhole_hole_actors
        if hasattr(self, '_drillhole_hole_actors') and self._drillhole_hole_actors:
            for hole_id, actor in self._drillhole_hole_actors.items():
                if actor is not None:
                    try:
                        bounds = actor.GetBounds()
                        if bounds and len(bounds) >= 6 and not all(b == 0 for b in bounds):
                            data_bounds.append(bounds)
                    except Exception:
                        pass

        if not data_bounds:
            return None
        
        # Compute union of all data layer bounds
        xmin = min(b[0] for b in data_bounds)
        xmax = max(b[1] for b in data_bounds)
        ymin = min(b[2] for b in data_bounds)
        ymax = max(b[3] for b in data_bounds)
        zmin = min(b[4] for b in data_bounds)
        zmax = max(b[5] for b in data_bounds)
        
        # Lock the bounds on first computation to prevent drift
        self._fixed_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
        logger.debug(f"Locked scene bounds: {self._fixed_bounds}")
        
        return self._fixed_bounds

        # OLD CODE: Temporarily hide overlays to compute bounds (removed)
        overlay_actors = self._get_overlay_actors()
        hidden_overlays: List[pv.Actor] = []
        for actor in overlay_actors:
            if actor is None:
                continue
            try:
                if actor.GetVisibility():
                    actor.VisibilityOff()
                    hidden_overlays.append(actor)
            except Exception:
                pass

        try:
            if self.plotter is not None and (self.current_model is None or self.current_model.bounds is None):
                try:
                    bounds = self.plotter.renderer.ComputeVisiblePropBounds()
                    if bounds and len(bounds) >= 6:
                        computed_bounds = tuple(float(bounds[i]) for i in range(6))
                        if not self._is_valid_bounds(computed_bounds):
                            return None
                        span = max(
                            computed_bounds[1] - computed_bounds[0],
                            computed_bounds[3] - computed_bounds[2],
                            computed_bounds[5] - computed_bounds[4]
                        )
                        if span > 1_000_000:
                            logger.warning(
                                f"Computed bounds span too large ({span:.1f}), likely including floating axes/scale bar actors. "
                                "Using model bounds instead."
                            )
                            return None
                        return computed_bounds
                except Exception:
                    pass
        finally:
            for actor in hidden_overlays:
                try:
                    actor.VisibilityOn()
                except Exception:
                    pass

        return None

    def _get_overlay_actors(self) -> List[pv.Actor]:
        """Collect overlay actors so they can be hidden during bounds computation."""
        actors: List[pv.Actor] = []
        if self.floating_axes is not None:
            actors.extend(self.floating_axes.get_actors())
        return actors

    # ---------------------------------------------------------------------#
    # Scene refresh & convenience accessors (used by controllers/tests)
    # ---------------------------------------------------------------------#

    def refresh_scene(self) -> None:
        """
        Lightweight scene refresh hook.

        This method is intentionally defensive so it can be called safely by the
        controller and tests without requiring a fully initialised plotter. It:
        - Updates scene bounds / overlay state
        - Updates floating axes bounds (if enabled)
        - Notifies any attached axis_manager of the latest bounds
        - Triggers a plotter.render() when available
        """
        try:
            # 1) Refresh cached scene bounds and overlay state
            try:
                self._update_scene_bounds()
            except Exception:
                logger.debug("Renderer.refresh_scene: _update_scene_bounds failed", exc_info=True)

            # 2) Keep floating axes in sync with current bounds
            try:
                if self.show_floating_axes:
                    self._update_floating_axes_bounds()
            except Exception:
                logger.debug("Renderer.refresh_scene: _update_floating_axes_bounds failed", exc_info=True)

            # 3) Notify axis manager (if wired) so elevation / scale bar widgets can update
            bounds = None
            try:
                bounds = self._get_scene_bounds()
            except Exception:
                bounds = None

            axis_mgr = getattr(self, "axis_manager", None)
            if axis_mgr is not None and bounds is not None:
                try:
                    axis_mgr.set_bounds(bounds)
                except Exception:
                    logger.debug("Renderer.refresh_scene: axis_manager.set_bounds failed", exc_info=True)

            # 4) Trigger a render if a plotter is present
            plotter = getattr(self, "plotter", None)
            if plotter is not None and hasattr(plotter, "render"):
                try:
                    plotter.render()
                except Exception:
                    logger.debug("Renderer.refresh_scene: plotter.render failed", exc_info=True)
        except Exception:
            logger.error("Renderer.refresh_scene failed", exc_info=True)

    # Properties used by tests / higher-level controllers

    @property
    def active_property(self) -> Optional[str]:
        """Return the currently active scalar property name, if any."""
        return getattr(self, "current_property", None)

    @property
    def current_metadata(self) -> Dict[str, Any]:
        """
        Return a minimal metadata dict describing the current scalar property.

        This intentionally mirrors what the legend / UI needs: a title and
        colormap name, and can be extended in future without breaking callers.
        """
        title = getattr(self, "current_property", None)
        cmap = getattr(self, "current_colormap", None)
        return {
            "property": title,
            "title": title or "",
            "colormap": cmap,
        }

    @property
    def current_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Public accessor for the current scene bounds.

        Delegates to `_get_scene_bounds` so callers do not need to know about
        the internal fixed/computed bounds logic.
        """
        try:
            return self._get_scene_bounds()
        except Exception:
            logger.debug("Renderer.current_bounds: _get_scene_bounds failed", exc_info=True)
            return None

    def _compute_drillhole_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Return bounds covering all drillhole actors."""
        if not self._drillhole_hole_actors:
            return None
        bounds_list: List[Tuple[float, float, float, float, float, float]] = []
        for actor in self._drillhole_hole_actors.values():
            try:
                actor_bounds = actor.GetBounds()
                if self._is_valid_bounds(actor_bounds):
                    bounds_list.append(actor_bounds[:6])
            except Exception:
                continue
        if not bounds_list:
            return None
        min_x = min(b[0] for b in bounds_list)
        max_x = max(b[1] for b in bounds_list)
        min_y = min(b[2] for b in bounds_list)
        max_y = max(b[3] for b in bounds_list)
        min_z = min(b[4] for b in bounds_list)
        max_z = max(b[5] for b in bounds_list)
        return (min_x, max_x, min_y, max_y, min_z, max_z)

    def _is_valid_bounds(self, bounds: Optional[Tuple[float, ...]]) -> bool:
        """Check if bounds tuple is well-formed."""
        if bounds is None or len(bounds) < 6:
            return False
        x0, x1, y0, y1, z0, z1 = bounds[:6]
        return (x1 > x0) and (y1 > y0) and (z1 > z0)
    
    def _sanitize_bounds(self, bounds: Optional[Tuple[float, ...]]) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Return a clean bounds tuple or None."""
        if not self._is_valid_bounds(bounds):
            return None
        cleaned = tuple(float(b) for b in bounds[:6])
        return cleaned
    
    def _maintain_clipping_range(self) -> None:
        """
        NEUTRALIZED: This method was causing infinite render loops and precision loss.
        The clipping range is now handled once during load in _setup_geology_camera_and_bounds.
        """
        return
    
    def _compute_world_per_pixel(self) -> Optional[float]:
        """
        Compute world-space units per screen pixel for scale bar.
        Returns None if cannot compute.
        """
        if self.plotter is None:
            return None
        
        try:
            bounds = self._get_scene_bounds()
            if bounds is None:
                return None
            
            # Get viewport size
            renderer = self.plotter.renderer
            if renderer is None:
                return None
            
            viewport = renderer.GetViewport()
            if viewport is None:
                return None
            
            # Get window size
            ren_win = renderer.GetRenderWindow()
            if ren_win is None:
                return None
            
            size = ren_win.GetSize()
            if size is None or len(size) < 2:
                return None
            
            width_px = float(size[0])
            height_px = float(size[1])
            
            if width_px <= 0 or height_px <= 0:
                return None
            
            # Compute span in world space
            x_span = bounds[1] - bounds[0]
            y_span = bounds[3] - bounds[2]
            z_span = bounds[5] - bounds[4]
            max_span = max(x_span, y_span, z_span)
            
            # Approximate: assume viewport shows roughly the max span
            # This is a simplification; for accurate scale bar, you'd need camera frustum math
            wpp = max_span / max(width_px, height_px)
            return wpp
        except Exception as e:
            logger.debug(f"Could not compute world per pixel: {e}")
            return None

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

    def _generate_world_axis_labels(self, local_min: float, local_max: float, offset: float, n_labels: int) -> str:
        """
        Generate a string showing the world coordinate range for axis labels.

        Args:
            local_min: Minimum local coordinate value
            local_max: Maximum local coordinate value
            offset: Offset to add (global_shift component) to convert to world coords
            n_labels: Number of labels (not used, kept for future)

        Returns:
            String like "500274-500301" showing world coordinate range
        """
        world_min = local_min + offset
        world_max = local_max + offset

        def _format_coord(val: float, force_full: bool = False) -> str:
            """Format coordinate value. If force_full, show actual number."""
            abs_val = abs(val)
            if force_full:
                # Show actual value without compact notation
                if abs_val >= 1000:
                    return f"{val:.0f}"
                elif abs_val >= 1:
                    return f"{val:.1f}".rstrip('0').rstrip('.')
                else:
                    return f"{val:.2f}".rstrip('0').rstrip('.')
            # Compact notation for large numbers
            if abs_val >= 1_000_000:
                return f"{val / 1_000_000:.1f}M".rstrip('0').rstrip('.')
            elif abs_val >= 10_000:
                return f"{val / 1_000:.0f}k"
            elif abs_val >= 1_000:
                return f"{val / 1_000:.1f}k".rstrip('0').rstrip('.')
            elif abs_val >= 100:
                return f"{val:.0f}"
            elif abs_val >= 1:
                return f"{val:.1f}".rstrip('0').rstrip('.')
            else:
                return f"{val:.0f}"

        # Try compact notation first
        min_str = _format_coord(world_min, force_full=False)
        max_str = _format_coord(world_max, force_full=False)

        # If compact notation makes them look identical, show actual values
        if min_str == max_str:
            min_str = _format_coord(world_min, force_full=True)
            max_str = _format_coord(world_max, force_full=True)

        return f"{min_str}-{max_str}"

    def _update_cube_axes_labels(self, label_offset: np.ndarray) -> None:
        """
        Update the CubeAxesActor labels to show world coordinates instead of local.

        PyVista's show_bounds() creates a VTK CubeAxesActor. This method finds that
        actor and updates its axis labels to show world coordinates by adding the
        label_offset to each tick value.

        Args:
            label_offset: [x, y, z] offset to add to local coords to get world coords
        """
        if self.plotter is None:
            return

        try:
            # Find the CubeAxesActor created by show_bounds
            cube_axes = None
            for actor in self.plotter.renderer.GetActors2D():
                if hasattr(actor, 'GetClassName') and 'CubeAxes' in actor.GetClassName():
                    cube_axes = actor
                    break

            # Also check regular actors
            if cube_axes is None:
                for actor in self.plotter.renderer.GetActors():
                    if hasattr(actor, 'GetClassName') and 'CubeAxes' in actor.GetClassName():
                        cube_axes = actor
                        break

            # Check PyVista's internal reference
            if cube_axes is None and hasattr(self.plotter, '_cube_axes_actor'):
                cube_axes = self.plotter._cube_axes_actor

            if cube_axes is None:
                logger.debug("CubeAxesActor not found, cannot update labels")
                return

            # VTK CubeAxesActor supports SetAxisLabels to set custom labels
            # We need to generate labels for each axis that show world coordinates
            import vtk

            def generate_axis_labels(local_min, local_max, offset, n_labels=5):
                """Generate VTK string array with world coordinate labels."""
                labels = vtk.vtkStringArray()
                step = (local_max - local_min) / max(1, n_labels - 1)
                for i in range(n_labels):
                    local_val = local_min + i * step
                    world_val = local_val + offset
                    # Format the world value
                    abs_val = abs(world_val)
                    if abs_val >= 1_000_000:
                        label = f"{world_val / 1_000_000:.1f}M"
                    elif abs_val >= 10_000:
                        label = f"{world_val / 1_000:.0f}k"
                    elif abs_val >= 1_000:
                        label = f"{world_val / 1_000:.1f}k"
                    elif abs_val >= 100:
                        label = f"{world_val:.0f}"
                    elif abs_val >= 1:
                        label = f"{world_val:.1f}"
                    else:
                        label = f"{world_val:.2f}"
                    label = label.rstrip('0').rstrip('.') if '.' in label else label
                    labels.InsertNextValue(label)
                return labels

            # Get current bounds from the actor
            actor_bounds = cube_axes.GetBounds()
            if actor_bounds and len(actor_bounds) >= 6:
                x_labels = generate_axis_labels(actor_bounds[0], actor_bounds[1], label_offset[0])
                y_labels = generate_axis_labels(actor_bounds[2], actor_bounds[3], label_offset[1])
                z_labels = generate_axis_labels(actor_bounds[4], actor_bounds[5], label_offset[2])

                cube_axes.SetAxisLabels(0, x_labels)  # X axis
                cube_axes.SetAxisLabels(1, y_labels)  # Y axis
                cube_axes.SetAxisLabels(2, z_labels)  # Z axis

                logger.debug("Updated CubeAxesActor labels to show world coordinates")

        except Exception as e:
            logger.debug(f"Could not update CubeAxesActor labels: {e}")

    def _update_scene_bounds(self) -> None:
        """
        Update scene bounds via the unified OverlayManager.
        
        This method computes bounds and delegates to OverlayManager, which
        handles all overlay updates and widget notifications.
        """
        # CRITICAL: Always use model bounds for HUD overlay, NEVER computed scene bounds
        # Computed scene bounds include HUD actors (with expanded bounds), causing feedback loop
        
        # Get bounds for overlays (can use computed bounds)
        computed_bounds = self._sanitize_bounds(self._get_scene_bounds())
        if computed_bounds is None:
            logger.debug("Scene bounds invalid from renderer; attempting fallbacks")
            candidate = self._sanitize_bounds(self.current_model.bounds if self.current_model and self.current_model.bounds else None)
            if candidate is None:
                candidate = self._sanitize_bounds(self._fixed_scene_bounds)
            if candidate is None:
                candidate = self._sanitize_bounds(self._compute_drillhole_bounds())
            if candidate is None:
                logger.debug("No scene bounds available to update")
                return
            computed_bounds = candidate
        
        # Lock bounds for future queries
        self._fixed_scene_bounds = computed_bounds
        
        logger.info(f"Updating scene bounds: {computed_bounds}")
        
        # Delegate to unified overlay manager (handles signals, widgets, and rebuilds)
        self.overlay_state.set_bounds(computed_bounds)
    
    def toggle_overlay(self, name: str, enabled: bool) -> None:
        """
        Unified overlay toggle API - routes to the OverlayManager.
        
        Args:
            name: Overlay name ('axes', 'ground_grid', 'grid', 'bounds', 'bounding_box', 'scale_bar', 'orientation_gizmo')
            enabled: True to show, False to hide
        """
        logger.info(f"Renderer.toggle_overlay: {name} = {enabled}")
        
        # Delegate to unified overlay manager
        self.overlay_state.toggle_overlay(name, enabled)
    
    # ============================================================================
    # PASSIVE OVERLAY API (called by OverlayManager)
    # ============================================================================
    
    def add_overlay_actor(self, actor: Any, overlay_type: str = 'other') -> None:
        """
        Add an overlay actor to the scene.
        
        This is a passive API called by OverlayManager. The Renderer does not
        decide what overlays to show - it simply draws what the OverlayManager
        tells it to draw.
        
        Args:
            actor: PyVista/VTK actor to add
            overlay_type: Type of overlay ('axes', 'scale_bar', 'grid', 'bounds', 'other')
        """
        if self.plotter is None:
            logger.debug("Cannot add overlay actor: plotter not initialized")
            return
        
        try:
            # Add to plotter
            if hasattr(self.plotter, 'add_actor'):
                self.plotter.add_actor(actor, pickable=False)
            
            # Track the actor
            if overlay_type not in self._overlay_actors:
                self._overlay_actors[overlay_type] = []
            self._overlay_actors[overlay_type].append(actor)
            
            logger.debug(f"Added overlay actor of type '{overlay_type}'")
        except Exception as e:
            logger.warning(f"Could not add overlay actor: {e}", exc_info=True)
    
    def remove_overlay_actor(self, actor: Any) -> None:
        """
        Remove a specific overlay actor from the scene.
        
        Args:
            actor: The actor to remove
        """
        if self.plotter is None:
            return
        
        try:
            # Remove from plotter
            if hasattr(self.plotter, 'remove_actor'):
                self.plotter.remove_actor(actor)
            
            # Remove from tracking
            for overlay_type, actors in self._overlay_actors.items():
                if actor in actors:
                    actors.remove(actor)
                    break
            
            logger.debug("Removed overlay actor")
        except Exception as e:
            logger.debug(f"Could not remove overlay actor: {e}")
    
    def clear_overlay_actors(self, overlay_type: Optional[str] = None) -> None:
        """
        Clear overlay actors from the scene.
        
        Args:
            overlay_type: If specified, clear only actors of this type.
                         If None, clear all overlay actors.
        """
        if self.plotter is None:
            return
        
        try:
            if overlay_type is not None:
                # Clear specific type
                actors = self._overlay_actors.get(overlay_type, [])
                for actor in actors:
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass
                self._overlay_actors[overlay_type] = []
                logger.debug(f"Cleared overlay actors of type '{overlay_type}'")
            else:
                # Clear all overlay actors
                for otype, actors in self._overlay_actors.items():
                    for actor in actors:
                        try:
                            self.plotter.remove_actor(actor)
                        except Exception:
                            pass
                    self._overlay_actors[otype] = []
                logger.debug("Cleared all overlay actors")
        except Exception as e:
            logger.debug(f"Could not clear overlay actors: {e}")
    
    def get_overlay_actors(self, overlay_type: Optional[str] = None) -> List[Any]:
        """
        Get current overlay actors.
        
        Args:
            overlay_type: If specified, get only actors of this type.
                         If None, get all overlay actors.
        
        Returns:
            List of overlay actors
        """
        if overlay_type is not None:
            return list(self._overlay_actors.get(overlay_type, []))
        
        all_actors = []
        for actors in self._overlay_actors.values():
            all_actors.extend(actors)
        return all_actors
    
    def fit_to_view(self) -> None:
        """Fit the model to the current view."""
        import time
        fit_start = time.time()
        if self.plotter is not None:
            try:
                # FIXED: Compute bounds from ALL data layers (block model, geology, AND drillholes)
                # Previously only used block model or geology bounds, causing drillholes to be clipped
                all_bounds = []
                
                # 1. Block model bounds (MUST transform to local coordinates)
                if self.current_model is not None and self.current_model.bounds is not None:
                    bm_bounds = self.current_model.bounds
                    # Transform to local coordinates if global shift is set
                    if self._global_shift is not None:
                        shift = self._global_shift
                        bm_bounds = (
                            bm_bounds[0] - shift[0], bm_bounds[1] - shift[0],
                            bm_bounds[2] - shift[1], bm_bounds[3] - shift[1],
                            bm_bounds[4] - shift[2], bm_bounds[5] - shift[2],
                        )
                        logger.debug(f"fit_to_view: Block model bounds transformed to local: {bm_bounds}")
                    all_bounds.append(bm_bounds)
                    logger.debug(f"fit_to_view: Using block model bounds: {bm_bounds}")
                
                # 2. Geology bounds
                if self._geology_bounds is not None:
                    all_bounds.append(self._geology_bounds)
                    logger.debug(f"fit_to_view: Using geology bounds: {self._geology_bounds}")
                
                # 3. Drillhole bounds (CRITICAL for visibility)
                if hasattr(self, '_drillhole_hole_actors') and self._drillhole_hole_actors:
                    for hid, actor in self._drillhole_hole_actors.items():
                        if actor is not None and hasattr(actor, 'GetBounds'):
                            ab = actor.GetBounds()
                            if ab and ab[0] < ab[1]:  # Valid bounds
                                all_bounds.append(ab)
                    if len(self._drillhole_hole_actors) > 0:
                        logger.debug(f"fit_to_view: Including {len(self._drillhole_hole_actors)} drillhole bounds")
                
                # Combine all bounds
                bounds = None
                if all_bounds:
                    xmin = min(b[0] for b in all_bounds)
                    xmax = max(b[1] for b in all_bounds)
                    ymin = min(b[2] for b in all_bounds)
                    ymax = max(b[3] for b in all_bounds)
                    zmin = min(b[4] for b in all_bounds)
                    zmax = max(b[5] for b in all_bounds)
                    bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
                    logger.info(f"fit_to_view: Combined bounds X[{xmin:.0f}-{xmax:.0f}] Y[{ymin:.0f}-{ymax:.0f}] Z[{zmin:.0f}-{zmax:.0f}]")
                
                # If we have bounds, set a good initial isometric view
                if bounds is not None:
                    center = [
                        (bounds[0] + bounds[1]) / 2, 
                        (bounds[2] + bounds[3]) / 2, 
                        (bounds[4] + bounds[5]) / 2
                    ]
                    
                    # Calculate appropriate camera distance
                    size = max(
                        bounds[1] - bounds[0], 
                        bounds[3] - bounds[2], 
                        bounds[5] - bounds[4]
                    )
                    
                    # Set camera to isometric view with proper distance
                    # Use plotter.camera_position property for efficient camera setting
                    # This is faster than direct camera manipulation and doesn't trigger expensive pipeline updates
                    camera_start = time.time()
                    distance = size * 2.0  # Ensure model is fully visible
                    
                    focal_point = center
                    position = [
                        center[0] + distance * 0.7, 
                        center[1] + distance * 0.7, 
                        center[2] + distance * 0.7
                    ]
                    up = [0, 0, 1]
                    
                    # Use direct VTK camera manipulation for reliability
                    try:
                        camera = self.plotter.renderer.GetActiveCamera()
                        if camera:
                            camera.SetPosition(position[0], position[1], position[2])
                            camera.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
                            camera.SetViewUp(up[0], up[1], up[2])
                            
                            # Set clipping range based on model bounds
                            model_size = size
                            cam_distance = distance
                            near_plane = max(size * 0.001, 1.0)  # At least 1 unit
                            far_plane = max(model_size * 50.0, cam_distance * 100.0)
                            camera.SetClippingRange(near_plane, far_plane)
                            
                            logger.info(f"fit_to_view: Camera at ({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})")
                    except Exception as e:
                        logger.debug(f"Could not set camera directly: {e}")
                    
                    # Also try PyVista method as backup
                    try:
                        self.plotter.camera_position = [position, focal_point, up]
                    except Exception:
                        pass
                    
                    camera_time = time.time() - camera_start
                    logger.info(f"PERF: Setting camera via camera_position property took {camera_time:.3f}s")
                    
                    # Force a render and Qt repaint to display immediately
                    try:
                        self.plotter.render()
                        if hasattr(self.plotter, 'interactor') and self.plotter.interactor is not None:
                            self.plotter.interactor.repaint()
                            self.plotter.interactor.update()
                    except Exception:
                        pass
                else:
                    # Fallback to reset_camera if no bounds available
                    # HUD actors are already hidden, so reset_camera will use model bounds only
                    reset_start = time.time()
                    self.plotter.reset_camera()
                    reset_time = time.time() - reset_start
                    logger.info(f"PERF: reset_camera() took {reset_time:.3f}s")
                    
                    # Force Qt repaint after reset_camera
                    try:
                        self.plotter.render()
                        if hasattr(self.plotter, 'interactor') and self.plotter.interactor is not None:
                            self.plotter.interactor.repaint()
                            self.plotter.interactor.update()
                    except Exception:
                        pass
                
                fit_time = time.time() - fit_start
                logger.info(f"PERF: fit_to_view total took {fit_time:.3f}s")
                logger.info("Fitted model to view")
                
                # CRITICAL: Remove any axes that might have been created by camera operations
                self._remove_all_axes_actors()
            except Exception as e:
                logger.warning(f"Error in fit_to_view: {e}, falling back to reset_camera")
                try:
                    self.plotter.reset_camera()
                    # Remove axes after reset_camera as well
                    self._remove_all_axes_actors()
                except Exception:
                    pass
    
    def set_view_preset(self, preset: str) -> None:
        """
        Set camera to a preset view.
        
        Args:
            preset: View name (Top, Bottom, Front, Back, Right, Left, Isometric)
        """
        if self.plotter is None:
            return
        
        try:
            preset_lower = preset.lower()
            
            if preset_lower == 'top':
                self.plotter.view_xy()
            elif preset_lower == 'bottom':
                self.plotter.view_xy(negative=True)
            elif preset_lower == 'front':
                self.plotter.view_xz()
            elif preset_lower == 'back':
                self.plotter.view_xz(negative=True)
            elif preset_lower == 'right':
                self.plotter.view_yz()
            elif preset_lower == 'left':
                self.plotter.view_yz(negative=True)
            elif preset_lower == 'side':
                # Map "side" to "right" view
                self.plotter.view_yz()
            elif preset_lower == 'isometric':
                self.plotter.view_isometric()
            else:
                logger.warning(f"Unknown view preset: {preset}")
                return
            
            logger.info(f"Set view preset: {preset}")
        except Exception as e:
            logger.error(f"Error setting view preset {preset}: {e}")
    
    def set_legend_settings(self, position: str = 'vertical', font_size: int = 12) -> None:
        """
        Set legend (scalar bar) settings.
        
        Args:
            position: 'vertical' or 'horizontal'
            font_size: Font size for legend text
        """
        self.legend_position = position
        self.legend_font_size = font_size
        logger.info(f"Set legend settings: position={position}, font_size={font_size}")
        
        # Refresh the legend to apply the new settings
        if self.current_property and self.legend_visible:
            self.refresh_legend()
    
    def update_legend_style(self, style_dict: dict) -> None:
        """
        Update comprehensive legend styling parameters.
        
        Args:
            style_dict: Dictionary with legend style parameters from LegendStyleState
                {
                    'count': int,              # Number of tick labels
                    'decimals': int,           # Decimal precision
                    'shadow': bool,            # Shadow effect
                    'outline': bool,           # Outline border
                    'background': tuple,       # RGB background color (0-1)
                    'background_opacity': float, # Background transparency
                    'orientation': str,        # 'vertical' or 'horizontal'
                    'font_size': int,          # Font size in points
                    'mode': str                # 'continuous' or 'discrete'
                }
        """
        try:
            # Update renderer legend parameters
            if 'orientation' in style_dict:
                self.legend_position = style_dict['orientation']
            
            if 'font_size' in style_dict:
                self.legend_font_size = int(style_dict['font_size'])
            
            if 'count' in style_dict:
                self.legend_label_count = int(style_dict['count'])
            
            if 'decimals' in style_dict:
                decimals = int(style_dict['decimals'])
                self.legend_label_format = f"%.{decimals}f"
            
            if 'shadow' in style_dict:
                self.legend_shadow = bool(style_dict['shadow'])
            
            if 'outline' in style_dict:
                self.legend_outline = bool(style_dict['outline'])
            
            if 'background' in style_dict:
                self.legend_background_color = tuple(style_dict['background'])
            
            if 'background_opacity' in style_dict:
                self.legend_background_opacity = float(style_dict['background_opacity'])
            
            if 'mode' in style_dict:
                self.legend_mode = style_dict['mode']
            
            logger.info(f"Updated legend style: orientation={self.legend_position}, font_size={self.legend_font_size}, count={getattr(self, 'legend_label_count', 5)}")
            
            # Force refresh the legend if a property is currently displayed
            if self.current_property and self.legend_visible:
                self.refresh_legend()
                
        except Exception as e:
            logger.error(f"Error updating legend style: {e}", exc_info=True)
    
    def _refresh_legend_from_active_layer(self) -> None:
        """
        Refresh legend from the currently active layer.
        Called by property panel and main window when layer properties change.
        """
        if not hasattr(self, 'legend_manager') or self.legend_manager is None:
            return
        
        try:
            # Find the active layer
            if not self.active_layers:
                return
            
            # Try to use the most recently updated layer or first available
            active_layer_name = None
            if hasattr(self, '_active_control_layer') and self._active_control_layer:
                active_layer_name = self._active_control_layer
            elif self.active_layers:
                # Prefer drillholes if present (commonly used for property exploration)
                if 'drillholes' in self.active_layers:
                    active_layer_name = 'drillholes'
                else:
                    active_layer_name = list(self.active_layers.keys())[0]
            
            if not active_layer_name or active_layer_name not in self.active_layers:
                return
            
            layer_info = self.active_layers[active_layer_name]
            layer_data = layer_info.get('data')
            
            # Use legend_manager.update_from_renderer to trigger refresh
            self.legend_manager.update_from_renderer(force=True, reason="manual_refresh")
            
            logger.debug(f"Refreshed legend from active layer: {active_layer_name}")
        except Exception as e:
            logger.debug(f"Failed to refresh legend from active layer: {e}")
    
    def refresh_legend(self) -> None:
        """
        Force refresh the current legend/scalar bar.
        Useful after updating legend style parameters.
        """
        if not self.current_property or not self.plotter:
            return
        
        try:
            # Re-apply the current property to rebuild the scalar bar with new settings
            logger.debug(f"Refreshing legend for property: {self.current_property}")
            self.color_by_property(self.current_property, refresh=True)
        except Exception as e:
            logger.error(f"Error refreshing legend: {e}", exc_info=True)
    
    def set_axis_font(self, family: str = None, size: int = None) -> None:
        """
        Set axis font settings.
        
        Args:
            family: Font family name (e.g., 'Arial', 'Times')
            size: Font size in points
        """
        if family is not None:
            self.axis_font_family = family
        if size is not None:
            self.axis_font_size = size
        
        # Refresh the display
        self.refresh_bounds()
        logger.info(f"Set axis font: {self.axis_font_family}, size={self.axis_font_size}")
    
    def set_axis_font_color(self, color) -> None:
        """
        Set axis font color.
        
        Args:
            color: Color as RGB tuple (0-1) or color name
        """
        self.axis_font_color = color
        
        # Refresh the display
        self.refresh_bounds()
        logger.info(f"Set axis font color: {color}")
    
    def set_opacity(self, opacity: float) -> None:
        """
        Set block opacity with instant update.
        
        Args:
            opacity: Opacity value (0.0 = transparent, 1.0 = opaque)
        """
        self.current_opacity = opacity
        self.set_transparency(opacity)

    def set_render_style(self, mode: str) -> None:
        """
        Set global mesh render style (solid / wireframe / solid_edges).

        This operates at the VTK property level so it applies consistently to
        whatever actors are currently in the scene.
        """
        if self.plotter is None or getattr(self.plotter, "renderer", None) is None:
            return
        style = (mode or "solid").lower()
        self._render_style = style
        try:
            for actor in self.plotter.renderer.actors.values():
                if not hasattr(actor, "GetProperty"):
                    continue
                prop = actor.GetProperty()
                if prop is None:
                    continue
                if style == "wireframe":
                    try:
                        prop.SetRepresentationToWireframe()
                    except Exception:
                        continue
                    try:
                        prop.SetEdgeVisibility(False)
                    except Exception:
                        pass
                elif style == "solid_edges":
                    try:
                        prop.SetRepresentationToSurface()
                    except Exception:
                        continue
                    try:
                        prop.SetEdgeVisibility(True)
                    except Exception:
                        pass
                else:  # "solid"
                    try:
                        prop.SetRepresentationToSurface()
                    except Exception:
                        continue
                    try:
                        prop.SetEdgeVisibility(False)
                    except Exception:
                        pass
            logger.info(f"Set global render style to '{style}'")
        except Exception:
            logger.debug("Renderer.set_render_style failed", exc_info=True)

    def set_shading_mode(self, mode: str | bool) -> None:
        """
        Set global shading mode to 'smooth' or 'flat'.

        Smooth shading uses Phong interpolation; flat shading uses face-based
        interpolation which is often preferred for geological contacts.
        """
        if isinstance(mode, bool):
            smooth = mode
        else:
            smooth = str(mode).lower().startswith("smooth")
        self.smooth_shading = bool(smooth)
        if self.plotter is None or getattr(self.plotter, "renderer", None) is None:
            return
        try:
            for actor in self.plotter.renderer.actors.values():
                if not hasattr(actor, "GetProperty"):
                    continue
                prop = actor.GetProperty()
                if prop is None:
                    continue
                try:
                    if self.smooth_shading:
                        # Phong shading for smooth appearance
                        prop.SetInterpolationToPhong()
                    else:
                        # Flat shading for faceted, geological look
                        prop.SetInterpolationToFlat()
                except Exception:
                    continue
            logger.info(f"Set shading mode to {'smooth' if self.smooth_shading else 'flat'}")
        except Exception:
            logger.debug("Renderer.set_shading_mode failed", exc_info=True)

    def set_line_width(self, width: float) -> None:
        """
        Set global line width for mesh edges, polylines and overlays.
        """
        try:
            w = float(width)
        except Exception:
            return
        w = max(1.0, min(w, 10.0))
        self.edge_width = w
        if self.plotter is None or getattr(self.plotter, "renderer", None) is None:
            return
        try:
            for actor in self.plotter.renderer.actors.values():
                if not hasattr(actor, "GetProperty"):
                    continue
                prop = actor.GetProperty()
                if prop is None:
                    continue
                try:
                    prop.SetLineWidth(w)
                except Exception:
                    continue
            logger.info(f"Set global line width to {w}")
        except Exception:
            logger.debug("Renderer.set_line_width failed", exc_info=True)

    def set_point_size(self, size: float) -> None:
        """
        Set global point size for collars, scatter points and point clouds.
        """
        try:
            s = float(size)
        except Exception:
            return
        s = max(1.0, min(s, 50.0))
        self.point_size = s
        if self.plotter is None or getattr(self.plotter, "renderer", None) is None:
            return
        try:
            for actor in self.plotter.renderer.actors.values():
                if not hasattr(actor, "GetProperty"):
                    continue
                prop = actor.GetProperty()
                if prop is None:
                    continue
                try:
                    prop.SetPointSize(s)
                except Exception:
                    continue
            logger.info(f"Set global point size to {s}")
        except Exception:
            logger.debug("Renderer.set_point_size failed", exc_info=True)
    
    def set_edge_visibility(self, visible: bool) -> None:
        """
        Toggle edge visibility for all actors in the scene.
        Used during resize to prevent GPU overload from edge recalculation.

        Args:
            visible: True to show edges, False to hide edges
        """
        if self.plotter is None:
            return

        edge_count = 0
        try:
            # Update all actors in active layers
            for layer_name, layer_info in self.active_layers.items():
                actor = layer_info.get('actor')
                if actor is not None:
                    try:
                        prop = actor.GetProperty()
                        if prop is not None:
                            # Store original edge visibility state if not already stored
                            if not hasattr(self, '_original_edge_states'):
                                self._original_edge_states = {}

                            # Save original state before first change
                            if layer_name not in self._original_edge_states:
                                self._original_edge_states[layer_name] = prop.GetEdgeVisibility()

                            # Set new visibility
                            prop.SetEdgeVisibility(visible)
                            edge_count += 1
                    except Exception as e:
                        logger.debug(f"Could not set edge visibility for {layer_name}: {e}")

            # Also update legacy mesh_actor if present
            if self.mesh_actor is not None:
                try:
                    prop = self.mesh_actor.GetProperty()
                    if prop is not None:
                        prop.SetEdgeVisibility(visible)
                        edge_count += 1
                except Exception as e:
                    logger.debug(f"Could not set edge visibility for mesh_actor: {e}")

            if edge_count > 0:
                logger.debug(f"[EDGE VISIBILITY] Set to {visible} for {edge_count} actors")
        except Exception as e:
            logger.warning(f"Error setting edge visibility: {e}")

    def get_edge_visibility(self) -> bool:
        """
        Get the current edge visibility state from the first available actor.
        Returns the user's original preference if edges were temporarily hidden.

        Returns:
            True if edges are visible (or should be visible), False otherwise
        """
        # If we have stored original states, return the first one (user's preference)
        if hasattr(self, '_original_edge_states') and self._original_edge_states:
            return next(iter(self._original_edge_states.values()))

        # Otherwise, check current state from first available actor
        try:
            for layer_name, layer_info in self.active_layers.items():
                actor = layer_info.get('actor')
                if actor is not None:
                    prop = actor.GetProperty()
                    if prop is not None:
                        return bool(prop.GetEdgeVisibility())
        except Exception:
            pass

        # Default to True (edges visible)
        return True

    def set_edge_color(self, color) -> None:
        """
        Set edge color with instant update.

        Args:
            color: RGB tuple (0-1) or color name
        """
        self.edge_color = color

        # Update mesh actor if available
        if self.mesh_actor is not None:
            try:
                self.mesh_actor.GetProperty().SetEdgeColor(color)
                # Defer render - PyVista will render on next frame automatically
                logger.info(f"Set edge color: {color}")
            except Exception as e:
                logger.warning(f"Could not set edge color: {e}")
    
    def set_background_color(self, color) -> None:
        """
        Set background color with instant update.
        
        Args:
            color: RGB tuple (0-1) or color name
        """
        self.background_color = color
        
        if self.plotter is not None:
            self.plotter.set_background(color)
            # Defer render - PyVista will render on next frame automatically
            logger.info(f"Set background color: {color}")
    
    def toggle_lighting(self, enabled: bool) -> None:
        """
        Toggle lighting on/off with instant update.
        
        Args:
            enabled: True to enable lighting, False to disable
        """
        self.lighting_enabled = enabled
        
        # Update mesh actor if available
        if self.mesh_actor is not None:
            try:
                self.mesh_actor.GetProperty().SetLighting(enabled)
                # Defer render - PyVista will render on next frame automatically
                logger.info(f"Set lighting: {enabled}")
            except Exception as e:
                logger.warning(f"Could not toggle lighting: {e}")
    
    def update_legend_font_size(self, size: int) -> None:
        """
        Update legend font size.
        
        Args:
            size: Font size in points
        """
        self.legend_font_size = size
        logger.info(f"Set legend font size: {size}")
        
        # Refresh the legend to apply the new font size
        if self.current_property and self.legend_visible:
            self.refresh_legend()
    
    def set_legend_visibility(self, visible: bool) -> None:
        """
        Set legend bar visibility.
        
        Args:
            visible: True to show, False to hide
        """
        self.legend_visible = visible
        
        if self.plotter is None:
            return
        
        # Toggle scalar bar visibility using VTK's visibility control
        try:
            if hasattr(self.plotter.renderer, '_scalar_bar'):
                scalar_bar = self.plotter.renderer._scalar_bar
                if scalar_bar:
                    scalar_bar.SetVisibility(1 if visible else 0)
                    # Defer render - PyVista will render on next frame automatically
                    # Note: We're not using PyVista scalar bars, but keeping for compatibility
                    logger.info(f"Set legend visibility: {visible}")
                    return
        except Exception as e:
            logger.debug(f"Could not toggle scalar bar visibility directly: {e}")
        
        # Fallback: Re-render with current property if we have one
        if self.current_property and visible:
            # Force re-render by re-applying current property
            self.color_by_property(self.current_property, refresh=True)
        elif not visible:
            # Just update the state, the next render will respect it
            logger.info(f"Legend visibility set to: {visible} (will apply on next render)")
    
    def set_legend_orientation(self, orientation: str) -> None:
        """
        Set legend bar orientation.
        
        Args:
            orientation: 'vertical' or 'horizontal'
        """
        if orientation not in ['vertical', 'horizontal']:
            logger.warning(f"Invalid legend orientation: {orientation}")
            return
        
        self.legend_position = orientation
        logger.info(f"Set legend orientation: {orientation}")
        
        # Refresh the legend to apply the new orientation
        if self.current_property and self.legend_visible:
            self.refresh_legend()
    
    def reset_legend_position(self) -> None:
        """Reset legend bar to default position."""
        # Default positions
        if self.legend_position == 'vertical':
            self.legend_x = 0.85
            self.legend_y = 0.05
        else:  # horizontal
            self.legend_x = 0.3
            self.legend_y = 0.05
        
        logger.info("Reset legend position to defaults")
    
    # ============================================================================
    # UNIFIED LAYER MANAGEMENT METHODS (DYNAMIC)
    # ============================================================================
    
    def add_layer(self, layer_name: str, actor, data=None, layer_type: str = 'default',
                  opacity: float = None) -> None:
        """
        Add or update a visualization layer dynamically.

        This method allows ANY visualization to be added as a layer in the scene.
        Works for drillholes, kriging, block models, IRR results, pit optimization, etc.

        Args:
            layer_name: Unique name for this layer (e.g., 'drillholes', 'irr_schedule', 'ultimate_pit')
            actor: PyVista actor for this visualization
            data: Optional data associated with this layer
            layer_type: Type of visualization ('drillhole', 'volume', 'blocks', 'mesh', 'schedule', 'pit', etc.)
            opacity: Optional opacity override (uses default for layer_type if not specified)
        """
        logger.debug(f"[LAYER] add_layer called on renderer {id(self)} for layer '{layer_name}' (type={layer_type})")
        # BUG FIX: For certain layer types, remove ALL existing layers of that type
        # to prevent stacking multiple layers on top of each other.
        #
        # MUTUALLY EXCLUSIVE LAYERS (only one at a time):
        # - Block Models: "Block Model: FE_PCT", "Block Model: AL2O3_PCT" → only show one
        # - Kriging: "Kriging: FE_PCT", "Kriging: AL2O3_PCT" → only show one result
        # - SGSIM: "SGSIM: FE_PCT", "SGSIM: AL2O3_PCT" → only show one simulation
        # - Classification: "Resource Classification" → only show one classification layer
        # - ANY blocks/volume/classification layer_type → all are mutually exclusive
        #
        # NON-EXCLUSIVE LAYERS (can show multiple):
        # - Geology: Users want to see ORE + WASTE + etc. together
        # - Faults/Folds/Unconformities: Users want to see multiple structural features
        # - Drillholes: Can have multiple drillhole datasets

        # Define all prefixes that represent block/volume visualizations
        block_volume_prefixes = ["Block Model", "Kriging", "SGSIM", "Resource Classification"]

        # Check if this is a block/volume layer (by name prefix OR by layer_type)
        is_block_volume_layer = (
            any(layer_name.startswith(prefix) for prefix in block_volume_prefixes) or
            layer_type in ('blocks', 'volume', 'classification')
        )

        if is_block_volume_layer:
            # Remove ALL existing block/volume layers (with ANY of the prefixes)
            layers_to_remove = [
                name for name in list(self.active_layers.keys())
                if any(name.startswith(prefix) for prefix in block_volume_prefixes) or
                   self.active_layers[name].get('type') in ('blocks', 'volume', 'classification')
            ]
            if layers_to_remove:
                logger.info(f"Removing {len(layers_to_remove)} existing block/volume layer(s) to prevent stacking")
                for old_layer in layers_to_remove:
                    logger.info(f"  - Removing: '{old_layer}'")
                    self.clear_layer(old_layer)
        else:
            # Not a mutually exclusive layer type - just remove if same name exists
            if layer_name in self.active_layers:
                self.clear_layer(layer_name)
        
        # Determine opacity
        if opacity is None:
            opacity = self.default_opacity.get(layer_type, self.default_opacity['default'])
        
        # Create layer entry
        self.active_layers[layer_name] = {
            'actor': actor,
            'data': data,
            'visible': True,
            'opacity': opacity,
            'type': layer_type
        }
        
        # CRITICAL FIX: Also register in scene_layers for AppState detection
        # This ensures _update_state_from_scene() sees the layer and transitions to RENDERED
        self.register_scene_layer(layer_name, actor, data, layer_type)
        
        logger.info(f"Added layer '{layer_name}' (type: {layer_type}, opacity: {opacity})")
        
        # Notify UI of layer change
        if self.layer_change_callback:
            logger.info(f"[LAYER CALLBACK] Invoking layer_change_callback after adding layer '{layer_name}'")
            try:
                self.layer_change_callback()
                logger.info(f"[LAYER CALLBACK] Callback completed successfully for layer '{layer_name}'")
            except Exception as cb_err:
                logger.error(f"[LAYER CALLBACK] Callback failed for layer '{layer_name}': {cb_err}", exc_info=True)
        else:
            logger.error(f"[LAYER CALLBACK] CRITICAL: No layer_change_callback set when adding layer '{layer_name}' - UI will not update!")
    
    def set_layer_change_callback(self, callback):
        """Set callback function to be called when layers change."""
        self.layer_change_callback = callback
    
    def set_layer_visibility(self, layer_name: str, visible: bool) -> None:
        """
        Toggle visibility of any layer.
        
        Args:
            layer_name: Name of the layer
            visible: Whether the layer should be visible
        """
        if layer_name not in self.active_layers:
            logger.warning(f"Unknown layer: {layer_name}")
            return
        
        layer = self.active_layers[layer_name]
        layer['visible'] = visible
        
        # Special handling for drillholes: there can be MANY individual actors per hole,
        # plus collar markers. The "drillholes" layer only tracks a representative actor,
        # so we need to propagate visibility to all stored actors.
        if layer_name == "drillholes":
            # Toggle representative actor
            if layer['actor'] is not None and self.plotter is not None:
                if visible:
                    layer['actor'].VisibilityOn()
                else:
                    layer['actor'].VisibilityOff()
            
            # Toggle per-hole tube actors
            for actor in getattr(self, "_drillhole_hole_actors", {}).values():
                try:
                    if visible:
                        actor.VisibilityOn()
                    else:
                        actor.VisibilityOff()
                except Exception:
                    continue
            
            # Toggle collar markers (handles merged mode)
            collar_actors = getattr(self, "_drillhole_collar_actors", {})
            for key, actor in collar_actors.items():
                if key == "_merged_hids":
                    continue  # Skip metadata key (list, not actor)
                try:
                    if visible:
                        actor.VisibilityOn()
                    else:
                        actor.VisibilityOff()
                except Exception:
                    continue
        else:
            # Generic case: single actor per layer
            if layer['actor'] is not None and self.plotter is not None:
                if visible:
                    layer['actor'].VisibilityOn()
                else:
                    layer['actor'].VisibilityOff()
            # Defer render - PyVista will render on next frame automatically
            # This improves performance during layer visibility changes
        
        logger.debug(f"Set {layer_name} visibility: {visible}")
    
    def set_layer_opacity(self, layer_name: str, opacity: float) -> None:
        """
        Set opacity for a specific layer.
        
        Args:
            layer_name: Name of the layer
            opacity: Opacity value (0.0 = transparent, 1.0 = opaque)
        """
        if layer_name not in self.active_layers:
            logger.warning(f"Unknown layer: {layer_name}")
            return
        
        layer = self.active_layers[layer_name]
        layer['opacity'] = opacity
        
        if layer['actor'] is not None:
            layer['actor'].GetProperty().SetOpacity(opacity)
            # Defer render - PyVista will render on next frame automatically
            # This improves performance during layer opacity changes
        
        logger.info(f"Set {layer_name} opacity: {opacity}")
    
    def set_active_layer_for_controls(self, layer_name: str):
        """
        Set the active layer for property controls.
        This makes the scalar bar visible ONLY for this layer.
        
        Args:
            layer_name: Name of the layer to make active
        """
        if layer_name not in self.active_layers:
            logger.warning(f"Unknown layer: {layer_name}")
            return
        
        # Remove all existing scalar bars
        try:
            self.plotter.remove_scalar_bar()
        except (AttributeError, RuntimeError, StopIteration) as e:
            logger.debug(f"No scalar bars to remove: {e}")
        
        layer = self.active_layers[layer_name]
        layer_data = layer.get('data')
        
        if layer_data is None:
            logger.warning(f"Layer '{layer_name}' has no data")
            return
        
        # Extract mesh from drillhole data dictionary if needed
        actual_mesh = layer_data
        if isinstance(layer_data, dict) and 'mesh' in layer_data:
            actual_mesh = layer_data.get('mesh')
            if actual_mesh is None:
                logger.warning(f"Layer '{layer_name}' has no mesh in data dictionary")
                return
        
        # Get the active scalars for this layer
        active_scalars = None
        if hasattr(actual_mesh, 'active_scalars_name'):
            active_scalars = actual_mesh.active_scalars_name
        
        # Find any scalar field to display
        if active_scalars is None:
            if hasattr(actual_mesh, 'cell_data') and len(actual_mesh.cell_data.keys()) > 0:
                active_scalars = list(actual_mesh.cell_data.keys())[0]
            elif hasattr(actual_mesh, 'point_data') and len(actual_mesh.point_data.keys()) > 0:
                active_scalars = list(actual_mesh.point_data.keys())[0]
            elif hasattr(actual_mesh, 'array_names') and len(actual_mesh.array_names) > 0:
                active_scalars = actual_mesh.array_names[0]
        
        if active_scalars:
            # DISABLED: PyVista scalar bars - use custom LegendManager instead
            # The LegendManager will handle legend display via the custom LegendWidget
            logger.info(f"Set active layer for controls: {layer_name} (scalar: {active_scalars}, using custom LegendManager)")
    
    def set_lod_quality(self, quality: float) -> None:
        """
        Set LOD quality setting (0.0 = low quality/aggressive LOD, 1.0 = high quality/minimal LOD).
        
        Args:
            quality: Quality value in [0.0, 1.0]
        """
        self.lod_config['lod_quality'] = max(0.0, min(1.0, quality))
        if self.lod_manager is not None:
            self.lod_manager.set_quality(self.lod_config['lod_quality'])
        logger.info(f"Set LOD quality to {self.lod_config['lod_quality']}")
    
    def set_sampling_factor(self, factor: int, enabled: bool = True) -> None:
        """
        Set block model sampling factor.
        
        Args:
            factor: Sampling factor (1 = no sampling, 2 = every 2nd block, etc.)
            enabled: Whether sampling is enabled
        """
        self.sampling_factor = max(1, factor)
        self.sampling_enabled = enabled
        logger.info(f"Set sampling factor to {self.sampling_factor} (enabled={enabled})")
        
        # Regenerate meshes if model is loaded
        if self.current_model is not None:
            self._generate_block_meshes()
            self._add_meshes_to_plotter()
    
    def set_max_blocks_render(self, max_blocks: int) -> None:
        """
        Set maximum number of blocks to render.
        
        Args:
            max_blocks: Maximum blocks to render (models exceeding this will be sampled)
        """
        self.max_blocks_render = max(1000, max_blocks)  # Minimum 1000 blocks
        logger.info(f"Set max blocks render to {self.max_blocks_render}")
        
        # Regenerate meshes if model is loaded and exceeds limit
        if self.current_model is not None and self.current_model.block_count > self.max_blocks_render:
            self._generate_block_meshes()
            self._add_meshes_to_plotter()
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """
        Get current performance settings.
        
        Returns:
            Dict with lod_quality, sampling_factor, sampling_enabled, max_blocks_render
        """
        return {
            'lod_quality': self.lod_config.get('lod_quality', 0.7),
            'sampling_factor': self.sampling_factor,
            'sampling_enabled': self.sampling_enabled,
            'max_blocks_render': self.max_blocks_render
        }
    
    def update_layer_property(self, layer_name: str, property_name: str, colormap: str = 'viridis', 
                              color_mode: str = 'continuous', custom_colors: Optional[Dict[Any, str]] = None) -> None:
        """
        Update the displayed property and colormap for a specific layer.
        
        Args:
            layer_name: Name of the layer to update
            property_name: Property/scalar name to visualize
            colormap: Colormap to use
            color_mode: 'continuous' or 'discrete'
            custom_colors: Optional dictionary mapping category values to color strings (hex or name)
                          Used for discrete mode to override colormap colors
        """
        if layer_name not in self.active_layers:
            logger.warning(f"Unknown layer: {layer_name}")
            return

        layer = self.active_layers[layer_name]
        layer_data = layer.get('data')
        layer_actor = layer.get('actor')

        # Handle drillhole layers specially BEFORE the actor null check
        # Drillholes don't have a single 'actor' - they have multiple actors in _drillhole_hole_actors
        if layer_name == "drillholes" or (isinstance(layer_data, dict) and 'database' in layer_data):
            self._update_drillhole_colors(property_name, colormap, color_mode, custom_colors)
            return

        if layer_data is None or layer_actor is None:
            logger.warning(f"Layer '{layer_name}' has no data or actor")
            return

        # Extract mesh from data dictionary if needed (MUST be before classification check!)
        actual_mesh = layer_data
        if isinstance(layer_data, dict) and 'mesh' in layer_data:
            actual_mesh = layer_data.get('mesh')
            if actual_mesh is None:
                logger.warning(f"Layer '{layer_name}' has no mesh in data dictionary")
                return

        # Handle classification layers specially
        # Note: add_layer stores as 'type', but some code uses 'layer_type' - check both
        layer_type = layer.get('layer_type', layer.get('type', ''))
        if layer_type == 'classification' and property_name in ('Classification', 'Category', 'CLASS'):
            self._update_classification_colors(layer_name, layer, actual_mesh, colormap, color_mode, custom_colors)
            return
        
        # Check if property exists in the data
        has_property = False
        if hasattr(actual_mesh, 'array_names') and property_name in actual_mesh.array_names:
            has_property = True
        elif hasattr(actual_mesh, 'cell_data') and property_name in actual_mesh.cell_data:
            has_property = True
        elif hasattr(actual_mesh, 'point_data') and property_name in actual_mesh.point_data:
            has_property = True
        
        if not has_property:
            logger.warning(f"Property '{property_name}' not found in layer '{layer_name}'")
            return
        
        # OPTIMIZATION: Try to update existing actor instead of removing/re-adding
        # This prevents double render and lag
        opacity = layer.get('opacity', 1.0)
        is_discrete = (color_mode == 'discrete')
        
        # Try to update existing actor's mapper (only for Block Model layers with UnstructuredGrid)
        if layer_name.startswith("Block Model") and layer_actor is not None and hasattr(actual_mesh, 'cell_data'):
            if property_name in actual_mesh.cell_data:
                try:
                    # Update existing mesh actor's mapper instead of removing/re-adding
                    mapper = layer_actor.GetMapper()
                    if mapper:
                        # Update scalar array
                        mapper.SelectColorArray(property_name)
                        mapper.SetScalarModeToUseCellData()
                        mapper.SetScalarVisibility(1)
                        
                        # Update colormap via lookup table
                        lut = mapper.GetLookupTable()
                        if lut is None:
                            import pyvista as pv
                            lut = pv._vtk.vtkLookupTable()
                            mapper.SetLookupTable(lut)
                        
                        # Get scalar range from the property
                        prop_values = actual_mesh.cell_data[property_name]
                        scalar_range = (float(np.nanmin(prop_values)), float(np.nanmax(prop_values)))
                        lut.SetRange(scalar_range)
                        lut.SetTableRange(scalar_range)
                        
                        # Apply colormap using matplotlib directly
                        try:
                            import matplotlib.cm as cm
                            cmap_matplotlib = cm.get_cmap(colormap)
                            lut.SetNumberOfTableValues(256)
                            lut.Build()
                            for i in range(256):
                                rgba = cmap_matplotlib(i / 255.0)
                                lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], rgba[3] if len(rgba) > 3 else 1.0)
                        except Exception as e:
                            logger.debug(f"Could not apply colormap '{colormap}' to LUT: {e}")
                            # Fallback to viridis
                            try:
                                import matplotlib.cm as cm
                                cmap_matplotlib = cm.get_cmap('viridis')
                                lut.SetNumberOfTableValues(256)
                                lut.Build()
                                for i in range(256):
                                    rgba = cmap_matplotlib(i / 255.0)
                                    lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], rgba[3] if len(rgba) > 3 else 1.0)
                            except Exception:
                                pass
                        
                        # Force mapper and actor update
                        mapper.Modified()
                        layer_actor.Modified()
                        
                        # Update opacity and lighting properties if changed
                        prop = layer_actor.GetProperty()
                        if prop is not None:
                            prop.SetOpacity(opacity)
                            # Ensure balanced lighting for better color perception
                            prop.SetAmbient(0.6)  # Reduced ambient for better color balance
                            prop.SetDiffuse(0.4)  # Add diffuse for depth perception
                            prop.SetSpecular(0.1)  # Small specular for subtle highlights
                            prop.SetSpecularPower(15)  # Control specular sharpness
                        
                        # Trigger render update
                        if hasattr(self.plotter, 'render'):
                            try:
                                self.plotter.render()
                            except Exception:
                                pass
                        
                        logger.info(f"Updated existing layer actor '{layer_name}' with property '{property_name}' and colormap '{colormap}' (no remove/re-add)")
                        
                        # Update layer metadata
                        layer['current_property'] = property_name
                        layer['current_colormap'] = colormap
                        
                        # Update scalar bar for active layer
                        self.set_active_layer_for_controls(layer_name)
                        
                        # Update custom legend widget via LegendManager
                        if hasattr(self, 'legend_manager') and self.legend_manager is not None:
                            try:
                                prop_array = np.asarray(prop_values)
                                if is_discrete:
                                    unique_values = np.unique(prop_array[~np.isnan(prop_array)])
                                    unique_values = unique_values[unique_values != 0]
                                    unique_values = sorted(unique_values)  # Returns a list
                                    if len(unique_values) <= 100:
                                        self.legend_manager.update_discrete(
                                            property_name=property_name,
                                            categories=list(unique_values),  # Ensure it's a list
                                            cmap_name=colormap
                                        )
                                else:
                                    finite_data = prop_array[np.isfinite(prop_array)]
                                    if len(finite_data) > 0:
                                        self.legend_manager.update_continuous(
                                            property_name=property_name,
                                            data=prop_array,
                                            cmap_name=colormap
                                        )
                            except Exception as e:
                                logger.debug(f"Could not update legend manager: {e}")
                        
                        # Update current property for tracking
                        self.current_property = property_name
                        
                        logger.info(f"Updated layer '{layer_name}' to show property '{property_name}' with colormap '{colormap}'")
                        return  # Successfully updated, no need to remove/re-add
                except Exception as e:
                    logger.warning(f"Could not update existing layer actor, falling back to remove/re-add: {e}")
        
        # Fallback: Remove old actor and re-add (only if update failed)
        if self.plotter is not None:
            self.plotter.remove_actor(layer_actor)
        
        # Handle custom colors for discrete mode
        actual_colormap = colormap
        # Track category colors for legend sync (used later)
        _category_colors_for_legend = None  
        if is_discrete and custom_colors is not None and len(custom_colors) > 0:
            try:
                # Extract property values to determine unique categories
                prop_values = None
                if hasattr(actual_mesh, 'cell_data') and property_name in actual_mesh.cell_data:
                    prop_values = actual_mesh.cell_data[property_name]
                elif hasattr(actual_mesh, 'point_data') and property_name in actual_mesh.point_data:
                    prop_values = actual_mesh.point_data[property_name]
                elif hasattr(actual_mesh, 'array_names') and property_name in actual_mesh.array_names:
                    try:
                        prop_values = actual_mesh[property_name]
                    except (KeyError, AttributeError) as e:
                        logger.debug(f"Property extraction failed for {property_name}: {e}")
                
                if prop_values is not None:
                    # Get unique values and create custom colormap
                    unique_values = np.unique(np.asarray(prop_values)[~np.isnan(np.asarray(prop_values))])
                    unique_values = unique_values[unique_values != 0]
                    unique_values = sorted(unique_values)
                    
                    # Create color list for ListedColormap
                    from matplotlib import cm
                    import matplotlib.colors as mcolors
                    
                    color_list = []
                    _category_colors_for_legend = {}  # Track RGBA for legend
                    for val in unique_values:
                        if val in custom_colors:
                            # Use custom color (convert hex to RGB if needed)
                            color_str = custom_colors[val]
                            if color_str.startswith('#'):
                                # Hex color
                                color_str = color_str.lstrip('#')
                                rgb = tuple(int(color_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                                rgba = rgb + (1.0,)
                                color_list.append(rgba)
                                _category_colors_for_legend[val] = rgba
                            else:
                                # Named color - try to get RGB
                                try:
                                    from matplotlib.colors import to_rgba
                                    rgba = to_rgba(color_str)
                                    color_list.append(rgba)
                                    _category_colors_for_legend[val] = rgba
                                except (ValueError, KeyError, TypeError) as e:
                                    # Fallback to gray for invalid color names
                                    logger.debug(f"Color conversion failed for '{color_str}': {e}, using gray")
                                    color_list.append((0.5, 0.5, 0.5, 1.0))
                                    _category_colors_for_legend[val] = (0.5, 0.5, 0.5, 1.0)
                        else:
                            # Use default colormap for values without custom colors
                            try:
                                default_cmap = cm.get_cmap(colormap)
                                idx = list(unique_values).index(val)
                                norm_idx = idx / max(1, len(unique_values) - 1)
                                rgba = default_cmap(norm_idx)
                                color_list.append(rgba)
                                _category_colors_for_legend[val] = tuple(rgba)
                            except (ValueError, KeyError) as e:
                                logger.debug(f"Colormap lookup failed for value {val}: {e}, using gray")
                                color_list.append((0.5, 0.5, 0.5, 1.0))
                                _category_colors_for_legend[val] = (0.5, 0.5, 0.5, 1.0)
                    
                    if len(color_list) > 0:
                        # Create custom ListedColormap
                        custom_cmap = mcolors.ListedColormap(color_list, name=f"{property_name}_custom")
                        actual_colormap = custom_cmap
                        logger.info(f"Applied custom colors for {len(custom_colors)} categories")
            except Exception as e:
                logger.warning(f"Failed to apply custom colors, using default colormap: {e}", exc_info=True)
                actual_colormap = colormap
                _category_colors_for_legend = None
        
        # Add mesh with new settings (NO scalar bar - will be managed separately)
        new_actor = self.plotter.add_mesh(
            actual_mesh,
            scalars=property_name,
            cmap=actual_colormap,
            show_edges=False,
            opacity=opacity,
            lighting=True,  # Enable lighting for better color balance
            ambient=0.6,  # Reduced ambient for better color balance
            diffuse=0.4,  # Add diffuse for depth perception
            specular=0.1,  # Small specular for subtle highlights
            show_scalar_bar=False,  # DON'T show scalar bar here
            n_colors=256 if not is_discrete else 20,
            clim=None  # Auto-scale
        )
        
        # Update layer with new actor
        layer['actor'] = new_actor
        
        # Set visibility based on previous state
        if not layer.get('visible', True):
            new_actor.VisibilityOff()
        
        # Ensure balanced lighting properties at VTK level
        if new_actor:
            prop = new_actor.GetProperty()
            if prop is not None:
                prop.SetInterpolationToFlat()  # Keep flat shading
                prop.SetAmbient(0.6)  # Reduced ambient for better color balance
                prop.SetDiffuse(0.4)  # Add diffuse for depth perception
                prop.SetSpecular(0.1)  # Small specular for subtle highlights
                prop.SetSpecularPower(15)  # Control specular sharpness
        
        # Update scalar bar for active layer
        self.set_active_layer_for_controls(layer_name)
        
        # Update custom legend widget via LegendManager
        if hasattr(self, 'legend_manager') and self.legend_manager is not None:
            try:
                # Extract property values from the mesh
                prop_values = None
                if hasattr(actual_mesh, 'cell_data') and property_name in actual_mesh.cell_data:
                    prop_values = actual_mesh.cell_data[property_name]
                elif hasattr(actual_mesh, 'point_data') and property_name in actual_mesh.point_data:
                    prop_values = actual_mesh.point_data[property_name]
                elif hasattr(actual_mesh, 'array_names') and property_name in actual_mesh.array_names:
                    # Try to get as array
                    try:
                        prop_values = actual_mesh[property_name]
                    except (KeyError, AttributeError) as e:
                        logger.debug(f"Property array extraction failed for {property_name}: {e}")
                
                if prop_values is not None:
                    prop_array = np.asarray(prop_values)
                    
                    if is_discrete:
                        # Update discrete legend
                        unique_values = np.unique(prop_array[~np.isnan(prop_array)])
                        unique_values = unique_values[unique_values != 0]
                        unique_values = sorted(unique_values)  # Returns a list
                        if len(unique_values) <= 100:  # Only if reasonable number of categories
                            # Use the same colormap that was used for rendering
                            # actual_colormap is defined earlier in the function scope
                            # Pass category_colors when custom colors were defined
                            self.legend_manager.update_discrete(
                                property_name=property_name,
                                categories=list(unique_values),  # Ensure it's a list
                                cmap_name=actual_colormap,
                                category_colors=_category_colors_for_legend  # Pass custom colors to legend
                            )
                    else:
                        # Update continuous legend - use the same colormap as rendering
                        finite_data = prop_array[np.isfinite(prop_array)]
                        if len(finite_data) > 0:
                            # Always use the colormap string name for the legend (not the colormap object)
                            # This ensures the legend uses the same colormap name as the property panel
                            # The actual_colormap might be a custom ListedColormap object, but we want the string name
                            legend_cmap = colormap  # Always use the string colormap name
                            logger.info(f"Updating legend with colormap='{colormap}' for property '{property_name}'")
                            self.legend_manager.update_continuous(
                                property_name=property_name,
                                data=prop_array,
                                cmap_name=legend_cmap
                            )
            except Exception as e:
                logger.warning(f"Failed to update legend in update_layer_property: {e}", exc_info=True)
        
        # Update current property for tracking
        self.current_property = property_name
        
        # CRITICAL FIX: Force render to display changes on screen
        # Without this, changes to block models and geology layers won't be visible
        if self.plotter is not None:
            try:
                self.plotter.render()
                logger.debug(f"Rendered scene after updating layer '{layer_name}'")
            except Exception as e:
                logger.warning(f"Failed to render after layer update: {e}")
        
        logger.info(f"Updated layer '{layer_name}' to show property '{property_name}' with colormap '{colormap}'")

    def _update_classification_colors(self, layer_name: str, layer: Dict, mesh, colormap: str,
                                    color_mode: str, custom_colors: Optional[Dict[Any, str]] = None) -> None:
        """
        Update colors for classification layers, handling custom color definitions.

        Args:
            layer_name: Name of the classification layer
            layer: Layer information dictionary
            mesh: PyVista mesh with classification data
            colormap: Colormap name (ignored for classification with custom colors)
            color_mode: Color mode (should be 'discrete' for classification)
            custom_colors: Dictionary mapping category names to colors
        """
        try:
            from ..models.jorc_classification_engine import CLASSIFICATION_COLORS, CLASSIFICATION_ORDER

            # Get the actor for this layer
            actor = layer.get('actor')
            if actor is None:
                logger.warning(f"No actor found for classification layer '{layer_name}'")
                return

            # Check for categorical data - try multiple field names
            # The visualization creates "Category" field with string names
            categories = None
            if "Category" in mesh.array_names:
                categories = mesh["Category"]
            elif "Classification_Categories" in mesh.array_names:
                categories = mesh["Classification_Categories"]

            # Get numeric scalars (Classification = 0,1,2,3)
            scalars = None
            if "Classification" in mesh.array_names:
                scalars = mesh["Classification"]
            elif "CLASS" in mesh.array_names:
                scalars = mesh["CLASS"]

            if scalars is None:
                logger.warning(f"No classification scalars found in layer '{layer_name}'")
                return

            # Classification mapping: Measured=0, Indicated=1, Inferred=2, Unclassified=3
            # This MUST match the order in _visualize_results in jorc_classification_panel.py
            category_order = ["Measured", "Indicated", "Inferred", "Unclassified"]

            # Build colormap in correct order (index 0=Measured, 1=Indicated, 2=Inferred, 3=Unclassified)
            def hex_to_rgb(hex_color):
                return [
                    int(hex_color[1:3], 16) / 255.0,
                    int(hex_color[3:5], 16) / 255.0,
                    int(hex_color[5:7], 16) / 255.0,
                    1.0
                ]

            # IMPORTANT: Merge new custom colors with existing ones stored in the layer
            # This prevents other colors from resetting when only one category is changed
            existing_colors = layer.get('classification_colors', {})
            if custom_colors:
                # Merge: new colors override existing ones
                merged_colors = {**existing_colors, **custom_colors}
            else:
                merged_colors = existing_colors

            # Store merged colors back in layer for future updates
            layer['classification_colors'] = merged_colors

            # Build colormap - use merged custom colors if provided, otherwise use defaults
            # PyVista cmap expects a list of color STRINGS (hex codes), not RGB tuples
            classification_cmap = []
            category_colors = {}  # For legend

            for cat in category_order:
                if merged_colors and cat in merged_colors:
                    color_hex = merged_colors[cat]
                    if not color_hex.startswith('#'):
                        color_hex = CLASSIFICATION_COLORS.get(cat, "#7f7f7f")
                else:
                    color_hex = CLASSIFICATION_COLORS[cat]

                # Store hex string for PyVista cmap (requires strings, not RGB tuples)
                classification_cmap.append(color_hex)
                # Store RGB tuple for legend
                rgb = hex_to_rgb(color_hex)
                category_colors[cat] = tuple(rgb)

            try:
                # Remove and re-add the actor with correct colormap
                self.plotter.remove_actor(actor)

                # DEBUG: Log actual scalar values in the mesh
                if "Classification" in mesh.array_names:
                    scalars = mesh["Classification"]
                    unique_vals, counts = np.unique(scalars, return_counts=True)
                    logger.info(f"[DEBUG] Mesh scalar 'Classification' unique values: {unique_vals.tolist()}")
                    logger.info(f"[DEBUG] Mesh scalar 'Classification' value counts: {dict(zip(unique_vals, counts))}")

                new_actor = self.plotter.add_mesh(
                    mesh,
                    name=layer_name,
                    scalars="Classification",
                    cmap=classification_cmap,
                    n_colors=4,  # CRITICAL: Force 4 discrete color bins (no interpolation)
                    clim=[-0.5, 3.5],  # Centered bins for 4 integer categories (0,1,2,3)
                    show_edges=True,
                    edge_color='black',
                    line_width=0.5,
                    opacity=layer.get('opacity', 1.0),
                    show_scalar_bar=False,
                    pickable=True,
                    lighting=True,
                )

                # Update the layer info with new actor
                layer['actor'] = new_actor

                logger.info(f"Updated classification layer '{layer_name}' with {'custom' if custom_colors else 'default'} colors")

            except Exception as e:
                logger.warning(f"Failed to update classification layer colors: {e}")
                return

            # Update legend with categorical names (not numbers!)
            if hasattr(self, 'legend_manager') and self.legend_manager is not None:
                try:
                    self.legend_manager.update_discrete(
                        property_name='Classification',
                        categories=category_order,
                        category_colors=category_colors,
                        subtitle="Resource Classification"
                    )
                    logger.info("Updated classification legend with category names")
                except Exception as e:
                    logger.warning(f"Could not update classification legend: {e}")

            # Force render to display the color changes
            if self.plotter is not None:
                try:
                    self.plotter.render()
                    logger.debug(f"Rendered classification layer '{layer_name}' after color update")
                except Exception as e:
                    logger.warning(f"Render failed after classification color update: {e}")

        except Exception as e:
            logger.error(f"Error updating classification colors for layer '{layer_name}': {e}", exc_info=True)

    def add_kriging_layer(self, grid, variable_name: str, scalars_name: str = None,
                          layer_name: str = None) -> None:
        """
        Add or update kriging results layer.

        Args:
            grid: PyVista grid with kriging estimates
            variable_name: Name of the variable (e.g., 'ZN')
            scalars_name: Name of the scalars to display (defaults to variable_name + '_estimate')
            layer_name: Optional custom layer name (defaults to 'Kriging: {variable_name}')
        """
        if scalars_name is None:
            scalars_name = f"{variable_name}_estimate"

        if layer_name is None:
            layer_name = f"Kriging: {variable_name}"

        # DIAGNOSTIC: Log input grid type
        logger.info(f"[KRIGING MESH] Input grid type: {type(grid).__name__}, n_cells: {grid.n_cells}")

        # CRITICAL FIX: Do NOT use threshold() - it converts to expensive UnstructuredGrid
        # For regular kriging grids (ImageData/RectilinearGrid), use the grid directly
        # Only threshold if we need to filter NaN values, but use a copy to preserve type
        if isinstance(grid, pv.UnstructuredGrid):
            # Already UnstructuredGrid - use threshold to filter
            render_grid = grid.threshold(value=(-np.inf, np.inf), scalars=scalars_name)
            show_edges = False  # NEVER show edges for UnstructuredGrid - extremely expensive
            logger.warning(f"[KRIGING MESH] Using UnstructuredGrid ({grid.n_cells} cells) - edges disabled for performance")
        else:
            # Regular grid (ImageData/RectilinearGrid) - use directly for best performance
            render_grid = grid
            show_edges = False  # Disable edges for all kriging layers
            logger.info(f"[KRIGING MESH] Using {type(grid).__name__} - optimal for regular grids")

        logger.info(f"[KRIGING MESH] Render grid type: {type(render_grid).__name__}, n_cells: {render_grid.n_cells}")

        actor = self.plotter.add_mesh(
            render_grid,
            scalars=scalars_name,
            cmap=self.current_colormap,
            opacity=self.default_opacity['volume'],
            show_edges=show_edges,  # Explicitly disable edges
            show_scalar_bar=False,  # Scalar bar managed separately
            name=layer_name
        )

        # Register layer using generic method
        self.add_layer(layer_name, actor, grid, layer_type='volume')

        # Register in scene_layers for global picking
        self.register_scene_layer(layer_name, actor, render_grid, 'volume')

        logger.info(f"Added kriging layer for {variable_name} (type={type(render_grid).__name__}, edges={show_edges})")
    
    def add_block_model_layer(self, block_grid, property_name: str, layer_name: str = None,
                               colormap: str = None) -> None:
        """
        Add or update block model layer.

        Args:
            block_grid: PyVista RectilinearGrid with block model
            property_name: Property to visualize
            layer_name: Optional custom layer name (defaults to 'Block Model: {property_name}')
            colormap: Optional colormap name. If None, auto-selects based on data type:
                      - 'tab20' for categorical data (Formation, Lithology, etc.)
                      - current_colormap (viridis) for continuous data
        """
        if layer_name is None:
            layer_name = f"Block Model: {property_name}"

        # Log ORIGINAL grid bounds for comparison with drillholes
        grid_bounds = block_grid.bounds
        logger.info(
            f"[GRID RENDER] Adding grid layer '{layer_name}': ORIGINAL bounds=({grid_bounds[0]:.2f}, {grid_bounds[1]:.2f}, "
            f"{grid_bounds[2]:.2f}, {grid_bounds[3]:.2f}, {grid_bounds[4]:.2f}, {grid_bounds[5]:.2f})"
        )

        # ================================================================
        # CRITICAL FIX: Apply coordinate shift to block model grid
        # ================================================================
        # Block models must use the SAME coordinate shift as drillholes and geology
        # to render together in the same scene. Without this, block models remain
        # at UTM coordinates (e.g., 500,000m) while other layers are shifted to local
        # coordinates (~0,0,0), causing block models to disappear from view.
        # ================================================================
        try:
            # Check if this grid has already been shifted (prevent double-shifting)
            already_shifted = getattr(block_grid, '_coordinate_shifted', False)

            if not already_shifted:
                # Get a representative point to lock/use the global shift
                center_point = np.array([
                    (grid_bounds[0] + grid_bounds[1]) / 2,
                    (grid_bounds[2] + grid_bounds[3]) / 2,
                    (grid_bounds[4] + grid_bounds[5]) / 2
                ]).reshape(1, 3)

                # Lock global shift if not already set, or use existing shift
                _ = self._to_local_precision(center_point)

                # Apply shift to the grid based on grid type
                if self._global_shift is not None:
                    shift = self._global_shift

                    if isinstance(block_grid, pv.RectilinearGrid):
                        # RectilinearGrid: shift the x, y, z edge arrays
                        block_grid.x = block_grid.x - shift[0]
                        block_grid.y = block_grid.y - shift[1]
                        block_grid.z = block_grid.z - shift[2]
                        logger.info(f"[GRID RENDER] Applied coordinate shift to RectilinearGrid: shift=[{shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f}]")

                    elif isinstance(block_grid, pv.ImageData):
                        # ImageData: shift the origin
                        origin = np.array(block_grid.origin)
                        block_grid.origin = tuple(origin - shift)
                        logger.info(f"[GRID RENDER] Applied coordinate shift to ImageData: shift=[{shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f}]")

                    elif hasattr(block_grid, 'points'):
                        # StructuredGrid/UnstructuredGrid: shift points directly
                        block_grid.points = block_grid.points - shift
                        logger.info(f"[GRID RENDER] Applied coordinate shift to grid points: shift=[{shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f}]")

                    # Mark grid as shifted to prevent double-shifting
                    block_grid._coordinate_shifted = True

                    # Log shifted bounds
                    shifted_bounds = block_grid.bounds
                    logger.info(
                        f"[GRID RENDER] SHIFTED bounds=({shifted_bounds[0]:.2f}, {shifted_bounds[1]:.2f}, "
                        f"{shifted_bounds[2]:.2f}, {shifted_bounds[3]:.2f}, {shifted_bounds[4]:.2f}, {shifted_bounds[5]:.2f})"
                    )
            else:
                logger.debug(f"[GRID RENDER] Grid already shifted, skipping coordinate transform")
        except Exception as e:
            logger.warning(f"[GRID RENDER] Failed to apply coordinate shift (non-fatal): {e}")
        
        # Determine data location (cell_data vs point_data)
        # StructuredGrid from simulation may store data in point_data
        data_preference = 'cell'
        if hasattr(block_grid, 'cell_data') and property_name in block_grid.cell_data:
            data_preference = 'cell'
        elif hasattr(block_grid, 'point_data') and property_name in block_grid.point_data:
            data_preference = 'point'
            logger.info(f"[GRID RENDER] Property '{property_name}' found in point_data, using point preference")
        
        # Auto-select colormap for categorical vs continuous data
        if colormap is None:
            # Detect categorical properties by name or data characteristics
            categorical_props = {'formation', 'lithology', 'lith', 'unit', 'domain', 'zone', 'classification'}
            prop_lower = property_name.lower()
            is_categorical = any(cat in prop_lower for cat in categorical_props)
            
            # Also check if data is integer with limited unique values
            if not is_categorical:
                try:
                    if data_preference == 'cell' and property_name in block_grid.cell_data:
                        data = block_grid.cell_data[property_name]
                    elif data_preference == 'point' and property_name in block_grid.point_data:
                        data = block_grid.point_data[property_name]
                    else:
                        data = None
                    
                    if data is not None:
                        # Integer data with <= 20 unique values is likely categorical
                        if np.issubdtype(data.dtype, np.integer):
                            n_unique = len(np.unique(data[~np.isnan(data.astype(float))]))
                            is_categorical = n_unique <= 20
                            logger.debug(f"Auto-detected categorical: {n_unique} unique integer values")
                except Exception:
                    pass
            
            colormap = 'tab20' if is_categorical else self.current_colormap
            logger.info(f"[GRID RENDER] Using colormap '{colormap}' for property '{property_name}' (categorical={is_categorical})")
        
        # DIAGNOSTIC: Log mesh details before adding
        n_cells = block_grid.n_cells if hasattr(block_grid, 'n_cells') else 0
        n_points = block_grid.n_points if hasattr(block_grid, 'n_points') else 0
        logger.info(f"[GRID RENDER] Mesh stats: {n_cells} cells, {n_points} points")

        # Check scalar data validity
        if data_preference == 'cell' and property_name in block_grid.cell_data:
            scalar_data = block_grid.cell_data[property_name]
            logger.info(f"[GRID RENDER] Scalars: min={np.nanmin(scalar_data)}, max={np.nanmax(scalar_data)}, "
                       f"unique={len(np.unique(scalar_data[~np.isnan(scalar_data.astype(float))]))}")

        # GPU-safe mode: Use single render for large models to prevent driver timeout (TDR)
        LARGE_MODEL_THRESHOLD = 25000  # GPU-safe mode above this (single render, no forced updates)
        is_large_model = n_cells > LARGE_MODEL_THRESHOLD

        if is_large_model:
            logger.warning(f"[GRID RENDER] Large model detected ({n_cells:,} cells) - using GPU-safe rendering mode")
            self._has_large_model = True  # Flag to disable aggressive renders globally

            # Activate the resize debounce filter (installed in initialize_plotter).
            # This monkey-patches the QtInteractor's resizeEvent to skip VTK Render()
            # during resize and fire a single deferred render after the user stops dragging.
            if self._resize_debounce is not None:
                self._resize_debounce.activate()
            else:
                logger.warning("[GRID RENDER] Resize debounce filter not installed — "
                               "panel resize may cause UI freeze with large models")

        # CRITICAL: Disable edges for models >50k cells
        # Edge rendering requires recalculating millions of wireframe vertices during resize
        # This is the primary cause of GPU TDR (Timeout Detection and Recovery) freezes
        render_edges = n_cells < 50000
        if n_cells >= 50000:
            logger.warning(f"[GRID RENDER] Disabling edges for large model ({n_cells:,} cells) to prevent GPU timeout")

        # Add new block model layer
        actor = self.plotter.add_mesh(
            block_grid,
            scalars=property_name,
            preference=data_preference,  # Dynamically select based on where data is stored
            cmap=colormap,
            opacity=self.default_opacity['blocks'],
            show_edges=render_edges,  # Dynamic based on model size
            edge_color='black',  # Black edges for better color balance
            line_width=1.0,
            show_scalar_bar=False,  # Scalar bar managed separately
            name=layer_name,
            render=not is_large_model,  # Defer rendering for large models
        )
        
        # DIAGNOSTIC: Verify actor was created
        if actor is None:
            logger.error(f"[GRID RENDER] FAILED: add_mesh returned None for '{layer_name}'!")
        else:
            # Check actor visibility
            try:
                visibility = actor.GetVisibility() if hasattr(actor, 'GetVisibility') else 'unknown'
                logger.info(f"[GRID RENDER] Actor created: visibility={visibility}")
            except Exception as e:
                logger.debug(f"Could not check actor visibility: {e}")
        
        # Register layer using generic method
        self.add_layer(layer_name, actor, block_grid, layer_type='blocks')
        self.current_property = property_name

        # Register in scene_layers for global picking
        self.register_scene_layer(layer_name, actor, block_grid, 'blocks')

        logger.info(f"Added block model layer: {property_name}")

        # GPU-SAFE RENDERING: Use single render for large models to prevent driver timeout (TDR)
        # The aggressive force_render_update() calls Render() 5+ times which can timeout on large meshes
        if is_large_model:
            logger.info(f"[GRID RENDER] Using single render pass for large model to prevent GPU timeout")
            try:
                self.plotter.render()  # Single render call
            except Exception as e:
                logger.warning(f"[GRID RENDER] Render warning (non-fatal): {e}")
        else:
            # CRITICAL FIX: Force synchronous render update to fix "invisible until resize" bug
            # This properly updates VTK render window, Qt widget, and processes events
            self.force_render_update()
    
    def add_geology_surface_layer(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        domain: str,
        color: Optional[Tuple[float, float, float]] = None,
        opacity: float = 1.0,
        layer_name: Optional[str] = None,
        show_edges: bool = False,
        ghost_mode: bool = False,
    ) -> None:
        """
        Add or update a geological surface layer (wireframe/isosurface).
        
        Creates a PyVista mesh from vertices and faces and adds it to the scene
        as a geological surface layer. Follows the same pattern as add_kriging_layer
        and add_block_model_layer for consistency.
        
        RENDERING RULE (Leapfrog-style):
        - Default opacity is 1.0 (fully opaque) to prevent overlapping visual chaos
        - Semi-transparency (ghost_mode) must be explicitly requested
        - Surfaces are extracted from mutually-exclusive voxel assignments,
          so they should NOT overlap geometrically by construction
        
        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of triangle face indices (0-based)
            domain: Name of the geological domain (e.g., 'ORE', 'WASTE')
            color: Optional RGB tuple (0-1 range). If None, uses deterministic color from domain name.
            opacity: Surface opacity (default 1.0 for OPAQUE surfaces - not semi-transparent!)
            layer_name: Optional custom layer name (defaults to 'Geology: {domain}')
            show_edges: Whether to show mesh edges (default False for smooth surfaces)
            ghost_mode: If True, render as semi-transparent "ghost" (0.15 opacity)
        """
        # Apply ghost mode if requested (for "show other domains faded" feature)
        if ghost_mode:
            opacity = 0.15
        import hashlib
        
        if layer_name is None:
            layer_name = f"Geology: {domain}"
        
        # Validate inputs
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=np.int64)

        if vertices.size == 0 or faces.size == 0:
            logger.warning(f"Cannot add geology surface '{layer_name}': empty vertices or faces")
            return

        # ================================================================
        # GPU-SAFE MODE: Check for large geology surfaces
        # ================================================================
        LARGE_MODEL_THRESHOLD = 25000
        n_faces = len(faces)
        if n_faces > LARGE_MODEL_THRESHOLD:
            logger.warning(f"[GEOLOGY SURFACE] Large surface detected ({n_faces:,} faces) - enabling GPU-safe mode")
            self._has_large_model = True
            if self._resize_debounce is not None:
                self._resize_debounce.activate()

        # ================================================================
        # CRITICAL FIX: Apply coordinate shift to geology surface vertices
        # ================================================================
        # Geology surfaces must use the SAME coordinate shift as drillholes
        # to render together in the same scene. Without this, geology surfaces
        # remain at world coordinates while drillholes are shifted to local,
        # causing them to render in different places.
        # ================================================================
        vertices = self._to_local_precision(vertices)
        logger.debug(f"Applied coordinate shift to geology surface '{layer_name}' ({len(vertices)} vertices)")

        # Create PyVista mesh from vertices and faces
        # PyVista requires faces in format: [n_verts, v0, v1, v2, n_verts, ...]
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
        mesh = pv.PolyData(vertices, faces_pv)
        
        # QUALITY FIX: Auto-orient normals outward first, then compute consistent normals
        # This ensures surfaces are visible from the correct side and prevents disappearing faces
        try:
            # Step 1: Auto-orient normals to point outward (critical for marching cubes surfaces)
            if hasattr(mesh, 'compute_normals'):
                mesh = mesh.compute_normals(
                    cell_normals=True,
                    point_normals=True,
                    split_vertices=False,  # Preserve connectivity
                    flip_normals=False,
                    consistent_normals=True,  # Ensure consistent orientation
                    auto_orient_normals=True,  # AUTO-ORIENT outward - fixes disappearing surfaces
                )
        except Exception as e:
            logger.debug(f"Could not compute normals for surface: {e}")
        
        # Deterministic color selection if not provided
        if color is None:
            palette = [
                (0.20, 0.60, 0.86),  # Blue
                (0.95, 0.40, 0.27),  # Red-orange
                (0.18, 0.80, 0.44),  # Green
                (0.62, 0.35, 0.71),  # Purple
                (0.98, 0.76, 0.19),  # Yellow
                (0.27, 0.74, 0.74),  # Cyan
                (0.90, 0.49, 0.13),  # Orange
                (0.55, 0.55, 0.55),  # Gray
            ]
            h = hashlib.md5(str(domain).encode("utf-8")).digest()[0]
            color = palette[h % len(palette)]
        
        # QUALITY FIX: Calculate unique offset for this layer to prevent z-fighting
        # Each layer gets a slightly different polygon offset based on its index
        layer_index = len([l for l in self.active_layers.keys() if 'Geology' in l or 'geology' in l])
        
        # Add mesh to plotter with STATIC colors (no view-dependent shading)
        # Geological models should show consistent colors regardless of view angle
        actor = self.plotter.add_mesh(
            mesh,
            name=layer_name,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            smooth_shading=False,  # Flat shading for consistent colors
            show_scalar_bar=False,
            reset_camera=False,
            lighting=True,        # Keep minimal lighting for depth perception
            ambient=0.85,         # HIGH ambient = colors stay consistent
            diffuse=0.15,         # LOW diffuse = minimal angle-dependent shading
            specular=0.0,         # NO specular = no shiny spots that move
            specular_power=1,     # Not used when specular=0
            pbr=False,            # Disable PBR
            render_lines_as_tubes=False,
            render_points_as_spheres=False,
        )
        
        # CRITICAL FIX: Configure VTK properties for proper surface visibility
        # This fixes surfaces disappearing at certain viewing angles
        if actor is not None:
            try:
                prop = actor.GetProperty()
                if prop:
                    # FIX 1: DISABLE BACKFACE CULLING - surfaces visible from ALL angles
                    # Without this, surfaces disappear when viewed from "behind"
                    prop.SetBackfaceCulling(0)  # 0 = disabled, surfaces always visible
                    prop.SetFrontfaceCulling(0)  # Also ensure front faces are never culled
                    
                    # FIX 2: ENABLE TWO-SIDED LIGHTING - back faces are properly lit
                    # Without this, back-facing surfaces appear dark/black
                    prop.BackfaceCullingOff()  # Redundant but explicit
                    
                    # FIX 3: Apply unique polygon offset to prevent z-fighting
                    # Each surface gets a progressively different offset to separate them in depth buffer
                    mapper = actor.GetMapper()
                    if mapper:
                        mapper.SetResolveCoincidentTopologyToPolygonOffset()
                        offset_factor = 1.0 + layer_index * 0.5  # Unique per layer
                        offset_units = 1.0 + layer_index * 0.5
                        mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(offset_factor, offset_units)
                        logger.debug(f"Applied polygon offset {offset_factor} to layer {layer_name}")
                    
                    logger.debug(f"Configured two-sided rendering for layer {layer_name}")
            except Exception as e:
                logger.debug(f"Could not configure VTK properties: {e}")
        
        # Register layer using generic method - use 'geology_surface' type for proper detection
        self.add_layer(layer_name, actor, mesh, layer_type='geology_surface', opacity=opacity)
        
        # Register in scene_layers for global picking
        self.register_scene_layer(layer_name, actor, mesh, 'mesh')
        
        logger.info(f"Added geology surface layer: {layer_name} ({len(vertices)} vertices, {len(faces)} faces)")
        
        # CRITICAL FIX: Force render AND Qt repaint to ensure visualization appears
        if self.plotter is not None:
            try:
                self.plotter.render()
                # Force Qt widget repaint - essential for visibility
                if hasattr(self.plotter, 'interactor') and self.plotter.interactor is not None:
                    self.plotter.interactor.repaint()
                logger.debug(f"Rendered and repainted geology surface layer '{layer_name}'")
            except Exception as e:
                logger.debug(f"Could not render after adding geology surface layer: {e}")
    
    def add_fault_plane_layer(
        self,
        fault_data: Dict[str, Any],
        layer_name: Optional[str] = None,
        extent: float = 500.0,
        resolution: int = 10,
        opacity: float = 0.6,
        show_edges: bool = True,
    ) -> None:
        """
        Add a fault plane visualization to the scene.
        
        Creates a semi-transparent planar mesh representing the fault,
        with edges shown by default for clear orientation.
        
        Args:
            fault_data: Dict with fault definition (from FaultPlane.to_dict())
                Required keys: name, point, normal
                Optional: dip, azimuth, throw, metadata
            layer_name: Custom layer name (defaults to 'Fault: {name}')
            extent: Size of the fault plane mesh in meters
            resolution: Number of grid divisions for the plane mesh
            opacity: Fault plane opacity (default 0.6 - semi-transparent)
            show_edges: Whether to show mesh edges (default True)
        """
        name = fault_data.get('name', 'Fault')
        if layer_name is None:
            layer_name = f"Fault: {name}"
        
        # Get fault geometry
        point = np.array(fault_data.get('point', [0, 0, 0]), dtype=float)
        normal = np.array(fault_data.get('normal', [0, 0, 1]), dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        
        # Build local coordinate system on the fault plane
        if abs(normal[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])
        
        u = np.cross(normal, up)
        u = u / (np.linalg.norm(u) + 1e-10)
        v = np.cross(normal, u)
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # Generate grid of vertices
        half = extent / 2
        vertices = []
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                s = -half + extent * i / resolution
                t = -half + extent * j / resolution
                vertex = point + s * u + t * v
                vertices.append(vertex)
        
        vertices = np.array(vertices, dtype=float)
        
        # Generate triangle faces
        faces = []
        for i in range(resolution):
            for j in range(resolution):
                idx = i * (resolution + 1) + j
                # Two triangles per quad
                faces.append([idx, idx + 1, idx + resolution + 1])
                faces.append([idx + 1, idx + resolution + 2, idx + resolution + 1])
        
        faces = np.array(faces, dtype=np.int64)
        
        # Create PyVista mesh
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
        mesh = pv.PolyData(vertices, faces_pv)
        
        # Compute normals for proper rendering
        try:
            mesh = mesh.compute_normals(
                cell_normals=True,
                point_normals=True,
                split_vertices=False,
                flip_normals=False,
                consistent_normals=True,
            )
        except Exception as e:
            logger.debug(f"Could not compute normals for fault plane: {e}")
        
        # Get color from metadata or use default fault color (red-orange)
        color = fault_data.get('metadata', {}).get('color', '#ff5555')
        if isinstance(color, str) and color.startswith('#'):
            # Convert hex to RGB tuple
            color = color.lstrip('#')
            rgb = tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        else:
            rgb = (0.9, 0.3, 0.3)  # Default red-orange
        
        # Add mesh to plotter
        actor = self.plotter.add_mesh(
            mesh,
            name=layer_name,
            color=rgb,
            opacity=opacity,
            show_edges=show_edges,
            edge_color=(0.5, 0.2, 0.2),  # Darker edge color
            line_width=1.5,
            smooth_shading=True,
            show_scalar_bar=False,
            reset_camera=False,
            lighting=True,
            ambient=0.3,
            diffuse=0.6,
            specular=0.1,
            pbr=False,
        )
        
        # Register layer
        self.add_layer(layer_name, actor, mesh, layer_type='fault_plane', opacity=opacity)
        self.register_scene_layer(layer_name, actor, mesh, 'fault')
        
        logger.info(f"Added fault plane layer: {layer_name} (dip={fault_data.get('dip', 'N/A')}°)")
    
    def add_structural_fault_layer(
        self,
        fault_feature,
        layer_name: Optional[str] = None,
        color: Optional[Tuple[float, float, float]] = None,
        opacity: float = 0.6,
        show_edges: bool = True,
    ) -> None:
        """
        Add a FaultFeature from structural CSV import to the scene.
        
        Renders fault surface points and orientation arrows.
        
        Args:
            fault_feature: FaultFeature object from structural.feature_types
            layer_name: Custom layer name (defaults to 'Fault: {name}')
            color: RGB tuple (0-1 range), defaults to red-orange
            opacity: Surface opacity (default 0.6)
            show_edges: Whether to show mesh edges
        """
        name = fault_feature.name
        if layer_name is None:
            layer_name = f"Fault: {name}"
        
        # Default fault color: red-orange
        if color is None:
            color = (0.9, 0.3, 0.3)
        
        # Render surface points as mesh if available
        if fault_feature.point_count >= 3:
            try:
                from scipy.spatial import Delaunay
                
                points = fault_feature.surface_points
                
                # Create Delaunay triangulation for surface
                try:
                    tri = Delaunay(points[:, :2])  # Project to XY for triangulation
                    faces = tri.simplices
                    
                    # Create PyVista mesh
                    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
                    mesh = pv.PolyData(points, faces_pv)
                    
                    # Compute normals
                    mesh = mesh.compute_normals(consistent_normals=True)
                    
                    # Add to scene
                    actor = self.plotter.add_mesh(
                        mesh,
                        name=layer_name,
                        color=color,
                        opacity=opacity,
                        show_edges=show_edges,
                        edge_color=(color[0] * 0.5, color[1] * 0.5, color[2] * 0.5),
                        smooth_shading=True,
                        lighting=True,
                        ambient=0.3,
                        diffuse=0.6,
                    )
                    
                    self.add_layer(layer_name, actor, mesh, layer_type='structural_fault', opacity=opacity)
                    logger.info(f"Added structural fault surface: {layer_name} ({len(points)} points)")
                    
                except Exception as e:
                    logger.warning(f"Could not triangulate fault surface: {e}")
                    # Fallback: render as point cloud
                    mesh = pv.PolyData(points)
                    actor = self.plotter.add_mesh(
                        mesh,
                        name=layer_name,
                        color=color,
                        point_size=8.0,
                        render_points_as_spheres=True,
                    )
                    self.add_layer(layer_name, actor, mesh, layer_type='structural_fault', opacity=1.0)
                    logger.info(f"Added structural fault points: {layer_name} ({len(points)} points)")
                    
            except ImportError:
                logger.warning("scipy not available for triangulation")
        
        # Render orientations as arrows if available
        if hasattr(fault_feature, 'orientations') and fault_feature.orientations:
            orient_layer = f"{layer_name}_orientations"
            self._render_orientation_arrows(
                fault_feature.orientations,
                orient_layer,
                color=color,
                scale=50.0,
            )
    
    def add_structural_fold_layer(
        self,
        fold_feature,
        layer_name: Optional[str] = None,
        color: Optional[Tuple[float, float, float]] = None,
        axis_color: Optional[Tuple[float, float, float]] = None,
        opacity: float = 0.5,
        show_axis: bool = True,
    ) -> None:
        """
        Add a FoldFeature from structural CSV import to the scene.
        
        Renders fold axis as a tube and optional surface points.
        
        Args:
            fold_feature: FoldFeature object from structural.feature_types
            layer_name: Custom layer name (defaults to 'Fold: {name}')
            color: RGB tuple for limb surfaces
            axis_color: RGB tuple for fold axis (defaults to purple)
            opacity: Surface opacity
            show_axis: Whether to render fold axis
        """
        name = fold_feature.name
        if layer_name is None:
            layer_name = f"Fold: {name}"
        
        # Default colors
        if color is None:
            color = (0.2, 0.6, 0.2)  # Green for fold limbs
        if axis_color is None:
            axis_color = (0.6, 0.2, 0.8)  # Purple for fold axis
        
        # Render fold axis if we have axis data
        if show_axis and hasattr(fold_feature, 'average_fold_axis') and fold_feature.average_fold_axis:
            plunge, trend = fold_feature.average_fold_axis
            
            if fold_feature.centroid is not None:
                center = fold_feature.centroid
                
                # Calculate axis direction vector
                plunge_rad = np.radians(plunge)
                trend_rad = np.radians(trend)
                
                axis_dir = np.array([
                    np.cos(plunge_rad) * np.sin(trend_rad),  # X (East)
                    np.cos(plunge_rad) * np.cos(trend_rad),  # Y (North)
                    -np.sin(plunge_rad),                      # Z (Down)
                ])
                
                # Create axis line
                axis_length = fold_feature.wavelength or 500.0
                start = center - axis_dir * (axis_length / 2)
                end = center + axis_dir * (axis_length / 2)
                
                axis_line = pv.Line(start, end)
                axis_tube = axis_line.tube(radius=10.0, n_sides=12)
                
                axis_layer = f"{layer_name}_axis"
                actor = self.plotter.add_mesh(
                    axis_tube,
                    name=axis_layer,
                    color=axis_color,
                    opacity=0.8,
                    smooth_shading=True,
                    lighting=True,
                )
                self.add_layer(axis_layer, actor, axis_tube, layer_type='fold_axis', opacity=0.8)
                logger.info(f"Added fold axis: {axis_layer} (plunge={plunge:.1f}°, trend={trend:.1f}°)")
        
        # Render surface points if available
        if fold_feature.point_count >= 3:
            points = fold_feature.surface_points
            mesh = pv.PolyData(points)
            
            actor = self.plotter.add_mesh(
                mesh,
                name=layer_name,
                color=color,
                point_size=6.0,
                render_points_as_spheres=True,
                opacity=opacity,
            )
            self.add_layer(layer_name, actor, mesh, layer_type='structural_fold', opacity=opacity)
            logger.info(f"Added fold points: {layer_name} ({len(points)} points)")
    
    def add_structural_unconformity_layer(
        self,
        unconformity_feature,
        layer_name: Optional[str] = None,
        color: Optional[Tuple[float, float, float]] = None,
        opacity: float = 0.7,
        show_edges: bool = True,
    ) -> None:
        """
        Add an UnconformityFeature from structural CSV import to the scene.
        
        Renders unconformity surface with distinct styling.
        
        Args:
            unconformity_feature: UnconformityFeature object from structural.feature_types
            layer_name: Custom layer name (defaults to 'Unconformity: {name}')
            color: RGB tuple (defaults to orange-yellow for erosional surfaces)
            opacity: Surface opacity
            show_edges: Whether to show mesh edges
        """
        name = unconformity_feature.name
        if layer_name is None:
            layer_name = f"Unconformity: {name}"
        
        # Default unconformity color: orange-yellow (erosion surface)
        if color is None:
            color = (0.95, 0.65, 0.2)
        
        # Render surface points as mesh if available
        if unconformity_feature.point_count >= 3:
            try:
                from scipy.spatial import Delaunay
                
                points = unconformity_feature.surface_points
                
                # Create Delaunay triangulation
                try:
                    tri = Delaunay(points[:, :2])
                    faces = tri.simplices
                    
                    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
                    mesh = pv.PolyData(points, faces_pv)
                    mesh = mesh.compute_normals(consistent_normals=True)
                    
                    actor = self.plotter.add_mesh(
                        mesh,
                        name=layer_name,
                        color=color,
                        opacity=opacity,
                        show_edges=show_edges,
                        edge_color=(color[0] * 0.6, color[1] * 0.6, color[2] * 0.6),
                        smooth_shading=True,
                        lighting=True,
                        ambient=0.35,
                        diffuse=0.55,
                    )
                    
                    self.add_layer(layer_name, actor, mesh, layer_type='structural_unconformity', opacity=opacity)
                    logger.info(f"Added unconformity surface: {layer_name} ({len(points)} points)")
                    
                except Exception as e:
                    logger.warning(f"Could not triangulate unconformity: {e}")
                    mesh = pv.PolyData(points)
                    actor = self.plotter.add_mesh(
                        mesh,
                        name=layer_name,
                        color=color,
                        point_size=8.0,
                        render_points_as_spheres=True,
                    )
                    self.add_layer(layer_name, actor, mesh, layer_type='structural_unconformity', opacity=1.0)
                    
            except ImportError:
                logger.warning("scipy not available for triangulation")
        
        # Render orientations as arrows if available
        if hasattr(unconformity_feature, 'orientations') and unconformity_feature.orientations:
            orient_layer = f"{layer_name}_orientations"
            self._render_orientation_arrows(
                unconformity_feature.orientations,
                orient_layer,
                color=color,
                scale=40.0,
            )
    
    def _render_orientation_arrows(
        self,
        orientations,
        layer_name: str,
        color: Tuple[float, float, float] = (0.8, 0.8, 0.2),
        scale: float = 50.0,
    ) -> None:
        """
        Render structural orientations as arrow glyphs.
        
        Args:
            orientations: List of StructuralOrientation objects
            layer_name: Layer name for the arrows
            color: Arrow color
            scale: Arrow length scale
        """
        if not orientations:
            return
        
        points = []
        vectors = []
        
        for orient in orientations:
            points.append([orient.x, orient.y, orient.z])
            
            # Convert dip/azimuth to vector
            dip_rad = np.radians(orient.dip)
            az_rad = np.radians(orient.azimuth)
            
            # Vector pointing in dip direction (downward)
            vx = np.sin(dip_rad) * np.sin(az_rad) * orient.polarity
            vy = np.sin(dip_rad) * np.cos(az_rad) * orient.polarity
            vz = -np.cos(dip_rad) * orient.polarity
            
            vectors.append([vx * scale, vy * scale, vz * scale])
        
        points = np.array(points)
        vectors = np.array(vectors)
        
        # Create glyph mesh
        glyphs = pv.PolyData(points)
        glyphs['vectors'] = vectors
        arrows = glyphs.glyph(orient='vectors', scale=False, factor=1.0)
        
        actor = self.plotter.add_mesh(
            arrows,
            name=layer_name,
            color=color,
            opacity=0.9,
            lighting=True,
        )
        
        self.add_layer(layer_name, actor, arrows, layer_type='orientation_arrows', opacity=0.9)
        logger.info(f"Added orientation arrows: {layer_name} ({len(orientations)} arrows)")
    
    def add_structural_features_collection(
        self,
        collection,
        prefix: str = "",
        fault_color: Optional[Tuple[float, float, float]] = None,
        fold_color: Optional[Tuple[float, float, float]] = None,
        unconformity_color: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Add all features from a StructuralFeatureCollection to the scene.
        
        Convenience method to render all faults, folds, and unconformities
        from a collection loaded via CSV import.
        
        Args:
            collection: StructuralFeatureCollection from structural.feature_types
            prefix: Optional prefix for layer names
            fault_color: Optional color override for faults
            fold_color: Optional color override for folds
            unconformity_color: Optional color override for unconformities
        """
        if collection is None:
            return
        
        # Add faults
        for fault in collection.faults:
            layer_name = f"{prefix}Fault: {fault.name}" if prefix else None
            self.add_structural_fault_layer(fault, layer_name=layer_name, color=fault_color)
        
        # Add folds
        for fold in collection.folds:
            layer_name = f"{prefix}Fold: {fold.name}" if prefix else None
            self.add_structural_fold_layer(fold, layer_name=layer_name, color=fold_color)
        
        # Add unconformities
        for unc in collection.unconformities:
            layer_name = f"{prefix}Unconformity: {unc.name}" if prefix else None
            self.add_structural_unconformity_layer(unc, layer_name=layer_name, color=unconformity_color)
        
        fc = collection.feature_count
        logger.info(f"Added structural feature collection: {fc['faults']} faults, {fc['folds']} folds, {fc['unconformities']} unconformities")
    
    def add_preview_layer(
        self,
        pv_object: Any,
        layer_name: str,
        color: Optional[Any] = None,
        opacity: float = 0.7,
        point_size: float = 8.0,
        line_width: float = 2.0,
    ) -> None:
        """
        Add a pre-solve preview layer to the scene.
        
        Handles various PyVista object types from presolve_preview module:
        - PolyData with points (contact points, drillhole traces)
        - PolyData with faces (fault plane previews)
        - Line objects (fold axes, extent cube edges)
        
        Args:
            pv_object: PyVista object (PolyData, Line, etc.) or dict with points/vectors
            layer_name: Unique name for this preview layer
            color: Optional color override
            opacity: Opacity for the preview element
            point_size: Size for point rendering
            line_width: Width for line rendering
        """
        if self.plotter is None:
            logger.warning(f"Cannot add preview layer '{layer_name}': plotter not initialized")
            return
        
        # Remove existing layer with same name
        if layer_name in self.active_layers:
            self.clear_layer(layer_name)
        
        try:
            # Handle dict format (for orientation vectors)
            if isinstance(pv_object, dict):
                if 'points' in pv_object and 'vectors' in pv_object:
                    # Orientation arrows
                    points = np.array(pv_object['points'])
                    vectors = np.array(pv_object['vectors'])
                    if len(points) > 0 and len(vectors) > 0:
                        glyphs = pv.PolyData(points)
                        glyphs['vectors'] = vectors
                        arrows = glyphs.glyph(orient='vectors', scale=False, factor=0.1)
                        actor = self.plotter.add_mesh(
                            arrows,
                            name=layer_name,
                            color=color or (0.2, 0.6, 0.2),
                            opacity=opacity,
                            show_scalar_bar=False,
                            reset_camera=False,
                        )
                        self.add_layer(layer_name, actor, arrows, layer_type='preview', opacity=opacity)
                        logger.info(f"Added preview orientation arrows: {layer_name} ({len(points)} arrows)")
                return
            
            # Check if it's a PyVista object
            if not hasattr(pv_object, 'n_points'):
                logger.warning(f"Unknown preview object type for '{layer_name}': {type(pv_object)}")
                return
            
            # Determine rendering style based on geometry
            n_points = pv_object.n_points
            n_cells = pv_object.n_cells if hasattr(pv_object, 'n_cells') else 0
            
            if n_points == 0:
                logger.debug(f"Empty preview object: {layer_name}")
                return
            
            # Default colors based on layer type
            if color is None:
                if 'contact' in layer_name.lower():
                    color = (0.8, 0.5, 0.2)  # Orange for contacts
                elif 'fault' in layer_name.lower():
                    color = (0.9, 0.2, 0.2)  # Red for faults
                elif 'fold' in layer_name.lower():
                    color = (0.2, 0.7, 0.2)  # Green for folds
                elif 'extent' in layer_name.lower():
                    color = (0.5, 0.5, 0.5)  # Gray for extent cube
                elif 'hole' in layer_name.lower():
                    color = (0.2, 0.4, 0.8)  # Blue for drillholes
                else:
                    color = (0.6, 0.6, 0.6)  # Default gray
            
            # Check if object has faces (surface) or just points/lines
            has_faces = False
            if hasattr(pv_object, 'faces') and pv_object.faces is not None:
                has_faces = len(pv_object.faces) > 0
            elif hasattr(pv_object, 'n_faces'):
                has_faces = pv_object.n_faces > 0
            
            if has_faces:
                # Render as surface mesh
                actor = self.plotter.add_mesh(
                    pv_object,
                    name=layer_name,
                    color=color,
                    opacity=opacity,
                    show_edges=True,
                    edge_color=(0.3, 0.3, 0.3),
                    line_width=1.0,
                    smooth_shading=True,
                    show_scalar_bar=False,
                    reset_camera=False,
                    lighting=True,
                )
                logger.info(f"Added preview surface: {layer_name} ({n_points} vertices)")
            elif n_cells > 0 and n_cells < n_points:
                # Likely lines (polylines)
                actor = self.plotter.add_mesh(
                    pv_object,
                    name=layer_name,
                    color=color,
                    line_width=line_width,
                    render_lines_as_tubes=True,
                    show_scalar_bar=False,
                    reset_camera=False,
                )
                logger.info(f"Added preview lines: {layer_name} ({n_cells} lines)")
            else:
                # Render as points
                actor = self.plotter.add_mesh(
                    pv_object,
                    name=layer_name,
                    color=color,
                    point_size=point_size,
                    render_points_as_spheres=True,
                    opacity=opacity,
                    show_scalar_bar=False,
                    reset_camera=False,
                )
                logger.info(f"Added preview points: {layer_name} ({n_points} points)")
            
            # Register layer
            self.add_layer(layer_name, actor, pv_object, layer_type='preview', opacity=opacity)
            
        except Exception as e:
            logger.error(f"Failed to add preview layer '{layer_name}': {e}", exc_info=True)
    
    def clear_preview_layers(self) -> None:
        """Remove all preview layers from the scene."""
        preview_layers = [name for name in self.active_layers if self.active_layers[name].get('type') == 'preview']
        for layer_name in preview_layers:
            self.clear_layer(layer_name)
        logger.info(f"Cleared {len(preview_layers)} preview layers")
    
    def add_fault_trace_line(
        self,
        fault_data: Dict[str, Any],
        z_level: float,
        extent: float = 1000.0,
        layer_name: Optional[str] = None,
        line_width: float = 3.0,
    ) -> None:
        """
        Add a fault trace line at a specific Z level (e.g., for section views).
        
        Args:
            fault_data: Dict with fault definition
            z_level: Z coordinate for the horizontal section
            extent: Length of the trace line
            layer_name: Custom layer name
            line_width: Width of the trace line
        """
        name = fault_data.get('name', 'Fault')
        if layer_name is None:
            layer_name = f"Fault Trace: {name} @ Z={z_level:.0f}"
        
        # Get fault geometry
        point = np.array(fault_data.get('point', [0, 0, 0]), dtype=float)
        normal = np.array(fault_data.get('normal', [0, 0, 1]), dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        
        # Calculate strike direction (horizontal perpendicular to dip direction)
        # Strike = cross product of normal with vertical
        vertical = np.array([0, 0, 1])
        strike = np.cross(normal, vertical)
        strike_norm = np.linalg.norm(strike)
        
        if strike_norm < 1e-6:
            # Vertical fault - strike is arbitrary
            strike = np.array([1, 0, 0])
        else:
            strike = strike / strike_norm
        
        # Find intersection point with z_level
        # The fault plane equation: normal . (p - point) = 0
        # At z = z_level, we need to find x, y on the plane
        if abs(normal[2]) > 1e-6:
            # Plane intersects z_level
            t = (z_level - point[2]) / normal[2] if abs(normal[2]) > 1e-6 else 0
            intersection = point + t * np.array([0, 0, 1])
        else:
            # Vertical plane - intersection is at any z
            intersection = np.array([point[0], point[1], z_level])
        
        # Create line points along strike
        half = extent / 2
        p1 = intersection - half * strike
        p2 = intersection + half * strike
        
        # Ensure z-level is respected
        p1[2] = z_level
        p2[2] = z_level
        
        # Create PyVista line
        line = pv.Line(p1, p2, resolution=1)
        
        # Get color
        color = fault_data.get('metadata', {}).get('color', '#ff5555')
        if isinstance(color, str) and color.startswith('#'):
            color = color.lstrip('#')
            rgb = tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        else:
            rgb = (0.9, 0.3, 0.3)
        
        # Add to plotter
        actor = self.plotter.add_mesh(
            line,
            name=layer_name,
            color=rgb,
            line_width=line_width,
            render_lines_as_tubes=True,
            show_scalar_bar=False,
            reset_camera=False,
        )
        
        # Register layer
        self.add_layer(layer_name, actor, line, layer_type='fault_trace', opacity=1.0)
        
        logger.info(f"Added fault trace line: {layer_name}")
    
    def visualize_fault_system(
        self,
        faults: List[Dict[str, Any]],
        extent: Optional[float] = None,
        auto_extent: bool = True,
    ) -> None:
        """
        Visualize multiple faults as a fault system.
        
        Args:
            faults: List of fault data dicts
            extent: Size of fault plane meshes (auto-calculated if None)
            auto_extent: Whether to auto-calculate extent from scene bounds
        """
        if not faults:
            logger.info("No faults to visualize")
            return
        
        # Calculate extent from scene bounds if needed
        if extent is None and auto_extent:
            try:
                bounds = self.plotter.bounds
                if bounds:
                    x_range = bounds[1] - bounds[0]
                    y_range = bounds[3] - bounds[2]
                    z_range = bounds[5] - bounds[4]
                    extent = max(x_range, y_range, z_range) * 0.5
                else:
                    extent = 500.0
            except Exception:
                extent = 500.0
        
        extent = extent or 500.0
        
        # Add each fault
        for fault_data in faults:
            if not fault_data.get('active', True):
                continue
            
            try:
                self.add_fault_plane_layer(
                    fault_data=fault_data,
                    extent=extent,
                    opacity=0.5,
                    show_edges=True,
                )
            except Exception as e:
                logger.warning(f"Failed to visualize fault '{fault_data.get('name', 'Unknown')}': {e}")
        
        logger.info(f"Visualized {len(faults)} fault planes")
    
    def remove_fault_layers(self) -> None:
        """Remove all fault-related layers from the scene."""
        layers_to_remove = []
        
        for layer_name, layer_info in self.active_layers.items():
            layer_type = layer_info.get('type', layer_info.get('layer_type', ''))
            if layer_type in ('fault_plane', 'fault_trace') or layer_name.startswith('Fault'):
                layers_to_remove.append(layer_name)
        
        for layer_name in layers_to_remove:
            self.remove_layer(layer_name)
        
        if layers_to_remove:
            logger.info(f"Removed {len(layers_to_remove)} fault layers")
    
    def update_geology_layer_color(self, domain: str, color: Tuple[float, float, float, float]) -> bool:
        """
        Update the color of a geology surface or solid layer by domain name.
        
        Args:
            domain: Domain name (e.g., 'BIF', 'FRESH')
            color: RGBA tuple (0-1 range)
            
        Returns:
            True if any layers were updated, False otherwise
        """
        updated = False
        
        # Try surface layer
        surface_layer_name = f"geology_surface_{domain}"
        if surface_layer_name in self.active_layers:
            layer_info = self.active_layers[surface_layer_name]
            actor = layer_info.get('actor')
            if actor and hasattr(actor, 'GetProperty'):
                prop = actor.GetProperty()
                if prop:
                    prop.SetColor(color[0], color[1], color[2])
                    if len(color) > 3:
                        prop.SetOpacity(color[3])
                    logger.info(f"Updated geology surface '{domain}' color to {color[:3]}")
                    updated = True
        
        # Try solid layer (voxels) - these use scalar coloring, need different approach
        solid_layer_name = "geology_solids_voxels"
        if solid_layer_name in self.active_layers:
            # For voxels with scalar coloring, we need to update the lookup table
            # This is more complex - just log for now
            logger.debug(f"Voxel solid color update for '{domain}' not yet implemented")
        
        # Render if anything was updated
        if updated:
            try:
                self.plotter.render()
            except Exception as e:
                logger.debug(f"Render after color update failed: {e}")
        
        return updated
    
    def set_geology_single_domain_view(self, active_domain: Optional[str] = None) -> None:
        """
        Set single-domain view mode for geological surfaces.
        
        When active_domain is specified:
        - That domain is shown at full opacity
        - All other domains are "ghosted" (semi-transparent, 0.15 opacity)
        
        When active_domain is None:
        - All domains are shown at full opacity (default view)
        
        This prevents the visual chaos of overlapping transparent surfaces.
        
        Args:
            active_domain: Domain name to show fully, or None to show all
        """
        geology_layers = {}
        
        # Find all geology surface layers
        for layer_name, layer_info in self.active_layers.items():
            layer_type = layer_info.get('type', layer_info.get('layer_type', ''))
            if layer_type == 'geology_surface' or layer_name.startswith('geology_surface_'):
                # Extract domain name from layer name
                # Format: "geology_surface_{domain}" or "Geology: {domain}"
                if layer_name.startswith('geology_surface_'):
                    domain = layer_name.replace('geology_surface_', '')
                elif layer_name.startswith('Geology: '):
                    domain = layer_name.replace('Geology: ', '')
                else:
                    domain = layer_name
                geology_layers[domain] = layer_info
        
        if not geology_layers:
            logger.debug("No geology surface layers found for single-domain view")
            return
        
        for domain, layer_info in geology_layers.items():
            actor = layer_info.get('actor')
            if not actor or not hasattr(actor, 'GetProperty'):
                continue
            
            prop = actor.GetProperty()
            if not prop:
                continue
            
            if active_domain is None:
                # Show all at full opacity
                prop.SetOpacity(1.0)
                logger.debug(f"Geology layer '{domain}': full opacity")
            elif domain == active_domain:
                # Active domain at full opacity
                prop.SetOpacity(1.0)
                logger.info(f"Geology layer '{domain}': ACTIVE (full opacity)")
            else:
                # Non-active domains ghosted
                prop.SetOpacity(0.15)
                logger.debug(f"Geology layer '{domain}': ghosted")
        
        # Render
        try:
            self.plotter.render()
        except Exception as e:
            logger.debug(f"Render after single-domain view failed: {e}")
    
    def set_geology_ghost_mode(self, enabled: bool, exclude_domain: Optional[str] = None) -> None:
        """
        Enable or disable ghost mode for all geology surfaces.
        
        Ghost mode: All surfaces rendered at 0.15 opacity to see through them.
        This is useful for inspecting drillholes within geological volumes.
        
        Args:
            enabled: True to enable ghost mode, False to restore full opacity
            exclude_domain: Optional domain to keep at full opacity even in ghost mode
        """
        opacity = 0.15 if enabled else 1.0
        
        for layer_name, layer_info in self.active_layers.items():
            layer_type = layer_info.get('type', layer_info.get('layer_type', ''))
            if layer_type != 'geology_surface' and not layer_name.startswith('geology_surface_'):
                continue
            
            # Check if this layer should be excluded
            if exclude_domain and (exclude_domain in layer_name):
                continue
            
            actor = layer_info.get('actor')
            if actor and hasattr(actor, 'GetProperty'):
                prop = actor.GetProperty()
                if prop:
                    prop.SetOpacity(opacity)
        
        try:
            self.plotter.render()
        except Exception as e:
            logger.debug(f"Render after ghost mode toggle failed: {e}")
        
        logger.info(f"Geology ghost mode: {'enabled' if enabled else 'disabled'}")
    
    def load_grid(self, payload) -> None:
        """
        Load a grid from a GridPayload (fallback for payload system).
        
        NOTE: This method is provided for backward compatibility with the payload system.
        New code should use add_block_model_layer() directly with a PyVista grid.
        
        Args:
            payload: GridPayload instance
            
        Returns:
            Actor or None
        """
        from .grid_adapter import GridAdapter
        from .render_payloads import GridPayload
        
        if not isinstance(payload, GridPayload):
            logger.error(f"load_grid: Expected GridPayload, got {type(payload)}")
            return None
        
        try:
            logger.info(f"load_grid: Loading grid '{payload.name}' via GridAdapter")
            
            # Use GridAdapter to convert payload to PyVista grid
            adapter = GridAdapter()
            grid, scalar_name = adapter.to_pv_grid(payload)
            
            if grid is None:
                logger.error("load_grid: GridAdapter returned None grid")
                return None
            
            if grid.n_cells == 0:
                logger.error("load_grid: Grid has 0 cells")
                return None
            
            logger.info(f"load_grid: Created grid with {grid.n_cells} cells, bounds={grid.bounds}")
            
            # Use scalar_name from payload if available, otherwise use payload name
            property_name = scalar_name if scalar_name else payload.name
            
            # Create layer name
            layer_name = payload.metadata.get('layer_name', f"Grid: {payload.name}")
            
            # Remove existing layer with same name
            if layer_name in self.active_layers:
                logger.info(f"load_grid: Removing existing layer '{layer_name}'")
                self.clear_layer(layer_name)
            
            # Use add_block_model_layer to add the grid (reuses existing logic)
            self.add_block_model_layer(grid, property_name, layer_name=layer_name)
            
            # Update legend if LegendManager is available
            if hasattr(self, 'legend_manager') and self.legend_manager is not None:
                try:
                    # Get data range for legend
                    if property_name in grid.cell_data:
                        data = grid.cell_data[property_name]
                    elif property_name in grid.point_data:
                        data = grid.point_data[property_name]
                    else:
                        logger.warning(f"load_grid: Property '{property_name}' not found in grid")
                        return None
                    
                    valid_data = data[np.isfinite(data)]
                    if len(valid_data) > 0:
                        vmin = float(np.nanmin(valid_data))
                        vmax = float(np.nanmax(valid_data))
                        colormap = payload.metadata.get('colormap', 'viridis')
                        
                        logger.info(f"load_grid: Updating legend for '{property_name}', range=[{vmin:.3f}, {vmax:.3f}]")
                        self.legend_manager.set_continuous(
                            field=property_name.upper(),
                            vmin=vmin,
                            vmax=vmax,
                            cmap_name=colormap
                        )
                except Exception as e:
                    logger.warning(f"load_grid: Could not update legend: {e}", exc_info=True)
            
            # Get actor from active layers
            if layer_name in self.active_layers:
                actor = self.active_layers[layer_name]['actor']
                logger.info(f"load_grid: Successfully loaded grid '{payload.name}' as layer '{layer_name}'")
                return actor
            else:
                logger.warning(f"load_grid: Layer '{layer_name}' was not registered in active_layers")
                return None
                
        except Exception as e:
            logger.error(f"load_grid: Error loading grid: {e}", exc_info=True)
            return None
    
    def load_mesh(self, payload) -> None:
        """
        Load a surface mesh from a MeshPayload.
        
        Supports geological surfaces from ChronosEngine, pit shells, and other
        mesh-based visualizations.
        
        Args:
            payload: MeshPayload instance with vertices, faces, etc.
            
        Returns:
            Actor or None
        """
        from .mesh_adapter import MeshAdapter
        from .render_payloads import MeshPayload
        
        if not isinstance(payload, MeshPayload):
            logger.error(f"load_mesh: Expected MeshPayload, got {type(payload)}")
            return None
        
        if self.plotter is None:
            logger.error("load_mesh: Plotter not initialized")
            return None
        
        try:
            logger.info(f"load_mesh: Loading mesh '{payload.name}'")
            
            # Use MeshAdapter to convert payload to PyVista mesh
            adapter = MeshAdapter()
            mesh, scalars = adapter.to_pv_mesh(payload)
            
            if mesh is None:
                logger.error("load_mesh: MeshAdapter returned None mesh")
                return None
            
            if mesh.n_points == 0:
                logger.error("load_mesh: Mesh has 0 points")
                return None
            
            logger.info(f"load_mesh: Created mesh with {mesh.n_points} points, {mesh.n_faces} faces")
            
            # Determine layer name
            layer_name = payload.metadata.get('layer_name', payload.name)
            
            # Remove existing layer with same name
            if layer_name in self.active_layers:
                logger.info(f"load_mesh: Removing existing layer '{layer_name}'")
                self.clear_layer(layer_name)
            
            # Determine visualization parameters
            # FIXED: Default to fully opaque (1.0) and no edges for clean geological rendering
            opacity = payload.opacity if payload.opacity is not None else 1.0
            color = payload.metadata.get('color', '#3498db')
            show_edges = payload.metadata.get('show_edges', False)

            # Contacts/point clouds need point rendering and colormap
            is_contacts = payload.metadata.get('type') == 'geology_contacts' or layer_name == 'GeoContacts'
            use_points = (mesh.n_faces == 0) or is_contacts
            point_size = payload.metadata.get('point_size', 8.0 if is_contacts else 5.0)
            cmap = payload.metadata.get('colormap') if payload.metadata.get('colormap') else None
            scalars_name = payload.name if scalars is not None else None
            
            # Add mesh to plotter
            actor = self.plotter.add_mesh(
                mesh,
                name=layer_name,
                color=None if scalars_name else color,
                scalars=scalars_name,
                cmap=cmap,
                opacity=opacity,
                show_edges=show_edges,
                edge_color='#333333',
                smooth_shading=not use_points,
                render_points_as_spheres=use_points,
                point_size=point_size if use_points else None,
                render=False  # Batch updates
            )
            
            # Register as layer
            layer_type = 'geology_contacts' if is_contacts else 'mesh'

            self.add_layer(
                layer_name,
                actor,
                data={'mesh': mesh, 'payload': payload},
                layer_type=layer_type,
                opacity=opacity
            )

            # Respect payload visibility flag
            try:
                actor.SetVisibility(1 if payload.visible else 0)
            except Exception:
                pass
            
            # Force render with full Qt/VTK synchronization
            self.force_render_update()
            
            logger.info(f"load_mesh: Successfully loaded mesh '{payload.name}' as layer '{layer_name}'")
            return actor
            
        except Exception as e:
            logger.error(f"load_mesh: Error loading mesh: {e}", exc_info=True)
            return None
    
    # =========================================================================
    # CP-Grade Geological Integration (Leapfrog/Vulcan-style)
    # =========================================================================
    
    def load_geology_package(self, package: Dict[str, Any], render_mode: str = "surfaces") -> None:
        """
        CP-GRADE INTEGRATION: Plots geology into the MAIN viewer.
        
        Handles surfaces, solids, and audit data from GeologicalModelRunner.
        This is the single entry point for all geological visualization,
        ensuring everything appears in the main 3D view alongside 
        drillholes and block models.
        
        RENDERING MODES:
        - "surfaces": Display isosurface meshes (2D contact boundaries) - DEFAULT
        - "solids": Display voxelized 3D block volumes using composite rendering
        - "both": Display both surfaces and solids (surfaces on top with lower opacity)
        
        Args:
            package: Dictionary containing:
                - 'surfaces': List of surface dicts with vertices, faces
                - 'solids': List of solid dicts with vertices, faces, volume_m3
                - 'report': AuditReport object (optional)
                - 'log': Provenance metadata (optional)
            render_mode: "surfaces", "solids", or "both"
        """
        if self.plotter is None:
            logger.error("load_geology_package: Plotter not initialized")
            return
        
        logger.info("=" * 60)
        logger.info(f"LOADING GEOLOGY PACKAGE INTO MAIN RENDERER (mode={render_mode})")
        logger.info("=" * 60)
        
        # DEBUG: Log coordinate shift state BEFORE loading geology
        logger.info(f"[GEOLOGY_LOAD] _global_shift before: {self._global_shift}")
        logger.info(f"[GEOLOGY_LOAD] Drillholes in active_layers: {'drillholes' in self.active_layers}")
        if hasattr(self, '_drillhole_hole_actors'):
            logger.info(f"[GEOLOGY_LOAD] Drillhole actors count: {len(self._drillhole_hole_actors)}")
        
        # DIAGNOSTIC: Check package contents
        n_surfaces = len(package.get('surfaces', []))
        n_solids = len(package.get('solids', []))
        unified_mesh = package.get('unified_mesh')
        logger.info(f"Package contains: {n_surfaces} surfaces, {n_solids} solids")
        logger.info(f"Unified mesh available: {unified_mesh is not None}")
        
        # DETAILED DEBUG: Check solid geometry
        for i, solid in enumerate(package.get('solids', [])):
            name = solid.get('unit_name', solid.get('name', f'Solid_{i}'))
            verts = solid.get('vertices')
            faces = solid.get('faces')
            v_count = len(verts) if verts is not None else 0
            f_count = len(faces) if faces is not None else 0
            logger.info(f"  Incoming solid '{name}': vertices={v_count}, faces={f_count}")
        
        # =========================================================================
        # UNIFIED MESH RENDERING (Voxel-Based Solids)
        # =========================================================================
        # The unified mesh has ONE Formation_ID per voxel - NO OVERLAP.
        # Used for "solids" mode (voxel view) and "both" mode (voxels + surfaces).
        # NOTE: "surfaces" mode uses SMOOTH marching cubes isosurfaces below.
        # =========================================================================
        if render_mode in ("solids", "both", "unified"):  # "surfaces" uses marching cubes
            # FIXED: Only use unified mesh if provided in the package
            # The model_runner is not available in the Renderer, so we can't extract on-demand
            # The package should already contain the unified_mesh if it was generated
            if unified_mesh is not None:
                logger.info("=" * 60)
                logger.info("USING UNIFIED PARTITION MESH (EXCLUSIVITY GUARANTEED)")
                logger.info("=" * 60)
                
                success = self._render_unified_geology_mesh(unified_mesh, package, render_mode=render_mode)
                if success:
                    # Unified mesh rendered successfully - skip individual solids
                    # Still render surfaces if mode is "both" and they're available
                    if render_mode == "both" and n_surfaces > 0:
                        logger.info("Also rendering surfaces overlay (mode='both')...")
                    else:
                        # Solid view complete - skip the rest
                        self._setup_geology_camera_and_bounds()
                        self.force_render_update()
                        self._force_repaint()
                        self._sync_geology_legend(package, render_mode, [])
                        logger.info("=" * 60)
                        logger.info(f"UNIFIED GEOLOGY MESH LOADED (BLEED-THROUGH ELIMINATED)")
                        logger.info("=" * 60)
                        return

        
        # 1. Clear existing geology layers to prevent overlap
        self.clear_geology_layers()
        
        # 2. Apply enhanced z-fighting prevention (depth peeling + polygon offset)
        self._apply_geology_z_fighting_fixes()
        
        # =========================================================================
        # CRITICAL FIX: LOCAL ORIGIN SHIFT FOR GPU FLOATING-POINT PRECISION
        # =========================================================================
        # Large UTM coordinates (500,000m+) exceed GPU 32-bit float precision.
        # The GPU can't distinguish sub-10cm differences at these magnitudes.
        # SOLUTION: Subtract model center from ALL vertices BEFORE GPU rendering.
        # This restores sub-millimeter precision by centering at (0, 0, 0).
        # =========================================================================
        self._local_origin = self._compute_local_origin(package)
        
        surfaces_added = 0
        solids_added = 0
        
        # Professional color palette for geological units
        geology_colors = [
            '#e74c3c',  # Red
            '#3498db',  # Blue
            '#2ecc71',  # Green
            '#f39c12',  # Orange
            '#9b59b6',  # Purple
            '#1abc9c',  # Teal
            '#e67e22',  # Dark Orange
            '#34495e',  # Dark Blue-Gray
        ]
        
        # 3. Plot Polished Surfaces (Isosurface contacts) if mode includes surfaces
        if render_mode in ("surfaces", "both"):
            # CRITICAL FIX: Use 100% OPAQUE rendering for stability
            # Transparent surfaces require Depth Peeling which causes driver issues.
            # 3. Plot Surfaces (Contacts/Faults)
            surfaces = package.get('surfaces', [])
            # Use the professional color palette defined above (geology_colors already set)
            
            # ====================================================================
            # GEOMETRIC SORTING: Force Physical Stacking Order (Top -> Bottom)
            # ====================================================================
            # Raw file order creates random Z-fighting ("Bottom layer on Top").
            # FIX: Sort surfaces by their physical Z-position (High to Low).
            # This ensures Index 0 is physically the TOP layer, Index N is BOTTOM.
            # Then, Index 0 gets Offset 0 (Front), Index N gets Offset N (Back).
            scored_surfaces = []
            for s in surfaces:
                verts = s.get('vertices')
                z_score = 0.0
                if verts is not None and len(verts) > 0:
                    # Use mean Z as the sorting key
                    z_score = np.mean(np.array(verts)[:, 2])
                scored_surfaces.append((z_score, s))
            
            # Sort DESCENDING (High Z first)
            scored_surfaces.sort(key=lambda x: x[0], reverse=True)
            sorted_surfaces = [x[1] for x in scored_surfaces]
            
            logger.info(f"Sorted {len(sorted_surfaces)} surfaces by Z-height for correct stacking")

            # Opaque surfaces + Polygon Offset = Industry Standard stability.
            surface_opacity = 1.0
            
            for i, surface in enumerate(sorted_surfaces):
                try:
                    name = surface.get('name', f'Contact_{i}')
                    # Use original color mapping (based on sorted order now)
                    # This keeps colors consistent with the physical stack
                    verts = np.asarray(surface['vertices'], dtype=np.float64)
                    faces = surface.get('faces')
                    
                    # DIAGNOSTIC: Check WORLD coordinate range BEFORE transform
                    if i == 0 and len(verts) > 0:
                        verts_world_min = np.min(verts, axis=0)
                        verts_world_max = np.max(verts, axis=0)
                        logger.info(f"DIAGNOSTIC: First surface '{name}' WORLD vertex range (before shift):")
                        logger.info(f"  X: [{verts_world_min[0]:.2f}, {verts_world_max[0]:.2f}]")
                        logger.info(f"  Y: [{verts_world_min[1]:.2f}, {verts_world_max[1]:.2f}]")
                        logger.info(f"  Z: [{verts_world_min[2]:.2f}, {verts_world_max[2]:.2f}]")
                        
                        # Check for suspicious coordinate ranges
                        x_range = verts_world_max[0] - verts_world_min[0]
                        y_range = verts_world_max[1] - verts_world_min[1]
                        
                        # If world coordinates are in [0,1] range, the inverse transform likely failed
                        if verts_world_max[0] < 10 and verts_world_max[1] < 10:
                            logger.error(
                                f"COORDINATE BUG DETECTED: Surface '{name}' has tiny world coordinates! "
                                f"Expected UTM-scale (e.g., 500000, 7000000) but got [{verts_world_max[0]:.2f}]. "
                                f"The inverse transform in ChronosEngine may have failed. "
                                f"Check the StandardScaler configuration."
                            )
                        
                        # If world coordinates span millions of meters, data may include outliers
                        if x_range > 1000000 or y_range > 1000000:
                            logger.warning(
                                f"COORDINATE WARNING: Surface '{name}' spans {x_range:.0f}m x {y_range:.0f}m. "
                                f"This may indicate outlier drillholes in the source data. "
                                f"Consider filtering outliers before modeling."
                            )
                    
                    # =========================================================
                    # CRITICAL: Apply Global Shift via Transform Gate
                    # =========================================================
                    # Use the Single Space Authority to ensure ALL data uses
                    # the SAME coordinate shift (prevents picking drift)
                    verts = self._to_local_precision(verts)
                    
                    # DIAGNOSTIC: Check coordinate range to verify local coordinates
                    if i == 0 and len(verts) > 0:
                        verts_min = np.min(verts, axis=0)
                        verts_max = np.max(verts, axis=0)
                        logger.info(f"DIAGNOSTIC: First surface '{name}' LOCAL vertex range (after shift):")
                        logger.info(f"  X: [{verts_min[0]:.2f}, {verts_max[0]:.2f}]")
                        logger.info(f"  Y: [{verts_min[1]:.2f}, {verts_max[1]:.2f}]")
                        logger.info(f"  Z: [{verts_min[2]:.2f}, {verts_max[2]:.2f}]")
                        
                        # Ideal local coordinates should be in the range of -5000 to +5000
                        local_range = max(
                            verts_max[0] - verts_min[0],
                            verts_max[1] - verts_min[1]
                        )
                        if local_range > 100000:
                            logger.warning(
                                f"LOCAL COORDINATE WARNING: Surface spans {local_range:.0f}m in local space. "
                                f"For optimal GPU precision, local coordinates should be < 10km from origin."
                            )
                    
                    if faces is None or len(faces) == 0:
                        logger.warning(f"Surface '{name}' has no faces, skipping")
                        continue
                    
                    faces = np.asarray(faces, dtype=np.int64)
                    
                    # Convert to PyVista format
                    if faces.ndim == 2 and faces.shape[1] == 3:
                        faces_pv = np.hstack([
                            np.full((len(faces), 1), 3, dtype=np.int64),
                            faces
                        ]).flatten()
                    else:
                        faces_pv = faces
                    
                    mesh = pv.PolyData(verts, faces_pv)
                    
                    # ================================================================
                    # EXCLUSIVE DOMAIN FIX: DISABLE REDUNDANT SMOOTHING
                    # ================================================================
                    # CRITICAL: Do NOT apply independent smoothing in the renderer.
                    # This preserves the exact mathematical alignment from ChronosEngine.
                    # surface.get('smoothed') will be True if Chronos handled it.
                    logger.debug(f"Geometric Exclusivity: Preserving engine geometry for '{name}'")
                    # mesh = mesh.smooth_taubin(...) <- REMOVED
                    
                    # Determine color
                    color = geology_colors[i % len(geology_colors)]
                    layer_name = f"GeoSurface: {name}"
                    
                    # ================================================================
                    # VOXEL-SHARPNESS SURFACE RENDERING (MATCHES UNIFIED MESH)
                    # ================================================================
                    # Use the SAME rendering settings as the unified mesh for consistency
                    # - High opacity for solid appearance
                    # - Flat shading (no smooth interpolation)
                    # - No color averaging (crisp boundaries)
                    # - High ambient (0.9) for vivid, static colors
                    actor = self.plotter.add_mesh(
                        mesh,
                        name=layer_name,
                        color=color,
                        opacity=surface_opacity,
                        show_edges=False,  # No edges for clean look
                        # ================================================================
                        # THE SHARPNESS SETTINGS (Eliminates blur and color bleeding)
                        # ================================================================
                        smooth_shading=False,         # FLAT shading - no blending at boundaries
                        interpolate_before_map=False, # CRITICAL: No color averaging
                        lighting=True,
                        ambient=0.9,   # HIGH ambient keeps colors vivid (SAME AS UNIFIED MESH)
                        diffuse=0.2,   # LOW diffuse - less view-angle variation (SAME AS UNIFIED MESH)
                        specular=0.0,  # No specular - pure solid color (SAME AS UNIFIED MESH)
                        # ================================================================
                        render=False,  # Defer render for batch processing
                        pbr=False,     # Disable PBR (causes flickering)
                    )
                    
                    # CRITICAL: Configure VTK properties for proper surface rendering
                    if actor is not None:
                        try:
                            # ========================================================
                            # FIX: DISABLE LOD (Level of Detail)
                            # ========================================================
                            # LOD optimization can delete small features at distance
                            if hasattr(actor, 'SetEnableLOD'):
                                actor.SetEnableLOD(0)
                            
                            prop = actor.GetProperty()
                            if prop:
                                # Disable backface culling - surfaces visible from all angles
                                prop.SetBackfaceCulling(0)
                                prop.SetFrontfaceCulling(0)
                                prop.SetOpacity(surface_opacity)
                                
                                # FLAT interpolation - safer for geological boundaries
                                prop.SetInterpolationToFlat()
                                
                                # For near-opaque surfaces, disable translucent rendering
                                if surface_opacity >= 0.9 and hasattr(prop, 'SetForceTranslucentOff'):
                                    prop.SetForceTranslucentOff()
                            
                            # ========================================================
                            # CASCADING POLYGON OFFSET (Layer Priority Fix)
                            # ========================================================
                            # Even with local origin shift, surfaces at same depth fight.
                            # SOLUTION: Force GPU to stack layers like a deck of cards.
                            # Higher index (younger stratigraphy) pushed toward camera.
                            # -2.0 multiplier ensures strong separation at all zoom levels.
                            mapper = actor.GetMapper()
                            if mapper:
                                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                                # GEOMETRIC SORTING APPLIED: 'i' is now truly Depth Rank
                                # i=0 is Top Surface (High Z)
                                # i=N is Bottom Surface (Low Z)
                                # Offset logic: Push deep layers AWAY from camera (Positive Offset).
                                offset = 1.0 * i
                                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0, offset)
                                # Use solid color, not scalars
                                mapper.SetScalarVisibility(False)
                            
                            # ========================================================
                            # FIX: Force Visibility and Pickability
                            # ========================================================
                            # PyVista/VTK may hide small actors or disable picking.
                            # Force both to ensure all geological features stay visible.
                            actor.SetVisibility(True)
                            actor.SetPickable(True)
                        except Exception as e:
                            logger.debug(f"Could not configure surface actor: {e}")
                    
                    # Register as layer in active_layers for Scene Inspector
                    self.add_layer(
                        layer_name,
                        actor,
                        data={'mesh': mesh, 'surface': surface, 'type': 'isosurface'},
                        layer_type='geology_surface',
                        opacity=surface_opacity
                    )
                    
                    surfaces_added += 1
                    logger.info(f"✓ Added surface '{name}' ({len(verts)} verts, {len(faces)} faces)")
                    
                except Exception as e:
                    logger.error(f"Failed to add surface '{name}': {e}")
        
        # 4. Plot Solid Volumes (3D voxel blocks) if mode includes solids
        # This uses COMPOSITE blending (not additive) for opaque solid appearance
        if render_mode in ("solids", "both"):
            solids_list = package.get('solids', [])
            if solids_list:
                solids_added = self._render_geology_solids(solids_list, geology_colors, render_mode)
        
        # Store solids data for volume queries even if not rendered
        self._geology_solids = package.get('solids', [])
        
        # 5. Compute combined bounds of all geology and set camera properly
        # This is CRITICAL - without proper camera setup, geology appears clipped/invisible
        self._setup_geology_camera_and_bounds()
        
        # 6. Force synchronous UI update and render (fixes "resize-to-view" bug)
        # The model is invisible until resize because VTK render window isn't notified
        # of new actors added from background threads. We force synchronous update here.
        # FIX: Reuse the robust main method which includes the 'Resize Toggle' fix
        self.force_render_update()
        
        # 7. Additional force repaint for guaranteed visibility
        self._force_repaint()
        
        # 8. SYNC LEGEND MANAGER WITH GEOLOGICAL UNITS
        # This ensures the legend shows the same colors as the 3D model
        self._sync_geology_legend(package, render_mode, geology_colors)
        
        # 9. FIX CAMERA PRECISION - DEPRECATED (Moved to _setup_geology_camera_and_bounds)
        # self._fix_camera_precision()
        
        # 10. Final processEvents to ensure legend is visible
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        # 11. Render Contact Points (combined with misfit coloring)
        contacts_added = 0
        report = package.get('report') or package.get('audit_report')
        if report is not None and hasattr(report, 'misfit_data'):
            contacts_added = self.render_geology_contacts(report)

        # 12. Log summary
        logger.info("=" * 60)
        logger.info(f"GEOLOGY PACKAGE LOADED: {surfaces_added} surfaces, {solids_added} solids, {contacts_added} contacts")
        
        # DEBUG: Log final state after loading
        logger.info(f"[GEOLOGY_LOAD] _global_shift after: {self._global_shift}")
        logger.info(f"[GEOLOGY_LOAD] Drillholes still in active_layers: {'drillholes' in self.active_layers}")
        logger.info(f"[GEOLOGY_LOAD] All active layers: {list(self.active_layers.keys())}")

        # Log audit status if available
        if report:
            status = getattr(report, 'status', 'Unknown')
            p90 = getattr(report, 'p90_error', 0)
            logger.info(f"AUDIT STATUS: {status} (P90: {p90:.2f}m)")
        
        logger.info("=" * 60)
    
    def clear_geology_layers(self) -> None:
        """
        Remove all geological layers from the scene.
        
        Clears both surfaces and solids to allow fresh geological model loading
        without overlap artifacts.
        """
        removed = 0
        prefixes = ("GeoSurface:", "GeoSolid:", "GeoUnified:", "GeoContacts", "GeoWireframe:", "geo_surface_", "geo_solid_", "geo_unified_")
        
        # DEBUG: Log all active layers before clearing
        logger.info(f"[CLEAR_GEOLOGY] Active layers before clear: {list(self.active_layers.keys())}")
        logger.info(f"[CLEAR_GEOLOGY] Drillholes present: {'drillholes' in self.active_layers}")
        
        for name in list(self.active_layers.keys()):
            if any(name.startswith(prefix) for prefix in prefixes):
                try:
                    self.clear_layer(name)
                    removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove layer '{name}': {e}")
        
        # DEBUG: Log active layers after clearing
        logger.info(f"[CLEAR_GEOLOGY] Active layers after clear: {list(self.active_layers.keys())}")
        logger.info(f"[CLEAR_GEOLOGY] Drillholes still present: {'drillholes' in self.active_layers}")
        
        # Clear stored geology data
        self._geology_bounds = None
        self._geology_solids = []
        # NOTE: Do NOT reset _local_origin or _global_shift here!
        # The coordinate transform must remain consistent across all data types
        # (drillholes, geology, blocks). Resetting it would cause coordinate
        # mismatch between existing drillholes and newly loaded geology.
        
        if removed > 0:
            logger.info(f"Cleared {removed} geological layers (surfaces + solids)")

    def render_geology_contacts(self, report) -> int:
        """
        Render contact points as color-coded spheres by residual error.

        Creates a GeoContacts layer with spheres at each contact point location.
        Color indicates the residual error (red = high error, green = low error).

        Args:
            report: AuditReport object with misfit_data DataFrame containing
                    X, Y, Z, residual_m, unit columns

        Returns:
            Number of contact points rendered
        """
        if self.plotter is None:
            logger.error("render_geology_contacts: Plotter not initialized")
            return 0

        if report is None:
            logger.warning("render_geology_contacts: No report provided")
            return 0

        # Get misfit_data DataFrame
        misfit_df = getattr(report, 'misfit_data', None)
        if misfit_df is None or len(misfit_df) == 0:
            logger.warning("render_geology_contacts: No misfit data in report")
            return 0

        try:
            # Extract coordinates
            required_cols = ['X', 'Y', 'Z', 'residual_m']
            if not all(col in misfit_df.columns for col in required_cols):
                logger.warning(f"render_geology_contacts: Missing columns. Has: {list(misfit_df.columns)}")
                return 0

            points = misfit_df[['X', 'Y', 'Z']].values.astype(np.float64)
            residuals = misfit_df['residual_m'].values.astype(np.float64)

            # Apply coordinate shift for GPU precision
            points = self._to_local_precision(points)

            # Create point cloud
            point_cloud = pv.PolyData(points)
            point_cloud['residual_m'] = residuals

            # Compute sphere radius based on model extent
            # Use 0.5% of the diagonal as sphere radius
            bounds = point_cloud.bounds
            diagonal = np.sqrt(
                (bounds[1] - bounds[0])**2 +
                (bounds[3] - bounds[2])**2 +
                (bounds[5] - bounds[4])**2
            )
            sphere_radius = max(diagonal * 0.005, 1.0)  # Minimum 1m radius

            # Create sphere glyphs
            sphere = pv.Sphere(radius=sphere_radius)
            glyphs = point_cloud.glyph(geom=sphere, scale=False, orient=False)

            # Copy scalars to glyphs
            glyphs['residual_m'] = np.repeat(residuals, sphere.n_cells)

            # Remove existing contacts layer
            if "GeoContacts" in self.active_layers:
                self.clear_layer("GeoContacts")

            # Add to scene with colormap (red = high error, green = low)
            actor = self.plotter.add_mesh(
                glyphs,
                name="GeoContacts",
                scalars='residual_m',
                cmap='RdYlGn_r',  # Red = high, Yellow = medium, Green = low
                opacity=0.9,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Misfit Error (m)',
                    'title_font_size': 12,
                    'label_font_size': 10,
                    'position_x': 0.85,
                    'position_y': 0.1,
                    'width': 0.1,
                    'height': 0.3,
                },
                smooth_shading=True,
                render=False
            )

            # Store report stats for UI display
            p90 = getattr(report, 'p90_error', 0)
            mean_err = getattr(report, 'mean_residual', 0)

            # Register layer
            self.add_layer(
                "GeoContacts",
                actor,
                data={
                    'points': points,
                    'residuals': residuals,
                    'count': len(points),
                    'p90_error': p90,
                    'mean_residual': mean_err,
                    'type': 'contacts'
                },
                layer_type='geology_contacts'
            )

            logger.info(f"✓ Rendered {len(points)} contact points (P90: {p90:.2f}m, Mean: {mean_err:.2f}m)")
            return len(points)

        except Exception as e:
            logger.error(f"Failed to render geology contacts: {e}", exc_info=True)
            return 0

    def _compute_local_origin(self, package: Dict[str, Any]) -> np.ndarray:
        """
        DEPRECATED: Use _to_local_precision() instead.
        
        This method is kept for backward compatibility but now delegates to
        the Single Space Authority (_to_local_precision).
        
        The new architecture locks the global shift on the FIRST dataset load,
        preventing coordinate drift and picking instability.
        
        Args:
            package: Dictionary containing surfaces and solids geometry
            
        Returns:
            np.ndarray: [x, y, z] center to subtract from all vertices
        """
        all_vertices = []
        
        # Collect vertices from surfaces
        for surface in package.get('surfaces', []):
            verts = surface.get('vertices')
            if verts is not None and len(verts) > 0:
                all_vertices.append(np.asarray(verts, dtype=np.float64))
        
        # Collect vertices from solids
        for solid in package.get('solids', []):
            verts = solid.get('vertices')
            if verts is not None and len(verts) > 0:
                all_vertices.append(np.asarray(verts, dtype=np.float64))
        
        if not all_vertices:
            logger.warning("No vertices found for local origin computation, using (0, 0, 0)")
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Compute the center of all geometry
        combined = np.vstack(all_vertices)
        
        # DELEGATE TO SINGLE SPACE AUTHORITY
        # This ensures the shift is locked on first call and reused for all subsequent data
        _ = self._to_local_precision(combined)  # Locks _global_shift if not set
        
        # Return the locked shift for backward compatibility
        return self._global_shift if self._global_shift is not None else np.mean(combined, axis=0)
    
    def get_world_coordinates(self, local_coords: np.ndarray) -> np.ndarray:
        """
        Convert local (GPU-shifted) coordinates back to real-world UTM coordinates.
        
        Use this for coordinate readouts to display original UTM values.
        
        Args:
            local_coords: Coordinates in local (shifted) space
            
        Returns:
            np.ndarray: Original UTM coordinates
        """
        # Use _global_shift (Single Space Authority)
        if self._global_shift is not None:
            return local_coords + self._global_shift
        # Fallback to legacy _local_origin for backward compatibility
        if self._local_origin is not None:
            return local_coords + self._local_origin
        return local_coords
    
    def get_local_coordinates(self, world_coords: np.ndarray) -> np.ndarray:
        """
        Convert real-world UTM coordinates to local (GPU-shifted) coordinates.
        
        Args:
            world_coords: Original UTM coordinates
            
        Returns:
            np.ndarray: Coordinates in local (shifted) space for GPU rendering
        """
        # Use _global_shift (Single Space Authority)
        if self._global_shift is not None:
            return world_coords - self._global_shift
        # Fallback to legacy _local_origin for backward compatibility
        if self._local_origin is not None:
            return world_coords - self._local_origin
        return world_coords
    
    def _apply_geology_z_fighting_fixes(self) -> None:
        """
        FIX: Balanced Depth Peeling + Polygon Offset for Sharp, Stable Rendering.
        
        PROBLEM:
        - Aggressive depth peeling (10 peels) caused fading artifacts during zoom
        - Disabling it entirely causes blurry/low-quality rendering
        
        SOLUTION:
        - Use CONSERVATIVE depth peeling (4 peels) for quality without overhead
        - Combine with PolygonOffset for coincident geometry
        - Enable multisampling for smooth, crisp edges
        - Use composite blending (not additive)
        
        This provides sharp rendering while preventing z-fighting and jitter.
        """
        try:
            renderer = self.plotter.renderer
            if renderer:
                # =====================================================
                # FIX 1: DISABLE Depth Peeling (Final)
                # =====================================================
                # Depth peeling causes artifacts with opaque solids.
                # Since we are enforcing 100% opacity, we don't need it.
                renderer.SetUseDepthPeeling(0)
                
                # =====================================================
                # FIX 2: Enable Two-Sided Lighting (Keep for illumination)
                # =====================================================
                # Ensures back-facing surfaces are illuminated correctly
                renderer.SetTwoSidedLighting(True)
                
                # =====================================================
                # FIX 3: Enhanced Global Polygon Offset (Final)
                # =====================================================
                # Use stronger offset parameters for better z-fighting prevention
                # Increased from (1.0, 1.0) to (2.0, 2.0) for large coordinate systems
                vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
                vtk.vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(2.0, 2.0)

                # Comprehensive diagnostic logging for z-fighting troubleshooting
                logger.info("=" * 60)
                logger.info("Z-FIGHTING DIAGNOSTIC REPORT")
                logger.info("=" * 60)
                logger.info(f"Depth Peeling: {renderer.GetUseDepthPeeling()} (0=DISABLED, 1=ENABLED)")
                logger.info(f"Two-Sided Lighting: {renderer.GetTwoSidedLighting()}")

                import vtk
                global_params = vtk.vtkMapper.GetResolveCoincidentTopologyPolygonOffsetParameters()
                logger.info(f"Global Polygon Offset: factor={global_params[0]}, units={global_params[1]}")

                camera = renderer.GetActiveCamera()
                clip_range = camera.GetClippingRange()
                clip_ratio = clip_range[1] / clip_range[0] if clip_range[0] > 0 else 0
                logger.info(f"Camera Clipping: near={clip_range[0]:.2f}m, far={clip_range[1]:.2f}m")
                logger.info(f"Clipping Ratio: {clip_ratio:.1f}:1")
                logger.info("=" * 60)

                logger.debug("Applied final z-fighting fixes: Depth Peeling OFF, Polygon Offset STRONG (2.0, 2.0)")
        except Exception as e:
            logger.debug(f"Could not apply z-fighting fixes: {e}")
    
    def _configure_geology_actor_properties(self, actor, layer_index: int) -> None:
        """
        Configure VTK actor properties for proper geological surface rendering.
        
        Applies per-actor settings to eliminate z-fighting and ensure correct
        visibility from all viewing angles.
        
        Args:
            actor: VTK actor to configure
            layer_index: Index used for unique polygon offset per layer
        """
        if actor is None:
            return
        
        try:
            prop = actor.GetProperty()
            if prop:
                # FIX: Disable backface culling - surfaces visible from ALL angles
                prop.SetBackfaceCulling(0)
                prop.SetFrontfaceCulling(0)
                
                # Ensure proper interpolation for smooth appearance
                prop.SetInterpolationToPhong()
            
            # FIX: Per-actor polygon offset to separate coincident surfaces
            mapper = actor.GetMapper()
            if mapper:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                # Each layer gets a unique offset to prevent z-fighting
                offset_factor = 1.0 + layer_index * 0.3
                offset_units = 1.0 + layer_index * 0.3
                mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(offset_factor, offset_units)
                
        except Exception as e:
            logger.debug(f"Could not configure actor properties: {e}")
    
    def _render_geology_solids(self, solids: List[Dict[str, Any]], colors: List[str], 
                               render_mode: str) -> int:
        """
        CP-GRADE SOLID VOLUME RENDERING (LoopStructural 1.6+ Compatible).
        
        CRITICAL FIXES FOR RENDERING ARTIFACTS:
        1. STOP ADDITIVE BLENDING - Uses opacity=1.0 with no transparency
        2. APPLY POLYGON OFFSETS - Stops Z-fighting/flickering
        3. COMPOSITE BLENDING - Prevents "fading" effect on overlapping solids
        
        This creates Leapfrog/Vulcan-style solid geological units where you
        see ONLY the frontmost unit, not overlapping transparent layers.
        
        Args:
            solids: List of solid dicts with vertices, faces, volume_m3
            colors: Color palette
            render_mode: "solids" or "both"
            
        Returns:
            Number of solids successfully rendered
        """
        solids_added = 0
        
        # CRITICAL FIX: For true solids, ALWAYS use FULL OPACITY (1.0)
        # Transparent solids invite depth peeling issues and driver timeouts.
        # Opaque solids are the industry-standard for stable block models.
        solid_opacity = 1.0
        
        for i, solid in enumerate(solids):
            try:
                unit_name = solid.get('unit_name', solid.get('name', f'Unit_{i}'))
                verts = solid.get('vertices')
                faces = solid.get('faces')
                
                if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
                    logger.warning(f"Solid '{unit_name}' has no geometry, skipping render")
                    continue
                
                verts = np.asarray(verts, dtype=np.float64)
                faces = np.asarray(faces, dtype=np.int64)
                
                # =========================================================
                # CRITICAL: Apply Global Shift via Transform Gate
                # =========================================================
                # Use the Single Space Authority to ensure ALL data uses
                # the SAME coordinate shift (prevents picking drift)
                verts = self._to_local_precision(verts)
                
                # Convert to PyVista format
                if faces.ndim == 2 and faces.shape[1] == 3:
                    faces_pv = np.hstack([
                        np.full((len(faces), 1), 3, dtype=np.int64),
                        faces
                    ]).flatten()
                else:
                    faces_pv = faces
                
                mesh = pv.PolyData(verts, faces_pv)
                
                # EXCLUSIVE DOMAIN FIX: DISABLE REDUNDANT SMOOTHING
                # ================================================================
                # CRITICAL: Do NOT apply independent smoothing in the renderer.
                # This preserves the exact mathematical alignment from ChronosEngine.
                logger.debug(f"Geometric Exclusivity: Preserving engine geometry for solid '{unit_name}'")
                # mesh = mesh.smooth_taubin(...) <- REMOVED
                
                # Fill holes to ensure closed manifold (looks like a solid block)
                try:
                    if hasattr(mesh, 'fill_holes'):
                        mesh = mesh.fill_holes(hole_size=1000)
                except Exception:
                    pass
                
                # ================================================================
                # FIX: COMPUTE NORMALS WITH AUTO-ORIENTATION (CRITICAL FOR VISIBILITY)
                # ================================================================
                # This prevents "disappearing islands" at far zoom distances
                # auto_orient_normals ensures surfaces point outward consistently
                # Without this, surfaces can vanish when viewed from certain angles/distances
                try:
                    mesh.compute_normals(
                        cell_normals=True,
                        point_normals=True,
                        inplace=True,
                        auto_orient_normals=True,
                        consistent_normals=True
                    )
                    logger.debug(f"Computed normals for solid '{unit_name}'")
                except Exception as e:
                    logger.debug(f"Could not compute normals: {e}")
                
                color = colors[i % len(colors)]
                layer_name = f"GeoSolid: {unit_name}"
                
                # ================================================================
                # VOXEL-SHARPNESS SOLID RENDERING (MATCHES UNIFIED MESH)
                # ================================================================
                # Use the SAME rendering settings as the unified mesh for consistency
                # - opacity=1.0 means NO TRANSPARENCY (stops "fading")
                # - interpolate_before_map=False: CRITICAL - no color averaging
                # - smooth_shading=False: Flat shading for sharp boundaries
                # - High ambient (0.9) ensures vivid, solid colors
                actor = self.plotter.add_mesh(
                    mesh,
                    name=layer_name,
                    color=color,
                    opacity=solid_opacity,
                    show_edges=False,
                    # ================================================================
                    # THE SHARPNESS SETTINGS (Eliminates blur and color bleeding)
                    # ================================================================
                    smooth_shading=False,         # FLAT shading - no blending at boundaries
                    interpolate_before_map=False, # CRITICAL: No color averaging
                    lighting=True,
                    ambient=0.9,   # HIGH ambient keeps colors vivid (SAME AS UNIFIED MESH)
                    diffuse=0.2,   # LOW diffuse - less view-angle variation (SAME AS UNIFIED MESH)
                    specular=0.0,  # No specular - pure solid color (SAME AS UNIFIED MESH)
                    # ================================================================
                    render=False,  # Batch updates
                    pbr=False,     # Disable PBR (causes flickering)
                )
                
                # ================================================================
                # CRITICAL FIX: VTK ACTOR CONFIGURATION FOR Z-FIGHTING PREVENTION
                # ================================================================
                if actor is not None:
                    try:
                        # ========================================================
                        # FIX: DISABLE LOD (Level of Detail)
                        # ========================================================
                        # LOD optimization can delete small features at distance
                        # Force full resolution rendering ALWAYS
                        if hasattr(actor, 'SetEnableLOD'):
                            actor.SetEnableLOD(0)
                        
                        prop = actor.GetProperty()
                        if prop:
                            # FIX: RE-ENABLE BACKFACE CULLING
                            # Opaque solids sharing a boundary have 2 surfaces (Unit A Front, Unit B Back).
                            # We MUST cull backfaces to see only the surface facing the camera.
                            prop.SetBackfaceCulling(1)
                            prop.SetFrontfaceCulling(0)
                            
                            # Force OPAQUE rendering (no transparency sorting issues)
                            prop.SetOpacity(solid_opacity)
                            
                            # FLAT interpolation - safer for geological boundaries
                            # Prevents color bleeding at small feature edges
                            prop.SetInterpolationToFlat()
                        
                        # ========================================================
                        # CASCADING POLYGON OFFSET (Layer Priority Fix)
                        # ========================================================
                        # FIX: DO NOT APPLY POLYGON OFFSET TO SOLIDS
                        # Applying offset to a closed volume pushes the back faces forward,
                        # causing them to Z-fight with the front faces or internal structure.
                        # Polygon Offset should ONLY be used for coplanar sheets (Surfaces).
                        mapper = actor.GetMapper()
                        if mapper:
                            # Disable offsets for solids
                            mapper.SetResolveCoincidentTopologyToDefault()
                            mapper.SetScalarVisibility(False)
                        
                        # ========================================================
                        # FIX: Set Opaque Rendering Mode via VTK
                        # ========================================================
                        # This prevents the "faded/washed-out" appearance AND bypasses Depth Peeling
                        # For fully opaque solids, disable translucent rendering
                        if prop:
                            prop.SetOpacity(1.0) # Ensure strictly 1.0
                            if hasattr(prop, 'SetForceTranslucentOff'):
                                # This is critical: Tells VTK "This is opaque, don't peel it"
                                prop.SetForceTranslucentOff()
                        
                        # ========================================================
                        # FIX: Force Visibility and Pickability
                        # ========================================================
                        # PyVista/VTK may hide small actors or disable picking.
                        # Force both to ensure all geological features stay visible.
                        actor.SetVisibility(True)
                        actor.SetPickable(True)
                            
                    except Exception as e:
                        logger.debug(f"Could not configure actor properties: {e}")
                
                # Register layer
                volume_m3 = solid.get('volume_m3', 0) or 0
                self.add_layer(
                    layer_name,
                    actor,
                    data={'mesh': mesh, 'solid': solid, 'type': 'volume', 'volume_m3': volume_m3},
                    layer_type='geology_solid',
                    opacity=solid_opacity
                )
                
                solids_added += 1
                logger.info(f"✓ Added OPAQUE solid '{unit_name}' ({len(verts)} verts, Vol: {volume_m3:,.0f} m³)")
                
            except Exception as e:
                logger.error(f"Failed to add solid '{unit_name}': {e}")
        
        return solids_added
    
    def _render_unified_geology_mesh(self, unified_mesh: Dict[str, Any], package: Dict[str, Any], render_mode: str = "solids") -> bool:
        """
        INDUSTRY-STANDARD RENDERING: Single mesh with Formation_ID (Leapfrog/Micromine style).
        
        This is the DEFINITIVE FIX for Z-fighting and disappearing features.
        Instead of rendering N overlapping meshes, we render ONE mesh where
        every voxel has a discrete Formation_ID. The GPU colors this using
        a categorical colormap - no overlap means no Z-fighting.
        
        Args:
            unified_mesh: Dict from ChronosEngine.extract_unified_geology_mesh()
            package: Full geology package (for color palette)
            render_mode: "surfaces", "solids", or "both"
            
        Returns:
            True if successful, False to fall back to multi-mesh mode
        """
        if self.plotter is None:
            return False
        
        logger.info("=" * 60)
        logger.info(f"RENDERING UNIFIED PARTITION MESH (Mode: {render_mode})")
        logger.info("Shared scalar field extraction - Exclusivity Guaranteed")
        logger.info("=" * 60)
        
        try:
            # Clear any existing geology first
            self.clear_geology_layers()
            
            # Get PyVista grid from unified mesh
            pv_grid = unified_mesh.get('_pyvista_grid')
            formation_names = unified_mesh.get('formation_names', {})
            n_units = unified_mesh.get('n_units', 0)
            is_world_coords = unified_mesh.get('_is_world_coordinates', False)
            
            if pv_grid is None:
                logger.error("Unified mesh has no PyVista grid - falling back to multi-mesh")
                return False
            
            # ================================================================
            # Step 1: Apply Local Origin Shift and Cast to UnstructuredGrid
            # CRITICAL FIX: Use _to_local_precision() to ensure the SAME
            # coordinate shift is used for geology as for drillholes.
            # ================================================================
            import pyvista as pv
            unstructured = pv_grid.cast_to_unstructured_grid()
            
            # ================================================================
            # SIMPLIFIED TRANSFORMATION LOGIC
            # ================================================================
            # The grid from GeoXIndustryModeler is now GUARANTEED to be in
            # World Coordinates (UTM). We only need to apply the Scene's
            # Global Shift to align it with the drillholes.
            # ================================================================
            
            # Verify the grid was flagged as world coordinates
            if not is_world_coords:
                logger.warning("[GEOLOGY RENDER] Grid NOT flagged as world coordinates! May need transformation.")
            
            # 1. Get points from the grid (should be World Coordinates)
            world_points = np.asarray(unstructured.points, dtype=np.float64)
            
            # 2. LOGGING: Check the bounds in the console/terminal
            b_min = world_points.min(axis=0)
            b_max = world_points.max(axis=0)
            logger.info(f"[GEOLOGY RENDER] Grid Bounds: X[{b_min[0]:.1f}, {b_max[0]:.1f}] Y[{b_min[1]:.1f}, {b_max[1]:.1f}] Z[{b_min[2]:.1f}, {b_max[2]:.1f}]")
            logger.info(f"[GEOLOGY RENDER] is_world_coordinates flag: {is_world_coords}")
            
            # Sanity check: if bounds are still small, warn about potential issue
            abs_max = np.max(np.abs(world_points))
            if abs_max < 1000:
                logger.error(
                    f"[GEOLOGY RENDER] COORDINATE BUG: Grid bounds too small (abs_max={abs_max:.2f}). "
                    f"Expected UTM-scale (>10000). Check industry_modeler._model_to_world!"
                )
            
            # 3. Apply the Global Shift (centers the model in the viewer)
            local_points = self._to_local_precision(world_points)
            logger.info(f"[GEOLOGY DEBUG] After _to_local_precision: min={local_points.min(axis=0)}, max={local_points.max(axis=0)}")
            logger.info(f"[GEOLOGY DEBUG] _global_shift = {self._global_shift}")
            
            # Update _local_origin for backward compatibility
            if self._global_shift is not None:
                self._local_origin = self._global_shift
            
            shifted_grid = unstructured.copy()
            shifted_grid.points = local_points
            
            logger.info(f"Applied coordinate shift to unified mesh: shift=[{self._global_shift[0]:.2f}, {self._global_shift[1]:.2f}, {self._global_shift[2]:.2f}]" if self._global_shift is not None else "Applied coordinate shift to unified mesh")
            
            # ================================================================
            # Step 2: Extract Geometry Based on Render Mode
            # ================================================================
            # "surfaces" mode: Per-formation colored surfaces (TRUE SURFACES)
            # "solids" mode: Hull with Formation_ID colormap (block model view)
            # "both": Both hull and per-formation surfaces
            
            boundaries = unified_mesh.get('boundaries', [])
            hull = None
            contacts = None
            per_formation_surfaces = []
            
            # ================================================================
            # SURFACES MODE: Extract each formation's surface separately
            # This gives proper colored surfaces, not just the unified hull
            # ================================================================
            if render_mode in ("surfaces", "both"):
                logger.info(f"Extracting {n_units} per-formation surfaces for surfaces mode...")
                for fid in range(n_units):
                    try:
                        # Threshold to extract just this formation's cells
                        formation_vol = shifted_grid.threshold(
                            value=[fid - 0.5, fid + 0.5],
                            scalars="Formation_ID"
                        )
                        if formation_vol and formation_vol.n_cells > 0:
                            formation_surface = formation_vol.extract_surface()
                            if formation_surface and formation_surface.n_cells > 0:
                                per_formation_surfaces.append({
                                    'fid': fid,
                                    'name': formation_names.get(fid, f"Unit_{fid}"),
                                    'mesh': formation_surface
                                })
                                logger.debug(f"Formation {fid} surface: {formation_surface.n_cells} faces")
                    except Exception as e:
                        logger.warning(f"Failed to extract surface for formation {fid}: {e}")
                logger.info(f"Extracted {len(per_formation_surfaces)} per-formation surfaces")
            
            # ================================================================
            # SOLIDS MODE: Extract hull (outer skin of entire model)
            # ================================================================
            if render_mode in ("solids", "both", "unified"):
                hull = shifted_grid.extract_surface()
                logger.info(f"Extracted hull: {hull.n_faces} faces")
            
            # ================================================================
            # CONTACTS: Extract internal boundaries between formations
            # FIXED: Validate boundaries are within scalar field range
            # ================================================================
            if boundaries and render_mode in ("surfaces", "both", "contacts"):
                try:
                    # Get actual scalar field range from cell data
                    if 'scalar' in shifted_grid.cell_data:
                        field_values = shifted_grid.cell_data['scalar']
                        field_min = float(np.nanmin(field_values))
                        field_max = float(np.nanmax(field_values))
                        
                        # Filter boundaries to those within the actual field range
                        valid_boundaries = [b for b in boundaries if field_min <= b <= field_max]
                        
                        if valid_boundaries:
                            logger.info(f"Extracting contacts at {len(valid_boundaries)} boundaries: {valid_boundaries}")
                            # Use point data for contour (ctp = cell to point)
                            point_grid = shifted_grid.cell_data_to_point_data()
                            contacts = point_grid.contour(isosurfaces=valid_boundaries, scalars="scalar")
                            logger.info(f"Extracted {contacts.n_cells if contacts else 0} contact faces")
                        else:
                            logger.warning(f"No boundaries within field range [{field_min:.2f}, {field_max:.2f}]")
                    else:
                        logger.warning("No 'scalar' field in cell_data for contact extraction")
                except Exception as e:
                    logger.warning(f"Contact extraction failed: {e}")
            
            logger.info(f"Geometry Extraction: Hull={hull.n_faces if hull else 0} faces, "
                       f"Contacts={contacts.n_faces if contacts else 0} faces, "
                       f"Per-formation={len(per_formation_surfaces)} surfaces")

            # ================================================================
            # GPU-SAFE MODE: Check for large geology meshes
            # ================================================================
            # Same threshold as block models - prevent driver timeout (TDR)
            LARGE_MODEL_THRESHOLD = 25000
            total_faces = (hull.n_faces if hull else 0) + sum(s['mesh'].n_cells for s in per_formation_surfaces)
            total_cells = shifted_grid.n_cells if shifted_grid else 0

            if total_cells > LARGE_MODEL_THRESHOLD or total_faces > LARGE_MODEL_THRESHOLD:
                logger.warning(f"[GEOLOGY RENDER] Large model detected ({total_cells:,} cells, {total_faces:,} faces) - enabling GPU-safe mode")
                self._has_large_model = True
                if self._resize_debounce is not None:
                    self._resize_debounce.activate()

            # ================================================================
            # Step 3: Configure Colors and Rendering
            # ================================================================
            geology_colors = [
                '#e74c3c', '#3498db', '#2ecc71', '#f39c12', 
                '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
            ]
            from matplotlib.colors import ListedColormap
            discrete_cmap = ListedColormap([geology_colors[i % len(geology_colors)] for i in range(n_units)])
            
            # ================================================================
            # Render Per-Formation Surfaces (TRUE SURFACES - colored by formation)
            # ================================================================
            if per_formation_surfaces:
                logger.info(f"Rendering {len(per_formation_surfaces)} per-formation surfaces...")
                for i, surf_data in enumerate(per_formation_surfaces):
                    fid = surf_data['fid']
                    name = surf_data['name']
                    mesh = surf_data['mesh']
                    color = geology_colors[fid % len(geology_colors)]
                    
                    layer_name = f"GeoSurface:{name}"
                    surf_actor = self.plotter.add_mesh(
                        mesh, name=layer_name, color=color,
                        opacity=1.0, smooth_shading=True,
                        lighting=True, ambient=0.6, diffuse=0.4,
                        render=False, pbr=False
                    )
                    if surf_actor:
                        # Apply polygon offset to prevent z-fighting between formations
                        mapper = surf_actor.GetMapper()
                        if mapper:
                            mapper.SetResolveCoincidentTopologyToPolygonOffset()
                            # Use formation index for offset to ensure proper stacking
                            mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0 + fid * 0.5, 1.0 + fid * 0.5)
                        self.add_layer(layer_name, surf_actor, data={'mesh': mesh, 'unit_name': name}, layer_type='geology_surface')
                        logger.debug(f"Added formation surface: {name} (color={color})")
            
            # ================================================================
            # Render Hull (Solid Formation View - block model style)
            # ================================================================
            if hull is not None and hull.n_cells > 0:
                hull_layer = "GeoUnified: Hull"
                hull_visible = render_mode in ("solids", "both", "unified")
                hull_actor = self.plotter.add_mesh(
                    hull, name=hull_layer, scalars="Formation_ID", cmap=discrete_cmap,
                    preference='cell', categories=True, smooth_shading=False,
                    opacity=1.0 if hull_visible else 0.0, 
                    lighting=True, ambient=0.9, diffuse=0.2,
                    render=False, pbr=False
                )
                if hull_actor:
                    # Set visibility after actor is created (PyVista API compatibility)
                    hull_actor.SetVisibility(hull_visible)
                    hull_actor.GetProperty().SetInterpolationToFlat()

                    # Apply polygon offset to unified mesh hull to prevent z-fighting
                    mapper = hull_actor.GetMapper()
                    if mapper:
                        mapper.SetResolveCoincidentTopologyToPolygonOffset()
                        mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(2.0, 2.0)
                        logger.debug("Applied polygon offset (2.0, 2.0) to unified hull mesh")

                    self.add_layer(hull_layer, hull_actor, data={'mesh': hull}, layer_type='geology_solid')
            
            # ================================================================
            # Render Shared Contacts (formation boundaries as colored surfaces)
            # ================================================================
            if contacts is not None and contacts.n_cells > 0:
                contact_layer = "GeoUnified: Contacts"
                contact_visible = render_mode in ("surfaces", "both", "contacts")
                # Render contacts with a distinct color (gold/yellow for visibility)
                contact_actor = self.plotter.add_mesh(
                    contacts, name=contact_layer, color="#FFD700",  # Gold color for contacts
                    opacity=1.0 if contact_visible else 0.0,
                    smooth_shading=True,
                    line_width=1,  # SAFE: Use 1.0 for driver compatibility
                    ambient=0.9, diffuse=0.1, render=False
                )
                if contact_actor:
                    # Contact surfaces need stronger offset since they overlay formations
                    mapper = contact_actor.GetMapper()
                    if mapper:
                        mapper.SetResolveCoincidentTopologyToPolygonOffset()
                        mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(4.0, 4.0)
                        logger.debug("Applied strong polygon offset (4.0, 4.0) to contact surfaces")

                    self.add_layer(contact_layer, contact_actor, data={'mesh': contacts}, layer_type='geology_contact')
                    logger.info(f"Added contact layer with {contacts.n_cells} faces")
                
            # Update bounds and camera
            self._geology_bounds = shifted_grid.bounds
            self._sync_unified_mesh_legend(unified_mesh, geology_colors)
            
            logger.info(f"UNIFIED MESH LOADED: {n_units} formations, Mode={render_mode}.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to render unified mesh: {e}", exc_info=True)
            return False
    
    def _sync_unified_mesh_legend(self, unified_mesh: Dict[str, Any], colors: List[str]) -> None:
        """
        Sync legend manager with unified mesh formation names and colors.
        
        Args:
            unified_mesh: Dict from ChronosEngine.extract_unified_geology_mesh()
            colors: Color palette
        """
        try:
            if not hasattr(self, 'legend_manager') or self.legend_manager is None:
                return
            
            formation_names = unified_mesh.get('formation_names', {})
            n_units = unified_mesh.get('n_units', 0)
            
            # Create legend entries
            legend_data = {}
            for fid in range(n_units):
                name = formation_names.get(fid, f"Unit_{fid}")
                color = colors[fid % len(colors)]
                legend_data[name] = color
            
            # Update legend manager
            if hasattr(self.legend_manager, 'update_categorical'):
                self.legend_manager.update_categorical(
                    property_name="Formation",
                    categories=legend_data
                )
                logger.info(f"Legend updated with {n_units} formations")
            
        except Exception as e:
            logger.debug(f"Could not sync unified mesh legend: {e}")
    
    def _setup_geology_camera_and_bounds(self) -> None:
        """
        Compute combined bounds of all visible layers and configure camera.
        
        This is CRITICAL - without proper camera setup, layers appear 
        clipped or invisible. Computes bounds from geology, drillholes,
        and block model layers to set appropriate clipping planes.
        
        FIXED: Now includes drillhole and block model bounds to prevent
        coordinate mismatch causing both geology and drillholes to disappear.
        """
        try:
            all_bounds = []
            
            # Collect bounds from all geology layers (including unified mesh)
            for layer_name, layer_info in self.active_layers.items():
                if layer_name.startswith("GeoSurface:") or layer_name.startswith("GeoSolid:") or layer_name.startswith("GeoUnified:"):
                    layer_data = layer_info.get('data', {})
                    if isinstance(layer_data, dict):
                        mesh = layer_data.get('mesh')
                    else:
                        mesh = layer_data
                    
                    if mesh is not None and hasattr(mesh, 'bounds'):
                        all_bounds.append(mesh.bounds)
                        logger.debug(f"Layer '{layer_name}' bounds: {mesh.bounds}")
            
            # CRITICAL FIX: Also include drillhole bounds for combined camera setup
            # This ensures drillholes and geology share the same clipping range
            if "drillholes" in self.active_layers:
                dh_info = self.active_layers.get("drillholes", {})
                dh_data = dh_info.get('data', {})
                if isinstance(dh_data, dict):
                    hole_polys = dh_data.get('hole_polys', {})
                    for hid, poly in hole_polys.items():
                        if poly is not None and hasattr(poly, 'bounds') and poly.n_points > 0:
                            all_bounds.append(poly.bounds)
                    if hole_polys:
                        logger.debug(f"Included {len(hole_polys)} drillhole bounds in camera setup")
            
            # Also check for drillhole hole actors bounds as fallback
            if hasattr(self, '_drillhole_hole_actors') and self._drillhole_hole_actors:
                for hid, actor in self._drillhole_hole_actors.items():
                    if actor is not None and hasattr(actor, 'GetBounds'):
                        bounds = actor.GetBounds()
                        if bounds and bounds[0] < bounds[1]:  # Valid bounds
                            all_bounds.append(bounds)
            
            if not all_bounds:
                logger.warning("No layer bounds found for camera setup!")
                return
            
            # Compute combined bounds
            xmin = min(b[0] for b in all_bounds)
            xmax = max(b[1] for b in all_bounds)
            ymin = min(b[2] for b in all_bounds)
            ymax = max(b[3] for b in all_bounds)
            zmin = min(b[4] for b in all_bounds)
            zmax = max(b[5] for b in all_bounds)
            
            geo_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
            self._geology_bounds = geo_bounds
            
            logger.info(f"Geology bounds: X[{xmin:.0f}-{xmax:.0f}] Y[{ymin:.0f}-{ymax:.0f}] Z[{zmin:.0f}-{zmax:.0f}]")
            
            # Calculate isometric camera position
            center = [
                (xmin + xmax) / 2,
                (ymin + ymax) / 2,
                (zmin + zmax) / 2
            ]
            
            size = max(xmax - xmin, ymax - ymin, zmax - zmin)
            distance = size * 2.5
            
            position = (
                center[0] + distance * 0.7,
                center[1] + distance * 0.7,
                center[2] + distance * 0.7
            )
            focal_point = tuple(center)
            up = (0, 0, 1)
            
            # Set camera via VTK (most reliable)
            camera = self.plotter.renderer.GetActiveCamera()
            if camera:
                camera.SetPosition(*position)
                camera.SetFocalPoint(*focal_point)
                camera.SetViewUp(*up)
                
                # Set proper clipping range
                # FIX: TIGHTER CLIPPING PLANES FOR Z-PRECISION
                # Use a much tighter fit to maximize depth buffer resolution (24-bit)
                # This is CRITICAL for preventing Z-fighting on coplanar surfaces.
                
                # Near plane: Move as close to object as possible without clipping
                near_plane = max(distance * 0.01, 0.1)  # Was size * 0.001
                
                # Far plane: Just enough to cover the object + margin
                far_plane = distance + size * 2.0      # Was size * 50.0 (!?)
                
                camera.SetClippingRange(near_plane, far_plane)
                logger.debug(f"High-Precision Clipping: Near={near_plane:.2f}, Far={far_plane:.2f}")
            
                camera.SetClippingRange(near_plane, far_plane)
                logger.debug(f"High-Precision Clipping: Near={near_plane:.2f}, Far={far_plane:.2f}")
            
            # REMOVED: self.plotter.reset_camera()
            # This was DESTROYING the precision clipping planes we just calculated above.
            # We must trust our manual calculations.
            
        except Exception as e:
            logger.warning(f"Could not set geology camera bounds: {e}")
    
    def _force_geology_render_update(self) -> None:
        """
        CP-GRADE FIX: Force synchronous UI update after geology loading.

        FIXES THE "RESIZE-TO-VIEW" BUG:
        The model is invisible after building until window resize because
        VTK render window isn't notified of new actors added from a background
        thread. We force synchronous render, repaint, and Qt event processing.

        This is a VTK threading issue - the Plotter is not "Modified" when
        the background thread finishes. We need to force a synchronous repaint.
        """
        try:
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QTimer

            if self.plotter is None:
                return

            # GPU-SAFE MODE: For large models, use single render to prevent driver timeout
            if getattr(self, '_has_large_model', False):
                logger.info("_force_geology_render_update: Using GPU-safe single render")
                self.plotter.render()
                return

            # =====================================================
            # Step 1: Force VTK render (updates internal state)
            # =====================================================
            self.plotter.render()
            
            # Force render window update - CRITICAL for background thread visibility
            if self.plotter.render_window:
                self.plotter.render_window.Render()
                # Modified() notifies VTK that the scene changed
                self.plotter.render_window.Modified()
            
            # =====================================================
            # FINAL: Safe, Single-Pass Update
            # =====================================================
            # 1. Notify VTK that data changed
            if hasattr(self.plotter.render_window, 'Modified'):
                self.plotter.render_window.Modified()
            
            # 2. Render VTK (Internal State Update)
            self.plotter.render()
            
            # 3. Process Qt Events once to clear pending paints
            QApplication.processEvents()
            
            # 4. Final Render to screen
            if self.plotter.render_window:
                self.plotter.render_window.Render()
            
            # 5. Fit to view if requested (safe)
            if hasattr(self, 'fit_to_view'):
                try:
                    self.fit_to_view()
                    self.plotter.render()
                except Exception:
                    pass
            
            # 6. Repaint Qt Widget (safe)
            if hasattr(self.plotter, 'interactor') and self.plotter.interactor:
                self.plotter.interactor.update()
                
            logger.info("force_render_update: Complete (Safe Mode)")
            
            logger.info("Forced synchronous render update complete")
            
        except Exception as e:
            logger.debug(f"Force render update: {e}")
    
    def _apply_geology_polygon_offset(self) -> None:
        """
        DEPRECATED: Use _apply_geology_z_fighting_fixes() instead.
        Kept for backward compatibility.
        """
        self._apply_geology_z_fighting_fixes()
    
    def _fix_camera_precision(self) -> None:
        """
        CP-GRADE FIX: Neutralized to prevent conflict with _setup_geology_camera_and_bounds.
        """
        # Delegating to the primary camera setup method
        self._setup_geology_camera_and_bounds()
    
    def _force_repaint(self) -> None:
        """
        CP-GRADE FIX: Synchronously forces the Qt Interactor to redraw the scene.

        This method should be called at the end of every geology build to ensure
        the model is visible immediately without requiring a window resize.

        Addresses the "Invisible until Resize" bug where VTK actors added in
        button callbacks don't trigger automatic repaints.
        """
        from PyQt6.QtWidgets import QApplication

        if self.plotter is None:
            return

        # GPU-SAFE MODE: For large models, use single render to prevent driver timeout
        if getattr(self, '_has_large_model', False):
            try:
                self.plotter.render()
            except Exception:
                pass
            return

        try:
            # 1. Update the VTK Pipeline
            self.plotter.render()
            
            # 2. Force render window notification
            if self.plotter.render_window:
                self.plotter.render_window.Render()
                self.plotter.render_window.Modified()
            
            # 3. Force the Qt Widget to repaint its frame
            if hasattr(self.plotter, 'interactor') and self.plotter.interactor:
                self.plotter.interactor.repaint()
                self.plotter.interactor.update()
            
            # 4. Flush the application event loop
            QApplication.processEvents()
            
            # 5. Reset camera if this is the first geological load
            if not hasattr(self, '_geology_camera_positioned') or not self._geology_camera_positioned:
                self.fit_to_view()
                self._geology_camera_positioned = True
            
            logger.debug("Force repaint completed")
            
        except Exception as e:
            logger.debug(f"Force repaint: {e}")
    
    def force_render_update(self) -> None:
        """
        PUBLIC API: Force synchronous UI update after adding meshes/actors.

        FIXES THE "INVISIBLE UNTIL RESIZE" BUG:
        When meshes are added to the VTK scene, the Qt widget doesn't automatically
        repaint to show the new content. This method forces VTK to recognize
        the render window size and re-render.
        """
        try:
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QTimer

            if self.plotter is None:
                return

            # GPU-SAFE MODE: For large models, use single render to prevent driver timeout (TDR)
            # The full force_render_update does 10+ render calls which can crash AMD/NVIDIA drivers
            # CRITICAL: Hard gate at 100k cells - NEVER do SetSize toggle trick for large models
            n_cells = 0
            if self.current_model:
                n_cells = self.current_model.block_count

            # NEW HARD GATE: If we have a large model, NEVER do the SetSize/toggle trick
            # It's better to have a 1-frame delay in visibility than a driver crash
            is_heavy = n_cells > 50000 or getattr(self, '_has_large_model', False)

            if is_heavy:
                logger.info(f"force_render_update: Using GPU-safe single render ({n_cells:,} cells)")
                try:
                    self.plotter.render()
                except Exception as e:
                    logger.debug(f"GPU-safe render: {e}")
                return

            interactor = getattr(self.plotter, 'interactor', None)
            render_window = self.plotter.render_window

            # Get the actual widget size
            w, h = 0, 0
            if interactor is not None:
                size = interactor.size()
                w, h = size.width(), size.height()

            # CRITICAL: Tell VTK render window about its actual size
            # This is what happens internally during a real resize
            if render_window is not None and w > 0 and h > 0:
                # SetSize triggers VTK's internal resize handling
                # FIX: Only set size if it actually changed to prevent recursive shrinking loop
                # CRITICAL: Must account for devicePixelRatio (HighDPI) to avoid shrinking loop
                dpr = interactor.devicePixelRatio()
                target_w = int(w * dpr)
                target_h = int(h * dpr)

                current_size = render_window.GetSize()
                # Check for mismatch (allow small tolerance just in case)
                if abs(current_size[0] - target_w) > 2 or abs(current_size[1] - target_h) > 2:
                    logger.info(f"Correcting VTK window size: {current_size} -> {target_w}x{target_h} (DPR={dpr})")
                    render_window.SetSize(target_w, target_h)
                else:
                    # FORCE WAKEUP: If size is already correct, toggle it by 1 pixel to force a repaint
                    # This simulates a manual resize which proves to fix the "invisible render" issue exhaustively
                    # NOTE: This is ONLY for small models (<50k cells) - large models skip this entirely above
                    logger.debug("Toggling VTK window size to force repaint")
                    render_window.SetSize(target_w - 1, target_h)
                    render_window.Render()
                    render_window.SetSize(target_w, target_h)
                
                # Invoke ConfigureEvent - this is what VTK fires on resize
                if hasattr(render_window, 'GetInteractor') and render_window.GetInteractor():
                    vtk_interactor = render_window.GetInteractor()
                    vtk_interactor.ConfigureEvent()
                    vtk_interactor.Render()
                
                render_window.Render()
                render_window.Modified()
            
            # 6. Repaint Qt Widget (safe)
            if hasattr(self.plotter, 'interactor') and self.plotter.interactor:
                self.plotter.interactor.update()
            
            # Render
            self.plotter.render()

            # Force Qt repaint
            if interactor is not None:
                interactor.repaint()
                interactor.update()

            # CRITICAL: Skip processEvents for massive grids (>50k cells)
            # processEvents during resize creates recursion loop that crashes GPU drivers
            if n_cells <= 50000 and not getattr(self, '_has_large_model', False):
                QApplication.processEvents()

            # Final render (only for small models)
            if n_cells <= 50000:
                self.plotter.render()

            # Schedule delayed refreshes - ONLY for small scenes to prevent GPU timeout
            # For large models, skip delayed renders entirely
            if not getattr(self, '_has_large_model', False) and n_cells <= 50000:
                def delayed_refresh():
                    try:
                        if self.plotter and self.plotter.render_window:
                            self.plotter.render_window.Render()
                            self.plotter.render()
                            if hasattr(self.plotter, 'interactor') and self.plotter.interactor:
                                self.plotter.interactor.repaint()
                            QApplication.processEvents()
                    except Exception:
                        pass

                QTimer.singleShot(100, delayed_refresh)
                QTimer.singleShot(300, delayed_refresh)
            else:
                logger.info("force_render_update: Skipped delayed refreshes (large model loaded)")

            logger.info(f"force_render_update: Complete (VTK SetSize {w}x{h})")
            
        except Exception as e:
            logger.warning(f"force_render_update error: {e}")
    
    def _sync_geology_legend(
        self, 
        package: Dict[str, Any], 
        render_mode: str, 
        geology_colors: List[str]
    ) -> None:
        """
        Sync the legend manager with geological unit colors.
        
        This ensures the legend immediately shows the same colors as the 3D model
        when geological surfaces/solids are loaded. The legend appears exactly
        when the 3D model appears.
        
        Args:
            package: Geology package with surfaces/solids
            render_mode: "surfaces", "solids", or "both"
            geology_colors: List of hex color strings used for the units
        """
        try:
            if not hasattr(self, 'legend_manager') or self.legend_manager is None:
                logger.debug("No legend_manager available for geology sync")
                return
            
            # 1. Collect unit names and their corresponding colors
            unit_names = []
            unit_colors = []
            
            # Select source based on render mode
            if render_mode == "solids":
                source = package.get('solids', [])
            else:
                source = package.get('surfaces', [])
            
            # Fallback to surfaces if solids is empty
            if not source and render_mode == "solids":
                source = package.get('surfaces', [])
            
            for i, item in enumerate(source):
                # Extract unit name (try multiple keys)
                name = item.get('unit_name') or item.get('name') or f'Unit_{i}'
                unit_names.append(name)
                unit_colors.append(geology_colors[i % len(geology_colors)])
            
            if not unit_names:
                logger.debug("No geological units found for legend sync")
                return
                
            # Determine legend title
            if render_mode == "solids":
                title = "Geology (Solids)"
            elif render_mode == "both":
                title = "Geology (All)"
            else:
                title = "Geology (Surfaces)"
            
            # Update the legend
            if hasattr(self.legend_manager, 'update_geology_legend'):
                self.legend_manager.update_geology_legend(unit_names, unit_colors, title)
                logger.info(f"Synced geology legend: {len(unit_names)} units -> {title}")
            elif hasattr(self.legend_manager, 'update_discrete'):
                # Fallback to update_discrete
                from PyQt6.QtGui import QColor
                category_colors = {}
                for name, hex_color in zip(unit_names, unit_colors):
                    try:
                        qc = QColor(hex_color)
                        if qc.isValid():
                            category_colors[name] = (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
                    except Exception:
                        category_colors[name] = (0.5, 0.5, 0.5, 1.0)
                
                self.legend_manager.update_discrete(
                    property_name=title,
                    categories=unit_names,
                    category_colors=category_colors,
                    subtitle="Geological Units"
                )
                logger.info(f"Synced geology legend via update_discrete: {len(unit_names)} units")
            
        except Exception as e:
            logger.warning(f"Failed to sync geology legend: {e}")
    
    def get_geology_layers(self) -> List[Dict[str, Any]]:
        """
        Get information about all geological layers.
        
        Returns:
            List of dictionaries with layer information for surfaces and solids
        """
        geology_layers = []
        
        for name, layer in self.active_layers.items():
            layer_type = layer.get('type', '')
            if layer_type in ('geology_surface', 'geology_solid', 'geology_volume'):
                layer_data = layer.get('data', {})
                if isinstance(layer_data, dict):
                    data = layer_data
                else:
                    data = {'mesh': layer_data}
                
                info = {
                    'name': name,
                    'type': layer_type,
                    'visible': layer.get('visible', True),
                    'opacity': layer.get('opacity', 1.0),
                }
                
                # Add volume for solids
                if layer_type == 'geology_solid':
                    info['volume_m3'] = data.get('volume_m3', 0)
                
                # Add render type
                info['render_type'] = data.get('type', 'unknown')
                
                geology_layers.append(info)
        
        return geology_layers
    
    def set_geology_render_mode(self, mode: str) -> None:
        """
        Toggle between surfaces and solids display for geology.

        This is the UI handler for "Surfaces" vs "Solids" button toggle.

        Args:
            mode: "surfaces", "solids", or "both"
        """
        # Get currently stored geology package
        package = {
            'surfaces': [],
            'solids': getattr(self, '_geology_solids', []),
        }

        # Collect surface data from current layers
        for name, layer in list(self.active_layers.items()):
            if name.startswith("GeoSurface:"):
                layer_data = layer.get('data', {})
                if isinstance(layer_data, dict) and 'surface' in layer_data:
                    package['surfaces'].append(layer_data['surface'])

        # Reload with new render mode
        if package['surfaces'] or package['solids']:
            self.load_geology_package(package, render_mode=mode)
            logger.info(f"Switched geology render mode to: {mode}")

    # =========================================================================
    # Geological Explorer Panel Methods
    # =========================================================================

    def set_geology_contacts_visible(self, visible: bool) -> None:
        """Toggle visibility of geology contact points."""
        layer_name = "GeoContacts"
        if layer_name in self.active_layers:
            layer = self.active_layers[layer_name]
            actor = layer.get('actor')
            if actor:
                actor.SetVisibility(visible)
                layer['visible'] = visible
                self.force_render_update()
                logger.debug(f"Geology contacts visibility: {visible}")

    def set_geology_surfaces_visible(self, visible: bool) -> None:
        """Toggle visibility of geological surfaces."""
        for name in list(self.active_layers.keys()):
            if name.startswith("GeoSurface:") or name.startswith("geo_surface_"):
                layer = self.active_layers[name]
                actor = layer.get('actor')
                if actor:
                    actor.SetVisibility(visible)
                    layer['visible'] = visible
        self.force_render_update()
        logger.debug(f"Geology surfaces visibility: {visible}")

    def set_geology_misfit_visible(self, visible: bool) -> None:
        """Toggle visibility of misfit glyphs."""
        layer_name = "GeoMisfit"
        if layer_name in self.active_layers:
            layer = self.active_layers[layer_name]
            actor = layer.get('actor')
            if actor:
                actor.SetVisibility(visible)
                layer['visible'] = visible
                self.force_render_update()
                logger.debug(f"Geology misfit visibility: {visible}")

    def filter_geology_formations(self, formations: List[str]) -> None:
        """
        Filter visible geological formations by name.

        Args:
            formations: List of formation names to show. Empty list = show all.
                       ["__NONE__"] = hide all.
        """
        show_all = not formations or formations == []
        hide_all = formations == ["__NONE__"]

        for name in list(self.active_layers.keys()):
            if name.startswith("GeoSurface:") or name.startswith("GeoSolid:"):
                # Extract formation name from layer name
                formation = name.split(":", 1)[1] if ":" in name else name
                layer = self.active_layers[name]
                actor = layer.get('actor')
                if actor:
                    if show_all:
                        actor.SetVisibility(True)
                        layer['visible'] = True
                    elif hide_all:
                        actor.SetVisibility(False)
                        layer['visible'] = False
                    else:
                        visible = formation in formations
                        actor.SetVisibility(visible)
                        layer['visible'] = visible

        self.force_render_update()
        logger.debug(f"Filtered geology to formations: {formations}")

    def set_geology_opacity(self, opacity: float) -> None:
        """
        Set opacity for all geological surface layers.

        Args:
            opacity: Opacity value 0.0-1.0
        """
        for name in list(self.active_layers.keys()):
            if name.startswith("GeoSurface:") or name.startswith("geo_surface_"):
                layer = self.active_layers[name]
                actor = layer.get('actor')
                if actor and hasattr(actor, 'GetProperty'):
                    actor.GetProperty().SetOpacity(opacity)
                    layer['opacity'] = opacity
        self.force_render_update()
        logger.debug(f"Geology opacity set to: {opacity}")

    def set_geology_color_palette(self, palette: str) -> None:
        """
        Set color palette for geological layers.

        Note: This stores the palette preference. Full re-coloring would
        require reloading the geology package with the new colormap.

        Args:
            palette: Palette name (Geologic, Viridis, Tab10, Spectral)
        """
        # Map palette names to colormap values
        palette_map = {
            "geologic": "tab10",
            "viridis": "viridis",
            "tab10": "tab10",
            "spectral": "Spectral",
        }
        cmap = palette_map.get(palette.lower(), "tab10")

        # Store for future use
        self._geology_color_palette = cmap
        logger.info(f"Geology color palette set to: {palette} (cmap={cmap})")

    def fit_to_geology(self) -> None:
        """Reset camera to fit geological model bounds."""
        if hasattr(self, '_geology_bounds') and self._geology_bounds:
            self.fit_to_bounds(self._geology_bounds)
        else:
            self.fit_to_view()
        logger.debug("Camera fit to geology")

    def clear_geology(self) -> None:
        """
        Remove all geological layers from the scene.

        Wrapper for clear_geology_layers() for Geological Explorer panel.
        """
        self.clear_geology_layers()
        logger.info("Cleared all geological layers")

    def set_geology_layer_visibility(self, layer_name: str, visible: bool) -> None:
        """
        Toggle visibility of a specific geology layer.

        Works for individual surfaces, solids, contacts, or wireframes.

        Args:
            layer_name: Full layer name (e.g., "GeoSurface:Surface_val_1.0")
            visible: Whether to show the layer
        """
        if layer_name not in self.active_layers:
            logger.warning(f"Unknown geology layer: {layer_name}")
            return

        layer = self.active_layers[layer_name]
        actor = layer.get('actor')
        if actor:
            actor.SetVisibility(visible)
            layer['visible'] = visible
            self.force_render_update()
            logger.debug(f"Set {layer_name} visibility: {visible}")

    def set_wireframe_visible(self, visible: bool) -> None:
        """
        Toggle wireframe visibility for all geological solid units.

        When enabled, adds wireframe overlays showing geological unit boundaries
        (not internal mesh triangulation) to all GeoSolid and GeoSurface layers.
        NOTE: Unified Hull mesh is skipped due to high cell count causing driver issues.
        When disabled, removes all GeoWireframe layers.

        Args:
            visible: Whether to show wireframes
        """
        # Get all solid layers (individual solids only, NOT unified mesh to avoid driver crashes)
        # The unified hull has 120³ = 1.7M+ cells which creates millions of edges
        solid_layers = [name for name in self.active_layers.keys()
                       if name.startswith("GeoSolid:") or name.startswith("GeoSurface:")]
        
        # Skip unified hull - it's too large for wireframe rendering
        solid_layers = [name for name in solid_layers if not name.startswith("GeoUnified:")]

        if visible:
            # Create wireframes for each solid/unified layer
            for layer_name in solid_layers:
                # Extract unit name from layer name
                if ":" in layer_name:
                    unit_name = layer_name.split(":", 1)[1].strip()
                else:
                    unit_name = layer_name

                layer_data = self.active_layers[layer_name].get('data', {})
                mesh = layer_data.get('mesh')
                if mesh:
                    self._render_solid_wireframe(unit_name, mesh)
        else:
            # Remove all wireframe layers
            wireframe_layers = [name for name in list(self.active_layers.keys())
                               if name.startswith("GeoWireframe:")]
            for layer_name in wireframe_layers:
                self.clear_layer(layer_name)

        self.force_render_update()
        logger.info(f"Wireframe visibility: {visible}")

    def _render_solid_wireframe(self, unit_name: str, mesh) -> bool:
        """
        Add wireframe overlay for a solid unit following geological boundaries.

        Uses extract_feature_edges() to show only meaningful geological boundaries
        (boundary edges, sharp edges) rather than all triangulation edges which
        would create a "box model" appearance.

        Args:
            unit_name: Name of the solid unit
            mesh: PyVista mesh for the solid

        Returns:
            True if wireframe was created successfully
        """
        if self.plotter is None or mesh is None:
            return False

        layer_name = f"GeoWireframe:{unit_name}"

        # Remove existing wireframe if present
        if layer_name in self.active_layers:
            self.clear_layer(layer_name)

        try:
            # Extract FEATURE edges (geological boundaries) instead of ALL edges
            # This shows only:
            # - Boundary edges (outer perimeter of the geological unit)
            # - Sharp edges (where surface normal changes significantly - geological contacts)
            # - Non-manifold edges (where surfaces meet)
            # This avoids showing internal triangulation which creates "box model" look
            edges = mesh.extract_feature_edges(
                boundary_edges=True,      # Show outer boundary of the unit
                feature_edges=True,       # Show sharp geological contacts
                manifold_edges=False,     # Don't show smooth internal edges
                non_manifold_edges=True,  # Show where surfaces intersect
                feature_angle=30.0        # Angle threshold for "sharp" edges (degrees)
            )

            # If no feature edges found, fall back to boundary edges only
            if edges.n_cells == 0:
                edges = mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=False,
                    manifold_edges=False,
                    non_manifold_edges=False
                )

            # SAFETY CHECK: Skip wireframe on large meshes to prevent driver crashes
            # Large meshes (>500k cells) create millions of edges that overwhelm GPUs
            if mesh.n_cells > 500000:
                logger.warning(f"Skipping wireframe for {unit_name}: mesh too large ({mesh.n_cells} cells)")
                return False
            
            # Create wireframe actor
            # DRIVER FIX: Use line_width=1 and opacity=1.0 to prevent GPU driver errors
            # Many drivers (especially Intel integrated) don't support line_width > 1
            # and transparent lines cause blending conflicts with depth buffer
            actor = self.plotter.add_mesh(
                edges,
                name=layer_name,
                color='black',
                line_width=1,  # SAFE: 1.0 is universally supported by all drivers
                opacity=1.0,   # SAFE: Opaque lines avoid blending/depth issues
                render=False
            )

            self.add_layer(
                layer_name,
                actor,
                data={'unit_name': unit_name},
                layer_type='geology_wireframe'
            )

            logger.debug(f"Added geological wireframe for solid: {unit_name} ({edges.n_cells} edges)")
            return True

        except Exception as e:
            logger.error(f"Failed to create wireframe for {unit_name}: {e}")
            return False

    def apply_view_mode(self, mode: str) -> None:
        """
        Apply a predefined view mode that sets visibility for multiple layer types.

        Args:
            mode: One of "surfaces_only", "solids_only", "contacts_only",
                  "surfaces_solids", "all"
        """
        # Define visibility settings for each mode
        modes = {
            "surfaces_only": {'surfaces': True, 'solids': False, 'contacts': False},
            "solids_only": {'surfaces': False, 'solids': True, 'contacts': False},
            "contacts_only": {'surfaces': False, 'solids': False, 'contacts': True},
            "surfaces_solids": {'surfaces': True, 'solids': True, 'contacts': False},
            "all": {'surfaces': True, 'solids': True, 'contacts': True},
        }

        if mode not in modes:
            logger.warning(f"Unknown view mode: {mode}")
            return

        settings = modes[mode]

        # Apply to surfaces
        for name in list(self.active_layers.keys()):
            if name.startswith("GeoSurface:") or name.startswith("geo_surface_"):
                self.set_geology_layer_visibility(name, settings['surfaces'])
            elif name.startswith("GeoSolid:") or name.startswith("geo_solid_"):
                self.set_geology_layer_visibility(name, settings['solids'])
            elif name.startswith("GeoWireframe:"):
                # Wireframes follow solids visibility
                self.set_geology_layer_visibility(name, settings['solids'])

        # Apply to contacts
        self.set_geology_contacts_visible(settings['contacts'])

        self.force_render_update()
        logger.info(f"Applied view mode: {mode}")

    def set_solids_opacity(self, opacity: float) -> None:
        """
        Set opacity for all geological solid layers.

        Args:
            opacity: Opacity value 0.0-1.0
        """
        for name in list(self.active_layers.keys()):
            if name.startswith("GeoSolid:") or name.startswith("geo_solid_"):
                layer = self.active_layers[name]
                actor = layer.get('actor')
                if actor and hasattr(actor, 'GetProperty'):
                    actor.GetProperty().SetOpacity(opacity)
                    layer['opacity'] = opacity
        self.force_render_update()
        logger.debug(f"Geology solids opacity set to: {opacity}")

    def get_geology_layer_names(self) -> dict:
        """
        Get lists of all geology layer names by type.

        Returns:
            Dict with keys 'surfaces', 'solids', 'contacts', 'wireframes'
            containing lists of layer names.
        """
        result = {
            'surfaces': [],
            'solids': [],
            'contacts': [],
            'wireframes': []
        }

        for name in self.active_layers.keys():
            if name.startswith("GeoSurface:"):
                result['surfaces'].append(name)
            elif name.startswith("GeoSolid:"):
                result['solids'].append(name)
            elif name == "GeoContacts":
                result['contacts'].append(name)
            elif name.startswith("GeoWireframe:"):
                result['wireframes'].append(name)

        return result

    def update_drillhole_layer(self, actor, data, layer_name: str = "Drillholes") -> None:
        """
        Update drillhole layer reference.
        
        Args:
            actor: PyVista actor for drillholes
            data: Drillhole data dictionary
            layer_name: Optional custom layer name
        """
        # Register layer using generic method
        self.add_layer(layer_name, actor, data, layer_type='drillhole')
        
        # Register in scene_layers for global picking
        # For drillholes, data is a dict with 'mesh' key containing the actual PyVista mesh
        mesh_data = data.get('mesh') if isinstance(data, dict) else data
        self.register_scene_layer(layer_name, actor, mesh_data, 'drillhole')
        
        logger.info(f"Updated drillhole layer: {layer_name}")
    
    def get_active_layers_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all active layers.
        
        Returns:
            List of dictionaries with layer information
        """
        layers_info = []
        
        for name, layer in self.active_layers.items():
            if layer['actor'] is not None:
                layers_info.append({
                    'name': name.replace('_', ' ').title(),
                    'type': layer['type'],
                    'visible': layer['visible'],
                    'opacity': layer['opacity']
                })
        
        return layers_info
    
    def clear_layer(self, layer_name: str) -> None:
        """
        Clear a specific layer from the scene and remove it from active layers.
        
        FIX CS-006: Resets property state if this was the active layer.
        
        Args:
            layer_name: Name of the layer to clear
        """
        if layer_name not in self.active_layers:
            logger.warning(f"Unknown layer: {layer_name}")
            return
        
        layer = self.active_layers[layer_name]
        
        # Remove actor from plotter
        if layer['actor'] is not None:
            try:
                self.plotter.remove_actor(layer['actor'])
            except (AttributeError, RuntimeError) as e:
                logger.debug(f"Actor removal failed for layer {layer_name}: {e}")
        
        # Remove from active_layers dictionary
        del self.active_layers[layer_name]
        
        # Also remove from scene_layers if present
        if layer_name in self.scene_layers:
            del self.scene_layers[layer_name]
        
        # FIX CS-006: If no layers remain, reset property state
        if not self.active_layers:
            self.current_property = None
            self.current_colormap = 'viridis'
            self.current_color_mode = 'continuous'
            self.current_custom_colors = None
            
            # Clear legend and overlays
            if hasattr(self, 'legend_manager') and self.legend_manager is not None:
                try:
                    self.legend_manager.clear()
                except Exception:
                    pass
            
            if hasattr(self, 'overlay_manager') and self.overlay_manager is not None:
                try:
                    self.overlay_manager.set_bounds(None)
                except Exception:
                    pass
        
        # Defer render - PyVista will render on next frame automatically
        # This improves performance during layer removal
        
        logger.info(f"Cleared and removed {layer_name} layer")
        
        # Notify UI to update layer controls
        if self.layer_change_callback:
            self.layer_change_callback()
    
    def clear_all_layers(self) -> None:
        """
        Clear all active layers from the scene and remove them from layer lists.
        
        FIX CS-006: Also resets active property state, legend, and overlay state.
        """
        layer_names = list(self.active_layers.keys())  # Copy list to avoid modification during iteration
        
        for layer_name in layer_names:
            # Remove actor from plotter
            layer = self.active_layers[layer_name]
            if layer['actor'] is not None:
                try:
                    self.plotter.remove_actor(layer['actor'])
                except (AttributeError, RuntimeError) as e:
                    logger.debug(f"Actor removal failed for layer {layer_name}: {e}")
        
        # Clear all layers dictionaries
        self.active_layers.clear()
        self.scene_layers.clear()
        
        # FIX CS-006: Reset active property state
        self.current_property = None
        self.current_colormap = 'viridis'
        self.current_color_mode = 'continuous'
        self.current_custom_colors = None
        
        # FIX CS-006: Clear legend manager
        if hasattr(self, 'legend_manager') and self.legend_manager is not None:
            try:
                self.legend_manager.clear()
            except Exception:
                pass
        
        # FIX CS-006: Clear overlay bounds
        if hasattr(self, 'overlay_manager') and self.overlay_manager is not None:
            try:
                self.overlay_manager.set_bounds(None)
            except Exception:
                pass
        
        # Defer render - PyVista will render on next frame automatically
        # This improves performance during bulk layer clearing
        
        logger.info("Cleared all layers, reset property state, cleared legend")
        
        # Notify UI to update layer controls
        if self.layer_change_callback:
            self.layer_change_callback()

    def cleanup_observers(self) -> None:
        """Remove all VTK observers to prevent memory leaks."""
        for observer_type, obj, tag in self._observer_tags:
            try:
                obj.RemoveObserver(tag)
                logger.debug(f"Removed {observer_type} observer {tag}")
            except Exception as e:
                logger.debug(f"Could not remove observer {tag}: {e}")
        self._observer_tags.clear()
        logger.info("Cleaned up all VTK observers")

    def _clear_block_meshes(self) -> None:
        """Clear block mesh data to free GPU memory."""
        if hasattr(self, 'block_meshes') and self.block_meshes:
            for mesh_id, mesh in self.block_meshes.items():
                if mesh is not None:
                    try:
                        if hasattr(mesh, 'Release'):
                            mesh.Release()
                    except Exception as e:
                        logger.debug(f"Could not release mesh {mesh_id}: {e}")
            self.block_meshes.clear()
            logger.debug("Released block mesh GPU memory")

    def close(self) -> None:
        """Close the renderer and free all resources."""
        logger.info("Closing renderer and cleaning up resources")
        self.cleanup_observers()
        self._clear_block_meshes()
        self.clear_all_layers()
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception as e:
                logger.debug(f"Error closing plotter: {e}")
        logger.info("Renderer closed successfully")

    def add_drillhole_layer(
        self,
        database: DrillholeDatabase,
        composite_df=None,
        radius: float = 1.0,
        color_mode: str = "Lithology",
        assay_field: Optional[str] = None,
        visible_holes: Optional[Set[str]] = None,
        legend_title: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        lith_filter: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add drillholes as a layer to the renderer.
        Uses individual hole actors for instant visibility toggling.
        
        Args:
            database: DrillholeDatabase containing collars, surveys, assays, lithology
            composite_df: Optional DataFrame with composite data
            radius: Tube radius for drillholes
            color_mode: "Lithology" or "Assay"
            visible_holes: Set of hole IDs to show (None = show all)
            lith_filter: Optional list of lithology codes to show (empty list or None = show all)
        """
        if self.plotter is None:
            logger.warning("Cannot add drillhole layer: plotter not initialized")
            return
        
        try:
            logger.debug(
                "[DRILLHOLE DEBUG] add_drillhole_layer start: radius=%s, color_mode=%s",
                radius,
                color_mode,
            )
            def _progress(fraction: float, message: str) -> None:
                if progress_callback is None:
                    return
                try:
                    progress_callback(max(0.0, min(1.0, fraction)), message)
                except Exception as e:
                    # DR-011 fix: Log exception instead of silently swallowing
                    logger.debug(f"Progress callback failed: {e}")
            _progress(0.02, "Preparing drillhole polylines")
            # Remove existing drillhole layer if present
            self.remove_drillhole_layer()
            
            # Build polylines using shared helper (cache this for radius updates)
            # CRITICAL: Pass registry for persistent interval IDs (GPU picking stability)
            registry = getattr(self, '_registry', None)
            result = build_drillhole_polylines(database, composite_df, assay_field_name=assay_field, registry=registry)
            hole_polys = result["hole_polys"]
            hole_segment_lith = result["hole_segment_lith"]
            hole_segment_assay = result["hole_segment_assay"]
            lith_colors = result["lith_colors"]
            lith_to_index = result["lith_to_index"]
            assay_field = result["assay_field"]
            assay_min = result["assay_min"]
            assay_max = result["assay_max"]
            hole_ids = result["hole_ids"]
            collar_coords = result["collar_coords"]
            logger.debug(
                "[DRILLHOLE DEBUG] Polylines built: holes=%d, collars=%d",
                len(hole_ids),
                len(collar_coords),
            )
            
            # ================================================================
            # CRITICAL FIX: Apply coordinate shift to drillhole polylines
            # ================================================================
            # Drillholes must use the SAME coordinate shift as geological surfaces
            # to render together in the same scene. Without this, drillholes remain
            # at UTM coordinates (e.g., 500,000m) while geology is shifted to ~(0,0,0),
            # causing both to disappear due to camera clipping range issues.
            # ================================================================
            all_points = []
            for poly in hole_polys.values():
                if poly.n_points > 0:
                    all_points.append(poly.points)
            
            if all_points:
                all_points_stacked = np.vstack(all_points)
                # Apply the SAME coordinate shift used by geological surfaces
                # This locks _global_shift if not already set (first dataset authority)
                shifted_origin = self._to_local_precision(all_points_stacked[:1])  # Lock shift
                
                # Now transform ALL polyline points to local coordinates
                for hid, poly in hole_polys.items():
                    if poly.n_points > 0:
                        local_points = self._to_local_precision(poly.points.copy())
                        poly.points = local_points
                
                # Also transform collar coordinates to local system
                if self._global_shift is not None:
                    for hid in collar_coords:
                        cx, cy, cz = collar_coords[hid]
                        collar_coords[hid] = (
                            cx - self._global_shift[0],
                            cy - self._global_shift[1],
                            cz - self._global_shift[2]
                        )
                    logger.info(f"[DRILLHOLE RENDER] Applied coordinate shift to {len(collar_coords)} collars")
                
                # Log shifted bounds for verification
                dh_x_min, dh_x_max = float(all_points_stacked[:, 0].min()), float(all_points_stacked[:, 0].max())
                dh_y_min, dh_y_max = float(all_points_stacked[:, 1].min()), float(all_points_stacked[:, 1].max())
                dh_z_min, dh_z_max = float(all_points_stacked[:, 2].min()), float(all_points_stacked[:, 2].max())
                logger.info(
                    f"[DRILLHOLE RENDER] Original drillhole bounds (world): X=[{dh_x_min:.2f}, {dh_x_max:.2f}], "
                    f"Y=[{dh_y_min:.2f}, {dh_y_max:.2f}], Z=[{dh_z_min:.2f}, {dh_z_max:.2f}]"
                )
                if self._global_shift is not None:
                    logger.info(
                        f"[DRILLHOLE RENDER] Applied global shift: [{self._global_shift[0]:.2f}, "
                        f"{self._global_shift[1]:.2f}, {self._global_shift[2]:.2f}]"
                    )
            
            _progress(0.15, "Caching drillhole data")
            
            # Cache polylines data for radius updates
            # Store color_mode if provided - colors will be applied after loading
            self._drillhole_polylines_cache = {
                "hole_polys": hole_polys,
                "hole_segment_lith": hole_segment_lith,
                "hole_segment_assay": hole_segment_assay,
                "lith_colors": lith_colors,
                "lith_to_index": lith_to_index,
                "assay_field": assay_field,
                "assay_min": assay_min,
                "assay_max": assay_max,
                "database": database,
                "composite_df": composite_df,
                "color_mode": color_mode,  # Store color_mode for later color application
                "collar_coords": collar_coords,
                "radius": radius,
            }
            
            # Determine which holes to show
            if visible_holes is None:
                visible_holes = set(hole_ids)
            
            # Apply lithology filter if specified
            # lith_filter is a list of lithology codes to show - if empty, show all
            if lith_filter and len(lith_filter) > 0:
                lith_filter_set = set(lith_filter)
                logger.info(f"Applying lithology filter: showing only {lith_filter}")
                
                # Filter out segments that don't match the lithology filter
                # This modifies hole_polys and hole_segment_lith in-place
                for hid in list(hole_polys.keys()):
                    segments = hole_segment_lith.get(hid, [])
                    assay_segments = hole_segment_assay.get(hid, [])
                    poly = hole_polys.get(hid)
                    
                    if poly is None or poly.n_points == 0:
                        continue
                    
                    # Find which segments match the filter
                    matching_indices = [i for i, lith in enumerate(segments) if lith in lith_filter_set]
                    
                    if not matching_indices:
                        # No matching segments - create empty polydata
                        hole_polys[hid] = pv.PolyData()
                        hole_segment_lith[hid] = []
                        hole_segment_assay[hid] = []
                        continue
                    
                    # Rebuild polylines with only matching segments
                    if len(matching_indices) < len(segments):
                        # Need to filter - rebuild the polyline from scratch
                        points = poly.points
                        original_lines = poly.lines
                        
                        # Parse original line connectivity
                        new_points = []
                        new_lines = []
                        new_lith = []
                        new_assay = []
                        point_map = {}  # old_idx -> new_idx
                        
                        line_idx = 0
                        seg_idx = 0
                        while line_idx < len(original_lines):
                            n_pts = original_lines[line_idx]
                            if seg_idx in matching_indices:
                                # Include this segment
                                line_pts = []
                                for j in range(1, n_pts + 1):
                                    old_pt_idx = original_lines[line_idx + j]
                                    if old_pt_idx not in point_map:
                                        point_map[old_pt_idx] = len(new_points)
                                        new_points.append(points[old_pt_idx])
                                    line_pts.append(point_map[old_pt_idx])
                                new_lines.extend([n_pts] + line_pts)
                                new_lith.append(segments[seg_idx])
                                new_assay.append(assay_segments[seg_idx] if seg_idx < len(assay_segments) else np.nan)
                            
                            line_idx += n_pts + 1
                            seg_idx += 1
                        
                        if new_points and new_lines:
                            new_poly = pv.PolyData(np.array(new_points, dtype=float))
                            new_poly.lines = np.array(new_lines, dtype=np.int64)
                            hole_polys[hid] = new_poly
                            hole_segment_lith[hid] = new_lith
                            hole_segment_assay[hid] = new_assay
                        else:
                            hole_polys[hid] = pv.PolyData()
                            hole_segment_lith[hid] = []
                            hole_segment_assay[hid] = []
                
                # Update cache with filtered data
                self._drillhole_polylines_cache["hole_polys"] = hole_polys
                self._drillhole_polylines_cache["hole_segment_lith"] = hole_segment_lith
                self._drillhole_polylines_cache["hole_segment_assay"] = hole_segment_assay
                
                # CRITICAL: Rebuild lith_to_index and lith_colors to match filtered lithologies
                # Otherwise property panel will show all lithologies, not just filtered ones
                filtered_unique_codes = sorted({code for codes in hole_segment_lith.values() for code in codes if code})
                
                if filtered_unique_codes:
                    # Rebuild lith_to_index with only filtered codes
                    filtered_lith_to_index = {code: idx for idx, code in enumerate(filtered_unique_codes)}
                    
                    # Rebuild lith_colors - preserve original colors for codes that remain
                    filtered_lith_colors = {}
                    for code in filtered_unique_codes:
                        if code in lith_colors:
                            filtered_lith_colors[code] = lith_colors[code]
                        else:
                            # Shouldn't happen, but provide fallback color just in case
                            filtered_lith_colors[code] = "#808080"  # Gray
                    
                    # Update cache and local variables with filtered mappings
                    lith_to_index = filtered_lith_to_index
                    lith_colors = filtered_lith_colors
                    self._drillhole_polylines_cache["lith_to_index"] = lith_to_index
                    self._drillhole_polylines_cache["lith_colors"] = lith_colors
                    
                    logger.info(f"Rebuilt lithology mappings after filter: {len(filtered_unique_codes)} codes remaining")
                else:
                    logger.warning("No lithology codes remaining after filter - using empty mappings")
                    lith_to_index = {}
                    lith_colors = {}
                    self._drillhole_polylines_cache["lith_to_index"] = lith_to_index
                    self._drillhole_polylines_cache["lith_colors"] = lith_colors
            
            # OPTION: Use GPU renderer for large datasets or if explicitly enabled
            # Auto-enable GPU renderer for datasets with >1000 holes or >10k intervals
            total_intervals = sum(len(segments) for segments in hole_segment_lith.values())
            # GPU renderer is currently experimental - only use if explicitly enabled
            # or for very large datasets where standard renderer would be too slow
            use_gpu = self._use_gpu_drillholes and (len(hole_ids) > 5000 or total_intervals > 100000)
            
            if use_gpu:
                try:
                    from .drillhole_gpu_renderer import (
                        DrillholeGPURenderer,
                        create_intervals_from_polyline_data,
                        get_drillhole_event_bus,
                        RenderQuality,
                    )
                    
                    logger.info(f"Using GPU renderer for {len(hole_ids)} holes ({total_intervals} intervals)")
                    
                    # Store original radius for transform scaling
                    self._drillhole_original_radius = radius
                    
                    # Convert to GPU intervals
                    intervals = create_intervals_from_polyline_data(result, radius=radius)
                    
                    # Create GPU renderer
                    self._gpu_drillhole_renderer = DrillholeGPURenderer(
                        self.plotter,
                        quality=RenderQuality.HIGH,
                        enable_gpu_picking=True
                    )
                    
                    # Load intervals
                    self._gpu_drillhole_renderer.load_intervals(intervals)
                    
                    # Apply colors if color_mode is provided
                    if color_mode and color_mode != "None":
                        logger.info(f"Applying colors to GPU drillholes with color_mode={color_mode}")
                        try:
                            # Determine property name and colormap based on color_mode
                            if color_mode == "Lithology":
                                property_name = "lithology"
                                colormap = "tab10"
                            else:  # Assay
                                property_name = "assay"  # GPU renderer uses "assay" as property name
                                colormap = "turbo"
                            
                            # Render with colormap first
                            self._gpu_drillhole_renderer.render(colormap=colormap, show_collars=True)
                            
                            # Update color property if method exists
                            if hasattr(self._gpu_drillhole_renderer, 'update_color_property'):
                                self._gpu_drillhole_renderer.update_color_property(property_name=property_name)
                            
                            # Update colormap if method exists
                            if hasattr(self._gpu_drillhole_renderer, 'update_colormap'):
                                self._gpu_drillhole_renderer.update_colormap(colormap=colormap)
                                
                        except Exception as e:
                            logger.warning(f"Failed to apply colors to GPU drillholes: {e}")
                            # Render without colormap as fallback
                            self._gpu_drillhole_renderer.render(colormap=None, show_collars=True)
                    else:
                        logger.info("Loading GPU drillholes without color assignment - colors will be applied when property is selected")
                        # Render without colormap - use uniform color
                        self._gpu_drillhole_renderer.render(colormap=None, show_collars=True)
                    
                    # Set up mouse event handlers for hover and click
                    self._setup_drillhole_interaction()
                    
                    # Connect to event bus for selection/hover feedback
                    event_bus = get_drillhole_event_bus()
                    event_bus.intervalSelected.connect(self._on_drillhole_interval_selected)
                    event_bus.intervalHovered.connect(self._on_drillhole_interval_hovered)
                    
                    # Store layer data with color_mode
                    layer_data = {
                        "database": database,
                        "composite_df": composite_df,
                        "radius": radius,
                        "color_mode": color_mode,  # Store color_mode
                        "hole_polys": hole_polys,
                        "hole_segment_lith": hole_segment_lith,
                        "hole_segment_assay": hole_segment_assay,
                        "lith_colors": lith_colors,
                        "lith_to_index": lith_to_index,
                        "assay_field": assay_field,
                        "assay_min": assay_min,
                        "assay_max": assay_max,
                        "hole_ids": hole_ids,
                        "collar_coords": collar_coords,
                        "visible_holes": visible_holes,
                        "gpu_renderer": self._gpu_drillhole_renderer,
                    }
                    
                    # Register GPU renderer actors with Visual Density Controller
                    if self.visual_density_controller is not None and self._gpu_drillhole_renderer._main_actor:
                        try:
                            # For GPU renderer, we have one main actor and collar actors
                            gpu_drillhole_actors = {"gpu_main": self._gpu_drillhole_renderer._main_actor}
                            gpu_collar_actors = getattr(self._gpu_drillhole_renderer, '_collar_actors', {})

                            self.visual_density_controller.register_actors(
                                drillhole_actors=gpu_drillhole_actors,
                                collar_actors=gpu_collar_actors,
                                label_actors=[],
                            )
                            logger.debug("Registered GPU drillhole actors with VisualDensityController")
                        except Exception as e:
                            logger.warning(f"Failed to register GPU actors with VisualDensityController: {e}")

                    # Use first actor from GPU renderer
                    if self._gpu_drillhole_renderer._main_actor:
                        self.add_layer("drillholes", self._gpu_drillhole_renderer._main_actor,
                                     data=layer_data, layer_type="drillhole", opacity=1.0)

                    logger.info("GPU drillhole renderer initialized with hover/click support")
                    _progress(1.0, "Drillholes ready (GPU accelerated)")
                    return layer_data
                    
                except Exception as e:
                    logger.warning(f"GPU renderer failed, falling back to standard renderer: {e}")
                    self._gpu_drillhole_renderer = None
                    # Continue with standard renderer below
            logger.debug(
                "[DRILLHOLE DEBUG] visible holes count = %d (total=%d)",
                len(visible_holes),
                len(hole_ids),
            )
            total_holes = len(hole_ids)
            
            # Store original radius for transform scaling
            self._drillhole_original_radius = radius
            
            # Apply colors if color_mode is provided, otherwise load without colors
            if color_mode and color_mode != "None":
                logger.info(f"Loading drillholes with color_mode={color_mode} - colors will be applied after loading")
                use_colors_during_loading = True
                loading_color_mode = color_mode
            else:
                logger.info("Loading drillholes without color assignment - colors will be applied when property is selected")
                use_colors_during_loading = False
                loading_color_mode = None
            scalar_name = None  # Will be set when colors are applied
            
            # ULTRA-OPTIMIZED: Merge polylines first, then apply ONE tube filter
            # This is 10-100x faster than creating individual tubes
            self._drillhole_hole_actors = {}
            all_bounds = []
            
            # Adaptive quality based on dataset size
            # Minimum 12 sides ensures tubes look cylindrical, not polygonal
            n_sides = 16  # Default high quality
            if total_holes > 500:
                n_sides = 12  # Medium-high quality for large datasets
            elif total_holes > 200:
                n_sides = 14  # High quality
            
            _progress(0.10, f"Preparing {total_holes} drillholes...")
            
            # Phase 1: Collect polylines WITH hole IDs (preserve mapping)
            # Colors will be assigned later when user selects a property
            polylines_with_ids = []  # List of (poly, hid) tuples to preserve mapping
            all_scalars = []
            
            for hid in hole_ids:
                poly = hole_polys.get(hid)
                if poly is None or poly.n_cells < 1:
                    continue
                
                # Do NOT assign any scalar data during loading
                # Scalar data will be assigned when user explicitly selects a property
                
                polylines_with_ids.append((poly, hid))
                all_bounds.append(poly.bounds)
            
            if not polylines_with_ids:
                logger.warning("No drillhole polylines to render")
                _progress(1.0, "No drillhole polylines to render")
                return
            
            # Phase 2: Build tubes for each hole (optimized with adaptive quality)
            _progress(0.20, f"Building {len(polylines_with_ids)} drillhole tubes...")
            logger.info(f"Building tubes for {len(polylines_with_ids)} holes...")
            
            self._drillhole_hole_actors = {}
            tubes_with_ids = []
            
            for idx, (poly, hid) in enumerate(polylines_with_ids):
                # Build tube with adaptive quality
                # Use capping=False for smooth tube junctions between segments
                # This prevents the "stacked boxes" appearance from capped segment ends
                tube = poly.tube(radius=radius, capping=False, n_sides=n_sides)
                if tube.n_cells < 1:
                    continue
                
                tubes_with_ids.append((tube, hid))
                
                # Progress update every 50 holes (throttled to avoid signal flooding)
                if (idx + 1) % 50 == 0:
                    frac = 0.20 + 0.50 * ((idx + 1) / len(polylines_with_ids))
                    _progress(frac, f"Building tubes: {idx + 1}/{len(polylines_with_ids)}")
                    # Force UI repaint using safer processEvents
                    from PyQt6.QtCore import QEventLoop
                    from PyQt6.QtWidgets import QApplication
                    QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            
            if not tubes_with_ids:
                logger.warning("No drillhole tubes to render")
                _progress(1.0, "No drillhole tubes to render")
                return
            
            # Phase 3: Create individual actors (enables individual visibility control)
            _progress(0.70, f"Creating {len(tubes_with_ids)} drillhole actors...")
            logger.info(f"Creating actors for {len(tubes_with_ids)} holes...")
            
            for idx, (tube, hid) in enumerate(tubes_with_ids):
                # Create actor for this hole WITHOUT any color assignment
                # Use uniform gray color - colors will be applied when user selects a property
                actor = self.plotter.add_mesh(
                    tube,
                    color='lightgray',  # Uniform color - no scalars
                    show_scalar_bar=False,
                    reset_camera=False,
                    smooth_shading=True,
                    pbr=False,
                    lighting=True,
                    specular=0.3,
                    specular_power=15,
                    ambient=0.3,
                    diffuse=0.8,
                    interpolate_before_map=True,
                    pickable=True,  # Enable picking for hover/click interaction
                )
                
                # Set visibility based on visible_holes set
                if hid in visible_holes:
                    actor.VisibilityOn()
                else:
                    actor.VisibilityOff()
                
                self._drillhole_hole_actors[hid] = actor
                all_bounds.append(tube.bounds)
                
                # Progress update every 50 actors (throttled to avoid signal flooding)
                if (idx + 1) % 50 == 0:
                    frac = 0.70 + 0.10 * ((idx + 1) / len(tubes_with_ids))
                    _progress(frac, f"Creating actors: {idx + 1}/{len(tubes_with_ids)}")
                    # Force UI repaint using safer processEvents
                    from PyQt6.QtCore import QEventLoop
                    from PyQt6.QtWidgets import QApplication
                    QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            
            if not actor:
                logger.warning("Failed to create drillhole actor")
                _progress(1.0, "Failed to create drillhole actor")
                return
            
            logger.info(f"Created {len(self._drillhole_hole_actors)} individual drillhole actors")
            
            # Add collar markers (professional touch - like Leapfrog/Surpac)
            # Use MERGED geometry for performance - single actor instead of one per collar
            _progress(0.85, "Adding collar markers")
            self._drillhole_collar_actors = {}
            collar_radius = radius * 1.5  # Slightly larger than tube
            collar_height = radius * 0.8  # Height of the cone/cylinder marker
            
            # Collect all visible collar positions for merged rendering
            collar_meshes = []
            visible_collar_hids = []
            
            for hid in hole_ids:
                if hid not in visible_holes:
                    continue
                collar = collar_coords.get(hid)
                if collar is None:
                    continue
                cx, cy, cz = collar
                visible_collar_hids.append(hid)
                
                # Create a small cone pointing down (like a survey marker/pin)
                # This is more distinctive than a flat disc
                cone = pv.Cone(
                    center=(cx, cy, cz + collar_height * 0.5),
                    direction=(0, 0, -1),
                    height=collar_height,
                    radius=collar_radius,
                    resolution=8,  # Low resolution for performance
                    capping=True
                )
                collar_meshes.append(cone)
            
            # Merge all collar meshes into one for fast rendering
            if collar_meshes:
                import time
                merge_start = time.perf_counter()
                
                # OPTIMIZED: Use batch pv.merge() - O(n) instead of O(n²) (DR-008 fix)
                if len(collar_meshes) == 1:
                    merged_collars = collar_meshes[0]
                else:
                    # Single batch merge call - much faster for large datasets
                    merged_collars = pv.merge(collar_meshes)
                
                merge_elapsed = (time.perf_counter() - merge_start) * 1000
                logger.info(f"Collar merge: {len(collar_meshes)} collars merged in {merge_elapsed:.1f}ms")
                
                # Single actor for all collars - much faster than individual actors
                collar_actor = self.plotter.add_mesh(
                    merged_collars,
                    color="#000000",  # Black color for collars
                    show_scalar_bar=False,
                    reset_camera=False,
                    lighting=True,
                    specular=0.6,
                    specular_power=20,
                    ambient=0.4,
                    diffuse=0.7,
                    pickable=True,
                    name="drillhole_collars_merged"
                )
                
                # Store single merged actor (use "_merged" key to indicate merged mode)
                self._drillhole_collar_actors["_merged"] = collar_actor
                self._drillhole_collar_actors["_merged_hids"] = visible_collar_hids
                
                logger.info(f"Created merged collar actor for {len(visible_collar_hids)} collars")

            # Register actors with Visual Density Controller for automatic LOD
            if self.visual_density_controller is not None:
                try:
                    self.visual_density_controller.register_actors(
                        drillhole_actors=self._drillhole_hole_actors,
                        collar_actors=self._drillhole_collar_actors,
                        label_actors=[],  # No label actors in standard renderer
                    )
                    logger.debug(f"Registered {len(self._drillhole_hole_actors)} drillhole actors with VisualDensityController")
                except Exception as e:
                    logger.warning(f"Failed to register actors with VisualDensityController: {e}")

            # Store layer data (using first actor as representative)
            first_actor = next(iter(self._drillhole_hole_actors.values()))
            layer_data = {
                "database": database,
                "composite_df": composite_df,
                "radius": radius,
                "color_mode": color_mode,
                "hole_polys": hole_polys,
                "hole_segment_lith": hole_segment_lith,
                "hole_segment_assay": hole_segment_assay,
                "lith_colors": lith_colors,
                "lith_to_index": lith_to_index,
                "assay_field": assay_field or result.get("assay_field"),
                "assay_min": assay_min,
                "assay_max": assay_max,
                "hole_ids": hole_ids,
                "collar_coords": collar_coords,
                "visible_holes": visible_holes,
            }
            
            self.add_layer("drillholes", first_actor, data=layer_data, layer_type="drillhole", opacity=1.0)
            
            # Register in scene_layers for global picking and state management
            # CRITICAL: Must register BEFORE triggering callback so state check finds the layer
            self.register_scene_layer("drillholes", first_actor, layer_data, "drillhole")
            
            # Trigger another callback now that scene_layers is populated
            # This ensures the app state transitions to RENDERED
            if self.layer_change_callback:
                self.layer_change_callback()
            
            # Explicitly ensure layer is visible
            if "drillholes" in self.active_layers:
                self.active_layers["drillholes"]['visible'] = True
            
            # Ensure all drillhole actors are visible (fix for visibility issue)
            # NOTE: After _update_drillhole_colors is called, individual actors may be replaced
            # with a single "_merged" actor. Handle both cases.
            if "_merged" in self._drillhole_hole_actors:
                # Merged actor mode (after color update)
                try:
                    self._drillhole_hole_actors["_merged"].VisibilityOn()
                    logger.debug("Enabled visibility for merged drillhole actor")
                except Exception as e:
                    logger.warning(f"Failed to enable visibility for merged drillhole actor: {e}")
            else:
                # Individual actor mode (before color update)
                for hid, actor in self._drillhole_hole_actors.items():
                    if hid in visible_holes:
                        try:
                            actor.VisibilityOn()
                        except Exception:
                            pass
            
            # Ensure collar actor is visible (merged mode)
            if "_merged" in self._drillhole_collar_actors:
                try:
                    self._drillhole_collar_actors["_merged"].VisibilityOn()
                except Exception:
                    pass
            else:
                # Legacy individual collar actors
                for hid, actor in self._drillhole_collar_actors.items():
                    if hid.startswith("_"):
                        continue  # Skip metadata keys
                    if hid in visible_holes:
                        try:
                            actor.VisibilityOn()
                        except Exception:
                            pass
            
            # Ensure the representative actor is visible (may be merged or individual)
            if first_actor is not None:
                try:
                    first_actor.VisibilityOn()
                except Exception:
                    pass
            
            # Force render to ensure visibility changes take effect
            if self.plotter is not None:
                try:
                    self.plotter.render()
                except Exception:
                    pass
            
            # Initialize state manager with drillhole data
            if self._drillhole_state_manager:
                try:
                    self._drillhole_state_manager.register_holes(list(hole_ids))
                    # Set visibility for all holes
                    for hid in hole_ids:
                        self._drillhole_state_manager.set_visibility(hid, hid in visible_holes)
                    self._drillhole_state_manager.set_color_property(
                        "lithology" if color_mode == "Lithology" else "assay"
                    )
                    self._drillhole_state_manager.set_colormap("tab10" if color_mode == "Lithology" else "turbo")
                    self._drillhole_state_manager.set_tube_radius(radius)
                except Exception as e:
                    logger.debug(f"State manager initialization failed: {e}")
            
            logger.debug(
                "[DRILLHOLE DEBUG] Added drillhole layer with %d actors; visible_holes=%d",
                len(self._drillhole_hole_actors),
                len(visible_holes),
            )
            
            # Set up click and hover for standard PyVista renderer
            self._setup_standard_drillhole_interaction(hole_polys, hole_segment_lith, hole_segment_assay, database)
            
            try:
                self._update_scene_bounds()
                logger.debug(
                    "[DRILLHOLE DEBUG] Scene bounds after drillhole update: %s",
                    self._fixed_scene_bounds,
                )
            except Exception as exc:
                logger.debug(f"Could not refresh scene bounds after drillhole load: {exc}")
            _progress(0.9, "Positioning camera for drillholes")
            
            # Reset camera to fit drillholes
            try:
                if all_bounds:
                    # Calculate combined bounds
                    min_x = min(b[0] for b in all_bounds)
                    max_x = max(b[1] for b in all_bounds)
                    min_y = min(b[2] for b in all_bounds)
                    max_y = max(b[3] for b in all_bounds)
                    min_z = min(b[4] for b in all_bounds)
                    max_z = max(b[5] for b in all_bounds)
                    
                    drill_bounds = (min_x, max_x, min_y, max_y, min_z, max_z)
                    if self._fixed_scene_bounds is None:
                        self._fixed_scene_bounds = drill_bounds
                    else:
                        fb = self._fixed_scene_bounds
                        self._fixed_scene_bounds = (
                            min(fb[0], drill_bounds[0]),
                            max(fb[1], drill_bounds[1]),
                            min(fb[2], drill_bounds[2]),
                            max(fb[3], drill_bounds[3]),
                            min(fb[4], drill_bounds[4]),
                            max(fb[5], drill_bounds[5]),
                        )
                    try:
                        self._maintain_clipping_range()
                        cam_clip = (
                            self.plotter.renderer.GetActiveCamera().GetClippingRange()
                            if self.plotter.renderer is not None
                            else None
                        )
                        logger.debug(
                            "[DRILLHOLE DEBUG] Camera clipping after drillhole bounds merge: %s",
                            cam_clip,
                        )
                    except Exception:
                        pass
                    
                    center = [
                        (min_x + max_x) / 2,
                        (min_y + max_y) / 2,
                        (min_z + max_z) / 2,
                    ]
                    size = max(max_x - min_x, max_y - min_y, max_z - min_z)
                    if size <= 0:
                        size = 1.0
                    
                    camera = self.plotter.renderer.GetActiveCamera()
                    if camera:
                        new_pos = [
                            center[0] + size * 1.5,
                            center[1] + size * 1.5,
                            center[2] + size * 1.5,
                        ]
                        camera.SetFocalPoint(center)
                        camera.SetPosition(new_pos)
                        camera.SetViewUp(0, 0, 1)
                        try:
                            self.plotter.renderer.ResetCameraClippingRange()
                        except Exception:
                            pass
                        logger.debug(
                            "[DRILLHOLE DEBUG] Camera positioned for drillholes: pos=%s focal=%s",
                            camera.GetPosition(),
                            camera.GetFocalPoint(),
                        )
                    else:
                        self.plotter.reset_camera(bounds=drill_bounds)
                else:
                    self.plotter.reset_camera()
            except Exception as e:
                logger.warning(f"Failed to position camera for drillholes: {e}")
                try:
                    self.plotter.reset_camera()
                except Exception:
                    pass
            
            # Force render
            try:
                self.plotter.render()
                try:
                    cam = self.plotter.renderer.GetActiveCamera()
                    if cam:
                        logger.debug(
                            "[DRILLHOLE DEBUG] Camera after render: pos=%s focal=%s clip=%s",
                            cam.GetPosition(),
                            cam.GetFocalPoint(),
                            cam.GetClippingRange(),
                        )
                except Exception:
                    pass
                _progress(1.0, "Drillholes ready")
            except Exception as e:
                logger.warning(f"Failed to render after adding drillholes: {e}")
            
            logger.info(f"Added drillhole layer with {len(self._drillhole_hole_actors)} individual actors")
            
            # Apply colors if color_mode was provided
            if color_mode and color_mode != "None" and len(self._drillhole_hole_actors) > 0:
                logger.info(f"Applying colors to drillholes with color_mode={color_mode}")
                try:
                    # Determine property name and colormap based on color_mode
                    if color_mode == "Lithology":
                        property_name = "lithology"
                        colormap = "tab10"
                        color_mode_param = "discrete"
                    else:  # Assay
                        property_name = assay_field or "assay"
                        colormap = "turbo"
                        color_mode_param = "continuous"
                    
                    # Apply colors using the update method
                    self._update_drillhole_colors(
                        property_name=property_name,
                        colormap=colormap,
                        color_mode=color_mode_param,
                        custom_colors=None
                    )
                    
                    # Update current property tracking
                    self.current_property = property_name
                    self.current_colormap = colormap
                    
                    # Create legend metadata with color information
                    # CRITICAL: Use lith_colors keys (which are updated by _update_drillhole_colors)
                    # instead of lith_to_index keys (which may not be updated)
                    cache = self._drillhole_polylines_cache
                    if color_mode == "Lithology":
                        lith_colors_from_cache = cache.get("lith_colors", {})
                        logger.info(f"[LEGEND CREATE] cache lith_colors has {len(lith_colors_from_cache)} entries: {list(lith_colors_from_cache.keys())[:5]}")
                        unique_liths = sorted(list(lith_colors_from_cache.keys()))
                        legend_metadata: Dict[str, Any] = {
                            "property": property_name,
                            "title": legend_title or "Drillholes",
                            "mode": "discrete",
                            "colormap": colormap,
                            "categories": unique_liths,
                            "category_colors": cache.get("lith_colors", {}),
                            "vmin": None,
                            "vmax": None,
                            "scalar_name": "lith_id",
                            "color_mode": color_mode,
                        }
                    else:
                        legend_metadata: Dict[str, Any] = {
                            "property": property_name,
                            "title": legend_title or "Drillholes",
                            "mode": "continuous",
                            "colormap": colormap,
                            "categories": None,
                            "category_colors": None,
                            "vmin": cache.get("assay_min", 0.0),
                            "vmax": cache.get("assay_max", 1.0),
                            "scalar_name": property_name,  # Use actual property name (Cu, Au, etc.) instead of generic "assay"
                            "color_mode": color_mode,
                        }
                    
                    # Store legend metadata
                    self._drillhole_legend_metadata = legend_metadata
                    
                    logger.info(f"Applied colors to drillholes: property={property_name}, colormap={colormap}")
                    
                    return legend_metadata
                    
                except Exception as e:
                    logger.warning(f"Failed to apply colors to drillholes: {e}", exc_info=True)
                    # Fall through to return empty legend metadata
            
            # No property assigned during loading - drillholes load without colors
            property_name = None  # No property selected initially
            self.current_property = None
            self.current_colormap = None
            
            # Skip legend creation during loading - legend will be created when user selects a property
            # Store empty legend metadata to indicate no colors are assigned
            legend_metadata: Dict[str, Any] = {
                "property": None,
                "title": legend_title or "Drillholes",
                "mode": None,
                "colormap": None,
                "categories": None,
                "category_colors": None,
                "vmin": None,
                "vmax": None,
                "scalar_name": None,
                "color_mode": color_mode if color_mode else None,
            }
            
            # Store legend metadata (empty - no colors assigned)
            self._drillhole_legend_metadata = legend_metadata
            
            # Do NOT update legend during loading - no colors are assigned
            # Legend will be updated when user explicitly selects a property
            logger.info("Skipping legend update during drillhole loading - no colors assigned")
            
            return legend_metadata
            
        except Exception as e:
            logger.error(f"Failed to add drillhole layer: {e}", exc_info=True)
            raise
    
    def _setup_drillhole_interaction(self) -> None:
        """
        Set up mouse event handlers for drillhole hover and click.
        
        PERFORMANCE: Uses PickingController for centralized LOD gating.
        Connects to GPU picker only when allowed by controller.
        
        NOTE: This sets up additional observers for drillhole-specific picking.
        The primary hover/click handling is done in ViewerWidget.
        """
        if self.plotter is None or self._gpu_drillhole_renderer is None:
            return
        
        try:
            # Get VTK interactor
            interactor = self.plotter.iren
            if interactor is None:
                logger.warning("Cannot setup drillhole interaction: no interactor")
                return
            
            # Store reference to GPU renderer and picking controller
            gpu_renderer = self._gpu_drillhole_renderer
            picking_ctrl = get_picking_controller()
            
            def on_mouse_move(obj, event):
                """Handle mouse move for hover - gated by PickingController."""
                try:
                    # Gate on PickingController
                    if not picking_ctrl.hover_allowed:
                        return
                    if gpu_renderer.picker is None:
                        return
                    
                    # Get mouse position and queue hover update
                    pos = interactor.GetEventPosition()
                    gpu_renderer.update_hover(pos[0], pos[1])
                except Exception as e:
                    logger.debug(f"Drillhole hover failed: {e}")
            
            def on_left_click(obj, event):
                """Handle left click for selection - gated by PickingController."""
                try:
                    # Gate on PickingController
                    if not picking_ctrl.click_allowed:
                        return
                    if gpu_renderer.picker is None:
                        return
                    
                    # Get click position
                    pos = interactor.GetEventPosition()
                    
                    # Perform pick with timing
                    with picking_ctrl.timed_click():
                        result = gpu_renderer.picker.pick_at_position(pos[0], pos[1])
                        
                        if result:
                            interval, world_pos = result
                            gpu_renderer.update_selection({interval.color_id})
                            logger.debug(f"Drillhole selected: {interval.hole_id}")
                            gpu_renderer.event_bus.emit_interval_selected(interval)
                        else:
                            gpu_renderer.update_selection(set())
                except Exception as e:
                    logger.debug(f"Drillhole click failed: {e}")
            
            # Connect VTK events
            interactor.AddObserver("MouseMoveEvent", on_mouse_move)
            interactor.AddObserver("LeftButtonPressEvent", on_left_click)
            
            logger.info("Drillhole mouse interaction handlers connected (PickingController gated)")
            
        except Exception as e:
            logger.warning(f"Failed to setup drillhole interaction: {e}")
    
    def _setup_standard_drillhole_interaction(self, hole_polys: Dict, hole_segment_lith: Dict, 
                                               hole_segment_assay: Dict, database: DrillholeDatabase) -> None:
        """
        Set up click and hover for standard PyVista renderer (non-GPU).
        
        Uses PyVista's cell picker for accurate drillhole interaction.
        PERFORMANCE: Gated by PickingController for LOD-aware interaction.
        """
        if self.plotter is None or not hasattr(self, '_drillhole_hole_actors'):
            return
        
        try:
            # Store data for picking
            self._drillhole_pick_data = {
                'hole_polys': hole_polys,
                'hole_segment_lith': hole_segment_lith,
                'hole_segment_assay': hole_segment_assay,
                'database': database
            }
            
            # Create mapping from actor to hole_id
            actor_to_hole = {actor: hole_id for hole_id, actor in self._drillhole_hole_actors.items()}
            picking_ctrl = get_picking_controller()
            
            # Set up hover callback using PROP picker (actor-level ONLY, fast)
            def on_hover(obj, event):
                """Handle hover over drillholes - PROP PICKER ONLY (never cell picker)."""
                try:
                    # Gate on PickingController
                    if not picking_ctrl.hover_allowed:
                        return
                    
                    iren = obj
                    pos = iren.GetEventPosition()
                    
                    # CRITICAL: Use PROP picker for hover, NEVER cell picker
                    prop_picker = picking_ctrl.get_prop_picker()
                    if prop_picker is None:
                        return
                    prop_picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
                    
                    picked_actor = prop_picker.GetActor()
                    if picked_actor and picked_actor in actor_to_hole:
                        hole_id = actor_to_hole[picked_actor]
                        
                        # Visual feedback: highlight hovered hole (actor-level only)
                        for hid, actor in self._drillhole_hole_actors.items():
                            prop = actor.GetProperty()
                            if prop:
                                if hid == hole_id:
                                    prop.SetAmbient(0.8)
                                    prop.SetDiffuse(0.9)
                                else:
                                    prop.SetAmbient(0.4)
                                    prop.SetDiffuse(0.6)
                        
                        if self.plotter:
                            self.plotter.render()
                except Exception as e:
                    logger.debug(f"Drillhole hover failed: {e}")
            
            # Set up click callback
            def on_click(obj, event):
                """Handle click on drillholes - uses PROP picker (actor-level sufficient)."""
                try:
                    # Gate on PickingController
                    if not picking_ctrl.click_allowed:
                        return
                    
                    iren = obj
                    pos = iren.GetEventPosition()
                    
                    # For drillholes, we only need actor identification (which hole)
                    # so prop picker is sufficient even for clicks
                    with picking_ctrl.timed_click():
                        prop_picker = picking_ctrl.get_prop_picker()
                        if prop_picker is None:
                            return
                        prop_picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
                        
                        picked_actor = prop_picker.GetActor()
                        if picked_actor and picked_actor in actor_to_hole:
                            hole_id = actor_to_hole[picked_actor]
                            
                            # Visual feedback: highlight selected hole
                            for hid, actor in self._drillhole_hole_actors.items():
                                prop = actor.GetProperty()
                                if prop:
                                    if hid == hole_id:
                                        prop.SetAmbient(1.0)
                                        prop.SetDiffuse(1.0)
                                        prop.SetEdgeColor(1.0, 1.0, 0.0)
                                        prop.SetEdgeVisibility(True)
                                    else:
                                        prop.SetAmbient(0.4)
                                        prop.SetDiffuse(0.6)
                                        prop.SetEdgeVisibility(False)
                            
                            if self.plotter:
                                self.plotter.render()
                            
                            logger.debug(f"Drillhole clicked: {hole_id}")
                            
                            # Emit selection event
                            if hasattr(self, 'layer_change_callback') and self.layer_change_callback:
                                try:
                                    self.layer_change_callback("drillhole_selected", {
                                        "hole_id": hole_id,
                                        "interval": None
                                    })
                                except Exception:
                                    pass
                        else:
                            # Clicked empty space - clear selection
                            for actor in self._drillhole_hole_actors.values():
                                prop = actor.GetProperty()
                                if prop:
                                    prop.SetAmbient(0.4)
                                    prop.SetDiffuse(0.6)
                                    prop.SetEdgeVisibility(False)
                            if self.plotter:
                                self.plotter.render()
                            
                except Exception as e:
                    logger.debug(f"Drillhole click failed: {e}")
            
            # Connect VTK events
            # Fix: With pyvistaqt, the actual VTK interactor can be in different places
            try:
                iren = None
                # Try multiple paths to find the actual VTK interactor
                if hasattr(self.plotter, 'iren') and hasattr(self.plotter.iren, 'AddObserver'):
                    iren = self.plotter.iren
                elif hasattr(self.plotter, 'interactor'):
                    # Try plotter.interactor.interactor (wrapped interactor)
                    if hasattr(self.plotter.interactor, 'interactor') and hasattr(self.plotter.interactor.interactor, 'AddObserver'):
                        iren = self.plotter.interactor.interactor
                    # Try plotter.interactor directly
                    elif hasattr(self.plotter.interactor, 'AddObserver'):
                        iren = self.plotter.interactor

                if iren and hasattr(iren, 'AddObserver'):
                    iren.AddObserver('MouseMoveEvent', on_hover)
                    iren.AddObserver('LeftButtonPressEvent', on_click)
                    logger.info("Standard drillhole click/hover enabled (PickingController gated)")
                else:
                    logger.warning("Could not get VTK interactor for standard drillhole interaction")
            except Exception as e:
                logger.warning(f"Could not set up standard drillhole interaction: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to setup standard drillhole interaction: {e}")
    
    def _on_drillhole_interval_selected(self, interval: Optional[Any]) -> None:
        """Handle drillhole interval selection event."""
        if interval is None:
            return
        
        logger.info(f"Drillhole interval selected: {interval.hole_id} [{interval.depth_from:.1f}-{interval.depth_to:.1f}m]")
        
        # Store selected interval for info panel
        self._selected_drillhole_interval = interval
        
        # Update state manager
        if self._drillhole_state_manager:
            try:
                self._drillhole_state_manager.set_selection({interval.color_id}, append=False)
                # Update selected hole IDs
                self._drillhole_state_manager._state.selected_hole_ids = {interval.hole_id}
            except Exception as e:
                logger.debug(f"State manager selection update failed: {e}")
        
        # Could emit signal or update UI here
        if hasattr(self, 'layer_change_callback') and self.layer_change_callback:
            try:
                self.layer_change_callback("drillhole_selected", {
                    "hole_id": interval.hole_id,
                    "depth_from": interval.depth_from,
                    "depth_to": interval.depth_to,
                    "lith_code": interval.lith_code,
                    "assay_value": interval.assay_value,
                    "interval": interval,
                })
            except Exception:
                pass
    
    def focus_on_drillhole_interval(self, hole_id: str, depth_from: float, depth_to: float) -> None:
        """
        Focus camera on a specific drillhole interval.
        
        Args:
            hole_id: Hole ID to focus on
            depth_from: Start depth of interval
            depth_to: End depth of interval
        """
        if self.plotter is None:
            return
        
        try:
            # Get hole bounds from cached polylines
            if self._drillhole_polylines_cache is None:
                logger.warning("Cannot focus: no drillhole cache")
                return
            
            hole_polys = self._drillhole_polylines_cache.get("hole_polys", {})
            poly = hole_polys.get(hole_id)
            
            if poly is None:
                logger.warning(f"Hole {hole_id} not found in cache")
                return
            
            # Get bounds of the hole
            bounds = poly.bounds
            
            # Calculate center point of the interval
            center_x = (bounds[0] + bounds[1]) / 2
            center_y = (bounds[2] + bounds[3]) / 2
            center_z = (depth_from + depth_to) / 2
            
            focal_point = (center_x, center_y, center_z)
            
            # Calculate camera distance
            interval_length = abs(depth_to - depth_from)
            hole_extent_x = bounds[1] - bounds[0]
            hole_extent_y = bounds[3] - bounds[2]
            max_extent = max(hole_extent_x, hole_extent_y, interval_length, 10.0)  # Min 10m
            
            distance = max_extent * 2.5
            camera_pos = (
                center_x + distance * 0.7,
                center_y + distance * 0.7,
                center_z + distance * 0.5
            )
            
            self.plotter.camera_position = [camera_pos, focal_point, [0, 0, 1]]
            
            # QUALITY FIX: Very small near plane for close-up drillhole inspection
            try:
                camera = self.plotter.renderer.GetActiveCamera()
                if camera:
                    near_plane = max(0.001, min(distance * 0.0001, max_extent * 0.00001))
                    far_plane = max(max_extent * 50.0, distance * 100.0)
                    camera.SetClippingRange(near_plane, far_plane)
            except Exception:
                pass
            
            self.plotter.render()
            logger.info(f"Focused camera on hole {hole_id} interval [{depth_from:.1f}-{depth_to:.1f}m]")
            
        except Exception as e:
            logger.error(f"Failed to focus on drillhole interval: {e}", exc_info=True)
    
    def select_drillholes_in_box(self, bounds: Tuple[float, float, float, float, float, float]) -> Set[str]:
        """
        Select drillholes within a 3D bounding box.
        
        Args:
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            
        Returns:
            Set of hole IDs within the box
        """
        if self._drillhole_polylines_cache is None:
            return set()
        
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        hole_polys = self._drillhole_polylines_cache.get("hole_polys", {})
        collar_coords = self._drillhole_polylines_cache.get("collar_coords", {})
        
        selected_holes = set()
        
        for hole_id, poly in hole_polys.items():
            # Check if any part of the hole is within bounds
            poly_bounds = poly.bounds
            
            # Check if polyline bounds overlap with selection box
            if (poly_bounds[1] >= xmin and poly_bounds[0] <= xmax and
                poly_bounds[3] >= ymin and poly_bounds[2] <= ymax and
                poly_bounds[5] >= zmin and poly_bounds[4] <= zmax):
                selected_holes.add(hole_id)
        
        logger.info(f"Selected {len(selected_holes)} drillholes in box bounds")
        return selected_holes
    
    def focus_on_selected_drillholes(self, hole_ids: Set[str]) -> None:
        """
        Focus camera on selected drillholes.
        
        Args:
            hole_ids: Set of hole IDs to focus on
        """
        if self.plotter is None or not hole_ids:
            return
        
        try:
            if self._drillhole_polylines_cache is None:
                logger.warning("Cannot focus: no drillhole cache")
                return
            
            hole_polys = self._drillhole_polylines_cache.get("hole_polys", {})
            all_bounds = []
            
            for hole_id in hole_ids:
                poly = hole_polys.get(hole_id)
                if poly is not None:
                    all_bounds.append(poly.bounds)
            
            if not all_bounds:
                logger.warning("No valid holes found for focus")
                return
            
            # Calculate combined bounds
            min_x = min(b[0] for b in all_bounds)
            max_x = max(b[1] for b in all_bounds)
            min_y = min(b[2] for b in all_bounds)
            max_y = max(b[3] for b in all_bounds)
            min_z = min(b[4] for b in all_bounds)
            max_z = max(b[5] for b in all_bounds)
            
            center = (
                (min_x + max_x) / 2,
                (min_y + max_y) / 2,
                (min_z + max_z) / 2
            )
            
            extent_x = max_x - min_x
            extent_y = max_y - min_y
            extent_z = max_z - min_z
            max_extent = max(extent_x, extent_y, extent_z, 10.0)
            
            distance = max_extent * 2.0
            camera_pos = (
                center[0] + distance * 0.7,
                center[1] + distance * 0.7,
                center[2] + distance * 0.7
            )
            
            self.plotter.camera_position = [camera_pos, center, [0, 0, 1]]
            
            # QUALITY FIX: Very small near plane for close-up inspection
            try:
                camera = self.plotter.renderer.GetActiveCamera()
                if camera:
                    near_plane = max(0.001, min(distance * 0.0001, max_extent * 0.00001))
                    far_plane = max(max_extent * 50.0, distance * 100.0)
                    camera.SetClippingRange(near_plane, far_plane)
            except Exception:
                pass
            
            self.plotter.render()
            logger.info(f"Focused camera on {len(hole_ids)} selected drillholes")
            
        except Exception as e:
            logger.error(f"Failed to focus on selected drillholes: {e}", exc_info=True)
    
    def _on_drillhole_interval_hovered(self, interval: Optional[Any]) -> None:
        """Handle drillhole interval hover event."""
        # Hover is handled visually by GPU renderer
        # Could update tooltip or status bar here
        if interval:
            logger.debug(f"Hovering drillhole: {interval.hole_id} [{interval.depth_from:.1f}-{interval.depth_to:.1f}m]")
    
    def remove_drillhole_layer(self) -> None:
        """Remove the drillhole layer if it exists."""
        # CRITICAL: Clear GPU renderer first if it exists (DR-001 fix)
        if hasattr(self, '_gpu_drillhole_renderer') and self._gpu_drillhole_renderer is not None:
            try:
                # Disconnect event bus signals to prevent stale handlers (DR-004 fix)
                from .drillhole_gpu_renderer import get_drillhole_event_bus
                event_bus = get_drillhole_event_bus()
                try:
                    event_bus.intervalSelected.disconnect(self._on_drillhole_interval_selected)
                except (TypeError, RuntimeError):
                    pass  # Already disconnected or never connected
                try:
                    event_bus.intervalHovered.disconnect(self._on_drillhole_interval_hovered)
                except (TypeError, RuntimeError):
                    pass
                
                # Clear GPU renderer (stops timers, removes actors, clears state)
                self._gpu_drillhole_renderer.clear()
                logger.debug("[DRILLHOLE DEBUG] Cleared GPU drillhole renderer.")
            except Exception as e:
                logger.warning(f"Error clearing GPU drillhole renderer: {e}")
            finally:
                self._gpu_drillhole_renderer = None
        
        # Remove all individual hole actors (standard renderer)
        if self.plotter is not None:
            for hole_id, actor in self._drillhole_hole_actors.items():
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        removed = len(self._drillhole_hole_actors)
        self._drillhole_hole_actors.clear()
        logger.debug("[DRILLHOLE DEBUG] Removed %d drillhole actors.", removed)
        
        # Remove collar marker actors
        if self.plotter is not None and hasattr(self, '_drillhole_collar_actors'):
            for key, collar_actor in self._drillhole_collar_actors.items():
                if key == "_merged_hids":
                    continue  # Skip metadata key (list of hids, not actor)
                try:
                    self.plotter.remove_actor(collar_actor)
                except Exception:
                    pass
            self._drillhole_collar_actors.clear()
            logger.debug("[DRILLHOLE DEBUG] Removed collar marker actors.")
        
        if "drillholes" in self.active_layers:
            self.clear_layer("drillholes")
        
        # Remove labels if they exist
        if hasattr(self, "_drillhole_label_actor") and self._drillhole_label_actor is not None:
            try:
                if self.plotter is not None:
                    self.plotter.remove_actor(self._drillhole_label_actor)
            except Exception as e:
                logger.debug(f"Could not remove drillhole label actor: {e}")
            self._drillhole_label_actor = None
        
        # Clear cache with proper memory release (Phase 2.2 fix)
        if self._drillhole_polylines_cache is not None:
            # Clear nested structures explicitly to help garbage collection
            for key in list(self._drillhole_polylines_cache.keys()):
                val = self._drillhole_polylines_cache[key]
                if isinstance(val, dict):
                    val.clear()
                elif isinstance(val, list):
                    val.clear()
                elif hasattr(val, '__del__'):
                    # Clear any objects with destructors (e.g., numpy arrays)
                    try:
                        del self._drillhole_polylines_cache[key]
                    except Exception:
                        pass
            self._drillhole_polylines_cache.clear()
            self._drillhole_polylines_cache = None
            
            # Force garbage collection for large datasets
            import gc
            gc.collect()
            logger.debug("[DRILLHOLE DEBUG] Cache cleared with memory released")
        
        try:
            self._update_scene_bounds()
        except Exception as exc:
            logger.debug(f"Could not refresh scene bounds after removing drillholes: {exc}")
        try:
            self._maintain_clipping_range()
        except Exception as e:
            logger.debug(f"Could not maintain clipping range: {e}")

    def get_drillhole_legend_metadata(self) -> Optional[Dict[str, Any]]:
        """Return the last computed drillhole legend metadata."""
        metadata = self._drillhole_legend_metadata
        if metadata:
            cats = metadata.get("categories", [])
            logger.info(f"[GET LEGEND METADATA] Returning metadata with {len(cats) if cats else 0} categories: {cats[:5] if cats else 'None'}")
        return metadata
    
    def set_drillhole_visibility(self, hole_id: str, visible: bool) -> None:
        """
        Toggle visibility for a single drillhole.
        
        Phase 3.2 Fix: Works in both merged and individual mode.
        In merged mode, rebuilds the mesh excluding hidden holes.
        
        Args:
            hole_id: The hole ID to toggle
            visible: Whether the hole should be visible
        """
        # Update state manager if available
        try:
            from ..core.state_manager import get_state_manager
            state_manager = get_state_manager()
            state_manager.set_drillhole_visibility(hole_id, visible)
        except Exception:
            pass
        
        # Check if we're in merged mode
        if "_merged" in self._drillhole_hole_actors:
            # Merged mode: need to rebuild with visibility filter
            self._rebuild_merged_drillholes_with_visibility()
        else:
            # Individual mode: toggle VTK actor visibility directly
            if hole_id in self._drillhole_hole_actors:
                actor = self._drillhole_hole_actors[hole_id]
                try:
                    if visible:
                        actor.VisibilityOn()
                    else:
                        actor.VisibilityOff()
                except Exception as e:
                    logger.debug(f"Could not toggle drillhole visibility: {e}")
            
            # Also handle collar actor
            if hole_id in self._drillhole_collar_actors:
                collar_actor = self._drillhole_collar_actors[hole_id]
                try:
                    if visible:
                        collar_actor.VisibilityOn()
                    else:
                        collar_actor.VisibilityOff()
                except Exception as e:
                    logger.debug(f"Could not toggle collar visibility: {e}")
        
        # Render the changes
        if self.plotter is not None:
            try:
                self.plotter.render()
            except Exception:
                pass
    
    def _rebuild_merged_drillholes_with_visibility(self) -> None:
        """
        Rebuild merged drillhole mesh excluding hidden holes.
        
        Phase 3.2 Fix: When drillholes are merged for performance,
        visibility toggling requires rebuilding the merged geometry.
        """
        # Get visible holes from state manager
        try:
            from ..core.state_manager import get_state_manager
            state_manager = get_state_manager()
            visible_holes = state_manager.get_visible_holes()
        except Exception:
            # Fallback: get from layer data
            if "drillholes" in self.active_layers:
                layer_data = self.active_layers["drillholes"].get("data", {})
                visible_holes = set(layer_data.get("visible_holes", set()))
            else:
                return
        
        if not visible_holes:
            # Hide the merged actor entirely
            merged_actor = self._drillhole_hole_actors.get("_merged")
            if merged_actor:
                try:
                    merged_actor.VisibilityOff()
                except Exception:
                    pass
            return
        
        # Get cached data for rebuild
        cache = self._drillhole_polylines_cache
        if cache is None:
            logger.warning("No drillhole cache available for visibility rebuild")
            return
        
        hole_polys = cache.get("hole_polys", {})
        hole_segment_lith = cache.get("hole_segment_lith", {})
        lith_colors = cache.get("lith_colors", {})
        lith_to_index = cache.get("lith_to_index", {})
        
        if not hole_polys:
            return
        
        # Build meshes only for visible holes
        import pyvista as pv
        meshes_to_merge = []
        
        radius = getattr(self, '_drillhole_original_radius', 1.0)
        tube_resolution = 12  # Standard resolution
        
        for hid in visible_holes:
            if hid not in hole_polys:
                continue
            
            try:
                line = hole_polys[hid]
                if line is None or line.n_points < 2:
                    continue
                
                # Create tube
                tube = line.tube(radius=radius, n_sides=tube_resolution)
                
                # Apply coloring
                if hid in hole_segment_lith:
                    seg_liths = hole_segment_lith[hid]
                    lith_ids = np.array([lith_to_index.get(l, 0) for l in seg_liths], dtype=np.int32)
                    # Expand to tube cells
                    n_original_cells = len(seg_liths)
                    expansion = max(1, tube.n_cells // max(1, n_original_cells))
                    expanded = np.repeat(lith_ids, expansion)[:tube.n_cells]
                    if len(expanded) < tube.n_cells:
                        expanded = np.pad(expanded, (0, tube.n_cells - len(expanded)), constant_values=expanded[-1])
                    tube.cell_data['lith_id'] = expanded
                
                meshes_to_merge.append(tube)
            except Exception as e:
                logger.debug(f"Could not build tube for hole {hid}: {e}")
        
        if not meshes_to_merge:
            return
        
        # Merge all visible holes
        try:
            merged_mesh = pv.merge(meshes_to_merge)
        except Exception as e:
            logger.warning(f"Could not merge drillhole meshes: {e}")
            return
        
        # Remove old merged actor
        old_actor = self._drillhole_hole_actors.get("_merged")
        if old_actor:
            try:
                self.plotter.remove_actor(old_actor)
            except Exception:
                pass
        
        # Add new merged mesh
        try:
            scalar_name = "lith_id" if "lith_id" in merged_mesh.cell_data else None
            new_actor = self.plotter.add_mesh(
                merged_mesh,
                scalars=scalar_name,
                cmap="tab20",
                show_scalar_bar=False,
                name="drillholes_merged_visible"
            )
            self._drillhole_hole_actors["_merged"] = new_actor
            logger.debug(f"Rebuilt merged drillholes with {len(visible_holes)} visible holes")
        except Exception as e:
            logger.warning(f"Could not add merged drillhole mesh: {e}")
    
    def set_drillhole_labels_visible(self, visible: bool) -> None:
        """Show or hide drillhole ID labels."""
        if "drillholes" not in self.active_layers:
            return
        
        layer_data = self.active_layers["drillholes"].get("data", {})
        collar_coords = layer_data.get("collar_coords", {})
        
        if not collar_coords:
            return
        
        # Remove existing labels
        if hasattr(self, "_drillhole_label_actor") and self._drillhole_label_actor is not None:
            try:
                if self.plotter is not None:
                    self.plotter.remove_actor(self._drillhole_label_actor)
            except Exception:
                pass
            self._drillhole_label_actor = None
        
        if not visible or self.plotter is None:
            return
        
        # Add labels
        try:
            import numpy as np
            points = np.array(list(collar_coords.values()), dtype=float)
            labels = list(collar_coords.keys())
            
            if len(points) > 0:
                self._drillhole_label_actor = self.plotter.add_point_labels(
                    points,
                    labels,
                    font_size=10,
                    show_points=False,
                    shape=None,
                    text_color="black",
                    name="drillhole_labels",
                )
                if self.plotter is not None:
                    self.plotter.render()
        except Exception as e:
            logger.warning(f"Failed to add drillhole labels: {e}")
    
    def set_drillhole_visibility(self, hole_id: str, visible: bool) -> None:
        """Toggle visibility of a single drillhole and its collar marker (instant, no re-render)."""
        updated = False
        
        # ✅ FIX: Handle merged actor case (single actor for all holes)
        if "_merged" in self._drillhole_hole_actors:
            # Merged mode: can't toggle individual holes, but we track visibility state
            # and show/hide the merged actor based on whether ANY holes are visible
            # For now, just ensure merged actor is visible if any hole should be visible
            merged_actor = self._drillhole_hole_actors["_merged"]
            if visible:
                merged_actor.VisibilityOn()
                updated = True
            # Note: In merged mode, we don't hide if one hole is invisible
            # The merged actor stays visible as long as any hole should be visible
        elif hole_id in self._drillhole_hole_actors:
            actor = self._drillhole_hole_actors[hole_id]
            if visible:
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()
            updated = True
        
        # Also toggle collar marker
        # In merged mode, collar is a single mesh - can't toggle individual collars
        if "_merged" in self._drillhole_collar_actors:
            # Merged collar actor stays visible as long as any holes are visible
            # Individual toggle not supported in merged mode
            pass
        elif hole_id in self._drillhole_collar_actors:
            collar_actor = self._drillhole_collar_actors[hole_id]
            if visible:
                collar_actor.VisibilityOn()
            else:
                collar_actor.VisibilityOff()
            updated = True
        
        # Update state manager
        if self._drillhole_state_manager:
            try:
                self._drillhole_state_manager.set_visibility(hole_id, visible)
            except Exception as e:
                logger.debug(f"State manager update failed: {e}")
        
        if updated and self.plotter is not None:
            self.plotter.render()
            logger.debug(f"Toggled drillhole {hole_id} visibility: {visible}")
        elif not updated:
            logger.warning(f"Drillhole {hole_id} not found in actors cache")
    
    def set_all_drillholes_visible(self, visible_holes: Set[str]) -> None:
        """
        ULTRA-FAST batch visibility update.
        
        Uses VTK visibility flags (no geometry rebuild).
        Target: <5ms for 200 holes.
        """
        import time
        start = time.perf_counter()
        
        if self.plotter is None:
            return
        
        updated = 0
        
        # ✅ FIX: Handle merged actor case (single actor for all holes)
        if "_merged" in self._drillhole_hole_actors:
            merged_actor = self._drillhole_hole_actors["_merged"]
            # Show merged actor if ANY holes should be visible
            should_show = len(visible_holes) > 0
            current_visible = merged_actor.GetVisibility() > 0
            if should_show != current_visible:
                if should_show:
                    merged_actor.VisibilityOn()
                else:
                    merged_actor.VisibilityOff()
                updated += 1
                logger.debug(f"Merged drillhole actor visibility: {should_show}")
        else:
            # Individual actors mode
            for hole_id, actor in self._drillhole_hole_actors.items():
                should_show = hole_id in visible_holes
                current_visible = actor.GetVisibility() > 0
                if should_show != current_visible:
                    if should_show:
                        actor.VisibilityOn()
                    else:
                        actor.VisibilityOff()
                    updated += 1
        
        # Update collar markers
        if "_merged" in self._drillhole_collar_actors:
            # Merged collar mode - show/hide based on whether ANY holes are visible
            merged_collar = self._drillhole_collar_actors["_merged"]
            should_show = len(visible_holes) > 0
            current_visible = merged_collar.GetVisibility() > 0
            if should_show != current_visible:
                if should_show:
                    merged_collar.VisibilityOn()
                else:
                    merged_collar.VisibilityOff()
        else:
            # Individual collar actors mode
            for hole_id, collar_actor in self._drillhole_collar_actors.items():
                if hole_id.startswith("_"):
                    continue  # Skip metadata keys
                should_show = hole_id in visible_holes
                current_visible = collar_actor.GetVisibility() > 0
                if should_show != current_visible:
                    if should_show:
                        collar_actor.VisibilityOn()
                    else:
                        collar_actor.VisibilityOff()
        
        # Update state manager
        if self._drillhole_state_manager:
            try:
                # Get hole IDs from merged metadata or collar keys
                if "_merged_hids" in self._drillhole_collar_actors:
                    all_hole_ids = set(self._drillhole_collar_actors["_merged_hids"])
                else:
                    all_hole_ids = set(k for k in self._drillhole_collar_actors.keys() if not k.startswith("_"))
                for hole_id in all_hole_ids:
                    self._drillhole_state_manager.set_visibility(hole_id, hole_id in visible_holes)
            except Exception as e:
                logger.debug(f"State manager batch update failed: {e}")
        
        # Single render for all updates
        if updated > 0 and self.plotter is not None:
            self.plotter.render()
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"FAST visibility update: {updated} holes in {elapsed:.1f}ms")
    
    def _update_drillhole_lut_only(self, colormap: str, is_lithology: bool, lith_colors: Dict[str, Any]) -> bool:
        """
        Fast path: Update only the lookup table without rebuilding geometry.
        
        Returns True if update succeeded, False if fallback needed.
        """
        if not self._drillhole_hole_actors:
            return False
        
        try:
            # Get unique actors (may be batched)
            unique_actors = set(self._drillhole_hole_actors.values())
            
            for actor in unique_actors:
                mapper = actor.GetMapper()
                if mapper is None:
                    continue
                
                lut = mapper.GetLookupTable()
                if lut is None:
                    continue
                
                if is_lithology:
                    # Update lithology colormap - REGENERATE colors from the new colormap
                    import matplotlib.cm as cm
                    try:
                        cmap_obj = cm.get_cmap(colormap)
                    except Exception:
                        logger.warning(f"Invalid colormap '{colormap}', using 'tab10' as fallback")
                        cmap_obj = cm.get_cmap("tab10")

                    # Get number of lithology categories
                    n_colors = len(lith_colors) if lith_colors else lut.GetNumberOfTableValues()
                    if n_colors < 1:
                        n_colors = lut.GetNumberOfTableValues()

                    # Regenerate colors from the NEW colormap
                    lut.SetNumberOfTableValues(n_colors)
                    for i in range(n_colors):
                        sample = i / max(1, n_colors - 1)
                        rgba = cmap_obj(sample)
                        lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], 1.0)

                    # Also update the lith_colors cache with new colors
                    if lith_colors:
                        new_lith_colors = {}
                        for i, lith_code in enumerate(sorted(lith_colors.keys())):
                            sample = i / max(1, len(lith_colors) - 1)
                            rgba = cmap_obj(sample)
                            new_lith_colors[lith_code] = (rgba[0], rgba[1], rgba[2])
                        lith_colors.clear()
                        lith_colors.update(new_lith_colors)
                else:
                    # Update assay colormap - use the provided colormap parameter
                    import matplotlib.cm as cm
                    try:
                        cmap_obj = cm.get_cmap(colormap)
                    except Exception:
                        # Fallback to turbo if colormap is invalid
                        logger.warning(f"Invalid colormap '{colormap}', using 'turbo' as fallback")
                        cmap_obj = cm.get_cmap("turbo")
                    
                    n_colors = lut.GetNumberOfTableValues()
                    for i in range(n_colors):
                        rgba = cmap_obj(i / (n_colors - 1))
                        lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], 1.0)
                
                lut.Modified()
                mapper.Modified()
                actor.Modified()  # Ensure actor knows about the change
            
            return True
        except Exception as e:
            logger.debug(f"LUT update failed: {e}")
            return False
    
    def _update_drillhole_legend_fast(self, property_name: str, colormap: str, is_lithology: bool, 
                                      lith_colors: Dict[str, Any], cache: Dict[str, Any]) -> None:
        """Fast legend update for colormap-only changes."""
        if not hasattr(self, 'legend_manager') or self.legend_manager is None:
            return
        
        try:
            if is_lithology:
                # Update discrete legend with new colormap colors
                categories = sorted(list(lith_colors.keys()))
                
                # Convert lith_colors to RGBA format
                from PyQt6.QtGui import QColor
                rgba_colors = {}
                for cat, color_val in lith_colors.items():
                    if isinstance(color_val, str):
                        qc = QColor(color_val)
                        rgba_colors[cat] = (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
                    elif isinstance(color_val, (tuple, list)):
                        if len(color_val) == 3:
                            rgba_colors[cat] = (*color_val, 1.0)
                        elif len(color_val) == 4:
                            rgba_colors[cat] = tuple(color_val)
                        else:
                            rgba_colors[cat] = (0.5, 0.5, 0.5, 1.0)
                    else:
                        rgba_colors[cat] = (0.5, 0.5, 0.5, 1.0)
                
                # Map property name to display name
                display_name = "Lithology" if property_name in ["lith_id", "Lithology", "Drillhole Lithology"] else property_name
                
                self.legend_manager.update_discrete(
                    property_name=display_name,
                    categories=categories,
                    cmap_name=colormap,
                    category_colors=rgba_colors
                )
            else:
                # Update continuous legend - USE SAME VALUES AS ACTORS
                # Get assay_min and assay_max from cache (same values used for clim in actors)
                assay_min = cache.get("assay_min")
                assay_max = cache.get("assay_max")
                
                if assay_min is not None and assay_max is not None:
                    # Use the exact same range as the actors
                    display_name = property_name if property_name not in ["assay", "Assay"] else "Assay"
                    # Create data array with min/max to ensure legend matches exactly
                    legend_data = np.array([assay_min, assay_max], dtype=np.float32)
                    self.legend_manager.update_continuous(
                        property_name=display_name,
                        data=legend_data,
                        cmap_name=colormap
                    )
                    logger.info(f"Updated legend with cached range: [{assay_min:.4f}, {assay_max:.4f}]")
                else:
                    # Fallback: recalculate from data
                    hole_polys = cache.get("hole_polys", {})
                    hole_segment_assay = cache.get("hole_segment_assay", {})
                    assay_data = np.concatenate([hole_segment_assay.get(hid, []) for hid in hole_polys.keys()])
                    
                    if len(assay_data) > 0:
                        display_name = property_name if property_name not in ["assay", "Assay"] else "Assay"
                        self.legend_manager.update_continuous(
                            property_name=display_name,
                            data=np.array(assay_data, dtype=np.float32),
                            cmap_name=colormap
                        )
                        # Update cache with calculated values
                        cache["assay_min"] = float(np.min(assay_data))
                        cache["assay_max"] = float(np.max(assay_data))
                        logger.info(f"Updated legend with calculated range: [{cache['assay_min']:.4f}, {cache['assay_max']:.4f}]")
            
            # Update metadata
            if is_lithology:
                from PyQt6.QtGui import QColor
                rgba_metadata_colors = {}
                for cat, color_val in lith_colors.items():
                    if isinstance(color_val, str):
                        qc = QColor(color_val)
                        rgba_metadata_colors[cat] = (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
                    elif isinstance(color_val, (tuple, list)):
                        rgba_metadata_colors[cat] = tuple(color_val[:4]) if len(color_val) >= 4 else (*color_val[:3], 1.0)
                    else:
                        rgba_metadata_colors[cat] = (0.5, 0.5, 0.5, 1.0)
                
                self._drillhole_legend_metadata = {
                    "property": property_name,
                    "title": property_name,
                    "mode": "discrete",
                    "colormap": colormap,
                    "categories": list(lith_colors.keys()),
                    "category_colors": rgba_metadata_colors,
                    "vmin": None,
                    "vmax": None,
                    "scalar_name": property_name,
                    "color_mode": "Lithology",
                }
            else:
                self._drillhole_legend_metadata = {
                    "property": property_name,
                    "title": property_name,
                    "mode": "continuous",
                    "colormap": colormap,
                    "categories": None,
                    "category_colors": None,
                    "vmin": cache.get("assay_min"),
                    "vmax": cache.get("assay_max"),
                    "scalar_name": property_name,
                    "color_mode": "Assay",
                }
        except Exception as e:
            logger.warning(f"Failed to update legend in fast path: {e}")
    
    # ============================================================================
    # VISUAL DENSITY CONTROLLER METHODS
    # ============================================================================

    def set_drillhole_visual_density_enabled(self, enabled: bool) -> None:
        """
        Enable or disable automatic visual density adjustment for drillholes.

        Args:
            enabled: Whether to enable visual density control
        """
        self.drillhole_visual_density_enabled = enabled

        if self.visual_density_controller is not None:
            self.visual_density_controller.set_enabled(enabled)
            logger.info(f"Drillhole visual density {'enabled' if enabled else 'disabled'}")

    def get_drillhole_visual_density_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics from the visual density controller.

        Returns:
            Controller statistics or None if not available
        """
        if self.visual_density_controller is not None:
            return self.visual_density_controller.get_stats()
        return None

    def set_drillhole_visual_density_debug(self, debug_enabled: bool) -> None:
        """
        Enable or disable debug logging for visual density controller.

        Args:
            debug_enabled: Whether to enable debug logging
        """
        if self.visual_density_controller is not None:
            # Update the debug logging setting
            self.visual_density_controller.debug_logging = debug_enabled
            logger.info(f"Drillhole visual density debug logging {'enabled' if debug_enabled else 'disabled'}")

            # Reset stats when enabling debug
            if debug_enabled:
                self.visual_density_controller.reset_stats()

    def update_selected_drillholes_for_density(self, selected_hole_ids: Set[str]) -> None:
        """
        Update the visual density controller with currently selected drillhole IDs.
        Selected drillholes always remain at full detail regardless of camera distance.

        Args:
            selected_hole_ids: Set of selected hole IDs
        """
        if self.visual_density_controller is not None:
            self.visual_density_controller.set_selected_holes(selected_hole_ids)

    def update_drillhole_selection(self, selected_hole_ids: Set[str]) -> None:
        """
        Update drillhole selection state for both rendering and visual density control.

        This method should be called whenever drillhole selections change in the UI.

        Args:
            selected_hole_ids: Set of currently selected hole IDs
        """
        # Update visual density controller (selected holes stay full detail)
        self.update_selected_drillholes_for_density(selected_hole_ids)

        # If using GPU renderer, update its selection state too
        if hasattr(self, '_gpu_drillhole_renderer') and self._gpu_drillhole_renderer is not None:
            try:
                # Convert hole IDs to color IDs for GPU renderer
                selected_color_ids = set()
                for hole_id in selected_hole_ids:
                    intervals = self._gpu_drillhole_renderer.state.get_intervals_by_hole(hole_id)
                    selected_color_ids.update(iv.color_id for iv in intervals)

                if selected_color_ids:
                    self._gpu_drillhole_renderer.update_selection(selected_color_ids)
            except Exception as e:
                logger.debug(f"Failed to update GPU renderer selection: {e}")

        logger.debug(f"Updated drillhole selection: {len(selected_hole_ids)} holes selected")

    # ============================================================================
    # DRILLHOLE RADIUS METHODS
    # ============================================================================

    def update_drillhole_radius(self, radius: float) -> None:
        """
        ACTOR PERSISTENCE: Update geometry in-place without removing actors.
        
        CRITICAL FIX: To keep picking stable, we MUST NOT remove/re-add actors.
        Instead, we update the VBO (Vertex Buffer Object) directly by copying
        new geometry into the existing actor's mapper.
        
        This prevents:
        - Picking drift (actor IDs stay constant)
        - Video jitter (no actor removal/creation mid-frame)
        - Selection loss (actor references remain valid)
        """
        if self._drillhole_polylines_cache is None:
            return
        
        if self.plotter is None:
            logger.warning("Cannot update radius: plotter not initialized")
            return
        
        cache = self._drillhole_polylines_cache
        
        # Loop protection: Check if radius is effectively unchanged
        current_radius = cache.get("radius", -1.0)
        if abs(current_radius - radius) < 1e-4:
            logger.debug(f"Drillhole radius {radius} unchanged, skipping update")
            return
        
        # Update cache radius
        cache["radius"] = radius
        
        # Check if we have a merged actor (most common case)
        if "_merged" not in self._drillhole_hole_actors:
            logger.debug("No merged drillhole actor found, falling back to full rebuild")
            self.update_drillhole_radius_full_rebuild(radius)
            return
        
        actor = self._drillhole_hole_actors["_merged"]
        hole_polys = cache["hole_polys"]
        
        # Extract color data from cache for scalar mapping
        hole_segment_lith = cache.get("hole_segment_lith", {})
        hole_segment_assay = cache.get("hole_segment_assay", {})
        lith_to_index = cache.get("lith_to_index", {})
        color_mode = cache.get("color_mode", "Lithology")
        scalar_name = "lith_id" if color_mode == "Lithology" else "assay"
        
        # Generate new geometry with updated radius
        # CRITICAL: Map scalars onto polylines BEFORE merging to preserve colors
        polylines_to_merge = []
        for hid, poly in hole_polys.items():
            if poly is None or poly.n_cells < 1:
                continue
            
            # Map color scalars onto polyline (same logic as full_rebuild)
            if color_mode == "Lithology":
                lith_ids = [lith_to_index.get(lit, -1) for lit in hole_segment_lith.get(hid, [])]
                if lith_ids:
                    repeats = max(1, poly.n_cells // len(lith_ids))
                    mapped = np.repeat(lith_ids, repeats)
                    if len(mapped) < poly.n_cells:
                        mapped = np.pad(mapped, (0, poly.n_cells - len(mapped)), mode="edge")
                    elif len(mapped) > poly.n_cells:
                        mapped = mapped[:poly.n_cells]
                    poly.cell_data[scalar_name] = mapped.astype(int)
                else:
                    poly.cell_data[scalar_name] = np.zeros(poly.n_cells, dtype=int)
            else:
                assay_vals = hole_segment_assay.get(hid, [])
                if assay_vals:
                    repeats = max(1, poly.n_cells // len(assay_vals))
                    mapped = np.repeat(assay_vals, repeats)
                    if len(mapped) < poly.n_cells:
                        mapped = np.pad(mapped, (0, poly.n_cells - len(mapped)), mode="edge")
                    elif len(mapped) > poly.n_cells:
                        mapped = mapped[:poly.n_cells]
                    poly.cell_data[scalar_name] = np.array(mapped, dtype=np.float32)
                else:
                    poly.cell_data[scalar_name] = np.zeros(poly.n_cells, dtype=np.float32)
            
            polylines_to_merge.append(poly)
        
        if not polylines_to_merge:
            logger.warning("No polylines to update")
            return
        
        try:
            # Merge polylines (preserves cell_data scalars) and tube with new radius
            merged_polylines = pv.merge(polylines_to_merge)
            new_mesh = merged_polylines.tube(radius=radius, capping=False, n_sides=16)
            
            # NOTE: Do NOT apply _to_local_precision here!
            # The cached polylines are in world coordinates, same as the initial render.
            # Applying _to_local_precision would double-shift the geometry, causing
            # drillholes to disappear when radius is updated.
            
            # IN-PLACE UPDATE: Set new geometry on the existing actor's mapper
            # This keeps the actor identity stable for picking
            mapper = actor.GetMapper()
            
            # Preserve the existing lookup table before updating input data
            existing_lut = mapper.GetLookupTable()
            
            mapper.SetInputData(new_mesh)
            
            # CRITICAL: Configure mapper to use the scalar array for coloring
            # Without this, the mapper loses color configuration after SetInputData
            if scalar_name in new_mesh.cell_data:
                mapper.SetScalarModeToUseCellData()
                mapper.SelectColorArray(scalar_name)
                mapper.ScalarVisibilityOn()
                
                # Restore the lookup table (colormap) that was configured initially
                if existing_lut is not None:
                    mapper.SetLookupTable(existing_lut)
                    # Update scalar range for continuous data (assay mode)
                    if color_mode != "Lithology":
                        assay_min = cache.get("assay_min", 0.0)
                        assay_max = cache.get("assay_max", 1.0)
                        mapper.SetScalarRange(assay_min, assay_max)
            
            mapper.Modified()
            actor.Modified()
            
            logger.info(f"Updated drillhole radius to {radius} (in-place, actor preserved, colors retained)")
            
            # Force render
            if self.plotter:
                self.plotter.render()
                
        except Exception as e:
            logger.warning(f"In-place update failed: {e}, falling back to full rebuild")
            self.update_drillhole_radius_full_rebuild(radius)
    
    def update_drillhole_radius_full_rebuild(self, radius: float) -> None:
        """
        FALLBACK: Full geometry rebuild for radius update (slow but accurate).
        
        Use this if transform-based scaling causes issues.
        Updates cache and layer data to keep radius synchronized.
        """
        import time
        start = time.perf_counter()
        
        if self._drillhole_polylines_cache is None:
            logger.warning("Cannot update radius: no cached polylines")
            return
        
        if self.plotter is None:
            logger.warning("Cannot update radius: plotter not initialized")
            return
        
        cache = self._drillhole_polylines_cache
        
        # Loop protection: Check if radius is effectively unchanged
        current_radius = cache.get("radius", -1.0)
        if abs(current_radius - radius) < 1e-4:
            logger.debug(f"Drillhole radius {radius} unchanged, skipping rebuild")
            return
            
        # CRITICAL: Update cache radius immediately to keep in sync
        cache["radius"] = radius
        
        # Also update layer data
        if "drillholes" in self.active_layers:
            self.active_layers["drillholes"]["data"]["radius"] = radius
        
        hole_polys = cache["hole_polys"]
        hole_segment_lith = cache["hole_segment_lith"]
        hole_segment_assay = cache["hole_segment_assay"]
        lith_colors = cache["lith_colors"]
        lith_to_index = cache["lith_to_index"]
        assay_min = cache["assay_min"]
        assay_max = cache["assay_max"]
        color_mode = cache["color_mode"]
        collar_coords = cache["collar_coords"]
        
        # Get current colormap from legend metadata or use default
        current_colormap = "viridis"
        if self._drillhole_legend_metadata:
            current_colormap = self._drillhole_legend_metadata.get("colormap", "viridis")
        
        scalar_name = "lith_id" if color_mode == "Lithology" else "assay"
        
        # Remove old actors
        for hole_id, actor in self._drillhole_hole_actors.items():
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        
        self._drillhole_hole_actors.clear()
        
        # Get current visible holes from layer data
        visible_holes = set()
        if "drillholes" in self.active_layers:
            layer_data = self.active_layers["drillholes"].get("data", {})
            visible_holes = set(layer_data.get("visible_holes", set()))
        
        # Adaptive quality based on dataset size
        # Minimum 12 sides ensures tubes look cylindrical, not polygonal
        total_holes = len(hole_polys)
        n_sides = 16
        if total_holes > 500:
            n_sides = 12
        elif total_holes > 200:
            n_sides = 14
        
        # ULTRA-OPTIMIZED: Merge polylines first, then apply single tube filter
        polylines_to_merge = []
        
        for hid in hole_polys.keys():
            poly = hole_polys.get(hid)
            if poly is None or poly.n_cells < 1:
                continue
            
            # Map colors on polyline (before tube)
            if color_mode == "Lithology":
                lith_ids = [lith_to_index.get(lit, -1) for lit in hole_segment_lith.get(hid, [])]
                if lith_ids:
                    repeats = max(1, poly.n_cells // len(lith_ids))
                    mapped = np.repeat(lith_ids, repeats)
                    if len(mapped) < poly.n_cells:
                        mapped = np.pad(mapped, (0, poly.n_cells - len(mapped)), mode="edge")
                    elif len(mapped) > poly.n_cells:
                        mapped = mapped[:poly.n_cells]
                    poly.cell_data[scalar_name] = mapped.astype(int)
                else:
                    poly.cell_data[scalar_name] = np.zeros(poly.n_cells, dtype=int)
            else:
                assay_vals = hole_segment_assay.get(hid, [])
                if assay_vals:
                    repeats = max(1, poly.n_cells // len(assay_vals))
                    mapped = np.repeat(assay_vals, repeats)
                    if len(mapped) < poly.n_cells:
                        mapped = np.pad(mapped, (0, poly.n_cells - len(mapped)), mode="edge")
                    elif len(mapped) > poly.n_cells:
                        mapped = mapped[:poly.n_cells]
                    poly.cell_data[scalar_name] = np.array(mapped, dtype=np.float32)
                else:
                    poly.cell_data[scalar_name] = np.zeros(poly.n_cells, dtype=np.float32)
            
            polylines_to_merge.append(poly)
        
        if not polylines_to_merge:
            logger.warning("No polylines to render after filtering")
            return
        
        # Merge polylines (FAST - just concatenates points/cells)
        try:
            merged_polylines = pv.merge(polylines_to_merge)
            logger.info(f"Radius update: merged {len(polylines_to_merge)} polylines")
        except Exception as e:
            logger.error(f"Polyline merge failed: {e}")
            return
        
        # Apply SINGLE tube filter
        # Use capping=False for smooth tube junctions between segments
        try:
            merged = merged_polylines.tube(radius=radius, capping=False, n_sides=n_sides)
            logger.info(f"Radius update: tube mesh has {merged.n_cells} cells")
        except Exception as e:
            logger.error(f"Tube filter failed: {e}")
            merged = None
        
        if merged is not None:
            # Single batched actor
            if color_mode == "Lithology":
                lith_cmap = self._build_lithology_colormap(current_colormap, lith_colors, lith_to_index)
                actor = self.plotter.add_mesh(
                    merged,
                    scalars=scalar_name,
                    cmap=lith_cmap,
                    show_scalar_bar=False,
                    reset_camera=False,
                    smooth_shading=True,
                    pbr=False,
                    lighting=True,
                    specular=0.3,
                    specular_power=15,
                    ambient=0.3,
                    diffuse=0.8,
                    interpolate_before_map=True,
                    name="drillholes_batched"
                )
            else:
                actor = self.plotter.add_mesh(
                    merged,
                    scalars=scalar_name,
                    clim=[assay_min, assay_max],
                    cmap="turbo",
                    show_scalar_bar=False,
                    reset_camera=False,
                    smooth_shading=True,
                    pbr=False,
                    lighting=True,
                    specular=0.3,
                    specular_power=15,
                    ambient=0.3,
                    diffuse=0.8,
                    interpolate_before_map=True,
                    name="drillholes_batched"
                )
            # Store single merged actor
            self._drillhole_merged_actor = actor
            self._drillhole_merged_mesh = merged
            # Keep dict for compatibility (single entry)
            self._drillhole_hole_actors = {"_merged": actor}
            
            # ✅ FIX: Ensure merged actor is visible after creation
            actor.VisibilityOn()
            actor.Modified()
            logger.info(f"Merged drillhole actor created and set visible")
        else:
            # Fallback: This shouldn't happen, but handle gracefully
            logger.error("Tube filter failed - no drillholes rendered")
            return
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Radius update completed: {len(polylines_to_merge)} holes in {elapsed:.1f}ms")
        
        # Update collar markers with new radius (MERGED for performance)
        if hasattr(self, '_drillhole_collar_actors'):
            for key, collar_actor in list(self._drillhole_collar_actors.items()):
                if key == "_merged_hids":
                    continue  # Skip metadata key
                try:
                    self.plotter.remove_actor(collar_actor)
                except Exception:
                    pass
            self._drillhole_collar_actors.clear()
            
            collar_radius = radius * 1.5
            collar_height = radius * 0.8
            
            # Collect all collar meshes for merging
            collar_meshes = []
            visible_collar_hids = []
            
            for hid in hole_polys.keys():
                collar = collar_coords.get(hid)
                if collar is None:
                    continue
                if visible_holes and hid not in visible_holes:
                    continue
                cx, cy, cz = collar
                visible_collar_hids.append(hid)
                
                # Create cone marker
                cone = pv.Cone(
                    center=(cx, cy, cz + collar_height * 0.5),
                    direction=(0, 0, -1),
                    height=collar_height,
                    radius=collar_radius,
                    resolution=8,
                    capping=True
                )
                collar_meshes.append(cone)
            
            # Merge all collars into single mesh
            if collar_meshes:
                if len(collar_meshes) == 1:
                    merged_collars = collar_meshes[0]
                else:
                    merged_collars = collar_meshes[0]
                    for m in collar_meshes[1:]:
                        merged_collars = merged_collars.merge(m)
                
                collar_actor = self.plotter.add_mesh(
                    merged_collars,
                    color="#000000",  # Black color for collars
                    show_scalar_bar=False,
                    reset_camera=False,
                    lighting=True,
                    specular=0.6,
                    specular_power=20,
                    ambient=0.4,
                    diffuse=0.7,
                    name="drillhole_collars_merged"
                )
                collar_actor.VisibilityOn()
                self._drillhole_collar_actors["_merged"] = collar_actor
                self._drillhole_collar_actors["_merged_hids"] = visible_collar_hids
        
        # Update layer data
        if "drillholes" in self.active_layers:
            self.active_layers["drillholes"]["data"]["radius"] = radius
        
        logger.info(f"Updated drillhole radius to {radius} for {len(self._drillhole_hole_actors)} holes")
        
        if self.plotter is not None:
            self.plotter.render()
    
    def _update_drillhole_colors(self, property_name: str, colormap: str, color_mode: str, custom_colors: Optional[Dict[Any, str]] = None) -> None:
        """
        Optimized drillhole color update - only updates colors/colormap, not geometry.
        
        Uses batched updates and only rebuilds if necessary (property switch).
        """
        import time
        start = time.perf_counter()
        
        if self._drillhole_polylines_cache is None or "drillholes" not in self.active_layers:
            logger.warning("Cannot update drillhole colors: no cached data or layer missing")
            return
        
        if self.plotter is None:
            logger.warning("Cannot update drillhole colors: plotter not initialized")
            return
        
        self.current_property = property_name
        self.current_colormap = colormap

        cache = self._drillhole_polylines_cache
        layer_data = self.active_layers["drillholes"].get("data", {})
        
        # Get current radius from layer data (may have been updated)
        radius = layer_data.get("radius", cache.get("radius", 1.0))
        cache["radius"] = radius  # Sync cache with layer data
        
        # Determine if we're switching between lithology and assay, OR changing assay field
        # Normalize property name - handle various formats from property panel
        normalized_property = property_name.lower().strip()
        is_lithology = (
            normalized_property in ["lithology", "lith_id", "lith_code", "drillhole lithology"] or
            "lith" in normalized_property
        )
        
        current_color_mode = cache.get("color_mode", "Lithology")
        current_assay_field = cache.get("assay_field", "")
        
        # Property switch = changing lithology↔assay OR changing assay field (FE→SIO2)
        is_mode_switch = (is_lithology and current_color_mode != "Lithology") or (not is_lithology and current_color_mode == "Lithology")
        # Compare assay fields case-insensitively to avoid spurious rebuilds
        is_assay_field_switch = (
            not is_lithology
            and (not current_assay_field or current_assay_field.lower() != property_name.lower())
        )
        is_property_switch = is_mode_switch or is_assay_field_switch
        
        # If we're in assay mode and the assay field changed, rebuild assay segments/range
        # so colors and legend reflect the newly selected element (e.g. FE vs SIO2).
        if not is_lithology and is_assay_field_switch:
            try:
                database = cache.get("database")
                composite_df = cache.get("composite_df")
                if database is not None:
                    from ..drillholes.drillhole_layer import build_drillhole_polylines
                    rebuild = build_drillhole_polylines(
                        database=database,
                        composite_df=composite_df,
                        assay_field_name=property_name,
                    )
                    
                    # CRITICAL FIX: Apply coordinate shift to rebuilt polylines
                    # Without this, the new polylines are in world coordinates while
                    # the scene uses local coordinates, causing a coordinate mismatch.
                    new_hole_polys = rebuild["hole_polys"]
                    new_collar_coords = rebuild["collar_coords"]
                    
                    if self._global_shift is not None:
                        for hid, poly in new_hole_polys.items():
                            if poly.n_points > 0:
                                local_points = self._to_local_precision(poly.points.copy())
                                poly.points = local_points
                        
                        # Also transform collar coordinates
                        for hid in new_collar_coords:
                            cx, cy, cz = new_collar_coords[hid]
                            new_collar_coords[hid] = (
                                cx - self._global_shift[0],
                                cy - self._global_shift[1],
                                cz - self._global_shift[2]
                            )
                        logger.debug("Applied coordinate shift to rebuilt drillhole polylines")
                    
                    # Update cache with freshly computed assay values and range
                    cache["hole_polys"] = new_hole_polys
                    cache["hole_segment_lith"] = rebuild["hole_segment_lith"]
                    cache["hole_segment_assay"] = rebuild["hole_segment_assay"]
                    cache["lith_colors"] = rebuild["lith_colors"]
                    cache["lith_to_index"] = rebuild["lith_to_index"]
                    cache["assay_field"] = rebuild["assay_field"]
                    cache["assay_min"] = rebuild["assay_min"]
                    cache["assay_max"] = rebuild["assay_max"]
                    cache["hole_ids"] = rebuild["hole_ids"]
                    cache["collar_coords"] = new_collar_coords
                    logger.info(
                        "Rebuilt drillhole assay data for field '%s' with range (%.4f, %.4f)",
                        cache["assay_field"],
                        cache["assay_min"],
                        cache["assay_max"],
                    )
            except Exception as e:
                logger.warning(f"Failed to rebuild assay data for field '{property_name}': {e}")
        
        # Update cache color mode and assay field
        cache["color_mode"] = "Lithology" if is_lithology else "Assay"
        if not is_lithology:
            cache["assay_field"] = property_name  # Update current assay field for tracking
        
        hole_polys = cache["hole_polys"]
        hole_segment_lith = cache["hole_segment_lith"]
        hole_segment_assay = cache["hole_segment_assay"]
        lith_colors = cache["lith_colors"]
        lith_to_index = cache["lith_to_index"]
        assay_field = cache.get("assay_field", "")
        assay_min = cache.get("assay_min", 0.0)
        assay_max = cache.get("assay_max", 1.0)
        visible_holes = set(layer_data.get("visible_holes", set()))
        
        # Check if actors have scalar data - if not, force full rebuild
        # This happens when actors are first created with color='lightgray' (no scalars)
        actors_have_scalars = False
        if self._drillhole_hole_actors:
            sample_actor = next(iter(self._drillhole_hole_actors.values()), None)
            if sample_actor:
                mapper = sample_actor.GetMapper()
                if mapper:
                    input_data = mapper.GetInput()
                    if input_data and input_data.GetCellData().GetScalars() is not None:
                        actors_have_scalars = True
        
        # OPTIMIZATION: If only colormap changed (not property), update LUT only
        # BUT only if actors already have scalar data AND no custom colors are defined
        # Custom colors require a full rebuild to apply the user-defined color mapping
        has_custom_colors = custom_colors is not None and len(custom_colors) > 0
        if not is_property_switch and len(self._drillhole_hole_actors) > 0 and actors_have_scalars and not has_custom_colors:
            # Fast path: Update lookup table only
            try:
                updated = self._update_drillhole_lut_only(colormap, is_lithology, lith_colors)
                if updated:
                    # CRITICAL: Also update legend even in fast path
                    self._update_drillhole_legend_fast(property_name, colormap, is_lithology, lith_colors, cache)

                    # Force render to ensure visual update
                    if self.plotter is not None:
                        self.plotter.render()

                    elapsed = (time.perf_counter() - start) * 1000
                    logger.info(f"Fast color update (LUT only): {elapsed:.1f}ms")
                    return
            except Exception as e:
                logger.debug(f"Fast LUT update failed, falling back to full rebuild: {e}")
        
        # Full rebuild path (property switch or first time)
        # Remove old actors
        for hole_id, actor in self._drillhole_hole_actors.items():
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        
        self._drillhole_hole_actors.clear()
        
        scalar_name = "lith_id" if is_lithology else "assay"
        
        # Apply colormap changes
        # Check if custom colors are defined for assay properties
        # If so, treat as discrete mode
        use_custom_assay_colors = (not is_lithology and color_mode == "discrete" and custom_colors and len(custom_colors) > 0)
        
        if is_lithology:
            if color_mode == "discrete" and custom_colors:
                updated_lith_colors = {}
                for lith_code, color_str in custom_colors.items():
                    if color_str.startswith('#'):
                        color_str_clean = color_str.lstrip('#')
                        rgb = tuple(int(color_str_clean[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                        updated_lith_colors[lith_code] = rgb
                    else:
                        try:
                            from matplotlib.colors import to_rgb
                            updated_lith_colors[lith_code] = to_rgb(color_str)
                        except Exception:
                            updated_lith_colors[lith_code] = (0.5, 0.5, 0.5)
                if updated_lith_colors:
                    lith_colors = updated_lith_colors
                    # Update cache with custom colors so legend uses them too
                    cache["lith_colors"] = lith_colors
                    logger.info(f"Applied custom colors for lithology: {list(updated_lith_colors.keys())}")
            else:
                # Regenerate colors from the new colormap when colormap changes
                # Only preserve original colors if custom_colors were explicitly provided (non-empty dict)
                if custom_colors is None or (isinstance(custom_colors, dict) and len(custom_colors) == 0):
                    # User changed colormap - regenerate colors from new colormap
                    unique_liths = sorted(list(set(lith_to_index.keys())))
                    try:
                        import matplotlib.cm as cm
                        cmap = cm.get_cmap(colormap)
                        n = len(unique_liths)
                        if n <= 1:
                            samples = [0.0]
                        else:
                            samples = np.linspace(0.0, 1.0, n)
                        lith_colors = {}
                        for i, lith_code in enumerate(unique_liths):
                            color = cmap(samples[i])
                            lith_colors[lith_code] = tuple(color[:3])  # RGB only
                        logger.info(f"Regenerated {len(lith_colors)} lithology colors from colormap '{colormap}'")
                    except Exception as e:
                        logger.warning(f"Failed to apply colormap '{colormap}' to lithology: {e}")
                        # Fallback: keep original colors if regeneration fails
                        if not lith_colors or len(lith_colors) == 0:
                            # Last resort: use default colors
                            unique_liths = sorted(list(set(lith_to_index.keys())))
                            for i, lith_code in enumerate(unique_liths):
                                lith_colors[lith_code] = (0.5, 0.5, 0.5)
                # else: custom_colors were provided, use those (handled above)

        # Handle custom colors for assay properties
        if use_custom_assay_colors:
            # Convert custom_colors to lith_colors format for discrete rendering
            updated_assay_colors = {}
            for key, color_str in custom_colors.items():
                if color_str.startswith('#'):
                    color_str = color_str.lstrip('#')
                    rgb = tuple(int(color_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                    updated_assay_colors[key] = rgb
                else:
                    try:
                        from matplotlib.colors import to_rgb
                        updated_assay_colors[key] = to_rgb(color_str)
                    except Exception:
                        updated_assay_colors[key] = (0.5, 0.5, 0.5)
            lith_colors = updated_assay_colors
            is_lithology = True  # Switch to discrete rendering mode
            scalar_name = "custom_assay_category"
        
        # ULTRA-OPTIMIZED: Merge polylines first, then apply single tube filter
        if not use_custom_assay_colors:
            scalar_name = "lith_id" if is_lithology else "assay"
        
        # Adaptive quality based on dataset size
        # Minimum 12 sides ensures tubes look cylindrical, not polygonal
        total_holes = len(hole_polys)
        n_sides = 16
        if total_holes > 500:
            n_sides = 12
        elif total_holes > 200:
            n_sides = 14
        
        # Collect polylines with scalar data
        polylines_to_merge = []
        
        for hid in hole_polys.keys():
            if hid not in visible_holes and visible_holes:
                continue  # Skip hidden holes
            
            poly = hole_polys.get(hid)
            if poly is None or poly.n_cells < 1:
                continue
            
            # Map scalars on polyline (before tube)
            if is_lithology:
                lith_ids = [lith_to_index.get(lit, -1) for lit in hole_segment_lith.get(hid, [])]
                if lith_ids:
                    repeats = max(1, poly.n_cells // len(lith_ids))
                    mapped = np.repeat(lith_ids, repeats)
                    if len(mapped) < poly.n_cells:
                        mapped = np.pad(mapped, (0, poly.n_cells - len(mapped)), mode="edge")
                    elif len(mapped) > poly.n_cells:
                        mapped = mapped[:poly.n_cells]
                    poly.cell_data[scalar_name] = mapped.astype(int)
                else:
                    poly.cell_data[scalar_name] = np.zeros(poly.n_cells, dtype=int)
            else:
                assay_vals = hole_segment_assay.get(hid, [])
                if assay_vals:
                    repeats = max(1, poly.n_cells // len(assay_vals))
                    mapped = np.repeat(assay_vals, repeats)
                    if len(mapped) < poly.n_cells:
                        mapped = np.pad(mapped, (0, poly.n_cells - len(mapped)), mode="edge")
                    elif len(mapped) > poly.n_cells:
                        mapped = mapped[:poly.n_cells]
                    poly.cell_data[scalar_name] = np.array(mapped, dtype=np.float32)
                else:
                    poly.cell_data[scalar_name] = np.zeros(poly.n_cells, dtype=np.float32)
            
            polylines_to_merge.append(poly)
        
        if not polylines_to_merge:
            logger.warning("No polylines to render after filtering")
            return
        
        # Merge polylines (FAST - just concatenates points/cells)
        try:
            merged_polylines = pv.merge(polylines_to_merge)
            logger.info(f"Color update: merged {len(polylines_to_merge)} polylines")
        except Exception as e:
            logger.error(f"Polyline merge failed: {e}")
            return
        
        # Apply SINGLE tube filter
        # Use capping=False for smooth tube junctions between segments
        try:
            merged = merged_polylines.tube(radius=radius, capping=False, n_sides=n_sides)
            logger.info(f"Color update: tube mesh has {merged.n_cells} cells")
        except Exception as e:
            logger.error(f"Tube filter failed: {e}")
            merged = None
        
        if merged is not None:
            # Single batched actor
            if is_lithology:
                lith_cmap = self._build_lithology_colormap(colormap, lith_colors, lith_to_index)
                actor = self.plotter.add_mesh(
                    merged,
                    scalars=scalar_name,
                    cmap=lith_cmap,
                    show_scalar_bar=False,
                    reset_camera=False,
                    smooth_shading=True,
                    pbr=False,
                    lighting=True,
                    specular=0.3,
                    specular_power=15,
                    ambient=0.3,
                    diffuse=0.8,
                    interpolate_before_map=True,
                    name="drillholes_batched"
                )
            else:
                actor = self.plotter.add_mesh(
                    merged,
                    scalars=scalar_name,
                    clim=[assay_min, assay_max],
                    cmap="turbo",
                    show_scalar_bar=False,
                    reset_camera=False,
                    smooth_shading=True,
                    pbr=False,
                    lighting=True,
                    specular=0.3,
                    specular_power=15,
                    ambient=0.3,
                    diffuse=0.8,
                    interpolate_before_map=True,
                    name="drillholes_batched"
                )
            # Store single merged actor
            self._drillhole_merged_actor = actor
            self._drillhole_merged_mesh = merged
            # Keep dict for compatibility (single entry)
            self._drillhole_hole_actors = {"_merged": actor}
            
            # CRITICAL FIX: Explicitly ensure the merged actor is visible
            # This is essential when rebuilding from gray actors to colored actors
            if actor is not None:
                try:
                    actor.VisibilityOn()
                    logger.info(f"Explicitly enabled visibility for merged drillhole actor (cells={merged.n_cells})")
                except Exception as e:
                    logger.warning(f"Failed to set visibility on merged actor: {e}")
            
            # Force immediate render to ensure actor appears
            if self.plotter is not None:
                try:
                    self.plotter.render()
                    logger.debug("Forced render after creating merged drillhole actor")
                except Exception as e:
                    logger.warning(f"Failed to force render: {e}")
        else:
            # Fallback: This shouldn't happen, but handle gracefully
            logger.error("Tube filter failed - no drillholes rendered")
            return
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Color update completed: {len(polylines_to_merge)} holes in {elapsed:.1f}ms")
        
        # Update cache with new colors and color mode
        cache["lith_colors"] = lith_colors
        cache["color_mode"] = "Lithology" if is_lithology else "Assay"
        logger.info(f"[CACHE UPDATE] Updated cache lith_colors with {len(lith_colors)} entries: {list(lith_colors.keys())[:5]}")
        
        # Update legend
        if hasattr(self, 'legend_manager') and self.legend_manager is not None:
            if is_lithology:
                # Sort categories for consistent ordering
                categories = sorted(list(lith_colors.keys()))
                
                # Convert lith_colors to RGBA format for legend
                from PyQt6.QtGui import QColor
                rgba_colors = {}
                for cat, color_val in lith_colors.items():
                    if isinstance(color_val, str):
                        # Hex string
                        qc = QColor(color_val)
                        rgba_colors[cat] = (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
                    elif isinstance(color_val, (tuple, list)):
                        if len(color_val) == 3:
                            # RGB - add alpha
                            rgba_colors[cat] = (*color_val, 1.0)
                        elif len(color_val) == 4:
                            # Already RGBA
                            rgba_colors[cat] = tuple(color_val)
                        else:
                            rgba_colors[cat] = (0.5, 0.5, 0.5, 1.0)
                    else:
                        rgba_colors[cat] = (0.5, 0.5, 0.5, 1.0)
                
                # Use actual property name from panel, not hardcoded string
                display_property = property_name if property_name else "Lithology"
                self.legend_manager.update_discrete(
                    property_name=display_property,
                    categories=categories,
                    cmap_name=colormap,
                    category_colors=rgba_colors
                )
            else:
                # For assay, use continuous legend
                assay_data = np.concatenate([hole_segment_assay.get(hid, []) for hid in hole_polys.keys()])
                if len(assay_data) > 0:
                    # Use actual property name from panel
                    display_property = property_name if property_name else "Assay"
                    self.legend_manager.update_continuous(
                        property_name=display_property,
                        data=np.array(assay_data, dtype=np.float32),
                        cmap_name=colormap
                    )

        # Build legend metadata with RGBA colors
        if is_lithology:
            # Convert to RGBA for metadata
            from PyQt6.QtGui import QColor
            rgba_metadata_colors = {}
            for cat, color_val in lith_colors.items():
                if isinstance(color_val, str):
                    qc = QColor(color_val)
                    rgba_metadata_colors[cat] = (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
                elif isinstance(color_val, (tuple, list)):
                    if len(color_val) == 3:
                        rgba_metadata_colors[cat] = (*color_val, 1.0)
                    elif len(color_val) == 4:
                        rgba_metadata_colors[cat] = tuple(color_val)
                    else:
                        rgba_metadata_colors[cat] = (0.5, 0.5, 0.5, 1.0)
                else:
                    rgba_metadata_colors[cat] = (0.5, 0.5, 0.5, 1.0)
            
            self._drillhole_legend_metadata = {
                "property": "Drillhole Lithology",
                "title": "Drillhole Lithology",
                "mode": "discrete",
                "colormap": colormap,
                "categories": list(lith_colors.keys()),
                "category_colors": rgba_metadata_colors,
                "vmin": None,
                "vmax": None,
                "scalar_name": property_name,
                "color_mode": "Lithology",
            }
        else:
            self._drillhole_legend_metadata = {
                "property": property_name,
                "title": property_name,
                "mode": "continuous",
                "colormap": colormap,
                "categories": None,
                "category_colors": None,
                "vmin": cache.get("assay_min"),
                "vmax": cache.get("assay_max"),
                "scalar_name": property_name,
                "color_mode": "Assay",
            }
        
        # Phase 3.3 Fix: Sync legend with state manager for centralized state
        try:
            from ..core.state_manager import get_state_manager
            state_manager = get_state_manager()
            
            metadata = self._drillhole_legend_metadata
            state_manager.update_legend(
                property_name=metadata.get("property", property_name),
                vmin=metadata.get("vmin"),
                vmax=metadata.get("vmax"),
                colormap=metadata.get("colormap", colormap),
                categories=metadata.get("categories"),
                category_colors=metadata.get("category_colors"),
            )
            logger.debug(f"Synced legend state for {property_name}")
        except Exception as e:
            logger.debug(f"Could not sync legend state: {e}")
        
        logger.info(f"Updated drillhole colors: property={property_name}, colormap={colormap}, mode={color_mode}")
        
        if self.plotter is not None:
            self.plotter.render()
    
    def _build_lithology_colormap(self, base_cmap: str, lith_colors: Dict[Any, Any],
                                    lith_to_index: Dict[Any, int] = None) -> Any:
        """Build a ListedColormap for lithology colors, fallback to base if not enough data.

        Args:
            base_cmap: Base colormap name for fallback
            lith_colors: Dictionary mapping lithology codes to colors (hex strings or RGB tuples)
            lith_to_index: Dictionary mapping lithology codes to indices (for correct ordering)

        Returns:
            ListedColormap with colors ordered by lith_to_index indices
        """
        import matplotlib.cm as cm
        from matplotlib.colors import to_rgb

        if not lith_colors:
            try:
                return cm.get_cmap(base_cmap)
            except Exception:
                return cm.get_cmap("tab10")

        # Build colors in the correct order based on lith_to_index
        # This ensures color[i] corresponds to scalar value i
        if lith_to_index:
            # Sort lithology codes by their index
            sorted_codes = sorted(lith_to_index.keys(), key=lambda x: lith_to_index[x])
            color_items = [(code, lith_colors.get(code, (0.5, 0.5, 0.5))) for code in sorted_codes]
        else:
            # Fallback: use sorted order of keys
            color_items = [(code, color) for code, color in sorted(lith_colors.items())]

        normalized = []
        try:
            for code, color in color_items:
                if isinstance(color, str):
                    # Handle hex string like "#E6194B" or color name
                    try:
                        rgb = to_rgb(color)
                        normalized.append(rgb)
                    except Exception:
                        normalized.append((0.5, 0.5, 0.5))  # Gray fallback
                elif isinstance(color, (list, tuple)) and len(color) >= 3:
                    # Handle RGB tuple (0-1 or 0-255 range)
                    channels = []
                    for c in color[:3]:
                        value = float(c)
                        if value > 1.0:
                            value = value / 255.0
                        channels.append(min(1.0, max(0.0, value)))
                    normalized.append(tuple(channels))
                else:
                    normalized.append((0.5, 0.5, 0.5))  # Gray fallback

            if not normalized:
                try:
                    return cm.get_cmap(base_cmap)
                except Exception:
                    return cm.get_cmap("tab10")

            return ListedColormap(normalized, name=f"{base_cmap}_drillholes")
        except Exception as e:
            logger.warning(f"Failed to build lithology colormap: {e}")
            try:
                return cm.get_cmap(base_cmap)
            except Exception:
                return cm.get_cmap("tab10")

    # ============================================================================
    # GLOBAL PICKING SYSTEM
    # ============================================================================
    
    def setup_global_picking(self):
        """
        Initialize global picking callbacks for all layers.
        
        PERFORMANCE: PyVista enable_point_picking/enable_cell_picking are DISABLED.
        Picking is handled by custom VTK picker in ViewerWidget._handle_block_click.
        This method only ensures mesh actors are set as pickable.
        """
        if not self.plotter:
            logger.warning("Cannot setup picking: plotter not initialized")
            return
        
        logger.info("Setting up global picking system (lightweight mode)...")
        
        try:
            # PERFORMANCE: Do NOT call enable_point_picking or enable_cell_picking
            # These cause double-picking, performance issues, and conflicts with camera controls.
            # Picking is handled by custom VTK picker in ViewerWidget._handle_block_click.
            
            # Just ensure mesh actor is pickable
            if hasattr(self, 'mesh_actor') and self.mesh_actor is not None:
                self.mesh_actor.SetPickable(1)
                logger.info(f"Mesh actor set pickable: {self.mesh_actor.GetPickable()}")
            
            logger.info("Global picking system ready (custom VTK picker mode)")
        except Exception as e:
            logger.error(f"Error setting up global picking: {e}", exc_info=True)
    
    def set_pick_callback(self, callback):
        """
        Set callback function for pick events.
        
        Args:
            callback: Function that receives a dictionary of pick information
        """
        self.pick_callback = callback
        logger.debug("Pick callback registered")
    
    def _on_pick_event(self, *args):
        """
        Handle global mouse pick events.
        
        This is called by PyVista when the user clicks on any mesh in the scene.
        PyVista stores the picked point in plotter.picked_point attribute.
        
        Args:
            *args: Variable arguments (PyVista may pass event object, but point is in plotter.picked_point)
        """
        logger.debug(f"_on_pick_event called! args={len(args)}, picking_enabled={self.picking_enabled}, pick_callback={self.pick_callback is not None}")
        
        if not self.picking_enabled:
            logger.debug("Picking is disabled!")
            return
        
        if not self.pick_callback:
            logger.debug("No pick callback registered!")
            return
        
        try:
            # PyVista stores the picked point in the plotter.picked_point attribute
            # This is more reliable than trying to extract from callback arguments
            picked_point = None
            if hasattr(self.plotter, 'picked_point') and self.plotter.picked_point is not None:
                picked_point = np.array(self.plotter.picked_point, dtype=float)
            elif args and len(args) > 0:
                # Fallback: try to extract from arguments if available
                if isinstance(args[0], (list, tuple, np.ndarray)) and len(args[0]) >= 3:
                    picked_point = np.array(args[0][:3], dtype=float)
            
            if picked_point is None or len(picked_point) < 3:
                logger.debug("Could not get valid picked point from plotter.picked_point or arguments")
                return
            
            # Get the picked mesh - try multiple methods
            picked = None
            
            # Method 1: Try plotter.picked_mesh (standard PyVista way)
            if hasattr(self.plotter, 'picked_mesh') and self.plotter.picked_mesh is not None:
                picked = self.plotter.picked_mesh
                logger.info(f"Got mesh from plotter.picked_mesh: {type(picked)}")
            
            # Method 2: Try to get from picked actor
            if picked is None:
                try:
                    # PyVista stores picked actor in plotter.picked_actor
                    if hasattr(self.plotter, 'picked_actor') and self.plotter.picked_actor is not None:
                        actor = self.plotter.picked_actor
                        mapper = actor.GetMapper()
                        if mapper is not None:
                            vtk_data = mapper.GetInput()
                            if vtk_data is not None:
                                import pyvista as pv
                                picked = pv.wrap(vtk_data)
                                logger.info(f"Got mesh from picked_actor: {type(picked)}")
                except Exception as e:
                    logger.debug(f"Could not get mesh from picked_actor: {e}")
            
            # Method 3: Try to get from renderer's mesh_actor
            if picked is None and hasattr(self, 'mesh_actor') and self.mesh_actor is not None:
                try:
                    # Check if mesh_actor matches any picked actor
                    if hasattr(self.plotter, 'picked_actor') and self.plotter.picked_actor == self.mesh_actor:
                        # Get mesh from block_meshes
                        picked = (
                            self.block_meshes.get('imagedata') or 
                            self.block_meshes.get('unstructured_grid') or 
                            self.block_meshes.get('point_cloud')
                        )
                        if picked is not None:
                            logger.info(f"Got mesh from block_meshes: {type(picked)}")
                except Exception as e:
                    logger.debug(f"Could not get mesh from mesh_actor: {e}")
            
            logger.info(f"Final picked mesh: {picked is not None}, type={type(picked) if picked is not None else 'None'}")
            if picked is None:
                logger.warning("No mesh was picked! Trying alternative picking method...")
                # Try using the picked point to find closest mesh manually
                if picked_point is not None:
                    # Check all active layers
                    for layer_name, layer in self.active_layers.items():
                        layer_data = layer.get('data')
                        if layer_data is not None:
                            try:
                                cell_id = layer_data.find_closest_cell(picked_point)
                                if cell_id >= 0:
                                    picked = layer_data
                                    logger.info(f"Found mesh via closest cell search: {layer_name}, cell_id={cell_id}")
                                    break
                            except Exception:
                                pass
                
                if picked is None:
                    logger.warning("Could not find any mesh for picked point")
                    return
            
            # Find the closest cell
            cell_id = picked.find_closest_cell(picked_point)
            logger.info(f"Found closest cell: cell_id={cell_id}, picked_point={picked_point}")
            if cell_id < 0:
                logger.warning(f"Invalid cell_id: {cell_id}")
                return
            
            # Determine which layer this mesh belongs to
            layer_name = None
            layer_type = 'unknown'
            
            # Check active_layers first
            for name, layer in self.active_layers.items():
                if layer['data'] is picked:
                    layer_name = name
                    layer_type = layer['type']
                    break
            
            # If not found, check scene_layers
            if layer_name is None:
                for name, layer in self.scene_layers.items():
                    if layer.get('data') is picked:
                        layer_name = name
                        layer_type = layer.get('type', 'unknown')
                        break
            
            if layer_name is None:
                layer_name = "Unknown Layer"
            
            # Build info dictionary
            info = {
                "Layer": layer_name,
                "Layer Type": layer_type,
                "Cell ID": cell_id,
                "Position": tuple(picked_point)
            }
            
            # Get all scalar properties at this cell
            if hasattr(picked, 'cell_data'):
                for key in picked.cell_data.keys():
                    arr = picked.cell_data[key]
                    if len(arr) > cell_id:
                        val = arr[cell_id]
                        # Preserve Original_ID as integer (not float)
                        if key == 'Original_ID':
                            info[key] = int(val)
                        else:
                            # Skip NaN values
                            if isinstance(val, (int, float, np.integer, np.floating)):
                                if not (isinstance(val, float) and np.isnan(val)):
                                    info[key] = float(val) if isinstance(val, (float, np.floating)) else int(val)
                            else:
                                info[key] = val
            
            # Also check point data (for some mesh types)
            if hasattr(picked, 'point_data'):
                # Find closest point
                point_id = picked.find_closest_point(picked_point)
                if point_id >= 0:
                    for key in picked.point_data.keys():
                        arr = picked.point_data[key]
                        if len(arr) > point_id:
                            # Only add if not already in cell_data
                            if key not in info:
                                info[f"Point_{key}"] = float(arr[point_id])
            
            # Call the registered callback with info
            self.pick_callback(info)
            
            logger.debug(f"Pick event: {layer_name}, cell {cell_id}")
            
        except Exception as e:
            logger.error(f"Error during picking: {e}", exc_info=True)
    
    def enable_picking(self):
        """Enable mouse picking."""
        self.picking_enabled = True
        logger.debug("Picking enabled")
    
    def disable_picking(self):
        """Disable mouse picking."""
        self.picking_enabled = False
        logger.debug("Picking disabled")
    
    def register_scene_layer(self, layer_name: str, actor, data, layer_type: str, properties: List[str] = None):
        """
        Register a layer in the scene registry for picking and inspection.
        
        Args:
            layer_name: Unique name for the layer
            actor: PyVista actor reference
            data: PyVista mesh/grid data
            layer_type: Type of layer ('block', 'drillhole', 'surface', 'volume', etc.)
            properties: List of property names available in this layer
        """
        if properties is None:
            properties = []
            # Auto-detect properties from data
            if hasattr(data, 'array_names'):
                properties = list(data.array_names)
            elif hasattr(data, 'cell_data'):
                properties = list(data.cell_data.keys())
            elif hasattr(data, 'point_data'):
                properties = list(data.point_data.keys())
        
        self.scene_layers[layer_name] = {
            'actor': actor,
            'data': data,
            'type': layer_type,
            'properties': properties
        }
        
        logger.debug(f"Registered scene layer: {layer_name} ({layer_type}), {len(properties)} properties")
    
    def unregister_scene_layer(self, layer_name: str):
        """
        Unregister a layer from the scene registry.
        
        Args:
            layer_name: Name of the layer to unregister
        """
        if layer_name in self.scene_layers:
            del self.scene_layers[layer_name]
            logger.debug(f"Unregistered scene layer: {layer_name}")
    
    def render_pushbacks(
        self,
        plan: Any,
        block_model: Optional[BlockModel] = None,
        style_config: Optional[Dict[str, Any]] = None,
        layer_name: str = "Pushbacks"
    ) -> Any:
        """
        Render pushbacks by coloring blocks by pushback ID (STEP 33).
        
        Args:
            plan: PushbackPlan instance
            block_model: BlockModel instance (uses current_model if None)
            style_config: Optional style configuration
            layer_name: Layer name
            
        Returns:
            SceneLayer
        """
        from .scene_layer import SceneLayer, LAYER_TYPE_PUSHBACK_PHASES
        from ..mine_planning.pushbacks.pushback_model import PushbackPlan
        
        try:
            if self.plotter is None:
                raise RuntimeError("Plotter not initialized")
            
            # Use provided block model or current model
            bm = block_model or self.current_model
            if bm is None:
                raise ValueError("No block model available")
            
            style_config = style_config or {}
            opacity = style_config.get("opacity", 0.8)
            
            # Create pushback mapping: shell_id -> pushback_id
            shell_to_pushback = {}
            for pushback in plan.pushbacks:
                for shell_id in pushback.shell_ids:
                    shell_to_pushback[shell_id] = pushback.id
            
            # Get block model as DataFrame
            df = bm.to_dataframe()
            
            # Create pushback ID array for blocks
            # Map phase/pit/shell properties to pushback IDs
            pushback_ids = []
            for idx in range(len(df)):
                # Try to match block phase/pit to pushback
                block_phase = None
                if "phase" in df.columns:
                    block_phase = df["phase"].iloc[idx]
                elif "PHASE" in df.columns:
                    block_phase = df["PHASE"].iloc[idx]
                elif "pit" in df.columns:
                    block_phase = df["pit"].iloc[idx]
                elif "PIT" in df.columns:
                    block_phase = df["PIT"].iloc[idx]
                
                pushback_id = None
                if block_phase and pd.notna(block_phase):
                    # Try direct match
                    if str(block_phase) in shell_to_pushback:
                        pushback_id = shell_to_pushback[str(block_phase)]
                    else:
                        # Try partial match (e.g., "S_30" matches "S_30")
                        for shell_id, pb_id in shell_to_pushback.items():
                            if str(block_phase) == shell_id or str(block_phase).startswith(shell_id):
                                pushback_id = pb_id
                                break
                
                pushback_ids.append(pushback_id if pushback_id else "unassigned")
            
            # Create color mapping
            pushback_colors = {}
            for pushback in plan.pushbacks:
                pushback_colors[pushback.id] = pushback.color
            
            # Create numeric array for coloring (map pushback IDs to indices)
            unique_pushback_ids = ["unassigned"] + [pb.id for pb in plan.pushbacks]
            pushback_to_index = {pb_id: idx for idx, pb_id in enumerate(unique_pushback_ids)}
            color_indices = np.array([pushback_to_index.get(pb_id, 0) for pb_id in pushback_ids])
            
            # Generate mesh with pushback colors
            if bm.positions is None or bm.dimensions is None:
                raise ValueError("Block model geometry not available")
            
            # Create mesh with pushback colors
            positions = bm.positions
            dimensions = bm.dimensions
            
            # Create mesh using helper method
            mesh = self._generate_optimized_meshes_for_pushbacks(positions, dimensions, pushback_ids, pushback_colors)
            
            # Add mesh to plotter
            actor = self.plotter.add_mesh(
                mesh,
                scalars="pushback_id",
                cmap="Set3",  # Discrete colormap
                show_edges=True,
                edge_color=self.edge_color,
                opacity=opacity,
                show_scalar_bar=False,  # DISABLED: Use custom LegendManager instead
                name=layer_name
            )
            
            # Create legend entries for pushbacks
            if self.legend_manager:
                legend_entries = []
                for pushback in plan.pushbacks:
                    legend_entries.append({
                        "label": pushback.name,
                        "color": pushback.color,
                        "value": pushback.id
                    })
                self.legend_manager.update_legend(legend_entries, property_name="pushback")
            
            logger.info(f"Rendered {len(plan.pushbacks)} pushbacks")
            
            layer = self.add_layer(
                layer_name,
                actor,
                mesh,
                layer_type=LAYER_TYPE_PUSHBACK_PHASES,
                opacity=opacity,
                properties=["pushback_id"]
            )
            
            return layer
            
        except Exception as e:
            logger.error(f"Failed to render pushbacks: {e}", exc_info=True)
            raise
    
    def render_stopes(
        self,
        stopes: List[Any],
        style_config: Optional[Dict[str, Any]] = None,
        layer_name: str = "SLOS Stopes"
    ) -> Any:
        """
        Render SLOS stopes (STEP 37).
        
        Args:
            stopes: List of StopeInstance objects
            style_config: Optional style configuration
            layer_name: Layer name
            
        Returns:
            SceneLayer
        """
        from .scene_layer import SceneLayer, LAYER_TYPE_UG_STOPE_LAYOUT
        
        try:
            if self.plotter is None:
                raise RuntimeError("Plotter not initialized")
            
            style_config = style_config or {}
            opacity = style_config.get("opacity", 0.7)
            color = style_config.get("color", "cyan")
            
            # Create boxes for each stope
            actors = []
            for stope in stopes:
                # Create box from stope center and template dimensions
                # Simplified: create a box at stope center
                # In production, would use actual template dimensions
                if isinstance(stope, dict):
                    center = stope.get('center', (0, 0, 0))
                else:
                    center = getattr(stope, 'center', (0, 0, 0))
                
                size = (50.0, 50.0, 30.0)  # Placeholder dimensions
                
                box = pv.Box(bounds=(
                    center[0] - size[0]/2, center[0] + size[0]/2,
                    center[1] - size[1]/2, center[1] + size[1]/2,
                    center[2] - size[2]/2, center[2] + size[2]/2
                ))
                
                stope_id = stope.get('id', '') if isinstance(stope, dict) else getattr(stope, 'id', '')
                actor = self.plotter.add_mesh(box, color=color, opacity=opacity, name=f"stope_{stope_id}")
                actors.append(actor)
            
            layer = SceneLayer(
                name=layer_name,
                layer_type=LAYER_TYPE_UG_STOPE_LAYOUT,
                actor=actors[0] if actors else None,
                data={"stopes": stopes},
                visible=True,
                opacity=opacity
            )
            
            # Store in scene layers
            self.scene_layers[layer_name] = layer
            
            logger.info(f"Rendered {len(stopes)} SLOS stopes")
            return layer
        
        except Exception as e:
            logger.error(f"Error rendering stopes: {e}", exc_info=True)
            return SceneLayer(name=layer_name, layer_type=LAYER_TYPE_UG_STOPE_LAYOUT, visible=False)
    
    def render_cave_footprint(
        self,
        footprint: Any,
        style_config: Optional[Dict[str, Any]] = None,
        layer_name: str = "Cave Footprint"
    ) -> Any:
        """
        Render cave footprint (STEP 37).
        
        Args:
            footprint: CaveFootprint instance
            style_config: Optional style configuration
            layer_name: Layer name
            
        Returns:
            SceneLayer
        """
        from .scene_layer import SceneLayer, LAYER_TYPE_UG_CAVE_FOOTPRINT
        
        try:
            if self.plotter is None:
                raise RuntimeError("Plotter not initialized")
            
            style_config = style_config or {}
            opacity = style_config.get("opacity", 0.6)
            color = style_config.get("color", "red")
            
            # Get cells
            if isinstance(footprint, dict):
                cells = footprint.get('cells', [])
            elif hasattr(footprint, 'cells'):
                cells = footprint.cells
            else:
                cells = []
            
            if not cells:
                return SceneLayer(name=layer_name, layer_type=LAYER_TYPE_UG_CAVE_FOOTPRINT, visible=False)
            
            # Create points for each cell
            points = []
            for cell in cells:
                if isinstance(cell, dict):
                    points.append([cell.get('x', 0), cell.get('y', 0), cell.get('level', 0)])
                else:
                    points.append([getattr(cell, 'x', 0), getattr(cell, 'y', 0), getattr(cell, 'level', 0)])
            
            point_cloud = pv.PolyData(np.array(points))
            
            # Add cell IDs as scalars
            cell_ids = np.arange(len(cells))
            point_cloud["cell_id"] = cell_ids
            
            actor = self.plotter.add_mesh(
                point_cloud,
                color=color,
                opacity=opacity,
                point_size=10,
                name=layer_name
            )
            
            layer = SceneLayer(
                name=layer_name,
                layer_type=LAYER_TYPE_UG_CAVE_FOOTPRINT,
                actor=actor,
                data={"footprint": footprint},
                visible=True,
                opacity=opacity
            )
            
            # Store in scene layers
            self.scene_layers[layer_name] = layer
            
            logger.info(f"Rendered cave footprint with {len(cells)} cells")
            return layer
        
        except Exception as e:
            logger.error(f"Error rendering cave footprint: {e}", exc_info=True)
            return SceneLayer(name=layer_name, layer_type=LAYER_TYPE_UG_CAVE_FOOTPRINT, visible=False)
    
    def render_void_state(
        self,
        void_series: Dict[str, List[float]],
        period_id: str,
        style_config: Optional[Dict[str, Any]] = None,
        layer_name: str = "Void State"
    ) -> Any:
        """
        Render void state for a specific period (STEP 37).
        
        Args:
            void_series: Dictionary with void volume time series
            period_id: Period identifier
            style_config: Optional style configuration
            layer_name: Layer name
            
        Returns:
            SceneLayer
        """
        from .scene_layer import SceneLayer, LAYER_TYPE_UG_VOID_STATE
        
        try:
            if self.plotter is None:
                raise RuntimeError("Plotter not initialized")
            
            style_config = style_config or {}
            opacity = style_config.get("opacity", 0.5)
            
            # This would visualize void volumes in 3D
            # For now, return a placeholder layer
            layer = SceneLayer(
                name=layer_name,
                layer_type=LAYER_TYPE_UG_VOID_STATE,
                data={"void_series": void_series, "period_id": period_id},
                visible=True,
                opacity=opacity
            )
            
            logger.info(f"Rendered void state for period {period_id}")
            return layer
        
        except Exception as e:
            logger.error(f"Error rendering void state: {e}", exc_info=True)
            return SceneLayer(name=layer_name, layer_type=LAYER_TYPE_UG_VOID_STATE, visible=False)
    
    def _generate_optimized_meshes_for_pushbacks(
        self,
        positions: np.ndarray,
        dimensions: np.ndarray,
        pushback_ids: List[str],
        pushback_colors: Dict[str, Tuple[float, float, float]]
    ) -> pv.UnstructuredGrid:
        """Generate optimized mesh for pushback visualization."""
        import pyvista as pv
        
        n_blocks = len(positions)
        cells = []
        cell_types = []
        points = []
        pushback_id_array = []
        color_array = []
        
        point_offset = 0
        
        for i in range(n_blocks):
            x, y, z = positions[i]
            dx, dy, dz = dimensions[i]
            
            # Create 8 vertices for hexahedron
            block_points = np.array([
                [x - dx/2, y - dy/2, z - dz/2],  # 0
                [x + dx/2, y - dy/2, z - dz/2],  # 1
                [x + dx/2, y + dy/2, z - dz/2],  # 2
                [x - dx/2, y + dy/2, z - dz/2],  # 3
                [x - dx/2, y - dy/2, z + dz/2],  # 4
                [x + dx/2, y - dy/2, z + dz/2],  # 5
                [x + dx/2, y + dy/2, z + dz/2],  # 6
                [x - dx/2, y + dy/2, z + dz/2],  # 7
            ])
            
            points.append(block_points)
            
            # Create hexahedron cell (VTK_HEXAHEDRON = 12)
            cell = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7]) + point_offset
            cells.extend(cell)
            cell_types.append(12)
            
            # Add pushback ID and color
            pb_id = pushback_ids[i] if i < len(pushback_ids) else "unassigned"
            pushback_id_array.append(pb_id)
            
            color = pushback_colors.get(pb_id, (0.5, 0.5, 0.5))
            color_array.append(color)
            
            point_offset += 8
        
        # Combine all points
        all_points = np.vstack(points)
        
        # Create unstructured grid
        mesh = pv.UnstructuredGrid(cells, cell_types, all_points)
        
        # Add pushback data
        mesh.cell_data["pushback_id"] = pushback_id_array
        mesh.cell_data["pushback_color"] = np.array(color_array)
        
        return mesh
    def _compute_grid_signature(self, grid_like: Any) -> Optional[Tuple[int, int]]:
        """Return a lightweight signature describing the current grid structure."""
        if grid_like is None:
            return None

        try:
            if hasattr(grid_like, 'n_cells') and hasattr(grid_like, 'n_points'):
                return (int(grid_like.n_cells), int(grid_like.n_points))

            get_cells = getattr(grid_like, 'GetNumberOfCells', None)
            get_points = getattr(grid_like, 'GetNumberOfPoints', None)
            if callable(get_cells) and callable(get_points):
                return (int(get_cells()), int(get_points()))
        except Exception as e:
            logger.debug(f"Could not compute grid signature: {e}")

        return None
    
    # ============================================================================
    # MEASUREMENT TOOLS - VISIBLE LINES AND FEEDBACK
    # ============================================================================
    
    def start_distance_measurement(self):
        """Start distance measurement mode (two clicks with visible line)."""
        if self.plotter is None:
            logger.warning("Cannot start measurement: plotter not initialized")
            return
        
        self.measure_mode = 'distance'
        self.measure_points = []
        self._clear_temporary_measurements()
        
        # Set up click callback
        self._setup_measurement_callbacks()
        logger.info("Distance measurement mode started - click two points")
    
    def start_area_measurement(self):
        """Start area measurement mode (multiple clicks with visible polygon)."""
        if self.plotter is None:
            logger.warning("Cannot start measurement: plotter not initialized")
            return
        
        self.measure_mode = 'area'
        self.measure_points = []
        self._clear_temporary_measurements()
        
        # Set up click callback
        self._setup_measurement_callbacks()
        logger.info("Area measurement mode started - click to add vertices")
    
    def finish_area_measurement(self):
        """Finish area measurement and compute area."""
        if self.measure_mode != 'area' or len(self.measure_points) < 3:
            logger.warning("Cannot finish area measurement: need at least 3 points")
            return
        
        try:
            import numpy as np
            points = np.array(self.measure_points)
            
            # Compute area from polygon
            area = self._compute_polygon_area(points)
            
            # Create permanent polygon with bright yellow fill
            polygon = self._create_polygon_mesh(points, color='yellow', opacity=0.4)
            actor = self.plotter.add_mesh(polygon, name=f"measurement_area_{len(self.measurements)}")
            self.measure_actors.append(actor)
            
            # Draw permanent lines
            for i in range(len(self.measure_points)):
                p1 = self.measure_points[i]
                p2 = self.measure_points[(i + 1) % len(self.measure_points)]
                line = pv.Line(p1, p2)
                line_actor = self.plotter.add_mesh(line, color='yellow', line_width=5, name=f'measure_line_{len(self.measurements)}_{i}')
                self.measure_actors.append(line_actor)
            
            # Add label with area
            center = points.mean(axis=0)
            label_text = f"Area: {area:.2f} m²"
            self._add_measurement_label(center, label_text)
            
            # Store measurement
            self.measurements.append({
                'type': 'area',
                'points': points.tolist(),
                'area': area,
                'actors': [actor] + [a for a in self.measure_actors[-len(self.measure_points):]]
            })
            
            # Clear temporary
            self._clear_temporary_measurements()
            self.measure_mode = None
            self.measure_points = []
            
            if self.plotter:
                self.plotter.render()
            
            logger.info(f"Area measurement completed: {area:.2f} m²")
            
        except Exception as e:
            logger.error(f"Error finishing area measurement: {e}", exc_info=True)
    
    def cancel_area_measurement(self):
        """Cancel current area measurement."""
        self._clear_temporary_measurements()
        # Remove temporary point markers
        for actor in list(self.measure_actors):
            if 'temp' in str(actor) or 'measure_point' in str(actor):
                try:
                    self.plotter.remove_actor(actor)
                    self.measure_actors.remove(actor)
                except Exception:
                    pass
        self.measure_mode = None
        self.measure_points = []
        if self.plotter:
            self.plotter.render()
        logger.info("Area measurement cancelled")
    
    def clear_measurements(self):
        """Clear all measurements."""
        if self.plotter is None:
            return
        
        # Remove all measurement actors
        for actor in self.measure_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        
        self.measure_actors.clear()
        self.measurements.clear()
        self._clear_temporary_measurements()
        self.measure_mode = None
        self.measure_points = []
        
        if self.plotter:
            self.plotter.render()
        
        logger.info("All measurements cleared")
    
    def export_measurements(self, filename: str, fmt: str = 'csv') -> int:
        """Export measurements to file."""
        if not self.measurements:
            return 0
        
        try:
            import pandas as pd
            
            data = []
            for i, meas in enumerate(self.measurements):
                if meas['type'] == 'distance':
                    data.append({
                        'ID': i + 1,
                        'Type': 'Distance',
                        'Point1': str(meas['points'][0]),
                        'Point2': str(meas['points'][1]),
                        'Distance (m)': meas['distance']
                    })
                elif meas['type'] == 'area':
                    data.append({
                        'ID': i + 1,
                        'Type': 'Area',
                        'Points': len(meas['points']),
                        'Area (m²)': meas['area']
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logger.info(f"Exported {len(self.measurements)} measurements to {filename}")
            return len(self.measurements)
            
        except Exception as e:
            logger.error(f"Error exporting measurements: {e}", exc_info=True)
            return 0
    
    def _setup_measurement_callbacks(self):
        """Set up mouse callbacks for measurement with visible feedback."""
        if self.plotter is None:
            return
        
        # Remove existing callback if any
        if self._measure_callback_id is not None:
            try:
                self.plotter.iren.remove_observer(self._measure_callback_id)
            except Exception:
                pass
        
        # Add click callback using PyVista's picker
        def on_click():
            try:
                if self.measure_mode is None:
                    return
                
                # Get click position using PyVista picker
                click_pos = self.plotter.pick_mouse_position()
                if click_pos is None:
                    return
                
                x, y, z = click_pos
                point = [x, y, z]
                
                if self.measure_mode == 'distance':
                    self.measure_points.append(point)
                    
                    # Add visible marker (bright red sphere)
                    marker = pv.Sphere(radius=1.0, center=point)
                    actor = self.plotter.add_mesh(marker, color='red', name=f'measure_point_{len(self.measure_points)}')
                    self.measure_actors.append(actor)
                    
                    if len(self.measure_points) == 2:
                        # Second point - draw bright yellow line
                        p1, p2 = self.measure_points[0], self.measure_points[1]
                        import numpy as np
                        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
                        
                        # Draw thick visible line
                        line = pv.Line(p1, p2)
                        actor = self.plotter.add_mesh(line, color='yellow', line_width=8, name='measure_line')
                        self.measure_actors.append(actor)
                        
                        # Add label
                        mid_point = [(a + b) / 2 for a, b in zip(p1, p2)]
                        label_text = f"{distance:.2f} m"
                        self._add_measurement_label(mid_point, label_text)
                        
                        # Store measurement
                        self.measurements.append({
                            'type': 'distance',
                            'points': [p1, p2],
                            'distance': distance
                        })
                        
                        # Reset for next measurement
                        self.measure_mode = None
                        self.measure_points = []
                        
                        if self.plotter:
                            self.plotter.render()
                        
                        logger.info(f"Distance measured: {distance:.2f} m")
                
                elif self.measure_mode == 'area':
                    self.measure_points.append(point)
                    
                    # Add visible cyan marker
                    marker = pv.Sphere(radius=1.0, center=point)
                    actor = self.plotter.add_mesh(marker, color='cyan', name=f'measure_point_{len(self.measure_points)}')
                    self.measure_actors.append(actor)
                    
                    # Update temporary polygon visualization
                    if len(self.measure_points) >= 2:
                        self._update_temporary_polygon()
                    
                    if self.plotter:
                        self.plotter.render()
                    
            except Exception as e:
                logger.error(f"Error in measurement callback: {e}", exc_info=True)
        
        # Connect to left click event using VTK observer
        try:
            iren = getattr(self.plotter, 'iren', None) or getattr(self.plotter, 'interactor', None)
            if iren:
                def vtk_callback(obj, event):
                    """VTK callback wrapper."""
                    try:
                        # Get pick position
                        picker = vtk.vtkWorldPointPicker()
                        pos = iren.GetEventPosition()
                        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
                        world_pos = picker.GetPickPosition()
                        
                        if world_pos[0] != 0 or world_pos[1] != 0 or world_pos[2] != 0:
                            point = list(world_pos)
                            
                            if self.measure_mode == 'distance':
                                self.measure_points.append(point)
                                
                                # Add visible marker (bright red sphere)
                                marker = pv.Sphere(radius=1.0, center=point)
                                actor = self.plotter.add_mesh(marker, color='red', name=f'measure_point_{len(self.measure_points)}')
                                self.measure_actors.append(actor)
                                
                                if len(self.measure_points) == 2:
                                    # Second point - draw bright yellow line
                                    p1, p2 = self.measure_points[0], self.measure_points[1]
                                    import numpy as np
                                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
                                    
                                    # Draw thick visible line
                                    line = pv.Line(p1, p2)
                                    actor = self.plotter.add_mesh(line, color='yellow', line_width=8, name='measure_line')
                                    self.measure_actors.append(actor)
                                    
                                    # Add label
                                    mid_point = [(a + b) / 2 for a, b in zip(p1, p2)]
                                    label_text = f"{distance:.2f} m"
                                    self._add_measurement_label(mid_point, label_text)
                                    
                                    # Store measurement
                                    self.measurements.append({
                                        'type': 'distance',
                                        'points': [p1, p2],
                                        'distance': distance
                                    })
                                    
                                    # Reset for next measurement
                                    self.measure_mode = None
                                    self.measure_points = []
                                    
                                    if self.plotter:
                                        self.plotter.render()
                                    
                                    logger.info(f"Distance measured: {distance:.2f} m")
                            
                            elif self.measure_mode == 'area':
                                self.measure_points.append(point)
                                
                                # Add visible cyan marker
                                marker = pv.Sphere(radius=1.0, center=point)
                                actor = self.plotter.add_mesh(marker, color='cyan', name=f'measure_point_{len(self.measure_points)}')
                                self.measure_actors.append(actor)
                                
                                # Update temporary polygon visualization
                                if len(self.measure_points) >= 2:
                                    self._update_temporary_polygon()
                                
                                if self.plotter:
                                    self.plotter.render()
                    except Exception as e:
                        logger.error(f"Error in measurement callback: {e}", exc_info=True)
                
                self._measure_callback_id = iren.AddObserver('LeftButtonPressEvent', vtk_callback)
                logger.debug("Measurement callback registered")
            else:
                logger.warning("Could not get interactor for measurement callback")
        except Exception as e:
            logger.warning(f"Could not set up measurement callback: {e}")
    
    def _update_temporary_polygon(self):
        """Update temporary polygon visualization with visible lines."""
        if len(self.measure_points) < 2:
            return
        
        # Remove old temporary polygon and lines
        temp_actors = [a for a in self.measure_actors if 'temp' in str(a)]
        for actor in temp_actors:
            try:
                self.plotter.remove_actor(actor)
                self.measure_actors.remove(actor)
            except Exception:
                pass
        
        import numpy as np
        points = np.array(self.measure_points)
        
        # Draw visible lines between points
        for i in range(len(self.measure_points)):
            p1 = self.measure_points[i]
            p2 = self.measure_points[(i + 1) % len(self.measure_points)]
            line = pv.Line(p1, p2)
            line_actor = self.plotter.add_mesh(line, color='cyan', line_width=5, name=f'temp_line_{i}')
            self.measure_actors.append(line_actor)
        
        # Create filled polygon if 3+ points
        if len(points) >= 3:
            polygon = self._create_polygon_mesh(points, color='cyan', opacity=0.2)
            if polygon:
                self._measure_polygon_actor = self.plotter.add_mesh(polygon, name='temp_measure_polygon')
                self.measure_actors.append(self._measure_polygon_actor)
    
    def _clear_temporary_measurements(self):
        """Clear temporary measurement visuals."""
        temp_actors = [a for a in self.measure_actors if 'temp' in str(a)]
        for actor in temp_actors:
            try:
                self.plotter.remove_actor(actor)
                self.measure_actors.remove(actor)
            except Exception:
                pass
        
        if self._measure_line_actor is not None:
            try:
                self.plotter.remove_actor(self._measure_line_actor)
            except Exception:
                pass
            self._measure_line_actor = None
        
        if self._measure_polygon_actor is not None:
            try:
                self.plotter.remove_actor(self._measure_polygon_actor)
            except Exception:
                pass
            self._measure_polygon_actor = None
    
    def _create_polygon_mesh(self, points, color='yellow', opacity=0.3):
        """Create a filled polygon mesh from points."""
        import numpy as np
        import pyvista as pv
        
        if len(points) < 3:
            return None
        
        # Close the polygon
        closed_points = np.vstack([points, points[0]])
        
        # Create polyline
        polyline = pv.PolyData(closed_points)
        polyline.lines = np.hstack([[len(closed_points)], np.arange(len(closed_points))])
        
        # Try to create filled polygon
        try:
            # Project to 2D (use XY plane)
            z_mean = points[:, 2].mean()
            points_2d = points[:, :2]
            polygon_2d = pv.PolyData(points_2d)
            polygon_2d = polygon_2d.delaunay_2d()
            
            # Extrude back to 3D
            if polygon_2d.n_cells > 0:
                # Add Z coordinate back
                points_3d = np.column_stack([polygon_2d.points, np.full(polygon_2d.n_points, z_mean)])
                polygon = pv.PolyData(points_3d, polygon_2d.cells)
                return polygon
        except Exception:
            pass
        
        # Fallback: just return polyline
        return polyline
    
    def _compute_polygon_area(self, points):
        """Compute polygon area using shoelace formula."""
        import numpy as np
        
        if len(points) < 3:
            return 0.0
        
        # Use XY plane for area calculation
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        return area
    
    def _add_measurement_label(self, position, text):
        """Add a visible text label at position."""
        try:
            # Use PyVista's text actor with high visibility
            actor = self.plotter.add_text(
                text,
                position=position,
                font_size=16,
                color='white',
                shadow=True,
                name=f'measure_label_{len(self.measure_actors)}'
            )
            self.measure_actors.append(actor)
        except Exception:
            # Fallback: just log
            logger.debug(f"Measurement label: {text} at {position}")

    # =====================================================================
    # Interactive Slicer Methods
    # =====================================================================

    def enable_interactive_slicer(self, widget_type: str = "plane", callback=None):
        """
        Enable interactive slicing with a draggable widget.

        Args:
            widget_type: Type of widget - "plane", "box", "sphere", or "cylinder"
            callback: Optional callback function called when widget is moved

        The widget can be dragged, rotated, and resized interactively.
        The model is clipped in real-time as the widget moves.
        """
        if self.plotter is None:
            raise RuntimeError("No plotter available. Load a model first.")

        # Store callback
        self._slicer_callback = callback
        self._slicer_widget_type = widget_type
        self._slicer_clip_mode = "normal"
        self._slicer_widget_opacity = 0.3

        # Get the mesh to clip (use the first active layer mesh)
        target_mesh = self._get_target_mesh_for_slicing()
        if target_mesh is None:
            raise RuntimeError("No mesh available to slice. Load a block model or drillholes first.")

        # Store original mesh
        self._original_mesh_for_slicing = target_mesh
        self._slicer_actor_name = None

        # Get mesh bounds for widget sizing
        bounds = target_mesh.bounds
        center = target_mesh.center

        # Enable the appropriate widget
        if widget_type == "plane":
            self._enable_plane_widget(center, bounds)
        elif widget_type == "box":
            self._enable_box_widget(bounds)
        elif widget_type == "sphere":
            self._enable_sphere_widget(center, bounds)
        elif widget_type == "cylinder":
            self._enable_cylinder_widget(center, bounds)
        else:
            raise ValueError(f"Unknown widget type: {widget_type}")

        logger.info(f"Interactive {widget_type} slicer enabled")

    def _get_target_mesh_for_slicing(self):
        """Get the target mesh for slicing (block model or drillholes)."""
        # Try to get block model mesh first
        if "blocks" in self.active_layers and "mesh" in self.active_layers["blocks"]:
            return self.active_layers["blocks"]["mesh"]

        # Try drillholes
        if "drillholes" in self.active_layers and "mesh" in self.active_layers["drillholes"]:
            return self.active_layers["drillholes"]["mesh"]

        # Try geology surfaces
        if "geology" in self.active_layers and "mesh" in self.active_layers["geology"]:
            return self.active_layers["geology"]["mesh"]

        # Fallback: search all actors for a large mesh
        for name, actor in self.plotter.actors.items():
            if hasattr(actor, 'mapper') and hasattr(actor.mapper, 'dataset'):
                mesh = actor.mapper.dataset
                if hasattr(mesh, 'n_cells') and mesh.n_cells > 0:
                    return mesh

        return None

    def _enable_plane_widget(self, center, bounds):
        """Enable interactive plane widget."""
        # Calculate plane size (diagonal of bounding box)
        import numpy as np
        diagonal = np.linalg.norm([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ])

        # Create plane widget
        def plane_callback(normal, origin):
            """Called when plane is moved."""
            self._apply_plane_clipping(normal, origin)
            if self._slicer_callback:
                self._slicer_callback({
                    'type': 'plane',
                    'normal': normal,
                    'origin': origin
                })

        self.plotter.add_plane_widget(
            callback=plane_callback,
            bounds=bounds,
            factor=1.2,
            normal='z',  # Initial normal
            origin=center,
            color='cyan',
            opacity=self._slicer_widget_opacity,
            implicit=True  # Clip the mesh
        )

        self._slicer_widget = 'plane'

    def _apply_plane_clipping(self, normal, origin):
        """Apply plane clipping to the mesh."""
        if self._original_mesh_for_slicing is None:
            return

        # Clip the mesh
        if self._slicer_clip_mode == "inverse":
            clipped = self._original_mesh_for_slicing.clip(normal=[-n for n in normal], origin=origin)
        else:
            clipped = self._original_mesh_for_slicing.clip(normal=normal, origin=origin)

        # Update the visualization
        # Remove old clipped actor
        if self._slicer_actor_name:
            try:
                self.plotter.remove_actor(self._slicer_actor_name, render=False)
            except:
                pass

        # Add clipped mesh
        self._slicer_actor_name = f"slicer_clipped_{id(clipped)}"

        # Get current active property for coloring
        active_scalars = None
        if hasattr(self._original_mesh_for_slicing, 'active_scalars_name'):
            active_scalars = self._original_mesh_for_slicing.active_scalars_name

        self.plotter.add_mesh(
            clipped,
            scalars=active_scalars,
            name=self._slicer_actor_name,
            show_edges=True,
            edge_color='white',
            opacity=1.0
        )

        self.plotter.render()

    def _enable_box_widget(self, bounds):
        """Enable interactive box widget."""
        def box_callback(box_bounds):
            """Called when box is moved/resized."""
            self._apply_box_clipping(box_bounds)
            if self._slicer_callback:
                self._slicer_callback({
                    'type': 'box',
                    'bounds': box_bounds
                })

        self.plotter.add_box_widget(
            callback=box_callback,
            bounds=bounds,
            factor=1.0,
            rotation_enabled=True,
            color='yellow',
            opacity=self._slicer_widget_opacity
        )

        self._slicer_widget = 'box'

    def _apply_box_clipping(self, box_bounds):
        """Apply box clipping to the mesh."""
        if self._original_mesh_for_slicing is None:
            return

        # Create box for clipping
        import pyvista as pv
        box = pv.Box(bounds=box_bounds)

        # Clip mesh
        if self._slicer_clip_mode == "inside" or self._slicer_clip_mode == "normal":
            clipped = self._original_mesh_for_slicing.clip_box(box_bounds, invert=False)
        else:  # outside or inverse
            clipped = self._original_mesh_for_slicing.clip_box(box_bounds, invert=True)

        # Update visualization
        if self._slicer_actor_name:
            try:
                self.plotter.remove_actor(self._slicer_actor_name, render=False)
            except:
                pass

        self._slicer_actor_name = f"slicer_clipped_{id(clipped)}"

        active_scalars = None
        if hasattr(self._original_mesh_for_slicing, 'active_scalars_name'):
            active_scalars = self._original_mesh_for_slicing.active_scalars_name

        self.plotter.add_mesh(
            clipped,
            scalars=active_scalars,
            name=self._slicer_actor_name,
            show_edges=True,
            edge_color='white',
            opacity=1.0
        )

        self.plotter.render()

    def _enable_sphere_widget(self, center, bounds):
        """Enable interactive sphere widget."""
        import numpy as np

        # Calculate initial radius (1/4 of diagonal)
        diagonal = np.linalg.norm([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ])
        radius = diagonal / 4.0

        def sphere_callback(center, radius):
            """Called when sphere is moved/resized."""
            self._apply_sphere_clipping(center, radius)
            if self._slicer_callback:
                self._slicer_callback({
                    'type': 'sphere',
                    'center': center,
                    'radius': radius
                })

        self.plotter.add_sphere_widget(
            callback=sphere_callback,
            center=center,
            radius=radius,
            color='magenta',
            opacity=self._slicer_widget_opacity
        )

        self._slicer_widget = 'sphere'

    def _apply_sphere_clipping(self, center, radius):
        """Apply sphere clipping to the mesh."""
        if self._original_mesh_for_slicing is None:
            return

        # Create implicit sphere function
        import vtk
        sphere = vtk.vtkSphere()
        sphere.SetCenter(center)
        sphere.SetRadius(radius)

        # Clip mesh
        if self._slicer_clip_mode == "inside" or self._slicer_clip_mode == "normal":
            clipped = self._original_mesh_for_slicing.clip_scalar(
                scalars='implicit_distance',
                value=0.0,
                invert=False,
                implicit_function=sphere
            )
        else:  # outside or inverse
            clipped = self._original_mesh_for_slicing.clip_scalar(
                scalars='implicit_distance',
                value=0.0,
                invert=True,
                implicit_function=sphere
            )

        # Update visualization
        if self._slicer_actor_name:
            try:
                self.plotter.remove_actor(self._slicer_actor_name, render=False)
            except:
                pass

        self._slicer_actor_name = f"slicer_clipped_{id(clipped)}"

        active_scalars = None
        if hasattr(self._original_mesh_for_slicing, 'active_scalars_name'):
            active_scalars = self._original_mesh_for_slicing.active_scalars_name

        self.plotter.add_mesh(
            clipped,
            scalars=active_scalars,
            name=self._slicer_actor_name,
            show_edges=True,
            edge_color='white',
            opacity=1.0
        )

        self.plotter.render()

    def _enable_cylinder_widget(self, center, bounds):
        """Enable interactive cylinder widget."""
        import numpy as np

        # Calculate initial radius and height
        diagonal = np.linalg.norm([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2]
        ])
        radius = diagonal / 4.0
        height = bounds[5] - bounds[4]

        # Note: PyVista doesn't have a built-in cylinder widget,
        # so we'll use a line widget to define the cylinder axis
        # This is a simplified implementation
        logger.warning("Cylinder widget not fully implemented - using plane widget instead")
        self._enable_plane_widget(center, bounds)

    def disable_interactive_slicer(self):
        """Disable the interactive slicer widget and restore original mesh."""
        if self.plotter is None:
            return

        # Clear all widgets
        self.plotter.clear_plane_widgets()
        self.plotter.clear_box_widgets()
        self.plotter.clear_sphere_widgets()

        # Remove clipped actor and restore original
        if hasattr(self, '_slicer_actor_name') and self._slicer_actor_name:
            try:
                self.plotter.remove_actor(self._slicer_actor_name, render=False)
            except:
                pass

        # Re-render original mesh if we have it
        if hasattr(self, '_original_mesh_for_slicing') and self._original_mesh_for_slicing:
            # The original mesh should still be in active_layers
            pass

        # Clean up
        self._slicer_widget = None
        self._slicer_callback = None
        self._original_mesh_for_slicing = None
        self._slicer_actor_name = None

        self.plotter.render()
        logger.info("Interactive slicer disabled")

    def set_slicer_clip_mode(self, mode: str):
        """
        Set the clipping mode for the slicer.

        Args:
            mode: "normal", "inverse", "inside", or "outside"
        """
        self._slicer_clip_mode = mode
        logger.info(f"Slicer clip mode set to: {mode}")

        # Reapply clipping with new mode
        # This would need to trigger the widget callback to refresh

    def set_slicer_widget_opacity(self, opacity: float):
        """Set the opacity of the slicer widget (0.0 to 1.0)."""
        self._slicer_widget_opacity = max(0.0, min(1.0, opacity))
        # Note: Changing widget opacity after creation may not be supported
        # by all PyVista widgets

    def reset_slicer_widget(self):
        """Reset the slicer widget to the center of the model."""
        if not hasattr(self, '_original_mesh_for_slicing') or self._original_mesh_for_slicing is None:
            return

        center = self._original_mesh_for_slicing.center
        bounds = self._original_mesh_for_slicing.bounds

        # Disable and re-enable with original position
        widget_type = getattr(self, '_slicer_widget_type', 'plane')
        callback = getattr(self, '_slicer_callback', None)

        self.disable_interactive_slicer()
        self.enable_interactive_slicer(widget_type=widget_type, callback=callback)

        logger.info("Slicer widget reset to center")

    def clear_all_clipping(self):
        """Clear all clipping and restore the full model."""
        self.disable_interactive_slicer()
        logger.info("All clipping cleared")

    # =====================================================================
    # Advanced Multi-Widget Slicer Methods
    # =====================================================================

    def add_slicer_widget(self, widget_type: str, widget_name: str, thickness: float = 0.0,
                         clip_mode: str = "normal", opacity: float = 0.3, callback=None) -> dict:
        """
        Add a named slicer widget (supports multiple simultaneous widgets).

        Args:
            widget_type: "plane", "box", "sphere", or "cylinder"
            widget_name: Unique name for this widget
            thickness: For plane widget, thickness of slab (0 = single plane)
            clip_mode: "normal", "inverse", "inside", or "outside"
            opacity: Widget opacity (0.0-1.0)
            callback: Optional callback(params) when widget moves

        Returns:
            dict: Widget parameters (origin, normal, bounds, etc.)
        """
        if self.plotter is None:
            raise RuntimeError("No plotter available")

        # Initialize multi-widget tracking if needed
        if not hasattr(self, '_slicer_widgets'):
            self._slicer_widgets = {}  # widget_name -> widget_info
            self._slicer_clipped_actors = {}  # widget_name -> actor_name
            self._slicer_widget_handlers = {}  # widget_name -> widget_object

        # Check for duplicate name
        if widget_name in self._slicer_widgets:
            raise ValueError(f"Widget '{widget_name}' already exists")

        # Get target mesh
        target_mesh = self._get_target_mesh_for_slicing()
        if target_mesh is None:
            raise RuntimeError("No mesh available to slice")

        # Store original mesh if first widget
        if not hasattr(self, '_original_mesh_for_multi_slicing'):
            self._original_mesh_for_multi_slicing = target_mesh

        bounds = target_mesh.bounds
        center = target_mesh.center

        # Create widget based on type
        widget_params = {}

        if widget_type == "plane":
            widget_params = self._add_plane_widget_multi(
                widget_name, center, bounds, thickness, opacity, callback
            )
        elif widget_type == "box":
            widget_params = self._add_box_widget_multi(
                widget_name, bounds, opacity, callback
            )
        elif widget_type == "sphere":
            widget_params = self._add_sphere_widget_multi(
                widget_name, center, bounds, opacity, callback
            )
        elif widget_type == "cylinder":
            widget_params = self._add_cylinder_widget_multi(
                widget_name, center, bounds, opacity, callback
            )
        else:
            raise ValueError(f"Unknown widget type: {widget_type}")

        # Store widget info
        self._slicer_widgets[widget_name] = {
            'type': widget_type,
            'thickness': thickness,
            'clip_mode': clip_mode,
            'opacity': opacity,
            'callback': callback,
            'params': widget_params
        }

        logger.info(f"Added slicer widget '{widget_name}' ({widget_type})")
        return widget_params

    def _add_plane_widget_multi(self, widget_name, center, bounds, thickness, opacity, callback):
        """Add a plane widget with optional slab thickness."""
        import numpy as np

        def plane_callback_wrapper(normal, origin):
            """Wrapper to handle thickness and multi-widget clipping."""
            params = {'normal': normal, 'origin': origin, 'thickness': thickness}

            # Apply multi-widget clipping
            self._apply_multi_widget_clipping()

            # Call user callback
            if callback:
                callback(params)

        # Add plane widget
        self.plotter.add_plane_widget(
            callback=plane_callback_wrapper,
            bounds=bounds,
            factor=1.2,
            normal='z',
            origin=center,
            color='cyan',
            implicit=False  # We handle clipping manually for multi-widget support
        )

        return {'origin': center, 'normal': [0, 0, 1], 'thickness': thickness}

    def _add_box_widget_multi(self, widget_name, bounds, opacity, callback):
        """Add a box widget."""
        def box_callback_wrapper(box_bounds):
            params = {'bounds': box_bounds}

            # Apply multi-widget clipping
            self._apply_multi_widget_clipping()

            if callback:
                callback(params)

        self.plotter.add_box_widget(
            callback=box_callback_wrapper,
            bounds=bounds,
            factor=0.8,
            rotation_enabled=True,
            color='yellow',
            opacity=opacity
        )

        return {'bounds': bounds}

    def _add_sphere_widget_multi(self, widget_name, center, bounds, opacity, callback):
        """Add a sphere widget."""
        import numpy as np

        diagonal = np.linalg.norm([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ])
        radius = diagonal / 4.0

        def sphere_callback_wrapper(center, radius):
            params = {'center': center, 'radius': radius}

            # Apply multi-widget clipping
            self._apply_multi_widget_clipping()

            if callback:
                callback(params)

        self.plotter.add_sphere_widget(
            callback=sphere_callback_wrapper,
            center=center,
            radius=radius,
            color='magenta',
            opacity=opacity
        )

        return {'center': center, 'radius': radius}

    def _add_cylinder_widget_multi(self, widget_name, center, bounds, opacity, callback):
        """Add a cylinder widget using line widget for axis."""
        import numpy as np

        # Calculate cylinder parameters
        diagonal_xy = np.linalg.norm([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2]
        ])
        radius = diagonal_xy / 6.0
        height = bounds[5] - bounds[4]

        # Define initial axis (vertical)
        point1 = [center[0], center[1], bounds[4]]
        point2 = [center[0], center[1], bounds[5]]

        def line_callback_wrapper(point1, point2):
            """Line widget defines the cylinder axis."""
            params = {
                'point1': point1,
                'point2': point2,
                'radius': radius,
                'axis': np.array(point2) - np.array(point1)
            }

            # Store for cylinder clipping
            self._slicer_widgets[widget_name]['params'] = params

            # Apply multi-widget clipping
            self._apply_multi_widget_clipping()

            if callback:
                callback(params)

        # Add line widget to define cylinder axis
        self.plotter.add_line_widget(
            callback=line_callback_wrapper,
            bounds=bounds,
            color='green',
            use_vertices=False
        )

        # Visualize cylinder (transparent)
        import pyvista as pv
        axis = np.array(point2) - np.array(point1)
        height_actual = np.linalg.norm(axis)
        direction = axis / height_actual if height_actual > 0 else [0, 0, 1]

        cylinder = pv.Cylinder(
            center=center,
            direction=direction,
            radius=radius,
            height=height_actual
        )

        self.plotter.add_mesh(
            cylinder,
            color='green',
            opacity=opacity,
            name=f'{widget_name}_cylinder_viz'
        )

        return {'point1': point1, 'point2': point2, 'radius': radius, 'axis': axis}

    def _apply_multi_widget_clipping(self):
        """Apply clipping from all active widgets combined."""
        if not hasattr(self, '_slicer_widgets') or not self._slicer_widgets:
            return

        if not hasattr(self, '_original_mesh_for_multi_slicing'):
            return

        mesh = self._original_mesh_for_multi_slicing

        # Apply each widget's clipping sequentially
        for widget_name, widget_info in self._slicer_widgets.items():
            widget_type = widget_info['type']
            params = widget_info['params']
            clip_mode = widget_info['clip_mode']
            thickness = widget_info.get('thickness', 0.0)

            try:
                if widget_type == "plane":
                    mesh = self._clip_with_plane(mesh, params, clip_mode, thickness)
                elif widget_type == "box":
                    mesh = self._clip_with_box(mesh, params, clip_mode)
                elif widget_type == "sphere":
                    mesh = self._clip_with_sphere(mesh, params, clip_mode)
                elif widget_type == "cylinder":
                    mesh = self._clip_with_cylinder(mesh, params, clip_mode)
            except Exception as e:
                logger.error(f"Error clipping with widget '{widget_name}': {e}")
                continue

        # Update visualization with combined clipped mesh
        self._update_clipped_visualization(mesh)

    def _clip_with_plane(self, mesh, params, clip_mode, thickness):
        """Clip mesh with plane (or slab if thickness > 0)."""
        import numpy as np

        normal = np.array(params['normal'])
        origin = np.array(params['origin'])

        if thickness > 0:
            # Slab clipping: keep blocks within thickness/2 on each side
            half_thick = thickness / 2.0

            # Create two planes
            origin1 = origin + normal * half_thick
            origin2 = origin - normal * half_thick

            # Clip between the two planes
            mesh = mesh.clip(normal=normal, origin=origin1, invert=False)
            mesh = mesh.clip(normal=-normal, origin=origin2, invert=False)
        else:
            # Single plane clipping
            if clip_mode == "inverse":
                mesh = mesh.clip(normal=-normal, origin=origin)
            else:
                mesh = mesh.clip(normal=normal, origin=origin)

        return mesh

    def _clip_with_box(self, mesh, params, clip_mode):
        """Clip mesh with box."""
        bounds = params['bounds']

        if clip_mode == "inside" or clip_mode == "normal":
            mesh = mesh.clip_box(bounds, invert=False)
        else:
            mesh = mesh.clip_box(bounds, invert=True)

        return mesh

    def _clip_with_sphere(self, mesh, params, clip_mode):
        """Clip mesh with sphere."""
        import vtk

        center = params['center']
        radius = params['radius']

        sphere = vtk.vtkSphere()
        sphere.SetCenter(center)
        sphere.SetRadius(radius)

        if clip_mode == "inside" or clip_mode == "normal":
            mesh = mesh.clip_scalar(
                scalars='implicit_distance',
                value=0.0,
                invert=False,
                implicit_function=sphere
            )
        else:
            mesh = mesh.clip_scalar(
                scalars='implicit_distance',
                value=0.0,
                invert=True,
                implicit_function=sphere
            )

        return mesh

    def _clip_with_cylinder(self, mesh, params, clip_mode):
        """Clip mesh with cylinder."""
        import vtk
        import numpy as np

        point1 = np.array(params['point1'])
        point2 = np.array(params['point2'])
        radius = params['radius']

        # Create cylinder implicit function
        axis = point2 - point1
        center = (point1 + point2) / 2.0

        cylinder = vtk.vtkCylinder()
        cylinder.SetCenter(center)
        cylinder.SetRadius(radius)
        cylinder.SetAxis(axis / np.linalg.norm(axis))

        if clip_mode == "inside" or clip_mode == "normal":
            mesh = mesh.clip_scalar(
                scalars='implicit_distance',
                value=0.0,
                invert=False,
                implicit_function=cylinder
            )
        else:
            mesh = mesh.clip_scalar(
                scalars='implicit_distance',
                value=0.0,
                invert=True,
                implicit_function=cylinder
            )

        return mesh

    def _update_clipped_visualization(self, clipped_mesh):
        """Update the visualization with the clipped mesh."""
        # Remove old clipped mesh
        if hasattr(self, '_multi_widget_clipped_actor'):
            try:
                self.plotter.remove_actor(self._multi_widget_clipped_actor, render=False)
            except:
                pass

        # Add new clipped mesh
        actor_name = f"multi_widget_clipped_{id(clipped_mesh)}"
        self._multi_widget_clipped_actor = actor_name

        # Get active scalars
        active_scalars = None
        if hasattr(self._original_mesh_for_multi_slicing, 'active_scalars_name'):
            active_scalars = self._original_mesh_for_multi_slicing.active_scalars_name

        self.plotter.add_mesh(
            clipped_mesh,
            scalars=active_scalars,
            name=actor_name,
            show_edges=True,
            edge_color='white',
            opacity=1.0
        )

        self.plotter.render()

    def remove_slicer_widget(self, widget_name: str):
        """Remove a specific slicer widget by name."""
        if not hasattr(self, '_slicer_widgets') or widget_name not in self._slicer_widgets:
            raise ValueError(f"Widget '{widget_name}' not found")

        # Remove widget info
        del self._slicer_widgets[widget_name]

        # TODO: Remove specific widget from plotter
        # PyVista doesn't provide easy way to remove specific widgets
        # For now, clear all and re-add remaining widgets

        # Reapply clipping with remaining widgets
        if self._slicer_widgets:
            self._apply_multi_widget_clipping()
        else:
            # No widgets left, restore original mesh
            self.remove_all_slicer_widgets()

        logger.info(f"Removed slicer widget '{widget_name}'")

    def remove_all_slicer_widgets(self):
        """Remove all slicer widgets and restore original mesh."""
        if self.plotter is None:
            return

        # Clear all widgets
        self.plotter.clear_plane_widgets()
        self.plotter.clear_box_widgets()
        self.plotter.clear_sphere_widgets()
        self.plotter.clear_line_widgets()

        # Remove clipped actor
        if hasattr(self, '_multi_widget_clipped_actor'):
            try:
                self.plotter.remove_actor(self._multi_widget_clipped_actor, render=False)
            except:
                pass
            delattr(self, '_multi_widget_clipped_actor')

        # Clear tracking
        if hasattr(self, '_slicer_widgets'):
            self._slicer_widgets.clear()

        if hasattr(self, '_original_mesh_for_multi_slicing'):
            delattr(self, '_original_mesh_for_multi_slicing')

        self.plotter.render()
        logger.info("All slicer widgets removed")

    def set_global_clip_mode(self, mode: str):
        """Set clipping mode for all widgets."""
        if not hasattr(self, '_slicer_widgets'):
            return

        # Update all widgets
        for widget_name in self._slicer_widgets:
            self._slicer_widgets[widget_name]['clip_mode'] = mode

        # Reapply clipping
        self._apply_multi_widget_clipping()

        logger.info(f"Global clip mode set to: {mode}")

    def set_all_widget_opacity(self, opacity: float):
        """Set opacity for all widgets."""
        # Note: PyVista doesn't support changing widget opacity after creation
        # This would need to be implemented by recreating widgets
        logger.warning("Changing widget opacity after creation not fully supported")

    def set_widgets_visibility(self, visible: bool):
        """Toggle visibility of all widget geometry (clipping remains active)."""
        # Note: PyVista doesn't directly support hiding widget geometry
        # while keeping them functional
        logger.warning("Widget visibility toggle not fully supported by PyVista")

    def export_clipped_section(self, filepath: str, format_type: str = "vtk"):
        """
        Export the clipped mesh to file.

        Args:
            filepath: Output file path
            format_type: "vtk" or "csv"
        """
        if not hasattr(self, '_multi_widget_clipped_actor'):
            raise RuntimeError("No clipped mesh available to export")

        # Get the clipped mesh
        clipped_mesh = None
        try:
            actor = self.plotter.actors.get(self._multi_widget_clipped_actor)
            if actor and hasattr(actor, 'mapper') and hasattr(actor.mapper, 'dataset'):
                clipped_mesh = actor.mapper.dataset
        except Exception as e:
            raise RuntimeError(f"Could not retrieve clipped mesh: {e}")

        if clipped_mesh is None:
            raise RuntimeError("Clipped mesh not found")

        if format_type == "vtk":
            # Export to VTK file
            clipped_mesh.save(filepath)
            logger.info(f"Exported clipped mesh to VTK: {filepath}")

        elif format_type == "csv":
            # Export to CSV (point data only)
            import pandas as pd
            import numpy as np

            # Extract points
            points = np.array(clipped_mesh.points)
            df_data = {
                'X': points[:, 0],
                'Y': points[:, 1],
                'Z': points[:, 2]
            }

            # Add scalar fields
            for array_name in clipped_mesh.array_names:
                try:
                    data = clipped_mesh[array_name]
                    if len(data) == len(points):
                        df_data[array_name] = data
                except:
                    pass

            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False)
            logger.info(f"Exported clipped mesh to CSV: {filepath} ({len(df)} points)")

        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def get_clipped_mesh(self):
        """Get the current clipped mesh (for external processing)."""
        if not hasattr(self, '_multi_widget_clipped_actor'):
            return None

        try:
            actor = self.plotter.actors.get(self._multi_widget_clipped_actor)
            if actor and hasattr(actor, 'mapper') and hasattr(actor.mapper, 'dataset'):
                return actor.mapper.dataset
        except:
            pass

        return None
