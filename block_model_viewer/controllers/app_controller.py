"""
App Controller - Thin orchestration layer between UI panels and the Renderer.

Provides centralized state management and signal routing for visualization operations.
Delegates domain-specific operations to specialized sub-controllers.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Sequence, Callable, Iterable, Union, List
from pathlib import Path
import logging
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyvista as pv

try:  # PyQt optional during headless tests
    from PyQt6.QtCore import QObject, pyqtSignal, QTimer
except Exception:  # pragma: no cover - fallback when Qt not available
    QObject = object  # type: ignore
    pyqtSignal = lambda *_, **__: None  # type: ignore

    class _FallbackTimer:
        @staticmethod
        def singleShot(_milliseconds, callback):
            if callable(callback):
                callback()

    QTimer = _FallbackTimer  # type: ignore

from ..irr_engine import run_irr, run_npv, run_pit_optimisation, IRRConfig, PitConfig, ScheduleConfig
from ..visualization.render_payloads import (
    MeshPayload, GridPayload, CrossSectionPayload, PitShellPayload,
    PointCloudPayload, LinePayload, SecondaryViewPayload
)
from ..utils.chunk_loader import ChunkLoader, assemble_chunks_into_model
from ..utils.profiling import start_section, end_section
from .job_worker import JobWorker
from .job_registry import JobRegistry
from .controller_signals import ControllerSignals
from .app_state import AppState, is_feature_enabled
from ..core.process_history_tracker import get_process_history_tracker

# Import sub-controllers
from .geostats_controller import GeostatsController
from .mining_controller import MiningController
from .vis_controller import VisController
from .data_controller import DataController
from .scan_controller import ScanController
from .survey_deformation_controller import SurveyDeformationController
from .insar_controller import InsarController

logger = logging.getLogger(__name__)

_ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=2)


@dataclass
class SessionState:
    """Centralized application state for visualization and data management."""
    project_path: Optional[str] = None
    dataset_meta: Dict[str, Any] = field(default_factory=dict)
    current_property: Optional[str] = None
    color_map: str = "viridis"
    transparency: float = 1.0
    legend_visible: bool = True
    slice_axis: str = "x"
    slice_position: float = 0.0
    filter_specs: Dict[str, tuple] = field(default_factory=dict)


@dataclass
class PerformanceSettings:
    """Performance optimization settings."""
    lod_quality: float = 0.7  # 0.0 = low quality, 1.0 = high quality
    max_render_cells: int = 1_000_000
    downsample_large_models: bool = True
    enable_async_loading: bool = True
    aggressive_compaction: bool = False


class AppController:
    """
    Thin orchestration layer between UI panels and the Renderer.
    
    Manages shared state and provides unified interface for visualization operations.
    Delegates domain-specific work to specialized sub-controllers:
    - GeostatsController: kriging, simulation, variogram, analysis
    - MiningController: resources, IRR/NPV, scheduling, planning
    - VisController: rendering, layers, legend, overlays
    - DataController: drillholes, geology, structural, geotech
    """
    
    def __init__(self, renderer, config=None, registry=None):
        """
        Initialize controller with renderer and optional config.
        
        Args:
            renderer: Renderer instance for visualization operations
            config: Optional configuration object
            registry: Optional DataRegistry instance (for dependency injection)
        """
        self.r = renderer
        self.cfg = config
        
        # Dependency Injection - registry should always be provided
        if registry:
            self.registry = registry
        else:
            # LEGACY FALLBACK ONLY: Singleton pattern for backward compatibility
            # New code should always pass registry via dependency injection
            # This fallback exists only for edge cases where Controller is instantiated directly
            from ..core.data_registry import DataRegistry
            self.registry = DataRegistry.instance()
            logger.warning("AppController initialized without registry - using singleton fallback. "
                         "This should only happen in legacy code. New code should use dependency injection.")
        
        self.s = SessionState()
        self.legend_manager = getattr(self.r, "legend_manager", None)
        # Unified overlay manager (consolidated from overlay_manager + axis_manager)
        self.overlay_manager = getattr(self.r, "overlay_state", None)
        # Backward compatibility: axis_manager now points to overlay_manager
        self.axis_manager = self.overlay_manager
        self._block_model = None
        self._viewer = None
        self._scene_listeners: list[weakref.ReferenceType] = []
        self._legend_listeners: list[weakref.ReferenceType] = []
        
        # Step 11: Unified worker pipeline
        self.signals = ControllerSignals()
        self.job_registry = JobRegistry()
        self._active_workers: Dict[str, JobWorker] = {}
        
        # Application state - single source of truth (initialized to EMPTY)
        self._app_state: AppState = AppState.EMPTY
        
        # STEP 18: Performance settings
        self.performance_settings = PerformanceSettings()
        self.chunk_loader: Optional[ChunkLoader] = None
        
        # STEP 31: Planning Dashboard & Scenario Manager
        from ..planning.scenario_store import ScenarioStore
        from ..planning.scenario_runner import ScenarioRunner
        
        scenario_base_path = Path.home() / ".block_model_viewer" / "scenarios"
        self.scenario_store = ScenarioStore(scenario_base_path)
        self.scenario_runner = ScenarioRunner(self, self.scenario_store)

        # Initialize sub-controllers
        self._geostats = GeostatsController(self)
        self._mining = MiningController(self)
        self._vis = VisController(self)
        self._data = DataController(self)
        self._scan = ScanController(self)
        self._survey_deformation = SurveyDeformationController(self)
        self._insar = InsarController(self)

        # Initialize job registry with sub-controllers
        self.job_registry.initialize_with_subcontrollers(
            self._geostats, self._mining, self._vis, self._data, self._scan, self._survey_deformation, self._insar
        )

        # Attempt to hydrate session from config if available
        if self.cfg and hasattr(self.cfg, 'get_visualization_settings'):
            try:
                vis = self.cfg.get_visualization_settings()
                self.s.color_map = vis.get("colormap", "viridis")
                self.s.transparency = vis.get("transparency", 1.0)
            except Exception as e:
                logger.warning(f"Could not load visualization settings from config: {e}")

        # Attach to renderer layer change events
        try:
            if hasattr(self.r, "set_layer_change_callback"):
                self.r.set_layer_change_callback(self.notify_panels_scene_changed)
        except Exception:
            logger.debug("Renderer does not support layer change callback multicast", exc_info=True)

    # ------------------------------------------------------------------
    # Sub-controller accessors
    # ------------------------------------------------------------------
    @property
    def geostats(self) -> GeostatsController:
        """Access geostatistics sub-controller."""
        return self._geostats
    
    @property
    def mining(self) -> MiningController:
        """Access mining/planning sub-controller."""
        return self._mining
    
    @property
    def vis(self) -> VisController:
        """Access visualization sub-controller."""
        return self._vis
    
    @property
    def data(self) -> DataController:
        """Access data (drillhole/geology/geotech) sub-controller."""
        return self._data

    @property
    def survey_deformation(self) -> SurveyDeformationController:
        """Access survey deformation & subsidence sub-controller."""
        return self._survey_deformation

    @property
    def insar(self) -> InsarController:
        """Access InSAR orchestration sub-controller."""
        return self._insar

    @property
    def scan_controller(self) -> ScanController:
        """Access scan analysis sub-controller."""
        return self._scan

    # ------------------------------------------------------------------
    # Application State Management
    # ------------------------------------------------------------------
    @property
    def app_state(self) -> AppState:
        """Get current application state."""
        return self._app_state
    
    def set_app_state(self, state: AppState) -> None:
        """
        Set application state and notify all listeners.
        
        This is the ONLY way to change application state. All UI panels
        must react to state changes via the app_state_changed signal.
        
        Args:
            state: New AppState value
        """
        if self._app_state == state:
            return
        
        old_state = self._app_state
        self._app_state = state
        logger.info(f"AppState changed: {old_state.name} -> {state.name}")
        
        # Emit signal to notify all listeners
        try:
            self.signals.app_state_changed.emit(int(state))
        except Exception as e:
            logger.warning(f"Failed to emit app_state_changed signal: {e}")
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a UI feature should be enabled for current state.
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature should be visible/enabled
        """
        return is_feature_enabled(self._app_state, feature)
    
    def _update_state_from_scene(self) -> None:
        """
        Update app state based on current scene state.
        
        Called internally after scene changes to ensure state consistency.
        """
        logger.debug(f"[STATE DEBUG] _update_state_from_scene called, current state: {self._app_state.name}")
        
        # Check if we have active workers (BUSY state)
        if self._active_workers:
            logger.debug(f"[STATE DEBUG] Active workers detected, setting BUSY state")
            self.set_app_state(AppState.BUSY)
            return
        
        # Check if we have rendered layers (RENDERED state)
        try:
            layers = self.get_scene_layers()
            logger.debug(f"[STATE DEBUG] get_scene_layers returned {len(layers) if layers else 0} layers")
            if layers and len(layers) > 0:
                logger.debug(f"[STATE DEBUG] Layers present, setting RENDERED state")
                self.set_app_state(AppState.RENDERED)
                return
        except Exception as e:
            logger.debug(f"[STATE DEBUG] get_scene_layers failed: {e}")
        
        # Check if we have data loaded (DATA_LOADED state)
        try:
            has_drillholes = self.registry.get_drillhole_data() is not None
            has_block_model = self._block_model is not None
            logger.debug(f"[STATE DEBUG] has_drillholes={has_drillholes}, has_block_model={has_block_model}")
            if has_drillholes or has_block_model:
                logger.debug(f"[STATE DEBUG] Data present, setting DATA_LOADED state")
                self.set_app_state(AppState.DATA_LOADED)
                return
        except Exception as e:
            logger.debug(f"[STATE DEBUG] Data check failed: {e}")
        
        # Default to EMPTY
        logger.debug(f"[STATE DEBUG] No data or layers, setting EMPTY state")
        self.set_app_state(AppState.EMPTY)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    @property
    def renderer(self):
        """Expose renderer for legacy callers while encouraging controller indirection."""
        return self.r

    @property
    def block_model(self):
        """Return the currently loaded block model if available."""
        if self._block_model is not None:
            return self._block_model
        return getattr(self.r, "current_model", None)
    
    @property
    def current_block_model(self):
        """Alias for block_model property."""
        return self.block_model

    # ------------------------------------------------------------------
    # Listener registration
    # ------------------------------------------------------------------
    def _make_weakref(self, callback: Callable):
        """Create a weak reference for a callback (handling bound/unbound)."""
        try:
            if hasattr(callback, "__self__") and callback.__self__ is not None:
                return weakref.WeakMethod(callback)
            return weakref.ref(callback)
        except TypeError:
            return lambda: callback

    def register_scene_listener(self, callback: Callable[[Sequence[Any]], None]) -> None:
        """Register a listener notified whenever scene layers change."""
        if callback is None:
            return
        ref = self._make_weakref(callback)
        for existing in list(self._scene_listeners):
            target = existing()
            if target is callback:
                return
        self._scene_listeners.append(ref)

    def _iter_listeners(self, store: list[weakref.ReferenceType]) -> Iterable[Callable]:
        """Yield alive callbacks and prune stale weak references."""
        alive: list[weakref.ReferenceType] = []
        for ref in store:
            try:
                cb = ref()
            except TypeError:
                cb = None
            if cb is None:
                continue
            alive.append(ref)
            yield cb
        store[:] = alive

    # ------------------------------------------------------------------
    # Scene layer helpers
    # ------------------------------------------------------------------
    def get_scene_layers(self) -> Sequence[Any]:
        """Return a list of scene layer objects managed by the renderer."""
        try:
            if hasattr(self.r, "scene_layers"):
                layers = getattr(self.r, "scene_layers")
                if isinstance(layers, dict):
                    return list(layers.values())
                return list(layers)
        except Exception:
            logger.debug("Failed to fetch scene layers from renderer", exc_info=True)
        return []
    
    def notify_panels_scene_changed(self) -> None:
        """Broadcast current scene layers to registered panels."""
        layers = tuple(self.get_scene_layers())
        for callback in self._iter_listeners(self._scene_listeners):
            try:
                callback(layers)
            except Exception:
                logger.debug("Scene listener raised", exc_info=True)
        
        # Update app state based on new scene state
        self._update_state_from_scene()
    
    # ---------- Data lifecycle ----------
    
    def load_block_model(self, block_model):
        """Load a block model into the renderer and apply visual preferences."""
        start_section("load_block_model")
        perf_controller_start = time.perf_counter()

        if self.performance_settings.aggressive_compaction and self._block_model:
            try:
                if hasattr(self._block_model, 'compact'):
                    self._block_model.compact()
            except Exception:
                pass
        
        render_start = time.perf_counter()
        self.r.load_block_model(block_model)
        render_elapsed = time.perf_counter() - render_start
        logger.info(f"PERF: Controller renderer.load_block_model took {render_elapsed:.3f}s")

        self._block_model = block_model

        axis_elapsed = 0.0
        try:
            axis_start = time.perf_counter()
            if self.axis_manager is not None and hasattr(self.r, "_get_scene_bounds"):
                bounds = self.r._get_scene_bounds()
                try:
                    self.axis_manager.set_bounds(bounds)
                except Exception:
                    logger.debug("Axis manager scene bounds update failed", exc_info=True)
            axis_elapsed = time.perf_counter() - axis_start
        except Exception:
            logger.debug("Axis manager scene bounds update failed", exc_info=True)

        prefs_start = time.perf_counter()
        self._vis._apply_visual_prefs()
        prefs_elapsed = time.perf_counter() - prefs_start

        notify_start = time.perf_counter()
        self.notify_panels_scene_changed()
        notify_elapsed = time.perf_counter() - notify_start

        logger.info(
            "Controller: Loaded block model with %d blocks (axis: %.3fs, prefs: %.3fs, listeners: %.3fs)",
            block_model.block_count,
            axis_elapsed,
            prefs_elapsed,
            notify_elapsed
        )

        total_elapsed = time.perf_counter() - perf_controller_start
        logger.info("PERF: Controller load_block_model internal total %.3fs", total_elapsed)

        end_section("load_block_model")
        
        # Update app state - block model loaded and rendered
        self._update_state_from_scene()
    
    def configure_performance(self, settings: PerformanceSettings) -> None:
        """Configure performance settings."""
        self.performance_settings = settings
        
        if hasattr(self.r, 'set_performance_settings'):
            self.r.set_performance_settings({
                'lod_quality': settings.lod_quality,
                'max_render_cells': settings.max_render_cells
            })
        
        if settings.enable_async_loading:
            if self.chunk_loader is None:
                self.chunk_loader = ChunkLoader(chunk_size=200_000)
        else:
            self.chunk_loader = None
        
        logger.info(f"Performance settings updated: LOD={settings.lod_quality}, "
                   f"Max cells={settings.max_render_cells}, Async={settings.enable_async_loading}")

    # ------------------------------------------------------------------
    # Task execution pipeline (Step 11)
    # ------------------------------------------------------------------
    
    def run_task(self, task: str, params: Dict[str, Any], callback: Optional[Callable[[Dict[str, Any]], None]] = None, progress_callback: Optional[Callable[[int, str], None]] = None) -> None:
        """
        Unified task execution pipeline.

        All analysis, resource, and planning tasks run through this single entry point.
        """
        func = self.job_registry.get(task)
        if func is None:
            error_msg = f"Unknown task '{task}'. Available: {self.job_registry.list_tasks()}"
            logger.error(error_msg)
            self.signals.task_error.emit(task, error_msg)
            return

        wrapped_params = params.copy()
        if progress_callback:
            wrapped_params['_progress_callback'] = progress_callback

        worker = JobWorker(func, wrapped_params, parent=None)

        # Patch: Expect percent, message from worker
        worker.progress.connect(lambda percent, message: self.signals.task_progress.emit(task, percent, message))
        worker.finished.connect(lambda result: self._on_task_complete(task, result, callback))
        worker.error.connect(lambda msg, tb: self._on_task_error(task, msg, tb))

        # Start process tracking
        process_tracker = get_process_history_tracker()
        process_id = process_tracker.start_process(task, params)

        # Store process_id in worker for completion tracking
        worker._process_id = process_id

        self._active_workers[task] = worker
        self.signals.task_started.emit(task)
        logger.info(f"Task '{task}' started")
        
        # Set BUSY state when task starts
        self.set_app_state(AppState.BUSY)
        
        worker.start()
    
    def cancel_task(self, task: str) -> bool:
        """Cancel a running task."""
        if task not in self._active_workers:
            logger.warning(f"Task '{task}' not found in active workers")
            return False
        
        worker = self._active_workers.pop(task, None)
        if worker:
            if hasattr(worker, 'cancel'):
                worker.cancel()
            logger.info(f"Task '{task}' cancellation requested")
            try:
                if worker.isRunning():
                    worker.quit()
                    worker.wait(5000)
                worker.deleteLater()
            except Exception as e:
                logger.debug(f"Worker cleanup for '{task}' failed: {e}")
            return True
        return False
    
    def _on_task_complete(self, task: str, result: Any, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """
        Handle task completion.

        This method is called on the main thread via Qt signal connection,
        so UI updates can be done directly.
        """
        worker = self._active_workers.pop(task, None)

        # Complete process tracking
        if worker and hasattr(worker, '_process_id'):
            process_tracker = get_process_history_tracker()
            # Generate result summary based on result type
            result_summary = self._generate_result_summary(task, result)
            process_tracker.complete_process(worker._process_id, success=True, result_summary=result_summary)

        logger.info(f"Task '{task}' completion handler called. Result type: {type(result)}")

        if result is None:
            logger.warning(f"Task '{task}' completed with no result")
            self.signals.task_error.emit(task, "Task returned no result")
            if callback:
                try:
                    callback(None)
                except Exception as e:
                    logger.error(f"Error callback for task '{task}' raised: {e}", exc_info=True)
            return
        
        # Call user callback FIRST (before visualization) so UI can update immediately
        # This ensures progress dialog closes before heavy visualization work
        if callback:
            try:
                logger.debug(f"Calling user callback for task '{task}'")
                callback(result)
                logger.debug(f"User callback completed for task '{task}'")
            except Exception as e:
                logger.error(f"Callback for task '{task}' raised: {e}", exc_info=True)
                # Still try to apply visualization even if callback failed
                self.signals.task_error.emit(task, f"Callback error: {e}")

        # Skip automatic 3D visualization for Simple Kriging (user must click "Visualize" button)
        # This prevents UI conflicts and gives users explicit control
        if task != "simple_kriging":
            try:
                self._vis.apply_results_to_model(result)
            except Exception as e:
                logger.error(f"Error applying visualization for task '{task}': {e}", exc_info=True)

        # Emit completion signal
        self.signals.task_finished.emit(task)

        logger.info(f"Task '{task}' completed successfully")
        
        # Update state after task completion (no longer BUSY)
        self._update_state_from_scene()
    
    def _on_task_error(self, task: str, msg: str, tb: Optional[str] = None) -> None:
        """Handle task error."""
        worker = self._active_workers.pop(task, None)

        # Mark process as failed in tracking
        if worker and hasattr(worker, '_process_id'):
            process_tracker = get_process_history_tracker()
            process_tracker.complete_process(worker._process_id, success=False, error_message=msg)

        logger.error(f"Task '{task}' failed: {msg}")
        if tb:
            logger.debug(f"Task '{task}' traceback:\n{tb}")
        self.signals.task_error.emit(task, msg)
        
        # Update state after task error (no longer BUSY)
        self._update_state_from_scene()

    def _generate_result_summary(self, task: str, result: Any) -> Optional[str]:
        """
        Generate a brief summary of task results for process history.

        Args:
            task: Task name
            result: Result object from task execution

        Returns:
            Brief summary string or None
        """
        try:
            if result is None:
                return None

            # Handle different result types based on task
            if task in ["variogram", "kriging", "sgsim", "simple_kriging", "cokriging", "indicator_kriging", "universal_kriging", "soft_kriging"]:
                # Geostatistics results
                if isinstance(result, dict):
                    if "model" in result:
                        return f"Generated {task.replace('_', ' ')} model"
                    elif "results" in result and isinstance(result["results"], dict):
                        results = result["results"]
                        if "block_model" in results:
                            bm = results["block_model"]
                            if hasattr(bm, "__len__"):
                                return f"Estimated {len(bm)} blocks"
                            else:
                                return f"Generated block model estimate"
                        elif len(results) > 0:
                            return f"Generated {len(results)} result datasets"
                return f"Completed {task.replace('_', ' ')} analysis"

            elif task in ["resources", "classify"]:
                # Resource calculation results
                if isinstance(result, dict) and "summary" in result:
                    summary = result["summary"]
                    if isinstance(summary, dict) and "total_tonnage" in summary:
                        tonnage = summary["total_tonnage"]
                        if isinstance(tonnage, (int, float)):
                            return f"Calculated {tonnage:,.0f} tonnes of resources"
                return f"Completed resource calculation"

            elif task in ["irr", "npv"]:
                # Financial analysis results
                if isinstance(result, dict):
                    if "irr" in result:
                        irr = result["irr"]
                        if isinstance(irr, (int, float)):
                            return f"Calculated IRR: {irr:.1f}%"
                    elif "npv" in result:
                        npv = result["npv"]
                        if isinstance(npv, (int, float)):
                            return f"Calculated NPV: ${npv:,.0f}"
                return f"Completed financial analysis"

            elif task in ["pit_opt"]:
                # Pit optimization results
                return "Generated pit optimization results"

            elif task in ["build_block_model", "load_file"]:
                # Data loading results
                if isinstance(result, dict) and "model" in result:
                    model = result["model"]
                    if hasattr(model, "nx") and hasattr(model, "ny") and hasattr(model, "nz"):
                        return f"Loaded {model.nx}×{model.ny}×{model.nz} block model"
                return f"Loaded data file"

            elif task in ["load_drillholes", "drillhole_import"]:
                # Drillhole loading results
                if isinstance(result, dict):
                    if "assay_count" in result:
                        return f"Loaded {result['assay_count']} assays"
                    elif "collar_count" in result:
                        return f"Loaded {result['collar_count']} collars"
                return "Loaded drillhole data"

            # Default summary
            if isinstance(result, dict):
                return f"Generated {len(result)} result datasets"
            elif hasattr(result, "__len__"):
                return f"Generated {len(result)} items"
            else:
                return f"Completed {task.replace('_', ' ')}"

        except Exception as e:
            logger.debug(f"Error generating result summary for {task}: {e}")
            return f"Completed {task.replace('_', ' ')}"

    # ==================================================================
    # BACKWARD-COMPATIBLE DELEGATION TO SUB-CONTROLLERS
    # All public methods below delegate to the appropriate sub-controller
    # ==================================================================
    
    # ------------------------------------------------------------------
    # Visualization delegates (to VisController)
    # ------------------------------------------------------------------
    def set_active_property(self, name: Optional[str]):
        """Set the active property for visualization."""
        self._vis.set_active_property(name)
    
    def set_current_property(self, name: Optional[str]):
        """Legacy method - delegates to set_active_property."""
        self._vis.set_current_property(name)
    
    def set_colormap(self, cmap: str):
        """Change the colormap."""
        self._vis.set_colormap(cmap)
    
    def set_transparency(self, alpha: float):
        """Set transparency/opacity."""
        self._vis.set_transparency(alpha)
    
    def toggle_legend(self, show: bool):
        """Toggle legend visibility."""
        self._vis.toggle_legend(show)
    
    def reset_scene(self):
        """Reset camera and re-apply visualization settings."""
        self._vis.reset_scene()
    
    def fit_to_view(self) -> None:
        """Fit current scene to viewport."""
        self._vis.fit_to_view()
    
    def set_view_preset(self, preset: str) -> None:
        """Apply a camera preset orientation."""
        self._vis.set_view_preset(preset)
    
    def set_projection_mode(self, orthographic: bool) -> None:
        """Enable or disable orthographic projection."""
        self._vis.set_projection_mode(orthographic)
    
    def set_trackball_mode(self, enabled: bool) -> None:
        """Switch interaction styles."""
        self._vis.set_trackball_mode(enabled)
    
    def apply_slice(self, axis: str, position: float):
        """Apply a spatial slice."""
        self._vis.apply_slice(axis, position)
    
    def apply_filters(self, filters: Dict[str, tuple]):
        """Apply property filters."""
        self._vis.apply_filters(filters)
    
    def refresh_scene(self) -> None:
        """Refresh the scene visualization."""
        self._vis.refresh_scene()
    
    def set_layer_visibility(self, layer_name: str, visible: bool) -> None:
        """Set layer visibility."""
        self._vis.set_layer_visibility(layer_name, visible)
    
    def set_layer_opacity(self, layer_name: str, opacity: float) -> None:
        """Set layer opacity."""
        self._vis.set_layer_opacity(layer_name, opacity)
    
    def set_active_layer(self, layer_name: str) -> None:
        """Set the active layer."""
        self._vis.set_active_layer(layer_name)
    
    def remove_layer(self, layer_name: str) -> None:
        """Remove a layer."""
        self._vis.remove_layer(layer_name)
    
    def set_legend_visibility(self, visible: bool) -> None:
        """Set legend visibility."""
        self._vis.set_legend_visibility(visible)
    
    def set_legend_orientation(self, orientation: str) -> None:
        """Set legend orientation."""
        self._vis.set_legend_orientation(orientation)
    
    def set_legend_font_size(self, size: int) -> None:
        """Set legend font size."""
        self._vis.set_legend_font_size(size)
    
    def reset_legend_position(self) -> None:
        """Reset legend position."""
        self._vis.reset_legend_position()
    
    def set_axes_visible(self, show: bool) -> None:
        """Set axes visibility."""
        self._vis.set_axes_visible(show)
    
    def set_bounds_visible(self, show: bool) -> None:
        """Set bounds visibility."""
        self._vis.set_bounds_visible(show)
    
    def set_ground_grid_visible(self, show: bool) -> None:
        """Set ground grid visibility."""
        self._vis.set_ground_grid_visible(show)
    
    def set_ground_grid_spacing(self, spacing: float) -> None:
        """Set ground grid spacing."""
        self._vis.set_ground_grid_spacing(spacing)
    
    def reset_ground_grid_spacing(self) -> None:
        """Reset ground grid spacing."""
        self._vis.reset_ground_grid_spacing()
    
    def set_overlay_units(self, units: str) -> None:
        """Set overlay units."""
        self._vis.set_overlay_units(units)
    
    def set_background_color(self, color: tuple) -> None:
        """Set background color."""
        self._vis.set_background_color(color)
    
    def set_edge_color(self, color: tuple) -> None:
        """Set edge color."""
        self._vis.set_edge_color(color)

    def set_edge_visibility(self, visible: bool) -> None:
        """Set edge visibility for block models."""
        self._vis.set_edge_visibility(visible)

    def set_lighting_enabled(self, enabled: bool) -> None:
        """Enable or disable lighting."""
        self._vis.set_lighting_enabled(enabled)
    
    def set_global_opacity(self, opacity: float) -> None:
        """Set global opacity."""
        self._vis.set_global_opacity(opacity)
    
    def configure_overlays(self, **kwargs):
        """Configure overlays."""
        self._vis.configure_overlays(**kwargs)
    
    def apply_render_payload(self, payload: Any) -> Any:
        """Apply a render payload."""
        return self._vis.apply_render_payload(payload)
    
    def apply_results_to_model(self, payload: Dict[str, Any]) -> None:
        """Apply analysis results to renderer."""
        self._vis.apply_results_to_model(payload)
    
    def export_screenshot(self, path: str, resolution=(1920, 1080)):
        """Export screenshot."""
        self._vis.export_screenshot(path, resolution)
    
    def show_schedule(self, schedule_df, mode: str = "Period"):
        """Show schedule visualization."""
        self._vis.show_schedule(schedule_df, mode)
    
    def show_optimal_pit(self, pit_df):
        """Show optimal pit."""
        self._vis.show_optimal_pit(pit_df)
    
    def render_pushback_layer(self, plan: Any, style_config: Optional[Dict[str, Any]] = None) -> None:
        """Render pushback layer."""
        self._vis.render_pushback_layer(plan, style_config)
    
    def export_resources(self, result: Any, path: Union[str, Path], excel: bool = False) -> None:
        """Export resources."""
        self._vis.export_resources(result, path, excel)
    
    def export_irr_results(self, result: Any, path: Union[str, Path], excel: bool = False) -> None:
        """Export IRR results."""
        self._vis.export_irr_results(result, path, excel)
    
    def export_analysis_results(self, result: Any, path: Union[str, Path], excel: bool = False, result_type: str = "analysis") -> None:
        """Export analysis results."""
        self._vis.export_analysis_results(result, path, excel, result_type)
    
    # ------------------------------------------------------------------
    # Geostatistics delegates (to GeostatsController)
    # ------------------------------------------------------------------
    
    def _get_estimation_ready_data(
        self, 
        task_name: str, 
        require_validation: bool = True,
        prefer_declustered: bool = False
    ) -> Optional[Any]:
        """
        UNIFIED DATA ACCESSOR for all estimation tasks.
        
        ENFORCEMENT:
        - Single entry point for kriging, SGSIM, variogram, etc.
        - Validates lineage before returning data (compositing required)
        - NEVER falls back to raw assays
        - Logs audit trail
        
        This method implements the data lineage gate for the geostatistical
        pipeline. It ensures all estimation engines use properly prepared data.
        
        Args:
            task_name: Name of requesting task (for audit/error messages)
            require_validation: If True, require validation to have passed
            prefer_declustered: If True, return declustered data if available
            
        Returns:
            Validated, composited (or declustered) DataFrame with provenance attrs
            Returns None if gates fail (error emitted via signals)
        """
        try:
            if prefer_declustered:
                df = self.registry.get_estimation_ready_data(
                    prefer_declustered=True,
                    require_validation=require_validation
                )
            else:
                df = self.registry.get_validated_composites(
                    require_validation=require_validation
                )
            
            # Log successful access for audit trail
            source_type = df.attrs.get('source_type', 'composites')
            validation_status = df.attrs.get('validation_status', 'unknown')
            logger.info(
                f"LINEAGE: {task_name} accessed {source_type} data. "
                f"Validation: {validation_status}, Rows: {len(df)}"
            )
            
            return df
            
        except ValueError as e:
            # Convert lineage gate failure to user-friendly error
            error_msg = str(e)
            logger.error(f"LINEAGE GATE: {task_name} blocked - {error_msg}")
            self.signals.task_error.emit(task_name, error_msg)
            return None
        except Exception as e:
            # Unexpected error - log and report
            logger.error(f"Unexpected error in _get_estimation_ready_data for {task_name}: {e}", exc_info=True)
            self.signals.task_error.emit(task_name, f"Unexpected error accessing data: {e}")
            return None
    
    def run_simple_kriging(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """
        Run Simple Kriging.
        
        CRITICAL: Uses lineage-enforced data accessor. NEVER uses raw assays.
        """
        # --- LINEAGE-ENFORCED DATA ACCESS ---
        if "data" not in params or params["data"] is None:
            df = self._get_estimation_ready_data("simple_kriging", require_validation=True)
            if df is None:
                return  # Error already emitted
            
            params["data"] = df
            logger.info(f"Controller: Injected validated composites ({len(df)} rows) into Simple Kriging.")
        # -----------------------------------------------
        
        self._geostats.run_simple_kriging(params, callback, progress_callback)
    
    def run_kriging(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """
        Run Ordinary Kriging.
        
        CRITICAL: Uses lineage-enforced data accessor. NEVER uses raw assays.
        """
        # --- LINEAGE-ENFORCED DATA ACCESS ---
        if "data_df" not in params or params["data_df"] is None:
            df = self._get_estimation_ready_data("kriging", require_validation=True)
            if df is None:
                return  # Error already emitted
            
            params["data_df"] = df
            logger.info(f"Controller: Injected validated composites ({len(df)} rows) into Ordinary Kriging.")
        # -----------------------------------------------
        
        self._geostats.run_kriging(params, callback, progress_callback)
    
    def run_universal_kriging(self, config: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """
        Run Universal Kriging.
        
        CRITICAL: Uses lineage-enforced data accessor. NEVER uses raw assays.
        """
        # --- LINEAGE-ENFORCED DATA ACCESS ---
        if "data_df" not in config or config["data_df"] is None:
            df = self._get_estimation_ready_data("universal_kriging", require_validation=True)
            if df is None:
                return  # Error already emitted
            
            config["data_df"] = df
            logger.info(f"Controller: Injected validated composites ({len(df)} rows) into Universal Kriging.")
        # -----------------------------------------------
        
        self._geostats.run_universal_kriging(config, callback, progress_callback)
    
    def run_cokriging(self, config: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """
        Run Co-Kriging.
        
        CRITICAL: Uses lineage-enforced data accessor. NEVER uses raw assays.
        """
        # --- LINEAGE-ENFORCED DATA ACCESS ---
        if "data_df" not in config or config["data_df"] is None:
            df = self._get_estimation_ready_data("cokriging", require_validation=True)
            if df is None:
                return  # Error already emitted
            
            config["data_df"] = df
            logger.info(f"Controller: Injected validated composites ({len(df)} rows) into Co-Kriging.")
        # -----------------------------------------------
        
        self._geostats.run_cokriging(config, callback, progress_callback)
    
    def run_indicator_kriging(self, config: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """
        Run Indicator Kriging.
        
        CRITICAL: Uses lineage-enforced data accessor. NEVER uses raw assays.
        """
        # --- LINEAGE-ENFORCED DATA ACCESS ---
        if "data_df" not in config or config["data_df"] is None:
            df = self._get_estimation_ready_data("indicator_kriging", require_validation=True)
            if df is None:
                return  # Error already emitted
            
            config["data_df"] = df
            logger.info(f"Controller: Injected validated composites ({len(df)} rows) into Indicator Kriging.")
        # -----------------------------------------------
        
        self._geostats.run_indicator_kriging(config, callback, progress_callback)
    
    def run_soft_kriging(self, config: Dict[str, Any], callback=None) -> None:
        """Run Soft/Bayesian Kriging."""
        self._geostats.run_soft_kriging(config, callback)

    def run_rbf_interpolation(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """
        Run RBF Interpolation.

        CRITICAL: Uses lineage-enforced data accessor. NEVER uses raw assays.
        """
        # --- LINEAGE-ENFORCED DATA ACCESS ---
        if "data" not in params or params["data"] is None:
            # RBF can use declustered data if available
            df = self._get_estimation_ready_data("rbf", require_validation=True, prefer_declustered=True)
            if df is None:
                return  # Error already emitted

            params = params.copy()  # Don't modify original
            params["data"] = df
            logger.info(f"Controller: Injected validated data ({len(df)} rows) into RBF Interpolation.")

        # Get grid spec from current block model if available
        if "grid_spec" not in params or not params["grid_spec"]:
            if hasattr(self, 'block_model') and self.block_model is not None:
                bm = self.block_model
                params["grid_spec"] = {
                    "nx": bm.nx, "ny": bm.ny, "nz": bm.nz,
                    "xmin": bm.xmin, "ymin": bm.ymin, "zmin": bm.zmin,
                    "xinc": bm.xinc, "yinc": bm.yinc, "zinc": bm.zinc
                }
            else:
                # Default grid spec
                params["grid_spec"] = {
                    "nx": 50, "ny": 50, "nz": 20,
                    "xmin": 0.0, "ymin": 0.0, "zmin": 0.0,
                    "xinc": 10.0, "yinc": 10.0, "zinc": 5.0
                }

        self._geostats.run_rbf_interpolation(params, callback, progress_callback)
    
    def run_sgsim(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """
        Run SGSIM.
        
        CRITICAL: Uses lineage-enforced data accessor. NEVER uses raw assays.
        """
        # --- LINEAGE-ENFORCED DATA ACCESS ---
        if "data_df" not in params or params["data_df"] is None:
            df = self._get_estimation_ready_data("sgsim", require_validation=True)
            if df is None:
                return  # Error already emitted
            
            params["data_df"] = df
            logger.info(f"Controller: Injected validated composites ({len(df)} rows) into SGSIM.")
        # -----------------------------------------------
        
        self._geostats.run_sgsim(params, callback, progress_callback)
    
    def run_ik_sgsim(self, config: Dict[str, Any], callback=None) -> None:
        """Run IK-based SGSIM."""
        self._geostats.run_ik_sgsim(config, callback)
    
    def run_cosgsim(self, config: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """
        Run Co-Simulation.
        
        CRITICAL: Uses lineage-enforced data accessor. NEVER uses raw assays.
        """
        # --- LINEAGE-ENFORCED DATA ACCESS ---
        if "data_df" not in config or config["data_df"] is None:
            df = self._get_estimation_ready_data("cosgsim", require_validation=True)
            if df is None:
                return  # Error already emitted
            
            config["data_df"] = df
            logger.info(f"Controller: Injected validated composites ({len(df)} rows) into Co-SGSIM.")
        # -----------------------------------------------
        
        self._geostats.run_cosgsim(config, callback, progress_callback)

    def run_sis(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Sequential Indicator Simulation."""
        self._geostats.run_sis(params, callback, progress_callback)

    def run_turning_bands(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Turning Bands Simulation."""
        self._geostats.run_turning_bands(params, callback, progress_callback)

    def run_dbs(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Direct Block Simulation."""
        self._geostats.run_dbs(params, callback, progress_callback)

    def run_grf(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Gaussian Random Field Simulation."""
        self._geostats.run_grf(params, callback, progress_callback)
    
    def run_variogram(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """
        Run Variogram analysis.
        
        CRITICAL: Uses lineage-enforced data accessor. NEVER uses raw assays.
        Variograms should be computed on declustered composites when available.
        """
        # --- LINEAGE-ENFORCED DATA ACCESS ---
        if "data_df" not in params or params["data_df"] is None:
            # Variogram prefers declustered data if available
            df = self._get_estimation_ready_data("variogram", require_validation=True, prefer_declustered=True)
            if df is None:
                return  # Error already emitted
            
            params["data_df"] = df
            source_type = df.attrs.get('source_type', 'composites')
            logger.info(f"Controller: Injected {source_type} data ({len(df)} rows) into Variogram analysis.")
        # -----------------------------------------------
        
        self._geostats.run_variogram(params, callback, progress_callback)
    
    def run_variogram_assistant(self, config: Dict[str, Any], callback=None) -> None:
        """Run Variogram Assistant."""
        self._geostats.run_variogram_assistant(config, callback)
    
    def run_uncertainty(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Uncertainty analysis."""
        self._geostats.run_uncertainty(params, callback, progress_callback)
    
    def run_economic_uncertainty(self, config: Dict[str, Any], callback=None) -> None:
        """Run Economic Uncertainty."""
        self._geostats.run_economic_uncertainty(config, callback)
    
    def run_grade_stats(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Grade Statistics."""
        self._geostats.run_grade_stats(params, callback, progress_callback)
    
    def run_grade_transform(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Grade Transformation."""
        self._geostats.run_grade_transform(params, callback, progress_callback)
    
    def run_swath_analysis(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Swath Analysis."""
        self._geostats.run_swath_analysis(params, callback, progress_callback)
    
    def run_kmeans(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run K-Means Clustering."""
        self._geostats.run_kmeans(params, callback, progress_callback)
    
    # ------------------------------------------------------------------
    # Mining delegates (to MiningController)
    # ------------------------------------------------------------------
    def calculate_resources(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Calculate resources."""
        self._mining.calculate_resources(params, callback, progress_callback)
    
    def classify_resources(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Classify resources."""
        self._mining.classify_resources(params, callback, progress_callback)
    
    def run_drillhole_resources(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run drillhole resources - REMOVED (depended on ResourceCalculator which has been deleted)."""
        raise NotImplementedError("Drillhole resources functionality has been removed (depended on ResourceCalculator).")
    
    def run_irr(self, config: Any, callback=None, progress_callback=None) -> None:
        """Run IRR analysis."""
        self._mining.run_irr(config, callback, progress_callback)
    
    def run_npv(self, config: Any, callback=None, progress_callback=None) -> None:
        """Run NPV analysis."""
        self._mining.run_npv(config, callback, progress_callback)
    
    def run_npvs(self, payload: Dict[str, Any], callback=None) -> None:
        """Run NPVS optimization."""
        self._mining.run_npvs(payload, callback)
    
    def run_pit_optimisation(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run pit optimization."""
        self._mining.run_pit_optimisation(params, callback, progress_callback)
    
    def run_underground_planning(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run underground planning."""
        self._mining.run_underground_planning(params, callback, progress_callback)
    
    def run_esg(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run ESG analysis."""
        self._mining.run_esg(params, callback, progress_callback)
    
    # Scheduling
    def run_strategic_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run strategic schedule."""
        self._mining.run_strategic_schedule(config, callback)
    
    def run_nested_shell_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run nested shell schedule."""
        self._mining.run_nested_shell_schedule(config, callback)
    
    def run_cutoff_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run cutoff schedule."""
        self._mining.run_cutoff_schedule(config, callback)
    
    def run_tactical_pushback_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run tactical pushback schedule."""
        self._mining.run_tactical_pushback_schedule(config, callback)
    
    def run_tactical_bench_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run tactical bench schedule."""
        self._mining.run_tactical_bench_schedule(config, callback)
    
    def run_tactical_dev_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run tactical development schedule."""
        self._mining.run_tactical_dev_schedule(config, callback)
    
    def run_short_term_digline_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run short-term digline schedule."""
        self._mining.run_short_term_digline_schedule(config, callback)
    
    def run_short_term_blend(self, config: Dict[str, Any], callback=None) -> None:
        """Run short-term blend."""
        self._mining.run_short_term_blend(config, callback)
    
    def run_shift_plan(self, config: Dict[str, Any], callback=None) -> None:
        """Run shift plan."""
        self._mining.run_shift_plan(config, callback)
    
    # Fleet & Haulage
    def compute_fleet_cycle_time(self, config: Dict[str, Any], callback=None) -> None:
        """Compute fleet cycle time."""
        self._mining.compute_fleet_cycle_time(config, callback)
    
    def run_fleet_dispatch(self, config: Dict[str, Any], callback=None) -> None:
        """Run fleet dispatch."""
        self._mining.run_fleet_dispatch(config, callback)
    
    def evaluate_haulage(self, config: Dict[str, Any], callback=None) -> None:
        """Evaluate haulage capacity."""
        self._mining.evaluate_haulage(config, callback)
    
    # Grade Control & Reconciliation
    def build_gc_support_model(self, config: Dict[str, Any], callback=None) -> None:
        """Build GC support model."""
        self._mining.build_gc_support_model(config, callback)
    
    def run_gc_ok(self, config: Dict[str, Any], callback=None) -> None:
        """Run GC OK."""
        self._mining.run_gc_ok(config, callback)
    
    def run_gc_sgsim(self, config: Dict[str, Any], callback=None) -> None:
        """Run GC SGSIM."""
        self._mining.run_gc_sgsim(config, callback)
    
    def classify_gc_ore_waste(self, config: Dict[str, Any], callback=None) -> None:
        """Classify GC ore/waste."""
        self._mining.classify_gc_ore_waste(config, callback)
    
    def summarise_gc_by_digpolygon(self, config: Dict[str, Any], callback=None) -> None:
        """Summarize GC by dig polygon."""
        self._mining.summarise_gc_by_digpolygon(config, callback)
    
    def run_recon_model_mine(self, config: Dict[str, Any], callback=None) -> None:
        """Run model-mine reconciliation."""
        self._mining.run_recon_model_mine(config, callback)
    
    def run_recon_mine_mill(self, config: Dict[str, Any], callback=None) -> None:
        """Run mine-mill reconciliation."""
        self._mining.run_recon_mine_mill(config, callback)
    
    def run_recon_metrics(self, config: Dict[str, Any], callback=None) -> None:
        """Run reconciliation metrics."""
        self._mining.run_recon_metrics(config, callback)
    
    # Underground
    def ug_generate_slos_stopes(self, config: Dict[str, Any], callback=None) -> None:
        """Generate SLOS stopes."""
        self._mining.ug_generate_slos_stopes(config, callback)
    
    def ug_run_slos_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run SLOS schedule."""
        self._mining.ug_run_slos_schedule(config, callback)
    
    def ug_build_cave_footprint(self, config: Dict[str, Any], callback=None) -> None:
        """Build cave footprint."""
        self._mining.ug_build_cave_footprint(config, callback)
    
    def ug_run_cave_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run cave schedule."""
        self._mining.ug_run_cave_schedule(config, callback)
    
    def ug_apply_dilution(self, config: Dict[str, Any], callback=None) -> None:
        """Apply dilution."""
        self._mining.ug_apply_dilution(config, callback)
    
    # Geometallurgy
    def assign_geomet_ore_types(self, config: Dict[str, Any], callback=None) -> None:
        """Assign geomet ore types."""
        self._mining.assign_geomet_ore_types(config, callback)
    
    def compute_geomet_block_attributes(self, config: Dict[str, Any], callback=None) -> None:
        """Compute geomet block attributes."""
        self._mining.compute_geomet_block_attributes(config, callback)
    
    def evaluate_geomet_plant_response(self, config: Dict[str, Any], callback=None) -> None:
        """Evaluate geomet plant response."""
        self._mining.evaluate_geomet_plant_response(config, callback)
    
    def run_geomet_chain(self, config: Dict[str, Any], callback=None) -> None:
        """Run geomet chain."""
        self._mining.run_geomet_chain(config, callback)
    
    # Scenario management
    def create_scenario(self, scenario: Any) -> None:
        """Create a scenario."""
        self._mining.create_scenario(scenario)
    
    def list_scenarios(self) -> list:
        """List scenarios."""
        return self._mining.list_scenarios()
    
    def run_scenario(self, name: str, version: str, callback=None) -> None:
        """Run a scenario."""
        self._mining.run_scenario(name, version, callback)
    
    def compare_scenarios(self, scenario_ids: list, callback=None) -> None:
        """Compare scenarios."""
        self._mining.compare_scenarios(scenario_ids, callback)
    
    def create_scenario_from_context(self, context: Dict[str, Any]) -> Any:
        """Build a PlanningScenario from current panel settings."""
        from ..planning.scenario_definition import PlanningScenario, ScenarioID, ScenarioInputs
        from datetime import datetime
        
        name = context.get("name", f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        version = context.get("version", "draft")
        
        scenario_id = ScenarioID(name=name, version=version)
        
        inputs = ScenarioInputs(
            model_name=context.get("model_name", "default"),
            value_mode=context.get("value_mode", "base"),
            value_field=context.get("value_field", "block_value"),
            pit_config=context.get("pit_config"),
            schedule_config=context.get("schedule_config"),
            cutoff_config=context.get("cutoff_config"),
            geomet_config=context.get("geomet_config"),
            gc_config=context.get("gc_config"),
            risk_config=context.get("risk_config"),
            esg_config=context.get("esg_config"),
            npvs_config=context.get("npvs_config"),
        )
        
        scenario = PlanningScenario(
            id=scenario_id,
            description=context.get("description", ""),
            tags=context.get("tags", []),
            inputs=inputs,
            status="new"
        )
        
        self.scenario_store.save(scenario)
        return scenario
    
    def build_pushback_plan(self, config: Dict[str, Any], callback=None) -> None:
        """Build pushback plan."""
        self._mining.build_pushback_plan(config, callback)
    
    def optimise_cutoff_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Optimise cutoff schedule."""
        self._mining.optimise_cutoff_schedule(config, callback)
    
    def align_production(self, config: Dict[str, Any], callback=None) -> None:
        """Align production."""
        self._mining.align_production(config, callback)
    
    # ------------------------------------------------------------------
    # Data delegates (to DataController)
    # ------------------------------------------------------------------
    def load_drillholes(self, config: Dict[str, Any], callback=None) -> None:
        """Load drillholes."""
        self._data.load_drillholes(config, callback)
    
    def run_drillhole_qaqc(self, db_config: Dict[str, Any], callback=None) -> None:
        """Run drillhole QAQC."""
        self._data.run_drillhole_qaqc(db_config, callback)
    
    def run_implicit_geology(self, config: Dict[str, Any], callback=None) -> None:
        """Run implicit geology."""
        self._data.run_implicit_geology(config, callback)
    
    def build_wireframes(self, config: Dict[str, Any], callback=None) -> None:
        """Build wireframes."""
        self._data.build_wireframes(config, callback)
    
    def run_structural_analysis(self, config: Dict[str, Any], callback=None) -> None:
        """Run structural analysis."""
        self._data.run_structural_analysis(config, callback)
    
    def run_slope_lem_2d(self, config: Dict[str, Any], callback=None) -> None:
        """Run 2D slope LEM."""
        self._data.run_slope_lem_2d(config, callback)
    
    def run_slope_lem_3d(self, config: Dict[str, Any], callback=None) -> None:
        """Run 3D slope LEM."""
        self._data.run_slope_lem_3d(config, callback)
    
    def run_slope_probabilistic(self, config: Dict[str, Any], callback=None) -> None:
        """Run probabilistic slope analysis."""
        self._data.run_slope_probabilistic(config, callback)
    
    def suggest_bench_design(self, config: Dict[str, Any], callback=None) -> None:
        """Suggest bench design."""
        self._data.suggest_bench_design(config, callback)
    
    def run_research_grid(self, grid_config: Dict[str, Any], callback=None) -> None:
        """Run research grid."""
        self._data.run_research_grid(grid_config, callback)
    
    # ------------------------------------------------------------------
    # Workflow Wizard helpers
    # ------------------------------------------------------------------
    def get_available_block_models(self) -> List[str]:
        """Get available block model names."""
        if self.block_model:
            return [self.block_model.name if hasattr(self.block_model, 'name') else "Current Model"]
        return []
    
    def get_available_scenarios(self) -> List[str]:
        """Get available scenario names."""
        if hasattr(self, 'scenario_store') and self.scenario_store:
            try:
                scenarios = self.scenario_store.list_scenarios()
                return [s.name if hasattr(s, 'name') else str(s.id) for s in scenarios]
            except Exception:
                return []
        return []
    
    def get_last_npvs_result(self):
        """Get last NPVS result."""
        return getattr(self, '_last_npvs_result', None)
    
    def get_last_gc_result(self):
        """Get last GC result."""
        return getattr(self, '_last_gc_result', None)
    
    def get_current_scenario(self):
        """Get current scenario."""
        if hasattr(self, 'scenario_store') and self.scenario_store:
            try:
                return getattr(self.scenario_store, 'current_scenario', None)
            except Exception:
                return None
        return None
    
    # ------------------------------------------------------------------
    # Legacy/utility methods
    # ------------------------------------------------------------------
    def request_task(self, name: str, **kwargs):
        """Placeholder async-task hook for panels."""
        logger.debug("Task requested '%s' (no dispatcher wired yet) kwargs=%s", name, kwargs)
    
    def _apply_visual_prefs(self):
        """Apply stored visual preferences."""
        self._vis._apply_visual_prefs()
    
    def _update_scalar_bar(self, title: str):
        """Update legend with current property."""
        self._vis._update_scalar_bar(title)
    
    def run_analysis_task(self, task: str, params: Dict[str, Any], callback=None, progress_callback=None):
        """Run analysis task (alias for run_task)."""
        self.run_task(task, params, callback, progress_callback)
