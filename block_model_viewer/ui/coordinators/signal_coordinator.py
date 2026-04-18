"""
Signal Coordinator — owns all signal wiring and handlers.

Extracted from MainWindow. Connects UI signals to controller,
controller signals to UI handlers, and owns the handler methods
for scene updates, task lifecycle, and state changes.

Usage:
    # In MainWindow.__init__:
    self._signal_coordinator = SignalCoordinator(self)

    # In _connect_signals:
    self._signal_coordinator.connect_all()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class SignalCoordinator(QObject):
    """
    Owns all signal wiring between UI components and the controller.

    Connects:
    1. UISignals → AppController methods
    2. ControllerSignals → UI handler methods
    3. DataRegistry signals → UI updates
    4. Panel-specific signals (cross-section, drillhole, legend)
    """

    def __init__(self, main_window: 'MainWindow'):
        super().__init__(main_window)
        self._mw = main_window

    # ═══════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════════

    def connect_all(self):
        """
        Wire all signals. Called once during MainWindow initialization.
        """
        mw = self._mw

        # 1. UI signals → Controller
        self._connect_ui_to_controller()

        # 2. Controller signals → UI handlers
        self._connect_controller_to_ui()

        # 3. DataRegistry signals
        self._connect_registry_signals()

        # 4. Panel-specific signals
        self._connect_panel_signals()

        logger.info("SignalCoordinator: all signals connected")

    # ═══════════════════════════════════════════════════════════════
    # UI → CONTROLLER
    # ═══════════════════════════════════════════════════════════════

    def _connect_ui_to_controller(self):
        """Connect UI signals to AppController methods."""
        mw = self._mw
        if not mw.signals or not mw.controller:
            logger.warning("Signals or controller not initialized, skipping UI→Controller connections")
            return

        mw.signals.propertySelected.connect(mw.controller.set_active_property)
        mw.signals.colormapChanged.connect(mw.controller.set_colormap)
        mw.signals.sliceChanged.connect(mw.controller.apply_slice)
        mw.signals.applyFilters.connect(mw.controller.apply_filters)
        mw.signals.exportScreenshot.connect(mw.controller.export_screenshot)
        mw.signals.blockModelLoaded.connect(mw.controller.load_block_model)

        logger.info("Connected UI signals to AppController")

    # ═══════════════════════════════════════════════════════════════
    # CONTROLLER → UI
    # ═══════════════════════════════════════════════════════════════

    def _connect_controller_to_ui(self):
        """Connect controller signals to UI handler methods."""
        mw = self._mw
        if not mw.controller:
            return

        signals = mw.controller.signals

        # Scene updates
        signals.scene_updated.connect(self._on_scene_updated)
        signals.block_model_changed.connect(self._on_block_model_changed)

        # Task lifecycle
        signals.task_started.connect(self._on_task_started)
        signals.task_finished.connect(self._on_task_finished)
        signals.task_error.connect(self._on_task_error)
        signals.task_progress.connect(self._on_task_progress)

        # App state
        signals.app_state_changed.connect(self._on_app_state_changed)

        logger.info("Connected controller signals to UI handlers")

    # ═══════════════════════════════════════════════════════════════
    # REGISTRY SIGNALS
    # ═══════════════════════════════════════════════════════════════

    def _connect_registry_signals(self):
        """Connect DataRegistry signals to UI updates.

        NOTE: blockModelLoaded is connected in dock_setup.py to
        main_window._on_block_model_loaded_from_registry.
        Do NOT duplicate that connection here.
        """
        pass

    # ═══════════════════════════════════════════════════════════════
    # PANEL-SPECIFIC SIGNALS
    # ═══════════════════════════════════════════════════════════════

    def _connect_panel_signals(self):
        """Connect panel-specific signals (cross-section, drillhole, etc.)."""
        mw = self._mw

        # Renderer layer changes
        try:
            if mw.viewer_widget and hasattr(mw.viewer_widget, 'renderer'):
                renderer = mw.viewer_widget.renderer
                if hasattr(renderer, 'layers_changed'):
                    renderer.layers_changed.connect(mw._on_renderer_layers_changed)
                if hasattr(renderer, 'drillhole_interval_selected'):
                    renderer.drillhole_interval_selected.connect(
                        self._on_drillhole_interval_selected
                    )
        except Exception as e:
            logger.debug(f"Could not connect renderer signals: {e}")

        # Property panel — layer switching via cached block model grids
        # CRITICAL: Without this, switching layers in the property panel dropdown
        # fails silently for cached SGSIM/Kriging/Classification layers.
        try:
            if hasattr(mw, 'property_panel') and mw.property_panel is not None:
                pp = mw.property_panel
                if hasattr(pp, 'request_visualization'):
                    pp.request_visualization.connect(
                        mw._handle_property_panel_visualization_request
                    )
                    logger.info("Connected property_panel.request_visualization signal to handler")
                # CRITICAL (from March Pictures): without these, changing
                # property/colormap in the panel doesn't repaint the 3D view.
                if hasattr(pp, 'property_changed') and hasattr(mw, 'on_property_changed'):
                    pp.property_changed.connect(mw.on_property_changed)
                if hasattr(pp, 'colormap_changed') and hasattr(mw, 'on_colormap_changed'):
                    pp.colormap_changed.connect(mw.on_colormap_changed)
                # Bind renderer for legend updates
                if hasattr(pp, 'set_renderer') and mw.viewer_widget:
                    try:
                        pp.set_renderer(mw.viewer_widget.renderer)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Could not connect property panel signal: {e}")

        # Scene inspector signals — reset-view / view-preset / projection /
        # scalar-bar toggle.  These are fully wired in the March (Pictures)
        # snapshot but completely absent from April (Documents); clicking
        # those buttons in the panel produces no effect without this block.
        try:
            if hasattr(mw, 'scene_inspector_panel') and mw.scene_inspector_panel is not None:
                si = mw.scene_inspector_panel
                if hasattr(si, 'reset_view_requested') and hasattr(mw, 'reset_camera'):
                    si.reset_view_requested.connect(mw.reset_camera)
                if hasattr(si, 'view_preset_requested') and hasattr(mw, 'set_view_preset'):
                    si.view_preset_requested.connect(mw.set_view_preset)
                if hasattr(si, 'projection_toggled') and hasattr(mw, 'on_projection_toggled'):
                    si.projection_toggled.connect(mw.on_projection_toggled)
                if hasattr(si, 'scalar_bar_toggled') and hasattr(mw, 'on_scalar_bar_toggled'):
                    si.scalar_bar_toggled.connect(mw.on_scalar_bar_toggled)
                logger.info("Connected scene_inspector_panel signals (reset/preset/projection/scalar bar)")
        except Exception as e:
            logger.debug(f"Could not connect scene inspector signals: {e}")

    # ═══════════════════════════════════════════════════════════════
    # HANDLER METHODS
    # ═══════════════════════════════════════════════════════════════

    def _on_scene_updated(self):
        """Handle scene update signal."""
        mw = self._mw
        if mw.viewer_widget:
            try:
                mw.viewer_widget.update()
            except Exception as exc:
                logger.debug(f"Failed to update viewer widget: {exc}")

    def _on_block_model_changed(self):
        """Handle block model changed signal."""
        mw = self._mw
        if hasattr(mw, 'property_panel') and mw.property_panel:
            try:
                mw.property_panel.refresh()
            except Exception as exc:
                logger.debug(f"Failed to refresh property panel: {exc}")

    def _on_task_started(self, task: str):
        """Handle task started signal."""
        logger.debug(f"Task '{task}' started")
        mw = self._mw
        if mw.status is not None:
            mw.status.show_message(f"Running: {task}...", 0)

    def _on_task_finished(self, task: str):
        """Handle task finished signal."""
        logger.debug(f"Task '{task}' finished")
        mw = self._mw
        if mw.status is not None:
            mw.status.show_message(f"Completed: {task}", 3000)

    def _on_task_error(self, task: str, error_msg: str):
        """Handle task error signal."""
        logger.error(f"Task '{task}' error: {error_msg}")
        QMessageBox.critical(
            self._mw, f"Task Error: {task}",
            f"The task '{task}' encountered an error:\n\n{error_msg}"
        )

    def _on_task_progress(self, task: str, progress: float):
        """Handle task progress signal."""
        logger.debug(f"Task '{task}' progress: {progress:.1f}%")

    def _on_app_state_changed(self, state: int):
        """
        Handle application state changes — propagate to all UI panels.
        Panels must NOT infer state from data presence.
        """
        from ...controllers.app_state import AppState

        mw = self._mw
        try:
            new_state = AppState(state)
            logger.info(f"App state → {new_state.name}")
        except ValueError:
            logger.warning(f"Invalid app state value: {state}")
            return

        # Propagate to panels that support state changes
        panels_to_update = [
            ('property_panel', mw.property_panel),
            ('scene_inspector_panel', getattr(mw, 'scene_inspector_panel', None)),
            ('drillhole_control_panel', getattr(mw, 'drillhole_control_panel', None)),
        ]
        for panel_name, panel in panels_to_update:
            if panel and hasattr(panel, 'on_app_state_changed'):
                try:
                    panel.on_app_state_changed(state)
                except Exception as e:
                    logger.debug(f"Failed to update {panel_name} state: {e}")

        # Update legend widget
        try:
            if mw.viewer_widget and hasattr(mw.viewer_widget, 'renderer'):
                renderer = mw.viewer_widget.renderer
                if hasattr(renderer, 'legend_manager') and renderer.legend_manager:
                    legend_widget = getattr(renderer.legend_manager, 'widget', None)
                    if legend_widget and hasattr(legend_widget, 'on_app_state_changed'):
                        legend_widget.on_app_state_changed(state)
        except Exception as e:
            logger.debug(f"Failed to update legend widget state: {e}")

        # Update status bar
        self._update_status_for_state(new_state)

    def _update_status_for_state(self, state):
        """Update status bar message based on app state."""
        from ...controllers.app_state import AppState

        messages = {
            AppState.EMPTY: "No file loaded",
            AppState.DATA_LOADED: "Data loaded - ready for visualization",
            AppState.RENDERED: "Ready",
            AppState.BUSY: "Processing...",
        }
        message = messages.get(state, "")
        if message and self._mw.status is not None:
            self._mw.status.show_message(message, 3000)

    def _on_drillhole_interval_selected(self, data: dict):
        """Handle drillhole interval selection from renderer."""
        logger.debug(f"Drillhole interval selected: {data}")
        try:
            hole_id = data.get("hole_id")
            if hole_id and hasattr(self._mw, 'drillhole_control_panel'):
                pass  # Could highlight in drillhole panel
        except Exception as e:
            logger.warning(f"Error handling drillhole selection: {e}")

    # ═══════════════════════════════════════════════════════════════
    # GENERIC PANEL SIGNAL HANDLERS
    # ═══════════════════════════════════════════════════════════════
    # These are reusable handlers that any panel can connect its signals
    # to.  See panel_mixin.open_*_panel() methods for where each handler
    # gets wired to a specific dialog instance.

    def on_panel_progress(self, percent: int, message: str = ""):
        """Generic progress handler for any panel emitting progress_updated(int, str)."""
        mw = self._mw
        if mw.status is not None:
            try:
                fraction = max(0.0, min(1.0, float(percent) / 100.0))
                mw.status.update_progress(message or f"Working... {percent}%", fraction)
            except Exception:
                pass

    def on_filters_changed(self, filter_state: dict):
        """Handler for block_model_filter_panel.filtersChanged.

        Applies the filter state to the renderer so filtered blocks become invisible.
        """
        mw = self._mw
        try:
            if mw.viewer_widget and mw.viewer_widget.renderer:
                renderer = mw.viewer_widget.renderer
                if hasattr(renderer, 'apply_block_filters'):
                    renderer.apply_block_filters(filter_state)
                elif hasattr(renderer, 'apply_filters'):
                    renderer.apply_filters(filter_state)
                else:
                    logger.debug("Renderer has no apply_block_filters/apply_filters method")
        except Exception as e:
            logger.warning(f"Failed to apply block filters: {e}")

    def on_section_updated(self, section_data):
        """Handler for cross_section_panel.section_updated."""
        mw = self._mw
        try:
            if mw.viewer_widget and mw.viewer_widget.renderer:
                renderer = mw.viewer_widget.renderer
                if hasattr(renderer, 'refresh_cross_section'):
                    renderer.refresh_cross_section(section_data)
        except Exception as e:
            logger.warning(f"Failed to refresh cross-section: {e}")

    def on_slicer_range_changed(self, axis: str, lo: float, hi: float):
        """Handler for interactive_slicer_panel.rangeChanged."""
        mw = self._mw
        try:
            if mw.viewer_widget:
                if hasattr(mw.viewer_widget, 'apply_spatial_slice'):
                    mw.viewer_widget.apply_spatial_slice(axis, lo, hi)
        except Exception as e:
            logger.warning(f"Failed to apply slice range: {e}")

    def on_slicer_clipping_changed(self, clip_state: dict):
        """Handler for interactive_slicer_panel.clipping_changed."""
        mw = self._mw
        try:
            if mw.viewer_widget and mw.viewer_widget.renderer:
                renderer = mw.viewer_widget.renderer
                if hasattr(renderer, 'apply_clip_state'):
                    renderer.apply_clip_state(clip_state)
        except Exception as e:
            logger.warning(f"Failed to apply clip state: {e}")

    def on_domain_codes_assigned(self, codes):
        """Handler for lithology_manager_panel.domainCodesAssigned."""
        mw = self._mw
        try:
            registry = mw.controller.registry if mw.controller else mw._registry
            if registry and hasattr(registry, 'register_domain_codes'):
                registry.register_domain_codes(codes, source_panel="LithologyManager")
            logger.info(f"Domain codes assigned: {len(codes) if hasattr(codes, '__len__') else '?'} entries")
        except Exception as e:
            logger.warning(f"Failed to register domain codes: {e}")

    def on_contacts_extracted(self, contacts):
        """Handler for lithology_manager_panel.contactsExtracted."""
        mw = self._mw
        try:
            registry = mw.controller.registry if mw.controller else mw._registry
            if registry and hasattr(registry, 'register_contact_set'):
                registry.register_contact_set(contacts, source_panel="LithologyManager")
            logger.info("Contacts extracted and registered")
        except Exception as e:
            logger.warning(f"Failed to register contacts: {e}")

    def on_frag_import_completed(self, result):
        """Handler for frag_import_panel.import_completed."""
        logger.info("Fragmentation import completed")
        mw = self._mw
        if mw.status is not None:
            mw.status.show_message("Fragmentation import complete", 3000)

    def on_frag_preprocessing_completed(self, result):
        """Handler for frag_preprocessing_panel.preprocessing_completed."""
        logger.info("Fragmentation preprocessing completed")
        mw = self._mw
        if mw.status is not None:
            mw.status.show_message("Fragmentation preprocessing complete", 3000)

    def on_frag_segmentation_completed(self, result):
        """Handler for frag_segmentation_panel.segmentation_completed."""
        logger.info("Fragmentation segmentation completed")
        mw = self._mw
        if mw.status is not None:
            mw.status.show_message("Fragmentation segmentation complete", 3000)

    def on_fragment_selected(self, fragment_id):
        """Handler for frag_results_panel.fragment_selected."""
        logger.info(f"Fragment selected: {fragment_id}")
        mw = self._mw
        try:
            if mw.viewer_widget and mw.viewer_widget.renderer:
                renderer = mw.viewer_widget.renderer
                if hasattr(renderer, 'highlight_fragment'):
                    renderer.highlight_fragment(fragment_id)
        except Exception as e:
            logger.debug(f"Failed to highlight fragment: {e}")

    # ═══════════════════════════════════════════════════════════════
    # PANEL SIGNAL BINDER
    # ═══════════════════════════════════════════════════════════════

    def wire_panel_dialog_signals(self, dialog):
        """Inspect a dialog for known signals and connect them to handlers.

        Idempotent: always disconnects before reconnecting, so it's safe to call
        this every time a panel dialog is opened.  Called from panel_mixin.open_*
        methods.
        """
        if dialog is None:
            return

        # Map of signal_name → handler (we connect only if the dialog emits it)
        signal_map = {
            'progress_updated': self.on_panel_progress,
            'filtersChanged': self.on_filters_changed,
            'section_updated': self.on_section_updated,
            'rangeChanged': self.on_slicer_range_changed,
            'clipping_changed': self.on_slicer_clipping_changed,
            'domainCodesAssigned': self.on_domain_codes_assigned,
            'contactsExtracted': self.on_contacts_extracted,
            'import_completed': self.on_frag_import_completed,
            'preprocessing_completed': self.on_frag_preprocessing_completed,
            'segmentation_completed': self.on_frag_segmentation_completed,
            'fragment_selected': self.on_fragment_selected,
        }

        for signal_name, handler in signal_map.items():
            sig = getattr(dialog, signal_name, None)
            if sig is None:
                continue
            # Only treat it as a signal if it has .connect()
            if not hasattr(sig, 'connect'):
                continue
            try:
                sig.disconnect(handler)
            except (TypeError, RuntimeError):
                pass
            try:
                sig.connect(handler)
                logger.debug(f"Connected {dialog.__class__.__name__}.{signal_name} → {handler.__name__}")
            except Exception as e:
                logger.debug(f"Could not connect {signal_name}: {e}")
