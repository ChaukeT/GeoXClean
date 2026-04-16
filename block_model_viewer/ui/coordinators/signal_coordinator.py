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
                if hasattr(mw.property_panel, 'request_visualization'):
                    mw.property_panel.request_visualization.connect(
                        mw._handle_property_panel_visualization_request
                    )
                    logger.info("Connected property_panel.request_visualization signal to handler")
        except Exception as e:
            logger.debug(f"Could not connect property panel signal: {e}")

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
