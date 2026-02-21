"""
StatusManager - Centralized status bar setup and updates for GeoX.

Responsibilities:
- Create and configure QStatusBar widgets
- Manage status update timer
- Update labels and progress indicator
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QStatusBar, QLabel, QProgressBar, QMainWindow

from ..workflow_status_widget import WorkflowStatusWidget

logger = logging.getLogger(__name__)


class StatusManager(QObject):
    """Manages status bar widgets and updates."""

    def __init__(self, main_window: QMainWindow, parent: Optional[QObject] = None):
        super().__init__(parent or main_window)
        self._main_window = main_window
        self._status_bar: Optional[QStatusBar] = None
        self._file_label: Optional[QLabel] = None
        self._blocks_label: Optional[QLabel] = None
        self._property_label: Optional[QLabel] = None
        self._camera_label: Optional[QLabel] = None
        self._lighting_label: Optional[QLabel] = None
        self._status_progress: Optional[QProgressBar] = None
        self._status_update_timer: Optional[QTimer] = None
        self._workflow_widget: Optional[WorkflowStatusWidget] = None

    @property
    def status_bar(self) -> Optional[QStatusBar]:
        return self._status_bar

    def setup(self) -> None:
        """Create and configure the status bar and widgets."""
        self._status_bar = QStatusBar()
        self._main_window.setStatusBar(self._status_bar)

        # Create status labels
        self._file_label = QLabel("No file loaded")
        self._blocks_label = QLabel("Blocks: 0")
        self._property_label = QLabel("Property: None")
        self._camera_label = QLabel("Camera: Perspective")
        self._lighting_label = QLabel("Lighting: On")

        # Add permanent widgets to status bar
        self._status_bar.addWidget(self._file_label, 2)
        self._status_bar.addWidget(self._blocks_label, 1)
        self._status_bar.addWidget(self._property_label, 1)
        self._status_bar.addWidget(self._camera_label, 1)
        self._status_bar.addWidget(self._lighting_label, 1)

        self._status_progress = QProgressBar()
        self._status_progress.setRange(0, 100)
        self._status_progress.setFixedWidth(140)
        self._status_progress.setTextVisible(False)
        self._status_progress.hide()
        self._status_bar.addPermanentWidget(self._status_progress)

        # Add workflow status indicator (geostatistics workflow progress)
        self._workflow_widget = WorkflowStatusWidget()
        self._workflow_widget.stage_clicked.connect(self._on_workflow_stage_clicked)
        self._status_bar.addPermanentWidget(self._workflow_widget)

        # Connect to registry if available
        registry = getattr(self._main_window, '_registry', None)
        if registry:
            self._workflow_widget.connect_registry(registry)

        # Setup status update timer
        self._status_update_timer = QTimer(self._main_window)
        self._status_update_timer.timeout.connect(self.update_status_bar)
        self._status_update_timer.start(1000)  # Update every second

        logger.info("Setup enhanced status bar")

    def stop(self) -> None:
        """Stop the status update timer."""
        if self._status_update_timer is not None:
            self._status_update_timer.stop()

    def update_status_bar(self) -> None:
        """Update status bar labels based on current app state."""
        try:
            # File info
            if getattr(self._main_window, "current_file_path", None):
                self._file_label.setText(f"File: {self._main_window.current_file_path.name}")
            else:
                self._file_label.setText("No file loaded")

            # Block count
            if getattr(self._main_window, "current_model", None):
                block_count = len(self._main_window.current_model.positions)
                self._blocks_label.setText(f"Blocks: {block_count:,}")
            else:
                self._blocks_label.setText("Blocks: 0")

            # Current property
            prop_panel = getattr(self._main_window, "property_panel", None)
            if prop_panel and getattr(prop_panel, "current_property", None):
                self._property_label.setText(f"Property: {prop_panel.current_property}")
            else:
                self._property_label.setText("Property: None")

            # Camera mode
            projection_action = getattr(self._main_window, "projection_action", None)
            is_ortho = projection_action.isChecked() if projection_action else False
            mode = "Orthographic" if is_ortho else "Perspective"
            self._camera_label.setText(f"Camera: {mode}")

            # Lighting status
            viewer_widget = getattr(self._main_window, "viewer_widget", None)
            if viewer_widget and getattr(viewer_widget, "renderer", None):
                self._lighting_label.setText("Lighting: On")
            else:
                self._lighting_label.setText("Lighting: -")

        except Exception as e:
            logger.debug(f"Status bar update error: {e}")

    def show_message(self, message: str, timeout: int = 3000) -> None:
        """Show a transient status message."""
        if self._status_bar is not None:
            try:
                self._status_bar.showMessage(message, timeout)
            except Exception:
                pass

    def update_progress(self, message: str, fraction: Optional[float] = None) -> None:
        """Update status bar message with optional progress."""
        if self._status_progress is None or self._status_bar is None:
            return
        if fraction is not None:
            value = max(0, min(100, int(fraction * 100)))
            self._status_progress.show()
            self._status_progress.setValue(value)
        self._status_bar.showMessage(message)

    def finish_progress(self, message: str, timeout: int = 3000) -> None:
        """Finish a status task and hide progress indicator."""
        if self._status_progress is not None:
            self._status_progress.hide()
        self.show_message(message, timeout)

    def _on_workflow_stage_clicked(self, stage_id: str):
        """Handle workflow stage click - open relevant panel."""
        # Map stage IDs to panel opening methods
        stage_panels = {
            'drillholes': 'open_drillhole_import_panel',
            'compositing': 'open_compositing_panel',
            'variogram': 'open_variogram_panel',
            'estimation': 'open_kriging_panel',
        }

        method_name = stage_panels.get(stage_id)
        if method_name and hasattr(self._main_window, method_name):
            try:
                getattr(self._main_window, method_name)()
                logger.info(f"Opened panel for workflow stage: {stage_id}")
            except Exception as e:
                logger.warning(f"Failed to open panel for {stage_id}: {e}")

    def is_progress_visible(self) -> bool:
        """Return True if progress indicator is visible."""
        return bool(self._status_progress and self._status_progress.isVisible())

    def set_camera_mode(self, mode: str) -> None:
        """Update camera label text."""
        if self._camera_label is not None:
            self._camera_label.setText(f"Camera: {mode}")

