"""
Shared scaffolding for analysis panels.

Provides common UI hooks, validation framework, and asynchronous execution
plumbing so specialised analysis dialogs (kriging, SGSIM, etc.) can focus on
domain-specific parameters while delegating execution to the AppController.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtWidgets import (
    QApplication, QProgressDialog, QHBoxLayout, QPushButton,
    QScrollArea, QWidget, QVBoxLayout
)

from .base_panel import BaseDockPanel

logger = logging.getLogger(__name__)


def _sanitize_params_for_log(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize parameters for logging (avoid logging huge dataframes, arrays, etc).
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Sanitized parameter dictionary safe for logging
    """
    sanitized = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            sanitized[key] = value
        elif isinstance(value, (list, tuple)) and len(value) < 10:
            sanitized[key] = value
        elif isinstance(value, dict) and len(value) < 10:
            sanitized[key] = _sanitize_params_for_log(value)
        elif hasattr(value, '__len__'):
            sanitized[key] = f"<{type(value).__name__}(length={len(value)})>"
        else:
            sanitized[key] = f"<{type(value).__name__}>"
    return sanitized


def log_registry_data_status(panel_name: str, data: Optional[Dict[str, Any]]) -> str:
    """
    Log diagnostic information about registry data contents.
    
    Args:
        panel_name: Name of the panel requesting data
        data: Registry drillhole data dictionary
        
    Returns:
        String describing the data source that will be used
    """
    if data is None:
        logger.warning(f"[{panel_name}] Registry data is None - no drillhole data loaded")
        return "none"
    
    if not isinstance(data, dict):
        logger.info(f"[{panel_name}] Registry data is DataFrame directly ({len(data) if hasattr(data, '__len__') else '?'} rows)")
        return "direct_dataframe"
    
    # Log all available keys
    keys = list(data.keys())
    logger.debug(f"[{panel_name}] Registry keys: {keys}")
    
    # Check composites
    composites = data.get('composites')
    composites_df = data.get('composites_df')
    comp_count = 0
    comp_source = None
    
    if isinstance(composites, pd.DataFrame) and not composites.empty:
        comp_count = len(composites)
        comp_source = 'composites'
    elif isinstance(composites_df, pd.DataFrame) and not composites_df.empty:
        comp_count = len(composites_df)
        comp_source = 'composites_df'
    
    # Check assays
    assays = data.get('assays')
    assays_df = data.get('assays_df')
    assay_count = 0
    assay_source = None
    
    if isinstance(assays, pd.DataFrame) and not assays.empty:
        assay_count = len(assays)
        assay_source = 'assays'
    elif isinstance(assays_df, pd.DataFrame) and not assays_df.empty:
        assay_count = len(assays_df)
        assay_source = 'assays_df'
    
    # Log status
    status_parts = []
    if comp_count > 0:
        status_parts.append(f"COMPOSITES: {comp_count} rows (key: '{comp_source}')")
    else:
        status_parts.append("COMPOSITES: None/Empty")
        
    if assay_count > 0:
        status_parts.append(f"RAW ASSAYS: {assay_count} rows (key: '{assay_source}')")
    else:
        status_parts.append("RAW ASSAYS: None/Empty")
    
    logger.info(f"[{panel_name}] Registry Data: {' | '.join(status_parts)}")
    
    # Determine what will be used
    if comp_count > 0:
        logger.info(f"[{panel_name}] → Will use COMPOSITES ({comp_count} samples)")
        return "composites"
    elif assay_count > 0:
        logger.warning(f"[{panel_name}] → Will use RAW ASSAYS ({assay_count} samples) - No composites available!")
        return "raw_assays"
    else:
        logger.error(f"[{panel_name}] → NO DATA AVAILABLE - Both composites and assays are empty!")
        return "none"


class BaseAnalysisPanel(BaseDockPanel):
    """
    Base class for analysis panels that run asynchronous workflows via the controller.

    Subclasses are expected to:
        * declare ``task_name`` (used by AppController dispatch)
        * override ``build_*`` helpers if they want the base class to assemble UI sections
        * implement ``gather_parameters`` and ``validate_inputs``
        * implement ``on_results`` to process controller payloads
    """

    task_name: str = ""

    def __init__(self, parent: Optional[Any] = None, *, panel_id: Optional[str] = None):
        # Set window flags BEFORE calling super().__init__ so they're applied correctly
        # Make analysis panels behave like independent windows that stay in taskbar
        # Use Window flag (not Dialog) so they appear in taskbar and stay when minimized
        if parent is None:
            # Only set window flags if no parent (standalone window)
            pass  # Will set after super().__init__
        
        super().__init__(parent=parent, panel_id=panel_id)
        self.progress_dialog: Optional[QProgressDialog] = None
        self._current_task: Optional[str] = None
        self._current_worker = None

        # Only set window flags if no parent (standalone window)
        # If panel has a parent (e.g., inside a QDialog), don't set window flags
        # as it will conflict with the parent container
        if parent is None:
            # Make analysis panels behave like independent windows that stay in taskbar
            # Use Window flag (not Dialog) so they appear in taskbar and stay when minimized
            self.setWindowFlags(
                Qt.WindowType.Window
                | Qt.WindowType.WindowMinimizeButtonHint
                | Qt.WindowType.WindowMaximizeButtonHint
                | Qt.WindowType.WindowCloseButtonHint
            )
            
            # Ensure panels stay in taskbar when minimized
            self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
            
            # Set proper sizing - fit to screen with reasonable defaults
            # Only for standalone windows, not when embedded in a dialog
            self._setup_panel_sizing()
        
        # Step 11: Connect to controller signals for unified progress/error handling
        # Note: controller may not be bound yet, so we'll connect in bind_controller
    
    def _setup_base_ui(self):
        """Override to wrap content in scroll area."""
        # Create scroll area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        
        # Create scrollable content widget
        scroll_content = QWidget()
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(5, 5, 5, 5)
        scroll_content_layout.setSpacing(10)
        
        # Set scroll content as the widget for scroll area
        scroll_area.setWidget(scroll_content)
        
        # Create a new main layout for the panel that contains the scroll area
        new_main_layout = QVBoxLayout(self)
        new_main_layout.setContentsMargins(0, 0, 0, 0)
        new_main_layout.setSpacing(0)
        new_main_layout.addWidget(scroll_area)
        
        # Set main_layout to scroll_content_layout so subclasses can use it
        self.main_layout = scroll_content_layout
        
        # Check if panel class defines _build_ui method (panels like geology_panel, wireframe_editor_panel use this)
        # If it does, don't call setup_ui here - the panel will call _build_ui() in its __init__ after super().__init__()
        # Otherwise, call setup_ui (standard BaseAnalysisPanel pattern)
        # CRITICAL: Check if this specific class (not parent) has _build_ui defined
        # This prevents calling setup_ui() which adds default UI elements when _build_ui() will add custom ones
        has_build_ui = '_build_ui' in self.__class__.__dict__
        
        if not has_build_ui:
            # Panel uses setup_ui (standard BaseAnalysisPanel pattern)
            # This adds default UI elements: property selector, filter section, parameter section, control buttons
            self.setup_ui()
        else:
            # Panel uses _build_ui() which will be called in its __init__ after super().__init__()
            # Don't call setup_ui() here to avoid duplicate UI elements
            # But we still need to call connect_signals() which is safe to call multiple times
            logger.debug(f"Panel {self.__class__.__name__} uses _build_ui(), skipping setup_ui() to avoid duplicate UI")
        
        self.connect_signals()
    
    def _setup_panel_sizing(self):
        """Setup proper panel sizing to fit screen."""
        # Get screen geometry
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            screen_width = screen_geometry.width()
            screen_height = screen_geometry.height()
            
            # Set default size (80% of screen, but not too large)
            default_width = min(1200, int(screen_width * 0.8))
            default_height = min(900, int(screen_height * 0.8))
            self.resize(default_width, default_height)
            
            # Set minimum size
            self.setMinimumSize(600, 400)
            
            # Set maximum size (don't exceed screen)
            max_width = min(1600, screen_width - 50)
            max_height = min(1200, screen_height - 50)
            self.setMaximumSize(max_width, max_height)
            
            logger.debug(f"Panel sizing: default=({default_width}, {default_height}), "
                        f"max=({max_width}, {max_height}), screen=({screen_width}, {screen_height})")
    

    def setup_ui(self):
        """Assemble standard sections for analysis panels."""
        layout = self.main_layout
        try:
            selector = self.build_property_selector()
            if selector:
                layout.addLayout(selector)
            filters = self.build_filter_section()
            if filters:
                layout.addLayout(filters)
            params_widget = self.build_parameter_section()
            if params_widget:
                layout.addLayout(params_widget)
            
            # Add control buttons (Stop and Close) at the bottom
            self._add_control_buttons(layout)
        except Exception:  # pragma: no cover - defensive
            logger.debug("BaseAnalysisPanel: UI assembly failed", exc_info=True)
    
    def _add_control_buttons(self, layout):
        """Add Stop and Close buttons to the panel."""
        button_layout = QHBoxLayout()
        
        # Stop button (only enabled when task is running)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        button_layout.addWidget(self.stop_button)
        
        button_layout.addStretch()
        
        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)

    # ------------------------------------------------------------------ #
    # Hooks for subclasses
    # ------------------------------------------------------------------ #
    def build_property_selector(self):
        """Return a layout/widget for property selection (optional)."""
        return None

    def build_filter_section(self):
        """Return a layout/widget for filter configuration (optional)."""
        return None

    def build_parameter_section(self):
        """Return a layout/widget for algorithm parameters (optional)."""
        return None

    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters required by the analysis task."""
        return {}

    def validate_inputs(self) -> bool:
        """Validate current UI inputs before running analysis."""
        return True

    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle successful analysis results (override required)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Execution helpers
    # ------------------------------------------------------------------ #
    def run_analysis(self) -> None:
        """Trigger asynchronous analysis via the controller - Step 11 unified pipeline."""
        # Log user action
        logger.info(f"USER ACTION: {self.__class__.__name__}.run_analysis() triggered for task='{self.task_name}'")
        
        if not self.controller:
            self.show_warning("Unavailable", "Controller is not connected; cannot run analysis.")
            logger.warning(f"Analysis aborted: No controller for {self.__class__.__name__}")
            return

        if not self.task_name:
            self.show_error("Configuration Error", "Panel must define task_name attribute.")
            logger.error(f"Configuration error: {self.__class__.__name__} has no task_name defined")
            return

        try:
            params = self.gather_parameters()
            # Log parameters for audit trail (sanitize sensitive data if needed)
            logger.info(f"ANALYSIS PARAMETERS: {self.__class__.__name__} task='{self.task_name}', params={_sanitize_params_for_log(params)}")
        except Exception as exc:
            logger.error("Failed to gather parameters for %s: %s", self.task_name or self.panel_id, exc, exc_info=True)
            self.show_error("Parameter Error", f"Failed to collect parameters:\n{exc}")
            return

        if not self.validate_inputs():
            logger.warning(f"Analysis inputs failed validation for {self.task_name or self.panel_id}")
            return

        label = "Running analysis..."
        if self.task_name:
            label = f"Running {self.task_name.replace('_', ' ').title()}..."
        self.show_progress(label)
        
        # Enable stop button
        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(True)
        self._current_task = self.task_name

        try:
            logger.info(f"ANALYSIS START: {self.__class__.__name__} dispatching task='{self.task_name}'")
            self.controller.run_analysis_task(
                task=self.task_name,
                params=params,
                callback=self.handle_results,
            )
            # Store reference to worker for cancellation
            if hasattr(self.controller, '_active_workers') and self.task_name in self.controller._active_workers:
                self._current_worker = self.controller._active_workers[self.task_name]
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to dispatch analysis task '%s': %s", self.task_name, exc, exc_info=True)
            self.hide_progress()
            self.show_error("Analysis Error", str(exc))
            if hasattr(self, 'stop_button'):
                self.stop_button.setEnabled(False)
            self._current_task = None

    def handle_results(self, payload: Optional[Dict[str, Any]]) -> None:
        """Callback invoked by the controller when analysis completes."""
        self.hide_progress()
        
        # Disable stop button
        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(False)
        self._current_task = None
        self._current_worker = None

        if not payload:
            logger.warning(f"ANALYSIS COMPLETED: {self.__class__.__name__} task='{self.task_name}' returned no payload")
            return

        if payload.get("error"):
            error_msg = payload["error"]
            logger.error(f"ANALYSIS FAILED: {self.__class__.__name__} task='{self.task_name}' - {error_msg}")
            self.show_error("Analysis Error", error_msg)
            return

        try:
            logger.info(f"ANALYSIS SUCCESS: {self.__class__.__name__} task='{self.task_name}' processing results...")
            self.on_results(payload)
            logger.debug(f"ANALYSIS RESULTS PROCESSED: {self.__class__.__name__} task='{self.task_name}'")
        except Exception as exc:
            logger.error(
                f"ANALYSIS RESULT PROCESSING FAILED: {self.__class__.__name__} task='{self.task_name}' - {exc}",
                exc_info=True
            )
            self.show_error("Result Processing Error", f"Failed to handle results:\n{exc}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Analysis result handling failed: %s", exc, exc_info=True)
            self.show_error("Result Handling Error", str(exc))
    
    def _on_stop_clicked(self):
        """Handle stop button click - cancel running task."""
        if self._current_task and self._current_worker:
            try:
                # Cancel the worker
                if hasattr(self._current_worker, 'cancel'):
                    self._current_worker.cancel()
                # Also try to cancel via controller
                if self.controller and hasattr(self.controller, 'cancel_task'):
                    self.controller.cancel_task(self._current_task)
                logger.info(f"Cancelled task: {self._current_task}")
                self.show_info("Task Cancelled", f"Task '{self._current_task}' has been cancelled.")
            except Exception as e:
                logger.error(f"Error cancelling task: {e}", exc_info=True)
                self.show_error("Cancellation Error", f"Failed to cancel task: {e}")
        
        self.hide_progress()
        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(False)
        self._current_task = None
        self._current_worker = None

    # ------------------------------------------------------------------ #
    # Progress dialog helpers
    # ------------------------------------------------------------------ #
    def show_progress(self, message: str) -> None:
        """Display a modal progress dialog with cancel button."""
        if self.progress_dialog is None:
            self.progress_dialog = QProgressDialog(message, "Cancel", 0, 0, self)
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress_dialog.canceled.connect(self._on_stop_clicked)
            self.progress_dialog.setMinimumDuration(0)
        else:
            self.progress_dialog.setLabelText(message)
        self.progress_dialog.forceShow()
        QApplication.processEvents()

    def hide_progress(self) -> None:
        """Close progress dialog when analysis completes."""
        if self.progress_dialog is not None:
            self.progress_dialog.hide()
            self.progress_dialog.deleteLater()
            self.progress_dialog = None

    # ============================================================================
    # PROJECT SAVE/RESTORE - Panel Settings
    # ============================================================================
    
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """
        Get panel settings for project save.
        
        Override this method in subclasses to return a dictionary of
        settings that should be saved with the project. Return None
        if this panel has no settings to save.
        
        Returns:
            Dictionary of settings (JSON-serializable) or None
        """
        return None
    
    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """
        Apply panel settings from project load.
        
        Override this method in subclasses to restore settings from
        a saved project. This will only be called if settings exist
        for this panel.
        
        Args:
            settings: Dictionary of settings from get_panel_settings()
        """
        pass

    # ------------------------------------------------------------------ #
    # Controller synchronisation
    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        """Refresh UI state (subclasses may extend)."""
        self.update_property_list()
    
    def refresh_from_registry(self) -> None:
        """
        Refresh panel data from DataRegistry.
        
        Subclasses should override to refresh their specific data.
        This is called when registry data is updated.
        """
        try:
            registry = self.get_registry()
            if registry:
                drillhole_data = registry.get_drillhole_data()
                if drillhole_data is not None:
                    if hasattr(self, '_on_drillhole_data_loaded'):
                        self._on_drillhole_data_loaded(drillhole_data)
        except Exception:
            pass
    
    def on_data_registry_updated(self, key: str) -> None:
        """
        Called when DataRegistry data is updated.
        
        Args:
            key: Data key that was updated (e.g., "drillholes", "block_model")
        """
        if key == "drillholes":
            self.refresh_from_registry()

    def update_property_list(self) -> None:
        """Update property selection widgets from the active block model."""
        if not hasattr(self, "property_combo"):
            return

        combo = getattr(self, "property_combo", None)
        if combo is None:
            return

        block_model = getattr(self.controller, "block_model", None) if self.controller else None
        if not block_model or not hasattr(block_model, "get_property_names"):
            return

        try:
            properties = block_model.get_property_names()
        except Exception:
            logger.debug("Failed to fetch property names from block model", exc_info=True)
            return

        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(properties)
        if current in properties:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    # ------------------------------------------------------------------ #
    # QWidget overrides
    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:  # pragma: no cover - GUI behaviour
        """Handle panel close event - cancel any running tasks and close properly."""
        # Cancel any running task
        if self._current_task and self._current_worker:
            try:
                if hasattr(self._current_worker, 'cancel'):
                    self._current_worker.cancel()
                if self.controller and hasattr(self.controller, 'cancel_task'):
                    self.controller.cancel_task(self._current_task)

                # Wait for the worker thread to finish to prevent QThread destruction warnings
                # This ensures the thread completes before the panel is destroyed
                if hasattr(self._current_worker, 'wait'):
                    logger.debug(f"Waiting for {self._current_task} worker thread to finish...")
                    # Wait up to 5 seconds for the thread to finish gracefully
                    if not self._current_worker.wait(5000):
                        logger.warning(f"Worker thread for {self._current_task} did not finish within timeout")
                    else:
                        logger.debug(f"Worker thread for {self._current_task} finished cleanly")
            except Exception as e:
                logger.debug(f"Error cancelling task on close: {e}")

        # Hide progress dialog
        self.hide_progress()

        # Accept the close event (actually close the window)
        event.accept()

        # Emit closed signal if available
        if hasattr(self, "dialog_closed"):
            try:
                QTimer.singleShot(0, getattr(self, "dialog_closed").emit)
            except Exception:
                logger.debug("Failed to emit dialog_closed signal", exc_info=True)

    def refresh_theme(self) -> None:
        """
        Refresh panel styles when theme changes.

        Subclasses should override this to update their specific widgets.
        By default, this applies the analysis panel stylesheet to the panel.
        """
        try:
            from .modern_styles import get_analysis_panel_stylesheet
            self.setStyleSheet(get_analysis_panel_stylesheet())
        except ImportError:
            logger.debug("Could not import get_analysis_panel_stylesheet for theme refresh")

