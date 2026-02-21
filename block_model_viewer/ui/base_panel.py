"""
Base panel class for all UI panels in Block Model Viewer.
Provides common functionality to reduce code duplication.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING, Any, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QGroupBox, QFormLayout,
    # Additional imports for clear_panel() functionality
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTextEdit, QPlainTextEdit, QTreeWidget, QListWidget, QTableWidget,
    QProgressBar, QRadioButton
)
from PyQt6.QtCore import pyqtSignal, Qt

from ..models.block_model import BlockModel

if TYPE_CHECKING:
    from ..controllers.app_controller import AppController

logger = logging.getLogger(__name__)


class BasePanel(QWidget):
    """
    Base class for all panel widgets.
    
    Provides common functionality:
    - Standardized setup/connection hooks
    - Block model management
    - Clear/reset functionality
    - Error handling
    - Status messages
    """
    
    # STEP 40: Diagnostic identifier for panel diagnostics
    PANEL_ID: str = "BasePanel"
    
    # Common signals
    status_message = pyqtSignal(str)  # Emit status messages to main window
    error_occurred = pyqtSignal(str)  # Emit error messages
    
    def __init__(self, parent: Optional[QWidget] = None, panel_id: Optional[str] = None):
        """Initialize base panel."""
        super().__init__(parent)
        
        # Common attributes
        self.controller: Optional["AppController"] = None
        self._block_model: Optional[Any] = None  # Can be BlockModel or DataFrame
        self._is_initialized = False
        self.panel_id: str = panel_id or self.__class__.__name__
        self.main_layout: Optional[QVBoxLayout] = None
        
        # STEP 40: Diagnostic tag for diagnostics
        self._diagnostic_tag = getattr(self.__class__, "PANEL_ID", self.__class__.__name__)
        
        # Setup UI (subclasses can override setup_ui/connect_signals)
        self._setup_base_ui()
        
        logger.debug("Initialized %s (panel_id=%s)", self.__class__.__name__, self.panel_id)
    
    def _setup_base_ui(self):
        """Setup base UI structure."""
        # Only create layout if one doesn't already exist (for panels that override)
        if self.layout() is None:
            self.main_layout = QVBoxLayout(self)
            self.main_layout.setContentsMargins(5, 5, 5, 5)
            self.main_layout.setSpacing(10)
        else:
            # Use existing layout (set by subclass before calling super().__init__)
            self.main_layout = self.layout()
        
        # Call subclass-specific hooks
        self.setup_ui()
        self.connect_signals()
        
        self._is_initialized = True
    
    # -------------------------------------------------------------------------
    # Template hooks - subclasses must override setup_ui
    # -------------------------------------------------------------------------
    def setup_ui(self):
        """
        Build widget layout. Subclasses must override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement setup_ui()"
        )
    
    def connect_signals(self):
        """Wire Qt signals. Subclasses can override when needed."""
        return
    
    def refresh(self):
        """Refresh panel state from latest data."""
        return
    
    # =========================================================================
    # Block Model Management
    # =========================================================================
    
    def bind_controller(self, controller: Optional["AppController"]) -> None:
        """
        Attach the shared application controller to this panel.
        
        Args:
            controller: AppController instance (or None to detach)
        """
        self.controller = controller
        logger.debug("%s bound to controller=%s", self.panel_id, bool(controller))
    
    def get_registry(self):
        """
        Get DataRegistry instance via dependency injection.
        
        Tries to get registry from:
        1. Controller (if bound) -> controller.registry
        2. MainWindow parent -> main_window.controller.registry
        3. Falls back to singleton for backward compatibility
        
        Returns:
            DataRegistry instance
        """
        # Try controller first (preferred - dependency injection)
        if self.controller and hasattr(self.controller, 'registry'):
            return self.controller.registry
        
        # Try MainWindow parent
        parent = self.parent()
        while parent:
            if hasattr(parent, 'controller') and parent.controller:
                if hasattr(parent.controller, 'registry'):
                    return parent.controller.registry
            # Check if parent is MainWindow
            if hasattr(parent, '_registry') and parent._registry:
                return parent._registry
            parent = parent.parent()
        
        # Fallback to singleton (backward compatibility)
        from ..core.data_registry import DataRegistry
        return DataRegistry.instance()
    
    def set_drillhole_data(self, data):
        """
        Legacy compatibility fallback for set_drillhole_data().
        
        This method prevents AttributeError when legacy code calls set_drillhole_data().
        New code should use registry pattern: self.get_registry().get_drillhole_data()
        and connect to registry.drillholeDataLoaded signal.
        
        Args:
            data: Drillhole data (ignored - panels should fetch from registry)
        """
        try:
            # Try to refresh from registry if panel implements on_data_registry_updated
            if hasattr(self, 'on_data_registry_updated'):
                self.on_data_registry_updated("drillholes")
            elif hasattr(self, 'refresh_from_registry'):
                self.refresh_from_registry()
            else:
                # Try to refresh drillhole data from registry
                try:
                    registry = self.get_registry()
                    if registry:
                        drillhole_data = registry.get_drillhole_data()
                        if drillhole_data is not None:
                            # Panel should implement _on_drillhole_data_loaded or similar
                            if hasattr(self, '_on_drillhole_data_loaded'):
                                self._on_drillhole_data_loaded(drillhole_data)
                            elif hasattr(self, '_on_drillhole_data_loaded_from_registry'):
                                self._on_drillhole_data_loaded_from_registry(drillhole_data)
                except Exception:
                    pass
        except Exception:
            # Silently ignore - prevents AttributeError in legacy code
            pass
    
    # =========================================================================
    # Block Model Management
    # =========================================================================
    
    def set_block_model(self, block_model: Optional[Any]):
        """
        Set the current block model.
        
        Args:
            block_model: BlockModel instance or DataFrame to use
        """
        self._block_model = block_model
        
        # Get block count safely (handles both BlockModel and DataFrame)
        block_count = 0
        if block_model is not None:
            try:
                # Check if it's a BlockModel object
                if hasattr(block_model, 'block_count'):
                    block_count = block_model.block_count
                # Check if it's a pandas DataFrame
                elif hasattr(block_model, '__len__') and hasattr(block_model, 'columns'):
                    block_count = len(block_model)
            except Exception:
                # Fallback: try to get length
                try:
                    block_count = len(block_model) if block_model is not None else 0
                except Exception:
                    block_count = 0
        
        self.on_block_model_changed()
        logger.info(
            f"{self.__class__.__name__}: Block model set with "
            f"{block_count} blocks"
        )
    
    def on_block_model_changed(self):
        """
        Called when block model changes.
        
        Subclasses can override to update UI.
        """
        pass
    
    # Backwards compatibility for panels overriding the old hook
    def _on_block_model_changed(self):  # pragma: no cover - compatibility shim
        self.on_block_model_changed()
    
    @property
    def block_model(self) -> Optional[BlockModel]:
        """Get current block model."""
        return self._block_model
    
    @property
    def has_block_model(self) -> bool:
        """Check if a block model is loaded."""
        return self._block_model is not None
    
    # =========================================================================
    # Clear/Reset Functionality
    # =========================================================================
    
    def clear(self):
        """
        Clear all panel data and reset to initial state.
        
        Subclasses should override to clear specific UI elements.
        """
        self._block_model = None
        logger.info(f"{self.__class__.__name__}: Cleared")
    
    def reset(self):
        """
        Reset panel to default settings (keep data).

        Subclasses can override for custom reset behavior.
        """
        logger.info(f"{self.__class__.__name__}: Reset to defaults")

    def clear_panel(self):
        """
        Clear all UI widgets to their default states.

        Automatically handles common Qt widgets:
        - QComboBox: reset to first item
        - QLineEdit: clear text
        - QSpinBox/QDoubleSpinBox: reset to minimum value
        - QCheckBox/QRadioButton: uncheck
        - QTextEdit/QPlainTextEdit: clear text
        - QTreeWidget/QListWidget: clear all items
        - QTableWidget: clear all rows
        - QProgressBar: reset to 0

        Subclasses should override and call super().clear_panel() first
        for custom behavior (e.g., clearing matplotlib figures, internal state).
        """
        self._block_model = None

        # Clear all child widgets recursively
        for widget in self.findChildren(QWidget):
            try:
                # Block signals to prevent cascading updates during clear
                widget.blockSignals(True)

                if isinstance(widget, QComboBox):
                    # Reset to first item (keep items, just reset selection)
                    if widget.count() > 0:
                        widget.setCurrentIndex(0)
                elif isinstance(widget, QLineEdit):
                    widget.clear()
                elif isinstance(widget, QSpinBox):
                    widget.setValue(widget.minimum())
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(widget.minimum())
                elif isinstance(widget, (QCheckBox, QRadioButton)):
                    widget.setChecked(False)
                elif isinstance(widget, QTextEdit):
                    widget.clear()
                elif isinstance(widget, QPlainTextEdit):
                    widget.clear()
                elif isinstance(widget, QTreeWidget):
                    widget.clear()
                elif isinstance(widget, QListWidget):
                    widget.clear()
                elif isinstance(widget, QTableWidget):
                    widget.setRowCount(0)
                elif isinstance(widget, QProgressBar):
                    widget.setValue(0)

                widget.blockSignals(False)
            except Exception:
                # Widget may have been deleted or has issues - skip
                pass

        logger.info(f"{self.__class__.__name__}: Panel UI cleared")

    # =========================================================================
    # Error Handling & Messages
    # =========================================================================
    
    def show_error(self, title: str, message: str):
        """
        Show an error message dialog.
        
        Args:
            title: Dialog title
            message: Error message
        """
        QMessageBox.critical(self, title, message)
        self.error_occurred.emit(f"{title}: {message}")
        logger.error(f"{self.__class__.__name__}: {title} - {message}")
    
    def show_warning(self, title: str, message: str):
        """
        Show a warning message dialog.
        
        Args:
            title: Dialog title
            message: Warning message
        """
        QMessageBox.warning(self, title, message)
        logger.warning(f"{self.__class__.__name__}: {title} - {message}")
    
    def show_info(self, title: str, message: str):
        """
        Show an information message dialog.
        
        Args:
            title: Dialog title
            message: Info message
        """
        QMessageBox.information(self, title, message)
        self.status_message.emit(message)
        logger.info(f"{self.__class__.__name__}: {title} - {message}")
    
    def emit_status(self, message: str):
        """
        Emit a status message (without showing dialog).
        
        Args:
            message: Status message
        """
        self.status_message.emit(message)
        logger.debug(f"{self.__class__.__name__}: {message}")

    def emit_error(self, message: str, details: Optional[str] = None):
        """
        Emit an error signal and log it (without dialog).
        
        Args:
            message: Short error summary
            details: Optional detail string
        """
        full_message = f"{message}\n\n{details}" if details else message
        self.error_occurred.emit(full_message)
        logger.error("%s: %s", self.__class__.__name__, full_message)
    
    def request_task(self, name: str, **kwargs):
        """
        Request an asynchronous/background task via the controller.
        
        Args:
            name: Task identifier
            **kwargs: Task parameters
        """
        if self.controller and hasattr(self.controller, "request_task"):
            try:
                self.controller.request_task(name, **kwargs)
            except Exception as exc:
                try:
                    exc_msg = str(exc)
                    logger.debug("Controller rejected task '%s': %s", name, exc_msg)
                except Exception:
                    logger.debug("Controller rejected task '%s': <unprintable error>", name)
    
    # =========================================================================
    # Validation Helpers
    # =========================================================================
    
    def validate_block_model_loaded(self) -> bool:
        """
        Validate that a block model is loaded.
        
        Returns:
            True if block model exists, False otherwise (shows error)
        """
        if not self.has_block_model:
            self.show_error(
                "No Block Model",
                "Please load a block model first.\n\n"
                "Go to: File → Open File"
            )
            return False
        return True
    
    def validate_property_exists(self, property_name: str) -> bool:
        """
        Validate that a property exists in the block model.
        
        Args:
            property_name: Property name to check
            
        Returns:
            True if property exists, False otherwise (shows error)
        """
        if not self.validate_block_model_loaded():
            return False
        
        if property_name not in self._block_model.get_property_names():
            self.show_error(
                "Property Not Found",
                f"Property '{property_name}' not found in block model.\n\n"
                f"Available properties:\n" +
                ", ".join(self._block_model.get_property_names()[:10])
            )
            return False
        return True
    
    # =========================================================================
    # UI Helper Methods
    # =========================================================================
    
    def create_group_box(self, title: str, layout: Optional[QVBoxLayout] = None) -> QGroupBox:
        """
        Create a standard group box.
        
        Args:
            title: Group box title
            layout: Optional layout to use (creates QVBoxLayout if None)
            
        Returns:
            QGroupBox with layout applied
        """
        group = QGroupBox(title)
        if layout is None:
            layout = QVBoxLayout()
        group.setLayout(layout)
        return group
    
    def create_form_layout(self) -> QFormLayout:
        """Create a standard form layout with consistent spacing."""
        form = QFormLayout()
        form.setSpacing(8)
        form.setContentsMargins(10, 10, 10, 10)
        return form
    
    def create_button_row(self, *button_specs) -> QHBoxLayout:
        """
        Create a horizontal layout with buttons.
        
        Args:
            button_specs: Tuples of (label, callback) or QPushButton instances
            
        Returns:
            QHBoxLayout with buttons
        """
        layout = QHBoxLayout()
        layout.addStretch()
        
        for spec in button_specs:
            if isinstance(spec, QPushButton):
                layout.addWidget(spec)
            else:
                label, callback = spec
                btn = QPushButton(label)
                btn.clicked.connect(callback)
                layout.addWidget(btn)
        
        return layout
    
    def add_separator(self):
        """Add a horizontal separator line to main layout."""
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #555;")
        self.main_layout.addWidget(line)
    
    # =========================================================================
    # STEP 40: Diagnostic Support
    # =========================================================================
    
    def diagnostic_name(self) -> str:
        """
        Get diagnostic name for this panel (STEP 40).

        Returns:
            Diagnostic identifier string
        """
        return self._diagnostic_tag

    def refresh_theme(self) -> None:
        """
        Refresh panel styles when theme changes.

        Override in subclasses to update widget-specific styles.
        The main_window._refresh_all_themed_widgets() method will call this
        automatically when the theme changes.
        """
        # Base implementation does nothing - subclasses should override
        pass


class BaseDockPanel(BasePanel):
    """
    Base class for dockable panels with unified lifecycle management.

    Extends BasePanel with dock-specific functionality and integrates with PanelManager.
    NEVER permanently closes - always hides instead of destroying.
    """

    # PanelManager integration
    PANEL_ID: str = "BaseDockPanel"  # Override in subclasses
    PANEL_NAME: str = "Base Dock Panel"  # Override in subclasses
    PANEL_CATEGORY = None  # Override with PanelCategory enum
    PANEL_ICON = None  # Icon name without extension
    PANEL_SHORTCUT = None  # Keyboard shortcut string
    PANEL_DEFAULT_DOCK_AREA = None  # DockArea enum
    PANEL_DEFAULT_VISIBLE = True
    PANEL_MINIMUM_WIDTH = 250
    PANEL_MINIMUM_HEIGHT = 200
    PANEL_TOOLTIP = None

    def __init__(self, parent: Optional[QWidget] = None, panel_id: Optional[str] = None):
        """Initialize base dock panel."""
        # Use provided panel_id or class attribute
        final_panel_id = panel_id or getattr(self.__class__, 'PANEL_ID', self.__class__.__name__)

        super().__init__(parent, panel_id=final_panel_id)

        # Set window title
        title = getattr(self.__class__, 'PANEL_NAME', self.__class__.__name__)
        if title.endswith('Panel'):
            title = title[:-5] + ' Panel'
        self.setWindowTitle(title)

        # PanelManager reference (set by manager during registration)
        self._panel_manager = None

        logger.debug(f"Initialized BaseDockPanel: {self.panel_id}")

    def setup_ui(self):
        """Base dock panels don't build UI themselves - subclasses must override."""
        pass

    # =========================================================================
    # Logging Helpers for User Actions
    # =========================================================================

    def _log_button_click(self, button_name: str, callback: Callable) -> Callable:
        """
        Wrap a button click callback with logging.
        
        Usage:
            button.clicked.connect(self._log_button_click("Run Analysis", self._do_analysis))
        """
        def wrapper(*args, **kwargs):
            logger.info(f"USER ACTION: {self.__class__.__name__}.{button_name} clicked")
            try:
                result = callback(*args, **kwargs)
                logger.debug(f"USER ACTION COMPLETED: {self.__class__.__name__}.{button_name}")
                return result
            except Exception as e:
                logger.error(
                    f"USER ACTION FAILED: {self.__class__.__name__}.{button_name} - {str(e)}",
                    exc_info=True
                )
                raise
        return wrapper

    def _log_combo_change(self, combo_name: str, callback: Callable) -> Callable:
        """
        Wrap a combo box change callback with logging.
        
        Usage:
            combo.currentTextChanged.connect(self._log_combo_change("Variable", self._on_variable_changed))
        """
        def wrapper(value, *args, **kwargs):
            logger.info(f"PARAMETER CHANGE: {self.__class__.__name__}.{combo_name} = {value}")
            return callback(value, *args, **kwargs)
        return wrapper

    def _log_value_change(self, param_name: str, callback: Callable) -> Callable:
        """
        Wrap a spinbox/slider value change callback with logging.
        
        Usage:
            spinbox.valueChanged.connect(self._log_value_change("NumRealizations", self._on_nreal_changed))
        """
        def wrapper(value, *args, **kwargs):
            logger.info(f"PARAMETER CHANGE: {self.__class__.__name__}.{param_name} = {value}")
            return callback(value, *args, **kwargs)
        return wrapper

    def on_register_in_panel_manager(self, manager):
        """
        Called when this panel is registered with the PanelManager.

        Args:
            manager: PanelManager instance
        """
        self._panel_manager = manager
        logger.debug(f"Panel {self.panel_id} registered with PanelManager")

    def resizeEvent(self, event):
        """Log panel resize events for audit trail."""
        old_size = event.oldSize()
        new_size = event.size()
        logger.debug(
            f"{self.__class__.__name__} ({self.panel_id}) resized: "
            f"{old_size.width()}x{old_size.height()} -> {new_size.width()}x{new_size.height()}"
        )
        super().resizeEvent(event)

    def showEvent(self, event):
        """Log panel show events for audit trail."""
        logger.info(f"{self.__class__.__name__} ({self.panel_id}) shown")
        super().showEvent(event)

    def hideEvent(self, event):
        """Log panel hide events for audit trail."""
        logger.info(f"{self.__class__.__name__} ({self.panel_id}) hidden")
        super().hideEvent(event)

    def closeEvent(self, event):
        """
        Handle panel close event.

        NEVER permanently close - hide instead of destroy.
        This enforces the professional panel lifecycle.
        """
        logger.info(f"{self.__class__.__name__} ({self.panel_id}): Close requested - hiding instead of destroying")

        # Hide the panel (PanelManager will handle the dock widget)
        self.hide()

        # Ignore the close event to prevent destruction
        event.ignore()

        # Notify PanelManager if available
        if self._panel_manager:
            panel_id = getattr(self, 'PANEL_ID', self.__class__.__name__)
            self._panel_manager.hide_panel(panel_id)

    def prevent_permanent_close(self):
        """
        Ensure this panel can never be permanently closed.

        Called automatically by PanelManager, but can be called manually.
        """
        # The closeEvent override above already prevents permanent closing
        # This method exists for API compatibility
        pass

    def save_state(self) -> dict:
        """
        Save panel state for persistence.

        Returns:
            Dictionary with panel state data
        """
        return {
            "panel_id": self.panel_id,
            "class_name": self.__class__.__name__,
            "timestamp": str(__import__('datetime').datetime.now())
        }

    def load_state(self, state: dict):
        """
        Load panel state from persistence.

        Args:
            state: State dictionary from save_state()
        """
        # Base implementation does nothing - subclasses can override
        logger.debug(f"Loading state for panel {self.panel_id}")

    # =========================================================================
    # PanelManager Integration Helpers
    # =========================================================================

    def get_panel_id(self) -> str:
        """Get the panel ID for PanelManager registration."""
        return getattr(self, 'PANEL_ID', self.__class__.__name__)

    def request_show_panel(self, panel_id: str):
        """Request that PanelManager show another panel."""
        if self._panel_manager:
            self._panel_manager.show_panel(panel_id)

    def request_hide_panel(self, panel_id: str):
        """Request that PanelManager hide another panel."""
        if self._panel_manager:
            self._panel_manager.hide_panel(panel_id)

    def request_toggle_panel(self, panel_id: str):
        """Request that PanelManager toggle another panel."""
        if self._panel_manager:
            self._panel_manager.toggle_panel(panel_id)


class BaseDialogPanel(BasePanel):
    """
    Base class for dialog panels (popup windows).
    
    Extends BasePanel with dialog-specific functionality.
    """
    
    # Signal for when dialog is closed
    dialog_closed = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize base dialog panel."""
        super().__init__(parent)
        
        # Set window flags for independent window
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        
        self.setWindowTitle(self.__class__.__name__.replace("Panel", ""))
    
    def setup_ui(self):
        """Base dialog panels don't build UI themselves - subclasses must override."""
        pass
    
    def resizeEvent(self, event):
        """Log dialog resize events for audit trail."""
        old_size = event.oldSize()
        new_size = event.size()
        logger.debug(
            f"{self.__class__.__name__} dialog resized: "
            f"{old_size.width()}x{old_size.height()} -> {new_size.width()}x{new_size.height()}"
        )
        super().resizeEvent(event)

    def showEvent(self, event):
        """Log dialog show events for audit trail."""
        logger.info(f"{self.__class__.__name__} dialog opened")
        super().showEvent(event)

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.dialog_closed.emit()
        logger.info(f"{self.__class__.__name__} dialog closed")
        event.accept()

















