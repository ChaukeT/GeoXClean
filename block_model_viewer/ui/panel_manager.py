"""
PanelManager: Unified Professional Lifecycle System for GeoX Panels

Provides centralized panel management with:
- Registry of all panel classes and instances
- Hide-on-close behavior (never destroy)
- State persistence (visibility, geometry, dock mode)
- Menu/toolbar toggle controls
- Keyboard shortcuts
- Reset layout functionality
- Professional logging and auditability
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QMenu, QToolBar, QStatusBar, QMessageBox
)
from PyQt6.QtGui import QAction
from PyQt6.QtGui import QActionGroup, QKeySequence, QIcon

from .base_panel import BasePanel

logger = logging.getLogger(__name__)


class PanelCategory(Enum):
    """Panel categories for organization."""
    VIEWER = "viewer"
    PROPERTY = "property"
    SCENE = "scene"
    LAYER = "layer"
    INFO = "info"
    ANALYSIS = "analysis"
    SELECTION = "selection"
    CROSS_SECTION = "cross_section"
    DISPLAY = "display"
    RESOURCE = "resource"
    PLANNING = "planning"
    GEOSTATS = "geostats"
    DRILLHOLE = "drillhole"
    ESG = "esg"
    GEOTECH = "geotech"
    OPTIMIZATION = "optimization"
    CHART = "chart"
    REPORT = "report"
    CONFIG = "config"
    OTHER = "other"


class DockArea(Enum):
    """Standard dock widget areas."""
    LEFT = Qt.DockWidgetArea.LeftDockWidgetArea
    RIGHT = Qt.DockWidgetArea.RightDockWidgetArea
    TOP = Qt.DockWidgetArea.TopDockWidgetArea
    BOTTOM = Qt.DockWidgetArea.BottomDockWidgetArea
    FLOATING = None  # Special case for floating


@dataclass
class PanelInfo:
    """Complete metadata for a panel."""
    panel_id: str
    name: str
    category: PanelCategory
    panel_class: Type[BasePanel]
    icon_name: Optional[str] = None
    shortcut: Optional[str] = None
    default_dock_area: DockArea = DockArea.LEFT
    default_visible: bool = True
    minimum_width: int = 250
    minimum_height: int = 200
    tooltip: Optional[str] = None
    factory: Optional[Callable[[], BasePanel]] = None

    # Runtime state
    instance: Optional[BasePanel] = None
    dock_widget: Optional[QDockWidget] = None
    menu_action: Optional[QAction] = None
    toolbar_action: Optional[QAction] = None
    is_visible: bool = False
    is_floating: bool = False
    geometry_data: Optional[str] = None

    def get_icon(self) -> QIcon:
        """Get icon for this panel."""
        if not self.icon_name:
            return QIcon()

        # Try multiple icon locations
        icon_paths = [
            Path(__file__).parent.parent / "assets" / "icons" / f"{self.icon_name}.svg",
            Path(__file__).parent / "icons" / f"{self.icon_name}.svg",
            Path(__file__).parent.parent / "assets" / "icons" / f"{self.icon_name}.png",
            Path(__file__).parent / "icons" / f"{self.icon_name}.png"
        ]

        for icon_path in icon_paths:
            if icon_path.exists():
                return QIcon(str(icon_path))

        return QIcon()


class PanelManager(QObject):
    """
    Central Panel Manager for GeoX Application.

    Manages the complete lifecycle of all UI panels:
    - Registration and instantiation
    - Hide-on-close behavior
    - State persistence
    - UI controls (menu, toolbar, shortcuts)
    - Reset layout functionality
    """

    # Signals
    panel_opened = pyqtSignal(str)  # panel_id
    panel_hidden = pyqtSignal(str)  # panel_id
    panel_docked = pyqtSignal(str)  # panel_id
    panel_floated = pyqtSignal(str)  # panel_id
    workspace_reset = pyqtSignal()

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        self._panels: Dict[str, PanelInfo] = {}
        self._settings_file = Path("panel_states.json")
        self._initialized = False

        # UI elements we'll create
        self._panels_menu: Optional[QMenu] = None
        self._panels_toolbar: Optional[QToolBar] = None
        self._shortcut_actions: Dict[str, QAction] = {}

        logger.info("PanelManager initialized")

    # =========================================================================
    # Panel Registration
    # =========================================================================

    def register_panel(self, panel_info: PanelInfo):
        """
        Register a panel with the manager.

        Args:
            panel_info: Complete panel metadata
        """
        if panel_info.panel_id in self._panels:
            logger.warning(f"Panel {panel_info.panel_id} already registered, replacing")

        # Set factory if not provided
        if panel_info.factory is None:
            panel_info.factory = lambda cls=panel_info.panel_class: cls(parent=self.main_window)

        self._panels[panel_info.panel_id] = panel_info
        logger.debug(f"Registered panel: {panel_info.panel_id} ({panel_info.name})")

    def register_panel_class(
        self,
        panel_class: Type[BasePanel],
        category: PanelCategory = PanelCategory.OTHER,
        icon_name: Optional[str] = None,
        shortcut: Optional[str] = None,
        default_dock_area: DockArea = DockArea.LEFT,
        default_visible: bool = True,
        minimum_width: int = 250,
        minimum_height: int = 200,
        tooltip: Optional[str] = None
    ):
        """
        Convenience method to register a panel class.

        Args:
            panel_class: The panel class to register
            category: Panel category
            icon_name: Icon filename (without extension)
            shortcut: Keyboard shortcut string
            default_dock_area: Default dock area
            default_visible: Whether panel is visible by default
            minimum_width: Minimum width in pixels
            minimum_height: Minimum height in pixels
            tooltip: Help tooltip
        """
        # Get panel ID from class
        panel_id = getattr(panel_class, 'PANEL_ID', panel_class.__name__)

        # Generate display name
        name = getattr(panel_class, 'PANEL_NAME', panel_class.__name__)
        if name.endswith('Panel'):
            name = name[:-5] + ' Panel'
        elif not name.endswith(' Panel'):
            name += ' Panel'

        panel_info = PanelInfo(
            panel_id=panel_id,
            name=name,
            category=category,
            panel_class=panel_class,
            icon_name=icon_name,
            shortcut=shortcut,
            default_dock_area=default_dock_area,
            default_visible=default_visible,
            minimum_width=minimum_width,
            minimum_height=minimum_height,
            tooltip=tooltip
        )

        self.register_panel(panel_info)

    def get_panel_info(self, panel_id: str) -> Optional[PanelInfo]:
        """Get panel information by ID."""
        return self._panels.get(panel_id)

    def get_panel_instance(self, panel_id: str):
        """Get the widget instance for a panel by ID.

        Args:
            panel_id: Panel identifier

        Returns:
            The panel widget instance, or None if not found/created.
        """
        panel_info = self._panels.get(panel_id)
        if panel_info:
            return panel_info.instance
        return None

    def get_all_panels(self) -> List[PanelInfo]:
        """Get all registered panels."""
        return list(self._panels.values())

    def get_panels_by_category(self, category: PanelCategory) -> List[PanelInfo]:
        """Get panels in a specific category."""
        return [p for p in self._panels.values() if p.category == category]

    # =========================================================================
    # Panel Lifecycle Management
    # =========================================================================

    def show_panel(self, panel_id: str, show: bool = True) -> bool:
        """
        Show or hide a panel.

        Args:
            panel_id: Panel identifier
            show: True to show, False to hide

        Returns:
            True if operation succeeded
        """
        panel_info = self.get_panel_info(panel_id)
        if not panel_info:
            logger.warning(f"Panel {panel_id} not registered")
            return False

        if show:
            return self._show_panel(panel_info)
        else:
            return self._hide_panel(panel_info)

    def hide_panel(self, panel_id: str) -> bool:
        """Hide a panel."""
        return self.show_panel(panel_id, False)

    def toggle_panel(self, panel_id: str) -> bool:
        """
        Toggle panel visibility.

        Args:
            panel_id: Panel identifier

        Returns:
            True if panel is now visible, False otherwise
        """
        panel_info = self.get_panel_info(panel_id)
        if not panel_info:
            logger.warning(f"Panel {panel_id} not registered")
            return False

        currently_visible = self.is_panel_visible(panel_id)
        return self.show_panel(panel_id, not currently_visible)

    def is_panel_visible(self, panel_id: str) -> bool:
        """Check if a panel is currently visible."""
        panel_info = self.get_panel_info(panel_id)
        if not panel_info:
            return False

        if panel_info.dock_widget:
            return panel_info.dock_widget.isVisible()
        return False

    def _show_panel(self, panel_info: PanelInfo) -> bool:
        """Show a panel (internal implementation)."""
        try:
            # Create panel if it doesn't exist
            if not panel_info.instance:
                if not self._create_panel_instance(panel_info):
                    return False

            # Create dock widget if needed
            if not panel_info.dock_widget:
                if not self._create_dock_widget(panel_info):
                    return False

            # Show the dock widget
            panel_info.dock_widget.show()
            panel_info.dock_widget.raise_()
            panel_info.is_visible = True

            # Update menu/toolbar state
            self._update_ui_state(panel_info)

            # Emit signal and log
            self.panel_opened.emit(panel_info.panel_id)
            logger.info(f"Panel opened: {panel_info.panel_id}")

            # Update status bar
            self._update_status_bar(f"Opened {panel_info.name}")

            return True

        except Exception as e:
            logger.error(f"Failed to show panel {panel_info.panel_id}: {e}")
            return False

    def _hide_panel(self, panel_info: PanelInfo) -> bool:
        """Hide a panel (internal implementation)."""
        try:
            if panel_info.dock_widget:
                panel_info.dock_widget.hide()
                panel_info.is_visible = False

                # Update menu/toolbar state
                self._update_ui_state(panel_info)

                # Save state
                self._save_panel_state(panel_info)

                # Emit signal and log
                self.panel_hidden.emit(panel_info.panel_id)
                logger.info(f"Panel hidden: {panel_info.panel_id}")

                # Update status bar
                self._update_status_bar(f"Hidden {panel_info.name}")

                return True
            return False

        except Exception as e:
            logger.error(f"Failed to hide panel {panel_info.panel_id}: {e}")
            return False

    def _create_panel_instance(self, panel_info: PanelInfo) -> bool:
        """Create the panel instance."""
        try:
            if not panel_info.factory:
                logger.error(f"No factory for panel {panel_info.panel_id}")
                return False

            panel_info.instance = panel_info.factory()

            # Register the panel with itself for lifecycle callbacks
            if hasattr(panel_info.instance, 'on_register_in_panel_manager'):
                panel_info.instance.on_register_in_panel_manager(self)

            logger.debug(f"Created panel instance: {panel_info.panel_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create panel instance {panel_info.panel_id}: {e}")
            return False

    def _create_dock_widget(self, panel_info: PanelInfo) -> bool:
        """Create the dock widget for a panel."""
        try:
            from .persistent_dock import PersistentDockWidget

            # Create persistent dock widget (handles hide-on-close)
            dock = PersistentDockWidget(
                key=panel_info.panel_id,
                title=panel_info.name,
                content=panel_info.instance,
                stage_text=panel_info.category.value.title()
            )

            # Configure dock widget
            dock.setObjectName(f"Dock_{panel_info.panel_id}")
            dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea |
                Qt.DockWidgetArea.RightDockWidgetArea |
                Qt.DockWidgetArea.TopDockWidgetArea |
                Qt.DockWidgetArea.BottomDockWidgetArea
            )
            dock.setMinimumWidth(panel_info.minimum_width)
            dock.setMinimumHeight(panel_info.minimum_height)

            # Set icon and tooltip
            icon = panel_info.get_icon()
            if not icon.isNull():
                dock.setWindowIcon(icon)
            if panel_info.tooltip:
                dock.setToolTip(panel_info.tooltip)

            # Connect signals for lifecycle tracking
            dock.visibilityChanged.connect(
                lambda visible, p=panel_info: self._on_dock_visibility_changed(p, visible)
            )

            # Store reference
            panel_info.dock_widget = dock

            # Add to main window
            dock_area = panel_info.default_dock_area.value if panel_info.default_dock_area.value else Qt.DockWidgetArea.LeftDockWidgetArea
            self.main_window.addDockWidget(dock_area, dock)

            # Restore geometry if available
            self._restore_panel_state(panel_info)

            logger.debug(f"Created dock widget: {panel_info.panel_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create dock widget for {panel_info.panel_id}: {e}")
            return False

    # =========================================================================
    # UI Creation and Management
    # =========================================================================

    def initialize_ui(self):
        """Initialize UI elements (menu, toolbar, shortcuts)."""
        if self._initialized:
            return

        self._create_panels_menu()
        self._create_panels_toolbar()
        self._create_keyboard_shortcuts()
        self._load_workspace_state()

        self._initialized = True
        logger.info("PanelManager UI initialized")

    def _create_panels_menu(self):
        """Create Window → Panels menu."""
        # Find or create Window menu
        window_menu = None
        for menu in self.main_window.menuBar().findChildren(QMenu):
            if menu.title() == "&Window":
                window_menu = menu
                break

        if not window_menu:
            window_menu = self.main_window.menuBar().addMenu("&Window")

        # Create Panels submenu
        self._panels_menu = window_menu.addMenu("&Panels")

        # Group panels by category
        categories = {}
        for panel_info in self.get_all_panels():
            if panel_info.category not in categories:
                categories[panel_info.category] = []
            categories[panel_info.category].append(panel_info)

        # Add panels to menu
        for category in PanelCategory:
            if category not in categories:
                continue

            # Add category submenu
            category_menu = self._panels_menu.addMenu(category.value.title())

            # Sort panels by name
            category_panels = sorted(categories[category], key=lambda p: p.name)

            for panel_info in category_panels:
                action = category_menu.addAction(panel_info.name)
                action.setCheckable(True)
                action.setChecked(panel_info.default_visible)
                action.triggered.connect(
                    lambda checked, pid=panel_info.panel_id: self.toggle_panel(pid)
                )
                panel_info.menu_action = action

        # Add separator and reset layout
        self._panels_menu.addSeparator()
        reset_action = self._panels_menu.addAction("Reset Layout")
        reset_action.triggered.connect(self.reset_layout)

    def _create_panels_toolbar(self):
        """Create panels toolbar with frequently used panel toggles."""
        self._panels_toolbar = self.main_window.addToolBar("Panels")
        self._panels_toolbar.setObjectName("PanelsToolbar")
        self._panels_toolbar.setToolTip("Panel visibility toggles")

        # Add frequently used panels (first 8)
        frequently_used = [
            "PropertyPanel", "SceneInspectorPanel",
            "SurveyDeformationPanel",
            "QCWindow", "KrigingPanel", "VariogramPanel"
        ]

        added_count = 0
        for panel_id in frequently_used:
            panel_info = self.get_panel_info(panel_id)
            if panel_info and added_count < 8:
                action = self._panels_toolbar.addAction(panel_info.name)
                action.setCheckable(True)
                action.setChecked(panel_info.default_visible)

                icon = panel_info.get_icon()
                if not icon.isNull():
                    action.setIcon(icon)

                action.triggered.connect(
                    lambda checked, pid=panel_id: self.toggle_panel(pid)
                )
                panel_info.toolbar_action = action
                added_count += 1

    def _create_keyboard_shortcuts(self):
        """Create keyboard shortcuts for panel toggling."""
        for panel_info in self.get_all_panels():
            if panel_info.shortcut:
                action = QAction(f"Toggle {panel_info.name}", self.main_window)
                action.setShortcut(QKeySequence(panel_info.shortcut))
                action.triggered.connect(
                    lambda checked, pid=panel_info.panel_id: self.toggle_panel(pid)
                )
                self.main_window.addAction(action)
                self._shortcut_actions[panel_info.panel_id] = action

    def _update_ui_state(self, panel_info: PanelInfo):
        """Update menu and toolbar state for a panel."""
        visible = self.is_panel_visible(panel_info.panel_id)

        if panel_info.menu_action:
            panel_info.menu_action.setChecked(visible)

        if panel_info.toolbar_action:
            panel_info.toolbar_action.setChecked(visible)

    # =========================================================================
    # State Persistence
    # =========================================================================

    def save_workspace_state(self):
        """Save current workspace state to file."""
        try:
            state = {
                "panels": {},
                "timestamp": str(Path(__file__).stat().st_mtime)
            }

            for panel_info in self.get_all_panels():
                panel_state = {
                    "visible": self.is_panel_visible(panel_info.panel_id),
                    "floating": panel_info.is_floating,
                    "geometry": panel_info.geometry_data
                }
                state["panels"][panel_info.panel_id] = panel_state

            self._settings_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
            logger.debug("Workspace state saved")

        except Exception as e:
            logger.error(f"Failed to save workspace state: {e}")

    def _load_workspace_state(self):
        """Load workspace state from file."""
        try:
            if not self._settings_file.exists():
                return

            state = json.loads(self._settings_file.read_text(encoding="utf-8"))

            # Apply saved visibility states
            for panel_id, panel_state in state.get("panels", {}).items():
                panel_info = self.get_panel_info(panel_id)
                if panel_info:
                    visible = panel_state.get("visible", panel_info.default_visible)
                    if visible:
                        # Defer showing to avoid UI conflicts during startup
                        QTimer.singleShot(100, lambda pid=panel_id: self.show_panel(pid))

        except Exception as e:
            logger.warning(f"Failed to load workspace state: {e}")

    def _save_panel_state(self, panel_info: PanelInfo):
        """Save state for a specific panel."""
        if panel_info.dock_widget:
            try:
                geometry = panel_info.dock_widget.saveGeometry()
                panel_info.geometry_data = bytes(geometry.toHex()).decode("ascii")
                panel_info.is_floating = panel_info.dock_widget.isFloating()
            except Exception as e:
                logger.debug(f"Failed to save geometry for {panel_info.panel_id}: {e}")

    def _restore_panel_state(self, panel_info: PanelInfo):
        """Restore state for a specific panel."""
        if panel_info.dock_widget and panel_info.geometry_data:
            try:
                geometry = bytes.fromhex(panel_info.geometry_data)
                panel_info.dock_widget.restoreGeometry(geometry)
                if panel_info.is_floating:
                    panel_info.dock_widget.setFloating(True)
            except Exception as e:
                logger.debug(f"Failed to restore geometry for {panel_info.panel_id}: {e}")

    # =========================================================================
    # Reset Layout Functionality
    # =========================================================================

    def reset_layout(self):
        """Reset to factory default layout."""
        try:
            # Hide all panels
            for panel_info in self.get_all_panels():
                if self.is_panel_visible(panel_info.panel_id):
                    self.hide_panel(panel_info.panel_id)

            # Clear saved state
            if self._settings_file.exists():
                self._settings_file.unlink()

            # Show default panels
            for panel_info in self.get_all_panels():
                if panel_info.default_visible:
                    self.show_panel(panel_info.panel_id)

            # Reset to default positions (this is complex, simplified for now)
            # In a full implementation, we'd have predefined dock positions

            self.workspace_reset.emit()
            logger.info("Workspace layout reset to factory defaults")

            self._update_status_bar("Workspace reset to factory layout")

        except Exception as e:
            logger.error(f"Failed to reset layout: {e}")
            QMessageBox.warning(
                self.main_window,
                "Reset Failed",
                f"Failed to reset workspace layout: {e}"
            )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_dock_visibility_changed(self, panel_info: PanelInfo, visible: bool):
        """Handle dock widget visibility changes."""
        panel_info.is_visible = visible
        self._update_ui_state(panel_info)

        if visible:
            self.panel_opened.emit(panel_info.panel_id)
            logger.info(f"Panel dock visibility changed: {panel_info.panel_id} -> visible")
        else:
            self.panel_hidden.emit(panel_info.panel_id)
            logger.info(f"Panel dock visibility changed: {panel_info.panel_id} -> hidden")

    def _update_status_bar(self, message: str):
        """Update status bar with panel message."""
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar().showMessage(message, 3000)  # 3 second timeout

    # =========================================================================
    # Cleanup
    # =========================================================================

    def shutdown(self):
        """Clean shutdown - save state."""
        self.save_workspace_state()
        logger.info("PanelManager shutdown complete")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_visible_panels(self) -> List[str]:
        """Get list of currently visible panel IDs."""
        return [pid for pid in self._panels.keys() if self.is_panel_visible(pid)]

    def clear_all_visible_panels(self) -> int:
        """
        Clear the UI state of all visible panels.

        Calls clear_panel(), clear_ui(), or clear() on each visible panel
        to reset their UI elements to default states.

        Returns:
            Number of panels successfully cleared
        """
        cleared_count = 0
        for panel_id in self.get_visible_panels():
            panel_info = self.get_panel_info(panel_id)
            if panel_info and panel_info.instance:
                # Try methods in order of preference
                for method_name in ('clear_panel', 'clear_ui', 'clear'):
                    method = getattr(panel_info.instance, method_name, None)
                    if callable(method):
                        try:
                            method()
                            cleared_count += 1
                            logger.info(f"Cleared panel: {panel_id}")
                        except Exception as e:
                            logger.warning(f"Failed to clear panel {panel_id}: {e}")
                        break

        self._update_status_bar(f"Cleared {cleared_count} panels")
        return cleared_count

    def get_panel_count(self) -> int:
        """Get total number of registered panels."""
        return len(self._panels)

    def prevent_permanent_close(self, panel_id: str):
        """
        Ensure a panel can never be permanently closed.

        This method should be called by panels that want to enforce
        the hide-on-close behavior.
        """
        panel_info = self.get_panel_info(panel_id)
        if panel_info and panel_info.dock_widget:
            # Override close event to hide instead of destroy
            original_close_event = panel_info.dock_widget.closeEvent

            def hide_on_close(event):
                panel_info.dock_widget.hide()
                event.ignore()
                logger.debug(f"Prevented permanent close of panel: {panel_id}")

            panel_info.dock_widget.closeEvent = hide_on_close
