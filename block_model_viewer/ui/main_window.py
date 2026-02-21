"""
Refactored Main Application Window with Menu-Driven Architecture.
Professional, minimal, structured layout for GeoX.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

# Try importing Pandas for type checking and runtime usage
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

if TYPE_CHECKING:
    import pandas as pd
import time

from PyQt6.QtCore import (
    QProcess,
    QSettings,
    QSignalBlocker,
    Qt,
    QTimer,
)
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from .modern_styles import get_theme_colors, ModernColors

# Step 12: PyVista removed from UI - all visualization via Renderer
# Legacy PyVista imports below are only for backward compatibility in try/except blocks
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from ..controllers.app_controller import AppController
from ..core.process_history_tracker import get_process_history_tracker

# Legend Controller removed - feature was unstable
from ..models.block_model import BlockModel
from ..utils.coordinate_manager import CoordinateManager
from ..utils.desurvey import interpolate_at_depth, minimum_curvature_desurvey
from .axes_scalebar_panel import AxesScaleBarPanel
from .block_resource_panel import BlockModelResourcePanel
from .bookmarks import BookmarkManager
from .charts_panel import ChartsPanel
from .cross_section_manager_panel import CrossSectionManagerPanel
from .data_registry_status_panel import DataRegistryStatusPanel
from .data_viewer_panel import DataViewerPanel
from .dialogs import DialogManager
from .dialogs.modern_search_dialog import ModernSearchDialog

from .drillhole_control_panel import DrillholeControlPanel
from .drillhole_import_panel import DrillholeImportPanel
from .drillhole_plotting_panel import DrillholePlottingPanel
from .drillhole_reporting_panel import DrillholeReportingPanel
from .esg_dashboard_panel import ESGDashboardPanel
from .gc_decision_panel import GCDecisionPanel
from .grade_tonnage_panel import GradeTonnagePanel
from .grade_tonnage_basic_panel import GradeTonnageBasicPanel
from .cutoff_optimization_panel import CutoffOptimizationPanel
from .variogram_panel import VariogramAnalysisPanel
from .swath_analysis_3d_panel import SwathAnalysis3DPanel
from .loopstructural_panel import LoopStructuralModelPanel
from .compositing_window import CompositingWindow
from .cross_section_panel import CrossSectionPanel
from .block_info_panel import BlockInfoPanel
from .grade_transformation_panel import GradeTransformationPanel
from .interaction import InteractionController
from .irr_panel import IRRPanel
from .jorc_classification_panel import JORCClassificationPanel
from .kmeans_clustering_panel import KMeansClusteringPanel
from .panel_manager import PanelManager
from .panel_registration import register_all_panels
from .panel_registry import get_panel_registry
from .persistent_dock import PersistentDockWidget
from .preferences_dialog import PreferencesDialog
from .project_loading_dialog import ProjectLoadingDialog
from .property_panel import PropertyPanel
from .qc_window import QCWindow
from .resource_classification_panel import JORCClassificationPanel as ResourceClassificationPanel
from .resource_reporting_panel import ResourceReportingPanel
from .block_property_calculator_panel import BlockPropertyCalculatorPanel
from .screenshot_export_dialog import ScreenshotExportDialog
from .data_export_dialog import DataExportDialog
from .selection_panel import SelectionPanel
from .shortcuts import Shortcuts
from .signals import UISignals

# DrillholeResourcePanel removed - functionality redundant with BlockModelResourcePanel
from .statistics_panel import StatisticsPanel
from .status import StatusManager
from .swath_panel import SwathPanel
from .table_viewer_panel import TableViewerPanel

# Removed: standalone_resource_calculator_panel.py (redundant, use BlockModelResourcePanel instead)
from .theme_manager import ThemeManager
from .underground_panel import UndergroundPanel
from .viewer_widget import ViewerWidget

# Icon loader for menu icons
try:
    from ..assets.icons.icon_loader import IconCategory, get_menu_icon
    ICONS_AVAILABLE = True
except ImportError:
    ICONS_AVAILABLE = False
    def get_menu_icon(category, name):
        return QIcon()
    class IconCategory:
        pass

logger = logging.getLogger(__name__)


class AxesScaleBarDialog(QDialog):
    """Floating window that hosts the Floating Axes & Scale Bar controls."""

    def __init__(self, parent: Optional[QWidget], action: QAction, controller: Optional["AppController"]):
        super().__init__(parent, Qt.WindowType.Tool)
        self._action = action
        self.setWindowTitle("Floating Axes & Scale Bar")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._panel = AxesScaleBarPanel(self)
        layout.addWidget(self._panel)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowModality(Qt.WindowModality.NonModal)

        if controller and self._panel:
            try:
                self._panel.bind_controller(controller)
            except Exception:
                logger.debug("Failed to bind controller to axes scalebar dialog", exc_info=True)



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
    def showEvent(self, event):
        logger.debug(f"{self.__class__.__name__} dialog shown")
        if self._panel:
            self._panel.refresh()
        super().showEvent(event)

    def closeEvent(self, event):
        logger.debug(f"{self.__class__.__name__} dialog closed")
        if self._action:
            self._action.setChecked(False)
        super().closeEvent(event)


# FileLoadThread removed - logic moved to DataController._prepare_load_file_payload
# File loading now uses controller.run_task('load_file')


def _safe_get_block_count(block_model) -> int:
    """
    Safely get block count from either BlockModel object or DataFrame.

    Args:
        block_model: BlockModel object or pandas DataFrame

    Returns:
        Number of blocks/rows

    Raises:
        ValueError: If block_model is None or unsupported type
    """
    if block_model is None:
        raise ValueError("block_model is None")

    # Check if it's a BlockModel object
    if hasattr(block_model, 'block_count'):
        return block_model.block_count

    # Check if it's a DataFrame
    if isinstance(block_model, pd.DataFrame):
        return len(block_model)

    # Unsupported type
    raise ValueError(f"Unsupported block_model type: {type(block_model)}")


class MainWindow(QMainWindow):
    """
    Refactored Main Window with menu-driven architecture.
    Clean, professional layout with dockable panels.
    """

    def __init__(self, registry=None):
        """
        Initialize MainWindow.
        
        Args:
            registry: Optional DataRegistry instance (for dependency injection).
                     If None, falls back to singleton pattern for backward compatibility.
        """
        super().__init__()
        self.setWindowTitle("GeoX")
        self.resize(1600, 1000)

        # Enable dock widget features for tabbing
        self.setDockOptions(
            QMainWindow.DockOption.AllowTabbedDocks |
            QMainWindow.DockOption.AllowNestedDocks |
            QMainWindow.DockOption.AnimatedDocks
        )
        # Set default tab position for all dock areas
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)

        # Track window state for logging
        self._last_logged_size = None

        # Store registry for dependency injection
        self._registry = registry

        # Data
        self.current_model: Optional[BlockModel] = None
        self.current_file_path: Optional[Path] = None
        self.current_project_path: Optional[Path] = None
        self.config = Config()

        # Project-level settings (for schema caching, etc.)
        self.project_settings: Dict[str, Any] = {}

        # Coordinate system manager for aligning datasets
        self.coordinate_manager = CoordinateManager()

        # Controller and Signals (NEW - Architecture Refactor)
        self.signals: Optional[UISignals] = None
        self.controller: Optional[AppController] = None
        self.interaction: Optional[InteractionController] = None
        self.dialogs: Optional[DialogManager] = None
        self.bookmarks: Optional[BookmarkManager] = None
        self.status: Optional[StatusManager] = None

        # UI Components
        self.viewer_widget: Optional[ViewerWidget] = None
        self.property_panel: Optional[PropertyPanel] = None
        self.scene_dock: Optional[QDockWidget] = None
        self.block_info_dock: Optional[QDockWidget] = None

        # Right dock tabs
        self.right_tab_widget: Optional[QTabWidget] = None
        self.gc_decision_panel: Optional[GCDecisionPanel] = None

        # Resource panels (popup windows)
        self.block_resource_panel: Optional[BlockModelResourcePanel] = None
        self.block_resource_dialog: Optional[QDialog] = None
        # DrillholeResourcePanel removed - functionality redundant with BlockModelResourcePanel
        self.irr_panel: Optional[IRRPanel] = None
        self.irr_dialog: Optional[QDialog] = None
        self.kmeans_panel: Optional[KMeansClusteringPanel] = None
        self.kmeans_dialog: Optional[QDialog] = None
        self.resource_classification_panel: Optional[ResourceClassificationPanel] = None
        self.resource_classification_dialog: Optional[QDialog] = None
        self.resource_reporting_panel: Optional[ResourceReportingPanel] = None
        self.resource_reporting_dialog: Optional[QDialog] = None
        # Removed: standalone_resource_calculator_panel (redundant, use block_resource_panel instead)

        # Drillhole panel registry (persistent docks)
        self._drillhole_panel_registry: Dict[str, "PersistentDockWidget"] = {}
        # Removed: standalone_resource_calculator_dialog (redundant)
        self.grade_tonnage_panel: Optional['GradeTonnagePanel'] = None
        self.grade_tonnage_dialog: Optional[QDialog] = None
        self.grade_tonnage_basic_panel: Optional['GradeTonnageBasicPanel'] = None
        self.grade_tonnage_basic_dialog: Optional[QDialog] = None
        self.cutoff_optimization_panel: Optional['CutoffOptimizationPanel'] = None
        self.cutoff_optimization_dialog: Optional[QDialog] = None
        self.pit_optimisation_dialog: Optional[QDialog] = None
        self.geotech_dialog: Optional[QDialog] = None
        self.variogram_dialog: Optional["VariogramAnalysisPanel"] = None
        self.variogram_panel: Optional["VariogramAnalysisPanel"] = None
        self.variogram_assistant_dialog: Optional[QDialog] = None
        self.soft_kriging_dialog: Optional[QDialog] = None
        self.ik_sgsim_dialog: Optional[QDialog] = None
        self.cosgsim_dialog: Optional[QDialog] = None
        self.sis_dialog: Optional[QDialog] = None
        self.turning_bands_dialog: Optional[QDialog] = None
        self.dbs_dialog: Optional[QDialog] = None
        self.mps_dialog: Optional[QDialog] = None
        self.grf_dialog: Optional[QDialog] = None
        self.uncertainty_propagation_dialog: Optional[QDialog] = None
        self.research_dashboard_dialog: Optional[QDialog] = None

        # Advanced features
        self.selection_panel: Optional[SelectionPanel] = None
        self.selection_dialog: Optional[QDialog] = None
        self.cross_section_manager_panel: Optional[CrossSectionManagerPanel] = None
        self.cross_section_manager_dialog: Optional[QDialog] = None

        # Track all open panels/dialogs for cleanup
        # Legacy list maintained for backward compatibility; DialogManager is authoritative
        self._open_panels: list[QDialog] = []

        # Data & Analysis panels (popup windows)
        self.statistics_panel: Optional[StatisticsPanel] = None
        self.statistics_dialog: Optional[QDialog] = None
        self.charts_panel: Optional[ChartsPanel] = None
        self.charts_dialog: Optional[QDialog] = None
        self.swath_panel: Optional[SwathPanel] = None
        self.swath_dialog: Optional[QDialog] = None
        self.swath_analysis_3d_panel: Optional['SwathAnalysis3DPanel'] = None
        self.swath_analysis_3d_dialog: Optional[QDialog] = None
        self.data_viewer_panel: Optional[DataViewerPanel] = None
        self.data_viewer_dialog: Optional[QDialog] = None

        # Drillhole panels (popup windows)
        self.domain_compositing_panel: Optional[DrillholeImportPanel] = None
        self.domain_compositing_dialog: Optional[QDialog] = None

        # LoopStructural Geological Modeling panel
        self.loopstructural_panel: Optional['LoopStructuralModelPanel'] = None
        self.loopstructural_dialog: Optional[QDialog] = None
        self.compositing_window: Optional['CompositingWindow'] = None
        self.cross_section_panel: Optional['CrossSectionPanel'] = None
        self.cross_section_dialog: Optional[QDialog] = None
        self.grade_transformation_panel: Optional[GradeTransformationPanel] = None
        self.grade_transformation_dialog: Optional[QDialog] = None

        # Drillhole panels
        self.drillhole_reporting_panel: Optional['DrillholeReportingPanel'] = None
        self.drillhole_reporting_dialog: Optional[QDialog] = None
        self.drillhole_plotting_panel: Optional['DrillholePlottingPanel'] = None
        self.drillhole_plotting_dialog: Optional[QDialog] = None

        self.data_registry_status_panel: Optional[DataRegistryStatusPanel] = None
        self.data_registry_status_dialog: Optional[QDialog] = None
        self.underground_panel_dialog: Optional[QDialog] = None  # Track underground panel dialog instance
        self.drillhole_control_panel: Optional[DrillholeControlPanel] = None
        self.drillhole_control_dock: Optional[QDockWidget] = None

        # Legend controller (popup window)
        # Legend Controller removed - feature was unstable

        # Screenshot manager
        from ..utils.screenshot_manager import ScreenshotManager
        self.screenshot_manager = ScreenshotManager()


        # Theme manager
        self.theme_manager = ThemeManager()
        self.theme_manager.set_application(QApplication.instance())

        # Panel registry and shortcuts (STEP 17)
        self.panel_registry = get_panel_registry()
        self.shortcuts = Shortcuts()

        # Panel manager - unified lifecycle system
        # All panel registrations happen in panel_registration.py
        self.panel_manager = PanelManager(self)
        register_all_panels(self.panel_manager)

        # Workspace layout state
        self.current_workspace_layout: str = "default"

        # Timers
        self.camera_update_timer: Optional[QTimer] = None
        self.status_update_timer: Optional[QTimer] = None
        self._autosave_timer: Optional[QTimer] = None
        self._dirty: bool = False
        self._drillhole_render_timer: Optional[QTimer] = None
        self._drillhole_render_start: float = 0.0

        # Background loading
        # FileLoadThread removed - now uses controller.run_task('load_file')

        # View bookmarks (camera positions) - initialized after viewer setup
        self.view_bookmarks: Dict[int, Dict[str, Any]] = {}

        # Enable drag and drop
        self.setAcceptDrops(True)
        # Pending renderer session state to apply after async load
        self._pending_session_state = None
        self._pending_drillhole_state = None
        self._pending_registry_models_state = None

        # Setup UI
        self._setup_ui()

        self._setup_docks()
        self._setup_toolbar()
        self._setup_menus()
        self._setup_status_bar()
        self._connect_signals()
        self._apply_styling()
        self._restore_state()

        # Fix geometry issues: allow MainWindow to shrink
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Diagnostic: find widgets with large minimum heights
        # Run this after window is shown to catch geometry issues
        def run_geometry_diagnostics():
            def dump_min(widget, name, indent=0):
                if widget is None:
                    return
                try:
                    min_size = widget.minimumSize()
                    min_hint = widget.minimumSizeHint()
                    obj_name = widget.objectName() or widget.__class__.__name__
                    if min_size.height() > 1000 or min_hint.height() > 1000:
                        prefix = "  " * indent
                        logger.warning(
                            f"{prefix}{name} ({obj_name}) min size = {min_size.width()}x{min_size.height()}, "
                            f"minHint = {min_hint.width()}x{min_hint.height()}"
                        )
                except Exception as e:
                    logger.debug(f"Error checking {name}: {e}")

            # Run diagnostics
            dump_min(self, "MAIN WINDOW")
            central = self.centralWidget()
            if central:
                dump_min(central, "CENTRAL WIDGET")
                for child in central.findChildren(QWidget):
                    dump_min(child, f"CHILD: {child.__class__.__name__}", indent=1)

            # Check all dock widgets
            for dock in self.findChildren(QDockWidget):
                dock_widget = dock.widget()
                if dock_widget:
                    dump_min(dock_widget, f"DOCK: {dock.objectName() or dock.windowTitle()}")

        # Run diagnostics after window is shown (deferred)
        QTimer.singleShot(500, run_geometry_diagnostics)

        # Defer session restore to after UI is up
        QTimer.singleShot(0, self._restore_session_on_startup)

        # Setup autosave every 2 minutes
        try:
            self._autosave_timer = QTimer(self)
            self._autosave_timer.timeout.connect(self._autosave_if_dirty)
            self._autosave_timer.start(120000)
        except Exception:
            pass

        logger.info("Initialized refactored main window")

    @property
    def registry(self):
        """
        Property to access the data registry.
        Returns the registry instance stored as self._registry.
        This property ensures backward compatibility with code that uses self.registry.
        """
        return self._registry

    # ================== STEP 40: Panel Creation Diagnostics ==================

    def _create_panel(self, panel_cls):
        """
        Create a panel with timing diagnostics (STEP 40).
        
        Args:
            panel_cls: Panel class to instantiate
            
        Returns:
            Panel instance
        """
        import time
        start = time.perf_counter()
        panel = panel_cls(parent=self)
        elapsed = time.perf_counter() - start
        panel_name = getattr(panel_cls, "PANEL_ID", panel_cls.__name__)
        if elapsed > 2.0:
            logger.warning("Panel %s took %.2fs to construct", panel_name, elapsed)
        return panel

    def _setup_ui(self):
        """Setup central widget (PyVista viewer only)."""
        # Central widget - PyVista 3D Viewer
        self.viewer_widget = ViewerWidget()
        self.setCentralWidget(self.viewer_widget)
        # Initialize Controller and Signals (NEW - Architecture Refactor)
        self.signals = UISignals()

        # Initialize InteractionController for mouse/camera operations
        self.interaction = InteractionController(
            viewer=self.viewer_widget,
            signals=self.signals,
            status_bar=None,  # Will be bound after status bar is created
            parent=self
        )

        # Initialize DialogManager for dialog lifecycle
        self.dialogs = DialogManager(parent=self)

        # Initialize BookmarkManager for view bookmarks
        self.bookmarks = BookmarkManager(viewer=self.viewer_widget, status_bar=None, parent=self)
        self.bookmarks.load_from_settings()
        # Maintain legacy reference for backward compatibility
        self.view_bookmarks = self.bookmarks.bookmarks

        # Initialize StatusManager for status bar setup and updates
        self.status = StatusManager(self, parent=self)

        self.controller = AppController(
            renderer=self.viewer_widget.renderer,
            config=self.config,
            registry=self._registry  # Dependency Injection
        )
        # Store viewer_widget reference for panel access (scale bar, north arrow widgets)
        self.controller.viewer_widget = self.viewer_widget

        # CRITICAL: Attach registry to renderer for persistent interval IDs (GPU picking stability)
        try:
            if hasattr(self.viewer_widget.renderer, 'attach_registry'):
                self.viewer_widget.renderer.attach_registry(self._registry)
                logger.info("Attached DataRegistry to renderer for stable GPU picking")
        except Exception as e:
            logger.warning(f"Could not attach registry to renderer: {e}")

        # Attach registry to legend manager for category label aliasing
        try:
            legend_mgr = getattr(self.controller, "legend_manager", None)
            if legend_mgr and hasattr(legend_mgr, 'attach_registry'):
                legend_mgr.attach_registry(self._registry)
                logger.info("Attached DataRegistry to legend manager for category label aliasing")
        except Exception as e:
            logger.debug(f"Could not attach registry to legend manager: {e}")

        # Mark project dirty when category labels change
        try:
            if hasattr(self._registry, 'categoryLabelMapsChanged'):
                self._registry.categoryLabelMapsChanged.connect(lambda ns: self._mark_dirty())
                logger.debug("Connected category label changes to project dirty flag")
        except Exception as e:
            logger.debug(f"Could not connect category label dirty marker: {e}")

        try:
            # Note: axis_manager is now unified into overlay_manager
            self.viewer_widget.bind_managers(
                getattr(self.controller, "legend_manager", None),
                getattr(self.controller, "overlay_manager", None),
            )
        except Exception:
            pass
        # Hook renderer state change callback (for undo/redo snapshots)
        try:
            if self.viewer_widget and self.viewer_widget.renderer:
                self.viewer_widget.renderer.set_state_change_callback(self._on_renderer_state_change)
        except Exception:
            pass

        # Connect signals to controller methods
        self._connect_signals_to_controller()

        # Step 11: Connect controller signals for unified pipeline
        self._connect_controller_signals()

        # Create property panel (will be added to bottom dock later)
        # Note: signals will be set after initialization
        self.property_panel = PropertyPanel(signals=self.signals)
        self.axes_scalebar_window: Optional[AxesScaleBarDialog] = None

        logger.info("Setup central viewer widget")

    def _setup_docks(self):
        """Setup all dockable panels.
        
        REFACTORED: Dock setup logic moved to ui/layout/dock_setup.py
        """
        from .layout.dock_setup import setup_docks
        setup_docks(self)

    def _setup_dock_hide_on_close(self, dock: QDockWidget):
        """Configure a dock widget to hide on close instead of destroying."""
        if not dock:
            return

        # Override closeEvent to hide instead of close
        original_close_event = dock.closeEvent

        def hide_on_close(event):
            dock.hide()
            event.ignore()
            # Update menu action checked state
            self._update_dock_menu_state(dock)

        dock.closeEvent = hide_on_close

        # Connect visibility changes to update menu state
        dock.visibilityChanged.connect(lambda visible: self._update_dock_menu_state(dock))

    def _toggle_dock(self, dock: QDockWidget, show: bool):
        """Toggle dock widget visibility."""
        if not dock:
            return

        if show:
            dock.show()
            dock.raise_()
        else:
            dock.hide()

        self._update_dock_menu_state(dock)

    def _update_dock_menu_state(self, dock: QDockWidget):
        """Update menu action checked state based on dock visibility."""
        if not dock:
            return

        visible = dock.isVisible()

        # Map dock widgets to their menu actions (actions may not exist yet during init)
        dock_to_action = {
            self.left_dock: getattr(self, 'controls_scene_action', None),
            self.gc_decision_dock: getattr(self, 'gc_decision_action', None),
            self.drillhole_control_dock: getattr(self, 'drillhole_explorer_action', None),
        }

        action = dock_to_action.get(dock)
        if action:
            action.setChecked(visible)

    def _update_all_dock_menu_states(self):
        """Update all dock menu action checked states."""
        if hasattr(self, 'left_dock'):
            self._update_dock_menu_state(self.left_dock)
        if hasattr(self, 'gc_decision_dock'):
            self._update_dock_menu_state(self.gc_decision_dock)
        if hasattr(self, 'drillhole_control_dock'):
            self._update_dock_menu_state(self.drillhole_control_dock)

    def _setup_toolbar(self):
        """Setup modern toolbar with icons, dropdowns, and status strip.
        
        REFACTORED: Toolbar setup logic moved to ui/layout/dock_setup.py
        """
        from .layout.dock_setup import setup_toolbar
        setup_toolbar(self)

    def _setup_menus(self):
        """Setup comprehensive menu bar.
        
        REFACTORED: Menus are now built by separate modules in ui/menus/
        This keeps MainWindow maintainable and prevents it from growing.
        """
        from .menus import (
            build_dashboards_menu,
            build_data_menu,
            build_drillholes_menu,
            build_edit_menu,
            build_estimations_menu,
            build_file_menu,
            build_geology_menu,
            build_geotech_menu,
            build_help_menu,
            build_layout_menu,
            build_mine_planning_menu,
            build_ml_menu,
            build_mouse_menu,
            build_panels_menu,
            build_remote_sensing_menu,
            build_resources_menu,
            build_scan_menu,
            build_search_menu,
            build_survey_menu,
            build_tools_menu,
            build_view_menu,
            build_workbench_menu,
            build_workflows_menu,
        )

        menubar = self.menuBar()
        
        # Enable hover-to-open behavior for menus
        self._setup_menubar_hover(menubar)

        # REFACTORED: Build all menus using extracted modules
        build_file_menu(self, menubar)
        build_layout_menu(self, menubar)  # Reports menu - placed early for visibility
        build_search_menu(self, menubar)
        build_edit_menu(self, menubar)
        build_view_menu(self, menubar)
        build_scan_menu(self, menubar)
        build_remote_sensing_menu(self, menubar)
        build_survey_menu(self, menubar)
        build_tools_menu(self, menubar)
        build_panels_menu(self, menubar)

        # Tools menu needs axes/scalebar action added after build
        # (This is a special case that needs MainWindow state)
        tools_menu = None
        for menu in menubar.findChildren(QMenu):
            if menu.title() == "&Tools":
                tools_menu = menu
                break

        if tools_menu:
            tools_menu.addSeparator()
            self.axes_scalebar_action = QAction(get_menu_icon("tools", "axes_scale"), "Floating &Axes & Scale Bar...", self)
            self.axes_scalebar_action.setCheckable(True)
            self.axes_scalebar_action.setChecked(False)
            self.axes_scalebar_action.setStatusTip("Show floating axes & scale bar controls")
            self.axes_scalebar_action.triggered.connect(self._toggle_axes_scalebar_window)
            tools_menu.addAction(self.axes_scalebar_action)
            self._create_axes_scalebar_window()

        build_data_menu(self, menubar)
        build_mouse_menu(self, menubar)
        build_drillholes_menu(self, menubar)
        build_geology_menu(self, menubar)
        build_resources_menu(self, menubar)
        build_estimations_menu(self, menubar)
        build_geotech_menu(self, menubar)
        build_mine_planning_menu(self, menubar)
        build_ml_menu(self, menubar)
        build_dashboards_menu(self, menubar)
        build_workbench_menu(self, menubar)
        build_workflows_menu(self, menubar)
        build_help_menu(self, menubar)

        # Initialize PanelManager UI (menu, toolbar, shortcuts)
        self.panel_manager.initialize_ui()

        logger.info("Setup comprehensive menu bar")

    def _setup_menubar_hover(self, menubar):
        """Enable hover-to-open behavior for top-level menu items.
        
        When hovering over menu bar items, the corresponding submenu opens automatically
        after a brief delay. Once any menu is open, hovering switches menus instantly.
        """
        from PyQt6.QtCore import QEvent, QObject, QTimer
        from PyQt6.QtWidgets import QMenuBar
        
        class MenuBarHoverFilter(QObject):
            """Event filter to enable hover-to-open on menu bar."""
            
            def __init__(self, menubar: QMenuBar):
                super().__init__(menubar)
                self.menubar = menubar
                self._hover_timer = QTimer()
                self._hover_timer.setSingleShot(True)
                self._hover_timer.setInterval(150)  # 150ms delay before opening
                self._hover_timer.timeout.connect(self._open_hovered_menu)
                self._pending_action = None
            
            def _open_hovered_menu(self):
                """Open the menu that's being hovered over."""
                if self._pending_action and self._pending_action.menu():
                    self.menubar.setActiveAction(self._pending_action)
            
            def eventFilter(self, obj, event):
                if obj == self.menubar:
                    if event.type() == QEvent.Type.MouseMove:
                        # Get the action under the cursor
                        action = self.menubar.actionAt(event.pos())
                        if action and action.menu():
                            active_action = self.menubar.activeAction()
                            if active_action and active_action.menu() and active_action.menu().isVisible():
                                # A menu is already open - switch immediately on hover
                                if active_action != action:
                                    self.menubar.setActiveAction(action)
                                    self._hover_timer.stop()
                            else:
                                # No menu open yet - start timer for delayed open
                                if self._pending_action != action:
                                    self._pending_action = action
                                    self._hover_timer.start()
                        else:
                            # Not hovering over a menu item
                            self._hover_timer.stop()
                            self._pending_action = None
                    elif event.type() == QEvent.Type.Leave:
                        self._hover_timer.stop()
                        self._pending_action = None
                return False
        
        # Install the event filter
        self._menubar_hover_filter = MenuBarHoverFilter(menubar)
        menubar.installEventFilter(self._menubar_hover_filter)

    def open_preferences(self):
        try:
            dialog = PreferencesDialog(self)
            self._setup_dialog_persistence(dialog, 'preferences_dialog')
            dialog.exec()
        except Exception as e:
            logger.error(f"Failed to open Preferences: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Preferences:\n{e}")

    # ==========================
    # Mouse/Interaction handlers
    # ==========================
    def _on_renderer_layers_changed(self):
        """Fan-out renderer layer change events to all interested panels."""
        logger.debug("_on_renderer_layers_changed called")

        try:
            if self.property_panel:
                self.property_panel.update_layer_controls()
                logger.debug("PropertyPanel layer controls updated")
        except Exception as e:
            logger.warning(f"Error updating PropertyPanel: {e}")

        # Also update scene inspector if available
        try:
            if hasattr(self, 'scene_inspector_panel') and self.scene_inspector_panel:
                if hasattr(self.scene_inspector_panel, 'update_layer_controls'):
                    self.scene_inspector_panel.update_layer_controls()
        except Exception:
            pass

        # CRITICAL FIX: Update app state when layers change
        # This triggers the state transition to RENDERED which enables UI controls
        try:
            if self.controller:
                self.controller._update_state_from_scene()
        except Exception:
            pass

        # Auto-show multi-legend when drillholes or block model loads
        try:
            self._auto_show_multi_legend_if_needed()
        except Exception as e:
            logger.debug(f"Auto-show multi-legend check failed: {e}")

    def _auto_show_multi_legend_if_needed(self):
        """
        Automatically show multi-legend and add elements when data loads.

        Called from _on_renderer_layers_changed to improve discoverability.
        Only triggers once per layer to avoid duplicates.
        """
        if not self.viewer_widget or not hasattr(self.viewer_widget, 'renderer'):
            logger.debug("Auto-legend: No viewer_widget or renderer")
            return

        renderer = self.viewer_widget.renderer
        if not hasattr(renderer, 'active_layers') or not renderer.active_layers:
            logger.debug("Auto-legend: No active_layers")
            return

        logger.debug(f"Auto-legend: Checking {len(renderer.active_layers)} active layers")

        # Track which layers we've already auto-added
        if not hasattr(self, '_auto_legend_layers'):
            self._auto_legend_layers = set()

        # Check for new drillhole or block model layers
        new_layers_to_add = []
        for layer_name, layer_info in renderer.active_layers.items():
            # Skip if we've already processed this layer
            if layer_name in self._auto_legend_layers:
                continue

            layer_type = layer_info.get('type', '').lower()
            layer_name_lower = layer_name.lower()

            # Check for drillholes
            if 'drillhole' in layer_type or 'drillholes' in layer_name_lower:
                new_layers_to_add.append((layer_name, None))  # No property for drillholes
                self._auto_legend_layers.add(layer_name)

            # Check for block models
            elif 'block' in layer_type or 'block model' in layer_name_lower:
                # Extract property from layer name if present (e.g., "Block Model: Au")
                property_name = None
                if ':' in layer_name:
                    property_name = layer_name.split(':', 1)[1].strip()
                new_layers_to_add.append((layer_name, property_name))
                self._auto_legend_layers.add(layer_name)

            # Check for geology/surface layers
            elif 'geology' in layer_type or 'surface' in layer_type:
                new_layers_to_add.append((layer_name, None))
                self._auto_legend_layers.add(layer_name)

        if not new_layers_to_add:
            logger.debug("Auto-legend: No new layers to add")
            return

        logger.info(f"Auto-legend: Found {len(new_layers_to_add)} new layers to add: {[l[0] for l in new_layers_to_add]}")

        # Enable multi-legend mode if not already visible
        if not self.viewer_widget.is_multi_legend_visible():
            self.viewer_widget.toggle_multi_legend(True)
            # Update menu action state
            if hasattr(self, 'multi_legend_action'):
                self.multi_legend_action.setChecked(True)
            if hasattr(self, 'classic_legend_action'):
                self.classic_legend_action.setChecked(False)
            logger.info("Auto-enabled multi-legend mode for loaded data")

        # Add legend elements for new layers
        legend_manager = None
        if hasattr(self.viewer_widget, '_legend_manager'):
            legend_manager = self.viewer_widget._legend_manager
        elif self.controller and hasattr(self.controller, 'legend_manager'):
            legend_manager = self.controller.legend_manager

        if legend_manager:
            for layer_name, property_name in new_layers_to_add:
                try:
                    element_id = legend_manager.add_legend_for_layer(layer_name, property_name)
                    if element_id:
                        logger.info(f"Auto-added legend element: {element_id}")
                except Exception as e:
                    logger.debug(f"Failed to auto-add legend for {layer_name}: {e}")

    # =========================================================================
    # MOUSE/INTERACTION MODE METHODS (Delegated to InteractionController)
    # =========================================================================
    # These methods delegate to self.interaction (InteractionController).
    # They are kept for backward compatibility with existing code that calls
    # MainWindow methods directly. New code should use self.interaction directly.
    # =========================================================================

    def set_mouse_mode_select(self) -> None:
        """Enable selection/clicking mode. Delegates to InteractionController."""
        if self.interaction is not None:
            self.interaction.set_mode_select()

    def set_mouse_mode_pan(self) -> None:
        """Enable pan mode. Delegates to InteractionController."""
        if self.interaction is not None:
            self.interaction.set_mode_pan()

    def set_mouse_mode_reset(self) -> None:
        """Restore original mouse mode. Delegates to InteractionController."""
        if self.interaction is not None:
            self.interaction.set_mode_reset()

    def set_mouse_mode_zoom_box(self) -> None:
        """Enable zoom box mode. Delegates to InteractionController."""
        if self.interaction is not None:
            self.interaction.set_mode_zoom_box()

    def update_mouse_action_checks(self, mode: Optional[str], show_message: bool = True) -> None:
        """
        Update mouse action checked states. Delegates to InteractionController.
        
        Args:
            mode: normalized mode string (e.g., 'select', 'pan', 'zoom_box', 'original')
            show_message: whether to show a brief statusBar message when updating
        """
        if self.interaction is not None:
            self.interaction.update_action_checks(mode, show_message)

    def _camera_zoom(self, factor: float) -> None:
        """Direct camera zoom. Delegates to InteractionController."""
        if self.interaction is not None:
            self.interaction.zoom(factor)

    def zoom_in(self) -> None:
        """Zoom in. Delegates to InteractionController."""
        if self.interaction is not None:
            self.interaction.zoom_in()

    def zoom_out(self) -> None:
        """Zoom out. Delegates to InteractionController."""
        if self.interaction is not None:
            self.interaction.zoom_out()

    def _create_axes_scalebar_window(self) -> None:
        """Create the floating axes/scale bar dialog."""
        if self.axes_scalebar_window is not None:
            return
        controller = getattr(self, "controller", None)
        self.axes_scalebar_window = AxesScaleBarDialog(self, self.axes_scalebar_action, controller)
        self.axes_scalebar_window.hide()

    def _toggle_axes_scalebar_window(self, visible: bool) -> None:
        """Show or hide the floating axes/scale bar dialog."""
        if visible and self.axes_scalebar_window is None:
            self._create_axes_scalebar_window()
        if self.axes_scalebar_window is None:
            return
        if visible:
            self.axes_scalebar_window.show()
            self.axes_scalebar_window.raise_()
        else:
            self.axes_scalebar_window.hide()

    def _setup_status_bar(self):
        """Setup enhanced status bar with multiple sections."""
        if self.status is None:
            self.status = StatusManager(self, parent=self)
        self.status.setup()

        # Preserve existing attributes for compatibility
        self.status_bar = self.status.status_bar

        # Bind status bar to managers
        if self.interaction is not None and self.status_bar is not None:
            self.interaction.bind_status_bar(self.status_bar)
        if self.bookmarks is not None and self.status_bar is not None:
            self.bookmarks.bind_status_bar(self.status_bar)

    def _update_status_progress(self, message: str, fraction: Optional[float] = None) -> None:
        """Update status bar message with optional progress."""
        if self.status is not None:
            self.status.update_progress(message, fraction)

    def _finish_status_progress(self, message: str, timeout: int = 3000) -> None:
        """Finish a status task and hide progress indicator."""
        if self.status is not None:
            self.status.finish_progress(message, timeout)

    def _connect_signals(self):
        """Connect all signals between components."""
        # Property panel to viewer (handles both block models and drillholes)
        if self.property_panel and self.viewer_widget:
            # Connect property panel to renderer for layer control
            self.property_panel.set_renderer(self.viewer_widget.renderer)

            # CRITICAL FIX: Connect property panel's request_visualization signal
            # This is emitted when switching to cached block model layers (SGSIM, Kriging, Classification)
            # that are not currently active in the renderer. Without this connection, layer switching fails silently.
            if hasattr(self.property_panel, 'request_visualization'):
                self.property_panel.request_visualization.connect(
                    self._handle_property_panel_visualization_request
                )
                logger.info("Connected property_panel.request_visualization signal to handler")

            try:
                if self.controller:
                    self.property_panel.bind_controller(self.controller)
            except Exception:
                pass

        # Drillhole control panel to viewer
        if self.drillhole_control_panel and self.viewer_widget:
            # Connect drillhole control panel to renderer for update button state
            self.drillhole_control_panel.set_renderer(self.viewer_widget.renderer)
            # Combine renderer layer-change updates for multiple panels
            try:
                if self.viewer_widget and self.viewer_widget.renderer:
                    # Set up layer change callback that handles both layer changes and drillhole selection
                    # NOTE: Renderer calls this with no args for layer changes, or with (event_type, data) for selections
                    # CRITICAL FIX: Store callback as instance variable to prevent garbage collection
                    # when stored as weakref in viewer_widget._layer_callback_refs
                    def layer_change_handler(event_type: str = None, data: Dict[str, Any] = None):
                        if event_type == "drillhole_selected":
                            self._on_drillhole_interval_selected(data)
                        else:
                            # Generic layer change - update UI controls and app state
                            self._on_renderer_layers_changed()
                    self._layer_change_handler = layer_change_handler  # Keep strong reference!
                    self.viewer_widget.renderer.set_layer_change_callback(self._layer_change_handler)
            except Exception:
                pass

            # Connect via UISignals bus instead of direct panel signals
            if self.signals:
                self.signals.propertyChanged.connect(self.on_property_changed)
                self.signals.colormapChanged.connect(self.on_colormap_changed)
                self.signals.colorModeChanged.connect(self.viewer_widget.set_color_mode)
                self.signals.filterChanged.connect(self.viewer_widget.apply_property_filter)
                self.signals.sliceChanged.connect(self.viewer_widget.apply_spatial_slice)
                self.signals.transparencyChanged.connect(self.viewer_widget.set_transparency)
                self.signals.blockSizeChanged.connect(self.viewer_widget.override_block_size)
                self.signals.legendSettingsChanged.connect(self.viewer_widget.set_legend_settings)
                self.signals.legendStyleChanged.connect(self.viewer_widget.update_legend_style)

            # Connect legend widget's colormap_changed signal to update property panel
            if hasattr(self.viewer_widget, '_legend_manager') and self.viewer_widget._legend_manager:
                if hasattr(self.viewer_widget._legend_manager, 'widget') and self.viewer_widget._legend_manager.widget:
                    try:
                        legend_widget = self.viewer_widget._legend_manager.widget
                        # Connect legend widget signal to UISignals bus
                        if self.signals:
                            legend_widget.colormap_changed.connect(self.signals.legendColormapChanged.emit)
                            self.signals.legendColormapChanged.connect(self._on_legend_colormap_changed)
                        else:
                            legend_widget.colormap_changed.connect(self._on_legend_colormap_changed)
                        logger.info("Connected legend colormap_changed signal to property panel")
                        # Connect mode_changed signal for menu item sync
                        legend_widget.mode_changed.connect(self._on_legend_mode_changed)
                        logger.info("Connected legend mode_changed signal for menu sync")
                    except Exception as e:
                        logger.warning(f"Could not connect legend colormap_changed to property panel: {e}")

        # Display settings to viewer
        # Display Settings signals are now handled by PropertyPanel's embedded display settings group
        # No additional connections needed here - PropertyPanel manages its own display settings

        # Scene inspector to viewer/main window - Connect via UISignals bus
        if self.signals:
            self.signals.resetViewRequested.connect(self.reset_camera)
            self.signals.fitViewRequested.connect(self.fit_to_view)
            self.signals.viewPresetRequested.connect(self.set_view_preset)
            self.signals.projectionToggled.connect(self.on_projection_toggled)
            self.signals.scalarBarToggled.connect(self.on_scalar_bar_toggled)
            # Note: axesToggled and gridToggled are handled by controller if needed

        # Selection - Connect via UISignals bus
        if self.signals:
            self.signals.clearSelectionRequested.connect(self.on_clear_block_selection)

        # Drillhole control panel - Connect via UISignals bus
        if self.signals:
            self.signals.drillholePlotRequested.connect(self._on_drillhole_control_plot)
            self.signals.drillholeClearRequested.connect(self._on_drillhole_control_clear)
            self.signals.drillholeRadiusChanged.connect(self._on_drillhole_radius_changed)
            self.signals.drillholeColorModeChanged.connect(self._on_drillhole_color_mode_changed)
            self.signals.drillholeAssayFieldChanged.connect(self._on_drillhole_assay_field_changed)
            self.signals.drillholeShowIdsToggled.connect(self._on_drillhole_show_ids_toggled)
            self.signals.drillholeVisibilityChanged.connect(self._on_drillhole_visibility_changed)
            self.signals.drillholeFocusRequested.connect(self._on_drillhole_focus_requested)
            self.signals.drillholeLithFilterChanged.connect(self._on_drillhole_lith_filter_changed)

        # Geological Explorer panel - Connect via UISignals bus
        if self.signals:
            if hasattr(self.signals, 'geologyRenderModeChanged'):
                self.signals.geologyRenderModeChanged.connect(self._on_geology_render_mode_changed)
            if hasattr(self.signals, 'geologyContactsVisibilityChanged'):
                self.signals.geologyContactsVisibilityChanged.connect(self._on_geology_contacts_visibility_changed)
            if hasattr(self.signals, 'geologySurfacesVisibilityChanged'):
                self.signals.geologySurfacesVisibilityChanged.connect(self._on_geology_surfaces_visibility_changed)
            if hasattr(self.signals, 'geologyMisfitVisibilityChanged'):
                self.signals.geologyMisfitVisibilityChanged.connect(self._on_geology_misfit_visibility_changed)
            if hasattr(self.signals, 'geologyFormationFilterChanged'):
                self.signals.geologyFormationFilterChanged.connect(self._on_geology_formation_filter_changed)
            if hasattr(self.signals, 'geologyOpacityChanged'):
                self.signals.geologyOpacityChanged.connect(self._on_geology_opacity_changed)
            if hasattr(self.signals, 'geologyColorPaletteChanged'):
                self.signals.geologyColorPaletteChanged.connect(self._on_geology_color_palette_changed)
            if hasattr(self.signals, 'geologyResetViewRequested'):
                self.signals.geologyResetViewRequested.connect(self._on_geology_reset_view)
            if hasattr(self.signals, 'geologyClearRequested'):
                self.signals.geologyClearRequested.connect(self._on_geology_clear)
            # New individual layer control signals
            if hasattr(self.signals, 'geologyLayerVisibilityChanged'):
                self.signals.geologyLayerVisibilityChanged.connect(self._on_geology_layer_visibility_changed)
            if hasattr(self.signals, 'geologyWireframeToggled'):
                self.signals.geologyWireframeToggled.connect(self._on_geology_wireframe_toggled)
            if hasattr(self.signals, 'geologySolidsOpacityChanged'):
                self.signals.geologySolidsOpacityChanged.connect(self._on_geology_solids_opacity_changed)
            if hasattr(self.signals, 'geologyViewModeChanged'):
                self.signals.geologyViewModeChanged.connect(self._on_geology_view_mode_changed)

        # Viewer signals - Connect viewer signals to UISignals bus
        if self.viewer_widget and self.signals:
            self.viewer_widget.block_picked.connect(self.signals.blockPicked.emit)
            self.signals.blockPicked.connect(self.on_block_picked)

            # Connect Block Info Panel to picking signals
            try:
                block_info_panel = self.panel_manager.get_panel_instance("BlockInfoPanel")
                if block_info_panel:
                    # Connect block_picked signal to update_block_info method
                    if hasattr(block_info_panel, 'update_block_info'):
                        self.viewer_widget.block_picked.connect(block_info_panel.update_block_info)
                        logger.info("Connected viewer_widget.block_picked to BlockInfoPanel.update_block_info")
                    # Connect global_pick_event signal for drillholes and geology
                    if hasattr(self.viewer_widget, 'global_pick_event') and hasattr(block_info_panel, 'on_pick_event'):
                        self.viewer_widget.global_pick_event.connect(block_info_panel.on_pick_event)
                        logger.info("Connected viewer_widget.global_pick_event to BlockInfoPanel.on_pick_event")
                else:
                    logger.warning("BlockInfoPanel instance not available for signal connection")
            except Exception as e:
                logger.warning(f"Failed to connect BlockInfoPanel signals: {e}")

            # Keep the Mouse menu in sync with the viewer's active interaction mode
            try:
                if hasattr(self.viewer_widget, 'mouse_mode_changed'):
                    self.viewer_widget.mouse_mode_changed.connect(self.signals.mouseModeChanged.emit)
                    self.signals.mouseModeChanged.connect(self.update_mouse_action_checks)
            except Exception:
                pass

        # Cross-section panels - Connect via UISignals bus
        if self.signals:
            self.signals.crossSectionUpdated.connect(self._on_cross_section_updated)
            self.signals.crossSectionPropertyChanged.connect(self._on_cross_section_property_changed)
            self.signals.crossSectionManagerRenderRequested.connect(self._on_cross_section_manager_render)
            self.signals.crossSectionManagerSectionCreated.connect(self._on_cross_section_manager_section_created)
            self.signals.crossSectionManagerSectionDeleted.connect(self._on_cross_section_manager_section_deleted)

        logger.info("Connected all component signals")

    # =========================================================================
    # Cross-Section Signal Handlers
    # =========================================================================

    def _on_cross_section_updated(self):
        """Handle cross-section parameter updates."""
        logger.debug("Cross-section parameters updated")
        # Could trigger re-rendering or status updates here

    def _on_cross_section_property_changed(self, property_name: str):
        """Handle cross-section property changes."""
        logger.info(f"Cross-section property changed to: {property_name}")
        # Could update visualization or trigger re-rendering

    def _on_cross_section_manager_render(self, section_name: str, property_name: str):
        """Handle cross-section manager render requests."""
        logger.info(f"Rendering cross-section '{section_name}' with property '{property_name}'")
        # This would integrate with the visualization system to render the section

    def _on_cross_section_manager_section_created(self, section_name: str, section_spec: dict):
        """Handle cross-section manager section creation."""
        logger.info(f"Cross-section '{section_name}' created: {section_spec}")
        # Could update UI or save to project

    def _on_cross_section_manager_section_deleted(self, section_name: str):
        """Handle cross-section manager section deletion."""
        logger.info(f"Cross-section '{section_name}' deleted")
        # Could update UI or remove from project

    # NOTE: _on_renderer_layers_changed is defined above in the Mouse/Interaction handlers section

    def _on_drillhole_interval_selected(self, data: dict):
        """Handle drillhole interval selection from renderer."""
        logger.debug(f"Drillhole interval selected: {data}")
        try:
            hole_id = data.get("hole_id")
            if hole_id and hasattr(self, 'drillhole_control_panel'):
                # Could highlight in drillhole panel or show details
                pass
        except Exception as e:
            logger.warning(f"Error handling drillhole selection: {e}")

    def _connect_signals_to_controller(self):
        """Connect UI signals to AppController methods (NEW - Architecture Refactor)."""
        if not self.signals or not self.controller:
            logger.warning("Signals or controller not initialized, skipping signal connections")
            return

        # Property and visualization signals
        # Step 12: Use standardized API name
        self.signals.propertySelected.connect(self.controller.set_active_property)
        self.signals.colormapChanged.connect(self.controller.set_colormap)
        # Note: transparency_changed signal removed (redundant with opacity)

        # Slicing and filtering
        self.signals.sliceChanged.connect(self.controller.apply_slice)
        self.signals.applyFilters.connect(self.controller.apply_filters)

        # Export
        self.signals.exportScreenshot.connect(self.controller.export_screenshot)

        # Data loading
        self.signals.blockModelLoaded.connect(self.controller.load_block_model)


        logger.info("Connected UI signals to AppController")

    def _connect_controller_signals(self):
        """Connect controller signals for unified pipeline - Step 11."""
        if not self.controller:
            return

        # Connect scene updates
        self.controller.signals.scene_updated.connect(self._on_scene_updated)
        self.controller.signals.block_model_changed.connect(self._on_block_model_changed)

        # Connect task lifecycle signals
        self.controller.signals.task_started.connect(self._on_task_started)
        self.controller.signals.task_finished.connect(self._on_task_finished)
        self.controller.signals.task_error.connect(self._on_task_error)
        self.controller.signals.task_progress.connect(self._on_task_progress)

        # Connect app state changes for UI gating
        self.controller.signals.app_state_changed.connect(self._on_app_state_changed)

        logger.info("Connected controller signals for unified pipeline")

    def _on_scene_updated(self):
        """Handle scene update signal - Step 11."""
        if self.viewer_widget:
            try:
                self.viewer_widget.update()
            except Exception as exc:
                try:
                    exc_msg = str(exc)
                    logger.debug(f"Failed to update viewer widget: {exc_msg}")
                except Exception:
                    logger.debug("Failed to update viewer widget: <unprintable error>")

    def _on_block_model_changed(self):
        """Handle block model changed signal - Step 11."""
        # Refresh property panels
        if hasattr(self, 'property_panel') and self.property_panel:
            try:
                self.property_panel.refresh()
            except Exception as exc:
                try:
                    exc_msg = str(exc)
                    logger.debug(f"Failed to refresh property panel: {exc_msg}")
                except Exception:
                    logger.debug("Failed to refresh property panel: <unprintable error>")

    def _on_task_started(self, task: str):
        """Handle task started signal - Step 11."""
        logger.debug(f"Task '{task}' started")
        # Could show status bar message here

    def _on_task_finished(self, task: str):
        """Handle task finished signal - Step 11."""
        logger.debug(f"Task '{task}' finished")
        # Could show status bar message here

    def _on_task_error(self, task: str, error_msg: str):
        """Handle task error signal - Step 11."""
        logger.error(f"Task '{task}' error: {error_msg}")
        QMessageBox.critical(
            self,
            f"Task Error: {task}",
            f"The task '{task}' encountered an error:\n\n{error_msg}"
        )

    def _on_task_progress(self, task: str, progress: float):
        """Handle task progress signal - Step 11."""
        # Progress is 0.0 to 1.0
        logger.debug(f"Task '{task}' progress: {progress:.1%}")
        # Could update progress dialog here if needed

    def _on_app_state_changed(self, state: int):
        """
        Handle application state changes - propagate to all UI panels.
        
        This is the central handler that ensures all panels react consistently
        to state changes. Panels must NOT infer state from data presence.
        
        Args:
            state: AppState enum value (as int for signal compatibility)
        """
        from ..controllers.app_state import AppState
        try:
            new_state = AppState(state)
            logger.info(f"MainWindow: App state changed to {new_state.name}")
        except ValueError:
            logger.warning(f"MainWindow: Invalid app state value: {state}")
            return

        # Propagate state change to all panels that support it
        panels_to_update = [
            ('property_panel', self.property_panel),
            ('scene_inspector_panel', getattr(self, 'scene_inspector_panel', None)),
            ('drillhole_control_panel', getattr(self, 'drillhole_control_panel', None)),
        ]

        for panel_name, panel in panels_to_update:
            if panel and hasattr(panel, 'on_app_state_changed'):
                try:
                    panel.on_app_state_changed(state)
                except Exception as e:
                    logger.debug(f"Failed to update {panel_name} state: {e}")

        # Update legend widget if it exists
        try:
            if self.viewer_widget and hasattr(self.viewer_widget, 'renderer'):
                renderer = self.viewer_widget.renderer
                if hasattr(renderer, 'legend_manager') and renderer.legend_manager:
                    legend_widget = getattr(renderer.legend_manager, '_widget', None)
                    if legend_widget and hasattr(legend_widget, 'on_app_state_changed'):
                        legend_widget.on_app_state_changed(state)
        except Exception as e:
            logger.debug(f"Failed to update legend widget state: {e}")

        # Update status bar message based on state
        self._update_status_for_state(new_state)

    def _update_status_for_state(self, state):
        """Update status bar message based on app state."""
        from ..controllers.app_state import AppState

        status_messages = {
            AppState.EMPTY: "No file loaded",
            AppState.DATA_LOADED: "Data loaded - ready for visualization",
            AppState.RENDERED: "Ready",
            AppState.BUSY: "Processing...",
        }

        message = status_messages.get(state, "")
        if message and self.status is not None:
            self.status.show_message(message, 3000)

    def _apply_styling(self):
        """Apply theme styling via ThemeManager."""
        try:
            # Load theme preference from config
            theme_name = self.config.get('ui.theme', 'dark')
            self.theme_manager.load_theme(theme_name)

            # Apply theme
            self.theme_manager.apply_theme(QApplication.instance())

            # Update theme menu check state
            if hasattr(self, 'light_theme_action') and hasattr(self, 'dark_theme_action'):
                self.light_theme_action.setChecked(theme_name == 'light')
                self.dark_theme_action.setChecked(theme_name == 'dark')

            # Connect theme change signal
            self.theme_manager.theme_changed.connect(self._on_theme_changed)

            logger.info(f"Applied theme: {theme_name}")
        except Exception as e:
            logger.error(f"Error applying theme: {e}", exc_info=True)
            # Fallback to basic styling
            self.setStyleSheet("")

    def set_theme(self, theme_name: str):
        """
        Set application theme.
        
        Args:
            theme_name: Theme name ('light' or 'dark')
        """
        try:
            self.theme_manager.load_theme(theme_name)
            self.theme_manager.apply_theme(QApplication.instance())

            # Save preference
            self.config.set('ui.theme', theme_name)
            self.config.save_config()

            # Update menu check state
            if hasattr(self, 'light_theme_action') and hasattr(self, 'dark_theme_action'):
                self.light_theme_action.setChecked(theme_name == 'light')
                self.dark_theme_action.setChecked(theme_name == 'dark')

            logger.info(f"Theme changed to: {theme_name}")
            self.status_bar.showMessage(f"Theme changed to {theme_name.capitalize()}", 2000)
        except Exception as e:
            logger.error(f"Error setting theme: {e}", exc_info=True)

    def _on_theme_changed(self, theme_name: str):
        """
        Handle theme change signal.

        Args:
            theme_name: New theme name
        """
        try:
            # Update legend widget colors if available
            if hasattr(self, 'viewer_widget') and self.viewer_widget:
                if hasattr(self.viewer_widget, 'renderer') and self.viewer_widget.renderer:
                    # Trigger legend refresh to use new theme colors
                    try:
                        self.viewer_widget.renderer._refresh_legend_from_active_layer()
                    except Exception:
                        pass

            # Update toolbar status strip
            if hasattr(self, 'toolbar_widget'):
                try:
                    camera_info = self.viewer_widget.renderer.get_camera_info() if self.viewer_widget and self.viewer_widget.renderer else None
                    camera_pos = camera_info.get('position', (0, 0, 0)) if camera_info else (0, 0, 0)
                    prop_name = self.current_property if hasattr(self, 'current_property') else ""
                    self.toolbar_widget.update_status_strip(
                        selection_count=0,
                        property_name=prop_name,
                        camera_pos=camera_pos,
                        theme_name=theme_name
                    )
                except Exception:
                    pass

            # Refresh all open windows and dialogs that have refresh_theme method
            self._refresh_all_themed_widgets()

            logger.debug(f"Theme changed to: {theme_name}")
        except Exception as e:
            logger.warning(f"Error handling theme change: {e}", exc_info=True)

    def _refresh_all_themed_widgets(self):
        """Refresh all widgets that have a refresh_theme method."""
        try:
            from PyQt6.QtWidgets import QApplication

            # Refresh all top-level widgets (windows, dialogs)
            for widget in QApplication.topLevelWidgets():
                self._refresh_widget_theme(widget)

            # Also refresh dock widgets in main window
            for dock in self.findChildren(QDockWidget):
                if dock.widget():
                    self._refresh_widget_theme(dock.widget())

        except Exception as e:
            logger.debug(f"Error refreshing themed widgets: {e}")

    def _refresh_widget_theme(self, widget):
        """Recursively refresh theme for a widget and its children."""
        try:
            # If widget has refresh_theme method, call it
            if hasattr(widget, 'refresh_theme') and callable(widget.refresh_theme):
                try:
                    widget.refresh_theme()
                except Exception as e:
                    logger.debug(f"Error refreshing theme for {widget.__class__.__name__}: {e}")

            # Recursively check children
            for child in widget.findChildren(QWidget):
                if hasattr(child, 'refresh_theme') and callable(child.refresh_theme):
                    try:
                        child.refresh_theme()
                    except Exception as e:
                        logger.debug(f"Error refreshing theme for {child.__class__.__name__}: {e}")

        except Exception as e:
            logger.debug(f"Error in _refresh_widget_theme: {e}")

    # STEP 17: Toolbar action handlers
    def _handle_scene_action(self, action: str):
        """Handle scene action from toolbar."""
        try:
            if action == "reset":
                self.reset_camera()
            elif action == "fit":
                self.fit_to_view()
            elif action == "wireframe":
                # Toggle wireframe mode
                if self.viewer_widget and self.viewer_widget.renderer:
                    # Implementation depends on renderer API
                    logger.debug("Wireframe toggle requested")
            elif action == "shading":
                # Toggle shading mode
                if self.viewer_widget and self.viewer_widget.renderer:
                    logger.debug("Shading toggle requested")
        except Exception as e:
            logger.error(f"Error handling scene action {action}: {e}", exc_info=True)

    def _handle_view_action(self, action: str):
        """Handle view action from toolbar."""
        try:
            if action == "block_data":
                self.open_data_viewer_window()
            elif action == "drillhole_data":
                self.open_drillhole_data_viewer_window()
            elif action == "statistics":
                self.open_statistics_window()
        except Exception as e:
            logger.error(f"Error handling view action {action}: {e}", exc_info=True)

    def _handle_panel_action(self, action: str):
        """Handle panel action from toolbar."""
        try:
            if action == "axes_panel":
                # Open axes/scale bar panel
                self._toggle_axes_scalebar_window(True)
        except Exception as e:
            logger.error(f"Error handling panel action {action}: {e}", exc_info=True)

    # STEP 17: Workspace layout management
    # REFACTORED: Workspace layout logic moved to ui/layout/workspace.py
    def load_workspace_layout(self, layout_name: str):
        """Load a predefined workspace layout.
        
        Args:
            layout_name: Layout name ("geology", "resource", "planning", "analytics")
        """
        from .layout.workspace import load_workspace_layout as _load_workspace_layout
        _load_workspace_layout(self, layout_name)

    def reset_workspace_layout(self):
        """Reset workspace to default layout."""
        from .layout.workspace import reset_workspace_layout as _reset_workspace_layout
        _reset_workspace_layout(self)

    def save_workspace_layout(self):
        """Save current workspace layout to file."""
        from .layout.workspace import save_workspace_layout as _save_workspace_layout
        _save_workspace_layout(self)

    def load_workspace_layout_file(self):
        """Load workspace layout from file."""
        from .layout.workspace import load_workspace_layout_file as _load_workspace_layout_file
        _load_workspace_layout_file(self)

    def _restore_state(self):
        """Restore window state and panel positions from config."""
        try:
            # Restore window geometry if saved
            settings = QSettings("GeoX", "MainWindow")
            geometry = settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)

            # Restore dock states
            state = settings.value("windowState")
            if state:
                self.restoreState(state)

            # Ensure drillhole control dock is visible after state restoration
            if self.drillhole_control_dock:
                self.drillhole_control_dock.setVisible(True)
                self.drillhole_control_dock.raise_()
            
            # Ensure geological explorer dock is also visible and tabified with drillhole dock
            if hasattr(self, 'geological_explorer_dock') and self.geological_explorer_dock:
                self.geological_explorer_dock.setVisible(True)
                # Ensure proper tabification at startup
                self.tabifyDockWidget(self.drillhole_control_dock, self.geological_explorer_dock)
                # Keep drillhole explorer as the default raised tab
                self.drillhole_control_dock.raise_()

            # Display settings are now managed by PropertyPanel
            # No separate display_settings_panel to restore

            logger.info("Restored window state")
        except Exception as e:
            logger.warning(f"Could not restore window state: {e}")

    def _save_state(self):
        """Save window state and panel positions to config."""
        try:
            settings = QSettings("GeoX", "MainWindow")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState())

            # Save display settings to config
            # self.config.save_config()

            logger.info("Saved window state")
        except Exception as e:
            logger.warning(f"Could not save window state: {e}")

    def _save_bookmark(self, bookmark_num: int):
        """
        Save current camera position as a view bookmark.

        Args:
            bookmark_num: Bookmark slot number (1-9)
        """
        if self.bookmarks is not None:
            self.bookmarks.save_bookmark(bookmark_num)

    def _load_bookmark(self, bookmark_num: int):
        """
        Load and restore a saved view bookmark.

        Args:
            bookmark_num: Bookmark slot number (1-9)
        """
        if self.bookmarks is not None:
            self.bookmarks.load_bookmark(bookmark_num)

    def _load_bookmarks(self):
        """Load saved view bookmarks from persistent storage."""
        if self.bookmarks is not None:
            self.bookmarks.load_from_settings()

    def _persist_bookmarks(self):
        """Save view bookmarks to persistent storage."""
        if self.bookmarks is not None:
            self.bookmarks.persist_bookmarks()

    def _save_layout(self):
        """Save window geometry and dock widget layout (NEW - Architecture Refactor)."""
        try:
            # Save to QSettings for persistence
            settings = QSettings("GeoX", "Layout")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("state", self.saveState())

            # Save all dialog window geometries
            self._save_dialog_geometries()

            logger.info("Saved window layout (geometry and dock state)")
        except Exception as e:
            logger.warning(f"Failed to save window layout: {e}")

    # =========================================================================
    # DIALOG LIFECYCLE METHODS (Delegated to DialogManager)
    # =========================================================================
    # These methods delegate to self.dialogs (DialogManager).
    # They are kept for backward compatibility with existing code.
    # New code should use self.dialogs directly.
    # =========================================================================

    def _save_dialog_geometries(self) -> None:
        """Save geometry for all dialog windows. Delegates to DialogManager."""
        if self.dialogs is not None:
            self.dialogs.save_all_geometries()

    def _restore_dialog_geometries(self) -> None:
        """Restore geometry for dialog windows when they're opened."""
        # Geometry is restored automatically by DialogManager.setup_persistence()
        logger.debug("Dialog geometry restoration ready (will restore when dialogs open)")

    def _is_dialog_valid(self, dialog) -> bool:
        """Check if a dialog widget is still valid. Delegates to DialogManager."""
        if self.dialogs is not None:
            return self.dialogs.is_valid(dialog)
        # Fallback
        if dialog is None:
            return False
        try:
            _ = dialog.isVisible()
            _ = dialog.windowTitle()
            return True
        except (RuntimeError, AttributeError):
            return False

    def _show_or_create_dialog(self, dialog_attr_name: str, create_callback):
        """Show existing dialog or create new one. Delegates to DialogManager."""
        if self.dialogs is not None:
            return self.dialogs.show_or_create(
                dialog_attr_name,
                create_callback,
                attr_holder=self,
                attr_name=dialog_attr_name
            )
        # Fallback for edge cases
        return create_callback()

    def _setup_dialog_persistence(self, dialog, dialog_name: str, panel_name: str = None) -> None:
        """Setup persistence for a dialog. Delegates to DialogManager."""
        if self.dialogs is not None:
            self.dialogs.setup_persistence(dialog, dialog_name, panel_name)
            # Also track in legacy list for backward compatibility
            if dialog not in self._open_panels:
                self._open_panels.append(dialog)
        else:
            # Fallback: minimal setup
            if dialog not in self._open_panels:
                self._open_panels.append(dialog)

    # ============================================================================
    # FILE OPERATIONS
    # ============================================================================

    def open_file(self):
        """Open a block model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Block Model File",
            "",
            "All Supported Files (*.csv *.txt *.vtk *.vtu *.obj *.gltf);;CSV Files (*.csv);;Text Files (*.txt);;VTK Files (*.vtk *.vtu);;3D Models (*.obj *.gltf)"
        )

        if not file_path:
            return

        self.load_file(Path(file_path))

    def load_file(self, file_path: Path):
        """Load a file in background thread.
        
        SECURITY: Validates file path and size before loading.
        """
        from ..utils.security import (
            FileSizeExceededError,
            SecurityError,
            validate_file_path,
            validate_file_size,
        )

        # SECURITY: Validate path first
        try:
            validated_path = validate_file_path(file_path, must_exist=True)
        except SecurityError as e:
            QMessageBox.critical(
                self,
                "Security Error",
                f"Cannot load file: {e}\n\nPlease select a valid file."
            )
            return
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "File Not Found",
                f"File not found: {file_path}"
            )
            return

        # SECURITY: Check file size with proper limits
        try:
            file_size = validate_file_size(validated_path, file_type='csv')
            file_size_mb = file_size / (1024 * 1024)
        except FileSizeExceededError as e:
            QMessageBox.critical(
                self,
                "File Too Large",
                f"{e}\n\nPlease use a smaller file or contact support."
            )
            return
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Cannot read file: {e}"
            )
            return

        # Warn for large files (but allow if under security limit)
        if file_size_mb > 50:
            reply = QMessageBox.question(
                self,
                "Large File Warning",
                f"This file is {file_size_mb:.1f} MB. Loading may take time. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Use validated path
        file_path = validated_path

        # Add to recent files
        self._add_recent_file(file_path)

        # Use controller task system for file loading
        if not self.controller:
            QMessageBox.critical(self, "Error", "Controller not available for file loading")
            return

        params = {
            "file_path": file_path
        }

        def on_load_complete(result: Dict[str, Any]):
            """
            Handle file load completion.
            
            This callback is already called on the main thread via Qt signals,
            so we can update UI directly without QTimer.singleShot.
            """
            logger.info(f"File load complete callback called. Result keys: {list(result.keys()) if result else 'None'}")

            if result is None or result.get("error"):
                error_msg = result.get("error", "Unknown error") if result else "No result"
                logger.error(f"File load error: {error_msg}")
                self.on_load_error(error_msg)
                return

            block_model = result.get("block_model")
            if block_model is not None:
                block_count = getattr(block_model, "block_count", None)
                if block_count is None:
                    try:
                        block_count = len(block_model)
                    except Exception:
                        block_count = 0
                logger.info(f"Block model loaded: {block_count} blocks")
                # Store file path before calling on_file_loaded
                result_file_path = result.get("file_path")
                if result_file_path:
                    self.current_file_path = Path(result_file_path)
                try:
                    self.on_file_loaded(block_model)
                except Exception as e:
                    # Avoid potential recursion in logging by not using exc_info=True
                    logger.error(f"Error in on_file_loaded: {type(e).__name__}: {str(e)}")
                    self.on_load_error(f"Failed to process loaded file: {e}")
            else:
                logger.error("No block model in result")
                self.on_load_error("No block model in result")

        self.controller.run_task('load_file', params, callback=on_load_complete)
        logger.info(f"Started loading file via task system: {file_path}")

    def on_file_loaded(self, block_model: BlockModel):
        """Handle successful file load."""
        import time
        handler_start = time.time()
        try:
            # Register block model with coordinate manager
            coord_start = time.time()
            # Get file path from current_file_path (set in on_load_complete)
            file_name = self.current_file_path.name if self.current_file_path else "Unknown"
            dataset_name = f"Block Model: {file_name}"
            dataset_info = self.coordinate_manager.register_dataset(
                dataset_name,
                block_model,
                'block_model'
            )
            coord_time = time.time() - coord_start
            logger.info(f"PERF: Coordinate registration took {coord_time:.3f}s")

            # Check if alignment is needed
            if not dataset_info.is_aligned:
                # Replace Unicode warning icons with ASCII for Windows console compatibility
                logger.warning("Block model requires coordinate alignment!")

                # Ask user if they want to auto-align
                reply = QMessageBox.question(
                    self,
                    "Coordinate Alignment Required",
                    f"The block model appears to be in a different coordinate system than existing data.\n\n"
                    f"Block model center: ({dataset_info.bounds.center[0]:,.2f}, {dataset_info.bounds.center[1]:,.2f}, {dataset_info.bounds.center[2]:,.2f})\n"
                    f"Required offset: ({dataset_info.offset[0]:,.2f}, {dataset_info.offset[1]:,.2f}, {dataset_info.offset[2]:,.2f}) meters\n\n"
                    f"Would you like to automatically align this block model with existing drillhole data?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )

                if reply == QMessageBox.StandardButton.Yes:
                    # Apply offset to align with reference frame
                    align_start = time.time()
                    self.coordinate_manager.apply_offset_to_block_model(block_model, dataset_info.offset)
                    dataset_info.is_aligned = True
                    align_time = time.time() - align_start
                    logger.info(f"PERF: Coordinate alignment took {align_time:.3f}s")
                    logger.info("Block model aligned to reference frame")

                    # Show alignment summary
                    summary = self.coordinate_manager.get_alignment_summary()
                    logger.info(summary)

            self.current_model = block_model
            # current_file_path already set in on_load_complete callback

            # Register block model with DataRegistry - this will emit blockModelLoaded signal
            # The signal handler (_on_block_model_loaded_from_registry) will update viewer and panels
            # This establishes DataRegistry as the single source of truth for updates
            try:
                # Use injected registry via controller (dependency injection)
                registry = self.controller.registry if self.controller else None
                if registry is not None:
                    registry.register_block_model(
                        block_model,
                        source_panel="MainWindow",
                        metadata={"source_path": str(self.current_file_path) if self.current_file_path else "Unknown"}
                    )
                    # Viewer and panel updates will be handled by DataRegistry.blockModelLoaded signal
                    # See _on_block_model_loaded_from_registry() handler
            except Exception as exc:
                logger.debug(f"Failed to register block model in DataRegistry: {exc}", exc_info=True)
                # Fallback: if registry registration fails, update viewer directly
                if self.viewer_widget:
                    self.viewer_widget.refresh_scene(block_model)
                if self.property_panel:
                    self.property_panel.set_block_model(block_model)


            # Update data analysis panel if already open

            handler_time = time.time() - handler_start
            logger.info(f"PERF: on_file_loaded total handler time: {handler_time:.3f}s")

            # Pass plotter reference for swath 3D linking
            if self.viewer_widget and hasattr(self.viewer_widget, 'plotter'):
                # Get grid and dataframe from renderer
                grid = None
                df = None
                if hasattr(self.viewer_widget, 'renderer') and self.viewer_widget.renderer:
                    if hasattr(self.viewer_widget.renderer, 'block_meshes'):
                        grid = self.viewer_widget.renderer.block_meshes.get('unstructured_grid')

                df = None
                if hasattr(block_model, 'to_dataframe'):
                    try:
                        df = block_model.to_dataframe()
                    except Exception:
                        pass

        except Exception as e:
            # Avoid potential recursion in logging by not using exc_info=True
            logger.error(f"Error in on_file_loaded: {type(e).__name__}: {str(e)}")
            QMessageBox.critical(self, "File Load Error", f"Error processing loaded file:\n{e}")

    def _on_block_model_loaded_from_registry(self, block_model: BlockModel):
        """
        Handle block model loaded signal from DataRegistry.
        
        This is the single source of truth for viewer and panel updates after
        a block model is registered with DataRegistry. This prevents race conditions
        and double-rendering issues.
        """
        import time
        handler_start = time.time()

        try:
            # Update viewer widget - single source of truth for rendering
            viewer_start = time.time()
            if self.viewer_widget:
                self.viewer_widget.refresh_scene(block_model)
            viewer_time = time.time() - viewer_start
            logger.info(f"PERF: ViewerWidget refresh_scene took {viewer_time:.3f}s")

            # CRITICAL: Force app state update after block model load
            # This ensures UI controls are enabled even if callback chain was broken
            try:
                if self.controller:
                    self.controller._update_state_from_scene()
            except Exception:
                pass

            # Update property panel
            panel_start = time.time()
            if self.property_panel:
                self.property_panel.set_block_model(block_model)
            panel_time = time.time() - panel_start
            logger.info(f"PERF: Property panel update took {panel_time:.3f}s")

            # Update data analysis panel if already open

            # Refresh Interactive Slicer panel if open
            self._refresh_slicer_panel()

            handler_time = time.time() - handler_start
            logger.info(f"PERF: _on_block_model_loaded_from_registry total handler time: {handler_time:.3f}s")

        except Exception as exc:
            # Avoid potential recursion in logging by not using exc_info=True
            logger.error(f"Error in _on_block_model_loaded_from_registry: {type(exc).__name__}: {str(exc)}")

            logger.info(f"Successfully loaded file: {self.current_file_path}")

            # Persist 'last_file' immediately for session restore resilience
            try:
                settings = QSettings("GeoX", "Session")
                if self.current_file_path:
                    settings.setValue("last_file", str(self.current_file_path))
            except Exception:
                pass

            # Apply any pending session state (camera, legend, filters, measurements)
            try:
                if getattr(self, '_pending_session_state', None) and self.viewer_widget and self.viewer_widget.renderer:
                    self.viewer_widget.renderer.apply_session_state(self._pending_session_state)
                    self._pending_session_state = None
            except Exception as e:
                logger.warning(f"Failed to apply pending session state: {e}")

            # Restore drillhole data if it was saved in project
            try:
                if getattr(self, '_pending_drillhole_state', None):
                    self._restore_drillhole_data(self._pending_drillhole_state)
                    self._pending_drillhole_state = None
            except Exception as e:
                logger.warning(f"Failed to restore drillhole data: {e}")

            # Note: Registry models restoration is handled elsewhere to avoid infinite loops
            # The _restore_registry_models call has been moved out of this signal handler

            # Enable actions that require a loaded block model
            try:
                if hasattr(self, 'view_data_action') and self.view_data_action:
                    self.view_data_action.setEnabled(True)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error loading model in viewer: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to load model: {e}"
            )

    def on_load_error(self, error_message: str):
        """Handle file load error."""
        QMessageBox.critical(
            self, "Load Error", f"Failed to load file:\n\n{error_message}"
        )
        logger.error(f"File load error: {error_message}")

    def _add_recent_file(self, file_path: Path):
        """Add a file to the recent files list."""
        try:
            file_str = str(file_path.absolute())
            recent_files = self.config.config.get('ui', {}).get('recent_files', [])

            # Remove if already in list
            if file_str in recent_files:
                recent_files.remove(file_str)

            # Add to front
            recent_files.insert(0, file_str)

            # Limit to max recent files
            max_recent = self.config.config.get('ui', {}).get('max_recent_files', 10)
            recent_files = recent_files[:max_recent]

            # Update config
            if 'ui' not in self.config.config:
                self.config.config['ui'] = {}
            self.config.config['ui']['recent_files'] = recent_files
            self.config.save_config()

            # Update menu
            self._update_recent_files_menu()

        except Exception as e:
            logger.warning(f"Failed to add recent file: {e}")

    def _update_recent_files_menu(self):
        """Update the recent files menu."""
        try:
            self.recent_files_menu.clear()
            recent_files = self.config.config.get('ui', {}).get('recent_files', [])

            if not recent_files:
                no_files_action = QAction("(No recent files)", self)
                no_files_action.setEnabled(False)
                self.recent_files_menu.addAction(no_files_action)
                return

            for i, file_path_str in enumerate(recent_files):
                file_path = Path(file_path_str)
                # Show number and filename
                action_text = f"{i+1}. {file_path.name}"
                action = QAction(action_text, self)
                action.setStatusTip(file_path_str)
                action.setToolTip(file_path_str)
                # Add keyboard shortcut for first 9 files
                if i < 9:
                    action.setShortcut(QKeySequence(f"Ctrl+{i+1}"))
                action.triggered.connect(lambda checked, fp=file_path: self._open_recent_file(fp))
                self.recent_files_menu.addAction(action)

            # Add separator and clear action
            self.recent_files_menu.addSeparator()
            clear_action = QAction("Clear Recent Files", self)
            clear_action.triggered.connect(self._clear_recent_files)
            self.recent_files_menu.addAction(clear_action)

        except Exception as e:
            logger.warning(f"Failed to update recent files menu: {e}")

    def _open_recent_file(self, file_path: Path):
        """Open a file from the recent files list."""
        if not file_path.exists():
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The file no longer exists:\n\n{file_path}"
            )
            # Remove from recent files
            try:
                recent_files = self.config.config.get('ui', {}).get('recent_files', [])
                file_str = str(file_path.absolute())
                if file_str in recent_files:
                    recent_files.remove(file_str)
                    self.config.config['ui']['recent_files'] = recent_files
                    self.config.save_config()
                    self._update_recent_files_menu()
            except Exception:
                pass
            return

        self.load_file(file_path)

    def _clear_recent_files(self):
        """Clear the recent files list."""
        reply = QMessageBox.question(
            self,
            "Clear Recent Files",
            "Are you sure you want to clear the recent files list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if 'ui' not in self.config.config:
                self.config.config['ui'] = {}
            self.config.config['ui']['recent_files'] = []
            self.config.save_config()
            self._update_recent_files_menu()
            self.status_bar.showMessage("Recent files cleared", 2000)

    def export_screenshot(self, filename: str = ""):
        """Export current view as screenshot."""
        if not filename:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Screenshot",
                "screenshot.png",
                "PNG Files (*.png);;JPEG Files (*.jpg)"
            )

        if filename and self.viewer_widget:
            self.viewer_widget.export_screenshot(filename)
            self.status_bar.showMessage(f"Screenshot saved: {filename}", 3000)
            logger.info(f"Screenshot saved: {filename}")

    def open_layout_composer(self):
        """Open the Layout Composer window for creating print layouts."""
        from .layout.layout_window import LayoutComposerWindow

        if not hasattr(self, '_layout_composer') or self._layout_composer is None:
            self._layout_composer = LayoutComposerWindow(main_window=self, parent=self)

        self._layout_composer.show()
        self._layout_composer.raise_()
        self._layout_composer.activateWindow()
        logger.info("Opened Layout Composer")

    def quick_layout_export(self, export_type: str = "pdf"):
        """
        Quick export with a standard layout template.

        Args:
            export_type: Export format - 'pdf', 'png', or 'png_hd' (600 DPI)
        """
        from pathlib import Path
        from ..layout.layout_document import LayoutDocument, ViewportItem, LegendItem, ScaleBarItem, TextItem
        from ..layout.layout_export import export_pdf, export_png

        # Determine DPI and format
        if export_type == "png_hd":
            dpi = 600
            ext = ".png"
            format_type = "png"
        elif export_type == "png":
            dpi = 300
            ext = ".png"
            format_type = "png"
        else:
            dpi = 300
            ext = ".pdf"
            format_type = "pdf"

        # Get file path
        default_name = f"GeoX_Export_{dpi}dpi{ext}"
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            f"Quick Export to {format_type.upper()}",
            str(Path.home() / default_name),
            f"{format_type.upper()} Files (*{ext})"
        )
        if not filepath:
            return

        filepath = Path(filepath)

        # Create quick layout document
        doc = LayoutDocument(name="Quick Export")
        doc.page.size = doc.page.size  # Use default A4 landscape

        # Add viewport
        viewport = ViewportItem(
            name="Main View",
            x_mm=10, y_mm=25,
            width_mm=200, height_mm=150,
        )
        # Capture camera state if viewer available
        if self.viewer_widget and hasattr(self.viewer_widget, 'renderer'):
            renderer = self.viewer_widget.renderer
            if hasattr(renderer, 'get_camera_info'):
                viewport.camera_state = renderer.get_camera_info()
            if hasattr(renderer, 'legend_manager') and hasattr(renderer.legend_manager, 'get_state'):
                viewport.legend_state = renderer.legend_manager.get_state()
                logger.info(f"[QUICK EXPORT] Captured legend state: {viewport.legend_state}")
        doc.add_item(viewport)

        # Add legend
        legend = LegendItem(
            name="Legend",
            x_mm=220, y_mm=25,
            width_mm=60, height_mm=100,
            legend_state=viewport.legend_state,
        )
        doc.add_item(legend)

        # Add scale bar
        scale_bar = ScaleBarItem(
            name="Scale Bar",
            x_mm=10, y_mm=185,
            width_mm=60, height_mm=12,
        )
        doc.add_item(scale_bar)

        # Add title
        title = TextItem(
            name="Title",
            text="GeoX Export",
            x_mm=10, y_mm=5,
            width_mm=277, height_mm=15,
            font_size=16,
            font_bold=True,
            alignment="center",
        )
        doc.add_item(title)

        # Prepare metadata
        import getpass
        from datetime import datetime
        metadata_values = {
            "project_name": "GeoX Project",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "author": getpass.getuser(),
            "software_version": "GeoX",
            "export_dpi": str(dpi),
        }

        try:
            if format_type == "pdf":
                export_pdf(doc, filepath, dpi, self.viewer_widget, metadata_values)
            else:
                export_png(doc, filepath, dpi, self.viewer_widget, metadata_values)

            self.status_bar.showMessage(f"Exported to {filepath}", 5000)
            logger.info(f"Quick layout exported to {filepath}")

            QMessageBox.information(
                self,
                "Export Complete",
                f"Layout exported to:\n{filepath}\n\nAudit record saved alongside."
            )

        except Exception as e:
            logger.error(f"Quick export failed: {e}")
            QMessageBox.warning(self, "Export Failed", f"Failed to export: {e}")

    def _has_valid_block_model(self):
        """Check if current_model exists and is not empty, or if there's simulated data in renderer layers."""
        import pandas as pd

        # First check current_model
        if self.current_model is not None:
            if isinstance(self.current_model, pd.DataFrame):
                if not self.current_model.empty:
                    return True
            else:
                return True

        # Check for block model data in renderer layers (SGSIM, kriging, classification, etc.)
        if self._has_block_model_in_layers():
            return True

        # Check registry for classified block model or block model
        try:
            registry = self.controller.registry if self.controller else None
            if registry:
                # Check for classified block model (from resource classification)
                classified = registry.get_classified_block_model(copy_data=False)
                if classified is not None:
                    if isinstance(classified, pd.DataFrame):
                        if not classified.empty:
                            return True
                    else:
                        return True

                # Check for regular block model in registry
                block_model = registry.get_block_model(copy_data=False)
                if block_model is not None:
                    if isinstance(block_model, pd.DataFrame):
                        if not block_model.empty:
                            return True
                    else:
                        return True
        except Exception as e:
            logger.debug(f"Error checking registry for block model: {e}")

        return False

    def _has_block_model_in_layers(self):
        """Check if there's block model data in renderer active layers."""
        try:
            if not (hasattr(self, 'viewer_widget') and self.viewer_widget and
                    hasattr(self.viewer_widget, 'renderer')):
                return False

            renderer = self.viewer_widget.renderer

            # Check active_layers for block model type data
            if hasattr(renderer, 'active_layers') and renderer.active_layers:
                for layer_name, layer_info in renderer.active_layers.items():
                    layer_type = layer_info.get('type', '')
                    # Block model types from all sources:
                    # - blocks/volume: loaded block models
                    # - sgsim/simulation: simulation results
                    # - kriging: estimation results
                    # - classification: resource classification
                    # - resource: resource reporting
                    if layer_type in ('blocks', 'volume', 'sgsim', 'kriging', 'simulation',
                                      'classification', 'resource', 'estimate'):
                        if layer_info.get('data') is not None:
                            return True
                    # Also check by layer name patterns for all block model sources
                    layer_lower = layer_name.lower()
                    block_model_patterns = [
                        'sgsim', 'kriging', 'simulation', 'block',
                        'classification', 'resource', 'estimate',
                        'measured', 'indicated', 'inferred',  # Resource categories
                        'ordinary', 'simple', 'universal', 'indicator',  # Kriging types
                        'cosgsim', 'sis', 'turning', 'dbs', 'mps', 'grf'  # Simulation types
                    ]
                    if any(pattern in layer_lower for pattern in block_model_patterns):
                        if layer_info.get('data') is not None:
                            return True

            # Check block_meshes for unstructured grid
            if hasattr(renderer, 'block_meshes') and renderer.block_meshes:
                if 'unstructured_grid' in renderer.block_meshes:
                    return True

        except Exception as e:
            logger.debug(f"Error checking block model in layers: {e}")

        return False

    def _get_block_model_from_layers(self):
        """Get block model DataFrame from renderer layers or registry if available."""
        import pandas as pd

        # First try renderer layers
        try:
            if (hasattr(self, 'viewer_widget') and self.viewer_widget and
                    hasattr(self.viewer_widget, 'renderer')):

                renderer = self.viewer_widget.renderer

                # Check active_layers for block model data from any source
                if hasattr(renderer, 'active_layers') and renderer.active_layers:
                    # Priority order for layer selection (most common first)
                    priority_patterns = [
                        # Simulation results
                        'sgsim: mean', 'sgsim: fe_sgsim_mean', 'sgsim:',
                        'cosgsim:', 'sis:', 'turning:', 'dbs:', 'mps:', 'grf:',
                        # Estimation results
                        'kriging', 'ordinary:', 'simple:', 'universal:', 'indicator:',
                        # Classification results
                        'classification', 'measured', 'indicated', 'inferred',
                        # Resource reporting
                        'resource', 'reserve',
                        # General block model
                        'block'
                    ]

                    for pattern in priority_patterns:
                        for layer_name, layer_info in renderer.active_layers.items():
                            if pattern in layer_name.lower():
                                grid_data = layer_info.get('data')
                                if grid_data is not None:
                                    # Try to convert to DataFrame
                                    df = self._grid_to_dataframe(grid_data, layer_name)
                                    if df is not None:
                                        logger.info(f"Got block model from layer: {layer_name}")
                                        return df

                    # Fallback: try any non-drillhole layer with block model type
                    for layer_name, layer_info in renderer.active_layers.items():
                        if 'drillhole' not in layer_name.lower():
                            layer_type = layer_info.get('type', '')
                            if layer_type in ('blocks', 'volume', 'sgsim', 'kriging', 'simulation',
                                              'classification', 'resource', 'estimate'):
                                grid_data = layer_info.get('data')
                                if grid_data is not None:
                                    df = self._grid_to_dataframe(grid_data, layer_name)
                                    if df is not None:
                                        logger.info(f"Got block model from layer (fallback): {layer_name}")
                                        return df

        except Exception as e:
            logger.debug(f"Error getting block model from layers: {e}")

        # Try registry as final fallback
        try:
            registry = self.controller.registry if self.controller else None
            if registry:
                # Try classified block model first (from resource classification)
                classified = registry.get_classified_block_model(copy_data=True)
                if classified is not None:
                    if isinstance(classified, pd.DataFrame) and not classified.empty:
                        logger.info("Got block model from registry (classified)")
                        return classified
                    elif hasattr(classified, 'to_dataframe'):
                        df = classified.to_dataframe()
                        if df is not None and not df.empty:
                            logger.info("Got block model from registry (classified, converted)")
                            return df

                # Try regular block model
                block_model = registry.get_block_model(copy_data=True)
                if block_model is not None:
                    if isinstance(block_model, pd.DataFrame) and not block_model.empty:
                        logger.info("Got block model from registry")
                        return block_model
                    elif hasattr(block_model, 'to_dataframe'):
                        df = block_model.to_dataframe()
                        if df is not None and not df.empty:
                            logger.info("Got block model from registry (converted)")
                            return df
        except Exception as e:
            logger.debug(f"Error getting block model from registry: {e}")

        return None

    def _grid_to_dataframe(self, grid_data, layer_name):
        """Convert PyVista grid to DataFrame."""
        try:
            import numpy as np
            import pandas as pd

            # Handle dict with 'mesh' key
            if isinstance(grid_data, dict) and 'mesh' in grid_data:
                grid_data = grid_data['mesh']

            # Try to extract cell centers and cell_data
            if hasattr(grid_data, 'cell_centers') and hasattr(grid_data, 'cell_data'):
                centers = grid_data.cell_centers()
                df_data = {
                    'X': centers.points[:, 0],
                    'Y': centers.points[:, 1],
                    'Z': centers.points[:, 2]
                }

                # Add all cell_data properties
                for prop_name in grid_data.cell_data.keys():
                    try:
                        prop_data = grid_data.cell_data[prop_name]
                        if np.issubdtype(prop_data.dtype, np.number):
                            df_data[prop_name] = prop_data
                    except Exception:
                        pass

                if df_data:
                    return pd.DataFrame(df_data)

        except Exception as e:
            logger.debug(f"Error converting grid to DataFrame: {e}")

        return None

    def export_filtered_data(self):
        """Open comprehensive data export dialog."""
        try:
            # Open the data export dialog
            dialog = DataExportDialog(self.registry, self)
            dialog.exec()
        except Exception as e:
            logger.error(f"Failed to open export dialog: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to open export dialog:\n{str(e)}"
            )

    def export_model(self):
        """Export 3D model to file."""
        if not self._has_valid_block_model():
            QMessageBox.warning(self, "No Model", "Load a model first.")
            return

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export 3D Model",
            "block_model.stl",
            "STL Files (*.stl);;OBJ Files (*.obj);;VTK Files (*.vtk)"
        )

        if filename and self.viewer_widget:
            try:
                self.viewer_widget.export_mesh_to_file(filename)
                self.status_bar.showMessage(f"Model exported: {filename}", 3000)
                logger.info(f"Exported 3D model: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export model:\n{e}")
                logger.error(f"Export error: {e}")

    def clear_scene(self):
        """Clear the scene and remove all models (block model and drillholes)."""
        # Check if there's anything to clear
        has_block_model = self._has_valid_block_model()
        has_drillhole_data = (hasattr(self.viewer_widget.renderer, 'drillhole_data') and
                             self.viewer_widget.renderer.drillhole_data is not None)

        if not has_block_model and not has_drillhole_data:
            QMessageBox.information(self, "No Model", "Scene is already empty.")
            return

        # Confirm action
        model_types = []
        if has_block_model:
            model_types.append("block model")
        if has_drillhole_data:
            model_types.append("drillholes")

        model_text = " and ".join(model_types)

        reply = QMessageBox.question(
            self,
            "Clear Scene",
            f"Are you sure you want to remove the {model_text} and clear the scene?\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            # Clear viewer
            if self.viewer_widget:
                self.viewer_widget.clear_scene()

            # Clear renderer drillhole data
            if hasattr(self.viewer_widget.renderer, 'drillhole_data'):
                self.viewer_widget.renderer.drillhole_data = None

            # Clear panels
            if self.property_panel:
                self.property_panel.clear()


            if self.data_viewer_panel and self.data_viewer_panel.isVisible():
                self.data_viewer_panel.clear()

            if self.scene_inspector_panel:
                self.scene_inspector_panel.clear_camera_info()

            # Clear current model
            self.current_model = None
            self.current_file_path = None

            # Disable actions that require a loaded block model
            try:
                if hasattr(self, 'view_data_action') and self.view_data_action:
                    self.view_data_action.setEnabled(False)
            except Exception:
                pass

            # Update status
            self.status_bar.showMessage("Scene cleared", 3000)
            self.setWindowTitle("GeoX")

            logger.info("Scene cleared successfully (block model and drillholes)")

        except Exception as e:
            QMessageBox.critical(self, "Clear Error", f"Failed to clear scene:\n{e}")
            logger.error(f"Clear scene error: {e}", exc_info=True)

    def on_property_changed(self, property_name: str):
        """Handle property change for both block models and drillholes."""
        # Check which layer is active in property panel
        active_layer = None
        if self.property_panel and hasattr(self.property_panel, 'active_layer_combo'):
            active_layer = self.property_panel.active_layer_combo.currentText()

        # Update block model if it's the active layer or if no active layer is set
        if self.current_model and (active_layer is None or "block" in active_layer.lower() or active_layer == "No layers active"):
            # Block model property coloring
            self.viewer_widget.set_property_coloring(property_name)

        # Update drillholes if drillholes layer is active or exists
        if "drillholes" in self.viewer_widget.renderer.active_layers and (active_layer is None or "drillhole" in active_layer.lower()):
            # Drillhole property coloring - check if drillholes layer is active
            layer_name = "drillholes"
            colormap = getattr(self.viewer_widget, 'current_colormap', 'viridis')
            if self.property_panel and hasattr(self.property_panel, 'colormap_combo'):
                colormap = self.property_panel.colormap_combo.currentText() or colormap

            color_mode = "discrete"
            if self.property_panel and hasattr(self.property_panel, 'color_mode_combo'):
                mode_text = self.property_panel.color_mode_combo.currentText()
                color_mode = "Lithology" if mode_text.lower() == "discrete" else "Assay"

            # Update the renderer's layer property
            self.viewer_widget.renderer.update_layer_property(
                layer_name, property_name, colormap, color_mode.lower()
            )

            # ✅ SYNC: Update Drillhole Control Panel to match Property Panel
            self._sync_drillhole_control_from_property_panel(property_name, color_mode)

            logger.info(f"Updated drillhole color to property '{property_name}'")

        try:
            if self.viewer_widget and self.viewer_widget.renderer:
                self.viewer_widget.renderer._refresh_legend_from_active_layer()
        except Exception:
            pass

    def _on_legend_colormap_changed(self, colormap: str):
        """
        Handle colormap change from legend widget.
        Updates the property panel's colormap dropdown and drillholes to stay synchronized.
        
        Args:
            colormap: New colormap name
        """
        try:
            if not colormap:
                return

            # Update property panel colormap dropdown
            if self.property_panel and hasattr(self.property_panel, 'set_colormap_from_external'):
                self.property_panel.set_colormap_from_external(colormap)
                logger.info(f"Updated property panel colormap dropdown to '{colormap}' from legend")

            # Check which layer is active
            active_layer = None
            if self.property_panel and hasattr(self.property_panel, 'active_layer_combo'):
                active_layer = self.property_panel.active_layer_combo.currentText()

            # Update drillholes if they are the active layer OR if legend is showing drillholes
            if self.viewer_widget and self.viewer_widget.renderer:
                # Check if drillholes layer exists and is active
                is_drillhole_active = (
                    "drillholes" in self.viewer_widget.renderer.active_layers and
                    (active_layer is None or "drillhole" in active_layer.lower() or active_layer == "drillholes")
                )

                if is_drillhole_active:
                    # Get current property and color mode from property panel
                    property_name = "Lithology"
                    color_mode = "discrete"
                    if self.property_panel:
                        if hasattr(self.property_panel, 'property_combo') and self.property_panel.property_combo:
                            prop_text = self.property_panel.property_combo.currentText()
                            if prop_text and prop_text != "No properties available":
                                property_name = prop_text
                        if hasattr(self.property_panel, 'color_mode_combo') and self.property_panel.color_mode_combo:
                            mode_text = self.property_panel.color_mode_combo.currentText()
                            if mode_text:
                                color_mode = mode_text.lower()

                    # Get custom colors if in discrete mode
                    custom_colors = None
                    if color_mode == "discrete" and self.property_panel:
                        if hasattr(self.property_panel, '_custom_discrete_colors'):
                            custom_colors = self.property_panel._custom_discrete_colors.get((active_layer or "drillholes", property_name))

                    # Update drillhole colors using update_layer_property for consistency
                    self.viewer_widget.renderer.update_layer_property(
                        "drillholes",
                        property_name,
                        colormap,
                        color_mode,
                        custom_colors=custom_colors
                    )
                    logger.info(f"Updated drillhole colors from legend colormap change: '{colormap}' for property '{property_name}'")
        except Exception as e:
            logger.error(f"Error updating from legend colormap change: {e}", exc_info=True)

    def on_colormap_changed(self, colormap: str):
        """Handle colormap change for both block models and drillholes."""
        # Check which layer is active in property panel
        active_layer = None
        if self.property_panel and hasattr(self.property_panel, 'active_layer_combo'):
            active_layer = self.property_panel.active_layer_combo.currentText()

        # Update block model if it's the active layer or if no active layer is set
        if self.current_model and (active_layer is None or "block" in active_layer.lower() or active_layer == "No layers active"):
            # Block model colormap
            self.viewer_widget.set_colormap(colormap)

        # Update drillholes if drillholes layer is active or exists
        if "drillholes" in self.viewer_widget.renderer.active_layers and (active_layer is None or "drillhole" in active_layer.lower()):
            # Drillhole colormap - use new layer-based approach
            layer_data = self.viewer_widget.renderer.active_layers["drillholes"].get("data", {})

            # Get current property and color mode from property panel
            property_name = "Lithology"
            color_mode = "discrete"
            if self.property_panel:
                if hasattr(self.property_panel, 'property_combo') and self.property_panel.property_combo:
                    property_name = self.property_panel.property_combo.currentText() or "Lithology"
                if hasattr(self.property_panel, 'color_mode_combo') and self.property_panel.color_mode_combo:
                    color_mode = self.property_panel.color_mode_combo.currentText().lower() or "discrete"

            # Get custom colors if in discrete mode
            custom_colors = None
            if color_mode == "discrete" and self.property_panel:
                layer = self.property_panel.active_layer_combo.currentText()
                if layer and hasattr(self.property_panel, '_custom_discrete_colors'):
                    custom_colors = self.property_panel._custom_discrete_colors.get((layer, property_name))

            # Update drillhole colors (will update legend too)
            self.viewer_widget.renderer.update_layer_property(
                "drillholes", property_name, colormap, color_mode, custom_colors=custom_colors
            )

            logger.info(f"Updated drillhole colormap to '{colormap}' for property '{property_name}'")

        try:
            if self.viewer_widget and self.viewer_widget.renderer:
                self.viewer_widget.renderer._refresh_legend_from_active_layer()
        except Exception:
            pass

    # ============================================================================
    # VIEW OPERATIONS
    # ============================================================================

    def reset_camera(self):
        """Reset camera to default view."""
        if self.viewer_widget:
            self.viewer_widget.reset_camera()
            self.status_bar.showMessage("View reset", 2000)

    def _get_block_model_layer_name(self) -> Optional[str]:
        """
        Find the current block model layer name in the renderer.

        Returns layer name like "Block Model: production_2024" or "Block Model",
        or None if no block model layer exists.
        """
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return None

        active_layers = self.viewer_widget.renderer.active_layers
        if not active_layers:
            return None

        # Find any layer that starts with "Block Model"
        for layer_name in active_layers.keys():
            if layer_name.startswith("Block Model"):
                logger.debug(f"Found block model layer: '{layer_name}'")
                return layer_name

        logger.debug("No block model layer found in active_layers")
        return None

    def fit_to_view(self):
        """Fit model to viewport."""
        if self.viewer_widget:
            self.viewer_widget.fit_to_view()
            self.status_bar.showMessage("Fitted to view", 2000)

    def set_view_preset(self, preset: str):
        """Set view to preset."""
        if self.viewer_widget:
            self.viewer_widget.set_view_preset(preset)
            self.status_bar.showMessage(f"View: {preset}", 2000)

    def toggle_projection(self, checked: bool):
        """Toggle projection mode."""
        if self.viewer_widget:
            self.viewer_widget.toggle_orthographic_projection(checked)
            mode = "Orthographic" if checked else "Perspective"
            if self.status is not None:
                self.status.set_camera_mode(mode)
            logger.info(f"Projection: {mode}")

    def on_projection_toggled(self, enabled: bool):
        """Handle projection toggle from scene inspector."""
        self.projection_action.setChecked(enabled)
        if self.viewer_widget:
            self.viewer_widget.toggle_orthographic_projection(enabled)


    def toggle_axes(self, checked: bool):
        """Toggle axes visibility."""
        if self.viewer_widget:
            self.viewer_widget.toggle_axes(checked)
            self.axes_action.setChecked(checked)

    def toggle_bounds(self, checked: bool):
        """Toggle bounding grid visibility."""
        if self.viewer_widget:
            self.viewer_widget.toggle_bounds(checked)
            self.grid_action.setChecked(checked)

    def toggle_legend(self, checked: bool):
        """Toggle legend visibility via LegendManager."""
        if self.controller and self.controller.legend_manager:
            self.controller.legend_manager.set_visibility(checked)
        elif self.viewer_widget:
            self.viewer_widget.toggle_scalar_bar_visibility(checked)

    def on_scalar_bar_toggled(self, visible: bool):
        """Handle legend toggle from scene inspector via LegendManager."""
        if self.controller and self.controller.legend_manager:
            self.controller.legend_manager.set_visibility(visible)
        elif self.viewer_widget:
            self.viewer_widget.toggle_scalar_bar_visibility(visible)

    def _toggle_multi_legend(self, checked: bool):
        """Toggle multi-legend panel visibility."""
        if self.viewer_widget:
            visible = self.viewer_widget.toggle_multi_legend(checked)
            # Update menu action state
            if hasattr(self, 'multi_legend_action'):
                self.multi_legend_action.setChecked(visible)
            # Keep classic legend action in sync (mutually exclusive)
            if visible and hasattr(self, 'classic_legend_action'):
                self.classic_legend_action.setChecked(False)

    def _toggle_classic_legend(self, checked: bool):
        """Toggle classic colorbar legend (switches out of multi-legend mode)."""
        if self.viewer_widget:
            if checked:
                # Switch to classic mode: disable multi-mode
                self.viewer_widget.toggle_multi_legend(False)
                if hasattr(self, 'multi_legend_action'):
                    self.multi_legend_action.setChecked(False)
                # Ensure the legend widget is visible in classic mode
                if (self.viewer_widget._legend_manager is not None
                        and self.viewer_widget._legend_manager.widget is not None):
                    self.viewer_widget._legend_manager.widget.show()
                    self.viewer_widget._legend_manager.widget.raise_()
            else:
                # Hide the classic legend entirely
                if (self.viewer_widget._legend_manager is not None
                        and self.viewer_widget._legend_manager.widget is not None):
                    self.viewer_widget._legend_manager.widget.hide()

    def _on_legend_mode_changed(self, is_multi: bool):
        """Handle legend mode changes from the LegendWidget context menu."""
        if hasattr(self, 'multi_legend_action'):
            self.multi_legend_action.setChecked(is_multi)
        if hasattr(self, 'classic_legend_action'):
            self.classic_legend_action.setChecked(not is_multi)

    def on_ground_grid_spacing_reset(self):
        """Reset ground grid spacing to a nice value and update the UI control."""
        try:
            if not self.viewer_widget or not self.viewer_widget.renderer:
                return
            rnd = self.viewer_widget.renderer
            # Trigger reset on renderer
            if hasattr(rnd, 'reset_ground_grid_spacing'):
                rnd.reset_ground_grid_spacing()
            else:
                # Fallback: compute directly from bounds
                b = getattr(rnd, '_get_scene_bounds', lambda: None)()
                if b is not None:
                    span = max((b[1]-b[0]), (b[3]-b[2]))
                    if hasattr(rnd, '_nice_number'):
                        rnd._ground_plane_spacing = float(rnd._nice_number(max(1e-6, span/10.0)))
            # Refresh grid if visible
            if getattr(rnd, 'show_grid', False):
                rnd._update_axes_bounds_for_scene()
                if getattr(rnd, 'plotter', None) is not None:
                    rnd.plotter.render()
            # Update the spin box to reflect current spacing
            try:
                spacing = float(getattr(rnd, '_ground_plane_spacing', 100.0) or 100.0)
                if spacing > 0 and self.scene_inspector_panel:
                    self.scene_inspector_panel.ground_grid_spacing_spin.setValue(spacing)
            except Exception:
                pass
        except Exception:
            pass

    def apply_lighting_preset(self, preset: str):
        """Apply lighting preset via renderer."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                # Delegate to renderer
                if hasattr(self.viewer_widget.renderer, 'apply_lighting_preset'):
                    self.viewer_widget.renderer.apply_lighting_preset(preset)
                    self.status_bar.showMessage(f"Applied lighting preset: {preset.capitalize()}", 2000)
                    logger.info(f"Applied lighting preset: {preset}")
                else:
                    logger.warning("Renderer does not support lighting presets")
            except Exception as e:
                logger.error(f"Error applying lighting preset: {e}", exc_info=True)
                self.status_bar.showMessage(f"Error applying lighting: {str(e)}", 3000)
        else:
            self.status_bar.showMessage("Viewer not ready", 2000)

    # ============================================================================
    # TOOLS
    # ============================================================================

    def open_slice_tool(self):
        """Open slice tool (focus property panel)."""
        if self.property_dock:
            self.property_dock.show()
            self.property_dock.raise_()
        self.status_bar.showMessage("Use Property Panel for slicing", 2000)

    def open_filter_tool(self):
        """Open filter tool (focus property panel)."""
        if self.property_dock:
            self.property_dock.show()
            self.property_dock.raise_()
        self.status_bar.showMessage("Use Property Panel for filtering", 2000)

    def toggle_pick_mode(self, checked: bool):
        """Toggle block selection mode."""
        mode = "enabled" if checked else "disabled"
        try:
            if self.viewer_widget and self.viewer_widget.renderer:
                # Record a friendly flag on the renderer for other code paths
                try:
                    self.viewer_widget.renderer.picking_enabled = bool(checked)
                except Exception:
                    pass

                if checked:
                    # Enable global picking via the renderer (safe: will no-op if plotter missing)
                    try:
                        self.viewer_widget.renderer.set_pick_callback(self.viewer_widget._on_global_pick if hasattr(self.viewer_widget, '_on_global_pick') else None)
                    except Exception:
                        # Fallback to the viewer forwarding handler
                        try:
                            self.viewer_widget.renderer.set_pick_callback(self.viewer_widget._on_global_pick)
                        except Exception:
                            pass
                    try:
                        self.viewer_widget.renderer.setup_global_picking()
                    except Exception:
                        pass
                    self.status_bar.showMessage("Selection mode enabled", 2000)
                else:
                    # Attempt to disable picking by clearing callbacks and overlays
                    try:
                        self.viewer_widget.renderer.pick_callback = None
                    except Exception:
                        pass
                    try:
                        self.viewer_widget.renderer.clear_pick_overlays(render=True)
                    except Exception:
                        pass
                    self.status_bar.showMessage("Selection mode disabled", 2000)

        except Exception:
            logger.exception("Failed to toggle pick mode")
        logger.info(f"Pick mode: {mode}")

    def show_statistics(self):
        """Show property statistics."""
        if not self._has_valid_block_model():
            QMessageBox.information(self, "No Model", "Load a model first to view statistics.")
            return

        try:
            # Build statistics text
            stats_text = "<h3>Model Statistics</h3>"
            stats_text += f"<p><b>Total Blocks:</b> {len(self.current_model.positions):,}</p>"

            # Get bounds
            try:
                bounds = self.current_model.get_bounds()
            except (AttributeError, ValueError) as e:
                # Calculate bounds manually if method doesn't exist
                logger.debug(f"Bounds method unavailable: {e}, calculating manually")
                positions = self.current_model.positions
                bounds = [
                    positions[:, 0].min(), positions[:, 0].max(),
                    positions[:, 1].min(), positions[:, 1].max(),
                    positions[:, 2].min(), positions[:, 2].max()
                ]

            if bounds:
                stats_text += "<h4>Model Extent:</h4>"
                stats_text += f"<p>X: {bounds[0]:.2f} to {bounds[1]:.2f}<br>"
                stats_text += f"Y: {bounds[2]:.2f} to {bounds[3]:.2f}<br>"
                stats_text += f"Z: {bounds[4]:.2f} to {bounds[5]:.2f}</p>"

            # Get property statistics
            stats_text += "<h4>Properties:</h4>"
            stats_text += f"<p><b>Count:</b> {len(self.current_model.get_property_names())}</p>"
            stats_text += "<ul>"
            for prop in self.current_model.get_property_names()[:10]:  # Show first 10
                stats_text += f"<li>{prop}</li>"
            stats_text += "</ul>"

            if len(self.current_model.get_property_names()) > 10:
                stats_text += f"<p><i>... and {len(self.current_model.get_property_names()) - 10} more</i></p>"

            QMessageBox.information(self, "Model Statistics", stats_text)

        except Exception as e:
            logger.error(f"Error showing statistics: {e}")
            QMessageBox.warning(
                self,
                "Statistics Error",
                f"Could not generate statistics:\n{str(e)}"
            )

    def open_selection_manager(self):
        """Open the Block Selection Manager dialog."""
        # Check for block model - either current_model or in layers
        if not self._has_valid_block_model():
            QMessageBox.information(self, "No Model", "Load a block model first to use selection tools.")
            return

        try:
            # Create dialog if it doesn't exist
            if self.selection_dialog is None:
                # Create as independent window (no parent) to prevent disappearing when minimized
                self.selection_dialog = QDialog()
                self.selection_dialog.setWindowTitle("Block Selection Manager")
                self.selection_dialog.resize(450, 800)

                # Set window flags to enable minimize button and keep window persistent
                self.selection_dialog.setWindowFlags(
                    Qt.WindowType.Window |
                    Qt.WindowType.WindowMinimizeButtonHint |
                    Qt.WindowType.WindowMaximizeButtonHint |
                    Qt.WindowType.WindowCloseButtonHint
                )

                # Set modeless to allow interaction with both windows
                self.selection_dialog.setModal(False)

                layout = QVBoxLayout(self.selection_dialog)
                self.selection_panel = SelectionPanel(self.selection_dialog)
                layout.addWidget(self.selection_panel)

                # Set main window reference for interactive features
                self.selection_panel.set_main_window(self)

                # Connect signals
                self.selection_panel.selection_changed.connect(self._on_selection_changed)

                logger.info("Created Block Selection Manager dialog")

            # Always update plotter and model data when opening
            if hasattr(self, 'viewer_widget') and hasattr(self.viewer_widget, 'renderer'):
                self.selection_panel.set_plotter(self.viewer_widget.renderer.plotter)
                logger.info("Set plotter for selection panel")
            else:
                logger.warning("viewer_widget or renderer not available for selection panel")

            # Set block model data - handle both BlockModel, DataFrame, and layers
            import pandas as pd
            block_df = None
            grid_spec = None

            if self.current_model is not None:
                if isinstance(self.current_model, pd.DataFrame):
                    block_df = self.current_model
                elif hasattr(self.current_model, 'to_dataframe'):
                    block_df = self.current_model.to_dataframe()
                    grid_spec = getattr(self.current_model, 'grid_spec', None)

            # If no current_model, try to get from layers
            if block_df is None:
                block_df = self._get_block_model_from_layers()
                if block_df is not None:
                    logger.info("Selection Manager: Loaded block model from renderer layers")

            if block_df is not None:
                self.selection_panel.set_block_model(block_df, grid_spec)

            self.selection_dialog.show()
            self.selection_dialog.raise_()
            self.selection_dialog.activateWindow()

        except Exception as e:
            logger.error(f"Error opening selection manager: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open Selection Manager:\n{str(e)}"
            )

    def open_cross_section_manager(self):
        """Open the Cross-Section Manager dialog."""
        if not self._has_valid_block_model():
            QMessageBox.information(self, "No Model", "Load a block model first to create cross-sections.")
            return

        try:
            # Create dialog if it doesn't exist
            if self.cross_section_manager_dialog is None:
                # Create as independent window (no parent) to prevent disappearing when minimized
                self.cross_section_manager_dialog = QDialog()
                self.cross_section_manager_dialog.setWindowTitle("Cross-Section Manager")
                self.cross_section_manager_dialog.resize(500, 750)

                # Set window flags to enable minimize button and keep window persistent
                self.cross_section_manager_dialog.setWindowFlags(
                    Qt.WindowType.Window |
                    Qt.WindowType.WindowMinimizeButtonHint |
                    Qt.WindowType.WindowMaximizeButtonHint |
                    Qt.WindowType.WindowCloseButtonHint
                )

                # Set modeless to allow interaction with both windows
                self.cross_section_manager_dialog.setModal(False)

                layout = QVBoxLayout(self.cross_section_manager_dialog)
                self.cross_section_manager_panel = CrossSectionManagerPanel(self.cross_section_manager_dialog, signals=self.signals)
                layout.addWidget(self.cross_section_manager_panel)

                # Set main window reference for interactive features
                self.cross_section_manager_panel.set_main_window(self)

                # Set block model data - handle both BlockModel and DataFrame
                import pandas as pd
                if isinstance(self.current_model, pd.DataFrame):
                    block_df = self.current_model
                elif hasattr(self.current_model, 'to_dataframe'):
                    block_df = self.current_model.to_dataframe()
                else:
                    block_df = None

                grid_spec = getattr(self.current_model, 'grid_spec', None) if not isinstance(self.current_model, pd.DataFrame) else None
                if block_df is not None:
                    self.cross_section_manager_panel.set_block_model(block_df, grid_spec)

                # Connect signals
                self.cross_section_manager_panel.section_render_requested.connect(self._on_section_render_requested)

                logger.info("Created Cross-Section Manager dialog")
            else:
                # Update with current model (same safe pattern as first-open)
                import pandas as pd
                if isinstance(self.current_model, pd.DataFrame):
                    block_df = self.current_model
                elif self.current_model is not None and hasattr(self.current_model, 'to_dataframe'):
                    block_df = self.current_model.to_dataframe()
                else:
                    block_df = None

                grid_spec = getattr(self.current_model, 'grid_spec', None) if self.current_model is not None and not isinstance(self.current_model, pd.DataFrame) else None
                if block_df is not None:
                    self.cross_section_manager_panel.set_block_model(block_df, grid_spec)

            self.cross_section_manager_dialog.show()
            self.cross_section_manager_dialog.raise_()
            self.cross_section_manager_dialog.activateWindow()

        except Exception as e:
            logger.error(f"Error opening cross-section manager: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open Cross-Section Manager:\n{str(e)}"
            )

    def open_interactive_slicer(self):
        """Open the Interactive Slicer panel."""
        try:
            # Show the panel via PanelManager
            self.panel_manager.show_panel("InteractiveSlicerPanel")

            # Get panel instance and set renderer reference
            panel_instance = self.panel_manager.get_panel_instance("InteractiveSlicerPanel")
            if panel_instance and hasattr(self, 'viewer_widget') and hasattr(self.viewer_widget, 'renderer'):
                panel_instance.set_renderer(self.viewer_widget.renderer)
                panel_instance.refresh_layers()
                logger.info("Interactive Slicer panel opened and connected to renderer")
            else:
                logger.warning("Interactive Slicer panel opened but renderer connection may be incomplete")

        except Exception as e:
            logger.error(f"Error opening interactive slicer: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open Interactive Slicer:\n{str(e)}"
            )

    def _refresh_slicer_panel(self):
        """Refresh the Interactive Slicer panel's layer list after new block data is loaded."""
        try:
            panel_instance = self.panel_manager.get_panel_instance("InteractiveSlicerPanel")
            if panel_instance and hasattr(panel_instance, 'refresh_layers'):
                panel_instance.refresh_layers()
                logger.debug("Refreshed Interactive Slicer panel layer list")
        except Exception as e:
            # Silent fail - panel might not be open, which is fine
            logger.debug(f"Could not refresh slicer panel (may not be open): {e}")

    def export_advanced_screenshot(self):
        """Open the Advanced Screenshot Export dialog."""
        try:
            # Set plotter for screenshot manager
            if self.viewer_widget and self.viewer_widget.renderer:
                self.screenshot_manager.set_plotter(self.viewer_widget.renderer.plotter)

            # Open dialog
            dialog = ScreenshotExportDialog(self.screenshot_manager, self)
            dialog.exec()

        except Exception as e:
            logger.error(f"Error opening advanced screenshot dialog: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open Advanced Screenshot Export:\n{str(e)}"
            )

    def _on_selection_changed(self, selected_indices: set):
        """Handle selection changes from selection manager."""
        logger.info(f"Selection changed: {len(selected_indices)} blocks selected")

        # Update visual highlight in viewer widget
        if self.viewer_widget:
            try:
                # Set selected blocks in viewer widget
                self.viewer_widget._selected_blocks = selected_indices
                # Update visualization
                self.viewer_widget._update_selection_visualization()
                logger.debug(f"Updated visual highlight for {len(selected_indices)} selected blocks")
            except Exception as e:
                logger.debug(f"Failed to update visual highlight: {e}", exc_info=True)

    def _on_section_render_requested(self, section_name: str, property_name: str):
        """Handle cross-section render request."""
        logger.info(f"Main window received section render request: {section_name}, {property_name}")

        try:
            # Get section mesh from cross-section manager
            mesh = self.cross_section_manager_panel.get_section_mesh(section_name, property_name)

            logger.info(f"Got mesh from section manager: {mesh is not None}")

            if mesh is None:
                QMessageBox.warning(
                    self,
                    "No Data",
                    f"Cross-section '{section_name}' contains no blocks.\n"
                    "Try adjusting the position or thickness."
                )
                return

            logger.info(f"Mesh has {mesh.n_cells} cells")

            # Add to 3D viewer via plotter + layer registration
            renderer = self.viewer_widget.renderer if self.viewer_widget else None
            if renderer and renderer.plotter:
                layer_name = f"Section_{section_name}"

                # Check if property exists in mesh
                scalars_to_use = property_name if property_name in mesh.array_names else None

                logger.info(f"Adding mesh to renderer with scalars: {scalars_to_use}")

                # Build kwargs for plotter.add_mesh (PyVista call)
                plotter_kwargs = {
                    'scalars': scalars_to_use,
                    'name': layer_name,
                    'cmap': 'viridis',
                    'show_scalar_bar': False,
                }
                try:
                    if hasattr(mesh, 'n_cells') and getattr(mesh, 'n_cells', 0) == 0 and getattr(mesh, 'n_points', 0) > 0:
                        plotter_kwargs.update({
                            'point_size': 18,
                            'render_points_as_spheres': True,
                            'show_edges': False,
                        })
                    else:
                        plotter_kwargs.update({
                            'show_edges': True,
                        })
                except Exception:
                    plotter_kwargs.update({'point_size': 12, 'render_points_as_spheres': True})

                # Use plotter.add_mesh (PyVista) then register as layer
                actor = renderer.plotter.add_mesh(mesh, **plotter_kwargs)

                if actor is None:
                    logger.error("Section rendering failed: plotter.add_mesh returned None")
                    QMessageBox.warning(
                        self,
                        "Section Render Error",
                        "The section could not be added to the 3D view. Please check logs for details."
                    )
                    return

                # Register in the renderer layer system so it appears in scene controls
                renderer.add_layer(layer_name, actor, data=mesh, layer_type='section')
                renderer.plotter.render()

                logger.info(f"Rendered cross-section '{section_name}' with property '{property_name}'")
                self.statusBar().showMessage(f"Rendered section: {section_name} ({mesh.n_cells} blocks)", 5000)

                QMessageBox.information(
                    self,
                    "Section Rendered",
                    f"Cross-section '{section_name}' has been added to the 3D view.\n"
                    f"Blocks in section: {mesh.n_cells}\n"
                    f"Property: {property_name}"
                )
            else:
                logger.error("viewer_widget or renderer not available")
                QMessageBox.warning(self, "Error", "3D viewer not available")

        except Exception as e:
            logger.error(f"Error rendering cross-section: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Render Error",
                f"Failed to render cross-section:\n{str(e)}"
            )

    # ============================================================================
    # PANELS
    # ============================================================================

    def reset_layout(self):
        """Reset all panels to default positions."""
        # Reset left dock
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.left_dock)
        self.left_dock.show()

        # Reset right side docks - tabified (Drillhole Explorer + Geological Explorer as tabs)
        if hasattr(self, 'drillhole_control_dock') and self.drillhole_control_dock:
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.drillhole_control_dock)
            self.drillhole_control_dock.show()
            self.drillhole_control_dock.setFloating(False)

        if hasattr(self, 'geological_explorer_dock') and self.geological_explorer_dock:
            self.geological_explorer_dock.setFloating(False)
            self.tabifyDockWidget(self.drillhole_control_dock, self.geological_explorer_dock)
            self.geological_explorer_dock.show()
            # Raise drillhole as default tab
            self.drillhole_control_dock.raise_()

        # Reset tabs to first tab
        self.left_tab_widget.setCurrentIndex(0)  # Property Controls

        self.status_bar.showMessage("Layout reset to default", 2000)
        logger.info("Reset panel layout")

    # ============================================================================
    # DATA & ANALYSIS
    # ============================================================================


    def open_statistics_window(self):
        """Open Statistics window."""
        # Check if panel already exists and is visible
        if hasattr(self, 'statistics_dialog') and self._is_dialog_valid(self.statistics_dialog):
            try:
                if self.statistics_panel:
                    self.statistics_panel._refresh_available_data()
                    if self.current_model:
                        self.statistics_panel.set_block_model(self.current_model)
            except Exception as exc:
                logger.debug(f"Failed to refresh Statistics panel on reopen: {exc}", exc_info=True)

            if self.statistics_dialog.isVisible():
                self.statistics_dialog.raise_()
                self.statistics_dialog.activateWindow()
            else:
                # Window exists but is minimized - restore it
                self.statistics_dialog.show()
                self.statistics_dialog.raise_()
                self.statistics_dialog.activateWindow()
            return
        elif hasattr(self, 'statistics_dialog'):
            # Dialog was destroyed - clear reference
            self.statistics_dialog = None

        # Create new panel window
        from PyQt6.QtWidgets import QVBoxLayout

        self.statistics_dialog = QDialog(None)  # No parent - independent window
        self.statistics_dialog.setWindowTitle("Statistics")
        self.statistics_dialog.resize(600, 800)
        self.statistics_dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.statistics_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.statistics_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored)
        self._setup_dialog_persistence(self.statistics_dialog, 'statistics_dialog')

        layout = QVBoxLayout(self.statistics_dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create panel
        registry = self.controller.registry if self.controller else self._registry
        self.statistics_panel = StatisticsPanel(registry=registry)
        layout.addWidget(self.statistics_panel)

        # Connect signals
        if self.viewer_widget:
            self.statistics_panel.property_changed.connect(
                self.viewer_widget.set_property_coloring
            )
            self.statistics_panel.filter_applied.connect(
                self.on_analysis_filter_applied
            )
            self.statistics_panel.filter_cleared.connect(
                self.on_analysis_filter_cleared
            )
            self.statistics_panel.slice_applied.connect(
                self.on_analysis_slice_applied
            )
            self.statistics_panel.slice_cleared.connect(
                self.on_analysis_slice_cleared
            )

        # Set block model if available
        if self.current_model:
            self.statistics_panel.set_block_model(self.current_model)
        else:
            try:
                self.statistics_panel._refresh_available_data()
            except Exception as exc:
                logger.debug(f"Failed to refresh Statistics panel after open: {exc}", exc_info=True)

        logger.info("Opened Statistics panel in separate window")

        # Show as non-modal dialog
        self.statistics_dialog.show()

    def open_charts_window(self):
        """Open Charts & Visualization window."""
        # Check if panel already exists and is visible
        if hasattr(self, 'charts_dialog') and self._is_dialog_valid(self.charts_dialog):
            if self.charts_dialog.isVisible():
                self.charts_dialog.raise_()
                self.charts_dialog.activateWindow()
            else:
                # Window exists but is minimized - restore it
                self.charts_dialog.show()
                self.charts_dialog.raise_()
                self.charts_dialog.activateWindow()
            return
        elif hasattr(self, 'charts_dialog'):
            # Dialog was destroyed - clear reference
            self.charts_dialog = None

        # Create new panel window
        from PyQt6.QtWidgets import QVBoxLayout

        self.charts_dialog = QDialog(None)  # No parent - independent window
        self.charts_dialog.setWindowTitle("Charts & Visualization")
        self.charts_dialog.resize(500, 700)
        self.charts_dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.charts_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.charts_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored)
        self._setup_dialog_persistence(self.charts_dialog, 'charts_dialog')

        layout = QVBoxLayout(self.charts_dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create panel
        self.charts_panel = ChartsPanel()
        layout.addWidget(self.charts_panel)

        # Set block model if available - try current_model first, then layers
        block_model_set = False

        # First try current_model
        if self.current_model is not None:
            import pandas as pd
            is_valid = True
            if isinstance(self.current_model, pd.DataFrame):
                is_valid = not self.current_model.empty

            if is_valid:
                if hasattr(self.charts_panel, 'set_block_model'):
                    self.charts_panel.set_block_model(self.current_model)
                else:
                    self.charts_panel._update_block_model(self.current_model)
                block_model_set = True

        # If no current_model, try to load from renderer layers (SGSIM, kriging, etc.)
        if not block_model_set:
            if (hasattr(self, 'viewer_widget') and self.viewer_widget and
                hasattr(self.viewer_widget, 'renderer') and
                hasattr(self.viewer_widget.renderer, 'active_layers')):

                active_layers = self.viewer_widget.renderer.active_layers

                # Look for block model layers from all sources
                layer_priority = [
                    "SGSIM: FE_SGSIM_MEAN", "SGSIM: MEAN", "SGSIM: Mean",
                    "SGSIM: P50", "SGSIM: P10", "SGSIM: P90",
                    "CoSGSIM:", "SIS:", "Turning:", "DBS:", "MPS:", "GRF:",
                    "Kriging", "Ordinary:", "Simple:", "Universal:", "Indicator:",
                    "Classification", "Resource"
                ]
                selected_layer = None

                # Try to find a priority layer (case-insensitive)
                for layer_name in active_layers.keys():
                    layer_lower = layer_name.lower()
                    for priority in layer_priority:
                        if priority.lower() in layer_lower:
                            selected_layer = layer_name
                            break
                    if selected_layer:
                        break

                # If no priority layer found, use first available block model layer
                if selected_layer is None and active_layers:
                    for layer_name, layer_info in active_layers.items():
                        # Skip drillhole layers
                        if 'drillhole' not in layer_name.lower():
                            layer_type = layer_info.get('type', '')
                            if layer_type in ('blocks', 'volume', 'sgsim', 'kriging', 'simulation',
                                              'classification', 'resource', 'estimate') or \
                               any(p in layer_name.lower() for p in ['sgsim', 'kriging', 'block']):
                                selected_layer = layer_name
                                break

                # Load the selected layer
                if selected_layer:
                    try:
                        layer_info = active_layers[selected_layer]
                        if 'data' in layer_info and layer_info['data'] is not None:
                            grid_data = layer_info['data']
                            if isinstance(grid_data, dict) and 'mesh' in grid_data:
                                grid_data = grid_data['mesh']

                            self.charts_panel.set_grid_data(grid_data, selected_layer)
                            block_model_set = True
                            logger.info(f"Loaded layer '{selected_layer}' into Charts panel")
                            self.status_bar.showMessage(f"Charts loaded with layer: {selected_layer}", 3000)
                        else:
                            logger.warning(f"Selected layer '{selected_layer}' has no data")
                    except Exception as e:
                        logger.error(f"Error loading layer '{selected_layer}' into Charts panel: {e}", exc_info=True)
                        QMessageBox.warning(
                            self,
                            "Layer Loading Error",
                            f"Could not load layer '{selected_layer}' for analysis.\n\n"
                            f"Error: {str(e)}\n\n"
                            f"Please load a traditional block model (CSV) for analysis."
                        )
                else:
                    logger.warning("No data layers available for Charts panel")

        logger.info("Opened Charts & Visualization panel in separate window")

        # Show as non-modal dialog
        self.charts_dialog.show()

    def open_swath_window(self):
        """Open Swath Plot Analysis window."""
        # Check if panel already exists and is visible
        if hasattr(self, 'swath_dialog') and self._is_dialog_valid(self.swath_dialog):
            if self.swath_dialog.isVisible():
                self.swath_dialog.raise_()
                self.swath_dialog.activateWindow()
            else:
                # Window exists but is minimized - restore it
                self.swath_dialog.show()
                self.swath_dialog.raise_()
                self.swath_dialog.activateWindow()
            return
        elif hasattr(self, 'swath_dialog'):
            # Dialog was destroyed - clear reference
            self.swath_dialog = None

        # Create new panel window
        from PyQt6.QtWidgets import QVBoxLayout

        self.swath_dialog = QDialog(None)  # No parent - independent window
        self.swath_dialog.setWindowTitle("Swath Plot Analysis")
        self.swath_dialog.resize(550, 700)
        self.swath_dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.swath_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.swath_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored)
        self._setup_dialog_persistence(self.swath_dialog, 'swath_dialog')

        layout = QVBoxLayout(self.swath_dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create panel
        self.swath_panel = SwathPanel()
        layout.addWidget(self.swath_panel)

        # Bind controller for analysis tasks
        if self.controller:
            self.swath_panel.bind_controller(self.controller)

        # Connect signals
        if self.viewer_widget:
            self.swath_panel.swath_highlight_requested.connect(
                self.on_swath_highlight_requested
            )
            self.swath_panel.swath_highlight_cleared.connect(
                self.on_swath_highlight_cleared
            )

        # Set block model if available - check both current_model and layers
        block_model_set = False

        # First try current_model
        if self.current_model is not None:
            import pandas as pd
            is_valid = True
            if isinstance(self.current_model, pd.DataFrame):
                is_valid = not self.current_model.empty

            if is_valid:
                self.swath_panel.set_block_model(self.current_model)
                block_model_set = True

                # Pass PyVista references for 3D highlighting
                if self.viewer_widget and self.viewer_widget.plotter:
                    if hasattr(self.viewer_widget.renderer, 'block_meshes'):
                        if 'unstructured_grid' in self.viewer_widget.renderer.block_meshes:
                            # Get DataFrame - handle both BlockModel and DataFrame
                            if isinstance(self.current_model, pd.DataFrame):
                                block_df = self.current_model
                            elif hasattr(self.current_model, 'to_dataframe'):
                                block_df = self.current_model.to_dataframe()
                            else:
                                block_df = None

                            if block_df is not None:
                                self.swath_panel.set_plotter_reference(
                                    self.viewer_widget.plotter,
                                    self.viewer_widget.renderer.block_meshes['unstructured_grid'],
                                    block_df
                                )

        # If no current_model, try to load from renderer layers (SGSIM, kriging, etc.)
        if not block_model_set:
            if (hasattr(self, 'viewer_widget') and self.viewer_widget and
                hasattr(self.viewer_widget, 'renderer') and
                hasattr(self.viewer_widget.renderer, 'active_layers')):

                active_layers = self.viewer_widget.renderer.active_layers

                # Look for block model layers from all sources (simulation, estimation, classification, etc.)
                layer_priority = [
                    # Simulation results
                    "SGSIM: FE_SGSIM_MEAN", "SGSIM: MEAN", "SGSIM: Mean",
                    "SGSIM: P50", "SGSIM: P10", "SGSIM: P90",
                    "CoSGSIM:", "SIS:", "Turning:", "DBS:", "MPS:", "GRF:",
                    # Estimation results
                    "Kriging", "Ordinary:", "Simple:", "Universal:", "Indicator:",
                    # Classification/Resource results
                    "Classification", "Measured", "Indicated", "Inferred",
                    "Resource", "Reserve"
                ]
                selected_layer = None

                # Try to find a priority layer (case-insensitive)
                for layer_name in active_layers.keys():
                    layer_lower = layer_name.lower()
                    for priority in layer_priority:
                        if priority.lower() in layer_lower:
                            selected_layer = layer_name
                            break
                    if selected_layer:
                        break

                # If no priority layer found, use first available block model layer
                if selected_layer is None and active_layers:
                    for layer_name, layer_info in active_layers.items():
                        # Skip drillhole layers
                        if 'drillhole' not in layer_name.lower():
                            layer_type = layer_info.get('type', '')
                            # Check for all block model types
                            valid_types = ('blocks', 'volume', 'sgsim', 'kriging', 'simulation',
                                           'classification', 'resource', 'estimate')
                            valid_patterns = ['sgsim', 'kriging', 'block', 'classification',
                                              'resource', 'measured', 'indicated', 'inferred']
                            if layer_type in valid_types or \
                               any(p in layer_name.lower() for p in valid_patterns):
                                selected_layer = layer_name
                                break

                # Load the selected layer
                if selected_layer:
                    try:
                        layer_info = active_layers[selected_layer]
                        if 'data' in layer_info and layer_info['data'] is not None:
                            grid_data = layer_info['data']
                            if isinstance(grid_data, dict) and 'mesh' in grid_data:
                                grid_data = grid_data['mesh']

                            self.swath_panel.set_grid_data(grid_data, selected_layer)
                            block_model_set = True
                            logger.info(f"Loaded layer '{selected_layer}' into Swath panel")
                            self.status_bar.showMessage(f"Swath plots loaded with layer: {selected_layer}", 3000)
                        else:
                            logger.warning(f"Selected layer '{selected_layer}' has no data")
                    except Exception as e:
                        logger.error(f"Error loading layer '{selected_layer}' into Swath panel: {e}", exc_info=True)
                        QMessageBox.warning(
                            self,
                            "Layer Loading Error",
                            f"Could not load layer '{selected_layer}' for analysis.\n\n"
                            f"Error: {str(e)}\n\n"
                            f"Please load a traditional block model (CSV) for analysis."
                        )
                else:
                    logger.warning("No data layers available for Swath panel")

        logger.info("Opened Swath Plot Analysis panel in separate window")

        # Show as non-modal dialog
        self.swath_dialog.show()

    def populate_registry_block_models_menu(self):
        """Populate View -> Registry Block Models submenu from DataRegistry."""
        menu = getattr(self, "registry_block_models_menu", None)
        if menu is None:
            return

        menu.clear()
        entries = self._get_registry_block_models_for_menu()

        if not entries:
            empty_action = QAction("No block models in registry", self)
            empty_action.setEnabled(False)
            menu.addAction(empty_action)
            return

        for entry in entries:
            action = QAction(entry["title"], self)
            action.setStatusTip(entry["status_tip"])
            key = entry["key"]
            action.triggered.connect(
                lambda checked=False, k=key, t=entry["title"]: self.visualize_registry_block_model_by_key(k, t)
            )
            menu.addAction(action)

        menu.addSeparator()
        status_action = QAction("Open Data Registry Status...", self)
        status_action.triggered.connect(self.open_data_registry_status_panel)
        menu.addAction(status_action)

    def _get_registry_block_models_for_menu(self) -> List[Dict[str, Any]]:
        """Return block model entries available in DataRegistry for menu display."""
        entries: List[Dict[str, Any]] = []

        registry = self.controller.registry if self.controller else self._registry
        if registry is None:
            return entries

        key_map = [
            ("block_model", "Block Model"),
            ("classified_block_model", "Classified Block Model"),
        ]

        for key, display_name in key_map:
            try:
                model = registry.get_data(key, copy_data=False)
            except Exception:
                model = None

            if model is None:
                continue

            source_panel = "Unknown"
            timestamp_text = "N/A"
            try:
                metadata = registry.get_metadata(key)
                if metadata:
                    source_panel = getattr(metadata, "source_panel", None) or source_panel
                    timestamp = getattr(metadata, "timestamp", None)
                    if timestamp is not None:
                        timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

            block_count = None
            try:
                if isinstance(model, BlockModel):
                    block_count = model.block_count
                elif PANDAS_AVAILABLE and isinstance(model, pd.DataFrame):
                    block_count = len(model)
                elif hasattr(model, "block_count"):
                    block_count = int(getattr(model, "block_count"))
                elif hasattr(model, "to_dataframe"):
                    df = model.to_dataframe()
                    if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
                        block_count = len(df)
            except Exception:
                block_count = None

            count_text = f"{block_count:,} blocks" if isinstance(block_count, int) else "available"

            entries.append(
                {
                    "title": f"{display_name} ({count_text})",
                    "status_tip": (
                        f"Visualize {display_name} from DataRegistry "
                        f"(source: {source_panel}, updated: {timestamp_text})"
                    ),
                    "key": key,
                }
            )

        return entries

    def _get_or_build_registry_block_model(self, registry_key: str) -> Optional[BlockModel]:
        """Get a normalized BlockModel for a DataRegistry key with lightweight caching."""
        registry = self.controller.registry if self.controller else self._registry
        if registry is None:
            return None

        try:
            raw_model = registry.get_data(registry_key, copy_data=False)
        except Exception:
            raw_model = None
        if raw_model is None:
            return None

        cache = getattr(self, "_registry_visualization_cache", None)
        if cache is None:
            cache = {}
            self._registry_visualization_cache = cache

        source_id = id(raw_model)
        cached = cache.get(registry_key)
        if cached and cached.get("source_id") == source_id and cached.get("model") is not None:
            return cached["model"]

        normalized_model = self._normalize_block_model_for_viewer(raw_model)
        if normalized_model is None:
            return None

        cache[registry_key] = {"source_id": source_id, "model": normalized_model}
        return normalized_model

    def visualize_registry_block_model_by_key(self, registry_key: str, model_label: str):
        """Resolve a registry key and visualize it in the main 3D viewer."""
        normalized_model = self._get_or_build_registry_block_model(registry_key)
        if normalized_model is None:
            QMessageBox.warning(
                self,
                "Model Not Available",
                f"Could not load '{registry_key}' from DataRegistry."
            )
            return
        self.visualize_registry_block_model(
            normalized_model,
            model_label=model_label,
            registry_key=registry_key,
        )

    def _normalize_block_model_for_viewer(self, block_model: Any) -> Optional[BlockModel]:
        """Normalize potential registry data into a BlockModel for DataViewerPanel."""
        if block_model is None:
            return None

        if isinstance(block_model, BlockModel):
            if getattr(block_model, "block_count", 0) > 0:
                return block_model
            return None

        if PANDAS_AVAILABLE and isinstance(block_model, pd.DataFrame):
            if block_model.empty:
                return None
            try:
                converted = BlockModel()
                converted.update_from_dataframe(block_model)
                return converted if converted.block_count > 0 else None
            except Exception:
                return None

        if hasattr(block_model, "to_dataframe"):
            try:
                df = block_model.to_dataframe()
            except Exception:
                return None
            if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    converted = BlockModel()
                    converted.update_from_dataframe(df)
                    return converted if converted.block_count > 0 else None
                except Exception:
                    return None

        return None

    def visualize_registry_block_model(
        self,
        block_model: Any,
        model_label: str = "Registry Block Model",
        registry_key: Optional[str] = None,
    ):
        """Load a registry-provided block model into the main 3D viewer."""
        normalized_model = self._normalize_block_model_for_viewer(block_model)
        if normalized_model is None:
            QMessageBox.warning(
                self,
                "Unsupported Model",
                "Selected registry entry is not a valid block model."
            )
            return

        if not self.viewer_widget:
            QMessageBox.warning(
                self,
                "Viewer Not Available",
                "3D viewer is not available."
            )
            return

        try:
            # Avoid redundant heavy reload when the same registry entry is already active.
            if registry_key and getattr(self, "_active_registry_model_key", None) == registry_key:
                renderer_model = getattr(self.viewer_widget.renderer, "current_model", None) if self.viewer_widget else None
                if renderer_model is normalized_model:
                    if hasattr(self, "status_bar") and self.status_bar:
                        self.status_bar.showMessage(f"{model_label} already active", 2000)
                    return

            self.current_model = normalized_model
            self.viewer_widget.refresh_scene(normalized_model)
            if self.property_panel:
                self.property_panel.set_block_model(normalized_model)
            self._active_registry_model_key = registry_key

            try:
                if hasattr(self, "view_data_action") and self.view_data_action:
                    self.view_data_action.setEnabled(True)
            except Exception:
                pass

            if hasattr(self, "status_bar") and self.status_bar:
                self.status_bar.showMessage(f"Visualized {model_label}", 3000)
            logger.info(f"Visualized registry block model: {model_label}")
        except Exception as exc:
            logger.error(f"Failed to visualize registry block model: {exc}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Failed to visualize selected block model:\n{exc}"
            )

    def open_data_viewer_window(self, checked: bool = False, block_model_override=None):
        """Open Block Model Data Viewer window."""
        _ = checked  # QAction.triggered(bool) compatibility
        normalized_override = None
        if block_model_override is not None:
            normalized_override = self._normalize_block_model_for_viewer(block_model_override)
            if normalized_override is None:
                QMessageBox.warning(
                    self,
                    "Unsupported Model",
                    "Selected registry entry is not a valid block model."
                )
                return

        # Check if panel already exists and is visible
        if hasattr(self, 'data_viewer_dialog') and self._is_dialog_valid(self.data_viewer_dialog):
            if self.data_viewer_dialog.isVisible():
                self.data_viewer_dialog.raise_()
                self.data_viewer_dialog.activateWindow()
            else:
                # Window exists but is minimized - restore it
                self.data_viewer_dialog.show()
                self.data_viewer_dialog.raise_()
                self.data_viewer_dialog.activateWindow()

            if normalized_override is not None and getattr(self, 'data_viewer_panel', None):
                self.data_viewer_panel.set_block_model(normalized_override)
            return

        # Ensure we have a block model; if none, try to extract from active layers
        if normalized_override is None and not self._has_valid_block_model():
            try:
                extracted = self._extract_block_model_from_layers()
            except Exception as e:
                logger.warning(f"Failed to auto-extract block model for data viewer: {e}")
                extracted = None
            if extracted is not None:
                self.current_model = extracted
                try:
                    # Use injected registry via controller (dependency injection)
                    registry = self.controller.registry if self.controller else None
                    if registry is not None:
                        registry.register_block_model_generated(
                            extracted,
                            source_panel="DataViewer",
                            metadata={"source": "auto_extracted_from_layers"},
                        )
                except Exception:
                    pass
                try:
                    if hasattr(self, "view_data_action") and self.view_data_action:
                        self.view_data_action.setEnabled(True)
                except Exception:
                    pass
            else:
                QMessageBox.information(
                    self,
                    "No Model Loaded",
                    "Please load a block model first before viewing data.\n\n"
                    "If you have SGSIM or kriging grids visible, build a block model or\n"
                    "use Resource Classification to extract a block model from those layers.",
                )
                return

        # Create new panel window
        from PyQt6.QtWidgets import QVBoxLayout

        self.data_viewer_dialog = QDialog(None)  # No parent - independent window
        self.data_viewer_dialog.setWindowTitle("Block Model Data Viewer")
        self.data_viewer_dialog.resize(900, 700)
        self.data_viewer_dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.data_viewer_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.data_viewer_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored)
        self._setup_dialog_persistence(self.data_viewer_dialog, 'data_viewer_dialog')

        layout = QVBoxLayout(self.data_viewer_dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create panel
        self.data_viewer_panel = DataViewerPanel(host_window=self)
        layout.addWidget(self.data_viewer_panel)

        # Connect signals (optional: sync with 3D view)
        if self.viewer_widget:
            self.data_viewer_panel.block_selected.connect(
                lambda idx: logger.info(f"Block {idx} selected in data viewer")
            )

        # Set block model
        if normalized_override is not None:
            self.data_viewer_panel.set_block_model(normalized_override)
            try:
                if hasattr(self, "view_data_action") and self.view_data_action:
                    self.view_data_action.setEnabled(True)
            except Exception:
                pass
        elif self.current_model:
            self.data_viewer_panel.set_block_model(self.current_model)

        logger.info("Opened Block Model Data Viewer in separate window")

        # Show as non-modal dialog
        self.data_viewer_dialog.show()

    def open_table_viewer_window_from_df(self, df, title: str = "Table Viewer"):
        """Open a generic table viewer window from a pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            QMessageBox.warning(
                self,
                "Pandas Not Available",
                "Pandas is required for table viewing. Please install pandas."
            )
            return

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            QMessageBox.information(self, "No Data", "No table data available to display.")
            return

        from PyQt6.QtWidgets import QVBoxLayout

        dialog = QDialog(None)
        dialog.setWindowTitle(title)
        dialog.resize(900, 700)
        dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        key_slug = ''.join(c.lower() if c.isalnum() else '_' for c in title)[:60]
        self._setup_dialog_persistence(dialog, f'table_viewer_dialog_{key_slug}')

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        panel = TableViewerPanel()
        layout.addWidget(panel)

        try:
            panel.set_dataframe(df, title=title)
        except Exception as e:
            logger.error(f"Failed to set dataframe for table viewer: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load table into viewer:\n{e}")
            return

        dialog.show()
        logger.info(f"Opened Table Viewer window for: {title}")

    def open_drillhole_data_viewer_window(self):
        """Open a table viewer for drillhole data with all data types combined."""
        try:
            from PyQt6.QtWidgets import QMessageBox

            # Get drillhole data from DataRegistry
            registry = self.controller.registry if self.controller else None
            if not registry:
                QMessageBox.information(
                    self,
                    "No Data Registry",
                    "Data registry is not available.\n\n"
                    "Please ensure the application is properly initialized."
                )
                return

            drillhole_data = registry.get_drillhole_data()
            if not drillhole_data:
                QMessageBox.information(
                    self,
                    "No Drillhole Data",
                    "No drillhole data is currently available.\n\n"
                    "Please load drillhole data via:\n"
                    "• Drillholes → Drillhole Loading"
                )
                return

            # Merge all drillhole data types into one combined DataFrame
            combined_df = self._merge_all_drillhole_data(drillhole_data)

            if combined_df is None or combined_df.empty:
                QMessageBox.information(
                    self,
                    "No Data to Merge",
                    "Could not merge drillhole data.\n\n"
                    "Please ensure you have at least assays or composites loaded."
                )
                return

            # Show combined data in table viewer
            self.open_table_viewer_window_from_df(
                combined_df,
                title="Drillhole Data (Desurveyed - Assays/Composites with Survey, Lithology, Structure)"
            )

        except Exception as e:
            logger.error(f"Error opening drillhole data viewer: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to open drillhole data viewer:\n{e}")

    def _merge_all_drillhole_data(self, drillhole_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Merge all drillhole data types into a clean, simple combined DataFrame.
        
        This creates a table showing each assay/composite interval with:
        - HoleID, From, To, X, Y, Z (desurveyed coordinates)
        - All assay/composite columns (grades, length, etc.)
        - Interpolated survey data (azimuth, dip) at interval midpoint
        - Lithology code matched by depth interval overlap
        - Structure data if available
        - Only unique collar info (max_depth, etc.) - no duplicate X/Y/Z
        
        Args:
            drillhole_data: Dictionary containing drillhole data with keys:
                - 'assays': DataFrame with assay intervals (already desurveyed)
                - 'collars': DataFrame with collar information
                - 'surveys': DataFrame with survey information
                - 'lithology': DataFrame with lithology intervals
                - 'composites': DataFrame with composite intervals
                - 'structures': DataFrame with structural measurements (optional)
        
        Returns:
            Combined DataFrame with all drillhole data merged, or None if no data available
        """
        import numpy as np

        try:
            # Helper: Find column by common names (case-insensitive)
            def find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
                """Find column by common name candidates."""
                if df is None or df.empty:
                    return None
                for col in df.columns:
                    if col.upper() in [c.upper() for c in candidates]:
                        return col
                return None

            def find_hole_id_col(df: pd.DataFrame) -> Optional[str]:
                return find_col(df, ['HOLEID', 'HOLE_ID', 'holeid', 'hole_id', 'HOLE'])

            def find_depth_col(df: pd.DataFrame) -> Optional[str]:
                return find_col(df, ['DEPTH', 'MD', 'MEASURED_DEPTH', 'FROM'])

            def find_from_to_cols(df: pd.DataFrame) -> tuple:
                from_col = find_col(df, ['FROM', 'DEPTH_FROM', 'MFROM'])
                to_col = find_col(df, ['TO', 'DEPTH_TO', 'MTO'])
                return from_col, to_col

            def find_azimuth_col(df: pd.DataFrame) -> Optional[str]:
                return find_col(df, ['AZIMUTH', 'AZI', 'AZ', 'BEARING', 'AZIM'])

            def find_dip_col(df: pd.DataFrame) -> Optional[str]:
                return find_col(df, ['DIP', 'INCLINATION', 'INCL', 'INC'])

            def find_lith_code_col(df: pd.DataFrame) -> Optional[str]:
                return find_col(df, ['LITH_CODE', 'LITHCODE', 'LITHOLOGY', 'LITH', 'ROCK_TYPE', 'ROCKTYPE', 'CODE'])

            # ================== Step 1: Get base data (composites or assays) ==================
            base_df = None
            data_source = None

            # Prefer composites (already processed/aggregated)
            if drillhole_data.get('composites') is not None and not drillhole_data['composites'].empty:
                base_df = drillhole_data['composites'].copy()
                data_source = "composites"
                logger.info(f"Using composites as base: {len(base_df)} rows")
            elif drillhole_data.get('assays') is not None and not drillhole_data['assays'].empty:
                base_df = drillhole_data['assays'].copy()
                data_source = "assays"
                logger.info(f"Using assays as base: {len(base_df)} rows")

            if base_df is None or base_df.empty:
                logger.warning("No assays or composites available for merging")
                return None

            hole_id_col = find_hole_id_col(base_df)
            base_from_col, base_to_col = find_from_to_cols(base_df)

            if not hole_id_col:
                logger.warning("Could not find hole ID column in base data")
                return base_df

            if not base_from_col or not base_to_col:
                logger.warning("Could not find FROM/TO columns in base data")
                return base_df

            # Calculate midpoint depth for each interval (needed for survey interpolation)
            base_df['_MID_DEPTH'] = (base_df[base_from_col] + base_df[base_to_col]) / 2.0

            # ================== Step 2: Add collar info (only unique columns) ==================
            collars_df = drillhole_data.get('collars')
            if collars_df is not None and not collars_df.empty:
                collar_hole_id = find_hole_id_col(collars_df)
                if collar_hole_id:
                    # Columns already in base that we should NOT duplicate
                    exclude_cols = {hole_id_col.upper(), 'X', 'Y', 'Z', 'EASTING', 'NORTHING',
                                   'ELEVATION', 'RL', 'EAST', 'NORTH'}

                    # Find useful collar columns to add (like max_depth, project, etc.)
                    collar_cols_to_add = []
                    for col in collars_df.columns:
                        if col.upper() not in exclude_cols and col != collar_hole_id:
                            # Check if column already exists in base
                            if col not in base_df.columns and col.upper() not in [c.upper() for c in base_df.columns]:
                                collar_cols_to_add.append(col)

                    if collar_cols_to_add:
                        collar_merge_df = collars_df[[collar_hole_id] + collar_cols_to_add].copy()
                        if collar_hole_id != hole_id_col:
                            collar_merge_df = collar_merge_df.rename(columns={collar_hole_id: hole_id_col})

                        base_df = base_df.merge(collar_merge_df, on=hole_id_col, how='left')
                        logger.info(f"Added {len(collar_cols_to_add)} unique collar columns: {collar_cols_to_add}")

            # ================== Step 3: Add survey data (interpolated at midpoints) ==================
            surveys_df = drillhole_data.get('surveys')
            if surveys_df is not None and not surveys_df.empty:
                survey_hole_id = find_hole_id_col(surveys_df)
                survey_depth_col = find_depth_col(surveys_df)
                survey_azi_col = find_azimuth_col(surveys_df)
                survey_dip_col = find_dip_col(surveys_df)

                if survey_hole_id and survey_depth_col and (survey_azi_col or survey_dip_col):
                    # Initialize columns
                    if survey_azi_col:
                        base_df['azimuth'] = np.nan
                    if survey_dip_col:
                        base_df['dip'] = np.nan

                    # For each hole, interpolate survey values at interval midpoints
                    for hid in base_df[hole_id_col].unique():
                        hole_surveys = surveys_df[surveys_df[survey_hole_id] == hid].sort_values(survey_depth_col)
                        if hole_surveys.empty:
                            continue

                        hole_mask = base_df[hole_id_col] == hid
                        hole_intervals = base_df.loc[hole_mask]

                        if len(hole_surveys) == 1:
                            # Single survey point - use constant values
                            if survey_azi_col:
                                base_df.loc[hole_mask, 'azimuth'] = hole_surveys[survey_azi_col].iloc[0]
                            if survey_dip_col:
                                base_df.loc[hole_mask, 'dip'] = hole_surveys[survey_dip_col].iloc[0]
                        else:
                            # Multiple survey points - interpolate at midpoints
                            survey_depths = hole_surveys[survey_depth_col].values
                            mid_depths = hole_intervals['_MID_DEPTH'].values

                            if survey_azi_col:
                                survey_azis = hole_surveys[survey_azi_col].values
                                interp_azis = np.interp(mid_depths, survey_depths, survey_azis)
                                base_df.loc[hole_mask, 'azimuth'] = interp_azis

                            if survey_dip_col:
                                survey_dips = hole_surveys[survey_dip_col].values
                                interp_dips = np.interp(mid_depths, survey_depths, survey_dips)
                                base_df.loc[hole_mask, 'dip'] = interp_dips

                    logger.info("Added interpolated survey data (azimuth, dip) at interval midpoints")

            # ================== Step 4: Add lithology by depth interval overlap ==================
            lithology_df = drillhole_data.get('lithology')
            if lithology_df is not None and not lithology_df.empty:
                lith_hole_id = find_hole_id_col(lithology_df)
                lith_from_col, lith_to_col = find_from_to_cols(lithology_df)
                lith_code_col = find_lith_code_col(lithology_df)

                if lith_hole_id and lith_from_col and lith_to_col and lith_code_col:
                    # Initialize lithology column
                    base_df['lith_code'] = None

                    # For each interval, find overlapping lithology
                    for hid in base_df[hole_id_col].unique():
                        hole_lith = lithology_df[lithology_df[lith_hole_id] == hid]
                        if hole_lith.empty:
                            continue

                        hole_mask = base_df[hole_id_col] == hid
                        hole_indices = base_df.loc[hole_mask].index

                        for idx in hole_indices:
                            interval_from = base_df.loc[idx, base_from_col]
                            interval_to = base_df.loc[idx, base_to_col]
                            interval_mid = (interval_from + interval_to) / 2.0

                            # Find lithology that contains the interval midpoint
                            mask = ((hole_lith[lith_from_col] <= interval_mid) &
                                   (hole_lith[lith_to_col] > interval_mid))
                            matching_lith = hole_lith[mask]

                            if not matching_lith.empty:
                                base_df.loc[idx, 'lith_code'] = matching_lith[lith_code_col].iloc[0]

                    # Also add sample_count if available in lithology
                    sample_count_col = find_col(lithology_df, ['SAMPLE_COUNT', 'SAMPLECOUNT', 'SAMPLES'])
                    if sample_count_col and sample_count_col not in base_df.columns:
                        base_df['sample_count'] = None
                        for hid in base_df[hole_id_col].unique():
                            hole_lith = lithology_df[lithology_df[lith_hole_id] == hid]
                            if hole_lith.empty:
                                continue
                            hole_mask = base_df[hole_id_col] == hid
                            hole_indices = base_df.loc[hole_mask].index
                            for idx in hole_indices:
                                interval_mid = base_df.loc[idx, '_MID_DEPTH']
                                mask = ((hole_lith[lith_from_col] <= interval_mid) &
                                       (hole_lith[lith_to_col] > interval_mid))
                                matching_lith = hole_lith[mask]
                                if not matching_lith.empty:
                                    base_df.loc[idx, 'sample_count'] = matching_lith[sample_count_col].iloc[0]

                    logger.info("Added lithology code by depth interval matching")

            # ================== Step 5: Add structure data if available ==================
            structures_df = drillhole_data.get('structures')
            if structures_df is not None and not structures_df.empty:
                struct_hole_id = find_hole_id_col(structures_df)
                struct_from_col, struct_to_col = find_from_to_cols(structures_df)
                struct_type_col = find_col(structures_df, ['FEATURE_TYPE', 'STRUCTURE_TYPE', 'TYPE', 'FEATURE'])

                if struct_hole_id and struct_from_col:
                    base_df['structure_type'] = None

                    for hid in base_df[hole_id_col].unique():
                        hole_struct = structures_df[structures_df[struct_hole_id] == hid]
                        if hole_struct.empty:
                            continue

                        hole_mask = base_df[hole_id_col] == hid
                        hole_indices = base_df.loc[hole_mask].index

                        for idx in hole_indices:
                            interval_from = base_df.loc[idx, base_from_col]
                            interval_to = base_df.loc[idx, base_to_col]

                            # Find structures within or overlapping the interval
                            if struct_to_col:
                                mask = ((hole_struct[struct_from_col] < interval_to) &
                                       (hole_struct[struct_to_col] > interval_from))
                            else:
                                # Point data - check if within interval
                                mask = ((hole_struct[struct_from_col] >= interval_from) &
                                       (hole_struct[struct_from_col] < interval_to))

                            matching_struct = hole_struct[mask]

                            if not matching_struct.empty and struct_type_col:
                                # Join multiple structure types if present
                                struct_types = matching_struct[struct_type_col].unique()
                                base_df.loc[idx, 'structure_type'] = ', '.join(str(s) for s in struct_types if pd.notna(s))

                    logger.info("Added structure data by depth interval matching")

            # ================== Step 6: Clean up and reorder columns ==================
            # Remove temporary columns
            if '_MID_DEPTH' in base_df.columns:
                base_df.drop('_MID_DEPTH', axis=1, inplace=True)

            # Define preferred column order for clean display
            preferred_order = [
                hole_id_col, base_from_col, base_to_col,  # Identity
                'X', 'Y', 'Z',  # Coordinates
                'LENGTH',  # Interval length
            ]

            # Add grade columns (anything numeric that's not coordinates/depth)
            grade_cols = []
            for col in base_df.columns:
                col_upper = col.upper()
                if col not in preferred_order and col_upper not in ['GLOBAL_INTERVAL_ID', 'METHOD']:
                    if base_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        if col_upper not in ['X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH', 'AZIMUTH', 'DIP',
                                            'SAMPLE_COUNT', 'CORE_THICKNESS_AT_ALPHA']:
                            grade_cols.append(col)

            preferred_order.extend(grade_cols)

            # Add method, lithology, survey, structure columns at the end
            metadata_cols = ['method', 'lith_code', 'sample_count', 'azimuth', 'dip', 'structure_type']
            for col in metadata_cols:
                if col in base_df.columns:
                    preferred_order.append(col)

            # Add any remaining columns
            for col in base_df.columns:
                if col not in preferred_order:
                    preferred_order.append(col)

            # Reorder (only include columns that exist)
            final_order = [col for col in preferred_order if col in base_df.columns]
            base_df = base_df[final_order]

            logger.info(f"Combined drillhole data: {len(base_df)} rows, {len(base_df.columns)} columns")
            logger.info(f"Final columns: {list(base_df.columns)}")
            return base_df

        except Exception as e:
            logger.error(f"Error merging drillhole data: {e}", exc_info=True)
            return None

    # ============================================================================
    # DRILLHOLES
    # ============================================================================
    def _get_or_create_drillhole_dock(self, key: str, widget_cls, title: str, stage_text: str = ""):
        """
        Create or retrieve a persistent dock for a drillhole panel.
        Panels are hidden (not destroyed) on close, preserving state.
        """
        if key in self._drillhole_panel_registry:
            dock = self._drillhole_panel_registry[key]
            dock.show()
            dock.raise_()
            dock.activateWindow()
            return dock, getattr(dock, "content_widget", None)

        try:
            content = widget_cls()
        except Exception as e:
            logger.error(f"Failed to create drillhole panel {key}: {e}", exc_info=True)
            QMessageBox.critical(self, "Panel Error", f"Could not open {title}:\n{e}")
            return None, None

        if self.controller and hasattr(content, "bind_controller"):
            try:
                content.bind_controller(self.controller)
            except Exception as e:
                logger.warning(f"Failed to bind controller to {key}: {e}", exc_info=True)

        dock = PersistentDockWidget(key, title, content, stage_text, parent=self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._drillhole_panel_registry[key] = dock
        dock.show()
        dock.raise_()
        dock.activateWindow()
        return dock, content

    def open_domain_compositing_panel(self):
        """Open Drillhole Loading panel in a separate window."""
        # Check if panel already exists and is visible
        if hasattr(self, 'domain_compositing_dialog') and self._is_dialog_valid(self.domain_compositing_dialog):
            if self.domain_compositing_dialog.isVisible():
                self.domain_compositing_dialog.raise_()
                self.domain_compositing_dialog.activateWindow()
            else:
                # Window exists but is minimized - restore it
                self.domain_compositing_dialog.show()
                self.domain_compositing_dialog.raise_()
                self.domain_compositing_dialog.activateWindow()
            return
        elif hasattr(self, 'domain_compositing_dialog'):
            # Dialog was destroyed - clear reference
            self.domain_compositing_dialog = None

        # Create new panel window
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        self.domain_compositing_dialog = QDialog(None)  # No parent - independent window
        self.domain_compositing_dialog.setWindowTitle("Drillhole Loading")
        # Set dialog size - panel won't set its own size when embedded
        self.domain_compositing_dialog.resize(1200, 800)
        self.domain_compositing_dialog.setMinimumSize(800, 600)
        self.domain_compositing_dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.domain_compositing_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.domain_compositing_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored)
        self._setup_dialog_persistence(self.domain_compositing_dialog, 'domain_compositing_dialog')

        layout = QVBoxLayout(self.domain_compositing_dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create panel WITH the dialog as parent so it knows it's embedded
        # This prevents window flags from being set incorrectly
        self.domain_compositing_panel = DrillholeImportPanel(parent=self.domain_compositing_dialog)
        # Ensure panel expands to fill dialog
        from PyQt6.QtWidgets import QSizePolicy
        self.domain_compositing_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.domain_compositing_panel)

        # Bind controller if available (though panel can work without it)
        if self.controller:
            self.domain_compositing_panel.bind_controller(self.controller)

        logger.info("Opened Drillhole Loading panel in separate window")

        # Show as non-modal dialog
        self.domain_compositing_dialog.show()

    def open_loopstructural_panel(self):
        """Open LoopStructural Geological Modeling panel in a separate window."""
        try:
            from .loopstructural_panel import LoopStructuralModelPanel

            # Check if dialog already exists and is valid
            if hasattr(self, 'loopstructural_dialog') and self._is_dialog_valid(self.loopstructural_dialog):
                if self.loopstructural_dialog.isVisible():
                    self.loopstructural_dialog.show()
                    self.loopstructural_dialog.raise_()
                    self.loopstructural_dialog.activateWindow()
                    return
                else:
                    self.loopstructural_dialog.show()
                    self.loopstructural_dialog.raise_()
                    self.loopstructural_dialog.activateWindow()
                return
            elif hasattr(self, 'loopstructural_dialog'):
                self.loopstructural_dialog = None

            # Create new panel window
            from PyQt6.QtWidgets import QDialog, QVBoxLayout

            self.loopstructural_dialog = QDialog(None)
            self.loopstructural_dialog.setWindowTitle("LoopStructural Geological Modeling")
            self.loopstructural_dialog.resize(1000, 800)
            self.loopstructural_dialog.setMinimumSize(900, 700)
            self.loopstructural_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )
            self.loopstructural_dialog.setWindowModality(Qt.WindowModality.NonModal)
            self.loopstructural_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.loopstructural_dialog, 'loopstructural_dialog')

            layout = QVBoxLayout(self.loopstructural_dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            # Create panel
            self.loopstructural_panel = LoopStructuralModelPanel(parent=self.loopstructural_dialog)
            from PyQt6.QtWidgets import QSizePolicy
            self.loopstructural_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(self.loopstructural_panel)

            # Bind controller if available
            if self.controller:
                self.loopstructural_panel.bind_controller(self.controller)

            # Connect geology package signal to main renderer
            self.loopstructural_panel.geology_package_ready.connect(self._on_geology_build_complete)

            logger.info("Opened LoopStructural Geological Modeling panel")

            self.loopstructural_dialog.show()

        except ImportError as e:
            logger.error(f"LoopStructural not available: {e}")
            QMessageBox.warning(
                self,
                "LoopStructural Not Available",
                "LoopStructural library is not installed.\n\n"
                "Install with: pip install LoopStructural>=1.6.0"
            )
        except Exception as e:
            logger.error(f"Failed to open LoopStructural panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open LoopStructural panel:\n{e}")

    def _on_geology_build_complete(self, result_package: dict):
        """
        Handle completion of geological model build from LoopStructural panel.
        
        Sends the geology package directly to the MAIN renderer, ensuring
        surfaces and solids appear in the central 3D viewer alongside
        drillholes and block models.
        
        Args:
            result_package: Dictionary with 'surfaces', 'solids', 'report', 'log'
        """
        try:
            logger.info("MainWindow: Received geology package from LoopStructural panel")

            # Load geology surfaces/solids into the main renderer
            # Use render_mode="both" to create all layers (surfaces + solids)
            # The GeologicalExplorer panel can then toggle visibility via apply_view_mode()
            if self.viewer_widget and self.viewer_widget.renderer:
                try:
                    self.viewer_widget.renderer.load_geology_package(result_package, render_mode="both")
                    logger.info("MainWindow: Geology surfaces and solids rendered in main viewer")
                except Exception as e:
                    logger.error(f"Failed to render geology in main viewer: {e}", exc_info=True)

            # Update Geological Explorer panel with new data
            if hasattr(self, 'geological_explorer_panel') and self.geological_explorer_panel:
                try:
                    self.geological_explorer_panel.on_geology_package_loaded(result_package)
                    # Show the dock if hidden
                    if hasattr(self, 'geological_explorer_dock') and self.geological_explorer_dock:
                        self.geological_explorer_dock.show()
                        self.geological_explorer_dock.raise_()
                    logger.info("Geological Explorer panel updated with geology package")
                except Exception as e:
                    logger.debug(f"Could not update Geological Explorer panel: {e}")

            # Update the property panel so the user can toggle layers
            if self.property_panel and hasattr(self.property_panel, 'update_layer_controls'):
                try:
                    self.property_panel.update_layer_controls()
                except Exception as e:
                    logger.debug(f"Could not update property panel: {e}")

            logger.info("MainWindow: Geology package notification received (already loaded by panel)")

            # 3. Fit view to geology bounds (panel already loaded the data)
            if self.viewer_widget and self.viewer_widget.renderer:
                try:
                    self.viewer_widget.renderer.fit_to_view()
                except Exception as e:
                    logger.debug(f"Could not fit to view: {e}")

            # 4. Log JORC Status in status bar
            report = result_package.get('report') or result_package.get('audit_report')
            if report:
                status = getattr(report, 'status', 'Unknown')
                p90 = getattr(report, 'p90_error', 0)
                self.status_bar.showMessage(
                    f"Geological Model Integrated: {status} (P90: {p90:.2f}m)",
                    10000  # Show for 10 seconds
                )
            else:
                surfaces_count = len(result_package.get('surfaces', []))
                solids_count = len(result_package.get('solids', []))
                self.status_bar.showMessage(
                    f"Geological Model: {surfaces_count} surfaces, {solids_count} solids loaded",
                    10000
                )

        except Exception as e:
            logger.error(f"MainWindow: Failed to plot geology in main viewer: {e}", exc_info=True)
            self.status_bar.showMessage(f"Geology integration error: {e}", 5000)

    def _show_loopstructural_info(self):
        """Show information about LoopStructural geological modeling."""
        QMessageBox.information(
            self,
            "About LoopStructural Modeling",
            "<h3>LoopStructural Geological Modeling</h3>"
            "<p>Industry-grade implicit geological modeling with JORC/SAMREC compliance.</p>"
            "<h4>Features:</h4>"
            "<ul>"
            "<li>Finite Difference Interpolation (FDI) for layered rocks</li>"
            "<li>Fault displacement fields</li>"
            "<li>Automatic coordinate normalization for UTM stability</li>"
            "<li>JORC/SAMREC compliant misfit auditing</li>"
            "<li>Automatic fault detection from model errors</li>"
            "<li>Watertight mesh extraction for volume calculation</li>"
            "</ul>"
            "<h4>Workflow:</h4>"
            "<ol>"
            "<li>Load contact and orientation data</li>"
            "<li>Configure stratigraphic sequence</li>"
            "<li>Add fault events (optional)</li>"
            "<li>Build model</li>"
            "<li>Validate compliance</li>"
            "<li>Export surfaces</li>"
            "</ol>"
            "<p><i>Powered by LoopStructural - https://loop3d.org</i></p>"
        )

    def open_block_model_import_panel(self):
        """Open Block Model Loading panel in a separate window."""
        try:
            from .block_model_import_panel import BlockModelImportPanel

            # Check if dialog already exists and is valid
            if hasattr(self, 'block_model_import_dialog') and self._is_dialog_valid(self.block_model_import_dialog):
                if self.block_model_import_dialog.isVisible():
                    # Window exists and is visible - just raise it
                    self.block_model_import_dialog.show()
                    self.block_model_import_dialog.raise_()
                    self.block_model_import_dialog.activateWindow()
                    return
                else:
                    # Window exists but is minimized - restore it
                    self.block_model_import_dialog.show()
                    self.block_model_import_dialog.raise_()
                    self.block_model_import_dialog.activateWindow()
                return
            elif hasattr(self, 'block_model_import_dialog'):
                # Dialog was destroyed - clear reference
                self.block_model_import_dialog = None

            # Create new panel window
            from PyQt6.QtWidgets import QDialog, QVBoxLayout

            self.block_model_import_dialog = QDialog(None)  # No parent - independent window
            self.block_model_import_dialog.setWindowTitle("Block Model Loading")
            # Set dialog size - panel won't set its own size when embedded
            self.block_model_import_dialog.resize(1200, 800)
            self.block_model_import_dialog.setMinimumSize(800, 600)
            self.block_model_import_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )
            self.block_model_import_dialog.setWindowModality(Qt.WindowModality.NonModal)
            self.block_model_import_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Setup dialog persistence (position/size will be saved/restored)
            self._setup_dialog_persistence(self.block_model_import_dialog, 'block_model_import_dialog')

            layout = QVBoxLayout(self.block_model_import_dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            # Create panel WITH the dialog as parent so it knows it's embedded
            # This prevents window flags from being set incorrectly
            self.block_model_import_panel = BlockModelImportPanel(parent=self.block_model_import_dialog)
            # Ensure panel expands to fill dialog
            from PyQt6.QtWidgets import QSizePolicy
            self.block_model_import_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(self.block_model_import_panel)

            # Bind controller if available (though panel can work without it)
            if self.controller:
                self.block_model_import_panel.bind_controller(self.controller)

            logger.info("Opened Block Model Loading panel in separate window")

            # Show as non-modal dialog
            self.block_model_import_dialog.show()
        except Exception as e:
            logger.error(f"Failed to open Block Model Loading panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Block Model Loading panel:\n{e}")

    def open_compositing_window(self):
        """Open Drillhole Compositing window."""
        try:
            from ..drillholes.compositing_engine import CompositingMethodEngine
            from ..drillholes.compositing_utils import get_intervals_from_registry
            from .compositing_window import CompositingWindow

            # Get registry reference first (needed for both new and existing window cases)
            registry = self.controller.registry if self.controller else None
            if not registry:
                from ..core.data_registry import DataRegistry
                registry = DataRegistry.instance()
            drillhole_data = registry.get_drillhole_data() if registry else None

            # Check if window already exists
            if hasattr(self, 'compositing_window') and self.compositing_window is not None:
                # Refresh intervals from current registry data to ensure fresh data
                # Pass registry so excluded_rows from validation state can be applied
                intervals = get_intervals_from_registry(drillhole_data=drillhole_data, registry=registry)
                self.compositing_window.refresh_intervals(intervals)

                if self.compositing_window.isVisible():
                    self.compositing_window.raise_()
                    self.compositing_window.activateWindow()
                else:
                    self.compositing_window.show()
                    self.compositing_window.raise_()
                    self.compositing_window.activateWindow()
                return

            # Convert drillhole data to intervals
            # Pass registry so excluded_rows from validation state can be applied
            intervals = get_intervals_from_registry(drillhole_data=drillhole_data, registry=registry)

            if not intervals:
                reply = QMessageBox.question(
                    self,
                    "No Drillhole Data",
                    "No drillhole data found in DataRegistry.\n\n"
                    "Please load drillhole data first via:\n"
                    "Drillholes → Drillhole Loading\n\n"
                    "Or run QC to clean existing data.\n\n"
                    "Do you want to open the window anyway (demo mode)?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
                intervals = []

            # Create compositing engine
            engine = CompositingMethodEngine()

            # Create window
            self.compositing_window = CompositingWindow(
                intervals=intervals,
                compositing_engine=engine,
                parent=self,
                controller=self.controller
            )

            # Setup persistence and close protection
            self._setup_dialog_persistence(self.compositing_window, 'compositing_window')

            # Apply saved compositing settings if any (from project load)
            if hasattr(self, '_saved_compositing_settings') and self._saved_compositing_settings:
                try:
                    self.compositing_window.apply_settings_state(self._saved_compositing_settings)
                    logger.info("Applied saved compositing settings to new window")
                except Exception as e:
                    logger.warning(f"Could not apply saved compositing settings: {e}")
                finally:
                    self._saved_compositing_settings = None

            self.compositing_window.show()
            self.status_bar.showMessage(f"Compositing window opened with {len(intervals)} intervals", 2000)
            logger.info(f"Opened Compositing window with {len(intervals)} intervals")

        except Exception as e:
            logger.error(f"Failed to open Compositing window: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Compositing window:\n{e}")

    def open_drillhole_qc_window(self):
        """Open Drillhole QC Window for data validation and quality control."""
        # Check if window already exists (even if minimized)
        if hasattr(self, 'qc_window') and self.qc_window:
            if self.qc_window.isVisible():
                self.qc_window.raise_()
                self.qc_window.activateWindow()
            else:
                # Window exists but is minimized - restore it
                self.qc_window.show()
                self.qc_window.raise_()
                self.qc_window.activateWindow()
            return

        # 1. Gather Data (Try to get from DataRegistry or existing panels)
        collars = None
        surveys = None
        assays = None
        lithology = None

        # Try fetching from domain compositing panel first (common source)
        if hasattr(self, 'domain_compositing_panel') and self.domain_compositing_panel:
            collars = getattr(self.domain_compositing_panel, 'collar_df', None)
            surveys = getattr(self.domain_compositing_panel, 'survey_df', None)
            assays = getattr(self.domain_compositing_panel, 'assay_df', None)
            # lithology = ... (if available)

        # If not found, check DataRegistry
        if collars is None or collars.empty:
            # Use injected registry via controller (dependency injection)
            registry = self.controller.registry if self.controller else None
            if registry:
                data = registry.get_drillhole_data()
                if isinstance(data, dict):
                    collars = data.get('collars')
                    surveys = data.get('surveys')
                    assays = data.get('assays')
                    lithology = data.get('lithology')

        if collars is None or collars.empty:
            QMessageBox.warning(
                self, "No Data",
                "Please load drillhole data (Collars, Surveys, Assays) first.\n\n"
                "Use 'Drillholes > Drillhole Loading' to load data."
            )
            return

        # 2. Map column names to validation engine expected format
        # Validation engine expects: hole_id, easting, northing, elevation, total_depth
        # Domain compositing uses: HOLEID, X, Y, Z, DEPTH (uppercase)
        # Also handle lowercase variants: hole_id, x, y, z, length, depth_from, depth_to, from_depth, to_depth

        collars_mapped = collars.copy()

        # Map hole_id (handle both HOLEID and hole_id)
        if 'HOLEID' in collars_mapped.columns and 'hole_id' not in collars_mapped.columns:
            collars_mapped['hole_id'] = collars_mapped['HOLEID'].copy()
        elif 'hole_id' not in collars_mapped.columns:
            # Try to find any column that might be hole_id
            hole_cols = [c for c in collars_mapped.columns if 'hole' in c.lower() or (c.lower() == 'id' or c.lower().endswith('_id'))]
            if hole_cols:
                collars_mapped['hole_id'] = collars_mapped[hole_cols[0]].copy()
            else:
                QMessageBox.warning(
                    self, "Invalid Collar Data",
                    f"Could not find 'hole_id' column in collar data.\n\n"
                    f"Available columns: {', '.join(collars_mapped.columns)}"
                )
                return

        # Map coordinates (handle both uppercase and lowercase, and common variations)
        # Create case-insensitive column lookup
        col_lower_map = {col.lower(): col for col in collars_mapped.columns}

        # Map easting
        if 'easting' not in collars_mapped.columns:
            for alt_name in ['x', 'east', 'easting']:
                if alt_name in col_lower_map:
                    collars_mapped['easting'] = collars_mapped[col_lower_map[alt_name]]
                    break

        # Map northing
        if 'northing' not in collars_mapped.columns:
            for alt_name in ['y', 'north', 'northing']:
                if alt_name in col_lower_map:
                    collars_mapped['northing'] = collars_mapped[col_lower_map[alt_name]]
                    break

        # Map elevation (handle RL, Z, ELEVATION, etc.)
        if 'elevation' not in collars_mapped.columns:
            for alt_name in ['z', 'rl', 'elev', 'elevation', 'elevation']:
                if alt_name in col_lower_map:
                    collars_mapped['elevation'] = collars_mapped[col_lower_map[alt_name]]
                    break

        # Map total_depth (handle MAX_DEPTH, DEPTH, depth, length, etc.)
        if 'total_depth' not in collars_mapped.columns:
            for alt_name in ['total_depth', 'max_depth', 'depth', 'length', 'totaldepth', 'maxdepth', 'eoh']:
                if alt_name in col_lower_map:
                    collars_mapped['total_depth'] = collars_mapped[col_lower_map[alt_name]]
                    break

        # Map surveys: handle DEPTH_FROM/DEPTH_TO, depth_from/depth_to, or DEPTH/depth
        if not PANDAS_AVAILABLE:
            QMessageBox.warning(self, "Pandas Required", "Pandas is required for drillhole data mapping.")
            return
        surveys_mapped = surveys.copy() if surveys is not None and not surveys.empty else pd.DataFrame()
        if not surveys_mapped.empty:
            # Map hole_id
            if 'HOLEID' in surveys_mapped.columns and 'hole_id' not in surveys_mapped.columns:
                surveys_mapped['hole_id'] = surveys_mapped['HOLEID'].copy()
            elif 'hole_id' not in surveys_mapped.columns:
                hole_cols = [c for c in surveys_mapped.columns if 'hole' in c.lower() or (c.lower() == 'id' or c.lower().endswith('_id'))]
                if hole_cols:
                    surveys_mapped['hole_id'] = surveys_mapped[hole_cols[0]].copy()
                else:
                    logger.warning(f"Could not find hole_id in surveys. Columns: {list(surveys_mapped.columns)}")

            # Map depth
            if 'DEPTH_FROM' in surveys_mapped.columns and 'DEPTH_TO' in surveys_mapped.columns:
                surveys_mapped['depth'] = (surveys_mapped['DEPTH_FROM'] + surveys_mapped['DEPTH_TO']) / 2
            elif 'DEPTH_FROM' in surveys_mapped.columns and 'depth' not in surveys_mapped.columns:
                surveys_mapped['depth'] = surveys_mapped['DEPTH_FROM']
            elif 'DEPTH' in surveys_mapped.columns and 'depth' not in surveys_mapped.columns:
                surveys_mapped['depth'] = surveys_mapped['DEPTH']
            elif 'depth_from' in surveys_mapped.columns and 'depth_to' in surveys_mapped.columns:
                surveys_mapped['depth'] = (surveys_mapped['depth_from'] + surveys_mapped['depth_to']) / 2
            elif 'depth_from' in surveys_mapped.columns and 'depth' not in surveys_mapped.columns:
                surveys_mapped['depth'] = surveys_mapped['depth_from']
            elif 'depth' not in surveys_mapped.columns:
                # Try to find any depth-like column
                depth_cols = [c for c in surveys_mapped.columns if 'depth' in c.lower()]
                if depth_cols:
                    surveys_mapped['depth'] = surveys_mapped[depth_cols[0]]

            # Map dip and azimuth
            if 'DIP' in surveys_mapped.columns and 'dip' not in surveys_mapped.columns:
                surveys_mapped['dip'] = surveys_mapped['DIP']
            if 'AZI' in surveys_mapped.columns and 'azimuth' not in surveys_mapped.columns:
                surveys_mapped['azimuth'] = surveys_mapped['AZI']
            elif 'AZIMUTH' in surveys_mapped.columns and 'azimuth' not in surveys_mapped.columns:
                surveys_mapped['azimuth'] = surveys_mapped['AZIMUTH']

        # Map assays: handle FROM/TO (uppercase) or from_depth/to_depth (lowercase)
        assays_mapped = assays.copy() if assays is not None and not assays.empty else pd.DataFrame()
        if not assays_mapped.empty:
            # Map hole_id
            if 'HOLEID' in assays_mapped.columns and 'hole_id' not in assays_mapped.columns:
                assays_mapped['hole_id'] = assays_mapped['HOLEID'].copy()
            elif 'hole_id' not in assays_mapped.columns:
                hole_cols = [c for c in assays_mapped.columns if 'hole' in c.lower() or (c.lower() == 'id' or c.lower().endswith('_id'))]
                if hole_cols:
                    assays_mapped['hole_id'] = assays_mapped[hole_cols[0]].copy()
                else:
                    logger.warning(f"Could not find hole_id in assays. Columns: {list(assays_mapped.columns)}")

            # Map from_depth/to_depth - handle multiple naming conventions
            if 'FROM' in assays_mapped.columns and 'from_depth' not in assays_mapped.columns:
                assays_mapped['from_depth'] = assays_mapped['FROM']
            elif 'depth_from' in assays_mapped.columns and 'from_depth' not in assays_mapped.columns:
                assays_mapped['from_depth'] = assays_mapped['depth_from']
            elif 'from_depth' not in assays_mapped.columns:
                # Try to find any from-like column
                from_cols = [c for c in assays_mapped.columns if 'from' in c.lower() and 'depth' in c.lower()]
                if from_cols:
                    assays_mapped['from_depth'] = assays_mapped[from_cols[0]]
                else:
                    QMessageBox.warning(
                        self, "Invalid Assay Data",
                        f"Assay data must have 'from_depth', 'depth_from', or 'FROM' column.\n\n"
                        f"Available columns: {', '.join(assays_mapped.columns)}"
                    )
                    return

            if 'TO' in assays_mapped.columns and 'to_depth' not in assays_mapped.columns:
                assays_mapped['to_depth'] = assays_mapped['TO']
            elif 'depth_to' in assays_mapped.columns and 'to_depth' not in assays_mapped.columns:
                assays_mapped['to_depth'] = assays_mapped['depth_to']
            elif 'to_depth' not in assays_mapped.columns:
                # Try to find any to-like column
                to_cols = [c for c in assays_mapped.columns if 'to' in c.lower() and 'depth' in c.lower()]
                if to_cols:
                    assays_mapped['to_depth'] = assays_mapped[to_cols[0]]
                else:
                    QMessageBox.warning(
                        self, "Invalid Assay Data",
                        f"Assay data must have 'to_depth', 'depth_to', or 'TO' column.\n\n"
                        f"Available columns: {', '.join(assays_mapped.columns)}"
                    )
                    return

        # Map lithology: ensure from_depth/to_depth exist
        lithology_mapped = lithology.copy() if lithology is not None and not lithology.empty else pd.DataFrame()
        if not lithology_mapped.empty:
            # Map hole_id
            if 'HOLEID' in lithology_mapped.columns and 'hole_id' not in lithology_mapped.columns:
                lithology_mapped['hole_id'] = lithology_mapped['HOLEID'].copy()
            elif 'hole_id' not in lithology_mapped.columns:
                hole_cols = [c for c in lithology_mapped.columns if 'hole' in c.lower() or (c.lower() == 'id' or c.lower().endswith('_id'))]
                if hole_cols:
                    lithology_mapped['hole_id'] = lithology_mapped[hole_cols[0]].copy()

            # Map from_depth/to_depth
            if 'FROM' in lithology_mapped.columns and 'from_depth' not in lithology_mapped.columns:
                lithology_mapped['from_depth'] = lithology_mapped['FROM']
            elif 'depth_from' in lithology_mapped.columns and 'from_depth' not in lithology_mapped.columns:
                lithology_mapped['from_depth'] = lithology_mapped['depth_from']

            if 'TO' in lithology_mapped.columns and 'to_depth' not in lithology_mapped.columns:
                lithology_mapped['to_depth'] = lithology_mapped['TO']
            elif 'depth_to' in lithology_mapped.columns and 'to_depth' not in lithology_mapped.columns:
                lithology_mapped['to_depth'] = lithology_mapped['depth_to']

        # 3. Validate required columns exist before creating QC Window (case-insensitive check)
        required_collar_cols = ['hole_id', 'easting', 'northing', 'elevation', 'total_depth']
        col_lower_set = {col.lower() for col in collars_mapped.columns}
        missing_collar_cols = [col for col in required_collar_cols if col.lower() not in col_lower_set]
        if missing_collar_cols:
            QMessageBox.warning(
                self, "Invalid Collar Data",
                f"Collars missing required columns: {', '.join(missing_collar_cols)}\n\n"
                f"Available columns: {', '.join(collars_mapped.columns)}"
            )
            return

        # Ensure hole_id exists in all non-empty dataframes
        if not surveys_mapped.empty and 'hole_id' not in surveys_mapped.columns:
            QMessageBox.warning(
                self, "Invalid Survey Data",
                f"Surveys missing 'hole_id' column.\n\n"
                f"Available columns: {', '.join(surveys_mapped.columns)}"
            )
            return

        if not assays_mapped.empty and 'hole_id' not in assays_mapped.columns:
            QMessageBox.warning(
                self, "Invalid Assay Data",
                f"Assays missing 'hole_id' column.\n\n"
                f"Available columns: {', '.join(assays_mapped.columns)}"
            )
            return

        if not lithology_mapped.empty and 'hole_id' not in lithology_mapped.columns:
            QMessageBox.warning(
                self, "Invalid Lithology Data",
                f"Lithology missing 'hole_id' column.\n\n"
                f"Available columns: {', '.join(lithology_mapped.columns)}"
            )
            return

        # 4. Create QC Window
        from ..drillholes.drillhole_validation import ValidationConfig

        cfg = ValidationConfig(
            max_interval_gap=0.10,
            max_small_overlap=0.02,
            standard_sample_length=1.0,
        )

        self.qc_window = QCWindow(
            collars=collars_mapped,
            surveys=surveys_mapped,
            assays=assays_mapped,
            lithology=lithology_mapped,
            cfg=cfg,
            user="GEOLOGIST",
            parent=None,  # No parent for independent window
            controller=self.controller
        )

        # Set window flags for proper minimize behavior (stay in taskbar)
        self.qc_window.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        # Ensure non-modal behavior
        self.qc_window.setWindowModality(Qt.WindowModality.NonModal)

        # Prevent window from being deleted when closed or minimized
        self.qc_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored)
        self._setup_dialog_persistence(self.qc_window, 'qc_window')

        logger.info("Opened Drillhole QC Window")
        self.qc_window.show()

    def open_drillhole_reporting_panel(self):
        """Open Drillhole Reporting panel as a persistent dock."""
        dock, panel = self._get_or_create_drillhole_dock(
            "drillhole_reporting",
            DrillholeReportingPanel,
            "Drillhole Reporting",
            "Using drillholes from: Drillhole Database Management",
        )
        if dock is None or panel is None:
            return
        self.drillhole_reporting_panel = panel


    def open_drillhole_plotting_panel(self):
        """Open Drillhole Plotting panel in a separate window."""
        # Check if panel already exists and is valid
        if hasattr(self, 'drillhole_plotting_dialog') and self._is_dialog_valid(self.drillhole_plotting_dialog):
            try:
                if self.drillhole_plotting_dialog.isVisible():
                    self.drillhole_plotting_dialog.raise_()
                    self.drillhole_plotting_dialog.activateWindow()
                else:
                    # Window exists but is minimized - restore it
                    self.drillhole_plotting_dialog.show()
                    self.drillhole_plotting_dialog.raise_()
                    self.drillhole_plotting_dialog.activateWindow()
                return
            except (AttributeError, RuntimeError):
                # Dialog was deleted or is invalid, recreate it
                self.drillhole_plotting_dialog = None

        # Create new panel window
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        self.drillhole_plotting_dialog = QDialog(None)  # No parent - independent window
        self.drillhole_plotting_dialog.setWindowTitle("Drillhole Plotting")
        # Sizing is handled by BaseAnalysisPanel._setup_panel_sizing()
        self.drillhole_plotting_dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.drillhole_plotting_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.drillhole_plotting_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored)
        self._setup_dialog_persistence(self.drillhole_plotting_dialog, 'drillhole_plotting_dialog')

        layout = QVBoxLayout(self.drillhole_plotting_dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create panel
        self.drillhole_plotting_panel = DrillholePlottingPanel()
        layout.addWidget(self.drillhole_plotting_panel)

        # Bind controller if available
        if self.controller:
            self.drillhole_plotting_panel.bind_controller(self.controller)

    def open_grade_transformation_panel(self):
        """Open Grade Transformation panel in a separate window."""
        # Check if panel already exists and is visible
        if hasattr(self, 'grade_transformation_dialog') and self._is_dialog_valid(self.grade_transformation_dialog):
            if self.grade_transformation_dialog.isVisible():
                self.grade_transformation_dialog.raise_()
                self.grade_transformation_dialog.activateWindow()
            else:
                # Window exists but is minimized - restore it
                self.grade_transformation_dialog.show()
                self.grade_transformation_dialog.raise_()
                self.grade_transformation_dialog.activateWindow()
            return

        # Create new panel window
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        self.grade_transformation_dialog = QDialog(None)  # No parent - independent window
        self.grade_transformation_dialog.setWindowTitle("Grade Data Transformation")
        self.grade_transformation_dialog.resize(1200, 800)
        self.grade_transformation_dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint
            # Close button removed - use close protection instead
        )
        self.grade_transformation_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.grade_transformation_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored) and close protection
        self._setup_dialog_persistence(self.grade_transformation_dialog, 'grade_transformation_dialog', 'Grade Transformation')

        layout = QVBoxLayout(self.grade_transformation_dialog)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create panel with reference to main window
        self.grade_transformation_panel = GradeTransformationPanel(main_window=self)
        layout.addWidget(self.grade_transformation_panel)

        # Try to get drillhole data
        drillhole_data = None

        # Method 1: Check domain compositing panel for composite / assay data
        if drillhole_data is None and hasattr(self, 'domain_compositing_panel') and self.domain_compositing_panel:
            # Try raw assay data
            if hasattr(self.domain_compositing_panel, 'assay_df'):
                assay_df = self.domain_compositing_panel.assay_df
                if assay_df is not None and not assay_df.empty:
                    drillhole_data = assay_df
                    logger.info(f"Loaded assay data from drillhole loading panel: {len(drillhole_data)} rows")

        # Method 2: Check stored drillhole data
        if drillhole_data is None and hasattr(self, 'stored_drillhole_data') and self.stored_drillhole_data is not None:
            drillhole_data = self.stored_drillhole_data
            logger.info(f"Loaded stored drillhole data: {len(drillhole_data)} rows")

        # Method 3: Check renderer for drillhole data
        if drillhole_data is None and hasattr(self.viewer_widget, 'renderer'):
            if hasattr(self.viewer_widget.renderer, 'drillhole_data'):
                dh_data = self.viewer_widget.renderer.drillhole_data
                if dh_data is not None:
                    if isinstance(dh_data, dict) and 'mesh' in dh_data:
                        # Extract data from mesh
                        mesh = dh_data['mesh']
                        # Try to get point data
                        if hasattr(mesh, 'point_data'):
                            # This is complex - would need to reconstruct DataFrame from mesh
                            pass
                    elif PANDAS_AVAILABLE and isinstance(dh_data, pd.DataFrame):
                        drillhole_data = dh_data
                        logger.info(f"Loaded drillhole data from renderer: {len(drillhole_data)} rows")

        # Method 4: DataRegistry (central source used by database panel)
        if drillhole_data is None:
            try:
                # Use injected registry via controller (dependency injection)
                registry = self.controller.registry if self.controller else None
                if registry is not None:
                    reg_data = registry.get_drillhole_data()
                    if isinstance(reg_data, dict):
                        comp = reg_data.get("composites")
                        assays = reg_data.get("assays")
                        if comp is not None and getattr(comp, "empty", False) is False:
                            drillhole_data = comp
                            logger.info(f"Loaded composites from DataRegistry: {len(drillhole_data)} rows")
                        elif assays is not None and getattr(assays, "empty", False) is False:
                            drillhole_data = assays
                            logger.info(f"Loaded assays from DataRegistry: {len(drillhole_data)} rows")
                    elif PANDAS_AVAILABLE and isinstance(reg_data, pd.DataFrame):
                        drillhole_data = reg_data
                        logger.info(f"Loaded drillhole DataFrame from DataRegistry: {len(drillhole_data)} rows")
            except Exception as e:
                logger.warning(f"Grade Transformation: failed to pull data from DataRegistry: {e}", exc_info=True)

        # Method 5: Current drillhole database (collars/assays/composites) if loaded
        # (Removed - database panel no longer available)

        # Data is now fetched from registry - panels connect to registry.drillholeDataLoaded signal
        # No need to call set_drillhole_data() - panels will refresh automatically when registry updates
        if drillhole_data is None:
            QMessageBox.information(
                self,
                "No Drillhole Data",
                "No drillhole data found.\n\n"
                "Please load drillhole data first:\n"
                "Drillholes → Drillhole Loading\n\n"
                "Or ensure drillholes have been visualized with XYZ coordinates."
            )

        logger.info("Opened Grade Transformation panel in separate window")

        # Show as non-modal dialog
        self.grade_transformation_dialog.show()

    def open_grade_transformation_panel_with_editor_data(self, assays_df: "pd.DataFrame"):
        """
        Open Grade Transformation panel using assays provided by the Drillhole Editor.
        
        The provided DataFrame should contain grade columns and may include
        HOLEID/FROM/TO for context.
        """
        self.open_grade_transformation_panel()

    # ============================================================================
    # RESOURCES
    # ============================================================================

    def open_block_resource_panel(self):
        """Open Block Model Resource calculation panel in a separate window."""
        # Check if panel already exists and is visible
        if hasattr(self, 'block_resource_dialog') and self._is_dialog_valid(self.block_resource_dialog):
            if self.block_resource_dialog.isVisible():
                self.block_resource_dialog.raise_()
                self.block_resource_dialog.activateWindow()
            else:
                # Window exists but is minimized - restore it
                self.block_resource_dialog.show()
                self.block_resource_dialog.raise_()
                self.block_resource_dialog.activateWindow()
            return

        # Create new panel window
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        self.block_resource_dialog = QDialog(None)  # No parent - independent window
        self.block_resource_dialog.setWindowTitle("Block Model Resources")
        self.block_resource_dialog.resize(600, 800)
        self.block_resource_dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.block_resource_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.block_resource_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored)
        self._setup_dialog_persistence(self.block_resource_dialog, 'block_resource_dialog')

        layout = QVBoxLayout(self.block_resource_dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create panel
        self.block_resource_panel = BlockModelResourcePanel()
        layout.addWidget(self.block_resource_panel)

        # Bind controller
        if self.controller:
            self.block_resource_panel.bind_controller(self.controller)

        # Connect signals
        self.block_resource_panel.highlight_blocks_requested.connect(
            self.on_block_resource_highlight
        )
        self.block_resource_panel.visualize_classification_requested.connect(
            self.on_visualize_resource_classification
        )

        # Set block model if available
        if self.current_model:
            self.block_resource_panel.set_block_model(self.current_model)

        logger.info("Opened Block Model Resource panel in separate window")

        # Show as non-modal dialog
        self.block_resource_dialog.show()


    def open_drillhole_resource_panel(self):
        """
        DrillholeResourcePanel has been removed - functionality is redundant.
        
        Use BlockModelResourcePanel instead:
        1. Build a block model from drillholes (Analysis → Resource Modelling → Build Block Model)
        2. Calculate resources using BlockModelResourcePanel (Analysis → Resource Modelling → Resource Calculator)
        """
        QMessageBox.information(
            self,
            "Panel Removed",
            "DrillholeResourcePanel has been removed as it was redundant.\n\n"
            "To calculate resources from drillholes:\n"
            "1. Build a block model from drillholes first\n"
            "2. Use BlockModelResourcePanel for resource calculation\n\n"
            "This consolidates resource calculation into a single, unified panel."
        )

    def open_sensitivity_panel(self):
        """Open Cut-off Sensitivity Analysis panel (placeholder)."""
        QMessageBox.information(
            self,
            "Sensitivity Analysis",
            "Cut-off Sensitivity Analysis\n\n"
            "This feature will allow you to:\n"
            "â€¢ Generate grade-tonnage curves\n"
            "â€¢ Analyze cut-off sensitivity for both block models and drillholes\n"
            "â€¢ Export results for reporting\n\n"
            "Coming soon!"
        )

    def open_irr_panel(self):
        """Open IRR Optimization panel in a separate window."""
        # Check if panel already exists and is visible
        if hasattr(self, 'irr_dialog') and self._is_dialog_valid(self.irr_dialog):
            if self.irr_dialog.isVisible():
                self.irr_dialog.raise_()
                self.irr_dialog.activateWindow()
            else:
                # Window exists but is minimized - restore it
                self.irr_dialog.show()
                self.irr_dialog.raise_()
                self.irr_dialog.activateWindow()
            return

        # Create new panel window
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        self.irr_dialog = QDialog(None)  # No parent - independent window
        self.irr_dialog.setWindowTitle("IRR Optimization - Risk-Adjusted Internal Rate of Return")

        # Set size to 85% of screen size for better fit
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        dialog_width = min(950, int(screen.width() * 0.85))
        dialog_height = min(700, int(screen.height() * 0.85))
        self.irr_dialog.resize(dialog_width, dialog_height)

        self.irr_dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.irr_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.irr_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored)
        self._setup_dialog_persistence(self.irr_dialog, 'irr_dialog')

        layout = QVBoxLayout(self.irr_dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create panel
        self.irr_panel = IRRPanel()

        # Bind controller
        if self.controller:
            self.irr_panel.bind_controller(self.controller)

        # Override the load method to use our current_model
        def load_from_main_window():
            if self.current_model:
                df = self.current_model.to_dataframe()
                self.irr_panel.set_block_model(df)
                logger.info(f"Loaded block model into IRR panel: {len(df)} blocks")
            else:
                QMessageBox.warning(
                    self.irr_panel,
                    "No Block Model",
                    "Please load a block model in the main viewer first.\n(File â†’ Load Block Model)"
                )

        # Replace the method
        self.irr_panel._load_block_model_from_viewer = load_from_main_window

        layout.addWidget(self.irr_panel)

        # Set block model if available
        if self.current_model:
            df = self.current_model.to_dataframe()
            self.irr_panel.set_block_model(df)

        # Connect signals
        self.irr_panel.schedule_visualization_requested.connect(
            self._on_schedule_visualization_requested
        )
        self.irr_panel.pit_visualization_requested.connect(
            self._on_pit_visualization_requested
        )

        logger.info("Opened IRR Optimization panel in separate window")

        # Show as non-modal dialog
        self.irr_dialog.show()

    def _on_schedule_visualization_requested(self, payload):
        """
        Integrate IRR schedule with the 3D viewer.
        schedule_df: pandas DataFrame with block_id, period, phase, destination, etc.
        """
        if isinstance(payload, tuple) and len(payload) == 2:
            schedule_df, mode = payload
        else:
            schedule_df = payload
            mode = "Period"

        try:
            if self.controller and hasattr(self.controller, "show_schedule"):
                self.controller.show_schedule(schedule_df, mode)
            elif self.viewer_widget and getattr(self.viewer_widget, "renderer", None):
                renderer = self.viewer_widget.renderer
                if hasattr(renderer, "show_schedule"):
                    renderer.show_schedule(schedule_df, mode)
                else:
                    raise AttributeError
            else:
                raise AttributeError
        except AttributeError:
            logger.error("Renderer does not implement show_schedule().")
            QMessageBox.warning(
                self,
                "3D Schedule",
                "The current viewer does not support schedule visualization.",
            )

    def _on_pit_visualization_requested(self, pit_df):
        """
        Show the IRR-optimal pit shell in the 3D viewer.
        pit_df: pandas DataFrame with block_id and in_optimal_pit flag (and phase if present).
        """
        try:
            if self.controller and hasattr(self.controller, "show_optimal_pit"):
                self.controller.show_optimal_pit(pit_df)
            elif self.viewer_widget and getattr(self.viewer_widget, "renderer", None):
                renderer = self.viewer_widget.renderer
                if hasattr(renderer, "show_optimal_pit"):
                    renderer.show_optimal_pit(pit_df)
                else:
                    raise AttributeError
            else:
                raise AttributeError
        except AttributeError:
            logger.error("Renderer does not implement show_optimal_pit().")
            QMessageBox.warning(
                self,
                "3D Pit",
                "The current viewer does not support optimal pit visualization.",
            )

    def open_3d_variogram(self):
        """Open 3D Variogram analysis."""
        drillhole_df = self._resolve_drillhole_dataframe()
        if drillhole_df is None:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes → Drillhole Loading\n\n"
                "You need to:\n"
                "1. Load collar, assay, and survey files\n"
                "2. Generate composites\n"
                "3. Visualize drillholes in 3D\n\n"
                "3D Kriging requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        QMessageBox.information(
            self,
            "3D Variogram Analysis Available!",
            "ðŸŽ¯ The 3D Variogram Analysis is already implemented!\n\n"
            "Please use the CORRECT menu:\n\n"
            "ðŸ“ Data & Analysis â†’ Variogram Tools â†’ Compute 3D Variogram\n\n"
            "Features:\n"
            "â€¢ Omnidirectional and directional variograms (N-S, E-W, Vertical)\n"
            "â€¢ Theoretical model fitting (Spherical, Exponential, Gaussian)\n"
            "â€¢ 3D Variogram cloud visualization\n"
            "â€¢ Export variogram tables to CSV\n\n"
            "Note: This menu item (Estimations) is for future advanced kriging features."
        )
        logger.info("3D Variogram menu clicked - redirected user to Data & Analysis menu")

    def open_ordinary_kriging(self):
        """Open Ordinary Kriging estimation."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes â†’ Drillhole Loading\n\n"
                "You need to:\n"
                "1. Load collar, assay, and survey files\n"
                "2. Visualize drillholes in 3D\n\n"
                "Ordinary Kriging uses drillhole assay data to estimate\n"
                "block model grades with optimal linear unbiased estimation."
            )
            return

        QMessageBox.information(
            self,
            "Ordinary Kriging",
            "Ordinary Kriging estimation tool coming soon!\n\n"
            "This feature will allow you to:\n"
            "â€¢ Perform OK grade estimation from drillhole data\n"
            "â€¢ Calculate kriging variance (uncertainty)\n"
            "â€¢ Define search ellipsoids with anisotropy\n"
            "â€¢ Set octant search strategies\n"
            "â€¢ Cross-validate estimates\n"
            "â€¢ Generate estimated block model\n"
            "â€¢ Export confidence intervals"
        )
        logger.info("Ordinary Kriging menu clicked - Coming Soon")

    def open_simple_kriging(self):
        """Open Simple Kriging estimation panel."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes → Drillhole Loading\n\n"
                "You need to:\n"
                "1. Load collar, assay, and survey files\n"
                "2. Generate composites\n"
                "3. Visualize drillholes in 3D\n\n"
                "Simple Kriging uses drillhole assay data to estimate\n"
                "block model grades assuming a known global mean."
            )
            return

        # Resolve drillhole data to DataFrame
        drillhole_df = self._resolve_drillhole_dataframe()
        if drillhole_df is None or drillhole_df.empty:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Could not resolve drillhole data.\n"
                "Please ensure composites or assays are loaded."
            )
            return

        # Check if panel already exists and is visible
        if not hasattr(self, 'simple_kriging_dialog') or self.simple_kriging_dialog is None or not self.simple_kriging_dialog.isVisible():
            from block_model_viewer.ui.simple_kriging_panel import SimpleKrigingPanel

            self.status_bar.showMessage("Opening Simple Kriging panel...", 2000)

            # Create as independent window (no parent) so it behaves like Drillhole Loading panel
            self.simple_kriging_dialog = SimpleKrigingPanel(None)
            self.simple_kriging_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.simple_kriging_dialog.bind_controller(self.controller)

            # Setup dialog persistence (position/size will be saved/restored)
            self._setup_dialog_persistence(self.simple_kriging_dialog, 'simple_kriging_dialog')

            # Auto-detect variable (first numeric column excluding X, Y, Z, system IDs, and compositing metadata)
            exclude_cols = ['X', 'Y', 'Z', 'FROM', 'TO', 'HOLEID', 'MID_DEPTH', 'DOMAIN', 'LENGTH', 'GLOBAL_INTERVAL_ID',
                           # Compositing metadata columns
                           'SAMPLE_COUNT', 'TOTAL_MASS', 'TOTAL_LENGTH', 'SUPPORT', 'IS_PARTIAL',
                           'METHOD', 'WEIGHTING', 'ELEMENT_WEIGHTS', 'MERGED_PARTIAL', 'MERGED_PARTIAL_AUTO']
            numeric_cols = drillhole_df.select_dtypes(include=[np.number]).columns
            grade_cols = [c for c in numeric_cols if c not in exclude_cols]
            variable = grade_cols[0] if grade_cols else None

            if variable:
                logger.info(f"Auto-detected variable '{variable}' for Simple Kriging")

            # Data is now fetched from registry - panel connects to registry.drillholeDataLoaded signal
            # No need to call set_drillhole_data() - panel will refresh automatically when registry updates
            self.simple_kriging_dialog.refresh()

            # Connect visualization signal (dict-based signal connects to visualize_kriging_results)
            if hasattr(self.simple_kriging_dialog, 'request_visualization'):
                try:
                    self.simple_kriging_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.simple_kriging_dialog.request_visualization.connect(self.visualize_kriging_results)
                logger.info("Connected Simple Kriging visualization signal")

            logger.info("Opened Simple Kriging panel")
            self.status_bar.showMessage(f"Simple Kriging panel ready - Variable: {variable} | Configure parameters and run", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.simple_kriging_dialog, 'request_visualization'):
                try:
                    self.simple_kriging_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.simple_kriging_dialog.request_visualization.connect(self.visualize_kriging_results)
            logger.info("Simple Kriging panel restored")
            self.status_bar.showMessage("Simple Kriging panel restored (parameters preserved)", 2000)

        # Show as non-modal dialog
        self.simple_kriging_dialog.show()
        self.simple_kriging_dialog.raise_()
        self.simple_kriging_dialog.activateWindow()

    def open_other_estimation_resources(self):
        """Open other geostatistical estimation resources."""
        QMessageBox.information(
            self,
            "Other Estimation Resources",
            "Additional geostatistical tools coming soon!\n\n"
            "All these methods use drillhole assay data to estimate block models:\n\n"
            "â€¢ Indicator Kriging (IK) - for non-linear grade estimation\n"
            "â€¢ Multiple Indicator Kriging (MIK) - categorical estimation\n"
            "â€¢ Co-Kriging - multi-variate estimation\n"
            "â€¢ Sequential Gaussian Simulation (SGS) - stochastic modeling\n"
            "â€¢ Inverse Distance Weighting (IDW) - deterministic interpolation\n"
            "â€¢ Nearest Neighbor (NN) - simple assignment\n"
            "â€¢ Grade domaining and contact analysis\n"
            "â€¢ Block support adjustments and change of support\n"
            "â€¢ Uniform Conditioning (UC) for recoverable resources\n"
            "â€¢ Cross-validation and quality assurance"
        )
        logger.info("Other Estimation Resources menu clicked - Coming Soon")

    # ============================================================================
    # VARIOGRAM ANALYSIS METHODS
    # ============================================================================

    def open_variogram_analysis(self):
        """Open 3D Variogram Analysis panel."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes â†’ Drillhole Loading\n\n"
                "You need to:\n"
                "1. Load collar, assay, and survey files\n"
                "2. Visualize drillholes in 3D\n\n"
                "3D Variogram analysis requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Check if panel already exists and is visible
        if not hasattr(self, 'variogram_dialog') or self.variogram_dialog is None or not self.variogram_dialog.isVisible():
            from block_model_viewer.ui.variogram_panel import VariogramAnalysisPanel

            self.status_bar.showMessage("Opening 3D Variogram Analysis panel...", 2000)

            # Create as top-level window so it has its own taskbar entry and standard window controls
            self.variogram_dialog = VariogramAnalysisPanel(parent=None, main_window=self)
            self.variogram_panel = self.variogram_dialog
            # Bind controller so run_analysis() works
            if self.controller:
                try:
                    self.variogram_dialog.bind_controller(self.controller)
                except Exception:
                    logger.debug("Failed to bind controller to variogram panel", exc_info=True)

            # Extract composites DataFrame from drillhole data dictionary.
            # Prefer renderer data (matches current 3D view), fall back to DataRegistry.
            drillhole_data_dict = getattr(self.viewer_widget.renderer, "drillhole_data", None)

            if drillhole_data_dict is None:
                try:

                    # Use injected registry via controller (dependency injection)
                    registry = self.controller.registry if self.controller else None
                    drillhole_data_dict = registry.get_drillhole_data() if registry else None
                except Exception:
                    drillhole_data_dict = None

            drillhole_df = None
            if isinstance(drillhole_data_dict, dict):
                # Newer loaders may use 'composites' or 'assays' keys
                if "composites_df" in drillhole_data_dict:
                    drillhole_df = drillhole_data_dict["composites_df"]
                elif "composites" in drillhole_data_dict:
                    drillhole_df = drillhole_data_dict["composites"]
                elif "assays" in drillhole_data_dict:
                    drillhole_df = drillhole_data_dict["assays"]
            else:
                drillhole_df = drillhole_data_dict  # Fallback if it's already a DataFrame

            # Data is now fetched from registry - panel connects to registry.drillholeDataLoaded signal
            # No need to call set_drillhole_data() - panel will refresh automatically when registry updates
            if drillhole_df is not None:
                pass  # Panel will fetch from registry

            logger.info("Opened 3D Variogram Analysis panel")
            self.status_bar.showMessage("Variogram panel ready - Results will be stored for kriging", 3000)
        else:
            self.status_bar.showMessage("Variogram panel restored (results preserved)", 2000)
            self.variogram_panel = self.variogram_dialog

        # Ensure controller is bound even when reusing an existing dialog
        if self.controller and hasattr(self, 'variogram_dialog'):
            try:
                self.variogram_dialog.bind_controller(self.controller)
            except Exception:
                logger.debug("Failed to (re)bind controller to variogram panel", exc_info=True)

        # Show as non-modal dialog
        self.variogram_dialog.show()
        # Open maximized so the plots use the full screen; users can restore if they prefer
        try:
            self.variogram_dialog.showMaximized()
        except Exception:
            # Fallback to default show if the platform rejects maximize
            pass
        self.variogram_dialog.raise_()
        self.variogram_dialog.activateWindow()

    def show_variogram_cloud(self):
        """Show variogram cloud directly."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes â†’ Drillhole Loading"
            )
            return

        # Open variogram panel and show cloud
        self.open_variogram_analysis()
        if hasattr(self, 'variogram_dialog') and self._is_dialog_valid(self.variogram_dialog) and self.variogram_dialog.isVisible():
            self.variogram_dialog.show_variogram_cloud()

    def fit_variogram_model(self):
        """Fit variogram model."""
        # Check if variogram panel is open and has results
        if not hasattr(self, 'variogram_dialog') or self.variogram_dialog is None or not self.variogram_dialog.isVisible():
            QMessageBox.information(
                self,
                "Variogram Analysis Required",
                "Please compute a variogram first.\n\n"
                "Go to: Data & Analysis â†’ Variogram Tools â†’ Compute 3D Variogram"
            )
            return

        if self.variogram_dialog.variogram_results is None:
            QMessageBox.information(
                self,
                "No Variogram Results",
                "Please compute a variogram first using the 'Compute Variogram' button."
            )
            return

        # Trigger recomputation with current parameters
        self.variogram_dialog.run_analysis()

    def export_variogram_tables(self):
        """Export variogram tables."""
        # Check if variogram panel is open and has results
        if not hasattr(self, 'variogram_dialog') or self.variogram_dialog is None or not self.variogram_dialog.isVisible():
            QMessageBox.information(
                self,
                "Variogram Analysis Required",
                "Please compute a variogram first.\n\n"
                "Go to: Data & Analysis â†’ Variogram Tools â†’ Compute 3D Variogram"
            )
            return

        if self.variogram_dialog.variogram_results is None:
            QMessageBox.information(
                self,
                "No Variogram Results",
                "Please compute a variogram first."
            )
            return

        # Trigger export
        self.variogram_dialog.export_variogram_tables()

    def _resolve_drillhole_dataframe(self):
        """Return composites DataFrame if drillhole data exists anywhere."""
        def _extract_df(payload):
            if isinstance(payload, dict):
                for key in ("composites_df", "composites", "assays", "data"):
                    if payload.get(key) is not None:
                        return payload[key]
            return payload

        renderer = getattr(self.viewer_widget, "renderer", None)
        renderer_data = getattr(renderer, "drillhole_data", None)
        if renderer_data is not None:
            df = _extract_df(renderer_data)
            if df is not None:
                return df

        if self.controller and hasattr(self.controller, "_drillhole_data"):
            df = _extract_df(self.controller._drillhole_data)
            if df is not None:
                if renderer is not None:
                    renderer.drillhole_data = {"composites_df": df}
                return df

        try:
            # Use injected registry via controller (dependency injection)
            registry = self.controller.registry if self.controller else None
            if registry:
                drillhole_data = registry.get_drillhole_data()
                df = _extract_df(drillhole_data)
            else:
                df = None
            if df is not None:
                if renderer is not None:
                    renderer.drillhole_data = {"composites_df": df}
                if self.controller:
                    self.controller._drillhole_data = df
                return df
        except Exception:
            pass
        return None

    def _has_drillhole_data(self) -> bool:
        """
        Check if drillhole data is available from any source.
        
        Checks:
        1. Renderer drillhole_data (visualized drillholes)
        2. DataRegistry (uploaded drillhole data)
        3. Controller cache
        
        Returns:
            True if drillhole data is available from any source
        """
        # Check renderer first
        renderer = getattr(self.viewer_widget, "renderer", None)
        if renderer and getattr(renderer, "drillhole_data", None) is not None:
            return True

        # Check controller cache
        if self.controller and getattr(self.controller, "_drillhole_data", None) is not None:
            return True

        # Check DataRegistry (use injected registry)
        try:
            registry = self.controller.registry if self.controller else None
            if registry and registry.has_drillhole_data():
                return True
        except Exception:
            pass

        return False

    # ============================================================================
    # KRIGING & ESTIMATION METHODS
    # ============================================================================

    def open_kriging_panel(self):
        """Open 3D Ordinary Kriging panel."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes → Drillhole Loading\n\n"
                "You need to:\n"
                "1. Load collar, assay, and survey files\n"
                "2. Generate composites\n"
                "3. Visualize drillholes in 3D\n\n"
                "3D Kriging requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Check if panel already exists and is visible
        if not hasattr(self, 'kriging_dialog') or self.kriging_dialog is None or not self.kriging_dialog.isVisible():
            from block_model_viewer.ui.kriging_panel import KrigingPanel

            self.status_bar.showMessage("Opening 3D Ordinary Kriging panel...", 2000)

            # Create as independent window (no parent) so it behaves like Drillhole Loading panel
            self.kriging_dialog = KrigingPanel(None)
            self.kriging_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.kriging_dialog.bind_controller(self.controller)

            # Setup dialog persistence (position/size will be saved/restored)
            self._setup_dialog_persistence(self.kriging_dialog, 'kriging_dialog')

            # Get drillhole data from any available source
            drillhole_df = self._resolve_drillhole_dataframe()
            # Data is now fetched from registry - panel connects to registry.drillholeDataLoaded signal
            # No need to call set_drillhole_data() - panel will refresh automatically when registry updates
            if drillhole_df is not None:
                pass  # Panel will fetch from registry

            # Link variogram results if available
            if hasattr(self, 'variogram_dialog') and self._is_dialog_valid(self.variogram_dialog):
                variogram_results = self.variogram_dialog.get_variogram_results()
                if variogram_results:
                    self.kriging_dialog.set_variogram_results(variogram_results)
                    logger.info("Linked variogram results to kriging panel")

            # Connect visualization signal
            if hasattr(self.kriging_dialog, 'request_visualization'):
                try:
                    self.kriging_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.kriging_dialog.request_visualization.connect(self.visualize_kriging_results)
                logger.info("Connected kriging visualization signal")

            logger.info("Opened 3D Ordinary Kriging panel")
            self.status_bar.showMessage("Kriging panel ready - Configure parameters and run", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.kriging_dialog, 'request_visualization'):
                try:
                    self.kriging_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.kriging_dialog.request_visualization.connect(self.visualize_kriging_results)
            self.status_bar.showMessage("Kriging panel restored (parameters preserved)", 2000)

        # Show as non-modal dialog
        self.kriging_dialog.show()
        self.kriging_dialog.raise_()
        self.kriging_dialog.activateWindow()

    def open_universal_kriging_panel(self):
        """Open Universal Kriging panel (STEP 22)."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes → Drillhole Loading\n\n"
                "Universal Kriging requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Check if panel already exists and is visible
        if not hasattr(self, 'universal_kriging_dialog') or self.universal_kriging_dialog is None or not self.universal_kriging_dialog.isVisible():
            from block_model_viewer.ui.universal_kriging_panel import UniversalKrigingPanel

            self.status_bar.showMessage("Opening Universal Kriging panel...", 2000)

            # Create as independent window (no parent) so it behaves like Drillhole Loading panel
            self.universal_kriging_dialog = UniversalKrigingPanel(None)
            self.universal_kriging_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.universal_kriging_dialog.bind_controller(self.controller)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.universal_kriging_dialog, 'universal_kriging_dialog')

            # Get drillhole data from any available source
            drillhole_df = self._resolve_drillhole_dataframe()
            if drillhole_df is not None:
                # Store drillhole data in controller for panel access
                if self.controller:
                    self.controller._drillhole_data = drillhole_df

                # Update variable combo (panel uses var_combo, not variable_combo)
                numeric_cols = drillhole_df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col.upper() not in ['X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH']]
                if hasattr(self.universal_kriging_dialog, 'var_combo'):
                    self.universal_kriging_dialog.var_combo.clear()
                    self.universal_kriging_dialog.var_combo.addItems(numeric_cols)

            # Connect visualization signal
            if hasattr(self.universal_kriging_dialog, 'request_visualization'):
                try:
                    self.universal_kriging_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.universal_kriging_dialog.request_visualization.connect(self.visualize_kriging_results)
                logger.info("Connected Universal Kriging visualization signal")

            logger.info("Opened Universal Kriging panel")
            self.status_bar.showMessage("Universal Kriging panel ready", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.universal_kriging_dialog, 'request_visualization'):
                try:
                    self.universal_kriging_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.universal_kriging_dialog.request_visualization.connect(self.visualize_kriging_results)
            self.status_bar.showMessage("Universal Kriging panel restored", 2000)

        self.universal_kriging_dialog.show()
        self.universal_kriging_dialog.raise_()
        self.universal_kriging_dialog.activateWindow()

    def open_cokriging_panel(self):
        """Open Co-Kriging panel (STEP 22)."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes → Drillhole Loading\n\n"
                "Co-Kriging requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Check if panel already exists and is visible
        if not hasattr(self, 'cokriging_dialog') or self.cokriging_dialog is None or not self.cokriging_dialog.isVisible():
            from block_model_viewer.ui.cokriging_panel import CoKrigingPanel

            self.status_bar.showMessage("Opening Co-Kriging panel...", 2000)

            # Create as independent window (no parent) so it behaves like Drillhole Loading panel
            self.cokriging_dialog = CoKrigingPanel(None)
            self.cokriging_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.cokriging_dialog.bind_controller(self.controller)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.cokriging_dialog, 'cokriging_dialog')

            # Get drillhole data from any available source
            drillhole_df = self._resolve_drillhole_dataframe()
            if drillhole_df is not None:
                # Store drillhole data in controller
                if self.controller:
                    self.controller._drillhole_data = drillhole_df

                # Update variable combos (only if UI is ready)
                numeric_cols = drillhole_df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col.upper() not in ['X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH']]
                if hasattr(self.cokriging_dialog, 'primary_combo') and hasattr(self.cokriging_dialog, 'secondary_combo'):
                    self.cokriging_dialog.primary_combo.clear()
                    self.cokriging_dialog.primary_combo.addItems(numeric_cols)
                    self.cokriging_dialog.secondary_combo.clear()
                    self.cokriging_dialog.secondary_combo.addItems(numeric_cols)

            # Connect visualization signal
            if hasattr(self.cokriging_dialog, 'request_visualization'):
                try:
                    self.cokriging_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.cokriging_dialog.request_visualization.connect(self.visualize_kriging_results)
                logger.info("Connected Co-Kriging visualization signal")

            logger.info("Opened Co-Kriging panel")
            self.status_bar.showMessage("Co-Kriging panel ready", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.cokriging_dialog, 'request_visualization'):
                try:
                    self.cokriging_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.cokriging_dialog.request_visualization.connect(self.visualize_kriging_results)
            self.status_bar.showMessage("Co-Kriging panel restored", 2000)

        self.cokriging_dialog.show()
        self.cokriging_dialog.raise_()
        self.cokriging_dialog.activateWindow()

    def open_indicator_kriging_panel(self):
        """Open Indicator Kriging panel (STEP 22)."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes → Drillhole Loading\n\n"
                "Indicator Kriging requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Check if panel already exists and is visible
        if not hasattr(self, 'indicator_kriging_dialog') or self.indicator_kriging_dialog is None or not self.indicator_kriging_dialog.isVisible():
            from block_model_viewer.ui.indicator_kriging_panel import IndicatorKrigingPanel

            self.status_bar.showMessage("Opening Indicator Kriging panel...", 2000)

            # Create as independent window (no parent) so it behaves like Drillhole Loading panel
            self.indicator_kriging_dialog = IndicatorKrigingPanel(None)
            self.indicator_kriging_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.indicator_kriging_dialog.bind_controller(self.controller)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.indicator_kriging_dialog, 'indicator_kriging_dialog')

            # Get drillhole data from any available source
            drillhole_df = self._resolve_drillhole_dataframe()
            if drillhole_df is not None:
                # Store drillhole data in controller
                if self.controller:
                    self.controller._drillhole_data = drillhole_df

                # Update variable combo (only if UI is ready)
                numeric_cols = drillhole_df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col.upper() not in ['X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH']]
                if hasattr(self.indicator_kriging_dialog, 'variable_combo'):
                    self.indicator_kriging_dialog.variable_combo.clear()
                    self.indicator_kriging_dialog.variable_combo.addItems(numeric_cols)

            # Connect visualization signal
            if hasattr(self.indicator_kriging_dialog, 'request_visualization'):
                try:
                    self.indicator_kriging_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.indicator_kriging_dialog.request_visualization.connect(self.visualize_kriging_results)
                logger.info("Connected Indicator Kriging visualization signal")

            logger.info("Opened Indicator Kriging panel")
            self.status_bar.showMessage("Indicator Kriging panel ready", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.indicator_kriging_dialog, 'request_visualization'):
                try:
                    self.indicator_kriging_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.indicator_kriging_dialog.request_visualization.connect(self.visualize_kriging_results)
            self.status_bar.showMessage("Indicator Kriging panel restored", 2000)

        self.indicator_kriging_dialog.show()
        self.indicator_kriging_dialog.raise_()
        self.indicator_kriging_dialog.activateWindow()

    def open_variogram_assistant_panel(self):
        """Open Variogram Modelling Assistant panel (STEP 23) as standalone window."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes → Drillhole Loading\n\n"
                "Variogram Assistant requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Get drillhole data
        drillhole_df = self._resolve_drillhole_dataframe()

        # Always create new panel (DeleteOnClose destroys it when closed)
        # Check if panel exists AND is visible; if closed, create new one
        need_new_panel = (
            not hasattr(self, 'variogram_assistant_dialog') or
            self.variogram_assistant_dialog is None or
            not self.variogram_assistant_dialog.isVisible()
        )

        if need_new_panel:
            from block_model_viewer.ui.variogram_assistant_panel import VariogramAssistantPanel

            self.status_bar.showMessage("Opening Variogram Assistant panel...", 2000)

            # Create as STANDALONE WINDOW (parent=None) with proper window flags
            self.variogram_assistant_dialog = VariogramAssistantPanel(parent=None)

            # Set window flags for standalone window behavior
            self.variogram_assistant_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )

            # CRITICAL: Bind controller BEFORE setting data
            if self.controller:
                bind_success = self.variogram_assistant_dialog.bind_controller(self.controller)
                if not bind_success:
                    logger.warning("Failed to bind controller to Variogram Assistant")
                else:
                    logger.info("Successfully bound controller to Variogram Assistant")

            # Set drillhole data on the panel
            if drillhole_df is not None:
                # Store in controller for task execution
                if self.controller:
                    self.controller._drillhole_data = drillhole_df

                # Set data directly on panel (triggers _populate_variables)
                self.variogram_assistant_dialog.set_drillhole_data(drillhole_df)

            logger.info("Opened Variogram Assistant as standalone window")
            self.status_bar.showMessage("Variogram Assistant panel ready", 3000)
        else:
            # Panel exists and is visible - just raise it
            self.status_bar.showMessage("Variogram Assistant panel restored", 2000)

            # Re-bind controller if needed (in case it was lost)
            if self.controller and not getattr(self.variogram_assistant_dialog, '_controller_bound', False):
                self.variogram_assistant_dialog.bind_controller(self.controller)

        # Show and activate
        self.variogram_assistant_dialog.show()
        self.variogram_assistant_dialog.raise_()
        self.variogram_assistant_dialog.activateWindow()

    def open_soft_kriging_panel(self):
        """Open Soft / Bayesian Kriging panel (STEP 23)."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes → Drillhole Loading\n\n"
                "Soft Kriging requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Check if panel already exists and is valid
        if hasattr(self, 'soft_kriging_dialog') and self._is_dialog_valid(self.soft_kriging_dialog):
            if self.soft_kriging_dialog.isVisible():
                # Dialog already visible, just bring to front
                self.soft_kriging_dialog.raise_()
                self.soft_kriging_dialog.activateWindow()
                return
            # Dialog exists but not visible - reuse it
            self.soft_kriging_dialog.show()
            self.soft_kriging_dialog.raise_()
            self.soft_kriging_dialog.activateWindow()
            return
        else:
            # Dialog was destroyed - clear reference
            if hasattr(self, 'soft_kriging_dialog'):
                self.soft_kriging_dialog = None

        # Create new dialog
        from block_model_viewer.ui.soft_kriging_panel import SoftKrigingPanel

        self.status_bar.showMessage("Opening Soft Kriging panel...", 2000)

        # Create as independent window (no parent) so it behaves like Drillhole Loading panel
        self.soft_kriging_dialog = SoftKrigingPanel(None)
        self.soft_kriging_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.soft_kriging_dialog.bind_controller(self.controller)

        # Setup dialog persistence
        self._setup_dialog_persistence(self.soft_kriging_dialog, 'soft_kriging_dialog')

        # Get drillhole data from any available source
        drillhole_df = self._resolve_drillhole_dataframe()
        if drillhole_df is not None:
            # Store drillhole data in controller
            if self.controller:
                self.controller._drillhole_data = drillhole_df

            # Update variable combo (only if UI is ready)
            numeric_cols = drillhole_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col.upper() not in ['X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH']]
            if hasattr(self.soft_kriging_dialog, 'variable_combo') and hasattr(self.soft_kriging_dialog, 'secondary_combo'):
                self.soft_kriging_dialog.variable_combo.clear()
                self.soft_kriging_dialog.variable_combo.addItems(numeric_cols)
                self.soft_kriging_dialog.secondary_combo.clear()
                self.soft_kriging_dialog.secondary_combo.addItems(numeric_cols)

        # Update block model property combo if block model is available (only if UI is ready)
        if self.current_model and hasattr(self.current_model, 'properties'):
            if hasattr(self.soft_kriging_dialog, 'bm_property_combo'):
                self.soft_kriging_dialog.bm_property_combo.clear()
                self.soft_kriging_dialog.bm_property_combo.addItems(list(self.current_model.properties.keys()))

        logger.info("Opened Soft Kriging panel")
        self.status_bar.showMessage("Soft Kriging panel ready", 3000)

        self.soft_kriging_dialog.show()
        self.soft_kriging_dialog.raise_()
        self.soft_kriging_dialog.activateWindow()

    def open_rbf_panel(self):
        """Open RBF Interpolation panel."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes → Drillhole Loading\n\n"
                "RBF Interpolation requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Check if panel already exists and is valid
        if hasattr(self, 'rbf_dialog') and self._is_dialog_valid(self.rbf_dialog):
            if self.rbf_dialog.isVisible():
                # Dialog already visible, just bring to front
                self.rbf_dialog.raise_()
                self.rbf_dialog.activateWindow()
                return
            # Dialog exists but not visible - reuse it
            self.rbf_dialog.show()
            self.rbf_dialog.raise_()
            self.rbf_dialog.activateWindow()
            return
        else:
            # Dialog was destroyed - clear reference
            if hasattr(self, 'rbf_dialog'):
                self.rbf_dialog = None

        # Create new dialog
        from block_model_viewer.ui.rbf_panel import RBFPanel

        self.status_bar.showMessage("Opening RBF Interpolation panel...", 2000)

        # Create as independent window (no parent) so it behaves like other estimation panels
        self.rbf_dialog = RBFPanel(None)
        self.rbf_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.rbf_dialog.bind_controller(self.controller)

        # Setup dialog persistence
        self._setup_dialog_persistence(self.rbf_dialog, 'rbf_dialog')

        # AUDIT FIX: Connect visualization signal (was missing - causing RBF to not display results)
        if hasattr(self.rbf_dialog, 'request_visualization'):
            try:
                self.rbf_dialog.request_visualization.disconnect()
            except (TypeError, RuntimeError):
                pass
            self.rbf_dialog.request_visualization.connect(self.visualize_rbf_results)
            logger.info("Connected RBF visualization signal")

        # Get drillhole data from any available source
        drillhole_df = self._resolve_drillhole_dataframe()
        if drillhole_df is not None:
            # Store drillhole data in controller
            if self.controller:
                self.controller._drillhole_data = drillhole_df

            # Update variable combo (only if UI is ready)
            self.rbf_dialog._update_variable_combo()

        logger.info("Opened RBF Interpolation panel")
        self.status_bar.showMessage("RBF Interpolation panel ready", 3000)

        self.rbf_dialog.show()
        self.rbf_dialog.raise_()
        self.rbf_dialog.activateWindow()

    def open_ik_sgsim_panel(self):
        """Open IK-based SGSIM panel."""
        # Check for drillhole data
        if not self._has_drillhole_data():
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "IK-SGSIM requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Check if panel already exists and is visible
        ik_dialog = getattr(self, 'ik_sgsim_dialog', None)
        if ik_dialog is None or not ik_dialog.isVisible():
            from block_model_viewer.ui.ik_sgsim_panel import IKSGSIMPanel

            self.status_bar.showMessage("Opening IK-SGSIM panel...", 2000)

            # Create as independent window (no parent) so it behaves like Drillhole Loading panel
            self.ik_sgsim_dialog = IKSGSIMPanel(None)
            self.ik_sgsim_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.ik_sgsim_dialog.bind_controller(self.controller)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.ik_sgsim_dialog, 'ik_sgsim_dialog')

            # Connect visualization signal
            if hasattr(self.ik_sgsim_dialog, 'request_visualization'):
                try:
                    self.ik_sgsim_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.ik_sgsim_dialog.request_visualization.connect(self.visualize_sgsim_results)
                logger.info("Connected IK-SGSIM visualization signal")

            logger.info("Opened IK-SGSIM panel")
            self.status_bar.showMessage("IK-SGSIM panel ready", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.ik_sgsim_dialog, 'request_visualization'):
                try:
                    self.ik_sgsim_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.ik_sgsim_dialog.request_visualization.connect(self.visualize_sgsim_results)
            self.status_bar.showMessage("IK-SGSIM panel restored", 2000)

        self.ik_sgsim_dialog.show()
        self.ik_sgsim_dialog.raise_()
        self.ik_sgsim_dialog.activateWindow()

    def open_cosgsim_panel(self):
        """Open Co-Simulation (CoSGSIM) panel."""
        # Check for drillhole data
        if not self._has_drillhole_data():
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Co-Simulation requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Check if panel already exists and is visible
        cosim_dialog = getattr(self, 'cosgsim_dialog', None)
        if cosim_dialog is None or not cosim_dialog.isVisible():
            from block_model_viewer.ui.cosgsim_panel import CoSGSIMPanel

            self.status_bar.showMessage("Opening Co-Simulation panel...", 2000)

            # Create as independent window (no parent) so it behaves like Drillhole Loading panel
            self.cosgsim_dialog = CoSGSIMPanel(None)
            self.cosgsim_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.cosgsim_dialog.bind_controller(self.controller)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.cosgsim_dialog, 'cosgsim_dialog')

            # Connect visualization signal
            if hasattr(self.cosgsim_dialog, 'request_visualization'):
                try:
                    self.cosgsim_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.cosgsim_dialog.request_visualization.connect(self.visualize_sgsim_results)
                logger.info("Connected Co-SGSIM visualization signal")

            logger.info("Opened Co-Simulation panel")
            self.status_bar.showMessage("Co-Simulation panel ready", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.cosgsim_dialog, 'request_visualization'):
                try:
                    self.cosgsim_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.cosgsim_dialog.request_visualization.connect(self.visualize_sgsim_results)
            self.status_bar.showMessage("Co-Simulation panel restored", 2000)

        self.cosgsim_dialog.show()
        self.cosgsim_dialog.raise_()
        self.cosgsim_dialog.activateWindow()

    def open_sis_panel(self):
        """Open Sequential Indicator Simulation (SIS) panel."""
        if not self._has_drillhole_data():
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "SIS requires drillhole/composite data with indicator transforms."
            )
            return

        sis_dialog = getattr(self, 'sis_dialog', None)
        if sis_dialog is None or not sis_dialog.isVisible():
            from block_model_viewer.ui.sis_panel import SISPanel

            self.status_bar.showMessage("Opening SIS panel...", 2000)

            self.sis_dialog = SISPanel(self)
            self.sis_dialog.bind_controller(self.controller)
            self._setup_dialog_persistence(self.sis_dialog, 'sis_dialog')

            # Connect visualization signal
            if hasattr(self.sis_dialog, 'request_visualization'):
                try:
                    self.sis_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.sis_dialog.request_visualization.connect(self.visualize_sgsim_results)
                logger.info("Connected SIS visualization signal")

            logger.info("Opened Sequential Indicator Simulation panel")
            self.status_bar.showMessage("SIS panel ready", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.sis_dialog, 'request_visualization'):
                try:
                    self.sis_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.sis_dialog.request_visualization.connect(self.visualize_sgsim_results)
            self.status_bar.showMessage("SIS panel restored", 2000)

        self.sis_dialog.show()
        self.sis_dialog.raise_()
        self.sis_dialog.activateWindow()

    def open_turning_bands_panel(self):
        """Open Turning Bands Simulation panel."""
        if not self._has_drillhole_data():
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Turning Bands requires conditioning data (optional) and variogram model."
            )
            return

        tb_dialog = getattr(self, 'turning_bands_dialog', None)
        if tb_dialog is None or not tb_dialog.isVisible():
            from block_model_viewer.ui.turning_bands_panel import TurningBandsPanel

            self.status_bar.showMessage("Opening Turning Bands panel...", 2000)

            self.turning_bands_dialog = TurningBandsPanel(self)
            self.turning_bands_dialog.bind_controller(self.controller)
            self._setup_dialog_persistence(self.turning_bands_dialog, 'turning_bands_dialog')

            # Connect visualization signal
            if hasattr(self.turning_bands_dialog, 'request_visualization'):
                try:
                    self.turning_bands_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.turning_bands_dialog.request_visualization.connect(self.visualize_sgsim_results)
                logger.info("Connected Turning Bands visualization signal")

            logger.info("Opened Turning Bands Simulation panel")
            self.status_bar.showMessage("Turning Bands panel ready", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.turning_bands_dialog, 'request_visualization'):
                try:
                    self.turning_bands_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.turning_bands_dialog.request_visualization.connect(self.visualize_sgsim_results)
            self.status_bar.showMessage("Turning Bands panel restored", 2000)

        self.turning_bands_dialog.show()
        self.turning_bands_dialog.raise_()
        self.turning_bands_dialog.activateWindow()

    def open_dbs_panel(self):
        """Open Direct Block Simulation (DBS) panel."""
        if not self._has_drillhole_data():
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "DBS requires conditioning data and block-support variogram."
            )
            return

        dbs_dialog = getattr(self, 'dbs_dialog', None)
        if dbs_dialog is None or not dbs_dialog.isVisible():
            from block_model_viewer.ui.dbs_panel import DBSPanel

            self.status_bar.showMessage("Opening DBS panel...", 2000)

            self.dbs_dialog = DBSPanel(self)
            self.dbs_dialog.bind_controller(self.controller)
            self._setup_dialog_persistence(self.dbs_dialog, 'dbs_dialog')

            # Connect visualization signal
            if hasattr(self.dbs_dialog, 'request_visualization'):
                try:
                    self.dbs_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.dbs_dialog.request_visualization.connect(self.visualize_sgsim_results)
                logger.info("Connected DBS visualization signal")

            logger.info("Opened Direct Block Simulation panel")
            self.status_bar.showMessage("DBS panel ready", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.dbs_dialog, 'request_visualization'):
                try:
                    self.dbs_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.dbs_dialog.request_visualization.connect(self.visualize_sgsim_results)
            self.status_bar.showMessage("DBS panel restored", 2000)

        self.dbs_dialog.show()
        self.dbs_dialog.raise_()
        self.dbs_dialog.activateWindow()

    def open_mps_panel(self):
        """Open Multiple-Point Simulation (MPS) panel."""
        mps_dialog = getattr(self, 'mps_dialog', None)
        if mps_dialog is None or not mps_dialog.isVisible():
            from block_model_viewer.ui.mps_panel import MPSPanel

            self.status_bar.showMessage("Opening MPS panel...", 2000)

            self.mps_dialog = MPSPanel(self)
            self.mps_dialog.bind_controller(self.controller)
            self._setup_dialog_persistence(self.mps_dialog, 'mps_dialog')

            # Connect visualization signal
            if hasattr(self.mps_dialog, 'request_visualization'):
                try:
                    self.mps_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.mps_dialog.request_visualization.connect(self.visualize_sgsim_results)
                logger.info("Connected MPS visualization signal")

            logger.info("Opened Multiple-Point Simulation panel")
            self.status_bar.showMessage("MPS panel ready", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.mps_dialog, 'request_visualization'):
                try:
                    self.mps_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.mps_dialog.request_visualization.connect(self.visualize_sgsim_results)
            self.status_bar.showMessage("MPS panel restored", 2000)

        self.mps_dialog.show()
        self.mps_dialog.raise_()
        self.mps_dialog.activateWindow()

    def open_grf_panel(self):
        """Open Gaussian Random Fields (GRF) panel."""
        grf_dialog = getattr(self, 'grf_dialog', None)
        if grf_dialog is None or not grf_dialog.isVisible():
            from block_model_viewer.ui.grf_panel import GRFPanel

            self.status_bar.showMessage("Opening GRF panel...", 2000)

            self.grf_dialog = GRFPanel(self)
            self.grf_dialog.bind_controller(self.controller)
            self._setup_dialog_persistence(self.grf_dialog, 'grf_dialog')

            # Connect visualization signal
            if hasattr(self.grf_dialog, 'request_visualization'):
                try:
                    self.grf_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.grf_dialog.request_visualization.connect(self.visualize_sgsim_results)
                logger.info("Connected GRF visualization signal")

            logger.info("Opened Gaussian Random Fields panel")
            self.status_bar.showMessage("GRF panel ready", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.grf_dialog, 'request_visualization'):
                try:
                    self.grf_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.grf_dialog.request_visualization.connect(self.visualize_sgsim_results)
            self.status_bar.showMessage("GRF panel restored", 2000)

        self.grf_dialog.show()
        self.grf_dialog.raise_()
        self.grf_dialog.activateWindow()

    def open_uncertainty_propagation_panel(self):
        """Open Uncertainty Propagation panel (STEP 24)."""
        # Check if block model is loaded
        if not self._has_valid_block_model():
            QMessageBox.warning(
                self,
                "No Block Model",
                "Please load a block model first.\n\n"
                "Uncertainty Propagation requires a block model with grade realisations."
            )
            return

        # Check if panel already exists and is visible
        uncertainty_dialog = getattr(self, 'uncertainty_propagation_dialog', None)
        if uncertainty_dialog is None or not uncertainty_dialog.isVisible():
            from block_model_viewer.ui.uncertainty_propagation_panel import (
                UncertaintyPropagationPanel,
            )

            self.status_bar.showMessage("Opening Uncertainty Propagation panel...", 2000)

            # Create as independent window (no parent) so it behaves like Drillhole Loading panel
            self.uncertainty_propagation_dialog = UncertaintyPropagationPanel(None)
            self.uncertainty_propagation_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.uncertainty_propagation_dialog.bind_controller(self.controller)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.uncertainty_propagation_dialog, 'uncertainty_propagation_dialog')

            # Update property combo (only if UI is ready)
            if self.current_model and hasattr(self.current_model, 'properties'):
                if hasattr(self.uncertainty_propagation_dialog, 'property_combo'):
                    properties = list(self.current_model.properties.keys())
                    self.uncertainty_propagation_dialog.property_combo.clear()
                    self.uncertainty_propagation_dialog.property_combo.addItems(properties)

            logger.info("Opened Uncertainty Propagation panel")
            self.status_bar.showMessage("Uncertainty Propagation panel ready", 3000)
        else:
            self.status_bar.showMessage("Uncertainty Propagation panel restored", 2000)

        self.uncertainty_propagation_dialog.show()
        self.uncertainty_propagation_dialog.raise_()
        self.uncertainty_propagation_dialog.activateWindow()

    def open_drillhole_panel(self):
        """
        DEPRECATED: Drillhole Database panel removed.
        
        Use "Drillholes > Drillhole Loading" instead, which provides:
        - Drillhole data loading (collar, survey, assay)
        - Trajectory computation
        - Domain-aware compositing
        - Full integration with the application
        
        This method redirects to Domain Compositing panel.
        """
        logger.info("open_drillhole_panel() called - redirecting to Domain Compositing panel")
        self.status_bar.showMessage("Redirecting to Drillholes > Drillhole Loading...", 2000)
        # Redirect to the working panel
        self.open_domain_compositing_panel()

    def open_structural_panel(self):
        """Open Structural Analysis panel (STEP 26)."""
        try:
            from .structural_panel import StructuralPanel

            if not hasattr(self, 'structural_dialog') or self.structural_dialog is None:
                self.structural_dialog = StructuralPanel(self)
                self.structural_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.structural_dialog, 'structural_dialog')

            self.structural_dialog.show()
            self.structural_dialog.raise_()
            self.structural_dialog.activateWindow()
            self.status_bar.showMessage("Structural Analysis panel opened", 2000)
            logger.info("Opened Structural Analysis panel")
        except Exception as e:
            logger.error(f"Failed to open Structural Analysis panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Structural Analysis panel:\n{e}")

    def open_slope_stability_panel(self):
        """Open Slope Stability panel (STEP 27)."""
        try:
            from .slope_stability_panel import SlopeStabilityPanel

            if not hasattr(self, 'slope_stability_dialog') or self.slope_stability_dialog is None:
                self.slope_stability_dialog = SlopeStabilityPanel(self)
                self.slope_stability_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.slope_stability_dialog, 'slope_stability_dialog')

            self.slope_stability_dialog.show()
            self.slope_stability_dialog.raise_()
            self.slope_stability_dialog.activateWindow()
            self.status_bar.showMessage("Slope Stability panel opened", 2000)
            logger.info("Opened Slope Stability panel")
        except Exception as e:
            logger.error(f"Failed to open Slope Stability panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Slope Stability panel:\n{e}")

    def open_bench_design_panel(self):
        """Open Bench Design panel (STEP 27)."""
        try:
            from .bench_design_panel import BenchDesignPanel

            if not hasattr(self, 'bench_design_dialog') or self.bench_design_dialog is None:
                self.bench_design_dialog = BenchDesignPanel(self)
                self.bench_design_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.bench_design_dialog, 'bench_design_dialog')

            self.bench_design_dialog.show()
            self.bench_design_dialog.raise_()
            self.bench_design_dialog.activateWindow()
            self.status_bar.showMessage("Bench Design panel opened", 2000)
            logger.info("Opened Bench Design panel")
        except Exception as e:
            logger.error(f"Failed to open Bench Design panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Bench Design panel:\n{e}")

    def open_geotech_summary_panel(self):
        """Open Geotech Summary panel (STEP 27)."""
        try:
            from .geotech_summary_panel import GeotechSummaryPanel

            if not hasattr(self, 'geotech_summary_dialog') or self.geotech_summary_dialog is None:
                self.geotech_summary_dialog = GeotechSummaryPanel(self)
                self.geotech_summary_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.geotech_summary_dialog, 'geotech_summary_dialog')

            self.geotech_summary_dialog.show()
            self.geotech_summary_dialog.raise_()
            self.geotech_summary_dialog.activateWindow()
            self.status_bar.showMessage("Geotech Summary panel opened", 2000)
            logger.info("Opened Geotech Summary panel")
        except Exception as e:
            logger.error(f"Failed to open Geotech Summary panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Geotech Summary panel:\n{e}")

    def open_geomet_domain_panel(self):
        """Open Geomet Domain Mapping panel (STEP 28)."""
        try:
            from .geomet_domain_panel import GeometDomainPanel

            if not hasattr(self, 'geomet_domain_dialog') or self.geomet_domain_dialog is None:
                self.geomet_domain_dialog = GeometDomainPanel(self)
                self.geomet_domain_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.geomet_domain_dialog, 'geomet_domain_dialog')

            self.geomet_domain_dialog.show()
            self.geomet_domain_dialog.raise_()
            self.geomet_domain_dialog.activateWindow()
            self.status_bar.showMessage("Geomet Domain Mapping panel opened", 2000)
            logger.info("Opened Geomet Domain Mapping panel")
        except Exception as e:
            logger.error(f"Failed to open Geomet Domain Mapping panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Geomet Domain Mapping panel:\n{e}")

    def open_geomet_plant_panel(self):
        """Open Geomet Plant Configuration panel (STEP 28)."""
        try:
            from .geomet_plant_panel import GeometPlantPanel

            if not hasattr(self, 'geomet_plant_dialog') or self.geomet_plant_dialog is None:
                self.geomet_plant_dialog = GeometPlantPanel(self)
                self.geomet_plant_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.geomet_plant_dialog, 'geomet_plant_dialog')

            self.geomet_plant_dialog.show()
            self.geomet_plant_dialog.raise_()
            self.geomet_plant_dialog.activateWindow()
            self.status_bar.showMessage("Geomet Plant Configuration panel opened", 2000)
            logger.info("Opened Geomet Plant Configuration panel")
        except Exception as e:
            logger.error(f"Failed to open Geomet Plant Configuration panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Geomet Plant Configuration panel:\n{e}")

    def open_geomet_panel(self):
        """Open Geomet Block Model panel (STEP 28)."""
        try:
            from .geomet_panel import GeometPanel

            if not hasattr(self, 'geomet_dialog') or self.geomet_dialog is None:
                self.geomet_dialog = GeometPanel(self)
                self.geomet_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.geomet_dialog, 'geomet_dialog')

            self.geomet_dialog.show()
            self.geomet_dialog.raise_()
            self.geomet_dialog.activateWindow()
            self.status_bar.showMessage("Geomet Block Model panel opened", 2000)
            logger.info("Opened Geomet Block Model panel")
        except Exception as e:
            logger.error(f"Failed to open Geomet Block Model panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Geomet Block Model panel:\n{e}")

    def open_grade_control_panel(self):
        """Open Grade Control panel (STEP 29)."""
        try:
            from .grade_control_panel import GradeControlPanel

            if not hasattr(self, 'gc_dialog') or self.gc_dialog is None:
                self.gc_dialog = GradeControlPanel(self)
                self.gc_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.gc_dialog, 'gc_dialog')

            self.gc_dialog.show()
            self.gc_dialog.raise_()
            self.gc_dialog.activateWindow()
            self.status_bar.showMessage("Grade Control panel opened", 2000)
            logger.info("Opened Grade Control panel")
        except Exception as e:
            logger.error(f"Failed to open Grade Control panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Grade Control panel:\n{e}")

    def open_digline_panel(self):
        """Open Digline panel (STEP 29)."""
        try:
            from .digline_panel import DiglinePanel

            if not hasattr(self, 'digline_dialog') or self.digline_dialog is None:
                self.digline_dialog = DiglinePanel(self)
                self.digline_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.digline_dialog, 'digline_dialog')

            self.digline_dialog.show()
            self.digline_dialog.raise_()
            self.digline_dialog.activateWindow()
            self.status_bar.showMessage("Digline panel opened", 2000)
            logger.info("Opened Digline panel")
        except Exception as e:
            logger.error(f"Failed to open Digline panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Digline panel:\n{e}")

    def open_reconciliation_panel(self):
        """Open Reconciliation panel (STEP 29)."""
        try:
            from .reconciliation_panel import ReconciliationPanel

            if not hasattr(self, 'recon_dialog') or self.recon_dialog is None:
                self.recon_dialog = ReconciliationPanel(self)
                self.recon_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.recon_dialog, 'recon_dialog')

            self.recon_dialog.show()
            self.recon_dialog.raise_()
            self.recon_dialog.activateWindow()
            self.status_bar.showMessage("Reconciliation panel opened", 2000)
            logger.info("Opened Reconciliation panel")
        except Exception as e:
            logger.error(f"Failed to open Reconciliation panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Reconciliation panel:\n{e}")

    def open_strategic_schedule_panel(self):
        """Open Strategic Schedule panel (STEP 30)."""
        try:
            from .strategic_schedule_panel import StrategicSchedulePanel

            if not hasattr(self, 'strategic_schedule_dialog') or self.strategic_schedule_dialog is None:
                self.strategic_schedule_dialog = StrategicSchedulePanel(self)
                self.strategic_schedule_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.strategic_schedule_dialog, 'strategic_schedule_dialog')

            self.strategic_schedule_dialog.show()
            self.strategic_schedule_dialog.raise_()
            self.strategic_schedule_dialog.activateWindow()
            self.status_bar.showMessage("Strategic Schedule panel opened", 2000)
            logger.info("Opened Strategic Schedule panel")
        except Exception as e:
            logger.error(f"Failed to open Strategic Schedule panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Strategic Schedule panel:\n{e}")

    def open_tactical_schedule_panel(self):
        """Open Tactical Schedule panel (STEP 30)."""
        try:
            from .tactical_schedule_panel import TacticalSchedulePanel

            if not hasattr(self, 'tactical_schedule_dialog') or self.tactical_schedule_dialog is None:
                self.tactical_schedule_dialog = TacticalSchedulePanel(self)
                self.tactical_schedule_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.tactical_schedule_dialog, 'tactical_schedule_dialog')

            self.tactical_schedule_dialog.show()
            self.tactical_schedule_dialog.raise_()
            self.tactical_schedule_dialog.activateWindow()
            self.status_bar.showMessage("Tactical Schedule panel opened", 2000)
            logger.info("Opened Tactical Schedule panel")
        except Exception as e:
            logger.error(f"Failed to open Tactical Schedule panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Tactical Schedule panel:\n{e}")

    def open_short_term_schedule_panel(self):
        """Open Short-Term Schedule panel (STEP 30)."""
        try:
            from .short_term_schedule_panel import ShortTermSchedulePanel

            if not hasattr(self, 'short_term_schedule_dialog') or self.short_term_schedule_dialog is None:
                self.short_term_schedule_dialog = ShortTermSchedulePanel(self)
                self.short_term_schedule_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.short_term_schedule_dialog, 'short_term_schedule_dialog')

            self.short_term_schedule_dialog.show()
            self.short_term_schedule_dialog.raise_()
            self.short_term_schedule_dialog.activateWindow()
            self.status_bar.showMessage("Short-Term Schedule panel opened", 2000)
            logger.info("Opened Short-Term Schedule panel")
        except Exception as e:
            logger.error(f"Failed to open Short-Term Schedule panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Short-Term Schedule panel:\n{e}")

    def open_fleet_panel(self):
        """Open Fleet panel (STEP 30)."""
        try:
            from .fleet_panel import FleetPanel

            if not hasattr(self, 'fleet_dialog') or self.fleet_dialog is None:
                self.fleet_dialog = FleetPanel(self)
                self.fleet_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.fleet_dialog, 'fleet_dialog')

            self.fleet_dialog.show()
            self.fleet_dialog.raise_()
            self.fleet_dialog.activateWindow()
            self.status_bar.showMessage("Fleet panel opened", 2000)
            logger.info("Opened Fleet panel")
        except Exception as e:
            logger.error(f"Failed to open Fleet panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Fleet panel:\n{e}")

    def open_planning_dashboard_panel(self):
        """Open Planning Dashboard panel (STEP 31)."""
        try:
            from .planning_dashboard_panel import PlanningDashboardPanel

            if not hasattr(self, 'planning_dashboard_dialog') or self.planning_dashboard_dialog is None:
                self.planning_dashboard_dialog = PlanningDashboardPanel(self)
                self.planning_dashboard_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.planning_dashboard_dialog, 'planning_dashboard_dialog')

            self.planning_dashboard_dialog.show()
            self.planning_dashboard_dialog.raise_()
            self.planning_dashboard_dialog.activateWindow()
            self.status_bar.showMessage("Planning Dashboard panel opened", 2000)
            logger.info("Opened Planning Dashboard panel")

            # Refresh scenario list on open
            if hasattr(self.planning_dashboard_dialog, '_refresh_scenario_list'):
                self.planning_dashboard_dialog._refresh_scenario_list()
        except Exception as e:
            logger.error(f"Failed to open Planning Dashboard panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Planning Dashboard panel:\n{e}")

    def open_npvs_panel(self):
        """Open NPVS Optimisation panel (STEP 32)."""
        try:
            from .npvs_panel import NPVSPanel

            if not hasattr(self, 'npvs_dialog') or self.npvs_dialog is None:
                self.npvs_dialog = NPVSPanel(self)
                self.npvs_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.npvs_dialog, 'npvs_dialog')

            self.npvs_dialog.show()
            self.npvs_dialog.raise_()
            self.npvs_dialog.activateWindow()
            self.status_bar.showMessage("NPVS Optimisation panel opened", 2000)
            logger.info("Opened NPVS Optimisation panel")
        except Exception as e:
            logger.error(f"Failed to open NPVS panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open NPVS Optimisation panel:\n{e}")

    def open_pushback_designer_panel(self):
        """Open Pushback Visual Designer panel (STEP 33)."""
        try:
            from .pushback_designer_panel import PushbackDesignerPanel

            if not hasattr(self, 'pushback_designer_dialog') or self.pushback_designer_dialog is None:
                self.pushback_designer_dialog = PushbackDesignerPanel(self)
                self.pushback_designer_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.pushback_designer_dialog, 'pushback_designer_dialog')

            self.pushback_designer_dialog.show()
            self.pushback_designer_dialog.raise_()
            self.pushback_designer_dialog.activateWindow()
            self.status_bar.showMessage("Pushback Visual Designer panel opened", 2000)
            logger.info("Opened Pushback Visual Designer panel")
        except Exception as e:
            logger.error(f"Failed to open Pushback Designer panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Pushback Visual Designer panel:\n{e}")

    def open_production_dashboard_panel(self):
        """Open Production Dashboard panel (STEP 36)."""
        try:
            from .production_dashboard_panel import ProductionDashboardPanel

            if not hasattr(self, 'production_dashboard_dialog') or self.production_dashboard_dialog is None:
                self.production_dashboard_dialog = ProductionDashboardPanel(self)
                self.production_dashboard_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.production_dashboard_dialog, 'production_dashboard_dialog')

            self.production_dashboard_dialog.show()
            self.production_dashboard_dialog.raise_()
            self.production_dashboard_dialog.activateWindow()
            self.status_bar.showMessage("Production Dashboard panel opened", 2000)
            logger.info("Opened Production Dashboard panel")
        except Exception as e:
            logger.error(f"Failed to open Production Dashboard panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Production Dashboard panel:\n{e}")

    def open_ug_advanced_panel(self):
        """Open Advanced Underground panel (STEP 37)."""
        try:
            from .ug_advanced_panel import UGAdvancedPanel

            if not hasattr(self, 'ug_advanced_dialog') or self.ug_advanced_dialog is None:
                self.ug_advanced_dialog = UGAdvancedPanel(self)
                self.ug_advanced_dialog.bind_controller(self.controller)
                # Set block model if available
                if hasattr(self, 'current_model') and self.current_model is not None:
                    self.ug_advanced_dialog.set_block_model(self.current_model)
                self._setup_dialog_persistence(self.ug_advanced_dialog, 'ug_advanced_dialog')
            else:
                # If dialog already exists, ensure block model is set
                if hasattr(self, 'current_model') and self.current_model is not None:
                    self.ug_advanced_dialog.set_block_model(self.current_model)

            self.ug_advanced_dialog.show()
            self.ug_advanced_dialog.raise_()
            self.ug_advanced_dialog.activateWindow()
            self.status_bar.showMessage("Advanced Underground panel opened", 2000)
            logger.info("Opened Advanced Underground panel")
        except Exception as e:
            logger.error(f"Failed to open Advanced Underground panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Advanced Underground panel:\n{e}")

    def open_geomet_chain_panel(self):
        """Open Geomet Chain panel (STEP 38)."""
        try:
            from .geomet_chain_panel import GeometChainPanel

            if not hasattr(self, 'geomet_chain_dialog') or self.geomet_chain_dialog is None:
                self.geomet_chain_dialog = GeometChainPanel(self)
                self.geomet_chain_dialog.bind_controller(self.controller)
                self._setup_dialog_persistence(self.geomet_chain_dialog, 'geomet_chain_dialog')

            self.geomet_chain_dialog.show()
            self.geomet_chain_dialog.raise_()
            self.geomet_chain_dialog.activateWindow()
            self.status_bar.showMessage("Geomet Chain panel opened", 2000)
            logger.info("Opened Geomet Chain panel")
        except Exception as e:
            logger.error(f"Failed to open Geomet Chain panel: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Geomet Chain panel:\n{e}")

    def open_research_dashboard_panel(self):
        """Open Research Mode Dashboard panel (STEP 25)."""
        # Check if panel already exists and is visible
        if not getattr(self, 'research_dashboard_dialog', None) or not getattr(self.research_dashboard_dialog, 'isVisible', lambda: False)():
            from block_model_viewer.ui.research_dashboard_panel import ResearchDashboardPanel

            self.status_bar.showMessage("Opening Research Mode Dashboard...", 2000)

            self.research_dashboard_dialog = ResearchDashboardPanel(self)
            self.research_dashboard_dialog.bind_controller(self.controller)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.research_dashboard_dialog, 'research_dashboard_dialog')

            logger.info("Opened Research Mode Dashboard panel")
            self.status_bar.showMessage("Research Mode Dashboard ready", 3000)
        else:
            self.status_bar.showMessage("Research Mode Dashboard restored", 2000)

        self.research_dashboard_dialog.show()
        self.research_dashboard_dialog.raise_()
        self.research_dashboard_dialog.activateWindow()

    def open_geotech_panel(self):
        """Open Geotechnical Dashboard panel (STEP 19)."""
        # Check if block model is loaded
        if not self._has_valid_block_model():
            QMessageBox.warning(
                self,
                "No Block Model",
                "Please load a block model first.\n\n"
                "Go to: File → Open\n\n"
                "Geotechnical interpolation requires a block model."
            )
            return

        # Check if panel already exists and is visible
        if not hasattr(self, 'geotech_dialog') or self.geotech_dialog is None or not self.geotech_dialog.isVisible():
            from block_model_viewer.ui.geotech_panel import GeotechPanel

            self.status_bar.showMessage("Opening Geotechnical Dashboard...", 2000)

            self.geotech_dialog = GeotechPanel(self)
            self.geotech_dialog.bind_controller(self.controller)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.geotech_dialog, 'geotech_dialog')

            logger.info("Opened Geotechnical Dashboard panel")
            self.status_bar.showMessage("Geotechnical Dashboard ready", 3000)
        else:
            self.status_bar.showMessage("Geotechnical Dashboard restored", 2000)

        self.geotech_dialog.show()
        self.geotech_dialog.raise_()
        self.geotech_dialog.activateWindow()

    def open_sgsim_panel(self):
        """Open SGSIM Simulation & Uncertainty Analysis panel."""
        # Check if drillhole data is loaded from any source
        has_drillhole_data = self._has_drillhole_data()

        if not has_drillhole_data:
            QMessageBox.warning(
                self,
                "No Drillhole Data",
                "Please load drillhole data first.\n\n"
                "Go to: Drillholes → Drillhole Loading\n\n"
                "You need to:\n"
                "1. Load collar, assay, and survey files\n"
                "2. Generate composites\n"
                "3. Visualize drillholes in 3D\n\n"
                "SGSIM requires drillhole assay data with X, Y, Z coordinates."
            )
            return

        # Check if panel already exists and is visible
        if not hasattr(self, 'sgsim_dialog') or self.sgsim_dialog is None or not self.sgsim_dialog.isVisible():
            from block_model_viewer.ui.sgsim_panel import SGSIMPanel

            self.status_bar.showMessage("Opening SGSIM Simulation panel...", 2000)

            # Create as independent window (no parent) so it behaves like Drillhole Loading panel
            self.sgsim_dialog = SGSIMPanel(None)
            self.sgsim_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.sgsim_dialog.bind_controller(self.controller)
            # CRITICAL: Pass main_window reference so panel can access renderer for accurate grid detection
            self.sgsim_dialog.main_window = self

            # Setup dialog persistence (position/size will be saved/restored)
            self._setup_dialog_persistence(self.sgsim_dialog, 'sgsim_dialog')

            # Get drillhole data from any available source
            drillhole_df = self._resolve_drillhole_dataframe()

            if drillhole_df is not None:
                # FIX: Use variogram variable if available, otherwise auto-detect
                variable = None

                # First, try to get variable from variogram results
                if hasattr(self, 'variogram_dialog') and self._is_dialog_valid(self.variogram_dialog):
                    variogram_results = self.variogram_dialog.get_variogram_results()
                    if variogram_results:
                        variable = variogram_results.get('variable', None)
                        if variable:
                            logger.info(f"Using variable '{variable}' from variogram analysis")

                # If no variogram variable, auto-detect (first numeric column excluding X, Y, Z)
                if variable is None:
                    exclude_cols = ['X', 'Y', 'Z', 'FROM', 'TO', 'HOLEID', 'MID_DEPTH', 'DOMAIN', 'LENGTH', 'GLOBAL_INTERVAL_ID',
                                   # Compositing metadata columns
                                   'SAMPLE_COUNT', 'TOTAL_MASS', 'TOTAL_LENGTH', 'SUPPORT', 'IS_PARTIAL',
                                   'METHOD', 'WEIGHTING', 'ELEMENT_WEIGHTS', 'MERGED_PARTIAL', 'MERGED_PARTIAL_AUTO']
                    numeric_cols = drillhole_df.select_dtypes(include=[np.number]).columns
                    grade_cols = [c for c in numeric_cols if c not in exclude_cols]
                    variable = grade_cols[0] if grade_cols else None
                    if variable:
                        logger.info(f"Auto-detected variable '{variable}' from drillhole data")

                # Data is now fetched from registry - panel connects to registry.drillholeDataLoaded signal
                # No need to call set_drillhole_data() - panel will refresh automatically when registry updates
                pass  # Panel will fetch from registry

            # Link variogram results if available
            if hasattr(self, 'variogram_dialog') and self._is_dialog_valid(self.variogram_dialog):
                variogram_results = self.variogram_dialog.get_variogram_results()
                if variogram_results:
                    logger.info("Auto-linking variogram results to SGSIM panel")
                    # The user can still manually load them if needed

            # Connect visualization signal if it exists (disconnect first to avoid duplicates)
            if hasattr(self.sgsim_dialog, 'request_visualization'):
                try:
                    self.sgsim_dialog.request_visualization.disconnect()
                    logger.debug("Disconnected existing SGSIM visualization signal")
                except (TypeError, RuntimeError):
                    pass  # Signal not connected yet
                self.sgsim_dialog.request_visualization.connect(self.visualize_sgsim_results)
                # Verify connection was successful
                if self.sgsim_dialog.receivers(self.sgsim_dialog.request_visualization) > 0:
                    logger.info(f"Connected SGSIM visualization signal to main_window.visualize_sgsim_results (receivers: {self.sgsim_dialog.receivers(self.sgsim_dialog.request_visualization)})")
                else:
                    logger.debug("SGSIM visualization signal connected but no receivers yet")
            else:
                logger.debug("SGSIM panel does not have request_visualization signal - visualization may be handled differently")

            variable_str = variable if variable else "None"
            logger.info(f"Opened SGSIM Simulation panel (variable: {variable_str})")
            self.status_bar.showMessage(f"SGSIM panel ready - Variable: {variable_str} | Configure parameters and run", 3000)
        else:
            # Ensure signal is connected even if panel already exists
            if hasattr(self.sgsim_dialog, 'request_visualization'):
                try:
                    self.sgsim_dialog.request_visualization.disconnect()
                except (TypeError, RuntimeError):
                    pass
                self.sgsim_dialog.request_visualization.connect(self.visualize_sgsim_results)
                logger.info("Reconnected SGSIM visualization signal to main_window.visualize_sgsim_results")
            self.status_bar.showMessage("SGSIM panel restored (parameters preserved)", 2000)

        # Show as non-modal dialog
        self.sgsim_dialog.show()
        self.sgsim_dialog.raise_()
        self.sgsim_dialog.activateWindow()

    def visualize_sgsim_results(self, grid, property_name: str):
        """
        Visualize SGSIM results in the main 3D viewer using the unified layer system.
        
        Parameters
        ----------
        grid : pv.RectilinearGrid
            PyVista grid containing SGSIM results
        property_name : str
            Property name to visualize (e.g., 'mean_bt', 'std_bt', 'p10_bt', 'p50_bt', 'p90_bt')
        """
        # Check if grid is None
        if grid is None:
            logger.error(f"SGSIM visualization failed: grid is None for property '{property_name}'")
            QMessageBox.warning(
                self,
                "Visualization Error",
                f"Cannot visualize '{property_name}': grid data is None.\n\n"
                "This may occur if:\n"
                "- SGSIM results are not available\n"
                "- Summary statistics were not computed\n"
                "- Grid creation failed"
            )
            return

        try:
            logger.info(f"RECEIVED SGSIM visualization request: property='{property_name}', grid has {grid.n_cells} cells, grid type={type(grid).__name__}")

            # Runtime checks for debugging
            if grid.n_cells == 0:
                QMessageBox.warning(self, "Empty Grid", "SGSIM grid has 0 cells. Check nx,ny,nz and edges.")
                return

            # GPU-SAFE: Warn about extremely large grids that can cause driver timeout
            VERY_LARGE_THRESHOLD = 200000  # 200k cells - warn user
            EXTREME_THRESHOLD = 500000     # 500k cells - strong warning

            if grid.n_cells > EXTREME_THRESHOLD:
                reply = QMessageBox.critical(
                    self,
                    "Extremely Large Grid - GPU Timeout Risk",
                    f"⚠️ This SGSIM grid has {grid.n_cells:,} cells!\n\n"
                    f"Visualizing this will likely cause GPU driver timeout (TDR) errors.\n\n"
                    f"Recommendation:\n"
                    f"• Create SGSIM on a coarser grid (increase cell size)\n"
                    f"• Target <100,000 cells for stable visualization\n"
                    f"• Use drillhole composite spacing as minimum cell size\n\n"
                    f"Proceed anyway? (NOT RECOMMENDED)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
            elif grid.n_cells > VERY_LARGE_THRESHOLD:
                QMessageBox.warning(
                    self,
                    "Large Grid Warning",
                    f"This SGSIM grid has {grid.n_cells:,} cells.\n\n"
                    f"Large grids may cause:\n"
                    f"• Slow rendering and interaction\n"
                    f"• GPU driver timeout during resize/rotation\n\n"
                    f"For best performance, use <100,000 cells.\n\n"
                    f"Proceeding with visualization..."
                )

            # Check both cell_data and point_data (StructuredGrid uses point_data when data size matches n_points)
            # Find property in grid (with case-insensitive matching)
            property_name_actual = None
            data_source = None  # Will be 'cell_data' or 'point_data'

            # Collect available properties from both sources
            cell_props = list(grid.cell_data.keys()) if hasattr(grid, 'cell_data') else []
            point_props = list(grid.point_data.keys()) if hasattr(grid, 'point_data') else []
            available_props = cell_props + point_props

            logger.info(f"Looking for property '{property_name}' in grid. Cell props: {cell_props}, Point props: {point_props}")

            # Check cell_data first (preferred for block models)
            if property_name in grid.cell_data:
                property_name_actual = property_name
                data_source = 'cell_data'
            # Then check point_data (StructuredGrid stores here when size matches n_points)
            elif property_name in grid.point_data:
                property_name_actual = property_name
                data_source = 'point_data'
            else:
                # Try case-insensitive match in cell_data
                property_upper = property_name.upper().replace(" ", "_")
                for prop in cell_props:
                    prop_normalized = prop.upper().replace(" ", "_")
                    if prop_normalized == property_upper:
                        property_name_actual = prop
                        data_source = 'cell_data'
                        break

                # Try case-insensitive match in point_data
                if property_name_actual is None:
                    for prop in point_props:
                        prop_normalized = prop.upper().replace(" ", "_")
                        if prop_normalized == property_upper:
                            property_name_actual = prop
                            data_source = 'point_data'
                            break

                # Fall back to first available property
                if property_name_actual is None:
                    if cell_props:
                        property_name_actual = cell_props[0]
                        data_source = 'cell_data'
                        logger.warning(f"Property '{property_name}' not found, using cell_data '{property_name_actual}'")
                    elif point_props:
                        property_name_actual = point_props[0]
                        data_source = 'point_data'
                        logger.warning(f"Property '{property_name}' not found, using point_data '{property_name_actual}'")

            if property_name_actual is None:
                QMessageBox.warning(
                    self,
                    "Property Not Found",
                    f"Property '{property_name}' not found in SGSIM grid.\n\n"
                    f"Available cell_data: {', '.join(cell_props)}\n"
                    f"Available point_data: {', '.join(point_props)}"
                )
                return

            # Get data from the correct source
            if data_source == 'cell_data':
                data = grid.cell_data[property_name_actual]
            else:
                data = grid.point_data[property_name_actual]

            logger.info(f"Using property '{property_name_actual}' from {data_source}")
            valid_mask = np.isfinite(data)

            if not valid_mask.any():
                QMessageBox.warning(self, "No Valid Data", f"All values in '{property_name_actual}' are NaN or infinite.")
                return

            # Create layer name
            layer_name = f"SGSIM: {property_name_actual}"

            # Remove existing SGSIM layer with same name
            if self.viewer_widget and self.viewer_widget.renderer:
                if layer_name in self.viewer_widget.renderer.active_layers:
                    self.viewer_widget.renderer.clear_layer(layer_name)

            # ✅ FIX: Use add_block_model_layer directly (same as block models)
            # This is the working method that block models use
            self.viewer_widget.renderer.add_block_model_layer(grid, property_name_actual, layer_name=layer_name)

            # Refresh Interactive Slicer panel if open
            self._refresh_slicer_panel()

            # Update legend via LegendManager
            if hasattr(self.viewer_widget.renderer, 'legend_manager') and self.viewer_widget.renderer.legend_manager is not None:
                try:
                    valid_data = data[valid_mask]
                    data_min = float(np.nanmin(valid_data))
                    data_max = float(np.nanmax(valid_data))
                    logger.info(f"SGSIM VIS: Updating legend for property '{property_name_actual}', range=[{data_min:.3f}, {data_max:.3f}]")
                    self.viewer_widget.renderer.legend_manager.set_continuous(
                        field=property_name_actual.upper(),
                        vmin=data_min,
                        vmax=data_max
                    )
                    logger.info("SGSIM VIS: Legend updated successfully")
                except Exception as e:
                    logger.warning(f"SGSIM VIS: Could not update legend: {e}", exc_info=True)

            # Reset camera to show all (SAME AS BUILD BLOCK MODEL)
            # IMPORTANT: Use same threshold as renderer.py GPU-safe mode (30,000)
            # Always reset camera for SGSIM so the user can see the new grid.
            # For very large models we use a lightweight reset_camera() which
            # only repositions the camera without re-rendering all actors.
            if self.viewer_widget and self.viewer_widget.renderer and self.viewer_widget.renderer.plotter:
                self.viewer_widget.renderer.plotter.reset_camera()
                logger.info(f"SGSIM VIS: Camera reset ({grid.n_cells:,} cells)")

            # Update property panel to show the new layer
            if hasattr(self, 'property_panel') and self.property_panel:
                try:
                    self.property_panel.update_layer_controls()
                    logger.info(f"Updated property panel with new SGSIM layer: {layer_name}")
                except AttributeError:
                    # Try fallback method name if it exists
                    if hasattr(self.property_panel, 'update_active_layers'):
                        self.property_panel.update_active_layers()
                        logger.info("Used fallback update_active_layers method")

            # Log active layers after registration
            logger.info(f"Active layers now: {list(self.viewer_widget.renderer.active_layers.keys())}")

            # Get data range for status message
            valid_data = data[valid_mask]
            data_min = float(np.nanmin(valid_data))
            data_max = float(np.nanmax(valid_data))

            self.status_bar.showMessage(
                f"SGSIM results added: {property_name_actual} ({grid.n_cells} blocks, "
                f"range=[{data_min:.2f}, {data_max:.2f}]) | Use Layer Controls to toggle visibility",
                5000
            )
            logger.info(f"Visualized SGSIM results: {property_name_actual}, bounds={grid.bounds}")

        except Exception as e:
            logger.error(f"Error visualizing SGSIM results: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Error visualizing SGSIM results:\n{str(e)}\n\n"
                "Check console logs for details."
            )

    def visualize_simple_kriging_results(self, grid, property_name: str):
        """
        Visualize Simple Kriging results in the main 3D viewer using the unified layer system.
        
        Parameters
        ----------
        grid : pv.StructuredGrid
            PyVista grid containing SK results
        property_name : str
            Property name to visualize (e.g., 'SK_est', 'SK_var')
        """
        try:

            logger.info(f"SIMPLE KRIGING VIS: Received visualization request: property='{property_name}', grid has {grid.n_cells} cells, grid type={type(grid).__name__}")

            # Runtime checks for debugging
            if grid is None:
                logger.error("SIMPLE KRIGING VIS: grid is None")
                QMessageBox.warning(
                    self,
                    "Visualization Error",
                    "Cannot visualize: grid data is None."
                )
                return

            if grid.n_cells == 0:
                logger.warning("SIMPLE KRIGING VIS: Grid has 0 cells")
                QMessageBox.warning(self, "Empty Grid", "Simple Kriging grid has 0 cells.")
                return

            # Find property in grid (with case-insensitive matching)
            property_name_actual = None
            available_props = list(grid.cell_data.keys()) if hasattr(grid, 'cell_data') else []

            logger.info(f"SIMPLE KRIGING VIS: Looking for property '{property_name}' in grid. Available properties: {available_props}")

            if property_name in grid.cell_data:
                property_name_actual = property_name
            else:
                # Try case-insensitive match
                property_upper = property_name.upper().replace(" ", "_")
                for prop in available_props:
                    prop_normalized = prop.upper().replace(" ", "_")
                    if prop_normalized == property_upper:
                        property_name_actual = prop
                        break

                if property_name_actual is None and available_props:
                    property_name_actual = available_props[0]
                    logger.warning(f"SIMPLE KRIGING VIS: Property '{property_name}' not found, using '{property_name_actual}'")

            if property_name_actual is None:
                QMessageBox.warning(
                    self,
                    "Property Not Found",
                    f"Property '{property_name}' not found in Simple Kriging grid.\n\n"
                    f"Available properties: {', '.join(available_props)}"
                )
                return

            # Get property data (prefer cell_data, fallback to point_data)
            if property_name_actual in grid.cell_data:
                data = grid.cell_data[property_name_actual]
                preference = 'cell'
            elif property_name_actual in grid.point_data:
                data = grid.point_data[property_name_actual]
                preference = 'point'
            else:
                logger.error(f"SIMPLE KRIGING VIS: Property '{property_name_actual}' not found in cell_data or point_data")
                QMessageBox.warning(
                    self,
                    "Property Not Found",
                    f"Property '{property_name_actual}' not found in grid."
                )
                return

            # Check for valid data
            valid_mask = np.isfinite(data)
            if not valid_mask.any():
                logger.warning(f"SIMPLE KRIGING VIS: All values in '{property_name_actual}' are NaN or infinite")
                QMessageBox.warning(self, "No Valid Data", f"All values in '{property_name_actual}' are NaN or infinite.")
                return

            valid_data = data[valid_mask]
            data_min = float(np.nanmin(valid_data))
            data_max = float(np.nanmax(valid_data))

            logger.info(f"SIMPLE KRIGING VIS: Property '{property_name_actual}' data range: [{data_min:.3f}, {data_max:.3f}], valid points: {valid_mask.sum()}/{len(data)}")

            # Create layer name
            layer_name = f"Simple Kriging: {property_name_actual}"

            # Remove existing layer with same name
            if self.viewer_widget and self.viewer_widget.renderer:
                if layer_name in self.viewer_widget.renderer.active_layers:
                    logger.info(f"SIMPLE KRIGING VIS: Removing existing layer '{layer_name}'")
                    self.viewer_widget.renderer.clear_layer(layer_name)

            # ✅ FIX: Use add_block_model_layer directly (same as SGSIM and Ordinary Kriging)
            # This is the working method that other visualizations use
            logger.info(f"SIMPLE KRIGING VIS: Adding layer '{layer_name}' with property '{property_name_actual}'")
            self.viewer_widget.renderer.add_block_model_layer(grid, property_name_actual, layer_name=layer_name)

            # Refresh Interactive Slicer panel if open
            self._refresh_slicer_panel()

            # Update legend via LegendManager
            if hasattr(self.viewer_widget.renderer, 'legend_manager') and self.viewer_widget.renderer.legend_manager is not None:
                try:
                    logger.info(f"SIMPLE KRIGING VIS: Updating legend for property '{property_name_actual}', range=[{data_min:.3f}, {data_max:.3f}]")
                    self.viewer_widget.renderer.legend_manager.set_continuous(
                        field=property_name_actual.upper(),
                        vmin=data_min,
                        vmax=data_max,
                        cmap_name='viridis'
                    )
                    logger.info("SIMPLE KRIGING VIS: Legend updated successfully")
                except Exception as e:
                    logger.warning(f"SIMPLE KRIGING VIS: Could not update legend: {e}", exc_info=True)

            # Reset camera to show all
            if self.viewer_widget and self.viewer_widget.renderer and self.viewer_widget.renderer.plotter:
                self.viewer_widget.renderer.plotter.reset_camera()
                logger.info("SIMPLE KRIGING VIS: Camera reset")

            # Update property panel to show the new layer
            if hasattr(self, 'property_panel') and self.property_panel:
                try:
                    self.property_panel.update_layer_controls()
                    logger.info(f"SIMPLE KRIGING VIS: Updated property panel with new layer: {layer_name}")
                except AttributeError:
                    # Try fallback method name if it exists
                    if hasattr(self.property_panel, 'update_active_layers'):
                        self.property_panel.update_active_layers()
                        logger.info("SIMPLE KRIGING VIS: Used fallback update_active_layers method")

            # Log active layers after registration
            logger.info(f"SIMPLE KRIGING VIS: Active layers now: {list(self.viewer_widget.renderer.active_layers.keys())}")

            self.status_bar.showMessage(
                f"Simple Kriging results added: {property_name_actual} ({grid.n_cells} blocks, "
                f"range=[{data_min:.2f}, {data_max:.2f}]) | Use Layer Controls to toggle visibility",
                5000
            )
            logger.info(f"SIMPLE KRIGING VIS: Successfully visualized Simple Kriging results: {property_name_actual}, bounds={grid.bounds}")

        except Exception as e:
            logger.error(f"SIMPLE KRIGING VIS: Error visualizing Simple Kriging results: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Error visualizing Simple Kriging results:\n{str(e)}\n\nCheck console logs for details."
            )

    def visualize_kriging_results(self, kriging_results: Dict):
        """
        Visualize kriging results in the main 3D viewer using the unified layer system.
        
        Parameters
        ----------
        kriging_results : dict
            Dictionary containing grid_x, grid_y, grid_z, estimates, variances, variable
        """
        # Disabled: rendering kriging grids directly is out of workflow (build block model instead)
        logger.info(
            "KRIGING VIS: Visualization skipped - please build a block model to view kriging results."
        )
        QMessageBox.information(
            self,
            "Visualization Disabled",
            "Direct 3D visualization of kriging grids is disabled.\n\n"
            "Next step: Open the Block Model Builder and load your kriging results to create a block model for viewing."
        )
        return
        try:
            import pyvista as pv

            logger.info(f"KRIGING VIS: Received kriging_results with keys: {list(kriging_results.keys())}")

            grid_x = kriging_results.get('grid_x')
            grid_y = kriging_results.get('grid_y')
            grid_z = kriging_results.get('grid_z')
            estimates = kriging_results.get('estimates')
            variances = kriging_results.get('variances')
            variable = kriging_results.get('variable', 'Unknown')

            # Validate inputs
            if grid_x is None or grid_y is None or grid_z is None:
                logger.error(f"KRIGING VIS: Missing grid coordinates. grid_x={grid_x is not None}, grid_y={grid_y is not None}, grid_z={grid_z is not None}")
                QMessageBox.warning(
                    self,
                    "Visualization Error",
                    "Missing grid coordinates in kriging results."
                )
                return

            if estimates is None:
                logger.error("KRIGING VIS: Missing estimates in kriging results")
                QMessageBox.warning(
                    self,
                    "Visualization Error",
                    "Missing estimates in kriging results."
                )
                return

            # Convert to numpy arrays if needed
            grid_x = np.asarray(grid_x)
            grid_y = np.asarray(grid_y)
            grid_z = np.asarray(grid_z)
            estimates = np.asarray(estimates)

            logger.info(f"KRIGING VIS: grid_x shape={grid_x.shape}, grid_y shape={grid_y.shape}, grid_z shape={grid_z.shape}, estimates shape={estimates.shape}")

            # Check for valid data
            valid_mask = np.isfinite(estimates)
            if not valid_mask.any():
                logger.warning("KRIGING VIS: All estimates are NaN or infinite")
                QMessageBox.warning(
                    self,
                    "No Valid Data",
                    "All kriging estimates are NaN or infinite. Cannot visualize."
                )
                return

            # Handle different grid coordinate formats
            # Case 1: 3D meshgrid arrays (from _prepare_kriging_payload) - use directly
            if grid_x.ndim == 3 and grid_y.ndim == 3 and grid_z.ndim == 3:
                logger.info("KRIGING VIS: Using 3D meshgrid arrays directly")
                # Ensure they have the same shape
                if grid_x.shape != grid_y.shape or grid_x.shape != grid_z.shape:
                    logger.error(f"KRIGING VIS: Grid coordinate shapes mismatch: {grid_x.shape}, {grid_y.shape}, {grid_z.shape}")
                    QMessageBox.warning(
                        self,
                        "Visualization Error",
                        f"Grid coordinate shapes mismatch: {grid_x.shape}, {grid_y.shape}, {grid_z.shape}"
                    )
                    return

                # Reshape estimates to match grid if needed
                if estimates.ndim == 1:
                    # Reshape estimates to match grid dimensions
                    nx, ny, nz = grid_x.shape
                    if len(estimates) == nx * ny * nz:
                        estimates = estimates.reshape(nx, ny, nz, order='C')
                    else:
                        logger.error(f"KRIGING VIS: Estimates length {len(estimates)} doesn't match grid size {nx * ny * nz}")
                        QMessageBox.warning(
                            self,
                            "Visualization Error",
                            f"Estimates length ({len(estimates)}) doesn't match grid size ({nx * ny * nz})"
                        )
                        return
                elif estimates.ndim == 3:
                    # Ensure shapes match
                    if estimates.shape != grid_x.shape:
                        logger.warning(f"KRIGING VIS: Estimates shape {estimates.shape} doesn't match grid shape {grid_x.shape}, attempting reshape")
                        # Try to reshape
                        if estimates.size == grid_x.size:
                            estimates = estimates.reshape(grid_x.shape, order='C')
                        else:
                            logger.error(f"KRIGING VIS: Cannot reshape estimates {estimates.shape} to match grid {grid_x.shape}")
                            QMessageBox.warning(
                                self,
                                "Visualization Error",
                                "Cannot reshape estimates to match grid dimensions."
                            )
                            return

                # Create PyVista StructuredGrid directly (same as _prepare_kriging_payload)
                grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
                property_name = f'{variable}_OK_est'

                # Add estimates as cell data (Fortran order for structured grids)
                estimates_flat = estimates.ravel(order='F')
                grid[property_name] = estimates_flat

                # Add variances if available
                if variances is not None:
                    variances = np.asarray(variances)
                    if variances.size == estimates.size:
                        variance_property = f'{variable}_OK_var'
                        if variances.ndim == 3:
                            variances_flat = variances.ravel(order='F')
                        else:
                            variances_flat = variances.ravel(order='F')
                        grid[variance_property] = variances_flat
                        logger.info(f"KRIGING VIS: Added variance property '{variance_property}'")

            # Case 2: 1D coordinate arrays - need to reconstruct grid
            elif grid_x.ndim == 1:
                logger.info("KRIGING VIS: Reconstructing grid from 1D coordinate arrays")
                # Check if we have enough information to reconstruct
                if len(grid_x) != len(grid_y) or len(grid_x) != len(grid_z):
                    logger.error(f"KRIGING VIS: Coordinate array lengths mismatch: {len(grid_x)}, {len(grid_y)}, {len(grid_z)}")
                    QMessageBox.warning(
                        self,
                        "Visualization Error",
                        "Grid coordinate arrays have different lengths."
                    )
                    return

                # Try to infer grid dimensions from unique values
                unique_x = np.unique(grid_x)
                unique_y = np.unique(grid_y)
                unique_z = np.unique(grid_z)
                nx, ny, nz = len(unique_x), len(unique_y), len(unique_z)

                logger.info(f"KRIGING VIS: Inferred grid dimensions: ({nx}, {ny}, {nz}), total points: {len(grid_x)}")

                if nx * ny * nz != len(grid_x):
                    logger.warning(f"KRIGING VIS: Grid size mismatch. Expected {nx * ny * nz}, got {len(grid_x)}")
                    # Try to create grid from point cloud instead
                    logger.info("KRIGING VIS: Creating point cloud mesh instead of structured grid")
                    points = np.column_stack([grid_x, grid_y, grid_z])
                    mesh = pv.PolyData(points)
                    mesh[property_name] = estimates if estimates.ndim == 1 else estimates.ravel()
                    grid = mesh
                else:
                    # Create meshgrid
                    X, Y, Z = np.meshgrid(unique_x, unique_y, unique_z, indexing='ij')
                    grid = pv.StructuredGrid(X, Y, Z)

                    # Reshape estimates to match grid
                    if estimates.ndim == 1 and len(estimates) == nx * ny * nz:
                        estimates_reshaped = estimates.reshape(nx, ny, nz, order='C')
                    else:
                        logger.error(f"KRIGING VIS: Cannot reshape estimates {estimates.shape} to grid ({nx}, {ny}, {nz})")
                        QMessageBox.warning(
                            self,
                            "Visualization Error",
                            "Cannot reshape estimates to match grid dimensions."
                        )
                        return

                    property_name = f'{variable}_OK_est'
                    grid[property_name] = estimates_reshaped.ravel(order='F')

                    # Add variances if available
                    if variances is not None:
                        variances = np.asarray(variances)
                        if variances.size == estimates.size:
                            variance_property = f'{variable}_OK_var'
                            if variances.ndim == 1:
                                variances_reshaped = variances.reshape(nx, ny, nz, order='C')
                            else:
                                variances_reshaped = variances
                            grid[variance_property] = variances_reshaped.ravel(order='F')
                            logger.info(f"KRIGING VIS: Added variance property '{variance_property}'")
            else:
                logger.error(f"KRIGING VIS: Unsupported grid coordinate format. grid_x.ndim={grid_x.ndim}, grid_y.ndim={grid_y.ndim}, grid_z.ndim={grid_z.ndim}")
                QMessageBox.warning(
                    self,
                    "Visualization Error",
                    f"Unsupported grid coordinate format (dimensions: {grid_x.ndim}, {grid_y.ndim}, {grid_z.ndim})"
                )
                return

            # Validate grid was created successfully
            if grid is None or grid.n_cells == 0:
                logger.error("KRIGING VIS: Failed to create grid or grid has 0 cells")
                QMessageBox.warning(
                    self,
                    "Visualization Error",
                    "Failed to create grid or grid has 0 cells."
                )
                return

            # Create layer name
            layer_name = f"Kriging: {variable}"

            # Remove existing kriging layer with same name
            if self.viewer_widget and self.viewer_widget.renderer:
                if layer_name in self.viewer_widget.renderer.active_layers:
                    logger.info(f"KRIGING VIS: Removing existing layer '{layer_name}'")
                    self.viewer_widget.renderer.clear_layer(layer_name)

            # ✅ CONVERT KRIGING RESULTS TO DRILLHOLE FORMAT
            # Flatten grid coordinates and estimates based on grid format
            if grid_x.ndim == 3:
                # 3D meshgrid: flatten all dimensions
                coords_flat = np.column_stack([
                    grid_x.ravel(),
                    grid_y.ravel(),
                    grid_z.ravel()
                ])
                estimates_flat = estimates.ravel()
            elif grid_x.ndim == 1:
                # 1D arrays: already flat
                coords_flat = np.column_stack([grid_x, grid_y, grid_z])
                estimates_flat = estimates.ravel() if estimates.ndim > 1 else estimates
            else:
                # Unsupported format
                logger.error(f"KRIGING VIS: Unsupported grid format for drillhole conversion. grid_x.ndim={grid_x.ndim}")
                QMessageBox.warning(self, "Visualization Error", "Unsupported grid format for drillhole visualization.")
                return

            # Filter to valid estimates only
            valid_mask = np.isfinite(estimates_flat)
            if not valid_mask.any():
                logger.warning("KRIGING VIS: No valid estimates after flattening")
                QMessageBox.warning(self, "No Valid Data", "No valid kriging estimates to visualize.")
                return

            coords_valid = coords_flat[valid_mask]
            estimates_valid = estimates_flat[valid_mask]

            # Calculate a small interval size based on grid spacing (use 10% of typical spacing)
            if grid_x.ndim == 3:
                # Estimate spacing from grid
                if grid_x.shape[0] > 1:
                    dx = abs(grid_x[1, 0, 0] - grid_x[0, 0, 0])
                else:
                    dx = 1.0
                if grid_y.shape[1] > 1:
                    dy = abs(grid_y[0, 1, 0] - grid_y[0, 0, 0])
                else:
                    dy = 1.0
                if grid_z.shape[2] > 1:
                    dz = abs(grid_z[0, 0, 1] - grid_z[0, 0, 0])
                else:
                    dz = 1.0
                interval_size = min(dx, dy, dz) * 0.1  # 10% of smallest spacing
            elif grid_x.ndim == 1:
                # For 1D arrays, estimate spacing from unique values
                unique_x = np.unique(grid_x)
                unique_y = np.unique(grid_y)
                unique_z = np.unique(grid_z)
                dx = np.diff(unique_x).min() if len(unique_x) > 1 else 1.0
                dy = np.diff(unique_y).min() if len(unique_y) > 1 else 1.0
                dz = np.diff(unique_z).min() if len(unique_z) > 1 else 1.0
                interval_size = min(dx, dy, dz) * 0.1  # 10% of smallest spacing
            else:
                # Fallback: use 1% of Z range
                z_range = coords_valid[:, 2].max() - coords_valid[:, 2].min()
                interval_size = max(z_range * 0.01, 0.1)  # At least 0.1m

            logger.info(f"KRIGING VIS: Converting {len(coords_valid)} grid points to drillhole format (interval_size={interval_size:.3f}m)")

            # Create drillhole DataFrame format
            # Each grid point becomes a small drillhole interval
            n_points = len(coords_valid)
            hole_ids = [f"KRIGED_{i+1:06d}" for i in range(n_points)]

            # Create assays DataFrame (drillhole format)
            # Note: DrillholeDatabase expects lowercase column names: hole_id, depth_from, depth_to, x, y, z
            assays_df = pd.DataFrame({
                'hole_id': hole_ids,
                'depth_from': coords_valid[:, 2] - interval_size / 2,  # Z - half interval
                'depth_to': coords_valid[:, 2] + interval_size / 2,     # Z + half interval
                'x': coords_valid[:, 0],
                'y': coords_valid[:, 1],
                'z': coords_valid[:, 2],
                variable: estimates_valid  # The kriged estimate as a grade column
            })

            # Create collars DataFrame (required for drillhole visualization)
            collars_df = pd.DataFrame({
                'hole_id': hole_ids,
                'x': coords_valid[:, 0],
                'y': coords_valid[:, 1],
                'z': coords_valid[:, 2]  # Collar at grid point Z
            })

            # Create DrillholeDatabase
            from block_model_viewer.drillholes.datamodel import DrillholeDatabase
            kriged_database = DrillholeDatabase()
            kriged_database.set_table('assays', assays_df)
            kriged_database.set_table('collars', collars_df)

            logger.info(f"KRIGING VIS: Created DrillholeDatabase with {len(assays_df)} intervals")

            # ✅ VISUALIZE AS BLOCK MODEL (avoid heavy drillhole conversion)
            property_min = float(np.nanmin(estimates))
            property_max = float(np.nanmax(estimates))
            logger.info(f"KRIGING VIS: Adding block model layer '{layer_name}' with property '{property_name}'")
            self.viewer_widget.renderer.add_block_model_layer(
                grid,
                property_name=property_name,
                layer_name=layer_name
            )

            # Refresh Interactive Slicer panel if open
            self._refresh_slicer_panel()

            # Update legend via LegendManager (if available)
            if hasattr(self.viewer_widget.renderer, 'legend_manager') and self.viewer_widget.renderer.legend_manager is not None:
                try:
                    self.viewer_widget.renderer.legend_manager.set_continuous(
                        field=property_name.upper(),
                        vmin=property_min,
                        vmax=property_max,
                        cmap_name='viridis'
                    )
                    logger.info("KRIGING VIS: Legend updated successfully")
                except Exception as e:
                    logger.warning(f"KRIGING VIS: Could not update legend: {e}", exc_info=True)

            # Reset camera to show all
            if self.viewer_widget and self.viewer_widget.renderer and self.viewer_widget.renderer.plotter:
                self.viewer_widget.renderer.plotter.reset_camera()
                logger.info("KRIGING VIS: Camera reset")

            # Update property panel to show the new layer
            if hasattr(self, 'property_panel') and self.property_panel:
                try:
                    self.property_panel.update_layer_controls()
                    logger.info(f"KRIGING VIS: Updated property panel with new layer: {layer_name}")
                except AttributeError:
                    if hasattr(self.property_panel, 'update_active_layers'):
                        self.property_panel.update_active_layers()
                        logger.info("KRIGING VIS: Used fallback update_active_layers method")

            # Log active layers after registration
            logger.info(f"KRIGING VIS: Active layers now: {list(self.viewer_widget.renderer.active_layers.keys())}")

            self.status_bar.showMessage(
                f"Kriging results visualized as block model: {variable} (cells={grid.n_cells}, "
                f"range=[{property_min:.2f}, {property_max:.2f}]) | Use Layer Controls to toggle visibility",
                5000
            )
            logger.info(f"KRIGING VIS: Successfully visualized kriging results as block model for {variable}, {grid.n_cells} cells")

        except Exception as e:
            logger.error(f"KRIGING VIS: Error visualizing kriging results: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Error visualizing kriging results:\n{str(e)}\n\nCheck console logs for details."
            )

    def visualize_rbf_results(self, rbf_results: Dict):
        """
        Visualize RBF interpolation results in the main 3D viewer.
        
        AUDIT FIX: This method was missing, causing RBF results to not be displayed.
        
        Parameters
        ----------
        rbf_results : dict
            Dictionary containing grid, grid_values, metadata, diagnostics
        """
        logger.info("RBF VIS: Visualization requested - guiding user to Block Model Builder")

        # Store RBF results in registry for later use
        try:
            if hasattr(self, 'controller') and self.controller:
                registry = getattr(self.controller, 'registry', None)
                if registry and hasattr(registry, 'register_rbf_results'):
                    # Results should already be registered by the panel, but ensure they're there
                    if registry.get_rbf_results(copy_data=False) is None:
                        registry.register_rbf_results(rbf_results, source_panel="MainWindow")
                        logger.info("RBF VIS: Stored RBF results in registry")
        except Exception as e:
            logger.warning(f"RBF VIS: Could not store results in registry: {e}")

        # Show guidance message (similar to kriging results)
        metadata = rbf_results.get('metadata', {})
        variable = metadata.get('variable', 'Unknown')
        n_samples = metadata.get('n_samples', 0)
        kernel = metadata.get('kernel', 'Unknown')

        # Build diagnostics message if available
        diag_msg = ""
        diagnostics = rbf_results.get('diagnostics')
        if diagnostics:
            r2 = diagnostics.get('R2', 'N/A')
            rmse = diagnostics.get('RMSE', 'N/A')
            mae = diagnostics.get('MAE', 'N/A')
            diag_msg = f"\n\nDiagnostics:\n  • R²: {r2:.4f}\n  • RMSE: {rmse:.4f}\n  • MAE: {mae:.4f}"

        QMessageBox.information(
            self,
            "RBF Interpolation Complete",
            f"RBF interpolation completed successfully.\n\n"
            f"Variable: {variable}\n"
            f"Kernel: {kernel}\n"
            f"Input Samples: {n_samples}{diag_msg}\n\n"
            f"Results have been stored in the data registry.\n\n"
            f"Next step: Open the Block Model Builder to create a block model "
            f"from your RBF results for 3D visualization."
        )

        self.status_bar.showMessage(
            f"RBF interpolation complete: {variable} ({kernel} kernel, {n_samples} samples)",
            5000
        )
        logger.info(f"RBF VIS: Interpolation complete for {variable}, {n_samples} samples")

    # ============================================================================
    # K-MEANS CLUSTERING METHODS
    # ============================================================================

    def open_kmeans_panel(self):
        """Open K-Means Clustering Analysis panel."""
        try:
            logger.info("=== KMEANS: Starting to open K-Means panel ===")

            # Check if block model is loaded
            block_model = self._extract_block_model_from_layers()
            if block_model is None:
                logger.warning("KMEANS: No block model loaded")
                QMessageBox.warning(
                    self,
                    "No Block Model",
                    "Please load a block model first.\n\n"
                    "Go to: File â†’ Load Block Model\n\n"
                    "K-Means clustering requires a block model with properties."
                )
                return

            logger.info(f"KMEANS: Block model loaded with {len(self.current_model.to_dataframe())} blocks")

            # Check if panel already exists and is valid (even if minimized)
            if hasattr(self, 'kmeans_dialog') and self._is_dialog_valid(self.kmeans_dialog):
                # Restore and show the existing dialog
                if self.kmeans_dialog.isMinimized():
                    self.kmeans_dialog.showNormal()
                else:
                    self.kmeans_dialog.show()
                self.kmeans_dialog.raise_()
                self.kmeans_dialog.activateWindow()
                logger.info("KMEANS: Dialog restored")
                return
            else:
                # Dialog was destroyed - clear reference
                if hasattr(self, 'kmeans_dialog'):
                    self.kmeans_dialog = None

            # Create new dialog
            logger.info("KMEANS: Creating new dialog...")
            self.status_bar.showMessage("Opening K-Means Clustering panel...", 2000)

            try:
                # Create dialog
                logger.info("KMEANS: Instantiating QDialog...")
                self.kmeans_dialog = QDialog(None)  # No parent for independent window
                self.kmeans_dialog.setWindowTitle("K-Means Clustering Analysis")
                logger.info("KMEANS: QDialog created successfully")

                # Set window flags for proper minimize behavior
                logger.info("KMEANS: Setting window flags...")
                self.kmeans_dialog.setWindowFlags(
                    Qt.WindowType.Window |
                    Qt.WindowType.WindowMinimizeButtonHint |
                    Qt.WindowType.WindowMaximizeButtonHint |
                    Qt.WindowType.WindowCloseButtonHint
                )
                logger.info("KMEANS: Window flags set")

                # Ensure non-modal behavior
                self.kmeans_dialog.setWindowModality(Qt.WindowModality.NonModal)

                # Prevent dialog from being deleted when closed or minimized
                self.kmeans_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

                # Setup dialog persistence (position/size will be saved/restored)
                self._setup_dialog_persistence(self.kmeans_dialog, 'kmeans_dialog')

                # Dynamic sizing based on screen
                logger.info("KMEANS: Computing window size...")
                screen = QApplication.primaryScreen().geometry()
                width = min(int(screen.width() * 0.6), 900)
                height = min(int(screen.height() * 0.75), 750)
                self.kmeans_dialog.resize(width, height)
                logger.info(f"KMEANS: Window sized to {width}x{height}")

                # Create layout
                logger.info("KMEANS: Creating layout...")
                layout = QVBoxLayout(self.kmeans_dialog)
                layout.setContentsMargins(10, 10, 10, 10)
                logger.info("KMEANS: Layout created")

                # Create panel
                logger.info("KMEANS: Creating KMeansClusteringPanel...")
                self.kmeans_panel = KMeansClusteringPanel()
                logger.info("KMEANS: KMeansClusteringPanel created successfully")

                logger.info("KMEANS: Adding panel to layout...")
                layout.addWidget(self.kmeans_panel)
                logger.info("KMEANS: Panel added to layout")

                # Bind controller
                logger.info("KMEANS: Binding controller to panel...")
                self.kmeans_panel.bind_controller(self.controller)
                logger.info("KMEANS: Controller bound to panel")

                # Set block model
                logger.info("KMEANS: Setting block model on panel...")
                self.kmeans_panel.set_block_model(self.current_model)
                logger.info("KMEANS: Block model set on panel")

                # Connect signals
                logger.info("KMEANS: Connecting signals...")
                self.kmeans_panel.clustering_complete.connect(self.on_clustering_complete)
                logger.info("KMEANS: Signals connected")

                logger.info("KMEANS: K-Means Clustering panel setup complete")
                self.status_bar.showMessage("K-Means Clustering panel ready - Select features and run", 3000)

                # Show as non-modal dialog
                logger.info("KMEANS: Showing dialog...")
                self.kmeans_dialog.show()
                self.kmeans_dialog.raise_()
                self.kmeans_dialog.activateWindow()
                logger.info("KMEANS: Dialog shown successfully")

            except Exception as e:
                logger.error(f"KMEANS: Error during panel creation: {e}", exc_info=True)
                QMessageBox.critical(
                    self,
                    "Panel Creation Error",
                    f"Failed to create K-Means panel:\n\n{str(e)}\n\nCheck the log for details."
                )
                return

        except Exception as e:
            logger.error(f"KMEANS: Fatal error in open_kmeans_panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Fatal Error",
                f"Failed to open K-Means panel:\n\n{str(e)}\n\nPlease check the log file."
            )

    def on_clustering_complete(self, property_name: str, labels: np.ndarray):
        """Handle clustering completion and visualize results."""
        try:
            logger.info(f"KMEANS: Clustering complete, visualizing property: {property_name}")

            # Reload the block model to include the new property
            if self.current_model is not None:
                logger.info("KMEANS: Reloading block model in renderer...")

                # Reload renderer using the correct method
                self.viewer_widget.renderer.load_block_model(self.current_model)
                logger.info("KMEANS: Block model reloaded in renderer")

                # Update property panel
                logger.info("KMEANS: Updating property panel...")
                self.property_panel.set_block_model(self.current_model)
                logger.info("KMEANS: Property panel updated")

                # Make sure the block model layer is active
                if hasattr(self.property_panel, 'active_layer_combo'):
                    block_model_layer = self._get_block_model_layer_name()
                    if block_model_layer:
                        logger.info(f"KMEANS: Setting active layer to '{block_model_layer}'...")
                        self.property_panel.active_layer_combo.setCurrentText(block_model_layer)
                        logger.info("KMEANS: Active layer set")
                    else:
                        logger.warning("KMEANS: Could not find block model layer")

                # Color by cluster property - changing the combo triggers the visualization
                logger.info(f"KMEANS: Setting property combo to {property_name}...")
                self.property_panel.property_combo.setCurrentText(property_name)
                logger.info("KMEANS: Property combo set - automatic coloring triggered")

                # Force a render update
                if hasattr(self.viewer_widget, 'plotter'):
                    logger.info("KMEANS: Forcing render update...")
                    self.viewer_widget.plotter.render()
                    logger.info("KMEANS: Render update complete")

                n_clusters = len(np.unique(labels[labels >= 0]))
                self.status_bar.showMessage(
                    f"âœ“ Clustering results visualized: {property_name} ({n_clusters} clusters)",
                    5000
                )

                logger.info(f"KMEANS: Visualization complete - {property_name} with {n_clusters} clusters")

                QMessageBox.information(
                    self,
                    "Clustering Applied",
                    f"K-Means clustering results have been applied!\n\n"
                    f"Property: {property_name}\n"
                    f"Clusters: {n_clusters}\n\n"
                    f"The 3D viewer now shows blocks colored by cluster."
                )

        except Exception as e:
            logger.error(f"KMEANS: Error visualizing clustering results: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Error visualizing clustering results:\n\n{str(e)}\n\n"
                f"The cluster property '{property_name}' was added to the model,\n"
                f"but visualization failed. Try manually selecting it from\n"
                f"the property panel."
            )

    def open_resource_classification_panel(self):
        """Open JORC/SAMREC-compliant Resource Classification panel."""
        try:
            logger.info("Opening JORC Resource Classification panel...")

            # Check if panel already exists and is valid
            if hasattr(self, 'jorc_classification_dialog') and \
               self._is_dialog_valid(self.jorc_classification_dialog):
                # Restore and show the existing dialog
                if self.jorc_classification_dialog.isMinimized():
                    self.jorc_classification_dialog.showNormal()
                else:
                    self.jorc_classification_dialog.show()
                self.jorc_classification_dialog.raise_()
                self.jorc_classification_dialog.activateWindow()
                logger.info("JORC Classification panel restored")
                return
            else:
                # Dialog was destroyed or doesn't exist - clear the reference
                if hasattr(self, 'jorc_classification_dialog'):
                    self.jorc_classification_dialog = None

            # Create new panel window
            self.jorc_classification_dialog = QDialog(None)  # No parent for independent window
            self.jorc_classification_dialog.setWindowTitle("Resource Classification (JORC/SAMREC)")

            # Set window flags for proper minimize behavior
            self.jorc_classification_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )

            # Ensure non-modal behavior
            self.jorc_classification_dialog.setWindowModality(Qt.WindowModality.NonModal)

            # Prevent dialog from being deleted when closed or minimized
            self.jorc_classification_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.jorc_classification_dialog, 'jorc_classification_dialog')

            # Dynamic sizing - larger for the new cockpit design
            screen = QApplication.primaryScreen().geometry()
            width = min(int(screen.width() * 0.75), 1400)
            height = min(int(screen.height() * 0.85), 900)
            self.jorc_classification_dialog.resize(width, height)

            # Create layout
            layout = QVBoxLayout(self.jorc_classification_dialog)
            layout.setContentsMargins(0, 0, 0, 0)

            # Create new JORC classification panel
            # Domain selection is optional - panel works with full model extent by default
            self.jorc_classification_panel = JORCClassificationPanel()
            layout.addWidget(self.jorc_classification_panel)

            # Bind controller if available
            if self.controller:
                try:
                    self.jorc_classification_panel.bind_controller(self.controller)
                except Exception:
                    pass  # Not all panels have bind_controller

            # Connect visualization request signal
            if hasattr(self.jorc_classification_panel, 'request_visualization'):
                self.jorc_classification_panel.request_visualization.connect(
                    self._handle_classification_visualization
                )

            # Connect classification complete signal
            if hasattr(self.jorc_classification_panel, 'classification_complete'):
                self.jorc_classification_panel.classification_complete.connect(
                    self.on_jorc_classification_complete
                )

            self.status_bar.showMessage("JORC Resource Classification panel ready", 3000)
            logger.info("JORC Classification panel opened successfully")

            # Show as non-modal dialog
            self.jorc_classification_dialog.show()

        except Exception as e:
            logger.error(f"Error opening JORC Classification panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Panel Error",
                f"Failed to open Resource Classification panel:\n\n{str(e)}"
            )

    def open_resource_reporting_panel(self):
        """Open Resource Reporting panel."""
        try:
            logger.info("Opening Resource Reporting panel...")

            # Check if panel already exists and is valid
            if hasattr(self, 'resource_reporting_dialog') and \
               self._is_dialog_valid(self.resource_reporting_dialog):
                # Restore and show the existing dialog
                if self.resource_reporting_dialog.isMinimized():
                    self.resource_reporting_dialog.showNormal()
                else:
                    self.resource_reporting_dialog.show()
                self.resource_reporting_dialog.raise_()
                self.resource_reporting_dialog.activateWindow()
                logger.info("Resource Reporting panel restored")
                return
            else:
                # Dialog was destroyed or doesn't exist - clear the reference
                if hasattr(self, 'resource_reporting_dialog'):
                    self.resource_reporting_dialog = None

            # Create new panel window
            self.resource_reporting_dialog = QDialog(None)  # No parent for independent window
            self.resource_reporting_dialog.setWindowTitle("Resource Summary (JORC/SAMREC)")

            # Set window flags for proper minimize behavior
            self.resource_reporting_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )

            # Ensure non-modal behavior
            self.resource_reporting_dialog.setWindowModality(Qt.WindowModality.NonModal)

            # Prevent dialog from being deleted when closed or minimized
            self.resource_reporting_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.resource_reporting_dialog, 'resource_reporting_dialog')

            # Dynamic sizing - similar to classification panel
            screen = QApplication.primaryScreen().geometry()
            width = min(int(screen.width() * 0.75), 1400)
            height = min(int(screen.height() * 0.85), 900)
            self.resource_reporting_dialog.resize(width, height)

            # Create layout
            layout = QVBoxLayout(self.resource_reporting_dialog)
            layout.setContentsMargins(0, 0, 0, 0)

            # Create new Resource Reporting panel
            self.resource_reporting_panel = ResourceReportingPanel()
            layout.addWidget(self.resource_reporting_panel)

            # Bind controller if available
            if self.controller:
                try:
                    self.resource_reporting_panel.bind_controller(self.controller)
                except Exception:
                    pass  # Not all panels have bind_controller

            # Connect reporting complete signal
            if hasattr(self.resource_reporting_panel, 'reporting_complete'):
                self.resource_reporting_panel.reporting_complete.connect(
                    self.on_resource_reporting_complete
                )

            self.status_bar.showMessage("Resource Reporting panel ready", 3000)
            logger.info("Resource Reporting panel opened successfully")

            # Show as non-modal dialog
            self.resource_reporting_dialog.show()

        except Exception as e:
            logger.error(f"Error opening Resource Reporting panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Panel Error",
                f"Failed to open Resource Reporting panel:\n\n{str(e)}"
            )

    def open_block_property_calculator_panel(self):
        """Open Block Property Calculator panel."""
        try:
            logger.info("Opening Block Property Calculator panel...")

            # Check if panel already exists and is valid
            if hasattr(self, 'block_property_calculator_dialog') and \
               self._is_dialog_valid(self.block_property_calculator_dialog):
                # Restore and show the existing dialog
                if self.block_property_calculator_dialog.isMinimized():
                    self.block_property_calculator_dialog.showNormal()
                else:
                    self.block_property_calculator_dialog.show()
                self.block_property_calculator_dialog.raise_()
                self.block_property_calculator_dialog.activateWindow()
                logger.info("Block Property Calculator panel restored")
                return
            else:
                # Dialog was destroyed or doesn't exist - clear the reference
                if hasattr(self, 'block_property_calculator_dialog'):
                    self.block_property_calculator_dialog = None

            # Create new panel window
            self.block_property_calculator_dialog = QDialog(None)  # No parent for independent window
            self.block_property_calculator_dialog.setWindowTitle("Block Property Calculator")

            # Set window flags for proper minimize behavior
            self.block_property_calculator_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )

            # Ensure non-modal behavior
            self.block_property_calculator_dialog.setWindowModality(Qt.WindowModality.NonModal)

            # Prevent dialog from being deleted when closed or minimized
            self.block_property_calculator_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Setup dialog persistence
            self._setup_dialog_persistence(self.block_property_calculator_dialog, 'block_property_calculator_dialog')

            # Dynamic sizing
            screen = QApplication.primaryScreen().geometry()
            width = min(int(screen.width() * 0.7), 1200)
            height = min(int(screen.height() * 0.8), 850)
            self.block_property_calculator_dialog.resize(width, height)

            # Create layout
            layout = QVBoxLayout(self.block_property_calculator_dialog)
            layout.setContentsMargins(0, 0, 0, 0)

            # Create new Block Property Calculator panel
            self.block_property_calculator_panel = BlockPropertyCalculatorPanel()
            layout.addWidget(self.block_property_calculator_panel)

            # Bind controller if available
            if self.controller:
                try:
                    self.block_property_calculator_panel.bind_controller(self.controller)
                except Exception:
                    pass  # Not all panels have bind_controller

            self.status_bar.showMessage("Block Property Calculator panel ready", 3000)
            logger.info("Block Property Calculator panel opened successfully")

            # Show as non-modal dialog
            self.block_property_calculator_dialog.show()

        except Exception as e:
            logger.error(f"Error opening Block Property Calculator panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Panel Error",
                f"Failed to open Block Property Calculator panel:\n\n{str(e)}"
            )

    def _handle_property_panel_visualization_request(self, grid, layer_name: str):
        """Handle visualization request from property panel when switching between cached block models.

        This handler is called when the user switches to a cached block model layer that's not
        currently active in the renderer. The property panel caches block models (SGSIM, Kriging,
        Classification) so they can be quickly switched without regeneration.

        Args:
            grid: PyVista grid containing the block model
            layer_name: Full layer name (e.g., "SGSIM: FE_PCT", "Resource Classification")
        """
        try:
            logger.info("="*60)
            logger.info(f"PROPERTY PANEL VISUALIZATION REQUEST: {layer_name}")
            logger.info(f"Grid type: {type(grid).__name__}")
            logger.info(f"Grid n_cells: {grid.n_cells if hasattr(grid, 'n_cells') else 'N/A'}")
            logger.info("="*60)

            # Validate viewer availability
            if not self.viewer_widget or not self.viewer_widget.renderer:
                logger.warning("Viewer not available for property panel visualization request")
                QMessageBox.warning(
                    self,
                    "Viewer Not Available",
                    "3D viewer is not initialized. Cannot visualize layer."
                )
                return

            renderer = self.viewer_widget.renderer
            if renderer.plotter is None:
                logger.warning("Plotter not available for property panel visualization request")
                QMessageBox.warning(
                    self,
                    "Plotter Not Available",
                    "3D plotter is not initialized. Cannot visualize layer."
                )
                return

            # Extract property name from layer_name
            # Layer names follow pattern: "METHOD: PROPERTY" (e.g., "SGSIM: FE_PCT", "Kriging: AL2O3_PCT")
            property_name = None
            if ": " in layer_name:
                property_name = layer_name.split(": ", 1)[1]
                logger.info(f"Extracted property name from layer: '{property_name}'")
            else:
                # Fallback: Try to extract from grid's cell_data
                if hasattr(grid, 'cell_data') and len(grid.cell_data.keys()) > 0:
                    property_name = list(grid.cell_data.keys())[0]
                    logger.info(f"Extracted property from grid cell_data: '{property_name}'")
                elif hasattr(grid, 'point_data') and len(grid.point_data.keys()) > 0:
                    property_name = list(grid.point_data.keys())[0]
                    logger.info(f"Extracted property from grid point_data: '{property_name}'")

            if not property_name:
                logger.error(f"Could not extract property name from layer '{layer_name}' or grid data")
                QMessageBox.warning(
                    self,
                    "Property Extraction Failed",
                    f"Could not determine which property to visualize for layer '{layer_name}'."
                )
                return

            # Validate property exists in grid
            has_property = False
            if hasattr(grid, 'cell_data') and property_name in grid.cell_data:
                has_property = True
            elif hasattr(grid, 'point_data') and property_name in grid.point_data:
                has_property = True

            if not has_property:
                available_props = []
                if hasattr(grid, 'cell_data'):
                    available_props.extend(list(grid.cell_data.keys()))
                if hasattr(grid, 'point_data'):
                    available_props.extend(list(grid.point_data.keys()))

                logger.error(f"Property '{property_name}' not found in grid. Available: {available_props}")
                QMessageBox.warning(
                    self,
                    "Property Not Found",
                    f"Property '{property_name}' not found in grid.\n\n"
                    f"Available properties: {', '.join(available_props)}"
                )
                return

            # ================================================================
            # CRITICAL FIX: Cache currently active block models BEFORE switching
            # ================================================================
            # When add_block_model_layer() is called, it removes other block models
            # due to mutual exclusivity. We need to cache them BEFORE removal so
            # they remain available in the layer dropdown for quick switching.
            # ================================================================
            if hasattr(self, 'property_panel') and hasattr(self.property_panel, '_block_layer_cache'):
                # Cache all currently active block-type layers before switching
                block_volume_prefixes = ["Block Model", "Kriging", "SGSIM", "Resource Classification"]
                for name, info in list(renderer.active_layers.items()):
                    # Check if this is a block-type layer
                    is_block_type = (
                        any(name.startswith(prefix) for prefix in block_volume_prefixes) or
                        info.get('type') in ('blocks', 'volume', 'classification')
                    )

                    if is_block_type:
                        layer_grid = info.get('data')
                        if layer_grid is not None and hasattr(layer_grid, 'bounds'):
                            # Cache this layer so it can be restored later
                            self.property_panel._block_layer_cache[name] = layer_grid
                            logger.info(f"Cached block model layer '{name}' before switching")

            logger.info(f"Adding block model layer: property='{property_name}', layer='{layer_name}'")

            # ================================================================
            # SPECIAL HANDLING: Resource Classification layers
            # ================================================================
            # Classification layers are categorical (Measured/Indicated/Inferred/Unclassified)
            # and need different visualization than continuous SGSIM/Kriging layers.
            # Delegate to the classification handler for proper categorical legend.
            # ================================================================
            if "Classification" in layer_name:
                logger.info(f"Detected Classification layer - delegating to classification handler")
                # Call the classification visualization handler which handles categorical legends
                # NOTE: This method already calls update_layer_controls() internally,
                # so we don't need to call it again afterward
                self._handle_classification_visualization(grid, layer_name)
            else:
                # Standard block model (SGSIM, Kriging, etc.) with continuous properties
                # Add block model layer using renderer
                # This will automatically:
                # 1. Remove other block models (mutual exclusivity)
                # 2. Apply coordinate shift
                # 3. Register in active_layers
                # 4. Update legend

                # Suppress property panel cache re-emission during add
                # (same stale-cache issue as classification layers)
                if hasattr(self, 'property_panel') and self.property_panel:
                    if hasattr(self.property_panel, '_block_layer_cache'):
                        self.property_panel._block_layer_cache[layer_name] = grid
                    if hasattr(self.property_panel, '_suppress_cache_reemit'):
                        self.property_panel._suppress_cache_reemit = True

                try:
                    renderer.add_block_model_layer(
                        grid,
                        property_name,
                        layer_name=layer_name
                    )
                finally:
                    if hasattr(self, 'property_panel') and self.property_panel:
                        if hasattr(self.property_panel, '_suppress_cache_reemit'):
                            self.property_panel._suppress_cache_reemit = False

                # Update property panel controls to reflect the new active layer
                # This will re-add cached layers to the dropdown
                # NOTE: Only call this for non-Classification layers since
                # _handle_classification_visualization() already calls it
                if hasattr(self, 'property_panel'):
                    self.property_panel.update_layer_controls()
                    logger.debug("Updated property panel layer controls")

            logger.info(f"Successfully visualized layer '{layer_name}'")

        except Exception as e:
            logger.error(f"Failed to handle property panel visualization request: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Failed to visualize layer '{layer_name}':\n\n{str(e)}"
            )

    def _handle_classification_visualization(self, mesh, name: str):
        """Handle visualization request from classification panel.

        Properly visualizes resource classification with:
        - Categorical colormap (Measured=green, Indicated=yellow, Inferred=red, Unclassified=gray)
        - Categorical legend
        - Layer control integration
        """
        try:
            logger.info("="*60)
            logger.info(f"CLASSIFICATION VISUALIZATION REQUESTED: {name}")
            logger.info(f"Mesh type: {type(mesh).__name__}")
            logger.info(f"Mesh n_cells: {mesh.n_cells if hasattr(mesh, 'n_cells') else 'N/A'}")
            logger.info(f"Mesh n_points: {mesh.n_points if hasattr(mesh, 'n_points') else 'N/A'}")
            logger.info("="*60)

            if not self.viewer_widget or not self.viewer_widget.renderer:
                logger.warning("Viewer not available for classification visualization")
                QMessageBox.warning(self, "Viewer Not Available",
                                  "3D viewer is not initialized. Cannot visualize classification.")
                return

            renderer = self.viewer_widget.renderer
            if renderer.plotter is None:
                logger.warning("Plotter not available for classification visualization")
                QMessageBox.warning(self, "Plotter Not Available",
                                  "3D plotter is not initialized. Cannot visualize classification.")
                return

            # Import classification colors
            try:
                from ..models.jorc_classification_engine import (
                    CLASSIFICATION_COLORS,
                    CLASSIFICATION_ORDER,
                )
            except ImportError:
                CLASSIFICATION_COLORS = {
                    "Measured": "#2ca02c",      # Green
                    "Indicated": "#ffbf00",     # Amber/Yellow
                    "Inferred": "#d62728",      # Red
                    "Unclassified": "#7f7f7f",  # Gray
                }
                CLASSIFICATION_ORDER = ["Measured", "Indicated", "Inferred", "Unclassified"]

            # Remove existing classification layer if present
            try:
                renderer.plotter.remove_actor(name)
                logger.debug(f"Removed existing '{name}' actor")
            except Exception:
                pass

            # ================================================================
            # CRITICAL FIX: Apply coordinate shift to classification mesh
            # ================================================================
            # Classification meshes must use the SAME coordinate shift as drillholes
            # and other layers. Without this, classification blocks remain at UTM
            # coordinates (e.g., 500,000m) while the camera focuses on local coords
            # (~0,0,0), causing classification to be invisible.
            # ================================================================
            try:
                import pyvista as pv
                already_shifted = getattr(mesh, '_coordinate_shifted', False)

                if not already_shifted and hasattr(renderer, '_to_local_precision'):
                    # Get mesh bounds
                    mesh_bounds = mesh.bounds
                    center_point = np.array([
                        (mesh_bounds[0] + mesh_bounds[1]) / 2,
                        (mesh_bounds[2] + mesh_bounds[3]) / 2,
                        (mesh_bounds[4] + mesh_bounds[5]) / 2
                    ]).reshape(1, 3)

                    # Lock global shift if not already set
                    _ = renderer._to_local_precision(center_point)

                    # Apply shift based on mesh type
                    if hasattr(renderer, '_global_shift') and renderer._global_shift is not None:
                        shift = renderer._global_shift

                        if isinstance(mesh, pv.RectilinearGrid):
                            mesh.x = mesh.x - shift[0]
                            mesh.y = mesh.y - shift[1]
                            mesh.z = mesh.z - shift[2]
                            logger.info(f"[CLASSIFICATION] Applied coordinate shift to RectilinearGrid: "
                                       f"shift=[{shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f}]")
                        elif isinstance(mesh, pv.UnstructuredGrid):
                            mesh.points = mesh.points - shift
                            logger.info(f"[CLASSIFICATION] Applied coordinate shift to UnstructuredGrid: "
                                       f"shift=[{shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f}]")
                        else:
                            # Generic fallback: shift points directly
                            if hasattr(mesh, 'points'):
                                mesh.points = mesh.points - shift
                                logger.info(f"[CLASSIFICATION] Applied coordinate shift to mesh points")

                        mesh._coordinate_shifted = True

                        # Log new bounds
                        new_bounds = mesh.bounds
                        logger.info(f"[CLASSIFICATION] Shifted bounds: ({new_bounds[0]:.2f}, {new_bounds[1]:.2f}, "
                                   f"{new_bounds[2]:.2f}, {new_bounds[3]:.2f}, {new_bounds[4]:.2f}, {new_bounds[5]:.2f})")
            except Exception as e:
                logger.warning(f"Could not apply coordinate shift to classification mesh: {e}")

            # Create custom colormap for classification categories
            # Map: 0=Measured(green), 1=Indicated(yellow), 2=Inferred(red), 3=Unclassified(gray)
            # PyVista expects hex color strings for categorical colormaps
            classification_cmap = [
                CLASSIFICATION_COLORS["Measured"],      # 0 - Green
                CLASSIFICATION_COLORS["Indicated"],     # 1 - Amber/Yellow
                CLASSIFICATION_COLORS["Inferred"],      # 2 - Red
                CLASSIFICATION_COLORS["Unclassified"],  # 3 - Gray
            ]

            # Determine scalar name
            # Check both cell_data and point_data for Classification scalars
            scalar_name = None
            if mesh.active_scalars_name:
                scalar_name = mesh.active_scalars_name
                logger.info(f"Using active scalars: {scalar_name}")
            elif "Classification" in mesh.cell_data:
                scalar_name = "Classification"
                mesh.set_active_scalars("Classification", preference="cell")
                logger.info("Using Classification from cell_data")
            elif "Classification" in mesh.point_data:
                scalar_name = "Classification"
                mesh.set_active_scalars("Classification", preference="point")
                logger.info("Using Classification from point_data")
            elif "CLASS" in mesh.cell_data:
                scalar_name = "CLASS"
                mesh.set_active_scalars("CLASS", preference="cell")
                logger.info("Using CLASS from cell_data")
            else:
                logger.warning(f"No Classification scalars found. Available cell_data: {list(mesh.cell_data.keys())}, point_data: {list(mesh.point_data.keys())}")

            # DEBUG: Log actual scalar values in the mesh
            if scalar_name and scalar_name in mesh.cell_data:
                scalars = mesh.cell_data[scalar_name]
                unique_vals, counts = np.unique(scalars, return_counts=True)
                logger.info(f"[DEBUG] Mesh scalar '{scalar_name}' unique values: {unique_vals.tolist()}")
                logger.info(f"[DEBUG] Mesh scalar '{scalar_name}' value counts: {dict(zip(unique_vals, counts))}")
                logger.info(f"[DEBUG] Mesh scalar range: [{scalars.min()}, {scalars.max()}]")

            # DEBUG: Log colormap being used
            logger.info(f"[DEBUG] Classification colormap: {classification_cmap}")
            logger.info(f"[DEBUG] PyVista add_mesh params: clim=[-0.5, 3.5], n_colors=4")

            # Add mesh with categorical colormap
            # Provide exactly 4 colors for values 0,1,2,3 with centered bins
            actor = renderer.plotter.add_mesh(
                mesh,
                name=name,
                scalars=scalar_name,
                cmap=classification_cmap,
                n_colors=4,  # CRITICAL: Force 4 discrete color bins (no interpolation)
                clim=[-0.5, 3.5],  # Centered bins for 4 integer categories (0,1,2,3)
                show_edges=True,
                edge_color='black',
                line_width=0.5,
                opacity=1.0,
                show_scalar_bar=False,  # LegendManager handles legends
                pickable=True,
                lighting=True,
            )

            # Update legend via LegendManager - use DISCRETE (categorical) mode
            if hasattr(renderer, 'legend_manager') and renderer.legend_manager is not None:
                try:
                    # Build category colors dict for the legend
                    category_colors = {}
                    for cat in CLASSIFICATION_ORDER:
                        hex_color = CLASSIFICATION_COLORS[cat]
                        # Convert hex to RGBA tuple (0-1 range)
                        r = int(hex_color[1:3], 16) / 255.0
                        g = int(hex_color[3:5], 16) / 255.0
                        b = int(hex_color[5:7], 16) / 255.0
                        category_colors[cat] = (r, g, b, 1.0)

                    renderer.legend_manager.update_discrete(
                        property_name='Classification',
                        categories=CLASSIFICATION_ORDER,
                        category_colors=category_colors,
                        subtitle="Resource Classification"
                    )
                    logger.info("Updated legend with categorical classification")
                except Exception as e:
                    logger.warning(f"Could not update legend: {e}")

            # ================================================================
            # CRITICAL FIX: Update property panel cache AND suppress re-emission
            # ================================================================
            # add_layer() has mutual exclusivity logic that removes old block layers.
            # This removal triggers the property panel's layer-change callback, which
            # detects the layer is gone and re-emits it from its _block_layer_cache.
            # If the cache holds a STALE grid from a previous classification run,
            # the old data overwrites the new classification — making it appear that
            # Measured/Indicated blocks are missing.
            #
            # TWO-PART FIX:
            # 1. Update the cache with the NEW mesh before add_layer fires.
            # 2. Suppress cache re-emission entirely during add_layer to avoid
            #    a redundant round-trip that can still race with the new layer.
            # ================================================================
            if hasattr(self, 'property_panel') and self.property_panel:
                if hasattr(self.property_panel, '_block_layer_cache'):
                    self.property_panel._block_layer_cache[name] = mesh
                    logger.info(f"[CLASSIFICATION] Updated property panel cache for '{name}' BEFORE add_layer")
                if hasattr(self.property_panel, '_suppress_cache_reemit'):
                    self.property_panel._suppress_cache_reemit = True

            # Register as layer for layer controls
            # IMPORTANT: Store mesh directly (same as other block models) for property panel compatibility
            try:
                renderer.add_layer(name, actor, data=mesh, layer_type='classification')
            finally:
                # Always restore re-emission capability
                if hasattr(self, 'property_panel') and self.property_panel:
                    if hasattr(self.property_panel, '_suppress_cache_reemit'):
                        self.property_panel._suppress_cache_reemit = False

            # Update property panel if available
            if hasattr(self, 'property_panel') and self.property_panel:
                try:
                    if hasattr(self.property_panel, 'update_layer_controls'):
                        self.property_panel.update_layer_controls()
                    elif hasattr(self.property_panel, 'update_active_layers'):
                        self.property_panel.update_active_layers()
                except Exception as e:
                    logger.debug(f"Could not update property panel: {e}")

            # Force render WITHOUT camera reset
            # CRITICAL FIX: Don't reset camera - it moves to wrong position
            # The blocks are already at the correct local coordinates
            renderer.plotter.render()
            logger.info("[CLASSIFICATION] Rendered classification (camera NOT reset - keeping current view)")

            # DON'T call reset_camera() or refresh_scene() - they cause the camera
            # to jump to the wrong position, making blocks disappear

            self.status_bar.showMessage(f"Classification visualization added: {name}", 3000)
            logger.info(f"Added classification visualization: {name} ({mesh.n_cells} blocks)")

        except Exception as e:
            logger.error(f"Error visualizing classification: {e}", exc_info=True)

    def on_jorc_classification_complete(self, result):
        """Handle JORC classification completion."""
        try:
            logger.info("JORC classification complete")

            # Register classified block model in registry so other panels can access it
            if hasattr(result, 'classified_df') and result.classified_df is not None:
                try:
                    if self.registry:
                        # Register as classified block model
                        self.registry.register_classified_block_model(
                            block_model=result.classified_df,
                            source_panel="JORCClassificationPanel",
                            metadata={
                                'ruleset': result.ruleset.to_dict() if hasattr(result.ruleset, 'to_dict') else {},
                                'variogram': {
                                    'range_major': result.variogram.range_major,
                                    'range_semi': result.variogram.range_semi,
                                    'range_minor': result.variogram.range_minor,
                                    'sill': result.variogram.sill,
                                },
                                'domain': result.domain_name,
                                'summary': result.summary,
                            }
                        )
                        logger.info("Registered classified block model in registry")
                except Exception as e:
                    logger.warning(f"Failed to register classified block model: {e}")

            # Get summary from result
            if hasattr(result, 'summary'):
                summary = result.summary
                measured = summary.get('Measured', {}).get('count', 0)
                indicated = summary.get('Indicated', {}).get('count', 0)
                inferred = summary.get('Inferred', {}).get('count', 0)
                unclass = summary.get('Unclassified', {}).get('count', 0)

                self.status_bar.showMessage(
                    f"Classification complete: Measured={measured}, Indicated={indicated}, "
                    f"Inferred={inferred}, Unclassified={unclass}",
                    8000
                )

            # Update legend if available
            try:
                pass
                # Legend integration would go here if legend panel exists
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error handling classification result: {e}", exc_info=True)

    def on_resource_reporting_complete(self, result):
        """Handle resource reporting completion."""
        try:
            logger.info("Resource reporting complete")

            # Register resource summary in registry for other panels to access
            if hasattr(result, 'metadata') and result.metadata:
                try:
                    if hasattr(self, 'registry') and self.registry:
                        # Register as resource summary
                        self.registry.register_resource_summary(
                            summary=result,
                            source_panel="ResourceReportingPanel",
                            metadata=result.metadata
                        )
                        logger.info("Registered resource summary in registry")
                except Exception as e:
                    logger.warning(f"Failed to register resource summary: {e}")

            # Update status bar with summary info
            if hasattr(result, 'rows') and result.rows:
                total_tonnage = sum(row.total_tonnage_t for row in result.rows)
                total_metal = sum(row.contained_metal_t for row in result.rows)

                self.status_bar.showMessage(
                    f"Resource summary complete: {len(result.rows)} categories, "
                    f"{total_tonnage:,.0f}t total tonnage, {total_metal:,.0f}t contained metal",
                    8000
                )

        except Exception as e:
            logger.error(f"Error handling resource reporting result: {e}", exc_info=True)


    def open_grade_tonnage_panel(self):
        """Open Grade-Tonnage Curves and Cut-off Sensitivity Analysis panel."""
        try:
            logger.info("Opening Grade-Tonnage Analysis panel...")

            # Check if block model is loaded (regular or classified)
            has_block_model = self._has_valid_block_model()
            if not has_block_model and hasattr(self, 'registry') and self.registry:
                # Check for classified block model
                classified_bm = self.registry.get_classified_block_model()
                has_block_model = classified_bm is not None

            if not has_block_model:
                QMessageBox.warning(
                    self,
                    "No Block Model",
                    "Please load or create a block model first.\n\n"
                    "Options:\n"
                    "• File → Load Block Model\n"
                    "• Analysis → Resource Modelling → Build Block Model\n"
                    "• Analysis → Resource Classification → Run Classification\n\n"
                    "The Grade-Tonnage Analysis needs a block model to analyze."
                )
                return

            # Check if panel already exists and is valid (even if minimized)
            if hasattr(self, 'grade_tonnage_dialog') and \
               self._is_dialog_valid(self.grade_tonnage_dialog):
                # Restore and show the existing dialog
                if self.grade_tonnage_dialog.isMinimized():
                    self.grade_tonnage_dialog.showNormal()
                else:
                    self.grade_tonnage_dialog.show()
                self.grade_tonnage_dialog.raise_()
                self.grade_tonnage_dialog.activateWindow()
                logger.info("Grade-Tonnage Analysis panel restored")
                return
            else:
                # Dialog was destroyed - clear reference
                if hasattr(self, 'grade_tonnage_dialog'):
                    self.grade_tonnage_dialog = None

            # Import here to avoid circular imports
            from .grade_tonnage_panel import GradeTonnagePanel

            # Create new panel window
            self.grade_tonnage_dialog = QDialog(None)  # No parent for independent window
            self.grade_tonnage_dialog.setWindowTitle("Grade-Tonnage Curves & Cut-off Sensitivity")

            # Set window flags for proper minimize behavior
            self.grade_tonnage_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )

            # Ensure non-modal behavior
            self.grade_tonnage_dialog.setWindowModality(Qt.WindowModality.NonModal)

            # Prevent dialog from being deleted when closed or minimized
            self.grade_tonnage_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Dynamic sizing - larger for charts
            screen = QApplication.primaryScreen().geometry()
            width = min(int(screen.width() * 0.7), 1200)
            height = min(int(screen.height() * 0.8), 900)
            self.grade_tonnage_dialog.resize(width, height)

            # Create layout
            layout = QVBoxLayout(self.grade_tonnage_dialog)
            layout.setContentsMargins(10, 10, 10, 10)

            # Create panel
            self.grade_tonnage_panel = GradeTonnagePanel()
            layout.addWidget(self.grade_tonnage_panel)

            # Bind controller so panel can access registry
            if hasattr(self, 'controller') and self.controller:
                self.grade_tonnage_panel.bind_controller(self.controller)

            # Set block model - try current_model first, then registry
            block_model = self.current_model
            if block_model is None and hasattr(self, 'registry') and self.registry:
                # Try classified first, then regular
                block_model = self.registry.get_classified_block_model()
                if block_model is None:
                    block_model = self.registry.get_block_model()

            if block_model is not None:
                self.grade_tonnage_panel.set_block_model(block_model)
                logger.info(f"Grade-Tonnage: Set block model ({type(block_model).__name__})")
            else:
                logger.warning("Grade-Tonnage: No block model available to set")

            self.status_bar.showMessage("Grade-Tonnage Analysis panel ready", 3000)
            logger.info("Grade-Tonnage Analysis panel opened successfully")

            # Show as non-modal dialog
            self.grade_tonnage_dialog.show()

        except Exception as e:
            logger.error(f"Error opening Grade-Tonnage Analysis panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Panel Error",
                f"Failed to open Grade-Tonnage Analysis panel:\n\n{str(e)}"
            )

    def open_grade_tonnage_basic_panel(self):
        """Open Basic Grade-Tonnage Analysis panel (simplified, no economic optimization)."""
        try:
            logger.info("Opening Basic Grade-Tonnage Analysis panel...")

            # Check if block model is loaded
            has_block_model = self._has_valid_block_model()
            if not has_block_model and hasattr(self, 'registry') and self.registry:
                classified_bm = self.registry.get_classified_block_model()
                has_block_model = classified_bm is not None

            if not has_block_model:
                QMessageBox.warning(
                    self,
                    "No Block Model",
                    "Please load or create a block model first.\n\n"
                    "Options:\n"
                    "• File → Load Block Model\n"
                    "• Analysis → Resource Modelling → Build Block Model"
                )
                return

            # Check if panel already exists
            if hasattr(self, 'grade_tonnage_basic_dialog') and \
               self._is_dialog_valid(self.grade_tonnage_basic_dialog):
                if self.grade_tonnage_basic_dialog.isMinimized():
                    self.grade_tonnage_basic_dialog.showNormal()
                else:
                    self.grade_tonnage_basic_dialog.show()
                self.grade_tonnage_basic_dialog.raise_()
                self.grade_tonnage_basic_dialog.activateWindow()
                logger.info("Basic Grade-Tonnage panel restored")
                return
            else:
                if hasattr(self, 'grade_tonnage_basic_dialog'):
                    self.grade_tonnage_basic_dialog = None

            from .grade_tonnage_basic_panel import GradeTonnageBasicPanel

            # Create new panel window
            self.grade_tonnage_basic_dialog = QDialog(None)
            self.grade_tonnage_basic_dialog.setWindowTitle("Basic Grade-Tonnage Curves")
            self.grade_tonnage_basic_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )
            self.grade_tonnage_basic_dialog.setWindowModality(Qt.WindowModality.NonModal)
            self.grade_tonnage_basic_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Dynamic sizing
            screen = QApplication.primaryScreen().geometry()
            width = min(int(screen.width() * 0.6), 1000)
            height = min(int(screen.height() * 0.7), 800)
            self.grade_tonnage_basic_dialog.resize(width, height)

            layout = QVBoxLayout(self.grade_tonnage_basic_dialog)
            layout.setContentsMargins(10, 10, 10, 10)

            self.grade_tonnage_basic_panel = GradeTonnageBasicPanel()
            layout.addWidget(self.grade_tonnage_basic_panel)

            if hasattr(self, 'controller') and self.controller:
                self.grade_tonnage_basic_panel.bind_controller(self.controller)

            # Set block model
            block_model = self.current_model
            if block_model is None and hasattr(self, 'registry') and self.registry:
                block_model = self.registry.get_classified_block_model()
                if block_model is None:
                    block_model = self.registry.get_block_model()

            if block_model is not None and hasattr(self.grade_tonnage_basic_panel, 'set_block_model'):
                self.grade_tonnage_basic_panel.set_block_model(block_model)
                logger.info(f"Basic Grade-Tonnage: Set block model ({type(block_model).__name__})")

            self.status_bar.showMessage("Basic Grade-Tonnage panel ready", 3000)
            logger.info("Basic Grade-Tonnage panel opened successfully")
            self.grade_tonnage_basic_dialog.show()

        except Exception as e:
            logger.error(f"Error opening Basic Grade-Tonnage panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Panel Error",
                f"Failed to open Basic Grade-Tonnage panel:\n\n{str(e)}"
            )

    def open_cutoff_optimization_panel(self):
        """Open Cut-off Optimization panel for economic analysis (NPV/IRR/Payback)."""
        try:
            logger.info("Opening Cut-off Optimization panel...")

            # Check if block model is loaded
            has_block_model = self._has_valid_block_model()
            if not has_block_model and hasattr(self, 'registry') and self.registry:
                classified_bm = self.registry.get_classified_block_model()
                has_block_model = classified_bm is not None

            if not has_block_model:
                QMessageBox.warning(
                    self,
                    "No Block Model",
                    "Please load or create a block model first.\n\n"
                    "Options:\n"
                    "• File → Load Block Model\n"
                    "• Analysis → Resource Modelling → Build Block Model\n\n"
                    "Cut-off optimization requires a block model with grade data."
                )
                return

            # Check if panel already exists
            if hasattr(self, 'cutoff_optimization_dialog') and \
               self._is_dialog_valid(self.cutoff_optimization_dialog):
                if self.cutoff_optimization_dialog.isMinimized():
                    self.cutoff_optimization_dialog.showNormal()
                else:
                    self.cutoff_optimization_dialog.show()
                self.cutoff_optimization_dialog.raise_()
                self.cutoff_optimization_dialog.activateWindow()
                logger.info("Cut-off Optimization panel restored")
                return
            else:
                if hasattr(self, 'cutoff_optimization_dialog'):
                    self.cutoff_optimization_dialog = None

            from .cutoff_optimization_panel import CutoffOptimizationPanel

            # Create new panel window
            self.cutoff_optimization_dialog = QDialog(None)
            self.cutoff_optimization_dialog.setWindowTitle("Cut-off Grade Optimization")
            self.cutoff_optimization_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )
            self.cutoff_optimization_dialog.setWindowModality(Qt.WindowModality.NonModal)
            self.cutoff_optimization_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Dynamic sizing - wider for economic inputs
            screen = QApplication.primaryScreen().geometry()
            width = min(int(screen.width() * 0.75), 1200)
            height = min(int(screen.height() * 0.85), 950)
            self.cutoff_optimization_dialog.resize(width, height)

            layout = QVBoxLayout(self.cutoff_optimization_dialog)
            layout.setContentsMargins(10, 10, 10, 10)

            self.cutoff_optimization_panel = CutoffOptimizationPanel()
            layout.addWidget(self.cutoff_optimization_panel)

            if hasattr(self, 'controller') and self.controller:
                self.cutoff_optimization_panel.bind_controller(self.controller)

            # Set block model
            block_model = self.current_model
            if block_model is None and hasattr(self, 'registry') and self.registry:
                block_model = self.registry.get_classified_block_model()
                if block_model is None:
                    block_model = self.registry.get_block_model()

            if block_model is not None and hasattr(self.cutoff_optimization_panel, 'set_block_model'):
                self.cutoff_optimization_panel.set_block_model(block_model)
                logger.info(f"Cut-off Optimization: Set block model ({type(block_model).__name__})")

            self.status_bar.showMessage("Cut-off Optimization panel ready", 3000)
            logger.info("Cut-off Optimization panel opened successfully")
            self.cutoff_optimization_dialog.show()

        except Exception as e:
            logger.error(f"Error opening Cut-off Optimization panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Panel Error",
                f"Failed to open Cut-off Optimization panel:\n\n{str(e)}"
            )

    def open_swath_analysis_3d_panel(self):
        """Open 3D Swath Analysis panel for geostatistical estimation reliability assessment."""
        try:
            logger.info("Opening 3D Swath Analysis panel...")

            # Check if block model is loaded
            if not self._has_valid_block_model():
                QMessageBox.warning(
                    self,
                    "No Block Model",
                    "Please load a block model first.\n\n"
                    "Go to: File → Load Block Model\n\n"
                    "3D Swath Analysis requires a block model with estimated grades."
                )
                return

            # Check if panel already exists and is valid (even if minimized)
            if hasattr(self, 'swath_analysis_3d_dialog') and \
               self._is_dialog_valid(self.swath_analysis_3d_dialog):
                # Restore and show the existing dialog
                if self.swath_analysis_3d_dialog.isMinimized():
                    self.swath_analysis_3d_dialog.showNormal()
                else:
                    self.swath_analysis_3d_dialog.show()
                self.swath_analysis_3d_dialog.raise_()
                self.swath_analysis_3d_dialog.activateWindow()
                logger.info("3D Swath Analysis panel restored")
                return
            else:
                # Dialog was destroyed - clear reference
                if hasattr(self, 'swath_analysis_3d_dialog'):
                    self.swath_analysis_3d_dialog = None

            # Import here to avoid circular imports
            from .swath_analysis_3d_panel import SwathAnalysis3DPanel

            # Create new panel window
            self.swath_analysis_3d_dialog = QDialog(None)  # No parent for independent window
            self.swath_analysis_3d_dialog.setWindowTitle("3D Swath Analysis - Geostatistical Reliability")

            # Set window flags for proper minimize behavior
            self.swath_analysis_3d_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )

            # Ensure non-modal behavior
            self.swath_analysis_3d_dialog.setWindowModality(Qt.WindowModality.NonModal)

            # Prevent dialog from being deleted when closed or minimized
            self.swath_analysis_3d_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Dynamic sizing - larger for charts and analysis
            screen = QApplication.primaryScreen().geometry()
            width = min(int(screen.width() * 0.8), 1400)
            height = min(int(screen.height() * 0.85), 1000)
            self.swath_analysis_3d_dialog.resize(width, height)

            # Create layout
            layout = QVBoxLayout(self.swath_analysis_3d_dialog)
            layout.setContentsMargins(10, 10, 10, 10)

            # Create panel
            self.swath_analysis_3d_panel = SwathAnalysis3DPanel()
            layout.addWidget(self.swath_analysis_3d_panel)

            # Bind controller for analysis tasks
            if self.controller:
                self.swath_analysis_3d_panel.bind_controller(self.controller)

            # Set block model - try current_model first, then layers
            block_model_set = False
            if self.current_model is not None:
                import pandas as pd
                is_valid = True
                if isinstance(self.current_model, pd.DataFrame):
                    is_valid = not self.current_model.empty
                if is_valid:
                    self.swath_analysis_3d_panel.set_block_model(self.current_model)
                    block_model_set = True

            # If no current_model, try to get from layers
            if not block_model_set:
                layer_df = self._get_block_model_from_layers()
                if layer_df is not None:
                    self.swath_analysis_3d_panel.set_block_model(layer_df)
                    block_model_set = True
                    logger.info("3D Swath Analysis: Loaded block model from renderer layers")

            # Check for existing drillhole composites and auto-load
            composite_loaded = False
            if (hasattr(self.viewer_widget.renderer, 'drillhole_data') and
                self.viewer_widget.renderer.drillhole_data is not None):
                drillhole_data = self.viewer_widget.renderer.drillhole_data
                if 'composites_df' in drillhole_data:
                    composites_df = drillhole_data['composites_df']
                    if composites_df is not None and not composites_df.empty:
                        # Auto-load composites into swath panel
                        self.swath_analysis_3d_panel.set_composite_data(
                            composites_df,
                            source="From drillholes"
                        )
                        composite_loaded = True
                        logger.info(f"Auto-loaded {len(composites_df)} composites from drillhole data")

            # Update status message based on composite detection
            if composite_loaded:
                self.status_bar.showMessage(
                    "3D Swath Analysis ready - Drillhole composites detected and loaded",
                    5000
                )
            else:
                self.status_bar.showMessage(
                    "3D Swath Analysis panel ready - Load composite data to begin",
                    5000
                )

            logger.info("3D Swath Analysis panel opened successfully")

            # Show as non-modal dialog
            self.swath_analysis_3d_dialog.show()

        except Exception as e:
            logger.error(f"Error opening 3D Swath Analysis panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Panel Error",
                f"Failed to open 3D Swath Analysis panel:\n\n{str(e)}"
            )

    def open_declustering_panel(self):
        """Open Declustering Analysis panel for statistical defensibility."""
        try:
            logger.info("Opening Declustering Analysis panel...")

            # Check if drillhole data is available
            registry = self.controller.registry if self.controller and hasattr(self.controller, "registry") else None
            if registry is None:
                try:
                    from ..core.data_registry import DataRegistry
                    registry = DataRegistry.instance()
                    self._registry = registry
                except Exception:
                    registry = None
            registry_data = registry.get_drillhole_data() if registry else None
            if not registry_data:
                QMessageBox.warning(
                    self,
                    "No Drillhole Data",
                    "No drillhole data available.\n\n"
                    "Please load drillhole data first:\n"
                    "Data → Drillholes → Import Drillholes\n\n"
                    "Declustering requires composited drillhole samples with X, Y coordinates."
                )
                return

            # Check if panel already exists and is valid (even if minimized)
            if hasattr(self, 'declustering_dialog') and \
               self._is_dialog_valid(self.declustering_dialog):
                # Restore and show the existing dialog
                if self.declustering_dialog.isMinimized():
                    self.declustering_dialog.showNormal()
                else:
                    self.declustering_dialog.show()
                self.declustering_dialog.raise_()
                self.declustering_dialog.activateWindow()
                logger.info("Declustering Analysis panel restored")
                return
            else:
                # Dialog was destroyed or doesn't exist - clear the reference
                if hasattr(self, 'declustering_dialog'):
                    self.declustering_dialog = None

            # Import here to avoid circular imports
            from .declustering_panel import DeclusteringPanel

            # Create new panel window
            self.declustering_dialog = QDialog(None)  # No parent for independent window
            self.declustering_dialog.setWindowTitle("Declustering Analysis - JORC/SAMREC Compliant")

            # Set window flags for proper minimize behavior
            self.declustering_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )

            # Ensure non-modal behavior
            self.declustering_dialog.setWindowModality(Qt.WindowModality.NonModal)

            # Prevent dialog from being deleted when closed or minimized
            self.declustering_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Create panel instance with dialog as parent
            panel = DeclusteringPanel(parent=self.declustering_dialog)
            # Bind controller so panel can access registry
            if self.controller:
                try:
                    panel.bind_controller(self.controller)
                except Exception:
                    logger.debug("Failed to bind controller to declustering panel", exc_info=True)
            elif registry is not None:
                try:
                    panel._registry = registry
                    if hasattr(panel, "_connect_registry"):
                        panel._connect_registry()
                except Exception:
                    logger.debug("Failed to attach registry fallback to declustering panel", exc_info=True)

            # Create layout
            from PyQt6.QtWidgets import QVBoxLayout
            layout = QVBoxLayout(self.declustering_dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(panel)

            # Set initial size
            self.declustering_dialog.resize(1000, 700)

            # Show dialog
            self.declustering_dialog.show()
            logger.info("Declustering Analysis panel opened successfully")

        except Exception as e:
            logger.error(f"Error opening Declustering Analysis panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Panel Error",
                f"Failed to open Declustering Analysis panel:\n\n{str(e)}"
            )

    def visualize_declustered_weights(self, viz_request: Dict[str, Any]):
        """
        Visualize declustered weights on drillholes in the 3D renderer.
        
        Parameters
        ----------
        viz_request : dict
            Dictionary containing:
            - 'declustered_dataframe': DataFrame with X, Y, Z and 'Declustering_Weight' column
            - 'summary': dict with declustering statistics
        """
        try:
            df = viz_request.get('declustered_dataframe')
            summary = viz_request.get('summary', {})

            if df is None or df.empty:
                QMessageBox.warning(
                    self, "No Data",
                    "No declustered data available for visualization."
                )
                return

            # Ensure required columns exist
            weight_col = 'Declustering_Weight'
            if weight_col not in df.columns:
                # Try alternative column names
                for alt in ['Weight', 'WEIGHT', 'weight', 'DeclusterWeight']:
                    if alt in df.columns:
                        weight_col = alt
                        break
                else:
                    QMessageBox.warning(
                        self, "Missing Column",
                        f"Cannot find declustering weight column in data.\n"
                        f"Available columns: {list(df.columns)}"
                    )
                    return

            # Check for coordinate columns
            if not all(col in df.columns for col in ['X', 'Y', 'Z']):
                QMessageBox.warning(
                    self, "Missing Coordinates",
                    "Declustered data must have X, Y, Z coordinate columns."
                )
                return

            logger.info(f"Visualizing declustered weights: {len(df)} samples, weight column: {weight_col}")

            # Get weight statistics
            weights = df[weight_col].values
            w_min, w_max = float(np.nanmin(weights)), float(np.nanmax(weights))
            w_mean = float(np.nanmean(weights))

            logger.info(f"Weight range: [{w_min:.4f}, {w_max:.4f}], mean: {w_mean:.4f}")

            # Create a PyVista point cloud from the declustered data
            import pyvista as pv

            points = df[['X', 'Y', 'Z']].values
            point_cloud = pv.PolyData(points)
            point_cloud[weight_col] = weights

            # Add to renderer as a new layer
            if self.viewer_widget and self.viewer_widget.renderer:
                layer_name = "Declustered Weights"

                # Remove existing layer if present
                if layer_name in self.viewer_widget.renderer.active_layers:
                    self.viewer_widget.renderer.clear_layer(layer_name)

                # Add point cloud with weights as scalar
                self.viewer_widget.renderer.plotter.add_mesh(
                    point_cloud,
                    scalars=weight_col,
                    cmap='RdYlGn',  # Red (low weight) -> Yellow -> Green (high weight)
                    point_size=8,
                    render_points_as_spheres=True,
                    name=layer_name,
                    clim=[w_min, w_max],
                    show_scalar_bar=True,
                    scalar_bar_args={
                        'title': 'Declustering Weight',
                        'title_font_size': 12,
                        'label_font_size': 10,
                        'color': 'white',
                        'position_x': 0.85,
                        'position_y': 0.1,
                        'width': 0.1,
                        'height': 0.7,
                        'fmt': '%.3f'
                    }
                )

                # Track the layer
                self.viewer_widget.renderer.active_layers[layer_name] = {
                    'type': 'points',
                    'data': point_cloud,
                    'visible': True
                }

                # Update legend
                if hasattr(self.viewer_widget.renderer, 'legend_manager') and self.viewer_widget.renderer.legend_manager:
                    try:
                        self.viewer_widget.renderer.legend_manager.set_continuous(
                            field='Declustering Weight',
                            vmin=w_min,
                            vmax=w_max,
                            cmap_name='RdYlGn'
                        )
                    except Exception as e:
                        logger.warning(f"Could not update legend: {e}")

                # Reset camera
                self.viewer_widget.renderer.plotter.reset_camera()

                # Show summary
                naïve_mean = summary.get('naive_mean', 'N/A')
                declustered_mean = summary.get('declustered_mean', 'N/A')

                self.status_bar.showMessage(
                    f"Declustered weights visualized: {len(df)} samples | "
                    f"Weight range: [{w_min:.3f}, {w_max:.3f}] | "
                    f"Declustered mean: {declustered_mean}",
                    5000
                )

                logger.info(f"Declustered weights visualization complete: {len(df)} samples")
            else:
                QMessageBox.warning(
                    self, "Renderer Not Available",
                    "3D renderer is not available. Please ensure the viewer is open."
                )

        except Exception as e:
            logger.error(f"Failed to visualize declustered weights: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Visualization Error",
                f"Failed to visualize declustered weights:\n\n{str(e)}"
            )

    def open_pit_optimisation_panel(self):
        """Open Pit Optimisation panel for Lerchs-Grossmann pit optimization."""
        try:
            logger.info("Opening Pit Optimisation panel...")

            # Resolve block model from current model, registry, or renderer layers.
            block_model = self._extract_block_model_from_layers()
            if block_model is None:
                QMessageBox.warning(
                    self,
                    "No Block Model",
                    "Please load a block model first.\n\n"
                    "Go to: File → Open File...\n\n"
                    "Or build a block model from estimation:\n"
                    "Estimations → Resource Modelling → Build Block Model\n\n"
                    "Pit optimization requires a block model with X, Y, Z coordinates and grade properties."
                )
                return

            # Check if panel already exists and is visible
            if hasattr(self, 'pit_optimisation_dialog') and self._is_dialog_valid(self.pit_optimisation_dialog):
                if self.pit_optimisation_dialog.isVisible():
                    self.pit_optimisation_dialog.raise_()
                    self.pit_optimisation_dialog.activateWindow()
                else:
                    # Window exists but is minimized - restore it
                    self.pit_optimisation_dialog.show()
                    self.pit_optimisation_dialog.raise_()
                    self.pit_optimisation_dialog.activateWindow()
                logger.info("Pit Optimisation panel restored")
                return

            # Import panel - use high-performance version
            try:
                from block_model_viewer.ui.pit_optimizer_panel import (
                    PitOptimizerPanel as PitOptimisationPanel,
                )
            except ImportError:
                # Fallback to original panel if new one not available
                from block_model_viewer.ui.pit_optimisation_panel import PitOptimisationPanel

            # Create dialog window
            self.pit_optimisation_dialog = QDialog(None)  # No parent for independent window
            self.pit_optimisation_dialog.setWindowTitle("Pit Optimisation - Lerchs-Grossmann")

            # Set window flags for proper minimize behavior
            self.pit_optimisation_dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )

            # Ensure non-modal behavior
            self.pit_optimisation_dialog.setWindowModality(Qt.WindowModality.NonModal)

            # Prevent dialog from being deleted when closed or minimized
            self.pit_optimisation_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Setup dialog persistence (position/size will be saved/restored)
            self._setup_dialog_persistence(self.pit_optimisation_dialog, 'pit_optimisation_dialog')

            # Dynamic sizing
            screen = QApplication.primaryScreen().geometry()
            width = min(int(screen.width() * 0.5), 600)
            height = min(int(screen.height() * 0.75), 900)
            self.pit_optimisation_dialog.resize(width, height)

            # Create layout
            layout = QVBoxLayout(self.pit_optimisation_dialog)
            layout.setContentsMargins(10, 10, 10, 10)

            # Create panel
            self.pit_optimisation_panel = PitOptimisationPanel(self)
            layout.addWidget(self.pit_optimisation_panel)

            # Convert block model to DataFrame
            if PANDAS_AVAILABLE and isinstance(block_model, pd.DataFrame):
                block_df = block_model.copy(deep=True)
            elif hasattr(block_model, "to_dataframe"):
                block_df = block_model.to_dataframe()
            else:
                raise ValueError(f"Unsupported block model type: {type(block_model)}")

            if block_df is None or block_df.empty:
                QMessageBox.warning(
                    self,
                    "No Block Model",
                    "Resolved block model is empty. Please load or generate a valid block model."
                )
                return

            # Normalize coordinate column names (case-insensitive, handles x/y/z, XC/YC/ZC, etc.)
            from block_model_viewer.models.pit_optimizer import normalize_coordinate_columns
            block_df = normalize_coordinate_columns(block_df)

            # Compute grid spec from block model
            # Get unique coordinates to determine grid spacing
            # normalize_coordinate_columns creates lowercase x, y, z
            if 'x' not in block_df.columns or 'y' not in block_df.columns or 'z' not in block_df.columns:
                QMessageBox.warning(
                    self,
                    "Missing Coordinates",
                    f"Could not find coordinate columns (x/y/z) after normalization.\n\n"
                    f"Available columns: {', '.join(block_df.columns.tolist())}"
                )
                return

            x_coords = np.unique(block_df['x'].values)
            y_coords = np.unique(block_df['y'].values)
            z_coords = np.unique(block_df['z'].values)

            nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)

            # Compute increments
            if nx > 1:
                xinc = float(np.mean(np.diff(np.sort(x_coords))))
                xmin = float(np.min(x_coords)) - xinc/2
            else:
                xinc = 1.0
                xmin = float(x_coords[0]) - xinc/2

            if ny > 1:
                yinc = float(np.mean(np.diff(np.sort(y_coords))))
                ymin = float(np.min(y_coords)) - yinc/2
            else:
                yinc = 1.0
                ymin = float(y_coords[0]) - yinc/2

            if nz > 1:
                zinc = float(np.mean(np.diff(np.sort(z_coords))))
                zmin = float(np.min(z_coords)) - zinc/2
            else:
                zinc = 1.0
                zmin = float(z_coords[0]) - zinc/2

            grid_spec = {
                'nx': nx,
                'ny': ny,
                'nz': nz,
                'xmin': xmin,
                'ymin': ymin,
                'zmin': zmin,
                'xinc': xinc,
                'yinc': yinc,
                'zinc': zinc
            }

            # Set block model data
            self.pit_optimisation_panel.set_block_model(block_df, grid_spec)

            # Connect visualization signal
            self.pit_optimisation_panel.request_visualization.connect(
                self.on_pit_optimisation_visualization
            )

            self.status_bar.showMessage("Pit Optimisation panel ready - Configure parameters and run", 3000)
            logger.info("Pit Optimisation panel opened successfully")

            # Show as non-modal dialog
            self.pit_optimisation_dialog.show()

        except Exception as e:
            logger.error(f"Error opening Pit Optimisation panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Panel Error",
                f"Failed to open Pit Optimisation panel:\n\n{str(e)}"
            )

    def on_pit_optimisation_visualization(self, mesh_or_grid, layer_name: str):
        """
        Handle visualization request from pit optimization panel.
        
        Args:
            mesh_or_grid: PyVista mesh/grid or DataFrame to visualize
            layer_name: Name for the layer or property name
        """
        try:
            import pandas as pd
            import pyvista as pv

            # Handle DataFrame input (from new high-performance panel)
            if isinstance(mesh_or_grid, pd.DataFrame):
                # Convert DataFrame to PyVista mesh
                # Find coordinate columns
                x_col = None
                y_col = None
                z_col = None

                for col in mesh_or_grid.columns:
                    col_upper = col.upper()
                    if col_upper in ['XC', 'X', 'EASTING', 'E', 'X_CENTROID']:
                        x_col = col
                    elif col_upper in ['YC', 'Y', 'NORTHING', 'N', 'Y_CENTROID']:
                        y_col = col
                    elif col_upper in ['ZC', 'Z', 'ELEVATION', 'ELEV', 'Z_CENTROID']:
                        z_col = col

                if not (x_col and y_col and z_col):
                    # Try lowercase versions
                    for col in mesh_or_grid.columns:
                        if col.lower() in ['x', 'xc', 'easting', 'e']:
                            x_col = col
                        elif col.lower() in ['y', 'yc', 'northing', 'n']:
                            y_col = col
                        elif col.lower() in ['z', 'zc', 'elevation', 'elev']:
                            z_col = col

                if not (x_col and y_col and z_col):
                    raise ValueError(f"Could not find coordinate columns. Available: {list(mesh_or_grid.columns)}")

                # Create point cloud
                points = mesh_or_grid[[x_col, y_col, z_col]].values
                point_cloud = pv.PolyData(points)

                # Add IN_PIT or specified property as scalar
                if layer_name in mesh_or_grid.columns:
                    point_cloud[layer_name] = mesh_or_grid[layer_name].values
                elif 'IN_PIT' in mesh_or_grid.columns:
                    point_cloud['IN_PIT'] = mesh_or_grid['IN_PIT'].values
                    point_cloud.set_active_scalars('IN_PIT')

                mesh_or_grid = point_cloud

            # Add mesh/grid as an actor
            actor = self.viewer_widget.plotter.add_mesh(
                mesh_or_grid,
                name=layer_name,
                show_edges=False,
                opacity=0.8,
                show_scalar_bar=False
            )

            # Add to layer system (will auto-update legend if mesh has scalars)
            self.viewer_widget.renderer.add_layer(
                layer_name,
                actor,
                data=mesh_or_grid,
                layer_type='pit'
            )

            # Explicitly check for scalars and update legend if needed
            try:
                if hasattr(mesh_or_grid, 'active_scalars_name') and mesh_or_grid.active_scalars_name:
                    scalar_name = mesh_or_grid.active_scalars_name
                    if hasattr(mesh_or_grid, 'cell_data') and scalar_name in mesh_or_grid.cell_data:
                        scalar_data = mesh_or_grid.cell_data[scalar_name]
                        self.viewer_widget.renderer.update_legend(f"{layer_name}: {scalar_name}", scalar_data)
                    elif hasattr(mesh_or_grid, 'point_data') and scalar_name in mesh_or_grid.point_data:
                        scalar_data = mesh_or_grid.point_data[scalar_name]
                        self.viewer_widget.renderer.update_legend(f"{layer_name}: {scalar_name}", scalar_data)
            except Exception as e:
                try:
                    e_msg = str(e)
                    logger.debug(f"Could not update legend for pit visualization: {e_msg}")
                except Exception:
                    logger.debug("Could not update legend for pit visualization: <unprintable error>")

            # Reset camera
            self.viewer_widget.plotter.reset_camera()

            self.status_bar.showMessage(
                f"Pit optimization result '{layer_name}' added to viewer (use Layer Controls to toggle)",
                5000
            )

            logger.info(f"Visualized pit optimization result: {layer_name}")

        except Exception as e:
            logger.error(f"Error visualizing pit optimization result: {e}", exc_info=True)
            QMessageBox.warning(
                self,
                "Visualization Error",
                f"Error visualizing pit optimization result:\n\n{str(e)}"
            )

    def open_uncertainty_panel(self):
        """Open Uncertainty Analysis panel for Monte Carlo and risk analysis."""
        try:
            logger.info("Opening Uncertainty Analysis panel...")

            # Check if block model is loaded
            block_model = self._extract_block_model_from_layers()
            if block_model is None:
                QMessageBox.warning(
                    self,
                    "No Block Model",
                    "Please load a block model before opening Uncertainty Analysis.\n\n"
                    "You can:\n"
                    "- Load a CSV file via File → Import\n"
                    "- Build a model via Estimation → Block Model Builder\n"
                    "- Generate estimates via Estimation → Kriging or SGSIM\n"
                    "- Create classification via Estimation → Resource Classification"
                )
                return

            # Convert BlockModel to DataFrame
            logger.info(f"Converting block model to DataFrame ({_safe_get_block_count(block_model)} blocks)")
            block_df = block_model.to_dataframe()

            logger.info(f"Block model ready with {len(block_df)} blocks and {len(block_df.columns)} attributes")

            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Uncertainty Analysis & Risk Dashboard")
            dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
            dialog.resize(1200, 800)

            # Create layout
            layout = QVBoxLayout()

            # Import panel
            from block_model_viewer.ui.uncertainty_panel import UncertaintyAnalysisPanel

            # Create panel
            panel = UncertaintyAnalysisPanel(parent=dialog)
            panel.set_block_model(block_df)
            layout.addWidget(panel)

            # Close button
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(close_btn)
            layout.addLayout(button_layout)

            dialog.setLayout(layout)

            # Show dialog
            logger.info("Uncertainty Analysis panel opened successfully")
            dialog.exec()

        except ImportError as e:
            logger.error(f"Import error for uncertainty analysis: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import uncertainty analysis module:\n\n{str(e)}\n\n"
                f"Please ensure all dependencies are installed:\n"
                f"pip install numpy pandas scipy matplotlib seaborn"
            )
        except Exception as e:
            logger.error(f"Error opening Uncertainty Analysis panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Error opening uncertainty analysis panel:\n\n{str(e)}"
            )

    def open_underground_panel(self):
        """Open Underground Mining panel for stope optimization and scheduling."""
        try:
            logger.info("Opening Underground Mining panel as floating dialog...")

            # Create dialog if it doesn't exist (prevent duplicates)
            if self.underground_panel_dialog is None:
                from PyQt6.QtWidgets import (
                    QApplication,
                    QDialog,
                    QHBoxLayout,
                    QPushButton,
                    QScrollArea,
                    QVBoxLayout,
                )

                dialog = QDialog(self)
                dialog.setWindowTitle("Underground Mining - Stope Optimization & Scheduling")
                # Enable minimize/maximize/close buttons
                dialog.setWindowFlags(
                    Qt.WindowType.Window |
                    Qt.WindowType.WindowMinimizeButtonHint |
                    Qt.WindowType.WindowMaximizeButtonHint |
                    Qt.WindowType.WindowCloseButtonHint
                )
                dialog.setModal(False)

                # Dynamic sizing
                screen = QApplication.primaryScreen().geometry()
                width = min(int(screen.width() * 0.85), 1500)
                height = min(int(screen.height() * 0.9), 1100)
                dialog.resize(width, height)

                layout = QVBoxLayout()

                # Scrollable content
                scroll = QScrollArea(dialog)
                scroll.setWidgetResizable(True)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

                panel = UndergroundPanel(parent=scroll, main_window=self)

                # Bind controller to panel (required for running analysis)
                if hasattr(self, 'controller') and self.controller:
                    panel.bind_controller(self.controller)

                scroll.setWidget(panel)
                layout.addWidget(scroll)

                # Attempt block model injection
                try:
                    bm = self._extract_block_model_from_layers()
                    if bm is not None and hasattr(panel, 'set_block_model'):
                        ok = panel.set_block_model(bm)
                        if ok:
                            logger.info("Injected existing block model into UndergroundPanel (dialog)")
                except Exception as e:
                    try:
                        e_msg = str(e)
                        logger.debug(f"Block model injection into UG dialog failed: {e_msg}")
                    except Exception:
                        logger.debug("Block model injection into UG dialog failed: <unprintable error>")

                # Buttons
                buttons = QHBoxLayout()
                buttons.addStretch()
                close_btn = QPushButton("Close")
                close_btn.clicked.connect(dialog.accept)
                buttons.addWidget(close_btn)
                layout.addLayout(buttons)

                dialog.setLayout(layout)

                # Store reference to prevent garbage collection and duplicate creation
                self.underground_panel_dialog = dialog
                if hasattr(self, '_open_panels'):
                    self._open_panels.append(dialog)

            # Show and raise the dialog
            if self.underground_panel_dialog.windowState() & Qt.WindowState.WindowMinimized:
                self.underground_panel_dialog.setWindowState(Qt.WindowState.WindowNoState)
            self.underground_panel_dialog.show()
            self.underground_panel_dialog.raise_()
            self.underground_panel_dialog.activateWindow()

            self.status_bar.showMessage(
                "Underground Mining popup opened - ready for stope optimization & scheduling", 4000
            )
            logger.info("Underground Mining dialog opened successfully")

        except ImportError as e:
            logger.error(f"Import error for underground mining: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import underground mining module:\n\n{str(e)}\n\n"
                f"Please ensure all dependencies are installed:\n"
                f"pip install numpy pandas networkx pyomo"
            )
        except Exception as e:
            logger.error(f"Error opening Underground Mining panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Error opening underground mining panel:\n\n{str(e)}"
            )

    def open_data_registry_status_panel(self):
        """Open Data Registry Status panel."""
        try:
            if self.data_registry_status_dialog is None:
                self.data_registry_status_dialog = QDialog(None)  # No parent - independent window
                self.data_registry_status_dialog.setWindowTitle("Data Registry Status")
                self.data_registry_status_dialog.setMinimumSize(900, 700)

                # Set window flags for proper behavior
                self.data_registry_status_dialog.setWindowFlags(
                    Qt.WindowType.Window |
                    Qt.WindowType.WindowMinimizeButtonHint |
                    Qt.WindowType.WindowCloseButtonHint |
                    Qt.WindowType.WindowTitleHint
                )

                layout = QVBoxLayout(self.data_registry_status_dialog)
                layout.setContentsMargins(0, 0, 0, 0)

                self.data_registry_status_panel = DataRegistryStatusPanel()
                layout.addWidget(self.data_registry_status_panel)

                # Store reference to prevent garbage collection
                self._open_panels.append(self.data_registry_status_dialog)

            # Show and raise the dialog
            if self.data_registry_status_dialog.windowState() & Qt.WindowState.WindowMinimized:
                self.data_registry_status_dialog.setWindowState(Qt.WindowState.WindowNoState)
            self.data_registry_status_dialog.show()
            self.data_registry_status_dialog.raise_()
            self.data_registry_status_dialog.activateWindow()

            logger.info("Opened Data Registry Status panel")
        except Exception as e:
            logger.error(f"Error opening Data Registry Status panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open Data Registry Status panel:\n{str(e)}"
            )

    # NOTE: closeEvent is defined in the LIFECYCLE section at line ~10909
    # This duplicate definition was removed to avoid confusion

    def restart_application(self):
        """Close and restart the application."""
        try:
            # Best-effort: save layout/state
            try:
                self._save_window_layout()
            except Exception:
                pass
            # Start a new detached process for this app
            import sys
            args = ["-m", "block_model_viewer"]
            started = QProcess.startDetached(sys.executable, args)
            if not started:
                raise RuntimeError("Failed to start new application process")
            # Quit current instance
            try:
                QApplication.quit()
            except Exception:
                self.close()
        except Exception as e:
            logger.error(f"Restart failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Restart Error", f"Restart failed:\n{e}")

    def open_esg_panel(self):
        """Open ESG Dashboard for environmental, social and governance metrics."""
        try:
            logger.info("Opening ESG Dashboard...")

            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("ESG Dashboard - Environmental, Social & Governance")
            # Enable minimize/maximize/close buttons
            dialog.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )
            dialog.setModal(False)

            # Dynamic sizing
            screen = QApplication.primaryScreen().geometry()
            width = min(int(screen.width() * 0.8), 1400)
            height = min(int(screen.height() * 0.85), 1000)
            dialog.resize(width, height)

            # Create layout
            layout = QVBoxLayout()

            # Scrollable content
            from PyQt6.QtWidgets import QScrollArea
            scroll = QScrollArea(dialog)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

            # Create panel inside scroll
            panel = ESGDashboardPanel(parent=scroll)
            scroll.setWidget(panel)
            layout.addWidget(scroll)

            # Close button
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(close_btn)
            layout.addLayout(button_layout)

            dialog.setLayout(layout)

            # FIX 4: Show dialog non-modally for safer freeze testing
            logger.info("ESG Dashboard opened successfully")
            self.status_bar.showMessage("ESG Dashboard ready - Load production schedule to calculate metrics", 3000)
            dialog.show()  # Non-modal is safer for freeze testing

        except ImportError as e:
            logger.error(f"Import error for ESG dashboard: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import ESG dashboard module:\n\n{str(e)}\n\n"
                f"Please ensure all dependencies are installed:\n"
                f"pip install numpy pandas matplotlib"
            )
        except Exception as e:
            logger.error(f"Error opening ESG Dashboard: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Error opening ESG dashboard:\n\n{str(e)}"
            )

    def _extract_block_model_from_layers(self) -> Optional[BlockModel]:
        """
        Extract block model from multiple sources in priority order.
        
        Checks the following sources:
        1. DataRegistry (SGSIM/kriging results registered there)
        2. Controller's current_model (uploaded/imported models)
        3. Block Model Builder results (buildmodel_panel)
        4. Resource Classification results
        5. Kriging/SGSIM estimation results (volume layers)
        6. Any other block model layers in the renderer
        
        Returns:
            BlockModel object if found, None otherwise
        """
        try:
            # 1. Check DataRegistry first (SGSIM/kriging + classified models).
            registry = None
            if hasattr(self, '_registry') and self._registry is not None:
                registry = self._registry
            elif hasattr(self, 'controller') and self.controller is not None:
                registry = getattr(self.controller, 'registry', None)

            if registry is not None:
                try:
                    classified_model = registry.get_classified_block_model()
                    if classified_model is not None:
                        logger.info("Using classified block model from DataRegistry")
                        return classified_model

                    registry_model = registry.get_block_model()
                    if registry_model is not None:
                        logger.info("Using block model from DataRegistry")
                        return registry_model
                except Exception as e:
                    logger.debug(f"DataRegistry check failed: {e}")

            # 2. Check controller's current_model (most common case for imported models)
            if hasattr(self, 'controller') and self.controller is not None:
                if hasattr(self.controller, 'current_model') and self.controller.current_model is not None:
                    logger.info("Using block model from controller.current_model")
                    return self.controller.current_model

            # 3. Check Block Model Builder panel
            if hasattr(self, 'buildmodel_panel') and self.buildmodel_panel is not None:
                if hasattr(self.buildmodel_panel, 'block_model') and self.buildmodel_panel.block_model is not None:
                    logger.info("Using block model from Block Model Builder")
                    return self.buildmodel_panel.block_model

            # 4. Check Resource Classification panel
            if hasattr(self, 'resource_classification_panel') and self.resource_classification_panel is not None:
                if hasattr(self.resource_classification_panel, 'classified_model') and \
                   self.resource_classification_panel.classified_model is not None:
                    logger.info("Using classified block model from Resource Classification")
                    return self.resource_classification_panel.classified_model

            # 4. Check renderer layers (for estimation results and other generated models)
            if not hasattr(self, 'viewer_widget') or not hasattr(self.viewer_widget, 'renderer'):
                logger.info("No viewer_widget or renderer available")
                return None

            renderer = self.viewer_widget.renderer
            if not hasattr(renderer, 'active_layers') or not renderer.active_layers:
                logger.info("No active layers in renderer")
                return None

            # Priority order: prefer SGSIM/Kriging results over generic blocks
            # Look for SGSIM mean first, then other SGSIM stats, then kriging, then any volume/blocks
            priority_patterns = [
                'SGSIM.*MEAN', 'SGSIM.*mean',
                'SGSIM.*P50', 'SGSIM.*p50',
                'SGSIM.*MEAN_BT', 'SGSIM.*mean_bt',
                'SGSIM',
                'KRIGING', 'Kriging', 'kriging',
                'Block Model', 'block_model'
            ]

            # First pass: try to find priority layers
            for pattern in priority_patterns:
                for layer_name, layer_info in renderer.active_layers.items():
                    layer_type = layer_info.get('type', '')
                    layer_data = layer_info.get('data', None)

                    # Check if layer name matches pattern (case-insensitive)
                    if not re.search(pattern, layer_name, re.IGNORECASE):
                        continue

                    # Accept both 'blocks' and 'volume' types for block models
                    if (layer_type in ('blocks', 'volume') and layer_data is not None):
                        block_model = self._extract_block_model_from_grid(layer_data, layer_name)
                        if block_model is not None:
                            return block_model

            # Second pass: check all layers if no priority layer found
            for layer_name, layer_info in renderer.active_layers.items():
                layer_type = layer_info.get('type', '')
                layer_data = layer_info.get('data', None)

                # Accept both 'blocks' and 'volume' types for block models
                if (layer_type in ('blocks', 'volume') and layer_data is not None):
                    block_model = self._extract_block_model_from_grid(layer_data, layer_name)
                    if block_model is not None:
                        return block_model

            logger.info("No block model found in active layers")
            return None

        except Exception as e:
            logger.error(f"Error extracting block model from layers: {e}", exc_info=True)
            return None

    def _extract_block_model_from_grid(self, grid, layer_name: str) -> Optional[BlockModel]:
        """
        Extract BlockModel from a PyVista grid (legacy method - PyVista isolation).
        
        Parameters
        ----------
        grid : PyVista grid or GridPayload
            Grid containing block model data (legacy PyVista or new payload)
        layer_name : str
            Name of the layer (for logging)
        
        Returns
        -------
        BlockModel or None
            Extracted block model if successful, None otherwise
        """
        try:
            # Handle GridPayload (new API)
            if hasattr(grid, 'grid') and hasattr(grid, 'scalars'):
                # It's a GridPayload - extract from grid attribute
                grid_points = grid.grid
                if isinstance(grid_points, tuple) and len(grid_points) == 3:
                    # Meshgrid format
                    xx, yy, zz = grid_points
                    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
                elif isinstance(grid_points, np.ndarray) and grid_points.ndim == 2:
                    # Already (N, 3) array
                    pass
                else:
                    return None

                cell_centers = grid_points
                # Estimate dimensions from grid spacing
                if len(cell_centers) > 1:
                    dx = np.mean(np.diff(np.unique(cell_centers[:, 0])))
                    dy = np.mean(np.diff(np.unique(cell_centers[:, 1])))
                    dz = np.mean(np.diff(np.unique(cell_centers[:, 2])))
                else:
                    dx = dy = dz = 1.0

                dimensions = np.full((len(cell_centers), 3), [dx, dy, dz])
                block_model = BlockModel()
                block_model.set_geometry(cell_centers, dimensions)

                if grid.scalars is not None:
                    block_model.add_property(layer_name, grid.scalars)

                return block_model

            # Legacy: Handle PyVista grid
            try:
                import pyvista as pv
            except ImportError:
                return None

            # Check if it's a valid grid type
            if not isinstance(grid, (pv.RectilinearGrid, pv.StructuredGrid, pv.UnstructuredGrid)):
                return None

            logger.info(f"Extracting block model from layer '{layer_name}' (grid type: {type(grid).__name__})")

            # Get cell centers (block centroids)
            cell_centers = grid.cell_centers().points  # (N, 3)

            # Get block dimensions from grid spacing
            if isinstance(grid, pv.RectilinearGrid):
                # RectilinearGrid has x, y, z coordinate arrays
                x_coords = grid.x
                y_coords = grid.y
                z_coords = grid.z

                # Calculate block dimensions (spacing between grid points)
                if len(x_coords) > 1:
                    dx = np.mean(np.diff(x_coords))
                else:
                    dx = 1.0

                if len(y_coords) > 1:
                    dy = np.mean(np.diff(y_coords))
                else:
                    dy = 1.0

                if len(z_coords) > 1:
                    dz = np.mean(np.diff(z_coords))
                else:
                    dz = 1.0

            elif isinstance(grid, pv.StructuredGrid):
                # StructuredGrid: estimate block size from cell bounds
                # Get bounds of first cell as reference
                bounds = grid.GetCell(0).GetBounds()
                if bounds:
                    dx = bounds[1] - bounds[0]
                    dy = bounds[3] - bounds[2]
                    dz = bounds[5] - bounds[4]
                else:
                    # Fallback: estimate from grid extent
                    grid_bounds = grid.bounds
                    if grid.dimensions[0] > 1:
                        dx = (grid_bounds[1] - grid_bounds[0]) / (grid.dimensions[0] - 1)
                    else:
                        dx = 1.0
                    if grid.dimensions[1] > 1:
                        dy = (grid_bounds[3] - grid_bounds[2]) / (grid.dimensions[1] - 1)
                    else:
                        dy = 1.0
                    if grid.dimensions[2] > 1:
                        dz = (grid_bounds[5] - grid_bounds[4]) / (grid.dimensions[2] - 1)
                    else:
                        dz = 1.0

            elif isinstance(grid, pv.UnstructuredGrid):
                # UnstructuredGrid: estimate block size from first cell bounds
                if grid.n_cells > 0:
                    bounds = grid.GetCell(0).GetBounds()
                    if bounds:
                        dx = bounds[1] - bounds[0]
                        dy = bounds[3] - bounds[2]
                        dz = bounds[5] - bounds[4]
                    else:
                        # Fallback: use a default size
                        dx = dy = dz = 10.0
                else:
                    return None

            else:
                return None

            # Create dimensions array (all blocks same size for regular grids)
            dimensions = np.full((len(cell_centers), 3), [dx, dy, dz])

            # Create BlockModel
            block_model = BlockModel()
            block_model.set_geometry(cell_centers, dimensions)

            # Extract all properties from cell_data
            for prop_name in grid.cell_data:
                prop_values = grid.cell_data[prop_name]
                if len(prop_values) == len(cell_centers):
                    block_model.add_property(prop_name, prop_values)
                    logger.info(f"Added property '{prop_name}' to extracted block model")

            logger.info(f"Extracted block model from layer '{layer_name}': {len(cell_centers)} blocks, "
                       f"dimensions=({dx:.2f}, {dy:.2f}, {dz:.2f})")
            return block_model

        except ImportError:
            logger.warning("PyVista not available for extracting block model from layers")
            return None
        except Exception as e:
            logger.warning(f"Error extracting block model from grid '{layer_name}': {e}")
            return None

    def _extract_drillhole_coordinates_for_classification(self) -> Optional[pd.DataFrame]:
        """
        Extract drillhole collar coordinates from any available source.
        
        Searches for drillhole data in:
        1. Domain compositing panel (composite data with X, Y, Z)
        2. Property panel drillhole data (if available)
        3. Renderer's drillhole_collar_data (legacy)
        4. Active layers (extract from composite midpoints, not mesh points)
        
        Returns:
            DataFrame with X, Y, Z columns representing drillhole collar locations,
            or None if no drillhole data found.
        """
        try:
            drillhole_coords = []

            # METHOD 1: Check Domain Compositing panel for composite data (most reliable!)
            if hasattr(self, 'domain_compositing_panel') and self.domain_compositing_panel:
                # Try assay data
                comp_df = None
                if hasattr(self.domain_compositing_panel, 'assay_df'):
                    comp_df = self.domain_compositing_panel.assay_df

                if comp_df is not None and not comp_df.empty:
                    if all(col in comp_df.columns for col in ['X', 'Y', 'Z']):
                        # Use assay coordinates
                        coords = comp_df[['X', 'Y', 'Z']].dropna()  # Drop NaN values
                        drillhole_coords = coords.to_dict('records')
                        logger.info(f"Extracted {len(drillhole_coords)} valid coordinates from assay data")
                    else:
                        logger.warning(f"Assay data missing X, Y, Z columns. Available: {comp_df.columns.tolist()}")

            # METHOD 1b: Check stored composites in main window
            if len(drillhole_coords) == 0 and hasattr(self, 'stored_drillhole_data') and self.stored_drillhole_data is not None:
                comp_df = self.stored_drillhole_data
                if not comp_df.empty and all(col in comp_df.columns for col in ['X', 'Y', 'Z']):
                    coords = comp_df[['X', 'Y', 'Z']].dropna()
                    drillhole_coords = coords.to_dict('records')
                    logger.info(f"Extracted {len(drillhole_coords)} valid coordinates from stored composites")

            # METHOD 2: Check Property Panel drillhole data
            if len(drillhole_coords) == 0 and hasattr(self, 'property_panel') and self.property_panel:
                if hasattr(self.property_panel, 'drillhole_df'):
                    dh_df = self.property_panel.drillhole_df
                    if dh_df is not None and not dh_df.empty:
                        if all(col in dh_df.columns for col in ['X', 'Y', 'Z']):
                            coords = dh_df[['X', 'Y', 'Z']].dropna()
                            drillhole_coords = coords.to_dict('records')
                            logger.info(f"Extracted {len(drillhole_coords)} valid coordinates from property panel")

            # METHOD 3: Check legacy drillhole_collar_data attribute
            if len(drillhole_coords) == 0:
                if hasattr(self.viewer_widget, 'renderer') and hasattr(self.viewer_widget.renderer, 'drillhole_collar_data'):
                    collar_df = self.viewer_widget.renderer.drillhole_collar_data
                    if collar_df is not None and not collar_df.empty:
                        if all(col in collar_df.columns for col in ['X', 'Y', 'Z']):
                            coords = collar_df[['X', 'Y', 'Z']].dropna()
                            drillhole_coords = coords.to_dict('records')
                            logger.info(f"Extracted {len(drillhole_coords)} valid coordinates from drillhole_collar_data")

            # Convert to DataFrame and validate
            if drillhole_coords:
                df = pd.DataFrame(drillhole_coords)

                # Remove any remaining NaN or inf values
                df = df.replace([np.inf, -np.inf], np.nan).dropna()

                if df.empty:
                    logger.warning("All drillhole coordinates contain NaN/inf values after cleaning")
                    return None

                logger.info(f"Total valid drillhole coordinates for classification: {len(df)}")
                return df
            else:
                logger.warning("No drillhole coordinate data found in any source")
                return None

        except Exception as e:
            logger.error(f"Error extracting drillhole coordinates: {e}", exc_info=True)
            return None

    def on_classification_complete(self, classified_df: pd.DataFrame):
        """Handle resource classification completion and visualize results."""
        try:
            logger.info("Classification complete, visualizing results in 3D...")

            # Add classification properties to block model
            if self.current_model is not None:
                # Add Category column
                self.current_model.add_property('Category', classified_df['Category'].values)

                # Add distance and count columns
                self.current_model.add_property('Nearest_DH_Dist', classified_df['Nearest_DH_Dist'].values)

                # Add drillhole count columns (all columns matching pattern)
                for col in classified_df.columns:
                    if col.startswith('DH_Count_'):
                        self.current_model.add_property(col, classified_df[col].values)

                logger.info("Added classification properties to block model")

                # Try to add property to existing grid WITHOUT full reload
                # (full reload is very slow for large models due to O(n*m) sampling)
                renderer = self.viewer_widget.renderer
                grid = None
                grid_source = None

                if renderer:
                    # Method 1: Check block_meshes (traditional block model)
                    if hasattr(renderer, 'block_meshes'):
                        grid = renderer.block_meshes.get('unstructured_grid')
                        if grid is not None:
                            grid_source = "block_meshes"

                    # Method 2: Check active_layers (SGSIM results stored as layers)
                    if grid is None and hasattr(renderer, 'active_layers'):
                        for layer_name, layer_data in renderer.active_layers.items():
                            if layer_name.startswith('Block Model:') and 'data' in layer_data:
                                candidate = layer_data.get('data')
                                if candidate is not None and hasattr(candidate, 'cell_data'):
                                    grid = candidate
                                    grid_source = f"active_layers['{layer_name}']"
                                    break

                if grid is not None and 'Category' in classified_df.columns:
                    # Map category strings to numeric values for coloring
                    category_map = {'Measured': 0, 'Indicated': 1, 'Inferred': 2, 'Unclassified': 3}
                    category_values = classified_df['Category'].map(category_map).fillna(3).astype(np.int32).values

                    # Add to existing grid (check size match)
                    if len(category_values) == grid.n_cells:
                        grid.cell_data['Category'] = category_values
                        grid.cell_data['Nearest_DH_Dist'] = classified_df['Nearest_DH_Dist'].values.astype(np.float32)
                        logger.info(f"Added Category property directly to existing grid from {grid_source} ({grid.n_cells} cells)")
                    else:
                        logger.warning(f"Category array length ({len(category_values)}) != grid cells ({grid.n_cells}), cannot add directly")
                        # DON'T fall back to full reload - it's too slow for large models
                        # Instead, show warning to user
                        QMessageBox.warning(
                            self,
                            "Size Mismatch",
                            f"Classification has {len(category_values)} blocks but grid has {grid.n_cells} cells.\n\n"
                            "Cannot apply classification to current visualization.\n"
                            "Please ensure the block model matches the classification data."
                        )
                        return
                else:
                    logger.warning("No existing grid found in renderer, cannot apply classification directly")
                    QMessageBox.warning(
                        self,
                        "No Grid",
                        "No block model grid found in viewer.\n\n"
                        "Please visualize a block model first before applying classification."
                    )
                    return

                # Update property panel
                self.property_panel.set_block_model(self.current_model)
                logger.info("Updated property panel")

                # Set active layer
                if hasattr(self.property_panel, 'active_layer_combo'):
                    block_model_layer = self._get_block_model_layer_name()
                    if block_model_layer:
                        self.property_panel.active_layer_combo.setCurrentText(block_model_layer)
                    else:
                        logger.warning("Could not find block model layer for category visualization")

                # Color by Category (use categorical colormap)
                self.property_panel.property_combo.setCurrentText('Category')
                # Use discrete/categorical colormap for Category
                if hasattr(self.viewer_widget.renderer, 'set_property_coloring'):
                    self.viewer_widget.renderer.set_property_coloring(
                        'Category',
                        colormap='Set1',  # Categorical colormap
                        discrete=True      # Treat as discrete categories
                    )
                logger.info("Set coloring to Category property with categorical colormap")

                # Force render update
                if hasattr(self.viewer_widget, 'plotter'):
                    self.viewer_widget.plotter.render()

                # Show success message
                summary = classified_df['Category'].value_counts()
                msg = "Resource Classification Visualized!\n\n"
                for cat in ['Measured', 'Indicated', 'Inferred', 'Unclassified']:
                    count = summary.get(cat, 0)
                    pct = count / len(classified_df) * 100
                    msg += f"• {cat}: {count} blocks ({pct:.1f}%)\n"

                msg += "\nThe 3D viewer now shows blocks colored by classification category."

                self.status_bar.showMessage("Resource classification visualized in 3D", 5000)
                logger.info("Classification visualization complete")

        except Exception as e:
            logger.error(f"Error visualizing classification results: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Error visualizing classification results:\n\n{str(e)}\n\n"
                "The classification data was added to the model,\n"
                "but visualization failed. Try manually selecting 'Category'\n"
                "from the property panel."
            )

    # ============================================================================
    # BLOCK MODEL BUILDER METHODS
    # ============================================================================

    def open_block_model_builder(self):
        """Open Block Model Builder panel."""
        # Check if kriging results are available
        has_kriging_data = (hasattr(self, 'kriging_dialog') and
                           self.kriging_dialog.kriging_results is not None)

        # Check if panel already exists and is visible
        if not hasattr(self, 'block_model_builder_dialog') or self.block_model_builder_dialog is None or not self.block_model_builder_dialog.isVisible():
            from block_model_viewer.ui.blockmodel_builder_panel import BlockModelBuilderPanel

            self.status_bar.showMessage("Opening Block Model Builder panel...", 2000)

            # Create as independent window (no parent) so it behaves like Drillhole Loading panel
            self.block_model_builder_dialog = BlockModelBuilderPanel(None, controller=self.controller)
            self.block_model_builder_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Setup dialog persistence (position/size will be saved/restored)
            self._setup_dialog_persistence(self.block_model_builder_dialog, 'block_model_builder_dialog')

            # Connect visualization request signal
            self.block_model_builder_dialog.request_visualization.connect(self.visualize_block_model)

            # If kriging results are available, automatically load them
            if has_kriging_data:
                kriging_results = self.kriging_dialog.kriging_results

                # Create estimation DataFrame from kriging results
                import pandas as pd

                grid_x = kriging_results['grid_x']
                grid_y = kriging_results['grid_y']
                grid_z = kriging_results['grid_z']
                estimates = kriging_results['estimates']
                variances = kriging_results['variances']
                variable = kriging_results['variable']

                # Flatten grids to DataFrame
                estimation_df = pd.DataFrame({
                    'X': grid_x.ravel(order='F'),
                    'Y': grid_y.ravel(order='F'),
                    'Z': grid_z.ravel(order='F'),
                    f'{variable}_est': estimates.ravel(order='F'),
                    'Variance': variances.ravel(order='F')
                })

                # Set estimation data
                self.block_model_builder_dialog.set_estimation_data(
                    estimation_df,
                    grade_col=f'{variable}_est',
                    var_col='Variance'
                )

                logger.info(f"Loaded kriging results into block model builder: {len(estimation_df)} points")
                self.status_bar.showMessage("Block model builder ready with kriging results", 3000)
            else:
                logger.info("Opened Block Model Builder panel (no kriging data available)")
                self.status_bar.showMessage("Block model builder ready - Load estimation data to begin", 3000)
        else:
            self.status_bar.showMessage("Block model builder restored (results preserved)", 2000)

        # Show as non-modal dialog
        self.block_model_builder_dialog.show()
        self.block_model_builder_dialog.raise_()
        self.block_model_builder_dialog.activateWindow()

    def _on_drillhole_control_plot(self, dataset_name: str) -> None:
        """Plot drillholes in the main renderer."""
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return

        # Use injected registry via controller (dependency injection)
        registry = self.controller.registry if self.controller else None
        if registry is None:
            QMessageBox.warning(self, "Error", "DataRegistry is not initialized.")
            return

        data = registry.get_drillhole_data()
        if data is None:
            QMessageBox.warning(self, "Error", "No drillhole data is registered.")
            return

        # Validate data structure
        if not isinstance(data, dict):
            QMessageBox.warning(
                self,
                "Error",
                f"Invalid drillhole data format. Expected dictionary, got {type(data).__name__}."
            )
            logger.error(f"Invalid drillhole data type: {type(data)}")
            return

        # Build database from registry
        from ..drillholes.registry_utils import build_database_from_registry
        try:
            db = build_database_from_registry(data)
        except Exception as e:
            logger.error(f"Failed to build drillhole database from registry: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to build drillhole database:\n\n{str(e)}\n\n"
                "Please check that your drillhole data has the required columns "
                "(holeid, x/y/z for collars, from/to for assays, etc.)."
            )
            return

        # Get composite data if needed
        composite_df = None
        if dataset_name == "Composites":
            # Accept both legacy and newer keys from the registry
            # Fix: Explicitly check for non-empty DataFrames to avoid ValueError with 'or' operator
            composites_from_registry = data.get("composites")
            composites_df_from_registry = data.get("composites_df")
            if isinstance(composites_from_registry, pd.DataFrame) and not composites_from_registry.empty:
                composite_df = composites_from_registry
            elif isinstance(composites_df_from_registry, pd.DataFrame) and not composites_df_from_registry.empty:
                composite_df = composites_df_from_registry
            else:
                composite_df = None

            if composite_df is None or (isinstance(composite_df, pd.DataFrame) and composite_df.empty):
                QMessageBox.warning(self, "Error", "No composite data available.")
                return
            logger.info(f"Using COMPOSITES dataset: {len(composite_df)} rows")
            # CRITICAL: Always replace db.assays with composites when composites are selected
            # build_drillhole_polylines uses database.assays, not composite_df parameter
            try:
                db.set_table("assays", composite_df.copy())
                logger.info(f"Replaced database.assays with composites ({len(composite_df)} rows) for visualization")
            except Exception as e:
                logger.warning(f"Failed to replace assays with composites: {e}", exc_info=True)
        else:
            logger.info("Using RAW ASSAYS dataset (composite_df=None)")
            # Ensure db.assays contains raw assays (not composites)
            # build_database_from_registry already populated db.assays with raw assays
            if db.assays is not None and not db.assays.empty:
                logger.info(f"Using raw assays from database ({len(db.assays)} rows)")
            else:
                logger.warning("No raw assays available in database")

        # Get radius from control panel (use get_radius() which returns properly scaled value)
        radius = 1.0
        if self.drillhole_control_panel:
            radius = self.drillhole_control_panel.get_radius()

        self._start_drillhole_render_status()
        success = False
        legend_metadata: Optional[Dict[str, Any]] = None
        try:
            self.viewer_widget.renderer.remove_drillhole_layer()
            visible_holes = self.drillhole_control_panel.get_visible_holes() if self.drillhole_control_panel else None
            color_mode = self.drillhole_control_panel.get_color_mode() if self.drillhole_control_panel else "Lithology"
            assay_field = self.drillhole_control_panel.get_assay_field() if self.drillhole_control_panel else None
            lith_filter = self.drillhole_control_panel.get_selected_lithologies() if self.drillhole_control_panel else []

            # Handle special "__NONE__" marker from lithology filter
            # This means user unchecked ALL lithology filters - don't render anything
            if lith_filter and "__NONE__" in lith_filter:
                logger.info("No lithologies selected - not rendering drillholes")
                QMessageBox.information(
                    self,
                    "No Lithologies Selected",
                    "No lithologies are selected to display.\n\n"
                    "Please select at least one lithology type in the Lithology Filter."
                )
                self._stop_drillhole_render_status(False)
                return

            # Log what we're plotting
            logger.info(f"Plotting drillholes: radius={radius}, color_mode={color_mode}, "
                       f"visible_holes={len(visible_holes) if visible_holes else 'all'}, "
                       f"lith_filter={lith_filter if lith_filter else 'ALL'}")

            # Warn if no holes are selected
            if visible_holes is not None and len(visible_holes) == 0:
                QMessageBox.warning(
                    self,
                    "No Holes Selected",
                    "No drillholes are selected to plot.\n\n"
                    "Please select at least one hole in the Drillhole Explorer panel."
                )
                self._stop_drillhole_render_status(False)
                return

            progress_cb = lambda frac, msg: self._update_status_progress(msg, frac)
            legend_metadata = self.viewer_widget.renderer.add_drillhole_layer(
                database=db,
                composite_df=composite_df,
                radius=radius,
                color_mode=color_mode,
                assay_field=assay_field,
                visible_holes=visible_holes,
                legend_title=f"{dataset_name} Drillholes",
                progress_callback=progress_cb,
                lith_filter=lith_filter,
            )

            # NOTE: Do NOT call _on_drillhole_data_loaded here - it will reset the user's dataset selection!
            # The control panel already has its own connection to drillholeDataLoaded signal
            # and will update itself when data changes. Calling it here after rendering
            # would reset the dataset combo selection (e.g., from "Composites" back to "Raw Assays")
            # if self.drillhole_control_panel:
            #     self.drillhole_control_panel._on_drillhole_data_loaded(data)

            logger.info(f"Added drillhole layer to renderer (dataset: {dataset_name})")

            # Enable interaction for drillholes if not already enabled
            if self.viewer_widget:
                # Get cell count from block model if loaded
                cell_count = 0
                if self.viewer_widget.current_model:
                    cell_count = self.viewer_widget.current_model.block_count
                self.viewer_widget.enable_interaction(
                    cell_count=cell_count,
                    has_block_model=self.viewer_widget.current_model is not None,
                    has_drillholes=True
                )

            success = True
        except Exception as e:
            logger.error(f"Failed to add drillhole layer: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to render drillholes: {e}")
        finally:
            # CRITICAL FIX: Get FRESH legend metadata AFTER colors have been applied
            # The metadata returned by add_drillhole_layer is stale (created before colors applied)
            # get_drillhole_legend_metadata() returns current state with correct lithology count
            fresh_metadata = self.viewer_widget.renderer.get_drillhole_legend_metadata() if success else None

            # Use fresh metadata for legend update
            if fresh_metadata and fresh_metadata.get("property") is not None:
                try:
                    self._update_legend_from_drillholes(fresh_metadata)
                except Exception as e:
                    logger.error(f"Failed to update legend from drillholes: {e}", exc_info=True)
            if success:
                self._activate_scene_inspector_legend()
                # Only update property panel if fresh_metadata exists
                if fresh_metadata:
                    try:
                        self._update_property_panel_for_drillholes(dataset_name, color_mode, fresh_metadata)
                    except Exception as e:
                        logger.error(f"Failed to update property panel for drillholes: {e}", exc_info=True)

                # CRITICAL FIX: Force app state update after drillhole rendering
                # This ensures PropertyPanel controls are enabled (state -> RENDERED)
                # The layer_change_callback chain is unreliable, so we update explicitly
                try:
                    if self.controller:
                        self.controller._update_state_from_scene()
                        logger.info("Forced app state update after drillhole rendering")
                except Exception as e:
                    logger.warning(f"Failed to update app state after drillhole render: {e}")
            self._stop_drillhole_render_status(success)

    def _on_drillhole_data_registered(self, data: Dict) -> None:
        """Handle drillhole data registration - auto-render like block models.

        NOTE: DrillholeControlPanel has its OWN connection to drillholeDataLoaded signal
        and will update itself. We do NOT call it here to avoid double-update.
        """
        logger.info("MainWindow._on_drillhole_data_registered: Signal received - will auto-render")

        # Defer UI updates and rendering to main thread
        def _update_ui_and_render():
            try:
                dataset_name = data.get("name", "Drillhole Data")
                self.status_bar.showMessage(f"Rendering drillholes: {dataset_name}...", 0)

                # Trigger app state update
                if self.controller:
                    try:
                        self.controller._update_state_from_scene()
                    except Exception as e:
                        logger.warning(f"Failed to update app state: {e}")

                # AUTO-RENDER: Call the plot function directly (like block models do)
                # Use "Raw Assays" as default dataset
                try:
                    self._on_drillhole_control_plot("Raw Assays")
                    logger.info(f"Auto-rendered drillholes: {dataset_name}")
                except Exception as e:
                    logger.error(f"Failed to auto-render drillholes: {e}", exc_info=True)
                    self.status_bar.showMessage(f"Drillhole data loaded: {dataset_name}", 5000)

            except Exception as e:
                logger.error(f"Error in _on_drillhole_data_registered: {e}", exc_info=True)

        # Delay slightly to ensure control panel has processed the data first
        QTimer.singleShot(100, _update_ui_and_render)

    def _on_drillhole_control_clear(self) -> None:
        """Clear drillholes from renderer."""
        if self.viewer_widget and self.viewer_widget.renderer:
            self.viewer_widget.renderer.remove_drillhole_layer()
            logger.info("Cleared drillhole layer")

    def _on_drillhole_radius_changed(self, radius: float) -> None:
        """Update drillhole radius in renderer (uses cached polylines for speed)."""
        if self.viewer_widget and self.viewer_widget.renderer:
            if "drillholes" in self.viewer_widget.renderer.active_layers:
                # Use optimized radius update (re-tubes cached polylines)
                self.viewer_widget.renderer.update_drillhole_radius(radius)
                # Reconnect legend after radius update
                metadata = self.viewer_widget.renderer.get_drillhole_legend_metadata()
                if metadata:
                    self._update_legend_from_drillholes(metadata)

    def _on_drillhole_color_mode_changed(self, mode: str) -> None:
        """Update drillhole color mode in renderer (FAST - no re-render)."""
        if self.viewer_widget and self.viewer_widget.renderer:
            if "drillholes" in self.viewer_widget.renderer.active_layers:
                # FAST PATH: Update colors only, no geometry rebuild
                property_name = "Lithology" if mode == "Lithology" else "FE"
                colormap = "tab10" if mode == "Lithology" else "turbo"

                # Get current assay field from control panel if in assay mode
                if mode != "Lithology" and self.drillhole_control_panel:
                    current_assay = self.drillhole_control_panel.get_assay_field()
                    if current_assay:
                        property_name = current_assay

                self.viewer_widget.renderer._update_drillhole_colors(
                    property_name=property_name,
                    colormap=colormap,
                    color_mode=mode,
                    custom_colors=None
                )

                # Update legend
                metadata = self.viewer_widget.renderer.get_drillhole_legend_metadata()
                if metadata:
                    self._update_legend_from_drillholes(metadata)

                # ✅ SYNC: Update Property Panel to match Drillhole Control Panel
                self._sync_property_panel_from_drillhole_control(mode, property_name, colormap)

                logger.info(f"Fast color mode change to '{mode}'")

    def _on_drillhole_assay_field_changed(self, field: str) -> None:
        """Update drillhole assay field (FAST - no re-render)."""
        if self.viewer_widget and self.viewer_widget.renderer:
            if "drillholes" in self.viewer_widget.renderer.active_layers:
                # FAST PATH: Update colors only, no geometry rebuild
                self.viewer_widget.renderer._update_drillhole_colors(
                    property_name=field,
                    colormap="turbo",
                    color_mode="Assay",
                    custom_colors=None
                )

                # Update legend
                metadata = self.viewer_widget.renderer.get_drillhole_legend_metadata()
                if metadata:
                    self._update_legend_from_drillholes(metadata)

                # ✅ SYNC: Update Property Panel to match Drillhole Control Panel
                self._sync_property_panel_from_drillhole_control("Assay", field, "turbo")

                logger.info(f"Fast assay field change to '{field}'")

    def _on_drillhole_show_ids_toggled(self, show: bool) -> None:
        """Toggle drillhole ID labels."""
        if self.viewer_widget and self.viewer_widget.renderer:
            self.viewer_widget.renderer.set_drillhole_labels_visible(show)

    def _on_drillhole_visibility_changed(self, hole_id: str, visible: bool) -> None:
        """Toggle individual hole visibility (instant, no re-render)."""
        if self.viewer_widget and self.viewer_widget.renderer:
            self.viewer_widget.renderer.set_drillhole_visibility(hole_id, visible)

    def _on_drillhole_focus_requested(self) -> None:
        """Focus camera on selected drillholes."""
        if self.viewer_widget and self.viewer_widget.renderer and self.drillhole_control_panel:
            visible_holes = self.drillhole_control_panel.get_visible_holes()
            if visible_holes:
                self.viewer_widget.renderer.focus_on_selected_drillholes(visible_holes)

    def _on_drillhole_lith_filter_changed(self, lith_codes: List[str]) -> None:
        """Handle lithology filter changes - re-render drillholes with filter applied."""
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return

        # Check if drillholes are currently rendered
        if "drillholes" not in self.viewer_widget.renderer.active_layers:
            logger.debug("Lithology filter changed but no drillholes rendered - will apply on next plot")
            return

        # Get current dataset and re-render with the filter
        dataset_name = self.drillhole_control_panel.get_dataset() if self.drillhole_control_panel else "Raw Assays"
        logger.info(f"Re-rendering drillholes with lithology filter: {lith_codes if lith_codes else 'ALL'}")

        # Trigger re-plot with new filter
        self._on_drillhole_control_plot(dataset_name)

    def _on_drillhole_info_focus_requested(self, hole_id: str, depth_from: float, depth_to: float) -> None:
        """Focus camera on specific drillhole interval."""
        if self.viewer_widget and self.viewer_widget.renderer:
            self.viewer_widget.renderer.focus_on_drillhole_interval(hole_id, depth_from, depth_to)

    def _on_drillhole_interval_selected(self, interval_data: Dict[str, Any]) -> None:
        """Handle drillhole interval selection."""
        # Info panel removed - no action needed
        pass

    def _on_drillhole_box_selection_completed(self, bounds: tuple) -> None:
        """Handle box selection completion - select drillholes within bounds."""
        # Box selection removed - this method kept for compatibility but does nothing
        pass


    def _start_drillhole_render_status(self) -> None:
        """Start status bar timer for drillhole rendering."""
        self._drillhole_render_start = time.perf_counter()
        if self._drillhole_render_timer is None:
            self._drillhole_render_timer = QTimer(self)
            self._drillhole_render_timer.setInterval(250)
            self._drillhole_render_timer.timeout.connect(self._update_drillhole_render_status)
        self._drillhole_render_timer.start()
        self._update_status_progress("Rendering drillholes...", 0.0)

    def _update_drillhole_render_status(self) -> None:
        """Update status bar text while drillholes are rendering."""
        if self._drillhole_render_timer and self._drillhole_render_timer.isActive():
            elapsed = time.perf_counter() - self._drillhole_render_start
            if self.status is None or not self.status.is_progress_visible():
                if self.status is not None:
                    self.status.show_message(f"Rendering drillholes... {elapsed:.1f}s")

    def _stop_drillhole_render_status(self, success: bool) -> None:
        """Stop the drillhole render timer and show final status."""
        if self._drillhole_render_timer:
            self._drillhole_render_timer.stop()
        elapsed = time.perf_counter() - self._drillhole_render_start
        if success:
            self._finish_status_progress(f"Drillholes rendered ({elapsed:.1f}s)", 4000)
        else:
            self._finish_status_progress("Drillhole rendering failed", 4000)

    def _update_legend_from_drillholes(self, metadata: Dict[str, Any]) -> None:
        """Broadcast legend metadata to the LegendManager."""
        # Skip if no property assigned (no colors during loading)
        if not metadata or metadata.get("property") is None or metadata.get("mode") is None:
            logger.info("Skipping legend update - no colors assigned during drillhole loading")
            return

        categories = metadata.get("categories")
        cats_count = len(categories) if categories else 0
        cats_preview = categories[:5] if categories else []
        logger.info(
            f"[LEGEND DEBUG] _update_legend_from_drillholes called with mode={metadata.get('mode')}, "
            f"{cats_count} categories: {cats_preview}..."
        )
        if self.controller and hasattr(self.controller, "legend_manager"):
            legend_manager = self.controller.legend_manager
            if legend_manager:
                # Use the appropriate update method based on mode
                if metadata.get("mode") == "discrete":
                    categories = metadata.get("categories", [])
                    category_colors = metadata.get("category_colors", {})
                    colormap = metadata.get("colormap", "tab10")

                    # DEBUG: Log received category_colors
                    logger.debug("[LEGEND DEBUG] MainWindow received category_colors sample:")
                    for cat, color_val in list(category_colors.items())[:3]:
                        logger.debug(f"[LEGEND DEBUG]   {cat}: {color_val} (type: {type(color_val)})")

                    # Convert category_colors from hex strings or RGB tuples to RGBA tuples
                    converted_colors = {}
                    for cat, color_val in category_colors.items():
                        if isinstance(color_val, str):
                            # Hex string like "#ff0000"
                            from PyQt6.QtGui import QColor
                            qc = QColor(color_val)
                            converted_colors[cat] = (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
                        elif isinstance(color_val, (tuple, list)):
                            if len(color_val) == 3:
                                # RGB tuple - add alpha
                                converted_colors[cat] = (*color_val, 1.0)
                            elif len(color_val) == 4:
                                # Already RGBA
                                converted_colors[cat] = tuple(color_val)
                        else:
                            # Fallback
                            converted_colors[cat] = (0.5, 0.5, 0.5, 1.0)

                    # DEBUG: Log converted colors before sending to legend_manager
                    logger.debug("[LEGEND DEBUG] MainWindow sending to legend_manager converted_colors sample:")
                    for cat, rgba in list(converted_colors.items())[:3]:
                        logger.debug(f"[LEGEND DEBUG]   {cat}: {rgba}")

                    property_name = metadata.get("property")
                    if property_name:
                        legend_manager.update_discrete(
                            property_name=property_name,
                            categories=categories,
                            cmap_name=colormap,
                            category_colors=converted_colors
                        )
                else:
                    # Continuous mode - USE SAME VALUES AS ACTORS
                    # Get vmin/vmax from metadata (these match the clim values used for actors)
                    vmin = metadata.get("vmin")
                    vmax = metadata.get("vmax")

                    property_name = metadata.get("property")
                    if property_name and vmin is not None and vmax is not None:
                        # Use the exact same range as the actors
                        import numpy as np
                        legend_data = np.array([vmin, vmax], dtype=np.float32)
                        legend_manager.update_continuous(
                            property_name=property_name,
                            data=legend_data,
                            cmap_name=metadata.get("colormap", "viridis")
                        )
                        logger.info(f"Updated legend from metadata: [{vmin:.4f}, {vmax:.4f}]")
                    else:
                        # Fallback: recalculate from data
                        if self.viewer_widget and self.viewer_widget.renderer:
                            if "drillholes" in self.viewer_widget.renderer.active_layers:
                                layer_data = self.viewer_widget.renderer.active_layers["drillholes"].get("data", {})
                                hole_segment_assay = layer_data.get("hole_segment_assay", {})
                                if hole_segment_assay:
                                    import numpy as np
                                    assay_data = np.concatenate([vals for vals in hole_segment_assay.values()])
                                    property_name = metadata.get("property")
                                    if property_name:
                                        legend_manager.update_continuous(
                                            property_name=property_name,
                                            data=assay_data,
                                            cmap_name=metadata.get("colormap", "viridis")
                                        )
                                        logger.info(f"Updated legend from recalculated data: [{np.min(assay_data):.4f}, {np.max(assay_data):.4f}]")
                # Also update via legacy method for compatibility (only if property exists)
                property_name = metadata.get("property")
                if property_name:
                    legend_manager.update_from_property(property_name, metadata)

    def _activate_scene_inspector_legend(self) -> None:
        """Ensure the scene inspector legend toggle indicates visibility."""
        panel = self.scene_inspector_panel
        if panel is None:
            return
        try:
            with QSignalBlocker(panel.toggle_scalar_bar):
                panel.toggle_scalar_bar.setChecked(True)
        except Exception:
            pass
        try:
            panel.scalar_bar_toggled.emit(True)
        except Exception:
            pass

    def _update_property_panel_for_drillholes(self, dataset_name: str, color_mode: str, legend_metadata: Dict[str, Any]) -> None:
        """Update property panel to show drillhole layer and properties."""
        if not self.property_panel:
            return

        try:
            # Block signals on active_layer_combo to prevent _on_active_layer_changed from clearing property combo
            if hasattr(self.property_panel, 'active_layer_combo') and self.property_panel.active_layer_combo:
                self.property_panel.active_layer_combo.blockSignals(True)

            # Update layer controls to include drillholes
            self.property_panel.update_layer_controls()

            # Set drillholes as active layer
            if hasattr(self.property_panel, 'active_layer_combo') and self.property_panel.active_layer_combo:
                drillhole_layer_name = "drillholes"
                index = self.property_panel.active_layer_combo.findText(drillhole_layer_name)
                if index >= 0:
                    self.property_panel.active_layer_combo.setCurrentIndex(index)
                else:
                    self.property_panel.active_layer_combo.addItem(drillhole_layer_name)
                    self.property_panel.active_layer_combo.setCurrentText(drillhole_layer_name)

            # Check if legend_metadata has valid data (not None values)
            # If no colors were assigned during loading, skip property/colormap updates
            if legend_metadata.get("property") is None or legend_metadata.get("scalar_name") is None:
                logger.info("Skipping property/colormap updates - no colors assigned during drillhole loading")
                # Just set the active layer, but don't set properties
                if hasattr(self.property_panel, 'active_layer_combo') and self.property_panel.active_layer_combo:
                    self.property_panel.active_layer_combo.blockSignals(False)
                return

            # Determine scalars and color mode
            scalar_name = legend_metadata.get("scalar_name", "lith_id" if color_mode == "Lithology" else "assay")
            mode_text = "Discrete" if color_mode == "Lithology" else "Continuous"

            # Set color mode first so property updates use correct mode
            if hasattr(self.property_panel, 'color_mode_combo') and self.property_panel.color_mode_combo:
                self.property_panel.color_mode_combo.blockSignals(True)
                self.property_panel.color_mode_combo.setCurrentText(mode_text)
                self.property_panel.color_mode_combo.blockSignals(False)

            # Update property combo - let property panel populate all available properties
            # Then set the current property from legend metadata
            if hasattr(self.property_panel, 'property_combo') and self.property_panel.property_combo:
                # Trigger property panel to populate all available properties for this layer
                # Get layer_data from renderer's active_layers
                layer_data = None
                if hasattr(self, 'viewer_widget') and self.viewer_widget and hasattr(self.viewer_widget, 'renderer'):
                    layer_info = self.viewer_widget.renderer.active_layers.get("drillholes", {})
                    layer_data = layer_info.get('data')

                # Call the public API instead of the non-existent private method
                self.property_panel.set_active_layer("drillholes", layer_data)

                # Set the current property from legend_metadata (actual element name like "Cu", "Au")
                current_property = legend_metadata.get("property", scalar_name)
                if current_property:
                    self.property_panel.property_combo.blockSignals(True)
                    index = self.property_panel.property_combo.findText(current_property)
                    if index >= 0:
                        self.property_panel.property_combo.setCurrentIndex(index)
                    else:
                        # Property not in list - add it
                        self.property_panel.property_combo.addItem(current_property)
                        self.property_panel.property_combo.setCurrentText(current_property)
                    self.property_panel.property_combo.blockSignals(False)

            # Unblock active_layer_combo signals now that property is set
            if hasattr(self.property_panel, 'active_layer_combo') and self.property_panel.active_layer_combo:
                self.property_panel.active_layer_combo.blockSignals(False)

            # Update colormap dropdown
            if hasattr(self.property_panel, 'colormap_combo') and self.property_panel.colormap_combo:
                colormap = legend_metadata.get("colormap", "viridis")
                if colormap:  # Only update if colormap is not None
                    index = self.property_panel.colormap_combo.findText(colormap)
                    if index >= 0:
                        self.property_panel.colormap_combo.blockSignals(True)
                        self.property_panel.colormap_combo.setCurrentIndex(index)
                        self.property_panel.colormap_combo.blockSignals(False)

            # NOTE: Do NOT call _on_property_changed here - it would regenerate colors from the colormap
            # instead of keeping the original HSV-generated colors from the drillhole layer.
            # The renderer already has the correct colors and legend metadata.

            logger.info(f"Updated property panel for drillholes: {dataset_name}, color_mode={color_mode}, property={scalar_name}")
        except Exception as e:
            logger.error(f"Failed to update property panel for drillholes: {e}", exc_info=True)

    def _sync_property_panel_from_drillhole_control(self, color_mode: str, property_name: str, colormap: str) -> None:
        """Sync Property Panel when Drillhole Control Panel changes."""
        if not self.property_panel or not hasattr(self.property_panel, 'active_layer_combo'):
            return

        # Only sync if drillholes layer is active
        current_layer = self.property_panel.active_layer_combo.currentText()
        if current_layer != "drillholes":
            return

        try:
            # Update color mode
            if hasattr(self.property_panel, 'color_mode_combo') and self.property_panel.color_mode_combo:
                mode_text = "Discrete" if color_mode == "Lithology" else "Continuous"
                with self.property_panel._block_signal(self.property_panel.color_mode_combo):
                    self.property_panel.color_mode_combo.setCurrentText(mode_text)

            # Update property
            if hasattr(self.property_panel, 'property_combo') and self.property_panel.property_combo:
                with self.property_panel._block_signal(self.property_panel.property_combo):
                    # Check if property exists, if not add it
                    index = self.property_panel.property_combo.findText(property_name)
                    if index < 0:
                        self.property_panel.property_combo.addItem(property_name)
                    self.property_panel.property_combo.setCurrentText(property_name)

            # Update colormap
            if hasattr(self.property_panel, 'colormap_combo') and self.property_panel.colormap_combo:
                with self.property_panel._block_signal(self.property_panel.colormap_combo):
                    index = self.property_panel.colormap_combo.findText(colormap)
                    if index >= 0:
                        self.property_panel.colormap_combo.setCurrentIndex(index)

            logger.debug(f"Synced Property Panel from Drillhole Control: mode={color_mode}, property={property_name}, colormap={colormap}")
        except Exception as e:
            logger.warning(f"Failed to sync Property Panel from Drillhole Control: {e}")

    def _sync_drillhole_control_from_property_panel(self, property_name: str, color_mode: str) -> None:
        """Sync Drillhole Control Panel when Property Panel changes drillhole properties."""
        if not self.drillhole_control_panel:
            return

        try:
            # Update color mode in Drillhole Control Panel
            if color_mode == "Lithology":
                if hasattr(self.drillhole_control_panel, 'color_mode_combo'):
                    with QSignalBlocker(self.drillhole_control_panel.color_mode_combo):
                        self.drillhole_control_panel.color_mode_combo.setCurrentText("Lithology")
            else:  # Assay mode
                if hasattr(self.drillhole_control_panel, 'color_mode_combo'):
                    with QSignalBlocker(self.drillhole_control_panel.color_mode_combo):
                        self.drillhole_control_panel.color_mode_combo.setCurrentText("Assay")

                # Update assay field if in assay mode
                if hasattr(self.drillhole_control_panel, 'assay_field_combo'):
                    with QSignalBlocker(self.drillhole_control_panel.assay_field_combo):
                        index = self.drillhole_control_panel.assay_field_combo.findText(property_name)
                        if index >= 0:
                            self.drillhole_control_panel.assay_field_combo.setCurrentIndex(index)
                        elif property_name and property_name != "Lithology":
                            # Add property if it doesn't exist
                            self.drillhole_control_panel.assay_field_combo.addItem(property_name)
                            self.drillhole_control_panel.assay_field_combo.setCurrentText(property_name)

            logger.debug(f"Synced Drillhole Control Panel from Property Panel: property={property_name}, mode={color_mode}")
        except Exception as e:
            logger.warning(f"Failed to sync Drillhole Control Panel from Property Panel: {e}")

    # =========================================================================
    # Geological Explorer Panel Signal Handlers
    # =========================================================================

    def _on_geology_render_mode_changed(self, mode: str) -> None:
        """Handle geology render mode change from Geological Explorer panel."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.set_geology_render_mode(mode)
                logger.info(f"Geology render mode changed to: {mode}")
            except Exception as e:
                logger.error(f"Failed to change geology render mode: {e}")

    def _on_geology_contacts_visibility_changed(self, visible: bool) -> None:
        """Handle geology contacts visibility toggle."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.set_geology_contacts_visible(visible)
            except Exception as e:
                logger.debug(f"Could not toggle contacts visibility: {e}")

    def _on_geology_surfaces_visibility_changed(self, visible: bool) -> None:
        """Handle geology surfaces visibility toggle."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.set_geology_surfaces_visible(visible)
            except Exception as e:
                logger.debug(f"Could not toggle surfaces visibility: {e}")

    def _on_geology_misfit_visibility_changed(self, visible: bool) -> None:
        """Handle geology misfit visibility toggle."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.set_geology_misfit_visible(visible)
            except Exception as e:
                logger.debug(f"Could not toggle misfit visibility: {e}")

    def _on_geology_formation_filter_changed(self, formations: list) -> None:
        """Handle geology formation filter change."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.filter_geology_formations(formations)
            except Exception as e:
                logger.debug(f"Could not filter formations: {e}")

    def _on_geology_opacity_changed(self, opacity: float) -> None:
        """Handle geology opacity change."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.set_geology_opacity(opacity)
            except Exception as e:
                logger.debug(f"Could not set geology opacity: {e}")

    def _on_geology_color_palette_changed(self, palette: str) -> None:
        """Handle geology color palette change."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.set_geology_color_palette(palette)
            except Exception as e:
                logger.debug(f"Could not set color palette: {e}")

    def _on_geology_reset_view(self) -> None:
        """Handle geology reset view request."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.fit_to_geology()
            except Exception as e:
                logger.debug(f"Could not reset geology view: {e}")

    def _on_geology_clear(self) -> None:
        """Handle geology clear request."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.clear_geology()
            except Exception as e:
                logger.debug(f"Could not clear geology: {e}")

    def _on_geology_layer_visibility_changed(self, layer_name: str, visible: bool) -> None:
        """Handle individual geology layer visibility toggle."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.set_geology_layer_visibility(layer_name, visible)
            except Exception as e:
                logger.debug(f"Could not toggle layer '{layer_name}' visibility: {e}")

    def _on_geology_wireframe_toggled(self, enabled: bool) -> None:
        """Handle wireframe toggle for solid domains."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.set_wireframe_visible(enabled)
            except Exception as e:
                logger.debug(f"Could not toggle wireframe: {e}")

    def _on_geology_solids_opacity_changed(self, opacity: float) -> None:
        """Handle solid domains opacity change."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.set_solids_opacity(opacity)
            except Exception as e:
                logger.debug(f"Could not set solids opacity: {e}")

    def _on_geology_view_mode_changed(self, mode: str) -> None:
        """Handle view mode preset change (surfaces_only, solids_only, etc.)."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                self.viewer_widget.renderer.apply_view_mode(mode)
                logger.info(f"Geology view mode changed to: {mode}")
            except Exception as e:
                logger.debug(f"Could not apply view mode: {e}")

    def _on_geology_package_ready(self, package: dict) -> None:
        """
        Handle geology package ready from LoopStructural panel.

        Updates the Geological Explorer panel and shows it.
        """
        try:
            # Update Geological Explorer panel with new data
            if hasattr(self, 'geological_explorer_panel') and self.geological_explorer_panel:
                self.geological_explorer_panel.on_geology_package_loaded(package)

                # Show the dock if hidden
                if hasattr(self, 'geological_explorer_dock') and self.geological_explorer_dock:
                    self.geological_explorer_dock.show()
                    self.geological_explorer_dock.raise_()

                logger.info("Geological Explorer panel updated with new geology package")
        except Exception as e:
            logger.error(f"Failed to update Geological Explorer panel: {e}", exc_info=True)

    def visualize_block_model(self, block_grid, property_name: str, method_name: str = "Block Model"):
        """
        Visualize block model in the main 3D viewer using the unified layer system.
        
        Parameters
        ----------
        block_grid : pv.RectilinearGrid
            PyVista grid containing block model
        property_name : str
            Property name to visualize
        method_name : str
            Method name (e.g., "Ordinary Kriging", "Simple Kriging") for layer naming
        """
        try:
            logger.info(f"BLOCK MODEL VIS: Received visualization request: property='{property_name}', grid has {block_grid.n_cells if block_grid else 0} cells")

            # Validate grid
            if block_grid is None:
                logger.error("BLOCK MODEL VIS: grid is None")
                QMessageBox.warning(
                    self,
                    "Visualization Error",
                    "Cannot visualize: grid data is None."
                )
                return

            if block_grid.n_cells == 0:
                logger.warning("BLOCK MODEL VIS: Grid has 0 cells")
                QMessageBox.warning(self, "Empty Grid", "Block model grid has 0 cells.")
                return

            # RectilinearGrid stores properties in cell_data (industry standard)
            # Find property in grid (with case-insensitive matching)
            property_name_actual = None
            available_props = list(block_grid.cell_data.keys()) if hasattr(block_grid, 'cell_data') else []

            logger.info(f"BLOCK MODEL VIS: Looking for property '{property_name}' in grid. Available properties: {available_props}")

            if property_name in block_grid.cell_data:
                property_name_actual = property_name
            else:
                # Try case-insensitive match
                property_upper = property_name.upper().replace(" ", "_")
                for prop in available_props:
                    prop_normalized = prop.upper().replace(" ", "_")
                    if prop_normalized == property_upper:
                        property_name_actual = prop
                        break

                if property_name_actual is None:
                    QMessageBox.warning(
                        self,
                        "Property Not Found",
                        f"Property '{property_name}' not found in block model.\n\n"
                        f"Available properties: {', '.join(available_props)}"
                    )
                    return

            # Get property data
            data = block_grid.cell_data[property_name_actual]
            valid_mask = np.isfinite(data)

            if not valid_mask.any():
                logger.warning(f"BLOCK MODEL VIS: All values in '{property_name_actual}' are NaN or infinite")
                QMessageBox.warning(self, "No Valid Data", f"All values in '{property_name_actual}' are NaN or infinite.")
                return

            valid_data = data[valid_mask]
            data_min = float(np.nanmin(valid_data))
            data_max = float(np.nanmax(valid_data))

            logger.info(f"BLOCK MODEL VIS: Property '{property_name_actual}' data range: [{data_min:.3f}, {data_max:.3f}], valid points: {valid_mask.sum()}/{len(data)}")

            # Create layer name using method name
            layer_name = f"{method_name}: {property_name_actual}"

            # Remove existing layer with same name
            if self.viewer_widget and self.viewer_widget.renderer:
                if layer_name in self.viewer_widget.renderer.active_layers:
                    logger.info(f"BLOCK MODEL VIS: Removing existing layer '{layer_name}'")
                    self.viewer_widget.renderer.clear_layer(layer_name)

            # Add block model layer using the unified layer system
            # This DOES NOT remove drillholes or kriging layers
            logger.info(f"BLOCK MODEL VIS: Adding layer '{layer_name}' with property '{property_name_actual}'")
            self.viewer_widget.renderer.add_block_model_layer(block_grid, property_name_actual, layer_name=layer_name)

            # Refresh Interactive Slicer panel if open
            self._refresh_slicer_panel()

            # Update legend via LegendManager
            if hasattr(self.viewer_widget.renderer, 'legend_manager') and self.viewer_widget.renderer.legend_manager is not None:
                try:
                    logger.info(f"BLOCK MODEL VIS: Updating legend for property '{property_name_actual}', range=[{data_min:.3f}, {data_max:.3f}]")
                    self.viewer_widget.renderer.legend_manager.set_continuous(
                        field=property_name_actual.upper(),
                        vmin=data_min,
                        vmax=data_max,
                        cmap_name='viridis'
                    )
                    logger.info("BLOCK MODEL VIS: Legend updated successfully")
                except Exception as e:
                    logger.warning(f"BLOCK MODEL VIS: Could not update legend: {e}", exc_info=True)

            # Reset camera to show all (skip for large models to prevent GPU timeout)
            # IMPORTANT: Align with GPU-safe mode threshold (30,000) in renderer.py
            CAMERA_RESET_THRESHOLD = 30000
            if self.viewer_widget and self.viewer_widget.renderer and self.viewer_widget.renderer.plotter:
                # DEBUG: Log grid bounds AFTER coordinate transformation
                transformed_bounds = block_grid.bounds
                logger.info(f"DEBUG BLOCK VIS: Grid bounds AFTER transform: {transformed_bounds}")
                logger.info(f"DEBUG BLOCK VIS: Grid n_cells: {block_grid.n_cells}")
                logger.info(f"DEBUG BLOCK VIS: Active layers: {list(self.viewer_widget.renderer.active_layers.keys())}")

                if block_grid.n_cells <= CAMERA_RESET_THRESHOLD:
                    # Get camera position before reset
                    cam_before = self.viewer_widget.renderer.plotter.camera_position
                    logger.info(f"DEBUG BLOCK VIS: Camera position BEFORE reset: {cam_before}")

                    self.viewer_widget.renderer.plotter.reset_camera()

                    # Get camera position after reset
                    cam_after = self.viewer_widget.renderer.plotter.camera_position
                    logger.info(f"DEBUG BLOCK VIS: Camera position AFTER reset: {cam_after}")
                    logger.info("BLOCK MODEL VIS: Camera reset")
                else:
                    logger.info(f"BLOCK MODEL VIS: Skipped camera reset for large model ({block_grid.n_cells:,} cells) to prevent GPU timeout")

            # Update property panel to show the new layer
            if hasattr(self, 'property_panel') and self.property_panel:
                try:
                    self.property_panel.update_layer_controls()
                    logger.info(f"BLOCK MODEL VIS: Updated property panel with new layer: {layer_name}")
                except AttributeError:
                    # Try fallback method name if it exists
                    if hasattr(self.property_panel, 'update_active_layers'):
                        self.property_panel.update_active_layers()
                        logger.info("BLOCK MODEL VIS: Used fallback update_active_layers method")

            # Log active layers after registration
            logger.info(f"BLOCK MODEL VIS: Active layers now: {list(self.viewer_widget.renderer.active_layers.keys())}")

            self.status_bar.showMessage(
                f"Block model added: {property_name_actual} ({block_grid.n_cells} blocks, "
                f"range=[{data_min:.2f}, {data_max:.2f}]) | Use Layer Controls to toggle visibility",
                5000
            )
            logger.info(f"BLOCK MODEL VIS: Successfully visualized block model: {property_name_actual}, bounds={block_grid.bounds}")

        except Exception as e:
            logger.error(f"BLOCK MODEL VIS: Error visualizing block model: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Error visualizing block model:\n{str(e)}\n\nCheck console logs for details."
            )

    def on_visualize_mining_schedule(self, schedule_df):
        """Visualize mining schedule in 3D viewer by coloring blocks by period."""
        if self.current_model is None:
            QMessageBox.warning(
                self,
                "No Block Model",
                "Please load a block model first before visualizing the mining schedule."
            )
            return

        try:
            logger.info(f"Visualizing mining schedule with {len(schedule_df)} entries")

            # Create BLOCK_ID in current model if it doesn't exist
            model_df = self.current_model.to_dataframe()
            if 'BLOCK_ID' not in model_df.columns:
                self.current_model.add_property('BLOCK_ID', np.arange(len(model_df)))
                model_df = self.current_model.to_dataframe()

            # Merge schedule with model
            merged = model_df.merge(schedule_df[['BLOCK_ID', 'PERIOD', 'MINED']], on='BLOCK_ID', how='left', validate='one_to_many')

            # Fill NaN values (blocks not in schedule) with -1 for period and 0 for mined
            merged['PERIOD'] = merged['PERIOD'].fillna(-1).astype(int)
            merged['MINED'] = merged['MINED'].fillna(0).astype(int)

            # Add the PERIOD property to the block model
            self.current_model.add_property('MINING_PERIOD', merged['PERIOD'].values)

            # Count mined blocks
            mined_blocks = merged[merged['MINED'] == 1]

            if len(mined_blocks) == 0:
                QMessageBox.information(
                    self,
                    "No Mined Blocks",
                    "The mining schedule doesn't contain any mined blocks to visualize."
                )
                return

            # Update the visualization to show the new property
            self.viewer_widget.renderer.load_block_model(self.current_model)

            # Color by mining period
            self.viewer_widget.set_property_coloring('MINING_PERIOD')

            # Fit to view
            self.viewer_widget.fit_to_view()

            num_periods = int(mined_blocks['PERIOD'].max() + 1)

            QMessageBox.information(
                self,
                "Schedule Visualized",
                f"Mining schedule visualized successfully!\n\n"
                f"ðŸ“Š Summary:\n"
                f"â€¢ Total blocks mined: {len(mined_blocks):,}\n"
                f"â€¢ Number of periods: {num_periods}\n"
                f"â€¢ Blocks per period (avg): {len(mined_blocks)/num_periods:,.0f}\n"
                f"â€¢ Unmined blocks: {len(merged[merged['MINED'] == 0]):,}\n\n"
                f"ðŸŽ¨ Visualization:\n"
                f"â€¢ Colored by mining period\n"
                f"â€¢ Period -1 (gray) = unmined\n"
                f"â€¢ Period 0-{num_periods-1} = mining sequence"
            )

            logger.info(f"Visualized mining schedule: {len(mined_blocks)} blocks, {num_periods} periods")

        except Exception as e:
            logger.error(f"Failed to visualize mining schedule: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Failed to visualize mining schedule:\n{str(e)}"
            )

    def on_visualize_ultimate_pit(self, pit_df):
        """Visualize ultimate pit in 3D viewer by merging pit results with block model."""
        if self.current_model is None:
            QMessageBox.warning(
                self,
                "No Block Model",
                "Please load a block model first before visualizing the ultimate pit."
            )
            return

        try:
            logger.info(f"Visualizing ultimate pit with {len(pit_df)} blocks")

            # Get current model DataFrame
            model_df = self.current_model.to_dataframe()

            # Ensure BLOCK_ID exists in both
            if 'BLOCK_ID' not in model_df.columns:
                self.current_model.add_property('BLOCK_ID', np.arange(len(model_df)))
                model_df = self.current_model.to_dataframe()

            if 'BLOCK_ID' not in pit_df.columns:
                pit_df = pit_df.copy()
                pit_df['BLOCK_ID'] = np.arange(len(pit_df))

            # Merge pit results with current model
            # Select columns from pit_df that we want to visualize
            pit_columns = ['BLOCK_ID', 'IN_PIT', 'VALUE']
            if 'SHELL_ID' in pit_df.columns:
                pit_columns.append('SHELL_ID')

            merged = model_df.merge(pit_df[pit_columns], on='BLOCK_ID', how='left', validate='one_to_one')

            # Fill NaN values for blocks not in pit
            merged['IN_PIT'] = merged['IN_PIT'].fillna(0).astype(int)
            merged['VALUE'] = merged['VALUE'].fillna(0)
            if 'SHELL_ID' in merged.columns:
                merged['SHELL_ID'] = merged['SHELL_ID'].fillna(0).astype(int)

            # Add properties to block model
            self.current_model.add_property('IN_PIT', merged['IN_PIT'].values)
            self.current_model.add_property('PIT_VALUE', merged['VALUE'].values)
            if 'SHELL_ID' in merged.columns:
                self.current_model.add_property('SHELL_ID', merged['SHELL_ID'].values)

            # Count pit blocks
            pit_blocks = merged[merged['IN_PIT'] == 1]

            if len(pit_blocks) == 0:
                QMessageBox.information(
                    self,
                    "No Pit Blocks",
                    "The ultimate pit calculation found no economic blocks."
                )
                return

            # Update the visualization
            self.viewer_widget.renderer.load_block_model(self.current_model)

            # Color by IN_PIT status
            self.viewer_widget.set_property_coloring('IN_PIT')

            # Fit to view
            self.viewer_widget.fit_to_view()

            # Build summary message
            total_value = pit_blocks['VALUE'].sum()
            avg_grade = pit_blocks['GRADE'].mean() if 'GRADE' in pit_blocks.columns else 0

            msg = "Ultimate pit visualized successfully!\n\n"
            msg += "ðŸ“Š Pit Summary:\n"
            msg += f"â€¢ Blocks in pit: {len(pit_blocks):,}\n"
            msg += f"â€¢ Blocks outside pit: {len(merged) - len(pit_blocks):,}\n"
            msg += f"â€¢ Total pit value: ${total_value:,.2f}\n"
            if avg_grade > 0:
                msg += f"â€¢ Average grade: {avg_grade:.2f} g/t\n"
            msg += "\nðŸŽ¨ Visualization:\n"
            msg += "â€¢ Colored by IN_PIT property\n"
            msg += "â€¢ 1 (colored) = in pit\n"
            msg += "â€¢ 0 (gray/hidden) = outside pit\n"

            if 'SHELL_ID' in merged.columns:
                num_shells = int(pit_blocks['SHELL_ID'].max())
                if num_shells > 0:
                    msg += "\nðŸŽ¯ Pit Shells:\n"
                    msg += f"â€¢ Number of shells: {num_shells}\n"
                    msg += "â€¢ Switch to 'SHELL_ID' property to see nested shells"

            QMessageBox.information(
                self,
                "Ultimate Pit Visualized",
                msg
            )

            logger.info(f"Visualized ultimate pit: {len(pit_blocks)} blocks")

        except Exception as e:
            logger.error(f"Failed to visualize ultimate pit: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Failed to visualize ultimate pit:\n{str(e)}"
            )

    def on_block_resource_highlight(self, mask):
        """Handle block resource highlight request."""
        if self.viewer_widget and self.viewer_widget.renderer:
            try:
                # Get the grid
                if 'unstructured_grid' in self.viewer_widget.renderer.block_meshes:
                    grid = self.viewer_widget.renderer.block_meshes['unstructured_grid']

                    # Extract selected blocks
                    selected_grid = grid.extract_cells(mask)

                    # Remove old highlight
                    if 'resource_highlight' in self.viewer_widget.renderer.block_meshes:
                        self.viewer_widget.plotter.remove_actor('resource_highlight')

                    # Add highlight
                    self.viewer_widget.plotter.add_mesh(
                        selected_grid,
                        color='lime',
                        opacity=0.7,
                        show_edges=True,
                        edge_color='darkgreen',
                        line_width=2.0,
                        name='resource_highlight',
                        pickable=False,
                        show_scalar_bar=False
                    )

                    self.viewer_widget.plotter.render()
                    logger.info("Highlighted resource blocks in 3D")

            except Exception as e:
                logger.error(f"Error highlighting resource blocks: {e}")

    def on_visualize_resource_classification(self, df: pd.DataFrame, color_map: dict):
        """
        Handle resource classification visualization request.
        
        NOTE: This method updates an existing grid with classification data.
        It modifies the existing 'unstructured_grid' rather than creating a new layer.
        """
        if not self.viewer_widget or not self.viewer_widget.renderer:
            logger.warning("RESOURCE CLASS VIS: Renderer not available")
            return

        try:

            logger.info("RESOURCE CLASS VIS: Received resource classification visualization request")

            # Get the grid
            if 'unstructured_grid' not in self.viewer_widget.renderer.block_meshes:
                logger.warning("RESOURCE CLASS VIS: No unstructured_grid found in renderer")
                QMessageBox.warning(
                    self,
                    "No Block Model",
                    "No block model grid found in viewer.\n\n"
                    "Please visualize a block model first before applying classification."
                )
                return

            grid = self.viewer_widget.renderer.block_meshes['unstructured_grid']
            logger.info(f"RESOURCE CLASS VIS: Found grid with {grid.n_cells} cells")

            # Add CLASS data to grid if not present
            if 'CLASS' not in grid.cell_data:
                if 'CLASS' in df.columns:
                    # Map class names to numeric values for PyVista
                    class_to_num = {
                        'Measured': 0,
                        'Indicated': 1,
                        'Inferred': 2,
                        'Unclassified': 3
                    }

                    class_numeric = df['CLASS'].map(class_to_num).fillna(3).astype(int).values

                    # Ensure length matches
                    if len(class_numeric) == grid.n_cells:
                        grid.cell_data['CLASS'] = class_numeric
                        logger.info(f"RESOURCE CLASS VIS: Added CLASS property to grid ({len(class_numeric)} values)")
                    else:
                        logger.error(f"RESOURCE CLASS VIS: Class data length {len(class_numeric)} != grid cells {grid.n_cells}")
                        QMessageBox.warning(
                            self,
                            "Size Mismatch",
                            f"Classification data ({len(class_numeric)} blocks) doesn't match grid ({grid.n_cells} cells)."
                        )
                        return
                else:
                    logger.warning("RESOURCE CLASS VIS: No 'CLASS' column in DataFrame")
                    return

            # Remove existing block actor
            try:
                self.viewer_widget.plotter.remove_actor('unstructured_grid')
                logger.info("RESOURCE CLASS VIS: Removed existing unstructured_grid actor")
            except Exception as e:
                logger.debug(f"RESOURCE CLASS VIS: Could not remove existing actor (may not exist): {e}")

            # Create color map for PyVista (RGB tuples)
            pv_cmap = [
                tuple(int(color_map['Measured'].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)),    # green
                tuple(int(color_map['Indicated'].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)),   # yellow
                tuple(int(color_map['Inferred'].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)),    # red
                tuple(int(color_map['Unclassified'].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) # gray
            ]

            # Add mesh with classification colors (use custom LegendWidget instead of PyVista scalar bar)
            actor = self.viewer_widget.plotter.add_mesh(
                grid,
                scalars='CLASS',
                cmap=pv_cmap,
                show_edges=True,
                edge_color='black',
                line_width=0.5,
                opacity=1.0,
                name='unstructured_grid',
                show_scalar_bar=False  # LegendManager handles scalar bars
            )
            logger.info("RESOURCE CLASS VIS: Added mesh with CLASS property")

            # Update legend via LegendManager (categorical)
            if hasattr(self.viewer_widget.renderer, 'legend_manager') and self.viewer_widget.renderer.legend_manager is not None:
                try:
                    categories = ['Measured', 'Indicated', 'Inferred', 'Unclassified']
                    logger.info(f"RESOURCE CLASS VIS: Updating legend with categories: {categories}")
                    self.viewer_widget.renderer.legend_manager.set_categorical(
                        field='CLASS',
                        categories=categories,
                        colormap='Set1'  # Categorical colormap
                    )
                    logger.info("RESOURCE CLASS VIS: Legend updated successfully")
                except Exception as e:
                    logger.warning(f"RESOURCE CLASS VIS: Could not update legend: {e}", exc_info=True)

            # Update property panel
            if hasattr(self, 'property_panel') and self.property_panel:
                try:
                    self.property_panel.update_layer_controls()
                    logger.info("RESOURCE CLASS VIS: Updated property panel")
                except AttributeError:
                    if hasattr(self.property_panel, 'update_active_layers'):
                        self.property_panel.update_active_layers()

            # Refresh Interactive Slicer panel if open
            self._refresh_slicer_panel()

            self.viewer_widget.plotter.render()
            logger.info("RESOURCE CLASS VIS: Successfully visualized resource classification in 3D")

            self.status_bar.showMessage("Resource classification visualized in 3D", 5000)

        except Exception as e:
            logger.error(f"RESOURCE CLASS VIS: Error visualizing classification: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Failed to visualize classification:\n{str(e)}\n\nCheck console logs for details."
            )

    def _ensure_domain_panel(self) -> bool:
        """Ensure the Domain Compositing panel exists and has data loaded."""
        if not hasattr(self, 'domain_compositing_panel') or self.domain_compositing_panel is None:
            # Attempt to open the panel
            try:
                self.open_domain_compositing_panel()
            except Exception:
                pass
        return hasattr(self, 'domain_compositing_panel') and self.domain_compositing_panel is not None

    def _get_clean_drillhole_df(self, prefer_composites: bool = True, require_xyz: bool = False):
        """
        Retrieve cleaned drillhole data (assays/composites) from central sources in priority order:
        1) Current drillhole database (recomputed trajectories from cleaned collars/surveys/assays)
        2) DataRegistry (preferred composites, then assays)
        3) Domain compositing panel (comp_domain_df or assay_xyz)
        """
        def _has_xyz(df) -> bool:
            if not require_xyz or df is None:
                return True
            cols_lower = {c.lower() for c in df.columns}
            return {"x", "y", "z"}.issubset(cols_lower)

        # 1) Current drillhole database (recompute XYZ from cleaned data)
        # (Removed - database panel no longer available)

        # 2) DataRegistry
        try:
            # Use injected registry via controller (dependency injection)
            registry = self.controller.registry if self.controller else None
            if registry is not None:
                reg_data = registry.get_drillhole_data()
                if isinstance(reg_data, dict):
                    assays = reg_data.get("assays")
                    if assays is not None and getattr(assays, "empty", False) is False and _has_xyz(assays):
                        return assays
                elif hasattr(reg_data, "empty") and not reg_data.empty and _has_xyz(reg_data):
                    return reg_data
        except Exception as e:
            logger.warning(f"Failed to read DataRegistry: {e}", exc_info=True)

        # 3) Drillhole loading panel (fallback)
        if self._ensure_domain_panel():
            panel = self.domain_compositing_panel
            if getattr(panel, "assay_df", None) is not None and _has_xyz(panel.assay_df):
                return panel.assay_df

        return None

    def open_cross_section_panel(self):
        """Open Cross-Section Tool panel in a separate window."""
        # Check if panel already exists and is visible
        if hasattr(self, 'cross_section_dialog') and self._is_dialog_valid(self.cross_section_dialog):
            if self.cross_section_dialog.isVisible():
                self.cross_section_dialog.raise_()
                self.cross_section_dialog.activateWindow()
            else:
                # Window exists but is minimized - restore it
                self.cross_section_dialog.show()
                self.cross_section_dialog.raise_()
                self.cross_section_dialog.activateWindow()
            return

        # Try to get mesh from block model first, then drillholes
        mesh = None
        scalar_field = None
        colormap = getattr(self.viewer_widget, 'current_colormap', 'viridis')

        # Check for block model
        if self.current_model and hasattr(self.viewer_widget.renderer, 'block_meshes'):
            block_meshes = self.viewer_widget.renderer.block_meshes
            if 'unstructured_grid' in block_meshes:
                mesh = block_meshes['unstructured_grid']
                # Get current property from viewer_widget or renderer
                scalar_field = getattr(self.viewer_widget, 'current_property', None)
                if scalar_field is None:
                    scalar_field = getattr(self.viewer_widget.renderer, 'current_property', None)
                # If no property set, try to get first property from model
                if scalar_field is None and self.current_model.properties:
                    scalar_field = list(self.current_model.properties.keys())[0]
                logger.info(f"Found block model mesh with {mesh.n_points} points, coloring by '{scalar_field}'")

        # If no block model, check for drillholes
        if mesh is None:
            drillhole_data = getattr(self.viewer_widget.renderer, 'drillhole_data', None)
            if drillhole_data is not None and 'mesh' in drillhole_data:
                mesh = drillhole_data['mesh']
                scalar_field = drillhole_data.get('color_by', None)
                logger.info(f"Found drillhole mesh with {mesh.n_points} points, coloring by '{scalar_field}'")

        # If still no mesh, show warning
        if mesh is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load and visualize data first before using the cross-section tool.\n\n"
                "For block models: File > Load Block Model\n"
                "For drillholes: Load drillhole data and use the viewer"
            )
            return

        # Create new panel window
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        from .cross_section_panel import CrossSectionPanel

        self.cross_section_dialog = QDialog(None)  # No parent - independent window
        self.cross_section_dialog.setWindowTitle("Cross-Section Tool")
        self.cross_section_dialog.resize(400, 600)
        self.cross_section_dialog.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.cross_section_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.cross_section_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Setup dialog persistence (position/size will be saved/restored)
        self._setup_dialog_persistence(self.cross_section_dialog, 'cross_section_dialog')

        layout = QVBoxLayout(self.cross_section_dialog)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create panel
        self.cross_section_panel = CrossSectionPanel(signals=self.signals)
        layout.addWidget(self.cross_section_panel)

        # Set mesh and plotter (works for both block models and drillholes)
        self.cross_section_panel.set_mesh(mesh, scalar_field=scalar_field, colormap=colormap)
        self.cross_section_panel.set_plotter(self.viewer_widget.plotter)

        logger.info("Opened Cross-Section Tool panel in separate window")

        # Show as non-modal dialog
        self.cross_section_dialog.show()

    def _desurvey_drillhole(self, x0: float, y0: float, z0: float, survey_df: pd.DataFrame) -> np.ndarray:
        """
        Desurvey a drillhole using Minimum Curvature algorithm.
        
        Uses the shared desurvey utility to ensure consistent coordinate
        calculations across the entire application.
        
        Args:
            x0, y0, z0: Collar coordinates
            survey_df: Survey data for this hole (DEPTH, AZI, DIP)
            
        Returns:
            Array of 3D points along the drillhole path
        """
        depths, xs, ys, zs = minimum_curvature_desurvey(x0, y0, z0, survey_df)

        if depths is None:
            # Fallback to collar point only
            return np.array([[x0, y0, z0]])

        return np.column_stack([xs, ys, zs])

    def _add_composite_coordinates(self, composites_df: pd.DataFrame, collar_df: pd.DataFrame,
                                   survey_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Add X, Y, Z coordinates to composite intervals.
        
        Calculates the 3D midpoint coordinates for each composite interval using
        collar and survey data (Minimum Curvature). This is required for variogram analysis.
        
        Uses the shared desurvey utility to ensure consistent coordinate
        calculations across the entire application.
        
        Args:
            composites_df: Composite intervals (HOLEID, FROM, TO, grades)
            collar_df: Collar data (HOLEID, X, Y, Z)
            survey_df: Survey data (HOLEID, DEPTH, AZI, DIP) or None
            
        Returns:
            DataFrame with added X, Y, Z columns
        """
        result_df = composites_df.copy()
        x_coords = []
        y_coords = []
        z_coords = []

        for idx, row in composites_df.iterrows():
            hole_id = row['HOLEID']
            from_depth = row['FROM']
            to_depth = row['TO']
            mid_depth = (from_depth + to_depth) / 2.0

            # Get collar for this hole
            collar = collar_df[collar_df['HOLEID'] == hole_id]
            if len(collar) == 0:
                # No collar - use 0,0,0
                x_coords.append(0.0)
                y_coords.append(0.0)
                z_coords.append(0.0)
                continue

            collar = collar.iloc[0]
            x0, y0, z0 = collar['X'], collar['Y'], collar['Z']

            # Desurvey to get 3D path using unified Minimum Curvature
            if survey_df is not None:
                hole_survey = survey_df[survey_df['HOLEID'] == hole_id].copy()
                if len(hole_survey) > 0:
                    depths, xs, ys, zs = minimum_curvature_desurvey(x0, y0, z0, hole_survey)

                    if depths is not None and len(depths) >= 2:
                        # Interpolate position at mid_depth
                        x_mid, y_mid, z_mid = interpolate_at_depth(
                            depths, xs, ys, zs, mid_depth
                        )
                    else:
                        # Not enough points - use collar with vertical assumption
                        x_mid, y_mid, z_mid = x0, y0, z0 - mid_depth
                else:
                    # No survey for this hole - assume vertical
                    x_mid, y_mid, z_mid = x0, y0, z0 - mid_depth
            else:
                # No survey data - assume vertical
                x_mid, y_mid, z_mid = x0, y0, z0 - mid_depth

            x_coords.append(x_mid)
            y_coords.append(y_mid)
            z_coords.append(z_mid)

        result_df['X'] = x_coords
        result_df['Y'] = y_coords
        result_df['Z'] = z_coords

        logger.info(f"Added X, Y, Z coordinates to {len(result_df)} composites")

        return result_df

    # ============================================================================
    # SCAN ANALYSIS
    # ============================================================================

    def load_scan_file(self):
        """Load a scan file from file dialog."""
        try:
            logger.info("Opening scan file dialog...")

            # Open file dialog
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter(
                "Scan files (*.las *.laz *.ply *.obj *.stl *.xyz);;"
                "LAS files (*.las *.laz);;"
                "Mesh files (*.ply *.obj *.stl);;"
                "Point clouds (*.ply *.xyz);;"
                "All files (*)"
            )

            if file_dialog.exec():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    filepath = Path(selected_files[0])
                    logger.info(f"Loading scan file: {filepath}")

                    # Ensure scan panel is open
                    self.open_scan_panel()

                    # Load the scan file through the panel
                    if hasattr(self, 'scan_panel') and self.scan_panel:
                        # Call the panel's load method
                        self.scan_panel._load_scan_file(filepath)
                    else:
                        QMessageBox.warning(
                            self,
                            "Panel Not Available",
                            "Scan panel could not be opened. Please try again."
                        )

        except Exception as e:
            logger.error(f"Error loading scan file: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load scan file:\n\n{str(e)}"
            )

    def open_scan_panel(self):
        """Open Scan Analysis panel for fragmentation and surface analysis."""
        try:
            logger.info("Opening Scan Analysis panel...")

            # Import here to avoid circular imports
            from .scan_panel import ScanPanel

            # Create panel if it doesn't exist
            if not hasattr(self, 'scan_panel'):
                self.scan_panel = ScanPanel(parent=self)
                self.scan_panel.bind_controller(self.controller)

            # Show the panel
            self.scan_panel.show()
            self.scan_panel.raise_()
            self.scan_panel.activateWindow()

            logger.info("Scan Analysis panel opened successfully")

        except Exception as e:
            logger.error(f"Error opening Scan Analysis panel: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Panel Error",
                f"Failed to open Scan Analysis panel:\n\n{str(e)}"
            )

    def export_scan_psd(self):
        """Export scan PSD and fragment metrics to CSV."""
        try:
            logger.info("Exporting scan PSD data...")

            if not hasattr(self, 'scan_panel') or not hasattr(self.scan_panel, '_fragment_metrics'):
                QMessageBox.warning(
                    self,
                    "No Data",
                    "No scan fragment metrics available for export.\n\n"
                    "Please load and analyze a scan first."
                )
                return

            # Get fragment metrics
            metrics = self.scan_panel._fragment_metrics
            if not metrics:
                QMessageBox.warning(
                    self,
                    "No Data",
                    "No fragment metrics available for export."
                )
                return

            # Prepare data for export
            import pandas as pd

            # Create DataFrame with fragment metrics
            data = []
            for fm in metrics:
                data.append({
                    'fragment_id': fm.fragment_id,
                    'point_count': fm.point_count,
                    'volume_m3': fm.volume_m3,
                    'equivalent_diameter_m': fm.equivalent_diameter_m,
                    'sphericity': fm.sphericity,
                    'elongation': fm.elongation,
                    'aspect_ratio_l': fm.aspect_ratio[0],
                    'aspect_ratio_w': fm.aspect_ratio[1],
                    'aspect_ratio_h': fm.aspect_ratio[2],
                    'confidence_score': fm.confidence_score,
                    'centroid_x': fm.centroid[0],
                    'centroid_y': fm.centroid[1],
                    'centroid_z': fm.centroid[2],
                    'surface_area_m2': fm.surface_area_m2,
                    'bounding_box_volume_m3': fm.bounding_box_volume_m3
                })

            df = pd.DataFrame(data)

            # Export to CSV
            from pathlib import Path

            from PyQt6.QtWidgets import QFileDialog

            file_dialog = QFileDialog(self)
            file_dialog.setDefaultSuffix("csv")
            file_dialog.setNameFilter("CSV files (*.csv)")
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)

            if file_dialog.exec():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    filepath = Path(selected_files[0])
                    df.to_csv(filepath, index=False)

                    QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Fragment metrics exported to:\n\n{filepath}"
                    )

                    logger.info(f"Exported {len(metrics)} fragment metrics to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting scan PSD: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export scan PSD data:\n\n{str(e)}"
            )

    def export_scan_fragments(self):
        """Export scan fragments as mesh files."""
        try:
            logger.info("Exporting scan fragments...")

            QMessageBox.information(
                self,
                "Not Implemented",
                "Fragment mesh export is not yet implemented.\n\n"
                "This feature will be available in a future update."
            )

        except Exception as e:
            logger.error(f"Error exporting scan fragments: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export scan fragments:\n\n{str(e)}"
            )

    def clear_scan_data(self):
        """Clear all scan data and results."""
        try:
            logger.info("Clearing scan data...")

            # Clear scan registry
            if hasattr(self, 'controller') and hasattr(self.controller, 'scan_controller'):
                # Get all scan IDs and delete them
                scans = self.controller.scan_controller.get_scan_list()
                for scan_metadata in scans:
                    self.controller.scan_controller.delete_scan(scan_metadata.scan_id)

            # Remove scan from renderer
            if self.viewer_widget and self.viewer_widget.renderer:
                try:
                    self.viewer_widget.renderer.clear_layer("Scan")
                except Exception:
                    pass

            # Reset scan panel if it exists
            if hasattr(self, 'scan_panel'):
                self.scan_panel._current_scan_id = None
                self.scan_panel._fragment_labels = None
                self.scan_panel._fragment_metrics = None
                self.scan_panel._current_state = self.scan_panel.ScanPanelState.EMPTY
                self.scan_panel._update_ui_for_state()
                self.scan_panel._update_scan_info()
                self.scan_panel._log_message("Scan data cleared")

        except Exception as e:
            logger.error(f"Error clearing scan data: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Clear Error",
                f"Failed to clear scan data:\n\n{str(e)}"
            )

    def _on_scan_loaded(self, metadata):
        """Handle scan loaded signal and render it in the viewer."""
        try:
            logger.info(f"Rendering scan: {metadata.scan_id}")

            # Get scan data from registry
            if not self.controller or not hasattr(self.controller, 'scan_controller'):
                logger.warning("Controller or scan_controller not available")
                return

            scan_registry = self.controller.scan_controller.registry
            result = scan_registry.get_scan(metadata.scan_id)

            if result is None:
                logger.warning(f"Scan {metadata.scan_id} not found in registry")
                return

            scan_metadata, scan_data = result

            # Convert scan data to PyVista mesh
            import pyvista as pv

            if scan_data.points is None:
                logger.warning("Scan has no points to render")
                return

            # Create PyVista mesh
            if scan_data.is_mesh() and scan_data.faces is not None:
                # Mesh with faces - convert to PyVista format
                faces = np.asarray(scan_data.faces, dtype=np.int64)

                # PyVista expects faces as flat array: [3, i0, i1, i2, 3, j0, j1, j2, ...]
                # where the first number is the vertex count per face
                if faces.ndim == 2:
                    n_faces = len(faces)
                    n_verts_per_face = faces.shape[1]  # Usually 3 for triangles

                    # Prepend vertex count to each face
                    faces_pv = np.hstack([
                        np.full((n_faces, 1), n_verts_per_face, dtype=np.int64),
                        faces.astype(np.int64)
                    ]).flatten()

                    logger.debug(f"Converted {n_faces} faces from (N,{n_verts_per_face}) to PyVista format")
                else:
                    # Assume already in PyVista format
                    faces_pv = faces.astype(np.int64)

                mesh = pv.PolyData(scan_data.points, faces_pv)
            else:
                # Point cloud
                mesh = pv.PolyData(scan_data.points)

            # Add colors if available
            if scan_data.colors is not None:
                mesh['colors'] = scan_data.colors

            # Add intensities if available
            if scan_data.intensities is not None:
                mesh['intensity'] = scan_data.intensities

            # Render in viewer
            if self.viewer_widget and self.viewer_widget.renderer:
                layer_name = f"Scan_{metadata.scan_id.hex[:8]}"

                # Remove existing scan layer if present
                try:
                    self.viewer_widget.renderer.clear_layer(layer_name)
                except Exception:
                    pass

                # Determine rendering style
                if scan_data.is_mesh():
                    # Render as mesh
                    actor = self.viewer_widget.renderer.add_mesh(
                        mesh,
                        name=layer_name,
                        layer_type="scan",
                        show_edges=False,
                        opacity=1.0,
                        color='lightblue' if scan_data.colors is None else None
                    )
                else:
                    # Render as point cloud
                    scalars = 'intensity' if scan_data.intensities is not None else None
                    actor = self.viewer_widget.renderer.add_mesh(
                        mesh,
                        name=layer_name,
                        layer_type="scan",
                        point_size=5,
                        render_points_as_spheres=True,
                        scalars=scalars,
                        opacity=1.0,
                        color='lightblue' if scalars is None else None
                    )

                if actor:
                    logger.info(f"Successfully rendered scan {metadata.scan_id} as layer '{layer_name}'")
                    # Fit camera to view
                    try:
                        self.viewer_widget.renderer.reset_camera()
                    except Exception:
                        pass
                else:
                    logger.warning(f"Failed to render scan {metadata.scan_id}")
            else:
                logger.warning("Viewer widget or renderer not available")

        except Exception as e:
            logger.error(f"Error rendering scan: {e}", exc_info=True)
            QMessageBox.warning(
                self,
                "Render Error",
                f"Failed to render scan in viewer:\n\n{str(e)}\n\n"
                "The scan was loaded but may not be visible. Check logs for details."
            )

    # ============================================================================
    # INSAR (REMOTE SENSING)
    # ============================================================================
    def _with_insar_panel(self, callback):
        """Ensure InSAR panel exists, then invoke callback with panel instance."""
        self.panel_manager.show_panel("InSARPanel")
        panel_info = self.panel_manager.get_panel_info("InSARPanel")
        if panel_info and panel_info.instance:
            callback(panel_info.instance)
            return

        def _try_invoke():
            info = self.panel_manager.get_panel_info("InSARPanel")
            if info and info.instance:
                callback(info.instance)
        QTimer.singleShot(200, _try_invoke)

    def open_insar_panel(self):
        """Open the InSAR Deformation panel."""
        self.panel_manager.show_panel("InSARPanel")

    def run_insar_job(self):
        """Run InSAR job using current panel settings."""
        self._with_insar_panel(lambda panel: getattr(panel, "run_job_from_menu", lambda: None)())

    def ingest_insar_results(self):
        """Ingest InSAR results from output directory."""
        self._with_insar_panel(lambda panel: getattr(panel, "ingest_results_from_menu", lambda: None)())

    # ============================================================================
    # SURVEY DEFORMATION & SUBSIDENCE
    # ============================================================================

    def import_subsidence_survey_data(self):
        """Load subsidence survey CSV/Excel into the registry."""
        try:
            if not PANDAS_AVAILABLE or pd is None:
                QMessageBox.critical(self, "Dependency Missing", "pandas is required to import subsidence surveys.")
                return
            from PyQt6.QtWidgets import QFileDialog
            file_dialog = QFileDialog(self)
            file_dialog.setNameFilters(["CSV files (*.csv)", "Excel files (*.xlsx *.xls)", "All files (*.*)"])
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

            if not file_dialog.exec():
                return
            selected = file_dialog.selectedFiles()
            if not selected:
                return

            path = Path(selected[0])
            if path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)

            key = self.controller.survey_deformation.ingest_subsidence_dataframe(df, source_file=path)
            QMessageBox.information(
                self,
                "Subsidence Data Loaded",
                f"Loaded {len(df):,} survey rows from:\n{path}\n\nRegistry key: {key}"
            )
            self.statusBar().showMessage(f"Subsidence survey data registered as {key}", 3000)
        except Exception as e:
            logger.error(f"Failed to import subsidence survey data: {e}", exc_info=True)
            QMessageBox.critical(self, "Import Error", f"Failed to import subsidence data:\n{e}")

    def import_groundwater_data(self):
        """Load groundwater well CSV/Excel into the registry."""
        try:
            if not PANDAS_AVAILABLE or pd is None:
                QMessageBox.critical(self, "Dependency Missing", "pandas is required to import groundwater data.")
                return
            from PyQt6.QtWidgets import QFileDialog
            file_dialog = QFileDialog(self)
            file_dialog.setNameFilters(["CSV files (*.csv)", "Excel files (*.xlsx *.xls)", "All files (*.*)"])
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

            if not file_dialog.exec():
                return
            selected = file_dialog.selectedFiles()
            if not selected:
                return

            path = Path(selected[0])
            if path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)

            key = self.controller.survey_deformation.ingest_groundwater_dataframe(df, source_file=path)
            QMessageBox.information(
                self,
                "Groundwater Data Loaded",
                f"Loaded {len(df):,} groundwater rows from:\n{path}\n\nRegistry key: {key}"
            )
            self.statusBar().showMessage(f"Groundwater data registered as {key}", 3000)
        except Exception as e:
            logger.error(f"Failed to import groundwater data: {e}", exc_info=True)
            QMessageBox.critical(self, "Import Error", f"Failed to import groundwater data:\n{e}")

    def run_subsidence_analysis(self):
        """Run subsidence time-series engine and register outputs."""
        try:
            result = self.controller.survey_deformation.run_subsidence_time_series()
            QMessageBox.information(
                self,
                "Subsidence Analysis Complete",
                f"Computed time series for {len(result.per_point_metrics)} points.\n"
                f"Registered keys: subsidence_timeseries, subsidence_metrics"
            )
            self.statusBar().showMessage("Subsidence analysis complete", 3000)
        except Exception as e:
            logger.error(f"Subsidence analysis failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Subsidence analysis failed:\n{e}")

    def run_control_stability(self):
        """Run control stability classification."""
        try:
            bundle = self.controller.survey_deformation.run_control_stability()
            df = bundle["control_stability"]
            QMessageBox.information(
                self,
                "Control Stability Complete",
                f"Classified {len(df)} points.\nRegistered key: control_stability"
            )
            self.statusBar().showMessage("Control stability computed", 3000)
        except Exception as e:
            logger.error(f"Control stability failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Control stability failed:\n{e}")

    def run_groundwater_analysis(self):
        """Run groundwater time-series engine."""
        try:
            bundle = self.controller.survey_deformation.run_groundwater_time_series()
            metrics = bundle["metrics"]
            QMessageBox.information(
                self,
                "Groundwater Analysis Complete",
                f"Computed rates for {len(metrics)} wells.\nRegistered keys: groundwater_timeseries, groundwater_metrics"
            )
            self.statusBar().showMessage("Groundwater analysis complete", 3000)
        except Exception as e:
            logger.error(f"Groundwater analysis failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Groundwater analysis failed:\n{e}")

    def run_coupled_interpretation(self):
        """Run subsidence-groundwater coupling."""
        try:
            bundle = self.controller.survey_deformation.run_coupling()
            df = bundle["coupling"]
            QMessageBox.information(
                self,
                "Coupled Interpretation Complete",
                f"Computed correlations for {len(df)} points.\nRegistered key: subsidence_groundwater_coupling"
            )
            self.statusBar().showMessage("Coupled interpretation complete", 3000)
        except Exception as e:
            logger.error(f"Coupling failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Coupled interpretation failed:\n{e}")

    def compute_deformation_index(self):
        """Compute deformation index and register results."""
        try:
            bundle = self.controller.survey_deformation.run_deformation_index()
            df = bundle["deformation_index"]
            QMessageBox.information(
                self,
                "Deformation Index Complete",
                f"Ranked {len(df)} points.\nRegistered key: deformation_index"
            )
            self.statusBar().showMessage("Deformation index computed", 3000)
        except Exception as e:
            logger.error(f"Deformation index failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Deformation index failed:\n{e}")

    # ============================================================================
    # HELP
    # ============================================================================

    def show_documentation(self):
        """Show documentation."""
        QMessageBox.information(
            self,
            "Documentation",
            "See QUICK_START_GUIDE.md and UX_ARCHITECTURE.md for detailed documentation."
        )

    def show_shortcuts(self):
        """Show keyboard shortcuts."""
        shortcuts_text = f"""
        <h3>Keyboard Shortcuts</h3>
        <table cellpadding="8" style="border-collapse: collapse;">
        <tr style=f"background-color: {ModernColors.CARD_BG};"><th align="left" colspan="2">File Operations</th></tr>
        <tr><td><b>Ctrl+O</b></td><td>Open File</td></tr>
        <tr><td><b>Ctrl+1...9</b></td><td>Open Recent File 1-9</td></tr>
        <tr><td><b>Ctrl+Shift+S</b></td><td>Export Screenshot</td></tr>
        <tr><td><b>Ctrl+W</b></td><td>Clear Scene</td></tr>
        <tr><td><b>Ctrl+D</b></td><td>View Block Model Data</td></tr>
        <tr><td><b>Ctrl+Q</b></td><td>Exit Application</td></tr>
        
        <tr style=f"background-color: {ModernColors.CARD_BG};"><th align="left" colspan="2">View Controls</th></tr>
        <tr><td><b>R</b></td><td>Reset View</td></tr>
        <tr><td><b>F</b></td><td>Fit to Model</td></tr>
        <tr><td><b>E</b></td><td>Zoom to Extents</td></tr>
        <tr><td><b>Home</b></td><td>Reset View to Default</td></tr>
        <tr><td><b>+/=</b></td><td>Zoom In</td></tr>
        <tr><td><b>-</b></td><td>Zoom Out</td></tr>
        
        <tr style=f"background-color: {ModernColors.CARD_BG};"><th align="left" colspan="2">Camera Navigation</th></tr>
        <tr><td><b>Arrow Keys</b></td><td>Nudge View (Pan)</td></tr>
        <tr><td><b>Ctrl+Arrow Keys</b></td><td>Rotate View 15°</td></tr>
        <tr><td><b>1</b></td><td>Top View</td></tr>
        <tr><td><b>2</b></td><td>Bottom View</td></tr>
        <tr><td><b>3</b></td><td>Front View</td></tr>
        <tr><td><b>4</b></td><td>Back View</td></tr>
        <tr><td><b>5</b></td><td>Right View</td></tr>
        <tr><td><b>6</b></td><td>Left View</td></tr>
        <tr><td><b>7</b></td><td>Isometric View</td></tr>
        
        <tr style=f"background-color: {ModernColors.CARD_BG};"><th align="left" colspan="2">View Bookmarks</th></tr>
        <tr><td><b>Ctrl+Shift+1...9</b></td><td>Save Current View to Bookmark Slot 1-9</td></tr>
        <tr><td><b>Ctrl+Shift+F1...F9</b></td><td>Load Saved View from Bookmark Slot 1-9</td></tr>
        
        <tr style=f"background-color: {ModernColors.CARD_BG};"><th align="left" colspan="2">Display Toggles</th></tr>
        <tr><td><b>O</b></td><td>Toggle Orthographic/Perspective</td></tr>
        <tr><td><b>A</b></td><td>Toggle Axes</td></tr>
        <tr><td><b>G</b></td><td>Toggle Grid</td></tr>
        
        <tr style=f"background-color: {ModernColors.CARD_BG};"><th align="left" colspan="2">Mouse Modes</th></tr>
        <tr><td><b>S</b></td><td>Select (Click) Mode</td></tr>
        <tr><td><b>P</b></td><td>Pan Mode</td></tr>
        <tr><td><b>Z</b></td><td>Zoom Box Mode</td></tr>
        <tr><td><b>Space</b></td><td>Toggle Select/Pan Mode</td></tr>
        
        <tr style=f"background-color: {ModernColors.CARD_BG};"><th align="left" colspan="2">Help</th></tr>
        <tr><td><b>F1</b></td><td>Show This Help</td></tr>
        </table>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(shortcuts_text)
        msg.exec()

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About GeoX",
            "<h2>GeoX</h2>"
            "<p>Professional geoscience visualization tool</p>"
            "<p>Version 1.0-alpha</p>"
            "<p>Built with PySide6 + PyVista</p>"
        )

    # ============================================================================
    # COORDINATE ALIGNMENT HELPERS
    # ============================================================================

    def register_drillhole_data(self, collar_df: pd.DataFrame, dataset_name: str = "Drillhole Data") -> None:
        """
        Register drillhole collar data with the coordinate manager and check alignment.
        
        Args:
            collar_df: DataFrame with X, Y, Z columns (collar coordinates)
            dataset_name: Name for the dataset in the coordinate manager
        """
        try:
            # Register with coordinate manager
            dataset_info = self.coordinate_manager.register_dataset(
                dataset_name,
                collar_df,
                'drillhole'
            )

            # Check if alignment is needed
            if not dataset_info.is_aligned:
                logger.warning("âš ï¸  Drillhole data requires coordinate alignment!")

                # Ask user if they want to auto-align
                reply = QMessageBox.question(
                    self,
                    "Coordinate Alignment Required",
                    f"The drillhole data appears to be in a different coordinate system than existing data.\n\n"
                    f"Drillhole center: ({dataset_info.bounds.center[0]:,.2f}, {dataset_info.bounds.center[1]:,.2f}, {dataset_info.bounds.center[2]:,.2f})\n"
                    f"Required offset: ({dataset_info.offset[0]:,.2f}, {dataset_info.offset[1]:,.2f}, {dataset_info.offset[2]:,.2f}) meters\n\n"
                    f"Would you like to automatically align this drillhole data with existing block model?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )

                if reply == QMessageBox.StandardButton.Yes:
                    # Return aligned dataframe for caller to use
                    aligned_df = self.coordinate_manager.apply_offset_to_dataframe(
                        collar_df,
                        dataset_info.offset,
                        coord_cols=('X', 'Y', 'Z')
                    )
                    dataset_info.is_aligned = True
                    logger.info("Drillhole data aligned to reference frame")

                    # Show alignment summary
                    summary = self.coordinate_manager.get_alignment_summary()
                    logger.info(summary)

                    return aligned_df

            return collar_df

        except Exception as e:
            logger.error(f"Error registering drillhole data: {e}", exc_info=True)
            return collar_df

    # ============================================================================
    # BLOCK SELECTION
    # ============================================================================

    def on_block_picked(self, block_id: int, block_data: dict):
        """Handle block selection."""
        if not self._has_valid_block_model():
            logger.warning("Block picked but no model is loaded")
            return

        try:
            positions = self.current_model.positions
            if positions is None or len(positions) == 0:
                logger.warning("Block picked but model has no positions")
                return

            if block_id < 0 or block_id >= len(positions):
                # Ignore invalid picks quietly (often from clicking empty space)
                return

            coordinates = tuple(positions[block_id])


            # Update GC Decision panel
            if self.gc_decision_panel:
                try:
                    self.gc_decision_panel.update_block_info(block_id, block_data, coordinates)
                except Exception as e:
                    logger.error(f"Error updating GC Decision panel: {e}", exc_info=True)

            # Update status bar
            x, y, z = coordinates
            self.status_bar.showMessage(
                f"Selected Block {block_id} at ({x:.2f}, {y:.2f}, {z:.2f})"
            )

            logger.info(f"Block {block_id} selected")
        except Exception as e:
            logger.error(f"Error handling block pick: {e}", exc_info=True)
            self.status_bar.showMessage(f"Error selecting block: {str(e)}", 3000)

    def on_clear_block_selection(self):
        """Clear block selection."""
        if self.viewer_widget:
            self.viewer_widget.clear_block_selection()

        if self.gc_decision_panel:
            self.gc_decision_panel.clear_info()

        self.status_bar.showMessage("Selection cleared")
        logger.info("Block selection cleared")

    # ========================================================================
    # DATA ANALYSIS EVENT HANDLERS
    # ========================================================================

    def on_analysis_filter_applied(self, property_name: str, min_val: float, max_val: float):
        """Handle filter application from data analysis panel."""
        if self.viewer_widget:
            self.viewer_widget.apply_property_filter(property_name, min_val, max_val)
            self.status_bar.showMessage(
                f"Filter applied: {min_val:.3f} â‰¤ {property_name} â‰¤ {max_val:.3f}",
                3000
            )
            logger.info(f"Applied filter from analysis panel: {property_name} [{min_val}, {max_val}]")

    def on_analysis_filter_cleared(self):
        """Handle filter clearing from data analysis panel."""
        if self.viewer_widget:
            self.viewer_widget.apply_property_filter("", 0, 0)  # Clear filter
            self.status_bar.showMessage("Filter cleared", 2000)
            logger.info("Cleared filter from analysis panel")

    def on_analysis_slice_applied(self, axis: str, position: float, keep_side: str):
        """Handle slice application from data analysis panel."""
        # Note: Slider-based slicing removed - use interactive 3D slicing in View menu
        self.status_bar.showMessage(
            "Old slice method deprecated. Use 'View > Interactive Slice' for 3D plane-based slicing",
            3000
        )
        logger.info(f"Slice request from analysis panel (deprecated): {axis} at {position}, keep {keep_side}")

    def on_analysis_slice_cleared(self):
        """Handle slice clearing from data analysis panel."""
        # Note: Slider-based slicing removed - use interactive 3D slicing in View menu
        self.status_bar.showMessage(
            "Old slice method deprecated. Use 'View > Interactive Slice' for 3D plane-based slicing",
            2000
        )
        logger.info("Slice clear request from analysis panel (deprecated)")

    def on_analysis_statistics_requested(self, property_name: str):
        """Handle statistics request from data analysis panel."""
        # Use the existing show_statistics method
        self.show_statistics()

    def on_swath_highlight_requested(self, axis: str, lower: float, upper: float):
        """Handle swath highlight request from data analysis panel."""
        if self.viewer_widget:
            self.viewer_widget.highlight_swath_interval(axis, lower, upper)
            logger.info(f"Highlighted swath interval: {axis} [{lower}, {upper}]")

    def on_swath_highlight_cleared(self):
        """Handle swath highlight clear request from data analysis panel."""
        if self.viewer_widget:
            self.viewer_widget.clear_swath_highlight()
            logger.info("Cleared swath highlight")

    # ============================================================================
    # UPDATES
    # ============================================================================

    def _update_camera_info(self):
        """Update camera info in scene inspector."""
        if not self.scene_inspector_panel or not self.viewer_widget:
            return

        try:
            camera_info = self.viewer_widget.get_camera_info()
            if camera_info:
                position = camera_info.get('position')
                focal_point = camera_info.get('focal_point')

                if position and focal_point:
                    self.scene_inspector_panel.update_camera_info(position, focal_point)
        except (RuntimeError, AttributeError):
            # Silently ignore when plotter is being modified
            pass
        except Exception as e:
            try:
                e_msg = str(e)
                logger.debug(f"Camera info update error: {e_msg}")
            except Exception:
                logger.debug("Camera info update error: <unprintable error>")

    def _update_status_bar(self):
        """Update status bar information."""
        if self.status is not None:
            self.status.update_status_bar()

    # ============================================================================
    # KEYBOARD NAVIGATION
    # ============================================================================

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for navigation and interaction."""
        from PyQt6.QtCore import Qt

        key = event.key()
        modifiers = event.modifiers()

        # Quick measurement controls: finish/cancel area measurement
        try:
            if self.viewer_widget and getattr(self.viewer_widget, 'renderer', None):
                mode = getattr(self.viewer_widget.renderer, 'measure_mode', None)
                if mode == 'area':
                    if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                        self.viewer_widget.renderer.finish_area_measurement()
                        self.statusBar().showMessage("Area measurement finished", 2000)
                        event.accept()
                        return
                    elif key == Qt.Key.Key_Escape:
                        self.viewer_widget.renderer.cancel_area_measurement()
                        self.statusBar().showMessage("Area measurement cancelled", 2000)
                        event.accept()
                        return
        except Exception:
            # Don't block other shortcuts if renderer isn't ready
            pass

        # Arrow keys - Nudge view slightly
        if key == Qt.Key.Key_Up:
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                # Ctrl+Arrow: Rotate view 15°
                if self.viewer_widget and self.viewer_widget.renderer:
                    self.viewer_widget.renderer.rotate_camera(0, 15)
                    self.statusBar().showMessage("Rotated view up 15°", 2000)
            else:
                # Arrow: Nudge view
                if self.viewer_widget and self.viewer_widget.renderer:
                    self.viewer_widget.renderer.pan_camera(0, 0.05)
                    self.statusBar().showMessage("View nudged up", 1000)
            event.accept()
            return

        elif key == Qt.Key.Key_Down:
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                if self.viewer_widget and self.viewer_widget.renderer:
                    self.viewer_widget.renderer.rotate_camera(0, -15)
                    self.statusBar().showMessage("Rotated view down 15°", 2000)
            else:
                if self.viewer_widget and self.viewer_widget.renderer:
                    self.viewer_widget.renderer.pan_camera(0, -0.05)
                    self.statusBar().showMessage("View nudged down", 1000)
            event.accept()
            return

        elif key == Qt.Key.Key_Left:
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                if self.viewer_widget and self.viewer_widget.renderer:
                    self.viewer_widget.renderer.rotate_camera(-15, 0)
                    self.statusBar().showMessage("Rotated view left 15°", 2000)
            else:
                if self.viewer_widget and self.viewer_widget.renderer:
                    self.viewer_widget.renderer.pan_camera(-0.05, 0)
                    self.statusBar().showMessage("View nudged left", 1000)
            event.accept()
            return

        elif key == Qt.Key.Key_Right:
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                if self.viewer_widget and self.viewer_widget.renderer:
                    self.viewer_widget.renderer.rotate_camera(15, 0)
                    self.statusBar().showMessage("Rotated view right 15°", 2000)
            else:
                if self.viewer_widget and self.viewer_widget.renderer:
                    self.viewer_widget.renderer.pan_camera(0.05, 0)
                    self.statusBar().showMessage("View nudged right", 1000)
            event.accept()
            return

        # +/- for zoom
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in()
            event.accept()
            return

        elif key == Qt.Key.Key_Minus:
            self.zoom_out()
            event.accept()
            return

        # Home key - Reset view
        elif key == Qt.Key.Key_Home:
            self.reset_camera()
            self.statusBar().showMessage("View reset to default (Home)", 2000)
            event.accept()
            return


        # Space - Toggle between select/pan modes
        elif key == Qt.Key.Key_Space:
            if not event.isAutoRepeat():  # Avoid triggering on key repeat
                # Toggle between select and pan modes
                if self.viewer_widget and self.viewer_widget.renderer:
                    current_mode = getattr(self.viewer_widget.renderer, '_current_mouse_mode', 'select')
                    if current_mode == 'select':
                        self.set_mouse_mode_pan()
                    else:
                        self.set_mouse_mode_select()
                event.accept()
                return

        # Call base implementation for other keys
        super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        """Handle drag enter events for file drops."""
        if event.mimeData().hasUrls():
            # Check if any of the URLs are valid file types
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())
                    # Accept CSV, VTK, and other supported formats
                    if file_path.suffix.lower() in ['.csv', '.vtk', '.vtp', '.vtu', '.txt']:
                        event.acceptProposedAction()
                        # Visual feedback - highlight window border
                        self.setStyleSheet("QMainWindow { border: 3px solid #4CAF50; }")
                        self.statusBar().showMessage("Drop file to load...", 2000)
                        return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave events."""
        # Remove visual feedback
        self.setStyleSheet("")
        self.statusBar().clearMessage()

    def dropEvent(self, event):
        """Handle file drop events."""
        # Remove visual feedback
        self.setStyleSheet("")

        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = Path(urls[0].toLocalFile())

                # Validate file type
                if file_path.suffix.lower() not in ['.csv', '.vtk', '.vtp', '.vtu', '.txt']:
                    QMessageBox.warning(
                        self,
                        "Invalid File Type",
                        f"Unsupported file type: {file_path.suffix}\n\n"
                        "Supported formats: CSV, VTK, VTP, VTU, TXT"
                    )
                    return

                # Check if file exists
                if not file_path.exists():
                    QMessageBox.warning(
                        self,
                        "File Not Found",
                        f"The file does not exist:\n{file_path}"
                    )
                    return

                # Load the file
                self.statusBar().showMessage(f"Loading {file_path.name}...", 3000)
                logger.info(f"File dropped: {file_path}")

                # Use existing load_file method
                try:
                    self.load_file(file_path)
                    event.acceptProposedAction()
                except Exception as e:
                    logger.error(f"Error loading dropped file: {e}")
                    QMessageBox.critical(
                        self,
                        "Load Error",
                        f"Failed to load file:\n{str(e)}"
                    )
        else:
            event.ignore()

    # ============================================================================
    # LIFECYCLE
    # ============================================================================

    def resizeEvent(self, event):
        """Log main window resize events for audit trail."""
        old_size = event.oldSize()
        new_size = event.size()
        
        # Only log significant size changes (avoid spam from tiny adjustments)
        size_tuple = (new_size.width(), new_size.height())
        if self._last_logged_size != size_tuple:
            logger.info(
                f"MainWindow resized: "
                f"{old_size.width()}x{old_size.height()} -> {new_size.width()}x{new_size.height()}"
            )
            self._last_logged_size = size_tuple
        
        super().resizeEvent(event)

    def showEvent(self, event):
        """Log main window show event for audit trail."""
        logger.info("MainWindow shown")
        super().showEvent(event)

    def closeEvent(self, event):
        """Handle window close event - comprehensive cleanup."""
        logger.info("Main window closeEvent triggered - starting cleanup")
        
        # Close legend widget if it exists (from original closeEvent)
        try:
            vw = getattr(self, "viewer_widget", None)
            if vw is not None:
                lm = getattr(vw, "_legend_manager", None) or getattr(vw, "renderer", None)
                legend_manager = None
                if lm and hasattr(lm, "widget"):
                    legend_manager = lm
                elif getattr(vw, "renderer", None) is not None and hasattr(vw.renderer, "legend_manager"):
                    legend_manager = vw.renderer.legend_manager
                if legend_manager and getattr(legend_manager, "widget", None) is not None:
                    try:
                        legend_manager.widget.close()
                    except Exception:
                        pass
        except Exception:
            pass

        # Stop timers
        if self.camera_update_timer:
            self.camera_update_timer.stop()
        if self.status is not None:
            self.status.stop()

        # Close all tracked panels/dialogs
        for dialog in self._open_panels[:]:  # Copy list to avoid modification during iteration
            if dialog is not None:
                try:
                    dialog.close()
                except Exception as e:
                    try:
                        e_msg = str(e)
                        logger.debug(f"Error closing dialog: {e_msg}")
                    except Exception:
                        logger.debug("Error closing dialog: <unprintable error>")

        # Also close explicitly tracked dialogs (backward compatibility)
        dialogs_to_close = [
            self.statistics_dialog,
            self.charts_dialog,
            self.swath_dialog,
            self.swath_analysis_3d_dialog,
            self.data_viewer_dialog,
            self.domain_compositing_dialog,
            self.drillhole_reporting_dialog,
            self.drillhole_plotting_dialog,
            self.grade_transformation_dialog,
            self.block_resource_dialog,
            # self.drillhole_resource_dialog,  # Removed - panel deleted
            self.irr_dialog,
            self.kmeans_dialog,
            self.resource_classification_dialog,
            self.resource_reporting_dialog,
            # Removed: standalone_resource_calculator_dialog (redundant)
            self.grade_tonnage_dialog,
            self.pit_optimisation_dialog,
            self.cross_section_dialog,
            self.selection_dialog,
            self.cross_section_manager_dialog,
            self.geotech_dialog,
            self.variogram_assistant_dialog,
            self.soft_kriging_dialog,
            self.uncertainty_propagation_dialog,
            self.research_dashboard_dialog,
            self.loopstructural_dialog,
            self.compositing_window,
            getattr(self, 'block_model_import_dialog', None),
            getattr(self, 'jorc_classification_dialog', None),
            getattr(self, 'declustering_dialog', None),
            getattr(self, 'data_registry_status_dialog', None),
            getattr(self, 'underground_panel_dialog', None),
            getattr(self, 'npvs_dialog', None),
            getattr(self, 'pushback_designer_dialog', None),
            getattr(self, 'production_dashboard_dialog', None),
            getattr(self, 'strategic_schedule_dialog', None),
            getattr(self, 'tactical_schedule_dialog', None),
            getattr(self, 'short_term_schedule_dialog', None),
            getattr(self, 'fleet_dialog', None),
            getattr(self, 'planning_dashboard_dialog', None),
            getattr(self, 'reconciliation_dialog', None),
            getattr(self, 'grade_control_dialog', None),
            getattr(self, 'digline_dialog', None),
            getattr(self, 'geomet_dialog', None),
            getattr(self, 'geomet_domain_dialog', None),
            getattr(self, 'geomet_plant_dialog', None),
            getattr(self, 'mps_dialog', None),
            getattr(self, 'grf_dialog', None),
        ]

        for dialog in dialogs_to_close:
            if dialog is not None and dialog not in self._open_panels:
                try:
                    dialog.close()
                except Exception as e:
                    try:
                        e_msg = str(e)
                        logger.debug(f"Error closing dialog: {e_msg}")
                    except Exception:
                        logger.debug("Error closing dialog: <unprintable error>")

        # Save state (includes dock layout persistence)
        self._save_state()
        self._save_layout()
        # Save session (last file, renderer state)
        self._save_session()
        # Ensure any floating overlay widgets are closed (legend, scale bar, etc.)
        try:
            if self.viewer_widget and getattr(self.viewer_widget, 'renderer', None):
                try:
                    self.viewer_widget.renderer.close_overlays()
                except Exception:
                    logger.debug('Failed to close renderer overlays during shutdown', exc_info=True)
        except Exception:
            logger.debug('Error while attempting to destroy overlays', exc_info=True)

        # Try to cleanly shut down the PyVista plotter/QtInteractor to avoid
        # noisy VTK framebuffer errors during application exit.
        try:
            if self.viewer_widget is not None:
                plotter = getattr(self.viewer_widget, 'plotter', None)
                if plotter is not None:
                    try:
                        # Attempt to clear actors and close plotter if supported
                        try:
                            plotter.clear()
                        except Exception:
                            pass
                        try:
                            plotter.close()
                        except Exception:
                            pass
                        try:
                            # If the interactor wrapper exists, try a graceful close
                            inter = getattr(plotter, 'interactor', None) or getattr(plotter, 'iren', None)
                            if inter is not None and hasattr(inter, 'Disable'):
                                try:
                                    inter.Disable()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception:
                        logger.debug('Error while attempting to destroy plotter', exc_info=True)
                    finally:
                        try:
                            self.viewer_widget.plotter = None
                        except Exception:
                            pass
        except Exception:
            logger.debug('Plotter shutdown sequence failed', exc_info=True)

        # Shutdown PanelManager
        try:
            self.panel_manager.shutdown()
        except Exception as e:
            logger.debug(f'PanelManager shutdown failed: {e}')

        # Close all dock widgets
        try:
            for dock in self.findChildren(QDockWidget):
                try:
                    dock.close()
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f'Error closing dock widgets: {e}')

        # Close drillhole panel registry
        try:
            for panel_id, dock_widget in list(self._drillhole_panel_registry.items()):
                try:
                    if dock_widget is not None:
                        dock_widget.close()
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f'Error closing drillhole panels: {e}')

        # Force close any remaining top-level windows associated with this app
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                # Close all windows except the main window (which is closing anyway)
                for widget in app.topLevelWidgets():
                    if widget is not self and widget.isVisible():
                        try:
                            widget.close()
                        except Exception:
                            pass
        except Exception as e:
            logger.debug(f'Error closing top-level widgets: {e}')

        logger.info("Application closing - all panels and windows closed")
        return super().closeEvent(event)

    # ============================================================================
    # SESSION AUTOSAVE / RESTORE
    # ============================================================================
    def _restore_session_on_startup(self):
        """If enabled, reopen last file and reapply renderer state."""
        try:
            settings = QSettings("GeoX", "Session")
            restore_flag = settings.value("restore_on_startup", True, type=bool)
            if not bool(restore_flag):
                return
            last_file = settings.value("last_file", "", type=str)
            state_json = settings.value("renderer_state", "", type=str)
            self._pending_session_state = None
            if state_json:
                try:
                    self._pending_session_state = json.loads(state_json)
                except Exception:
                    self._pending_session_state = None
            if last_file:
                p = Path(last_file)
                if p.exists():
                    logger.info(f"Restoring last session: {p}")
                    self.load_file(p)
                    # Schedule a delayed state update to catch any layers loaded during session restore
                    # This runs after file loading task completes
                    QTimer.singleShot(500, self._force_state_update_after_restore)
                else:
                    logger.info("Last session file not found; skipping restore")
                    self._pending_session_state = None
        except Exception as e:
            try:
                e_msg = str(e)
                logger.debug(f"Session restore skipped: {e_msg}")
            except Exception:
                logger.debug("Session restore skipped: <unprintable error>")

    def _force_state_update_after_restore(self):
        """Force app state update after session restore - ensures UI is enabled."""
        try:
            if self.controller:
                self.controller._update_state_from_scene()
                logger.info("Forced app state update after session restore")
        except Exception as e:
            logger.debug(f"Failed to force state update: {e}")

    def _save_session(self):
        """Persist last file and renderer state for next launch."""
        try:
            settings = QSettings("GeoX", "Session")
            if self.current_file_path:
                settings.setValue("last_file", str(self.current_file_path))
            # Persist renderer state
            try:
                if self.viewer_widget and self.viewer_widget.renderer:
                    state = self.viewer_widget.renderer.get_session_state()
                    settings.setValue("renderer_state", json.dumps(state))
            except Exception:
                pass
            # Also persist the toggle state, in case it changed programmatically
            try:
                if hasattr(self, 'restore_session_action') and self.restore_session_action is not None:
                    settings.setValue("restore_on_startup", bool(self.restore_session_action.isChecked()))
            except Exception:
                pass
        except Exception as e:
            try:
                e_msg = str(e)
                logger.debug(f"Failed to save session: {e_msg}")
            except Exception:
                logger.debug("Failed to save session: <unprintable error>")

    # ============================================================================
    # UNDO / REDO
    # ============================================================================
    def _on_renderer_state_change(self, snapshot: Dict[str, Any], reason: str = ""):
        """Receive pre-change snapshots from renderer and push onto undo stack."""
        try:
            if not hasattr(self, '_undo_stack'):
                self._undo_stack = []
                self._redo_stack = []
                self._max_history = 50
            # Push snapshot and trim
            self._undo_stack.append(snapshot)
            if len(self._undo_stack) > getattr(self, '_max_history', 50):
                self._undo_stack = self._undo_stack[-self._max_history:]
            # Clear redo on new action
            self._redo_stack.clear()
            # Update UI state
            if hasattr(self, 'undo_action'):
                self.undo_action.setEnabled(len(self._undo_stack) > 0)
            if hasattr(self, 'redo_action'):
                self.redo_action.setEnabled(len(self._redo_stack) > 0)
            # Optional status
            self.statusBar().showMessage("Change captured for undo", 1500)
            # Mark project dirty
            self._mark_dirty()
        except Exception as e:
            try:
                e_msg = str(e)
                logger.debug(f"Undo snapshot push failed: {e_msg}")
            except Exception:
                logger.debug("Undo snapshot push failed: <unprintable error>")

    def _mark_dirty(self):
        """Mark current project as having unsaved changes and update title."""
        try:
            self._dirty = True
            base = "GeoX"
            if self.current_project_path:
                base += f" - {self.current_project_path.stem}"
            self.setWindowTitle(base + "*")
        except Exception:
            pass

    def _clear_dirty(self):
        try:
            self._dirty = False
            base = "GeoX"
            if self.current_project_path:
                base += f" - {self.current_project_path.stem}"
            self.setWindowTitle(base)
        except Exception:
            pass

    def _autosave_if_dirty(self):
        """Periodically write an autosave backup if there are unsaved changes."""
        try:
            if not self._dirty:
                return
            if not self.current_project_path:
                return
            # Write to sidecar autosave file next to project
            autosave_path = self.current_project_path.with_suffix(self.current_project_path.suffix + ".autosave")
            state = self._collect_project_state()
            autosave_path.parent.mkdir(parents=True, exist_ok=True)
            with open(autosave_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Autosaved project backup: {autosave_path}")
        except Exception as e:
            try:
                e_msg = str(e)
                logger.debug(f"Autosave skipped: {e_msg}")
            except Exception:
                logger.debug("Autosave skipped: <unprintable error>")

    # ============================================================================
    # PROJECT: Save / Load
    # ============================================================================
    def _collect_project_state(self) -> Dict[str, Any]:
        """Collect the current application state for project serialization."""
        data_files: Dict[str, Any] = {}
        if self.current_file_path:
            data_files['block_model'] = str(self.current_file_path)

        # Save drillhole data if present - check multiple sources
        drillhole_state = None
        try:
            drillhole_data = None
            composites_df = None
            assays_df = None
            collars_df = None
            surveys_df = None
            trajectories_df = None
            color_by = ''
            radius = 5.0

            # Method 1: Check compositing panel for composited data
            if hasattr(self, 'domain_compositing_panel') and self.domain_compositing_panel:
                if hasattr(self.domain_compositing_panel, 'comp_domain_df') and self.domain_compositing_panel.comp_domain_df is not None:
                    composites_df = self.domain_compositing_panel.comp_domain_df
                    logger.info(f"Found {len(composites_df)} composites in compositing panel")

                if hasattr(self.domain_compositing_panel, 'assay_df') and self.domain_compositing_panel.assay_df is not None:
                    assays_df = self.domain_compositing_panel.assay_df
                    logger.info(f"Found {len(assays_df)} assays in drillhole loading panel")

                if hasattr(self.domain_compositing_panel, 'collar_df') and self.domain_compositing_panel.collar_df is not None:
                    collars_df = self.domain_compositing_panel.collar_df
                    logger.info(f"Found {len(collars_df)} collars in drillhole loading panel")

                if hasattr(self.domain_compositing_panel, 'survey_df') and self.domain_compositing_panel.survey_df is not None:
                    surveys_df = self.domain_compositing_panel.survey_df
                    logger.info(f"Found {len(surveys_df)} surveys in drillhole loading panel")

            # Method 2: Check DataRegistry for drillhole data (using proper getter method)
            try:
                from ..core.data_registry import DataRegistry
                if hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None:
                    registry = DataRegistry._instance
                    # Use proper getter method - get_drillhole_data() returns data from _data_store
                    reg_data = registry.get_drillhole_data(copy_data=False)
                    if reg_data is not None:
                        if assays_df is None and 'assays' in reg_data and reg_data['assays'] is not None:
                            assays_df = reg_data['assays']
                            logger.info(f"Found {len(assays_df)} assays in DataRegistry")
                        if collars_df is None and 'collars' in reg_data and reg_data['collars'] is not None:
                            collars_df = reg_data['collars']
                            logger.info(f"Found {len(collars_df)} collars in DataRegistry")
                        if surveys_df is None and 'surveys' in reg_data and reg_data['surveys'] is not None:
                            surveys_df = reg_data['surveys']
                            logger.info(f"Found {len(surveys_df)} surveys in DataRegistry")
                        if trajectories_df is None and 'trajectories' in reg_data and reg_data['trajectories'] is not None:
                            trajectories_df = reg_data['trajectories']
                            logger.info(f"Found {len(trajectories_df)} trajectories in DataRegistry")
                        if composites_df is None and 'composites' in reg_data and reg_data['composites'] is not None:
                            composites_df = reg_data['composites']
                            logger.info(f"Found {len(composites_df)} composites in DataRegistry")
                        if 'lithology' in reg_data and reg_data['lithology'] is not None:
                            # Save lithology as well
                            pass  # Will be handled in save logic below
            except Exception as reg_error:
                try:
                    reg_msg = str(reg_error)
                    logger.debug(f"Could not check DataRegistry: {reg_msg}")
                except Exception:
                    logger.debug("Could not check DataRegistry: <unprintable error>")

            # Method 3: Check renderer for drillhole data (existing method)
            if (hasattr(self.viewer_widget.renderer, 'drillhole_data') and
                self.viewer_widget.renderer.drillhole_data is not None):
                drillhole_data = self.viewer_widget.renderer.drillhole_data

                if composites_df is None and 'composites_df' in drillhole_data:
                    composites_df = drillhole_data['composites_df']
                    logger.info(f"Found {len(composites_df)} composites in renderer")

                # Get visualization settings from renderer
                color_by = drillhole_data.get('color_by', '')
                radius = float(drillhole_data.get('radius', 5.0))

            # Save all found drillhole data to CSV files (only if files don't exist)
            if self.current_project_path and (composites_df is not None or assays_df is not None or
                                             collars_df is not None or surveys_df is not None or
                                             trajectories_df is not None):
                # Create a drillhole_data subfolder next to the project file
                drillhole_folder = self.current_project_path.parent / f"{self.current_project_path.stem}_drillholes"
                drillhole_folder.mkdir(exist_ok=True)

                saved_files = {}

                # Save composites (only if file doesn't exist)
                if composites_df is not None and not composites_df.empty:
                    composites_path = drillhole_folder / "composites.csv"
                    if not composites_path.exists():
                        from ..utils.export_helpers import export_dataframe_to_csv
                        export_dataframe_to_csv(composites_df, composites_path, process_events=False)
                        logger.info(f"Exported {len(composites_df)} composites to {composites_path}")
                    saved_files['composites'] = str(composites_path)

                # Save assays (only if file doesn't exist)
                if assays_df is not None and not assays_df.empty:
                    assays_path = drillhole_folder / "assays.csv"
                    if not assays_path.exists():
                        from ..utils.export_helpers import export_dataframe_to_csv
                        export_dataframe_to_csv(assays_df, assays_path, process_events=False)
                        logger.info(f"Exported {len(assays_df)} assays to {assays_path}")
                    saved_files['assays'] = str(assays_path)

                # Save collars (only if file doesn't exist)
                if collars_df is not None and not collars_df.empty:
                    collars_path = drillhole_folder / "collars.csv"
                    if not collars_path.exists():
                        from ..utils.export_helpers import export_dataframe_to_csv
                        export_dataframe_to_csv(collars_df, collars_path, process_events=False)
                        logger.info(f"Exported {len(collars_df)} collars to {collars_path}")
                    saved_files['collars'] = str(collars_path)

                # Save surveys (only if file doesn't exist)
                if surveys_df is not None and not surveys_df.empty:
                    surveys_path = drillhole_folder / "surveys.csv"
                    if not surveys_path.exists():
                        from ..utils.export_helpers import export_dataframe_to_csv
                        export_dataframe_to_csv(surveys_df, surveys_path, process_events=False)
                        logger.info(f"Exported {len(surveys_df)} surveys to {surveys_path}")
                    saved_files['surveys'] = str(surveys_path)

                # Save trajectories (only if file doesn't exist)
                if trajectories_df is not None and not trajectories_df.empty:
                    trajectories_path = drillhole_folder / "trajectories.csv"
                    if not trajectories_path.exists():
                        from ..utils.export_helpers import export_dataframe_to_csv
                        export_dataframe_to_csv(trajectories_df, trajectories_path, process_events=False)
                        logger.info(f"Exported {len(trajectories_df)} trajectories to {trajectories_path}")
                    saved_files['trajectories'] = str(trajectories_path)

                # Save lithology (from DataRegistry) (only if file doesn't exist)
                try:
                    from ..core.data_registry import DataRegistry
                    if hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None:
                        registry = DataRegistry._instance
                        reg_data = registry.get_drillhole_data(copy_data=False)
                        if reg_data is not None and 'lithology' in reg_data and reg_data['lithology'] is not None:
                            lithology_df = reg_data['lithology']
                            if not lithology_df.empty:
                                lithology_path = drillhole_folder / "lithology.csv"
                                if not lithology_path.exists():
                                    from ..utils.export_helpers import export_dataframe_to_csv
                                    export_dataframe_to_csv(lithology_df, lithology_path, process_events=False)
                                    logger.info(f"Exported {len(lithology_df)} lithology records to {lithology_path}")
                                saved_files['lithology'] = str(lithology_path)
                except Exception as e:
                    logger.debug(f"Could not save lithology: {e}")

                # Create drillhole state dict
                if saved_files:
                    drillhole_state = saved_files.copy()
                    # Also save visualization settings if available
                    if color_by:
                        drillhole_state['color_by'] = color_by
                    drillhole_state['radius'] = radius

                    logger.info(f"Saved drillhole data: {', '.join(saved_files.keys())}")
        except Exception as e:
            logger.error(f"Error saving drillhole data: {e}", exc_info=True)

        # Save all DataRegistry models and results
        registry_models_state = None
        try:
            import pickle

            import pandas as pd

            from ..core.data_registry import DataRegistry

            if hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None:
                registry = DataRegistry._instance

                if self.current_project_path:
                    models_folder = self.current_project_path.parent / f"{self.current_project_path.stem}_models"
                    models_folder.mkdir(exist_ok=True)

                    saved_models = {}

                    # Helper function to save a registry data item (only if file doesn't exist)
                    def save_registry_item(data_key: str, filename: str, display_name: str):
                        """Save a registry data item using proper getter methods."""
                        try:
                            data = registry.get_data(data_key, copy_data=False)
                            if data is not None:
                                save_path = models_folder / filename
                                # For DataFrames, try to save as CSV for readability
                                if isinstance(data, pd.DataFrame):
                                    csv_path = save_path.with_suffix('.csv')
                                    if not csv_path.exists():
                                        from ..utils.export_helpers import export_dataframe_to_csv
                                        export_dataframe_to_csv(data, csv_path, process_events=False)
                                        logger.info(f"Exported {display_name} ({len(data)} rows) to project")
                                    saved_models[data_key] = str(csv_path)
                                else:
                                    # Use pickle for other data types
                                    pkl_path = save_path.with_suffix('.pkl')
                                    if not pkl_path.exists():
                                        with open(pkl_path, 'wb') as f:
                                            pickle.dump(data, f)
                                        logger.info(f"Exported {display_name} to project")
                                    saved_models[data_key] = str(pkl_path)
                        except Exception as e:
                            logger.warning(f"Could not save {display_name}: {e}")

                    # Save variogram results (only if file doesn't exist)
                    variogram_data = registry.get_variogram_results(copy_data=False)
                    if variogram_data is not None:
                        variogram_path = models_folder / "variogram_results.pkl"
                        if not variogram_path.exists():
                            with open(variogram_path, 'wb') as f:
                                pickle.dump(variogram_data, f)
                            logger.info("Exported variogram results to project")
                        saved_models['variogram_results'] = str(variogram_path)

                    # Save drillhole validation state (only if file doesn't exist)
                    validation_state = registry.get_drillholes_validation_state()
                    if validation_state is not None:
                        validation_path = models_folder / "drillholes_validation_state.pkl"
                        if not validation_path.exists():
                            with open(validation_path, 'wb') as f:
                                pickle.dump(validation_state, f)
                            logger.info("Exported drillhole validation state to project")
                        saved_models['drillholes_validation_state'] = str(validation_path)

                    # Save declustering results (only if file doesn't exist)
                    declustering_data = registry.get_declustering_results(copy_data=False)
                    if declustering_data is not None:
                        declustering_path = models_folder / "declustering_results.pkl"
                        if not declustering_path.exists():
                            with open(declustering_path, 'wb') as f:
                                pickle.dump(declustering_data, f)
                            logger.info("Exported declustering results to project")
                        saved_models['declustering_results'] = str(declustering_path)

                    # Save transformation metadata (only if file doesn't exist)
                    transform_data = registry.get_transformation_metadata(copy_data=False)
                    if transform_data is not None:
                        transform_path = models_folder / "transformation_metadata.pkl"
                        if not transform_path.exists():
                            with open(transform_path, 'wb') as f:
                                pickle.dump(transform_data, f)
                            logger.info("Exported transformation metadata to project")
                        saved_models['transformation_metadata'] = str(transform_path)

                    # Save transformers (special case - not in base class storage) (only if file doesn't exist)
                    transformers = registry.get_transformers()
                    if transformers is not None:
                        transformers_path = models_folder / "transformers.pkl"
                        if not transformers_path.exists():
                            with open(transformers_path, 'wb') as f:
                                pickle.dump(transformers, f)
                            logger.info("Exported transformers to project")
                        saved_models['transformers'] = str(transformers_path)

                    # Save kriging results (only if file doesn't exist)
                    kriging_data = registry.get_kriging_results(copy_data=False)
                    if kriging_data is not None:
                        kriging_path = models_folder / "kriging_results.pkl"
                        if not kriging_path.exists():
                            with open(kriging_path, 'wb') as f:
                                pickle.dump(kriging_data, f)
                            logger.info("Exported kriging results to project")
                        saved_models['kriging_results'] = str(kriging_path)

                    # Save SGSIM results (only if file doesn't exist)
                    sgsim_data = registry.get_sgsim_results(copy_data=False)
                    if sgsim_data is not None:
                        sgsim_path = models_folder / "sgsim_results.pkl"
                        if not sgsim_path.exists():
                            with open(sgsim_path, 'wb') as f:
                                pickle.dump(sgsim_data, f)
                            logger.info("Exported SGSIM results to project")
                        saved_models['sgsim_results'] = str(sgsim_path)

                    # Save simple kriging results (only if file doesn't exist)
                    simple_kriging_data = registry.get_simple_kriging_results(copy_data=False)
                    if simple_kriging_data is not None:
                        simple_kriging_path = models_folder / "simple_kriging_results.pkl"
                        if not simple_kriging_path.exists():
                            with open(simple_kriging_path, 'wb') as f:
                                pickle.dump(simple_kriging_data, f)
                            logger.info("Exported simple kriging results to project")
                        saved_models['simple_kriging_results'] = str(simple_kriging_path)

                    # Save co-kriging results (only if file doesn't exist)
                    cokriging_data = registry.get_cokriging_results(copy_data=False)
                    if cokriging_data is not None:
                        cokriging_path = models_folder / "cokriging_results.pkl"
                        if not cokriging_path.exists():
                            with open(cokriging_path, 'wb') as f:
                                pickle.dump(cokriging_data, f)
                            logger.info("Exported co-kriging results to project")
                        saved_models['cokriging_results'] = str(cokriging_path)

                    # Save indicator kriging results (only if file doesn't exist)
                    indicator_kriging_data = registry.get_indicator_kriging_results(copy_data=False)
                    if indicator_kriging_data is not None:
                        indicator_kriging_path = models_folder / "indicator_kriging_results.pkl"
                        if not indicator_kriging_path.exists():
                            with open(indicator_kriging_path, 'wb') as f:
                                pickle.dump(indicator_kriging_data, f)
                            logger.info("Exported indicator kriging results to project")
                        saved_models['indicator_kriging_results'] = str(indicator_kriging_path)

                    # Save universal kriging results (only if file doesn't exist)
                    universal_kriging_data = registry.get_universal_kriging_results(copy_data=False)
                    if universal_kriging_data is not None:
                        universal_kriging_path = models_folder / "universal_kriging_results.pkl"
                        if not universal_kriging_path.exists():
                            with open(universal_kriging_path, 'wb') as f:
                                pickle.dump(universal_kriging_data, f)
                            logger.info("Exported universal kriging results to project")
                        saved_models['universal_kriging_results'] = str(universal_kriging_path)

                    # Save soft kriging results (only if file doesn't exist)
                    soft_kriging_data = registry.get_soft_kriging_results(copy_data=False)
                    if soft_kriging_data is not None:
                        soft_kriging_path = models_folder / "soft_kriging_results.pkl"
                        if not soft_kriging_path.exists():
                            with open(soft_kriging_path, 'wb') as f:
                                pickle.dump(soft_kriging_data, f)
                            logger.info("Exported soft kriging results to project")
                        saved_models['soft_kriging_results'] = str(soft_kriging_path)

                    # Save domain model (only if file doesn't exist)
                    domain_model = registry.get_domain_model(copy_data=False)
                    if domain_model is not None:
                        domain_path = models_folder / "domain_model.pkl"
                        if not domain_path.exists():
                            with open(domain_path, 'wb') as f:
                                pickle.dump(domain_model, f)
                            logger.info("Exported domain model to project")
                        saved_models['domain_model'] = str(domain_path)

                    # Save contact set (only if file doesn't exist)
                    contact_set = registry.get_contact_set(copy_data=False)
                    if contact_set is not None:
                        contact_path = models_folder / "contact_set.pkl"
                        if not contact_path.exists():
                            with open(contact_path, 'wb') as f:
                                pickle.dump(contact_set, f)
                            logger.info("Exported contact set to project")
                        saved_models['contact_set'] = str(contact_path)

                    # Save all registered block models (multi-model support)
                    all_models = registry.get_block_model_list()
                    for model_info in all_models:
                        model_id = model_info['model_id']
                        model = registry.get_block_model(model_id=model_id, copy_data=False)

                        if model is not None:
                            try:
                                # Filename: block_model_{model_id}.csv
                                filename = f"block_model_{model_id}.csv"
                                model_path = models_folder / filename

                                # Convert to DataFrame if needed
                                if hasattr(model, 'to_dataframe'):
                                    df = model.to_dataframe()
                                elif isinstance(model, pd.DataFrame):
                                    df = model
                                else:
                                    # Fallback to pickle
                                    filename = f"block_model_{model_id}.pkl"
                                    model_path = models_folder / filename
                                    if not model_path.exists():
                                        with open(model_path, 'wb') as f:
                                            pickle.dump(model, f)
                                        logger.info(f"Exported block model '{model_id}' (pickle) to project")
                                    saved_models[f'block_model_{model_id}'] = str(model_path)
                                    continue

                                # Export CSV
                                if not model_path.exists():
                                    from ..utils.export_helpers import export_dataframe_to_csv
                                    export_dataframe_to_csv(df, model_path, process_events=False)
                                    logger.info(f"Exported block model '{model_id}' to project")
                                saved_models[f'block_model_{model_id}'] = str(model_path)
                            except Exception as e:
                                logger.warning(f"Could not save block model '{model_id}': {e}")

                    # Save current model pointer
                    current_id = registry.get_current_block_model_id()
                    if current_id:
                        saved_models['current_block_model_id'] = current_id

                    # Backward compat: Also save current model as "block_model.csv"
                    if current_id:
                        current_model = registry.get_block_model(copy_data=False)
                        if current_model is not None:
                            try:
                                if hasattr(current_model, 'to_dataframe'):
                                    df = current_model.to_dataframe()
                                elif isinstance(current_model, pd.DataFrame):
                                    df = current_model
                                else:
                                    df = None

                                if df is not None:
                                    legacy_path = models_folder / "block_model.csv"
                                    if not legacy_path.exists():
                                        from ..utils.export_helpers import export_dataframe_to_csv
                                        export_dataframe_to_csv(df, legacy_path, process_events=False)
                                        logger.info("Exported current block model as legacy 'block_model.csv'")
                                    saved_models['block_model'] = str(legacy_path)
                            except Exception:
                                pass

                    # Save classified block model (only if file doesn't exist)
                    classified_model = registry.get_classified_block_model(copy_data=False)
                    if classified_model is not None:
                        try:
                            if hasattr(classified_model, 'to_dataframe'):
                                classified_path = models_folder / "classified_block_model.csv"
                                if not classified_path.exists():
                                    df = classified_model.to_dataframe()
                                    from ..utils.export_helpers import export_dataframe_to_csv
                                    export_dataframe_to_csv(df, classified_path, process_events=False)
                                    logger.info("Exported classified block model to project")
                                saved_models['classified_block_model'] = str(classified_path)
                            elif isinstance(classified_model, pd.DataFrame):
                                classified_path = models_folder / "classified_block_model.csv"
                                if not classified_path.exists():
                                    from ..utils.export_helpers import export_dataframe_to_csv
                                    export_dataframe_to_csv(classified_model, classified_path, process_events=False)
                                    logger.info("Exported classified block model DataFrame to project")
                                saved_models['classified_block_model'] = str(classified_path)
                            else:
                                classified_path = models_folder / "classified_block_model.pkl"
                                if not classified_path.exists():
                                    with open(classified_path, 'wb') as f:
                                        pickle.dump(classified_model, f)
                                    logger.info("Exported classified block model (pickle) to project")
                                saved_models['classified_block_model'] = str(classified_path)
                        except Exception as e:
                            logger.warning(f"Could not save classified block model: {e}")

                    # Save resource summary (only if file doesn't exist)
                    resource_summary = registry.get_resource_summary(copy_data=False)
                    if resource_summary is not None:
                        resource_path = models_folder / "resource_summary.pkl"
                        if not resource_path.exists():
                            with open(resource_path, 'wb') as f:
                                pickle.dump(resource_summary, f)
                            logger.info("Exported resource summary to project")
                        saved_models['resource_summary'] = str(resource_path)

                    # Save geomet results (only if file doesn't exist)
                    geomet_data = registry.get_geomet_results(copy_data=False)
                    if geomet_data is not None:
                        geomet_path = models_folder / "geomet_results.pkl"
                        if not geomet_path.exists():
                            with open(geomet_path, 'wb') as f:
                                pickle.dump(geomet_data, f)
                            logger.info("Exported geomet results to project")
                        saved_models['geomet_results'] = str(geomet_path)

                    # Save geomet ore types (only if file doesn't exist)
                    geomet_ore_types = registry.get_geomet_ore_types(copy_data=False)
                    if geomet_ore_types is not None:
                        geomet_ore_path = models_folder / "geomet_ore_types.pkl"
                        if not geomet_ore_path.exists():
                            with open(geomet_ore_path, 'wb') as f:
                                pickle.dump(geomet_ore_types, f)
                            logger.info("Exported geomet ore types to project")
                        saved_models['geomet_ore_types'] = str(geomet_ore_path)

                    # Save pit optimization results (only if file doesn't exist)
                    pit_data = registry.get_pit_optimization_results(copy_data=False)
                    if pit_data is not None:
                        pit_path = models_folder / "pit_optimization_results.pkl"
                        if not pit_path.exists():
                            with open(pit_path, 'wb') as f:
                                pickle.dump(pit_data, f)
                            logger.info("Exported pit optimization results to project")
                        saved_models['pit_optimization_results'] = str(pit_path)

                    # Save schedule (only if file doesn't exist)
                    schedule_data = registry.get_schedule(copy_data=False)
                    if schedule_data is not None:
                        schedule_path = models_folder / "schedule.pkl"
                        if not schedule_path.exists():
                            with open(schedule_path, 'wb') as f:
                                pickle.dump(schedule_data, f)
                            logger.info("Exported schedule to project")
                        saved_models['schedule'] = str(schedule_path)

                    # Save IRR results (only if file doesn't exist)
                    irr_data = registry.get_irr_results(copy_data=False)
                    if irr_data is not None:
                        irr_path = models_folder / "irr_results.pkl"
                        if not irr_path.exists():
                            with open(irr_path, 'wb') as f:
                                pickle.dump(irr_data, f)
                            logger.info("Exported IRR results to project")
                        saved_models['irr_results'] = str(irr_path)

                    # Save reconciliation results (only if file doesn't exist)
                    reconciliation_data = registry.get_reconciliation_results(copy_data=False)
                    if reconciliation_data is not None:
                        reconciliation_path = models_folder / "reconciliation_results.pkl"
                        if not reconciliation_path.exists():
                            with open(reconciliation_path, 'wb') as f:
                                pickle.dump(reconciliation_data, f)
                            logger.info("Exported reconciliation results to project")
                        saved_models['reconciliation_results'] = str(reconciliation_path)

                    # Save haulage evaluation (only if file doesn't exist)
                    haulage_data = registry.get_haulage_evaluation(copy_data=False)
                    if haulage_data is not None:
                        haulage_path = models_folder / "haulage_evaluation.pkl"
                        if not haulage_path.exists():
                            with open(haulage_path, 'wb') as f:
                                pickle.dump(haulage_data, f)
                            logger.info("Exported haulage evaluation to project")
                        saved_models['haulage_evaluation'] = str(haulage_path)

                    # Save experiment results (only if file doesn't exist)
                    experiment_data = registry.get_experiment_results(copy_data=False)
                    if experiment_data is not None:
                        experiment_path = models_folder / "experiment_results.pkl"
                        if not experiment_path.exists():
                            with open(experiment_path, 'wb') as f:
                                pickle.dump(experiment_data, f)
                            logger.info("Exported experiment results to project")
                        saved_models['experiment_results'] = str(experiment_path)

                    # Save category label maps (only if file doesn't exist)
                    category_label_maps = registry.get_category_label_maps()
                    if category_label_maps:
                        label_maps_path = models_folder / "category_label_maps.pkl"
                        if not label_maps_path.exists():
                            with open(label_maps_path, 'wb') as f:
                                pickle.dump(category_label_maps, f)
                            logger.info(f"Exported {len(category_label_maps)} category label namespaces to project")
                        saved_models['category_label_maps'] = str(label_maps_path)

                    if saved_models:
                        registry_models_state = saved_models
                        logger.info(f"Saved {len(saved_models)} models/results to project: {', '.join(saved_models.keys())}")
        except Exception as e:
            logger.error(f"Error saving DataRegistry models: {e}", exc_info=True)

        try:
            renderer_state = self.viewer_widget.renderer.get_session_state() if (self.viewer_widget and self.viewer_widget.renderer) else {}
        except Exception:
            renderer_state = {}

        # Save process history
        process_history_state = None
        try:
            process_tracker = get_process_history_tracker()
            process_history_state = process_tracker.to_dict()
            logger.info(f"Saved process history: {process_tracker.get_process_count()} processes")
        except Exception as e:
            logger.warning(f"Could not save process history: {e}")

        # Save compositing settings (if compositing window is open/initialized)
        compositing_settings = None
        try:
            if hasattr(self, 'compositing_window') and self.compositing_window is not None:
                if hasattr(self.compositing_window, 'get_settings_state'):
                    compositing_settings = self.compositing_window.get_settings_state()
                    if compositing_settings:
                        logger.info("Saved compositing settings to project")
        except Exception as e:
            logger.warning(f"Could not save compositing settings: {e}")

        # Save panel settings from all registered panels
        panel_settings = {}
        try:
            from .panel_manager import get_panel_manager
            panel_manager = get_panel_manager()
            if panel_manager:
                for panel_info in panel_manager.get_all_panels():
                    panel_id = panel_info.panel_id
                    panel_instance = panel_manager.get_panel_instance(panel_id)
                    if panel_instance and hasattr(panel_instance, 'get_panel_settings'):
                        try:
                            settings = panel_instance.get_panel_settings()
                            if settings:
                                panel_settings[panel_id] = settings
                                logger.debug(f"Saved settings for panel: {panel_id}")
                        except Exception as e:
                            logger.warning(f"Could not save settings for panel {panel_id}: {e}")
        except Exception as e:
            logger.warning(f"Could not collect panel settings: {e}")

        return {
            'version': 1,
            'app': 'BlockModelViewer',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'data': data_files,
            'drillhole_data': drillhole_state,
            'registry_models': registry_models_state,
            'renderer_state': renderer_state,
            'bookmarks': self.bookmarks.get_bookmarks() if self.bookmarks is not None else self.view_bookmarks,
            'process_history': process_history_state,
            'compositing_settings': compositing_settings,
            'panel_settings': panel_settings if panel_settings else None,
        }

    def _restore_drillhole_data(self, drillhole_state: Dict[str, Any]):
        """Restore drillhole data from saved project state."""
        try:
            if not drillhole_state:
                return

            # Handle old format (composites_file) and new format (composites, assays, etc.)
            composites_file = drillhole_state.get('composites_file') or drillhole_state.get('composites')
            assays_file = drillhole_state.get('assays')
            collars_file = drillhole_state.get('collars')
            surveys_file = drillhole_state.get('surveys')
            trajectories_file = drillhole_state.get('trajectories')

            # Restore composites (required for old format compatibility)
            composites_df = None
            if composites_file:
                if not Path(composites_file).exists():
                    logger.warning(f"Drillhole composites file not found: {composites_file}")
                    return

                # Load composites DataFrame
                import pandas as pd
                composites_df = pd.read_csv(composites_file)
                logger.info(f"Loaded {len(composites_df)} composites from {composites_file}")

            # Load other drillhole data files
            assays_df = None
            if assays_file and Path(assays_file).exists():
                import pandas as pd
                assays_df = pd.read_csv(assays_file)
                logger.info(f"Loaded {len(assays_df)} assays from {assays_file}")

            collars_df = None
            if collars_file and Path(collars_file).exists():
                import pandas as pd
                collars_df = pd.read_csv(collars_file)
                logger.info(f"Loaded {len(collars_df)} collars from {collars_file}")

            surveys_df = None
            if surveys_file and Path(surveys_file).exists():
                import pandas as pd
                surveys_df = pd.read_csv(surveys_file)
                logger.info(f"Loaded {len(surveys_df)} surveys from {surveys_file}")

            trajectories_df = None
            if trajectories_file and Path(trajectories_file).exists():
                import pandas as pd
                trajectories_df = pd.read_csv(trajectories_file)
                logger.info(f"Loaded {len(trajectories_df)} trajectories from {trajectories_file}")

            # Store loaded data in compositing panel if available
            if hasattr(self, 'domain_compositing_panel') and self.domain_compositing_panel:
                if composites_df is not None:
                    self.domain_compositing_panel.comp_domain_df = composites_df
                    logger.info("Restored composites to compositing panel")
                if assays_df is not None:
                    self.domain_compositing_panel.assay_xyz = assays_df
                    logger.info("Restored assays to compositing panel")
                if collars_df is not None:
                    self.domain_compositing_panel.collar_df = collars_df
                    logger.info("Restored collars to drillhole loading panel")
                if surveys_df is not None:
                    self.domain_compositing_panel.survey_df = surveys_df
                    logger.info("Restored surveys to compositing panel")

            # Load lithology if present
            lithology_file = drillhole_state.get('lithology')
            lithology_df = None
            if lithology_file and Path(lithology_file).exists():
                import pandas as pd
                lithology_df = pd.read_csv(lithology_file)
                logger.info(f"Loaded {len(lithology_df)} lithology records from {lithology_file}")

            # Register drillhole data with DataRegistry using proper method
            try:
                from ..core.data_registry import DataRegistry
                if hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None:
                    registry = DataRegistry._instance
                    drillhole_data = {}
                    if assays_df is not None:
                        drillhole_data['assays'] = assays_df
                    if collars_df is not None:
                        drillhole_data['collars'] = collars_df
                    if surveys_df is not None:
                        drillhole_data['surveys'] = surveys_df
                    if trajectories_df is not None:
                        drillhole_data['trajectories'] = trajectories_df
                    if composites_df is not None:
                        drillhole_data['composites'] = composites_df
                    if lithology_df is not None:
                        drillhole_data['lithology'] = lithology_df

                    if drillhole_data:
                        # Use proper registration method to store in _data_store
                        registry.register_drillhole_data(drillhole_data, source_panel="ProjectRestore")
                        logger.info("Restored drillhole data to DataRegistry")
            except Exception as reg_error:
                try:
                    reg_msg = str(reg_error)
                    logger.debug(f"Could not restore to DataRegistry: {reg_msg}")
                except Exception:
                    logger.debug("Could not restore to DataRegistry: <unprintable error>")

            # Data has been restored to DataRegistry above
            # Do NOT render drillholes here - let user click "Plot 3D" in Drillhole Control Panel
            # This avoids duplicate drillhole layers (one from restore, one from Plot 3D)
            # The Plot 3D button uses add_drillhole_layer() which has proper lithology coloring

            n_composites = len(composites_df) if composites_df is not None else 0
            n_assays = len(assays_df) if assays_df is not None else 0
            logger.info(f"Drillhole data restored to registry: {n_composites} composites, {n_assays} assays")
            logger.info("Click 'Plot 3D' in Drillhole Control Panel to visualize")
            self.statusBar().showMessage("Drillhole data loaded - click 'Plot 3D' to visualize", 5000)

        except Exception as e:
            logger.error(f"Error restoring drillhole data: {e}", exc_info=True)

    def _restore_compositing_settings(self, compositing_settings: Dict[str, Any]):
        """Restore compositing window settings from saved project state.
        
        This method initializes the compositing window (if needed) and applies
        the saved settings so that compositing parameters are preserved across
        project save/load cycles.
        
        Args:
            compositing_settings: Dictionary containing compositing tab settings
        """
        try:
            if not compositing_settings:
                logger.debug("No compositing settings to restore")
                return

            # If the compositing window is already open, apply settings directly
            if hasattr(self, 'compositing_window') and self.compositing_window is not None:
                if hasattr(self.compositing_window, 'apply_settings_state'):
                    self.compositing_window.apply_settings_state(compositing_settings)
                    logger.info("Applied compositing settings to existing window")
                return

            # Store settings for later application when compositing window is opened
            self._saved_compositing_settings = compositing_settings
            logger.info("Stored compositing settings for deferred application")

        except Exception as e:
            logger.warning(f"Error restoring compositing settings: {e}")

    def _restore_panel_settings(self, panel_settings: Dict[str, Dict[str, Any]]):
        """Restore settings for all panels from saved project state.
        
        This method applies saved settings to panels when they are opened.
        Settings are stored and applied when panels are initialized.
        
        Args:
            panel_settings: Dictionary mapping panel_id to settings dict
        """
        try:
            if not panel_settings:
                return

            # Store settings for deferred application (panels may not be open yet)
            if not hasattr(self, '_saved_panel_settings'):
                self._saved_panel_settings = {}

            self._saved_panel_settings.update(panel_settings)
            logger.info(f"Stored settings for {len(panel_settings)} panels for deferred application")

            # Try to apply settings to panels that are already open
            from .panel_manager import get_panel_manager
            panel_manager = get_panel_manager()
            if panel_manager:
                for panel_id, settings in panel_settings.items():
                    panel_instance = panel_manager.get_panel_instance(panel_id)
                    if panel_instance and hasattr(panel_instance, 'apply_panel_settings'):
                        try:
                            panel_instance.apply_panel_settings(settings)
                            logger.debug(f"Applied settings to open panel: {panel_id}")
                        except Exception as e:
                            logger.warning(f"Could not apply settings to panel {panel_id}: {e}")

        except Exception as e:
            logger.warning(f"Error restoring panel settings: {e}")

    def _restore_registry_models(self, registry_models_state: Dict[str, Any]):
        """Restore DataRegistry models from saved project state.
        
        This method restores all saved models (variogram, kriging, SGSIM, resource,
        classification, etc.) back into the DataRegistry when a project is loaded.
        """
        try:
            if not registry_models_state:
                logger.debug("No registry models to restore")
                return

            import pickle

            import pandas as pd

            from ..core.data_registry import DataRegistry

            if not hasattr(DataRegistry, '_instance') or DataRegistry._instance is None:
                logger.warning("DataRegistry not available for restoring models")
                return

            registry = DataRegistry._instance
            restored_count = 0

            # Helper function to load a pickle file
            def load_pickle(file_path: str) -> Any:
                """Load data from a pickle file with security validation.
                
                SECURITY: Validates path and file size before loading.
                WARNING: Pickle can execute arbitrary code. Only load trusted files.
                """
                from ..utils.security import SecurityError, validate_pickle_file

                try:
                    validated_path, file_size = validate_pickle_file(
                        Path(file_path),
                        allowed_base=None  # Project files are in project directory
                    )
                    logger.debug(f"Loading pickle file: {validated_path} ({file_size} bytes)")
                    with open(validated_path, 'rb') as f:
                        return pickle.load(f)
                except SecurityError as e:
                    logger.error(f"Security error loading pickle file {file_path}: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Error loading pickle file {file_path}: {e}")
                    return None

            # Helper function to load a CSV file
            def load_csv(file_path: str) -> Optional[pd.DataFrame]:
                """Load data from a CSV file."""
                if not Path(file_path).exists():
                    logger.warning(f"File not found: {file_path}")
                    return None
                return pd.read_csv(file_path)

            # Restore variogram results
            if 'variogram_results' in registry_models_state:
                data = load_pickle(registry_models_state['variogram_results'])
                if data is not None:
                    registry.register_variogram_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored variogram results from project")

            # Restore drillhole validation state
            if 'drillholes_validation_state' in registry_models_state:
                data = load_pickle(registry_models_state['drillholes_validation_state'])
                if data is not None and isinstance(data, dict):
                    # Unpack the validation state dict and pass as kwargs
                    registry.set_drillholes_validation_state(
                        status=data.get('status', 'NOT_RUN'),
                        timestamp=data.get('timestamp', ''),
                        config_hash=data.get('config_hash', ''),
                        fatal_count=data.get('fatal_count', 0),
                        warn_count=data.get('warn_count', 0),
                        info_count=data.get('info_count', 0),
                        violations_summary=data.get('violations_summary'),
                        tables_validated=data.get('tables_validated'),
                        schema_errors=data.get('schema_errors'),
                        excluded_rows=data.get('excluded_rows')
                    )
                    restored_count += 1
                    logger.info(f"Restored drillhole validation state from project (status: {data.get('status', 'UNKNOWN')})")

            # Restore declustering results
            if 'declustering_results' in registry_models_state:
                data = load_pickle(registry_models_state['declustering_results'])
                if data is not None:
                    # Declustering results are stored as dict with 'weighted_dataframe' and 'summary'
                    if isinstance(data, dict) and 'weighted_dataframe' in data and 'summary' in data:
                        registry.register_declustering_results(
                            (data['weighted_dataframe'], data['summary']),
                            source_panel="ProjectRestore"
                        )
                    else:
                        # Legacy format
                        registry.register_model('declustering_results', data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored declustering results from project")

            # Restore transformation metadata
            if 'transformation_metadata' in registry_models_state:
                data = load_pickle(registry_models_state['transformation_metadata'])
                if data is not None:
                    registry.register_transformation_metadata(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored transformation metadata from project")

            # Restore transformers (special case)
            if 'transformers' in registry_models_state:
                data = load_pickle(registry_models_state['transformers'])
                if data is not None:
                    registry.register_transformers(data)
                    restored_count += 1
                    logger.info("Restored transformers from project")

            # Restore kriging results
            if 'kriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['kriging_results'])
                if data is not None:
                    registry.register_kriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored kriging results from project")

            # Restore SGSIM results
            if 'sgsim_results' in registry_models_state:
                data = load_pickle(registry_models_state['sgsim_results'])
                if data is not None:
                    registry.register_sgsim_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored SGSIM results from project")

            # Restore simple kriging results
            if 'simple_kriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['simple_kriging_results'])
                if data is not None:
                    registry.register_simple_kriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored simple kriging results from project")

            # Restore co-kriging results
            if 'cokriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['cokriging_results'])
                if data is not None:
                    registry.register_cokriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored co-kriging results from project")

            # Restore indicator kriging results
            if 'indicator_kriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['indicator_kriging_results'])
                if data is not None:
                    registry.register_indicator_kriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored indicator kriging results from project")

            # Restore universal kriging results
            if 'universal_kriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['universal_kriging_results'])
                if data is not None:
                    registry.register_universal_kriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored universal kriging results from project")

            # Restore soft kriging results
            if 'soft_kriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['soft_kriging_results'])
                if data is not None:
                    registry.register_soft_kriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored soft kriging results from project")

            # Restore domain model
            if 'domain_model' in registry_models_state:
                data = load_pickle(registry_models_state['domain_model'])
                if data is not None:
                    registry.register_domain_model(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored domain model from project")

            # Restore contact set
            if 'contact_set' in registry_models_state:
                data = load_pickle(registry_models_state['contact_set'])
                if data is not None:
                    registry.register_contact_set(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored contact set from project")

            # Restore all block models (multi-model support)
            block_models_restored = 0
            current_model_id = None

            # Check if this is a legacy project (single block_model.csv)
            has_legacy = 'block_model' in registry_models_state
            has_multi = any(k.startswith('block_model_') and k != 'current_block_model_id'
                            for k in registry_models_state)

            if has_multi:
                # New multi-model project
                for key, file_path in registry_models_state.items():
                    if key.startswith('block_model_') and key != 'current_block_model_id':
                        # Extract model_id: "block_model_sgsim_FE_mean" -> "sgsim_FE_mean"
                        model_id = key[len('block_model_'):]

                        # Load data
                        try:
                            if file_path.endswith('.csv'):
                                data = load_csv(file_path)
                            else:
                                data = load_pickle(file_path)

                            if data is not None:
                                registry.register_block_model(
                                    data,
                                    source_panel="ProjectRestore",
                                    model_id=model_id,
                                    set_as_current=False
                                )
                                block_models_restored += 1
                                logger.info(f"Restored block model '{model_id}' from project")
                        except Exception as e:
                            logger.warning(f"Could not restore block model '{model_id}': {e}")

                # Restore current model pointer
                if 'current_block_model_id' in registry_models_state:
                    current_model_id = registry_models_state['current_block_model_id']
                    if registry.set_current_block_model(current_model_id):
                        logger.info(f"Set current block model to '{current_model_id}'")
                elif block_models_restored > 0:
                    # No saved current model - set first restored model as current
                    first_model_id = registry.get_block_model_list()[0]['model_id']
                    registry.set_current_block_model(first_model_id)
                    logger.info(f"Set first restored model '{first_model_id}' as current (no saved current)")

            elif has_legacy and not registry.has_block_model():
                # Legacy project - auto-migrate as "default" model
                try:
                    file_path = registry_models_state['block_model']
                    if file_path.endswith('.csv'):
                        data = load_csv(file_path)
                    else:
                        data = load_pickle(file_path)

                    if data is not None:
                        registry.register_block_model(
                            data,
                            source_panel="ProjectRestore",
                            model_id="default",
                            set_as_current=True
                        )
                        block_models_restored = 1
                        logger.info("Restored legacy block model as 'default'")
                except Exception as e:
                    logger.warning(f"Could not restore legacy block model: {e}")

            if block_models_restored > 0:
                restored_count += block_models_restored
                logger.info(f"Restored {block_models_restored} block model(s) from project")

            # Restore classified block model (check for both CSV and pickle)
            if 'classified_block_model' in registry_models_state:
                file_path = registry_models_state['classified_block_model']
                if file_path.endswith('.csv'):
                    data = load_csv(file_path)
                else:
                    data = load_pickle(file_path)
                if data is not None:
                    registry.register_classified_block_model(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored classified block model from project")

            # Restore resource summary
            if 'resource_summary' in registry_models_state:
                data = load_pickle(registry_models_state['resource_summary'])
                if data is not None:
                    registry.register_resource_summary(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored resource summary from project")

            # Restore geomet results
            if 'geomet_results' in registry_models_state:
                data = load_pickle(registry_models_state['geomet_results'])
                if data is not None:
                    registry.register_geomet_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored geomet results from project")

            # Restore geomet ore types
            if 'geomet_ore_types' in registry_models_state:
                data = load_pickle(registry_models_state['geomet_ore_types'])
                if data is not None:
                    registry.register_geomet_ore_types(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored geomet ore types from project")

            # Restore pit optimization results
            if 'pit_optimization_results' in registry_models_state:
                data = load_pickle(registry_models_state['pit_optimization_results'])
                if data is not None:
                    registry.register_pit_optimization_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored pit optimization results from project")

            # Restore schedule
            if 'schedule' in registry_models_state:
                data = load_pickle(registry_models_state['schedule'])
                if data is not None:
                    registry.register_schedule(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored schedule from project")

            # Restore IRR results
            if 'irr_results' in registry_models_state:
                data = load_pickle(registry_models_state['irr_results'])
                if data is not None:
                    registry.register_irr_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored IRR results from project")

            # Restore reconciliation results
            if 'reconciliation_results' in registry_models_state:
                data = load_pickle(registry_models_state['reconciliation_results'])
                if data is not None:
                    registry.register_reconciliation_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored reconciliation results from project")

            # Restore haulage evaluation
            if 'haulage_evaluation' in registry_models_state:
                data = load_pickle(registry_models_state['haulage_evaluation'])
                if data is not None:
                    registry.register_haulage_evaluation(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored haulage evaluation from project")

            # Restore experiment results
            if 'experiment_results' in registry_models_state:
                data = load_pickle(registry_models_state['experiment_results'])
                if data is not None:
                    registry.register_experiment_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored experiment results from project")

            # Restore category label maps
            if 'category_label_maps' in registry_models_state:
                data = load_pickle(registry_models_state['category_label_maps'])
                if data is not None:
                    registry.register_model("category_label_maps", data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info(f"Restored {len(data)} category label namespaces from project")

            if restored_count > 0:
                logger.info(f"Successfully restored {restored_count} models/results from project")
                self.statusBar().showMessage(f"Restored {restored_count} project items", 3000)

        except Exception as e:
            logger.error(f"Error restoring registry models: {e}", exc_info=True)

    def save_project(self):
        """Save the current project to disk. Uses existing path or prompts if none."""
        try:
            if not self.current_project_path:
                return self.save_project_as()

            # Show progress dialog for large saves
            from PyQt6.QtCore import Qt
            from PyQt6.QtWidgets import QProgressDialog
            progress = QProgressDialog("Saving project...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(500)  # Only show if takes > 500ms
            progress.setValue(0)
            QApplication.processEvents()

            try:
                # Collect project state (this does CSV exports)
                progress.setLabelText("Collecting project data...")
                progress.setValue(10)
                QApplication.processEvents()

                state = self._collect_project_state()

                progress.setLabelText("Writing project file...")
                progress.setValue(90)
                QApplication.processEvents()

                # Write JSON project file
                self.current_project_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.current_project_path, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2)

                progress.setValue(100)
                QApplication.processEvents()

            finally:
                progress.close()

            self.statusBar().showMessage(f"Project saved: {self.current_project_path.name}", 3000)
            logger.info(f"Saved project to {self.current_project_path}")
            # Clear dirty flag after successful save
            try:
                self._clear_dirty()
            except Exception:
                pass
            # Remember last project
            try:
                settings = QSettings("GeoX", "Project")
                settings.setValue("last_project", str(self.current_project_path))
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error saving project: {e}")
            QMessageBox.warning(self, "Save Project", f"Failed to save project:\n{e}")

    def save_project_as(self):
        """Prompt for a project filename and save the project."""
        try:
            default_dir = str(self.current_file_path.parent) if self.current_file_path else str(Path.home())
            default_name = (self.current_file_path.stem if self.current_file_path else 'project') + ".bmvproj"
            path_str, _ = QFileDialog.getSaveFileName(self, "Save Project As", str(Path(default_dir) / default_name), "Block Model Viewer Project (*.bmvproj);;JSON (*.json)")
            if not path_str:
                return
            self.current_project_path = Path(path_str)
            self.save_project()
            # Reflect in window title
            try:
                self.setWindowTitle(f"GeoX - {self.current_project_path.stem}")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error in Save Project As: {e}")
            QMessageBox.warning(self, "Save Project As", f"Failed to save project:\n{e}")

    def open_project(self):
        """Open a project file, load data and reapply renderer state."""
        try:
            path_str, _ = QFileDialog.getOpenFileName(self, "Open Project", str(Path.home()), "Block Model Viewer Project (*.bmvproj);;JSON (*.json)")
            if not path_str:
                return
            self._load_project_from_path(Path(path_str))
        except Exception as e:
            logger.error(f"Error opening project: {e}")
            QMessageBox.warning(self, "Open Project", f"Failed to open project:\n{e}")

    def _load_block_model_with_progress(self, file_path: Path, dialog, current_step: int, total_steps: int):
        """
        Load a block model with detailed progress tracking in the loading dialog.

        Args:
            file_path: Path to the block model file
            dialog: ProjectLoadingDialog instance
            current_step: Current step counter
            total_steps: Total steps for progress calculation
        """
        # Add to recent files
        self._add_recent_file(file_path)

        # Use controller task system for file loading with progress callback
        if not self.controller:
            QMessageBox.critical(self, "Error", "Controller not available for file loading")
            return

        params = {
            "file_path": file_path
        }

        def progress_callback(progress_percent: int, message: str):
            """Update loading dialog with file loading progress."""
            dialog.set_detailed_progress(
                current_step,
                total_steps,
                f"Loading block model: {message}"
            )

        def on_load_complete(result: Dict[str, Any]):
            """
            Handle file load completion.

            This callback is already called on the main thread via Qt signals,
            so we can update UI directly without QTimer.singleShot.
            """
            logger.info(f"Block model load complete. Result keys: {list(result.keys()) if result else 'None'}")

            if result is None or result.get("error"):
                error_msg = result.get("error", "Unknown error") if result else "No result"
                logger.error(f"Block model load error: {error_msg}")
                dialog.set_detailed_progress(current_step + 1, total_steps, f"Block model loading failed: {error_msg}")
                return

            block_model = result.get("block_model")
            if block_model is not None:
                block_count = getattr(block_model, "block_count", None)
                if block_count is None:
                    try:
                        block_count = len(block_model)
                    except Exception:
                        block_count = 0
                logger.info(f"Block model loaded: {block_count} blocks")
                # Store file path before calling on_file_loaded
                result_file_path = result.get("file_path")
                if result_file_path:
                    self.current_file_path = Path(result_file_path)
                try:
                    self.on_file_loaded(block_model)
                    dialog.set_detailed_progress(current_step + 1, total_steps, f"Block model loaded: {int(block_count):,} blocks")
                except Exception as e:
                    # Avoid potential recursion in logging by not using exc_info=True
                    logger.error(f"Error in on_file_loaded: {type(e).__name__}: {str(e)}")
                    dialog.set_detailed_progress(current_step + 1, total_steps, f"Block model processing failed: {e}")
            else:
                logger.error("No block model in result")
                dialog.set_detailed_progress(current_step + 1, total_steps, "Block model loading failed: No data")

        # Run the task with progress callback
        self.controller.run_task('load_file', params, callback=on_load_complete, progress_callback=progress_callback)

    def _restore_project_data_without_block_model(self, dialog=None, current_step=None, total_steps=None):
        """Restore project data (drillholes, registry models, renderer state) when no block model is loaded.

        This is called when a project is loaded that either has no block model or the block model file is missing.
        It ensures drillhole data and all registry models are still restored.

        Args:
            dialog: Optional ProjectLoadingDialog for progress updates
            current_step: Current step counter for progress
            total_steps: Total steps for progress calculation
        """
        try:
            # Apply renderer state
            if getattr(self, '_pending_session_state', None):
                if dialog and current_step is not None and total_steps is not None:
                    dialog.set_detailed_progress(current_step, total_steps, "Applying visual settings...")
                try:
                    if self.viewer_widget and self.viewer_widget.renderer:
                        self.viewer_widget.renderer.apply_session_state(self._pending_session_state)
                        self._pending_session_state = None
                        logger.info("Applied renderer session state from project")
                except Exception as e:
                    logger.warning(f"Failed to apply pending session state: {e}")

            # Restore drillhole data
            if getattr(self, '_pending_drillhole_state', None):
                if dialog and current_step is not None and total_steps is not None:
                    dialog.set_detailed_progress(current_step, total_steps, "Restoring drillhole data...")
                try:
                    self._restore_drillhole_data(self._pending_drillhole_state)
                    self._pending_drillhole_state = None
                    logger.info("Restored drillhole data from project")
                except Exception as e:
                    logger.warning(f"Failed to restore drillhole data: {e}")

            # Restore DataRegistry models
            if getattr(self, '_pending_registry_models_state', None):
                if dialog and current_step is not None and total_steps is not None:
                    dialog.set_detailed_progress(current_step, total_steps, "Restoring analysis results...")
                try:
                    self._restore_registry_models(self._pending_registry_models_state)
                    self._pending_registry_models_state = None
                    logger.info("Restored registry models from project")
                except Exception as e:
                    logger.warning(f"Failed to restore registry models: {e}")

        except Exception as e:
            logger.error(f"Error restoring project data without block model: {e}", exc_info=True)

    def _load_project_from_path(self, project_path: Path):
        """
        Load a project from file with detailed progress indication.

        Shows a progress dialog with loading steps and displays the processes
        that were run on the project.
        """

        # Create and show progress dialog
        dialog = ProjectLoadingDialog(project_path.stem, self)
        dialog.show()

        # Set up detailed progress tracking
        total_steps = 8  # More granular steps for better progress indication
        dialog.set_total_steps(total_steps)
        current_step = 0

        try:
            # Step 1: Reading project file
            current_step += 1
            dialog.set_detailed_progress(current_step, total_steps, "Reading project file...")

            with open(project_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.current_project_path = project_path

            # Step 2: Loading project metadata
            current_step += 1
            dialog.set_detailed_progress(current_step, total_steps, "Loading project metadata...")

            # Update window title
            try:
                self.setWindowTitle(f"GeoX - {project_path.stem}")
            except Exception:
                pass

            # Restore bookmarks
            try:
                bookmarks = data.get('bookmarks') or {}
                if isinstance(bookmarks, dict):
                    if self.bookmarks is not None:
                        self.bookmarks.set_bookmarks(bookmarks)
                        self.view_bookmarks = self.bookmarks.bookmarks
                    else:
                        self.view_bookmarks = bookmarks
                        self._persist_bookmarks()
            except Exception:
                pass

            # Step 3: Loading process history
            current_step += 1
            dialog.set_detailed_progress(current_step, total_steps, "Loading process history...")

            try:
                # Restore process history
                process_history_data = data.get('process_history')
                if process_history_data:
                    process_tracker = get_process_history_tracker()
                    process_tracker.from_dict(process_history_data)
                    logger.info("Restored process history from project")

                    # Display processes in the dialog
                    history = process_tracker.get_history()
                    dialog.set_process_history([p.to_dict() for p in history])
                else:
                    # Clear process history for new/legacy projects
                    process_tracker = get_process_history_tracker()
                    process_tracker.clear_history()
                    dialog.set_process_history([])
            except Exception as e:
                logger.warning(f"Could not restore process history: {e}")
                dialog.set_process_history([])

            # Prepare renderer state to apply after data load
            self._pending_session_state = data.get('renderer_state') or None
            self._pending_drillhole_state = data.get('drillhole_data') or None
            self._pending_registry_models_state = data.get('registry_models') or None
            self._pending_compositing_settings = data.get('compositing_settings') or None
            self._pending_panel_settings = data.get('panel_settings') or {}

            # Step 4-6: Loading data components
            data_section = data.get('data') or {}
            bm_path = data_section.get('block_model')

            if bm_path:
                current_step += 1
                dialog.set_detailed_progress(current_step, total_steps, "Preparing block model data...")

                p = Path(bm_path)
                if p.exists():
                    current_step += 1
                    dialog.set_detailed_progress(current_step, total_steps, "Loading block model file...")

                    # Load block model with progress tracking
                    self._load_block_model_with_progress(p, dialog, current_step, total_steps)
                    current_step += 1  # Increment after block model loading completes
                else:
                    QMessageBox.warning(self, "Open Project", f"Block model file not found:\n{p}")
                    # Still restore other data even if block model not found
                    current_step += 2
                    dialog.set_detailed_progress(current_step, total_steps, "Loading project data (block model missing)...")
                    self._restore_project_data_without_block_model(dialog, current_step, total_steps)
            else:
                # No block model - restore drillhole data, registry models, and renderer state directly
                current_step += 1
                dialog.set_detailed_progress(current_step, total_steps, "Loading drillhole data...")
                current_step += 1
                dialog.set_detailed_progress(current_step, total_steps, "Loading registry models...")
                self._restore_project_data_without_block_model(dialog, current_step, total_steps)

            # Step 7: Applying renderer state
            current_step += 1
            dialog.set_detailed_progress(current_step, total_steps, "Applying visual settings...")

            # Apply any pending renderer state
            if self._pending_session_state:
                try:
                    if self.viewer_widget and self.viewer_widget.renderer:
                        self.viewer_widget.renderer.apply_session_state(self._pending_session_state)
                        logger.info("Applied saved renderer state")
                except Exception as e:
                    logger.warning(f"Could not apply renderer state: {e}")

            # Step 8: Finalizing project load
            current_step += 1
            dialog.set_detailed_progress(current_step, total_steps, "Finalizing project load...")

            # Restore compositing settings if saved
            if self._pending_compositing_settings:
                try:
                    self._restore_compositing_settings(self._pending_compositing_settings)
                    logger.info("Restored compositing settings from project")
                except Exception as e:
                    logger.warning(f"Could not restore compositing settings: {e}")
                finally:
                    self._pending_compositing_settings = None

            # Restore panel settings from all panels
            if self._pending_panel_settings:
                try:
                    self._restore_panel_settings(self._pending_panel_settings)
                    logger.info(f"Restored settings for {len(self._pending_panel_settings)} panels")
                except Exception as e:
                    logger.warning(f"Could not restore panel settings: {e}")
                finally:
                    self._pending_panel_settings = {}

            # Remember last project
            try:
                settings = QSettings("GeoX", "Project")
                settings.setValue("last_project", str(project_path))
            except Exception:
                pass

            self.statusBar().showMessage(f"Project loaded: {project_path.name}", 3000)
            logger.info(f"Loaded project from {project_path}")

            # Auto-show process history panel if there are processes
            try:
                process_tracker = get_process_history_tracker()
                if process_tracker.get_process_count() > 0:
                    from .panel_manager import get_panel_manager
                    panel_manager = get_panel_manager()
                    panel_manager.show_panel("ProcessHistoryPanel")
                    logger.info("Auto-showing process history panel with saved processes")
            except Exception as e:
                logger.debug(f"Could not auto-show process history panel: {e}")

            # Loaded project should start clean
            try:
                self._clear_dirty()
            except Exception:
                pass

            # Complete the loading
            dialog.complete_loading()

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            dialog.close()
            QMessageBox.warning(self, "Open Project", f"Failed to load project:\n{e}")

    def _new_project(self):
        """Start a new project by clearing scene and resetting project path."""
        try:
            self.clear_scene()
            self.current_project_path = None
            self.statusBar().showMessage("New project created", 2000)
            self.setWindowTitle("GeoX")
            try:
                self._clear_dirty()
            except Exception:
                pass
        except Exception as e:
            try:
                e_msg = str(e)
                logger.debug(f"New project error: {e_msg}")
            except Exception:
                logger.debug("New project error: <unprintable error>")

    def _undo(self):
        try:
            if not hasattr(self, '_undo_stack') or not self._undo_stack:
                return
            if not self.viewer_widget or not self.viewer_widget.renderer:
                return
            current = self.viewer_widget.renderer.get_session_state()
            prev = self._undo_stack.pop()
            # Apply previous state and push current to redo
            self.viewer_widget.renderer.apply_session_state(prev)
            if not hasattr(self, '_redo_stack'):
                self._redo_stack = []
            self._redo_stack.append(current)
            # Update UI
            self.undo_action.setEnabled(len(self._undo_stack) > 0)
            self.redo_action.setEnabled(len(self._redo_stack) > 0)
            self.statusBar().showMessage("Undid last change", 1500)
        except Exception as e:
            logger.warning(f"Undo failed: {e}")

    def _redo(self):
        try:
            if not hasattr(self, '_redo_stack') or not self._redo_stack:
                return
            if not self.viewer_widget or not self.viewer_widget.renderer:
                return
            current = self.viewer_widget.renderer.get_session_state()
            nxt = self._redo_stack.pop()
            # Apply next state and push current to undo
            self.viewer_widget.renderer.apply_session_state(nxt)
            if not hasattr(self, '_undo_stack'):
                self._undo_stack = []
            self._undo_stack.append(current)
            # Update UI
            self.undo_action.setEnabled(len(self._undo_stack) > 0)
            self.redo_action.setEnabled(len(self._redo_stack) > 0)
            self.statusBar().showMessage("Redid last change", 1500)
        except Exception as e:
            logger.warning(f"Redo failed: {e}")

    # ================== STEP 39: WORKBENCH & WORKFLOWS ==================

    def apply_workbench_profile(self, profile_id: str) -> None:
        """Apply a workbench profile (STEP 39)."""
        try:
            from .workbench_manager import WorkbenchManager

            if not hasattr(self, 'workbench_manager'):
                self.workbench_manager = WorkbenchManager(self)

            self.workbench_manager.apply_profile(profile_id)
            self.statusBar().showMessage(f"Applied workbench profile: {profile_id}", 2000)
        except Exception as e:
            logger.error(f"Failed to apply workbench profile: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to apply workbench profile:\n{e}")

    def save_current_layout(self) -> None:
        """Save current layout as a profile (STEP 39)."""
        try:
            from PyQt6.QtWidgets import QInputDialog

            from .workbench_manager import WorkbenchManager

            if not hasattr(self, 'workbench_manager'):
                self.workbench_manager = WorkbenchManager(self)

            profile_id, ok = QInputDialog.getText(self, "Save Layout", "Profile ID:")
            if ok and profile_id:
                name, ok2 = QInputDialog.getText(self, "Save Layout", "Display Name:")
                if ok2 and name:
                    self.workbench_manager.save_current_layout_as_profile(profile_id, name)
                    self.statusBar().showMessage(f"Saved layout as profile: {name}", 2000)
        except Exception as e:
            logger.error(f"Failed to save layout: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save layout:\n{e}")

    def start_workflow(self, workflow_id: str) -> None:
        """Start a workflow wizard (STEP 39)."""
        try:
            from .workflow_wizard import WorkflowWizard, get_workflow_definition

            workflow_def = get_workflow_definition(workflow_id)
            if not workflow_def:
                QMessageBox.warning(self, "Workflow Not Found", f"Workflow '{workflow_id}' not found.")
                return

            wizard = WorkflowWizard(self, workflow_def)
            wizard.start()
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to start workflow:\n{e}")

    def save_session_template(self) -> None:
        """Save current session as a template (STEP 39)."""
        try:
            from PyQt6.QtWidgets import QInputDialog

            from .templates_manager import TemplatesManager

            if not hasattr(self, 'templates_manager'):
                self.templates_manager = TemplatesManager()

            template_id, ok = QInputDialog.getText(self, "Save Template", "Template ID:")
            if ok and template_id:
                name, ok2 = QInputDialog.getText(self, "Save Template", "Display Name:")
                if ok2 and name:
                    description, ok3 = QInputDialog.getText(self, "Save Template", "Description (optional):")
                    desc = description if ok3 else ""

                    template = self.templates_manager.create_template_from_current_session(
                        template_id=template_id,
                        name=name,
                        description=desc,
                        main_window=self,
                        controller=self.controller
                    )
                    self.templates_manager.save_template(template)
                    self.statusBar().showMessage(f"Saved template: {name}", 2000)
        except Exception as e:
            logger.error(f"Failed to save template: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save template:\n{e}")

    def load_session_template(self) -> None:
        """Load a saved template (STEP 39)."""
        try:
            from PyQt6.QtWidgets import QInputDialog

            from .templates_manager import TemplatesManager

            if not hasattr(self, 'templates_manager'):
                self.templates_manager = TemplatesManager()

            templates = self.templates_manager.list_templates()
            if not templates:
                QMessageBox.information(self, "No Templates", "No saved templates found.")
                return

            template_names = [f"{t.name} ({t.id})" for t in templates]
            template_name, ok = QInputDialog.getItem(
                self, "Load Template", "Select template:", template_names, 0, False
            )

            if ok and template_name:
                # Extract template ID
                template_id = template_name.split("(")[1].rstrip(")")
                template = self.templates_manager.load_template(template_id)

                if template:
                    # Restore layout
                    if template.layout_snapshot:
                        import base64
                        layout_bytes = base64.b64decode(template.layout_snapshot)
                        self.restoreState(layout_bytes)

                        # Ensure drillhole control dock is visible after template restoration
                        if self.drillhole_control_dock:
                            self.drillhole_control_dock.setVisible(True)
                            self.drillhole_control_dock.raise_()

                    # Restore scenario if available
                    if template.scenario_snapshot and self.controller:
                        # Would restore scenario state here
                        pass

                    self.statusBar().showMessage(f"Loaded template: {template.meta.name}", 2000)
        except Exception as e:
            logger.error(f"Failed to load template: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load template:\n{e}")

    def delete_session_template(self) -> None:
        """Delete a saved template (STEP 39)."""
        try:
            from PyQt6.QtWidgets import QInputDialog

            from .templates_manager import TemplatesManager

            if not hasattr(self, 'templates_manager'):
                self.templates_manager = TemplatesManager()

            templates = self.templates_manager.list_templates()
            if not templates:
                QMessageBox.information(self, "No Templates", "No saved templates found.")
                return

            template_names = [f"{t.name} ({t.id})" for t in templates]
            template_name, ok = QInputDialog.getItem(
                self, "Delete Template", "Select template to delete:", template_names, 0, False
            )

            if ok and template_name:
                reply = QMessageBox.question(
                    self, "Confirm Delete",
                    f"Delete template '{template_name}'?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply == QMessageBox.StandardButton.Yes:
                    template_id = template_name.split("(")[1].rstrip(")")
                    self.templates_manager.delete_template(template_id)
                    self.statusBar().showMessage(f"Deleted template: {template_name}", 2000)
        except Exception as e:
            logger.error(f"Failed to delete template: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to delete template:\n{e}")

    # -------------------------------------------------------------
    # Search menu helper
    # -------------------------------------------------------------
    def _search_modules(self):
        """Search menu actions/modules by name and open the selected one."""
        actions = self.findChildren(QAction)
        items = []
        valid_actions = []
        for act in actions:
            label = (act.text() or "").replace("&", "")
            # Skip empty labels
            if not label:
                continue
            
            # Skip menu actions that don't have actual functionality
            # Menu actions are QActions that just open menus and don't trigger real functionality
            if act.menu() is not None:
                # This is a menu action - skip it since it just opens a menu
                continue
            
            # Skip actions without connections (they won't do anything when triggered)
            if not act.receivers(act.triggered):
                continue
            
            # Add the action to search results (including checkable actions like dock toggles)
            items.append((label, act.statusTip() or ""))
            valid_actions.append(act)

        dialog = ModernSearchDialog(items, self, actions=valid_actions)
        if dialog.exec() and dialog.get_selected_item():
            self._on_search_module_selected(dialog.get_selected_item())

    def _on_search_module_selected(self, item_name: str):
        """Handle the selection of a module from the search dialog."""
        if not item_name:
            return
        actions = self.findChildren(QAction)
        for act in actions:
            label = (act.text() or "").replace("&", "")
            if label == item_name:
                act.trigger()
                break
