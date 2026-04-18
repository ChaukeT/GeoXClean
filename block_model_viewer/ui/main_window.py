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
    QToolBar,
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
from .bookmarks import BookmarkManager
from .charts_panel import ChartsPanel
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
from .cutoff_optimization_panel import CutoffOptimizationPanel
from .variogram_panel import VariogramAnalysisPanel
from .swath_analysis_3d_panel import SwathAnalysis3DPanel
from .loopstructural_panel import LoopStructuralModelPanel
from .compositing_window import CompositingWindow
from .grade_transformation_panel import GradeTransformationPanel
from .interaction import InteractionController
from .irr_panel import IRRPanel
from .jorc_classification_panel import JORCClassificationPanel
from .kmeans_clustering_panel import KMeansClusteringPanel
from .panel_manager import PanelManager
from .panel_registration import register_all_panels
from .persistent_dock import PersistentDockWidget
from .preferences_dialog import PreferencesDialog
from .project_loading_dialog import ProjectLoadingDialog
from .property_panel import PropertyPanel
from .qc_window import QCWindow
from .jorc_classification_panel import JORCClassificationPanel as ResourceClassificationPanel
from .resource_reporting_panel import ResourceReportingPanel
from .block_property_calculator_panel import BlockPropertyCalculatorPanel
from .screenshot_export_dialog import ScreenshotExportDialog
from .data_export_dialog import DataExportDialog
from .shortcuts import Shortcuts
from .signals import UISignals

from .statistics_panel import StatisticsPanel
from .status import StatusManager
from .swath_panel import SwathPanel
from .table_viewer_panel import TableViewerPanel

from .theme_manager import ThemeManager
from .underground_panel import UndergroundPanel
from .viewer_widget import ViewerWidget

# ---- Decomposition: mixins provide panel + file-I/O methods ----
from .mixins.panel_mixin import PanelMixin
from .mixins.file_mixin import FileMixin

# ---- Decomposition: coordinator objects own signal/menu/workspace wiring ----
from .coordinators import (
    MenuCoordinator,
    SignalCoordinator,
    WorkspaceCoordinator,
)

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
        """No-op. App QSS handles theming."""
        pass
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


class MainWindow(PanelMixin, FileMixin, QMainWindow):
    """
    Refactored Main Window with menu-driven architecture.
    Clean, professional layout with dockable panels.

    Decomposed into focused mixin/coordinator files:
      ui/mixins/panel_mixin.py   — all open_* panel methods (209 methods)
      ui/mixins/file_mixin.py    — project save/load & file I/O (19 methods)
      ui/coordinators/signal_coordinator.py   — signal wiring
      ui/coordinators/menu_coordinator.py     — menu-bar construction
      ui/coordinators/workspace_coordinator.py — geometry + workbench
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
        self.irr_panel: Optional[IRRPanel] = None
        self.irr_dialog: Optional[QDialog] = None
        self.kmeans_panel: Optional[KMeansClusteringPanel] = None
        self.kmeans_dialog: Optional[QDialog] = None
        self.resource_classification_panel: Optional[ResourceClassificationPanel] = None
        self.resource_classification_dialog: Optional[QDialog] = None
        self.resource_reporting_panel: Optional[ResourceReportingPanel] = None
        self.resource_reporting_dialog: Optional[QDialog] = None

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
        self.theme_manager = ThemeManager(app=QApplication.instance())

        # Shortcuts
        self.shortcuts = Shortcuts()

        # Panel manager - unified lifecycle system
        # All panel registrations happen in panel_registration.py
        self.panel_manager = PanelManager(self)
        register_all_panels(self.panel_manager)

        # Workspace layout state
        self.current_workspace_layout: str = "default"

        # ---- Coordinators (own signal wiring, menu build, workspace state) ----
        self._signal_coordinator = SignalCoordinator(self)
        self._menu_coordinator = MenuCoordinator(self)
        self._workspace_coordinator = WorkspaceCoordinator(self)

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
        self._restoring_registry_models = False  # Guard flag to suppress re-renders during bulk restore

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

        # Show toast warning when coordinate mismatch is detected between datasets
        try:
            sig = getattr(self._registry, 'coordinateMismatchDetected', None)
            if sig is not None:
                sig.connect(self._on_coordinate_mismatch)
                logger.debug("Connected coordinate mismatch warning signal")
        except Exception as e:
            logger.debug(f"Could not connect coordinate mismatch signal: {e}")

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

        # Connect legend widget's colormap_changed to signal bus
        try:
            legend_mgr = getattr(self.controller, "legend_manager", None)
            if legend_mgr and hasattr(legend_mgr, 'widget') and legend_mgr.widget:
                legend_widget = legend_mgr.widget
                if hasattr(legend_widget, 'colormap_changed') and self.signals:
                    legend_widget.colormap_changed.connect(self.signals.legendColormapChanged.emit)
        except Exception:
            logger.debug("Could not connect legend widget colormap signal", exc_info=True)

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
        """Delegate to MenuCoordinator (see ui/coordinators/menu_coordinator.py)."""
        self._menu_coordinator.setup_menus()

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
        action = getattr(self, "axes_scalebar_action", None)
        self.axes_scalebar_window = AxesScaleBarDialog(self, action, controller)
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
        """Delegate to SignalCoordinator (see ui/coordinators/signal_coordinator.py)."""
        self._signal_coordinator.connect_all()

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
        self.signals.opacityChanged.connect(self.controller.set_global_opacity)

        # Slicing and filtering
        self.signals.sliceChanged.connect(self.controller.apply_slice)
        self.signals.applyFilters.connect(self.controller.apply_filters)

        # Export
        self.signals.exportScreenshot.connect(self.controller.export_screenshot)

        # Data loading
        self.signals.blockModelLoaded.connect(self.controller.load_block_model)

        # Drillhole control panel signals — core operations
        self.signals.drillholePlotRequested.connect(self._on_drillhole_control_plot)
        self.signals.drillholeClearRequested.connect(self._on_drillhole_control_clear)
        self.signals.drillholeRadiusChanged.connect(self._on_drillhole_radius_changed)
        self.signals.drillholeShowIdsToggled.connect(self._on_drillhole_show_ids_toggled)
        self.signals.drillholeVisibilityChanged.connect(self._on_drillhole_visibility_changed)
        self.signals.drillholeFocusRequested.connect(self._on_drillhole_focus_requested)
        # Drillhole control panel signals — rendering toggles
        self.signals.drillholeColorModeChanged.connect(self._on_drillhole_color_mode_changed)
        self.signals.drillholeAssayFieldChanged.connect(self._on_drillhole_assay_field_changed)
        self.signals.drillholeCollarToggled.connect(self._on_drillhole_collar_toggled)
        self.signals.drillholePbrToggled.connect(self._on_drillhole_pbr_toggled)
        self.signals.drillholeSsaoToggled.connect(self._on_drillhole_ssao_toggled)
        self.signals.drillholeEdlToggled.connect(self._on_drillhole_edl_toggled)
        self.signals.drillholeHideBarrenToggled.connect(self._on_drillhole_hide_barren_toggled)

        # Legend colormap signal
        self.signals.legendColormapChanged.connect(self._on_legend_colormap_changed)

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

    def _on_coordinate_mismatch(self, message: str):
        """Show a persistent toast warning when coordinate mismatch is detected."""
        try:
            from .toast import ToastWidget
            ToastWidget.show_message(self, message, duration=8000)
        except Exception:
            pass
        # Also show in status bar for persistence
        try:
            self.statusBar().showMessage(message, 15000)
        except Exception:
            pass

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
                    legend_widget = getattr(renderer.legend_manager, 'widget', None)
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
            self.theme_manager.apply_theme()

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
            self.theme_manager.apply_theme()

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

    def _get_viewer_bg_for_theme(self, theme_name: str | None = None) -> str:
        """Return the 3D viewport background color for the given theme.

        If SSAO or EDL is active the dark viewport background is kept
        regardless of UI theme (those effects require a dark canvas).
        """
        from .design_tokens import tokens
        if theme_name is None:
            theme_name = tokens.theme

        # SSAO/EDL require dark background to work visually
        ssao_on = (hasattr(self, 'drillhole_control_panel')
                   and self.drillhole_control_panel
                   and hasattr(self.drillhole_control_panel, 'ssao_check')
                   and self.drillhole_control_panel.ssao_check.isChecked())
        edl_on = (hasattr(self, 'drillhole_control_panel')
                  and self.drillhole_control_panel
                  and hasattr(self.drillhole_control_panel, 'edl_check')
                  and self.drillhole_control_panel.edl_check.isChecked())
        if ssao_on or edl_on:
            return '#1a1a1e'

        return '#1a1a1e' if theme_name == 'dark' else '#c8cad0'

    def _on_theme_changed(self, theme_name: str):
        """
        Handle theme change signal.

        Args:
            theme_name: New theme name
        """
        try:
            # Update 3D viewport background to match theme
            if hasattr(self, 'viewer_widget') and self.viewer_widget:
                if hasattr(self.viewer_widget, 'renderer') and self.viewer_widget.renderer:
                    try:
                        bg = self._get_viewer_bg_for_theme(theme_name)
                        self.viewer_widget.renderer.set_background_color(bg)
                        if self.viewer_widget.renderer.plotter:
                            self.viewer_widget.renderer.plotter.render()
                    except Exception:
                        pass

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

    # STEP 17b: Zoom & mouse-mode toolbar handlers
    def _handle_zoom_in(self):
        """Zoom in by a fixed step (same as scroll wheel up)."""
        if self.interaction is not None:
            self.interaction.zoom_in()
        elif self.viewer_widget:
            self.viewer_widget.zoom_in()
            self.status_bar.showMessage("Zoomed in", 1500)

    def _handle_zoom_out(self):
        """Zoom out by a fixed step (same as scroll wheel down)."""
        if self.interaction is not None:
            self.interaction.zoom_out()
        elif self.viewer_widget:
            self.viewer_widget.zoom_out()
            self.status_bar.showMessage("Zoomed out", 1500)

    def _handle_mouse_mode(self, mode: str):
        """Set mouse interaction mode from toolbar buttons.

        Delegates to the InteractionController which handles VTK interactor
        style changes, cursor updates, and picking state.
        """
        try:
            if self.interaction is not None:
                self.interaction.set_mode_from_string(mode)
            elif self.viewer_widget and hasattr(self.viewer_widget, 'set_interaction_mode'):
                self.viewer_widget.set_interaction_mode(mode)
        except Exception as e:
            logger.error(f"Error setting mouse mode '{mode}': {e}", exc_info=True)

    def _restore_state(self):
        """Delegate to WorkspaceCoordinator (see ui/coordinators/workspace_coordinator.py)."""
        self._workspace_coordinator.restore_state()

    def _save_state(self):
        """Delegate to WorkspaceCoordinator (see ui/coordinators/workspace_coordinator.py)."""
        self._workspace_coordinator.save_state()

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

    def save_current_layout(self):
        """Save the current dock/panel layout as a named profile."""
        self._save_layout()

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
        """Open a block model or surface file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "All Supported Files (*.csv *.txt *.vtk *.vtu *.obj *.gltf *.tif *.tiff *.asc *.grd *.dxf);;"
            "CSV Files (*.csv);;Text Files (*.txt);;"
            "VTK Files (*.vtk *.vtu);;3D Models (*.obj *.gltf);;"
            "Topographic Surfaces (*.tif *.tiff *.asc *.grd);;"
            "DXF CAD Files (*.dxf)"
        )

        if not file_path:
            return

        # Route surface/CAD formats to the dedicated surface importer
        ext = Path(file_path).suffix.lower()
        if ext in ('.tif', '.tiff', '.asc', '.grd', '.dxf'):
            self.import_surface_file_path(Path(file_path))
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
            from ..utils.coordinate_manager import CoordinateBounds, AlignmentStatus
            file_name = self.current_file_path.name if self.current_file_path else "Unknown"
            dataset_name = f"Block Model: {file_name}"
            bounds = block_model.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            cb = CoordinateBounds(
                xmin=bounds[0], xmax=bounds[1],
                ymin=bounds[2], ymax=bounds[3],
                zmin=bounds[4], zmax=bounds[5],
            )
            dataset_info = self.coordinate_manager.register_dataset(
                dataset_name, cb, block_model.block_count, 'block_model'
            )
            coord_time = time.time() - coord_start
            logger.info(f"PERF: Coordinate registration took {coord_time:.3f}s")

            # Check if alignment is needed
            if dataset_info.alignment_status == AlignmentStatus.POTENTIALLY_MISALIGNED:
                logger.warning("Block model may be in a different coordinate system!")

                reply = QMessageBox.question(
                    self,
                    "Coordinate Alignment Warning",
                    f"The block model appears to be in a different coordinate system than existing data.\n\n"
                    f"Block model center: ({dataset_info.bounds.center[0]:,.2f}, {dataset_info.bounds.center[1]:,.2f}, {dataset_info.bounds.center[2]:,.2f})\n\n"
                    f"Continue loading anyway? (You can verify alignment in the Coordinate Manager panel.)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self.coordinate_manager.confirm_alignment(
                        dataset_name, "User confirmed at load time"
                    )
                    logger.info("User confirmed block model alignment")
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
        # Skip expensive viewer refreshes when we're bulk-restoring registry
        # models during project load.  A final refresh with the correct current
        # model happens after _restore_registry_models() completes.
        if getattr(self, '_restoring_registry_models', False):
            logger.debug("Skipping viewer refresh during registry model restoration")
            return

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

            # Initialize toolbar grade cutoff range from block model data
            try:
                if hasattr(self, 'toolbar_widget') and self.toolbar_widget and self.viewer_widget:
                    br = getattr(self.viewer_widget.renderer, '_block_renderer', None)
                    if br and br._current_clim:
                        lo, hi = br._current_clim
                        self.toolbar_widget.set_cutoff_range(lo, hi)
                        logger.info(f"Toolbar cutoff range set: [{lo:.2f}, {hi:.2f}]")
            except Exception as e:
                logger.debug(f"Failed to set toolbar cutoff range: {e}")

            handler_time = time.time() - handler_start
            logger.info(f"PERF: _on_block_model_loaded_from_registry total handler time: {handler_time:.3f}s")

            logger.info(f"Successfully loaded file: {self.current_file_path}")

            # Persist 'last_file' immediately for session restore resilience
            try:
                settings = QSettings("GeoX", "Session")
                if self.current_file_path:
                    settings.setValue("last_file", str(self.current_file_path))
            except Exception:
                pass

            # Restore drillhole data if it was saved in project
            try:
                if getattr(self, '_pending_drillhole_state', None):
                    self._restore_drillhole_data(self._pending_drillhole_state)
                    self._pending_drillhole_state = None
            except Exception as e:
                logger.warning(f"Failed to restore drillhole data: {e}")

            # Restore registry models (variogram, kriging, SGSIM results, etc.)
            # Clear the pending state BEFORE restoring to prevent any re-entrant calls.
            # Set guard flag so re-entrant blockModelLoaded signals during restore
            # don't trigger expensive viewer refreshes for each intermediate model.
            try:
                _pending = getattr(self, '_pending_registry_models_state', None)
                if _pending:
                    self._pending_registry_models_state = None
                    self._restoring_registry_models = True
                    try:
                        self._restore_registry_models(_pending)
                    finally:
                        self._restoring_registry_models = False
            except Exception as e:
                logger.warning(f"Failed to restore registry models: {e}")

            # After ALL data is restored, refresh the viewer with the CURRENT
            # block model from the registry.  This ensures the viewer shows
            # the correct model with all properties (not a stale intermediate).
            try:
                registry = self.controller.registry if self.controller else None
                if registry is not None:
                    current_model = registry.get_block_model(copy_data=False)
                    if current_model is not None and self.viewer_widget:
                        self.viewer_widget.refresh_scene(current_model)
                        if self.property_panel:
                            self.property_panel.set_block_model(current_model)
                        logger.info("Final viewer refresh with current registry model after project restore")
            except Exception as e:
                logger.warning(f"Failed final viewer refresh after restore: {e}")

            # Rebuild every tagged scene layer from the freshly restored
            # registry (ARBF/kriging/SGSIM/etc) BEFORE applying session state.
            # This ensures layer visibility + active-property restoration
            # lands on layers that actually exist in the scene.
            try:
                pending_state = getattr(self, '_pending_session_state', None) or {}
                manifest = pending_state.get('scene_manifest') if isinstance(pending_state, dict) else None
                if manifest and self.viewer_widget and self.viewer_widget.renderer:
                    registry_inst = self.controller.registry if self.controller else None
                    if registry_inst is not None:
                        kind_handlers = {
                            'classification': self._handle_classification_visualization,
                        }
                        n = self.viewer_widget.renderer.rebuild_scene_from_manifest(
                            manifest, registry_inst, kind_handlers=kind_handlers,
                        )
                        logger.info(f"Scene rebuilder reconstructed {n} tagged layers")
            except Exception as e:
                logger.warning(f"Scene rebuild from manifest failed: {e}", exc_info=True)

            # Apply renderer state AFTER all data is restored and viewer is
            # refreshed.  The saved state references properties (kr_Cu,
            # PIT_SHELL, etc.) that only exist on the enhanced model.
            try:
                if getattr(self, '_pending_session_state', None) and self.viewer_widget and self.viewer_widget.renderer:
                    self.viewer_widget.renderer.apply_session_state(self._pending_session_state)
                    self._pending_session_state = None
            except Exception as e:
                logger.warning(f"Failed to apply pending session state: {e}")

            # Enable actions that require a loaded block model
            try:
                if hasattr(self, 'view_data_action') and self.view_data_action:
                    self.view_data_action.setEnabled(True)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error loading model in viewer: {type(e).__name__}: {e}")
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

    def _clear_for_project_load(self):
        """Silently clear all scene data, registry, and panels before loading a new project.

        Unlike clear_scene(), this skips user confirmation and the "already empty"
        check so that project loading always starts from a clean slate.
        """
        try:
            renderer = getattr(self.viewer_widget, 'renderer', None) if self.viewer_widget else None

            # Clear 3D viewer (actors, meshes, picking state)
            if self.viewer_widget:
                self.viewer_widget.clear_scene()

            # Clear all renderer active_layers (drillholes, blocks, geology, etc.)
            if renderer and hasattr(renderer, 'clear_all_layers'):
                renderer.clear_all_layers()

            # Clear DataRegistry so old estimation/drillhole data doesn't bleed through
            try:
                from ..core.data_registry import DataRegistry
                if hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None:
                    DataRegistry._instance.clear_all()
                    logger.info("DataRegistry cleared for project load")
            except Exception as e:
                logger.debug(f"Could not clear DataRegistry: {e}")

            # ── SIA-001: Clear ALL instantiated panels, not just 3 hardcoded ones ──
            # Hidden panels cache data too; clearing only visible/hardcoded panels
            # left stale caches in ChartsPanel, StatisticsPanel, KrigingPanel,
            # SGSIMPanel, VariogramPanel, GradeTonnagePanel, etc.
            if hasattr(self, 'panel_manager') and self.panel_manager is not None:
                try:
                    self.panel_manager.clear_all_panels()
                except Exception as e:
                    logger.debug(f"PanelManager clear_all_panels failed: {e}")

            # Legacy fallback for panels not registered with PanelManager
            if self.property_panel:
                self.property_panel.clear()
            if self.data_viewer_panel and hasattr(self.data_viewer_panel, 'clear'):
                self.data_viewer_panel.clear()
            if self.scene_inspector_panel and hasattr(self.scene_inspector_panel, 'clear_camera_info'):
                self.scene_inspector_panel.clear_camera_info()

            # Clear process history
            try:
                from ..core.process_history_tracker import get_process_history_tracker
                get_process_history_tracker().clear_history()
            except Exception:
                pass

            # Reset current model / file references
            self.current_model = None
            self.current_file_path = None

            logger.info("Scene cleared silently for project load")

        except Exception as e:
            logger.warning(f"Error during pre-load clear: {e}", exc_info=True)

    def clear_scene(self):
        """Clear the scene and remove all models (block model and drillholes)."""
        # Check if there's anything to clear
        has_block_model = self._has_valid_block_model()
        has_drillhole_data = False
        has_any_layers = False

        # Check renderer active_layers for drillholes and other content
        renderer = getattr(self.viewer_widget, 'renderer', None) if self.viewer_widget else None
        if renderer and hasattr(renderer, 'active_layers') and renderer.active_layers:
            has_any_layers = True
            has_drillhole_data = 'drillholes' in renderer.active_layers

        if not has_block_model and not has_drillhole_data and not has_any_layers:
            QMessageBox.information(self, "No Model", "Scene is already empty.")
            return

        # Confirm action
        model_types = []
        if has_block_model:
            model_types.append("block model")
        if has_drillhole_data:
            model_types.append("drillholes")
        if has_any_layers and not model_types:
            model_types.append("all layers")

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
            # Clear viewer (plotter actors, block meshes, picking state)
            if self.viewer_widget:
                self.viewer_widget.clear_scene()

            # Clear all renderer active_layers (drillholes, blocks, geology, etc.)
            if renderer and hasattr(renderer, 'clear_all_layers'):
                renderer.clear_all_layers()

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

    # ------------------------------------------------------------------
    # Drillhole core handlers (plot, clear, radius, visibility, ids, focus)
    # ------------------------------------------------------------------

    def _on_drillhole_data_registered(self, data: Dict) -> None:
        """Handle drillhole data registration - auto-render like block models."""
        logger.info("MainWindow._on_drillhole_data_registered: Signal received - will auto-render")

        def _update_ui_and_render():
            try:
                dataset_name = data.get("name", "Drillhole Data")
                self.status_bar.showMessage(f"Rendering drillholes: {dataset_name}...", 0)

                if self.controller:
                    try:
                        self.controller._update_state_from_scene()
                    except Exception as e:
                        logger.warning(f"Failed to update app state: {e}")

                try:
                    self._on_drillhole_control_plot("Raw Assays")
                    logger.info(f"Auto-rendered drillholes: {dataset_name}")
                except Exception as e:
                    logger.error(f"Failed to auto-render drillholes: {e}", exc_info=True)
                    self.status_bar.showMessage(f"Drillhole data loaded: {dataset_name}", 5000)

            except Exception as e:
                logger.error(f"Error in _on_drillhole_data_registered: {e}", exc_info=True)

        QTimer.singleShot(100, _update_ui_and_render)

    def _on_drillhole_control_plot(self, dataset_name: str) -> None:
        """Plot drillholes in the main renderer."""
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return
        registry = self.controller.registry if self.controller else None
        if registry is None:
            QMessageBox.warning(self, "Error", "DataRegistry is not initialized.")
            return
        data = registry.get_drillhole_data()
        if data is None:
            QMessageBox.warning(self, "Error", "No drillhole data is registered.")
            return
        if not isinstance(data, dict):
            QMessageBox.warning(self, "Error", f"Invalid drillhole data format: {type(data).__name__}")
            return

        from ..drillholes.registry_utils import build_database_from_registry
        try:
            db = build_database_from_registry(data)
        except Exception as e:
            logger.error(f"Failed to build drillhole database: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to build drillhole database:\n{e}")
            return

        composite_df = None
        if dataset_name == "Composites":
            composites_from_registry = data.get("composites")
            composites_df_from_registry = data.get("composites_df")
            if isinstance(composites_from_registry, pd.DataFrame) and not composites_from_registry.empty:
                composite_df = composites_from_registry
            elif isinstance(composites_df_from_registry, pd.DataFrame) and not composites_df_from_registry.empty:
                composite_df = composites_df_from_registry
            if composite_df is None or (isinstance(composite_df, pd.DataFrame) and composite_df.empty):
                QMessageBox.warning(self, "Error", "No composite data available.")
                return
            try:
                db.set_table("assays", composite_df.copy())
            except Exception as e:
                logger.warning(f"Failed to replace assays with composites: {e}")

        radius = 10.0
        if self.drillhole_control_panel:
            radius = self.drillhole_control_panel.get_radius()

        self._start_drillhole_render_status()
        success = False
        try:
            self.viewer_widget.renderer.remove_drillhole_layer()
            visible_holes = self.drillhole_control_panel.get_visible_holes() if self.drillhole_control_panel else None
            color_mode = self.drillhole_control_panel.get_color_mode() if self.drillhole_control_panel else "Lithology"
            assay_field = self.drillhole_control_panel.get_assay_field() if self.drillhole_control_panel else None
            lith_filter = self.drillhole_control_panel.get_selected_lithologies() if self.drillhole_control_panel else []

            # Sync hide_barren flag to renderer BEFORE rendering
            if self.drillhole_control_panel and hasattr(self.drillhole_control_panel, "hide_barren_check"):
                self.viewer_widget.renderer._hide_barren_intervals = (
                    self.drillhole_control_panel.hide_barren_check.isChecked()
                )

            if lith_filter and "__NONE__" in lith_filter:
                self._stop_drillhole_render_status(False)
                return

            if visible_holes is not None and len(visible_holes) == 0:
                self._stop_drillhole_render_status(False)
                return

            progress_cb = lambda frac, msg: self._update_status_progress(msg, frac)
            self.viewer_widget.renderer.add_drillhole_layer(
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
            if self.viewer_widget:
                cell_count = 0
                if self.viewer_widget.current_model:
                    cell_count = self.viewer_widget.current_model.block_count
                self.viewer_widget.enable_interaction(
                    cell_count=cell_count,
                    has_block_model=self.viewer_widget.current_model is not None,
                    has_drillholes=True,
                )
            success = True
        except Exception as e:
            logger.error(f"Failed to add drillhole layer: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to render drillholes: {e}")
        finally:
            fresh_metadata = self.viewer_widget.renderer.get_drillhole_legend_metadata() if success else None
            if fresh_metadata and fresh_metadata.get("property") is not None:
                try:
                    self._update_legend_from_drillholes(fresh_metadata)
                except Exception as e:
                    logger.error(f"Failed to update legend: {e}", exc_info=True)
            if success:
                self._activate_scene_inspector_legend()
                if fresh_metadata:
                    try:
                        self._update_property_panel_for_drillholes(dataset_name, color_mode, fresh_metadata)
                    except Exception:
                        pass
                try:
                    if self.controller:
                        self.controller._update_state_from_scene()
                except Exception:
                    pass
            self._stop_drillhole_render_status(success)

    def _on_drillhole_control_clear(self) -> None:
        """Clear drillholes from renderer."""
        if self.viewer_widget and self.viewer_widget.renderer:
            self.viewer_widget.renderer.remove_drillhole_layer()
            logger.info("Cleared drillhole layer")

    def _on_drillhole_radius_changed(self, radius: float) -> None:
        """Update drillhole radius in renderer."""
        if self.viewer_widget and self.viewer_widget.renderer:
            if "drillholes" in self.viewer_widget.renderer.active_layers:
                self.viewer_widget.renderer.update_drillhole_radius(radius)
                metadata = self.viewer_widget.renderer.get_drillhole_legend_metadata()
                if metadata:
                    self._update_legend_from_drillholes(metadata)

    def _on_drillhole_show_ids_toggled(self, show: bool) -> None:
        """Toggle drillhole ID labels."""
        if self.viewer_widget and self.viewer_widget.renderer:
            self.viewer_widget.renderer.set_drillhole_labels_visible(show)

    def _on_drillhole_visibility_changed(self, hole_id: str, visible: bool) -> None:
        """Toggle individual hole visibility."""
        if self.viewer_widget and self.viewer_widget.renderer:
            self.viewer_widget.renderer.set_drillhole_visibility(hole_id, visible)

    def _on_drillhole_focus_requested(self) -> None:
        """Focus camera on selected drillholes."""
        if self.viewer_widget and self.viewer_widget.renderer and self.drillhole_control_panel:
            visible_holes = self.drillhole_control_panel.get_visible_holes()
            if visible_holes:
                self.viewer_widget.renderer.focus_on_selected_drillholes(visible_holes)

    def _on_drillhole_color_mode_changed(self, mode: str) -> None:
        """Handle drillhole color mode change — update renderer, legend, and property panel."""
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return
        if "drillholes" not in self.viewer_widget.renderer.active_layers:
            return

        property_name = "Lithology" if mode == "Lithology" else "FE"
        # Lithology uses categorical colormap; assay uses turbo
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
            custom_colors=None,
        )

        # Update legend
        metadata = self.viewer_widget.renderer.get_drillhole_legend_metadata()
        if metadata:
            self._update_legend_from_drillholes(metadata)

        # Sync property panel to match
        self._sync_property_panel_from_drillhole_control(mode, property_name, colormap)
        logger.info(f"Drillhole color mode changed to '{mode}', property='{property_name}'")

    def _on_drillhole_assay_field_changed(self, field: str) -> None:
        """Handle drillhole assay field change — update renderer, legend, and property panel."""
        if not field:
            return
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return
        if "drillholes" not in self.viewer_widget.renderer.active_layers:
            return

        cmap = "turbo"

        self.viewer_widget.renderer._update_drillhole_colors(
            property_name=field,
            colormap=cmap,
            color_mode="Assay",
            custom_colors=None,
        )

        # Update legend
        metadata = self.viewer_widget.renderer.get_drillhole_legend_metadata()
        if metadata:
            self._update_legend_from_drillholes(metadata)

        # Sync property panel to match
        self._sync_property_panel_from_drillhole_control("Assay", field, cmap)
        logger.info(f"Drillhole assay field changed to '{field}'")

    def _on_drillhole_pbr_toggled(self, enabled: bool) -> None:
        """Toggle PBR smooth shading on drillhole tube actors."""
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return
        renderer = self.viewer_widget.renderer
        # Toggle PBR on all drillhole hole actors
        actors = getattr(renderer, '_drillhole_hole_actors', {})
        for actor in actors.values():
            try:
                prop = actor.GetProperty()
                if enabled:
                    prop.SetInterpolationToPBR()
                    prop.SetMetallic(0.1)
                    prop.SetRoughness(0.5)
                else:
                    prop.SetInterpolationToPhong()
                    prop.SetSpecular(0.3)
                    prop.SetSpecularPower(15)
                    prop.SetAmbient(0.3)
                    prop.SetDiffuse(0.8)
            except Exception:
                pass
        if renderer.plotter:
            try:
                renderer.plotter.render()
            except Exception:
                pass
        logger.info(f"Drillhole PBR {'enabled' if enabled else 'disabled'}")

    def _on_drillhole_collar_toggled(self, visible: bool) -> None:
        """Show/hide collar sphere markers."""
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return
        try:
            self.viewer_widget.renderer.set_collar_visibility(visible)
        except Exception as e:
            logger.debug(f"Could not toggle collar visibility: {e}")

    def _on_drillhole_ssao_toggled(self, enabled: bool) -> None:
        """Toggle screen-space ambient occlusion."""
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return
        if enabled:
            self.viewer_widget.renderer.enable_ssao()
            # Switch to dark background for SSAO/EDL visibility
            self.viewer_widget.renderer.set_background_color('#1a1a1e')
        else:
            self.viewer_widget.renderer.disable_ssao()
            # Revert to theme-appropriate background if EDL is also off
            self.viewer_widget.renderer.set_background_color(
                self._get_viewer_bg_for_theme()
            )
        if self.viewer_widget.renderer.plotter:
            self.viewer_widget.renderer.plotter.render()

    def _on_drillhole_edl_toggled(self, enabled: bool) -> None:
        """Toggle eye-dome lighting."""
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return
        if enabled:
            self.viewer_widget.renderer.enable_edl()
            # Switch to dark background for SSAO/EDL visibility
            self.viewer_widget.renderer.set_background_color('#1a1a1e')
        else:
            self.viewer_widget.renderer.disable_edl()
            # Revert to theme-appropriate background if SSAO is also off
            self.viewer_widget.renderer.set_background_color(
                self._get_viewer_bg_for_theme()
            )
        if self.viewer_widget.renderer.plotter:
            self.viewer_widget.renderer.plotter.render()

    def _on_drillhole_hide_barren_toggled(self, hide: bool) -> None:
        """Toggle barren interval visibility — triggers full color rebuild."""
        if not self.viewer_widget or not self.viewer_widget.renderer:
            return
        renderer = self.viewer_widget.renderer
        # Store the flag on the renderer so the color update path can read it
        renderer._hide_barren_intervals = hide
        # Trigger a full color rebuild to apply barren handling
        if "drillholes" not in renderer.active_layers:
            return
        cache = renderer._drillhole_polylines_cache
        if cache is None:
            return
        color_mode = cache.get("color_mode", "Lithology")
        if color_mode == "Lithology":
            property_name = "Lithology"
        else:
            property_name = cache.get("assay_field", "assay")
        colormap = "tab10" if color_mode == "Lithology" else "turbo"
        renderer._update_drillhole_colors(
            property_name=property_name,
            colormap=colormap,
            color_mode=color_mode,
            custom_colors=None,
        )
        metadata = renderer.get_drillhole_legend_metadata()
        if metadata:
            self._update_legend_from_drillholes(metadata)
        logger.info(f"Barren intervals {'hidden' if hide else 'shown'}")

    def _update_legend_from_drillholes(self, metadata: Dict[str, Any]) -> None:
        """Broadcast drillhole legend metadata to the LegendManager."""
        if not metadata or metadata.get("property") is None or metadata.get("mode") is None:
            return

        legend_manager = None
        if self.controller and hasattr(self.controller, "legend_manager"):
            legend_manager = self.controller.legend_manager
        if not legend_manager:
            return

        try:
            if metadata.get("mode") == "discrete":
                categories = metadata.get("categories", [])
                category_colors = metadata.get("category_colors", {})
                colormap = metadata.get("colormap", "tab10")

                # Convert category_colors to RGBA tuples
                converted_colors = {}
                for cat, color_val in category_colors.items():
                    if isinstance(color_val, str):
                        from PyQt6.QtGui import QColor
                        qc = QColor(color_val)
                        converted_colors[cat] = (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
                    elif isinstance(color_val, (tuple, list)):
                        if len(color_val) == 3:
                            converted_colors[cat] = (*color_val, 1.0)
                        elif len(color_val) == 4:
                            converted_colors[cat] = tuple(color_val)
                        else:
                            converted_colors[cat] = (0.5, 0.5, 0.5, 1.0)
                    else:
                        converted_colors[cat] = (0.5, 0.5, 0.5, 1.0)

                legend_manager.update_discrete(
                    property_name=metadata.get("property", "Lithology"),
                    categories=categories,
                    category_colors=converted_colors if converted_colors else None,
                    cmap_name=colormap,
                    subtitle="Drillholes"
                )
            else:
                # Continuous mode
                data = metadata.get("data")
                colormap = metadata.get("colormap", "turbo")
                legend_manager.update_continuous(
                    property_name=metadata.get("property", "Assay"),
                    data=data,
                    cmap_name=colormap
                )
        except Exception as e:
            logger.error(f"Failed to update legend from drillholes: {e}", exc_info=True)

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
            # Block signals on active_layer_combo to prevent clearing property combo
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

            # Skip property/colormap updates if no colors assigned
            if legend_metadata.get("property") is None or legend_metadata.get("scalar_name") is None:
                logger.info("Skipping property/colormap updates - no colors assigned during drillhole loading")
                if hasattr(self.property_panel, 'active_layer_combo') and self.property_panel.active_layer_combo:
                    self.property_panel.active_layer_combo.blockSignals(False)
                return

            scalar_name = legend_metadata.get("scalar_name", "lith_id" if color_mode == "Lithology" else "assay")
            mode_text = "Discrete" if color_mode == "Lithology" else "Continuous"

            # Set color mode
            if hasattr(self.property_panel, 'color_mode_combo') and self.property_panel.color_mode_combo:
                self.property_panel.color_mode_combo.blockSignals(True)
                self.property_panel.color_mode_combo.setCurrentText(mode_text)
                self.property_panel.color_mode_combo.blockSignals(False)

            # Update property combo
            if hasattr(self.property_panel, 'property_combo') and self.property_panel.property_combo:
                layer_data = None
                if hasattr(self, 'viewer_widget') and self.viewer_widget and hasattr(self.viewer_widget, 'renderer'):
                    layer_info = self.viewer_widget.renderer.active_layers.get("drillholes", {})
                    layer_data = layer_info.get('data')

                self.property_panel.set_active_layer("drillholes", layer_data)

                current_property = legend_metadata.get("property", scalar_name)
                if current_property:
                    self.property_panel.property_combo.blockSignals(True)
                    index = self.property_panel.property_combo.findText(current_property)
                    if index >= 0:
                        self.property_panel.property_combo.setCurrentIndex(index)
                    else:
                        self.property_panel.property_combo.addItem(current_property)
                        self.property_panel.property_combo.setCurrentText(current_property)
                    self.property_panel.property_combo.blockSignals(False)

            # Unblock active_layer_combo signals
            if hasattr(self.property_panel, 'active_layer_combo') and self.property_panel.active_layer_combo:
                self.property_panel.active_layer_combo.blockSignals(False)

            # Update colormap dropdown
            if hasattr(self.property_panel, 'colormap_combo') and self.property_panel.colormap_combo:
                colormap = legend_metadata.get("colormap", "viridis")
                if colormap:
                    index = self.property_panel.colormap_combo.findText(colormap)
                    if index >= 0:
                        self.property_panel.colormap_combo.blockSignals(True)
                        self.property_panel.colormap_combo.setCurrentIndex(index)
                        self.property_panel.colormap_combo.blockSignals(False)

            logger.info(f"Updated property panel for drillholes: {dataset_name}, color_mode={color_mode}, property={scalar_name}")
        except Exception as e:
            logger.error(f"Failed to update property panel for drillholes: {e}", exc_info=True)

    def _sync_property_panel_from_drillhole_control(self, color_mode: str, property_name: str, colormap: str) -> None:
        """Sync Property Panel dropdowns when Drillhole Control Panel changes color/property."""
        if not self.property_panel or not hasattr(self.property_panel, 'active_layer_combo'):
            return

        # Only sync if drillholes layer is active in property panel
        current_layer = self.property_panel.active_layer_combo.currentText()
        if current_layer != "drillholes":
            return

        try:
            # Update color mode dropdown
            if hasattr(self.property_panel, 'color_mode_combo') and self.property_panel.color_mode_combo:
                mode_text = "Discrete" if color_mode == "Lithology" else "Continuous"
                with self.property_panel._block_signal(self.property_panel.color_mode_combo):
                    self.property_panel.color_mode_combo.setCurrentText(mode_text)

            # Update property dropdown
            if hasattr(self.property_panel, 'property_combo') and self.property_panel.property_combo:
                with self.property_panel._block_signal(self.property_panel.property_combo):
                    index = self.property_panel.property_combo.findText(property_name)
                    if index < 0:
                        self.property_panel.property_combo.addItem(property_name)
                    self.property_panel.property_combo.setCurrentText(property_name)

            # Update colormap dropdown
            if hasattr(self.property_panel, 'colormap_combo') and self.property_panel.colormap_combo:
                with self.property_panel._block_signal(self.property_panel.colormap_combo):
                    index = self.property_panel.colormap_combo.findText(colormap)
                    if index >= 0:
                        self.property_panel.colormap_combo.setCurrentIndex(index)
        except Exception as e:
            logger.error(f"Failed to sync property panel from drillhole control: {e}", exc_info=True)

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


    # =========================================================================
    # Search -- helpers for the Search menu
    # =========================================================================


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

    # ── File / Project ────────────────────────────────────────────

    def _new_project(self):
        """Create a new project, clearing the current scene."""
        reply = QMessageBox.question(
            self, "New Project",
            "Start a new project? Unsaved changes will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.clear_scene()
            self.registry.clear() if hasattr(self.registry, 'clear') else None
            self.setWindowTitle("GeoX")
            logger.info("New project created")

    # open_project, save_project, save_project_as are inherited from FileMixin
    # (block_model_viewer/ui/mixins/file_mixin.py) which handles full project
    # serialization including drillholes, registry models, and renderer state.

    # ── Edit / Undo ───────────────────────────────────────────────

    def _undo(self):
        """Undo the last action via controller undo manager."""
        try:
            ctrl = getattr(self, 'controller', None)
            if ctrl and hasattr(ctrl, 'undo_manager'):
                ctrl.undo_manager.undo()
            elif ctrl and hasattr(ctrl, 'undo'):
                ctrl.undo()
        except Exception as e:
            logger.debug(f"Undo failed: {e}")

    def _redo(self):
        """Redo the last undone action via controller undo manager."""
        try:
            ctrl = getattr(self, 'controller', None)
            if ctrl and hasattr(ctrl, 'undo_manager'):
                ctrl.undo_manager.redo()
            elif ctrl and hasattr(ctrl, 'redo'):
                ctrl.redo()
        except Exception as e:
            logger.debug(f"Redo failed: {e}")

    def show_find_dialog(self):
        """Show the module search / find dialog."""
        self._search_modules()

    def select_all(self):
        """Select all visible blocks in the scene."""
        try:
            if hasattr(self, 'viewer_widget') and hasattr(self.viewer_widget, 'select_all'):
                self.viewer_widget.select_all()
        except Exception as e:
            logger.debug(f"Select all failed: {e}")

    def deselect_all(self):
        """Clear current block selection."""
        try:
            if hasattr(self, 'viewer_widget') and hasattr(self.viewer_widget, 'deselect_all'):
                self.viewer_widget.deselect_all()
        except Exception as e:
            logger.debug(f"Deselect all failed: {e}")

    def invert_selection(self):
        """Invert the current block selection."""
        try:
            if hasattr(self, 'viewer_widget') and hasattr(self.viewer_widget, 'invert_selection'):
                self.viewer_widget.invert_selection()
        except Exception as e:
            logger.debug(f"Invert selection failed: {e}")

    # ── Help ──────────────────────────────────────────────────────

    def show_whats_new(self):
        """Show the What's New / changelog dialog."""
        QMessageBox.information(
            self, "What's New in GeoX",
            "See the release notes at:\nhttps://github.com/geox/geox/releases",
        )

    def check_for_updates(self):
        """Check for software updates."""
        QMessageBox.information(
            self, "Check for Updates",
            "You are running the latest version of GeoX.",
        )

    def report_bug(self):
        """Open the bug report page."""
        import webbrowser
        webbrowser.open("https://github.com/geox/geox/issues/new")

    def show_license_info(self):
        """Show the software license information."""
        QMessageBox.information(
            self, "License Information",
            "GeoX — Commercial Software\n\n"
            "This software is proprietary. Unauthorized copying, modification, "
            "or distribution is strictly prohibited.",
        )

    # ── Panel openers (not in panel_mixin) ───────────────────────

    def open_drillhole_import_panel(self):
        """Open the Drillhole Import panel."""
        from .drillhole_import_panel import DrillholeImportPanel
        self._show_or_create_dialog(
            'drillhole_import_dialog',
            lambda: DrillholeImportPanel(parent=None),
        )

    def import_leapfrog_project(self):
        """Import drillhole data and block models from a Leapfrog .aproj file."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
        from PyQt6.QtCore import Qt
        from pathlib import Path
        import logging as _logging

        _logger = _logging.getLogger(__name__)

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Leapfrog Project",
            "",
            "Leapfrog Projects (*.aproj);;All Files (*)",
        )
        if not path:
            return

        # Show progress dialog
        progress = QProgressDialog(
            "Reading Leapfrog project...", "Cancel", 0, 0, self
        )
        progress.setWindowTitle("Importing Leapfrog Project")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        try:
            from ..drillholes.leapfrog_importer import read_leapfrog_project

            result = read_leapfrog_project(path)

            progress.close()

            collars = result.get('collars')
            surveys = result.get('surveys')
            assays = result.get('assays')
            bm_names = result.get('block_model_names', [])
            metadata = result.get('metadata', {})

            has_collars = collars is not None and not collars.empty
            has_bm_defs = len(bm_names) > 0

            if not has_collars and not has_bm_defs:
                QMessageBox.warning(
                    self, "Import Warning",
                    "No drillhole or block model data found in the "
                    "Leapfrog project file.\n\n"
                    "The file may not contain supported data, or the "
                    "format may not be recognized."
                )
                return

            # Build summary
            summary_lines = [f"Source: {Path(path).name}"]

            if has_collars:
                summary_lines.append(f"Collars: {len(collars):,} holes")
                if surveys is not None and not surveys.empty:
                    summary_lines.append(f"Surveys: {len(surveys):,} records")
                if assays is not None and not assays.empty:
                    grade_cols = [c for c in assays.columns
                                  if c not in ('HOLEID', 'FROM', 'TO')]
                    summary_lines.append(
                        f"Assays: {len(assays):,} intervals, "
                        f"grades: {', '.join(grade_cols)}"
                    )

            if has_bm_defs:
                summary_lines.append("")
                for name in bm_names:
                    summary_lines.append(
                        f"Block Model: '{name}' (exported CSV required)"
                    )

            # Show summary and confirm
            msg = QMessageBox(self)
            msg.setWindowTitle("Leapfrog Import")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Successfully read Leapfrog project data:")
            msg.setInformativeText("\n".join(summary_lines))
            msg.setStandardButtons(
                QMessageBox.StandardButton.Ok
                | QMessageBox.StandardButton.Cancel
            )
            msg.setDefaultButton(QMessageBox.StandardButton.Ok)
            msg.button(QMessageBox.StandardButton.Ok).setText("Import into GeoX")

            if msg.exec() != QMessageBox.StandardButton.Ok:
                return

            # ── Prompt for block model CSV files ──
            bm_registered = 0
            if has_bm_defs:
                bm_registered = self._import_leapfrog_block_model_csvs(
                    bm_names, path, metadata, _logger,
                )

            # ── Register drillhole data ──
            if has_collars:
                progress2 = QProgressDialog(
                    "Building drillhole database...", None, 0, 0, self
                )
                progress2.setWindowTitle("Importing Leapfrog Project")
                progress2.setWindowModality(Qt.WindowModality.WindowModal)
                progress2.setMinimumDuration(0)
                progress2.show()

                try:
                    params = {
                        'collar_df': collars,
                        'survey_df': surveys,
                        'assay_df': assays,
                        'lithology_df': None,
                        'structures_df': None,
                    }

                    controller = getattr(self, 'controller', None)
                    if controller is None:
                        raise RuntimeError("Application controller not available")

                    def on_complete(task_result):
                        progress2.close()
                        if task_result is None or task_result.get('error'):
                            err = (task_result or {}).get('error', 'Unknown error')
                            QMessageBox.critical(
                                self, "Import Error",
                                f"Failed to build drillhole database:\n{err}"
                            )
                            return

                        drillhole_data = task_result.get('drillhole_data')
                        if drillhole_data is None:
                            QMessageBox.critical(
                                self, "Import Error",
                                "No drillhole data produced by import pipeline."
                            )
                            return

                        try:
                            reg = self.registry
                            if reg is None:
                                raise ValueError("DataRegistry not available")

                            import_metadata = {
                                'source': 'Leapfrog Geo (.aproj)',
                                'file': path,
                                **metadata,
                            }
                            reg.register_drillhole_data(
                                drillhole_data,
                                source_panel="Leapfrog Import",
                                metadata=import_metadata,
                            )

                            parts = [
                                f"Drillhole data imported ({len(collars)} collars)."
                            ]
                            if bm_registered > 0:
                                parts.append(
                                    f"{bm_registered} block model(s) imported."
                                )
                            QMessageBox.information(
                                self, "Import Complete",
                                "\n".join(parts)
                                + "\n\nUse Data > Drillholes > Drillhole Viewer "
                                "to visualize drillhole data."
                            )
                            _logger.info(
                                f"Leapfrog import complete: "
                                f"{len(collars)} collars, "
                                f"{bm_registered} block models"
                            )

                        except Exception as e:
                            _logger.error(
                                f"Failed to register Leapfrog data: {e}",
                                exc_info=True,
                            )
                            QMessageBox.critical(
                                self, "Registration Error",
                                f"Failed to register data:\n{e}"
                            )

                    controller.run_task(
                        'drillhole_import', params, callback=on_complete
                    )

                except Exception as e:
                    progress2.close()
                    raise
            else:
                # No drillholes — show block-model-only success
                if bm_registered > 0:
                    QMessageBox.information(
                        self, "Import Complete",
                        f"{bm_registered} block model(s) imported from "
                        f"Leapfrog project.\n\n"
                        f"No drillhole data was found in the project."
                    )

        except Exception as e:
            progress.close()
            _logger.error(f"Leapfrog import failed: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Import Error",
                f"Failed to import Leapfrog project:\n\n{e}"
            )

    def _import_leapfrog_block_model_csvs(
        self, bm_names, aproj_path, metadata, _logger,
    ) -> int:
        """Prompt user for Leapfrog block model CSV exports and register them.

        Leapfrog .aproj files store block model definitions but NOT the
        computed cell data (grades, coordinates).  Users must export block
        models as CSV from Leapfrog Geo, then select those CSV files here.

        Returns the number of block models successfully registered.
        """
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path

        names_str = ", ".join(f"'{n}'" for n in bm_names)
        reply = QMessageBox.question(
            self, "Block Models Detected",
            f"Block model(s) found in the Leapfrog project:\n"
            f"{names_str}\n\n"
            f"Leapfrog stores block model definitions but not the "
            f"computed cell data. To import block models, please "
            f"provide the exported CSV files (exported from Leapfrog "
            f"Geo via right-click > Export).\n\n"
            f"Would you like to select block model CSV files now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return 0

        # Open file dialog starting from the .aproj directory
        start_dir = str(Path(aproj_path).parent)
        csv_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Leapfrog Block Model CSV Files",
            start_dir,
            "CSV Files (*.csv);;All Files (*)",
        )

        if not csv_paths:
            return 0

        reg = self.registry
        if reg is None:
            _logger.warning("DataRegistry not available")
            return 0

        registered = 0
        for idx, csv_path in enumerate(csv_paths):
            try:
                csv_name = Path(csv_path).stem
                _logger.info(f"Loading Leapfrog block model CSV: {csv_path}")

                # Use the existing CSV parser which handles Leapfrog headers
                # (rotation, block size, etc.) automatically
                from ..parsers import parser_registry
                block_model = parser_registry.parse_file(Path(csv_path))

                if block_model is None:
                    _logger.warning(f"Failed to parse CSV: {csv_path}")
                    continue

                errors = block_model.validate()
                if errors:
                    _logger.warning(
                        f"Block model validation errors for '{csv_name}': "
                        f"{errors}"
                    )

                safe_name = (
                    csv_name.replace(' ', '_')
                    .replace('/', '_')
                    .replace('\\', '_')
                    .replace(':', '_')
                )
                reg.register_block_model(
                    block_model,
                    source_panel="Leapfrog Import",
                    metadata={
                        'source': 'Leapfrog Geo (CSV export)',
                        'aproj_file': aproj_path,
                        'csv_file': csv_path,
                        'block_model_name': csv_name,
                        **metadata,
                    },
                    model_id=f"leapfrog_{safe_name}",
                    set_as_current=(idx == 0),
                )
                registered += 1
                _logger.info(
                    f"Registered block model '{csv_name}': "
                    f"{block_model.block_count} blocks"
                )

            except Exception as e:
                _logger.error(
                    f"Failed to import block model CSV '{csv_path}': {e}",
                    exc_info=True,
                )
                QMessageBox.warning(
                    self, "Block Model Import Warning",
                    f"Failed to import '{Path(csv_path).name}':\n{e}"
                )

        if registered > 0:
            _logger.info(f"Registered {registered} block model(s) from CSV")

        return registered

    def open_geological_explorer_panel(self):
        """Open the Geological Explorer panel."""
        from .geological_explorer_panel import GeologicalExplorerPanel
        self._show_or_create_dialog(
            'geological_explorer_dialog',
            lambda: GeologicalExplorerPanel(registry=self.registry, parent=None),
        )

    def open_slope_risk_panel(self):
        """Open the Slope Risk panel."""
        from .slope_risk_panel import SlopeRiskPanel
        self._show_or_create_dialog(
            'slope_risk_dialog',
            lambda: SlopeRiskPanel(registry=self.registry, parent=None),
        )

    def open_stope_stability_panel(self):
        """Open the Stope Stability panel."""
        from .stope_stability_panel import StopeStabilityPanel
        self._show_or_create_dialog(
            'stope_stability_dialog',
            lambda: StopeStabilityPanel(registry=self.registry, parent=None),
        )

    def open_rockburst_panel(self):
        """Open the Rockburst panel."""
        from .rockburst_panel import RockburstPanel
        self._show_or_create_dialog(
            'rockburst_dialog',
            lambda: RockburstPanel(registry=self.registry, parent=None),
        )

    def open_seismic_panel(self):
        """Open the Seismic panel."""
        from .seismic_panel import SeismicPanel
        self._show_or_create_dialog(
            'seismic_dialog',
            lambda: SeismicPanel(registry=self.registry, parent=None),
        )

    # ── View toggles ──────────────────────────────────────────────

    def refresh_view(self):
        """Refresh / redraw the 3D scene."""
        try:
            if hasattr(self, 'viewer_widget') and self.viewer_widget is not None:
                self.viewer_widget.update()
                if hasattr(self.viewer_widget, 'renderer') and self.viewer_widget.renderer is not None:
                    self.viewer_widget.renderer.render()
        except Exception as e:
            logger.debug(f"refresh_view failed: {e}")

    def toggle_toolbars(self, checked: bool = True):
        """Show or hide all toolbars."""
        for tb in self.findChildren(QToolBar):
            tb.setVisible(checked)

    def toggle_statusbar(self, checked: bool = True):
        """Show or hide the status bar."""
        sb = self.statusBar()
        if sb is not None:
            sb.setVisible(checked)

    def toggle_fullscreen(self, checked: bool = False):
        """Toggle full-screen mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        fs_action = getattr(self, 'fullscreen_action', None)
        if fs_action is not None:
            fs_action.setChecked(self.isFullScreen())
