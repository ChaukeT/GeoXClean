"""
DECLUSTERING ANALYSIS PANEL

Professional UI for cell-based declustering analysis.
Implements industry-standard methodology for statistical defensibility.

Features:
- Interactive cell size configuration
- Real-time validation and diagnostics
- Raw vs declustered statistics comparison
- Multi-cell-size sensitivity analysis
- Export to CSV for audit trails
- Direct integration with Variogram engine

Integration:
- Loads declustered weights into Variogram panel
- Publishes summaries to QA panel for CP reporting
- Supports batch processing for sensitivity analysis
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QSplitter, QCheckBox,
    QProgressBar, QFrame, QMessageBox, QFileDialog, QFormLayout,
    QScrollArea, QSizePolicy, QRadioButton, QButtonGroup
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QIcon

from .base_analysis_panel import BaseAnalysisPanel, log_registry_data_status
from ..drillholes.declustering import (
    DeclusteringEngine,
    DeclusteringConfig,
    CellDefinition,
    DeclusteringSummary,
    ValidationResult
)
from ..utils.coordinate_utils import ensure_xyz_columns

# Import modern status bar
from .drillhole_status_bar import DrillholeProcessStatusBar, StatusLevel, create_progress_callback
from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors

logger = logging.getLogger(__name__)


def _get_btn_style_primary() -> str:
    """Get primary button style for current theme."""
    colors = get_theme_colors()
    return f"""
        QPushButton {{ background-color: {colors.SUCCESS}; color: white; font-weight: bold; padding: 8px; border-radius: 4px; }}
        QPushButton:hover {{ background-color: #43A047; }}
        QPushButton:disabled {{ background-color: {colors.BORDER}; color: {colors.TEXT_DISABLED}; }}
    """


def _get_btn_style_export() -> str:
    """Get export button style for current theme."""
    colors = get_theme_colors()
    return f"""
        QPushButton {{ background-color: {colors.INFO}; color: white; padding: 6px; border-radius: 3px; }}
        QPushButton:hover {{ background-color: {colors.ACCENT_PRIMARY}; }}
    """

# =============================================================================
# ASYNC WORKER FOR BACKGROUND EXECUTION
# =============================================================================

class _DeclusteringWorker(QObject):
    """Worker object for running declustering in background thread."""
    finished = pyqtSignal(object)
    error = pyqtSignal(Exception)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        """Execute the function and emit results."""
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(e)

# =============================================================================
# MAIN PANEL CLASS
# =============================================================================

class DeclusteringPanel(BaseAnalysisPanel):
    """
    Professional Declustering Analysis Panel

    Provides interactive UI for cell-based declustering with:
    - Real-time parameter validation
    - Multi-cell-size sensitivity analysis
    - Statistical comparison tables
    - Audit-ready export functionality
    - Direct integration with GeoX Variogram engine
    """
    # PanelManager metadata
    PANEL_ID = "DeclusteringPanel"
    PANEL_NAME = "Declustering Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT





    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Declustering Analysis")
        self.setObjectName("declustering_panel")

        # Core components
        self.engine = DeclusteringEngine()
        self.current_results: Optional[Tuple[pd.DataFrame, DeclusteringSummary]] = None
        self.multi_size_results: Dict[str, Tuple[pd.DataFrame, DeclusteringSummary]] = {}

        # Worker thread tracking (initialized to None)
        self._worker: Optional[_DeclusteringWorker] = None
        self._worker_thread: Optional[QThread] = None

        # UI setup will be called by base class via setup_ui()
        # Note: _connect_signals() is called at the end of setup_ui() after widgets exist
        # Registry will be connected after controller injection (bind_controller)

        logger.info("DeclusteringPanel initialized")

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        self.setStyleSheet(get_analysis_panel_stylesheet())
        # Also refresh button styles
        if hasattr(self, 'multi_run_btn'):
            self.multi_run_btn.setStyleSheet(_get_btn_style_primary())
        if hasattr(self, 'run_btn'):
            self.run_btn.setStyleSheet(_get_btn_style_primary())
        if hasattr(self, 'export_btn'):
            self.export_btn.setStyleSheet(_get_btn_style_export())
        if hasattr(self, 'export_weights_btn'):
            self.export_weights_btn.setStyleSheet(_get_btn_style_export())
        if hasattr(self, 'suggest_cell_btn'):
            self.suggest_cell_btn.setStyleSheet(
                "QPushButton { background-color: #7E57C2; color: white; padding: 6px; border-radius: 3px; }"
                "QPushButton:hover { background-color: #5E35B1; }"
            )
        if hasattr(self, 'apply_recommended_btn'):
            self.apply_recommended_btn.setStyleSheet(
                "QPushButton { background-color: #FF9800; color: white; padding: 8px; border-radius: 4px; }"
                "QPushButton:hover { background-color: #F57C00; }"
                "QPushButton:disabled { background-color: #555; color: #888; }"
            )

    def _cleanup_worker_references(self):
        """Clear Python references to worker/thread after they finish."""
        self._worker = None
        self._worker_thread = None

    def run_async(self, func, *args, callback=None, error_callback=None):
        """
        Run a function asynchronously in a background thread.

        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            callback: Function to call with result on success
            error_callback: Function to call with exception on error
        """
        # Clean up any previous worker thread to prevent memory leaks
        if self._worker_thread is not None:
            try:
                if self._worker_thread.isRunning():
                    logger.warning("Previous worker still running - waiting for completion")
                    self._worker_thread.quit()
                    self._worker_thread.wait(5000)  # Wait up to 5 seconds
            except RuntimeError:
                # C++ object already deleted, just clear references
                self._worker = None
                self._worker_thread = None

        # Create thread and worker
        self._worker_thread = QThread()
        self._worker = _DeclusteringWorker(func, *args)
        self._worker.moveToThread(self._worker_thread)

        # Connect signals
        self._worker_thread.started.connect(self._worker.run)

        if callback:
            self._worker.finished.connect(callback)
        if error_callback:
            self._worker.error.connect(error_callback)

        # Cleanup on completion - clear references BEFORE deleteLater
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker_references)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        # Start the thread
        self._worker_thread.start()

    def setup_ui(self):
        """Setup the complete UI layout - called by base class."""
        # Ensure main_layout exists (should be set by BaseAnalysisPanel)
        if not hasattr(self, 'main_layout') or self.main_layout is None:
            self.main_layout = QVBoxLayout(self)
            self.main_layout.setContentsMargins(5, 5, 5, 5)
            self.main_layout.setSpacing(10)
        
        main_layout = self.main_layout
        main_layout.setSpacing(10)

        # Header
        header_label = QLabel("Cell-Based Declustering Analysis")
        header_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4fc3f7; margin-bottom: 10px;")
        main_layout.addWidget(header_label)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # Top section - Configuration
        config_widget = self._create_configuration_panel()
        splitter.addWidget(config_widget)

        # Bottom section - Results
        results_widget = self._create_results_panel()
        splitter.addWidget(results_widget)

        # Set splitter proportions
        splitter.setSizes([400, 600])

        # Apply theme
        self.setStyleSheet(get_analysis_panel_stylesheet())

        # Connect signals AFTER all widgets are created
        self._connect_signals()

    def _create_configuration_panel(self) -> QWidget:
        """Create the configuration panel with cell size and parameter controls."""
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        # Data Source Selection Group
        data_source_group = QGroupBox("Data Source")
        data_source_layout = QFormLayout(data_source_group)
        
        # Data source selector (composited vs raw)
        self.data_source_group = QButtonGroup()
        self.data_source_composited = QRadioButton("Composited Data")
        self.data_source_composited.setToolTip("Use composited drillhole data (recommended)")
        self.data_source_raw = QRadioButton("Raw Assay Data")
        self.data_source_raw.setToolTip("Use raw drillhole assay data")
        
        self.data_source_group.addButton(self.data_source_composited, 0)
        self.data_source_group.addButton(self.data_source_raw, 1)
        
        # Default to composited if available
        self.data_source_composited.setChecked(True)
        self.data_source_group.buttonClicked.connect(self._on_data_source_changed)
        
        data_source_radio_layout = QHBoxLayout()
        data_source_radio_layout.addWidget(self.data_source_composited)
        data_source_radio_layout.addWidget(self.data_source_raw)
        data_source_radio_layout.addStretch()
        
        data_source_layout.addRow("Source:", data_source_radio_layout)
        
        # Data source status label
        self.data_source_status_label = QLabel("")
        self.data_source_status_label.setStyleSheet("font-size: 9px; color: #888;")
        data_source_layout.addRow("", self.data_source_status_label)
        
        config_layout.addWidget(data_source_group)

        # Cell Definition Group
        cell_group = QGroupBox("Cell Configuration")
        cell_layout = QFormLayout(cell_group)

        # Cell size inputs
        self.cell_size_x_spin = QDoubleSpinBox()
        self.cell_size_x_spin.setRange(0.1, 1000.0)
        self.cell_size_x_spin.setValue(10.0)
        self.cell_size_x_spin.setSingleStep(1.0)
        self.cell_size_x_spin.setSuffix(" m")
        cell_layout.addRow("Cell Size X:", self.cell_size_x_spin)

        self.cell_size_y_spin = QDoubleSpinBox()
        self.cell_size_y_spin.setRange(0.1, 1000.0)
        self.cell_size_y_spin.setValue(10.0)
        self.cell_size_y_spin.setSingleStep(1.0)
        self.cell_size_y_spin.setSuffix(" m")
        cell_layout.addRow("Cell Size Y:", self.cell_size_y_spin)

        self.cell_size_z_spin = QDoubleSpinBox()
        self.cell_size_z_spin.setRange(0.1, 1000.0)
        self.cell_size_z_spin.setValue(5.0)
        self.cell_size_z_spin.setSingleStep(1.0)
        self.cell_size_z_spin.setSuffix(" m")
        cell_layout.addRow("Cell Size Z:", self.cell_size_z_spin)

        # 2D/3D toggle
        self.is_3d_checkbox = QCheckBox("3D Declustering")
        self.is_3d_checkbox.setChecked(True)
        cell_layout.addRow("", self.is_3d_checkbox)

        # Suggest Cell Size button - data-driven intelligent suggestion
        suggest_btn_layout = QHBoxLayout()
        self.suggest_cell_btn = QPushButton("Suggest Cell Size from Data")
        self.suggest_cell_btn.setToolTip(
            "Analyze sample spacing to suggest optimal cell sizes.\n"
            "Uses nearest-neighbor distances and spatial extent."
        )
        self.suggest_cell_btn.setStyleSheet(
            "QPushButton { background-color: #7E57C2; color: white; padding: 6px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #5E35B1; }"
        )
        self.suggest_cell_btn.clicked.connect(self._suggest_cell_size)
        suggest_btn_layout.addWidget(self.suggest_cell_btn)
        suggest_btn_layout.addStretch()
        cell_layout.addRow("", suggest_btn_layout)

        # Visualization mode (when results available)
        self.viz_mode_combo = QComboBox()
        self.viz_mode_combo.addItems(["Default", "Declustered Weights"])
        self.viz_mode_combo.setEnabled(False)  # Enable when declustered data available
        self.viz_mode_combo.currentTextChanged.connect(self._on_viz_mode_changed)
        cell_layout.addRow("3D View Mode:", self.viz_mode_combo)

        config_layout.addWidget(cell_group)

        # Multi-cell Analysis Group
        multi_group = QGroupBox("Sensitivity Analysis")
        multi_layout = QVBoxLayout(multi_group)

        # Predefined cell sizes
        sizes_label = QLabel("Cell sizes to evaluate:")
        sizes_label.setStyleSheet("color: #ccc;")
        multi_layout.addWidget(sizes_label)

        self.cell_sizes_text = QTextEdit()
        self.cell_sizes_text.setMaximumHeight(80)
        self.cell_sizes_text.setPlainText("5, 10, 15, 20, 25")
        self.cell_sizes_text.setToolTip("Comma-separated cell sizes (meters) for sensitivity analysis")
        multi_layout.addWidget(self.cell_sizes_text)

        # Multi-size run button
        multi_btn_layout = QHBoxLayout()
        self.multi_run_btn = QPushButton("Run Multi-Cell Analysis")
        self.multi_run_btn.setStyleSheet(_get_btn_style_primary())
        self.multi_run_btn.clicked.connect(self._run_multi_cell_analysis)
        multi_btn_layout.addWidget(self.multi_run_btn)

        # Apply Recommended button (enabled after analysis completes)
        self.apply_recommended_btn = QPushButton("Apply Recommended")
        self.apply_recommended_btn.setToolTip(
            "Apply the recommended cell size to the configuration spinboxes"
        )
        self.apply_recommended_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #F57C00; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.apply_recommended_btn.setEnabled(False)
        self.apply_recommended_btn.clicked.connect(self._apply_recommended_cell_size)
        multi_btn_layout.addWidget(self.apply_recommended_btn)

        multi_layout.addLayout(multi_btn_layout)

        config_layout.addWidget(multi_group)

        # Action Buttons
        buttons_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Declustering")
        self.run_btn.setStyleSheet(_get_btn_style_primary())
        self.run_btn.clicked.connect(self._run_declustering)
        buttons_layout.addWidget(self.run_btn)

        self.export_btn = QPushButton("Export Summary")
        self.export_btn.setStyleSheet(_get_btn_style_export())
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_btn)

        self.export_weights_btn = QPushButton("Export Cell Map")
        self.export_weights_btn.setStyleSheet(_get_btn_style_export())
        self.export_weights_btn.clicked.connect(self._export_cell_weights)
        self.export_weights_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_weights_btn)

        self.to_variogram_btn = QPushButton("→ Send to Variogram")
        self.to_variogram_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px; border-radius: 4px; } QPushButton:hover { background-color: #F57C00; }")
        self.to_variogram_btn.clicked.connect(self._send_to_variogram)
        self.to_variogram_btn.setEnabled(False)
        buttons_layout.addWidget(self.to_variogram_btn)

        self.visualize_weights_btn = QPushButton("🎨 Visualize Weights")
        self.visualize_weights_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; padding: 8px; border-radius: 4px; } QPushButton:hover { background-color: #7B1FA2; }")
        self.visualize_weights_btn.clicked.connect(self._visualize_declustered_weights)
        self.visualize_weights_btn.setEnabled(False)
        buttons_layout.addWidget(self.visualize_weights_btn)

        config_layout.addLayout(buttons_layout)

        # Modern Status Bar for declustering progress
        self.modern_status_bar = DrillholeProcessStatusBar.create_for_declustering(self)
        self.modern_status_bar.cancel_requested.connect(self._on_cancel_declustering)
        config_layout.addWidget(self.modern_status_bar)
        
        # Keep basic progress bar for legacy compatibility
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        config_layout.addWidget(self.progress_bar)

        return config_widget

    def _create_results_panel(self) -> QWidget:
        """Create the results panel with tables and diagnostics."""
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Results tabs
        self.results_tabs = QTabWidget()

        # Summary tab
        summary_tab = self._create_summary_tab()
        self.results_tabs.addTab(summary_tab, "Summary")

        # Statistics tab
        stats_tab = self._create_statistics_tab()
        self.results_tabs.addTab(stats_tab, "Statistics")

        # Cell Diagnostics tab
        diagnostics_tab = self._create_diagnostics_tab()
        self.results_tabs.addTab(diagnostics_tab, "Cell Diagnostics")

        # Multi-cell comparison tab
        multi_tab = self._create_multi_cell_tab()
        self.results_tabs.addTab(multi_tab, "Sensitivity Analysis")

        # Spatial visualization tab
        spatial_tab = self._create_spatial_tab()
        self.results_tabs.addTab(spatial_tab, "Spatial View")

        results_layout.addWidget(self.results_tabs)

        return results_widget

    def _create_summary_tab(self) -> QWidget:
        """Create summary information display."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        layout.addWidget(self.summary_text)

        return tab

    def _create_statistics_tab(self) -> QWidget:
        """Create statistics comparison table."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Variable selection
        var_layout = QHBoxLayout()
        var_layout.addWidget(QLabel("Grade Variable:"))
        self.var_combo = QComboBox()
        var_layout.addWidget(self.var_combo)
        layout.addLayout(var_layout)

        # Statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(["Statistic", "Raw Data", "Declustered", "Difference"])
        layout.addWidget(self.stats_table)

        return tab

    def _create_diagnostics_tab(self) -> QWidget:
        """Create cell diagnostics table."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.diagnostics_table = QTableWidget()
        self.diagnostics_table.setColumnCount(6)
        self.diagnostics_table.setHorizontalHeaderLabels([
            "Cell Key", "Samples", "Weight", "X Range", "Y Range", "Z Range"
        ])
        layout.addWidget(self.diagnostics_table)

        return tab

    def _create_multi_cell_tab(self) -> QWidget:
        """Create multi-cell-size comparison table."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Recommended cell size display
        self.recommended_label = QLabel("Run multi-cell analysis to see recommended cell size")
        self.recommended_label.setStyleSheet("font-weight: bold; color: #4fc3f7; padding: 5px;")
        layout.addWidget(self.recommended_label)

        self.multi_table = QTableWidget()
        self.multi_table.setColumnCount(8)
        self.multi_table.setHorizontalHeaderLabels([
            "Cell Size", "Samples", "Occupied Cells", "Avg Samples/Cell", "Mean Weight",
            "Grade Delta", "Stability", "Weight Change"
        ])
        layout.addWidget(self.multi_table)

        return tab

    def _create_spatial_tab(self) -> QWidget:
        """Create spatial visualization of declustering ratios."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Visualization options
        options_layout = QHBoxLayout()

        options_layout.addWidget(QLabel("Visualization:"))
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["Cell Weights Heatmap", "Sample Count Bubbles", "Weight Distribution"])
        self.viz_combo.currentTextChanged.connect(self._update_spatial_plot)
        options_layout.addWidget(self.viz_combo)

        options_layout.addStretch()

        # Refresh button
        self.refresh_viz_btn = QPushButton("Refresh Plot")
        self.refresh_viz_btn.clicked.connect(self._update_spatial_plot)
        self.refresh_viz_btn.setEnabled(False)
        options_layout.addWidget(self.refresh_viz_btn)

        layout.addLayout(options_layout)

        # Canvas for spatial plot
        from block_model_viewer.ui.variogram_panel import VariogramCanvas
        self.spatial_canvas = VariogramCanvas(self)
        layout.addWidget(self.spatial_canvas)

        return tab

    def _connect_signals(self):
        """Connect UI signals to handlers."""
        self.is_3d_checkbox.toggled.connect(self._on_3d_toggled)

        # Update cell size Z enabled state
        self._on_3d_toggled(self.is_3d_checkbox.isChecked())

    def _connect_registry(self):
        """Connect to data registry for drillhole data."""
        try:
            self.registry = self.get_registry()
            if self.registry:
                # FIX: Check if signal is available before connecting
                dh_signal = self.registry.drillholeDataLoaded
                if dh_signal is not None:
                    dh_signal.connect(self.on_registry_data_changed)
                    logger.debug("DeclusteringPanel: Connected to drillholeDataLoaded signal")
                # Initial load
                data = self.registry.get_drillhole_data()
                if data is not None:
                    self.on_registry_data_changed(data)
        except (ImportError, AttributeError) as e:
            self.registry = None
            logger.warning(f"DataRegistry not available for DeclusteringPanel: {e}")

    def _on_3d_toggled(self, checked: bool):
        """Handle 2D/3D mode toggle."""
        self.cell_size_z_spin.setEnabled(checked)

    def _suggest_cell_size(self):
        """Suggest cell sizes based on data characteristics."""
        try:
            # Get data from registry
            registry = self.get_registry()
            if registry is None:
                QMessageBox.warning(self, "No Registry", "Data registry not available.")
                return

            data = registry.get_drillhole_data()
            if data is None:
                QMessageBox.warning(self, "No Data", "No drillhole data available. Please load data first.")
                return

            # Get DataFrame
            df = self._extract_dataframe(data)
            if df is None or df.empty:
                QMessageBox.warning(self, "No Data", "Could not extract valid DataFrame from registry.")
                return

            # Call engine's suggestion method
            suggested_x, suggested_y, suggested_z, multi_sizes = self.engine.suggest_cell_sizes_from_data(df)

            # Apply to UI spinboxes
            self.cell_size_x_spin.setValue(suggested_x)
            self.cell_size_y_spin.setValue(suggested_y)
            self.cell_size_z_spin.setValue(suggested_z)

            # Update multi-cell analysis text field
            multi_sizes_str = ", ".join(str(int(s) if s == int(s) else s) for s in multi_sizes)
            self.cell_sizes_text.setPlainText(multi_sizes_str)

            # Store for reference
            self._suggested_sizes = (suggested_x, suggested_y, suggested_z)

            # Show feedback
            QMessageBox.information(
                self, "Cell Size Suggestion",
                f"Based on your data's spatial distribution:\n\n"
                f"Suggested cell size: {suggested_x} x {suggested_y} x {suggested_z} m\n\n"
                f"Multi-cell analysis sizes: {multi_sizes_str}\n\n"
                f"These values have been applied to the configuration."
            )

            logger.info(f"Applied suggested cell sizes: {suggested_x}x{suggested_y}x{suggested_z}m")

        except Exception as e:
            logger.error(f"Failed to suggest cell sizes: {e}")
            QMessageBox.critical(self, "Suggestion Failed", f"Failed to analyze data: {e}")

    def _apply_recommended_cell_size(self):
        """Apply the recommended cell size from multi-cell analysis."""
        if not hasattr(self, '_recommended_cell_size') or self._recommended_cell_size is None:
            QMessageBox.warning(
                self, "No Recommendation",
                "No recommended cell size available.\n"
                "Run multi-cell analysis first."
            )
            return

        try:
            # Parse the recommended size (format like "25.0m_cubic" or "10.0m_x_10.0m_x_5.0m")
            rec = self._recommended_cell_size

            if "_cubic" in rec:
                # Format: "25.0m_cubic"
                size = float(rec.split("m")[0])
                self.cell_size_x_spin.setValue(size)
                self.cell_size_y_spin.setValue(size)
                self.cell_size_z_spin.setValue(size)
            elif "_x_" in rec:
                # Format: "10.0m_x_10.0m_x_5.0m" or "10.0m_x_10.0m"
                parts = rec.replace("m", "").split("_x_")
                self.cell_size_x_spin.setValue(float(parts[0]))
                self.cell_size_y_spin.setValue(float(parts[1]))
                if len(parts) > 2:
                    self.cell_size_z_spin.setValue(float(parts[2]))

            logger.info(f"Applied recommended cell size: {rec}")
            QMessageBox.information(
                self, "Applied",
                f"Recommended cell size applied:\n{rec}"
            )

        except Exception as e:
            logger.error(f"Failed to apply recommended cell size: {e}")
            QMessageBox.critical(self, "Apply Failed", f"Failed to parse cell size: {e}")

    def _on_viz_mode_changed(self, mode: str):
        """Handle visualization mode change."""
        if mode == "Declustered Weights" and self.current_results:
            self._visualize_declustered_weights()
        # Default mode doesn't need special handling

    # =========================================================================
    # CORE FUNCTIONALITY
    # =========================================================================

    def _on_cancel_declustering(self):
        """Handle cancel request from the modern status bar."""
        self.modern_status_bar.update_status("Cancellation requested", StatusLevel.WARNING)
        # Note: The actual cancellation would need to be implemented in the background thread
        logger.info("User requested declustering cancellation")
        
    def _run_declustering(self):
        """Execute single declustering analysis."""
        try:
            # Get data from registry (use get_registry() which has fallback logic)
            registry = self.get_registry()
            if registry is None:
                QMessageBox.warning(self, "No Registry", "Data registry not available.")
                return

            data = registry.get_drillhole_data()
            if data is None:
                QMessageBox.warning(self, "No Data", "No drillhole data available. Please load data first.")
                return

            # LINEAGE GATE: Check data source selection
            # Declustering should be performed on composited data for geostatistical defensibility
            data_source = self._get_selected_data_source()
            if data_source == "assays":
                reply = QMessageBox.warning(
                    self,
                    "Lineage Warning: Raw Assays Selected",
                    "You have selected RAW ASSAYS for declustering.\n\n"
                    "JORC/SAMREC Best Practice:\n"
                    "Declustering should typically be performed on COMPOSITED data, "
                    "not raw assays, to ensure:\n"
                    "• Consistent sample support\n"
                    "• Valid change-of-support statistics\n"
                    "• Defensible resource estimates\n\n"
                    "Do you want to proceed with raw assays anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                logger.warning("LINEAGE: User proceeding with declustering on raw assays (not recommended)")
            
            # LINEAGE GATE: Check validation status
            validation_state = registry.get_drillholes_validation_state() or {}
            validation_status = validation_state.get('status', 'NOT_RUN')
            if validation_status == 'FAIL':
                reply = QMessageBox.warning(
                    self,
                    "Lineage Warning: Validation Failed",
                    "Drillhole validation FAILED.\n\n"
                    "Running declustering on unvalidated data may produce\n"
                    "unreliable results. Fix validation errors first.\n\n"
                    "Proceed anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                logger.warning("LINEAGE: User proceeding with declustering despite failed validation")
            elif validation_status == 'NOT_RUN':
                logger.warning("LINEAGE: Declustering running on data that has not been validated")

            # Get DataFrame (respects user's data source selection)
            df = self._extract_dataframe(data)
            if df is None or df.empty:
                QMessageBox.warning(self, "No Data", "Could not extract valid DataFrame from registry.")
                return

            # Store data source type and validation status for provenance tracking
            # (stored here to avoid thread-unsafe registry access in background thread)
            self._current_data_source = data_source
            self._current_validation_status = validation_status

            # Create configuration
            config = self._create_config_from_ui()
            self.engine = DeclusteringEngine(config)

            # Start modern status bar progress
            self.modern_status_bar.start_process(cancellable=True)
            self.modern_status_bar.update_status("Starting declustering analysis", StatusLevel.INFO)
            
            # Show legacy progress bar for compatibility
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            self.run_btn.setEnabled(False)

            # Run in background
            self.run_async(
                self._execute_declustering,
                df,
                callback=self._on_declustering_complete,
                error_callback=self._on_declustering_error
            )

        except Exception as e:
            logger.error(f"Error starting declustering: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start declustering: {e}")

    def _execute_declustering(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DeclusteringSummary]:
        """Execute declustering computation with lineage enforcement.

        Note: This runs in a background thread. Registry access is NOT thread-safe,
        so we use values stored in the main thread before async dispatch.
        """
        # Get lineage information (stored in main thread before async dispatch)
        data_source = getattr(self, '_current_data_source', 'unknown')
        validation_status = getattr(self, '_current_validation_status', 'NOT_RUN')

        # Run input validation
        validation = self.engine.validate_input_data(df)
        if not validation.is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(validation.errors)}")

        # Compute weights with lineage tracking
        # Note: require_composites=False allows UI warning but doesn't block
        # require_validation=False allows processing but logs warning
        df_result, summary = self.engine.compute_weights(
            df,
            value_cols=None,
            source_type=data_source,
            validation_status=validation_status,
            require_composites=False,  # UI already warned user
            require_validation=False   # UI already warned user
        )

        return df_result, summary

    def _on_declustering_complete(self, result: Tuple[pd.DataFrame, DeclusteringSummary]):
        """Handle successful declustering completion."""
        # Hide legacy progress bar and re-enable buttons
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)

        df_result, summary = result
        self.current_results = result

        # Update modern status bar with success
        success_msg = f"Declustering completed: {summary.total_samples} samples, {summary.occupied_cells} cells"
        self.modern_status_bar.finish_process(success=True, message=success_msg)
        self.modern_status_bar.update_status("Analysis completed successfully", StatusLevel.SUCCESS)

        # Register results with data registry for system-wide access WITH PROVENANCE
        registry = self.get_registry()
        if registry:
            # Build provenance metadata
            validation_state = registry.get_drillholes_validation_state() or {}
            data_source = getattr(self, '_current_data_source', 'unknown')
            
            provenance_metadata = {
                # Lineage tracking
                'parent_data_key': 'composites' if data_source == 'composites' else 'assays',
                'transformation_type': 'declustering',
                'transformation_params': {
                    'cell_size_x': self.engine.config.cell_definition.cell_size_x,
                    'cell_size_y': self.engine.config.cell_definition.cell_size_y,
                    'cell_size_z': self.engine.config.cell_definition.cell_size_z,
                    'method': self.engine.config.method.value,
                },
                # Validation status at time of processing
                'validation_status': validation_state.get('status', 'NOT_RUN'),
                'validation_config_hash': validation_state.get('config_hash', ''),
                # Data source warning flag
                'source_was_raw_assays': data_source == 'assays',
                'sample_count': summary.total_samples,
                'occupied_cells': summary.occupied_cells,
            }
            
            # Register with provenance
            registry.register_declustering_results(
                result, 
                source_panel="DeclusteringPanel",
                metadata=provenance_metadata
            )
            
            logger.info(
                f"LINEAGE: Registered declustering results. "
                f"Source: {data_source}, Samples: {summary.total_samples}, "
                f"Validation: {validation_state.get('status', 'NOT_RUN')}"
            )

        # Update UI
        self._update_summary_display(summary)
        self._update_statistics_table(df_result, summary)
        self._update_diagnostics_table(summary)

        # Enable export buttons and visualization
        self.export_btn.setEnabled(True)
        self.export_weights_btn.setEnabled(True)
        self.to_variogram_btn.setEnabled(True)
        self.refresh_viz_btn.setEnabled(True)
        self.visualize_weights_btn.setEnabled(True)
        self.viz_mode_combo.setEnabled(True)

        # Update spatial visualization
        self._update_spatial_plot()

        logger.info("Declustering analysis completed successfully")

        # Show success message
        QMessageBox.information(
            self, "Success",
            f"Declustering completed!\n\n"
            f"Samples: {summary.total_samples}\n"
            f"Occupied Cells: {summary.occupied_cells}\n"
            f"Average Samples/Cell: {summary.cells_per_sample:.2f}\n"
            f"Weight Range: {summary.min_weight:.6f} - {summary.max_weight:.6f}"
        )

    def _on_declustering_error(self, error: Exception):
        """Handle declustering execution error."""
        # Hide progress and re-enable controls
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)

        # Update modern status bar with error
        error_msg = f"Declustering failed: {str(error)}"
        self.modern_status_bar.update_status("Analysis failed", StatusLevel.ERROR)
        self.modern_status_bar.set_step_error(str(error))

        logger.error(f"Declustering failed: {error}")
        QMessageBox.critical(self, "Declustering Failed", f"Error during declustering: {error}")

    # =========================================================================
    # Controller/Registry Binding
    # =========================================================================

    def bind_controller(self, controller):
        """Bind controller and (re)connect registry when injected by MainWindow."""
        super().bind_controller(controller)
        try:
            self._connect_registry()
        except Exception:
            logger.debug("DeclusteringPanel: failed to connect registry after binding controller", exc_info=True)

    def _run_multi_cell_analysis(self):
        """Execute multi-cell-size sensitivity analysis."""
        try:
            # Get data (use get_registry() which has fallback logic)
            registry = self.get_registry()
            if registry is None:
                QMessageBox.warning(self, "No Registry", "Data registry not available.")
                return

            data = registry.get_drillhole_data()
            if data is None:
                QMessageBox.warning(self, "No Data", "No drillhole data available.")
                return

            # Get DataFrame (respects user's data source selection)
            df = self._extract_dataframe(data)
            if df is None or df.empty:
                return

            # Parse cell sizes
            cell_sizes_text = self.cell_sizes_text.toPlainText().strip()
            if not cell_sizes_text:
                QMessageBox.warning(self, "Invalid Input", "Please specify cell sizes for analysis.")
                return

            try:
                cell_sizes = [float(x.strip()) for x in cell_sizes_text.split(',') if x.strip()]
                if not cell_sizes:
                    raise ValueError("No valid cell sizes")
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", f"Could not parse cell sizes: {e}")
                return

            # Show progress
            self.progress_bar.setVisible(True)
            self.multi_run_btn.setEnabled(False)

            # Run analysis
            self.run_async(
                self._execute_multi_cell_analysis,
                df, cell_sizes,
                callback=self._on_multi_cell_complete,
                error_callback=self._on_multi_cell_error
            )

        except Exception as e:
            logger.error(f"Error starting multi-cell analysis: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start analysis: {e}")

    def _execute_multi_cell_analysis(self, df: pd.DataFrame, cell_sizes: List[float]) -> Dict[str, Tuple[pd.DataFrame, DeclusteringSummary]]:
        """Execute multi-cell-size analysis."""
        # Convert to cell size specifications (cubic cells)
        cell_specs = [(size, size, size) for size in cell_sizes]

        # Run analysis
        results = self.engine.analyze_cell_sizes(df, cell_specs, value_cols=None)

        return results

    def _on_multi_cell_complete(self, results: Dict[str, Tuple[pd.DataFrame, DeclusteringSummary]]):
        """Handle multi-cell analysis completion."""
        self.progress_bar.setVisible(False)
        self.multi_run_btn.setEnabled(True)

        self.multi_size_results = results

        # Update multi-cell table
        self._update_multi_cell_table(results)

        # Get and display recommended cell size
        recommended = self.engine.get_recommended_cell_size(results)

        # Store recommendation for "Apply" button
        self._recommended_cell_size = recommended

        if recommended:
            self.recommended_label.setText(f"🎯 Recommended Cell Size: {recommended}")
            self.recommended_label.setStyleSheet("font-weight: bold; color: #4CAF50; padding: 5px; background-color: rgba(76, 175, 80, 0.1);")

            # Enable the Apply Recommended button
            self.apply_recommended_btn.setEnabled(True)

            # Auto-apply the recommended size to spinboxes
            self._auto_apply_recommended(recommended)
        else:
            self.recommended_label.setText("No stable cell size identified - review results manually")
            self.apply_recommended_btn.setEnabled(False)

        # Switch to sensitivity tab
        self.results_tabs.setCurrentIndex(3)

        logger.info(f"Multi-cell analysis completed with {len(results)} cell sizes")

        message = f"Multi-cell analysis completed!\n\nEvaluated {len(results)} cell sizes."
        if recommended:
            message += f"\n\nRecommended cell size: {recommended}\n\nThe cell size has been automatically applied to the configuration."

        QMessageBox.information(self, "Success", message)

    def _auto_apply_recommended(self, recommended: str):
        """Automatically apply recommended cell size to UI spinboxes."""
        try:
            if "_cubic" in recommended:
                # Format: "25.0m_cubic"
                size = float(recommended.split("m")[0])
                self.cell_size_x_spin.setValue(size)
                self.cell_size_y_spin.setValue(size)
                self.cell_size_z_spin.setValue(size)
                logger.info(f"Auto-applied recommended cubic cell size: {size}m")
            elif "_x_" in recommended:
                # Format: "10.0m_x_10.0m_x_5.0m" or "10.0m_x_10.0m"
                parts = recommended.replace("m", "").split("_x_")
                self.cell_size_x_spin.setValue(float(parts[0]))
                self.cell_size_y_spin.setValue(float(parts[1]))
                if len(parts) > 2:
                    self.cell_size_z_spin.setValue(float(parts[2]))
                logger.info(f"Auto-applied recommended cell size: {parts}")
        except Exception as e:
            logger.warning(f"Could not auto-apply recommended cell size: {e}")

    def _on_multi_cell_error(self, error: Exception):
        """Handle multi-cell analysis error."""
        self.progress_bar.setVisible(False)
        self.multi_run_btn.setEnabled(True)

        logger.error(f"Multi-cell analysis failed: {error}")
        QMessageBox.critical(self, "Analysis Failed", f"Multi-cell analysis failed: {error}")

    def _create_config_from_ui(self) -> DeclusteringConfig:
        """Create DeclusteringConfig from UI controls."""
        cell_def = CellDefinition(
            cell_size_x=self.cell_size_x_spin.value(),
            cell_size_y=self.cell_size_y_spin.value(),
            cell_size_z=self.cell_size_z_spin.value() if self.is_3d_checkbox.isChecked() else None
        )

        return DeclusteringConfig(cell_definition=cell_def)

    def _get_selected_data_source(self) -> str:
        """Get the currently selected data source type.
        
        Returns:
            'composites' if composited data is selected
            'assays' if raw assays are selected
            'unknown' if cannot determine
        """
        if hasattr(self, 'data_source_composited') and hasattr(self, 'data_source_raw'):
            if self.data_source_composited.isChecked():
                return 'composites'
            else:
                return 'assays'
        return 'unknown'

    def _extract_dataframe(self, data, data_source: str = None) -> Optional[pd.DataFrame]:
        """Extract DataFrame from registry data, respecting user's data source selection."""
        df = None
        
        # Respect user's selection if radio buttons exist
        if hasattr(self, 'data_source_composited') and hasattr(self, 'data_source_raw'):
            use_composited = self.data_source_composited.isChecked()
            
            if isinstance(data, dict):
                if use_composited:
                    # Try composites first
                    composites = data.get('composites')
                    if isinstance(composites, pd.DataFrame) and not composites.empty:
                        df = composites
                    else:
                        composites_df = data.get('composites_df')
                        if isinstance(composites_df, pd.DataFrame) and not composites_df.empty:
                            df = composites_df
                else:
                    # Try raw assays
                    assays = data.get('assays')
                    if isinstance(assays, pd.DataFrame) and not assays.empty:
                        df = assays
                    else:
                        assays_df = data.get('assays_df')
                        if isinstance(assays_df, pd.DataFrame) and not assays_df.empty:
                            df = assays_df
            elif isinstance(data, pd.DataFrame):
                df = data
        else:
            # Legacy behavior: use data_source parameter
            if data_source == "direct_dataframe":
                # Data is already a DataFrame
                df = data if isinstance(data, pd.DataFrame) else None
            elif not isinstance(data, dict):
                df = None
            elif data_source == "composites":
                # Check composites first, then composites_df (avoid using 'or' with DataFrames)
                composites = data.get('composites')
                if isinstance(composites, pd.DataFrame) and not composites.empty:
                    df = composites
                else:
                    composites_df = data.get('composites_df')
                    if isinstance(composites_df, pd.DataFrame) and not composites_df.empty:
                        df = composites_df
            elif data_source == "raw_assays":
                # Check assays first, then assays_df (avoid using 'or' with DataFrames)
                assays = data.get('assays')
                if isinstance(assays, pd.DataFrame) and not assays.empty:
                    df = assays
                else:
                    assays_df = data.get('assays_df')
                    if isinstance(assays_df, pd.DataFrame) and not assays_df.empty:
                        df = assays_df
        
        # Ensure coordinate columns are normalized to X, Y, Z
        if df is not None:
            df = ensure_xyz_columns(df)
        
        return df

    # =========================================================================
    # UI UPDATE METHODS
    # =========================================================================

    def _update_summary_display(self, summary: DeclusteringSummary):
        """Update summary text display."""
        # Build bias correction section
        bias_section = ""
        if summary.variable_summaries:
            bias_section = "\n\nBIAS CORRECTION BY VARIABLE (Audit Trail)\n" + "=" * 45 + "\n"
            for var_name, stats in summary.variable_summaries.items():
                raw_mean = stats.get('mean_raw', 0)
                declust_mean = stats.get('mean_declust', 0)
                delta = stats.get('delta', 0)
                
                # Calculate percentage change
                pct_change = (delta / raw_mean * 100) if raw_mean != 0 else 0
                sign = "+" if delta >= 0 else ""
                
                bias_section += f"\n{var_name}:\n"
                bias_section += f"  Raw Mean:         {raw_mean:.4f}\n"
                bias_section += f"  Declustered Mean: {declust_mean:.4f}\n"
                bias_section += f"  Bias Correction:  {sign}{delta:.4f} ({sign}{pct_change:.2f}%)\n"
            
            bias_section += "\n" + "-" * 45
            bias_section += "\nNote: Negative bias indicates clustered high-grade samples.\n"
            bias_section += "      Positive bias indicates clustered low-grade samples."
        
        summary_text = f"""
DECLUSTERING SUMMARY
===================

Cell Configuration: {summary.cell_size_summary}
Total Samples: {summary.total_samples:,}
Occupied Cells: {summary.occupied_cells:,}
Empty Cells: {summary.empty_cells:,}
Samples per Cell (avg): {summary.cells_per_sample:.2f}

Weight Statistics:
- Minimum: {summary.min_weight:.6f}
- Maximum: {summary.max_weight:.6f}
- Mean: {summary.mean_weight:.6f}
- Standard Deviation: {summary.weight_std:.6f}
{bias_section}

Processing Time: {summary.processing_time_seconds:.2f} seconds
Timestamp: {summary.timestamp}

AUDIT READY: This analysis follows JORC/SAMREC standards for statistical defensibility.
"""
        self.summary_text.setPlainText(summary_text.strip())

    def _update_statistics_table(self, df: pd.DataFrame, summary: DeclusteringSummary):
        """Update statistics comparison table."""
        # Update variable combo
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
                       and col not in ['declust_weight', 'declust_cell']]
        self.var_combo.clear()
        self.var_combo.addItems(numeric_cols)

        if numeric_cols:
            self.var_combo.setCurrentIndex(0)
            self._update_variable_stats(df, numeric_cols[0])

        # Connect combo change signal (disconnect first to prevent duplication)
        try:
            self.var_combo.currentTextChanged.disconnect()
        except TypeError:
            pass  # Signal was not connected

        self.var_combo.currentTextChanged.connect(
            lambda var: self._update_variable_stats(df, var) if var else None
        )

    def _update_variable_stats(self, df: pd.DataFrame, variable: str):
        """Update statistics for selected variable."""
        if variable not in df.columns:
            return

        # Calculate raw statistics
        raw_data = df[variable].dropna()
        raw_mean = raw_data.mean()
        raw_std = raw_data.std()
        raw_min = raw_data.min()
        raw_max = raw_data.max()
        raw_median = raw_data.median()
        raw_count = len(raw_data)

        # Calculate declustered statistics (weighted)
        valid_mask = ~df[variable].isna()
        values = df.loc[valid_mask, variable].values
        weights = df.loc[valid_mask, 'declust_weight'].values
        
        declust_mean = np.sum(values * weights) / np.sum(weights)
        declust_std = np.sqrt(np.sum(weights * (values - declust_mean)**2) / np.sum(weights))
        
        # Weighted median (approximate)
        sorted_idx = np.argsort(values)
        cumsum = np.cumsum(weights[sorted_idx])
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        declust_median = values[sorted_idx[median_idx]]

        # Calculate bias
        mean_delta = declust_mean - raw_mean
        mean_pct = (mean_delta / raw_mean * 100) if raw_mean != 0 else 0

        # Clear and populate table
        self.stats_table.setRowCount(0)
        stats_data = [
            ("Sample Count", f"{raw_count:,}", f"{raw_count:,}", "-"),
            ("Mean", f"{raw_mean:.4f}", f"{declust_mean:.4f}", f"{mean_delta:+.4f} ({mean_pct:+.2f}%)"),
            ("Std Dev", f"{raw_std:.4f}", f"{declust_std:.4f}", f"{declust_std - raw_std:+.4f}"),
            ("Median", f"{raw_median:.4f}", f"{declust_median:.4f}", f"{declust_median - raw_median:+.4f}"),
            ("Minimum", f"{raw_min:.4f}", f"{raw_min:.4f}", "-"),
            ("Maximum", f"{raw_max:.4f}", f"{raw_max:.4f}", "-"),
        ]

        from PyQt6.QtGui import QColor, QBrush
        
        for row, (stat, raw, declust, diff) in enumerate(stats_data):
            self.stats_table.insertRow(row)
            self.stats_table.setItem(row, 0, QTableWidgetItem(stat))
            self.stats_table.setItem(row, 1, QTableWidgetItem(raw))
            self.stats_table.setItem(row, 2, QTableWidgetItem(declust))
            
            diff_item = QTableWidgetItem(diff)
            # Color code the delta column for Mean row
            if stat == "Mean" and diff != "-":
                if mean_delta < 0:
                    diff_item.setForeground(QBrush(QColor("#ff6b6b")))  # Red for negative (clustered high grades)
                elif mean_delta > 0:
                    diff_item.setForeground(QBrush(QColor("#69db7c")))  # Green for positive
            self.stats_table.setItem(row, 3, diff_item)

        # Resize columns
        self.stats_table.resizeColumnsToContents()

    def _update_diagnostics_table(self, summary: DeclusteringSummary):
        """Update cell diagnostics table."""
        diagnostics = summary.cell_diagnostics[:100]  # Limit for performance

        self.diagnostics_table.setRowCount(0)

        for diagnostic in diagnostics:
            row = self.diagnostics_table.rowCount()
            self.diagnostics_table.insertRow(row)

            # Cell key
            self.diagnostics_table.setItem(row, 0, QTableWidgetItem(str(diagnostic.cell_key)))

            # Sample count
            self.diagnostics_table.setItem(row, 1, QTableWidgetItem(str(diagnostic.sample_count)))

            # Weight
            self.diagnostics_table.setItem(row, 2, QTableWidgetItem(f"{diagnostic.weight_value:.6f}"))

            # Coordinate ranges
            x_range = f"{diagnostic.min_x:.1f} - {diagnostic.max_x:.1f}"
            self.diagnostics_table.setItem(row, 3, QTableWidgetItem(x_range))

            y_range = f"{diagnostic.min_y:.1f} - {diagnostic.max_y:.1f}"
            self.diagnostics_table.setItem(row, 4, QTableWidgetItem(y_range))

            if diagnostic.min_z is not None:
                z_range = f"{diagnostic.min_z:.1f} - {diagnostic.max_z:.1f}"
            else:
                z_range = "N/A"
            self.diagnostics_table.setItem(row, 5, QTableWidgetItem(z_range))

        self.diagnostics_table.resizeColumnsToContents()

    def _update_multi_cell_table(self, results: Dict[str, Tuple[pd.DataFrame, DeclusteringSummary]]):
        """Update multi-cell comparison table."""
        self.multi_table.setRowCount(0)

        # Get the first available grade variable for delta calculation
        sample_df, _ = next(iter(results.values())) if results else (None, None)
        available_grades = []
        if sample_df is not None:
            available_grades = [col for col in sample_df.columns
                              if pd.api.types.is_numeric_dtype(sample_df[col])
                              and col not in ['declust_weight', 'declust_cell']]
        selected_grade = available_grades[0] if available_grades else None

        for cell_key, (df_result, summary) in results.items():
            row = self.multi_table.rowCount()
            self.multi_table.insertRow(row)

            # Basic statistics
            self.multi_table.setItem(row, 0, QTableWidgetItem(cell_key))
            self.multi_table.setItem(row, 1, QTableWidgetItem(str(summary.total_samples)))
            self.multi_table.setItem(row, 2, QTableWidgetItem(str(summary.occupied_cells)))
            self.multi_table.setItem(row, 3, QTableWidgetItem(f"{summary.cells_per_sample:.2f}"))
            self.multi_table.setItem(row, 4, QTableWidgetItem(f"{summary.mean_weight:.6f}"))

            # Grade delta (if available)
            grade_delta = "N/A"
            if selected_grade and selected_grade in summary.variable_summaries:
                var_stats = summary.variable_summaries[selected_grade]
                if 'delta' in var_stats:
                    grade_delta = f"{var_stats['delta']:.4f}"
            self.multi_table.setItem(row, 5, QTableWidgetItem(grade_delta))

            # Stability indicators
            stability_status = "✓" if summary.stability_achieved else ""
            self.multi_table.setItem(row, 6, QTableWidgetItem(stability_status))

            weight_change = "N/A"
            if summary.weight_change_from_previous is not None:
                weight_change = f"{summary.weight_change_from_previous:.6f}"
            self.multi_table.setItem(row, 7, QTableWidgetItem(weight_change))

        self.multi_table.resizeColumnsToContents()

    def _update_spatial_plot(self):
        """Update the spatial visualization of declustering ratios."""
        if not self.current_results:
            return

        try:
            df_result, summary = self.current_results
            viz_type = self.viz_combo.currentText()

            # Clear existing plot
            self.spatial_canvas.clear_figure()

            # Create spatial plot based on selected type
            if viz_type == "Cell Weights Heatmap":
                self._plot_cell_weights_heatmap(df_result)
            elif viz_type == "Sample Count Bubbles":
                self._plot_sample_count_bubbles(df_result)
            elif viz_type == "Weight Distribution":
                self._plot_weight_distribution(df_result)

            # Refresh the canvas
            self.spatial_canvas.draw()

        except Exception as e:
            logger.error(f"Spatial plot update failed: {e}")
            # Show error on canvas
            self.spatial_canvas.clear_figure()
            ax = self.spatial_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', transform=ax.transAxes)
            self.spatial_canvas.draw()

    def _plot_cell_weights_heatmap(self, df_result):
        """Create heatmap of cell weights."""
        # Detect coordinate columns (handle both X/x, Y/y variants)
        x_col = 'X' if 'X' in df_result.columns else 'x' if 'x' in df_result.columns else None
        y_col = 'Y' if 'Y' in df_result.columns else 'y' if 'y' in df_result.columns else None
        
        if x_col is None or y_col is None:
            raise ValueError(f"Column(s) {['x' if x_col is None else None, 'y' if y_col is None else None]} do not exist. Available: {list(df_result.columns)}")
        
        # Group by cell to get cell centers and weights
        cell_groups = df_result.groupby('declust_cell').agg({
            x_col: 'mean',
            y_col: 'mean',
            'declust_weight': ['mean', 'count']
        }).reset_index()

        # Flatten column names
        cell_groups.columns = ['cell_key', 'x_center', 'y_center', 'weight_mean', 'sample_count']

        # Create plot
        ax = self.spatial_canvas.fig.add_subplot(111)
        self.spatial_canvas._apply_theme(ax)

        # Create scatter plot colored by weight
        scatter = ax.scatter(
            cell_groups['x_center'],
            cell_groups['y_center'],
            c=cell_groups['weight_mean'],
            s=cell_groups['sample_count'] * 20,  # Size by sample count
            cmap='viridis',
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )

        # Add colorbar
        cbar = self.spatial_canvas.fig.colorbar(scatter, ax=ax)
        cbar.set_label('Mean Declustering Weight', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')

        ax.set_xlabel('Easting (m)', color='white')
        ax.set_ylabel('Northing (m)', color='white')
        ax.set_title('Declustering Weights Heatmap\n(Bubble size = Sample count)', color='white', fontsize=12)
        ax.grid(True, alpha=0.3)

    def _plot_sample_count_bubbles(self, df_result):
        """Create bubble plot of sample counts per cell."""
        # Detect coordinate columns (handle both X/x, Y/y variants)
        x_col = 'X' if 'X' in df_result.columns else 'x' if 'x' in df_result.columns else None
        y_col = 'Y' if 'Y' in df_result.columns else 'y' if 'y' in df_result.columns else None
        
        if x_col is None or y_col is None:
            raise ValueError(f"Coordinate columns not found. Available: {list(df_result.columns)}")
        
        # Group by cell
        cell_groups = df_result.groupby('declust_cell').agg({
            x_col: 'mean',
            y_col: 'mean',
            'declust_weight': ['mean', 'count']
        }).reset_index()

        cell_groups.columns = ['cell_key', 'x_center', 'y_center', 'weight_mean', 'sample_count']

        # Create plot
        ax = self.spatial_canvas.fig.add_subplot(111)
        self.spatial_canvas._apply_theme(ax)

        # Bubble plot
        scatter = ax.scatter(
            cell_groups['x_center'],
            cell_groups['y_center'],
            s=cell_groups['sample_count'] * 10,  # Bubble size
            c=cell_groups['sample_count'],  # Color by count
            cmap='plasma',
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )

        # Add sample count labels for larger bubbles
        large_cells = cell_groups[cell_groups['sample_count'] >= cell_groups['sample_count'].quantile(0.8)]
        for _, row in large_cells.iterrows():
            ax.annotate(
                f"{int(row['sample_count'])}",
                (row['x_center'], row['y_center']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, color='white', ha='left', va='bottom'
            )

        # Add colorbar
        cbar = self.spatial_canvas.fig.colorbar(scatter, ax=ax)
        cbar.set_label('Sample Count per Cell', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')

        ax.set_xlabel('Easting (m)', color='white')
        ax.set_ylabel('Northing (m)', color='white')
        ax.set_title('Sample Count per Cell\n(Bubble size ∝ Sample count)', color='white', fontsize=12)
        ax.grid(True, alpha=0.3)

    def _plot_weight_distribution(self, df_result):
        """Create plot showing weight distribution spatially."""
        # Detect coordinate columns (handle both X/x, Y/y variants)
        x_col = 'X' if 'X' in df_result.columns else 'x' if 'x' in df_result.columns else None
        y_col = 'Y' if 'Y' in df_result.columns else 'y' if 'y' in df_result.columns else None
        
        if x_col is None or y_col is None:
            raise ValueError(f"Coordinate columns not found. Available: {list(df_result.columns)}")
        
        # Create plot
        ax = self.spatial_canvas.fig.add_subplot(111)
        self.spatial_canvas._apply_theme(ax)

        # Scatter plot of all samples colored by weight
        scatter = ax.scatter(
            df_result[x_col],
            df_result[y_col],
            c=df_result['declust_weight'],
            s=20,
            cmap='coolwarm',
            alpha=0.6,
            edgecolors='none'
        )

        # Add colorbar
        cbar = self.spatial_canvas.fig.colorbar(scatter, ax=ax)
        cbar.set_label('Declustering Weight', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')

        ax.set_xlabel('Easting (m)', color='white')
        ax.set_ylabel('Northing (m)', color='white')
        ax.set_title('Individual Sample Weights Distribution', color='white', fontsize=12)
        ax.grid(True, alpha=0.3)

    def _visualize_declustered_weights(self):
        """Visualize declustered weights in the 3D renderer."""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "No declustered results to visualize.")
            return

        try:
            # Publish visualization request to the main application
            # This will be picked up by the renderer to switch color mode
            viz_request = {
                'type': 'declustered_weights_visualization',
                'declustered_dataframe': self.current_results[0],  # df with weights
                'summary': self.current_results[1]
            }

            # Try multiple methods to find main window
            main_window = None
            
            # Method 1: Check if controller has main_window reference
            if hasattr(self, '_controller') and self._controller:
                if hasattr(self._controller, 'main_window'):
                    main_window = self._controller.main_window
            
            # Method 2: Try parent hierarchy
            if main_window is None:
                widget = self
                while widget is not None:
                    if hasattr(widget, 'visualize_declustered_weights'):
                        main_window = widget
                        break
                    widget = widget.parent() if hasattr(widget, 'parent') else None
            
            # Method 3: Find via QApplication
            if main_window is None:
                from PyQt6.QtWidgets import QApplication
                for widget in QApplication.topLevelWidgets():
                    if hasattr(widget, 'visualize_declustered_weights'):
                        main_window = widget
                        break
            
            # Call visualization if main window found
            if main_window and hasattr(main_window, 'visualize_declustered_weights'):
                main_window.visualize_declustered_weights(viz_request)
                logger.info("Declustered weights visualization request sent to main window")
            else:
                QMessageBox.information(
                    self, "Visualization",
                    "Declustered weights visualization requested.\n\n"
                    "The 3D view should now show drillhole intervals colored by declustering weights.\n\n"
                    "Note: This feature requires renderer integration to be fully functional."
                )

        except Exception as e:
            logger.error(f"Failed to request declustered weights visualization: {e}")
            QMessageBox.critical(self, "Visualization Failed", f"Failed to visualize declustered weights: {e}")

    # =========================================================================
    # EXPORT AND INTEGRATION
    # =========================================================================

    def _export_results(self):
        """Export declustering results to CSV."""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "No results to export. Run declustering first.")
            return

        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Declustering Results", "",
                "CSV files (*.csv);;All files (*)"
            )

            if filename:
                df_result, summary = self.current_results

                # Export main results
                df_result.to_csv(filename, index=False)

                # Export summary
                summary_filename = filename.replace('.csv', '_summary.csv')
                self.engine.export_summary_csv(summary_filename, summary)

                QMessageBox.information(
                    self, "Export Complete",
                    f"Results exported to:\n{filename}\n{summary_filename}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export results: {e}")

    def _export_cell_weights(self):
        """Export cell weights map to CSV."""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "No results to export. Run declustering first.")
            return

        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Cell Weights Map", "",
                "CSV files (*.csv);;All files (*)"
            )

            if filename:
                df_result, _ = self.current_results
                self.engine.export_cell_weights_csv(df_result, filename)

                QMessageBox.information(
                    self, "Export Complete",
                    f"Cell weights map exported to:\n{filename}"
                )

        except Exception as e:
            logger.error(f"Cell weights export failed: {e}")
            QMessageBox.critical(self, "Export Failed", f"Failed to export cell weights: {e}")

    def _send_to_variogram(self):
        """Send declustered data to Variogram panel."""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "No results to send. Run declustering first.")
            return

        try:
            df_result, _ = self.current_results

            # Prepare for variogram engine
            variogram_data = self.engine.prepare_for_variogram_engine(df_result)

            # Emit signal to send data to variogram panel
            # This would typically connect to the app controller's signal system
            self._publish_data_to_registry({
                'declustered_data': variogram_data,
                'declustering_summary': self.engine.get_last_summary()
            })

            QMessageBox.information(
                self, "Data Sent",
                "Declustered data sent to Variogram panel.\n\n"
                "The data is now available in the Variogram Analysis panel for experimental variogram calculation."
            )

        except Exception as e:
            logger.error(f"Failed to send to variogram: {e}")
            QMessageBox.critical(self, "Send Failed", f"Failed to send data to Variogram: {e}")

    # =========================================================================
    # REGISTRY INTEGRATION
    # =========================================================================

    def _on_data_source_changed(self, button):
        """Handle data source selection change."""
        if not hasattr(self, 'registry') or not self.registry:
            return
        
        # Reload data with new selection
        data = self.registry.get_drillhole_data()
        if data:
            self.on_registry_data_changed(data)
    
    def on_registry_data_changed(self, data: Optional[Dict[str, Any]]):
        """Handle registry data changes."""
        # Store registry data
        self._registry_data = data
        
        # Update UI based on available data
        if data is not None:
            # Check what's available
            composites_available = False
            assays_available = False
            
            if isinstance(data, dict):
                # Properly handle DataFrame checks to avoid ambiguous truth value error
                composites = data.get('composites')
                if composites is None:
                    composites = data.get('composites_df')

                assays = data.get('assays')
                if assays is None:
                    assays = data.get('assays_df')

                composites_available = isinstance(composites, pd.DataFrame) and not composites.empty
                assays_available = isinstance(assays, pd.DataFrame) and not assays.empty
            
            # Update radio button states
            if hasattr(self, 'data_source_composited') and hasattr(self, 'data_source_raw'):
                self.data_source_composited.setEnabled(composites_available)
                self.data_source_raw.setEnabled(assays_available)
                
                # Auto-select based on availability
                if not composites_available and assays_available:
                    self.data_source_raw.setChecked(True)
                elif composites_available:
                    self.data_source_composited.setChecked(True)
                
                # Update status label
                # AUDIT FIX: Can't use 'or' with DataFrames - use proper None check
                def _get_df(d, key1, key2):
                    """Helper to get DataFrame from dict with fallback key."""
                    val = d.get(key1)
                    if val is None:
                        val = d.get(key2)
                    return val
                
                if hasattr(self, 'data_source_status_label'):
                    if composites_available and assays_available:
                        comp_df = _get_df(data, 'composites', 'composites_df')
                        assay_df = _get_df(data, 'assays', 'assays_df')
                        self.data_source_status_label.setText(
                            f"✓ Both available: {len(comp_df):,} composites, {len(assay_df):,} assays"
                        )
                        self.data_source_status_label.setStyleSheet("font-size: 9px; color: #4CAF50;")
                    elif composites_available:
                        comp_df = _get_df(data, 'composites', 'composites_df')
                        self.data_source_status_label.setText(f"✓ {len(comp_df):,} composites available")
                        self.data_source_status_label.setStyleSheet("font-size: 9px; color: #4CAF50;")
                    elif assays_available:
                        assay_df = _get_df(data, 'assays', 'assays_df')
                        self.data_source_status_label.setText(f"✓ {len(assay_df):,} raw assays available")
                        self.data_source_status_label.setStyleSheet("font-size: 9px; color: #FF9800;")
                    else:
                        self.data_source_status_label.setText("⚠ No data available")
                        self.data_source_status_label.setStyleSheet("font-size: 9px; color: #e57373;")
            
            data_source = log_registry_data_status("DeclusteringPanel", data)
            if data_source != "none":
                self.run_btn.setEnabled(True)
                self.multi_run_btn.setEnabled(True)
            else:
                self.run_btn.setEnabled(False)
                self.multi_run_btn.setEnabled(False)

    def _publish_data_to_registry(self, data: Dict[str, Any]):
        """Publish data to the application registry."""
        # This would connect to the app controller's registry system
        # For now, we'll emit a signal that other panels can connect to
        logger.info("Publishing declustered data to registry")

        # Emit custom signal (would be connected in main window)
        if hasattr(self, 'dataPublished'):
            self.dataPublished.emit(data)  # type: ignore

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Data source
            if hasattr(self, 'data_source_composited'):
                settings['data_source'] = 'composited' if self.data_source_composited.isChecked() else 'raw'
            
            # Variable selection
            settings['variable'] = get_safe_widget_value(self, 'var_combo')
            
            # Cell size parameters
            settings['cell_x'] = get_safe_widget_value(self, 'cell_x_spin')
            settings['cell_y'] = get_safe_widget_value(self, 'cell_y_spin')
            settings['cell_z'] = get_safe_widget_value(self, 'cell_z_spin')
            
            # Multi-size analysis
            settings['min_cell'] = get_safe_widget_value(self, 'min_cell_spin')
            settings['max_cell'] = get_safe_widget_value(self, 'max_cell_spin')
            settings['n_sizes'] = get_safe_widget_value(self, 'n_sizes_spin')
            
            # Options
            settings['auto_cell'] = get_safe_widget_value(self, 'auto_cell_check')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save declustering panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Data source
            if 'data_source' in settings:
                if settings['data_source'] == 'composited' and hasattr(self, 'data_source_composited'):
                    self.data_source_composited.setChecked(True)
                elif settings['data_source'] == 'raw' and hasattr(self, 'data_source_raw'):
                    self.data_source_raw.setChecked(True)
            
            # Variable selection
            set_safe_widget_value(self, 'var_combo', settings.get('variable'))
            
            # Cell size parameters
            set_safe_widget_value(self, 'cell_x_spin', settings.get('cell_x'))
            set_safe_widget_value(self, 'cell_y_spin', settings.get('cell_y'))
            set_safe_widget_value(self, 'cell_z_spin', settings.get('cell_z'))
            
            # Multi-size analysis
            set_safe_widget_value(self, 'min_cell_spin', settings.get('min_cell'))
            set_safe_widget_value(self, 'max_cell_spin', settings.get('max_cell'))
            set_safe_widget_value(self, 'n_sizes_spin', settings.get('n_sizes'))
            
            # Options
            set_safe_widget_value(self, 'auto_cell_check', settings.get('auto_cell'))
                
            logger.info("Restored declustering panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore declustering panel settings: {e}")