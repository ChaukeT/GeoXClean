"""
RBF Interpolation Panel

Provides UI for configuring and running 3D RBF (Radial Basis Function) interpolation via the controller.
Supports anisotropy, polynomial drift, global/local/GPU modes, and continuous/classification modes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Optional dependencies
from .panel_manager import PanelCategory, DockArea
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    FigureCanvasQTAgg = None
    Figure = None

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QMessageBox,
    QTextEdit, QCheckBox, QFileDialog, QWidget, QSplitter,
    QScrollArea, QFrame, QDialog, QProgressBar, QRadioButton, QButtonGroup,
    QTabWidget, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from datetime import datetime

from ..utils.coordinate_utils import ensure_xyz_columns
from ..utils.variable_utils import (
    get_grade_columns, validate_variable, populate_variable_combo,
    get_variable_from_combo_or_fallback
)
from .base_analysis_panel import BaseAnalysisPanel, log_registry_data_status

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class RBFPanel(BaseAnalysisPanel):
    """
    Panel for configuring and launching 3D RBF Interpolation.
    Supports advanced RBF features including anisotropy, polynomial drift,
    and multiple computation modes.
    """
    # PanelManager metadata
    PANEL_ID = "RBFPanel"
    PANEL_NAME = "RBF Interpolation"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    task_name = "rbf"
    request_visualization = pyqtSignal(dict)  # Signal to request visualization in main viewer
    progress_updated = pyqtSignal(int, str)

    def __init__(self, parent=None):
        # Initialize state BEFORE calling super().__init__
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.variogram_results: Optional[Dict[str, Any]] = None
        self.rbf_results: Optional[Dict[str, Any]] = None
        self.transformation_metadata: Optional[Dict[str, Any]] = None
        self.registry = None

        # Pending payloads (if data loaded before UI ready)
        self._pending_drillhole_data = None
        self._pending_variogram_results = None
        self._ui_ready = False

        super().__init__(parent=parent, panel_id="rbf")

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, 'setStyleSheet'):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, 'refresh_theme'):
                child.refresh_theme()
        self.setWindowTitle("RBF Interpolation")
        self.resize(1200, 800)

        # Build UI (required when using _build_ui pattern)
        self._build_ui()

        # UI is now built
        self._ui_ready = True

        self._init_registry_connections()

        # Connect progress signal to update method
        self.progress_updated.connect(self._update_progress)
        self._process_pending_data()

    def _build_ui(self):
        """Build the RBF panel UI with tabbed layout."""
        # Use the main_layout set up by base class (inside scroll area)
        layout = self.main_layout

        # Create tab widget for organization
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Tab 1: Data & Variable Selection
        self._build_data_tab()

        # Tab 2: RBF Parameters
        self._build_rbf_parameters_tab()

        # Tab 3: Anisotropy Settings
        self._build_anisotropy_tab()

        # Tab 4: Computation Settings
        self._build_computation_tab()

        # Tab 5: Results & Diagnostics
        self._build_results_tab()

        # Progress and control buttons at bottom
        self._build_control_section(layout)

    def _build_data_tab(self):
        """Build data selection tab."""
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)

        # Data source section
        data_group = QGroupBox("Data Source")
        data_layout = QFormLayout(data_group)

        self.data_source_combo = QComboBox()
        self.data_source_combo.currentTextChanged.connect(self._on_data_source_changed)
        data_layout.addRow("Data Source:", self.data_source_combo)

        self.variable_combo = QComboBox()
        self.variable_combo.currentTextChanged.connect(self._on_variable_changed)
        data_layout.addRow("Variable:", self.variable_combo)

        layout.addWidget(data_group)

        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.data_preview_text = QTextEdit()
        self.data_preview_text.setMaximumHeight(200)
        self.data_preview_text.setReadOnly(True)
        preview_layout.addWidget(self.data_preview_text)

        layout.addWidget(preview_group)
        layout.addStretch()

        self.tab_widget.addTab(data_tab, "Data")

    def _build_rbf_parameters_tab(self):
        """Build RBF parameters tab."""
        params_tab = QWidget()
        layout = QVBoxLayout(params_tab)

        # RBF Kernel settings
        kernel_group = QGroupBox("RBF Kernel")
        kernel_layout = QFormLayout(kernel_group)

        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems([
            "thin_plate_spline",
            "linear",
            "cubic",
            "gaussian",
            "multiquadric",
            "inverse_multiquadric",
            "quintic"
        ])
        self.kernel_combo.setCurrentText("thin_plate_spline")
        kernel_layout.addRow("Kernel:", self.kernel_combo)

        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setRange(0.0, 1000.0)
        self.smoothing_spin.setValue(0.0)
        self.smoothing_spin.setSingleStep(0.1)
        kernel_layout.addRow("Smoothing:", self.smoothing_spin)

        layout.addWidget(kernel_group)

        # Polynomial drift
        drift_group = QGroupBox("Polynomial Drift")
        drift_layout = QFormLayout(drift_group)

        self.trend_degree_combo = QComboBox()
        self.trend_degree_combo.addItems(["None", "Constant (0)", "Linear (1)", "Quadratic (2)"])
        self.trend_degree_combo.setCurrentText("None")
        drift_layout.addRow("Degree:", self.trend_degree_combo)

        layout.addWidget(drift_group)

        # Mode settings
        mode_group = QGroupBox("Interpolation Mode")
        mode_layout = QFormLayout(mode_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["auto", "global", "local"])
        self.method_combo.setCurrentText("auto")
        mode_layout.addRow("Method:", self.method_combo)

        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(5, 1000)
        self.neighbors_spin.setValue(50)
        mode_layout.addRow("Neighbors (local):", self.neighbors_spin)

        self.classification_check = QCheckBox("Binary Classification Mode")
        self.classification_check.setToolTip("Treat values as binary classes (0/1) for classification")
        mode_layout.addRow(self.classification_check)

        layout.addWidget(mode_group)
        layout.addStretch()

        self.tab_widget.addTab(params_tab, "Parameters")

    def _build_anisotropy_tab(self):
        """Build anisotropy settings tab."""
        aniso_tab = QWidget()
        layout = QVBoxLayout(aniso_tab)

        # Anisotropy enable/disable
        self.anisotropy_enabled_check = QCheckBox("Enable Anisotropy")
        self.anisotropy_enabled_check.setChecked(False)
        self.anisotropy_enabled_check.stateChanged.connect(self._on_anisotropy_toggled)
        layout.addWidget(self.anisotropy_enabled_check)

        # Anisotropy parameters
        self.anisotropy_group = QGroupBox("Anisotropy Parameters")
        aniso_layout = QFormLayout(self.anisotropy_group)

        # Ranges
        self.range_x_spin = QDoubleSpinBox()
        self.range_x_spin.setRange(0.1, 10000.0)
        self.range_x_spin.setValue(50.0)
        aniso_layout.addRow("Range X:", self.range_x_spin)

        self.range_y_spin = QDoubleSpinBox()
        self.range_y_spin.setRange(0.1, 10000.0)
        self.range_y_spin.setValue(30.0)
        aniso_layout.addRow("Range Y:", self.range_y_spin)

        self.range_z_spin = QDoubleSpinBox()
        self.range_z_spin.setRange(0.1, 10000.0)
        self.range_z_spin.setValue(10.0)
        aniso_layout.addRow("Range Z:", self.range_z_spin)

        # Rotation angles (future extension)
        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(0.0, 360.0)
        self.azimuth_spin.setValue(0.0)
        self.azimuth_spin.setEnabled(False)  # Disabled for now
        aniso_layout.addRow("Azimuth (°):", self.azimuth_spin)

        self.dip_spin = QDoubleSpinBox()
        self.dip_spin.setRange(-90.0, 90.0)
        self.dip_spin.setValue(0.0)
        self.dip_spin.setEnabled(False)  # Disabled for now
        aniso_layout.addRow("Dip (°):", self.dip_spin)

        self.plunge_spin = QDoubleSpinBox()
        self.plunge_spin.setRange(-90.0, 90.0)
        self.plunge_spin.setValue(0.0)
        self.plunge_spin.setEnabled(False)  # Disabled for now
        aniso_layout.addRow("Plunge (°):", self.plunge_spin)

        layout.addWidget(self.anisotropy_group)

        # Status label for anisotropy loading
        self.anisotropy_status_label = QLabel("Anisotropy parameters will be auto-loaded from variogram results when available.")
        self.anisotropy_status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.anisotropy_status_label)

        # Info text
        info_label = QLabel(
            "Note: Currently supports range-based anisotropy only.\n"
            "Rotation angles will be implemented in future updates."
        )
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)

        layout.addStretch()
        self._on_anisotropy_toggled()  # Initialize state

        self.tab_widget.addTab(aniso_tab, "Anisotropy")

    def _init_registry_connections(self) -> None:
        """Connect to the DataRegistry."""
        try:
            self.registry = self.get_registry()
            if not self.registry:
                return

            # FIX: Check if signals are available before connecting
            dh_signal = self.registry.drillholeDataLoaded
            if dh_signal is not None:
                # AUDIT FIX: Use lambda to adapt signal signature (signal sends 1 arg, method expects 2)
                dh_signal.connect(lambda data: self.on_data_updated(data, "drillhole"))
                logger.debug("RBFPanel: Connected to drillholeDataLoaded signal")
            
            vario_signal = getattr(self.registry, 'variogramResultsLoaded', None)
            if vario_signal is not None:
                vario_signal.connect(self.on_variogram_results)
                logger.debug("RBFPanel: Connected to variogramResultsLoaded signal")

            # Load initial state
            existing_data = self.registry.get_drillhole_data()
            if existing_data:
                self.on_data_updated(existing_data, "drillhole")

            existing_vario = self.registry.get_variogram_results()
            if existing_vario:
                self.on_variogram_results(existing_vario)

            # Update data source combo
            self._update_data_source_combo()

            # Initialize anisotropy UI state
            self._on_anisotropy_toggled()

        except Exception as e:
            logger.debug(f"Registry init failed: {e}")

    def _process_pending_data(self):
        """Process any data that was loaded before UI was ready."""
        if self._pending_drillhole_data:
            self.on_data_updated(self._pending_drillhole_data, "drillhole")
            self._pending_drillhole_data = None
        if self._pending_variogram_results:
            self.on_variogram_results(self._pending_variogram_results)
            self._pending_variogram_results = None

    def _update_progress(self, percentage: int, message: str):
        """Update progress display."""
        # This method is connected to the progress_updated signal
        pass

    def _load_anisotropy_from_variogram(self, results: Dict[str, Any]):
        """Load anisotropy parameters from variogram results."""
        if not results or not self._ui_ready:
            return

        try:
            # Try to get anisotropy from combined_3d_model first
            combined = results.get('combined_3d_model', {})
            if combined:
                self._set_anisotropy_from_model(combined)
                logger.info("RBF Panel: Loaded anisotropy from combined_3d_model")
                return

            # Fallback to major variogram
            major_vario = results.get('major_variogram', {})
            if major_vario:
                self._set_anisotropy_from_model(major_vario)
                logger.info("RBF Panel: Loaded anisotropy from major variogram")
                return

            # Final fallback to omni variogram (isotropic)
            omni_vario = results.get('omni_variogram', {})
            if omni_vario:
                self._set_anisotropy_from_model(omni_vario)
                logger.info("RBF Panel: Loaded isotropic ranges from omni variogram")

        except Exception as e:
            logger.warning(f"RBF Panel: Failed to load anisotropy from variogram: {e}")

    def _set_anisotropy_from_model(self, model: Dict[str, Any]):
        """Set anisotropy UI controls from variogram model parameters."""
        # Extract ranges
        major_range = model.get('major_range', model.get('range', 50.0))
        minor_range = model.get('minor_range', major_range)
        vertical_range = model.get('vertical_range', major_range)

        # Extract angles if available (future extension)
        azimuth = model.get('azimuth', 0.0)
        dip = model.get('dip', 0.0)
        plunge = model.get('plunge', 0.0)

        # Update UI controls
        if hasattr(self, 'range_x_spin'):
            self.range_x_spin.setValue(float(major_range))
        if hasattr(self, 'range_y_spin'):
            self.range_y_spin.setValue(float(minor_range))
        if hasattr(self, 'range_z_spin'):
            self.range_z_spin.setValue(float(vertical_range))

        # Enable anisotropy if we have anisotropic ranges
        if hasattr(self, 'anisotropy_enabled_check'):
            has_anisotropy = abs(major_range - minor_range) > 1e-6 or abs(major_range - vertical_range) > 1e-6
            self.anisotropy_enabled_check.setChecked(has_anisotropy)

        # Update status label
        if hasattr(self, 'anisotropy_status_label'):
            if has_anisotropy:
                self.anisotropy_status_label.setText("✓ Anisotropy parameters loaded from variogram results.")
                self.anisotropy_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            else:
                self.anisotropy_status_label.setText("✓ Isotropic ranges loaded from variogram results.")
                self.anisotropy_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")

        logger.info(f"RBF Panel: Set anisotropy - X:{major_range:.1f}, Y:{minor_range:.1f}, Z:{vertical_range:.1f}")

    def _build_computation_tab(self):
        """Build computation settings tab."""
        comp_tab = QWidget()
        layout = QVBoxLayout(comp_tab)

        # Performance settings
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QFormLayout(perf_group)

        self.use_gpu_check = QCheckBox("Use GPU acceleration (CuPy)")
        self.use_gpu_check.setChecked(False)
        self.use_gpu_check.setToolTip("Requires CuPy installation for GPU acceleration")
        perf_layout.addRow(self.use_gpu_check)

        self.large_n_threshold_spin = QSpinBox()
        self.large_n_threshold_spin.setRange(1000, 1000000)
        self.large_n_threshold_spin.setValue(10000)
        perf_layout.addRow("Large N threshold:", self.large_n_threshold_spin)

        self.local_threshold_spin = QSpinBox()
        self.local_threshold_spin.setRange(1000, 100000)
        self.local_threshold_spin.setValue(5000)
        perf_layout.addRow("Local mode threshold:", self.local_threshold_spin)

        layout.addWidget(perf_group)

        # Grid settings
        grid_group = QGroupBox("Grid Settings")
        grid_layout = QFormLayout(grid_group)

        # We'll get grid info from the block model or allow manual specification
        self.grid_info_label = QLabel("Grid will be determined from block model or variogram results")
        self.grid_info_label.setStyleSheet("color: #666; font-style: italic;")
        grid_layout.addRow(self.grid_info_label)

        layout.addWidget(grid_group)

        # Diagnostics
        diag_group = QGroupBox("Diagnostics")
        diag_layout = QVBoxLayout(diag_group)

        self.run_diagnostics_check = QCheckBox("Run cross-validation diagnostics")
        self.run_diagnostics_check.setChecked(True)
        self.run_diagnostics_check.setToolTip("Compute MAE, RMSE, and R² on held-out data")
        diag_layout.addWidget(self.run_diagnostics_check)

        layout.addWidget(diag_group)
        layout.addStretch()

        self.tab_widget.addTab(comp_tab, "Computation")

    def _build_results_tab(self):
        """Build results and diagnostics tab."""
        results_tab = QWidget()
        layout = QVBoxLayout(results_tab)

        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(300)
        results_layout.addWidget(self.results_text)

        # Export buttons
        export_layout = QHBoxLayout()
        self.export_grid_btn = QPushButton("Export Grid (.npy)")
        self.export_grid_btn.clicked.connect(self._export_grid)
        self.export_grid_btn.setEnabled(False)
        export_layout.addWidget(self.export_grid_btn)

        self.export_csv_btn = QPushButton("Export Results (.csv)")
        self.export_csv_btn.clicked.connect(self._export_results)
        self.export_csv_btn.setEnabled(False)
        export_layout.addWidget(self.export_csv_btn)

        results_layout.addLayout(export_layout)
        layout.addWidget(results_group)

        # Diagnostics display
        diag_group = QGroupBox("Diagnostics")
        diag_layout = QVBoxLayout(diag_group)

        self.diagnostics_text = QTextEdit()
        self.diagnostics_text.setReadOnly(True)
        self.diagnostics_text.setMaximumHeight(200)
        diag_layout.addWidget(self.diagnostics_text)

        layout.addWidget(diag_group)
        layout.addStretch()

        self.tab_widget.addTab(results_tab, "Results")

    def _build_control_section(self, main_layout):
        """Build progress bar and control buttons."""
        # Progress section
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)

        main_layout.addLayout(progress_layout)

        # Control buttons
        button_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run RBF Interpolation")
        self.run_btn.clicked.connect(self._run_rbf)
        self.run_btn.setEnabled(False)
        button_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_rbf)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self._clear_results)
        button_layout.addWidget(self.clear_btn)

        main_layout.addLayout(button_layout)

    # Event handlers
    def _on_data_source_changed(self):
        """Handle data source selection change."""
        if not self._registry_data or not isinstance(self._registry_data, dict):
            return

        selected_source = self.data_source_combo.currentText()

        # Switch to selected data source
        new_data = None
        if selected_source == "Declustered Data":
            new_data = self._registry_data.get('declustered')
        elif selected_source == "Composite Data":
            new_data = self._registry_data.get('composites')
        elif selected_source == "Raw Assay Data":
            new_data = self._registry_data.get('assays')

        if new_data is not None and new_data is not self.drillhole_data:
            self.drillhole_data = new_data
            self._update_variable_combo()
            self._update_data_preview()
            self._validate_run_conditions()
            logger.info(f"RBF Panel: Switched to {selected_source}")

    def _on_variable_changed(self):
        """Handle variable selection change."""
        self._validate_run_conditions()

    def _on_anisotropy_toggled(self):
        """Handle anisotropy enable/disable."""
        enabled = self.anisotropy_enabled_check.isChecked()
        self.anisotropy_group.setEnabled(enabled)

    def _update_variable_combo(self):
        """Update variable combo box based on selected data source."""
        if self.drillhole_data is not None and hasattr(self.drillhole_data, 'empty'):
            populate_variable_combo(self.variable_combo, self.drillhole_data)
        else:
            # Clear combo if no valid data
            if hasattr(self, 'variable_combo'):
                self.variable_combo.clear()
        self._validate_run_conditions()

    def _update_data_source_combo(self):
        """Update data source combo box with available data sources."""
        if not hasattr(self, 'data_source_combo'):
            return

        self.data_source_combo.clear()

        # Add available data sources
        data_sources = []

        if self._registry_data:
            if isinstance(self._registry_data, dict):
                if self._registry_data.get('declustered') is not None:
                    data_sources.append("Declustered Data")
                if self._registry_data.get('composites') is not None:
                    data_sources.append("Composite Data")
                if self._registry_data.get('assays') is not None:
                    data_sources.append("Raw Assay Data")
            else:
                data_sources.append("Drillhole Data")

        if not data_sources:
            data_sources.append("No data available")

        self.data_source_combo.addItems(data_sources)

        # Set current selection based on what's loaded
        if self.drillhole_data is not None and self._registry_data:
            if isinstance(self._registry_data, dict):
                if self._registry_data.get('declustered') is not None and \
                   self.drillhole_data is self._registry_data.get('declustered'):
                    self.data_source_combo.setCurrentText("Declustered Data")
                elif self._registry_data.get('composites') is not None and \
                     self.drillhole_data is self._registry_data.get('composites'):
                    self.data_source_combo.setCurrentText("Composite Data")
                elif self._registry_data.get('assays') is not None and \
                     self.drillhole_data is self._registry_data.get('assays'):
                    self.data_source_combo.setCurrentText("Raw Assay Data")

    def _update_data_preview(self):
        """Update data preview text."""
        if self.drillhole_data is None or self.drillhole_data.empty:
            self.data_preview_text.setPlainText("No data available")
            return

        # Show basic statistics
        preview_text = f"Data shape: {self.drillhole_data.shape}\n\n"

        # Column info
        numeric_cols = self.drillhole_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            preview_text += "Numeric columns:\n"
            for col in numeric_cols[:10]:  # Show first 10
                stats = self.drillhole_data[col].describe()
                preview_text += f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, n={stats['count']:.0f}\n"
            if len(numeric_cols) > 10:
                preview_text += f"  ... and {len(numeric_cols) - 10} more\n"

        self.data_preview_text.setPlainText(preview_text)

    def _validate_run_conditions(self):
        """Validate if we can run RBF interpolation."""
        can_run = (
            self.drillhole_data is not None and
            not self.drillhole_data.empty and
            self.variable_combo.currentText() != ""
        )
        self.run_btn.setEnabled(can_run)

    def _run_rbf(self):
        """Execute RBF interpolation."""
        try:
            # Collect parameters
            params = self._collect_rbf_params()

            # Validate parameters
            if not self._validate_params(params):
                return

            # Update UI state
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Starting RBF interpolation...")

            # Connect progress signal
            self.progress_updated.connect(self._on_progress_update)

            # Run via controller
            self.controller.run_rbf_interpolation(
                params=params,
                callback=self._on_rbf_complete,
                progress_callback=self._on_progress_update
            )

        except Exception as e:
            logger.error(f"RBF execution failed: {e}")
            QMessageBox.critical(self, "RBF Error", f"Failed to start RBF interpolation:\n{str(e)}")
            self._reset_ui_state()

    def _stop_rbf(self):
        """Stop RBF interpolation."""
        # TODO: Implement stopping mechanism
        self._reset_ui_state()

    def _clear_results(self):
        """Clear results and diagnostics."""
        self.results_text.clear()
        self.diagnostics_text.clear()
        self.export_grid_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.rbf_results = None

    def _collect_rbf_params(self) -> Dict[str, Any]:
        """Collect all RBF parameters from UI."""
        variable_result = get_variable_from_combo_or_fallback(self.variable_combo, self.drillhole_data)
        if not variable_result.is_valid:
            raise ValueError(f"Invalid variable selection: {variable_result.error_message}")
        variable = variable_result.variable

        # RBF parameters
        params = {
            "variable": variable,
            "kernel": self.kernel_combo.currentText(),
            "smoothing": self.smoothing_spin.value(),
            "method": self.method_combo.currentText(),
            "neighbors": self.neighbors_spin.value(),
            "use_gpu": self.use_gpu_check.isChecked(),
            "classification": self.classification_check.isChecked(),
            "run_diagnostics": self.run_diagnostics_check.isChecked(),
            "random_state": 42,  # Fixed for reproducibility
        }

        # Trend degree
        trend_text = self.trend_degree_combo.currentText()
        if trend_text == "None":
            params["trend_degree"] = None
        else:
            params["trend_degree"] = int(trend_text.split("(")[1].split(")")[0])

        # Anisotropy
        params["anisotropy_enabled"] = self.anisotropy_enabled_check.isChecked()
        if params["anisotropy_enabled"]:
            params.update({
                "range_x": self.range_x_spin.value(),
                "range_y": self.range_y_spin.value(),
                "range_z": self.range_z_spin.value(),
                "azimuth": self.azimuth_spin.value(),
                "dip": self.dip_spin.value(),
                "plunge": self.plunge_spin.value(),
            })

        # Grid specification (will be set by controller based on block model)
        params["grid_spec"] = self._get_grid_spec()

        return params

    def _get_grid_spec(self) -> Dict[str, Any]:
        """Get grid specification from block model or default."""
        # Try to get from block model via controller
        if self.controller and hasattr(self.controller, 'block_model') and self.controller.block_model is not None:
            bm = self.controller.block_model
            return {
                "nx": bm.nx, "ny": bm.ny, "nz": bm.nz,
                "xmin": bm.xmin, "ymin": bm.ymin, "zmin": bm.zmin,
                "xinc": bm.xinc, "yinc": bm.yinc, "zinc": bm.zinc
            }

        # Default grid (shouldn't happen in normal usage)
        return {
            "nx": 50, "ny": 50, "nz": 20,
            "xmin": 0.0, "ymin": 0.0, "zmin": 0.0,
            "xinc": 10.0, "yinc": 10.0, "zinc": 5.0
        }

    def _validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate RBF parameters."""
        # Check if variable exists and has data
        if not params.get("variable"):
            QMessageBox.warning(self, "Validation Error", "Please select a variable to interpolate.")
            return False

        if self.drillhole_data is None or self.drillhole_data.empty:
            QMessageBox.warning(self, "Validation Error", "No drillhole data available.")
            return False

        variable = params["variable"]
        if variable not in self.drillhole_data.columns:
            QMessageBox.warning(self, "Validation Error", f"Variable '{variable}' not found in data.")
            return False

        # AUDIT FIX: Check data quality with better validation
        # First ensure coordinate columns exist
        df = self.drillhole_data
        coord_cols = ['X', 'Y', 'Z']
        missing_coords = [c for c in coord_cols if c not in df.columns]
        if missing_coords:
            QMessageBox.warning(
                self, "Validation Error",
                f"Missing coordinate columns: {missing_coords}\n\n"
                f"Ensure data has X, Y, Z columns (case-sensitive)."
            )
            return False

        valid_data = df.dropna(subset=["X", "Y", "Z", variable])
        n_valid = len(valid_data)
        
        # Minimum points based on polynomial degree
        trend_degree = params.get("trend_degree")
        min_points_map = {None: 4, 0: 4, 1: 10, 2: 20}  # Conservative minimums
        min_required = min_points_map.get(trend_degree, 10)
        
        if n_valid < min_required:
            QMessageBox.warning(
                self, "Validation Error",
                f"Insufficient valid data points: {n_valid}\n\n"
                f"For polynomial degree {trend_degree}, need at least {min_required} points.\n\n"
                f"Try:\n"
                f"  • Setting 'Polynomial Drift' to 'None'\n"
                f"  • Loading more drillhole data"
            )
            return False
        
        # AUDIT FIX: Check for data spread (warn about potential singularity)
        try:
            coords = valid_data[['X', 'Y', 'Z']].to_numpy()
            coord_ranges = coords.max(axis=0) - coords.min(axis=0)
            
            # Check if data is essentially 2D (one dimension has no spread)
            small_range_dims = coord_ranges < 1e-6 * max(coord_ranges)
            if small_range_dims.any() and trend_degree is not None and trend_degree >= 1:
                dim_names = ['X', 'Y', 'Z']
                flat_dims = [dim_names[i] for i, is_flat in enumerate(small_range_dims) if is_flat]
                reply = QMessageBox.question(
                    self, "Geometry Warning",
                    f"Data has minimal spread in: {', '.join(flat_dims)}\n\n"
                    f"This may cause 'Singular matrix' errors with polynomial drift.\n\n"
                    f"Recommended: Set 'Polynomial Drift' to 'None' or 'Constant (0)'.\n\n"
                    f"Continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return False
        except Exception as e:
            logger.warning(f"RBF geometry check failed: {e}")

        return True

    def _on_progress_update(self, percentage: int, message: str):
        """Handle progress updates."""
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(message)

    def _on_rbf_complete(self, result: Dict[str, Any]):
        """Handle RBF completion."""
        try:
            self._reset_ui_state()

            if "error" in result:
                QMessageBox.critical(self, "RBF Error", f"RBF interpolation failed:\n{result['error']}")
                return

            # Store results
            self.rbf_results = result

            # Update results display
            self._update_results_display(result)

            # Update diagnostics
            if result.get("diagnostics"):
                self._update_diagnostics_display(result["diagnostics"])

            # Enable export buttons
            self.export_grid_btn.setEnabled(True)
            self.export_csv_btn.setEnabled(True)

            # Register results with data registry
            try:
                if self.registry:
                    self.registry.register_rbf_results(result, source_panel="RBFPanel")
                    logger.info("RBF results registered with data registry")
            except Exception as e:
                logger.warning(f"Failed to register RBF results with registry: {e}")

            # Request visualization
            self.request_visualization.emit({
                "type": "rbf_results",
                "data": result
            })

            QMessageBox.information(self, "Success", "RBF interpolation completed successfully!")

        except Exception as e:
            logger.error(f"Error handling RBF results: {e}")
            QMessageBox.critical(self, "Results Error", f"Failed to process RBF results:\n{str(e)}")

    def _update_results_display(self, result: Dict[str, Any]):
        """Update results text display."""
        metadata = result.get("metadata", {})

        results_text = "RBF Interpolation Results\n"
        results_text += "=" * 50 + "\n\n"

        results_text += f"Method: {metadata.get('method', 'RBF Interpolation')}\n"
        results_text += f"Kernel: {metadata.get('kernel', 'N/A')}\n"
        results_text += f"Smoothing: {metadata.get('smoothing', 0.0)}\n"
        results_text += f"Method Type: {metadata.get('method_type', 'auto')}\n"
        results_text += f"GPU Enabled: {metadata.get('use_gpu', False)}\n"
        results_text += f"Classification: {metadata.get('classification', False)}\n"
        results_text += f"Sample Count: {metadata.get('n_samples', 0)}\n"
        results_text += f"Variable: {metadata.get('variable', 'N/A')}\n\n"

        if metadata.get("anisotropy_enabled"):
            results_text += "Anisotropy Settings:\n"
            results_text += f"  Range X: {metadata.get('range_x', 'N/A')}\n"
            results_text += f"  Range Y: {metadata.get('range_y', 'N/A')}\n"
            results_text += f"  Range Z: {metadata.get('range_z', 'N/A')}\n\n"

        if metadata.get("trend_degree") is not None:
            results_text += f"Polynomial Drift Degree: {metadata.get('trend_degree')}\n\n"

        # Grid info
        grid = result.get("grid")
        if grid is not None:
            results_text += f"Output Grid: {grid.dimensions}\n"
            results_text += f"Grid Bounds: X({grid.bounds[0]:.1f}, {grid.bounds[1]:.1f}) "
            results_text += f"Y({grid.bounds[2]:.1f}, {grid.bounds[3]:.1f}) "
            results_text += f"Z({grid.bounds[4]:.1f}, {grid.bounds[5]:.1f})\n\n"

        self.results_text.setPlainText(results_text)

    def _update_diagnostics_display(self, diagnostics: Dict[str, float]):
        """Update diagnostics text display."""
        diag_text = "Cross-Validation Diagnostics\n"
        diag_text += "=" * 30 + "\n\n"

        diag_text += f"Test Samples: {diagnostics.get('n_test', 'N/A')}\n"
        diag_text += f"MAE: {diagnostics.get('MAE', 'N/A'):.4f}\n"
        diag_text += f"RMSE: {diagnostics.get('RMSE', 'N/A'):.4f}\n"
        diag_text += f"R²: {diagnostics.get('R2', 'N/A'):.4f}\n\n"

        # Interpretation
        r2 = diagnostics.get('R2', 0.0)
        if r2 > 0.8:
            diag_text += "Excellent fit (R² > 0.8)\n"
        elif r2 > 0.6:
            diag_text += "Good fit (R² > 0.6)\n"
        elif r2 > 0.3:
            diag_text += "Moderate fit (R² > 0.3)\n"
        else:
            diag_text += "Poor fit (R² < 0.3)\n"

        self.diagnostics_text.setPlainText(diag_text)

    def _export_grid(self):
        """Export grid to .npy file."""
        if not self.rbf_results or "grid_values" not in self.rbf_results:
            QMessageBox.warning(self, "Export Error", "No grid data available for export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export RBF Grid", "", "NumPy files (*.npy);;All files (*)"
        )

        if filename:
            try:
                from ..geostats.rbf_interpolation import RBFModel3D
                grid_values = self.rbf_results["grid_values"]
                RBFModel3D.export_grid(None, grid_values, filename)  # Static method call
                QMessageBox.information(self, "Export Success",
                                      f"Grid exported successfully to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export grid:\n{str(e)}")

    def _export_results(self):
        """Export results to CSV."""
        if not self.rbf_results:
            QMessageBox.warning(self, "Export Error", "No results available for export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export RBF Results", "", "CSV files (*.csv);;All files (*)"
        )

        if filename:
            try:
                # Create results DataFrame
                results_data = {
                    "parameter": [],
                    "value": []
                }

                metadata = self.rbf_results.get("metadata", {})
                for key, value in metadata.items():
                    results_data["parameter"].append(key)
                    results_data["value"].append(str(value))

                if self.rbf_results.get("diagnostics"):
                    for key, value in self.rbf_results["diagnostics"].items():
                        results_data["parameter"].append(f"diagnostic_{key}")
                        results_data["value"].append(str(value))

                df = pd.DataFrame(results_data)
                df.to_csv(filename, index=False)

                QMessageBox.information(self, "Export Success",
                                      f"Results exported successfully to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")

    def _reset_ui_state(self):
        """Reset UI to ready state."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")

    # BaseAnalysisPanel interface methods
    def on_data_updated(self, data, data_type: str):
        """Handle data updates from registry."""
        if data_type == "drillhole":
            self._process_drillhole_data(data)

    def _process_drillhole_data(self, data):
        """
        Handle drillhole data loaded from registry.
        Prefers declustered data if available, otherwise uses composites/assays.
        """
        if not self._ui_ready:
            self._pending_drillhole_data = data
            return

        # Store registry data
        self._registry_data = data

        # Extract DataFrame - prefer declustered data
        df = None

        if isinstance(data, dict):
            # Priority: declustered > composites > assays
            declustered = data.get('declustered')
            composites = data.get('composites')
            assays = data.get('assays')

            if declustered is not None and hasattr(declustered, 'empty') and not declustered.empty:
                df = declustered
                logger.info("RBF Panel: Using declustered data")
            elif composites is not None and hasattr(composites, 'empty') and not composites.empty:
                df = composites
                logger.info("RBF Panel: Using composite data")
            elif assays is not None and hasattr(assays, 'empty') and not assays.empty:
                df = assays
                logger.info("RBF Panel: Using assay data")
            elif declustered is not None:
                df = declustered  # Fallback
            elif composites is not None:
                df = composites  # Fallback
            elif assays is not None:
                df = assays  # Fallback
        else:
            # Assume it's already a DataFrame
            df = data

        self.drillhole_data = df
        self._update_data_source_combo()
        self._update_variable_combo()
        self._update_data_preview()
        self._validate_run_conditions()

    def on_variogram_results(self, results: Dict[str, Any]):
        """Handle variogram results and auto-load anisotropy parameters."""
        self.variogram_results = results
        self._load_anisotropy_from_variogram(results)

    def get_required_data_types(self) -> list[str]:
        """Return required data types for this panel."""
        return ["drillhole"]
