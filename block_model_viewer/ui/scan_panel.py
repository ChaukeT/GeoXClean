"""
Scan Analysis Panel

Provides the user interface for scan analysis pipeline.
Implements state machine UI with workflow steps and progress reporting.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from uuid import UUID

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QCheckBox, QRadioButton, QButtonGroup,
    QFileDialog, QMessageBox, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea
)

from .base_panel import BasePanel
from ..scans.scan_models import ScanProcessingMode, RegionGrowingParams, DBSCANParams

logger = logging.getLogger(__name__)


class ScanPanelState:
    """Enumeration of panel states."""
    EMPTY = "empty"  # No scan loaded
    FILE_LOADED = "file_loaded"  # File loaded but not validated
    VALIDATED = "validated"  # Validation complete
    CLEANED = "cleaned"  # Cleaning complete
    SEGMENTED = "segmented"  # Segmentation complete
    METRICS_READY = "metrics_ready"  # Metrics computed


class ScanPanel(BasePanel):
    """
    Scan Analysis Panel with state machine UI.

    Provides workflow-based interface for scan processing pipeline.
    """

    PANEL_ID = "scan_panel"

    # Signals
    scan_loaded = pyqtSignal(UUID)  # scan_id
    scan_processed = pyqtSignal(UUID)  # scan_id

    def __init__(self, parent=None, panel_id=None):
        self._current_state = ScanPanelState.EMPTY
        self._current_scan_id: Optional[UUID] = None
        self._fragment_labels: Optional[List[int]] = None
        self._fragment_metrics = None

        super().__init__(parent=parent, panel_id=panel_id)

    def setup_ui(self):
        """Set up the panel UI."""
        # Use existing layout from BasePanel, but customize margins/spacing
        if self.main_layout is None:
            self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Create UI sections
        self._create_header_section()
        self._create_workflow_section()
        self._create_progress_section()
        self._create_log_section()

        # Initialize to empty state
        self._update_ui_for_state()

    def _create_header_section(self):
        """Create header with title and current state info."""
        header_group = QGroupBox("Scan Analysis")
        header_layout = QVBoxLayout(header_group)

        self.state_label = QLabel("Status: No scan loaded")
        self.state_label.setStyleSheet("font-weight: bold; color: #666;")
        header_layout.addWidget(self.state_label)

        self.scan_info_label = QLabel("")
        header_layout.addWidget(self.scan_info_label)

        self.main_layout.addWidget(header_group)

    def _create_workflow_section(self):
        """Create workflow steps section."""
        workflow_group = QGroupBox("Analysis Workflow")
        workflow_layout = QVBoxLayout(workflow_group)

        # Step 1: Load Scan
        self.load_step_group = QGroupBox("Step 1: Load Scan")
        load_layout = QVBoxLayout(self.load_step_group)

        load_button_layout = QHBoxLayout()
        self.load_scan_button = QPushButton("Load Scan File...")
        self.load_scan_button.clicked.connect(self._on_load_scan)
        load_button_layout.addWidget(self.load_scan_button)

        self.clear_scan_button = QPushButton("Clear Scan")
        self.clear_scan_button.clicked.connect(self._on_clear_scan)
        self.clear_scan_button.setEnabled(False)
        load_button_layout.addWidget(self.clear_scan_button)

        load_layout.addLayout(load_button_layout)

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Processing Mode:"))

        self.preview_mode_radio = QRadioButton("Preview")
        self.analysis_mode_radio = QRadioButton("Analysis")
        self.preview_mode_radio.setChecked(True)

        mode_button_group = QButtonGroup()
        mode_button_group.addButton(self.preview_mode_radio)
        mode_button_group.addButton(self.analysis_mode_radio)

        mode_layout.addWidget(self.preview_mode_radio)
        mode_layout.addWidget(self.analysis_mode_radio)
        mode_layout.addStretch()

        load_layout.addLayout(mode_layout)

        workflow_layout.addWidget(self.load_step_group)

        # Step 2: Validate
        self.validate_step_group = QGroupBox("Step 2: Validate Data")
        validate_layout = QVBoxLayout(self.validate_step_group)

        self.validate_button = QPushButton("Validate Scan")
        self.validate_button.clicked.connect(self._on_validate_scan)
        self.validate_button.setEnabled(False)
        validate_layout.addWidget(self.validate_button)

        self.validation_results_text = QTextEdit()
        self.validation_results_text.setMaximumHeight(100)
        self.validation_results_text.setReadOnly(True)
        validate_layout.addWidget(self.validation_results_text)

        workflow_layout.addWidget(self.validate_step_group)

        # Step 3: Clean
        self.clean_step_group = QGroupBox("Step 3: Clean & Compute Normals")
        clean_layout = QVBoxLayout(self.clean_step_group)

        # Outlier removal
        outlier_layout = QFormLayout()
        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems(["statistical", "radius", "none"])
        outlier_layout.addRow("Outlier Method:", self.outlier_method_combo)

        # Normal estimation
        self.normal_method_combo = QComboBox()
        self.normal_method_combo.addItems(["auto", "pca", "open3d", "none"])
        outlier_layout.addRow("Normal Estimation:", self.normal_method_combo)

        clean_layout.addLayout(outlier_layout)

        self.clean_button = QPushButton("Clean & Compute Normals")
        self.clean_button.clicked.connect(self._on_clean_scan)
        self.clean_button.setEnabled(False)
        clean_layout.addWidget(self.clean_button)

        workflow_layout.addWidget(self.clean_step_group)

        # Step 4: Segment
        self.segment_step_group = QGroupBox("Step 4: Fragment Segmentation")
        segment_layout = QVBoxLayout(self.segment_step_group)

        # Strategy selection
        strategy_layout = QFormLayout()
        self.segmentation_strategy_combo = QComboBox()
        self.segmentation_strategy_combo.addItems(["region_growing", "dbscan"])
        self.segmentation_strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        strategy_layout.addRow("Strategy:", self.segmentation_strategy_combo)

        segment_layout.addLayout(strategy_layout)

        # Region growing parameters
        self.rg_params_group = QGroupBox("Region Growing Parameters")
        rg_layout = QFormLayout(self.rg_params_group)

        self.rg_normal_threshold_spin = QDoubleSpinBox()
        self.rg_normal_threshold_spin.setRange(0, 180)
        self.rg_normal_threshold_spin.setValue(30.0)
        rg_layout.addRow("Normal Threshold (°):", self.rg_normal_threshold_spin)

        self.rg_curvature_threshold_spin = QDoubleSpinBox()
        self.rg_curvature_threshold_spin.setRange(0, 1)
        self.rg_curvature_threshold_spin.setValue(0.01)
        self.rg_curvature_threshold_spin.setSingleStep(0.001)
        rg_layout.addRow("Curvature Threshold:", self.rg_curvature_threshold_spin)

        self.rg_k_neighbors_spin = QSpinBox()
        self.rg_k_neighbors_spin.setRange(3, 50)
        self.rg_k_neighbors_spin.setValue(15)
        rg_layout.addRow("K-Neighbors:", self.rg_k_neighbors_spin)

        segment_layout.addWidget(self.rg_params_group)

        # DBSCAN parameters
        self.dbscan_params_group = QGroupBox("DBSCAN Parameters")
        dbscan_layout = QFormLayout(self.dbscan_params_group)

        self.dbscan_epsilon_spin = QDoubleSpinBox()
        self.dbscan_epsilon_spin.setRange(0.001, 10.0)
        self.dbscan_epsilon_spin.setValue(0.05)
        self.dbscan_epsilon_spin.setSingleStep(0.01)
        dbscan_layout.addRow("Epsilon (m):", self.dbscan_epsilon_spin)

        self.dbscan_min_points_spin = QSpinBox()
        self.dbscan_min_points_spin.setRange(1, 100)
        self.dbscan_min_points_spin.setValue(20)
        dbscan_layout.addRow("Min Points:", self.dbscan_min_points_spin)

        self.dbscan_use_hdbscan_check = QCheckBox("Use HDBSCAN")
        dbscan_layout.addRow(self.dbscan_use_hdbscan_check)

        segment_layout.addWidget(self.dbscan_params_group)

        self.segment_button = QPushButton("Run Segmentation")
        self.segment_button.clicked.connect(self._on_segment_scan)
        self.segment_button.setEnabled(False)
        segment_layout.addWidget(self.segment_button)

        workflow_layout.addWidget(self.segment_step_group)

        # Step 5: Compute Metrics
        self.metrics_step_group = QGroupBox("Step 5: Compute Fragment Metrics")
        metrics_layout = QVBoxLayout(self.metrics_step_group)

        self.compute_metrics_button = QPushButton("Compute Metrics & PSD")
        self.compute_metrics_button.clicked.connect(self._on_compute_metrics)
        self.compute_metrics_button.setEnabled(False)
        metrics_layout.addWidget(self.compute_metrics_button)

        # Results display
        self.psd_results_label = QLabel("")
        metrics_layout.addWidget(self.psd_results_label)

        workflow_layout.addWidget(self.metrics_step_group)

        # Export section
        export_group = QGroupBox("Export Results")
        export_layout = QHBoxLayout(export_group)

        self.export_psd_button = QPushButton("Export PSD (CSV)")
        self.export_psd_button.clicked.connect(self._on_export_psd)
        self.export_psd_button.setEnabled(False)
        export_layout.addWidget(self.export_psd_button)

        self.export_fragments_button = QPushButton("Export Fragments")
        self.export_fragments_button.clicked.connect(self._on_export_fragments)
        self.export_fragments_button.setEnabled(False)
        export_layout.addWidget(self.export_fragments_button)

        workflow_layout.addWidget(export_group)

        self.main_layout.addWidget(workflow_group)

    def _create_progress_section(self):
        """Create progress reporting section."""
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_operation)
        self.cancel_button.setEnabled(False)
        progress_layout.addWidget(self.cancel_button)

        self.main_layout.addWidget(progress_group)

    def _create_log_section(self):
        """Create logging section."""
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        self.main_layout.addWidget(log_group)

    def _update_ui_for_state(self):
        """Update UI elements based on current state."""
        state = self._current_state

        # Update state label
        state_descriptions = {
            ScanPanelState.EMPTY: "No scan loaded",
            ScanPanelState.FILE_LOADED: "Scan file loaded",
            ScanPanelState.VALIDATED: "Scan validated",
            ScanPanelState.CLEANED: "Scan cleaned",
            ScanPanelState.SEGMENTED: "Fragments identified",
            ScanPanelState.METRICS_READY: "Metrics computed"
        }

        self.state_label.setText(f"Status: {state_descriptions.get(state, 'Unknown')}")

        # Enable/disable buttons based on state
        self.load_scan_button.setEnabled(state == ScanPanelState.EMPTY)
        self.clear_scan_button.setEnabled(state != ScanPanelState.EMPTY)
        self.validate_button.setEnabled(state == ScanPanelState.FILE_LOADED)
        self.clean_button.setEnabled(state == ScanPanelState.VALIDATED)
        self.segment_button.setEnabled(state == ScanPanelState.CLEANED)
        self.compute_metrics_button.setEnabled(state == ScanPanelState.SEGMENTED)

        # Export buttons
        self.export_psd_button.setEnabled(state == ScanPanelState.METRICS_READY)
        self.export_fragments_button.setEnabled(state == ScanPanelState.METRICS_READY)

        # Update step group styles
        self._update_step_group_styles()

        # Show/hide parameter groups based on strategy
        self._on_strategy_changed()

    def _update_step_group_styles(self):
        """Update visual styling of step groups based on completion."""
        state_order = [
            ScanPanelState.EMPTY,
            ScanPanelState.FILE_LOADED,
            ScanPanelState.VALIDATED,
            ScanPanelState.CLEANED,
            ScanPanelState.SEGMENTED,
            ScanPanelState.METRICS_READY
        ]

        current_index = state_order.index(self._current_state) if self._current_state in state_order else -1

        step_groups = [
            self.load_step_group,
            self.validate_step_group,
            self.clean_step_group,
            self.segment_step_group,
            self.metrics_step_group
        ]

        for i, group in enumerate(step_groups):
            if i < current_index:
                # Completed step
                group.setStyleSheet("QGroupBox { font-weight: bold; color: green; }")
            elif i == current_index:
                # Current step
                group.setStyleSheet("QGroupBox { font-weight: bold; color: blue; }")
            else:
                # Future step
                group.setStyleSheet("QGroupBox { color: #666; }")

    def _on_strategy_changed(self):
        """Handle segmentation strategy change."""
        strategy = self.segmentation_strategy_combo.currentText()

        # Show/hide parameter groups
        self.rg_params_group.setVisible(strategy == "region_growing")
        self.dbscan_params_group.setVisible(strategy == "dbscan")

    # Event handlers
    def _on_load_scan(self):
        """Handle load scan button click."""
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
                self._load_scan_file(filepath)

    def _load_scan_file(self, filepath: Path):
        """Load scan file and update state."""
        try:
            self._log_message(f"Loading scan file: {filepath}")

            # Get processing mode
            mode = ScanProcessingMode.PREVIEW if self.preview_mode_radio.isChecked() else ScanProcessingMode.ANALYSIS

            # Check if controller is available
            if self.controller:
                if hasattr(self.controller, 'scan_controller'):
                    # Use controller to load scan
                    scan_id = self.controller.scan_controller.load_scan(filepath)

                    if scan_id:
                        self._current_scan_id = scan_id
                        self._current_state = ScanPanelState.FILE_LOADED
                        self._update_ui_for_state()
                        self._update_scan_info()

                        self._log_message(f"Scan loaded successfully: {scan_id}")
                        self.scan_loaded.emit(scan_id)
                    else:
                        self._log_message("Failed to load scan file", error=True)
                else:
                    self._log_message("Controller has no scan_controller", error=True)
            else:
                self._log_message("Controller not available", error=True)

        except Exception as e:
            self._log_message(f"Error loading scan: {str(e)}", error=True)
            logger.error(f"Error loading scan file {filepath}: {e}", exc_info=True)

    def _on_clear_scan(self):
        """Handle clear scan button click."""
        if self._current_scan_id and self.controller:
            self.controller.scan_controller.delete_scan(self._current_scan_id)

        self._current_scan_id = None
        self._fragment_labels = None
        self._fragment_metrics = None
        self._current_state = ScanPanelState.EMPTY
        self._update_ui_for_state()
        self._update_scan_info()
        self._log_message("Scan cleared")

    def _on_validate_scan(self):
        """Handle validate scan button click."""
        if not self._current_scan_id or not self.controller:
            return

        self._run_async_task("validate_scan", {
            "scan_id": str(self._current_scan_id)
        })

    def _on_clean_scan(self):
        """Handle clean scan button click."""
        if not self._current_scan_id or not self.controller:
            return

        outlier_method = self.outlier_method_combo.currentText()
        normal_method = self.normal_method_combo.currentText()

        self._run_async_task("clean_scan", {
            "scan_id": str(self._current_scan_id),
            "outlier_method": outlier_method,
            "normal_method": normal_method
        })

    def _on_segment_scan(self):
        """Handle segment scan button click."""
        if not self._current_scan_id or not self.controller:
            return

        strategy = self.segmentation_strategy_combo.currentText()
        strategy_params = {}

        if strategy == "region_growing":
            strategy_params = {
                "normal_threshold_deg": self.rg_normal_threshold_spin.value(),
                "curvature_threshold": self.rg_curvature_threshold_spin.value(),
                "k_neighbors": self.rg_k_neighbors_spin.value()
            }
        elif strategy == "dbscan":
            strategy_params = {
                "epsilon": self.dbscan_epsilon_spin.value(),
                "min_points": self.dbscan_min_points_spin.value(),
                "use_hdbscan": self.dbscan_use_hdbscan_check.isChecked()
            }

        self._run_async_task("segment_scan", {
            "scan_id": str(self._current_scan_id),
            "strategy": strategy,
            "strategy_params": strategy_params
        })

    def _on_compute_metrics(self):
        """Handle compute metrics button click."""
        if not self._current_scan_id or not self._fragment_labels or not self.controller:
            return

        self._run_async_task("compute_metrics", {
            "scan_id": str(self._current_scan_id),
            "fragment_labels": self._fragment_labels
        })

    def _on_export_psd(self):
        """Handle export PSD button click."""
        # TODO: Implement PSD export
        self._log_message("PSD export not yet implemented")

    def _on_export_fragments(self):
        """Handle export fragments button click."""
        # TODO: Implement fragment export
        self._log_message("Fragment export not yet implemented")

    def _on_cancel_operation(self):
        """Handle cancel button click."""
        if self.controller:
            # TODO: Implement cancellation
            self._log_message("Cancellation not yet implemented")

    def _run_async_task(self, task_name: str, params: Dict[str, Any]):
        """Run an asynchronous task via the controller."""
        if not self.controller:
            self._log_message("Controller not available", error=True)
            return

        self._log_message(f"Starting task: {task_name}")
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Running {task_name}...")
        self.cancel_button.setEnabled(True)

        # Run task via controller
        try:
            if task_name == "validate_scan":
                result = self.controller.scan_controller.validate_scan(UUID(params["scan_id"]))
                self._on_validation_complete(result)
            elif task_name == "clean_scan":
                result = self.controller.scan_controller.clean_scan(
                    UUID(params["scan_id"]),
                    params["outlier_method"],
                    params["normal_method"]
                )
                self._on_cleaning_complete(result)
            elif task_name == "segment_scan":
                result = self.controller.scan_controller.segment_scan(
                    UUID(params["scan_id"]),
                    params["strategy"],
                    params["strategy_params"]
                )
                self._on_segmentation_complete(result)
            elif task_name == "compute_metrics":
                result = self.controller.scan_controller.compute_fragment_metrics(
                    UUID(params["scan_id"]),
                    params["fragment_labels"]
                )
                self._on_metrics_complete(result)

        except Exception as e:
            self._log_message(f"Task {task_name} failed: {str(e)}", error=True)
        finally:
            self.progress_bar.setValue(100)
            self.progress_label.setText("Complete")
            self.cancel_button.setEnabled(False)

    def _on_validation_complete(self, result):
        """Handle validation completion."""
        if result and result.is_valid:
            self._current_state = ScanPanelState.VALIDATED
            self._log_message("Validation successful")
        else:
            self._log_message("Validation failed", error=True)

        # Display validation results
        if result:
            violations_text = []
            for violation in result.violations:
                prefix = "ERROR:" if violation.is_error() else "WARNING:"
                violations_text.append(f"{prefix} {violation.message}")

            self.validation_results_text.setPlainText("\n".join(violations_text))

        self._update_ui_for_state()

    def _on_cleaning_complete(self, result):
        """Handle cleaning completion."""
        if result:
            self._current_scan_id = result  # New scan ID
            self._current_state = ScanPanelState.CLEANED
            self._log_message("Cleaning successful")
        else:
            self._log_message("Cleaning failed", error=True)

        self._update_ui_for_state()

    def _on_segmentation_complete(self, result):
        """Handle segmentation completion."""
        if result:
            self._current_scan_id, self._fragment_labels = result
            self._current_state = ScanPanelState.SEGMENTED
            self._log_message("Segmentation successful")
        else:
            self._log_message("Segmentation failed", error=True)

        self._update_ui_for_state()

    def _on_metrics_complete(self, result):
        """Handle metrics computation completion."""
        if result:
            results_id, self._fragment_metrics = result
            self._current_state = ScanPanelState.METRICS_READY
            self._log_message("Metrics computation successful")
            self._display_psd_results()
        else:
            self._log_message("Metrics computation failed", error=True)

        self._update_ui_for_state()

    def _display_psd_results(self):
        """Display PSD results in the UI."""
        if not self._fragment_metrics:
            return

        # Simple display of key metrics
        fragment_count = len(self._fragment_metrics)
        volumes = [fm.volume_m3 for fm in self._fragment_metrics if fm.volume_m3 > 0]

        if volumes:
            total_volume = sum(volumes)
            self.psd_results_label.setText(
                f"Fragments: {fragment_count} | "
                f"Total Volume: {total_volume:.3f} m³ | "
                f"P10: {self._compute_percentile_display(volumes, 10)} | "
                f"P50: {self._compute_percentile_display(volumes, 50)} | "
                f"P80: {self._compute_percentile_display(volumes, 80)}"
            )

    def _compute_percentile_display(self, values: List[float], percentile: float) -> str:
        """Compute percentile and format for display."""
        if not values:
            return "N/A"

        sorted_values = sorted(values)
        n = len(sorted_values)
        index = (n - 1) * (percentile / 100.0)

        if index.is_integer():
            return f"{sorted_values[int(index)]:.3f}"
        else:
            lower_idx = int(index)
            upper_idx = min(lower_idx + 1, n - 1)
            weight = index - lower_idx
            value = sorted_values[lower_idx] * (1 - weight) + sorted_values[upper_idx] * weight
            return f"{value:.3f}"

    def _update_scan_info(self):
        """Update scan information display."""
        if self._current_scan_id and self.controller:
            metadata = self.controller.scan_controller.registry.get_scan_metadata(self._current_scan_id)
            if metadata:
                info_text = (
                    f"File: {metadata.source_file.name} | "
                    f"Format: {metadata.file_format} | "
                    f"Points: {metadata.point_count:,}"
                )
                if metadata.crs:
                    info_text += f" | CRS: {metadata.crs}"
                self.scan_info_label.setText(info_text)
            else:
                self.scan_info_label.setText("")
        else:
            self.scan_info_label.setText("")

    def _log_message(self, message: str, error: bool = False):
        """Add message to log."""
        color = "red" if error else "black"
        self.log_text.append(f'<span style="color: {color};">{message}</span>')
        logger.info(message)
