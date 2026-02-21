"""
InSAR Deformation Panel (ISCE-2 Orchestration).

Provides UI for configuring inputs, launching external ISCE-2 jobs via WSL2,
and ingesting results into the registry for viewing and export.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .base_panel import BaseDockPanel

logger = logging.getLogger(__name__)


class InSARPanel(BaseDockPanel):
    PANEL_ID = "InSARPanel"
    PANEL_NAME = "InSAR Deformation"

    def __init__(self, parent: Optional[QWidget] = None):
        # Safety: initialize private state only
        self._last_job_spec = None
        self._last_results = []
        super().__init__(parent)
        if hasattr(parent, "controller"):
            self.bind_controller(parent.controller)

    def setup_ui(self):
        self.setMinimumWidth(420)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        layout.addWidget(self._build_project_setup())
        layout.addWidget(self._build_data_manager())
        layout.addWidget(self._build_processing_control())
        layout.addWidget(self._build_results_viewer())
        layout.addWidget(self._build_live_log())

        layout.addStretch()
        self.setLayout(layout)

    def connect_signals(self):
        return

    # ------------------------------------------------------------------ #
    # UI Sections
    # ------------------------------------------------------------------ #
    def _build_project_setup(self) -> QGroupBox:
        group = QGroupBox("Project Setup")
        form = QFormLayout()

        self.aoi_mode = QComboBox()
        self.aoi_mode.addItems(["Polygon File", "Bounding Box"])
        form.addRow("AOI Mode", self.aoi_mode)

        self.aoi_polygon_path = QLineEdit()
        browse_aoi = QPushButton("Browse")
        browse_aoi.clicked.connect(self._browse_aoi_polygon)
        aoi_row = QHBoxLayout()
        aoi_row.addWidget(self.aoi_polygon_path)
        aoi_row.addWidget(browse_aoi)
        aoi_container = QWidget()
        aoi_container.setLayout(aoi_row)
        form.addRow("AOI Polygon", aoi_container)

        self.aoi_bbox = QLineEdit()
        self.aoi_bbox.setPlaceholderText("minx,miny,maxx,maxy")
        form.addRow("AOI BBox", self.aoi_bbox)

        self.dem_path = QLineEdit()
        browse_dem = QPushButton("Browse")
        browse_dem.clicked.connect(self._browse_dem)
        dem_row = QHBoxLayout()
        dem_row.addWidget(self.dem_path)
        dem_row.addWidget(browse_dem)
        dem_container = QWidget()
        dem_container.setLayout(dem_row)
        form.addRow("DEM", dem_container)

        self.orbit_source = QComboBox()
        self.orbit_source.addItems(["ESA", "ASF", "Auto"])
        form.addRow("Orbit Source", self.orbit_source)

        self.processing_mode = QComboBox()
        self.processing_mode.addItems(["PS", "SBAS", "Time-Series"])
        form.addRow("Processing Mode", self.processing_mode)

        group.setLayout(form)
        return group

    def _build_data_manager(self) -> QGroupBox:
        group = QGroupBox("Data Manager")
        layout = QVBoxLayout()

        self.sentinel_list = QListWidget()
        layout.addWidget(QLabel("Sentinel-1 Products"))
        layout.addWidget(self.sentinel_list)

        button_row = QHBoxLayout()
        add_btn = QPushButton("Add SAFE")
        add_btn.clicked.connect(self._add_sentinel_product)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_sentinel_product)
        button_row.addWidget(add_btn)
        button_row.addWidget(remove_btn)
        layout.addLayout(button_row)

        layout.addWidget(QLabel("Temporal/Perpendicular Baseline Summary"))
        self.baseline_summary = QLabel("No baseline summary yet.")
        self.baseline_summary.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.baseline_summary)

        group.setLayout(layout)
        return group

    def _build_processing_control(self) -> QGroupBox:
        group = QGroupBox("Processing Control")
        layout = QVBoxLayout()

        form = QFormLayout()
        self.workflow = QComboBox()
        self.workflow.addItems(["stripmapApp", "topsApp", "stackSentinel"])
        form.addRow("ISCE Workflow", self.workflow)

        self.wsl_distro = QLineEdit()
        self.wsl_distro.setPlaceholderText("Ubuntu-22.04 (optional)")
        form.addRow("WSL Distro", self.wsl_distro)

        self.output_dir = QLineEdit()
        browse_output = QPushButton("Browse")
        browse_output.clicked.connect(self._browse_output_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_dir)
        out_row.addWidget(browse_output)
        out_container = QWidget()
        out_container.setLayout(out_row)
        form.addRow("Output Directory", out_container)

        layout.addLayout(form)

        self.param_editor = QTextEdit()
        self.param_editor.setPlaceholderText("Advanced parameters (JSON)")
        self.param_editor.setText("{}")
        layout.addWidget(self.param_editor)

        control_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run_job)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._pause_job)
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.clicked.connect(self._resume_job)
        control_row.addWidget(self.run_btn)
        control_row.addWidget(self.pause_btn)
        control_row.addWidget(self.resume_btn)
        layout.addLayout(control_row)

        group.setLayout(layout)
        return group

    def _build_results_viewer(self) -> QGroupBox:
        group = QGroupBox("Results Viewer")
        layout = QVBoxLayout()

        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Type", "Registry Key", "Path", "Status"])
        self.results_table.setSortingEnabled(True)
        layout.addWidget(self.results_table)

        button_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Results")
        refresh_btn.clicked.connect(self._refresh_results)
        export_btn = QPushButton("Export Selected")
        export_btn.clicked.connect(self._export_selected)
        button_row.addWidget(refresh_btn)
        button_row.addWidget(export_btn)
        layout.addLayout(button_row)

        group.setLayout(layout)
        return group

    def _build_live_log(self) -> QGroupBox:
        group = QGroupBox("Live Log")
        layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(120)
        layout.addWidget(self.log_output)
        group.setLayout(layout)
        return group

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def run_job_from_menu(self):
        self._run_job()

    def ingest_results_from_menu(self):
        self._ingest_results()

    def _run_job(self):
        if not self.controller or not hasattr(self.controller, "insar"):
            self.show_warning("Unavailable", "InSAR controller is not available.")
            return

        params = self._gather_parameters()
        if not params:
            return

        try:
            result = self.controller.insar.run_job(params)
            self._append_log(result.get("summary", "InSAR run completed."))
            self._last_job_spec = result.get("job_spec")
            self._refresh_results()
        except Exception as exc:
            logger.error("InSAR run failed", exc_info=True)
            QMessageBox.critical(self, "Run Error", f"InSAR job failed:\n{exc}")

    def _pause_job(self):
        if self.controller and hasattr(self.controller, "insar"):
            self.controller.insar.pause_job()
        self._append_log("Pause requested (stub).")

    def _resume_job(self):
        if self.controller and hasattr(self.controller, "insar"):
            self.controller.insar.resume_job()
        self._append_log("Resume requested (stub).")

    def _refresh_results(self):
        if not self.controller or not hasattr(self.controller, "insar"):
            return
        results = self.controller.insar.list_results()
        self._last_results = results
        self._populate_results_table(results)

    def _ingest_results(self):
        if not self.controller or not hasattr(self.controller, "insar"):
            return
        output_dir = self.output_dir.text().strip()
        if not output_dir:
            QMessageBox.information(self, "Output Directory", "Select an output directory first.")
            return
        try:
            self.controller.insar.ingest_results(output_dir, self._gather_parameters() or {})
            self._append_log(f"Ingested results from {output_dir}")
            self._refresh_results()
        except Exception as exc:
            QMessageBox.critical(self, "Ingest Error", f"Failed to ingest results:\n{exc}")

    def _export_selected(self):
        row = self.results_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Select Result", "Select a result to export.")
            return
        path_item = self.results_table.item(row, 2)
        if not path_item:
            return
        export_path = QFileDialog.getSaveFileName(self, "Export Result")[0]
        if not export_path:
            return
        try:
            if self.controller and hasattr(self.controller, "insar"):
                self.controller.insar.export_result(path_item.text(), export_path)
            self._append_log(f"Exported to {export_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Export failed:\n{exc}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _gather_parameters(self) -> Optional[dict]:
        try:
            params = json.loads(self.param_editor.toPlainText() or "{}")
        except json.JSONDecodeError as exc:
            QMessageBox.warning(self, "Parameters", f"Invalid JSON:\n{exc}")
            return None

        params.update(
            {
                "aoi_mode": self.aoi_mode.currentText(),
                "aoi_polygon": self.aoi_polygon_path.text().strip() or None,
                "aoi_bbox": self.aoi_bbox.text().strip() or None,
                "dem_path": self.dem_path.text().strip() or None,
                "orbit_source": self.orbit_source.currentText(),
                "processing_mode": self.processing_mode.currentText(),
                "workflow": self.workflow.currentText(),
                "wsl_distro": self.wsl_distro.text().strip() or None,
                "output_dir": self.output_dir.text().strip() or None,
                "sentinel_products": [self.sentinel_list.item(i).text() for i in range(self.sentinel_list.count())],
            }
        )
        return params

    def _populate_results_table(self, results: list[dict]):
        self.results_table.setRowCount(0)
        for item in results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(item.get("type", "")))
            self.results_table.setItem(row, 1, QTableWidgetItem(item.get("key", "")))
            self.results_table.setItem(row, 2, QTableWidgetItem(item.get("path", "")))
            self.results_table.setItem(row, 3, QTableWidgetItem(item.get("status", "")))
        self.results_table.resizeColumnsToContents()

    def _append_log(self, message: str):
        existing = self.log_output.toPlainText()
        new_text = message if not existing else existing + "\n" + message
        self.log_output.setPlainText(new_text)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def _browse_aoi_polygon(self):
        path = QFileDialog.getOpenFileName(self, "Select AOI Polygon", "", "Vector files (*.geojson *.shp *.gpkg);;All files (*.*)")[0]
        if path:
            self.aoi_polygon_path.setText(path)

    def _browse_dem(self):
        path = QFileDialog.getOpenFileName(self, "Select DEM", "", "Raster files (*.tif *.tiff);;All files (*.*)")[0]
        if path:
            self.dem_path.setText(path)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir.setText(path)

    def _add_sentinel_product(self):
        path = QFileDialog.getOpenFileName(self, "Select Sentinel-1 SAFE", "", "SAFE (*.SAFE);;All files (*.*)")[0]
        if path:
            self.sentinel_list.addItem(path)

    def _remove_sentinel_product(self):
        row = self.sentinel_list.currentRow()
        if row >= 0:
            self.sentinel_list.takeItem(row)

