"""
Fragmentation Import Panel
===========================

Panel for importing LiDAR point clouds and RGB imagery into a new FragmentDataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QComboBox, QFileDialog, QMessageBox,
    QDateEdit, QProgressBar,
)

from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea

logger = logging.getLogger(__name__)


class FragImportPanel(BaseDockPanel):
    """Panel for importing LiDAR/RGB data into a FragmentDataset."""

    PANEL_ID = "FragImportPanel"
    PANEL_NAME = "Fragmentation Import"
    PANEL_CATEGORY = PanelCategory.ANALYSIS
    PANEL_ICON = "scan"
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT
    PANEL_DEFAULT_VISIBLE = False
    PANEL_TOOLTIP = "Import LiDAR point cloud and RGB imagery for fragmentation analysis"

    import_completed = pyqtSignal(dict)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, panel_id=kwargs.get("panel_id"))

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Point Cloud / DXF File ---
        lidar_group = QGroupBox("Point Cloud / CAD Data")
        lidar_form = QFormLayout()

        self._las_path_edit = QLineEdit()
        self._las_path_edit.setPlaceholderText("Select LAS, PLY, DXF, or other point cloud file...")
        self._las_path_edit.setReadOnly(True)
        las_browse = QPushButton("Browse")
        las_browse.clicked.connect(self._browse_las)
        las_row = QHBoxLayout()
        las_row.addWidget(self._las_path_edit, 1)
        las_row.addWidget(las_browse)
        lidar_form.addRow("File:", las_row)

        self._point_count_label = QLabel("--")
        lidar_form.addRow("Points:", self._point_count_label)

        lidar_group.setLayout(lidar_form)
        layout.addWidget(lidar_group)

        # --- RGB Imagery ---
        rgb_group = QGroupBox("RGB Imagery (Optional)")
        rgb_form = QFormLayout()

        self._rgb_path_edit = QLineEdit()
        self._rgb_path_edit.setPlaceholderText("Select image files...")
        self._rgb_path_edit.setReadOnly(True)
        rgb_browse = QPushButton("Browse")
        rgb_browse.clicked.connect(self._browse_rgb)
        rgb_row = QHBoxLayout()
        rgb_row.addWidget(self._rgb_path_edit, 1)
        rgb_row.addWidget(rgb_browse)
        rgb_form.addRow("Images:", rgb_row)

        rgb_group.setLayout(rgb_form)
        layout.addWidget(rgb_group)

        # --- Metadata ---
        meta_group = QGroupBox("Metadata")
        meta_form = QFormLayout()

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Lift2_Blast_2026-03-15")
        meta_form.addRow("Dataset Name:", self._name_edit)

        self._crs_combo = QComboBox()
        self._crs_combo.setEditable(True)
        self._crs_combo.addItems([
            "", "EPSG:32633", "EPSG:32634", "EPSG:32635",
            "EPSG:32736", "EPSG:32737", "EPSG:4326",
        ])
        self._crs_combo.setCurrentText("")
        meta_form.addRow("CRS:", self._crs_combo)

        self._date_edit = QDateEdit()
        self._date_edit.setCalendarPopup(True)
        from PyQt6.QtCore import QDate
        self._date_edit.setDate(QDate.currentDate())
        meta_form.addRow("Acquisition Date:", self._date_edit)

        meta_group.setLayout(meta_form)
        layout.addWidget(meta_group)

        # --- Import Button ---
        self._import_btn = QPushButton("Import Dataset")
        self._import_btn.setMinimumHeight(36)
        self._import_btn.setEnabled(False)
        self._import_btn.clicked.connect(self._on_import)
        layout.addWidget(self._import_btn)

        # --- Progress ---
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # --- Status ---
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        layout.addStretch()

        if self.main_layout:
            self.main_layout.addLayout(layout)
        else:
            self.setLayout(layout)

    # ------------------------------------------------------------------
    # File Browsing
    # ------------------------------------------------------------------

    def _browse_las(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Point Cloud / CAD File", "",
            "All Supported (*.las *.laz *.ply *.obj *.xyz *.dxf);;"
            "LiDAR Files (*.las *.laz);;"
            "Point Cloud (*.ply *.obj *.xyz);;"
            "DXF CAD Files (*.dxf);;"
            "All Files (*)"
        )
        if path:
            self._las_path_edit.setText(path)
            self._update_preview(path)
            self._import_btn.setEnabled(True)
            if not self._name_edit.text():
                self._name_edit.setText(Path(path).stem)

    def _browse_rgb(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select RGB Images", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff);;All Files (*)"
        )
        if paths:
            self._rgb_path_edit.setText("; ".join(paths))

    def _update_preview(self, file_path: str):
        """Show quick point count preview."""
        ext = Path(file_path).suffix.lower()
        try:
            if ext in ('.las', '.laz'):
                import laspy
                with laspy.open(file_path) as f:
                    count = f.header.point_count
                self._point_count_label.setText(f"{count:,}")
            elif ext == '.dxf':
                from ..parsers.dxf_parser import DXFParser
                result = DXFParser().parse(Path(file_path))
                n_pts = 0
                for s in result.get('surfaces', []):
                    n_pts += s.n_points
                for l in result.get('lines', []):
                    n_pts += l.n_points
                pts = result.get('points')
                if pts is not None:
                    n_pts += pts.n_points
                self._point_count_label.setText(f"{n_pts:,} (DXF vertices)")
            else:
                self._point_count_label.setText("(preview on import)")
        except Exception:
            self._point_count_label.setText("(preview unavailable)")

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _on_import(self):
        las_path = self._las_path_edit.text()
        if not las_path:
            QMessageBox.warning(self, "No File", "Select a LiDAR file first.")
            return

        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not connected.")
            return

        from datetime import datetime

        params = {
            "las_path": las_path,
            "name": self._name_edit.text() or Path(las_path).stem,
            "crs": self._crs_combo.currentText() or None,
            "acquisition_date": datetime(
                self._date_edit.date().year(),
                self._date_edit.date().month(),
                self._date_edit.date().day(),
            ),
        }

        self._import_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Importing...")

        def on_progress(pct, msg):
            self._progress.setValue(int(pct))
            self._status_label.setText(msg)

        params["_progress_callback"] = on_progress

        self.controller.run_task(
            "frag_import", params,
            callback=self._on_import_complete,
            progress_callback=on_progress,
        )

    def _on_import_complete(self, result):
        self._progress.setVisible(False)
        self._import_btn.setEnabled(True)

        if isinstance(result, dict) and "error" not in result:
            count = result.get("point_count", 0)
            self._status_label.setText(f"Imported: {count:,} points")
            self.import_completed.emit(result)
            QMessageBox.information(
                self, "Import Complete",
                f"Dataset '{result.get('name', '')}' imported with {count:,} points."
            )
        else:
            error = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            self._status_label.setText(f"Error: {error}")
            QMessageBox.critical(self, "Import Failed", str(error))
