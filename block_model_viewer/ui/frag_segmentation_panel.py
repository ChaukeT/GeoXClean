"""
Fragmentation Segmentation Panel
==================================

Select and configure segmentation algorithm(s), run segmentation,
and inspect results with a summary view.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QRadioButton, QButtonGroup, QStackedWidget,
    QDoubleSpinBox, QSpinBox, QCheckBox, QProgressBar, QMessageBox,
)

from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea

logger = logging.getLogger(__name__)


class FragSegmentationPanel(BaseDockPanel):
    """Panel for configuring and running fragment segmentation."""

    PANEL_ID = "FragSegmentationPanel"
    PANEL_NAME = "Fragmentation Segmentation"
    PANEL_CATEGORY = PanelCategory.ANALYSIS
    PANEL_ICON = "scan"
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT
    PANEL_DEFAULT_VISIBLE = False
    PANEL_TOOLTIP = "Segment point cloud into individual rock fragments"

    segmentation_completed = pyqtSignal(dict)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, panel_id=kwargs.get("panel_id"))

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Method Selection ---
        method_group = QGroupBox("Segmentation Method")
        method_layout = QVBoxLayout()

        self._method_group = QButtonGroup(self)
        self._radio_geometric = QRadioButton("Geometric (Region Growing + RANSAC)")
        self._radio_watershed = QRadioButton("Watershed (Image-based)")
        self._radio_geometric.setChecked(True)
        self._method_group.addButton(self._radio_geometric, 0)
        self._method_group.addButton(self._radio_watershed, 1)
        method_layout.addWidget(self._radio_geometric)
        method_layout.addWidget(self._radio_watershed)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # --- Parameter Stacks ---
        self._param_stack = QStackedWidget()

        # Page 0: Geometric params
        geom_page = self._build_geometric_params()
        self._param_stack.addWidget(geom_page)

        # Page 1: Watershed params
        ws_page = self._build_watershed_params()
        self._param_stack.addWidget(ws_page)

        self._method_group.idToggled.connect(
            lambda id, checked: self._param_stack.setCurrentIndex(id) if checked else None
        )

        layout.addWidget(self._param_stack)

        # --- Run + Size Extraction ---
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Segmentation")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._on_run_segmentation)
        btn_row.addWidget(self._run_btn)

        self._extract_btn = QPushButton("Extract Size Metrics")
        self._extract_btn.setMinimumHeight(36)
        self._extract_btn.setEnabled(False)
        self._extract_btn.clicked.connect(self._on_extract_sizes)
        btn_row.addWidget(self._extract_btn)

        layout.addLayout(btn_row)

        # --- Progress ---
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # --- Summary ---
        summary_group = QGroupBox("Results Summary")
        summary_form = QFormLayout()
        self._frag_count_label = QLabel("--")
        self._noise_label = QLabel("--")
        self._d50_label = QLabel("--")
        self._d80_label = QLabel("--")
        summary_form.addRow("Fragments:", self._frag_count_label)
        summary_form.addRow("Noise Points:", self._noise_label)
        summary_form.addRow("D50:", self._d50_label)
        summary_form.addRow("D80:", self._d80_label)
        summary_group.setLayout(summary_form)
        layout.addWidget(summary_group)

        # --- Status ---
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)
        layout.addStretch()

        if self.main_layout:
            self.main_layout.addLayout(layout)
        else:
            self.setLayout(layout)

    # ------------------------------------------------------------------
    # Parameter Pages
    # ------------------------------------------------------------------

    def _build_geometric_params(self):
        from PyQt6.QtWidgets import QWidget
        page = QWidget()
        form = QFormLayout(page)

        self._geom_angle = QDoubleSpinBox()
        self._geom_angle.setRange(5, 90)
        self._geom_angle.setValue(30)
        self._geom_angle.setSuffix(" deg")
        form.addRow("Normal Angle Threshold:", self._geom_angle)

        self._geom_curv = QDoubleSpinBox()
        self._geom_curv.setRange(0.001, 1.0)
        self._geom_curv.setDecimals(4)
        self._geom_curv.setSingleStep(0.001)
        self._geom_curv.setValue(0.01)
        form.addRow("Curvature Threshold:", self._geom_curv)

        self._geom_min_size = QSpinBox()
        self._geom_min_size.setRange(10, 10000)
        self._geom_min_size.setValue(100)
        form.addRow("Min Region Size:", self._geom_min_size)

        self._geom_max_size = QSpinBox()
        self._geom_max_size.setRange(1000, 10_000_000)
        self._geom_max_size.setValue(1_000_000)
        form.addRow("Max Region Size:", self._geom_max_size)

        self._geom_k = QSpinBox()
        self._geom_k.setRange(5, 100)
        self._geom_k.setValue(15)
        form.addRow("Neighbours (k):", self._geom_k)

        self._geom_ransac = QCheckBox("Enable RANSAC planes")
        form.addRow(self._geom_ransac)

        self._geom_dbscan = QCheckBox("DBSCAN fallback for residuals")
        self._geom_dbscan.setChecked(True)
        form.addRow(self._geom_dbscan)

        self._geom_merge = QDoubleSpinBox()
        self._geom_merge.setRange(0.0, 1.0)
        self._geom_merge.setSingleStep(0.05)
        self._geom_merge.setValue(0.8)
        form.addRow("Merge Threshold:", self._geom_merge)

        return page

    def _build_watershed_params(self):
        from PyQt6.QtWidgets import QWidget
        page = QWidget()
        form = QFormLayout(page)

        self._ws_canny_low = QSpinBox()
        self._ws_canny_low.setRange(10, 200)
        self._ws_canny_low.setValue(50)
        form.addRow("Canny Low Threshold:", self._ws_canny_low)

        self._ws_canny_high = QSpinBox()
        self._ws_canny_high.setRange(50, 500)
        self._ws_canny_high.setValue(150)
        form.addRow("Canny High Threshold:", self._ws_canny_high)

        self._ws_dist_ratio = QDoubleSpinBox()
        self._ws_dist_ratio.setRange(0.1, 0.9)
        self._ws_dist_ratio.setSingleStep(0.05)
        self._ws_dist_ratio.setValue(0.5)
        form.addRow("Distance Transform Ratio:", self._ws_dist_ratio)

        self._ws_min_size = QSpinBox()
        self._ws_min_size.setRange(10, 10000)
        self._ws_min_size.setValue(50)
        form.addRow("Min Fragment Size:", self._ws_min_size)

        return page

    # ------------------------------------------------------------------
    # Gather Parameters
    # ------------------------------------------------------------------

    def _gather_params(self) -> Dict[str, Any]:
        if self._radio_geometric.isChecked():
            return {
                "method": "geometric",
                "normal_threshold_deg": self._geom_angle.value(),
                "curvature_threshold": self._geom_curv.value(),
                "min_region_size": self._geom_min_size.value(),
                "max_region_size": self._geom_max_size.value(),
                "k_neighbors": self._geom_k.value(),
                "enable_ransac": self._geom_ransac.isChecked(),
                "enable_dbscan_fallback": self._geom_dbscan.isChecked(),
                "merge_threshold": self._geom_merge.value(),
            }
        else:
            return {
                "method": "watershed",
                "canny_low": self._ws_canny_low.value(),
                "canny_high": self._ws_canny_high.value(),
                "dist_thresh_ratio": self._ws_dist_ratio.value(),
                "min_fragment_size": self._ws_min_size.value(),
            }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _on_run_segmentation(self):
        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not connected.")
            return

        registry = self.get_registry()
        if not registry or not registry.has_fragment_dataset():
            QMessageBox.warning(self, "No Dataset", "Import and preprocess data first.")
            return

        params = self._gather_params()
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Segmenting...")

        def on_progress(pct, msg):
            self._progress.setValue(int(pct))
            self._status_label.setText(msg)

        params["_progress_callback"] = on_progress

        self.controller.run_task(
            "frag_segment", params,
            callback=self._on_segment_complete,
            progress_callback=on_progress,
        )

    def _on_segment_complete(self, result):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)

        if isinstance(result, dict) and "error" not in result:
            self._frag_count_label.setText(str(result.get("fragment_count", 0)))
            self._noise_label.setText(str(result.get("noise_points", 0)))
            self._extract_btn.setEnabled(True)
            self._status_label.setText(
                f"Segmentation complete: {result.get('fragment_count', 0)} fragments"
            )
            self.segmentation_completed.emit(result)
        else:
            error = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            self._status_label.setText(f"Error: {error}")
            QMessageBox.critical(self, "Segmentation Failed", str(error))

    def _on_extract_sizes(self):
        if not self.controller:
            return

        self._extract_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Extracting size metrics...")

        params = {}

        def on_progress(pct, msg):
            self._progress.setValue(int(pct))
            self._status_label.setText(msg)

        params["_progress_callback"] = on_progress

        self.controller.run_task(
            "frag_size", params,
            callback=self._on_size_complete,
            progress_callback=on_progress,
        )

    def _on_size_complete(self, result):
        self._progress.setVisible(False)
        self._extract_btn.setEnabled(True)

        if isinstance(result, dict) and "error" not in result:
            self._d50_label.setText(f"{result.get('d50', 0):.3f} m")
            self._d80_label.setText(f"{result.get('d80', 0):.3f} m")
            self._status_label.setText(
                f"Size metrics: D50={result.get('d50', 0):.3f}m, D80={result.get('d80', 0):.3f}m"
            )
        else:
            error = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            self._status_label.setText(f"Error: {error}")
            QMessageBox.critical(self, "Size Extraction Failed", str(error))
