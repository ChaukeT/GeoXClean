"""
Fragmentation Preprocessing Panel
===================================

Configure and run the LiDAR cleaning, normal estimation, and RGB fusion pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QCheckBox, QDoubleSpinBox, QSpinBox,
    QProgressBar, QMessageBox,
)

from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea

logger = logging.getLogger(__name__)


class FragPreprocessingPanel(BaseDockPanel):
    """Panel for configuring and running fragmentation preprocessing."""

    PANEL_ID = "FragPreprocessingPanel"
    PANEL_NAME = "Fragmentation Preprocessing"
    PANEL_CATEGORY = PanelCategory.ANALYSIS
    PANEL_ICON = "scan"
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT
    PANEL_DEFAULT_VISIBLE = False
    PANEL_TOOLTIP = "Clean, filter, and fuse LiDAR point cloud data"

    preprocessing_completed = pyqtSignal(dict)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, panel_id=kwargs.get("panel_id"))

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- 1. LiDAR Cleaning ---
        clean_group = QGroupBox("1. LiDAR Cleaning")
        clean_form = QFormLayout()

        self._sor_check = QCheckBox("Statistical Outlier Removal")
        self._sor_check.setChecked(True)
        clean_form.addRow(self._sor_check)

        self._sor_k = QSpinBox()
        self._sor_k.setRange(5, 100)
        self._sor_k.setValue(20)
        clean_form.addRow("  SOR Neighbours (k):", self._sor_k)

        self._sor_std = QDoubleSpinBox()
        self._sor_std.setRange(0.5, 5.0)
        self._sor_std.setSingleStep(0.1)
        self._sor_std.setValue(2.0)
        clean_form.addRow("  SOR Std Ratio:", self._sor_std)

        self._ror_check = QCheckBox("Radius Outlier Removal")
        self._ror_check.setChecked(False)
        clean_form.addRow(self._ror_check)

        self._ror_radius = QDoubleSpinBox()
        self._ror_radius.setRange(0.01, 10.0)
        self._ror_radius.setSingleStep(0.1)
        self._ror_radius.setValue(0.5)
        self._ror_radius.setSuffix(" m")
        clean_form.addRow("  ROR Radius:", self._ror_radius)

        self._ror_min = QSpinBox()
        self._ror_min.setRange(1, 50)
        self._ror_min.setValue(6)
        clean_form.addRow("  ROR Min Points:", self._ror_min)

        self._voxel_check = QCheckBox("Voxel Downsample")
        self._voxel_check.setChecked(False)
        clean_form.addRow(self._voxel_check)

        self._voxel_size = QDoubleSpinBox()
        self._voxel_size.setRange(0.001, 1.0)
        self._voxel_size.setSingleStep(0.01)
        self._voxel_size.setValue(0.05)
        self._voxel_size.setSuffix(" m")
        clean_form.addRow("  Voxel Size:", self._voxel_size)

        clean_group.setLayout(clean_form)
        layout.addWidget(clean_group)

        # --- 2. Ground Filtering ---
        ground_group = QGroupBox("2. Ground Filtering")
        ground_form = QFormLayout()

        self._ground_check = QCheckBox("Enable CSF Ground Filter")
        self._ground_check.setChecked(False)
        ground_form.addRow(self._ground_check)

        self._csf_res = QDoubleSpinBox()
        self._csf_res.setRange(0.1, 5.0)
        self._csf_res.setSingleStep(0.1)
        self._csf_res.setValue(0.5)
        self._csf_res.setSuffix(" m")
        ground_form.addRow("  Cloth Resolution:", self._csf_res)

        self._csf_thresh = QDoubleSpinBox()
        self._csf_thresh.setRange(0.1, 5.0)
        self._csf_thresh.setSingleStep(0.1)
        self._csf_thresh.setValue(0.5)
        self._csf_thresh.setSuffix(" m")
        ground_form.addRow("  Threshold:", self._csf_thresh)

        ground_group.setLayout(ground_form)
        layout.addWidget(ground_group)

        # --- 3. Normal Estimation ---
        normal_group = QGroupBox("3. Normal & Curvature Estimation")
        normal_form = QFormLayout()

        self._normal_k = QSpinBox()
        self._normal_k.setRange(5, 100)
        self._normal_k.setValue(30)
        normal_form.addRow("Neighbours (k):", self._normal_k)

        normal_group.setLayout(normal_form)
        layout.addWidget(normal_group)

        # --- Run Button ---
        self._run_btn = QPushButton("Run Preprocessing")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

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
    # Gather Parameters
    # ------------------------------------------------------------------

    def _gather_config(self) -> Dict[str, Any]:
        """Collect UI values into a PreprocessingConfig."""
        from ..scans.preprocessing import (
            PreprocessingConfig, SORConfig, RORConfig, CSFConfig,
            VoxelConfig, NormalConfig,
        )

        config = PreprocessingConfig(
            enable_sor=self._sor_check.isChecked(),
            sor=SORConfig(k_neighbors=self._sor_k.value(), std_ratio=self._sor_std.value()),
            enable_ror=self._ror_check.isChecked(),
            ror=RORConfig(radius=self._ror_radius.value(), min_points=self._ror_min.value()),
            enable_ground_filter=self._ground_check.isChecked(),
            csf=CSFConfig(cloth_resolution=self._csf_res.value(), threshold=self._csf_thresh.value()),
            enable_voxel_downsample=self._voxel_check.isChecked(),
            voxel=VoxelConfig(voxel_size=self._voxel_size.value()),
            normals=NormalConfig(k_neighbors=self._normal_k.value()),
        )
        return config

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _on_run(self):
        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not connected.")
            return

        # Check that a dataset exists
        registry = self.get_registry()
        if not registry or not registry.has_fragment_dataset():
            QMessageBox.warning(
                self, "No Dataset",
                "Import a LiDAR file first using the Fragmentation Import panel."
            )
            return

        config = self._gather_config()
        params = {"config": config}

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Preprocessing...")

        def on_progress(pct, msg):
            self._progress.setValue(int(pct))
            self._status_label.setText(msg)

        params["_progress_callback"] = on_progress

        self.controller.run_task(
            "frag_preprocess", params,
            callback=self._on_complete,
            progress_callback=on_progress,
        )

    def _on_complete(self, result):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)

        if isinstance(result, dict) and "error" not in result:
            stats = result.get("stats", {})
            pts = result.get("point_count", 0)
            self._status_label.setText(
                f"Done: {pts:,} points after preprocessing"
            )
            self.preprocessing_completed.emit(result)
        else:
            error = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            self._status_label.setText(f"Error: {error}")
            QMessageBox.critical(self, "Preprocessing Failed", str(error))
