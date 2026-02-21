"""
SGSIM Simulation & Uncertainty Analysis Panel
==============================================
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QLabel,
    QMessageBox, QTabWidget, QWidget, QTextEdit,
    QFileDialog, QCheckBox, QLineEdit, QSplitter, QScrollArea, QFrame,
    QProgressBar, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication

from .panel_manager import PanelCategory, DockArea
from ..utils.coordinate_utils import ensure_xyz_columns
from ..utils.variable_utils import get_grade_columns, populate_variable_combo
from .base_analysis_panel import BaseAnalysisPanel, log_registry_data_status
from .modern_styles import get_theme_colors, get_analysis_panel_stylesheet

logger = logging.getLogger(__name__)

def get_sgsim_panel_stylesheet() -> str:
    colors = get_theme_colors()
    return f"""
        QLabel {{ color: #FFFFFF; font-size: 10pt; }}
        QLabel#NewDataBanner {{
            background-color: #1a3a5a; color: #4fc3f7; padding: 10px;
            border: 2px solid #2196F3; border-radius: 6px; font-weight: bold;
        }}
        QGroupBox {{
            font-weight: bold; border: 1px solid #444; border-radius: 6px;
            margin-top: 15px; padding-top: 15px; background-color: #222222;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; subcontrol-position: top left;
            left: 10px; color: #3498db; padding: 0 5px;
        }}
        QDoubleSpinBox, QSpinBox, QComboBox {{
            background-color: #111; border: 1px solid #555; color: white; padding: 4px;
        }}
        QPushButton#RefreshBtn {{
            background-color: #2c3e50; border: 1px solid #3498db; color: #3498db;
            font-weight: bold; font-size: 14pt;
        }}
    """

class SGSIMPanel(BaseAnalysisPanel):
    PANEL_ID = "SGSIMPanel"
    PANEL_NAME = "SGSIM Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    task_name = "sgsim"
    request_visualization = pyqtSignal(object, str)
    progress_updated = pyqtSignal(int, str)

    def __init__(self, parent=None):
        self.drillhole_data = None
        self.variable = None
        self.sgsim_results = None
        self.results_ready = False
        self.transformation_metadata = {}
        self.variogram_results = None
        self.block_grid_spec = None
        self.main_window = None

        super().__init__(parent=parent, panel_id="sgsim")
        self.setWindowTitle("SGSIM Simulation")
        self.resize(1200, 800)
        self._build_ui()
        self._init_registry()
        self.progress_updated.connect(self._update_progress)

    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if not self.registry: return

            self.registry.drillholeDataLoaded.connect(self._on_data_loaded)
            self.registry.variogramResultsLoaded.connect(self._on_vario_loaded)

            if hasattr(self.registry, 'transformationMetadataLoaded'):
                self.registry.transformationMetadataLoaded.connect(self._on_transformation_loaded)

            # Initial load
            d = self.registry.get_drillhole_data()
            if d is not None: self._on_data_loaded(d)
        except Exception as e:
            logger.warning(f"Registry connection failed: {e}")

    def _build_ui(self):
        # Clear any existing layout from base class (BaseAnalysisPanel creates a scroll area)
        old_layout = self.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.hide()
                        widget.setParent(None)
                        widget.deleteLater()
                    del item
            QWidget().setLayout(old_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT SIDE
        left = QWidget()
        left.setStyleSheet(get_sgsim_panel_stylesheet())
        l_lay = QVBoxLayout(left)

        # Data Status Card
        status_card = QFrame()
        status_card.setStyleSheet("background-color: #1a1a1a; border-radius: 8px; border: 1px solid #333;")
        sc_lay = QVBoxLayout(status_card)

        self.refresh_btn = QPushButton("🔄 Refresh Data")
        self.refresh_btn.setObjectName("RefreshBtn")
        self.refresh_btn.clicked.connect(self._manual_refresh)

        self.data_source_group = QButtonGroup()
        self.data_source_composited = QRadioButton("Composited Data")
        self.data_source_raw = QRadioButton("Raw Assay Data")
        self.data_source_group.addButton(self.data_source_composited)
        self.data_source_group.addButton(self.data_source_raw)
        self.data_source_composited.setChecked(True)

        self.data_source_status_label = QLabel("Initializing...")

        sc_lay.addWidget(self.refresh_btn)
        sc_lay.addWidget(self.data_source_composited)
        sc_lay.addWidget(self.data_source_raw)
        sc_lay.addWidget(self.data_source_status_label)
        l_lay.addWidget(status_card)

        # Scroll Config
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        cont = QWidget()
        s_lay = QVBoxLayout(cont)

        self._create_sim_settings(s_lay)
        self._create_grid_group(s_lay)
        self._create_variogram_group(s_lay)
        self._create_search_group(s_lay)
        self._create_cutoff_group(s_lay)

        scroll.setWidget(cont)
        l_lay.addWidget(scroll)

        # RIGHT SIDE
        right = QWidget()
        r_lay = QVBoxLayout(right)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        r_lay.addWidget(self.progress_bar)
        r_lay.addWidget(self.progress_label)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        r_lay.addWidget(QLabel("<b>Event Log</b>"))
        r_lay.addWidget(self.results_text)

        self.tabs = QTabWidget()
        self._create_viz_tab()
        self._create_uncert_tab()
        r_lay.addWidget(self.tabs)

        act_lay = QHBoxLayout()
        self.run_btn = QPushButton("RUN SGSIM")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        self.run_btn.clicked.connect(self.run_analysis)

        self.export_btn = QPushButton("Export")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_results_menu)

        act_lay.addWidget(self.run_btn)
        act_lay.addWidget(self.export_btn)
        r_lay.addLayout(act_lay)

        splitter.addWidget(left)
        splitter.addWidget(right)
        layout.addWidget(splitter)

    def _create_sim_settings(self, layout):
        g = QGroupBox("1. Simulation Settings")
        l = QFormLayout(g)
        self.variable_combo = QComboBox()
        self.nreal_spin = QSpinBox()
        self.nreal_spin.setRange(1, 1000)
        self.nreal_spin.setValue(50)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999999)
        self.seed_spin.setValue(12345)

        l.addRow("Variable:", self.variable_combo)
        l.addRow("Realizations:", self.nreal_spin)
        l.addRow("Seed:", self.seed_spin)
        layout.addWidget(g)

    def _create_grid_group(self, layout):
        g = QGroupBox("2. Grid Specification")
        l = QVBoxLayout(g)
        h1 = QHBoxLayout()
        self.xmin_spin = QDoubleSpinBox()
        self.ymin_spin = QDoubleSpinBox()
        self.zmin_spin = QDoubleSpinBox()
        h1.addWidget(QLabel("X0:")); h1.addWidget(self.xmin_spin)
        h1.addWidget(QLabel("Y0:")); h1.addWidget(self.ymin_spin)
        h1.addWidget(QLabel("Z0:")); h1.addWidget(self.zmin_spin)
        l.addLayout(h1)

        h2 = QHBoxLayout()
        self.nx = QSpinBox(); self.ny = QSpinBox(); self.nz = QSpinBox()
        self.nx.setRange(1, 500); self.ny.setRange(1, 500); self.nz.setRange(1, 500)
        h2.addWidget(QLabel("NX:")); h2.addWidget(self.nx)
        h2.addWidget(QLabel("NY:")); h2.addWidget(self.ny)
        h2.addWidget(QLabel("NZ:")); h2.addWidget(self.nz)
        l.addLayout(h2)

        h3 = QHBoxLayout()
        self.dx = QDoubleSpinBox(); self.dy = QDoubleSpinBox(); self.dz = QDoubleSpinBox()
        self.dx.setValue(10); self.dy.setValue(10); self.dz.setValue(5)
        h3.addWidget(QLabel("DX:")); h3.addWidget(self.dx)
        h3.addWidget(QLabel("DY:")); h3.addWidget(self.dy)
        h3.addWidget(QLabel("DZ:")); h3.addWidget(self.dz)
        l.addLayout(h3)

        btn = QPushButton("Auto-Detect Grid")
        btn.clicked.connect(self._auto_detect_grid)
        l.addWidget(btn)
        layout.addWidget(g)

    def _create_variogram_group(self, layout):
        g = QGroupBox("3. Variogram")
        l = QFormLayout(g)
        self.vario_type = QComboBox()
        self.vario_type.addItems(["Spherical", "Exponential", "Gaussian"])
        self.rmaj = QDoubleSpinBox(); self.rmaj.setRange(0, 5000); self.rmaj.setValue(100)
        self.rmin = QDoubleSpinBox(); self.rmin.setRange(0, 5000); self.rmin.setValue(80)
        self.rver = QDoubleSpinBox(); self.rver.setRange(0, 5000); self.rver.setValue(20)
        self.nug = QDoubleSpinBox(); self.nug.setRange(0, 1); self.nug.setSingleStep(0.01)
        self.sill = QDoubleSpinBox(); self.sill.setRange(0, 10); self.sill.setValue(1.0)
        self.azim = QDoubleSpinBox(); self.dip = QDoubleSpinBox()

        l.addRow("Type:", self.vario_type)
        l.addRow("Range Major:", self.rmaj)
        l.addRow("Range Minor:", self.rmin)
        l.addRow("Range Vert:", self.rver)
        l.addRow("Nugget:", self.nug)
        l.addRow("Sill:", self.sill)
        l.addRow("Azimuth:", self.azim)
        l.addRow("Dip:", self.dip)
        layout.addWidget(g)

    def _create_search_group(self, layout):
        g = QGroupBox("4. Search")
        l = QHBoxLayout(g)
        self.min_n = QSpinBox(); self.min_n.setValue(8)
        self.max_n = QSpinBox(); self.max_n.setValue(16)
        self.rad = QDoubleSpinBox(); self.rad.setValue(200); self.rad.setRange(0, 5000)
        l.addWidget(QLabel("Min:")); l.addWidget(self.min_n)
        l.addWidget(QLabel("Max:")); l.addWidget(self.max_n)
        l.addWidget(QLabel("Radius:")); l.addWidget(self.rad)
        layout.addWidget(g)

    def _create_cutoff_group(self, layout):
        g = QGroupBox("5. Cutoffs")
        l = QHBoxLayout(g)
        self.cutoff_edit = QLineEdit("0.5, 1.0, 2.0")
        l.addWidget(self.cutoff_edit)
        layout.addWidget(g)

    def _create_viz_tab(self):
        tab = QWidget()
        l = QVBoxLayout(tab)
        self.back_transform_btn = QPushButton("Back-Transform")
        self.back_transform_btn.setEnabled(False)
        l.addWidget(self.back_transform_btn)

        self.viz_mean = QPushButton("Mean"); self.viz_mean.clicked.connect(lambda: self._visualize_summary("mean"))
        self.viz_std = QPushButton("Std Dev"); self.viz_std.clicked.connect(lambda: self._visualize_summary("std"))
        self.viz_p10 = QPushButton("P10"); self.viz_p10.clicked.connect(lambda: self._visualize_summary("p10"))
        self.viz_p50 = QPushButton("P50"); self.viz_p50.clicked.connect(lambda: self._visualize_summary("p50"))
        self.viz_p90 = QPushButton("P90"); self.viz_p90.clicked.connect(lambda: self._visualize_summary("p90"))

        for b in [self.viz_mean, self.viz_std, self.viz_p10, self.viz_p50, self.viz_p90]:
            b.setEnabled(False)
            l.addWidget(b)
        self.tabs.addTab(tab, "Visualization")

    def _create_uncert_tab(self):
        tab = QWidget()
        l = QVBoxLayout(tab)
        self.uncert_text = QTextEdit(); self.uncert_text.setReadOnly(True)
        self.prob_layout = QHBoxLayout()
        l.addWidget(self.uncert_text)
        l.addLayout(self.prob_layout)
        self.tabs.addTab(tab, "Uncertainty")

    # LOGIC METHODS
    def _on_data_loaded(self, data):
        if not data: return
        df = data if isinstance(data, pd.DataFrame) else data.get('composites')
        if df is not None:
            self.drillhole_data = ensure_xyz_columns(df)
            populate_variable_combo(self.variable_combo, df)
            self.data_source_status_label.setText(f"Data Loaded: {len(df)} samples")

    def _on_vario_loaded(self, results):
        """Handle variogram results loaded from registry"""
        self.variogram_results = results
        # Could auto-populate variogram parameters here if needed

    def _on_transformation_loaded(self, metadata):
        """Handle transformation metadata loaded from registry"""
        self.transformation_metadata = metadata

    def _manual_refresh(self):
        if self.registry:
            self._on_data_loaded(self.registry.get_drillhole_data())

    def _auto_detect_grid(self):
        if self.drillhole_data is None: return
        df = self.drillhole_data
        self.xmin_spin.setValue(df.X.min())
        self.ymin_spin.setValue(df.Y.min())
        self.zmin_spin.setValue(df.Z.min())
        self.nx.setValue(int((df.X.max() - df.X.min()) / 10) + 2)
        self.ny.setValue(int((df.Y.max() - df.Y.min()) / 10) + 2)
        self.nz.setValue(int((df.Z.max() - df.Z.min()) / 5) + 2)

    def run_analysis(self):
        if not self.controller: return
        params = {
            "data_df": self.drillhole_data,
            "variable": self.variable_combo.currentText(),
            "nreal": self.nreal_spin.value(),
            "seed": self.seed_spin.value(),
            "nx": self.nx.value(), "ny": self.ny.value(), "nz": self.nz.value(),
            "xmin": self.xmin_spin.value(), "ymin": self.ymin_spin.value(), "zmin": self.zmin_spin.value(),
            "xinc": self.dx.value(), "yinc": self.dy.value(), "zinc": self.dz.value(),
            "variogram_type": self.vario_type.currentText().lower(),
            "range_major": self.rmaj.value(), "range_minor": self.rmin.value(), "range_vert": self.rver.value(),
            "nugget": self.nug.value(), "sill": self.sill.value(),
            "azimuth": self.azim.value(), "dip": self.dip.value(),
            "min_neighbors": self.min_n.value(), "max_neighbors": self.max_n.value(),
            "max_search_radius": self.rad.value(),
            "cutoffs": [float(x.strip()) for x in self.cutoff_edit.text().split(",") if x.strip()]
        }
        self.run_btn.setEnabled(False)
        self.controller.run_sgsim(params, callback=self.handle_results,
                                progress_callback=lambda p, m: self.progress_updated.emit(p, m))

    def handle_results(self, payload):
        self.run_btn.setEnabled(True)
        self.sgsim_results = payload.get("results")
        if not self.sgsim_results:
            self._log_event("Simulation failed or returned no results", "error")
            return

        self.results_ready = True
        self.export_btn.setEnabled(True)
        self.back_transform_btn.setEnabled(True)
        for b in [self.viz_mean, self.viz_std, self.viz_p10, self.viz_p50, self.viz_p90]:
            b.setEnabled(True)

        self._log_event("SGSIM Simulation Complete", "success")
        self._display_uncertainty()

    def _auto_back_transform(self):
        """Auto-trigger back-transformation if transformation metadata is available"""
        if self.transformation_metadata:
            self._log_event("Auto-triggering back-transformation...", "info")
            # Implementation of back-transform logic would go here
            # This is a placeholder for the actual transformation code

    def _display_uncertainty(self):
        if not self.sgsim_results: return
        self.uncert_text.setText("Simulation Summary Calculated.")

    def _visualize_summary(self, stat):
        import pyvista as pv
        if not self.sgsim_results: return
        summary = self.sgsim_results.get('summary', {})
        data = summary.get(stat)
        if data is None: return

        grid = pv.ImageData(
            dimensions=(self.nx.value(), self.ny.value(), self.nz.value()),
            spacing=(self.dx.value(), self.dy.value(), self.dz.value()),
            origin=(self.xmin_spin.value(), self.ymin_spin.value(), self.zmin_spin.value())
        )
        grid.cell_data[f"SGSIM_{stat.upper()}"] = data.flatten(order="C")
        self.request_visualization.emit(grid, f"SGSIM_{stat.upper()}")

    def _update_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def _log_event(self, msg, level="info"):
        color = "#81c784" if level == "success" else "#e57373" if level == "error" else "#FFFFFF"
        self.results_text.append(f"<span style='color:{color}'>{msg}</span>")

    def _export_results_menu(self):
        """Placeholder for export functionality"""
        QMessageBox.information(self, "Export", "Export functionality to be implemented")

    def get_panel_settings(self) -> Dict[str, Any]:
        return {
            "variable": self.variable_combo.currentText(),
            "nreal": self.nreal_spin.value(),
            "seed": self.seed_spin.value(),
            "nx": self.nx.value(), "ny": self.ny.value(), "nz": self.nz.value(),
            "xmin": self.xmin_spin.value(), "ymin": self.ymin_spin.value(), "zmin": self.zmin_spin.value(),
            "dx": self.dx.value(), "dy": self.dy.value(), "dz": self.dz.value(),
            "vario_type": self.vario_type.currentText(),
            "rmaj": self.rmaj.value(), "rmin": self.rmin.value(), "rver": self.rver.value(),
            "nugget": self.nug.value(), "sill": self.sill.value()
        }

    def apply_panel_settings(self, settings: Dict[str, Any]):
        if not settings: return
        self.variable_combo.setCurrentText(settings.get("variable", ""))
        self.nreal_spin.setValue(settings.get("nreal", 50))
        self.seed_spin.setValue(settings.get("seed", 12345))
        self.nx.setValue(settings.get("nx", 50))
        self.ny.setValue(settings.get("ny", 50))
        self.nz.setValue(settings.get("nz", 20))
        self.xmin_spin.setValue(settings.get("xmin", 0))
        self.ymin_spin.setValue(settings.get("ymin", 0))
        self.zmin_spin.setValue(settings.get("zmin", 0))
        self.dx.setValue(settings.get("dx", 10))
        self.dy.setValue(settings.get("dy", 10))
        self.dz.setValue(settings.get("dz", 5))
