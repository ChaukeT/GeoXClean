"""
Fragmentation Results Panel
=============================

Displays fragment size metrics table, interactive FSD curves with
Rosin-Rammler and Swebrec fits, and export controls.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFileDialog, QMessageBox, QTabWidget, QWidget,
)

from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea

logger = logging.getLogger(__name__)


class FragResultsPanel(BaseDockPanel):
    """Panel displaying fragment metrics, FSD curves, and export controls."""

    PANEL_ID = "FragResultsPanel"
    PANEL_NAME = "Fragmentation Results"
    PANEL_CATEGORY = PanelCategory.ANALYSIS
    PANEL_ICON = "chart"
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT
    PANEL_DEFAULT_VISIBLE = False
    PANEL_TOOLTIP = "View fragment size metrics, FSD curves, and export results"

    fragment_selected = pyqtSignal(int)  # Emits fragment_id for 3D highlight

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, panel_id=kwargs.get("panel_id"))

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Tabs: Table | FSD Plot | Summary
        self._tabs = QTabWidget()

        # --- Tab 1: Fragment Table ---
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        self._table = QTableWidget()
        self._table.setColumnCount(9)
        self._table.setHorizontalHeaderLabels([
            "ID", "Points", "D_equiv (m)", "Feret Max (m)", "Feret Min (m)",
            "Aspect Ratio", "Sphericity", "Method", "Confidence",
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setSortingEnabled(True)
        self._table.cellDoubleClicked.connect(self._on_row_double_click)
        table_layout.addWidget(self._table)
        self._tabs.addTab(table_tab, "Fragment Table")

        # --- Tab 2: FSD Plot ---
        fsd_tab = QWidget()
        fsd_layout = QVBoxLayout(fsd_tab)
        self._fsd_widget = None  # Will be a matplotlib canvas if available
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(5, 4), dpi=100)
            self._ax = self._fig.add_subplot(111)
            self._fsd_widget = FigureCanvasQTAgg(self._fig)
            fsd_layout.addWidget(self._fsd_widget)
        except ImportError:
            fsd_layout.addWidget(QLabel("matplotlib not available for FSD plot"))
        self._tabs.addTab(fsd_tab, "FSD Curve")

        # --- Tab 3: Summary ---
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self._summary_text = QLabel("Load fragment data to see summary.")
        self._summary_text.setWordWrap(True)
        self._summary_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        summary_layout.addWidget(self._summary_text)
        self._tabs.addTab(summary_tab, "Summary")

        layout.addWidget(self._tabs)

        # --- 3D Size Visualisation ---
        vis_group = QGroupBox("3D Size Visualisation")
        vis_layout = QHBoxLayout()

        from PyQt6.QtWidgets import QComboBox
        self._size_metric_combo = QComboBox()
        self._size_metric_combo.addItems([
            "equiv_diameter", "feret_max", "feret_min",
            "volume", "sphericity", "aspect_ratio",
        ])
        vis_layout.addWidget(QLabel("Metric:"))
        vis_layout.addWidget(self._size_metric_combo, 1)

        self._vis_btn = QPushButton("Show in 3D")
        self._vis_btn.clicked.connect(self._on_visualise_sizes)
        vis_layout.addWidget(self._vis_btn)

        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        # --- Export Buttons ---
        export_row = QHBoxLayout()
        self._export_csv_btn = QPushButton("Export CSV")
        self._export_csv_btn.clicked.connect(self._export_csv)
        export_row.addWidget(self._export_csv_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._load_from_registry)
        export_row.addWidget(self._refresh_btn)

        layout.addLayout(export_row)

        if self.main_layout:
            self.main_layout.addLayout(layout)
        else:
            self.setLayout(layout)

    def on_panel_shown(self):
        """Auto-refresh when panel becomes visible."""
        self._load_from_registry()

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def _load_from_registry(self):
        registry = self.get_registry()
        if not registry or not registry.has_fragment_dataset():
            return

        dataset = registry.get_fragment_dataset(copy_data=False)
        if dataset is None:
            return

        self._populate_table(dataset)
        self._plot_fsd(dataset)
        self._update_summary(dataset)

    def _populate_table(self, dataset):
        fragments = dataset.fragments
        self._table.setRowCount(len(fragments))

        for row, f in enumerate(fragments):
            self._table.setItem(row, 0, self._num_item(f.fragment_id))
            self._table.setItem(row, 1, self._num_item(f.point_count))
            self._table.setItem(row, 2, self._float_item(f.equiv_diameter, 4))
            self._table.setItem(row, 3, self._float_item(f.feret_max, 4))
            self._table.setItem(row, 4, self._float_item(f.feret_min, 4))
            self._table.setItem(row, 5, self._float_item(f.aspect_ratio, 2))
            self._table.setItem(row, 6, self._float_item(f.sphericity, 3))
            self._table.setItem(row, 7, QTableWidgetItem(f.source_method.value))
            self._table.setItem(row, 8, self._float_item(f.confidence, 2))

    def _plot_fsd(self, dataset):
        if self._fsd_widget is None or dataset.fsd_global is None:
            return

        fsd = dataset.fsd_global
        ax = self._ax
        ax.clear()

        # Cumulative passing curve
        ax.semilogx(fsd.diameters_sorted, fsd.cumulative_passing * 100,
                     'b-', linewidth=2, label="FSD")

        # D-markers
        for pname, dval, color in [
            ("D10", fsd.d10, "green"), ("D50", fsd.d50, "orange"),
            ("D80", fsd.d80, "red"), ("D90", fsd.d90, "darkred"),
        ]:
            if dval > 0:
                ax.axvline(dval, color=color, linestyle="--", alpha=0.7, label=f"{pname}={dval:.3f}m")

        # Rosin-Rammler fit
        if fsd.rr_n is not None and fsd.rr_xc is not None:
            x_fit = np.logspace(
                np.log10(fsd.diameters_sorted[0]),
                np.log10(fsd.diameters_sorted[-1]),
                200,
            )
            y_rr = (1 - np.exp(-((x_fit / fsd.rr_xc) ** fsd.rr_n))) * 100
            ax.semilogx(x_fit, y_rr, 'r--', alpha=0.6,
                        label=f"R-R (n={fsd.rr_n:.2f}, xc={fsd.rr_xc:.3f})")

        # Swebrec fit
        if fsd.swebrec_xmax is not None and fsd.swebrec_x50 is not None and fsd.swebrec_b is not None:
            x_fit = np.logspace(
                np.log10(fsd.diameters_sorted[0]),
                np.log10(min(fsd.diameters_sorted[-1], fsd.swebrec_xmax * 0.95)),
                200,
            )
            ratio = np.log(np.clip(fsd.swebrec_xmax / x_fit, 1.001, None)) / \
                    np.log(np.clip(fsd.swebrec_xmax / fsd.swebrec_x50, 1.001, None))
            y_sw = 100.0 / (1.0 + ratio**fsd.swebrec_b)
            ax.semilogx(x_fit, y_sw, 'g--', alpha=0.6,
                        label=f"Swebrec (b={fsd.swebrec_b:.2f})")

        ax.set_xlabel("Fragment Size (m)")
        ax.set_ylabel("Cumulative Passing (%)")
        ax.set_title("Fragment Size Distribution")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, alpha=0.3)

        self._fig.tight_layout()
        self._fsd_widget.draw()

    def _update_summary(self, dataset):
        parts = []
        parts.append(f"Dataset: {dataset.name}")
        parts.append(f"Points: {dataset.point_count:,}")
        parts.append(f"Fragments: {dataset.fragment_count}")
        if dataset.fsd_global:
            fsd = dataset.fsd_global
            parts.append(f"D10: {fsd.d10:.4f} m")
            parts.append(f"D50: {fsd.d50:.4f} m")
            parts.append(f"D80: {fsd.d80:.4f} m")
            parts.append(f"D90: {fsd.d90:.4f} m")
            if fsd.rr_n is not None:
                parts.append(f"Rosin-Rammler: n={fsd.rr_n:.3f}, xc={fsd.rr_xc:.4f} m")
            if fsd.swebrec_b is not None:
                parts.append(f"Swebrec: xmax={fsd.swebrec_xmax:.4f}, x50={fsd.swebrec_x50:.4f}, b={fsd.swebrec_b:.3f}")
        parts.append(f"Validation: {dataset.validation_status.value}")
        parts.append(f"Provenance steps: {len(dataset.provenance.entries)}")
        self._summary_text.setText("\n".join(parts))

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_row_double_click(self, row, col):
        """Emit signal to highlight fragment in 3D viewer."""
        id_item = self._table.item(row, 0)
        if id_item:
            frag_id = int(id_item.data(Qt.ItemDataRole.UserRole))
            self.fragment_selected.emit(frag_id)

    def _on_visualise_sizes(self):
        """Render the point cloud coloured by the selected size metric."""
        registry = self.get_registry()
        if not registry or not registry.has_fragment_dataset():
            QMessageBox.warning(self, "No Data", "No fragment data available.")
            return

        dataset = registry.get_fragment_dataset(copy_data=False)
        if not dataset or not dataset.fragments or dataset.fragment_labels is None:
            QMessageBox.warning(self, "No Fragments", "Run segmentation and size extraction first.")
            return

        metric = self._size_metric_combo.currentText()

        try:
            from ..visualization.renderer.fragment_renderer import FragmentRenderer

            renderer = None
            if self.controller and hasattr(self.controller, 'r'):
                renderer = self.controller.r

            if renderer is None:
                QMessageBox.warning(self, "No Renderer", "3D renderer not available.")
                return

            frag_renderer = FragmentRenderer(renderer)

            # Get global shift for coordinate precision
            global_shift = getattr(renderer, '_global_shift', None)

            frag_renderer.render_size_map(
                cloud=dataset.fused_cloud,
                fragment_labels=dataset.fragment_labels,
                fragments=dataset.fragments,
                metric=metric,
                point_size=3.0,
                global_shift=global_shift,
            )

            # Force render update
            plotter = getattr(renderer, 'plotter', None)
            if plotter:
                plotter.render()

            self._status_label = getattr(self, '_status_label', None)

        except Exception as e:
            logger.error("Size visualisation failed: %s", e)
            QMessageBox.critical(self, "Visualisation Failed", str(e))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_csv(self):
        registry = self.get_registry()
        if not registry or not registry.has_fragment_dataset():
            QMessageBox.warning(self, "No Data", "No fragment data to export.")
            return

        dataset = registry.get_fragment_dataset(copy_data=False)
        if not dataset or not dataset.fragments:
            QMessageBox.warning(self, "No Fragments", "Run segmentation and size extraction first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Fragments CSV", f"{dataset.name}_fragments.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "fragment_id", "x", "y", "z", "point_count",
                    "equiv_diameter_m", "feret_max_m", "feret_min_m",
                    "projected_area_m2", "volume_m3", "surface_area_m2",
                    "aspect_ratio", "sphericity", "elongation",
                    "source_method", "confidence",
                ])
                for fr in dataset.fragments:
                    writer.writerow([
                        fr.fragment_id,
                        f"{fr.centroid_xyz[0]:.4f}",
                        f"{fr.centroid_xyz[1]:.4f}",
                        f"{fr.centroid_xyz[2]:.4f}",
                        fr.point_count,
                        f"{fr.equiv_diameter:.6f}",
                        f"{fr.feret_max:.6f}",
                        f"{fr.feret_min:.6f}",
                        f"{fr.projected_area:.6f}",
                        f"{fr.volume_estimate:.6f}",
                        f"{fr.surface_area:.6f}",
                        f"{fr.aspect_ratio:.4f}",
                        f"{fr.sphericity:.4f}",
                        f"{fr.elongation:.4f}",
                        fr.source_method.value,
                        f"{fr.confidence:.4f}",
                    ])
            QMessageBox.information(self, "Export Complete", f"Exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    # ------------------------------------------------------------------
    # Table Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _num_item(value) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setData(Qt.ItemDataRole.DisplayRole, int(value))
        item.setData(Qt.ItemDataRole.UserRole, int(value))
        return item

    @staticmethod
    def _float_item(value, decimals=4) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setData(Qt.ItemDataRole.DisplayRole, round(float(value), decimals))
        item.setData(Qt.ItemDataRole.UserRole, float(value))
        return item
