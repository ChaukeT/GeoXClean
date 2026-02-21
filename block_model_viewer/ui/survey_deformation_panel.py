"""
Survey Deformation & Subsidence Panel.

Provides an end-to-end workflow UI for:
- Ingesting subsidence survey data (Point ID, Easting, Northing, Elevation, Survey Date)
- Ingesting groundwater well data (Well ID, Easting, Northing, Date, Water Level)
- Running subsidence time-series, control stability, groundwater stats, coupling, and deformation index
- Surfacing basic results/row counts with registry-backed provenance
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QLineEdit,
)

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


class SurveyDeformationPanel(BaseAnalysisPanel):
    PANEL_ID = "SurveyDeformationPanel"
    PANEL_NAME = "Survey Deformation & Subsidence"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Auto-bind controller if parent is a MainWindow
        if hasattr(parent, "controller"):
            self.bind_controller(parent.controller)

    def setup_ui(self):
        self.setMinimumWidth(420)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Data status
        status_group = QGroupBox("Data Status")
        status_layout = QFormLayout()
        self.subsidence_status = QLabel("Subsidence: not loaded")
        self.groundwater_status = QLabel("Groundwater: not loaded")
        status_layout.addRow(self.subsidence_status)
        status_layout.addRow(self.groundwater_status)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Import controls
        import_group = QGroupBox("Import")
        import_layout = QHBoxLayout()
        self.import_subs_btn = QPushButton("Import Subsidence Surveys")
        self.import_subs_btn.clicked.connect(self._import_subsidence)
        self.import_gw_btn = QPushButton("Import Groundwater Wells")
        self.import_gw_btn.clicked.connect(self._import_groundwater)
        import_layout.addWidget(self.import_subs_btn)
        import_layout.addWidget(self.import_gw_btn)
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        self.ma_window = QSpinBox()
        self.ma_window.setRange(1, 365)
        self.ma_window.setValue(3)
        self.survey_precision = QDoubleSpinBox()
        self.survey_precision.setDecimals(2)
        self.survey_precision.setRange(0.1, 1000.0)
        self.survey_precision.setValue(5.0)
        self.sigma = QDoubleSpinBox()
        self.sigma.setDecimals(2)
        self.sigma.setRange(0.5, 6.0)
        self.sigma.setValue(2.0)
        self.max_distance = QDoubleSpinBox()
        self.max_distance.setDecimals(1)
        self.max_distance.setRange(10.0, 10000.0)
        self.max_distance.setValue(1500.0)

        params_layout.addRow("Moving average window (epochs)", self.ma_window)
        params_layout.addRow("Survey precision (mm)", self.survey_precision)
        params_layout.addRow("Significance (sigma)", self.sigma)
        params_layout.addRow("Coupling max distance (m)", self.max_distance)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Actions
        actions_group = QGroupBox("Run Analysis")
        actions_layout = QVBoxLayout()
        self.run_subs_btn = QPushButton("Run Subsidence Time-Series")
        self.run_subs_btn.clicked.connect(self._run_subsidence)
        self.run_stability_btn = QPushButton("Run Control Stability")
        self.run_stability_btn.clicked.connect(self._run_stability)
        self.run_gw_btn = QPushButton("Run Groundwater Analysis")
        self.run_gw_btn.clicked.connect(self._run_groundwater)
        self.run_coupling_btn = QPushButton("Run Coupled Interpretation")
        self.run_coupling_btn.clicked.connect(self._run_coupling)
        self.run_index_btn = QPushButton("Compute Deformation Index")
        self.run_index_btn.clicked.connect(self._run_index)

        for btn in [
            self.run_subs_btn,
            self.run_stability_btn,
            self.run_gw_btn,
            self.run_coupling_btn,
            self.run_index_btn,
        ]:
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            actions_layout.addWidget(btn)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.results_tabs = QTabWidget()
        self.table_timeseries = self._create_table_widget()
        self.table_metrics = self._create_table_widget()
        self.table_stability = self._create_table_widget()
        self.table_groundwater = self._create_table_widget()
        self.table_coupling = self._create_table_widget()
        self.table_index = self._create_table_widget()
        self.results_tabs.addTab(self.table_metrics, "Subsidence Metrics")
        self.results_tabs.addTab(self.table_timeseries, "Timeseries")
        self.results_tabs.addTab(self.table_stability, "Control Stability")
        self.results_tabs.addTab(self.table_groundwater, "Groundwater Metrics")
        self.results_tabs.addTab(self.table_coupling, "Coupling")
        self.results_tabs.addTab(self.table_index, "Deformation Index")
        results_layout.addWidget(self.results_tabs)

        # Plot preview
        preview_group = QGroupBox("2D Preview")
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Point ID:"))
        self.point_input = QLineEdit()
        self.point_input.setPlaceholderText("Enter point_id")
        preview_layout.addWidget(self.point_input)
        self.preview_btn = QPushButton("Plot Elevation & ΔZ")
        self.preview_btn.clicked.connect(self._plot_point)
        preview_layout.addWidget(self.preview_btn)
        preview_group.setLayout(preview_layout)
        results_layout.addWidget(preview_group)

        self.plot_label = QLabel("Run analysis and pick a point to preview.")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setMinimumHeight(180)
        results_layout.addWidget(self.plot_label)

        # Text log
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(100)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()
        self.setLayout(layout)
        self._refresh_status()

    def connect_signals(self):
        return

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def _import_subsidence(self):
        try:
            df, path = self._load_dataframe()
            if df is None:
                return
            key = self.controller.survey_deformation.ingest_subsidence_dataframe(df, source_file=path)
            self._append_result(f"Subsidence loaded: {len(df):,} rows -> {key}")
            self._refresh_status()
        except Exception as e:
            logger.error("Failed to import subsidence data", exc_info=True)
            QMessageBox.critical(self, "Import Error", f"Failed to import subsidence data:\n{e}")

    def _import_groundwater(self):
        try:
            df, path = self._load_dataframe()
            if df is None:
                return
            key = self.controller.survey_deformation.ingest_groundwater_dataframe(df, source_file=path)
            self._append_result(f"Groundwater loaded: {len(df):,} rows -> {key}")
            self._refresh_status()
        except Exception as e:
            logger.error("Failed to import groundwater data", exc_info=True)
            QMessageBox.critical(self, "Import Error", f"Failed to import groundwater data:\n{e}")

    def _run_subsidence(self):
        try:
            result = self.controller.survey_deformation.run_subsidence_time_series(
                parameters={"ma_window": self.ma_window.value()}
            )
            self._append_result(
                f"Subsidence metrics computed for {len(result.per_point_metrics)} points "
                f"(timeseries rows: {len(result.timeseries)})"
            )
            self._populate_tables()
            self._refresh_status()
        except Exception as e:
            logger.error("Subsidence analysis failed", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Subsidence analysis failed:\n{e}")

    def _run_stability(self):
        try:
            bundle = self.controller.survey_deformation.run_control_stability(
                survey_precision_mm=self.survey_precision.value(),
                significance_sigma=self.sigma.value(),
            )
            df = bundle["control_stability"]
            self._append_result(f"Control stability classified {len(df)} points")
            self._populate_tables()
            self._refresh_status()
        except Exception as e:
            logger.error("Control stability failed", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Control stability failed:\n{e}")

    def _run_groundwater(self):
        try:
            bundle = self.controller.survey_deformation.run_groundwater_time_series()
            df = bundle["metrics"]
            self._append_result(f"Groundwater rates computed for {len(df)} wells")
            self._populate_tables()
            self._refresh_status()
        except Exception as e:
            logger.error("Groundwater analysis failed", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Groundwater analysis failed:\n{e}")

    def _run_coupling(self):
        try:
            bundle = self.controller.survey_deformation.run_coupling(max_distance=self.max_distance.value())
            df = bundle["coupling"]
            self._append_result(f"Coupling computed for {len(df)} points")
            self._populate_tables()
            self._refresh_status()
        except Exception as e:
            logger.error("Coupling failed", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Coupling failed:\n{e}")

    def _run_index(self):
        try:
            bundle = self.controller.survey_deformation.run_deformation_index()
            df = bundle["deformation_index"]
            self._append_result(f"Deformation index ranked {len(df)} points")
            self._populate_tables()
            self._refresh_status()
        except Exception as e:
            logger.error("Deformation index failed", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Deformation index failed:\n{e}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _load_dataframe(self) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
        dlg = QFileDialog(self)
        dlg.setNameFilters(["CSV files (*.csv)", "Excel files (*.xlsx *.xls)", "All files (*.*)"])
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        if not dlg.exec():
            return None, None
        selected = dlg.selectedFiles()
        if not selected:
            return None, None
        path = Path(selected[0])
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        return df, path

    def _refresh_status(self):
        reg = self.get_registry()
        subs_df = reg.get_data(self.controller.survey_deformation.SUBSIDENCE_RAW_KEY) if reg else None
        gw_df = reg.get_data(self.controller.survey_deformation.GROUNDWATER_RAW_KEY) if reg else None
        subs_count = len(subs_df) if subs_df is not None else 0
        gw_count = len(gw_df) if gw_df is not None else 0
        self.subsidence_status.setText(f"Subsidence: {'loaded' if subs_df is not None else 'not loaded'} ({subs_count:,} rows)")
        self.groundwater_status.setText(f"Groundwater: {'loaded' if gw_df is not None else 'not loaded'} ({gw_count:,} rows)")

    def _append_result(self, text: str):
        existing = self.results_text.toPlainText()
        new_text = text if not existing else existing + "\n" + text
        self.results_text.setPlainText(new_text)
        self.results_text.verticalScrollBar().setValue(self.results_text.verticalScrollBar().maximum())

    # ------------------------------------------------------------------ #
    # Table & plotting helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_table_widget() -> QTableWidget:
        table = QTableWidget()
        table.setSortingEnabled(True)
        table.setMinimumHeight(140)
        return table

    def _populate_table(self, table: QTableWidget, df: Optional[pd.DataFrame], max_rows: int = 500):
        table.clear()
        if df is None or df.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        df_show = df.head(max_rows)
        table.setColumnCount(len(df_show.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df_show.columns])
        table.setRowCount(len(df_show))
        for i, (_, row) in enumerate(df_show.iterrows()):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                table.setItem(i, j, item)
        table.resizeColumnsToContents()

    def _populate_tables(self):
        reg = self.get_registry()
        if reg is None:
            return
        self._populate_table(self.table_metrics, reg.get_data(self.controller.survey_deformation.SUBSIDENCE_METRICS_KEY))
        self._populate_table(self.table_timeseries, reg.get_data(self.controller.survey_deformation.SUBSIDENCE_TIMESERIES_KEY))
        self._populate_table(self.table_stability, reg.get_data(self.controller.survey_deformation.CONTROL_STABILITY_KEY))
        self._populate_table(self.table_groundwater, reg.get_data(self.controller.survey_deformation.GROUNDWATER_METRICS_KEY))
        self._populate_table(self.table_coupling, reg.get_data(self.controller.survey_deformation.COUPLING_KEY))
        self._populate_table(self.table_index, reg.get_data(self.controller.survey_deformation.DEFORMATION_INDEX_KEY))

    def _plot_point(self):
        point_id = self.point_input.text().strip()
        if not point_id:
            QMessageBox.information(self, "Select Point", "Enter a point_id to plot.")
            return
        reg = self.get_registry()
        if reg is None:
            QMessageBox.warning(self, "Registry Missing", "No registry available.")
            return
        ts = reg.get_data(self.controller.survey_deformation.SUBSIDENCE_TIMESERIES_KEY)
        if ts is None or ts.empty:
            QMessageBox.information(self, "No Data", "Run subsidence analysis first.")
            return
        subset = ts[ts["point_id"] == point_id]
        if subset.empty:
            QMessageBox.information(self, "Not Found", f"No records for point_id '{point_id}'.")
            return
        fig, ax = plt.subplots(2, 1, figsize=(4, 3.6), constrained_layout=True)
        ax[0].plot(subset["survey_date"], subset["elevation"], marker="o")
        ax[0].set_title(f"Elevation vs Time ({point_id})")
        ax[0].set_ylabel("Elevation")
        ax[1].plot(subset["survey_date"], subset["delta_z_mm"], marker="o", color="darkred")
        ax[1].set_title("ΔZ (mm)")
        ax[1].set_ylabel("ΔZ (mm)")
        ax[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        for axis in ax:
            axis.tick_params(axis="x", labelrotation=25)
            axis.grid(True, linestyle="--", alpha=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read(), "PNG")
        self.plot_label.setPixmap(pixmap)
        self.plot_label.setScaledContents(True)
