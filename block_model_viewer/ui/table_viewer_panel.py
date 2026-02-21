"""
Generic Table Viewer Panel
--------------------------
Reusable panel for viewing any pandas DataFrame in a searchable/filterable table.
Intended for drillhole composites, IRR results, and other tabular outputs.
"""

from __future__ import annotations

import logging
from typing import Optional
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox, QHeaderView,
    QComboBox, QSpinBox, QGroupBox, QProgressDialog, QApplication
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QColor, QCursor

logger = logging.getLogger(__name__)


class TableViewerPanel(QWidget):
    """Panel for viewing arbitrary DataFrame data in table format."""

    def __init__(self):
        super().__init__()
        self.df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None

        self._setup_ui()
        logger.info("Initialized Generic Table Viewer panel")

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Header with info
        header_layout = QHBoxLayout()
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        header_layout.addWidget(self.info_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Search and filter controls
        filter_group = QGroupBox("Search & Filter")
        filter_layout = QVBoxLayout()

        # Search bar
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in all columns...")
        self.search_input.textChanged.connect(self._apply_search)
        search_layout.addWidget(self.search_input)

        self.clear_search_btn = QPushButton("Clear")
        self.clear_search_btn.clicked.connect(self._clear_search)
        search_layout.addWidget(self.clear_search_btn)
        filter_layout.addLayout(search_layout)

        # Property filter (numeric columns)
        property_filter_layout = QHBoxLayout()
        property_filter_layout.addWidget(QLabel("Filter Property:"))
        self.property_combo = QComboBox()
        self.property_combo.addItem("-- Select Property --")
        self.property_combo.currentTextChanged.connect(self._on_property_selected)
        property_filter_layout.addWidget(self.property_combo)

        property_filter_layout.addWidget(QLabel("Min:"))
        self.min_value_spin = QSpinBox()
        self.min_value_spin.setRange(-999999999, 999999999)
        self.min_value_spin.setEnabled(False)
        property_filter_layout.addWidget(self.min_value_spin)

        property_filter_layout.addWidget(QLabel("Max:"))
        self.max_value_spin = QSpinBox()
        self.max_value_spin.setRange(-999999999, 999999999)
        self.max_value_spin.setEnabled(False)
        property_filter_layout.addWidget(self.max_value_spin)

        self.apply_filter_btn = QPushButton("Apply Filter")
        self.apply_filter_btn.setEnabled(False)
        self.apply_filter_btn.clicked.connect(self._apply_property_filter)
        property_filter_layout.addWidget(self.apply_filter_btn)

        self.clear_filter_btn = QPushButton("Clear Filter")
        self.clear_filter_btn.setEnabled(False)
        self.clear_filter_btn.clicked.connect(self._clear_property_filter)
        property_filter_layout.addWidget(self.clear_filter_btn)

        filter_layout.addLayout(property_filter_layout)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Table widget
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # Bottom toolbar
        toolbar_layout = QHBoxLayout()
        self.rows_label = QLabel("Rows: 0")
        toolbar_layout.addWidget(self.rows_label)
        toolbar_layout.addStretch()

        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_to_csv)
        toolbar_layout.addWidget(self.export_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.clicked.connect(self._refresh_table)
        toolbar_layout.addWidget(self.refresh_btn)

        layout.addLayout(toolbar_layout)

    # ---------------------- Public API ----------------------
    def set_dataframe(self, dataframe: pd.DataFrame, title: Optional[str] = None):
        """Set the DataFrame to display."""
        try:
            if dataframe is None or dataframe.empty:
                self.df = None
                self.filtered_df = None
                self._clear_table_only()
                self.info_label.setText("No data loaded")
                self.rows_label.setText("Rows: 0")
                self.export_btn.setEnabled(False)
                self.refresh_btn.setEnabled(False)
                return

            self.df = dataframe.copy()
            self.filtered_df = self.df.copy()

            # Update info
            self.info_label.setText(
                f"{title or 'Table'}: {len(self.df)} rows, {len(self.df.columns)} columns"
            )

            # Populate property combo with numeric columns
            self.property_combo.clear()
            self.property_combo.addItem("-- Select Property --")
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols:
                self.property_combo.addItem(col)

            # Populate table
            self._populate_table(self.filtered_df)

            # Enable controls
            self.export_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)

            logger.info(
                f"Table Viewer loaded dataframe with {len(self.df)} rows and {len(self.df.columns)} columns"
            )
        except Exception as e:
            logger.error(f"Error setting dataframe in table viewer: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load table data:\n{e}")

    # ---------------------- Internals ----------------------
    def _populate_table(self, dataframe: pd.DataFrame):
        progress = None
        try:
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            self.table.setSortingEnabled(False)
            self.table.setUpdatesEnabled(False)

            # Limit rows for performance (configurable via Preferences)
            try:
                s = QSettings("GeoX", "TableViewer")
                max_rows = int(s.value("row_limit", 5000))
            except Exception:
                max_rows = 5000
            display_df = dataframe.head(max_rows) if len(dataframe) > max_rows else dataframe

            progress = QProgressDialog("Loading table data...", "Cancel", 0, len(display_df), self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(500)

            # Set dimensions
            self.table.setRowCount(len(display_df))
            self.table.setColumnCount(len(display_df.columns))
            self.table.setHorizontalHeaderLabels(display_df.columns.tolist())
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

            # Populate cells
            chunk_size = 100
            for row_idx in range(len(display_df)):
                if row_idx % chunk_size == 0:
                    if progress.wasCanceled():
                        self.table.setRowCount(0)
                        break
                    progress.setValue(row_idx)
                    QApplication.processEvents()

                try:
                    for col_idx in range(len(display_df.columns)):
                        value = display_df.iloc[row_idx, col_idx]

                        if pd.isna(value):
                            item_text = ""
                        elif isinstance(value, float):
                            item_text = f"{value:.6g}"
                        else:
                            item_text = str(value)

                        item = QTableWidgetItem(item_text)
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                        # Optional dark-theme-friendly color coding for numeric values
                        col_name = display_df.columns[col_idx]
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            if value > 0:
                                item.setBackground(QColor(60, 90, 60))   # Dark green tint
                            elif value < 0:
                                item.setBackground(QColor(90, 60, 60))   # Dark red tint

                        self.table.setItem(row_idx, col_idx, item)
                except Exception as e:
                    logger.warning(f"Error populating row {row_idx}: {e}")
                    continue

            if not progress.wasCanceled():
                progress.setValue(len(display_df))

            # Resize first 10 columns to contents
            for col in range(min(len(display_df.columns), 10)):
                self.table.resizeColumnToContents(col)

            # Update row count label
            if len(dataframe) > max_rows:
                self.rows_label.setText(f"Rows: {len(display_df)} of {len(dataframe)} (showing first {max_rows})")
            else:
                self.rows_label.setText(f"Rows: {len(dataframe)}")

            self.table.setUpdatesEnabled(True)
            self.table.setSortingEnabled(True)
        except Exception as e:
            self.table.setUpdatesEnabled(True)
            logger.error(f"Error populating table: {e}", exc_info=True)
            QMessageBox.warning(self, "Table Error", f"Failed to populate table:\n{e}")
        finally:
            if progress:
                progress.close()
            QApplication.restoreOverrideCursor()

    def _apply_search(self, text: str):
        if self.df is None or text.strip() == "":
            if self.df is not None:
                self._populate_table(self.df)
            return

        try:
            mask = self.df.astype(str).apply(lambda x: x.str.contains(text, case=False, na=False)).any(axis=1)
            self.filtered_df = self.df[mask]
            self._populate_table(self.filtered_df)
        except Exception as e:
            logger.error(f"Error applying search: {e}")

    def _clear_search(self):
        self.search_input.clear()
        if self.df is not None:
            self.filtered_df = self.df.copy()
            self._populate_table(self.filtered_df)

    def _on_property_selected(self, property_name: str):
        if not property_name or property_name == "-- Select Property --" or self.df is None:
            self.min_value_spin.setEnabled(False)
            self.max_value_spin.setEnabled(False)
            self.apply_filter_btn.setEnabled(False)
            self.clear_filter_btn.setEnabled(False)
            return

        if property_name not in self.df.columns:
            logger.warning(f"Property '{property_name}' not found in dataframe")
            self.min_value_spin.setEnabled(False)
            self.max_value_spin.setEnabled(False)
            self.apply_filter_btn.setEnabled(False)
            self.clear_filter_btn.setEnabled(False)
            return

        try:
            min_val = int(self.df[property_name].min())
            max_val = int(self.df[property_name].max())

            self.min_value_spin.setRange(min_val, max_val)
            self.min_value_spin.setValue(min_val)
            self.max_value_spin.setRange(min_val, max_val)
            self.max_value_spin.setValue(max_val)

            self.min_value_spin.setEnabled(True)
            self.max_value_spin.setEnabled(True)
            self.apply_filter_btn.setEnabled(True)
            self.clear_filter_btn.setEnabled(True)
        except Exception as e:
            logger.error(f"Error setting property filter range: {e}", exc_info=True)
            self.min_value_spin.setEnabled(False)
            self.max_value_spin.setEnabled(False)
            self.apply_filter_btn.setEnabled(False)
            self.clear_filter_btn.setEnabled(False)

    def _apply_property_filter(self):
        if self.df is None:
            return

        property_name = self.property_combo.currentText()
        if property_name == "-- Select Property --":
            return

        try:
            min_val = self.min_value_spin.value()
            max_val = self.max_value_spin.value()

            mask = (self.df[property_name] >= min_val) & (self.df[property_name] <= max_val)
            self.filtered_df = self.df[mask]
            self._populate_table(self.filtered_df)

            logger.info(f"Applied filter: {property_name} in [{min_val}, {max_val}]")
        except Exception as e:
            logger.error(f"Error applying property filter: {e}")
            QMessageBox.warning(self, "Error", f"Failed to apply filter:\n{e}")

    def _clear_property_filter(self):
        self.property_combo.setCurrentIndex(0)
        if self.df is not None:
            self.filtered_df = self.df.copy()
            self._populate_table(self.filtered_df)

    def _refresh_table(self):
        if self.df is not None:
            self.set_dataframe(self.df)

    def _export_to_csv(self):
        if self.filtered_df is None or self.filtered_df.empty:
            QMessageBox.warning(self, "No Data", "No data available to export.")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Table Data",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )

            if file_path:
                # Step 12: Use ExportHelpers
                from ..utils.export_helpers import export_dataframe_to_csv
                export_dataframe_to_csv(self.filtered_df, file_path)
                logger.info(f"Exported {len(self.filtered_df)} rows to {file_path}")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Exported {len(self.filtered_df)} rows to:\n{file_path}"
                )
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{e}")

    def clear(self):
        self.df = None
        self.filtered_df = None
        self._clear_table_only()
        self.info_label.setText("No data loaded")
        self.rows_label.setText("Rows: 0")
        self.search_input.clear()
        self.property_combo.clear()
        self.property_combo.addItem("-- Select Property --")
        self.export_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        logger.info("Cleared table viewer")

    def clear_panel(self):
        """Clear all panel UI and state to initial defaults."""
        self.clear()
        super().clear_panel()
        logger.info("TableViewerPanel: Panel fully cleared")

    def _clear_table_only(self):
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors
        self.setStyleSheet(get_analysis_panel_stylesheet())
