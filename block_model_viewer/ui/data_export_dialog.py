"""
Data Export Dialog - Comprehensive export functionality.

Allows users to export various data types (block models, drillholes, etc.)
with control over format, resolution, and specific datasets.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QComboBox, QLabel, QPushButton, QFileDialog, QMessageBox,
    QCheckBox, QSpinBox, QProgressDialog, QButtonGroup
)
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class DataExportDialog(QDialog):
    """Dialog for exporting various data types from the application."""

    def __init__(self, registry, parent=None):
        """
        Initialize the export dialog.

        Args:
            registry: DataRegistry instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.registry = registry
        self.setWindowTitle("Export Data")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self._build_ui()
        self._update_dataset_options()

    def _build_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)

        # Data Type Selection
        data_type_group = QGroupBox("1. Select Data Type")
        data_type_layout = QVBoxLayout()

        self.data_type_group = QButtonGroup(self)
        self.rb_block_model = QRadioButton("Block Model")
        self.rb_drillholes = QRadioButton("Drillhole Data")
        self.rb_estimation = QRadioButton("Estimation Results")
        self.rb_statistics = QRadioButton("Statistical Results")

        self.data_type_group.addButton(self.rb_block_model, 0)
        self.data_type_group.addButton(self.rb_drillholes, 1)
        self.data_type_group.addButton(self.rb_estimation, 2)
        self.data_type_group.addButton(self.rb_statistics, 3)

        data_type_layout.addWidget(self.rb_block_model)
        data_type_layout.addWidget(self.rb_drillholes)
        data_type_layout.addWidget(self.rb_estimation)
        data_type_layout.addWidget(self.rb_statistics)
        data_type_group.setLayout(data_type_layout)
        layout.addWidget(data_type_group)

        # Dataset Selection
        dataset_group = QGroupBox("2. Select Specific Dataset")
        dataset_layout = QVBoxLayout()

        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(400)
        dataset_layout.addWidget(self.dataset_combo)

        self.dataset_info_label = QLabel("")
        self.dataset_info_label.setWordWrap(True)
        self.dataset_info_label.setStyleSheet("color: #666; font-size: 10pt;")
        dataset_layout.addWidget(self.dataset_info_label)

        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # File Format Selection
        format_group = QGroupBox("3. Select File Format")
        format_layout = QVBoxLayout()

        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "CSV (*.csv)",
            "Excel (*.xlsx)",
            "Parquet (*.parquet)",
            "VTK Grid (*.vtk)",
            "VTU Unstructured Grid (*.vtu)",
            "Pickle (*.pkl)"
        ])
        format_layout.addWidget(self.format_combo)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Options Group (for block models)
        self.options_group = QGroupBox("4. Export Options")
        options_layout = QVBoxLayout()

        # Resolution control for block models
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Downsampling Factor:"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setMinimum(1)
        self.downsample_spin.setMaximum(10)
        self.downsample_spin.setValue(1)
        self.downsample_spin.setToolTip("Reduce resolution by this factor (1=full resolution, 2=half, etc.)")
        resolution_layout.addWidget(self.downsample_spin)
        resolution_layout.addStretch()
        options_layout.addLayout(resolution_layout)

        # Include metadata checkbox
        self.include_metadata_check = QCheckBox("Include metadata and provenance")
        self.include_metadata_check.setChecked(True)
        options_layout.addWidget(self.include_metadata_check)

        # Include coordinates checkbox (for drillholes)
        self.include_coords_check = QCheckBox("Include 3D coordinates")
        self.include_coords_check.setChecked(True)
        options_layout.addWidget(self.include_coords_check)

        self.options_group.setLayout(options_layout)
        layout.addWidget(self.options_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.export_btn = QPushButton("Export...")
        self.export_btn.setDefault(True)
        self.export_btn.clicked.connect(self._on_export)
        button_layout.addWidget(self.export_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        # Connect signals
        self.data_type_group.buttonClicked.connect(self._on_data_type_changed)
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)

        # Set default selection
        self.rb_block_model.setChecked(True)
        self._on_data_type_changed()

    def _on_data_type_changed(self):
        """Handle data type selection change."""
        self._update_dataset_options()
        self._update_options_visibility()

    def _update_dataset_options(self):
        """Update the dataset combo box based on selected data type."""
        self.dataset_combo.clear()

        if self.rb_block_model.isChecked():
            # Block model options
            datasets = []

            # Check for all registered block models (multi-model support)
            models = self.registry.get_block_model_list()
            if models:
                for model_info in models:
                    model_id = model_info['model_id']
                    row_count = model_info['row_count']
                    is_current = model_info['is_current']

                    display_name = f"{model_id}{' (current)' if is_current else ''}"
                    data_info = f"{model_id} ({row_count:,} blocks)"
                    # Store model_id in dataset combo for retrieval
                    datasets.append((display_name, model_id))

            # Check for classified block model (legacy support)
            classified = self.registry.get_classified_block_model(copy_data=False)
            if classified is not None:
                # Check if already in list
                already_added = any('classified' in name.lower() for name, _ in datasets)
                if not already_added:
                    if hasattr(classified, 'block_count'):
                        datasets.append(("Classified Block Model", "classified_block_model"))
                    else:
                        datasets.append(("Classified Block Model", "classified_block_model"))

            # Add estimation results that are block models
            estimation_types = [
                ("kriging_results", "Kriging"),
                ("sgsim_results", "SGSIM"),
                ("simple_kriging_results", "Simple Kriging"),
                ("cokriging_results", "Co-Kriging"),
                ("indicator_kriging_results", "Indicator Kriging"),
                ("universal_kriging_results", "Universal Kriging"),
            ]

            for key, name in estimation_types:
                results = self.registry.get_data(key, copy_data=False)
                if results and isinstance(results, dict):
                    grid = results.get('grid') or results.get('block_model')
                    if grid is not None:
                        if hasattr(grid, 'n_cells'):
                            datasets.append((name, f"{key} ({grid.n_cells:,} cells)"))
                        elif hasattr(grid, '__len__'):
                            datasets.append((name, f"{key} ({len(grid):,} rows)"))

            if not datasets:
                self.dataset_combo.addItem("No block models available")
                self.export_btn.setEnabled(False)
            else:
                for display_name, data_info in datasets:
                    self.dataset_combo.addItem(display_name, data_info)
                self.export_btn.setEnabled(True)

        elif self.rb_drillholes.isChecked():
            # Drillhole data options
            drillhole_data = self.registry.get_data("drillhole_data", copy_data=False)

            if drillhole_data is None:
                self.dataset_combo.addItem("No drillhole data available")
                self.export_btn.setEnabled(False)
            else:
                datasets = []

                # Check each drillhole data type
                for key in ['composites', 'assays', 'collars', 'surveys', 'trajectories', 'lithology']:
                    df = drillhole_data.get(key)
                    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                        datasets.append((key.capitalize(), f"{key} ({len(df):,} rows)"))

                if not datasets:
                    self.dataset_combo.addItem("No drillhole data available")
                    self.export_btn.setEnabled(False)
                else:
                    for display_name, data_info in datasets:
                        self.dataset_combo.addItem(display_name, data_info)
                    self.export_btn.setEnabled(True)

        elif self.rb_estimation.isChecked():
            # Estimation results (variograms, etc.)
            datasets = []

            # Variogram results
            variogram = self.registry.get_variogram_results(copy_data=False)
            if variogram:
                datasets.append(("Variogram Results", "variogram_results"))

            # Declustering results
            decluster = self.registry.get_declustering_results(copy_data=False)
            if decluster:
                datasets.append(("Declustering Results", "declustering_results"))

            if not datasets:
                self.dataset_combo.addItem("No estimation results available")
                self.export_btn.setEnabled(False)
            else:
                for display_name, data_info in datasets:
                    self.dataset_combo.addItem(display_name, data_info)
                self.export_btn.setEnabled(True)

        elif self.rb_statistics.isChecked():
            # Statistical results
            datasets = []

            # Resource summary
            resource = self.registry.get_resource_summary(copy_data=False)
            if resource:
                datasets.append(("Resource Summary", "resource_summary"))

            # GeometOr results
            geomet = self.registry.get_geomet_results(copy_data=False)
            if geomet:
                datasets.append(("GeometOR Results", "geomet_results"))

            if not datasets:
                self.dataset_combo.addItem("No statistical results available")
                self.export_btn.setEnabled(False)
            else:
                for display_name, data_info in datasets:
                    self.dataset_combo.addItem(display_name, data_info)
                self.export_btn.setEnabled(True)

        self._on_dataset_changed()

    def _on_dataset_changed(self):
        """Handle dataset selection change."""
        if self.dataset_combo.count() == 0:
            self.dataset_info_label.setText("")
            return

        current_data = self.dataset_combo.currentData()
        if current_data:
            self.dataset_info_label.setText(f"Data: {current_data}")
        else:
            self.dataset_info_label.setText("")

    def _update_options_visibility(self):
        """Update visibility of options based on data type."""
        is_block_model = self.rb_block_model.isChecked()
        is_drillholes = self.rb_drillholes.isChecked()

        # Show downsampling only for block models
        self.downsample_spin.setVisible(is_block_model)
        self.downsample_spin.parent().parent().setVisible(is_block_model)  # The layout

        # Show coordinate options only for drillholes
        self.include_coords_check.setVisible(is_drillholes)

    def _on_format_changed(self):
        """Handle format selection change."""
        format_text = self.format_combo.currentText()

        # Adjust available options based on format
        if "VTK" in format_text or "VTU" in format_text:
            # VTK formats require grid data
            if not self.rb_block_model.isChecked():
                QMessageBox.warning(
                    self,
                    "Format Incompatible",
                    "VTK/VTU formats are only available for block model exports."
                )
                self.format_combo.setCurrentIndex(0)  # Reset to CSV

    def _on_export(self):
        """Handle export button click."""
        try:
            # Determine data to export
            data_type = self._get_selected_data_type()
            dataset_key = self._get_selected_dataset_key()

            if not dataset_key:
                QMessageBox.warning(self, "No Selection", "Please select a dataset to export.")
                return

            # Get file format
            format_text = self.format_combo.currentText()
            extension = self._get_extension_from_format(format_text)

            # Get default filename
            default_filename = f"{dataset_key}{extension}"

            # Get save location
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Export Data",
                default_filename,
                format_text
            )

            if not filepath:
                return  # User cancelled

            # Perform export
            success = self._export_data(dataset_key, filepath, format_text)

            if success:
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Data exported successfully to:\n{filepath}"
                )
                self.accept()
            else:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    "Failed to export data. Check the log for details."
                )

        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred during export:\n{str(e)}"
            )

    def _get_selected_data_type(self) -> str:
        """Get the selected data type."""
        if self.rb_block_model.isChecked():
            return "block_model"
        elif self.rb_drillholes.isChecked():
            return "drillholes"
        elif self.rb_estimation.isChecked():
            return "estimation"
        elif self.rb_statistics.isChecked():
            return "statistics"
        return ""

    def _get_selected_dataset_key(self) -> Optional[str]:
        """Get the registry key for the selected dataset."""
        if self.dataset_combo.count() == 0:
            return None

        current_text = self.dataset_combo.currentText()
        current_data = self.dataset_combo.currentData()

        if "No" in current_text and "available" in current_text:
            return None

        # Map display names to registry keys
        if self.rb_block_model.isChecked():
            if "Block Model" in current_text and "Classified" not in current_text:
                return "block_model"
            elif "Classified" in current_text:
                return "classified_block_model"
            elif "Kriging" in current_text:
                return "kriging_results"
            elif "SGSIM" in current_text:
                return "sgsim_results"
            elif "Simple Kriging" in current_text:
                return "simple_kriging_results"
            elif "Co-Kriging" in current_text:
                return "cokriging_results"
            elif "Indicator Kriging" in current_text:
                return "indicator_kriging_results"
            elif "Universal Kriging" in current_text:
                return "universal_kriging_results"

        elif self.rb_drillholes.isChecked():
            return current_text.lower()  # composites, assays, etc.

        elif self.rb_estimation.isChecked():
            if "Variogram" in current_text:
                return "variogram_results"
            elif "Declustering" in current_text:
                return "declustering_results"

        elif self.rb_statistics.isChecked():
            if "Resource" in current_text:
                return "resource_summary"
            elif "GeometOR" in current_text:
                return "geomet_results"

        return None

    def _get_extension_from_format(self, format_text: str) -> str:
        """Extract file extension from format string."""
        if "*.csv" in format_text:
            return ".csv"
        elif "*.xlsx" in format_text:
            return ".xlsx"
        elif "*.parquet" in format_text:
            return ".parquet"
        elif "*.vtk" in format_text:
            return ".vtk"
        elif "*.vtu" in format_text:
            return ".vtu"
        elif "*.pkl" in format_text:
            return ".pkl"
        return ".csv"

    def _export_data(self, dataset_key: str, filepath: str, format_text: str) -> bool:
        """
        Export the selected data to file.

        Args:
            dataset_key: Registry key for the dataset
            filepath: Output file path
            format_text: Format description

        Returns:
            True if successful, False otherwise
        """
        progress = None
        try:
            logger.info(f"Exporting {dataset_key} to {filepath}")

            # Show progress dialog
            progress = QProgressDialog("Preparing data for export...", "Cancel", 0, 0, self)
            progress.setWindowTitle("Exporting Data")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(500)  # Show after 500ms
            progress.setValue(0)

            # Get the data
            data = None

            if self.rb_block_model.isChecked():
                # Try to get as multi-model first (dataset_key = model_id)
                data = self.registry.get_block_model(model_id=dataset_key, copy_data=True)

                # Fallback to legacy keys if not found
                if data is None:
                    if dataset_key == "block_model":
                        data = self.registry.get_block_model(copy_data=True)
                    elif dataset_key == "classified_block_model":
                        data = self.registry.get_classified_block_model(copy_data=True)
                    else:
                        # Estimation result that contains a grid
                        results = self.registry.get_data(dataset_key, copy_data=True)
                        if results and isinstance(results, dict):
                            data = results.get('grid') or results.get('block_model')

            elif self.rb_drillholes.isChecked():
                drillhole_data = self.registry.get_data("drillhole_data", copy_data=True)
                if drillhole_data:
                    data = drillhole_data.get(dataset_key)

            elif self.rb_estimation.isChecked() or self.rb_statistics.isChecked():
                data = self.registry.get_data(dataset_key, copy_data=True)

            if data is None:
                logger.error(f"No data found for {dataset_key}")
                if progress:
                    progress.close()
                return False

            # Check if user cancelled
            if progress.wasCanceled():
                return False

            # Apply downsampling for block models
            downsample_factor = self.downsample_spin.value()
            if self.rb_block_model.isChecked() and downsample_factor > 1:
                progress.setLabelText(f"Downsampling by factor {downsample_factor}...")
                data = self._downsample_block_model(data, downsample_factor)

            # Check if user cancelled
            if progress.wasCanceled():
                return False

            # Update progress
            progress.setLabelText(f"Writing to {Path(filepath).name}...")

            # Export based on format
            result = False
            if "CSV" in format_text:
                result = self._export_csv(data, filepath)
            elif "Excel" in format_text:
                result = self._export_excel(data, filepath)
            elif "Parquet" in format_text:
                result = self._export_parquet(data, filepath)
            elif "VTK" in format_text or "VTU" in format_text:
                result = self._export_vtk(data, filepath)
            elif "Pickle" in format_text:
                result = self._export_pickle(data, filepath)
            else:
                logger.error(f"Unsupported format: {format_text}")
                result = False

            if progress:
                progress.close()

            return result

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            if progress:
                progress.close()
            return False

    def _downsample_block_model(self, data, factor: int):
        """Downsample block model data."""
        try:
            import pyvista as pv

            if hasattr(data, 'n_cells'):  # PyVista grid
                # Downsample by extracting every Nth cell
                indices = list(range(0, data.n_cells, factor))
                return data.extract_cells(indices)
            elif isinstance(data, pd.DataFrame):
                # Downsample DataFrame by taking every Nth row
                return data.iloc[::factor].copy()
            else:
                logger.warning(f"Cannot downsample data type: {type(data)}")
                return data
        except Exception as e:
            logger.error(f"Downsampling failed: {e}")
            return data

    def _export_csv(self, data, filepath: str) -> bool:
        """Export data as CSV."""
        try:
            import pyvista as pv

            if isinstance(data, pd.DataFrame):
                # Filter out any geometry columns that can't be serialized to CSV
                df_to_export = data.copy()

                # Remove any columns that are objects/geometries
                for col in df_to_export.columns:
                    if df_to_export[col].dtype == 'object':
                        # Check if it's a string column or geometry
                        if len(df_to_export[col]) > 0:
                            first_val = df_to_export[col].iloc[0]
                            if first_val is not None and not isinstance(first_val, (str, int, float, bool)):
                                logger.warning(f"Skipping non-serializable column: {col}")
                                df_to_export = df_to_export.drop(columns=[col])

                df_to_export.to_csv(filepath, index=False)
                logger.info(f"Exported {len(df_to_export)} rows to CSV: {filepath}")
                return True
            elif hasattr(data, 'points'):  # PyVista grid
                # Convert to DataFrame
                df = pd.DataFrame(data.points, columns=['X', 'Y', 'Z'])

                # Add cell data
                for array_name in data.array_names:
                    try:
                        array_data = data[array_name]
                        # Only add if it's 1D array (scalar per cell)
                        if len(array_data.shape) == 1:
                            df[array_name] = array_data
                        else:
                            logger.warning(f"Skipping multi-dimensional array: {array_name}")
                    except Exception as e:
                        logger.warning(f"Skipping array {array_name}: {e}")

                df.to_csv(filepath, index=False)
                logger.info(f"Exported {len(df)} blocks to CSV: {filepath}")
                return True
            elif isinstance(data, dict):
                # For dict results, try to extract a DataFrame
                for key in ['dataframe', 'data', 'results']:
                    if key in data and isinstance(data[key], pd.DataFrame):
                        data[key].to_csv(filepath, index=False)
                        logger.info(f"Exported {len(data[key])} rows to CSV: {filepath}")
                        return True
                # If no DataFrame found, save as JSON
                import json
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                return True
            else:
                logger.error(f"Unsupported data type for CSV export: {type(data)}")
                return False
        except Exception as e:
            logger.error(f"CSV export failed: {e}", exc_info=True)
            return False

    def _export_excel(self, data, filepath: str) -> bool:
        """Export data as Excel."""
        try:
            if isinstance(data, pd.DataFrame):
                data.to_excel(filepath, index=False, engine='openpyxl')
                logger.info(f"Exported {len(data)} rows to Excel: {filepath}")
                return True
            elif hasattr(data, 'points'):  # PyVista grid
                # Convert to DataFrame
                df = pd.DataFrame(data.points, columns=['X', 'Y', 'Z'])

                # Add cell data
                for array_name in data.array_names:
                    df[array_name] = data[array_name]

                df.to_excel(filepath, index=False, engine='openpyxl')
                logger.info(f"Exported {len(df)} blocks to Excel: {filepath}")
                return True
            elif isinstance(data, dict):
                # For dict results, create multi-sheet Excel
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    for key, value in data.items():
                        if isinstance(value, pd.DataFrame):
                            value.to_excel(writer, sheet_name=key[:31], index=False)  # Excel sheet name limit
                logger.info(f"Exported multi-sheet Excel: {filepath}")
                return True
            else:
                logger.error(f"Unsupported data type for Excel export: {type(data)}")
                return False
        except Exception as e:
            logger.error(f"Excel export failed: {e}", exc_info=True)
            return False

    def _export_parquet(self, data, filepath: str) -> bool:
        """Export data as Parquet."""
        try:
            if isinstance(data, pd.DataFrame):
                data.to_parquet(filepath, index=False)
                logger.info(f"Exported {len(data)} rows to Parquet: {filepath}")
                return True
            elif hasattr(data, 'points'):  # PyVista grid
                # Convert to DataFrame
                df = pd.DataFrame(data.points, columns=['X', 'Y', 'Z'])

                # Add cell data
                for array_name in data.array_names:
                    df[array_name] = data[array_name]

                df.to_parquet(filepath, index=False)
                logger.info(f"Exported {len(df)} blocks to Parquet: {filepath}")
                return True
            else:
                logger.error(f"Unsupported data type for Parquet export: {type(data)}")
                return False
        except Exception as e:
            logger.error(f"Parquet export failed: {e}", exc_info=True)
            return False

    def _export_vtk(self, data, filepath: str) -> bool:
        """Export data as VTK/VTU."""
        try:
            import pyvista as pv

            if hasattr(data, 'save'):  # PyVista grid
                data.save(filepath)
                logger.info(f"Exported PyVista grid to VTK: {filepath}")
                return True
            else:
                logger.error(f"VTK export requires PyVista grid, got: {type(data)}")
                return False
        except Exception as e:
            logger.error(f"VTK export failed: {e}", exc_info=True)
            return False

    def _export_pickle(self, data, filepath: str) -> bool:
        """Export data as Pickle."""
        try:
            import pickle

            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Exported data to Pickle: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Pickle export failed: {e}", exc_info=True)
            return False
