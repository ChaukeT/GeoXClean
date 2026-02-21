"""
Data Viewer Panel - Displays block model data in a tabular format.
Shows all block properties with search, filter, and export capabilities.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView,
    QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox, QHeaderView,
    QComboBox, QSpinBox, QGroupBox, QFormLayout, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QSortFilterProxyModel
from PyQt6.QtGui import QColor, QCursor

from ..models.block_model import BlockModel
from .block_model_table_model import BlockModelTableModel
from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea

logger = logging.getLogger(__name__)


class DataViewerPanel(BaseDockPanel):
    """Panel for viewing and interacting with block model data in table format."""
    # PanelManager metadata
    PANEL_ID = "DataViewerPanel"
    PANEL_NAME = "DataViewer Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT


    
    # Signals
    block_selected = pyqtSignal(int)  # Emits block index when row is clicked
    
    def __init__(self, parent=None, panel_id=None, host_window=None):
        # Initialize attributes before calling super().__init__() which calls setup_ui()
        # Use _block_model (private backing field) - block_model is a read-only @property in BasePanel
        self._block_model: Optional[BlockModel] = None
        self._host_window = host_window
        
        # Use lazy-loading table model instead of DataFrame
        self.table_model = BlockModelTableModel()
        self._available_models: Dict[str, Any] = {}
        self._suppress_model_combo_signal = False
        # Create proxy_model without parent first (will set parent after super().__init__())
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.table_model)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        # Avoid expensive automatic re-sorting while typing/filtering on large tables.
        self.proxy_model.setDynamicSortFilter(False)
        
        # Now call super().__init__() which will call setup_ui()
        super().__init__(parent=parent, panel_id=panel_id)
        
        # Set parent for proxy_model now that self is fully initialized
        self.proxy_model.setParent(self)
        
        # Subscribe to block model updates from DataRegistry
        self._connect_to_registry()
        
        logger.info("Initialized Data Viewer panel")
    
    def _connect_to_registry(self):
        """Connect to DataRegistry for block model updates."""
        try:
            registry = self.get_registry()
            if registry:
                registry.blockModelGenerated.connect(self._on_block_model_generated)
                registry.blockModelLoaded.connect(self._on_block_model_loaded)
                registry.blockModelClassified.connect(self._on_block_model_classified)

                # Multi-model support - refresh when current model changes
                if hasattr(registry, 'currentBlockModelChanged'):
                    registry.currentBlockModelChanged.connect(self._refresh_available_block_models)

                # Load existing block models from all sources
                self._refresh_available_block_models()
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
    
    def _refresh_available_block_models(self):
        """Refresh the list of available block models from all sources."""
        try:
            registry = self.get_registry()
            if not registry:
                return
            
            # Check regular block model
            block_model = registry.get_block_model()
            if block_model is not None:
                self._register_model_option("Registry Block Model", block_model, select=False)
            
            # Check classified block model
            try:
                classified_model = registry.get_classified_block_model()
                if classified_model is not None:
                    self._register_model_option("Classified Block Model", classified_model, select=False)
            except Exception as e:
                logger.error(f"Failed to register classified block model: {e}", exc_info=True)
            
            # Also check renderer layers for block models that might be displayed
            self._check_renderer_layers_for_block_models()

            # Choose a default model once options are populated.
            if self.model_combo.count() > 1 and self.block_model is None:
                self.model_combo.setCurrentIndex(1)
            
        except Exception as e:
            logger.warning(f"Failed to refresh available block models: {e}")
    
    def _check_renderer_layers_for_block_models(self):
        """Check renderer layers for block models that might be displayed."""
        try:
            # Try to get renderer from parent/main window
            parent = self._host_window if self._host_window is not None else self.parent()
            while parent:
                if hasattr(parent, 'viewer_widget') and hasattr(parent.viewer_widget, 'renderer'):
                    renderer = parent.viewer_widget.renderer
                    if hasattr(renderer, 'active_layers') and renderer.active_layers:
                        # Look for block model layers (SGSIM, kriging, etc.)
                        for layer_name, layer_info in renderer.active_layers.items():
                            layer_type = layer_info.get('type', '')
                            layer_data = layer_info.get('data', None)
                            
                            # Check if it's a block model layer
                            if layer_type in ('blocks', 'volume') and layer_data is not None:
                                if hasattr(parent, '_extract_block_model_from_grid'):
                                    block_model = parent._extract_block_model_from_grid(layer_data, layer_name)
                                    if block_model is not None:
                                        self._register_model_option(f"Layer: {layer_name}", block_model, select=False)
                                        logger.info(f"Found block model in renderer layer: {layer_name}")
                    break
                parent = parent.parent() if parent else None
        except Exception as e:
            logger.debug(f"Could not check renderer layers: {e}")
    
    def _on_block_model_classified(self, block_model):
        """Handle block model classification changes."""
        self._register_model_option("Classified Block Model", block_model, select=True)
    
    def setup_ui(self):
        """Setup the user interface."""
        # BasePanel already creates the root layout in _setup_base_ui().
        # Reuse it to avoid building UI into an unattached second layout.
        layout = self.main_layout if self.main_layout is not None else QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header with info
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("-- Select Block Model --")
        self.model_combo.currentTextChanged.connect(self._on_model_selected)
        self.model_combo.setMinimumWidth(260)
        header_layout.addWidget(self.model_combo)

        self.info_label = QLabel("No block model loaded")
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
        
        # Property filter
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
        
        # Table view with lazy-loading model
        self.table = QTableView()
        self.table.setModel(self.proxy_model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        # Keep sorting opt-in; enabling by default can stall with very large models.
        self.table.setSortingEnabled(False)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.clicked.connect(self._on_cell_clicked)
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
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        logger.info("Data Viewer Panel received generated block model from DataRegistry")
        self._register_model_option("Generated Block Model", block_model, select=True)
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._register_model_option("Registry Block Model", block_model, select=True)

    def _register_model_option(self, name: str, block_model: Any, select: bool = False):
        """Register/update a block model source in the selector."""
        if block_model is None:
            return

        self._available_models[name] = block_model

        existing_idx = self.model_combo.findText(name)
        if existing_idx < 0:
            self._suppress_model_combo_signal = True
            try:
                self.model_combo.addItem(name)
            finally:
                self._suppress_model_combo_signal = False
            existing_idx = self.model_combo.findText(name)

        if select and existing_idx >= 0:
            self.model_combo.setCurrentIndex(existing_idx)

    def _on_model_selected(self, model_name: str):
        """Load selected model from the model selector."""
        if self._suppress_model_combo_signal:
            return
        if not model_name or model_name == "-- Select Block Model --":
            return

        model = self._available_models.get(model_name)
        if model is None:
            return
        self.set_block_model(model)
    
    def set_block_model(self, block_model):
        """Set the block model and update the table model."""
        try:
            logger.info("Setting block model in data viewer...")

            # DataRegistry may provide a pandas DataFrame. Normalize to BlockModel.
            if isinstance(block_model, pd.DataFrame):
                if block_model.empty:
                    raise ValueError("Block model DataFrame is empty")
                converted = BlockModel()
                converted.update_from_dataframe(block_model)
                block_model = converted
            
            if not isinstance(block_model, BlockModel):
                raise TypeError(f"Unsupported block model type: {type(block_model).__name__}")
            
            self._block_model = block_model  # Use private backing field (property contract)
            
            # Set block model in table model (lazy loading - no DataFrame copy)
            self.table_model.set_block_model(block_model)
            
            # Update info label
            num_blocks = block_model.block_count
            num_props = len(block_model.get_property_names())
            self.info_label.setText(f"Block Model: {num_blocks} blocks, {num_props} properties")
            
            # Populate property combo from block model properties
            logger.info("Populating property combo...")
            self.property_combo.clear()
            self.property_combo.addItem("-- Select Property --")
            property_names = block_model.get_property_names()
            for prop_name in sorted(property_names):
                # Check if property is numeric
                prop_values = block_model.get_property(prop_name)
                if prop_values is not None and np.issubdtype(prop_values.dtype, np.number):
                    self.property_combo.addItem(prop_name)
            
            # Full resize-to-contents can freeze on very large models.
            # Use it only for smaller datasets.
            if num_blocks <= 20000:
                self.table.resizeColumnsToContents()
            else:
                header = self.table.horizontalHeader()
                header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
                for i in range(self.table_model.columnCount()):
                    self.table.setColumnWidth(i, 110)
            
            # Update row count label
            self.rows_label.setText(f"Rows: {num_blocks}")
            
            # Enable controls
            self.export_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            
            logger.info(f"Successfully set block model with {num_blocks} blocks in data viewer")
            
        except Exception as e:
            logger.error(f"Error setting block model in data viewer: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load block model data:\n{e}")
    
    def _apply_search(self, text: str):
        """Apply search filter to the table using proxy model."""
        if self.block_model is None:
            return
        
        if text.strip() == "":
            # Clear search - show all rows
            self.proxy_model.setFilterFixedString("")
            self.rows_label.setText(f"Rows: {self.table_model.rowCount()}")
        else:
            # Apply search filter (searches across all columns)
            self.proxy_model.setFilterFixedString(text)
            visible_count = self.proxy_model.rowCount()
            total_count = self.table_model.rowCount()
            self.rows_label.setText(f"Rows: {visible_count} of {total_count}")
    
    def _clear_search(self):
        """Clear the search filter."""
        self.search_input.clear()
        self.proxy_model.setFilterFixedString("")
        if self.block_model is not None:
            self.rows_label.setText(f"Rows: {self.table_model.rowCount()}")
    
    def _on_property_selected(self, property_name: str):
        """Handle property selection for filtering."""
        if not property_name or property_name == "-- Select Property --" or self.block_model is None:
            self.min_value_spin.setEnabled(False)
            self.max_value_spin.setEnabled(False)
            self.apply_filter_btn.setEnabled(False)
            self.clear_filter_btn.setEnabled(False)
            return
        
        # Verify property exists in block model
        if property_name not in self.block_model.get_property_names():
            logger.warning(f"Property '{property_name}' not found in block model")
            self.min_value_spin.setEnabled(False)
            self.max_value_spin.setEnabled(False)
            self.apply_filter_btn.setEnabled(False)
            self.clear_filter_btn.setEnabled(False)
            return
        
        try:
            # Get property values to determine min/max
            prop_values = self.block_model.get_property(property_name)
            if prop_values is None:
                raise ValueError(f"Property {property_name} has no values")
            
            # Filter out NaN values for range calculation
            valid_values = prop_values[~np.isnan(prop_values)]
            if len(valid_values) == 0:
                raise ValueError(f"Property {property_name} has no valid values")
            
            min_val = int(np.min(valid_values))
            max_val = int(np.max(valid_values))
        
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
        """Apply property range filter."""
        if self.block_model is None:
            return
        
        property_name = self.property_combo.currentText()
        if property_name == "-- Select Property --":
            return
        
        try:
            min_val = self.min_value_spin.value()
            max_val = self.max_value_spin.value()
            
            # Get property values and create filter mask
            prop_values = self.block_model.get_property(property_name)
            if prop_values is None:
                raise ValueError(f"Property {property_name} not found")
            
            mask = (prop_values >= min_val) & (prop_values <= max_val)
            # Also include NaN values (user might want to see them)
            mask = mask | np.isnan(prop_values)
            
            # Apply filter to table model
            self.table_model.apply_filter(mask)
            
            visible_count = self.table_model.rowCount()
            total_count = self.block_model.block_count
            self.rows_label.setText(f"Rows: {visible_count} of {total_count}")
            
            logger.info(f"Applied filter: {property_name} in [{min_val}, {max_val}]")
            
        except Exception as e:
            logger.error(f"Error applying property filter: {e}")
            QMessageBox.warning(self, "Error", f"Failed to apply filter:\n{e}")
    
    def _clear_property_filter(self):
        """Clear the property filter."""
        self.property_combo.setCurrentIndex(0)
        self.table_model.clear_filter()
        if self.block_model is not None:
            self.rows_label.setText(f"Rows: {self.block_model.block_count}")
    
    def _refresh_table(self):
        """Refresh the table with current block model."""
        if self.block_model is not None:
            self.set_block_model(self.block_model)
    
    def _export_to_csv(self):
        """Export the currently displayed data to CSV."""
        if self.block_model is None:
            QMessageBox.warning(self, "No Data", "No data available to export.")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Block Model Data",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                # Create DataFrame only when exporting (lazy loading)
                # Use the table model's to_dataframe method which respects filters
                export_df = self.table_model.to_dataframe()
                
                # Step 12: Use ExportHelpers
                from ..utils.export_helpers import export_dataframe_to_csv
                export_dataframe_to_csv(export_df, file_path)
                logger.info(f"Exported {len(export_df)} rows to {file_path}")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Exported {len(export_df)} rows to:\n{file_path}"
                )
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{e}")
    
    def _on_cell_clicked(self, index):
        """Handle cell click - emit block selected signal."""
        try:
            # Get the source model index (accounting for proxy model)
            source_index = self.proxy_model.mapToSource(index)
            if source_index.isValid():
                block_idx = self.table_model.get_block_index_for_row(source_index.row())
                if block_idx is not None:
                    self.block_selected.emit(block_idx)
                    logger.debug(f"Selected block index: {block_idx}")
        except Exception as e:
            logger.error(f"Error handling cell click: {e}")
    
    def clear(self):
        """Clear the data viewer."""
        self._block_model = None  # Use _block_model instead of block_model property
        self.table_model.set_block_model(None)
        self.proxy_model.setFilterFixedString("")
        self.info_label.setText("No block model loaded")
        self.rows_label.setText("Rows: 0")
        self.property_combo.clear()
        self.property_combo.addItem("-- Select Property --")
        self.search_input.clear()
        self.export_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        logger.info("Cleared data viewer")

    def clear_panel(self):
        """Clear all panel UI and state to initial defaults."""
        self.clear()
        super().clear_panel()
        logger.info("DataViewerPanel: Panel fully cleared")

