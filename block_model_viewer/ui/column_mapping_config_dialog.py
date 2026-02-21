"""
Dialog for configuring block model column mapping.

Allows users to specify which columns in their data correspond to
standard fields like coordinates, volume, density, etc.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QLabel, QPushButton, QGroupBox,
    QMessageBox, QCheckBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

from ..models.pit_optimizer import ColumnMapping


class ColumnMappingConfigDialog(QDialog):
    """
    Dialog for configuring block model column mappings.
    
    Allows users to map their custom column names to standard fields
    required for pit optimization and other analyses.
    """
    
    def __init__(self, df: pd.DataFrame, current_mapping: Optional[ColumnMapping] = None, parent=None):
        """
        Initialize the column mapping dialog.
        
        Parameters:
        -----------
        df : DataFrame
            Block model data with available columns
        current_mapping : ColumnMapping, optional
            Existing column mapping to edit
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.df = df
        self.current_mapping = current_mapping or ColumnMapping()
        self.column_mapping = None  # Will be set when accepted
        self.constant_densities = {}  # Dict mapping domain/lithology values to density
        
        self.setWindowTitle("Configure Block Model Columns")
        self.setModal(True)
        self.resize(700, 650)
        
        self._init_ui()
        self._auto_detect_columns()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "Map your block model columns to standard fields.\n"
            "The optimizer will use these mappings to locate required data."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Required fields group
        required_group = QGroupBox("Required Fields")
        required_layout = QFormLayout()
        
        self.x_combo = self._create_column_combo("X Coordinate")
        self.y_combo = self._create_column_combo("Y Coordinate")
        self.z_combo = self._create_column_combo("Z Coordinate")
        self.volume_combo = self._create_column_combo("Block Volume")
        
        required_layout.addRow("X Coordinate:", self.x_combo)
        required_layout.addRow("Y Coordinate:", self.y_combo)
        required_layout.addRow("Z Coordinate:", self.z_combo)
        required_layout.addRow("Block Volume:", self.volume_combo)
        
        required_group.setLayout(required_layout)
        layout.addWidget(required_group)
        
        # Density configuration group
        density_group = QGroupBox("Density Configuration")
        density_layout = QVBoxLayout()
        
        # Radio buttons for density source
        self.density_button_group = QButtonGroup()
        self.density_from_column_radio = QRadioButton("Use density from column:")
        self.density_constant_radio = QRadioButton("Use constant density value(s)")
        self.density_button_group.addButton(self.density_from_column_radio, 0)
        self.density_button_group.addButton(self.density_constant_radio, 1)
        self.density_from_column_radio.setChecked(True)
        
        density_layout.addWidget(self.density_from_column_radio)
        
        # Density column selector
        density_col_layout = QHBoxLayout()
        density_col_layout.addSpacing(20)
        self.density_combo = self._create_column_combo("Density/SG", allow_none=True)
        density_col_layout.addWidget(self.density_combo)
        density_layout.addLayout(density_col_layout)
        
        density_layout.addWidget(self.density_constant_radio)
        
        # Constant density options
        constant_density_layout = QVBoxLayout()
        constant_density_layout.setContentsMargins(20, 0, 0, 0)
        
        # Single constant density
        single_density_layout = QHBoxLayout()
        self.single_density_radio = QRadioButton("Single density for all blocks:")
        self.single_density_spin = QDoubleSpinBox()
        self.single_density_spin.setRange(0.1, 20.0)
        self.single_density_spin.setDecimals(2)
        self.single_density_spin.setValue(2.70)
        self.single_density_spin.setSuffix(" t/m³")
        self.single_density_spin.setEnabled(False)
        single_density_layout.addWidget(self.single_density_radio)
        single_density_layout.addWidget(self.single_density_spin)
        single_density_layout.addStretch()
        constant_density_layout.addLayout(single_density_layout)
        
        # Multiple densities by domain/lithology
        multi_density_layout = QVBoxLayout()
        self.multi_density_radio = QRadioButton("Different densities by domain/lithology:")
        multi_density_layout.addWidget(self.multi_density_radio)
        
        domain_selector_layout = QHBoxLayout()
        domain_selector_layout.addSpacing(20)
        domain_selector_layout.addWidget(QLabel("Zone column:"))
        self.zone_column_combo = self._create_column_combo("Column for zones/domains/lithology", allow_none=True)
        self.zone_column_combo.setEnabled(False)
        domain_selector_layout.addWidget(self.zone_column_combo)
        domain_selector_layout.addStretch()
        multi_density_layout.addLayout(domain_selector_layout)
        
        # Table for domain-specific densities
        table_layout = QHBoxLayout()
        table_layout.addSpacing(20)
        self.density_table = QTableWidget()
        self.density_table.setColumnCount(2)
        self.density_table.setHorizontalHeaderLabels(["Zone/Domain/Lithology", "Density (t/m³)"])
        self.density_table.horizontalHeader().setStretchLastSection(True)
        self.density_table.setMaximumHeight(150)
        self.density_table.setEnabled(False)
        table_layout.addWidget(self.density_table)
        multi_density_layout.addLayout(table_layout)
        
        btn_layout = QHBoxLayout()
        btn_layout.addSpacing(20)
        self.btn_populate_zones = QPushButton("Auto-populate from data")
        self.btn_populate_zones.setEnabled(False)
        self.btn_populate_zones.clicked.connect(self._populate_density_table)
        btn_layout.addWidget(self.btn_populate_zones)
        btn_layout.addStretch()
        multi_density_layout.addLayout(btn_layout)
        
        constant_density_layout.addLayout(multi_density_layout)
        density_layout.addLayout(constant_density_layout)
        
        density_group.setLayout(density_layout)
        layout.addWidget(density_group)
        
        # Connect signals for density mode switching
        self.density_from_column_radio.toggled.connect(self._on_density_mode_changed)
        self.density_constant_radio.toggled.connect(self._on_density_mode_changed)
        self.single_density_radio.toggled.connect(self._on_constant_density_type_changed)
        self.multi_density_radio.toggled.connect(self._on_constant_density_type_changed)
        self.zone_column_combo.currentIndexChanged.connect(self._on_zone_column_changed)
        
        # Group for constant density type
        self.constant_density_button_group = QButtonGroup()
        self.constant_density_button_group.addButton(self.single_density_radio, 0)
        self.constant_density_button_group.addButton(self.multi_density_radio, 1)
        self.single_density_radio.setChecked(True)
        
        # Optional fields group
        optional_group = QGroupBox("Optional Fields")
        optional_layout = QFormLayout()
        
        self.tonnes_combo = self._create_column_combo("Tonnes (calculated if not provided)", allow_none=True)
        self.block_id_combo = self._create_column_combo("Block ID (auto-generated if not provided)", allow_none=True)
        self.domain_combo = self._create_column_combo("Domain/Zone", allow_none=True)
        self.rock_type_combo = self._create_column_combo("Rock Type", allow_none=True)
        
        optional_layout.addRow("Tonnes:", self.tonnes_combo)
        optional_layout.addRow("Block ID:", self.block_id_combo)
        optional_layout.addRow("Domain:", self.domain_combo)
        optional_layout.addRow("Rock Type:", self.rock_type_combo)
        
        optional_group.setLayout(optional_layout)
        layout.addWidget(optional_group)
        
        # Auto-detect checkbox
        self.auto_detect_check = QCheckBox("Auto-detect column mappings")
        self.auto_detect_check.setChecked(True)
        self.auto_detect_check.stateChanged.connect(self._on_auto_detect_changed)
        layout.addWidget(self.auto_detect_check)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(self.btn_reset)
        
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_ok.setDefault(True)
        button_layout.addWidget(self.btn_ok)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _create_column_combo(self, tooltip: str, allow_none: bool = False) -> QComboBox:
        """Create a combobox for column selection."""
        combo = QComboBox()
        combo.setToolTip(tooltip)
        
        if allow_none:
            combo.addItem("<None - Auto-calculate>", None)
        
        for col in self.df.columns:
            combo.addItem(col, col)
        
        return combo
    
    def _auto_detect_columns(self):
        """Auto-detect common column naming patterns."""
        if not self.auto_detect_check.isChecked():
            return
        
        columns_lower = {col: col.lower().strip() for col in self.df.columns}
        
        # X coordinate patterns
        x_patterns = ['x', 'xc', 'x_centre', 'x_center', 'xcenter', 'xcoord', 'easting', 'east']
        for col, col_lower in columns_lower.items():
            if col_lower in x_patterns or 'east' in col_lower or col_lower.startswith('x'):
                self._set_combo_by_value(self.x_combo, col)
                break
        
        # Y coordinate patterns
        y_patterns = ['y', 'yc', 'y_centre', 'y_center', 'ycenter', 'ycoord', 'northing', 'north']
        for col, col_lower in columns_lower.items():
            if col_lower in y_patterns or 'north' in col_lower or col_lower.startswith('y'):
                self._set_combo_by_value(self.y_combo, col)
                break
        
        # Z coordinate patterns
        z_patterns = ['z', 'zc', 'z_centre', 'z_center', 'zcenter', 'zcoord', 'zmid', 'rl', 'elevation', 'elev', 'centroid']
        for col, col_lower in columns_lower.items():
            if col_lower in z_patterns or 'elev' in col_lower or 'rl' in col_lower or col_lower.startswith('z') or 'centroid' in col_lower:
                self._set_combo_by_value(self.z_combo, col)
                break
        
        # Volume patterns
        vol_patterns = ['volume', 'vol', 'block_volume', 'block_vol', 'vol_m3', 'volume_m3']
        for col, col_lower in columns_lower.items():
            if col_lower in vol_patterns or 'vol' in col_lower:
                self._set_combo_by_value(self.volume_combo, col)
                break
        
        # Density patterns
        dens_patterns = ['density', 'dens', 'sg', 'specific_gravity', 'spec_grav', 'densit', 'rho']
        for col, col_lower in columns_lower.items():
            if col_lower in dens_patterns or 'dens' in col_lower or col_lower == 'sg':
                self._set_combo_by_value(self.density_combo, col)
                break
        
        # Optional: Tonnes
        tonnes_patterns = ['tonnes', 'tons', 'tonnage', 'mass', 'tonne', 'ton']
        for col, col_lower in columns_lower.items():
            if col_lower in tonnes_patterns or 'ton' in col_lower:
                self._set_combo_by_value(self.tonnes_combo, col)
                break
        
        # Optional: Block ID
        id_patterns = ['block_id', 'blockid', 'blk_id', 'id', 'bid', 'block_no']
        for col, col_lower in columns_lower.items():
            if col_lower in id_patterns or 'block' in col_lower and 'id' in col_lower:
                self._set_combo_by_value(self.block_id_combo, col)
                break
        
        # Optional: Domain
        domain_patterns = ['domain', 'zone', 'zone_code', 'dom', 'zone_id']
        for col, col_lower in columns_lower.items():
            if col_lower in domain_patterns or 'domain' in col_lower or 'zone' in col_lower:
                self._set_combo_by_value(self.domain_combo, col)
                break
        
        # Optional: Rock type
        rock_patterns = ['rock_type', 'rocktype', 'rock', 'lithology', 'lith', 'rock_code']
        for col, col_lower in columns_lower.items():
            if col_lower in rock_patterns or 'rock' in col_lower or 'lith' in col_lower:
                self._set_combo_by_value(self.rock_type_combo, col)
                break
    
    def _set_combo_by_value(self, combo: QComboBox, value: str):
        """Set combobox to a specific value."""
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                break
    
    def _on_auto_detect_changed(self, state):
        """Handle auto-detect checkbox state change."""
        if state == Qt.CheckState.Checked.value:
            self._auto_detect_columns()
    
    def _on_density_mode_changed(self):
        """Handle density mode radio button changes."""
        use_column = self.density_from_column_radio.isChecked()
        
        # Enable/disable density column selector
        self.density_combo.setEnabled(use_column)
        
        # Enable/disable constant density options
        self.single_density_radio.setEnabled(not use_column)
        self.multi_density_radio.setEnabled(not use_column)
        
        if not use_column:
            self._on_constant_density_type_changed()
    
    def _on_constant_density_type_changed(self):
        """Handle constant density type changes."""
        if not self.density_constant_radio.isChecked():
            return
        
        use_single = self.single_density_radio.isChecked()
        
        # Enable/disable single density spin
        self.single_density_spin.setEnabled(use_single)
        
        # Enable/disable multi-density options
        self.zone_column_combo.setEnabled(not use_single)
        self.density_table.setEnabled(not use_single)
        self.btn_populate_zones.setEnabled(not use_single)
    
    def _on_zone_column_changed(self):
        """Handle zone column selection change."""
        if self.multi_density_radio.isChecked():
            # Clear the table when zone column changes
            self.density_table.setRowCount(0)
            self.constant_densities.clear()
    
    def _populate_density_table(self):
        """Populate density table with unique values from zone column."""
        zone_col = self.zone_column_combo.currentData()
        
        if zone_col is None or zone_col not in self.df.columns:
            QMessageBox.warning(
                self,
                "No Zone Column",
                "Please select a zone/domain/lithology column first."
            )
            return
        
        # Get unique values from zone column
        unique_zones = sorted(self.df[zone_col].dropna().unique())
        
        if len(unique_zones) == 0:
            QMessageBox.warning(
                self,
                "No Data",
                f"Column '{zone_col}' has no data."
            )
            return
        
        if len(unique_zones) > 50:
            reply = QMessageBox.question(
                self,
                "Many Zones",
                f"Found {len(unique_zones)} unique zones. This is a lot!\n\n"
                f"Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Populate table
        self.density_table.setRowCount(len(unique_zones))
        
        for i, zone in enumerate(unique_zones):
            # Zone name
            zone_item = QTableWidgetItem(str(zone))
            zone_item.setFlags(zone_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.density_table.setItem(i, 0, zone_item)
            
            # Density value - use existing value if available, otherwise default
            default_density = self.constant_densities.get(str(zone), 2.70)
            density_item = QTableWidgetItem(f"{default_density:.2f}")
            self.density_table.setItem(i, 1, density_item)
        
        QMessageBox.information(
            self,
            "Zones Loaded",
            f"Loaded {len(unique_zones)} zones from column '{zone_col}'.\n\n"
            f"Please enter the density for each zone in the table."
        )
    
    def _reset_to_defaults(self):
        """Reset all combos to default (first item)."""
        for combo in [self.x_combo, self.y_combo, self.z_combo, 
                      self.volume_combo, self.density_combo]:
            combo.setCurrentIndex(0)
        
        for combo in [self.tonnes_combo, self.block_id_combo,
                      self.domain_combo, self.rock_type_combo]:
            combo.setCurrentIndex(0)  # <None> for optional fields
        
        if self.auto_detect_check.isChecked():
            self._auto_detect_columns()
    
    def accept(self):
        """Validate and accept the dialog."""
        # Get selected values
        x_col = self.x_combo.currentData()
        y_col = self.y_combo.currentData()
        z_col = self.z_combo.currentData()
        volume_col = self.volume_combo.currentData()
        
        # Validate required coordinates and volume
        if not all([x_col, y_col, z_col, volume_col]):
            QMessageBox.warning(
                self,
                "Missing Required Fields",
                "Please select values for all required fields:\n"
                "- X Coordinate\n"
                "- Y Coordinate\n"
                "- Z Coordinate\n"
                "- Block Volume"
            )
            return
        
        # Handle density configuration
        density_col = None
        use_constant_density = self.density_constant_radio.isChecked()
        
        if use_constant_density:
            # Validate constant density configuration
            if self.single_density_radio.isChecked():
                # Single constant density
                single_density = self.single_density_spin.value()
                self.constant_densities = {'__DEFAULT__': single_density}
            else:
                # Multiple densities by zone
                zone_col = self.zone_column_combo.currentData()
                if zone_col is None:
                    QMessageBox.warning(
                        self,
                        "Missing Zone Column",
                        "Please select a zone/domain/lithology column for density mapping."
                    )
                    return
                
                # Read densities from table
                if self.density_table.rowCount() == 0:
                    QMessageBox.warning(
                        self,
                        "No Density Values",
                        "Please populate the density table by clicking 'Auto-populate from data'."
                    )
                    return
                
                self.constant_densities = {}
                for row in range(self.density_table.rowCount()):
                    zone_item = self.density_table.item(row, 0)
                    density_item = self.density_table.item(row, 1)
                    
                    if zone_item and density_item:
                        try:
                            zone = zone_item.text()
                            density = float(density_item.text())
                            if density <= 0:
                                raise ValueError("Density must be positive")
                            self.constant_densities[zone] = density
                        except ValueError as e:
                            QMessageBox.warning(
                                self,
                                "Invalid Density",
                                f"Invalid density value for zone '{zone}': {density_item.text()}\n\n"
                                f"Please enter a positive number."
                            )
                            return
                
                # Store zone column for later use
                self.constant_densities['__ZONE_COLUMN__'] = zone_col
        else:
            # Using density column
            density_col = self.density_combo.currentData()
            if not density_col:
                QMessageBox.warning(
                    self,
                    "Missing Density",
                    "Please either:\n"
                    "1. Select a density column, OR\n"
                    "2. Choose 'Use constant density value(s)' and specify density"
                )
                return
        
        # Check for duplicates in coordinate fields
        required_cols = [x_col, y_col, z_col, volume_col]
        if density_col:
            required_cols.append(density_col)
        
        non_none_cols = [c for c in required_cols if c is not None]
        if len(non_none_cols) != len(set(non_none_cols)):
            QMessageBox.warning(
                self,
                "Duplicate Columns",
                "Each required field must map to a different column."
            )
            return
        
        # Create column mapping
        self.column_mapping = ColumnMapping(
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            volume_col=volume_col,
            density_col=density_col if density_col else 'density',  # Use dummy name if constant
            tonnes_col=self.tonnes_combo.currentData(),
            block_id_col=self.block_id_combo.currentData(),
            domain_col=self.domain_combo.currentData(),
            rock_type_col=self.rock_type_combo.currentData()
        )
        
        super().accept()
    
    def get_column_mapping(self) -> Optional[ColumnMapping]:
        """Get the configured column mapping."""
        return self.column_mapping
    
    def get_constant_densities(self) -> Dict[str, float]:
        """Get the constant density values configured by user."""
        return self.constant_densities


def show_column_mapping_dialog(df: pd.DataFrame, 
                                current_mapping: Optional[ColumnMapping] = None,
                                parent=None) -> tuple[Optional[ColumnMapping], Dict[str, float]]:
    """
    Show column mapping configuration dialog.
    
    Parameters:
    -----------
    df : DataFrame
        Block model data
    current_mapping : ColumnMapping, optional
        Current mapping to edit
    parent : QWidget, optional
        Parent widget
    
    Returns:
    --------
    tuple: (ColumnMapping or None, constant_densities dict)
        Returns column mapping and constant density configuration.
        If cancelled, returns (None, {})
    """
    dialog = ColumnMappingConfigDialog(df, current_mapping, parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_column_mapping(), dialog.get_constant_densities()
    return None, {}
