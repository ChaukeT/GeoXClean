"""
Fault Definition Panel
======================

UI panel for defining faults in geological models.

Features:
- Manual fault definition (point + dip/azimuth + throw)
- Import fault surfaces from files (DXF, OBJ, VTK)
- Fault relationship editor
- 3D preview of fault planes
- Integration with GemPy and geological wizard

This panel is designed to be used as a dockable panel in the main window
or embedded within the geological modeling wizard.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtGui import QColor, QDoubleValidator, QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyleOptionSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


# =============================================================================
# FAULT LIST ITEM WIDGET
# =============================================================================

class FaultListItem(QFrame):
    """Custom widget for displaying a fault in the list."""
    
    edited = pyqtSignal(str)  # Fault name
    deleted = pyqtSignal(str)  # Fault name
    toggled = pyqtSignal(str, bool)  # Fault name, active state
    
    def __init__(self, fault_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self._fault_data = fault_data
        self._setup_ui()
    


    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()
    def _setup_ui(self):
        """Build the item UI."""
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            FaultListItem {
                background: #2d2d30;
                border: 1px solid #3d3d40;
                border-radius: 4px;
                margin: 2px;
            }
            FaultListItem:hover {
                border-color: #0078d4;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Active checkbox
        self.active_check = QCheckBox()
        self.active_check.setChecked(self._fault_data.get('active', True))
        self.active_check.setToolTip("Toggle fault active state")
        self.active_check.stateChanged.connect(self._on_toggled)
        layout.addWidget(self.active_check)
        
        # Color indicator
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(20, 20)
        color = self._fault_data.get('metadata', {}).get('color', '#ff5555')
        self.color_btn.setStyleSheet(f"background-color: {color}; border: 1px solid #555; border-radius: 2px;")
        self.color_btn.setToolTip("Click to change color")
        self.color_btn.clicked.connect(self._pick_color)
        layout.addWidget(self.color_btn)
        
        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(0)
        
        name = self._fault_data.get('name', 'Unnamed')
        fault_type = self._fault_data.get('fault_type', 'unknown')
        
        name_lbl = QLabel(f"<b>{name}</b>")
        name_lbl.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY};")
        info_layout.addWidget(name_lbl)
        
        # Details line
        dip = self._fault_data.get('dip', 0)
        azimuth = self._fault_data.get('azimuth', 0)
        throw = self._fault_data.get('throw_magnitude', 0)
        
        details = f"{fault_type.title()} | Dip: {dip:.0f}° @ {azimuth:.0f}° | Throw: {throw:.1f}m"
        details_lbl = QLabel(details)
        details_lbl.setStyleSheet("color: #888; font-size: 10px;")
        info_layout.addWidget(details_lbl)
        
        layout.addLayout(info_layout, 1)
        
        # Edit button
        edit_btn = QToolButton()
        edit_btn.setText("✎")
        edit_btn.setToolTip("Edit fault")
        edit_btn.clicked.connect(lambda: self.edited.emit(self._fault_data.get('name', '')))
        layout.addWidget(edit_btn)
        
        # Delete button
        del_btn = QToolButton()
        del_btn.setText("✕")
        del_btn.setToolTip("Delete fault")
        del_btn.setStyleSheet("color: #ff5555;")
        del_btn.clicked.connect(lambda: self.deleted.emit(self._fault_data.get('name', '')))
        layout.addWidget(del_btn)
    
    def _on_toggled(self, state):
        """Handle active state toggle."""
        self.toggled.emit(self._fault_data.get('name', ''), state == Qt.CheckState.Checked.value)
    
    def _pick_color(self):
        """Open color picker."""
        current = self._fault_data.get('metadata', {}).get('color', '#ff5555')
        color = QColorDialog.getColor(QColor(current), self, "Select Fault Color")
        if color.isValid():
            hex_color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555; border-radius: 2px;")
            if 'metadata' not in self._fault_data:
                self._fault_data['metadata'] = {}
            self._fault_data['metadata']['color'] = hex_color
    
    @property
    def fault_data(self) -> Dict[str, Any]:
        """Get the current fault data."""
        return self._fault_data


# =============================================================================
# FAULT EDITOR DIALOG
# =============================================================================

class FaultEditorDialog(QDialog):
    """Dialog for editing a fault definition."""
    
    def __init__(self, fault_data: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self._fault_data = fault_data or {}
        self._is_new = fault_data is None
        self.setWindowTitle("New Fault" if self._is_new else f"Edit Fault: {fault_data.get('name', '')}")
        self.setMinimumWidth(450)
        self._setup_ui()
        self._load_data()
    
    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Tabs for different input methods
        tabs = QTabWidget()
        tabs.addTab(self._create_manual_tab(), "Manual Definition")
        tabs.addTab(self._create_import_tab(), "Import Surface")
        layout.addWidget(tabs)
        
        # Button box
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _create_manual_tab(self) -> QWidget:
        """Create manual fault definition tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Basic info group
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Main_Fault_1")
        basic_layout.addRow("Fault Name:", self.name_edit)
        
        self.fault_type_combo = QComboBox()
        self.fault_type_combo.addItems([
            "normal", "reverse", "strike_slip", "oblique", "listric", "thrust", "unknown"
        ])
        basic_layout.addRow("Fault Type:", self.fault_type_combo)
        
        layout.addWidget(basic_group)
        
        # Geometry group
        geom_group = QGroupBox("Fault Plane Geometry")
        geom_layout = QFormLayout(geom_group)
        
        # Point on fault
        point_layout = QHBoxLayout()
        self.point_x = QDoubleSpinBox()
        self.point_x.setRange(-1e9, 1e9)
        self.point_x.setDecimals(2)
        self.point_y = QDoubleSpinBox()
        self.point_y.setRange(-1e9, 1e9)
        self.point_y.setDecimals(2)
        self.point_z = QDoubleSpinBox()
        self.point_z.setRange(-1e9, 1e9)
        self.point_z.setDecimals(2)
        point_layout.addWidget(QLabel("X:"))
        point_layout.addWidget(self.point_x)
        point_layout.addWidget(QLabel("Y:"))
        point_layout.addWidget(self.point_y)
        point_layout.addWidget(QLabel("Z:"))
        point_layout.addWidget(self.point_z)
        geom_layout.addRow("Point on Fault:", point_layout)
        
        # Dip
        self.dip_spin = QDoubleSpinBox()
        self.dip_spin.setRange(0, 90)
        self.dip_spin.setDecimals(1)
        self.dip_spin.setSuffix("°")
        self.dip_spin.setValue(60)
        geom_layout.addRow("Dip Angle:", self.dip_spin)
        
        # Azimuth (dip direction)
        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(0, 360)
        self.azimuth_spin.setDecimals(1)
        self.azimuth_spin.setSuffix("°")
        self.azimuth_spin.setValue(90)
        geom_layout.addRow("Dip Direction:", self.azimuth_spin)
        
        layout.addWidget(geom_group)
        
        # Displacement group
        disp_group = QGroupBox("Displacement")
        disp_layout = QFormLayout(disp_group)
        
        self.throw_spin = QDoubleSpinBox()
        self.throw_spin.setRange(0, 10000)
        self.throw_spin.setDecimals(1)
        self.throw_spin.setSuffix(" m")
        self.throw_spin.setValue(10)
        disp_layout.addRow("Throw Magnitude:", self.throw_spin)
        
        self.throw_dir_combo = QComboBox()
        self.throw_dir_combo.addItems(["normal", "reverse", "dextral", "sinistral"])
        disp_layout.addRow("Throw Direction:", self.throw_dir_combo)
        
        self.influence_spin = QDoubleSpinBox()
        self.influence_spin.setRange(0, 1000)
        self.influence_spin.setDecimals(1)
        self.influence_spin.setSuffix(" m")
        self.influence_spin.setValue(50)
        disp_layout.addRow("Influence Distance:", self.influence_spin)
        
        layout.addWidget(disp_group)
        
        # Active checkbox
        self.active_check = QCheckBox("Fault is active in modeling")
        self.active_check.setChecked(True)
        layout.addWidget(self.active_check)
        
        layout.addStretch()
        return widget
    
    def _create_import_tab(self) -> QWidget:
        """Create surface import tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel(
            "Import a fault surface mesh from a file.\n"
            "Supported formats: OBJ, VTK, DXF, STL"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888;")
        layout.addWidget(info_label)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("Select a mesh file...")
        file_layout.addWidget(self.file_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(browse_btn)
        
        layout.addLayout(file_layout)
        
        # Import settings
        settings_group = QGroupBox("Import Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.import_name_edit = QLineEdit()
        self.import_name_edit.setPlaceholderText("Auto-detect from filename")
        settings_layout.addRow("Fault Name:", self.import_name_edit)
        
        self.import_throw_spin = QDoubleSpinBox()
        self.import_throw_spin.setRange(0, 10000)
        self.import_throw_spin.setDecimals(1)
        self.import_throw_spin.setSuffix(" m")
        self.import_throw_spin.setValue(10)
        settings_layout.addRow("Default Throw:", self.import_throw_spin)
        
        layout.addWidget(settings_group)
        
        # Preview area
        preview_group = QGroupBox("File Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(100)
        self.preview_text.setPlaceholderText("File information will appear here after selection...")
        preview_layout.addWidget(self.preview_text)
        layout.addWidget(preview_group)
        
        layout.addStretch()
        return widget
    
    def _browse_file(self):
        """Browse for a mesh file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Fault Surface Mesh",
            "",
            "Mesh Files (*.obj *.vtk *.dxf *.stl);;OBJ Files (*.obj);;VTK Files (*.vtk);;All Files (*)"
        )
        
        if file_path:
            self.file_edit.setText(file_path)
            self._preview_file(file_path)
    
    def _preview_file(self, file_path: str):
        """Preview the selected file."""
        try:
            path = Path(file_path)
            ext = path.suffix.lower()
            size = path.stat().st_size
            
            info = f"File: {path.name}\n"
            info += f"Size: {size / 1024:.1f} KB\n"
            info += f"Format: {ext}\n"
            
            # Try to read basic stats
            if ext == '.obj':
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    vertices = sum(1 for l in lines if l.startswith('v '))
                    faces = sum(1 for l in lines if l.startswith('f '))
                    info += f"Vertices: {vertices}\n"
                    info += f"Faces: {faces}"
            
            self.preview_text.setText(info)
            
            # Auto-fill name from filename
            if not self.import_name_edit.text():
                self.import_name_edit.setText(path.stem)
                
        except Exception as e:
            self.preview_text.setText(f"Error reading file: {e}")
    
    def _load_data(self):
        """Load existing fault data into the form."""
        if not self._fault_data:
            return
        
        self.name_edit.setText(self._fault_data.get('name', ''))
        
        fault_type = self._fault_data.get('fault_type', 'unknown')
        idx = self.fault_type_combo.findText(fault_type)
        if idx >= 0:
            self.fault_type_combo.setCurrentIndex(idx)
        
        point = self._fault_data.get('point', [0, 0, 0])
        if isinstance(point, (list, np.ndarray)) and len(point) >= 3:
            self.point_x.setValue(float(point[0]))
            self.point_y.setValue(float(point[1]))
            self.point_z.setValue(float(point[2]))
        
        self.dip_spin.setValue(float(self._fault_data.get('dip', 60)))
        self.azimuth_spin.setValue(float(self._fault_data.get('azimuth', 90)))
        self.throw_spin.setValue(float(self._fault_data.get('throw_magnitude', 10)))
        self.influence_spin.setValue(float(self._fault_data.get('influence', 50)))
        self.active_check.setChecked(self._fault_data.get('active', True))
    
    def get_fault_data(self) -> Dict[str, Any]:
        """Get the fault data from the form."""
        try:
            from ..geology.faults import FaultPlane
        except ImportError:
            raise ImportError("Geology module removed - fault functionality unavailable")
        
        name = self.name_edit.text().strip() or f"Fault_{id(self)}"
        
        # Create FaultPlane using dip/azimuth constructor
        fault = FaultPlane.from_dip_azimuth(
            name=name,
            point=np.array([self.point_x.value(), self.point_y.value(), self.point_z.value()]),
            dip=self.dip_spin.value(),
            azimuth=self.azimuth_spin.value(),
            throw_magnitude=self.throw_spin.value(),
            throw_direction=self.throw_dir_combo.currentText(),
            influence=self.influence_spin.value(),
        )
        fault.active = self.active_check.isChecked()
        
        # Preserve metadata like color
        if 'metadata' in self._fault_data:
            fault.metadata = dict(self._fault_data['metadata'])
        
        return fault.to_dict()


# =============================================================================
# FAULT RELATIONSHIPS DIALOG
# =============================================================================

class FaultRelationshipsDialog(QDialog):
    """Dialog for editing fault cutting relationships."""
    
    def __init__(self, faults: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self._faults = faults
        self.setWindowTitle("Fault Relationships")
        self.setMinimumSize(500, 400)
        self._relationships: Dict[Tuple[str, str], str] = {}
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        
        info_label = QLabel(
            "Define cutting relationships between faults.\n"
            "Younger faults cut older faults."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Relationship table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Fault 1", "Relationship", "Fault 2"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        # Add relationship button
        add_layout = QHBoxLayout()
        
        self.fault1_combo = QComboBox()
        self.fault1_combo.addItems([f.get('name', '') for f in self._faults])
        add_layout.addWidget(self.fault1_combo)
        
        self.relation_combo = QComboBox()
        self.relation_combo.addItems(["cuts", "cut_by", "terminates_at", "merges_with"])
        add_layout.addWidget(self.relation_combo)
        
        self.fault2_combo = QComboBox()
        self.fault2_combo.addItems([f.get('name', '') for f in self._faults])
        add_layout.addWidget(self.fault2_combo)
        
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_relationship)
        add_layout.addWidget(add_btn)
        
        layout.addLayout(add_layout)
        
        # Button box
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _add_relationship(self):
        """Add a relationship to the table."""
        f1 = self.fault1_combo.currentText()
        f2 = self.fault2_combo.currentText()
        rel = self.relation_combo.currentText()
        
        if f1 == f2:
            QMessageBox.warning(self, "Invalid", "Cannot create relationship between same fault.")
            return
        
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(f1))
        self.table.setItem(row, 1, QTableWidgetItem(rel))
        self.table.setItem(row, 2, QTableWidgetItem(f2))
        
        self._relationships[(f1, f2)] = rel
    
    def get_relationships(self) -> Dict[Tuple[str, str], str]:
        """Get the defined relationships."""
        return dict(self._relationships)


# =============================================================================
# MAIN FAULT DEFINITION PANEL
# =============================================================================

class FaultDefinitionPanel(BaseAnalysisPanel):
    """
    Panel for defining faults in geological models.
    
    Features:
    - Manual fault definition (point + dip/azimuth + throw)
    - Import fault surfaces from files
    - Fault relationship editor
    - Integration with GemPy and geological wizard
    """
    
    task_name = "fault_definition"
    
    # Signals
    faults_updated = pyqtSignal(list)  # Emits list of fault dicts
    preview_requested = pyqtSignal(dict)  # Emits fault dict for 3D preview
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="fault_definition")
        self.setWindowTitle("Fault Definition")
        
        self._faults: List[Dict[str, Any]] = []
        self._relationships: Dict[Tuple[str, str], str] = {}
        
        # Try to get registry
        try:
            self.registry = self.get_registry()
        except Exception:
            self.registry = None
        
        self._build_ui()
        self._load_from_registry()
    
    def _build_ui(self):
        """Build the panel UI."""
        layout = self.main_layout
        
        # Header
        header = QLabel("Fault Network Definition")
        header.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)
        
        info = QLabel(
            "Define faults to be incorporated into your geological model.\n"
            "Faults will displace geological contacts across the fault plane."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(info)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        add_btn = QPushButton("+ Add Fault")
        add_btn.clicked.connect(self._add_fault)
        toolbar_layout.addWidget(add_btn)
        
        import_btn = QPushButton("Import...")
        import_btn.clicked.connect(self._import_faults)
        toolbar_layout.addWidget(import_btn)
        
        relationships_btn = QPushButton("Relationships...")
        relationships_btn.clicked.connect(self._edit_relationships)
        toolbar_layout.addWidget(relationships_btn)
        
        toolbar_layout.addStretch()
        
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet("color: #ff5555;")
        clear_btn.clicked.connect(self._clear_all)
        toolbar_layout.addWidget(clear_btn)
        
        layout.addLayout(toolbar_layout)
        
        # Fault list
        self.fault_list_widget = QWidget()
        self.fault_list_layout = QVBoxLayout(self.fault_list_widget)
        self.fault_list_layout.setContentsMargins(0, 0, 0, 0)
        self.fault_list_layout.setSpacing(4)
        
        scroll = QScrollArea()
        scroll.setWidget(self.fault_list_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        layout.addWidget(scroll, 1)
        
        # Empty state label
        self.empty_label = QLabel("No faults defined.\nClick '+ Add Fault' to create one.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #666; font-style: italic;")
        self.fault_list_layout.addWidget(self.empty_label)
        
        # Summary
        self.summary_label = QLabel("0 faults defined")
        self.summary_label.setStyleSheet("color: #888; margin-top: 10px;")
        layout.addWidget(self.summary_label)
        
        # Apply button
        apply_btn = QPushButton("Apply to Model")
        apply_btn.setStyleSheet("background-color: #0078d4; padding: 8px;")
        apply_btn.clicked.connect(self._apply_to_model)
        layout.addWidget(apply_btn)
    
    def _load_from_registry(self):
        """Load existing faults from the registry."""
        if not self.registry:
            return
        
        try:
            geo_model = self.registry.get_model('GeoModel')
            if geo_model and 'fault_system' in geo_model:
                fault_system = geo_model['fault_system']
                faults = fault_system.get('faults', {})
                if isinstance(faults, dict):
                    self._faults = list(faults.values())
                elif isinstance(faults, list):
                    self._faults = faults
                
                self._relationships = fault_system.get('relationships', {})
                self._refresh_list()
        except Exception as e:
            logger.debug(f"Could not load faults from registry: {e}")
    
    def _refresh_list(self):
        """Refresh the fault list display."""
        # Clear existing items
        while self.fault_list_layout.count() > 0:
            item = self.fault_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self._faults:
            self.empty_label = QLabel("No faults defined.\nClick '+ Add Fault' to create one.")
            self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.empty_label.setStyleSheet("color: #666; font-style: italic;")
            self.fault_list_layout.addWidget(self.empty_label)
        else:
            for fault_data in self._faults:
                item = FaultListItem(fault_data)
                item.edited.connect(self._edit_fault)
                item.deleted.connect(self._delete_fault)
                item.toggled.connect(self._toggle_fault)
                self.fault_list_layout.addWidget(item)
        
        # Add stretch at end
        self.fault_list_layout.addStretch()
        
        # Update summary
        active = sum(1 for f in self._faults if f.get('active', True))
        self.summary_label.setText(f"{len(self._faults)} faults defined ({active} active)")
    
    def _add_fault(self):
        """Add a new fault."""
        dialog = FaultEditorDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            fault_data = dialog.get_fault_data()
            
            # Assign a color if not present
            if 'metadata' not in fault_data:
                fault_data['metadata'] = {}
            if 'color' not in fault_data.get('metadata', {}):
                # Generate color from index
                hue = (len(self._faults) * 0.618033988749895) % 1.0
                import colorsys
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                fault_data['metadata']['color'] = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                )
            
            self._faults.append(fault_data)
            self._refresh_list()
            self.faults_updated.emit(self._faults)
    
    def _edit_fault(self, name: str):
        """Edit an existing fault."""
        fault_data = next((f for f in self._faults if f.get('name') == name), None)
        if not fault_data:
            return
        
        dialog = FaultEditorDialog(fault_data=fault_data, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_data = dialog.get_fault_data()
            
            # Update in list
            for i, f in enumerate(self._faults):
                if f.get('name') == name:
                    # Preserve color
                    if 'metadata' in f and 'color' in f['metadata']:
                        if 'metadata' not in new_data:
                            new_data['metadata'] = {}
                        new_data['metadata']['color'] = f['metadata']['color']
                    self._faults[i] = new_data
                    break
            
            self._refresh_list()
            self.faults_updated.emit(self._faults)
    
    def _delete_fault(self, name: str):
        """Delete a fault."""
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete fault '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._faults = [f for f in self._faults if f.get('name') != name]
            
            # Remove relationships involving this fault
            self._relationships = {
                k: v for k, v in self._relationships.items()
                if name not in k
            }
            
            self._refresh_list()
            self.faults_updated.emit(self._faults)
    
    def _toggle_fault(self, name: str, active: bool):
        """Toggle a fault's active state."""
        for f in self._faults:
            if f.get('name') == name:
                f['active'] = active
                break
        
        self._refresh_list()
        self.faults_updated.emit(self._faults)
    
    def _import_faults(self):
        """Import faults from file."""
        # For now, just open the editor with import tab
        dialog = FaultEditorDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            fault_data = dialog.get_fault_data()
            self._faults.append(fault_data)
            self._refresh_list()
            self.faults_updated.emit(self._faults)
    
    def _edit_relationships(self):
        """Open the relationships editor."""
        if len(self._faults) < 2:
            QMessageBox.information(
                self, "Not Enough Faults",
                "You need at least 2 faults to define relationships."
            )
            return
        
        dialog = FaultRelationshipsDialog(self._faults, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._relationships = dialog.get_relationships()
    
    def _clear_all(self):
        """Clear all faults."""
        if not self._faults:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Clear",
            f"Delete all {len(self._faults)} faults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._faults.clear()
            self._relationships.clear()
            self._refresh_list()
            self.faults_updated.emit(self._faults)
    
    def _apply_to_model(self):
        """Apply faults to the geological model in registry."""
        if not self.registry:
            QMessageBox.warning(self, "No Registry", "Data registry not available.")
            return
        
        try:
            from ..geology.geo_model_registry import ensure_geo_model, update_geo_model_component
            
            ensure_geo_model(self.registry)
            
            fault_system = {
                'enabled': True,
                'faults': {f.get('name', f'fault_{i}'): f for i, f in enumerate(self._faults)},
                'relationships': {f"{k[0]}:{k[1]}": v for k, v in self._relationships.items()},
            }
            
            update_geo_model_component(self.registry, 'fault_system', fault_system)
            
            QMessageBox.information(
                self, "Success",
                f"Applied {len(self._faults)} faults to the geological model."
            )
            
        except Exception as e:
            logger.error(f"Failed to apply faults: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to apply faults:\n{e}")
    
    def get_faults(self) -> List[Dict[str, Any]]:
        """Get the current list of fault data."""
        return list(self._faults)
    
    def set_faults(self, faults: List[Dict[str, Any]]):
        """Set the fault list."""
        self._faults = list(faults)
        self._refresh_list()
    
    def get_fault_data_for_gempy(self) -> List[Any]:
        """
        Get faults in format suitable for GemPy engine.
        
        Returns:
            List of FaultData objects for GemPy
        """
        try:
            from ..geology.gempy_engine import FaultData, OrientationData
            from ..geology.faults import dip_azimuth_to_normal
        except ImportError:
            raise ImportError("Geology module removed - fault data functionality unavailable")
        
        fault_data_list = []
        
        for fault in self._faults:
            if not fault.get('active', True):
                continue
            
            point = np.array(fault.get('point', [0, 0, 0]), dtype=float)
            dip = float(fault.get('dip', 60))
            azimuth = float(fault.get('azimuth', 90))
            
            # Create orientation at the fault center
            normal = dip_azimuth_to_normal(dip, azimuth)
            orientation = OrientationData(
                x=point[0],
                y=point[1],
                z=point[2],
                dip=dip,
                azimuth=azimuth,
                polarity=1,
                formation=fault.get('name', 'Fault'),
                source='user_defined',
            )
            
            # Create FaultData
            # For simple faults, generate a grid of surface points
            # This is a simplified representation
            surface_points = np.array([point])  # Just the center for now
            
            affected = fault.get('affected_formations', [])
            
            fault_data = FaultData(
                name=fault.get('name', 'Fault'),
                surface_points=surface_points,
                orientations=[orientation],
                affected_formations=affected,
                fault_type='fault',
                metadata=fault.get('metadata', {}),
            )
            
            fault_data_list.append(fault_data)
        
        return fault_data_list


