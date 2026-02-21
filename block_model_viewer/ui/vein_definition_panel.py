"""
Vein Definition Panel
=====================

UI panel for defining veins and dykes in geological models.

Features:
- Manual vein definition (centerline, thickness, orientation)
- Import from drillhole intercepts
- Vein type classification
- 3D preview
- Integration with GemPy
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


# =============================================================================
# VEIN LIST ITEM WIDGET
# =============================================================================

class VeinListItem(QFrame):
    """Custom widget for displaying a vein in the list."""
    
    edited = pyqtSignal(str)
    deleted = pyqtSignal(str)
    toggled = pyqtSignal(str, bool)
    
    def __init__(self, vein_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self._vein_data = vein_data
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
            VeinListItem {
                background: #2d2d30;
                border: 1px solid #3d3d40;
                border-radius: 4px;
                margin: 2px;
            }
            VeinListItem:hover {
                border-color: #0078d4;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Active checkbox
        self.active_check = QCheckBox()
        self.active_check.setChecked(self._vein_data.get('active', True))
        self.active_check.stateChanged.connect(self._on_toggled)
        layout.addWidget(self.active_check)
        
        # Color indicator
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(20, 20)
        color = self._vein_data.get('metadata', {}).get('color', '#ffaa00')
        self.color_btn.setStyleSheet(f"background-color: {color}; border: 1px solid #555; border-radius: 2px;")
        self.color_btn.clicked.connect(self._pick_color)
        layout.addWidget(self.color_btn)
        
        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(0)
        
        name = self._vein_data.get('name', 'Unnamed')
        body_type = self._vein_data.get('body_type', 'vein')
        
        name_lbl = QLabel(f"<b>{name}</b>")
        name_lbl.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY};")
        info_layout.addWidget(name_lbl)
        
        thickness = self._vein_data.get('thickness', 1.0)
        dip = self._vein_data.get('dip', 90)
        dip_dir = self._vein_data.get('dip_direction', 0)
        
        details = f"{body_type.title()} | {thickness:.1f}m thick | Dip: {dip:.0f}° @ {dip_dir:.0f}°"
        details_lbl = QLabel(details)
        details_lbl.setStyleSheet("color: #888; font-size: 10px;")
        info_layout.addWidget(details_lbl)
        
        layout.addLayout(info_layout, 1)
        
        # Edit button
        edit_btn = QToolButton()
        edit_btn.setText("✎")
        edit_btn.clicked.connect(lambda: self.edited.emit(self._vein_data.get('name', '')))
        layout.addWidget(edit_btn)
        
        # Delete button
        del_btn = QToolButton()
        del_btn.setText("✕")
        del_btn.setStyleSheet("color: #ff5555;")
        del_btn.clicked.connect(lambda: self.deleted.emit(self._vein_data.get('name', '')))
        layout.addWidget(del_btn)
    
    def _on_toggled(self, state):
        self.toggled.emit(self._vein_data.get('name', ''), state == Qt.CheckState.Checked.value)
    
    def _pick_color(self):
        current = self._vein_data.get('metadata', {}).get('color', '#ffaa00')
        color = QColorDialog.getColor(QColor(current), self, "Select Vein Color")
        if color.isValid():
            hex_color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555; border-radius: 2px;")
            if 'metadata' not in self._vein_data:
                self._vein_data['metadata'] = {}
            self._vein_data['metadata']['color'] = hex_color
    
    @property
    def vein_data(self) -> Dict[str, Any]:
        return self._vein_data


# =============================================================================
# VEIN EDITOR DIALOG
# =============================================================================

class VeinEditorDialog(QDialog):
    """Dialog for editing a vein definition."""
    
    def __init__(self, vein_data: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self._vein_data = vein_data or {}
        self._is_new = vein_data is None
        self.setWindowTitle("New Vein/Dyke" if self._is_new else f"Edit: {vein_data.get('name', '')}")
        self.setMinimumWidth(450)
        self._setup_ui()
        self._load_data()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Basic info
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Vein_A1")
        basic_layout.addRow("Name:", self.name_edit)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["vein", "dyke", "sill", "sheet", "pegmatite", "quartz"])
        basic_layout.addRow("Type:", self.type_combo)
        
        layout.addWidget(basic_group)
        
        # Geometry
        geom_group = QGroupBox("Geometry")
        geom_layout = QFormLayout(geom_group)
        
        # Center point
        center_layout = QHBoxLayout()
        self.center_x = QDoubleSpinBox()
        self.center_x.setRange(-1e9, 1e9)
        self.center_x.setDecimals(2)
        self.center_y = QDoubleSpinBox()
        self.center_y.setRange(-1e9, 1e9)
        self.center_y.setDecimals(2)
        self.center_z = QDoubleSpinBox()
        self.center_z.setRange(-1e9, 1e9)
        self.center_z.setDecimals(2)
        center_layout.addWidget(QLabel("X:"))
        center_layout.addWidget(self.center_x)
        center_layout.addWidget(QLabel("Y:"))
        center_layout.addWidget(self.center_y)
        center_layout.addWidget(QLabel("Z:"))
        center_layout.addWidget(self.center_z)
        geom_layout.addRow("Center Point:", center_layout)
        
        # Thickness
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.1, 1000)
        self.thickness_spin.setDecimals(2)
        self.thickness_spin.setSuffix(" m")
        self.thickness_spin.setValue(1.0)
        geom_layout.addRow("Thickness:", self.thickness_spin)
        
        # Strike length
        self.strike_length_spin = QDoubleSpinBox()
        self.strike_length_spin.setRange(0, 100000)
        self.strike_length_spin.setDecimals(0)
        self.strike_length_spin.setSuffix(" m")
        self.strike_length_spin.setValue(100)
        geom_layout.addRow("Strike Length:", self.strike_length_spin)
        
        # Down-dip extent
        self.dip_extent_spin = QDoubleSpinBox()
        self.dip_extent_spin.setRange(0, 10000)
        self.dip_extent_spin.setDecimals(0)
        self.dip_extent_spin.setSuffix(" m")
        self.dip_extent_spin.setValue(50)
        geom_layout.addRow("Down-dip Extent:", self.dip_extent_spin)
        
        layout.addWidget(geom_group)
        
        # Orientation
        orient_group = QGroupBox("Orientation")
        orient_layout = QFormLayout(orient_group)
        
        self.dip_spin = QDoubleSpinBox()
        self.dip_spin.setRange(0, 90)
        self.dip_spin.setDecimals(1)
        self.dip_spin.setSuffix("°")
        self.dip_spin.setValue(90)
        orient_layout.addRow("Dip:", self.dip_spin)
        
        self.dip_dir_spin = QDoubleSpinBox()
        self.dip_dir_spin.setRange(0, 360)
        self.dip_dir_spin.setDecimals(1)
        self.dip_dir_spin.setSuffix("°")
        self.dip_dir_spin.setValue(0)
        orient_layout.addRow("Dip Direction:", self.dip_dir_spin)
        
        layout.addWidget(orient_group)
        
        # Contact type
        contact_group = QGroupBox("Contact Properties")
        contact_layout = QFormLayout(contact_group)
        
        self.contact_combo = QComboBox()
        self.contact_combo.addItems(["sharp", "gradational", "chilled", "alteration"])
        contact_layout.addRow("Contact Type:", self.contact_combo)
        
        self.geometry_combo = QComboBox()
        self.geometry_combo.addItems(["planar", "sigmoidal", "ptygmatic", "breccia", "stockwork"])
        contact_layout.addRow("Geometry:", self.geometry_combo)
        
        self.mineralization_edit = QLineEdit()
        self.mineralization_edit.setPlaceholderText("e.g., Au, Cu-Mo")
        contact_layout.addRow("Mineralization:", self.mineralization_edit)
        
        layout.addWidget(contact_group)
        
        # Active checkbox
        self.active_check = QCheckBox("Active in modeling")
        self.active_check.setChecked(True)
        layout.addWidget(self.active_check)
        
        layout.addStretch()
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _load_data(self):
        if not self._vein_data:
            return
        
        self.name_edit.setText(self._vein_data.get('name', ''))
        
        body_type = self._vein_data.get('body_type', 'vein')
        idx = self.type_combo.findText(body_type)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        
        centerline = self._vein_data.get('centerline', [[0, 0, 0]])
        if isinstance(centerline, (list, np.ndarray)):
            centerline = np.asarray(centerline)
            if centerline.ndim == 2 and len(centerline) > 0:
                center = centerline.mean(axis=0)
                self.center_x.setValue(float(center[0]))
                self.center_y.setValue(float(center[1]))
                self.center_z.setValue(float(center[2]))
        
        self.thickness_spin.setValue(float(self._vein_data.get('thickness', 1.0)))
        self.strike_length_spin.setValue(float(self._vein_data.get('strike_length', 100) or 100))
        self.dip_extent_spin.setValue(float(self._vein_data.get('down_dip_extent', 50) or 50))
        
        self.dip_spin.setValue(float(self._vein_data.get('dip', 90)))
        self.dip_dir_spin.setValue(float(self._vein_data.get('dip_direction', 0)))
        
        contact_type = self._vein_data.get('contact_type', 'sharp')
        idx = self.contact_combo.findText(contact_type)
        if idx >= 0:
            self.contact_combo.setCurrentIndex(idx)
        
        geometry = self._vein_data.get('geometry', 'planar')
        idx = self.geometry_combo.findText(geometry)
        if idx >= 0:
            self.geometry_combo.setCurrentIndex(idx)
        
        self.mineralization_edit.setText(self._vein_data.get('mineralization', '') or '')
        self.active_check.setChecked(self._vein_data.get('active', True))
    
    def get_vein_data(self) -> Dict[str, Any]:
        try:
            from ..geology.veins import ThinBody, ThinBodyType, ContactType, VeinGeometry
        except ImportError:
            raise ImportError("Geology module removed - vein functionality unavailable")
        
        name = self.name_edit.text().strip() or f"Vein_{id(self)}"
        
        try:
            body_type = ThinBodyType(self.type_combo.currentText())
        except ValueError:
            body_type = ThinBodyType.VEIN
        
        try:
            contact_type = ContactType(self.contact_combo.currentText())
        except ValueError:
            contact_type = ContactType.SHARP
        
        try:
            geometry = VeinGeometry(self.geometry_combo.currentText())
        except ValueError:
            geometry = VeinGeometry.PLANAR
        
        centerline = np.array([[
            self.center_x.value(),
            self.center_y.value(),
            self.center_z.value()
        ]])
        
        body = ThinBody(
            name=name,
            body_type=body_type,
            centerline=centerline,
            thickness=self.thickness_spin.value(),
            dip=self.dip_spin.value(),
            dip_direction=self.dip_dir_spin.value(),
            contact_type=contact_type,
            geometry=geometry,
            mineralization=self.mineralization_edit.text().strip() or None,
            strike_length=self.strike_length_spin.value() or None,
            down_dip_extent=self.dip_extent_spin.value() or None,
            active=self.active_check.isChecked(),
        )
        
        if 'metadata' in self._vein_data:
            body.metadata = dict(self._vein_data['metadata'])
        
        return body.to_dict()


# =============================================================================
# MAIN VEIN DEFINITION PANEL
# =============================================================================

class VeinDefinitionPanel(BaseAnalysisPanel):
    """Panel for defining veins and dykes."""
    
    task_name = "vein_definition"
    
    veins_updated = pyqtSignal(list)
    preview_requested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="vein_definition")
        self.setWindowTitle("Vein/Dyke Definition")
        
        self._veins: List[Dict[str, Any]] = []
        
        try:
            self.registry = self.get_registry()
        except Exception:
            self.registry = None
        
        self._build_ui()
        self._load_from_registry()
    
    def _build_ui(self):
        layout = self.main_layout
        
        header = QLabel("Vein & Dyke Definition")
        header.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)
        
        info = QLabel(
            "Define veins, dykes, and other thin intrusive bodies.\n"
            "These are modeled as separate geological series in GemPy."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(info)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        add_btn = QPushButton("+ Add Vein/Dyke")
        add_btn.clicked.connect(self._add_vein)
        toolbar_layout.addWidget(add_btn)
        
        toolbar_layout.addStretch()
        
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet("color: #ff5555;")
        clear_btn.clicked.connect(self._clear_all)
        toolbar_layout.addWidget(clear_btn)
        
        layout.addLayout(toolbar_layout)
        
        # List
        self.vein_list_widget = QWidget()
        self.vein_list_layout = QVBoxLayout(self.vein_list_widget)
        self.vein_list_layout.setContentsMargins(0, 0, 0, 0)
        self.vein_list_layout.setSpacing(4)
        
        scroll = QScrollArea()
        scroll.setWidget(self.vein_list_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        layout.addWidget(scroll, 1)
        
        self.empty_label = QLabel("No veins defined.\nClick '+ Add Vein/Dyke' to create one.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #666; font-style: italic;")
        self.vein_list_layout.addWidget(self.empty_label)
        
        self.summary_label = QLabel("0 veins/dykes defined")
        self.summary_label.setStyleSheet("color: #888; margin-top: 10px;")
        layout.addWidget(self.summary_label)
        
        apply_btn = QPushButton("Apply to Model")
        apply_btn.setStyleSheet("background-color: #0078d4; padding: 8px;")
        apply_btn.clicked.connect(self._apply_to_model)
        layout.addWidget(apply_btn)
    
    def _load_from_registry(self):
        if not self.registry:
            return
        
        try:
            geo_model = self.registry.get_model('GeoModel')
            if geo_model and 'intrusions' in geo_model:
                intrusions = geo_model['intrusions']
                veins = intrusions.get('veins', {})
                if isinstance(veins, dict):
                    self._veins = list(veins.values())
                elif isinstance(veins, list):
                    self._veins = veins
                self._refresh_list()
        except Exception as e:
            logger.debug(f"Could not load veins from registry: {e}")
    
    def _refresh_list(self):
        while self.vein_list_layout.count() > 0:
            item = self.vein_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self._veins:
            self.empty_label = QLabel("No veins defined.\nClick '+ Add Vein/Dyke' to create one.")
            self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.empty_label.setStyleSheet("color: #666; font-style: italic;")
            self.vein_list_layout.addWidget(self.empty_label)
        else:
            for vein_data in self._veins:
                item = VeinListItem(vein_data)
                item.edited.connect(self._edit_vein)
                item.deleted.connect(self._delete_vein)
                item.toggled.connect(self._toggle_vein)
                self.vein_list_layout.addWidget(item)
        
        self.vein_list_layout.addStretch()
        
        active = sum(1 for v in self._veins if v.get('active', True))
        self.summary_label.setText(f"{len(self._veins)} veins/dykes defined ({active} active)")
    
    def _add_vein(self):
        dialog = VeinEditorDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            vein_data = dialog.get_vein_data()
            
            if 'metadata' not in vein_data:
                vein_data['metadata'] = {}
            if 'color' not in vein_data.get('metadata', {}):
                hue = 0.08 + (len(self._veins) * 0.05) % 0.2
                import colorsys
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                vein_data['metadata']['color'] = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                )
            
            self._veins.append(vein_data)
            self._refresh_list()
            self.veins_updated.emit(self._veins)
    
    def _edit_vein(self, name: str):
        vein_data = next((v for v in self._veins if v.get('name') == name), None)
        if not vein_data:
            return
        
        dialog = VeinEditorDialog(vein_data=vein_data, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_data = dialog.get_vein_data()
            
            for i, v in enumerate(self._veins):
                if v.get('name') == name:
                    if 'metadata' in v and 'color' in v['metadata']:
                        if 'metadata' not in new_data:
                            new_data['metadata'] = {}
                        new_data['metadata']['color'] = v['metadata']['color']
                    self._veins[i] = new_data
                    break
            
            self._refresh_list()
            self.veins_updated.emit(self._veins)
    
    def _delete_vein(self, name: str):
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._veins = [v for v in self._veins if v.get('name') != name]
            self._refresh_list()
            self.veins_updated.emit(self._veins)
    
    def _toggle_vein(self, name: str, active: bool):
        for v in self._veins:
            if v.get('name') == name:
                v['active'] = active
                break
        
        self._refresh_list()
        self.veins_updated.emit(self._veins)
    
    def _clear_all(self):
        if not self._veins:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Clear",
            f"Delete all {len(self._veins)} veins/dykes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._veins.clear()
            self._refresh_list()
            self.veins_updated.emit(self._veins)
    
    def _apply_to_model(self):
        if not self.registry:
            QMessageBox.warning(self, "No Registry", "Data registry not available.")
            return
        
        try:
            from ..geology.geo_model_registry import ensure_geo_model, update_geo_model_component
            
            ensure_geo_model(self.registry)
            
            intrusion_system = {
                'enabled': True,
                'veins': {v.get('name', f'vein_{i}'): v for i, v in enumerate(self._veins)},
                'dykes': {},
            }
            
            update_geo_model_component(self.registry, 'intrusions', intrusion_system)
            
            QMessageBox.information(
                self, "Success",
                f"Applied {len(self._veins)} veins/dykes to the geological model."
            )
            
        except Exception as e:
            logger.error(f"Failed to apply veins: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to apply veins:\n{e}")
    
    def get_veins(self) -> List[Dict[str, Any]]:
        return list(self._veins)
    
    def set_veins(self, veins: List[Dict[str, Any]]):
        self._veins = list(veins)
        self._refresh_list()


