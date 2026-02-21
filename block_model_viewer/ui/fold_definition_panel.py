"""
Fold Definition Panel
=====================

UI panel for defining fold structures in geological models.

Features:
- Manual fold definition (axis points, plunge/trend, interlimb angle)
- Import fold axes from polylines
- Fold type classification
- 3D preview of fold axes
- Integration with GemPy and geological wizard
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
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


# =============================================================================
# FOLD LIST ITEM WIDGET
# =============================================================================

class FoldListItem(QFrame):
    """Custom widget for displaying a fold in the list."""
    
    edited = pyqtSignal(str)  # Fold name
    deleted = pyqtSignal(str)  # Fold name
    toggled = pyqtSignal(str, bool)  # Fold name, active state
    
    def __init__(self, fold_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self._fold_data = fold_data
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
            FoldListItem {
                background: #2d2d30;
                border: 1px solid #3d3d40;
                border-radius: 4px;
                margin: 2px;
            }
            FoldListItem:hover {
                border-color: #0078d4;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Active checkbox
        self.active_check = QCheckBox()
        self.active_check.setChecked(self._fold_data.get('active', True))
        self.active_check.setToolTip("Toggle fold active state")
        self.active_check.stateChanged.connect(self._on_toggled)
        layout.addWidget(self.active_check)
        
        # Color indicator
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(20, 20)
        color = self._fold_data.get('metadata', {}).get('color', '#55aa55')
        self.color_btn.setStyleSheet(f"background-color: {color}; border: 1px solid #555; border-radius: 2px;")
        self.color_btn.setToolTip("Click to change color")
        self.color_btn.clicked.connect(self._pick_color)
        layout.addWidget(self.color_btn)
        
        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(0)
        
        name = self._fold_data.get('name', 'Unnamed')
        fold_type = self._fold_data.get('fold_type', 'unknown')
        
        name_lbl = QLabel(f"<b>{name}</b>")
        name_lbl.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY};")
        info_layout.addWidget(name_lbl)
        
        # Details line
        plunge = self._fold_data.get('plunge', 0)
        trend = self._fold_data.get('trend', 0)
        interlimb = self._fold_data.get('interlimb_angle', 90)
        
        details = f"{fold_type.title()} | Plunge: {plunge:.0f}° → {trend:.0f}° | Interlimb: {interlimb:.0f}°"
        details_lbl = QLabel(details)
        details_lbl.setStyleSheet("color: #888; font-size: 10px;")
        info_layout.addWidget(details_lbl)
        
        layout.addLayout(info_layout, 1)
        
        # Edit button
        edit_btn = QToolButton()
        edit_btn.setText("✎")
        edit_btn.setToolTip("Edit fold")
        edit_btn.clicked.connect(lambda: self.edited.emit(self._fold_data.get('name', '')))
        layout.addWidget(edit_btn)
        
        # Delete button
        del_btn = QToolButton()
        del_btn.setText("✕")
        del_btn.setToolTip("Delete fold")
        del_btn.setStyleSheet("color: #ff5555;")
        del_btn.clicked.connect(lambda: self.deleted.emit(self._fold_data.get('name', '')))
        layout.addWidget(del_btn)
    
    def _on_toggled(self, state):
        """Handle active state toggle."""
        self.toggled.emit(self._fold_data.get('name', ''), state == Qt.CheckState.Checked.value)
    
    def _pick_color(self):
        """Open color picker."""
        current = self._fold_data.get('metadata', {}).get('color', '#55aa55')
        color = QColorDialog.getColor(QColor(current), self, "Select Fold Color")
        if color.isValid():
            hex_color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555; border-radius: 2px;")
            if 'metadata' not in self._fold_data:
                self._fold_data['metadata'] = {}
            self._fold_data['metadata']['color'] = hex_color
    
    @property
    def fold_data(self) -> Dict[str, Any]:
        """Get the current fold data."""
        return self._fold_data


# =============================================================================
# FOLD EDITOR DIALOG
# =============================================================================

class FoldEditorDialog(QDialog):
    """Dialog for editing a fold definition."""
    
    def __init__(self, fold_data: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self._fold_data = fold_data or {}
        self._is_new = fold_data is None
        self.setWindowTitle("New Fold" if self._is_new else f"Edit Fold: {fold_data.get('name', '')}")
        self.setMinimumWidth(450)
        self._setup_ui()
        self._load_data()
    
    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Basic info group
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Main_Anticline")
        basic_layout.addRow("Fold Name:", self.name_edit)
        
        self.fold_type_combo = QComboBox()
        self.fold_type_combo.addItems([
            "anticline", "syncline", "monocline", 
            "antiform", "synform", "overturned",
            "recumbent", "dome", "basin", "unknown"
        ])
        basic_layout.addRow("Fold Type:", self.fold_type_combo)
        
        layout.addWidget(basic_group)
        
        # Axis geometry group
        axis_group = QGroupBox("Fold Axis Geometry")
        axis_layout = QFormLayout(axis_group)
        
        # Hinge point
        hinge_layout = QHBoxLayout()
        self.hinge_x = QDoubleSpinBox()
        self.hinge_x.setRange(-1e9, 1e9)
        self.hinge_x.setDecimals(2)
        self.hinge_y = QDoubleSpinBox()
        self.hinge_y.setRange(-1e9, 1e9)
        self.hinge_y.setDecimals(2)
        self.hinge_z = QDoubleSpinBox()
        self.hinge_z.setRange(-1e9, 1e9)
        self.hinge_z.setDecimals(2)
        hinge_layout.addWidget(QLabel("X:"))
        hinge_layout.addWidget(self.hinge_x)
        hinge_layout.addWidget(QLabel("Y:"))
        hinge_layout.addWidget(self.hinge_y)
        hinge_layout.addWidget(QLabel("Z:"))
        hinge_layout.addWidget(self.hinge_z)
        axis_layout.addRow("Hinge Point:", hinge_layout)
        
        # Plunge
        self.plunge_spin = QDoubleSpinBox()
        self.plunge_spin.setRange(0, 90)
        self.plunge_spin.setDecimals(1)
        self.plunge_spin.setSuffix("°")
        self.plunge_spin.setValue(0)
        self.plunge_spin.setToolTip("Angle below horizontal (0° = horizontal, 90° = vertical)")
        axis_layout.addRow("Plunge:", self.plunge_spin)
        
        # Trend (azimuth of plunge direction)
        self.trend_spin = QDoubleSpinBox()
        self.trend_spin.setRange(0, 360)
        self.trend_spin.setDecimals(1)
        self.trend_spin.setSuffix("°")
        self.trend_spin.setValue(0)
        self.trend_spin.setToolTip("Azimuth direction of plunge (0° = North)")
        axis_layout.addRow("Trend:", self.trend_spin)
        
        # Axis length
        self.axis_length_spin = QDoubleSpinBox()
        self.axis_length_spin.setRange(0, 100000)
        self.axis_length_spin.setDecimals(0)
        self.axis_length_spin.setSuffix(" m")
        self.axis_length_spin.setValue(500)
        axis_layout.addRow("Axis Length:", self.axis_length_spin)
        
        layout.addWidget(axis_group)
        
        # Fold geometry group
        geom_group = QGroupBox("Fold Geometry")
        geom_layout = QFormLayout(geom_group)
        
        # Interlimb angle
        self.interlimb_spin = QDoubleSpinBox()
        self.interlimb_spin.setRange(0, 180)
        self.interlimb_spin.setDecimals(1)
        self.interlimb_spin.setSuffix("°")
        self.interlimb_spin.setValue(90)
        self.interlimb_spin.setToolTip("Angle between fold limbs (180° = gentle, 0° = isoclinal)")
        geom_layout.addRow("Interlimb Angle:", self.interlimb_spin)
        
        # Wavelength (optional)
        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(0, 100000)
        self.wavelength_spin.setDecimals(0)
        self.wavelength_spin.setSuffix(" m")
        self.wavelength_spin.setValue(0)
        self.wavelength_spin.setSpecialValueText("Not specified")
        self.wavelength_spin.setToolTip("Distance between adjacent hinges (for periodic folds)")
        geom_layout.addRow("Wavelength:", self.wavelength_spin)
        
        # Amplitude (optional)
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0, 10000)
        self.amplitude_spin.setDecimals(0)
        self.amplitude_spin.setSuffix(" m")
        self.amplitude_spin.setValue(0)
        self.amplitude_spin.setSpecialValueText("Not specified")
        self.amplitude_spin.setToolTip("Height/depth of fold from hinge to trough")
        geom_layout.addRow("Amplitude:", self.amplitude_spin)
        
        layout.addWidget(geom_group)
        
        # Axial surface group
        axial_group = QGroupBox("Axial Surface")
        axial_layout = QFormLayout(axial_group)
        
        self.axial_dip_spin = QDoubleSpinBox()
        self.axial_dip_spin.setRange(0, 90)
        self.axial_dip_spin.setDecimals(1)
        self.axial_dip_spin.setSuffix("°")
        self.axial_dip_spin.setValue(90)
        self.axial_dip_spin.setToolTip("Dip of axial surface (90° = upright, 0° = recumbent)")
        axial_layout.addRow("Axial Surface Dip:", self.axial_dip_spin)
        
        self.axial_azimuth_spin = QDoubleSpinBox()
        self.axial_azimuth_spin.setRange(0, 360)
        self.axial_azimuth_spin.setDecimals(1)
        self.axial_azimuth_spin.setSuffix("°")
        self.axial_azimuth_spin.setValue(0)
        axial_layout.addRow("Axial Surface Azimuth:", self.axial_azimuth_spin)
        
        layout.addWidget(axial_group)
        
        # Active checkbox
        self.active_check = QCheckBox("Fold is active in modeling")
        self.active_check.setChecked(True)
        layout.addWidget(self.active_check)
        
        layout.addStretch()
        
        # Button box
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _load_data(self):
        """Load existing fold data into the form."""
        if not self._fold_data:
            return
        
        self.name_edit.setText(self._fold_data.get('name', ''))
        
        fold_type = self._fold_data.get('fold_type', 'unknown')
        idx = self.fold_type_combo.findText(fold_type)
        if idx >= 0:
            self.fold_type_combo.setCurrentIndex(idx)
        
        # Axis points - use centroid as hinge
        axis_points = self._fold_data.get('axis_points', [[0, 0, 0]])
        if isinstance(axis_points, (list, np.ndarray)):
            axis_points = np.asarray(axis_points)
            if axis_points.ndim == 2 and len(axis_points) > 0:
                hinge = axis_points.mean(axis=0)
                self.hinge_x.setValue(float(hinge[0]))
                self.hinge_y.setValue(float(hinge[1]))
                self.hinge_z.setValue(float(hinge[2]))
        
        self.plunge_spin.setValue(float(self._fold_data.get('plunge', 0)))
        self.trend_spin.setValue(float(self._fold_data.get('trend', 0)))
        self.axis_length_spin.setValue(float(self._fold_data.get('axis_length', 500)))
        
        self.interlimb_spin.setValue(float(self._fold_data.get('interlimb_angle', 90)))
        self.wavelength_spin.setValue(float(self._fold_data.get('wavelength', 0) or 0))
        self.amplitude_spin.setValue(float(self._fold_data.get('amplitude', 0) or 0))
        
        self.axial_dip_spin.setValue(float(self._fold_data.get('axial_surface_dip', 90)))
        self.axial_azimuth_spin.setValue(float(self._fold_data.get('axial_surface_azimuth', 0)))
        
        self.active_check.setChecked(self._fold_data.get('active', True))
    
    def get_fold_data(self) -> Dict[str, Any]:
        """Get the fold data from the form."""
        try:
            from ..geology.folds import FoldAxis, FoldType, plunge_trend_to_vector
        except ImportError:
            raise ImportError("Geology module removed - fold functionality unavailable")
        
        name = self.name_edit.text().strip() or f"Fold_{id(self)}"
        
        # Calculate axis endpoints from hinge, plunge/trend, and length
        hinge = np.array([self.hinge_x.value(), self.hinge_y.value(), self.hinge_z.value()])
        plunge = self.plunge_spin.value()
        trend = self.trend_spin.value()
        length = self.axis_length_spin.value()
        
        axis_vec = plunge_trend_to_vector(plunge, trend)
        half_length = length / 2
        
        start = hinge - half_length * axis_vec
        end = hinge + half_length * axis_vec
        axis_points = np.vstack([start, hinge, end])
        
        # Get fold type
        try:
            fold_type = FoldType(self.fold_type_combo.currentText())
        except ValueError:
            fold_type = FoldType.UNKNOWN
        
        fold = FoldAxis(
            name=name,
            axis_points=axis_points,
            plunge=plunge,
            trend=trend,
            fold_type=fold_type,
            interlimb_angle=self.interlimb_spin.value(),
            wavelength=self.wavelength_spin.value() if self.wavelength_spin.value() > 0 else None,
            amplitude=self.amplitude_spin.value() if self.amplitude_spin.value() > 0 else None,
            axial_surface_dip=self.axial_dip_spin.value(),
            axial_surface_azimuth=self.axial_azimuth_spin.value(),
            active=self.active_check.isChecked(),
        )
        
        # Preserve metadata like color
        if 'metadata' in self._fold_data:
            fold.metadata = dict(self._fold_data['metadata'])
        
        return fold.to_dict()


# =============================================================================
# MAIN FOLD DEFINITION PANEL
# =============================================================================

class FoldDefinitionPanel(BaseAnalysisPanel):
    """
    Panel for defining fold structures in geological models.
    """
    
    task_name = "fold_definition"
    
    # Signals
    folds_updated = pyqtSignal(list)  # Emits list of fold dicts
    preview_requested = pyqtSignal(dict)  # Emits fold dict for 3D preview
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="fold_definition")
        self.setWindowTitle("Fold Definition")
        
        self._folds: List[Dict[str, Any]] = []
        
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
        header = QLabel("Fold Structure Definition")
        header.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)
        
        info = QLabel(
            "Define fold axes to be incorporated into your geological model.\n"
            "Folds generate synthetic orientations along their limbs for GemPy."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(info)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        add_btn = QPushButton("+ Add Fold")
        add_btn.clicked.connect(self._add_fold)
        toolbar_layout.addWidget(add_btn)
        
        toolbar_layout.addStretch()
        
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet("color: #ff5555;")
        clear_btn.clicked.connect(self._clear_all)
        toolbar_layout.addWidget(clear_btn)
        
        layout.addLayout(toolbar_layout)
        
        # Fold list
        self.fold_list_widget = QWidget()
        self.fold_list_layout = QVBoxLayout(self.fold_list_widget)
        self.fold_list_layout.setContentsMargins(0, 0, 0, 0)
        self.fold_list_layout.setSpacing(4)
        
        scroll = QScrollArea()
        scroll.setWidget(self.fold_list_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        layout.addWidget(scroll, 1)
        
        # Empty state label
        self.empty_label = QLabel("No folds defined.\nClick '+ Add Fold' to create one.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #666; font-style: italic;")
        self.fold_list_layout.addWidget(self.empty_label)
        
        # Summary
        self.summary_label = QLabel("0 folds defined")
        self.summary_label.setStyleSheet("color: #888; margin-top: 10px;")
        layout.addWidget(self.summary_label)
        
        # Apply button
        apply_btn = QPushButton("Apply to Model")
        apply_btn.setStyleSheet("background-color: #0078d4; padding: 8px;")
        apply_btn.clicked.connect(self._apply_to_model)
        layout.addWidget(apply_btn)
    
    def _load_from_registry(self):
        """Load existing folds from the registry."""
        if not self.registry:
            return
        
        try:
            geo_model = self.registry.get_model('GeoModel')
            if geo_model and 'folds' in geo_model:
                fold_system = geo_model['folds']
                folds = fold_system.get('folds', {})
                if isinstance(folds, dict):
                    self._folds = list(folds.values())
                elif isinstance(folds, list):
                    self._folds = folds
                
                self._refresh_list()
        except Exception as e:
            logger.debug(f"Could not load folds from registry: {e}")
    
    def _refresh_list(self):
        """Refresh the fold list display."""
        # Clear existing items
        while self.fold_list_layout.count() > 0:
            item = self.fold_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self._folds:
            self.empty_label = QLabel("No folds defined.\nClick '+ Add Fold' to create one.")
            self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.empty_label.setStyleSheet("color: #666; font-style: italic;")
            self.fold_list_layout.addWidget(self.empty_label)
        else:
            for fold_data in self._folds:
                item = FoldListItem(fold_data)
                item.edited.connect(self._edit_fold)
                item.deleted.connect(self._delete_fold)
                item.toggled.connect(self._toggle_fold)
                self.fold_list_layout.addWidget(item)
        
        # Add stretch at end
        self.fold_list_layout.addStretch()
        
        # Update summary
        active = sum(1 for f in self._folds if f.get('active', True))
        self.summary_label.setText(f"{len(self._folds)} folds defined ({active} active)")
    
    def _add_fold(self):
        """Add a new fold."""
        dialog = FoldEditorDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            fold_data = dialog.get_fold_data()
            
            # Assign a color if not present
            if 'metadata' not in fold_data:
                fold_data['metadata'] = {}
            if 'color' not in fold_data.get('metadata', {}):
                # Generate color from index (greenish for folds)
                hue = 0.3 + (len(self._folds) * 0.1) % 0.3
                import colorsys
                rgb = colorsys.hsv_to_rgb(hue, 0.6, 0.8)
                fold_data['metadata']['color'] = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                )
            
            self._folds.append(fold_data)
            self._refresh_list()
            self.folds_updated.emit(self._folds)
    
    def _edit_fold(self, name: str):
        """Edit an existing fold."""
        fold_data = next((f for f in self._folds if f.get('name') == name), None)
        if not fold_data:
            return
        
        dialog = FoldEditorDialog(fold_data=fold_data, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_data = dialog.get_fold_data()
            
            # Update in list
            for i, f in enumerate(self._folds):
                if f.get('name') == name:
                    # Preserve color
                    if 'metadata' in f and 'color' in f['metadata']:
                        if 'metadata' not in new_data:
                            new_data['metadata'] = {}
                        new_data['metadata']['color'] = f['metadata']['color']
                    self._folds[i] = new_data
                    break
            
            self._refresh_list()
            self.folds_updated.emit(self._folds)
    
    def _delete_fold(self, name: str):
        """Delete a fold."""
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete fold '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._folds = [f for f in self._folds if f.get('name') != name]
            self._refresh_list()
            self.folds_updated.emit(self._folds)
    
    def _toggle_fold(self, name: str, active: bool):
        """Toggle a fold's active state."""
        for f in self._folds:
            if f.get('name') == name:
                f['active'] = active
                break
        
        self._refresh_list()
        self.folds_updated.emit(self._folds)
    
    def _clear_all(self):
        """Clear all folds."""
        if not self._folds:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Clear",
            f"Delete all {len(self._folds)} folds?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._folds.clear()
            self._refresh_list()
            self.folds_updated.emit(self._folds)
    
    def _apply_to_model(self):
        """Apply folds to the geological model in registry."""
        if not self.registry:
            QMessageBox.warning(self, "No Registry", "Data registry not available.")
            return
        
        try:
            from ..geology.geo_model_registry import ensure_geo_model, update_geo_model_component
            
            ensure_geo_model(self.registry)
            
            fold_system = {
                'enabled': True,
                'folds': {f.get('name', f'fold_{i}'): f for i, f in enumerate(self._folds)},
            }
            
            update_geo_model_component(self.registry, 'folds', fold_system)
            
            QMessageBox.information(
                self, "Success",
                f"Applied {len(self._folds)} folds to the geological model."
            )
            
        except Exception as e:
            logger.error(f"Failed to apply folds: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to apply folds:\n{e}")
    
    def get_folds(self) -> List[Dict[str, Any]]:
        """Get the current list of fold data."""
        return list(self._folds)
    
    def set_folds(self, folds: List[Dict[str, Any]]):
        """Set the fold list."""
        self._folds = list(folds)
        self._refresh_list()
    
    def get_fold_orientations_for_gempy(self, formations: List[str]) -> List[Dict[str, Any]]:
        """
        Get synthetic orientations for all active folds.
        
        Args:
            formations: List of formation names to generate orientations for
            
        Returns:
            List of orientation dicts for GemPy
        """
        try:
            from ..geology.folds import FoldAxis, generate_fold_orientations
        except ImportError:
            raise ImportError("Geology module removed - fold orientation functionality unavailable")
        
        all_orientations = []
        
        for fold_data in self._folds:
            if not fold_data.get('active', True):
                continue
            
            try:
                fold = FoldAxis.from_dict(fold_data)
                
                # Generate orientations for each affected formation
                affected = fold.affected_formations or formations
                
                for formation in affected:
                    orientations = generate_fold_orientations(
                        fold=fold,
                        formation=formation,
                        n_orientations=10,
                        limb_extent=100.0,
                    )
                    all_orientations.extend(orientations)
                    
            except Exception as e:
                logger.warning(f"Failed to generate orientations for fold '{fold_data.get('name', 'Unknown')}': {e}")
        
        return all_orientations


