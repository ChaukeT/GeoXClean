"""
Cross-section panel for drillhole and block model visualization.

Redesigned:
- Status label replaces MessageBox spam
- Offset slider uses float-scaled integers for proper precision
- Batch export uses np.arange to avoid float drift
- Slider drag auto-applies section for interactive feel
- DataFrame block models are handled properly
- Controls disabled until mesh/plotter set
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QComboBox, QPushButton, QGroupBox, QFormLayout, QCheckBox, QSlider,
    QFileDialog, QMessageBox, QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from typing import Optional, List
import numpy as np
import logging
from pathlib import Path

from ..visualization.cross_section import (
    render_cross_section, get_orientation_normal,
    export_cross_section_data, export_cross_section_screenshot,
)

try:
    from .base_analysis_panel import BaseAnalysisPanel
    from .signals import UISignals
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False
    UISignals = None

logger = logging.getLogger(__name__)

STATUS_FLASH_MS = 4000

# Offset slider resolution: how many slider ticks per metre
_SLIDER_TICKS_PER_UNIT = 10


def _safe_panel_category():
    """Return PanelCategory string if the panel manager is available."""
    try:
        from block_model_viewer.ui.panel_manager import PanelCategory  # noqa: F401
        return "Visualization"
    except Exception:
        return None


def _safe_dock_area():
    try:
        from block_model_viewer.ui.panel_manager import DockArea  # noqa: F401
        return "Right"
    except Exception:
        return None


class CrossSectionPanel(BaseAnalysisPanel if BASE_AVAILABLE else QWidget):
    """Panel for controlling cross-section slicing of 3D data."""

    # PanelManager metadata
    PANEL_ID = "CrossSectionPanel"
    PANEL_NAME = "Cross-Section Panel"
    PANEL_CATEGORY = _safe_panel_category()
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = _safe_dock_area()

    # Signals
    section_updated = pyqtSignal()

    def __init__(self, parent=None, signals: Optional['UISignals'] = None):
        # Data attributes — set BEFORE super().__init__
        self.drillhole_mesh = None
        self.scalar_field = None
        self.colormap = 'viridis'
        self.plotter = None
        self._base_origin = None
        self._section_applied = False
        self._last_sliced_mesh = None
        self._last_orientation_info = {}
        self._block_model = None
        self.available_properties: List[str] = []
        self.export_directory = Path("exports")
        self.signals = signals

        if BASE_AVAILABLE:
            super().__init__(parent=parent, panel_id="cross_section")
        else:
            super().__init__(parent)

        self._connect_registry()
        self._setup_ui()
        self._set_controls_enabled(False)

    # ==================================================================
    # Registry integration
    # ==================================================================

    def _connect_registry(self):
        """Connect to DataRegistry for automatic block model detection."""
        self.registry = None
        if not BASE_AVAILABLE:
            return
        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.blockModelLoaded.connect(self._on_block_model_changed_signal)
                self.registry.blockModelGenerated.connect(self._on_block_model_changed_signal)
                self.registry.blockModelClassified.connect(self._on_block_model_changed_signal)
                self._refresh_available_block_models()
                logger.info("Cross-section panel connected to DataRegistry")
        except Exception as e:
            logger.warning(f"Registry connection failed: {e}")
            self.registry = None

    def _on_block_model_changed_signal(self, block_model):
        """Single handler for any block-model change signal."""
        self._update_block_model(block_model)
        logger.info("Cross-section panel updated with block model")

    def _refresh_available_block_models(self):
        """Load the first available block model from the registry."""
        if not self.registry:
            return
        try:
            model = self.registry.get_block_model()
            if model is None:
                try:
                    model = self.registry.get_classified_block_model()
                except Exception:
                    pass
            if model is not None:
                self._update_block_model(model)
        except Exception as e:
            logger.warning(f"Failed to refresh block models: {e}")

    def _update_block_model(self, block_model):
        """Update internal block model reference and available properties."""
        self._block_model = block_model
        # Notify parent if applicable
        if hasattr(self, 'on_block_model_changed'):
            self.on_block_model_changed()

        if block_model is not None:
            if hasattr(block_model, 'columns'):
                self.available_properties = [
                    c for c in block_model.columns
                    if c not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ')
                ]
            elif hasattr(block_model, 'properties'):
                self.available_properties = list(block_model.properties.keys())
            else:
                self.available_properties = []

            if hasattr(self, 'property_combo'):
                self.property_combo.clear()
                self.property_combo.addItems([''] + self.available_properties)
            logger.info(f"Block model updated: {len(self.available_properties)} properties")
        else:
            self.available_properties = []
            if hasattr(self, 'property_combo'):
                self.property_combo.clear()

    # ==================================================================
    # UI
    # ==================================================================

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Status label ---
        self.status_label = QLabel("No mesh loaded")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "background-color: #2b2b2b; padding: 6px; border-radius: 3px;"
        )
        layout.addWidget(self.status_label)

        # --- Section Mode ---
        orient_group = QGroupBox("Section Mode")
        orient_layout = QFormLayout(orient_group)

        self.section_mode_combo = QComboBox()
        self.section_mode_combo.addItems([
            'Vertical N–S', 'Vertical E–W', 'Custom Azimuth', 'Strike/Dip (Geological)',
        ])
        self.section_mode_combo.setToolTip(
            "Vertical N–S: Section along N-S axis (E-W plane)\n"
            "Vertical E–W: Section along E-W axis (N-S plane)\n"
            "Custom Azimuth: User-defined vertical plane azimuth\n"
            "Strike/Dip: Geological orientation"
        )
        self.section_mode_combo.currentTextChanged.connect(self._on_section_mode_changed)
        orient_layout.addRow("Section Mode:", self.section_mode_combo)

        # Azimuth (custom only)
        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(0.0, 360.0)
        self.azimuth_spin.setSingleStep(5.0)
        self.azimuth_spin.setSuffix("°")
        self.azimuth_spin.setEnabled(False)
        orient_layout.addRow("Azimuth:", self.azimuth_spin)

        # Strike / dip row
        sd_row = QHBoxLayout()
        self.strike_spin = QDoubleSpinBox()
        self.strike_spin.setRange(0.0, 360.0)
        self.strike_spin.setSingleStep(5.0)
        self.strike_spin.setSuffix("°")
        self.strike_spin.setEnabled(False)

        self.dip_spin = QDoubleSpinBox()
        self.dip_spin.setRange(0.0, 90.0)
        self.dip_spin.setSingleStep(5.0)
        self.dip_spin.setSuffix("°")
        self.dip_spin.setValue(90.0)
        self.dip_spin.setEnabled(False)

        sd_row.addWidget(QLabel("Strike:"))
        sd_row.addWidget(self.strike_spin)
        sd_row.addWidget(QLabel("Dip:"))
        sd_row.addWidget(self.dip_spin)
        sd_row.addStretch()
        orient_layout.addRow("", sd_row)

        layout.addWidget(orient_group)

        # --- Plane Position ---
        pos_group = QGroupBox("Plane Position")
        pos_layout = QFormLayout(pos_group)

        self.x_spin = self._make_pos_spin(" m")
        pos_layout.addRow("X (Easting):", self.x_spin)
        self.y_spin = self._make_pos_spin(" m")
        pos_layout.addRow("Y (Northing):", self.y_spin)
        self.z_spin = self._make_pos_spin(" m")
        pos_layout.addRow("Z (Elevation):", self.z_spin)

        # Offset — slider + spin, with float-scaled slider
        off_row = QHBoxLayout()
        off_row.addWidget(QLabel("Offset:"))

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setDecimals(1)
        self.offset_spin.setRange(-500.0, 500.0)
        self.offset_spin.setSingleStep(1.0)
        self.offset_spin.setSuffix(" m")
        self.offset_spin.valueChanged.connect(self._on_offset_spin_changed)
        off_row.addWidget(self.offset_spin)

        self.offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.offset_slider.setMinimum(-500 * _SLIDER_TICKS_PER_UNIT)
        self.offset_slider.setMaximum(500 * _SLIDER_TICKS_PER_UNIT)
        self.offset_slider.setValue(0)
        self.offset_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.offset_slider.setTickInterval(50 * _SLIDER_TICKS_PER_UNIT)
        self.offset_slider.valueChanged.connect(self._on_offset_slider_changed)
        off_row.addWidget(self.offset_slider)

        pos_layout.addRow("", off_row)

        self.center_btn = QPushButton("Center on Mesh")
        self.center_btn.setToolTip("Set plane origin to mesh centre and reset offset")
        self.center_btn.clicked.connect(self._center_on_mesh)
        pos_layout.addRow("", self.center_btn)

        layout.addWidget(pos_group)

        # --- Options ---
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        prop_row = QHBoxLayout()
        prop_row.addWidget(QLabel("Property:"))
        self.property_combo = QComboBox()
        self.property_combo.setEditable(False)
        self.property_combo.setToolTip("Property to colour the cross-section with")
        self.property_combo.currentTextChanged.connect(self._on_property_changed)
        prop_row.addWidget(self.property_combo)
        options_layout.addLayout(prop_row)

        self.overlay_check = QCheckBox("Overlay view (show full mesh as context)")
        self.overlay_check.setChecked(True)
        options_layout.addWidget(self.overlay_check)

        self.invert_check = QCheckBox("Show opposite side")
        options_layout.addWidget(self.invert_check)

        self.auto_apply_check = QCheckBox("Auto-apply on slider change")
        self.auto_apply_check.setChecked(True)
        self.auto_apply_check.setToolTip(
            "When enabled, the section updates in real time as you drag the offset slider"
        )
        options_layout.addWidget(self.auto_apply_check)

        layout.addWidget(options_group)

        # --- Apply / Clear ---
        btn_row = QHBoxLayout()

        self.apply_btn = QPushButton("Apply Section")
        self.apply_btn.clicked.connect(self.apply_section)
        btn_row.addWidget(self.apply_btn)

        self.clear_btn = QPushButton("Clear Section")
        self.clear_btn.clicked.connect(self.clear_section)
        btn_row.addWidget(self.clear_btn)

        layout.addLayout(btn_row)

        # --- Export ---
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        export_btns = QHBoxLayout()

        self.export_data_btn = QPushButton("Export Data (CSV)")
        self.export_data_btn.clicked.connect(self.export_data)
        self.export_data_btn.setEnabled(False)
        export_btns.addWidget(self.export_data_btn)

        self.export_screenshot_btn = QPushButton("Export Screenshot")
        self.export_screenshot_btn.clicked.connect(self.export_screenshot)
        self.export_screenshot_btn.setEnabled(False)
        export_btns.addWidget(self.export_screenshot_btn)

        self.export_both_btn = QPushButton("Export Both")
        self.export_both_btn.clicked.connect(self.export_both)
        self.export_both_btn.setEnabled(False)
        export_btns.addWidget(self.export_both_btn)

        export_layout.addLayout(export_btns)

        # Directory row
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Export Directory:"))
        self.export_dir_label = QLabel("exports")
        self.export_dir_label.setStyleSheet("color: grey;")
        dir_row.addWidget(self.export_dir_label)
        self.browse_dir_btn = QPushButton("Browse…")
        self.browse_dir_btn.clicked.connect(self._browse_export_directory)
        dir_row.addWidget(self.browse_dir_btn)
        export_layout.addLayout(dir_row)

        layout.addWidget(export_group)

        # --- Batch Export ---
        batch_group = QGroupBox("Batch Export")
        batch_layout = QFormLayout(batch_group)

        self.batch_start_spin = self._make_batch_spin(-200.0)
        batch_layout.addRow("Start offset:", self.batch_start_spin)

        self.batch_end_spin = self._make_batch_spin(200.0)
        batch_layout.addRow("End offset:", self.batch_end_spin)

        self.batch_step_spin = QDoubleSpinBox()
        self.batch_step_spin.setDecimals(2)
        self.batch_step_spin.setRange(0.1, 5000.0)
        self.batch_step_spin.setSingleStep(10.0)
        self.batch_step_spin.setValue(50.0)
        self.batch_step_spin.setSuffix(" m")
        batch_layout.addRow("Step:", self.batch_step_spin)

        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        batch_layout.addRow(self.batch_progress)

        self.export_batch_btn = QPushButton("Export Batch")
        self.export_batch_btn.setToolTip("Sweep offsets and export CSV + screenshot at each slice")
        self.export_batch_btn.clicked.connect(self.export_batch)
        batch_layout.addRow(self.export_batch_btn)

        layout.addWidget(batch_group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pos_spin(suffix: str = " m") -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(2)
        s.setRange(-1e6, 1e6)
        s.setSingleStep(10.0)
        s.setSuffix(suffix)
        return s

    @staticmethod
    def _make_batch_spin(default: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(2)
        s.setRange(-5000.0, 5000.0)
        s.setSingleStep(10.0)
        s.setValue(default)
        s.setSuffix(" m")
        return s

    # ==================================================================
    # Status helpers
    # ==================================================================

    def _flash(self, msg: str):
        """Show a temporary status message."""
        self.status_label.setText(msg)
        QTimer.singleShot(STATUS_FLASH_MS, self._restore_status)

    def _restore_status(self):
        if self._section_applied:
            info = self._last_orientation_info
            self.status_label.setText(
                f"Section active — {info.get('mode', '?')}, offset {self.offset_spin.value():.1f} m"
            )
        elif self.drillhole_mesh is not None:
            self.status_label.setText("Mesh loaded — ready to section")
        else:
            self.status_label.setText("No mesh loaded")

    def _set_controls_enabled(self, enabled: bool):
        """Enable/disable action controls."""
        for w in (
            self.apply_btn, self.clear_btn, self.center_btn,
            self.export_batch_btn,
        ):
            w.setEnabled(enabled)

    def _set_export_buttons(self, enabled: bool):
        self.export_data_btn.setEnabled(enabled)
        self.export_screenshot_btn.setEnabled(enabled)
        self.export_both_btn.setEnabled(enabled)

    # ==================================================================
    # Section mode changes
    # ==================================================================

    def _on_section_mode_changed(self, text: str):
        self.azimuth_spin.setEnabled(text == 'Custom Azimuth')
        is_sd = (text == 'Strike/Dip (Geological)')
        self.strike_spin.setEnabled(is_sd)
        self.dip_spin.setEnabled(is_sd)

    def _on_property_changed(self, property_name: str):
        self.scalar_field = property_name if property_name else None
        if self.signals and property_name:
            self.signals.crossSectionPropertyChanged.emit(property_name)

    # ==================================================================
    # Offset synchronisation (spin ↔ slider)
    # ==================================================================

    def _on_offset_spin_changed(self, value: float):
        self.offset_slider.blockSignals(True)
        self.offset_slider.setValue(int(round(value * _SLIDER_TICKS_PER_UNIT)))
        self.offset_slider.blockSignals(False)

    def _on_offset_slider_changed(self, tick_value: int):
        real_value = tick_value / _SLIDER_TICKS_PER_UNIT
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(real_value)
        self.offset_spin.blockSignals(False)
        # Auto-apply if enabled and a section is already shown
        if self.auto_apply_check.isChecked() and self._section_applied:
            self.apply_section()

    # ==================================================================
    # Public API — called by main window
    # ==================================================================

    def set_plotter(self, plotter):
        self.plotter = plotter
        self._set_controls_enabled(self.drillhole_mesh is not None and plotter is not None)

    def set_drillhole_mesh(self, mesh, scalar_field: str = None, colormap: str = 'viridis'):
        """Legacy entry point; delegates to set_mesh."""
        self.set_mesh(mesh, scalar_field, colormap)

    def set_mesh(self, mesh, scalar_field: str = None, colormap: str = 'viridis'):
        """Set the mesh for cross-sectioning."""
        self.drillhole_mesh = mesh
        self.scalar_field = scalar_field
        self.colormap = colormap

        if mesh is not None:
            center = mesh.center
            self.x_spin.setValue(center[0])
            self.y_spin.setValue(center[1])
            self.z_spin.setValue(center[2])
            self._base_origin = tuple(center)

            bounds = mesh.bounds
            self.x_spin.setRange(bounds[0], bounds[1])
            self.y_spin.setRange(bounds[2], bounds[3])
            self.z_spin.setRange(bounds[4], bounds[5])

            # Scale offset and batch ranges to mesh size
            span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            half = max(100.0, span * 0.5)
            self.offset_spin.setRange(-half, half)
            self.offset_slider.setMinimum(int(-half * _SLIDER_TICKS_PER_UNIT))
            self.offset_slider.setMaximum(int(half * _SLIDER_TICKS_PER_UNIT))

            self.batch_start_spin.setValue(-half)
            self.batch_end_spin.setValue(half)
            self.batch_step_spin.setValue(max(10.0, span / 10.0))

            # Auto-select property
            if hasattr(self, 'property_combo') and self.available_properties:
                if scalar_field and scalar_field in self.available_properties:
                    self.property_combo.setCurrentText(scalar_field)
                else:
                    self.property_combo.setCurrentText(self.available_properties[0])
                    self.scalar_field = self.available_properties[0]

            self._set_controls_enabled(self.plotter is not None)
            self._flash(f"Mesh loaded: {mesh.n_points:,} points")
        else:
            self._set_controls_enabled(False)
            self._restore_status()

    def set_block_model(self, block_model, property_name: str = None):
        """Set a block model for cross-sectioning."""
        self._block_model = block_model
        if hasattr(self, 'on_block_model_changed'):
            self.on_block_model_changed()

        if block_model is None:
            self.set_mesh(None, None)
            return

        self._update_block_model(block_model)

        try:
            import pyvista as pv

            mesh = None
            if hasattr(block_model, 'positions') and hasattr(block_model, 'block_count'):
                # BlockModel API → ImageData
                mesh = pv.ImageData()
                mesh.dimensions = tuple(block_model.grid_size)
                mesh.origin = block_model.origin
                mesh.spacing = block_model.spacing
                prop = property_name or next(iter(block_model.properties), 'values')
                mesh.cell_data[prop] = block_model.properties.get(
                    prop, np.zeros(block_model.block_count)
                )
            elif hasattr(block_model, 'columns'):
                # DataFrame — create PolyData point cloud so sectioning works
                x_col = _first_matching(block_model.columns, ['X', 'XC', 'x', 'xc', 'EASTING'])
                y_col = _first_matching(block_model.columns, ['Y', 'YC', 'y', 'yc', 'NORTHING'])
                z_col = _first_matching(block_model.columns, ['Z', 'ZC', 'z', 'zc', 'RL', 'ELEVATION'])
                if all([x_col, y_col, z_col]):
                    points = np.column_stack([
                        block_model[x_col].values,
                        block_model[y_col].values,
                        block_model[z_col].values,
                    ])
                    mesh = pv.PolyData(points)
                    for col in block_model.columns:
                        if col not in (x_col, y_col, z_col):
                            try:
                                mesh[col] = block_model[col].values
                            except Exception:
                                pass

            self.set_mesh(mesh, property_name)
        except ImportError:
            logger.warning("PyVista not available for block model mesh creation")
            self.set_mesh(None, property_name)

    # ==================================================================
    # Centre on mesh
    # ==================================================================

    def _center_on_mesh(self):
        if self.drillhole_mesh is None:
            return
        center = self.drillhole_mesh.center
        self.x_spin.setValue(center[0])
        self.y_spin.setValue(center[1])
        self.z_spin.setValue(center[2])
        self._base_origin = tuple(center)
        self.offset_spin.setValue(0.0)
        self.offset_slider.setValue(0)
        self._flash("Plane centred on mesh")

    # ==================================================================
    # Apply / Clear section
    # ==================================================================

    def _get_normal(self):
        """Compute the cutting-plane normal from current UI settings."""
        mode = self.section_mode_combo.currentText()
        kwargs = {}
        if mode == 'Vertical N–S':
            key = 'north-south'
        elif mode == 'Vertical E–W':
            key = 'east-west'
        elif mode == 'Custom Azimuth':
            key = 'custom'
            kwargs['azimuth'] = self.azimuth_spin.value()
        elif mode == 'Strike/Dip (Geological)':
            key = 'strike-dip'
            kwargs['strike'] = self.strike_spin.value()
            kwargs['dip'] = self.dip_spin.value()
        else:
            key = 'north-south'
        return get_orientation_normal(key, **kwargs), key, kwargs

    def apply_section(self):
        """Apply cross-section to the plotter."""
        if self.drillhole_mesh is None or self.plotter is None:
            self._flash("Cannot apply: mesh or plotter not set")
            return

        normal, orient_key, kwargs = self._get_normal()

        base = (self.x_spin.value(), self.y_spin.value(), self.z_spin.value())
        offset = self.offset_spin.value()
        origin = tuple(base[i] + offset * normal[i] for i in range(3))

        mode_text = self.section_mode_combo.currentText()

        try:
            sliced = render_cross_section(
                self.drillhole_mesh,
                normal=normal,
                origin=origin,
                scalar=self.scalar_field,
                cmap=self.colormap,
                invert=self.invert_check.isChecked(),
                plotter=self.plotter,
                show_scalar_bar=False,
                overlay=self.overlay_check.isChecked(),
                full_mesh_opacity=0.25,
                full_mesh_color="lightgrey",
            )

            if sliced is not None:
                self.plotter.render()
                self._section_applied = True
                self._last_sliced_mesh = sliced
                self._last_orientation_info = {
                    "mode": mode_text,
                    "strike": kwargs.get('strike'),
                    "dip": kwargs.get('dip'),
                    "orientation_key": orient_key,
                }
                self._set_export_buttons(True)
                self._flash(f"Section applied — {mode_text}, offset {offset:.1f} m")

                if self.signals:
                    self.signals.crossSectionUpdated.emit()
                self.section_updated.emit()
            else:
                self._section_applied = False
                self._last_sliced_mesh = None
                self._set_export_buttons(False)
                self._flash("Section did not intersect mesh")

        except Exception as e:
            logger.error(f"Error applying section: {e}", exc_info=True)
            self._flash(f"Error: {e}")

    def clear_section(self):
        if self.plotter is None:
            return
        try:
            self.plotter.remove_actor('cross_section', render=False)
            self.plotter.remove_actor('cross_section_full_mesh', render=False)
            self.plotter.render()
        except Exception as e:
            logger.debug(f"Error clearing section: {e}")

        self._section_applied = False
        self._last_sliced_mesh = None
        self._set_export_buttons(False)

        if self.signals:
            self.signals.crossSectionUpdated.emit()
        self.section_updated.emit()

        self._flash("Section cleared")

    # ==================================================================
    # Export helpers
    # ==================================================================

    def _browse_export_directory(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", str(self.export_directory.absolute())
        )
        if d:
            self.export_directory = Path(d)
            self.export_dir_label.setText(d)
            self.export_dir_label.setStyleSheet("")

    def _export_orientation_kwargs(self) -> dict:
        return dict(
            strike=self._last_orientation_info.get("strike"),
            dip=self._last_orientation_info.get("dip"),
            orientation=self._last_orientation_info.get("mode", "unknown"),
            save_dir=self.export_directory,
        )

    def export_data(self):
        if self._last_sliced_mesh is None:
            self._flash("No section data to export")
            return
        try:
            result = export_cross_section_data(
                self._last_sliced_mesh,
                scalar_field=self.scalar_field,
                **self._export_orientation_kwargs(),
            )
            if result and "csv" in result:
                self._flash(f"CSV exported → {Path(result['csv']).name}")
            else:
                self._flash("CSV export failed")
        except Exception as e:
            logger.error(f"Export data error: {e}", exc_info=True)
            self._flash(f"Export error: {e}")

    def export_screenshot(self):
        if not self._section_applied or self.plotter is None:
            self._flash("No section visible to screenshot")
            return
        try:
            result = export_cross_section_screenshot(
                self.plotter, **self._export_orientation_kwargs()
            )
            if result and "image" in result:
                self._flash(f"Screenshot exported → {Path(result['image']).name}")
            else:
                self._flash("Screenshot export failed")
        except Exception as e:
            logger.error(f"Screenshot export error: {e}", exc_info=True)
            self._flash(f"Export error: {e}")

    def export_both(self):
        if self._last_sliced_mesh is None or not self._section_applied:
            self._flash("No section data to export")
            return
        self.export_data()
        self.export_screenshot()
        self._flash("Exported data + screenshot")

    # ==================================================================
    # Batch export
    # ==================================================================

    def export_batch(self):
        """Sweep offsets and export CSV + screenshot for each slice."""
        if self.drillhole_mesh is None or self.plotter is None:
            self._flash("Set mesh and plotter before batch export")
            return

        start = self.batch_start_spin.value()
        end = self.batch_end_spin.value()
        step = self.batch_step_spin.value()
        if step <= 0:
            self._flash("Step must be positive")
            return

        # Use np.arange to avoid floating-point drift
        if start <= end:
            offsets = np.arange(start, end + step * 0.5, step)
        else:
            offsets = np.arange(start, end - step * 0.5, -step)

        self.export_directory.mkdir(parents=True, exist_ok=True)

        self.batch_progress.setRange(0, len(offsets))
        self.batch_progress.setValue(0)
        self.batch_progress.setVisible(True)
        self.export_batch_btn.setEnabled(False)

        total = 0
        try:
            for i, off in enumerate(offsets):
                off = float(round(off, 6))
                self.offset_spin.setValue(off)
                self.apply_section()

                if not self._section_applied or self._last_sliced_mesh is None:
                    self.batch_progress.setValue(i + 1)
                    continue

                kw = self._export_orientation_kwargs()

                try:
                    data_result = export_cross_section_data(
                        self._last_sliced_mesh, scalar_field=self.scalar_field, **kw
                    )
                    shot_result = export_cross_section_screenshot(self.plotter, **kw)

                    suffix = f"_off_{off:+.2f}m"
                    for r, key in [(data_result, "csv"), (shot_result, "image")]:
                        if r and key in r:
                            src = Path(r[key])
                            dst = src.with_stem(src.stem + suffix)
                            src.replace(dst)
                    total += 1
                except Exception as e:
                    logger.debug(f"Batch slice failed at offset {off}: {e}")

                self.batch_progress.setValue(i + 1)
                # Let the event loop breathe so the UI doesn't freeze completely
                from PyQt6.QtWidgets import QApplication
                QApplication.processEvents()

        finally:
            self.batch_progress.setVisible(False)
            self.export_batch_btn.setEnabled(True)

        self._flash(f"Batch export complete: {total} / {len(offsets)} slices → {self.export_directory}")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _first_matching(columns, candidates):
    """Return the first column name from *candidates* that exists in *columns*."""
    for c in candidates:
        if c in columns:
            return c
    return None
