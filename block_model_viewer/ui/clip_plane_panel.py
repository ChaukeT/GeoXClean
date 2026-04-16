"""
Clip Plane Window — interactive ParaView-style clipping for block models.

Standalone QDialog window (like Compositing Window). Provides directional
slicing (X/Y/Z/custom normal), origin slider, invert/flip controls, and
export of the clipped section (VTK, CSV, screenshot).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox, QCheckBox, QDialog, QDoubleSpinBox, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSlider, QSpinBox, QVBoxLayout, QFormLayout, QFrame, QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification LUT builder
# ---------------------------------------------------------------------------
_CLASSIFICATION_CANDIDATES = ("Classification", "CLASS", "Category", "JORC_Classification")

_FALLBACK_COLORS = {
    "Measured": "#2ca02c",
    "Indicated": "#ffbf00",
    "Inferred": "#d62728",
    "Unclassified": "#7f7f7f",
}


def _build_classification_lut():
    """Build a VTK LookupTable for classification (int 0-3 -> RGBA)."""
    try:
        from ..models.jorc_classification_engine import CLASSIFICATION_COLORS
    except ImportError:
        CLASSIFICATION_COLORS = _FALLBACK_COLORS

    import vtk as _vtk
    lut = _vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(4)
    lut.SetTableRange(0, 3)
    for idx, cat in enumerate(["Measured", "Indicated", "Inferred", "Unclassified"]):
        h = CLASSIFICATION_COLORS[cat].lstrip("#")
        r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
        lut.SetTableValue(idx, r, g, b, 1.0)
    lut.Build()
    return lut


def _find_classification_scalar(mesh) -> Optional[str]:
    """Return the name of the classification cell-data array, or None."""
    if not hasattr(mesh, "cell_data"):
        return None
    for name in _CLASSIFICATION_CANDIDATES:
        if name in mesh.cell_data:
            return name
    return None


_PIT_SHELL_CANDIDATES = ("PIT_SHELL", "pit_shell", "SHELL", "shell_number")
_PIT_PROB_CANDIDATES  = ("PIT_PROBABILITY", "pit_probability", "LG_VALUE", "lg_value")


def _find_pit_scalars(mesh) -> Dict[str, str]:
    """
    Detect pit-optimisation scalars in *mesh*.

    Returns a dict with any of the following keys that are present:
        'shell'  → name of the PIT_SHELL integer array
        'prob'   → name of the PIT_PROBABILITY / LG_VALUE float array
    """
    result: Dict[str, str] = {}
    if not hasattr(mesh, "cell_data"):
        return result
    for name in _PIT_SHELL_CANDIDATES:
        if name in mesh.cell_data:
            result['shell'] = name
            break
    for name in _PIT_PROB_CANDIDATES:
        if name in mesh.cell_data:
            result['prob'] = name
            break
    return result


_DOMAIN_MASK_CANDIDATES = ("domain_mask", "DOMAIN_MASK", "informed", "data_support")


def _find_domain_mask_scalar(mesh) -> Optional[str]:
    """Return the name of the domain-mask cell-data array, or None."""
    if not hasattr(mesh, "cell_data"):
        return None
    for name in _DOMAIN_MASK_CANDIDATES:
        if name in mesh.cell_data:
            return name
    return None


def _find_nan_scalar(mesh) -> Optional[str]:
    """Return the name of a float cell-data array that contains NaN values, or None.

    Checks all float cell-data arrays (not just active scalar) because the
    active scalar may not be set or may be a non-grade array.
    """
    if not hasattr(mesh, "cell_data"):
        return None
    for name in mesh.cell_data:
        try:
            arr = np.asarray(mesh.cell_data[name])
            if np.issubdtype(arr.dtype, np.floating) and np.any(np.isnan(arr)):
                return name
        except Exception:
            continue
    return None


_GEO_PREFIXES = ("GeoSurface:", "GeoSolid:", "GeoUnified:", "geo_surface_", "geo_solid_")


def _find_block_layers(renderer) -> Dict[str, Any]:
    """Return {layer_name: layer_info} for all clippable layers.

    Includes:
    - Block models / voxel volumes / classification grids
    - Geological model surfaces and domain solids (GeoSurface:, GeoSolid:, ...)
    """
    if renderer is None or not hasattr(renderer, "active_layers"):
        return {}
    out: Dict[str, Any] = {}
    for name, info in renderer.active_layers.items():
        ltype = info.get("type", info.get("layer_type", ""))
        lname_lower = name.lower()

        # Block / volume / classification layers
        is_block = (
            ltype in ("blocks", "volume", "classification")
            or any(k in lname_lower for k in ("block", "sgsim", "kriging", "classification"))
        ) and "drillhole" not in lname_lower

        # Geological model surfaces and solids
        is_geo = (
            ltype in ("surface", "solid", "mesh", "geology")
            or any(name.startswith(pfx) for pfx in _GEO_PREFIXES)
        ) and "drillhole" not in lname_lower and "contact" not in lname_lower

        if is_block or is_geo:
            out[name] = info
    return out


# ============================================================================
# ClipPlaneWindow
# ============================================================================

_ACTOR_NAME = "__clip_plane_result__"


class ClipPlaneWindow(QDialog):
    """Standalone window for interactive clip-plane slicing."""

    def __init__(self, renderer, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Clip Plane (ParaView-style)")
        self.resize(420, 680)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self.renderer = renderer

        # Clip state
        self._source_mesh = None
        self._source_layer: str = ""
        self._clipped_mesh = None
        self._is_active = False
        self._classification_lut = None
        self._classification_scalar: Optional[str] = None
        self._is_classification = False

        self._save_counter = 0  # counter for naming saved clip layers

        # Domain mask (data support) detection
        self._domain_mask_scalar: Optional[str] = None
        self._nan_mask_scalar: Optional[str] = None  # scalar name used for NaN-based mask
        self._domain_mask_count: int = 0   # number of informed blocks
        self._domain_mask_total: int = 0   # total blocks

        # Pit optimisation scalar detection
        self._pit_scalars: Dict[str, str] = {}   # {'shell': 'PIT_SHELL', 'prob': ...}
        self._pit_shell_max: int = 0              # max shell index found in mesh

        # Debounce rapid slider/spinbox changes
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)
        self._debounce.timeout.connect(self._apply_clip)

        self._build_ui()
        self._refresh_layer_list()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def set_renderer(self, renderer):
        """Update the renderer reference (e.g. after viewer reload)."""
        self.renderer = renderer
        self._refresh_layer_list()

    def refresh_layers(self):
        """Re-scan active layers (call after new data is loaded)."""
        self._refresh_layer_list()

    # ------------------------------------------------------------------ #
    #  Close handling — deactivate clip before hiding
    # ------------------------------------------------------------------ #
    def closeEvent(self, event):
        """Deactivate clip plane on close so the original layer is restored."""
        if self._is_active:
            self._deactivate()
        event.accept()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Scroll area for all controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # ── Status ──────────────────────────────────────────────
        self.status_label = QLabel("Select a layer and press Activate.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # ── Layer selector ──────────────────────────────────────
        layer_group = QGroupBox("Source Layer")
        layer_lay = QVBoxLayout(layer_group)
        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip("Select the layer to clip (block model, geological surface, or solid)")
        self.layer_combo.currentTextChanged.connect(self._on_layer_combo_changed)
        layer_lay.addWidget(self.layer_combo)
        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Layers")
        self.refresh_btn.setToolTip("Re-scan active layers from the 3D viewer")
        self.refresh_btn.clicked.connect(self._refresh_layer_list)
        btn_row.addWidget(self.refresh_btn)
        layer_lay.addLayout(btn_row)
        layout.addWidget(layer_group)

        # ── Direction / Normal ──────────────────────────────────
        dir_group = QGroupBox("Clip Direction")
        dir_lay = QFormLayout(dir_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["X-Axis", "Y-Axis", "Z-Axis", "Custom"])
        self.preset_combo.setToolTip("Quick preset for clip normal direction")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        dir_lay.addRow("Preset:", self.preset_combo)

        normal_row = QHBoxLayout()
        self.nx_spin = QDoubleSpinBox()
        self.ny_spin = QDoubleSpinBox()
        self.nz_spin = QDoubleSpinBox()
        for spin, val, label in [
            (self.nx_spin, 1.0, "X"),
            (self.ny_spin, 0.0, "Y"),
            (self.nz_spin, 0.0, "Z"),
        ]:
            spin.setRange(-1.0, 1.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(3)
            spin.setValue(val)
            spin.setToolTip(f"Normal {label} component")
            spin.valueChanged.connect(self._on_normal_changed)
            normal_row.addWidget(QLabel(f"{label}:"))
            normal_row.addWidget(spin)
        dir_lay.addRow("Normal:", normal_row)

        self.flip_check = QCheckBox("Flip Normal (invert clip side)")
        self.flip_check.setToolTip("Show the other half of the model")
        self.flip_check.toggled.connect(self._schedule_clip)
        dir_lay.addRow(self.flip_check)

        layout.addWidget(dir_group)

        # ── Position ────────────────────────────────────────────
        pos_group = QGroupBox("Position")
        pos_lay = QVBoxLayout(pos_group)

        slider_row = QHBoxLayout()
        self.origin_slider = QSlider(Qt.Orientation.Horizontal)
        self.origin_slider.setRange(0, 1000)
        self.origin_slider.setValue(500)
        self.origin_slider.setToolTip("Slide the clip plane along the normal direction")
        self.origin_slider.valueChanged.connect(self._schedule_clip)
        slider_row.addWidget(QLabel("Offset:"))
        slider_row.addWidget(self.origin_slider)
        self.offset_label = QLabel("50.0%")
        self.offset_label.setMinimumWidth(50)
        slider_row.addWidget(self.offset_label)
        pos_lay.addLayout(slider_row)

        layout.addWidget(pos_group)

        # ── Display ─────────────────────────────────────────────
        disp_group = QGroupBox("Display")
        disp_lay = QVBoxLayout(disp_group)

        self.edges_check = QCheckBox("Show edges on clipped mesh")
        self.edges_check.toggled.connect(self._schedule_clip)
        disp_lay.addWidget(self.edges_check)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setToolTip("Clipped mesh opacity (0-100%)")
        self.opacity_slider.valueChanged.connect(self._schedule_clip)
        opacity_row.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("100%")
        self.opacity_label.setMinimumWidth(40)
        opacity_row.addWidget(self.opacity_label)
        disp_lay.addLayout(opacity_row)

        layout.addWidget(disp_group)

        # ── Activate / Deactivate ───────────────────────────────
        ctrl_row = QHBoxLayout()
        self.activate_btn = QPushButton("Activate Clip Plane")
        self.activate_btn.setToolTip("Start clipping the selected layer")
        self.activate_btn.clicked.connect(self._activate)
        ctrl_row.addWidget(self.activate_btn)

        self.deactivate_btn = QPushButton("Deactivate")
        self.deactivate_btn.setToolTip("Remove clip plane and restore original model")
        self.deactivate_btn.clicked.connect(self._deactivate)
        self.deactivate_btn.setEnabled(False)
        ctrl_row.addWidget(self.deactivate_btn)
        layout.addLayout(ctrl_row)

        # ── Category Filter (classification layers only) ───────
        # Placed after Activate so it appears right when clip is activated
        self.category_group = QGroupBox("Category Filter (Classification)")
        cat_lay = QVBoxLayout(self.category_group)

        cat_info = QLabel("Uncheck categories to hide them from the clipped view:")
        cat_info.setWordWrap(True)
        cat_lay.addWidget(cat_info)

        self._category_checks: Dict[str, QCheckBox] = {}
        for cat_name, color_hex in [
            ("Measured", "#2ca02c"),
            ("Indicated", "#ffbf00"),
            ("Inferred", "#d62728"),
            ("Unclassified", "#7f7f7f"),
        ]:
            cb = QCheckBox(cat_name)
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {color_hex}; font-weight: bold; }}")
            cb.toggled.connect(self._schedule_clip)
            cat_lay.addWidget(cb)
            self._category_checks[cat_name] = cb

        cat_btn_row = QHBoxLayout()
        select_all_btn = QPushButton("All")
        select_all_btn.setToolTip("Select all categories")
        select_all_btn.clicked.connect(self._select_all_categories)
        cat_btn_row.addWidget(select_all_btn)

        select_none_btn = QPushButton("None")
        select_none_btn.setToolTip("Deselect all categories")
        select_none_btn.clicked.connect(self._select_no_categories)
        cat_btn_row.addWidget(select_none_btn)
        cat_lay.addLayout(cat_btn_row)

        # Hidden by default — shown only when a classification layer is activated
        self.category_group.setVisible(False)
        layout.addWidget(self.category_group)

        # ── Pit Shell / Probability Filter ──────────────────────
        self.pit_group = QGroupBox("Pit Shell Filter")
        pit_lay = QFormLayout(self.pit_group)

        pit_info = QLabel(
            "Filter blocks by pit shell number or probability.\n"
            "Shell 0 = waste outside pit. Shell 1 = innermost core."
        )
        pit_info.setWordWrap(True)
        pit_lay.addRow(pit_info)

        self.pit_min_shell_spin = QSpinBox()
        self.pit_min_shell_spin.setRange(0, 99)
        self.pit_min_shell_spin.setValue(1)
        self.pit_min_shell_spin.setToolTip(
            "Show only blocks with PIT_SHELL >= this value.\n"
            "Set to 1 to hide waste (shell 0).\n"
            "Set to 0 to show all blocks."
        )
        self.pit_min_shell_spin.valueChanged.connect(self._schedule_clip)
        pit_lay.addRow("Min shell (≥):", self.pit_min_shell_spin)

        self.pit_max_shell_spin = QSpinBox()
        self.pit_max_shell_spin.setRange(0, 99)
        self.pit_max_shell_spin.setValue(99)
        self.pit_max_shell_spin.setToolTip(
            "Show only blocks with PIT_SHELL <= this value.\n"
            "Set to 99 to show all shells up to max."
        )
        self.pit_max_shell_spin.valueChanged.connect(self._schedule_clip)
        pit_lay.addRow("Max shell (≤):", self.pit_max_shell_spin)

        shell_btn_row = QHBoxLayout()
        self.pit_show_all_btn = QPushButton("All shells")
        self.pit_show_all_btn.setToolTip("Show all shells including waste (shell 0)")
        self.pit_show_all_btn.clicked.connect(self._pit_show_all)
        shell_btn_row.addWidget(self.pit_show_all_btn)

        self.pit_hide_waste_btn = QPushButton("Hide waste (≥1)")
        self.pit_hide_waste_btn.setToolTip("Hide shell 0 (waste outside pit)")
        self.pit_hide_waste_btn.clicked.connect(self._pit_hide_waste)
        shell_btn_row.addWidget(self.pit_hide_waste_btn)
        pit_lay.addRow(shell_btn_row)

        self.pit_shell_info = QLabel("")
        self.pit_shell_info.setWordWrap(True)
        pit_lay.addRow(self.pit_shell_info)

        # Hidden by default — shown only when PIT_SHELL is detected
        self.pit_group.setVisible(False)
        layout.addWidget(self.pit_group)

        # ── Data Support Filter (domain mask) ────────────────────
        self.mask_group = QGroupBox("Data Support Filter")
        mask_lay = QVBoxLayout(self.mask_group)

        self.mask_hide_check = QCheckBox("Hide blocks outside data support")
        self.mask_hide_check.setToolTip(
            "Remove blocks that are outside the estimation data support.\n"
            "Uses the domain_mask array if available, otherwise hides\n"
            "blocks with NaN values (uninformed/unestimated blocks)."
        )
        self.mask_hide_check.setChecked(True)
        self.mask_hide_check.stateChanged.connect(self._schedule_clip)
        mask_lay.addWidget(self.mask_hide_check)

        self.mask_info_label = QLabel("")
        self.mask_info_label.setWordWrap(True)
        mask_lay.addWidget(self.mask_info_label)

        # Hidden by default — shown only when domain_mask is detected
        self.mask_group.setVisible(False)
        layout.addWidget(self.mask_group)

        # ── Save / Export ───────────────────────────────────────
        export_group = QGroupBox("Save / Export")
        export_lay = QVBoxLayout(export_group)

        self.save_layer_btn = QPushButton("Save Clip as New Layer")
        self.save_layer_btn.setToolTip(
            "Add the current clipped section to the 3D viewer as a permanent layer.\n"
            "It will appear in the Property Panel layer list."
        )
        self.save_layer_btn.clicked.connect(self._save_as_layer)
        self.save_layer_btn.setEnabled(False)
        export_lay.addWidget(self.save_layer_btn)

        self.save_vtk_btn = QPushButton("Save Clipped Mesh (VTK)")
        self.save_vtk_btn.setToolTip("Save the clipped section as a VTK file")
        self.save_vtk_btn.clicked.connect(self._export_vtk)
        self.save_vtk_btn.setEnabled(False)
        export_lay.addWidget(self.save_vtk_btn)

        self.save_csv_btn = QPushButton("Export Data (CSV)")
        self.save_csv_btn.setToolTip("Export clipped block data as CSV")
        self.save_csv_btn.clicked.connect(self._export_csv)
        self.save_csv_btn.setEnabled(False)
        export_lay.addWidget(self.save_csv_btn)

        self.screenshot_btn = QPushButton("Screenshot")
        self.screenshot_btn.setToolTip("Save a screenshot of the current 3D view")
        self.screenshot_btn.clicked.connect(self._export_screenshot)
        self.screenshot_btn.setEnabled(False)
        export_lay.addWidget(self.screenshot_btn)

        layout.addWidget(export_group)

        # ── Info / Stats ────────────────────────────────────────
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll)

    # ------------------------------------------------------------------ #
    #  Layer combo change — preview pit scalars before activating
    # ------------------------------------------------------------------ #
    def _on_layer_combo_changed(self, layer_name: str):
        """
        When the user picks a layer, scan its mesh for pit scalars and
        show/hide the Pit Shell Filter group immediately — before the user
        even clicks Activate.  This way the filter is always visible when
        pit optimisation data is present.
        """
        if self._is_active:
            return  # don't interfere while a clip is live

        if not layer_name or self.renderer is None:
            self.pit_group.setVisible(False)
            self.mask_group.setVisible(False)
            return

        try:
            layer_info = self.renderer.active_layers.get(layer_name, {})
            data = layer_info.get('data')
            mesh = data.get('mesh') if isinstance(data, dict) and 'mesh' in data else data
            if mesh is None:
                self.pit_group.setVisible(False)
                self.mask_group.setVisible(False)
                return

            pit = _find_pit_scalars(mesh)
            if pit.get('shell'):
                shell_arr = mesh.cell_data[pit['shell']]
                max_s = int(np.max(shell_arr)) if len(shell_arr) else 0
                self._pit_scalars = pit
                self._pit_shell_max = max_s
                self.pit_max_shell_spin.blockSignals(True)
                self.pit_min_shell_spin.blockSignals(True)
                self.pit_max_shell_spin.setMaximum(max(99, max_s))
                self.pit_max_shell_spin.setValue(max_s)
                self.pit_min_shell_spin.setMaximum(max(99, max_s))
                self.pit_min_shell_spin.setValue(0)
                self.pit_max_shell_spin.blockSignals(False)
                self.pit_min_shell_spin.blockSignals(False)
                self.pit_shell_info.setText(
                    f"{pit['shell']} detected — shells 0–{max_s}. "
                    f"Click Activate to enable live filtering."
                )
                self.pit_group.setVisible(True)
            else:
                self._pit_scalars = {}
                self._pit_shell_max = 0
                self.pit_group.setVisible(False)

            # Preview domain mask (explicit array or NaN-derived)
            dm_scalar = _find_domain_mask_scalar(mesh)
            if dm_scalar:
                mask_arr = np.asarray(mesh.cell_data[dm_scalar])
                n_total = len(mask_arr)
                n_in = int(np.sum(mask_arr > 0))
                pct = (n_in / max(1, n_total)) * 100
                self.mask_info_label.setText(
                    f"Informed blocks: {n_in:,} / {n_total:,} ({pct:.1f}%) — "
                    f"Click Activate to enable filtering."
                )
                self.mask_group.setVisible(True)
            elif _find_nan_scalar(mesh):
                nan_s = _find_nan_scalar(mesh)
                arr = np.asarray(mesh.cell_data[nan_s])
                finite_mask = np.isfinite(arr)
                n_total = len(arr)
                n_in = int(finite_mask.sum())
                pct = (n_in / max(1, n_total)) * 100
                self.mask_info_label.setText(
                    f"Informed blocks: {n_in:,} / {n_total:,} ({pct:.1f}%) — "
                    f"Click Activate to enable filtering."
                )
                self.mask_group.setVisible(True)
            else:
                self.mask_group.setVisible(False)
        except Exception as exc:
            logger.debug("_on_layer_combo_changed scan error: %s", exc)
            self.pit_group.setVisible(False)
            self.mask_group.setVisible(False)

    # ------------------------------------------------------------------ #
    #  Layer list
    # ------------------------------------------------------------------ #
    def _refresh_layer_list(self):
        """Scan renderer for block-model layers and populate combo."""
        self.layer_combo.blockSignals(True)
        current = self.layer_combo.currentText()
        self.layer_combo.clear()
        layers = _find_block_layers(self.renderer)
        for name in layers:
            self.layer_combo.addItem(name)
        idx = self.layer_combo.findText(current)
        if idx >= 0:
            self.layer_combo.setCurrentIndex(idx)
        self.layer_combo.blockSignals(False)

        if self.layer_combo.count() == 0:
            self._status("No clippable layers found. Load a block model or build a geological model first.")
            self.pit_group.setVisible(False)
            self.mask_group.setVisible(False)
        else:
            self._status(f"Found {self.layer_combo.count()} layer(s). Press Activate to clip.")
            # Preview pit scalars for whichever layer is currently selected
            QTimer.singleShot(0, lambda: self._on_layer_combo_changed(
                self.layer_combo.currentText()))

    # ------------------------------------------------------------------ #
    #  Preset handling
    # ------------------------------------------------------------------ #
    _PRESETS = {
        "X-Axis": (1.0, 0.0, 0.0),
        "Y-Axis": (0.0, 1.0, 0.0),
        "Z-Axis": (0.0, 0.0, 1.0),
    }

    def _on_preset_changed(self, text: str):
        preset = self._PRESETS.get(text)
        if preset is None:
            return
        self.nx_spin.blockSignals(True)
        self.ny_spin.blockSignals(True)
        self.nz_spin.blockSignals(True)
        self.nx_spin.setValue(preset[0])
        self.ny_spin.setValue(preset[1])
        self.nz_spin.setValue(preset[2])
        self.nx_spin.blockSignals(False)
        self.ny_spin.blockSignals(False)
        self.nz_spin.blockSignals(False)
        self._schedule_clip()

    def _on_normal_changed(self):
        current_normal = (self.nx_spin.value(), self.ny_spin.value(), self.nz_spin.value())
        for name, preset in self._PRESETS.items():
            if all(abs(a - b) < 0.001 for a, b in zip(current_normal, preset)):
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentText(name)
                self.preset_combo.blockSignals(False)
                self._schedule_clip()
                return
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)
        self._schedule_clip()

    # ------------------------------------------------------------------ #
    #  Clip geometry helpers
    # ------------------------------------------------------------------ #
    def _get_normal(self) -> np.ndarray:
        n = np.array([self.nx_spin.value(), self.ny_spin.value(), self.nz_spin.value()])
        length = np.linalg.norm(n)
        if length < 1e-9:
            n = np.array([1.0, 0.0, 0.0])
        else:
            n = n / length
        if self.flip_check.isChecked():
            n = -n
        return n

    def _get_origin(self) -> np.ndarray:
        if self._source_mesh is None:
            return np.zeros(3)

        normal = self._get_normal()
        bounds = self._source_mesh.bounds
        corners = np.array([
            [bounds[0], bounds[2], bounds[4]],
            [bounds[1], bounds[2], bounds[4]],
            [bounds[0], bounds[3], bounds[4]],
            [bounds[1], bounds[3], bounds[4]],
            [bounds[0], bounds[2], bounds[5]],
            [bounds[1], bounds[2], bounds[5]],
            [bounds[0], bounds[3], bounds[5]],
            [bounds[1], bounds[3], bounds[5]],
        ])
        projections = corners @ normal
        proj_min, proj_max = projections.min(), projections.max()
        center = np.array(self._source_mesh.center)

        t = self.origin_slider.value() / 1000.0
        offset = proj_min + t * (proj_max - proj_min)

        center_proj = np.dot(center, normal)
        origin = center + (offset - center_proj) * normal
        return origin

    # ------------------------------------------------------------------ #
    #  Activate / Deactivate
    # ------------------------------------------------------------------ #
    def _activate(self):
        if self._is_active:
            self._deactivate()

        layer_name = self.layer_combo.currentText()
        if not layer_name:
            self._status("No layer selected.")
            return

        if self.renderer is None or not hasattr(self.renderer, "active_layers"):
            self._status("No renderer available.")
            return

        layer_info = self.renderer.active_layers.get(layer_name)
        if layer_info is None:
            self._status(f"Layer '{layer_name}' not found in renderer.")
            return

        mesh = layer_info.get("data")
        if mesh is None or not hasattr(mesh, "bounds"):
            self._status(f"Layer '{layer_name}' has no mesh data.")
            return

        self._source_mesh = mesh
        self._source_layer = layer_name

        # Detect classification — check layer type AND mesh cell_data
        ltype = layer_info.get("type", layer_info.get("layer_type", ""))
        self._classification_scalar = _find_classification_scalar(mesh)
        self._classification_lut = None

        # Classification if layer type says so OR if mesh contains a classification scalar
        self._is_classification = (
            ltype == "classification"
            or self._classification_scalar is not None
        )

        if self._is_classification and self._classification_scalar:
            try:
                self._classification_lut = _build_classification_lut()
                logger.info("[CLIP PLANE] Built classification LUT, scalar='%s'",
                            self._classification_scalar)
            except Exception as e:
                logger.warning("[CLIP PLANE] Failed to build classification LUT: %s", e)
                self._is_classification = False
        elif self._is_classification and not self._classification_scalar:
            self._is_classification = False

        # Detect pit-optimisation scalars
        self._pit_scalars = _find_pit_scalars(mesh)
        if self._pit_scalars.get('shell'):
            shell_arr = mesh.cell_data[self._pit_scalars['shell']]
            self._pit_shell_max = int(np.max(shell_arr)) if len(shell_arr) else 0
            self.pit_max_shell_spin.blockSignals(True)
            self.pit_max_shell_spin.setMaximum(max(99, self._pit_shell_max))
            self.pit_max_shell_spin.setValue(self._pit_shell_max)
            self.pit_max_shell_spin.blockSignals(False)
            self.pit_min_shell_spin.blockSignals(True)
            self.pit_min_shell_spin.setMaximum(max(99, self._pit_shell_max))
            self.pit_min_shell_spin.setValue(0)
            self.pit_min_shell_spin.blockSignals(False)
            self.pit_shell_info.setText(
                f"Detected: {self._pit_scalars['shell']} "
                f"(shells 0–{self._pit_shell_max})"
            )
            self.pit_group.setVisible(True)
            logger.info("[CLIP PLANE] Pit shell scalar='%s', max=%d",
                        self._pit_scalars['shell'], self._pit_shell_max)
        else:
            self._pit_scalars = {}
            self._pit_shell_max = 0
            self.pit_group.setVisible(False)

        # Detect domain mask (data support)
        self._domain_mask_scalar = _find_domain_mask_scalar(mesh)
        self._nan_mask_scalar = None
        if self._domain_mask_scalar:
            mask_arr = np.asarray(mesh.cell_data[self._domain_mask_scalar])
            self._domain_mask_total = len(mask_arr)
            self._domain_mask_count = int(np.sum(mask_arr > 0))
            pct = (self._domain_mask_count / max(1, self._domain_mask_total)) * 100
            self.mask_info_label.setText(
                f"Informed blocks: {self._domain_mask_count:,} / "
                f"{self._domain_mask_total:,} ({pct:.1f}%)"
            )
            self.mask_group.setVisible(True)
            self.mask_hide_check.setChecked(True)
            logger.info("[CLIP PLANE] Domain mask scalar='%s', informed=%d/%d",
                        self._domain_mask_scalar, self._domain_mask_count,
                        self._domain_mask_total)
        else:
            # No explicit domain_mask — check if any float scalar has NaN
            # blocks (uninformed/outside data support)
            nan_scalar = _find_nan_scalar(mesh)
            if nan_scalar:
                self._nan_mask_scalar = nan_scalar
                arr = np.asarray(mesh.cell_data[nan_scalar])
                finite_mask = np.isfinite(arr)
                self._domain_mask_total = len(arr)
                self._domain_mask_count = int(finite_mask.sum())
                pct = (self._domain_mask_count / max(1, self._domain_mask_total)) * 100
                self.mask_info_label.setText(
                    f"Informed blocks: {self._domain_mask_count:,} / "
                    f"{self._domain_mask_total:,} ({pct:.1f}%)"
                )
                self.mask_group.setVisible(True)
                self.mask_hide_check.setChecked(True)
                logger.info("[CLIP PLANE] NaN-based domain mask via '%s', "
                            "informed=%d/%d", nan_scalar,
                            self._domain_mask_count, self._domain_mask_total)
            else:
                self._domain_mask_count = 0
                self._domain_mask_total = 0
                self.mask_group.setVisible(False)

        # Hide original layer
        if hasattr(self.renderer, "set_layer_visibility"):
            self.renderer.set_layer_visibility(layer_name, False)

        self._is_active = True
        self.activate_btn.setEnabled(False)
        self.deactivate_btn.setEnabled(True)
        self.save_layer_btn.setEnabled(True)
        self.save_vtk_btn.setEnabled(True)
        self.save_csv_btn.setEnabled(True)
        self.screenshot_btn.setEnabled(True)
        self.layer_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)

        # Show/hide category filter for classification layers
        self.category_group.setVisible(self._is_classification)

        self._apply_clip()
        self._status(f"Clip plane active on '{layer_name}'. Adjust controls to slice.")
        logger.info("[CLIP PLANE] Activated on layer '%s' (%d cells)",
                    layer_name, mesh.n_cells if hasattr(mesh, "n_cells") else 0)

    def _deactivate(self):
        plotter = getattr(self.renderer, "plotter", None) if self.renderer else None

        if plotter is not None:
            try:
                plotter.remove_actor(_ACTOR_NAME)
            except Exception:
                pass
            plotter.render()

        if self._source_layer and self.renderer and hasattr(self.renderer, "set_layer_visibility"):
            self.renderer.set_layer_visibility(self._source_layer, True)

        self._is_active = False
        self._source_mesh = None
        self._source_layer = ""
        self._clipped_mesh = None
        self._classification_lut = None
        self._classification_scalar = None
        self._is_classification = False
        self._pit_scalars = {}
        self._pit_shell_max = 0
        self.pit_group.setVisible(False)
        self._domain_mask_scalar = None
        self._nan_mask_scalar = None
        self._domain_mask_count = 0
        self._domain_mask_total = 0
        self.mask_group.setVisible(False)

        self.activate_btn.setEnabled(True)
        self.deactivate_btn.setEnabled(False)
        self.save_layer_btn.setEnabled(False)
        self.save_vtk_btn.setEnabled(False)
        self.save_csv_btn.setEnabled(False)
        self.screenshot_btn.setEnabled(False)
        self.layer_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.info_label.setText("")
        self.category_group.setVisible(False)

        self._status("Clip plane deactivated. Original model restored.")
        logger.info("[CLIP PLANE] Deactivated")

    # ------------------------------------------------------------------ #
    #  Clipping
    # ------------------------------------------------------------------ #
    def _schedule_clip(self, *_args):
        if not self._is_active:
            return
        self.offset_label.setText(f"{self.origin_slider.value() / 10.0:.1f}%")
        self.opacity_label.setText(f"{self.opacity_slider.value()}%")
        self._debounce.start()

    def _apply_clip(self):
        if not self._is_active or self._source_mesh is None:
            return

        plotter = getattr(self.renderer, "plotter", None) if self.renderer else None
        if plotter is None:
            return

        normal = self._get_normal()
        origin = self._get_origin()
        opacity = self.opacity_slider.value() / 100.0
        show_edges = self.edges_check.isChecked()

        try:
            clipped = self._source_mesh.clip(
                normal=normal, origin=origin, crinkle=True
            )
            if clipped is None or clipped.n_cells == 0:
                self.info_label.setText("Clip produced 0 cells (plane outside model bounds).")
                return

            # Category filtering for classification layers
            if self._is_classification and self._classification_scalar:
                clipped = self._filter_by_category(clipped)
                if clipped is None or clipped.n_cells == 0:
                    self.info_label.setText("No cells match the selected categories.")
                    try:
                        plotter.remove_actor(_ACTOR_NAME)
                        plotter.render()
                    except Exception:
                        pass
                    return

            # Domain mask (data support) filtering
            if (self._domain_mask_scalar or self._nan_mask_scalar) and self.mask_hide_check.isChecked():
                clipped = self._filter_by_domain_mask(clipped)
                if clipped is None or clipped.n_cells == 0:
                    self.info_label.setText("No informed blocks in clipped region.")
                    try:
                        plotter.remove_actor(_ACTOR_NAME)
                        plotter.render()
                    except Exception:
                        pass
                    return

            # Pit shell threshold filtering
            if self._pit_scalars.get('shell'):
                clipped = self._filter_by_shell(clipped)
                if clipped is None or clipped.n_cells == 0:
                    self.info_label.setText("No cells match the shell filter range.")
                    try:
                        plotter.remove_actor(_ACTOR_NAME)
                        plotter.render()
                    except Exception:
                        pass
                    return

            self._clipped_mesh = clipped

            # ── Reuse the source layer's colormap / LUT / clim ──
            # Instead of building our own legend, pull everything from
            # the source layer's actor so colors are identical to the
            # main 3D view.  The main legend stays authoritative.
            source_lut = None
            source_layer = self.renderer.active_layers.get(self._source_layer, {})
            source_actor = source_layer.get('actor')
            if source_actor is not None:
                src_mapper = source_actor.GetMapper()
                if src_mapper is not None:
                    source_lut = src_mapper.GetLookupTable()

            scalar_name = None
            cmap = getattr(self.renderer, "current_colormap", "turbo") or "turbo"
            clim = None
            solid_color = None  # used for geo surfaces rendered with flat color

            if self._is_classification and self._classification_scalar:
                if hasattr(clipped, "cell_data") and self._classification_scalar in clipped.cell_data:
                    clipped.set_active_scalars(self._classification_scalar, preference="cell")
                    scalar_name = self._classification_scalar
                    cmap = "tab10"
                    clim = [0, 3]
            else:
                scalar_name = clipped.active_scalars_name
                # Use SOURCE mesh scalar range so colours stay consistent
                # across different clip positions (not just the clipped subset)
                if scalar_name and hasattr(self._source_mesh, "cell_data"):
                    if scalar_name in self._source_mesh.cell_data:
                        src_vals = np.asarray(self._source_mesh.cell_data[scalar_name])
                        finite = src_vals[np.isfinite(src_vals)]
                        if len(finite) > 0:
                            _p2 = float(np.nanpercentile(finite, 2))
                            _p98 = float(np.nanpercentile(finite, 98))
                            if _p98 > _p2:
                                clim = [_p2, _p98]
                            else:
                                clim = [float(np.nanmin(finite)), float(np.nanmax(finite))]

                # For geo surfaces/solids with no scalar data, preserve the
                # actor's flat color rather than using the turbo colormap
                if scalar_name is None and source_actor is not None:
                    try:
                        vtk_prop = source_actor.GetProperty()
                        if vtk_prop is not None:
                            src_mapper = source_actor.GetMapper()
                            # Only use flat color when scalar visibility is off
                            if src_mapper is None or not src_mapper.GetScalarVisibility():
                                r, g, b = vtk_prop.GetColor()
                                solid_color = (r, g, b)
                    except Exception:
                        pass

            # Never show a scalar bar — the main legend already covers this
            mesh_kwargs = dict(
                name=_ACTOR_NAME,
                show_edges=show_edges,
                opacity=opacity,
                show_scalar_bar=False,
                reset_camera=False,
                pickable=True,
                lighting=True,
            )
            if solid_color is not None:
                mesh_kwargs["color"] = solid_color
            else:
                mesh_kwargs.update(scalars=scalar_name, cmap=cmap, clim=clim)

            actor = plotter.add_mesh(clipped, **mesh_kwargs)

            # Copy the source layer's LUT so colours match exactly
            if source_lut is not None and scalar_name:
                mapper = actor.GetMapper()
                if mapper is not None:
                    mapper.SetLookupTable(source_lut)
                    if clim:
                        mapper.SetScalarRange(clim[0], clim[1])
                    mapper.SetScalarModeToUseCellData()
                    mapper.SelectColorArray(scalar_name)
                    mapper.SetScalarVisibility(True)
                    mapper.Modified()
            elif self._is_classification and self._classification_lut is not None and scalar_name:
                mapper = actor.GetMapper()
                if mapper is not None:
                    mapper.SetLookupTable(self._classification_lut)
                    mapper.SetScalarRange(0, 3)
                    mapper.SetScalarModeToUseCellData()
                    mapper.SelectColorArray(scalar_name)
                    mapper.SetScalarVisibility(True)
                    mapper.Modified()

            total = self._source_mesh.n_cells if hasattr(self._source_mesh, "n_cells") else 0
            clipped_n = clipped.n_cells
            self.info_label.setText(
                f"Clipped: {clipped_n:,} of {total:,} cells "
                f"({clipped_n / total * 100:.1f}%)"
                if total > 0 else f"Clipped: {clipped_n:,} cells"
            )

        except Exception as e:
            logger.warning("[CLIP PLANE] Clip error: %s", e, exc_info=True)
            self.info_label.setText(f"Clip error: {e}")

    # ------------------------------------------------------------------ #
    #  Category filtering
    # ------------------------------------------------------------------ #
    _CATEGORY_INDEX = {"Measured": 0, "Indicated": 1, "Inferred": 2, "Unclassified": 3}

    def _select_all_categories(self):
        for cb in self._category_checks.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self._schedule_clip()

    def _select_no_categories(self):
        for cb in self._category_checks.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self._schedule_clip()

    def _filter_by_category(self, mesh):
        """Filter mesh to only include cells matching the checked categories.

        Returns the filtered mesh, or None if nothing is selected.
        """
        if not self._classification_scalar:
            return mesh
        if not hasattr(mesh, "cell_data") or self._classification_scalar not in mesh.cell_data:
            return mesh

        # Check if all categories are selected — skip filtering entirely
        all_checked = all(cb.isChecked() for cb in self._category_checks.values())
        if all_checked:
            return mesh

        # Build mask of allowed category indices
        allowed_indices = set()
        for cat_name, cb in self._category_checks.items():
            if cb.isChecked():
                allowed_indices.add(self._CATEGORY_INDEX[cat_name])

        if not allowed_indices:
            return None

        class_data = mesh.cell_data[self._classification_scalar]
        mask = np.isin(class_data, list(allowed_indices))

        if not mask.any():
            return None

        # Extract cells matching the mask
        filtered = mesh.extract_cells(np.where(mask)[0])
        return filtered

    # ------------------------------------------------------------------ #
    #  Pit shell filtering
    # ------------------------------------------------------------------ #

    def _filter_by_shell(self, mesh):
        """
        Keep only cells whose PIT_SHELL value is within [min_shell, max_shell].

        Returns the filtered mesh, or None if nothing matches.
        """
        scalar = self._pit_scalars.get('shell')
        if not scalar:
            return mesh
        if not hasattr(mesh, "cell_data") or scalar not in mesh.cell_data:
            return mesh

        min_s = self.pit_min_shell_spin.value()
        max_s = self.pit_max_shell_spin.value()

        # Skip filtering when full range is selected
        if min_s == 0 and max_s >= self._pit_shell_max:
            return mesh

        shell_data = np.asarray(mesh.cell_data[scalar])
        mask = (shell_data >= min_s) & (shell_data <= max_s)
        if not mask.any():
            return None

        filtered = mesh.extract_cells(np.where(mask)[0])
        n_in = int(mask.sum())
        n_total = len(mask)
        self.pit_shell_info.setText(
            f"Shells {min_s}–{max_s}: {n_in:,} of {n_total:,} blocks "
            f"({n_in / n_total * 100:.1f}%)"
        )
        return filtered

    def _pit_show_all(self):
        """Reset shell filter to show all shells."""
        self.pit_min_shell_spin.blockSignals(True)
        self.pit_max_shell_spin.blockSignals(True)
        self.pit_min_shell_spin.setValue(0)
        self.pit_max_shell_spin.setValue(max(99, self._pit_shell_max))
        self.pit_min_shell_spin.blockSignals(False)
        self.pit_max_shell_spin.blockSignals(False)
        self._schedule_clip()

    def _pit_hide_waste(self):
        """Set min shell = 1 to hide waste blocks (shell 0)."""
        self.pit_min_shell_spin.blockSignals(True)
        self.pit_max_shell_spin.blockSignals(True)
        self.pit_min_shell_spin.setValue(1)
        self.pit_max_shell_spin.setValue(max(99, self._pit_shell_max))
        self.pit_min_shell_spin.blockSignals(False)
        self.pit_max_shell_spin.blockSignals(False)
        self._schedule_clip()

    # ------------------------------------------------------------------ #
    #  Domain mask (data support) filtering
    # ------------------------------------------------------------------ #

    def _filter_by_domain_mask(self, mesh):
        """Keep only informed cells (domain_mask > 0 or finite scalar values).

        Uses explicit domain_mask array if available, otherwise derives
        the mask from NaN values in the grade scalar (blocks outside
        data support have NaN grade values after domain masking).

        Returns the filtered mesh, or None if nothing matches.
        """
        scalar = self._domain_mask_scalar

        if scalar and hasattr(mesh, "cell_data") and scalar in mesh.cell_data:
            # Explicit domain_mask array
            mask_data = np.asarray(mesh.cell_data[scalar])
            mask = mask_data > 0
        elif self._nan_mask_scalar:
            # Derive mask from NaN values in the detected scalar
            nan_key = self._nan_mask_scalar
            if not hasattr(mesh, "cell_data") or nan_key not in mesh.cell_data:
                return mesh
            arr = np.asarray(mesh.cell_data[nan_key])
            if not np.issubdtype(arr.dtype, np.floating):
                return mesh
            mask = np.isfinite(arr)
        else:
            return mesh

        n_in = int(mask.sum())
        n_total = len(mask)

        if not mask.any():
            self.mask_info_label.setText(
                f"Informed blocks: 0 / {n_total:,} (0.0%)"
            )
            return None

        pct = n_in / max(1, n_total) * 100
        self.mask_info_label.setText(
            f"Informed blocks: {n_in:,} / {n_total:,} ({pct:.1f}%)"
        )
        return mesh.extract_cells(np.where(mask)[0])

    # ------------------------------------------------------------------ #
    #  Save as Layer
    # ------------------------------------------------------------------ #
    def _save_as_layer(self):
        """Register the current clipped mesh as a new permanent layer in the renderer.

        The layer uses type ``'clip'`` which is non-exclusive — it will NOT
        remove existing block/volume/classification layers.  After saving,
        the property panel is explicitly refreshed so the new layer appears
        in the Active Layer dropdown immediately.
        """
        if self._clipped_mesh is None:
            self._status("No clipped mesh to save. Adjust the clip plane first.")
            return

        if self.renderer is None or not hasattr(self.renderer, "plotter"):
            self._status("No renderer available.")
            return

        plotter = self.renderer.plotter
        if plotter is None:
            self._status("No plotter available.")
            return

        self._save_counter += 1
        layer_name = f"Clip Section {self._save_counter}"
        mesh = self._clipped_mesh.copy()

        try:
            # Reuse source layer's LUT for consistent colours
            source_lut = None
            source_layer = self.renderer.active_layers.get(self._source_layer, {})
            source_actor = source_layer.get('actor')
            if source_actor is not None:
                src_mapper = source_actor.GetMapper()
                if src_mapper is not None:
                    source_lut = src_mapper.GetLookupTable()

            scalar_name = None
            cmap = getattr(self.renderer, "current_colormap", "turbo") or "turbo"
            clim = None
            solid_color = None

            if self._is_classification and self._classification_scalar:
                if hasattr(mesh, "cell_data") and self._classification_scalar in mesh.cell_data:
                    mesh.set_active_scalars(self._classification_scalar, preference="cell")
                    scalar_name = self._classification_scalar
                    cmap = "tab10"
                    clim = [0, 3]
            else:
                scalar_name = mesh.active_scalars_name
                if scalar_name and self._source_mesh is not None and hasattr(self._source_mesh, "cell_data"):
                    if scalar_name in self._source_mesh.cell_data:
                        src_vals = np.asarray(self._source_mesh.cell_data[scalar_name])
                        finite = src_vals[np.isfinite(src_vals)]
                        if len(finite) > 0:
                            _p2 = float(np.nanpercentile(finite, 2))
                            _p98 = float(np.nanpercentile(finite, 98))
                            if _p98 > _p2:
                                clim = [_p2, _p98]
                            else:
                                clim = [float(np.nanmin(finite)), float(np.nanmax(finite))]

                if scalar_name is None and source_actor is not None:
                    try:
                        vtk_prop = source_actor.GetProperty()
                        if vtk_prop is not None:
                            src_mapper = source_actor.GetMapper()
                            if src_mapper is None or not src_mapper.GetScalarVisibility():
                                r, g, b = vtk_prop.GetColor()
                                solid_color = (r, g, b)
                    except Exception:
                        pass

            save_kwargs = dict(
                name=layer_name,
                show_edges=self.edges_check.isChecked(),
                opacity=self.opacity_slider.value() / 100.0,
                show_scalar_bar=False,
                reset_camera=False,
                pickable=True,
                lighting=True,
            )
            if solid_color is not None:
                save_kwargs["color"] = solid_color
            else:
                save_kwargs.update(scalars=scalar_name, cmap=cmap, clim=clim)

            actor = plotter.add_mesh(mesh, **save_kwargs)

            # Copy source LUT for exact colour match
            if source_lut is not None and scalar_name:
                mapper = actor.GetMapper()
                if mapper is not None:
                    mapper.SetLookupTable(source_lut)
                    if clim:
                        mapper.SetScalarRange(clim[0], clim[1])
                    mapper.SetScalarModeToUseCellData()
                    mapper.SelectColorArray(scalar_name)
                    mapper.SetScalarVisibility(True)
                    mapper.Modified()
            elif self._is_classification and self._classification_lut is not None and scalar_name:
                mapper = actor.GetMapper()
                if mapper is not None:
                    mapper.SetLookupTable(self._classification_lut)
                    mapper.SetScalarRange(0, 3)
                    mapper.SetScalarModeToUseCellData()
                    mapper.SelectColorArray(scalar_name)
                    mapper.SetScalarVisibility(True)
                    mapper.Modified()

            # Register in renderer's active_layers dict directly so the
            # property panel sees it even if add_layer's callback path fails.
            self.renderer.active_layers[layer_name] = {
                'actor': actor,
                'data': mesh,
                'visible': True,
                'opacity': self.opacity_slider.value() / 100.0,
                'type': 'clip',
            }

            # Also register in scene_layers for picking
            if hasattr(self.renderer, 'register_scene_layer'):
                try:
                    self.renderer.register_scene_layer(layer_name, actor, mesh, 'clip')
                except Exception:
                    pass

            # Explicitly notify UI — call the layer-change callback AND
            # directly refresh the property panel (belt-and-suspenders).
            if hasattr(self.renderer, 'layer_change_callback') and self.renderer.layer_change_callback:
                try:
                    self.renderer.layer_change_callback()
                except Exception as cb_err:
                    logger.debug("[CLIP PLANE] layer_change_callback error: %s", cb_err)

            # Direct property panel refresh as fallback
            main_win = self.parent()
            if main_win is not None:
                pp = getattr(main_win, 'property_panel', None)
                if pp is not None and hasattr(pp, 'update_layer_controls'):
                    pp.update_layer_controls()
                    logger.debug("[CLIP PLANE] Directly refreshed property panel")

            n_cells = mesh.n_cells if hasattr(mesh, "n_cells") else 0
            self._status(f"Saved '{layer_name}' ({n_cells:,} cells) — visible in Property Panel.")
            logger.info("[CLIP PLANE] Saved layer '%s' (%d cells)", layer_name, n_cells)

        except Exception as e:
            self._status(f"Save as layer error: {e}")
            logger.error("[CLIP PLANE] Save as layer failed: %s", e, exc_info=True)

    # ------------------------------------------------------------------ #
    #  Export to file
    # ------------------------------------------------------------------ #
    def _export_vtk(self):
        if self._clipped_mesh is None:
            self._status("No clipped mesh to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Clipped Mesh",
            "clipped_section.vtk",
            "VTK Files (*.vtk);;VTU Files (*.vtu);;All Files (*)"
        )
        if not path:
            return
        try:
            self._clipped_mesh.save(path)
            self._status(f"Saved clipped mesh to {Path(path).name}")
            logger.info("[CLIP PLANE] Exported VTK: %s (%d cells)", path, self._clipped_mesh.n_cells)
        except Exception as e:
            self._status(f"Export error: {e}")
            logger.error("[CLIP PLANE] VTK export failed: %s", e, exc_info=True)

    def _export_csv(self):
        if self._clipped_mesh is None:
            self._status("No clipped mesh to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Clipped Data",
            "clipped_section.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            import pandas as pd
            data = {}
            if hasattr(self._clipped_mesh, "cell_data"):
                for key in self._clipped_mesh.cell_data.keys():
                    arr = self._clipped_mesh.cell_data[key]
                    if arr.ndim == 1:
                        data[key] = arr

            if hasattr(self._clipped_mesh, "cell_centers"):
                centers = self._clipped_mesh.cell_centers().points
                data["X"] = centers[:, 0]
                data["Y"] = centers[:, 1]
                data["Z"] = centers[:, 2]

            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
            self._status(f"Exported {len(df)} rows to {Path(path).name}")
            logger.info("[CLIP PLANE] Exported CSV: %s (%d rows)", path, len(df))
        except Exception as e:
            self._status(f"CSV export error: {e}")
            logger.error("[CLIP PLANE] CSV export failed: %s", e, exc_info=True)

    def _export_screenshot(self):
        plotter = getattr(self.renderer, "plotter", None) if self.renderer else None
        if plotter is None:
            self._status("No plotter available.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot",
            "clip_plane_screenshot.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        if not path:
            return
        try:
            plotter.screenshot(path)
            self._status(f"Screenshot saved to {Path(path).name}")
            logger.info("[CLIP PLANE] Screenshot: %s", path)
        except Exception as e:
            self._status(f"Screenshot error: {e}")
            logger.error("[CLIP PLANE] Screenshot failed: %s", e, exc_info=True)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _status(self, msg: str):
        self.status_label.setText(msg)
