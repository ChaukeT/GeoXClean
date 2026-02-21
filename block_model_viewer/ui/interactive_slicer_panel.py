"""
Interactive Slicer Panel — axis-aligned block-model slicer.

This panel is fully self-contained.  It reads PyVista meshes from the
renderer's ``active_layers`` dict and clips them with PyVista's
``clip_box`` — no custom renderer methods required.

Two modes
---------
**Box Clip** — six sliders (X/Y/Z min & max) crop the model to a sub-volume.
**Cross-Section** — a single plane along one axis, with adjustable thickness,
  shows a slab through the model.

The original mesh is hidden while the slicer is active and restored on reset.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QFormLayout, QCheckBox, QSlider,
    QDoubleSpinBox, QFrame, QSizePolicy, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

try:
    from .base_analysis_panel import BaseAnalysisPanel
    from .signals import UISignals
    from .modern_styles import ModernColors, get_button_stylesheet
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False
    UISignals = None

    class ModernColors:
        TEXT_PRIMARY = "#e0e0e0"
        TEXT_SECONDARY = "#a0a0a0"
        TEXT_HINT = "#707070"
        ELEVATED_BG = "#252530"
        DIVIDER = "#303038"
        ACCENT = "#4da6ff"

    def get_button_stylesheet(_kind="primary"):
        return ""

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _find_block_layers(renderer) -> Dict[str, Any]:
    """Return {layer_name: layer_info} for all block-type layers."""
    if renderer is None or not hasattr(renderer, "active_layers"):
        return {}
    out = {}
    for name, info in renderer.active_layers.items():
        lname = name.lower()
        ltype = info.get("layer_type", info.get("type", ""))
        is_block = (
            ltype in ("blocks", "classification")
            or any(k in lname for k in ("block", "sgsim", "kriging", "classification"))
        ) and "drillhole" not in lname
        if is_block:
            out[name] = info
    return out


class _RangeSliderPair(QWidget):
    """Two synchronised QSliders representing a min/max range.

    The widget maps an integer slider range [0 … ``steps``] to the
    real-valued range [``real_min`` … ``real_max``].
    """
    rangeChanged = pyqtSignal(float, float)

    STEPS = 500  # slider resolution

    def __init__(
        self,
        axis_label: str,
        real_min: float = 0.0,
        real_max: float = 1.0,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._real_min = real_min
        self._real_max = real_max

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        # --- header row: "X:  [123.4] — [456.7]"
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        lbl = QLabel(f"<b>{axis_label}</b>")
        lbl.setFixedWidth(22)
        lbl.setStyleSheet(f"color: {ModernColors.ACCENT};")
        hdr.addWidget(lbl)
        self.lo_label = QLabel()
        self.lo_label.setMinimumWidth(70)
        self.lo_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lo_label.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-family: monospace;")
        hdr.addWidget(self.lo_label)
        hdr.addWidget(QLabel("—"))
        self.hi_label = QLabel()
        self.hi_label.setMinimumWidth(70)
        self.hi_label.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-family: monospace;")
        hdr.addWidget(self.hi_label)
        hdr.addStretch()
        layout.addLayout(hdr)

        # --- sliders
        self.lo_slider = self._make_slider()
        self.hi_slider = self._make_slider()
        self.lo_slider.setValue(0)
        self.hi_slider.setValue(self.STEPS)

        self.lo_slider.valueChanged.connect(self._on_lo)
        self.hi_slider.valueChanged.connect(self._on_hi)

        layout.addWidget(self.lo_slider)
        layout.addWidget(self.hi_slider)

        self._refresh_labels()

    # ---- public API ----
    def set_range(self, real_min: float, real_max: float):
        self._real_min = real_min
        self._real_max = real_max
        self.lo_slider.blockSignals(True)
        self.hi_slider.blockSignals(True)
        self.lo_slider.setValue(0)
        self.hi_slider.setValue(self.STEPS)
        self.lo_slider.blockSignals(False)
        self.hi_slider.blockSignals(False)
        self._refresh_labels()

    def real_values(self) -> Tuple[float, float]:
        lo = self._to_real(self.lo_slider.value())
        hi = self._to_real(self.hi_slider.value())
        return (lo, hi)

    def reset(self):
        self.set_range(self._real_min, self._real_max)

    # ---- internals ----
    def _make_slider(self) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, self.STEPS)
        s.setMinimumHeight(20)
        return s

    def _to_real(self, tick: int) -> float:
        frac = tick / self.STEPS
        return self._real_min + frac * (self._real_max - self._real_min)

    def _on_lo(self, val):
        if val > self.hi_slider.value():
            self.lo_slider.blockSignals(True)
            self.lo_slider.setValue(self.hi_slider.value())
            self.lo_slider.blockSignals(False)
        self._refresh_labels()
        self._emit()

    def _on_hi(self, val):
        if val < self.lo_slider.value():
            self.hi_slider.blockSignals(True)
            self.hi_slider.setValue(self.lo_slider.value())
            self.hi_slider.blockSignals(False)
        self._refresh_labels()
        self._emit()

    def _refresh_labels(self):
        lo, hi = self.real_values()
        self.lo_label.setText(f"{lo:.1f}")
        self.hi_label.setText(f"{hi:.1f}")

    def _emit(self):
        lo, hi = self.real_values()
        self.rangeChanged.emit(lo, hi)


# ------------------------------------------------------------------ #
#  Panel
# ------------------------------------------------------------------ #

class InteractiveSlicerPanel(BaseAnalysisPanel if BASE_AVAILABLE else QWidget):
    """Axis-aligned block-model slicer with live preview."""

    # PanelManager metadata
    PANEL_ID = "InteractiveSlicerPanel"
    PANEL_NAME = "Interactive Slicer"
    PANEL_CATEGORY = "Visualization"
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = "Right"

    clipping_changed = pyqtSignal(dict)

    # ------------------------------------------------------------------ #
    #  Init
    # ------------------------------------------------------------------ #
    def __init__(self, parent=None, signals: Optional[UISignals] = None):
        self.signals = signals
        self.renderer = None

        # Slicer state
        self._source_mesh = None        # original PyVista mesh (never mutated)
        self._source_layer: str = ""     # name of the layer being sliced
        self._slice_actor = None         # actor for the clipped mesh
        self._is_active = False          # True while slicer is engaged
        self._debounce = QTimer()        # coalesce rapid slider moves
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(60)   # ms
        self._debounce.timeout.connect(self._apply_clip)

        if BASE_AVAILABLE:
            super().__init__(parent=parent, panel_id="interactive_slicer")
        else:
            super().__init__(parent)

        self._build_ui()
        self._set_controls_enabled(False)

    # ------------------------------------------------------------------ #
    #  Public
    # ------------------------------------------------------------------ #
    def set_renderer(self, renderer):
        """Called by MainWindow once the viewer is ready."""
        self.renderer = renderer
        has_plotter = (
            renderer is not None
            and hasattr(renderer, "plotter")
            and renderer.plotter is not None
        )
        self._set_controls_enabled(has_plotter)
        if has_plotter:
            self._refresh_layer_list()
            self._status("Ready — choose a layer and press <b>Activate</b>.")
        else:
            self._status("Waiting for 3-D viewer…")

    def refresh_layers(self):
        """Re-scan the renderer for block-model layers (call after new data)."""
        self._refresh_layer_list()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = self.main_layout if BASE_AVAILABLE else QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # ── Status ────────────────────────────────────────────────────
        self.status_label = QLabel("Waiting for 3-D viewer…")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            f"background: {ModernColors.ELEVATED_BG}; padding: 8px; "
            f"border-radius: 4px; color: {ModernColors.TEXT_PRIMARY};"
        )
        layout.addWidget(self.status_label)

        # ── Layer selector ────────────────────────────────────────────
        layer_grp = QGroupBox("Target Layer")
        layer_lay = QHBoxLayout(layer_grp)
        layer_lay.setContentsMargins(10, 14, 10, 10)

        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumHeight(30)
        self.layer_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layer_lay.addWidget(self.layer_combo)

        self.refresh_btn = QPushButton("⟳")
        self.refresh_btn.setFixedSize(30, 30)
        self.refresh_btn.setToolTip("Refresh layer list")
        self.refresh_btn.clicked.connect(self._refresh_layer_list)
        layer_lay.addWidget(self.refresh_btn)
        layout.addWidget(layer_grp)

        # ── Axis sliders ──────────────────────────────────────────────
        axis_grp = QGroupBox("Clip Bounds")
        axis_lay = QVBoxLayout(axis_grp)
        axis_lay.setContentsMargins(10, 18, 10, 10)
        axis_lay.setSpacing(8)

        self.x_range = _RangeSliderPair("X")
        self.y_range = _RangeSliderPair("Y")
        self.z_range = _RangeSliderPair("Z")

        for w in (self.x_range, self.y_range, self.z_range):
            w.rangeChanged.connect(self._on_slider_moved)
            axis_lay.addWidget(w)

        layout.addWidget(axis_grp)

        # ── Cross-section mode ────────────────────────────────────────
        cs_grp = QGroupBox("Cross-Section Mode")
        cs_lay = QVBoxLayout(cs_grp)
        cs_lay.setContentsMargins(10, 18, 10, 10)
        cs_lay.setSpacing(6)

        self.cs_check = QCheckBox("Enable single-plane cross-section")
        self.cs_check.setToolTip(
            "Reduce one axis to a thin slab.\n"
            "Use the Position slider to sweep through the model."
        )
        self.cs_check.toggled.connect(self._on_cross_section_toggled)
        cs_lay.addWidget(self.cs_check)

        self.cs_axis_combo = QComboBox()
        self.cs_axis_combo.addItems(["X axis", "Y axis", "Z axis"])
        self.cs_axis_combo.setCurrentIndex(2)  # Z default
        self.cs_axis_combo.currentIndexChanged.connect(self._on_cs_axis_changed)

        self.cs_pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.cs_pos_slider.setRange(0, _RangeSliderPair.STEPS)
        self.cs_pos_slider.setValue(_RangeSliderPair.STEPS // 2)
        self.cs_pos_slider.valueChanged.connect(self._on_slider_moved)

        self.cs_pos_label = QLabel("—")
        self.cs_pos_label.setMinimumWidth(80)
        self.cs_pos_label.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-family: monospace;")

        self.cs_thick_spin = QDoubleSpinBox()
        self.cs_thick_spin.setRange(0.1, 99999)
        self.cs_thick_spin.setDecimals(1)
        self.cs_thick_spin.setValue(10.0)
        self.cs_thick_spin.setSuffix(" m")
        self.cs_thick_spin.setToolTip("Slab thickness")
        self.cs_thick_spin.valueChanged.connect(self._on_slider_moved)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Axis:"))
        row1.addWidget(self.cs_axis_combo)
        row1.addSpacing(12)
        row1.addWidget(QLabel("Thickness:"))
        row1.addWidget(self.cs_thick_spin)
        cs_lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Position:"))
        row2.addWidget(self.cs_pos_slider, stretch=1)
        row2.addWidget(self.cs_pos_label)
        cs_lay.addLayout(row2)

        # Initially disabled
        for w in (self.cs_axis_combo, self.cs_pos_slider, self.cs_thick_spin):
            w.setEnabled(False)

        layout.addWidget(cs_grp)

        # ── Action buttons ────────────────────────────────────────────
        btn_lay = QHBoxLayout()
        btn_lay.setSpacing(8)

        self.activate_btn = QPushButton("✂  Activate Slicer")
        self.activate_btn.setMinimumHeight(40)
        self.activate_btn.setStyleSheet(
            "QPushButton { background: #1a6b1a; color: white; font-weight: bold; "
            "border-radius: 5px; } QPushButton:hover { background: #228B22; }"
        )
        self.activate_btn.clicked.connect(self._on_activate)
        btn_lay.addWidget(self.activate_btn)

        self.reset_btn = QPushButton("↺  Reset")
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.setStyleSheet(
            "QPushButton { background: #5c1a1a; color: white; font-weight: bold; "
            "border-radius: 5px; } QPushButton:hover { background: #8B2222; }"
        )
        self.reset_btn.clicked.connect(self._on_reset)
        self.reset_btn.setEnabled(False)
        btn_lay.addWidget(self.reset_btn)

        layout.addLayout(btn_lay)

        # ── Invert checkbox ───────────────────────────────────────────
        self.invert_check = QCheckBox("Invert (show outside the box)")
        self.invert_check.setToolTip("Show blocks outside the clipping box instead of inside")
        self.invert_check.toggled.connect(self._on_slider_moved)
        layout.addWidget(self.invert_check)

        # ── Tips ──────────────────────────────────────────────────────
        tips = QLabel(
            "<b>How to use</b><br>"
            "1. Select a block-model layer and press <b>Activate</b>.<br>"
            "2. Drag the <b>X / Y / Z</b> sliders to crop the volume.<br>"
            "3. Toggle <b>Cross-Section</b> mode to sweep a thin slab.<br>"
            "4. Press <b>Reset</b> to restore the full model."
        )
        tips.setWordWrap(True)
        tips.setStyleSheet(
            f"background: #111118; padding: 10px; border-radius: 4px; "
            f"color: {ModernColors.TEXT_HINT}; font-size: 10pt;"
        )
        layout.addWidget(tips)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    #  Layer management
    # ------------------------------------------------------------------ #
    def _refresh_layer_list(self):
        current = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()

        layers = _find_block_layers(self.renderer)
        if layers:
            for name in layers:
                self.layer_combo.addItem(name)
            # Restore previous selection if still available
            idx = self.layer_combo.findText(current)
            if idx >= 0:
                self.layer_combo.setCurrentIndex(idx)
        else:
            self.layer_combo.addItem("No block layers found")

        self.layer_combo.blockSignals(False)

    def _get_mesh_for_layer(self, layer_name: str):
        """Return the PyVista mesh for *layer_name*, or None."""
        if not self.renderer or not hasattr(self.renderer, "active_layers"):
            return None
        info = self.renderer.active_layers.get(layer_name)
        if info is None:
            return None
        data = info.get("data")
        if data is not None and hasattr(data, "bounds"):
            return data
        return None

    # ------------------------------------------------------------------ #
    #  Activate / Reset
    # ------------------------------------------------------------------ #
    def _on_activate(self):
        layer_name = self.layer_combo.currentText()
        if not layer_name or layer_name.startswith("No "):
            self._status("⚠ Select a valid block-model layer first.")
            return

        mesh = self._get_mesh_for_layer(layer_name)
        if mesh is None:
            self._status(f"⚠ Could not read mesh from '{layer_name}'.")
            return

        self._source_mesh = mesh
        self._source_layer = layer_name
        self._is_active = True

        # Read bounds and configure sliders
        b = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        self.x_range.set_range(b[0], b[1])
        self.y_range.set_range(b[2], b[3])
        self.z_range.set_range(b[4], b[5])

        # Set cross-section defaults to centre
        self.cs_pos_slider.setValue(_RangeSliderPair.STEPS // 2)
        self._update_cs_label()

        # Hide original layer
        if hasattr(self.renderer, "set_layer_visibility"):
            self.renderer.set_layer_visibility(layer_name, False)

        # Show initial (full) clip
        self._apply_clip()

        self.activate_btn.setEnabled(False)
        self.layer_combo.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self._status(f"Slicing <b>{layer_name}</b> — drag sliders to clip.")

    def _on_reset(self):
        """Restore the original mesh and deactivate the slicer."""
        # Remove clipped actor
        self._remove_slice_actor()

        # Show original layer
        if self._source_layer and hasattr(self.renderer, "set_layer_visibility"):
            self.renderer.set_layer_visibility(self._source_layer, True)

        if hasattr(self.renderer, "plotter") and self.renderer.plotter:
            self.renderer.plotter.render()

        self._source_mesh = None
        self._source_layer = ""
        self._is_active = False

        # Reset sliders
        self.x_range.reset()
        self.y_range.reset()
        self.z_range.reset()

        self.activate_btn.setEnabled(True)
        self.layer_combo.setEnabled(True)
        self.reset_btn.setEnabled(False)
        self._refresh_layer_list()
        self._status("Slicer deactivated — original model restored.")

    # ------------------------------------------------------------------ #
    #  Clipping logic
    # ------------------------------------------------------------------ #
    def _on_slider_moved(self, *_args):
        """Called on every slider tick — debounce before heavy work."""
        if not self._is_active:
            return
        self._update_cs_label()
        self._debounce.start()

    def _apply_clip(self):
        """Perform the actual PyVista clip and update the plotter."""
        if not self._is_active or self._source_mesh is None:
            return

        plotter = getattr(self.renderer, "plotter", None)
        if plotter is None:
            return

        mesh = self._source_mesh

        # ── Determine bounds ──────────────────────────────────────────
        if self.cs_check.isChecked():
            # Cross-section mode: one thin slab
            xlo, xhi = self.x_range.real_values()
            ylo, yhi = self.y_range.real_values()
            zlo, zhi = self.z_range.real_values()
            bounds = [xlo, xhi, ylo, yhi, zlo, zhi]

            axis_idx = self.cs_axis_combo.currentIndex()  # 0=X, 1=Y, 2=Z
            pos = self._cs_real_pos()
            half = self.cs_thick_spin.value() / 2.0

            # Override the relevant axis with the cross-section slab
            bounds[axis_idx * 2] = pos - half
            bounds[axis_idx * 2 + 1] = pos + half
        else:
            xlo, xhi = self.x_range.real_values()
            ylo, yhi = self.y_range.real_values()
            zlo, zhi = self.z_range.real_values()
            bounds = [xlo, xhi, ylo, yhi, zlo, zhi]

        invert = self.invert_check.isChecked()

        # ── Clip ──────────────────────────────────────────────────────
        try:
            clipped = mesh.clip_box(bounds, invert=invert)
        except Exception as e:
            logger.warning(f"clip_box failed: {e}")
            return

        if clipped is None or clipped.n_cells == 0:
            self._remove_slice_actor()
            plotter.render()
            self._status("Clip region is empty — widen the sliders.")
            return

        # ── Display ───────────────────────────────────────────────────
        self._remove_slice_actor()

        try:
            # Preserve original colormap / scalars if possible
            scalars = None
            cmap = "turbo"
            clim = None
            if mesh.active_scalars_name and mesh.active_scalars_name in clipped.array_names:
                scalars = mesh.active_scalars_name
                arr = mesh.active_scalars
                if arr is not None and len(arr) > 0:
                    clim = [float(np.nanmin(arr)), float(np.nanmax(arr))]

            self._slice_actor = plotter.add_mesh(
                clipped,
                scalars=scalars,
                cmap=cmap,
                clim=clim,
                show_edges=False,
                name="__slicer_clip__",
                reset_camera=False,
            )
            plotter.render()
            n = clipped.n_cells
            self._status(f"Showing <b>{n:,}</b> blocks.")

        except Exception as e:
            logger.error(f"Failed to display clipped mesh: {e}", exc_info=True)
            self._status(f"⚠ Render error: {e}")

        self.clipping_changed.emit({"bounds": bounds, "invert": invert})

    def _remove_slice_actor(self):
        plotter = getattr(self.renderer, "plotter", None)
        if plotter is None:
            return
        try:
            plotter.remove_actor("__slicer_clip__")
        except Exception:
            pass
        self._slice_actor = None

    # ------------------------------------------------------------------ #
    #  Cross-section helpers
    # ------------------------------------------------------------------ #
    def _on_cross_section_toggled(self, on: bool):
        for w in (self.cs_axis_combo, self.cs_pos_slider, self.cs_thick_spin):
            w.setEnabled(on)
        if self._is_active:
            self._debounce.start()

    def _on_cs_axis_changed(self, _idx):
        if self._is_active:
            self._debounce.start()

    def _cs_real_pos(self) -> float:
        """Map the cross-section slider tick to a real coordinate."""
        if self._source_mesh is None:
            return 0.0
        axis = self.cs_axis_combo.currentIndex()
        b = self._source_mesh.bounds
        lo = b[axis * 2]
        hi = b[axis * 2 + 1]
        frac = self.cs_pos_slider.value() / _RangeSliderPair.STEPS
        return lo + frac * (hi - lo)

    def _update_cs_label(self):
        self.cs_pos_label.setText(f"{self._cs_real_pos():.1f}")

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def _set_controls_enabled(self, enabled: bool):
        self.layer_combo.setEnabled(enabled)
        self.activate_btn.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled)

    def _status(self, html: str):
        self.status_label.setText(html)

    def refresh_theme(self):
        pass
