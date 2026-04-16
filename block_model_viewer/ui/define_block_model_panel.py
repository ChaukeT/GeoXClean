"""
Define Block Model Panel — single source of truth for block model geometry.

This panel is where the user specifies the shared block model grid that ALL
estimation methods will use.  No estimation panel should define its own grid.

Accessible from: Resources → Define Block Model
"""

import logging
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QLabel, QPushButton, QMessageBox,
    QFrame,
)
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WARN_BLOCKS = 1_000_000
_HARD_CAP = 5_000_000


class DefineBlockModelPanel(QWidget):
    """Panel for defining the shared block model grid geometry."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Define Block Model")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Header ───────────────────────────────────────────────
        header = QLabel(
            "<b>Block Model Definition</b><br>"
            "<span style='color: #888;'>All estimation methods share this grid.</span>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # ── Current definition info bar ──────────────────────────
        self._info_bar = QLabel("No block model defined.")
        self._info_bar.setWordWrap(True)
        self._info_bar.setStyleSheet(
            "background: #2a3a4a; padding: 6px; border-radius: 4px; color: #ccc;"
        )
        layout.addWidget(self._info_bar)

        # ── Origin group ─────────────────────────────────────────
        origin_group = QGroupBox("Grid Origin (corner of first block)")
        origin_form = QFormLayout(origin_group)
        self.x0_spin = self._make_coord_spin()
        self.y0_spin = self._make_coord_spin()
        self.z0_spin = self._make_coord_spin()
        origin_form.addRow("X\u2080 (m):", self.x0_spin)
        origin_form.addRow("Y\u2080 (m):", self.y0_spin)
        origin_form.addRow("Z\u2080 (m):", self.z0_spin)
        layout.addWidget(origin_group)

        # ── Block Size group ─────────────────────────────────────
        size_group = QGroupBox("Block Size")
        size_form = QFormLayout(size_group)
        self.dx_spin = self._make_size_spin(25.0)
        self.dy_spin = self._make_size_spin(25.0)
        self.dz_spin = self._make_size_spin(10.0)
        size_form.addRow("DX (m):", self.dx_spin)
        size_form.addRow("DY (m):", self.dy_spin)
        size_form.addRow("DZ (m):", self.dz_spin)
        layout.addWidget(size_group)

        # ── Block Count group ────────────────────────────────────
        count_group = QGroupBox("Number of Blocks")
        count_form = QFormLayout(count_group)
        self.nx_spin = self._make_count_spin(50)
        self.ny_spin = self._make_count_spin(50)
        self.nz_spin = self._make_count_spin(20)
        count_form.addRow("NX:", self.nx_spin)
        count_form.addRow("NY:", self.ny_spin)
        count_form.addRow("NZ:", self.nz_spin)
        layout.addWidget(count_group)

        # ── Summary display ──────────────────────────────────────
        summary_group = QGroupBox("Summary")
        summary_lay = QVBoxLayout(summary_group)
        self._total_label = QLabel("Total blocks: --")
        self._extent_label = QLabel("Extent: --")
        self._memory_label = QLabel("Memory per property: --")
        summary_lay.addWidget(self._total_label)
        summary_lay.addWidget(self._extent_label)
        summary_lay.addWidget(self._memory_label)
        layout.addWidget(summary_group)

        # ── Auto-fit button ──────────────────────────────────────
        auto_row = QHBoxLayout()
        self._auto_fit_btn = QPushButton("Auto-Fit to Drillhole Data")
        self._auto_fit_btn.setToolTip(
            "Compute origin and extent from drillhole/composite data "
            "with configurable padding."
        )
        self._auto_fit_btn.clicked.connect(self._auto_fit)

        self._auto_size_btn = QPushButton("Auto-Suggest Block Size")
        self._auto_size_btn.setToolTip(
            "Suggest block sizes from median composite spacing."
        )
        self._auto_size_btn.clicked.connect(self._auto_suggest_size)
        auto_row.addWidget(self._auto_fit_btn)
        auto_row.addWidget(self._auto_size_btn)
        layout.addLayout(auto_row)

        # ── Padding spin for auto-fit ────────────────────────────
        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("Auto-fit padding (%):"))
        self._padding_spin = QSpinBox()
        self._padding_spin.setRange(0, 50)
        self._padding_spin.setValue(5)
        pad_row.addWidget(self._padding_spin)
        pad_row.addStretch()
        layout.addLayout(pad_row)

        # ── Rotation (disabled — future) ─────────────────────────
        rot_label = QLabel(
            "<span style='color: #777;'>Rotation: not yet supported. "
            "Grid is axis-aligned.</span>"
        )
        rot_label.setWordWrap(True)
        layout.addWidget(rot_label)

        # ── Apply / Clear buttons ────────────────────────────────
        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply Definition")
        self._apply_btn.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; }"
        )
        self._apply_btn.clicked.connect(self._apply)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._clear_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

        # ── Connect spin changes to summary update ───────────────
        for spin in (self.x0_spin, self.y0_spin, self.z0_spin,
                     self.dx_spin, self.dy_spin, self.dz_spin,
                     self.nx_spin, self.ny_spin, self.nz_spin):
            spin.valueChanged.connect(self._update_summary)

        self._update_summary()
        self._refresh_info_bar()

    # ------------------------------------------------------------------
    # Widget factories
    # ------------------------------------------------------------------

    @staticmethod
    def _make_coord_spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-1e9, 1e9)
        s.setDecimals(2)
        s.setValue(0.0)
        s.setSingleStep(10.0)
        return s

    @staticmethod
    def _make_size_spin(default: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.01, 10000.0)
        s.setDecimals(2)
        s.setValue(default)
        s.setSingleStep(1.0)
        return s

    @staticmethod
    def _make_count_spin(default: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(1, 2000)
        s.setValue(default)
        return s

    # ------------------------------------------------------------------
    # Summary / info
    # ------------------------------------------------------------------

    def _update_summary(self):
        nx = self.nx_spin.value()
        ny = self.ny_spin.value()
        nz = self.nz_spin.value()
        dx = self.dx_spin.value()
        dy = self.dy_spin.value()
        dz = self.dz_spin.value()
        x0 = self.x0_spin.value()
        y0 = self.y0_spin.value()
        z0 = self.z0_spin.value()

        total = nx * ny * nz
        mem_mb = total * 8 / (1024 * 1024)

        colour = "#e0e0e0"
        if total > _HARD_CAP:
            colour = "#f44336"
        elif total > _WARN_BLOCKS:
            colour = "#ff9800"

        self._total_label.setText(
            f"Total blocks: <span style='color:{colour};'>{total:,}</span>"
        )
        self._extent_label.setText(
            f"Extent: X [{x0:.1f}, {x0 + nx * dx:.1f}] "
            f"Y [{y0:.1f}, {y0 + ny * dy:.1f}] "
            f"Z [{z0:.1f}, {z0 + nz * dz:.1f}]"
        )
        self._memory_label.setText(
            f"Memory per property: ~{mem_mb:.1f} MB "
            f"(10 properties \u2248 {10 * mem_mb:.0f} MB)"
        )

    def _refresh_info_bar(self):
        reg = self._get_registry()
        if reg is None:
            self._info_bar.setText("No registry available.")
            return
        defn = reg.get_block_model_definition()
        if defn is None:
            self._info_bar.setText(
                "\u26a0 No block model defined. Estimation methods cannot run."
            )
            self._info_bar.setStyleSheet(
                "background: #4a2a2a; padding: 6px; border-radius: 4px; color: #faa;"
            )
        else:
            self._info_bar.setText(
                f"\u2705 Active: {defn.nx}\u00d7{defn.ny}\u00d7{defn.nz} "
                f"@ {defn.dx}\u00d7{defn.dy}\u00d7{defn.dz}m "
                f"= {defn.n_blocks:,} blocks"
            )
            self._info_bar.setStyleSheet(
                "background: #1b3a1b; padding: 6px; border-radius: 4px; color: #afa;"
            )

    # ------------------------------------------------------------------
    # Auto-fit
    # ------------------------------------------------------------------

    def _auto_fit(self):
        """Compute origin and block counts from drillhole/composite data."""
        reg = self._get_registry()
        if reg is None:
            QMessageBox.warning(self, "No Data", "Registry not available.")
            return

        # Try composites first, then drillholes
        df = reg.get_data("composites", copy_data=False)
        if df is None or (hasattr(df, "empty") and df.empty):
            df = reg.get_data("drillhole_data", copy_data=False)
        if df is None or not hasattr(df, "columns"):
            QMessageBox.warning(
                self, "No Data",
                "Load drillhole or composite data before auto-fitting."
            )
            return

        # Find coordinate columns
        x_col = y_col = z_col = None
        for col in df.columns:
            cl = col.lower()
            if cl in ("x", "xc", "east", "easting"):
                x_col = col
            elif cl in ("y", "yc", "north", "northing"):
                y_col = col
            elif cl in ("z", "zc", "elev", "elevation", "rl"):
                z_col = col
        if not all([x_col, y_col, z_col]):
            QMessageBox.warning(
                self, "Missing Coordinates",
                "Cannot find X/Y/Z coordinate columns in data."
            )
            return

        coords = df[[x_col, y_col, z_col]].dropna().values.astype(float)
        if len(coords) == 0:
            QMessageBox.warning(self, "No Data", "All coordinate values are NaN.")
            return

        pad_pct = self._padding_spin.value() / 100.0
        dx = self.dx_spin.value()
        dy = self.dy_spin.value()
        dz = self.dz_spin.value()

        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        zmin, zmax = coords[:, 2].min(), coords[:, 2].max()

        # Add padding
        x_range = xmax - xmin
        y_range = ymax - ymin
        z_range = zmax - zmin
        xmin -= x_range * pad_pct
        xmax += x_range * pad_pct
        ymin -= y_range * pad_pct
        ymax += y_range * pad_pct
        zmin -= z_range * pad_pct
        zmax += z_range * pad_pct

        # Snap origin to block boundary
        x0 = np.floor(xmin / dx) * dx
        y0 = np.floor(ymin / dy) * dy
        z0 = np.floor(zmin / dz) * dz

        nx = max(1, int(np.ceil((xmax - x0) / dx)))
        ny = max(1, int(np.ceil((ymax - y0) / dy)))
        nz = max(1, int(np.ceil((zmax - z0) / dz)))

        self.x0_spin.setValue(x0)
        self.y0_spin.setValue(y0)
        self.z0_spin.setValue(z0)
        self.nx_spin.setValue(min(nx, 2000))
        self.ny_spin.setValue(min(ny, 2000))
        self.nz_spin.setValue(min(nz, 2000))

        logger.info(
            "Auto-fit: origin=(%.1f, %.1f, %.1f), dims=(%d, %d, %d) from %d points",
            x0, y0, z0, nx, ny, nz, len(coords),
        )

    def _auto_suggest_size(self):
        """Suggest block sizes from median composite spacing."""
        reg = self._get_registry()
        if reg is None:
            return
        df = reg.get_data("composites", copy_data=False)
        if df is None or (hasattr(df, "empty") and df.empty):
            df = reg.get_data("drillhole_data", copy_data=False)
        if df is None or not hasattr(df, "columns"):
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        for col_name, spin in [("X", self.dx_spin), ("Y", self.dy_spin), ("Z", self.dz_spin)]:
            col = None
            for c in df.columns:
                if c.lower() == col_name.lower():
                    col = c
                    break
            if col is None:
                continue
            vals = df[col].dropna().values
            unique_vals = np.sort(np.unique(np.round(vals, 4)))
            if len(unique_vals) > 1:
                diffs = np.diff(unique_vals)
                median_spacing = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 10.0
                suggested = max(0.5, round(median_spacing / 2.0, 2))
                spin.setValue(suggested)

    # ------------------------------------------------------------------
    # Apply / Clear
    # ------------------------------------------------------------------

    def _apply(self):
        """Validate and register the block model definition."""
        from ..models.block_model_definition import BlockModelDefinition

        origin = (self.x0_spin.value(), self.y0_spin.value(), self.z0_spin.value())
        spacing = (self.dx_spin.value(), self.dy_spin.value(), self.dz_spin.value())
        dims = (self.nx_spin.value(), self.ny_spin.value(), self.nz_spin.value())

        defn = BlockModelDefinition.from_params(origin, spacing, dims)
        errors = defn.validate()
        if errors:
            QMessageBox.critical(
                self, "Validation Failed",
                "Cannot create block model:\n\n" + "\n".join(errors),
            )
            return

        total = defn.n_blocks
        if total > _HARD_CAP:
            QMessageBox.critical(
                self, "Too Many Blocks",
                f"Block model has {total:,} blocks (limit: {_HARD_CAP:,}).\n"
                "Increase block size or reduce extent.",
            )
            return

        if total > _WARN_BLOCKS:
            reply = QMessageBox.question(
                self, "Large Block Model",
                f"Block model has {total:,} blocks.\n"
                f"Memory: ~{defn.memory_per_property_mb * 10:.0f} MB for 10 properties.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Warn about stale results
        reg = self._get_registry()
        if reg is not None:
            old_defn = reg.get_block_model_definition()
            if old_defn is not None and not old_defn.geometry_matches(defn):
                reply = QMessageBox.warning(
                    self, "Definition Changed",
                    "Changing block model definition will invalidate existing "
                    "estimation results on the old grid.\n\n"
                    "Previous results will be cleared. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                # Clear stale block models from registry
                reg.clear_block_model()

            reg.register_block_model_definition(defn)

        self._refresh_info_bar()
        logger.info("Applied block model definition: %s", defn)

    def _clear(self):
        reg = self._get_registry()
        if reg is not None:
            reg.clear_block_model_definition()
        self._refresh_info_bar()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_registry(self):
        if self.main_window is not None:
            ctrl = getattr(self.main_window, "controller", None)
            if ctrl is not None:
                return getattr(ctrl, "registry", None)
            return getattr(self.main_window, "registry", None)
        # Fallback to singleton
        try:
            from ..core.data_registry import DataRegistry
            return DataRegistry.instance()
        except Exception:
            return None

    def populate_from_definition(self, defn):
        """Fill spinboxes from an existing BlockModelDefinition."""
        if defn is None:
            return
        self.x0_spin.setValue(defn.origin[0])
        self.y0_spin.setValue(defn.origin[1])
        self.z0_spin.setValue(defn.origin[2])
        self.dx_spin.setValue(defn.dx)
        self.dy_spin.setValue(defn.dy)
        self.dz_spin.setValue(defn.dz)
        self.nx_spin.setValue(defn.nx)
        self.ny_spin.setValue(defn.ny)
        self.nz_spin.setValue(defn.nz)
        self._update_summary()
        self._refresh_info_bar()
