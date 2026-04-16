"""
Block Model Filter Panel
=========================

Universal visual-only block model filter that works with any estimated block
model (SGSIM, FastRBF, Kriging, etc.).  Filters are stacked with AND logic
and only affect the 3D display — the full block model data is always preserved.

Filters:
    - Data envelope (distance to nearest drillhole)
    - Grade threshold (min grade)
    - Elevation slice (Z range)
    - Classification filter (JORC categories)

Fixes over original:
    - Envelope filter computes distances on-the-fly via KDTree (Bug #1)
    - Original layer reliably hidden with multiple fallback strategies (Bug #2)
    - Previous filtered layer removed before adding new one (Bug #3)
    - Grade slider uses log-scale mapping for ppm-range data (Bug #4)
    - NaN/invalid blocks excluded by default (new)
    - Auto-applies on checkbox toggle, not just Apply button (UX)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QDoubleSpinBox, QLabel, QPushButton,
    QSlider, QWidget, QFrame, QComboBox, QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from .base_display_panel import BaseDisplayPanel
from .design_tokens import tokens

logger = logging.getLogger(__name__)

# ── Theme-aware semantic colors for filter panel ────────────────────
_c = tokens.colors()
_CLR_HINT = _c.TEXT_TERTIARY          # hint / description text
_CLR_SECONDARY = _c.TEXT_SECONDARY    # info labels, summary
_CLR_SUCCESS = _c.STATUS_SUCCESS      # stats labels, positive feedback
_CLR_ERROR = _c.STATUS_ERROR          # error / warning states
_CLR_WARNING = _c.STATUS_WARNING      # moderate warning
_CLR_ACCENT = _c.ACCENT               # primary action button


class BlockModelFilterPanel(BaseDisplayPanel):
    """Visual-only block model filter panel for the 3D viewer."""

    PANEL_ID = "BlockModelFilterPanel"

    # Emitted when filters change — payload is the filter spec dict
    filtersChanged = pyqtSignal(dict)

    # ------------------------------------------------------------------ init
    def __init__(self, parent: Optional[Any] = None):
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(250)
        self._debounce_timer.timeout.connect(self._apply_filters)

        # State
        self._full_grid = None
        self._filtered_actor_name = None
        self._filtered_layer_name = None
        self._available_scalars: List[str] = []
        self._scalar_ranges: Dict[str, tuple] = {}

        # FIX #1: drillhole points for on-the-fly distance computation
        self._drillhole_points: Optional[np.ndarray] = None
        self._cached_distances: Optional[np.ndarray] = None

        super().__init__(parent=parent, panel_id=self.PANEL_ID)

    # ------------------------------------------------------------ UI build
    def setup_ui(self) -> None:
        root = self.main_layout
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        title = QLabel("Block Model Filter")
        title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        root.addWidget(title)

        hint = QLabel("Visual-only — full block model preserved for reporting.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_CLR_HINT}; font-size: 8pt;")
        root.addWidget(hint)

        # ── Active layer selector ────────────────────────────────────
        layer_grp = QGroupBox("Active Block Model")
        layer_lay = QVBoxLayout(layer_grp)
        layer_lay.setContentsMargins(8, 10, 8, 8)

        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip("Select which block model layer to filter")
        self.layer_combo.currentTextChanged.connect(self._on_layer_selected)
        layer_lay.addWidget(self.layer_combo)

        self.layer_info_label = QLabel("No block model loaded")
        self.layer_info_label.setWordWrap(True)
        self.layer_info_label.setStyleSheet(f"color: {_CLR_SECONDARY}; font-size: 8pt;")
        layer_lay.addWidget(self.layer_info_label)

        refresh_row = QHBoxLayout()
        self.refresh_layers_btn = QPushButton("Refresh Layers")
        self.refresh_layers_btn.setToolTip(
            "Scan renderer for available block model layers"
        )
        self.refresh_layers_btn.clicked.connect(self._refresh_layer_list)
        refresh_row.addWidget(self.refresh_layers_btn)
        refresh_row.addStretch()
        layer_lay.addLayout(refresh_row)

        root.addWidget(layer_grp)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        self._filter_layout = QVBoxLayout(container)
        self._filter_layout.setSpacing(10)

        self._build_envelope_group()
        self._build_grade_group()
        self._build_elevation_group()
        self._build_classification_group()
        self._build_nan_group()

        self._filter_layout.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        # Action buttons
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setStyleSheet(
            f"background-color: {_CLR_ACCENT}; color: {_c.TEXT_ON_ACCENT}; "
            "font-weight: bold; padding: 8px;"
        )
        self.apply_btn.clicked.connect(self._apply_filters)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_filters)

        btn_row.addWidget(self.apply_btn, stretch=2)
        btn_row.addWidget(self.reset_btn, stretch=1)
        root.addLayout(btn_row)

        # Summary
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            f"color: {_CLR_SECONDARY}; font-size: 9pt; margin-top: 4px;"
        )
        root.addWidget(self.summary_label)

    # ── Envelope filter ──────────────────────────────────────────────────

    def _build_envelope_group(self):
        grp = QGroupBox("Data Envelope")
        grp.setCheckable(True)
        grp.setChecked(False)
        grp.toggled.connect(self._schedule_filter)
        self.envelope_group = grp

        lay = QVBoxLayout(grp)

        desc = QLabel("Hide blocks far from drillhole data")
        desc.setStyleSheet(f"color: {_CLR_HINT}; font-size: 8pt;")
        lay.addWidget(desc)

        # Drillhole status indicator
        self.dh_status_label = QLabel("No drillhole data loaded")
        self.dh_status_label.setStyleSheet(f"color: {_CLR_ERROR}; font-size: 8pt;")
        lay.addWidget(self.dh_status_label)

        # Distance slider
        row = QHBoxLayout()
        row.addWidget(QLabel("Max distance:"))
        self.envelope_slider = QSlider(Qt.Orientation.Horizontal)
        self.envelope_slider.setRange(10, 1000)
        self.envelope_slider.setValue(200)
        self.envelope_slider.setTickInterval(50)
        self.envelope_slider.valueChanged.connect(self._on_envelope_slider)
        row.addWidget(self.envelope_slider, stretch=1)
        self.envelope_value_label = QLabel("200 m")
        self.envelope_value_label.setMinimumWidth(50)
        row.addWidget(self.envelope_value_label)
        lay.addLayout(row)

        self.envelope_stats = QLabel("")
        self.envelope_stats.setStyleSheet(f"color: {_CLR_SUCCESS}; font-size: 8pt;")
        lay.addWidget(self.envelope_stats)

        self._filter_layout.addWidget(grp)

    # ── Grade threshold filter ───────────────────────────────────────────

    def _build_grade_group(self):
        grp = QGroupBox("Grade Threshold")
        grp.setCheckable(True)
        grp.setChecked(False)
        grp.toggled.connect(self._schedule_filter)
        self.grade_group = grp

        lay = QVBoxLayout(grp)

        desc = QLabel("Hide barren blocks below minimum grade")
        desc.setStyleSheet(f"color: {_CLR_HINT}; font-size: 8pt;")
        lay.addWidget(desc)

        form = QFormLayout()
        self.grade_property_combo = QComboBox()
        self.grade_property_combo.setToolTip("Select the estimated grade property")
        self.grade_property_combo.currentTextChanged.connect(
            self._on_grade_property_changed
        )
        form.addRow("Property:", self.grade_property_combo)

        self.grade_min_spin = QDoubleSpinBox()
        self.grade_min_spin.setDecimals(4)
        self.grade_min_spin.setRange(0.0, 1e9)
        self.grade_min_spin.setValue(0.0)
        self.grade_min_spin.setSuffix("")
        self.grade_min_spin.valueChanged.connect(self._schedule_filter)
        form.addRow("Min grade:", self.grade_min_spin)
        lay.addLayout(form)

        # FIX #4: Log-scale grade slider
        self.grade_slider = QSlider(Qt.Orientation.Horizontal)
        self.grade_slider.setRange(0, 1000)
        self.grade_slider.setValue(0)
        self.grade_slider.valueChanged.connect(self._on_grade_slider)
        lay.addWidget(self.grade_slider)

        self.grade_stats = QLabel("")
        self.grade_stats.setStyleSheet(f"color: {_CLR_SUCCESS}; font-size: 8pt;")
        lay.addWidget(self.grade_stats)

        self._filter_layout.addWidget(grp)

    # ── Elevation slice filter ───────────────────────────────────────────

    def _build_elevation_group(self):
        grp = QGroupBox("Elevation Slice")
        grp.setCheckable(True)
        grp.setChecked(False)
        grp.toggled.connect(self._schedule_filter)
        self.elevation_group = grp

        lay = QVBoxLayout(grp)

        desc = QLabel("Clip blocks outside elevation range")
        desc.setStyleSheet(f"color: {_CLR_HINT}; font-size: 8pt;")
        lay.addWidget(desc)

        form = QFormLayout()
        self.z_min_spin = QDoubleSpinBox()
        self.z_min_spin.setDecimals(1)
        self.z_min_spin.setRange(-10000, 10000)
        self.z_min_spin.setValue(0)
        self.z_min_spin.setSuffix(" m")
        self.z_min_spin.valueChanged.connect(self._schedule_filter)
        form.addRow("From RL:", self.z_min_spin)

        self.z_max_spin = QDoubleSpinBox()
        self.z_max_spin.setDecimals(1)
        self.z_max_spin.setRange(-10000, 10000)
        self.z_max_spin.setValue(1000)
        self.z_max_spin.setSuffix(" m")
        self.z_max_spin.valueChanged.connect(self._schedule_filter)
        form.addRow("To RL:", self.z_max_spin)
        lay.addLayout(form)

        # Dual sliders
        row = QHBoxLayout()
        self.z_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_min_slider.valueChanged.connect(self._on_z_min_slider)
        row.addWidget(QLabel("Min"))
        row.addWidget(self.z_min_slider, stretch=1)
        self.z_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_max_slider.valueChanged.connect(self._on_z_max_slider)
        row.addWidget(QLabel("Max"))
        row.addWidget(self.z_max_slider, stretch=1)
        lay.addLayout(row)

        self.elevation_stats = QLabel("")
        self.elevation_stats.setStyleSheet(f"color: {_CLR_SUCCESS}; font-size: 8pt;")
        lay.addWidget(self.elevation_stats)

        self._filter_layout.addWidget(grp)

    # ── Classification filter ────────────────────────────────────────────

    def _build_classification_group(self):
        grp = QGroupBox("Classification Filter")
        grp.setCheckable(True)
        grp.setChecked(False)
        grp.toggled.connect(self._schedule_filter)
        self.classification_group = grp

        lay = QVBoxLayout(grp)

        desc = QLabel("Filter by JORC resource category")
        desc.setStyleSheet(f"color: {_CLR_HINT}; font-size: 8pt;")
        lay.addWidget(desc)

        self.class_measured = QCheckBox("Measured")
        self.class_measured.setChecked(True)
        self.class_measured.toggled.connect(self._schedule_filter)
        lay.addWidget(self.class_measured)

        self.class_indicated = QCheckBox("Indicated")
        self.class_indicated.setChecked(True)
        self.class_indicated.toggled.connect(self._schedule_filter)
        lay.addWidget(self.class_indicated)

        self.class_inferred = QCheckBox("Inferred")
        self.class_inferred.setChecked(False)
        self.class_inferred.toggled.connect(self._schedule_filter)
        lay.addWidget(self.class_inferred)

        self.classification_stats = QLabel("")
        self.classification_stats.setStyleSheet(f"color: {_CLR_SUCCESS}; font-size: 8pt;")
        lay.addWidget(self.classification_stats)

        self._filter_layout.addWidget(grp)

    # ── NaN/Invalid filter (NEW) ─────────────────────────────────────────

    def _build_nan_group(self):
        grp = QGroupBox("Exclude Invalid Blocks")
        grp.setCheckable(True)
        grp.setChecked(True)  # ON by default
        grp.toggled.connect(self._schedule_filter)
        self.nan_group = grp

        lay = QVBoxLayout(grp)
        desc = QLabel(
            "Remove blocks with NaN, negative, or zero estimated grades. "
            "These are blocks that were not estimated (outside search range)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {_CLR_HINT}; font-size: 8pt;")
        lay.addWidget(desc)

        self.nan_stats = QLabel("")
        self.nan_stats.setStyleSheet(f"color: {_CLR_SUCCESS}; font-size: 8pt;")
        lay.addWidget(self.nan_stats)

        self._filter_layout.addWidget(grp)

    # ================================================================
    # Data binding
    # ================================================================

    def set_block_model_grid(self, grid, layer_name: str):
        """
        Receive the full (unfiltered) block model grid.

        Parameters
        ----------
        grid : pv.DataSet
            Full PyVista grid (ImageData, RectilinearGrid, UnstructuredGrid)
        layer_name : str
            Name of the layer in the renderer
        """
        self._full_grid = grid
        self._filtered_actor_name = layer_name
        self._cached_distances = None  # invalidate distance cache

        # Discover available scalar arrays
        self._available_scalars = []
        self._scalar_ranges = {}
        for source_name in ('cell_data', 'point_data'):
            source = getattr(grid, source_name, None)
            if source is None:
                continue
            for key in source.keys():
                arr = source[key]
                if not np.issubdtype(arr.dtype, np.number):
                    continue
                self._available_scalars.append(key)
                finite = arr[np.isfinite(arr)]
                if len(finite) > 0:
                    self._scalar_ranges[key] = (
                        float(finite.min()),
                        float(finite.max()),
                    )

        self._populate_grade_combo()
        self._populate_elevation_range()
        self._update_summary_counts()
        self._update_layer_info(layer_name, grid)
        logger.info(
            "BlockModelFilterPanel: received grid with %d cells, "
            "%d scalar arrays",
            grid.n_cells,
            len(self._available_scalars),
        )

    def set_drillhole_points(self, points: np.ndarray):
        """
        Receive drillhole sample XYZ positions for envelope computation.

        Call this from the estimation panel or main_window after drillholes
        are loaded / composites are computed.

        Parameters
        ----------
        points : Nx3 array of drillhole sample midpoints
        """
        self._drillhole_points = np.asarray(points, dtype=float)
        self._cached_distances = None  # invalidate cache
        n = len(self._drillhole_points)
        self.dh_status_label.setText(
            f"<span style='color:{_CLR_SUCCESS}'>{n:,} drillhole samples loaded</span>"
        )

        # Compute and show suggested distance
        if n > 1:
            from scipy.spatial import cKDTree

            tree = cKDTree(self._drillhole_points)
            dd, _ = tree.query(self._drillhole_points, k=2)
            median_spacing = float(np.median(dd[:, 1]))
            suggested = median_spacing * 2.0
            self.envelope_slider.blockSignals(True)
            self.envelope_slider.setValue(int(suggested))
            self.envelope_slider.blockSignals(False)
            self.envelope_value_label.setText(f"{int(suggested)} m")
            self.dh_status_label.setText(
                f"<span style='color:{_CLR_SUCCESS}'>"
                f"{n:,} samples | spacing ~{median_spacing:.0f}m | "
                f"suggested: {suggested:.0f}m</span>"
            )
        logger.info(
            "BlockModelFilter: received %d drillhole points", n
        )

    def _update_layer_info(self, layer_name: str, grid):
        """Update the layer info label and ensure the combo shows this layer."""
        idx = self.layer_combo.findText(layer_name)
        if idx < 0:
            self.layer_combo.blockSignals(True)
            self.layer_combo.addItem(layer_name)
            idx = self.layer_combo.count() - 1
            self.layer_combo.blockSignals(False)
        self.layer_combo.blockSignals(True)
        self.layer_combo.setCurrentIndex(idx)
        self.layer_combo.blockSignals(False)

        n_cells = grid.n_cells if grid is not None else 0
        n_scalars = len(self._available_scalars)
        bounds = grid.bounds if grid is not None else (0, 0, 0, 0, 0, 0)
        x_ext = bounds[1] - bounds[0]
        y_ext = bounds[3] - bounds[2]
        z_ext = bounds[5] - bounds[4]
        self.layer_info_label.setText(
            f"{n_cells:,} cells  |  {n_scalars} properties\n"
            f"Extent: {x_ext:.0f} x {y_ext:.0f} x {z_ext:.0f} m"
        )

    def _on_layer_selected(self, layer_name: str):
        """User picked a different layer from the combo."""
        if not layer_name:
            return
        renderer = self.renderer
        if renderer is None:
            return
        active_layers = getattr(renderer, 'active_layers', {})
        layer_info = active_layers.get(layer_name)
        if layer_info is None:
            return
        grid = self._extract_grid_from_layer(renderer, layer_name, layer_info)
        if grid is not None and hasattr(grid, 'n_cells') and grid.n_cells > 0:
            logger.info(
                "BlockModelFilter: selected layer '%s' with %d cells",
                layer_name, grid.n_cells,
            )
            self.set_block_model_grid(grid, layer_name)

    def _refresh_layer_list(self):
        """Scan renderer for all block model layers and populate the combo."""
        renderer = self.renderer
        if renderer is None:
            self.layer_info_label.setText(
                "No renderer available — bind controller first"
            )
            return
        active_layers = getattr(renderer, 'active_layers', {})
        block_tags = (
            "SGSIM", "Block Model", "Kriging", "RBF", "ARBF", "Filtered:",
        )
        block_types = ('blocks', 'volume')

        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        found_layers = []
        for name, info in active_layers.items():
            lt = info.get('type', info.get('layer_type', ''))
            is_block = lt in block_types or any(
                tag in name for tag in block_tags
            )
            if is_block:
                self.layer_combo.addItem(name)
                found_layers.append(name)
        self.layer_combo.blockSignals(False)

        if not found_layers:
            self.layer_info_label.setText(
                "No block model layers found in 3D viewer"
            )
            return

        current = self._filtered_actor_name
        if current and current in found_layers:
            self.layer_combo.setCurrentText(current)
        else:
            self._on_layer_selected(found_layers[0])

    def _populate_grade_combo(self):
        """Fill the grade property combo with available numeric scalars."""
        self.grade_property_combo.blockSignals(True)
        self.grade_property_combo.clear()
        for name in self._available_scalars:
            self.grade_property_combo.addItem(name)
        self.grade_property_combo.blockSignals(False)
        if self._available_scalars:
            self._on_grade_property_changed(self._available_scalars[0])

    def _populate_elevation_range(self):
        """Set elevation spin/slider ranges from grid bounds."""
        if self._full_grid is None:
            return
        try:
            bounds = self._full_grid.bounds
            z_min, z_max = bounds[4], bounds[5]
        except Exception:
            z_min, z_max = 0.0, 1000.0

        for spin in (self.z_min_spin, self.z_max_spin):
            spin.blockSignals(True)
        self.z_min_spin.setRange(z_min, z_max)
        self.z_min_spin.setValue(z_min)
        self.z_max_spin.setRange(z_min, z_max)
        self.z_max_spin.setValue(z_max)
        for spin in (self.z_min_spin, self.z_max_spin):
            spin.blockSignals(False)

        steps = max(1, int((z_max - z_min) * 10))
        for slider in (self.z_min_slider, self.z_max_slider):
            slider.blockSignals(True)
            slider.setRange(0, steps)
        self.z_min_slider.setValue(0)
        self.z_max_slider.setValue(steps)
        for slider in (self.z_min_slider, self.z_max_slider):
            slider.blockSignals(False)

    # ================================================================
    # Slider ↔ spin synchronisation
    # ================================================================

    def _on_envelope_slider(self, value: int):
        self.envelope_value_label.setText(f"{value} m")
        self._schedule_filter()

    def _on_grade_slider(self, value: int):
        """FIX #4: Map slider 0-1000 using LOG scale for ppm-range data."""
        prop = self.grade_property_combo.currentText()
        rng = self._scalar_ranges.get(prop)
        if rng is None:
            return
        lo, hi = rng
        # Use log scale if range spans more than 2 orders of magnitude
        if hi > 0 and lo >= 0 and (hi / max(lo, 0.001)) > 100:
            # Log-scale mapping
            log_lo = np.log10(max(lo, 0.1))
            log_hi = np.log10(max(hi, 1.0))
            log_val = log_lo + (log_hi - log_lo) * value / 1000.0
            grade = 10 ** log_val
        else:
            # Linear mapping for small ranges
            grade = lo + (hi - lo) * value / 1000.0
        self.grade_min_spin.blockSignals(True)
        self.grade_min_spin.setValue(grade)
        self.grade_min_spin.blockSignals(False)
        self._schedule_filter()

    def _on_grade_property_changed(self, prop_name: str):
        rng = self._scalar_ranges.get(prop_name)
        if rng is None:
            return
        lo, hi = rng
        self.grade_min_spin.blockSignals(True)
        self.grade_min_spin.setRange(lo, hi)
        self.grade_min_spin.setValue(lo)
        self.grade_min_spin.blockSignals(False)
        self.grade_slider.blockSignals(True)
        self.grade_slider.setValue(0)
        self.grade_slider.blockSignals(False)

    def _on_z_min_slider(self, value: int):
        if self._full_grid is None:
            return
        z_min = self._full_grid.bounds[4]
        z_max = self._full_grid.bounds[5]
        steps = max(1, self.z_min_slider.maximum())
        z = z_min + (z_max - z_min) * value / steps
        self.z_min_spin.blockSignals(True)
        self.z_min_spin.setValue(z)
        self.z_min_spin.blockSignals(False)
        self._schedule_filter()

    def _on_z_max_slider(self, value: int):
        if self._full_grid is None:
            return
        z_min = self._full_grid.bounds[4]
        z_max = self._full_grid.bounds[5]
        steps = max(1, self.z_max_slider.maximum())
        z = z_min + (z_max - z_min) * value / steps
        self.z_max_spin.blockSignals(True)
        self.z_max_spin.setValue(z)
        self.z_max_spin.blockSignals(False)
        self._schedule_filter()

    # ================================================================
    # Distance computation (FIX #1)
    # ================================================================

    def _get_or_compute_distances(self, grid, centers: np.ndarray) -> Optional[np.ndarray]:
        """
        Get distance-to-nearest-drillhole for each block.

        Strategy:
        1. Check if grid already has a DistToHole scalar -> use it
        2. Check if we have drillhole points -> compute via KDTree
        3. Return None if neither available
        """
        # Strategy 1: pre-computed on grid
        dist_arr = self._find_scalar(grid, [
            "DistToHole", "dist_to_hole", "DIST_TO_HOLE",
            "distance_to_data", "DistToData", "DISTANCE",
        ])
        if dist_arr is not None:
            return dist_arr

        # Strategy 2: compute from drillhole points
        if self._drillhole_points is not None and len(self._drillhole_points) > 0:
            # Use cache if grid hasn't changed
            if (
                self._cached_distances is not None
                and len(self._cached_distances) == len(centers)
            ):
                return self._cached_distances

            logger.info(
                "BlockModelFilter: computing distances for %d blocks "
                "from %d drillhole points...",
                len(centers), len(self._drillhole_points),
            )

            try:
                from scipy.spatial import cKDTree

                tree = cKDTree(self._drillhole_points)
                distances, _ = tree.query(centers, k=1, workers=-1)
                self._cached_distances = distances.astype(np.float32)

                # Also attach to grid so other panels can use it
                try:
                    if hasattr(grid, 'cell_data'):
                        grid.cell_data["DistToHole"] = self._cached_distances
                except Exception:
                    pass

                logger.info(
                    "BlockModelFilter: distances computed. "
                    "Range: %.0f - %.0f m, median: %.0f m",
                    float(distances.min()),
                    float(distances.max()),
                    float(np.median(distances)),
                )
                return self._cached_distances

            except ImportError:
                logger.error(
                    "BlockModelFilter: scipy not available for KDTree"
                )
                return None
            except Exception as e:
                logger.error(
                    "BlockModelFilter: distance computation failed: %s", e
                )
                return None

        return None

    # ================================================================
    # Filter application (debounced)
    # ================================================================

    def _schedule_filter(self, *_args):
        """Restart the debounce timer so we don't re-filter on every pixel."""
        self._debounce_timer.start()

    def _apply_filters(self):
        """
        Build a combined filter mask and replace the displayed mesh.

        All filters are AND-combined. The original grid is never modified.
        """
        # TODO: REBUILD — block model rendering removed for clean reimplementation
        #
        # Previously this method:
        # - Built combined filter mask and swapped displayed mesh in renderer
        #
        # Rebuild using: PyVista ImageData pipeline
        logger.debug("Block model filter: operation deferred (rendering rebuild in progress)")

    def _reset_filters(self):
        """Disable all filter groups, remove filtered layer, restore original."""
        # TODO: REBUILD — block model rendering removed for clean reimplementation
        #
        # Previously this method:
        # - Disabled all filter groups, removed filtered layer, restored original in renderer
        #
        # Rebuild using: PyVista ImageData pipeline
        logger.debug("Block model filter: operation deferred (rendering rebuild in progress)")
        self._refresh_property_panel()

        n = self._full_grid.n_cells if self._full_grid else 0
        self._update_summary(n, n)

    # ================================================================
    # Renderer interaction (FIX #2 + #3)
    # ================================================================

    def _swap_filtered_layer(self, mesh):
        """
        Replace the displayed mesh with the filtered version.

        FIX #2: Multiple fallback strategies to hide the original.
        FIX #3: Remove previous filtered layer before adding new.
        """
        # TODO: REBUILD — block model rendering removed for clean reimplementation
        #
        # Previously this method:
        # - Swapped renderer layer with filtered mesh, hid original, added new layer
        #
        # Rebuild using: PyVista ImageData pipeline
        logger.debug("Block model filter: operation deferred (rendering rebuild in progress)")

    def _hide_original_layer(self):
        """FIX #2: Hide the original layer using multiple fallback strategies."""
        # TODO: REBUILD — block model rendering removed for clean reimplementation
        #
        # Previously this method:
        # - Hid original block model layer via renderer with multiple fallback strategies
        #
        # Rebuild using: PyVista ImageData pipeline
        logger.debug("Block model filter: operation deferred (rendering rebuild in progress)")

    def _show_original_layer(self):
        """Restore visibility of the original layer."""
        # TODO: REBUILD — block model rendering removed for clean reimplementation
        #
        # Previously this method:
        # - Restored visibility of the original block model layer in renderer
        #
        # Rebuild using: PyVista ImageData pipeline
        logger.debug("Block model filter: operation deferred (rendering rebuild in progress)")

    def _remove_filtered_layer(self):
        """Remove the currently active filtered layer."""
        # TODO: REBUILD — block model rendering removed for clean reimplementation
        #
        # Previously this method:
        # - Removed the filtered layer from renderer via clear_layer or actor removal
        #
        # Rebuild using: PyVista ImageData pipeline
        logger.debug("Block model filter: operation deferred (rendering rebuild in progress)")

    def _refresh_property_panel(self):
        """Tell the property panel to rescan renderer layers."""
        # TODO: REBUILD — block model rendering removed for clean reimplementation
        #
        # Previously this method:
        # - Walked parent hierarchy to find property panel and triggered layer rescan
        #
        # Rebuild using: PyVista ImageData pipeline
        logger.debug("Block model filter: operation deferred (rendering rebuild in progress)")

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _find_scalar(grid, candidates: list):
        """Return the first matching scalar array from cell_data or point_data."""
        for source_name in ('cell_data', 'point_data'):
            source = getattr(grid, source_name, None)
            if source is None:
                continue
            for name in candidates:
                if name in source:
                    return np.asarray(source[name])
        return None

    def _update_summary(self, n_shown: int, n_total: int):
        pct = 100.0 * n_shown / n_total if n_total > 0 else 0.0
        colour = (
            _CLR_SUCCESS if pct >= 50
            else _CLR_WARNING if pct >= 20
            else _CLR_ERROR
        )
        self.summary_label.setText(
            f"<span style='color:{colour}; font-weight:bold'>"
            f"Blocks displayed: {n_shown:,} / {n_total:,} ({pct:.1f}%)"
            f"</span>"
        )

    def _update_summary_counts(self):
        n = self._full_grid.n_cells if self._full_grid else 0
        self._update_summary(n, n)

    def _build_filter_spec(self, n_shown: int, n_total: int) -> dict:
        spec: dict = {"n_shown": n_shown, "n_total": n_total}
        if self.envelope_group.isChecked():
            spec["envelope_max_distance"] = self.envelope_slider.value()
        if self.grade_group.isChecked():
            spec["grade_property"] = self.grade_property_combo.currentText()
            spec["grade_min"] = self.grade_min_spin.value()
        if self.elevation_group.isChecked():
            spec["z_min"] = self.z_min_spin.value()
            spec["z_max"] = self.z_max_spin.value()
        if self.classification_group.isChecked():
            spec["classification"] = {
                "measured": self.class_measured.isChecked(),
                "indicated": self.class_indicated.isChecked(),
                "inferred": self.class_inferred.isChecked(),
            }
        return spec

    # ================================================================
    # Panel lifecycle
    # ================================================================

    def connect_signals(self) -> None:
        pass

    def bind_controller(self, controller) -> None:
        """Override to auto-grab the active block model grid."""
        super().bind_controller(controller)
        self._grab_active_grid()

    def showEvent(self, event):
        """When the panel becomes visible, grab the active grid."""
        super().showEvent(event)
        if self._full_grid is None:
            self._grab_active_grid()

    def _grab_active_grid(self):
        """Scan the renderer's active_layers for a block model grid."""
        self._refresh_layer_list()

    @staticmethod
    def _extract_grid_from_layer(renderer, layer_name: str, layer_info: dict):
        """Try to extract a PyVista grid from a renderer layer."""
        grid = (
            layer_info.get('grid')
            or layer_info.get('data')
            or layer_info.get('mesh')
        )
        if grid is not None and hasattr(grid, 'n_cells'):
            return grid

        plotter = getattr(renderer, 'plotter', None)
        if plotter is None:
            return None

        actor = layer_info.get('actor')
        if actor is None:
            return None

        try:
            import pyvista as pv

            mapper = actor.GetMapper()
            if mapper is not None:
                dataset = mapper.GetInput()
                if dataset is not None:
                    grid = pv.wrap(dataset)
                    if hasattr(grid, 'n_cells') and grid.n_cells > 0:
                        return grid
        except Exception:
            pass

        return None

    def refresh(self) -> None:
        """Sync UI from current renderer state."""
        self._update_summary_counts()

    def connect_layer_events(self) -> None:
        """Subscribe to renderer layer changes."""
        if self.controller is None:
            return
        signals = getattr(self.controller, 'signals', None)
        if signals and hasattr(signals, 'layerAdded'):
            try:
                signals.layerAdded.connect(self._on_layer_added)
            except Exception:
                pass

    def _on_layer_added(self, layer_name: str, grid):
        """Auto-detect new block model layers and offer filtering."""
        if grid is not None and hasattr(grid, 'n_cells') and grid.n_cells > 0:
            self.set_block_model_grid(grid, layer_name)
