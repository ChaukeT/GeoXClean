"""
Method Comparison Dialog.

Allows users to select two estimation results, compute block-by-block
differences, view statistics, and render the difference map in the 3D viewer
with a diverging colormap centred on zero.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QGroupBox, QMessageBox,
)
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class ComparisonDialog(QDialog):
    """Compare two estimation results side by side.

    Parameters
    ----------
    registry : DataRegistry
        Used to list available block models and their properties.
    vis_controller : optional
        If provided, difference maps are sent to the 3D viewer.
    parent : QWidget, optional
    """

    def __init__(self, registry, vis_controller=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Method Comparison")
        self.resize(700, 500)
        self._registry = registry
        self._vis = vis_controller
        self._build_ui()
        self._populate_combos()

    # ── UI ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Selection group
        sel = QGroupBox("Select Two Results")
        form = QFormLayout(sel)

        self._combo_a = QComboBox()
        self._combo_b = QComboBox()
        form.addRow("Result A:", self._combo_a)
        form.addRow("Result B:", self._combo_b)

        self._prop_combo_a = QComboBox()
        self._prop_combo_b = QComboBox()
        form.addRow("Property A:", self._prop_combo_a)
        form.addRow("Property B:", self._prop_combo_b)

        self._combo_a.currentIndexChanged.connect(
            lambda: self._populate_props(self._combo_a, self._prop_combo_a)
        )
        self._combo_b.currentIndexChanged.connect(
            lambda: self._populate_props(self._combo_b, self._prop_combo_b)
        )

        layout.addWidget(sel)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_compute = QPushButton("Compute Difference (A \u2212 B)")
        self._btn_compute.clicked.connect(self._compute_difference)
        btn_row.addWidget(self._btn_compute)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Results table
        res = QGroupBox("Comparison Statistics")
        rl = QVBoxLayout(res)
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        rl.addWidget(self._table)
        layout.addWidget(res)

        # Summary table for both results
        summ = QGroupBox("Individual Result Summaries")
        sl = QVBoxLayout(summ)
        self._summary_table = QTableWidget(0, 3)
        self._summary_table.setHorizontalHeaderLabels(["Metric", "Result A", "Result B"])
        self._summary_table.horizontalHeader().setStretchLastSection(True)
        sl.addWidget(self._summary_table)
        layout.addWidget(summ)

    # ── Data ─────────────────────────────────────────────────────────

    def _populate_combos(self) -> None:
        """Fill result combos from registry block model list."""
        if self._registry is None:
            return
        try:
            models = self._registry.get_block_model_list()
        except Exception:
            models = []

        self._combo_a.clear()
        self._combo_b.clear()
        self._models = {}

        for m in models:
            mid = m.get("model_id", "?")
            src = m.get("source_panel", "?")
            label = f"{src} ({mid})"
            self._combo_a.addItem(label, mid)
            self._combo_b.addItem(label, mid)
            self._models[mid] = m

        # Auto-select different items if possible
        if self._combo_b.count() > 1:
            self._combo_b.setCurrentIndex(1)

    def _populate_props(self, model_combo: QComboBox, prop_combo: QComboBox) -> None:
        """Fill property combo from selected block model's columns."""
        prop_combo.clear()
        mid = model_combo.currentData()
        if mid is None:
            return
        try:
            bm = self._registry.get_block_model(model_id=mid, copy_data=False)
            if bm is None:
                return
            if isinstance(bm, pd.DataFrame):
                skip = {"X", "Y", "Z", "XC", "YC", "ZC"}
                for col in bm.columns:
                    if col.upper() not in skip:
                        prop_combo.addItem(col)
            elif hasattr(bm, "cell_data"):
                for key in bm.cell_data.keys():
                    prop_combo.addItem(key)
        except Exception as exc:
            logger.debug("Failed to populate properties: %s", exc)

    # ── Computation ──────────────────────────────────────────────────

    def _compute_difference(self) -> None:
        mid_a = self._combo_a.currentData()
        mid_b = self._combo_b.currentData()
        prop_a = self._prop_combo_a.currentText()
        prop_b = self._prop_combo_b.currentText()

        if not mid_a or not mid_b or not prop_a or not prop_b:
            QMessageBox.warning(self, "Selection Required",
                                "Please select two results and a property from each.")
            return

        try:
            bm_a = self._registry.get_block_model(model_id=mid_a, copy_data=False)
            bm_b = self._registry.get_block_model(model_id=mid_b, copy_data=False)
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", str(exc))
            return

        vals_a = self._extract_values(bm_a, prop_a)
        vals_b = self._extract_values(bm_b, prop_b)

        if vals_a is None or vals_b is None:
            QMessageBox.warning(self, "Property Error",
                                "Could not extract values from one or both results.")
            return

        if len(vals_a) != len(vals_b):
            QMessageBox.warning(
                self, "Size Mismatch",
                f"Result A has {len(vals_a)} blocks, Result B has {len(vals_b)}. "
                "Cannot compute block-by-block difference on different grids.",
            )
            return

        diff = vals_a - vals_b
        valid = np.isfinite(diff)
        n_valid = int(valid.sum())

        if n_valid == 0:
            QMessageBox.warning(self, "No Valid Blocks",
                                "No blocks have finite values in both results.")
            return

        d = diff[valid]

        # Populate difference stats table
        stats = [
            ("Blocks compared", f"{n_valid}"),
            ("Mean difference (A\u2212B)", f"{np.mean(d):.4f}"),
            ("Median difference", f"{np.median(d):.4f}"),
            ("Std deviation", f"{np.std(d):.4f}"),
            ("Min difference", f"{np.min(d):.4f}"),
            ("Max difference", f"{np.max(d):.4f}"),
            ("Max |difference|", f"{np.max(np.abs(d)):.4f}"),
            ("RMSD", f"{np.sqrt(np.mean(d**2)):.4f}"),
        ]
        self._table.setRowCount(len(stats))
        for i, (name, val) in enumerate(stats):
            self._table.setItem(i, 0, QTableWidgetItem(name))
            self._table.setItem(i, 1, QTableWidgetItem(val))

        # Populate individual summaries
        summaries = [
            ("Count (finite)", f"{int(np.isfinite(vals_a).sum())}", f"{int(np.isfinite(vals_b).sum())}"),
            ("Mean", f"{np.nanmean(vals_a):.4f}", f"{np.nanmean(vals_b):.4f}"),
            ("Std", f"{np.nanstd(vals_a):.4f}", f"{np.nanstd(vals_b):.4f}"),
            ("Min", f"{np.nanmin(vals_a):.4f}", f"{np.nanmin(vals_b):.4f}"),
            ("Max", f"{np.nanmax(vals_a):.4f}", f"{np.nanmax(vals_b):.4f}"),
        ]
        self._summary_table.setRowCount(len(summaries))
        for i, (name, va, vb) in enumerate(summaries):
            self._summary_table.setItem(i, 0, QTableWidgetItem(name))
            self._summary_table.setItem(i, 1, QTableWidgetItem(va))
            self._summary_table.setItem(i, 2, QTableWidgetItem(vb))

        # Send difference to 3D viewer
        self._render_difference(bm_a, diff, prop_a, prop_b)

        logger.info(
            "Comparison: %s vs %s — mean_diff=%.4f, RMSD=%.4f, n=%d",
            prop_a, prop_b, np.mean(d), np.sqrt(np.mean(d**2)), n_valid,
        )

    def _extract_values(self, bm, prop_name: str) -> Optional[np.ndarray]:
        if bm is None:
            return None
        if isinstance(bm, pd.DataFrame):
            if prop_name in bm.columns:
                return bm[prop_name].to_numpy(dtype=float)
        elif hasattr(bm, "cell_data") and prop_name in bm.cell_data:
            return np.asarray(bm.cell_data[prop_name], dtype=float)
        return None

    def _render_difference(
        self, bm_a, diff: np.ndarray, prop_a: str, prop_b: str,
    ) -> None:
        """Add the difference as a new property and render with diverging colormap."""
        if self._vis is None:
            return
        try:
            import pyvista as pv

            diff_name = f"DIFF_{prop_a}_vs_{prop_b}"

            # Try to add to existing mesh
            if hasattr(bm_a, "cell_data"):
                grid = bm_a.copy()
                grid.cell_data[diff_name] = diff
            elif isinstance(bm_a, pd.DataFrame) and {"X", "Y", "Z"}.issubset(bm_a.columns):
                coords = bm_a[["X", "Y", "Z"]].to_numpy()
                grid = pv.PolyData(coords)
                grid[diff_name] = diff
            else:
                logger.warning("Cannot render difference — unknown block model format")
                return

            # Symmetric color range centred on zero
            abs_max = float(np.nanmax(np.abs(diff[np.isfinite(diff)])))
            clim = (-abs_max, abs_max)

            if hasattr(self._vis, "apply_results_to_model"):
                self._vis.apply_results_to_model({
                    "visualization": {
                        "mesh": grid,
                        "layer_name": diff_name,
                        "property": diff_name,
                    },
                    "metadata": {
                        "variable": diff_name,
                        "colormap": "RdBu",
                        "clim": clim,
                    },
                })
                logger.info("Rendered difference map: %s", diff_name)

        except Exception as exc:
            logger.warning("Failed to render difference: %s", exc)
