"""
DRILLHOLE PLOTTING PANEL

Provides UI for generating downhole plots, strip logs, and fence diagrams
for drillhole data visualization. Uses Matplotlib for 2D plotting.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path

# Matplotlib integration (backend is set in main.py)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFileDialog, QMessageBox, QWidget
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal

from .base_analysis_panel import BaseAnalysisPanel
from ..drillholes.datamodel import DrillholeDatabase

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)

# =========================================================
# PLOTTER CLASSES (Internal Implementation)
# =========================================================


class DownholePlotter:
    def __init__(self, db: DrillholeDatabase):
        self.db = db

    def create_downhole_plot(self, hole_id: str, element: str, fig: Figure):
        """Generate a downhole line plot for a specific element."""
        ax = fig.add_subplot(111)
        
        # Extract data for hole
        depths = []
        values = []
        
        assays = self.db.get_assays_for(hole_id) if hasattr(self.db, "get_assays_for") else self.db.assays
        if isinstance(assays, pd.DataFrame):
            if not assays.empty and element in assays.columns:
                for _, row in assays.iterrows():
                    val = row.get(element)
                    if pd.notna(val):
                        mid_depth = (float(row.get("depth_from", 0.0)) + float(row.get("depth_to", 0.0))) / 2
                        depths.append(mid_depth)
                        values.append(float(val))
        else:
            for assay in assays:
                if str(assay.hole_id) == hole_id:
                    val = assay.values.get(element)
                    if val is not None:
                        mid_depth = (assay.depth_from + assay.depth_to) / 2
                        depths.append(mid_depth)
                        values.append(val)
        
        if not depths:
            ax.text(0.5, 0.5, f"No data for {element} in {hole_id}", ha='center')
            return

        # Sort by depth
        sorted_indices = np.argsort(depths)
        depths = np.array(depths)[sorted_indices]
        values = np.array(values)[sorted_indices]

        ax.plot(values, depths, marker='o', linestyle='-', markersize=4, color='#4fc3f7')
        ax.set_ylim(max(depths), min(depths)) # Invert Y for depth
        ax.set_xlabel(f"{element} Value")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Downhole Plot: {hole_id} - {element}")
        ax.grid(True, linestyle='--', alpha=0.5)


class StripLogPlotter:
    def __init__(self, db: DrillholeDatabase):
        self.db = db

    def create_strip_log(self, hole_id: str, elements: List[str], fig: Figure):
        """Generate a strip log (lithology + grade)."""
        # Layout: Lithology track + Grade tracks
        n_tracks = 1 + len(elements)
        axs = fig.subplots(1, n_tracks, sharey=True)
        if n_tracks == 1: axs = [axs]
        
        # 1. Lithology Track
        lith_ax = axs[0]
        lith_data = []
        lithology = self.db.get_lithology_for(hole_id) if hasattr(self.db, "get_lithology_for") else self.db.lithology
        if isinstance(lithology, pd.DataFrame):
            for _, row in lithology.iterrows():
                depth_from = row.get("depth_from")
                depth_to = row.get("depth_to")
                if pd.notna(depth_from) and pd.notna(depth_to):
                    lith_data.append((float(depth_from), float(depth_to), str(row.get("lith_code", "Unknown"))))
        else:
            for lith in lithology:
                if str(lith.hole_id) == hole_id:
                    lith_data.append((lith.depth_from, lith.depth_to, lith.lith_code))
        
        # Plot simple colored bars for lithology
        # In a real app, we'd map codes to colors/patterns
        unique_liths = sorted(list(set(d[2] for d in lith_data)))
        lith_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_liths)))
        color_map = dict(zip(unique_liths, lith_colors))
        
        for start, end, code in lith_data:
            lith_ax.barh(y=(start+end)/2, width=1, height=end-start, color=color_map.get(code, 'gray'), align='center')
            # Add text label
            lith_ax.text(0.5, (start+end)/2, code, ha='center', va='center', fontsize=8, color='white')
            
        lith_ax.set_ylim(max([d[1] for d in lith_data], default=100), 0)
        lith_ax.set_title("Lithology")
        lith_ax.set_xticks([])
        
        # 2. Grade Tracks
        for i, elem in enumerate(elements):
            ax = axs[i+1]
            depths = []
            values = []
            assays = self.db.get_assays_for(hole_id) if hasattr(self.db, "get_assays_for") else self.db.assays
            if isinstance(assays, pd.DataFrame):
                if not assays.empty and elem in assays.columns:
                    for _, row in assays.iterrows():
                        val = row.get(elem)
                        if pd.notna(val):
                            depths.append((float(row.get("depth_from", 0.0)) + float(row.get("depth_to", 0.0))) / 2)
                            values.append(float(val))
            else:
                for assay in assays:
                    if str(assay.hole_id) == hole_id:
                        val = assay.values.get(elem)
                        if val is not None:
                            depths.append((assay.depth_from + assay.depth_to) / 2)
                            values.append(val)
            
            if depths:
                # Step plot for grades
                indices = np.argsort(depths)
                d_sorted = np.array(depths)[indices]
                v_sorted = np.array(values)[indices]
                ax.step(v_sorted, d_sorted, where='mid', color='red')
                ax.set_title(elem)
                ax.grid(True, linestyle=':', alpha=0.5)
            else:
                ax.text(0.5, 0.5, "No Data", ha='center')

        fig.suptitle(f"Strip Log: {hole_id}")


class FenceDiagramPlotter:
    def __init__(self, db: DrillholeDatabase):
        self.db = db

    def create_fence_diagram(self, hole_ids: List[str], element: str, fig: Figure):
        """Generate a simple 2D fence diagram (projected section)."""
        ax = fig.add_subplot(111)
        
        # Simple projection: flatten onto X-Z plane (Distance along section vs Depth)
        # For a real fence diagram, we'd project onto a section line.
        # Here we just stack them side-by-side with spacing.
        
        spacing = 20.0 # Gap between holes
        offset = 0.0
        
        for hid in hole_ids:
            depths = []
            values = []
            
            # Get collar Z to adjust relative depth
            collar_z = 0.0
            collar = self.db.get_collar(hid) if hasattr(self.db, "get_collar") else None
            if collar is not None:
                collar_z = float(collar.get("z", 0.0))
            else:
                for col in self.db.collars:
                    if str(col.hole_id) == hid:
                        collar_z = col.z
                        break
            
            assays = self.db.get_assays_for(hid) if hasattr(self.db, "get_assays_for") else self.db.assays
            if isinstance(assays, pd.DataFrame):
                if not assays.empty and element in assays.columns:
                    for _, row in assays.iterrows():
                        val = row.get(element)
                        if pd.notna(val):
                            mid_depth = (float(row.get("depth_from", 0.0)) + float(row.get("depth_to", 0.0))) / 2
                            # Plot relative elevation
                            depths.append(collar_z - mid_depth)
                            values.append(float(val))
            else:
                for assay in assays:
                    if str(assay.hole_id) == hid:
                        val = assay.values.get(element)
                        if val is not None:
                            mid_depth = (assay.depth_from + assay.depth_to) / 2
                            # Plot relative elevation
                            depths.append(collar_z - mid_depth)
                            values.append(val)
            
            if depths:
                # Normalize values for width visualization (log or linear)
                # Plot as a vertical trace with width = grade
                v_norm = np.array(values) / (max(values) if max(values) > 0 else 1.0) * 5.0 # Scale width
                
                # Draw trace line
                ax.plot([offset]*len(depths), depths, 'k-', alpha=0.3)
                
                # Draw "bars" or filled curve
                ax.fill_betweenx(depths, offset, offset + v_norm, color='green', alpha=0.6)
                ax.text(offset, max(depths), hid, ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            offset += spacing
            
        ax.set_xlabel("Hole Sequence")
        ax.set_ylabel("Elevation (Z)")
        ax.set_title(f"Fence Diagram: {element}")
        ax.set_xticks([])


# =========================================================
# PANEL IMPLEMENTATION
# =========================================================


class DrillholePlottingPanel(BaseAnalysisPanel):

    task_name = "drillhole_plotting"
    plot_generated = pyqtSignal(str)

    # PanelManager metadata
    PANEL_ID = "DrillholePlottingPanel"
    PANEL_NAME = "Drillhole Plotting Panel"
    PANEL_CATEGORY = PanelCategory.DRILLHOLE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="drillhole_plotting")
        self.setWindowTitle("Drillhole Plotting")
        
        # State
        self.current_database = None
        
        # Internal Plotters
        self.downhole_plotter = None
        self.strip_log_plotter = None
        self.fence_plotter = None
        
        self._build_ui()

        self._init_registry()
        


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
    def _build_ui(self):
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Controls Group
        gb_ctrl = QGroupBox("Plot Controls")
        form = QHBoxLayout(gb_ctrl)
        
        self.db_label = QLabel("No Database Loaded")
        self.db_label.setStyleSheet("color: gray; font-style: italic;")
        form.addWidget(self.db_label)
        
        self.hole_combo = QComboBox()
        self.hole_combo.setMinimumWidth(120)
        form.addWidget(QLabel("Hole:"))
        form.addWidget(self.hole_combo)
        
        self.element_combo = QComboBox()
        self.element_combo.setMinimumWidth(100)
        form.addWidget(QLabel("Element:"))
        form.addWidget(self.element_combo)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Downhole Plot", "Strip Log", "Fence Diagram"])
        form.addWidget(QLabel("Type:"))
        form.addWidget(self.type_combo)
        
        self.btn_gen = QPushButton("Generate")
        self.btn_gen.clicked.connect(self._generate_plot)
        self.btn_gen.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        form.addWidget(self.btn_gen)
        
        layout.addWidget(gb_ctrl)
        
        # Plot Canvas
        self.figure = Figure(figsize=(8, 6), facecolor='#f0f0f0')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Export
        self.btn_export = QPushButton("Export Plot to PDF/PNG")
        self.btn_export.clicked.connect(self._export_plot)
        layout.addWidget(self.btn_export)

    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.drillholeDataLoaded.connect(self._on_data_loaded)
                d = self.registry.get_drillhole_data()
                if d is not None:
                    self._on_data_loaded(d)
        except Exception as e:
            logger.error(f"Failed to connect drillhole data signal: {e}", exc_info=True)

    def _on_data_loaded(self, data):
        """Build database object from registry dictionary."""
        from ..drillholes.registry_utils import build_database_from_registry
        try:
            db = build_database_from_registry(data)
            self.set_database(db)
        except Exception as e:
            logger.error(f"Failed to load database for plotting: {e}")

    def set_database(self, db: DrillholeDatabase):
        self.current_database = db
        if db:
            # Initialize internal plotters
            self.downhole_plotter = DownholePlotter(db)
            self.strip_log_plotter = StripLogPlotter(db)
            self.fence_plotter = FenceDiagramPlotter(db)
            
            # Update UI
            self.db_label.setText(f"DB Loaded: {len(db.collars)} holes")
            self.db_label.setStyleSheet("color: green; font-weight: bold;")
            
            self.hole_combo.clear()
            self.hole_combo.addItems(sorted(db.get_hole_ids()))
            
            self.element_combo.clear()
            elements = set()
            if isinstance(db.assays, pd.DataFrame):
                exclude = {"hole_id", "depth_from", "depth_to", "x", "y", "z", "metadata"}
                for col in db.assays.columns:
                    if col not in exclude and pd.api.types.is_numeric_dtype(db.assays[col]):
                        elements.add(str(col))
            else:
                for a in db.assays:
                    elements.update(a.values.keys())
            self.element_combo.addItems(sorted(list(elements)))

    def _generate_plot(self):
        if not self.current_database:
            return
            
        hole = self.hole_combo.currentText()
        elem = self.element_combo.currentText()
        p_type = self.type_combo.currentText()
        
        self.figure.clear()
        
        try:
            if p_type == "Downhole Plot":
                self.downhole_plotter.create_downhole_plot(hole, elem, self.figure)
            elif p_type == "Strip Log":
                self.strip_log_plotter.create_strip_log(hole, [elem], self.figure)
            elif p_type == "Fence Diagram":
                # For fence, select multiple holes (logic simplified here to current + neighbors?)
                # Just pass current hole list for now
                holes = [self.hole_combo.itemText(i) for i in range(self.hole_combo.count())][:5] # Limit to 5
                self.fence_plotter.create_fence_diagram(holes, elem, self.figure)
                
            self.canvas.draw()
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", str(e))

    def _export_plot(self):
        if not self.figure: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "plot.png", "PNG Image (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Saved", f"Plot saved to {path}")
            self.plot_generated.emit(path)
