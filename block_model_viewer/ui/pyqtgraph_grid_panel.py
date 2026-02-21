"""
PyQtGraph-based Grid Viewer with Tri-Orthogonal Grid Planes.

Lightweight OpenGL-accelerated alternative to PyVista for visualizing block models
with orthogonal grid planes (XY, XZ, YZ) intersecting at the data center.
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QSpinBox, QPushButton, QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QLinearGradient

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    # CRITICAL FIX: Removed dangerous 'from OpenGL.GL import *' - not used in this module
    # PyQtGraph's opengl module handles OpenGL imports internally
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

from .base_panel import BaseDialogPanel
from ..models.block_model import BlockModel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


def _normalize(values):
    """Normalize array to [0,1] range, handling edge cases."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.array([])
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10:
        return np.full_like(values, 0.5)
    return (values - vmin) / (vmax - vmin)


def _viridis_rgba(x):
    """
    Approximation of Viridis colormap.
    x: float in [0,1] or array of such values
    Returns: RGBA array (N,4) with float values in [0,1]
    """
    x = np.clip(x, 0, 1)
    
    # Viridis control points (approximate)
    colors = np.array([
        [0.267004, 0.004874, 0.329415, 1.0],
        [0.282623, 0.140926, 0.457517, 1.0],
        [0.253935, 0.265254, 0.529983, 1.0],
        [0.206756, 0.371758, 0.553117, 1.0],
        [0.163625, 0.471133, 0.558148, 1.0],
        [0.127568, 0.566949, 0.550556, 1.0],
        [0.134692, 0.658636, 0.517649, 1.0],
        [0.266941, 0.748751, 0.440573, 1.0],
        [0.477504, 0.821444, 0.318195, 1.0],
        [0.741388, 0.873449, 0.149561, 1.0],
        [0.993248, 0.906157, 0.143936, 1.0]
    ], dtype=np.float32)
    
    # Interpolate
    indices = x * (len(colors) - 1)
    i0 = np.floor(indices).astype(int)
    i1 = np.minimum(i0 + 1, len(colors) - 1)
    alpha = (indices - i0)[:, None]
    
    result = colors[i0] * (1 - alpha) + colors[i1] * alpha
    return result.astype(np.float32)


class _LegendOverlay(QWidget):
    """
    Colorbar legend overlay for PyQtGraph GLViewWidget.
    
    Now uses LegendManager for consistent legend across all views (Step 9).
    """
    
    def __init__(self, parent=None, legend_manager=None):
        super().__init__(parent)
        self.setFixedWidth(90)
        self.vmin = 0.0
        self.vmax = 1.0
        self.title = "Property"
        self.colormap = "viridis"
        self._legend_manager = legend_manager
        
        # Connect to LegendManager if provided
        if self._legend_manager:
            try:
                self._legend_manager.legend_changed.connect(self._on_legend_changed)
            except Exception:
                pass
    


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
    def _on_legend_changed(self, payload: dict):
        """Update legend from LegendManager payload."""
        if not payload:
            return
        self.vmin = payload.get("vmin", self.vmin)
        self.vmax = payload.get("vmax", self.vmax)
        self.title = payload.get("title") or payload.get("property", self.title)
        self.colormap = payload.get("colormap", self.colormap)
        self.update()
        
    def set_range(self, vmin, vmax, title="Property"):
        """DEPRECATED: Use LegendManager instead."""
        self.vmin = vmin
        self.vmax = vmax
        self.title = title
        self.update()
    
    def paintEvent(self, event):
        painter = None
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Use LegendManager colormap if available, otherwise fallback to viridis
            from matplotlib import cm
            try:
                cmap = cm.get_cmap(self.colormap)
            except Exception:
                cmap = cm.get_cmap("viridis")
            
            # Draw colorbar gradient using LegendManager colormap
            gradient = QLinearGradient(0, 0, 0, self.height())
            n_stops = 20
            for i in range(n_stops + 1):
                t = i / n_stops
                rgba = cmap(1.0 - t)  # Reverse: top = high, bottom = low
                color = QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 255)
                gradient.setColorAt(t, color)
            
            painter.fillRect(10, 20, 20, self.height() - 60, gradient)
            
            # Draw labels
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(35, 30, f"{self.vmax:.2f}")
            painter.drawText(35, self.height() // 2, f"{(self.vmin + self.vmax) / 2:.2f}")
            painter.drawText(35, self.height() - 45, f"{self.vmin:.2f}")
            painter.drawText(5, 15, self.title)
        finally:
            if painter is not None:
                painter.end()  # Properly close QPainter to prevent warnings


class PyQtGraphGridPanel(BaseDialogPanel):
    """
    PyQtGraph-based grid viewer with tri-orthogonal grid planes.
    Provides lightweight OpenGL rendering for block models with XY/XZ/YZ reference grids.
    """
    
    def __init__(self, parent=None, legend_manager=None):
        super().__init__(parent)
        self.setWindowTitle("Grid Viewer (PyQtGraph)")
        self.resize(1000, 700)
        
        # Use _block_model (private backing field) - block_model is a read-only @property in BasePanel
        self._block_model: Optional[BlockModel] = None
        self.current_property = None
        self.gl_view = None
        self.scatter_item = None
        self.grid_items = []
        self.legend_overlay = None
        self._legend_manager = legend_manager
        
        if not PYQTGRAPH_AVAILABLE:
            self._show_install_message()
            return
    
    def _show_install_message(self):
        """Show installation instructions if PyQtGraph not available."""
        layout = QVBoxLayout()
        label = QLabel(
            "<h2>PyQtGraph Not Installed</h2>"
            "<p>To use the Grid Viewer, install PyQtGraph and PyOpenGL:</p>"
            "<pre>pip install pyqtgraph PyOpenGL PyOpenGL_accelerate</pre>"
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        self.setLayout(layout)
    
    def setup_ui(self):
        """Create the user interface."""
        main_layout = QVBoxLayout()
        
        # Control panel
        controls = QHBoxLayout()
        
        controls.addWidget(QLabel("Property:"))
        self.property_combo = QComboBox()
        self.property_combo.currentTextChanged.connect(self._update_plot)
        controls.addWidget(self.property_combo)
        
        controls.addWidget(QLabel("Downsample:"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 100)
        self.downsample_spin.setValue(1)
        self.downsample_spin.setSuffix(" blocks")
        self.downsample_spin.valueChanged.connect(self._update_plot)
        controls.addWidget(self.downsample_spin)
        
        controls.addWidget(QLabel("Point Size:"))
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(3)
        self.point_size_spin.valueChanged.connect(self._update_plot)
        controls.addWidget(self.point_size_spin)
        
        self.grid_checkbox = QCheckBox("Show Grid Planes")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.toggled.connect(self._toggle_grids)
        controls.addWidget(self.grid_checkbox)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._update_plot)
        controls.addWidget(refresh_btn)
        
        controls.addStretch()
        
        main_layout.addLayout(controls)
        
        # GLViewWidget for 3D rendering
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor('k')
        main_layout.addWidget(self.gl_view)
        
        # Add RGB axes (X=red, Y=green, Z=blue)
        axis_length = 100
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]), 
                                   color=(1, 0, 0, 1), width=2)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]), 
                                   color=(0, 1, 0, 1), width=2)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]), 
                                   color=(0, 0, 1, 1), width=2)
        self.gl_view.addItem(x_axis)
        self.gl_view.addItem(y_axis)
        self.gl_view.addItem(z_axis)
        
        # Legend overlay (now uses LegendManager for consistency)
        self.legend_overlay = _LegendOverlay(self.gl_view, legend_manager=self._legend_manager)
        self.legend_overlay.setGeometry(self.gl_view.width() - 100, 10, 90, self.gl_view.height() - 20)
        self.legend_overlay.show()
        
        self.setLayout(main_layout)
    
    def set_block_model(self, block_model: BlockModel):
        """Set the block model and update the viewer."""
        self._block_model = block_model  # Use private backing field (property contract)
        self._on_block_model_changed()
        self._update_plot()
    
    def _on_block_model_changed(self):
        """Handle block model change - populate property dropdown."""
        if not self.block_model or not PYQTGRAPH_AVAILABLE:
            return
        
        self.property_combo.blockSignals(True)
        self.property_combo.clear()
        
        # Add numeric properties
        for prop_name in self.block_model.properties.keys():
            values = self.block_model.properties[prop_name]
            if np.issubdtype(values.dtype, np.number):
                self.property_combo.addItem(prop_name)
        
        self.property_combo.blockSignals(False)
        
        if self.property_combo.count() > 0:
            self.property_combo.setCurrentIndex(0)
    
    def _update_plot(self):
        """Update the 3D plot with current settings."""
        # LAG FIX: Skip update if panel is hidden to avoid unnecessary work
        if not self.isVisible():
            return
        if not self.block_model or not PYQTGRAPH_AVAILABLE:
            return
        
        # Get current property
        self.current_property = self.property_combo.currentText()
        if not self.current_property:
            return
        
        # Get positions and values
        positions = np.column_stack([
            self.block_model.x_centroids,
            self.block_model.y_centroids,
            self.block_model.z_centroids
        ])
        
        values = self.block_model.properties[self.current_property]
        
        # Apply downsampling
        downsample = self.downsample_spin.value()
        if downsample > 1:
            indices = np.arange(0, len(positions), downsample)
            positions = positions[indices]
            values = values[indices]
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        positions = positions[valid_mask]
        values = values[valid_mask]
        
        if len(positions) == 0:
            logger.warning(f"No valid data for property '{self.current_property}'")
            return
        
        # Normalize and color
        values_norm = _normalize(values)
        colors = _viridis_rgba(values_norm)  # (N, 4) float32 RGBA
        
        # Remove old scatter if exists
        if self.scatter_item:
            self.gl_view.removeItem(self.scatter_item)
        
        # Create new scatter plot
        # Note: GLScatterPlotItem expects colors as (N,4) float array [0,1]
        self.scatter_item = gl.GLScatterPlotItem(
            pos=positions,
            color=colors,
            size=self.point_size_spin.value(),
            pxMode=True  # Pixel mode for consistent size
        )
        self.gl_view.addItem(self.scatter_item)
        
        # Update legend (if LegendManager not available, use local update)
        vmin, vmax = values.min(), values.max()
        if self._legend_manager:
            # LegendManager will update via signal, but we can also update directly for immediate feedback
            metadata = {
                'vmin': vmin,
                'vmax': vmax,
                'colormap': 'viridis',  # Default, will be overridden by LegendManager
                'title': self.current_property
            }
            self._legend_manager.update_from_property(self.current_property, metadata)
        else:
            # Fallback to local legend update
            self.legend_overlay.set_range(vmin, vmax, self.current_property)
        
        # Update grid planes
        self._update_grids(positions)
        
        # Auto-fit camera
        center = positions.mean(axis=0)
        span_x = positions[:, 0].max() - positions[:, 0].min()
        span_y = positions[:, 1].max() - positions[:, 1].min()
        span_z = positions[:, 2].max() - positions[:, 2].min()
        diagonal = np.sqrt(span_x**2 + span_y**2 + span_z**2)
        
        self.gl_view.opts['center'] = pg.Vector(center[0], center[1], center[2])
        self.gl_view.opts['distance'] = diagonal * 1.5
        
        logger.info(f"Rendered {len(positions)} blocks with property '{self.current_property}'")
    
    def _update_grids(self, positions):
        """Update the tri-orthogonal grid planes."""
        # Remove old grids
        for grid in self.grid_items:
            self.gl_view.removeItem(grid)
        self.grid_items.clear()
        
        if len(positions) == 0:
            return
        
        # Calculate grid parameters
        center = positions.mean(axis=0)
        span_x = positions[:, 0].max() - positions[:, 0].min()
        span_y = positions[:, 1].max() - positions[:, 1].min()
        span_z = positions[:, 2].max() - positions[:, 2].min()
        
        grid_size = max(span_x, span_y, span_z, 100.0)
        spacing = grid_size / 20
        
        # XY plane (horizontal)
        grid_xy = gl.GLGridItem()
        grid_xy.setSize(grid_size, grid_size)
        grid_xy.setSpacing(spacing, spacing)
        grid_xy.translate(center[0], center[1], center[2])
        grid_xy.scale(1, 1, 0.01)  # Thin in Z
        self.grid_items.append(grid_xy)
        self.gl_view.addItem(grid_xy)
        
        # XZ plane (vertical, E-W)
        grid_xz = gl.GLGridItem()
        grid_xz.setSize(grid_size, grid_size)
        grid_xz.setSpacing(spacing, spacing)
        grid_xz.rotate(90, 1, 0, 0)  # Rotate around X axis
        grid_xz.translate(center[0], center[1], center[2])
        self.grid_items.append(grid_xz)
        self.gl_view.addItem(grid_xz)
        
        # YZ plane (vertical, N-S)
        grid_yz = gl.GLGridItem()
        grid_yz.setSize(grid_size, grid_size)
        grid_yz.setSpacing(spacing, spacing)
        grid_yz.rotate(90, 0, 1, 0)  # Rotate around Y axis
        grid_yz.translate(center[0], center[1], center[2])
        self.grid_items.append(grid_yz)
        self.gl_view.addItem(grid_yz)
        
        logger.info(f"Created 3 grid planes at center ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    
    def _toggle_grids(self, visible):
        """Toggle visibility of all grid planes."""
        for grid in self.grid_items:
            grid.setVisible(visible)
