"""
Structural Panel - Industry-Grade Structural Analysis UI.

Features:
- Stereonet visualization (Schmidt/Wulff, poles/planes, density contours)
- Rose diagram visualization
- Clustering with Fisher statistics
- Kinematic feasibility (planar, wedge, toppling)
- Export with audit metadata

Uses geox.structural.core engine for computations.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QMessageBox, QTextEdit,
    QDoubleSpinBox, QSpinBox, QComboBox, QFormLayout,
    QTabWidget, QCheckBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from .base_analysis_panel import BaseAnalysisPanel

# Matplotlib imports
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    FigureCanvas = None
    Figure = None

logger = logging.getLogger(__name__)


class StereonetCanvas(QWidget):
    """Matplotlib canvas for stereonet visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        if not MATPLOTLIB_AVAILABLE:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Matplotlib not available"))
            return
        
        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, aspect='equal')
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        
        self._setup_stereonet()
    
    def _setup_stereonet(self):
        """Set up the stereonet axes."""
        self.ax.clear()
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Draw primitive circle
        theta = np.linspace(0, 2 * np.pi, 100)
        self.ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
        
        # Draw graticule
        for dip in range(10, 90, 10):
            r = np.sqrt(2) * np.sin(np.radians(dip) / 2)
            self.ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=0.3, alpha=0.3)
        
        # Draw N-S and E-W lines
        self.ax.plot([0, 0], [-1, 1], 'k-', linewidth=0.5, alpha=0.5)
        self.ax.plot([-1, 1], [0, 0], 'k-', linewidth=0.5, alpha=0.5)
        
        # Add cardinal directions
        self.ax.text(0, 1.05, 'N', ha='center', va='bottom', fontsize=10)
        self.ax.text(0, -1.05, 'S', ha='center', va='top', fontsize=10)
        self.ax.text(1.05, 0, 'E', ha='left', va='center', fontsize=10)
        self.ax.text(-1.05, 0, 'W', ha='right', va='center', fontsize=10)
        
        self.canvas.draw()
    
    def plot_poles(self, x: np.ndarray, y: np.ndarray, 
                   colors: Optional[np.ndarray] = None,
                   labels: Optional[List[str]] = None):
        """Plot poles on stereonet."""
        self._setup_stereonet()
        
        if colors is not None:
            scatter = self.ax.scatter(x, y, c=colors, s=30, cmap='Set1', alpha=0.7)
        else:
            self.ax.scatter(x, y, c='blue', s=30, alpha=0.7)
        
        self.canvas.draw()
    
    def plot_density(self, x_grid: np.ndarray, y_grid: np.ndarray, 
                     density: np.ndarray, levels: int = 6):
        """Plot density contours."""
        # Mask points outside the net
        r = np.sqrt(x_grid**2 + y_grid**2)
        density_masked = np.ma.masked_where(r > 1.0, density)
        
        self.ax.contourf(x_grid, y_grid, density_masked, levels=levels, 
                         cmap='YlOrRd', alpha=0.6)
        self.ax.contour(x_grid, y_grid, density_masked, levels=levels, 
                        colors='darkred', linewidths=0.5, alpha=0.5)
        
        self.canvas.draw()
    
    def plot_great_circle(self, x: np.ndarray, y: np.ndarray, 
                          color: str = 'blue', linewidth: float = 1.0):
        """Plot a great circle."""
        self.ax.plot(x, y, color=color, linewidth=linewidth)
        self.canvas.draw()
    
    def plot_kinematic_envelope(self, x: np.ndarray, y: np.ndarray,
                                label: str, color: str = 'red'):
        """Plot kinematic envelope (daylight, friction cone, etc.)."""
        self.ax.plot(x, y, color=color, linewidth=2, linestyle='--', label=label)
        self.canvas.draw()
    
    def save_figure(self, filepath: str, dpi: int = 300):
        """Save the figure to file."""
        self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight')


class RoseCanvas(QWidget):
    """Matplotlib canvas for rose diagram visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        if not MATPLOTLIB_AVAILABLE:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Matplotlib not available"))
            return
        
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='polar')
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        
        self._setup_rose()
    
    def _setup_rose(self):
        """Set up the rose diagram axes."""
        self.ax.clear()
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)  # Clockwise
        self.canvas.draw()
    
    def plot_rose(self, bin_centers: np.ndarray, counts: np.ndarray,
                  color: str = 'steelblue', alpha: float = 0.7):
        """Plot rose diagram."""
        self._setup_rose()
        
        # Convert to radians
        theta = np.radians(bin_centers)
        width = np.radians(360 / len(bin_centers))
        
        self.ax.bar(theta, counts, width=width, color=color, alpha=alpha,
                    edgecolor='darkblue', linewidth=0.5)
        
        self.canvas.draw()
    
    def save_figure(self, filepath: str, dpi: int = 300):
        """Save the figure to file."""
        self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight')


class StructuralPanel(BaseAnalysisPanel):
    """
    Industry-grade structural analysis panel.
    
    Features:
    - Stereonet visualization (Schmidt/Wulff)
    - Rose diagram
    - Clustering with set statistics
    - Kinematic feasibility analysis
    - Export with audit metadata
    """
    
    task_name = "structural"
    analysis_complete = pyqtSignal(dict)  # Signal for analysis completion
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="structural")
        self.setWindowTitle("Structural Analysis")
        
        # Private state (Panel Safety Rules)
        self._dataset = None
        self._normals = None
        self._current_result = None
        self._audit_bundle = None
        
        # Subscribe to drillhole data from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
            
            # Load existing drillhole data if available
            existing_data = self.registry.get_drillhole_data()
            if existing_data:
                self._on_drillhole_data_loaded(existing_data)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self._build_ui()
    
    def bind_controller(self, controller):
        """
        Bind to controller for data and signals.
        
        Called by MainWindow after panel creation.
        """
        super().bind_controller(controller)
        
        # Connect to structural features signals from controller
        if controller and hasattr(controller, 'signals'):
            try:
                controller.signals.structural_features_loaded.connect(self._on_structural_features_loaded)
                logger.info("StructuralPanel connected to controller structural features signal")
            except Exception as e:
                logger.warning(f"Failed to connect structural features signal: {e}")
        
        # Also load any existing structural features
        if controller and hasattr(controller, '_app'):
            if hasattr(controller._app, '_structural_features'):
                features = controller._app._structural_features
                if features and hasattr(features, 'total_points') and features.total_points > 0:
                    self._on_structural_features_loaded(features)
    
    def _on_structural_features_loaded(self, features):
        """
        Handle structural features loaded from controller.
        
        Converts StructuralFeatureCollection to normals for analysis.
        """
        logger.info(f"StructuralPanel received structural features")
        try:
            # Extract plane measurements from features
            planes = []
            
            # Get planes from faults
            if hasattr(features, 'faults'):
                for fault in features.faults:
                    if hasattr(fault, 'get_plane_measurements'):
                        planes.extend(fault.get_plane_measurements())
            
            # Get planes from unconformities
            if hasattr(features, 'unconformities'):
                for unconf in features.unconformities:
                    if hasattr(unconf, 'get_plane_measurements'):
                        planes.extend(unconf.get_plane_measurements())
            
            if not planes:
                logger.info("No plane measurements found in structural features")
                return
            
            # Convert to normals
            try:
                from geox.structural.core import dip_dipdir_to_normal
                dips = np.array([p.dip for p in planes])
                dip_dirs = np.array([p.dip_direction for p in planes])
                self._normals = dip_dipdir_to_normal(dips, dip_dirs)
            except ImportError:
                # Fallback
                dips = np.array([p.dip for p in planes])
                dip_dirs = np.array([p.dip_direction for p in planes])
                dip_rad = np.radians(dips)
                dd_rad = np.radians(dip_dirs)
                nx = np.sin(dip_rad) * np.sin(dd_rad)
                ny = np.sin(dip_rad) * np.cos(dd_rad)
                nz = -np.cos(dip_rad)
                self._normals = np.column_stack([nx, ny, nz])
            
            self._update_data_summary()
            self.status_label.setText(f"Loaded {len(self._normals)} measurements from structural features")
            logger.info(f"Extracted {len(self._normals)} plane measurements from structural features")
            
        except Exception as e:
            logger.warning(f"Failed to process structural features: {e}")
    
    def _on_drillhole_data_loaded(self, drillhole_data):
        """Receive drillhole data when loaded from DataRegistry."""
        logger.info("Structural Panel received drillhole data from DataRegistry")
        
        # drillhole_data is a dictionary with keys: collars, surveys, assays, lithology, structures
        structures_df = None
        
        if isinstance(drillhole_data, dict):
            structures_df = drillhole_data.get('structures')
        elif hasattr(drillhole_data, 'structures'):
            structures_df = drillhole_data.structures
        
        if structures_df is not None and not structures_df.empty:
            # Extract normals from the structures DataFrame
            # Requires columns: normal_x, normal_y, normal_z
            required_cols = ['normal_x', 'normal_y', 'normal_z']
            if all(col in structures_df.columns for col in required_cols):
                normals = structures_df[required_cols].values
                self._normals = normals
                self._update_data_summary()
                self.status_label.setText(f"Loaded {len(normals)} structural measurements from drillholes")
                logger.info(f"Loaded {len(normals)} structural measurements from drillhole data")
            else:
                logger.warning(f"Structures DataFrame missing required normal columns. Has: {list(structures_df.columns)}")
        else:
            logger.info("Drillhole data does not contain structural measurements")
    
    def bind_data(self, dataset=None, normals=None):
        """
        Bind data to the panel (Controller calls this).
        
        Args:
            dataset: Legacy StructuralDataset
            normals: Nx3 array of unit normal vectors (preferred)
        """
        if normals is not None:
            self._normals = np.asarray(normals)
        if dataset is not None:
            self._dataset = dataset
            # Convert to normals
            try:
                from geox.structural.core import dip_dipdir_to_normal
                dips = np.array([p.dip for p in dataset.planes])
                dip_dirs = np.array([p.dip_direction for p in dataset.planes])
                self._normals = dip_dipdir_to_normal(dips, dip_dirs)
            except Exception as e:
                logger.warning(f"Could not convert dataset to normals: {e}")
        
        self._update_data_summary()
    
    def _update_data_summary(self):
        """Update the data summary display."""
        if hasattr(self, 'data_text'):
            n = len(self._normals) if self._normals is not None else 0
            self.data_text.setText(f"Loaded: {n} structural measurements")
    
    def _build_ui(self):
        """Build the UI."""
        layout = self.main_layout
        
        title = QLabel("Structural Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Main tabs
        tabs = QTabWidget()
        tabs.addTab(self._create_data_tab(), "Data")
        tabs.addTab(self._create_stereonet_tab(), "Stereonet")
        tabs.addTab(self._create_rose_tab(), "Rose Diagram")
        tabs.addTab(self._create_clustering_tab(), "Sets")
        tabs.addTab(self._create_kinematic_tab(), "Kinematics")
        
        layout.addWidget(tabs)
        
        # Status bar
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.export_btn = QPushButton("Export Results...")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        status_layout.addWidget(self.export_btn)
        
        layout.addWidget(status_frame)
    
    def _create_data_tab(self) -> QWidget:
        """Create data loading tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Load options
        load_group = QGroupBox("Load Data")
        load_layout = QVBoxLayout()
        
        load_csv_btn = QPushButton("Load from CSV...")
        load_csv_btn.clicked.connect(self._load_csv)
        load_layout.addWidget(load_csv_btn)
        
        load_drillhole_btn = QPushButton("Load from Drillhole Structures")
        load_drillhole_btn.clicked.connect(self._load_from_drillholes)
        load_layout.addWidget(load_drillhole_btn)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Data summary
        self.data_text = QTextEdit()
        self.data_text.setReadOnly(True)
        self.data_text.setMaximumHeight(150)
        layout.addWidget(self.data_text)
        
        layout.addStretch()
        return widget
    
    def _create_stereonet_tab(self) -> QWidget:
        """Create stereonet visualization tab."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left: controls
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls.setMaximumWidth(250)
        
        # Net type
        net_group = QGroupBox("Projection")
        net_layout = QFormLayout()
        
        self.net_type = QComboBox()
        self.net_type.addItems(["Schmidt (Equal-Area)", "Wulff (Equal-Angle)"])
        net_layout.addRow("Net Type:", self.net_type)
        
        self.hemisphere = QComboBox()
        self.hemisphere.addItems(["Lower", "Upper"])
        net_layout.addRow("Hemisphere:", self.hemisphere)
        
        net_group.setLayout(net_layout)
        controls_layout.addWidget(net_group)
        
        # Display options
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()
        
        self.show_poles = QCheckBox("Show Poles")
        self.show_poles.setChecked(True)
        display_layout.addWidget(self.show_poles)
        
        self.show_planes = QCheckBox("Show Planes (Great Circles)")
        display_layout.addWidget(self.show_planes)
        
        self.show_density = QCheckBox("Show Density Contours")
        self.show_density.setChecked(True)
        display_layout.addWidget(self.show_density)
        
        self.color_by_set = QCheckBox("Color by Set")
        display_layout.addWidget(self.color_by_set)
        
        display_group.setLayout(display_layout)
        controls_layout.addWidget(display_group)
        
        # Plot button
        plot_btn = QPushButton("Update Stereonet")
        plot_btn.clicked.connect(self._update_stereonet)
        controls_layout.addWidget(plot_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls)
        
        # Right: canvas
        self.stereonet_canvas = StereonetCanvas()
        layout.addWidget(self.stereonet_canvas, stretch=1)
        
        return widget
    
    def _create_rose_tab(self) -> QWidget:
        """Create rose diagram tab."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left: controls
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls.setMaximumWidth(250)
        
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        self.rose_bins = QSpinBox()
        self.rose_bins.setRange(8, 72)
        self.rose_bins.setValue(36)
        params_layout.addRow("Number of Bins:", self.rose_bins)
        
        self.rose_direction = QComboBox()
        self.rose_direction.addItems(["Dip Direction", "Strike"])
        params_layout.addRow("Direction:", self.rose_direction)
        
        self.rose_axial = QCheckBox("Axial Data (0-180°)")
        self.rose_axial.setChecked(True)
        params_layout.addRow("", self.rose_axial)
        
        self.rose_weighting = QComboBox()
        self.rose_weighting.addItems(["Count", "Length-Weighted"])
        params_layout.addRow("Weighting:", self.rose_weighting)
        
        params_group.setLayout(params_layout)
        controls_layout.addWidget(params_group)
        
        plot_btn = QPushButton("Update Rose Diagram")
        plot_btn.clicked.connect(self._update_rose)
        controls_layout.addWidget(plot_btn)
        
        # Statistics display
        self.rose_stats = QTextEdit()
        self.rose_stats.setReadOnly(True)
        self.rose_stats.setMaximumHeight(150)
        controls_layout.addWidget(self.rose_stats)
        
        controls_layout.addStretch()
        layout.addWidget(controls)
        
        # Right: canvas
        self.rose_canvas = RoseCanvas()
        layout.addWidget(self.rose_canvas, stretch=1)
        
        return widget
    
    def _create_clustering_tab(self) -> QWidget:
        """Create clustering tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        params_group = QGroupBox("Clustering Parameters")
        params_layout = QFormLayout()
        
        self.cluster_method = QComboBox()
        self.cluster_method.addItems(["Auto (HDBSCAN/DBSCAN)", "K-Means", "DBSCAN"])
        params_layout.addRow("Method:", self.cluster_method)
        
        self.n_sets = QSpinBox()
        self.n_sets.setRange(1, 10)
        self.n_sets.setValue(3)
        params_layout.addRow("Number of Sets (K-Means):", self.n_sets)
        
        self.min_cluster_size = QSpinBox()
        self.min_cluster_size.setRange(2, 50)
        self.min_cluster_size.setValue(5)
        params_layout.addRow("Min Cluster Size:", self.min_cluster_size)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        cluster_btn = QPushButton("Run Clustering")
        cluster_btn.clicked.connect(self._run_clustering)
        layout.addWidget(cluster_btn)
        
        self.cluster_results = QTextEdit()
        self.cluster_results.setReadOnly(True)
        layout.addWidget(self.cluster_results)
        
        return widget
    
    def _create_kinematic_tab(self) -> QWidget:
        """Create kinematic analysis tab with full parameter controls."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left: controls
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls.setMaximumWidth(300)
        
        # Slope parameters
        slope_group = QGroupBox("Slope Parameters")
        slope_layout = QFormLayout()
        
        self.slope_dip = QDoubleSpinBox()
        self.slope_dip.setRange(0, 90)
        self.slope_dip.setValue(55.0)
        self.slope_dip.setSuffix("°")
        self.slope_dip.setDecimals(1)
        slope_layout.addRow("Slope Dip:", self.slope_dip)
        
        self.slope_direction = QDoubleSpinBox()
        self.slope_direction.setRange(0, 360)
        self.slope_direction.setValue(0.0)
        self.slope_direction.setSuffix("°")
        self.slope_direction.setDecimals(1)
        slope_layout.addRow("Slope Dip Direction:", self.slope_direction)
        
        slope_group.setLayout(slope_layout)
        controls_layout.addWidget(slope_group)
        
        # Geotechnical parameters
        geotech_group = QGroupBox("Geotechnical Parameters")
        geotech_layout = QFormLayout()
        
        self.friction_angle = QDoubleSpinBox()
        self.friction_angle.setRange(0, 90)
        self.friction_angle.setValue(35.0)
        self.friction_angle.setSuffix("°")
        self.friction_angle.setDecimals(1)
        geotech_layout.addRow("Friction Angle (φ):", self.friction_angle)
        
        self.lateral_limits = QDoubleSpinBox()
        self.lateral_limits.setRange(0, 90)
        self.lateral_limits.setValue(20.0)
        self.lateral_limits.setSuffix("°")
        self.lateral_limits.setDecimals(1)
        geotech_layout.addRow("Lateral Limits:", self.lateral_limits)
        
        geotech_group.setLayout(geotech_layout)
        controls_layout.addWidget(geotech_group)
        
        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout()
        
        self.analyze_planar = QCheckBox("Planar Sliding")
        self.analyze_planar.setChecked(True)
        analysis_layout.addWidget(self.analyze_planar)
        
        self.analyze_wedge = QCheckBox("Wedge Sliding")
        self.analyze_wedge.setChecked(True)
        analysis_layout.addWidget(self.analyze_wedge)
        
        self.analyze_toppling = QCheckBox("Toppling")
        self.analyze_toppling.setChecked(True)
        analysis_layout.addWidget(self.analyze_toppling)
        
        self.show_envelopes = QCheckBox("Show Envelopes on Stereonet")
        self.show_envelopes.setChecked(True)
        analysis_layout.addWidget(self.show_envelopes)
        
        analysis_group.setLayout(analysis_layout)
        controls_layout.addWidget(analysis_group)
        
        analyze_btn = QPushButton("Run Kinematic Analysis")
        analyze_btn.clicked.connect(self._run_kinematic)
        controls_layout.addWidget(analyze_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls)
        
        # Right: results
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        self.kinematic_results = QTextEdit()
        self.kinematic_results.setReadOnly(True)
        results_layout.addWidget(self.kinematic_results)
        
        layout.addWidget(results_widget, stretch=1)
        
        return widget
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def _load_csv(self):
        """Load structural data from CSV."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Structural Data", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return
        
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            # Try to find dip/dip-direction columns
            dip_col = None
            dd_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'dip' in col_lower and 'direction' not in col_lower:
                    dip_col = col
                if 'dip' in col_lower and 'direction' in col_lower:
                    dd_col = col
                if col_lower in ['dipdir', 'dip_direction', 'dip_dir', 'dipazimuth']:
                    dd_col = col
            
            if dip_col is None or dd_col is None:
                QMessageBox.warning(self, "Error", 
                    "Could not find dip and dip-direction columns.\n"
                    "Expected columns containing 'dip' and 'direction'.")
                return
            
            dips = df[dip_col].values
            dip_dirs = df[dd_col].values
            
            # Convert to normals
            try:
                from geox.structural.core import dip_dipdir_to_normal
                self._normals = dip_dipdir_to_normal(dips, dip_dirs)
            except ImportError:
                # Fallback
                dip_rad = np.radians(dips)
                dd_rad = np.radians(dip_dirs)
                nx = np.sin(dip_rad) * np.sin(dd_rad)
                ny = np.sin(dip_rad) * np.cos(dd_rad)
                nz = -np.cos(dip_rad)
                self._normals = np.column_stack([nx, ny, nz])
            
            self._update_data_summary()
            self.status_label.setText(f"Loaded {len(self._normals)} measurements from {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load CSV: {e}")
            logger.exception("CSV load error")
    
    def _load_from_drillholes(self):
        """Load structural data from drillhole database."""
        if self.registry is None:
            QMessageBox.warning(self, "Error", "DataRegistry not available")
            return
        
        try:
            drillhole_data = self.registry.get_drillhole_data()
            if drillhole_data is None:
                QMessageBox.warning(self, "Error", "No drillhole data loaded")
                return
            
            if hasattr(drillhole_data, 'get_structures_as_normals'):
                normals = drillhole_data.get_structures_as_normals()
                if len(normals) == 0:
                    QMessageBox.warning(self, "Error", "No structural measurements in drillhole data")
                    return
                self._normals = normals
                self._update_data_summary()
                self.status_label.setText(f"Loaded {len(self._normals)} measurements from drillholes")
            else:
                QMessageBox.warning(self, "Error", "Drillhole data does not contain structural measurements")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load from drillholes: {e}")
    
    # =========================================================================
    # Stereonet
    # =========================================================================
    
    def _update_stereonet(self):
        """Update stereonet plot."""
        if self._normals is None or len(self._normals) == 0:
            QMessageBox.warning(self, "Error", "No structural data loaded")
            return
        
        try:
            from geox.structural.core import (
                project_schmidt, project_wulff,
                spherical_kde_grid, canonicalize_to_lower_hemisphere,
                NetType, Hemisphere
            )
            
            normals = canonicalize_to_lower_hemisphere(self._normals)
            
            # Select projection
            net_type = NetType.SCHMIDT if "Schmidt" in self.net_type.currentText() else NetType.WULFF
            hemisphere = Hemisphere.LOWER if "Lower" in self.hemisphere.currentText() else Hemisphere.UPPER
            
            # Project poles
            if net_type == NetType.SCHMIDT:
                x, y = project_schmidt(normals, hemisphere)
            else:
                x, y = project_wulff(normals, hemisphere)
            
            # Setup canvas
            self.stereonet_canvas._setup_stereonet()
            
            # Plot density
            if self.show_density.isChecked():
                lat_grid, lon_grid, density = spherical_kde_grid(normals)
                
                # Convert grid to projected coordinates
                from geox.structural.core import dip_dipdir_to_normal
                grid_x = []
                grid_y = []
                for i in range(lat_grid.shape[0]):
                    row_x = []
                    row_y = []
                    for j in range(lat_grid.shape[1]):
                        n = dip_dipdir_to_normal(lat_grid[i, j], lon_grid[i, j]).flatten()
                        if net_type == NetType.SCHMIDT:
                            px, py = project_schmidt(n.reshape(1, 3), hemisphere)
                        else:
                            px, py = project_wulff(n.reshape(1, 3), hemisphere)
                        row_x.append(px[0])
                        row_y.append(py[0])
                    grid_x.append(row_x)
                    grid_y.append(row_y)
                
                self.stereonet_canvas.plot_density(
                    np.array(grid_x), np.array(grid_y), density
                )
            
            # Plot poles
            if self.show_poles.isChecked():
                self.stereonet_canvas.plot_poles(x, y)
            
            self.status_label.setText("Stereonet updated")
            
        except ImportError:
            QMessageBox.warning(self, "Error", "geox.structural.core not available")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update stereonet: {e}")
            logger.exception("Stereonet error")
    
    # =========================================================================
    # Rose Diagram
    # =========================================================================
    
    def _update_rose(self):
        """Update rose diagram."""
        if self._normals is None or len(self._normals) == 0:
            QMessageBox.warning(self, "Error", "No structural data loaded")
            return
        
        try:
            from geox.structural.core import (
                normal_to_dip_dipdir, strike_from_dip_direction,
                compute_rose_histogram, WeightingMode
            )
            
            # Get directions
            _, dip_dirs = normal_to_dip_dipdir(self._normals)
            
            if "Strike" in self.rose_direction.currentText():
                directions = strike_from_dip_direction(dip_dirs)
            else:
                directions = dip_dirs
            
            # Compute histogram
            weighting = WeightingMode.COUNT
            if "Length" in self.rose_weighting.currentText():
                weighting = WeightingMode.LENGTH
            
            result = compute_rose_histogram(
                directions,
                n_bins=self.rose_bins.value(),
                is_axial=self.rose_axial.isChecked(),
                weighting=weighting
            )
            
            # Plot
            self.rose_canvas.plot_rose(result.bin_centers, result.counts)
            
            # Update stats
            self.rose_stats.clear()
            self.rose_stats.append(f"Mean Direction: {result.mean_direction:.1f}°")
            self.rose_stats.append(f"Mean Resultant Length: {result.mean_resultant_length:.3f}")
            self.rose_stats.append(f"Circular Variance: {result.circular_variance:.3f}")
            self.rose_stats.append(f"N = {len(directions)}")
            
            self.status_label.setText("Rose diagram updated")
            
        except ImportError:
            QMessageBox.warning(self, "Error", "geox.structural.core not available")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update rose diagram: {e}")
            logger.exception("Rose error")
    
    # =========================================================================
    # Clustering
    # =========================================================================
    
    def _run_clustering(self):
        """Run structural clustering."""
        if self._normals is None or len(self._normals) == 0:
            QMessageBox.warning(self, "Error", "No structural data loaded")
            return
        
        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not available")
            return
        
        method_text = self.cluster_method.currentText()
        if "Auto" in method_text:
            method = "auto"
        elif "K-Means" in method_text:
            method = "kmeans"
        else:
            method = "dbscan"
        
        config = {
            "normals": self._normals,
            "method": method,
            "n_sets": self.n_sets.value() if method == "kmeans" else None,
            "min_cluster_size": self.min_cluster_size.value(),
            "analysis_type": "clusters"
        }
        
        self.status_label.setText("Running clustering...")
        self.controller.run_structural_analysis(config, self._on_clustering_complete)
    
    def _on_clustering_complete(self, result: Dict[str, Any]):
        """Handle clustering complete."""
        self.cluster_results.clear()
        
        if "structural_sets" in result:
            sets = result["structural_sets"]
            self.cluster_results.append(f"Identified {len(sets)} structural sets\n")
            self.cluster_results.append("-" * 40 + "\n")
            
            for s in sets:
                self.cluster_results.append(f"\n{s['set_id'].upper()}")
                self.cluster_results.append(f"  N = {s['n_members']}")
                self.cluster_results.append(f"  Mean Dip: {s['mean_dip']:.1f}°")
                self.cluster_results.append(f"  Mean Dip Dir: {s['mean_dip_direction']:.1f}°")
                self.cluster_results.append(f"  κ (concentration): {s['kappa']:.1f}")
                self.cluster_results.append(f"  95% Cone: {s['confidence_cone_95']:.1f}°")
                self.cluster_results.append(f"  Dispersion: {s['dispersion']:.1f}°")
            
            self._current_result = result
            self._audit_bundle = result.get("audit_bundle")
            self.export_btn.setEnabled(True)
            self.status_label.setText(f"Clustering complete: {len(sets)} sets")
        else:
            self.cluster_results.append("Clustering failed")
            self.status_label.setText("Clustering failed")
    
    # =========================================================================
    # Kinematic Analysis
    # =========================================================================
    
    def _run_kinematic(self):
        """Run kinematic analysis."""
        if self._normals is None or len(self._normals) == 0:
            QMessageBox.warning(self, "Error", "No structural data loaded")
            return
        
        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not available")
            return
        
        # Build analysis type
        analysis_type = "all"
        if self.analyze_planar.isChecked() and not self.analyze_wedge.isChecked() and not self.analyze_toppling.isChecked():
            analysis_type = "plane"
        elif self.analyze_wedge.isChecked() and not self.analyze_planar.isChecked() and not self.analyze_toppling.isChecked():
            analysis_type = "wedge"
        elif self.analyze_toppling.isChecked() and not self.analyze_planar.isChecked() and not self.analyze_wedge.isChecked():
            analysis_type = "toppling"
        
        config = {
            "normals": self._normals,
            "slope_dip": self.slope_dip.value(),
            "slope_dip_direction": self.slope_direction.value(),
            "phi": self.friction_angle.value(),
            "lateral_limits": self.lateral_limits.value(),
            "analysis_type": analysis_type,
        }
        
        self.status_label.setText("Running kinematic analysis...")
        self.controller.run_structural_analysis(config, self._on_kinematic_complete)
    
    def _on_kinematic_complete(self, result: Dict[str, Any]):
        """Handle kinematic analysis complete."""
        self.kinematic_results.clear()
        
        if "summary" in result:
            summary = result["summary"]
            kinematic = result.get("kinematic_result", {})
            
            self.kinematic_results.append("KINEMATIC ANALYSIS RESULTS")
            self.kinematic_results.append("=" * 40)
            
            self.kinematic_results.append(f"\nSlope Parameters:")
            self.kinematic_results.append(f"  Dip: {summary['slope_parameters']['dip']:.1f}°")
            self.kinematic_results.append(f"  Dip Direction: {summary['slope_parameters']['dip_direction']:.1f}°")
            self.kinematic_results.append(f"  Friction Angle: {summary['slope_parameters']['friction_angle']:.1f}°")
            
            self.kinematic_results.append(f"\nTotal Measurements: {summary['n_measurements']}")
            
            self.kinematic_results.append(f"\n--- PLANAR SLIDING ---")
            self.kinematic_results.append(f"  Feasible: {summary['planar']['n_feasible']} ({summary['planar']['fraction']:.1%})")
            
            self.kinematic_results.append(f"\n--- WEDGE SLIDING ---")
            self.kinematic_results.append(f"  Pairs Analyzed: {summary['wedge']['n_pairs_analyzed']}")
            self.kinematic_results.append(f"  Feasible: {summary['wedge']['n_feasible']} ({summary['wedge']['fraction']:.1%})")
            
            self.kinematic_results.append(f"\n--- TOPPLING ---")
            self.kinematic_results.append(f"  Feasible: {summary['toppling']['n_feasible']} ({summary['toppling']['fraction']:.1%})")
            
            self.kinematic_results.append(f"\n" + "=" * 40)
            self.kinematic_results.append(f"RISK LEVEL: {summary['risk_level']}")
            self.kinematic_results.append(f"Dominant Mode: {summary['dominant_mode']}")
            
            self._current_result = result
            self._audit_bundle = result.get("audit_bundle")
            self.export_btn.setEnabled(True)
            self.status_label.setText(f"Analysis complete - Risk: {summary['risk_level']}")
        else:
            # Legacy result format
            if "result" in result:
                r = result["result"]
                self.kinematic_results.append(f"Feasible: {r.get('feasible_count', 0)} / {r.get('total_count', 0)}")
                self.kinematic_results.append(f"Fraction: {r.get('feasible_fraction', 0):.1%}")
            else:
                self.kinematic_results.append("Analysis failed")
            self.status_label.setText("Analysis complete")
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def _export_results(self):
        """Export results with audit metadata."""
        if self._current_result is None:
            QMessageBox.warning(self, "Error", "No results to export")
            return
        
        # Get save location
        file_path, filter_used = QFileDialog.getSaveFileName(
            self, "Export Results", "", 
            "JSON Files (*.json);;PNG Image (*.png);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.json') or 'JSON' in filter_used:
                # Export JSON with audit bundle
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "result": self._current_result,
                }
                if self._audit_bundle:
                    export_data["audit_bundle"] = self._audit_bundle
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.status_label.setText(f"Exported to {os.path.basename(file_path)}")
            
            elif file_path.endswith('.png'):
                # Export current figure
                if hasattr(self, 'stereonet_canvas'):
                    self.stereonet_canvas.save_figure(file_path)
                    
                    # Also save JSON sidecar
                    json_path = file_path.replace('.png', '_metadata.json')
                    if self._audit_bundle:
                        with open(json_path, 'w') as f:
                            json.dump(self._audit_bundle, f, indent=2, default=str)
                    
                    self.status_label.setText(f"Exported figure and metadata")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Export failed: {e}")
            logger.exception("Export error")
