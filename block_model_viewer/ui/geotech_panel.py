"""
Geotechnical Dashboard Panel

High-level panel for managing rock-mass properties and geotechnical analysis.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QFileDialog, QTabWidget, QWidget,
    QTextEdit, QTableWidget, QTableWidgetItem, QDoubleSpinBox, QSpinBox
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


class GeotechPanel(BaseAnalysisPanel):
    """
    Geotechnical Dashboard Panel.
    
    Provides tabs for:
    - Rock Mass Model management
    - Stope Stability analysis
    - Slope Risk assessment
    """
    # PanelManager metadata
    PANEL_ID = "GeotechPanel"
    PANEL_NAME = "Geotech Panel"
    PANEL_CATEGORY = PanelCategory.GEOTECH
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "geotech_interpolation"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="geotech")
        
        self.rock_mass_model = None
        self.current_variable = "RMR"
        self.interpolation_method = "IDW"
        self.drillhole_data = None
        
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
        
        # setup_ui() is called automatically by BaseAnalysisPanel.__init__
        logger.info("Initialized Geotech Dashboard panel")
    
    def _on_drillhole_data_loaded(self, drillhole_data):
        """
        Automatically receive drillhole data when it's loaded from DataRegistry.
        
        Args:
            drillhole_data: Drillhole data dict with 'composites', 'assays', etc.
        """
        logger.info("Geotech Panel received drillhole data from DataRegistry")
        self.drillhole_data = drillhole_data
        # Geotech data might be in lithology or domain columns
        if isinstance(drillhole_data, dict):
            composites = drillhole_data.get('composites')
            if composites is not None and not composites.empty:
                self.drillhole_data = composites
                logger.info("Geotech Panel loaded composites from drillhole data")
    
    def setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("<b>Geotechnical Dashboard</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_rock_mass_tab(), "Rock Mass Model")
        tabs.addTab(self._create_interpolation_tab(), "Interpolation")
        tabs.addTab(self._create_summary_tab(), "Summary")
        layout.addWidget(tabs)
        
        self.tabs = tabs
    
    def _create_rock_mass_tab(self) -> QWidget:
        """Create rock mass model management tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Load data section
        load_group = QGroupBox("Load Geotechnical Data")
        load_layout = QVBoxLayout(load_group)
        
        load_btn = QPushButton("Load from CSV")
        load_btn.clicked.connect(self._load_geotech_data)
        load_layout.addWidget(load_btn)
        
        self.data_status_label = QLabel("No data loaded")
        self.data_status_label.setStyleSheet("color: orange;")
        load_layout.addWidget(self.data_status_label)
        
        layout.addWidget(load_group)
        
        # Property selection
        prop_group = QGroupBox("Available Properties")
        prop_layout = QFormLayout(prop_group)
        
        self.property_combo = QComboBox()
        self.property_combo.addItems(["RQD", "Q", "RMR", "GSI"])
        self.property_combo.currentTextChanged.connect(self._on_property_changed)
        prop_layout.addRow("Property:", self.property_combo)
        
        layout.addWidget(prop_group)
        
        layout.addStretch()
        return widget
    
    def _create_interpolation_tab(self) -> QWidget:
        """Create interpolation settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Interpolation settings
        interp_group = QGroupBox("Interpolation Settings")
        interp_layout = QFormLayout(interp_group)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["IDW", "OK"])
        self.method_combo.setToolTip("IDW: Inverse Distance Weighting\nOK: Ordinary Kriging")
        interp_layout.addRow("Method:", self.method_combo)
        
        self.power_spinbox = QDoubleSpinBox()
        self.power_spinbox.setRange(1.0, 5.0)
        self.power_spinbox.setValue(2.0)
        self.power_spinbox.setSingleStep(0.5)
        self.power_spinbox.setToolTip("Power parameter for IDW (higher = more local influence)")
        interp_layout.addRow("IDW Power:", self.power_spinbox)
        
        self.max_neighbors_spinbox = QSpinBox()
        self.max_neighbors_spinbox.setRange(5, 50)
        self.max_neighbors_spinbox.setValue(20)
        self.max_neighbors_spinbox.setToolTip("Maximum number of neighbors to use")
        interp_layout.addRow("Max Neighbors:", self.max_neighbors_spinbox)
        
        layout.addWidget(interp_group)
        
        # Run interpolation
        run_btn = QPushButton("Interpolate to Block Model")
        run_btn.clicked.connect(self._run_interpolation)
        layout.addWidget(run_btn)
        
        layout.addStretch()
        return widget
    
    def _create_summary_tab(self) -> QWidget:
        """Create summary statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["Property", "Mean", "Std Dev"])
        layout.addWidget(self.summary_table)
        
        return widget
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect interpolation parameters."""
        params = {
            'variable': self.property_combo.currentText(),
            'method': self.method_combo.currentText(),
            'power': self.power_spinbox.value(),
            'max_neighbors': self.max_neighbors_spinbox.value()
        }
        # Add rock mass model if available
        if self.rock_mass_model:
            params['rock_mass_model'] = self.rock_mass_model
        return params
    
    def validate_inputs(self) -> bool:
        """Validate inputs before running interpolation."""
        if not self.validate_block_model_loaded():
            return False
        
        if self.rock_mass_model is None or len(self.rock_mass_model.points) == 0:
            self.show_error("No Data", "Please load geotechnical data first.")
            return False
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle interpolation results."""
        if payload.get('error'):
            self.show_error("Interpolation Error", payload['error'])
            return
        
        property_name = payload.get('property_name', 'Unknown')
        visualization = payload.get('visualization', {})
        
        # Geotech properties are added directly to the block model
        # The property will be available in the property panel for visualization
        # No need to add a separate mesh - just refresh the scene
        
        # Update summary
        if 'summary_stats' in payload:
            self._update_summary_table(payload['summary_stats'])
        
        self.show_info("Success", f"Geotechnical interpolation completed: {property_name}\n\n"
                                  f"The property '{property_name}' has been added to the block model.\n"
                                  f"Select it from the Property panel to visualize.")
        
        # Publish updated block model to DataRegistry (with new geotech property)
        if hasattr(self, 'registry') and self.registry:
            try:
                # Get current block model and update it
                existing_block_model = self.registry.get_block_model()
                if existing_block_model is not None:
                    # The property has been added to the block model by the controller
                    # Re-register the block model to notify subscribers
                    self.registry.register_block_model(existing_block_model, source_panel="GeotechPanel")
                    logger.info(f"Geotech Panel published updated block model with property '{property_name}' to DataRegistry")
            except Exception as e:
                logger.warning(f"Failed to register geotech interpolation results: {e}")
        
        # Trigger visualization refresh and property panel update
        if self.controller:
            self.controller.refresh_scene()
            
            # Refresh property panel if available
            if hasattr(self.controller, 'r') and self.controller.r:
                # The property is now in the block model, so it will appear in property panel
                logger.info(f"Geotechnical property '{property_name}' added to block model")
    
    def _load_geotech_data(self):
        """Load geotechnical data from CSV."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Geotechnical Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            from ..geotech.rock_mass_model import RockMassModel
            
            self.rock_mass_model = RockMassModel()
            self.rock_mass_model.load_from_csv(Path(file_path))
            
            # Update status
            n_points = len(self.rock_mass_model.points)
            self.data_status_label.setText(f"Loaded {n_points} points")
            self.data_status_label.setStyleSheet("color: green;")
            
            # Update summary
            stats = self.rock_mass_model.get_summary_statistics()
            self._update_summary_table(stats)
            
            self.show_info("Success", f"Loaded {n_points} geotechnical data points.")
            
        except Exception as e:
            logger.error(f"Error loading geotech data: {e}", exc_info=True)
            self.show_error("Load Error", f"Failed to load data: {e}")
    
    def _on_property_changed(self, property_name: str):
        """Handle property selection change."""
        self.current_variable = property_name
    
    def _run_interpolation(self):
        """Run interpolation analysis."""
        self.run_analysis()
    
    def _update_summary_table(self, stats: Dict[str, Dict[str, float]]):
        """Update summary statistics table."""
        self.summary_table.setRowCount(len(stats))
        
        row = 0
        for prop_name, prop_stats in stats.items():
            self.summary_table.setItem(row, 0, QTableWidgetItem(prop_name))
            self.summary_table.setItem(row, 1, QTableWidgetItem(f"{prop_stats['mean']:.2f}"))
            self.summary_table.setItem(row, 2, QTableWidgetItem(f"{prop_stats['std']:.2f}"))
            row += 1

