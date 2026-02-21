"""
Rockburst Index Analysis Panel

Panel for computing and visualizing rockburst indices at stopes, drives, and other excavations.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QTableWidget, QTableWidgetItem, QCheckBox
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class RockburstPanel(BaseAnalysisPanel):
    """
    Rockburst Index Analysis Panel.
    
    Provides:
    - Target geometry selection (stopes, drives, levels)
    - Rockburst index computation
    - Deterministic and probabilistic analysis
    - Results visualization
    """
    
    task_name = "rockburst_index"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="rockburst")
        
        self.current_results = []
        self.hazard_volume = None
        self._setup_ui()
        logger.info("Initialized Rockburst Index panel")
    


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
    def setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("<b>Rockburst Index Analysis</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # Target selection
        target_group = QGroupBox("Target Geometry")
        target_layout = QFormLayout(target_group)
        
        self.target_combo = QComboBox()
        self.target_combo.addItems(["Stopes", "Drives", "Levels", "Custom Points"])
        target_layout.addRow("Target Type:", self.target_combo)
        
        self.use_existing_stopes = QCheckBox("Use existing UG stopes")
        self.use_existing_stopes.setChecked(True)
        target_layout.addRow(self.use_existing_stopes)
        
        layout.addWidget(target_group)
        
        # Data sources
        source_group = QGroupBox("Data Sources")
        source_layout = QVBoxLayout(source_group)
        
        self.hazard_source_combo = QComboBox()
        self.hazard_source_combo.addItems(["From Seismic Panel", "Load Hazard Volume"])
        source_layout.addWidget(QLabel("Hazard Volume:"))
        source_layout.addWidget(self.hazard_source_combo)
        
        self.rock_mass_source_combo = QComboBox()
        self.rock_mass_source_combo.addItems(["From Geotech Panel", "None"])
        source_layout.addWidget(QLabel("Rock Mass Grid:"))
        source_layout.addWidget(self.rock_mass_source_combo)
        
        layout.addWidget(source_group)
        
        # Parameters
        params_group = QGroupBox("Index Parameters")
        params_layout = QFormLayout(params_group)
        
        self.hazard_weight_spinbox = QDoubleSpinBox()
        self.hazard_weight_spinbox.setRange(0.0, 1.0)
        self.hazard_weight_spinbox.setValue(0.6)
        self.hazard_weight_spinbox.setSingleStep(0.1)
        params_layout.addRow("Hazard Weight:", self.hazard_weight_spinbox)
        
        self.rock_mass_weight_spinbox = QDoubleSpinBox()
        self.rock_mass_weight_spinbox.setRange(0.0, 1.0)
        self.rock_mass_weight_spinbox.setValue(0.3)
        self.rock_mass_weight_spinbox.setSingleStep(0.1)
        params_layout.addRow("Rock Mass Weight:", self.rock_mass_weight_spinbox)
        
        self.stress_weight_spinbox = QDoubleSpinBox()
        self.stress_weight_spinbox.setRange(0.0, 1.0)
        self.stress_weight_spinbox.setValue(0.1)
        self.stress_weight_spinbox.setSingleStep(0.1)
        params_layout.addRow("Stress Weight:", self.stress_weight_spinbox)
        
        self.rmr_threshold_spinbox = QDoubleSpinBox()
        self.rmr_threshold_spinbox.setRange(0.0, 100.0)
        self.rmr_threshold_spinbox.setValue(60.0)
        params_layout.addRow("RMR Threshold:", self.rmr_threshold_spinbox)
        
        layout.addWidget(params_group)
        
        # Analysis buttons
        btn_layout = QHBoxLayout()
        
        self.deterministic_btn = QPushButton("Run Deterministic Analysis")
        self.deterministic_btn.clicked.connect(self._run_deterministic)
        btn_layout.addWidget(self.deterministic_btn)
        
        self.probabilistic_btn = QPushButton("Run Probabilistic Analysis")
        self.probabilistic_btn.clicked.connect(self._run_probabilistic)
        btn_layout.addWidget(self.probabilistic_btn)
        
        self.n_realizations_spinbox = QSpinBox()
        self.n_realizations_spinbox.setRange(10, 10000)
        self.n_realizations_spinbox.setValue(100)
        btn_layout.addWidget(QLabel("Realizations:"))
        btn_layout.addWidget(self.n_realizations_spinbox)
        
        layout.addLayout(btn_layout)
        
        # Results table
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Location", "Index", "Class", "Events", "Notes"
        ])
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect rockburst index parameters."""
        return {
            'target_type': self.target_combo.currentText(),
            'use_existing_stopes': self.use_existing_stopes.isChecked(),
            'hazard_weight': self.hazard_weight_spinbox.value(),
            'rock_mass_weight': self.rock_mass_weight_spinbox.value(),
            'stress_weight': self.stress_weight_spinbox.value(),
            'rmr_threshold': self.rmr_threshold_spinbox.value(),
            'analysis_type': 'deterministic'
        }
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        # Check weights sum to ~1.0
        total_weight = (
            self.hazard_weight_spinbox.value() +
            self.rock_mass_weight_spinbox.value() +
            self.stress_weight_spinbox.value()
        )
        
        if abs(total_weight - 1.0) > 0.01:
            self.show_warning("Weights", "Weights should sum to approximately 1.0. Normalizing...")
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle analysis results."""
        if payload.get('error'):
            self.show_error("Analysis Error", payload['error'])
            return
        
        results = payload.get('results', [])
        if results:
            self.current_results = results
            self._update_results_table()
        
        # Request visualization
        if self.controller and hasattr(self.controller, 'renderer'):
            try:
                renderer = self.controller.renderer
                if hasattr(renderer, 'render_rockburst_index'):
                    renderer.render_rockburst_index(self.current_results)
            except Exception as e:
                logger.warning(f"Failed to visualize rockburst index: {e}", exc_info=True)
        
        self.show_info("Success", "Rockburst index analysis completed.")
    
    def _run_deterministic(self):
        """Run deterministic analysis."""
        self.task_name = "rockburst_index"
        self.run_analysis()
    
    def _run_probabilistic(self):
        """Run probabilistic analysis."""
        params = self.gather_parameters()
        params['n_realizations'] = self.n_realizations_spinbox.value()
        params['analysis_type'] = 'probabilistic'
        
        if not self.validate_inputs():
            return
        
        if self.controller:
            self.controller.run_task("rockburst_index_mc", params, self.handle_results)
    
    def _update_results_table(self):
        """Update results table."""
        self.results_table.setRowCount(len(self.current_results))
        
        for row, result in enumerate(self.current_results):
            # Location
            loc_str = f"({result['location'][0]:.1f}, {result['location'][1]:.1f}, {result['location'][2]:.1f})"
            self.results_table.setItem(row, 0, QTableWidgetItem(loc_str))
            
            # Index value
            index_value = result.get('index_value', 0.0)
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{index_value:.3f}"))
            
            # Class
            index_class = result.get('index_class', 'Unknown')
            self.results_table.setItem(row, 2, QTableWidgetItem(index_class))
            
            # Contributing events
            n_events = result.get('contributing_events', 0)
            self.results_table.setItem(row, 3, QTableWidgetItem(str(n_events)))
            
            # Notes (truncated)
            notes = result.get('notes', '')
            notes_short = notes[:50] + "..." if len(notes) > 50 else notes
            self.results_table.setItem(row, 4, QTableWidgetItem(notes_short))
            
            # Color code by class
            if index_class == 'Low':
                color = "#90EE90"  # Light green
            elif index_class == 'Moderate':
                color = "#FFD700"  # Gold
            elif index_class == 'High':
                color = "#FFA500"  # Orange
            else:  # Extreme
                color = "#FF6347"  # Tomato
            
            # Apply color (simplified, would use proper styling)

