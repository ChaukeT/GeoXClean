"""
Slope Risk Assessment Panel

Panel for evaluating slope stability risk indicators.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QCheckBox, QTableWidget, QTableWidgetItem
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class SlopeRiskPanel(BaseAnalysisPanel):
    """
    Panel for slope risk assessment.
    
    Allows users to:
    - Define slope sectors
    - Set geometry and rock mass properties
    - Run deterministic and probabilistic risk analysis
    - View risk classifications and recommendations
    """
    # PanelManager metadata
    PANEL_ID = "SlopeRiskPanel"
    PANEL_NAME = "SlopeRisk Panel"
    PANEL_CATEGORY = PanelCategory.GEOTECH
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "slope_risk"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="slope_risk")
        
        self.current_results = []
        self._setup_ui()
        logger.info("Initialized Slope Risk panel")
    


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
        title_label = QLabel("<b>Slope Risk Assessment</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # Input section
        input_group = QGroupBox("Slope Geometry")
        input_layout = QFormLayout(input_group)
        
        self.bench_height_spinbox = QDoubleSpinBox()
        self.bench_height_spinbox.setRange(1.0, 50.0)
        self.bench_height_spinbox.setValue(10.0)
        self.bench_height_spinbox.setSuffix(" m")
        input_layout.addRow("Bench Height:", self.bench_height_spinbox)
        
        self.slope_angle_spinbox = QDoubleSpinBox()
        self.slope_angle_spinbox.setRange(10.0, 90.0)
        self.slope_angle_spinbox.setValue(45.0)
        self.slope_angle_spinbox.setSuffix("°")
        input_layout.addRow("Overall Slope Angle:", self.slope_angle_spinbox)
        
        self.orientation_spinbox = QDoubleSpinBox()
        self.orientation_spinbox.setRange(0.0, 360.0)
        self.orientation_spinbox.setValue(0.0)
        self.orientation_spinbox.setSuffix("°")
        self.orientation_spinbox.setToolTip("Wall orientation (azimuth)")
        input_layout.addRow("Wall Orientation:", self.orientation_spinbox)
        
        layout.addWidget(input_group)
        
        # Rock mass properties
        rm_group = QGroupBox("Rock Mass Properties")
        rm_layout = QFormLayout(rm_group)
        
        self.rmr_spinbox = QDoubleSpinBox()
        self.rmr_spinbox.setRange(0.0, 100.0)
        self.rmr_spinbox.setValue(50.0)
        rm_layout.addRow("RMR:", self.rmr_spinbox)
        
        self.q_spinbox = QDoubleSpinBox()
        self.q_spinbox.setRange(0.01, 1000.0)
        self.q_spinbox.setValue(10.0)
        rm_layout.addRow("Q-value:", self.q_spinbox)
        
        layout.addWidget(rm_group)
        
        # Environmental factors
        env_group = QGroupBox("Environmental Factors")
        env_layout = QVBoxLayout(env_group)
        
        self.water_checkbox = QCheckBox("Water Present")
        env_layout.addWidget(self.water_checkbox)
        
        self.major_faults_checkbox = QCheckBox("Major Faults Present")
        env_layout.addWidget(self.major_faults_checkbox)
        
        self.adverse_joints_checkbox = QCheckBox("Adverse Joint Orientations")
        env_layout.addWidget(self.adverse_joints_checkbox)
        
        layout.addWidget(env_group)
        
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
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Sector", "Risk Index", "Risk Class", "P(Failure)"
        ])
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect slope risk parameters."""
        return {
            'bench_height': self.bench_height_spinbox.value(),
            'overall_slope_angle': self.slope_angle_spinbox.value(),
            'pit_wall_orientation': self.orientation_spinbox.value(),
            'rock_mass_properties': {
                'RMR': self.rmr_spinbox.value(),
                'Q': self.q_spinbox.value()
            },
            'water_present': self.water_checkbox.isChecked(),
            'structural_features': {
                'major_faults': self.major_faults_checkbox.isChecked(),
                'adverse_joints': self.adverse_joints_checkbox.isChecked()
            },
            'analysis_type': 'deterministic'
        }
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        if self.bench_height_spinbox.value() <= 0:
            self.show_error("Invalid Input", "Bench height must be positive.")
            return False
        if self.slope_angle_spinbox.value() <= 0 or self.slope_angle_spinbox.value() >= 90:
            self.show_error("Invalid Input", "Slope angle must be between 0 and 90 degrees.")
            return False
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle analysis results."""
        if payload.get('error'):
            self.show_error("Analysis Error", payload['error'])
            return
        
        result = payload.get('result')
        if result:
            self.current_results = [result] if not isinstance(result, list) else result
            self._update_results_table()
        
        self.show_info("Success", "Slope risk analysis completed.")
    
    def _run_deterministic(self):
        """Run deterministic analysis."""
        self.task_name = "slope_risk"
        self.run_analysis()
    
    def _run_probabilistic(self):
        """Run probabilistic analysis."""
        params = self.gather_parameters()
        params['n_realizations'] = self.n_realizations_spinbox.value()
        params['analysis_type'] = 'probabilistic'
        
        if not self.validate_inputs():
            return
        
        if self.controller:
            self.controller.run_task("slope_risk_mc", params, self.handle_results)
    
    def _update_results_table(self):
        """Update results table."""
        self.results_table.setRowCount(len(self.current_results))
        
        for row, result in enumerate(self.current_results):
            sector_name = result.get('sector_name', f"Sector {row+1}")
            risk_index = result.get('risk_index', 0.0)
            risk_class = result.get('qualitative_class', 'Unknown')
            prob_failure = result.get('probability_of_failure', 0.0)
            
            self.results_table.setItem(row, 0, QTableWidgetItem(str(sector_name)))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{risk_index:.1f}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(risk_class))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{prob_failure:.1%}"))
            
            # Color code by risk
            if risk_index < 25:
                color = "#90EE90"  # Light green
            elif risk_index < 50:
                color = "#FFD700"  # Gold
            elif risk_index < 75:
                color = "#FFA500"  # Orange
            else:
                color = "#FF6347"  # Tomato
            
            for col in range(4):
                item = self.results_table.item(row, col)
                if item:
                    item.setBackground(Qt.GlobalColor.transparent)  # Will be styled

