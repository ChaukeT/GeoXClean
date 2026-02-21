"""
Schedule Risk Panel

Panel for building period-by-period risk profiles from mine schedules.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QCheckBox, QTextEdit
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


class ScheduleRiskPanel(BaseAnalysisPanel):
    """
    Schedule Risk Analysis Panel.
    
    Provides:
    - Schedule selection (pit / UG / scenario)
    - Hazard source selection
    - Aggregation configuration
    - Risk profile generation
    - Results table display
    """
    
    task_name = "schedule_risk_profile"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="schedule_risk")
        
        self.schedule = None
        
        # Subscribe to schedule from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.scheduleGenerated.connect(self._on_schedule_generated)
            
            # Load existing schedule if available
            existing_schedule = self.registry.get_schedule()
            if existing_schedule:
                self._on_schedule_generated(existing_schedule)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self.current_profile = None
        self._setup_ui()
        logger.info("Initialized Schedule Risk panel")
    
    def _on_schedule_generated(self, schedule):
        """
        Automatically receive schedule when it's generated.
        
        Args:
            schedule: Production schedule from DataRegistry
        """
        logger.info("Schedule Risk Panel received schedule from DataRegistry")
        self.schedule = schedule
    
    def setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("<b>Schedule Risk Analysis</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # Schedule selection
        schedule_group = QGroupBox("Schedule Selection")
        schedule_layout = QFormLayout(schedule_group)
        
        self.schedule_type_combo = QComboBox()
        self.schedule_type_combo.addItems(["Pit Schedule", "UG Schedule", "Custom Schedule"])
        schedule_layout.addRow("Schedule Type:", self.schedule_type_combo)
        
        self.schedule_id_edit = QTextEdit()
        self.schedule_id_edit.setMaximumHeight(60)
        self.schedule_id_edit.setPlaceholderText("Enter schedule ID or select from available schedules...")
        schedule_layout.addRow("Schedule ID:", self.schedule_id_edit)
        
        layout.addWidget(schedule_group)
        
        # Hazard sources
        hazard_group = QGroupBox("Hazard Sources")
        hazard_layout = QVBoxLayout(hazard_group)
        
        self.use_seismic_checkbox = QCheckBox("Use Seismic Hazard Volume")
        self.use_seismic_checkbox.setChecked(True)
        hazard_layout.addWidget(self.use_seismic_checkbox)
        
        self.use_rockburst_checkbox = QCheckBox("Use Rockburst Index Results")
        self.use_rockburst_checkbox.setChecked(True)
        hazard_layout.addWidget(self.use_rockburst_checkbox)
        
        self.use_slope_risk_checkbox = QCheckBox("Use Slope Risk Results")
        hazard_layout.addWidget(self.use_slope_risk_checkbox)
        
        layout.addWidget(hazard_group)
        
        # Aggregation settings
        agg_group = QGroupBox("Aggregation Settings")
        agg_layout = QFormLayout(agg_group)
        
        self.aggregation_combo = QComboBox()
        self.aggregation_combo.addItems(["mean", "max", "tonnage_weighted"])
        agg_layout.addRow("Method:", self.aggregation_combo)
        
        # Risk weights
        self.seismic_weight_spinbox = QDoubleSpinBox()
        self.seismic_weight_spinbox.setRange(0.0, 1.0)
        self.seismic_weight_spinbox.setValue(0.4)
        self.seismic_weight_spinbox.setSingleStep(0.1)
        agg_layout.addRow("Seismic Weight:", self.seismic_weight_spinbox)
        
        self.rockburst_weight_spinbox = QDoubleSpinBox()
        self.rockburst_weight_spinbox.setRange(0.0, 1.0)
        self.rockburst_weight_spinbox.setValue(0.4)
        self.rockburst_weight_spinbox.setSingleStep(0.1)
        agg_layout.addRow("Rockburst Weight:", self.rockburst_weight_spinbox)
        
        self.slope_weight_spinbox = QDoubleSpinBox()
        self.slope_weight_spinbox.setRange(0.0, 1.0)
        self.slope_weight_spinbox.setValue(0.2)
        self.slope_weight_spinbox.setSingleStep(0.1)
        agg_layout.addRow("Slope Weight:", self.slope_weight_spinbox)
        
        self.period_days_spinbox = QDoubleSpinBox()
        self.period_days_spinbox.setRange(1.0, 365.0)
        self.period_days_spinbox.setValue(30.0)
        self.period_days_spinbox.setSuffix(" days")
        agg_layout.addRow("Period Length:", self.period_days_spinbox)
        
        layout.addWidget(agg_group)
        
        # Run analysis
        run_btn = QPushButton("Build Risk Profile")
        run_btn.clicked.connect(self._build_risk_profile)
        layout.addWidget(run_btn)
        
        # Results table
        results_group = QGroupBox("Risk Profile Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Period", "Tonnage", "Metal", "Seismic", "Rockburst", "Slope", "Combined", "Notes"
        ])
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect risk profile parameters."""
        schedule_id = self.schedule_id_edit.toPlainText().strip() or "schedule_1"
        
        return {
            'schedule_id': schedule_id,
            'schedule_type': self.schedule_type_combo.currentText().lower().replace(' ', '_'),
            'aggregation_method': self.aggregation_combo.currentText(),
            'risk_weights': {
                'seismic_weight': self.seismic_weight_spinbox.value(),
                'rockburst_weight': self.rockburst_weight_spinbox.value(),
                'slope_weight': self.slope_weight_spinbox.value()
            },
            'period_days': self.period_days_spinbox.value(),
            'use_seismic': self.use_seismic_checkbox.isChecked(),
            'use_rockburst': self.use_rockburst_checkbox.isChecked(),
            'use_slope_risk': self.use_slope_risk_checkbox.isChecked()
        }
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        if not self.validate_block_model_loaded():
            return False
        
        # Check weights sum to ~1.0
        total_weight = (
            self.seismic_weight_spinbox.value() +
            self.rockburst_weight_spinbox.value() +
            self.slope_weight_spinbox.value()
        )
        
        if abs(total_weight - 1.0) > 0.01:
            self.show_warning("Weights", "Weights should sum to approximately 1.0. Normalizing...")
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle risk profile results."""
        if payload.get('error'):
            self.show_error("Risk Profile Error", payload['error'])
            return
        
        profile = payload.get('profile')
        if profile:
            self.current_profile = profile
            self._update_results_table()
            self.show_info("Success", f"Risk profile built: {len(profile.periods)} periods")
        else:
            self.show_error("Error", "No profile returned from analysis")
    
    def _build_risk_profile(self):
        """Build risk profile."""
        self.run_analysis()
    
    def _update_results_table(self):
        """Update results table with period risk data."""
        if not self.current_profile:
            return
        
        periods = self.current_profile.periods
        self.results_table.setRowCount(len(periods))
        
        for row, period in enumerate(periods):
            # Period index
            self.results_table.setItem(row, 0, QTableWidgetItem(str(period.period_index)))
            
            # Tonnage
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{period.mined_tonnage:,.0f}"))
            
            # Metal
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{period.metal:,.2f}"))
            
            # Seismic hazard
            seismic_val = f"{period.seismic_hazard_index:.3f}" if period.seismic_hazard_index is not None else "N/A"
            self.results_table.setItem(row, 3, QTableWidgetItem(seismic_val))
            
            # Rockburst risk
            rockburst_val = f"{period.rockburst_risk_index:.3f}" if period.rockburst_risk_index is not None else "N/A"
            self.results_table.setItem(row, 4, QTableWidgetItem(rockburst_val))
            
            # Slope risk
            slope_val = f"{period.slope_risk_index:.2f}" if period.slope_risk_index is not None else "N/A"
            self.results_table.setItem(row, 5, QTableWidgetItem(slope_val))
            
            # Combined risk
            combined_val = f"{period.combined_risk_score:.3f}" if period.combined_risk_score is not None else "N/A"
            item = QTableWidgetItem(combined_val)
            
            # Color code by risk level
            if period.combined_risk_score is not None:
                if period.combined_risk_score < 0.25:
                    item.setBackground(Qt.GlobalColor.green)
                elif period.combined_risk_score < 0.5:
                    item.setBackground(Qt.GlobalColor.yellow)
                elif period.combined_risk_score < 0.75:
                    item.setBackground(Qt.GlobalColor.orange)
                else:
                    item.setBackground(Qt.GlobalColor.red)
            
            self.results_table.setItem(row, 6, item)
            
            # Notes
            notes_short = period.notes[:50] + "..." if len(period.notes) > 50 else period.notes
            self.results_table.setItem(row, 7, QTableWidgetItem(notes_short))

