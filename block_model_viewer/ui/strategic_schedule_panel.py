"""
Strategic Schedule Panel (STEP 30)

Annual LOM scheduling with MILP, nested shells, and cutoff optimization.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from datetime import date, timedelta

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView,
    QMessageBox
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


def _has_model_data(model: Any) -> bool:
    """Safe model presence check that handles pandas DataFrames."""
    if model is None:
        return False
    try:
        return not bool(getattr(model, "empty"))
    except Exception:
        return True


class StrategicSchedulePanel(BaseAnalysisPanel):
    """
    Panel for strategic scheduling (annual LOM).
    """
    
    task_name = "strategic_milp_schedule"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="strategic_schedule")
        self.schedule_result = None
        self._block_model = None  # Use _block_model instead of block_model property
        self.pit_results = None
        
        # Subscribe to block model and pit optimization updates from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.blockModelClassified.connect(self._on_block_model_loaded)
            self.registry.pitOptimizationResultsLoaded.connect(self._on_pit_optimization_results_loaded)
            
            # Prefer classified block model when available.
            existing_block_model = self.registry.get_classified_block_model()
            if not _has_model_data(existing_block_model):
                existing_block_model = self.registry.get_block_model()
            if _has_model_data(existing_block_model):
                self._on_block_model_loaded(existing_block_model)
            
            existing_pit = self.registry.get_pit_optimization_results()
            if existing_pit:
                self._on_pit_optimization_results_loaded(existing_pit)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        logger.info("Initialized Strategic Schedule Panel")
    


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
        """Setup the user interface."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Strategic Schedule Panel: NPV-maximizing annual LOM schedule using MILP, nested shells, or cutoff optimization."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Planning Horizon
        horizon_group = QGroupBox("Planning Horizon")
        horizon_layout = QFormLayout()
        
        self.num_years = QSpinBox()
        self.num_years.setRange(1, 50)
        self.num_years.setValue(20)
        horizon_layout.addRow("Number of Years:", self.num_years)
        
        self.start_year = QSpinBox()
        self.start_year.setRange(2000, 2100)
        self.start_year.setValue(2025)
        horizon_layout.addRow("Start Year:", self.start_year)
        
        horizon_group.setLayout(horizon_layout)
        layout.addWidget(horizon_group)
        
        # Schedule Configuration
        config_group = QGroupBox("Schedule Configuration")
        config_layout = QFormLayout()
        
        self.schedule_method = QComboBox()
        self.schedule_method.addItems(["MILP", "Nested Shells", "Cutoff Optimization"])
        config_layout.addRow("Method:", self.schedule_method)
        
        self.value_field = QComboBox()
        self.value_field.addItems(["value", "geomet_value"])
        config_layout.addRow("Value Field:", self.value_field)
        
        self.discount_rate = QDoubleSpinBox()
        self.discount_rate.setRange(0.0, 1.0)
        self.discount_rate.setSingleStep(0.01)
        self.discount_rate.setValue(0.10)
        self.discount_rate.setSuffix(" (10% = 0.10)")
        config_layout.addRow("Discount Rate:", self.discount_rate)
        
        self.mine_capacity = QDoubleSpinBox()
        self.mine_capacity.setRange(0.0, 100000000.0)
        self.mine_capacity.setValue(10000000.0)
        self.mine_capacity.setSuffix(" tpy")
        config_layout.addRow("Mining Capacity:", self.mine_capacity)
        
        self.plant_capacity = QDoubleSpinBox()
        self.plant_capacity.setRange(0.0, 100000000.0)
        self.plant_capacity.setValue(8000000.0)
        self.plant_capacity.setSuffix(" tpy")
        config_layout.addRow("Plant Capacity:", self.plant_capacity)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Strategic Schedule")
        self.run_btn.clicked.connect(self._on_run_schedule)
        button_layout.addWidget(self.run_btn)
        
        self.export_btn = QPushButton("Export Schedule")
        self.export_btn.clicked.connect(self._on_export)
        button_layout.addWidget(self.export_btn)
        
        self.save_scenario_btn = QPushButton("Save as Scenario")
        self.save_scenario_btn.clicked.connect(self._on_save_as_scenario)
        button_layout.addWidget(self.save_scenario_btn)
        
        layout.addLayout(button_layout)
        
        # Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Period", "Tonnes", "Grade", "Value", "NPV"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("Schedule Results:"))
        layout.addWidget(self.results_table)
        
        # Summary
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(QLabel("Summary:"))
        layout.addWidget(self.summary_text)
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        from ..mine_planning.scheduling.types import TimePeriod
        
        # Create periods
        periods = []
        for year in range(self.num_years.value()):
            start = date(self.start_year.value() + year, 1, 1)
            end = date(self.start_year.value() + year + 1, 1, 1) - timedelta(days=1)
            period = TimePeriod(
                id=f"Y{year+1:02d}",
                index=year,
                start_date=start,
                end_date=end,
                duration_days=365.0
            )
            periods.append(period)
        
        return {
            "periods": [{"id": p.id, "index": p.index, "duration_days": p.duration_days} for p in periods],
            "discount_rate": self.discount_rate.value(),
            "mine_capacity_tpy": self.mine_capacity.value(),
            "plant_capacity_tpy": self.plant_capacity.value(),
            "block_source": "long_model",
            "block_value_field": self.value_field.currentText()
        }
    
    def validate_inputs(self) -> bool:
        """
        Validate inputs for strategic scheduling.
        
        MP-006 FIX: Comprehensive validation including:
        - Block model presence
        - Value field existence
        - Economic parameter sanity
        - Classification status check
        """
        # Check controller and block model
        if not self.controller:
            self.show_warning("No Controller", "Controller not available.")
            return False
        
        if not self.controller.current_block_model:
            self.show_warning("No Model", "Please load a block model first.")
            return False
        
        block_model = self.controller.current_block_model
        
        # MP-011 related: Validate value field exists
        value_field = self.value_field.currentText()
        if hasattr(block_model, 'get_property_names'):
            available_props = block_model.get_property_names()
        elif hasattr(block_model, 'columns'):
            available_props = list(block_model.columns)
        else:
            available_props = []
        
        if value_field and value_field not in available_props:
            # Try case-insensitive match
            matched = [p for p in available_props if p.lower() == value_field.lower()]
            if not matched:
                self.show_warning(
                    "Value Field Missing",
                    f"Value field '{value_field}' not found in block model.\n\n"
                    f"Available properties: {', '.join(available_props[:10])}{'...' if len(available_props) > 10 else ''}\n\n"
                    "Please select a valid value field or run pit optimization first."
                )
                return False
        
        # Validate economic parameters
        discount_rate = self.discount_rate.value()
        if discount_rate < 0 or discount_rate > 1:
            self.show_warning("Invalid Discount Rate", "Discount rate must be between 0 and 1.")
            return False
        
        mining_capacity = self.mine_capacity.value()
        plant_capacity = self.plant_capacity.value()
        
        if mining_capacity <= 0:
            self.show_warning("Invalid Mining Capacity", "Mining capacity must be positive.")
            return False
        
        if plant_capacity <= 0:
            self.show_warning("Invalid Plant Capacity", "Plant capacity must be positive.")
            return False
        
        # Check if block model has tonnage for scheduling
        tonnage_fields = ['TONNAGE', 'TONNES', 'T', 'tonnage', 'tonnes']
        has_tonnage = any(t in available_props for t in tonnage_fields)
        if not has_tonnage:
            response = QMessageBox.warning(
                self, "Missing Tonnage",
                "Block model does not have a TONNAGE field.\n\n"
                "Scheduling requires tonnage per block. Results may be invalid.\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if response != QMessageBox.StandardButton.Yes:
                return False
        
        return True
    
    def _on_run_schedule(self):
        """Run strategic schedule."""
        if not self.validate_inputs():
            return
        
        method = self.schedule_method.currentText()
        params = {
            "block_model": self.controller.current_block_model,
            "config": self.gather_parameters()
        }
        
        if method == "MILP":
            self.controller.run_strategic_schedule(params, self._on_schedule_complete)
        elif method == "Nested Shells":
            # MP-016 FIX: Validate shell tonnage for nested shells
            if self.pit_results is None or not self.pit_results.get("shell_tonnage"):
                response = QMessageBox.warning(
                    self, "Shell Tonnage Missing",
                    "Nested shell scheduling requires pit shell tonnage data.\n\n"
                    "Please run pit optimization with nested shells first, "
                    "or provide shell tonnage data.\n\n"
                    "Continue with empty shell tonnage? (may produce invalid results)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if response != QMessageBox.StandardButton.Yes:
                    return
                params["shell_tonnage"] = {}
            else:
                params["shell_tonnage"] = self.pit_results["shell_tonnage"]
            self.controller.run_nested_shell_schedule(params, self._on_schedule_complete)
        else:  # Cutoff
            params["config"]["candidate_cutoffs"] = [30.0, 40.0, 50.0, 60.0]
            self.controller.run_cutoff_schedule(params, self._on_schedule_complete)
    
    def _on_schedule_complete(self, result: Dict[str, Any]):
        """Handle schedule completion."""
        schedule_result = result.get("result")
        if schedule_result:
            self.schedule_result = schedule_result
            self._display_results(schedule_result)
        else:
            self.show_warning("No Result", "Schedule completed but no result returned.")
    
    def _display_results(self, schedule_result):
        """Display schedule results."""
        from ..mine_planning.scheduling.types import ScheduleResult
        
        if not isinstance(schedule_result, ScheduleResult):
            self.show_warning("Invalid Result", "Schedule result is not a valid ScheduleResult object.")
            return
        
        # Populate table
        tonnes_by_period = schedule_result.get_tonnes_by_period()
        
        self.results_table.setRowCount(len(tonnes_by_period))
        
        for row, (period_id, tonnes) in enumerate(tonnes_by_period.items()):
            self.results_table.setItem(row, 0, QTableWidgetItem(period_id))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{tonnes:,.0f}"))
            # Would add grade and value columns
        
        # Summary
        total_tonnes = schedule_result.get_total_tonnes()
        npv = schedule_result.metadata.get("objective_value", 0.0)
        
        self.summary_text.setText(
            f"Total Tonnes: {total_tonnes:,.0f}\n"
            f"NPV: ${npv:,.0f}\n"
            f"Periods: {len(schedule_result.periods)}\n"
            f"Decisions: {len(schedule_result.decisions)}"
        )
        
        # Publish schedule to DataRegistry
        try:
            if hasattr(self, 'registry') and self.registry:
                self.registry.register_schedule(schedule_result, source_panel="StrategicSchedulePanel")
                logger.info("Published strategic schedule to DataRegistry")
        except Exception as e:
            logger.warning(f"Failed to publish strategic schedule to DataRegistry: {e}")
    
    def _on_export(self):
        """Export schedule."""
        if not self.schedule_result:
            self.show_warning("No Schedule", "Please run schedule first.")
            return
        
        self.show_info("Export", "Export functionality would be implemented here.")
    
    def _on_save_as_scenario(self):
        """Save current configuration as planning scenario (STEP 31)."""
        if not self.controller:
            self.show_warning("No Controller", "Controller not available.")
            return
        
        # Build context from current panel state
        context = {
            "name": f"strategic_schedule_{self.start_year.value()}",
            "version": "draft",
            "description": f"Strategic schedule from {self.start_year.value()}",
            "model_name": "default",  # Would get from controller
            "value_mode": self.value_field.currentText().split("_")[0] if "_" in self.value_field.currentText() else "base",
            "value_field": self.value_field.currentText(),
            "schedule_config": {
                "strategic": {
                    "enabled": True,
                    "method": self.schedule_method.currentText(),
                    "discount_rate": self.discount_rate.value(),
                    "mine_capacity_tpy": self.mine_capacity.value(),
                    "plant_capacity_tpy": self.plant_capacity.value(),
                }
            }
        }
        
        try:
            scenario = self.controller.create_scenario_from_context(context)
            self.show_info("Scenario Saved", f"Scenario '{scenario.id.name}' saved successfully.")
        except Exception as e:
            self.show_error("Save Failed", f"Failed to save scenario:\n{e}")
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass
    
    def _on_block_model_generated(self, block_model):
        """Handle block model generation."""
        self._on_block_model_loaded(block_model)
    
    def _on_block_model_loaded(self, block_model):
        """
        Handle block model loading.
        
        Updates value field combo with available properties.
        """
        self._block_model = block_model
        logger.info(f"Strategic Schedule Panel: Block model loaded")
        
        # Update value field combo with available properties
        if hasattr(block_model, 'get_property_names'):
            available_props = block_model.get_property_names()
        elif hasattr(block_model, 'columns'):
            available_props = list(block_model.columns)
        else:
            available_props = []
        
        # Identify likely value fields
        value_candidates = []
        value_keywords = ['VALUE', 'NPV', 'LG_VALUE', 'BLOCK_VALUE', 'GEOMET_VALUE']
        for prop in available_props:
            if any(kw in prop.upper() for kw in value_keywords):
                value_candidates.append(prop)
        
        # Update combo
        self.value_field.clear()
        if value_candidates:
            self.value_field.addItems(value_candidates)
        else:
            # Add some generic options
            self.value_field.addItems(["value", "geomet_value"])
            logger.warning("No value fields found in block model - using defaults")
        
        # Invalidate previous schedule result
        self.schedule_result = None
    
    def _on_pit_optimization_results_loaded(self, pit_results):
        """
        Handle pit optimization results.
        
        Updates shell tonnage data for nested shell scheduling.
        """
        self.pit_results = pit_results
        logger.info("Strategic Schedule Panel: Pit optimization results loaded")
        
        # If pit results contain value field, ensure it's in the combo
        if pit_results and 'result_df' in pit_results:
            result_df = pit_results['result_df']
            if hasattr(result_df, 'columns') and 'LG_VALUE' in result_df.columns:
                current_items = [self.value_field.itemText(i) for i in range(self.value_field.count())]
                if 'LG_VALUE' not in current_items:
                    self.value_field.addItem('LG_VALUE')

