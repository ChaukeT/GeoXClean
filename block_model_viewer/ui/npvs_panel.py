"""
NPVS Optimisation Panel (STEP 32)

UI panel for NPVS (Net Present Value Scheduling) optimization.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QPushButton, QDoubleSpinBox,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import pyqtSlot

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class NPVSPanel(BaseAnalysisPanel):
    """
    NPVS Optimisation Panel
    
    - Builds period structure
    - Configures destinations (plant, stockpile, waste)
    - Defines capacities, recoveries
    - Calls AppController.run_npvs()
    """
    PANEL_ID = "NPVSPanel"  # STEP 40
    task_name = "npvs_run"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="npvs")
        
        # Subscribe to block model and pit optimization updates from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.pitOptimizationResultsLoaded.connect(self._on_pit_optimization_results_loaded)
            
            # Load existing data if available
            existing_block_model = self.registry.get_block_model()
            if existing_block_model:
                self._on_block_model_loaded(existing_block_model)
            
            existing_pit = self.registry.get_pit_optimization_results()
            if existing_pit:
                self._on_pit_optimization_results_loaded(existing_pit)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self._build_ui()
    


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
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        logger.info("NPVS Panel received block model from DataRegistry")
        self._block_model = block_model  # Use private backing field (property contract)
        
        # Update property combo with available value fields
        if hasattr(block_model, 'get_property_names'):
            available_props = block_model.get_property_names()
        elif hasattr(block_model, 'columns'):
            available_props = list(block_model.columns)
        else:
            available_props = []
        
        # Identify likely value fields
        value_keywords = ['VALUE', 'NPV', 'BLOCK_VALUE', 'GEOMET_VALUE', 'LG_VALUE']
        value_candidates = []
        for prop in available_props:
            if any(kw in prop.upper() for kw in value_keywords):
                value_candidates.append(prop)
        
        # Update combo
        if hasattr(self, 'property_combo'):
            self.property_combo.clear()
            if value_candidates:
                self.property_combo.addItems(value_candidates)
                logger.info(f"NPVS Panel: Found value fields: {value_candidates}")
            else:
                # Add defaults
                self.property_combo.addItems(["block_value", "geomet_value"])
                logger.warning("NPVS Panel: No value fields found - using defaults")
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def _on_pit_optimization_results_loaded(self, pit_results):
        """
        Automatically receive pit optimization results when they're loaded.
        
        Args:
            pit_results: Pit optimization results from DataRegistry
        """
        logger.info("NPVS Panel received pit optimization results from DataRegistry")
        self.pit_results = pit_results
    
    # --------------------------------------------------------
    # UI BUILD
    # --------------------------------------------------------
    
    def _build_ui(self):
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "NPVS Optimisation: Maximize Net Present Value by scheduling blocks "
            "across periods with destination routing (plant, stockpile, waste)."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # 1. Block model and value selector
        prop_group = self._build_property_selector_group()
        if prop_group:
            layout.addWidget(prop_group)
        
        # 2. Period configuration
        layout.addWidget(self._build_period_group())
        
        # 3. Destination configuration
        layout.addWidget(self._build_destinations_group())
        
        # 4. Capacities
        layout.addWidget(self._build_capacity_group())
        
        # 5. Run button
        self.run_btn = QPushButton("Run NPVS Optimisation")
        self.run_btn.clicked.connect(self._run_npvs_clicked)
        layout.addWidget(self.run_btn)
        
        # 6. Output table
        self.result_table = QTableWidget()
        layout.addWidget(QLabel("<b>Results:</b>"))
        layout.addWidget(self.result_table)
    
    # --------------------------------------------------------
    # Subsections
    # --------------------------------------------------------
    
    def _build_property_selector_group(self) -> QWidget:
        """Build property selector group."""
        box = QGroupBox("Block Model & Value Field")
        lay = QVBoxLayout(box)
        
        self.property_combo = QComboBox()
        self.property_combo.addItems(["block_value", "geomet_value"])
        lay.addWidget(QLabel("Value Field:"))
        lay.addWidget(self.property_combo)
        
        return box
    
    def _build_period_group(self) -> QWidget:
        """Build period configuration group."""
        box = QGroupBox("Periods")
        lay = QHBoxLayout(box)
        
        self.num_periods = QSpinBox()
        self.num_periods.setValue(10)
        self.num_periods.setMinimum(1)
        self.num_periods.setMaximum(50)
        
        self.discount_rate = QDoubleSpinBox()
        self.discount_rate.setValue(0.08)
        self.discount_rate.setDecimals(4)
        self.discount_rate.setSingleStep(0.01)
        self.discount_rate.setRange(0.0, 1.0)
        
        lay.addWidget(QLabel("Number of periods:"))
        lay.addWidget(self.num_periods)
        
        lay.addWidget(QLabel("Discount rate:"))
        lay.addWidget(self.discount_rate)
        
        lay.addStretch()
        
        return box
    
    def _build_destinations_group(self) -> QWidget:
        """
        Config table:
           Destination | Type | Capacity t/y | Recovery Cu | Recovery Au | Proc Cost
        """
        box = QGroupBox("Destinations")
        lay = QVBoxLayout(box)
        
        self.dest_table = QTableWidget(3, 6)  # plant, stockpile, waste
        self.dest_table.setHorizontalHeaderLabels([
            "Destination", "Type", "Capacity t/y",
            "Rec Cu", "Rec Au", "Proc Cost"
        ])
        self.dest_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # default rows
        defaults = [
            ("plant", "plant", 10_000_000, 0.90, 0.85, 12),
            ("stockpile_lg", "stockpile", 30_000_000, 0, 0, 0),
            ("waste", "waste", 1_000_000_000, 0, 0, 0),
        ]
        for r, row in enumerate(defaults):
            for c, val in enumerate(row):
                self.dest_table.setItem(r, c, QTableWidgetItem(str(val)))
        
        lay.addWidget(self.dest_table)
        return box
    
    def _build_capacity_group(self) -> QWidget:
        """Build capacity configuration group."""
        box = QGroupBox("Global Capacities")
        lay = QHBoxLayout(box)
        
        self.mining_capacity = QDoubleSpinBox()
        self.mining_capacity.setRange(0, 1e12)
        self.mining_capacity.setValue(15_000_000)
        self.mining_capacity.setSuffix(" t/y")
        
        self.plant_capacity = QDoubleSpinBox()
        self.plant_capacity.setRange(0, 1e12)
        self.plant_capacity.setValue(10_000_000)
        self.plant_capacity.setSuffix(" t/y")
        
        lay.addWidget(QLabel("Mining capacity:"))
        lay.addWidget(self.mining_capacity)
        
        lay.addWidget(QLabel("Plant capacity:"))
        lay.addWidget(self.plant_capacity)
        
        lay.addStretch()
        
        return box
    
    # --------------------------------------------------------
    # Build NPVS config payload
    # --------------------------------------------------------
    
    def _collect_periods(self) -> list[dict]:
        """Collect period configuration."""
        n = self.num_periods.value()
        r = self.discount_rate.value()
        
        periods = []
        for i in range(n):
            periods.append({
                "id": f"Y{i+1:02}",
                "index": i,
                "duration_years": 1.0,
                "discount_factor": 1 / ((1+r)**i)
            })
        return periods
    
    def _collect_destinations(self) -> list[dict]:
        """Collect destination configuration."""
        rows = self.dest_table.rowCount()
        dests = []
        
        for r in range(rows):
            dest_id = self.dest_table.item(r, 0)
            dest_type = self.dest_table.item(r, 1)
            capacity = self.dest_table.item(r, 2)
            rec_cu = self.dest_table.item(r, 3)
            rec_au = self.dest_table.item(r, 4)
            proc_cost = self.dest_table.item(r, 5)
            
            if not all([dest_id, dest_type, capacity, rec_cu, rec_au, proc_cost]):
                continue
            
            dests.append({
                "id": dest_id.text(),
                "type": dest_type.text(),
                "capacity_tpy": float(capacity.text()),
                "recovery_by_element": {
                    "Cu": float(rec_cu.text()),
                    "Au": float(rec_au.text()),
                },
                "processing_cost_per_t": float(proc_cost.text())
            })
        
        return dests
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters for NPVS optimization."""
        value_field = self.property_combo.currentText()
        
        config = {
            "periods": self._collect_periods(),
            "destinations": self._collect_destinations(),
            "discount_rate": self.discount_rate.value(),
            "mining_capacity_tpy": self.mining_capacity.value(),
            "plant_capacity_tpy": self.plant_capacity.value(),
        }
        
        return {
            "block_model_property": value_field,
            "config": config,
        }
    
    def validate_inputs(self) -> bool:
        """
        Validate inputs before running.
        
        MP-011 FIX: Comprehensive validation including value field existence check.
        """
        if self.num_periods.value() < 1:
            self.show_warning("Invalid Input", "Number of periods must be at least 1.")
            return False
        
        if self.discount_rate.value() < 0 or self.discount_rate.value() >= 1.0:
            self.show_warning("Invalid Input", "Discount rate must be between 0 and 1.")
            return False
        
        destinations = self._collect_destinations()
        if len(destinations) == 0:
            self.show_warning("Invalid Input", "At least one destination must be configured.")
            return False
        
        # Validate destination capacities
        for dest in destinations:
            if dest['capacity_tpy'] <= 0 and dest['type'] not in ['waste']:
                self.show_warning(
                    "Invalid Destination",
                    f"Destination '{dest['id']}' has zero or negative capacity."
                )
                return False
        
        # MP-011 FIX: Validate value field exists in block model
        if not self.controller:
            self.show_warning("No Controller", "Controller not available.")
            return False
        
        block_model = self.controller.block_model
        if block_model is None:
            self.show_warning("No Block Model", "Please load a block model first.")
            return False
        
        value_field = self.property_combo.currentText()
        
        # Get available properties
        if hasattr(block_model, 'get_property_names'):
            available_props = block_model.get_property_names()
        elif hasattr(block_model, 'columns'):
            available_props = list(block_model.columns)
        else:
            available_props = []
        
        # Check if value field exists (case-insensitive)
        value_field_found = any(p.lower() == value_field.lower() for p in available_props)
        if not value_field_found:
            self.show_warning(
                "Value Field Missing",
                f"Value field '{value_field}' not found in block model.\n\n"
                f"Available properties: {', '.join(available_props[:15])}{'...' if len(available_props) > 15 else ''}\n\n"
                "Please run pit optimization or geomet evaluation first to calculate block values."
            )
            return False
        
        # Check for tonnage field
        tonnage_fields = ['TONNAGE', 'TONNES', 'T', 'tonnage', 'tonnes']
        has_tonnage = any(t.lower() in [p.lower() for p in available_props] for t in tonnage_fields)
        if not has_tonnage:
            logger.warning("NPVS: No TONNAGE field found - scheduling may fail or produce incorrect results")
        
        return True
    
    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------
    
    @pyqtSlot()
    def _run_npvs_clicked(self):
        """Handle run button click."""
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        if not self.validate_inputs():
            return
        
        params = self.gather_parameters()
        
        # Get block model from controller
        block_model = self.controller.block_model
        if block_model is None:
            self.show_error("No Block Model", "Please load a block model first.")
            return
        
        params["block_model"] = block_model
        
        self.show_progress("Running NPVS optimisation...")
        
        try:
            self.controller.run_npvs(params, self._on_npvs_done)
        except Exception as e:
            logger.error(f"Failed to run NPVS: {e}", exc_info=True)
            self.hide_progress()
            self.show_error("NPVS Error", f"Failed to run NPVS optimisation:\n{e}")
    
    @pyqtSlot(dict)
    def _on_npvs_done(self, result: dict):
        """
        Handle NPVS completion.
        
        result = {
            "schedule": ScheduleResult serialized
            "npv": float,
        }
        """
        self.hide_progress()
        
        # Handle result payload structure
        if isinstance(result, dict) and "result" in result:
            # Result wrapped in payload
            actual_result = result["result"]
        else:
            actual_result = result
        
        if actual_result.get("error"):
            self.show_error("NPVS Error", actual_result["error"])
            return
        
        self.emit_status("NPVS complete")
        self._fill_table(actual_result)
    
    # --------------------------------------------------------
    # Output table
    # --------------------------------------------------------
    
    def _fill_table(self, result: dict):
        """Fill results table."""
        schedule = result.get("schedule", {})
        npv = result.get("npv", None)
        
        decisions = schedule.get("decisions", [])
        
        self.result_table.clear()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels([
            "Period", "Block", "Tonnes", "Destination", "Value"
        ])
        self.result_table.setRowCount(len(decisions))
        
        for r, d in enumerate(decisions):
            self.result_table.setItem(r, 0, QTableWidgetItem(str(d.get("period_id", ""))))
            self.result_table.setItem(r, 1, QTableWidgetItem(str(d.get("unit_id", ""))))
            self.result_table.setItem(r, 2, QTableWidgetItem(str(round(d.get("tonnes", 0), 2))))
            self.result_table.setItem(r, 3, QTableWidgetItem(str(d.get("destination", ""))))
            # Note: ScheduleDecision doesn't have a value field, so we'll leave it empty or calculate from tonnes
            self.result_table.setItem(r, 4, QTableWidgetItem(""))  # Value not available in decision
        
        self.result_table.resizeColumnsToContents()
        
        if npv is not None:
            self.show_info("NPVS Complete", f"NPVS Optimisation completed — NPV = ${npv:,.2f}")
            
            # Publish schedule to DataRegistry
            try:
                if hasattr(self, 'registry') and self.registry:
                    schedule_data = {
                        'schedule': schedule,
                        'npv': npv,
                        'decisions': decisions
                    }
                    self.registry.register_schedule(schedule_data, source_panel="NPVSPanel")
                    logger.info("Published NPVS schedule to DataRegistry")
            except Exception as e:
                logger.warning(f"Failed to publish NPVS schedule to DataRegistry: {e}")
            
            # Add button to open Production Dashboard
            if not hasattr(self, 'prod_dashboard_btn'):
                self.prod_dashboard_btn = QPushButton("Open Production Dashboard with this schedule")
                self.prod_dashboard_btn.clicked.connect(self._on_open_production_dashboard)
                self.main_layout.addWidget(self.prod_dashboard_btn)
    
    @pyqtSlot()
    def _on_open_production_dashboard(self):
        """Open Production Dashboard with current NPVS schedule."""
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        try:
            from .production_dashboard_panel import ProductionDashboardPanel
            
            # Open production dashboard
            if not hasattr(self.controller, 'production_dashboard_dialog') or self.controller.production_dashboard_dialog is None:
                dialog = ProductionDashboardPanel(self)
                dialog.bind_controller(self.controller)
                self.controller.production_dashboard_dialog = dialog
            
            # Pre-populate with current schedule
            # Would pass schedule result reference here
            self.controller.production_dashboard_dialog.show()
            self.controller.production_dashboard_dialog.raise_()
            self.controller.production_dashboard_dialog.activateWindow()
            
            self.show_info("Production Dashboard", "Opened Production Dashboard")
        
        except Exception as e:
            logger.error(f"Failed to open Production Dashboard: {e}", exc_info=True)
            self.show_error("Error", f"Failed to open Production Dashboard:\n{e}")
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results from controller."""
        self._on_npvs_done(payload)

