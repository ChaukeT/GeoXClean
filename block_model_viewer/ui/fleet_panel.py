"""
Fleet Panel (STEP 30)

Edit fleet configuration and compute cycle times and dispatch.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView,
    QMessageBox
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class FleetPanel(BaseAnalysisPanel):
    """
    Panel for fleet management and haulage.
    """
    
    task_name = "fleet_cycle_time"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="fleet")
        self.fleet_config = None
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
        
        logger.info("Initialized Fleet Panel")
    


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
    def _on_schedule_generated(self, schedule):
        """
        Automatically receive schedule when it's generated.
        
        Args:
            schedule: Production schedule from DataRegistry
        """
        logger.info("Fleet Panel received schedule from DataRegistry")
        self.schedule = schedule
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Fleet Panel: Edit fleet configuration, compute cycle times, and allocate trucks to routes."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Fleet Configuration
        fleet_group = QGroupBox("Fleet Configuration")
        fleet_layout = QFormLayout()
        
        self.shift_hours = QDoubleSpinBox()
        self.shift_hours.setRange(8.0, 24.0)
        self.shift_hours.setValue(12.0)
        self.shift_hours.setSuffix(" hours")
        fleet_layout.addRow("Shift Hours:", self.shift_hours)
        
        fleet_group.setLayout(fleet_layout)
        layout.addWidget(fleet_group)
        
        # Truck Table
        self.truck_table = QTableWidget()
        self.truck_table.setColumnCount(6)
        self.truck_table.setHorizontalHeaderLabels([
            "ID", "Payload (t)", "Speed Loaded (km/h)", "Speed Empty (km/h)", "Availability", "Utilisation"
        ])
        self.truck_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("Trucks:"))
        layout.addWidget(self.truck_table)
        
        # Route Configuration
        route_group = QGroupBox("Route Configuration")
        route_layout = QFormLayout()
        
        self.route_source = QLineEdit()
        route_layout.addRow("Source:", self.route_source)
        
        self.route_destination = QLineEdit()
        route_layout.addRow("Destination:", self.route_destination)
        
        self.route_distance = QDoubleSpinBox()
        self.route_distance.setRange(0.0, 100.0)
        self.route_distance.setValue(5.0)
        self.route_distance.setSuffix(" km")
        route_layout.addRow("Distance:", self.route_distance)
        
        route_group.setLayout(route_layout)
        layout.addWidget(route_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cycle_time_btn = QPushButton("Compute Cycle Time")
        self.cycle_time_btn.clicked.connect(self._on_compute_cycle_time)
        button_layout.addWidget(self.cycle_time_btn)
        
        self.dispatch_btn = QPushButton("Run Dispatch")
        self.dispatch_btn.clicked.connect(self._on_run_dispatch)
        button_layout.addWidget(self.dispatch_btn)
        
        layout.addLayout(button_layout)
        
        # Results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_text)
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        return {
            "shift_hours": self.shift_hours.value()
        }
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def _on_compute_cycle_time(self):
        """Compute cycle time."""
        # Would get truck and route from UI
        params = {
            "truck": None,  # Would get from truck_table
            "route": {
                "id": "route_1",
                "source": self.route_source.text(),
                "destination": self.route_destination.text(),
                "distance_km": self.route_distance.value()
            },
            "parameters": {}
        }
        
        self.controller.compute_fleet_cycle_time(params, self._on_cycle_time_complete)
    
    def _on_cycle_time_complete(self, result: Dict[str, Any]):
        """Handle cycle time completion."""
        cycle_result = result.get("result")
        if cycle_result:
            self.results_text.append(
                f"Cycle Time: {cycle_result.truck_cycle_minutes:.1f} minutes\n"
                f"Tonnes per Hour: {cycle_result.tonnes_per_hour:.1f}"
            )
    
    def _on_run_dispatch(self):
        """Run dispatch."""
        params = {
            "fleet": self.fleet_config,
            "routes": [],  # Would get from UI
            "production_targets": {}  # Would get from schedule
        }
        
        self.controller.run_fleet_dispatch(params, self._on_dispatch_complete)
    
    def _on_dispatch_complete(self, result: Dict[str, Any]):
        """Handle dispatch completion."""
        allocation = result.get("result", {})
        self.results_text.append(f"Dispatch allocation complete: {len(allocation)} routes")
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        # After haulage evaluation, add button to send to Production Dashboard
        if not hasattr(self, 'prod_dashboard_btn'):
            self.prod_dashboard_btn = QPushButton("Send to Production Dashboard")
            self.prod_dashboard_btn.clicked.connect(self._on_send_to_production_dashboard)
            self.main_layout.addWidget(self.prod_dashboard_btn)
    
    def _on_send_to_production_dashboard(self):
        """Send haulage evaluation result to Production Dashboard."""
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
            
            # Pre-populate with haulage result
            # Would pass haulage eval result reference here
            self.controller.production_dashboard_dialog.show()
            self.controller.production_dashboard_dialog.raise_()
            self.controller.production_dashboard_dialog.activateWindow()
            
            self.show_info("Production Dashboard", "Sent haulage evaluation to Production Dashboard")
        
        except Exception as e:
            logger.error(f"Failed to open Production Dashboard: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Production Dashboard:\n{e}")

