"""
Planning Dashboard Panel (STEP 31)

Main dashboard for managing and comparing planning scenarios.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QLineEdit, QTextEdit,
    QTableWidget, QTableWidgetItem, QTabWidget, QHeaderView,
    QMessageBox, QCheckBox, QDoubleSpinBox, QSpinBox
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class PlanningDashboardPanel(BaseAnalysisPanel):
    """
    Panel for Planning Dashboard & Scenario Manager.
    """
    # PanelManager metadata
    PANEL_ID = "PlanningDashboardPanel"
    PANEL_NAME = "PlanningDashboard Panel"
    PANEL_CATEGORY = PanelCategory.PLANNING
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    PANEL_ID = "PlanningDashboardPanel"  # STEP 40
    task_name = "planning_run_scenario"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="planning_dashboard")
        self.current_scenario = None
        self.scenarios = []
        
        # Subscribe to data updates from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.irrResultsLoaded.connect(self._on_irr_results_loaded)
            self.registry.pitOptimizationResultsLoaded.connect(self._on_pit_optimization_results_loaded)
            self.registry.scheduleGenerated.connect(self._on_schedule_generated)
            self.registry.geometResultsLoaded.connect(self._on_geomet_results_loaded)
            
            # Load existing data if available
            existing_irr = self.registry.get_irr_results()
            if existing_irr:
                self._on_irr_results_loaded(existing_irr)
            
            existing_pit = self.registry.get_pit_optimization_results()
            if existing_pit:
                self._on_pit_optimization_results_loaded(existing_pit)
            
            existing_schedule = self.registry.get_schedule()
            if existing_schedule:
                self._on_schedule_generated(existing_schedule)
            
            existing_geomet = self.registry.get_geomet_results()
            if existing_geomet:
                self._on_geomet_results_loaded(existing_geomet)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        logger.info("Initialized Planning Dashboard Panel")
    


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
            "Planning Dashboard: Define, run, compare, and export planning scenarios across IRR, pit optimization, scheduling, geomet, GC, reconciliation, and risk."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Main split: Scenario list (left) and Editor/Results (right)
        main_split = QHBoxLayout()
        
        # Left: Scenario list
        left_panel = self._create_scenario_list_panel()
        main_split.addWidget(left_panel, 1)
        
        # Right: Editor and Results tabs
        right_panel = self._create_editor_results_panel()
        main_split.addWidget(right_panel, 2)
        
        layout.addLayout(main_split)
    
    def _on_irr_results_loaded(self, irr_results):
        """
        Automatically receive IRR results when they're loaded.
        
        Args:
            irr_results: IRR results from DataRegistry
        """
        logger.info("Planning Dashboard received IRR results from DataRegistry")
        self.irr_results = irr_results
        # Update scenario with IRR data if applicable
    
    def _on_pit_optimization_results_loaded(self, pit_results):
        """
        Automatically receive pit optimization results when they're loaded.
        
        Args:
            pit_results: Pit optimization results from DataRegistry
        """
        logger.info("Planning Dashboard received pit optimization results from DataRegistry")
        self.pit_results = pit_results
        # Update scenario with pit data if applicable
    
    def _on_schedule_generated(self, schedule):
        """
        Automatically receive schedule when it's generated.
        
        Args:
            schedule: Production schedule from DataRegistry
        """
        logger.info("Planning Dashboard received schedule from DataRegistry")
        self.schedule = schedule
        # Update scenario with schedule data if applicable
    
    def _on_geomet_results_loaded(self, geomet_results):
        """
        Automatically receive geomet results when they're loaded.
        
        Args:
            geomet_results: Geomet results from DataRegistry
        """
        logger.info("Planning Dashboard received geomet results from DataRegistry")
        self.geomet_results = geomet_results
        # Update scenario with geomet data if applicable
    
    def _create_scenario_list_panel(self) -> QWidget:
        """Create scenario list panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        layout.addWidget(QLabel("<b>Scenarios</b>"))
        
        # Buttons
        button_layout = QHBoxLayout()
        
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._on_new_scenario)
        button_layout.addWidget(new_btn)
        
        duplicate_btn = QPushButton("Duplicate")
        duplicate_btn.clicked.connect(self._on_duplicate_scenario)
        button_layout.addWidget(duplicate_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._on_delete_scenario)
        button_layout.addWidget(delete_btn)
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._on_export_scenario)
        button_layout.addWidget(export_btn)
        
        layout.addLayout(button_layout)
        
        # Scenario table
        self.scenario_table = QTableWidget()
        self.scenario_table.setColumnCount(5)
        self.scenario_table.setHorizontalHeaderLabels([
            "Name", "Version", "Status", "Tags", "Modified"
        ])
        self.scenario_table.horizontalHeader().setStretchLastSection(True)
        self.scenario_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.scenario_table.itemSelectionChanged.connect(self._on_scenario_selected)
        layout.addWidget(self.scenario_table)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_scenario_list)
        layout.addWidget(refresh_btn)
        
        return widget
    
    def _create_editor_results_panel(self) -> QWidget:
        """Create editor and results panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Tabs
        tabs = QTabWidget()
        
        # Editor tab
        editor_tab = self._create_scenario_editor_tab()
        tabs.addTab(editor_tab, "Scenario Editor")
        
        # Results tab
        results_tab = self._create_results_tab()
        tabs.addTab(results_tab, "Single Scenario Results")
        
        # Comparison tab
        comparison_tab = self._create_comparison_tab()
        tabs.addTab(comparison_tab, "Scenario Comparison")
        
        layout.addWidget(tabs)
        
        return widget
    
    def _create_scenario_editor_tab(self) -> QWidget:
        """Create scenario editor tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Basic Info
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout()
        
        self.scenario_name = QLineEdit()
        basic_layout.addRow("Name:", self.scenario_name)
        
        self.scenario_version = QLineEdit()
        self.scenario_version.setText("v1")
        basic_layout.addRow("Version:", self.scenario_version)
        
        self.scenario_description = QTextEdit()
        self.scenario_description.setMaximumHeight(60)
        basic_layout.addRow("Description:", self.scenario_description)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # Block Model & Value
        model_group = QGroupBox("Block Model & Value")
        model_layout = QFormLayout()
        
        self.model_name = QComboBox()
        self.model_name.addItems(["default"])
        model_layout.addRow("Model:", self.model_name)
        
        self.value_mode = QComboBox()
        self.value_mode.addItems(["base", "geomet"])
        self.value_mode.currentTextChanged.connect(self._on_value_mode_changed)
        model_layout.addRow("Value Mode:", self.value_mode)
        
        self.value_field = QComboBox()
        self.value_field.addItems(["block_value", "geomet_value"])
        model_layout.addRow("Value Field:", self.value_field)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Pit & Shells
        pit_group = QGroupBox("Pit & Shells")
        pit_layout = QFormLayout()
        
        self.include_pit = QCheckBox("Include Pit Optimization")
        pit_layout.addRow(self.include_pit)
        
        pit_group.setLayout(pit_layout)
        layout.addWidget(pit_group)
        
        # Scheduling
        schedule_group = QGroupBox("Scheduling")
        schedule_layout = QFormLayout()
        
        self.include_strategic = QCheckBox("Include Strategic Schedule")
        self.include_strategic.setChecked(True)
        schedule_layout.addRow(self.include_strategic)
        
        self.include_tactical = QCheckBox("Include Tactical Schedule")
        schedule_layout.addRow(self.include_tactical)
        
        self.include_short_term = QCheckBox("Include Short-Term Schedule")
        schedule_layout.addRow(self.include_short_term)
        
        self.mine_capacity = QDoubleSpinBox()
        self.mine_capacity.setRange(0.0, 100000000.0)
        self.mine_capacity.setValue(10000000.0)
        self.mine_capacity.setSuffix(" tpy")
        schedule_layout.addRow("Mining Capacity:", self.mine_capacity)
        
        self.plant_capacity = QDoubleSpinBox()
        self.plant_capacity.setRange(0.0, 100000000.0)
        self.plant_capacity.setValue(8000000.0)
        self.plant_capacity.setSuffix(" tpy")
        schedule_layout.addRow("Plant Capacity:", self.plant_capacity)
        
        schedule_group.setLayout(schedule_layout)
        layout.addWidget(schedule_group)
        
        # Geomet
        geomet_group = QGroupBox("Geometallurgy")
        geomet_layout = QFormLayout()
        
        self.include_geomet = QCheckBox("Include Geomet Attributes")
        geomet_layout.addRow(self.include_geomet)
        
        geomet_group.setLayout(geomet_layout)
        layout.addWidget(geomet_group)
        
        # GC & Reconciliation
        gc_group = QGroupBox("GC & Reconciliation")
        gc_layout = QFormLayout()
        
        self.include_gc = QCheckBox("Include GC Model & Reconciliation")
        gc_layout.addRow(self.include_gc)
        
        gc_group.setLayout(gc_layout)
        layout.addWidget(gc_group)
        
        # Risk
        risk_group = QGroupBox("Risk & Uncertainty")
        risk_layout = QFormLayout()
        
        self.include_risk = QCheckBox("Include Risk/Uncertainty Analysis")
        risk_layout.addRow(self.include_risk)
        
        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)
        
        # NPVS (STEP 32)
        npvs_group = QGroupBox("NPVS Optimisation")
        npvs_layout = QFormLayout()
        
        self.include_npvs = QCheckBox("Include NPVS Optimisation")
        npvs_layout.addRow(self.include_npvs)
        
        npvs_group.setLayout(npvs_layout)
        layout.addWidget(npvs_group)
        
        # Run button
        run_btn = QPushButton("Run Scenario")
        run_btn.clicked.connect(self._on_run_scenario)
        layout.addWidget(run_btn)
        
        layout.addStretch()
        return widget
    
    def _create_results_tab(self) -> QWidget:
        """Create results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        return widget
    
    def _create_comparison_tab(self) -> QWidget:
        """Create comparison tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Comparison table
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(8)
        self.comparison_table.setHorizontalHeaderLabels([
            "Name", "Version", "NPV", "IRR", "Payback", "LOM", "Peak Prod", "Status"
        ])
        self.comparison_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("Scenario Comparison:"))
        layout.addWidget(self.comparison_table)
        
        # Compare button
        compare_btn = QPushButton("Compare Selected Scenarios")
        compare_btn.clicked.connect(self._on_compare_scenarios)
        layout.addWidget(compare_btn)
        
        return widget
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        if not self.scenario_name.text():
            self.show_warning("Missing Name", "Please enter a scenario name.")
            return False
        return True
    
    def _on_new_scenario(self):
        """Create new scenario."""
        self.scenario_name.clear()
        self.scenario_version.setText("v1")
        self.scenario_description.clear()
        self.current_scenario = None
    
    def _on_duplicate_scenario(self):
        """Duplicate selected scenario."""
        if not self.current_scenario:
            self.show_warning("No Selection", "Please select a scenario to duplicate.")
            return
        
        # Would duplicate scenario with new version
        self.show_info("Duplicate", "Duplication functionality would be implemented here.")
    
    def _on_delete_scenario(self):
        """Delete selected scenario."""
        if not self.current_scenario:
            self.show_warning("No Selection", "Please select a scenario to delete.")
            return
        
        reply = QMessageBox.question(
            self, "Delete Scenario",
            f"Delete scenario {self.current_scenario.id.name} v{self.current_scenario.id.version}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.controller.scenario_store.delete(
                    self.current_scenario.id.name,
                    self.current_scenario.id.version
                )
                self._refresh_scenario_list()
                self.current_scenario = None
            except Exception as e:
                self.show_error("Delete Failed", f"Failed to delete scenario:\n{e}")
    
    def _on_export_scenario(self):
        """Export scenario."""
        if not self.current_scenario:
            self.show_warning("No Selection", "Please select a scenario to export.")
            return
        
        self.show_info("Export", "Export functionality would be implemented here.")
    
    def _on_scenario_selected(self):
        """Handle scenario selection."""
        selected = self.scenario_table.selectedItems()
        if not selected:
            return
        
        row = selected[0].row()
        name_item = self.scenario_table.item(row, 0)
        version_item = self.scenario_table.item(row, 1)
        
        if name_item and version_item:
            name = name_item.text()
            version = version_item.text()
            
            scenario = self.controller.scenario_store.get(name, version)
            if scenario:
                self.current_scenario = scenario
                self._load_scenario_to_editor(scenario)
    
    def _load_scenario_to_editor(self, scenario):
        """Load scenario into editor."""
        self.scenario_name.setText(scenario.id.name)
        self.scenario_version.setText(scenario.id.version)
        self.scenario_description.setText(scenario.description)
        self.model_name.setCurrentText(scenario.inputs.model_name)
        self.value_mode.setCurrentText(scenario.inputs.value_mode)
        self.value_field.setCurrentText(scenario.inputs.value_field)
        self.include_pit.setChecked(scenario.inputs.pit_config is not None)
        self.include_strategic.setChecked(scenario.inputs.schedule_config is not None)
        self.include_geomet.setChecked(scenario.inputs.geomet_config is not None)
        self.include_gc.setChecked(scenario.inputs.gc_config is not None)
        self.include_risk.setChecked(scenario.inputs.risk_config is not None)
        self.include_npvs.setChecked(scenario.inputs.npvs_config is not None if hasattr(scenario.inputs, 'npvs_config') else False)
    
    def _on_value_mode_changed(self, mode: str):
        """Handle value mode change."""
        if mode == "geomet":
            self.include_geomet.setChecked(True)
    
    def _on_run_scenario(self):
        """Run scenario."""
        if not self.validate_inputs():
            return
        
        # Build scenario from editor
        scenario = self._build_scenario_from_editor()
        
        # Save scenario
        self.controller.create_scenario(scenario)
        
        # Run scenario
        self.controller.run_scenario(
            scenario.id.name,
            scenario.id.version,
            self._on_scenario_complete
        )
    
    def _build_scenario_from_editor(self):
        """Build scenario from editor inputs."""
        from ..planning.scenario_definition import PlanningScenario, ScenarioID, ScenarioInputs
        
        scenario_id = ScenarioID(
            name=self.scenario_name.text(),
            version=self.scenario_version.text()
        )
        
        inputs = ScenarioInputs(
            model_name=self.model_name.currentText(),
            value_mode=self.value_mode.currentText(),
            value_field=self.value_field.currentText(),
            pit_config={"enabled": self.include_pit.isChecked()} if self.include_pit.isChecked() else None,
            schedule_config={
                "strategic": {"enabled": self.include_strategic.isChecked()},
                "tactical": {"enabled": self.include_tactical.isChecked()},
                "short_term": {"enabled": self.include_short_term.isChecked()},
                "mine_capacity_tpy": self.mine_capacity.value(),
                "plant_capacity_tpy": self.plant_capacity.value(),
            } if (self.include_strategic.isChecked() or self.include_tactical.isChecked() or self.include_short_term.isChecked()) else None,
            geomet_config={"enabled": self.include_geomet.isChecked()} if self.include_geomet.isChecked() else None,
            gc_config={"enabled": self.include_gc.isChecked()} if self.include_gc.isChecked() else None,
            risk_config={"enabled": self.include_risk.isChecked()} if self.include_risk.isChecked() else None,
            npvs_config={"enabled": self.include_npvs.isChecked()} if self.include_npvs.isChecked() else None,
        )
        
        scenario = PlanningScenario(
            id=scenario_id,
            description=self.scenario_description.toPlainText(),
            tags=[],
            inputs=inputs,
            status="new"
        )
        
        return scenario
    
    def _on_scenario_complete(self, result: Dict[str, Any]):
        """Handle scenario completion."""
        scenario_result = result.get("result")
        if scenario_result:
            self.results_text.append(f"Scenario {scenario_result.id.name} completed.\n")
            self._refresh_scenario_list()
    
    def _on_compare_scenarios(self):
        """Compare selected scenarios."""
        selected = self.scenario_table.selectedItems()
        if not selected or len(selected) < 2:
            self.show_warning("Selection", "Please select at least 2 scenarios to compare.")
            return
        
        # Get selected scenario IDs
        scenario_ids = []
        rows = set(item.row() for item in selected if item.column() == 0)
        
        for row in rows:
            name_item = self.scenario_table.item(row, 0)
            version_item = self.scenario_table.item(row, 1)
            if name_item and version_item:
                from ..planning.scenario_definition import ScenarioID
                scenario_ids.append(ScenarioID(name=name_item.text(), version=version_item.text()))
        
        if len(scenario_ids) < 2:
            self.show_warning("Selection", "Please select at least 2 scenarios to compare.")
            return
        
        self.controller.compare_scenarios(scenario_ids, self._on_comparison_complete)
    
    def _on_comparison_complete(self, result: Dict[str, Any]):
        """Handle comparison completion."""
        comparison = result.get("result", {})
        
        # Populate comparison table
        scenarios = comparison.get("scenarios", [])
        self.comparison_table.setRowCount(len(scenarios))
        
        for row, scenario_metrics in enumerate(scenarios):
            self.comparison_table.setItem(row, 0, QTableWidgetItem(scenario_metrics.get("name", "")))
            self.comparison_table.setItem(row, 1, QTableWidgetItem(scenario_metrics.get("version", "")))
            self.comparison_table.setItem(row, 2, QTableWidgetItem(f"{scenario_metrics.get('npv', 0):,.0f}" if scenario_metrics.get('npv') else "N/A"))
            self.comparison_table.setItem(row, 3, QTableWidgetItem(f"{scenario_metrics.get('irr', 0):.2%}" if scenario_metrics.get('irr') else "N/A"))
            self.comparison_table.setItem(row, 4, QTableWidgetItem(str(scenario_metrics.get('payback_period', 'N/A'))))
            self.comparison_table.setItem(row, 5, QTableWidgetItem(str(scenario_metrics.get('lom_years', 'N/A'))))
            self.comparison_table.setItem(row, 6, QTableWidgetItem(f"{scenario_metrics.get('peak_annual_production', 0):,.0f}" if scenario_metrics.get('peak_annual_production') else "N/A"))
            self.comparison_table.setItem(row, 7, QTableWidgetItem("Completed"))
    
    def _refresh_scenario_list(self):
        """Refresh scenario list."""
        if not self.controller:
            return
        
        scenarios = self.controller.list_scenarios()
        self.scenarios = scenarios
        
        self.scenario_table.setRowCount(len(scenarios))
        
        for row, scenario in enumerate(scenarios):
            self.scenario_table.setItem(row, 0, QTableWidgetItem(scenario.id.name))
            self.scenario_table.setItem(row, 1, QTableWidgetItem(scenario.id.version))
            self.scenario_table.setItem(row, 2, QTableWidgetItem(scenario.status))
            self.scenario_table.setItem(row, 3, QTableWidgetItem(", ".join(scenario.tags)))
            self.scenario_table.setItem(row, 4, QTableWidgetItem(scenario.modified_at.strftime("%Y-%m-%d %H:%M") if isinstance(scenario.modified_at, datetime) else str(scenario.modified_at)))
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass

