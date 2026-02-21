"""
Experiment Configuration Panel

Build and edit experiment definitions for research mode.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
import numpy as np

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QLineEdit, QTableWidget, QTableWidgetItem, QCheckBox,
    QMessageBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel
from ..research.experiment_definitions import (
    ExperimentDefinition, ExperimentParameter, ScenarioGrid
)

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class ExperimentConfigPanel(BaseAnalysisPanel):
    """Panel for configuring experiments."""
    
    task_name = "experiment_config"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="experiment_config")
        self.setWindowTitle("Experiment Configuration")
        
        self._block_model = None
        
        # Subscribe to block model from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            
            # Load existing block model if available
            existing_block_model = self.registry.get_block_model()
            if existing_block_model:
                self._on_block_model_loaded(existing_block_model)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self.current_definition: Optional[ExperimentDefinition] = None
        logger.info("Initialized Experiment Config panel")
    


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
        logger.info("Experiment Config Panel received block model from DataRegistry")
        self._block_model = block_model
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)

    @property
    def block_model(self):
        """Block model associated with this experiment (read/write)."""
        return self._block_model

    @block_model.setter
    def block_model(self, value):
        self._block_model = value
    
    def setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Basic info
        info_group = QGroupBox("Experiment Information")
        info_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        info_layout.addRow("Name:", self.name_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        info_layout.addRow("Description:", self.description_edit)
        
        self.domain_combo = QComboBox()
        self.domain_combo.addItems(["geostats", "planning", "uncertainty", "geotech"])
        info_layout.addRow("Domain:", self.domain_combo)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(4)
        self.params_table.setHorizontalHeaderLabels(["Name", "Type", "Values", "Notes"])
        self.params_table.horizontalHeader().setStretchLastSection(True)
        params_layout.addWidget(self.params_table)
        
        param_actions = QHBoxLayout()
        add_param_btn = QPushButton("Add Parameter")
        add_param_btn.clicked.connect(self._add_parameter)
        param_actions.addWidget(add_param_btn)
        
        remove_param_btn = QPushButton("Remove Parameter")
        remove_param_btn.clicked.connect(self._remove_parameter)
        param_actions.addWidget(remove_param_btn)
        
        param_actions.addStretch()
        params_layout.addLayout(param_actions)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Pipelines
        pipelines_group = QGroupBox("Pipelines")
        pipelines_layout = QVBoxLayout()
        
        self.pipelines_list = QListWidget()
        pipelines_layout.addWidget(self.pipelines_list)
        
        pipeline_actions = QHBoxLayout()
        add_pipeline_btn = QPushButton("Add Pipeline")
        add_pipeline_btn.clicked.connect(self._add_pipeline)
        pipeline_actions.addWidget(add_pipeline_btn)
        
        remove_pipeline_btn = QPushButton("Remove Pipeline")
        remove_pipeline_btn.clicked.connect(self._remove_pipeline)
        pipeline_actions.addWidget(remove_pipeline_btn)
        
        pipeline_actions.addStretch()
        pipelines_layout.addLayout(pipeline_actions)
        
        pipelines_group.setLayout(pipelines_layout)
        layout.addWidget(pipelines_group)
        
        # Metrics
        metrics_group = QGroupBox("Metrics to Record")
        metrics_layout = QVBoxLayout()
        
        self.metrics_list = QListWidget()
        self.metrics_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        metrics_layout.addWidget(self.metrics_list)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Actions
        actions_layout = QHBoxLayout()
        
        generate_btn = QPushButton("Generate Scenario Grid")
        generate_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")
        generate_btn.clicked.connect(self._generate_grid)
        actions_layout.addWidget(generate_btn)
        
        run_btn = QPushButton("Run Experiment")
        run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        run_btn.clicked.connect(self._run_experiment)
        actions_layout.addWidget(run_btn)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
        
        # Initialize metrics list
        self._populate_metrics_list()
    
    def _populate_metrics_list(self):
        """Populate available metrics list."""
        metrics = [
            "rmse_cv", "mae_cv", "smoothing_index", "nugget_to_sill_ratio",
            "gt_loss", "npv_mean", "npv_std", "npv_p10", "npv_p50", "npv_p90",
            "irr_mean", "irr_std", "irr_p10", "irr_p50", "irr_p90",
            "risk_adjusted_npv", "schedule_exposure", "risk_percentile"
        ]
        
        self.metrics_list.clear()
        for metric in metrics:
            self.metrics_list.addItem(metric)
    
    def _add_parameter(self):
        """Add a new parameter."""
        # Simple dialog would go here - for now, add empty row
        row = self.params_table.rowCount()
        self.params_table.insertRow(row)
        self.params_table.setItem(row, 0, QTableWidgetItem(""))
        self.params_table.setItem(row, 1, QTableWidgetItem("categorical"))
        self.params_table.setItem(row, 2, QTableWidgetItem(""))
        self.params_table.setItem(row, 3, QTableWidgetItem(""))
    
    def _remove_parameter(self):
        """Remove selected parameter."""
        current_row = self.params_table.currentRow()
        if current_row >= 0:
            self.params_table.removeRow(current_row)
    
    def _add_pipeline(self):
        """Add a pipeline."""
        pipelines = ["ok", "uk", "cok", "ik", "loocv", "pit_optimization", "npv", "irr",
                    "sgsim", "ik_sgsim", "cosgsim", "economic_uncertainty"]
        
        pipeline, ok = QMessageBox.getItem(
            self, "Add Pipeline", "Select pipeline:", pipelines, 0, False
        )
        
        if ok and pipeline:
            self.pipelines_list.addItem(pipeline)
    
    def _remove_pipeline(self):
        """Remove selected pipeline."""
        current_item = self.pipelines_list.currentItem()
        if current_item:
            self.pipelines_list.takeItem(self.pipelines_list.row(current_item))
    
    def _generate_grid(self):
        """Generate scenario grid from definition."""
        definition = self._build_definition()
        if not definition:
            return
        
        grid = ScenarioGrid.from_definition(definition)
        
        QMessageBox.information(
            self,
            "Scenario Grid Generated",
            f"Generated {len(grid.instances)} experiment instances"
        )
    
    def _run_experiment(self):
        """Run the experiment."""
        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not connected")
            return
        
        definition = self._build_definition()
        if not definition:
            return
        
        grid = ScenarioGrid.from_definition(definition)
        
        # Convert to dict for controller
        grid_dict = {
            'definition': {
                'id': grid.definition.id,
                'name': grid.definition.name,
                'description': grid.definition.description,
                'domain': grid.definition.domain,
                'parameters': [
                    {
                        'name': p.name,
                        'values': p.values,
                        'type': p.type,
                        'notes': p.notes
                    }
                    for p in grid.definition.parameters
                ],
                'base_config': grid.definition.base_config,
                'metrics': grid.definition.metrics,
                'pipelines': grid.definition.pipelines
            },
            'instances': [
                {
                    'definition_id': inst.definition_id,
                    'index': inst.index,
                    'parameter_values': inst.parameter_values,
                    'seed': inst.seed,
                    'metadata': inst.metadata
                }
                for inst in grid.instances
            ]
        }
        
        self.controller.run_research_grid(grid_dict, self._on_experiment_results)
    
    def _build_definition(self) -> Optional[ExperimentDefinition]:
        """Build experiment definition from UI."""
        name = self.name_edit.text()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter experiment name")
            return None
        
        # Build parameters
        parameters = []
        for row in range(self.params_table.rowCount()):
            name_item = self.params_table.item(row, 0)
            type_item = self.params_table.item(row, 1)
            values_item = self.params_table.item(row, 2)
            
            if name_item and values_item:
                param_name = name_item.text()
                param_type = type_item.text() if type_item else "categorical"
                values_str = values_item.text()
                
                # Parse values
                if param_type == "numeric":
                    # Parse numeric range: min,max,step or list
                    try:
                        if ',' in values_str:
                            parts = [p.strip() for p in values_str.split(',')]
                            if len(parts) == 3:
                                # Range
                                min_val, max_val, step = map(float, parts)
                                values = list(np.arange(min_val, max_val + step, step))
                            else:
                                # List
                                values = [float(p) for p in parts]
                        else:
                            values = [float(values_str)]
                    except ValueError:
                        QMessageBox.warning(self, "Error", f"Invalid numeric values for parameter {param_name}")
                        return None
                else:
                    # Categorical: comma-separated
                    values = [v.strip() for v in values_str.split(',')]
                
                param = ExperimentParameter(
                    name=param_name,
                    values=values,
                    type=param_type
                )
                parameters.append(param)
        
        # Get pipelines
        pipelines = []
        for i in range(self.pipelines_list.count()):
            pipelines.append(self.pipelines_list.item(i).text())
        
        # Get metrics
        metrics = []
        for item in self.metrics_list.selectedItems():
            metrics.append(item.text())
        
        definition = ExperimentDefinition(
            name=name,
            description=self.description_edit.toPlainText(),
            domain=self.domain_combo.currentText(),
            parameters=parameters,
            base_config={},  # Would be populated from base config UI
            metrics=metrics,
            pipelines=pipelines
        )
        
        self.current_definition = definition
        return definition
    
    def load_definition(self, definition_dict: Dict[str, Any]):
        """Load an experiment definition."""
        # Populate UI from definition dict
        self.name_edit.setText(definition_dict.get('name', ''))
        self.description_edit.setPlainText(definition_dict.get('description', ''))
        
        domain = definition_dict.get('domain', 'geostats')
        index = self.domain_combo.findText(domain)
        if index >= 0:
            self.domain_combo.setCurrentIndex(index)
        
        # Load parameters
        self.params_table.setRowCount(0)
        for param_dict in definition_dict.get('parameters', []):
            row = self.params_table.rowCount()
            self.params_table.insertRow(row)
            self.params_table.setItem(row, 0, QTableWidgetItem(param_dict.get('name', '')))
            self.params_table.setItem(row, 1, QTableWidgetItem(param_dict.get('type', 'categorical')))
            values_str = ','.join([str(v) for v in param_dict.get('values', [])])
            self.params_table.setItem(row, 2, QTableWidgetItem(values_str))
            self.params_table.setItem(row, 3, QTableWidgetItem(param_dict.get('notes', '')))
        
        # Load pipelines
        self.pipelines_list.clear()
        for pipeline in definition_dict.get('pipelines', []):
            self.pipelines_list.addItem(pipeline)
        
        # Load metrics
        for i in range(self.metrics_list.count()):
            item = self.metrics_list.item(i)
            if item.text() in definition_dict.get('metrics', []):
                item.setSelected(True)
    
    def _on_experiment_results(self, payload: Dict[str, Any]):
        """Handle experiment results."""
        if payload.get('error'):
            self.show_error("Experiment Error", payload['error'])
            return
        
        # Forward to results panel
        try:
            from .experiment_results_panel import ExperimentResultsPanel
            
            results_panel = ExperimentResultsPanel(parent=self.parent())
            results_panel.bind_controller(self.controller)
            results_panel.load_results(payload)
            results_panel.show()
        except Exception as e:
            logger.error(f"Failed to open results: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to open results:\n{e}")
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Gather parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass

