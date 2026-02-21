"""
Uncertainty Analysis Panel

Comprehensive GUI for mine planning uncertainty analysis including:
- Monte Carlo simulation with dynamic re-optimization
- Bootstrap confidence intervals
- Latin Hypercube Sampling
- Probabilistic pit shells
- Interactive risk dashboard

Integrates with existing BlockModelViewer architecture.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QCheckBox, QRadioButton, QButtonGroup,
    QScrollArea, QLineEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

try:
    # Matplotlib backend is set in main.py
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


class UncertaintyAnalysisPanel(BaseAnalysisPanel):
    """
    Panel for comprehensive uncertainty analysis in mine planning.
    """
    
    task_name = "uncertainty"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="uncertainty")
        
        # Use _block_model (private backing field) - block_model is a read-only @property in BaseAnalysisPanel
        self._block_model: Optional[pd.DataFrame] = None
        self.results: Optional[Dict] = None
        self.current_analysis_type: Optional[str] = None  # 'monte_carlo', 'bootstrap', 'lhs', 'prob_shells'
        self.pit_results = None
        self.kriging_results = None
        self.sgsim_results = None
        self.loaded_schedule = None
        
        # Subscribe to data from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.blockModelClassified.connect(self._on_block_model_classified)
            self.registry.pitOptimizationResultsLoaded.connect(self._on_pit_optimization_results_loaded)
            self.registry.krigingResultsLoaded.connect(self._on_kriging_results_loaded)
            self.registry.sgsimResultsLoaded.connect(self._on_sgsim_results_loaded)
            self.registry.scheduleGenerated.connect(self._on_schedule_generated)
            
            # Load existing data if available
            self._refresh_available_block_models()
            
            existing_pit = self.registry.get_pit_optimization_results()
            if existing_pit:
                self._on_pit_optimization_results_loaded(existing_pit)
            
            existing_kriging = self.registry.get_kriging_results()
            if existing_kriging:
                self._on_kriging_results_loaded(existing_kriging)
            
            existing_sgsim = self.registry.get_sgsim_results()
            if existing_sgsim:
                self._on_sgsim_results_loaded(existing_sgsim)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
    
    def _refresh_available_block_models(self):
        """Refresh the list of available block models from all sources."""
        if not self.registry:
            return
        
        try:
            # Check regular block model
            block_model = self.registry.get_block_model()
            if block_model is not None:
                self._on_block_model_loaded(block_model)
            
            # Check classified block model
            try:
                classified_model = self.registry.get_classified_block_model()
                if classified_model is not None:
                    self._on_block_model_classified(classified_model)
            except Exception as e:
                logger.error(f"Failed to load classified block model: {e}", exc_info=True)
            
            # Also check renderer layers for block models that might be displayed
            self._check_renderer_layers_for_block_models()
            
        except Exception as e:
            logger.warning(f"Failed to refresh available block models: {e}")
    
    def _check_renderer_layers_for_block_models(self):
        """Check renderer layers for block models that might be displayed."""
        try:
            # Try to get renderer from parent/main window
            parent = self.parent()
            while parent:
                if hasattr(parent, 'viewer_widget') and hasattr(parent.viewer_widget, 'renderer'):
                    renderer = parent.viewer_widget.renderer
                    if hasattr(renderer, 'active_layers') and renderer.active_layers:
                        # Look for block model layers (SGSIM, kriging, etc.)
                        for layer_name, layer_info in renderer.active_layers.items():
                            layer_type = layer_info.get('type', '')
                            layer_data = layer_info.get('data', None)
                            
                            # Check if it's a block model layer
                            if layer_type in ('blocks', 'volume') and layer_data is not None:
                                # If we don't have a block model yet, try to use this one
                                if self.block_model is None:
                                    if hasattr(parent, '_extract_block_model_from_grid'):
                                        block_model = parent._extract_block_model_from_grid(layer_data, layer_name)
                                        if block_model is not None:
                                            self._on_block_model_generated(block_model)
                                            logger.info(f"Found block model in renderer layer: {layer_name}")
                    break
                parent = parent.parent() if parent else None
        except Exception as e:
            logger.debug(f"Could not check renderer layers: {e}")
    
    def _on_block_model_classified(self, block_model):
        """Handle block model classification changes."""
        self._on_block_model_loaded(block_model)
        
        try:
            from ..uncertainty_engine import UncertaintyDashboard
            self.dashboard = UncertaintyDashboard()
        except ImportError:
            self.dashboard = None
        
        self._setup_ui()
        
        logger.info("Initialized Uncertainty Analysis panel")
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        logger.info("Uncertainty Analysis Panel received block model from DataRegistry")
        # Convert to DataFrame if needed
        if hasattr(block_model, 'to_dataframe'):
            self._block_model = block_model.to_dataframe()  # Use private backing field (property contract)
        elif isinstance(block_model, pd.DataFrame):
            self._block_model = block_model  # Use private backing field (property contract)
        else:
            logger.warning(f"Unexpected block model type: {type(block_model)}")
    
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
        logger.info("Uncertainty Analysis Panel received pit optimization results from DataRegistry")
        self.pit_results = pit_results
    
    def _on_kriging_results_loaded(self, kriging_results):
        """
        Automatically receive kriging results when they're loaded.
        
        Args:
            kriging_results: Kriging results from DataRegistry
        """
        logger.info("Uncertainty Analysis Panel received kriging results from DataRegistry")
        self.kriging_results = kriging_results
    
    def _on_sgsim_results_loaded(self, sgsim_results):
        """
        Automatically receive SGSIM results when they're loaded.
        
        Args:
            sgsim_results: SGSIM results from DataRegistry
        """
        logger.info("Uncertainty Analysis Panel received SGSIM results from DataRegistry")
        self.sgsim_results = sgsim_results
    
    def _on_schedule_generated(self, schedule):
        """
        Automatically receive schedule when it's generated.
        
        Args:
            schedule: Schedule from DataRegistry
        """
        logger.info("Uncertainty Analysis Panel received schedule from DataRegistry")
        self.loaded_schedule = schedule
    
    def _setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("<b>Uncertainty Analysis & Risk Dashboard</b>")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_monte_carlo_tab(), "Monte Carlo")
        tabs.addTab(self._create_bootstrap_tab(), "Bootstrap CI")
        tabs.addTab(self._create_lhs_tab(), "Latin Hypercube")
        tabs.addTab(self._create_prob_shells_tab(), "Probabilistic Shells")
        tabs.addTab(self._create_dashboard_tab(), "Risk Dashboard")
        
        layout.addWidget(tabs)
        
        # Progress section
        progress_group = self._create_progress_section()
        layout.addWidget(progress_group)
    
    def _create_monte_carlo_tab(self) -> QWidget:
        """Create Monte Carlo simulation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Scroll area for long form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Basic settings
        basic_group = QGroupBox("Simulation Settings")
        basic_layout = QFormLayout()
        
        self.mc_n_simulations_spin = QSpinBox()
        self.mc_n_simulations_spin.setRange(10, 10000)
        self.mc_n_simulations_spin.setValue(1000)
        self.mc_n_simulations_spin.setSingleStep(100)
        basic_layout.addRow("Number of Simulations:", self.mc_n_simulations_spin)
        
        self.mc_mode_combo = QComboBox()
        self.mc_mode_combo.addItems(["Dynamic Re-optimize", "Fixed Schedule"])
        basic_layout.addRow("Simulation Mode:", self.mc_mode_combo)
        
        self.mc_seed_spin = QSpinBox()
        self.mc_seed_spin.setRange(0, 999999)
        self.mc_seed_spin.setValue(42)
        self.mc_seed_spin.setSpecialValueText("Random")
        basic_layout.addRow("Random Seed:", self.mc_seed_spin)
        
        self.mc_parallel_check = QCheckBox("Use Parallel Processing")
        self.mc_parallel_check.setChecked(True)
        basic_layout.addRow("", self.mc_parallel_check)
        
        basic_group.setLayout(basic_layout)
        scroll_layout.addWidget(basic_group)
        
        # Parameter distributions
        param_group = QGroupBox("Parameter Uncertainty")
        param_layout = QVBoxLayout()
        
        param_layout.addWidget(QLabel("<i>Define uncertain parameters:</i>"))
        
        # Price uncertainty
        price_group = self._create_parameter_group("Price ($/t)", "price")
        param_layout.addWidget(price_group)
        
        # Cost uncertainty
        cost_group = self._create_parameter_group("Mining Cost ($/t)", "mining_cost")
        param_layout.addWidget(cost_group)
        
        # Recovery uncertainty
        recovery_group = self._create_parameter_group("Recovery (%)", "recovery")
        param_layout.addWidget(recovery_group)
        
        # Grade uncertainty
        grade_group = QGroupBox("Grade Uncertainty")
        grade_layout = QFormLayout()
        
        self.mc_use_realizations = QCheckBox("Use Pre-generated Grade Realizations")
        grade_layout.addRow("", self.mc_use_realizations)
        
        self.mc_n_realizations_spin = QSpinBox()
        self.mc_n_realizations_spin.setRange(10, 1000)
        self.mc_n_realizations_spin.setValue(100)
        self.mc_n_realizations_spin.setEnabled(False)
        self.mc_use_realizations.toggled.connect(self.mc_n_realizations_spin.setEnabled)
        grade_layout.addRow("Number of Realizations:", self.mc_n_realizations_spin)
        
        grade_group.setLayout(grade_layout)
        param_layout.addWidget(grade_group)
        
        param_group.setLayout(param_layout)
        scroll_layout.addWidget(param_group)
        
        # Output settings
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout()
        
        self.mc_track_annual_check = QCheckBox("Track Annual Metrics")
        self.mc_track_annual_check.setChecked(True)
        output_layout.addRow("", self.mc_track_annual_check)
        
        self.mc_track_blocks_check = QCheckBox("Track Block-Level Risk")
        self.mc_track_blocks_check.setChecked(True)
        output_layout.addRow("", self.mc_track_blocks_check)
        
        output_group.setLayout(output_layout)
        scroll_layout.addWidget(output_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Run button
        run_btn = QPushButton("Run Monte Carlo Simulation")
        run_btn.clicked.connect(lambda: self._start_analysis('monte_carlo'))
        layout.addWidget(run_btn)
        
        return widget
    
    def _create_parameter_group(self, label: str, param_name: str) -> QGroupBox:
        """Create a parameter uncertainty input group."""
        group = QGroupBox(label)
        layout = QFormLayout()
        
        # Distribution type
        dist_combo = QComboBox()
        dist_combo.addItems(["Normal", "Triangular", "Uniform", "Log-Normal", "Deterministic"])
        dist_combo.setObjectName(f"{param_name}_dist")
        layout.addRow("Distribution:", dist_combo)
        
        # Base value
        base_spin = QDoubleSpinBox()
        base_spin.setRange(-1e9, 1e9)
        base_spin.setValue(100.0)
        base_spin.setDecimals(2)
        base_spin.setObjectName(f"{param_name}_base")
        layout.addRow("Base Value:", base_spin)
        
        # Std dev (for Normal)
        std_spin = QDoubleSpinBox()
        std_spin.setRange(0, 1e9)
        std_spin.setValue(10.0)
        std_spin.setDecimals(2)
        std_spin.setObjectName(f"{param_name}_std")
        layout.addRow("Std Dev:", std_spin)
        
        # Min/Max (for Triangular, Uniform)
        min_spin = QDoubleSpinBox()
        min_spin.setRange(-1e9, 1e9)
        min_spin.setValue(80.0)
        min_spin.setDecimals(2)
        min_spin.setObjectName(f"{param_name}_min")
        layout.addRow("Min Value:", min_spin)
        
        max_spin = QDoubleSpinBox()
        max_spin.setRange(-1e9, 1e9)
        max_spin.setValue(120.0)
        max_spin.setDecimals(2)
        max_spin.setObjectName(f"{param_name}_max")
        layout.addRow("Max Value:", max_spin)
        
        # Enable checkbox
        enable_check = QCheckBox("Include in Analysis")
        enable_check.setObjectName(f"{param_name}_enable")
        layout.addRow("", enable_check)
        
        group.setLayout(layout)
        return group
    
    def _create_bootstrap_tab(self) -> QWidget:
        """Create Bootstrap confidence intervals tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Settings
        settings_group = QGroupBox("Bootstrap Settings")
        settings_layout = QFormLayout()
        
        self.boot_iterations_spin = QSpinBox()
        self.boot_iterations_spin.setRange(100, 10000)
        self.boot_iterations_spin.setValue(1000)
        self.boot_iterations_spin.setSingleStep(100)
        settings_layout.addRow("Iterations:", self.boot_iterations_spin)
        
        self.boot_confidence_spin = QDoubleSpinBox()
        self.boot_confidence_spin.setRange(0.80, 0.99)
        self.boot_confidence_spin.setValue(0.95)
        self.boot_confidence_spin.setSingleStep(0.01)
        self.boot_confidence_spin.setDecimals(2)
        settings_layout.addRow("Confidence Level:", self.boot_confidence_spin)
        
        self.boot_method_combo = QComboBox()
        self.boot_method_combo.addItems(["Simple", "Stratified", "Block", "Parametric"])
        settings_layout.addRow("Method:", self.boot_method_combo)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Column selection
        columns_group = QGroupBox("Variables to Analyze")
        columns_layout = QVBoxLayout()
        
        self.boot_columns_list = QTextEdit()
        self.boot_columns_list.setPlaceholderText("Enter column names (one per line):\ngrade\ntonnage\nvalue")
        self.boot_columns_list.setMaximumHeight(100)
        columns_layout.addWidget(self.boot_columns_list)
        
        columns_group.setLayout(columns_layout)
        layout.addWidget(columns_group)
        
        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.boot_results_table = QTableWidget()
        self.boot_results_table.setColumnCount(5)
        self.boot_results_table.setHorizontalHeaderLabels([
            "Variable", "Original", "Bootstrap Mean", "CI Lower", "CI Upper"
        ])
        results_layout.addWidget(self.boot_results_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Run button
        run_btn = QPushButton("Run Bootstrap Analysis")
        run_btn.clicked.connect(lambda: self._start_analysis('bootstrap'))
        layout.addWidget(run_btn)
        
        layout.addStretch()
        return widget
    
    def _create_lhs_tab(self) -> QWidget:
        """Create Latin Hypercube Sampling tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel(
            "<b>Latin Hypercube Sampling</b><br>"
            "Efficient stratified sampling for multi-variate parameter space exploration.<br>"
            "Use for sensitivity analysis or as input to Monte Carlo simulation."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Settings
        settings_group = QGroupBox("LHS Settings")
        settings_layout = QFormLayout()
        
        self.lhs_samples_spin = QSpinBox()
        self.lhs_samples_spin.setRange(10, 1000)
        self.lhs_samples_spin.setValue(100)
        settings_layout.addRow("Number of Samples:", self.lhs_samples_spin)
        
        self.lhs_criterion_combo = QComboBox()
        self.lhs_criterion_combo.addItems(["Maximin", "Correlation", "Random"])
        settings_layout.addRow("Optimization Criterion:", self.lhs_criterion_combo)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Export button
        export_btn = QPushButton("Generate & Export Samples")
        export_btn.clicked.connect(lambda: self._start_analysis('lhs'))
        layout.addWidget(export_btn)
        
        layout.addStretch()
        return widget
    
    def _create_prob_shells_tab(self) -> QWidget:
        """Create probabilistic pit shells tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel(
            "<b>Probabilistic Pit Shells</b><br>"
            "Generate multiple pit shells under uncertainty to quantify block-level mining risk.<br>"
            "Results show mining probability and risk classification (Robust/Marginal/Fringe)."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()
        
        self.ps_realizations_spin = QSpinBox()
        self.ps_realizations_spin.setRange(10, 500)
        self.ps_realizations_spin.setValue(100)
        settings_layout.addRow("Number of Realizations:", self.ps_realizations_spin)
        
        self.ps_use_realizations_check = QCheckBox("Use Grade Realizations")
        self.ps_use_realizations_check.setChecked(True)
        settings_layout.addRow("", self.ps_use_realizations_check)
        
        self.ps_robust_spin = QDoubleSpinBox()
        self.ps_robust_spin.setRange(0.5, 1.0)
        self.ps_robust_spin.setValue(0.80)
        self.ps_robust_spin.setSingleStep(0.05)
        self.ps_robust_spin.setDecimals(2)
        settings_layout.addRow("Robust Threshold:", self.ps_robust_spin)
        
        self.ps_fringe_spin = QDoubleSpinBox()
        self.ps_fringe_spin.setRange(0.0, 0.5)
        self.ps_fringe_spin.setValue(0.20)
        self.ps_fringe_spin.setSingleStep(0.05)
        self.ps_fringe_spin.setDecimals(2)
        settings_layout.addRow("Fringe Threshold:", self.ps_fringe_spin)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Run button
        run_btn = QPushButton("Run Probabilistic Analysis")
        run_btn.clicked.connect(lambda: self._start_analysis('prob_shells'))
        layout.addWidget(run_btn)
        
        layout.addStretch()
        return widget
    
    def _create_dashboard_tab(self) -> QWidget:
        """Create risk dashboard tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Dashboard controls
        controls_group = QGroupBox("Dashboard Controls")
        controls_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh Dashboard")
        refresh_btn.clicked.connect(self._refresh_dashboard)
        controls_layout.addWidget(refresh_btn)
        
        export_btn = QPushButton("Export Figures")
        export_btn.clicked.connect(self._export_dashboard)
        controls_layout.addWidget(export_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Summary table
        summary_group = QGroupBox("KPI Summary")
        summary_layout = QVBoxLayout()
        
        self.dashboard_summary_table = QTableWidget()
        summary_layout.addWidget(self.dashboard_summary_table)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Visualization area
        if MATPLOTLIB_AVAILABLE:
            viz_group = QGroupBox("Visualizations")
            viz_layout = QVBoxLayout()
            
            self.dashboard_canvas = FigureCanvas(Figure(figsize=(10, 6)))
            viz_layout.addWidget(self.dashboard_canvas)
            
            viz_group.setLayout(viz_layout)
            layout.addWidget(viz_group)
        
        return widget
    
    def _create_progress_section(self) -> QGroupBox:
        """Create progress monitoring section."""
        group = QGroupBox("Analysis Progress")
        layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        layout.addWidget(self.progress_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        group.setLayout(layout)
        return group
    
    def set_block_model(self, block_model: pd.DataFrame):
        """Set the block model for analysis."""
        self._block_model = block_model  # Use private backing field (property contract)
        logger.info(f"Block model set: {len(block_model)} blocks")
    
    # ------------------------------------------------------------------
    # BaseAnalysisPanel overrides
    # ------------------------------------------------------------------
    
    def _start_analysis(self, analysis_type: str):
        """Start analysis of specified type."""
        self.current_analysis_type = analysis_type
        
        # Use optimized path for Fixed Schedule Monte Carlo
        if analysis_type == 'monte_carlo' and self.mc_mode_combo.currentText() == "Fixed Schedule":
            self._run_optimized_monte_carlo()
        else:
            self.run_analysis()
    
    def _run_optimized_monte_carlo(self):
        """Run optimized Monte Carlo simulation for Fixed Schedule mode."""
        if not self.validate_inputs():
            return
        
        try:
            from ..uncertainty_engine.monte_carlo import MonteCarloSimulator, MonteCarloConfig, SimulationMode
            
            # Get schedule from registry
            loaded_schedule = None
            if self.registry:
                loaded_schedule = self.registry.get_schedule()
            
            if loaded_schedule is None:
                self.show_error("No Schedule", "Please load or generate a schedule first for Fixed Schedule mode.")
                return
            
            # Convert schedule to DataFrame if needed
            if not isinstance(loaded_schedule, pd.DataFrame):
                if hasattr(loaded_schedule, 'to_dataframe'):
                    schedule_df = loaded_schedule.to_dataframe()
                elif isinstance(loaded_schedule, dict) and 'schedule' in loaded_schedule:
                    schedule_df = loaded_schedule['schedule']
                else:
                    self.show_error("Invalid Schedule", "Schedule format not recognized.")
                    return
            else:
                schedule_df = loaded_schedule
            
            # Ensure schedule has required columns
            if 'block_id' not in schedule_df.columns or 'period' not in schedule_df.columns:
                self.show_error("Invalid Schedule", "Schedule must have 'block_id' and 'period' columns.")
                return
            
            # Create config
            config = MonteCarloConfig(
                n_simulations=self.mc_n_simulations_spin.value(),
                mode=SimulationMode.FIXED_SCHEDULE,
                random_seed=self.mc_seed_spin.value() if self.mc_seed_spin.value() > 0 else None,
                parallel=False  # Optimized path doesn't use parallel processing
            )
            
            # Create simulator
            simulator = MonteCarloSimulator(config)
            
            # Collect parameter specs
            param_specs = self._collect_parameter_specs()
            
            if not param_specs:
                self.show_error("No Parameters", "Please define at least one parameter for Monte Carlo simulation.")
                return
            
            # Update progress
            self._update_progress(0, 100, "Running optimized Monte Carlo simulation...")
            self.cancel_btn.setEnabled(True)
            
            # Run optimized simulation
            npv_results, param_df = simulator.run_fixed_schedule_optimized(
                self.block_model,
                schedule_df,
                self.mc_n_simulations_spin.value(),
                param_specs
            )
            
            # Create a simplified result structure for display
            from ..uncertainty_engine.monte_carlo import MonteCarloResults, SimulationResult
            
            # Create mock simulation results for compatibility
            simulations = []
            for i, npv in enumerate(npv_results):
                sim_result = SimulationResult(
                    simulation_id=i,
                    parameters=param_df.iloc[i].to_dict(),
                    npv=float(npv)
                )
                simulations.append(sim_result)
            
            mc_results = MonteCarloResults(
                config=config,
                simulations=simulations
            )
            mc_results.compute_summary_statistics()
            mc_results.compute_risk_metrics()
            
            # Display results
            self._update_progress(100, 100, "Simulation complete!")
            self._on_analysis_complete({
                'type': 'monte_carlo',
                'results': mc_results
            })
            
        except Exception as e:
            logger.exception("Error in optimized Monte Carlo simulation")
            self._on_analysis_error(str(e))
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all parameters from the UI based on current analysis type."""
        if self.current_analysis_type == 'monte_carlo':
            return {
                "analysis_type": "monte_carlo",
                "block_model": self.block_model,
                "n_simulations": self.mc_n_simulations_spin.value(),
                "mode": 'dynamic' if self.mc_mode_combo.currentText() == "Dynamic Re-optimize" else 'fixed',
                "random_seed": self.mc_seed_spin.value() if self.mc_seed_spin.value() > 0 else None,
                "parallel": self.mc_parallel_check.isChecked(),
                "parameters": self._collect_parameter_specs()
            }
        elif self.current_analysis_type == 'bootstrap':
            columns_text = self.boot_columns_list.toPlainText()
            columns = [c.strip() for c in columns_text.split('\n') if c.strip()]
            return {
                "analysis_type": "bootstrap",
                "block_model": self.block_model,
                "n_iterations": self.boot_iterations_spin.value(),
                "confidence_level": self.boot_confidence_spin.value(),
                "method": self.boot_method_combo.currentText().lower(),
                "columns": columns
            }
        elif self.current_analysis_type == 'lhs':
            return {
                "analysis_type": "lhs",
                "block_model": self.block_model,
                "n_samples": self.lhs_samples_spin.value(),
                "parameters": self._collect_parameter_specs()
            }
        elif self.current_analysis_type == 'prob_shells':
            return {
                "analysis_type": "prob_shells",
                "block_model": self.block_model,
                "n_realizations": self.ps_realizations_spin.value(),
                "use_grade_realizations": self.ps_use_realizations_check.isChecked(),
                "robust_threshold": self.ps_robust_spin.value(),
                "fringe_threshold": self.ps_fringe_spin.value(),
            }
        else:
            raise ValueError(f"Unknown analysis type: {self.current_analysis_type}")
    
    def validate_inputs(self) -> bool:
        """Validate collected parameters."""
        if not super().validate_inputs():
            return False
        
        if self.block_model is None or self.block_model.empty:
            self.show_error("No Data", "Please load a block model first.")
            return False
        
        if self.current_analysis_type == 'bootstrap':
            columns_text = self.boot_columns_list.toPlainText()
            columns = [c.strip() for c in columns_text.split('\n') if c.strip()]
            if not columns:
                self.show_error("No Columns", "Please specify columns to analyze.")
                return False
        elif self.current_analysis_type == 'lhs':
            param_specs = self._collect_parameter_specs()
            if not param_specs:
                self.show_error("No Parameters", "Please define parameters for LHS.")
                return False
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Process and display uncertainty analysis results."""
        self.results = payload.get("results")
        analysis_type = payload.get("analysis_type", self.current_analysis_type)
        
        if self.results:
            self._on_analysis_complete({'type': analysis_type, 'results': self.results})
            self.show_info("Analysis Complete", f"{analysis_type.replace('_', ' ').title()} analysis completed successfully.")
    
    def _collect_parameter_specs(self) -> Dict:
        """Collect parameter specifications from UI."""
        params = {}
        
        # Find all parameter groups
        for param_name in ['price', 'mining_cost', 'recovery']:
            enable_check = self.findChild(QCheckBox, f"{param_name}_enable")
            if not enable_check or not enable_check.isChecked():
                continue
            
            dist_combo = self.findChild(QComboBox, f"{param_name}_dist")
            base_spin = self.findChild(QDoubleSpinBox, f"{param_name}_base")
            std_spin = self.findChild(QDoubleSpinBox, f"{param_name}_std")
            min_spin = self.findChild(QDoubleSpinBox, f"{param_name}_min")
            max_spin = self.findChild(QDoubleSpinBox, f"{param_name}_max")
            
            if not all([dist_combo, base_spin]):
                continue
            
            dist_type = dist_combo.currentText().lower().replace('-', '')
            
            params[param_name] = {
                'base_value': base_spin.value(),
                'distribution': dist_type
            }
            
            if dist_type == 'normal':
                params[param_name]['std_dev'] = std_spin.value() if std_spin else 0.0
            elif dist_type == 'triangular':
                params[param_name]['min_value'] = min_spin.value() if min_spin else base_spin.value() * 0.8
                params[param_name]['mode_value'] = base_spin.value()
                params[param_name]['max_value'] = max_spin.value() if max_spin else base_spin.value() * 1.2
            elif dist_type == 'uniform':
                params[param_name]['min_value'] = min_spin.value() if min_spin else base_spin.value() * 0.8
                params[param_name]['max_value'] = max_spin.value() if max_spin else base_spin.value() * 1.2
            elif dist_type == 'lognormal':
                params[param_name]['std_dev'] = std_spin.value() if std_spin else 0.0
        
        return params
    
    def _update_progress(self, current: int, total: int, message: str):
        """Update progress bar and label."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)
    
    def _on_analysis_complete(self, results: Dict):
        """Handle analysis completion."""
        self.results = results
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Analysis complete!")
        
        analysis_type = results.get('type')
        
        if analysis_type == 'monte_carlo':
            self._display_monte_carlo_results(results['results'])
        elif analysis_type == 'bootstrap':
            self._display_bootstrap_results(results['results'])
        elif analysis_type == 'lhs':
            self._export_lhs_results(results['results'])
        
        QMessageBox.information(self, "Success", f"{analysis_type.replace('_', ' ').title()} analysis complete!")
    
    def _on_analysis_error(self, error_msg: str):
        """Handle analysis error."""
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Analysis failed")
        QMessageBox.critical(self, "Analysis Error", f"Error during analysis:\n\n{error_msg}")
    
    def _cancel_analysis(self):
        """Cancel running analysis."""
        if self.worker:
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.progress_label.setText("Cancelled")
    
    def _display_monte_carlo_results(self, results):
        """Display Monte Carlo results."""
        # Update summary table
        if results.summary_stats is not None:
            self._populate_table(self.dashboard_summary_table, results.summary_stats)
        
        # Generate dashboard
        self._refresh_dashboard()
    
    def _display_bootstrap_results(self, results: Dict):
        """Display bootstrap results."""
        self.boot_results_table.setRowCount(len(results))
        
        for i, (var_name, result) in enumerate(results.items()):
            self.boot_results_table.setItem(i, 0, QTableWidgetItem(var_name))
            self.boot_results_table.setItem(i, 1, QTableWidgetItem(f"{result.original_statistic:.4f}"))
            self.boot_results_table.setItem(i, 2, QTableWidgetItem(f"{result.bootstrap_mean:.4f}"))
            self.boot_results_table.setItem(i, 3, QTableWidgetItem(f"{result.ci_lower:.4f}"))
            self.boot_results_table.setItem(i, 4, QTableWidgetItem(f"{result.ci_upper:.4f}"))
    
    def _export_lhs_results(self, samples_df: pd.DataFrame):
        """Export LHS samples to file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export LHS Samples", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if filepath:
            # Step 10: Use ExportHelpers
            from ..utils.export_helpers import export_dataframe_to_csv
            export_dataframe_to_csv(samples_df, filepath)
            QMessageBox.information(self, "Exported", f"LHS samples exported to:\n{filepath}")
    
    def _refresh_dashboard(self):
        """Refresh dashboard visualizations."""
        if not self.results or not MATPLOTLIB_AVAILABLE:
            return
        
        results = self.results.get('results')
        if not results:
            return
        
        # Generate dashboard figures
        figures = self.dashboard.create_full_dashboard(results)
        
        # Display first figure in canvas
        if figures and self.dashboard_canvas:
            first_fig = list(figures.values())[0]
            self.dashboard_canvas.figure = first_fig
            self.dashboard_canvas.draw()
    
    def _export_dashboard(self):
        """Export dashboard figures."""
        if not self.results:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        
        if output_dir:
            results = self.results.get('results')
            figures = self.dashboard.create_full_dashboard(results, Path(output_dir))
            QMessageBox.information(self, "Exported", f"Dashboard exported to:\n{output_dir}")
    
    def _populate_table(self, table: QTableWidget, dataframe: pd.DataFrame):
        """Populate QTableWidget from DataFrame."""
        table.setRowCount(len(dataframe))
        table.setColumnCount(len(dataframe.columns) + 1)  # +1 for index
        
        # Set headers
        headers = ["Metric"] + list(dataframe.columns)
        table.setHorizontalHeaderLabels(headers)
        
        # Fill data
        for i, (index, row) in enumerate(dataframe.iterrows()):
            table.setItem(i, 0, QTableWidgetItem(str(index)))
            for j, value in enumerate(row):
                table.setItem(i, j + 1, QTableWidgetItem(f"{value:.2f}"))
        
        table.resizeColumnsToContents()
