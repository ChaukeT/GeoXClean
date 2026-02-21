"""
Research Dashboard Panel
========================

Comprehensive research and validation panel for geostatistical methods.
Provides professional analysis tools for estimation and simulation results.

Features:
1. Estimation Results Analysis (Kriging/CoK/UK/SK)
   - Kriging Summary Tables
   - Slope of Regression Diagnostics
   - Cross-Validation (LOOCV)
   - Swath Plots
   - Kriging Variance Analysis

2. Simulation Results Analysis (SGSIM/SIS/DBS/MPS)
   - Realization Statistics
   - Histogram Reproduction
   - Variogram Reproduction
   - Connectivity Analysis
   - Realization Viewer

3. Uncertainty & Risk Analysis
   - Uncertainty Grids
   - Economic Uncertainty
   - Volume/Tonnage Uncertainty
   - Spatial Risk Maps

4. Professional Reporting
   - Variogram Reports
   - Estimation Reports
   - Simulation Reports
   - Resource Classification Packs
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QListWidget, QListWidgetItem, QMessageBox,
    QTabWidget, QWidget, QTextEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QSplitter, QScrollArea, QFrame, QDoubleSpinBox,
    QSpinBox, QCheckBox, QLineEdit
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .base_analysis_panel import BaseAnalysisPanel
from ..research.analysis_tools import (
    compute_kriging_summary_table,
    compute_slope_of_regression,
    leave_one_out_cross_validation,
    compute_swath_plot_data,
    compute_simulation_reproduction_stats,
    compute_uncertainty_grids
)

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class ResearchDashboardPanel(BaseAnalysisPanel):
    """
    Comprehensive Research Dashboard for geostatistical analysis and validation.
    
    Provides professional tools for comparing methods, validating results,
    and generating audit-ready reports.
    """
    # PanelManager metadata
    PANEL_ID = "ResearchDashboardPanel"
    PANEL_NAME = "ResearchDashboard Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "research_dashboard"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="research_dashboard")
        self.setWindowTitle("Research & Validation Dashboard")
        self.resize(1400, 900)
        
        # Data storage
        self.kriging_results: Dict[str, Any] = {}
        self.simulation_results: Dict[str, Any] = {}
        self.composites_data: Optional[pd.DataFrame] = None
        
        # Connect to registry
        self._init_registry()
        
        # Build UI
        self._build_ui()
        
        logger.info("Initialized Research Dashboard panel")
    


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
    def _build_ui(self):
        """Build the comprehensive research dashboard UI."""
        self._setup_ui()
    
    def _init_registry(self):
        """Initialize DataRegistry connections."""
        try:
            self.registry = self.get_registry()
            if self.registry:
                # Connect to all relevant signals
                self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
                self.registry.variogramResultsLoaded.connect(self._on_variogram_loaded)
                self.registry.krigingResultsLoaded.connect(self._on_kriging_results_loaded)
                self.registry.simpleKrigingResultsLoaded.connect(self._on_kriging_results_loaded)
                self.registry.universalKrigingResultsLoaded.connect(self._on_kriging_results_loaded)
                self.registry.cokrigingResultsLoaded.connect(self._on_kriging_results_loaded)
                self.registry.sgsimResultsLoaded.connect(self._on_simulation_results_loaded)
                
                # Load existing data
                drillhole_data = self.registry.get_drillhole_data()
                if drillhole_data:
                    self._on_drillhole_data_loaded(drillhole_data)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
    
    def _setup_ui(self):
        """Set up the main UI with tabs for different analysis types."""
        # Clear existing widgets from main_layout (which is the scroll content layout from base class)
        if self.main_layout:
            while self.main_layout.count():
                item = self.main_layout.takeAt(0)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.hide()
                        widget.setParent(None)
                        widget.deleteLater()
        
        # Use main_layout which is already set up by BaseAnalysisPanel's _setup_base_ui
        main_layout = self.main_layout
        if main_layout is None:
            # Fallback: create layout if somehow main_layout wasn't set
            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(10, 10, 10, 10)
            self.main_layout = main_layout
        
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Research & Validation Dashboard")
        title.setStyleSheet("font-size: 18pt; font-weight: bold; padding: 10px;")
        main_layout.addWidget(title)
        
        # Create main tabs
        tabs = QTabWidget()
        
        # Tab 1: Estimation Analysis
        est_tab = self._create_estimation_analysis_tab()
        tabs.addTab(est_tab, "Estimation Analysis")
        
        # Tab 2: Simulation Analysis
        sim_tab = self._create_simulation_analysis_tab()
        tabs.addTab(sim_tab, "Simulation Analysis")
        
        # Tab 3: Uncertainty & Risk
        unc_tab = self._create_uncertainty_analysis_tab()
        tabs.addTab(unc_tab, "Uncertainty & Risk")
        
        # Tab 4: Reporting
        report_tab = self._create_reporting_tab()
        tabs.addTab(report_tab, "Reporting")
        
        main_layout.addWidget(tabs)
    
    def _create_estimation_analysis_tab(self) -> QWidget:
        """Create estimation analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Splitter for left (controls) and right (results)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # LEFT: Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Data selection
        data_group = QGroupBox("1. Data Selection")
        data_layout = QVBoxLayout()
        
        self.kriging_method_combo = QComboBox()
        self.kriging_method_combo.addItems(["Select Kriging Method...", "Ordinary Kriging", "Simple Kriging", 
                                           "Universal Kriging", "Co-Kriging"])
        data_layout.addWidget(QLabel("Kriging Method:"))
        data_layout.addWidget(self.kriging_method_combo)
        
        self.load_kriging_btn = QPushButton("Load Kriging Results")
        self.load_kriging_btn.clicked.connect(self._load_kriging_results)
        data_layout.addWidget(self.load_kriging_btn)
        
        data_group.setLayout(data_layout)
        scroll_layout.addWidget(data_group)
        
        # Analysis options
        analysis_group = QGroupBox("2. Analysis Options")
        analysis_layout = QVBoxLayout()
        
        self.summary_table_check = QCheckBox("Kriging Summary Tables")
        self.summary_table_check.setChecked(True)
        analysis_layout.addWidget(self.summary_table_check)
        
        self.slope_regression_check = QCheckBox("Slope of Regression Diagnostics")
        self.slope_regression_check.setChecked(True)
        analysis_layout.addWidget(self.slope_regression_check)
        
        self.cross_validation_check = QCheckBox("Cross-Validation (LOOCV)")
        analysis_layout.addWidget(self.cross_validation_check)
        
        self.swath_plots_check = QCheckBox("Swath Plots")
        analysis_layout.addWidget(self.swath_plots_check)
        
        self.variance_analysis_check = QCheckBox("Kriging Variance Analysis")
        analysis_layout.addWidget(self.variance_analysis_check)
        
        analysis_group.setLayout(analysis_layout)
        scroll_layout.addWidget(analysis_group)
        
        # Domain selection
        domain_group = QGroupBox("3. Domain Selection")
        domain_layout = QVBoxLayout()
        
        self.domain_combo = QComboBox()
        self.domain_combo.addItem("All Domains")
        domain_layout.addWidget(QLabel("Group By:"))
        domain_layout.addWidget(self.domain_combo)
        
        domain_group.setLayout(domain_layout)
        scroll_layout.addWidget(domain_group)
        
        # Run analysis button
        run_btn = QPushButton("Run Analysis")
        run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        run_btn.clicked.connect(self._run_estimation_analysis)
        scroll_layout.addWidget(run_btn)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)
        
        # RIGHT: Results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Results tabs
        results_tabs = QTabWidget()
        
        # Summary table tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.summary_table = QTableWidget()
        summary_layout.addWidget(self.summary_table)
        results_tabs.addTab(summary_tab, "Summary Tables")
        
        # Slope of regression tab
        slope_tab = QWidget()
        slope_layout = QVBoxLayout(slope_tab)
        self.slope_table = QTableWidget()
        slope_layout.addWidget(self.slope_table)
        results_tabs.addTab(slope_tab, "Slope of Regression")
        
        # Cross-validation tab
        cv_tab = QWidget()
        cv_layout = QVBoxLayout(cv_tab)
        self.cv_text = QTextEdit()
        self.cv_text.setReadOnly(True)
        cv_layout.addWidget(self.cv_text)
        results_tabs.addTab(cv_tab, "Cross-Validation")
        
        # Swath plots tab
        swath_tab = QWidget()
        swath_layout = QVBoxLayout(swath_tab)
        self.swath_text = QTextEdit()
        self.swath_text.setReadOnly(True)
        swath_layout.addWidget(self.swath_text)
        results_tabs.addTab(swath_tab, "Swath Plots")
        
        right_layout.addWidget(results_tabs)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        return widget
    
    def _create_simulation_analysis_tab(self) -> QWidget:
        """Create simulation analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        label = QLabel("Simulation Analysis - Coming Soon")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        return widget
    
    def _create_uncertainty_analysis_tab(self) -> QWidget:
        """Create uncertainty & risk analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        label = QLabel("Uncertainty & Risk Analysis - Coming Soon")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        return widget
    
    def _create_reporting_tab(self) -> QWidget:
        """Create professional reporting tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Report generation controls
        controls_group = QGroupBox("Generate Reports")
        controls_layout = QVBoxLayout()
        
        report_types = [
            ("Variogram Report", self._generate_variogram_report),
            ("Estimation Report", self._generate_estimation_report),
            ("Simulation Report", self._generate_simulation_report),
            ("Resource Classification Pack", self._generate_classification_pack)
        ]
        
        for report_name, callback in report_types:
            btn = QPushButton(f"Generate {report_name}")
            btn.clicked.connect(callback)
            controls_layout.addWidget(btn)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Report preview
        preview_group = QGroupBox("Report Preview")
        preview_layout = QVBoxLayout()
        
        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        self.report_preview.setFont(QFont("Courier", 9))
        preview_layout.addWidget(self.report_preview)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        return widget
    
    def _on_drillhole_data_loaded(self, data: Dict[str, Any]):
        """Handle drillhole data loaded from registry."""
        if isinstance(data, dict):
            # Fix: Explicitly check for non-empty DataFrames to avoid ValueError
            composites = data.get('composites')
            assays = data.get('assays')
            if isinstance(composites, pd.DataFrame) and not composites.empty:
                self.composites_data = composites
            elif isinstance(assays, pd.DataFrame) and not assays.empty:
                self.composites_data = assays
            else:
                self.composites_data = None
        else:
            self.composites_data = data
        
        logger.info("Research Dashboard: Loaded drillhole data")
    
    def _on_kriging_results_loaded(self, results: Dict[str, Any]):
        """Handle kriging results loaded from registry."""
        method = results.get('method', 'Ordinary Kriging')
        self.kriging_results[method] = results
        logger.info(f"Research Dashboard: Loaded {method} results")
    
    def _on_simulation_results_loaded(self, results: Dict[str, Any]):
        """Handle simulation results loaded from registry."""
        method = results.get('method', 'SGSIM')
        self.simulation_results[method] = results
        logger.info(f"Research Dashboard: Loaded {method} results")
    
    def _on_variogram_loaded(self, results: Dict[str, Any]):
        """Handle variogram results loaded."""
        self.variogram_results = results
    
    def _load_kriging_results(self):
        """Load kriging results from registry."""
        if not self.registry:
            QMessageBox.warning(self, "No Registry", "DataRegistry not available")
            return
        
        method = self.kriging_method_combo.currentText()
        if method == "Select Kriging Method...":
            QMessageBox.warning(self, "No Method", "Please select a kriging method")
            return
        
        # Try to get results from registry
        if method == "Ordinary Kriging":
            results = self.registry.get_kriging_results()
        elif method == "Simple Kriging":
            results = self.registry.get_simple_kriging_results()
        elif method == "Universal Kriging":
            results = self.registry.get_universal_kriging_results()
        elif method == "Co-Kriging":
            results = self.registry.get_cokriging_results()
        else:
            results = None
        
        if results:
            self.kriging_results[method] = results
            QMessageBox.information(self, "Loaded", f"Loaded {method} results")
        else:
            QMessageBox.warning(self, "No Results", f"No {method} results found in registry")
    
    def _run_estimation_analysis(self):
        """Run estimation analysis based on selected options."""
        if self.composites_data is None:
            QMessageBox.warning(self, "No Data", "Please load drillhole/composite data first")
            return
        
        if not self.kriging_results:
            QMessageBox.warning(self, "No Results", "Please load kriging results first")
            return
        
        # Get selected kriging method
        method = self.kriging_method_combo.currentText()
        if method == "Select Kriging Method..." or method not in self.kriging_results:
            QMessageBox.warning(self, "No Results", f"No results loaded for {method}")
            return
        
        results = self.kriging_results[method]
        estimates = results.get('estimates')
        variable = results.get('variable', 'Grade')
        
        if estimates is None:
            QMessageBox.warning(self, "Invalid Results", "Results do not contain estimates")
            return
        
        # Run selected analyses
        if self.summary_table_check.isChecked():
            self._compute_summary_tables(results, variable)
        
        if self.slope_regression_check.isChecked():
            self._compute_slope_of_regression(results, variable)
        
        if self.cross_validation_check.isChecked():
            self._run_cross_validation(results, variable)
        
        if self.swath_plots_check.isChecked():
            self._compute_swath_plots(results, variable)
    
    def _compute_summary_tables(self, results: Dict[str, Any], variable: str):
        """Compute and display kriging summary tables."""
        try:
            estimates = results.get('estimates')
            if estimates is None:
                return
            
            domain_col = self.domain_combo.currentText() if self.domain_combo.currentText() != "All Domains" else None
            
            df = compute_kriging_summary_table(
                self.composites_data,
                estimates,
                variable,
                domain_col
            )
            
            # Display in table
            self.summary_table.setRowCount(len(df))
            self.summary_table.setColumnCount(len(df.columns))
            self.summary_table.setHorizontalHeaderLabels(df.columns.tolist())
            
            for i, row in df.iterrows():
                for j, col in enumerate(df.columns):
                    item = QTableWidgetItem(str(row[col]))
                    self.summary_table.setItem(i, j, item)
            
            self.summary_table.resizeColumnsToContents()
            
        except Exception as e:
            logger.error(f"Error computing summary tables: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to compute summary tables:\n{e}")
    
    def _compute_slope_of_regression(self, results: Dict[str, Any], variable: str):
        """Compute and display slope of regression diagnostics."""
        try:
            estimates = results.get('estimates')
            if estimates is None:
                return
            
            group_by = self.domain_combo.currentText() if self.domain_combo.currentText() != "All Domains" else None
            
            df = compute_slope_of_regression(
                self.composites_data,
                estimates,
                variable,
                group_by
            )
            
            # Display in table
            self.slope_table.setRowCount(len(df))
            self.slope_table.setColumnCount(len(df.columns))
            self.slope_table.setHorizontalHeaderLabels(df.columns.tolist())
            
            for i, row in df.iterrows():
                for j, col in enumerate(df.columns):
                    item = QTableWidgetItem(str(row[col]))
                    self.slope_table.setItem(i, j, item)
            
            self.slope_table.resizeColumnsToContents()
            
        except Exception as e:
            logger.error(f"Error computing slope of regression: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to compute slope of regression:\n{e}")
    
    def _run_cross_validation(self, results: Dict[str, Any], variable: str):
        """Run leave-one-out cross-validation."""
        try:
            # This would require access to original data coordinates and kriging function
            # For now, show a placeholder
            self.cv_text.setText("Cross-Validation requires access to original kriging function.\n"
                               "This feature will be implemented with full integration.")
        except Exception as e:
            logger.error(f"Error running cross-validation: {e}", exc_info=True)
    
    def _compute_swath_plots(self, results: Dict[str, Any], variable: str):
        """Compute swath plot data."""
        try:
            estimates = results.get('estimates')
            grid_x = results.get('grid_x')
            grid_y = results.get('grid_y')
            grid_z = results.get('grid_z')
            
            if estimates is None or grid_x is None:
                return
            
            # Create grid coordinates
            if grid_x.ndim == 1:
                grid_coords = np.column_stack([grid_x, grid_y, grid_z])
            else:
                grid_coords = np.column_stack([
                    grid_x.ravel(order='F'),
                    grid_y.ravel(order='F'),
                    grid_z.ravel(order='F')
                ])
            
            # Compute swath for each direction
            swath_data = {}
            for direction in ['X', 'Y', 'Z']:
                df = compute_swath_plot_data(
                    self.composites_data,
                    estimates.ravel(order='F') if estimates.ndim > 1 else estimates,
                    grid_coords,
                    direction
                )
                swath_data[direction] = df
            
            # Display summary
            text = "Swath Plot Data Computed:\n\n"
            for direction, df in swath_data.items():
                text += f"{direction}-Direction Swath:\n"
                text += f"  Bins: {len(df)}\n"
                text += f"  Composite Mean Range: [{df['Comp_Mean'].min():.2f}, {df['Comp_Mean'].max():.2f}]\n"
                text += f"  Estimate Mean Range: [{df['Est_Mean'].min():.2f}, {df['Est_Mean'].max():.2f}]\n\n"
            
            self.swath_text.setText(text)
            
        except Exception as e:
            logger.error(f"Error computing swath plots: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to compute swath plots:\n{e}")
    
    def _generate_variogram_report(self):
        """Generate variogram report."""
        if not hasattr(self, 'variogram_results') or not self.variogram_results:
            QMessageBox.warning(self, "No Data", "No variogram results available")
            return
        
        report = "VARIogram REPORT\n"
        report += "=" * 70 + "\n\n"
        report += "This is a placeholder for the full variogram report.\n"
        report += "Full implementation will include:\n"
        report += "- Experimental variograms (all directions)\n"
        report += "- Fitted variogram models\n"
        report += "- Anisotropy tables\n"
        report += "- Nested structures\n"
        report += "- Variogram fitting diagnostics\n"
        
        self.report_preview.setText(report)
    
    def _generate_estimation_report(self):
        """Generate estimation report."""
        report = "ESTIMATION REPORT\n"
        report += "=" * 70 + "\n\n"
        report += "This is a placeholder for the full estimation report.\n"
        self.report_preview.setText(report)
    
    def _generate_simulation_report(self):
        """Generate simulation report."""
        report = "SIMULATION REPORT\n"
        report += "=" * 70 + "\n\n"
        report += "This is a placeholder for the full simulation report.\n"
        self.report_preview.setText(report)
    
    def _generate_classification_pack(self):
        """Generate resource classification pack."""
        report = "RESOURCE CLASSIFICATION PACK\n"
        report += "=" * 70 + "\n\n"
        report += "This is a placeholder for the classification pack.\n"
        self.report_preview.setText(report)
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Gather parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass
