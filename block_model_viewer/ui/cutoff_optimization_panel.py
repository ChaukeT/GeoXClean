"""
Cut-off Optimization Panel for Economic Analysis.

This panel provides economic optimization features for cut-off grade analysis:
- NPV, IRR, and payback period calculations
- Multi-period DCF analysis (audit-grade)
- Economic parameter sensitivity analysis
- Optimal cut-off determination

Requires a grade-tonnage curve from GradeTonnageBasicPanel or GradeTonnagePanel.

Author: GeoX Mining Software
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QCheckBox,
    QSplitter, QTabWidget, QFormLayout, QFrame, QScrollArea,
    QProgressBar, QTextEdit, QSpinBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

# Matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.ticker
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..mine_planning.cutoff.geostats_grade_tonnage import (
    CutoffSensitivityEngine,
    CutoffSensitivityAnalysis,
    CutoffOptimizationMethod,
    GradeTonnageCurve,
    GeostatsGradeTonnageEngine,
    GeostatsGradeTonnageConfig,
    DataMode
)

from ..mine_planning.cutoff.mine_economics import (
    MineEconomicsEngine,
    MineEconomicsConfig,
    EconomicParameters,
    MineCapacity,
    CapitalExpenditure,
    TaxParameters
)

from ..mine_planning.cutoff.advanced_visualization import (
    MiningVisualizationCoordinator,
    PlotConfig,
    ColorScheme
)

from .base_panel import BasePanel
from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors

logger = logging.getLogger(__name__)


class CutoffOptimizationPanel(BasePanel):
    """
    Cut-off Optimization Panel for economic analysis.

    Provides NPV/IRR optimization, sensitivity analysis,
    and multi-period DCF calculations.
    """

    # Signals
    optimizationCompleted = pyqtSignal(object, float)  # sensitivity_result, optimal_cutoff
    analysisProgress = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grade_tonnage_curve: Optional[GradeTonnageCurve] = None
        self.sensitivity_result: Optional[CutoffSensitivityAnalysis] = None

        # Apply modern high-contrast stylesheet
        self._apply_modern_stylesheet()

        # Engines
        self.sensitivity_engine = CutoffSensitivityEngine()
        self.economics_engine: Optional[MineEconomicsEngine] = None
        self.viz_coordinator = MiningVisualizationCoordinator()
        self.gt_engine = GeostatsGradeTonnageEngine()



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
    def _apply_modern_stylesheet(self):
        """Apply modern high-contrast stylesheet for better visibility."""
        colors = get_theme_colors()
        base_style = get_analysis_panel_stylesheet()

        # Add panel-specific enhancements
        panel_specific = f"""
            /* High-contrast labels */
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 10pt;
            }}

            /* Professional GroupBox styling */
            QGroupBox {{
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #222222;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                color: #3498db;
                padding: 0 5px;
            }}

            /* Enhanced inputs */
            QDoubleSpinBox, QSpinBox, QComboBox {{
                background-color: #111;
                border: 1px solid #555;
                color: white;
                padding: 4px;
                min-height: 25px;
            }}

            /* Button styling */
            QPushButton {{
                background-color: #2c3e50;
                color: white;
                border: 1px solid #555;
                padding: 8px;
                border-radius: 4px;
            }}

            QPushButton:hover {{
                background-color: #34495e;
                border-color: #3498db;
            }}
        """

        self.setStyleSheet(base_style + panel_specific)

    def on_block_model_changed(self):
        """Called when block model is set via set_block_model().

        Auto-compute a basic GT curve if block model is available.
        """
        if self._block_model is None:
            return

        try:
            # Convert to DataFrame if needed
            if hasattr(self._block_model, 'to_dataframe'):
                df = self._block_model.to_dataframe()
            elif isinstance(self._block_model, pd.DataFrame):
                df = self._block_model
            else:
                logger.warning(f"Unknown block model type: {type(self._block_model)}")
                return

            if df.empty:
                return

            # Auto-detect columns and compute GT curve
            numeric_cols = df.select_dtypes(include=[float, 'int64', 'int32', 'int']).columns.tolist()

            # Find grade, tonnage, and coordinate columns
            grade_col = None
            tonnage_col = None
            x_col = None
            y_col = None
            z_col = None

            for col in numeric_cols:
                col_upper = col.upper()
                if grade_col is None and any(p in col_upper for p in ['GRADE', 'AU', 'AG', 'CU', 'FE', 'ZN', '_EST', '_OK']):
                    grade_col = col
                if tonnage_col is None and any(p in col_upper for p in ['TONNAGE', 'TONNES', 'TONS', 'WEIGHT']):
                    tonnage_col = col
                # Coordinate columns - check for common patterns
                if x_col is None and any(p == col_upper or col_upper.endswith(p) for p in ['X', 'XC', 'XCENTRE', 'EAST', 'EASTING']):
                    x_col = col
                if y_col is None and any(p == col_upper or col_upper.endswith(p) for p in ['Y', 'YC', 'YCENTRE', 'NORTH', 'NORTHING']):
                    y_col = col
                if z_col is None and any(p == col_upper or col_upper.endswith(p) for p in ['Z', 'ZC', 'ZCENTRE', 'RL', 'ELEV', 'ELEVATION']):
                    z_col = col

            if grade_col is None:
                self.status_text.setPlainText(
                    "Block model loaded but no grade column detected.\n"
                    "Use Basic GT Panel first to generate a curve."
                )
                return

            # Use defaults if coordinates not found
            if x_col is None:
                x_col = 'X' if 'X' in df.columns else 'x'
            if y_col is None:
                y_col = 'Y' if 'Y' in df.columns else 'y'
            if z_col is None:
                z_col = 'Z' if 'Z' in df.columns else 'z'

            # Auto-compute GT curve with intelligent cutoff range
            grades = df[grade_col].dropna()
            p5 = grades.quantile(0.05)
            p95 = grades.quantile(0.95)
            grade_max = grades.max()

            # Start from 80% of P5 (or 0), end at P95
            cutoff_min = max(0, p5 * 0.8)
            cutoff_max = p95

            # Round to sensible values based on grade magnitude
            if grade_max > 10:
                cutoff_min = np.floor(cutoff_min)
                cutoff_max = np.ceil(cutoff_max)
            elif grade_max > 1:
                cutoff_min = np.floor(cutoff_min * 10) / 10
                cutoff_max = np.ceil(cutoff_max * 10) / 10
            else:
                cutoff_min = np.floor(cutoff_min * 100) / 100
                cutoff_max = np.ceil(cutoff_max * 100) / 100

            cutoff_range = np.linspace(cutoff_min, cutoff_max, 50)

            self.grade_tonnage_curve = self.gt_engine.calculate_grade_tonnage_curve(
                df,
                cutoff_range=cutoff_range,
                element_column=grade_col,
                tonnage_column=tonnage_col,
                x_column=x_col,
                y_column=y_col,
                z_column=z_col
            )

            self.run_btn.setEnabled(True)
            total_tonnage = self.grade_tonnage_curve.global_statistics.get('total_tonnage', 0)
            self.status_text.setPlainText(
                f"Auto-computed GT curve from block model:\n"
                f"Grade column: {grade_col} (range: {grades.min():.2f} - {grades.max():.2f})\n"
                f"Cutoff range: {cutoff_min:.2f} - {cutoff_max:.2f}\n"
                f"{len(self.grade_tonnage_curve.points)} points, "
                f"{total_tonnage:,.0f} tonnes total"
            )
            logger.info(f"CutoffOptimizationPanel: Auto-computed GT curve with {len(self.grade_tonnage_curve.points)} points")

        except Exception as e:
            logger.warning(f"Could not auto-compute GT curve: {e}")
            self.status_text.setPlainText(
                f"Block model loaded but could not auto-compute GT curve.\n"
                f"Error: {str(e)}\n"
                f"Use Basic GT Panel first to generate a curve."
            )

    def setup_ui(self):
        """Set up the UI."""
        layout = self.main_layout if hasattr(self, 'main_layout') else QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Economic parameters
        left_panel = self._create_economics_panel()
        splitter.addWidget(left_panel)

        # Right: Results
        right_panel = self._create_results_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        layout.addWidget(splitter)

    def _create_economics_panel(self) -> QWidget:
        """Create economic parameters panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(10)

        # Basic Economics
        self._create_basic_economics(config_layout)

        # Advanced Economics (collapsible)
        self._create_advanced_economics(config_layout)

        # Optimization Options
        self._create_optimization_options(config_layout)

        config_layout.addStretch()
        scroll.setWidget(config_widget)
        layout.addWidget(scroll)

        return panel

    def _create_basic_economics(self, layout):
        """Create basic economic parameters group."""
        group = QGroupBox("1. Economic Parameters")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #f57c00;
                border: 2px solid #f57c00;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        form = QFormLayout(group)
        form.setSpacing(8)

        # Metal price
        self.metal_price = QDoubleSpinBox()
        self.metal_price.setRange(0, 10000)
        self.metal_price.setValue(50)
        self.metal_price.setPrefix("$/")
        self.metal_price.setSuffix(" unit")
        form.addRow("Metal Price:", self.metal_price)

        # Operating costs
        cost_layout = QHBoxLayout()

        self.mining_cost = QDoubleSpinBox()
        self.mining_cost.setRange(0, 1000)
        self.mining_cost.setValue(15)
        self.mining_cost.setSuffix(" $/t")
        cost_layout.addWidget(QLabel("Mining:"))
        cost_layout.addWidget(self.mining_cost)

        self.processing_cost = QDoubleSpinBox()
        self.processing_cost.setRange(0, 1000)
        self.processing_cost.setValue(25)
        self.processing_cost.setSuffix(" $/t")
        cost_layout.addWidget(QLabel("Process:"))
        cost_layout.addWidget(self.processing_cost)

        form.addRow("Costs:", cost_layout)

        # Recovery and discount
        param_layout = QHBoxLayout()

        self.recovery = QDoubleSpinBox()
        self.recovery.setRange(0, 1)
        self.recovery.setValue(0.85)
        self.recovery.setSingleStep(0.01)
        param_layout.addWidget(QLabel("Recovery:"))
        param_layout.addWidget(self.recovery)

        self.discount_rate = QDoubleSpinBox()
        self.discount_rate.setRange(0, 0.5)
        self.discount_rate.setValue(0.10)
        self.discount_rate.setSingleStep(0.01)
        param_layout.addWidget(QLabel("Discount:"))
        param_layout.addWidget(self.discount_rate)

        form.addRow("Parameters:", param_layout)

        layout.addWidget(group)

    def _create_advanced_economics(self, layout):
        """Create advanced economics group (collapsible)."""
        group = QGroupBox("2. Multi-Period DCF (Advanced)")
        group.setCheckable(True)
        group.setChecked(False)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #c62828;
                border: 2px solid #c62828;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        form = QFormLayout(group)
        form.setSpacing(8)

        # Mine capacity
        capacity_layout = QHBoxLayout()
        self.annual_capacity = QDoubleSpinBox()
        self.annual_capacity.setRange(100_000, 500_000_000)
        self.annual_capacity.setValue(10_000_000)
        self.annual_capacity.setDecimals(0)
        self.annual_capacity.setSuffix(" t/yr")
        capacity_layout.addWidget(self.annual_capacity)

        self.ramp_up_years = QSpinBox()
        self.ramp_up_years.setRange(1, 10)
        self.ramp_up_years.setValue(2)
        capacity_layout.addWidget(QLabel("Ramp-up:"))
        capacity_layout.addWidget(self.ramp_up_years)
        capacity_layout.addWidget(QLabel("yrs"))

        form.addRow("Capacity:", capacity_layout)

        # Capital expenditure
        capex_layout = QHBoxLayout()
        self.initial_capex = QDoubleSpinBox()
        self.initial_capex.setRange(0, 50_000_000_000)
        self.initial_capex.setValue(500_000_000)
        self.initial_capex.setDecimals(0)
        self.initial_capex.setPrefix("$")
        capex_layout.addWidget(self.initial_capex)

        self.sustaining_capex = QDoubleSpinBox()
        self.sustaining_capex.setRange(0, 100)
        self.sustaining_capex.setValue(2)
        self.sustaining_capex.setSuffix(" $/t")
        capex_layout.addWidget(QLabel("Sustaining:"))
        capex_layout.addWidget(self.sustaining_capex)

        form.addRow("CAPEX:", capex_layout)

        # Tax
        tax_layout = QHBoxLayout()
        self.tax_rate = QDoubleSpinBox()
        self.tax_rate.setRange(0, 0.5)
        self.tax_rate.setValue(0.30)
        self.tax_rate.setSingleStep(0.01)
        tax_layout.addWidget(QLabel("Tax:"))
        tax_layout.addWidget(self.tax_rate)

        self.royalty_rate = QDoubleSpinBox()
        self.royalty_rate.setRange(0, 0.2)
        self.royalty_rate.setValue(0.05)
        self.royalty_rate.setSingleStep(0.01)
        tax_layout.addWidget(QLabel("Royalty:"))
        tax_layout.addWidget(self.royalty_rate)

        form.addRow("Tax/Royalty:", tax_layout)

        # Mining modifying factors
        mmf_layout = QHBoxLayout()
        self.dilution = QDoubleSpinBox()
        self.dilution.setRange(0, 0.5)
        self.dilution.setValue(0.05)
        self.dilution.setSingleStep(0.01)
        mmf_layout.addWidget(QLabel("Dilution:"))
        mmf_layout.addWidget(self.dilution)

        self.mining_loss = QDoubleSpinBox()
        self.mining_loss.setRange(0, 0.3)
        self.mining_loss.setValue(0.02)
        self.mining_loss.setSingleStep(0.01)
        mmf_layout.addWidget(QLabel("Loss:"))
        mmf_layout.addWidget(self.mining_loss)

        form.addRow("MMF:", mmf_layout)

        self.advanced_economics_group = group
        layout.addWidget(group)

    def _create_optimization_options(self, layout):
        """Create optimization options and run button."""
        group = QGroupBox("3. Optimization")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #7b1fa2;
                border: 2px solid #7b1fa2;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        form = QFormLayout(group)

        # Optimization method
        self.optimization_method = QComboBox()
        self.optimization_method.addItems([
            "NPV Maximization",
            "IRR Maximization",
            "Payback Minimization"
        ])
        form.addRow("Method:", self.optimization_method)

        # Run button
        self.run_btn = QPushButton("Run Cut-off Optimization")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.run_btn.clicked.connect(self._run_optimization)
        self.run_btn.setEnabled(False)
        form.addRow("", self.run_btn)

        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        form.addRow("", self.export_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        form.addRow("", self.progress_bar)

        layout.addWidget(group)

    def _create_results_panel(self) -> QWidget:
        """Create results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.results_tabs = QTabWidget()

        # NPV Curve tab - with matplotlib chart
        npv_tab = QWidget()
        npv_layout = QVBoxLayout(npv_tab)

        # Create matplotlib figure for NPV curve - professional dark theme
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(10, 5), dpi=100)
            self.figure.patch.set_facecolor('#0d1117')
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            # Create subplot for NPV curve
            self.ax_npv = self.figure.add_subplot(111)
            self._style_npv_axes()

            # Toolbar for interactive navigation (zoom, pan, save)
            toolbar_container = QWidget()
            toolbar_layout = QHBoxLayout(toolbar_container)
            toolbar_layout.setContentsMargins(0, 0, 0, 0)

            self.chart_toolbar = NavigationToolbar(self.canvas, self)
            self.chart_toolbar.setStyleSheet("""
                QToolBar { background: #161b22; border: none; spacing: 5px; }
                QToolButton { background: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 4px; color: #c9d1d9; }
                QToolButton:hover { background: #30363d; }
                QToolButton:pressed { background: #388bfd; }
            """)
            toolbar_layout.addWidget(self.chart_toolbar)

            # Export chart button
            self.export_chart_btn = QPushButton("Export Chart")
            self.export_chart_btn.setStyleSheet("""
                QPushButton { background: #238636; color: white; border: none; border-radius: 4px; padding: 6px 12px; font-weight: 500; }
                QPushButton:hover { background: #2ea043; }
            """)
            self.export_chart_btn.clicked.connect(self._export_chart)
            toolbar_layout.addWidget(self.export_chart_btn)
            toolbar_layout.addStretch()

            npv_layout.addWidget(toolbar_container)
            npv_layout.addWidget(self.canvas, stretch=3)
        else:
            self.npv_placeholder = QLabel("Matplotlib not available - install matplotlib for charts")
            self.npv_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.npv_placeholder.setStyleSheet("""
                QLabel {
                    color: #666;
                    font-size: 14px;
                    padding: 40px;
                    border: 2px dashed #ccc;
                    border-radius: 10px;
                }
            """)
            npv_layout.addWidget(self.npv_placeholder)

        self.optimal_display = QLabel()
        self.optimal_display.setStyleSheet("""
            QLabel {
                background-color: #161b22;
                border: 1px solid #238636;
                border-left: 4px solid #3fb950;
                border-radius: 6px;
                padding: 15px 20px;
                font-weight: 500;
                font-size: 13px;
                color: #c9d1d9;
            }
        """)
        self.optimal_display.setVisible(False)
        npv_layout.addWidget(self.optimal_display)

        self.results_tabs.addTab(npv_tab, "NPV Optimization")

        # Sensitivity tab
        sens_tab = QWidget()
        sens_layout = QVBoxLayout(sens_tab)

        self.sensitivity_table = QTableWidget(0, 6)
        self.sensitivity_table.setHorizontalHeaderLabels([
            "Cutoff", "Tonnage", "Grade", "NPV", "IRR", "Payback"
        ])
        sens_layout.addWidget(self.sensitivity_table)

        self.results_tabs.addTab(sens_tab, "Sensitivity Data")

        layout.addWidget(self.results_tabs)

        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(80)
        self.status_text.setReadOnly(True)
        self.status_text.setPlainText("Load a GT curve to enable optimization...")
        layout.addWidget(self.status_text)

        return panel

    def set_grade_tonnage_curve(self, curve: GradeTonnageCurve):
        """Set the grade-tonnage curve for optimization."""
        self.grade_tonnage_curve = curve
        self.run_btn.setEnabled(curve is not None)

        if curve:
            total_tonnage = curve.global_statistics.get('total_tonnage', 0)
            self.status_text.setPlainText(
                f"GT Curve loaded: {len(curve.points)} points, "
                f"{total_tonnage:,.0f} tonnes total"
            )

    def _run_optimization(self):
        """Run cut-off optimization."""
        if self.grade_tonnage_curve is None:
            QMessageBox.warning(self, "No Data", "Load a GT curve first.")
            return

        try:
            self.run_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)

            # Update economic parameters
            self.sensitivity_engine.economic_params = {
                "metal_price": self.metal_price.value(),
                "mining_cost": self.mining_cost.value(),
                "processing_cost": self.processing_cost.value(),
                "recovery": self.recovery.value(),
                "discount_rate": self.discount_rate.value(),
                "admin_cost": 0,
                "transport_cost": 0
            }

            self.progress_bar.setValue(30)

            # Get optimization method
            method_map = {
                0: CutoffOptimizationMethod.NPV_MAXIMIZATION,
                1: CutoffOptimizationMethod.IRR_MAXIMIZATION,
                2: CutoffOptimizationMethod.PAYBACK_MINIMIZATION
            }
            method = method_map.get(self.optimization_method.currentIndex(),
                                   CutoffOptimizationMethod.NPV_MAXIMIZATION)

            # Run sensitivity analysis
            cutoffs = np.array([pt.cutoff_grade for pt in self.grade_tonnage_curve.points])

            self.sensitivity_result = self.sensitivity_engine.perform_sensitivity_analysis(
                self.grade_tonnage_curve,
                cutoffs,
                method
            )

            self.progress_bar.setValue(80)

            # If advanced economics enabled, run multi-period DCF
            if self.advanced_economics_group.isChecked():
                self._run_multi_period_dcf()

            self._update_results_display()

            self.progress_bar.setValue(100)
            self.export_btn.setEnabled(True)

            self.optimizationCompleted.emit(
                self.sensitivity_result,
                self.sensitivity_result.optimal_cutoff
            )

        except Exception as e:
            self.status_text.setPlainText(f"Error: {str(e)}")
            logger.exception("Optimization error")

        finally:
            self.run_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _run_multi_period_dcf(self):
        """Run multi-period DCF analysis."""
        econ_params = EconomicParameters(
            metal_price=self.metal_price.value(),
            mining_cost_ore=self.mining_cost.value(),
            processing_cost=self.processing_cost.value(),
            recovery_rate=self.recovery.value(),
            dilution_factor=self.dilution.value(),
            mining_loss_factor=self.mining_loss.value(),
            discount_rate=self.discount_rate.value()
        )

        capacity = MineCapacity(
            annual_ore_capacity=self.annual_capacity.value(),
            ramp_up_years=self.ramp_up_years.value()
        )

        capex = CapitalExpenditure(
            initial_capex=self.initial_capex.value(),
            sustaining_capex_per_tonne=self.sustaining_capex.value()
        )

        tax = TaxParameters(
            corporate_tax_rate=self.tax_rate.value(),
            royalty_rate=self.royalty_rate.value()
        )

        config = MineEconomicsConfig(
            economic_params=econ_params,
            capacity=capacity,
            capex=capex,
            tax=tax
        )

        self.economics_engine = MineEconomicsEngine(config)

        # Find optimal cutoff using multi-period NPV
        optimal_cutoff, optimal_result = self.economics_engine.find_optimal_cutoff(
            self.grade_tonnage_curve
        )

        # Update sensitivity result with multi-period optimal
        self.sensitivity_result.optimal_cutoff = optimal_cutoff
        self.sensitivity_result.economic_parameters['multi_period_npv'] = optimal_result.npv
        self.sensitivity_result.economic_parameters['multi_period_irr'] = optimal_result.irr
        self.sensitivity_result.economic_parameters['mine_life'] = optimal_result.mine_life_years

    def _update_results_display(self):
        """Update results display."""
        if self.sensitivity_result is None:
            return

        optimal = self.sensitivity_result.optimal_cutoff
        max_npv = max(self.sensitivity_result.npv_by_cutoff)

        # Find optimal point data
        optimal_idx = np.argmax(self.sensitivity_result.npv_by_cutoff)
        optimal_tonnage = 0
        optimal_grade = 0
        for pt in self.grade_tonnage_curve.points:
            if abs(pt.cutoff_grade - optimal) < 0.01:
                optimal_tonnage = pt.tonnage
                optimal_grade = pt.avg_grade
                break

        # Update optimal display
        display_text = f"""
OPTIMAL CUT-OFF GRADE: {optimal:.2f}

NPV: ${max_npv:,.0f}
Tonnage: {optimal_tonnage:,.0f} t
Average Grade: {optimal_grade:.3f}
"""
        if self.advanced_economics_group.isChecked():
            econ = self.sensitivity_result.economic_parameters
            display_text += f"""
Multi-Period NPV: ${econ.get('multi_period_npv', 0):,.0f}
IRR: {(econ.get('multi_period_irr', 0) or 0) * 100:.1f}%
Mine Life: {econ.get('mine_life', 0)} years
"""

        self.optimal_display.setText(display_text)
        self.optimal_display.setVisible(True)

        # Plot NPV curve
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'figure'):
            self._plot_npv_curve()

        # Update sensitivity table
        cutoffs = self.sensitivity_result.cutoff_range
        npvs = self.sensitivity_result.npv_by_cutoff
        irrs = self.sensitivity_result.irr_by_cutoff
        paybacks = self.sensitivity_result.payback_by_cutoff

        self.sensitivity_table.setRowCount(len(cutoffs))

        for i, cutoff in enumerate(cutoffs):
            # Find corresponding GT point
            tonnage = grade = 0
            for pt in self.grade_tonnage_curve.points:
                if abs(pt.cutoff_grade - cutoff) < 0.01:
                    tonnage = pt.tonnage
                    grade = pt.avg_grade
                    break

            self.sensitivity_table.setItem(i, 0, QTableWidgetItem(f"{cutoff:.2f}"))
            self.sensitivity_table.setItem(i, 1, QTableWidgetItem(f"{tonnage:,.0f}"))
            self.sensitivity_table.setItem(i, 2, QTableWidgetItem(f"{grade:.3f}"))
            self.sensitivity_table.setItem(i, 3, QTableWidgetItem(f"${npvs[i]:,.0f}"))
            self.sensitivity_table.setItem(i, 4, QTableWidgetItem(
                f"{irrs[i]*100:.1f}%" if irrs is not None else "-"
            ))
            self.sensitivity_table.setItem(i, 5, QTableWidgetItem(
                f"{paybacks[i]:.1f}" if paybacks is not None and paybacks[i] < 999 else "-"
            ))

        self.status_text.setPlainText(
            f"Optimization complete. Optimal cutoff: {optimal:.2f}"
        )

    def _style_npv_axes(self):
        """Apply professional dark theme styling to NPV axes."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Professional dark theme colors
        bg_color = '#0d1117'
        text_color = '#c9d1d9'
        grid_color = '#21262d'

        self.ax_npv.set_facecolor(bg_color)
        self.ax_npv.tick_params(colors=text_color, labelsize=10)

        for spine in ['top', 'right']:
            self.ax_npv.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            self.ax_npv.spines[spine].set_color(grid_color)
            self.ax_npv.spines[spine].set_linewidth(0.8)

        self.figure.patch.set_facecolor(bg_color)
        self.figure.tight_layout(pad=2.0)

    def _plot_npv_curve(self):
        """Plot professional NPV optimization curve."""
        if self.sensitivity_result is None:
            return

        # Professional color scheme
        bg_color = '#0d1117'
        text_color = '#c9d1d9'
        grid_color = '#21262d'
        npv_color = '#3fb950'      # Green for positive NPV
        npv_fill = '#238636'
        optimal_color = '#f0883e'   # Gold for optimal point
        negative_color = '#f85149'  # Red for negative NPV

        cutoffs = self.sensitivity_result.cutoff_range
        npvs = self.sensitivity_result.npv_by_cutoff
        optimal = self.sensitivity_result.optimal_cutoff
        optimal_idx = np.argmax(npvs)
        optimal_npv = npvs[optimal_idx]

        # Clear previous plot
        self.ax_npv.clear()
        self.ax_npv.set_facecolor(bg_color)

        # Create gradient fill - positive above zero, negative below
        positive_mask = npvs >= 0
        negative_mask = npvs < 0

        # Fill positive region
        self.ax_npv.fill_between(
            cutoffs, 0, np.where(positive_mask, npvs, 0),
            alpha=0.2, color=npv_fill, linewidth=0
        )

        # Fill negative region (if any)
        if np.any(negative_mask):
            self.ax_npv.fill_between(
                cutoffs, 0, np.where(negative_mask, npvs, 0),
                alpha=0.2, color=negative_color, linewidth=0
            )

        # Plot NPV curve
        self.ax_npv.plot(
            cutoffs, npvs, '-', color=npv_color,
            linewidth=2.5, label='NPV', solid_capstyle='round'
        )

        # Add markers at key points
        n_markers = min(10, len(cutoffs))
        marker_indices = np.linspace(0, len(cutoffs)-1, n_markers, dtype=int)
        self.ax_npv.scatter(
            cutoffs[marker_indices], npvs[marker_indices],
            color=npv_color, s=35, zorder=4, edgecolors='white', linewidths=1
        )

        # Mark optimal point with prominent styling
        self.ax_npv.axvline(
            x=optimal, color=optimal_color, linestyle='--',
            linewidth=1.5, alpha=0.8, zorder=3
        )
        self.ax_npv.scatter(
            [optimal], [optimal_npv], color=optimal_color,
            s=150, zorder=6, marker='*', edgecolors='white', linewidths=1.5
        )

        # Add annotation for optimal point
        def format_value(x):
            if abs(x) >= 1e9:
                return f'${x/1e9:.1f}B'
            elif abs(x) >= 1e6:
                return f'${x/1e6:.1f}M'
            elif abs(x) >= 1e3:
                return f'${x/1e3:.0f}K'
            return f'${x:.0f}'

        self.ax_npv.annotate(
            f'Optimal: {optimal:.2f}\nNPV: {format_value(optimal_npv)}',
            xy=(optimal, optimal_npv),
            xytext=(15, 15), textcoords='offset points',
            fontsize=9, color=text_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22',
                     edgecolor=optimal_color, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color=optimal_color, lw=1)
        )

        # Add zero line
        self.ax_npv.axhline(y=0, color='#484f58', linestyle='-', linewidth=1, alpha=0.8)

        # Labels and title
        self.ax_npv.set_xlabel('Cutoff Grade', fontsize=11, color=text_color, fontweight='medium')
        self.ax_npv.set_ylabel('Net Present Value', fontsize=11, color=text_color, fontweight='medium')
        self.ax_npv.set_title(
            'NPV Optimization Curve', fontsize=13, color=text_color,
            fontweight='bold', pad=15
        )

        # Grid
        self.ax_npv.grid(True, alpha=0.15, color=grid_color, linestyle='-', linewidth=0.5)
        self.ax_npv.set_axisbelow(True)

        # Format y-axis with currency (auto-scale to M/B)
        self.ax_npv.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format_value(x))
        )

        # Tick styling
        self.ax_npv.tick_params(colors=text_color, labelsize=10)

        # Spine styling
        for spine in ['top', 'right']:
            self.ax_npv.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            self.ax_npv.spines[spine].set_color(grid_color)
            self.ax_npv.spines[spine].set_linewidth(0.8)

        # Legend
        legend = self.ax_npv.legend(
            loc='upper left', frameon=True,
            facecolor='#161b22', edgecolor=grid_color,
            fontsize=9, labelcolor=text_color
        )
        legend.get_frame().set_alpha(0.9)

        # Set axis limits
        self.ax_npv.set_xlim(cutoffs.min(), cutoffs.max())
        y_range = npvs.max() - npvs.min()
        self.ax_npv.set_ylim(npvs.min() - y_range * 0.1, npvs.max() * 1.15)

        self.figure.tight_layout(pad=1.5)
        self.canvas.draw()

    def _export_results(self):
        """Export optimization results."""
        if self.sensitivity_result is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv)"
        )

        if filename:
            try:
                data = []
                for i, cutoff in enumerate(self.sensitivity_result.cutoff_range):
                    row = {
                        'cutoff_grade': cutoff,
                        'npv': self.sensitivity_result.npv_by_cutoff[i]
                    }
                    if self.sensitivity_result.irr_by_cutoff is not None:
                        row['irr'] = self.sensitivity_result.irr_by_cutoff[i]
                    if self.sensitivity_result.payback_by_cutoff is not None:
                        row['payback'] = self.sensitivity_result.payback_by_cutoff[i]
                    data.append(row)

                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)

                QMessageBox.information(self, "Export Complete", f"Exported to {filename}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _export_chart(self):
        """Export the NPV chart as an image file."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'figure'):
            QMessageBox.warning(self, "Export Error", "No chart available to export.")
            return

        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Chart",
            "npv_optimization_curve",
            "PNG Image (*.png);;SVG Vector (*.svg);;PDF Document (*.pdf);;All Files (*)"
        )

        if filename:
            try:
                # Ensure correct extension
                if not any(filename.lower().endswith(ext) for ext in ['.png', '.svg', '.pdf']):
                    if 'PNG' in selected_filter:
                        filename += '.png'
                    elif 'SVG' in selected_filter:
                        filename += '.svg'
                    elif 'PDF' in selected_filter:
                        filename += '.pdf'
                    else:
                        filename += '.png'

                # Export with high DPI for quality
                self.figure.savefig(
                    filename,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor=self.figure.get_facecolor(),
                    edgecolor='none'
                )

                QMessageBox.information(self, "Export Complete", f"Chart exported to {filename}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
